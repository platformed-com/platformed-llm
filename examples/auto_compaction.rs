//! Interactive REPL with auto-compaction.
//!
//! Demonstrates the compaction primitives shipped with the lib:
//!
//! - [`Provider::capabilities`] tells us the model's context window.
//! - [`Usage::total_tokens`] reports how much the last turn consumed.
//! - [`Capabilities::context_usage_fraction`] turns that into a
//!   `[0.0, 1.0]`-ish ratio.
//!
//! When the ratio crosses `COMPACTION_THRESHOLD`, we ask the model to
//! summarize the conversation so far into a compact memo and restart
//! with `[system + summary + last user message]`. This is the same
//! pattern the Claude CLI's `/compact` uses — collapse history into
//! a dense recap, keep recent turns verbatim so the active focus
//! doesn't get lost.
//!
//! ## Usage
//!
//! ```text
//! cargo run --example auto_compaction --features openai
//! # or
//! PROVIDER_TYPE=anthropic GOOGLE_CLOUD_PROJECT=... \
//!   cargo run --example auto_compaction --features anthropic-vertex
//! ```
//!
//! Type messages at the `>` prompt. Type `quit` or send EOF to exit.
//! After each turn we print the context utilization; once it crosses
//! the threshold the loop prints `(compacting…)` and rebuilds the
//! prompt before the next turn.
//!
//! ## Things to look at
//!
//! - [`compact`] is the only meaningful logic — everything else is
//!   plumbing.
//! - We hold the last user message back from the summary and re-attach
//!   it post-compaction, so the model's next turn still sees what the
//!   user just asked. (Without this the summary would absorb the live
//!   question into "the user is asking X" and the next response would
//!   answer from the meta level.)
//! - The summary instruction is deliberately terse and asks for a
//!   memo — model-supplied preamble like "Here's a summary:" would
//!   leak into the rebuilt prompt and confuse the next turn.

use std::error::Error;
use std::io::{self, BufRead, Write};

use platformed_llm::{generate, Capabilities, Config, Prompt, Provider, ProviderFactory};

/// Trigger compaction when `context_usage_fraction` crosses this
/// threshold. 0.7 (= 70%) leaves ~30% headroom for the next turn's
/// input + output before we'd actually hit the context window.
const COMPACTION_THRESHOLD: f32 = 0.7;

/// The summarization instruction is appended as a final user turn
/// when we want a recap. Kept terse — preamble like "Here's the
/// summary:" would leak into the rebuilt prompt.
const SUMMARIZATION_INSTRUCTION: &str = "\
The conversation above will be discarded to free up context space. \
Reply with a single dense, complete memo that captures every fact, \
decision, open question, and named entity the next turn needs to \
continue this conversation. Do not address the user; do not write \
in second person; do not include any preamble like 'Here's a \
summary'. Output only the memo.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenvy::dotenv().ok();
    let provider = ProviderFactory::from_env().await?;

    // We need both `&dyn Provider` (for `generate`) and the
    // provider's name handle (for `capabilities`). Resolve the
    // model once up front.
    let model = std::env::var("MODEL_NAME").unwrap_or_else(|_| {
        match std::env::var("PROVIDER_TYPE").as_deref() {
            Ok("anthropic") => "claude-sonnet-4-5".to_string(),
            Ok("google") => "gemini-2.5-flash".to_string(),
            _ => "gpt-4o-mini".to_string(),
        }
    });
    let caps = provider.capabilities(&model);
    let config = Config::builder(&model).build();

    println!(
        "Model: {} ({} token context, compaction at {:.0}%)",
        model,
        caps.context_window_tokens,
        COMPACTION_THRESHOLD * 100.0,
    );
    println!("Type a message at the prompt. `quit` or Ctrl-D to exit.\n");

    let mut conversation = Prompt::system(
        "You are a helpful assistant. Reply concisely unless the user asks for detail.",
    );

    let stdin = io::stdin();
    let mut input = stdin.lock();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush()?;
        let mut line = String::new();
        if input.read_line(&mut line)? == 0 {
            break; // EOF
        }
        let user_msg = line.trim();
        if user_msg.is_empty() {
            continue;
        }
        if matches!(user_msg, "quit" | "exit" | ":q") {
            break;
        }

        conversation = conversation.with_user(user_msg);
        let response = generate(&*provider, &conversation, &config)
            .await?
            .buffer()
            .await?;
        println!("\n{}\n", response.text());

        let fraction = caps.context_usage_fraction(&response.usage);
        println!(
            "  [context: {} input + {} output = {} / {} tokens ({:.1}% used)]",
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.total_tokens(),
            caps.context_window_tokens,
            fraction * 100.0,
        );

        conversation = conversation.with_response(&response);

        if fraction > COMPACTION_THRESHOLD {
            println!("  [compacting…]");
            conversation = compact(&*provider, &config, &caps, conversation, user_msg).await?;
            println!("  [done; conversation rebuilt from summary]\n");
        }
    }

    Ok(())
}

/// Replace the conversation with `[system, "earlier conversation: <summary>", last_user_msg]`.
///
/// The summary is produced by the same model — we ask it to recap
/// the whole conversation into a single memo, then drop the entire
/// history and start over with that memo as a synthetic prior turn.
/// The most recent user message is kept verbatim so the next
/// response answers it directly rather than the meta-level "the
/// user is asking X" framing it would get if it were swept into the
/// summary.
async fn compact(
    provider: &dyn Provider,
    config: &Config,
    caps: &Capabilities,
    history: Prompt,
    last_user_msg: &str,
) -> Result<Prompt, Box<dyn Error>> {
    // Ask the model for the recap. We append the instruction as a
    // final user turn to the existing history so the model has the
    // full conversation in context when producing the summary.
    let summary_prompt = history.with_user(SUMMARIZATION_INSTRUCTION);
    let summary_response = generate(provider, &summary_prompt, config)
        .await?
        .buffer()
        .await?;
    let summary = summary_response.text();
    let summary_tokens = summary_response.usage.output_tokens;
    let post_summary_fraction = caps.context_usage_fraction(&summary_response.usage);

    eprintln!(
        "  [summary: {summary_tokens} output tokens, prompt was {:.1}% full when generating it]",
        post_summary_fraction * 100.0,
    );

    // Rebuild: system + summary-as-prior-context + the last user turn.
    Ok(Prompt::system(
        "You are a helpful assistant. The next user turn references an earlier conversation; \
         use the assistant turn below as your memory of what was said previously.",
    )
    .with_assistant(format!(
        "Memo of the earlier conversation:\n\n{}",
        summary.trim()
    ))
    .with_user(last_user_msg))
}
