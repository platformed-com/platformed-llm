//! Interactive REPL with auto-compaction.
//!
//! Demonstrates [`Compactor`] driving a long-running conversation:
//!
//! - After every successful turn, check [`Compactor::should_compact`]
//!   against the response's [`Usage`] and the model's
//!   [`Capabilities`]. If over the threshold, summarize-and-rebuild
//!   before the next user message.
//! - If a turn fails with [`Error::ContextWindowExceeded`] (the
//!   single message itself overflowed the window), compact the prior
//!   history, then re-attach the live user message and retry once.
//!   `Compactor::compact` returns `[system, user-memo]`; the
//!   caller appends the next turn — see `send_with_recovery` below.
//!
//! The compaction prompt itself lives in the library
//! ([`DEFAULT_SUMMARIZATION_INSTRUCTION`]), informed by aider's
//! "user retelling" framing and the leaked Claude Code `/compact`
//! anti-drift rules. See [`compaction`](platformed_llm::compaction)
//! for the rationale.
//!
//! ## Usage
//!
//! ```text
//! cargo run --example auto_compaction --features openai
//! ```
//!
//! Type messages at the `>` prompt. `quit`, `exit`, `:q`, or Ctrl-D
//! to exit. After each turn the loop prints the context utilization;
//! once it crosses the threshold the loop prints `[compacting…]`
//! and rebuilds the prompt before the next turn.

use std::error::Error as StdError;
use std::io::{self, BufRead, Write};

use platformed_llm::compaction::DEFAULT_SUMMARIZATION_INSTRUCTION;
use platformed_llm::{
    generate, Capabilities, Compactor, CompleteResponse, Config, Error, Prompt, Provider,
    ProviderFactory,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn StdError>> {
    dotenvy::dotenv().ok();
    let provider = ProviderFactory::from_env().await?;

    let model = std::env::var("MODEL_NAME").unwrap_or_else(|_| {
        match std::env::var("PROVIDER_TYPE").as_deref() {
            Ok("anthropic") => "claude-sonnet-4-5".to_string(),
            Ok("google") => "gemini-2.5-flash".to_string(),
            _ => "gpt-4o-mini".to_string(),
        }
    });
    let caps = provider.capabilities(&model);
    let config = Config::builder(&model).build();
    let compactor = Compactor::new(); // defaults: 0.7 threshold, library prompts

    println!(
        "Model: {} ({} token context, compaction at {:.0}%)",
        model,
        caps.context_window_tokens,
        compactor.threshold() * 100.0,
    );
    println!(
        "Compaction instruction (first line): {}",
        DEFAULT_SUMMARIZATION_INSTRUCTION
            .lines()
            .next()
            .unwrap_or(""),
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
        let user_msg = line.trim().to_string();
        if user_msg.is_empty() {
            continue;
        }
        if matches!(user_msg.as_str(), "quit" | "exit" | ":q") {
            break;
        }

        // `take()` the conversation so we can either thread it
        // through the happy path or hand it to the compactor and
        // rebuild — either way we replace it before the next loop.
        let (response, next_conversation) =
            send_with_recovery(&*provider, &config, &compactor, conversation, &user_msg).await?;
        conversation = next_conversation;

        println!("\n{}\n", response.text());
        report_usage(&caps, &response);

        // Proactive compaction once the threshold is crossed.
        if compactor.should_compact(&caps, &response.usage) {
            println!(
                "  [compacting at {:.1}% context…]",
                caps.context_usage_fraction(&response.usage) * 100.0,
            );
            conversation = compactor.compact(&*provider, &config, conversation).await?;
            println!("  [done; conversation rebuilt from summary]\n");
        }
    }

    Ok(())
}

/// Send `user_msg` against `conversation`. Returns
/// `(response, updated conversation including the response)`.
///
/// Recovers from `Error::ContextWindowExceeded` once: the live user
/// message couldn't fit alongside the existing history, so we
/// compact the prior history, append the live message on top of the
/// rebuilt prompt, and retry. Subsequent failures propagate.
async fn send_with_recovery(
    provider: &dyn Provider,
    config: &Config,
    compactor: &Compactor,
    conversation: Prompt,
    user_msg: &str,
) -> Result<(CompleteResponse, Prompt), Error> {
    // `generate()` itself can fail (pre-flight provider rejection),
    // and `buffer()` can fail (in-stream error — e.g. OpenAI's
    // 200-OK-then-SSE-`event:error` shape). Funnel both into one
    // Result so the recovery branch handles either case uniformly.
    let pending = generate(provider, &conversation.clone().with_user(user_msg), config).await;
    let attempt = match pending {
        Ok(response) => response.buffer().await,
        Err(e) => Err(e),
    };

    match attempt {
        Ok(response) => {
            // Happy path: commit user message + response to history.
            let next = conversation.with_user(user_msg).with_response(&response);
            Ok((response, next))
        }
        Err(Error::ContextWindowExceeded { message, .. }) => {
            eprintln!("  [context window exceeded mid-turn: {message}]");
            eprintln!("  [compacting prior history and retrying…]");
            // Compact returns `[system, user-memo]`; we attach
            // the live user message on top and retry. This is the
            // same shape as the happy path's
            // `conversation.with_user(user_msg)` — the only
            // difference is `conversation` is now the compacted
            // rebuild rather than the bloated original.
            let rebuilt = compactor.compact(provider, config, conversation).await?;
            let retry = rebuilt.clone().with_user(user_msg);
            let response = generate(provider, &retry, config).await?.buffer().await?;
            let next = rebuilt.with_user(user_msg).with_response(&response);
            Ok((response, next))
        }
        Err(e) => Err(e),
    }
}

/// Print a one-line summary of the model's reported usage and the
/// resulting context fraction.
fn report_usage(caps: &Capabilities, response: &CompleteResponse) {
    let fraction = caps.context_usage_fraction(&response.usage);
    println!(
        "  [context: {} input + {} output = {} / {} tokens ({:.1}% used)]",
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.usage.total_tokens(),
        caps.context_window_tokens,
        fraction * 100.0,
    );
}
