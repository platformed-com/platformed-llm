//! Conversation compaction — collapse a long prompt into a dense
//! summary so a long-running session stays under the model's context
//! window.
//!
//! Use [`Compactor`] when you're driving a multi-turn conversation
//! and want to keep going past the point where the raw history
//! wouldn't fit. The typical loop:
//!
//! 1. Send the user's message and get a response (via
//!    [`crate::generate`]).
//! 2. After the response, call [`Compactor::should_compact`] with
//!    the model's [`Capabilities`] and the response's [`Usage`].
//!    If `true`, call [`Compactor::compact`] to rewrite the prompt.
//! 3. If a request fails with [`crate::Error::ContextWindowExceeded`]
//!    (a single turn that itself overflowed the window), call
//!    [`Compactor::compact`] with the *pre-request* history and the
//!    in-flight user message as the held-out tail — then retry.
//!
//! [`examples/auto_compaction.rs`](../../examples/auto_compaction.rs)
//! demonstrates both paths against a live provider.
//!
//! ## Prompt design
//!
//! The default summarization instruction is informed by the patterns
//! used by aider (<https://github.com/Aider-AI/aider/blob/main/aider/prompts.py>)
//! and the leaked Claude Code `/compact` prompt
//! (<https://github.com/Piebald-AI/claude-code-system-prompts/blob/main/system-prompts/agent-prompt-conversation-summarization.md>):
//!
//! - **Aider's "user retelling" framing** — phrase the summary as
//!   the user recapping what happened ("I asked you to…"), so the
//!   model picks it up naturally as user context rather than a
//!   meta-narrative.
//! - **Claude Code's anti-drift rules** — preserve explicit user
//!   requests *in order*, preserve security-relevant instructions
//!   verbatim, anchor next steps in direct quotes from recent turns.
//! - **No structural tags** (Claude Code's `<analysis>` / `<summary>`)
//!   — those work when the consumer parses them out; here the memo
//!   feeds straight back into the prompt as plain text.
//!
//! Override [`Compactor::with_summarization_instruction`] when your
//! domain calls for something different (e.g. coding agents may want
//! Claude Code's verbose 9-section format; chat assistants probably
//! don't).
//!
//! ## Security note
//!
//! Compaction has a well-documented sharp edge: instructions buried
//! in the conversation (file contents, tool output, malicious user
//! input) can survive the summary as if the user had said them, and
//! the post-compaction model treats them as authoritative. See
//! <https://www.straiker.ai/blog/claude-code-source-leak-with-great-agency-comes-great-responsibility>
//! for a write-up of the Claude Code variant of this issue. The
//! default prompt mitigates by explicitly asking for user requests
//! and security constraints to be preserved *verbatim*, but
//! callers running compaction over untrusted content should layer
//! their own defenses (input sanitization, post-summary review).

use crate::{generate, Capabilities, Config, Error, InputItem, Prompt, Provider, Usage};

/// Default fraction of the context window past which proactive
/// compaction kicks in. 0.7 leaves ~30% headroom for the next turn's
/// input + output before the window would actually be exceeded.
pub const DEFAULT_COMPACTION_THRESHOLD: f32 = 0.7;

/// Default summarization instruction sent to the model when running
/// [`Compactor::compact`]. Synthesizes the patterns from aider's
/// chat-history summarization and Claude Code's `/compact`:
///
/// - phrased as a user retelling so the summary slots into the
///   rebuilt prompt naturally,
/// - asks for explicit user requests + named entities + open
///   questions + most recent focus,
/// - preserves any security-relevant instructions verbatim,
/// - forbids preamble like "Here's a summary" so the output drops
///   straight into the rebuilt prompt without leaking framing.
pub const DEFAULT_SUMMARIZATION_INSTRUCTION: &str = "\
The conversation above is about to be discarded to free up context space. Write a dense, \
detailed memo that captures everything the next turn needs to continue without losing \
important information.

Include:
- Every explicit user request, in the order it was made.
- Key facts, decisions, and named entities (files, URLs, names, identifiers, code).
- Any open questions or pending tasks, with what was decided about each.
- The most recent topic of focus and what was happening immediately before this memo.

Preserve any security or safety instructions the user gave (e.g. \"do not read X\", \
\"never call Y\", \"the password is Z, don't echo it\") verbatim — they MUST still apply \
after the memo replaces the conversation.

Phrase the memo as the user recapping what happened so far in first person (\"I asked you \
to…\", \"We decided to…\"). This makes the memo usable as a synthetic prior turn without \
the model treating it as a meta-narrative.

Output ONLY the memo. Do not address the user; do not include preamble like \
\"Here's a summary\"; do not wrap the memo in markdown fences.";

/// Default prefix applied to the summary text when it's inserted as
/// the synthetic user turn during rebuild. Signals to the
/// post-compaction model that the user turn is a memo recapping
/// earlier conversation, not a fresh request.
pub const DEFAULT_MEMO_PREFIX: &str = "[Compacted memo of earlier conversation]\n\n";

/// Configurable conversation compactor.
///
/// Holds the compaction threshold and the prompts used during
/// summarization. Cheap to construct; the default constructor returns
/// a sensible configuration for general-purpose chat (see
/// [`DEFAULT_COMPACTION_THRESHOLD`], [`DEFAULT_SUMMARIZATION_INSTRUCTION`],
/// [`DEFAULT_MEMO_PREFIX`]). Override individual fields via the
/// builder methods when your domain calls for something different.
#[derive(Debug, Clone)]
pub struct Compactor {
    threshold: f32,
    summarization_instruction: String,
    memo_prefix: String,
}

impl Default for Compactor {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_COMPACTION_THRESHOLD,
            summarization_instruction: DEFAULT_SUMMARIZATION_INSTRUCTION.to_string(),
            memo_prefix: DEFAULT_MEMO_PREFIX.to_string(),
        }
    }
}

impl Compactor {
    /// New compactor with library defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Trigger compaction when [`Capabilities::context_usage_fraction`]
    /// reaches `threshold`. Default is [`DEFAULT_COMPACTION_THRESHOLD`]
    /// (0.7).
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Override the summarization instruction sent to the model as
    /// the final user turn during [`Self::compact`]. See
    /// [`DEFAULT_SUMMARIZATION_INSTRUCTION`] for the prompt-design
    /// rationale.
    pub fn with_summarization_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.summarization_instruction = instruction.into();
        self
    }

    /// Override the prefix applied to the summary text when it's
    /// inserted as a synthetic user turn during rebuild.
    pub fn with_memo_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.memo_prefix = prefix.into();
        self
    }

    /// Current threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// `true` when `usage` (from the most recent turn) indicates the
    /// conversation should be proactively compacted before the next
    /// request. Compare `usage` against `caps.context_window_tokens`
    /// — once the fraction reaches [`Self::threshold`], the next
    /// turn's response is at risk of exceeding the window.
    pub fn should_compact(&self, caps: &Capabilities, usage: &Usage) -> bool {
        caps.context_usage_fraction(usage) >= self.threshold
    }

    /// Rewrite `history` as a compacted prompt: ask the model to
    /// summarize the conversation so far into a dense memo, then
    /// rebuild as `[original system, synthetic user memo]`.
    ///
    /// The memo lands as a `User` turn rather than `Assistant` for
    /// two reasons. First, it matches the voice of the default
    /// summarization instruction (aider's "user retelling" framing:
    /// "I asked you to…"), so the role and the content agree.
    /// Second, it's load-bearing on Anthropic-via-Vertex, which
    /// hoists `system` to a top-level field. If the rebuild started
    /// with an assistant turn the wire `messages` array would lead
    /// with `assistant`, and Anthropic's Messages API rejects that
    /// with a 400 (`first message must use the "user" role`). With
    /// the memo as a user turn the wire array leads with `user`,
    /// which every provider accepts; the caller's subsequent
    /// `with_user(live)` append produces two consecutive user turns,
    /// which Anthropic and Google both merge transparently and
    /// OpenAI accepts as-is.
    ///
    /// The returned prompt is *ready to continue from* — callers
    /// append whatever the next turn looks like (a fresh user
    /// message, a multipart user turn with attachments, a
    /// tool-result, etc.) using the normal `Prompt` builder methods.
    /// The library doesn't take an opinion on what that next turn
    /// is, because user turns aren't always plain strings (they may
    /// carry images, tool results, cache breakpoints) and callers
    /// already own that construction.
    ///
    /// The recovery pattern after a mid-turn
    /// [`Error::ContextWindowExceeded`]:
    ///
    /// ```ignore
    /// let rebuilt = compactor
    ///     .compact(&*provider, &config, history)
    ///     .await?
    ///     .with_user(live_user_msg); // the request that failed
    /// let response = generate(&*provider, &rebuilt, &config).await?.buffer().await?;
    /// ```
    ///
    /// The summarization request itself goes through the same
    /// `provider` + `config` (so it'll honor the active middleware
    /// chain). If summarization fails — including with
    /// [`Error::ContextWindowExceeded`] when the history is *already*
    /// too big to summarize in one shot — the error propagates and
    /// the caller must split the work themselves.
    pub async fn compact(
        &self,
        provider: &dyn Provider,
        config: &Config,
        history: Prompt,
    ) -> Result<Prompt, Error> {
        let original_system = extract_first_system(&history);

        let summary_prompt = history.with_user(&self.summarization_instruction);
        let summary_response = generate(provider, &summary_prompt, config)
            .await?
            .buffer()
            .await?;
        let summary = summary_response.text();

        let rebuilt = match original_system {
            Some(s) => Prompt::system(s),
            None => Prompt::new(),
        };
        Ok(rebuilt.with_user(format!("{}{}", self.memo_prefix, summary.trim())))
    }
}

/// Return the content of the first `InputItem::System` in `prompt`,
/// if any. Used to preserve the caller's system instruction across
/// a compaction rebuild.
fn extract_first_system(prompt: &Prompt) -> Option<String> {
    prompt.items().iter().find_map(|item| match item {
        InputItem::System(s) => Some(s.clone()),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::mock::{MockProvider, MockResponse};
    use crate::types::Usage;
    use crate::{InputItem, UserPart};

    fn caps_128k() -> Capabilities {
        Capabilities {
            context_window_tokens: 128_000,
            max_output_tokens: 16_384,
            ..Capabilities::default()
        }
    }

    #[test]
    fn should_compact_fires_at_threshold() {
        let c = Compactor::new(); // threshold = 0.7
        let caps = caps_128k();
        let under = Usage {
            input_tokens: 80_000,
            output_tokens: 1_000,
            ..Usage::default()
        };
        let over = Usage {
            input_tokens: 100_000,
            output_tokens: 1_000,
            ..Usage::default()
        };
        assert!(!c.should_compact(&caps, &under));
        assert!(c.should_compact(&caps, &over));
    }

    #[test]
    fn with_threshold_overrides_default() {
        let c = Compactor::new().with_threshold(0.5);
        assert_eq!(c.threshold(), 0.5);
        let caps = caps_128k();
        // 70k input is 54.7% of 128k — under default 0.7, over our 0.5.
        let usage = Usage {
            input_tokens: 70_000,
            output_tokens: 0,
            ..Usage::default()
        };
        assert!(c.should_compact(&caps, &usage));
    }

    #[tokio::test]
    async fn compact_summarizes_and_rebuilds_prompt() {
        let provider = MockProvider::builder()
            .reply(MockResponse::text("Memo body: we discussed weather."))
            .build();
        let config = Config::builder("test-model").build();
        let history = Prompt::system("You are helpful.")
            .with_user("What's the weather?")
            .with_assistant("Sunny.");

        let compacted = Compactor::new()
            .compact(&provider, &config, history)
            .await
            .unwrap();
        let items = compacted.items();
        // [system, user(memo)].
        assert_eq!(items.len(), 2);
        assert!(matches!(&items[0], InputItem::System(s) if s == "You are helpful."));
        match &items[1] {
            InputItem::User { content } => {
                assert_eq!(content.len(), 1);
                match &content[0] {
                    UserPart::Text(t) => {
                        assert!(t.starts_with(DEFAULT_MEMO_PREFIX));
                        assert!(t.contains("Memo body"));
                    }
                    other => panic!("expected text part, got {other:?}"),
                }
            }
            other => panic!("expected user turn, got {other:?}"),
        }
    }

    /// `compact()` returns just `[system, user-memo]` — the caller
    /// appends the next turn themselves. This test pins that shape
    /// and demonstrates the standard caller-side append.
    #[tokio::test]
    async fn caller_attaches_next_turn_after_compaction() {
        let provider = MockProvider::builder()
            .reply(MockResponse::text("memo"))
            .build();
        let config = Config::builder("test-model").build();
        let history = Prompt::system("sys").with_user("earlier");

        let compacted = Compactor::new()
            .compact(&provider, &config, history)
            .await
            .unwrap();
        // Rebuilt prompt is exactly [system, user-memo].
        assert_eq!(compacted.items().len(), 2);

        // Callers attach whatever the next turn looks like — here a
        // multipart user message demonstrates that the lib doesn't
        // need a string-shaped hook.
        let with_next = compacted.with_item(InputItem::User {
            content: vec![
                UserPart::Text("what should I do next?".into()),
                UserPart::Text(" (with structure)".into()),
            ],
        });
        assert_eq!(with_next.items().len(), 3);
        match with_next.items().last().unwrap() {
            InputItem::User { content } => assert_eq!(content.len(), 2),
            other => panic!("expected user turn, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn compact_works_without_system_message() {
        let provider = MockProvider::builder()
            .reply(MockResponse::text("memo"))
            .build();
        let config = Config::builder("test-model").build();
        // No system message in the history.
        let history = Prompt::user("hi").with_assistant("hello");

        let compacted = Compactor::new()
            .compact(&provider, &config, history)
            .await
            .unwrap();
        // First item is the user-memo (no system was preserved).
        match &compacted.items()[0] {
            InputItem::User { .. } => {}
            other => panic!("expected user first, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn compact_records_the_summarization_request() {
        // Confirm the call we send to the model includes the
        // summarization instruction as a final user turn.
        let provider = MockProvider::builder()
            .reply(MockResponse::text("memo"))
            .build();
        let log = provider.call_log();
        let config = Config::builder("test-model").build();
        let history = Prompt::system("sys").with_user("ask").with_assistant("ans");
        let _ = Compactor::new()
            .compact(&provider, &config, history)
            .await
            .unwrap();
        let calls = log.calls();
        assert_eq!(calls.len(), 1);
        // The last item must be a user turn carrying the instruction.
        let items = calls[0].prompt.items();
        match items.last().unwrap() {
            InputItem::User { content } => match &content[0] {
                UserPart::Text(t) => assert!(
                    t.contains("dense, detailed memo"),
                    "summarization instruction missing"
                ),
                other => panic!("expected text user part, got {other:?}"),
            },
            other => panic!("expected user turn last, got {other:?}"),
        }
    }
}
