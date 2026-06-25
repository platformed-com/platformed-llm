//! Conversation compaction — collapse a long prompt into a dense
//! summary so a long-running session stays under the model's context
//! window.
//!
//! [`Compactor`] takes a prompt about to be sent to a model and
//! returns a smaller drop-in replacement. The lib holds out the
//! trailing in-flight exchange so the caller doesn't have to manage
//! that bookkeeping manually:
//!
//! - When the prompt's tail is a user turn (the typical shape of a
//!   prompt about to be sent — a live question, or a tool result the
//!   model is expected to react to), the lib pops that user turn
//!   before summarising and re-attaches it on top of the memo.
//! - When the tail is a [`UserPart::ToolResult`], the lib also
//!   preserves the immediately-preceding assistant turn that emitted
//!   the matching `tool_call`. Compaction without this would leave
//!   an orphaned tool_result whose `call_id` references nothing —
//!   OpenAI 400s, Anthropic 400s, and Google silently drops the
//!   result (see the orphan handling in `providers::vertex::google::push_part`).
//! - When the tail is an assistant turn (proactive compaction
//!   immediately after a turn completed), the assistant turn is the
//!   held-out group; the caller appends the next user turn on top.
//!
//! The number of recent turns kept verbatim is configurable via
//! [`Compactor::with_keep_recent_turns`] (default 3). This matches
//! the consensus in other compaction implementations — Microsoft
//! Agent Framework's `MinimumPreserved`, Inspect AI's
//! `keep_tool_uses`, aider's token-budget-driven head/tail split.
//!
//! Typical workflows:
//!
//! ```ignore
//! // After a turn completes, check whether the next call needs compacting.
//! if compactor.should_compact(&caps, &response.usage) {
//!     conversation = compactor.compact(provider, config, conversation).await?;
//! }
//!
//! // Recovery: a request failed with ContextWindowExceeded; compact
//! // and retry. The live user message rides through as the held-out
//! // tail, so the retry is a drop-in resend.
//! match generate(provider, &prompt, config).await {
//!     Err(Error::ContextWindowExceeded { .. }) => {
//!         let retry = compactor.compact(provider, config, prompt).await?;
//!         generate(provider, &retry, config).await?
//!     }
//!     Ok(r) => r,
//! }
//! ```
//!
//! [`examples/auto_compaction.rs`](../../examples/auto_compaction.rs)
//! demonstrates both paths against a live provider.
//!
//! ## Scope
//!
//! This module is **summarisation-only**: it replaces older turns
//! with a memo. Callers who need other compaction strategies — tool
//! result collapsing, sliding-window truncation, hard-limit
//! truncation, or a layered pipeline of all of these — should build
//! them on top. See Microsoft Agent Framework's `CompactionStrategy`
//! taxonomy for prior art on a fuller toolbox.
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

/// Default number of trailing message groups held out from
/// summarisation and preserved verbatim in the rebuilt prompt. Each
/// group is one of: a `User` turn, a plain-text `Assistant` turn, or
/// an atomic `(Assistant tool_call, User tool_result)` pair. System
/// messages are always preserved and don't count toward this budget.
///
/// 3 is the consensus floor across other compaction implementations:
/// Microsoft Agent Framework's `SummarizationCompactionStrategy`
/// defaults `MinimumPreserved` to 4, Inspect AI's Edit Compaction
/// defaults `keep_tool_uses` to 3. Trades a small amount of
/// compression for substantially better continuation quality.
pub const DEFAULT_KEEP_RECENT_TURNS: usize = 3;

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
    keep_recent_turns: usize,
}

impl Default for Compactor {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_COMPACTION_THRESHOLD,
            summarization_instruction: DEFAULT_SUMMARIZATION_INSTRUCTION.to_string(),
            memo_prefix: DEFAULT_MEMO_PREFIX.to_string(),
            keep_recent_turns: DEFAULT_KEEP_RECENT_TURNS,
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

    /// Number of trailing message groups to preserve verbatim — the
    /// rest gets summarised into a single memo. Default is
    /// [`DEFAULT_KEEP_RECENT_TURNS`] (3). A "group" is a `User` turn,
    /// a plain-text `Assistant` turn, or an atomic
    /// `(Assistant tool_call, User tool_result)` pair. System
    /// messages are always preserved and never count toward this
    /// budget. Setting `0` summarises everything; setting a value
    /// larger than the prompt's group count makes `compact` a
    /// no-op.
    pub fn with_keep_recent_turns(mut self, keep_recent_turns: usize) -> Self {
        self.keep_recent_turns = keep_recent_turns;
        self
    }

    /// Current threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Current `keep_recent_turns` setting.
    pub fn keep_recent_turns(&self) -> usize {
        self.keep_recent_turns
    }

    /// `true` when `usage` (from the most recent turn) indicates the
    /// conversation should be proactively compacted before the next
    /// request. Compare `usage` against `caps.context_window_tokens`
    /// — once the fraction reaches [`Self::threshold`], the next
    /// turn's response is at risk of exceeding the window.
    pub fn should_compact(&self, caps: &Capabilities, usage: &Usage) -> bool {
        caps.context_usage_fraction(usage) >= self.threshold
    }

    /// Rewrite `prompt` as a compacted prompt: hold out the last
    /// [`Self::keep_recent_turns`] message groups verbatim, ask the
    /// model to summarise everything older into a dense memo, then
    /// rebuild as `[original system, user(memo), …held-out groups…]`.
    ///
    /// The returned prompt is a drop-in replacement for `prompt`: the
    /// caller hands it to [`crate::generate`] in place of the
    /// original. The held-out tail (whether that's a live user
    /// question, a tool-call/result pair mid-agentic-loop, or just a
    /// trailing assistant turn after a completed exchange) rides
    /// through verbatim — callers don't have to manage that
    /// bookkeeping themselves.
    ///
    /// The recovery pattern after a mid-turn
    /// [`Error::ContextWindowExceeded`]:
    ///
    /// ```ignore
    /// let retry = compactor
    ///     .compact(&*provider, &config, failed_prompt)
    ///     .await?;
    /// let response = generate(&*provider, &retry, &config).await?.buffer().await?;
    /// ```
    ///
    /// No-op fast path: if the prompt has at most
    /// `keep_recent_turns` non-system groups, the prompt is returned
    /// unchanged without invoking the summarisation model.
    ///
    /// The summarisation request itself goes through the same
    /// `provider` + `config` (so it honours the active middleware
    /// chain). If summarisation fails — including with
    /// [`Error::ContextWindowExceeded`] when the history is *already*
    /// too big to summarise in one shot — the error propagates and
    /// the caller must split the work themselves.
    pub async fn compact(
        &self,
        provider: &dyn Provider,
        config: &Config,
        prompt: Prompt,
    ) -> Result<Prompt, Error> {
        // 1. Split into system (preserved verbatim, doesn't count
        //    toward the keep_recent_turns budget) and the rest.
        let (system, rest) = split_off_system(prompt);
        // 2. Partition `rest` into atomic groups: User, AssistantText,
        //    and ToolCall (assistant tool_call + matching user
        //    tool_result fused into one group).
        let groups = group_items(rest);
        // 3. If we don't have more groups than keep_recent_turns,
        //    there's nothing to summarise — fast path.
        if groups.len() <= self.keep_recent_turns {
            return Ok(reassemble(system, Vec::new(), None, groups));
        }
        // 4. Split into [to_summarise, to_keep]: everything before
        //    the last keep_recent_turns groups goes to summarisation.
        let split_at = groups.len() - self.keep_recent_turns;
        let mut iter = groups.into_iter();
        let to_summarise: Vec<Group> = iter.by_ref().take(split_at).collect();
        let to_keep: Vec<Group> = iter.collect();
        // 5. Build the summarisation request: system (if any) + the
        //    items-to-summarise + the summarisation instruction as a
        //    final user turn. Because to_summarise was popped before
        //    the instruction lands, the instruction is always a
        //    standalone directive — no role-merging surprises.
        let mut summary_prompt = match &system {
            Some(s) => Prompt::system(s.clone()),
            None => Prompt::new(),
        };
        for g in &to_summarise {
            for item in g.items() {
                summary_prompt = summary_prompt.with_item(item.clone());
            }
        }
        let summary_prompt = summary_prompt.with_user(&self.summarization_instruction);
        let summary_response = generate(provider, &summary_prompt, config)
            .await?
            .buffer()
            .await?;
        // Guard: refuse to commit a degenerate memo. Either branch
        // here means the summarisation model didn't produce a usable
        // memo body, so swapping it in for older history would
        // silently destroy context. Propagate the failure with a
        // descriptive reason so the caller can recover (retry with a
        // larger summarisation budget, switch summarisation model,
        // surface to user).
        if summary_response.was_truncated() {
            return Err(Error::compaction(
                "summarisation response was truncated (FinishReason::Length); \
                 retry with a larger summarisation max_tokens or smaller history",
            ));
        }
        let summary = summary_response.text();
        let trimmed = summary.trim();
        if trimmed.is_empty() {
            return Err(Error::compaction(
                "summarisation response produced no usable text \
                 (empty / whitespace / refusal / pure tool-call)",
            ));
        }
        let memo = format!("{}{}", self.memo_prefix, trimmed);
        // 6. Rebuild: system + user(memo) + held-out groups.
        Ok(reassemble(system, to_summarise, Some(memo), to_keep))
    }
}

/// Atomic message group. System messages are handled separately
/// (always preserved, never counted toward `keep_recent_turns`).
#[derive(Debug)]
enum Group {
    /// A standalone user turn (text / image / cache breakpoint / etc.).
    /// Does NOT include user turns whose content is wrapped into a
    /// `ToolCall` group below.
    User(InputItem),
    /// A plain-text assistant turn (no tool calls).
    Assistant(InputItem),
    /// Atomic `(assistant tool_call, user tool_result)` exchange.
    /// Both items ride through compaction together so call_id
    /// integrity holds — OpenAI 400s on `function_call_output.call_id`
    /// mismatch, Anthropic on `tool_use_id` mismatch, and Google
    /// silently drops orphaned results client-side via
    /// `push_part`.
    ToolPair {
        assistant: InputItem,
        user_results: InputItem,
    },
}

impl Group {
    /// The InputItems this group expands to, in order.
    fn items(&self) -> Vec<&InputItem> {
        match self {
            Group::User(i) | Group::Assistant(i) => vec![i],
            Group::ToolPair {
                assistant,
                user_results,
            } => vec![assistant, user_results],
        }
    }

    fn into_items(self) -> Vec<InputItem> {
        match self {
            Group::User(i) | Group::Assistant(i) => vec![i],
            Group::ToolPair {
                assistant,
                user_results,
            } => vec![assistant, user_results],
        }
    }
}

/// Pop the first `InputItem::System` (if any) off the prompt, returning
/// its content plus the remaining items. System messages elsewhere in
/// the prompt are left in place (a caller that puts multiple system
/// messages in the middle of the conversation is doing something
/// unusual; we just preserve the first one for the rebuild).
fn split_off_system(prompt: Prompt) -> (Option<String>, Vec<InputItem>) {
    let mut system = None;
    let mut rest = Vec::new();
    for item in prompt.into_items() {
        match (&system, &item) {
            (None, InputItem::System(s)) => {
                system = Some(s.clone());
            }
            _ => rest.push(item),
        }
    }
    (system, rest)
}

/// Walk a flat item list and bucket consecutive items into atomic
/// `Group`s. The interesting case is `(assistant with ToolCall, user
/// with matching ToolResult)` pairs — those fuse into a single
/// `ToolPair` group. Everything else is one item per group.
///
/// Edge cases:
/// - An assistant turn with tool_calls whose immediately-following
///   user turn doesn't have matching tool_results: treat the
///   assistant as a standalone Assistant group (don't fuse).
/// - An assistant turn with tool_calls that's the last item: same
///   — standalone Assistant group, no pair.
/// - System messages in the rest list: shouldn't happen after
///   `split_off_system`, but if one slips through, treat as its own
///   group via the catch-all User branch (won't compile actually —
///   System isn't User; we just preserve it as a "User-like" group
///   for the simple fall-through).
fn group_items(items: Vec<InputItem>) -> Vec<Group> {
    let mut groups = Vec::new();
    let mut iter = items.into_iter().peekable();
    while let Some(item) = iter.next() {
        match item {
            InputItem::Assistant { ref content } if has_tool_call(content) => {
                // Try to fuse with the next user turn IF that user
                // turn's content has any ToolResult parts.
                if iter.peek().is_some_and(is_user_with_tool_result) {
                    let user_results = iter.next().expect("peeked Some");
                    groups.push(Group::ToolPair {
                        assistant: item,
                        user_results,
                    });
                } else {
                    groups.push(Group::Assistant(item));
                }
            }
            InputItem::Assistant { .. } => {
                groups.push(Group::Assistant(item));
            }
            InputItem::User { .. } => {
                groups.push(Group::User(item));
            }
            // System slipping through here is unusual but we preserve
            // it as a User-shaped pass-through so the rebuild doesn't
            // drop it silently.
            InputItem::System(_) => {
                groups.push(Group::User(item));
            }
        }
    }
    groups
}

fn has_tool_call(content: &[crate::AssistantPart]) -> bool {
    use crate::AssistantPart;
    content
        .iter()
        .any(|p| matches!(p, AssistantPart::ToolCall(_)))
}

fn is_user_with_tool_result(item: &InputItem) -> bool {
    use crate::UserPart;
    match item {
        InputItem::User { content } => content
            .iter()
            .any(|p| matches!(p, UserPart::ToolResult { .. })),
        _ => false,
    }
}

/// Build the final prompt: optional system + optional memo + held-out
/// groups. When `memo` is `None` we're on the no-op fast path —
/// `to_summarise` is empty and we reassemble the original input.
fn reassemble(
    system: Option<String>,
    to_summarise: Vec<Group>,
    memo: Option<String>,
    to_keep: Vec<Group>,
) -> Prompt {
    let mut out = match system {
        Some(s) => Prompt::system(s),
        None => Prompt::new(),
    };
    if let Some(memo_text) = memo {
        out = out.with_user(memo_text);
    } else {
        // No-op path: the items we'd otherwise summarise need to be
        // re-emitted verbatim.
        for g in to_summarise {
            for item in g.into_items() {
                out = out.with_item(item);
            }
        }
    }
    for g in to_keep {
        for item in g.into_items() {
            out = out.with_item(item);
        }
    }
    out
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

    // =====================================================================
    // Compaction spec
    // =====================================================================
    //
    // The tests below specify how `Compactor::compact` should behave as a
    // drop-in compaction primitive an agentic caller (Claude Code, aider,
    // our own auto_compaction example) can hand any in-flight prompt to.
    // They reflect what the public surface needs to do.
    //
    // Contract:
    //
    // 1. `compact(prompt)` takes the prompt the caller was about to send
    //    and returns a smaller drop-in replacement. The lib holds out the
    //    last `keep_recent_turns` message groups verbatim and summarises
    //    everything older into a single user-role memo, so the caller
    //    doesn't have to do any bookkeeping.
    //
    // 2. A "group" is the atomic unit of the message index:
    //
    //      - `System`     — preserved unconditionally, doesn't count toward
    //                       `keep_recent_turns`.
    //      - `User`       — a single `InputItem::User` (text, image,
    //                       cache breakpoint, …) that is not part of a
    //                       tool exchange.
    //      - `Assistant`  — a plain-text `InputItem::Assistant` with no
    //                       tool calls.
    //      - `ToolCall`   — an `InputItem::Assistant` containing one or
    //                       more `AssistantPart::ToolCall`s plus the
    //                       immediately-following `InputItem::User`
    //                       carrying the matching `UserPart::ToolResult`s.
    //                       Treated as ONE atomic group: either both
    //                       items ride through compaction together or
    //                       both go into the memo. Splitting them would
    //                       leave a `call_id` orphan — OpenAI 400s on
    //                       `function_call_output.call_id` mismatch,
    //                       Anthropic 400s on `tool_use_id` mismatch,
    //                       and Google silently drops the result
    //                       client-side (see google.rs::push_part).
    //
    //    Group atomicity matches the model used by Microsoft Agent
    //    Framework's `MessageGroup` / `MessageGroupKind` and OpenClaw's
    //    compaction-chunk boundary handling.
    //
    // 3. The summarization request the lib sends to the model includes
    //    every item EXCEPT the held-out groups, with the summarization
    //    instruction appended as a final user turn. Because the tail
    //    groups were popped first, the request always ends in
    //    assistant / system / empty before the instruction lands —
    //    review comment #2 (the instruction getting glued onto a user
    //    tail by provider-side merging) becomes structurally impossible.
    //
    // 4. Held-out content rides through verbatim. Multipart user turns
    //    (text + images + cache breakpoints + tool results) are preserved
    //    item-for-item, part-for-part. Cache breakpoints survive. A
    //    `ToolCall` group rides through with all its parts (text +
    //    tool_calls + reasoning), not selectively split.
    //
    // 5. The memo itself is a single `User` text part prefixed with the
    //    configured `memo_prefix`. It's a user turn (not assistant) so
    //    the wire array leads with `user` after every provider's
    //    system-hoist behaviour — see commit 8d694c7 for why.
    //
    // 6. When the prompt has fewer non-system groups than
    //    `keep_recent_turns`, `compact` is a no-op: it returns the
    //    input unchanged without calling the summarisation model.
    //    Avoids a round-trip when there's nothing to compact.

    /// **Spec 1 — typical conversational tail.** A prompt that ends in a
    /// user question is the common shape: caller built `conversation +
    /// next_question` and is about to send it. With `keep_recent_turns=1`
    /// only the trailing user turn rides through; everything before it
    /// gets summarised. The verbatim live question stays at the tail.
    #[tokio::test]
    async fn user_text_tail_is_held_out_after_memo() {
        let provider = MockProvider::builder()
            .reply(MockResponse::text("dense memo body"))
            .build();
        let config = Config::builder("test-model").build();
        let prompt = Prompt::system("be helpful")
            .with_user("first question")
            .with_assistant("first answer")
            .with_user("second question")
            .with_assistant("second answer")
            .with_user("the live question");

        let compacted = Compactor::new()
            .with_keep_recent_turns(1)
            .compact(&provider, &config, prompt)
            .await
            .unwrap();
        let items = compacted.items();

        // Shape: [system, user(memo), user(live)]
        assert_eq!(items.len(), 3, "{items:?}");
        assert!(matches!(&items[0], InputItem::System(s) if s == "be helpful"));
        match &items[1] {
            InputItem::User { content } => {
                assert_eq!(content.len(), 1);
                match &content[0] {
                    UserPart::Text(t) => {
                        assert!(t.starts_with(DEFAULT_MEMO_PREFIX));
                        assert!(t.contains("dense memo body"));
                    }
                    other => panic!("memo must be a single text part, got {other:?}"),
                }
            }
            other => panic!("memo must land as user turn, got {other:?}"),
        }
        match &items[2] {
            InputItem::User { content } => match &content[0] {
                UserPart::Text(t) => assert_eq!(t, "the live question"),
                other => panic!("tail's first part must be the verbatim question, got {other:?}"),
            },
            other => panic!("tail must be a user turn, got {other:?}"),
        }
    }

    /// **Spec 2 — mid-tool-loop tail.** When the prompt ends in a
    /// `tool_result`, the assistant turn that emitted the matching
    /// `tool_call` rides through compaction alongside it. Without this
    /// the rebuilt prompt has an orphaned tool_result: OpenAI 400s,
    /// Anthropic 400s, Google silently drops it. The compaction must
    /// preserve call_id integrity end-to-end.
    #[tokio::test]
    async fn tool_result_tail_holds_out_with_its_matching_assistant_tool_call() {
        use crate::{AssistantPart, FunctionCall};
        let provider = MockProvider::builder()
            .reply(MockResponse::text("memo body"))
            .build();
        let config = Config::builder("test-model").build();
        let prompt = Prompt::system("be helpful")
            .with_user("look up old data")
            .with_assistant_tool_call(FunctionCall {
                call_id: "call_old".into(),
                name: "search".into(),
                arguments: r#"{"q":"old"}"#.into(),
                provider_signature: None,
            })
            .with_tool_result("call_old", "old result")
            .with_assistant("here you go")
            .with_user("now look up new data")
            .with_assistant_tool_call(FunctionCall {
                call_id: "call_pending".into(),
                name: "search".into(),
                arguments: r#"{"q":"new"}"#.into(),
                provider_signature: None,
            })
            .with_tool_result("call_pending", "fresh result");

        let compacted = Compactor::new()
            .with_keep_recent_turns(1)
            .compact(&provider, &config, prompt)
            .await
            .unwrap();
        let items = compacted.items();

        // Shape: [system, user(memo), assistant(call_pending), user(tool_result for call_pending)]
        // The pending tool_call + result is one atomic group; with
        // keep_recent_turns=1 it's the single held-out group.
        assert_eq!(items.len(), 4, "{items:?}");
        assert!(matches!(&items[0], InputItem::System(_)));
        assert!(matches!(&items[1], InputItem::User { .. }));

        // The pending tool_call rides through; the OLDER call_old is
        // summarized into the memo, NOT preserved as a turn.
        match &items[2] {
            InputItem::Assistant { content } => {
                let calls: Vec<&FunctionCall> = content
                    .iter()
                    .filter_map(|p| match p {
                        AssistantPart::ToolCall(c) => Some(c),
                        _ => None,
                    })
                    .collect();
                assert_eq!(calls.len(), 1, "expected exactly one preserved tool_call");
                assert_eq!(
                    calls[0].call_id, "call_pending",
                    "only the immediately-preceding call should be preserved"
                );
            }
            other => panic!("expected preserved assistant tool_call, got {other:?}"),
        }
        match &items[3] {
            InputItem::User { content } => {
                let results: Vec<&str> = content
                    .iter()
                    .filter_map(|p| match p {
                        UserPart::ToolResult { call_id, .. } => Some(call_id.as_str()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(results, vec!["call_pending"]);
            }
            other => panic!("expected preserved tool_result, got {other:?}"),
        }
    }

    /// **Spec 3 — parallel tool calls.** Providers canonically pair one
    /// assistant turn carrying N `ToolCall`s with one user turn carrying
    /// N matching `ToolResult`s. Preserving "the last assistant + user
    /// pair" as a unit handles this naturally — the whole multi-call
    /// block rides through, all call_ids stay matched.
    #[tokio::test]
    async fn parallel_tool_calls_in_tail_preserve_whole_block_as_unit() {
        use crate::{AssistantPart, FunctionCall};
        let provider = MockProvider::builder()
            .reply(MockResponse::text("memo body"))
            .build();
        let config = Config::builder("test-model").build();
        let parallel_assistant = InputItem::Assistant {
            content: vec![
                AssistantPart::ToolCall(FunctionCall {
                    call_id: "call_a".into(),
                    name: "get_weather".into(),
                    arguments: r#"{"city":"Paris"}"#.into(),
                    provider_signature: None,
                }),
                AssistantPart::ToolCall(FunctionCall {
                    call_id: "call_b".into(),
                    name: "get_weather".into(),
                    arguments: r#"{"city":"London"}"#.into(),
                    provider_signature: None,
                }),
            ],
        };
        let parallel_results = InputItem::User {
            content: vec![
                UserPart::ToolResult {
                    call_id: "call_a".into(),
                    content: vec![UserPart::Text("sunny".into())],
                },
                UserPart::ToolResult {
                    call_id: "call_b".into(),
                    content: vec![UserPart::Text("rainy".into())],
                },
            ],
        };
        let prompt = Prompt::system("sys")
            .with_user("warm up")
            .with_assistant("ready")
            .with_user("weather in Paris and London?")
            .with_item(parallel_assistant)
            .with_item(parallel_results);

        let compacted = Compactor::new()
            .with_keep_recent_turns(1)
            .compact(&provider, &config, prompt)
            .await
            .unwrap();
        let items = compacted.items();

        // Shape: [system, user(memo), assistant(call_a + call_b), user(result_a + result_b)]
        assert_eq!(items.len(), 4, "{items:?}");
        match &items[2] {
            InputItem::Assistant { content } => {
                let ids: Vec<&str> = content
                    .iter()
                    .filter_map(|p| match p {
                        AssistantPart::ToolCall(c) => Some(c.call_id.as_str()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(ids, vec!["call_a", "call_b"]);
            }
            other => panic!("expected parallel assistant block, got {other:?}"),
        }
        match &items[3] {
            InputItem::User { content } => {
                let ids: Vec<&str> = content
                    .iter()
                    .filter_map(|p| match p {
                        UserPart::ToolResult { call_id, .. } => Some(call_id.as_str()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(ids, vec!["call_a", "call_b"]);
            }
            other => panic!("expected parallel results block, got {other:?}"),
        }
    }

    /// **Spec 4 — assistant-terminated history (proactive path).** Caller
    /// just finished a normal turn and is checking `should_compact`
    /// against the response's usage; the conversation ends in
    /// `assistant`. With `keep_recent_turns=1` the trailing assistant
    /// turn rides through; the rebuilt prompt ends in `assistant`,
    /// ready for the caller to append the next user turn on top. The
    /// rebuild's tail role matches the input's tail role.
    #[tokio::test]
    async fn assistant_tail_compacts_with_held_out_assistant_turn() {
        let provider = MockProvider::builder()
            .reply(MockResponse::text("memo body"))
            .build();
        let config = Config::builder("test-model").build();
        let prompt = Prompt::system("sys")
            .with_user("q1")
            .with_assistant("a1")
            .with_user("q2")
            .with_assistant("the trailing answer");

        let compacted = Compactor::new()
            .with_keep_recent_turns(1)
            .compact(&provider, &config, prompt)
            .await
            .unwrap();
        let items = compacted.items();

        // Shape: [system, user(memo), assistant(trailing)]
        assert_eq!(items.len(), 3, "{items:?}");
        assert!(matches!(&items[0], InputItem::System(_)));
        assert!(matches!(&items[1], InputItem::User { .. }));
        match &items[2] {
            InputItem::Assistant { content } => {
                use crate::AssistantPart;
                assert!(content.iter().any(|p| matches!(
                    p,
                    AssistantPart::Text { content: t, .. } if t == "the trailing answer"
                )));
            }
            other => panic!("expected held-out assistant tail, got {other:?}"),
        }
    }

    /// **Spec 5 — multipart user tail.** The tail user turn can carry
    /// text + images + cache breakpoints in any combination. Every part
    /// rides through compaction verbatim — the lib doesn't peek inside
    /// and re-serialize.
    #[tokio::test]
    async fn multipart_user_tail_preserves_every_part_verbatim() {
        use crate::FileSource;
        let provider = MockProvider::builder()
            .reply(MockResponse::text("memo body"))
            .build();
        let config = Config::builder("test-model").build();
        let multipart_tail = InputItem::User {
            content: vec![
                UserPart::Text("look at this:".into()),
                UserPart::Image(FileSource::Url("https://example.com/img.png".into())),
                UserPart::CacheBreakpoint,
                UserPart::Text("what do you see?".into()),
            ],
        };
        let prompt = Prompt::system("sys")
            .with_user("warm up")
            .with_assistant("ready")
            .with_user("more context")
            .with_assistant("noted")
            .with_item(multipart_tail.clone());

        let compacted = Compactor::new()
            .with_keep_recent_turns(1)
            .compact(&provider, &config, prompt)
            .await
            .unwrap();
        let items = compacted.items();
        assert_eq!(items.len(), 3, "{items:?}");
        // Tail at index 2 must equal the multipart input we supplied —
        // same parts in the same order.
        match (&items[2], &multipart_tail) {
            (InputItem::User { content: actual }, InputItem::User { content: expected }) => {
                assert_eq!(actual.len(), expected.len(), "tail part count drifted");
                for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                    match (a, e) {
                        (UserPart::Text(at), UserPart::Text(et)) => {
                            assert_eq!(at, et, "text part {i} drifted")
                        }
                        (UserPart::Image(_), UserPart::Image(_)) => {}
                        (UserPart::CacheBreakpoint, UserPart::CacheBreakpoint) => {}
                        (a, e) => panic!("part {i} variant changed: {a:?} vs {e:?}"),
                    }
                }
            }
            (other, _) => panic!("expected user tail, got {other:?}"),
        }
    }

    /// **Spec 6 — summarization request excludes the held-out tail.**
    /// The model summarizing the conversation should see the items that
    /// are being discarded (so it knows what to capture in the memo) but
    /// NOT the held-out tail (which is going to ride through anyway and
    /// would just bias the summary toward whatever the live question
    /// happens to be).
    #[tokio::test]
    async fn summarization_request_omits_the_held_out_segment() {
        let provider = MockProvider::builder()
            .reply(MockResponse::text("memo"))
            .build();
        let log = provider.call_log();
        let config = Config::builder("test-model").build();
        let prompt = Prompt::system("sys")
            .with_user("first thing")
            .with_assistant("first reply")
            .with_user("second thing")
            .with_assistant("second reply")
            .with_user("LIVE QUESTION SHOULD NOT APPEAR IN SUMMARY INPUT");

        let _ = Compactor::new()
            .with_keep_recent_turns(1)
            .compact(&provider, &config, prompt)
            .await
            .unwrap();

        let calls = log.calls();
        assert_eq!(calls.len(), 1);
        let request_items = calls[0].prompt.items();
        // The held-out tail must not be in the summarization request.
        for item in request_items {
            if let InputItem::User { content } = item {
                for part in content {
                    if let UserPart::Text(t) = part {
                        assert!(
                            !t.contains("LIVE QUESTION SHOULD NOT APPEAR IN SUMMARY INPUT"),
                            "held-out tail leaked into summarization request"
                        );
                    }
                }
            }
        }
        // Sanity: the items being discarded ARE in the summarization
        // request (so the model has something to summarize).
        let saw_discarded = request_items.iter().any(|i| match i {
            InputItem::User { content } => content
                .iter()
                .any(|p| matches!(p, UserPart::Text(t) if t.contains("first thing"))),
            _ => false,
        });
        assert!(
            saw_discarded,
            "discarded turns must be present in summarization request"
        );
    }

    /// **Spec 7 — system message persists.** Whatever system message
    /// the caller had in the input prompt (if any) appears as item 0 in
    /// the rebuild. Without a system message in the input, the rebuild
    /// just doesn't have one — no synthetic system is injected.
    #[tokio::test]
    async fn system_message_persists_through_compaction() {
        let provider = MockProvider::builder()
            .reply(MockResponse::text("memo"))
            .reply(MockResponse::text("memo"))
            .build();
        let config = Config::builder("test-model").build();

        // With system.
        let with_sys = Prompt::system("you are X")
            .with_user("hi")
            .with_assistant("hello")
            .with_user("more")
            .with_assistant("ok")
            .with_user("live");
        let out = Compactor::new()
            .with_keep_recent_turns(1)
            .compact(&provider, &config, with_sys)
            .await
            .unwrap();
        assert!(matches!(&out.items()[0], InputItem::System(s) if s == "you are X"));

        // Without system — no synthetic system is fabricated.
        let no_sys = Prompt::user("hi")
            .with_assistant("hello")
            .with_user("more")
            .with_assistant("ok")
            .with_user("live");
        let out = Compactor::new()
            .with_keep_recent_turns(1)
            .compact(&provider, &config, no_sys)
            .await
            .unwrap();
        assert!(
            !matches!(&out.items()[0], InputItem::System(_)),
            "no synthetic system should appear when input had none"
        );
    }

    /// **Spec 8 — `keep_recent_turns` preserves N groups.** With
    /// `keep_recent_turns=3` (the library default), the last three
    /// non-system groups ride through verbatim and only older
    /// content gets summarised. Mirrors Microsoft Agent Framework's
    /// `MinimumPreserved` / Inspect AI's `keep_tool_uses` semantics.
    #[tokio::test]
    async fn keep_recent_turns_preserves_n_groups() {
        let provider = MockProvider::builder()
            .reply(MockResponse::text("memo body"))
            .build();
        let config = Config::builder("test-model").build();
        // 6 non-system groups: user, assistant, user, assistant, user, assistant.
        let prompt = Prompt::system("sys")
            .with_user("q1")
            .with_assistant("a1")
            .with_user("q2")
            .with_assistant("a2")
            .with_user("q3")
            .with_assistant("a3");

        let compacted = Compactor::new()
            .with_keep_recent_turns(3)
            .compact(&provider, &config, prompt)
            .await
            .unwrap();
        let items = compacted.items();

        // Shape: [system, user(memo), user(q2), assistant(a2), user(q3), assistant(a3)]
        // — the last 3 groups (a2 onward: actually user(q3)/assistant(a3) plus
        // assistant(a2) — wait, last 3 groups counting BACK from the end are
        // assistant(a3), user(q3), assistant(a2). Adding memo and system:
        // [sys, memo, assistant(a2), user(q3), assistant(a3)].
        assert_eq!(items.len(), 5, "{items:?}");
        assert!(matches!(&items[0], InputItem::System(_)));
        assert!(matches!(&items[1], InputItem::User { .. }));
        // The three preserved groups in original order.
        use crate::AssistantPart;
        assert!(matches!(
            &items[2],
            InputItem::Assistant { content } if content.iter().any(|p| matches!(
                p,
                AssistantPart::Text { content: t, .. } if t == "a2"
            ))
        ));
        assert!(matches!(
            &items[3],
            InputItem::User { content } if content.iter().any(|p| matches!(
                p,
                UserPart::Text(t) if t == "q3"
            ))
        ));
        assert!(matches!(
            &items[4],
            InputItem::Assistant { content } if content.iter().any(|p| matches!(
                p,
                AssistantPart::Text { content: t, .. } if t == "a3"
            ))
        ));
    }

    /// **Spec 9 — no-op when there's not enough to compact.** If the
    /// prompt has at most `keep_recent_turns` non-system groups,
    /// there's nothing older to summarise; the lib returns the
    /// prompt unchanged without invoking the summarisation model.
    /// Avoids a wasted round-trip and a spurious empty memo.
    #[tokio::test]
    async fn no_op_when_history_smaller_than_keep_recent_turns() {
        let provider = MockProvider::builder()
            // Will panic if called — the test asserts the model isn't.
            .reply(MockResponse::text("this should never appear"))
            .build();
        let log = provider.call_log();
        let config = Config::builder("test-model").build();
        // 3 non-system groups; keep_recent_turns=3 → nothing to summarise.
        let prompt = Prompt::system("sys")
            .with_user("q1")
            .with_assistant("a1")
            .with_user("live");

        let original_items = prompt.items().to_vec();
        let compacted = Compactor::new()
            .with_keep_recent_turns(3)
            .compact(&provider, &config, prompt)
            .await
            .unwrap();
        assert_eq!(
            compacted.items().len(),
            original_items.len(),
            "no-op compaction must preserve item count"
        );
        assert_eq!(
            log.calls().len(),
            0,
            "no-op compaction must not call the summarisation model"
        );
    }

    /// **Spec 10 — empty summary fails fast.** A summarisation
    /// response with no usable text (refusal, content-filtered,
    /// pure tool-call, all-whitespace) means we can't build a
    /// meaningful memo. Compacting anyway would land
    /// `[system, user("[Compacted memo of earlier conversation]\n\n")]`
    /// — the prefix label with no body — and silently drop the
    /// older history. The lib must refuse the rebuild and surface
    /// the failure so the caller can decide (retry with a larger
    /// budget, switch summarisation model, surface to user).
    /// PR-review #6.
    #[tokio::test]
    async fn empty_summary_response_errors_without_destroying_history() {
        let provider = MockProvider::builder()
            // Reply with empty text — accumulator finishes with Stop.
            .reply(MockResponse::text(""))
            .build();
        let config = Config::builder("test-model").build();
        let prompt = Prompt::system("sys")
            .with_user("q1")
            .with_assistant("a1")
            .with_user("q2")
            .with_assistant("a2")
            .with_user("live");

        let result = Compactor::new()
            .with_keep_recent_turns(1)
            .compact(&provider, &config, prompt)
            .await;
        match result {
            Err(Error::Compaction { reason }) => {
                assert!(
                    reason.to_ascii_lowercase().contains("empty")
                        || reason.to_ascii_lowercase().contains("no usable"),
                    "error must name the empty-summary cause: got {reason:?}"
                );
            }
            other => panic!("expected Error::Compaction for empty summary, got {other:?}"),
        }
    }

    /// **Spec 11 — truncated summary fails fast.** If the
    /// summarisation model hit its output-token budget mid-memo
    /// (`FinishReason::Length`), the resulting text is a
    /// mid-sentence fragment. Committing it as the memo would
    /// silently truncate the historical context every caller relies
    /// on. The lib refuses the rebuild and surfaces the failure so
    /// the caller can retry with a larger summarisation budget.
    /// PR-review #6.
    #[tokio::test]
    async fn truncated_summary_response_errors_without_destroying_history() {
        use crate::{AssistantPart, FinishReason};
        let truncated = MockResponse::from_parts(
            vec![AssistantPart::Text {
                content: "I asked you to compute 4+4 and you said the".to_string(),
                annotations: Vec::new(),
            }],
            FinishReason::Length,
        );
        let provider = MockProvider::builder().reply(truncated).build();
        let config = Config::builder("test-model").build();
        let prompt = Prompt::system("sys")
            .with_user("q1")
            .with_assistant("a1")
            .with_user("q2")
            .with_assistant("a2")
            .with_user("live");

        let result = Compactor::new()
            .with_keep_recent_turns(1)
            .compact(&provider, &config, prompt)
            .await;
        match result {
            Err(Error::Compaction { reason }) => {
                assert!(
                    reason.to_ascii_lowercase().contains("truncated")
                        || reason.to_ascii_lowercase().contains("length"),
                    "error must name the truncation cause: got {reason:?}"
                );
            }
            other => panic!("expected Error::Compaction for truncated summary, got {other:?}"),
        }
    }

    /// **Spec 12 — whitespace-only summary fails fast.** Same
    /// destructive failure mode as the empty case — the model
    /// emitted text but only whitespace, so the trimmed memo body
    /// is empty. Treat as empty.
    #[tokio::test]
    async fn whitespace_only_summary_response_errors() {
        let provider = MockProvider::builder()
            .reply(MockResponse::text("   \n  \t  "))
            .build();
        let config = Config::builder("test-model").build();
        let prompt = Prompt::system("sys")
            .with_user("q1")
            .with_assistant("a1")
            .with_user("q2")
            .with_assistant("a2")
            .with_user("live");

        let result = Compactor::new()
            .with_keep_recent_turns(1)
            .compact(&provider, &config, prompt)
            .await;
        assert!(
            matches!(result, Err(Error::Compaction { .. })),
            "expected Error::Compaction for whitespace-only summary, got {result:?}"
        );
    }
}
