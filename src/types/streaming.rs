//! Streaming event representation.
//!
//! Every event names its target `AssistantPart` by integer index.
//! Indices are monotonically increasing within a single assistant turn
//! (0, 1, 2, …). The accumulator becomes a straight-line dispatch on
//! variant — no implicit "currently-active part" state.

use crate::types::{Annotation, FinishReason, ProviderBuiltin, ProviderContinuation, Usage};

/// Events emitted by [`crate::Response`] streams.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A new assistant content part is opening. `index` is monotonically
    /// increasing within the turn. One-shot parts
    /// ([`PartKind::RedactedReasoning`]) carry all their data in `kind`
    /// and emit no subsequent Delta / PartUpdate for this index.
    PartStart {
        /// Monotonic part index within the turn.
        index: u32,
        /// What kind of part is opening.
        kind: PartKind,
    },

    /// Append data to the part at `index`. The interpretation depends on
    /// the part's kind:
    /// - [`PartKind::Text`] / [`PartKind::Refusal`] → text delta.
    /// - [`PartKind::Reasoning`] → reasoning text delta.
    /// - [`PartKind::ToolCall`] → JSON-argument delta.
    Delta {
        /// Index of the part being extended.
        index: u32,
        /// The delta payload to append.
        delta: String,
    },

    /// Out-of-band metadata for a part. Always arrives before the
    /// matching `PartEnd`. May arrive multiple times.
    PartUpdate {
        /// Index of the part being updated.
        index: u32,
        /// Metadata update to apply.
        update: PartUpdate,
    },

    /// No further events will arrive for this part.
    PartEnd {
        /// Index of the part that just closed.
        index: u32,
    },

    /// The assistant turn is complete.
    Done {
        /// Why the model stopped.
        finish_reason: FinishReason,
        /// Final usage counters for the turn.
        usage: Usage,
    },

    /// Mid-stream fatal error.
    Error {
        /// Human-readable error description.
        error: String,
    },
}

/// Kind of part being streamed. Mirrors [`crate::AssistantPart`] but in
/// "header" form — the content arrives via subsequent [`StreamEvent`]s.
#[derive(Debug, Clone, PartialEq)]
pub enum PartKind {
    /// Visible text part.
    Text,
    /// Reasoning / chain-of-thought part.
    Reasoning,
    /// Anthropic opaque thinking blob. One-shot — no subsequent deltas.
    RedactedReasoning {
        /// Opaque server-encrypted thinking blob.
        data: String,
    },
    /// OpenAI-style refusal part.
    Refusal,
    /// Tool call header. Arguments stream via `Delta` events.
    ToolCall {
        /// Identifier the model assigns to the call.
        call_id: String,
        /// Tool name.
        name: String,
    },
    /// Builtin tool invocation the provider executed natively (web
    /// search, code execution, …). Arguments stream via `Delta`;
    /// result (when present) arrives via
    /// [`PartUpdate::BuiltinToolResult`].
    BuiltinToolCall {
        /// Which builtin tool the provider executed.
        kind: ProviderBuiltin,
    },
    /// Provider-issued resumption hint identifying this assistant
    /// turn. One-shot — all data lives in `kind`, no subsequent
    /// `Delta` / `PartUpdate` events for this index.
    Continuation(ProviderContinuation),
}

/// Metadata update for a streaming part.
#[derive(Debug, Clone)]
pub enum PartUpdate {
    /// Opaque provider signature for the part being updated. On a
    /// [`PartKind::Reasoning`] part it carries Anthropic's thinking
    /// signature (arriving after the last reasoning delta); on a
    /// [`PartKind::ToolCall`] part it carries Gemini's `thoughtSignature`.
    /// Lands on the corresponding part's signature field.
    Signature(String),
    /// Citation / annotation on the text up to this point.
    Annotation(Annotation),
    /// Result payload for a [`PartKind::BuiltinToolCall`] part — JSON,
    /// shape depends on the builtin (`{"outcome": "...", "output":
    /// "..."}` for code execution).
    BuiltinToolResult(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Usage;

    #[test]
    fn done_carries_finish_and_usage() {
        let ev = StreamEvent::Done {
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        };
        assert!(matches!(ev, StreamEvent::Done { .. }));
    }

    #[test]
    fn tool_call_kind_carries_id_and_name() {
        let kind = PartKind::ToolCall {
            call_id: "call_abc".into(),
            name: "get_weather".into(),
        };
        match kind {
            PartKind::ToolCall { call_id, name } => {
                assert_eq!(call_id, "call_abc");
                assert_eq!(name, "get_weather");
            }
            _ => panic!("wrong kind"),
        }
    }
}
