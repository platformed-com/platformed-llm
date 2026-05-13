//! Streaming event representation.
//!
//! Every event names its target `AssistantPart` by integer index.
//! Indices are monotonically increasing within a single assistant turn
//! (0, 1, 2, …). The accumulator becomes a straight-line dispatch on
//! variant — no implicit "currently-active part" state.

use crate::types::{Annotation, FinishReason, Usage};

/// Events emitted by [`crate::Response`] streams.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A new assistant content part is opening. `index` is monotonically
    /// increasing within the turn. One-shot parts
    /// ([`PartKind::RedactedReasoning`]) carry all their data in `kind`
    /// and emit no subsequent Delta / PartUpdate for this index.
    PartStart { index: u32, kind: PartKind },

    /// Append data to the part at `index`. The interpretation depends on
    /// the part's kind:
    /// - [`PartKind::Text`] / [`PartKind::Refusal`] → text delta.
    /// - [`PartKind::Reasoning`] → reasoning text delta.
    /// - [`PartKind::ToolCall`] → JSON-argument delta.
    Delta { index: u32, delta: String },

    /// Out-of-band metadata for a part. Always arrives before the
    /// matching `PartEnd`. May arrive multiple times.
    PartUpdate { index: u32, update: PartUpdate },

    /// No further events will arrive for this part.
    PartEnd { index: u32 },

    /// The assistant turn is complete.
    Done {
        finish_reason: FinishReason,
        usage: Usage,
    },

    /// Mid-stream fatal error.
    Error { error: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum PartKind {
    Text,
    Reasoning,
    /// Anthropic opaque thinking blob. One-shot — no subsequent deltas.
    RedactedReasoning { data: String },
    Refusal,
    /// Tool call header. Arguments stream via `Delta` events.
    ToolCall { call_id: String, name: String },
}

#[derive(Debug, Clone)]
pub enum PartUpdate {
    /// Anthropic thinking signature, arriving after the last reasoning
    /// delta.
    Signature(String),
    /// Citation / annotation on the text up to this point.
    Annotation(Annotation),
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
