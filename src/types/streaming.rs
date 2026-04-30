//! Types for streaming responses.

use crate::types::{FinishReason, FunctionCall, Usage};

/// Events emitted by [`crate::Response`] streams.
///
/// The expected ordering for a single response is:
///
/// 1. Zero or more output items, each announced by [`StreamEvent::OutputItemAdded`].
///    - For [`OutputItemInfo::Text`], one or more [`StreamEvent::ContentDelta`]
///      events follow with the next text chunks.
///    - For [`OutputItemInfo::Reasoning`], one or more
///      [`StreamEvent::ReasoningDelta`] events follow with chain-of-thought
///      text. May carry an opaque `signature` that must be echoed back in
///      subsequent turns to preserve reasoning continuity (Anthropic).
///    - For [`OutputItemInfo::FunctionCall`], a single
///      [`StreamEvent::FunctionCallComplete`] follows once the model has
///      finished assembling the arguments JSON.
/// 2. Exactly one terminal [`StreamEvent::Done`] event with the finish reason
///    and final usage tally.
///
/// [`StreamEvent::Error`] terminates the stream out-of-band and must be
/// treated as fatal for the current response.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Incremental text appended to the most recent text output item.
    ContentDelta { delta: String },
    /// Incremental chain-of-thought text appended to the most recent
    /// reasoning output item.
    ReasoningDelta { delta: String },
    /// Opaque signature attached to the current reasoning output item.
    /// Anthropic emits this once at the end of a thinking block; it must
    /// be echoed back unchanged in subsequent turns to preserve thinking
    /// continuity. Other providers ignore it.
    ReasoningSignature { signature: String },
    /// A new output item is starting. Subsequent `ContentDelta` /
    /// `ReasoningDelta` / `FunctionCallComplete` events apply to this item.
    OutputItemAdded { item: OutputItemInfo },
    /// A function call has completed. The `call_id` matches the one carried
    /// in the corresponding `OutputItemAdded { item: FunctionCall { id } }`
    /// for some providers (Anthropic) and shares a base UUID with it for
    /// others (Gemini); for OpenAI the `id` and `call_id` are distinct
    /// surfaces of the same call.
    FunctionCallComplete { call: FunctionCall },
    /// The stream has finished. This is the last event.
    Done {
        finish_reason: FinishReason,
        usage: Usage,
    },
    /// A fatal streaming-level error. The stream will yield no further
    /// events for this response.
    Error { error: String },
}

/// Information about an output item being added.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputItemInfo {
    /// Text/message output item. Filled by subsequent `ContentDelta`
    /// events.
    Text,
    /// Function call output item. Followed by exactly one
    /// `FunctionCallComplete` carrying the assembled arguments.
    FunctionCall { name: String, id: String },
    /// Chain-of-thought reasoning output. Filled by subsequent
    /// `ReasoningDelta` events.
    Reasoning,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_event_properties() {
        let content_event = StreamEvent::ContentDelta {
            delta: "test".to_string(),
        };
        assert!(matches!(content_event, StreamEvent::ContentDelta { .. }));

        let done_event = StreamEvent::Done {
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        };
        assert!(matches!(done_event, StreamEvent::Done { .. }));
    }

    #[test]
    fn test_output_item_added_event() {
        // Test OutputItemAdded event creation and properties
        let text_event = StreamEvent::OutputItemAdded {
            item: OutputItemInfo::Text,
        };

        let function_event = StreamEvent::OutputItemAdded {
            item: OutputItemInfo::FunctionCall {
                name: "get_weather".to_string(),
                id: "fc_123".to_string(),
            },
        };

        // Test pattern matching
        assert!(matches!(text_event, StreamEvent::OutputItemAdded { .. }));
        assert!(matches!(
            function_event,
            StreamEvent::OutputItemAdded { .. }
        ));

        // Test that OutputItemInfo works correctly
        match &function_event {
            StreamEvent::OutputItemAdded {
                item: OutputItemInfo::FunctionCall { name, id },
            } => {
                assert_eq!(name, "get_weather");
                assert_eq!(id, "fc_123");
            }
            _ => panic!("Expected OutputItemAdded event with FunctionCall"),
        }

        match &text_event {
            StreamEvent::OutputItemAdded {
                item: OutputItemInfo::Text,
            } => {
                // Text items don't have additional data
            }
            _ => panic!("Expected OutputItemAdded event with Text"),
        }
    }
}
