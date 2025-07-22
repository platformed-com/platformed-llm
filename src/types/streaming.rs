//! Types for streaming responses.

use crate::types::{FinishReason, FunctionCall, Usage};

/// Events that can be emitted during streaming.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A chunk of content was received.
    ContentDelta { delta: String },
    /// A new output item was added (text or function call).
    OutputItemAdded { item: OutputItemInfo },
    /// A function call has completed with full arguments.
    FunctionCallComplete { call: FunctionCall },
    /// The stream has finished.
    Done {
        finish_reason: FinishReason,
        usage: Usage,
    },
    /// An error occurred during streaming.
    Error { error: String },
}

/// Information about an output item being added.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputItemInfo {
    /// Text/message output item.
    Text,
    /// Function call output item with name and ID.
    FunctionCall { name: String, id: String },
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
