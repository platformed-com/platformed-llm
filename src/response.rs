//! Response handling for LLM generations.

use crate::{Error, FinishReason, FunctionCall, StreamEvent, Usage};
use futures_util::stream::Stream;
use std::pin::Pin;

/// A complete response from an LLM provider.
#[derive(Debug, Clone)]
pub struct CompleteResponse {
    /// Ordered sequence of output items (text, function calls, etc.)
    pub output: Vec<OutputItem>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

/// An item in the LLM response output.
#[derive(Debug, Clone)]
pub enum OutputItem {
    /// Text content
    Text { content: String },
    /// Function call
    FunctionCall { call: FunctionCall },
}

impl OutputItem {
    /// Convert this output item to a message.
    /// Text items become assistant messages, function calls become function call messages.
    pub fn to_input_item(&self) -> crate::types::InputItem {
        match self {
            OutputItem::Text { content } => {
                crate::types::InputItem::Message(crate::types::Message {
                    role: crate::types::Role::Assistant,
                    content: content.clone(),
                })
            }
            OutputItem::FunctionCall { call } => {
                crate::types::InputItem::FunctionCall(call.clone())
            }
        }
    }
}

impl CompleteResponse {
    /// Get all text content concatenated together.
    pub fn content(&self) -> String {
        self.output
            .iter()
            .filter_map(|item| match item {
                OutputItem::Text { content } => Some(content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get all function calls in order.
    pub fn function_calls(&self) -> Vec<&FunctionCall> {
        self.output
            .iter()
            .filter_map(|item| match item {
                OutputItem::FunctionCall { call } => Some(call),
                _ => None,
            })
            .collect()
    }

    /// Convert this response to a sequence of input items.
    /// This preserves the ordering of text and function calls.
    pub fn to_items(&self) -> Vec<crate::types::InputItem> {
        self.output
            .iter()
            .map(|item| item.to_input_item())
            .collect()
    }
}

/// Response from an LLM generation that can be streamed or buffered.
/// All responses are internally streaming.
pub struct Response {
    stream: Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send>>,
}

impl Response {
    /// Create a new response from a stream of events.
    pub fn from_stream<S>(stream: S) -> Self
    where
        S: Stream<Item = Result<StreamEvent, Error>> + Send + 'static,
    {
        Self {
            stream: Box::pin(stream),
        }
    }

    /// Buffer the entire response by consuming the stream.
    pub async fn buffer(self) -> Result<CompleteResponse, Error> {
        Self::buffer_stream(self.stream).await
    }

    /// Get just the text content (convenience method).
    pub async fn text(self) -> Result<String, Error> {
        let complete = self.buffer().await?;
        Ok(complete.content())
    }

    /// Stream the response events.
    pub fn stream(self) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send>> {
        self.stream
    }

    /// Buffer a streaming response by consuming all events.
    async fn buffer_stream(
        mut stream: Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send>>,
    ) -> Result<CompleteResponse, Error> {
        use futures_util::StreamExt;

        let mut accumulator = crate::accumulator::ResponseAccumulator::new();

        while let Some(event_result) = stream.next().await {
            let event = event_result?;

            match &event {
                StreamEvent::Done { .. } => {
                    accumulator.process_event(event)?;
                    break;
                }
                StreamEvent::Error { error } => {
                    return Err(Error::streaming(error.clone()));
                }
                _ => {
                    accumulator.process_event(event)?;
                }
            }
        }

        accumulator.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_response_creation() {
        let response = CompleteResponse {
            output: vec![OutputItem::Text {
                content: "Hello, world!".to_string(),
            }],
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        };

        assert_eq!(response.content(), "Hello, world!");
        assert!(response.function_calls().is_empty());
    }

    #[tokio::test]
    async fn test_response_buffering() {
        // Create a mock stream that represents a complete response
        let events = vec![
            Ok(StreamEvent::OutputItemAdded {
                item: crate::types::OutputItemInfo::Text,
            }),
            Ok(StreamEvent::ContentDelta {
                delta: "Test response".to_string(),
            }),
            Ok(StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
            }),
        ];

        let stream = futures_util::stream::iter(events);
        let response = Response::from_stream(stream);
        let text = response.text().await.unwrap();
        assert_eq!(text, "Test response");
    }

    #[test]
    fn test_mixed_output_ordering() {
        use crate::types::FunctionCall;

        // Test that ordering is preserved: text -> function call -> more text
        let response = CompleteResponse {
            output: vec![
                OutputItem::Text {
                    content: "I'll help you with that. ".to_string(),
                },
                OutputItem::FunctionCall {
                    call: FunctionCall {
                        id: "fc_123".to_string(),
                        call_id: "call_123".to_string(),
                        name: "get_weather".to_string(),
                        arguments: "{\"location\":\"Paris\"}".to_string(),
                    },
                },
                OutputItem::Text {
                    content: " Let me also check something else.".to_string(),
                },
            ],
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        };

        // Test content concatenation
        assert_eq!(
            response.content(),
            "I'll help you with that.  Let me also check something else."
        );

        // Test function calls extraction
        let calls = response.function_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");

        // Test output items preserve order
        assert_eq!(response.output.len(), 3);
        match &response.output[0] {
            OutputItem::Text { content } => assert_eq!(content, "I'll help you with that. "),
            _ => panic!("Expected text item"),
        }
        match &response.output[1] {
            OutputItem::FunctionCall { call } => assert_eq!(call.name, "get_weather"),
            _ => panic!("Expected function call item"),
        }
        match &response.output[2] {
            OutputItem::Text { content } => {
                assert_eq!(content, " Let me also check something else.")
            }
            _ => panic!("Expected text item"),
        }
    }

    #[test]
    fn test_to_items() {
        use crate::types::{FunctionCall, Role};

        // Test text-only response
        let text_response = CompleteResponse {
            output: vec![OutputItem::Text {
                content: "Hello, world!".to_string(),
            }],
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        };

        let items = text_response.to_items();
        assert_eq!(items.len(), 1);
        match &items[0] {
            crate::types::InputItem::Message(msg) => {
                assert_eq!(msg.role(), Role::Assistant);
                assert_eq!(msg.content(), Some("Hello, world!".to_string()));
            }
            _ => panic!("Expected message"),
        }

        // Test response with function calls
        let mixed_response = CompleteResponse {
            output: vec![
                OutputItem::Text {
                    content: "I'll help you with that. ".to_string(),
                },
                OutputItem::FunctionCall {
                    call: FunctionCall {
                        id: "fc_123".to_string(),
                        call_id: "call_123".to_string(),
                        name: "get_weather".to_string(),
                        arguments: "{\"location\":\"Paris\"}".to_string(),
                    },
                },
            ],
            finish_reason: FinishReason::ToolCalls,
            usage: Usage::default(),
        };

        let items = mixed_response.to_items();
        assert_eq!(items.len(), 2); // One text message, one function call

        // First item should be text
        match &items[0] {
            crate::types::InputItem::Message(msg) => {
                assert_eq!(msg.role(), Role::Assistant);
                assert_eq!(msg.content(), Some("I'll help you with that. ".to_string()));
            }
            _ => panic!("Expected message"),
        }

        // Second item should be function call
        match &items[1] {
            crate::types::InputItem::FunctionCall(call) => {
                assert_eq!(call.name, "get_weather");
            }
            _ => panic!("Expected function call"),
        }
    }

    #[test]
    fn test_to_items_mixed() {
        use crate::types::{FunctionCall, Role};

        // Test response with mixed content
        let mixed_response = CompleteResponse {
            output: vec![
                OutputItem::Text {
                    content: "I'll help you with that. ".to_string(),
                },
                OutputItem::FunctionCall {
                    call: FunctionCall {
                        id: "fc_123".to_string(),
                        call_id: "call_123".to_string(),
                        name: "get_weather".to_string(),
                        arguments: "{\"location\":\"Paris\"}".to_string(),
                    },
                },
                OutputItem::Text {
                    content: "Let me check that for you.".to_string(),
                },
            ],
            finish_reason: FinishReason::ToolCalls,
            usage: Usage::default(),
        };

        let items = mixed_response.to_items();
        assert_eq!(items.len(), 3);

        // Check content types
        match &items[0] {
            crate::types::InputItem::Message(msg) => {
                assert_eq!(msg.role(), Role::Assistant);
                assert_eq!(msg.content(), Some("I'll help you with that. ".to_string()));
            }
            _ => panic!("Expected message"),
        }

        match &items[1] {
            crate::types::InputItem::FunctionCall(call) => {
                assert_eq!(call.name, "get_weather");
            }
            _ => panic!("Expected function call"),
        }

        match &items[2] {
            crate::types::InputItem::Message(msg) => {
                assert_eq!(msg.role(), Role::Assistant);
                assert_eq!(
                    msg.content(),
                    Some("Let me check that for you.".to_string())
                );
            }
            _ => panic!("Expected message"),
        }
    }
}
