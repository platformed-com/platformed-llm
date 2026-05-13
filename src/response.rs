//! Response handling for LLM generations.

use crate::types::{
    AssistantPart, FinishReason, FunctionCall, InputItem, ProviderContinuation, Usage,
};
use crate::{Error, StreamEvent};
use futures_util::stream::Stream;
use std::pin::Pin;

/// A complete (buffered) response from an LLM provider.
#[derive(Debug, Clone)]
pub struct CompleteResponse {
    /// The assistant's output. Currently always a single
    /// [`OutputItem::Assistant`] (one assistant turn); modeled as a
    /// `Vec` for forward-compatibility.
    pub output: Vec<OutputItem>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    /// Provider-specific continuation hint for optional optimization on
    /// the next request — always optional, always safe to ignore.
    pub continuation: Option<ProviderContinuation>,
}

/// One assistant turn in the response output.
#[derive(Debug, Clone)]
pub enum OutputItem {
    Assistant { content: Vec<AssistantPart> },
}

impl OutputItem {
    pub fn to_input_item(&self) -> InputItem {
        match self {
            OutputItem::Assistant { content } => InputItem::Assistant {
                content: content.clone(),
            },
        }
    }
}

impl CompleteResponse {
    /// Concatenated text of all `AssistantPart::Text` parts.
    pub fn text(&self) -> String {
        self.output
            .iter()
            .flat_map(|item| match item {
                OutputItem::Assistant { content } => content.iter(),
            })
            .filter_map(|part| match part {
                AssistantPart::Text { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Concatenated text — back-compat alias for `text()`.
    pub fn content(&self) -> String {
        self.text()
    }

    /// All tool calls emitted by the assistant, in emit order.
    pub fn function_calls(&self) -> Vec<&FunctionCall> {
        self.output
            .iter()
            .flat_map(|item| match item {
                OutputItem::Assistant { content } => content.iter(),
            })
            .filter_map(|part| match part {
                AssistantPart::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect()
    }

    pub fn to_items(&self) -> Vec<InputItem> {
        self.output.iter().map(|item| item.to_input_item()).collect()
    }
}

/// A streaming response.
pub struct Response {
    stream: Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send>>,
    continuation: Option<ProviderContinuation>,
}

impl Response {
    pub fn from_stream<S>(stream: S) -> Self
    where
        S: Stream<Item = Result<StreamEvent, Error>> + Send + 'static,
    {
        Self {
            stream: Box::pin(stream),
            continuation: None,
        }
    }

    pub fn with_continuation<S>(stream: S, continuation: Option<ProviderContinuation>) -> Self
    where
        S: Stream<Item = Result<StreamEvent, Error>> + Send + 'static,
    {
        Self {
            stream: Box::pin(stream),
            continuation,
        }
    }

    pub async fn buffer(self) -> Result<CompleteResponse, Error> {
        Ok(self.collect().await?.1)
    }

    pub async fn text(self) -> Result<String, Error> {
        let complete = self.buffer().await?;
        Ok(complete.text())
    }

    /// Drain the stream and return BOTH the full event sequence and the
    /// accumulated [`CompleteResponse`]. Use when you need to render
    /// streaming UI events live AND want a buffered final result without
    /// double-consuming the response.
    pub async fn collect(self) -> Result<(Vec<StreamEvent>, CompleteResponse), Error> {
        let continuation = self.continuation.clone();
        let mut accumulator = crate::accumulator::ResponseAccumulator::new();
        let mut events = Vec::new();

        use futures_util::StreamExt;
        let mut stream = self.stream;
        while let Some(event_result) = stream.next().await {
            let event = event_result?;
            match &event {
                StreamEvent::Done { .. } => {
                    events.push(event.clone());
                    accumulator.process_event(event)?;
                    break;
                }
                StreamEvent::Error { error } => {
                    return Err(Error::streaming(error.clone()));
                }
                _ => {
                    events.push(event.clone());
                    accumulator.process_event(event)?;
                }
            }
        }

        let mut response = accumulator.finalize()?;
        response.continuation = continuation;
        Ok((events, response))
    }

    pub fn stream(self) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send>> {
        self.stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FunctionCall, PartKind, Usage};

    #[tokio::test]
    async fn buffers_a_text_only_response() {
        let events: Vec<Result<StreamEvent, Error>> = vec![
            Ok(StreamEvent::PartStart {
                index: 0,
                kind: PartKind::Text,
            }),
            Ok(StreamEvent::Delta {
                index: 0,
                delta: "Test response".to_string(),
            }),
            Ok(StreamEvent::PartEnd { index: 0 }),
            Ok(StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
            }),
        ];
        let stream = futures_util::stream::iter(events);
        let text = Response::from_stream(stream).text().await.unwrap();
        assert_eq!(text, "Test response");
    }

    #[tokio::test]
    async fn buffer_propagates_mid_stream_error() {
        let events: Vec<Result<StreamEvent, Error>> = vec![
            Ok(StreamEvent::PartStart {
                index: 0,
                kind: PartKind::Text,
            }),
            Ok(StreamEvent::Delta {
                index: 0,
                delta: "partial".to_string(),
            }),
            Err(Error::streaming("connection reset mid-stream")),
            Ok(StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
            }),
        ];
        let stream = futures_util::stream::iter(events);
        let err = Response::from_stream(stream).buffer().await.expect_err("");
        assert!(matches!(err, Error::Streaming(_)));
        assert!(err.to_string().contains("connection reset"));
    }

    #[tokio::test]
    async fn buffer_propagates_stream_error_event() {
        let events: Vec<Result<StreamEvent, Error>> = vec![
            Ok(StreamEvent::PartStart {
                index: 0,
                kind: PartKind::Text,
            }),
            Ok(StreamEvent::Delta {
                index: 0,
                delta: "partial".to_string(),
            }),
            Ok(StreamEvent::Error {
                error: "model internal error".to_string(),
            }),
            Ok(StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
            }),
        ];
        let stream = futures_util::stream::iter(events);
        let err = Response::from_stream(stream).buffer().await.expect_err("");
        assert!(err.to_string().contains("model internal error"));
    }

    #[test]
    fn text_concatenates_across_parts() {
        let response = CompleteResponse {
            output: vec![OutputItem::Assistant {
                content: vec![
                    AssistantPart::Text {
                        content: "Hello, ".to_string(),
                        annotations: Vec::new(),
                    },
                    AssistantPart::Text {
                        content: "world!".to_string(),
                        annotations: Vec::new(),
                    },
                ],
            }],
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
            continuation: None,
        };
        assert_eq!(response.text(), "Hello, world!");
    }

    #[tokio::test]
    async fn collect_returns_both_events_and_buffered_response() {
        let events: Vec<Result<StreamEvent, Error>> = vec![
            Ok(StreamEvent::PartStart {
                index: 0,
                kind: PartKind::Text,
            }),
            Ok(StreamEvent::Delta {
                index: 0,
                delta: "hello".to_string(),
            }),
            Ok(StreamEvent::PartEnd { index: 0 }),
            Ok(StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
            }),
        ];
        let stream = futures_util::stream::iter(events);
        let (events, complete) = Response::from_stream(stream).collect().await.unwrap();
        // Stream events include both lifecycle and content events.
        assert_eq!(events.len(), 4);
        assert!(matches!(events[0], StreamEvent::PartStart { .. }));
        // Buffered response has the accumulated text.
        assert_eq!(complete.text(), "hello");
    }

    #[test]
    fn function_calls_iter_returns_in_order() {
        let response = CompleteResponse {
            output: vec![OutputItem::Assistant {
                content: vec![
                    AssistantPart::ToolCall(FunctionCall {
                        call_id: "call_1".to_string(),
                        name: "get_weather".to_string(),
                        arguments: "{}".to_string(),
                    }),
                    AssistantPart::ToolCall(FunctionCall {
                        call_id: "call_2".to_string(),
                        name: "get_news".to_string(),
                        arguments: "{}".to_string(),
                    }),
                ],
            }],
            finish_reason: FinishReason::ToolCalls,
            usage: Usage::default(),
            continuation: None,
        };
        let calls = response.function_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "get_news");
    }
}
