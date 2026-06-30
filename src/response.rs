//! Response handling for LLM generations.

use crate::types::{
    AssistantPart, FinishReason, FunctionCall, InputItem, ProviderContinuation, Usage,
};
use crate::{Error, StreamEvent};
use futures_util::stream::Stream;
use std::pin::Pin;

/// A complete (buffered) response from an LLM provider — a single
/// assistant turn's worth of [`AssistantPart`]s plus terminal
/// metadata.
///
/// Fields are public; the lib doesn't maintain hidden invariants on
/// this struct. The accessor methods ([`Self::text`],
/// [`Self::function_calls`], [`Self::continuation`], [`Self::to_items`])
/// are convenience views over `content` — readers can pick whichever
/// is more ergonomic. Callers that need to mutate the response should
/// edit `content` directly.
#[derive(Debug, Clone)]
pub struct CompleteResponse {
    /// The assistant's emitted parts in order: text, reasoning, tool
    /// calls, continuation marker, etc.
    pub content: Vec<AssistantPart>,
    /// Why the model stopped generating.
    pub finish_reason: FinishReason,
    /// Token accounting for the turn.
    pub usage: Usage,
}

impl CompleteResponse {
    /// Concatenated text of all `AssistantPart::Text` parts.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|part| match part {
                AssistantPart::Text { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect()
    }

    /// `true` when the model stopped because it hit a token budget
    /// (`max_tokens` cap or the context window itself) rather than
    /// completing naturally. Tells callers the response was likely
    /// cut short and they should consider raising `max_tokens`,
    /// compacting the conversation, or retrying with a larger model.
    ///
    /// Mirrors `FinishReason::Length`; provided as a method so
    /// downstream code doesn't have to depend on the enum directly
    /// for this common check.
    pub fn was_truncated(&self) -> bool {
        matches!(self.finish_reason, FinishReason::Length)
    }

    /// All tool calls emitted by the assistant, in emit order.
    pub fn function_calls(&self) -> Vec<&FunctionCall> {
        self.content
            .iter()
            .filter_map(|part| match part {
                AssistantPart::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect()
    }

    /// The provider's resumption hint for the next conversation turn,
    /// if any. Surfaces the latest [`AssistantPart::Continuation`] in
    /// the assistant content. Always optional, always safe to ignore.
    pub fn continuation(&self) -> Option<&ProviderContinuation> {
        self.content.iter().rev().find_map(|part| match part {
            AssistantPart::Continuation(c) => Some(c),
            _ => None,
        })
    }

    /// Convert the response into a list of input items suitable for
    /// appending to the next [`crate::Prompt`]. Returns a single
    /// `InputItem::Assistant { content }`; any
    /// [`AssistantPart::Continuation`] inside is automatically picked
    /// up by the next same-provider request and elides prior history.
    pub fn to_items(&self) -> Vec<InputItem> {
        if self.content.is_empty() {
            Vec::new()
        } else {
            vec![InputItem::Assistant {
                content: self.content.clone(),
            }]
        }
    }
}

/// A streaming response.
pub struct Response {
    stream: Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send>>,
}

impl Response {
    /// Wrap an arbitrary stream of [`StreamEvent`]s as a [`Response`].
    /// Mainly useful for tests and for callers that drive their own
    /// transport.
    pub fn from_stream<S>(stream: S) -> Self
    where
        S: Stream<Item = Result<StreamEvent, Error>> + Send + 'static,
    {
        Self {
            stream: Box::pin(stream),
        }
    }

    /// Drain the stream and return the buffered [`CompleteResponse`].
    ///
    /// Unlike [`Self::collect`] this does not build (and clone every
    /// event into) an event log it would only discard — it feeds the
    /// accumulator by value. This is the common path behind
    /// [`Self::text`].
    pub async fn buffer(self) -> Result<CompleteResponse, Error> {
        use futures_util::StreamExt;
        let mut accumulator = crate::accumulator::ResponseAccumulator::new();
        let mut stream = self.stream;
        while let Some(event_result) = stream.next().await {
            let event = event_result?;
            let done = matches!(event, StreamEvent::Done { .. });
            accumulator.process_event(event)?;
            if done {
                break;
            }
        }
        accumulator.finalize()
    }

    /// Drain the stream and return the concatenated text of all text parts.
    pub async fn text(self) -> Result<String, Error> {
        let complete = self.buffer().await?;
        Ok(complete.text())
    }

    /// Drain the stream up to **and including** the terminal `Done`
    /// and return the event log alongside the buffered
    /// [`CompleteResponse`]. Any events a transport emits *after*
    /// `Done` are not collected (the log stops at `Done`, mirroring
    /// the buffered result).
    ///
    /// Returns *after* the stream completes — the `Vec<StreamEvent>` is
    /// a post-hoc record, not a live feed. Use it for inspection,
    /// snapshot testing, or audit logging. For the buffered result
    /// alone prefer [`Self::buffer`] (no event-log allocation). If you
    /// need live event handling while the model streams, consume
    /// [`Self::stream`] directly and feed events into a
    /// [`crate::accumulator::ResponseAccumulator`] yourself.
    pub async fn collect(self) -> Result<(Vec<StreamEvent>, CompleteResponse), Error> {
        let mut accumulator = crate::accumulator::ResponseAccumulator::new();
        let mut events = Vec::new();

        use futures_util::StreamExt;
        let mut stream = self.stream;
        while let Some(event_result) = stream.next().await {
            let event = event_result?;
            let done = matches!(event, StreamEvent::Done { .. });
            events.push(event.clone());
            accumulator.process_event(event)?;
            if done {
                break;
            }
        }

        let response = accumulator.finalize()?;
        Ok((events, response))
    }

    /// Unwrap to the raw event stream for direct consumption.
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

    /// A mid-stream `Err` must propagate out of `buffer` and discard
    /// any events that arrive after it — including a `Done`. Without
    /// the short-circuit, a malformed provider that emitted both an
    /// `Err` *and* a `Done` could trick callers into seeing a
    /// successful finish.
    #[tokio::test]
    async fn buffer_propagates_mid_stream_error_and_stops_at_err() {
        let events: Vec<Result<StreamEvent, Error>> = vec![
            Ok(StreamEvent::PartStart {
                index: 0,
                kind: PartKind::Text,
            }),
            Ok(StreamEvent::Delta {
                index: 0,
                delta: "partial".to_string(),
            }),
            Err(Error::provider("OpenAI", "connection reset mid-stream")),
            // A spurious post-error Done that buffer() must NOT see.
            Ok(StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
            }),
        ];
        let stream = futures_util::stream::iter(events);
        let err = Response::from_stream(stream).buffer().await.expect_err("");
        assert!(
            matches!(
                err,
                Error::Provider {
                    provider: "OpenAI",
                    ..
                }
            ),
            "must propagate the upstream provider name, got {err:?}",
        );
        assert!(err.to_string().contains("connection reset"));
    }

    #[test]
    fn was_truncated_reports_length_finish_reason() {
        let empty_text = AssistantPart::Text {
            content: String::new(),
            annotations: Vec::new(),
        };
        let truncated = CompleteResponse {
            content: vec![empty_text.clone()],
            finish_reason: FinishReason::Length,
            usage: Usage::default(),
        };
        assert!(truncated.was_truncated());

        for reason in [
            FinishReason::Stop,
            FinishReason::ToolCalls,
            FinishReason::ContentFilter,
        ] {
            let r = CompleteResponse {
                content: vec![empty_text.clone()],
                finish_reason: reason,
                usage: Usage::default(),
            };
            assert!(
                !r.was_truncated(),
                "was_truncated should only fire for Length"
            );
        }
    }

    #[test]
    fn usage_total_tokens_sums_input_and_output() {
        let usage = Usage {
            input_tokens: 100,
            output_tokens: 50,
            ..Usage::default()
        };
        assert_eq!(usage.total_tokens(), 150);
        assert_eq!(Usage::default().total_tokens(), 0);
    }

    #[test]
    fn text_concatenates_across_parts() {
        let response = CompleteResponse {
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
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
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

    /// `to_items()` wraps the response's content in a single
    /// `InputItem::Assistant`, preserving any `Continuation` part so
    /// the next same-provider request picks it up and elides prior
    /// history.
    #[test]
    fn to_items_includes_continuation_marker() {
        use crate::types::{InputItem, ProviderContinuation};
        let response = CompleteResponse {
            content: vec![
                AssistantPart::Text {
                    content: "hi".into(),
                    annotations: Vec::new(),
                },
                AssistantPart::Continuation(ProviderContinuation::OpenAI {
                    response_id: "resp_42".into(),
                }),
            ],
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        };
        let items = response.to_items();
        assert_eq!(items.len(), 1);
        match &items[0] {
            InputItem::Assistant { content } => {
                assert_eq!(content.len(), 2);
                assert!(matches!(&content[0], AssistantPart::Text { .. }));
                match &content[1] {
                    AssistantPart::Continuation(ProviderContinuation::OpenAI { response_id }) => {
                        assert_eq!(response_id, "resp_42");
                    }
                    other => panic!("expected continuation part, got {other:?}"),
                }
            }
            other => panic!("expected assistant item, got {other:?}"),
        }
        // And the accessor matches.
        assert!(matches!(
            response.continuation(),
            Some(ProviderContinuation::OpenAI { response_id }) if response_id == "resp_42"
        ));
    }

    #[test]
    fn function_calls_iter_returns_in_order() {
        let response = CompleteResponse {
            content: vec![
                AssistantPart::ToolCall(FunctionCall {
                    call_id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: "{}".to_string(),
                    provider_signature: None,
                }),
                AssistantPart::ToolCall(FunctionCall {
                    call_id: "call_2".to_string(),
                    name: "get_news".to_string(),
                    arguments: "{}".to_string(),
                    provider_signature: None,
                }),
            ],
            finish_reason: FinishReason::ToolCalls,
            usage: Usage::default(),
        };
        let calls = response.function_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "get_news");
    }
}
