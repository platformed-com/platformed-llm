//! Text a model writes after a tool call, in the same turn, was written
//! before the call's result existed: a JSON answer or a claim that the
//! work is done, either way premature. Text before the turn's first call
//! is a preamble.
//!
//! `drop_text_after_tool_calls` enforces this over any provider's event
//! stream as it arrives. Each text part is judged at its `PartStart`, so
//! nothing is held: a part that starts after the turn's first `ToolCall`
//! is dropped with its deltas and annotations, and later parts are
//! renumbered to keep the part indices contiguous.

use std::borrow::Cow;
use std::collections::HashMap;
use std::pin::Pin;

use futures_util::stream::{Stream, StreamExt};

use crate::types::{PartKind, RawConfig, Tool};
use crate::{Capabilities, Error, Prompt, Response, StreamEvent};

use super::{Middleware, ResponseTransform};

pub(crate) type EventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send>>;

/// Applies `drop_text_after_tool_calls` to a request with function
/// tools; a turn without them makes no tool call.
#[derive(Debug, Default)]
pub struct TextAfterToolCallsMiddleware;

impl Middleware for TextAfterToolCallsMiddleware {
    fn name(&self) -> &str {
        "text_after_tool_calls"
    }

    fn apply<'a>(
        &self,
        _prompt: &mut Cow<'a, Prompt>,
        config: &mut Cow<'a, RawConfig>,
        _caps: &Capabilities,
    ) -> Result<Option<ResponseTransform>, Error> {
        let has_function_tools = config
            .tools
            .as_ref()
            .is_some_and(|tools| tools.iter().any(|t| matches!(t, Tool::Function(_))));
        if !has_function_tools {
            return Ok(None);
        }
        let model = config.model.clone();
        Ok(Some(Box::new(move |response: Response| {
            Response::from_stream(drop_text_after_tool_calls(response.stream(), model))
        })))
    }
}

/// See the module docs.
pub(crate) fn drop_text_after_tool_calls(inner: EventStream, model: String) -> EventStream {
    let mut dropper = TextDropper {
        model,
        slots: HashMap::new(),
        next_index: 0,
        saw_tool_call: false,
    };
    Box::pin(
        inner
            .map(move |event| event.map(|event| dropper.on_event(event)))
            .filter_map(|event| async move {
                match event {
                    Ok(Some(event)) => Some(Ok(event)),
                    Ok(None) => None,
                    Err(error) => Some(Err(error)),
                }
            }),
    )
}

enum Slot {
    Sent(u32),
    Dropped { bytes: usize },
}

struct TextDropper {
    model: String,
    slots: HashMap<u32, Slot>,
    next_index: u32,
    saw_tool_call: bool,
}

impl TextDropper {
    fn on_event(&mut self, event: StreamEvent) -> Option<StreamEvent> {
        let index = match &event {
            StreamEvent::PartStart { index, kind } => {
                let slot = if *kind == PartKind::Text && self.saw_tool_call {
                    Slot::Dropped { bytes: 0 }
                } else {
                    self.saw_tool_call |= matches!(kind, PartKind::ToolCall { .. });
                    self.next_index += 1;
                    Slot::Sent(self.next_index - 1)
                };
                self.slots.insert(*index, slot);
                *index
            }
            StreamEvent::Delta { index, .. }
            | StreamEvent::PartUpdate { index, .. }
            | StreamEvent::PartEnd { index } => *index,
            StreamEvent::Done { .. } => return Some(event),
        };
        match self.slots.get_mut(&index) {
            Some(Slot::Sent(sent)) => Some(reindex(event, *sent)),
            Some(Slot::Dropped { bytes }) => {
                match &event {
                    StreamEvent::Delta { delta, .. } => *bytes += delta.len(),
                    StreamEvent::PartEnd { .. } => tracing::debug!(
                        model = %self.model,
                        bytes = *bytes,
                        "dropping text written after a tool call",
                    ),
                    _ => {}
                }
                None
            }
            None => Some(event),
        }
    }
}

/// Move a part's event to part `index`.
pub(crate) fn reindex(event: StreamEvent, index: u32) -> StreamEvent {
    match event {
        StreamEvent::PartStart { kind, .. } => StreamEvent::PartStart { index, kind },
        StreamEvent::Delta { delta, .. } => StreamEvent::Delta { index, delta },
        StreamEvent::PartUpdate { update, .. } => StreamEvent::PartUpdate { index, update },
        StreamEvent::PartEnd { .. } => StreamEvent::PartEnd { index },
        StreamEvent::Done { .. } => event,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AssistantPart, FinishReason, Usage};
    use crate::CompleteResponse;

    fn text(index: u32, deltas: &[&str]) -> Vec<StreamEvent> {
        let mut events = vec![StreamEvent::PartStart {
            index,
            kind: PartKind::Text,
        }];
        events.extend(deltas.iter().map(|delta| StreamEvent::Delta {
            index,
            delta: delta.to_string(),
        }));
        events.push(StreamEvent::PartEnd { index });
        events
    }

    fn call(index: u32) -> Vec<StreamEvent> {
        vec![
            StreamEvent::PartStart {
                index,
                kind: PartKind::ToolCall {
                    call_id: format!("c{index}"),
                    name: "lookup".into(),
                },
            },
            StreamEvent::Delta {
                index,
                delta: "{}".into(),
            },
            StreamEvent::PartEnd { index },
        ]
    }

    fn done() -> StreamEvent {
        StreamEvent::Done {
            finish_reason: FinishReason::ToolCalls,
            usage: Usage::default(),
        }
    }

    async fn turn(parts: Vec<Vec<StreamEvent>>) -> CompleteResponse {
        let mut events: Vec<Result<StreamEvent, Error>> =
            parts.into_iter().flatten().map(Ok).collect();
        events.push(Ok(done()));
        let stream = drop_text_after_tool_calls(
            Box::pin(futures_util::stream::iter(events)),
            "model".into(),
        );
        Response::from_stream(stream).buffer().await.unwrap()
    }

    /// Text parts as `"text"` and calls as their id, in part order.
    fn shape(response: &CompleteResponse) -> Vec<String> {
        response
            .content
            .iter()
            .map(|part| match part {
                AssistantPart::Text { content, .. } => format!("text:{content}"),
                AssistantPart::ToolCall(call) => call.call_id.clone(),
                part => panic!("unexpected part {part:?}"),
            })
            .collect()
    }

    /// Feed `events` through the stage without ending the turn, and
    /// return the first `n` events it emits.
    async fn emitted_before_done(events: Vec<StreamEvent>, n: usize) -> Vec<StreamEvent> {
        let (tx, rx) = futures::channel::mpsc::unbounded();
        for event in events {
            tx.unbounded_send(Ok(event)).unwrap();
        }
        let mut out = drop_text_after_tool_calls(Box::pin(rx), "model".into());
        let mut emitted = Vec::new();
        for _ in 0..n {
            emitted.push(out.next().await.unwrap().unwrap());
        }
        emitted
    }

    #[tokio::test]
    async fn a_preamble_before_a_call_streams_and_is_kept() {
        let emitted = emitted_before_done(text(0, &["Checking", " now."]), 4).await;
        assert!(
            matches!(&emitted[1], StreamEvent::Delta { index: 0, delta } if delta == "Checking"),
            "{emitted:?}"
        );
        let response = turn(vec![text(0, &["Checking now."]), call(1)]).await;
        assert_eq!(shape(&response), ["text:Checking now.", "c1"]);
    }

    #[tokio::test]
    async fn an_answer_only_turn_streams() {
        let emitted = emitted_before_done(text(0, &[r#"{"answer":"#, r#""done"}"#]), 3).await;
        assert!(
            matches!(&emitted[2], StreamEvent::Delta { delta, .. } if delta == r#""done"}"#),
            "{emitted:?}"
        );
    }

    #[tokio::test]
    async fn text_after_a_call_is_dropped() {
        let response = turn(vec![call(0), text(1, &[r#"{"answer":"done"}"#])]).await;
        assert_eq!(shape(&response), ["c0"]);
    }

    #[tokio::test]
    async fn text_between_two_calls_is_dropped() {
        let response = turn(vec![call(0), text(1, &["It now passes."]), call(2)]).await;
        assert_eq!(shape(&response), ["c0", "c2"]);
    }

    #[tokio::test]
    async fn parts_after_a_dropped_text_are_renumbered() {
        let response = turn(vec![
            text(0, &["Checking."]),
            call(1),
            text(2, &["Done."]),
            call(3),
            call(4),
        ])
        .await;
        assert_eq!(shape(&response), ["text:Checking.", "c1", "c3", "c4"]);
    }

    /// Chat with tools on a provider that writes its text before its
    /// calls passes through the default middleware unchanged.
    #[tokio::test]
    async fn text_before_calls_passes_through_the_default_middleware() {
        use crate::providers::mock::{MockProvider, MockResponse};
        use crate::types::FunctionCall;

        let provider = MockProvider::always(MockResponse::from_parts(
            vec![
                AssistantPart::Text {
                    content: "Let me look that up.".into(),
                    annotations: Vec::new(),
                },
                AssistantPart::ToolCall(FunctionCall {
                    call_id: "c1".into(),
                    name: "lookup".into(),
                    arguments: "{}".into(),
                    provider_signature: None,
                }),
            ],
            FinishReason::ToolCalls,
        ));
        let config = crate::Config::builder("claude-sonnet-4-5")
            .tools(vec![Tool::function(
                "lookup",
                None,
                Cow::Owned(serde_json::value::RawValue::from_string("{}".into()).unwrap()),
            )])
            .build();
        let response = crate::generate(&provider, &Prompt::user("hi"), &config)
            .await
            .unwrap()
            .buffer()
            .await
            .unwrap();
        assert_eq!(shape(&response), ["text:Let me look that up.", "c1"]);
        assert_eq!(response.finish_reason, FinishReason::ToolCalls);
    }
}
