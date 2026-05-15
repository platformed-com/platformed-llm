//! Delta accumulation for streaming responses.
//!
//! Every event names its target part by index. Reconstruction is a
//! straight-line dispatch — no implicit "currently-active part" state.

use crate::response::CompleteResponse;
use crate::types::{
    AssistantPart, FinishReason, FunctionCall, PartKind, PartUpdate, StreamEvent, Usage,
};
use crate::Error;

/// Reassembles a sequence of [`StreamEvent`]s into a [`CompleteResponse`].
///
/// Useful when you want to consume a stream incrementally but also produce
/// the final buffered response at the end.
#[derive(Debug, Default)]
pub struct ResponseAccumulator {
    parts: Vec<AssistantPart>,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
}

impl ResponseAccumulator {
    /// Create an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a single stream event. Returns an error if the event references
    /// a part index that wasn't opened by a preceding `PartStart`, or if the
    /// stream itself reported an error.
    pub fn process_event(&mut self, event: StreamEvent) -> Result<(), Error> {
        match event {
            StreamEvent::PartStart { index, kind } => {
                if index as usize != self.parts.len() {
                    return Err(Error::streaming(format!(
                        "PartStart out of order: index={index}, parts.len()={}",
                        self.parts.len()
                    )));
                }
                self.parts.push(open_part(kind));
            }
            StreamEvent::Delta { index, delta } => {
                let part = self.part_mut(index)?;
                append_delta(part, &delta);
            }
            StreamEvent::PartUpdate { index, update } => {
                let part = self.part_mut(index)?;
                apply_update(part, update);
            }
            StreamEvent::PartEnd { index } => {
                let part = self.part_mut(index)?;
                finalize_part(part);
            }
            StreamEvent::Done {
                finish_reason,
                usage,
            } => {
                self.finish_reason = Some(finish_reason);
                self.usage = Some(usage);
            }
            StreamEvent::Error { error } => {
                return Err(Error::streaming(error));
            }
        }
        Ok(())
    }

    fn part_mut(&mut self, index: u32) -> Result<&mut AssistantPart, Error> {
        let len = self.parts.len();
        self.parts.get_mut(index as usize).ok_or_else(|| {
            Error::streaming(format!(
                "stream event references unknown part index {index} (have {len} parts)",
            ))
        })
    }

    /// Consume the accumulator and produce the final response. If `Done` was
    /// never observed, the finish reason defaults to [`FinishReason::Stop`]
    /// and usage to zeros.
    pub fn finalize(self) -> Result<CompleteResponse, Error> {
        Ok(CompleteResponse {
            content: self.parts,
            finish_reason: self.finish_reason.unwrap_or(FinishReason::Stop),
            usage: self.usage.unwrap_or_default(),
        })
    }

    /// Concatenation of all accumulated text-part content so far. Intended
    /// for live previews while streaming is still in flight.
    pub fn current_content(&self) -> String {
        self.parts
            .iter()
            .filter_map(|p| match p {
                AssistantPart::Text { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect()
    }

    /// All function-call parts seen so far, cloned out. Note that the
    /// `arguments` JSON is only guaranteed to be complete once the
    /// corresponding `PartEnd` event has been processed.
    pub fn completed_function_calls(&self) -> Vec<FunctionCall> {
        self.parts
            .iter()
            .filter_map(|p| match p {
                AssistantPart::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect()
    }
}

fn open_part(kind: PartKind) -> AssistantPart {
    match kind {
        PartKind::Text => AssistantPart::Text {
            content: String::new(),
            annotations: Vec::new(),
        },
        PartKind::Reasoning => AssistantPart::Reasoning {
            content: String::new(),
            signature: None,
        },
        PartKind::RedactedReasoning { data } => AssistantPart::RedactedReasoning { data },
        PartKind::Refusal => AssistantPart::Refusal(String::new()),
        PartKind::ToolCall { call_id, name } => AssistantPart::ToolCall(FunctionCall {
            call_id,
            name,
            arguments: String::new(),
        }),
        PartKind::BuiltinToolCall { kind } => AssistantPart::BuiltinToolCall {
            kind,
            arguments: String::new(),
            result: None,
        },
        PartKind::Continuation(c) => AssistantPart::Continuation(c),
    }
}

fn append_delta(part: &mut AssistantPart, delta: &str) {
    match part {
        AssistantPart::Text { content, .. } => content.push_str(delta),
        AssistantPart::Reasoning { content, .. } => content.push_str(delta),
        AssistantPart::Refusal(content) => content.push_str(delta),
        AssistantPart::ToolCall(call) => call.arguments.push_str(delta),
        AssistantPart::BuiltinToolCall { arguments, .. } => arguments.push_str(delta),
        AssistantPart::RedactedReasoning { .. }
        | AssistantPart::Continuation(_)
        | AssistantPart::CacheBreakpoint => {}
    }
}

fn apply_update(part: &mut AssistantPart, update: PartUpdate) {
    match (part, update) {
        (AssistantPart::Reasoning { signature, .. }, PartUpdate::Signature(sig)) => {
            *signature = Some(sig);
        }
        (AssistantPart::Text { annotations, .. }, PartUpdate::Annotation(ann)) => {
            annotations.push(ann);
        }
        (AssistantPart::BuiltinToolCall { result, .. }, PartUpdate::BuiltinToolResult(r)) => {
            *result = Some(r);
        }
        _ => {}
    }
}

fn finalize_part(part: &mut AssistantPart) {
    if let AssistantPart::ToolCall(call) = part {
        if !call.arguments.is_empty() {
            if let Err(e) = serde_json::from_str::<serde_json::Value>(&call.arguments) {
                tracing::debug!(
                    call_id = %call.call_id,
                    error = %e,
                    "tool call arguments did not parse as JSON; passing through verbatim",
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_text_part() {
        let mut acc = ResponseAccumulator::new();
        acc.process_event(StreamEvent::PartStart {
            index: 0,
            kind: PartKind::Text,
        })
        .unwrap();
        acc.process_event(StreamEvent::Delta {
            index: 0,
            delta: "Hello, ".into(),
        })
        .unwrap();
        acc.process_event(StreamEvent::Delta {
            index: 0,
            delta: "world!".into(),
        })
        .unwrap();
        acc.process_event(StreamEvent::PartEnd { index: 0 })
            .unwrap();
        assert_eq!(acc.current_content(), "Hello, world!");
    }

    #[test]
    fn accumulates_tool_call_arguments_via_deltas() {
        let mut acc = ResponseAccumulator::new();
        acc.process_event(StreamEvent::PartStart {
            index: 0,
            kind: PartKind::ToolCall {
                call_id: "call_1".into(),
                name: "get_weather".into(),
            },
        })
        .unwrap();
        acc.process_event(StreamEvent::Delta {
            index: 0,
            delta: r#"{"city":"#.into(),
        })
        .unwrap();
        acc.process_event(StreamEvent::Delta {
            index: 0,
            delta: r#" "Paris"}"#.into(),
        })
        .unwrap();
        acc.process_event(StreamEvent::PartEnd { index: 0 })
            .unwrap();

        let calls = acc.completed_function_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, r#"{"city": "Paris"}"#);
    }

    #[test]
    fn accumulates_reasoning_with_signature() {
        let mut acc = ResponseAccumulator::new();
        acc.process_event(StreamEvent::PartStart {
            index: 0,
            kind: PartKind::Reasoning,
        })
        .unwrap();
        acc.process_event(StreamEvent::Delta {
            index: 0,
            delta: "Thinking...".into(),
        })
        .unwrap();
        acc.process_event(StreamEvent::PartUpdate {
            index: 0,
            update: PartUpdate::Signature("sig_abc".into()),
        })
        .unwrap();
        acc.process_event(StreamEvent::PartEnd { index: 0 })
            .unwrap();

        let response = acc.finalize().unwrap();
        match &response.content[0] {
            AssistantPart::Reasoning { signature, content } => {
                assert_eq!(content, "Thinking...");
                assert_eq!(signature.as_deref(), Some("sig_abc"));
            }
            _ => panic!("wrong part"),
        }
    }

    #[test]
    fn part_start_must_be_in_order() {
        let mut acc = ResponseAccumulator::new();
        let err = acc
            .process_event(StreamEvent::PartStart {
                index: 1,
                kind: PartKind::Text,
            })
            .expect_err("");
        assert!(err.to_string().contains("out of order"));
    }

    #[test]
    fn delta_to_unknown_index_errors() {
        let mut acc = ResponseAccumulator::new();
        let err = acc
            .process_event(StreamEvent::Delta {
                index: 0,
                delta: "hi".into(),
            })
            .expect_err("");
        assert!(err.to_string().contains("unknown part index"));
    }

    #[test]
    fn redacted_reasoning_one_shot() {
        let mut acc = ResponseAccumulator::new();
        acc.process_event(StreamEvent::PartStart {
            index: 0,
            kind: PartKind::RedactedReasoning {
                data: "opaque-blob".into(),
            },
        })
        .unwrap();
        acc.process_event(StreamEvent::PartEnd { index: 0 })
            .unwrap();

        let response = acc.finalize().unwrap();
        match &response.content[0] {
            AssistantPart::RedactedReasoning { data } => assert_eq!(data, "opaque-blob"),
            _ => panic!("wrong"),
        }
    }
}
