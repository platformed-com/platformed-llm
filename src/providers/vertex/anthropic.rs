use futures_util::StreamExt;
use std::collections::HashMap;

use super::anthropic_types::*;
use super::transport::VertexTransport;
use crate::provider::LLMProvider;
use crate::sse_stream::SseStream;
use crate::types::{FinishReason, FunctionCall, InputItem, Role, Usage};
use crate::{Error, LLMRequest, Response, StreamEvent};

/// Anthropic Claude provider implementation via Vertex AI.
pub struct AnthropicViaVertexProvider {
    transport: VertexTransport,
}

impl AnthropicViaVertexProvider {
    /// Create a new Anthropic provider with access token authentication.
    pub fn new(project_id: String, location: String, access_token: String) -> Result<Self, Error> {
        Ok(Self {
            transport: VertexTransport::with_access_token(project_id, location, access_token)?,
        })
    }

    /// Create a new Anthropic provider with a custom base URL (for testing).
    pub fn new_with_base_url(
        project_id: String,
        location: String,
        access_token: String,
        base_url: String,
    ) -> Result<Self, Error> {
        Ok(Self {
            transport: VertexTransport::with_access_token(project_id, location, access_token)?
                .with_base_url(base_url),
        })
    }

    /// Create a new Anthropic provider with Application Default Credentials.
    pub async fn with_adc(project_id: String, location: String) -> Result<Self, Error> {
        Ok(Self {
            transport: VertexTransport::with_adc(project_id, location).await?,
        })
    }

    /// Convert internal request to Anthropic format.
    fn convert_request(&self, request: &LLMRequest) -> Result<AnthropicRequest, Error> {
        let mut messages = Vec::new();
        let mut system_message = None;

        for item in &request.messages {
            match item {
                InputItem::Message(msg) => {
                    match msg.role {
                        Role::System => {
                            // Anthropic uses separate system field for system messages
                            system_message = Some(msg.content.clone());
                        }
                        Role::User => {
                            messages.push(AnthropicMessage {
                                role: "user".to_string(),
                                content: AnthropicContent::Text(msg.content.clone()),
                            });
                        }
                        Role::Assistant => {
                            messages.push(AnthropicMessage {
                                role: "assistant".to_string(),
                                content: AnthropicContent::Text(msg.content.clone()),
                            });
                        }
                    }
                }
                InputItem::FunctionCall(call) => {
                    // Add tool use to the last assistant response or create a new one
                    let tool_use_block = AnthropicContentBlock::ToolUse {
                        id: call.call_id.clone(),
                        name: call.name.clone(),
                        input: serde_json::from_str(&call.arguments).map_err(|e| {
                            Error::provider(
                                "Anthropic",
                                format!("Invalid function arguments: {e}"),
                            )
                        })?,
                    };

                    if let Some(last_msg) = messages.last_mut() {
                        if last_msg.role == "assistant" {
                            // Convert existing content to blocks and add tool use
                            match &mut last_msg.content {
                                AnthropicContent::Text(text) => {
                                    let mut blocks =
                                        vec![AnthropicContentBlock::Text { text: text.clone() }];
                                    blocks.push(tool_use_block);
                                    last_msg.content = AnthropicContent::Blocks(blocks);
                                }
                                AnthropicContent::Blocks(blocks) => {
                                    blocks.push(tool_use_block);
                                }
                            }
                        } else {
                            // Create new assistant message with tool use
                            messages.push(AnthropicMessage {
                                role: "assistant".to_string(),
                                content: AnthropicContent::Blocks(vec![tool_use_block]),
                            });
                        }
                    } else {
                        // Create new assistant message with tool use
                        messages.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: AnthropicContent::Blocks(vec![tool_use_block]),
                        });
                    }
                }
                InputItem::FunctionCallOutput { call_id, output } => {
                    // Add tool result to a user message
                    let tool_result_block = AnthropicContentBlock::ToolResult {
                        tool_use_id: call_id.clone(),
                        content: output.clone(),
                    };

                    // Check if the last message is already a user message with tool results
                    let should_append = if let Some(last_msg) = messages.last() {
                        last_msg.role == "user"
                            && match &last_msg.content {
                                AnthropicContent::Blocks(blocks) => blocks.iter().any(|b| {
                                    matches!(b, AnthropicContentBlock::ToolResult { .. })
                                }),
                                _ => false,
                            }
                    } else {
                        false
                    };

                    if should_append {
                        // Add to existing user message with tool results
                        if let Some(last_msg) = messages.last_mut() {
                            if let AnthropicContent::Blocks(blocks) = &mut last_msg.content {
                                blocks.push(tool_result_block);
                            }
                        }
                    } else {
                        // Create new user message with tool result
                        messages.push(AnthropicMessage {
                            role: "user".to_string(),
                            content: AnthropicContent::Blocks(vec![tool_result_block]),
                        });
                    }
                }
            }
        }

        let tools = request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|tool| AnthropicTool {
                    name: tool.function.name.clone(),
                    description: tool.function.description.clone(),
                    input_schema: tool.function.parameters.clone(),
                })
                .collect()
        });

        let anthropic_request = AnthropicRequest {
            messages,
            max_tokens: request.max_tokens.unwrap_or(1024),
            anthropic_version: "vertex-2023-10-16".to_string(),
            system: system_message,
            temperature: request.temperature,
            top_p: request.top_p,
            tools,
            stream: Some(true), // Enable streaming for SSE responses
        };

        Ok(anthropic_request)
    }
}

#[async_trait::async_trait]
impl LLMProvider for AnthropicViaVertexProvider {
    async fn generate(&self, request: &LLMRequest) -> Result<Response, Error> {
        let anthropic_request = self.convert_request(request)?;

        let endpoint = self.transport.endpoint(
            "anthropic",
            &request.model,
            "streamRawPredict",
            Some("alt=sse"),
        );

        let builder = self
            .transport
            .client()
            .post(&endpoint)
            .header("Content-Type", "application/json")
            .json(&anthropic_request);
        let request_builder = self.transport.authorize(builder).await?;

        let response = request_builder.send().await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(Error::provider(
                "Anthropic",
                format!("API error: {error_text}"),
            ));
        }

        // Create SSE stream from response
        let byte_stream = response.bytes_stream();
        let sse_stream = SseStream::new(byte_stream);

        // Create a stateful processor for function call tracking
        let mut state = StreamState::default();

        let event_stream = sse_stream
            .map(move |sse_result| {
                match sse_result {
                    Ok(sse_event) => {
                        let data = sse_event.data.trim();

                        // Skip empty events
                        if data.is_empty() {
                            return vec![];
                        }

                        // Parse the SSE data as Anthropic stream event
                        match serde_json::from_str::<AnthropicStreamEvent>(data) {
                            Ok(stream_event) => {
                                match convert_stream_event_stateful(stream_event, &mut state) {
                                    Ok(events) => events.into_iter().map(Ok).collect(),
                                    Err(e) => vec![Err(e)],
                                }
                            }
                            Err(e) => {
                                // Skip unparseable events (might be connection keep-alive or other data)
                                if !data.starts_with('{') {
                                    vec![]
                                } else {
                                    vec![Err(Error::provider(
                                        "Anthropic",
                                        format!("Failed to parse SSE event: {e}"),
                                    ))]
                                }
                            }
                        }
                    }
                    Err(e) => vec![Err(e)],
                }
            })
            .map(|events| futures_util::stream::iter(events.into_iter()))
            .flatten();

        Ok(Response::from_stream(event_stream))
    }
}

/// State for tracking streaming progress across content blocks and the
/// message-level metadata that arrives on `message_delta`.
///
/// Anthropic delivers `stop_reason` and the cumulative `usage` on
/// `message_delta` events, which fire **before** `message_stop`. We have to
/// stash them here so the final `Done` event we synthesize at `message_stop`
/// reflects what the model actually said.
#[derive(Debug, Default)]
pub(crate) struct StreamState {
    /// In-progress function calls indexed by content block index.
    in_progress_calls: HashMap<u32, InProgressFunctionCall>,
    /// Cumulative usage merged from `message_start` and `message_delta`.
    pending_usage: Usage,
    /// `stop_reason` captured from `message_delta`.
    pending_stop_reason: Option<String>,
}

/// Map an Anthropic `stop_reason` string onto our unified [`FinishReason`].
///
/// Until [`FinishReason`] is extended (Phase 5), `stop_sequence` and
/// `pause_turn` collapse to `Stop` — the closest existing variant.
pub(crate) fn map_anthropic_stop_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("end_turn") => FinishReason::Stop,
        Some("tool_use") => FinishReason::ToolCalls,
        Some("max_tokens") => FinishReason::Length,
        Some("stop_sequence") => FinishReason::Stop,
        Some("pause_turn") => FinishReason::Stop,
        Some("refusal") => FinishReason::ContentFilter,
        Some(other) => {
            tracing::warn!(stop_reason = other, "unknown Anthropic stop_reason");
            FinishReason::Stop
        }
        None => FinishReason::Stop,
    }
}

/// Merge an Anthropic `usage` object into the running [`Usage`] tally.
///
/// Anthropic's streaming protocol reports `input_tokens` once on
/// `message_start` and a cumulative `output_tokens` on the final
/// `message_delta`. We always overwrite with the latest non-`None` value so
/// the `Done` event reflects the model's authoritative final counts.
fn merge_anthropic_usage(target: &mut Usage, src: &AnthropicUsage) {
    if let Some(t) = src.input_tokens {
        target.input_tokens = t;
    }
    if let Some(t) = src.output_tokens {
        target.output_tokens = t;
    }
    // Note: `cache_creation_input_tokens` is currently aliased onto
    // `cached_tokens` for parity with the existing `From` impl; Phase 3.4 will
    // split this out properly into separate cache_read / cache_create fields.
    if let Some(t) = src.cache_creation_input_tokens {
        target.cached_tokens = Some(t);
    }
}

/// A function call that's being built incrementally from streaming events.
///
/// Per the streaming protocol the `input` field on `content_block_start` is
/// always `{}` — the actual JSON arrives in `input_json_delta` chunks. We
/// always start with an empty buffer and append every delta verbatim.
#[derive(Debug)]
struct InProgressFunctionCall {
    id: String,
    name: String,
    input_buffer: String,
}

/// Convert an Anthropic stream event into our unified `StreamEvent`s, with
/// state tracking for incremental function-call argument accumulation.
///
/// `pub(crate)` so unit tests can drive this directly with synthetic events
/// rather than going through the full SSE plumbing.
pub(crate) fn convert_stream_event_stateful(
    event: AnthropicStreamEvent,
    state: &mut StreamState,
) -> Result<Vec<StreamEvent>, Error> {
    let mut events = Vec::new();

    match event {
        AnthropicStreamEvent::MessageStart { message } => {
            // `message_start` carries the initial `input_tokens`.
            if let Some(usage) = &message.usage {
                merge_anthropic_usage(&mut state.pending_usage, usage);
            }
        }
        AnthropicStreamEvent::ContentBlockStart {
            content_block,
            index,
        } => {
            match content_block {
                AnthropicContentBlock::ToolUse { id, name, input } => {
                    events.push(StreamEvent::OutputItemAdded {
                        item: crate::types::OutputItemInfo::FunctionCall {
                            name: name.clone(),
                            id: id.clone(),
                        },
                    });

                    // Per the streaming protocol the `input` here is always
                    // `{}`. Anything else is off-spec — log and proceed with
                    // an empty buffer; `input_json_delta` events are the
                    // source of truth.
                    let nonempty = !(input.is_null()
                        || (input.is_object()
                            && input.as_object().map(|o| o.is_empty()).unwrap_or(true)));
                    if nonempty {
                        tracing::warn!(
                            ?input,
                            "Anthropic content_block_start carried non-empty `input`; \
                             ignoring and relying on input_json_delta accumulation"
                        );
                    }

                    state.in_progress_calls.insert(
                        index,
                        InProgressFunctionCall {
                            id: id.clone(),
                            name: name.clone(),
                            input_buffer: String::new(),
                        },
                    );
                }
                AnthropicContentBlock::Text { text } => {
                    events.push(StreamEvent::OutputItemAdded {
                        item: crate::types::OutputItemInfo::Text,
                    });
                    // Handle initial text content if any
                    if !text.is_empty() {
                        events.push(StreamEvent::ContentDelta { delta: text });
                    }
                }
                AnthropicContentBlock::ToolResult { .. } => {
                    // Tool results are handled in request construction, not in responses
                }
            }
        }
        AnthropicStreamEvent::ContentBlockDelta { delta, index } => match delta {
            AnthropicContentDelta::TextDelta { text } => {
                if !text.is_empty() {
                    events.push(StreamEvent::ContentDelta { delta: text });
                }
            }
            AnthropicContentDelta::InputJsonDelta { partial_json } => {
                if let Some(in_progress) = state.in_progress_calls.get_mut(&index) {
                    in_progress.input_buffer.push_str(&partial_json);
                }
            }
        },
        AnthropicStreamEvent::ContentBlockStop { index } => {
            // Content block finished - emit FunctionCallComplete if this was a function call
            if let Some(in_progress) = state.in_progress_calls.remove(&index) {
                let function_call = FunctionCall {
                    call_id: in_progress.id, // Use the same ID
                    name: in_progress.name,
                    arguments: in_progress.input_buffer,
                };
                events.push(StreamEvent::FunctionCallComplete {
                    call: function_call,
                });
            }
        }
        AnthropicStreamEvent::MessageDelta { delta, usage } => {
            // `message_delta` carries the canonical `stop_reason` and the
            // cumulative `output_tokens`. Stash them — the final `Done` event
            // gets emitted on `message_stop`.
            if let Some(reason) = delta.stop_reason {
                state.pending_stop_reason = Some(reason);
            }
            if let Some(usage) = usage {
                merge_anthropic_usage(&mut state.pending_usage, &usage);
            }
        }
        AnthropicStreamEvent::MessageStop => {
            let finish_reason =
                map_anthropic_stop_reason(state.pending_stop_reason.as_deref());
            let usage = std::mem::take(&mut state.pending_usage);
            events.push(StreamEvent::Done {
                finish_reason,
                usage,
            });
        }
        AnthropicStreamEvent::Ping => {
            // Keep-alive event - ignore
        }
    }

    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{stream, StreamExt};

    fn parse(json: &str) -> AnthropicStreamEvent {
        serde_json::from_str(json).expect("event JSON should parse")
    }

    fn drain(events: Vec<AnthropicStreamEvent>) -> Vec<StreamEvent> {
        let mut state = StreamState::default();
        let mut out = Vec::new();
        for event in events {
            out.extend(
                convert_stream_event_stateful(event, &mut state)
                    .expect("conversion should succeed"),
            );
        }
        out
    }

    /// `message_delta` carries the canonical `stop_reason` (and on the wire,
    /// `usage` is a sibling of `delta`, not nested inside it). `message_stop`
    /// must surface those fields on the final `Done` event — the previous
    /// implementation hard-coded `Stop` / `Usage::default()` regardless.
    #[test]
    fn message_stop_emits_actual_finish_reason_and_usage() {
        let events = drain(vec![
            parse(
                r#"{
                    "type":"message_start",
                    "message":{
                        "id":"m1","model":"c","role":"assistant","content":[],
                        "usage":{"input_tokens":10,"output_tokens":0}
                    }
                }"#,
            ),
            parse(
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            ),
            parse(
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
            ),
            parse(r#"{"type":"content_block_stop","index":0}"#),
            parse(
                r#"{
                    "type":"message_delta",
                    "delta":{"stop_reason":"tool_use","stop_sequence":null},
                    "usage":{"output_tokens":42}
                }"#,
            ),
            parse(r#"{"type":"message_stop"}"#),
        ]);

        let done = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::Done {
                    finish_reason,
                    usage,
                } => Some((finish_reason, usage)),
                _ => None,
            })
            .expect("expected a Done event");

        assert_eq!(
            *done.0,
            FinishReason::ToolCalls,
            "stop_reason 'tool_use' should map to FinishReason::ToolCalls",
        );
        assert_eq!(
            done.1.input_tokens, 10,
            "input_tokens should be carried from message_start",
        );
        assert_eq!(
            done.1.output_tokens, 42,
            "output_tokens should be carried from message_delta",
        );
    }

    /// Per the streaming protocol, multiple `input_json_delta` events for one
    /// tool_use block must **accumulate** — the JSON string is delivered in
    /// pieces. The previous implementation had a path that *replaced* the
    /// buffer on each delta whenever `content_block_start` had carried any
    /// initial input (real or imagined), which silently dropped earlier
    /// deltas on the floor.
    #[test]
    fn multiple_input_json_deltas_concatenate_not_replace() {
        let mut state = StreamState::default();
        // Off-spec but historically triggered the buggy "replace" branch.
        let evt = parse(
            r#"{"type":"content_block_start","index":0,
                "content_block":{"type":"tool_use","id":"toolu_1","name":"f","input":{"a":1}}}"#,
        );
        let _ = convert_stream_event_stateful(evt, &mut state).unwrap();
        let evt = parse(
            r#"{"type":"content_block_delta","index":0,
                "delta":{"type":"input_json_delta","partial_json":"FIRST"}}"#,
        );
        let _ = convert_stream_event_stateful(evt, &mut state).unwrap();
        let evt = parse(
            r#"{"type":"content_block_delta","index":0,
                "delta":{"type":"input_json_delta","partial_json":"SECOND"}}"#,
        );
        let _ = convert_stream_event_stateful(evt, &mut state).unwrap();
        let evt = parse(r#"{"type":"content_block_stop","index":0}"#);
        let events = convert_stream_event_stateful(evt, &mut state).unwrap();

        let call = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::FunctionCallComplete { call } => Some(call),
                _ => None,
            })
            .expect("expected FunctionCallComplete");
        assert!(
            call.arguments.contains("FIRST") && call.arguments.contains("SECOND"),
            "deltas must accumulate, got: {:?}",
            call.arguments,
        );
    }

    #[test]
    fn finish_reason_mapping_covers_known_stop_reasons() {
        let cases = [
            ("end_turn", FinishReason::Stop),
            ("tool_use", FinishReason::ToolCalls),
            ("max_tokens", FinishReason::Length),
            ("stop_sequence", FinishReason::Stop),
            ("pause_turn", FinishReason::Stop),
            ("refusal", FinishReason::ContentFilter),
        ];
        for (input, expected) in cases {
            assert_eq!(
                map_anthropic_stop_reason(Some(input)),
                expected,
                "stop_reason {input:?}",
            );
        }
    }

    #[tokio::test]
    async fn test_streaming_content_parsing() {
        // Simulate realistic Anthropic streaming response chunks
        let start_event = r#"{"type":"message_start","message":{"id":"msg_123","model":"claude-sonnet-4","role":"assistant","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let content_start =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let text_delta1 = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let text_delta2 = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}"#;
        let content_stop = r#"{"type":"content_block_stop","index":0}"#;
        let message_stop = r#"{"type":"message_stop"}"#;

        let byte_chunks: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
            Ok(bytes::Bytes::from(format!("data: {start_event}\n\n"))),
            Ok(bytes::Bytes::from(format!("data: {content_start}\n\n"))),
            Ok(bytes::Bytes::from(format!("data: {text_delta1}\n\n"))),
            Ok(bytes::Bytes::from(format!("data: {text_delta2}\n\n"))),
            Ok(bytes::Bytes::from(format!("data: {content_stop}\n\n"))),
            Ok(bytes::Bytes::from(format!("data: {message_stop}\n\n"))),
        ];

        let byte_stream = stream::iter(byte_chunks);
        let sse_stream = crate::sse_stream::SseStream::new(byte_stream);

        // Process events through our Anthropic SSE handler
        let mut events = Vec::new();
        let mut state = StreamState::default();

        // Collect all events using StreamExt::next
        let mut sse_stream = sse_stream;
        while let Some(sse_result) = sse_stream.next().await {
            let sse_event = sse_result.expect("SSE should parse correctly");
            let data = sse_event.data.trim();

            if data.is_empty() {
                continue;
            }

            // Parse as AnthropicStreamEvent
            let stream_event: AnthropicStreamEvent =
                serde_json::from_str(data).expect("JSON should parse");
            let stream_events = convert_stream_event_stateful(stream_event, &mut state)
                .expect("conversion should succeed");
            events.extend(stream_events);
        }

        // Verify we got the expected events
        let content_events: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentDelta { delta } => Some(delta.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(content_events, vec!["Hello", " world"]);

        // Verify we got exactly one Done event at the end
        let done_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::Done { .. }))
            .collect();

        assert_eq!(done_events.len(), 1);

        // The Done event should be the last event
        assert!(matches!(events.last(), Some(StreamEvent::Done { .. })));
    }
}
