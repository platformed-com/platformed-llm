use futures_util::StreamExt;
use std::collections::HashMap;

use super::anthropic_types::*;
use super::endpoint::VertexEndpoint;
use crate::provider::LLMProvider;
use crate::sse_stream::SseStream;
use crate::transport::{Transport, TransportRequest};
use crate::types::{
    AssistantPart, FinishReason, FunctionCall, InputItem, PartKind, PartUpdate, ReasoningEffort,
    UserPart, Usage,
};
use crate::{Error, LLMRequest, Response, StreamEvent};

/// Anthropic Claude provider implementation via Vertex AI.
pub struct AnthropicViaVertexProvider {
    endpoint: VertexEndpoint,
    transport: Transport,
}

impl AnthropicViaVertexProvider {
    /// Create a new Anthropic provider with access token authentication.
    pub fn new(project_id: String, location: String, access_token: String) -> Result<Self, Error> {
        Ok(Self {
            endpoint: VertexEndpoint::with_access_token(project_id, location, access_token),
            transport: Transport::reqwest()?,
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
            endpoint: VertexEndpoint::with_access_token(project_id, location, access_token)
                .with_base_url(base_url),
            transport: Transport::reqwest()?,
        })
    }

    /// Create a new Anthropic provider with Application Default Credentials.
    pub async fn with_adc(project_id: String, location: String) -> Result<Self, Error> {
        Ok(Self {
            endpoint: VertexEndpoint::with_adc(project_id, location).await?,
            transport: Transport::reqwest()?,
        })
    }

    /// Create a new Anthropic provider with a caller-supplied [`Transport`]
    /// and pre-built [`VertexEndpoint`].
    pub fn with_transport(endpoint: VertexEndpoint, transport: Transport) -> Self {
        Self { endpoint, transport }
    }

    /// Convert internal request to Anthropic format.
    fn convert_request(&self, request: &LLMRequest) -> Result<AnthropicRequest, Error> {
        let mut messages = Vec::new();
        let mut system_message = None;

        for item in &request.messages {
            match item {
                InputItem::System(content) => {
                    system_message = Some(content.clone());
                }
                InputItem::User { content } => {
                    let blocks = build_user_blocks(content)?;
                    if blocks.is_empty() {
                        continue;
                    }
                    if blocks.len() == 1 {
                        if let AnthropicContentBlock::Text { text } = &blocks[0] {
                            messages.push(AnthropicMessage {
                                role: "user".to_string(),
                                content: AnthropicContent::Text(text.clone()),
                            });
                            continue;
                        }
                    }
                    messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContent::Blocks(blocks),
                    });
                }
                InputItem::Assistant { content } => {
                    let blocks = build_assistant_blocks(content)?;
                    if blocks.is_empty() {
                        continue;
                    }
                    if blocks.len() == 1 {
                        if let AnthropicContentBlock::Text { text } = &blocks[0] {
                            messages.push(AnthropicMessage {
                                role: "assistant".to_string(),
                                content: AnthropicContent::Text(text.clone()),
                            });
                            continue;
                        }
                    }
                    messages.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: AnthropicContent::Blocks(blocks),
                    });
                }
            }
        }

        let tools = request.tools.as_ref().and_then(|tools| {
            use crate::types::{ProviderBuiltin, Tool};
            let converted: Vec<AnthropicTool> = tools
                .iter()
                .filter_map(|tool| match tool {
                    Tool::Function(f) => Some(AnthropicTool {
                        name: f.name.clone(),
                        description: f.description.clone().unwrap_or_default(),
                        input_schema: f.parameters.clone(),
                    }),
                    Tool::Builtin(b) => {
                        // WebSearch / ComputerUse are Anthropic builtins
                        // but use distinct wire shapes (separate tools array
                        // entries with type=web_search_20250305 etc.).
                        // Wiring is a Phase 5 follow-up; for now drop with
                        // a tracing note so model-switching doesn't break.
                        if !matches!(
                            b,
                            ProviderBuiltin::WebSearch | ProviderBuiltin::ComputerUse
                        ) {
                            tracing::debug!(
                                ?b,
                                "Anthropic provider dropping unsupported builtin tool"
                            );
                        }
                        None
                    }
                })
                .collect();
            if converted.is_empty() {
                None
            } else {
                Some(converted)
            }
        });

        // Map our unified ReasoningConfig onto Anthropic's `thinking` field.
        // We derive budget_tokens from `effort` with sensible defaults;
        // callers needing precise control can construct providers directly.
        let thinking = request.reasoning.as_ref().map(|cfg| {
            let budget_tokens = match cfg.effort.unwrap_or(ReasoningEffort::Medium) {
                ReasoningEffort::Low => 2048,
                ReasoningEffort::Medium => 8192,
                ReasoningEffort::High => 16384,
            };
            AnthropicThinking::Enabled { budget_tokens }
        });

        // Anthropic requires temperature == 1 when thinking is enabled.
        // Override with a warning rather than erroring; better DX.
        let temperature = if thinking.is_some() {
            if matches!(request.temperature, Some(t) if (t - 1.0).abs() > f32::EPSILON) {
                tracing::warn!(
                    requested = ?request.temperature,
                    "Anthropic requires temperature=1 when extended thinking is enabled; \
                     overriding"
                );
            }
            Some(1.0)
        } else {
            request.temperature
        };

        let anthropic_request = AnthropicRequest {
            messages,
            max_tokens: request.max_tokens.unwrap_or(1024),
            anthropic_version: "vertex-2023-10-16".to_string(),
            system: system_message,
            temperature,
            top_p: request.top_p,
            tools,
            stream: Some(true), // Enable streaming for SSE responses
            thinking,
        };

        Ok(anthropic_request)
    }
}

/// Translate user-side parts into Anthropic content blocks. Text parts
/// coalesce; tool results map to `tool_result` blocks; multi-modal
/// parts are not yet wired (Phase 5 follow-up).
fn build_user_blocks(parts: &[UserPart]) -> Result<Vec<AnthropicContentBlock>, Error> {
    let mut blocks = Vec::new();
    for part in parts {
        match part {
            UserPart::Text(s) => blocks.push(AnthropicContentBlock::Text { text: s.clone() }),
            UserPart::ToolResult { call_id, content } => {
                let text = flatten_user_parts_to_text(content);
                blocks.push(AnthropicContentBlock::ToolResult {
                    tool_use_id: call_id.clone(),
                    content: AnthropicToolResultContent::Text(text),
                    is_error: None,
                });
            }
            UserPart::Image(_) | UserPart::Audio(_) | UserPart::Document(_) => {
                tracing::debug!("Anthropic provider dropping unsupported user part");
            }
            UserPart::CacheBreakpoint => {
                // Phase 5 follow-up: emit `cache_control: ephemeral` on
                // the preceding block.
                tracing::debug!("Anthropic provider has no cache_control wiring yet");
            }
        }
    }
    Ok(blocks)
}

/// Translate assistant-side parts into Anthropic content blocks. Text +
/// reasoning + tool_use are all expressed as blocks; reasoning carries
/// its signature when present.
fn build_assistant_blocks(parts: &[AssistantPart]) -> Result<Vec<AnthropicContentBlock>, Error> {
    let mut blocks = Vec::new();
    for part in parts {
        match part {
            AssistantPart::Text { content, .. } => {
                blocks.push(AnthropicContentBlock::Text {
                    text: content.clone(),
                });
            }
            AssistantPart::Reasoning { content, signature } => {
                blocks.push(AnthropicContentBlock::Thinking {
                    thinking: content.clone(),
                    signature: signature.clone(),
                });
            }
            AssistantPart::RedactedReasoning { data } => {
                blocks.push(AnthropicContentBlock::RedactedThinking { data: data.clone() });
            }
            AssistantPart::Refusal(s) => {
                // Anthropic has no typed refusal channel; surface as text.
                blocks.push(AnthropicContentBlock::Text { text: s.clone() });
            }
            AssistantPart::ToolCall(call) => {
                let input = serde_json::from_str(&call.arguments).map_err(|e| {
                    Error::provider("Anthropic", format!("Invalid function arguments: {e}"))
                })?;
                blocks.push(AnthropicContentBlock::ToolUse {
                    id: call.call_id.clone(),
                    name: call.name.clone(),
                    input,
                });
            }
            AssistantPart::CacheBreakpoint => {
                tracing::debug!("Anthropic provider has no cache_control wiring yet");
            }
        }
    }
    Ok(blocks)
}

fn flatten_user_parts_to_text(parts: &[UserPart]) -> String {
    let mut out = String::new();
    for part in parts {
        if let UserPart::Text(s) = part {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(s);
        }
    }
    out
}

#[async_trait::async_trait]
impl LLMProvider for AnthropicViaVertexProvider {
    async fn generate(&self, request: &LLMRequest) -> Result<Response, Error> {
        let anthropic_request = self.convert_request(request)?;

        let url = self.endpoint.url(
            "anthropic",
            &request.model,
            "streamRawPredict",
            Some("alt=sse"),
        );

        let body = serde_json::to_vec(&anthropic_request)?;
        let req = TransportRequest {
            url,
            headers: vec![
                self.endpoint.auth_header().await?,
                ("Content-Type".to_string(), "application/json".to_string()),
            ],
            body,
        };
        let response = self.transport.send(req).await?;

        if !(200..300).contains(&response.status) {
            let status = response.status;
            let body_bytes = response.collect_body().await.unwrap_or_default();
            let body_text = String::from_utf8_lossy(&body_bytes);
            return Err(match status {
                401 | 403 => {
                    Error::auth_with_status(status, format!("Anthropic {status}: {body_text}"))
                }
                404 => Error::ModelNotAvailable(format!("Anthropic 404: {body_text}")),
                _ => Error::provider_with_status(
                    "Anthropic",
                    status,
                    format!("API error: {body_text}"),
                ),
            });
        }

        // Create SSE stream from response
        let sse_stream = SseStream::new(response.body);

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

                        // Anthropic's wire format only emits JSON event
                        // payloads (including `{"type":"ping"}` for keep-
                        // alives). The SSE parser already filters comment
                        // lines, so anything that fails to parse here is a
                        // genuine surprise — surface it.
                        match serde_json::from_str::<AnthropicStreamEvent>(data) {
                            Ok(stream_event) => {
                                match convert_stream_event_stateful(stream_event, &mut state) {
                                    Ok(events) => events.into_iter().map(Ok).collect(),
                                    Err(e) => vec![Err(e)],
                                }
                            }
                            Err(e) => vec![Err(Error::provider(
                                "Anthropic",
                                format!("Failed to parse SSE event: {e}"),
                            ))],
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

/// State for tracking streaming progress.
///
/// Anthropic delivers `stop_reason` and the cumulative `usage` on
/// `message_delta` events which fire **before** `message_stop`. We
/// stash them so the final `Done` event reflects what the model
/// actually said.
#[derive(Debug, Default)]
pub(crate) struct StreamState {
    /// Maps Anthropic's content-block index to our lib-side part index.
    tracker: crate::providers::part_tracker::PartTracker<u32>,
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
    if let Some(t) = src.cache_read_input_tokens {
        target.cache_read_input_tokens = Some(t);
    }
    if let Some(t) = src.cache_creation_input_tokens {
        target.cache_creation_input_tokens = Some(t);
    }
}

/// Convert an Anthropic stream event into our unified `StreamEvent`s.
///
/// `pub(crate)` so unit tests can drive this directly with synthetic events.
pub(crate) fn convert_stream_event_stateful(
    event: AnthropicStreamEvent,
    state: &mut StreamState,
) -> Result<Vec<StreamEvent>, Error> {
    let mut events = Vec::new();

    match event {
        AnthropicStreamEvent::MessageStart { message } => {
            if let Some(usage) = &message.usage {
                merge_anthropic_usage(&mut state.pending_usage, usage);
            }
        }
        AnthropicStreamEvent::ContentBlockStart {
            content_block,
            index,
        } => match content_block {
            AnthropicContentBlock::Text { text } => {
                let (lib_idx, ev) = state.tracker.open(index, PartKind::Text);
                events.push(ev);
                if !text.is_empty() {
                    events.push(StreamEvent::Delta {
                        index: lib_idx,
                        delta: text,
                    });
                }
            }
            AnthropicContentBlock::ToolUse { id, name, input } => {
                let (_lib_idx, ev) = state.tracker.open(
                    index,
                    PartKind::ToolCall {
                        call_id: id,
                        name,
                    },
                );
                events.push(ev);
                // Per the streaming protocol the initial `input` is `{}`.
                // Arguments arrive via input_json_delta.
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
            }
            AnthropicContentBlock::Thinking { thinking, signature } => {
                let (lib_idx, ev) = state.tracker.open(index, PartKind::Reasoning);
                events.push(ev);
                if !thinking.is_empty() {
                    events.push(StreamEvent::Delta {
                        index: lib_idx,
                        delta: thinking,
                    });
                }
                if let Some(sig) = signature {
                    events.push(StreamEvent::PartUpdate {
                        index: lib_idx,
                        update: PartUpdate::Signature(sig),
                    });
                }
            }
            AnthropicContentBlock::RedactedThinking { data } => {
                let (_lib_idx, ev) = state
                    .tracker
                    .open(index, PartKind::RedactedReasoning { data });
                events.push(ev);
            }
            AnthropicContentBlock::ToolResult { .. } => {
                // Request-side blocks; not expected on the response stream.
            }
            AnthropicContentBlock::Image { .. } => {
                // Request-side blocks; not expected on the response stream.
            }
        },
        AnthropicStreamEvent::ContentBlockDelta { delta, index } => {
            let lib_idx = match state.tracker.index_of(&index) {
                Some(i) => i,
                None => {
                    return Err(Error::streaming(format!(
                        "Anthropic content_block_delta for unknown index {index}"
                    )));
                }
            };
            match delta {
                AnthropicContentDelta::TextDelta { text } => {
                    if !text.is_empty() {
                        events.push(StreamEvent::Delta {
                            index: lib_idx,
                            delta: text,
                        });
                    }
                }
                AnthropicContentDelta::InputJsonDelta { partial_json } => {
                    events.push(StreamEvent::Delta {
                        index: lib_idx,
                        delta: partial_json,
                    });
                }
                AnthropicContentDelta::ThinkingDelta { thinking } => {
                    if !thinking.is_empty() {
                        events.push(StreamEvent::Delta {
                            index: lib_idx,
                            delta: thinking,
                        });
                    }
                }
                AnthropicContentDelta::SignatureDelta { signature } => {
                    events.push(StreamEvent::PartUpdate {
                        index: lib_idx,
                        update: PartUpdate::Signature(signature),
                    });
                }
            }
        }
        AnthropicStreamEvent::ContentBlockStop { index } => {
            if let Some(ev) = state.tracker.close(&index) {
                events.push(ev);
            }
        }
        AnthropicStreamEvent::MessageDelta { delta, usage } => {
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
        AnthropicStreamEvent::Error { error } => {
            return Err(Error::provider(
                "Anthropic",
                format!("{}: {}", error.error_type, error.message),
            ));
        }
    }

    Ok(events)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LLMRequest, Prompt};

    fn provider() -> AnthropicViaVertexProvider {
        AnthropicViaVertexProvider::new(
            "p".to_string(),
            "us-east5".to_string(),
            "tok".to_string(),
        )
        .unwrap()
    }

    #[test]
    fn map_anthropic_stop_reason_known_values() {
        assert_eq!(map_anthropic_stop_reason(Some("end_turn")), FinishReason::Stop);
        assert_eq!(map_anthropic_stop_reason(Some("tool_use")), FinishReason::ToolCalls);
        assert_eq!(map_anthropic_stop_reason(Some("max_tokens")), FinishReason::Length);
        assert_eq!(map_anthropic_stop_reason(Some("refusal")), FinishReason::ContentFilter);
        assert_eq!(map_anthropic_stop_reason(None), FinishReason::Stop);
    }

    #[test]
    fn convert_simple_text_request() {
        let req = LLMRequest::from_prompt("claude", &Prompt::user("hi"));
        let body = provider().convert_request(&req).unwrap();
        assert_eq!(body.messages.len(), 1);
        assert_eq!(body.messages[0].role, "user");
    }

    /// A signature_delta on a thinking block emits PartUpdate::Signature
    /// pointing at the correct part index.
    #[test]
    fn signature_delta_emits_part_update() {
        let mut state = StreamState::default();
        let start = AnthropicStreamEvent::ContentBlockStart {
            index: 0,
            content_block: AnthropicContentBlock::Thinking {
                thinking: String::new(),
                signature: None,
            },
        };
        let _ = convert_stream_event_stateful(start, &mut state).unwrap();
        let sig = AnthropicStreamEvent::ContentBlockDelta {
            index: 0,
            delta: AnthropicContentDelta::SignatureDelta {
                signature: "sig_abc".to_string(),
            },
        };
        let events = convert_stream_event_stateful(sig, &mut state).unwrap();
        match &events[0] {
            StreamEvent::PartUpdate {
                index: 0,
                update: PartUpdate::Signature(s),
            } => assert_eq!(s, "sig_abc"),
            other => panic!("expected PartUpdate(Signature), got {other:?}"),
        }
    }

    /// A `tool_use` content block opens a `PartKind::ToolCall` part with
    /// the wire `id` carried as our `call_id`.
    #[test]
    fn tool_use_opens_tool_call_part() {
        let mut state = StreamState::default();
        let start = AnthropicStreamEvent::ContentBlockStart {
            index: 0,
            content_block: AnthropicContentBlock::ToolUse {
                id: "toolu_xyz".to_string(),
                name: "get_weather".to_string(),
                input: ijson::ijson!({}),
            },
        };
        let events = convert_stream_event_stateful(start, &mut state).unwrap();
        match &events[0] {
            StreamEvent::PartStart {
                index: 0,
                kind: PartKind::ToolCall { call_id, name },
            } => {
                assert_eq!(call_id, "toolu_xyz");
                assert_eq!(name, "get_weather");
            }
            other => panic!("expected PartStart(ToolCall), got {other:?}"),
        }
    }
}
