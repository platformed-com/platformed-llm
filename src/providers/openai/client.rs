use super::types::{OpenAIReasoning, OpenAIToolChoice, ResponsesRequest, ResponsesStreamEvent};
use crate::provider::LLMProvider;
use crate::transport::{Transport, TransportRequest};
use crate::types::{PartKind, ReasoningConfig, ReasoningEffort, ReasoningSummary, ToolChoice};
use crate::{Error, LLMRequest, Response, StreamEvent};
use futures_util::StreamExt as _;
use std::sync::{Arc, Mutex};
use tracing::debug;

/// OpenAI provider implementation.
pub struct OpenAIProvider {
    transport: Transport,
    api_key: String,
    base_url: String,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider with the default reqwest-backed transport.
    pub fn new(api_key: String) -> Result<Self, Error> {
        Ok(Self {
            transport: Transport::reqwest()?,
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
        })
    }

    /// Create a new OpenAI provider with a custom base URL and the default transport.
    pub fn new_with_base_url(api_key: String, base_url: String) -> Result<Self, Error> {
        Ok(Self {
            transport: Transport::reqwest()?,
            api_key,
            base_url,
        })
    }

    /// Create a new OpenAI provider with a caller-supplied transport. Lets
    /// downstream consumers (or tests) plug in a recording / replaying /
    /// retrying [`Transport`] without touching the rest of the provider.
    pub fn with_transport(api_key: String, base_url: String, transport: Transport) -> Self {
        Self {
            transport,
            api_key,
            base_url,
        }
    }

    /// Convert internal request to OpenAI Responses API format.
    fn convert_request(&self, request: &LLMRequest) -> ResponsesRequest {
        let mut input: Vec<crate::providers::openai::types::OpenAIInputMessage> = Vec::new();
        for item in &request.messages {
            Self::flatten_input_item(item, &mut input);
        }

        // ProviderContinuation::OpenAI carries previous_response_id when
        // the caller wants to chain via server-side state. Other variants
        // (none exist yet) are silently ignored — that's the
        // model-switching contract.
        let previous_response_id = match &request.continuation {
            Some(crate::types::ProviderContinuation::OpenAI { response_id }) => {
                Some(response_id.clone())
            }
            _ => None,
        };

        ResponsesRequest {
            model: request.model.clone(),
            input,
            instructions: None,
            temperature: request.temperature,
            max_output_tokens: request.max_tokens,
            top_p: request.top_p,
            tools: request
                .tools
                .as_ref()
                .map(|tools| Self::convert_tools(tools)),
            tool_choice: request.tool_choice.as_ref().map(convert_tool_choice),
            parallel_tool_calls: request.parallel_tool_calls,
            previous_response_id,
            stream: None,
            store: Some(request.store.unwrap_or(false)),
            reasoning: request.reasoning.as_ref().map(convert_reasoning),
            stop: request.stop.clone(),
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
        }
    }

    /// Flatten one canonical `InputItem` into one or more OpenAI input
    /// items. OpenAI's wire model puts function_call / function_call_output
    /// as siblings of message items, so an Assistant turn with mixed
    /// text + tool calls splits into a message + N function_call items
    /// here (preserving emit order).
    fn flatten_input_item(
        item: &crate::types::InputItem,
        out: &mut Vec<crate::providers::openai::types::OpenAIInputMessage>,
    ) {
        use crate::providers::openai::types::OpenAIInputMessage;
        use crate::types::{AssistantPart, InputItem, UserPart};

        match item {
            InputItem::System(content) => {
                out.push(OpenAIInputMessage::Regular {
                    role: "system".to_string(),
                    content: crate::providers::openai::types::OpenAIMessageContent::Text(
                        content.clone(),
                    ),
                });
            }
            InputItem::User { content } => {
                use crate::providers::openai::types::{OpenAIContentPart, OpenAIMessageContent};
                // Build a content-parts list. Tool results become their own
                // top-level items; text and images become InputText /
                // InputImage parts. If we end up with just one text part
                // we collapse to a bare string for the common case.
                let mut parts: Vec<OpenAIContentPart> = Vec::new();
                for part in content {
                    match part {
                        UserPart::Text(s) => parts.push(OpenAIContentPart::InputText {
                            text: s.clone(),
                        }),
                        UserPart::Image(src) => {
                            let url = match src {
                                crate::types::ImageSource::Url(u) => u.clone(),
                                crate::types::ImageSource::Base64 { data, media_type } => {
                                    format!("data:{media_type};base64,{data}")
                                }
                            };
                            parts.push(OpenAIContentPart::InputImage {
                                image_url: Some(url),
                            });
                        }
                        UserPart::ToolResult { call_id, content } => {
                            push_user_parts(out, &mut parts);
                            out.push(OpenAIInputMessage::FunctionCallOutput {
                                call_id: call_id.clone(),
                                output: flatten_user_parts_to_text(content),
                            });
                        }
                        UserPart::Audio(_) | UserPart::Document(_) => {
                            tracing::debug!(
                                "OpenAI provider dropping audio/document user part during request build"
                            );
                        }
                        UserPart::CacheBreakpoint => {
                            // OpenAI maps cache breakpoints onto a per-
                            // request `prompt_cache_key` rather than
                            // per-block markers; nothing to emit at the
                            // part level.
                        }
                    }
                }
                push_user_parts(out, &mut parts);
            }
            InputItem::Assistant { content } => {
                let mut buffered_text = String::new();
                for part in content {
                    match part {
                        AssistantPart::Text { content, .. } => {
                            if !buffered_text.is_empty() {
                                buffered_text.push('\n');
                            }
                            buffered_text.push_str(content);
                        }
                        AssistantPart::Refusal(s) => {
                            if !buffered_text.is_empty() {
                                buffered_text.push('\n');
                            }
                            buffered_text.push_str(s);
                        }
                        AssistantPart::ToolCall(call) => {
                            if !buffered_text.is_empty() {
                                out.push(OpenAIInputMessage::Regular {
                                    role: "assistant".to_string(),
                                    content: crate::providers::openai::types::OpenAIMessageContent::Text(
                                        std::mem::take(&mut buffered_text),
                                    ),
                                });
                            }
                            out.push(OpenAIInputMessage::FunctionCall {
                                call_id: call.call_id.clone(),
                                name: call.name.clone(),
                                arguments: call.arguments.clone(),
                            });
                        }
                        AssistantPart::Reasoning { .. }
                        | AssistantPart::RedactedReasoning { .. }
                        | AssistantPart::CacheBreakpoint => {
                            tracing::debug!(
                                "OpenAI provider dropping unsupported assistant part during request build"
                            );
                        }
                    }
                }
                if !buffered_text.is_empty() {
                    out.push(OpenAIInputMessage::Regular {
                        role: "assistant".to_string(),
                        content: crate::providers::openai::types::OpenAIMessageContent::Text(
                            buffered_text,
                        ),
                    });
                }
            }
        }
    }

    /// Convert our internal tools to OpenAI Responses API format.
    /// Builtin tools that OpenAI offers (`web_search`, `computer_use`)
    /// emit their typed wire shape; builtins OpenAI doesn't offer are
    /// silently dropped — model-switching contract.
    #[allow(clippy::ptr_arg)]
    fn convert_tools(tools: &[crate::types::Tool]) -> Vec<super::types::OpenAITool> {
        use crate::types::{ProviderBuiltin, Tool};
        let mut out = Vec::new();
        for tool in tools {
            match tool {
                Tool::Function(f) => out.push(super::types::OpenAITool::Function {
                    name: f.name.clone(),
                    description: f.description.clone().unwrap_or_default(),
                    parameters: f.parameters.clone(),
                }),
                Tool::Builtin(b) => match b {
                    ProviderBuiltin::WebSearch => {
                        out.push(super::types::OpenAITool::WebSearchPreview);
                    }
                    ProviderBuiltin::ComputerUse => {
                        out.push(super::types::OpenAITool::ComputerUsePreview);
                    }
                    ProviderBuiltin::GoogleSearch | ProviderBuiltin::CodeExecution => {
                        tracing::debug!(
                            ?b,
                            "OpenAI provider dropping unsupported builtin tool"
                        );
                    }
                },
            }
        }
        out
    }
}

/// Map an OpenAI HTTP error response onto our [`Error`] variants.
///
/// OpenAI returns `{"error":{"message":..., "type":..., "code":...}}` on
/// failure. We parse it best-effort and pick a typed variant from the
/// HTTP status:
///
/// - 401 → [`Error::Auth`]
/// - 429 → [`Error::RateLimit`] (carries `Retry-After` if present)
/// - any other → [`Error::Provider`] with status, type, and message
///
/// The full body is preserved in the message so callers can still extract
/// the unparsed structured fields if they need them.
pub(crate) fn parse_openai_error(
    status: u16,
    retry_after_seconds: Option<u64>,
    body: &str,
) -> Error {
    #[derive(serde::Deserialize)]
    struct Outer<'a> {
        #[serde(borrow)]
        error: Option<Inner<'a>>,
    }
    #[derive(serde::Deserialize)]
    struct Inner<'a> {
        #[serde(default, borrow)]
        message: Option<&'a str>,
        #[serde(default, rename = "type", borrow)]
        kind: Option<&'a str>,
        #[serde(default, borrow)]
        code: Option<&'a str>,
    }
    let parsed = serde_json::from_str::<Outer>(body)
        .ok()
        .and_then(|o| o.error);
    let message = parsed
        .as_ref()
        .and_then(|e| e.message)
        .unwrap_or(body)
        .to_string();
    let kind = parsed.as_ref().and_then(|e| e.kind).unwrap_or("");
    let code = parsed.as_ref().and_then(|e| e.code).unwrap_or("");

    match status {
        401 => Error::auth_with_status(401, format!("OpenAI 401 ({kind} {code}): {message}")),
        429 => Error::rate_limit(
            retry_after_seconds,
            format!("OpenAI 429 ({kind} {code}): {message}"),
        ),
        _ => Error::provider_with_status(
            "OpenAI",
            status,
            format!("HTTP {status} ({kind} {code}): {message}"),
        ),
    }
}

/// Flush a buffered user-content-parts list as an
/// `OpenAIInputMessage::Regular`. Collapses single-text to the bare-
/// string content form (OpenAI accepts both, but the bare string is
/// the canonical shape and matches our pre-multi-modal request bodies).
fn push_user_parts(
    out: &mut Vec<crate::providers::openai::types::OpenAIInputMessage>,
    parts: &mut Vec<crate::providers::openai::types::OpenAIContentPart>,
) {
    use crate::providers::openai::types::{
        OpenAIContentPart, OpenAIInputMessage, OpenAIMessageContent,
    };
    if parts.is_empty() {
        return;
    }
    let drained: Vec<OpenAIContentPart> = std::mem::take(parts);
    // Single text part → bare string; anything else → parts array.
    if drained.len() == 1 {
        if let OpenAIContentPart::InputText { text } = &drained[0] {
            out.push(OpenAIInputMessage::Regular {
                role: "user".to_string(),
                content: OpenAIMessageContent::Text(text.clone()),
            });
            return;
        }
    }
    out.push(OpenAIInputMessage::Regular {
        role: "user".to_string(),
        content: OpenAIMessageContent::Parts(drained),
    });
}

/// Best-effort flatten of a tool-result content array into a single string.
/// Multi-modal tool results aren't currently representable on OpenAI's
/// function_call_output (which takes a plain string `output`), so non-text
/// parts are dropped with a tracing note.
fn flatten_user_parts_to_text(parts: &[crate::types::UserPart]) -> String {
    use crate::types::UserPart;
    let mut out = String::new();
    for part in parts {
        match part {
            UserPart::Text(s) => {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(s);
            }
            _ => {
                tracing::debug!("dropping non-text tool result part for OpenAI flatten");
            }
        }
    }
    out
}

fn convert_reasoning(cfg: &ReasoningConfig) -> OpenAIReasoning {
    OpenAIReasoning {
        effort: cfg.effort.map(|e| match e {
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
        }),
        summary: cfg.summary.map(|s| match s {
            ReasoningSummary::Auto => "auto",
            ReasoningSummary::Concise => "concise",
            ReasoningSummary::Detailed => "detailed",
        }),
    }
}

fn convert_tool_choice(choice: &ToolChoice) -> OpenAIToolChoice {
    match choice {
        ToolChoice::Auto => OpenAIToolChoice::Mode("auto"),
        ToolChoice::None => OpenAIToolChoice::Mode("none"),
        ToolChoice::Required => OpenAIToolChoice::Mode("required"),
        ToolChoice::Function { name } => OpenAIToolChoice::Function {
            kind: "function",
            name: name.clone(),
        },
    }
}

/// Streaming state for an in-flight OpenAI response.
///
/// OpenAI's wire model has two-level nesting: top-level items
/// (`message`, `function_call`, `reasoning`) plus per-message content
/// parts (`output_text`, `refusal`). We map both to our flat part-index
/// space via `(output_index, content_index)` keys — `content_index` is
/// `None` for top-level items that don't have nested content parts.
pub(crate) struct OpenAIStreamState {
    tracker: crate::providers::part_tracker::PartTracker<(u32, Option<u32>)>,
    /// Recorded continuation hint from the most recent response payload.
    continuation: Option<crate::types::ProviderContinuation>,
}

impl OpenAIStreamState {
    pub(crate) fn new() -> Self {
        Self {
            tracker: crate::providers::part_tracker::PartTracker::new(),
            continuation: None,
        }
    }

    pub(crate) fn continuation(&self) -> Option<crate::types::ProviderContinuation> {
        self.continuation.clone()
    }

    /// Process one OpenAI wire event into 0 or more `StreamEvent`s.
    pub(crate) fn process(
        &mut self,
        event: ResponsesStreamEvent,
    ) -> Result<Vec<StreamEvent>, Error> {
        match event.r#type.as_str() {
            "error" => {
                let (kind, message) = match &event.error {
                    Some(e) => (e.r#type.as_str(), e.message.as_str()),
                    None => ("unknown", "Unknown error occurred"),
                };
                return Err(Error::provider("OpenAI", format!("{kind}: {message}")));
            }
            "response.created" | "response.in_progress" => {
                if let Some(response) = &event.response {
                    self.continuation = Some(crate::types::ProviderContinuation::OpenAI {
                        response_id: response.id.clone(),
                    });
                }
            }
            "response.output_item.added" => {
                let Some(item) = event.item else { return Ok(vec![]); };
                let Some(output_index) = event.output_index else { return Ok(vec![]); };
                match item.r#type.as_str() {
                    "function_call" => {
                        let call_id = item.call_id.ok_or_else(|| {
                            Error::provider(
                                "OpenAI",
                                format!(
                                    "function_call item is missing call_id (item id: {})",
                                    item.id
                                ),
                            )
                        })?;
                        let name = item.name.unwrap_or_else(|| "unknown".to_string());
                        let (_idx, ev) = self.tracker.open(
                            (output_index, None),
                            PartKind::ToolCall { call_id, name },
                        );
                        return Ok(vec![ev]);
                    }
                    "reasoning" => {
                        let (_idx, ev) = self
                            .tracker
                            .open((output_index, None), PartKind::Reasoning);
                        return Ok(vec![ev]);
                    }
                    "message" => {
                        // Wait for response.content_part.added — message
                        // items contain an array of content parts, each of
                        // which maps to its own AssistantPart.
                    }
                    _ => {}
                }
            }
            "response.content_part.added" => {
                let Some(output_index) = event.output_index else { return Ok(vec![]); };
                let Some(content_index) = event.content_index else { return Ok(vec![]); };
                let kind = match event.part.as_ref().map(|p| p.r#type.as_str()) {
                    Some("output_text") => PartKind::Text,
                    Some("refusal") => PartKind::Refusal,
                    _ => PartKind::Text,
                };
                let (_idx, ev) = self
                    .tracker
                    .open((output_index, Some(content_index)), kind);
                return Ok(vec![ev]);
            }
            "response.output_text.delta" => {
                let Some(output_index) = event.output_index else { return Ok(vec![]); };
                let Some(content_index) = event.content_index else { return Ok(vec![]); };
                let Some(delta) = event.delta else { return Ok(vec![]); };
                if delta.is_empty() {
                    return Ok(vec![]);
                }
                let key = (output_index, Some(content_index));
                let index = self
                    .tracker
                    .index_of(&key)
                    .ok_or_else(|| Error::streaming(format!(
                        "output_text.delta for unknown content part {key:?}"
                    )))?;
                return Ok(vec![StreamEvent::Delta { index, delta }]);
            }
            "response.refusal.delta" => {
                let Some(output_index) = event.output_index else { return Ok(vec![]); };
                let Some(content_index) = event.content_index else { return Ok(vec![]); };
                let Some(delta) = event.delta else { return Ok(vec![]); };
                if delta.is_empty() {
                    return Ok(vec![]);
                }
                let key = (output_index, Some(content_index));
                let index = self
                    .tracker
                    .index_of(&key)
                    .ok_or_else(|| Error::streaming(format!(
                        "refusal.delta for unknown content part {key:?}"
                    )))?;
                return Ok(vec![StreamEvent::Delta { index, delta }]);
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let Some(output_index) = event.output_index else { return Ok(vec![]); };
                let Some(delta) = event.delta else { return Ok(vec![]); };
                if delta.is_empty() {
                    return Ok(vec![]);
                }
                let key = (output_index, None);
                let index = self
                    .tracker
                    .index_of(&key)
                    .ok_or_else(|| Error::streaming(format!(
                        "reasoning delta for unknown reasoning part {key:?}"
                    )))?;
                return Ok(vec![StreamEvent::Delta { index, delta }]);
            }
            "response.function_call_arguments.delta" => {
                let Some(output_index) = event.output_index else { return Ok(vec![]); };
                let Some(delta) = event.delta else { return Ok(vec![]); };
                if delta.is_empty() {
                    return Ok(vec![]);
                }
                let key = (output_index, None);
                let index = self
                    .tracker
                    .index_of(&key)
                    .ok_or_else(|| Error::streaming(format!(
                        "function_call_arguments.delta for unknown tool part {key:?}"
                    )))?;
                return Ok(vec![StreamEvent::Delta { index, delta }]);
            }
            "response.content_part.done" => {
                let Some(output_index) = event.output_index else { return Ok(vec![]); };
                let Some(content_index) = event.content_index else { return Ok(vec![]); };
                if let Some(ev) = self.tracker.close(&(output_index, Some(content_index))) {
                    return Ok(vec![ev]);
                }
            }
            "response.output_item.done" => {
                let Some(output_index) = event.output_index else { return Ok(vec![]); };
                if let Some(ev) = self.tracker.close(&(output_index, None)) {
                    return Ok(vec![ev]);
                }
            }
            "response.output_text.done" | "response.reasoning_summary_text.done"
            | "response.reasoning_text.done" | "response.refusal.done"
            | "response.function_call_arguments.done" => {
                // Final canonical value — we already received it via deltas.
            }
            "response.completed" => {
                if let Some(response) = event.response {
                    self.continuation = Some(crate::types::ProviderContinuation::OpenAI {
                        response_id: response.id.clone(),
                    });
                    let finish_reason =
                        if response.output.iter().any(|o| o.r#type == "function_call") {
                            crate::types::FinishReason::ToolCalls
                        } else {
                            crate::types::FinishReason::Stop
                        };
                    return Ok(vec![StreamEvent::Done {
                        finish_reason,
                        usage: response.usage.unwrap_or_default(),
                    }]);
                }
            }
            "response.incomplete" => {
                if let Some(response) = event.response {
                    self.continuation = Some(crate::types::ProviderContinuation::OpenAI {
                        response_id: response.id.clone(),
                    });
                    let finish_reason = match response
                        .incomplete_details
                        .as_ref()
                        .map(|d| d.reason.as_str())
                    {
                        Some("max_output_tokens") => crate::types::FinishReason::Length,
                        Some("content_filter") => crate::types::FinishReason::ContentFilter,
                        _ => crate::types::FinishReason::Stop,
                    };
                    return Ok(vec![StreamEvent::Done {
                        finish_reason,
                        usage: response.usage.unwrap_or_default(),
                    }]);
                }
            }
            "response.failed" => {
                if let Some(response) = event.response {
                    let message = response
                        .error
                        .as_ref()
                        .map(|e| format!("{}: {}", e.r#type, e.message))
                        .unwrap_or_else(|| "response failed without error details".to_string());
                    return Err(Error::provider("OpenAI", format!("response.failed — {message}")));
                }
                if let Some(error) = &event.error {
                    return Err(Error::provider(
                        "OpenAI",
                        format!("response.failed — {}: {}", error.r#type, error.message),
                    ));
                }
                return Err(Error::provider("OpenAI", "response.failed without details"));
            }
            _ => {}
        }
        Ok(vec![])
    }
}

#[async_trait::async_trait]
impl LLMProvider for OpenAIProvider {
    /// Generate a chat completion (internally always streams).
    async fn generate(&self, request: &LLMRequest) -> Result<Response, Error> {
        let mut openai_request = self.convert_request(request);
        openai_request.stream = Some(true);

        debug!(
            request = ?openai_request,
            "sending OpenAI Responses API request"
        );

        let body = serde_json::to_vec(&openai_request)?;
        let req = TransportRequest {
            url: format!("{}/responses", self.base_url),
            headers: vec![
                (
                    "Authorization".to_string(),
                    format!("Bearer {}", self.api_key),
                ),
                ("Content-Type".to_string(), "application/json".to_string()),
            ],
            body,
        };
        let response = self.transport.send(req).await?;

        if !(200..300).contains(&response.status) {
            let status = response.status;
            let retry_after = response
                .header("retry-after")
                .and_then(|s| s.parse::<u64>().ok());
            let body_bytes = response.collect_body().await.unwrap_or_default();
            let body_str = String::from_utf8_lossy(&body_bytes).into_owned();
            return Err(parse_openai_error(status, retry_after, &body_str));
        }

        use crate::sse_stream::SseStreamExt;
        let state = Arc::new(Mutex::new(OpenAIStreamState::new()));
        let state_for_stream = state.clone();
        let event_stream = response
            .body
            .sse_events()
            .map(move |sse_result| -> Result<Vec<StreamEvent>, Error> {
                let sse_event = sse_result?;
                debug!(event = ?sse_event, "received OpenAI SSE event");
                let stream_event =
                    serde_json::from_str::<ResponsesStreamEvent>(&sse_event.data)?;
                state_for_stream.lock().unwrap().process(stream_event)
            })
            .flat_map(|result| match result {
                Ok(events) => futures_util::stream::iter(
                    events.into_iter().map(Ok).collect::<Vec<_>>(),
                ),
                Err(e) => futures_util::stream::iter(vec![Err(e)]),
            });

        // We can't read the continuation off the state until the stream
        // is fully consumed (response.completed sets it). That's fine for
        // most callers (they want the stream itself); buffer() picks up
        // the continuation when finalizing — see Response::with_continuation
        // for the wiring. We attach an empty continuation here; the
        // accumulator's finalize() can be extended in a follow-up to
        // poll the state at end-of-stream if we want this populated on
        // the streaming-only path too.
        let _ = state; // keep state alive (the closure also clones it)
        Ok(Response::from_stream(event_stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Prompt;

    #[test]
    fn test_provider_creation() {
        let provider = OpenAIProvider::new("test-key".to_string());
        assert!(provider.is_ok());
    }

    fn provider() -> OpenAIProvider {
        OpenAIProvider::new("k".to_string()).unwrap()
    }

    /// HTTP 429 with an OpenAI-shaped error body should produce
    /// [`Error::RateLimit`] (not the generic [`Error::Provider`]) so
    /// retry-aware callers can branch on it.
    #[test]
    fn http_429_maps_to_rate_limit() {
        let body = r#"{"error":{"message":"Rate limited","type":"rate_limit_error","code":"rate_limit_exceeded"}}"#;
        let err = parse_openai_error(429, Some(30), body);
        match err {
            Error::RateLimit {
                retry_after,
                message,
            } => {
                assert_eq!(retry_after, Some(std::time::Duration::from_secs(30)));
                assert!(message.contains("Rate limited"));
                assert!(message.contains("rate_limit_error"));
            }
            other => panic!("expected RateLimit, got {other:?}"),
        }
    }

    #[test]
    fn http_401_maps_to_auth() {
        let body =
            r#"{"error":{"message":"Bad key","type":"invalid_request_error","code":"invalid_api_key"}}"#;
        let err = parse_openai_error(401, None, body);
        assert!(matches!(err, Error::Auth { status: Some(401), .. }), "got {err:?}");
        assert!(format!("{err}").contains("Bad key"));
    }

    /// Non-JSON / non-conforming bodies still produce a useful error rather
    /// than swallowing the status code.
    #[test]
    fn unparseable_error_body_still_carries_status_and_body() {
        let err = parse_openai_error(500, None, "<html>500 Server Error</html>");
        match &err {
            Error::Provider { message, .. } => {
                assert!(message.contains("500"));
                assert!(message.contains("<html>"));
            }
            other => panic!("expected Provider, got {other:?}"),
        }
    }

    fn request_with_tool_choice(choice: ToolChoice) -> LLMRequest {
        LLMRequest::from_prompt("gpt-4", &Prompt::user("hi")).tool_choice(choice)
    }

    /// `tool_choice` must serialize to OpenAI's expected wire forms:
    /// the bare strings `"auto"` / `"none"` / `"required"` for modes, and
    /// `{"type":"function","name":"…"}` for a forced specific tool.
    #[test]
    fn tool_choice_serializes_modes_as_strings() {
        for (choice, expected) in [
            (ToolChoice::Auto, serde_json::json!("auto")),
            (ToolChoice::None, serde_json::json!("none")),
            (ToolChoice::Required, serde_json::json!("required")),
        ] {
            let req = provider().convert_request(&request_with_tool_choice(choice.clone()));
            let json = serde_json::to_value(&req).unwrap();
            assert_eq!(
                json["tool_choice"], expected,
                "ToolChoice::{choice:?} should serialize to {expected}",
            );
        }
    }

    #[test]
    fn tool_choice_serializes_function_as_typed_object() {
        let req = provider().convert_request(&request_with_tool_choice(ToolChoice::Function {
            name: "get_weather".to_string(),
        }));
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(
            json["tool_choice"],
            serde_json::json!({"type": "function", "name": "get_weather"}),
        );
    }

    /// `reasoning` configuration must reach the wire as
    /// `{"effort": "...", "summary": "..."}`. Both fields are optional.
    #[test]
    fn reasoning_config_serializes_to_correct_shape() {
        use crate::types::{ReasoningConfig, ReasoningEffort, ReasoningSummary};
        let req = provider().convert_request(
            &LLMRequest::from_prompt("gpt-5", &Prompt::user("hi")).reasoning(ReasoningConfig {
                effort: Some(ReasoningEffort::High),
                summary: Some(ReasoningSummary::Auto),
            }),
        );
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(
            json["reasoning"],
            serde_json::json!({"effort": "high", "summary": "auto"}),
        );
    }

    /// OpenAI's reasoning streaming events should open a Reasoning part
    /// and stream deltas into it.
    #[test]
    fn reasoning_summary_text_delta_routes_to_reasoning_part() {
        let mut state = OpenAIStreamState::new();
        let added: ResponsesStreamEvent = serde_json::from_str(
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1"}}"#,
        )
        .unwrap();
        let events = state.process(added).unwrap();
        assert!(
            matches!(&events[0], StreamEvent::PartStart { index: 0, kind: PartKind::Reasoning }),
            "expected PartStart(Reasoning), got {:?}", events,
        );

        let delta: ResponsesStreamEvent = serde_json::from_str(
            r#"{"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"hmm,"}"#,
        )
        .unwrap();
        let events = state.process(delta).unwrap();
        match &events[0] {
            StreamEvent::Delta { index, delta } => {
                assert_eq!(*index, 0);
                assert_eq!(delta, "hmm,");
            }
            other => panic!("expected Delta, got {other:?}"),
        }
    }

    /// `parallel_tool_calls` and `store` should reach the wire when the
    /// caller sets them, and stay absent when not. Default `store` is false
    /// (opt-out) so we don't unintentionally retain prompts server-side.
    #[test]
    fn parallel_tool_calls_and_store_are_caller_controlled() {
        let req = provider().convert_request(
            &LLMRequest::from_prompt("gpt-4", &Prompt::user("hi"))
                .parallel_tool_calls(false)
                .store(true),
        );
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["parallel_tool_calls"], false);
        assert_eq!(json["store"], true);
    }

    /// A `response.output_item.added` for a `function_call` opens a
    /// `ToolCall` part carrying the API's `call_id` — not the `fc_…`
    /// output item id. The two are not interchangeable.
    #[test]
    fn function_call_opens_with_api_call_id() {
        let mut state = OpenAIStreamState::new();
        let json = r#"{
            "type":"response.output_item.added",
            "output_index":0,
            "item":{
                "type":"function_call",
                "id":"fc_123",
                "name":"get_weather",
                "arguments":"",
                "call_id":"call_abc"
            }
        }"#;
        let event: ResponsesStreamEvent = serde_json::from_str(json).unwrap();
        let events = state.process(event).unwrap();
        match &events[0] {
            StreamEvent::PartStart {
                index: 0,
                kind: PartKind::ToolCall { call_id, name },
            } => {
                assert_eq!(call_id, "call_abc");
                assert_eq!(name, "get_weather");
            }
            other => panic!("expected PartStart(ToolCall), got {other:?}"),
        }
    }

    /// A function_call item with no `call_id` is malformed per the
    /// Responses API — silently substituting the `fc_*` id breaks
    /// multi-turn tool calls invisibly. Surface it as an error.
    #[test]
    fn function_call_added_without_call_id_errors() {
        let mut state = OpenAIStreamState::new();
        let json = r#"{
            "type":"response.output_item.added",
            "output_index":0,
            "item":{
                "type":"function_call",
                "id":"fc_123",
                "name":"get_weather",
                "arguments":""
            }
        }"#;
        let event: ResponsesStreamEvent = serde_json::from_str(json).unwrap();
        let result = state.process(event);
        assert!(
            result.is_err(),
            "function_call without call_id must error, got: {result:?}",
        );
    }

    #[test]
    fn test_request_conversion() {
        let provider = OpenAIProvider::new("test-key".to_string()).unwrap();
        let prompt = Prompt::user("Hello");
        let request = LLMRequest::from_prompt("gpt-4", &prompt)
            .temperature(0.7)
            .max_tokens(100);

        let openai_request = provider.convert_request(&request);
        assert_eq!(openai_request.model, "gpt-4");
        assert_eq!(openai_request.temperature, Some(0.7));
        assert_eq!(openai_request.max_output_tokens, Some(100));
    }
}
