use super::types::{OpenAIReasoning, OpenAIToolChoice, ResponsesRequest, ResponsesStreamEvent};
use crate::provider::LLMProvider;
use crate::types::{ReasoningConfig, ReasoningEffort, ReasoningSummary, ToolChoice};
use crate::{Error, LLMRequest, Response, StreamEvent};
use futures::TryStreamExt as _;
use reqwest::Client;
use std::time::Duration;
use tracing::debug;

/// OpenAI provider implementation.
pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

/// Connect timeout for the underlying HTTP client.
///
/// We deliberately do **not** set a total request timeout — streaming
/// reasoning responses (gpt-5 / o-series) can legitimately run for many
/// minutes, and a whole-request timeout aborts them mid-stream.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

fn build_client() -> Result<Client, Error> {
    Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .build()
        .map_err(Error::from)
}

impl OpenAIProvider {
    /// Create a new OpenAI provider.
    pub fn new(api_key: String) -> Result<Self, Error> {
        Ok(Self {
            client: build_client()?,
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
        })
    }

    /// Create a new OpenAI provider with custom base URL.
    pub fn new_with_base_url(api_key: String, base_url: String) -> Result<Self, Error> {
        Ok(Self {
            client: build_client()?,
            api_key,
            base_url,
        })
    }

    /// Convert internal request to OpenAI Responses API format.
    fn convert_request(&self, request: &LLMRequest) -> ResponsesRequest {
        // Convert items to OpenAI format
        let input: Vec<crate::providers::openai::types::OpenAIInputMessage> =
            request.messages.iter().map(Self::convert_message).collect();

        ResponsesRequest {
            model: request.model.clone(),
            input,
            instructions: None, // System messages will be in input array
            temperature: request.temperature,
            max_output_tokens: request.max_tokens,
            top_p: request.top_p,
            tools: request
                .tools
                .as_ref()
                .map(|tools| Self::convert_tools(tools)),
            tool_choice: request.tool_choice.as_ref().map(convert_tool_choice),
            parallel_tool_calls: request.parallel_tool_calls,
            previous_response_id: None, // Will be set when we add conversation support
            stream: None,               // Will be set by the generate methods
            // Default to opt-out so callers don't accidentally have prompts
            // retained server-side. Override via LLMRequest::store(true) when
            // intentionally chaining via `previous_response_id`.
            store: Some(request.store.unwrap_or(false)),
            reasoning: request.reasoning.as_ref().map(convert_reasoning),
        }
    }

    /// Convert our internal InputItem to OpenAI format.
    fn convert_message(
        item: &crate::types::InputItem,
    ) -> crate::providers::openai::types::OpenAIInputMessage {
        use crate::providers::openai::types::OpenAIInputMessage;

        match item {
            crate::types::InputItem::Message(msg) => {
                let role = match msg.role {
                    crate::types::Role::System => "system",
                    crate::types::Role::User => "user",
                    crate::types::Role::Assistant => "assistant",
                };

                let text = msg.text_content();

                OpenAIInputMessage::Regular {
                    role: role.to_string(),
                    content: text,
                }
            }
            crate::types::InputItem::FunctionCall(call) => OpenAIInputMessage::FunctionCall {
                call_id: call.call_id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            },
            crate::types::InputItem::FunctionCallOutput { call_id, output } => {
                OpenAIInputMessage::FunctionCallOutput {
                    call_id: call_id.clone(),
                    output: output.clone(),
                }
            }
        }
    }

    /// Convert our internal tools to OpenAI Responses API format.
    #[allow(clippy::ptr_arg)]
    fn convert_tools(tools: &[crate::types::Tool]) -> Vec<super::types::OpenAITool> {
        tools
            .iter()
            .map(|tool| {
                super::types::OpenAITool {
                    r#type: "function".to_string(), // OpenAI Responses API expects "function"
                    name: tool.function.name.clone(),
                    description: tool.function.description.clone().unwrap_or_default(),
                    parameters: tool.function.parameters.clone(),
                }
            })
            .collect()
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
        401 => Error::auth(format!("OpenAI 401 ({kind} {code}): {message}")),
        429 => Error::rate_limit(
            retry_after_seconds,
            format!("OpenAI 429 ({kind} {code}): {message}"),
        ),
        _ => Error::provider(
            "OpenAI",
            format!("HTTP {status} ({kind} {code}): {message}"),
        ),
    }
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

impl OpenAIProvider {
    /// Convert an OpenAI streaming event to our `StreamEvent`s.
    ///
    /// `pub(crate)` so unit tests can drive this with synthetic events.
    pub(crate) fn convert_stream_event(
        event: ResponsesStreamEvent,
    ) -> Result<Option<StreamEvent>, Error> {
        match event.r#type.as_str() {
            "error" => {
                let (type_, message) = if let Some(error) = &event.error {
                    (error.r#type.as_str(), error.message.as_str())
                } else {
                    ("unknown", "Unknown error occurred")
                };
                return Err(Error::provider("OpenAI", format!("{type_}: {message}")));
            }
            "response.output_text.delta" => {
                if let Some(delta) = event.delta {
                    if !delta.is_empty() {
                        return Ok(Some(StreamEvent::ContentDelta { delta }));
                    }
                }
            }
            "response.output_item.added" => {
                // Handle new output item being added
                if let Some(item) = event.item {
                    // Emit OutputItemAdded event with type-specific info
                    let item_info = match item.r#type.as_str() {
                        "function_call" => {
                            // For function calls, include the name and ID if available
                            let name = item.name.unwrap_or_else(|| "unknown".to_string());
                            crate::types::OutputItemInfo::FunctionCall { name, id: item.id }
                        }
                        "reasoning" => crate::types::OutputItemInfo::Reasoning,
                        "message" => crate::types::OutputItemInfo::Text,
                        _ => crate::types::OutputItemInfo::Text,
                    };

                    return Ok(Some(StreamEvent::OutputItemAdded { item: item_info }));
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                if let Some(delta) = event.delta {
                    if !delta.is_empty() {
                        return Ok(Some(StreamEvent::ReasoningDelta { delta }));
                    }
                }
            }
            "response.reasoning_summary_text.done" | "response.reasoning_text.done" => {
                // Final canonical text — we already accumulated via deltas.
            }
            "response.function_call_arguments.delta" => {
                // We no longer emit FunctionCallArguments events
                // Arguments are accumulated internally and only complete calls are emitted
                // This event is ignored for now
            }
            "response.function_call_arguments.done" => {
                // Function call arguments are complete but no complete data here
            }
            "response.output_item.done" => {
                // Output item is done - emit FunctionCallComplete for function calls.
                if let Some(item) = event.item {
                    if item.r#type == "function_call" {
                        if let (Some(name), Some(arguments)) = (item.name, item.arguments) {
                            // The Responses API always populates `call_id`
                            // (`call_…`) on function-call items. The `id`
                            // field (`fc_…`) is the output-item id and is
                            // NOT interchangeable — subsequent
                            // `function_call_output` items must reference
                            // the `call_…` id or the model can't correlate
                            // them. Refuse to silently substitute.
                            let call_id = item.call_id.ok_or_else(|| {
                                Error::provider(
                                    "OpenAI",
                                    format!(
                                        "function_call output item is missing required \
                                         `call_id` field (item id: {})",
                                        item.id,
                                    ),
                                )
                            })?;
                            let call = crate::types::FunctionCall {
                                call_id,
                                name,
                                arguments,
                            };
                            return Ok(Some(StreamEvent::FunctionCallComplete { call }));
                        }
                    }
                }
            }
            "response.completed" => {
                // The response is complete - emit any completed function calls
                if let Some(response) = event.response {
                    // Determine finish reason
                    let finish_reason =
                        if response.output.iter().any(|o| o.r#type == "function_call") {
                            crate::types::FinishReason::ToolCalls
                        } else {
                            crate::types::FinishReason::Stop
                        };

                    return Ok(Some(StreamEvent::Done {
                        finish_reason,
                        usage: response.usage.unwrap_or_default(),
                    }));
                }
            }
            _ => {
                // Ignore other event types for now
            }
        }

        Ok(None)
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

        let response = self
            .client
            .post(format!("{}/responses", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&openai_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            let body = response.text().await.unwrap_or_default();
            return Err(parse_openai_error(status, retry_after, &body));
        }

        // Create a stream from the response bytes
        let byte_stream = response.bytes_stream();

        // Use the clean SSE stream adapter
        use crate::sse_stream::SseStreamExt;
        let event_stream = byte_stream
            .sse_events()
            .try_filter_map(|sse_event| async move {
                debug!(event = ?sse_event, "received OpenAI SSE event");
                let stream_event = serde_json::from_str::<ResponsesStreamEvent>(&sse_event.data)?;
                OpenAIProvider::convert_stream_event(stream_event)
            });

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
                retry_after_seconds,
                message,
            } => {
                assert_eq!(retry_after_seconds, Some(30));
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
        assert!(matches!(err, Error::Auth(_)), "got {err:?}");
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

    /// OpenAI's reasoning streaming events should be routed onto the
    /// unified `ReasoningDelta` channel; the corresponding
    /// `response.output_item.added` for type `reasoning` should produce a
    /// `Reasoning` `OutputItemInfo` so the accumulator opens a reasoning
    /// item before deltas arrive.
    #[test]
    fn reasoning_summary_text_delta_emits_reasoning_delta() {
        let added: ResponsesStreamEvent = serde_json::from_str(
            r#"{"type":"response.output_item.added","item":{"type":"reasoning","id":"rs_1"}}"#,
        )
        .unwrap();
        match OpenAIProvider::convert_stream_event(added).unwrap().unwrap() {
            StreamEvent::OutputItemAdded {
                item: crate::types::OutputItemInfo::Reasoning,
            } => {}
            other => panic!("expected OutputItemAdded(Reasoning), got {other:?}"),
        }

        let delta: ResponsesStreamEvent = serde_json::from_str(
            r#"{"type":"response.reasoning_summary_text.delta","delta":"hmm,"}"#,
        )
        .unwrap();
        match OpenAIProvider::convert_stream_event(delta).unwrap().unwrap() {
            StreamEvent::ReasoningDelta { delta } => assert_eq!(delta, "hmm,"),
            other => panic!("expected ReasoningDelta, got {other:?}"),
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

    /// On a `response.output_item.done` for a `function_call`, the unified
    /// `FunctionCallComplete.call_id` MUST come from the API's `call_id`
    /// (`call_…`) — not the output item id (`fc_…`). The two are not
    /// interchangeable; subsequent `function_call_output` items have to
    /// reference the `call_…` id or the model can't correlate them.
    #[test]
    fn function_call_done_uses_api_call_id() {
        let json = r#"{
            "type":"response.output_item.done",
            "item":{
                "type":"function_call",
                "id":"fc_123",
                "name":"get_weather",
                "arguments":"{\"city\":\"Paris\"}",
                "call_id":"call_abc"
            }
        }"#;
        let event: ResponsesStreamEvent = serde_json::from_str(json).unwrap();
        let stream_event = OpenAIProvider::convert_stream_event(event)
            .expect("conversion should succeed")
            .expect("should produce a StreamEvent");

        match stream_event {
            StreamEvent::FunctionCallComplete { call } => {
                assert_eq!(
                    call.call_id, "call_abc",
                    "call_id must come from API's call_id field, not item.id",
                );
                assert_ne!(
                    call.call_id, "fc_123",
                    "call_id must NOT alias to the fc_* output item id",
                );
            }
            other => panic!("expected FunctionCallComplete, got {other:?}"),
        }
    }

    /// A `function_call` item with no `call_id` is malformed per the
    /// Responses API — silently substituting the `fc_*` id breaks multi-turn
    /// tool calls invisibly. Surface it as an error instead.
    #[test]
    fn function_call_done_without_call_id_errors() {
        let json = r#"{
            "type":"response.output_item.done",
            "item":{
                "type":"function_call",
                "id":"fc_123",
                "name":"get_weather",
                "arguments":"{}"
            }
        }"#;
        let event: ResponsesStreamEvent = serde_json::from_str(json).unwrap();
        let result = OpenAIProvider::convert_stream_event(event);
        assert!(
            result.is_err(),
            "function_call without call_id must error, got: {result:?}",
        );
    }

    #[test]
    fn test_request_conversion() {
        let provider = OpenAIProvider::new("test-key".to_string()).unwrap();
        let prompt = Prompt::user("Hello");
        let request = LLMRequest {
            model: "gpt-4".to_string(),
            messages: prompt.items().to_vec(),
            temperature: Some(0.7),
            max_tokens: Some(100),
            top_p: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            store: None,
            reasoning: None,
        };

        let openai_request = provider.convert_request(&request);
        assert_eq!(openai_request.model, "gpt-4");
        assert_eq!(openai_request.temperature, Some(0.7));
        assert_eq!(openai_request.max_output_tokens, Some(100));
    }
}
