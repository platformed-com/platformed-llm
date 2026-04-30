use super::types::{ResponsesRequest, ResponsesStreamEvent};
use crate::provider::LLMProvider;
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

// Removed function call tracking structs - no longer needed since we handle complete calls only

impl OpenAIProvider {
    /// Create a new OpenAI provider.
    pub fn new(api_key: String) -> Result<Self, Error> {
        let client = Client::builder().timeout(Duration::from_secs(60)).build()?;

        Ok(Self {
            client,
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
        })
    }

    /// Create a new OpenAI provider with custom base URL.
    pub fn new_with_base_url(api_key: String, base_url: String) -> Result<Self, Error> {
        let client = Client::builder().timeout(Duration::from_secs(60)).build()?;

        Ok(Self {
            client,
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
            tool_choice: None, // Will be set later when we add function calling
            parallel_tool_calls: Some(true),
            previous_response_id: None, // Will be set when we add conversation support
            stream: None,               // Will be set by the generate methods
            store: Some(false),         // Don't store by default for our abstraction
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
    fn convert_tools(tools: &[crate::types::Tool]) -> Vec<super::types::OpenAITool> {
        tools
            .iter()
            .map(|tool| {
                super::types::OpenAITool {
                    r#type: "function".to_string(), // OpenAI Responses API expects "function"
                    name: tool.function.name.clone(),
                    description: tool.function.description.clone(),
                    parameters: tool.function.parameters.clone(),
                }
            })
            .collect()
    }

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
                        "message" => crate::types::OutputItemInfo::Text,
                        _ => crate::types::OutputItemInfo::Text,
                    };

                    return Ok(Some(StreamEvent::OutputItemAdded { item: item_info }));
                }
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
            let error_text = response.text().await?;
            return Err(Error::provider(
                "OpenAI",
                format!("API error: {error_text}"),
            ));
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
        };

        let openai_request = provider.convert_request(&request);
        assert_eq!(openai_request.model, "gpt-4");
        assert_eq!(openai_request.temperature, Some(0.7));
        assert_eq!(openai_request.max_output_tokens, Some(100));
    }
}
