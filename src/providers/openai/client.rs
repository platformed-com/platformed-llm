use std::time::Duration;
use reqwest::Client;
use crate::{Error, LLMRequest, Response, StreamEvent};
use crate::provider::LLMProvider;
use super::types::{ResponsesRequest, ResponsesStreamEvent};
use futures_util::StreamExt;

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
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()?;
        
        Ok(Self {
            client,
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
        })
    }
    
    /// Create a new OpenAI provider with custom base URL.
    pub fn new_with_base_url(api_key: String, base_url: String) -> Result<Self, Error> {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()?;
        
        Ok(Self {
            client,
            api_key,
            base_url,
        })
    }
    
    /// Convert internal request to OpenAI Responses API format.
    fn convert_request(&self, request: &LLMRequest) -> ResponsesRequest {
        // Convert items to OpenAI format
        let input: Vec<crate::providers::openai::types::OpenAIInputMessage> = request.messages.iter()
            .map(Self::convert_message)
            .collect();
        
        ResponsesRequest {
            model: request.model.clone(),
            input,
            instructions: None, // System messages will be in input array
            temperature: request.temperature,
            max_output_tokens: request.max_tokens,
            top_p: request.top_p,
            tools: request.tools.as_ref().map(|tools| Self::convert_tools(tools)),
            tool_choice: None, // Will be set later when we add function calling
            parallel_tool_calls: Some(true),
            previous_response_id: None, // Will be set when we add conversation support
            stream: None, // Will be set by the generate methods
            store: Some(false), // Don't store by default for our abstraction
        }
    }
    
    /// Convert our internal InputItem to OpenAI format.
    fn convert_message(item: &crate::types::InputItem) -> crate::providers::openai::types::OpenAIInputMessage {
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
            },
            crate::types::InputItem::FunctionCall(call) => {
                OpenAIInputMessage::FunctionCall {
                    id: call.id.clone(),
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                }
            },
            crate::types::InputItem::FunctionCallOutput { call_id, output } => {
                OpenAIInputMessage::FunctionCallOutput {
                    call_id: call_id.clone(),
                    output: output.clone(),
                }
            },
        }
    }
    
    /// Convert our internal tools to OpenAI Responses API format.
    fn convert_tools(tools: &[crate::types::Tool]) -> Vec<super::types::OpenAITool> {
        tools.iter().map(|tool| {
            super::types::OpenAITool {
                r#type: "function".to_string(), // OpenAI Responses API expects "function"
                name: tool.function.name.clone(),
                description: tool.function.description.clone(),
                parameters: tool.function.parameters.clone(),
            }
        }).collect()
    }
    
    /// Convert an OpenAI streaming event to our StreamEvents (static version).
    fn convert_stream_event_static(event: ResponsesStreamEvent) -> Result<Vec<StreamEvent>, Error> {
        match event.r#type.as_str() {
            "response.output_text.delta" => {
                if let Some(delta) = event.delta {
                    if !delta.is_empty() {
                        return Ok(vec![StreamEvent::ContentDelta { delta }]);
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
                            crate::types::OutputItemInfo::FunctionCall {
                                name,
                                id: item.id,
                            }
                        }
                        "message" => crate::types::OutputItemInfo::Text,
                        _ => crate::types::OutputItemInfo::Text,
                    };
                    
                    return Ok(vec![StreamEvent::OutputItemAdded {
                        item: item_info,
                    }]);
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
                // Output item is done - this might contain complete function call data
                if let Some(item) = event.item {
                    if item.r#type == "function_call" {
                        if let (Some(name), Some(args)) = (&item.name, &item.arguments) {
                            let function_call = crate::types::FunctionCall {
                                id: item.id.clone(), // Use the actual ID (starts with "fc_")
                                call_id: item.call_id.clone().unwrap_or_else(|| item.id.clone()), // Use call_id for function results
                                name: name.clone(),
                                arguments: args.clone(),
                            };
                            return Ok(vec![StreamEvent::FunctionCallComplete { call: function_call }]);
                        }
                    }
                }
            }
            "response.completed" => {
                // The response is complete - emit any completed function calls
                if let Some(response) = event.response {
                    let mut events = Vec::new();
                    
                    // Extract completed function calls from the response
                    for output in &response.output {
                        if output.r#type == "function_call" {
                            // Function call is a separate output item
                            if let (Some(name), Some(args)) = (&output.name, &output.arguments) {
                                let function_call = crate::types::FunctionCall {
                                    id: output.id.clone(), // Use the actual ID (starts with "fc_")
                                    call_id: output.call_id.clone().unwrap_or_else(|| output.id.clone()), // Use call_id for function results
                                    name: name.clone(),
                                    arguments: args.clone(),
                                };
                                events.push(StreamEvent::FunctionCallComplete { call: function_call });
                            }
                        }
                    }
                    
                    // Determine finish reason
                    let finish_reason = if !events.is_empty() {
                        crate::types::FinishReason::ToolCalls
                    } else {
                        crate::types::FinishReason::Stop
                    };
                    
                    // Add the done event
                    events.push(StreamEvent::Done { 
                        finish_reason,
                        usage: response.usage 
                    });
                    
                    return Ok(events);
                }
            }
            _ => {
                // Ignore other event types for now
            }
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
        
        let response = self.client
            .post(format!("{}/responses", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&openai_request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(Error::provider("OpenAI", format!("API error: {error_text}")));
        }
        
        // Create a stream from the response bytes
        let byte_stream = response.bytes_stream();
        
        // Use the clean SSE stream adapter
        use crate::sse_stream::SseStreamExt;
        let event_stream = byte_stream
            .sse_events()
            .filter_map(|sse_result| async move {
                match sse_result {
                    Ok(sse_event) => {
                        if sse_event.is_done() {
                            // End of stream
                            return None;
                        }
                        
                        // Parse the JSON data as a ResponsesStreamEvent
                        if let Ok(event) = serde_json::from_str::<ResponsesStreamEvent>(&sse_event.data) {
                            match OpenAIProvider::convert_stream_event_static(event) {
                                Ok(events) => Some(Ok(events)),
                                Err(e) => Some(Err(e)),
                            }
                        } else {
                            // Skip unparseable events (might be comments or other SSE data)
                            None
                        }
                    }
                    Err(e) => Some(Err(e)),
                }
            })
            .map(|events_result| {
                match events_result {
                    Ok(events) => events.into_iter().map(Ok).collect::<Vec<_>>(),
                    Err(e) => vec![Err(e)],
                }
            })
            .map(|events| {
                futures_util::stream::iter(events.into_iter())
            })
            .flatten();
        
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