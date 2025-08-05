use futures_util::StreamExt;
use gcp_auth::TokenProvider;
use reqwest::Client;
use std::sync::Arc;
use std::time::Duration;

use super::anthropic_types::*;
use crate::provider::LLMProvider;
use crate::sse_stream::SseStream;
use crate::types::{FinishReason, FunctionCall, InputItem, Role};
use crate::{Error, LLMRequest, Response, StreamEvent};

/// Authentication method for Anthropic provider via Vertex AI.
#[derive(Debug)]
pub enum AnthropicViaVertexAuth {
    /// Use access token (passed as Bearer header)
    AccessToken(String),
    /// Use Application Default Credentials (ADC)
    ApplicationDefault,
}

/// Anthropic Claude provider implementation via Vertex AI.
pub struct AnthropicViaVertexProvider {
    client: Client,
    project_id: String,
    location: String,
    auth: AnthropicViaVertexAuth,
    auth_manager: Option<Arc<dyn TokenProvider>>,
    base_url: Option<String>,
}

impl AnthropicViaVertexProvider {
    /// Create a new Anthropic provider with access token authentication.
    pub fn new(project_id: String, location: String, access_token: String) -> Result<Self, Error> {
        Self::with_auth(
            project_id,
            location,
            AnthropicViaVertexAuth::AccessToken(access_token),
        )
    }

    /// Create a new Anthropic provider with custom base URL (for testing).
    pub fn new_with_base_url(
        project_id: String,
        location: String,
        access_token: String,
        base_url: String,
    ) -> Result<Self, Error> {
        let mut provider = Self::with_auth(
            project_id,
            location,
            AnthropicViaVertexAuth::AccessToken(access_token),
        )?;
        provider.base_url = Some(base_url);
        Ok(provider)
    }

    /// Create a new Anthropic provider with Application Default Credentials.
    pub async fn with_adc(project_id: String, location: String) -> Result<Self, Error> {
        Self::with_auth_async(
            project_id,
            location,
            AnthropicViaVertexAuth::ApplicationDefault,
        )
        .await
    }

    /// Create a new Anthropic provider with specific authentication method (sync for access tokens).
    pub fn with_auth(
        project_id: String,
        location: String,
        auth: AnthropicViaVertexAuth,
    ) -> Result<Self, Error> {
        match auth {
            AnthropicViaVertexAuth::AccessToken(_) => {
                let client = Client::builder().timeout(Duration::from_secs(60)).build()?;

                Ok(Self {
                    client,
                    project_id,
                    location,
                    auth,
                    auth_manager: None,
                    base_url: None,
                })
            }
            AnthropicViaVertexAuth::ApplicationDefault => Err(Error::config(
                "Use with_auth_async() for Application Default Credentials",
            )),
        }
    }

    /// Create a new Anthropic provider with specific authentication method (async for ADC).
    pub async fn with_auth_async(
        project_id: String,
        location: String,
        auth: AnthropicViaVertexAuth,
    ) -> Result<Self, Error> {
        let client = Client::builder().timeout(Duration::from_secs(60)).build()?;

        let auth_manager = match &auth {
            AnthropicViaVertexAuth::ApplicationDefault => {
                Some(gcp_auth::provider().await.map_err(|e| {
                    Error::provider("Anthropic", format!("Failed to create auth manager: {e}"))
                })?)
            }
            AnthropicViaVertexAuth::AccessToken(_) => None,
        };

        Ok(Self {
            client,
            project_id,
            location,
            auth,
            auth_manager,
            base_url: None,
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
                            Error::provider("Anthropic", format!("Invalid function arguments: {e}"))
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
                                AnthropicContent::Blocks(blocks) => blocks
                                    .iter()
                                    .any(|b| matches!(b, AnthropicContentBlock::ToolResult { .. })),
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

    /// Get the API endpoint for the Anthropic model.
    fn get_endpoint(&self, stream: bool, model: &str) -> String {
        let method = if stream {
            "streamRawPredict"
        } else {
            "rawPredict"
        };
        let sse_param = if stream { "?alt=sse" } else { "" };

        if let Some(base_url) = &self.base_url {
            // Use custom base URL for testing
            format!(
                "{}/v1/projects/{}/locations/{}/publishers/anthropic/models/{}:{}{}",
                base_url.trim_end_matches('/'),
                self.project_id,
                self.location,
                model,
                method,
                sse_param
            )
        } else {
            // Use default Vertex AI endpoint
            format!(
                "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/anthropic/models/{}:{}{}",
                self.location, self.project_id, self.location, model, method, sse_param
            )
        }
    }
}

#[async_trait::async_trait]
impl LLMProvider for AnthropicViaVertexProvider {
    async fn generate(&self, request: &LLMRequest) -> Result<Response, Error> {
        let anthropic_request = self.convert_request(request)?;

        let endpoint = self.get_endpoint(true, &request.model);

        let mut request_builder = self
            .client
            .post(&endpoint)
            .header("Content-Type", "application/json")
            .json(&anthropic_request);

        // Add authentication based on the method
        request_builder = match &self.auth {
            AnthropicViaVertexAuth::AccessToken(token) => {
                request_builder.header("Authorization", format!("Bearer {token}"))
            }
            AnthropicViaVertexAuth::ApplicationDefault => {
                let auth_manager = self.auth_manager.as_ref().ok_or_else(|| {
                    Error::provider("Anthropic", "Auth manager not initialized for ADC")
                })?;

                let token = auth_manager
                    .token(&["https://www.googleapis.com/auth/cloud-platform"])
                    .await
                    .map_err(|e| {
                        Error::provider("Anthropic", format!("Failed to get ADC token: {e}"))
                    })?;

                request_builder.header("Authorization", format!("Bearer {}", token.as_str()))
            }
        };

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
                                match Self::convert_stream_event_stateful(stream_event, &mut state)
                                {
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

/// State for tracking in-progress function calls during streaming.
#[derive(Debug, Default)]
struct StreamState {
    /// In-progress function calls indexed by content block index
    in_progress_calls: std::collections::HashMap<u32, InProgressFunctionCall>,
}

/// A function call that's being built incrementally from streaming events.
#[derive(Debug)]
struct InProgressFunctionCall {
    id: String,
    name: String,
    input_buffer: String,    // Accumulates InputJsonDelta events
    has_initial_input: bool, // Whether we started with complete input
}

impl AnthropicViaVertexProvider {
    /// Convert stream event with state tracking for function calls.
    fn convert_stream_event_stateful(
        event: AnthropicStreamEvent,
        state: &mut StreamState,
    ) -> Result<Vec<StreamEvent>, Error> {
        let mut events = Vec::new();

        match event {
            AnthropicStreamEvent::MessageStart { .. } => {
                // Start of message - no events needed for now
            }
            AnthropicStreamEvent::ContentBlockStart {
                content_block,
                index,
            } => {
                match content_block {
                    AnthropicContentBlock::ToolUse { id, name, input } => {
                        // Handle tool use block start
                        events.push(StreamEvent::OutputItemAdded {
                            item: crate::types::OutputItemInfo::FunctionCall {
                                name: name.clone(),
                                id: id.clone(),
                            },
                        });

                        // Start tracking this function call - don't emit FunctionCallComplete yet
                        // Parameters may be streamed incrementally via InputJsonDelta events

                        // Check if we have initial input or if it will be streamed
                        let (initial_input, has_initial) = if input.is_null()
                            || (input.is_object() && input.as_object().unwrap().is_empty())
                        {
                            // No initial input, will be streamed via InputJsonDelta
                            (String::new(), false)
                        } else {
                            // We have complete initial input
                            let json = serde_json::to_string(&input).map_err(|e| {
                                Error::provider(
                                    "Anthropic",
                                    format!("Failed to serialize initial function input: {e}"),
                                )
                            })?;
                            (json, true)
                        };

                        state.in_progress_calls.insert(
                            index,
                            InProgressFunctionCall {
                                id: id.clone(),
                                name: name.clone(),
                                input_buffer: initial_input,
                                has_initial_input: has_initial,
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
            AnthropicStreamEvent::ContentBlockDelta { delta, index } => {
                match delta {
                    AnthropicContentDelta::TextDelta { text } => {
                        if !text.is_empty() {
                            events.push(StreamEvent::ContentDelta { delta: text });
                        }
                    }
                    AnthropicContentDelta::InputJsonDelta { partial_json } => {
                        // Handle function parameter updates
                        if let Some(in_progress) = state.in_progress_calls.get_mut(&index) {
                            if in_progress.has_initial_input {
                                // We already had complete input in ContentBlockStart
                                // InputJsonDelta is providing the same data again (or updates)
                                // Replace with the new data
                                in_progress.input_buffer = partial_json;
                            } else {
                                // We're building the input incrementally
                                // Append the partial JSON
                                in_progress.input_buffer.push_str(&partial_json);
                            }
                        }
                    }
                }
            }
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
            AnthropicStreamEvent::MessageDelta { delta } => {
                // Handle usage updates and stop reason
                if let Some(_usage) = delta.usage {
                    // Don't emit Done event here, wait for MessageStop
                }
            }
            AnthropicStreamEvent::MessageStop => {
                // Message is complete - emit done event
                events.push(StreamEvent::Done {
                    finish_reason: FinishReason::Stop, // TODO: Map actual stop reason
                    usage: crate::types::Usage::default(), // TODO: Get actual usage from message_delta
                });
            }
            AnthropicStreamEvent::Ping => {
                // Keep-alive event - ignore
            }
        }

        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{stream, StreamExt};

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

        // Collect all events using StreamExt::next
        let mut sse_stream = sse_stream;
        while let Some(sse_result) = sse_stream.next().await {
            let sse_event = sse_result.expect("SSE should parse correctly");
            let data = sse_event.data.trim();

            if data.is_empty() {
                continue;
            }

            // Parse as AnthropicStreamEvent
            match serde_json::from_str::<AnthropicStreamEvent>(data) {
                Ok(stream_event) => {
                    let mut state = StreamState::default();
                    match AnthropicViaVertexProvider::convert_stream_event_stateful(
                        stream_event,
                        &mut state,
                    ) {
                        Ok(stream_events) => {
                            events.extend(stream_events);
                        }
                        Err(e) => panic!("Should parse successfully: {e}"),
                    }
                }
                Err(e) => panic!("Should parse JSON successfully: {e}"),
            }
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
