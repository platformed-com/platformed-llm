use futures_util::StreamExt;
use ijson::ijson;
use std::collections::HashSet;
use uuid::Uuid;

use super::google_types::*;
use super::transport::VertexTransport;
use crate::provider::LLMProvider;
use crate::sse_stream::SseStream;
use crate::types::{FinishReason, FunctionCall, InputItem, Role};
use crate::{Error, LLMRequest, Response, StreamEvent};

/// Google provider implementation via Vertex AI (for Gemini models).
pub struct GoogleProvider {
    transport: VertexTransport,
}

impl GoogleProvider {
    /// Create a new Google provider with access token authentication.
    pub fn new(project_id: String, location: String, access_token: String) -> Result<Self, Error> {
        Ok(Self {
            transport: VertexTransport::with_access_token(project_id, location, access_token)?,
        })
    }

    /// Create a new Google provider with a custom base URL (for testing).
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

    /// Create a new Google provider with Application Default Credentials.
    pub async fn with_adc(project_id: String, location: String) -> Result<Self, Error> {
        Ok(Self {
            transport: VertexTransport::with_adc(project_id, location).await?,
        })
    }

    /// Convert internal request to Google format.
    fn convert_request(&self, request: &LLMRequest) -> Result<GoogleRequest, Error> {
        let mut contents = Vec::new();
        let mut system_instruction = None;

        for item in &request.messages {
            match item {
                InputItem::Message(msg) => {
                    match msg.role {
                        Role::System => {
                            // Google uses system_instruction field for system messages
                            system_instruction = Some(GoogleContent {
                                role: "user".to_string(), // System instructions are treated as user content
                                parts: vec![GooglePart::Text {
                                    text: msg.content.clone(),
                                }],
                            });
                        }
                        Role::User => {
                            contents.push(GoogleContent {
                                role: "user".to_string(),
                                parts: vec![GooglePart::Text {
                                    text: msg.content.clone(),
                                }],
                            });
                        }
                        Role::Assistant => {
                            contents.push(GoogleContent {
                                role: "model".to_string(),
                                parts: vec![GooglePart::Text {
                                    text: msg.content.clone(),
                                }],
                            });
                        }
                    }
                }
                InputItem::FunctionCall(call) => {
                    // Add function call to the last model response or create a new one
                    if let Some(last_content) = contents.last_mut() {
                        if last_content.role == "model" {
                            last_content.parts.push(GooglePart::FunctionCall {
                                function_call: GoogleFunctionCall {
                                    name: call.name.clone(),
                                    args: serde_json::from_str(&call.arguments).map_err(|e| {
                                        Error::provider(
                                            "Google",
                                            format!("Invalid function arguments: {e}"),
                                        )
                                    })?,
                                },
                            });
                        } else {
                            // Create a new model content with the function call
                            contents.push(GoogleContent {
                                role: "model".to_string(),
                                parts: vec![GooglePart::FunctionCall {
                                    function_call: GoogleFunctionCall {
                                        name: call.name.clone(),
                                        args: serde_json::from_str(&call.arguments).map_err(
                                            |e| {
                                                Error::provider(
                                                    "Google",
                                                    format!("Invalid function arguments: {e}"),
                                                )
                                            },
                                        )?,
                                    },
                                }],
                            });
                        }
                    } else {
                        contents.push(GoogleContent {
                            role: "model".to_string(),
                            parts: vec![GooglePart::FunctionCall {
                                function_call: GoogleFunctionCall {
                                    name: call.name.clone(),
                                    args: serde_json::from_str(&call.arguments).map_err(|e| {
                                        Error::provider(
                                            "Google",
                                            format!("Invalid function arguments: {e}"),
                                        )
                                    })?,
                                },
                            }],
                        });
                    }
                }
                InputItem::FunctionCallOutput { call_id, output } => {
                    // Find the function name for this call_id
                    let function_name = self
                        .find_function_name_by_call_id(&contents, call_id)
                        .unwrap_or_else(|| "unknown".to_string());

                    // Check if the last content is already a user message with function responses
                    let should_append = if let Some(last_content) = contents.last() {
                        last_content.role == "user"
                            && last_content
                                .parts
                                .iter()
                                .any(|p| matches!(p, GooglePart::FunctionResponse { .. }))
                    } else {
                        false
                    };

                    if should_append {
                        // Add this response to the existing user message
                        if let Some(last_content) = contents.last_mut() {
                            last_content.parts.push(GooglePart::FunctionResponse {
                                function_response: GoogleFunctionResponse {
                                    name: function_name,
                                    response: ijson!({ "result": output }),
                                },
                            });
                        }
                    } else {
                        // Create a new user message with the function response
                        contents.push(GoogleContent {
                            role: "user".to_string(),
                            parts: vec![GooglePart::FunctionResponse {
                                function_response: GoogleFunctionResponse {
                                    name: function_name,
                                    response: ijson!({ "result": output }),
                                },
                            }],
                        });
                    }
                }
            }
        }

        let generation_config = Some(GoogleGenerationConfig {
            temperature: request.temperature,
            max_output_tokens: request.max_tokens,
            top_p: request.top_p,
        });

        let tools = request.tools.as_ref().map(|tools| {
            vec![GoogleTool {
                function_declarations: tools
                    .iter()
                    .map(|tool| GoogleFunctionDeclaration {
                        name: tool.function.name.clone(),
                        description: tool.function.description.clone(),
                        parameters: tool.function.parameters.clone(),
                    })
                    .collect(),
            }]
        });

        let google_request = GoogleRequest {
            contents,
            generation_config,
            tools,
            system_instruction,
        };

        Ok(google_request)
    }

    /// Find the function name associated with a call_id.
    /// This is a simplified implementation that assumes function responses are processed
    /// in the same order as function calls were made.
    fn find_function_name_by_call_id(
        &self,
        contents: &[GoogleContent],
        _call_id: &str,
    ) -> Option<String> {
        // Count how many function responses we've already processed
        let response_count = contents
            .iter()
            .filter(|c| c.role == "user")
            .flat_map(|c| &c.parts)
            .filter(|p| matches!(p, GooglePart::FunctionResponse { .. }))
            .count();

        // Find the corresponding function call
        let mut call_count = 0;
        for content in contents {
            if content.role == "model" {
                for part in &content.parts {
                    if let GooglePart::FunctionCall { function_call } = part {
                        if call_count == response_count {
                            return Some(function_call.name.clone());
                        }
                        call_count += 1;
                    }
                }
            }
        }
        None
    }
}

#[async_trait::async_trait]
impl LLMProvider for GoogleProvider {
    async fn generate(&self, request: &LLMRequest) -> Result<Response, Error> {
        let google_request = self.convert_request(request)?;

        let endpoint = self.transport.endpoint(
            "google",
            &request.model,
            "streamGenerateContent",
            Some("alt=sse"),
        );

        let builder = self
            .transport
            .client()
            .post(&endpoint)
            .header("Content-Type", "application/json")
            .json(&google_request);
        let request_builder = self.transport.authorize(builder).await?;

        let response = request_builder.send().await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(Error::provider(
                "Google",
                format!("API error: {error_text}"),
            ));
        }

        // Create SSE stream from response (Gemini supports ?alt=sse)
        let byte_stream = response.bytes_stream();
        let sse_stream = SseStream::new(byte_stream);

        // Create a stateful processor for tracking output items
        let mut state = GoogleStreamState::default();

        let event_stream = sse_stream
            .map(move |sse_result| {
                match sse_result {
                    Ok(sse_event) => {
                        let data = sse_event.data.trim();

                        // Skip [DONE] events and empty events
                        if data == "[DONE]" || data.is_empty() {
                            return vec![];
                        }

                        // Parse the SSE data as GoogleResponse
                        match serde_json::from_str::<GoogleResponse>(data) {
                            Ok(google_response) => {
                                match convert_response_stateful(google_response, &mut state) {
                                    Ok(stream_events) => {
                                        stream_events.into_iter().map(Ok).collect()
                                    }
                                    Err(e) => vec![Err(e)],
                                }
                            }
                            Err(e) => {
                                vec![Err(Error::provider(
                                    "Google",
                                    format!("Failed to parse SSE event: {e}"),
                                ))]
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

/// State for tracking output items during streaming to avoid duplicate OutputItemAdded events.
#[derive(Debug, Default)]
pub(crate) struct GoogleStreamState {
    /// Whether we've started text output
    has_text_output: bool,
    /// Set of function call IDs we've already announced
    announced_function_calls: HashSet<String>,
}

/// Stateful version of convert_response that tracks output items to emit OutputItemAdded only once.
///
/// `pub(crate)` so unit tests in this module (and future fixture-driven tests in
/// `tests/`) can drive the conversion directly with synthetic `GoogleResponse`
/// values, without standing up a transport or a mock server.
pub(crate) fn convert_response_stateful(
    response: GoogleResponse,
    state: &mut GoogleStreamState,
) -> Result<Vec<StreamEvent>, Error> {
    let mut events = Vec::new();

    if let Some(candidate) = response.candidates.first() {
        for part in &candidate.content.parts {
            match part {
                GooglePart::Text { text } => {
                    // Only emit OutputItemAdded if we haven't started text output yet
                    if !state.has_text_output {
                        events.push(StreamEvent::OutputItemAdded {
                            item: crate::types::OutputItemInfo::Text,
                        });
                        state.has_text_output = true;
                    }
                    if !text.is_empty() {
                        events.push(StreamEvent::ContentDelta {
                            delta: text.clone(),
                        });
                    }
                }
                GooglePart::FunctionCall { function_call } => {
                    // Use a deterministic ID based on function name and arguments for tracking
                    let function_key = format!(
                        "{}:{}",
                        function_call.name,
                        serde_json::to_string(&function_call.args).unwrap_or_default()
                    );

                    // Use a single UUID for both id and call_id to ensure consistency
                    let base_id = Uuid::new_v4().simple().to_string();
                    let fc_id = format!("fc_{base_id}");
                    let call_id = format!("call_{base_id}");

                    // Only emit OutputItemAdded if we haven't announced this function call yet
                    if !state.announced_function_calls.contains(&function_key) {
                        events.push(StreamEvent::OutputItemAdded {
                            item: crate::types::OutputItemInfo::FunctionCall {
                                name: function_call.name.clone(),
                                id: fc_id.clone(),
                            },
                        });
                        state.announced_function_calls.insert(function_key);
                    }

                    // Convert function call
                    let function_call_obj = FunctionCall {
                        call_id,
                        name: function_call.name.clone(),
                        arguments: serde_json::to_string(&function_call.args).map_err(|e| {
                            Error::provider(
                                "Google",
                                format!("Failed to serialize function args: {e}"),
                            )
                        })?,
                    };
                    events.push(StreamEvent::FunctionCallComplete {
                        call: function_call_obj,
                    });
                }
                GooglePart::FunctionResponse { .. } => {
                    // Function responses are typically not part of the model's output
                }
            }
        }

        // Only add a Done event if this response has a finish_reason (indicates end of stream)
        if let Some(finish_reason_str) = &candidate.finish_reason {
            let finish_reason = match finish_reason_str.as_str() {
                "STOP" => FinishReason::Stop,
                "MAX_TOKENS" => FinishReason::Length,
                "SAFETY" => FinishReason::ContentFilter,
                _ => FinishReason::Stop, // Default to Stop for unknown reasons
            };

            let usage = response
                .usage_metadata
                .map(|meta| meta.into())
                .unwrap_or_default();

            events.push(StreamEvent::Done {
                finish_reason,
                usage,
            });
        }
    } else if let Some(feedback) = &response.prompt_feedback {
        // Prompt was safety-blocked. Surface as ContentFilter regardless of
        // the specific reason (SAFETY / BLOCKLIST / PROHIBITED_CONTENT / SPII
        // / OTHER) — they all mean "the model declined to respond".
        if let Some(reason) = &feedback.block_reason {
            tracing::warn!(
                block_reason = %reason,
                message = ?feedback.block_reason_message,
                "Gemini prompt was blocked",
            );
        }
        let usage = response
            .usage_metadata
            .map(|meta| meta.into())
            .unwrap_or_default();
        events.push(StreamEvent::Done {
            finish_reason: FinishReason::ContentFilter,
            usage,
        });
    } else if response.usage_metadata.is_some() {
        // If no candidates but we have usage metadata, this might be a final response
        let usage = response
            .usage_metadata
            .map(|meta| meta.into())
            .unwrap_or_default();
        events.push(StreamEvent::Done {
            finish_reason: FinishReason::Stop,
            usage,
        });
    }

    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{stream, StreamExt};

    #[tokio::test]
    async fn test_streaming_content_parsing() {
        // Simulate realistic Google streaming response chunks
        let chunk1 = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Google"}]}}]}"#;
        let chunk2 =
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":" Gemini"}]}}]}"#;
        let chunk3 = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":" is"}]}}]}"#;
        let final_chunk = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":" an AI."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"totalTokenCount":30}}"#;

        let byte_chunks: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
            Ok(bytes::Bytes::from(format!("data: {chunk1}\n\n"))),
            Ok(bytes::Bytes::from(format!("data: {chunk2}\n\n"))),
            Ok(bytes::Bytes::from(format!("data: {chunk3}\n\n"))),
            Ok(bytes::Bytes::from(format!("data: {final_chunk}\n\n"))),
            Ok(bytes::Bytes::from("data: [DONE]\n\n")),
        ];

        let byte_stream = stream::iter(byte_chunks);
        let sse_stream = crate::sse_stream::SseStream::new(byte_stream);

        // Process events through our Gemini SSE handler
        let mut events = Vec::new();
        let mut shared_state = GoogleStreamState::default(); // Share state across SSE events like real usage

        // Collect all events using StreamExt::next
        let mut sse_stream = sse_stream;
        while let Some(sse_result) = sse_stream.next().await {
            let sse_event = sse_result.expect("SSE should parse correctly");
            let data = sse_event.data.trim();

            if data == "[DONE]" || data.is_empty() {
                continue;
            }

            // Parse as GoogleResponse
            match serde_json::from_str::<GoogleResponse>(data) {
                Ok(response) => match convert_response_stateful(response, &mut shared_state) {
                    Ok(stream_events) => {
                        events.extend(stream_events);
                    }
                    Err(e) => panic!("Should parse successfully: {e}"),
                },
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

        assert_eq!(content_events, vec!["Google", " Gemini", " is", " an AI."]);

        // Verify we got exactly one Done event at the end
        let done_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::Done { .. }))
            .collect();

        assert_eq!(done_events.len(), 1);

        // The Done event should be the last event
        assert!(matches!(events.last(), Some(StreamEvent::Done { .. })));
    }

    /// When Vertex blocks a prompt at the safety layer, it returns
    /// `{"promptFeedback":{...}}` with no `candidates` array. Deserializing
    /// that previously failed because `candidates: Vec<...>` was required —
    /// surfaced to callers as a misleading "Failed to parse SSE event"
    /// instead of a real `Done`/blocked signal.
    #[test]
    fn prompt_feedback_only_response_parses() {
        let json = r#"{"promptFeedback":{"blockReason":"SAFETY"}}"#;
        let response: GoogleResponse = serde_json::from_str(json)
            .expect("safety-blocked response with no candidates must parse");

        let mut state = GoogleStreamState::default();
        let events = convert_response_stateful(response, &mut state).unwrap();

        let done = events.iter().find_map(|e| match e {
            StreamEvent::Done { finish_reason, .. } => Some(finish_reason.clone()),
            _ => None,
        });
        assert_eq!(
            done,
            Some(FinishReason::ContentFilter),
            "promptFeedback.blockReason should yield Done with ContentFilter",
        );
    }

    /// Vertex sometimes emits a candidate with `finishReason: SAFETY` and
    /// `content: {role: model}` (no `parts`). `parts` was a required field;
    /// the parse failure leaked through as a generic "Failed to parse SSE
    /// event" error.
    #[test]
    fn candidate_with_no_parts_parses() {
        let json = r#"{
            "candidates":[{"content":{"role":"model"},"finishReason":"SAFETY"}]
        }"#;
        let response: GoogleResponse =
            serde_json::from_str(json).expect("candidate without parts must parse");

        let mut state = GoogleStreamState::default();
        let events = convert_response_stateful(response, &mut state).unwrap();
        let done = events.iter().find_map(|e| match e {
            StreamEvent::Done { finish_reason, .. } => Some(finish_reason.clone()),
            _ => None,
        });
        assert_eq!(done, Some(FinishReason::ContentFilter));
    }

    #[test]
    fn request_body_uses_camel_case_keys() {
        // Vertex AI's REST API expects camelCase keys: `generationConfig`,
        // `systemInstruction`, `functionDeclarations`. snake_case keys are
        // silently ignored, which means temperature / maxOutputTokens / topP
        // / system instructions / tool definitions all silently drop on the
        // floor. Lock in the correct shape.
        use crate::types::{Function, InputItem, Tool, ToolType};

        let tool = Tool {
            r#type: ToolType::Function,
            function: Function {
                name: "get_weather".to_string(),
                description: "Get the weather".to_string(),
                parameters: serde_json::from_str(
                    r#"{"type":"object","properties":{}}"#,
                )
                .unwrap(),
            },
        };
        let request = LLMRequest {
            model: "gemini-1.5-pro".to_string(),
            messages: vec![
                InputItem::system("You are helpful."),
                InputItem::user("Hi"),
            ],
            temperature: Some(0.5),
            max_tokens: Some(128),
            top_p: Some(0.9),
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: Some(vec![tool]),
        };

        let provider = GoogleProvider::new(
            "p".to_string(),
            "us-east1".to_string(),
            "tok".to_string(),
        )
        .unwrap();
        let google_request = provider.convert_request(&request).unwrap();
        let json = serde_json::to_value(&google_request).unwrap();

        // Top-level keys
        assert!(
            json.get("generationConfig").is_some(),
            "request must serialize 'generationConfig' (camelCase), got: {json}",
        );
        assert!(
            json.get("systemInstruction").is_some(),
            "request must serialize 'systemInstruction' (camelCase), got: {json}",
        );
        assert!(
            json.get("generation_config").is_none(),
            "request must NOT contain snake_case 'generation_config'",
        );
        assert!(
            json.get("system_instruction").is_none(),
            "request must NOT contain snake_case 'system_instruction'",
        );

        // Inside generationConfig
        let gen_cfg = &json["generationConfig"];
        assert!(gen_cfg.get("temperature").is_some(), "missing temperature");
        assert!(gen_cfg.get("maxOutputTokens").is_some(), "missing maxOutputTokens");
        assert!(gen_cfg.get("topP").is_some(), "missing topP");
        assert!(
            gen_cfg.get("max_output_tokens").is_none(),
            "generationConfig must NOT contain snake_case 'max_output_tokens'",
        );
        assert!(
            gen_cfg.get("top_p").is_none(),
            "generationConfig must NOT contain snake_case 'top_p'",
        );

        // Inside tools[0]
        let tool_obj = &json["tools"][0];
        assert!(
            tool_obj.get("functionDeclarations").is_some(),
            "tools[0] must serialize 'functionDeclarations' (camelCase)",
        );
        assert!(
            tool_obj.get("function_declarations").is_none(),
            "tools[0] must NOT contain snake_case 'function_declarations'",
        );
    }

    #[tokio::test]
    async fn test_real_google_sse_response_format() {
        // Test the exact Google SSE format you provided to see where the concatenation happens
        let real_google_sse = r#"data: {"candidates": [{"content": {"role": "model","parts": [{"text": "The"}]}}],"usageMetadata": {"trafficType": "ON_DEMAND"},"modelVersion": "gemini-1.5-pro-002","createTime": "2025-07-22T11:00:54.652094Z","responseId": "Zm9_aL7mJ8bAxN8PnfeY2AY"}

data: {"candidates": [{"content": {"role": "model","parts": [{"text": " weather in Tokyo"}]}}],"usageMetadata": {"trafficType": "ON_DEMAND"},"modelVersion": "gemini-1.5-pro-002","createTime": "2025-07-22T11:00:54.652094Z","responseId": "Zm9_aL7mJ8bAxN8PnfeY2AY"}

data: {"candidates": [{"content": {"role": "model","parts": [{"text": ", Japan is currently sunny"}]}}],"usageMetadata": {"trafficType": "ON_DEMAND"},"modelVersion": "gemini-1.5-pro-002","createTime": "2025-07-22T11:00:54.652094Z","responseId": "Zm9_aL7mJ8bAxN8PnfeY2AY"}

"#;

        let byte_chunks: Vec<Result<bytes::Bytes, std::io::Error>> =
            vec![Ok(bytes::Bytes::from(real_google_sse))];

        let byte_stream = stream::iter(byte_chunks);
        let sse_stream = crate::sse_stream::SseStream::new(byte_stream);

        let mut all_parsed_events = Vec::new();
        let mut event_count = 0;
        let mut shared_state = GoogleStreamState::default();

        let mut sse_stream = sse_stream;
        while let Some(sse_result) = sse_stream.next().await {
            let sse_event = sse_result.expect("SSE should parse correctly");
            event_count += 1;
            let data = sse_event.data.trim();
            if data == "[DONE]" || data.is_empty() {
                continue;
            }
            let google_response: GoogleResponse =
                serde_json::from_str(data).expect("JSON should parse");
            let stream_events = convert_response_stateful(google_response, &mut shared_state)
                .expect("conversion should succeed");
            all_parsed_events.extend(stream_events);
        }

        assert_eq!(event_count, 3, "Should have 3 separate SSE events");
        assert_eq!(
            all_parsed_events.len(),
            4,
            "Should generate 4 stream events (1 OutputItemAdded + 3 ContentDelta)"
        );
    }
}
