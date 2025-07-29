use futures_util::StreamExt;
use gcp_auth::AuthenticationManager;
use reqwest::Client;
use std::time::Duration;
use uuid::Uuid;

use super::google_types::*;
use crate::provider::LLMProvider;
use crate::sse_stream::SseStream;
use crate::types::{FinishReason, FunctionCall, InputItem, Role};
use crate::{Error, LLMRequest, Response, StreamEvent};

/// Authentication method for Google provider.
#[derive(Debug)]
pub enum GoogleAuth {
    /// Use access token (passed as Bearer header)
    AccessToken(String),
    /// Use Application Default Credentials (ADC)
    ApplicationDefault,
}

/// Google provider implementation via Vertex AI (for Gemini models).
pub struct GoogleProvider {
    client: Client,
    project_id: String,
    location: String,
    auth: GoogleAuth,
    auth_manager: Option<AuthenticationManager>,
    base_url: Option<String>,
}

impl GoogleProvider {
    /// Create a new Google provider with access token authentication.
    pub fn new(project_id: String, location: String, access_token: String) -> Result<Self, Error> {
        Self::with_auth(project_id, location, GoogleAuth::AccessToken(access_token))
    }

    /// Create a new Google provider with custom base URL (for testing).
    pub fn new_with_base_url(
        project_id: String,
        location: String,
        access_token: String,
        base_url: String,
    ) -> Result<Self, Error> {
        let mut provider =
            Self::with_auth(project_id, location, GoogleAuth::AccessToken(access_token))?;
        provider.base_url = Some(base_url);
        Ok(provider)
    }

    /// Create a new Google provider with Application Default Credentials.
    pub async fn with_adc(project_id: String, location: String) -> Result<Self, Error> {
        Self::with_auth_async(project_id, location, GoogleAuth::ApplicationDefault).await
    }

    /// Create a new Google provider with specific authentication method (sync for access tokens).
    pub fn with_auth(
        project_id: String,
        location: String,
        auth: GoogleAuth,
    ) -> Result<Self, Error> {
        match auth {
            GoogleAuth::AccessToken(_) => {
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
            GoogleAuth::ApplicationDefault => Err(Error::config(
                "Use with_auth_async() for Application Default Credentials",
            )),
        }
    }

    /// Create a new Google provider with specific authentication method (async for ADC).
    pub async fn with_auth_async(
        project_id: String,
        location: String,
        auth: GoogleAuth,
    ) -> Result<Self, Error> {
        let client = Client::builder().timeout(Duration::from_secs(60)).build()?;

        let auth_manager = match &auth {
            GoogleAuth::ApplicationDefault => {
                Some(AuthenticationManager::new().await.map_err(|e| {
                    Error::provider("Google", format!("Failed to create auth manager: {e}"))
                })?)
            }
            GoogleAuth::AccessToken(_) => None,
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
                                    response: serde_json::json!({ "result": output }),
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
                                    response: serde_json::json!({ "result": output }),
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

    /// Get the API endpoint for the Google model.
    fn get_endpoint(&self, stream: bool, model: &str) -> String {
        let method = if stream {
            "streamGenerateContent"
        } else {
            "generateContent"
        };
        let sse_param = if stream { "?alt=sse" } else { "" };

        if let Some(base_url) = &self.base_url {
            // Use custom base URL for testing
            format!(
                "{}/v1/projects/{}/locations/{}/publishers/google/models/{}:{}{}",
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
                "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:{}{}",
                self.location, self.project_id, self.location, model, method, sse_param
            )
        }
    }
}

#[async_trait::async_trait]
impl LLMProvider for GoogleProvider {
    async fn generate(&self, request: &LLMRequest) -> Result<Response, Error> {
        let google_request = self.convert_request(request)?;

        let endpoint = self.get_endpoint(true, &request.model);

        let mut request_builder = self
            .client
            .post(&endpoint)
            .header("Content-Type", "application/json")
            .json(&google_request);

        // Add authentication based on the method
        request_builder = match &self.auth {
            GoogleAuth::AccessToken(token) => {
                request_builder.header("Authorization", format!("Bearer {token}"))
            }
            GoogleAuth::ApplicationDefault => {
                let auth_manager = self.auth_manager.as_ref().ok_or_else(|| {
                    Error::provider("Google", "Auth manager not initialized for ADC")
                })?;

                let token = auth_manager
                    .get_token(&["https://www.googleapis.com/auth/cloud-platform"])
                    .await
                    .map_err(|e| {
                        Error::provider("Google", format!("Failed to get ADC token: {e}"))
                    })?;

                request_builder.header("Authorization", format!("Bearer {}", token.as_str()))
            }
        };

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
                                match Self::convert_response_stateful(google_response, &mut state) {
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
struct GoogleStreamState {
    /// Whether we've started text output
    has_text_output: bool,
    /// Set of function call IDs we've already announced
    announced_function_calls: std::collections::HashSet<String>,
}

impl GoogleProvider {
    /// Stateful version of convert_response that tracks output items to emit OutputItemAdded only once.
    fn convert_response_stateful(
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
                Ok(response) => {
                    match GoogleProvider::convert_response_stateful(response, &mut shared_state) {
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

    #[tokio::test]
    async fn test_trailing_characters_error_simulation() {
        // Simulate the exact scenario that would cause a "trailing characters" error
        // This simulates what might happen if Google's API sends extra whitespace or formatting

        // Test case 1: JSON with trailing whitespace
        let problematic_chunk1 =
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Test"}]}}]}  "#; // Extra spaces

        // Test case 2: JSON with trailing newline
        let problematic_chunk2 =
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Test2"}]}}]}
"#
            .to_string(); // Extra newline

        // Test case 3: Multiple JSON objects (should trigger fallback)
        let problematic_chunk3 = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Test3"}]}}]}
{"candidates":[{"content":{"role":"model","parts":[{"text":"Test4"}]}}]}"#;

        let test_cases = vec![
            ("trailing spaces", problematic_chunk1),
            ("trailing newline", &problematic_chunk2),
            ("multiple json", problematic_chunk3),
        ];

        for (case_name, json_data) in test_cases {
            println!("Testing case: {case_name}");

            // Try parsing directly to see what error we get
            match serde_json::from_str::<GoogleResponse>(json_data) {
                Ok(_) => println!("  ✅ Case '{case_name}' parsed successfully (no error)"),
                Err(e) => {
                    println!("  ❌ Case '{case_name}' failed with error: {e}");
                    if e.to_string().contains("trailing characters") {
                        println!("  🎯 Found trailing characters error!");

                        // Test our fallback parsing logic
                        let mut all_events = Vec::new();
                        for line in json_data.lines() {
                            let line = line.trim();
                            if line.is_empty() || line == "[DONE]" {
                                continue;
                            }

                            match serde_json::from_str::<GoogleResponse>(line) {
                                Ok(google_response) => {
                                    let mut test_state = GoogleStreamState::default();
                                    match GoogleProvider::convert_response_stateful(
                                        google_response,
                                        &mut test_state,
                                    ) {
                                        Ok(stream_events) => {
                                            all_events.extend(stream_events);
                                            println!(
                                                "  ✅ Successfully parsed line: '{}'",
                                                line.chars().take(50).collect::<String>()
                                            );
                                        }
                                        Err(e) => {
                                            println!("  ❌ Failed to convert response: {e}");
                                        }
                                    }
                                }
                                Err(line_err) => {
                                    println!(
                                        "  ⚠️ Failed to parse line '{}': {}",
                                        line.chars().take(30).collect::<String>(),
                                        line_err
                                    );
                                }
                            }
                        }
                        println!("  📊 Fallback parsing produced {} events", all_events.len());
                    }
                }
            }
            println!();
        }
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

        // Process events through our Google SSE handler to see where the issue occurs
        let mut all_parsed_events = Vec::new();
        let mut event_count = 0;
        let mut shared_state = GoogleStreamState::default(); // Share state across all SSE events like in real usage

        let mut sse_stream = sse_stream;
        while let Some(sse_result) = sse_stream.next().await {
            match sse_result {
                Ok(sse_event) => {
                    event_count += 1;
                    let data = sse_event.data.trim();
                    println!("SSE Event #{}: data length = {}", event_count, data.len());
                    println!(
                        "SSE Event #{}: first 100 chars = {:?}",
                        event_count,
                        data.chars().take(100).collect::<String>()
                    );

                    if data == "[DONE]" || data.is_empty() {
                        continue;
                    }

                    // This is the exact logic from the Google provider
                    match serde_json::from_str::<GoogleResponse>(data) {
                        Ok(google_response) => {
                            match GoogleProvider::convert_response_stateful(
                                google_response,
                                &mut shared_state,
                            ) {
                                Ok(stream_events) => {
                                    println!(
                                        "✅ SSE Event #{}: Successfully parsed {} stream events",
                                        event_count,
                                        stream_events.len()
                                    );
                                    all_parsed_events.extend(stream_events);
                                }
                                Err(e) => {
                                    println!("❌ SSE Event #{event_count}: Failed to convert: {e}");
                                }
                            }
                        }
                        Err(e) => {
                            println!("❌ SSE Event #{event_count}: JSON parse error: {e}");
                            if e.to_string().contains("trailing characters") {
                                println!("🚨 SSE Event #{event_count}: This is the trailing characters error!");
                                println!("🔍 Data causing error: {data:?}");
                            }
                        }
                    }
                }
                Err(e) => {
                    println!("❌ SSE parsing error: {e}");
                }
            }
        }

        println!("📊 Total SSE events processed: {event_count}");
        println!(
            "📊 Total stream events generated: {}",
            all_parsed_events.len()
        );

        // Verify we got separate events, not concatenated ones
        assert_eq!(event_count, 3, "Should have 3 separate SSE events");
        assert_eq!(
            all_parsed_events.len(),
            4,
            "Should generate 4 stream events (1 OutputItemAdded + 3 ContentDelta)"
        );
    }

    #[tokio::test]
    async fn test_potential_edge_cases_causing_trailing_chars() {
        // Test various edge cases that might cause the trailing characters error in production

        // Case 1: CRLF line endings (Windows-style)
        let crlf_sse = "data: {\"candidates\": [{\"content\": {\"role\": \"model\",\"parts\": [{\"text\": \"Test1\"}]}}]}\r\n\r\ndata: {\"candidates\": [{\"content\": {\"role\": \"model\",\"parts\": [{\"text\": \"Test2\"}]}}]}\r\n\r\n";

        // Case 2: Mixed line endings
        let mixed_sse = "data: {\"candidates\": [{\"content\": {\"role\": \"model\",\"parts\": [{\"text\": \"Test1\"}]}}]}\r\n\ndata: {\"candidates\": [{\"content\": {\"role\": \"model\",\"parts\": [{\"text\": \"Test2\"}]}}]}\n\r\n";

        // Case 3: UTF-8 BOM at the start
        let bom_sse = "\u{FEFF}data: {\"candidates\": [{\"content\": {\"role\": \"model\",\"parts\": [{\"text\": \"Test1\"}]}}]}\n\n";

        // Case 4: Extra whitespace in data field
        let whitespace_sse = "data:  {\"candidates\": [{\"content\": {\"role\": \"model\",\"parts\": [{\"text\": \"Test1\"}]}}]}  \n\n";

        let test_cases = vec![
            ("CRLF endings", crlf_sse),
            ("Mixed endings", mixed_sse),
            ("UTF-8 BOM", bom_sse),
            ("Extra whitespace", whitespace_sse),
        ];

        for (case_name, sse_data) in test_cases {
            println!("Testing case: {case_name}");

            let byte_chunks: Vec<Result<bytes::Bytes, std::io::Error>> =
                vec![Ok(bytes::Bytes::from(sse_data))];

            let byte_stream = stream::iter(byte_chunks);
            let sse_stream = crate::sse_stream::SseStream::new(byte_stream);

            let mut events_processed = 0;
            let mut sse_stream = sse_stream;

            while let Some(sse_result) = sse_stream.next().await {
                match sse_result {
                    Ok(sse_event) => {
                        events_processed += 1;
                        let data = sse_event.data.trim();

                        if data.is_empty() || data == "[DONE]" {
                            continue;
                        }

                        // Try parsing the data
                        match serde_json::from_str::<GoogleResponse>(data) {
                            Ok(_) => {
                                println!("  ✅ Event {events_processed}: Parsed successfully");
                            }
                            Err(e) => {
                                println!("  ❌ Event {events_processed}: Parse error: {e}");
                                if e.to_string().contains("trailing characters") {
                                    println!("  🚨 Found trailing characters error in case '{case_name}'!");
                                    println!("  🔍 Raw data bytes: {:?}", data.as_bytes());
                                    println!("  🔍 Data repr: {data:?}");
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("  ❌ SSE parse error: {e}");
                    }
                }
            }

            println!("  📊 Total events processed: {events_processed}");
            println!();
        }
    }
}
