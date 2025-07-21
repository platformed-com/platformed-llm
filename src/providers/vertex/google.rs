use std::time::Duration;
use reqwest::Client;
use futures_util::StreamExt;
use uuid::Uuid;
use gcp_auth::AuthenticationManager;

use crate::{Error, LLMRequest, Response, StreamEvent};
use crate::provider::LLMProvider;
use crate::types::{FinishReason, FunctionCall, InputItem, Role};
use crate::sse_stream::SseStream;
use super::google_types::*;

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
    model_id: String,
    auth: GoogleAuth,
    auth_manager: Option<AuthenticationManager>,
    base_url: Option<String>,
}

impl GoogleProvider {
    /// Create a new Google provider with access token authentication.
    pub fn new(project_id: String, location: String, model_id: String, access_token: String) -> Result<Self, Error> {
        Self::with_auth(project_id, location, model_id, GoogleAuth::AccessToken(access_token))
    }
    
    /// Create a new Google provider with custom base URL (for testing).
    pub fn new_with_base_url(project_id: String, location: String, model_id: String, access_token: String, base_url: String) -> Result<Self, Error> {
        let mut provider = Self::with_auth(project_id, location, model_id, GoogleAuth::AccessToken(access_token))?;
        provider.base_url = Some(base_url);
        Ok(provider)
    }
    
    /// Create a new Google provider with Application Default Credentials.
    pub async fn with_adc(project_id: String, location: String, model_id: String) -> Result<Self, Error> {
        Self::with_auth_async(project_id, location, model_id, GoogleAuth::ApplicationDefault).await
    }
    
    /// Create a new Google provider with specific authentication method (sync for access tokens).
    pub fn with_auth(project_id: String, location: String, model_id: String, auth: GoogleAuth) -> Result<Self, Error> {
        match auth {
            GoogleAuth::AccessToken(_) => {
                let client = Client::builder()
                    .timeout(Duration::from_secs(60))
                    .build()?;
                
                Ok(Self {
                    client,
                    project_id,
                    location,
                    model_id,
                    auth,
                    auth_manager: None,
                    base_url: None,
                })
            }
            GoogleAuth::ApplicationDefault => {
                Err(Error::config("Use with_auth_async() for Application Default Credentials"))
            }
        }
    }
    
    /// Create a new Google provider with specific authentication method (async for ADC).
    pub async fn with_auth_async(project_id: String, location: String, model_id: String, auth: GoogleAuth) -> Result<Self, Error> {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()?;
        
        let auth_manager = match &auth {
            GoogleAuth::ApplicationDefault => {
                Some(AuthenticationManager::new().await
                    .map_err(|e| Error::provider("Google", format!("Failed to create auth manager: {e}")))?)
            }
            GoogleAuth::AccessToken(_) => None,
        };
        
        Ok(Self {
            client,
            project_id,
            location,
            model_id,
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
                                parts: vec![GooglePart::Text { text: msg.content.clone() }],
                            });
                        }
                        Role::User => {
                            contents.push(GoogleContent {
                                role: "user".to_string(),
                                parts: vec![GooglePart::Text { text: msg.content.clone() }],
                            });
                        }
                        Role::Assistant => {
                            contents.push(GoogleContent {
                                role: "model".to_string(),
                                parts: vec![GooglePart::Text { text: msg.content.clone() }],
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
                                    args: serde_json::from_str(&call.arguments)
                                        .map_err(|e| Error::provider("Google", format!("Invalid function arguments: {e}")))?,
                                },
                            });
                        } else {
                            // Create a new model content with the function call
                            contents.push(GoogleContent {
                                role: "model".to_string(),
                                parts: vec![GooglePart::FunctionCall {
                                    function_call: GoogleFunctionCall {
                                        name: call.name.clone(),
                                        args: serde_json::from_str(&call.arguments)
                                            .map_err(|e| Error::provider("Google", format!("Invalid function arguments: {e}")))?,
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
                                    args: serde_json::from_str(&call.arguments)
                                        .map_err(|e| Error::provider("Google", format!("Invalid function arguments: {e}")))?,
                                },
                            }],
                        });
                    }
                }
                InputItem::FunctionCallOutput { call_id, output } => {
                    // Find the function name for this call_id
                    let function_name = self.find_function_name_by_call_id(&contents, call_id)
                        .unwrap_or_else(|| "unknown".to_string());
                    
                    // Check if the last content is already a user message with function responses
                    let should_append = if let Some(last_content) = contents.last() {
                        last_content.role == "user" && 
                        last_content.parts.iter().any(|p| matches!(p, GooglePart::FunctionResponse { .. }))
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
                function_declarations: tools.iter().map(|tool| {
                    GoogleFunctionDeclaration {
                        name: tool.function.name.clone(),
                        description: tool.function.description.clone(),
                        parameters: tool.function.parameters.clone(),
                    }
                }).collect(),
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
    fn find_function_name_by_call_id(&self, contents: &[GoogleContent], _call_id: &str) -> Option<String> {
        // Count how many function responses we've already processed
        let response_count = contents.iter()
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
    fn get_endpoint(&self, stream: bool) -> String {
        let method = if stream { "streamGenerateContent" } else { "generateContent" };
        let sse_param = if stream { "?alt=sse" } else { "" };
        
        if let Some(base_url) = &self.base_url {
            // Use custom base URL for testing
            format!(
                "{}/v1/projects/{}/locations/{}/publishers/google/models/{}:{}{}",
                base_url.trim_end_matches('/'), self.project_id, self.location, self.model_id, method, sse_param
            )
        } else {
            // Use default Vertex AI endpoint
            format!(
                "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:{}{}",
                self.location, self.project_id, self.location, self.model_id, method, sse_param
            )
        }
    }
}

#[async_trait::async_trait]
impl LLMProvider for GoogleProvider {
    async fn generate(&self, request: &LLMRequest) -> Result<Response, Error> {
        let google_request = self.convert_request(request)?;
        
        let endpoint = self.get_endpoint(true);
        
        let mut request_builder = self.client
            .post(&endpoint)
            .header("Content-Type", "application/json")
            .json(&google_request);
        
        // Add authentication based on the method
        request_builder = match &self.auth {
            GoogleAuth::AccessToken(token) => {
                request_builder.header("Authorization", format!("Bearer {token}"))
            }
            GoogleAuth::ApplicationDefault => {
                let auth_manager = self.auth_manager.as_ref()
                    .ok_or_else(|| Error::provider("Google", "Auth manager not initialized for ADC"))?;
                
                let token = auth_manager.get_token(&["https://www.googleapis.com/auth/cloud-platform"]).await
                    .map_err(|e| Error::provider("Google", format!("Failed to get ADC token: {e}")))?;
                
                request_builder.header("Authorization", format!("Bearer {}", token.as_str()))
            }
        };
        
        let response = request_builder.send().await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(Error::provider("Google", format!("API error: {error_text}")));
        }
        
        // Create SSE stream from response (Gemini supports ?alt=sse)
        let byte_stream = response.bytes_stream();
        let sse_stream = SseStream::new(byte_stream);
        
        let event_stream = sse_stream
            .map(|sse_result| {
                match sse_result {
                    Ok(sse_event) => {
                        // Handle special Gemini SSE events
                        let data = sse_event.data.trim();
                        
                        // Skip [DONE] events - they signal end of stream
                        if data == "[DONE]" {
                            return vec![];
                        }
                        
                        // Skip empty events
                        if data.is_empty() {
                            return vec![];
                        }
                        
                        // Parse the SSE data as Google JSON response
                        // Handle cases where there might be multiple lines with JSON objects
                        let mut all_events = Vec::new();
                        
                        // Try parsing as a single JSON object first
                        match serde_json::from_str::<GoogleResponse>(data) {
                            Ok(google_response) => {
                                match Self::convert_response_static(google_response) {
                                    Ok(stream_events) => {
                                        stream_events.into_iter().map(Ok).collect::<Vec<_>>()
                                    }
                                    Err(e) => vec![Err(e)],
                                }
                            }
                            Err(_) => {
                                // If single JSON parsing fails, try parsing each line separately
                                for line in data.lines() {
                                    let line = line.trim();
                                    if line.is_empty() || line == "[DONE]" {
                                        continue;
                                    }
                                    
                                    match serde_json::from_str::<GoogleResponse>(line) {
                                        Ok(google_response) => {
                                            match Self::convert_response_static(google_response) {
                                                Ok(stream_events) => {
                                                    all_events.extend(stream_events.into_iter().map(Ok));
                                                }
                                                Err(e) => {
                                                    all_events.push(Err(e));
                                                }
                                            }
                                        }
                                        Err(_) => {
                                            // Try as stream chunk
                                            match serde_json::from_str::<GoogleStreamChunk>(line) {
                                                Ok(stream_chunk) => {
                                                    let response = GoogleResponse {
                                                        candidates: stream_chunk.candidates,
                                                        usage_metadata: stream_chunk.usage_metadata,
                                                    };
                                                    match Self::convert_response_static(response) {
                                                        Ok(stream_events) => {
                                                            all_events.extend(stream_events.into_iter().map(Ok));
                                                        }
                                                        Err(e) => {
                                                            all_events.push(Err(e));
                                                        }
                                                    }
                                                }
                                                Err(e) => {
                                                    // Skip lines that can't be parsed as JSON
                                                    // This might be [DONE] markers or other control messages
                                                    if !line.starts_with('[') && !line.starts_with('{') {
                                                        continue;
                                                    }
                                                    all_events.push(Err(Error::provider("Google", format!("Failed to parse SSE event: {e}"))));
                                                }
                                            }
                                        }
                                    }
                                }
                                
                                all_events
                            }
                        }
                    }
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

impl GoogleProvider {
    /// Static version of convert_response for use in stream processing.
    fn convert_response_static(response: GoogleResponse) -> Result<Vec<StreamEvent>, Error> {
        let mut events = Vec::new();
        
        if let Some(candidate) = response.candidates.first() {
            for part in &candidate.content.parts {
                match part {
                    GooglePart::Text { text } => {
                        if !text.is_empty() {
                            events.push(StreamEvent::ContentDelta { delta: text.clone() });
                        }
                    }
                    GooglePart::FunctionCall { function_call } => {
                        // Convert function call
                        let function_call_obj = FunctionCall {
                            id: format!("fc_{}", Uuid::new_v4().simple()),
                            call_id: format!("call_{}", Uuid::new_v4().simple()),
                            name: function_call.name.clone(),
                            arguments: serde_json::to_string(&function_call.args)
                                .map_err(|e| Error::provider("Google", format!("Failed to serialize function args: {e}")))?,
                        };
                        events.push(StreamEvent::FunctionCallComplete { call: function_call_obj });
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
                
                let usage = response.usage_metadata
                    .map(|meta| meta.into())
                    .unwrap_or_default();
                
                events.push(StreamEvent::Done { finish_reason, usage });
            }
        } else if response.usage_metadata.is_some() {
            // If no candidates but we have usage metadata, this might be a final response
            let usage = response.usage_metadata
                .map(|meta| meta.into())
                .unwrap_or_default();
            events.push(StreamEvent::Done { finish_reason: FinishReason::Stop, usage });
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
        let chunk2 = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":" Gemini"}]}}]}"#;
        let chunk3 = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":" is"}]}}]}"#;
        let final_chunk = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":" an AI."}]},"finish_reason":"STOP"}],"usage_metadata":{"prompt_token_count":10,"candidates_token_count":20,"total_token_count":30}}"#;
        
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
                    match GoogleProvider::convert_response_static(response) {
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
        let content_events: Vec<_> = events.iter()
            .filter_map(|e| match e {
                StreamEvent::ContentDelta { delta } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        
        assert_eq!(content_events, vec!["Google", " Gemini", " is", " an AI."]);
        
        // Verify we got exactly one Done event at the end
        let done_events: Vec<_> = events.iter()
            .filter(|e| matches!(e, StreamEvent::Done { .. }))
            .collect();
        
        assert_eq!(done_events.len(), 1);
        
        // The Done event should be the last event
        assert!(matches!(events.last(), Some(StreamEvent::Done { .. })));
    }
}