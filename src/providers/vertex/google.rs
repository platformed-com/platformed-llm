use futures_util::StreamExt;
use ijson::{ijson, IValue};
use uuid::Uuid;

use super::endpoint::VertexEndpoint;
use super::google_types::*;
use crate::provider::LLMProvider;
use crate::sse_stream::SseStream;
use crate::transport::{Transport, TransportRequest};
use crate::types::{
    AssistantPart, FinishReason, FunctionCall, InputItem, PartKind, UserPart,
};
use crate::{Error, LLMRequest, Response, StreamEvent};

/// Google provider implementation via Vertex AI (for Gemini models).
pub struct GoogleProvider {
    endpoint: VertexEndpoint,
    transport: Transport,
}

impl GoogleProvider {
    /// Create a new Google provider with access token authentication.
    pub fn new(project_id: String, location: String, access_token: String) -> Result<Self, Error> {
        Ok(Self {
            endpoint: VertexEndpoint::with_access_token(project_id, location, access_token),
            transport: Transport::reqwest()?,
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
            endpoint: VertexEndpoint::with_access_token(project_id, location, access_token)
                .with_base_url(base_url),
            transport: Transport::reqwest()?,
        })
    }

    /// Create a new Google provider with Application Default Credentials.
    pub async fn with_adc(project_id: String, location: String) -> Result<Self, Error> {
        Ok(Self {
            endpoint: VertexEndpoint::with_adc(project_id, location).await?,
            transport: Transport::reqwest()?,
        })
    }

    /// Create a new Google provider with a caller-supplied [`Transport`]
    /// and pre-built [`VertexEndpoint`]. Lets downstream consumers / tests
    /// plug in custom recording / replaying / retrying transports.
    pub fn with_transport(endpoint: VertexEndpoint, transport: Transport) -> Self {
        Self { endpoint, transport }
    }

    /// Convert internal request to Google format.
    fn convert_request(&self, request: &LLMRequest) -> Result<GoogleRequest, Error> {
        let mut contents: Vec<GoogleContent> = Vec::new();
        let mut system_instruction = None;

        // Gemini's `functionCall` parts have no `id` field on the wire, so
        // we synthesize call_ids on the response side. To send results back
        // we have to recover the function name by call_id from the
        // conversation history. Build the mapping in a single pass.
        let mut call_id_to_name: std::collections::HashMap<&str, &str> =
            std::collections::HashMap::new();

        // Append a part to the last content with the same role, otherwise
        // start a new content. Vertex rejects consecutive same-role contents,
        // and this also matches how function_call / function_response
        // sequences need to fold into the surrounding model/user turns.
        fn push_part(contents: &mut Vec<GoogleContent>, role: &str, part: GooglePart) {
            if let Some(last) = contents.last_mut() {
                if last.role == role {
                    last.parts.push(part);
                    return;
                }
            }
            contents.push(GoogleContent {
                role: role.to_string(),
                parts: vec![part],
            });
        }

        // First pass: collect the assistant turns' tool_call name + id
        // pairs so we can echo them back on the corresponding user
        // ToolResult (Gemini's `functionResponse` part requires the
        // function name on the wire).
        for item in &request.messages {
            if let InputItem::Assistant { content } = item {
                for part in content {
                    if let AssistantPart::ToolCall(call) = part {
                        call_id_to_name.insert(call.call_id.as_str(), call.name.as_str());
                    }
                }
            }
        }

        for item in &request.messages {
            match item {
                InputItem::System(content) => {
                    system_instruction = Some(GoogleContent {
                        role: "system".to_string(),
                        parts: vec![GooglePart::Text {
                            text: content.clone(),
                        }],
                    });
                }
                InputItem::User { content } => {
                    for part in content {
                        match part {
                            UserPart::Text(s) => {
                                push_part(
                                    &mut contents,
                                    "user",
                                    GooglePart::Text { text: s.clone() },
                                );
                            }
                            UserPart::ToolResult { call_id, content } => {
                                let function_name = call_id_to_name
                                    .get(call_id.as_str())
                                    .ok_or_else(|| {
                                        Error::provider(
                                            "Google",
                                            format!(
                                                "ToolResult references unknown call_id \
                                                 {call_id:?} — no prior tool_call matches",
                                            ),
                                        )
                                    })?
                                    .to_string();
                                let output_text = flatten_user_parts_to_text(content);
                                push_part(
                                    &mut contents,
                                    "user",
                                    GooglePart::FunctionResponse {
                                        function_response: GoogleFunctionResponse {
                                            name: function_name,
                                            response: encode_function_output(&output_text),
                                        },
                                    },
                                );
                            }
                            UserPart::Image(src) => {
                                let part = match src {
                                    crate::types::ImageSource::Base64 { data, media_type } => {
                                        GooglePart::InlineData {
                                            inline_data: GoogleInlineData {
                                                mime_type: media_type.clone(),
                                                data: data.clone(),
                                            },
                                        }
                                    }
                                    crate::types::ImageSource::Url(u) => GooglePart::FileData {
                                        file_data: GoogleFileData {
                                            mime_type: "image/*".to_string(),
                                            file_uri: u.clone(),
                                        },
                                    },
                                };
                                push_part(&mut contents, "user", part);
                            }
                            UserPart::Audio(_)
                            | UserPart::Document(_)
                            | UserPart::CacheBreakpoint => {
                                tracing::debug!(
                                    "Google provider dropping unsupported user part"
                                );
                            }
                        }
                    }
                }
                InputItem::Assistant { content } => {
                    for part in content {
                        match part {
                            AssistantPart::Text { content, .. } => {
                                push_part(
                                    &mut contents,
                                    "model",
                                    GooglePart::Text {
                                        text: content.clone(),
                                    },
                                );
                            }
                            AssistantPart::Refusal(s) => {
                                push_part(
                                    &mut contents,
                                    "model",
                                    GooglePart::Text { text: s.clone() },
                                );
                            }
                            AssistantPart::ToolCall(call) => {
                                let args = serde_json::from_str(&call.arguments).map_err(|e| {
                                    Error::provider(
                                        "Google",
                                        format!("Invalid function arguments: {e}"),
                                    )
                                })?;
                                push_part(
                                    &mut contents,
                                    "model",
                                    GooglePart::FunctionCall {
                                        function_call: GoogleFunctionCall {
                                            name: call.name.clone(),
                                            args,
                                        },
                                    },
                                );
                            }
                            AssistantPart::Reasoning { .. }
                            | AssistantPart::RedactedReasoning { .. }
                            | AssistantPart::CacheBreakpoint => {
                                tracing::debug!(
                                    "Google provider dropping unsupported assistant part"
                                );
                            }
                        }
                    }
                }
            }
        }

        let thinking_config = request.reasoning.as_ref().map(|cfg| {
            let thinking_budget = match cfg.effort.unwrap_or(crate::types::ReasoningEffort::Medium) {
                crate::types::ReasoningEffort::Low => 2048,
                crate::types::ReasoningEffort::Medium => 8192,
                crate::types::ReasoningEffort::High => 16384,
            };
            GoogleThinkingConfig { thinking_budget }
        });

        let generation_config = Some(GoogleGenerationConfig {
            temperature: request.temperature,
            max_output_tokens: request.max_tokens,
            top_p: request.top_p,
            stop_sequences: request.stop.clone(),
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            thinking_config,
        });

        let tools = request.tools.as_ref().and_then(|tools| {
            use crate::types::{ProviderBuiltin, Tool};
            let mut function_decls: Vec<GoogleFunctionDeclaration> = Vec::new();
            let mut entries: Vec<GoogleTool> = Vec::new();
            for tool in tools {
                match tool {
                    Tool::Function(f) => function_decls.push(GoogleFunctionDeclaration {
                        name: f.name.clone(),
                        description: f.description.clone().unwrap_or_default(),
                        parameters: f.parameters.clone(),
                    }),
                    Tool::Builtin(ProviderBuiltin::GoogleSearch) => {
                        entries.push(GoogleTool::GoogleSearch {
                            google_search: GoogleEmptyConfig::default(),
                        });
                    }
                    Tool::Builtin(ProviderBuiltin::CodeExecution) => {
                        entries.push(GoogleTool::CodeExecution {
                            code_execution: GoogleEmptyConfig::default(),
                        });
                    }
                    Tool::Builtin(b) => {
                        tracing::debug!(?b, "Google provider dropping unsupported builtin tool");
                    }
                }
            }
            if !function_decls.is_empty() {
                entries.insert(
                    0,
                    GoogleTool::Functions {
                        function_declarations: function_decls,
                    },
                );
            }
            if entries.is_empty() {
                None
            } else {
                Some(entries)
            }
        });

        let tool_config =
            request
                .tool_choice
                .as_ref()
                .map(|choice| match choice {
                    crate::types::ToolChoice::Auto => GoogleToolConfig {
                        function_calling_config: GoogleFunctionCallingConfig {
                            mode: "AUTO",
                            allowed_function_names: None,
                        },
                    },
                    crate::types::ToolChoice::None => GoogleToolConfig {
                        function_calling_config: GoogleFunctionCallingConfig {
                            mode: "NONE",
                            allowed_function_names: None,
                        },
                    },
                    crate::types::ToolChoice::Required => GoogleToolConfig {
                        function_calling_config: GoogleFunctionCallingConfig {
                            mode: "ANY",
                            allowed_function_names: None,
                        },
                    },
                    crate::types::ToolChoice::Function { name } => GoogleToolConfig {
                        function_calling_config: GoogleFunctionCallingConfig {
                            mode: "ANY",
                            allowed_function_names: Some(vec![name.clone()]),
                        },
                    },
                });

        let google_request = GoogleRequest {
            contents,
            generation_config,
            tools,
            system_instruction,
            tool_config,
        };

        Ok(google_request)
    }

}

/// Flatten a tool-result's content array into a single string. Non-text
/// parts (images, etc.) aren't representable in Gemini's
/// `functionResponse.response` shape and are dropped with a debug note.
fn flatten_user_parts_to_text(parts: &[UserPart]) -> String {
    let mut out = String::new();
    for part in parts {
        match part {
            UserPart::Text(s) => {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(s);
            }
            _ => tracing::debug!("dropping non-text tool result part for Gemini flatten"),
        }
    }
    out
}

/// Shape a tool's output for Gemini's `functionResponse.response` field,
/// which the API requires to be a JSON object.
///
/// - JSON objects pass through unchanged so the model receives structured
///   data it can reason about.
/// - JSON non-objects (numbers, arrays, strings, bools, null) are wrapped
///   under `{"result": <value>}` so we still satisfy the object requirement
///   without losing structure.
/// - Non-JSON strings are wrapped under `{"result": "<string>"}`.
fn encode_function_output(output: &str) -> IValue {
    match serde_json::from_str::<IValue>(output) {
        Ok(value) if value.is_object() => value,
        Ok(value) => ijson!({ "result": value }),
        Err(_) => ijson!({ "result": output }),
    }
}

#[async_trait::async_trait]
impl LLMProvider for GoogleProvider {
    async fn generate(&self, request: &LLMRequest) -> Result<Response, Error> {
        let google_request = self.convert_request(request)?;

        let url = self.endpoint.url(
            "google",
            &request.model,
            "streamGenerateContent",
            Some("alt=sse"),
        );

        let body = serde_json::to_vec(&google_request)?;
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
            // Vertex 4xx envelopes carry `"status": "UNAUTHENTICATED"` /
            // `"NOT_FOUND"` — map them onto our typed variants where the
            // mapping is clear.
            return Err(match status {
                401 | 403 => {
                    Error::auth_with_status(status, format!("Google {status}: {body_text}"))
                }
                404 => Error::ModelNotAvailable(format!("Google 404: {body_text}")),
                _ => Error::provider_with_status(
                    "Google",
                    status,
                    format!("API error: {body_text}"),
                ),
            });
        }

        // Create SSE stream from response (Gemini supports ?alt=sse)
        let sse_stream = SseStream::new(response.body);

        // Create a stateful processor for tracking output items
        let mut state = GoogleStreamState::default();

        let event_stream = sse_stream
            .map(move |sse_result| {
                match sse_result {
                    Ok(sse_event) => {
                        let data = sse_event.data.trim();

                        // Vertex's SSE channel terminates by stream close;
                        // there is no `[DONE]` sentinel (that is an OpenAI
                        // convention). Empty events do still occur for
                        // keep-alives.
                        if data.is_empty() {
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

/// Stream state for Gemini's `streamGenerateContent`. Gemini emits each
/// chunk's parts in a `parts` array — consecutive text parts continue
/// the same logical text span, and `functionCall` parts arrive
/// already-complete. The state tracks which part-index is open and
/// closes it when the part changes or the stream ends.
#[derive(Debug, Default)]
pub(crate) struct GoogleStreamState {
    /// Synthetic counter for next-part-index allocation.
    next_index: u32,
    /// Index of the currently-open text part, if any.
    text_part: Option<u32>,
}

impl GoogleStreamState {
    fn open_text(&mut self, out: &mut Vec<StreamEvent>) -> u32 {
        if let Some(idx) = self.text_part {
            return idx;
        }
        let idx = self.next_index;
        self.next_index += 1;
        self.text_part = Some(idx);
        out.push(StreamEvent::PartStart {
            index: idx,
            kind: PartKind::Text,
        });
        idx
    }

    fn close_text(&mut self, out: &mut Vec<StreamEvent>) {
        if let Some(idx) = self.text_part.take() {
            out.push(StreamEvent::PartEnd { index: idx });
        }
    }

    fn open_close_tool_call(
        &mut self,
        out: &mut Vec<StreamEvent>,
        call_id: String,
        name: String,
        arguments: String,
    ) {
        let idx = self.next_index;
        self.next_index += 1;
        out.push(StreamEvent::PartStart {
            index: idx,
            kind: PartKind::ToolCall {
                call_id,
                name: name.clone(),
            },
        });
        if !arguments.is_empty() {
            out.push(StreamEvent::Delta {
                index: idx,
                delta: arguments,
            });
        }
        out.push(StreamEvent::PartEnd { index: idx });
    }
}

/// Stateful per-chunk conversion. `pub(crate)` so unit tests can drive
/// synthetic `GoogleResponse` values directly.
pub(crate) fn convert_response_stateful(
    response: GoogleResponse,
    state: &mut GoogleStreamState,
) -> Result<Vec<StreamEvent>, Error> {
    let mut events = Vec::new();

    if let Some(candidate) = response.candidates.first() {
        for part in &candidate.content.parts {
            match part {
                GooglePart::Text { text } => {
                    if text.is_empty() {
                        continue;
                    }
                    let idx = state.open_text(&mut events);
                    events.push(StreamEvent::Delta {
                        index: idx,
                        delta: text.clone(),
                    });
                }
                GooglePart::FunctionCall { function_call } => {
                    // Close any open text part before starting a tool call.
                    state.close_text(&mut events);
                    let base_id = Uuid::new_v4().simple().to_string();
                    let call_id = format!("call_{base_id}");
                    let arguments = serde_json::to_string(&function_call.args).map_err(|e| {
                        Error::provider("Google", format!("Failed to serialize function args: {e}"))
                    })?;
                    state.open_close_tool_call(
                        &mut events,
                        call_id,
                        function_call.name.clone(),
                        arguments,
                    );
                }
                GooglePart::FunctionResponse { .. }
                | GooglePart::InlineData { .. }
                | GooglePart::FileData { .. } => {
                    // Request-side parts; not expected on response stream.
                }
            }
        }

        // Only add a Done event if this response has a finish_reason (indicates end of stream)
        if let Some(finish_reason_str) = &candidate.finish_reason {
            // Close any still-open text part before emitting Done.
            state.close_text(&mut events);

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

    #[test]
    fn convert_simple_text_request() {
        let provider = GoogleProvider::new(
            "p".to_string(),
            "us-east1".to_string(),
            "tok".to_string(),
        )
        .unwrap();
        let req = LLMRequest::from_prompt("gemini", &crate::Prompt::user("hi"));
        let body = provider.convert_request(&req).unwrap();
        assert_eq!(body.contents.len(), 1);
        assert_eq!(body.contents[0].role, "user");
    }

    #[test]
    fn system_instruction_role_is_not_user() {
        let provider = GoogleProvider::new(
            "p".to_string(),
            "us-east1".to_string(),
            "tok".to_string(),
        )
        .unwrap();
        let req = LLMRequest::from_prompt(
            "gemini",
            &crate::Prompt::system("you are helpful").with_user("hi"),
        );
        let body = provider.convert_request(&req).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        let role = json["systemInstruction"]["role"].as_str();
        assert!(
            role != Some("user"),
            "systemInstruction must not carry role: 'user' (got {role:?})",
        );
    }

    #[tokio::test]
    async fn streaming_text_yields_partstart_delta_partend() {
        let chunk1 = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}]}"#;
        let chunk2 = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":" world"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}"#;
        let mut state = GoogleStreamState::default();
        let r1: GoogleResponse = serde_json::from_str(chunk1).unwrap();
        let r2: GoogleResponse = serde_json::from_str(chunk2).unwrap();
        let events: Vec<StreamEvent> = convert_response_stateful(r1, &mut state)
            .unwrap()
            .into_iter()
            .chain(convert_response_stateful(r2, &mut state).unwrap())
            .collect();
        // We expect: PartStart Text, Delta "Hello", Delta " world", PartEnd, Done
        assert!(matches!(events[0], StreamEvent::PartStart { kind: PartKind::Text, .. }));
        assert!(matches!(events[1], StreamEvent::Delta { .. }));
        assert!(matches!(events.last(), Some(StreamEvent::Done { .. })));
    }

    fn provider() -> GoogleProvider {
        GoogleProvider::new(
            "p".to_string(),
            "us-east1".to_string(),
            "tok".to_string(),
        )
        .unwrap()
    }

    #[test]
    fn stop_sequences_threaded_through_request() {
        let req = LLMRequest::from_prompt("gemini", &crate::Prompt::user("hi"))
            .stop(vec!["END".to_string(), "STOP".to_string()]);
        let body = provider().convert_request(&req).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["generationConfig"]["stopSequences"],
            serde_json::json!(["END", "STOP"]),
        );
    }

    #[test]
    fn presence_and_frequency_penalty_threaded_through() {
        let req = LLMRequest::from_prompt("gemini", &crate::Prompt::user("hi"))
            .presence_penalty(0.5)
            .frequency_penalty(0.25);
        let body = provider().convert_request(&req).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        // Use exact float values (0.5, 0.25) that survive f32 → JSON without
        // representation drift.
        assert_eq!(json["generationConfig"]["presencePenalty"], 0.5);
        assert_eq!(json["generationConfig"]["frequencyPenalty"], 0.25);
    }

    #[test]
    fn reasoning_config_emits_thinking_budget() {
        use crate::types::{ReasoningConfig, ReasoningEffort};
        let req = LLMRequest::from_prompt("gemini-2.5-flash", &crate::Prompt::user("hi"))
            .reasoning(ReasoningConfig {
                effort: Some(ReasoningEffort::High),
                summary: None,
            });
        let body = provider().convert_request(&req).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            16384,
        );
    }

    #[test]
    fn tool_choice_required_maps_to_any_mode() {
        use crate::types::ToolChoice;
        let req = LLMRequest::from_prompt("gemini", &crate::Prompt::user("hi"))
            .tool_choice(ToolChoice::Required);
        let body = provider().convert_request(&req).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["toolConfig"]["functionCallingConfig"]["mode"],
            "ANY",
        );
    }

    #[test]
    fn tool_choice_function_restricts_allowed_names() {
        use crate::types::ToolChoice;
        let req = LLMRequest::from_prompt("gemini", &crate::Prompt::user("hi"))
            .tool_choice(ToolChoice::Function {
                name: "get_weather".to_string(),
            });
        let body = provider().convert_request(&req).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"],
            serde_json::json!(["get_weather"]),
        );
    }

    #[test]
    fn google_search_builtin_emits_separate_tool_entry() {
        use crate::types::{ProviderBuiltin, Tool};
        let req = LLMRequest::from_prompt("gemini", &crate::Prompt::user("hi"))
            .tools(vec![Tool::builtin(ProviderBuiltin::GoogleSearch)]);
        let body = provider().convert_request(&req).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["tools"], serde_json::json!([{ "googleSearch": {} }]));
    }

    #[test]
    fn code_execution_builtin_emits_separate_tool_entry() {
        use crate::types::{ProviderBuiltin, Tool};
        let req = LLMRequest::from_prompt("gemini", &crate::Prompt::user("hi"))
            .tools(vec![Tool::builtin(ProviderBuiltin::CodeExecution)]);
        let body = provider().convert_request(&req).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["tools"], serde_json::json!([{ "codeExecution": {} }]));
    }
}
