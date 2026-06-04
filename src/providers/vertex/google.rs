use futures_util::StreamExt;
use ijson::{ijson, IValue};
use uuid::Uuid;

use super::endpoint::VertexEndpoint;
use super::google_types::*;
use crate::provider::Provider;
use crate::sse_stream::SseStream;
use crate::transport::{Transport, TransportRequest};
use crate::types::{
    Annotation, AnnotationKind, AssistantPart, FinishReason, InputItem, PartKind, PartUpdate,
    UserPart,
};
use crate::{Error, RawConfig, Response, StreamEvent};

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
        Self {
            endpoint,
            transport,
        }
    }

    /// Swap the static access token before it expires (GCP tokens
    /// last ~1h). Errors if this provider was built with ADC, which
    /// refreshes automatically. See [`VertexEndpoint::set_access_token`].
    pub fn set_access_token(&self, token: impl Into<String>) -> Result<(), Error> {
        self.endpoint.set_access_token(token)
    }

    /// Convert internal request to Google format.
    fn convert_request(
        &self,
        prompt: &crate::Prompt,
        config: &RawConfig,
    ) -> Result<GoogleRequest, Error> {
        let messages = prompt.items();

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

        // Scan history for the latest Gemini continuation. Items at
        // and before its index are elided — the server already has them
        // via the referenced `cachedContent` resource. Continuations
        // for other providers are ignored.
        let (cached_content, start_index) = find_latest_gemini_continuation(messages);
        let active_messages = &messages[start_index..];

        // First pass: collect the assistant turns' tool_call name + id
        // pairs so we can echo them back on the corresponding user
        // ToolResult (Gemini's `functionResponse` part requires the
        // function name on the wire). Scan the *full* history, not
        // just `active_messages`: a continuation can elide the
        // assistant turn that made the call while the first
        // non-elided user item is its ToolResult — the name still
        // has to resolve.
        for item in messages {
            if let InputItem::Assistant { content } = item {
                for part in content {
                    if let AssistantPart::ToolCall(call) = part {
                        call_id_to_name.insert(call.call_id.as_str(), call.name.as_str());
                    }
                }
            }
        }

        for item in active_messages {
            match item {
                InputItem::System(content) => {
                    // `role: "system"` here is confirmed accepted by
                    // the live Vertex API — see the captured real
                    // exchange in
                    // tests/cross_provider/traces/google/system_and_user.*
                    // (request sends this shape; response is a valid
                    // 200). Don't "fix" to drop the role without a
                    // fresh capture proving it's required.
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
                                // No matching tool_call anywhere in
                                // history (e.g. the originating call
                                // was a provider-builtin dropped on a
                                // cross-provider switch). Skip the
                                // orphaned result with a warning rather
                                // than hard-erroring the whole request
                                // — matches the model-switching "drop
                                // what doesn't translate" contract.
                                let Some(function_name) =
                                    call_id_to_name.get(call_id.as_str()).map(|s| s.to_string())
                                else {
                                    tracing::warn!(
                                        call_id = %call_id,
                                        "Gemini: dropping ToolResult with no matching \
                                         tool_call (orphaned across model switch?)",
                                    );
                                    continue;
                                };
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
                            UserPart::Audio(src) => {
                                let part = match src {
                                    crate::types::AudioSource::Base64 { data, media_type } => {
                                        GooglePart::InlineData {
                                            inline_data: GoogleInlineData {
                                                mime_type: media_type.clone(),
                                                data: data.clone(),
                                            },
                                        }
                                    }
                                    crate::types::AudioSource::Url(u) => GooglePart::FileData {
                                        file_data: GoogleFileData {
                                            mime_type: "audio/*".to_string(),
                                            file_uri: u.clone(),
                                        },
                                    },
                                };
                                push_part(&mut contents, "user", part);
                            }
                            UserPart::Document(src) => {
                                let part = match src {
                                    crate::types::DocumentSource::Base64 { data, media_type } => {
                                        GooglePart::InlineData {
                                            inline_data: GoogleInlineData {
                                                mime_type: media_type.clone(),
                                                data: data.clone(),
                                            },
                                        }
                                    }
                                    crate::types::DocumentSource::Url(u) => GooglePart::FileData {
                                        file_data: GoogleFileData {
                                            mime_type: "application/pdf".to_string(),
                                            file_uri: u.clone(),
                                        },
                                    },
                                };
                                push_part(&mut contents, "user", part);
                            }
                            UserPart::CacheBreakpoint => {
                                // Gemini uses a separate cachedContent
                                // API surface; not wired yet.
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
                            | AssistantPart::BuiltinToolCall { .. }
                            | AssistantPart::Continuation(_)
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

        let thinking_config = config.reasoning.as_ref().map(|cfg| {
            let thinking_budget = match cfg.effort.unwrap_or(crate::types::ReasoningEffort::Medium)
            {
                crate::types::ReasoningEffort::Low => 2048,
                crate::types::ReasoningEffort::Medium => 8192,
                crate::types::ReasoningEffort::High => 16384,
            };
            GoogleThinkingConfig { thinking_budget }
        });

        let (response_mime_type, response_schema) = match &config.response_format {
            Some(crate::types::ResponseFormat::JsonObject) => {
                (Some("application/json".to_string()), None)
            }
            Some(crate::types::ResponseFormat::JsonSchema { schema, .. }) => {
                (Some("application/json".to_string()), Some(schema.clone()))
            }
            // ResponseFormat::Text or None — leave unset.
            Some(crate::types::ResponseFormat::Text) | None => (None, None),
        };

        let generation_config = Some(GoogleGenerationConfig {
            temperature: config.temperature,
            max_output_tokens: config.max_tokens,
            top_p: config.top_p,
            stop_sequences: config.stop.clone(),
            presence_penalty: config.presence_penalty,
            frequency_penalty: config.frequency_penalty,
            thinking_config,
            response_mime_type,
            response_schema,
        });

        let tools = config.tools.as_ref().and_then(|tools| {
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

        let tool_config = config.tool_choice.as_ref().map(|choice| match choice {
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

        // `cached_content` was extracted up-front from the message
        // history; nothing more to do here.

        let google_request = GoogleRequest {
            contents,
            generation_config,
            tools,
            system_instruction,
            tool_config,
            cached_content,
        };

        Ok(google_request)
    }
}

use crate::providers::flatten_user_parts_to_text;

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
impl Provider for GoogleProvider {
    async fn generate(
        &self,
        prompt: &crate::Prompt,
        config: &RawConfig,
    ) -> Result<Response, Error> {
        let google_request = self.convert_request(prompt, config)?;

        let url = self.endpoint.url(
            "google",
            &config.model,
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
            // Read Retry-After before `collect_body` consumes the response.
            let retry_after = crate::transport::parse_retry_after(response.header("retry-after"));
            let body_bytes = response.collect_body().await.unwrap_or_default();
            let body_text = String::from_utf8_lossy(&body_bytes);
            // Vertex 4xx envelopes carry `"status": "UNAUTHENTICATED"` /
            // `"NOT_FOUND"` / `"RESOURCE_EXHAUSTED"` — map them onto our
            // typed variants where the mapping is clear.
            return Err(match status {
                401 | 403 => {
                    Error::auth_with_status(status, format!("Google {status}: {body_text}"))
                }
                404 => Error::ModelNotAvailable(format!("Google 404: {body_text}")),
                429 => Error::rate_limit(
                    retry_after,
                    format!("Google 429 (RESOURCE_EXHAUSTED): {body_text}"),
                ),
                _ => {
                    Error::provider_with_status("Google", status, format!("API error: {body_text}"))
                }
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

/// Walk the history right-to-left for the most recent
/// [`InputItem::Assistant`] containing an
/// [`AssistantPart::Continuation`] of
/// [`crate::types::ProviderContinuation::Gemini`]. Returns the cached-
/// content resource name plus the index of the first item the provider
/// should send (one past the assistant turn — the server has it via
/// the cached content). Non-Gemini continuation parts are transparently
/// skipped.
fn find_latest_gemini_continuation(
    messages: &[crate::types::InputItem],
) -> (Option<String>, usize) {
    use crate::types::{AssistantPart, InputItem, ProviderContinuation};
    for (i, item) in messages.iter().enumerate().rev() {
        if let InputItem::Assistant { content } = item {
            for part in content.iter().rev() {
                if let AssistantPart::Continuation(ProviderContinuation::Gemini {
                    cached_content,
                }) = part
                {
                    return (Some(cached_content.clone()), i + 1);
                }
            }
        }
    }
    (None, 0)
}

/// Convert Gemini's batched `groundingMetadata` payload into one or
/// more flat [`Annotation`]s. Each `groundingSupport` (span) yields one
/// annotation per cited chunk, so a span that draws from N sources
/// surfaces as N URL citations covering the same byte range.
fn flatten_grounding_metadata(meta: &GoogleGroundingMetadata) -> Vec<Annotation> {
    let mut out = Vec::new();
    for support in &meta.grounding_supports {
        for &chunk_idx in &support.grounding_chunk_indices {
            let Some(chunk) = meta.grounding_chunks.get(chunk_idx as usize) else {
                continue;
            };
            let Some(web) = &chunk.web else {
                continue;
            };
            out.push(Annotation {
                kind: AnnotationKind::UrlCitation,
                start: support.segment.start_index,
                end: support.segment.end_index,
                source: web.uri.clone(),
                title: web.title.clone(),
            });
        }
    }
    out
}

/// Slot key for [`GoogleStreamState::tracker`]. Gemini doesn't carry
/// part identifiers on the wire (parts are anonymous entries in the
/// `parts` array), so the lib uses fixed slots for the two
/// long-running part kinds (text spans and code-execution call/result
/// pairs) plus a fresh-key namespace for one-shot tool calls.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum GoogleSlot {
    /// Open text span; multiple consecutive `Text` parts on the wire
    /// continue the same logical part until a non-text emission closes
    /// it.
    Text,
    /// Open `BuiltinToolCall(CodeExecution)` part. Gemini emits
    /// `executableCode` and `codeExecutionResult` as sibling parts;
    /// the slot keeps the call open until the result lands.
    CodeExecution,
}

/// Stream state for Gemini's `streamGenerateContent`. Single
/// [`PartTracker`] manages all part-index allocation so there's no
/// chance of drift between separately-tracked counters.
#[derive(Debug)]
pub(crate) struct GoogleStreamState {
    tracker: crate::providers::part_tracker::PartTracker<GoogleSlot>,
    /// Index of the most recently opened text part, retained even
    /// after it's closed. Gemini batches grounding metadata onto the
    /// final chunk; if a function call / code execution interleaved
    /// and closed the text part, the citation target would otherwise
    /// be lost (`index_of(Text)` is `None` at finish).
    last_text_index: Option<u32>,
}

impl Default for GoogleStreamState {
    fn default() -> Self {
        Self {
            tracker: crate::providers::part_tracker::PartTracker::new(),
            last_text_index: None,
        }
    }
}

impl GoogleStreamState {
    fn open_text(&mut self, out: &mut Vec<StreamEvent>) -> u32 {
        if let Some(idx) = self.tracker.index_of(&GoogleSlot::Text) {
            return idx;
        }
        let (idx, ev) = self.tracker.open(GoogleSlot::Text, PartKind::Text);
        self.last_text_index = Some(idx);
        out.push(ev);
        idx
    }

    fn close_text(&mut self, out: &mut Vec<StreamEvent>) {
        if let Some(ev) = self.tracker.close(&GoogleSlot::Text) {
            out.push(ev);
        }
    }

    fn open_code_execution(&mut self, out: &mut Vec<StreamEvent>) -> u32 {
        let (idx, ev) = self.tracker.open(
            GoogleSlot::CodeExecution,
            PartKind::BuiltinToolCall {
                kind: crate::types::ProviderBuiltin::CodeExecution,
            },
        );
        out.push(ev);
        idx
    }

    fn code_execution_index(&self) -> Option<u32> {
        self.tracker.index_of(&GoogleSlot::CodeExecution)
    }

    fn close_code_execution(&mut self, out: &mut Vec<StreamEvent>) {
        if let Some(ev) = self.tracker.close(&GoogleSlot::CodeExecution) {
            out.push(ev);
        }
    }

    fn open_close_tool_call(
        &mut self,
        out: &mut Vec<StreamEvent>,
        call_id: String,
        name: String,
        mut arguments: String,
    ) {
        let events = self.tracker.open_one_shot(PartKind::ToolCall {
            call_id,
            name: name.clone(),
        });
        // open_one_shot emits PartStart then PartEnd; splice the
        // arguments Delta in right after the PartStart. Forward each
        // event and inject once, without depending (via panicking
        // `expect`/`unreachable!`) on the exact event count — a
        // future PartTracker change shouldn't crash the stream.
        for ev in events {
            if let StreamEvent::PartStart { index, .. } = &ev {
                let index = *index;
                out.push(ev);
                if !arguments.is_empty() {
                    out.push(StreamEvent::Delta {
                        index,
                        delta: std::mem::take(&mut arguments),
                    });
                }
            } else {
                out.push(ev);
            }
        }
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
                    // Text following a code-execution call ends the
                    // call's lifecycle; close it before opening text.
                    state.close_code_execution(&mut events);
                    let idx = state.open_text(&mut events);
                    events.push(StreamEvent::Delta {
                        index: idx,
                        delta: text.clone(),
                    });
                }
                GooglePart::FunctionCall { function_call } => {
                    // Close any open text part before starting a tool call.
                    state.close_text(&mut events);
                    state.close_code_execution(&mut events);
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
                GooglePart::ExecutableCode { executable_code } => {
                    // `executableCode` opens a CodeExecution
                    // BuiltinToolCall; the matching
                    // `codeExecutionResult` (if any) populates its
                    // `result` via PartUpdate before we close.
                    state.close_text(&mut events);
                    state.close_code_execution(&mut events);
                    let idx = state.open_code_execution(&mut events);
                    let arguments = serde_json::json!({
                        "language": executable_code.language,
                        "code": executable_code.code,
                    })
                    .to_string();
                    events.push(StreamEvent::Delta {
                        index: idx,
                        delta: arguments,
                    });
                }
                GooglePart::CodeExecutionResult {
                    code_execution_result,
                } => {
                    let result = serde_json::json!({
                        "outcome": code_execution_result.outcome,
                        "output": code_execution_result.output,
                    })
                    .to_string();
                    let idx = match state.code_execution_index() {
                        Some(idx) => idx,
                        // Unpaired result — open a synthetic part so
                        // the data isn't silently lost.
                        None => state.open_code_execution(&mut events),
                    };
                    events.push(StreamEvent::PartUpdate {
                        index: idx,
                        update: PartUpdate::BuiltinToolResult(result),
                    });
                    state.close_code_execution(&mut events);
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
            // Flush grounding annotations onto the open text part before
            // closing it. Gemini batches grounding metadata on the final
            // chunk; we replay each support as a PartUpdate so the
            // accumulator can attach citations to AssistantPart::Text.
            // Prefer the still-open text part; fall back to the last
            // text part we opened (a tool call may have closed it
            // before the grounding-bearing final chunk arrived).
            let text_idx = state
                .tracker
                .index_of(&GoogleSlot::Text)
                .or(state.last_text_index);
            if let (Some(text_idx), Some(meta)) = (text_idx, &candidate.grounding_metadata) {
                for annotation in flatten_grounding_metadata(meta) {
                    events.push(StreamEvent::PartUpdate {
                        index: text_idx,
                        update: PartUpdate::Annotation(annotation),
                    });
                }
            }

            // Close any still-open text part before emitting Done.
            state.close_text(&mut events);
            state.close_code_execution(&mut events);

            let finish_reason = match finish_reason_str.as_str() {
                "STOP" => FinishReason::Stop,
                "MAX_TOKENS" => FinishReason::Length,
                // All of these mean "the model declined / output was
                // suppressed", not a clean stop — surfacing them as
                // Stop would let callers treat a censored or truncated
                // answer as complete.
                "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII"
                | "IMAGE_SAFETY" => FinishReason::ContentFilter,
                other => {
                    tracing::warn!(
                        finish_reason = other,
                        "Gemini: unknown candidate finishReason; treating as Incomplete",
                    );
                    FinishReason::Incomplete
                }
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
    use crate::types::Config;

    #[test]
    fn convert_simple_text_request() {
        let provider =
            GoogleProvider::new("p".to_string(), "us-east1".to_string(), "tok".to_string())
                .unwrap();
        let prompt = crate::Prompt::user("hi");
        let cfg = Config::builder("gemini").build();
        let body = provider.convert_request(&prompt, cfg.raw()).unwrap();
        assert_eq!(body.contents.len(), 1);
        assert_eq!(body.contents[0].role, "user");
    }

    #[test]
    fn system_instruction_role_is_not_user() {
        let provider =
            GoogleProvider::new("p".to_string(), "us-east1".to_string(), "tok".to_string())
                .unwrap();
        let prompt = crate::Prompt::system("you are helpful").with_user("hi");
        let cfg = Config::builder("gemini").build();
        let body = provider.convert_request(&prompt, cfg.raw()).unwrap();
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
        assert!(matches!(
            events[0],
            StreamEvent::PartStart {
                kind: PartKind::Text,
                ..
            }
        ));
        assert!(matches!(events[1], StreamEvent::Delta { .. }));
        assert!(matches!(events.last(), Some(StreamEvent::Done { .. })));
    }

    fn provider() -> GoogleProvider {
        GoogleProvider::new("p".to_string(), "us-east1".to_string(), "tok".to_string()).unwrap()
    }

    #[test]
    fn stop_sequences_threaded_through_request() {
        let prompt = crate::Prompt::user("hi");
        let cfg = Config::builder("gemini")
            .stop(vec!["END".to_string(), "STOP".to_string()])
            .build();
        let body = provider().convert_request(&prompt, cfg.raw()).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["generationConfig"]["stopSequences"],
            serde_json::json!(["END", "STOP"]),
        );
    }

    #[test]
    fn presence_and_frequency_penalty_threaded_through() {
        let prompt = crate::Prompt::user("hi");
        let cfg = Config::builder("gemini")
            .presence_penalty(0.5)
            .frequency_penalty(0.25)
            .build();
        let body = provider().convert_request(&prompt, cfg.raw()).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        // Use exact float values (0.5, 0.25) that survive f32 → JSON without
        // representation drift.
        assert_eq!(json["generationConfig"]["presencePenalty"], 0.5);
        assert_eq!(json["generationConfig"]["frequencyPenalty"], 0.25);
    }

    #[test]
    fn reasoning_config_emits_thinking_budget() {
        use crate::types::{ReasoningConfig, ReasoningEffort};
        let prompt = crate::Prompt::user("hi");
        let cfg = Config::builder("gemini-2.5-flash")
            .reasoning(ReasoningConfig {
                effort: Some(ReasoningEffort::High),
                summary: None,
            })
            .build();
        let body = provider().convert_request(&prompt, cfg.raw()).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            16384,
        );
    }

    #[test]
    fn tool_choice_required_maps_to_any_mode() {
        use crate::types::ToolChoice;
        let prompt = crate::Prompt::user("hi");
        let cfg = Config::builder("gemini")
            .tool_choice(ToolChoice::Required)
            .build();
        let body = provider().convert_request(&prompt, cfg.raw()).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["toolConfig"]["functionCallingConfig"]["mode"], "ANY",);
    }

    #[test]
    fn tool_choice_function_restricts_allowed_names() {
        use crate::types::ToolChoice;
        let prompt = crate::Prompt::user("hi");
        let cfg = Config::builder("gemini")
            .tool_choice(ToolChoice::Function {
                name: "get_weather".to_string(),
            })
            .build();
        let body = provider().convert_request(&prompt, cfg.raw()).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"],
            serde_json::json!(["get_weather"]),
        );
    }

    #[test]
    fn google_search_builtin_emits_separate_tool_entry() {
        use crate::types::{ProviderBuiltin, Tool};
        let prompt = crate::Prompt::user("hi");
        let cfg = Config::builder("gemini")
            .tools(vec![Tool::builtin(ProviderBuiltin::GoogleSearch)])
            .build();
        let body = provider().convert_request(&prompt, cfg.raw()).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["tools"], serde_json::json!([{ "googleSearch": {} }]));
    }

    #[test]
    fn code_execution_builtin_emits_separate_tool_entry() {
        use crate::types::{ProviderBuiltin, Tool};
        let prompt = crate::Prompt::user("hi");
        let cfg = Config::builder("gemini")
            .tools(vec![Tool::builtin(ProviderBuiltin::CodeExecution)])
            .build();
        let body = provider().convert_request(&prompt, cfg.raw()).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["tools"], serde_json::json!([{ "codeExecution": {} }]));
    }

    #[test]
    fn cached_content_continuation_threaded_through_request() {
        use crate::types::{InputItem, ProviderContinuation};
        let prompt = crate::Prompt::user("hi").with_item(InputItem::assistant_continuation(
            ProviderContinuation::Gemini {
                cached_content: "projects/p/locations/l/cachedContents/abc".to_string(),
            },
        ));
        let cfg = Config::builder("gemini").build();
        let body = provider().convert_request(&prompt, cfg.raw()).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["cachedContent"],
            "projects/p/locations/l/cachedContents/abc",
        );
    }

    #[test]
    fn response_format_json_object_sets_mime_type() {
        use crate::types::ResponseFormat;
        let prompt = crate::Prompt::user("hi");
        let cfg = Config::builder("gemini")
            .response_format(ResponseFormat::JsonObject)
            .build();
        let body = provider().convert_request(&prompt, cfg.raw()).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["generationConfig"]["responseMimeType"],
            "application/json",
        );
        assert!(json["generationConfig"]
            .get("responseSchema")
            .map(|v| v.is_null())
            .unwrap_or(true));
    }

    #[test]
    fn response_format_json_schema_emits_schema() {
        use crate::types::ResponseFormat;
        use std::borrow::Cow;
        let schema_raw = serde_json::value::RawValue::from_string(
            r#"{"type":"object","properties":{"x":{"type":"number"}}}"#.to_string(),
        )
        .unwrap();
        let prompt = crate::Prompt::user("hi");
        let cfg = Config::builder("gemini")
            .response_format(ResponseFormat::JsonSchema {
                name: "Point".to_string(),
                schema: Cow::Owned(schema_raw),
                strict: true,
            })
            .build();
        let body = provider().convert_request(&prompt, cfg.raw()).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["generationConfig"]["responseMimeType"],
            "application/json",
        );
        assert_eq!(json["generationConfig"]["responseSchema"]["type"], "object");
    }

    /// An OpenAI continuation part is ignored by Gemini — the
    /// model-switching contract: hints from the wrong provider degrade
    /// silently to a full-history request.
    #[test]
    fn openai_continuation_ignored_by_gemini() {
        use crate::types::{InputItem, ProviderContinuation};
        let prompt = crate::Prompt::user("hi").with_item(InputItem::assistant_continuation(
            ProviderContinuation::OpenAI {
                response_id: "resp_abc".to_string(),
            },
        ));
        let cfg = Config::builder("gemini").build();
        let body = provider().convert_request(&prompt, cfg.raw()).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert!(json.get("cachedContent").is_none());
    }

    /// A Gemini continuation deep inside the history should elide all
    /// prior items: the request must contain only the items that
    /// follow the assistant turn carrying the marker.
    #[test]
    fn gemini_continuation_elides_prior_history() {
        use crate::types::{InputItem, ProviderContinuation};
        let prompt = crate::Prompt::user("first turn")
            .with_assistant("first answer")
            .with_item(InputItem::assistant_continuation(
                ProviderContinuation::Gemini {
                    cached_content: "cached/1".to_string(),
                },
            ))
            .with_user("follow-up");
        let cfg = Config::builder("gemini").build();
        let body = provider().convert_request(&prompt, cfg.raw()).unwrap();
        assert_eq!(body.contents.len(), 1);
        assert_eq!(body.contents[0].role, "user");
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["cachedContent"], "cached/1");
    }

    /// Full roundtrip: a Gemini `CompleteResponse` folded into the
    /// next prompt via `with_response()` should have its continuation
    /// picked up and prior history elided automatically.
    #[test]
    fn with_response_threads_continuation_into_next_request() {
        use crate::response::CompleteResponse;
        use crate::types::{AssistantPart, FinishReason, ProviderContinuation, Usage};
        let prior = CompleteResponse {
            content: vec![
                AssistantPart::Text {
                    content: "first answer".into(),
                    annotations: Vec::new(),
                },
                AssistantPart::Continuation(ProviderContinuation::Gemini {
                    cached_content: "cached/prior".into(),
                }),
            ],
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        };
        let prompt = crate::Prompt::user("first turn")
            .with_response(&prior)
            .with_user("follow-up");
        let cfg = Config::builder("gemini").build();
        let body = provider().convert_request(&prompt, cfg.raw()).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["cachedContent"], "cached/prior");
        assert_eq!(body.contents.len(), 1);
        if let GooglePart::Text { text } = &body.contents[0].parts[0] {
            assert_eq!(text, "follow-up");
        } else {
            panic!("expected text part, got {:?}", body.contents[0].parts[0]);
        }
    }

    /// The *most recent* matching continuation wins. Older markers of
    /// the same type are superseded — only history after the latest
    /// one is sent.
    #[test]
    fn latest_gemini_continuation_wins() {
        use crate::types::{InputItem, ProviderContinuation};
        let prompt = crate::Prompt::user("a")
            .with_item(InputItem::assistant_continuation(
                ProviderContinuation::Gemini {
                    cached_content: "cached/old".to_string(),
                },
            ))
            .with_user("b")
            .with_item(InputItem::assistant_continuation(
                ProviderContinuation::Gemini {
                    cached_content: "cached/new".to_string(),
                },
            ))
            .with_user("c");
        let cfg = Config::builder("gemini").build();
        let body = provider().convert_request(&prompt, cfg.raw()).unwrap();
        assert_eq!(body.contents.len(), 1);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["cachedContent"], "cached/new");
        // Only the items after `cached/new` are sent.
        assert_eq!(body.contents[0].parts.len(), 1);
        if let GooglePart::Text { text } = &body.contents[0].parts[0] {
            assert_eq!(text, "c");
        } else {
            panic!("expected text part, got {:?}", body.contents[0].parts[0]);
        }
    }
}
