use futures_util::StreamExt;
use ijson::{ijson, IValue};
use serde_json::value::RawValue;
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

        // Gemini enforces that every model turn with K `functionCall`
        // parts is immediately followed by a user turn with exactly K
        // `functionResponse` parts; an unmatched pairing surfaces only
        // as an opaque 400 at the HTTP layer. Catch it here, before the
        // round trip, with a typed error that points at the offending
        // turn.
        validate_gemini_tool_pairing(active_messages)?;

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
                                            thought_signature: call.provider_signature.clone(),
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
                        // Gemini's `functionDeclarations[].parameters` accepts
                        // only the property keywords of JSON Schema — it
                        // rejects the meta-fields `$schema`, `$ref`, and
                        // `$defs` with a 400. Normalise before sending:
                        // drop the meta-fields and inline any `$ref`s against
                        // the schema's `$defs` / `definitions`.
                        parameters: normalize_gemini_tool_schema(&f.parameters),
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

        // Gemini splits the system prompt into a top-level
        // `system_instruction` and requires at least one `contents`
        // entry (a user or model turn). A prompt that carries only
        // `System` items — or only items that don't translate to a
        // Gemini turn — leaves `contents` empty and gets a 400 ("at
        // least one contents field is required"). Surface it as a typed
        // error before the round trip. A `cached_content` continuation
        // references prior turns server-side, so empty `contents`
        // alongside one is left to the API rather than rejected here.
        if contents.is_empty() && cached_content.is_none() {
            return Err(Error::invalid_prompt(
                "Gemini requires at least one user or assistant turn; the prompt contained \
                 only system or non-content items",
            ));
        }

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

/// Validate that the prompt satisfies Gemini's function call / response
/// pairing rule: each model turn with K `functionCall` parts must be
/// immediately followed by a user turn with K `functionResponse` parts.
///
/// Runs on the items actually being sent (post-continuation elision), so
/// a continuation that elides the assistant turn carrying a call — while
/// keeping its result — doesn't false-positive (a lone user `ToolResult`
/// with no preceding assistant turn isn't flagged). Only the
/// forward direction (assistant calls → matching results) is checked;
/// orphaned results are handled separately by the request builder, which
/// drops them with a warning per the model-switching contract.
fn validate_gemini_tool_pairing(messages: &[crate::types::InputItem]) -> Result<(), Error> {
    for (i, item) in messages.iter().enumerate() {
        let InputItem::Assistant { content } = item else {
            continue;
        };
        let call_count = content
            .iter()
            .filter(|p| matches!(p, AssistantPart::ToolCall(_)))
            .count();
        if call_count == 0 {
            continue;
        }
        let result_count = match messages.get(i + 1) {
            Some(InputItem::User { content }) => content
                .iter()
                .filter(|p| matches!(p, UserPart::ToolResult { .. }))
                .count(),
            _ => 0,
        };
        if result_count != call_count {
            return Err(Error::invalid_prompt(format!(
                "Gemini requires each assistant turn's function_call count to match the \
                 following user turn's function_response count: assistant turn at index {i} \
                 has {call_count} tool call(s) but the following turn has {result_count} tool \
                 result(s)"
            )));
        }
    }
    Ok(())
}

/// Normalise a function tool's JSON-Schema `parameters` into the subset
/// Gemini's `functionDeclarations[].parameters` accepts. Gemini takes
/// only the property keywords of JSON Schema and rejects the meta-fields
/// `$schema`, `$ref`, and `$defs` with a 400, so we:
///
/// - drop root/nested meta-fields (`$schema`, `$id`, `$comment`,
///   `$anchor`, `$defs`, `definitions`), and
/// - inline every local `$ref` (`#/$defs/Name` or `#/definitions/Name`)
///   against the collected definitions, merging any sibling keywords
///   over the resolved definition.
///
/// Recursive `$ref` cycles can't be expressed in Gemini's flattened
/// schema; they degrade to a permissive open object with a warning.
/// Anything that fails to parse or re-serialise is passed through
/// unchanged — normalisation is best-effort and never blocks the request.
fn normalize_gemini_tool_schema(raw: &RawValue) -> std::borrow::Cow<'static, RawValue> {
    use serde_json::Value;

    let Ok(mut value) = serde_json::from_str::<Value>(raw.get()) else {
        return std::borrow::Cow::Owned(raw.to_owned());
    };

    // Collect `$defs` / `definitions` off the root so refs can resolve
    // against them after they're stripped from the output.
    let mut defs = serde_json::Map::new();
    if let Some(obj) = value.as_object_mut() {
        for key in ["$defs", "definitions"] {
            if let Some(Value::Object(map)) = obj.remove(key) {
                defs.extend(map);
            }
        }
    }

    let mut stack: Vec<String> = Vec::new();
    let resolved = resolve_and_strip_schema(value, &defs, &mut stack);

    match serde_json::value::to_raw_value(&resolved) {
        Ok(rv) => std::borrow::Cow::Owned(rv),
        Err(_) => std::borrow::Cow::Owned(raw.to_owned()),
    }
}

/// The name component of a local definition `$ref`, if it is one.
fn schema_ref_name(reference: &str) -> Option<&str> {
    reference
        .strip_prefix("#/$defs/")
        .or_else(|| reference.strip_prefix("#/definitions/"))
}

/// Recursively strip JSON-Schema meta-fields and inline local `$ref`s.
/// `stack` holds the ref names currently being resolved, to break cycles.
fn resolve_and_strip_schema(
    value: serde_json::Value,
    defs: &serde_json::Map<String, serde_json::Value>,
    stack: &mut Vec<String>,
) -> serde_json::Value {
    use serde_json::Value;

    match value {
        Value::Object(mut obj) => {
            for key in [
                "$schema",
                "$id",
                "$comment",
                "$anchor",
                "$defs",
                "definitions",
            ] {
                obj.remove(key);
            }

            if let Some(Value::String(reference)) = obj.remove("$ref") {
                let Some(name) = schema_ref_name(&reference) else {
                    tracing::warn!(%reference, "Gemini: dropping non-local $ref from tool schema");
                    return resolve_object_children(obj, defs, stack);
                };
                if stack.iter().any(|n| n == name) {
                    tracing::warn!(
                        %reference,
                        "Gemini: dropping recursive $ref from tool schema; inlining as open object",
                    );
                    return serde_json::json!({ "type": "object" });
                }
                let Some(def) = defs.get(name) else {
                    tracing::warn!(%reference, "Gemini: unresolved $ref in tool schema; dropping");
                    return resolve_object_children(obj, defs, stack);
                };
                stack.push(name.to_string());
                let resolved = resolve_and_strip_schema(def.clone(), defs, stack);
                stack.pop();
                // Merge any sibling keywords over the resolved definition
                // (siblings win), recursing into each.
                if let Value::Object(mut resolved_obj) = resolved {
                    for (k, v) in obj {
                        resolved_obj.insert(k, resolve_and_strip_schema(v, defs, stack));
                    }
                    return Value::Object(resolved_obj);
                }
                return resolved;
            }

            resolve_object_children(obj, defs, stack)
        }
        Value::Array(arr) => Value::Array(
            arr.into_iter()
                .map(|v| resolve_and_strip_schema(v, defs, stack))
                .collect(),
        ),
        other => other,
    }
}

/// Recurse into the children of an already-meta-stripped object.
fn resolve_object_children(
    obj: serde_json::Map<String, serde_json::Value>,
    defs: &serde_json::Map<String, serde_json::Value>,
    stack: &mut Vec<String>,
) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for (k, v) in obj {
        out.insert(k, resolve_and_strip_schema(v, defs, stack));
    }
    serde_json::Value::Object(out)
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
            // typed variants where the mapping is clear. Context-window
            // exceeded is a 400 with INVALID_ARGUMENT and a free-form
            // message; detect via wording match (no typed code from
            // the upstream).
            if status == 400 && is_google_context_exceeded(&body_text) {
                return Err(Error::context_window_exceeded(
                    "Google",
                    body_text.to_string(),
                ));
            }
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
        mut signature: Option<String>,
    ) {
        let events = self.tracker.open_one_shot(PartKind::ToolCall {
            call_id,
            name: name.clone(),
        });
        // open_one_shot emits PartStart then PartEnd; splice the
        // arguments Delta (and Gemini's thoughtSignature, if any) in
        // right after the PartStart. Forward each event and inject once,
        // without depending (via panicking `expect`/`unreachable!`) on
        // the exact event count — a future PartTracker change shouldn't
        // crash the stream.
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
                if let Some(sig) = signature.take() {
                    out.push(StreamEvent::PartUpdate {
                        index,
                        update: PartUpdate::Signature(sig),
                    });
                }
            } else {
                out.push(ev);
            }
        }
    }
}

/// Heuristic match for Vertex's "input too long" 400. Vertex returns
/// an `INVALID_ARGUMENT` envelope with a free-form message; we look
/// for the documented wording. Conservative — a near-miss falls
/// through to a generic provider error.
///
/// The accepted clauses all anchor on a token-specific phrase:
/// - `token count` (matches "input token count exceeds the maximum")
/// - `input token`
/// - `context length`
///
/// An earlier version also matched a bare `exceeds the maximum`,
/// which false-positived on unrelated `INVALID_ARGUMENT` parameter
/// validation errors (`candidate_count exceeds the maximum allowed
/// value of 8`, `max_output_tokens exceeds the maximum…`). Dropped —
/// the three token-anchored clauses above cover the real
/// context-exceeded shape, and matches Anthropic's detector
/// requiring co-occurrence with a token word.
fn is_google_context_exceeded(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("invalid_argument")
        && (lower.contains("token count")
            || lower.contains("input token")
            || lower.contains("context length"))
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
                        function_call.thought_signature.clone(),
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
    fn detect_context_exceeded_in_invalid_argument_error() {
        let body = r#"{"error":{"code":400,"message":"The input token count (1500000) exceeds the maximum number of tokens allowed (1000000).","status":"INVALID_ARGUMENT"}}"#;
        assert!(is_google_context_exceeded(body));

        let body2 = r#"{"error":{"code":400,"message":"context length 1100000 exceeds limit","status":"INVALID_ARGUMENT"}}"#;
        assert!(is_google_context_exceeded(body2));

        // Unrelated INVALID_ARGUMENT should not match.
        let body3 =
            r#"{"error":{"code":400,"message":"model not found","status":"INVALID_ARGUMENT"}}"#;
        assert!(!is_google_context_exceeded(body3));

        // Non-INVALID_ARGUMENT status should not match even with token-ish wording.
        let body4 = r#"{"error":{"code":500,"message":"token count high","status":"INTERNAL"}}"#;
        assert!(!is_google_context_exceeded(body4));
    }

    /// PR-review #4: the bare `exceeds the maximum` clause matched
    /// unrelated `INVALID_ARGUMENT` parameter-validation errors —
    /// e.g. a `candidate_count` cap or any other `> max allowed value`
    /// shape Vertex surfaces. The detector must require co-occurrence
    /// with a token-related word, matching the Anthropic detector's
    /// `&& (tokens || input length)` shape.
    #[test]
    fn candidate_count_validation_error_is_not_context_exceeded() {
        let body = r#"{"error":{"code":400,"message":"The value of candidate_count exceeds the maximum allowed value of 8.","status":"INVALID_ARGUMENT"}}"#;
        assert!(
            !is_google_context_exceeded(body),
            "unrelated parameter-cap validation error must not classify as context-exceeded"
        );
    }

    /// `max_output_tokens` validation errors mention `tokens` in the
    /// parameter NAME, but they're an output-config issue, not a
    /// context-window one. The detector must distinguish them from
    /// real input-too-long errors.
    #[test]
    fn max_output_tokens_validation_error_is_not_context_exceeded() {
        let body = r#"{"error":{"code":400,"message":"The value of max_output_tokens (200000) exceeds the maximum allowed value of 65536.","status":"INVALID_ARGUMENT"}}"#;
        assert!(
            !is_google_context_exceeded(body),
            "output-token-cap validation error must not classify as context-exceeded"
        );
    }

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

    fn tool_with_schema(schema: &str) -> crate::types::Tool {
        use std::borrow::Cow;
        let raw = serde_json::value::RawValue::from_string(schema.to_string()).unwrap();
        crate::types::Tool::function("f", None, Cow::Owned(raw))
    }

    /// #1: Gemini rejects `$schema` / `$ref` / `$defs` meta-fields in
    /// tool parameters. The Google provider must strip the meta-fields
    /// and inline local `$ref`s against `$defs` before sending.
    #[test]
    fn tool_schema_meta_fields_stripped_and_refs_inlined() {
        let schema = r##"{
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": { "source": { "$ref": "#/$defs/SourceId" } },
            "$defs": { "SourceId": { "type": "string", "format": "uuid" } }
        }"##;
        let prompt = crate::Prompt::user("hi");
        let cfg = Config::builder("gemini")
            .tools(vec![tool_with_schema(schema)])
            .build();
        let body = provider().convert_request(&prompt, cfg.raw()).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        let params = &json["tools"][0]["functionDeclarations"][0]["parameters"];

        assert!(params.get("$schema").is_none(), "$schema must be stripped");
        assert!(params.get("$defs").is_none(), "$defs must be stripped");
        // The `$ref` is inlined to the resolved definition.
        assert_eq!(params["properties"]["source"]["type"], "string");
        assert_eq!(params["properties"]["source"]["format"], "uuid");
        assert!(
            params["properties"]["source"].get("$ref").is_none(),
            "$ref must be resolved away"
        );
    }

    /// A recursive `$ref` can't be expressed in Gemini's flattened
    /// schema; normalisation must terminate (not stack-overflow) and
    /// degrade the cycle to a permissive object.
    #[test]
    fn tool_schema_recursive_ref_terminates() {
        let schema = r##"{
            "type": "object",
            "properties": { "child": { "$ref": "#/$defs/Node" } },
            "$defs": { "Node": { "type": "object", "properties": { "next": { "$ref": "#/$defs/Node" } } } }
        }"##;
        let prompt = crate::Prompt::user("hi");
        let cfg = Config::builder("gemini")
            .tools(vec![tool_with_schema(schema)])
            .build();
        let body = provider().convert_request(&prompt, cfg.raw()).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        let params = &json["tools"][0]["functionDeclarations"][0]["parameters"];
        // First level inlines; the recursive self-reference degrades to
        // an open object rather than looping forever.
        assert_eq!(params["properties"]["child"]["type"], "object");
        assert!(params.get("$defs").is_none());
    }

    /// #2: a prompt with only a system instruction (no user/assistant
    /// turn) leaves Gemini's `contents` empty; surface a typed
    /// `InvalidPrompt` before the round trip instead of a raw 400.
    #[test]
    fn system_only_prompt_is_rejected_with_typed_error() {
        let prompt = crate::Prompt::system("you are helpful");
        let cfg = Config::builder("gemini").build();
        let err = provider()
            .convert_request(&prompt, cfg.raw())
            .expect_err("system-only prompt must be rejected");
        assert!(matches!(err, Error::InvalidPrompt(_)), "got {err:?}");
    }

    /// #3: an assistant turn whose tool call has no matching tool result
    /// in the following user turn violates Gemini's pairing rule. Caught
    /// before the round trip as a typed error.
    #[test]
    fn unmatched_tool_call_is_rejected_with_typed_error() {
        use crate::types::FunctionCall;
        let prompt = crate::Prompt::user("hi").with_assistant_tool_call(FunctionCall {
            call_id: "c1".into(),
            name: "f".into(),
            arguments: "{}".into(),
            provider_signature: None,
        });
        let cfg = Config::builder("gemini").build();
        let err = provider()
            .convert_request(&prompt, cfg.raw())
            .expect_err("unmatched tool call must be rejected");
        assert!(matches!(err, Error::InvalidPrompt(_)), "got {err:?}");
    }

    /// A matched tool call / result pair passes validation.
    #[test]
    fn matched_tool_call_and_result_passes() {
        use crate::types::FunctionCall;
        let prompt = crate::Prompt::user("hi")
            .with_assistant_tool_call(FunctionCall {
                call_id: "c1".into(),
                name: "f".into(),
                arguments: "{}".into(),
                provider_signature: None,
            })
            .with_tool_result("c1", "ok");
        let cfg = Config::builder("gemini").build();
        assert!(provider().convert_request(&prompt, cfg.raw()).is_ok());
    }

    /// #4 (request side): a tool call carrying a `provider_signature` is
    /// echoed back as Gemini's `thoughtSignature` on the wire.
    #[test]
    fn provider_signature_echoed_as_thought_signature() {
        use crate::types::FunctionCall;
        let prompt = crate::Prompt::user("hi")
            .with_assistant_tool_call(FunctionCall {
                call_id: "c1".into(),
                name: "f".into(),
                arguments: "{}".into(),
                provider_signature: Some("sig_xyz".into()),
            })
            .with_tool_result("c1", "ok");
        let cfg = Config::builder("gemini").build();
        let body = provider().convert_request(&prompt, cfg.raw()).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        // contents: [user "hi", model [functionCall], user [functionResponse]]
        assert_eq!(
            json["contents"][1]["parts"][0]["functionCall"]["thoughtSignature"],
            "sig_xyz",
        );
    }

    /// #4 (response side): Gemini's `thoughtSignature` on a `functionCall`
    /// part is captured and surfaced via a `PartUpdate::Signature`, which
    /// the accumulator lands on `FunctionCall::provider_signature`.
    #[test]
    fn thought_signature_parsed_into_provider_signature() {
        let chunk = r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"Paris"},"thoughtSignature":"sig_abc"}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}"#;
        let mut state = GoogleStreamState::default();
        let r: GoogleResponse = serde_json::from_str(chunk).unwrap();
        let events = convert_response_stateful(r, &mut state).unwrap();

        let mut acc = crate::accumulator::ResponseAccumulator::new();
        for ev in events {
            acc.process_event(ev).unwrap();
        }
        let resp = acc.finalize().unwrap();
        let call = resp
            .content
            .iter()
            .find_map(|p| match p {
                AssistantPart::ToolCall(c) => Some(c),
                _ => None,
            })
            .expect("expected a tool call");
        assert_eq!(call.provider_signature.as_deref(), Some("sig_abc"));
    }
}
