use futures_util::{Stream, StreamExt};
use ijson::{ijson, IValue};
use serde_json::value::RawValue;
use uuid::Uuid;

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;

use super::endpoint::VertexEndpoint;
use super::google_types::*;
use crate::factory::ProviderType;
use crate::provider::Provider;
use crate::providers::file_resolve::{
    media_type_extension, percent_encode, resolve_refs, NoLibraryUpload, ProviderUploader,
    ResolvedRef,
};
use crate::sse_stream::SseStream;
use crate::transport::{Method, Transport, TransportRequest, UploadRequest};
use crate::types::{
    Annotation, AnnotationKind, AssistantPart, FileResolver, FileSource, FinishReason, InputItem,
    PartKind, PartUpdate, ProviderScope, ResolvedHandle, UserPart,
};
use crate::{Error, RawConfig, Response, StreamEvent};

/// Google provider implementation via Vertex AI (for Gemini models).
pub struct GoogleProvider {
    endpoint: VertexEndpoint,
    transport: Transport,
    /// Optional caller-held file registry for resolving `Ref` file inputs.
    file_resolver: Option<Arc<dyn FileResolver>>,
    /// Cloud Storage bucket the lib uploads streamed `Ref` files into. When
    /// set, a `Ref` resolving to a stream is uploaded to GCS (using the same
    /// Vertex OAuth token) and referenced as a `gs://` URI. When unset, a
    /// streamed `Ref` is rejected (caller must supply a handle/URL).
    gcs_bucket: Option<String>,
    /// Optional object-name prefix for uploaded files (default
    /// `platformed-llm/`).
    gcs_prefix: Option<String>,
    /// Cooperative rate limiter consulted before every send.
    rate_limiter: crate::rate_limit::SharedRateLimiter,
}

impl GoogleProvider {
    /// Create a new Google provider with access token authentication.
    pub fn new(project_id: String, location: String, access_token: String) -> Result<Self, Error> {
        Ok(Self {
            endpoint: VertexEndpoint::with_access_token(project_id, location, access_token),
            transport: Transport::reqwest()?,
            file_resolver: None,
            gcs_bucket: None,
            gcs_prefix: None,
            rate_limiter: crate::rate_limit::default_shared_limiter(),
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
            file_resolver: None,
            gcs_bucket: None,
            gcs_prefix: None,
            rate_limiter: crate::rate_limit::default_shared_limiter(),
        })
    }

    /// Create a new Google provider with Application Default Credentials.
    pub async fn with_adc(project_id: String, location: String) -> Result<Self, Error> {
        Ok(Self {
            endpoint: VertexEndpoint::with_adc(project_id, location).await?,
            transport: Transport::reqwest()?,
            file_resolver: None,
            gcs_bucket: None,
            gcs_prefix: None,
            rate_limiter: crate::rate_limit::default_shared_limiter(),
        })
    }

    /// Create a new Google provider with a caller-supplied [`Transport`]
    /// and pre-built [`VertexEndpoint`]. Lets downstream consumers / tests
    /// plug in custom recording / replaying / retrying transports.
    pub fn with_transport(endpoint: VertexEndpoint, transport: Transport) -> Self {
        Self {
            endpoint,
            transport,
            file_resolver: None,
            gcs_bucket: None,
            gcs_prefix: None,
            rate_limiter: crate::rate_limit::default_shared_limiter(),
        }
    }

    /// Attach a [`FileResolver`] so the provider can resolve
    /// [`FileSource::Ref`](crate::FileSource::Ref) file inputs.
    ///
    /// If a [GCS bucket is configured](Self::with_gcs_bucket), a `Ref` that
    /// resolves to a stream is uploaded to Cloud Storage (using the same
    /// Vertex OAuth token) and referenced as a `gs://` URI. Otherwise the
    /// resolver must return a durable `gs://` URI via
    /// [`ResolvedFile::ProviderHandle`](crate::ResolvedFile::ProviderHandle)
    /// (or, for a genuinely public file, a public URL via
    /// [`ResolvedFile::Url`](crate::ResolvedFile::Url)); a streaming payload is
    /// rejected.
    pub fn with_file_resolver(mut self, resolver: Arc<dyn FileResolver>) -> Self {
        self.file_resolver = Some(resolver);
        self
    }

    /// Configure the Cloud Storage bucket the lib uploads streamed `Ref` files
    /// into. The bucket must live in the same GCP project as the Vertex
    /// endpoint (Gemini only fetches `gs://` objects from the requesting
    /// project, or publicly-readable ones). Uploads reuse the endpoint's
    /// OAuth token — the `cloud-platform` scope covers both Vertex and GCS.
    pub fn with_gcs_bucket(mut self, bucket: impl Into<String>) -> Self {
        self.gcs_bucket = Some(bucket.into());
        self
    }

    /// Override the object-name prefix for uploaded files. Defaults to
    /// `platformed-llm/`. A trailing `/` makes it a folder.
    pub fn with_gcs_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.gcs_prefix = Some(prefix.into());
        self
    }

    /// Attach a shared [`crate::rate_limit::RateLimiter`]. See the
    /// equivalent method on the OpenAI provider for the model — same
    /// trait, same semantics.
    pub fn with_rate_limiter(mut self, limiter: crate::rate_limit::SharedRateLimiter) -> Self {
        self.rate_limiter = limiter;
        self
    }

    /// The [`ProviderScope`] file handles are valid within — the GCP
    /// project + region.
    fn scope(&self) -> ProviderScope {
        ProviderScope::new(
            ProviderType::Google,
            format!(
                "{}/{}",
                self.endpoint.project_id(),
                self.endpoint.location()
            ),
        )
    }


    /// Swap the static access token before it expires (GCP tokens
    /// last ~1h). Errors if this provider was built with ADC, which
    /// refreshes automatically. See [`VertexEndpoint::set_access_token`].
    pub fn set_access_token(&self, token: impl Into<String>) -> Result<(), Error> {
        self.endpoint.set_access_token(token)
    }

    /// Convert internal request to Google format.
    ///
    /// `resolved` maps each file-`Ref` id to its wire-ready reference, built
    /// by the async [`resolve_refs`] pre-pass in [`Self::generate`].
    fn convert_request(
        &self,
        prompt: &crate::Prompt,
        config: &RawConfig,
        resolved: &HashMap<String, ResolvedRef>,
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

        // The function_call / function_response pairing invariant Gemini
        // enforces is checked provider-agnostically in
        // [`crate::middleware::validate_prompt`] (every provider requires
        // it), so it is not re-validated here.

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
                            // Image / audio / document / video all map the same
                            // way (inlineData for base64, fileData for URL/Ref);
                            // only the fallback MIME differs.
                            UserPart::Image(src) => {
                                if let Some(part) = file_source_to_part(src, "image/*", resolved) {
                                    push_part(&mut contents, "user", part);
                                }
                            }
                            UserPart::Audio(src) => {
                                if let Some(part) = file_source_to_part(src, "audio/*", resolved) {
                                    push_part(&mut contents, "user", part);
                                }
                            }
                            UserPart::Document(src) => {
                                if let Some(part) =
                                    file_source_to_part(src, "application/pdf", resolved)
                                {
                                    push_part(&mut contents, "user", part);
                                }
                            }
                            UserPart::Video(src) => {
                                if let Some(part) = file_source_to_part(src, "video/*", resolved) {
                                    push_part(&mut contents, "user", part);
                                }
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
            // Vertex's `generationConfig.responseSchema` uses the same
            // OpenAPI-subset dialect as `functionDeclarations[].parameters`
            // — it rejects `$schema` / `$ref` / `$defs` with a 400. Run
            // the same normaliser so a typed-struct response schema
            // doesn't reach the wire raw.
            Some(crate::types::ResponseFormat::JsonSchema { schema, .. }) => (
                Some("application/json".to_string()),
                Some(normalize_gemini_tool_schema(schema)),
            ),
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
        //
        // The "prompt must contain at least one user/assistant turn"
        // requirement is enforced provider-agnostically in
        // [`crate::middleware::validate_prompt`], so a system-only prompt
        // is rejected uniformly across providers before reaching here —
        // it is not re-checked at this layer.

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

/// Normalise a function tool's JSON-Schema `parameters` into the subset
/// Gemini's `functionDeclarations[].parameters` accepts. Gemini takes
/// only the property keywords of JSON Schema and rejects the meta-fields
/// `$schema`, `$ref`, and `$defs` with a 400, so we:
///
/// - drop meta-fields (`$schema`, `$id`, `$comment`, `$anchor`, `$defs`,
///   `definitions`) wherever they appear in *schema* position, and
/// - inline every local `$ref` (`#/$defs/Name` or `#/definitions/Name`)
///   against the definitions in scope, merging any sibling keywords over
///   the resolved definition.
///
/// The walk is **position-aware**: it descends only into keywords that
/// actually hold subschemas (`properties`, `items`, `additionalProperties`,
/// `allOf`/`anyOf`/`oneOf`, `$defs`, …) and copies every other keyword
/// verbatim. So data that merely *looks* like schema — a `const` /
/// `default` / `enum` value that happens to contain a `$ref` key, or a
/// property literally named `$defs` — is left untouched rather than being
/// silently mangled. `$defs` are collected lexically (root *and* nested),
/// so a ref to a definition nested inside a subschema still resolves.
///
/// Recursive `$ref` cycles can't be expressed in Gemini's flattened
/// schema; they degrade to a permissive open object with a warning.
/// Anything that fails to parse or re-serialise is passed through
/// unchanged — normalisation is best-effort and never blocks the request.
fn normalize_gemini_tool_schema(raw: &RawValue) -> std::borrow::Cow<'static, RawValue> {
    use serde_json::Value;

    let Ok(value) = serde_json::from_str::<Value>(raw.get()) else {
        return std::borrow::Cow::Owned(raw.to_owned());
    };

    let empty = serde_json::Map::new();
    let mut stack: Vec<String> = Vec::new();
    let resolved = normalize_schema(value, &empty, &mut stack);

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

/// Keywords whose value is a *single* subschema (or, for
/// `additionalProperties` / `items`, possibly a boolean — copied as-is).
const SINGLE_SUBSCHEMA_KEYS: &[&str] = &[
    "additionalProperties",
    "additionalItems",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contains",
    "propertyNames",
    "if",
    "then",
    "else",
    "not",
];

/// Keywords whose value is an array of subschemas.
const SUBSCHEMA_ARRAY_KEYS: &[&str] = &["allOf", "anyOf", "oneOf", "prefixItems"];

/// Keywords whose value is a `{ name -> subschema }` map (the map keys are
/// arbitrary names, not schema keywords, so they're preserved verbatim).
const SUBSCHEMA_MAP_KEYS: &[&str] = &["properties", "patternProperties", "dependentSchemas"];

/// Normalise one JSON-Schema node. `defs_in_scope` are the definitions
/// visible here (root + every enclosing `$defs`); `stack` holds the ref
/// names currently being expanded, to break cycles.
fn normalize_schema(
    value: serde_json::Value,
    defs_in_scope: &serde_json::Map<String, serde_json::Value>,
    stack: &mut Vec<String>,
) -> serde_json::Value {
    use serde_json::Value;

    let Value::Object(obj) = value else {
        // Non-objects in schema position (e.g. a boolean schema) pass
        // through unchanged.
        return value;
    };

    // Extend the scope with this node's local `$defs` / `definitions` so
    // refs anywhere in this subtree (including sibling defs) resolve.
    let mut local = serde_json::Map::new();
    for key in ["$defs", "definitions"] {
        if let Some(Value::Object(map)) = obj.get(key) {
            for (k, v) in map {
                local.insert(k.clone(), v.clone());
            }
        }
    }
    let scope: std::borrow::Cow<serde_json::Map<String, Value>> = if local.is_empty() {
        std::borrow::Cow::Borrowed(defs_in_scope)
    } else {
        let mut merged = defs_in_scope.clone();
        merged.extend(local);
        std::borrow::Cow::Owned(merged)
    };
    let defs = scope.as_ref();

    if let Some(Value::String(reference)) = obj.get("$ref") {
        let reference = reference.clone();
        return resolve_ref(&reference, &obj, defs, stack);
    }

    Value::Object(normalize_schema_keys(obj, defs, stack))
}

/// Rebuild a schema object minus meta-fields, recursing only into
/// subschema-bearing keywords and copying everything else verbatim.
/// Assumes any `$ref` has already been handled by the caller.
fn normalize_schema_keys(
    obj: serde_json::Map<String, serde_json::Value>,
    defs: &serde_json::Map<String, serde_json::Value>,
    stack: &mut Vec<String>,
) -> serde_json::Map<String, serde_json::Value> {
    use serde_json::Value;

    let mut out = serde_json::Map::new();
    for (k, v) in obj {
        match k.as_str() {
            // Meta-fields and definition blocks: stripped from the output
            // (defs were already lifted into scope by the caller).
            "$schema" | "$id" | "$comment" | "$anchor" | "$defs" | "definitions" | "$ref" => {}
            _ if SUBSCHEMA_MAP_KEYS.contains(&k.as_str()) => {
                let mapped = match v {
                    Value::Object(m) => Value::Object(
                        m.into_iter()
                            .map(|(name, sub)| (name, normalize_schema(sub, defs, stack)))
                            .collect(),
                    ),
                    other => other,
                };
                out.insert(k, mapped);
            }
            _ if SUBSCHEMA_ARRAY_KEYS.contains(&k.as_str()) => {
                out.insert(k, map_subschema_array(v, defs, stack));
            }
            // `items` is a single subschema in 2020-12 but an array of
            // subschemas in draft-07; handle both shapes.
            "items" => {
                let mapped = match v {
                    Value::Array(_) => map_subschema_array(v, defs, stack),
                    Value::Object(_) => normalize_schema(v, defs, stack),
                    other => other,
                };
                out.insert(k, mapped);
            }
            _ if SINGLE_SUBSCHEMA_KEYS.contains(&k.as_str()) => {
                // Subschema, or a bare bool (e.g. `additionalProperties:
                // false`) which is copied as-is.
                let mapped = match v {
                    Value::Object(_) => normalize_schema(v, defs, stack),
                    other => other,
                };
                out.insert(k, mapped);
            }
            // Data / scalar validation keyword (`type`, `enum`, `const`,
            // `default`, `examples`, `required`, `format`, …): copy
            // verbatim — never strip or resolve inside a data position.
            _ => {
                out.insert(k, v);
            }
        }
    }
    out
}

/// Map [`normalize_schema`] over an array of subschemas.
fn map_subschema_array(
    value: serde_json::Value,
    defs: &serde_json::Map<String, serde_json::Value>,
    stack: &mut Vec<String>,
) -> serde_json::Value {
    match value {
        serde_json::Value::Array(arr) => serde_json::Value::Array(
            arr.into_iter()
                .map(|s| normalize_schema(s, defs, stack))
                .collect(),
        ),
        other => other,
    }
}

/// Resolve a `$ref` against `defs`, inlining the (normalised) definition
/// and merging any sibling keywords of the `$ref` object over it.
fn resolve_ref(
    reference: &str,
    obj: &serde_json::Map<String, serde_json::Value>,
    defs: &serde_json::Map<String, serde_json::Value>,
    stack: &mut Vec<String>,
) -> serde_json::Value {
    use serde_json::Value;

    let drop_ref_and_normalize = |stack: &mut Vec<String>| {
        let mut without_ref = obj.clone();
        without_ref.remove("$ref");
        Value::Object(normalize_schema_keys(without_ref, defs, stack))
    };

    let Some(name) = schema_ref_name(reference) else {
        tracing::warn!(%reference, "Gemini: dropping non-local $ref from tool schema");
        return drop_ref_and_normalize(stack);
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
        return drop_ref_and_normalize(stack);
    };

    stack.push(name.to_string());
    let resolved = normalize_schema(def.clone(), defs, stack);
    stack.pop();

    // Merge sibling keywords (everything but `$ref`) over the resolved
    // definition — siblings win, matching draft 2020-12 where `$ref` is an
    // applicator that composes with adjacent keywords.
    let Value::Object(mut resolved_obj) = resolved else {
        return resolved;
    };
    let mut without_ref = obj.clone();
    without_ref.remove("$ref");
    for (k, v) in normalize_schema_keys(without_ref, defs, stack) {
        resolved_obj.insert(k, v);
    }
    Value::Object(resolved_obj)
}

#[async_trait::async_trait]
impl Provider for GoogleProvider {
    async fn generate(
        &self,
        prompt: &crate::Prompt,
        config: &RawConfig,
    ) -> Result<Response, Error> {
        // Upload streamed Refs to GCS when a bucket is configured; otherwise
        // require the resolver to supply a durable handle/URL.
        let no_upload = NoLibraryUpload { provider: "Google" };
        let uploader: &dyn ProviderUploader = if self.gcs_bucket.is_some() {
            self
        } else {
            &no_upload
        };
        let resolved = resolve_refs(
            prompt.items(),
            &self.scope(),
            self.file_resolver.as_deref(),
            uploader,
        )
        .await?;
        let google_request = self.convert_request(prompt, config, &resolved)?;

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

        let scope = crate::rate_limit::RateScope {
            // Vertex regions have independent quotas — include the
            // location so per-region buckets stay separate.
            bucket_key: format!(
                "Vertex-Google/{}/{}",
                self.endpoint.location(),
                config.model,
            ),
            tenant: config.tenant.unwrap_or(uuid::Uuid::nil()),
            priority: config.priority.unwrap_or_default(),
        };
        let permit = self.rate_limiter.acquire(&scope).await?;
        let response = match self.transport.send(req).await {
            Ok(r) => r,
            Err(e) => {
                permit.observe(crate::rate_limit::RateOutcome::OtherFailure);
                return Err(e);
            }
        };

        if !(200..300).contains(&response.status) {
            let status = response.status;
            // Read Retry-After before `collect_body` consumes the response.
            let retry_after = crate::transport::parse_retry_after(response.header("retry-after"));
            if status == 429 {
                permit.observe(crate::rate_limit::RateOutcome::RateLimited {
                    retry_after: retry_after.map(std::time::Duration::from_secs),
                    info: crate::rate_limit::ProviderRateInfo::default(),
                });
            } else {
                permit.observe(crate::rate_limit::RateOutcome::OtherFailure);
            }
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
                // 5xx (and any other status) may carry a
                // `Retry-After` per RFC 7231; thread it through so
                // the retry helper honours the server hint.
                _ => Error::provider_with_retry_after(
                    "Google",
                    status,
                    retry_after,
                    format!("API error: {body_text}"),
                ),
            });
        }

        // Success path: defer the limiter observation to stream-end
        // — see `rate_limit::observe_stream`. We do this even though
        // Vertex Gemini doesn't have a known mid-stream rate-limit
        // signal yet, so transport drops mid-response are reported as
        // `OtherFailure` rather than `Success`.

        // Create SSE stream from response (Gemini supports ?alt=sse)
        let sse_stream = SseStream::new("Google", response.body);

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

        let observed = crate::rate_limit::observe_response_stream(
            event_stream,
            permit,
            crate::rate_limit::ProviderRateInfo::default(),
        );
        Ok(Response::from_stream(observed))
    }
}

/// Cloud Storage JSON-API upload host. Auth is the same `cloud-platform`
/// bearer used for Vertex.
const GCS_UPLOAD_HOST: &str = "https://storage.googleapis.com";

#[async_trait]
impl ProviderUploader for GoogleProvider {
    /// Stream `body` to a Cloud Storage object and return its `gs://` URI.
    ///
    /// Uses the GCS JSON API single-request media upload
    /// (`uploadType=media`), authenticated with the endpoint's Vertex OAuth
    /// token. The object lives in the configured bucket under the configured
    /// prefix with a random name; Gemini reads it via `fileData.fileUri`.
    async fn upload(
        &self,
        media_type: &str,
        content_length: Option<u64>,
        body: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>>,
    ) -> Result<ResolvedHandle, Error> {
        let bucket = self.gcs_bucket.as_deref().ok_or_else(|| {
            Error::config("GoogleProvider GCS upload called without a configured bucket")
        })?;
        let prefix = self.gcs_prefix.as_deref().unwrap_or("platformed-llm/");
        let object = match media_type_extension(media_type) {
            "" => format!("{prefix}{}", Uuid::new_v4()),
            ext => format!("{prefix}{}.{ext}", Uuid::new_v4()),
        };
        let url = format!(
            "{GCS_UPLOAD_HOST}/upload/storage/v1/b/{bucket}/o?uploadType=media&name={}",
            percent_encode(&object),
        );
        let headers = vec![
            self.endpoint.auth_header().await?,
            ("Content-Type".to_string(), media_type.to_string()),
        ];
        let req = UploadRequest {
            method: Method::Post,
            url,
            headers,
            content_length,
            body,
        };
        let response = self.transport.send_upload(req).await?;
        let status = response.status;
        let bytes = response.collect_body().await.unwrap_or_default();
        if !(200..300).contains(&status) {
            let body_str = String::from_utf8_lossy(&bytes).into_owned();
            return Err(Error::provider_with_status(
                "Google",
                status,
                format!("GCS upload failed: {body_str}"),
            ));
        }
        Ok(ResolvedHandle {
            uri: format!("gs://{bucket}/{object}"),
            media_type: media_type.to_string(),
            expires_at: None,
        })
    }
}

/// Convert a [`FileSource`] (any modality) to a Gemini part: `inlineData` for
/// inline base64, `fileData` for a URL or a resolved `Ref`. `fallback_mime` is
/// used for URL/Ref inputs that don't carry their own MIME type.
fn file_source_to_part(
    src: &FileSource,
    fallback_mime: &str,
    resolved: &HashMap<String, ResolvedRef>,
) -> Option<GooglePart> {
    match src {
        FileSource::Base64 { data, media_type } => Some(GooglePart::InlineData {
            inline_data: GoogleInlineData {
                mime_type: media_type.clone(),
                data: data.clone(),
            },
        }),
        FileSource::Url(u) => Some(GooglePart::FileData {
            file_data: GoogleFileData {
                mime_type: fallback_mime.to_string(),
                file_uri: u.clone(),
            },
        }),
        FileSource::Ref(id) => ref_to_file_data(resolved, id, fallback_mime),
    }
}

/// Resolve a file `Ref` to a Gemini `fileData` part, or `None` (logged) when
/// the id wasn't resolved. Both handle and URL results become a `fileData`
/// `fileUri` — a `gs://` or `https` URI Vertex fetches at request time.
fn ref_to_file_data(
    resolved: &HashMap<String, ResolvedRef>,
    id: &str,
    fallback_mime: &str,
) -> Option<GooglePart> {
    match resolved.get(id) {
        Some(ResolvedRef::Handle { uri, media_type })
        | Some(ResolvedRef::Url { uri, media_type }) => {
            let mime = if media_type.is_empty() {
                fallback_mime.to_string()
            } else {
                media_type.clone()
            };
            Some(GooglePart::FileData {
                file_data: GoogleFileData {
                    mime_type: mime,
                    file_uri: uri.clone(),
                },
            })
        }
        None => {
            tracing::debug!("Gemini: unresolved file Ref {id}; dropping");
            None
        }
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
        let body = provider
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
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
        let body = provider
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
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
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["generationConfig"]["stopSequences"],
            serde_json::json!(["END", "STOP"]),
        );
    }

    /// A resolved document `Ref` lands as a `fileData` part carrying the
    /// resolved URI and real MIME type (handle and URL both map here).
    #[test]
    fn resolved_ref_emits_file_data() {
        use crate::providers::file_resolve::ResolvedRef;
        use crate::types::{FileSource, InputItem, UserPart};

        let prompt = crate::Prompt::new().with_item(InputItem::User {
            content: vec![UserPart::Document(FileSource::Ref("doc1".into()))],
        });
        let mut resolved = std::collections::HashMap::new();
        resolved.insert(
            "doc1".to_string(),
            ResolvedRef::Handle {
                uri: "gs://bucket/x.pdf".into(),
                media_type: "application/pdf".into(),
            },
        );
        let cfg = Config::builder("gemini").build();
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &resolved)
            .unwrap();
        let json = serde_json::to_value(&body).unwrap();
        let part = &json["contents"][0]["parts"][0]["fileData"];
        assert_eq!(part["fileUri"], "gs://bucket/x.pdf");
        assert_eq!(part["mimeType"], "application/pdf");
    }

    /// Video inputs map like the other modalities: a URL → `fileData` (with the
    /// `video/*` fallback mime), inline base64 → `inlineData`.
    #[test]
    fn video_maps_to_gemini_parts() {
        use crate::types::{FileSource, InputItem, UserPart};

        let prompt = crate::Prompt::new().with_item(InputItem::User {
            content: vec![
                UserPart::Video(FileSource::Url("gs://bucket/clip.mp4".into())),
                UserPart::Video(FileSource::Base64 {
                    data: "AAAA".into(),
                    media_type: "video/mp4".into(),
                }),
            ],
        });
        let cfg = Config::builder("gemini").build();
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
        let json = serde_json::to_value(&body).unwrap();
        let parts = &json["contents"][0]["parts"];
        assert_eq!(parts[0]["fileData"]["fileUri"], "gs://bucket/clip.mp4");
        assert_eq!(parts[0]["fileData"]["mimeType"], "video/*");
        assert_eq!(parts[1]["inlineData"]["mimeType"], "video/mp4");
        assert_eq!(parts[1]["inlineData"]["data"], "AAAA");
    }

    #[test]
    fn presence_and_frequency_penalty_threaded_through() {
        let prompt = crate::Prompt::user("hi");
        let cfg = Config::builder("gemini")
            .presence_penalty(0.5)
            .frequency_penalty(0.25)
            .build();
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
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
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
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
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
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
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
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
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
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
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
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
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
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
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
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
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["generationConfig"]["responseMimeType"],
            "application/json",
        );
        assert_eq!(json["generationConfig"]["responseSchema"]["type"], "object");
    }

    /// Vertex's `generationConfig.responseSchema` uses the same OpenAPI-
    /// subset dialect as `functionDeclarations[].parameters` — it rejects
    /// `$schema` / `$ref` / `$defs` with a 400. A response schema built
    /// from a typed struct (the same `$ref`-emitting shape the tool-param
    /// normaliser was added for) must flow through the same normaliser
    /// rather than reaching the wire raw.
    #[test]
    fn response_format_json_schema_is_normalized() {
        use crate::types::ResponseFormat;
        use std::borrow::Cow;
        let schema_raw = serde_json::value::RawValue::from_string(
            r##"{
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": { "source": { "$ref": "#/$defs/SourceId" } },
                "$defs": { "SourceId": { "type": "string", "format": "uuid" } }
            }"##
            .to_string(),
        )
        .unwrap();
        let prompt = crate::Prompt::user("hi");
        let cfg = Config::builder("gemini")
            .response_format(ResponseFormat::JsonSchema {
                name: "Out".to_string(),
                schema: Cow::Owned(schema_raw),
                strict: true,
            })
            .build();
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
        let json = serde_json::to_value(&body).unwrap();
        let schema = &json["generationConfig"]["responseSchema"];
        assert!(
            schema.get("$schema").is_none(),
            "responseSchema must have $schema stripped: {schema}",
        );
        assert!(
            schema.get("$defs").is_none(),
            "responseSchema must have $defs stripped: {schema}",
        );
        assert_eq!(schema["properties"]["source"]["type"], "string");
        assert_eq!(schema["properties"]["source"]["format"], "uuid");
        assert!(
            schema["properties"]["source"].get("$ref").is_none(),
            "responseSchema must have $ref inlined: {schema}",
        );
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
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
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
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
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
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
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
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
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

    /// Run a tool-parameter schema through the full request conversion and
    /// return the normalised `parameters` object Gemini would receive.
    fn normalized_params(schema: &str) -> serde_json::Value {
        let prompt = crate::Prompt::user("hi");
        let cfg = Config::builder("gemini")
            .tools(vec![tool_with_schema(schema)])
            .build();
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
        let json = serde_json::to_value(&body).unwrap();
        json["tools"][0]["functionDeclarations"][0]["parameters"].clone()
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
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
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
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
        let json = serde_json::to_value(&body).unwrap();
        let params = &json["tools"][0]["functionDeclarations"][0]["parameters"];
        // First level inlines; the recursive self-reference degrades to
        // an open object rather than looping forever.
        assert_eq!(params["properties"]["child"]["type"], "object");
        assert!(params.get("$defs").is_none());
    }

    /// Review finding #1: a `$ref` to a definition nested inside a
    /// subschema (not at the root) must still resolve — `$defs` are
    /// collected lexically, root and nested.
    #[test]
    fn tool_schema_nested_defs_resolve() {
        let schema = r##"{
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "$defs": { "Inner": { "type": "string", "format": "uuid" } },
                    "properties": { "id": { "$ref": "#/$defs/Inner" } }
                }
            }
        }"##;
        let params = normalized_params(schema);
        let inner = &params["properties"]["outer"]["properties"]["id"];
        assert_eq!(
            inner["type"], "string",
            "nested $ref must resolve: {params}"
        );
        assert_eq!(inner["format"], "uuid");
        assert!(inner.get("$ref").is_none());
        assert!(params["properties"]["outer"].get("$defs").is_none());
    }

    /// Review finding #6: meta-field *names* used as ordinary property
    /// keys (or keys inside `const` / `default` data) must NOT be stripped
    /// — only schema-position meta-fields are. A property literally named
    /// `$defs` / `$ref` survives.
    #[test]
    fn tool_schema_preserves_data_named_like_meta_fields() {
        let schema = r##"{
            "type": "object",
            "properties": {
                "$ref": { "type": "string" },
                "$defs": { "type": "number" },
                "config": {
                    "type": "object",
                    "default": { "$ref": "literal-data", "$schema": "also-data" },
                    "const": { "$defs": "still-data" }
                }
            }
        }"##;
        let params = normalized_params(schema);
        let props = &params["properties"];
        // Properties whose *names* collide with meta-fields are real
        // properties, not schema keywords — they must remain.
        assert_eq!(props["$ref"]["type"], "string", "got: {params}");
        assert_eq!(props["$defs"]["type"], "number");
        // `default` / `const` are data positions: their contents are
        // copied verbatim, meta-looking keys and all.
        assert_eq!(props["config"]["default"]["$ref"], "literal-data");
        assert_eq!(props["config"]["default"]["$schema"], "also-data");
        assert_eq!(props["config"]["const"]["$defs"], "still-data");
    }

    /// `enum` values are data and pass through untouched.
    #[test]
    fn tool_schema_preserves_enum_values() {
        let schema = r##"{ "type": "string", "enum": ["a", "b", "c"] }"##;
        let params = normalized_params(schema);
        assert_eq!(params["enum"], serde_json::json!(["a", "b", "c"]));
    }

    /// Sibling keywords on a `$ref` object merge over the resolved
    /// definition (siblings win).
    #[test]
    fn tool_schema_ref_siblings_merge_over_definition() {
        let schema = r##"{
            "type": "object",
            "properties": {
                "x": { "$ref": "#/$defs/Base", "description": "the x field" }
            },
            "$defs": { "Base": { "type": "integer", "description": "base desc" } }
        }"##;
        let params = normalized_params(schema);
        let x = &params["properties"]["x"];
        assert_eq!(x["type"], "integer", "resolved def type kept: {params}");
        assert_eq!(x["description"], "the x field", "sibling overrides def");
        assert!(x.get("$ref").is_none());
    }

    /// `additionalProperties` as a subschema with a `$ref` resolves; as a
    /// bare boolean it is copied unchanged.
    #[test]
    fn tool_schema_additional_properties_handled() {
        let schema_ref = r##"{
            "type": "object",
            "additionalProperties": { "$ref": "#/$defs/V" },
            "$defs": { "V": { "type": "boolean" } }
        }"##;
        let params = normalized_params(schema_ref);
        assert_eq!(params["additionalProperties"]["type"], "boolean");
        assert!(params.get("$defs").is_none());

        let schema_bool = r##"{ "type": "object", "additionalProperties": false }"##;
        let params = normalized_params(schema_bool);
        assert_eq!(params["additionalProperties"], serde_json::json!(false));
    }

    /// `allOf` / `anyOf` arrays and array `items` recurse and resolve refs.
    #[test]
    fn tool_schema_combinators_and_items_recurse() {
        let schema = r##"{
            "type": "object",
            "properties": {
                "tags": { "type": "array", "items": { "$ref": "#/$defs/Tag" } },
                "either": { "anyOf": [ { "$ref": "#/$defs/Tag" }, { "type": "null" } ] }
            },
            "$defs": { "Tag": { "type": "string", "minLength": 1 } }
        }"##;
        let params = normalized_params(schema);
        assert_eq!(params["properties"]["tags"]["items"]["type"], "string");
        assert_eq!(params["properties"]["tags"]["items"]["minLength"], 1);
        assert_eq!(params["properties"]["either"]["anyOf"][0]["type"], "string");
        assert_eq!(params["properties"]["either"]["anyOf"][1]["type"], "null");
        assert!(params.get("$defs").is_none());
    }

    /// An unresolvable / non-local `$ref` degrades gracefully: the `$ref`
    /// is dropped (so Gemini doesn't 400 on it) and sibling keywords are
    /// retained, rather than crashing.
    #[test]
    fn tool_schema_unresolved_ref_drops_gracefully() {
        let schema = r##"{
            "type": "object",
            "properties": {
                "a": { "$ref": "https://example.com/external", "type": "string" },
                "b": { "$ref": "#/$defs/Missing", "title": "B" }
            }
        }"##;
        let params = normalized_params(schema);
        // Non-local ref dropped, sibling `type` kept.
        assert_eq!(params["properties"]["a"]["type"], "string");
        assert!(params["properties"]["a"].get("$ref").is_none());
        // Unresolved local ref dropped, sibling `title` kept.
        assert_eq!(params["properties"]["b"]["title"], "B");
        assert!(params["properties"]["b"].get("$ref").is_none());
    }

    /// Malformed (non-JSON) parameters are passed through unchanged rather
    /// than blocking the request.
    #[test]
    fn tool_schema_non_json_passes_through() {
        let raw = serde_json::value::RawValue::from_string("true".to_string()).unwrap();
        let out = normalize_gemini_tool_schema(&raw);
        assert_eq!(out.get(), "true");
    }

    /// A matched tool call / result pair converts cleanly. (The pairing
    /// invariant itself is enforced provider-agnostically in
    /// `crate::middleware::validate_prompt`.)
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
        assert!(provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .is_ok());
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
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
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
