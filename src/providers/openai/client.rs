use super::types::{
    OpenAIAnnotation, OpenAIReasoning, OpenAIStreamEvent, OpenAIToolChoice, ResponsesRequest,
};
use crate::factory::ProviderType;
use crate::provider::Provider;
use crate::providers::file_resolve::{
    media_type_extension, resolve_refs, ProviderUploader, ResolvedRef,
};
use crate::transport::{Method, Transport, TransportRequest, UploadRequest};
use crate::types::{
    Annotation, AnnotationKind, FileResolver, PartKind, PartUpdate, ProviderBuiltin, ProviderScope,
    ReasoningConfig, ReasoningEffort, ReasoningSummary, ResolvedHandle, ToolChoice,
};
use crate::{Error, RawConfig, Response, StreamEvent};
use bytes::Bytes;
use futures_util::{Stream, StreamExt as _};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tracing::{debug, trace};

/// OpenAI provider implementation.
pub struct OpenAIProvider {
    transport: Transport,
    api_key: String,
    base_url: String,
    /// Optional `OpenAI-Organization` header value for multi-org keys.
    organization: Option<String>,
    /// Optional `OpenAI-Project` header value for project-scoped keys.
    project: Option<String>,
    /// Optional caller-held file registry for resolving `Ref` file inputs.
    file_resolver: Option<Arc<dyn FileResolver>>,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider with the default reqwest-backed transport.
    pub fn new(api_key: String) -> Result<Self, Error> {
        Ok(Self {
            transport: Transport::reqwest()?,
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
            organization: None,
            project: None,
            file_resolver: None,
        })
    }

    /// Create a new OpenAI provider with a custom base URL and the default transport.
    pub fn new_with_base_url(api_key: String, base_url: String) -> Result<Self, Error> {
        Ok(Self {
            transport: Transport::reqwest()?,
            api_key,
            base_url,
            organization: None,
            project: None,
            file_resolver: None,
        })
    }

    /// Create a new OpenAI provider with a caller-supplied transport. Lets
    /// downstream consumers (or tests) plug in a recording / replaying /
    /// retrying [`Transport`] without touching the rest of the provider.
    pub fn with_transport(api_key: String, base_url: String, transport: Transport) -> Self {
        Self {
            transport,
            api_key,
            base_url,
            organization: None,
            project: None,
            file_resolver: None,
        }
    }

    /// Attach an `OpenAI-Organization` header. Required for keys that
    /// have access to multiple organizations.
    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// Attach an `OpenAI-Project` header. Required for project-scoped keys.
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Attach a [`FileResolver`] so the provider can resolve
    /// [`FileSource::Ref`](crate::FileSource::Ref) file inputs —
    /// uploading them to `POST /v1/files` on a registry miss and referencing
    /// the resulting `file_id`.
    pub fn with_file_resolver(mut self, resolver: Arc<dyn FileResolver>) -> Self {
        self.file_resolver = Some(resolver);
        self
    }

    /// The [`ProviderScope`] handles minted by this client are valid within —
    /// the base URL plus any org/project scoping.
    fn scope(&self) -> ProviderScope {
        let mut account = self.base_url.clone();
        if let Some(org) = &self.organization {
            account.push('|');
            account.push_str(org);
        }
        if let Some(project) = &self.project {
            account.push('|');
            account.push_str(project);
        }
        ProviderScope::new(ProviderType::OpenAI, account)
    }

    /// Convert internal request to OpenAI Responses API format.
    ///
    /// `resolved` maps each file-`Ref` id to its wire-ready reference, built
    /// by the async [`resolve_refs`] pre-pass in [`Self::generate`].
    fn convert_request(
        &self,
        prompt: &crate::Prompt,
        config: &RawConfig,
        resolved: &HashMap<String, ResolvedRef>,
    ) -> ResponsesRequest {
        let messages = prompt.items();

        // Scan history for the latest InputItem::Continuation carrying
        // an OpenAI hint. Items at and before that index are elided —
        // the server already has them via `previous_response_id`.
        // Continuation markers for other providers are ignored.
        let (previous_response_id, start_index) = find_latest_openai_continuation(messages);

        let mut input: Vec<crate::providers::openai::types::OpenAIInputMessage> = Vec::new();
        for item in &messages[start_index..] {
            Self::flatten_input_item(item, &mut input, resolved);
        }

        ResponsesRequest {
            model: config.model.clone(),
            input,
            instructions: None,
            temperature: config.temperature,
            max_output_tokens: config.max_tokens,
            top_p: config.top_p,
            tools: config
                .tools
                .as_ref()
                .map(|tools| Self::convert_tools(tools)),
            tool_choice: config.tool_choice.as_ref().map(convert_tool_choice),
            parallel_tool_calls: config.parallel_tool_calls,
            previous_response_id,
            stream: None,
            store: Some(config.store.unwrap_or(false)),
            reasoning: config.reasoning.as_ref().map(convert_reasoning),
            stop: config.stop.clone(),
            presence_penalty: config.presence_penalty,
            frequency_penalty: config.frequency_penalty,
            prompt_cache_key: derive_prompt_cache_key(messages),
            text: config
                .response_format
                .as_ref()
                .and_then(convert_response_format),
        }
    }

    /// Flatten one canonical `InputItem` into one or more OpenAI input
    /// items. OpenAI's wire model puts function_call / function_call_output
    /// as siblings of message items, so an Assistant turn with mixed
    /// text + tool calls splits into a message + N function_call items
    /// here (preserving emit order).
    fn flatten_input_item(
        item: &crate::types::InputItem,
        out: &mut Vec<crate::providers::openai::types::OpenAIInputMessage>,
        resolved: &HashMap<String, ResolvedRef>,
    ) {
        use crate::providers::openai::types::OpenAIInputMessage;
        use crate::types::{AssistantPart, InputItem, UserPart};

        match item {
            InputItem::System(content) => {
                out.push(OpenAIInputMessage::Regular {
                    role: "system".to_string(),
                    content: crate::providers::openai::types::OpenAIMessageContent::Text(
                        content.clone(),
                    ),
                });
            }
            InputItem::User { content } => {
                use crate::providers::openai::types::OpenAIContentPart;
                // Build a content-parts list. Tool results become their own
                // top-level items; text and images become InputText /
                // InputImage parts. If we end up with just one text part
                // we collapse to a bare string for the common case.
                let mut parts: Vec<OpenAIContentPart> = Vec::new();
                for part in content {
                    match part {
                        UserPart::Text(s) => {
                            parts.push(OpenAIContentPart::InputText { text: s.clone() })
                        }
                        UserPart::Image(src) => match src {
                            crate::types::FileSource::Url(u) => {
                                parts.push(OpenAIContentPart::InputImage {
                                    image_url: Some(u.clone()),
                                    file_id: None,
                                });
                            }
                            crate::types::FileSource::Base64 { data, media_type } => {
                                parts.push(OpenAIContentPart::InputImage {
                                    image_url: Some(format!("data:{media_type};base64,{data}")),
                                    file_id: None,
                                });
                            }
                            crate::types::FileSource::Ref(id) => match resolved.get(id) {
                                Some(ResolvedRef::Handle { uri, .. }) => {
                                    parts.push(OpenAIContentPart::InputImage {
                                        image_url: None,
                                        file_id: Some(uri.clone()),
                                    });
                                }
                                Some(ResolvedRef::Url { uri, .. }) => {
                                    parts.push(OpenAIContentPart::InputImage {
                                        image_url: Some(uri.clone()),
                                        file_id: None,
                                    });
                                }
                                None => {
                                    tracing::debug!("OpenAI: unresolved image Ref {id}; dropping")
                                }
                            },
                        },
                        UserPart::ToolResult { call_id, content } => {
                            // A user turn mixing free text with a tool
                            // result (legitimate on Anthropic/Gemini,
                            // and how round-tripped history can look)
                            // is split here into ordered top-level
                            // items: message / function_call_output /
                            // message. This is NOT rejected — doing so
                            // would break a unified prompt only on
                            // OpenAI. Open question (needs live-API
                            // verification, can't be confirmed offline):
                            // whether the Responses API accepts that
                            // interleaving as-is or wants the
                            // function_call_output reordered ahead of
                            // trailing user text. Left as-is until
                            // verified rather than risk regressing a
                            // working path on an unverified assumption.
                            push_user_parts(out, &mut parts);
                            out.push(OpenAIInputMessage::FunctionCallOutput {
                                call_id: call_id.clone(),
                                output: flatten_user_parts_to_text(content),
                            });
                        }
                        UserPart::Audio(_) => {
                            // Rejected up front in generate() via
                            // reject_unsupported_modalities (the Responses API
                            // has no audio input — verified HTTP 400). Defensive
                            // drop for any direct convert_request caller.
                            tracing::debug!("OpenAI: dropping unsupported audio part");
                        }
                        UserPart::Document(src) => match src {
                            crate::types::FileSource::Url(u) => {
                                parts.push(OpenAIContentPart::InputFile {
                                    file_url: Some(u.clone()),
                                    file_data: None,
                                    file_id: None,
                                    filename: None,
                                });
                            }
                            crate::types::FileSource::Base64 { data, media_type } => {
                                parts.push(OpenAIContentPart::InputFile {
                                    file_url: None,
                                    file_data: Some(format!("data:{media_type};base64,{data}")),
                                    file_id: None,
                                    filename: None,
                                });
                            }
                            crate::types::FileSource::Ref(id) => match resolved.get(id) {
                                Some(ResolvedRef::Handle { uri, .. }) => {
                                    parts.push(OpenAIContentPart::InputFile {
                                        file_url: None,
                                        file_data: None,
                                        file_id: Some(uri.clone()),
                                        filename: None,
                                    });
                                }
                                Some(ResolvedRef::Url { uri, .. }) => {
                                    parts.push(OpenAIContentPart::InputFile {
                                        file_url: Some(uri.clone()),
                                        file_data: None,
                                        file_id: None,
                                        filename: None,
                                    });
                                }
                                None => {
                                    tracing::debug!(
                                        "OpenAI: unresolved document Ref {id}; dropping"
                                    )
                                }
                            },
                        },
                        UserPart::Video(_) => {
                            // Rejected up front in generate() (no video input on
                            // the Responses API). Defensive drop for any direct
                            // convert_request caller.
                            tracing::debug!("OpenAI: dropping unsupported video part");
                        }
                        UserPart::CacheBreakpoint => {
                            // OpenAI maps cache breakpoints onto a per-
                            // request `prompt_cache_key` rather than
                            // per-block markers; nothing to emit at the
                            // part level.
                        }
                    }
                }
                push_user_parts(out, &mut parts);
            }
            InputItem::Assistant { content } => {
                let mut buffered_text = String::new();
                for part in content {
                    match part {
                        AssistantPart::Text { content, .. } => {
                            if !buffered_text.is_empty() {
                                buffered_text.push('\n');
                            }
                            buffered_text.push_str(content);
                        }
                        AssistantPart::Refusal(s) => {
                            if !buffered_text.is_empty() {
                                buffered_text.push('\n');
                            }
                            buffered_text.push_str(s);
                        }
                        AssistantPart::ToolCall(call) => {
                            if !buffered_text.is_empty() {
                                out.push(OpenAIInputMessage::Regular {
                                    role: "assistant".to_string(),
                                    content:
                                        crate::providers::openai::types::OpenAIMessageContent::Text(
                                            std::mem::take(&mut buffered_text),
                                        ),
                                });
                            }
                            out.push(OpenAIInputMessage::FunctionCall {
                                call_id: call.call_id.clone(),
                                name: call.name.clone(),
                                arguments: call.arguments.clone(),
                            });
                        }
                        AssistantPart::Reasoning { .. }
                        | AssistantPart::RedactedReasoning { .. }
                        | AssistantPart::BuiltinToolCall { .. }
                        | AssistantPart::Continuation(_)
                        | AssistantPart::CacheBreakpoint => {
                            tracing::debug!(
                                "OpenAI provider dropping unsupported assistant part during request build"
                            );
                        }
                    }
                }
                if !buffered_text.is_empty() {
                    out.push(OpenAIInputMessage::Regular {
                        role: "assistant".to_string(),
                        content: crate::providers::openai::types::OpenAIMessageContent::Text(
                            buffered_text,
                        ),
                    });
                }
            }
        }
    }

    /// Convert our internal tools to OpenAI Responses API format.
    /// Builtin tools that OpenAI offers (`web_search`, `computer_use`)
    /// emit their typed wire shape; builtins OpenAI doesn't offer are
    /// silently dropped — model-switching contract.
    #[allow(clippy::ptr_arg)]
    fn convert_tools(tools: &[crate::types::Tool]) -> Vec<super::types::OpenAITool> {
        use crate::types::{ProviderBuiltin, Tool};
        let mut out = Vec::new();
        for tool in tools {
            match tool {
                Tool::Function(f) => out.push(super::types::OpenAITool::Function {
                    name: f.name.clone(),
                    description: f.description.clone().unwrap_or_default(),
                    parameters: f.parameters.clone(),
                }),
                Tool::Builtin(b) => match b {
                    ProviderBuiltin::WebSearch => {
                        out.push(super::types::OpenAITool::WebSearchPreview);
                    }
                    ProviderBuiltin::ComputerUse(cfg) => {
                        out.push(super::types::OpenAITool::ComputerUsePreview {
                            display_width: cfg.display_width,
                            display_height: cfg.display_height,
                            environment: cfg.environment.clone(),
                        });
                    }
                    ProviderBuiltin::GoogleSearch | ProviderBuiltin::CodeExecution => {
                        tracing::debug!(?b, "OpenAI provider dropping unsupported builtin tool");
                    }
                },
            }
        }
        out
    }
}

/// Map an OpenAI HTTP error response onto our [`Error`] variants.
///
/// OpenAI returns `{"error":{"message":..., "type":..., "code":...}}` on
/// failure. We parse it best-effort and pick a typed variant from the
/// HTTP status:
///
/// - 401 → [`Error::Auth`]
/// - 429 → [`Error::RateLimit`] (carries `Retry-After` if present)
/// - any other → [`Error::Provider`] with status, type, and message
///
/// The full body is preserved in the message so callers can still extract
/// the unparsed structured fields if they need them.
pub(crate) fn parse_openai_error(
    status: u16,
    retry_after_seconds: Option<u64>,
    body: &str,
) -> Error {
    #[derive(serde::Deserialize)]
    struct Outer<'a> {
        #[serde(borrow)]
        error: Option<Inner<'a>>,
    }
    #[derive(serde::Deserialize)]
    struct Inner<'a> {
        #[serde(default, borrow)]
        message: Option<&'a str>,
        #[serde(default, rename = "type", borrow)]
        kind: Option<&'a str>,
        #[serde(default, borrow)]
        code: Option<&'a str>,
    }
    let parsed = serde_json::from_str::<Outer>(body)
        .ok()
        .and_then(|o| o.error);
    let message = parsed
        .as_ref()
        .and_then(|e| e.message)
        .unwrap_or(body)
        .to_string();
    let kind = parsed.as_ref().and_then(|e| e.kind).unwrap_or("");
    let code = parsed.as_ref().and_then(|e| e.code).unwrap_or("");

    // Context-window detection: OpenAI reliably sets
    // `code: "context_length_exceeded"` for this case; surface as a
    // typed variant so callers driving long conversations can trigger
    // compaction without parsing strings.
    if code == "context_length_exceeded" {
        return Error::context_window_exceeded("OpenAI", format!("HTTP {status}: {message}"));
    }

    match status {
        401 => Error::auth_with_status(401, format!("OpenAI 401 ({kind} {code}): {message}")),
        429 => Error::rate_limit(
            retry_after_seconds,
            format!("OpenAI 429 ({kind} {code}): {message}"),
        ),
        _ => Error::provider_with_status(
            "OpenAI",
            status,
            format!("HTTP {status} ({kind} {code}): {message}"),
        ),
    }
}

/// Derive a stable cache key from the message prefix that precedes
/// the first [`crate::UserPart::CacheBreakpoint`]. Returns `None` when
/// no breakpoint is present (callers who don't opt into caching get
/// the OpenAI default of no key).
///
/// Uses `std::hash::DefaultHasher` (SipHash-1-3, fixed seed) — stable
/// within a single build of the consuming binary, which is the unit
/// of deployment that wants consistent cache hits. *Not* stable
/// across Rust/std versions; don't persist these keys.
fn derive_prompt_cache_key(messages: &[crate::types::InputItem]) -> Option<String> {
    use crate::types::{AssistantPart, InputItem, UserPart};
    use std::hash::{Hash, Hasher};

    // Common case: no breakpoint anywhere → no key, and skip hashing
    // the entire history (this runs on every request).
    let has_breakpoint = messages.iter().any(|item| match item {
        InputItem::User { content } => content
            .iter()
            .any(|p| matches!(p, UserPart::CacheBreakpoint)),
        InputItem::Assistant { content } => content
            .iter()
            .any(|p| matches!(p, AssistantPart::CacheBreakpoint)),
        InputItem::System(_) => false,
    });
    if !has_breakpoint {
        return None;
    }

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let mut saw_breakpoint = false;

    'outer: for item in messages {
        match item {
            InputItem::System(s) => {
                "system".hash(&mut hasher);
                s.hash(&mut hasher);
            }
            InputItem::User { content } => {
                "user".hash(&mut hasher);
                for part in content {
                    match part {
                        UserPart::Text(s) => s.hash(&mut hasher),
                        UserPart::Image(_)
                        | UserPart::Audio(_)
                        | UserPart::Document(_)
                        | UserPart::Video(_) => {
                            // Skip multi-modal payloads from the hash —
                            // their base64 representation would dominate
                            // and small re-encodings would defeat the key.
                            "<media>".hash(&mut hasher);
                        }
                        UserPart::ToolResult { call_id, content } => {
                            call_id.hash(&mut hasher);
                            for inner in content {
                                if let UserPart::Text(s) = inner {
                                    s.hash(&mut hasher);
                                }
                            }
                        }
                        UserPart::CacheBreakpoint => {
                            saw_breakpoint = true;
                            break 'outer;
                        }
                    }
                }
            }
            InputItem::Assistant { content } => {
                "assistant".hash(&mut hasher);
                for part in content {
                    match part {
                        AssistantPart::Text { content, .. } => content.hash(&mut hasher),
                        AssistantPart::ToolCall(call) => {
                            call.call_id.hash(&mut hasher);
                            call.name.hash(&mut hasher);
                            call.arguments.hash(&mut hasher);
                        }
                        AssistantPart::Reasoning { content, .. } => content.hash(&mut hasher),
                        AssistantPart::Refusal(s) => s.hash(&mut hasher),
                        AssistantPart::RedactedReasoning { data } => data.hash(&mut hasher),
                        AssistantPart::BuiltinToolCall {
                            kind,
                            arguments,
                            result,
                        } => {
                            // Provider-side; nothing the caller controls
                            // changes the cacheable prefix here, but
                            // hash the fields anyway so distinct calls
                            // don't collide.
                            format!("{kind:?}").hash(&mut hasher);
                            arguments.hash(&mut hasher);
                            result.hash(&mut hasher);
                        }
                        AssistantPart::Continuation(c) => {
                            "continuation".hash(&mut hasher);
                            format!("{c:?}").hash(&mut hasher);
                        }
                        AssistantPart::CacheBreakpoint => {
                            saw_breakpoint = true;
                            break 'outer;
                        }
                    }
                }
            }
        }
    }

    if saw_breakpoint {
        Some(format!("{:x}", hasher.finish()))
    } else {
        None
    }
}

/// Flush a buffered user-content-parts list as an
/// `OpenAIInputMessage::Regular`. Collapses single-text to the bare-
/// string content form (OpenAI accepts both, but the bare string is
/// the canonical shape and matches our pre-multi-modal request bodies).
fn push_user_parts(
    out: &mut Vec<crate::providers::openai::types::OpenAIInputMessage>,
    parts: &mut Vec<crate::providers::openai::types::OpenAIContentPart>,
) {
    use crate::providers::openai::types::{
        OpenAIContentPart, OpenAIInputMessage, OpenAIMessageContent,
    };
    if parts.is_empty() {
        return;
    }
    let drained: Vec<OpenAIContentPart> = std::mem::take(parts);
    // Single text part → bare string; anything else → parts array.
    if drained.len() == 1 {
        if let OpenAIContentPart::InputText { text } = &drained[0] {
            out.push(OpenAIInputMessage::Regular {
                role: "user".to_string(),
                content: OpenAIMessageContent::Text(text.clone()),
            });
            return;
        }
    }
    out.push(OpenAIInputMessage::Regular {
        role: "user".to_string(),
        content: OpenAIMessageContent::Parts(drained),
    });
}

use crate::providers::flatten_user_parts_to_text;

fn convert_response_format(
    rf: &crate::types::ResponseFormat,
) -> Option<crate::providers::openai::types::OpenAITextConfig> {
    use crate::providers::openai::types::{OpenAITextConfig, OpenAITextFormat};
    use crate::types::ResponseFormat;
    let format = match rf {
        ResponseFormat::Text => return None,
        ResponseFormat::JsonObject => OpenAITextFormat::JsonObject,
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => OpenAITextFormat::JsonSchema {
            name: name.clone(),
            schema: schema.clone(),
            strict: *strict,
        },
    };
    Some(OpenAITextConfig { format })
}

fn convert_reasoning(cfg: &ReasoningConfig) -> OpenAIReasoning {
    OpenAIReasoning {
        effort: cfg.effort.map(|e| match e {
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
        }),
        summary: cfg.summary.map(|s| match s {
            ReasoningSummary::Auto => "auto",
            ReasoningSummary::Concise => "concise",
            ReasoningSummary::Detailed => "detailed",
        }),
    }
}

/// Walk the history right-to-left for the most recent
/// [`InputItem::Assistant`] containing an
/// [`AssistantPart::Continuation`] of [`ProviderContinuation::OpenAI`].
/// Returns the response ID plus the index of the first item the
/// provider should actually send (one past the assistant turn that
/// carried the marker — the server already has that turn and
/// everything before it). Other providers' continuation parts are
/// transparently skipped.
fn find_latest_openai_continuation(
    messages: &[crate::types::InputItem],
) -> (Option<String>, usize) {
    use crate::types::{AssistantPart, InputItem, ProviderContinuation};
    for (i, item) in messages.iter().enumerate().rev() {
        if let InputItem::Assistant { content } = item {
            for part in content.iter().rev() {
                if let AssistantPart::Continuation(ProviderContinuation::OpenAI { response_id }) =
                    part
                {
                    return (Some(response_id.clone()), i + 1);
                }
            }
        }
    }
    (None, 0)
}

/// Map an OpenAI annotation onto the unified [`Annotation`] surface.
///
/// `Other` variants (forward-compat tag values we don't recognize) are
/// dropped rather than fabricated as broken citations.
fn map_openai_annotation(a: OpenAIAnnotation) -> Option<Annotation> {
    match a {
        OpenAIAnnotation::UrlCitation {
            start_index,
            end_index,
            url,
            title,
        } => Some(Annotation {
            kind: AnnotationKind::UrlCitation,
            start: start_index,
            end: end_index,
            source: url,
            title,
        }),
        OpenAIAnnotation::FileCitation {
            file_id,
            filename,
            index,
        } => Some(Annotation {
            kind: AnnotationKind::FileCitation,
            // Point anchor: OpenAI file citations carry a single
            // offset, not a span. Zero-width start == end.
            start: index,
            end: index,
            source: file_id,
            title: filename,
        }),
        OpenAIAnnotation::Other => None,
    }
}

fn convert_tool_choice(choice: &ToolChoice) -> OpenAIToolChoice {
    match choice {
        ToolChoice::Auto => OpenAIToolChoice::Mode("auto"),
        ToolChoice::None => OpenAIToolChoice::Mode("none"),
        ToolChoice::Required => OpenAIToolChoice::Mode("required"),
        ToolChoice::Function { name } => OpenAIToolChoice::Function {
            kind: "function",
            name: name.clone(),
        },
    }
}

/// Streaming state for an in-flight OpenAI response.
///
/// OpenAI's wire model has two-level nesting: top-level items
/// (`message`, `function_call`, `reasoning`) plus per-message content
/// parts (`output_text`, `refusal`). We map both to our flat part-index
/// space via `(output_index, content_index)` keys — `content_index` is
/// `None` for top-level items that don't have nested content parts.
pub(crate) struct OpenAIStreamState {
    tracker: crate::providers::part_tracker::PartTracker<(u32, Option<u32>)>,
    /// Whether we've already emitted the one-shot
    /// [`PartKind::Continuation`] for this response. OpenAI carries the
    /// `response.id` on `created` / `in_progress` / `completed` frames;
    /// we surface it once at end-of-stream so the marker lands *after*
    /// the assistant content in the resulting `AssistantPart` order.
    emitted_continuation: bool,
    /// Keys of `function_call` parts that received at least one
    /// `function_call_arguments.delta`. On `output_item.done` a key
    /// *not* in this set means the args never streamed incrementally
    /// — we reconcile from the item's complete `arguments` so they
    /// aren't silently lost.
    fn_args_streamed: std::collections::HashSet<(u32, Option<u32>)>,
}

impl OpenAIStreamState {
    pub(crate) fn new() -> Self {
        Self {
            tracker: crate::providers::part_tracker::PartTracker::new(),
            emitted_continuation: false,
            fn_args_streamed: std::collections::HashSet::new(),
        }
    }

    /// Open and immediately close a one-shot Continuation part. The
    /// PartTracker doesn't have a slot for keyless one-shot parts, so
    /// we go through `open_one_shot` instead.
    fn continuation_events(&mut self, response_id: &str) -> Vec<StreamEvent> {
        if self.emitted_continuation {
            return Vec::new();
        }
        self.emitted_continuation = true;
        self.tracker.open_one_shot(PartKind::Continuation(
            crate::types::ProviderContinuation::OpenAI {
                response_id: response_id.to_string(),
            },
        ))
    }

    /// Process one OpenAI wire event into 0 or more `StreamEvent`s.
    pub(crate) fn process(&mut self, event: OpenAIStreamEvent) -> Result<Vec<StreamEvent>, Error> {
        match event {
            OpenAIStreamEvent::Error { error } => {
                // OpenAI's Responses API returns 200 OK and emits the
                // context-length error inside the SSE stream as an
                // `event: error` frame with `code:
                // context_length_exceeded`. Detect it here so callers
                // driving long conversations get the typed
                // `ContextWindowExceeded` variant instead of a generic
                // streaming/provider error.
                if error.code.as_deref() == Some("context_length_exceeded") {
                    return Err(Error::context_window_exceeded(
                        "OpenAI",
                        format!("{}: {}", error.r#type, error.message),
                    ));
                }
                Err(Error::provider(
                    "OpenAI",
                    format!("{}: {}", error.r#type, error.message),
                ))
            }

            // `response.id` is stable across created/in_progress/
            // completed frames — emit the Continuation part at
            // end-of-stream (response.completed) so it lands after the
            // assistant content in the final part order.
            OpenAIStreamEvent::ResponseCreated | OpenAIStreamEvent::ResponseInProgress => {
                Ok(vec![])
            }

            OpenAIStreamEvent::OutputItemAdded { output_index, item } => {
                match item.r#type.as_str() {
                    "function_call" => {
                        let call_id = item.call_id.ok_or_else(|| {
                            Error::provider(
                                "OpenAI",
                                format!(
                                    "function_call item is missing call_id (item id: {})",
                                    item.id
                                ),
                            )
                        })?;
                        // Mirror the `call_id` handling: a missing
                        // name is a malformed item — error rather than
                        // fabricate `"unknown"`, which would only fail
                        // tool dispatch confusingly downstream.
                        let name = item.name.ok_or_else(|| {
                            Error::provider(
                                "OpenAI",
                                format!(
                                    "function_call item is missing name (item id: {})",
                                    item.id
                                ),
                            )
                        })?;
                        let (_idx, ev) = self
                            .tracker
                            .open((output_index, None), PartKind::ToolCall { call_id, name });
                        Ok(vec![ev])
                    }
                    "web_search_call" => {
                        let (_idx, ev) = self.tracker.open(
                            (output_index, None),
                            PartKind::BuiltinToolCall {
                                kind: ProviderBuiltin::WebSearch,
                            },
                        );
                        Ok(vec![ev])
                    }
                    // `reasoning` items contain one or more
                    // `reasoning_summary_part` children — each is its
                    // own Reasoning AssistantPart. `message` items
                    // contain content parts — each opens via
                    // `content_part.added`. Both outer items are
                    // wire-level wrappers; don't open a part for them.
                    "reasoning" | "message" => Ok(vec![]),
                    other => {
                        tracing::debug!(
                            item_type = other,
                            "OpenAI output_item.added with unhandled item type — open ignored"
                        );
                        Ok(vec![])
                    }
                }
            }

            OpenAIStreamEvent::OutputItemDone { output_index, item } => {
                let key = (output_index, None);
                let mut out = Vec::new();
                // For web_search_call items, emit the final action
                // payload as a Delta against the BuiltinToolCall part
                // before closing. That's the only point in the stream
                // where the queries arrive — `output_item.added` only
                // carries the in-progress shell.
                if let Some(idx) = self.tracker.index_of(&key) {
                    if item.r#type == "web_search_call" {
                        if let Some(action) = &item.action {
                            if let Ok(delta) = serde_json::to_string(action) {
                                out.push(StreamEvent::Delta { index: idx, delta });
                            }
                        }
                    }
                    // Reconcile function-call arguments: if no
                    // `function_call_arguments.delta` ever streamed for
                    // this part, the complete `arguments` arrives only
                    // here. Without this the call's arguments would be
                    // silently empty (short calls / wire variations).
                    if item.r#type == "function_call" && !self.fn_args_streamed.contains(&key) {
                        if let Some(args) = item.arguments.as_deref() {
                            if !args.is_empty() {
                                out.push(StreamEvent::Delta {
                                    index: idx,
                                    delta: args.to_string(),
                                });
                            }
                        }
                    }
                }
                if let Some(ev) = self.tracker.close(&key) {
                    out.push(ev);
                }
                Ok(out)
            }

            OpenAIStreamEvent::ContentPartAdded {
                output_index,
                content_index,
                part,
            } => {
                let kind = match part.r#type.as_str() {
                    "output_text" => PartKind::Text,
                    "refusal" => PartKind::Refusal,
                    other => {
                        tracing::warn!(
                            part_type = other,
                            "unknown content_part.added type — treating as text"
                        );
                        PartKind::Text
                    }
                };
                let (_idx, ev) = self.tracker.open((output_index, Some(content_index)), kind);
                Ok(vec![ev])
            }
            OpenAIStreamEvent::ContentPartDone {
                output_index,
                content_index,
            } => Ok(self
                .tracker
                .close(&(output_index, Some(content_index)))
                .into_iter()
                .collect()),

            OpenAIStreamEvent::ReasoningSummaryPartAdded {
                output_index,
                summary_index,
            } => {
                let (_idx, ev) = self
                    .tracker
                    .open((output_index, Some(summary_index)), PartKind::Reasoning);
                Ok(vec![ev])
            }
            OpenAIStreamEvent::ReasoningSummaryPartDone {
                output_index,
                summary_index,
            } => Ok(self
                .tracker
                .close(&(output_index, Some(summary_index)))
                .into_iter()
                .collect()),

            OpenAIStreamEvent::OutputTextDelta {
                output_index,
                content_index,
                delta,
            } => self.text_or_refusal_delta(output_index, content_index, delta, "output_text"),
            OpenAIStreamEvent::RefusalDelta {
                output_index,
                content_index,
                delta,
            } => self.text_or_refusal_delta(output_index, content_index, delta, "refusal"),

            OpenAIStreamEvent::ReasoningSummaryTextDelta {
                output_index,
                summary_index,
                delta,
            } => self.reasoning_delta(output_index, Some(summary_index), delta),
            OpenAIStreamEvent::ReasoningTextDelta {
                output_index,
                delta,
            } => self.reasoning_delta(output_index, None, delta),

            OpenAIStreamEvent::FunctionCallArgumentsDelta {
                output_index,
                delta,
            } => {
                if delta.is_empty() {
                    return Ok(vec![]);
                }
                let key = (output_index, None);
                let index = self.tracker.index_of(&key).ok_or_else(|| {
                    Error::streaming(format!(
                        "function_call_arguments.delta for unknown tool part {key:?}"
                    ))
                })?;
                self.fn_args_streamed.insert(key);
                Ok(vec![StreamEvent::Delta { index, delta }])
            }

            OpenAIStreamEvent::OutputTextAnnotationAdded {
                output_index,
                content_index,
                annotation,
            } => {
                let key = (output_index, Some(content_index));
                let index = self.tracker.index_of(&key).ok_or_else(|| {
                    Error::streaming(format!("annotation.added for unknown content part {key:?}"))
                })?;
                let Some(annotation) = map_openai_annotation(annotation) else {
                    return Ok(vec![]);
                };
                Ok(vec![StreamEvent::PartUpdate {
                    index,
                    update: PartUpdate::Annotation(annotation),
                }])
            }

            OpenAIStreamEvent::ResponseCompleted { response } => {
                let mut out = self.continuation_events(&response.id);
                let finish_reason = if response.output.iter().any(|o| o.r#type == "function_call") {
                    crate::types::FinishReason::ToolCalls
                } else {
                    crate::types::FinishReason::Stop
                };
                out.push(StreamEvent::Done {
                    finish_reason,
                    usage: response.usage.map(Into::into).unwrap_or_default(),
                });
                Ok(out)
            }
            OpenAIStreamEvent::ResponseIncomplete { response } => {
                let mut out = self.continuation_events(&response.id);
                let finish_reason = match response
                    .incomplete_details
                    .as_ref()
                    .map(|d| d.reason.as_str())
                {
                    Some("max_output_tokens") => crate::types::FinishReason::Length,
                    Some("content_filter") => crate::types::FinishReason::ContentFilter,
                    _ => crate::types::FinishReason::Stop,
                };
                out.push(StreamEvent::Done {
                    finish_reason,
                    usage: response.usage.map(Into::into).unwrap_or_default(),
                });
                Ok(out)
            }
            OpenAIStreamEvent::ResponseFailed { response, error } => {
                let message = response
                    .as_ref()
                    .and_then(|r| r.error.as_ref())
                    .map(|e| format!("{}: {}", e.r#type, e.message))
                    .or_else(|| {
                        error
                            .as_ref()
                            .map(|e| format!("{}: {}", e.r#type, e.message))
                    })
                    .unwrap_or_else(|| "response failed without error details".to_string());
                Err(Error::provider(
                    "OpenAI",
                    format!("response.failed — {message}"),
                ))
            }

            // Final-canonical-value / lifecycle frames whose payload
            // we don't need (we already accumulated via deltas, or
            // they're informational).
            OpenAIStreamEvent::OutputTextDone
            | OpenAIStreamEvent::ReasoningSummaryTextDone
            | OpenAIStreamEvent::ReasoningTextDone
            | OpenAIStreamEvent::RefusalDone
            | OpenAIStreamEvent::FunctionCallArgumentsDone
            | OpenAIStreamEvent::WebSearchCallInProgress
            | OpenAIStreamEvent::WebSearchCallSearching
            | OpenAIStreamEvent::WebSearchCallCompleted => Ok(vec![]),

            OpenAIStreamEvent::Unknown => {
                tracing::warn!(
                    "received an OpenAI stream event with an unrecognised `type` — \
                     ignoring. Inspect the captured `.response.sse` for the wire shape."
                );
                Ok(vec![])
            }
        }
    }

    fn text_or_refusal_delta(
        &self,
        output_index: u32,
        content_index: u32,
        delta: String,
        kind_label: &str,
    ) -> Result<Vec<StreamEvent>, Error> {
        if delta.is_empty() {
            return Ok(vec![]);
        }
        let key = (output_index, Some(content_index));
        let index = self.tracker.index_of(&key).ok_or_else(|| {
            Error::streaming(format!(
                "{kind_label}.delta for unknown content part {key:?}"
            ))
        })?;
        Ok(vec![StreamEvent::Delta { index, delta }])
    }

    fn reasoning_delta(
        &mut self,
        output_index: u32,
        summary_index: Option<u32>,
        delta: String,
    ) -> Result<Vec<StreamEvent>, Error> {
        if delta.is_empty() {
            return Ok(vec![]);
        }
        let key = (output_index, summary_index);
        if let Some(index) = self.tracker.index_of(&key) {
            return Ok(vec![StreamEvent::Delta { index, delta }]);
        }
        // The non-summary `response.reasoning_text.delta` channel has
        // no preceding part-added frame (unlike the summary channel,
        // whose part is opened by `ReasoningSummaryPartAdded`). Open
        // the Reasoning part lazily on first delta rather than
        // hard-aborting the whole stream. It's closed later by the
        // enclosing `reasoning` item's `OutputItemDone` (same
        // `(output_index, None)` key).
        if summary_index.is_none() {
            let (idx, start) = self.tracker.open(key, PartKind::Reasoning);
            return Ok(vec![start, StreamEvent::Delta { index: idx, delta }]);
        }
        Err(Error::streaming(format!(
            "reasoning delta for unknown reasoning part {key:?}"
        )))
    }
}

/// Multipart boundary for `POST /v1/files` uploads. A fixed token is fine —
/// the body is binary file content, and a collision with this exact ASCII
/// run is vanishingly unlikely.
const MULTIPART_BOUNDARY: &str = "platformedllmFormBoundary8x4mZqW2pT";

/// Best-effort filename (OpenAI requires one) derived from the MIME type.
fn filename_for(media_type: &str) -> String {
    match media_type_extension(media_type) {
        "" => "file.bin".to_string(),
        ext => format!("file.{ext}"),
    }
}

#[async_trait::async_trait]
impl ProviderUploader for OpenAIProvider {
    /// Stream `body` to `POST /v1/files` as `multipart/form-data` (never
    /// buffering it whole) and return the resulting `file_id`.
    ///
    /// The multipart framing is hand-rolled because the `reqwest` `multipart`
    /// feature isn't enabled and the library streams the file part. Best-effort
    /// against the documented `/v1/files` contract; needs live-API verification.
    async fn upload(
        &self,
        media_type: &str,
        content_length: Option<u64>,
        body: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>>,
    ) -> Result<ResolvedHandle, Error> {
        let boundary = MULTIPART_BOUNDARY;
        // Images referenced via `input_image` must be uploaded with
        // `purpose: "vision"`; documents (`input_file`) use `user_data`.
        let purpose = if media_type.starts_with("image/") {
            "vision"
        } else {
            "user_data"
        };
        let head = format!(
            "--{b}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\n{purpose}\r\n\
             --{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fname}\"\r\n\
             Content-Type: {mt}\r\n\r\n",
            b = boundary,
            fname = filename_for(media_type),
            mt = media_type,
        );
        let tail = format!("\r\n--{boundary}--\r\n");
        let head_b = Bytes::from(head.into_bytes());
        let tail_b = Bytes::from(tail.into_bytes());
        // A known body length yields a Content-Length; an unknown one (None)
        // falls back to chunked transfer-encoding, which `/v1/files` may reject
        // — resolvers should set `content_length` whenever they can.
        let total_len = content_length.map(|n| head_b.len() as u64 + n + tail_b.len() as u64);
        let stream_body = futures_util::stream::once(async move { Ok(head_b) })
            .chain(body)
            .chain(futures_util::stream::once(async move { Ok(tail_b) }));
        let stream_body: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>> =
            Box::pin(stream_body);

        let mut headers = vec![
            (
                "Authorization".to_string(),
                format!("Bearer {}", self.api_key),
            ),
            (
                "Content-Type".to_string(),
                format!("multipart/form-data; boundary={boundary}"),
            ),
        ];
        if let Some(org) = &self.organization {
            headers.push(("OpenAI-Organization".to_string(), org.clone()));
        }
        if let Some(project) = &self.project {
            headers.push(("OpenAI-Project".to_string(), project.clone()));
        }

        let req = UploadRequest {
            method: Method::Post,
            url: format!("{}/files", self.base_url),
            headers,
            content_length: total_len,
            body: stream_body,
        };
        let response = self.transport.send_upload(req).await?;
        let status = response.status;
        let retry_after = crate::transport::parse_retry_after(response.header("retry-after"));
        let bytes = response.collect_body().await.unwrap_or_default();
        if !(200..300).contains(&status) {
            let body_str = String::from_utf8_lossy(&bytes).into_owned();
            return Err(parse_openai_error(status, retry_after, &body_str));
        }

        #[derive(serde::Deserialize)]
        struct FileObj {
            id: String,
        }
        let obj: FileObj = serde_json::from_slice(&bytes)?;
        Ok(ResolvedHandle {
            uri: obj.id,
            media_type: media_type.to_string(),
            expires_at: None,
        })
    }
}

#[async_trait::async_trait]
impl Provider for OpenAIProvider {
    /// Generate a chat completion (internally always streams).
    async fn generate(
        &self,
        prompt: &crate::Prompt,
        config: &RawConfig,
    ) -> Result<Response, Error> {
        // The Responses API accepts only image / document inputs — reject
        // audio / video up front rather than dropping them.
        crate::providers::reject_unsupported_modalities(prompt.items(), "OpenAI", false, false)?;

        // Resolve any file `Ref`s to provider handles (uploading on a miss)
        // before the sync request build.
        let resolved = resolve_refs(
            prompt.items(),
            &self.scope(),
            self.file_resolver.as_deref(),
            self,
        )
        .await?;
        let mut openai_request = self.convert_request(prompt, config, &resolved);
        openai_request.stream = Some(true);

        debug!(
            model = %openai_request.model,
            messages = openai_request.input.len(),
            "sending OpenAI Responses API request"
        );
        trace!(
            request = ?openai_request,
            "full OpenAI request body"
        );

        let body = serde_json::to_vec(&openai_request)?;
        let mut headers = vec![
            (
                "Authorization".to_string(),
                format!("Bearer {}", self.api_key),
            ),
            ("Content-Type".to_string(), "application/json".to_string()),
        ];
        if let Some(org) = &self.organization {
            headers.push(("OpenAI-Organization".to_string(), org.clone()));
        }
        if let Some(project) = &self.project {
            headers.push(("OpenAI-Project".to_string(), project.clone()));
        }
        let req = TransportRequest {
            url: format!("{}/responses", self.base_url),
            headers,
            body,
        };
        let response = self.transport.send(req).await?;

        if !(200..300).contains(&response.status) {
            let status = response.status;
            let retry_after = crate::transport::parse_retry_after(response.header("retry-after"));
            let body_bytes = response.collect_body().await.unwrap_or_default();
            let body_str = String::from_utf8_lossy(&body_bytes).into_owned();
            return Err(parse_openai_error(status, retry_after, &body_str));
        }

        use crate::sse_stream::SseStreamExt;
        let state = Arc::new(Mutex::new(OpenAIStreamState::new()));
        let state_for_stream = state.clone();
        let event_stream = response
            .body
            .sse_events()
            .map(move |sse_result| -> Result<Vec<StreamEvent>, Error> {
                let sse_event = sse_result?;
                trace!(event = ?sse_event, "received OpenAI SSE event");
                let stream_event = serde_json::from_str::<OpenAIStreamEvent>(&sse_event.data)?;
                // A poisoned lock means `process` panicked on a prior
                // event; surface it as a stream error instead of
                // panicking this task too.
                let mut guard = state_for_stream
                    .lock()
                    .map_err(|_| Error::streaming("OpenAI stream state lock poisoned"))?;
                guard.process(stream_event)
            })
            .flat_map(|result| match result {
                Ok(events) => {
                    futures_util::stream::iter(events.into_iter().map(Ok).collect::<Vec<_>>())
                }
                Err(e) => futures_util::stream::iter(vec![Err(e)]),
            });

        // We can't read the continuation off the state until the stream
        // is fully consumed (response.completed sets it). That's fine for
        // most callers (they want the stream itself); buffer() picks up
        // the continuation when finalizing — see Response::with_continuation
        // for the wiring. We attach an empty continuation here; the
        // accumulator's finalize() can be extended in a follow-up to
        // poll the state at end-of-stream if we want this populated on
        // the streaming-only path too.
        let _ = state; // keep state alive (the closure also clones it)
        Ok(Response::from_stream(event_stream))
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::ResponseItem;
    use super::*;
    use crate::types::{Config, Prompt};

    #[test]
    fn test_provider_creation() {
        let provider = OpenAIProvider::new("test-key".to_string());
        assert!(provider.is_ok());
    }

    fn provider() -> OpenAIProvider {
        OpenAIProvider::new("k".to_string()).unwrap()
    }

    /// `generate()` rejects audio (and video) with a typed
    /// [`Error::UnsupportedInput`] before any network call — the Responses API
    /// can't take them.
    #[tokio::test]
    async fn generate_rejects_unsupported_audio_input() {
        use crate::types::{FileSource, InputItem, UserPart};
        let prompt = Prompt::new().with_item(InputItem::User {
            content: vec![UserPart::Audio(FileSource::Url(
                "http://x/a.mp3".to_string(),
            ))],
        });
        let cfg = Config::builder("gpt-4o-mini").build();
        let err = match provider().generate(&prompt, cfg.raw()).await {
            Ok(_) => panic!("audio is unsupported on the Responses API"),
            Err(e) => e,
        };
        assert!(
            matches!(
                err,
                Error::UnsupportedInput {
                    provider: "OpenAI",
                    modality: "audio"
                }
            ),
            "got: {err:?}"
        );
    }

    /// HTTP 429 with an OpenAI-shaped error body should produce
    /// [`Error::RateLimit`] (not the generic [`Error::Provider`]) so
    /// retry-aware callers can branch on it.
    #[test]
    fn http_429_maps_to_rate_limit() {
        let body = r#"{"error":{"message":"Rate limited","type":"rate_limit_error","code":"rate_limit_exceeded"}}"#;
        let err = parse_openai_error(429, Some(30), body);
        match err {
            Error::RateLimit {
                retry_after,
                message,
            } => {
                assert_eq!(retry_after, Some(std::time::Duration::from_secs(30)));
                assert!(message.contains("Rate limited"));
                assert!(message.contains("rate_limit_error"));
            }
            other => panic!("expected RateLimit, got {other:?}"),
        }
    }

    /// OpenAI reliably sets `code: "context_length_exceeded"` for
    /// over-budget prompts — surface that as the typed
    /// [`Error::ContextWindowExceeded`] so long-conversation callers
    /// can branch without parsing strings.
    #[test]
    fn http_400_context_length_exceeded_is_typed() {
        let body = r#"{"error":{"message":"This model's maximum context length is 128000 tokens.","type":"invalid_request_error","code":"context_length_exceeded"}}"#;
        let err = parse_openai_error(400, None, body);
        match err {
            Error::ContextWindowExceeded { provider, message } => {
                assert_eq!(provider, "OpenAI");
                assert!(message.contains("maximum context length"));
            }
            other => panic!("expected ContextWindowExceeded, got {other:?}"),
        }
    }

    /// OpenAI's Responses API doesn't always return a 4xx for
    /// over-budget prompts — it can return HTTP 200 OK and emit the
    /// failure inside the SSE stream as an `event: error` with
    /// `code: "context_length_exceeded"`. The in-stream branch of
    /// `OpenAIStreamState::process` must produce the same typed
    /// `ContextWindowExceeded` variant as the HTTP-400 path.
    #[test]
    fn in_stream_context_length_exceeded_is_typed() {
        use crate::providers::openai::types::ErrorDetails;
        let mut state = OpenAIStreamState::new();
        let err = state
            .process(OpenAIStreamEvent::Error {
                error: ErrorDetails {
                    message: "Your input exceeds the context window of this model.".to_string(),
                    r#type: "invalid_request_error".to_string(),
                    param: Some("input".to_string()),
                    code: Some("context_length_exceeded".to_string()),
                },
            })
            .expect_err("Error event must produce an Err");
        match err {
            Error::ContextWindowExceeded { provider, message } => {
                assert_eq!(provider, "OpenAI");
                assert!(message.contains("context window"));
            }
            other => panic!("expected ContextWindowExceeded, got {other:?}"),
        }
    }

    /// In-stream `Error` events with codes *other than*
    /// `context_length_exceeded` must still fall through to the
    /// generic `Error::Provider` path — the typed variant is reserved
    /// for the one signal compaction callers care about.
    #[test]
    fn in_stream_unrelated_error_stays_generic_provider() {
        use crate::providers::openai::types::ErrorDetails;
        let mut state = OpenAIStreamState::new();
        let err = state
            .process(OpenAIStreamEvent::Error {
                error: ErrorDetails {
                    message: "model overloaded".to_string(),
                    r#type: "server_error".to_string(),
                    param: None,
                    code: Some("server_overloaded".to_string()),
                },
            })
            .expect_err("Error event must produce an Err");
        match err {
            Error::Provider {
                provider: "OpenAI", ..
            } => {}
            other => panic!("expected Provider, got {other:?}"),
        }
    }

    #[test]
    fn http_401_maps_to_auth() {
        let body = r#"{"error":{"message":"Bad key","type":"invalid_request_error","code":"invalid_api_key"}}"#;
        let err = parse_openai_error(401, None, body);
        assert!(
            matches!(
                err,
                Error::Auth {
                    status: Some(401),
                    ..
                }
            ),
            "got {err:?}"
        );
        assert!(format!("{err}").contains("Bad key"));
    }

    /// Non-JSON / non-conforming bodies still produce a useful error rather
    /// than swallowing the status code.
    #[test]
    fn unparseable_error_body_still_carries_status_and_body() {
        let err = parse_openai_error(500, None, "<html>500 Server Error</html>");
        match &err {
            Error::Provider { message, .. } => {
                assert!(message.contains("500"));
                assert!(message.contains("<html>"));
            }
            other => panic!("expected Provider, got {other:?}"),
        }
    }

    /// `tool_choice` must serialize to OpenAI's expected wire forms:
    /// the bare strings `"auto"` / `"none"` / `"required"` for modes, and
    /// `{"type":"function","name":"…"}` for a forced specific tool.
    #[test]
    fn tool_choice_serializes_modes_as_strings() {
        let prompt = Prompt::user("hi");
        for (choice, expected) in [
            (ToolChoice::Auto, serde_json::json!("auto")),
            (ToolChoice::None, serde_json::json!("none")),
            (ToolChoice::Required, serde_json::json!("required")),
        ] {
            let cfg = Config::builder("gpt-4").tool_choice(choice.clone()).build();
            let req =
                provider().convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new());
            let json = serde_json::to_value(&req).unwrap();
            assert_eq!(
                json["tool_choice"], expected,
                "ToolChoice::{choice:?} should serialize to {expected}",
            );
        }
    }

    #[test]
    fn tool_choice_serializes_function_as_typed_object() {
        let prompt = Prompt::user("hi");
        let cfg = Config::builder("gpt-4")
            .tool_choice(ToolChoice::Function {
                name: "get_weather".to_string(),
            })
            .build();
        let req = provider().convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new());
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(
            json["tool_choice"],
            serde_json::json!({"type": "function", "name": "get_weather"}),
        );
    }

    /// `reasoning` configuration must reach the wire as
    /// `{"effort": "...", "summary": "..."}`. Both fields are optional.
    #[test]
    fn reasoning_config_serializes_to_correct_shape() {
        use crate::types::{ReasoningConfig, ReasoningEffort, ReasoningSummary};
        let prompt = Prompt::user("hi");
        let cfg = Config::builder("gpt-5")
            .reasoning(ReasoningConfig {
                effort: Some(ReasoningEffort::High),
                summary: Some(ReasoningSummary::Auto),
            })
            .build();
        let req = provider().convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new());
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(
            json["reasoning"],
            serde_json::json!({"effort": "high", "summary": "auto"}),
        );
    }

    /// OpenAI's reasoning streaming should open one Reasoning part per
    /// `reasoning_summary_part.added` (not per outer `reasoning` item),
    /// since a single reasoning item often emits multiple summaries.
    #[test]
    fn reasoning_summary_text_delta_routes_to_reasoning_part() {
        let mut state = OpenAIStreamState::new();
        // Outer reasoning item — no part opens here.
        let added: OpenAIStreamEvent = serde_json::from_str(
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1"}}"#,
        )
        .unwrap();
        assert!(state.process(added).unwrap().is_empty());

        // First summary opens.
        let summary_added: OpenAIStreamEvent = serde_json::from_str(
            r#"{"type":"response.reasoning_summary_part.added","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}}"#,
        )
        .unwrap();
        let events = state.process(summary_added).unwrap();
        assert!(
            matches!(
                &events[0],
                StreamEvent::PartStart {
                    index: 0,
                    kind: PartKind::Reasoning
                }
            ),
            "expected PartStart(Reasoning), got {:?}",
            events,
        );

        let delta: OpenAIStreamEvent = serde_json::from_str(
            r#"{"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":"hmm,"}"#,
        )
        .unwrap();
        let events = state.process(delta).unwrap();
        match &events[0] {
            StreamEvent::Delta { index, delta } => {
                assert_eq!(*index, 0);
                assert_eq!(delta, "hmm,");
            }
            other => panic!("expected Delta, got {other:?}"),
        }
    }

    /// `parallel_tool_calls` and `store` should reach the wire when the
    /// caller sets them, and stay absent when not. Default `store` is false
    /// (opt-out) so we don't unintentionally retain prompts server-side.
    #[test]
    fn parallel_tool_calls_and_store_are_caller_controlled() {
        let prompt = Prompt::user("hi");
        let cfg = Config::builder("gpt-4")
            .parallel_tool_calls(false)
            .store(true)
            .build();
        let req = provider().convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new());
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["parallel_tool_calls"], false);
        assert_eq!(json["store"], true);
    }

    /// A `response.output_item.added` for a `function_call` opens a
    /// `ToolCall` part carrying the API's `call_id` — not the `fc_…`
    /// output item id. The two are not interchangeable.
    #[test]
    fn function_call_opens_with_api_call_id() {
        let mut state = OpenAIStreamState::new();
        let json = r#"{
            "type":"response.output_item.added",
            "output_index":0,
            "item":{
                "type":"function_call",
                "id":"fc_123",
                "name":"get_weather",
                "arguments":"",
                "call_id":"call_abc"
            }
        }"#;
        let event: OpenAIStreamEvent = serde_json::from_str(json).unwrap();
        let events = state.process(event).unwrap();
        match &events[0] {
            StreamEvent::PartStart {
                index: 0,
                kind: PartKind::ToolCall { call_id, name },
            } => {
                assert_eq!(call_id, "call_abc");
                assert_eq!(name, "get_weather");
            }
            other => panic!("expected PartStart(ToolCall), got {other:?}"),
        }
    }

    /// A function_call item with no `call_id` is malformed per the
    /// Responses API — silently substituting the `fc_*` id breaks
    /// multi-turn tool calls invisibly. Surface it as an error.
    #[test]
    fn function_call_added_without_call_id_errors() {
        let mut state = OpenAIStreamState::new();
        let json = r#"{
            "type":"response.output_item.added",
            "output_index":0,
            "item":{
                "type":"function_call",
                "id":"fc_123",
                "name":"get_weather",
                "arguments":""
            }
        }"#;
        let event: OpenAIStreamEvent = serde_json::from_str(json).unwrap();
        let result = state.process(event);
        assert!(
            result.is_err(),
            "function_call without call_id must error, got: {result:?}",
        );
    }

    /// A `Continuation` part inside an assistant turn threads through
    /// as `previous_response_id` *and* elides that assistant turn plus
    /// every item before it.
    #[test]
    fn openai_continuation_elides_prior_history() {
        use crate::types::{InputItem, ProviderContinuation};
        let prompt = Prompt::user("first turn")
            .with_assistant("first answer")
            .with_item(InputItem::assistant_continuation(
                ProviderContinuation::OpenAI {
                    response_id: "resp_1".to_string(),
                },
            ))
            .with_user("follow-up");
        let cfg = Config::builder("gpt-5").build();
        let body =
            provider().convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new());
        assert_eq!(body.previous_response_id.as_deref(), Some("resp_1"));
        // Only the items after the assistant turn carrying the
        // continuation reach the wire.
        assert_eq!(body.input.len(), 1);
    }

    /// Full roundtrip: a `CompleteResponse` from a prior turn, folded
    /// into the next prompt via `with_response()`, should have its
    /// continuation picked up and prior history elided automatically —
    /// no caller-side bookkeeping required.
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
                AssistantPart::Continuation(ProviderContinuation::OpenAI {
                    response_id: "resp_prior".into(),
                }),
            ],
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        };
        let prompt = Prompt::user("first turn")
            .with_response(&prior)
            .with_user("follow-up");
        let cfg = Config::builder("gpt-5").build();
        let body =
            provider().convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new());
        assert_eq!(body.previous_response_id.as_deref(), Some("resp_prior"));
        // Only the follow-up reaches the wire — everything else is
        // covered by the server-side response state.
        assert_eq!(body.input.len(), 1);
    }

    /// The *most recent* matching continuation wins; older markers are
    /// superseded.
    #[test]
    fn latest_openai_continuation_wins() {
        use crate::types::{InputItem, ProviderContinuation};
        let prompt = Prompt::user("a")
            .with_item(InputItem::assistant_continuation(
                ProviderContinuation::OpenAI {
                    response_id: "resp_old".to_string(),
                },
            ))
            .with_user("b")
            .with_item(InputItem::assistant_continuation(
                ProviderContinuation::OpenAI {
                    response_id: "resp_new".to_string(),
                },
            ))
            .with_user("c");
        let cfg = Config::builder("gpt-5").build();
        let body =
            provider().convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new());
        assert_eq!(body.previous_response_id.as_deref(), Some("resp_new"));
        // Only items strictly after the latest matching assistant turn.
        assert_eq!(body.input.len(), 1);
    }

    /// A Gemini-flavored continuation part in the history is ignored
    /// by OpenAI and does *not* elide history — the model-switching
    /// contract.
    #[test]
    fn gemini_continuation_ignored_by_openai() {
        use crate::types::{InputItem, ProviderContinuation};
        let prompt = Prompt::user("a")
            .with_item(InputItem::assistant_continuation(
                ProviderContinuation::Gemini {
                    cached_content: "cached/x".to_string(),
                },
            ))
            .with_user("b");
        let cfg = Config::builder("gpt-5").build();
        let body =
            provider().convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new());
        assert!(body.previous_response_id.is_none());
        // Both user items still on the wire (continuation part drops out).
        assert_eq!(body.input.len(), 2);
    }

    #[test]
    fn computer_use_builtin_carries_config_on_openai() {
        use crate::types::{ComputerUseConfig, ProviderBuiltin, Tool};
        let prompt = Prompt::user("hi");
        let cfg = Config::builder("gpt-5")
            .tools(vec![Tool::builtin(ProviderBuiltin::ComputerUse(
                ComputerUseConfig {
                    display_width: 1280,
                    display_height: 800,
                    environment: "browser".to_string(),
                },
            ))])
            .build();
        let body =
            provider().convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new());
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["tools"][0]["type"], "computer_use_preview");
        assert_eq!(json["tools"][0]["display_width"], 1280);
        assert_eq!(json["tools"][0]["display_height"], 800);
        assert_eq!(json["tools"][0]["environment"], "browser");
    }

    #[test]
    fn response_format_json_object_emits_text_format() {
        use crate::types::ResponseFormat;
        let prompt = Prompt::user("hi");
        let cfg = Config::builder("gpt-5")
            .response_format(ResponseFormat::JsonObject)
            .build();
        let body =
            provider().convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new());
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["text"]["format"]["type"], "json_object");
    }

    #[test]
    fn response_format_json_schema_emits_schema_block() {
        use crate::types::ResponseFormat;
        use std::borrow::Cow;
        let schema_raw =
            serde_json::value::RawValue::from_string(r#"{"type":"object"}"#.to_string()).unwrap();
        let prompt = Prompt::user("hi");
        let cfg = Config::builder("gpt-5")
            .response_format(ResponseFormat::JsonSchema {
                name: "Point".to_string(),
                schema: Cow::Owned(schema_raw),
                strict: true,
            })
            .build();
        let body =
            provider().convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new());
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["text"]["format"]["type"], "json_schema");
        assert_eq!(json["text"]["format"]["name"], "Point");
        assert_eq!(json["text"]["format"]["strict"], true);
    }

    /// `UserPart::CacheBreakpoint` should produce a stable
    /// `prompt_cache_key` derived from the prefix bytes — same prefix
    /// → same key, so OpenAI can group cacheable prefixes server-side.
    #[test]
    fn cache_breakpoint_derives_stable_prompt_cache_key() {
        use crate::types::{InputItem, UserPart};
        let make_prompt = || {
            let mut p = Prompt::system("system instructions");
            p = p.with_item(InputItem::User {
                content: vec![
                    UserPart::Text("cached context".into()),
                    UserPart::CacheBreakpoint,
                    UserPart::Text("variable suffix".into()),
                ],
            });
            p
        };
        let prompt1 = make_prompt();
        let prompt2 = make_prompt();
        let cfg = Config::builder("gpt-5").build();
        let req1 =
            provider().convert_request(&prompt1, cfg.raw(), &std::collections::HashMap::new());
        let req2 =
            provider().convert_request(&prompt2, cfg.raw(), &std::collections::HashMap::new());
        assert!(req1.prompt_cache_key.is_some());
        assert_eq!(req1.prompt_cache_key, req2.prompt_cache_key);
    }

    #[test]
    fn no_cache_breakpoint_means_no_prompt_cache_key() {
        let prompt = Prompt::user("hi");
        let cfg = Config::builder("gpt-5").build();
        let req = provider().convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new());
        assert!(req.prompt_cache_key.is_none());
    }

    /// Different prefixes BEFORE the breakpoint must produce different
    /// keys; otherwise OpenAI would group unrelated requests.
    #[test]
    fn different_prefix_produces_different_key() {
        use crate::types::{InputItem, UserPart};
        let make_prompt = |prefix: &str| {
            Prompt::system(prefix).with_item(InputItem::User {
                content: vec![UserPart::Text("ctx".into()), UserPart::CacheBreakpoint],
            })
        };
        let cfg = Config::builder("gpt-5").build();
        let p1 = make_prompt("system one");
        let p2 = make_prompt("system two");
        let k1 = provider()
            .convert_request(&p1, cfg.raw(), &std::collections::HashMap::new())
            .prompt_cache_key;
        let k2 = provider()
            .convert_request(&p2, cfg.raw(), &std::collections::HashMap::new())
            .prompt_cache_key;
        assert!(k1.is_some());
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_request_conversion() {
        let provider = OpenAIProvider::new("test-key".to_string()).unwrap();
        let prompt = Prompt::user("Hello");
        let cfg = Config::builder("gpt-4")
            .temperature(0.7)
            .max_tokens(100)
            .build();
        let openai_request =
            provider.convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new());
        assert_eq!(openai_request.model, "gpt-4");
        assert_eq!(openai_request.temperature, Some(0.7));
        assert_eq!(openai_request.max_output_tokens, Some(100));
    }

    #[test]
    fn cache_key_is_none_without_breakpoint() {
        // The common (no-breakpoint) path short-circuits to None.
        let p = Prompt::system("sys").with_user("hello");
        let cfg = Config::builder("gpt-5").build();
        let k = provider()
            .convert_request(&p, cfg.raw(), &std::collections::HashMap::new())
            .prompt_cache_key;
        assert_eq!(k, None);
    }

    /// A resolved document `Ref` lands as an `input_file` referencing the
    /// uploaded `file_id`; an image `Ref` as an `input_image` with `file_id`.
    #[test]
    fn resolved_refs_emit_file_id_wire_shapes() {
        use crate::providers::file_resolve::ResolvedRef;
        use crate::types::{FileSource, InputItem, UserPart};

        let prompt = Prompt::new().with_item(InputItem::User {
            content: vec![
                UserPart::Document(FileSource::Ref("doc1".into())),
                UserPart::Image(FileSource::Ref("img1".into())),
            ],
        });
        let mut resolved = std::collections::HashMap::new();
        resolved.insert(
            "doc1".to_string(),
            ResolvedRef::Handle {
                uri: "file-doc".into(),
                media_type: "application/pdf".into(),
            },
        );
        resolved.insert(
            "img1".to_string(),
            ResolvedRef::Handle {
                uri: "file-img".into(),
                media_type: "image/png".into(),
            },
        );
        let cfg = Config::builder("gpt-5").build();
        let req = provider().convert_request(&prompt, cfg.raw(), &resolved);
        let json = serde_json::to_value(&req).unwrap();
        let parts = &json["input"][0]["content"];
        assert_eq!(parts[0]["type"], "input_file");
        assert_eq!(parts[0]["file_id"], "file-doc");
        assert_eq!(parts[1]["type"], "input_image");
        assert_eq!(parts[1]["file_id"], "file-img");
    }

    /// A resolved `Ref` that came back as a plain URL falls back to the URL
    /// wire form (no upload handle).
    #[test]
    fn resolved_url_ref_falls_back_to_url_forms() {
        use crate::providers::file_resolve::ResolvedRef;
        use crate::types::{FileSource, InputItem, UserPart};

        let prompt = Prompt::new().with_item(InputItem::User {
            content: vec![UserPart::Document(FileSource::Ref("doc1".into()))],
        });
        let mut resolved = std::collections::HashMap::new();
        resolved.insert(
            "doc1".to_string(),
            ResolvedRef::Url {
                uri: "https://example.com/x.pdf".into(),
                media_type: "application/pdf".into(),
            },
        );
        let cfg = Config::builder("gpt-5").build();
        let req = provider().convert_request(&prompt, cfg.raw(), &resolved);
        let json = serde_json::to_value(&req).unwrap();
        let part = &json["input"][0]["content"][0];
        assert_eq!(part["type"], "input_file");
        assert_eq!(part["file_url"], "https://example.com/x.pdf");
        assert!(part["file_id"].is_null());
    }

    fn fn_item(call_id: Option<&str>, name: Option<&str>, arguments: Option<&str>) -> ResponseItem {
        ResponseItem {
            r#type: "function_call".to_string(),
            id: "fc_1".to_string(),
            name: name.map(str::to_string),
            call_id: call_id.map(str::to_string),
            action: None,
            arguments: arguments.map(str::to_string),
        }
    }

    #[test]
    fn reasoning_text_delta_opens_part_lazily_no_abort() {
        // H5: the non-summary reasoning_text channel has no
        // part-added frame; the first delta must open a Reasoning
        // part instead of hard-aborting the stream.
        let mut st = OpenAIStreamState::new();
        let evs = st
            .process(OpenAIStreamEvent::ReasoningTextDelta {
                output_index: 0,
                delta: "thinking…".to_string(),
            })
            .expect("must not error");
        assert!(
            matches!(
                evs[0],
                StreamEvent::PartStart {
                    kind: PartKind::Reasoning,
                    ..
                }
            ),
            "expected lazy PartStart(Reasoning), got {evs:#?}"
        );
        assert!(matches!(evs[1], StreamEvent::Delta { .. }));
    }

    #[test]
    fn function_call_args_reconciled_from_done_when_no_deltas() {
        // H4: no function_call_arguments.delta streamed → the
        // complete arguments on output_item.done must still surface.
        let mut st = OpenAIStreamState::new();
        st.process(OpenAIStreamEvent::OutputItemAdded {
            output_index: 0,
            item: fn_item(Some("call_1"), Some("get_weather"), None),
        })
        .unwrap();
        let evs = st
            .process(OpenAIStreamEvent::OutputItemDone {
                output_index: 0,
                item: fn_item(
                    Some("call_1"),
                    Some("get_weather"),
                    Some(r#"{"city":"Paris"}"#),
                ),
            })
            .unwrap();
        let arg_delta = evs.iter().find_map(|e| match e {
            StreamEvent::Delta { delta, .. } => Some(delta.as_str()),
            _ => None,
        });
        assert_eq!(arg_delta, Some(r#"{"city":"Paris"}"#));
    }

    #[test]
    fn function_call_args_not_duplicated_when_streamed() {
        // If args *did* stream, output_item.done must NOT re-emit them.
        let mut st = OpenAIStreamState::new();
        st.process(OpenAIStreamEvent::OutputItemAdded {
            output_index: 0,
            item: fn_item(Some("call_1"), Some("f"), None),
        })
        .unwrap();
        st.process(OpenAIStreamEvent::FunctionCallArgumentsDelta {
            output_index: 0,
            delta: r#"{"a":1}"#.to_string(),
        })
        .unwrap();
        let evs = st
            .process(OpenAIStreamEvent::OutputItemDone {
                output_index: 0,
                item: fn_item(Some("call_1"), Some("f"), Some(r#"{"a":1}"#)),
            })
            .unwrap();
        assert!(
            !evs.iter().any(|e| matches!(e, StreamEvent::Delta { .. })),
            "args already streamed; done must not re-emit them: {evs:#?}"
        );
    }

    #[test]
    fn function_call_missing_name_errors() {
        let mut st = OpenAIStreamState::new();
        let err = st
            .process(OpenAIStreamEvent::OutputItemAdded {
                output_index: 0,
                item: fn_item(Some("call_1"), None, None),
            })
            .expect_err("missing name must error, not fabricate 'unknown'");
        assert!(err.to_string().contains("missing name"), "{err}");
    }
}
