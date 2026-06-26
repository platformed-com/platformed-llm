use std::collections::HashMap;
use std::sync::Arc;

use futures_util::StreamExt;

use super::anthropic_types::*;
use super::endpoint::VertexEndpoint;
use crate::factory::ProviderType;
use crate::provider::Provider;
use crate::providers::file_resolve::{resolve_refs, NoLibraryUpload, ResolvedRef};
use crate::sse_stream::SseStream;
use crate::transport::{Transport, TransportRequest};
use crate::types::{
    AssistantPart, FileResolver, FinishReason, InputItem, PartKind, PartUpdate, ProviderScope,
    ReasoningEffort, Usage, UserPart,
};
use crate::{Error, RawConfig, Response, StreamEvent};

/// Anthropic Claude provider implementation via Vertex AI.
pub struct AnthropicViaVertexProvider {
    endpoint: VertexEndpoint,
    transport: Transport,
    /// Comma-separated `anthropic-beta` header values. Used to opt into
    /// beta features (computer use, fine-grained tool streaming, etc.).
    beta: Vec<String>,
    /// Optional caller-held file registry for resolving `Ref` file inputs.
    file_resolver: Option<Arc<dyn FileResolver>>,
    /// Cooperative rate limiter consulted before every send.
    rate_limiter: crate::rate_limit::SharedRateLimiter,
}

impl AnthropicViaVertexProvider {
    /// Create a new Anthropic provider with access token authentication.
    pub fn new(project_id: String, location: String, access_token: String) -> Result<Self, Error> {
        Ok(Self {
            endpoint: VertexEndpoint::with_access_token(project_id, location, access_token),
            transport: Transport::reqwest()?,
            beta: Vec::new(),
            file_resolver: None,
            rate_limiter: crate::rate_limit::default_shared_limiter(),
        })
    }

    /// Create a new Anthropic provider with a custom base URL (for testing).
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
            beta: Vec::new(),
            file_resolver: None,
            rate_limiter: crate::rate_limit::default_shared_limiter(),
        })
    }

    /// Create a new Anthropic provider with Application Default Credentials.
    pub async fn with_adc(project_id: String, location: String) -> Result<Self, Error> {
        Ok(Self {
            endpoint: VertexEndpoint::with_adc(project_id, location).await?,
            transport: Transport::reqwest()?,
            beta: Vec::new(),
            file_resolver: None,
            rate_limiter: crate::rate_limit::default_shared_limiter(),
        })
    }

    /// Create a new Anthropic provider with a caller-supplied [`Transport`]
    /// and pre-built [`VertexEndpoint`].
    pub fn with_transport(endpoint: VertexEndpoint, transport: Transport) -> Self {
        Self {
            endpoint,
            transport,
            beta: Vec::new(),
            file_resolver: None,
            rate_limiter: crate::rate_limit::default_shared_limiter(),
        }
    }

    /// Attach a shared [`crate::rate_limit::RateLimiter`]. See the
    /// equivalent method on the OpenAI provider for the model — same
    /// trait, same semantics.
    pub fn with_rate_limiter(mut self, limiter: crate::rate_limit::SharedRateLimiter) -> Self {
        self.rate_limiter = limiter;
        self
    }

    /// Swap the static access token before it expires (GCP tokens
    /// last ~1h). Errors if this provider was built with ADC, which
    /// refreshes automatically. See [`VertexEndpoint::set_access_token`].
    pub fn set_access_token(&self, token: impl Into<String>) -> Result<(), Error> {
        self.endpoint.set_access_token(token)
    }

    /// Opt into Anthropic beta features. Each `beta_id` (e.g.
    /// `"computer-use-2025-01-24"`) appears as a comma-separated value
    /// in the `anthropic-beta` header.
    pub fn with_beta(mut self, beta_ids: impl IntoIterator<Item = String>) -> Self {
        self.beta.extend(beta_ids);
        self
    }

    /// Attach a [`FileResolver`] so the provider can resolve
    /// [`FileSource::Ref`](crate::FileSource::Ref) file inputs.
    ///
    /// Anthropic-via-Vertex has no library-owned file store, so the resolver
    /// should return a durable provider handle via
    /// [`ResolvedFile::ProviderHandle`](crate::ResolvedFile::ProviderHandle)
    /// (or, for a genuinely public file, a public URL via
    /// [`ResolvedFile::Url`](crate::ResolvedFile::Url)); a streaming payload is
    /// rejected.
    pub fn with_file_resolver(mut self, resolver: Arc<dyn FileResolver>) -> Self {
        self.file_resolver = Some(resolver);
        self
    }

    /// The [`ProviderScope`] file handles are valid within — the GCP
    /// project + region.
    fn scope(&self) -> ProviderScope {
        ProviderScope::new(
            ProviderType::Anthropic,
            format!(
                "{}/{}",
                self.endpoint.project_id(),
                self.endpoint.location()
            ),
        )
    }

    /// Convert internal request to Anthropic format.
    ///
    /// `resolved` maps each file-`Ref` id to its wire-ready reference, built
    /// by the async [`resolve_refs`] pre-pass in [`Self::generate`].
    fn convert_request(
        &self,
        prompt: &crate::Prompt,
        config: &RawConfig,
        resolved: &HashMap<String, ResolvedRef>,
    ) -> Result<AnthropicRequest, Error> {
        let mut messages = Vec::new();
        let mut system_message = None;

        for item in prompt.items() {
            match item {
                InputItem::System(content) => {
                    system_message = Some(content.clone());
                }
                InputItem::User { content } => {
                    let blocks = build_user_blocks(content, resolved)?;
                    if blocks.is_empty() {
                        continue;
                    }
                    if blocks.len() == 1 {
                        if let AnthropicContentBlock::Text { text, .. } = &blocks[0] {
                            messages.push(AnthropicMessage {
                                role: "user".to_string(),
                                content: AnthropicContent::Text(text.clone()),
                            });
                            continue;
                        }
                    }
                    messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContent::Blocks(blocks),
                    });
                }
                InputItem::Assistant { content } => {
                    let blocks = build_assistant_blocks(content)?;
                    if blocks.is_empty() {
                        continue;
                    }
                    if blocks.len() == 1 {
                        if let AnthropicContentBlock::Text { text, .. } = &blocks[0] {
                            messages.push(AnthropicMessage {
                                role: "assistant".to_string(),
                                content: AnthropicContent::Text(text.clone()),
                            });
                            continue;
                        }
                    }
                    messages.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: AnthropicContent::Blocks(blocks),
                    });
                }
            }
        }

        let tools = config.tools.as_ref().and_then(|tools| {
            use crate::types::{ProviderBuiltin, Tool};
            let converted: Vec<AnthropicTool> = tools
                .iter()
                .filter_map(|tool| match tool {
                    Tool::Function(f) => Some(AnthropicTool::Function {
                        name: f.name.clone(),
                        description: f.description.clone().unwrap_or_default(),
                        input_schema: f.parameters.clone(),
                    }),
                    Tool::Builtin(ProviderBuiltin::WebSearch) => Some(AnthropicTool::Builtin {
                        r#type: "web_search_20250305",
                        name: "web_search",
                    }),
                    Tool::Builtin(ProviderBuiltin::ComputerUse(cfg)) => {
                        Some(AnthropicTool::Computer {
                            r#type: "computer_20250124",
                            name: "computer",
                            display_width_px: cfg.display_width,
                            display_height_px: cfg.display_height,
                        })
                    }
                    Tool::Builtin(b) => {
                        tracing::debug!(?b, "Anthropic provider dropping unsupported builtin");
                        None
                    }
                })
                .collect();
            if converted.is_empty() {
                None
            } else {
                Some(converted)
            }
        });

        // Map our unified ReasoningConfig onto Anthropic's `thinking` field.
        // We derive budget_tokens from `effort` with sensible defaults;
        // callers needing precise control can construct providers directly.
        let thinking = config.reasoning.as_ref().map(|cfg| {
            let budget_tokens = match cfg.effort.unwrap_or(ReasoningEffort::Medium) {
                ReasoningEffort::Low => 2048,
                ReasoningEffort::Medium => 8192,
                ReasoningEffort::High => 16384,
            };
            AnthropicThinking::Enabled { budget_tokens }
        });

        // Anthropic requires temperature == 1 when thinking is enabled.
        // Override with a warning rather than erroring; better DX.
        let temperature = if thinking.is_some() {
            if matches!(config.temperature, Some(t) if (t - 1.0).abs() > f32::EPSILON) {
                tracing::warn!(
                    requested = ?config.temperature,
                    "Anthropic requires temperature=1 when extended thinking is enabled; \
                     overriding"
                );
            }
            Some(1.0)
        } else {
            config.temperature
        };

        let tool_choice = config.tool_choice.as_ref().map(|choice| match choice {
            crate::types::ToolChoice::Auto => AnthropicToolChoice::Auto,
            crate::types::ToolChoice::None => AnthropicToolChoice::None,
            crate::types::ToolChoice::Required => AnthropicToolChoice::Any,
            crate::types::ToolChoice::Function { name } => {
                AnthropicToolChoice::Tool { name: name.clone() }
            }
        });

        let anthropic_request = AnthropicRequest {
            messages,
            max_tokens: config.max_tokens.unwrap_or(1024),
            anthropic_version: "vertex-2023-10-16",
            system: system_message,
            temperature,
            top_p: config.top_p,
            tools,
            stream: Some(true), // Enable streaming for SSE responses
            thinking,
            stop_sequences: config.stop.clone(),
            tool_choice,
        };

        if config.presence_penalty.is_some() || config.frequency_penalty.is_some() {
            tracing::debug!(
                "Anthropic provider does not support presence/frequency penalty; dropping"
            );
        }
        // `config.response_format` is silently ignored here. Callers
        // that want structured output on Anthropic should drive the
        // request through `platformed_llm::generate`, which runs the
        // default `JsonCoercionMiddleware` and rewrites the request as
        // a forced-tool-call (unwrapped back to text on the response
        // side). Calling `Provider::generate` directly bypasses
        // middleware and the field is just dropped — same as before.

        Ok(anthropic_request)
    }
}

/// Translate user-side parts into Anthropic content blocks. A
/// `CacheBreakpoint` attaches `cache_control: {type: "ephemeral"}` to
/// the most recently emitted block.
fn build_user_blocks(
    parts: &[UserPart],
    resolved: &HashMap<String, ResolvedRef>,
) -> Result<Vec<AnthropicContentBlock>, Error> {
    let mut blocks = Vec::new();
    for part in parts {
        match part {
            UserPart::Text(s) => blocks.push(AnthropicContentBlock::Text {
                text: s.clone(),
                cache_control: None,
            }),
            UserPart::Image(src) => {
                let source = match src {
                    crate::types::FileSource::Url(u) => Some(ijson::ijson!({
                        "type": "url",
                        "url": u.clone(),
                    })),
                    crate::types::FileSource::Base64 { data, media_type } => Some(ijson::ijson!({
                        "type": "base64",
                        "media_type": media_type.clone(),
                        "data": data.clone(),
                    })),
                    crate::types::FileSource::Ref(id) => ref_to_source(resolved, id),
                };
                if let Some(source) = source {
                    blocks.push(AnthropicContentBlock::Image {
                        source,
                        cache_control: None,
                    });
                }
            }
            UserPart::ToolResult { call_id, content } => {
                let text = flatten_user_parts_to_text(content);
                blocks.push(AnthropicContentBlock::ToolResult {
                    tool_use_id: call_id.clone(),
                    content: AnthropicToolResultContent::Text(text),
                    is_error: None,
                });
            }
            // Audio / video are rejected up front in generate() via
            // reject_unsupported_modalities; these are defensive drops for any
            // direct convert_request caller.
            UserPart::Audio(_) => {
                tracing::debug!("Anthropic: dropping unsupported audio part");
            }
            UserPart::Video(_) => {
                tracing::debug!("Anthropic: dropping unsupported video part");
            }
            UserPart::Document(src) => {
                let source = match src {
                    crate::types::FileSource::Url(u) => Some(ijson::ijson!({
                        "type": "url",
                        "url": u.clone(),
                    })),
                    crate::types::FileSource::Base64 { data, media_type } => Some(ijson::ijson!({
                        "type": "base64",
                        "media_type": media_type.clone(),
                        "data": data.clone(),
                    })),
                    crate::types::FileSource::Ref(id) => ref_to_source(resolved, id),
                };
                if let Some(source) = source {
                    blocks.push(AnthropicContentBlock::Document {
                        source,
                        cache_control: None,
                    });
                }
            }
            UserPart::CacheBreakpoint => attach_cache_control(blocks.last_mut()),
        }
    }
    Ok(blocks)
}

/// Resolve a file `Ref` to an Anthropic content-block `source`, or `None`
/// (logged) when the id wasn't resolved. A provider handle becomes a
/// `{type:"file", file_id}` source; a URL becomes `{type:"url", url}`.
fn ref_to_source(resolved: &HashMap<String, ResolvedRef>, id: &str) -> Option<ijson::IValue> {
    match resolved.get(id) {
        Some(ResolvedRef::Handle { uri, .. }) => Some(ijson::ijson!({
            "type": "file",
            "file_id": uri.clone(),
        })),
        Some(ResolvedRef::Url { uri, .. }) => Some(ijson::ijson!({
            "type": "url",
            "url": uri.clone(),
        })),
        None => {
            tracing::debug!("Anthropic: unresolved file Ref {id}; dropping");
            None
        }
    }
}

/// Attach a `cache_control: {type: "ephemeral"}` hint to the most-
/// recently-emitted block (the one immediately before the
/// CacheBreakpoint in source order). Anthropic recognises this on
/// text, tool_use, and image blocks; other variants silently ignore
/// because the wire shape doesn't model the hint there.
fn attach_cache_control(last: Option<&mut AnthropicContentBlock>) {
    if let Some(block) = last {
        let hint = AnthropicCacheControl {
            r#type: "ephemeral".to_string(),
        };
        match block {
            AnthropicContentBlock::Text { cache_control, .. } => {
                *cache_control = Some(hint);
            }
            AnthropicContentBlock::ToolUse { cache_control, .. } => {
                *cache_control = Some(hint);
            }
            AnthropicContentBlock::Image { cache_control, .. } => {
                *cache_control = Some(hint);
            }
            AnthropicContentBlock::Document { cache_control, .. } => {
                *cache_control = Some(hint);
            }
            _ => {
                tracing::debug!(
                    "CacheBreakpoint preceded by a block type that doesn't accept cache_control; ignoring"
                );
            }
        }
    } else {
        tracing::debug!("CacheBreakpoint with no preceding block; ignoring");
    }
}

/// Translate assistant-side parts into Anthropic content blocks. Text +
/// reasoning + tool_use are all expressed as blocks; reasoning carries
/// its signature when present.
fn build_assistant_blocks(parts: &[AssistantPart]) -> Result<Vec<AnthropicContentBlock>, Error> {
    let mut blocks = Vec::new();
    for part in parts {
        match part {
            AssistantPart::Text { content, .. } => {
                blocks.push(AnthropicContentBlock::Text {
                    text: content.clone(),
                    cache_control: None,
                });
            }
            AssistantPart::Reasoning { content, signature } => {
                blocks.push(AnthropicContentBlock::Thinking {
                    thinking: content.clone(),
                    signature: signature.clone(),
                });
            }
            AssistantPart::RedactedReasoning { data } => {
                blocks.push(AnthropicContentBlock::RedactedThinking { data: data.clone() });
            }
            AssistantPart::Refusal(s) => {
                // Anthropic has no typed refusal channel; surface as text.
                blocks.push(AnthropicContentBlock::Text {
                    text: s.clone(),
                    cache_control: None,
                });
            }
            AssistantPart::ToolCall(call) => {
                let input = serde_json::from_str(&call.arguments).map_err(|e| {
                    Error::provider("Anthropic", format!("Invalid function arguments: {e}"))
                })?;
                blocks.push(AnthropicContentBlock::ToolUse {
                    id: call.call_id.clone(),
                    name: call.name.clone(),
                    input,
                    cache_control: None,
                });
            }
            AssistantPart::BuiltinToolCall { .. } => {
                // Provider-side tool calls don't round-trip through
                // history on Anthropic; drop them per the
                // model-switching contract.
                tracing::debug!("Anthropic provider dropping BuiltinToolCall during request build");
            }
            AssistantPart::Continuation(_) => {
                // Anthropic has no equivalent server-side resumption
                // surface; drop the continuation marker silently.
            }
            AssistantPart::CacheBreakpoint => attach_cache_control(blocks.last_mut()),
        }
    }
    Ok(blocks)
}

use crate::providers::flatten_user_parts_to_text;

#[async_trait::async_trait]
impl Provider for AnthropicViaVertexProvider {
    async fn generate(
        &self,
        prompt: &crate::Prompt,
        config: &RawConfig,
    ) -> Result<Response, Error> {
        // Claude accepts only image / document inputs — reject audio / video
        // up front rather than dropping them.
        crate::providers::reject_unsupported_modalities(prompt.items(), "Anthropic", false, false)?;

        let resolved = resolve_refs(
            prompt.items(),
            &self.scope(),
            self.file_resolver.as_deref(),
            &NoLibraryUpload {
                provider: "Anthropic",
            },
        )
        .await?;
        let anthropic_request = self.convert_request(prompt, config, &resolved)?;

        let url = self.endpoint.url(
            "anthropic",
            &config.model,
            "streamRawPredict",
            Some("alt=sse"),
        );

        let body = serde_json::to_vec(&anthropic_request)?;
        let mut headers = vec![
            self.endpoint.auth_header().await?,
            ("Content-Type".to_string(), "application/json".to_string()),
        ];
        if !self.beta.is_empty() {
            headers.push(("anthropic-beta".to_string(), self.beta.join(",")));
        }
        let req = TransportRequest { url, headers, body };

        let scope = crate::rate_limit::RateScope {
            // Vertex regions have independent quotas — include the
            // location so per-region buckets stay separate.
            bucket_key: format!(
                "Vertex-Anthropic/{}/{}",
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
            // Anthropic doesn't expose a typed code for "too many input
            // tokens" — detect via message-string match on 400s. The
            // canonical phrasing as of 2026 is "prompt is too long" but
            // the upstream may rephrase; this is best-effort.
            if status == 400 && is_anthropic_context_exceeded(&body_text) {
                return Err(Error::context_window_exceeded(
                    "Anthropic",
                    body_text.to_string(),
                ));
            }
            return Err(match status {
                401 | 403 => {
                    Error::auth_with_status(status, format!("Anthropic {status}: {body_text}"))
                }
                404 => Error::ModelNotAvailable(format!("Anthropic 404: {body_text}")),
                429 => Error::rate_limit(
                    retry_after,
                    format!("Anthropic 429 (rate limited): {body_text}"),
                ),
                // 5xx (and any other non-special status) may carry
                // a `Retry-After` per RFC 7231; thread it through so
                // the retry helper honours the server hint.
                _ => Error::provider_with_retry_after(
                    "Anthropic",
                    status,
                    retry_after,
                    format!("API error: {body_text}"),
                ),
            });
        }

        // Success path: defer the limiter observation until the
        // stream terminates so a mid-stream `overloaded_error` /
        // `rate_limit_error` (which we map to `Error::RateLimit`
        // below) is fed back as `RateLimited`, not `Success`. See
        // `rate_limit::observe_stream`.

        // Create SSE stream from response
        let sse_stream = SseStream::new("Anthropic", response.body);

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

                        // Anthropic's wire format only emits JSON event
                        // payloads (including `{"type":"ping"}` for keep-
                        // alives). The SSE parser already filters comment
                        // lines, so anything that fails to parse here is a
                        // genuine surprise — surface it.
                        match serde_json::from_str::<AnthropicStreamEvent>(data) {
                            Ok(stream_event) => {
                                match convert_stream_event_stateful(stream_event, &mut state) {
                                    Ok(events) => events.into_iter().map(Ok).collect(),
                                    Err(e) => vec![Err(e)],
                                }
                            }
                            Err(e) => vec![Err(Error::provider(
                                "Anthropic",
                                format!("Failed to parse SSE event: {e}"),
                            ))],
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

/// State for tracking streaming progress.
///
/// Anthropic delivers `stop_reason` and the cumulative `usage` on
/// `message_delta` events which fire **before** `message_stop`. We
/// stash them so the final `Done` event reflects what the model
/// actually said.
#[derive(Debug, Default)]
pub(crate) struct StreamState {
    /// Maps Anthropic's content-block index to our lib-side part index.
    tracker: crate::providers::part_tracker::PartTracker<u32>,
    /// Cumulative usage merged from `message_start` and `message_delta`.
    pending_usage: Usage,
    /// `stop_reason` captured from `message_delta`.
    pending_stop_reason: Option<String>,
}

/// Heuristic match for "input too long" 400s. Anthropic returns
/// `invalid_request_error` with a free-form message and no typed code,
/// so we look for the documented wording patterns. Conservative — a
/// near-miss falls through to a generic provider error rather than
/// claiming a context-window cause we're not sure of.
///
/// The set of accepted phrases is intentionally narrow:
/// - `prompt is too long`
/// - `input is too long`
/// - `context window`
///
/// An earlier version included a loose `maximum && (tokens || input
/// length)` clause as a catch-all. It false-positived on Anthropic's
/// output-cap validation error (`max_tokens: N > M, which is the
/// maximum allowed number of output tokens`), which would have made
/// a compaction-aware caller destroy history in response to an
/// output-config error. Dropped — the three explicit phrases above
/// cover the documented context-exceeded responses, and a near-miss
/// falling through to `Error::Provider` is the safer default.
fn is_anthropic_context_exceeded(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    (lower.contains("prompt is too long")
        || lower.contains("input is too long")
        || lower.contains("context window"))
        && lower.contains("invalid_request_error")
}

/// Map an Anthropic `stop_reason` string onto our unified [`FinishReason`].
///
/// Until [`FinishReason`] is extended (Phase 5), `stop_sequence` and
/// `pause_turn` collapse to `Stop` — the closest existing variant.
pub(crate) fn map_anthropic_stop_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("end_turn") => FinishReason::Stop,
        Some("tool_use") => FinishReason::ToolCalls,
        Some("max_tokens") => FinishReason::Length,
        Some("stop_sequence") => FinishReason::Stop,
        Some("pause_turn") => FinishReason::Stop,
        Some("refusal") => FinishReason::ContentFilter,
        Some(other) => {
            tracing::warn!(stop_reason = other, "unknown Anthropic stop_reason");
            FinishReason::Stop
        }
        None => FinishReason::Stop,
    }
}

/// Merge an Anthropic `usage` object into the running [`Usage`] tally.
///
/// Anthropic's streaming protocol reports `input_tokens` and the
/// cache fields together on `message_start`, and a cumulative
/// `output_tokens` on the final `message_delta`. We always overwrite
/// with the latest non-`None` value so the `Done` event reflects
/// the model's authoritative final counts.
///
/// `input_tokens` on the wire is the UNCACHED remainder — Anthropic
/// reports the cache fields as additive. The unified [`Usage`] type
/// documents `input_tokens` as the total prompt (with cache_* fields
/// as breakdowns / subsets), so this merge normalises by adding the
/// cache fields when `input_tokens` is set.
fn merge_anthropic_usage(target: &mut Usage, src: &AnthropicUsage) {
    if let Some(t) = src.cache_read_input_tokens {
        target.cache_read_input_tokens = Some(t);
    }
    if let Some(t) = src.cache_creation_input_tokens {
        target.cache_creation_input_tokens = Some(t);
    }
    if let Some(t) = src.input_tokens {
        // Cache fields are always emitted alongside `input_tokens`
        // at `message_start`, so reading them off `src` here gives
        // the right total.
        target.input_tokens = t
            .saturating_add(src.cache_read_input_tokens.unwrap_or(0))
            .saturating_add(src.cache_creation_input_tokens.unwrap_or(0));
    }
    if let Some(t) = src.output_tokens {
        target.output_tokens = t;
    }
}

/// Convert an Anthropic stream event into our unified `StreamEvent`s.
///
/// `pub(crate)` so unit tests can drive this directly with synthetic events.
pub(crate) fn convert_stream_event_stateful(
    event: AnthropicStreamEvent,
    state: &mut StreamState,
) -> Result<Vec<StreamEvent>, Error> {
    let mut events = Vec::new();

    match event {
        AnthropicStreamEvent::MessageStart { message } => {
            if let Some(usage) = &message.usage {
                merge_anthropic_usage(&mut state.pending_usage, usage);
            }
        }
        AnthropicStreamEvent::ContentBlockStart {
            content_block,
            index,
        } => match content_block {
            AnthropicContentBlock::Text { text, .. } => {
                let (lib_idx, ev) = state.tracker.open(index, PartKind::Text);
                events.push(ev);
                if !text.is_empty() {
                    events.push(StreamEvent::Delta {
                        index: lib_idx,
                        delta: text,
                    });
                }
            }
            AnthropicContentBlock::ToolUse {
                id, name, input, ..
            } => {
                let (_lib_idx, ev) = state
                    .tracker
                    .open(index, PartKind::ToolCall { call_id: id, name });
                events.push(ev);
                // Per the streaming protocol the initial `input` is `{}`.
                // Arguments arrive via input_json_delta.
                let nonempty = !(input.is_null()
                    || (input.is_object()
                        && input.as_object().map(|o| o.is_empty()).unwrap_or(true)));
                if nonempty {
                    tracing::warn!(
                        ?input,
                        "Anthropic content_block_start carried non-empty `input`; \
                         ignoring and relying on input_json_delta accumulation"
                    );
                }
            }
            AnthropicContentBlock::Thinking {
                thinking,
                signature,
            } => {
                let (lib_idx, ev) = state.tracker.open(index, PartKind::Reasoning);
                events.push(ev);
                if !thinking.is_empty() {
                    events.push(StreamEvent::Delta {
                        index: lib_idx,
                        delta: thinking,
                    });
                }
                if let Some(sig) = signature {
                    events.push(StreamEvent::PartUpdate {
                        index: lib_idx,
                        update: PartUpdate::Signature(sig),
                    });
                }
            }
            AnthropicContentBlock::RedactedThinking { data } => {
                let (_lib_idx, ev) = state
                    .tracker
                    .open(index, PartKind::RedactedReasoning { data });
                events.push(ev);
            }
            AnthropicContentBlock::ToolResult { .. }
            | AnthropicContentBlock::Image { .. }
            | AnthropicContentBlock::Document { .. } => {
                // Request-side blocks; not expected on the response stream.
            }
        },
        AnthropicStreamEvent::ContentBlockDelta { delta, index } => {
            let lib_idx = match state.tracker.index_of(&index) {
                Some(i) => i,
                None => {
                    return Err(Error::provider(
                        "Anthropic",
                        format!("content_block_delta for unknown index {index}"),
                    ));
                }
            };
            match delta {
                AnthropicContentDelta::TextDelta { text } => {
                    if !text.is_empty() {
                        events.push(StreamEvent::Delta {
                            index: lib_idx,
                            delta: text,
                        });
                    }
                }
                AnthropicContentDelta::InputJsonDelta { partial_json } => {
                    events.push(StreamEvent::Delta {
                        index: lib_idx,
                        delta: partial_json,
                    });
                }
                AnthropicContentDelta::ThinkingDelta { thinking } => {
                    if !thinking.is_empty() {
                        events.push(StreamEvent::Delta {
                            index: lib_idx,
                            delta: thinking,
                        });
                    }
                }
                AnthropicContentDelta::SignatureDelta { signature } => {
                    events.push(StreamEvent::PartUpdate {
                        index: lib_idx,
                        update: PartUpdate::Signature(signature),
                    });
                }
            }
        }
        AnthropicStreamEvent::ContentBlockStop { index } => {
            if let Some(ev) = state.tracker.close(&index) {
                events.push(ev);
            }
        }
        AnthropicStreamEvent::MessageDelta { delta, usage } => {
            if let Some(reason) = delta.stop_reason {
                state.pending_stop_reason = Some(reason);
            }
            if let Some(usage) = usage {
                merge_anthropic_usage(&mut state.pending_usage, &usage);
            }
        }
        AnthropicStreamEvent::MessageStop => {
            let finish_reason = map_anthropic_stop_reason(state.pending_stop_reason.as_deref());
            let usage = std::mem::take(&mut state.pending_usage);
            events.push(StreamEvent::Done {
                finish_reason,
                usage,
            });
        }
        AnthropicStreamEvent::Ping => {
            // Keep-alive event - ignore
        }
        AnthropicStreamEvent::Error { error } => {
            // Mid-stream rate limits (`overloaded_error` /
            // `rate_limit_error`) arrive after a 200 has already gone
            // out; normalise to the typed `Error::RateLimit` variant
            // so caller-level retry loops and the rate limiter can
            // both treat them like a pre-stream 429. Other mid-stream
            // errors stay as `Error::Provider`.
            //
            // Note: the rate-limit permit was already observed at the
            // HTTP-200 success site, so the limiter's AIMD model
            // doesn't yet learn from this mid-stream event. Holding
            // the permit through the stream consumption would fix
            // that — tracked as a v2 refactor; for now the caller's
            // retry loop carries the resilience signal.
            if error.error_type == "rate_limit_error" || error.error_type == "overloaded_error" {
                return Err(Error::rate_limit(
                    None,
                    format!(
                        "Anthropic mid-stream {}: {}",
                        error.error_type, error.message
                    ),
                ));
            }
            return Err(Error::provider(
                "Anthropic",
                format!("{}: {}", error.error_type, error.message),
            ));
        }
    }

    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, Prompt};

    fn provider() -> AnthropicViaVertexProvider {
        AnthropicViaVertexProvider::new("p".to_string(), "us-east5".to_string(), "tok".to_string())
            .unwrap()
    }

    /// Mid-stream `overloaded_error` and `rate_limit_error` events
    /// must surface as the typed [`Error::RateLimit`] so caller-level
    /// retry loops and the rate limiter can both recognise them.
    /// Other mid-stream errors should still surface as the generic
    /// [`Error::Provider`].
    #[test]
    fn mid_stream_rate_limit_normalises_to_typed_error() {
        use crate::providers::vertex::anthropic_types::AnthropicErrorPayload;
        let mut state = StreamState::default();
        for kind in ["overloaded_error", "rate_limit_error"] {
            let err = convert_stream_event_stateful(
                AnthropicStreamEvent::Error {
                    error: AnthropicErrorPayload {
                        error_type: kind.to_string(),
                        message: format!("simulated {kind}"),
                    },
                },
                &mut state,
            )
            .expect_err("error event must produce Err");
            assert!(
                matches!(err, Error::RateLimit { .. }),
                "{kind} should map to Error::RateLimit, got {err:?}",
            );
        }
        // A non-rate-limit error stays generic so callers can branch
        // appropriately (auth issues, server errors, etc.).
        let err = convert_stream_event_stateful(
            AnthropicStreamEvent::Error {
                error: AnthropicErrorPayload {
                    error_type: "api_error".to_string(),
                    message: "internal".to_string(),
                },
            },
            &mut state,
        )
        .expect_err("error event must produce Err");
        assert!(
            matches!(err, Error::Provider { .. }),
            "non-rate-limit error should stay generic Provider, got {err:?}",
        );
    }

    #[test]
    fn detect_context_exceeded_in_invalid_request_error() {
        // Documented Anthropic phrasing.
        let body = r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 250000 tokens > 200000 maximum"}}"#;
        assert!(is_anthropic_context_exceeded(body));

        // Alternate phrasing.
        let body2 = r#"{"type":"error","error":{"type":"invalid_request_error","message":"input is too long for the model's context window"}}"#;
        assert!(is_anthropic_context_exceeded(body2));

        // A *different* invalid_request_error must not match.
        let body3 = r#"{"type":"error","error":{"type":"invalid_request_error","message":"messages must contain at least one item"}}"#;
        assert!(!is_anthropic_context_exceeded(body3));

        // An error from a different category must not match either.
        let body4 = r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#;
        assert!(!is_anthropic_context_exceeded(body4));
    }

    /// PR-review #3: the `max_tokens > model_max_output` validation
    /// error contains `maximum`, `tokens`, and `invalid_request_error`
    /// — under the loose conjunction `maximum && (tokens || input
    /// length)` it false-positived as `ContextWindowExceeded`,
    /// which would lead a compaction-aware caller to destroy
    /// history in response to an *output*-config error and then
    /// retry into the same failure. The detector must reject this.
    #[test]
    fn output_token_cap_error_is_not_context_exceeded() {
        let body = r#"{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: 100000 > 64000, which is the maximum allowed number of output tokens for claude-sonnet-4-5"}}"#;
        assert!(
            !is_anthropic_context_exceeded(body),
            "output-token-cap error must not classify as context-exceeded"
        );
    }

    /// Pin the "context window" wording path — Anthropic documents
    /// at least one variant that uses that exact phrase rather than
    /// "input/prompt too long". The detector keeps that branch.
    #[test]
    fn context_window_phrase_still_matches() {
        let body = r#"{"type":"error","error":{"type":"invalid_request_error","message":"this request exceeds the model's context window"}}"#;
        assert!(is_anthropic_context_exceeded(body));
    }

    /// PR-review #5. The streaming `merge_anthropic_usage` must
    /// normalise `input_tokens` to be the union of uncached +
    /// cache_read + cache_creation — matching the
    /// `From<AnthropicUsage>` branch and the unified `Usage`
    /// invariant.
    #[test]
    fn merge_normalises_cache_into_input_tokens() {
        let mut target = Usage::default();
        // message_start: input_tokens + cache fields.
        let start = AnthropicUsage {
            input_tokens: Some(5_000),
            output_tokens: None,
            cache_read_input_tokens: Some(150_000),
            cache_creation_input_tokens: Some(10_000),
        };
        merge_anthropic_usage(&mut target, &start);
        assert_eq!(
            target.input_tokens, 165_000,
            "merge must add cache_read + cache_creation to uncached input"
        );
        assert_eq!(target.cache_read_input_tokens, Some(150_000));
        assert_eq!(target.cache_creation_input_tokens, Some(10_000));

        // message_delta: only output_tokens. Must not clobber the
        // normalised input_tokens.
        let delta = AnthropicUsage {
            input_tokens: None,
            output_tokens: Some(2_000),
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
        };
        merge_anthropic_usage(&mut target, &delta);
        assert_eq!(target.input_tokens, 165_000, "input_tokens must persist");
        assert_eq!(target.output_tokens, 2_000);
    }

    /// Bridge from #5 to the compaction threshold: a warm-cache
    /// Anthropic Usage that pre-fix reported a context fraction of
    /// ~0.035 must now correctly report ~0.785 against a 200k
    /// window, so [`crate::Compactor::should_compact`] fires.
    #[test]
    fn warm_cache_usage_triggers_compaction_threshold() {
        use crate::Capabilities;
        let mut usage = Usage::default();
        merge_anthropic_usage(
            &mut usage,
            &AnthropicUsage {
                input_tokens: Some(5_000),
                output_tokens: Some(2_000),
                cache_read_input_tokens: Some(150_000),
                cache_creation_input_tokens: None,
            },
        );
        let caps = Capabilities {
            context_window_tokens: 200_000,
            ..Capabilities::default()
        };
        let fraction = caps.context_usage_fraction(&usage);
        assert!(
            fraction > 0.7,
            "warm-cache usage should report >0.7 (got {fraction}); pre-fix this was ~0.035"
        );
    }

    #[test]
    fn map_anthropic_stop_reason_known_values() {
        assert_eq!(
            map_anthropic_stop_reason(Some("end_turn")),
            FinishReason::Stop
        );
        assert_eq!(
            map_anthropic_stop_reason(Some("tool_use")),
            FinishReason::ToolCalls
        );
        assert_eq!(
            map_anthropic_stop_reason(Some("max_tokens")),
            FinishReason::Length
        );
        assert_eq!(
            map_anthropic_stop_reason(Some("refusal")),
            FinishReason::ContentFilter
        );
        assert_eq!(map_anthropic_stop_reason(None), FinishReason::Stop);
    }

    #[test]
    fn convert_simple_text_request() {
        let prompt = Prompt::user("hi");
        let cfg = Config::builder("claude").build();
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &std::collections::HashMap::new())
            .unwrap();
        assert_eq!(body.messages.len(), 1);
        assert_eq!(body.messages[0].role, "user");
    }

    /// A resolved document `Ref` (handle) lands as a `{type:"file", file_id}`
    /// source; a URL result as `{type:"url", url}`.
    #[test]
    fn resolved_ref_emits_file_and_url_sources() {
        use crate::providers::file_resolve::ResolvedRef;
        use crate::types::{FileSource, InputItem, UserPart};

        // Handle -> file source.
        let prompt = Prompt::new().with_item(InputItem::User {
            content: vec![UserPart::Document(FileSource::Ref("doc1".into()))],
        });
        let mut resolved = std::collections::HashMap::new();
        resolved.insert(
            "doc1".to_string(),
            ResolvedRef::Handle {
                uri: "file-abc".into(),
                media_type: "application/pdf".into(),
            },
        );
        let cfg = Config::builder("claude").build();
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &resolved)
            .unwrap();
        let json = serde_json::to_value(&body).unwrap();
        let source = &json["messages"][0]["content"][0]["source"];
        assert_eq!(source["type"], "file");
        assert_eq!(source["file_id"], "file-abc");

        // URL -> url source.
        let mut resolved_url = std::collections::HashMap::new();
        resolved_url.insert(
            "doc1".to_string(),
            ResolvedRef::Url {
                uri: "https://example.com/x.pdf".into(),
                media_type: "application/pdf".into(),
            },
        );
        let body = provider()
            .convert_request(&prompt, cfg.raw(), &resolved_url)
            .unwrap();
        let json = serde_json::to_value(&body).unwrap();
        let source = &json["messages"][0]["content"][0]["source"];
        assert_eq!(source["type"], "url");
        assert_eq!(source["url"], "https://example.com/x.pdf");
    }

    /// A signature_delta on a thinking block emits PartUpdate::Signature
    /// pointing at the correct part index.
    #[test]
    fn signature_delta_emits_part_update() {
        let mut state = StreamState::default();
        let start = AnthropicStreamEvent::ContentBlockStart {
            index: 0,
            content_block: AnthropicContentBlock::Thinking {
                thinking: String::new(),
                signature: None,
            },
        };
        let _ = convert_stream_event_stateful(start, &mut state).unwrap();
        let sig = AnthropicStreamEvent::ContentBlockDelta {
            index: 0,
            delta: AnthropicContentDelta::SignatureDelta {
                signature: "sig_abc".to_string(),
            },
        };
        let events = convert_stream_event_stateful(sig, &mut state).unwrap();
        match &events[0] {
            StreamEvent::PartUpdate {
                index: 0,
                update: PartUpdate::Signature(s),
            } => assert_eq!(s, "sig_abc"),
            other => panic!("expected PartUpdate(Signature), got {other:?}"),
        }
    }

    /// A `tool_use` content block opens a `PartKind::ToolCall` part with
    /// the wire `id` carried as our `call_id`.
    #[test]
    fn tool_use_opens_tool_call_part() {
        let mut state = StreamState::default();
        let start = AnthropicStreamEvent::ContentBlockStart {
            index: 0,
            content_block: AnthropicContentBlock::ToolUse {
                id: "toolu_xyz".to_string(),
                name: "get_weather".to_string(),
                input: ijson::ijson!({}),
                cache_control: None,
            },
        };
        let events = convert_stream_event_stateful(start, &mut state).unwrap();
        match &events[0] {
            StreamEvent::PartStart {
                index: 0,
                kind: PartKind::ToolCall { call_id, name },
            } => {
                assert_eq!(call_id, "toolu_xyz");
                assert_eq!(name, "get_weather");
            }
            other => panic!("expected PartStart(ToolCall), got {other:?}"),
        }
    }
}
