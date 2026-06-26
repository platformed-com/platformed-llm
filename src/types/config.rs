use serde::{Deserialize, Serialize};

/// Token usage information across providers.
///
/// Not every provider populates every field — fields specific to one
/// provider's billing model (cache create/read, reasoning) are `Option`
/// and stay `None` for providers that don't report them.
///
/// **Invariant**: `input_tokens` is the total prompt — the union of
/// uncached + cache-read + cache-creation. `cache_read_input_tokens`
/// and `cache_creation_input_tokens` are the breakdown (subsets of
/// `input_tokens`), not additive. Wire-level differences are
/// normalised on ingest: OpenAI and Google already report this way;
/// Anthropic's wire format reports `input_tokens` as the uncached
/// remainder with cache fields additive, and the provider's
/// `From<AnthropicUsage>` / `merge_anthropic_usage` add the cache
/// fields to match the unified invariant. This keeps
/// [`Self::total_tokens`] correct across providers and prevents
/// `Capabilities::context_usage_fraction` from under-firing on
/// cache-warm Anthropic conversations.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Usage {
    /// Total prompt tokens — the union of uncached input, cache-read,
    /// and cache-creation tokens.
    pub input_tokens: u32,
    /// Tokens in the completion.
    pub output_tokens: u32,
    /// Subset of [`Self::input_tokens`] served from the prompt cache
    /// (charged at a discount). Reported by Anthropic as
    /// `cache_read_input_tokens`, by OpenAI under
    /// `input_tokens_details.cached_tokens`, and by Gemini as
    /// `cachedContentTokenCount`.
    pub cache_read_input_tokens: Option<u32>,
    /// Subset of [`Self::input_tokens`] *written* to the cache on this
    /// request (Anthropic-only; charged at a 1.25× premium). Reported
    /// as `cache_creation_input_tokens`.
    pub cache_creation_input_tokens: Option<u32>,
    /// Output tokens spent on the model's internal reasoning (gpt-5 /
    /// o-series and Gemini thinking). OpenAI reports this under
    /// `output_tokens_details.reasoning_tokens`; Gemini reports it as
    /// `thoughtsTokenCount`.
    pub reasoning_tokens: Option<u32>,
}

impl Usage {
    /// `input_tokens + output_tokens` — the total tokens charged for
    /// this turn. Cache-read / cache-creation / reasoning fields are
    /// already counted inside input / output respectively, so this
    /// gives the right number for "how much of the context window did
    /// this turn touch?".
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

/// Reasoning configuration for models that support chain-of-thought
/// (gpt-5 / o-series, Claude extended thinking, Gemini thinking).
///
/// Each provider has a different shape; this is the unified surface and
/// each provider's `convert_request` translates it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ReasoningConfig {
    /// How much effort to spend reasoning. Maps to OpenAI's `effort` and
    /// to Anthropic / Gemini's `budget_tokens` (rough mapping).
    pub effort: Option<ReasoningEffort>,
    /// Whether (and how) to surface reasoning summaries (OpenAI). Anthropic
    /// returns thinking content unconditionally when enabled; Gemini's
    /// thinking is not exposed to clients.
    pub summary: Option<ReasoningSummary>,
}

/// Coarse "how hard to think" knob. Each provider maps it onto its own
/// budget / effort parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    /// Minimal reasoning, optimized for latency.
    Low,
    /// Balanced reasoning budget — the typical default.
    Medium,
    /// Maximum reasoning budget; useful for hard problems.
    High,
}

/// Verbosity level for reasoning summaries that are surfaced to the
/// caller (OpenAI Responses API).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningSummary {
    /// Let the provider pick a summary style.
    Auto,
    /// Short, high-level summary.
    Concise,
    /// Long, step-by-step summary.
    Detailed,
}

/// Provider-specific continuation hint that the caller carries from a
/// [`crate::CompleteResponse`] into the next conversation turn by
/// appending an [`crate::AssistantPart::Continuation`] part on the
/// assistant turn that produced it.
///
/// When the next request targets the *same* provider that issued the
/// hint, the provider uses it as an optimization (e.g. OpenAI's
/// `previous_response_id` to elide the message history). When it
/// targets a *different* provider, the hint is silently ignored and
/// the lib falls back to sending the full conversation. Continuations
/// are *always* optional — the lib works the same with or without one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderContinuation {
    /// OpenAI Responses API `previous_response_id`.
    OpenAI {
        /// The `id` field of the previous OpenAI response.
        response_id: String,
    },
    /// Gemini `cachedContent` resource name (pre-created via the
    /// Vertex AI CachedContent API). When the next request targets the
    /// same Gemini deployment, Vertex looks up the cached prefix and
    /// elides the message history that produced it.
    Gemini {
        /// Fully-qualified cached-content resource name.
        cached_content: String,
    },
}

/// Structured output mode for the response.
///
/// Mapping per provider:
/// - **OpenAI**: maps to `text.format` on the Responses API. `Text` is
///   the default (no constraint); `JsonObject` sets `{"type":
///   "json_object"}`; `JsonSchema` sets `{"type": "json_schema",
///   "json_schema": {...}}`.
/// - **Gemini**: maps to `generationConfig.responseMimeType:
///   "application/json"` plus `responseSchema` for `JsonSchema`.
/// - **Anthropic**: no native JSON mode. The default middleware chain
///   polyfills this via tool-use coercion (see
///   [`crate::JsonCoercionMiddleware`]).
#[derive(Debug, Clone)]
pub enum ResponseFormat {
    /// Default — unconstrained text output.
    Text,
    /// Bare JSON mode — model returns valid JSON but no schema is enforced.
    JsonObject,
    /// JSON Schema constraint. `strict` requests strict-mode validation
    /// on providers that support it.
    JsonSchema {
        /// Schema identifier surfaced to providers that require one.
        name: String,
        /// The JSON Schema itself.
        schema: std::borrow::Cow<'static, serde_json::value::RawValue>,
        /// Request strict-mode validation where supported.
        strict: bool,
    },
}

/// Strategy for how the model should use available tools.
///
/// Each provider has its own wire shape for this; the conversion happens
/// inside each provider's `convert_request`.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoice {
    /// Default. The model picks whether to call a tool.
    Auto,
    /// Disable tools for this request even if `tools` is non-empty.
    None,
    /// Force the model to call exactly one tool (any tool).
    Required,
    /// Force the model to call this specific tool.
    Function {
        /// Name of the required tool.
        name: String,
    },
}

/// The request payload that flows through the middleware chain and
/// into the provider.
///
/// Held inside [`Config`] as the `raw` template; copied at the top of
/// [`crate::generate`] into a `Cow<RawConfig>` that middleware
/// modify. Capabilities are *not* a field here — they're resolved by
/// the provider per call (see [`crate::Provider::capabilities`]) and
/// flow alongside `RawConfig` through the pipeline.
#[derive(Debug, Clone)]
pub struct RawConfig {
    /// Provider-specific model identifier (e.g. `"gpt-4o"`,
    /// `"gemini-2.5-pro"`, `"claude-sonnet-4-5"`).
    pub model: String,
    /// Sampling temperature (`0.0` = deterministic, higher = more random).
    pub temperature: Option<f32>,
    /// Hard cap on output tokens.
    pub max_tokens: Option<u32>,
    /// Nucleus sampling — restrict to the smallest token set whose
    /// cumulative probability is `top_p`.
    pub top_p: Option<f32>,
    /// Stop sequences. The model halts as soon as it would emit any of these.
    pub stop: Option<Vec<String>>,
    /// Penalty for tokens that have already appeared in the response.
    pub presence_penalty: Option<f32>,
    /// Penalty proportional to a token's prior occurrence count.
    pub frequency_penalty: Option<f32>,
    /// Functions / builtins the model may call.
    pub tools: Option<Vec<super::message::Tool>>,
    /// How the model should choose among tools.
    pub tool_choice: Option<ToolChoice>,
    /// Whether to allow more than one tool call per turn (OpenAI). `None`
    /// uses the provider's default.
    pub parallel_tool_calls: Option<bool>,
    /// Whether OpenAI should retain the response server-side for use with
    /// `previous_response_id` chaining. `None` uses the provider's default
    /// (which is currently `true` for OpenAI).
    pub store: Option<bool>,
    /// Reasoning configuration. Only meaningful for models that support
    /// chain-of-thought reasoning.
    pub reasoning: Option<ReasoningConfig>,
    /// Structured-output constraint (JSON mode / JSON schema). `None`
    /// means unconstrained text output.
    pub response_format: Option<ResponseFormat>,
    /// Opaque tenant key consulted by the provider's
    /// [`crate::rate_limit::RateLimiter`] for fair queueing. `None`
    /// collapses to a single anonymous tenant — fine for
    /// single-tenant deployments; multi-tenant callers should set
    /// this per request so the limiter can isolate one tenant's
    /// burst from another. Stored as [`Arc<str>`](std::sync::Arc) so the same
    /// tenant identifier can be cheaply cloned into many requests.
    pub tenant: Option<std::sync::Arc<str>>,
    /// Latency priority for the rate limiter. Defaults to
    /// [`crate::Priority::Interactive`] when unset — most callers
    /// run user-facing requests, so we minimise their queueing
    /// latency by default. Background batches should explicitly
    /// pick [`crate::Priority::Background`].
    pub priority: Option<crate::rate_limit::Priority>,
}

/// User-facing request spec. Bundles the [`RawConfig`] payload with
/// an optional middleware override. Capabilities are *not* per-call
/// — they're owned by the provider (see
/// [`crate::Provider::capabilities`]) and resolved at
/// [`crate::generate`] time. Middleware default to
/// [`crate::middleware::default_middleware`] applied to the resolved
/// caps unless the caller pinned a specific chain via
/// [`ConfigBuilder::with_middleware`].
///
/// Constructed via [`Config::builder`]. The struct itself is
/// inspectable but immutable from the caller's side; use the builder
/// to make changes.
#[derive(Clone)]
pub struct Config {
    raw: RawConfig,
    #[allow(clippy::type_complexity)]
    middleware_override: Option<Vec<std::sync::Arc<dyn crate::middleware::Middleware>>>,
}

impl Config {
    /// Start a builder targeting `model`. Equivalent to
    /// [`ConfigBuilder::new`].
    pub fn builder(model: impl Into<String>) -> ConfigBuilder {
        ConfigBuilder::new(model)
    }

    /// Borrow the [`RawConfig`] payload. This is what gets threaded
    /// through middleware and reaches the provider.
    pub fn raw(&self) -> &RawConfig {
        &self.raw
    }

    /// Caller's middleware override, if any. `None` means
    /// [`crate::middleware::default_middleware`] is derived from the
    /// resolved capabilities at `generate()` time.
    pub fn middleware_override(
        &self,
    ) -> Option<&[std::sync::Arc<dyn crate::middleware::Middleware>]> {
        self.middleware_override.as_deref()
    }
}

// `Config` carries an `Arc<dyn Middleware>` vector; `dyn Middleware: Debug`
// is a trait supertype so this would work, but the wrapper struct doesn't
// derive `Debug` automatically because of the trait-object indirection. A
// manual impl prints the raw fields plus a count of installed middleware
// so logs are useful without dumping every middleware's Debug body.
impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("raw", &self.raw)
            .field(
                "middleware_override_count",
                &self.middleware_override.as_ref().map(|m| m.len()),
            )
            .finish()
    }
}

/// Builder for [`Config`]. Created via [`ConfigBuilder::new`] or
/// [`Config::builder`]; finalized via [`ConfigBuilder::build`].
///
/// All settings except `model` default to `None` (provider default).
/// Capability and middleware overrides default to "use the library's
/// auto-derived value"; setting either pins the chain to the value
/// you supply, including `Vec::new()` to disable polyfills entirely.
#[derive(Clone)]
pub struct ConfigBuilder {
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    stop: Option<Vec<String>>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    tools: Option<Vec<super::message::Tool>>,
    tool_choice: Option<ToolChoice>,
    parallel_tool_calls: Option<bool>,
    store: Option<bool>,
    reasoning: Option<ReasoningConfig>,
    response_format: Option<ResponseFormat>,
    tenant: Option<std::sync::Arc<str>>,
    priority: Option<crate::rate_limit::Priority>,
    #[allow(clippy::type_complexity)]
    middleware_override: Option<Vec<std::sync::Arc<dyn crate::middleware::Middleware>>>,
}

impl ConfigBuilder {
    /// Build a config targeting `model`. All other fields default to
    /// `None` (provider default); set them via the chainable builder
    /// methods.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            store: None,
            reasoning: None,
            response_format: None,
            tenant: None,
            priority: None,
            middleware_override: None,
        }
    }

    /// Set the temperature (randomness) parameter.
    ///
    /// Must be finite and in `0.0..=2.0`. Passing a value outside
    /// that range (or NaN/∞) is a caller logic error and panics —
    /// it's never a valid request. Note providers impose their own
    /// tighter limits (Anthropic caps at 1.0); those still surface
    /// server-side.
    pub fn temperature(mut self, temperature: f32) -> Self {
        assert!(
            temperature.is_finite() && (0.0..=2.0).contains(&temperature),
            "temperature must be finite and in 0.0..=2.0, got {temperature}",
        );
        self.temperature = Some(temperature);
        self
    }

    /// Set the maximum tokens to generate.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the top_p (nucleus sampling) parameter.
    ///
    /// Must be finite and in `0.0..=1.0` (it's a probability mass).
    /// Out-of-range / NaN is a caller logic error and panics.
    pub fn top_p(mut self, top_p: f32) -> Self {
        assert!(
            top_p.is_finite() && (0.0..=1.0).contains(&top_p),
            "top_p must be finite and in 0.0..=1.0, got {top_p}",
        );
        self.top_p = Some(top_p);
        self
    }

    /// Set stop sequences.
    pub fn stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Set presence penalty. Must be finite and in `-2.0..=2.0`
    /// (the widest range any supported provider accepts);
    /// out-of-range / NaN is a caller logic error and panics.
    pub fn presence_penalty(mut self, presence_penalty: f32) -> Self {
        assert!(
            presence_penalty.is_finite() && (-2.0..=2.0).contains(&presence_penalty),
            "presence_penalty must be finite and in -2.0..=2.0, got {presence_penalty}",
        );
        self.presence_penalty = Some(presence_penalty);
        self
    }

    /// Set frequency penalty. Must be finite and in `-2.0..=2.0`
    /// (the widest range any supported provider accepts);
    /// out-of-range / NaN is a caller logic error and panics.
    pub fn frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        assert!(
            frequency_penalty.is_finite() && (-2.0..=2.0).contains(&frequency_penalty),
            "frequency_penalty must be finite and in -2.0..=2.0, got {frequency_penalty}",
        );
        self.frequency_penalty = Some(frequency_penalty);
        self
    }

    /// Set tools/functions for function calling.
    pub fn tools(mut self, tools: Vec<super::message::Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set the tool choice strategy.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Allow or disallow parallel tool calls (OpenAI).
    pub fn parallel_tool_calls(mut self, parallel: bool) -> Self {
        self.parallel_tool_calls = Some(parallel);
        self
    }

    /// Whether to store the response server-side (OpenAI).
    pub fn store(mut self, store: bool) -> Self {
        self.store = Some(store);
        self
    }

    /// Configure reasoning (chain-of-thought) for the request.
    pub fn reasoning(mut self, reasoning: ReasoningConfig) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Constrain the response to a structured shape (JSON mode / schema).
    pub fn response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    /// Set the opaque tenant key the provider's
    /// [`crate::rate_limit::RateLimiter`] uses for fair queueing.
    /// Required for multi-tenant deployments — a missing tenant
    /// collapses every request into one anonymous tenant, so a
    /// single noisy caller can starve every other request through
    /// the shared limiter. The string is opaque; pick whatever
    /// identifies a quota-isolated unit (user id, workspace id, …).
    ///
    /// Accepts anything that converts into [`Arc<str>`](std::sync::Arc) (e.g. a
    /// `&str` or `String`); the same tenant string can be cheaply
    /// reused across many requests via `Arc::clone`.
    pub fn tenant(mut self, tenant: impl Into<std::sync::Arc<str>>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }

    /// Set the latency priority. See [`crate::Priority`] for the
    /// scheduling model — interactive beats standard beats
    /// background, strictly, across tenants.
    pub fn priority(mut self, priority: crate::rate_limit::Priority) -> Self {
        self.priority = Some(priority);
        self
    }

    /// Override the middleware chain. Pass `Vec::new()` to disable all
    /// polyfills (validation will still run and surface unsupported
    /// requests as `Error::Config`). Pass a custom list to add your
    /// own middleware or reorder the defaults.
    pub fn with_middleware(
        mut self,
        middleware: Vec<std::sync::Arc<dyn crate::middleware::Middleware>>,
    ) -> Self {
        self.middleware_override = Some(middleware);
        self
    }

    /// Finalize into a [`Config`]. Cheap — no capability resolution
    /// or middleware derivation happens here. Both are deferred to
    /// [`crate::generate`], which asks the provider for the model's
    /// capabilities and derives the default middleware list from
    /// them. Use [`Self::with_middleware`] to pin a specific chain
    /// before the call.
    pub fn build(self) -> Config {
        Config {
            raw: RawConfig {
                model: self.model,
                temperature: self.temperature,
                max_tokens: self.max_tokens,
                top_p: self.top_p,
                stop: self.stop,
                presence_penalty: self.presence_penalty,
                frequency_penalty: self.frequency_penalty,
                tools: self.tools,
                tool_choice: self.tool_choice,
                parallel_tool_calls: self.parallel_tool_calls,
                store: self.store,
                reasoning: self.reasoning,
                response_format: self.response_format,
                tenant: self.tenant,
                priority: self.priority,
            },
            middleware_override: self.middleware_override,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_chains() {
        let cfg = Config::builder("gpt-4")
            .temperature(0.8)
            .max_tokens(500)
            .top_p(0.9)
            .build();
        assert_eq!(cfg.raw().model, "gpt-4");
        assert_eq!(cfg.raw().temperature, Some(0.8));
        assert_eq!(cfg.raw().max_tokens, Some(500));
        assert_eq!(cfg.raw().top_p, Some(0.9));
        assert!(cfg.raw().tools.is_none());
    }

    #[test]
    #[should_panic(expected = "temperature must be finite")]
    fn temperature_out_of_range_panics() {
        ConfigBuilder::new("m").temperature(5.0);
    }

    #[test]
    #[should_panic(expected = "temperature must be finite")]
    fn temperature_nan_panics() {
        ConfigBuilder::new("m").temperature(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "top_p must be finite")]
    fn top_p_out_of_range_panics() {
        ConfigBuilder::new("m").top_p(1.5);
    }

    #[test]
    fn boundary_values_are_accepted() {
        let cfg = ConfigBuilder::new("m")
            .temperature(2.0)
            .top_p(0.0)
            .presence_penalty(-2.0)
            .frequency_penalty(2.0)
            .build();
        assert_eq!(cfg.raw().temperature, Some(2.0));
    }

    #[test]
    fn build_records_middleware_override() {
        let cfg = Config::builder("claude-sonnet-4-5")
            .with_middleware(Vec::new())
            .build();
        assert_eq!(cfg.middleware_override().map(|m| m.len()), Some(0));
    }
}
