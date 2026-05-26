use serde::{Deserialize, Serialize};

/// Token usage information across providers.
///
/// Not every provider populates every field — fields specific to one
/// provider's billing model (cache create/read, reasoning) are `Option`
/// and stay `None` for providers that don't report them.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Usage {
    /// Tokens in the prompt.
    pub input_tokens: u32,
    /// Tokens in the completion.
    pub output_tokens: u32,
    /// Cached input tokens that were *read* from the prompt cache (charged
    /// at a discount). Reported by Anthropic as `cache_read_input_tokens`,
    /// by OpenAI under `input_tokens_details.cached_tokens`, and by Gemini
    /// as `cachedContentTokenCount`.
    pub cache_read_input_tokens: Option<u32>,
    /// Input tokens *written* to the cache on this request (Anthropic-only;
    /// charged at a 1.25× premium). Reported as
    /// `cache_creation_input_tokens`.
    pub cache_creation_input_tokens: Option<u32>,
    /// Output tokens spent on the model's internal reasoning (gpt-5 /
    /// o-series and Gemini thinking). OpenAI reports this under
    /// `output_tokens_details.reasoning_tokens`; Gemini reports it as
    /// `thoughtsTokenCount`.
    pub reasoning_tokens: Option<u32>,
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
/// - **Anthropic**: silently dropped — Anthropic has no native JSON
///   mode. Callers wanting structured output on Anthropic should use
///   tool-use coercion (a function tool with the schema).
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

/// Model selection plus sampling / tool / structured-output settings
/// for an LLM call. Independent of the prompt so a single `Config`
/// can be reused across many prompts targeting the same model.
///
/// `model` is required; every other field is `Option` and `None` means
/// "use the provider's default."
#[derive(Debug, Clone)]
pub struct Config {
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
}

impl Config {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_config_builder_chains() {
        let cfg = Config::new("gpt-4")
            .temperature(0.8)
            .max_tokens(500)
            .top_p(0.9);
        assert_eq!(cfg.model, "gpt-4");
        assert_eq!(cfg.temperature, Some(0.8));
        assert_eq!(cfg.max_tokens, Some(500));
        assert_eq!(cfg.top_p, Some(0.9));
        assert!(cfg.tools.is_none());
    }

    #[test]
    #[should_panic(expected = "temperature must be finite")]
    fn temperature_out_of_range_panics() {
        Config::new("m").temperature(5.0);
    }

    #[test]
    #[should_panic(expected = "temperature must be finite")]
    fn temperature_nan_panics() {
        Config::new("m").temperature(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "top_p must be finite")]
    fn top_p_out_of_range_panics() {
        Config::new("m").top_p(1.5);
    }

    #[test]
    fn boundary_values_are_accepted() {
        let cfg = Config::new("m")
            .temperature(2.0)
            .top_p(0.0)
            .presence_penalty(-2.0)
            .frequency_penalty(2.0);
        assert_eq!(cfg.temperature, Some(2.0));
    }
}
