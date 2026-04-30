use serde::{Deserialize, Serialize};

/// Configuration for different LLM providers.
#[derive(Debug, Clone)]
pub enum ProviderConfig {
    OpenAI {
        api_key: String,
    },
    Gemini {
        project_id: String,
        location: String,
    },
    AnthropicVertex {
        project_id: String,
        location: String,
    },
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningSummary {
    Auto,
    Concise,
    Detailed,
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
    Function { name: String },
}

/// Request structure used by LLM providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMRequest {
    pub model: String,
    pub messages: Vec<super::message::InputItem>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub tools: Option<Vec<super::message::Tool>>,
    /// How the model should choose among tools.
    #[serde(default, skip_serializing_if = "Option::is_none", skip)]
    pub tool_choice: Option<ToolChoice>,
    /// Whether to allow more than one tool call per turn (OpenAI). `None`
    /// uses the provider's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Whether OpenAI should retain the response server-side for use with
    /// `previous_response_id` chaining. `None` uses the provider's default
    /// (which is currently `true` for OpenAI).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Reasoning configuration. Only meaningful for models that support
    /// chain-of-thought reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none", skip)]
    pub reasoning: Option<ReasoningConfig>,
}

impl LLMRequest {
    /// Create a new request with required fields.
    pub fn new(model: impl Into<String>, messages: Vec<super::message::InputItem>) -> Self {
        Self {
            model: model.into(),
            messages,
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
        }
    }

    /// Create a new request from a Prompt.
    pub fn from_prompt(model: impl Into<String>, prompt: &crate::Prompt) -> Self {
        Self::new(model, prompt.items().to_vec())
    }

    /// Set the temperature (randomness) parameter.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the maximum tokens to generate.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the top_p (nucleus sampling) parameter.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set stop sequences.
    pub fn stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Set presence penalty.
    pub fn presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.presence_penalty = Some(presence_penalty);
        self
    }

    /// Set frequency penalty.
    pub fn frequency_penalty(mut self, frequency_penalty: f32) -> Self {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InputItem, Prompt};

    #[test]
    fn test_llm_request_builder() {
        let prompt = Prompt::user("Hello");

        let request = LLMRequest::from_prompt("gpt-4", &prompt)
            .temperature(0.8)
            .max_tokens(500)
            .top_p(0.9);

        assert_eq!(request.model, "gpt-4");
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.temperature, Some(0.8));
        assert_eq!(request.max_tokens, Some(500));
        assert_eq!(request.top_p, Some(0.9));
        assert!(request.tools.is_none());
    }

    #[test]
    fn test_llm_request_minimal() {
        let messages = vec![InputItem::user("Test")];

        let request = LLMRequest::new("gpt-3.5-turbo", messages);

        assert_eq!(request.model, "gpt-3.5-turbo");
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.temperature, None);
        assert_eq!(request.max_tokens, None);
    }
}
