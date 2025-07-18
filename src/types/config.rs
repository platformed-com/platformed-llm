use serde::{Deserialize, Serialize};

/// Configuration for different LLM providers.
#[derive(Debug, Clone)]
pub enum ProviderConfig {
    OpenAI { api_key: String },
    Gemini { project_id: String, location: String },
    AnthropicVertex { project_id: String, location: String },
}

/// Token usage information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cached_tokens: Option<u32>,
}

/// Internal request structure used by providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalRequest {
    pub model: String,
    pub messages: Vec<super::message::InputItem>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub tools: Option<Vec<super::message::Tool>>,
}