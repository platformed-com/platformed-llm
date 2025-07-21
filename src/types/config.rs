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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Prompt, InputItem};

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