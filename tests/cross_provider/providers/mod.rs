pub mod anthropic;
pub mod google;
pub mod openai;

use platformed_llm::{Function, LLMProvider, Tool, ToolType};
use std::pin::Pin;
use wiremock::MockServer;

/// Create a weather function tool for testing
pub fn create_weather_tool() -> Tool {
    Tool {
        r#type: ToolType::Function,
        function: Function {
            name: "get_weather".to_string(),
            description: "Get the current weather for a location".to_string(),
            parameters: serde_json::from_str(
                r#"{
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }"#,
            )
            .unwrap(),
        },
    }
}

/// Provider configuration for cross-provider testing
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub name: &'static str,
    pub model: &'static str,
    pub supports_custom_base_url: bool,
}

/// Trait for provider-specific test setup
#[async_trait::async_trait]
pub trait ProviderTestSetup {
    /// Get the provider configuration
    fn get_config() -> ProviderConfig;

    /// Create the provider instance
    fn create_provider(base_url: &str) -> Pin<Box<dyn LLMProvider>>;

    /// Mount the required mocks for function calling test on the provided mock server
    async fn mount_function_calling_mocks(
        mock_server: &MockServer,
    ) -> Result<(), Box<dyn std::error::Error>>;
}
