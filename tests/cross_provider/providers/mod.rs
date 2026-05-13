pub mod anthropic;
pub mod google;
pub mod openai;

use platformed_llm::{Function, LLMProvider, Tool, ToolType};
use std::pin::Pin;

/// Create a weather function tool for testing
pub fn create_weather_tool() -> Tool {
    Tool {
        r#type: ToolType::Function,
        function: Function {
            name: "get_weather".to_string(),
            description: Some("Get the current weather for a location".to_string()),
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
}

/// Provider-specific test setup. Each impl builds a fully-wired provider
/// backed by a `ScriptedTransport` that asserts the lib's emitted request
/// body matches the expected payload for each of the two scripted turns
/// (initial tool-emitting call + follow-up after the tool result).
pub trait ProviderTestSetup {
    fn get_config() -> ProviderConfig;
    fn build_provider() -> Pin<Box<dyn LLMProvider>>;
}
