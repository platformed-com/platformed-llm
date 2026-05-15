use super::{create_weather_tool, ProviderConfig, ProviderTestSetup};
use crate::cross_provider::scripted::{load_fixture, ScriptedTransport, ScriptedTurn};
use platformed_llm::providers::{AnthropicViaVertexProvider, VertexEndpoint};
use platformed_llm::transport::Transport;
use platformed_llm::Provider;
use serde_json::json;
use std::pin::Pin;

pub struct AnthropicTestSetup;

impl ProviderTestSetup for AnthropicTestSetup {
    fn get_config() -> ProviderConfig {
        ProviderConfig {
            name: "Anthropic",
            model: "claude-3-5-sonnet-v2@20241022",
        }
    }

    fn build_provider() -> Pin<Box<dyn Provider>> {
        let weather_tool = create_weather_tool();
        let initial = json!({
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather like in Paris?"
                }
            ],
            "max_tokens": 150,
            "anthropic_version": "vertex-2023-10-16",
            "system": "You have access to weather data. Use the get_weather function when asked about weather.",
            "temperature": 0.7,
            "tools": [
                {
                    "name": weather_tool.as_function().unwrap().name,
                    "description": weather_tool.as_function().unwrap().description,
                    "input_schema": weather_tool.as_function().unwrap().parameters
                }
            ],
            "stream": true
        });

        let followup = json!({
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather like in Paris?"
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I'll help you get the weather for Paris."
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_123456",
                            "name": "get_weather",
                            "input": {"location": "Paris"}
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123456",
                            "content": "The weather in Paris is sunny with a temperature of 22°C (72°F). Perfect weather for sightseeing!"
                        }
                    ]
                }
            ],
            "max_tokens": 150,
            "anthropic_version": "vertex-2023-10-16",
            "system": "You have access to weather data. Use the get_weather function when asked about weather.",
            "temperature": 0.7,
            "stream": true
        });

        let scripted = ScriptedTransport::new(vec![
            ScriptedTurn {
                expected_body: initial,
                response_sse: load_fixture(
                    "tests/cross_provider/fixtures/anthropic/function_call_response.sse",
                ),
            },
            ScriptedTurn {
                expected_body: followup,
                response_sse: load_fixture(
                    "tests/cross_provider/fixtures/anthropic/followup_response.sse",
                ),
            },
        ]);
        let endpoint = VertexEndpoint::with_access_token(
            "test-project".to_string(),
            "europe-west1".to_string(),
            "test-access-token".to_string(),
        );
        let provider =
            AnthropicViaVertexProvider::with_transport(endpoint, Transport::new(scripted));
        Box::pin(provider)
    }
}
