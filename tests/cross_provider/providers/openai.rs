use super::{create_weather_tool, ProviderConfig, ProviderTestSetup};
use crate::cross_provider::scripted::{load_fixture, ScriptedTransport, ScriptedTurn};
use platformed_llm::{LLMProvider, OpenAIProvider, Transport};
use serde_json::json;
use std::pin::Pin;

pub struct OpenAITestSetup;

impl ProviderTestSetup for OpenAITestSetup {
    fn get_config() -> ProviderConfig {
        ProviderConfig {
            name: "OpenAI",
            model: "gpt-4o-mini",
        }
    }

    fn build_provider() -> Pin<Box<dyn LLMProvider>> {
        let weather_tool = create_weather_tool();
        let initial = json!({
            "model": "gpt-4o-mini",
            "input": [
                {
                    "type": "message",
                    "role": "system",
                    "content": "You have access to weather data. Use the get_weather function when asked about weather."
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": "What's the weather like in Paris?"
                }
            ],
            "temperature": 0.7,
            "max_output_tokens": 150,
            "tools": [
                {
                    "type": "function",
                    "name": weather_tool.function.name,
                    "description": weather_tool.function.description,
                    "parameters": weather_tool.function.parameters
                }
            ],
            "stream": true,
            "store": false
        });

        let followup = json!({
            "model": "gpt-4o-mini",
            "input": [
                {
                    "type": "message",
                    "role": "system",
                    "content": "You have access to weather data. Use the get_weather function when asked about weather."
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": "What's the weather like in Paris?"
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "I'll help you get the weather for Paris."
                },
                {
                    "type": "function_call",
                    "call_id": "call_abc123def456",
                    "name": "get_weather",
                    "arguments": "{\"location\": \"Paris\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_abc123def456",
                    "output": "The weather in Paris is sunny with a temperature of 22°C (72°F). Perfect weather for sightseeing!"
                }
            ],
            "temperature": 0.7,
            "max_output_tokens": 150,
            "stream": true,
            "store": false
        });

        let scripted = ScriptedTransport::new(vec![
            ScriptedTurn {
                expected_body: initial,
                response_sse: load_fixture(
                    "tests/cross_provider/fixtures/openai/function_call_response.sse",
                ),
            },
            ScriptedTurn {
                expected_body: followup,
                response_sse: load_fixture(
                    "tests/cross_provider/fixtures/openai/followup_response.sse",
                ),
            },
        ]);
        let provider = OpenAIProvider::with_transport(
            "test-api-key".to_string(),
            "http://placeholder".to_string(),
            Transport::new(scripted),
        );
        Box::pin(provider)
    }
}
