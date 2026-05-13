use super::{create_weather_tool, ProviderConfig, ProviderTestSetup};
use crate::cross_provider::scripted::{load_fixture, ScriptedTransport, ScriptedTurn};
use platformed_llm::providers::vertex::GoogleProvider;
use platformed_llm::{LLMProvider, Transport, VertexEndpoint};
use serde_json::json;
use std::pin::Pin;

pub struct GoogleTestSetup;

impl ProviderTestSetup for GoogleTestSetup {
    fn get_config() -> ProviderConfig {
        ProviderConfig {
            name: "Google",
            model: "gemini-1.5-pro",
        }
    }

    fn build_provider() -> Pin<Box<dyn LLMProvider>> {
        let weather_tool = create_weather_tool();
        let initial = json!({
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "What's the weather like in Paris?"}]
                }
            ],
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": "You have access to weather data. Use the get_weather function when asked about weather."}]
            },
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 150
            },
            "tools": [{
                "functionDeclarations": [{
                    "name": weather_tool.function.name,
                    "description": weather_tool.function.description,
                    "parameters": weather_tool.function.parameters
                }]
            }]
        });

        let followup = json!({
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "What's the weather like in Paris?"}]
                },
                {
                    "role": "model",
                    "parts": [
                        {"text": "I'll help you get the weather for Paris."},
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"location": "Paris"}
                            }
                        }
                    ]
                },
                {
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": "get_weather",
                            "response": {"result": "The weather in Paris is sunny with a temperature of 22°C (72°F). Perfect weather for sightseeing!"}
                        }
                    }]
                }
            ],
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": "You have access to weather data. Use the get_weather function when asked about weather."}]
            },
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 150
            }
        });

        let scripted = ScriptedTransport::new(vec![
            ScriptedTurn {
                expected_body: initial,
                response_sse: load_fixture(
                    "tests/cross_provider/fixtures/google/function_call_response.sse",
                ),
            },
            ScriptedTurn {
                expected_body: followup,
                response_sse: load_fixture(
                    "tests/cross_provider/fixtures/google/followup_response.sse",
                ),
            },
        ]);
        let endpoint = VertexEndpoint::with_access_token(
            "test-project".to_string(),
            "europe-west1".to_string(),
            "test-access-token".to_string(),
        );
        let provider = GoogleProvider::with_transport(endpoint, Transport::new(scripted));
        Box::pin(provider)
    }
}
