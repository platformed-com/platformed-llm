use super::{create_weather_tool, ProviderConfig, ProviderTestSetup};
use crate::cross_provider::scripted::{load_fixture, ScriptedTransport, ScriptedTurn};
use platformed_llm::providers::OpenAIProvider;
use platformed_llm::transport::Transport;
use platformed_llm::Provider;
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

    fn build_provider() -> Pin<Box<dyn Provider>> {
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
                    "name": weather_tool.as_function().unwrap().name,
                    "description": weather_tool.as_function().unwrap().description,
                    "parameters": weather_tool.as_function().unwrap().parameters
                }
            ],
            "stream": true,
            "store": false
        });

        // The first turn's `response.completed` carries
        // `"id":"resp_1"`. The lib lifts that into a
        // `ProviderContinuation::OpenAI` on the CompleteResponse;
        // `with_response()` folds it into the conversation as an
        // `InputItem::Continuation`. The follow-up request therefore
        // sends ONLY the tool result + `previous_response_id` —
        // server-side state covers the elided prefix. (Prior to this
        // wiring, the lib silently dropped the continuation and re-
        // sent the full history, which is the bug this fixture now
        // pins against.)
        let followup = json!({
            "model": "gpt-4o-mini",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_abc123def456",
                    "output": "The weather in Paris is sunny with a temperature of 22°C (72°F). Perfect weather for sightseeing!"
                }
            ],
            "temperature": 0.7,
            "max_output_tokens": 150,
            "previous_response_id": "resp_1",
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
