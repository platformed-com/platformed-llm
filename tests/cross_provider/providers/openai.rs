use super::{create_weather_tool, ProviderConfig, ProviderTestSetup};
use platformed_llm::{LLMProvider, OpenAIProvider};
use serde_json::json;
use std::pin::Pin;
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

pub struct OpenAITestSetup;

/// Load test fixture from file
fn load_fixture(filename: &str) -> String {
    std::fs::read_to_string(filename)
        .unwrap_or_else(|_| panic!("Failed to load test fixture: {filename}"))
}

#[async_trait::async_trait]
impl ProviderTestSetup for OpenAITestSetup {
    fn get_config() -> ProviderConfig {
        ProviderConfig {
            name: "OpenAI",
            model: "gpt-4o-mini",
            supports_custom_base_url: true,
        }
    }

    fn create_provider(base_url: &str) -> Pin<Box<dyn LLMProvider>> {
        let provider =
            OpenAIProvider::new_with_base_url("test-api-key".to_string(), base_url.to_string())
                .expect("Failed to create OpenAI provider");
        Box::pin(provider)
    }

    async fn mount_function_calling_mocks(
        mock_server: &MockServer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let weather_tool = create_weather_tool();
        let initial_request_payload = json!({
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
            "parallel_tool_calls": true,
            "stream": true,
            "store": false
        });

        let followup_request_payload = json!({
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
            "parallel_tool_calls": true,
            "stream": true,
            "store": false
        });

        // Mount initial request mock
        Mock::given(method("POST"))
            .and(path("/responses"))
            .and(body_json(initial_request_payload))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(load_fixture(
                        "tests/cross_provider/fixtures/openai/function_call_response.sse",
                    ))
                    .insert_header("content-type", "text/event-stream")
                    .insert_header("cache-control", "no-cache"),
            )
            .expect(1)
            .mount(mock_server)
            .await;

        // Mount follow-up request mock
        Mock::given(method("POST"))
            .and(path("/responses"))
            .and(body_json(followup_request_payload))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(load_fixture(
                        "tests/cross_provider/fixtures/openai/followup_response.sse",
                    ))
                    .insert_header("content-type", "text/event-stream")
                    .insert_header("cache-control", "no-cache"),
            )
            .expect(1)
            .mount(mock_server)
            .await;

        Ok(())
    }
}
