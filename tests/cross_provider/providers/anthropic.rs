use super::{create_weather_tool, ProviderConfig, ProviderTestSetup};
use platformed_llm::providers::vertex::AnthropicProvider;
use platformed_llm::LLMProvider;
use serde_json::json;
use std::pin::Pin;
use wiremock::matchers::{body_json, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

pub struct AnthropicTestSetup;

/// Load test fixture from file
fn load_fixture(filename: &str) -> String {
    std::fs::read_to_string(filename)
        .unwrap_or_else(|_| panic!("Failed to load test fixture: {filename}"))
}

#[async_trait::async_trait]
impl ProviderTestSetup for AnthropicTestSetup {
    fn get_config() -> ProviderConfig {
        ProviderConfig {
            name: "Anthropic",
            model: "claude-3-5-sonnet-v2@20241022",
            supports_custom_base_url: true,
        }
    }

    fn create_provider(base_url: &str) -> Pin<Box<dyn LLMProvider>> {
        let provider = AnthropicProvider::new_with_base_url(
            "test-project".to_string(),
            "europe-west1".to_string(),
            "claude-3-5-sonnet-v2@20241022".to_string(),
            "test-access-token".to_string(),
            base_url.to_string(),
        )
        .expect("Failed to create Anthropic provider");
        Box::pin(provider)
    }

    async fn mount_function_calling_mocks(
        mock_server: &MockServer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let weather_tool = create_weather_tool();

        let initial_request_payload = json!({
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
                    "name": weather_tool.function.name,
                    "description": weather_tool.function.description,
                    "input_schema": weather_tool.function.parameters
                }
            ],
            "stream": true
        });

        let followup_request_payload = json!({
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

        // Mount initial request mock
        Mock::given(method("POST"))
            .and(path("/v1/projects/test-project/locations/europe-west1/publishers/anthropic/models/claude-3-5-sonnet-v2@20241022:streamRawPredict"))
            .and(query_param("alt", "sse"))
            .and(body_json(initial_request_payload))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(load_fixture("tests/cross_provider/fixtures/anthropic/function_call_response.sse"))
                    .insert_header("content-type", "text/event-stream")
                    .insert_header("cache-control", "no-cache")
            )
            .expect(1)
            .mount(mock_server)
            .await;

        // Mount follow-up request mock
        Mock::given(method("POST"))
            .and(path("/v1/projects/test-project/locations/europe-west1/publishers/anthropic/models/claude-3-5-sonnet-v2@20241022:streamRawPredict"))
            .and(query_param("alt", "sse"))
            .and(body_json(followup_request_payload))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(load_fixture("tests/cross_provider/fixtures/anthropic/followup_response.sse"))
                    .insert_header("content-type", "text/event-stream")
                    .insert_header("cache-control", "no-cache")
            )
            .expect(1)
            .mount(mock_server)
            .await;

        Ok(())
    }
}
