use super::{create_weather_tool, ProviderConfig, ProviderTestSetup};
use platformed_llm::providers::vertex::GoogleProvider;
use platformed_llm::LLMProvider;
use serde_json::json;
use std::pin::Pin;
use wiremock::matchers::{body_json, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

pub struct GoogleTestSetup;

/// Load test fixture from file
fn load_fixture(filename: &str) -> String {
    std::fs::read_to_string(filename)
        .unwrap_or_else(|_| panic!("Failed to load test fixture: {filename}"))
}

#[async_trait::async_trait]
impl ProviderTestSetup for GoogleTestSetup {
    fn get_config() -> ProviderConfig {
        ProviderConfig {
            name: "Google",
            model: "gemini-1.5-pro",
            supports_custom_base_url: true, // Now supports custom base URLs!
        }
    }

    fn create_provider(base_url: &str) -> Pin<Box<dyn LLMProvider>> {
        let provider = GoogleProvider::new_with_base_url(
            "test-project".to_string(),
            "europe-west1".to_string(),
            "gemini-1.5-pro".to_string(),
            "test-access-token".to_string(),
            base_url.to_string(),
        )
        .expect("Failed to create Google provider");
        Box::pin(provider)
    }

    async fn mount_function_calling_mocks(
        mock_server: &MockServer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let weather_tool = create_weather_tool();
        let initial_request_payload = json!({
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "What's the weather like in Paris?"}]
                }
            ],
            "system_instruction": {
                "role": "user",
                "parts": [{"text": "You have access to weather data. Use the get_weather function when asked about weather."}]
            },
            "generation_config": {
                "temperature": 0.7,
                "max_output_tokens": 150
            },
            "tools": [{
                "function_declarations": [{
                    "name": weather_tool.function.name,
                    "description": weather_tool.function.description,
                    "parameters": weather_tool.function.parameters
                }]
            }]
        });

        let followup_request_payload = json!({
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
            "system_instruction": {
                "role": "user",
                "parts": [{"text": "You have access to weather data. Use the get_weather function when asked about weather."}]
            },
            "generation_config": {
                "temperature": 0.7,
                "max_output_tokens": 150
            }
        });

        // Mount initial request mock with Google-specific query parameter
        Mock::given(method("POST"))
            .and(path("/v1/projects/test-project/locations/europe-west1/publishers/google/models/gemini-1.5-pro:streamGenerateContent"))
            .and(query_param("alt", "sse"))
            .and(body_json(initial_request_payload))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(load_fixture("tests/cross_provider/fixtures/google/function_call_response.sse"))
                    .insert_header("content-type", "text/event-stream")
                    .insert_header("cache-control", "no-cache")
            )
            .expect(1)
            .mount(mock_server)
            .await;

        // Mount follow-up request mock
        Mock::given(method("POST"))
            .and(path("/v1/projects/test-project/locations/europe-west1/publishers/google/models/gemini-1.5-pro:streamGenerateContent"))
            .and(query_param("alt", "sse"))
            .and(body_json(followup_request_payload))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(load_fixture("tests/cross_provider/fixtures/google/followup_response.sse"))
                    .insert_header("content-type", "text/event-stream")
                    .insert_header("cache-control", "no-cache")
            )
            .expect(1)
            .mount(mock_server)
            .await;

        Ok(())
    }
}
