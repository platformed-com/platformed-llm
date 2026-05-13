//! Two-turn function-calling test driver, one variant per provider.
//!
//! Verifies that the lib's `convert_request` emits the exact JSON shape
//! each provider expects (asserted inside [`ScriptedTransport`]) and that
//! the round-trip — model emits tool call → caller appends the tool
//! result → caller sends the follow-up — produces a non-empty final
//! response.

use futures_util::StreamExt;
use platformed_llm::accumulator::ResponseAccumulator;
use platformed_llm::{InputItem, LLMRequest, Prompt};

use super::providers::{
    anthropic::AnthropicTestSetup, create_weather_tool, google::GoogleTestSetup,
    openai::OpenAITestSetup, ProviderTestSetup,
};

async fn run_function_calling_test<T: ProviderTestSetup>() -> Result<(), Box<dyn std::error::Error>>
{
    let config = T::get_config();
    let provider = T::build_provider();

    let mut conversation = Prompt::system(
        "You have access to weather data. Use the get_weather function when asked about weather.",
    )
    .with_user("What's the weather like in Paris?");

    let request = LLMRequest::from_prompt(config.model, &conversation)
        .temperature(0.7)
        .max_tokens(150)
        .tools(vec![create_weather_tool()]);

    // First turn: ScriptedTransport asserts the lib's emitted request
    // body matches the expected initial payload, then returns the
    // canned function-call SSE.
    let response = provider.generate(&request).await?;

    let mut accumulator = ResponseAccumulator::new();
    let mut stream = response.stream();
    while let Some(event_result) = stream.next().await {
        accumulator.process_event(event_result?)?;
    }

    let function_calls = accumulator.completed_function_calls();
    assert!(
        !function_calls.is_empty(),
        "{}: expected at least one function call",
        config.name,
    );
    let weather_call = &function_calls[0];
    assert_eq!(
        weather_call.name, "get_weather",
        "{}: function name",
        config.name,
    );
    assert!(
        weather_call.arguments.contains("Paris"),
        "{}: function arguments should contain Paris, got {}",
        config.name,
        weather_call.arguments,
    );
    if config.name == "OpenAI" {
        assert_eq!(
            weather_call.call_id, "call_abc123def456",
            "OpenAI: call_id should match fixture",
        );
    }

    // Fold the model's tool emission back into the conversation, append
    // the tool result, and send the follow-up.
    let complete_response = accumulator.finalize()?;
    conversation = conversation.with_response(&complete_response);
    conversation = conversation.with_item(InputItem::function_call_output(
        weather_call.call_id.clone(),
        "The weather in Paris is sunny with a temperature of 22°C (72°F). \
         Perfect weather for sightseeing!"
            .to_string(),
    ));

    let followup_request = LLMRequest::from_prompt(config.model, &conversation)
        .temperature(0.7)
        .max_tokens(150);

    // Second turn: ScriptedTransport asserts the follow-up body shape.
    let followup_response = provider.generate(&followup_request).await?;
    let followup_text = followup_response.text().await?;
    assert!(
        !followup_text.trim().is_empty(),
        "{}: follow-up response should not be empty",
        config.name,
    );

    Ok(())
}

#[tokio::test]
async fn test_openai_function_calling_e2e() {
    run_function_calling_test::<OpenAITestSetup>()
        .await
        .expect("OpenAI function calling test failed");
}

#[tokio::test]
async fn test_google_function_calling_e2e() {
    run_function_calling_test::<GoogleTestSetup>()
        .await
        .expect("Google function calling test failed");
}

#[tokio::test]
async fn test_anthropic_function_calling_e2e() {
    run_function_calling_test::<AnthropicTestSetup>()
        .await
        .expect("Anthropic function calling test failed");
}
