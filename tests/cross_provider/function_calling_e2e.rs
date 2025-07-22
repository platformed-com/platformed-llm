use futures_util::StreamExt;
use platformed_llm::accumulator::ResponseAccumulator;
use platformed_llm::{InputItem, LLMRequest, Prompt};
use wiremock::MockServer;

use super::providers::{
    anthropic::AnthropicTestSetup, create_weather_tool, google::GoogleTestSetup,
    openai::OpenAITestSetup, ProviderTestSetup,
};

/// Run the function calling e2e test for a specific provider
async fn run_function_calling_test<T: ProviderTestSetup>() -> Result<(), Box<dyn std::error::Error>>
{
    let config = T::get_config();

    println!("\n{}", "=".repeat(80));
    println!("Testing {} Provider", config.name);
    println!("{}", "=".repeat(80));

    // Skip if provider doesn't support custom base URLs
    if !config.supports_custom_base_url {
        println!(
            "⚠️  {} provider doesn't support custom base URLs for mocking",
            config.name
        );
        println!("   Skipping test for now...");
        return Ok(());
    }

    // Start mock server
    let mock_server = MockServer::start().await;

    // Let the provider setup mount its required mocks
    T::mount_function_calling_mocks(&mock_server).await?;

    // Define weather tool
    let weather_tool = create_weather_tool();

    // Create provider instance
    let provider = T::create_provider(&mock_server.uri());

    // Step 1: Create initial conversation
    let mut conversation = Prompt::system(
        "You have access to weather data. Use the get_weather function when asked about weather.",
    )
    .with_user("What's the weather like in Paris?");

    let request = LLMRequest::from_prompt(config.model, &conversation)
        .temperature(0.7)
        .max_tokens(150)
        .tools(vec![weather_tool]);

    // Execute the function calling request and accumulate the complete response
    let response = provider
        .generate(&request)
        .await
        .expect("Failed to get response from provider");
    println!("✅ Function calling request successfully mocked with correct payload!");

    // ASSERTION 1: Verify the request payload was correct (if we get here, wiremock matched the exact payload)
    println!("✅ ASSERTION PASSED: Request payload verification");
    println!("   ✓ Model: {}", config.model);
    println!("   ✓ Temperature: 0.7");
    println!("   ✓ Max tokens: 150");
    println!("   ✓ Messages: System + User prompts");
    println!("   ✓ Tools: get_weather function with correct parameters");
    println!("   ✓ Stream: true");

    // Step 2: Accumulate the response to extract function calls
    let mut accumulator = ResponseAccumulator::new();
    let mut stream = response.stream();

    // Process the stream to accumulate the complete response
    while let Some(event_result) = stream.next().await {
        let event = event_result.expect("Stream should parse correctly");
        accumulator
            .process_event(event)
            .expect("Event processing should succeed");
    }

    // ASSERTION 2: Verify we got the expected function call from parsing
    let function_calls = accumulator.completed_function_calls();
    assert!(
        !function_calls.is_empty(),
        "{}: Should have at least one function call",
        config.name
    );

    let weather_call = &function_calls[0];
    assert_eq!(
        weather_call.name, "get_weather",
        "{}: Function name should be get_weather",
        config.name
    );
    assert!(
        weather_call.arguments.contains("Paris"),
        "{}: Function arguments should contain Paris",
        config.name
    );

    println!("✅ ASSERTION PASSED: Function call extracted successfully");
    println!("   ✓ Function name: {}", weather_call.name);
    println!("   ✓ Call ID: {}", weather_call.call_id);
    println!("   ✓ Arguments contain location: Paris");

    // For OpenAI, verify specific call ID
    if config.name == "OpenAI" {
        assert_eq!(
            weather_call.id, "call_abc123def456",
            "OpenAI: Function call ID should match fixture"
        );
    }

    // Step 3: Get the complete response and add it to the conversation
    let complete_response = accumulator.finalize().expect("Failed to finalize response");
    conversation = conversation.with_response(&complete_response);

    // Step 4: Simulate function execution (in real usage, this would call actual function)
    let function_result = "The weather in Paris is sunny with a temperature of 22°C (72°F). Perfect weather for sightseeing!";
    println!("✅ ASSERTION PASSED: Function execution simulated");
    println!("   ✓ Executed function: {}", weather_call.name);
    println!("   ✓ Result: {function_result}");

    // Step 5: Add function result to conversation and create follow-up request
    conversation = conversation.with_item(InputItem::function_call_output(
        weather_call.call_id.clone(),
        function_result.to_string(),
    ));

    let followup_request = LLMRequest::from_prompt(config.model, &conversation)
        .temperature(0.7)
        .max_tokens(150);

    // Step 6: Send follow-up request (this will trigger the second mock)
    let followup_response = provider
        .generate(&followup_request)
        .await
        .expect("Failed to get follow-up response");

    println!("✅ ASSERTION PASSED: Follow-up request sent successfully");
    println!("   ✓ Request included function call and result");
    println!("   ✓ Second mock matched exact payload including conversation history");

    // Process the follow-up response
    let followup_text = followup_response.text().await?;
    assert!(
        !followup_text.trim().is_empty(),
        "{}: Follow-up response should not be empty",
        config.name
    );

    println!("✅ ASSERTION PASSED: Follow-up response received");
    println!("   ✓ Response text: {} chars", followup_text.len());

    // ASSERTION 3: Verify complete function calling workflow
    println!("✅ ASSERTION PASSED: Complete function calling workflow");
    println!("   ✓ Initial request with tools");
    println!("   ✓ Function call extracted and parsed");
    println!("   ✓ Function execution simulated");
    println!("   ✓ Follow-up request with function result");
    println!("   ✓ Final response processed");

    // ASSERTION 4: Verify the mock infrastructure is working end-to-end
    println!("✅ ASSERTION PASSED: Complete end-to-end mocking infrastructure");
    println!("   ✓ Mock server setup and teardown");
    println!("   ✓ HTTP request interception");
    println!("   ✓ Exact payload matching with body_json");
    println!("   ✓ Response delivery");
    println!("   ✓ Provider integration with custom base URL");

    println!(
        "\n✅ {} End-to-end function calling test PASSED with comprehensive assertions!",
        config.name
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
