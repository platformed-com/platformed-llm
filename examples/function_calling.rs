//! Simple example demonstrating function calling with the OpenAI provider.

use futures_util::StreamExt;
use platformed_llm::accumulator::ResponseAccumulator;
use platformed_llm::types::InputItem;
use platformed_llm::{Function, InternalRequest, OpenAIProvider, Prompt, Tool, ToolType};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load API key from environment
    dotenv::dotenv().ok();
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set");

    let provider = OpenAIProvider::new(api_key)?;

    // Define a simple weather function
    let weather_tool = Tool {
        r#type: ToolType::Function,
        function: Function {
            name: "get_weather".to_string(),
            description: "Get the current weather for a location".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }),
        },
    };

    // Step 1: Make a request that triggers function calling
    println!("=== Function Calling Example ===");
    let conversation = Prompt::system("You have access to weather data. Use the get_weather function when asked about weather.")
        .with_user("What's the weather like in Paris?");

    let request = InternalRequest {
        model: "gpt-4o-mini".to_string(),
        messages: conversation.items().to_vec(),
        temperature: Some(0.7),
        max_tokens: Some(150),
        top_p: None,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        tools: Some(vec![weather_tool]),
    };

    let response = provider.generate(&request).await?;
    let mut stream = response.stream();
    let mut accumulator = ResponseAccumulator::new();

    println!("AI response:");
    while let Some(event_result) = stream.next().await {
        let event = event_result?;
        accumulator.process_event(event.clone())?;

        match event {
            platformed_llm::StreamEvent::ContentDelta { delta } => {
                print!("{delta}");
            }
            platformed_llm::StreamEvent::FunctionCallComplete { call } => {
                println!("\\n[Function call: {} with args: {}]", call.name, call.arguments);
            }
            platformed_llm::StreamEvent::Done { .. } => {
                println!("\\n[Done]");
                break;
            }
            _ => {}
        }
    }

    // Step 2: Provide function results and continue the conversation
    println!("\\n=== Providing Function Results ===");
    let response = accumulator.finalize()?;
    let function_calls = response.function_calls();

    if !function_calls.is_empty() {
        let call = &function_calls[0];
        println!("Executing function: {}", call.name);
        
        // Mock function execution
        let result = if call.name == "get_weather" {
            "The weather in Paris is 22°C and partly cloudy."
        } else {
            "Function result not available."
        };
        
        println!("Function result: {}", result);

        // Continue conversation with function result
        let conversation = Prompt::new()
            .with_items(response.to_items())
            .with_item(InputItem::FunctionCallOutput {
                call_id: call.call_id.clone(),
                output: result.to_string(),
            });

        let followup_request = InternalRequest {
            model: "gpt-4o-mini".to_string(),
            messages: conversation.items().to_vec(),
            temperature: Some(0.7),
            max_tokens: Some(150),
            top_p: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: None, // No more tools needed
        };

        let followup_response = provider.generate(&followup_request).await?;
        let text = followup_response.text().await?;
        println!("Final AI response: {}", text);
    }

    Ok(())
}