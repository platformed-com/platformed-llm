use futures_util::StreamExt;
use ijson::IValue;
use platformed_llm::accumulator::ResponseAccumulator;
use platformed_llm::{
    generate, Config, Error, Function, Prompt, ProviderFactory, StreamEvent, Tool,
};

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Load environment variables
    dotenvy::dotenv().ok();

    println!("🚀 Universal Function Calling Example");
    println!("📋 This example works with any configured LLM provider");
    println!();

    // Check environment variables first
    println!("🔍 Environment check:");
    println!(
        "  OPENAI_API_KEY: {}",
        if std::env::var("OPENAI_API_KEY").is_ok() {
            "Set"
        } else {
            "Not set"
        }
    );
    println!(
        "  VERTEX_ACCESS_TOKEN: {}",
        if std::env::var("VERTEX_ACCESS_TOKEN").is_ok() {
            "Set"
        } else {
            "Not set"
        }
    );
    println!(
        "  GOOGLE_APPLICATION_CREDENTIALS: {}",
        if std::env::var("GOOGLE_APPLICATION_CREDENTIALS").is_ok() {
            "Set"
        } else {
            "Not set"
        }
    );
    println!("  PROVIDER_TYPE: {:?}", std::env::var("PROVIDER_TYPE"));

    // Create provider automatically from environment
    let provider = match ProviderFactory::from_env().await {
        Ok(provider) => {
            println!("✅ Provider created successfully from environment");
            provider
        }
        Err(e) => {
            println!("❌ Could not create provider: {e}");
            println!();
            println!("💡 Set one of these environment variables:");
            println!("   • OPENAI_API_KEY - for OpenAI GPT models");
            println!("   • VERTEX_ACCESS_TOKEN + PROVIDER_TYPE=anthropic - for Anthropic Claude");
            println!("   • VERTEX_ACCESS_TOKEN + PROVIDER_TYPE=google - for Google Gemini");
            println!("   • Or configure Application Default Credentials");
            return Err(e);
        }
    };

    println!();

    // Define function tools
    let get_weather = Tool::Function(Function {
        name: "get_weather".to_string(),
        description: Some("Get the current weather for a location".to_string()),
        parameters: serde_json::from_str(
            r#"{
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state/country, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }"#,
        )
        .unwrap(),
    });

    let calculate = Tool::Function(Function {
        name: "calculate".to_string(),
        description: Some("Perform mathematical calculations".to_string()),
        parameters: serde_json::from_str(
            r#"{
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. '2 + 2'"
                    }
                },
                "required": ["expression"]
            }"#,
        )
        .unwrap(),
    });

    // Start a conversation with function calling
    println!("🛠️ Function Calling Demo");
    println!("{}", "─".repeat(50));

    let mut conversation = Prompt::system(
        "You are a helpful assistant with access to weather information and a calculator. \
         Use the provided functions when users ask about weather or need calculations.",
    )
    .with_user("What's the weather like in Tokyo? Also, what's 15 multiplied by 23?");

    // Get model name based on configured provider type
    let model_name = std::env::var("MODEL_NAME").unwrap_or_else(|_| {
        // Check PROVIDER_TYPE first (most reliable)
        if let Ok(provider_type) = std::env::var("PROVIDER_TYPE") {
            match provider_type.to_lowercase().as_str() {
                "openai" => {
                    println!("🔍 Using OpenAI provider (explicit)");
                    "gpt-4o-mini".to_string()
                }
                "google" => {
                    println!("🔍 Using Google provider (explicit)");
                    "gemini-1.5-pro".to_string()
                }
                "anthropic" => {
                    println!("🔍 Using Anthropic provider (explicit)");
                    "claude-3-5-sonnet-v2@20241022".to_string()
                }
                _ => {
                    println!("🔍 Unknown PROVIDER_TYPE, using OpenAI fallback");
                    "gpt-4o-mini".to_string()
                }
            }
        } else {
            // Fall back to credential-based detection
            if std::env::var("OPENAI_API_KEY").is_ok() {
                println!("🔍 Detected OpenAI provider (credentials)");
                "gpt-4o-mini".to_string()
            } else if std::env::var("VERTEX_ACCESS_TOKEN").is_ok()
                || std::env::var("GOOGLE_APPLICATION_CREDENTIALS").is_ok()
            {
                // Without explicit PROVIDER_TYPE, default to Google for Vertex
                println!("🔍 Detected Google provider (credentials)");
                "gemini-1.5-pro".to_string()
            } else {
                println!("🔍 No credentials found, using OpenAI fallback");
                "gpt-4o-mini".to_string()
            }
        }
    });

    println!("🎯 Using model: {model_name}");

    let cfg = Config::builder(&model_name)
        .temperature(0.2)
        .max_tokens(300)
        .tools(vec![get_weather.clone(), calculate.clone()])
        .build();

    println!("👤 User: What's the weather like in Tokyo? Also, what's 15 multiplied by 23?");
    println!();

    // Generate response with function calling
    println!("📡 Making API request...");
    let response = match generate(&*provider, &conversation, &cfg).await {
        Ok(response) => {
            println!("✅ API request successful");
            response
        }
        Err(e) => {
            println!("❌ API request failed: {e}");
            return Err(e);
        }
    };

    // Process function calls in a loop until no more are needed
    let mut current_response = response;
    let mut round = 0;

    loop {
        round += 1;
        println!("🔄 Round {round} - Processing response...");

        let mut accumulator = ResponseAccumulator::new();
        let mut stream = current_response.stream();
        let mut event_count = 0;
        let mut text_output = String::new();

        while let Some(event_result) = stream.next().await {
            let event = event_result?;
            event_count += 1;
            println!("📥 Event #{event_count}: {event:?}");

            // Track text output (filter by part kind via accumulator afterwards;
            // here we just check if it's a text delta by looking at the part).
            if let StreamEvent::Delta { delta, .. } = &event {
                text_output.push_str(delta);
            }

            accumulator.process_event(event)?;
        }

        println!("🏁 Processed {event_count} total events");

        // Print any text output from this round
        if !text_output.trim().is_empty() {
            println!("💬 AI said: {}", text_output.trim());
        }

        // Get function calls before finalizing
        let function_calls = accumulator.completed_function_calls();

        // Get the complete response
        let complete_response = accumulator.finalize()?;

        // Add the AI response to conversation
        conversation = conversation.with_response(&complete_response);

        if function_calls.is_empty() {
            // No more function calls, we're done
            println!("✅ No more function calls needed");
            if !complete_response.text().trim().is_empty()
                && complete_response.text() != text_output
            {
                println!("🤖 Final AI response: {}", complete_response.text());
            }
            break;
        }

        println!("🔧 AI wants to call {} function(s):", function_calls.len());

        for (i, call) in function_calls.iter().enumerate() {
            println!(
                "  {}. {} with arguments: {}",
                i + 1,
                call.name,
                call.arguments
            );
        }

        println!();

        // Simulate function execution and collect all results
        let mut function_results = Vec::new();

        for call in &function_calls {
            let result = match call.name.as_str() {
                "get_weather" => {
                    // Parse the location from arguments
                    let args: IValue = serde_json::from_str(&call.arguments)?;
                    let location = args["location"]
                        .as_string()
                        .map(|s| &**s)
                        .unwrap_or("Unknown");

                    println!("🌤️ Calling weather API for {location}...");

                    // Simulate weather API response
                    match location.to_lowercase().as_str() {
                        l if l.contains("tokyo") => {
                            "The weather in Tokyo, Japan is currently sunny with a temperature of 24°C (75°F). \
                             Humidity is at 65% with light winds from the east at 8 km/h. Perfect weather for exploring the city!"
                        }
                        l if l.contains("london") => {
                            "The weather in London, UK is currently cloudy with a temperature of 16°C (61°F). \
                             There's a 40% chance of light rain. Humidity is at 78% with gentle winds from the southwest."
                        }
                        l if l.contains("new york") => {
                            "The weather in New York, NY is currently partly cloudy with a temperature of 22°C (72°F). \
                             Clear skies expected later today. Humidity is at 58% with moderate winds from the west."
                        }
                        _ => {
                            "The weather is partly cloudy with a temperature of 20°C (68°F). \
                             Humidity is moderate with light winds. A pleasant day overall!"
                        }
                    }
                }
                "calculate" => {
                    // Parse the expression from arguments
                    let args: IValue = serde_json::from_str(&call.arguments)?;
                    let expression = args["expression"].as_string().map(|s| &**s).unwrap_or("0");

                    println!("🧮 Calculating '{expression}'...");

                    // Simple calculator simulation
                    match expression.trim() {
                        e if e.contains("15") && e.contains("23") => "15 × 23 = 345",
                        e if e.contains("2 + 2") || e.contains("2+2") => "2 + 2 = 4",
                        e if e.contains("10 * 5") || e.contains("10*5") => "10 × 5 = 50",
                        _ => "Calculation completed: The result depends on the specific expression provided.",
                    }
                }
                _ => "Function not implemented in this demo",
            };

            println!("✅ Function result: {result}");
            function_results.push((call.call_id.clone(), result.to_string()));
        }

        // Add all function results to the conversation at once
        for (call_id, result) in function_results {
            conversation = conversation.with_tool_result(call_id, result);
        }

        // Continue with the next request
        println!();
        println!("🔁 Sending function results back to AI...");

        current_response = generate(&*provider, &conversation, &cfg).await?;
    }

    println!();
    println!("🎉 Universal function calling example completed!");
    println!();
    println!("💡 Key Features:");
    println!("   • Works with any configured LLM provider");
    println!("   • Automatic provider detection from environment");
    println!("   • Unified function calling interface");
    println!("   • Complete conversation flow with function results");
    println!();
    println!("🔧 Supported Providers:");
    println!("   • OpenAI GPT models (set OPENAI_API_KEY)");
    println!("   • Google Gemini via Vertex AI (set VERTEX_ACCESS_TOKEN or use ADC)");
    println!("   • Easy to extend for future providers");

    Ok(())
}
