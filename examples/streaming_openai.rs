//! Streaming example of using the platformed-llm library with OpenAI.
//! 
//! This example demonstrates Phase 2 functionality:
//! - Creating an OpenAI provider with streaming support
//! - Building prompts
//! - Making streaming requests
//! - Handling streaming responses (both stream() and buffer())
//!
//! To run this example, you need to set the OPENAI_API_KEY environment variable:
//! 
//! ```bash
//! export OPENAI_API_KEY=your_api_key_here
//! cargo run --example streaming_openai
//! ```

use platformed_llm::{OpenAIProvider, InternalRequest, Prompt, Error, StreamEvent};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Load .env file if it exists
    dotenv::dotenv().ok();
    
    // Get API key from environment variable (use placeholder if not set)
    let api_key = std::env::var("OPENAI_API_KEY")
        .unwrap_or_else(|_| {
            println!("Note: OPENAI_API_KEY not set, using placeholder (API calls will fail)");
            "placeholder-key".to_string()
        });
    
    // Create OpenAI provider
    println!("Creating OpenAI provider...");
    let provider = OpenAIProvider::new(api_key)?;
    
    // Build a prompt using the builder pattern
    let prompt = Prompt::system("You are a helpful assistant that responds concisely.")
        .with_user("Tell me a short story about a robot learning to paint.");
    
    println!("Prompt has {} items", prompt.items().len());
    
    // Create an internal request
    let request = InternalRequest {
        model: "gpt-3.5-turbo".to_string(),
        messages: prompt.items().to_vec(),
        temperature: Some(0.7),
        max_tokens: Some(150),
        top_p: None,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        tools: None,
    };
    
    // Demonstrate streaming (this will fail without a real API key)
    println!("\n=== Streaming Example ===");
    match provider.generate(&request).await {
        Ok(response) => {
            println!("Streaming response received!");
            
            // Option 1: Stream the events
            println!("\n--- Streaming Events ---");
            let mut stream = response.stream();
            while let Some(event_result) = stream.next().await {
                match event_result {
                    Ok(event) => {
                        match event {
                            StreamEvent::ContentDelta { delta } => {
                                print!("{delta}");
                                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                            }
                            StreamEvent::OutputItemAdded { item } => {
                                match item {
                                    platformed_llm::types::OutputItemInfo::Text => {
                                        println!("\n[Text output item added]");
                                    }
                                    platformed_llm::types::OutputItemInfo::FunctionCall { name, id } => {
                                        println!("\n[Function call output item added: {name} (ID: {id})]");
                                    }
                                }
                            }
                            StreamEvent::Done { finish_reason, usage } => {
                                println!("\n\nStream finished with reason: {finish_reason:?}");
                                println!("Usage: {usage:?}");
                                break;
                            }
                            StreamEvent::FunctionCallComplete { call } => {
                                println!("[Function call completed: {name} with args: {arguments}]", name=call.name, arguments=call.arguments);
                            }
                            StreamEvent::Error { error } => {
                                println!("\n[Stream error: {error}]");
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        println!("\nStream error: {e}");
                        break;
                    }
                }
            }
        }
        Err(e) => {
            println!("Error (expected without real API key): {e}");
            demonstrate_non_streaming().await?;
        }
    }
    
    // Option 2: Buffer the entire response (new request)
    println!("\n\n=== Buffered Example ===");
    let buffered_request = InternalRequest {
        model: "gpt-3.5-turbo".to_string(),
        messages: Prompt::system("You are a helpful assistant.")
            .with_user("What is 2+2? Answer in one sentence.")
            .items().to_vec(),
        temperature: Some(0.7),
        max_tokens: Some(50),
        top_p: None,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        tools: None,
    };
    
    match provider.generate(&buffered_request).await {
        Ok(response) => {
            let complete = response.buffer().await?;
            println!("Complete response: {}", complete.content());
            println!("Finish reason: {:?}", complete.finish_reason);
            println!("Usage: {:?}", complete.usage);
            
            if !complete.function_calls().is_empty() {
                println!("Function calls:");
                for call in &complete.function_calls() {
                    println!("  - {}: {}", call.name, call.arguments);
                }
            }
        }
        Err(e) => {
            println!("Error (expected without real API key): {e}");
        }
    }
    
    // Option 3: Get just the text (convenience method - new request)
    println!("\n=== Text-only Example ===");
    let text_request = InternalRequest {
        model: "gpt-3.5-turbo".to_string(),
        messages: Prompt::system("You are a helpful assistant.")
            .with_user("Say hello in French.")
            .items().to_vec(),
        temperature: Some(0.7),
        max_tokens: Some(20),
        top_p: None,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        tools: None,
    };
    
    match provider.generate(&text_request).await {
        Ok(response) => {
            let text = response.text().await?;
            println!("Response text: {text}");
        }
        Err(e) => {
            println!("Error (expected without real API key): {e}");
        }
    }
    
    println!("\nPhase 2 streaming functionality demonstrated!");
    Ok(())
}

/// Demonstrate the streaming architecture with mock data when API calls fail.
async fn demonstrate_non_streaming() -> Result<(), Error> {
    println!("\n--- Mock Streaming Demo ---");
    println!("Since the API call failed, here's how streaming would work:");
    
    // Create mock stream events
    let mock_events = vec![
        Ok(StreamEvent::ContentDelta { delta: "Once ".to_string() }),
        Ok(StreamEvent::ContentDelta { delta: "upon ".to_string() }),
        Ok(StreamEvent::ContentDelta { delta: "a ".to_string() }),
        Ok(StreamEvent::ContentDelta { delta: "time, ".to_string() }),
        Ok(StreamEvent::ContentDelta { delta: "there ".to_string() }),
        Ok(StreamEvent::ContentDelta { delta: "was ".to_string() }),
        Ok(StreamEvent::ContentDelta { delta: "a ".to_string() }),
        Ok(StreamEvent::ContentDelta { delta: "robot ".to_string() }),
        Ok(StreamEvent::ContentDelta { delta: "who ".to_string() }),
        Ok(StreamEvent::ContentDelta { delta: "learned ".to_string() }),
        Ok(StreamEvent::ContentDelta { delta: "to ".to_string() }),
        Ok(StreamEvent::ContentDelta { delta: "paint.".to_string() }),
        Ok(StreamEvent::Done { 
            finish_reason: platformed_llm::FinishReason::Stop, 
            usage: platformed_llm::Usage::default() 
        }),
    ];
    
    let mock_stream = futures_util::stream::iter(mock_events);
    let response = platformed_llm::Response::from_stream(mock_stream);
    
    // Demonstrate streaming
    println!("Streaming: ");
    let mut stream = response.stream();
    while let Some(event_result) = stream.next().await {
        match event_result? {
            StreamEvent::ContentDelta { delta } => {
                print!("{delta}");
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
            StreamEvent::Done { .. } => {
                println!("\n[Done]");
                break;
            }
            _ => {}
        }
    }
    
    Ok(())
}