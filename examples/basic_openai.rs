//! Basic example of using the platformed-llm library with OpenAI.
//! 
//! This example demonstrates Phase 1 functionality:
//! - Creating an OpenAI provider
//! - Building prompts
//! - Making non-streaming requests
//! - Handling responses
//!
//! To run this example, you need to set the OPENAI_API_KEY environment variable:
//! 
//! ```bash
//! export OPENAI_API_KEY=your_api_key_here
//! cargo run --example basic_openai
//! ```

use platformed_llm::{OpenAIProvider, InternalRequest, Prompt, Error};

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
        .with_user("What is the capital of France?");
    
    println!("Prompt has {} items", prompt.items().len());
    
    // Create an internal request
    let request = InternalRequest {
        model: "gpt-3.5-turbo".to_string(),
        messages: prompt.items().to_vec(),
        temperature: Some(0.7),
        max_tokens: Some(100),
        top_p: None,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        tools: None,
    };
    
    // Make the request (this will fail without a real API key)
    println!("Making request to OpenAI...");
    match provider.generate(&request).await {
        Ok(response) => {
            let text = response.text().await?;
            println!("Response: {text}");
        }
        Err(e) => {
            println!("Error (expected without real API key): {e}");
            
            // Demonstrate error handling
            match e {
                Error::Provider { provider, message } => {
                    println!("Provider '{provider}' error: {message}");
                }
                Error::Http(_) => {
                    println!("HTTP error occurred");
                }
                _ => {
                    println!("Other error: {e}");
                }
            }
        }
    }
    
    // Demonstrate different prompt creation methods
    println!("\nDemonstrating prompt creation methods:");
    
    // From string
    let prompt1: Prompt = "Hello, world!".into();
    println!("From string: {} items", prompt1.items().len());
    
    // Complex conversation
    let prompt2 = Prompt::new()
        .with_system("You are a helpful assistant")
        .with_user("What is 2+2?")
        .with_assistant("2+2 equals 4.")
        .with_user("What about 3+3?");
    println!("Complex conversation: {} items", prompt2.items().len());
    
    println!("\nPhase 1 basic functionality demonstrated!");
    Ok(())
}