//! Minimal example showing the simplest usage of the library.

use platformed_llm::{OpenAIProvider, InternalRequest, Prompt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load API key from environment
    dotenv::dotenv().ok();
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set");

    // Create provider and prompt
    let provider = OpenAIProvider::new(api_key)?;
    let prompt = Prompt::user("What is the capital of France?");

    // Make request
    let request = InternalRequest {
        model: "gpt-4o-mini".to_string(),
        messages: prompt.items().to_vec(),
        temperature: Some(0.7),
        max_tokens: Some(100),
        top_p: None,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        tools: None,
    };

    // Get response
    let response = provider.generate(&request).await?;
    let text = response.text().await?;
    println!("AI: {}", text);

    Ok(())
}