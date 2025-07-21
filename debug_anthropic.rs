use platformed_llm::{Error, LLMRequest, Prompt, ProviderFactory};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Error> {
    dotenv::dotenv().ok();
    
    println!("🔍 Debugging Anthropic Provider Response");
    
    // Create provider
    let provider = match ProviderFactory::from_env().await {
        Ok(provider) => {
            println!("✅ Provider created successfully");
            provider
        }
        Err(e) => {
            println!("❌ Failed to create provider: {}", e);
            return Err(e);
        }
    };
    
    // Check environment variables
    println!("🔍 Environment check:");
    println!("  PROVIDER_TYPE: {:?}", std::env::var("PROVIDER_TYPE"));
    println!("  VERTEX_ACCESS_TOKEN: {}", if std::env::var("VERTEX_ACCESS_TOKEN").is_ok() { "Set" } else { "Not set" });
    println!("  GOOGLE_APPLICATION_CREDENTIALS: {}", if std::env::var("GOOGLE_APPLICATION_CREDENTIALS").is_ok() { "Set" } else { "Not set" });
    println!("  GOOGLE_CLOUD_PROJECT: {:?}", std::env::var("GOOGLE_CLOUD_PROJECT"));
    println!("  ANTHROPIC_MODEL: {:?}", std::env::var("ANTHROPIC_MODEL"));
    
    // Create simple request
    let conversation = Prompt::user("Hello! Please just say 'Hi there!' back to me.");
    let request = LLMRequest::from_prompt("claude-3-5-sonnet-v2@20241022", &conversation)
        .temperature(0.0)
        .max_tokens(50);
    
    println!("\n📤 Sending request...");
    
    // Generate response
    match provider.generate(&request).await {
        Ok(response) => {
            let mut stream = response.stream();
            
            println!("📥 Processing stream events:");
            let mut event_count = 0;
            
            while let Some(event_result) = stream.next().await {
                event_count += 1;
                match event_result {
                    Ok(event) => {
                        println!("  Event #{}: {:?}", event_count, event);
                    }
                    Err(e) => {
                        println!("  Error #{}: {:?}", event_count, e);
                        return Err(e);
                    }
                }
            }
            
            println!("🏁 Stream ended with {} total events", event_count);
            
            // Try to get text content from new response
            println!("\n📄 Getting text content from new request...");
            let response2 = provider.generate(&request).await?;
            let text = response2.text().await?;
            println!("📄 Final text content: '{}'", text);
            println!("📄 Text length: {} characters", text.len());
        }
        Err(e) => {
            println!("❌ Request failed: {}", e);
            return Err(e);
        }
    }
    
    Ok(())
}