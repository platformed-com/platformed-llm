use platformed_llm::{Error, LLMRequest, Prompt, ProviderFactory, ResponseAccumulator};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Error> {
    dotenv::dotenv().ok();
    
    println!("🔍 Debug Streaming");
    
    // Check environment
    println!("Environment check:");
    println!("  OPENAI_API_KEY: {}", if std::env::var("OPENAI_API_KEY").is_ok() { "Set" } else { "Not set" });
    println!("  VERTEX_ACCESS_TOKEN: {}", if std::env::var("VERTEX_ACCESS_TOKEN").is_ok() { "Set" } else { "Not set" });
    println!("  PROVIDER_TYPE: {:?}", std::env::var("PROVIDER_TYPE"));
    
    // Create provider
    let provider = ProviderFactory::from_env().await?;
    println!("✅ Provider created");
    
    // Simple request
    let conversation = Prompt::user("Say 'Hello World' exactly.");
    let request = LLMRequest::from_prompt("gpt-4o-mini", &conversation)
        .temperature(0.0)
        .max_tokens(10);
    
    println!("📡 Making request...");
    
    // Generate response
    match provider.generate(&request).await {
        Ok(response) => {
            println!("✅ Request succeeded");
            
            // Test direct text method
            let response_clone = provider.generate(&request).await?;
            let text = response_clone.text().await?;
            println!("📄 Direct text result: '{}'", text);
            println!("📄 Text length: {}", text.len());
            
            // Test streaming
            let mut stream = response.stream();
            let mut accumulator = ResponseAccumulator::new();
            let mut event_count = 0;
            
            println!("📥 Stream events:");
            while let Some(event_result) = stream.next().await {
                event_count += 1;
                match event_result {
                    Ok(event) => {
                        println!("  #{}: {:?}", event_count, event);
                        accumulator.process_event(event)?;
                    }
                    Err(e) => {
                        println!("  Error #{}: {}", event_count, e);
                        return Err(e);
                    }
                }
            }
            
            println!("🏁 {} total events", event_count);
            
            let complete_response = accumulator.finalize()?;
            let content = complete_response.content();
            println!("📄 Accumulated content: '{}'", content);
        }
        Err(e) => {
            println!("❌ Request failed: {}", e);
            return Err(e);
        }
    }
    
    Ok(())
}