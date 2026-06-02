use futures_util::StreamExt;
use platformed_llm::accumulator::ResponseAccumulator;
use platformed_llm::{Config, Error, Prompt, ProviderFactory};

#[tokio::main]
async fn main() -> Result<(), Error> {
    dotenvy::dotenv().ok();

    println!("🔍 Debug Streaming");

    // Check environment
    println!("Environment check:");
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
    println!("  PROVIDER_TYPE: {:?}", std::env::var("PROVIDER_TYPE"));

    // Create provider
    let provider = ProviderFactory::from_env().await?;
    println!("✅ Provider created");

    // Simple request
    let conversation = Prompt::user("Say 'Hello World' exactly.");
    let cfg = Config::new("gpt-4o-mini")
        .temperature(0.0)
        .max_tokens(10)
        .build();

    println!("📡 Making request...");

    // Generate response
    match provider.generate(&conversation, cfg.raw()).await {
        Ok(response) => {
            println!("✅ Request succeeded");

            // Test direct text method
            let response_clone = provider.generate(&conversation, cfg.raw()).await?;
            let text = response_clone.text().await?;
            println!("📄 Direct text result: '{text}'");
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
                        println!("  #{event_count}: {event:?}");
                        accumulator.process_event(event)?;
                    }
                    Err(e) => {
                        println!("  Error #{event_count}: {e}");
                        return Err(e);
                    }
                }
            }

            println!("🏁 {event_count} total events");

            let complete_response = accumulator.finalize()?;
            let text = complete_response.text();
            println!("📄 Accumulated content: '{text}'");
        }
        Err(e) => {
            println!("❌ Request failed: {e}");
            return Err(e);
        }
    }

    Ok(())
}
