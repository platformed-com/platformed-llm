use futures_util::StreamExt;
use platformed_llm::providers::vertex::AnthropicViaVertexProvider;
use platformed_llm::{Error, LLMProvider, LLMRequest, Prompt, ResponseAccumulator};

#[tokio::main]
async fn main() -> Result<(), Error> {
    dotenv::dotenv().ok();

    println!("🔍 Simple Anthropic Test");

    // Check environment
    let project_id =
        std::env::var("GOOGLE_CLOUD_PROJECT").expect("GOOGLE_CLOUD_PROJECT must be set");
    let location =
        std::env::var("GOOGLE_CLOUD_REGION").unwrap_or_else(|_| "europe-west1".to_string());
    let access_token =
        std::env::var("VERTEX_ACCESS_TOKEN").expect("VERTEX_ACCESS_TOKEN must be set");

    println!("📋 Configuration:");
    println!("  Project: {project_id}");
    println!("  Location: {location}");
    println!(
        "  Token: {}...",
        &access_token[..20.min(access_token.len())]
    );

    // Create provider directly
    let provider = AnthropicViaVertexProvider::new(project_id, location, access_token)?;

    println!("✅ Anthropic provider created");

    // Simple request
    let conversation = Prompt::user("Please just say 'Hello world!' - nothing else.");
    let request = LLMRequest::from_prompt("claude-3-5-sonnet-v2@20241022", &conversation)
        .temperature(0.0)
        .max_tokens(20);

    println!("📤 Making simple request...");

    // Generate response
    let response = provider.generate(&request).await?;
    let mut stream = response.stream();

    println!("📥 Processing stream:");
    let mut accumulator = ResponseAccumulator::new();
    let mut event_count = 0;

    while let Some(event_result) = stream.next().await {
        event_count += 1;
        match event_result {
            Ok(event) => {
                println!("  Event #{event_count}: {event:?}");
                accumulator.process_event(event)?;
            }
            Err(e) => {
                println!("  Error #{event_count}: {e}");
                return Err(e);
            }
        }
    }

    println!("🏁 Processed {event_count} events");

    let complete_response = accumulator.finalize()?;
    let content = complete_response.content();

    println!("📄 Response content: '{content}'");
    println!("📄 Content length: {} characters", content.len());

    if content.is_empty() {
        println!("❌ Empty response detected!");
    } else {
        println!("✅ Got response content");
    }

    Ok(())
}
