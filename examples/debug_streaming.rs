use futures_util::StreamExt;
use platformed_llm::accumulator::ResponseAccumulator;
use platformed_llm::{generate, retry, Config, Error, Prompt, ProviderFactory, RetryPolicy};

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
    let cfg = Config::builder("gpt-4o-mini")
        .temperature(0.0)
        .max_tokens(10)
        .build();

    println!("📡 Making request...");

    // Wrap the entire generate-and-stream-consumption in `retry` so
    // a transient 429 / 5xx restarts cleanly. The closure is the
    // unit of retry: any state it owns (the event counter, the
    // accumulator) is discarded on each attempt, so a partial
    // stream from a previous attempt can't leak into the accumulated
    // text. `retry` honours `Retry-After` via the policy and gives
    // up on terminal errors (auth, bad config, …).
    //
    // Caveat: mid-body connection drops surface as
    // `Error::Transport(is_body())`, which the default policy
    // treats as non-retryable (it can't distinguish a real drop from
    // a deterministic decode failure). See the `What doesn't` section
    // of `mod retry` — for production code that *knows* its stream
    // is safe to re-issue on any body-level error, drive
    // `RetryPolicy::delay_after` yourself and treat
    // `Error::Transport` as retryable in your own classifier.
    let policy = RetryPolicy::standard();
    let text = retry(policy, async |attempt| {
        if attempt > 1 {
            println!("🔁 retry attempt {attempt}");
        }
        let response = generate(&*provider, &conversation, &cfg).await?;
        let mut stream = response.stream();
        let mut accumulator = ResponseAccumulator::new();
        let mut event_count = 0;
        println!("📥 Stream events:");
        while let Some(event) = stream.next().await {
            event_count += 1;
            let event = event?;
            println!("  #{event_count}: {event:?}");
            accumulator.process_event(event)?;
        }
        println!("🏁 {event_count} total events");
        Ok(accumulator.finalize()?.text())
    })
    .await?;

    println!("📄 Accumulated content: '{text}'");
    println!("📄 Text length: {}", text.len());

    Ok(())
}
