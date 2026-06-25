//! Demonstrates the `mock` provider: testing code that depends on
//! `Provider` without any network or credentials.
//!
//! Also doubles as the buffered example of the [`retry`] helper:
//! each agent turn is wrapped in a retry loop with a standard
//! exponential-backoff policy, so a scripted [`Error::RateLimit`]
//! from the mock transparently triggers a sleep + retry — same shape
//! a real provider's 429 would take.
//!
//! Run with: `cargo run --example mock_provider --features mock`

use platformed_llm::providers::mock::{Chunking, MockProvider, MockResponse};
use platformed_llm::{generate, retry, Config, Error, FunctionCall, Prompt, Provider, RetryPolicy};

/// A toy "agent loop" — exactly the kind of code you'd want to test
/// against a mock. Each turn is wrapped in [`retry`]: a transient
/// failure (rate limit, 5xx, transport blip, mid-stream drop)
/// transparently re-issues *that turn only*, without re-running
/// already-committed turns earlier in the conversation. Terminal
/// errors propagate immediately.
async fn run_agent(provider: &dyn Provider, question: &str) -> Result<String, Error> {
    let config = Config::builder("test-model").build();
    let policy = RetryPolicy::standard();
    let mut prompt = Prompt::user(question);

    loop {
        let response = retry(&policy, async |_attempt| {
            generate(provider, &prompt, &config).await?.buffer().await
        })
        .await?;
        match response.function_calls().first() {
            Some(call) => {
                // Pretend we executed the tool.
                let result = format!("result for {}", call.name);
                prompt = prompt
                    .with_response(&response)
                    .with_tool_result(&call.call_id, result);
            }
            None => return Ok(response.text()),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    // A scripted queue: turn 1 hits a (synthetic) rate limit the retry
    // loop will swallow; turn 1 retry returns a tool call; turn 2
    // answers. The provider is called three times overall, but the
    // agent loop sees two semantically-distinct turns.
    let provider = MockProvider::builder()
        .chunking(Chunking::Words)
        .fail(Error::rate_limit(Some(0), "synthetic 429"))
        .reply(MockResponse::tool_call(FunctionCall {
            call_id: "call_1".into(),
            name: "get_weather".into(),
            arguments: r#"{"city":"Paris"}"#.into(),
            provider_signature: None,
        }))
        .reply("It is sunny in Paris.")
        .build();

    let log = provider.call_log();
    let answer = run_agent(&provider, "What's the weather in Paris?").await?;

    println!("answer: {answer}");
    println!("the agent made {} provider call(s)", log.len());
    assert_eq!(answer, "It is sunny in Paris.");
    // 3 calls: synthetic 429 → retry → tool call → final answer.
    assert_eq!(log.len(), 3);

    Ok(())
}
