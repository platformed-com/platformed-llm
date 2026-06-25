//! Demonstrates the `mock` provider: testing code that depends on
//! `Provider` without any network or credentials.
//!
//! Doubles as a worked example of:
//!
//! - The [`retry`] helper: each agent turn in Part 1 runs inside
//!   `retry(&policy, …)`, so a scripted [`Error::RateLimit`] from the
//!   mock transparently triggers a sleep + retry — same shape a real
//!   provider's 429 would take.
//! - The shared [`InMemoryRateLimiter`]: Part 2 has two tenants
//!   competing for a single shared mock "provider"; strict priority
//!   lets a late high-priority `Interactive` request from one tenant
//!   preempt a queued backlog of `Background` work from another.
//!
//! Run with: `cargo run --example mock_provider --features mock`

use std::sync::Arc;
use std::time::Instant;

use platformed_llm::providers::mock::{Chunking, MockProvider, MockResponse};
use platformed_llm::{
    generate, retry, Config, Error, FunctionCall, InMemoryRateLimiter, Priority, Prompt, Provider,
    RetryPolicy, SharedRateLimiter,
};

/// A toy "agent loop" — exactly the kind of code you'd want to test
/// against a mock. It asks the model, runs any tool call, feeds the
/// result back, and returns the final text. The retry policy is
/// applied **outside** at the caller's discretion (see Part 1) — that
/// keeps the spawned-task pattern in Part 2 simple (no higher-ranked
/// lifetime headaches from async closures crossing `tokio::spawn`).
async fn run_agent(
    provider: &dyn Provider,
    tenant: &str,
    priority: Priority,
    question: &str,
) -> Result<String, Error> {
    let config = Config::builder("test-model")
        .tenant(tenant)
        .priority(priority)
        .build();
    let mut prompt = Prompt::user(question);

    loop {
        let response = generate(provider, &prompt, &config).await?.buffer().await?;
        match response.function_calls().first() {
            Some(call) => {
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
    // === Part 1: retry demo. ===
    //
    // Scripted queue: the first scripted reply is a synthetic 429.
    // We wrap the agent call in `retry(...)`, so the 429 is
    // transparently swallowed and the agent restarts; the second
    // attempt sees a tool call, then the final answer. The
    // *provider* is called 3 times overall (429 + tool call +
    // answer), but the *agent loop* runs to completion once.
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
    let policy = RetryPolicy::standard();
    let answer = retry(&policy, async |_attempt| {
        run_agent(
            &provider,
            "weather-service",
            Priority::Interactive,
            "What's the weather in Paris?",
        )
        .await
    })
    .await?;

    println!("answer: {answer}");
    println!("the agent made {} provider call(s)\n", log.len());
    assert_eq!(answer, "It is sunny in Paris.");
    // 3 calls: synthetic 429 → retry of the agent loop → tool call →
    // final answer. The retry restarts the agent from the user
    // message, which is fine because the agent loop is idempotent.
    assert_eq!(log.len(), 3);

    // === Part 2: two tenants share a rate-limited mock provider. ===
    //
    // The limiter starts at 1 req/s by default — slow enough to make
    // queueing visible. Tenant `loud` floods with 4 Background
    // requests; tenant `vip` fires a single Interactive request a
    // moment later. Strict priority means `vip` jumps the queue.
    let limiter: SharedRateLimiter = Arc::new(InMemoryRateLimiter::new());
    let provider = Arc::new(MockProvider::with_text("ok").with_rate_limiter(limiter.clone()));

    let start = Instant::now();
    let mut tasks = Vec::new();

    // Loud tenant: 4 background requests, kicked off first.
    for i in 0..4 {
        let p = provider.clone();
        tasks.push(tokio::spawn(async move {
            run_agent(
                &*p,
                "loud",
                Priority::Background,
                &format!("loud request {i}"),
            )
            .await
            .map(|_| ("loud", Instant::now()))
        }));
    }
    // Give them a moment to enqueue.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    // VIP tenant: one interactive request, fired late.
    let p = provider.clone();
    tasks.push(tokio::spawn(async move {
        run_agent(&*p, "vip", Priority::Interactive, "vip request")
            .await
            .map(|_| ("vip", Instant::now()))
    }));

    let mut results = Vec::new();
    for task in tasks {
        results.push(task.await.unwrap()?);
    }
    results.sort_by_key(|(_, t)| *t);

    println!("dispatch order under the shared limiter:");
    for (tenant, time) in &results {
        println!(
            "  +{:>4}ms  {tenant}",
            time.duration_since(start).as_millis(),
        );
    }
    // Strict priority: vip should land before all but the very first
    // loud request (which had a head-start of ~50ms before the limiter
    // was queueing anything).
    let vip_position = results
        .iter()
        .position(|(t, _)| *t == "vip")
        .expect("vip dispatched");
    assert!(
        vip_position <= 1,
        "strict-priority preemption: vip should dispatch within the first \
         two slots, got position {vip_position}",
    );

    Ok(())
}
