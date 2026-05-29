//! Demonstrates the `mock` provider: testing code that depends on
//! `Provider` without any network or credentials.
//!
//! Run with: `cargo run --example mock_provider --features mock`

use platformed_llm::providers::mock::{Chunking, MockProvider, MockResponse};
use platformed_llm::{Config, FunctionCall, Prompt, Provider};

/// A toy "agent loop" — exactly the kind of code you'd want to test
/// against a mock. It asks the model, runs any tool call, feeds the
/// result back, and returns the final text.
async fn run_agent(
    provider: &dyn Provider,
    question: &str,
) -> Result<String, platformed_llm::Error> {
    let config = Config::new("test-model");
    let mut prompt = Prompt::user(question);

    loop {
        let response = provider.generate(&prompt, &config).await?.buffer().await?;
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
async fn main() -> Result<(), platformed_llm::Error> {
    // A scripted queue: first turn calls a tool, second turn answers.
    let provider = MockProvider::builder()
        .chunking(Chunking::Words)
        .reply(MockResponse::tool_call(FunctionCall {
            call_id: "call_1".into(),
            name: "get_weather".into(),
            arguments: r#"{"city":"Paris"}"#.into(),
        }))
        .reply("It is sunny in Paris.")
        .build();

    let log = provider.call_log();
    let answer = run_agent(&provider, "What's the weather in Paris?").await?;

    println!("answer: {answer}");
    println!("the agent made {} provider call(s)", log.len());
    assert_eq!(answer, "It is sunny in Paris.");
    assert_eq!(log.len(), 2);

    Ok(())
}
