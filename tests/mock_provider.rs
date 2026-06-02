//! Integration test exercising the public `mock` provider the way a
//! downstream crate would: drive a tool-call loop against canned
//! responses, then assert on what was sent.
#![cfg(feature = "mock")]

use platformed_llm::providers::mock::{Chunking, MockProvider, MockResponse};
use platformed_llm::{Config, FunctionCall, InputItem, Prompt, Provider, UserPart};

#[tokio::test]
async fn drives_a_tool_call_loop() {
    let provider = MockProvider::builder()
        .chunking(Chunking::Words)
        .reply(MockResponse::tool_call(FunctionCall {
            call_id: "call_1".into(),
            name: "lookup".into(),
            arguments: r#"{"q":"answer"}"#.into(),
        }))
        .reply("The answer is 42.")
        .build();

    let log = provider.call_log();
    let config = Config::new("test-model");

    // Turn 1: expect a tool call.
    let mut prompt = Prompt::user("what is the answer?");
    let first = provider
        .generate(&prompt, &config)
        .await
        .unwrap()
        .buffer()
        .await
        .unwrap();
    let call = first.function_calls()[0].clone();
    assert_eq!(call.name, "lookup");

    // Turn 2: feed the tool result back, expect final text.
    prompt = prompt
        .with_response(&first)
        .with_tool_result(&call.call_id, "42");
    let final_text = provider
        .generate(&prompt, &config)
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(final_text, "The answer is 42.");

    // The mock recorded both calls; the second carried the tool result.
    let calls = log.calls();
    assert_eq!(calls.len(), 2);
    let second_has_tool_result = calls[1].prompt.items().iter().any(|item| {
        matches!(item, InputItem::User { content }
            if content.iter().any(|p| matches!(p, UserPart::ToolResult { .. })))
    });
    assert!(
        second_has_tool_result,
        "second call should carry the tool result"
    );
}

#[tokio::test]
async fn surfaces_scripted_errors() {
    let provider = MockProvider::builder()
        .fail(platformed_llm::Error::provider_with_status(
            "MockProvider",
            429,
            "rate limited",
        ))
        .build();

    let err = provider
        .generate(&Prompt::user("x"), &Config::new("m"))
        .await
        .map(|_| ())
        .expect_err("scripted failure");
    assert!(err.to_string().contains("rate limited"));
}
