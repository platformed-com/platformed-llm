#![cfg(feature = "llama-gguf")]
//! Tests for the `LlamaGgufProvider` spawn_blocking ↔ channel bridge
//! and finish-reason inference, via a mock `LocalEngine` (no real
//! model). Covers the integration glue that the chat-template
//! generator unit tests don't: engine-error propagation through the
//! whole pipeline, and Length-vs-Stop inference from the token
//! budget.

use std::sync::Arc;

use platformed_llm::providers::local::LocalEngine;
use platformed_llm::providers::LlamaGgufProvider;
use platformed_llm::{generate, Config, FinishReason, Prompt};

/// Mock engine yielding a fixed script of token results.
struct ScriptEngine(Vec<Result<String, String>>);

impl LocalEngine for ScriptEngine {
    fn generate_streaming<'a>(
        &'a self,
        _prompt: &str,
        _max_tokens: usize,
    ) -> Box<dyn Iterator<Item = Result<String, String>> + Send + 'a> {
        Box::new(self.0.clone().into_iter())
    }
}

fn provider(script: Vec<Result<String, String>>) -> LlamaGgufProvider {
    LlamaGgufProvider::from_local_engine(Arc::new(ScriptEngine(script)))
}

#[tokio::test]
async fn budget_exhausted_reports_length() {
    // 3 chunks emitted, max_tokens 2 → budget exhausted → Length.
    let p = provider(vec![Ok("a".into()), Ok("b".into()), Ok("c".into())]);
    let cfg = Config::builder("m").max_tokens(2).build();
    let resp = generate(&p, &Prompt::user("hi"), &cfg).await.unwrap();
    let complete = resp.buffer().await.unwrap();
    assert_eq!(complete.finish_reason, FinishReason::Length);
}

#[tokio::test]
async fn natural_stop_reports_stop() {
    let p = provider(vec![Ok("hello".into())]);
    let cfg = Config::builder("m").max_tokens(100).build();
    let resp = generate(&p, &Prompt::user("hi"), &cfg).await.unwrap();
    let complete = resp.buffer().await.unwrap();
    assert_eq!(complete.finish_reason, FinishReason::Stop);
}

#[tokio::test]
async fn engine_error_propagates_and_suppresses_done() {
    let p = provider(vec![Ok("partial".into()), Err("boom".into())]);
    let cfg = Config::builder("m").max_tokens(100).build();
    let resp = generate(&p, &Prompt::user("hi"), &cfg).await.unwrap();
    // The mid-stream engine error must surface as a stream error
    // (collect returns Err), not a silent clean finish.
    let err = resp
        .collect()
        .await
        .map(|_| ())
        .expect_err("engine error must surface");
    assert!(
        err.to_string().contains("boom"),
        "error should carry the engine message, got: {err}"
    );
}
