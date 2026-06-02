//! End-to-end smoke test for the [`LlamaGgufProvider`].
//!
//! Loads a small cached GGUF model from `target/test-models/` (or
//! wherever `PLATFORMED_LLM_MODEL_CACHE` points), runs one short
//! generation, and asserts the model emits at least one token.
//!
//! ## Setup
//!
//! Models must be downloaded first via:
//!
//! ```text
//! cargo run --bin fetch-test-models --features fetch-test-models
//! ```
//!
//! Or supply a local file ad-hoc:
//!
//! ```text
//! PLATFORMED_LLM_TEST_MODEL_PATH=~/path/to/model.gguf \
//!   cargo test --features llama-gguf --test llama_gguf_e2e
//! ```
//!
//! The test reports `skip: <reason>` and exits cleanly when no model
//! is available — so this file is safe to leave enabled in CI runs
//! that haven't populated the cache.

#![cfg(feature = "llama-gguf")]

mod common;

use platformed_llm::providers::LlamaGgufProvider;
use platformed_llm::{Config, Prompt, Provider};

use common::test_models;

#[tokio::test]
async fn local_generation_smoke_test() {
    let model_path = match test_models::require(test_models::SMOLLM2_135M_INSTRUCT_Q8) {
        Ok(p) => p,
        Err(reason) => {
            eprintln!("skip: {reason}");
            return;
        }
    };
    eprintln!("using GGUF model at {}", model_path.display());

    let provider = match LlamaGgufProvider::from_gguf(model_path.to_string_lossy().into_owned()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skip: LlamaGgufProvider::from_gguf failed: {e}");
            return;
        }
    };

    let prompt = Prompt::user("Say hello in one word.");
    let cfg = Config::new("smollm2-135m-instruct").max_tokens(16).build();
    let response = provider
        .generate(&prompt, cfg.raw())
        .await
        .expect("generate succeeded");
    let text = response.text().await.expect("buffered text");
    eprintln!("model output: {text:?}");
    assert!(
        !text.trim().is_empty(),
        "local model should emit at least one token, got empty output",
    );
}
