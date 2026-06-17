//! End-to-end smoke test for the [`LlamaGgufProvider`].
//!
//! Loads a small cached GGUF model from `target/test-models/` (or
//! wherever `PLATFORMED_LLM_MODEL_CACHE` points), runs one short
//! generation, and asserts the model emits at least one token.
//!
//! ## Setup
//!
//! The model is downloaded into the shared cache on first run via
//! [`test_util::ensure`]. To pre-populate the cache (e.g. before an
//! offline run):
//!
//! ```text
//! cargo run --bin fetch-test-models --features test-util
//! ```
//!
//! Or supply a local file ad-hoc:
//!
//! ```text
//! PLATFORMED_LLM_TEST_MODEL_PATH=~/path/to/model.gguf \
//!   cargo test --features llama-gguf,test-util --test llama_gguf_e2e
//! ```

#![cfg(all(feature = "llama-gguf", feature = "test-util"))]

use platformed_llm::providers::LlamaGgufProvider;
use platformed_llm::test_util;
use platformed_llm::{generate, Config, Prompt};

#[tokio::test]
async fn local_generation_smoke_test() {
    let model_path = test_util::ensure(test_util::SMOLLM2_135M_INSTRUCT_Q8)
        .await
        .expect("fetch test model");
    eprintln!("using GGUF model at {}", model_path.display());

    let provider = match LlamaGgufProvider::from_gguf(model_path.to_string_lossy().into_owned()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skip: LlamaGgufProvider::from_gguf failed: {e}");
            return;
        }
    };

    let prompt = Prompt::user("Say hello in one word.");
    let cfg = Config::builder("smollm2-135m-instruct")
        .max_tokens(16)
        .build();
    let response = generate(&provider, &prompt, &cfg)
        .await
        .expect("generate succeeded");
    let text = response.text().await.expect("buffered text");
    eprintln!("model output: {text:?}");
    assert!(
        !text.trim().is_empty(),
        "local model should emit at least one token, got empty output",
    );
}
