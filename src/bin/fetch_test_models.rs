//! Pre-download every test GGUF model so the integration tests don't
//! have to (and don't fail when the runner is offline).
//!
//! Run before `cargo test`:
//!
//! ```text
//! cargo run --bin fetch-test-models --features test-util
//! ```
//!
//! Subsequent runs are a no-op when every model is already cached. Set
//! `PLATFORMED_LLM_MODEL_CACHE=/some/dir` to override the cache location
//! (defaults to `${CARGO_TARGET_DIR:-target}/test-models/`).
//!
//! This is a thin wrapper over the same [`test_util::ensure`] entry
//! point downstream crates use: the model registry, download logic, and
//! progress reporting all live in [`platformed_llm::test_util`], so
//! there's a single source of truth for which models exist and where
//! they come from.

use std::process::ExitCode;

use platformed_llm::test_util;

#[tokio::main]
async fn main() -> ExitCode {
    let mut had_error = false;
    for &model in test_util::MODELS {
        // `ensure` reports cache hits / download progress to stderr; we
        // only need to surface failures and set the exit code.
        if let Err(e) = test_util::ensure(model).await {
            eprintln!("FAILED {}: {e}", model.filename);
            had_error = true;
        }
    }

    if had_error {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
