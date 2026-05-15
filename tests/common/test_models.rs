//! Lookup helpers for test-only model files cached on disk.
//!
//! Models are populated by the `fetch-test-models` binary:
//!
//! ```text
//! cargo run --bin fetch-test-models --features fetch-test-models
//! ```
//!
//! …which downloads each known model into the cache directory. Tests
//! then call [`path`] / [`require`] to locate a model file, falling
//! back to a clean skip when one isn't present.
//!
//! The single source of truth for which models exist and where they
//! come from lives in [`src/bin/fetch_test_models.rs`](../../src/bin/fetch_test_models.rs);
//! the constants below name the cached filenames so tests get
//! typo-safe references.

use std::path::PathBuf;

/// SmolLM2 135M Instruct (Q8 quant, ~150 MB). The smallest fully-
/// functional GGUF chat model the test suite tracks — fast to
/// download, fast to load, runs on CPU in seconds.
pub const SMOLLM2_135M_INSTRUCT_Q8: &str = "smollm2-135m-instruct-q8_0.gguf";

/// Directory test models are cached in. Honors
/// `PLATFORMED_LLM_MODEL_CACHE`, then `CARGO_TARGET_DIR/test-models`,
/// then `target/test-models`.
pub fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("PLATFORMED_LLM_MODEL_CACHE") {
        return PathBuf::from(dir);
    }
    let target = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
    PathBuf::from(target).join("test-models")
}

/// Path to a cached model file, or `None` if it hasn't been
/// downloaded yet. Tests should typically use [`require`] which emits
/// a useful skip message.
pub fn path(filename: &str) -> Option<PathBuf> {
    let p = cache_dir().join(filename);
    p.exists().then_some(p)
}

/// Resolve a model the test needs, or return a printable skip
/// reason. Callers print the reason and return early — that's the
/// project convention for tests that depend on out-of-band resources.
///
/// `PLATFORMED_LLM_TEST_MODEL_PATH` short-circuits the lookup: when
/// set, it's used verbatim regardless of `filename`. Useful when
/// running a single test against a different model than the suite
/// would normally fetch.
pub fn require(filename: &str) -> Result<PathBuf, String> {
    if std::env::var("PLATFORMED_LLM_SKIP_GGUF_TESTS").is_ok() {
        return Err("PLATFORMED_LLM_SKIP_GGUF_TESTS is set".to_string());
    }
    if let Ok(override_path) = std::env::var("PLATFORMED_LLM_TEST_MODEL_PATH") {
        let p = PathBuf::from(override_path);
        if !p.exists() {
            return Err(format!(
                "PLATFORMED_LLM_TEST_MODEL_PATH={} does not exist",
                p.display()
            ));
        }
        return Ok(p);
    }
    match path(filename) {
        Some(p) => Ok(p),
        None => Err(format!(
            "test model {filename:?} not in {} — run \
             `cargo run --bin fetch-test-models --features fetch-test-models` \
             to populate the cache",
            cache_dir().display()
        )),
    }
}
