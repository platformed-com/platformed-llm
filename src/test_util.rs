//! Test-only helpers for locating — and, on a cache miss, downloading —
//! the small GGUF models the integration suite runs against.
//!
//! This module is gated behind the `test-util` feature. Rust has no
//! cross-crate "test mode" (`#[cfg(test)]` only applies to a crate's
//! own tests), so downstream crates that want to exercise the local
//! [`LlamaGgufProvider`](crate::providers::LlamaGgufProvider) against
//! the *same* model this crate's tests use enable the feature under
//! their `[dev-dependencies]`:
//!
//! ```toml
//! [dev-dependencies]
//! platformed-llm = { version = "...", features = ["llama-gguf", "test-util"] }
//! ```
//!
//! …and then resolve a model in a test:
//!
//! ```no_run
//! # async fn doc() {
//! use platformed_llm::test_util;
//!
//! // Downloads to the shared cache on first run, no-op thereafter.
//! let path = test_util::ensure(test_util::SMOLLM2_135M_INSTRUCT_Q8)
//!     .await
//!     .expect("fetch test model");
//! # }
//! ```
//!
//! The `test-util` feature pulls in `reqwest/rustls-tls` plus a
//! multi-thread Tokio runtime so [`ensure`] is self-contained — the
//! lib itself never picks a TLS backend, so this stays opt-in.
//!
//! ## Environment overrides
//!
//! - `PLATFORMED_LLM_MODEL_CACHE` — directory models are cached in.
//!   Defaults to `${CARGO_TARGET_DIR:-target}/test-models/`.
//! - `PLATFORMED_LLM_TEST_MODEL_PATH` — short-circuits [`ensure`]: when
//!   set, it's used verbatim for *any* model (handy for running one
//!   test against a different file than the suite would fetch).

use std::io::Write as _;
use std::path::{Path, PathBuf};

use futures_util::StreamExt as _;

/// One downloadable test model: a cached `filename` and the public URL
/// to fetch it from on a cache miss.
///
/// [`MODELS`] is the single source of truth for the set the suite
/// tracks; the `pub const` values below name individual entries so
/// tests get typo-safe references.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TestModel {
    /// Filename under the cache directory.
    pub filename: &'static str,
    /// Where to download from on a cache miss. Public URLs only —
    /// HuggingFace `resolve/main/<file>` is the canonical pattern.
    pub url: &'static str,
}

/// SmolLM2 135M Instruct (Q8 quant, ~150 MB). The smallest fully-
/// functional GGUF chat model the test suite tracks — fast to
/// download, fast to load, runs on CPU in seconds.
///
/// Pulled from bartowski's mirror — the original HuggingFaceTB upload
/// was gated after release.
pub const SMOLLM2_135M_INSTRUCT_Q8: TestModel = TestModel {
    filename: "smollm2-135m-instruct-q8_0.gguf",
    url: "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf",
};

/// Every test model the suite knows how to fetch. The `fetch-test-models`
/// binary pre-downloads this whole set; tests resolve individual
/// entries with [`ensure`].
pub const MODELS: &[TestModel] = &[SMOLLM2_135M_INSTRUCT_Q8];

/// Resolve a model, downloading it into the shared cache on a miss and
/// returning the cached path. A no-op (just the path) when already
/// cached. Progress — cache hit, download start, bytes fetched — is
/// reported to stderr so test runs and the `fetch-test-models` binary
/// narrate what they're doing.
///
/// A download failure is a hard error — the returned `Err` is meant to
/// be `.expect`ed so the test fails. `PLATFORMED_LLM_TEST_MODEL_PATH`
/// short-circuits to a verbatim path. Concurrent callers are safe: the
/// download writes to a `.partial` file and atomically renames on
/// success, so a half-written file is never observed as cached.
pub async fn ensure(model: TestModel) -> Result<PathBuf, String> {
    if let Some(p) = override_path()? {
        eprintln!(
            "test model: using PLATFORMED_LLM_TEST_MODEL_PATH={}",
            p.display()
        );
        return Ok(p);
    }
    let cache = cache_dir();
    let dest = cache.join(model.filename);
    if dest.exists() {
        eprintln!(
            "test model {} cached ({})",
            model.filename,
            human_size(file_size(&dest))
        );
        return Ok(dest);
    }
    std::fs::create_dir_all(&cache).map_err(|e| format!("create {}: {e}", cache.display()))?;
    eprintln!(
        "test model {} fetching from {} ...",
        model.filename, model.url
    );
    let bytes = download_to(model.url, &dest).await?;
    eprintln!(
        "test model {} fetched ({})",
        model.filename,
        human_size(bytes)
    );
    Ok(dest)
}

/// Directory test models are cached in. Honors
/// `PLATFORMED_LLM_MODEL_CACHE`, then `CARGO_TARGET_DIR/test-models`,
/// then `target/test-models`.
fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("PLATFORMED_LLM_MODEL_CACHE") {
        return PathBuf::from(dir);
    }
    let target = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
    PathBuf::from(target).join("test-models")
}

/// Resolve `PLATFORMED_LLM_TEST_MODEL_PATH` if set, erroring when it
/// points at a nonexistent file. `Ok(None)` means the override is
/// unset and normal cache resolution should proceed.
fn override_path() -> Result<Option<PathBuf>, String> {
    match std::env::var("PLATFORMED_LLM_TEST_MODEL_PATH") {
        Ok(override_path) => {
            let p = PathBuf::from(override_path);
            if !p.exists() {
                return Err(format!(
                    "PLATFORMED_LLM_TEST_MODEL_PATH={} does not exist",
                    p.display()
                ));
            }
            Ok(Some(p))
        }
        Err(_) => Ok(None),
    }
}

/// Stream-to-disk download. Writes to `<dest>.partial` first and
/// renames on success so a `Ctrl-C` mid-fetch (or a concurrent reader)
/// never observes a corrupt cache entry. Returns the byte count.
async fn download_to(url: &str, dest: &Path) -> Result<u64, String> {
    let partial = dest.with_extension(format!(
        "{}.partial",
        dest.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("download")
    ));

    let response = reqwest::get(url)
        .await
        .map_err(|e| format!("HTTP request failed: {e}"))?;
    if !response.status().is_success() {
        return Err(format!("HTTP {} from {url}", response.status()));
    }

    let mut file = std::fs::File::create(&partial)
        .map_err(|e| format!("create {}: {e}", partial.display()))?;
    let mut stream = response.bytes_stream();
    let mut written: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("read chunk: {e}"))?;
        file.write_all(&chunk)
            .map_err(|e| format!("write {}: {e}", partial.display()))?;
        written += chunk.len() as u64;
    }
    file.sync_all().ok();
    drop(file);

    std::fs::rename(&partial, dest)
        .map_err(|e| format!("rename {} -> {}: {e}", partial.display(), dest.display()))?;
    Ok(written)
}

fn file_size(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

fn human_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}
