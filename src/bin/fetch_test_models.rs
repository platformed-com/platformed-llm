//! Pre-download test GGUF models so the integration tests don't have
//! to (and don't fail when the runner is offline).
//!
//! Run before `cargo test`:
//!
//! ```text
//! cargo run --bin fetch-test-models --features fetch-test-models
//! ```
//!
//! Subsequent runs are a no-op when every model is already cached.
//! Set `PLATFORMED_LLM_MODEL_CACHE=/some/dir` to override the cache
//! location (defaults to `${CARGO_TARGET_DIR:-target}/test-models/`).
//!
//! The `fetch-test-models` feature is required because the bin pulls
//! in `reqwest/rustls-tls` and a multi-thread Tokio runtime — neither
//! of which the lib itself wants in its default dependency tree.
//!
//! The model list below is the *single source of truth* — tests look
//! up cached files by their `filename` via
//! [`tests/common/test_models.rs`](../../tests/common/test_models.rs),
//! so adding a new model means updating both this `MODELS` array and
//! the matching `pub const` in `tests/common/test_models.rs`.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use futures_util::StreamExt as _;

/// One downloadable test model.
struct TestModel {
    /// Filename under the cache directory. *Must* match the
    /// corresponding `pub const` in `tests/common/test_models.rs`.
    filename: &'static str,
    /// Where to download from on a cache miss. Public URLs only —
    /// HuggingFace `resolve/main/<file>` is the canonical pattern.
    url: &'static str,
}

const MODELS: &[TestModel] = &[
    // SmolLM2 135M Instruct, Q8 quantisation. ~150 MB. Used by
    // `tests/llama_gguf_e2e.rs`. Pulled from bartowski's mirror —
    // the original HuggingFaceTB upload was gated after release.
    TestModel {
        filename: "smollm2-135m-instruct-q8_0.gguf",
        url: "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf",
    },
];

#[tokio::main]
async fn main() -> ExitCode {
    let cache = cache_dir();
    if let Err(e) = std::fs::create_dir_all(&cache) {
        eprintln!("failed to create {}: {e}", cache.display());
        return ExitCode::FAILURE;
    }
    println!("cache directory: {}", cache.display());

    let mut had_error = false;
    for model in MODELS {
        let dest = cache.join(model.filename);
        if dest.exists() {
            println!(
                "cached:  {} ({})",
                model.filename,
                human_size(file_size(&dest))
            );
            continue;
        }
        print!("fetching {} ... ", model.filename);
        std::io::stdout().flush().ok();
        match download_to(model.url, &dest).await {
            Ok(bytes) => println!("ok ({} bytes)", bytes),
            Err(e) => {
                println!("FAILED");
                eprintln!("  url: {}", model.url);
                eprintln!("  err: {e}");
                had_error = true;
            }
        }
    }

    if had_error {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("PLATFORMED_LLM_MODEL_CACHE") {
        return PathBuf::from(dir);
    }
    let target = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
    PathBuf::from(target).join("test-models")
}

/// Stream-to-disk download. Writes to `<dest>.partial` first and
/// renames on success so a `Ctrl-C` mid-fetch doesn't leave a
/// corrupt cache entry for the next run.
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
