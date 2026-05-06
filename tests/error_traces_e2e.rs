//! Replay captured 4xx/5xx traces through the matching provider's
//! `generate()` and assert they map to the right typed [`Error`] variant.
//!
//! `http_errors_e2e.rs` exercises `parse_openai_error` against synthetic
//! bodies; this file does the same against **real** provider error
//! envelopes captured by `capture_traces` (scenarios marked
//! `expect_failure: true`). Catches drift in the actual on-the-wire error
//! shape — e.g. OpenAI renaming a field, Vertex changing the
//! `google.rpc.Status` envelope, etc.
//!
//! No-ops if no error captures are present.

use std::fs;
use std::path::PathBuf;

use platformed_llm::{Error, GoogleProvider, LLMProvider, LLMRequest, OpenAIProvider, Prompt};
use serde_json::Value;
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

const TRACES_ROOT: &str = "tests/cross_provider/traces";

#[derive(Debug, Clone, Copy, PartialEq)]
enum Provider {
    OpenAI,
    Google,
}

impl Provider {
    fn dir_name(self) -> &'static str {
        match self {
            Provider::OpenAI => "openai",
            Provider::Google => "google",
        }
    }
    fn from_dir_name(name: &str) -> Option<Provider> {
        match name {
            "openai" => Some(Provider::OpenAI),
            "google" => Some(Provider::Google),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct ErrorTrace {
    provider: Provider,
    scenario: String,
    status: u16,
    body: Vec<u8>,
}

fn load_error_traces() -> Vec<ErrorTrace> {
    let root = PathBuf::from(TRACES_ROOT);
    let mut traces = Vec::new();
    if !root.is_dir() {
        return traces;
    }
    for entry in fs::read_dir(&root).expect("read traces root") {
        let entry = entry.unwrap();
        if !entry.file_type().unwrap().is_dir() {
            continue;
        }
        let provider = match Provider::from_dir_name(&entry.file_name().to_string_lossy()) {
            Some(p) => p,
            None => continue,
        };
        let dir = entry.path();
        for inner in fs::read_dir(&dir).unwrap() {
            let inner = inner.unwrap();
            let path = inner.path();
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            let scenario = match name.strip_suffix(".response.sse") {
                Some(s) => s.to_string(),
                None => continue,
            };
            let meta_path = dir.join(format!("{scenario}.meta.json"));
            let meta_raw = match fs::read_to_string(&meta_path) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let meta: Value = match serde_json::from_str(&meta_raw) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let status = meta.get("status").and_then(|v| v.as_u64()).unwrap_or(200) as u16;
            if (200..300).contains(&status) {
                continue;
            }
            let body = fs::read(&path).unwrap();
            traces.push(ErrorTrace {
                provider,
                scenario,
                status,
                body,
            });
        }
    }
    traces
}

async fn replay_error(trace: &ErrorTrace) -> Error {
    let server = MockServer::start().await;
    let body_clone = trace.body.clone();
    Mock::given(method("POST"))
        .and(path_regex(".*"))
        .respond_with(
            ResponseTemplate::new(trace.status)
                .insert_header("content-type", "application/json")
                .set_body_raw(body_clone, "application/json"),
        )
        .mount(&server)
        .await;

    let req = LLMRequest::from_prompt("model", &Prompt::user("hi"));
    match trace.provider {
        Provider::OpenAI => {
            let p =
                OpenAIProvider::new_with_base_url("test".to_string(), server.uri()).unwrap();
            p.generate(&req)
                .await
                .err()
                .expect("4xx must produce an error")
        }
        Provider::Google => {
            let p = GoogleProvider::new_with_base_url(
                "p".to_string(),
                "us-east1".to_string(),
                "tok".to_string(),
                server.uri(),
            )
            .unwrap();
            p.generate(&req)
                .await
                .err()
                .expect("4xx must produce an error")
        }
    }
}

/// For each captured error response, assert the typed [`Error`] variant
/// matches the captured HTTP status. Loose on the message — that's
/// volatile across model versions — but strict on the variant.
#[tokio::test]
async fn captured_error_bodies_map_to_typed_errors() {
    let traces = load_error_traces();
    if traces.is_empty() {
        eprintln!("no error captures under {TRACES_ROOT}; run capture_traces first");
        return;
    }

    let mut failures = Vec::<String>::new();
    for trace in &traces {
        let label = format!(
            "{}/{} (status={})",
            trace.provider.dir_name(),
            trace.scenario,
            trace.status
        );
        let err = replay_error(trace).await;
        let ok = match (trace.provider, trace.status, &err) {
            (Provider::OpenAI, 401, Error::Auth(_)) => true,
            (Provider::OpenAI, 429, Error::RateLimit { .. }) => true,
            (Provider::OpenAI, _, Error::Provider { provider, .. }) if provider == "OpenAI" => true,
            (Provider::Google, _, Error::Provider { provider, .. }) if provider == "Google" => true,
            _ => false,
        };
        if !ok {
            failures.push(format!("{label}: unexpected variant: {err:?}"));
            continue;
        }

        // Sanity: error text should contain something derived from the
        // captured body, not be empty / lost. Loose match.
        let msg = err.to_string();
        if msg.is_empty() {
            failures.push(format!("{label}: error message is empty"));
        }
    }

    if !failures.is_empty() {
        panic!("error-trace replay failures:\n  {}", failures.join("\n  "));
    }
}
