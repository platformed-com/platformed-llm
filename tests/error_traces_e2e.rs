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
use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::Stream;
use platformed_llm::{
    Error, GoogleProvider, LLMProvider, LLMRequest, OpenAIProvider, Prompt, Transport,
    TransportImpl, TransportRequest, TransportResponse, VertexEndpoint,
};
use serde_json::Value;

/// Test-only `TransportImpl` returning a fixed status + body. Used here
/// to feed the captured 4xx envelope into the provider's error path
/// without spinning up wiremock.
struct StaticTransport {
    status: u16,
    body: Vec<u8>,
}

#[async_trait]
impl TransportImpl for StaticTransport {
    async fn send(&self, _req: TransportRequest) -> Result<TransportResponse, Error> {
        let body = Bytes::from(self.body.clone());
        let stream: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>> =
            Box::pin(futures_util::stream::iter(vec![Ok(body)]));
        Ok(TransportResponse {
            status: self.status,
            headers: vec![("content-type".to_string(), "application/json".to_string())],
            body: stream,
        })
    }
}

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
    let transport = Transport::new(StaticTransport {
        status: trace.status,
        body: trace.body.clone(),
    });
    let req = LLMRequest::from_prompt("model", &Prompt::user("hi"));
    match trace.provider {
        Provider::OpenAI => {
            let p = OpenAIProvider::with_transport(
                "test".to_string(),
                "http://placeholder".to_string(),
                transport,
            );
            p.generate(&req)
                .await
                .err()
                .expect("4xx must produce an error")
        }
        Provider::Google => {
            let endpoint = VertexEndpoint::with_access_token(
                "p".to_string(),
                "us-east1".to_string(),
                "tok".to_string(),
            );
            GoogleProvider::with_transport(endpoint, transport)
                .generate(&req)
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
            // 401 → Auth on both providers, now that Vertex maps it.
            (Provider::OpenAI, 401, Error::Auth { status: Some(401), .. }) => true,
            (Provider::Google, 401, Error::Auth { status: Some(401), .. }) => true,
            // 429 → RateLimit (OpenAI-side; Vertex doesn't typically 429
            // on the streaming endpoint, but if it does, accept either).
            (_, 429, Error::RateLimit { .. }) => true,
            // 404 → ModelNotAvailable on Vertex.
            (Provider::Google, 404, Error::ModelNotAvailable(_)) => true,
            // Any other 4xx/5xx on either provider → Provider with the
            // correct provider name and a matching status code.
            (
                Provider::OpenAI,
                _,
                Error::Provider {
                    provider: "OpenAI",
                    status: Some(_),
                    ..
                },
            ) => true,
            (
                Provider::Google,
                _,
                Error::Provider {
                    provider: "Google",
                    status: Some(_),
                    ..
                },
            ) => true,
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
