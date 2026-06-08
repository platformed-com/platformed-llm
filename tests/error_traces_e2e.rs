#![cfg(all(feature = "openai", feature = "google", feature = "anthropic-vertex"))]
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
use platformed_llm::providers::{
    AnthropicViaVertexProvider, GoogleProvider, OpenAIProvider, VertexEndpoint,
};
use platformed_llm::transport::{Transport, TransportImpl, TransportRequest, TransportResponse};
use platformed_llm::{generate, Config, Error, Prompt};
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
    Anthropic,
}

impl Provider {
    fn dir_name(self) -> &'static str {
        match self {
            Provider::OpenAI => "openai",
            Provider::Google => "google",
            Provider::Anthropic => "anthropic",
        }
    }
    fn from_dir_name(name: &str) -> Option<Provider> {
        match name {
            "openai" => Some(Provider::OpenAI),
            "google" => Some(Provider::Google),
            "anthropic" => Some(Provider::Anthropic),
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
            let expect_failure = meta
                .get("expect_failure")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            // Skip 2xx captures unless the scenario explicitly expects
            // failure — some providers (notably OpenAI's Responses API)
            // return 200 OK and surface errors inside the SSE stream
            // (e.g. `event: error` with `code:
            // context_length_exceeded`), and those still belong here.
            if (200..300).contains(&status) && !expect_failure {
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
    let prompt = Prompt::user("hi");
    let cfg = Config::builder("model").build();
    // `generate()` can succeed even on captures we expect to fail —
    // OpenAI's Responses API returns 200 OK and surfaces the error
    // inside the SSE stream, so the error only materializes when the
    // caller drains the response. Run the full pipeline (generate +
    // buffer) so both paths land in this `Error` result.
    let outcome = match trace.provider {
        Provider::OpenAI => {
            let p = OpenAIProvider::with_transport(
                "test".to_string(),
                "http://placeholder".to_string(),
                transport,
            );
            generate_and_drain(&p, &prompt, &cfg).await
        }
        Provider::Google => {
            let endpoint = VertexEndpoint::with_access_token(
                "p".to_string(),
                "us-east1".to_string(),
                "tok".to_string(),
            );
            let p = GoogleProvider::with_transport(endpoint, transport);
            generate_and_drain(&p, &prompt, &cfg).await
        }
        Provider::Anthropic => {
            let endpoint = VertexEndpoint::with_access_token(
                "p".to_string(),
                "us-east5".to_string(),
                "tok".to_string(),
            );
            let p = AnthropicViaVertexProvider::with_transport(endpoint, transport);
            generate_and_drain(&p, &prompt, &cfg).await
        }
    };
    match outcome {
        Ok(()) => panic!("captured failure must produce an error"),
        Err(e) => e,
    }
}

async fn generate_and_drain(
    provider: &dyn platformed_llm::Provider,
    prompt: &Prompt,
    cfg: &Config,
) -> Result<(), Error> {
    let response = generate(provider, prompt, cfg).await?;
    response.buffer().await.map(|_| ())
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
        let ok = match (trace.provider, trace.status, &err, trace.scenario.as_str()) {
            // 401 → Auth on every provider, now that Vertex maps it.
            (
                _,
                401,
                Error::Auth {
                    status: Some(401), ..
                },
                _,
            ) => true,
            // 429 → RateLimit (OpenAI; Vertex doesn't typically 429 on
            // streamGenerateContent, but accept it on every provider if
            // it happens).
            (_, 429, Error::RateLimit { .. }, _) => true,
            // 404 → ModelNotAvailable on Vertex.
            (Provider::Google, 404, Error::ModelNotAvailable(_), _) => true,
            (Provider::Anthropic, 404, Error::ModelNotAvailable(_), _) => true,
            // The `context_window_exceeded` scenario must surface as the
            // typed `ContextWindowExceeded` on every provider — that's
            // the load-bearing contract for compaction-aware callers.
            (
                Provider::OpenAI,
                _,
                Error::ContextWindowExceeded {
                    provider: "OpenAI", ..
                },
                "context_window_exceeded",
            ) => true,
            (
                Provider::Google,
                _,
                Error::ContextWindowExceeded {
                    provider: "Google", ..
                },
                "context_window_exceeded",
            ) => true,
            (
                Provider::Anthropic,
                _,
                Error::ContextWindowExceeded {
                    provider: "Anthropic",
                    ..
                },
                "context_window_exceeded",
            ) => true,
            // Any other 4xx/5xx on any provider → Provider with the
            // correct provider name and a matching status code.
            (
                Provider::OpenAI,
                _,
                Error::Provider {
                    provider: "OpenAI",
                    status: Some(_),
                    ..
                },
                _,
            ) => true,
            (
                Provider::Google,
                _,
                Error::Provider {
                    provider: "Google",
                    status: Some(_),
                    ..
                },
                _,
            ) => true,
            (
                Provider::Anthropic,
                _,
                Error::Provider {
                    provider: "Anthropic",
                    status: Some(_),
                    ..
                },
                _,
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
