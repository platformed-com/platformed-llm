//! Snapshot the unified `StreamEvent` sequence produced by replaying each
//! captured wire trace.
//!
//! The trace files under `tests/cross_provider/traces/` are real bytes
//! that came over the wire. This test replays each one through its
//! provider's pipeline (parser → state machine → accumulator) and
//! compares the resulting `Vec<StreamEvent>` against a checked-in
//! `.events.txt` golden file. Any drift — a deleted event, a renamed
//! field, an off-by-one delta — shows up directly in the PR diff.
//!
//! Set `UPDATE_SNAPSHOTS=1` to overwrite the golden files (e.g. after a
//! deliberate change to the unified event shape or a re-capture).
//!
//! No-ops if no traces are present.

use std::fs;
use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use platformed_llm::{
    AnthropicViaVertexProvider, GoogleProvider, LLMProvider, LLMRequest, OpenAIProvider, Prompt,
    StreamEvent,
};
use serde_json::Value;
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, ResponseTemplate};

const TRACES_ROOT: &str = "tests/cross_provider/traces";

#[derive(Debug, Clone, Copy)]
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
struct Trace {
    provider: Provider,
    scenario: String,
    response_sse: Vec<u8>,
    snapshot_path: PathBuf,
    request_model: String,
}

fn load_all_traces() -> Vec<Trace> {
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
            let request_path = dir.join(format!("{scenario}.request.json"));
            let request_model = read_json(&request_path)
                .as_ref()
                .and_then(|v| v.get("model"))
                .and_then(|v| v.as_str())
                .unwrap_or("model")
                .to_string();
            // Error captures don't have an SSE-shaped body. They're
            // exercised by `error_traces_e2e` instead.
            let meta = read_json(&dir.join(format!("{scenario}.meta.json")));
            let status = meta
                .as_ref()
                .and_then(|v| v.get("status"))
                .and_then(|v| v.as_u64())
                .unwrap_or(200);
            if status != 200 {
                continue;
            }
            let response_sse = fs::read(&path).unwrap();
            let snapshot_path = dir.join(format!("{scenario}.events.txt"));
            traces.push(Trace {
                provider,
                scenario,
                response_sse,
                snapshot_path,
                request_model,
            });
        }
    }
    traces
}

fn read_json(path: &Path) -> Option<Value> {
    let raw = fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

async fn replay(trace: &Trace) -> Vec<StreamEvent> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path_regex(".*"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(trace.response_sse.clone(), "text/event-stream"),
        )
        .mount(&server)
        .await;

    let req = LLMRequest::from_prompt(trace.request_model.clone(), &Prompt::user("replay"));
    let response = match trace.provider {
        Provider::OpenAI => {
            let p = OpenAIProvider::new_with_base_url("test".to_string(), server.uri()).unwrap();
            p.generate(&req).await.unwrap()
        }
        Provider::Google => GoogleProvider::new_with_base_url(
            "p".to_string(),
            "us-east1".to_string(),
            "tok".to_string(),
            server.uri(),
        )
        .unwrap()
        .generate(&req)
        .await
        .unwrap(),
        Provider::Anthropic => AnthropicViaVertexProvider::new_with_base_url(
            "p".to_string(),
            "us-east1".to_string(),
            "tok".to_string(),
            server.uri(),
        )
        .unwrap()
        .generate(&req)
        .await
        .unwrap(),
    };

    let mut events = Vec::new();
    let mut stream = response.stream();
    while let Some(ev) = stream.next().await {
        match ev {
            Ok(e) => events.push(e),
            Err(e) => panic!("stream error during snapshot replay: {e}"),
        }
    }
    events
}

/// Mask volatile IDs (Gemini synthesizes a fresh UUID per parse, so
/// raw values would make every test run produce a "drift") while
/// preserving the *correlation* between `OutputItemAdded.id` and the
/// matching `FunctionCallComplete.call_id`. Two identical real IDs map
/// to the same placeholder; two different reals map to distinct ones.
struct IdMasker {
    map: std::collections::HashMap<String, String>,
}

impl IdMasker {
    fn new() -> Self {
        Self {
            map: std::collections::HashMap::new(),
        }
    }
    fn mask(&mut self, id: &str) -> String {
        if id.is_empty() {
            return String::from("\"\"");
        }
        let next = format!("<id-{}>", self.map.len() + 1);
        let placeholder = self
            .map
            .entry(id.to_string())
            .or_insert(next)
            .clone();
        format!("{placeholder:?}")
    }
}

/// Format the unified event sequence as a deterministic, line-oriented
/// text snapshot. Stable under cosmetic changes (fields are explicitly
/// named, ordering is fixed) so unrelated `Debug` derive churn doesn't
/// break the snapshots.
fn format_events(events: &[StreamEvent]) -> String {
    use platformed_llm::OutputItemInfo;
    let mut masker = IdMasker::new();
    let mut out = String::new();
    for ev in events {
        match ev {
            StreamEvent::OutputItemAdded { item } => match item {
                OutputItemInfo::Text => out.push_str("OutputItemAdded text\n"),
                OutputItemInfo::Reasoning => out.push_str("OutputItemAdded reasoning\n"),
                OutputItemInfo::FunctionCall { name, id } => {
                    let id = masker.mask(id);
                    out.push_str(&format!(
                        "OutputItemAdded function_call name={name:?} id={id}\n"
                    ));
                }
            },
            StreamEvent::ContentDelta { delta } => {
                out.push_str(&format!("ContentDelta {delta:?}\n"));
            }
            StreamEvent::ReasoningDelta { delta } => {
                out.push_str(&format!("ReasoningDelta {delta:?}\n"));
            }
            StreamEvent::ReasoningSignature { signature } => {
                // Signatures are model-version-volatile opaque blobs; record
                // only the byte length so the snapshot stays stable across
                // re-captures while still asserting one was emitted.
                out.push_str(&format!("ReasoningSignature len={}\n", signature.len()));
            }
            StreamEvent::FunctionCallComplete { call } => {
                let call_id = masker.mask(&call.call_id);
                out.push_str(&format!(
                    "FunctionCallComplete call_id={call_id} name={:?} arguments={:?}\n",
                    call.name, call.arguments
                ));
            }
            StreamEvent::Done {
                finish_reason,
                usage,
            } => {
                out.push_str(&format!(
                    "Done finish={finish_reason:?} input={} output={}",
                    usage.input_tokens, usage.output_tokens
                ));
                if let Some(v) = usage.cache_read_input_tokens {
                    out.push_str(&format!(" cache_read={v}"));
                }
                if let Some(v) = usage.cache_creation_input_tokens {
                    out.push_str(&format!(" cache_creation={v}"));
                }
                if let Some(v) = usage.reasoning_tokens {
                    out.push_str(&format!(" reasoning={v}"));
                }
                out.push('\n');
            }
            StreamEvent::Error { error } => {
                out.push_str(&format!("Error {error:?}\n"));
            }
        }
    }
    out
}

/// Shape sanity that must hold for any captured success-path replay,
/// independent of the snapshot. Run *before* write/compare so a buggy
/// bootstrap can't silently bake bad behaviour into the golden file.
fn validate_event_sequence(events: &[StreamEvent]) -> Result<(), String> {
    let dones = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::Done { .. }))
        .count();
    if dones != 1 {
        return Err(format!("expected exactly one Done event, got {dones}"));
    }
    let last_is_done = matches!(events.last(), Some(StreamEvent::Done { .. }));
    if !last_is_done {
        return Err("Done event is not the last event in the stream".to_string());
    }
    Ok(())
}

#[tokio::test]
async fn unified_event_snapshots_match() {
    let traces = load_all_traces();
    if traces.is_empty() {
        eprintln!("no traces under {TRACES_ROOT} — run capture_traces first");
        return;
    }
    let update = std::env::var("UPDATE_SNAPSHOTS").is_ok();

    let mut failures = Vec::<String>::new();
    for trace in &traces {
        let label = format!("{}/{}", trace.provider.dir_name(), trace.scenario);
        let events = replay(trace).await;

        if let Err(msg) = validate_event_sequence(&events) {
            failures.push(format!("{label}: {msg}"));
            continue;
        }

        let actual = format_events(&events);

        let snapshot_exists = trace.snapshot_path.exists();
        if update || !snapshot_exists {
            fs::write(&trace.snapshot_path, &actual).expect("write snapshot");
            if !snapshot_exists {
                eprintln!("bootstrapped snapshot for {label}");
            }
            continue;
        }

        let expected = fs::read_to_string(&trace.snapshot_path).expect("read snapshot");
        if expected != actual {
            failures.push(format!(
                "{label}: snapshot drift\n--- expected ({})\n{}\n--- actual\n{}\n",
                trace.snapshot_path.display(),
                expected,
                actual,
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "{} snapshot mismatch(es); rerun with UPDATE_SNAPSHOTS=1 to update:\n\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}
