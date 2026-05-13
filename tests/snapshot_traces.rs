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
use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use platformed_llm::accumulator::ResponseAccumulator;
use platformed_llm::{
    AnthropicViaVertexProvider, CompleteResponse, Error, GoogleProvider, LLMProvider, LLMRequest,
    OpenAIProvider, OutputItem, Prompt, StreamEvent, Transport, TransportImpl, TransportRequest,
    TransportResponse, VertexEndpoint,
};
use serde_json::Value;

/// Test-only `TransportImpl` that always returns a fixed status + body,
/// regardless of the request. Replaces wiremock for the snapshot replay —
/// we don't need an actual HTTP server to feed bytes into the lib's
/// parser pipeline.
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
            headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
            body: stream,
        })
    }
}

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
    let transport = Transport::new(StaticTransport {
        status: 200,
        body: trace.response_sse.clone(),
    });
    let req = LLMRequest::from_prompt(trace.request_model.clone(), &Prompt::user("replay"));
    let response = match trace.provider {
        Provider::OpenAI => {
            let p = OpenAIProvider::with_transport(
                "test".to_string(),
                "http://placeholder".to_string(),
                transport,
            );
            p.generate(&req).await.unwrap()
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
                .unwrap()
        }
        Provider::Anthropic => {
            let endpoint = VertexEndpoint::with_access_token(
                "p".to_string(),
                "us-east1".to_string(),
                "tok".to_string(),
            );
            AnthropicViaVertexProvider::with_transport(endpoint, transport)
                .generate(&req)
                .await
                .unwrap()
        }
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
/// text snapshot.
fn format_events(events: &[StreamEvent]) -> String {
    use platformed_llm::{PartKind, PartUpdate};
    let mut masker = IdMasker::new();
    let mut out = String::new();
    for ev in events {
        match ev {
            StreamEvent::PartStart { index, kind } => match kind {
                PartKind::Text => out.push_str(&format!("PartStart[{index}] text\n")),
                PartKind::Reasoning => out.push_str(&format!("PartStart[{index}] reasoning\n")),
                PartKind::Refusal => out.push_str(&format!("PartStart[{index}] refusal\n")),
                PartKind::RedactedReasoning { data } => out.push_str(&format!(
                    "PartStart[{index}] redacted_reasoning len={}\n",
                    data.len()
                )),
                PartKind::ToolCall { call_id, name } => {
                    let masked = masker.mask(call_id);
                    out.push_str(&format!(
                        "PartStart[{index}] tool_call call_id={masked} name={name:?}\n"
                    ));
                }
            },
            StreamEvent::Delta { index, delta } => {
                out.push_str(&format!("Delta[{index}] {delta:?}\n"));
            }
            StreamEvent::PartUpdate { index, update } => match update {
                PartUpdate::Signature(s) => out.push_str(&format!(
                    "PartUpdate[{index}] signature len={}\n",
                    s.len()
                )),
                PartUpdate::Annotation(a) => out.push_str(&format!(
                    "PartUpdate[{index}] annotation kind={:?} source={:?}\n",
                    a.kind, a.source
                )),
            },
            StreamEvent::PartEnd { index } => {
                out.push_str(&format!("PartEnd[{index}]\n"));
            }
            StreamEvent::Done { finish_reason, .. } => {
                // Usage masked to keep snapshots stable across re-captures.
                out.push_str(&format!(
                    "Done finish={finish_reason:?} input=<n> output=<n>\n"
                ));
            }
            StreamEvent::Error { error } => {
                out.push_str(&format!("Error {error:?}\n"));
            }
        }
    }
    out
}

/// Format the final `CompleteResponse` produced by feeding the events
/// through `ResponseAccumulator`.
fn format_complete(complete: &CompleteResponse) -> String {
    use platformed_llm::AssistantPart;
    let mut out = String::new();
    out.push_str(&format!("finish={:?}\n", complete.finish_reason));
    for (i, item) in complete.output.iter().enumerate() {
        match item {
            OutputItem::Assistant { content } => {
                for (j, part) in content.iter().enumerate() {
                    match part {
                        AssistantPart::Text { content, annotations } => out.push_str(&format!(
                            "output[{i}].part[{j}] text {content:?} annotations={}\n",
                            annotations.len()
                        )),
                        AssistantPart::Reasoning { content, signature } => {
                            let sig_len = signature.as_ref().map(|s| s.len()).unwrap_or(0);
                            out.push_str(&format!(
                                "output[{i}].part[{j}] reasoning len={} signature_len={sig_len}\n",
                                content.len()
                            ));
                        }
                        AssistantPart::RedactedReasoning { data } => out.push_str(&format!(
                            "output[{i}].part[{j}] redacted_reasoning len={}\n",
                            data.len()
                        )),
                        AssistantPart::Refusal(s) => out.push_str(&format!(
                            "output[{i}].part[{j}] refusal {s:?}\n"
                        )),
                        AssistantPart::ToolCall(call) => out.push_str(&format!(
                            "output[{i}].part[{j}] tool_call name={:?} arguments={:?}\n",
                            call.name, call.arguments
                        )),
                        AssistantPart::CacheBreakpoint => out.push_str(&format!(
                            "output[{i}].part[{j}] cache_breakpoint\n"
                        )),
                    }
                }
            }
        }
    }
    out
}

/// Shape sanity that must hold for any captured success-path replay,
/// independent of the snapshot. Run *before* write/compare so a buggy
/// bootstrap can't silently bake bad behaviour into the golden file.
///
/// Also catches "usage parsing silently broke" — the snapshot masks
/// usage numeric values to keep re-captures stable, so without this
/// check a parser regression that zeroed out token counts would slip
/// through.
fn validate_event_sequence(events: &[StreamEvent]) -> Result<(), String> {
    let dones: Vec<&StreamEvent> = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::Done { .. }))
        .collect();
    if dones.len() != 1 {
        return Err(format!(
            "expected exactly one Done event, got {}",
            dones.len()
        ));
    }
    let last_is_done = matches!(events.last(), Some(StreamEvent::Done { .. }));
    if !last_is_done {
        return Err("Done event is not the last event in the stream".to_string());
    }
    if let StreamEvent::Done { usage, .. } = dones[0] {
        if usage.input_tokens == 0 {
            return Err(
                "Done.usage.input_tokens is 0 — usage block was probably not parsed".to_string(),
            );
        }
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

        // Run the same events through the accumulator to produce a
        // `CompleteResponse` — the surface most callers actually see
        // (`response.buffer()` / `response.text()`). Snapshotting it
        // closes the trace → unified events → CompleteResponse chain.
        let mut accumulator = ResponseAccumulator::new();
        let mut accumulator_failed = false;
        for ev in &events {
            if let Err(e) = accumulator.process_event(ev.clone()) {
                failures.push(format!("{label}: accumulator: {e}"));
                accumulator_failed = true;
                break;
            }
        }
        if accumulator_failed {
            continue;
        }
        let complete = match accumulator.finalize() {
            Ok(c) => c,
            Err(e) => {
                failures.push(format!("{label}: accumulator.finalize: {e}"));
                continue;
            }
        };

        let actual = format!(
            "{}\n=== final ===\n{}",
            format_events(&events),
            format_complete(&complete),
        );

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
