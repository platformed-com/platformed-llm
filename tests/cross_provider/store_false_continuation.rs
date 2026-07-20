//! Regression test for OpenAI response-id chaining under `store: false`.
//!
//! OpenAI's Responses API returns a `response.completed` frame carrying an
//! `id` *even when the request set `store: false`* — the recorded
//! `traces/openai/function_call.response.sse` (captured with `store: false`,
//! per its `function_call.request.json`) contains one. The accumulator
//! surfaces that id as a `ProviderContinuation::OpenAI`, so folding the
//! response back into the next turn puts a continuation marker in history.
//!
//! The bug: a follow-up would then thread that id through as
//! `previous_response_id` while still `store: false`. Nothing was retained
//! server-side to chain from, so OpenAI rejects the request with
//! `previous_response_not_found`. This test drives the real consumer flow —
//! replay the recorded bytes → accumulate → `with_response` → follow-up —
//! over ground-truth captured bytes, and pins that the follow-up does **not**
//! chain under `store: false`.

use std::fs;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use platformed_llm::accumulator::ResponseAccumulator;
use platformed_llm::providers::OpenAIProvider;
use platformed_llm::transport::{Transport, TransportImpl, TransportRequest, TransportResponse};
use platformed_llm::{generate, Config, Error, Prompt, ProviderContinuation};
use serde_json::Value;

/// Transport that records the outbound request body and replays a fixed
/// response stream — lets the test inspect what the lib emitted while
/// feeding it canned bytes.
struct CapturingTransport {
    body: Arc<Mutex<Option<Vec<u8>>>>,
    response: Vec<u8>,
}

impl CapturingTransport {
    fn new(response: Vec<u8>) -> (Self, Arc<Mutex<Option<Vec<u8>>>>) {
        let body = Arc::new(Mutex::new(None));
        (
            CapturingTransport {
                body: body.clone(),
                response,
            },
            body,
        )
    }
}

#[async_trait]
impl TransportImpl for CapturingTransport {
    async fn send(&self, req: TransportRequest) -> Result<TransportResponse, Error> {
        *self.body.lock().unwrap() = Some(req.body);
        let body = Bytes::from(self.response.clone());
        let stream: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>> =
            Box::pin(futures_util::stream::iter(vec![Ok(body)]));
        Ok(TransportResponse {
            status: 200,
            headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
            body: stream,
        })
    }
}

/// Minimal well-formed OpenAI Responses stream for the follow-up turn — we
/// only care about the request the lib *sends*, not this reply.
const TRIVIAL_RESPONSE: &str = concat!(
    r#"data: {"type":"response.created","response":{"id":"resp_followup","object":"response","created_at":1,"status":"in_progress","model":"gpt-4o-mini","output":[]}}"#,
    "\n\n",
    r#"data: {"type":"response.completed","response":{"id":"resp_followup","object":"response","created_at":1,"status":"completed","model":"gpt-4o-mini","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}"#,
    "\n\n",
);

fn openai_provider(response: Vec<u8>) -> (OpenAIProvider, Arc<Mutex<Option<Vec<u8>>>>) {
    let (transport, body) = CapturingTransport::new(response);
    let provider = OpenAIProvider::with_transport(
        "test-key".to_string(),
        "http://placeholder".to_string(),
        Transport::new(transport),
    );
    (provider, body)
}

#[tokio::test]
async fn store_false_response_id_is_not_chained_on_followup() {
    // Turn 1: replay the recorded `store: false` capture through the real
    // pipeline to obtain a genuine `CompleteResponse`.
    let recorded = fs::read("tests/cross_provider/traces/openai/function_call.response.sse")
        .expect("recorded openai function_call trace present");
    let (provider1, _) = openai_provider(recorded);
    let turn1 = Prompt::system("You have access to weather data.")
        .with_user("What's the weather like in Paris?");
    // `store` defaults to false — the production default and the mode the
    // trace was captured in.
    let cfg = Config::builder("gpt-4o-mini").max_tokens(256).build();

    let response = generate(&provider1, &turn1, &cfg)
        .await
        .expect("turn 1 generate");
    let mut accumulator = ResponseAccumulator::new();
    let mut stream = response.stream();
    while let Some(ev) = stream.next().await {
        accumulator
            .process_event(ev.expect("stream event"))
            .expect("accumulate");
    }
    let tool_call_id = accumulator
        .completed_function_calls()
        .first()
        .map(|c| c.call_id.clone());
    let complete = accumulator.finalize().expect("finalize turn 1");

    // The captured `store: false` response still carried a response id, and
    // the lib surfaced it as an OpenAI continuation. This is the fact that
    // makes the chaining bug reachable at all — pin it against real bytes.
    assert!(
        matches!(
            complete.continuation(),
            Some(ProviderContinuation::OpenAI { .. })
        ),
        "recorded store:false response should surface an OpenAI continuation, got {:?}",
        complete.continuation(),
    );

    // Turn 2: fold the real response back in the way a consumer does, add the
    // tool result, and send the follow-up — still `store: false`.
    let mut turn2 = turn1.with_response(&complete);
    if let Some(call_id) = tool_call_id {
        turn2 = turn2.with_tool_result(call_id, r#"{"temperature_c":22,"condition":"sunny"}"#);
    }
    turn2 = turn2.with_user("Summarise it in one short sentence.");

    let (provider2, sent_body) = openai_provider(TRIVIAL_RESPONSE.as_bytes().to_vec());
    let _ = generate(&provider2, &turn2, &cfg)
        .await
        .expect("turn 2 generate");

    let sent: Value = serde_json::from_slice(
        &sent_body
            .lock()
            .unwrap()
            .clone()
            .expect("follow-up body captured"),
    )
    .expect("follow-up body is JSON");

    // The store gate must hold: with nothing stored server-side, the
    // follow-up may not chain on `previous_response_id` (doing so is exactly
    // what OpenAI rejects with `previous_response_not_found`).
    assert!(
        sent.get("previous_response_id").is_none(),
        "under store:false the follow-up must not chain on previous_response_id, body: {sent}",
    );
    assert_eq!(
        sent.get("store"),
        Some(&Value::Bool(false)),
        "follow-up should be sent with store:false",
    );
}
