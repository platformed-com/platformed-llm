//! Cross-provider model-switching contract tests.
//!
//! When a [`CompleteResponse`] from provider A is folded into a
//! [`Prompt`] and sent to provider B, parts that are A-specific
//! (provider-builtin calls, foreign continuation markers, reasoning on
//! providers without a reasoning input channel) must be silently
//! dropped from the wire body; parts that B *does* have an equivalent
//! for (e.g. Anthropic's Thinking block ↔ `AssistantPart::Reasoning`)
//! must round-trip.
//!
//! These tests are the safety net against the kind of bug that
//! `with_response`-style refactors can introduce: snapshots / scripted
//! fixtures can bake in *broken* behaviour by accident (the continuation
//! roundtrip bug was an example), so we assert directly on the wire
//! shape instead of comparing to a frozen fixture.

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::Stream;
use platformed_llm::providers::{
    AnthropicViaVertexProvider, GoogleProvider, OpenAIProvider, VertexEndpoint,
};
use platformed_llm::transport::{Transport, TransportImpl, TransportRequest, TransportResponse};
use platformed_llm::{
    AssistantPart, CompleteResponse, Config, Error, FinishReason, Prompt, Provider,
    ProviderBuiltin, ProviderContinuation, Usage,
};
use serde_json::Value;

/// Transport that captures the outbound request body and returns a
/// trivial empty-stream response. Lets tests inspect what the lib
/// *would* have sent, without scripting an exact byte match.
struct CapturingTransport {
    body: Arc<Mutex<Option<Vec<u8>>>>,
    response_sse: Vec<u8>,
}

impl CapturingTransport {
    fn new(response_sse: &str) -> (Self, Arc<Mutex<Option<Vec<u8>>>>) {
        let body = Arc::new(Mutex::new(None));
        (
            CapturingTransport {
                body: body.clone(),
                response_sse: response_sse.as_bytes().to_vec(),
            },
            body,
        )
    }
}

#[async_trait]
impl TransportImpl for CapturingTransport {
    async fn send(&self, req: TransportRequest) -> Result<TransportResponse, Error> {
        *self.body.lock().unwrap() = Some(req.body);
        let body = Bytes::from(self.response_sse.clone());
        let stream: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>> =
            Box::pin(futures_util::stream::iter(vec![Ok(body)]));
        Ok(TransportResponse {
            status: 200,
            headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
            body: stream,
        })
    }
}

/// Build a `CompleteResponse` carrying every assistant-part kind we
/// care about for cross-provider switching: visible Text, an internal
/// Reasoning summary, a BuiltinToolCall (provider-side), and a
/// `Continuation` marker (provider-side) — all on a single assistant
/// turn.
fn rich_assistant_turn() -> CompleteResponse {
    CompleteResponse {
        content: vec![
            AssistantPart::Reasoning {
                content: "Internal scratch: 2+2 is 4.".into(),
                signature: Some("sig_abc".into()),
            },
            AssistantPart::Text {
                content: "The answer is 4.".into(),
                annotations: Vec::new(),
            },
            AssistantPart::BuiltinToolCall {
                kind: ProviderBuiltin::WebSearch,
                arguments: r#"{"queries":["2+2"]}"#.into(),
                result: None,
            },
            AssistantPart::Continuation(ProviderContinuation::OpenAI {
                response_id: "resp_prior".into(),
            }),
        ],
        finish_reason: FinishReason::Stop,
        usage: Usage::default(),
    }
}

/// Walk an OpenAI Responses-API request body and collect every input
/// item type (`"message"`, `"function_call"`, …).
fn openai_input_item_types(body: &Value) -> Vec<&str> {
    body["input"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v["type"].as_str()).collect())
        .unwrap_or_default()
}

/// Walk a Gemini request body and collect every part-key on the
/// `contents` array entries (`"text"`, `"functionCall"`, etc.).
fn gemini_part_keys(body: &Value) -> Vec<String> {
    body["contents"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .flat_map(|c| c["parts"].as_array().cloned().unwrap_or_default())
                .flat_map(|p| {
                    p.as_object()
                        .map(|o| o.keys().cloned().collect::<Vec<_>>())
                        .unwrap_or_default()
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Walk an Anthropic request body's `messages` array, return the
/// content-block types emitted for the assistant turn (the second
/// message). Returns an empty vec if absent.
fn anthropic_assistant_block_types(body: &Value) -> Vec<&str> {
    let Some(messages) = body["messages"].as_array() else {
        return Vec::new();
    };
    let Some(assistant) = messages.iter().find(|m| m["role"] == "assistant") else {
        return Vec::new();
    };
    // Anthropic content is either a bare string or an array of blocks.
    if assistant["content"].is_string() {
        return vec!["text"];
    }
    assistant["content"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v["type"].as_str()).collect())
        .unwrap_or_default()
}

const OPENAI_TRIVIAL_RESPONSE: &str = concat!(
    r#"data: {"type":"response.created","response":{"id":"resp_x","object":"response","created_at":1,"status":"in_progress","model":"gpt-4","output":[]}}"#,
    "\n\n",
    r#"data: {"type":"response.completed","response":{"id":"resp_x","object":"response","created_at":1,"status":"completed","model":"gpt-4","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}"#,
    "\n\n",
);

const GEMINI_TRIVIAL_RESPONSE: &str = concat!(
    r#"data: {"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}"#,
    "\n\n",
);

const ANTHROPIC_TRIVIAL_RESPONSE: &str = concat!(
    "event: message_start\n",
    r#"data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}"#,
    "\n\n",
    "event: message_delta\n",
    r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}"#,
    "\n\n",
    "event: message_stop\n",
    r#"data: {"type":"message_stop"}"#,
    "\n\n",
);

async fn send_to_openai(prompt: &Prompt) -> Value {
    let (transport, body) = CapturingTransport::new(OPENAI_TRIVIAL_RESPONSE);
    let provider = OpenAIProvider::with_transport(
        "k".into(),
        "http://placeholder".into(),
        Transport::new(transport),
    );
    let cfg = Config::new("gpt-4");
    let _ = provider
        .generate(prompt, &cfg)
        .await
        .expect("generate succeeded");
    let bytes = body.lock().unwrap().clone().expect("body captured");
    serde_json::from_slice(&bytes).expect("body is JSON")
}

async fn send_to_gemini(prompt: &Prompt) -> Value {
    let (transport, body) = CapturingTransport::new(GEMINI_TRIVIAL_RESPONSE);
    let endpoint = VertexEndpoint::with_access_token("p".into(), "us-east1".into(), "tok".into());
    let provider = GoogleProvider::with_transport(endpoint, Transport::new(transport));
    let cfg = Config::new("gemini");
    let _ = provider
        .generate(prompt, &cfg)
        .await
        .expect("generate succeeded");
    let bytes = body.lock().unwrap().clone().expect("body captured");
    serde_json::from_slice(&bytes).expect("body is JSON")
}

async fn send_to_anthropic(prompt: &Prompt) -> Value {
    let (transport, body) = CapturingTransport::new(ANTHROPIC_TRIVIAL_RESPONSE);
    let endpoint = VertexEndpoint::with_access_token("p".into(), "us-east1".into(), "tok".into());
    let provider = AnthropicViaVertexProvider::with_transport(endpoint, Transport::new(transport));
    let cfg = Config::new("claude-3");
    let _ = provider
        .generate(prompt, &cfg)
        .await
        .expect("generate succeeded");
    let bytes = body.lock().unwrap().clone().expect("body captured");
    serde_json::from_slice(&bytes).expect("body is JSON")
}

/// On OpenAI: Reasoning, BuiltinToolCall, and a foreign-provider
/// Continuation all silently drop. The assistant turn is reduced to
/// the bare text, and history continues with the follow-up user
/// message — `previous_response_id` is set to `resp_prior` because
/// it's an OpenAI continuation.
#[tokio::test]
async fn openai_drops_unsupported_parts() {
    let prompt = Prompt::user("hi")
        .with_response(&rich_assistant_turn())
        .with_user("follow-up");
    let body = send_to_openai(&prompt).await;
    let serialized = serde_json::to_string(&body).unwrap();

    // The continuation marker carries an OpenAI hint, so prior history
    // is elided and `previous_response_id` is set on the wire.
    assert_eq!(
        body["previous_response_id"], "resp_prior",
        "OpenAI: continuation should thread through as previous_response_id",
    );
    // Only the follow-up user message reaches the wire.
    assert_eq!(
        openai_input_item_types(&body),
        vec!["message"],
        "OpenAI body should contain only the post-continuation user turn, got: {serialized}",
    );
    // The dropped parts must not appear anywhere in the wire body —
    // an easy regex check against the raw JSON catches accidental
    // leakage (e.g. a Reasoning content slipping into a system-style
    // message).
    assert!(
        !serialized.contains("Internal scratch"),
        "OpenAI body leaked Reasoning content: {serialized}",
    );
    assert!(
        !serialized.contains("queries"),
        "OpenAI body leaked BuiltinToolCall arguments: {serialized}",
    );
}

/// On Gemini: an OpenAI continuation marker is unrecognised
/// (model-switching contract) so it doesn't elide history. Reasoning
/// and BuiltinToolCall still drop because Gemini has no input channel
/// for them.
#[tokio::test]
async fn gemini_drops_unsupported_parts_and_ignores_foreign_continuation() {
    let prompt = Prompt::user("hi")
        .with_response(&rich_assistant_turn())
        .with_user("follow-up");
    let body = send_to_gemini(&prompt).await;
    let serialized = serde_json::to_string(&body).unwrap();

    // No `cachedContent` — the continuation is an OpenAI marker, not a
    // Gemini one.
    assert!(
        body.get("cachedContent").is_none(),
        "Gemini should ignore foreign continuation; got cachedContent={:?}",
        body.get("cachedContent"),
    );
    // The assistant turn's surviving parts on Gemini are just the
    // visible text. Reasoning/BuiltinToolCall drop; the user turns
    // remain ("hi" and "follow-up").
    let part_keys = gemini_part_keys(&body);
    // Only `text` parts allowed.
    assert!(
        part_keys.iter().all(|k| k == "text"),
        "Gemini body contains non-text part keys after switching: {part_keys:?}\nbody={serialized}",
    );
    // Reasoning content didn't leak.
    assert!(
        !serialized.contains("Internal scratch"),
        "Gemini body leaked Reasoning content: {serialized}",
    );
    // BuiltinToolCall didn't leak.
    assert!(
        !serialized.contains("queries"),
        "Gemini body leaked BuiltinToolCall arguments: {serialized}",
    );
}

/// On Anthropic: Reasoning *is* preserved (it maps to Anthropic's
/// `thinking` content block, signature included). BuiltinToolCall and
/// the foreign continuation marker drop.
#[tokio::test]
async fn anthropic_preserves_reasoning_drops_provider_side_parts() {
    let prompt = Prompt::user("hi")
        .with_response(&rich_assistant_turn())
        .with_user("follow-up");
    let body = send_to_anthropic(&prompt).await;
    let serialized = serde_json::to_string(&body).unwrap();

    // The assistant turn surfaces a `thinking` + `text` block pair.
    let block_types = anthropic_assistant_block_types(&body);
    assert!(
        block_types.contains(&"thinking"),
        "Anthropic should preserve Reasoning as a thinking block; got: {block_types:?}\nbody={serialized}",
    );
    assert!(
        block_types.contains(&"text"),
        "Anthropic should preserve Text alongside Reasoning; got: {block_types:?}",
    );
    // No block for the BuiltinToolCall — provider-side, drops.
    assert!(
        !block_types.contains(&"tool_use"),
        "Anthropic body should not include a tool_use block from BuiltinToolCall: {block_types:?}",
    );
    // Reasoning signature comes along.
    assert!(
        serialized.contains("sig_abc"),
        "Anthropic body should include reasoning signature: {serialized}",
    );
    // Builtin tool arguments must not leak.
    assert!(
        !serialized.contains("queries"),
        "Anthropic body leaked BuiltinToolCall arguments: {serialized}",
    );
}

/// `RedactedReasoning` only has a typed home on Anthropic. Sending it
/// to OpenAI or Gemini must drop without leaking the opaque blob into
/// the wire body.
#[tokio::test]
async fn redacted_reasoning_drops_on_non_anthropic_providers() {
    let prior = CompleteResponse {
        content: vec![
            AssistantPart::RedactedReasoning {
                data: "OPAQUE_BLOB_DATA_12345".into(),
            },
            AssistantPart::Text {
                content: "visible answer".into(),
                annotations: Vec::new(),
            },
        ],
        finish_reason: FinishReason::Stop,
        usage: Usage::default(),
    };
    let prompt = Prompt::user("hi")
        .with_response(&prior)
        .with_user("follow-up");

    let openai_body = send_to_openai(&prompt).await;
    let openai_serialized = serde_json::to_string(&openai_body).unwrap();
    assert!(
        !openai_serialized.contains("OPAQUE_BLOB"),
        "OpenAI body leaked redacted reasoning blob: {openai_serialized}",
    );

    let gemini_body = send_to_gemini(&prompt).await;
    let gemini_serialized = serde_json::to_string(&gemini_body).unwrap();
    assert!(
        !gemini_serialized.contains("OPAQUE_BLOB"),
        "Gemini body leaked redacted reasoning blob: {gemini_serialized}",
    );

    // Anthropic *should* round-trip the blob.
    let anthropic_body = send_to_anthropic(&prompt).await;
    let anthropic_serialized = serde_json::to_string(&anthropic_body).unwrap();
    assert!(
        anthropic_serialized.contains("OPAQUE_BLOB"),
        "Anthropic should preserve the redacted reasoning blob: {anthropic_serialized}",
    );
}
