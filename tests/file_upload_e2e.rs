#![cfg(feature = "openai")]
//! Offline replay of the real OpenAI file-upload (`Ref`) round-trip.
//!
//! Uses the bytes captured by
//! `cargo run --example capture_traces -- openai file_ref_image`:
//! - `file_ref_image.upload.response.json` — the real `POST /v1/files`
//!   response (the file object), and
//! - `file_ref_image.response.sse` — the real generate SSE stream.
//!
//! Replays the full **resolve → upload → reference → generate → parse** flow
//! against those bytes (no network) and asserts the generate request
//! referenced the uploaded `file_id`. This is the offline guard that the
//! lazy-upload wiring keeps producing the shape the live API accepted.
//!
//! No-ops if the capture hasn't been run (traces absent).

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use platformed_llm::providers::OpenAIProvider;
use platformed_llm::transport::{
    Transport, TransportImpl, TransportRequest, TransportResponse, UploadRequest,
};
use platformed_llm::{
    generate, Config, Error, FileResolver, FileSource, InputItem, Prompt, ProviderScope,
    ResolvedFile, ResolvedHandle, StreamEvent, UserPart,
};
use serde_json::Value;

const TRACE_DIR: &str = "tests/cross_provider/traces/openai";

fn trace_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(TRACE_DIR)
        .join(name)
}

/// Transport that replays the recorded upload + generate bytes and captures
/// the generate request body so the test can inspect the wire shape.
struct ReplayTransport {
    upload_response: Vec<u8>,
    generate_sse: Vec<u8>,
    generate_request_body: Arc<Mutex<Option<Vec<u8>>>>,
    upload_called: Arc<Mutex<bool>>,
}

fn static_response(status: u16, body: Vec<u8>) -> TransportResponse {
    let stream: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>> =
        Box::pin(futures_util::stream::iter(vec![Ok(Bytes::from(body))]));
    TransportResponse {
        status,
        headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
        body: stream,
    }
}

#[async_trait]
impl TransportImpl for ReplayTransport {
    async fn send(&self, req: TransportRequest) -> Result<TransportResponse, Error> {
        *self.generate_request_body.lock().unwrap() = Some(req.body.clone());
        Ok(static_response(200, self.generate_sse.clone()))
    }

    async fn send_upload(&self, req: UploadRequest) -> Result<TransportResponse, Error> {
        // Drain the multipart body so the streamed upload is genuinely
        // consumed, then return the recorded file object.
        let mut body = req.body;
        while let Some(chunk) = body.next().await {
            chunk?;
        }
        *self.upload_called.lock().unwrap() = true;
        Ok(static_response(200, self.upload_response.clone()))
    }
}

/// Resolver that always misses (forcing an upload) and opens a trivial
/// in-memory stream — the bytes are irrelevant offline since the
/// [`ReplayTransport`] returns the recorded file object regardless.
struct LocalFileResolver;

#[async_trait]
impl FileResolver for LocalFileResolver {
    async fn lookup(
        &self,
        _id: &str,
        _scope: &ProviderScope,
    ) -> Result<Option<ResolvedHandle>, Error> {
        Ok(None)
    }

    async fn open(&self, _id: &str, _scope: &ProviderScope) -> Result<ResolvedFile, Error> {
        Ok(ResolvedFile::Stream {
            media_type: "image/png".to_string(),
            content_length: Some(3),
            body: Box::pin(futures_util::stream::once(async {
                Ok(Bytes::from_static(b"png"))
            })),
        })
    }

    async fn store(
        &self,
        _id: &str,
        _scope: &ProviderScope,
        _handle: ResolvedHandle,
    ) -> Result<(), Error> {
        Ok(())
    }
}

#[tokio::test]
async fn file_ref_uploads_and_references_recorded_file_id() {
    let upload_path = trace_path("file_ref_image.upload.response.json");
    let sse_path = trace_path("file_ref_image.response.sse");
    if !upload_path.exists() || !sse_path.exists() {
        eprintln!("file_ref_image traces absent — run `capture_traces -- openai file_ref_image`");
        return;
    }
    let upload_response = std::fs::read(&upload_path).unwrap();
    let generate_sse = std::fs::read(&sse_path).unwrap();

    // The file_id the live upload returned — the value the generate request
    // must reference.
    let file_obj: Value = serde_json::from_slice(&upload_response).unwrap();
    let expected_file_id = file_obj["id"].as_str().expect("upload id").to_string();

    let gen_body = Arc::new(Mutex::new(None));
    let upload_called = Arc::new(Mutex::new(false));
    let transport = Transport::new(ReplayTransport {
        upload_response,
        generate_sse,
        generate_request_body: gen_body.clone(),
        upload_called: upload_called.clone(),
    });

    let provider = OpenAIProvider::with_transport(
        "test".to_string(),
        "http://placeholder/v1".to_string(),
        transport,
    )
    .with_file_resolver(Arc::new(LocalFileResolver));

    let prompt = Prompt::new().with_item(InputItem::User {
        content: vec![
            UserPart::Text("Briefly describe what you see in this image.".to_string()),
            UserPart::Image(FileSource::Ref("img-1".to_string())),
        ],
    });
    let cfg = Config::builder("gpt-4o-mini").max_tokens(256).build();

    let response = generate(&provider, &prompt, &cfg).await.expect("generate");
    let mut events = Vec::new();
    let mut stream = response.stream();
    while let Some(ev) = stream.next().await {
        events.push(ev.expect("stream event"));
    }

    // 1. The Ref triggered a real upload.
    assert!(
        *upload_called.lock().unwrap(),
        "send_upload should have been called to resolve the file Ref"
    );

    // 2. The generate request referenced the uploaded file_id via input_image
    //    (and not as an inline image_url).
    let body = gen_body
        .lock()
        .unwrap()
        .clone()
        .expect("generate request body captured");
    let req: Value = serde_json::from_slice(&body).unwrap();
    let parts = req["input"][0]["content"]
        .as_array()
        .expect("user content parts");
    let image = parts
        .iter()
        .find(|p| p["type"] == "input_image")
        .expect("an input_image part");
    assert_eq!(
        image["file_id"], expected_file_id,
        "generate must reference the uploaded file_id"
    );
    assert!(
        image.get("image_url").map(|v| v.is_null()).unwrap_or(true),
        "a resolved handle must not also emit an inline image_url"
    );

    // 3. The recorded response parsed into a terminal Done event.
    assert!(
        matches!(events.last(), Some(StreamEvent::Done { .. })),
        "replayed stream should end with Done"
    );
    assert!(events.len() > 1, "expected a non-trivial event sequence");
}
