#![cfg(feature = "google")]
//! Offline exercise of the Gemini (Vertex) GCS upload path for file `Ref`s.
//!
//! A configured `GoogleProvider` resolves an image `Ref` by streaming it to
//! Cloud Storage and referencing the resulting `gs://` URI. This test drives
//! that whole flow with a mock transport: the `send_upload` call (the GCS
//! media upload) is answered with a canned 2xx object, and the generate call
//! replays a real recorded Gemini response. It asserts the upload hit the GCS
//! endpoint and the generate request referenced the `gs://` URI via
//! `fileData.fileUri`.

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use platformed_llm::providers::{GoogleProvider, VertexEndpoint};
use platformed_llm::transport::{
    Transport, TransportImpl, TransportRequest, TransportResponse, UploadRequest,
};
use platformed_llm::{
    generate, Config, Error, FileResolver, FileSource, InputItem, Prompt, ProviderScope,
    ResolvedFile, ResolvedHandle, StreamEvent, UserPart,
};
use serde_json::Value;

fn google_trace(name: &str) -> Vec<u8> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/cross_provider/traces/google")
        .join(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

struct GcsReplayTransport {
    generate_sse: Vec<u8>,
    generate_request_body: Arc<Mutex<Option<Vec<u8>>>>,
    upload_url: Arc<Mutex<Option<String>>>,
}

fn static_response(status: u16, body: Vec<u8>) -> TransportResponse {
    let stream: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>> =
        Box::pin(futures_util::stream::iter(vec![Ok(Bytes::from(body))]));
    TransportResponse {
        status,
        headers: vec![("content-type".to_string(), "application/json".to_string())],
        body: stream,
    }
}

#[async_trait]
impl TransportImpl for GcsReplayTransport {
    async fn send(&self, req: TransportRequest) -> Result<TransportResponse, Error> {
        *self.generate_request_body.lock().unwrap() = Some(req.body.clone());
        Ok(static_response(200, self.generate_sse.clone()))
    }

    async fn send_upload(&self, req: UploadRequest) -> Result<TransportResponse, Error> {
        *self.upload_url.lock().unwrap() = Some(req.url.clone());
        // Drain the streamed object bytes, then return a GCS object metadata
        // body (the uploader builds the gs:// URI from bucket + name itself).
        let mut body = req.body;
        while let Some(chunk) = body.next().await {
            chunk?;
        }
        let object = br#"{"kind":"storage#object","bucket":"test-bucket"}"#.to_vec();
        Ok(static_response(200, object))
    }
}

/// Resolver that always misses and opens a trivial in-memory stream.
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
async fn gemini_ref_uploads_to_gcs_and_references_gs_uri() {
    let generate_sse = google_trace("multi_modal_image.response.sse");

    let gen_body = Arc::new(Mutex::new(None));
    let upload_url = Arc::new(Mutex::new(None));
    let transport = Transport::new(GcsReplayTransport {
        generate_sse,
        generate_request_body: gen_body.clone(),
        upload_url: upload_url.clone(),
    });

    let endpoint = VertexEndpoint::with_access_token(
        "proj-1".to_string(),
        "us-east1".to_string(),
        "tok".to_string(),
    );
    let provider = GoogleProvider::with_transport(endpoint, transport)
        .with_gcs_bucket("test-bucket")
        .with_file_resolver(Arc::new(LocalFileResolver));

    let prompt = Prompt::new().with_item(InputItem::User {
        content: vec![
            UserPart::Text("Describe this image.".to_string()),
            UserPart::Image(FileSource::Ref("img-1".to_string())),
        ],
    });
    let cfg = Config::builder("gemini-2.5-flash").max_tokens(256).build();

    let response = generate(&provider, &prompt, &cfg).await.expect("generate");
    let mut events = Vec::new();
    let mut stream = response.stream();
    while let Some(ev) = stream.next().await {
        events.push(ev.expect("stream event"));
    }

    // 1. The Ref was uploaded to the GCS media-upload endpoint.
    let url = upload_url
        .lock()
        .unwrap()
        .clone()
        .expect("send_upload should have been called");
    assert!(
        url.contains("storage.googleapis.com/upload/storage/v1/b/test-bucket/o")
            && url.contains("uploadType=media"),
        "upload should target the GCS media-upload endpoint, got: {url}"
    );

    // 2. The generate request referenced the uploaded object as a gs:// URI.
    let body = gen_body
        .lock()
        .unwrap()
        .clone()
        .expect("generate request body captured");
    let req: Value = serde_json::from_slice(&body).unwrap();
    let mut found_gs = None;
    for content in req["contents"].as_array().expect("contents") {
        for part in content["parts"].as_array().expect("parts") {
            if let Some(uri) = part.get("fileData").and_then(|f| f.get("fileUri")) {
                found_gs = uri.as_str().map(str::to_string);
            }
        }
    }
    let gs = found_gs.expect("a fileData.fileUri part");
    assert!(
        gs.starts_with("gs://test-bucket/"),
        "Ref must resolve to a gs:// URI in the configured bucket, got: {gs}"
    );

    // 3. The recorded Gemini response parsed to a terminal Done.
    assert!(
        matches!(events.last(), Some(StreamEvent::Done { .. })),
        "replayed stream should end with Done"
    );
}
