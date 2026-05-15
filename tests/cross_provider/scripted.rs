//! `ScriptedTransport` — the wiremock replacement for the cross-provider
//! function-calling tests.
//!
//! Each scripted turn is a pair of `(expected_request_body, response_sse)`.
//! On each `send()`, the transport:
//! 1. Pops the next expected/response pair.
//! 2. Deserializes the actual request body as JSON.
//! 3. Asserts it equals the expected — a request-shape regression panics
//!    here with a deep `assert_eq!` diff.
//! 4. Returns the response SSE as a single-chunk streaming body.
//!
//! This is the exact moral equivalent of wiremock's `body_json` matcher
//! plus `ResponseTemplate::set_body_string`, but in-process and ~30 LOC.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Mutex;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::Stream;
use platformed_llm::transport::{TransportImpl, TransportRequest, TransportResponse};
use platformed_llm::Error;
use serde_json::Value;

pub struct ScriptedTurn {
    pub expected_body: Value,
    pub response_sse: Vec<u8>,
}

pub struct ScriptedTransport {
    turns: Mutex<VecDeque<ScriptedTurn>>,
}

impl ScriptedTransport {
    pub fn new(turns: Vec<ScriptedTurn>) -> Self {
        Self {
            turns: Mutex::new(turns.into()),
        }
    }
}

#[async_trait]
impl TransportImpl for ScriptedTransport {
    async fn send(&self, req: TransportRequest) -> Result<TransportResponse, Error> {
        let turn = self
            .turns
            .lock()
            .unwrap()
            .pop_front()
            .expect("ScriptedTransport called more times than scripted");

        let actual: Value =
            serde_json::from_slice(&req.body).expect("request body sent by lib was not valid JSON");
        assert_eq!(
            actual, turn.expected_body,
            "request body did not match expected payload",
        );

        let body = Bytes::from(turn.response_sse);
        let stream: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>> =
            Box::pin(futures_util::stream::iter(vec![Ok(body)]));
        Ok(TransportResponse {
            status: 200,
            headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
            body: stream,
        })
    }
}

/// Read a fixture file relative to the project root.
pub fn load_fixture(filename: &str) -> Vec<u8> {
    std::fs::read(filename).unwrap_or_else(|_| panic!("failed to load test fixture: {filename}"))
}
