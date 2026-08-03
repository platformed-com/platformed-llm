#![cfg(feature = "reqwest")]
//! Retry classification for mid-body connection drops.
//!
//! hyper reports a connection lost partway through a response body as
//! a reqwest *decode* error ("error decoding response body") wrapping
//! the transport cause, so `Error::is_retryable` must look through the
//! source chain: a `hyper`/IO cause means a transient drop, while a
//! payload that genuinely failed to parse (serde) is terminal.
//! `reqwest::Error` offers no public constructor, so both shapes are
//! produced against a real socket: a raw `tokio::net::TcpListener`
//! (the `cancellation.rs` pattern) that either truncates the body or
//! returns a complete-but-unparseable one.

use platformed_llm::transport::{Transport, TransportRequest};
use platformed_llm::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Serve one connection: read the request headers, write `response`,
/// then drop the socket.
async fn serve_once(response: Vec<u8>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        // Drain the request headers — read until \r\n\r\n.
        let mut buf = Vec::new();
        let mut tmp = [0u8; 1024];
        loop {
            let n = socket.read(&mut tmp).await.unwrap();
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        socket.write_all(&response).await.unwrap();
        socket.flush().await.unwrap();
        // Dropping the socket closes the connection; with a
        // truncated body the client sees the drop mid-read.
    });
    format!("http://127.0.0.1:{port}")
}

#[tokio::test]
async fn mid_body_connection_drop_is_retryable() {
    // Content-Length promises 1000 bytes; only a few arrive before
    // the connection closes.
    let base_url = serve_once(
        b"HTTP/1.1 200 OK\r\n\
          Content-Type: text/event-stream\r\n\
          Content-Length: 1000\r\n\
          \r\n\
          data: partial"
            .to_vec(),
    )
    .await;

    let transport = Transport::reqwest().unwrap();
    let response = transport
        .send(TransportRequest {
            url: base_url,
            headers: vec![],
            body: vec![],
        })
        .await
        .expect("headers arrive fine; the failure is mid-body");
    assert_eq!(response.status, 200);

    let err = response
        .collect_body()
        .await
        .expect_err("truncated body must surface an error");
    let Error::Transport(ref e) = err else {
        panic!("expected Error::Transport, got: {err:?}");
    };
    assert!(
        e.is_decode(),
        "hyper surfaces the truncation as a decode-kind error \
         (the shape under test); got: {e:?}",
    );
    assert!(
        err.is_retryable(),
        "a connection lost mid-body is transient and must be retryable",
    );
}

#[tokio::test]
async fn corrupt_payload_decode_stays_terminal() {
    // A complete, correctly-delimited response whose body simply
    // isn't JSON. `reqwest::Response::json()` wraps the serde failure
    // as a decode-kind error with no transport-level cause.
    let base_url = serve_once(
        b"HTTP/1.1 200 OK\r\n\
          Content-Type: application/json\r\n\
          Content-Length: 8\r\n\
          Connection: close\r\n\
          \r\n\
          not json"
            .to_vec(),
    )
    .await;

    let reqwest_err = reqwest::Client::new()
        .get(&base_url)
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .expect_err("body is not JSON");
    assert!(reqwest_err.is_decode());

    let err = Error::from(reqwest_err);
    assert!(
        !err.is_retryable(),
        "a payload that failed to parse fails identically on a \
         re-read — must stay terminal",
    );
}
