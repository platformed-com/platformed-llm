//! Verify that dropping a `Response` mid-stream actually closes the
//! underlying HTTP connection.
//!
//! `wiremock` is HTTP-level and doesn't expose connection-close signals,
//! so this test stands up a raw `tokio::net::TcpListener`, hand-rolls a
//! single HTTP/1.1 chunked SSE response, sends one event, and then calls
//! `read()` on the socket. If the client dropped the stream, `read()`
//! returns `Ok(0)` (peer FIN); if not, the test times out.
//!
//! This is the cancellation contract: reqwest+hyper must propagate the
//! drop down to a TCP close so the server stops uselessly producing
//! tokens.

use std::time::Duration;

use futures_util::StreamExt;
use platformed_llm::{LLMProvider, LLMRequest, OpenAIProvider, Prompt, StreamEvent};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

#[tokio::test]
async fn dropping_response_closes_underlying_connection() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let base_url = format!("http://127.0.0.1:{port}");

    let (observed_close_tx, observed_close_rx) = oneshot::channel::<bool>();

    let server = tokio::spawn(async move {
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

        // Send a minimal HTTP/1.1 chunked SSE response head.
        let head = b"HTTP/1.1 200 OK\r\n\
                     Content-Type: text/event-stream\r\n\
                     Transfer-Encoding: chunked\r\n\
                     Cache-Control: no-cache\r\n\
                     \r\n";
        socket.write_all(head).await.unwrap();

        // One real OpenAI-shaped SSE event the parser will accept.
        let frame = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n";
        let chunked = format!("{:X}\r\n{}\r\n", frame.len(), frame);
        socket.write_all(chunked.as_bytes()).await.unwrap();
        socket.flush().await.unwrap();

        // Now wait for the client to close. A clean FIN from the peer
        // shows up as `read()` returning Ok(0). If the client doesn't
        // drop the connection on `Response::drop`, this read blocks and
        // the outer timeout fires.
        let result = tokio::time::timeout(Duration::from_secs(3), socket.read(&mut tmp)).await;
        let observed_close = matches!(result, Ok(Ok(0)));
        let _ = observed_close_tx.send(observed_close);
    });

    let provider = OpenAIProvider::new_with_base_url("test-key".to_string(), base_url).unwrap();
    let req = LLMRequest::from_prompt("gpt-4o-mini", &Prompt::user("hi"));
    let response = provider.generate(&req).await.expect("generate should succeed");
    let mut stream = response.stream();

    let first = stream
        .next()
        .await
        .expect("server sent one event")
        .expect("event parses cleanly");
    match first {
        StreamEvent::ContentDelta { delta } => assert_eq!(delta, "hi"),
        other => panic!("unexpected first event: {other:?}"),
    }

    drop(stream);

    let observed = tokio::time::timeout(Duration::from_secs(5), observed_close_rx)
        .await
        .expect("server task never reported back")
        .expect("oneshot sender dropped — server task panicked");

    assert!(
        observed,
        "server did not observe a connection close after Response stream was dropped — \
         cancellation is not propagating to the underlying TCP connection",
    );

    server.await.unwrap();
}
