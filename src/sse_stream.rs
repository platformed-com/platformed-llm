//! Stream adapter for parsing SSE (Server-Sent Events) from byte chunks.

use crate::Error;
use futures_util::{Stream, StreamExt};
use memchr::memchr2;
use std::collections::VecDeque;
use std::mem;
use std::pin::Pin;
use std::task::{ready, Context, Poll};

/// A Server-Sent Events (SSE) event.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SseEvent {
    /// Event type (optional).
    pub event_type: String,
    /// Event data.
    pub data: String,
    /// Event ID (optional).
    pub id: String,
    /// Retry delay in milliseconds (optional).
    pub retry: Option<u64>,
}

impl SseEvent {
    /// Create a new SSE event with just data.
    pub fn new(data: String) -> Self {
        Self {
            data,
            ..Default::default()
        }
    }

    /// Create a new SSE event with event type and data.
    pub fn with_type(event_type: String, data: String) -> Self {
        Self {
            event_type,
            data,
            ..Default::default()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
            && self.event_type.is_empty()
            && self.id.is_empty()
            && self.retry.is_none()
    }
}

/// A stream adapter that parses SSE events from a byte stream.
/// Uses line-based processing with state machine for robust line ending detection.
pub struct SseStream<S> {
    /// The underlying byte stream
    inner: S,
    /// Buffer for incomplete raw bytes from previous chunks
    line_buffer: Vec<u8>,
    /// Line ending detection state (preserved across buffer boundaries)
    last_seen_cr: bool,
    /// Parsed events ready to be yielded
    events: EventBuffer,
}

struct EventBuffer {
    current_event: SseEvent,
    events: VecDeque<SseEvent>,
}

impl EventBuffer {
    /// Create a new empty event buffer.
    fn new() -> Self {
        Self {
            current_event: SseEvent::default(),
            events: VecDeque::new(),
        }
    }

    /// Get the next event from the buffer, if available.
    fn pop(&mut self) -> Option<SseEvent> {
        self.events.pop_front()
    }

    fn dispatch_event(&mut self) {
        if self.current_event.data.is_empty() {
            // Ignore events with empty data field as per spec
            return;
        }
        if self.current_event.data.ends_with('\n') {
            // Remove trailing newline as per spec
            self.current_event.data.pop();
        }
        if self.current_event.event_type.is_empty() {
            // Default to "message" event type if not set
            self.current_event.event_type = "message".to_string();
        }
        self.events.push_back(mem::take(&mut self.current_event));
    }

    fn process_line(&mut self, line: &[u8]) -> Result<(), Error> {
        let line = std::str::from_utf8(line)
            .map_err(|e| Error::streaming(format!("Invalid UTF-8 in SSE event: {e}")))?;

        if line.is_empty() {
            self.dispatch_event();
        }

        let (field, mut value) = line.split_once(':').unwrap_or((line, ""));

        // Remove optional leading space after colon
        if value.starts_with(' ') {
            value = &value[1..];
        }

        match field {
            "" => {
                // Comment, do nothing
            }
            "event" => {
                self.current_event.event_type = value.to_string();
            }
            "data" => {
                self.current_event.data.push_str(value);
                self.current_event.data.push('\n');
            }
            "id" => {
                self.current_event.id = value.trim().to_string();
            }
            "retry" => {
                if let Ok(retry) = value.trim().parse() {
                    self.current_event.retry = Some(retry);
                }
            }
            _ => {} // Ignore unknown fields
        }
        Ok(())
    }
}

impl<S> SseStream<S> {
    /// Create a new SSE stream from a byte stream.
    pub fn new(stream: S) -> Self {
        Self {
            inner: stream,
            line_buffer: Vec::new(),
            last_seen_cr: false,
            events: EventBuffer::new(),
        }
    }

    /// Process the buffer using a state machine to detect line endings robustly.
    /// State is preserved across calls to handle line endings split across buffer boundaries.
    fn parse_buffer(&mut self, mut buffer: &[u8]) -> Result<(), Error> {
        while let Some(idx) = memchr2(b'\n', b'\r', buffer) {
            let is_nl = buffer[idx] == b'\n';
            if self.last_seen_cr && idx == 0 && is_nl {
                // Do nothing, found matching LF after CR
            } else if self.line_buffer.is_empty() {
                // No previous line buffer, process directly
                self.events.process_line(&buffer[..idx])?;
            } else {
                // We have a previous line buffer, combine it with the current line
                self.line_buffer.extend_from_slice(&buffer[..idx]);
                self.events.process_line(&self.line_buffer)?;
                self.line_buffer.clear();
            }

            self.last_seen_cr = !is_nl;
            buffer = &buffer[idx + 1..];
        }

        // Add any remaining bytes to the line buffer
        self.line_buffer.extend_from_slice(buffer);

        Ok(())
    }
}

impl<S, E> Stream for SseStream<S>
where
    S: Stream<Item = Result<bytes::Bytes, E>> + Unpin,
    E: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    type Item = Result<SseEvent, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            // First, yield any already-parsed events (FIFO order)
            if let Some(event) = self.events.pop() {
                return Poll::Ready(Some(Ok(event)));
            }

            // No buffered events, poll the underlying stream for more data
            if let Some(chunk) = ready!(self
                .inner
                .poll_next_unpin(cx)
                .map_err(|e| Error::streaming(format!("Stream error: {}", e.into())))?)
            {
                self.parse_buffer(&chunk)?;
            } else {
                if !self.line_buffer.is_empty() {
                    return Poll::Ready(Some(Err(Error::streaming(format!(
                        "Incomplete line buffer at end of stream: {}",
                        String::from_utf8_lossy(&self.line_buffer)
                    )))));
                }

                if !self.events.current_event.is_empty() {
                    return Poll::Ready(Some(Err(Error::streaming(format!(
                        "Incomplete event at end of stream: {:?}",
                        self.events.current_event
                    )))));
                }

                return Poll::Ready(None);
            };
        }
    }
}

/// Extension trait to add SSE parsing to byte streams.
pub trait SseStreamExt: Stream {
    /// Parse this byte stream as SSE events.
    fn sse_events(self) -> SseStream<Self>
    where
        Self: Sized,
    {
        SseStream::new(self)
    }
}

impl<S: Stream> SseStreamExt for S {}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;

    #[tokio::test]
    async fn test_sse_stream_complete_events() {
        let chunks: Vec<Result<bytes::Bytes, std::io::Error>> =
            vec![Ok(bytes::Bytes::from("data: Hello\n\ndata: World\n\n"))];
        let byte_stream = stream::iter(chunks);
        let mut sse_stream = byte_stream.sse_events();

        let event1 = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event1.data, "Hello");

        let event2 = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event2.data, "World");

        assert!(sse_stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_sse_stream_split_events() {
        let chunks: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
            Ok(bytes::Bytes::from("data: Hel")),
            Ok(bytes::Bytes::from("lo World\n\ndata: ")),
            Ok(bytes::Bytes::from("Second\n\n")),
        ];
        let byte_stream = stream::iter(chunks);
        let mut sse_stream = byte_stream.sse_events();

        let event1 = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event1.data, "Hello World");

        let event2 = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event2.data, "Second");

        assert!(sse_stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_sse_stream_multiline() {
        let chunks: Vec<Result<bytes::Bytes, std::io::Error>> =
            vec![Ok(bytes::Bytes::from("data: Line 1\ndata: Line 2\n\n"))];
        let byte_stream = stream::iter(chunks);
        let mut sse_stream = byte_stream.sse_events();

        let event = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event.data, "Line 1\nLine 2");
    }

    #[tokio::test]
    async fn test_sse_stream_with_fields() {
        let chunks: Vec<Result<bytes::Bytes, std::io::Error>> = vec![Ok(bytes::Bytes::from(
            "event: custom\ndata: Test\nid: 123\n\n",
        ))];
        let byte_stream = stream::iter(chunks);
        let mut sse_stream = byte_stream.sse_events();

        let event = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event.event_type, "custom".to_string());
        assert_eq!(event.data, "Test");
        assert_eq!(event.id, "123".to_string());
    }

    #[tokio::test]
    async fn test_sse_stream_utf8_boundary() {
        // Test UTF-8 character split across chunk boundaries
        // Using Euro symbol (€) which is 3 bytes in UTF-8: E2 82 AC
        let euro_bytes = "€".as_bytes();

        let chunks: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
            // Split the Euro symbol across chunks
            Ok(bytes::Bytes::from(
                [b"data: Price: ".as_slice(), &euro_bytes[..2]].concat(),
            )),
            Ok(bytes::Bytes::from([&euro_bytes[2..], b"100\n\n"].concat())),
        ];

        let byte_stream = stream::iter(chunks);
        let mut sse_stream = byte_stream.sse_events();

        let event = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event.data, "Price: €100");

        assert!(sse_stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_sse_stream_invalid_utf8_error() {
        // Test that invalid UTF-8 in a complete event properly returns an error
        let chunks: Vec<Result<bytes::Bytes, std::io::Error>> = vec![Ok(bytes::Bytes::from(
            b"data: Valid start \xFF\xFE invalid bytes\n\n".to_vec(),
        ))];

        let byte_stream = stream::iter(chunks);
        let mut sse_stream = byte_stream.sse_events();

        // Should get an error for invalid UTF-8
        let result = sse_stream.next().await.unwrap();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_line_ending_variations() {
        // Test comprehensive line ending handling
        let test_cases = vec![
            // (name, data, expected_events)
            (
                "Unix LF",
                "data: event1\n\ndata: event2\n\n",
                vec!["event1", "event2"],
            ),
            (
                "Windows CRLF",
                "data: event1\r\n\r\ndata: event2\r\n\r\n",
                vec!["event1", "event2"],
            ),
            (
                "Classic Mac CR",
                "data: event1\r\rdata: event2\r\r",
                vec!["event1", "event2"],
            ),
            (
                "Mixed LF first",
                "data: event1\n\ndata: event2\r\n\r\n",
                vec!["event1", "event2"],
            ),
            (
                "Mixed CRLF first",
                "data: event1\r\n\r\ndata: event2\n\n",
                vec!["event1", "event2"],
            ),
            ("Single event LF", "data: single\n\n", vec!["single"]),
            ("Single event CRLF", "data: single\r\n\r\n", vec!["single"]),
        ];

        for (test_name, data, expected) in test_cases {
            println!("Testing: {test_name}");

            let chunks: Vec<Result<bytes::Bytes, std::io::Error>> =
                vec![Ok(bytes::Bytes::from(data))];
            let byte_stream = stream::iter(chunks);
            let mut sse_stream = byte_stream.sse_events();

            let mut received_events = Vec::new();
            while let Some(result) = sse_stream.next().await {
                match result {
                    Ok(event) => received_events.push(event.data),
                    Err(e) => panic!("Unexpected error in {test_name}: {e}"),
                }
            }

            assert_eq!(
                received_events, expected,
                "Test '{test_name}' failed. Expected: {expected:?}, Got: {received_events:?}"
            );
        }
    }

    #[tokio::test]
    async fn test_buffer_boundary_splitting() {
        // Test the critical case where line separators are split across buffer boundaries

        // Case 1: CRLF split across boundaries (\r | \n\r\n)
        let chunks1: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
            Ok(bytes::Bytes::from("data: event1\r")), // Ends with \r
            Ok(bytes::Bytes::from("\n\r\ndata: event2\r\n\r\n")), // Starts with \n, contains complete event
        ];
        let byte_stream1 = stream::iter(chunks1);
        let mut sse_stream1 = byte_stream1.sse_events();

        let event1 = sse_stream1.next().await.unwrap().unwrap();
        assert_eq!(event1.data, "event1");

        let event2 = sse_stream1.next().await.unwrap().unwrap();
        assert_eq!(event2.data, "event2");

        assert!(sse_stream1.next().await.is_none());

        // Case 2: LF split across boundaries (data\n | \ndata)
        let chunks2: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
            Ok(bytes::Bytes::from("data: event1\n")),     // Ends with \n
            Ok(bytes::Bytes::from("\ndata: event2\n\n")), // Starts with \n, contains complete event
        ];
        let byte_stream2 = stream::iter(chunks2);
        let mut sse_stream2 = byte_stream2.sse_events();

        let event1 = sse_stream2.next().await.unwrap().unwrap();
        assert_eq!(event1.data, "event1");

        let event2 = sse_stream2.next().await.unwrap().unwrap();
        assert_eq!(event2.data, "event2");

        assert!(sse_stream2.next().await.is_none());

        // Case 3: Complex split (\r\n | \r\n)
        let chunks3: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
            Ok(bytes::Bytes::from("data: event1\r\n")), // Ends with \r\n
            Ok(bytes::Bytes::from("\r\ndata: event2\r\n\r\n")), // Starts with \r\n
        ];
        let byte_stream3 = stream::iter(chunks3);
        let mut sse_stream3 = byte_stream3.sse_events();

        let event1 = sse_stream3.next().await.unwrap().unwrap();
        assert_eq!(event1.data, "event1");

        let event2 = sse_stream3.next().await.unwrap().unwrap();
        assert_eq!(event2.data, "event2");

        assert!(sse_stream3.next().await.is_none());
    }

    #[tokio::test]
    async fn test_robust_separator_precedence() {
        // Test that our robust separator finding handles precedence correctly

        // Case 1: Mixed separators - CRLF should take precedence over LF within it
        let mixed_data = "data: event1\r\n\r\ndata: event2\n\ndata: event3\r\r";
        let chunks: Vec<Result<bytes::Bytes, std::io::Error>> =
            vec![Ok(bytes::Bytes::from(mixed_data))];
        let byte_stream = stream::iter(chunks);
        let mut sse_stream = byte_stream.sse_events();

        // Should correctly parse all three events with different separators
        let event1 = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event1.data, "event1");

        let event2 = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event2.data, "event2");

        // Note: The third event will be parsed by the stream end logic since \r\r is at the end
        let event3 = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event3.data, "event3");

        assert!(sse_stream.next().await.is_none());

        // Case 2: Ensure \n\n within \r\n\r\n doesn't cause false matches
        let tricky_data = "data: tricky\r\n\r\ndata: event\r\n\r\n";
        let chunks2: Vec<Result<bytes::Bytes, std::io::Error>> =
            vec![Ok(bytes::Bytes::from(tricky_data))];
        let byte_stream2 = stream::iter(chunks2);
        let mut sse_stream2 = byte_stream2.sse_events();

        let event1 = sse_stream2.next().await.unwrap().unwrap();
        assert_eq!(event1.data, "tricky");

        let event2 = sse_stream2.next().await.unwrap().unwrap();
        assert_eq!(event2.data, "event");

        assert!(sse_stream2.next().await.is_none());
    }

    #[tokio::test]
    async fn test_incomplete_crlf_sequences() {
        // Test the key edge cases for the state machine

        // Case 1: \r\n\r at end of buffer (incomplete, should not be treated as separator)
        let incomplete_data = "data: event1\r\n\r";
        let chunks1: Vec<Result<bytes::Bytes, std::io::Error>> =
            vec![Ok(bytes::Bytes::from(incomplete_data))];
        let byte_stream1 = stream::iter(chunks1);
        let mut sse_stream1 = byte_stream1.sse_events();

        // Should parse the incomplete event when stream ends
        let event1 = sse_stream1.next().await.unwrap().unwrap();
        assert_eq!(event1.data, "event1");
        assert!(sse_stream1.next().await.is_none());

        // Case 2: \r\n\r\n complete separator should work normally
        let complete_data = "data: event1\r\n\r\ndata: event2\r\n\r\n";
        let chunks2: Vec<Result<bytes::Bytes, std::io::Error>> =
            vec![Ok(bytes::Bytes::from(complete_data))];
        let byte_stream2 = stream::iter(chunks2);
        let mut sse_stream2 = byte_stream2.sse_events();

        let event1 = sse_stream2.next().await.unwrap().unwrap();
        assert_eq!(event1.data, "event1");

        let event2 = sse_stream2.next().await.unwrap().unwrap();
        assert_eq!(event2.data, "event2");

        assert!(sse_stream2.next().await.is_none());

        // Case 3: Mixed separators should all work
        let mixed_data = "data: event1\n\ndata: event2\r\rdata: event3\r\n\r\n";
        let chunks3: Vec<Result<bytes::Bytes, std::io::Error>> =
            vec![Ok(bytes::Bytes::from(mixed_data))];
        let byte_stream3 = stream::iter(chunks3);
        let mut sse_stream3 = byte_stream3.sse_events();

        let event1 = sse_stream3.next().await.unwrap().unwrap();
        assert_eq!(event1.data, "event1");

        let event2 = sse_stream3.next().await.unwrap().unwrap();
        assert_eq!(event2.data, "event2");

        let event3 = sse_stream3.next().await.unwrap().unwrap();
        assert_eq!(event3.data, "event3");

        assert!(sse_stream3.next().await.is_none());
    }
}
