//! Stream adapter for parsing SSE (Server-Sent Events) from byte chunks.

use futures_util::{Stream, StreamExt};
use memchr::memmem;
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{ready, Context, Poll};
use crate::Error;

/// A Server-Sent Events (SSE) event.
#[derive(Debug, Clone, PartialEq)]
pub struct SseEvent {
    /// Event type (optional).
    pub event_type: Option<String>,
    /// Event data.
    pub data: String,
    /// Event ID (optional).
    pub id: Option<String>,
    /// Retry delay in milliseconds (optional).
    pub retry: Option<u64>,
}

impl SseEvent {
    /// Create a new SSE event with just data.
    pub fn new(data: String) -> Self {
        Self {
            event_type: None,
            data,
            id: None,
            retry: None,
        }
    }
    
    /// Create a new SSE event with event type and data.
    pub fn with_type(event_type: String, data: String) -> Self {
        Self {
            event_type: Some(event_type),
            data,
            id: None,
            retry: None,
        }
    }
    
    /// Check if this is a "done" event (used by OpenAI to signal end of stream).
    pub fn is_done(&self) -> bool {
        self.data.trim() == "[DONE]"
    }
}

/// A stream adapter that parses SSE events from a byte stream.
/// Maintains internal state to handle events split across chunks.
pub struct SseStream<S> {
    /// The underlying byte stream
    inner: S,
    /// Buffer for incomplete raw bytes from previous chunks
    buffer: Vec<u8>,
    /// Parsed events ready to be yielded
    events: VecDeque<SseEvent>,
}

impl<S> SseStream<S> {
    /// Create a new SSE stream from a byte stream.
    pub fn new(stream: S) -> Self {
        Self {
            inner: stream,
            buffer: Vec::new(),
            events: VecDeque::new(),
        }
    }
    
    /// Parse complete SSE events from the buffer.
    /// Adds parsed events directly to the internal event list.
    fn parse_buffer(&mut self) -> Result<(), Error> {
        // SSE event separator is "\n\n" (two consecutive newlines)
        let separator = b"\n\n";
        let finder = memmem::Finder::new(separator);
        let mut start = 0;
        
        // Find complete events using memmem for efficient byte pattern matching
        while let Some(pos) = finder.find(&self.buffer[start..]) {
            let event_end = start + pos;
            let event_bytes = &self.buffer[start..event_end];
            
            // Convert this event's bytes to UTF-8 string
            let event_text = std::str::from_utf8(event_bytes)
                .map_err(|e| Error::streaming(format!("Invalid UTF-8 in SSE event: {e}")))?;
            
            // Parse the event
            if let Some(event) = Self::parse_single_event(event_text) {
                self.events.push_back(event);
            }
            
            // Move past this event (including the separator)
            start = event_end + separator.len();
        }
        
        // Remove processed bytes from buffer
        if start > 0 {
            self.buffer.drain(..start);
        }
        
        Ok(())
    }
    
    /// Parse a single complete SSE event from its text representation.
    fn parse_single_event(event_text: &str) -> Option<SseEvent> {
        let mut event_type = None;
        let mut data_lines = Vec::new();
        let mut id = None;
        let mut retry = None;
        
        for line in event_text.lines() {
            let line = line.trim_end(); // Only trim end to preserve intentional leading spaces
            
            // Skip empty lines and comments
            if line.is_empty() || line.starts_with(':') {
                continue;
            }
            
            // Parse field:value pairs using split_once
            if let Some((field, mut value)) = line.split_once(':') {
                // Remove optional leading space after colon
                if value.starts_with(' ') {
                    value = &value[1..];
                }
                
                match field {
                    "event" => event_type = Some(value.to_string()),
                    "data" => data_lines.push(value.to_string()),
                    "id" => id = Some(value.to_string()),
                    "retry" => retry = value.parse().ok(),
                    _ => {} // Ignore unknown fields
                }
            }
        }
        
        // No data means no event
        if data_lines.is_empty() {
            return None;
        }
        
        Some(SseEvent {
            event_type,
            data: data_lines.join("\n"),
            id,
            retry,
        })
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
            if let Some(event) = self.events.pop_front() {
                return Poll::Ready(Some(Ok(event)));
            }
            
            // No buffered events, poll the underlying stream for more data
            let chunk = match ready!(self.inner.poll_next_unpin(cx)) {
                Some(Ok(chunk)) => chunk,
                Some(Err(e)) => {
                    return Poll::Ready(Some(Err(Error::streaming(
                        format!("Stream error: {}", e.into())
                    ))));
                }
                None => {
                    // Stream ended - try to parse any remaining data as a final event
                    if !self.buffer.is_empty() {
                        if let Ok(text) = std::str::from_utf8(&self.buffer) {
                            let text = text.trim();
                            if !text.is_empty() {
                                // Try to parse the remaining buffer as a final event
                                if let Some(event) = Self::parse_single_event(text) {
                                    self.buffer.clear();
                                    return Poll::Ready(Some(Ok(event)));
                                }
                                // Only warn if the remaining data doesn't parse as a valid event
                                // This handles cases where the stream ends without the final \n\n
                            }
                        }
                        self.buffer.clear();
                    }
                    return Poll::Ready(None);
                }
            };
            
            // Append raw bytes to buffer
            self.buffer.extend_from_slice(&chunk);
            
            // Check buffer size limit
            if self.buffer.len() > 1_000_000 {
                self.buffer.clear();
                return Poll::Ready(Some(Err(Error::streaming(
                    "SSE buffer exceeded maximum size".to_string()
                ))));
            }
            
            // Parse any complete events and continue loop
            if let Err(e) = self.parse_buffer() {
                return Poll::Ready(Some(Err(e)));
            }
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
        let chunks: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
            Ok(bytes::Bytes::from("data: Hello\n\ndata: World\n\n")),
        ];
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
        let chunks: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
            Ok(bytes::Bytes::from("data: Line 1\ndata: Line 2\n\n")),
        ];
        let byte_stream = stream::iter(chunks);
        let mut sse_stream = byte_stream.sse_events();
        
        let event = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event.data, "Line 1\nLine 2");
    }
    
    #[tokio::test]
    async fn test_sse_stream_with_fields() {
        let chunks: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
            Ok(bytes::Bytes::from("event: custom\ndata: Test\nid: 123\n\n")),
        ];
        let byte_stream = stream::iter(chunks);
        let mut sse_stream = byte_stream.sse_events();
        
        let event = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event.event_type, Some("custom".to_string()));
        assert_eq!(event.data, "Test");
        assert_eq!(event.id, Some("123".to_string()));
    }
    
    #[tokio::test]
    async fn test_sse_stream_utf8_boundary() {
        // Test UTF-8 character split across chunk boundaries
        // Using Euro symbol (€) which is 3 bytes in UTF-8: E2 82 AC
        let euro_bytes = "€".as_bytes();
        
        let chunks: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
            // Split the Euro symbol across chunks
            Ok(bytes::Bytes::from([b"data: Price: ".as_slice(), &euro_bytes[..2]].concat())),
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
        let chunks: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
            Ok(bytes::Bytes::from(b"data: Valid start \xFF\xFE invalid bytes\n\n".to_vec())),
        ];
        
        let byte_stream = stream::iter(chunks);
        let mut sse_stream = byte_stream.sse_events();
        
        // Should get an error for invalid UTF-8
        let result = sse_stream.next().await.unwrap();
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_sse_stream_ends_without_final_newline() {
        // Test that streams ending without final \n\n are handled gracefully
        // This simulates the Gemini SSE format that ends with "data: [DONE]" without final newlines
        let chunks: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
            Ok(bytes::Bytes::from("data: First event\n\n")),
            Ok(bytes::Bytes::from("data: [DONE]")), // No final \n\n
        ];
        
        let byte_stream = stream::iter(chunks);
        let mut sse_stream = byte_stream.sse_events();
        
        // Should get the first complete event
        let event1 = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event1.data, "First event");
        
        // Should get the final event even without trailing \n\n
        let event2 = sse_stream.next().await.unwrap().unwrap();
        assert_eq!(event2.data, "[DONE]");
        
        // Stream should end cleanly without warnings
        assert!(sse_stream.next().await.is_none());
    }
}