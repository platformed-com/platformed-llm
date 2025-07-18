//! Streaming utilities for Server-Sent Events (SSE) parsing.

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

/// Convert a stream of bytes to SSE events.
/// This is a simplified implementation for Phase 2.
pub fn parse_sse_chunk(chunk: &[u8]) -> Result<Vec<SseEvent>, Error> {
    let text = std::str::from_utf8(chunk)
        .map_err(|e| Error::streaming(format!("Invalid UTF-8: {e}")))?;
    
    let mut events = Vec::new();
    
    for line in text.lines() {
        let line = line.trim();
        
        // Skip empty lines and comments
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        
        // Parse data lines (most common case for OpenAI)
        if let Some(data) = line.strip_prefix("data: ") {
            events.push(SseEvent::new(data.to_string()));
        }
        // Parse event type lines
        else if let Some(event_type) = line.strip_prefix("event: ") {
            // For now, create an event with just the type
            events.push(SseEvent::with_type(event_type.to_string(), String::new()));
        }
    }
    
    Ok(events)
}

/// Parse a single SSE line into an event.
/// This is a simplified parser that handles the most common case: "data: {json}"
#[cfg(test)]
fn parse_sse_line(line: &str) -> Option<SseEvent> {
    if let Some(data) = line.strip_prefix("data: ") {
        Some(SseEvent::new(data.to_string()))
    } else {
        line.strip_prefix("event: ").map(|event_type| SseEvent::with_type(event_type.to_string(), String::new()))
    }
}

/// Parse a full SSE event from multiple lines.
/// This handles the complete SSE format with multiple fields.
pub fn parse_sse_event(lines: &[String]) -> Option<SseEvent> {
    let mut event_type = None;
    let mut data_lines = Vec::new();
    let mut id = None;
    let mut retry = None;
    
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        
        if let Some(value) = line.strip_prefix("event: ") {
            event_type = Some(value.to_string());
        } else if let Some(value) = line.strip_prefix("data: ") {
            data_lines.push(value.to_string());
        } else if let Some(value) = line.strip_prefix("id: ") {
            id = Some(value.to_string());
        } else if let Some(value) = line.strip_prefix("retry: ") {
            retry = value.parse().ok();
        }
    }
    
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sse_event_creation() {
        let event = SseEvent::new("test data".to_string());
        assert_eq!(event.data, "test data");
        assert!(event.event_type.is_none());
        
        let event_with_type = SseEvent::with_type("message".to_string(), "test data".to_string());
        assert_eq!(event_with_type.event_type, Some("message".to_string()));
        assert_eq!(event_with_type.data, "test data");
    }
    
    #[test]
    fn test_done_event() {
        let done_event = SseEvent::new("[DONE]".to_string());
        assert!(done_event.is_done());
        
        let normal_event = SseEvent::new("normal data".to_string());
        assert!(!normal_event.is_done());
    }
    
    #[test]
    fn test_parse_sse_line() {
        let data_line = "data: {\"message\": \"hello\"}";
        let event = parse_sse_line(data_line).unwrap();
        assert_eq!(event.data, "{\"message\": \"hello\"}");
        
        let event_line = "event: message";
        let event = parse_sse_line(event_line).unwrap();
        assert_eq!(event.event_type, Some("message".to_string()));
    }
    
    #[test]
    fn test_parse_sse_event() {
        let lines = vec![
            "event: message".to_string(),
            "data: first line".to_string(),
            "data: second line".to_string(),
            "id: 123".to_string(),
        ];
        
        let event = parse_sse_event(&lines).unwrap();
        assert_eq!(event.event_type, Some("message".to_string()));
        assert_eq!(event.data, "first line\nsecond line");
        assert_eq!(event.id, Some("123".to_string()));
    }
    
    #[test]
    fn test_parse_sse_chunk() {
        let chunk = b"data: test message 1\n\ndata: test message 2\n\n";
        let events = parse_sse_chunk(chunk).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "test message 1");
        assert_eq!(events[1].data, "test message 2");
    }
}