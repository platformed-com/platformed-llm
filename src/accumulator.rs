//! Delta accumulation logic for streaming responses.

use crate::types::{StreamEvent, Usage, FinishReason, FunctionCall};
use crate::{CompleteResponse, OutputItem};
use crate::Error;

/// Accumulates streaming deltas into a complete response.
#[derive(Debug, Default)]
pub struct ResponseAccumulator {
    /// Ordered output items (text, function calls, etc.)
    output_items: Vec<OutputItem>,
    /// Final finish reason (if received).
    finish_reason: Option<FinishReason>,
    /// Final usage statistics (if received).
    usage: Option<Usage>,
}

// Removed PartialFunctionCallBuilder - no longer needed since we handle complete calls only

impl ResponseAccumulator {
    /// Create a new response accumulator.
    pub fn new() -> Self {
        Self::default()
    }
    
    
    /// Process a stream event and update the accumulation.
    pub fn process_event(&mut self, event: StreamEvent) -> Result<(), Error> {
        match event {
            StreamEvent::ContentDelta { delta } => {
                // Append text to the most recent text output item
                match self.output_items.last_mut() {
                    Some(OutputItem::Text { content }) => {
                        // Append to existing text item
                        content.push_str(&delta);
                    }
                    _ => {
                        // No text item to append to - this shouldn't happen if OutputItemAdded events are properly sent
                        // But for robustness, create a new text item
                        self.output_items.push(OutputItem::Text { content: delta });
                    }
                }
            }
            StreamEvent::OutputItemAdded { item } => {
                // When a new output item is added, create the appropriate empty item
                match item {
                    crate::types::OutputItemInfo::Text => {
                        // Add an empty text item that will be filled by subsequent ContentDelta events
                        self.output_items.push(OutputItem::Text { content: String::new() });
                    }
                    crate::types::OutputItemInfo::FunctionCall { .. } => {
                        // Function call items will be replaced when FunctionCallComplete arrives
                        // We don't add a placeholder here since we handle it in FunctionCallComplete
                    }
                }
            }
            StreamEvent::FunctionCallComplete { call } => {
                // Add the complete function call as an output item
                self.output_items.push(OutputItem::FunctionCall { call });
            }
            StreamEvent::Done { finish_reason, usage } => {
                self.finish_reason = Some(finish_reason);
                self.usage = Some(usage);
            }
            StreamEvent::Error { .. } => {
                // Handle error events if needed
            }
        }
        
        Ok(())
    }
    
    /// Finalize and return the complete response.
    pub fn finalize(self) -> Result<CompleteResponse, Error> {
        Ok(CompleteResponse {
            output: self.output_items,
            finish_reason: self.finish_reason.unwrap_or(FinishReason::Stop),
            usage: self.usage.unwrap_or_default(),
        })
    }
    
    /// Get the current accumulated content (concatenated text only).
    /// This is a convenience method for accessing content during streaming.
    pub fn current_content(&self) -> String {
        let mut content = String::new();
        
        // Add text from all text output items
        for item in &self.output_items {
            if let OutputItem::Text { content: text } = item {
                content.push_str(text);
            }
        }
        
        content
    }
    
    /// Get the completed function calls so far.
    /// This is a convenience method for accessing function calls during streaming.
    pub fn completed_function_calls(&self) -> Vec<FunctionCall> {
        self.output_items.iter()
            .filter_map(|item| match item {
                OutputItem::FunctionCall { call } => Some(call.clone()),
                _ => None,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_content_accumulation() {
        let mut accumulator = ResponseAccumulator::new();
        
        // Without OutputItemAdded, content should still work (for backward compatibility)
        let event1 = StreamEvent::ContentDelta { delta: "Hello ".to_string() };
        accumulator.process_event(event1).unwrap();
        assert_eq!(accumulator.current_content(), "Hello ");
        
        let event2 = StreamEvent::ContentDelta { delta: "world!".to_string() };
        accumulator.process_event(event2).unwrap();
        assert_eq!(accumulator.current_content(), "Hello world!");
    }
    
    #[test]
    fn test_output_item_based_accumulation() {
        let mut accumulator = ResponseAccumulator::new();
        
        // Add first text output item
        let add_text1 = StreamEvent::OutputItemAdded { 
            item: crate::types::OutputItemInfo::Text,
        };
        accumulator.process_event(add_text1).unwrap();
        
        // Add content to first text item
        let delta1 = StreamEvent::ContentDelta { delta: "First item. ".to_string() };
        accumulator.process_event(delta1).unwrap();
        assert_eq!(accumulator.current_content(), "First item. ");
        
        // Add function call output item
        let add_func = StreamEvent::OutputItemAdded { 
            item: crate::types::OutputItemInfo::FunctionCall {
                name: "test_func".to_string(),
                id: "fc_123".to_string(),
            },
        };
        accumulator.process_event(add_func).unwrap();
        
        // Complete the function call
        let func_complete = StreamEvent::FunctionCallComplete {
            call: FunctionCall {
                id: "fc_123".to_string(),
                call_id: "call_123".to_string(),
                name: "test_func".to_string(),
                arguments: "{}".to_string(),
            },
        };
        accumulator.process_event(func_complete).unwrap();
        
        // Add second text output item
        let add_text2 = StreamEvent::OutputItemAdded { 
            item: crate::types::OutputItemInfo::Text,
        };
        accumulator.process_event(add_text2).unwrap();
        
        // Add content to second text item
        let delta2 = StreamEvent::ContentDelta { delta: "Second item.".to_string() };
        accumulator.process_event(delta2).unwrap();
        
        // Verify content accumulation
        assert_eq!(accumulator.current_content(), "First item. Second item.");
        
        // Verify output structure
        let response = accumulator.finalize().unwrap();
        assert_eq!(response.output.len(), 3);
        
        match &response.output[0] {
            OutputItem::Text { content } => assert_eq!(content, "First item. "),
            _ => panic!("Expected text item"),
        }
        
        match &response.output[1] {
            OutputItem::FunctionCall { call } => assert_eq!(call.name, "test_func"),
            _ => panic!("Expected function call"),
        }
        
        match &response.output[2] {
            OutputItem::Text { content } => assert_eq!(content, "Second item."),
            _ => panic!("Expected text item"),
        }
    }
    
    #[test]
    fn test_function_call_accumulation() {
        let mut accumulator = ResponseAccumulator::new();
        
        // Complete function call
        let complete_event = StreamEvent::FunctionCallComplete {
            call: FunctionCall {
                id: "fc_123".to_string(),
                call_id: "call_123".to_string(),
                name: "get_weather".to_string(),
                arguments: "{\"location\": \"Paris\"}".to_string(),
            },
        };
        accumulator.process_event(complete_event).unwrap();
        
        assert_eq!(accumulator.completed_function_calls().len(), 1);
        assert_eq!(accumulator.completed_function_calls()[0].name, "get_weather");
    }
    
    #[test]
    fn test_finalization() {
        let mut accumulator = ResponseAccumulator::new();
        
        // Add some content
        let content_event = StreamEvent::ContentDelta { delta: "Test response".to_string() };
        accumulator.process_event(content_event).unwrap();
        
        // Add done event
        let done_event = StreamEvent::Done {
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        };
        accumulator.process_event(done_event).unwrap();
        
        // Finalize
        let complete = accumulator.finalize().unwrap();
        assert_eq!(complete.content(), "Test response");
        assert_eq!(complete.finish_reason, FinishReason::Stop);
    }
    
    #[test]
    fn test_to_assistant_message() {
        use crate::types::Role;
        
        let mut accumulator = ResponseAccumulator::new();
        
        // Add some content
        let content_event = StreamEvent::ContentDelta { delta: "Hello world".to_string() };
        accumulator.process_event(content_event).unwrap();
        
        // Add a function call
        let function_call = StreamEvent::FunctionCallComplete {
            call: FunctionCall {
                id: "fc_123".to_string(),
                call_id: "call_123".to_string(),
                name: "test_function".to_string(),
                arguments: "{}".to_string(),
            },
        };
        accumulator.process_event(function_call).unwrap();
        
        // Finalize and test the response
        let response = accumulator.finalize().unwrap();
        assert_eq!(response.content(), "Hello world");
        
        // Test the proper way to get all items
        let items = response.to_items();
        assert_eq!(items.len(), 2); // One text message, one function call
        
        // First item should be text
        match &items[0] {
            crate::types::InputItem::Message(msg) => {
                assert_eq!(msg.role(), Role::Assistant);
                assert_eq!(msg.content(), Some("Hello world".to_string()));
            }
            _ => panic!("Expected message"),
        }
        
        // Second item should be function call
        match &items[1] {
            crate::types::InputItem::FunctionCall(call) => {
                assert_eq!(call.name, "test_function");
            }
            _ => panic!("Expected function call"),
        }
    }
    
    #[test]
    fn test_to_items() {
        use crate::types::Role;
        
        let mut accumulator = ResponseAccumulator::new();
        
        // Add some content
        let content_event = StreamEvent::ContentDelta { delta: "Hello world".to_string() };
        accumulator.process_event(content_event).unwrap();
        
        // Add a function call
        let function_call = StreamEvent::FunctionCallComplete {
            call: FunctionCall {
                id: "fc_123".to_string(),
                call_id: "call_123".to_string(),
                name: "test_function".to_string(),
                arguments: "{}".to_string(),
            },
        };
        accumulator.process_event(function_call).unwrap();
        
        // Finalize and convert to items
        let response = accumulator.finalize().unwrap();
        let items = response.to_items();
        assert_eq!(items.len(), 2);
        
        // Check content types
        match &items[0] {
            crate::types::InputItem::Message(msg) => {
                assert_eq!(msg.role(), Role::Assistant);
                assert_eq!(msg.content(), Some("Hello world".to_string()));
            }
            _ => panic!("Expected message"),
        }
        
        match &items[1] {
            crate::types::InputItem::FunctionCall(call) => {
                assert_eq!(call.name, "test_function");
            }
            _ => panic!("Expected function call"),
        }
    }
}