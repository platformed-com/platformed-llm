use serde::Serialize;

use super::message::InputItem;

/// A structured prompt containing a sequence of input items.
#[derive(Debug, Clone, Serialize)]
pub struct Prompt {
    items: Vec<InputItem>,
}

impl Prompt {
    /// Create a new empty prompt.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Create a prompt with a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            items: vec![InputItem::system(content.into())],
        }
    }

    /// Create a prompt with a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            items: vec![InputItem::user(content.into())],
        }
    }

    /// Add a system message.
    pub fn with_system(mut self, content: impl Into<String>) -> Self {
        self.items.push(InputItem::system(content.into()));
        self
    }

    /// Add a user message.
    pub fn with_user(mut self, content: impl Into<String>) -> Self {
        self.items.push(InputItem::user(content.into()));
        self
    }

    /// Add an assistant message.
    pub fn with_assistant(mut self, content: impl Into<String>) -> Self {
        self.items.push(InputItem::assistant(content.into()));
        self
    }

    /// Add an input item.
    pub fn with_item(mut self, item: InputItem) -> Self {
        self.items.push(item);
        self
    }

    /// Add multiple input items.
    pub fn with_items(mut self, items: Vec<InputItem>) -> Self {
        self.items.extend(items);
        self
    }

    /// Add a response to the conversation.
    /// This converts the response to a sequence of input items, preserving the ordering of text and function calls.
    pub fn with_response(mut self, response: &crate::response::CompleteResponse) -> Self {
        self.items.extend(response.to_items());
        self
    }

    /// Get the input items.
    pub fn items(&self) -> &[InputItem] {
        &self.items
    }
}

impl Default for Prompt {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&str> for Prompt {
    fn from(s: &str) -> Self {
        Prompt::user(s)
    }
}

impl From<String> for Prompt {
    fn from(s: String) -> Self {
        Prompt::user(s)
    }
}

impl From<InputItem> for Prompt {
    fn from(item: InputItem) -> Self {
        Prompt { items: vec![item] }
    }
}

impl From<Vec<InputItem>> for Prompt {
    fn from(items: Vec<InputItem>) -> Self {
        Prompt { items }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::response::{CompleteResponse, OutputItem};
    use crate::types::{FinishReason, FunctionCall, Message, Role, Usage};

    #[test]
    fn builder_stacks_items_in_order() {
        let prompt = Prompt::system("you are a helpful assistant")
            .with_user("what is the capital of france?");
        assert_eq!(prompt.items().len(), 2);

        let from_str: Prompt = "hello".into();
        assert_eq!(from_str.items().len(), 1);

        let from_string: Prompt = "hello".to_string().into();
        assert_eq!(from_string.items().len(), 1);
    }

    #[test]
    fn with_response_appends_assistant_message() {
        let prompt =
            Prompt::system("you are a helpful assistant").with_user("what is the capital?");
        let response = CompleteResponse {
            output: vec![OutputItem::Text {
                content: "AI response".to_string(),
            }],
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        };
        let extended = prompt.with_response(&response);
        assert_eq!(extended.items().len(), 3);
        match &extended.items()[2] {
            InputItem::Message(msg) => {
                assert_eq!(msg.role(), Role::Assistant);
                assert_eq!(msg.content(), Some("AI response".to_string()));
            }
            _ => panic!("expected message"),
        }
    }

    #[test]
    fn with_items_appends_each_in_order() {
        let prompt = Prompt::system("you are a helpful assistant").with_user("hello world");
        let extended = prompt.with_items(vec![
            InputItem::Message(Message::assistant("first assistant message")),
            InputItem::FunctionCall(FunctionCall {
                call_id: "call_123".to_string(),
                name: "test_function".to_string(),
                arguments: "{}".to_string(),
            }),
            InputItem::Message(Message::assistant("second assistant message")),
        ]);
        assert_eq!(extended.items().len(), 5);

        let added = &extended.items()[2..];
        match &added[0] {
            InputItem::Message(msg) => {
                assert_eq!(msg.role(), Role::Assistant);
                assert_eq!(msg.content(), Some("first assistant message".to_string()));
            }
            _ => panic!("expected message"),
        }
        match &added[1] {
            InputItem::FunctionCall(call) => assert_eq!(call.name, "test_function"),
            _ => panic!("expected function call"),
        }
        match &added[2] {
            InputItem::Message(msg) => {
                assert_eq!(msg.content(), Some("second assistant message".to_string()))
            }
            _ => panic!("expected message"),
        }
    }
}
