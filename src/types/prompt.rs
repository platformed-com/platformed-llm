use serde::Serialize;

use super::message::{FunctionCall, InputItem};

/// A structured prompt containing a sequence of input items.
#[derive(Debug, Clone, Serialize)]
pub struct Prompt {
    items: Vec<InputItem>,
}

impl Prompt {
    /// Build an empty prompt.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Start a prompt with a single system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            items: vec![InputItem::system(content)],
        }
    }

    /// Start a prompt with a single user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            items: vec![InputItem::user(content)],
        }
    }

    /// Append a system message.
    pub fn with_system(mut self, content: impl Into<String>) -> Self {
        self.items.push(InputItem::system(content));
        self
    }

    /// Append a user message.
    pub fn with_user(mut self, content: impl Into<String>) -> Self {
        self.items.push(InputItem::user(content));
        self
    }

    /// Append a plain-text assistant message (e.g. a previous reply).
    pub fn with_assistant(mut self, content: impl Into<String>) -> Self {
        self.items.push(InputItem::assistant(content));
        self
    }

    /// Append a pre-built [`InputItem`] verbatim.
    pub fn with_item(mut self, item: InputItem) -> Self {
        self.items.push(item);
        self
    }

    /// Append a sequence of pre-built [`InputItem`]s verbatim.
    pub fn with_items(mut self, items: Vec<InputItem>) -> Self {
        self.items.extend(items);
        self
    }

    /// Append a completed assistant response (text, tool calls,
    /// continuation marker — everything preserved).
    pub fn with_response(mut self, response: &crate::response::CompleteResponse) -> Self {
        self.items.extend(response.to_items());
        self
    }

    /// Append a tool result for a previously-emitted assistant tool call.
    pub fn with_tool_result(
        mut self,
        call_id: impl Into<String>,
        output: impl Into<String>,
    ) -> Self {
        self.items.push(InputItem::tool_result(call_id, output));
        self
    }

    /// Append an assistant turn whose only content is a single tool call —
    /// useful when manually reconstructing conversation history.
    pub fn with_assistant_tool_call(mut self, call: FunctionCall) -> Self {
        self.items.push(InputItem::assistant_tool_call(call));
        self
    }

    /// Borrow the accumulated items.
    pub fn items(&self) -> &[InputItem] {
        &self.items
    }

    /// Consume the prompt and return its underlying items.
    pub fn into_items(self) -> Vec<InputItem> {
        self.items
    }
}

impl Default for Prompt {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&Prompt> for Prompt {
    fn from(p: &Prompt) -> Self {
        p.clone()
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
    use crate::response::CompleteResponse;
    use crate::types::{AssistantPart, FinishReason, Usage};

    #[test]
    fn builder_stacks_items_in_order() {
        let prompt = Prompt::system("be helpful").with_user("hi");
        assert_eq!(prompt.items().len(), 2);
        assert!(matches!(prompt.items()[0], InputItem::System(_)));
        assert!(matches!(prompt.items()[1], InputItem::User { .. }));
    }

    #[test]
    fn from_str_creates_single_user_item() {
        let p: Prompt = "hello".into();
        assert_eq!(p.items().len(), 1);
        assert!(matches!(p.items()[0], InputItem::User { .. }));
    }

    #[test]
    fn with_response_appends_assistant_turn() {
        let prompt = Prompt::system("be helpful").with_user("hi");
        let response = CompleteResponse {
            content: vec![AssistantPart::Text {
                content: "AI response".to_string(),
                annotations: Vec::new(),
            }],
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        };
        let extended = prompt.with_response(&response);
        assert_eq!(extended.items().len(), 3);
    }
}
