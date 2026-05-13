use serde::Serialize;

use super::message::{FunctionCall, InputItem};

/// A structured prompt containing a sequence of input items.
#[derive(Debug, Clone, Serialize)]
pub struct Prompt {
    items: Vec<InputItem>,
}

impl Prompt {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            items: vec![InputItem::system(content)],
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            items: vec![InputItem::user(content)],
        }
    }

    pub fn with_system(mut self, content: impl Into<String>) -> Self {
        self.items.push(InputItem::system(content));
        self
    }

    pub fn with_user(mut self, content: impl Into<String>) -> Self {
        self.items.push(InputItem::user(content));
        self
    }

    pub fn with_assistant(mut self, content: impl Into<String>) -> Self {
        self.items.push(InputItem::assistant(content));
        self
    }

    pub fn with_item(mut self, item: InputItem) -> Self {
        self.items.push(item);
        self
    }

    pub fn with_items(mut self, items: Vec<InputItem>) -> Self {
        self.items.extend(items);
        self
    }

    pub fn with_response(mut self, response: &crate::response::CompleteResponse) -> Self {
        self.items.extend(response.to_items());
        self
    }

    pub fn with_tool_result(
        mut self,
        call_id: impl Into<String>,
        output: impl Into<String>,
    ) -> Self {
        self.items.push(InputItem::tool_result(call_id, output));
        self
    }

    pub fn with_assistant_tool_call(mut self, call: FunctionCall) -> Self {
        self.items.push(InputItem::assistant_tool_call(call));
        self
    }

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
            output: vec![OutputItem::Assistant {
                content: vec![AssistantPart::Text {
                    content: "AI response".to_string(),
                    annotations: Vec::new(),
                }],
            }],
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
            continuation: None,
        };
        let extended = prompt.with_response(&response);
        assert_eq!(extended.items().len(), 3);
    }
}
