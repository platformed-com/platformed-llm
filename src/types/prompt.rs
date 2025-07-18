use super::message::InputItem;

/// A structured prompt containing a sequence of input items.
#[derive(Debug, Clone)]
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
            items: vec![InputItem::system(content.into())]
        }
    }
    
    /// Create a prompt with a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            items: vec![InputItem::user(content.into())]
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