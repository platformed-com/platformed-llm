use serde::{Deserialize, Serialize};

/// An input item in a conversation (mirrors OutputItem).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputItem {
    /// A message with role and content
    Message(Message),
    /// A function call
    FunctionCall(FunctionCall),
    /// Output from a function call
    FunctionCallOutput { call_id: String, output: String },
}

/// A message with role and content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}



impl InputItem {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        InputItem::Message(Message {
            role: Role::System,
            content: content.into(),
        })
    }
    
    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        InputItem::Message(Message {
            role: Role::User,
            content: content.into(),
        })
    }
    
    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        InputItem::Message(Message {
            role: Role::Assistant,
            content: content.into(),
        })
    }
    
    /// Create a function call item.
    pub fn function_call(call: FunctionCall) -> Self {
        InputItem::FunctionCall(call)
    }
    
    /// Create a function call output item.
    pub fn function_call_output(call_id: String, output: String) -> Self {
        InputItem::FunctionCallOutput { call_id, output }
    }
    
    /// Get the role of this item (if it's a message).
    pub fn role(&self) -> Option<Role> {
        match self {
            InputItem::Message(msg) => Some(msg.role),
            _ => None,
        }
    }
    
    /// Get the text content of this item (if any).
    pub fn content(&self) -> Option<String> {
        match self {
            InputItem::Message(msg) => {
                if msg.content.is_empty() {
                    None
                } else {
                    Some(msg.content.clone())
                }
            },
            InputItem::FunctionCallOutput { output, .. } => Some(output.clone()),
            InputItem::FunctionCall(_) => None,
        }
    }
    
    /// Get the function call from this item (if any).
    pub fn get_function_call(&self) -> Option<&FunctionCall> {
        match self {
            InputItem::FunctionCall(call) => Some(call),
            _ => None,
        }
    }
    
    /// Get the function call ID from this item (if any).
    pub fn function_call_id(&self) -> Option<&str> {
        match self {
            InputItem::FunctionCallOutput { call_id, .. } => Some(call_id),
            _ => None,
        }
    }
    
}

impl Message {
    /// Create a new message with role and text content.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Message {
            role,
            content: content.into(),
        }
    }
    
    /// Add text content to this message.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        if !self.content.is_empty() {
            self.content.push(' ');
        }
        self.content.push_str(&text.into());
        self
    }
    
    /// Get all text content.
    pub fn text_content(&self) -> String {
        self.content.clone()
    }
    
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Message {
            role: Role::System,
            content: content.into(),
        }
    }
    
    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Message {
            role: Role::User,
            content: content.into(),
        }
    }
    
    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Message {
            role: Role::Assistant,
            content: content.into(),
        }
    }
    
    /// Get the role of this message.
    pub fn role(&self) -> Role {
        self.role
    }
    
    /// Get the text content of this message (if any).
    pub fn content(&self) -> Option<String> {
        if self.content.is_empty() {
            None
        } else {
            Some(self.content.clone())
        }
    }
}


/// Role of a message participant.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Tool definition for function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub r#type: ToolType,
    pub function: Function,
}

/// Type of tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    Function,
}

/// Function definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON Schema
}

/// Function call information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub id: String,
    pub call_id: String, // The call_id used for function results
    pub name: String,
    pub arguments: String, // JSON string
}

/// Reason why generation finished.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}