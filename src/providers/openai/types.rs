use crate::types::Usage;
use serde::{Deserialize, Serialize};

/// OpenAI input message format for Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OpenAIInputMessage {
    /// Regular message with role and content
    #[serde(rename = "message")]
    Regular { role: String, content: String },
    /// Function call output message
    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: String },
    /// Function call message (when sending previous function calls back)
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
}

/// OpenAI tool format for Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAITool {
    pub r#type: String, // "function"
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// OpenAI Responses API request.
#[derive(Debug, Clone, Serialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: Vec<OpenAIInputMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
}

/// OpenAI Responses API response.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // Will be used in non-streaming mode later
pub struct ResponsesResponse {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub status: String,
    pub model: String,
    pub output: Vec<ResponseOutput>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
}

/// Output item in a Responses API response.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // Will be used in non-streaming mode later
pub struct ResponseOutput {
    pub r#type: String, // "message" or "function_call"
    pub id: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ResponseContent>>,
    // For function call outputs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
}

/// Content item in a Responses API output.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // Will be used in non-streaming mode later
pub struct ResponseContent {
    pub r#type: String, // "output_text", "tool_call", etc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<serde_json::Value>>,
    // For tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// OpenAI error response.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // For error handling and debugging
pub struct OpenAIError {
    pub error: ErrorDetails,
}

/// Error details from OpenAI API.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // For error handling and debugging
pub struct ErrorDetails {
    pub message: String,
    pub r#type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

/// OpenAI streaming Responses API event.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // Some fields used for metadata
pub struct ResponsesStreamEvent {
    pub r#type: String, // Event type
    pub sequence_number: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<ResponsesResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item: Option<ResponseItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub part: Option<ResponseContent>,
}

/// Function call item in streaming response.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ResponseItem {
    pub id: String,
    pub r#type: String, // "function_call"
    pub status: String, // "in_progress", "completed"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Tool call delta in a streaming response.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // Some fields used for metadata
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

/// Function call delta in a streaming response.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // Used for completeness of API response structure
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}
