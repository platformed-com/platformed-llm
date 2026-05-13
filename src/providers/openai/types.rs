use std::borrow::Cow;

use crate::types::Usage;
use ijson::IValue;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

/// OpenAI input message format for Responses API.
///
/// The `Regular` variant supports both a bare string content (the
/// common text-only case) and an array of content parts (used when any
/// part is non-text — images, files, etc.) — distinguished by
/// `OpenAIMessageContent`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OpenAIInputMessage {
    /// Regular message with role and content.
    #[serde(rename = "message")]
    Regular {
        role: String,
        content: OpenAIMessageContent,
    },
    /// Function call output message.
    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: String },
    /// Function call message (when sending previous function calls back).
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
}

/// OpenAI message content: either a bare string (text-only) or an
/// explicit array of content parts (mixed text + images / files /
/// audio).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIMessageContent {
    Text(String),
    Parts(Vec<OpenAIContentPart>),
}

/// Tagged content part within an OpenAI message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OpenAIContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "input_image")]
    InputImage {
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
    },
    #[serde(rename = "input_audio")]
    InputAudio { input_audio: OpenAIInputAudio },
    #[serde(rename = "input_file")]
    InputFile {
        #[serde(skip_serializing_if = "Option::is_none")]
        file_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
}

/// OpenAI inline audio data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIInputAudio {
    pub data: String,
    pub format: String,
}

/// OpenAI tool entry in the Responses API `tools` array.
///
/// `Function` is the caller-defined case; the other variants are
/// OpenAI's pre-baked builtin tools. Each builtin has its own wire
/// shape — `web_search_preview` is a bare type marker, while
/// `computer_use_preview` takes display dimensions and an environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAITool {
    Function {
        name: String,
        description: String,
        parameters: Cow<'static, RawValue>,
    },
    WebSearchPreview,
    ComputerUsePreview {
        display_width: u32,
        display_height: u32,
        environment: String,
    },
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
    pub tool_choice: Option<OpenAIToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<OpenAIReasoning>,
    /// Stop sequences. OpenAI Responses API does not support `stop` on
    /// reasoning models; for non-reasoning models it accepts up to 4
    /// strings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Presence / frequency penalty (non-reasoning models only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Stable caller-side cache key. OpenAI groups requests that share
    /// the same key for prefix-cache lookup. We derive it from a hash
    /// of the message prefix before the first
    /// [`crate::UserPart::CacheBreakpoint`] so two requests with the
    /// same prefix produce the same key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// `text.format` block — JSON mode / JSON schema constraint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<OpenAITextConfig>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAITextConfig {
    pub format: OpenAITextFormat,
}

/// `text.format` wire shape on OpenAI Responses API.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAITextFormat {
    /// Bare JSON mode — model returns valid JSON without schema enforcement.
    JsonObject,
    /// Strict JSON Schema. Carries the schema inline.
    JsonSchema {
        name: String,
        schema: Cow<'static, RawValue>,
        strict: bool,
    },
}

/// OpenAI's `reasoning` request field for chain-of-thought models.
#[derive(Debug, Clone, Serialize)]
pub struct OpenAIReasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<&'static str>,
}

/// OpenAI's `tool_choice` accepts either a string mode or a typed object
/// pointing at a specific tool.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum OpenAIToolChoice {
    /// `"auto"`, `"none"`, or `"required"`.
    Mode(&'static str),
    /// `{"type":"function","name":"..."}`.
    Function {
        #[serde(rename = "type")]
        kind: &'static str,
        name: String,
    },
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
    pub output: Vec<ResponseItem>,
    pub usage: Option<Usage>,
    /// Populated by `response.incomplete` events with `{ reason: ... }`
    /// where `reason` is one of `max_output_tokens`, `content_filter`,
    /// or similar. We map this onto `FinishReason::Length` /
    /// `ContentFilter` so callers see a sensible terminal state.
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetails>,
    /// Populated by `response.failed` events with details on why the
    /// model run could not complete (rate limits, server error, etc.).
    /// Mapped onto `FinishReason::ContentFilter` only when the failure
    /// was an explicit safety/policy block; otherwise treated as a
    /// streaming-level error.
    #[serde(default)]
    pub error: Option<ErrorDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
}

/// Output item in a Responses API response.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // Will be used in non-streaming mode later
pub struct ResponseItem {
    pub r#type: String, // "message" or "function_call"
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
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
    pub annotations: Option<Vec<IValue>>,
}

/// Error details from OpenAI API.
#[derive(Debug, Clone, Deserialize)]
pub struct ErrorDetails {
    pub message: String,
    pub r#type: String,
    #[allow(unused)]
    pub param: Option<String>,
    #[allow(unused)]
    pub code: Option<String>,
}

/// `response.incomplete_details` payload — the model didn't run to
/// completion. `reason` is `"max_output_tokens"`, `"content_filter"`, or
/// (rarely) something else; treat unknown values as `Stop` so the
/// terminal event still fires.
#[derive(Debug, Clone, Deserialize)]
pub struct IncompleteDetails {
    pub reason: String,
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
    pub logprobs: Option<Vec<IValue>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<ResponsesResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item: Option<ResponseItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub part: Option<ResponseContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorDetails>,
}

