use std::borrow::Cow;

use crate::types::Usage;
use ijson::IValue;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

/// Google Vertex AI request format (for Gemini models).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleRequest {
    pub contents: Vec<GoogleContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GoogleGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GoogleTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GoogleContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_config: Option<GoogleToolConfig>,
    /// Reference to a pre-created `CachedContent` resource on Vertex.
    /// When set, Vertex looks up the cached prefix and elides the
    /// message history that produced it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content: Option<String>,
}

/// Gemini `toolConfig`. Forces or disables tool calling per request.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleToolConfig {
    pub function_calling_config: GoogleFunctionCallingConfig,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleFunctionCallingConfig {
    /// `AUTO` (default, model decides), `ANY` (model must call a tool),
    /// `NONE` (no tool calls).
    pub mode: &'static str,
    /// When `mode == "ANY"`, restricts to a specific allowed set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_function_names: Option<Vec<String>>,
}

/// Google content (message) format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleContent {
    pub role: String, // "user", "model"
    /// Vertex sometimes returns a candidate with `content: {role: model}`
    /// and no `parts` (e.g. when a candidate is safety-blocked mid-stream).
    /// Default to empty so we don't fail to parse those frames.
    #[serde(default)]
    pub parts: Vec<GooglePart>,
}

/// Part of a Google content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GooglePart {
    Text {
        text: String,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GoogleFunctionCall,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GoogleFunctionResponse,
    },
    /// Inline binary content (base64) — images, audio, video.
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: GoogleInlineData,
    },
    /// File reference by URI (Cloud Storage, etc.).
    FileData {
        #[serde(rename = "fileData")]
        file_data: GoogleFileData,
    },
    /// Code the model wrote to execute via the `codeExecution` builtin.
    ExecutableCode {
        #[serde(rename = "executableCode")]
        executable_code: GoogleExecutableCode,
    },
    /// Result of running `ExecutableCode`.
    CodeExecutionResult {
        #[serde(rename = "codeExecutionResult")]
        code_execution_result: GoogleCodeExecutionResult,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleExecutableCode {
    /// `"PYTHON"` is the only documented value today.
    pub language: String,
    pub code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleCodeExecutionResult {
    /// `"OUTCOME_OK"`, `"OUTCOME_FAILED"`, etc.
    pub outcome: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleInlineData {
    pub mime_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleFileData {
    pub mime_type: String,
    pub file_uri: String,
}

/// Google function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleFunctionCall {
    pub name: String,
    pub args: IValue,
}

/// Google function response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleFunctionResponse {
    pub name: String,
    pub response: IValue,
}

/// Google generation configuration.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Gemini 2.5 thinking budget. Mirrors `ReasoningConfig.effort` via
    /// rough mapping (Low → 2048, Medium → 8192, High → 16384).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_config: Option<GoogleThinkingConfig>,
    /// MIME type the model should constrain its response to. The
    /// canonical value for JSON output is `"application/json"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    /// Optional JSON Schema describing the response shape. Only
    /// meaningful when `response_mime_type` is `"application/json"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<Cow<'static, RawValue>>,
}

/// Gemini thinking config.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleThinkingConfig {
    pub thinking_budget: u32,
}

/// Google tool entry in the `tools` array.
///
/// Gemini distinguishes function tools from builtins by which key is
/// present on the entry. Serializes via `#[serde(untagged)]` so each
/// variant produces the wire shape Gemini expects.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum GoogleTool {
    /// Caller-defined function tools.
    Functions {
        #[serde(rename = "functionDeclarations")]
        function_declarations: Vec<GoogleFunctionDeclaration>,
    },
    /// Built-in Google Search retrieval.
    GoogleSearch {
        #[serde(rename = "googleSearch")]
        google_search: GoogleEmptyConfig,
    },
    /// Built-in code execution.
    CodeExecution {
        #[serde(rename = "codeExecution")]
        code_execution: GoogleEmptyConfig,
    },
}

/// Placeholder empty-object wire shape Gemini uses for parameterless
/// builtin tools. Always serializes to `{}`.
#[derive(Debug, Clone, Serialize, Default)]
pub struct GoogleEmptyConfig {}

/// Google function declaration.
#[derive(Debug, Clone, Serialize)]
pub struct GoogleFunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: Cow<'static, RawValue>,
}

/// Google API response.
///
/// `candidates` defaults to empty so we still parse safety-blocked-at-prompt
/// frames, which carry only `promptFeedback` and no candidates array.
#[derive(Debug, Clone, Deserialize)]
pub struct GoogleResponse {
    #[serde(default)]
    pub candidates: Vec<GoogleCandidate>,
    #[serde(default, rename = "usageMetadata")]
    pub usage_metadata: Option<GoogleUsageMetadata>,
    #[serde(default, rename = "promptFeedback")]
    pub prompt_feedback: Option<GooglePromptFeedback>,
}

/// Returned in place of (or alongside) candidates when the prompt itself was
/// blocked at the safety layer.
#[derive(Debug, Clone, Deserialize)]
pub struct GooglePromptFeedback {
    #[serde(default, rename = "blockReason")]
    pub block_reason: Option<String>,
    #[serde(default, rename = "blockReasonMessage")]
    pub block_reason_message: Option<String>,
}

/// Google response candidate.
#[derive(Debug, Clone, Deserialize)]
pub struct GoogleCandidate {
    pub content: GoogleContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "finishReason")]
    pub finish_reason: Option<String>,
    /// Grounding metadata attached when `googleSearch` (or other
    /// retrieval) builtin tools fire. Maps to per-span URL citations
    /// on the unified surface.
    #[serde(default, rename = "groundingMetadata")]
    pub grounding_metadata: Option<GoogleGroundingMetadata>,
}

/// `groundingMetadata` payload attached to a candidate.
///
/// Wire shape: a flat list of `groundingChunks` (the sources the model
/// drew from) plus a list of `groundingSupports` that map text spans
/// onto chunk indices.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleGroundingMetadata {
    #[serde(default)]
    pub grounding_chunks: Vec<GoogleGroundingChunk>,
    #[serde(default)]
    pub grounding_supports: Vec<GoogleGroundingSupport>,
}

/// A single grounding source. Only the `web` variant is documented
/// today; future tags (e.g. retrieval over uploaded files) deserialize
/// into the catch-all so unknown shapes don't fail the parse.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleGroundingChunk {
    #[serde(default)]
    pub web: Option<GoogleGroundingWebChunk>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleGroundingWebChunk {
    pub uri: String,
    #[serde(default)]
    pub title: Option<String>,
}

/// One text-span → chunks mapping. `segment.start_index` /
/// `segment.end_index` are byte offsets into the candidate's text.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleGroundingSupport {
    pub segment: GoogleGroundingSegment,
    #[serde(default)]
    pub grounding_chunk_indices: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleGroundingSegment {
    #[serde(default)]
    pub start_index: usize,
    #[serde(default)]
    pub end_index: usize,
}

/// Google usage metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct GoogleUsageMetadata {
    #[serde(default, rename = "promptTokenCount")]
    pub prompt_token_count: Option<u32>,
    #[serde(default, rename = "candidatesTokenCount")]
    pub candidates_token_count: Option<u32>,
    #[serde(default, rename = "totalTokenCount")]
    pub total_token_count: Option<u32>,
    /// Output tokens used for the model's internal reasoning, on Gemini 2.5
    /// thinking models.
    #[serde(default, rename = "thoughtsTokenCount")]
    pub thoughts_token_count: Option<u32>,
    /// Tokens served from the prompt cache.
    #[serde(default, rename = "cachedContentTokenCount")]
    pub cached_content_token_count: Option<u32>,
}

impl From<GoogleUsageMetadata> for Usage {
    fn from(metadata: GoogleUsageMetadata) -> Self {
        Usage {
            input_tokens: metadata.prompt_token_count.unwrap_or(0),
            output_tokens: metadata.candidates_token_count.unwrap_or(0),
            cache_read_input_tokens: metadata.cached_content_token_count,
            cache_creation_input_tokens: None,
            reasoning_tokens: metadata.thoughts_token_count,
        }
    }
}
