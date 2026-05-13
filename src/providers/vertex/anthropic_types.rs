use std::borrow::Cow;

use crate::types::Usage;
use ijson::IValue;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

/// Anthropic Claude request format via Vertex AI.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicRequest {
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    pub anthropic_version: String, // Always "vertex-2023-10-16"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinking>,
    /// Stop sequences. Anthropic does not expose presence/frequency
    /// penalties, only stops.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// Forced tool mode. Anthropic accepts `auto`, `any`, `tool` (with
    /// `name`), or `none`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolChoice {
    Auto,
    Any,
    Tool { name: String },
    None,
}

/// Anthropic extended-thinking config.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicThinking {
    Enabled { budget_tokens: u32 },
}

/// Anthropic message format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String, // "user" or "assistant"
    pub content: AnthropicContent,
}

/// Anthropic content can be string or array of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

/// Anthropic content block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        /// Anthropic prompt-caching hint. Marks the prefix up to this
        /// block as cacheable. Up to 4 breakpoints per request.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: IValue,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: AnthropicToolResultContent,
        /// Set to `true` to signal the tool failed. The model uses this to
        /// distinguish error returns from success returns.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    /// Extended-thinking block. The model's chain-of-thought reasoning;
    /// `signature` must be echoed back unchanged in subsequent turns to
    /// preserve thinking continuity.
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Encrypted thinking that's been classified as sensitive. Pass-through
    /// only — opaque blob to be echoed back unchanged.
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    /// Image content block (request side).
    #[serde(rename = "image")]
    Image {
        source: IValue,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
}

/// Anthropic cache-control hint on a content block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicCacheControl {
    #[serde(rename = "type")]
    pub r#type: String,
}

/// `tool_result.content` accepts either a plain string or an array of
/// content blocks (the latter is used when a tool returns multi-modal
/// content — e.g. an image alongside text).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicToolResultContent {
    Text(String),
    Blocks(Vec<AnthropicToolResultBlock>),
}

/// Block forms acceptable inside a `tool_result.content` array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicToolResultBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: IValue },
}

/// Anthropic tool definition.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicTool {
    pub name: String,
    pub description: String,
    pub input_schema: Cow<'static, RawValue>,
}

/// Anthropic API response.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    pub model: String,
    pub role: String, // Always "assistant"
    pub content: Vec<AnthropicContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<AnthropicUsage>,
}

/// Anthropic usage information.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

/// Anthropic streaming events.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicResponse },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: u32,
        content_block: AnthropicContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: u32,
        delta: AnthropicContentDelta,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: u32 },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: AnthropicMessageDelta,
        // On the wire, `usage` is a top-level sibling of `delta` on
        // `message_delta` events — NOT nested inside the delta. Decoding it as
        // a sibling here means the cumulative output_tokens reported by
        // Anthropic actually reaches our state machine.
        #[serde(default)]
        usage: Option<AnthropicUsage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
    /// Operational error delivered mid-stream (e.g. `overloaded_error`,
    /// `api_error`). Surfaces as a stream `Err` rather than being parsed as
    /// a "failed to parse SSE event".
    #[serde(rename = "error")]
    Error { error: AnthropicErrorPayload },
}

/// Payload of a mid-stream `event: error` frame.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicErrorPayload {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

/// Delta for content blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicContentDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    /// Incremental update to a `thinking` content block.
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    /// Signature appended to a `thinking` block at end-of-block. Required to
    /// be echoed back unchanged in subsequent turns.
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
}

/// Delta for message-level changes carried by a `message_delta` event.
///
/// Note: `usage` is intentionally NOT a field here. Per Anthropic's wire
/// format, the `usage` object on `message_delta` is a sibling of `delta`, not
/// nested inside it — the parent enum variant carries it.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessageDelta {
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub stop_sequence: Option<String>,
}

impl From<AnthropicUsage> for Usage {
    fn from(usage: AnthropicUsage) -> Self {
        Usage {
            input_tokens: usage.input_tokens.unwrap_or(0),
            output_tokens: usage.output_tokens.unwrap_or(0),
            cache_read_input_tokens: usage.cache_read_input_tokens,
            cache_creation_input_tokens: usage.cache_creation_input_tokens,
            reasoning_tokens: None,
        }
    }
}
