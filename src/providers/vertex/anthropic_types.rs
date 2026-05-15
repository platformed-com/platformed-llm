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
    pub anthropic_version: &'static str, // Always "vertex-2023-10-16"
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
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        text: String,
        /// Anthropic prompt-caching hint. Marks the prefix up to this
        /// block as cacheable. Up to 4 breakpoints per request.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: IValue,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
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
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Encrypted thinking that's been classified as sensitive. Pass-through
    /// only — opaque blob to be echoed back unchanged.
    RedactedThinking { data: String },
    /// Image content block (request side).
    Image {
        source: IValue,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    /// Document (PDF) content block (request side).
    Document {
        source: IValue,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
}

/// Anthropic cache-control hint on a content block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicCacheControl {
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
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolResultBlock {
    Text { text: String },
    Image { source: IValue },
}

/// Anthropic tool entry. Function tools serialize without a `type`
/// field (Anthropic infers it from the absence of a typed type); the
/// builtin variants carry typed `type` markers like
/// `"web_search_20250305"`.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum AnthropicTool {
    Function {
        name: String,
        description: String,
        input_schema: Cow<'static, RawValue>,
    },
    /// Parameterless builtin (`web_search_20250305`).
    Builtin {
        r#type: &'static str,
        name: &'static str,
    },
    /// Anthropic `computer_20250124` — display dimensions + environment.
    Computer {
        r#type: &'static str,
        name: &'static str,
        display_width_px: u32,
        display_height_px: u32,
    },
}

/// Anthropic API response shell as it arrives on `message_start`.
/// Only [`Self::usage`] is consumed today; other top-level fields
/// (`id`, `model`, `role`, `content`, `stop_reason`) are present on
/// the wire but stripped by serde since the streaming converter
/// reconstructs them from the per-block events.
// Deserialize-only: `skip_serializing_if` would be dead here.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicResponse {
    /// Initial usage snapshot — Anthropic reports `input_tokens` here
    /// and accumulates `output_tokens` via `message_delta` events.
    pub usage: Option<AnthropicUsage>,
}

/// Anthropic usage information.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub cache_creation_input_tokens: Option<u32>,
    pub cache_read_input_tokens: Option<u32>,
}

/// Anthropic streaming events.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicStreamEvent {
    MessageStart {
        message: AnthropicResponse,
    },
    ContentBlockStart {
        index: u32,
        content_block: AnthropicContentBlock,
    },
    ContentBlockDelta {
        index: u32,
        delta: AnthropicContentDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: AnthropicMessageDelta,
        // On the wire, `usage` is a top-level sibling of `delta` on
        // `message_delta` events — NOT nested inside the delta. Decoding it as
        // a sibling here means the cumulative output_tokens reported by
        // Anthropic actually reaches our state machine.
        #[serde(default)]
        usage: Option<AnthropicUsage>,
    },
    MessageStop,
    Ping,
    /// Operational error delivered mid-stream (e.g. `overloaded_error`,
    /// `api_error`). Surfaces as a stream `Err` rather than being parsed as
    /// a "failed to parse SSE event".
    Error {
        error: AnthropicErrorPayload,
    },
}

/// Payload of a mid-stream `event: error` frame.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicErrorPayload {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

/// Delta for content blocks. Variant names mirror the wire `type`
/// discriminator (`text_delta`, `input_json_delta`, …); the shared
/// `Delta` suffix is intentional.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(clippy::enum_variant_names)]
pub enum AnthropicContentDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    /// Incremental update to a `thinking` content block.
    ThinkingDelta {
        thinking: String,
    },
    /// Signature appended to a `thinking` block at end-of-block. Required to
    /// be echoed back unchanged in subsequent turns.
    SignatureDelta {
        signature: String,
    },
}

/// Delta for message-level changes carried by a `message_delta` event.
///
/// Note: `usage` is intentionally NOT a field here. Per Anthropic's wire
/// format, the `usage` object on `message_delta` is a sibling of `delta`, not
/// nested inside it — the parent enum variant carries it.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessageDelta {
    /// Reason the model stopped (`end_turn`, `max_tokens`, …).
    #[serde(default)]
    pub stop_reason: Option<String>,
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
