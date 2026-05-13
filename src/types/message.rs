//! Canonical message model.
//!
//! `InputItem` is variant-by-role: `System`, `User`, `Assistant`. The
//! content of `User` and `Assistant` items is a `Vec` of typed parts so
//! the model can represent interleaved text + reasoning + tool calls + …
//! within a single turn — the way Anthropic emits its content blocks.
//!
//! Provider-specific parts (`UserPart::CacheBreakpoint`,
//! `AssistantPart::Reasoning::signature`, etc.) are carried losslessly
//! when the conversation round-trips through the same provider and
//! silently dropped or best-effort translated when the lib sends to a
//! different provider. That's the explicit contract that makes
//! switching models mid-conversation always work — see FOLLOWUPS Phase 5
//! for the full drop / translate matrix.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

/// A single item in a conversation history.
///
/// Each item is one logical turn. The variant encodes the role; the
/// content is a sequence of typed parts (for `User` and `Assistant`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputItem {
    /// System / developer instruction.
    System(String),
    /// User turn. Contains text, multimedia, tool results, and optional
    /// cache breakpoints in emit order.
    User { content: Vec<UserPart> },
    /// Assistant turn. Contains the model's emissions in the order they
    /// were produced — text, reasoning, refusals, tool calls all
    /// interleaved as parts.
    Assistant { content: Vec<AssistantPart> },
}

impl InputItem {
    /// Build a system instruction.
    pub fn system(content: impl Into<String>) -> Self {
        InputItem::System(content.into())
    }

    /// Build a user turn from a single text string.
    pub fn user(content: impl Into<String>) -> Self {
        InputItem::User {
            content: vec![UserPart::Text(content.into())],
        }
    }

    /// Build an assistant turn from a single text string.
    pub fn assistant(content: impl Into<String>) -> Self {
        InputItem::Assistant {
            content: vec![AssistantPart::Text {
                content: content.into(),
                annotations: Vec::new(),
            }],
        }
    }

    /// Build a tool-result message (a user turn with a single
    /// `UserPart::ToolResult`).
    pub fn tool_result(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        InputItem::User {
            content: vec![UserPart::ToolResult {
                call_id: call_id.into(),
                content: vec![UserPart::Text(output.into())],
            }],
        }
    }

    /// Build an assistant turn that emitted a single tool call.
    pub fn assistant_tool_call(call: FunctionCall) -> Self {
        InputItem::Assistant {
            content: vec![AssistantPart::ToolCall(call)],
        }
    }
}

/// A part of a user turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserPart {
    Text(String),
    Image(ImageSource),
    Audio(AudioSource),
    Document(DocumentSource),
    /// Result of a tool the assistant previously called. `call_id`
    /// correlates with a prior `AssistantPart::ToolCall`.
    ToolResult {
        call_id: String,
        content: Vec<UserPart>,
    },
    /// Anthropic-only: marks the end of a cacheable prefix in the
    /// surrounding message. Best-effort on OpenAI (derives a stable
    /// `prompt_cache_key`); dropped on Gemini.
    CacheBreakpoint,
}

/// A part of an assistant turn. Parts appear in the order the model
/// emitted them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssistantPart {
    /// Visible text. Annotations attach citations to specific spans.
    Text {
        content: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        annotations: Vec<Annotation>,
    },
    /// Chain-of-thought reasoning. The optional `signature` is
    /// Anthropic's thinking signature; dropped on cross-provider
    /// conversion.
    Reasoning {
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Anthropic redacted thinking — opaque blob passed back unchanged.
    RedactedReasoning { data: String },
    /// Typed refusal (OpenAI). Translated to plain text on providers
    /// that don't model refusals separately.
    Refusal(String),
    /// A tool call the model emitted.
    ToolCall(FunctionCall),
    /// Anthropic cache breakpoint marker, same semantics as
    /// [`UserPart::CacheBreakpoint`].
    CacheBreakpoint,
}

/// Citation or annotation attached to a span within an
/// [`AssistantPart::Text`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub kind: AnnotationKind,
    pub start: usize,
    pub end: usize,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnnotationKind {
    UrlCitation,
    FileCitation,
    WebSearch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageSource {
    Url(String),
    Base64 { data: String, media_type: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioSource {
    Url(String),
    Base64 { data: String, media_type: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentSource {
    Url(String),
    Base64 { data: String, media_type: String },
}

/// Tool definition the model can call.
///
/// Most tools are caller-defined functions (`Tool::Function`). Some
/// providers offer pre-baked tools (web search, computer use, code
/// execution) configured by name — those land on `Tool::Builtin`, which
/// is silently dropped from the tools array on providers that don't
/// offer the same builtin.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Tool {
    Function(Function),
    Builtin(ProviderBuiltin),
}

impl Tool {
    /// Convenience: build a function tool from name, description, and
    /// a parsed JSON-schema parameters value.
    pub fn function(
        name: impl Into<String>,
        description: impl Into<Option<String>>,
        parameters: Cow<'static, RawValue>,
    ) -> Self {
        Tool::Function(Function {
            name: name.into(),
            description: description.into(),
            parameters,
        })
    }

    /// Convenience: a builtin tool by kind.
    pub fn builtin(kind: ProviderBuiltin) -> Self {
        Tool::Builtin(kind)
    }

    /// Borrow the inner [`Function`] if this is a function tool.
    pub fn as_function(&self) -> Option<&Function> {
        match self {
            Tool::Function(f) => Some(f),
            _ => None,
        }
    }
}

/// Legacy alias retained for documentation symmetry. New code should
/// match on `Tool` directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Cow<'static, RawValue>,
}

/// Provider-builtin tools — pre-baked tool definitions the provider
/// invokes natively rather than calling out to the caller. Dropped from
/// the tools array on providers that don't offer the same builtin.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderBuiltin {
    /// OpenAI / Anthropic web search.
    WebSearch,
    /// Google Search retrieval (Gemini).
    GoogleSearch,
    /// Code execution (Gemini).
    CodeExecution,
    /// Computer use (OpenAI / Anthropic). Carries the virtual display
    /// dimensions and the environment the model is acting against.
    ComputerUse(ComputerUseConfig),
}

/// Configuration for the `computer_use` builtin tool. Required by
/// both OpenAI's `computer_use_preview` and Anthropic's
/// `computer_20250124` tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct ComputerUseConfig {
    pub display_width: u32,
    pub display_height: u32,
    /// Environment the model controls — `"browser"`, `"mac"`,
    /// `"windows"`, `"ubuntu"` on OpenAI; Anthropic accepts the same
    /// labels.
    pub environment: String,
}

/// A tool call emitted by the assistant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub call_id: String,
    pub name: String,
    /// JSON-encoded argument object.
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}
