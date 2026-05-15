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
    User {
        /// Ordered user-turn parts.
        content: Vec<UserPart>,
    },
    /// Assistant turn. Contains the model's emissions in the order they
    /// were produced — text, reasoning, refusals, tool calls,
    /// continuation markers, all interleaved as parts.
    Assistant {
        /// Ordered assistant-turn parts.
        content: Vec<AssistantPart>,
    },
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

    /// Build an assistant turn whose only content is a provider
    /// continuation marker. Useful for stitching a resumption hint into
    /// the conversation history outside of a real model turn (e.g. in
    /// tests).
    pub fn assistant_continuation(continuation: super::config::ProviderContinuation) -> Self {
        InputItem::Assistant {
            content: vec![AssistantPart::Continuation(continuation)],
        }
    }
}

/// A part of a user turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserPart {
    /// Plain text content.
    Text(String),
    /// Image input by URL or inline base64.
    Image(ImageSource),
    /// Audio input by URL or inline base64.
    Audio(AudioSource),
    /// Document (e.g. PDF) input by URL or inline base64.
    Document(DocumentSource),
    /// Result of a tool the assistant previously called. `call_id`
    /// correlates with a prior `AssistantPart::ToolCall`.
    ToolResult {
        /// Identifier of the originating tool call.
        call_id: String,
        /// Result payload, modelled as user parts so it can include
        /// text, images, etc.
        content: Vec<UserPart>,
    },
    /// Anthropic-only: marks the end of a cacheable prefix in the
    /// surrounding message. Best-effort on OpenAI (derives a stable
    /// `prompt_cache_key`); dropped on Gemini.
    ///
    /// Input-only: there is no [`PartKind`](super::PartKind)
    /// `CacheBreakpoint`, so this is never produced by the streaming
    /// accumulator. It survives a round-trip only via direct
    /// `content` construction, not by re-accumulating a stream.
    CacheBreakpoint,
}

/// A part of an assistant turn. Parts appear in the order the model
/// emitted them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssistantPart {
    /// Visible text. Annotations attach citations to specific spans.
    Text {
        /// The text body.
        content: String,
        /// Citations / annotations over byte spans of `content`.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        annotations: Vec<Annotation>,
    },
    /// Chain-of-thought reasoning. The optional `signature` is
    /// Anthropic's thinking signature; dropped on cross-provider
    /// conversion.
    Reasoning {
        /// The reasoning content.
        content: String,
        /// Anthropic's opaque thinking signature, when supplied.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Anthropic redacted thinking — opaque blob passed back unchanged.
    RedactedReasoning {
        /// Opaque server-encrypted thinking blob.
        data: String,
    },
    /// Typed refusal (OpenAI). Translated to plain text on providers
    /// that don't model refusals separately.
    Refusal(String),
    /// A tool call the model emitted.
    ToolCall(FunctionCall),
    /// A provider-builtin tool invocation — the provider executed the
    /// tool natively (no client round-trip). `arguments` is JSON; the
    /// shape depends on `kind` (`{"queries":[...]}` for web search,
    /// `{"language":"python","code":"..."}` for code execution).
    /// `result` is populated when the builtin returns a payload
    /// directly (Gemini's `codeExecutionResult`); web search instead
    /// surfaces sources via [`Annotation`]s on the trailing
    /// [`AssistantPart::Text`].
    BuiltinToolCall {
        /// Which builtin tool was invoked.
        kind: ProviderBuiltin,
        /// JSON-encoded arguments the model passed to the builtin.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        arguments: String,
        /// Inline result payload, when the provider returns one directly.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<String>,
    },
    /// Provider-issued resumption hint that identifies this assistant
    /// turn server-side (e.g. OpenAI's `previous_response_id`). When
    /// the caller sends a follow-up to the *same* provider, the
    /// provider scans assistant history for the latest matching
    /// continuation, sets the provider-side field, and elides this
    /// turn (and every item before it) from the wire body — the
    /// server already has them. Cross-provider markers are silently
    /// dropped (model-switching contract).
    Continuation(super::config::ProviderContinuation),
    /// Anthropic cache breakpoint marker, same semantics as
    /// [`UserPart::CacheBreakpoint`] — input-only; never produced by
    /// the streaming accumulator.
    CacheBreakpoint,
}

/// Citation or annotation attached to a span within an
/// [`AssistantPart::Text`]. `start` / `end` are byte offsets into the
/// text content; both providers report them inclusive-of-start /
/// exclusive-of-end.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    /// What kind of citation this annotation represents.
    pub kind: AnnotationKind,
    /// Inclusive start byte offset into the annotated text.
    pub start: usize,
    /// Exclusive end byte offset into the annotated text.
    pub end: usize,
    /// Primary identifier — URL for [`AnnotationKind::UrlCitation`],
    /// file ID for [`AnnotationKind::FileCitation`].
    pub source: String,
    /// Human-readable label, when the provider supplies one (e.g. page
    /// title for a URL citation, filename for a file citation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// Kind of citation an [`Annotation`] represents.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AnnotationKind {
    /// Cites a web URL (e.g. web search result).
    UrlCitation,
    /// Cites a previously uploaded file.
    FileCitation,
}

/// Source for an image input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageSource {
    /// HTTP(S) URL the provider will fetch.
    Url(String),
    /// Inline base64-encoded payload.
    Base64 {
        /// Base64-encoded image bytes.
        data: String,
        /// MIME type (e.g. `image/png`, `image/jpeg`).
        media_type: String,
    },
}

/// Source for an audio input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioSource {
    /// HTTP(S) URL the provider will fetch.
    Url(String),
    /// Inline base64-encoded payload.
    Base64 {
        /// Base64-encoded audio bytes.
        data: String,
        /// MIME type (e.g. `audio/mpeg`, `audio/wav`).
        media_type: String,
    },
}

/// Source for a document input (PDF, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentSource {
    /// HTTP(S) URL the provider will fetch.
    Url(String),
    /// Inline base64-encoded payload.
    Base64 {
        /// Base64-encoded document bytes.
        data: String,
        /// MIME type (e.g. `application/pdf`).
        media_type: String,
    },
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
    /// Caller-defined function tool.
    Function(Function),
    /// Provider-builtin tool (web search, code execution, etc.).
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

/// Caller-defined function tool the model can call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    /// Identifier the model uses to call the function.
    pub name: String,
    /// Optional natural-language description shown to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema describing the argument object.
    pub parameters: Cow<'static, RawValue>,
}

/// Provider-builtin tools — pre-baked tool definitions the provider
/// invokes natively rather than calling out to the caller. Dropped from
/// the tools array on providers that don't offer the same builtin.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
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
    /// Virtual display width in pixels.
    pub display_width: u32,
    /// Virtual display height in pixels.
    pub display_height: u32,
    /// Environment the model controls — `"browser"`, `"mac"`,
    /// `"windows"`, `"ubuntu"` on OpenAI; Anthropic accepts the same
    /// labels.
    pub environment: String,
}

/// A tool call emitted by the assistant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Identifier the model assigns to this call; echoed back in the
    /// matching `UserPart::ToolResult`.
    pub call_id: String,
    /// Name of the tool the model is invoking.
    pub name: String,
    /// JSON-encoded argument object.
    pub arguments: String,
}

/// Why the model stopped generating.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FinishReason {
    /// Natural end of the response (hit a stop sequence or the model decided to stop).
    Stop,
    /// Hit `max_tokens` / provider-side length cap before finishing.
    Length,
    /// The turn ended because the model emitted one or more tool calls.
    ToolCalls,
    /// The provider's content filter blocked or truncated the response.
    ContentFilter,
    /// The stream ended without a terminal `Done`/stop signal — the
    /// response is *incomplete* (connection dropped, task cancelled,
    /// or a local engine cut off mid-emit). Distinct from [`Self::Stop`]
    /// so callers driving tool-call loops or billing don't mistake a
    /// truncated turn for a clean finish.
    Incomplete,
}
