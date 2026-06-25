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

use crate::ProviderType;

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
    /// File input — image, audio, video, or document. The modality is
    /// derived from [`FileInput::mime_type`] by [`modality_from_mime`]
    /// (providers map per-modality and the capability check gates on it),
    /// so a single variant carries every binary input kind.
    File(FileInput),
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

/// A file input to the model — image, audio, video, or document.
///
/// `mime_type` is top-level and authoritative: every provider mapping
/// carries it onto the wire verbatim, so a `gs://` or HTTP URL is no
/// longer forced to a hard-coded MIME the way the old per-kind `Url`
/// variants were. The modality (image / audio / video / document) is
/// derived from it via [`modality_from_mime`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInput {
    /// The file's MIME type (e.g. `image/png`, `audio/mpeg`,
    /// `application/pdf`). Authoritative — providers send this exact
    /// value, and the modality / capability gate is derived from it.
    pub mime_type: String,
    /// Where the bytes come from.
    pub source: FileSource,
}

impl FileInput {
    /// Build a [`FileInput`] from inline bytes and a MIME type.
    pub fn bytes(mime_type: impl Into<String>, bytes: impl Into<bytes::Bytes>) -> Self {
        Self {
            mime_type: mime_type.into(),
            source: FileSource::Bytes(bytes.into()),
        }
    }

    /// Build a [`FileInput`] referencing a fetch-by-URL source. For
    /// Gemini-via-Vertex this is the only place `gs://` URIs work; for
    /// OpenAI / Anthropic it is an HTTP(S) URL the provider fetches.
    pub fn url(mime_type: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            mime_type: mime_type.into(),
            source: FileSource::Url(url.into()),
        }
    }

    /// Build a [`FileInput`] referencing a previously
    /// [`Provider::upload`](crate::Provider::upload)ed provider file.
    pub fn uploaded(mime_type: impl Into<String>, file_ref: ProviderFileRef) -> Self {
        Self {
            mime_type: mime_type.into(),
            source: FileSource::Uploaded(file_ref),
        }
    }

    /// The modality this file represents, derived from [`Self::mime_type`].
    pub fn modality(&self) -> Modality {
        modality_from_mime(&self.mime_type)
    }
}

/// Where a [`FileInput`]'s bytes come from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileSource {
    /// Inline bytes — sent as base64 in the provider's inline format.
    Bytes(bytes::Bytes),
    /// A URL the provider fetches. HTTP(S) on OpenAI / Anthropic;
    /// additionally `gs://` Cloud Storage URIs on Gemini-via-Vertex
    /// (`fileData.fileUri`).
    Url(String),
    /// A file already uploaded to a provider's Files API, referenced by
    /// id. The [`ProviderFileRef::provider`] stamp is validated against
    /// the sending provider so a wrong-provider id can't silently leak
    /// onto the wire.
    Uploaded(ProviderFileRef),
}

/// A handle to a file uploaded to a specific provider's Files API,
/// returned by [`Provider::upload`](crate::Provider::upload). The
/// `provider` stamp records which provider owns `id`; providers reject a
/// reference whose `provider` doesn't match their own type rather than
/// sending an id the upstream won't recognise.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderFileRef {
    /// Which provider this file id belongs to.
    pub provider: ProviderType,
    /// The provider-assigned file identifier (e.g. OpenAI's `file-…`).
    pub id: String,
}

/// The coarse modality of a file input, derived from its MIME type by
/// [`modality_from_mime`]. Providers map per-modality and the
/// capability gate (`Capabilities::files`) is keyed on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    /// `image/*`.
    Image,
    /// `audio/*`.
    Audio,
    /// `video/*`.
    Video,
    /// Anything else (`application/pdf`, `text/*`, …) — treated as a
    /// document.
    Document,
}

/// Derive the [`Modality`] from a MIME type by its top-level type:
/// `image/*` → image, `audio/*` → audio, `video/*` → video; everything
/// else (PDFs, text, office formats) → document. Comparison is
/// case-insensitive on the top-level type.
pub fn modality_from_mime(mime_type: &str) -> Modality {
    let top = mime_type.split('/').next().unwrap_or("");
    if top.eq_ignore_ascii_case("image") {
        Modality::Image
    } else if top.eq_ignore_ascii_case("audio") {
        Modality::Audio
    } else if top.eq_ignore_ascii_case("video") {
        Modality::Video
    } else {
        Modality::Document
    }
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
    /// Identifier correlating this call to its
    /// [`UserPart::ToolResult`]. **Opaque and provider-specific** —
    /// correlate by this value (or by position within the assistant
    /// turn), never parse it or assume it maps to a provider-side
    /// identifier:
    ///
    /// - OpenAI / Anthropic carry a stable id on the wire; it round-trips
    ///   unchanged.
    /// - Gemini's `functionCall` parts have **no** id on the wire, so the
    ///   Google provider synthesizes a fresh `call_<uuid>` per parsed call
    ///   and rebuilds a `call_id → name` map from assistant history on each
    ///   request. Such an id is meaningful only within the round trip that
    ///   produced it; on echo, Gemini matches by function *name* and
    ///   position, not by this id. Duplicate-named calls in one turn are
    ///   therefore matched positionally — preserve `(ToolCall, ToolResult)`
    ///   ordering in the event log.
    pub call_id: String,
    /// Name of the tool the model is invoking.
    pub name: String,
    /// JSON-encoded argument object.
    pub arguments: String,
    /// Opaque provider-specific signature attached to this call, when the
    /// provider emits one. Currently carries Gemini 2.5+ thinking models'
    /// `thoughtSignature` — a cryptographic blob bound to the call that the
    /// provider may require echoed back to preserve thinking continuity.
    /// Populated by the Google response parser and echoed back verbatim,
    /// **unconditionally**, by the Google request serializer.
    ///
    /// There is no provenance tracking: the field records neither the
    /// provider nor the model that produced it, and it is never cleared
    /// from history. It crosses a provider switch only *emergently* — the
    /// OpenAI and Anthropic tool-call serializers read `call_id` / `name` /
    /// `arguments` and ignore this field, so it is simply not transmitted
    /// to them (mirroring how [`AssistantPart::Reasoning`]'s `signature` is
    /// dropped because those providers drop the whole reasoning part). The
    /// value still lives on the call in history, so switching away from
    /// Gemini and back re-echoes the original signature, and switching
    /// between Gemini models echoes a signature a different model produced.
    /// Gemini currently tolerates a stale or absent `thoughtSignature`, so
    /// neither is rejected today; if that changes this needs an origin
    /// (provider + model) guard before echo. `None` for providers that
    /// don't emit one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_signature: Option<String>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modality_from_mime_splits_on_top_level_type() {
        assert_eq!(modality_from_mime("image/png"), Modality::Image);
        assert_eq!(modality_from_mime("image/jpeg"), Modality::Image);
        assert_eq!(modality_from_mime("audio/mpeg"), Modality::Audio);
        assert_eq!(modality_from_mime("audio/wav"), Modality::Audio);
        assert_eq!(modality_from_mime("video/mp4"), Modality::Video);
        assert_eq!(modality_from_mime("video/webm"), Modality::Video);
    }

    #[test]
    fn modality_from_mime_treats_non_media_as_document() {
        // PDFs, office formats, text and bare/garbage values all map to
        // Document — the catch-all branch.
        for m in [
            "application/pdf",
            "text/plain",
            "text/csv",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/octet-stream",
            "",
            "notamime",
        ] {
            assert_eq!(modality_from_mime(m), Modality::Document, "{m}");
        }
    }

    #[test]
    fn modality_from_mime_is_case_insensitive_on_top_level() {
        assert_eq!(modality_from_mime("IMAGE/PNG"), Modality::Image);
        assert_eq!(modality_from_mime("Audio/Mpeg"), Modality::Audio);
        assert_eq!(modality_from_mime("VIDEO/mp4"), Modality::Video);
    }

    #[test]
    fn file_input_helpers_build_expected_sources() {
        let b = FileInput::bytes("image/png", bytes::Bytes::from_static(b"\x89PNG"));
        assert_eq!(b.mime_type, "image/png");
        assert_eq!(b.modality(), Modality::Image);
        assert!(matches!(b.source, FileSource::Bytes(_)));

        let u = FileInput::url("application/pdf", "gs://bucket/doc.pdf");
        assert_eq!(u.modality(), Modality::Document);
        match u.source {
            FileSource::Url(ref s) => assert_eq!(s, "gs://bucket/doc.pdf"),
            _ => panic!("expected Url source"),
        }

        let r = ProviderFileRef {
            provider: ProviderType::OpenAI,
            id: "file-abc".into(),
        };
        let up = FileInput::uploaded("audio/mpeg", r.clone());
        assert_eq!(up.modality(), Modality::Audio);
        match up.source {
            FileSource::Uploaded(ref got) => assert_eq!(got, &r),
            _ => panic!("expected Uploaded source"),
        }
    }
}
