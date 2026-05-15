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

/// Tagged content part within an OpenAI message. Variant names mirror
/// the wire `type` discriminator (`input_text`, `input_image`, …) so
/// the Rust-side naming stays in lock-step with the API; the shared
/// `Input` prefix is intentional.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(clippy::enum_variant_names)]
pub enum OpenAIContentPart {
    InputText {
        text: String,
    },
    InputImage {
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
    },
    InputAudio {
        input_audio: OpenAIInputAudio,
    },
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

/// OpenAI Responses API response. Only carries the fields the
/// streaming converter actually reads — extra metadata (`object`,
/// `created_at`, `model`, …) is on the wire but stripped by serde
/// since nothing consumes it today.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub output: Vec<ResponseItem>,
    pub usage: Option<OpenAIUsage>,
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
}

/// OpenAI usage wire shape. The `*_tokens_details` sub-objects
/// surface cached-prompt and reasoning-output counts that the
/// canonical [`Usage`] flattens into top-level fields.
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub input_tokens_details: Option<OpenAIInputTokensDetails>,
    #[serde(default)]
    pub output_tokens_details: Option<OpenAIOutputTokensDetails>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIInputTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIOutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
}

impl From<OpenAIUsage> for Usage {
    fn from(u: OpenAIUsage) -> Self {
        Usage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            cache_read_input_tokens: u.input_tokens_details.and_then(|d| d.cached_tokens),
            cache_creation_input_tokens: None,
            reasoning_tokens: u.output_tokens_details.and_then(|d| d.reasoning_tokens),
        }
    }
}

/// Output item in a Responses API response. Carries only the fields
/// the streaming converter reads today; additional metadata fields are
/// stripped by serde.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseItem {
    pub r#type: String, // "message" / "function_call" / "web_search_call" / "reasoning"
    pub id: String,
    /// Function name on a `function_call` item.
    #[serde(default)]
    pub name: Option<String>,
    /// Function `call_id` on a `function_call` item — distinct from
    /// `id` (`fc_…`) and the one that must be echoed back as the
    /// `call_id` on the matching `function_call_output`.
    #[serde(default)]
    pub call_id: Option<String>,
    /// Builtin-tool call payload. Populated on items like
    /// `web_search_call` (search action with queries) or
    /// `code_interpreter_call`. Preserved as an opaque JSON value so
    /// new builtins don't fail the parse.
    #[serde(default)]
    pub action: Option<IValue>,
}

/// Content item in a Responses API output. Currently only the `type`
/// discriminator is consumed (to pick `PartKind::Text` vs
/// `PartKind::Refusal` on `response.content_part.added`).
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseContent {
    pub r#type: String, // "output_text", "refusal", etc.
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

/// OpenAI streaming Responses API event — one variant per wire event
/// type. Replaces the previous string-matching dispatch over an
/// any-of-these-fields-may-be-set struct: the enum makes per-event
/// required fields explicit, dispatches via serde, and surfaces
/// unknown event types through [`Self::Unknown`] (logged at warning
/// level) instead of silently dropping into a catch-all.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum OpenAIStreamEvent {
    /// Wire-level error frame.
    #[serde(rename = "error")]
    Error { error: ErrorDetails },

    /// Initial frame — carries the response shell with its id. The
    /// id is stable across created/in_progress/completed, so we lift
    /// the continuation only at end-of-stream; this variant is
    /// acknowledged but its payload isn't consumed.
    #[serde(rename = "response.created")]
    ResponseCreated,
    /// Heartbeat-style status frame; payload unused (see `ResponseCreated`).
    #[serde(rename = "response.in_progress")]
    ResponseInProgress,

    /// New output item opening (message / function_call / reasoning /
    /// web_search_call / …).
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        output_index: u32,
        item: ResponseItem,
    },
    /// Output item closing — final canonical item is in `item`.
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        output_index: u32,
        item: ResponseItem,
    },

    /// Content part within a `message` item opening.
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        output_index: u32,
        content_index: u32,
        part: ResponseContent,
    },
    /// Content part closing.
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        output_index: u32,
        content_index: u32,
    },

    /// Token-by-token text delta on an `output_text` content part.
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        output_index: u32,
        content_index: u32,
        delta: String,
    },
    /// Token-by-token refusal delta on a `refusal` content part.
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta {
        output_index: u32,
        content_index: u32,
        delta: String,
    },

    /// New reasoning summary part inside a `reasoning` item opening.
    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded {
        output_index: u32,
        summary_index: u32,
    },
    /// Reasoning summary part closing.
    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone {
        output_index: u32,
        summary_index: u32,
    },
    /// Token delta within an open reasoning summary part.
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        output_index: u32,
        summary_index: u32,
        delta: String,
    },
    /// Alternative reasoning channel (some models emit
    /// `reasoning_text.delta` instead of `reasoning_summary_text.delta`).
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta { output_index: u32, delta: String },

    /// JSON-argument delta on a function_call item.
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta { output_index: u32, delta: String },

    /// Citation / file reference added to an `output_text` part.
    #[serde(rename = "response.output_text.annotation.added")]
    OutputTextAnnotationAdded {
        output_index: u32,
        content_index: u32,
        annotation: OpenAIAnnotation,
    },

    /// Terminal success frame. Carries the final response with usage.
    #[serde(rename = "response.completed")]
    ResponseCompleted { response: ResponsesResponse },
    /// Terminal frame for premature termination
    /// (max_output_tokens, content_filter, …). Carries
    /// `incomplete_details.reason`.
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete { response: ResponsesResponse },
    /// Terminal error frame mid-generation (rate limit, server error,
    /// safety block). Either `response.error` or top-level `error` is
    /// populated.
    #[serde(rename = "response.failed")]
    ResponseFailed {
        #[serde(default)]
        response: Option<ResponsesResponse>,
        #[serde(default)]
        error: Option<ErrorDetails>,
    },

    // ---- intentionally-ignored frames (no `StreamEvent` produced) ----
    /// Final-canonical-value frame for `output_text` — we already have
    /// the content via deltas.
    #[serde(rename = "response.output_text.done")]
    OutputTextDone,
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone,
    #[serde(rename = "response.reasoning_text.done")]
    ReasoningTextDone,
    #[serde(rename = "response.refusal.done")]
    RefusalDone,
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone,
    /// Lifecycle frames for `web_search_call` items — status is folded
    /// into the queries `Delta` emitted at `output_item.done`.
    #[serde(rename = "response.web_search_call.in_progress")]
    WebSearchCallInProgress,
    #[serde(rename = "response.web_search_call.searching")]
    WebSearchCallSearching,
    #[serde(rename = "response.web_search_call.completed")]
    WebSearchCallCompleted,

    /// Unknown event type — OpenAI added something we don't yet
    /// recognise. The stream-state logs the variant at `warn` level
    /// and continues; investigate via the captured `.response.sse`
    /// file when it fires.
    #[serde(other)]
    Unknown,
}

/// Tagged annotation payload OpenAI emits with `output_text` parts. The
/// concrete variants follow OpenAI's `type` discriminator:
/// `url_citation` (web_search grounding), `file_citation`
/// (retrieval / file_search tools), `container_file_citation` (code
/// interpreter), `file_path` (file_search output references).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIAnnotation {
    UrlCitation {
        #[serde(default)]
        start_index: usize,
        #[serde(default)]
        end_index: usize,
        url: String,
        #[serde(default)]
        title: Option<String>,
    },
    FileCitation {
        file_id: String,
        #[serde(default)]
        filename: Option<String>,
    },
    /// Unknown / future annotation variants — captured generically so
    /// new shapes don't fail the parse.
    #[serde(other)]
    Other,
}
