//! Anthropic Claude capability table.
//!
//! Anthropic has no native JSON mode; structured output is expressed
//! via tool-use coercion (see [`crate::JsonCoercionMiddleware`]).
//!
//! Context windows are the **default** values without any beta
//! headers. Anthropic gates the 1M-token window for Sonnet 4.6 /
//! Opus 4.6+ behind the `context-1m-2025-08-07` (or successor)
//! `anthropic-beta` header — callers who opt into that header should
//! override [`crate::Provider::capabilities`] on their provider to
//! report the wider window. Defaulting to 200k here under-promises:
//! the headroom helpers trigger compaction earlier, which is safer
//! than over-promising and rejecting a request the model would
//! otherwise accept under the beta path.

use super::{Capabilities, ModelEntry, ModelMatch, ThinkingSupport};
use crate::types::ReasoningEffort::{High, Low, Max, Medium, XHigh};
use ModelMatch::Prefix;
use ThinkingSupport::Efforts;

/// Build an Anthropic capabilities entry.
const fn caps(context: u32, output: u32, thinking: ThinkingSupport) -> Capabilities {
    Capabilities {
        native_json_mode: false,
        response_schema: false,
        response_schema_with_tools: false,
        context_window_tokens: context,
        max_output_tokens: output,
        thinking,
    }
}

/// Extended thinking: `budget_tokens` of at least 1024 and below `max_tokens`.
const fn budget(output: u32) -> ThinkingSupport {
    ThinkingSupport::Budget {
        min: 1024,
        max: output - 1,
        can_disable: true,
    }
}

const LOW_TO_MAX: ThinkingSupport = Efforts(&[Low, Medium, High, XHigh, Max]);
const NO_XHIGH: ThinkingSupport = Efforts(&[Low, Medium, High, Max]);

/// Anthropic Claude model table, ordered most-specific first.
pub(super) static MODELS: &[ModelEntry] = &[
    // ----- Claude 4.x Opus -----
    // 4.6+ go to 1M with the context beta header — we under-promise
    // to the no-beta default (200k). See module-doc above.
    (
        Prefix("claude-opus-4-8"),
        caps(200_000, 128_000, LOW_TO_MAX),
    ),
    (
        Prefix("claude-opus-4-7"),
        caps(200_000, 128_000, LOW_TO_MAX),
    ),
    (Prefix("claude-opus-4-6"), caps(200_000, 128_000, NO_XHIGH)),
    (
        Prefix("claude-opus-4-1"),
        caps(200_000, 32_000, budget(32_000)),
    ),
    (
        Prefix("claude-opus-4"),
        caps(200_000, 32_000, budget(32_000)),
    ),
    // ----- Claude 4.x Sonnet (same beta caveat as Opus 4.6+) -----
    (
        Prefix("claude-sonnet-4-6"),
        caps(200_000, 128_000, NO_XHIGH),
    ),
    (
        Prefix("claude-sonnet-4-5"),
        caps(200_000, 64_000, budget(64_000)),
    ),
    (
        Prefix("claude-sonnet-4"),
        caps(200_000, 64_000, budget(64_000)),
    ),
    // ----- Claude 4.5 Haiku -----
    (
        Prefix("claude-haiku-4-5"),
        caps(200_000, 64_000, budget(64_000)),
    ),
    // ----- Claude 3.7 / 3.5 -----
    (
        Prefix("claude-3-7-sonnet"),
        caps(200_000, 64_000, budget(64_000)),
    ),
    (
        Prefix("claude-3-5-sonnet"),
        caps(200_000, 8192, ThinkingSupport::None),
    ),
    (
        Prefix("claude-3-5-haiku"),
        caps(200_000, 8192, ThinkingSupport::None),
    ),
    // ----- Claude 3 (legacy) -----
    (
        Prefix("claude-3"),
        caps(200_000, 4096, ThinkingSupport::None),
    ),
    // ----- Family catch-all -----
    (
        Prefix("claude-"),
        caps(200_000, 8192, ThinkingSupport::Unknown),
    ),
];

/// Fallback when nothing in [`MODELS`] matches.
pub(super) const FALLBACK: Capabilities = caps(200_000, 8192, ThinkingSupport::Unknown);
