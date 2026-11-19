//! Google / Gemini capability table.
//!
//! Sourced from Google Cloud / AI Studio docs as of 2026-06, thinking
//! support as of 2026-10. Per
//! Google's docs, the 3.x series supports combining `responseSchema`
//! with function-calling tools (preview), while 2.5 / 2.0 / 1.5
//! support schema-constrained output but **not** in combination with
//! tools.

use super::{Capabilities, ModelEntry, ModelMatch, ThinkingSupport};
use crate::types::ReasoningEffort::{High, Low, Medium, Minimal};
use ModelMatch::Prefix;
use ThinkingSupport::Efforts;

/// Build a Gemini capabilities entry with the supplied feature /
/// limit combination.
const fn caps(
    schema_with_tools: bool,
    context: u32,
    output: u32,
    thinking: ThinkingSupport,
) -> Capabilities {
    Capabilities {
        native_json_mode: true,
        response_schema: true,
        response_schema_with_tools: schema_with_tools,
        context_window_tokens: context,
        max_output_tokens: output,
        thinking,
    }
}

const ALL_LEVELS: ThinkingSupport = Efforts(&[Minimal, Low, Medium, High]);
const NO_MINIMAL: ThinkingSupport = Efforts(&[Low, Medium, High]);
// The levels every Gemini 3 model accepts, for variants not listed here.
const COMMON_LEVELS: ThinkingSupport = Efforts(&[Low, High]);

/// Google / Gemini model table, ordered most-specific first.
pub(super) static MODELS: &[ModelEntry] = &[
    // ----- Gemini 3.x — supports schema + tools (preview as of 2026-06) -----
    (
        Prefix("gemini-3.8-flash"),
        caps(true, 1_048_576, 65_536, NO_MINIMAL),
    ),
    (
        Prefix("gemini-3.5-flash"),
        caps(true, 1_048_576, 65_536, ALL_LEVELS),
    ),
    (
        Prefix("gemini-3.1-flash-lite"),
        caps(true, 1_000_000, 64_000, ALL_LEVELS),
    ),
    (
        Prefix("gemini-3.1-pro"),
        caps(true, 1_048_576, 65_536, NO_MINIMAL),
    ),
    (
        Prefix("gemini-3-flash"),
        caps(true, 200_000, 64_000, ALL_LEVELS),
    ),
    (
        Prefix("gemini-3-pro"),
        caps(true, 1_000_000, 64_000, COMMON_LEVELS),
    ),
    (
        Prefix("gemini-3"),
        caps(true, 1_000_000, 64_000, COMMON_LEVELS),
    ),
    // ----- Gemini 2.5 -----
    (
        Prefix("gemini-2.5-flash-lite"),
        caps(false, 1_048_576, 65_535, budget(512, 24_576, true)),
    ),
    (
        Prefix("gemini-2.5-flash"),
        caps(false, 1_048_576, 65_535, budget(1, 24_576, true)),
    ),
    (
        Prefix("gemini-2.5-pro"),
        caps(false, 1_048_576, 65_536, budget(128, 32_768, false)),
    ),
    (
        Prefix("gemini-2.5"),
        caps(false, 1_048_576, 65_535, budget(512, 24_576, true)),
    ),
    // ----- Gemini 2.0 -----
    (
        Prefix("gemini-2.0"),
        caps(false, 1_000_000, 8_192, ThinkingSupport::None),
    ),
    // ----- Gemini 1.5 -----
    (
        Prefix("gemini-1.5-pro"),
        caps(false, 2_000_000, 8192, ThinkingSupport::None),
    ),
    (
        Prefix("gemini-1.5-flash-8b"),
        caps(false, 1_000_000, 8192, ThinkingSupport::None),
    ),
    (
        Prefix("gemini-1.5-flash"),
        caps(false, 1_000_000, 8192, ThinkingSupport::None),
    ),
    (
        Prefix("gemini-1.5"),
        caps(false, 1_000_000, 8192, ThinkingSupport::None),
    ),
    // ----- Family catch-all -----
    (
        Prefix("gemini-"),
        caps(false, 1_000_000, 8192, ThinkingSupport::Unknown),
    ),
];

const fn budget(min: u32, max: u32, can_disable: bool) -> ThinkingSupport {
    ThinkingSupport::Budget {
        min,
        max,
        can_disable,
    }
}

/// Fallback when nothing in [`MODELS`] matches.
pub(super) const FALLBACK: Capabilities = caps(false, 1_000_000, 8192, ThinkingSupport::Unknown);
