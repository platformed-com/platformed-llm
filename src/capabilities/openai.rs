//! OpenAI capability table.
//!
//! Sourced from <https://developers.openai.com/api/docs/models> as of
//! 2026-08. Keep the per-row comments accurate — they're the audit
//! trail for the next refresh.

use super::{Capabilities, ModelEntry, ModelMatch, ThinkingSupport};
use crate::types::ReasoningEffort::{High, Low, Max, Medium, Minimal, None as NoEffort, XHigh};
use ModelMatch::{Exact, Prefix};
use ThinkingSupport::Efforts;

/// Build an OpenAI capabilities entry. Every modern OpenAI Chat /
/// Responses model supports native JSON mode, JSON schema, and schema
/// + tools combined; only the token limits vary.
const fn caps(context: u32, output: u32, thinking: ThinkingSupport) -> Capabilities {
    Capabilities {
        native_json_mode: true,
        response_schema: true,
        response_schema_with_tools: true,
        context_window_tokens: context,
        max_output_tokens: output,
        thinking,
    }
}

const NONE_TO_MAX: ThinkingSupport = Efforts(&[NoEffort, Low, Medium, High, XHigh, Max]);
const NONE_TO_XHIGH: ThinkingSupport = Efforts(&[NoEffort, Low, Medium, High, XHigh]);
const MEDIUM_TO_XHIGH: ThinkingSupport = Efforts(&[Medium, High, XHigh]);
const MINIMAL_TO_HIGH: ThinkingSupport = Efforts(&[Minimal, Low, Medium, High]);
const LOW_TO_HIGH: ThinkingSupport = Efforts(&[Low, Medium, High]);
const NO_REASONING: ThinkingSupport = ThinkingSupport::None;
const UNKNOWN: ThinkingSupport = ThinkingSupport::Unknown;

/// OpenAI model table, ordered most-specific first.
pub(super) static MODELS: &[ModelEntry] = &[
    // ----- GPT-5 family (released 2025; gpt-5.5 added 2026-04, gpt-5.6 added 2026-07) -----
    // `gpt-5.6` covers the sol / terra / luna tiers, which share limits.
    (Prefix("gpt-5.6"), caps(1_050_000, 128_000, NONE_TO_MAX)),
    (
        Prefix("gpt-5.5-pro"),
        caps(1_050_000, 128_000, MEDIUM_TO_XHIGH),
    ),
    (Prefix("gpt-5.5"), caps(1_050_000, 128_000, NONE_TO_XHIGH)),
    (
        Prefix("gpt-5.4-pro"),
        caps(1_050_000, 128_000, MEDIUM_TO_XHIGH),
    ),
    (
        Prefix("gpt-5.4-mini"),
        caps(400_000, 128_000, NONE_TO_XHIGH),
    ),
    (Prefix("gpt-5.4-nano"), caps(400_000, 128_000, UNKNOWN)),
    (Prefix("gpt-5.4"), caps(1_050_000, 128_000, NONE_TO_XHIGH)),
    (
        Prefix("gpt-5-mini"),
        caps(400_000, 128_000, MINIMAL_TO_HIGH),
    ),
    (
        Prefix("gpt-5-nano"),
        caps(400_000, 128_000, MINIMAL_TO_HIGH),
    ),
    (Prefix("gpt-5"), caps(400_000, 128_000, MINIMAL_TO_HIGH)),
    // ----- GPT-4.1 family (1M context) -----
    (Prefix("gpt-4.1"), caps(1_047_576, 32_768, NO_REASONING)),
    // ----- GPT-4o family -----
    (Prefix("gpt-4o-mini"), caps(128_000, 16_384, NO_REASONING)),
    (Prefix("gpt-4o"), caps(128_000, 16_384, NO_REASONING)),
    (Prefix("chatgpt-4o"), caps(128_000, 16_384, NO_REASONING)),
    // ----- GPT-4 turbo / preview / vision (all 128k context) -----
    // These all share GPT-4 Turbo's 128k window. Listed *before* the
    // `gpt-4-` legacy catch-all so dated snapshots / preview tags
    // pick up their real cap rather than the 8k fallback.
    (Prefix("gpt-4-turbo"), caps(128_000, 4096, NO_REASONING)),
    (
        Prefix("gpt-4-vision-preview"),
        caps(128_000, 4096, NO_REASONING),
    ),
    (
        Prefix("gpt-4-1106-preview"),
        caps(128_000, 4096, NO_REASONING),
    ),
    (
        Prefix("gpt-4-0125-preview"),
        caps(128_000, 4096, NO_REASONING),
    ),
    // gpt-4-32k (and its dated snapshots) — 32k context.
    (Prefix("gpt-4-32k"), caps(32_768, 8192, NO_REASONING)),
    // ----- GPT-4 legacy (8k context) -----
    (Exact("gpt-4"), caps(8192, 8192, NO_REASONING)),
    (Prefix("gpt-4-"), caps(8192, 8192, NO_REASONING)),
    // ----- o-series reasoning models -----
    (Prefix("o1-mini"), caps(128_000, 65_536, NO_REASONING)),
    (Prefix("o1-preview"), caps(128_000, 32_768, NO_REASONING)),
    (Prefix("o1"), caps(200_000, 100_000, LOW_TO_HIGH)),
    (Prefix("o3-mini"), caps(200_000, 100_000, LOW_TO_HIGH)),
    (Prefix("o3"), caps(200_000, 100_000, LOW_TO_HIGH)),
    (Prefix("o4-mini"), caps(200_000, 100_000, LOW_TO_HIGH)),
    (Prefix("o4"), caps(200_000, 100_000, LOW_TO_HIGH)),
    // ----- Family catch-all -----
    // No `Prefix("o")` row on purpose — it would over-match names like
    // `openai-experimental` or `oracle-x` and disagree with
    // `for_model`'s `o<digit>` dispatch. Unknown o-series fall to
    // [`FALLBACK`] (same conservative caps + a tracing breadcrumb).
    (Prefix("gpt-"), caps(128_000, 16_384, UNKNOWN)),
    (Prefix("chatgpt-"), caps(128_000, 16_384, UNKNOWN)),
];

/// Fallback when nothing in [`MODELS`] matches.
pub(super) const FALLBACK: Capabilities = caps(128_000, 16_384, UNKNOWN);
