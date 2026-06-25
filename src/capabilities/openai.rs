//! OpenAI capability table.
//!
//! Sourced from <https://developers.openai.com/api/docs/models> as of
//! 2026-06. Keep the per-row comments accurate — they're the audit
//! trail for the next refresh.

use super::{Capabilities, FileCapabilities, ModelEntry, ModelMatch};
use ModelMatch::{Exact, Prefix};

/// Build an OpenAI capabilities entry. Every modern OpenAI Chat /
/// Responses model supports native JSON mode, JSON schema, and schema
/// + tools combined; only the token limits vary.
///
/// File support: OpenAI exposes a real Files API (`upload: true`) and
/// vision-class models accept image / audio / document inputs; video is
/// not a supported input modality on the Responses API.
const fn caps(context: u32, output: u32) -> Capabilities {
    Capabilities {
        native_json_mode: true,
        response_schema: true,
        response_schema_with_tools: true,
        context_window_tokens: context,
        max_output_tokens: output,
        files: FileCapabilities {
            upload: true,
            image: true,
            audio: true,
            video: false,
            document: true,
        },
    }
}

/// OpenAI model table, ordered most-specific first.
pub(super) static MODELS: &[ModelEntry] = &[
    // ----- GPT-5 family (released 2025; gpt-5.5 added 2026-04) -----
    (Prefix("gpt-5.5"), caps(1_050_000, 128_000)),
    (Prefix("gpt-5.4-mini"), caps(400_000, 128_000)),
    (Prefix("gpt-5.4-nano"), caps(400_000, 128_000)),
    (Prefix("gpt-5.4"), caps(1_050_000, 128_000)),
    (Prefix("gpt-5-mini"), caps(400_000, 128_000)),
    (Prefix("gpt-5-nano"), caps(400_000, 128_000)),
    (Prefix("gpt-5"), caps(400_000, 128_000)),
    // ----- GPT-4.1 family (1M context) -----
    (Prefix("gpt-4.1"), caps(1_047_576, 32_768)),
    // ----- GPT-4o family -----
    (Prefix("gpt-4o-mini"), caps(128_000, 16_384)),
    (Prefix("gpt-4o"), caps(128_000, 16_384)),
    (Prefix("chatgpt-4o"), caps(128_000, 16_384)),
    // ----- GPT-4 turbo / preview / vision (all 128k context) -----
    // These all share GPT-4 Turbo's 128k window. Listed *before* the
    // `gpt-4-` legacy catch-all so dated snapshots / preview tags
    // pick up their real cap rather than the 8k fallback.
    (Prefix("gpt-4-turbo"), caps(128_000, 4096)),
    (Prefix("gpt-4-vision-preview"), caps(128_000, 4096)),
    (Prefix("gpt-4-1106-preview"), caps(128_000, 4096)),
    (Prefix("gpt-4-0125-preview"), caps(128_000, 4096)),
    // gpt-4-32k (and its dated snapshots) — 32k context.
    (Prefix("gpt-4-32k"), caps(32_768, 8192)),
    // ----- GPT-4 legacy (8k context) -----
    (Exact("gpt-4"), caps(8192, 8192)),
    (Prefix("gpt-4-"), caps(8192, 8192)),
    // ----- o-series reasoning models -----
    (Prefix("o1-mini"), caps(128_000, 65_536)),
    (Prefix("o1-preview"), caps(128_000, 32_768)),
    (Prefix("o1"), caps(200_000, 100_000)),
    (Prefix("o3-mini"), caps(200_000, 100_000)),
    (Prefix("o3"), caps(200_000, 100_000)),
    (Prefix("o4-mini"), caps(200_000, 100_000)),
    (Prefix("o4"), caps(200_000, 100_000)),
    // ----- Family catch-all -----
    // No `Prefix("o")` row on purpose — it would over-match names like
    // `openai-experimental` or `oracle-x` and disagree with
    // `for_model`'s `o<digit>` dispatch. Unknown o-series fall to
    // [`FALLBACK`] (same conservative caps + a tracing breadcrumb).
    (Prefix("gpt-"), caps(128_000, 16_384)),
    (Prefix("chatgpt-"), caps(128_000, 16_384)),
];

/// Fallback when nothing in [`MODELS`] matches.
pub(super) const FALLBACK: Capabilities = caps(128_000, 16_384);
