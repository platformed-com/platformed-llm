//! Google / Gemini capability table.
//!
//! Sourced from Google Cloud / AI Studio docs as of 2026-06. Per
//! Google's docs, the 3.x series supports combining `responseSchema`
//! with function-calling tools (preview), while 2.5 / 2.0 / 1.5
//! support schema-constrained output but **not** in combination with
//! tools.

use super::{Capabilities, FileCapabilities, ModelEntry, ModelMatch};
use ModelMatch::Prefix;

/// Build a Gemini capabilities entry with the supplied feature /
/// limit combination.
///
/// File support: Gemini is natively multimodal — image / audio / video
/// / document are all accepted. `upload` is `false`: this crate's
/// Gemini provider runs **via Vertex AI**, which has no `files.upload`
/// Files API (that surface is AI Studio's
/// `generativelanguage.googleapis.com`); Vertex references files by
/// `gs://` Cloud Storage URI through `fileData.fileUri` instead — i.e.
/// [`crate::FileSource::Url`], not an upload.
const fn caps(schema_with_tools: bool, context: u32, output: u32) -> Capabilities {
    Capabilities {
        native_json_mode: true,
        response_schema: true,
        response_schema_with_tools: schema_with_tools,
        context_window_tokens: context,
        max_output_tokens: output,
        files: FileCapabilities {
            upload: false,
            image: true,
            audio: true,
            video: true,
            document: true,
        },
    }
}

/// Google / Gemini model table, ordered most-specific first.
pub(super) static MODELS: &[ModelEntry] = &[
    // ----- Gemini 3.x — supports schema + tools (preview as of 2026-06) -----
    (Prefix("gemini-3-flash"), caps(true, 200_000, 64_000)),
    (Prefix("gemini-3-pro"), caps(true, 1_000_000, 64_000)),
    (Prefix("gemini-3"), caps(true, 1_000_000, 64_000)),
    // ----- Gemini 2.5 -----
    (Prefix("gemini-2.5-flash"), caps(false, 1_048_576, 65_535)),
    (Prefix("gemini-2.5-pro"), caps(false, 1_048_576, 65_536)),
    (Prefix("gemini-2.5"), caps(false, 1_048_576, 65_535)),
    // ----- Gemini 2.0 -----
    (Prefix("gemini-2.0"), caps(false, 1_000_000, 8_192)),
    // ----- Gemini 1.5 -----
    (Prefix("gemini-1.5-pro"), caps(false, 2_000_000, 8192)),
    (Prefix("gemini-1.5-flash-8b"), caps(false, 1_000_000, 8192)),
    (Prefix("gemini-1.5-flash"), caps(false, 1_000_000, 8192)),
    (Prefix("gemini-1.5"), caps(false, 1_000_000, 8192)),
    // ----- Family catch-all -----
    (Prefix("gemini-"), caps(false, 1_000_000, 8192)),
];

/// Fallback when nothing in [`MODELS`] matches.
pub(super) const FALLBACK: Capabilities = caps(false, 1_000_000, 8192);
