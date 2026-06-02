//! Per-model capability table.
//!
//! Providers expose families of models with different feature support
//! — e.g. Gemini 3.x can combine `responseSchema` with function-calling
//! tools but Gemini 2.5 cannot; Anthropic has no native JSON mode at
//! all. [`Capabilities`] is the factual answer to *"what does this
//! model support?"*; what to *do* about the answer (drop the field,
//! warn, polyfill via tool-coercion, …) is the policy decision
//! applied by [`crate::middleware`] above the provider layer.
//!
//! Resolution: at [`crate::generate`] time the active
//! [`crate::Provider`] is asked via
//! [`crate::Provider::capabilities`]. Hosted providers inherit the
//! default impl which delegates to [`Capabilities::for_model`] —
//! which dispatches into a per-family lookup table.
//!
//! The tables (one per family — OpenAI, Google, Anthropic) hold an
//! ordered list of `(ModelMatch, Capabilities)` entries. The walker
//! tries each in order and returns the first hit; entries should be
//! ordered most-specific first (`Exact` before `Prefix`, longer
//! prefixes before shorter). A final catch-all entry covers
//! family-shaped names the table doesn't know about. Adding a new
//! model = appending one row.

/// How a [`ModelEntry`] matches a model name.
///
/// Lookups are case-insensitive — names are lowercased before
/// comparison so `"GPT-4o"` and `"gpt-4o"` resolve to the same caps.
#[derive(Debug, Clone, Copy)]
pub enum ModelMatch {
    /// Match if `model.to_ascii_lowercase() == self.0`.
    Exact(&'static str),
    /// Match if `model.to_ascii_lowercase().starts_with(self.0)`.
    Prefix(&'static str),
}

impl ModelMatch {
    fn matches(self, lowered: &str) -> bool {
        match self {
            ModelMatch::Exact(s) => lowered == s,
            ModelMatch::Prefix(s) => lowered.starts_with(s),
        }
    }
}

/// One row in a per-family capability table.
type ModelEntry = (ModelMatch, Capabilities);

/// Feature support flags for a specific model.
///
/// Boolean fields default to the most-restrictive value (`false`);
/// numeric token-limit fields default to deliberately conservative
/// values (see [`Self::default`]) so the headroom helpers err on the
/// side of triggering compaction earlier than necessary when the
/// caller hasn't picked specific values.
///
/// Marked `#[non_exhaustive]` so the library can add new capability
/// flags in a minor release without breaking external callers'
/// struct-literal construction. Construct via [`Self::default`] +
/// field assignment, or via the per-family helpers
/// ([`Self::for_model`], [`Self::openai`], [`Self::google`],
/// [`Self::anthropic`]). Within this crate full struct-literal
/// construction is still permitted, so the matcher tables can read
/// naturally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Capabilities {
    /// Model natively supports bare-JSON output (no schema enforcement).
    /// Maps to `response_mime_type: application/json` on Gemini and
    /// `{"type":"json_object"}` on OpenAI.
    pub native_json_mode: bool,
    /// Model natively supports schema-constrained output. Maps to
    /// `responseSchema` on Gemini and `{"type":"json_schema",...}` on
    /// OpenAI.
    pub response_schema: bool,
    /// Schema-constrained output may be combined with function-calling
    /// tools in the same request. Gemini's documented sharp edge: only
    /// the 3.x family supports this combination.
    pub response_schema_with_tools: bool,
    /// Total context-window size (input + output combined) in tokens.
    /// Used by compaction heuristics — see
    /// [`Self::context_usage_fraction`] and
    /// [`Self::would_exceed_context`].
    pub context_window_tokens: u32,
    /// Hard cap on output tokens in a single response. Most providers
    /// cap output well below the full context window; setting
    /// `max_tokens` higher than this is a caller error that will
    /// surface server-side.
    pub max_output_tokens: u32,
}

impl Default for Capabilities {
    /// Most-restrictive defaults: no native JSON / schema support, and
    /// conservative token windows (`4096` context, `1024` output) that
    /// roughly match the smallest model families anyone is still
    /// using. Always overriding-friendly — the headroom helpers
    /// against these values err on the side of triggering compaction
    /// earlier than necessary, which is the safe direction for a
    /// fallback.
    fn default() -> Self {
        Self {
            native_json_mode: false,
            response_schema: false,
            response_schema_with_tools: false,
            context_window_tokens: 4096,
            max_output_tokens: 1024,
        }
    }
}

impl Capabilities {
    /// Fraction of the context window consumed by `usage.input_tokens
    /// + usage.output_tokens` on the most recent turn.
    ///
    /// A value approaching 1.0 means the *next* turn — which has to
    /// re-send this turn's history plus the assistant's response — is
    /// at risk of either being truncated or rejected for exceeding
    /// the context window. Typical compaction trigger is
    /// `fraction > 0.7` or so.
    pub fn context_usage_fraction(&self, usage: &crate::Usage) -> f32 {
        if self.context_window_tokens == 0 {
            return f32::INFINITY;
        }
        usage.total_tokens() as f32 / self.context_window_tokens as f32
    }

    /// `true` if `tokens` strictly exceeds the model's
    /// [`Self::context_window_tokens`].
    pub fn would_exceed_context(&self, tokens: u32) -> bool {
        tokens > self.context_window_tokens
    }
}

impl Capabilities {
    /// Provider-agnostic entry point. Dispatches by model-name prefix:
    /// - `gpt-*`, `o1*`, `o3*`, `o4*`, `chatgpt-*` → [`Self::openai`].
    /// - `gemini-*` → [`Self::google`].
    /// - `claude-*` (any case) → [`Self::anthropic`].
    /// - otherwise → [`Self::default`] (everything off) with a debug log.
    ///
    /// In practice these namespaces don't collide across providers, so
    /// prefix routing is reliable; callers using a fine-tune or a model
    /// the table doesn't recognize should override
    /// [`crate::Provider::capabilities`] on their provider instead.
    pub fn for_model(model: &str) -> Self {
        let m = model.to_ascii_lowercase();
        if m.starts_with("gpt-") || is_openai_o_series(&m) || m.starts_with("chatgpt-") {
            return Self::openai(model);
        }
        if m.starts_with("gemini-") {
            return Self::google(model);
        }
        if m.starts_with("claude-") || m.contains("claude") {
            return Self::anthropic(model);
        }
        tracing::debug!(
            model,
            "model name does not match any known provider family; assuming no capabilities"
        );
        Self::default()
    }

    /// Capabilities for an OpenAI model name. Walks [`OPENAI_MODELS`];
    /// falls back to `OPENAI_FALLBACK` on no match.
    pub fn openai(model: &str) -> Self {
        lookup(model, OPENAI_MODELS, OPENAI_FALLBACK, "OpenAI")
    }

    /// Capabilities for a Google / Gemini model name. Walks
    /// [`GOOGLE_MODELS`]; falls back to `GOOGLE_FALLBACK` on no match.
    pub fn google(model: &str) -> Self {
        lookup(model, GOOGLE_MODELS, GOOGLE_FALLBACK, "Google")
    }

    /// Capabilities for an Anthropic Claude model name. Walks
    /// [`ANTHROPIC_MODELS`]; falls back to `ANTHROPIC_FALLBACK` on no
    /// match.
    pub fn anthropic(model: &str) -> Self {
        lookup(model, ANTHROPIC_MODELS, ANTHROPIC_FALLBACK, "Anthropic")
    }
}

/// True for OpenAI reasoning ("o-series") model names: a leading `o`
/// immediately followed by a digit (`o1`, `o3`, `o4-mini`, and any
/// future `o5`/`o6`/…). Matching the digit rather than enumerating
/// known versions means a newly released o-series model routes to the
/// permissive OpenAI capability set instead of falling through to the
/// all-off default (which would spuriously trigger JSON coercion on a
/// model with native schema support). `model` is assumed already
/// lowercased.
fn is_openai_o_series(model: &str) -> bool {
    let mut chars = model.chars();
    chars.next() == Some('o') && chars.next().is_some_and(|c| c.is_ascii_digit())
}

/// Walk `table` in order, returning the caps of the first matching
/// row. Falls back to `default` with a debug log if no row matches —
/// callers reach this when the model name is unknown within an
/// otherwise-recognized family namespace (e.g. an OpenAI fine-tune).
fn lookup(
    model: &str,
    table: &[ModelEntry],
    default: Capabilities,
    family: &'static str,
) -> Capabilities {
    let lowered = model.to_ascii_lowercase();
    for (matcher, caps) in table {
        if matcher.matches(&lowered) {
            return *caps;
        }
    }
    tracing::debug!(
        model,
        family,
        "no model-table entry matched; using family fallback caps"
    );
    default
}

// =====================================================================
// Per-family capability tables
//
// Entries are ordered MOST-SPECIFIC FIRST. The walker takes the first
// hit, so put longer / more specific prefixes before shorter ones, and
// `Exact` before `Prefix` when both could match the same string.
//
// Values are sourced from each provider's public docs as of 2026-06.
// They're under active drift — refresh when adding a model and please
// keep the references in the per-row comments accurate so the next
// audit goes faster.
// =====================================================================

use ModelMatch::{Exact, Prefix};

// ---------- OpenAI ----------

/// Build an OpenAI capabilities entry. Every modern OpenAI Chat /
/// Responses model supports native JSON mode, JSON schema, and schema
/// + tools combined; only the token limits vary.
const fn openai_caps(context: u32, output: u32) -> Capabilities {
    Capabilities {
        native_json_mode: true,
        response_schema: true,
        response_schema_with_tools: true,
        context_window_tokens: context,
        max_output_tokens: output,
    }
}

/// OpenAI model table, ordered most-specific first.
static OPENAI_MODELS: &[ModelEntry] = &[
    // ----- GPT-5 family (released 2025; gpt-5.5 added 2026-04) -----
    (Prefix("gpt-5.5"), openai_caps(1_050_000, 128_000)),
    (Prefix("gpt-5.4-mini"), openai_caps(400_000, 128_000)),
    (Prefix("gpt-5.4-nano"), openai_caps(400_000, 128_000)),
    (Prefix("gpt-5.4"), openai_caps(1_050_000, 128_000)),
    (Prefix("gpt-5-mini"), openai_caps(400_000, 128_000)),
    (Prefix("gpt-5-nano"), openai_caps(400_000, 128_000)),
    (Prefix("gpt-5"), openai_caps(400_000, 128_000)),
    // ----- GPT-4.1 family (1M context) -----
    (Prefix("gpt-4.1"), openai_caps(1_047_576, 32_768)),
    // ----- GPT-4o family -----
    (Prefix("gpt-4o-mini"), openai_caps(128_000, 16_384)),
    (Prefix("gpt-4o"), openai_caps(128_000, 16_384)),
    (Prefix("chatgpt-4o"), openai_caps(128_000, 16_384)),
    // ----- GPT-4 turbo / legacy -----
    (Prefix("gpt-4-turbo"), openai_caps(128_000, 4096)),
    (Exact("gpt-4"), openai_caps(8192, 8192)),
    (Prefix("gpt-4-"), openai_caps(8192, 8192)),
    // ----- o-series reasoning models -----
    (Prefix("o1-mini"), openai_caps(128_000, 65_536)),
    (Prefix("o1-preview"), openai_caps(128_000, 32_768)),
    (Prefix("o1"), openai_caps(200_000, 100_000)),
    (Prefix("o3-mini"), openai_caps(200_000, 100_000)),
    (Prefix("o3"), openai_caps(200_000, 100_000)),
    (Prefix("o4-mini"), openai_caps(200_000, 100_000)),
    (Prefix("o4"), openai_caps(200_000, 100_000)),
    // ----- Family catch-all -----
    // Anything still starting with `gpt-` / `chatgpt-` / `o*` falls
    // here. Conservative numbers + full feature flags.
    (Prefix("gpt-"), openai_caps(128_000, 16_384)),
    (Prefix("chatgpt-"), openai_caps(128_000, 16_384)),
    (Prefix("o"), openai_caps(128_000, 16_384)),
];

/// Fallback when nothing in [`OPENAI_MODELS`] matches.
const OPENAI_FALLBACK: Capabilities = openai_caps(128_000, 16_384);

// ---------- Google / Gemini ----------

/// Build a Gemini capabilities entry with the supplied feature /
/// limit combination.
const fn gemini_caps(schema_with_tools: bool, context: u32, output: u32) -> Capabilities {
    Capabilities {
        native_json_mode: true,
        response_schema: true,
        response_schema_with_tools: schema_with_tools,
        context_window_tokens: context,
        max_output_tokens: output,
    }
}

/// Google / Gemini model table, ordered most-specific first.
static GOOGLE_MODELS: &[ModelEntry] = &[
    // ----- Gemini 3.x — supports schema + tools (preview as of 2026-06) -----
    (Prefix("gemini-3-flash"), gemini_caps(true, 200_000, 64_000)),
    (Prefix("gemini-3-pro"), gemini_caps(true, 1_000_000, 64_000)),
    (Prefix("gemini-3"), gemini_caps(true, 1_000_000, 64_000)),
    // ----- Gemini 2.5 -----
    (
        Prefix("gemini-2.5-flash"),
        gemini_caps(false, 1_048_576, 65_535),
    ),
    (
        Prefix("gemini-2.5-pro"),
        gemini_caps(false, 1_048_576, 65_536),
    ),
    (Prefix("gemini-2.5"), gemini_caps(false, 1_048_576, 65_535)),
    // ----- Gemini 2.0 -----
    (Prefix("gemini-2.0"), gemini_caps(false, 1_000_000, 8_192)),
    // ----- Gemini 1.5 -----
    (
        Prefix("gemini-1.5-pro"),
        gemini_caps(false, 2_000_000, 8192),
    ),
    (
        Prefix("gemini-1.5-flash-8b"),
        gemini_caps(false, 1_000_000, 8192),
    ),
    (
        Prefix("gemini-1.5-flash"),
        gemini_caps(false, 1_000_000, 8192),
    ),
    (Prefix("gemini-1.5"), gemini_caps(false, 1_000_000, 8192)),
    // ----- Family catch-all -----
    (Prefix("gemini-"), gemini_caps(false, 1_000_000, 8192)),
];

/// Fallback when nothing in [`GOOGLE_MODELS`] matches.
const GOOGLE_FALLBACK: Capabilities = gemini_caps(false, 1_000_000, 8192);

// ---------- Anthropic ----------

/// Build an Anthropic capabilities entry. Anthropic has no native JSON
/// mode; structured output is expressed via tool-use coercion (see
/// [`crate::JsonCoercionMiddleware`]).
const fn anthropic_caps(context: u32, output: u32) -> Capabilities {
    Capabilities {
        native_json_mode: false,
        response_schema: false,
        response_schema_with_tools: false,
        context_window_tokens: context,
        max_output_tokens: output,
    }
}

/// Anthropic Claude model table, ordered most-specific first.
static ANTHROPIC_MODELS: &[ModelEntry] = &[
    // ----- Claude 4.x Opus (1M context on 4.6+, 128k output) -----
    (
        Prefix("claude-opus-4-8"),
        anthropic_caps(1_000_000, 128_000),
    ),
    (
        Prefix("claude-opus-4-7"),
        anthropic_caps(1_000_000, 128_000),
    ),
    (
        Prefix("claude-opus-4-6"),
        anthropic_caps(1_000_000, 128_000),
    ),
    (Prefix("claude-opus-4-1"), anthropic_caps(200_000, 32_000)),
    (Prefix("claude-opus-4"), anthropic_caps(200_000, 32_000)),
    // ----- Claude 4.x Sonnet (1M context on 4.6, otherwise 200k) -----
    (
        Prefix("claude-sonnet-4-6"),
        anthropic_caps(1_000_000, 128_000),
    ),
    (Prefix("claude-sonnet-4-5"), anthropic_caps(200_000, 64_000)),
    (Prefix("claude-sonnet-4"), anthropic_caps(200_000, 64_000)),
    // ----- Claude 4.5 Haiku -----
    (Prefix("claude-haiku-4-5"), anthropic_caps(200_000, 64_000)),
    // ----- Claude 3.7 / 3.5 -----
    (Prefix("claude-3-7-sonnet"), anthropic_caps(200_000, 64_000)),
    (Prefix("claude-3-5-sonnet"), anthropic_caps(200_000, 8192)),
    (Prefix("claude-3-5-haiku"), anthropic_caps(200_000, 8192)),
    // ----- Claude 3 (legacy) -----
    (Prefix("claude-3"), anthropic_caps(200_000, 4096)),
    // ----- Family catch-all -----
    (Prefix("claude-"), anthropic_caps(200_000, 8192)),
];

/// Fallback when nothing in [`ANTHROPIC_MODELS`] matches.
const ANTHROPIC_FALLBACK: Capabilities = anthropic_caps(200_000, 8192);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_most_restrictive() {
        let c = Capabilities::default();
        assert!(!c.native_json_mode);
        assert!(!c.response_schema);
        assert!(!c.response_schema_with_tools);
        assert_eq!(c.context_window_tokens, 4096);
        assert_eq!(c.max_output_tokens, 1024);
    }

    #[test]
    fn openai_exact_known_models() {
        // gpt-4 (legacy) — Exact entry; must NOT pick up gpt-4o caps.
        let c = Capabilities::openai("gpt-4");
        assert_eq!(c.context_window_tokens, 8192);
        assert_eq!(c.max_output_tokens, 8192);

        // gpt-4o and gpt-4o-mini share caps.
        for m in ["gpt-4o", "gpt-4o-mini"] {
            let c = Capabilities::openai(m);
            assert_eq!(c.context_window_tokens, 128_000, "{m}");
            assert_eq!(c.max_output_tokens, 16_384, "{m}");
        }

        // gpt-5 family: 5.4 / 5.5 are 1.05M context; base 5 is 400k.
        assert_eq!(Capabilities::openai("gpt-5").context_window_tokens, 400_000);
        assert_eq!(
            Capabilities::openai("gpt-5.4").context_window_tokens,
            1_050_000
        );
        assert_eq!(
            Capabilities::openai("gpt-5.5").context_window_tokens,
            1_050_000
        );

        // gpt-4.1 — 1M context.
        assert_eq!(
            Capabilities::openai("gpt-4.1").context_window_tokens,
            1_047_576
        );
    }

    #[test]
    fn openai_prefix_fallback_for_version_suffix() {
        // Dated / pinned variants must match the family prefix entry.
        let c = Capabilities::openai("gpt-4o-2024-08-06");
        assert_eq!(c.context_window_tokens, 128_000);

        let c = Capabilities::openai("o3-2025-04-16");
        assert_eq!(c.context_window_tokens, 200_000);
    }

    #[test]
    fn openai_family_fallback_for_unknown_name() {
        // Truly novel name not matching any table row: falls through.
        let c = Capabilities::openai("openai-future-model");
        assert_eq!(c, OPENAI_FALLBACK);
    }

    #[test]
    fn google_three_series_supports_schema_with_tools() {
        for m in ["gemini-3-pro", "gemini-3-flash"] {
            let c = Capabilities::google(m);
            assert!(c.response_schema_with_tools, "{m}");
        }
    }

    #[test]
    fn google_two_five_blocks_schema_with_tools() {
        for m in ["gemini-2.5-flash", "gemini-2.5-pro"] {
            let c = Capabilities::google(m);
            assert!(!c.response_schema_with_tools, "{m}");
        }
    }

    #[test]
    fn google_token_limits_per_model() {
        // 2.5 pro / flash differ by 1 token of output (65535 vs 65536).
        assert_eq!(
            Capabilities::google("gemini-2.5-flash").max_output_tokens,
            65_535
        );
        assert_eq!(
            Capabilities::google("gemini-2.5-pro").max_output_tokens,
            65_536
        );
        // 1.5 pro has the legendary 2M window.
        assert_eq!(
            Capabilities::google("gemini-1.5-pro").context_window_tokens,
            2_000_000
        );
        // 3 pro is 1M; 3 flash drops to 200k.
        assert_eq!(
            Capabilities::google("gemini-3-pro").context_window_tokens,
            1_000_000
        );
        assert_eq!(
            Capabilities::google("gemini-3-flash").context_window_tokens,
            200_000
        );
    }

    #[test]
    fn anthropic_has_no_native_json_anywhere() {
        for m in [
            "claude-3-5-sonnet",
            "claude-3-7-sonnet",
            "claude-sonnet-4-5",
            "claude-opus-4-7",
            "claude-haiku-4-5",
        ] {
            let c = Capabilities::anthropic(m);
            assert!(!c.native_json_mode, "{m}");
            assert!(!c.response_schema, "{m}");
        }
    }

    #[test]
    fn anthropic_4_6_plus_have_million_token_context() {
        for m in [
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
        ] {
            let c = Capabilities::anthropic(m);
            assert_eq!(c.context_window_tokens, 1_000_000, "{m}");
            assert_eq!(c.max_output_tokens, 128_000, "{m}");
        }
    }

    #[test]
    fn anthropic_legacy_4_x_keeps_200k() {
        // 4.0 / 4.5 sonnet, 4.0 / 4.1 opus stay at 200k.
        assert_eq!(
            Capabilities::anthropic("claude-sonnet-4-5").context_window_tokens,
            200_000
        );
        assert_eq!(
            Capabilities::anthropic("claude-opus-4-1").context_window_tokens,
            200_000
        );
    }

    #[test]
    fn anthropic_family_fallback_for_unknown_name() {
        let c = Capabilities::anthropic("claude-experimental-future");
        // Falls through to `claude-` catch-all.
        assert_eq!(c.context_window_tokens, 200_000);
        assert_eq!(c.max_output_tokens, 8192);
    }

    #[test]
    fn for_model_dispatches_to_correct_family() {
        assert!(Capabilities::for_model("gpt-5").response_schema_with_tools);
        assert!(Capabilities::for_model("o4-mini").response_schema_with_tools);
        assert!(Capabilities::for_model("chatgpt-4o").response_schema_with_tools);
        assert!(Capabilities::for_model("gemini-3-pro").response_schema_with_tools);
        assert!(!Capabilities::for_model("gemini-2.5-pro").response_schema_with_tools);
        assert!(!Capabilities::for_model("claude-sonnet-4-5").native_json_mode);
        // Unknown model — empty caps.
        assert_eq!(
            Capabilities::for_model("mistral-7b-instruct"),
            Capabilities::default()
        );
    }

    #[test]
    fn matcher_is_case_insensitive() {
        assert_eq!(Capabilities::openai("GPT-5"), Capabilities::openai("gpt-5"));
        assert_eq!(
            Capabilities::google("Gemini-3-Pro"),
            Capabilities::google("gemini-3-pro")
        );
    }

    #[test]
    fn context_usage_fraction_computes_against_window() {
        use crate::Usage;
        let caps = Capabilities::openai("gpt-4o"); // 128k window
        let usage = Usage {
            input_tokens: 96_000,
            output_tokens: 16_000,
            ..Usage::default()
        };
        let frac = caps.context_usage_fraction(&usage);
        assert!((frac - 0.875).abs() < 0.001, "got {frac}");
    }

    #[test]
    fn context_usage_fraction_uses_default_window_when_unset() {
        use crate::Usage;
        let caps = Capabilities::default();
        let usage = Usage {
            input_tokens: 1000,
            output_tokens: 1000,
            ..Usage::default()
        };
        let frac = caps.context_usage_fraction(&usage);
        assert!((frac - (2000.0 / 4096.0)).abs() < 0.001, "got {frac}");
    }

    #[test]
    fn would_exceed_context_compares_to_window() {
        let caps = Capabilities::anthropic("claude-sonnet-4-5"); // 200k
        assert!(!caps.would_exceed_context(150_000));
        assert!(caps.would_exceed_context(200_001));
        assert!(!caps.would_exceed_context(200_000));
    }
    /// A future o-series model (`o5`, `o6`, …) must route to the
    /// permissive OpenAI caps rather than falling through to the
    /// all-off default — otherwise JSON coercion would fire spuriously
    /// and downgrade a model with native strict-schema support.
    #[test]
    fn for_model_routes_unknown_o_series_permissive() {
        for model in ["o5", "o5-mini", "o6-pro", "O7"] {
            let c = Capabilities::for_model(model);
            assert!(
                c.native_json_mode && c.response_schema && c.response_schema_with_tools,
                "{model}: should route to permissive OpenAI caps"
            );
        }
    }

    /// `is_openai_o_series` keys off `o` + digit, not a leading `o`
    /// alone — non-o-series names that merely start with `o` must not
    /// be misrouted to OpenAI.
    #[test]
    fn o_series_matcher_requires_digit() {
        assert!(is_openai_o_series("o1"));
        assert!(is_openai_o_series("o3-mini"));
        assert!(is_openai_o_series("o5"));
        assert!(!is_openai_o_series("opus"));
        assert!(!is_openai_o_series("o"));
        assert!(!is_openai_o_series("openai-thing"));
    }
}
