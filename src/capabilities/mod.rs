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

/// How a `ModelEntry` matches a model name.
///
/// Lookups are case-insensitive — names are lowercased before
/// comparison so `"GPT-4o"` and `"gpt-4o"` resolve to the same caps.
#[derive(Debug, Clone, Copy)]
pub enum ModelMatch {
    /// Match if `model.to_ascii_lowercase() == self.0`.
    Exact(&'static str),
    /// Match if the lower-cased model name starts with `self.0`
    /// *and* the prefix ends on a word boundary — i.e. either the
    /// model name is exactly the prefix, or the next character is
    /// non-alphanumeric (`-`, `.`, `_`, `:`, `@`, `/`, …).
    ///
    /// The boundary check is only enforced when the prefix itself
    /// ends in an alphanumeric character. If the prefix ends in a
    /// separator (e.g. `"gpt-"`) anything can follow.
    ///
    /// Concretely:
    /// - `Prefix("o1")` matches `o1`, `o1-mini`, `o1.preview`;
    ///   it does **not** match `o12`, `o100`, `o1foo`.
    /// - `Prefix("gpt-")` matches `gpt-5`, `gpt-anything` — the
    ///   trailing `-` already encodes the boundary.
    /// - `Prefix("gpt-4")` matches `gpt-4`, `gpt-4-turbo`; it does
    ///   not match `gpt-4o` or `gpt-4.1` (those are alphanumeric
    ///   continuations of `gpt-4`).
    ///
    /// This lets the tables encode "the model is exactly `o1` or a
    /// versioned variant" without colliding with `o10`/`o11`/`o100`.
    Prefix(&'static str),
}

impl ModelMatch {
    fn matches(self, lowered: &str) -> bool {
        match self {
            ModelMatch::Exact(s) => lowered == s,
            ModelMatch::Prefix(s) => {
                if !lowered.starts_with(s) {
                    return false;
                }
                // The boundary check only kicks in when the prefix
                // ends mid-word (its last char is alphanumeric). A
                // prefix ending in a separator already has its
                // boundary baked in.
                let ends_in_word = s
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_ascii_alphanumeric());
                if !ends_in_word {
                    return true;
                }
                // Either the model name IS the prefix (no next char),
                // or the next char is non-alphanumeric.
                match lowered[s.len()..].chars().next() {
                    None => true,
                    Some(c) => !c.is_ascii_alphanumeric(),
                }
            }
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
    /// File-input support — which modalities the model accepts and
    /// whether the provider exposes a Files API for uploads. Consulted
    /// by each provider's part mapping: a [`crate::UserPart::File`]
    /// whose mime-derived modality is `false` here is an error (files
    /// are payload), not a silent drop.
    pub files: FileCapabilities,
}

/// Per-model file-input capability flags.
///
/// `upload` reports whether [`crate::Provider::upload`] can hit a real
/// Files API for this provider; the four modality flags report which
/// kinds of [`crate::UserPart::File`] the model accepts as input. All
/// default to `false` (the most-restrictive stance — unknown models
/// support nothing) to match [`Capabilities::default`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FileCapabilities {
    /// Provider exposes a Files API that [`crate::Provider::upload`] can
    /// use. `false` means `upload` returns an unsupported error.
    pub upload: bool,
    /// Model accepts `image/*` inputs.
    pub image: bool,
    /// Model accepts `audio/*` inputs.
    pub audio: bool,
    /// Model accepts `video/*` inputs.
    pub video: bool,
    /// Model accepts document inputs (PDF and other non-media types).
    pub document: bool,
}

impl FileCapabilities {
    /// Whether the given [`crate::Modality`] is accepted as input.
    pub fn accepts(&self, modality: crate::Modality) -> bool {
        match modality {
            crate::Modality::Image => self.image,
            crate::Modality::Audio => self.audio,
            crate::Modality::Video => self.video,
            crate::Modality::Document => self.document,
        }
    }
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
            files: FileCapabilities {
                upload: false,
                image: false,
                audio: false,
                video: false,
                document: false,
            },
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
    /// - `gpt-*`, `chatgpt-*` → [`Self::openai`].
    /// - `o<digit>...` (o-series reasoning models — `o1`, `o3`,
    ///   `o4`, and any future `oN`) → [`Self::openai`].
    /// - `gemini-*` → [`Self::google`].
    /// - `claude-*` (or names containing `claude`) → [`Self::anthropic`].
    /// - otherwise → [`Self::default`] (everything off) with a debug log.
    ///
    /// The o-series check is intentionally `o + digit` rather than a
    /// bare `o` prefix so unrelated names that happen to start with
    /// `o` (e.g. `openai-future-model`, `oracle-x`) fall through to
    /// the unknown-model path instead of getting routed to the OpenAI
    /// matcher. This keeps `for_model` consistent with the
    /// per-family table walkers.
    ///
    /// In practice these namespaces don't collide across providers, so
    /// prefix routing is reliable; callers using a fine-tune or a model
    /// the table doesn't recognize should override
    /// [`crate::Provider::capabilities`] on their provider instead.
    pub fn for_model(model: &str) -> Self {
        let m = model.to_ascii_lowercase();
        if m.starts_with("gpt-") || m.starts_with("chatgpt-") || is_openai_o_series(&m) {
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

    /// Capabilities for an OpenAI model name. Walks the `openai`
    /// submodule's `MODELS` table; falls back to its `FALLBACK`
    /// constant on no match.
    pub fn openai(model: &str) -> Self {
        lookup(model, openai::MODELS, openai::FALLBACK, "OpenAI")
    }

    /// Capabilities for a Google / Gemini model name. Walks the
    /// `google` submodule's table.
    pub fn google(model: &str) -> Self {
        lookup(model, google::MODELS, google::FALLBACK, "Google")
    }

    /// Capabilities for an Anthropic Claude model name. Walks the
    /// `anthropic` submodule's table.
    pub fn anthropic(model: &str) -> Self {
        lookup(model, anthropic::MODELS, anthropic::FALLBACK, "Anthropic")
    }
}

/// `true` when `lowered` looks like an OpenAI o-series reasoning
/// model name — `o` followed by at least one digit (`o1`, `o3-mini`,
/// `o4-mini-2025-…`). Used by [`Capabilities::for_model`] dispatch
/// and by the OPENAI table's catch-all entry so the two paths agree.
fn is_openai_o_series(lowered: &str) -> bool {
    let mut chars = lowered.chars();
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
// Each submodule owns one family's `MODELS` table + `FALLBACK`
// constant. Values are sourced from each provider's public docs as of
// 2026-06 — refresh when adding a model and keep the per-row
// comments accurate so the next audit goes faster.
//
// Entries are ordered MOST-SPECIFIC FIRST in each table: the walker
// returns the first hit, so put longer / more specific prefixes
// before shorter ones, and `Exact` before `Prefix` when both could
// match the same string.
// =====================================================================

mod anthropic;
mod google;
mod openai;

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
        // Truly novel name not matching any table row: falls through
        // the table to openai::FALLBACK. The `Prefix("o")` catch-all
        // was deliberately removed so `openai-future-model` no longer
        // collides with the o-series.
        let c = Capabilities::openai("openai-future-model");
        assert_eq!(c, openai::FALLBACK);
        // A novel future o-series variant also falls to the fallback
        // (same caps as the openai::FALLBACK constant, but reaches it
        // via the explicit fallback path rather than over-matching).
        let c = Capabilities::openai("o9-future-mini");
        assert_eq!(c, openai::FALLBACK);
    }

    /// `Capabilities::for_model("oN…")` must route to `openai()` for
    /// every `N` that's a digit — covers o-series releases the
    /// dispatch list doesn't enumerate explicitly. The contract is
    /// "any future `oN` model gets OpenAI caps from the table walker,
    /// not the all-zeros default."
    #[test]
    fn for_model_routes_all_o_series_to_openai() {
        for m in [
            "o1",
            "o2",
            "o3",
            "o4",
            "o5",
            "o6",
            "o7",
            "o9-mini",
            "o3-2025-04-16",
        ] {
            let via_for_model = Capabilities::for_model(m);
            let via_direct = Capabilities::openai(m);
            assert_eq!(
                via_for_model, via_direct,
                "{m}: for_model and openai() must agree"
            );
            assert!(
                via_for_model.context_window_tokens >= 128_000,
                "{m}: expected OpenAI-class caps, got {via_for_model:?}"
            );
        }
    }

    /// Names that look o-series-ish but aren't (e.g.
    /// `openai-experimental`, `oracle-x`) must NOT route to the
    /// OpenAI matcher — they fall to the unknown-model default.
    #[test]
    fn for_model_rejects_non_digit_o_prefix() {
        for m in ["openai-future-model", "oracle-x", "octopus", "o"] {
            assert_eq!(
                Capabilities::for_model(m),
                Capabilities::default(),
                "{m}: should be treated as unknown"
            );
        }
    }

    /// Legacy `gpt-4-*` variants whose context isn't 8k must get
    /// their real value (not the gpt-4 family fallback's 8192).
    #[test]
    fn legacy_gpt_4_variants_get_real_context() {
        // 32k variant family.
        for m in ["gpt-4-32k", "gpt-4-32k-0613"] {
            assert_eq!(
                Capabilities::openai(m).context_window_tokens,
                32_768,
                "{m}: gpt-4-32k variant"
            );
        }
        // 128k preview / vision variants.
        for m in [
            "gpt-4-1106-preview",
            "gpt-4-0125-preview",
            "gpt-4-vision-preview",
            "gpt-4-turbo-preview",
            "gpt-4-turbo-2024-04-09",
        ] {
            assert_eq!(
                Capabilities::openai(m).context_window_tokens,
                128_000,
                "{m}: 128k turbo/preview variant"
            );
        }
        // Bare `gpt-4` and `gpt-4-0613` stay at 8k.
        for m in ["gpt-4", "gpt-4-0613"] {
            assert_eq!(
                Capabilities::openai(m).context_window_tokens,
                8192,
                "{m}: 8k legacy"
            );
        }
    }

    /// Per the table doc-comment, 4.6+ models stay at the no-beta
    /// default of 200k context — the 1M beta isn't on by default and
    /// callers opting in must override caps on their Provider.
    #[test]
    fn anthropic_4_6_plus_stay_at_200k_no_beta() {
        for m in [
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
        ] {
            let c = Capabilities::anthropic(m);
            assert_eq!(
                c.context_window_tokens, 200_000,
                "{m}: should default to no-beta context"
            );
            assert_eq!(c.max_output_tokens, 128_000, "{m}");
        }
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

    /// `Prefix("o1")` must match `o1` and `o1-mini`, but NOT `o12`
    /// or `o100` — the boundary keeps short numeric prefixes from
    /// accidentally swallowing wider numeric ranges.
    #[test]
    fn prefix_match_respects_word_boundary() {
        let m = ModelMatch::Prefix("o1");
        assert!(m.matches("o1"));
        assert!(m.matches("o1-mini"));
        assert!(m.matches("o1.preview"));
        assert!(m.matches("o1_v2"));
        assert!(!m.matches("o12"));
        assert!(!m.matches("o100-mini"));
        assert!(!m.matches("o1foo"));
    }

    /// `Prefix("gpt-")` ends in a separator already, so any next
    /// character (including alphanumeric) is allowed — otherwise
    /// the catch-all rows would stop catching real models like
    /// `gpt-5` or `gpt-4o`.
    #[test]
    fn prefix_match_lets_separator_prefix_match_anything() {
        let m = ModelMatch::Prefix("gpt-");
        assert!(m.matches("gpt-5"));
        assert!(m.matches("gpt-x"));
        assert!(m.matches("gpt-anything"));
        assert!(m.matches("gpt-"));
    }

    /// `Prefix("gpt-4")` must NOT swallow `gpt-4o` or `gpt-4.1` —
    /// they're distinct model families that share a leading
    /// substring. Previously this only worked because of table
    /// ordering; now the boundary check enforces it independent of
    /// order.
    #[test]
    fn prefix_match_does_not_swallow_alphanumeric_extensions() {
        let m = ModelMatch::Prefix("gpt-4");
        assert!(m.matches("gpt-4"));
        assert!(m.matches("gpt-4-turbo"));
        assert!(m.matches("gpt-4-0613"));
        // `gpt-4o` is a *different* family and must NOT match.
        assert!(!m.matches("gpt-4o"));
        assert!(!m.matches("gpt-4o-mini"));
        // `gpt-4.1` is also a different family (next char `.` is non-
        // alphanumeric → it WOULD match). We want it to match because
        // there's a more-specific row for it, and `Prefix("gpt-4")`
        // is a reasonable fallback if that row were missing.
        assert!(m.matches("gpt-4.1"));
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

    #[test]
    fn default_files_all_false() {
        let f = Capabilities::default().files;
        assert!(!f.upload);
        assert!(!f.image);
        assert!(!f.audio);
        assert!(!f.video);
        assert!(!f.document);
    }

    /// OpenAI: real Files API + image / audio / document inputs; no video.
    #[test]
    fn openai_files_caps() {
        for m in ["gpt-4o", "gpt-5", "o4-mini"] {
            let f = Capabilities::openai(m).files;
            assert!(f.upload, "{m}: upload");
            assert!(f.image, "{m}: image");
            assert!(f.audio, "{m}: audio");
            assert!(f.document, "{m}: document");
            assert!(!f.video, "{m}: video unsupported");
        }
    }

    /// Gemini (via Vertex): multimodal inputs but NO upload (no Vertex
    /// Files API — files go by `gs://` URL).
    #[test]
    fn google_files_caps() {
        for m in ["gemini-2.5-pro", "gemini-3-flash"] {
            let f = Capabilities::google(m).files;
            assert!(!f.upload, "{m}: no Vertex Files API");
            assert!(f.image, "{m}: image");
            assert!(f.audio, "{m}: audio");
            assert!(f.video, "{m}: video");
            assert!(f.document, "{m}: document");
        }
    }

    /// Anthropic (via Vertex): image + document only, no audio/video, no
    /// upload (Files API not on Vertex).
    #[test]
    fn anthropic_files_caps() {
        for m in ["claude-sonnet-4-5", "claude-opus-4-8"] {
            let f = Capabilities::anthropic(m).files;
            assert!(!f.upload, "{m}: no Vertex Files API");
            assert!(f.image, "{m}: image");
            assert!(f.document, "{m}: document");
            assert!(!f.audio, "{m}: audio unsupported");
            assert!(!f.video, "{m}: video unsupported");
        }
    }

    #[test]
    fn file_capabilities_accepts_maps_modalities() {
        use crate::Modality;
        let f = Capabilities::anthropic("claude-sonnet-4-5").files;
        assert!(f.accepts(Modality::Image));
        assert!(f.accepts(Modality::Document));
        assert!(!f.accepts(Modality::Audio));
        assert!(!f.accepts(Modality::Video));
    }
}
