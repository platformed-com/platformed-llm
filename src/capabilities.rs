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
//! prefix-routed by name (`gpt-*`/`o*`/`chatgpt-*` → OpenAI,
//! `gemini-*` → Google, `claude-*` → Anthropic). Providers whose
//! models don't follow such a namespace (local inference, custom
//! fine-tunes) override the trait method to report whatever they
//! actually support.
//!
//! Matchers use prefix matching from most-recent to oldest family, with
//! a final "unknown" fallback that returns the most-restrictive caps
//! for that family plus a `tracing::debug!`. New minor versions within
//! a known family inherit the family's caps automatically. Unknown
//! model names entirely (no family prefix match) fall back to
//! [`Capabilities::default`] (everything off) — the safe-by-default
//! stance for a model the library has no information about.

/// Feature support flags for a specific model.
///
/// Fields default to the most-restrictive value (`false`) so a default
/// [`Capabilities`] is safe to use anywhere — features must be
/// explicitly opted into per model.
///
/// Marked `#[non_exhaustive]` so the library can add new capability
/// flags in a minor release without breaking external callers'
/// struct-literal construction. Construct via [`Self::default`] +
/// field assignment, or via the per-family helpers
/// ([`Self::for_model`], [`Self::openai`], [`Self::google`],
/// [`Self::anthropic`]). Within this crate full struct-literal
/// construction is still permitted, so the matcher functions
/// continue to read naturally.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
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
        if m.starts_with("gpt-")
            || m.starts_with("o1")
            || m.starts_with("o3")
            || m.starts_with("o4")
            || m.starts_with("chatgpt-")
        {
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

    /// Capabilities for an OpenAI model name (Responses API).
    ///
    /// All modern OpenAI chat / responses models support native JSON
    /// mode, schema-constrained output, and combining schema with
    /// tools. The matcher is permissive: unknown `gpt-*` / `o*` /
    /// `chatgpt-*` IDs fall back to the full feature set.
    pub fn openai(model: &str) -> Self {
        let m = model.to_ascii_lowercase();
        if m.starts_with("gpt-")
            || m.starts_with("o1")
            || m.starts_with("o3")
            || m.starts_with("o4")
            || m.starts_with("chatgpt-")
        {
            return Self {
                native_json_mode: true,
                response_schema: true,
                response_schema_with_tools: true,
            };
        }
        tracing::debug!(
            model,
            "unknown OpenAI model; assuming permissive capabilities"
        );
        Self {
            native_json_mode: true,
            response_schema: true,
            response_schema_with_tools: true,
        }
    }

    /// Capabilities for a Google / Gemini model name.
    ///
    /// Per Google's docs as of 2026: the 3.x series supports combining
    /// `responseSchema` with function-calling tools (preview), while
    /// 2.5 / 2.0 / 1.5 support schema-constrained output but **not**
    /// in combination with tools. Unknown Gemini IDs fall back to the
    /// 2.5 baseline (the most-restrictive *currently shipping* family)
    /// with a `tracing::debug!`.
    pub fn google(model: &str) -> Self {
        let m = model.to_ascii_lowercase();
        if m.starts_with("gemini-3") {
            return Self {
                native_json_mode: true,
                response_schema: true,
                response_schema_with_tools: true,
            };
        }
        if m.starts_with("gemini-2") || m.starts_with("gemini-1.5") {
            return Self {
                native_json_mode: true,
                response_schema: true,
                response_schema_with_tools: false,
            };
        }
        tracing::debug!(
            model,
            "unknown Gemini model; falling back to 2.5-series capabilities"
        );
        Self {
            native_json_mode: true,
            response_schema: true,
            response_schema_with_tools: false,
        }
    }

    /// Capabilities for an Anthropic Claude model name (via Vertex or
    /// direct API).
    ///
    /// Anthropic has no native JSON mode and no native schema-constrained
    /// output on any model — structured output is expressed by defining
    /// a function tool whose `input_schema` is the target shape. Every
    /// known Claude model therefore reports all JSON-related capabilities
    /// as `false`.
    pub fn anthropic(model: &str) -> Self {
        let m = model.to_ascii_lowercase();
        if !(m.starts_with("claude-") || m.contains("claude")) {
            tracing::debug!(
                model,
                "unknown Anthropic model; falling back to baseline capabilities"
            );
        }
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_most_restrictive() {
        let c = Capabilities::default();
        assert!(!c.native_json_mode);
        assert!(!c.response_schema);
        assert!(!c.response_schema_with_tools);
    }

    #[test]
    fn openai_known_models_are_permissive() {
        for model in [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-5",
            "gpt-5-mini",
            "o1",
            "o3-mini",
            "o4-mini",
            "chatgpt-4o-latest",
        ] {
            let c = Capabilities::openai(model);
            assert!(c.native_json_mode, "{model}: native_json_mode");
            assert!(c.response_schema, "{model}: response_schema");
            assert!(
                c.response_schema_with_tools,
                "{model}: response_schema_with_tools"
            );
        }
    }

    #[test]
    fn google_three_series_supports_schema_with_tools() {
        for model in [
            "gemini-3.1-pro-preview",
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
        ] {
            let c = Capabilities::google(model);
            assert!(c.response_schema, "{model}: response_schema");
            assert!(
                c.response_schema_with_tools,
                "{model}: response_schema_with_tools"
            );
        }
    }

    #[test]
    fn google_two_series_blocks_schema_with_tools() {
        for model in [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
        ] {
            let c = Capabilities::google(model);
            assert!(c.response_schema, "{model}: response_schema");
            assert!(
                !c.response_schema_with_tools,
                "{model}: must not claim schema+tools support"
            );
        }
    }

    #[test]
    fn google_unknown_falls_back_to_restrictive_baseline() {
        let c = Capabilities::google("gemini-experimental-foo");
        assert!(c.response_schema);
        assert!(!c.response_schema_with_tools);
    }

    #[test]
    fn anthropic_has_no_native_json_anywhere() {
        for model in [
            "claude-3-5-sonnet",
            "claude-3-7-sonnet",
            "claude-sonnet-4-5",
            "claude-opus-4-7",
            "claude-haiku-4-5",
        ] {
            let c = Capabilities::anthropic(model);
            assert!(!c.native_json_mode, "{model}: native_json_mode");
            assert!(!c.response_schema, "{model}: response_schema");
            assert!(
                !c.response_schema_with_tools,
                "{model}: response_schema_with_tools"
            );
        }
    }

    #[test]
    fn for_model_dispatches_to_correct_family() {
        // OpenAI namespace
        assert!(Capabilities::for_model("gpt-5").response_schema_with_tools);
        assert!(Capabilities::for_model("o4-mini").response_schema_with_tools);
        assert!(Capabilities::for_model("chatgpt-4o-latest").response_schema_with_tools);

        // Google namespace — 3.x permissive
        assert!(Capabilities::for_model("gemini-3.5-flash").response_schema_with_tools);
        // 2.5 restricted
        assert!(!Capabilities::for_model("gemini-2.5-pro").response_schema_with_tools);

        // Anthropic namespace
        assert!(!Capabilities::for_model("claude-sonnet-4-5").native_json_mode);

        // Unknown model — empty caps
        let unknown = Capabilities::for_model("mistral-7b-instruct");
        assert_eq!(unknown, Capabilities::default());
    }

    #[test]
    fn matcher_is_case_insensitive() {
        assert_eq!(Capabilities::openai("GPT-5"), Capabilities::openai("gpt-5"));
        assert_eq!(
            Capabilities::google("Gemini-3.5-Flash"),
            Capabilities::google("gemini-3.5-flash")
        );
    }
}
