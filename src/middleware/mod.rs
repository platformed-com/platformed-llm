//! Request/response middleware applied above the [`crate::Provider`]
//! layer.
//!
//! The provider trait is intentionally dumb — it translates a
//! `(Prompt, RawConfig)` to whatever wire shape the upstream API
//! expects and emits stream events back. Anything that bridges a
//! *gap* between the caller's intent and what the model natively
//! supports — JSON coercion via tool-use, schema-vs-tools conflict
//! reconciliation, future retry/cache/redact behaviors — belongs here
//! as a [`Middleware`].
//!
//! Each concrete middleware lives in its own submodule (e.g.
//! [`crate::middleware::json_coercion`] for the JSON-via-tool-coercion
//! polyfill); this module owns the trait, the [`generate`] entry
//! point, the [`crate::middleware::validate`] post-middleware gate,
//! and the [`crate::middleware::default_middleware`] derivation from
//! a capability set.
//!
//! Pipeline (see [`generate`]):
//! 1. Borrow the caller's `prompt` and `config.raw()` as `Cow::Borrowed`.
//!    No clone yet.
//! 2. Walk each middleware's [`Middleware::apply`] in order. A
//!    middleware that wants to change something calls
//!    [`std::borrow::Cow::to_mut`] on the relevant Cow (cloning once,
//!    paying for the modification it actually makes); no-op middleware
//!    leave both Cows borrowed and add zero allocation.
//! 3. Run [`crate::middleware::validate`] against the *post-middleware*
//!    config. If a gap remains, error fast — the provider never sees
//!    an unsupported request.
//! 4. Call the provider with `&Prompt` and `&RawConfig` (Cow Deref).
//! 5. Apply response transforms in **reverse** order (onion: the last
//!    middleware to touch the request is the first to unwrap the
//!    response).

use std::borrow::Cow;
use std::sync::Arc;

use crate::provider::Provider;
use crate::types::{RawConfig, ResponseFormat, Tool};
use crate::{Capabilities, Error, Prompt, Response};

pub mod json_coercion;

pub use json_coercion::JsonCoercionMiddleware;

/// A response-stream wrapper produced by a middleware during request
/// rewriting. Captures any per-request state (e.g. the synthetic tool
/// name a polyfill injected) so the response side can undo the
/// rewrite.
pub type ResponseTransform = Box<dyn FnOnce(Response) -> Response + Send>;

/// Sits between the caller and the [`crate::Provider`]. Implementors
/// rewrite the outgoing request and optionally return a closure that
/// wraps the response stream to undo or interpret the rewrite.
///
/// Receives `&mut Cow<Prompt>` and `&mut Cow<RawConfig>`. The Cow
/// starts as `Borrowed` so a middleware that doesn't need to change
/// anything imposes zero overhead — return `Ok(None)` and the chain
/// continues with the original references. A middleware that does
/// need to change something calls `prompt.to_mut()` / `config.to_mut()`
/// which clones the underlying value once and yields a `&mut T` for
/// in-place edits.
///
/// `capabilities` is the resolved capability set for `config.model`
/// (from the provider, or a caller override). It's passed alongside
/// the request rather than embedded in `RawConfig` so the wire-level
/// payload stays clean — middleware that needs to know what the model
/// supports should consult this parameter.
pub trait Middleware: Send + Sync + std::fmt::Debug {
    /// Short human-readable name. Used in tracing / debug output only.
    fn name(&self) -> &str;

    /// Inspect / rewrite the outgoing request.
    ///
    /// Return `Ok(None)` if no response-side rewriting is needed. The
    /// `ResponseTransform` closure runs after the provider's response
    /// is available; it has full control over the event stream (and
    /// captures whatever state from request-rewriting time it needs).
    fn apply<'a>(
        &self,
        prompt: &mut Cow<'a, Prompt>,
        config: &mut Cow<'a, RawConfig>,
        capabilities: &Capabilities,
    ) -> Result<Option<ResponseTransform>, Error>;
}

/// Default middleware list derived from a capability set.
///
/// Any middleware whose presence depends only on the model's caps
/// (and not on the specific request) gets included unconditionally
/// when the cap is missing. Middleware are responsible for being
/// cheap when they have nothing to do — e.g. [`JsonCoercionMiddleware`]
/// no-ops when `response_format` is unset.
pub fn default_middleware(caps: &Capabilities) -> Vec<Arc<dyn Middleware>> {
    let mut out: Vec<Arc<dyn Middleware>> = Vec::new();
    if !caps.response_schema || !caps.response_schema_with_tools || !caps.native_json_mode {
        out.push(Arc::new(JsonCoercionMiddleware));
    }
    out
}

/// Validate that `config` is achievable given `caps`. Run **after**
/// middleware so a polyfill that already cleared / rewrote the
/// offending field passes through silently.
///
/// Returns `Err(Error::Config)` with a precise message when a gap
/// remains.
pub fn validate(config: &RawConfig, caps: &Capabilities) -> Result<(), Error> {
    // Vertex rejects *controlled generation of any form* combined with
    // function calling on the restricted Gemini families — the wire
    // error is literally "Function calling with a response mime type:
    // 'application/json' is unsupported". That covers both
    // `responseMimeType` (our `JsonObject`) and `responseSchema` (our
    // `JsonSchema`), so both must gate the combined-request check.
    // `response_schema_with_tools` is the flag for "controlled
    // generation may be combined with tools".
    let asks_for_schema = matches!(
        config.response_format,
        Some(ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. })
    );
    let has_function_tools = config
        .tools
        .as_ref()
        .is_some_and(|ts| ts.iter().any(|t| matches!(t, Tool::Function(_))));

    if let Some(ResponseFormat::JsonObject) = &config.response_format {
        if !caps.native_json_mode && !caps.response_schema {
            return Err(Error::config(format!(
                "model '{}' does not support response_format=JsonObject and no middleware \
                 polyfilled it",
                config.model
            )));
        }
    }
    if let Some(ResponseFormat::JsonSchema { .. }) = &config.response_format {
        if !caps.response_schema {
            return Err(Error::config(format!(
                "model '{}' does not support response_format=JsonSchema and no middleware \
                 polyfilled it",
                config.model
            )));
        }
    }
    if asks_for_schema && has_function_tools && !caps.response_schema_with_tools {
        return Err(Error::config(format!(
            "model '{}' does not support combining response_format with function-calling \
             tools and no middleware reconciled them",
            config.model
        )));
    }
    Ok(())
}

/// Top-level entry point. Threads `prompt` and `config.raw()` through
/// the middleware chain (as `Cow`s, so pass-through is free),
/// validates the post-middleware request against the baked-in
/// capabilities, calls the underlying provider, and unwraps the
/// response stream in reverse middleware order.
///
/// `provider` is any `&dyn Provider`. Calling
/// [`crate::Provider::generate`] directly bypasses middleware — use
/// that only if you've already run the pipeline yourself or you know
/// the model natively supports everything in the config.
pub async fn generate(
    provider: &dyn Provider,
    prompt: &Prompt,
    config: &crate::Config,
) -> Result<Response, Error> {
    // Capabilities are owned by the provider — ask it.
    let capabilities = provider.capabilities(&config.raw().model);

    // Resolve middleware: caller override wins, otherwise derive from
    // the resolved caps.
    let owned_default;
    let middleware: &[Arc<dyn Middleware>] = match config.middleware_override() {
        Some(m) => m,
        None => {
            owned_default = default_middleware(&capabilities);
            &owned_default
        }
    };

    let mut prompt_cow: Cow<'_, Prompt> = Cow::Borrowed(prompt);
    let mut raw_cow: Cow<'_, RawConfig> = Cow::Borrowed(config.raw());
    let mut response_transforms: Vec<ResponseTransform> = Vec::new();
    for m in middleware {
        if let Some(rt) = m.apply(&mut prompt_cow, &mut raw_cow, &capabilities)? {
            response_transforms.push(rt);
        }
    }

    validate(&raw_cow, &capabilities)?;

    let response = provider.generate(&prompt_cow, &raw_cow).await?;

    let response = response_transforms
        .into_iter()
        .rev()
        .fold(response, |r, transform| transform(r));
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FinishReason, Function, Usage};
    use crate::{Config, StreamEvent};
    use async_trait::async_trait;
    use std::sync::Mutex;

    #[derive(Debug)]
    struct MockProvider {
        last_raw: Arc<Mutex<Option<RawConfig>>>,
        events: Mutex<Option<Vec<Result<StreamEvent, Error>>>>,
    }

    impl MockProvider {
        fn new(events: Vec<Result<StreamEvent, Error>>) -> Self {
            Self {
                last_raw: Arc::new(Mutex::new(None)),
                events: Mutex::new(Some(events)),
            }
        }
        fn last_raw(&self) -> RawConfig {
            self.last_raw.lock().unwrap().clone().expect("called")
        }
        fn was_called(&self) -> bool {
            self.last_raw.lock().unwrap().is_some()
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
        async fn generate(&self, _prompt: &Prompt, config: &RawConfig) -> Result<Response, Error> {
            *self.last_raw.lock().unwrap() = Some(config.clone());
            let events = self.events.lock().unwrap().take().unwrap_or_default();
            Ok(Response::from_stream(futures_util::stream::iter(events)))
        }
    }

    fn json_schema_rf(name: &str, schema_json: &str) -> ResponseFormat {
        let schema = serde_json::value::RawValue::from_string(schema_json.to_string()).unwrap();
        ResponseFormat::JsonSchema {
            name: name.to_string(),
            schema: std::borrow::Cow::Owned(schema),
            strict: true,
        }
    }

    #[test]
    fn validate_rejects_unsupported_schema() {
        let cfg = Config::builder("claude-sonnet-4-5")
            .response_format(json_schema_rf("X", r#"{"type":"object"}"#))
            .build();
        let caps = Capabilities::for_model(&cfg.raw().model);
        let err = validate(cfg.raw(), &caps).expect_err("anthropic has no schema support");
        assert!(err.to_string().contains("JsonSchema"), "got: {err}");
    }

    #[test]
    fn validate_rejects_schema_plus_tools_on_pre_3_gemini() {
        let cfg = Config::builder("gemini-2.5-flash")
            .response_format(json_schema_rf("X", r#"{"type":"object"}"#))
            .tools(vec![Tool::Function(Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: std::borrow::Cow::Owned(
                    serde_json::value::RawValue::from_string("{}".to_string()).unwrap(),
                ),
            })])
            .build();
        let caps = Capabilities::for_model(&cfg.raw().model);
        let err = validate(cfg.raw(), &caps).expect_err("2.5 doesn't allow schema+tools");
        assert!(err.to_string().contains("combining"), "got: {err}");
    }

    /// Vertex rejects controlled generation combined with function
    /// calling on Gemini 2.x — and that restriction covers
    /// `responseMimeType` (`JsonObject`), not just `responseSchema`
    /// (the wire error is "Function calling with a response mime type:
    /// 'application/json' is unsupported"). So `validate()` must reject
    /// `JsonObject` + tools on 2.5 when no middleware reconciled it,
    /// exactly as it does for `JsonSchema` + tools. (With the default
    /// middleware present `JsonCoercionMiddleware` polyfills it; this
    /// guards the safety net for callers who disable middleware.)
    #[test]
    fn validate_rejects_json_object_plus_tools_on_pre_3_gemini() {
        let cfg = Config::builder("gemini-2.5-flash")
            .response_format(ResponseFormat::JsonObject)
            .tools(vec![Tool::Function(Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: std::borrow::Cow::Owned(
                    serde_json::value::RawValue::from_string("{}".to_string()).unwrap(),
                ),
            })])
            .build();
        let caps = Capabilities::for_model(&cfg.raw().model);
        assert!(caps.native_json_mode && !caps.response_schema_with_tools);
        let err = validate(cfg.raw(), &caps)
            .expect_err("JsonObject + tools is rejected on 2.5 controlled generation");
        assert!(err.to_string().contains("combining"), "got: {err}");
    }

    #[test]
    fn validate_passes_after_json_coercion_clears_response_format() {
        let prompt = Prompt::user("hi");
        let cfg = Config::builder("claude-sonnet-4-5")
            .response_format(json_schema_rf("X", r#"{"type":"object"}"#))
            .build();
        let mut prompt_cow: Cow<'_, Prompt> = Cow::Borrowed(&prompt);
        let mut raw_cow: Cow<'_, RawConfig> = Cow::Borrowed(cfg.raw());
        let caps = Capabilities::for_model(&cfg.raw().model);
        let _ = JsonCoercionMiddleware
            .apply(&mut prompt_cow, &mut raw_cow, &caps)
            .unwrap();
        validate(&raw_cow, &caps).expect("post-middleware should validate");
    }

    #[tokio::test]
    async fn generate_errors_when_caller_disables_polyfill() {
        let provider = MockProvider::new(Vec::new());
        let prompt = Prompt::user("hi");
        let config = Config::builder("claude-sonnet-4-5")
            .response_format(json_schema_rf("X", r#"{"type":"object"}"#))
            .with_middleware(Vec::new())
            .build();
        let err = match generate(&provider, &prompt, &config).await {
            Ok(_) => panic!("should reject pre-flight"),
            Err(e) => e,
        };
        assert!(matches!(err, Error::Config(_)));
    }

    /// When a middleware returns `Err`, [`generate`] short-circuits
    /// before reaching the provider. Uses the
    /// `JsonCoercionMiddleware` conflict case (caller pinned
    /// `tool_choice` to a specific function) as the trigger.
    #[tokio::test]
    async fn middleware_error_short_circuits_provider() {
        let schema =
            serde_json::value::RawValue::from_string(r#"{"type":"object"}"#.to_string()).unwrap();
        let provider = MockProvider::new(Vec::new());
        let prompt = Prompt::user("hi");
        let config = Config::builder("claude-sonnet-4-5")
            .response_format(ResponseFormat::JsonSchema {
                name: "X".to_string(),
                schema: std::borrow::Cow::Owned(schema),
                strict: true,
            })
            .tools(vec![Tool::Function(Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: std::borrow::Cow::Owned(
                    serde_json::value::RawValue::from_string("{}".to_string()).unwrap(),
                ),
            })])
            .tool_choice(crate::types::ToolChoice::Function {
                name: "get_weather".to_string(),
            })
            .build();

        let err = match generate(&provider, &prompt, &config).await {
            Ok(_) => panic!("middleware should error before provider runs"),
            Err(e) => e,
        };
        assert!(matches!(err, Error::Config(_)), "got: {err}");
        assert!(
            !provider.was_called(),
            "provider must not be called after middleware error"
        );
    }

    #[tokio::test]
    async fn generate_passes_through_cows_when_no_middleware_modifies() {
        let provider = MockProvider::new(vec![Ok(StreamEvent::Done {
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        })]);
        let prompt = Prompt::user("hi");
        let config = Config::builder("claude-sonnet-4-5")
            .temperature(0.5)
            .build();
        let _ = generate(&provider, &prompt, &config).await.unwrap();
        let recv = provider.last_raw();
        assert_eq!(recv.model, "claude-sonnet-4-5");
        assert_eq!(recv.temperature, Some(0.5));
        assert!(recv.response_format.is_none());
    }
}
