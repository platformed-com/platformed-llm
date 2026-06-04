//! Local GGUF inference via the [`llama-gguf`] crate.
//!
//! Unlike the hosted providers (OpenAI / Vertex), this runs the model
//! locally on the caller's machine. Trade-offs:
//!
//! - **No network.** Construction blocks while the GGUF file maps and
//!   tokenizer initialises, but every subsequent
//!   [`Provider::generate`] is in-process.
//! - **Synchronous inference.** [`llama_gguf::Engine`]'s generation
//!   API is a blocking iterator; we run it on a `spawn_blocking`
//!   worker and pipe decoded tokens through a `mpsc` channel into the
//!   unified [`crate::Response`] stream.
//! - **Limited config surface.** Sampling parameters (`temperature`,
//!   `top_p`, etc.) are baked into [`llama_gguf::EngineConfig`] at
//!   load time, *not* per-call. The provider snapshots an
//!   [`EngineConfig`] on construction and ignores per-call sampling
//!   knobs on [`crate::Config`]. `max_tokens` is the only per-call
//!   setting that's honoured (passed through to `generate_streaming`).
//! - **No multi-modal, no continuations.** Image / audio / document
//!   parts, and any `ProviderContinuation` items in the prompt are
//!   silently dropped (the model-switching contract).
//! - **Prompt formatting + tool calling.** Owned by [`ChatTemplate`].
//!   The default is [`ChatMlTemplate`] (ChatML with Hermes/Qwen-style
//!   `<tool_call>` blocks); supply your own via
//!   [`Self::with_chat_template`] when the model expects something
//!   different.
//!
//! [`llama-gguf`]: https://github.com/Lexmata/llama-gguf

use std::sync::Arc;

use async_trait::async_trait;
use futures::channel::oneshot;
use llama_gguf::engine::EngineConfig;

use crate::provider::Provider;
use crate::types::{FinishReason, Prompt, RawConfig, Tool};
use crate::{Error, Response};

use super::chat_template::{function_tools, translate_to_events, ChatTemplate, TokenStream};
use super::chatml::ChatMlTemplate;
use super::engine::{LlamaGgufEngine, LocalEngine};

/// Provider for local GGUF inference.
///
/// Construction is synchronous and loads the entire model into
/// memory; do it once and reuse the provider across many calls.
///
/// The engine itself is held behind the [`LocalEngine`] trait so
/// callers can substitute a different backend (or a test mock) via
/// [`Self::from_local_engine`]. `from_gguf` and `from_engine_config`
/// are convenience constructors that wire up the
/// [`llama_gguf::Engine`] backend by default.
pub struct LlamaGgufProvider {
    engine: Arc<dyn LocalEngine>,
    template: Arc<dyn ChatTemplate>,
}

impl LlamaGgufProvider {
    /// Load a GGUF model from disk with the lib's default sampling
    /// settings (temperature 0.7, top_p 0.9, top_k 40, no GPU). Use
    /// [`Self::from_engine_config`] when you need finer control.
    pub fn from_gguf(path: impl Into<String>) -> Result<Self, Error> {
        Self::from_engine_config(EngineConfig {
            model_path: path.into(),
            tokenizer_path: None,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.1,
            max_tokens: 512,
            seed: None,
            use_gpu: false,
            max_context_len: None,
            kv_cache_type: Default::default(),
        })
    }

    /// Load a GGUF model with a caller-supplied [`EngineConfig`]. The
    /// `max_tokens` field is overridden per-call by
    /// [`crate::RawConfig::max_tokens`]; everything else is locked in at
    /// load time.
    pub fn from_engine_config(config: EngineConfig) -> Result<Self, Error> {
        let engine = llama_gguf::engine::Engine::load(config)
            .map_err(|e| Error::provider("llama-gguf", format!("failed to load model: {e}")))?;
        Ok(Self::from_local_engine(Arc::new(LlamaGgufEngine::new(
            engine,
        ))))
    }

    /// Build a provider over a caller-supplied [`LocalEngine`]. The
    /// canonical use is plugging in a test mock or a non-GGUF local
    /// backend; see [`LocalEngine`] for the contract.
    pub fn from_local_engine(engine: Arc<dyn LocalEngine>) -> Self {
        Self {
            engine,
            template: Arc::new(ChatMlTemplate::new()),
        }
    }

    /// Swap in a custom [`ChatTemplate`]. The default is
    /// [`ChatMlTemplate`]; override when the loaded model was trained
    /// against a different chat format (e.g. Llama 3 instruct, raw
    /// completion).
    ///
    /// Caveat: the underlying `llama_gguf::Engine` runs its own
    /// `wrap_prompt` on the rendered string, which is a no-op only
    /// when the string already contains a recognised marker
    /// (`<|im_start|>`, `<|user|>`, `[INST]`). `ChatMlTemplate`
    /// emits `<|im_start|>` so it's safe. A custom template whose
    /// output contains none of those markers (e.g. a raw-completion
    /// or Llama-3 `<|start_header_id|>` template) will be *re-wrapped*
    /// by the engine — include a recognised marker in the rendered
    /// output, or load the GGUF with a chat template that no-ops.
    pub fn with_chat_template(mut self, template: Arc<dyn ChatTemplate>) -> Self {
        self.template = template;
        self
    }
}

#[async_trait]
impl Provider for LlamaGgufProvider {
    /// Local llama-gguf has no native JSON mode, no schema-constrained
    /// output, no schema+tools. Return [`crate::Capabilities::default`]
    /// (everything false) — the default middleware chain will
    /// polyfill `response_format` via tool-use coercion, which the
    /// ChatML template supports natively.
    fn capabilities(&self, _model: &str) -> crate::Capabilities {
        crate::Capabilities::default()
    }

    async fn generate(&self, prompt: &Prompt, config: &RawConfig) -> Result<Response, Error> {
        let tools = config
            .tools
            .as_deref()
            .map(function_tools)
            .unwrap_or_default();
        // Builtin tools are dropped by `function_tools`; log once so
        // the loss is visible in traces.
        if let Some(all) = config.tools.as_deref() {
            let dropped = all
                .iter()
                .filter(|t| !matches!(t, Tool::Function(_)))
                .count();
            if dropped > 0 {
                tracing::debug!(
                    "LlamaGgufProvider: dropping {dropped} builtin tool(s) — no local support"
                );
            }
        }

        let prompt_text = self
            .template
            .render(prompt, &tools, config.tool_choice.as_ref());
        let max_tokens = config.max_tokens.map(|n| n as usize).unwrap_or(512);
        let engine = self.engine.clone();

        // The llama-gguf streaming iterator is synchronous (it does
        // tensor math on the calling thread). Run it on a
        // `spawn_blocking` worker and bridge into async via a
        // *bounded* tokio mpsc: `blocking_send` parks the worker when
        // the consumer falls behind (backpressure — no unbounded
        // buffering) and returns `Err` *immediately* if the receiver
        // is dropped, even while parked, so cancellation is observed
        // promptly (worst case: one in-flight token's compute). The
        // `reason` oneshot carries the finish reason out-of-band so
        // the terminal `Done` reports `Length` vs `Stop` truthfully
        // (llama-gguf's iterator ends identically for EOS and the
        // max_tokens cap, so we infer it from the token budget).
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, Error>>(32);
        let (reason_tx, reason_rx) = oneshot::channel::<FinishReason>();
        tokio::task::spawn_blocking(move || {
            let stream = engine.generate_streaming(&prompt_text, max_tokens);
            let mut emitted: usize = 0;
            for tok in stream {
                let item =
                    tok.map_err(|e| Error::provider("llama-gguf", format!("generation: {e}")));
                let is_err = item.is_err();
                // Err here ⇒ receiver dropped (consumer abandoned us)
                // ⇒ stop generating; nobody is listening for a reason.
                if tx.blocking_send(item).is_err() {
                    return;
                }
                if is_err {
                    // Error already forwarded; translate_to_events
                    // suppresses the trailing Done on error, so the
                    // finish reason is moot — drop reason_tx.
                    return;
                }
                emitted += 1;
            }
            // Natural end. We can't see *why* llama-gguf stopped, so
            // infer: if we emitted at least `max_tokens` chunks the
            // budget was exhausted (each chunk is >= 1 token, so this
            // never false-positives Length); otherwise the model
            // stopped on its own (EOS / stop sequence).
            let reason = if max_tokens > 0 && emitted >= max_tokens {
                FinishReason::Length
            } else {
                FinishReason::Stop
            };
            let _ = reason_tx.send(reason);
            // tx drops here; rx sees end-of-stream on next recv.
        });

        // Adapt the tokio Receiver into a futures Stream (no
        // tokio-stream dep needed). `recv().await` yields None once
        // the worker's `tx` drops → clean end-of-stream.
        let token_stream = futures_util::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });

        // Chain: tokens → ParsedDelta → StreamEvent (+ trailing Done
        // carrying the engine-reported finish reason). The decode
        // stages are `stream!` generators in `chat_template`; this
        // provider is purely glue.
        let tokens: TokenStream = Box::pin(token_stream);
        let events = translate_to_events(self.template.decode(tokens), Some(reason_rx));

        Ok(Response::from_stream(events))
    }
}
