//! Local-inference backend abstraction.
//!
//! Decouples [`LlamaGgufProvider`](super::LlamaGgufProvider) from the
//! concrete [`llama_gguf::Engine`] so:
//!
//! - Tests can substitute a scripted token-emitter without spinning up
//!   a real GGUF model (see the cross-provider test suite).
//! - A future local backend (mistral.rs, candle, raw `llama.cpp`
//!   bindings, …) can drop into the same provider just by
//!   implementing [`LocalEngine`].
//!
//! The trait is intentionally a thin shim — one synchronous,
//! iterator-returning method that mirrors
//! [`llama_gguf::Engine::generate_streaming`]. Anything more
//! sophisticated (sampling overrides, mid-generation cancellation
//! beyond dropping the iterator) belongs on the concrete engine
//! impl, not on the abstraction.

use llama_gguf::engine::Engine;

/// Backend that produces tokens for a flat prompt string.
///
/// The returned iterator is consumed synchronously inside a
/// [`spawn_blocking`](tokio::task::spawn_blocking) worker, so it
/// needs to be `Send` but not `Sync`. Errors are surfaced as a plain
/// `String` description — provider-level wrapping into
/// [`Error`](crate::Error) is the caller's job.
pub trait LocalEngine: Send + Sync {
    /// Generate up to `max_tokens` tokens for `prompt`. Each
    /// `Ok(String)` yielded by the iterator is one decoded token or
    /// multi-token chunk (the engine decides).
    ///
    /// Cancellation: the provider stops pulling and drops the
    /// iterator when the consumer goes away. Because inference is
    /// synchronous and uninterruptible *within* a token, cancellation
    /// takes effect at the next token boundary — i.e. at most one
    /// in-flight token's worth of compute is wasted, not an unbounded
    /// amount. Implementors should drop cleanly between tokens; no
    /// finer-grained interruption is required (or expected).
    fn generate_streaming<'a>(
        &'a self,
        prompt: &str,
        max_tokens: usize,
    ) -> Box<dyn Iterator<Item = Result<String, String>> + Send + 'a>;
}

/// [`LocalEngine`] adapter around [`llama_gguf::Engine`]. Held by
/// [`LlamaGgufProvider`](super::LlamaGgufProvider) when constructed
/// via `from_gguf` / `from_engine_config`; callers wanting a
/// different backend implement [`LocalEngine`] directly and pass
/// their type to
/// [`LlamaGgufProvider::from_local_engine`](super::LlamaGgufProvider::from_local_engine).
pub struct LlamaGgufEngine {
    inner: Engine,
}

impl LlamaGgufEngine {
    /// Wrap a loaded [`llama_gguf::Engine`].
    pub fn new(engine: Engine) -> Self {
        Self { inner: engine }
    }

    /// Borrow the underlying [`Engine`] (e.g. for metadata inspection).
    pub fn inner(&self) -> &Engine {
        &self.inner
    }
}

impl LocalEngine for LlamaGgufEngine {
    fn generate_streaming<'a>(
        &'a self,
        prompt: &str,
        max_tokens: usize,
    ) -> Box<dyn Iterator<Item = Result<String, String>> + Send + 'a> {
        // `EngineError` doesn't implement `Display` directly so go
        // via `Debug` for the textual form.
        let iter = self
            .inner
            .generate_streaming(prompt, max_tokens)
            .map(|r| r.map_err(|e| format!("{e:?}")));
        Box::new(iter)
    }
}
