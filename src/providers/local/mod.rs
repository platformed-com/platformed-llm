//! Local in-process providers (no network round-trip).
//!
//! Today this is just the GGUF-backed [`LlamaGgufProvider`], but the
//! prompt/decoding machinery is deliberately split out so any future
//! local backend (mistral.rs, candle, …) can reuse it:
//!
//! - [`chat_template`] — the general abstraction: the [`ChatTemplate`]
//!   trait, the [`scan_delimited`] streaming-framing primitive, and
//!   the `ParsedDelta` → `StreamEvent` translation.
//! - [`chatml`] — the concrete ChatML/Hermes [`ChatMlTemplate`]
//!   implementation.
//! - [`engine`] — the [`LocalEngine`] backend trait.

pub mod chat_template;
pub mod chatml;
pub mod engine;
mod llama_gguf;

pub use chat_template::{
    function_tools, scan_delimited, ChatTemplate, DeltaStream, ParsedDelta, ParsedDeltaStreamExt,
    Segment, TokenStream,
};
pub use chatml::ChatMlTemplate;
pub use engine::{LlamaGgufEngine, LocalEngine};
pub use llama_gguf::LlamaGgufProvider;
