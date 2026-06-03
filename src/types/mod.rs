//! Core types used throughout the library.

/// Request configuration, reasoning options, response formats, and usage
/// accounting.
pub mod config;
/// Caller-held file registry: portable file IDs resolved to
/// provider-specific upload handles at request time.
pub mod files;
pub mod message;
/// Convenience builder for assembling a [`prompt::Prompt`] without writing
/// out the underlying `Vec<InputItem>` by hand.
pub mod prompt;
pub mod streaming;

// Explicit re-exports — no globs so that adding a `pub` item inside a
// module doesn't accidentally leak into the public surface.

pub use config::{
    Config, ConfigBuilder, ProviderContinuation, RawConfig, ReasoningConfig, ReasoningEffort,
    ReasoningSummary, ResponseFormat, ToolChoice, Usage,
};
pub use files::{FileResolver, LruFileResolver, ProviderScope, ResolvedFile, ResolvedHandle};
pub use message::{
    Annotation, AnnotationKind, AssistantPart, ComputerUseConfig, FileSource, FinishReason,
    Function, FunctionCall, InputItem, ProviderBuiltin, Tool, UserPart,
};
pub use prompt::Prompt;
pub use streaming::{PartKind, PartUpdate, StreamEvent};
