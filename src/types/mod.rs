//! Core types used throughout the library.

/// Request configuration, reasoning options, response formats, and usage
/// accounting.
pub mod config;
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
pub use message::{
    modality_from_mime, Annotation, AnnotationKind, AssistantPart, ComputerUseConfig, FileInput,
    FileSource, FinishReason, Function, FunctionCall, InputItem, Modality, ProviderBuiltin,
    ProviderFileRef, Tool, UserPart,
};
pub use prompt::Prompt;
pub use streaming::{PartKind, PartUpdate, StreamEvent};
