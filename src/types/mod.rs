//! Core types used throughout the library.

pub mod config;
pub mod message;
pub mod prompt;
pub mod streaming;

// Explicit re-exports — no globs so that adding a `pub` item inside a
// module doesn't accidentally leak into the public surface.

pub use config::{
    LLMRequest, ProviderContinuation, ReasoningConfig, ReasoningEffort, ReasoningSummary,
    ResponseFormat, ToolChoice, Usage,
};
pub use message::{
    Annotation, AnnotationKind, AssistantPart, AudioSource, ComputerUseConfig, DocumentSource,
    FinishReason, Function, FunctionCall, ImageSource, InputItem, ProviderBuiltin, Tool, ToolType,
    UserPart,
};
pub use prompt::Prompt;
pub use streaming::{PartKind, PartUpdate, StreamEvent};
