# platformed-llm Design Document

## Overview

`platformed-llm` is a Rust library providing a unified abstraction over multiple LLM providers. It offers a consistent API for interacting with OpenAI, Google Gemini (via Vertex AI), and Anthropic Claude (via Vertex AI), with support for streaming responses and function calling.

## Key Design Principles

1. **Single Entry Point**: All interactions go through the `LLM` struct, created via builder pattern
2. **Builder Pattern**: Fluent API for configuration with methods like `.model()`, `.temperature()`, `.function()`
3. **Streaming by Default**: All providers use streaming internally for consistency
4. **Flexible Response Handling**: Single `Response` type that can be streamed or buffered
5. **Hidden Complexity**: Delta merging and stream processing handled internally
6. **Type Safety**: Leverage Rust's type system while maintaining ease of use

## Goals

1. **Unified Interface**: Single API that works across all supported providers
2. **Type Safety**: Leverage Rust's type system for compile-time guarantees
3. **Async First**: Built on tokio for high-performance async operations
4. **Streaming Support**: First-class support for streaming responses
5. **Function Calling**: Consistent interface for tool use across providers
6. **Extensibility**: Easy to add new providers without breaking changes

## Architecture

### Core Components

```
platformed-llm/
├── src/
│   ├── lib.rs              # Public API exports
│   ├── llm.rs              # Main LLM entry point and builder
│   ├── provider.rs         # Core trait definitions
│   ├── response.rs         # Unified response type with streaming
│   ├── types/              # Common types and models
│   │   ├── mod.rs
│   │   ├── message.rs      # Message types
│   │   ├── function.rs     # Function calling types
│   │   └── config.rs       # Configuration types
│   ├── providers/          # Provider implementations
│   │   ├── mod.rs
│   │   ├── openai/
│   │   │   ├── mod.rs
│   │   │   ├── client.rs
│   │   │   └── streaming.rs
│   │   └── vertex/
│   │       ├── mod.rs
│   │       ├── auth.rs     # Shared Vertex auth
│   │       ├── gemini/
│   │       │   ├── mod.rs
│   │       │   └── client.rs
│   │       └── anthropic/
│   │           ├── mod.rs
│   │           └── client.rs
│   ├── error.rs            # Error types
│   └── utils/              # Utility functions
│       ├── mod.rs
│       ├── sse.rs          # SSE parsing utilities
│       └── delta.rs        # Delta merging utilities
```

### Main API Entry Point

```rust
// llm.rs - Single entry point with builder pattern
pub struct LLM {
    provider: Box<dyn LLMProvider>,
    default_model: Option<String>,
    default_temperature: Option<f32>,
    default_max_tokens: Option<u32>,
    tools: Vec<Tool>,
}

impl LLM {
    /// Create a new LLM instance with OpenAI
    pub fn openai(api_key: impl Into<String>) -> LLMBuilder {
        LLMBuilder::new(ProviderConfig::OpenAI { 
            api_key: api_key.into() 
        })
    }
    
    /// Create a new LLM instance with Google Gemini via Vertex
    pub fn gemini(project_id: impl Into<String>, location: impl Into<String>) -> LLMBuilder {
        LLMBuilder::new(ProviderConfig::Gemini {
            project_id: project_id.into(),
            location: location.into(),
        })
    }
    
    /// Create a new LLM instance with Anthropic via Vertex
    pub fn anthropic_vertex(project_id: impl Into<String>, location: impl Into<String>) -> LLMBuilder {
        LLMBuilder::new(ProviderConfig::AnthropicVertex {
            project_id: project_id.into(),
            location: location.into(),
        })
    }
    
    /// Generate a response - this is the main method users interact with
    pub async fn generate(&self, prompt: impl Into<Prompt>) -> Result<Response, Error> {
        let prompt = prompt.into();
        let request = self.build_request(prompt);
        
        // Always use streaming internally
        let stream = self.provider.stream(request).await?;
        Ok(Response::new(stream))
    }
    
    fn build_request(&self, prompt: Prompt) -> InternalRequest {
        InternalRequest {
            model: self.default_model.clone().unwrap_or_else(|| "gpt-3.5-turbo".to_string()),
            messages: prompt.messages().to_vec(),
            temperature: self.default_temperature,
            max_tokens: self.default_max_tokens,
            top_p: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: if self.tools.is_empty() { None } else { Some(self.tools.clone()) },
        }
    }
}

pub struct LLMBuilder {
    config: ProviderConfig,
    model: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    tools: Vec<Tool>,
    system_prompt: Option<String>,
}

impl LLMBuilder {
    pub fn new(config: ProviderConfig) -> Self {
        Self {
            config,
            model: None,
            temperature: None,
            max_tokens: None,
            tools: Vec::new(),
            system_prompt: None,
        }
    }
    
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
    
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }
    
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }
    
    pub fn system(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }
    
    /// Add a function the LLM can call
    pub fn function<F>(mut self, name: impl Into<String>, description: impl Into<String>, parameters: F) -> Self 
    where 
        F: Into<serde_json::Value>
    {
        self.tools.push(Tool {
            r#type: ToolType::Function,
            function: Function {
                name: name.into(),
                description: description.into(),
                parameters: parameters.into(),
            },
        });
        self
    }
    
    /// Build the LLM instance
    pub async fn build(self) -> Result<LLM, Error> {
        let provider = match self.config {
            ProviderConfig::OpenAI { api_key } => {
                Box::new(OpenAIProvider::new(api_key)?) as Box<dyn LLMProvider>
            }
            ProviderConfig::Gemini { project_id, location } => {
                Box::new(GeminiProvider::new(project_id, location).await?) as Box<dyn LLMProvider>
            }
            ProviderConfig::AnthropicVertex { project_id, location } => {
                Box::new(AnthropicVertexProvider::new(project_id, location).await?) as Box<dyn LLMProvider>
            }
        };
        
        Ok(LLM {
            provider,
            default_model: self.model,
            default_temperature: self.temperature,
            default_max_tokens: self.max_tokens,
            tools: self.tools,
        })
    }
}

// Consistent prompt representation with convenience methods
#[derive(Debug, Clone)]
pub struct Prompt {
    messages: Vec<Message>,
}

impl Prompt {
    /// Create a new empty prompt
    pub fn new() -> Self {
        Self { messages: Vec::new() }
    }
    
    /// Create a prompt with a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            messages: vec![Message::system(content.into())]
        }
    }
    
    /// Create a prompt with a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            messages: vec![Message::user(content.into())]
        }
    }
    
    /// Add a system message
    pub fn with_system(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::system(content.into()));
        self
    }
    
    /// Add a user message
    pub fn with_user(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::user(content.into()));
        self
    }
    
    /// Add an assistant message
    pub fn with_assistant(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::assistant(content.into()));
        self
    }
    
    /// Add a message
    pub fn with_message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }
    
    /// Add multiple messages
    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages.extend(messages);
        self
    }
    
    /// Get the messages
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }
    
    /// Get mutable messages
    pub fn messages_mut(&mut self) -> &mut Vec<Message> {
        &mut self.messages
    }
    
    /// Add tool results to the prompt
    pub fn with_tool_results(mut self, tool_results: Vec<(String, String)>) -> Self {
        for (tool_call_id, result) in tool_results {
            self.messages.push(Message::tool_result(tool_call_id, result));
        }
        self
    }
    
    /// Add tool result messages
    pub fn with_tool_result_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages.extend(messages);
        self
    }
}

impl From<&str> for Prompt {
    fn from(s: &str) -> Self {
        Prompt::user(s)
    }
}

impl From<String> for Prompt {
    fn from(s: String) -> Self {
        Prompt::user(s)
    }
}

impl From<Vec<Message>> for Prompt {
    fn from(messages: Vec<Message>) -> Self {
        Prompt { messages }
    }
}

impl From<Message> for Prompt {
    fn from(message: Message) -> Self {
        Prompt { messages: vec![message] }
    }
}
```

### Unified Response Type

```rust
// response.rs - Handles both streaming and buffering
use futures::{Stream, StreamExt};
use std::pin::Pin;

pub struct Response {
    stream: Pin<Box<dyn Stream<Item = Result<StreamChunk, Error>> + Send>>,
}

impl Response {
    pub(crate) fn new(stream: impl Stream<Item = Result<StreamChunk, Error>> + Send + 'static) -> Self {
        Self {
            stream: Box::pin(stream),
        }
    }
    
    /// Stream the response chunk by chunk
    pub fn stream(self) -> impl Stream<Item = Result<StreamEvent, Error>> {
        self.stream.flat_map(|chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    let events = StreamEvent::from_chunk(chunk);
                    futures::stream::iter(events.into_iter().map(Ok))
                }
                Err(e) => futures::stream::iter(vec![Err(e)].into_iter()),
            }
        })
    }
    
    /// Buffer the entire response
    pub async fn buffer(mut self) -> Result<CompleteResponse, Error> {
        let mut accumulated = AccumulatedResponse::default();
        
        // Consume the stream and accumulate
        while let Some(chunk_result) = self.stream.next().await {
            let chunk = chunk_result?;
            accumulated.merge_chunk(&chunk);
        }
        
        Ok(accumulated.to_complete_response())
    }
    
    /// Get just the text content (convenience method)
    pub async fn text(self) -> Result<String, Error> {
        let complete = self.buffer().await?;
        Ok(complete.content)
    }
    
    /// Add this response to a prompt for continued conversation
    pub async fn add_to_prompt(self, mut prompt: Prompt) -> Result<(Prompt, CompleteResponse), Error> {
        let complete = self.buffer().await?;
        
        // Add assistant message to prompt
        let assistant_message = if complete.function_calls.is_empty() {
            Message::assistant(complete.content.clone())
        } else {
            // Create message with function calls
            Message::assistant_with_tools(complete.content.clone(), complete.function_calls.clone())
        };
        
        prompt.messages_mut().push(assistant_message);
        Ok((prompt, complete))
    }
    
    /// Create a conversation handler for back-and-forth chat
    pub fn chat(self) -> ChatHandler {
        ChatHandler::new(self)
    }
    
    /// Handle function calls with a closure and continue conversation
    pub async fn handle_function_calls<F, Fut>(
        self, 
        mut prompt: Prompt, 
        handler: F
    ) -> Result<Prompt, Error>
    where
        F: Fn(FunctionCall) -> Fut,
        Fut: std::future::Future<Output = Result<String, Error>>,
    {
        let complete = self.buffer().await?;
        
        // Add assistant message to prompt
        let assistant_message = if complete.function_calls.is_empty() {
            Message::assistant(complete.content.clone())
        } else {
            Message::assistant_with_tools(complete.content.clone(), complete.function_calls.clone())
        };
        prompt.messages_mut().push(assistant_message);
        
        // Handle function calls
        for call in complete.function_calls {
            let result = handler(call.clone()).await?;
            prompt.messages_mut().push(Message::tool_result(call.id, result));
        }
        
        Ok(prompt)
    }
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Text content delta
    Text(String),
    /// Function call started
    FunctionCallStart { id: String, name: String },
    /// Function call completed with full arguments
    FunctionCallEnd { id: String, name: String, arguments: String },
    /// Response finished
    Finished { reason: FinishReason },
}

impl StreamEvent {
    pub fn from_chunk(chunk: StreamChunk) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        
        for choice in chunk.choices {
            // Handle text content
            if let Some(content) = choice.delta.content {
                events.push(StreamEvent::Text(content));
            }
            
            // Handle tool calls - this is simplified, real implementation would need
            // to track partial function calls and emit start/end events appropriately
            if let Some(tool_calls) = choice.delta.tool_calls {
                for tool_call in tool_calls {
                    if let Some(id) = tool_call.id {
                        if let Some(function) = &tool_call.function {
                            if let Some(name) = &function.name {
                                events.push(StreamEvent::FunctionCallStart {
                                    id: id.clone(),
                                    name: name.clone(),
                                });
                            }
                        }
                    }
                }
            }
            
            // Handle finish reason
            if let Some(reason) = choice.finish_reason {
                events.push(StreamEvent::Finished { reason });
            }
        }
        
        events
    }
}

#[derive(Debug, Clone)]
pub struct CompleteResponse {
    pub content: String,
    pub function_calls: Vec<FunctionCall>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

impl CompleteResponse {
    /// Check if this response contains function calls
    pub fn has_function_calls(&self) -> bool {
        !self.function_calls.is_empty()
    }
    
    /// Get function calls by name
    pub fn get_function_calls(&self, name: &str) -> Vec<&FunctionCall> {
        self.function_calls.iter()
            .filter(|call| call.name == name)
            .collect()
    }
    
    /// Handle function calls with a closure and return tool results
    pub async fn handle_function_calls<F, Fut>(
        &self, 
        handler: F
    ) -> Result<Vec<(String, String)>, Error>
    where
        F: Fn(&FunctionCall) -> Fut,
        Fut: std::future::Future<Output = Result<String, Error>>,
    {
        let mut results = Vec::new();
        
        for call in &self.function_calls {
            let result = handler(call).await?;
            results.push((call.id.clone(), result));
        }
        
        Ok(results)
    }
    
    /// Create tool result messages from function calls
    pub fn create_tool_results<F>(&self, handler: F) -> Vec<Message>
    where
        F: Fn(&FunctionCall) -> String,
    {
        self.function_calls.iter()
            .map(|call| {
                let result = handler(call);
                Message::tool_result(call.id.clone(), result)
            })
            .collect()
    }
}

// Internal accumulator that handles delta merging
#[derive(Default)]
struct AccumulatedResponse {
    content: String,
    function_calls: HashMap<String, PartialFunctionCall>,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
}

impl AccumulatedResponse {
    fn merge_chunk(&mut self, chunk: &StreamChunk) {
        // Implementation handles all the complexity of merging deltas
        // Users never see this complexity
    }
    
    fn to_complete_response(self) -> CompleteResponse {
        CompleteResponse {
            content: self.content,
            function_calls: self.function_calls.into_values().map(|p| p.into()).collect(),
            finish_reason: self.finish_reason.unwrap_or(FinishReason::Stop),
            usage: self.usage.unwrap_or_default(),
        }
    }
}

/// Handler for streaming conversations that accumulates messages
pub struct ChatHandler {
    stream: Pin<Box<dyn Stream<Item = Result<StreamChunk, Error>> + Send>>,
    accumulated: AccumulatedResponse,
    prompt: Prompt,
}

impl ChatHandler {
    pub(crate) fn new(response: Response) -> Self {
        Self {
            stream: response.stream,
            accumulated: AccumulatedResponse::default(),
            prompt: Prompt::new(),
        }
    }
    
    /// Create a chat handler with initial prompt
    pub fn with_prompt(response: Response, prompt: Prompt) -> Self {
        Self {
            stream: response.stream,
            accumulated: AccumulatedResponse::default(),
            prompt,
        }
    }
    
    /// Stream events while building up the conversation
    pub fn stream_events(self) -> impl Stream<Item = Result<ChatEvent, Error>> {
        self.stream.map(|chunk_result| {
            match chunk_result {
                Ok(chunk) => {
                    // This would need to be handled differently in real implementation
                    // since we need mutable access to accumulated
                    Ok(ChatEvent::Stream(StreamEvent::from_chunk(chunk)))
                }
                Err(e) => Err(e),
            }
        })
    }
    
    /// Complete the conversation and return the updated prompt
    pub async fn complete(mut self) -> Result<(Prompt, CompleteResponse), Error> {
        // Consume the stream and accumulate
        while let Some(chunk_result) = self.stream.next().await {
            let chunk = chunk_result?;
            self.accumulated.merge_chunk(&chunk);
        }
        
        let complete = self.accumulated.to_complete_response();
        
        // Add assistant message to prompt
        let assistant_message = if complete.function_calls.is_empty() {
            Message::assistant(complete.content.clone())
        } else {
            Message::assistant_with_tools(complete.content.clone(), complete.function_calls.clone())
        };
        
        self.prompt.messages_mut().push(assistant_message);
        Ok((self.prompt, complete))
    }
    
    /// Add tool results and continue the conversation
    pub async fn add_tool_results(mut self, tool_results: Vec<(String, String)>) -> Result<Prompt, Error> {
        let (mut prompt, _) = self.complete().await?;
        
        // Add tool result messages
        for (tool_call_id, result) in tool_results {
            prompt.messages_mut().push(Message::tool_result(tool_call_id, result));
        }
        
        Ok(prompt)
    }
}

#[derive(Debug, Clone)]
pub enum ChatEvent {
    Stream(StreamEvent),
    Complete(CompleteResponse),
}
```

### Core Provider Trait

```rust
// provider.rs - Simplified internal trait
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

#[async_trait]
pub(crate) trait LLMProvider: Send + Sync {
    /// All providers must implement streaming
    async fn stream(
        &self,
        request: InternalRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, Error>> + Send>>, Error>;
}

#[async_trait]
pub trait Authenticator: Send + Sync {
    async fn get_token(&self) -> Result<String, Error>;
}
```

### Type Definitions

```rust
// types/config.rs
#[derive(Debug, Clone)]
pub enum ProviderConfig {
    OpenAI { api_key: String },
    Gemini { project_id: String, location: String },
    AnthropicVertex { project_id: String, location: String },
}

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cached_tokens: Option<u32>,
}

// types/message.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Content,
}

impl Message {
    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Content::Text { text: content.into() },
        }
    }
    
    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Content::Text { text: content.into() },
        }
    }
    
    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Content::Text { text: content.into() },
        }
    }
    
    /// Create a tool result message
    pub fn tool_result(tool_use_id: String, content: String) -> Self {
        Self {
            role: Role::Tool,
            content: Content::Multipart { 
                parts: vec![ContentPart::ToolResult {
                    tool_use_id,
                    content,
                }] 
            },
        }
    }
    
    /// Create a message with multipart content
    pub fn multipart(role: Role, parts: Vec<ContentPart>) -> Self {
        Self {
            role,
            content: Content::Multipart { parts },
        }
    }
    
    /// Create an assistant message with tool calls
    pub fn assistant_with_tools(content: String, tool_calls: Vec<FunctionCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: Content::AssistantWithTools { 
                text: content, 
                tool_calls 
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Content {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "multipart")]
    Multipart { parts: Vec<ContentPart> },
    #[serde(rename = "assistant_with_tools")]
    AssistantWithTools { 
        text: String, 
        tool_calls: Vec<FunctionCall> 
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    Image { image_url: ImageUrl },
    ToolResult { tool_use_id: String, content: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    pub detail: Option<String>, // "low", "high", "auto"
}

// types/config.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub tools: Option<Vec<Tool>>,
}

// types/response.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub id: String,
    pub model: String,
    pub created: u64,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}

// types/streaming.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub model: String,
    pub created: u64,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    pub role: Option<Role>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub id: Option<String>,
    pub r#type: Option<String>,
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

// Internal accumulation types
#[derive(Debug, Clone)]
struct PartialFunctionCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl From<PartialFunctionCall> for FunctionCall {
    fn from(partial: PartialFunctionCall) -> Self {
        FunctionCall {
            id: partial.id,
            name: partial.name,
            arguments: partial.arguments,
        }
    }
}

// types/function.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolType {
    Function,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON Schema
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: ToolType,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub id: String,
    pub name: String,
    pub arguments: String, // JSON string
}
```

### Error Handling

```rust
// error.rs
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    
    #[error("Authentication failed: {0}")]
    Auth(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Provider error: {provider} - {message}")]
    Provider { provider: String, message: String },
    
    #[error("Invalid configuration: {0}")]
    Config(String),
    
    #[error("Streaming error: {0}")]
    Streaming(String),
    
    #[error("Rate limit exceeded")]
    RateLimit,
    
    #[error("Model not available: {0}")]
    ModelNotAvailable(String),
}
```

## Provider Implementations

### OpenAI Provider

```rust
// providers/openai/client.rs
pub struct OpenAIProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl OpenAIProvider {
    pub fn new(api_key: String) -> Result<Self, Error> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()?;
        
        Ok(Self {
            client,
            api_key,
            base_url: "https://api.openai.com/v1".to_string(),
        })
    }
}
```

### Vertex AI Providers

```rust
// providers/vertex/auth.rs
pub struct VertexAuthenticator {
    provider: gcp_auth::AuthenticationManager,
}

impl VertexAuthenticator {
    pub async fn new() -> Result<Self, Error> {
        let provider = gcp_auth::provider().await
            .map_err(|e| Error::Auth(e.to_string()))?;
        Ok(Self { provider })
    }
}

// providers/vertex/gemini/client.rs
pub struct GeminiProvider {
    client: reqwest::Client,
    auth: Arc<VertexAuthenticator>,
    project_id: String,
    location: String,
}

// providers/vertex/anthropic/client.rs
pub struct AnthropicVertexProvider {
    client: reqwest::Client,
    auth: Arc<VertexAuthenticator>,
    project_id: String,
    location: String,
}
```

## Dependencies

```toml
[dependencies]
# Core async runtime
tokio = { version = "1.40", features = ["full"] }
async-trait = "0.1"

# HTTP client
reqwest = { version = "0.12", features = ["json", "stream"] }

# Streaming
futures = "0.3"
futures-util = "0.3"
tokio-stream = "0.1"
eventsource-stream = "0.2"  # For SSE parsing

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Error handling
thiserror = "2.0"

# GCP authentication
gcp_auth = "0.13"

# Utilities
url = "2.5"
base64 = "0.22"

[dev-dependencies]
tokio-test = "0.4"
wiremock = "0.6"
```

## Usage Examples

### Basic Generation

```rust
use platformed_llm::LLM;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create LLM instance with builder pattern
    let llm = LLM::openai(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o")
        .temperature(0.7)
        .build()
        .await?;
    
    // Simple text generation - buffered response
    let response = llm.generate("Hello, how are you?").await?;
    println!("{}", response.text().await?);
    
    Ok(())
}
```

### Streaming Responses

```rust
use platformed_llm::{LLM, StreamEvent};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = LLM::gemini(
        std::env::var("GCP_PROJECT_ID")?,
        "europe-west1"
    )
    .model("gemini-2.0-flash")
    .build()
    .await?;
    
    // Stream response tokens as they arrive
    let response = llm.generate("Tell me a story").await?;
    let mut stream = response.stream();
    
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::Text(text) => print!("{}", text),
            StreamEvent::Finished { reason } => {
                println!("\n\nFinished: {:?}", reason);
            }
            _ => {} // Handle function calls if needed
        }
    }
    
    Ok(())
}
```

### Function Calling

```rust
use platformed_llm::{LLM, StreamEvent};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build LLM with functions
    let llm = LLM::anthropic_vertex(
        std::env::var("GCP_PROJECT_ID")?,
        "us-east5"
    )
    .model("claude-3-5-sonnet")
    .function(
        "get_weather",
        "Get current weather for a location",
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state"
                }
            },
            "required": ["location"]
        })
    )
    .function(
        "search_web",
        "Search the web for information",
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        })
    )
    .build()
    .await?;
    
    // Get buffered response with function calls
    let response = llm.generate("What's the weather in San Francisco?").await?;
    let complete = response.buffer().await?;
    
    // Process function calls
    for call in &complete.function_calls {
        println!("Function: {}", call.name);
        println!("Arguments: {}", call.arguments);
        
        // Execute function and continue conversation...
    }
    
    Ok(())
}
```

### Advanced: Multi-turn Conversation

```rust
use platformed_llm::{LLM, Prompt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = LLM::openai(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o")
        .system("You are a helpful assistant")
        .build()
        .await?;
    
    // Build conversation using Prompt builder
    let prompt = Prompt::system("You are a helpful assistant")
        .with_user("What is the capital of France?")
        .with_assistant("The capital of France is Paris.")
        .with_user("What is its population?");
    
    // Generate response with conversation history
    let response = llm.generate(prompt).await?;
    println!("{}", response.text().await?);
    
    Ok(())
}
```

### Prompt Building Examples

```rust
use platformed_llm::{LLM, Prompt, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = LLM::openai(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o")
        .build()
        .await?;
    
    // Simple string prompt (creates user message)
    let response1 = llm.generate("Hello!").await?;
    
    // System prompt with user message
    let prompt2 = Prompt::system("You are a helpful assistant")
        .with_user("Explain quantum computing");
    let response2 = llm.generate(prompt2).await?;
    
    // Complex conversation
    let prompt3 = Prompt::new()
        .with_system("You are a coding assistant")
        .with_user("How do I iterate over a vector in Rust?")
        .with_assistant("You can use a for loop: `for item in &vector { ... }`")
        .with_user("What about with indices?");
    let response3 = llm.generate(prompt3).await?;
    
    // From existing messages
    let messages = vec![
        Message::user("First message"),
        Message::assistant("Response"),
    ];
    let prompt4 = Prompt::from(messages)
        .with_user("Follow-up question");
    let response4 = llm.generate(prompt4).await?;
    
    Ok(())
}
```

### Back-and-Forth Conversations

```rust
use platformed_llm::{LLM, Prompt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = LLM::openai(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o")
        .build()
        .await?;
    
    // Start conversation
    let mut prompt = Prompt::system("You are a helpful assistant")
        .with_user("What is the capital of France?");
    
    let response = llm.generate(prompt).await?;
    
    // Add response to conversation and continue
    let (updated_prompt, complete) = response.add_to_prompt(prompt).await?;
    println!("Assistant: {}", complete.content);
    
    // Continue conversation
    let prompt = updated_prompt.with_user("What is its population?");
    let response = llm.generate(prompt).await?;
    println!("Assistant: {}", response.text().await?);
    
    Ok(())
}
```

### Streaming Conversations

```rust
use platformed_llm::{LLM, Prompt, StreamEvent, ChatEvent};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = LLM::anthropic_vertex(
        std::env::var("GCP_PROJECT_ID")?,
        "us-east5"
    )
    .model("claude-3-5-sonnet")
    .build()
    .await?;
    
    let mut prompt = Prompt::system("You are a helpful assistant")
        .with_user("Tell me a story");
    
    let response = llm.generate(prompt).await?;
    
    // Use chat handler for streaming conversation
    let chat = ChatHandler::with_prompt(response, prompt);
    let mut stream = chat.stream_events();
    
    while let Some(event) = stream.next().await {
        match event? {
            ChatEvent::Stream(StreamEvent::Text(text)) => {
                print!("{}", text);
            }
            ChatEvent::Stream(StreamEvent::Finished { .. }) => {
                break;
            }
            ChatEvent::Complete(complete) => {
                println!("\n\nConversation updated with: {}", complete.content);
            }
            _ => {}
        }
    }
    
    // Get the updated prompt for next turn
    let (updated_prompt, _) = chat.complete().await?;
    
    Ok(())
}
```

### Function Calling with Conversation History

```rust
use platformed_llm::{LLM, Prompt};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = LLM::openai(std::env::var("OPENAI_API_KEY")?)
        .function("get_weather", "Get current weather", json!({
            "type": "object",
            "properties": {
                "location": { "type": "string" }
            },
            "required": ["location"]
        }))
        .function("get_time", "Get current time", json!({
            "type": "object",
            "properties": {
                "timezone": { "type": "string" }
            },
            "required": ["timezone"]
        }))
        .build()
        .await?;
    
    let prompt = Prompt::user("What's the weather in San Francisco and what time is it there?");
    
    // Method 1: Using the convenience handler
    let updated_prompt = llm.generate(prompt.clone()).await?
        .handle_function_calls(prompt, |call| async move {
            match call.name.as_str() {
                "get_weather" => Ok("Sunny, 72°F".to_string()),
                "get_time" => Ok("3:42 PM PST".to_string()),
                _ => Err(Error::Provider { 
                    provider: "user".to_string(), 
                    message: "Unknown function".to_string() 
                }),
            }
        }).await?;
    
    // Continue conversation with function results
    let response = llm.generate(updated_prompt).await?;
    println!("Final response: {}", response.text().await?);
    
    Ok(())
}
```

### Alternative Function Calling Patterns

```rust
use platformed_llm::{LLM, Prompt};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = LLM::openai(std::env::var("OPENAI_API_KEY")?)
        .function("calculate", "Perform calculations", json!({
            "type": "object",
            "properties": {
                "expression": { "type": "string" }
            },
            "required": ["expression"]
        }))
        .build()
        .await?;
    
    let prompt = Prompt::user("What is 25 * 37 + 14?");
    let response = llm.generate(prompt.clone()).await?;
    let (updated_prompt, complete) = response.add_to_prompt(prompt).await?;
    
    // Method 2: Using CompleteResponse convenience methods
    if complete.has_function_calls() {
        let tool_results = complete.handle_function_calls(|call| async move {
            match call.name.as_str() {
                "calculate" => {
                    // Parse arguments and perform calculation
                    let args: serde_json::Value = serde_json::from_str(&call.arguments)?;
                    let expression = args["expression"].as_str().unwrap_or("");
                    
                    // Mock calculation
                    Ok(format!("Result: {}", match expression {
                        "25 * 37 + 14" => "939",
                        _ => "Unable to calculate"
                    }))
                }
                _ => Err(Error::Provider { 
                    provider: "user".to_string(), 
                    message: "Unknown function".to_string() 
                }),
            }
        }).await?;
        
        let final_prompt = updated_prompt.with_tool_results(tool_results);
        let response = llm.generate(final_prompt).await?;
        println!("Final response: {}", response.text().await?);
    }
    
    Ok(())
}
```

### Synchronous Function Handling

```rust
use platformed_llm::{LLM, Prompt};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = LLM::openai(std::env::var("OPENAI_API_KEY")?)
        .function("lookup_user", "Look up user information", json!({
            "type": "object",
            "properties": {
                "user_id": { "type": "string" }
            },
            "required": ["user_id"]
        }))
        .build()
        .await?;
    
    let prompt = Prompt::user("Look up information for user 'alice123'");
    let response = llm.generate(prompt.clone()).await?;
    let (updated_prompt, complete) = response.add_to_prompt(prompt).await?;
    
    // Method 3: Using create_tool_results for synchronous functions
    let tool_messages = complete.create_tool_results(|call| {
        match call.name.as_str() {
            "lookup_user" => {
                let args: serde_json::Value = serde_json::from_str(&call.arguments).unwrap();
                let user_id = args["user_id"].as_str().unwrap_or("");
                format!("User {}: Active since 2023, Premium member", user_id)
            }
            _ => "Function not found".to_string(),
        }
    });
    
    let final_prompt = updated_prompt.with_tool_result_messages(tool_messages);
    let response = llm.generate(final_prompt).await?;
    println!("Final response: {}", response.text().await?);
    
    Ok(())
}
```

### Streaming Function Calls

```rust
use platformed_llm::{LLM, Prompt, StreamEvent};
use futures::StreamExt;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = LLM::openai(std::env::var("OPENAI_API_KEY")?)
        .function("calculate", "Perform math calculations", json!({
            "type": "object",
            "properties": {
                "expression": { "type": "string" }
            },
            "required": ["expression"]
        }))
        .function("get_weather", "Get weather information", json!({
            "type": "object",
            "properties": {
                "location": { "type": "string" }
            },
            "required": ["location"]
        }))
        .build()
        .await?;
    
    let prompt = Prompt::user("What is 25 * 37 and what's the weather in Tokyo?");
    let response = llm.generate(prompt).await?;
    let mut stream = response.stream();
    
    let mut function_calls = Vec::new();
    
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::Text(text) => {
                print!("{}", text);
                io::stdout().flush()?;
            }
            StreamEvent::FunctionCallStart { id, name } => {
                println!("\n🔧 Starting function call: {}", name);
            }
            StreamEvent::FunctionCallEnd { id, name, arguments } => {
                println!("✅ Function call completed: {} with args: {}", name, arguments);
                
                // Store function call for later execution
                function_calls.push(FunctionCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                });
                
                // Execute function immediately (optional)
                let result = match name.as_str() {
                    "calculate" => {
                        let args: serde_json::Value = serde_json::from_str(&arguments)?;
                        let expression = args["expression"].as_str().unwrap_or("");
                        match expression {
                            "25 * 37" => "925",
                            _ => "Unable to calculate"
                        }
                    }
                    "get_weather" => {
                        let args: serde_json::Value = serde_json::from_str(&arguments)?;
                        let location = args["location"].as_str().unwrap_or("");
                        match location {
                            "Tokyo" => "Sunny, 22°C",
                            _ => "Weather data not available"
                        }
                    }
                    _ => "Function not found"
                };
                
                println!("📤 Function result: {}", result);
            }
            StreamEvent::Finished { reason } => {
                println!("\n🏁 Stream finished: {:?}", reason);
                break;
            }
        }
    }
    
    Ok(())
}
```

### Advanced Chat Loop

```rust
use platformed_llm::{LLM, Prompt, Message};
use std::io::{self, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = LLM::openai(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o")
        .system("You are a helpful assistant")
        .build()
        .await?;
    
    let mut prompt = Prompt::system("You are a helpful assistant");
    
    loop {
        // Get user input
        print!("You: ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input.is_empty() || input == "quit" {
            break;
        }
        
        // Add user message
        prompt = prompt.with_user(input);
        
        // Generate response
        let response = llm.generate(prompt.clone()).await?;
        
        // Stream the response
        let mut stream = response.stream();
        print!("Assistant: ");
        let mut accumulated_content = String::new();
        
        while let Some(event) = stream.next().await {
            match event? {
                StreamEvent::Text(text) => {
                    print!("{}", text);
                    accumulated_content.push_str(&text);
                    io::stdout().flush()?;
                }
                StreamEvent::Finished { .. } => {
                    println!();
                    break;
                }
                _ => {}
            }
        }
        
        // Add assistant response to prompt for next turn
        prompt = prompt.with_assistant(accumulated_content);
    }
    
    Ok(())
}
```

## Testing Strategy

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test against provider APIs using wiremock
3. **Provider Tests**: Optional tests against real APIs (behind feature flags)
4. **Stream Tests**: Ensure streaming behavior is correct
5. **Error Tests**: Verify error handling and recovery

## Performance Considerations

1. **Connection Pooling**: Reuse HTTP connections via reqwest
2. **Token Caching**: Cache authentication tokens (Vertex)
3. **Streaming Efficiency**: Use buffered reads for SSE parsing
4. **Async Design**: Non-blocking throughout the stack

## Security Considerations

1. **API Key Management**: Never log or expose API keys
2. **TLS**: Always use HTTPS connections
3. **Input Validation**: Validate request parameters
4. **Token Refresh**: Handle token expiry gracefully

## Future Extensions

1. **Additional Providers**: AWS Bedrock, Azure OpenAI, Cohere
2. **Batch Processing**: Support batch API endpoints
3. **Caching**: Response caching layer
4. **Metrics**: Integration with metrics systems
5. **Retry Logic**: Configurable retry strategies
6. **Rate Limiting**: Built-in rate limit handling