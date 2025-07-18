# platformed-llm Implementation Plan

## Overview

This implementation plan follows an incremental approach, building core functionality first and adding complexity in stages. Each phase is designed to be functional and testable before moving to the next.

## Phase 1: Core Foundation (Week 1-2)

### Goals
- Establish basic project structure
- Implement core types and error handling
- Create basic OpenAI provider (simplest to implement)
- Basic streaming and buffering

### Tasks

#### 1.1 Project Setup
- [ ] Initialize Cargo project with proper dependencies
- [ ] Set up basic module structure
- [ ] Configure CI/CD pipeline
- [ ] Add basic documentation

#### 1.2 Core Types
- [ ] Implement `Error` enum with `thiserror`
- [ ] Create `Message`, `Role`, `Content` types
- [ ] Implement `Prompt` struct with builder methods
- [ ] Create `Usage` and basic response types

#### 1.3 Basic OpenAI Provider
- [ ] Implement `OpenAIProvider` struct
- [ ] Create HTTP client with proper authentication
- [ ] Implement basic chat completion endpoint
- [ ] Add request/response serialization

#### 1.4 Simple Response Handling
- [ ] Implement `Response` struct with buffering only
- [ ] Add `text()` convenience method
- [ ] Create basic integration tests

### Success Criteria
- Can make basic chat completion requests to OpenAI
- Can buffer and return complete responses
- Basic error handling works
- All tests pass

## Phase 2: Streaming Support (Week 3)

### Goals
- Add streaming support to OpenAI provider
- Implement `StreamEvent` and delta processing
- Add streaming response API

### Tasks

#### 2.1 Streaming Infrastructure
- [ ] Implement SSE parsing utilities
- [ ] Create `StreamChunk` and `StreamEvent` types
- [ ] Add delta accumulation logic
- [ ] Implement stream-to-event conversion

#### 2.2 Response Streaming
- [ ] Add `stream()` method to `Response`
- [ ] Implement `StreamEvent` generation
- [ ] Add streaming tests

#### 2.3 OpenAI Streaming
- [ ] Implement OpenAI SSE streaming
- [ ] Handle streaming response format
- [ ] Add streaming integration tests

### Success Criteria
- Can stream responses token by token
- Delta accumulation works correctly
- Both buffered and streaming APIs work
- Performance is acceptable

## Phase 3: Function Calling (Week 4)

### Goals
- Add function calling support to OpenAI provider
- Implement tool definition and calling
- Add function calling response handling

### Tasks

#### 3.1 Function Types
- [ ] Implement `Tool`, `Function`, `FunctionCall` types
- [ ] Add function calling to request/response types
- [ ] Create function calling serialization

#### 3.2 Function Calling Logic
- [ ] Add function calls to `StreamEvent`
- [ ] Implement function call accumulation
- [ ] Add function calling to OpenAI provider

#### 3.3 Convenience Methods
- [ ] Add `has_function_calls()` and helper methods
- [ ] Implement `handle_function_calls()` patterns
- [ ] Add comprehensive function calling tests

### Success Criteria
- Can define and call functions via OpenAI
- Function calling works in both streaming and buffered modes
- Multiple function calls per response work
- Function calling examples work end-to-end

## Phase 4: LLM Builder Interface (Week 5)

### Goals
- Implement the main `LLM` struct and builder pattern
- Add configuration management
- Create the unified API entry point

### Tasks

#### 4.1 Builder Pattern
- [ ] Implement `LLMBuilder` struct
- [ ] Add configuration methods (model, temperature, etc.)
- [ ] Implement function registration
- [ ] Add builder validation

#### 4.2 LLM Interface
- [ ] Implement `LLM` struct
- [ ] Add `generate()` method
- [ ] Implement request building logic
- [ ] Add provider abstraction

#### 4.3 Provider Abstraction
- [ ] Create `LLMProvider` trait
- [ ] Implement trait for OpenAI provider
- [ ] Add provider factory logic

### Success Criteria
- Builder pattern works intuitively
- Can configure LLM instances with all options
- Function registration works
- Clean API matches design examples

## Phase 5: Conversation Support (Week 6)

### Goals
- Add conversation management
- Implement prompt building and response integration
- Add conversation examples

### Tasks

#### 5.1 Conversation Flow
- [ ] Implement `add_to_prompt()` method
- [ ] Add conversation building utilities
- [ ] Create conversation examples

#### 5.2 Tool Results
- [ ] Implement tool result message creation
- [ ] Add tool result convenience methods
- [ ] Add multi-turn conversation tests

### Success Criteria
- Can maintain conversation state across turns
- Function calling works in conversations
- Tool results integrate properly
- Conversation examples work

## Phase 6: Additional Providers (Week 7-8)

### Goals
- Add Google Gemini (Vertex) provider
- Add Anthropic (Vertex) provider
- Ensure provider abstraction works

### Tasks

#### 6.1 Vertex AI Authentication
- [ ] Implement `VertexAuthenticator`
- [ ] Add GCP auth integration
- [ ] Handle token caching and refresh

#### 6.2 Gemini Provider
- [ ] Implement `GeminiProvider`
- [ ] Add Gemini API integration
- [ ] Map Gemini responses to common format
- [ ] Add Gemini-specific tests

#### 6.3 Anthropic Provider
- [ ] Implement `AnthropicVertexProvider`
- [ ] Add Anthropic via Vertex integration
- [ ] Map Anthropic responses to common format
- [ ] Add Anthropic-specific tests

### Success Criteria
- All three providers work with same API
- Provider switching is transparent
- All examples work with all providers
- Provider-specific features are handled

## Phase 7: Polish and Optimization (Week 9)

### Goals
- Performance optimization
- Documentation completion
- Error handling improvements

### Tasks

#### 7.1 Performance
- [ ] Optimize streaming performance
- [ ] Add connection pooling
- [ ] Implement token caching
- [ ] Add performance benchmarks

#### 7.2 Documentation
- [ ] Complete API documentation
- [ ] Add comprehensive examples
- [ ] Create usage guide
- [ ] Add troubleshooting guide

#### 7.3 Error Handling
- [ ] Improve error messages
- [ ] Add proper error context
- [ ] Handle edge cases
- [ ] Add error recovery examples

### Success Criteria
- Performance is production-ready
- Documentation is comprehensive
- Error handling is robust
- Ready for public release

## Design Fixes

### Missing Type Definitions

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

// types/streaming.rs
#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub id: String,
    pub model: String,
    pub created: u64,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Clone)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone)]
pub struct Delta {
    pub role: Option<Role>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Clone)]
pub struct ToolCallDelta {
    pub id: Option<String>,
    pub r#type: Option<String>,
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Clone)]
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
```

### Fixed Content Type

```rust
// types/message.rs
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
```

### Fixed FunctionCall Type

```rust
// types/function.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub id: String,
    pub name: String,
    pub arguments: String, // JSON string
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: ToolType,
    pub function: FunctionCall,
}
```

### Missing LLM Methods

```rust
// llm.rs
impl LLM {
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
}
```

### Fixed StreamEvent Conversion

```rust
// response.rs
impl StreamEvent {
    pub fn from_chunk(chunk: StreamChunk) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        
        for choice in chunk.choices {
            // Handle text content
            if let Some(content) = choice.delta.content {
                events.push(StreamEvent::Text(content));
            }
            
            // Handle tool calls
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
```

## Implementation Guidelines

### 1. **Start Simple**
- Begin with the most basic functionality
- Add complexity only when the foundation is solid
- Test each phase thoroughly before moving on

### 2. **Test-Driven Development**
- Write tests for each component as you build it
- Use mocking for external services during development
- Add integration tests with real APIs last

### 3. **Documentation First**
- Document APIs as you create them
- Include examples for every major feature
- Keep documentation up-to-date with code

### 4. **Performance Considerations**
- Profile early and often
- Optimize for common use cases
- Don't over-optimize prematurely

### 5. **Error Handling**
- Design error types early
- Provide helpful error messages
- Handle edge cases gracefully

This incremental approach ensures each phase builds on a solid foundation, making the implementation more manageable and reducing the risk of architectural issues later.