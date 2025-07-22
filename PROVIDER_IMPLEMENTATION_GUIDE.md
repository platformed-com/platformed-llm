# Provider Implementation Guide

This guide provides a comprehensive checklist for implementing new LLM providers in the platformed-llm library. Follow this guide to ensure your provider implementation is complete, consistent, and up to the same standard as existing providers.

## Overview

The platformed-llm library provides a unified interface for multiple LLM providers through:
- A common `LLMProvider` trait that all providers must implement
- Standardized request/response types that abstract provider-specific formats
- Streaming support with consistent event types
- Function calling capabilities across all providers
- Unified error handling and authentication patterns

## 📋 Implementation Checklist

### Phase 1: Project Structure Setup

- [ ] **Create provider module structure**
  - [ ] Create `src/providers/{provider_name}/` directory
  - [ ] Create `src/providers/{provider_name}/mod.rs` with public exports
  - [ ] Create `src/providers/{provider_name}/client.rs` for main provider implementation
  - [ ] Create `src/providers/{provider_name}/{provider_name}_types.rs` for provider-specific types
  - [ ] Update `src/providers/mod.rs` to include and re-export your provider

- [ ] **Add provider to factory system**
  - [ ] Add new variant to `ProviderType` enum in `src/factory.rs`
  - [ ] Add configuration methods to `ProviderConfig` (e.g., `new_provider_name()`)
  - [ ] Update `ProviderConfig::from_env()` to detect your provider from environment variables
  - [ ] Add provider creation logic to `ProviderFactory::create()`
  - [ ] Update `src/lib.rs` to re-export your provider types

### Phase 2: Core Type Definitions

- [ ] **Define provider-specific types** in `{provider_name}_types.rs`
  - [ ] Request types that match the provider's API format
  - [ ] Response types for both streaming and non-streaming responses
  - [ ] Authentication-related types if needed
  - [ ] Content/message format types
  - [ ] Function calling types (if supported)
  - [ ] Usage/metadata types
  - [ ] Add proper `#[derive(Debug, Clone, Serialize, Deserialize)]` annotations
  - [ ] Add conversion implementations to/from standard types when needed

- [ ] **Define authentication enum** (if multiple auth methods supported)
  ```rust
  #[derive(Debug)]
  pub enum YourProviderAuth {
      ApiKey(String),
      AccessToken(String),
      ApplicationDefault,
      // Add other auth methods as needed
  }
  ```

### Phase 3: Provider Implementation

- [ ] **Create main provider struct** in `client.rs`
  ```rust
  pub struct YourProvider {
      client: Client,
      auth: YourProviderAuth,
      // Add other required fields
  }
  ```

- [ ] **Implement constructor methods**
  - [ ] `new()` - Primary constructor with explicit parameters
  - [ ] `new_with_base_url()` - Constructor with custom base URL for testing
  - [ ] `with_auth()` - Constructor with specific authentication method
  - [ ] `with_auth_async()` - Async constructor if needed for some auth methods
  - [ ] Handle `reqwest::Client` creation with appropriate timeout (60 seconds)

- [ ] **Implement request conversion**
  - [ ] `convert_request(&self, request: &LLMRequest) -> Result<ProviderRequest, Error>`
  - [ ] Convert system/user/assistant messages to provider format
  - [ ] Handle function calls and function call outputs properly
  - [ ] Convert tools/functions to provider's function declaration format
  - [ ] Map temperature, max_tokens, top_p parameters
  - [ ] Enable streaming in the request format

- [ ] **Implement API endpoint construction**
  - [ ] `get_endpoint(&self, stream: bool) -> String`
  - [ ] Support both streaming and non-streaming endpoints
  - [ ] Handle custom base URLs for testing
  - [ ] Add appropriate query parameters (e.g., `?alt=sse` for streaming)

### Phase 4: LLMProvider Trait Implementation

- [ ] **Implement the `LLMProvider` trait**
  ```rust
  #[async_trait::async_trait]
  impl LLMProvider for YourProvider {
      async fn generate(&self, request: &LLMRequest) -> Result<Response, Error> {
          // Implementation here
      }
  }
  ```

- [ ] **HTTP request handling**
  - [ ] Convert internal request to provider format
  - [ ] Set up HTTP client with proper headers
  - [ ] Handle different authentication methods in headers
  - [ ] Send POST request to appropriate endpoint
  - [ ] Handle non-success HTTP responses with detailed error messages

- [ ] **Streaming implementation**
  - [ ] Create SSE stream from response bytes
  - [ ] Parse provider-specific SSE events
  - [ ] Convert to standardized `StreamEvent`s
  - [ ] Handle connection errors and malformed events gracefully
  - [ ] Return `Response` with properly configured event stream

### Phase 5: Event Conversion Logic

- [ ] **Implement stateful stream processing** (if provider sends incremental updates)
  - [ ] Create state tracking struct for in-progress items
  - [ ] Track when text output starts (emit `OutputItemAdded` only once)
  - [ ] Track function calls to avoid duplicate announcements
  - [ ] Handle incremental content updates properly

- [ ] **Convert provider events to `StreamEvent`s**
  - [ ] `StreamEvent::OutputItemAdded` - When new text or function call starts
  - [ ] `StreamEvent::ContentDelta` - For text content chunks  
  - [ ] `StreamEvent::FunctionCallComplete` - When function call is ready
  - [ ] `StreamEvent::Done` - When response is complete with usage info
  - [ ] `StreamEvent::Error` - For any streaming errors

- [ ] **Handle function calling correctly**
  - [ ] Generate consistent UUIDs for function call IDs
  - [ ] Emit `OutputItemAdded` before `FunctionCallComplete`
  - [ ] Handle both incremental and complete function parameter delivery
  - [ ] Map provider-specific function formats to `FunctionCall` struct

### Phase 6: Testing Implementation

- [ ] **Unit tests in provider module**
  - [ ] Test request conversion with various input types
  - [ ] Test streaming event parsing with realistic API responses
  - [ ] Test error handling for malformed responses
  - [ ] Test authentication methods
  - [ ] Test function calling scenarios
  - [ ] Use realistic test data that matches actual API responses

- [ ] **Cross-provider E2E tests**
  - [ ] Create test setup struct implementing `ProviderTestSetup` trait
  - [ ] Add to `tests/cross_provider/providers/mod.rs`
  - [ ] Implement mock server responses for function calling scenarios
  - [ ] Add test cases to `tests/cross_provider/function_calling_e2e.rs`
  - [ ] Ensure consistency with other providers

- [ ] **Test fixtures**
  - [ ] Create `.sse` fixture files with realistic streaming responses
  - [ ] Use correct line endings for the provider (LF vs CRLF)
  - [ ] Include both function calling and text-only scenarios
  - [ ] Test edge cases like empty responses, errors, etc.

### Phase 7: Documentation and Examples

- [ ] **Provider documentation**
  - [ ] Add comprehensive doc comments to all public types and methods
  - [ ] Document authentication requirements and setup
  - [ ] Provide usage examples in doc comments
  - [ ] Document any provider-specific quirks or limitations

- [ ] **Update main documentation**
  - [ ] Add provider to README.md
  - [ ] Update CLAUDE.md with provider-specific information
  - [ ] Add environment variable documentation
  - [ ] Update examples to show new provider usage

- [ ] **Example integration**
  - [ ] Update `examples/function_calling.rs` to support the new provider
  - [ ] Add provider detection logic
  - [ ] Add appropriate model name defaults
  - [ ] Test the example with your provider

### Phase 8: Quality Assurance

- [ ] **Code quality checks**
  - [ ] Run `cargo clippy` and fix all warnings
  - [ ] Run `cargo fmt` to ensure consistent formatting
  - [ ] Ensure no `TODO` or `FIXME` comments remain
  - [ ] Remove any debug print statements

- [ ] **Testing completeness**
  - [ ] Run full test suite: `cargo test`
  - [ ] Run provider-specific tests: `cargo test providers::your_provider`
  - [ ] Run cross-provider tests: `cargo test cross_provider`
  - [ ] Test examples: `cargo build --examples`

- [ ] **Error handling verification**
  - [ ] Test with invalid credentials
  - [ ] Test with network failures
  - [ ] Test with malformed API responses
  - [ ] Ensure all error messages are helpful and specific

### Phase 9: Integration and Deployment

- [ ] **Environment variable support**
  - [ ] Document required environment variables
  - [ ] Test automatic provider detection from environment
  - [ ] Verify fallback behavior works correctly

- [ ] **Performance considerations**
  - [ ] Verify reasonable timeout settings
  - [ ] Test with large responses
  - [ ] Check memory usage with streaming responses
  - [ ] Profile critical paths if needed

## 🔍 Implementation Patterns

### Authentication Patterns

```rust
// Pattern 1: Multiple auth methods
pub enum YourProviderAuth {
    ApiKey(String),
    AccessToken(String),
    ApplicationDefault,
}

// Pattern 2: Auth handling in requests
request_builder = match &self.auth {
    YourProviderAuth::ApiKey(key) => {
        request_builder.header("Authorization", format!("Bearer {key}"))
    }
    YourProviderAuth::AccessToken(token) => {
        request_builder.header("Authorization", format!("Bearer {token}"))
    }
    // Handle other auth types...
}
```

### Streaming Event Processing

```rust
// Pattern 1: Stateful processing for incremental updates
#[derive(Debug, Default)]
struct StreamState {
    has_text_output: bool,
    in_progress_calls: HashMap<String, InProgressCall>,
}

// Pattern 2: Event conversion with state tracking
fn convert_stream_event(event: ProviderEvent, state: &mut StreamState) -> Result<Vec<StreamEvent>, Error> {
    let mut events = Vec::new();
    
    match event {
        ProviderEvent::TextStart { text } => {
            if !state.has_text_output {
                events.push(StreamEvent::OutputItemAdded {
                    item: OutputItemInfo::Text,
                });
                state.has_text_output = true;
            }
            events.push(StreamEvent::ContentDelta { delta: text });
        }
        // Handle other event types...
    }
    
    Ok(events)
}
```

### Error Handling Pattern

```rust
// Consistent error handling
let response = request_builder.send().await?;

if !response.status().is_success() {
    let error_text = response.text().await?;
    return Err(Error::provider(
        "YourProvider",
        format!("API error: {error_text}"),
    ));
}
```

## 🚨 Common Pitfalls

1. **OutputItemAdded timing**: Only emit when a new item actually starts, not before every content delta
2. **Function call IDs**: Use consistent UUID generation patterns and handle both `id` and `call_id` fields
3. **State management**: For providers with incremental updates, ensure state persists across all SSE events
4. **Line endings**: Match the provider's actual SSE format (LF vs CRLF) in test fixtures
5. **Authentication**: Handle async auth setup properly and provide both sync/async constructors as needed
6. **Error messages**: Make errors provider-specific and informative

## ✅ Verification Steps

Before considering your implementation complete:

1. **All existing tests pass**: `cargo test`
2. **Examples work**: `cargo run --example function_calling`
3. **Cross-provider consistency**: Your provider behaves like others in the function calling example
4. **No clippy warnings**: `cargo clippy`
5. **Proper documentation**: All public APIs documented
6. **Environment integration**: Provider auto-detectable from environment variables

## 📁 File Structure Reference

```
src/providers/your_provider/
├── mod.rs                 # Public exports
├── client.rs             # Main provider implementation
└── your_provider_types.rs # Provider-specific types

tests/cross_provider/providers/
├── mod.rs                # Test provider registry
├── your_provider.rs      # Test setup implementation
└── fixtures/
    └── your_provider/
        ├── function_call_response.sse
        └── followup_response.sse
```

This guide ensures your provider implementation will be robust, consistent, and well-integrated with the rest of the platformed-llm ecosystem.