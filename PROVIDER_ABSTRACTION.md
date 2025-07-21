# Provider Abstraction

This document explains how to use the provider abstraction layer in `platformed-llm`.

## Overview

The library now provides a unified `LLMProvider` trait that abstracts over different LLM providers, making it easy to switch between providers or use them interchangeably.

## Supported Providers

- **OpenAI**: GPT models via OpenAI API
- **Google Gemini**: Gemini models via Vertex AI

## Basic Usage

### Using Specific Providers

```rust
use platformed_llm::{LLMProvider, OpenAIProvider, GeminiProvider, LLMRequest, Prompt};

// OpenAI Provider
let openai_provider = OpenAIProvider::new(api_key)?;

// Gemini Provider with access token
let gemini_provider = GeminiProvider::new(project_id, location, model_id, access_token)?;

// Gemini Provider with Application Default Credentials
let gemini_provider = GeminiProvider::with_adc(project_id, location, model_id).await?;

// Both implement the same LLMProvider trait
let request = LLMRequest::from_prompt("gpt-4o-mini", &Prompt::user("Hello"));
let response = provider.generate(&request).await?;
```

### Using the Provider Factory

The factory provides automatic provider selection based on environment variables:

```rust
use platformed_llm::{ProviderFactory, ProviderConfig};

// Automatic provider creation from environment
let provider = ProviderFactory::from_env()?;

// Manual provider configuration
let config = ProviderConfig::openai(api_key);
let provider = ProviderFactory::create(&config)?;
```

## Environment Variables

### OpenAI Provider
- `OPENAI_API_KEY`: Your OpenAI API key

### Gemini Provider (via Vertex AI)

**Authentication Options:**
- `VERTEX_ACCESS_TOKEN`: Your access token (passed as Authorization header)
- Application Default Credentials (ADC) - automatically detected

**Configuration:**
- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud project ID
- `GOOGLE_CLOUD_REGION`: The region for Vertex AI (default: `europe-west1`)
- `GEMINI_MODEL`: The Gemini model to use (default: `gemini-1.5-pro`)

**Authentication Methods:**
1. **Access Token**: Direct token using `Authorization: Bearer` header
2. **Application Default Credentials**: Automatic credential discovery
   - Interactive: `gcloud auth application-default login`
   - Service Account: Set `GOOGLE_APPLICATION_CREDENTIALS`
   - Compute Engine: Automatically available

## Benefits

1. **Provider Independence**: Write code once, run with any provider
2. **Easy Testing**: Mock the `LLMProvider` trait for unit tests
3. **Flexible Configuration**: Switch providers via environment variables
4. **Consistent API**: Same methods and response format across all providers
5. **Future-Proof**: Easy to add new providers without changing existing code

## Examples

See the `examples/` directory for complete working examples:

- `function_calling.rs`: **Universal function calling** - works with any provider
- `provider_abstraction.rs`: Basic provider abstraction demo
- `provider_factory.rs`: Using the provider factory for automatic selection
- `openai_function_calling.rs`: OpenAI-specific function calling example
- `gemini_function_calling.rs`: Gemini-specific function calling example
- `gemini_auth_simple.rs`: Demonstrates access token and ADC authentication
- All examples work with the unified trait system

## Function Calling

All providers support function calling through the same interface:

```rust
let request = LLMRequest::from_prompt("gpt-4o-mini", &prompt)
    .tools(vec![your_tool_definition]);

let response = provider.generate(&request).await?;
// Handle function calls in the response...
```

The function calling format is automatically converted between provider-specific formats internally.