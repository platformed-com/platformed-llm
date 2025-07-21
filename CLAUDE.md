# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`platformed-llm` is a Rust library that provides a unified abstraction over multiple LLM providers. It implements a common interface for OpenAI, Google Gemini (via Vertex), and Anthropic (via Vertex), with extensibility for future providers.

### Core Features
- **Unified API** across all providers using OpenAI's "Responses" API pattern
- **Streaming responses** support for real-time output
- **Function calling** capabilities across all providers
- **Provider abstraction** designed for easy addition of new LLM providers
- **Flexible authentication** supporting access tokens and Application Default Credentials for Vertex AI

## Development Commands

### Build
```bash
cargo build
cargo build --release  # For optimized builds
```

### Test
```bash
cargo test              # Run all tests
cargo test -- --nocapture  # Show println! output during tests
cargo test [test_name]  # Run specific test
```

### Check & Lint
```bash
cargo check            # Fast compilation check without producing binaries
cargo clippy          # Rust linter with helpful suggestions
cargo fmt             # Format code according to Rust standards
```

### Documentation
```bash
cargo doc --open      # Generate and open documentation
```

## Project Structure

The project follows standard Rust library conventions:
- `src/lib.rs`: Main library entry point
- `Cargo.toml`: Package manifest and dependencies
- `target/`: Build artifacts (gitignored)

## Architecture Design

### Provider Structure
Implement each provider as a separate module:
- `providers::openai` - OpenAI API implementation
- `providers::vertex::gemini` - Google Gemini via Vertex AI
- `providers::vertex::anthropic` - Anthropic Claude via Vertex AI

### Core Abstractions
1. **Common Trait**: Define a `LLMProvider` trait that all providers implement
2. **Response Types**: Mirror OpenAI's Responses API structure for consistency
3. **Streaming**: Use tokio-stream or futures-stream for async streaming
4. **Function Calling**: Unified function schema representation across providers
5. **Authentication**: Support access tokens and Application Default Credentials for Vertex AI

### Key Dependencies to Consider
- `reqwest` or `hyper` for HTTP requests
- `tokio` for async runtime
- `serde` + `serde_json` for serialization
- `futures` or `tokio-stream` for streaming
- No external auth libraries required - supports direct API keys and access tokens

### Implementation Priority
1. Define core traits and types
2. Implement OpenAI provider as reference
3. Add Vertex providers (sharing auth/transport logic)
4. Ensure streaming and function calling work across all providers

## Important Instructions

### API Design Principles
- **No legacy compatibility**: When refactoring APIs, update all usages directly without preserving old names or adding backward compatibility aliases
- **Clean refactoring**: Remove deprecated code completely rather than maintaining it
- **Direct updates**: Update all files that use old APIs rather than adding compatibility layers