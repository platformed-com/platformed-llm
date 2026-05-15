//! Cross-provider test setup for the local llama-gguf provider.
//!
//! The hosted-provider variants in this directory use
//! [`ScriptedTransport`](crate::cross_provider::scripted::ScriptedTransport)
//! to assert the lib's HTTP request shape. The local provider doesn't
//! flow through `Transport`, so we substitute at the next layer
//! down: a [`ScriptedLocalEngine`] takes the place of
//! [`llama_gguf::Engine`], asserts the rendered ChatML prompt
//! contains expected substrings, and replays a canned sequence of
//! tokens back through the chat-template parser. The rest of the
//! cross-provider test harness is unchanged.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use platformed_llm::providers::local::LocalEngine;
use platformed_llm::providers::LlamaGgufProvider;
use platformed_llm::Provider;

use super::{ProviderConfig, ProviderTestSetup};

/// One scripted generation. Pops off the engine's queue on each
/// `generate_streaming` call.
pub struct ScriptedTurn {
    /// Substrings the rendered prompt must contain. Each one is
    /// asserted individually so a failure points at the missing
    /// piece, not just "prompt didn't match."
    pub expected_prompt_contains: Vec<&'static str>,
    /// Tokens to emit back through the chat-template parser. Each
    /// element is yielded as one `Ok(String)` from the iterator;
    /// concatenated they form the model's "output" for this turn.
    pub tokens: Vec<&'static str>,
}

/// `LocalEngine` implementation that:
///   1. Pops the next scripted turn off the queue.
///   2. Asserts the rendered prompt contains each expected
///      substring (a chat-template regression panics here with a
///      pointed message).
///   3. Returns an iterator that yields the canned tokens one by
///      one.
pub struct ScriptedLocalEngine {
    turns: Mutex<VecDeque<ScriptedTurn>>,
}

impl ScriptedLocalEngine {
    pub fn new(turns: Vec<ScriptedTurn>) -> Self {
        Self {
            turns: Mutex::new(turns.into()),
        }
    }
}

impl LocalEngine for ScriptedLocalEngine {
    fn generate_streaming<'a>(
        &'a self,
        prompt: &str,
        _max_tokens: usize,
    ) -> Box<dyn Iterator<Item = Result<String, String>> + Send + 'a> {
        let turn = self
            .turns
            .lock()
            .unwrap()
            .pop_front()
            .expect("ScriptedLocalEngine called more times than scripted");
        for needle in &turn.expected_prompt_contains {
            assert!(
                prompt.contains(needle),
                "prompt did not contain expected substring {needle:?}\n--- full prompt ---\n{prompt}"
            );
        }
        Box::new(
            turn.tokens
                .into_iter()
                .map(|s| Ok(s.to_string()))
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }
}

pub struct LlamaGgufTestSetup;

impl ProviderTestSetup for LlamaGgufTestSetup {
    fn get_config() -> ProviderConfig {
        ProviderConfig {
            name: "LlamaGguf",
            model: "scripted-local",
        }
    }

    fn build_provider() -> Pin<Box<dyn Provider>> {
        let scripted = ScriptedLocalEngine::new(vec![
            // Turn 1: tools manifest renders into the system prompt;
            // the user question round-trips verbatim. Model "emits"
            // a Hermes-style tool_call block split across multiple
            // tokens to exercise the streaming parser.
            ScriptedTurn {
                expected_prompt_contains: vec![
                    "get_weather",
                    "What's the weather like in Paris?",
                    "<tools>",
                    "<tool_call>",
                ],
                tokens: vec![
                    "<tool_",
                    "call>",
                    r#"{"name":"get_weather","arguments":{"location":"Paris"}}"#,
                    "</tool_call>",
                ],
            },
            // Turn 2: prior assistant tool call + tool result round-
            // trip into the prompt via ChatML; model "emits" a plain
            // text response.
            ScriptedTurn {
                expected_prompt_contains: vec![
                    "<tool_call>",
                    "get_weather",
                    "<tool_response>",
                    "sunny",
                ],
                tokens: vec!["It's sunny in Paris", " with a temperature of 22°C."],
            },
        ]);
        Box::pin(LlamaGgufProvider::from_local_engine(Arc::new(scripted)))
    }
}
