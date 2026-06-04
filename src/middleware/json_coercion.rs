//! JSON-via-tool-coercion polyfill.
//!
//! Many providers don't natively support `response_format`
//! (Anthropic, older Gemini paths) but every provider that supports
//! function-calling tools can be coerced into producing JSON by:
//!
//! 1. defining a tool whose `parameters` is the requested schema,
//! 2. forcing `tool_choice` to that tool, and
//! 3. unwrapping the resulting tool-call back into a text part on the
//!    response side.
//!
//! This module implements that polyfill as a [`Middleware`] so it
//! works above any provider transparently.

use std::borrow::Cow;
use std::collections::HashMap;
use std::pin::Pin;

use futures_util::stream::{Stream, StreamExt};

use crate::types::{FinishReason, Function, PartKind, RawConfig, ResponseFormat, Tool, ToolChoice};
use crate::{Capabilities, Error, Prompt, Response, StreamEvent};

use super::{Middleware, ResponseTransform};

/// When the caller set `response_format` but the model has no native
/// support for it (per `config.capabilities`), this middleware:
/// 1. Synthesizes a `Function` tool whose `parameters` is the
///    requested schema (or an open-object schema for
///    [`ResponseFormat::JsonObject`]).
/// 2. Appends it to `config.tools` with a name that doesn't collide
///    with any caller-supplied tool.
/// 3. Sets `tool_choice`:
///    - to `Function { name: synth }` when there are no other
///      function tools (no real choice to give the model);
///    - to `Required` when the caller has their own function tools,
///      so the model can pick *either* a real tool (to gather info)
///      *or* the synth tool (to emit the final structured answer).
///      Multi-turn conversations work naturally: each follow-up turn
///      the middleware re-adds the synth tool until the model picks it.
/// 4. Clears `config.response_format` (the provider would reject /
///    drop it).
/// 5. Wraps the response stream so the synthetic tool-call events are
///    rewritten as a text part — the caller sees the JSON arguments
///    as the assistant's text output, never knows a tool was involved.
///    Other tool calls pass through unchanged.
///
/// No-ops when `response_format` is `None` / `Text`, or when caps say
/// the model already supports the requested format natively.
///
/// Returns `Err(Error::Config)` when the polyfill cannot be applied
/// because the caller pinned `tool_choice` to a specific function
/// other than the synth tool — that forcing is fundamentally
/// incompatible with also guaranteeing a structured response. Drop
/// one of the constraints or use a model with native
/// `response_schema_with_tools` support.
#[derive(Debug, Default)]
pub struct JsonCoercionMiddleware;

impl Middleware for JsonCoercionMiddleware {
    fn name(&self) -> &str {
        "json_coercion"
    }

    fn apply<'a>(
        &self,
        _prompt: &mut Cow<'a, Prompt>,
        config: &mut Cow<'a, RawConfig>,
        caps: &Capabilities,
    ) -> Result<Option<ResponseTransform>, Error> {
        let Some(rf) = config.response_format.clone() else {
            return Ok(None);
        };
        let has_caller_function_tools = config
            .tools
            .as_ref()
            .is_some_and(|ts| ts.iter().any(|t| matches!(t, Tool::Function(_))));

        // Pick the schema we'll use as the synthetic tool's parameters.
        // Match `rf` *by value* so the `JsonSchema` arm can move the
        // caller's already-owned `Cow<'static, RawValue>` straight into
        // `Function.parameters` — same type, no string round-trip.
        let parameters: Cow<'static, serde_json::value::RawValue> = match rf {
            ResponseFormat::Text => return Ok(None),
            ResponseFormat::JsonObject => {
                if caps.native_json_mode {
                    return Ok(None);
                }
                Cow::Owned(
                    serde_json::value::RawValue::from_string(
                        r#"{"type":"object","additionalProperties":true}"#.to_string(),
                    )
                    .map_err(|e| Error::config(format!("synth schema: {e}")))?,
                )
            }
            ResponseFormat::JsonSchema { schema, .. } => {
                let supported = if has_caller_function_tools {
                    caps.response_schema_with_tools
                } else {
                    caps.response_schema
                };
                if supported {
                    return Ok(None);
                }
                schema
            }
        };

        // If the caller pinned `tool_choice` to a specific function, we
        // can't reconcile: forcing exactly *their* tool conflicts with
        // also guaranteeing the structured response gets emitted via
        // the synth tool. Fail fast with an actionable message rather
        // than silently breaking one of the constraints.
        if let Some(ToolChoice::Function { name }) = &config.tool_choice {
            return Err(Error::config(format!(
                "JsonCoercionMiddleware cannot reconcile response_format with \
                 tool_choice=Function {{ name: \"{name}\" }} — the forced tool would \
                 displace the structured-response tool. Drop one of these constraints \
                 or use a model with native response_schema_with_tools support."
            )));
        }

        let synth_tool_name = unique_tool_name("respond_with_json", config.tools.as_deref());
        let synth_tool = Tool::Function(Function {
            name: synth_tool_name.clone(),
            description: Some(
                "Respond to the user's request by invoking this tool. The arguments object \
                 IS the answer — there is no further reasoning step. Do not emit any other \
                 output."
                    .to_string(),
            ),
            parameters,
        });

        // First mutation — clones config exactly once if it was borrowed.
        let cfg = config.to_mut();
        cfg.response_format = None;
        let mut tools = cfg.tools.take().unwrap_or_default();
        tools.push(synth_tool);
        cfg.tools = Some(tools);
        // When the caller has their own function tools the model needs
        // the freedom to call them first and emit the structured
        // answer only when ready — set `Required` so it must call
        // *some* tool, but let it pick which. Otherwise force the
        // synth tool directly.
        cfg.tool_choice = if has_caller_function_tools {
            Some(ToolChoice::Required)
        } else {
            Some(ToolChoice::Function {
                name: synth_tool_name.clone(),
            })
        };

        let transform: ResponseTransform = Box::new(move |response| {
            Response::from_stream(rewrite_synth_tool_stream(
                response.stream(),
                synth_tool_name,
            ))
        });
        Ok(Some(transform))
    }
}

/// Find a tool name that doesn't collide with any caller-supplied
/// function tool. Avoids depending on UUID / RNG for a property that
/// only needs to hold within a single request.
fn unique_tool_name(base: &str, existing: Option<&[Tool]>) -> String {
    let taken = |candidate: &str| -> bool {
        existing.is_some_and(|tools| {
            tools
                .iter()
                .any(|t| matches!(t, Tool::Function(f) if f.name == candidate))
        })
    };
    if !taken(base) {
        return base.to_string();
    }
    let mut n = 2u32;
    loop {
        let candidate = format!("{base}_{n}");
        if !taken(&candidate) {
            return candidate;
        }
        n += 1;
    }
}

/// Stream adapter that turns the coerced provider stream back into
/// what a caller who asked for structured output expects to see.
///
/// - The synthetic tool call's events are relabeled from `ToolCall` to
///   `Text`, so its JSON arguments surface as the assistant's text.
/// - **Genuine** visible `Text` parts are suppressed entirely. The
///   provider can emit free text before / around the forced tool call
///   (`tool_choice: Required` is Gemini ANY mode, which doesn't gag
///   text; even a forced Anthropic tool call can be preceded by text
///   when thinking is on). Without this, [`crate::CompleteResponse::text`]
///   — which concatenates *all* text parts — would return
///   `preamble + json`, silently unparseable. Dropping them guarantees
///   `text()` is exactly the synth JSON.
/// - Other (real) tool calls pass through unchanged.
///
/// Because suppressing a part would leave a hole in the otherwise
/// contiguous `PartStart` index sequence the accumulator requires, the
/// kept parts are **renumbered** to stay 0,1,2,…
///
/// Terminal handling once `Done` arrives:
/// - synth tool fired and was the *only* tool → rewrite
///   `FinishReason::ToolCalls` to `Stop` (looks like a plain text turn).
/// - synth tool fired alongside real tools → leave the finish reason
///   as-is (multi-turn tool use in progress).
/// - synth tool didn't fire but real tools did → legitimate
///   intermediate turn (model is gathering info); pass through.
/// - neither fired → the model free-texted instead of producing the
///   structured answer. Surface an error rather than returning the
///   silently-suppressed text as an empty response.
fn rewrite_synth_tool_stream(
    inner: Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send>>,
    synth_tool_name: String,
) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, Error>> + Send>> {
    let mut synth_seen = false;
    let mut other_tool_seen = false;
    // Original part index -> remapped index (`Some`) or suppressed (`None`).
    let mut index_map: HashMap<u32, Option<u32>> = HashMap::new();
    let mut next_index: u32 = 0;
    Box::pin(
        inner
            .map(move |ev_result| -> Option<Result<StreamEvent, Error>> {
                let ev = match ev_result {
                    Ok(ev) => ev,
                    Err(e) => return Some(Err(e)),
                };
                match ev {
                    StreamEvent::PartStart { index, kind } => match kind {
                        // Synthetic tool call -> relabel to a text part.
                        PartKind::ToolCall { ref name, .. } if name == &synth_tool_name => {
                            synth_seen = true;
                            let mapped = next_index;
                            next_index += 1;
                            index_map.insert(index, Some(mapped));
                            Some(Ok(StreamEvent::PartStart {
                                index: mapped,
                                kind: PartKind::Text,
                            }))
                        }
                        // Genuine visible text -> suppress (see fn docs).
                        PartKind::Text => {
                            index_map.insert(index, None);
                            None
                        }
                        // Real tool call -> keep, but renumber.
                        PartKind::ToolCall { .. } => {
                            other_tool_seen = true;
                            let mapped = next_index;
                            next_index += 1;
                            index_map.insert(index, Some(mapped));
                            Some(Ok(StreamEvent::PartStart {
                                index: mapped,
                                kind,
                            }))
                        }
                        // Reasoning / refusal / builtin / continuation ->
                        // keep, renumber.
                        _ => {
                            let mapped = next_index;
                            next_index += 1;
                            index_map.insert(index, Some(mapped));
                            Some(Ok(StreamEvent::PartStart {
                                index: mapped,
                                kind,
                            }))
                        }
                    },
                    StreamEvent::Delta { index, delta } => match index_map.get(&index) {
                        Some(Some(mapped)) => Some(Ok(StreamEvent::Delta {
                            index: *mapped,
                            delta,
                        })),
                        _ => None,
                    },
                    StreamEvent::PartUpdate { index, update } => match index_map.get(&index) {
                        Some(Some(mapped)) => Some(Ok(StreamEvent::PartUpdate {
                            index: *mapped,
                            update,
                        })),
                        _ => None,
                    },
                    StreamEvent::PartEnd { index } => match index_map.get(&index) {
                        Some(Some(mapped)) => Some(Ok(StreamEvent::PartEnd { index: *mapped })),
                        _ => None,
                    },
                    StreamEvent::Done {
                        finish_reason,
                        usage,
                    } => {
                        if synth_seen {
                            let finish_reason = if matches!(finish_reason, FinishReason::ToolCalls)
                                && !other_tool_seen
                            {
                                FinishReason::Stop
                            } else {
                                finish_reason
                            };
                            Some(Ok(StreamEvent::Done {
                                finish_reason,
                                usage,
                            }))
                        } else if other_tool_seen {
                            // Intermediate multi-turn turn: the model
                            // called a real tool and hasn't emitted the
                            // structured answer yet. Pass through.
                            Some(Ok(StreamEvent::Done {
                                finish_reason,
                                usage,
                            }))
                        } else {
                            Some(Err(Error::streaming(
                                "json_coercion: model did not invoke the structured-response \
                                 tool and made no other tool call — the request asked for \
                                 structured output but the model returned free-form text",
                            )))
                        }
                    }
                    StreamEvent::Error { error } => Some(Ok(StreamEvent::Error { error })),
                }
            })
            .filter_map(futures_util::future::ready),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::generate;
    use crate::types::{FunctionCall, PartKind, Usage};
    use crate::{AssistantPart, Capabilities, Config, ConfigBuilder};
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};

    #[derive(Debug)]
    struct MockProvider {
        last_raw: Arc<Mutex<Option<RawConfig>>>,
        events: Mutex<Option<Vec<Result<StreamEvent, Error>>>>,
    }

    impl MockProvider {
        fn new(events: Vec<Result<StreamEvent, Error>>) -> Self {
            Self {
                last_raw: Arc::new(Mutex::new(None)),
                events: Mutex::new(Some(events)),
            }
        }
        fn last_raw(&self) -> RawConfig {
            self.last_raw.lock().unwrap().clone().expect("called")
        }
    }

    #[async_trait]
    impl crate::Provider for MockProvider {
        async fn generate(&self, _prompt: &Prompt, config: &RawConfig) -> Result<Response, Error> {
            *self.last_raw.lock().unwrap() = Some(config.clone());
            let events = self.events.lock().unwrap().take().unwrap_or_default();
            Ok(Response::from_stream(futures_util::stream::iter(events)))
        }
    }

    fn json_schema_rf(name: &str, schema_json: &str) -> ResponseFormat {
        let schema = serde_json::value::RawValue::from_string(schema_json.to_string()).unwrap();
        ResponseFormat::JsonSchema {
            name: name.to_string(),
            schema: std::borrow::Cow::Owned(schema),
            strict: true,
        }
    }

    #[test]
    fn unique_tool_name_appends_counter_on_collision() {
        let f = |n: &str| {
            Tool::Function(Function {
                name: n.to_string(),
                description: None,
                parameters: std::borrow::Cow::Owned(
                    serde_json::value::RawValue::from_string("{}".to_string()).unwrap(),
                ),
            })
        };
        assert_eq!(
            unique_tool_name("respond_with_json", None),
            "respond_with_json"
        );
        assert_eq!(
            unique_tool_name("respond_with_json", Some(&[f("other")])),
            "respond_with_json"
        );
        assert_eq!(
            unique_tool_name("respond_with_json", Some(&[f("respond_with_json")])),
            "respond_with_json_2"
        );
        assert_eq!(
            unique_tool_name(
                "respond_with_json",
                Some(&[f("respond_with_json"), f("respond_with_json_2")])
            ),
            "respond_with_json_3"
        );
    }

    #[test]
    fn noops_when_response_format_unset() {
        let prompt = Prompt::user("hi");
        let cfg = Config::builder("claude-sonnet-4-5").build();
        let mut prompt_cow: Cow<'_, Prompt> = Cow::Borrowed(&prompt);
        let mut raw_cow: Cow<'_, RawConfig> = Cow::Borrowed(cfg.raw());
        let transform = JsonCoercionMiddleware
            .apply(
                &mut prompt_cow,
                &mut raw_cow,
                &Capabilities::for_model(&cfg.raw().model),
            )
            .unwrap();
        assert!(transform.is_none());
        assert!(matches!(prompt_cow, Cow::Borrowed(_)));
        assert!(matches!(raw_cow, Cow::Borrowed(_)));
    }

    #[test]
    fn noops_when_caps_natively_support_schema() {
        let prompt = Prompt::user("hi");
        let cfg = Config::builder("gpt-5")
            .response_format(json_schema_rf("Point", r#"{"type":"object"}"#))
            .build();
        let mut prompt_cow: Cow<'_, Prompt> = Cow::Borrowed(&prompt);
        let mut raw_cow: Cow<'_, RawConfig> = Cow::Borrowed(cfg.raw());
        let transform = JsonCoercionMiddleware
            .apply(
                &mut prompt_cow,
                &mut raw_cow,
                &Capabilities::for_model(&cfg.raw().model),
            )
            .unwrap();
        assert!(transform.is_none());
        assert!(matches!(raw_cow, Cow::Borrowed(_)));
    }

    #[test]
    fn rewrites_request_on_anthropic() {
        let prompt = Prompt::user("hi");
        let cfg = Config::builder("claude-sonnet-4-5")
            .response_format(json_schema_rf("Point", r#"{"type":"object"}"#))
            .build();
        let mut prompt_cow: Cow<'_, Prompt> = Cow::Borrowed(&prompt);
        let mut raw_cow: Cow<'_, RawConfig> = Cow::Borrowed(cfg.raw());
        let transform = JsonCoercionMiddleware
            .apply(
                &mut prompt_cow,
                &mut raw_cow,
                &Capabilities::for_model(&cfg.raw().model),
            )
            .unwrap();
        assert!(matches!(raw_cow, Cow::Owned(_)));

        let raw = &*raw_cow;
        assert!(raw.response_format.is_none());
        let tools = raw.tools.as_ref().expect("synth tool added");
        assert_eq!(tools.len(), 1);
        let Tool::Function(synth) = &tools[0] else {
            panic!("synth tool should be a function");
        };
        assert!(synth.name.starts_with("respond_with_json"));
        match &raw.tool_choice {
            Some(ToolChoice::Function { name }) => assert_eq!(name, &synth.name),
            other => panic!("expected forced tool_choice, got {other:?}"),
        }
        assert!(transform.is_some());
    }

    #[tokio::test]
    async fn stream_rewrites_tool_call_to_text() {
        let synth_name = "respond_with_json".to_string();
        let events: Vec<Result<StreamEvent, Error>> = vec![
            Ok(StreamEvent::PartStart {
                index: 0,
                kind: PartKind::ToolCall {
                    call_id: "c1".to_string(),
                    name: synth_name.clone(),
                },
            }),
            Ok(StreamEvent::Delta {
                index: 0,
                delta: r#"{"answer":"#.to_string(),
            }),
            Ok(StreamEvent::Delta {
                index: 0,
                delta: r#" 42}"#.to_string(),
            }),
            Ok(StreamEvent::PartEnd { index: 0 }),
            Ok(StreamEvent::Done {
                finish_reason: FinishReason::ToolCalls,
                usage: Usage::default(),
            }),
        ];
        let provider = MockProvider::new(events);
        let prompt = Prompt::user("what is the meaning of life?");
        let config = Config::builder("claude-sonnet-4-5")
            .response_format(json_schema_rf("Answer", r#"{"type":"object"}"#))
            .build();

        let response = generate(&provider, &prompt, &config).await.unwrap();
        let complete = response.buffer().await.unwrap();

        assert_eq!(complete.text(), r#"{"answer": 42}"#);
        assert!(matches!(complete.finish_reason, FinishReason::Stop));
        assert!(complete.function_calls().is_empty());

        let recv = provider.last_raw();
        assert!(recv.response_format.is_none());
        let tools = recv.tools.expect("tools forwarded");
        assert!(tools
            .iter()
            .any(|t| matches!(t, Tool::Function(f) if f.name.starts_with("respond_with_json"))));
        assert!(matches!(
            recv.tool_choice,
            Some(ToolChoice::Function { .. })
        ));
    }

    #[tokio::test]
    async fn preserves_other_tool_calls() {
        let synth_name = "respond_with_json".to_string();
        let events: Vec<Result<StreamEvent, Error>> = vec![
            Ok(StreamEvent::PartStart {
                index: 0,
                kind: PartKind::ToolCall {
                    call_id: "c1".to_string(),
                    name: synth_name.clone(),
                },
            }),
            Ok(StreamEvent::Delta {
                index: 0,
                delta: r#"{"x":1}"#.to_string(),
            }),
            Ok(StreamEvent::PartEnd { index: 0 }),
            Ok(StreamEvent::PartStart {
                index: 1,
                kind: PartKind::ToolCall {
                    call_id: "c2".to_string(),
                    name: "get_weather".to_string(),
                },
            }),
            Ok(StreamEvent::Delta {
                index: 1,
                delta: r#"{"city":"NY"}"#.to_string(),
            }),
            Ok(StreamEvent::PartEnd { index: 1 }),
            Ok(StreamEvent::Done {
                finish_reason: FinishReason::ToolCalls,
                usage: Usage::default(),
            }),
        ];
        let provider = MockProvider::new(events);
        let prompt = Prompt::user("hi");
        let config = ConfigBuilder::new("claude-sonnet-4-5")
            .response_format(json_schema_rf("Answer", r#"{"type":"object"}"#))
            .tools(vec![Tool::Function(Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: std::borrow::Cow::Owned(
                    serde_json::value::RawValue::from_string("{}".to_string()).unwrap(),
                ),
            })])
            .build();

        let response = generate(&provider, &prompt, &config).await.unwrap();
        let complete = response.buffer().await.unwrap();

        let mut saw_text = false;
        let mut saw_real_call = false;
        for part in &complete.content {
            if let AssistantPart::Text { content, .. } = part {
                assert_eq!(content, r#"{"x":1}"#);
                saw_text = true;
            }
            if let AssistantPart::ToolCall(FunctionCall { name, .. }) = part {
                if name == "get_weather" {
                    saw_real_call = true;
                }
            }
        }
        assert!(saw_text);
        assert!(saw_real_call);
        assert!(matches!(complete.finish_reason, FinishReason::ToolCalls));
    }

    #[test]
    fn included_in_default_middleware_when_caps_deficient() {
        use crate::middleware::default_middleware;
        let caps_full = Capabilities {
            native_json_mode: true,
            response_schema: true,
            response_schema_with_tools: true,
        };
        assert!(default_middleware(&caps_full).is_empty());

        let caps_anthropic = Capabilities::anthropic("claude-sonnet-4-5");
        assert_eq!(default_middleware(&caps_anthropic).len(), 1);
    }

    /// On a model that supports schema natively but not schema+tools
    /// (Gemini 2.5), the polyfill adds the synth tool alongside the
    /// caller's tools and switches `tool_choice` to `Required` — the
    /// model can call either a real tool or the synth tool when ready
    /// to emit the structured answer.
    #[test]
    fn schema_plus_tools_uses_tool_choice_required() {
        let prompt = Prompt::user("hi");
        let cfg = ConfigBuilder::new("gemini-2.5-flash")
            .response_format(json_schema_rf("X", r#"{"type":"object"}"#))
            .tools(vec![Tool::Function(Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: std::borrow::Cow::Owned(
                    serde_json::value::RawValue::from_string("{}".to_string()).unwrap(),
                ),
            })])
            .build();
        let mut prompt_cow: Cow<'_, Prompt> = Cow::Borrowed(&prompt);
        let mut raw_cow: Cow<'_, RawConfig> = Cow::Borrowed(cfg.raw());
        let transform = JsonCoercionMiddleware
            .apply(
                &mut prompt_cow,
                &mut raw_cow,
                &Capabilities::for_model(&cfg.raw().model),
            )
            .unwrap();
        assert!(transform.is_some());
        let raw = &*raw_cow;
        assert!(raw.response_format.is_none());
        assert!(matches!(raw.tool_choice, Some(ToolChoice::Required)));
        let tools = raw.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 2);
        assert!(tools
            .iter()
            .any(|t| matches!(t, Tool::Function(f) if f.name == "get_weather")));
        assert!(tools
            .iter()
            .any(|t| matches!(t, Tool::Function(f) if f.name.starts_with("respond_with_json"))));
    }

    /// When no caller function tools are present, the polyfill still
    /// forces `tool_choice` to the synth tool (the existing
    /// no-other-tools behavior). This guards against regressing the
    /// single-tool path while introducing the multi-tool one.
    #[test]
    fn schema_without_caller_tools_forces_synth_tool() {
        let prompt = Prompt::user("hi");
        let cfg = ConfigBuilder::new("claude-sonnet-4-5")
            .response_format(json_schema_rf("X", r#"{"type":"object"}"#))
            .build();
        let mut prompt_cow: Cow<'_, Prompt> = Cow::Borrowed(&prompt);
        let mut raw_cow: Cow<'_, RawConfig> = Cow::Borrowed(cfg.raw());
        let _ = JsonCoercionMiddleware
            .apply(
                &mut prompt_cow,
                &mut raw_cow,
                &Capabilities::for_model(&cfg.raw().model),
            )
            .unwrap();
        let raw = &*raw_cow;
        let synth_name = match &raw.tool_choice {
            Some(ToolChoice::Function { name }) => name.clone(),
            other => panic!("expected forced synth tool, got {other:?}"),
        };
        assert!(synth_name.starts_with("respond_with_json"));
    }

    /// `tool_choice: Function { name }` pins a specific caller tool —
    /// the polyfill cannot reconcile that with structured output and
    /// must fail loudly rather than silently breaking one constraint.
    #[test]
    fn forced_specific_tool_choice_errors() {
        let prompt = Prompt::user("hi");
        let cfg = ConfigBuilder::new("claude-sonnet-4-5")
            .response_format(json_schema_rf("X", r#"{"type":"object"}"#))
            .tools(vec![Tool::Function(Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: std::borrow::Cow::Owned(
                    serde_json::value::RawValue::from_string("{}".to_string()).unwrap(),
                ),
            })])
            .tool_choice(ToolChoice::Function {
                name: "get_weather".to_string(),
            })
            .build();
        let mut prompt_cow: Cow<'_, Prompt> = Cow::Borrowed(&prompt);
        let mut raw_cow: Cow<'_, RawConfig> = Cow::Borrowed(cfg.raw());
        let caps = Capabilities::for_model(&cfg.raw().model);
        let err = match JsonCoercionMiddleware.apply(&mut prompt_cow, &mut raw_cow, &caps) {
            Ok(_) => panic!("forced tool conflicts with structured response"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(msg.contains("tool_choice=Function"), "got: {msg}");
        assert!(msg.contains("get_weather"), "got: {msg}");
    }

    /// End-to-end: caller has both a real tool and a schema. Turn 1
    /// the model picks the real tool to gather info; the caller
    /// returns a `ToolResult` and re-invokes `generate()` with the
    /// same config. Turn 2 the model picks the synth tool, the
    /// polyfill unwraps it to text, and the caller sees the
    /// structured answer.
    ///
    /// This is the load-bearing test for the schema+tools polyfill
    /// claim — "multi-turn works naturally because the middleware
    /// re-adds the synth tool every turn."
    #[tokio::test]
    async fn multi_turn_schema_plus_tools_real_then_synth() {
        use crate::types::UserPart;

        let config = ConfigBuilder::new("claude-sonnet-4-5")
            .response_format(json_schema_rf(
                "Answer",
                r#"{"type":"object","properties":{"answer":{"type":"string"}}}"#,
            ))
            .tools(vec![Tool::Function(Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: std::borrow::Cow::Owned(
                    serde_json::value::RawValue::from_string(
                        r#"{"type":"object","properties":{"city":{"type":"string"}}}"#.to_string(),
                    )
                    .unwrap(),
                ),
            })])
            .build();

        // Turn 1: model decides to call the real tool to gather info.
        let turn1_provider = MockProvider::new(vec![
            Ok(StreamEvent::PartStart {
                index: 0,
                kind: PartKind::ToolCall {
                    call_id: "c1".to_string(),
                    name: "get_weather".to_string(),
                },
            }),
            Ok(StreamEvent::Delta {
                index: 0,
                delta: r#"{"city":"Paris"}"#.to_string(),
            }),
            Ok(StreamEvent::PartEnd { index: 0 }),
            Ok(StreamEvent::Done {
                finish_reason: FinishReason::ToolCalls,
                usage: Usage::default(),
            }),
        ]);
        let prompt = Prompt::user("What's the weather in Paris? Answer in JSON.");
        let turn1_complete = generate(&turn1_provider, &prompt, &config)
            .await
            .unwrap()
            .buffer()
            .await
            .unwrap();

        // The real tool call passes through; the synth tool was offered
        // but not picked by the model.
        let calls = turn1_complete.function_calls();
        assert_eq!(calls.len(), 1, "expected 1 tool call, got {calls:?}");
        assert_eq!(calls[0].name, "get_weather");
        assert!(matches!(
            turn1_complete.finish_reason,
            FinishReason::ToolCalls
        ));

        // Verify the synth tool was offered to the model on turn 1 and
        // tool_choice was set to Required (the multi-tool path).
        let turn1_raw = turn1_provider.last_raw();
        let turn1_tools = turn1_raw.tools.as_ref().unwrap();
        assert!(turn1_tools
            .iter()
            .any(|t| matches!(t, Tool::Function(f) if f.name == "get_weather")));
        assert!(turn1_tools
            .iter()
            .any(|t| matches!(t, Tool::Function(f) if f.name == "respond_with_json")));
        assert!(matches!(turn1_raw.tool_choice, Some(ToolChoice::Required)));

        // Caller appends turn 1's assistant message + a tool result.
        let turn2_prompt = prompt.clone().with_response(&turn1_complete).with_item(
            crate::types::InputItem::User {
                content: vec![UserPart::ToolResult {
                    call_id: "c1".to_string(),
                    content: vec![UserPart::Text(
                        r#"{"temp":22,"condition":"sunny"}"#.to_string(),
                    )],
                }],
            },
        );

        // Turn 2: model is done gathering info and picks the synth
        // tool. Polyfill unwraps it to text.
        let turn2_provider = MockProvider::new(vec![
            Ok(StreamEvent::PartStart {
                index: 0,
                kind: PartKind::ToolCall {
                    call_id: "c2".to_string(),
                    name: "respond_with_json".to_string(),
                },
            }),
            Ok(StreamEvent::Delta {
                index: 0,
                delta: r#"{"answer":"sunny, 22C"}"#.to_string(),
            }),
            Ok(StreamEvent::PartEnd { index: 0 }),
            Ok(StreamEvent::Done {
                finish_reason: FinishReason::ToolCalls,
                usage: Usage::default(),
            }),
        ]);
        let turn2_complete = generate(&turn2_provider, &turn2_prompt, &config)
            .await
            .unwrap()
            .buffer()
            .await
            .unwrap();

        // Caller sees a text reply with the structured answer; no tool
        // call surfaced; finish_reason rewritten to Stop.
        assert_eq!(turn2_complete.text(), r#"{"answer":"sunny, 22C"}"#);
        assert!(turn2_complete.function_calls().is_empty());
        assert!(matches!(turn2_complete.finish_reason, FinishReason::Stop));

        // Turn 2 also had the synth tool offered (middleware idempotent
        // across turns) and Required tool_choice.
        let turn2_raw = turn2_provider.last_raw();
        let turn2_tools = turn2_raw.tools.as_ref().unwrap();
        assert!(turn2_tools
            .iter()
            .any(|t| matches!(t, Tool::Function(f) if f.name == "respond_with_json")));
        assert!(matches!(turn2_raw.tool_choice, Some(ToolChoice::Required)));
    }

    /// `tool_choice: Required` is compatible with the polyfill — the
    /// caller already said "must call a tool" and we keep that, just
    /// adding the synth as another option in the pool.
    #[test]
    fn tool_choice_required_is_preserved() {
        let prompt = Prompt::user("hi");
        let cfg = ConfigBuilder::new("gemini-2.5-flash")
            .response_format(json_schema_rf("X", r#"{"type":"object"}"#))
            .tools(vec![Tool::Function(Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: std::borrow::Cow::Owned(
                    serde_json::value::RawValue::from_string("{}".to_string()).unwrap(),
                ),
            })])
            .tool_choice(ToolChoice::Required)
            .build();
        let mut prompt_cow: Cow<'_, Prompt> = Cow::Borrowed(&prompt);
        let mut raw_cow: Cow<'_, RawConfig> = Cow::Borrowed(cfg.raw());
        let _ = JsonCoercionMiddleware
            .apply(
                &mut prompt_cow,
                &mut raw_cow,
                &Capabilities::for_model(&cfg.raw().model),
            )
            .unwrap();
        assert!(matches!(raw_cow.tool_choice, Some(ToolChoice::Required)));
    }

    /// Regression: the provider emits free text *before* the synth tool
    /// call in the same turn (`tool_choice: Required` / ANY mode doesn't
    /// gag text, and forced Anthropic tool use can be preceded by text
    /// with thinking on). `CompleteResponse::text()` concatenates all
    /// text parts, so without suppression the caller would get
    /// `preamble + json` — unparseable. The genuine text part must be
    /// dropped so `text()` is exactly the synth JSON.
    #[tokio::test]
    async fn text_before_synth_call_is_suppressed() {
        let synth_name = "respond_with_json".to_string();
        let events: Vec<Result<StreamEvent, Error>> = vec![
            // Genuine visible text the model emitted first.
            Ok(StreamEvent::PartStart {
                index: 0,
                kind: PartKind::Text,
            }),
            Ok(StreamEvent::Delta {
                index: 0,
                delta: "Here you go: ".to_string(),
            }),
            Ok(StreamEvent::PartEnd { index: 0 }),
            // Then the synth tool call carrying the structured answer.
            Ok(StreamEvent::PartStart {
                index: 1,
                kind: PartKind::ToolCall {
                    call_id: "c1".to_string(),
                    name: synth_name.clone(),
                },
            }),
            Ok(StreamEvent::Delta {
                index: 1,
                delta: r#"{"answer":42}"#.to_string(),
            }),
            Ok(StreamEvent::PartEnd { index: 1 }),
            Ok(StreamEvent::Done {
                finish_reason: FinishReason::ToolCalls,
                usage: Usage::default(),
            }),
        ];
        let provider = MockProvider::new(events);
        let prompt = Prompt::user("what is the meaning of life?");
        let config = Config::builder("claude-sonnet-4-5")
            .response_format(json_schema_rf("Answer", r#"{"type":"object"}"#))
            .build();

        let complete = generate(&provider, &prompt, &config)
            .await
            .unwrap()
            .buffer()
            .await
            .unwrap();

        // The preamble is gone; text() is exactly the synth JSON.
        assert_eq!(complete.text(), r#"{"answer":42}"#);
        assert!(matches!(complete.finish_reason, FinishReason::Stop));
        assert!(complete.function_calls().is_empty());
    }

    /// When the model never invokes the synth tool and makes no other
    /// tool call — it just free-texted instead of producing structured
    /// output — the suppressed text must not surface as a silent empty
    /// response. The coerced stream errors instead.
    #[tokio::test]
    async fn free_text_without_synth_tool_errors() {
        let events: Vec<Result<StreamEvent, Error>> = vec![
            Ok(StreamEvent::PartStart {
                index: 0,
                kind: PartKind::Text,
            }),
            Ok(StreamEvent::Delta {
                index: 0,
                delta: "I'd rather just chat.".to_string(),
            }),
            Ok(StreamEvent::PartEnd { index: 0 }),
            Ok(StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
            }),
        ];
        let provider = MockProvider::new(events);
        let prompt = Prompt::user("hi");
        let config = Config::builder("claude-sonnet-4-5")
            .response_format(json_schema_rf("Answer", r#"{"type":"object"}"#))
            .build();

        let err = generate(&provider, &prompt, &config)
            .await
            .unwrap()
            .buffer()
            .await
            .expect_err("free-text-only response should error");
        assert!(matches!(err, Error::Streaming(_)));
        assert!(err.to_string().contains("structured output"), "got: {err}");
    }
}
