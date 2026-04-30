use futures_util::StreamExt;
use ijson::{ijson, IValue};
use uuid::Uuid;

use super::google_types::*;
use super::transport::VertexTransport;
use crate::provider::LLMProvider;
use crate::sse_stream::SseStream;
use crate::types::{FinishReason, FunctionCall, InputItem, Role};
use crate::{Error, LLMRequest, Response, StreamEvent};

/// Google provider implementation via Vertex AI (for Gemini models).
pub struct GoogleProvider {
    transport: VertexTransport,
}

impl GoogleProvider {
    /// Create a new Google provider with access token authentication.
    pub fn new(project_id: String, location: String, access_token: String) -> Result<Self, Error> {
        Ok(Self {
            transport: VertexTransport::with_access_token(project_id, location, access_token)?,
        })
    }

    /// Create a new Google provider with a custom base URL (for testing).
    pub fn new_with_base_url(
        project_id: String,
        location: String,
        access_token: String,
        base_url: String,
    ) -> Result<Self, Error> {
        Ok(Self {
            transport: VertexTransport::with_access_token(project_id, location, access_token)?
                .with_base_url(base_url),
        })
    }

    /// Create a new Google provider with Application Default Credentials.
    pub async fn with_adc(project_id: String, location: String) -> Result<Self, Error> {
        Ok(Self {
            transport: VertexTransport::with_adc(project_id, location).await?,
        })
    }

    /// Convert internal request to Google format.
    fn convert_request(&self, request: &LLMRequest) -> Result<GoogleRequest, Error> {
        let mut contents: Vec<GoogleContent> = Vec::new();
        let mut system_instruction = None;

        // Gemini's `functionCall` parts have no `id` field on the wire, so
        // we synthesize call_ids on the response side. To send results back
        // we have to recover the function name by call_id from the
        // conversation history. Build the mapping in a single pass.
        let mut call_id_to_name: std::collections::HashMap<&str, &str> =
            std::collections::HashMap::new();

        // Append a part to the last content with the same role, otherwise
        // start a new content. Vertex rejects consecutive same-role contents,
        // and this also matches how function_call / function_response
        // sequences need to fold into the surrounding model/user turns.
        fn push_part(contents: &mut Vec<GoogleContent>, role: &str, part: GooglePart) {
            if let Some(last) = contents.last_mut() {
                if last.role == role {
                    last.parts.push(part);
                    return;
                }
            }
            contents.push(GoogleContent {
                role: role.to_string(),
                parts: vec![part],
            });
        }

        for item in &request.messages {
            match item {
                InputItem::Message(msg) => match msg.role {
                    Role::System => {
                        // Vertex's `systemInstruction` doesn't take a role.
                        // Setting it to "system" is the conventional choice
                        // when a role is required by the type; "user" was
                        // both wrong and misleading.
                        system_instruction = Some(GoogleContent {
                            role: "system".to_string(),
                            parts: vec![GooglePart::Text {
                                text: msg.content.clone(),
                            }],
                        });
                    }
                    Role::User => push_part(
                        &mut contents,
                        "user",
                        GooglePart::Text {
                            text: msg.content.clone(),
                        },
                    ),
                    Role::Assistant => push_part(
                        &mut contents,
                        "model",
                        GooglePart::Text {
                            text: msg.content.clone(),
                        },
                    ),
                },
                InputItem::FunctionCall(call) => {
                    call_id_to_name.insert(call.call_id.as_str(), call.name.as_str());
                    let args = serde_json::from_str(&call.arguments).map_err(|e| {
                        Error::provider("Google", format!("Invalid function arguments: {e}"))
                    })?;
                    push_part(
                        &mut contents,
                        "model",
                        GooglePart::FunctionCall {
                            function_call: GoogleFunctionCall {
                                name: call.name.clone(),
                                args,
                            },
                        },
                    );
                }
                InputItem::FunctionCallOutput { call_id, output } => {
                    let function_name = call_id_to_name
                        .get(call_id.as_str())
                        .ok_or_else(|| {
                            Error::provider(
                                "Google",
                                format!(
                                    "FunctionCallOutput references unknown call_id {call_id:?} \
                                     — no prior FunctionCall in the conversation matches it",
                                ),
                            )
                        })?
                        .to_string();
                    push_part(
                        &mut contents,
                        "user",
                        GooglePart::FunctionResponse {
                            function_response: GoogleFunctionResponse {
                                name: function_name,
                                response: encode_function_output(output),
                            },
                        },
                    );
                }
            }
        }

        let generation_config = Some(GoogleGenerationConfig {
            temperature: request.temperature,
            max_output_tokens: request.max_tokens,
            top_p: request.top_p,
        });

        let tools = request.tools.as_ref().map(|tools| {
            vec![GoogleTool {
                function_declarations: tools
                    .iter()
                    .map(|tool| GoogleFunctionDeclaration {
                        name: tool.function.name.clone(),
                        description: tool.function.description.clone(),
                        parameters: tool.function.parameters.clone(),
                    })
                    .collect(),
            }]
        });

        let google_request = GoogleRequest {
            contents,
            generation_config,
            tools,
            system_instruction,
        };

        Ok(google_request)
    }

}

/// Shape a tool's output for Gemini's `functionResponse.response` field,
/// which the API requires to be a JSON object.
///
/// - JSON objects pass through unchanged so the model receives structured
///   data it can reason about.
/// - JSON non-objects (numbers, arrays, strings, bools, null) are wrapped
///   under `{"result": <value>}` so we still satisfy the object requirement
///   without losing structure.
/// - Non-JSON strings are wrapped under `{"result": "<string>"}`.
fn encode_function_output(output: &str) -> IValue {
    match serde_json::from_str::<IValue>(output) {
        Ok(value) if value.is_object() => value,
        Ok(value) => ijson!({ "result": value }),
        Err(_) => ijson!({ "result": output }),
    }
}

#[async_trait::async_trait]
impl LLMProvider for GoogleProvider {
    async fn generate(&self, request: &LLMRequest) -> Result<Response, Error> {
        let google_request = self.convert_request(request)?;

        let endpoint = self.transport.endpoint(
            "google",
            &request.model,
            "streamGenerateContent",
            Some("alt=sse"),
        );

        let builder = self
            .transport
            .client()
            .post(&endpoint)
            .header("Content-Type", "application/json")
            .json(&google_request);
        let request_builder = self.transport.authorize(builder).await?;

        let response = request_builder.send().await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(Error::provider(
                "Google",
                format!("API error: {error_text}"),
            ));
        }

        // Create SSE stream from response (Gemini supports ?alt=sse)
        let byte_stream = response.bytes_stream();
        let sse_stream = SseStream::new(byte_stream);

        // Create a stateful processor for tracking output items
        let mut state = GoogleStreamState::default();

        let event_stream = sse_stream
            .map(move |sse_result| {
                match sse_result {
                    Ok(sse_event) => {
                        let data = sse_event.data.trim();

                        // Vertex's SSE channel terminates by stream close;
                        // there is no `[DONE]` sentinel (that is an OpenAI
                        // convention). Empty events do still occur for
                        // keep-alives.
                        if data.is_empty() {
                            return vec![];
                        }

                        // Parse the SSE data as GoogleResponse
                        match serde_json::from_str::<GoogleResponse>(data) {
                            Ok(google_response) => {
                                match convert_response_stateful(google_response, &mut state) {
                                    Ok(stream_events) => {
                                        stream_events.into_iter().map(Ok).collect()
                                    }
                                    Err(e) => vec![Err(e)],
                                }
                            }
                            Err(e) => {
                                vec![Err(Error::provider(
                                    "Google",
                                    format!("Failed to parse SSE event: {e}"),
                                ))]
                            }
                        }
                    }
                    Err(e) => vec![Err(e)],
                }
            })
            .map(|events| futures_util::stream::iter(events.into_iter()))
            .flatten();

        Ok(Response::from_stream(event_stream))
    }
}

/// State for tracking output items during streaming.
#[derive(Debug, Default)]
pub(crate) struct GoogleStreamState {
    /// Whether we've started text output. Only the first text part across
    /// the whole stream emits an `OutputItemAdded`; subsequent ones append
    /// to the same item via `ContentDelta`.
    has_text_output: bool,
}

/// Stateful version of convert_response that tracks output items to emit OutputItemAdded only once.
///
/// `pub(crate)` so unit tests in this module (and future fixture-driven tests in
/// `tests/`) can drive the conversion directly with synthetic `GoogleResponse`
/// values, without standing up a transport or a mock server.
pub(crate) fn convert_response_stateful(
    response: GoogleResponse,
    state: &mut GoogleStreamState,
) -> Result<Vec<StreamEvent>, Error> {
    let mut events = Vec::new();

    if let Some(candidate) = response.candidates.first() {
        for part in &candidate.content.parts {
            match part {
                GooglePart::Text { text } => {
                    // Only emit OutputItemAdded if we haven't started text output yet
                    if !state.has_text_output {
                        events.push(StreamEvent::OutputItemAdded {
                            item: crate::types::OutputItemInfo::Text,
                        });
                        state.has_text_output = true;
                    }
                    if !text.is_empty() {
                        events.push(StreamEvent::ContentDelta {
                            delta: text.clone(),
                        });
                    }
                }
                GooglePart::FunctionCall { function_call } => {
                    // Synthesize a single UUID for the call and prefix it
                    // for the two surfaces — `fc_<base>` for the
                    // OutputItemAdded item id, `call_<base>` for the
                    // FunctionCallComplete call_id. Sharing the base lets a
                    // consumer correlate the two events for one call.
                    //
                    // Each `functionCall` part in each chunk is treated as
                    // a fresh emission. Gemini's streaming protocol uses
                    // delta semantics (text parts only contain new text), so
                    // function calls don't repeat across chunks; if they
                    // ever do, we'd over-emit, but the previous behaviour
                    // was strictly worse (orphan Complete events).
                    let base_id = Uuid::new_v4().simple().to_string();
                    let fc_id = format!("fc_{base_id}");
                    let call_id = format!("call_{base_id}");

                    events.push(StreamEvent::OutputItemAdded {
                        item: crate::types::OutputItemInfo::FunctionCall {
                            name: function_call.name.clone(),
                            id: fc_id,
                        },
                    });
                    let function_call_obj = FunctionCall {
                        call_id,
                        name: function_call.name.clone(),
                        arguments: serde_json::to_string(&function_call.args).map_err(|e| {
                            Error::provider(
                                "Google",
                                format!("Failed to serialize function args: {e}"),
                            )
                        })?,
                    };
                    events.push(StreamEvent::FunctionCallComplete {
                        call: function_call_obj,
                    });
                }
                GooglePart::FunctionResponse { .. } => {
                    // Function responses are typically not part of the model's output
                }
            }
        }

        // Only add a Done event if this response has a finish_reason (indicates end of stream)
        if let Some(finish_reason_str) = &candidate.finish_reason {
            let finish_reason = match finish_reason_str.as_str() {
                "STOP" => FinishReason::Stop,
                "MAX_TOKENS" => FinishReason::Length,
                "SAFETY" => FinishReason::ContentFilter,
                _ => FinishReason::Stop, // Default to Stop for unknown reasons
            };

            let usage = response
                .usage_metadata
                .map(|meta| meta.into())
                .unwrap_or_default();

            events.push(StreamEvent::Done {
                finish_reason,
                usage,
            });
        }
    } else if let Some(feedback) = &response.prompt_feedback {
        // Prompt was safety-blocked. Surface as ContentFilter regardless of
        // the specific reason (SAFETY / BLOCKLIST / PROHIBITED_CONTENT / SPII
        // / OTHER) — they all mean "the model declined to respond".
        if let Some(reason) = &feedback.block_reason {
            tracing::warn!(
                block_reason = %reason,
                message = ?feedback.block_reason_message,
                "Gemini prompt was blocked",
            );
        }
        let usage = response
            .usage_metadata
            .map(|meta| meta.into())
            .unwrap_or_default();
        events.push(StreamEvent::Done {
            finish_reason: FinishReason::ContentFilter,
            usage,
        });
    } else if response.usage_metadata.is_some() {
        // If no candidates but we have usage metadata, this might be a final response
        let usage = response
            .usage_metadata
            .map(|meta| meta.into())
            .unwrap_or_default();
        events.push(StreamEvent::Done {
            finish_reason: FinishReason::Stop,
            usage,
        });
    }

    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{stream, StreamExt};

    #[tokio::test]
    async fn test_streaming_content_parsing() {
        // Simulate realistic Google streaming response chunks
        let chunk1 = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Google"}]}}]}"#;
        let chunk2 =
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":" Gemini"}]}}]}"#;
        let chunk3 = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":" is"}]}}]}"#;
        let final_chunk = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":" an AI."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"totalTokenCount":30}}"#;

        let byte_chunks: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
            Ok(bytes::Bytes::from(format!("data: {chunk1}\n\n"))),
            Ok(bytes::Bytes::from(format!("data: {chunk2}\n\n"))),
            Ok(bytes::Bytes::from(format!("data: {chunk3}\n\n"))),
            Ok(bytes::Bytes::from(format!("data: {final_chunk}\n\n"))),
            Ok(bytes::Bytes::from("data: [DONE]\n\n")),
        ];

        let byte_stream = stream::iter(byte_chunks);
        let sse_stream = crate::sse_stream::SseStream::new(byte_stream);

        // Process events through our Gemini SSE handler
        let mut events = Vec::new();
        let mut shared_state = GoogleStreamState::default(); // Share state across SSE events like real usage

        // Collect all events using StreamExt::next
        let mut sse_stream = sse_stream;
        while let Some(sse_result) = sse_stream.next().await {
            let sse_event = sse_result.expect("SSE should parse correctly");
            let data = sse_event.data.trim();

            if data == "[DONE]" || data.is_empty() {
                continue;
            }

            // Parse as GoogleResponse
            match serde_json::from_str::<GoogleResponse>(data) {
                Ok(response) => match convert_response_stateful(response, &mut shared_state) {
                    Ok(stream_events) => {
                        events.extend(stream_events);
                    }
                    Err(e) => panic!("Should parse successfully: {e}"),
                },
                Err(e) => panic!("Should parse JSON successfully: {e}"),
            }
        }

        // Verify we got the expected events
        let content_events: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ContentDelta { delta } => Some(delta.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(content_events, vec!["Google", " Gemini", " is", " an AI."]);

        // Verify we got exactly one Done event at the end
        let done_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::Done { .. }))
            .collect();

        assert_eq!(done_events.len(), 1);

        // The Done event should be the last event
        assert!(matches!(events.last(), Some(StreamEvent::Done { .. })));
    }

    /// Vertex's `systemInstruction` should NOT carry `role: "user"`. The
    /// canonical role for a system instruction is `"system"` (or omitted).
    /// The previous code mislabeled it as `"user"`, which is misleading and
    /// may confuse some models.
    #[test]
    fn system_instruction_role_is_not_user() {
        use crate::types::InputItem;
        let req = LLMRequest {
            model: "gemini".to_string(),
            messages: vec![
                InputItem::system("you are helpful"),
                InputItem::user("hi"),
            ],
            temperature: None,
            max_tokens: None,
            top_p: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            store: None,
        };
        let provider = GoogleProvider::new(
            "p".to_string(),
            "us-east1".to_string(),
            "tok".to_string(),
        )
        .unwrap();
        let body = provider.convert_request(&req).unwrap();
        let json = serde_json::to_value(&body).unwrap();
        let role = json["systemInstruction"]["role"].as_str();
        assert!(
            role != Some("user"),
            "systemInstruction must not carry role: 'user' (got {role:?})",
        );
    }

    /// Vertex rejects consecutive same-role contents in `contents`. If the
    /// caller hands us two adjacent assistant messages, fold them into a
    /// single content entry with multiple parts.
    #[test]
    fn consecutive_same_role_messages_are_merged() {
        use crate::types::InputItem;
        let req = LLMRequest {
            model: "gemini".to_string(),
            messages: vec![
                InputItem::user("first user"),
                InputItem::assistant("first assistant"),
                InputItem::assistant("second assistant"),
                InputItem::user("second user"),
            ],
            temperature: None,
            max_tokens: None,
            top_p: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            store: None,
        };
        let provider = GoogleProvider::new(
            "p".to_string(),
            "us-east1".to_string(),
            "tok".to_string(),
        )
        .unwrap();
        let body = provider.convert_request(&req).unwrap();
        // Expect exactly 3 contents: user, model (with 2 parts), user.
        assert_eq!(
            body.contents.len(),
            3,
            "consecutive same-role messages should fold; got {} entries: {:?}",
            body.contents.len(),
            body.contents.iter().map(|c| &c.role).collect::<Vec<_>>(),
        );
        assert_eq!(body.contents[1].role, "model");
        assert_eq!(
            body.contents[1].parts.len(),
            2,
            "merged model content should have two text parts",
        );
    }

    /// Each emitted function call must produce a paired `OutputItemAdded`
    /// and `FunctionCallComplete` whose ids derive from the same base UUID.
    /// The previous implementation deduped `OutputItemAdded` by name+args
    /// while regenerating the call_id on every chunk, so a repeated
    /// emission produced an orphan `FunctionCallComplete` with no
    /// preceding `Added`.
    #[test]
    fn function_call_emits_paired_ids_per_occurrence() {
        fn pair(events: &[StreamEvent]) -> (String, String) {
            assert_eq!(
                events.len(),
                2,
                "expected [Added, Complete] for one call, got {events:?}",
            );
            match (&events[0], &events[1]) {
                (
                    StreamEvent::OutputItemAdded {
                        item: crate::types::OutputItemInfo::FunctionCall { id, .. },
                    },
                    StreamEvent::FunctionCallComplete { call },
                ) => (id.clone(), call.call_id.clone()),
                other => panic!("expected [Added, Complete], got {other:?}"),
            }
        }

        let chunk = r#"{"candidates":[{"content":{"role":"model","parts":[
            {"functionCall":{"name":"get_weather","args":{"city":"Paris"}}}
        ]}}]}"#;

        let mut state = GoogleStreamState::default();
        let r1: GoogleResponse = serde_json::from_str(chunk).unwrap();
        let events1 = convert_response_stateful(r1, &mut state).unwrap();
        let (added1, complete1) = pair(&events1);

        // Second emission of the same call — historically this lost the
        // `Added` event (deduped) but still emitted `Complete` (orphan).
        let r2: GoogleResponse = serde_json::from_str(chunk).unwrap();
        let events2 = convert_response_stateful(r2, &mut state).unwrap();
        let (added2, complete2) = pair(&events2);

        for (added, complete) in [(&added1, &complete1), (&added2, &complete2)] {
            let base_added = added.strip_prefix("fc_").expect("fc_*");
            let base_complete = complete.strip_prefix("call_").expect("call_*");
            assert_eq!(
                base_added, base_complete,
                "OutputItemAdded and FunctionCallComplete must share a UUID base",
            );
        }
        assert_ne!(
            added1, added2,
            "two distinct emissions should yield distinct ids",
        );
    }

    /// Gemini doesn't return IDs on `functionCall` parts, so we have to
    /// rebuild the call_id → name map ourselves from the conversation
    /// history. The previous code matched positionally (Nth response
    /// pairs with Nth call), which silently mis-matches when the caller
    /// sends responses out of order. Use distinct function names so the
    /// bug surfaces if positional matching ever creeps back in.
    #[test]
    fn function_call_output_resolves_name_via_call_id_not_position() {
        use crate::types::{FunctionCall, InputItem};
        let req = LLMRequest::new(
            "gemini",
            vec![
                InputItem::user("hi"),
                InputItem::FunctionCall(FunctionCall {
                    call_id: "call_one".to_string(),
                    name: "get_weather".to_string(),
                    arguments: "{}".to_string(),
                }),
                InputItem::FunctionCall(FunctionCall {
                    call_id: "call_two".to_string(),
                    name: "get_time".to_string(),
                    arguments: "{}".to_string(),
                }),
                // Responses arrive in the reverse order.
                InputItem::function_call_output("call_two".to_string(), r#"{"x":1}"#.to_string()),
                InputItem::function_call_output("call_one".to_string(), r#"{"y":2}"#.to_string()),
            ],
        );
        let provider = GoogleProvider::new(
            "p".to_string(),
            "us-east1".to_string(),
            "tok".to_string(),
        )
        .unwrap();
        let body = provider.convert_request(&req).unwrap();
        let last = body.contents.last().unwrap();
        let names: Vec<&str> = last
            .parts
            .iter()
            .filter_map(|p| match p {
                GooglePart::FunctionResponse { function_response } => {
                    Some(function_response.name.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            names,
            vec!["get_time", "get_weather"],
            "responses should resolve via call_id, not position",
        );
    }

    /// Sending a FunctionCallOutput for an unknown call_id is a programmer
    /// bug — the previous code substituted the literal string "unknown"
    /// (which Vertex would then reject). Now we surface a clear error.
    #[test]
    fn function_call_output_with_unknown_call_id_errors() {
        use crate::types::InputItem;
        let req = LLMRequest::new(
            "gemini",
            vec![
                InputItem::user("hi"),
                InputItem::function_call_output("call_unknown".to_string(), "result".to_string()),
            ],
        );
        let provider = GoogleProvider::new(
            "p".to_string(),
            "us-east1".to_string(),
            "tok".to_string(),
        )
        .unwrap();
        let err = provider
            .convert_request(&req)
            .expect_err("unknown call_id should produce an error");
        let msg = format!("{err}");
        assert!(
            msg.contains("call_unknown"),
            "error message should reference the unknown id; got: {msg}",
        );
    }

    /// Gemini 2.5 thinking models report tokens spent on internal
    /// reasoning under `thoughtsTokenCount`, and prompt-cache hits under
    /// `cachedContentTokenCount`. Both must reach the unified `Usage`.
    #[test]
    fn gemini_thoughts_and_cache_tokens_propagate() {
        let json = r#"{
            "candidates":[{
                "content":{"role":"model","parts":[{"text":""}]},
                "finishReason":"STOP"
            }],
            "usageMetadata":{
                "promptTokenCount":10,
                "candidatesTokenCount":20,
                "thoughtsTokenCount":15,
                "cachedContentTokenCount":5
            }
        }"#;
        let response: GoogleResponse = serde_json::from_str(json).unwrap();
        let mut state = GoogleStreamState::default();
        let events = convert_response_stateful(response, &mut state).unwrap();
        let usage = events
            .iter()
            .find_map(|e| match e {
                StreamEvent::Done { usage, .. } => Some(usage),
                _ => None,
            })
            .expect("Done event with usage");
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(usage.reasoning_tokens, Some(15));
        assert_eq!(usage.cache_read_input_tokens, Some(5));
    }

    /// Build a request with one FunctionCall + one FunctionCallOutput so we
    /// can assert how the latter gets shaped into Vertex's `functionResponse`.
    fn request_with_tool_output(call_id: &str, output: &str) -> LLMRequest {
        use crate::types::{FunctionCall, InputItem};
        LLMRequest::new(
            "gemini",
            vec![
                InputItem::user("hi"),
                InputItem::FunctionCall(FunctionCall {
                    call_id: call_id.to_string(),
                    name: "get_weather".to_string(),
                    arguments: r#"{"city":"Paris"}"#.to_string(),
                }),
                InputItem::function_call_output(call_id.to_string(), output.to_string()),
            ],
        )
    }

    fn extract_function_response(req: &GoogleRequest) -> serde_json::Value {
        let json = serde_json::to_value(req).unwrap();
        let contents = json["contents"].as_array().unwrap();
        for content in contents.iter().rev() {
            for part in content["parts"].as_array().unwrap() {
                if let Some(fr) = part.get("functionResponse") {
                    return fr["response"].clone();
                }
            }
        }
        panic!("no functionResponse part found in: {json}");
    }

    /// JSON-shaped tool output (the common case) should reach Gemini as
    /// structured JSON. The previous behaviour wrapped every output under
    /// `{"result": "<json string>"}` so the model only ever saw a stringified
    /// blob it had to re-parse — and silently double-encoded JSON outputs.
    #[test]
    fn function_call_output_with_json_object_is_unwrapped() {
        let provider = GoogleProvider::new(
            "p".to_string(),
            "us-east1".to_string(),
            "tok".to_string(),
        )
        .unwrap();
        let req = provider
            .convert_request(&request_with_tool_output(
                "c1",
                r#"{"temp":72,"unit":"F"}"#,
            ))
            .unwrap();
        let response = extract_function_response(&req);
        assert_eq!(
            response,
            serde_json::json!({"temp":72,"unit":"F"}),
            "JSON object outputs should be passed through, not wrapped",
        );
    }

    /// Non-JSON string outputs still need to satisfy Gemini's "response must
    /// be a JSON object" requirement, so wrap under `{"result": "..."}`.
    #[test]
    fn function_call_output_with_plain_string_is_wrapped_under_result() {
        let provider = GoogleProvider::new(
            "p".to_string(),
            "us-east1".to_string(),
            "tok".to_string(),
        )
        .unwrap();
        let req = provider
            .convert_request(&request_with_tool_output("c1", "Sunny, 72F"))
            .unwrap();
        let response = extract_function_response(&req);
        assert_eq!(
            response,
            serde_json::json!({"result": "Sunny, 72F"}),
            "non-JSON outputs are wrapped under 'result'",
        );
    }

    /// When Vertex blocks a prompt at the safety layer, it returns
    /// `{"promptFeedback":{...}}` with no `candidates` array. Deserializing
    /// that previously failed because `candidates: Vec<...>` was required —
    /// surfaced to callers as a misleading "Failed to parse SSE event"
    /// instead of a real `Done`/blocked signal.
    #[test]
    fn prompt_feedback_only_response_parses() {
        let json = r#"{"promptFeedback":{"blockReason":"SAFETY"}}"#;
        let response: GoogleResponse = serde_json::from_str(json)
            .expect("safety-blocked response with no candidates must parse");

        let mut state = GoogleStreamState::default();
        let events = convert_response_stateful(response, &mut state).unwrap();

        let done = events.iter().find_map(|e| match e {
            StreamEvent::Done { finish_reason, .. } => Some(finish_reason.clone()),
            _ => None,
        });
        assert_eq!(
            done,
            Some(FinishReason::ContentFilter),
            "promptFeedback.blockReason should yield Done with ContentFilter",
        );
    }

    /// Vertex sometimes emits a candidate with `finishReason: SAFETY` and
    /// `content: {role: model}` (no `parts`). `parts` was a required field;
    /// the parse failure leaked through as a generic "Failed to parse SSE
    /// event" error.
    #[test]
    fn candidate_with_no_parts_parses() {
        let json = r#"{
            "candidates":[{"content":{"role":"model"},"finishReason":"SAFETY"}]
        }"#;
        let response: GoogleResponse =
            serde_json::from_str(json).expect("candidate without parts must parse");

        let mut state = GoogleStreamState::default();
        let events = convert_response_stateful(response, &mut state).unwrap();
        let done = events.iter().find_map(|e| match e {
            StreamEvent::Done { finish_reason, .. } => Some(finish_reason.clone()),
            _ => None,
        });
        assert_eq!(done, Some(FinishReason::ContentFilter));
    }

    #[test]
    fn request_body_uses_camel_case_keys() {
        // Vertex AI's REST API expects camelCase keys: `generationConfig`,
        // `systemInstruction`, `functionDeclarations`. snake_case keys are
        // silently ignored, which means temperature / maxOutputTokens / topP
        // / system instructions / tool definitions all silently drop on the
        // floor. Lock in the correct shape.
        use crate::types::{Function, InputItem, Tool, ToolType};

        let tool = Tool {
            r#type: ToolType::Function,
            function: Function {
                name: "get_weather".to_string(),
                description: "Get the weather".to_string(),
                parameters: serde_json::from_str(
                    r#"{"type":"object","properties":{}}"#,
                )
                .unwrap(),
            },
        };
        let request = LLMRequest {
            model: "gemini-1.5-pro".to_string(),
            messages: vec![
                InputItem::system("You are helpful."),
                InputItem::user("Hi"),
            ],
            temperature: Some(0.5),
            max_tokens: Some(128),
            top_p: Some(0.9),
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: Some(vec![tool]),
            tool_choice: None,
            parallel_tool_calls: None,
            store: None,
        };

        let provider = GoogleProvider::new(
            "p".to_string(),
            "us-east1".to_string(),
            "tok".to_string(),
        )
        .unwrap();
        let google_request = provider.convert_request(&request).unwrap();
        let json = serde_json::to_value(&google_request).unwrap();

        // Top-level keys
        assert!(
            json.get("generationConfig").is_some(),
            "request must serialize 'generationConfig' (camelCase), got: {json}",
        );
        assert!(
            json.get("systemInstruction").is_some(),
            "request must serialize 'systemInstruction' (camelCase), got: {json}",
        );
        assert!(
            json.get("generation_config").is_none(),
            "request must NOT contain snake_case 'generation_config'",
        );
        assert!(
            json.get("system_instruction").is_none(),
            "request must NOT contain snake_case 'system_instruction'",
        );

        // Inside generationConfig
        let gen_cfg = &json["generationConfig"];
        assert!(gen_cfg.get("temperature").is_some(), "missing temperature");
        assert!(gen_cfg.get("maxOutputTokens").is_some(), "missing maxOutputTokens");
        assert!(gen_cfg.get("topP").is_some(), "missing topP");
        assert!(
            gen_cfg.get("max_output_tokens").is_none(),
            "generationConfig must NOT contain snake_case 'max_output_tokens'",
        );
        assert!(
            gen_cfg.get("top_p").is_none(),
            "generationConfig must NOT contain snake_case 'top_p'",
        );

        // Inside tools[0]
        let tool_obj = &json["tools"][0];
        assert!(
            tool_obj.get("functionDeclarations").is_some(),
            "tools[0] must serialize 'functionDeclarations' (camelCase)",
        );
        assert!(
            tool_obj.get("function_declarations").is_none(),
            "tools[0] must NOT contain snake_case 'function_declarations'",
        );
    }

    #[tokio::test]
    async fn test_real_google_sse_response_format() {
        // Test the exact Google SSE format you provided to see where the concatenation happens
        let real_google_sse = r#"data: {"candidates": [{"content": {"role": "model","parts": [{"text": "The"}]}}],"usageMetadata": {"trafficType": "ON_DEMAND"},"modelVersion": "gemini-1.5-pro-002","createTime": "2025-07-22T11:00:54.652094Z","responseId": "Zm9_aL7mJ8bAxN8PnfeY2AY"}

data: {"candidates": [{"content": {"role": "model","parts": [{"text": " weather in Tokyo"}]}}],"usageMetadata": {"trafficType": "ON_DEMAND"},"modelVersion": "gemini-1.5-pro-002","createTime": "2025-07-22T11:00:54.652094Z","responseId": "Zm9_aL7mJ8bAxN8PnfeY2AY"}

data: {"candidates": [{"content": {"role": "model","parts": [{"text": ", Japan is currently sunny"}]}}],"usageMetadata": {"trafficType": "ON_DEMAND"},"modelVersion": "gemini-1.5-pro-002","createTime": "2025-07-22T11:00:54.652094Z","responseId": "Zm9_aL7mJ8bAxN8PnfeY2AY"}

"#;

        let byte_chunks: Vec<Result<bytes::Bytes, std::io::Error>> =
            vec![Ok(bytes::Bytes::from(real_google_sse))];

        let byte_stream = stream::iter(byte_chunks);
        let sse_stream = crate::sse_stream::SseStream::new(byte_stream);

        let mut all_parsed_events = Vec::new();
        let mut event_count = 0;
        let mut shared_state = GoogleStreamState::default();

        let mut sse_stream = sse_stream;
        while let Some(sse_result) = sse_stream.next().await {
            let sse_event = sse_result.expect("SSE should parse correctly");
            event_count += 1;
            let data = sse_event.data.trim();
            if data == "[DONE]" || data.is_empty() {
                continue;
            }
            let google_response: GoogleResponse =
                serde_json::from_str(data).expect("JSON should parse");
            let stream_events = convert_response_stateful(google_response, &mut shared_state)
                .expect("conversion should succeed");
            all_parsed_events.extend(stream_events);
        }

        assert_eq!(event_count, 3, "Should have 3 separate SSE events");
        assert_eq!(
            all_parsed_events.len(),
            4,
            "Should generate 4 stream events (1 OutputItemAdded + 3 ContentDelta)"
        );
    }
}
