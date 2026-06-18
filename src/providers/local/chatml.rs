//! ChatML chat template with Hermes/Qwen-style tool calling.
//!
//! The default [`ChatTemplate`] implementation. ChatML is the
//! dominant format across open instruction-tuned models — Qwen,
//! Hermes, SmolLM, OLMo, many Mistral fine-tunes. Turns are
//! `<|im_start|>role\n…<|im_end|>\n`, closed with a trailing
//! `<|im_start|>assistant\n` to invite the model's reply.
//!
//! ## Tool calling
//!
//! ChatML has no native tool concept; the convention here is the
//! Hermes / Qwen 2.5 one: tools are declared inside `<tools>…</tools>`
//! in the system turn and the model emits calls as
//! `<tool_call>{"name": …, "arguments": …}</tool_call>` blocks. Tool
//! *results* round-trip as a user turn wrapping
//! `<tool_response>…</tool_response>`.
//!
//! Decoding is a stateless map over [`scan_delimited`]'s segments
//! — the grammar `( text | <tool_call> json </tool_call> )*` is tiny
//! and the JSON body goes straight to `serde_json`. All the
//! streaming-framing difficulty lives in the generic scanner, not
//! here.
//!
//! Caveat: `tool_choice` / `parallel_tool_calls` enforcement needs
//! grammar-constrained decoding, which CPU-only local engines don't
//! support. Both are passed to the model as prompt hints only.

use std::borrow::Cow;

use async_stream::stream;
use futures_util::pin_mut;
use futures_util::stream::{Stream, StreamExt};
use serde::Serialize;
use serde_json::value::RawValue;

use crate::types::{AssistantPart, Function, InputItem, Prompt, ToolChoice, UserPart};
use crate::Error;

use super::chat_template::{
    scan_delimited, ChatTemplate, DeltaStream, ParsedDelta, Segment, TokenStream,
};

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// ChatML chat template with Hermes/Qwen-style `<tool_call>` blocks.
///
/// When `tools` is non-empty, a manifest is appended to the system
/// turn (or a synthetic system turn is created):
///
/// ```text
/// # Tools
/// You may call any of the following functions. To call one, emit a
/// single block of the form:
/// <tool_call>
/// {"name": "<function_name>", "arguments": <json-object>}
/// </tool_call>
///
/// Available functions:
/// <tools>
/// [{"type": "function", "function": {…}}, …]
/// </tools>
/// ```
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ChatMlTemplate;

impl ChatMlTemplate {
    /// Construct a default ChatML template.
    pub fn new() -> Self {
        Self
    }
}

impl ChatTemplate for ChatMlTemplate {
    fn render(
        &self,
        prompt: &Prompt,
        tools: &[&Function],
        tool_choice: Option<&ToolChoice>,
    ) -> String {
        let mut out = String::new();
        let tool_hint = render_tool_hint(tools, tool_choice);
        let mut tool_hint_emitted = tool_hint.is_none();

        for item in prompt.items() {
            match item {
                InputItem::System(content) => {
                    out.push_str("<|im_start|>system\n");
                    out.push_str(content);
                    if let Some(hint) = &tool_hint {
                        if !tool_hint_emitted {
                            out.push_str("\n\n");
                            out.push_str(hint);
                            tool_hint_emitted = true;
                        }
                    }
                    out.push_str("<|im_end|>\n");
                }
                InputItem::User { content } => {
                    // No system turn yet but tools are declared — emit
                    // a synthetic system turn before the first user
                    // turn so the model sees the tools manifest.
                    if let Some(hint) = &tool_hint {
                        if !tool_hint_emitted {
                            out.push_str("<|im_start|>system\n");
                            out.push_str(hint);
                            out.push_str("<|im_end|>\n");
                            tool_hint_emitted = true;
                        }
                    }
                    render_user_turn(&mut out, content);
                }
                InputItem::Assistant { content } => {
                    render_assistant_turn(&mut out, content);
                }
            }
        }

        // Open an assistant turn for the model to complete.
        out.push_str("<|im_start|>assistant\n");
        out
    }

    fn decode(&self, tokens: TokenStream) -> DeltaStream {
        Box::pin(chatml_decode(tokens))
    }
}

fn render_user_turn(out: &mut String, parts: &[UserPart]) {
    // Tool results are split out of the same user turn so the model
    // sees them inside `<tool_response>` blocks — matches the Qwen
    // 2.5 convention.
    for part in parts {
        if let UserPart::ToolResult { content, .. } = part {
            out.push_str("<|im_start|>user\n<tool_response>\n");
            out.push_str(&user_parts_to_text(content));
            out.push_str("\n</tool_response><|im_end|>\n");
        }
    }
    // Everything else gets a normal user turn — skip if there's no
    // non-tool-result content.
    let text = user_parts_text_only(parts);
    if !text.is_empty() {
        out.push_str("<|im_start|>user\n");
        out.push_str(&text);
        out.push_str("<|im_end|>\n");
    }
}

fn render_assistant_turn(out: &mut String, parts: &[AssistantPart]) {
    out.push_str("<|im_start|>assistant\n");
    for part in parts {
        match part {
            AssistantPart::Text { content, .. } => out.push_str(content),
            AssistantPart::ToolCall(call) => {
                out.push_str(TOOL_CALL_OPEN);
                out.push('\n');
                // Serialise the {"name","arguments"} block via serde
                // so `name` is correctly escaped and `arguments` is
                // spliced as a real JSON value (not string-formatted,
                // which broke on names containing `"`/`\` and on
                // non-JSON arguments).
                #[derive(Serialize)]
                struct ToolCallLine<'a> {
                    name: &'a str,
                    arguments: &'a RawValue,
                }
                let empty = RawValue::from_string("{}".to_string()).unwrap();
                let args: &RawValue = if call.arguments.trim().is_empty() {
                    &empty
                } else {
                    serde_json::from_str::<&RawValue>(&call.arguments).unwrap_or(&empty)
                };
                let line = serde_json::to_string(&ToolCallLine {
                    name: &call.name,
                    arguments: args,
                })
                .unwrap_or_else(|_| r#"{"name":"","arguments":{}}"#.to_string());
                out.push_str(&line);
                out.push('\n');
                out.push_str(TOOL_CALL_CLOSE);
            }
            // Reasoning / refusals / continuations / cache breakpoints
            // / builtin tool calls have no chat-template slot; drop
            // them silently. The model-switching contract.
            _ => tracing::debug!("ChatMlTemplate dropping non-renderable assistant part"),
        }
    }
    out.push_str("<|im_end|>\n");
}

fn user_parts_text_only(parts: &[UserPart]) -> String {
    let mut out = String::new();
    for part in parts {
        match part {
            UserPart::Text(s) => out.push_str(s),
            UserPart::ToolResult { .. } => {} // handled separately
            _ => tracing::debug!("ChatMlTemplate dropping non-text user part"),
        }
    }
    out
}

fn user_parts_to_text(parts: &[UserPart]) -> String {
    let mut out = String::new();
    for part in parts {
        match part {
            UserPart::Text(s) => out.push_str(s),
            UserPart::ToolResult { content, .. } => out.push_str(&user_parts_to_text(content)),
            _ => tracing::debug!("ChatMlTemplate dropping non-text user part"),
        }
    }
    out
}

/// Manifest snippet appended to the system turn when the caller
/// supplies function tools. Returns `None` for an empty toolset (or
/// `ToolChoice::None`) so callers can skip the manifest entirely.
fn render_tool_hint(tools: &[&Function], tool_choice: Option<&ToolChoice>) -> Option<String> {
    if tools.is_empty() {
        return None;
    }
    if matches!(tool_choice, Some(ToolChoice::None)) {
        return None;
    }

    #[derive(Serialize)]
    struct ToolDecl<'a> {
        #[serde(rename = "type")]
        kind: &'static str,
        function: FnDecl<'a>,
    }
    #[derive(Serialize)]
    struct FnDecl<'a> {
        name: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<&'a str>,
        parameters: &'a Cow<'static, RawValue>,
    }

    let decls: Vec<ToolDecl<'_>> = tools
        .iter()
        .map(|f| ToolDecl {
            kind: "function",
            function: FnDecl {
                name: &f.name,
                description: f.description.as_deref(),
                parameters: &f.parameters,
            },
        })
        .collect();
    let manifest = serde_json::to_string(&decls).unwrap_or_else(|_| "[]".to_string());

    let mut hint = String::from(
        "# Tools\n\
         You may call any of the following functions. To call one, emit a single block:\n\
         <tool_call>\n\
         {\"name\": \"<function_name>\", \"arguments\": <json-object>}\n\
         </tool_call>\n\n\
         Available functions:\n<tools>\n",
    );
    hint.push_str(&manifest);
    hint.push_str("\n</tools>");

    match tool_choice {
        Some(ToolChoice::Required) => {
            hint.push_str("\n\nYou MUST call one of the functions above before replying.");
        }
        Some(ToolChoice::Function { name }) => {
            hint.push_str(&format!(
                "\n\nYou MUST call the `{name}` function before replying."
            ));
        }
        _ => {}
    }
    Some(hint)
}

/// ChatML decoder: token chunks → [`ParsedDelta`]s.
///
/// A stateless map over [`scan_delimited`]'s segments: text runs
/// become bracketed text deltas; a block's payload is parsed as the
/// tool-call JSON. The only state is `text_open` — the lazy
/// [`ParsedDelta::TextStart`] / [`ParsedDelta::TextEnd`] bracketing
/// policy, which is a `ParsedDelta` concern, not a parsing one.
pub fn chatml_decode<S>(tokens: S) -> impl Stream<Item = Result<ParsedDelta, Error>> + Send
where
    S: Stream<Item = Result<String, Error>> + Send + 'static,
{
    stream! {
        let segments = scan_delimited(tokens, TOOL_CALL_OPEN, TOOL_CALL_CLOSE);
        pin_mut!(segments);
        let mut text_open = false;
        while let Some(seg) = segments.next().await {
            match seg {
                Err(e) => {
                    yield Err(e);
                    return;
                }
                Ok(Segment::Text(s)) => {
                    if s.is_empty() {
                        continue;
                    }
                    if !text_open {
                        yield Ok(ParsedDelta::TextStart);
                        text_open = true;
                    }
                    yield Ok(ParsedDelta::TextDelta(s));
                }
                Ok(Segment::Block(json)) => {
                    if text_open {
                        yield Ok(ParsedDelta::TextEnd);
                        text_open = false;
                    }
                    match parse_tool_call(&json) {
                        Some(call) => yield Ok(call),
                        None => tracing::warn!(
                            "chatml_decode: dropping <tool_call> with unparseable JSON"
                        ),
                    }
                }
            }
        }
        if text_open {
            yield Ok(ParsedDelta::TextEnd);
        }
    }
}

/// Pull `name` + `arguments` out of the JSON body of a `<tool_call>`
/// block. Returns `None` on malformed input — the caller logs and
/// drops.
fn parse_tool_call(json: &str) -> Option<ParsedDelta> {
    #[derive(serde::Deserialize)]
    struct Wire {
        name: String,
        // Capture as RawValue so the model's exact argument bytes
        // (key order, number formatting) pass through verbatim to
        // the tool — re-serialising via serde_json::Value would
        // reorder/normalise them.
        #[serde(default)]
        arguments: Option<Box<RawValue>>,
    }
    let body = json.trim();
    let parsed: Wire = match serde_json::from_str(body) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("chatml_decode: failed to parse tool_call JSON: {e}");
            return None;
        }
    };
    let arguments = match parsed.arguments {
        None => "{}".to_string(),
        Some(rv) if rv.get().trim() == "null" => "{}".to_string(),
        Some(rv) => rv.get().to_string(),
    };
    Some(ParsedDelta::ToolCall {
        name: parsed.name,
        arguments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FunctionCall;

    fn raw(s: &str) -> Cow<'static, RawValue> {
        Cow::Owned(RawValue::from_string(s.to_string()).unwrap())
    }

    // -- Rendering -----------------------------------------------

    #[test]
    fn render_basic_system_user() {
        let p = Prompt::system("be brief").with_user("hi");
        let out = ChatMlTemplate::new().render(&p, &[], None);
        let expected = "<|im_start|>system\nbe brief<|im_end|>\n\
                        <|im_start|>user\nhi<|im_end|>\n\
                        <|im_start|>assistant\n";
        assert_eq!(out, expected);
    }

    #[test]
    fn render_tool_manifest_appended_to_system_turn() {
        let f = Function {
            name: "get_weather".into(),
            description: Some("Get the weather".into()),
            parameters: raw(r#"{"type":"object","properties":{}}"#),
        };
        let tools = vec![&f];
        let p = Prompt::system("be brief").with_user("hi");
        let out = ChatMlTemplate::new().render(&p, &tools, None);
        assert!(out.contains("be brief\n\n# Tools\n"));
        assert!(out.contains("<tool_call>"));
        assert!(out.contains("get_weather"));
        assert!(out.contains("<tools>\n["));
    }

    #[test]
    fn render_synthesises_system_turn_when_only_user_supplied() {
        let f = Function {
            name: "ping".into(),
            description: None,
            parameters: raw(r#"{}"#),
        };
        let tools = vec![&f];
        let p = Prompt::user("hi");
        let out = ChatMlTemplate::new().render(&p, &tools, None);
        let sys = out.find("<|im_start|>system").unwrap();
        let user = out.find("<|im_start|>user").unwrap();
        assert!(sys < user, "expected system turn before user turn");
    }

    #[test]
    fn render_tool_choice_required_adds_hint() {
        let f = Function {
            name: "ping".into(),
            description: None,
            parameters: raw(r#"{}"#),
        };
        let out =
            ChatMlTemplate::new().render(&Prompt::user("hi"), &[&f], Some(&ToolChoice::Required));
        assert!(out.contains("MUST call one of the functions"));
    }

    #[test]
    fn render_tool_choice_none_drops_manifest() {
        let f = Function {
            name: "ping".into(),
            description: None,
            parameters: raw(r#"{}"#),
        };
        let out = ChatMlTemplate::new().render(&Prompt::user("hi"), &[&f], Some(&ToolChoice::None));
        assert!(!out.contains("<tools>"));
    }

    #[test]
    fn render_assistant_tool_call_serialised_as_block() {
        let call = FunctionCall {
            call_id: "call_0".into(),
            name: "get_weather".into(),
            arguments: r#"{"city":"Paris"}"#.into(),
            provider_signature: None,
        };
        let p = Prompt::user("hi")
            .with_item(InputItem::assistant_tool_call(call))
            .with_tool_result("call_0", "sunny");
        let out = ChatMlTemplate::new().render(&p, &[], None);
        assert!(out.contains(r#"<tool_call>"#));
        assert!(out.contains(r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#));
        assert!(out.contains("<tool_response>\nsunny\n</tool_response>"));
    }

    // -- Decode test helpers -------------------------------------
    //
    // chatml_decode is a `stream!` generator; tests drive it by
    // feeding a `stream::iter` of chunks and collecting. Most tests
    // don't care about the exact `TextStart` / `TextDelta` /
    // `TextEnd` boundaries — only the *logical* sequence of
    // [text-span | tool-call] groups. `normalize` collapses the raw
    // delta stream into [`Group`]s so the assertion holds regardless
    // of how the input was chunked.

    #[derive(Debug, PartialEq)]
    enum Group {
        Text(String),
        ToolCall { name: String, arguments: String },
    }

    fn normalize(events: Vec<ParsedDelta>) -> Vec<Group> {
        let mut out: Vec<Group> = Vec::new();
        let mut buf = String::new();
        for d in events {
            match d {
                ParsedDelta::TextStart | ParsedDelta::TextEnd => {}
                ParsedDelta::TextDelta(s) => buf.push_str(&s),
                ParsedDelta::ToolCall { name, arguments } => {
                    if !buf.is_empty() {
                        out.push(Group::Text(std::mem::take(&mut buf)));
                    }
                    out.push(Group::ToolCall { name, arguments });
                }
            }
        }
        if !buf.is_empty() {
            out.push(Group::Text(buf));
        }
        out
    }

    /// Drive `chatml_decode` over a fixed chunk sequence; raw items
    /// (errors included, not unwrapped).
    fn decode_chunks(chunks: &[&str]) -> Vec<Result<ParsedDelta, Error>> {
        let items: Vec<Result<String, Error>> = chunks.iter().map(|c| Ok(c.to_string())).collect();
        let s = chatml_decode(futures_util::stream::iter(items));
        futures::executor::block_on(s.collect::<Vec<_>>())
    }

    /// `decode_chunks` unwrapping each item (panics on a surfaced
    /// error — only for inputs that don't error).
    fn parse_in_chunks(chunks: &[&str]) -> Vec<ParsedDelta> {
        decode_chunks(chunks)
            .into_iter()
            .map(Result::unwrap)
            .collect()
    }

    /// Parse `input` fed as one chunk.
    fn parse_whole(input: &str) -> Vec<ParsedDelta> {
        parse_in_chunks(&[input])
    }

    /// Parse `input` one char at a time (worst-case chunking).
    fn parse_per_char(input: &str) -> Vec<ParsedDelta> {
        let chunks: Vec<String> = input.chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
        parse_in_chunks(&refs)
    }

    /// The fundamental invariant: the *normalised* output must not
    /// depend on how the input is chunked.
    fn assert_chunk_invariance(input: &str) {
        let whole = normalize(parse_whole(input));
        let per_char = normalize(parse_per_char(input));
        assert_eq!(
            whole, per_char,
            "chunk-invariance violated for input: {input:?}\n  whole = {whole:#?}\n  per-char = {per_char:#?}",
        );
    }

    // -- Basic shape ---------------------------------------------

    #[test]
    fn parser_passes_text_through() {
        assert_eq!(
            normalize(parse_whole("Hello, world!")),
            vec![Group::Text("Hello, world!".into())],
        );
    }

    #[test]
    fn parser_handles_empty_input() {
        assert!(normalize(parse_whole("")).is_empty());
    }

    #[test]
    fn parser_empty_chunks_are_no_ops() {
        assert_eq!(
            normalize(parse_in_chunks(&["", "hi", "", " there", ""])),
            vec![Group::Text("hi there".into())],
        );
    }

    // -- Tool call extraction ------------------------------------

    #[test]
    fn parser_extracts_lone_tool_call() {
        assert_eq!(
            normalize(parse_whole(
                r#"<tool_call>{"name":"x","arguments":{"a":1}}</tool_call>"#
            )),
            vec![Group::ToolCall {
                name: "x".into(),
                arguments: r#"{"a":1}"#.into(),
            }],
        );
    }

    #[test]
    fn parser_extracts_text_around_tool_call() {
        assert_eq!(
            normalize(parse_whole(
                r#"before <tool_call>{"name":"x","arguments":{}}</tool_call> after"#
            )),
            vec![
                Group::Text("before ".into()),
                Group::ToolCall {
                    name: "x".into(),
                    arguments: "{}".into(),
                },
                Group::Text(" after".into()),
            ],
        );
    }

    #[test]
    fn parser_handles_back_to_back_tool_calls() {
        assert_eq!(
            normalize(parse_whole(
                r#"<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{"x":1}}</tool_call>"#
            )),
            vec![
                Group::ToolCall {
                    name: "a".into(),
                    arguments: "{}".into(),
                },
                Group::ToolCall {
                    name: "b".into(),
                    arguments: r#"{"x":1}"#.into(),
                },
            ],
        );
    }

    #[test]
    fn parser_handles_interleaved_text_and_tool_calls() {
        let input = concat!(
            "thinking... ",
            r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#,
            " ok so ",
            r#"<tool_call>{"name":"b","arguments":{}}</tool_call>"#,
            " done."
        );
        let groups = normalize(parse_whole(input));
        assert_eq!(groups.len(), 5, "got {groups:#?}");
        assert!(matches!(&groups[0], Group::Text(s) if s == "thinking... "));
        assert!(matches!(&groups[1], Group::ToolCall { name, .. } if name == "a"));
        assert!(matches!(&groups[2], Group::Text(s) if s == " ok so "));
        assert!(matches!(&groups[3], Group::ToolCall { name, .. } if name == "b"));
        assert!(matches!(&groups[4], Group::Text(s) if s == " done."));
    }

    #[test]
    fn parser_handles_whitespace_inside_tool_call_block() {
        let input = "<tool_call>\n  {\"name\":\"x\",\"arguments\":{}}\n</tool_call>";
        assert_eq!(
            normalize(parse_whole(input)),
            vec![Group::ToolCall {
                name: "x".into(),
                arguments: "{}".into(),
            }],
        );
    }

    #[test]
    fn parser_normalises_null_arguments_to_empty_object() {
        assert_eq!(
            normalize(parse_whole(
                r#"<tool_call>{"name":"x","arguments":null}</tool_call>"#
            )),
            vec![Group::ToolCall {
                name: "x".into(),
                arguments: "{}".into(),
            }],
        );
    }

    #[test]
    fn parser_defaults_arguments_to_empty_object_when_omitted() {
        assert_eq!(
            normalize(parse_whole(r#"<tool_call>{"name":"x"}</tool_call>"#)),
            vec![Group::ToolCall {
                name: "x".into(),
                arguments: "{}".into(),
            }],
        );
    }

    #[test]
    fn parser_drops_malformed_tool_call_json_keeping_surrounding_text() {
        // `{"name":"x"` is unparseable; the call is dropped, the
        // surrounding text both survives. Since there's no ToolCall
        // between them after the drop, `normalize` (correctly) fuses
        // the two spans into a single Text group.
        let input = r#"hi <tool_call>{"name":"x"</tool_call> bye"#;
        assert_eq!(
            normalize(parse_whole(input)),
            vec![Group::Text("hi  bye".into())],
        );
    }

    #[test]
    fn parser_finish_inside_tool_call_does_best_effort_parse() {
        assert_eq!(
            normalize(parse_whole(r#"<tool_call>{"name":"x","arguments":{}}"#)),
            vec![Group::ToolCall {
                name: "x".into(),
                arguments: "{}".into(),
            }],
        );
    }

    #[test]
    fn parser_finish_inside_unparseable_tool_call_drops_silently() {
        assert!(normalize(parse_whole("<tool_call>{garbage")).is_empty());
    }

    // -- UTF-8 + content safety ----------------------------------

    #[test]
    fn parser_passes_multi_byte_utf8_through_text() {
        assert_eq!(
            normalize(parse_whole("héllo 🌍 世界")),
            vec![Group::Text("héllo 🌍 世界".into())],
        );
    }

    #[test]
    fn parser_passes_tool_call_with_unicode_arguments() {
        let input = r#"<tool_call>{"name":"say","arguments":{"msg":"héllo 🌍"}}</tool_call>"#;
        assert_eq!(
            normalize(parse_whole(input)),
            vec![Group::ToolCall {
                name: "say".into(),
                arguments: r#"{"msg":"héllo 🌍"}"#.into(),
            }],
        );
    }

    // -- Chunk invariance ----------------------------------------

    #[test]
    fn parser_chunk_invariance_pure_text() {
        assert_chunk_invariance("Hello, world!");
    }

    #[test]
    fn parser_chunk_invariance_text_with_angle_bracket() {
        assert_chunk_invariance("a<b<c<<>>tool_callz");
    }

    #[test]
    fn parser_chunk_invariance_text_with_marker_prefix_runs() {
        assert_chunk_invariance("see <tool_kit and <tool_box and done.");
    }

    #[test]
    fn parser_chunk_invariance_with_tool_call() {
        assert_chunk_invariance(
            r#"hi there <tool_call>{"name":"x","arguments":{"a":1}}</tool_call> bye!"#,
        );
    }

    #[test]
    fn parser_chunk_invariance_multiple_tool_calls() {
        assert_chunk_invariance(concat!(
            "first ",
            r#"<tool_call>{"name":"a","arguments":{}}</tool_call>"#,
            " middle ",
            r#"<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>"#,
            " last"
        ));
    }

    #[test]
    fn parser_chunk_invariance_tool_call_at_boundaries() {
        assert_chunk_invariance(r#"<tool_call>{"name":"x","arguments":{}}</tool_call>"#);
    }

    #[test]
    fn parser_chunk_invariance_with_unicode() {
        assert_chunk_invariance(
            r#"héllo 🌍 <tool_call>{"name":"x","arguments":{"city":"São Paulo"}}</tool_call> 世界"#,
        );
    }

    #[test]
    fn parser_chunk_invariance_with_unterminated_tool_call() {
        assert_chunk_invariance(r#"prefix <tool_call>{"name":"x","arguments":{}}"#);
    }

    // -- Known limitations ---------------------------------------

    #[test]
    fn parser_documents_nested_tool_call_in_arguments_limitation() {
        // The scanner closes on the *first* `</tool_call>`, so a JSON
        // string value containing `</tool_call>` prematurely closes
        // the block. Real ChatML-trained models don't emit this; if
        // one did we'd need a JSON-aware scanner. Just assert no
        // crash and document the behaviour.
        let input = r#"<tool_call>{"name":"x","arguments":{"s":"a</tool_call>b"}}</tool_call>"#;
        let groups = normalize(parse_whole(input));
        assert!(!groups.is_empty(), "parser should still produce something");
    }

    // -- Exact-output regression pins ----------------------------

    #[test]
    fn parser_emits_text_start_end_around_text() {
        assert_eq!(
            parse_whole("hi"),
            vec![
                ParsedDelta::TextStart,
                ParsedDelta::TextDelta("hi".into()),
                ParsedDelta::TextEnd,
            ],
        );
    }

    #[test]
    fn parser_does_not_open_text_part_for_pure_tool_call() {
        assert_eq!(
            parse_whole(r#"<tool_call>{"name":"x","arguments":{}}</tool_call>"#),
            vec![ParsedDelta::ToolCall {
                name: "x".into(),
                arguments: "{}".into(),
            }],
        );
    }

    #[test]
    fn decode_forwards_errors_then_stops() {
        // Clean text alone produces no error.
        assert!(decode_chunks(&["partial"]).iter().all(|r| r.is_ok()));

        // With an injected error mid-stream, the post-error chunk is
        // never processed.
        let items = vec![
            Ok::<_, Error>("partial".into()),
            Err(Error::streaming("kaboom")),
            Ok(" later".into()),
        ];
        let out = futures::executor::block_on(
            chatml_decode(futures_util::stream::iter(items)).collect::<Vec<_>>(),
        );
        assert!(
            out.iter().any(|r| r.is_err()),
            "expected forwarded error, got {out:#?}",
        );
        for item in &out {
            if let Ok(ParsedDelta::TextDelta(s)) = item {
                assert!(!s.contains("later"), "post-error chunk leaked: {s:?}");
            }
        }
    }
}
