//! The local-provider chat-template abstraction and shared pipeline.
//!
//! Hosted providers (OpenAI / Vertex) speak structured wire formats —
//! the request carries `messages`, `tools`, etc., and the provider
//! serialises it. A local provider like
//! [`LlamaGgufProvider`](super::LlamaGgufProvider) has no such
//! affordance: the model only sees a flat token stream. So a local
//! provider needs two model-specific halves, both owned by a
//! [`ChatTemplate`] implementation:
//!
//! 1. **render** — flatten a [`Prompt`] (+ tools) into the single
//!    string the model was trained to expect.
//! 2. **decode** — turn the model's raw token stream back into typed
//!    [`ParsedDelta`]s.
//!
//! This module is the *general* machinery: the [`ChatTemplate`] trait,
//! the [`ParsedDelta`] vocabulary, the reusable streaming-framing
//! primitive [`scan_delimited`], and the provider-agnostic
//! [`translate_to_events`] stage that turns deltas into the unified
//! [`StreamEvent`] flow. The concrete ChatML/Hermes implementation
//! lives in the sibling [`chatml`](super::chatml) module; other
//! templates (Llama 3 chat, Vicuna, raw completion) would be
//! additional implementations here.
//!
//! ## The pipeline
//!
//! ```text
//!   engine tokens               (Stream<Result<String>>)
//!     │  ChatTemplate::decode  (e.g. scan_delimited + a stateless map)
//!   parsed deltas               (Stream<Result<ParsedDelta>>)
//!     │  .into_stream_events()  (→ translate_to_events)
//!   unified events + final Done (Stream<Result<StreamEvent>>)
//! ```
//!
//! The only genuinely tricky part is *streaming framing* — splitting a
//! chunked stream on a literal delimiter pair when delimiters can
//! straddle chunk boundaries. That lives, once, in the generic
//! [`scan_delimited`]; a template-specific decoder is then a stateless
//! `match` over its [`Segment`]s.

use std::fmt;
use std::pin::Pin;

use async_stream::stream;
use futures::channel::oneshot;
use futures_util::pin_mut;
use futures_util::stream::{Stream, StreamExt};

use crate::types::{
    FinishReason, Function, PartKind, Prompt, StreamEvent, Tool, ToolChoice, Usage,
};
use crate::Error;

/// A boxed, `Send` stream of raw model-output token chunks — the
/// input side of a [`ChatTemplate::decode`] pipeline.
pub type TokenStream = Pin<Box<dyn Stream<Item = Result<String, Error>> + Send>>;

/// A boxed, `Send` stream of decoded [`ParsedDelta`]s — the output of
/// [`ChatTemplate::decode`], fed into [`into_stream_events`].
///
/// [`into_stream_events`]: ParsedDeltaStreamExt::into_stream_events
pub type DeltaStream = Pin<Box<dyn Stream<Item = Result<ParsedDelta, Error>> + Send>>;

/// Renders a [`Prompt`] (plus available tools) into a model-specific
/// prompt string, and decodes the model's raw token stream back into
/// typed [`ParsedDelta`]s.
///
/// Implementations should be cheap to clone-via-`Box`/`Arc` and
/// stateless — all per-call state lives in the stream `decode`
/// returns, not on the template. The default implementation is
/// [`ChatMlTemplate`](super::chatml::ChatMlTemplate).
pub trait ChatTemplate: Send + Sync + fmt::Debug {
    /// Render the conversation history (with optional tool declarations
    /// and an optional tool-choice hint) into a single prompt string
    /// suitable for feeding to the local engine.
    ///
    /// `tools` is filtered to function tools only — provider builtins
    /// (web search, code execution, …) have no representation in a
    /// pure local-text world and are dropped by the caller (see
    /// [`function_tools`]).
    fn render(
        &self,
        prompt: &Prompt,
        tools: &[&Function],
        tool_choice: Option<&ToolChoice>,
    ) -> String;

    /// Decode a stream of raw model-output token chunks into a stream
    /// of typed [`ParsedDelta`]s.
    ///
    /// The returned stream is responsible for buffering partial
    /// multi-character markers that straddle chunk boundaries,
    /// forwarding an upstream `Err` once and then ending, and
    /// flushing any trailing content at end-of-input. [`scan_delimited`]
    /// handles all of this for delimiter-based templates.
    fn decode(&self, tokens: TokenStream) -> DeltaStream;
}

/// Output of [`ChatTemplate::decode`] — the building blocks
/// [`translate_to_events`] turns into [`StreamEvent`]s.
///
/// Tool calls arrive as a single [`Self::ToolCall`] with the full
/// `arguments` JSON — no intra-call streaming. Local models emit tool
/// calls as one atomic block, so per-token argument deltas would be
/// useless.
#[derive(Debug, Clone, PartialEq)]
pub enum ParsedDelta {
    /// Start of a text part.
    TextStart,
    /// Append to the current text part.
    TextDelta(String),
    /// End of the current text part.
    TextEnd,
    /// A fully-parsed tool call. The provider assigns a `call_id`.
    ToolCall {
        /// Function name the model wants to invoke.
        name: String,
        /// JSON-encoded argument object.
        arguments: String,
    },
}

/// Keep just the function tools out of a generic `[Tool]` slice.
/// Provider builtins have no place in a local-text chat template and
/// are dropped here; the result is what's passed to
/// [`ChatTemplate::render`].
pub fn function_tools(tools: &[Tool]) -> Vec<&Function> {
    tools.iter().filter_map(Tool::as_function).collect()
}

// ---------------------------------------------------------------------------
// Streaming-framing primitive
// ---------------------------------------------------------------------------

/// One piece of a [`scan_delimited`] stream.
#[derive(Debug, Clone, PartialEq)]
pub enum Segment {
    /// A run of text outside any `open`…`close` block, surfaced
    /// incrementally as it arrives (a held-back tail that could be
    /// the start of `open` is withheld until disambiguated).
    Text(String),
    /// The complete inner payload of one `open`…`close` block. An
    /// unterminated block at end-of-stream still yields its buffered
    /// payload (best-effort).
    Block(String),
}

/// Largest byte count from the front of `buf` that can't be the start
/// of `marker`. Holds back the last `marker.len() - 1` bytes, clamped
/// down to a char boundary so a multi-byte UTF-8 sequence is never
/// split. `0` means "withhold everything for now".
///
/// Precondition: `marker` is ASCII (true for all delimiters this
/// crate uses). A multibyte `marker` could under-hold — a partial
/// multibyte delimiter straddling a chunk boundary might leak as
/// text and then never re-match. `scan_delimited` only takes
/// `&'static str` delimiters chosen by template authors, so this is
/// a documented contract, not a runtime check.
fn safe_prefix_end(buf: &str, marker: &str) -> usize {
    let hold = marker.len().saturating_sub(1);
    let mut end = buf.len().saturating_sub(hold);
    while end > 0 && !buf.is_char_boundary(end) {
        end -= 1;
    }
    end
}

/// Split a chunked text stream into [`Segment`]s on a literal
/// `open`/`close` delimiter pair, robust to a delimiter that straddles
/// a chunk boundary.
///
/// This is the *only* place with a buffer or boundary reasoning. The
/// two loops are the two states (outside / inside a block) — the
/// program counter is the state, no enum. Outside text streams out
/// incrementally minus a held-back possible-`open` tail; a block's
/// payload is buffered whole (it isn't shown incrementally anyway).
/// An upstream `Err` is forwarded once and ends the stream; clean EOF
/// flushes trailing text or an unterminated block's payload.
pub fn scan_delimited<S>(
    tokens: S,
    open: &'static str,
    close: &'static str,
) -> impl Stream<Item = Result<Segment, Error>> + Send
where
    S: Stream<Item = Result<String, Error>> + Send + 'static,
{
    stream! {
        pin_mut!(tokens);
        let mut buf = String::new();
        'outside: loop {
            // Outside a block: stream text until `open` appears.
            loop {
                if let Some(i) = buf.find(open) {
                    let head: String = buf.drain(..i).collect();
                    if !head.is_empty() {
                        yield Ok(Segment::Text(head));
                    }
                    buf.drain(..open.len());
                    break; // → inside
                }
                let end = safe_prefix_end(&buf, open);
                if end > 0 {
                    yield Ok(Segment::Text(buf.drain(..end).collect()));
                }
                match tokens.next().await {
                    Some(Ok(chunk)) => buf.push_str(&chunk),
                    Some(Err(e)) => {
                        yield Err(e);
                        return;
                    }
                    None => {
                        if !buf.is_empty() {
                            yield Ok(Segment::Text(std::mem::take(&mut buf)));
                        }
                        return;
                    }
                }
            }
            // Inside a block: buffer until `close` (never emit partial
            // payload, so no held-back tail needed here).
            loop {
                if let Some(i) = buf.find(close) {
                    let body: String = buf.drain(..i).collect();
                    buf.drain(..close.len());
                    yield Ok(Segment::Block(body));
                    continue 'outside;
                }
                match tokens.next().await {
                    Some(Ok(chunk)) => buf.push_str(&chunk),
                    Some(Err(e)) => {
                        yield Err(e);
                        return;
                    }
                    None => {
                        // Unterminated block — surface the buffered
                        // payload best-effort rather than dropping it.
                        yield Ok(Segment::Block(std::mem::take(&mut buf)));
                        return;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ParsedDelta → StreamEvent translation
// ---------------------------------------------------------------------------

/// Translate a [`ParsedDelta`] stream into the unified
/// [`StreamEvent`] flow: assign monotonic part indices, mint
/// synthetic `call_id`s, and append a terminal [`StreamEvent::Done`]
/// once the input drains.
///
/// `finish_signal` controls the terminal [`StreamEvent::Done`]'s
/// `finish_reason`:
///
/// - `Some(rx)` — the producer (e.g. a local engine) reports the
///   authoritative reason out-of-band; whatever it resolves to wins
///   (a truncated turn must report [`FinishReason::Length`] even if a
///   tool call also parsed). If the sender is dropped without a value
///   the derived fallback is used.
/// - `None` — derive: [`FinishReason::ToolCalls`] if any tool call
///   passed through, else [`FinishReason::Stop`]. Used by templates /
///   tests with no engine signal.
///
/// Usage stays zero — local engines don't surface token counts. An
/// upstream `Err` is forwarded once and suppresses the trailing
/// `Done` (downstream handles error termination).
///
/// Prefer the [`ParsedDeltaStreamExt::into_stream_events`] sugar when
/// there's no engine signal.
pub fn translate_to_events<S>(
    deltas: S,
    finish_signal: Option<oneshot::Receiver<FinishReason>>,
) -> impl Stream<Item = Result<StreamEvent, Error>> + Send
where
    S: Stream<Item = Result<ParsedDelta, Error>> + Send + 'static,
{
    stream! {
        pin_mut!(deltas);
        let mut next_index: u32 = 0;
        let mut text_index: Option<u32> = None;
        let mut tool_calls: u32 = 0;
        let mut emitted_tool_call = false;

        while let Some(item) = deltas.next().await {
            let d = match item {
                Ok(d) => d,
                Err(e) => {
                    yield Err(e);
                    return; // suppress the trailing Done
                }
            };
            match d {
                ParsedDelta::TextStart => {
                    let idx = next_index;
                    next_index += 1;
                    text_index = Some(idx);
                    yield Ok(StreamEvent::PartStart {
                        index: idx,
                        kind: PartKind::Text,
                    });
                }
                ParsedDelta::TextDelta(delta) => {
                    if let Some(idx) = text_index {
                        if !delta.is_empty() {
                            yield Ok(StreamEvent::Delta { index: idx, delta });
                        }
                    }
                }
                ParsedDelta::TextEnd => {
                    if let Some(idx) = text_index.take() {
                        yield Ok(StreamEvent::PartEnd { index: idx });
                    }
                }
                ParsedDelta::ToolCall { name, arguments } => {
                    emitted_tool_call = true;
                    let idx = next_index;
                    next_index += 1;
                    let call_id = format!("call_{tool_calls}");
                    tool_calls += 1;
                    yield Ok(StreamEvent::PartStart {
                        index: idx,
                        kind: PartKind::ToolCall { call_id, name },
                    });
                    if !arguments.is_empty() {
                        yield Ok(StreamEvent::Delta {
                            index: idx,
                            delta: arguments,
                        });
                    }
                    yield Ok(StreamEvent::PartEnd { index: idx });
                }
            }
        }

        let derived = if emitted_tool_call {
            FinishReason::ToolCalls
        } else {
            FinishReason::Stop
        };
        let finish_reason = match finish_signal {
            // Engine-reported reason is authoritative (e.g. Length on
            // a max_tokens-truncated turn). Sender dropped without a
            // value → fall back to the derived reason.
            Some(rx) => rx.await.unwrap_or(derived),
            None => derived,
        };
        yield Ok(StreamEvent::Done {
            finish_reason,
            usage: Usage::default(),
        });
    }
}

/// Pipeline sugar: `delta_stream.into_stream_events()` instead of
/// `translate_to_events(delta_stream)`, so a provider reads as
/// `template.decode(tokens).into_stream_events()`.
pub trait ParsedDeltaStreamExt:
    Stream<Item = Result<ParsedDelta, Error>> + Send + Sized + 'static
{
    /// See [`translate_to_events`]. No engine finish-signal — the
    /// reason is derived (Stop / ToolCalls).
    fn into_stream_events(self) -> impl Stream<Item = Result<StreamEvent, Error>> + Send {
        translate_to_events(self, None)
    }
}

impl<S> ParsedDeltaStreamExt for S where
    S: Stream<Item = Result<ParsedDelta, Error>> + Send + 'static
{
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- scan_delimited primitive --------------------------------
    //
    // The streaming-framing logic, exercised in isolation: arbitrary
    // delimiters, no JSON, no template. This is where the only
    // buffering/holdback code is proven correct.

    fn scan(chunks: &[&str], open: &'static str, close: &'static str) -> Vec<Segment> {
        let items: Vec<Result<String, Error>> = chunks.iter().map(|c| Ok(c.to_string())).collect();
        let s = scan_delimited(futures_util::stream::iter(items), open, close);
        futures::executor::block_on(s.collect::<Vec<_>>())
            .into_iter()
            .map(Result::unwrap)
            .collect()
    }

    /// Merge adjacent `Text` so output is comparable regardless of
    /// how text was chunked.
    fn merge_text(v: Vec<Segment>) -> Vec<Segment> {
        let mut out: Vec<Segment> = Vec::new();
        for s in v {
            match (out.last_mut(), s) {
                (Some(Segment::Text(prev)), Segment::Text(cur)) => prev.push_str(&cur),
                (_, s) => out.push(s),
            }
        }
        out
    }

    #[test]
    fn scan_text_only_has_no_blocks() {
        assert_eq!(
            merge_text(scan(&["hello ", "world"], "[[", "]]")),
            vec![Segment::Text("hello world".into())],
        );
    }

    #[test]
    fn scan_splits_text_and_block() {
        assert_eq!(
            merge_text(scan(&["a[[X]]b"], "[[", "]]")),
            vec![
                Segment::Text("a".into()),
                Segment::Block("X".into()),
                Segment::Text("b".into()),
            ],
        );
    }

    #[test]
    fn scan_adjacent_blocks_emit_no_empty_text() {
        assert_eq!(
            scan(&["[[A]][[B]]"], "[[", "]]"),
            vec![Segment::Block("A".into()), Segment::Block("B".into())],
        );
    }

    #[test]
    fn scan_handles_open_straddling_chunk_boundary() {
        // `<<` open delimiter split across two chunks. Concatenated
        // input is `x<<y>>z`: text "x", block "y", text "z".
        assert_eq!(
            merge_text(scan(&["x<", "<y>", ">z"], "<<", ">>")),
            vec![
                Segment::Text("x".into()),
                Segment::Block("y".into()),
                Segment::Text("z".into()),
            ],
        );
    }

    #[test]
    fn scan_handles_close_straddling_chunk_boundary() {
        assert_eq!(
            merge_text(scan(&["[[pay", "lo", "ad]", "]tail"], "[[", "]]")),
            vec![
                Segment::Block("payload".into()),
                Segment::Text("tail".into()),
            ],
        );
    }

    #[test]
    fn scan_unterminated_block_flushes_payload() {
        assert_eq!(
            merge_text(scan(&["pre[[half"], "[[", "]]")),
            vec![Segment::Text("pre".into()), Segment::Block("half".into())],
        );
    }

    #[test]
    fn scan_is_chunk_invariant() {
        let input = "alpha [[one]] beta [[two]] gamma";
        let whole = merge_text(scan(&[input], "[[", "]]"));

        let chars: Vec<String> = input.chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = chars.iter().map(|s| s.as_str()).collect();
        let by_char = merge_text(scan(&refs, "[[", "]]"));

        assert_eq!(whole, by_char);
        assert_eq!(
            whole,
            vec![
                Segment::Text("alpha ".into()),
                Segment::Block("one".into()),
                Segment::Text(" beta ".into()),
                Segment::Block("two".into()),
                Segment::Text(" gamma".into()),
            ],
        );
    }

    // -- translate_to_events -------------------------------------
    //
    // Driven with synthetic ParsedDelta input so the translator is
    // tested independently of any decoder.

    fn events_from(deltas: Vec<Result<ParsedDelta, Error>>) -> Vec<Result<StreamEvent, Error>> {
        let s = futures_util::stream::iter(deltas);
        futures::executor::block_on(translate_to_events(s, None).collect::<Vec<_>>())
    }

    #[test]
    fn translate_terminates_with_done_stop() {
        let events: Vec<StreamEvent> = events_from(vec![
            Ok(ParsedDelta::TextStart),
            Ok(ParsedDelta::TextDelta("hello".into())),
            Ok(ParsedDelta::TextEnd),
        ])
        .into_iter()
        .map(Result::unwrap)
        .collect();
        match events.last().unwrap() {
            StreamEvent::Done { finish_reason, .. } => {
                assert_eq!(*finish_reason, FinishReason::Stop);
            }
            _ => panic!("expected Done last, got {events:#?}"),
        }
    }

    #[test]
    fn translate_picks_tool_calls_finish_when_call_emitted() {
        let events: Vec<StreamEvent> = events_from(vec![Ok(ParsedDelta::ToolCall {
            name: "x".into(),
            arguments: "{}".into(),
        })])
        .into_iter()
        .map(Result::unwrap)
        .collect();
        match events.last().unwrap() {
            StreamEvent::Done { finish_reason, .. } => {
                assert_eq!(*finish_reason, FinishReason::ToolCalls);
            }
            _ => panic!("expected Done last"),
        }
    }

    #[test]
    fn translate_assigns_monotonic_indices_and_call_ids() {
        // text → part 0, tool call → part 1, tool call → part 2.
        let events: Vec<StreamEvent> = events_from(vec![
            Ok(ParsedDelta::TextStart),
            Ok(ParsedDelta::TextDelta("hi ".into())),
            Ok(ParsedDelta::TextEnd),
            Ok(ParsedDelta::ToolCall {
                name: "a".into(),
                arguments: "{}".into(),
            }),
            Ok(ParsedDelta::ToolCall {
                name: "b".into(),
                arguments: "{}".into(),
            }),
        ])
        .into_iter()
        .map(Result::unwrap)
        .collect();
        let part_starts: Vec<u32> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::PartStart { index, .. } => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(part_starts, vec![0, 1, 2]);
        let call_ids: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::PartStart {
                    kind: PartKind::ToolCall { call_id, .. },
                    ..
                } => Some(call_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(call_ids, vec!["call_0", "call_1"]);
    }

    #[test]
    fn translate_suppresses_done_after_upstream_error() {
        let events = events_from(vec![Err(Error::provider("Stream", "boom"))]);
        assert!(
            events.iter().any(|r| r.is_err()),
            "expected an error to surface"
        );
        for r in events.iter().filter_map(|r| r.as_ref().ok()) {
            assert!(
                !matches!(r, StreamEvent::Done { .. }),
                "Done must not follow an error"
            );
        }
    }

    #[test]
    fn translate_engine_finish_signal_overrides_derived_reason() {
        // A tool call was emitted (derived would be ToolCalls) but the
        // engine reports Length (max_tokens) — the engine signal wins.
        let (tx, rx) = oneshot::channel();
        tx.send(FinishReason::Length).unwrap();
        let s = futures_util::stream::iter(vec![Ok(ParsedDelta::ToolCall {
            name: "x".into(),
            arguments: "{}".into(),
        })]);
        let events: Vec<StreamEvent> =
            futures::executor::block_on(translate_to_events(s, Some(rx)).collect::<Vec<_>>())
                .into_iter()
                .map(Result::unwrap)
                .collect();
        match events.last().unwrap() {
            StreamEvent::Done { finish_reason, .. } => {
                assert_eq!(*finish_reason, FinishReason::Length);
            }
            _ => panic!("expected Done last"),
        }
    }

    #[test]
    fn translate_dropped_finish_signal_falls_back_to_derived() {
        // Sender dropped without sending → derived reason (Stop here).
        let (tx, rx) = oneshot::channel::<FinishReason>();
        drop(tx);
        let s = futures_util::stream::iter(vec![
            Ok(ParsedDelta::TextStart),
            Ok(ParsedDelta::TextDelta("hi".into())),
            Ok(ParsedDelta::TextEnd),
        ]);
        let events: Vec<StreamEvent> =
            futures::executor::block_on(translate_to_events(s, Some(rx)).collect::<Vec<_>>())
                .into_iter()
                .map(Result::unwrap)
                .collect();
        match events.last().unwrap() {
            StreamEvent::Done { finish_reason, .. } => {
                assert_eq!(*finish_reason, FinishReason::Stop);
            }
            _ => panic!("expected Done last"),
        }
    }
}
