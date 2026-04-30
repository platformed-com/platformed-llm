# Captured provider traces

Real raw `(request, response)` pairs from each provider, captured by
`cargo run --example capture_traces`. These are the **ground truth** the
crate is tested against — every `<scenario>.response.sse` here is bytes
that came over the wire, not something synthesized from spec.

Layout (one subdirectory per provider, three files per scenario):

```
tests/cross_provider/traces/
  openai/
    text_only.request.json     # JSON body we sent
    text_only.response.sse     # raw response stream, byte-for-byte
    text_only.meta.json        # status, headers, model, latency
    function_call.*
    ...
  google/
    ...
  anthropic/
    ...
```

## Regenerating

```sh
# Set OPENAI_API_KEY, GOOGLE_PROJECT_ID, GOOGLE_REGION, optionally
# GOOGLE_SERVICE_ACCOUNT_EMAIL in .env. Then:
cargo run --example capture_traces

# Subset:
cargo run --example capture_traces -- openai
cargo run --example capture_traces -- text_only
cargo run --example capture_traces -- openai function_call
```

Captures are committed to git so the test suite is reproducible without
network access.

## Replaying

```sh
cargo test --test replay_traces
```

This walks the directory, replays each `.response.sse` through the
matching provider's pipeline (parser → state machine → accumulator),
and asserts a sane unified-event sequence (one `Done`, non-empty
content for text scenarios, etc.).

## Adding scenarios

Edit `tests/scenarios.json`. Re-run capture. No Rust changes required.

## Caveats

- Captures contain real model output (sometimes verbose). Diff carefully
  on regeneration — model versions drift over time, even at the same
  identifier.
- The `.meta.json` files contain volatile fields (`captured_at_unix`,
  `latency_ms`, request IDs). Tests should not depend on those.
- Don't commit captures from accounts with PII or sensitive prompts.
- **Anthropic captures are not yet checked in.** Capturing requires a
  GCP project with at least one Claude model enabled in Vertex Model
  Garden. If your `GOOGLE_PROJECT_ID` doesn't have Claude enabled,
  `cargo run --example capture_traces -- anthropic` will return 404
  ("model not found OR your project does not have access"). Set
  `ANTHROPIC_REGION` (commonly `us-east5`) and enable a Claude model on
  the project, then re-run.
- Scenarios use `max_tokens: 256` to leave headroom for thinking-model
  overhead. Gemini 2.5 burns reasoning tokens before producing output;
  too tight a budget can cause `FinishReason::Length` with zero output
  text.
