//! Capture raw SSE traces from real LLM providers for use as test fixtures.
//!
//! ## Usage
//!
//! ```text
//! cargo run --example capture_traces                     # all providers, all scenarios
//! cargo run --example capture_traces -- openai           # one provider
//! cargo run --example capture_traces -- text_only        # one scenario
//! cargo run --example capture_traces -- openai text_only # combined
//! ```
//!
//! ## Required env vars (read from .env or process env)
//!
//! - `OPENAI_API_KEY` — bearer token for `api.openai.com`.
//! - `GOOGLE_PROJECT_ID` and `GOOGLE_REGION` — Vertex AI project + region.
//! - `ANTHROPIC_REGION` — *optional*. Override `GOOGLE_REGION` for
//!   Anthropic-on-Vertex calls. Anthropic models are only enabled in some
//!   Vertex regions (commonly `us-east5`); set this if your default region
//!   doesn't have Anthropic enabled.
//! - `GOOGLE_SERVICE_ACCOUNT_EMAIL` — *optional*. If set, we shell out to
//!   `gcloud auth print-access-token --impersonate-service-account=$EMAIL`
//!   to mint a Vertex token. Otherwise we use whatever ADC resolves to via
//!   `gcp_auth::provider()` (i.e. `GOOGLE_APPLICATION_CREDENTIALS`,
//!   `gcloud auth application-default login`, or GCE metadata).
//!
//! ## Outputs
//!
//! For each `(provider, scenario)` the script writes three files under
//! `tests/cross_provider/traces/<provider>/`:
//!
//! - `<scenario>.request.json` — the JSON body we sent (pretty-printed).
//! - `<scenario>.response.sse` — the raw HTTP response body, byte-for-byte.
//! - `<scenario>.meta.json` — status, selected headers, model, endpoint,
//!   capture timestamp, latency. Useful for debugging; tests should NOT
//!   read volatile fields from this.
//!
//! Scenarios live in `tests/scenarios.json` (data, not code). See the
//! comment at the top of that file for the full schema, including
//! per-provider overrides and `expect_failure` for error-path captures.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

#[derive(Debug, Deserialize)]
struct Config {
    defaults: Defaults,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Deserialize)]
struct Defaults {
    models: ModelMap,
}

#[derive(Debug, Deserialize)]
struct ModelMap {
    openai: String,
    google: String,
    anthropic: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Scenario {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    expect_failure: bool,
    #[serde(default)]
    providers: BTreeMap<String, ProviderOverride>,
    messages: Vec<ScenarioMessage>,
    #[serde(default)]
    tools: Vec<ScenarioTool>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct ProviderOverride {
    #[serde(default)]
    skip: bool,
    #[serde(default)]
    model: Option<String>,
    /// Literal token to send instead of the resolved bearer. Useful for 401
    /// captures. Never written to the saved request body.
    #[serde(default)]
    auth_override: Option<String>,
    /// Free-form provider-specific JSON merged into the request body just
    /// before send (e.g. `parallel_tool_calls`, `reasoning`,
    /// `thinkingConfig`).
    #[serde(default)]
    extra_body: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ScenarioMessage {
    role: String,
    #[serde(default)]
    content: Option<String>,
    /// Assistant role: prior tool calls the model emitted in this turn.
    #[serde(default)]
    tool_calls: Vec<ScenarioToolCall>,
    /// `tool` role: which assistant tool call this message answers.
    #[serde(default)]
    tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ScenarioToolCall {
    id: String,
    name: String,
    /// JSON-encoded string. Matches OpenAI's wire format; we re-parse for
    /// Gemini/Anthropic.
    arguments: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ScenarioTool {
    name: String,
    #[serde(default)]
    description: String,
    parameters: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Provider {
    OpenAI,
    Google,
    Anthropic,
}

impl Provider {
    const ALL: &'static [Provider] = &[Provider::OpenAI, Provider::Google, Provider::Anthropic];

    fn name(&self) -> &'static str {
        match self {
            Provider::OpenAI => "openai",
            Provider::Google => "google",
            Provider::Anthropic => "anthropic",
        }
    }

    fn from_arg(s: &str) -> Option<Provider> {
        match s {
            "openai" => Some(Provider::OpenAI),
            "google" | "gemini" => Some(Provider::Google),
            "anthropic" | "claude" => Some(Provider::Anthropic),
            _ => None,
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();

    let cwd = std::env::current_dir()?;
    let scenarios_path = cwd.join("tests").join("scenarios.json");
    let raw = fs::read_to_string(&scenarios_path)
        .map_err(|e| format!("read {}: {e}", scenarios_path.display()))?;
    let config: Config = serde_json::from_str(&raw)?;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let providers = parse_providers(&args);
    let scenarios = parse_scenarios(&args, &config.scenarios);

    let traces_root = cwd.join("tests/cross_provider/traces");
    fs::create_dir_all(&traces_root)?;

    let mut had_error = false;
    for &provider in &providers {
        let dir = traces_root.join(provider.name());
        fs::create_dir_all(&dir)?;

        for scenario in &scenarios {
            let overrides = scenario
                .providers
                .get(provider.name())
                .cloned()
                .unwrap_or_default();
            if overrides.skip {
                println!("[{}] {} ... skip", provider.name(), scenario.name);
                continue;
            }
            let model = overrides.model.clone().unwrap_or_else(|| {
                match provider {
                    Provider::OpenAI => config.defaults.models.openai.clone(),
                    Provider::Google => config.defaults.models.google.clone(),
                    Provider::Anthropic => config.defaults.models.anthropic.clone(),
                }
            });
            print!("[{}] {} ... ", provider.name(), scenario.name);
            std::io::Write::flush(&mut std::io::stdout()).ok();
            let started = Instant::now();
            match capture_one(provider, scenario, &model, &overrides, &dir).await {
                Ok(status) => println!("ok ({} in {:?})", status, started.elapsed()),
                Err(e) => {
                    println!("ERR: {e}");
                    had_error = true;
                }
            }
        }
    }

    if had_error {
        std::process::exit(1);
    }
    Ok(())
}

fn parse_providers(args: &[String]) -> Vec<Provider> {
    let want: Vec<Provider> = args
        .iter()
        .filter_map(|a| Provider::from_arg(a))
        .collect();
    if want.is_empty() {
        Provider::ALL.to_vec()
    } else {
        want
    }
}

fn parse_scenarios<'a>(args: &[String], all: &'a [Scenario]) -> Vec<&'a Scenario> {
    let names: Vec<&str> = args
        .iter()
        .filter(|a| Provider::from_arg(a).is_none())
        .map(|s| s.as_str())
        .collect();
    if names.is_empty() {
        all.iter().collect()
    } else {
        all.iter()
            .filter(|s| names.iter().any(|n| *n == s.name))
            .collect()
    }
}

async fn capture_one(
    provider: Provider,
    scenario: &Scenario,
    model: &str,
    overrides: &ProviderOverride,
    dir: &Path,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut body = build_body(provider, scenario, model);
    if let Some(extra) = &overrides.extra_body {
        merge_into(&mut body, extra);
    }
    let endpoint = endpoint_for(provider, model)?;
    let token = match &overrides.auth_override {
        Some(t) => t.clone(),
        None => token_for(provider).await?,
    };

    let client = Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .build()?;

    let started = Instant::now();
    let response = client
        .post(&endpoint)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&body)?)
        .send()
        .await?;

    let status = response.status();
    let mut headers = BTreeMap::<String, Value>::new();
    for (k, v) in response.headers() {
        if interesting_header(k.as_str()) {
            if let Ok(s) = v.to_str() {
                headers.insert(k.as_str().to_string(), Value::String(s.to_string()));
            }
        }
    }

    let mut response_bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        response_bytes.extend_from_slice(&chunk);
    }
    let elapsed = started.elapsed();

    let req_path = dir.join(format!("{}.request.json", scenario.name));
    let resp_path = dir.join(format!("{}.response.sse", scenario.name));
    let meta_path = dir.join(format!("{}.meta.json", scenario.name));

    fs::write(&req_path, serde_json::to_string_pretty(&body)?)?;
    fs::write(&resp_path, &response_bytes)?;

    let captured_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let meta = json!({
        "provider": provider.name(),
        "scenario": scenario.name,
        "model": model,
        "endpoint": endpoint,
        "status": status.as_u16(),
        "headers": headers,
        "captured_at_unix": captured_at,
        "latency_ms": elapsed.as_millis(),
        "response_bytes": response_bytes.len(),
        "expect_failure": scenario.expect_failure,
    });
    fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;

    let succeeded_http = status.is_success();
    if scenario.expect_failure {
        if succeeded_http {
            return Err(format!(
                "expected non-2xx but got HTTP {} (saved body to {})",
                status,
                resp_path.display(),
            )
            .into());
        }
        return Ok(format!(
            "{} expect_failure status={}",
            resp_path.display(),
            status.as_u16(),
        ));
    }
    if !succeeded_http {
        return Err(format!(
            "HTTP {} (saved body to {})",
            status,
            resp_path.display(),
        )
        .into());
    }
    Ok(format!("{} status={}", resp_path.display(), status.as_u16()))
}

/// Recursively merge `src` into `dst`. Object keys from `src` overwrite
/// scalars and merge into nested objects; arrays are replaced wholesale.
fn merge_into(dst: &mut Value, src: &Value) {
    match (dst, src) {
        (Value::Object(d), Value::Object(s)) => {
            for (k, v) in s {
                merge_into(d.entry(k.clone()).or_insert(Value::Null), v);
            }
        }
        (slot, other) => {
            *slot = other.clone();
        }
    }
}

/// Headers worth keeping. Skip cookies, opaque IDs that vary per request, etc.
fn interesting_header(name: &str) -> bool {
    matches!(
        name,
        "content-type"
            | "openai-version"
            | "openai-model"
            | "openai-processing-ms"
            | "openai-organization"
            | "x-request-id"
            | "anthropic-ratelimit-requests-limit"
            | "anthropic-ratelimit-tokens-limit"
            | "retry-after"
    )
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

async fn token_for(provider: Provider) -> Result<String, Box<dyn std::error::Error>> {
    match provider {
        Provider::OpenAI => Ok(std::env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY not set in env or .env")?),
        Provider::Google | Provider::Anthropic => vertex_token().await,
    }
}

async fn vertex_token() -> Result<String, Box<dyn std::error::Error>> {
    // Preference order: explicit gcloud impersonation if the email is set
    // and gcloud is on PATH; otherwise fall back to whatever ADC resolves
    // to (GOOGLE_APPLICATION_CREDENTIALS, gcloud user creds, GCE metadata).
    if let Ok(email) = std::env::var("GOOGLE_SERVICE_ACCOUNT_EMAIL") {
        if !email.is_empty() {
            match gcloud_impersonation_token(&email) {
                Ok(token) => return Ok(token),
                Err(e) => {
                    eprintln!("(impersonation via gcloud failed: {e}; falling back to ADC)");
                }
            }
        }
    }
    let provider = gcp_auth::provider().await?;
    let token = provider
        .token(&["https://www.googleapis.com/auth/cloud-platform"])
        .await?;
    Ok(token.as_str().to_string())
}

fn gcloud_impersonation_token(email: &str) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("gcloud")
        .args([
            "auth",
            "print-access-token",
            "--impersonate-service-account",
            email,
        ])
        .output()
        .map_err(|e| format!("could not launch gcloud: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "gcloud exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim(),
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

// ---------------------------------------------------------------------------
// Endpoint resolution
// ---------------------------------------------------------------------------

fn endpoint_for(provider: Provider, model: &str) -> Result<String, Box<dyn std::error::Error>> {
    match provider {
        Provider::OpenAI => Ok("https://api.openai.com/v1/responses".to_string()),
        Provider::Google => {
            let (project, region) = vertex_project_region(provider)?;
            Ok(format!(
                "{host}/v1/projects/{project}/locations/{region}/publishers/google/models/{model}:streamGenerateContent?alt=sse",
                host = vertex_host(&region),
            ))
        }
        Provider::Anthropic => {
            let (project, region) = vertex_project_region(provider)?;
            // Match the lib's URL exactly (including `?alt=sse`) so the
            // capture verifies what we actually send, not an idealized
            // shape.
            Ok(format!(
                "{host}/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:streamRawPredict?alt=sse",
                host = vertex_host(&region),
            ))
        }
    }
}

fn vertex_project_region(
    provider: Provider,
) -> Result<(String, String), Box<dyn std::error::Error>> {
    let project = std::env::var("GOOGLE_PROJECT_ID")
        .map_err(|_| "GOOGLE_PROJECT_ID not set in env or .env")?;
    // Anthropic models are only enabled in some Vertex regions; allow
    // ANTHROPIC_REGION to override GOOGLE_REGION specifically for Anthropic.
    let region = match provider {
        Provider::Anthropic => std::env::var("ANTHROPIC_REGION")
            .or_else(|_| std::env::var("GOOGLE_REGION"))
            .map_err(|_| "neither ANTHROPIC_REGION nor GOOGLE_REGION is set")?,
        _ => std::env::var("GOOGLE_REGION")
            .map_err(|_| "GOOGLE_REGION not set in env or .env")?,
    };
    Ok((project, region))
}

fn vertex_host(region: &str) -> String {
    if region == "global" {
        "https://aiplatform.googleapis.com".to_string()
    } else {
        format!("https://{region}-aiplatform.googleapis.com")
    }
}

// ---------------------------------------------------------------------------
// Body construction (independent of the lib's convert_request — the point of
// capture is to verify our wire shape end-to-end against a real API).
// ---------------------------------------------------------------------------

fn build_body(provider: Provider, scenario: &Scenario, model: &str) -> Value {
    match provider {
        Provider::OpenAI => build_openai(scenario, model),
        Provider::Google => build_gemini(scenario, model),
        Provider::Anthropic => build_anthropic(scenario, model),
    }
}

fn build_openai(scenario: &Scenario, model: &str) -> Value {
    // OpenAI Responses API: function_call and function_call_output are
    // top-level input items, not nested under role=assistant. Walk the
    // unified scenario messages and split assistant turns with tool_calls
    // into a message + N function_call items.
    let mut input: Vec<Value> = Vec::new();
    for m in &scenario.messages {
        match m.role.as_str() {
            "tool" => {
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": m.tool_call_id.clone().unwrap_or_default(),
                    "output": m.content.clone().unwrap_or_default(),
                }));
            }
            "assistant" if !m.tool_calls.is_empty() => {
                if let Some(text) = m.content.as_ref().filter(|s| !s.is_empty()) {
                    input.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": text,
                    }));
                }
                for tc in &m.tool_calls {
                    input.push(json!({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }));
                }
            }
            other => {
                input.push(json!({
                    "type": "message",
                    "role": other,
                    "content": m.content.clone().unwrap_or_default(),
                }));
            }
        }
    }

    let mut body = json!({
        "model": model,
        "input": input,
        "stream": true,
        "store": false,
    });
    if let Some(t) = scenario.temperature {
        body["temperature"] = json!(t);
    }
    if let Some(m) = scenario.max_tokens {
        body["max_output_tokens"] = json!(m);
    }
    if !scenario.tools.is_empty() {
        let tools: Vec<Value> = scenario
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                })
            })
            .collect();
        body["tools"] = Value::Array(tools);
    }
    body
}

fn build_gemini(scenario: &Scenario, _model: &str) -> Value {
    // Gemini routes the system message to a separate `systemInstruction`
    // field; everything else lands in `contents` with role=user|model.
    // Tool results go in role=user content with a functionResponse part;
    // assistant tool calls go in role=model content with functionCall parts.
    let mut contents: Vec<Value> = Vec::new();
    let mut system_instruction: Option<Value> = None;
    for m in &scenario.messages {
        match m.role.as_str() {
            "system" => {
                if let Some(text) = m.content.as_ref() {
                    system_instruction = Some(json!({
                        "role": "system",
                        "parts": [{ "text": text }],
                    }));
                }
            }
            "tool" => {
                // functionResponse always lands under role=user per the
                // Gemini contract. The `name` field must echo the original
                // tool name; we don't have it directly here, so we leave it
                // empty and rely on tests that need it to do their own
                // wiring. In practice scenarios that exercise tool round-
                // trip should pre-populate the `name` via the
                // ScenarioMessage.tool_call_id mapped from the prior
                // assistant turn.
                let response = serde_json::from_str::<Value>(
                    m.content.as_deref().unwrap_or("{}"),
                )
                .unwrap_or_else(|_| json!({ "result": m.content.clone().unwrap_or_default() }));
                let tool_name = lookup_tool_name(scenario, m.tool_call_id.as_deref());
                contents.push(json!({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": tool_name,
                            "response": response,
                        }
                    }],
                }));
            }
            "assistant" if !m.tool_calls.is_empty() => {
                let mut parts: Vec<Value> = Vec::new();
                if let Some(text) = m.content.as_ref().filter(|s| !s.is_empty()) {
                    parts.push(json!({ "text": text }));
                }
                for tc in &m.tool_calls {
                    let args = serde_json::from_str::<Value>(&tc.arguments)
                        .unwrap_or_else(|_| Value::String(tc.arguments.clone()));
                    parts.push(json!({
                        "functionCall": {
                            "name": tc.name,
                            "args": args,
                        }
                    }));
                }
                contents.push(json!({ "role": "model", "parts": parts }));
            }
            "assistant" => {
                if let Some(text) = m.content.as_ref() {
                    contents.push(json!({
                        "role": "model",
                        "parts": [{ "text": text }],
                    }));
                }
            }
            _ => {
                if let Some(text) = m.content.as_ref() {
                    contents.push(json!({
                        "role": "user",
                        "parts": [{ "text": text }],
                    }));
                }
            }
        }
    }
    let mut body = json!({ "contents": contents });
    if let Some(si) = system_instruction {
        body["systemInstruction"] = si;
    }
    let mut gen_cfg = Map::new();
    if let Some(t) = scenario.temperature {
        gen_cfg.insert("temperature".to_string(), json!(t));
    }
    if let Some(m) = scenario.max_tokens {
        gen_cfg.insert("maxOutputTokens".to_string(), json!(m));
    }
    if !gen_cfg.is_empty() {
        body["generationConfig"] = Value::Object(gen_cfg);
    }
    if !scenario.tools.is_empty() {
        let decls: Vec<Value> = scenario
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                })
            })
            .collect();
        body["tools"] = json!([{ "functionDeclarations": decls }]);
    }
    body
}

/// Find the tool name that matches a given `tool_call_id` by scanning the
/// scenario's prior assistant `tool_calls`. Falls back to the first
/// declared tool if the id isn't found, which is the best we can do
/// without a richer schema. Gemini requires the name on functionResponse.
fn lookup_tool_name(scenario: &Scenario, call_id: Option<&str>) -> String {
    if let Some(id) = call_id {
        for m in &scenario.messages {
            for tc in &m.tool_calls {
                if tc.id == id {
                    return tc.name.clone();
                }
            }
        }
    }
    scenario
        .tools
        .first()
        .map(|t| t.name.clone())
        .unwrap_or_default()
}

fn build_anthropic(scenario: &Scenario, _model: &str) -> Value {
    // Anthropic via Vertex omits the top-level `model` field (model is in
    // the URL) and pins `anthropic_version`. Assistant tool calls become
    // tool_use blocks; tool results become user messages with tool_result
    // blocks.
    let mut messages = Vec::<Value>::new();
    let mut system: Option<String> = None;
    for m in &scenario.messages {
        match m.role.as_str() {
            "system" => system = m.content.clone(),
            "tool" => {
                messages.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
                        "content": m.content.clone().unwrap_or_default(),
                    }],
                }));
            }
            "assistant" if !m.tool_calls.is_empty() => {
                let mut blocks: Vec<Value> = Vec::new();
                if let Some(text) = m.content.as_ref().filter(|s| !s.is_empty()) {
                    blocks.push(json!({ "type": "text", "text": text }));
                }
                for tc in &m.tool_calls {
                    let input = serde_json::from_str::<Value>(&tc.arguments)
                        .unwrap_or_else(|_| Value::String(tc.arguments.clone()));
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": input,
                    }));
                }
                messages.push(json!({ "role": "assistant", "content": blocks }));
            }
            role => {
                if let Some(text) = m.content.as_ref() {
                    messages.push(json!({ "role": role, "content": text }));
                }
            }
        }
    }
    let mut body = json!({
        "messages": messages,
        "max_tokens": scenario.max_tokens.unwrap_or(1024),
        "anthropic_version": "vertex-2023-10-16",
        "stream": true,
    });
    if let Some(s) = system {
        body["system"] = json!(s);
    }
    if let Some(t) = scenario.temperature {
        body["temperature"] = json!(t);
    }
    if !scenario.tools.is_empty() {
        let tools: Vec<Value> = scenario
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                })
            })
            .collect();
        body["tools"] = Value::Array(tools);
    }
    body
}
