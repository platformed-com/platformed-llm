//! Capture raw SSE traces from real LLM providers for use as test fixtures.
//!
//! Each scenario is sent through the lib's `provider.generate()` with a
//! [`RecordingTransport`] that tees the actual outgoing request and the
//! actual incoming response bytes to disk. Capturing therefore exercises
//! the full lib pipeline (request building, URL routing, auth, HTTP send,
//! response parsing) end-to-end against the real provider — if the lib's
//! request shape is wrong, the real provider rejects it at capture time.
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
//! - `ANTHROPIC_REGION` — *optional*. Override `GOOGLE_REGION` for Anthropic
//!   on Vertex (Claude models are only enabled in some regions).
//! - `GOOGLE_SERVICE_ACCOUNT_EMAIL` — *optional*. If set, we shell out to
//!   `gcloud auth print-access-token --impersonate-service-account=$EMAIL`
//!   to mint a Vertex token. Otherwise we use whatever ADC resolves to via
//!   `gcp_auth::provider()`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::pin::Pin;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use platformed_llm::providers::{
    AnthropicViaVertexProvider, GoogleProvider, OpenAIProvider, VertexEndpoint,
};
use platformed_llm::transport::{Transport, TransportImpl, TransportRequest, TransportResponse};
use platformed_llm::{
    Config, Error, FunctionCall, InputItem, Prompt, ReasoningConfig, ReasoningEffort,
    ReasoningSummary, Tool,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Scenario schema (mirrors tests/scenarios.json)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ScenarioFile {
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
    /// Provider-builtin tools to add alongside `tools`. Each entry is
    /// one of: `"web_search"`, `"google_search"`, `"code_execution"`,
    /// `"computer_use"`. Providers that don't offer the named builtin
    /// silently drop it (model-switching contract).
    #[serde(default)]
    builtin_tools: Vec<String>,
    /// Optional structured-output constraint. `{"type":"json_object"}`
    /// or `{"type":"json_schema","name":...,"schema":...,"strict":...}`.
    #[serde(default)]
    response_format: Option<ScenarioResponseFormat>,
    /// If set, the last user message's `content` is expanded with
    /// repeated filler text up to roughly this many UTF-8 bytes
    /// before sending. Lets a scenario deliberately overflow a
    /// model's context window without dragging multi-megabyte
    /// strings into scenarios.json. Rough rule of thumb: English
    /// text averages ~4 bytes per token, so e.g. `600_000` bytes
    /// reliably exceeds OpenAI's 128k-token gpt-4o-mini context
    /// window. Used by `context_window_exceeded`.
    #[serde(default)]
    oversize_user_content_bytes: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ScenarioResponseFormat {
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        #[serde(default)]
        strict: bool,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct ProviderOverride {
    #[serde(default)]
    skip: bool,
    #[serde(default)]
    model: Option<String>,
    /// Literal token to send instead of the resolved bearer. Useful for 401
    /// captures.
    #[serde(default)]
    auth_override: Option<String>,
    /// Free-form provider-specific JSON. Currently translated into the
    /// lib's `Config::reasoning` and `Config::parallel_tool_calls` fields;
    /// any unknown key fails the scenario so the schema stays honest.
    #[serde(default)]
    extra_body: Option<Value>,
    /// Override the scenario-level `oversize_user_content_bytes`.
    /// Context-window sizes vary by 3 orders of magnitude across
    /// providers (gpt-4 at 8k vs gemini-2.5 at 1M), so one
    /// scenario-wide value would either skip the smaller models or
    /// upload pointless megabytes to the larger ones.
    #[serde(default)]
    oversize_user_content_bytes: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ScenarioMessage {
    role: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ScenarioToolCall>,
    #[serde(default)]
    tool_call_id: Option<String>,
    /// Multi-modal attachments. Appended to the user message after the
    /// text `content`.
    #[serde(default)]
    attachments: Vec<ScenarioAttachment>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ScenarioAttachment {
    Image { data: String, media_type: String },
    ImageUrl { url: String },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ScenarioToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ScenarioTool {
    name: String,
    #[serde(default)]
    description: String,
    parameters: Value,
}

// ---------------------------------------------------------------------------
// Provider routing
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Recording transport: wraps another Transport, tees request + response
// bytes into in-memory buffers so the capture binary can serialize them
// after the lib finishes consuming the stream.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Recording {
    request: Option<TransportRequest>,
    response_status: u16,
    response_headers: Vec<(String, String)>,
    response_body: Vec<u8>,
}

struct RecordingTransport {
    inner: Transport,
    recording: Arc<Mutex<Recording>>,
}

impl RecordingTransport {
    fn new(inner: Transport) -> (Self, Arc<Mutex<Recording>>) {
        let recording = Arc::new(Mutex::new(Recording::default()));
        (
            Self {
                inner,
                recording: recording.clone(),
            },
            recording,
        )
    }
}

#[async_trait]
impl TransportImpl for RecordingTransport {
    async fn send(&self, req: TransportRequest) -> Result<TransportResponse, Error> {
        // Snapshot the request before forwarding. TransportRequest is Clone,
        // so the recording owns its own copy.
        self.recording.lock().unwrap().request = Some(req.clone());

        let response = self.inner.send(req).await?;

        {
            let mut rec = self.recording.lock().unwrap();
            rec.response_status = response.status;
            rec.response_headers = response.headers.clone();
        }

        // Tee the body. As the lib's SSE parser pulls chunks, the same
        // bytes are appended to the recording.
        let recording = self.recording.clone();
        let teed = response.body.map(move |chunk| {
            if let Ok(bytes) = &chunk {
                recording
                    .lock()
                    .unwrap()
                    .response_body
                    .extend_from_slice(bytes);
            }
            chunk
        });
        let teed: Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>> = Box::pin(teed);

        Ok(TransportResponse {
            status: response.status,
            headers: response.headers,
            body: teed,
        })
    }
}

// ---------------------------------------------------------------------------
// Scenario → (Prompt, Config) translation
// ---------------------------------------------------------------------------

/// Pad `seed` with repeated filler so the result is at least
/// `target_bytes` UTF-8 bytes. Used by scenarios that deliberately
/// overflow a model's context window — keeps multi-megabyte strings
/// out of scenarios.json.
///
/// The filler is plain English (`"Lorem ipsum dolor sit amet, ..."`)
/// so tokenization is realistic; each word is roughly one token, so
/// `target_bytes` ≈ `4 * target_tokens` for the languages most
/// frontier models train on.
fn oversize_filler(seed: &str, target_bytes: usize) -> String {
    const FILLER: &str = " Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.";
    let mut out = String::with_capacity(target_bytes + seed.len());
    out.push_str(seed);
    while out.len() < target_bytes {
        out.push_str(FILLER);
    }
    out
}

fn scenario_to_llm_request(
    scenario: &Scenario,
    model: &str,
    overrides: &ProviderOverride,
) -> Result<(Prompt, Config), String> {
    // Walk messages, splitting assistant turns with tool_calls into
    // assistant message + N FunctionCall items, and turning tool-role
    // messages into FunctionCallOutput.
    let mut prompt = Prompt::new();
    for m in &scenario.messages {
        match m.role.as_str() {
            "tool" => {
                let id = m
                    .tool_call_id
                    .clone()
                    .ok_or_else(|| "tool message missing tool_call_id".to_string())?;
                let output = m.content.clone().unwrap_or_default();
                prompt = prompt.with_tool_result(id, output);
            }
            "assistant" if !m.tool_calls.is_empty() => {
                use platformed_llm::AssistantPart;
                let mut content: Vec<AssistantPart> = Vec::new();
                if let Some(text) = m.content.as_ref().filter(|s| !s.is_empty()) {
                    content.push(AssistantPart::Text {
                        content: text.clone(),
                        annotations: Vec::new(),
                    });
                }
                for tc in &m.tool_calls {
                    content.push(AssistantPart::ToolCall(FunctionCall {
                        call_id: tc.id.clone(),
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                    }));
                }
                prompt = prompt.with_item(InputItem::Assistant { content });
            }
            "assistant" => {
                prompt = prompt.with_assistant(m.content.clone().unwrap_or_default());
            }
            "system" => {
                prompt = prompt.with_system(m.content.clone().unwrap_or_default());
            }
            _ => {
                use platformed_llm::UserPart;
                let mut content: Vec<UserPart> = Vec::new();
                if let Some(text) = m.content.as_ref().filter(|s| !s.is_empty()) {
                    // Per-provider override wins; falls back to the
                    // scenario-level value.
                    let target = overrides
                        .oversize_user_content_bytes
                        .or(scenario.oversize_user_content_bytes);
                    let payload = match target {
                        Some(t) if t > text.len() => oversize_filler(text, t),
                        _ => text.clone(),
                    };
                    content.push(UserPart::Text(payload));
                }
                for att in &m.attachments {
                    match att {
                        ScenarioAttachment::Image { data, media_type } => {
                            content.push(UserPart::Image(platformed_llm::ImageSource::Base64 {
                                data: data.clone(),
                                media_type: media_type.clone(),
                            }));
                        }
                        ScenarioAttachment::ImageUrl { url } => {
                            content.push(UserPart::Image(platformed_llm::ImageSource::Url(
                                url.clone(),
                            )));
                        }
                    }
                }
                if content.is_empty() {
                    // Empty user message: skip rather than emit a no-content turn.
                    continue;
                }
                prompt = prompt.with_item(InputItem::User { content });
            }
        }
    }

    let mut cfg = Config::builder(model);
    if let Some(t) = scenario.temperature {
        cfg = cfg.temperature(t);
    }
    if let Some(m) = scenario.max_tokens {
        cfg = cfg.max_tokens(m);
    }
    let mut tools: Vec<Tool> = Vec::new();
    for t in &scenario.tools {
        let raw = serde_json::value::RawValue::from_string(t.parameters.to_string())
            .map_err(|e| format!("tool {} parameters: {e}", t.name))?;
        tools.push(Tool::Function(platformed_llm::Function {
            name: t.name.clone(),
            description: if t.description.is_empty() {
                None
            } else {
                Some(t.description.clone())
            },
            parameters: std::borrow::Cow::Owned(raw),
        }));
    }
    for b in &scenario.builtin_tools {
        let builtin = match b.as_str() {
            "web_search" => platformed_llm::ProviderBuiltin::WebSearch,
            "google_search" => platformed_llm::ProviderBuiltin::GoogleSearch,
            "code_execution" => platformed_llm::ProviderBuiltin::CodeExecution,
            other => {
                return Err(format!("unknown builtin_tool: {other}"));
            }
        };
        tools.push(Tool::Builtin(builtin));
    }
    if !tools.is_empty() {
        cfg = cfg.tools(tools);
    }

    if let Some(rf) = &scenario.response_format {
        let translated = match rf {
            ScenarioResponseFormat::JsonObject => platformed_llm::ResponseFormat::JsonObject,
            ScenarioResponseFormat::JsonSchema {
                name,
                schema,
                strict,
            } => {
                let raw = serde_json::value::RawValue::from_string(schema.to_string())
                    .map_err(|e| format!("response_format schema: {e}"))?;
                platformed_llm::ResponseFormat::JsonSchema {
                    name: name.clone(),
                    schema: std::borrow::Cow::Owned(raw),
                    strict: *strict,
                }
            }
        };
        cfg = cfg.response_format(translated);
    }

    // Translate the supported subset of `extra_body`. Anything else is
    // rejected so the schema stays honest about which features actually
    // route through the lib.
    if let Some(extra) = &overrides.extra_body {
        if let Some(obj) = extra.as_object() {
            for (k, v) in obj {
                match k.as_str() {
                    "parallel_tool_calls" => {
                        let b = v
                            .as_bool()
                            .ok_or_else(|| "parallel_tool_calls must be a bool".to_string())?;
                        cfg = cfg.parallel_tool_calls(b);
                    }
                    "reasoning" => {
                        let r = parse_reasoning(v)?;
                        cfg = cfg.reasoning(r);
                    }
                    other => {
                        return Err(format!(
                            "extra_body field `{other}` is not currently routed through Config; \
                             extend scenario_to_llm_request to support it"
                        ));
                    }
                }
            }
        }
    }

    Ok((prompt, cfg.build()))
}

fn parse_reasoning(v: &Value) -> Result<ReasoningConfig, String> {
    let obj = v.as_object().ok_or("reasoning must be an object")?;
    let effort = match obj.get("effort").and_then(|x| x.as_str()) {
        Some("low") => Some(ReasoningEffort::Low),
        Some("medium") => Some(ReasoningEffort::Medium),
        Some("high") => Some(ReasoningEffort::High),
        Some(other) => return Err(format!("unknown reasoning.effort: {other}")),
        None => None,
    };
    let summary = match obj.get("summary").and_then(|x| x.as_str()) {
        Some("auto") => Some(ReasoningSummary::Auto),
        Some("concise") => Some(ReasoningSummary::Concise),
        Some("detailed") => Some(ReasoningSummary::Detailed),
        Some(other) => return Err(format!("unknown reasoning.summary: {other}")),
        None => None,
    };
    Ok(ReasoningConfig { effort, summary })
}

// ---------------------------------------------------------------------------
// Auth resolution (Vertex via gcloud impersonation or ADC; OpenAI via env)
// ---------------------------------------------------------------------------

async fn openai_token() -> Result<String, Box<dyn std::error::Error>> {
    Ok(std::env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY not set in env or .env")?)
}

async fn vertex_token() -> Result<String, Box<dyn std::error::Error>> {
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

fn vertex_project_region(
    provider: Provider,
) -> Result<(String, String), Box<dyn std::error::Error>> {
    let project = std::env::var("GOOGLE_PROJECT_ID")
        .map_err(|_| "GOOGLE_PROJECT_ID not set in env or .env")?;
    let region = match provider {
        Provider::Anthropic => std::env::var("ANTHROPIC_REGION")
            .or_else(|_| std::env::var("GOOGLE_REGION"))
            .map_err(|_| "neither ANTHROPIC_REGION nor GOOGLE_REGION is set")?,
        _ => std::env::var("GOOGLE_REGION").map_err(|_| "GOOGLE_REGION not set in env or .env")?,
    };
    Ok((project, region))
}

// ---------------------------------------------------------------------------
// Capture entry points
// ---------------------------------------------------------------------------

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
    let scenario_file: ScenarioFile = serde_json::from_str(&raw)?;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let providers = parse_providers(&args);
    let scenarios = parse_scenarios(&args, &scenario_file.scenarios);

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
            let model = overrides.model.clone().unwrap_or_else(|| match provider {
                Provider::OpenAI => scenario_file.defaults.models.openai.clone(),
                Provider::Google => scenario_file.defaults.models.google.clone(),
                Provider::Anthropic => scenario_file.defaults.models.anthropic.clone(),
            });
            print!("[{}] {} ... ", provider.name(), scenario.name);
            std::io::Write::flush(&mut std::io::stdout()).ok();
            let started = Instant::now();
            match capture_one(provider, scenario, &model, &overrides, &dir).await {
                Ok(status) => println!("ok ({status} in {:?})", started.elapsed()),
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
    let want: Vec<Provider> = args.iter().filter_map(|a| Provider::from_arg(a)).collect();
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
    let (prompt, cfg) = scenario_to_llm_request(scenario, model, overrides)
        .map_err(|e| format!("scenario translation: {e}"))?;

    // Build the recording transport that wraps the default reqwest one.
    let (recorder, recording) = RecordingTransport::new(Transport::reqwest()?);
    let transport = Transport::new(recorder);

    // Resolve auth (or accept a deliberate override for 4xx captures) and
    // construct the right provider with our recording transport.
    let started = Instant::now();
    let generate_result = match provider {
        Provider::OpenAI => {
            let api_key = match &overrides.auth_override {
                Some(t) => t.clone(),
                None => openai_token().await?,
            };
            let p = OpenAIProvider::with_transport(
                api_key,
                "https://api.openai.com/v1".to_string(),
                transport,
            );
            run_provider(&p, &prompt, &cfg).await
        }
        Provider::Google => {
            let (project_id, region) = vertex_project_region(provider)?;
            let token = match &overrides.auth_override {
                Some(t) => t.clone(),
                None => vertex_token().await?,
            };
            let endpoint = VertexEndpoint::with_access_token(project_id, region, token);
            let p = GoogleProvider::with_transport(endpoint, transport);
            run_provider(&p, &prompt, &cfg).await
        }
        Provider::Anthropic => {
            let (project_id, region) = vertex_project_region(provider)?;
            let token = match &overrides.auth_override {
                Some(t) => t.clone(),
                None => vertex_token().await?,
            };
            let endpoint = VertexEndpoint::with_access_token(project_id, region, token);
            let p = AnthropicViaVertexProvider::with_transport(endpoint, transport);
            run_provider(&p, &prompt, &cfg).await
        }
    };
    let elapsed = started.elapsed();

    // The recording is now populated regardless of whether generate() ok'd
    // or err'd: success-path drained the body via SSE, error-path drained
    // it via collect_body inside the lib. Either way, our tee saw the
    // bytes flow through.
    let captured = recording.lock().unwrap();
    let request = captured
        .request
        .as_ref()
        .ok_or("recorded request is missing — provider didn't call Transport::send")?
        .clone();
    let status = captured.response_status;
    let headers = captured.response_headers.clone();
    let response_body = captured.response_body.clone();
    drop(captured);

    let req_path = dir.join(format!("{}.request.json", scenario.name));
    let resp_path = dir.join(format!("{}.response.sse", scenario.name));
    let meta_path = dir.join(format!("{}.meta.json", scenario.name));

    // Pretty-print the request body (it's serialized JSON internally).
    let req_pretty = match serde_json::from_slice::<Value>(&request.body) {
        Ok(v) => serde_json::to_string_pretty(&v)?,
        Err(_) => String::from_utf8_lossy(&request.body).into_owned(),
    };
    fs::write(&req_path, req_pretty)?;
    fs::write(&resp_path, &response_body)?;

    let captured_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let header_filter: BTreeMap<String, Value> = headers
        .iter()
        .filter(|(k, _)| interesting_header(k))
        .map(|(k, v)| (k.to_lowercase(), Value::String(v.clone())))
        .collect();
    let meta = json!({
        "provider": provider.name(),
        "scenario": scenario.name,
        "model": model,
        "endpoint": request.url,
        "status": status,
        "headers": header_filter,
        "captured_at_unix": captured_at,
        "latency_ms": elapsed.as_millis(),
        "response_bytes": response_body.len(),
        "expect_failure": scenario.expect_failure,
    });
    fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;

    let succeeded_http = (200..300).contains(&status);
    if scenario.expect_failure {
        if succeeded_http {
            return Err(format!(
                "expected non-2xx but got HTTP {status} (saved body to {})",
                resp_path.display(),
            )
            .into());
        }
        return Ok(format!(
            "{} expect_failure status={status}",
            resp_path.display()
        ));
    }
    // Re-raise any error from the lib so the run is marked failed.
    generate_result.map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?;
    Ok(format!("{} status={status}", resp_path.display()))
}

/// Drive `platformed_llm::generate()` and consume the resulting stream
/// so the recording transport sees every byte. Discards the unified
/// events — the snapshot test will replay the captured bytes through
/// the lib later.
async fn run_provider(
    provider: &dyn platformed_llm::Provider,
    prompt: &Prompt,
    config: &Config,
) -> Result<(), Error> {
    let response = platformed_llm::generate(provider, prompt, config).await?;
    let mut stream = response.stream();
    while let Some(ev) = stream.next().await {
        // A streaming-level error becomes our error too.
        ev?;
    }
    Ok(())
}

/// Headers worth keeping in meta.json. Skip cookies and opaque per-request IDs.
fn interesting_header(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    matches!(
        lower.as_str(),
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
