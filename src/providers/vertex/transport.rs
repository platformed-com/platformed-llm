//! Shared transport for Vertex AI providers.
//!
//! Both `GoogleProvider` (Gemini) and `AnthropicViaVertexProvider` (Claude on
//! Vertex) hit `https://{location}-aiplatform.googleapis.com` with the same
//! auth, the same client, and almost identical URL shapes. This module owns
//! that plumbing in one place so each provider only has to translate
//! request/response payloads.
//!
//! The transport supports both static access tokens and Application Default
//! Credentials (via `gcp_auth`). For tests, callers can override the host with
//! [`VertexTransport::with_base_url`].

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use gcp_auth::TokenProvider;
use reqwest::{Client, RequestBuilder};

use crate::Error;

/// OAuth scope used for all Vertex AI calls.
const VERTEX_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// Connect timeout for the underlying HTTP client.
///
/// We deliberately do **not** set a total request timeout — streaming
/// responses (especially with extended thinking / reasoning) can run for
/// many minutes, and a whole-request timeout would abort them mid-stream.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Authentication state for Vertex AI.
#[derive(Clone)]
pub enum VertexAuth {
    /// A pre-fetched access token. The caller is responsible for refresh.
    Static(String),
    /// Application Default Credentials. The provider caches and refreshes
    /// tokens internally.
    Adc(Arc<dyn TokenProvider>),
}

impl fmt::Debug for VertexAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VertexAuth::Static(_) => f.debug_tuple("Static").field(&"<redacted>").finish(),
            VertexAuth::Adc(_) => f.debug_struct("Adc").finish_non_exhaustive(),
        }
    }
}

/// Shared transport for Vertex AI providers.
#[derive(Debug, Clone)]
pub struct VertexTransport {
    client: Client,
    project_id: String,
    location: String,
    base_url: Option<String>,
    auth: VertexAuth,
}

impl VertexTransport {
    /// Build a transport from a static access token (sync — no network calls).
    pub fn with_access_token(
        project_id: String,
        location: String,
        access_token: String,
    ) -> Result<Self, Error> {
        Ok(Self {
            client: build_client()?,
            project_id,
            location,
            base_url: None,
            auth: VertexAuth::Static(access_token),
        })
    }

    /// Build a transport using Application Default Credentials. Async because
    /// `gcp_auth::provider()` may need to discover the credential source.
    pub async fn with_adc(project_id: String, location: String) -> Result<Self, Error> {
        let provider = gcp_auth::provider()
            .await
            .map_err(|e| Error::auth(format!("failed to create ADC provider: {e}")))?;
        Ok(Self {
            client: build_client()?,
            project_id,
            location,
            base_url: None,
            auth: VertexAuth::Adc(provider),
        })
    }

    /// Override the base URL (scheme + host). Intended for tests using a mock
    /// server. The path that follows is still constructed by [`endpoint`].
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// The HTTP client.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The configured location/region (e.g. `us-east1`, `global`).
    pub fn location(&self) -> &str {
        &self.location
    }

    /// Construct the URL for a Vertex AI prediction endpoint.
    ///
    /// - `publisher` is the model publisher (`"google"` or `"anthropic"`).
    /// - `model` is the model id.
    /// - `method` is the verb (`generateContent`, `streamGenerateContent`,
    ///   `rawPredict`, `streamRawPredict`).
    /// - `query` is appended verbatim after `?`, or omitted when `None`.
    pub fn endpoint(
        &self,
        publisher: &str,
        model: &str,
        method: &str,
        query: Option<&str>,
    ) -> String {
        let host = self
            .base_url
            .as_deref()
            .map(|b| b.trim_end_matches('/').to_owned())
            .unwrap_or_else(|| default_host(&self.location));
        let mut url = format!(
            "{host}/v1/projects/{project}/locations/{location}/publishers/{publisher}/models/{model}:{method}",
            project = self.project_id,
            location = self.location,
        );
        if let Some(q) = query {
            url.push('?');
            url.push_str(q);
        }
        url
    }

    /// Resolve an access token. For ADC this delegates to the cached
    /// `gcp_auth::TokenProvider`.
    pub async fn access_token(&self) -> Result<String, Error> {
        match &self.auth {
            VertexAuth::Static(token) => Ok(token.clone()),
            VertexAuth::Adc(provider) => {
                let token = provider
                    .token(&[VERTEX_SCOPE])
                    .await
                    .map_err(|e| Error::auth(format!("ADC token fetch failed: {e}")))?;
                Ok(token.as_str().to_string())
            }
        }
    }

    /// Attach `Authorization: Bearer …` to a request builder.
    pub async fn authorize(&self, builder: RequestBuilder) -> Result<RequestBuilder, Error> {
        let token = self.access_token().await?;
        Ok(builder.header("Authorization", format!("Bearer {token}")))
    }
}

fn build_client() -> Result<Client, Error> {
    Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .build()
        .map_err(Error::from)
}

/// Resolve the default Vertex AI host for a location.
///
/// - `global` → `https://aiplatform.googleapis.com`
/// - any other location → `https://{location}-aiplatform.googleapis.com`
///
/// Note: regional multi-region aliases (`us`, `eu`) currently fall through to
/// the regional pattern. If/when we hit a regression on those, the fix lives
/// here.
fn default_host(location: &str) -> String {
    if location == "global" {
        "https://aiplatform.googleapis.com".to_string()
    } else {
        format!("https://{location}-aiplatform.googleapis.com")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transport(location: &str) -> VertexTransport {
        VertexTransport::with_access_token(
            "proj-1".to_string(),
            location.to_string(),
            "tok".to_string(),
        )
        .unwrap()
    }

    #[test]
    fn endpoint_regional_with_query() {
        let t = transport("us-east1");
        assert_eq!(
            t.endpoint("google", "gemini-1.5-pro", "streamGenerateContent", Some("alt=sse")),
            "https://us-east1-aiplatform.googleapis.com/v1/projects/proj-1/locations/us-east1/publishers/google/models/gemini-1.5-pro:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn endpoint_regional_no_query() {
        let t = transport("us-east1");
        assert_eq!(
            t.endpoint("anthropic", "claude-sonnet-4", "streamRawPredict", None),
            "https://us-east1-aiplatform.googleapis.com/v1/projects/proj-1/locations/us-east1/publishers/anthropic/models/claude-sonnet-4:streamRawPredict"
        );
    }

    #[test]
    fn endpoint_global_uses_unprefixed_host() {
        let t = transport("global");
        assert_eq!(
            t.endpoint("anthropic", "claude-sonnet-4", "streamRawPredict", None),
            "https://aiplatform.googleapis.com/v1/projects/proj-1/locations/global/publishers/anthropic/models/claude-sonnet-4:streamRawPredict"
        );
    }

    #[test]
    fn endpoint_respects_base_url_override() {
        let t = transport("us-east1").with_base_url("http://localhost:1234");
        assert_eq!(
            t.endpoint("google", "gemini", "generateContent", None),
            "http://localhost:1234/v1/projects/proj-1/locations/us-east1/publishers/google/models/gemini:generateContent"
        );
    }

    #[test]
    fn endpoint_strips_trailing_slash_from_base_url() {
        let t = transport("us-east1").with_base_url("http://localhost:1234/");
        assert_eq!(
            t.endpoint("google", "gemini", "generateContent", None),
            "http://localhost:1234/v1/projects/proj-1/locations/us-east1/publishers/google/models/gemini:generateContent"
        );
    }

    #[tokio::test]
    async fn access_token_returns_static_token() {
        let t = transport("us-east1");
        assert_eq!(t.access_token().await.unwrap(), "tok");
    }
}
