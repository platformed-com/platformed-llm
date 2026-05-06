//! Shared auth + URL routing for Vertex AI providers.
//!
//! Both `GoogleProvider` (Gemini) and `AnthropicViaVertexProvider` (Claude on
//! Vertex) hit `https://{location}-aiplatform.googleapis.com` with the same
//! Google auth flow and almost identical URL shapes. This module owns those
//! two pieces — *not* the HTTP client, which is now a top-level
//! [`crate::Transport`] each provider holds independently.
//!
//! The endpoint supports both static access tokens and Application Default
//! Credentials (via `gcp_auth`). Tests can override the host with
//! [`VertexEndpoint::with_base_url`].
//!
//! Renamed from `VertexTransport` once the actual HTTP transport became a
//! lib-wide concept; calling this a "transport" was misleading because it
//! never carried bytes.

use std::fmt;
use std::sync::Arc;

use gcp_auth::TokenProvider;

use crate::Error;

/// OAuth scope used for all Vertex AI calls.
const VERTEX_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

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

/// Vertex AI auth + URL helper. Cheap to clone (auth state is `Arc`d
/// internally for the ADC variant).
#[derive(Debug, Clone)]
pub struct VertexEndpoint {
    project_id: String,
    location: String,
    base_url: Option<String>,
    auth: VertexAuth,
}

impl VertexEndpoint {
    /// Build from a static access token (sync — no network calls).
    pub fn with_access_token(
        project_id: String,
        location: String,
        access_token: String,
    ) -> Self {
        Self {
            project_id,
            location,
            base_url: None,
            auth: VertexAuth::Static(access_token),
        }
    }

    /// Build using Application Default Credentials. Async because
    /// `gcp_auth::provider()` may need to discover the credential source.
    pub async fn with_adc(project_id: String, location: String) -> Result<Self, Error> {
        let provider = gcp_auth::provider()
            .await
            .map_err(|e| Error::auth(format!("failed to create ADC provider: {e}")))?;
        Ok(Self {
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
    pub fn url(
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

    /// Build the `Authorization: Bearer …` header tuple. Sugar over
    /// [`access_token`] for the common case where the provider just needs
    /// to attach it to a `TransportRequest.headers`.
    pub async fn auth_header(&self) -> Result<(String, String), Error> {
        let token = self.access_token().await?;
        Ok(("Authorization".to_string(), format!("Bearer {token}")))
    }
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

    fn endpoint(location: &str) -> VertexEndpoint {
        VertexEndpoint::with_access_token(
            "proj-1".to_string(),
            location.to_string(),
            "tok".to_string(),
        )
    }

    #[test]
    fn url_regional_with_query() {
        let t = endpoint("us-east1");
        assert_eq!(
            t.url("google", "gemini-1.5-pro", "streamGenerateContent", Some("alt=sse")),
            "https://us-east1-aiplatform.googleapis.com/v1/projects/proj-1/locations/us-east1/publishers/google/models/gemini-1.5-pro:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn url_regional_no_query() {
        let t = endpoint("us-east1");
        assert_eq!(
            t.url("anthropic", "claude-sonnet-4", "streamRawPredict", None),
            "https://us-east1-aiplatform.googleapis.com/v1/projects/proj-1/locations/us-east1/publishers/anthropic/models/claude-sonnet-4:streamRawPredict"
        );
    }

    #[test]
    fn url_global_uses_unprefixed_host() {
        let t = endpoint("global");
        assert_eq!(
            t.url("anthropic", "claude-sonnet-4", "streamRawPredict", None),
            "https://aiplatform.googleapis.com/v1/projects/proj-1/locations/global/publishers/anthropic/models/claude-sonnet-4:streamRawPredict"
        );
    }

    #[test]
    fn url_respects_base_url_override() {
        let t = endpoint("us-east1").with_base_url("http://localhost:1234");
        assert_eq!(
            t.url("google", "gemini", "generateContent", None),
            "http://localhost:1234/v1/projects/proj-1/locations/us-east1/publishers/google/models/gemini:generateContent"
        );
    }

    #[test]
    fn url_strips_trailing_slash_from_base_url() {
        let t = endpoint("us-east1").with_base_url("http://localhost:1234/");
        assert_eq!(
            t.url("google", "gemini", "generateContent", None),
            "http://localhost:1234/v1/projects/proj-1/locations/us-east1/publishers/google/models/gemini:generateContent"
        );
    }

    #[tokio::test]
    async fn access_token_returns_static_token() {
        let t = endpoint("us-east1");
        assert_eq!(t.access_token().await.unwrap(), "tok");
    }

    #[tokio::test]
    async fn auth_header_sugar() {
        let t = endpoint("us-east1");
        assert_eq!(
            t.auth_header().await.unwrap(),
            ("Authorization".to_string(), "Bearer tok".to_string()),
        );
    }
}
