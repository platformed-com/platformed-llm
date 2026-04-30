use crate::providers::vertex::{AnthropicViaVertexProvider, GoogleProvider};
use crate::{Error, LLMProvider, OpenAIProvider};
use std::env;

/// Supported LLM providers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderType {
    OpenAI,
    Google,
    Anthropic,
}

impl ProviderType {
    /// Check if this provider type is supported via Vertex AI.
    pub fn is_supported_via_vertex(&self) -> bool {
        matches!(self, ProviderType::Google | ProviderType::Anthropic)
    }
}

/// Configuration for creating providers.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub provider_type: ProviderType,
    pub api_key: Option<String>,
    pub project_id: Option<String>,
    pub location: Option<String>,
    pub access_token: Option<String>,
}

impl ProviderConfig {
    /// Create configuration for OpenAI provider.
    pub fn openai(api_key: String) -> Self {
        Self {
            provider_type: ProviderType::OpenAI,
            api_key: Some(api_key),
            project_id: None,
            location: None,
            access_token: None,
        }
    }

    /// Create configuration for any Vertex AI provider with access token.
    ///
    /// Returns `Err` if `provider_type` is not supported via Vertex AI
    /// (e.g. `ProviderType::OpenAI`).
    pub fn vertex(
        provider_type: ProviderType,
        project_id: String,
        location: String,
        access_token: String,
    ) -> Result<Self, Error> {
        if !provider_type.is_supported_via_vertex() {
            return Err(Error::config(format!(
                "{provider_type:?} is not a Vertex AI provider; use ProviderConfig::openai()",
            )));
        }
        Ok(Self {
            provider_type,
            api_key: None,
            project_id: Some(project_id),
            location: Some(location),
            access_token: Some(access_token),
        })
    }

    /// Create configuration for any Vertex AI provider with Application
    /// Default Credentials.
    ///
    /// Returns `Err` if `provider_type` is not supported via Vertex AI.
    pub fn vertex_with_adc(
        provider_type: ProviderType,
        project_id: String,
        location: String,
    ) -> Result<Self, Error> {
        if !provider_type.is_supported_via_vertex() {
            return Err(Error::config(format!(
                "{provider_type:?} is not a Vertex AI provider; use ProviderConfig::openai()",
            )));
        }
        Ok(Self {
            provider_type,
            api_key: None,
            project_id: Some(project_id),
            location: Some(location),
            access_token: None,
        })
    }

    /// Create configuration from environment variables.
    pub fn from_env() -> Result<Self, Error> {
        // Check for explicit provider type first
        if let Ok(provider_type) = env::var("PROVIDER_TYPE") {
            match provider_type.to_lowercase().as_str() {
                "openai" => {
                    let api_key = env::var("OPENAI_API_KEY").map_err(|_| {
                        Error::config(
                            "OPENAI_API_KEY environment variable is required for OpenAI provider",
                        )
                    })?;
                    return Ok(Self::openai(api_key));
                }
                "google" => {
                    let project_id = env::var("GOOGLE_CLOUD_PROJECT")
                        .map_err(|_| Error::config("GOOGLE_CLOUD_PROJECT environment variable is required for Google provider"))?;
                    let location = env::var("GOOGLE_CLOUD_REGION")
                        .unwrap_or_else(|_| "europe-west1".to_string());

                    if let Ok(access_token) = env::var("VERTEX_ACCESS_TOKEN") {
                        return Self::vertex(
                            ProviderType::Google,
                            project_id,
                            location,
                            access_token,
                        );
                    } else {
                        return Self::vertex_with_adc(
                            ProviderType::Google,
                            project_id,
                            location,
                        );
                    }
                }
                "anthropic" => {
                    let project_id = env::var("GOOGLE_CLOUD_PROJECT")
                        .map_err(|_| Error::config("GOOGLE_CLOUD_PROJECT environment variable is required for Anthropic provider"))?;
                    let location = env::var("GOOGLE_CLOUD_REGION")
                        .unwrap_or_else(|_| "europe-west1".to_string());

                    if let Ok(access_token) = env::var("VERTEX_ACCESS_TOKEN") {
                        return Self::vertex(
                            ProviderType::Anthropic,
                            project_id,
                            location,
                            access_token,
                        );
                    } else {
                        return Self::vertex_with_adc(
                            ProviderType::Anthropic,
                            project_id,
                            location,
                        );
                    }
                }
                _ => {
                    return Err(Error::config(format!(
                        "Invalid PROVIDER_TYPE '{provider_type}'. Valid values are: openai, google, anthropic"
                    )));
                }
            }
        }

        // Fallback to credential-based inference for backward compatibility
        // Try OpenAI first
        if let Ok(api_key) = env::var("OPENAI_API_KEY") {
            return Ok(Self::openai(api_key));
        }

        // Try Google/Vertex with access token
        if let Ok(access_token) = env::var("VERTEX_ACCESS_TOKEN") {
            let project_id = env::var("GOOGLE_CLOUD_PROJECT").map_err(|_| {
                Error::config("GOOGLE_CLOUD_PROJECT environment variable is required for Google")
            })?;
            let location =
                env::var("GOOGLE_CLOUD_REGION").unwrap_or_else(|_| "europe-west1".to_string());

            return Self::vertex(ProviderType::Google, project_id, location, access_token);
        }

        // Try Anthropic/Vertex with access token
        if let Ok(access_token) = env::var("ANTHROPIC_VERTEX_ACCESS_TOKEN") {
            let project_id = env::var("GOOGLE_CLOUD_PROJECT").map_err(|_| {
                Error::config("GOOGLE_CLOUD_PROJECT environment variable is required for Anthropic")
            })?;
            let location =
                env::var("GOOGLE_CLOUD_REGION").unwrap_or_else(|_| "europe-west1".to_string());

            return Self::vertex(ProviderType::Anthropic, project_id, location, access_token);
        }

        // Try Google/Vertex with Application Default Credentials
        if env::var("GOOGLE_APPLICATION_CREDENTIALS").is_ok()
            || env::var("GOOGLE_CLOUD_PROJECT").is_ok()
        {
            let project_id = env::var("GOOGLE_CLOUD_PROJECT").map_err(|_| {
                Error::config("GOOGLE_CLOUD_PROJECT environment variable is required for Google")
            })?;
            let location =
                env::var("GOOGLE_CLOUD_REGION").unwrap_or_else(|_| "europe-west1".to_string());

            // Check if this should be Anthropic instead of Google
            if env::var("ANTHROPIC_MODEL").is_ok() {
                return Self::vertex_with_adc(ProviderType::Anthropic, project_id, location);
            } else {
                return Self::vertex_with_adc(ProviderType::Google, project_id, location);
            }
        }

        Err(Error::config("No valid API credentials found in environment. Set PROVIDER_TYPE (openai/google/anthropic) with appropriate credentials"))
    }
}

/// Factory for creating LLM providers.
pub struct ProviderFactory;

impl ProviderFactory {
    /// Create a provider from configuration.
    pub async fn create(config: &ProviderConfig) -> Result<Box<dyn LLMProvider>, Error> {
        match config.provider_type {
            ProviderType::OpenAI => {
                let api_key = config
                    .api_key
                    .as_ref()
                    .ok_or_else(|| Error::config("API key required for OpenAI provider"))?;
                let provider = OpenAIProvider::new(api_key.clone())?;
                Ok(Box::new(provider))
            }
            ProviderType::Google => {
                let project_id = config
                    .project_id
                    .as_ref()
                    .ok_or_else(|| Error::config("Project ID required for Google provider"))?;
                let location = config
                    .location
                    .as_ref()
                    .ok_or_else(|| Error::config("Location required for Google provider"))?;
                // Determine authentication method
                let provider = if let Some(access_token) = &config.access_token {
                    GoogleProvider::new(project_id.clone(), location.clone(), access_token.clone())?
                } else {
                    // Use Application Default Credentials
                    GoogleProvider::with_adc(project_id.clone(), location.clone()).await?
                };
                Ok(Box::new(provider))
            }
            ProviderType::Anthropic => {
                let project_id = config
                    .project_id
                    .as_ref()
                    .ok_or_else(|| Error::config("Project ID required for Anthropic provider"))?;
                let location = config
                    .location
                    .as_ref()
                    .ok_or_else(|| Error::config("Location required for Anthropic provider"))?;
                // Determine authentication method
                let provider = if let Some(access_token) = &config.access_token {
                    AnthropicViaVertexProvider::new(
                        project_id.clone(),
                        location.clone(),
                        access_token.clone(),
                    )?
                } else {
                    // Use Application Default Credentials
                    AnthropicViaVertexProvider::with_adc(project_id.clone(), location.clone())
                        .await?
                };
                Ok(Box::new(provider))
            }
        }
    }

    /// Create a provider from environment variables.
    pub async fn from_env() -> Result<Box<dyn LLMProvider>, Error> {
        let config = ProviderConfig::from_env()?;
        Self::create(&config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_with_explicit_provider_types() {
        // Test direct vertex() method with Google
        let google_config = ProviderConfig::vertex(
            ProviderType::Google,
            "test-project".to_string(),
            "europe-west1".to_string(),
            "test-token".to_string(),
        )
        .unwrap();
        assert!(matches!(google_config.provider_type, ProviderType::Google));

        // Test direct vertex() method with Anthropic
        let anthropic_config = ProviderConfig::vertex(
            ProviderType::Anthropic,
            "test-project".to_string(),
            "us-east5".to_string(),
            "test-token".to_string(),
        )
        .unwrap();
        assert!(matches!(
            anthropic_config.provider_type,
            ProviderType::Anthropic
        ));
    }

    #[test]
    fn test_vertex_returns_err_on_openai() {
        let err = ProviderConfig::vertex(
            ProviderType::OpenAI,
            "test-project".to_string(),
            "us-east1".to_string(),
            "test-token".to_string(),
        )
        .expect_err("OpenAI is not a Vertex provider");
        assert!(format!("{err}").contains("not a Vertex AI provider"));
    }

    #[test]
    fn test_vertex_with_adc_returns_err_on_openai() {
        let err = ProviderConfig::vertex_with_adc(
            ProviderType::OpenAI,
            "test-project".to_string(),
            "us-east1".to_string(),
        )
        .expect_err("OpenAI is not a Vertex provider");
        assert!(format!("{err}").contains("not a Vertex AI provider"));
    }

    #[test]
    fn test_openai_config_unchanged() {
        let config = ProviderConfig::openai("test-api-key".to_string());

        assert!(matches!(config.provider_type, ProviderType::OpenAI));
        assert_eq!(config.api_key, Some("test-api-key".to_string()));
        assert_eq!(config.project_id, None);
        assert_eq!(config.location, None);
        assert_eq!(config.access_token, None);
    }

    #[test]
    fn test_is_supported_via_vertex() {
        assert!(ProviderType::Google.is_supported_via_vertex());
        assert!(ProviderType::Anthropic.is_supported_via_vertex());
        assert!(!ProviderType::OpenAI.is_supported_via_vertex());
    }
}
