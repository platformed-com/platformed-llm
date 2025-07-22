use crate::providers::vertex::{AnthropicProvider, GoogleProvider};
use crate::{Error, LLMProvider, OpenAIProvider};
use std::env;

/// Supported LLM providers.
#[derive(Debug, Clone)]
pub enum ProviderType {
    OpenAI,
    Google,
    Anthropic,
}

/// Configuration for creating providers.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub provider_type: ProviderType,
    pub api_key: Option<String>,
    pub project_id: Option<String>,
    pub location: Option<String>,
    pub model_id: Option<String>,
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
            model_id: None,
            access_token: None,
        }
    }

    /// Create configuration for Google provider with access token.
    pub fn google(
        project_id: String,
        location: String,
        model_id: String,
        access_token: String,
    ) -> Self {
        Self {
            provider_type: ProviderType::Google,
            api_key: None,
            project_id: Some(project_id),
            location: Some(location),
            model_id: Some(model_id),
            access_token: Some(access_token),
        }
    }

    /// Create configuration for Google provider with Application Default Credentials.
    pub fn google_with_adc(project_id: String, location: String, model_id: String) -> Self {
        Self {
            provider_type: ProviderType::Google,
            api_key: None,
            project_id: Some(project_id),
            location: Some(location),
            model_id: Some(model_id),
            access_token: None,
        }
    }

    /// Create configuration for Anthropic provider with access token.
    pub fn anthropic(
        project_id: String,
        location: String,
        model_id: String,
        access_token: String,
    ) -> Self {
        Self {
            provider_type: ProviderType::Anthropic,
            api_key: None,
            project_id: Some(project_id),
            location: Some(location),
            model_id: Some(model_id),
            access_token: Some(access_token),
        }
    }

    /// Create configuration for Anthropic provider with Application Default Credentials.
    pub fn anthropic_with_adc(project_id: String, location: String, model_id: String) -> Self {
        Self {
            provider_type: ProviderType::Anthropic,
            api_key: None,
            project_id: Some(project_id),
            location: Some(location),
            model_id: Some(model_id),
            access_token: None,
        }
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
                    let model_id =
                        env::var("GOOGLE_MODEL").unwrap_or_else(|_| "gemini-1.5-pro".to_string());

                    if let Ok(access_token) = env::var("VERTEX_ACCESS_TOKEN") {
                        return Ok(Self::google(project_id, location, model_id, access_token));
                    } else {
                        return Ok(Self::google_with_adc(project_id, location, model_id));
                    }
                }
                "anthropic" => {
                    let project_id = env::var("GOOGLE_CLOUD_PROJECT")
                        .map_err(|_| Error::config("GOOGLE_CLOUD_PROJECT environment variable is required for Anthropic provider"))?;
                    let location = env::var("GOOGLE_CLOUD_REGION")
                        .unwrap_or_else(|_| "europe-west1".to_string());
                    let model_id = env::var("ANTHROPIC_MODEL")
                        .unwrap_or_else(|_| "claude-3-5-sonnet-v2@20241022".to_string());

                    if let Ok(access_token) = env::var("VERTEX_ACCESS_TOKEN") {
                        return Ok(Self::anthropic(
                            project_id,
                            location,
                            model_id,
                            access_token,
                        ));
                    } else {
                        return Ok(Self::anthropic_with_adc(project_id, location, model_id));
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
            let model_id =
                env::var("GOOGLE_MODEL").unwrap_or_else(|_| "gemini-1.5-pro".to_string());

            return Ok(Self::google(project_id, location, model_id, access_token));
        }

        // Try Anthropic/Vertex with access token
        if let Ok(access_token) = env::var("ANTHROPIC_VERTEX_ACCESS_TOKEN") {
            let project_id = env::var("GOOGLE_CLOUD_PROJECT").map_err(|_| {
                Error::config("GOOGLE_CLOUD_PROJECT environment variable is required for Anthropic")
            })?;
            let location =
                env::var("GOOGLE_CLOUD_REGION").unwrap_or_else(|_| "europe-west1".to_string());
            let model_id = env::var("ANTHROPIC_MODEL")
                .unwrap_or_else(|_| "claude-3-5-sonnet-v2@20241022".to_string());

            return Ok(Self::anthropic(
                project_id,
                location,
                model_id,
                access_token,
            ));
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
                let model_id = env::var("ANTHROPIC_MODEL")
                    .unwrap_or_else(|_| "claude-3-5-sonnet-v2@20241022".to_string());
                return Ok(Self::anthropic_with_adc(project_id, location, model_id));
            } else {
                let model_id =
                    env::var("GOOGLE_MODEL").unwrap_or_else(|_| "gemini-1.5-pro".to_string());
                return Ok(Self::google_with_adc(project_id, location, model_id));
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
                let model_id = config
                    .model_id
                    .as_ref()
                    .ok_or_else(|| Error::config("Model ID required for Google provider"))?;
                // Determine authentication method
                let provider = if let Some(access_token) = &config.access_token {
                    GoogleProvider::new(
                        project_id.clone(),
                        location.clone(),
                        model_id.clone(),
                        access_token.clone(),
                    )?
                } else {
                    // Use Application Default Credentials
                    GoogleProvider::with_adc(project_id.clone(), location.clone(), model_id.clone())
                        .await?
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
                let model_id = config
                    .model_id
                    .as_ref()
                    .ok_or_else(|| Error::config("Model ID required for Anthropic provider"))?;
                // Determine authentication method
                let provider = if let Some(access_token) = &config.access_token {
                    AnthropicProvider::new(
                        project_id.clone(),
                        location.clone(),
                        model_id.clone(),
                        access_token.clone(),
                    )?
                } else {
                    // Use Application Default Credentials
                    AnthropicProvider::with_adc(
                        project_id.clone(),
                        location.clone(),
                        model_id.clone(),
                    )
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
