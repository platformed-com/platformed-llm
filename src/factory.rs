#[cfg(feature = "anthropic-vertex")]
use crate::providers::AnthropicViaVertexProvider;
#[cfg(feature = "google")]
use crate::providers::GoogleProvider;
#[cfg(feature = "openai")]
use crate::providers::OpenAIProvider;
use crate::{Error, Provider};
use std::{env, fmt};

/// Supported LLM providers.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ProviderType {
    /// OpenAI's hosted API (`api.openai.com`).
    OpenAI,
    /// Google Gemini via Vertex AI.
    Google,
    /// Anthropic Claude via Vertex AI.
    Anthropic,
}

impl ProviderType {
    /// Check if this provider type is supported via Vertex AI.
    pub fn is_supported_via_vertex(&self) -> bool {
        matches!(self, ProviderType::Google | ProviderType::Anthropic)
    }
}

/// Configuration for creating providers.
///
/// Fields are public for inspection but the safe way to *construct*
/// values is via [`ProviderConfig::openai`] / [`ProviderConfig::vertex`]
/// / [`ProviderConfig::vertex_with_adc`] — those validate that the
/// credential set matches the provider type. Direct struct literals
/// can build inconsistent states (e.g. `provider_type: OpenAI` paired
/// with `access_token: Some(_)`) which [`ProviderFactory::create`]
/// will then surface as a missing-credential error.
#[derive(Clone)]
pub struct ProviderConfig {
    /// Which backend to instantiate.
    pub provider_type: ProviderType,
    /// API key for direct-API providers (OpenAI).
    pub api_key: Option<String>,
    /// GCP project ID for Vertex providers.
    pub project_id: Option<String>,
    /// GCP region for Vertex providers (e.g. `europe-west1`, `us-east5`).
    pub location: Option<String>,
    /// Pre-fetched OAuth access token for Vertex providers. When absent,
    /// the factory uses Application Default Credentials.
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
    ///
    /// **`PROVIDER_TYPE` is required.** Set it to one of `openai`,
    /// `google`, or `anthropic`. The credential-sniffing fallback that
    /// used to guess from which credential variable happened to be set
    /// has been removed — it was ambiguous when multiple credentials
    /// were present, and silently picked the wrong provider on
    /// dev machines with leftover env state.
    ///
    /// Per-provider env vars:
    /// - **openai**: `OPENAI_API_KEY` (required).
    /// - **google** / **anthropic**: `GOOGLE_CLOUD_PROJECT` (required),
    ///   `GOOGLE_CLOUD_REGION` (default `europe-west1`),
    ///   `VERTEX_ACCESS_TOKEN` (optional — uses ADC when absent).
    pub fn from_env() -> Result<Self, Error> {
        // A var set to an empty/whitespace-only string is as good as
        // unset — reject it here with a clear config error instead of
        // deferring to a confusing provider 401.
        fn required(name: &str) -> Result<String, Error> {
            match env::var(name) {
                Ok(v) if !v.trim().is_empty() => Ok(v),
                _ => Err(Error::config(format!(
                    "{name} environment variable is required and must be non-empty"
                ))),
            }
        }

        let provider_type = required("PROVIDER_TYPE").map_err(|_| {
            Error::config(
                "PROVIDER_TYPE environment variable is required (openai, google, or anthropic)",
            )
        })?;
        match provider_type.to_lowercase().as_str() {
            "openai" => {
                let api_key = required("OPENAI_API_KEY")?;
                Ok(Self::openai(api_key))
            }
            kind @ ("google" | "anthropic") => {
                let provider = if kind == "google" {
                    ProviderType::Google
                } else {
                    ProviderType::Anthropic
                };
                let project_id = required("GOOGLE_CLOUD_PROJECT").map_err(|_| {
                    Error::config(format!(
                        "GOOGLE_CLOUD_PROJECT environment variable is required for {kind} provider"
                    ))
                })?;
                let location = match env::var("GOOGLE_CLOUD_REGION") {
                    Ok(v) if !v.trim().is_empty() => v,
                    _ => "europe-west1".to_string(),
                };
                // An empty VERTEX_ACCESS_TOKEN is treated as absent
                // (fall through to ADC) rather than a blank bearer.
                match env::var("VERTEX_ACCESS_TOKEN") {
                    Ok(token) if !token.trim().is_empty() => {
                        Self::vertex(provider, project_id, location, token)
                    }
                    _ => Self::vertex_with_adc(provider, project_id, location),
                }
            }
            other => Err(Error::config(format!(
                "Invalid PROVIDER_TYPE '{other}'. Valid values are: openai, google, anthropic"
            ))),
        }
    }
}

impl fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            provider_type,
            api_key,
            project_id,
            location,
            access_token,
        } = self;

        f.debug_struct("ProviderConfig")
            .field("provider_type", &provider_type)
            .field("api_key", &api_key.as_ref().map(|_| "[redacted]"))
            .field("project_id", &project_id)
            .field("location", &location)
            .field("access_token", &access_token.as_ref().map(|_| "[redacted]"))
            .finish()
    }
}

/// Factory for creating LLM providers.
pub struct ProviderFactory;

impl ProviderFactory {
    /// Create a provider from configuration.
    ///
    /// Returns `Error::Config` when the requested `provider_type`
    /// targets a backend whose Cargo feature is not enabled in this
    /// build.
    pub async fn create(config: &ProviderConfig) -> Result<Box<dyn Provider>, Error> {
        match config.provider_type {
            #[cfg(feature = "openai")]
            ProviderType::OpenAI => {
                let api_key = config
                    .api_key
                    .as_ref()
                    .ok_or_else(|| Error::config("API key required for OpenAI provider"))?;
                let provider = OpenAIProvider::new(api_key.clone())?;
                Ok(Box::new(provider))
            }
            #[cfg(not(feature = "openai"))]
            ProviderType::OpenAI => Err(Error::config(
                "OpenAI provider is not enabled in this build \
                 (rebuild with `--features openai`)",
            )),

            #[cfg(feature = "google")]
            ProviderType::Google => {
                let project_id = config
                    .project_id
                    .as_ref()
                    .ok_or_else(|| Error::config("Project ID required for Google provider"))?;
                let location = config
                    .location
                    .as_ref()
                    .ok_or_else(|| Error::config("Location required for Google provider"))?;
                let provider = if let Some(access_token) = &config.access_token {
                    GoogleProvider::new(project_id.clone(), location.clone(), access_token.clone())?
                } else {
                    GoogleProvider::with_adc(project_id.clone(), location.clone()).await?
                };
                Ok(Box::new(provider))
            }
            #[cfg(not(feature = "google"))]
            ProviderType::Google => Err(Error::config(
                "Google provider is not enabled in this build \
                 (rebuild with `--features google`)",
            )),

            #[cfg(feature = "anthropic-vertex")]
            ProviderType::Anthropic => {
                let project_id = config
                    .project_id
                    .as_ref()
                    .ok_or_else(|| Error::config("Project ID required for Anthropic provider"))?;
                let location = config
                    .location
                    .as_ref()
                    .ok_or_else(|| Error::config("Location required for Anthropic provider"))?;
                let provider = if let Some(access_token) = &config.access_token {
                    AnthropicViaVertexProvider::new(
                        project_id.clone(),
                        location.clone(),
                        access_token.clone(),
                    )?
                } else {
                    AnthropicViaVertexProvider::with_adc(project_id.clone(), location.clone())
                        .await?
                };
                Ok(Box::new(provider))
            }
            #[cfg(not(feature = "anthropic-vertex"))]
            ProviderType::Anthropic => Err(Error::config(
                "Anthropic provider is not enabled in this build \
                 (rebuild with `--features anthropic-vertex`)",
            )),
        }
    }

    /// Create a provider from environment variables.
    pub async fn from_env() -> Result<Box<dyn Provider>, Error> {
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

    // ---------------------------------------------------------------------
    // `ProviderFactory::create()` construction tests.
    //
    // These confirm the factory wires each config variant to the right
    // concrete provider without making any network calls. We use the
    // explicit-access-token path because it lets the provider be built
    // synchronously (no ADC fetch); the ADC fallback is still exercised
    // by the `*_with_adc_paths` test which expects an error in offline
    // CI environments.
    // ---------------------------------------------------------------------

    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn create_openai_succeeds() {
        let config = ProviderConfig::openai("sk-test".into());
        let provider = ProviderFactory::create(&config)
            .await
            .expect("create openai");
        // We can't inspect the boxed concrete type without downcasting,
        // but reaching `Ok` proves the OpenAI branch wired up.
        drop(provider);
    }

    #[cfg(feature = "google")]
    #[tokio::test]
    async fn create_google_with_access_token_succeeds() {
        let config = ProviderConfig::vertex(
            ProviderType::Google,
            "test-project".into(),
            "us-east1".into(),
            "ya29.token".into(),
        )
        .unwrap();
        let provider = ProviderFactory::create(&config)
            .await
            .expect("create google");
        drop(provider);
    }

    #[cfg(feature = "anthropic-vertex")]
    #[tokio::test]
    async fn create_anthropic_with_access_token_succeeds() {
        let config = ProviderConfig::vertex(
            ProviderType::Anthropic,
            "test-project".into(),
            "europe-west1".into(),
            "ya29.token".into(),
        )
        .unwrap();
        let provider = ProviderFactory::create(&config)
            .await
            .expect("create anthropic");
        drop(provider);
    }

    #[cfg(feature = "openai")]
    #[tokio::test]
    async fn create_openai_without_api_key_errors() {
        let config = ProviderConfig {
            provider_type: ProviderType::OpenAI,
            api_key: None,
            project_id: None,
            location: None,
            access_token: None,
        };
        let err = ProviderFactory::create(&config)
            .await
            .map(|_| ())
            .expect_err("openai needs api_key");
        assert!(err.to_string().contains("API key"), "got: {err}");
    }

    #[cfg(feature = "google")]
    #[tokio::test]
    async fn create_google_without_project_id_errors() {
        let config = ProviderConfig {
            provider_type: ProviderType::Google,
            api_key: None,
            project_id: None,
            location: Some("us-east1".into()),
            access_token: Some("tok".into()),
        };
        let err = ProviderFactory::create(&config)
            .await
            .map(|_| ())
            .expect_err("google needs project_id");
        assert!(err.to_string().contains("Project ID"), "got: {err}");
    }

    #[cfg(feature = "google")]
    #[tokio::test]
    async fn create_google_without_location_errors() {
        let config = ProviderConfig {
            provider_type: ProviderType::Google,
            api_key: None,
            project_id: Some("p".into()),
            location: None,
            access_token: Some("tok".into()),
        };
        let err = ProviderFactory::create(&config)
            .await
            .map(|_| ())
            .expect_err("google needs location");
        assert!(err.to_string().contains("Location"), "got: {err}");
    }

    #[cfg(feature = "anthropic-vertex")]
    #[tokio::test]
    async fn create_anthropic_without_project_id_errors() {
        let config = ProviderConfig {
            provider_type: ProviderType::Anthropic,
            api_key: None,
            project_id: None,
            location: Some("us-east1".into()),
            access_token: Some("tok".into()),
        };
        let err = ProviderFactory::create(&config)
            .await
            .map(|_| ())
            .expect_err("anthropic needs project_id");
        assert!(err.to_string().contains("Project ID"), "got: {err}");
    }

    // ---------------------------------------------------------------------
    // from_env tests
    //
    // Env vars are process-global, so these tests serialize on a single
    // mutex (`ENV_LOCK`). Each test wraps its mutations in `EnvGuard` which
    // snapshots and restores the relevant vars on drop, so an unrelated
    // run-after test isn't poisoned by left-over state.
    //
    // `env::set_var` / `env::remove_var` are `unsafe` since Rust 1.81
    // because the underlying syscall is not thread-safe with concurrent
    // readers. The mutex above gives us that exclusion: while we hold the
    // lock, no other test in this binary is reading env vars via
    // `from_env`. Outside threads (e.g. spawned by tokio runtimes inside
    // tests) reading the same vars *could* race, but `from_env` is sync
    // and runs only on the test thread under the lock, so the
    // SAFETY-requirements are met.
    // ---------------------------------------------------------------------

    use std::sync::Mutex;
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Vars `from_env` reads. Cleared on construction so each test sees a
    /// known empty environment; original values restored on drop.
    const TRACKED: &[&str] = &[
        "PROVIDER_TYPE",
        "OPENAI_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_REGION",
        "VERTEX_ACCESS_TOKEN",
    ];

    struct EnvGuard {
        saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
    }

    impl EnvGuard {
        fn fresh() -> Self {
            let saved: Vec<_> = TRACKED.iter().map(|v| (*v, env::var_os(v))).collect();
            for v in TRACKED {
                // SAFETY: serialized by ENV_LOCK; see module-level note.
                unsafe { env::remove_var(v) };
            }
            Self { saved }
        }

        fn set(&self, key: &str, value: &str) {
            // SAFETY: serialized by ENV_LOCK; see module-level note.
            unsafe { env::set_var(key, value) };
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.saved {
                // SAFETY: serialized by ENV_LOCK; see module-level note.
                unsafe {
                    match v {
                        Some(val) => env::set_var(k, val),
                        None => env::remove_var(k),
                    }
                }
            }
        }
    }

    /// Lock the mutex, recovering from a poisoned lock left by a previously
    /// panicking test so one failure doesn't cascade through the suite.
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[test]
    fn from_env_openai_explicit() {
        let _l = lock();
        let g = EnvGuard::fresh();
        g.set("PROVIDER_TYPE", "openai");
        g.set("OPENAI_API_KEY", "sk-test-key");

        let config = ProviderConfig::from_env().expect("openai config");
        assert!(matches!(config.provider_type, ProviderType::OpenAI));
        assert_eq!(config.api_key, Some("sk-test-key".to_string()));
        assert_eq!(config.project_id, None);
    }

    #[test]
    fn from_env_openai_missing_api_key_errors() {
        let _l = lock();
        let g = EnvGuard::fresh();
        g.set("PROVIDER_TYPE", "openai");

        let err = ProviderConfig::from_env().expect_err("missing key");
        assert!(err.to_string().contains("OPENAI_API_KEY"), "got: {err}");
    }

    #[test]
    fn from_env_google_with_access_token() {
        let _l = lock();
        let g = EnvGuard::fresh();
        g.set("PROVIDER_TYPE", "google");
        g.set("GOOGLE_CLOUD_PROJECT", "proj-1");
        g.set("GOOGLE_CLOUD_REGION", "us-east1");
        g.set("VERTEX_ACCESS_TOKEN", "ya29.tok");

        let config = ProviderConfig::from_env().expect("google config");
        assert!(matches!(config.provider_type, ProviderType::Google));
        assert_eq!(config.project_id, Some("proj-1".to_string()));
        assert_eq!(config.location, Some("us-east1".to_string()));
        assert_eq!(config.access_token, Some("ya29.tok".to_string()));
    }

    #[test]
    fn from_env_google_defaults_region_when_absent() {
        let _l = lock();
        let g = EnvGuard::fresh();
        g.set("PROVIDER_TYPE", "google");
        g.set("GOOGLE_CLOUD_PROJECT", "proj-1");
        g.set("VERTEX_ACCESS_TOKEN", "ya29.tok");

        let config = ProviderConfig::from_env().expect("google config");
        assert_eq!(config.location, Some("europe-west1".to_string()));
    }

    #[test]
    fn from_env_google_falls_back_to_adc_when_no_access_token() {
        let _l = lock();
        let g = EnvGuard::fresh();
        g.set("PROVIDER_TYPE", "google");
        g.set("GOOGLE_CLOUD_PROJECT", "proj-1");

        let config = ProviderConfig::from_env().expect("google adc config");
        assert!(matches!(config.provider_type, ProviderType::Google));
        assert_eq!(config.access_token, None);
        assert_eq!(config.project_id, Some("proj-1".to_string()));
    }

    #[test]
    fn from_env_anthropic_explicit() {
        let _l = lock();
        let g = EnvGuard::fresh();
        g.set("PROVIDER_TYPE", "anthropic");
        g.set("GOOGLE_CLOUD_PROJECT", "proj-1");
        g.set("VERTEX_ACCESS_TOKEN", "ya29.tok");

        let config = ProviderConfig::from_env().expect("anthropic config");
        assert!(matches!(config.provider_type, ProviderType::Anthropic));
        assert_eq!(config.access_token, Some("ya29.tok".to_string()));
    }

    #[test]
    fn from_env_invalid_provider_type_errors() {
        let _l = lock();
        let g = EnvGuard::fresh();
        g.set("PROVIDER_TYPE", "bogus");

        let err = ProviderConfig::from_env().expect_err("invalid provider");
        assert!(
            err.to_string().contains("Invalid PROVIDER_TYPE"),
            "got: {err}"
        );
    }

    #[test]
    fn from_env_without_provider_type_errors() {
        let _l = lock();
        let g = EnvGuard::fresh();
        // Even with a credential set, no PROVIDER_TYPE is an error now —
        // the credential-sniffing fallback has been removed.
        g.set("OPENAI_API_KEY", "sk-fallback");

        let err = ProviderConfig::from_env().expect_err("missing PROVIDER_TYPE");
        assert!(
            err.to_string().contains("PROVIDER_TYPE"),
            "error should mention PROVIDER_TYPE, got: {err}"
        );
    }

    #[test]
    fn from_env_anthropic_via_vertex_access_token() {
        let _l = lock();
        let g = EnvGuard::fresh();
        g.set("PROVIDER_TYPE", "anthropic");
        g.set("GOOGLE_CLOUD_PROJECT", "proj-1");
        g.set("VERTEX_ACCESS_TOKEN", "ya29.anthropic");

        let config = ProviderConfig::from_env().expect("anthropic config");
        assert!(matches!(config.provider_type, ProviderType::Anthropic));
        assert_eq!(config.access_token, Some("ya29.anthropic".to_string()));
    }

    #[test]
    fn from_env_with_nothing_set_errors() {
        let _l = lock();
        let _g = EnvGuard::fresh();
        let err = ProviderConfig::from_env().expect_err("no creds");
        assert!(
            err.to_string().contains("PROVIDER_TYPE"),
            "error should mention PROVIDER_TYPE, got: {err}"
        );
    }

    #[test]
    fn openai_config_debug_redacts_secrets() {
        let config = ProviderConfig::openai("sk-super-secret-123".to_string());
        let log_entry = format!("{:?}", config);
        assert!(
            log_entry.contains(r#"api_key: Some("[redacted]")"#),
            "api_key should be redacted, got: {log_entry}"
        );
        assert!(
            !log_entry.contains("super-secret"),
            "api_key should not leak the supplied key, got: {log_entry}"
        );
        assert!(
            log_entry.contains(r#"access_token: None"#),
            "access_token should be None, got: {log_entry}"
        );
    }

    #[test]
    fn vertex_config_debug_redacts_secrets() -> Result<(), Error> {
        let config = ProviderConfig::vertex(
            ProviderType::Google,
            "123".to_string(),
            "eu1".to_string(),
            "sk-super-secret-456".to_string(),
        )?;
        let log_entry = format!("{:?}", config);
        assert!(
            log_entry.contains(r#"api_key: None"#),
            "api_key should be None, got: {log_entry}"
        );
        assert!(
            !log_entry.contains("super-secret"),
            "access_token should not leak the supplied token, got: {log_entry}"
        );
        assert!(
            log_entry.contains(r#"access_token: Some("[redacted]")"#),
            "access_token should be redacted, got: {log_entry}"
        );

        Ok(())
    }
}
