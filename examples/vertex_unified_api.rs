use platformed_llm::{Error, ProviderConfig, ProviderType};

#[tokio::main]
async fn main() -> Result<(), Error> {
    dotenvy::dotenv().ok();

    // Vertex AI providers (Google Gemini, Anthropic Claude) — explicit
    // provider_type selection through the unified factory.
    let google_config = ProviderConfig::vertex(
        ProviderType::Google,
        "my-project".to_string(),
        "us-central1".to_string(),
        "fake-access-token".to_string(),
    )?;
    println!("Google config: {:?}", google_config.provider_type);

    let anthropic_config = ProviderConfig::vertex(
        ProviderType::Anthropic,
        "my-project".to_string(),
        "us-east5".to_string(),
        "fake-access-token".to_string(),
    )?;
    println!("Anthropic config: {:?}", anthropic_config.provider_type);

    // Application Default Credentials variant.
    let google_adc = ProviderConfig::vertex_with_adc(
        ProviderType::Google,
        "my-project".to_string(),
        "europe-west1".to_string(),
    )?;
    println!(
        "Google (ADC): {:?} in {:?}",
        google_adc.provider_type, google_adc.location
    );

    // Passing a non-Vertex provider type returns Err rather than panicking.
    let err = ProviderConfig::vertex(
        ProviderType::OpenAI,
        "my-project".to_string(),
        "us-central1".to_string(),
        "tok".to_string(),
    )
    .unwrap_err();
    println!("OpenAI is rejected: {err}");

    Ok(())
}
