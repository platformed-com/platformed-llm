use platformed_llm::{Error, ProviderConfig, ProviderType};

#[tokio::main]
async fn main() -> Result<(), Error> {
    dotenv::dotenv().ok();

    println!("🔍 Vertex Unified API Example");
    println!("Demonstrating the new unified vertex() constructors\n");

    // Example 1: Using the unified vertex() constructor with explicit provider type
    println!("📋 Example 1: Unified vertex() constructor");

    let google_config = ProviderConfig::vertex(
        ProviderType::Google,
        "my-project".to_string(),
        "us-central1".to_string(),
        "fake-access-token".to_string(),
    );

    let anthropic_config = ProviderConfig::vertex(
        ProviderType::Anthropic,
        "my-project".to_string(),
        "us-east5".to_string(),
        "fake-access-token".to_string(),
    );

    println!("✅ Google config: {:?}", google_config.provider_type);
    println!("✅ Anthropic config: {:?}", anthropic_config.provider_type);

    // Example 2: Using convenience methods
    println!("\n📋 Example 2: Convenience methods");

    let google_convenience = ProviderConfig::vertex_with_adc(
        ProviderType::Google,
        "my-project".to_string(),
        "europe-west1".to_string(),
    );

    let anthropic_convenience = ProviderConfig::vertex(
        ProviderType::Anthropic,
        "my-project".to_string(),
        "us-east5".to_string(),
        "fake-access-token".to_string(),
    );

    println!(
        "✅ Google (with ADC): {:?} in {:?}",
        google_convenience.provider_type, google_convenience.location
    );
    println!(
        "✅ Anthropic (with token): {:?} in {:?}",
        anthropic_convenience.provider_type, anthropic_convenience.location
    );

    // Example 3: Logic error protection - trying to use OpenAI with vertex() panics
    println!("\n📋 Example 3: Logic error protection (normally panics)");

    println!("✅ vertex() with OpenAI would panic - this is intentional!");
    println!("   Using panic! ensures logic errors are caught at development time");
    println!("   Only Google and Anthropic provider types are supported with vertex()");

    // Example 4: Different authentication methods
    println!("\n📋 Example 4: Authentication methods");

    let with_token = ProviderConfig::vertex(
        ProviderType::Google,
        "my-project".to_string(),
        "us-central1".to_string(),
        "access-token".to_string(),
    );

    let with_adc = ProviderConfig::vertex_with_adc(
        ProviderType::Google,
        "my-project".to_string(),
        "us-central1".to_string(),
    );

    println!(
        "✅ Access token: provider={:?}, has_token={}",
        with_token.provider_type,
        with_token.access_token.is_some()
    );
    println!(
        "✅ ADC: provider={:?}, has_token={}",
        with_adc.provider_type,
        with_adc.access_token.is_some()
    );

    println!("\n🎉 All examples completed successfully!");
    println!("\n💡 Benefits of the unified API:");
    println!("   - Only vertex() and vertex_with_adc() methods needed");
    println!("   - Explicit provider type selection");
    println!("   - Panics on logic errors (compile-time safety)");
    println!("   - Clean, minimal API surface");

    Ok(())
}
