//! End-to-end tests for HTTP error mapping.
//!
//! `parse_openai_error` has unit tests, but those don't exercise the path
//! from a real HTTP response (status, headers, body) through
//! `OpenAIProvider::generate()` to a typed [`Error`]. The wiring on
//! `client.rs:352-361` reads `retry-after`, drains the body, and calls the
//! parser — that whole pipeline is untested. This file fills that gap with
//! `wiremock`.

use platformed_llm::{Error, LLMProvider, LLMRequest, OpenAIProvider, Prompt};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn openai_against<F>(setup: F) -> Result<(), Error>
where
    F: FnOnce(ResponseTemplate) -> ResponseTemplate,
{
    let server = MockServer::start().await;
    let template = setup(ResponseTemplate::new(200));
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(template)
        .mount(&server)
        .await;

    let provider =
        OpenAIProvider::new_with_base_url("test-key".to_string(), server.uri()).unwrap();
    let req = LLMRequest::from_prompt("gpt-4o-mini", &Prompt::user("hi"));
    provider.generate(&req).await.map(|_| ())
}

#[tokio::test]
async fn http_429_with_retry_after_surfaces_as_rate_limit() {
    let body = r#"{"error":{"message":"Rate limited","type":"rate_limit_error","code":"rate_limit_exceeded"}}"#;
    let err = openai_against(|_| {
        ResponseTemplate::new(429)
            .insert_header("retry-after", "42")
            .insert_header("content-type", "application/json")
            .set_body_string(body)
    })
    .await
    .expect_err("429 must produce an error");

    match err {
        Error::RateLimit {
            retry_after_seconds,
            message,
        } => {
            assert_eq!(retry_after_seconds, Some(42));
            assert!(
                message.contains("Rate limited"),
                "message should contain provider text, got: {message}",
            );
        }
        other => panic!("expected RateLimit, got {other:?}"),
    }
}

#[tokio::test]
async fn http_429_without_retry_after_still_maps_to_rate_limit() {
    let body = r#"{"error":{"message":"Slow down","type":"rate_limit_error"}}"#;
    let err = openai_against(|_| ResponseTemplate::new(429).set_body_string(body))
        .await
        .expect_err("429 must error");

    match err {
        Error::RateLimit {
            retry_after_seconds,
            ..
        } => assert_eq!(retry_after_seconds, None),
        other => panic!("expected RateLimit, got {other:?}"),
    }
}

#[tokio::test]
async fn http_401_surfaces_as_auth() {
    let body = r#"{"error":{"message":"Bad key","type":"invalid_request_error","code":"invalid_api_key"}}"#;
    let err = openai_against(|_| ResponseTemplate::new(401).set_body_string(body))
        .await
        .expect_err("401 must error");

    match err {
        Error::Auth(message) => assert!(
            message.contains("Bad key"),
            "auth message lost provider text: {message}",
        ),
        other => panic!("expected Auth, got {other:?}"),
    }
}

#[tokio::test]
async fn http_500_surfaces_as_provider() {
    let body = r#"{"error":{"message":"boom","type":"server_error"}}"#;
    let err = openai_against(|_| ResponseTemplate::new(500).set_body_string(body))
        .await
        .expect_err("500 must error");

    match err {
        Error::Provider { provider, message } => {
            assert_eq!(provider, "OpenAI");
            assert!(message.contains("500"), "should mention status: {message}");
            assert!(message.contains("boom"), "should mention body: {message}");
        }
        other => panic!("expected Provider, got {other:?}"),
    }
}

/// Even when the body isn't JSON we should still produce a typed error and
/// preserve the raw text. Catches a regression where a malformed body would
/// panic the deserializer or get swallowed.
#[tokio::test]
async fn non_json_error_body_is_preserved() {
    let err = openai_against(|_| {
        ResponseTemplate::new(503).set_body_string("upstream proxy timeout")
    })
    .await
    .expect_err("503 must error");

    match err {
        Error::Provider { provider, message } => {
            assert_eq!(provider, "OpenAI");
            assert!(message.contains("upstream proxy timeout"), "got: {message}");
        }
        other => panic!("expected Provider, got {other:?}"),
    }
}
