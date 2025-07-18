use platformed_llm::{OpenAIProvider, InternalRequest, Prompt, Error, CompleteResponse, OutputItem, FinishReason, Usage, Role, FunctionCall};
use platformed_llm::types::{InputItem, Message};

#[tokio::test]
async fn test_openai_provider_creation() {
    let provider = OpenAIProvider::new("test-api-key".to_string());
    assert!(provider.is_ok());
}

#[tokio::test]
async fn test_request_building() {
    let _provider = OpenAIProvider::new("test-api-key".to_string()).unwrap();
    
    let prompt = Prompt::system("You are a helpful assistant")
        .with_user("Hello, world!");
    
    let request = InternalRequest {
        model: "gpt-4".to_string(),
        messages: prompt.items().to_vec(),
        temperature: Some(0.7),
        max_tokens: Some(100),
        top_p: None,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        tools: None,
    };
    
    // We can't actually make an API call without a real API key,
    // but we can test the request building
    assert_eq!(request.model, "gpt-4");
    assert_eq!(request.messages.len(), 2);
    assert_eq!(request.temperature, Some(0.7));
}

#[test]
fn test_prompt_builder() {
    let prompt = Prompt::system("You are a helpful assistant")
        .with_user("What is the capital of France?");
    
    let items = prompt.items();
    assert_eq!(items.len(), 2);
    
    // Test From implementations
    let prompt_from_str: Prompt = "Hello".into();
    assert_eq!(prompt_from_str.items().len(), 1);
    
    let prompt_from_string: Prompt = "Hello".to_string().into();
    assert_eq!(prompt_from_string.items().len(), 1);
    
    // Test with_response method
    let response = CompleteResponse {
        output: vec![OutputItem::Text { content: "AI response".to_string() }],
        finish_reason: FinishReason::Stop,
        usage: Usage::default(),
    };
    
    let prompt_with_response = prompt.with_response(&response);
    assert_eq!(prompt_with_response.items().len(), 3); // Original 2 + 1 response
    
    // Check that the added item is an assistant message
    let last_item = &prompt_with_response.items()[2];
    match last_item {
        InputItem::Message(msg) => {
            assert_eq!(msg.role(), Role::Assistant);
            assert_eq!(msg.content(), Some("AI response".to_string()));
        }
        _ => panic!("Expected message"),
    }
    
    // Test with_items method
    let input_items = vec![
        InputItem::Message(Message::assistant("First assistant message")),
        InputItem::FunctionCall(FunctionCall {
            id: "fc_123".to_string(),
            call_id: "call_123".to_string(),
            name: "test_function".to_string(),
            arguments: "{}".to_string(),
        }),
        InputItem::Message(Message::assistant("Second assistant message")),
    ];
    
    let original_prompt = Prompt::system("You are a helpful assistant")
        .with_user("Hello world");
    let prompt_with_items = original_prompt.with_items(input_items);
    assert_eq!(prompt_with_items.items().len(), 5); // Original 2 + 3 items
    
    // Check the added items
    let added_items = &prompt_with_items.items()[2..];
    match &added_items[0] {
        InputItem::Message(msg) => {
            assert_eq!(msg.role(), Role::Assistant);
            assert_eq!(msg.content(), Some("First assistant message".to_string()));
        }
        _ => panic!("Expected message"),
    }
    
    match &added_items[1] {
        InputItem::FunctionCall(call) => {
            assert_eq!(call.name, "test_function");
        }
        _ => panic!("Expected function call"),
    }
    
    match &added_items[2] {
        InputItem::Message(msg) => {
            assert_eq!(msg.role(), Role::Assistant);
            assert_eq!(msg.content(), Some("Second assistant message".to_string()));
        }
        _ => panic!("Expected message"),
    }
}

#[test]
fn test_error_creation() {
    let error = Error::provider("OpenAI", "Test error");
    assert!(error.to_string().contains("OpenAI"));
    assert!(error.to_string().contains("Test error"));
    
    let config_error = Error::config("Invalid model name");
    assert!(config_error.to_string().contains("Invalid configuration"));
}