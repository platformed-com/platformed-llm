use std::borrow::Cow;

use crate::types::Usage;
use ijson::IValue;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

/// Google Vertex AI request format (for Gemini models).
#[derive(Debug, Clone, Serialize)]
pub struct GoogleRequest {
    pub contents: Vec<GoogleContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GoogleGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GoogleTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GoogleContent>,
}

/// Google content (message) format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleContent {
    pub role: String, // "user", "model"
    pub parts: Vec<GooglePart>,
}

/// Part of a Google content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GooglePart {
    Text {
        text: String,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GoogleFunctionCall,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GoogleFunctionResponse,
    },
}

/// Google function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleFunctionCall {
    pub name: String,
    pub args: IValue,
}

/// Google function response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleFunctionResponse {
    pub name: String,
    pub response: IValue,
}

/// Google generation configuration.
#[derive(Debug, Clone, Serialize)]
pub struct GoogleGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

/// Google tool definition.
#[derive(Debug, Clone, Serialize)]
pub struct GoogleTool {
    pub function_declarations: Vec<GoogleFunctionDeclaration>,
}

/// Google function declaration.
#[derive(Debug, Clone, Serialize)]
pub struct GoogleFunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: Cow<'static, RawValue>,
}

/// Google API response.
#[derive(Debug, Clone, Deserialize)]
pub struct GoogleResponse {
    pub candidates: Vec<GoogleCandidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<GoogleUsageMetadata>,
}

/// Google response candidate.
#[derive(Debug, Clone, Deserialize)]
pub struct GoogleCandidate {
    pub content: GoogleContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "finishReason")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "safetyRatings")]
    pub safety_ratings: Option<Vec<IValue>>,
}

/// Google usage metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct GoogleUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    pub prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    pub candidates_token_count: Option<u32>,
    #[serde(rename = "totalTokenCount")]
    pub total_token_count: Option<u32>,
}

impl From<GoogleUsageMetadata> for Usage {
    fn from(metadata: GoogleUsageMetadata) -> Self {
        Usage {
            input_tokens: metadata.prompt_token_count.unwrap_or(0),
            output_tokens: metadata.candidates_token_count.unwrap_or(0),
            cached_tokens: None, // Google doesn't provide cached token info
        }
    }
}
