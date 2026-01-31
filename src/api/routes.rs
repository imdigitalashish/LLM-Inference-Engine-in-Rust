use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use tracing::{error, info};
use uuid::Uuid;

use super::types::*;
use super::AppState;
use crate::config::GenerationConfig;
use crate::device::device_info;
use crate::generation::TextGenerator;
use crate::model::LanguageModel;

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state)
}

async fn health_check(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: state.model_id.clone(),
        device: device_info(&state.device),
    })
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_id.clone(),
            object: "model".to_string(),
            created,
            owned_by: "local".to_string(),
        }],
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let request_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    info!(
        "Chat completion request: {} messages, max_tokens={}",
        request.messages.len(),
        request.max_tokens
    );

    // Format messages into a prompt
    let prompt = format_chat_prompt(&request.messages);

    let config = GenerationConfig {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: 40,
        repeat_penalty: request.frequency_penalty.max(1.0),
        seed: None,
        stop_sequences: request.stop.unwrap_or_default(),
    };

    // Non-streaming response
    let mut model = state.model.lock().await;
    let mut generator = TextGenerator::new(
        model.as_mut(),
        state.tokenizer.as_ref(),
        &state.device,
    );

    match generator.generate(&prompt, &config) {
        Ok(output) => {
            let response = ChatCompletionResponse {
                id: request_id,
                object: "chat.completion".to_string(),
                created,
                model: state.model_id.clone(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: output.text,
                    },
                    finish_reason: "stop".to_string(),
                }],
                usage: Usage {
                    prompt_tokens: state.tokenizer.encode(&prompt).unwrap_or_default().len(),
                    completion_tokens: output.tokens.len(),
                    total_tokens: state.tokenizer.encode(&prompt).unwrap_or_default().len()
                        + output.tokens.len(),
                },
            };
            Json(response).into_response()
        }
        Err(e) => {
            error!("Generation error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: e.to_string(),
                        r#type: "server_error".to_string(),
                        code: None,
                    },
                }),
            )
                .into_response()
        }
    }
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> Response {
    let request_id = format!("cmpl-{}", Uuid::new_v4());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    info!(
        "Completion request: prompt_len={}, max_tokens={}",
        request.prompt.len(),
        request.max_tokens
    );

    let config = GenerationConfig {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: 40,
        repeat_penalty: 1.1,
        seed: None,
        stop_sequences: request.stop.unwrap_or_default(),
    };

    let mut model = state.model.lock().await;
    let mut generator = TextGenerator::new(
        model.as_mut(),
        state.tokenizer.as_ref(),
        &state.device,
    );

    match generator.generate(&request.prompt, &config) {
        Ok(output) => {
            let response = CompletionResponse {
                id: request_id,
                object: "text_completion".to_string(),
                created,
                model: state.model_id.clone(),
                choices: vec![CompletionChoice {
                    index: 0,
                    text: output.text,
                    finish_reason: "stop".to_string(),
                }],
                usage: Usage {
                    prompt_tokens: state.tokenizer.encode(&request.prompt).unwrap_or_default().len(),
                    completion_tokens: output.tokens.len(),
                    total_tokens: state.tokenizer.encode(&request.prompt).unwrap_or_default().len()
                        + output.tokens.len(),
                },
            };
            Json(response).into_response()
        }
        Err(e) => {
            error!("Generation error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: e.to_string(),
                        r#type: "server_error".to_string(),
                        code: None,
                    },
                }),
            )
                .into_response()
        }
    }
}

fn format_chat_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                // Gemma 3 format
                prompt.push_str(&format!("<start_of_turn>user\n{}<end_of_turn>\n", msg.content));
            }
            "user" => {
                prompt.push_str(&format!("<start_of_turn>user\n{}<end_of_turn>\n", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!("<start_of_turn>model\n{}<end_of_turn>\n", msg.content));
            }
            _ => {
                prompt.push_str(&format!("{}\n", msg.content));
            }
        }
    }

    // Add model turn start for generation
    prompt.push_str("<start_of_turn>model\n");
    prompt
}
