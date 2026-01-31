mod routes;
mod types;

pub use routes::create_router;

use std::sync::Arc;
use tokio::sync::Mutex;

use crate::model::LanguageModel;
use crate::tokenizer::TokenizerWrapper;
use candle_core::Device;

/// Shared application state
pub struct AppState {
    pub model: Arc<Mutex<Box<dyn LanguageModel>>>,
    pub tokenizer: Arc<TokenizerWrapper>,
    pub device: Device,
    pub model_id: String,
}
