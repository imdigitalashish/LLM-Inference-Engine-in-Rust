mod gemma;
mod gemma2;
mod gemma3;
mod loader;
mod mistral;

pub use gemma::GemmaModel;
pub use gemma2::Gemma2Model;
pub use gemma3::Gemma3Model;
pub use loader::load_model;
pub use mistral::MistralModel;

use candle_core::Tensor;
use crate::error::Result;

/// Trait for language models that can generate next token logits
pub trait LanguageModel: Send + Sync {
    /// Forward pass to get logits for the next token
    fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor>;

    /// Reset the model's KV cache
    fn reset_cache(&mut self);

    /// Get the end-of-sequence token ID
    fn eos_token_id(&self) -> u32;

    /// Get the model's vocabulary size
    fn vocab_size(&self) -> usize;
}

/// Supported model architectures
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelArchitecture {
    Mistral,
    Gemma,
    Gemma2,
    Gemma3,
}

/// Detect model architecture from config.json
pub fn detect_architecture(config_path: &std::path::Path) -> Result<ModelArchitecture> {
    let config_str = std::fs::read_to_string(config_path)?;
    let config: serde_json::Value = serde_json::from_str(&config_str)?;

    // Check architectures field
    if let Some(archs) = config.get("architectures").and_then(|v| v.as_array()) {
        for arch in archs {
            if let Some(arch_str) = arch.as_str() {
                let arch_lower = arch_str.to_lowercase();
                if arch_lower.contains("gemma3") {
                    return Ok(ModelArchitecture::Gemma3);
                } else if arch_lower.contains("gemma2") {
                    return Ok(ModelArchitecture::Gemma2);
                } else if arch_lower.contains("gemma") {
                    return Ok(ModelArchitecture::Gemma);
                } else if arch_lower.contains("mistral") {
                    return Ok(ModelArchitecture::Mistral);
                }
            }
        }
    }

    // Check model_type field
    if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
        let model_type_lower = model_type.to_lowercase();
        if model_type_lower.contains("gemma3") {
            return Ok(ModelArchitecture::Gemma3);
        } else if model_type_lower.contains("gemma2") {
            return Ok(ModelArchitecture::Gemma2);
        } else if model_type_lower.contains("gemma") {
            return Ok(ModelArchitecture::Gemma);
        } else if model_type_lower.contains("mistral") {
            return Ok(ModelArchitecture::Mistral);
        }
    }

    // Default to Mistral
    Ok(ModelArchitecture::Mistral)
}
