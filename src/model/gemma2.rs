use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma2::{Config, Model};
use tracing::info;

use super::LanguageModel;
use crate::error::Result;

pub struct Gemma2Model {
    model: Model,
    eos_token_id: u32,
}

impl Gemma2Model {
    pub fn load(
        config_path: &PathBuf,
        weight_paths: &[PathBuf],
        device: &Device,
    ) -> Result<Self> {
        info!("Loading Gemma2 model configuration...");
        let config_str = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;

        // Try to read eos_token_id from config JSON
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
        let eos_token_id = config_json
            .get("eos_token_id")
            .and_then(|v| {
                v.as_u64().or_else(|| v.as_array().and_then(|arr| arr.first()?.as_u64()))
            })
            .unwrap_or(1) as u32;

        info!(
            "Gemma2 config: vocab_size={}, hidden_size={}, num_layers={}, num_heads={}",
            config.vocab_size,
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads
        );

        info!("Loading model weights...");
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(weight_paths, dtype, device)?
        };

        info!("Building Gemma2 model...");
        let model = Model::new(false, &config, vb)?;

        info!("Gemma2 model loaded successfully");
        Ok(Self {
            model,
            eos_token_id,
        })
    }
}

impl LanguageModel for Gemma2Model {
    fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        let logits = self.model.forward(input_ids, position)?;
        Ok(logits)
    }

    fn reset_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    fn vocab_size(&self) -> usize {
        0
    }
}
