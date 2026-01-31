use std::path::PathBuf;

use hf_hub::{api::sync::Api, api::sync::ApiBuilder, Repo, RepoType};
use tracing::info;

use crate::error::{InferenceError, Result};

pub struct ModelFiles {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: Vec<PathBuf>,
}

/// Load model files from HuggingFace Hub
pub fn load_model(
    model_id: &str,
    revision: &str,
    token: Option<&str>,
) -> Result<ModelFiles> {
    info!("Loading model: {} (revision: {})", model_id, revision);

    let api = match token {
        Some(t) => ApiBuilder::new()
            .with_token(Some(t.to_string()))
            .build()
            .map_err(|e| InferenceError::Hub(e.to_string()))?,
        None => Api::new().map_err(|e| InferenceError::Hub(e.to_string()))?,
    };

    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    // Download config.json
    info!("Downloading config.json...");
    let config = repo
        .get("config.json")
        .map_err(|e| InferenceError::Hub(format!("Failed to download config.json: {}", e)))?;

    // Download tokenizer files
    info!("Downloading tokenizer...");
    let tokenizer = repo
        .get("tokenizer.json")
        .map_err(|e| InferenceError::Hub(format!("Failed to download tokenizer.json: {}", e)))?;

    // Try to find weight files - prefer safetensors
    info!("Downloading model weights...");
    let weights = download_weights(&repo)?;

    info!(
        "Model files downloaded: config={:?}, tokenizer={:?}, weights={} files",
        config,
        tokenizer,
        weights.len()
    );

    Ok(ModelFiles {
        config,
        tokenizer,
        weights,
    })
}

fn download_weights(repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<PathBuf>> {
    // Try single safetensors file first
    if let Ok(path) = repo.get("model.safetensors") {
        return Ok(vec![path]);
    }

    // Try sharded safetensors
    let mut weights = Vec::new();
    for i in 1..=100 {
        // Try common patterns for sharded models
        for total in [2, 3, 4, 5, 8, 10, 19, 37] {
            let filename = format!("model-{:05}-of-{:05}.safetensors", i, total);
            if let Ok(path) = repo.get(&filename) {
                weights.push(path);
                break;
            }
        }
    }

    if !weights.is_empty() {
        return Ok(weights);
    }

    // Try pytorch_model.bin as fallback
    if let Ok(path) = repo.get("pytorch_model.bin") {
        return Ok(vec![path]);
    }

    Err(InferenceError::Hub(
        "Could not find model weights (tried model.safetensors, sharded safetensors, pytorch_model.bin)".to_string()
    ))
}

