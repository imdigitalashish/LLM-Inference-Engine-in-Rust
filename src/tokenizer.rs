use std::path::PathBuf;

use tokenizers::Tokenizer;
use tracing::info;

use crate::error::{InferenceError, Result};

pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
}

impl TokenizerWrapper {
    pub fn load(path: &PathBuf) -> Result<Self> {
        info!("Loading tokenizer from {:?}", path);
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| InferenceError::Tokenizer(e.to_string()))?;

        info!("Tokenizer loaded with {} tokens", tokenizer.get_vocab_size(true));
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| InferenceError::Tokenizer(e.to_string()))?;

        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| InferenceError::Tokenizer(e.to_string()))
    }

    pub fn decode_single(&self, id: u32) -> Result<String> {
        self.tokenizer
            .decode(&[id], false)
            .map_err(|e| InferenceError::Tokenizer(e.to_string()))
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn get_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.tokenizer
            .token_to_id("</s>")
            .or_else(|| self.tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| self.tokenizer.token_to_id("<eos>"))
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.tokenizer
            .token_to_id("<s>")
            .or_else(|| self.tokenizer.token_to_id("<|startoftext|>"))
            .or_else(|| self.tokenizer.token_to_id("<bos>"))
    }
}
