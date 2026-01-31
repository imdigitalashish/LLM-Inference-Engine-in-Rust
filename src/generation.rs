use std::time::Instant;

use candle_core::{Device, Tensor};
use tracing::{debug, info};

use crate::config::GenerationConfig;
use crate::error::{InferenceError, Result};
use crate::model::LanguageModel;
use crate::sampling::{apply_repeat_penalty, Sampler};
use crate::tokenizer::TokenizerWrapper;

pub struct GenerationOutput {
    pub text: String,
    pub tokens: Vec<u32>,
    pub tokens_per_second: f64,
    pub total_time_ms: u128,
}

pub struct TextGenerator<'a> {
    model: &'a mut dyn LanguageModel,
    tokenizer: &'a TokenizerWrapper,
    device: &'a Device,
}

impl<'a> TextGenerator<'a> {
    pub fn new(
        model: &'a mut dyn LanguageModel,
        tokenizer: &'a TokenizerWrapper,
        device: &'a Device,
    ) -> Self {
        Self {
            model,
            tokenizer,
            device,
        }
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<GenerationOutput> {
        let start_time = Instant::now();

        self.model.reset_cache();

        let prompt_tokens = self.tokenizer.encode(prompt)?;
        let prompt_len = prompt_tokens.len();
        info!("Prompt tokens: {}", prompt_len);

        if prompt_tokens.is_empty() {
            return Err(InferenceError::Inference("Empty prompt".to_string()));
        }

        let mut sampler = Sampler::new(
            config.temperature,
            config.top_p,
            config.top_k,
            config.seed,
        );

        let eos_token = self.tokenizer.eos_token_id().unwrap_or(self.model.eos_token_id());
        // Gemma 3 uses token 106 for <end_of_turn>
        const END_OF_TURN_TOKEN: u32 = 106;

        let mut all_tokens = prompt_tokens.clone();
        let mut generated_tokens: Vec<u32> = Vec::new();

        // Process prompt (prefill)
        let input_tensor = Tensor::new(prompt_tokens.as_slice(), self.device)?
            .unsqueeze(0)?;
        let mut logits = self.model.forward(&input_tensor, 0)?;

        let generation_start = Instant::now();

        // Generate tokens autoregressively
        for i in 0..config.max_tokens {
            // Apply repeat penalty
            let penalized_logits = apply_repeat_penalty(
                &logits,
                config.repeat_penalty,
                &all_tokens,
            )?;

            // Sample next token
            let next_token = sampler.sample(&penalized_logits)?;

            // Check for EOS or end_of_turn token
            if next_token == eos_token || next_token == END_OF_TURN_TOKEN {
                debug!("Stop token {} generated at position {}", next_token, i);
                break;
            }

            // Check for stop sequences
            generated_tokens.push(next_token);
            all_tokens.push(next_token);

            let partial_text = self.tokenizer.decode(&generated_tokens)?;
            if config
                .stop_sequences
                .iter()
                .any(|s| partial_text.contains(s))
            {
                debug!("Stop sequence found at position {}", i);
                break;
            }

            // Forward pass for next token
            let input = Tensor::new(&[next_token], self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&input, prompt_len + i)?;
        }

        let generation_time = generation_start.elapsed();
        let total_time = start_time.elapsed();

        let tokens_per_second = if generation_time.as_secs_f64() > 0.0 {
            generated_tokens.len() as f64 / generation_time.as_secs_f64()
        } else {
            0.0
        };

        let output_text = self.tokenizer.decode(&generated_tokens)?;

        info!(
            "Generated {} tokens in {:?} ({:.2} tokens/sec)",
            generated_tokens.len(),
            generation_time,
            tokens_per_second
        );

        Ok(GenerationOutput {
            text: output_text,
            tokens: generated_tokens,
            tokens_per_second,
            total_time_ms: total_time.as_millis(),
        })
    }

    /// Generate with streaming callback
    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        config: &GenerationConfig,
        mut callback: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(&str) -> bool, // Returns false to stop generation
    {
        let start_time = Instant::now();

        self.model.reset_cache();

        let prompt_tokens = self.tokenizer.encode(prompt)?;
        let prompt_len = prompt_tokens.len();

        if prompt_tokens.is_empty() {
            return Err(InferenceError::Inference("Empty prompt".to_string()));
        }

        let mut sampler = Sampler::new(
            config.temperature,
            config.top_p,
            config.top_k,
            config.seed,
        );

        let eos_token = self.tokenizer.eos_token_id().unwrap_or(self.model.eos_token_id());
        // Gemma 3 uses token 106 for <end_of_turn>
        const END_OF_TURN_TOKEN: u32 = 106;

        let mut all_tokens = prompt_tokens.clone();
        let mut generated_tokens: Vec<u32> = Vec::new();

        let input_tensor = Tensor::new(prompt_tokens.as_slice(), self.device)?
            .unsqueeze(0)?;
        let mut logits = self.model.forward(&input_tensor, 0)?;

        let generation_start = Instant::now();
        let mut prev_text = String::new();

        for i in 0..config.max_tokens {
            let penalized_logits = apply_repeat_penalty(
                &logits,
                config.repeat_penalty,
                &all_tokens,
            )?;

            let next_token = sampler.sample(&penalized_logits)?;

            // Check for EOS or end_of_turn token
            if next_token == eos_token || next_token == END_OF_TURN_TOKEN {
                break;
            }

            generated_tokens.push(next_token);
            all_tokens.push(next_token);

            // Stream using delta decoding (decode all tokens and emit the difference)
            let current_text = self.tokenizer.decode(&generated_tokens)?;
            if current_text.len() > prev_text.len() {
                let new_text = &current_text[prev_text.len()..];
                if !callback(new_text) {
                    break;
                }
            }
            prev_text = current_text.clone();

            if config
                .stop_sequences
                .iter()
                .any(|s| current_text.contains(s))
            {
                break;
            }

            let input = Tensor::new(&[next_token], self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&input, prompt_len + i)?;
        }

        let generation_time = generation_start.elapsed();
        let total_time = start_time.elapsed();

        let tokens_per_second = if generation_time.as_secs_f64() > 0.0 {
            generated_tokens.len() as f64 / generation_time.as_secs_f64()
        } else {
            0.0
        };

        let output_text = self.tokenizer.decode(&generated_tokens)?;

        Ok(GenerationOutput {
            text: output_text,
            tokens: generated_tokens,
            tokens_per_second,
            total_time_ms: total_time.as_millis(),
        })
    }
}
