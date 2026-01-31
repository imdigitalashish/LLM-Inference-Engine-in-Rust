use candle_core::{DType, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::error::Result;

pub struct Sampler {
    temperature: f64,
    top_p: f64,
    top_k: usize,
    rng: StdRng,
}

impl Sampler {
    pub fn new(temperature: f64, top_p: f64, top_k: usize, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        Self {
            temperature,
            top_p,
            top_k,
            rng,
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let logits = get_last_logits(logits)?;
        let logits = logits.to_dtype(DType::F32)?;

        let logits = if self.temperature > 0.0 && self.temperature != 1.0 {
            (logits / self.temperature)?
        } else {
            logits
        };

        let logits_vec: Vec<f32> = logits.to_vec1()?;

        if self.temperature == 0.0 {
            let (idx, _) = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            return Ok(idx as u32);
        }

        let mut indexed: Vec<(usize, f32)> =
            logits_vec.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let indexed = if self.top_k > 0 && self.top_k < indexed.len() {
            indexed[..self.top_k].to_vec()
        } else {
            indexed
        };

        let max_logit = indexed[0].1;
        let mut probs: Vec<(usize, f32)> = indexed
            .iter()
            .map(|(i, l)| (*i, (l - max_logit).exp()))
            .collect();

        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in &mut probs {
            *p /= sum;
        }

        let top_p_f32 = self.top_p as f32;
        let probs = if self.top_p < 1.0 {
            let mut cumsum = 0.0f32;
            let cutoff_idx = probs
                .iter()
                .position(|(_, p)| {
                    cumsum += p;
                    cumsum > top_p_f32
                })
                .unwrap_or(probs.len());

            let cutoff_idx = cutoff_idx.max(1);
            probs[..cutoff_idx].to_vec()
        } else {
            probs
        };

        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        let probs: Vec<(usize, f32)> = probs
            .into_iter()
            .map(|(i, p)| (i, p / sum))
            .collect();

        let r: f32 = self.rng.gen();
        let mut cumsum = 0.0;
        for (idx, prob) in probs.iter() {
            cumsum += prob;
            if r < cumsum {
                return Ok(*idx as u32);
            }
        }

        Ok(probs.last().map(|(i, _)| *i as u32).unwrap_or(0))
    }
}

pub fn get_last_logits(logits: &Tensor) -> Result<Tensor> {
    let dims = logits.dims();
    match dims.len() {
        1 => Ok(logits.clone()),
        2 => {
            let last_idx = dims[0] - 1;
            Ok(logits.get(last_idx)?)
        }
        3 => {
            let seq_len = dims[1];
            let logits = logits.get(0)?;
            Ok(logits.get(seq_len - 1)?)
        }
        _ => {
            let mut result = logits.clone();
            while result.dims().len() > 1 {
                if result.dims()[0] == 1 {
                    result = result.squeeze(0)?;
                } else {
                    break;
                }
            }
            Ok(result)
        }
    }
}

pub fn apply_repeat_penalty(
    logits: &Tensor,
    repeat_penalty: f32,
    context: &[u32],
) -> Result<Tensor> {
    if repeat_penalty == 1.0 || context.is_empty() {
        return Ok(logits.clone());
    }

    let original_dims = logits.dims().to_vec();
    let device = logits.device();

    let last_logits = get_last_logits(logits)?;
    let mut logits_vec: Vec<f32> = last_logits.to_dtype(DType::F32)?.to_vec1()?;

    for &token_id in context {
        let idx = token_id as usize;
        if idx < logits_vec.len() {
            let score = logits_vec[idx];
            logits_vec[idx] = if score > 0.0 {
                score / repeat_penalty
            } else {
                score * repeat_penalty
            };
        }
    }

    let vocab_size = logits_vec.len();
    let result = Tensor::from_vec(logits_vec, (vocab_size,), device)?;

    match original_dims.len() {
        1 => Ok(result),
        2 => Ok(result.unsqueeze(0)?),
        3 => Ok(result.unsqueeze(0)?.unsqueeze(0)?),
        _ => Ok(result.unsqueeze(0)?.unsqueeze(0)?),
    }
}
