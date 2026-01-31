use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Module, Result as CandleResult, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};

// Custom RMSNorm for Gemma 3 that uses (1 + weight) instead of just weight
struct Gemma3RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Gemma3RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> CandleResult<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;

        // Compute RMS: sqrt(mean(x^2) + eps)
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let rms = (variance + self.eps)?.sqrt()?;
        let x_normed = x.broadcast_div(&rms)?;

        // Gemma uses (1 + weight) instead of just weight
        let weight = (&self.weight.to_dtype(DType::F32)? + 1.0)?;
        let result = x_normed.broadcast_mul(&weight)?;

        result.to_dtype(dtype)
    }
}
use serde::Deserialize;
use tracing::info;

use super::LanguageModel;
use crate::error::Result;

// Gemma 3 Configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rope_local_base_freq")]
    pub rope_local_base_freq: f64,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_query_pre_attn_scalar")]
    pub query_pre_attn_scalar: usize,
}

fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_rope_theta() -> f64 { 1_000_000.0 }
fn default_rope_local_base_freq() -> f64 { 10_000.0 }
fn default_sliding_window() -> usize { 1024 }
fn default_max_position_embeddings() -> usize { 32768 }
fn default_query_pre_attn_scalar() -> usize { 256 }

impl Gemma3Config {
    /// Returns true if this layer should use global attention (every 6th layer)
    fn is_global_layer(&self, layer_idx: usize) -> bool {
        // Pattern: 5 local, 1 global (layers 5, 11, 17, 23, ...)
        (layer_idx + 1) % 6 == 0
    }

    fn rope_theta_for_layer(&self, layer_idx: usize) -> f64 {
        if self.is_global_layer(layer_idx) {
            self.rope_theta // 1M for global
        } else {
            self.rope_local_base_freq // 10k for local
        }
    }
}

// Rotary Position Embedding (HuggingFace-style split-half method)
struct RotaryEmbedding {
    sin: Tensor,  // [max_seq_len, dim/2]
    cos: Tensor,  // [max_seq_len, dim/2]
}

impl RotaryEmbedding {
    fn new(dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> CandleResult<Self> {
        let half_dim = dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq, device)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|x| x as f32).collect();
        let positions = Tensor::new(positions, device)?.unsqueeze(1)?;

        let freqs = positions.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;

        Ok(Self { sin, cos })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> CandleResult<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let sin = self.sin.i(offset..offset + seq_len)?;
        let cos = self.cos.i(offset..offset + seq_len)?;

        let q_embed = Self::apply_rotary_emb(q, &sin, &cos)?;
        let k_embed = Self::apply_rotary_emb(k, &sin, &cos)?;

        Ok((q_embed, k_embed))
    }

    /// HuggingFace-style RoPE: split-half method
    /// rotate_half(x) = cat(-x[..., dim/2:], x[..., :dim/2])
    /// output = x * cos + rotate_half(x) * sin
    fn apply_rotary_emb(x: &Tensor, sin: &Tensor, cos: &Tensor) -> CandleResult<Tensor> {
        let (b, h, seq_len, dim) = x.dims4()?;
        let half_dim = dim / 2;

        // Split x into first and second half
        let x1 = x.narrow(D::Minus1, 0, half_dim)?;
        let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;

        // rotate_half: [-x2, x1]
        let x2_neg = x2.neg()?;
        let x_rotated = Tensor::cat(&[x2_neg, x1], D::Minus1)?;

        // Expand sin/cos to full dimension: [seq, dim/2] -> [1, 1, seq, dim]
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;  // [1, 1, seq, dim/2]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;

        // Duplicate sin/cos to match full dim: [1, 1, seq, dim/2] -> [1, 1, seq, dim]
        let sin_full = Tensor::cat(&[&sin, &sin], D::Minus1)?;
        let cos_full = Tensor::cat(&[&cos, &cos], D::Minus1)?;

        // Apply: x * cos + rotate_half(x) * sin
        let result = (x.broadcast_mul(&cos_full)? + x_rotated.broadcast_mul(&sin_full)?)?;
        Ok(result)
    }
}

// MLP with GELU activation
struct Gemma3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Gemma3MLP {
    fn new(config: &Gemma3Config, vb: VarBuilder) -> CandleResult<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self { gate_proj, up_proj, down_proj })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = gate.gelu_erf()?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden)
    }
}

// Attention with QK-Norm
struct Gemma3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Gemma3RmsNorm,
    k_norm: Gemma3RmsNorm,
    rotary: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    is_global: bool,
    sliding_window: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Gemma3Attention {
    fn new(config: &Gemma3Config, layer_idx: usize, vb: VarBuilder, device: &Device) -> CandleResult<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        let q_norm = Gemma3RmsNorm::new(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = Gemma3RmsNorm::new(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?;

        let is_global = config.is_global_layer(layer_idx);
        let rope_theta = config.rope_theta_for_layer(layer_idx);
        let rotary = RotaryEmbedding::new(head_dim, config.max_position_embeddings, rope_theta, device)?;

        let scale = 1.0 / (config.query_pre_attn_scalar as f64).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            is_global,
            sliding_window: config.sliding_window,
            kv_cache: None,
        })
    }

    fn forward(&mut self, x: &Tensor, position: usize) -> CandleResult<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [batch, heads, seq, head_dim]
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply QK-Norm (per-head normalization)
        let q = self.apply_qk_norm(&q, &self.q_norm)?;
        let k = self.apply_qk_norm(&k, &self.k_norm)?;

        // Apply rotary embeddings
        let (q, k) = self.rotary.apply(&q, &k, position)?;

        // Update KV cache
        let (k, v) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Expand KV heads for GQA
        let k = self.repeat_kv(&k)?;
        let v = self.repeat_kv(&v)?;

        // Compute attention scores
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let scores = (scores * self.scale)?;

        // Apply causal mask (and sliding window for local attention)
        let scores = self.apply_mask(&scores)?;

        // Softmax and attention output
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }

    fn apply_qk_norm(&self, x: &Tensor, norm: &Gemma3RmsNorm) -> CandleResult<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        let x = x.reshape((b * h * s, d))?;
        let x = norm.forward(&x)?;
        x.reshape((b, h, s, d))
    }

    fn repeat_kv(&self, x: &Tensor) -> CandleResult<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let (b, h, s, d) = x.dims4()?;
        let x = x.unsqueeze(2)?;
        let x = x.expand((b, h, n_rep, s, d))?;
        x.reshape((b, h * n_rep, s, d))
    }

    fn apply_mask(&self, scores: &Tensor) -> CandleResult<Tensor> {
        let (_, _, seq_len, kv_len) = scores.dims4()?;
        let device = scores.device();
        let dtype = scores.dtype();

        let mut mask = vec![vec![0f32; kv_len]; seq_len];
        for i in 0..seq_len {
            for j in 0..kv_len {
                let query_pos = i;
                let key_pos = j;

                // Causal: can't attend to future
                if key_pos > query_pos + (kv_len - seq_len) {
                    mask[i][j] = f32::NEG_INFINITY;
                }
                // Sliding window for local attention
                else if !self.is_global {
                    let distance = (query_pos + (kv_len - seq_len)) as i64 - key_pos as i64;
                    if distance > self.sliding_window as i64 {
                        mask[i][j] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        let mask: Vec<f32> = mask.into_iter().flatten().collect();
        let mask = Tensor::from_vec(mask, (1, 1, seq_len, kv_len), device)?
            .to_dtype(dtype)?;

        scores.broadcast_add(&mask)
    }

    fn clear_cache(&mut self) {
        self.kv_cache = None;
    }
}

// Transformer layer
struct Gemma3DecoderLayer {
    self_attn: Gemma3Attention,
    mlp: Gemma3MLP,
    input_layernorm: Gemma3RmsNorm,
    post_attention_layernorm: Gemma3RmsNorm,
    pre_feedforward_layernorm: Gemma3RmsNorm,
    post_feedforward_layernorm: Gemma3RmsNorm,
}

impl Gemma3DecoderLayer {
    fn new(config: &Gemma3Config, layer_idx: usize, vb: VarBuilder, device: &Device) -> CandleResult<Self> {
        let self_attn = Gemma3Attention::new(config, layer_idx, vb.pp("self_attn"), device)?;
        let mlp = Gemma3MLP::new(config, vb.pp("mlp"))?;

        let input_layernorm = Gemma3RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = Gemma3RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        let pre_feedforward_layernorm = Gemma3RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("pre_feedforward_layernorm"))?;
        let post_feedforward_layernorm = Gemma3RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("post_feedforward_layernorm"))?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    fn forward(&mut self, x: &Tensor, position: usize, _layer_idx: usize) -> CandleResult<Tensor> {
        // Pre-norm attention
        let residual = x;
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, position)?;
        let attn_normed = self.post_attention_layernorm.forward(&attn_out)?;
        let hidden_after_attn = (residual + attn_normed)?;

        // Pre-norm MLP
        let residual = &hidden_after_attn;
        let ff_normed = self.pre_feedforward_layernorm.forward(&hidden_after_attn)?;
        let mlp_out = self.mlp.forward(&ff_normed)?;
        let mlp_normed = self.post_feedforward_layernorm.forward(&mlp_out)?;

        residual + mlp_normed
    }

    fn clear_cache(&mut self) {
        self.self_attn.clear_cache();
    }
}

// Full Gemma 3 Model
pub struct Gemma3Model {
    embed_tokens: Embedding,
    layers: Vec<Gemma3DecoderLayer>,
    norm: Gemma3RmsNorm,
    lm_head: Linear,
    config: Gemma3Config,
    eos_token_id: u32,
}

impl Gemma3Model {
    pub fn load(
        config_path: &PathBuf,
        weight_paths: &[PathBuf],
        device: &Device,
    ) -> Result<Self> {
        info!("Loading Gemma 3 model configuration...");
        let config_str = std::fs::read_to_string(config_path)?;
        let config: Gemma3Config = serde_json::from_str(&config_str)?;

        // Get eos_token_id from config
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
        let eos_token_id = config_json
            .get("eos_token_id")
            .and_then(|v| v.as_u64().or_else(|| v.as_array().and_then(|arr| arr.first()?.as_u64())))
            .unwrap_or(1) as u32;

        info!(
            "Gemma 3 config: vocab_size={}, hidden_size={}, num_layers={}, num_heads={}, head_dim={}",
            config.vocab_size,
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim
        );

        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(weight_paths, dtype, device)? };

        info!("Building Gemma 3 model...");

        // Build model components
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("model.embed_tokens"))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = Gemma3DecoderLayer::new(&config, i, vb.pp(format!("model.layers.{}", i)), device)?;
            layers.push(layer);
        }

        let norm = Gemma3RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;

        // lm_head may be tied to embed_tokens
        let lm_head = match linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head")) {
            Ok(lm_head) => lm_head,
            Err(_) => {
                // Tied weights - use embed_tokens weight transposed
                let weight = vb.pp("model.embed_tokens").get((config.vocab_size, config.hidden_size), "weight")?;
                Linear::new(weight, None)
            }
        };

        info!("Gemma 3 model loaded successfully!");
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config,
            eos_token_id,
        })
    }
}

impl LanguageModel for Gemma3Model {
    fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        // Embed tokens with Gemma scaling
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let scale = (self.config.hidden_size as f64).sqrt();
        hidden = (hidden * scale)?;

        // Run through transformer layers
        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden = layer.forward(&hidden, position, i)?;
        }

        // Final norm and logits
        hidden = self.norm.forward(&hidden)?;
        let logits = self.lm_head.forward(&hidden)?;

        Ok(logits)
    }

    fn reset_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}
