use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Production LLM Inference Engine")]
pub struct Args {
    #[arg(short, long, default_value = "crumb/nano-mistral")]
    pub model: String,

    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    #[arg(long)]
    pub cpu: bool,

    #[arg(long, env)]
    pub hf_token: Option<String>,

    #[arg(short, long)]
    pub interactive: bool,

    #[arg(short, long)]
    pub prompt: Option<String>,

    #[arg(long, default_value_t = 256)]
    pub max_tokens: usize,

    #[arg(long, default_value_t = 0.7)]
    pub temperature: f64,

    #[arg(long, default_value_t = 0.9)]
    pub top_p: f64,

    #[arg(long, default_value_t = 40)]
    pub top_k: usize,

    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,

    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    #[arg(long, default_value = "main")]
    pub revision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repeat_penalty: f32,
    pub seed: Option<u64>,
    pub stop_sequences: Vec<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            seed: None,
            stop_sequences: vec![],
        }
    }
}

impl From<&Args> for GenerationConfig {
    fn from(args: &Args) -> Self {
        let stop_sequences = vec![
            "<end_of_turn>".to_string(),
            "<|endoftext|>".to_string(),
            "</s>".to_string(),
        ];

        Self {
            max_tokens: args.max_tokens,
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            repeat_penalty: args.repeat_penalty,
            seed: if args.seed == 0 { None } else { Some(args.seed) },
            stop_sequences,
        }
    }
}
