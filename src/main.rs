mod api;
mod config;
mod device;
mod error;
mod generation;
mod model;
mod sampling;
mod tokenizer;

use std::io::{self, Write};
use std::net::SocketAddr;
use std::sync::Arc;

use clap::Parser;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::api::AppState;
use crate::config::{Args, GenerationConfig};
use crate::device::{device_info, get_device};
use crate::generation::TextGenerator;
use crate::model::{
    detect_architecture, load_model, GemmaModel, Gemma2Model, Gemma3Model,
    LanguageModel, MistralModel, ModelArchitecture,
};
use crate::tokenizer::TokenizerWrapper;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            "llm_inference_engine=info,tower_http=debug".into()
        }))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    println!(
        r#"
╔═══════════════════════════════════════════════════════════════╗
║           LLM Inference Engine - Rust Edition                 ║
║                  Powered by Candle                            ║
╚═══════════════════════════════════════════════════════════════╝
"#
    );

    // Get device
    let device = get_device(args.cpu)?;
    info!("Using device: {}", device_info(&device));

    // Download model files
    info!("Loading model: {}", args.model);
    let model_files = load_model(
        &args.model,
        &args.revision,
        args.hf_token.as_deref(),
    )?;

    // Load tokenizer
    let tokenizer = TokenizerWrapper::load(&model_files.tokenizer)?;

    // Detect architecture and load appropriate model
    let architecture = detect_architecture(&model_files.config)?;
    info!("Detected architecture: {:?}", architecture);

    let model: Box<dyn LanguageModel> = match architecture {
        ModelArchitecture::Mistral => {
            Box::new(MistralModel::load(
                &model_files.config,
                &model_files.weights,
                &device,
            )?)
        }
        ModelArchitecture::Gemma => {
            Box::new(GemmaModel::load(
                &model_files.config,
                &model_files.weights,
                &device,
            )?)
        }
        ModelArchitecture::Gemma2 => {
            Box::new(Gemma2Model::load(
                &model_files.config,
                &model_files.weights,
                &device,
            )?)
        }
        ModelArchitecture::Gemma3 => {
            Box::new(Gemma3Model::load(
                &model_files.config,
                &model_files.weights,
                &device,
            )?)
        }
    };

    info!("Model loaded successfully!");
    info!("Vocabulary size: {}", tokenizer.vocab_size());

    if args.interactive {
        run_interactive_mode(model, &tokenizer, &device, &args)?;
    } else if let Some(prompt) = &args.prompt {
        run_single_prompt(model, &tokenizer, &device, prompt, &args)?;
    } else {
        run_server(model, tokenizer, device, &args).await?;
    }

    Ok(())
}

fn run_single_prompt(
    mut model: Box<dyn LanguageModel>,
    tokenizer: &TokenizerWrapper,
    device: &candle_core::Device,
    prompt: &str,
    args: &Args,
) -> anyhow::Result<()> {
    let config = GenerationConfig::from(args);

    println!("\n📝 Prompt: {}", prompt);
    println!("{}", "─".repeat(60));

    let mut generator = TextGenerator::new(
        model.as_mut(),
        tokenizer,
        device,
    );

    let output = generator.generate(prompt, &config)?;

    println!("\n🤖 Response:\n{}", output.text);
    println!("{}", "─".repeat(60));
    println!(
        "📊 Stats: {} tokens, {:.2} tokens/sec, {}ms total",
        output.tokens.len(),
        output.tokens_per_second,
        output.total_time_ms
    );

    Ok(())
}

fn run_interactive_mode(
    mut model: Box<dyn LanguageModel>,
    tokenizer: &TokenizerWrapper,
    device: &candle_core::Device,
    args: &Args,
) -> anyhow::Result<()> {
    let config = GenerationConfig::from(args);

    println!("\n🎮 Interactive Mode - Type your prompts (Ctrl+C to exit)\n");

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            println!("Goodbye!");
            break;
        }

        // Format with Gemma 3 chat template
        let formatted_prompt = format!(
            "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
            input
        );

        let mut generator = TextGenerator::new(
            model.as_mut(),
            tokenizer,
            device,
        );

        print!("AI: ");
        io::stdout().flush()?;

        match generator.generate_stream(&formatted_prompt, &config, |token| {
            print!("{}", token);
            io::stdout().flush().ok();
            true
        }) {
            Ok(output) => {
                println!(
                    "\n[{} tokens, {:.1} tok/s]\n",
                    output.tokens.len(),
                    output.tokens_per_second
                );
            }
            Err(e) => {
                error!("\nError: {}", e);
            }
        }
    }

    Ok(())
}

async fn run_server(
    model: Box<dyn LanguageModel>,
    tokenizer: TokenizerWrapper,
    device: candle_core::Device,
    args: &Args,
) -> anyhow::Result<()> {
    let state = Arc::new(AppState {
        model: Arc::new(Mutex::new(model)),
        tokenizer: Arc::new(tokenizer),
        device,
        model_id: args.model.clone(),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = api::create_router(state)
        .layer(TraceLayer::new_for_http())
        .layer(cors);

    let addr = SocketAddr::new(args.host.parse()?, args.port);

    println!(
        r#"
🚀 Server starting...
   ├─ Address: http://{}
   ├─ Model: {}
   └─ Endpoints:
      ├─ GET  /health              - Health check
      ├─ GET  /v1/models           - List models
      ├─ POST /v1/completions      - Text completion
      └─ POST /v1/chat/completions - Chat completion (OpenAI compatible)

Press Ctrl+C to stop the server.
"#,
        addr, args.model
    );

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
