# LLM Inference Engine

A production-grade LLM inference engine written in Rust, powered by [Candle](https://github.com/huggingface/candle) - Hugging Face's minimalist ML framework.

## Features

- Load models directly from HuggingFace Hub (like `crumb/nano-mistral`)
- OpenAI-compatible REST API
- Interactive CLI mode with streaming output
- Configurable sampling (temperature, top-p, top-k, repeat penalty)
- CPU and GPU support (CUDA, Metal, MKL, Accelerate)
- Efficient memory-mapped model loading

## Installation

```bash
# Build in release mode (recommended)
cargo build --release

# Build with GPU support
cargo build --release --features cuda    # NVIDIA CUDA
cargo build --release --features metal   # Apple Metal
cargo build --release --features mkl     # Intel MKL
cargo build --release --features accelerate  # Apple Accelerate
```

## Usage

### Server Mode (OpenAI-compatible API)

```bash
# Start with default model (crumb/nano-mistral)
./target/release/llm-engine

# Start with a specific model
./target/release/llm-engine --model crumb/nano-mistral --port 8080
```

The server exposes:
- `GET /health` - Health check
- `GET /v1/models` - List available models
- `POST /v1/completions` - Text completion
- `POST /v1/chat/completions` - Chat completion (OpenAI compatible)

Example API call:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Single Prompt Mode

```bash
./target/release/llm-engine --prompt "Once upon a time" --max-tokens 100
```

### Interactive Mode

```bash
./target/release/llm-engine --interactive
```

## CLI Options

```
Usage: llm-engine [OPTIONS]

Options:
  -m, --model <MODEL>              HuggingFace model ID [default: crumb/nano-mistral]
      --host <HOST>                Server host [default: 127.0.0.1]
      --port <PORT>                Server port [default: 8080]
      --cpu                        Force CPU (disable GPU)
      --hf-token <HF_TOKEN>        HuggingFace token for private models [env: HF_TOKEN]
  -i, --interactive                Interactive mode
  -p, --prompt <PROMPT>            Single prompt mode
      --max-tokens <MAX_TOKENS>    Maximum tokens to generate [default: 256]
      --temperature <TEMPERATURE>  Sampling temperature [default: 0.7]
      --top-p <TOP_P>              Nucleus sampling [default: 0.9]
      --top-k <TOP_K>              Top-k sampling [default: 40]
      --repeat-penalty <PENALTY>   Repeat penalty [default: 1.1]
      --seed <SEED>                Random seed [default: 0]
      --revision <REVISION>        Model revision [default: main]
  -h, --help                       Print help
  -V, --version                    Print version
```

## Supported Models

This engine supports Mistral-architecture models from HuggingFace:

- `crumb/nano-mistral` (200M params - great for testing)
- `mistralai/Mistral-7B-v0.1`
- `mistralai/Mistral-7B-Instruct-v0.1`
- And other Mistral-compatible models

## Environment Variables

- `HF_TOKEN` - HuggingFace API token for private models
- `RUST_LOG` - Logging level (e.g., `RUST_LOG=info`)

## Architecture

```
src/
├── main.rs          # Entry point, CLI, server setup
├── config.rs        # Configuration and CLI args
├── device.rs        # Device selection (CPU/GPU)
├── error.rs         # Error types
├── generation.rs    # Text generation with sampling
├── sampling.rs      # Top-k, top-p, temperature sampling
├── tokenizer.rs     # Tokenizer wrapper
├── model/
│   ├── mod.rs       # LanguageModel trait
│   ├── loader.rs    # HuggingFace Hub model loading
│   └── mistral.rs   # Mistral model implementation
└── api/
    ├── mod.rs       # API state
    ├── routes.rs    # HTTP endpoints
    └── types.rs     # Request/Response types
```

## License

MIT
