use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use paramecia_engine::{
    Device, DeviceOffloadMode, KvCacheQuantization, ModelEngineBuilder, ModelInput,
    TokenOutputStream,
};
use serde::Serialize;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const BATCH_SIZE: usize = 32;
const RUST_BOOK_CONTENT: &str = "Welcome to The Rust Programming Language, an introductory book about Rust. The Rust programming language helps you write faster, more reliable software. High-level ergonomics and low-level control are often at odds in programming language design; Rust challenges that conflict. Through balancing powerful technical capacity and a great developer experience, Rust gives you the option to control low-level details without all the hassle traditionally associated with such control.";

#[cfg(feature = "qwen3next_80b_a3b")]
const DEFAULT_TOKENIZER_REPO: &str = "Qwen/Qwen3-Next-80B-A3B-Instruct";
#[cfg(not(feature = "qwen3next_80b_a3b"))]
const DEFAULT_TOKENIZER_REPO: &str = "Qwen/Qwen3.5-35B-A3B";

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
    Metal,
    Vulkan,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OffloadArg {
    Auto,
    None,
    Experts,
    Down,
    Updown,
}

impl OffloadArg {
    fn to_mode(self) -> DeviceOffloadMode {
        match self {
            Self::Auto => DeviceOffloadMode::default(),
            Self::None => DeviceOffloadMode::FullGpu,
            Self::Experts => DeviceOffloadMode::ExpertsOnCpu,
            Self::Down => DeviceOffloadMode::DownProjectionsOnCpu,
            Self::Updown => DeviceOffloadMode::UpDownProjectionsOnCpu,
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "train-eval")]
#[command(about = "Benchmark a training-loaded model with a fixed batch size of 32.")]
struct Args {
    /// Path to local GGUF model weights.
    #[arg(long)]
    model_path: PathBuf,

    /// Path to local tokenizer.json file (overrides tokenizer_repo).
    #[arg(long)]
    tokenizer_path: Option<PathBuf>,

    /// HuggingFace tokenizer repo to download tokenizer.json from.
    #[arg(long, default_value = DEFAULT_TOKENIZER_REPO)]
    tokenizer_repo: String,

    /// Device selection for model execution.
    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    /// Expert offload mode.
    #[arg(long, value_enum, default_value_t = OffloadArg::Auto)]
    offload: OffloadArg,

    /// KV-cache quantization mode (f16, bf16, q8_0, q4k, ...).
    #[arg(long, default_value = "q8_0")]
    kv_cache_quant: String,

    /// Number of Rust Book tokens to use in each system prompt.
    #[arg(long, default_value_t = 1000)]
    prefill_system_tokens: usize,

    /// Number of tokens to generate for every sequence in the batch.
    #[arg(long, default_value_t = 100)]
    generate_tokens: usize,

    /// Temperature for token sampling.
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Top-p (nucleus) sampling threshold.
    #[arg(long)]
    top_p: Option<f64>,

    /// Top-k sampling limit.
    #[arg(long, default_value_t = 64)]
    top_k: usize,

    /// Number of tail samples for distribution reporting.
    #[arg(long, default_value_t = 16)]
    tail_samples: usize,

    /// RNG seed.
    #[arg(long, default_value_t = 299_792_458)]
    seed: u64,
}

#[derive(Debug, Serialize)]
struct TrainEvalOutput {
    batch_size: usize,
    prefill_tokens_per_second: f64,
    generation_tokens_per_second: f64,
    generated_text: Vec<String>,
}

#[tokio::main]
async fn main() {
    if let Err(err) = run(Args::parse()).await {
        eprintln!("train-eval failed: {err:#}");
        std::process::exit(1);
    }
}

async fn run(args: Args) -> Result<()> {
    if args.prefill_system_tokens == 0 {
        bail!("--prefill-system-tokens must be greater than 0");
    }
    if args.generate_tokens == 0 {
        bail!("--generate-tokens must be greater than 0");
    }

    let mut builder = ModelEngineBuilder::new(&args.model_path)
        .offload_mode(args.offload.to_mode())
        .kv_cache_quant(parse_kv_cache_quant(&args.kv_cache_quant)?)
        .temperature(args.temperature)
        .top_k(args.top_k)
        .tail_samples(args.tail_samples)
        .seed(args.seed);

    if let Some(top_p) = args.top_p {
        builder = builder.top_p(top_p);
    }
    if let Some(path) = args.tokenizer_path.as_ref() {
        builder = builder.tokenizer_path(path);
    } else {
        builder = builder.tokenizer_repo(&args.tokenizer_repo);
    }

    builder = match args.device {
        DeviceArg::Auto => builder,
        DeviceArg::Cpu => builder.cpu(true),
        DeviceArg::Cuda => builder
            .device(Device::cuda_if_available(0).context("failed to initialize CUDA device 0")?),
        DeviceArg::Metal => {
            builder.device(Device::new_metal(0).context("failed to initialize Metal device 0")?)
        }
        DeviceArg::Vulkan => {
            builder.device(Device::new_vulkan(0).context("failed to initialize Vulkan device 0")?)
        }
    };

    let engine = builder
        .build_for_training()
        .map_err(|e| anyhow!("failed to build model engine for training: {e}"))?;

    let tokenizer = engine.tokenizer().clone();
    let system_tokens = select_system_prompt_tokens(&tokenizer, args.prefill_system_tokens)?;
    let prompt_tokens = build_chatml_prompt_tokens(&tokenizer, &system_tokens)?;
    let inputs = vec![vec![ModelInput::Tokens(prompt_tokens.clone())]; BATCH_SIZE];
    let mut token_streams = (0..BATCH_SIZE)
        .map(|_| TokenOutputStream::new(tokenizer.clone()))
        .collect::<Vec<_>>();
    let mut generated_text = vec![String::new(); BATCH_SIZE];

    let prefill_start = Instant::now();
    let (mut prediction_rx, cancel_tx) = engine
        .predict_completions_batched(&inputs)
        .await
        .map_err(|e| anyhow!("batched benchmark failed to start: {e}"))?;

    let mut first_step_elapsed = None;
    let mut generation_start = None;
    let mut generation_steps = 0usize;
    let mut cancel_tx = Some(cancel_tx);
    let mut reached_limit = false;

    while let Some(update) = prediction_rx.recv().await {
        let predictions = match update {
            Ok(predictions) => predictions,
            Err(_) if reached_limit => break,
            Err(e) => return Err(anyhow!("batched benchmark failed: {e}")),
        };

        if first_step_elapsed.is_none() {
            first_step_elapsed = Some(prefill_start.elapsed());
            generation_start = Some(Instant::now());
        } else {
            generation_steps += 1;
        }

        for ((prediction, stream), text) in predictions
            .into_iter()
            .zip(&mut token_streams)
            .zip(&mut generated_text)
        {
            if let Some(piece) = stream.next_token(prediction.token_id)? {
                text.push_str(&piece);
            }
        }

        if generation_steps + 1 >= args.generate_tokens {
            reached_limit = true;
            if let Some(tx) = cancel_tx.take() {
                let _ = tx.send(());
            }
        }
    }

    if !reached_limit {
        bail!(
            "all sequences stopped after {} generation steps (requested {})",
            generation_steps + 1,
            args.generate_tokens
        );
    }

    for (stream, text) in token_streams.iter().zip(&mut generated_text) {
        if let Some(rest) = stream.decode_rest()? {
            text.push_str(&rest);
        }
    }

    let prefill_elapsed = first_step_elapsed.context("benchmark produced no tokens")?;
    let generation_elapsed = generation_start
        .context("benchmark produced no tokens")?
        .elapsed();
    let output = TrainEvalOutput {
        batch_size: BATCH_SIZE,
        prefill_tokens_per_second: tokens_per_second(
            prompt_tokens.len() * BATCH_SIZE,
            prefill_elapsed,
        ),
        generation_tokens_per_second: tokens_per_second(
            generation_steps * BATCH_SIZE,
            generation_elapsed,
        ),
        generated_text,
    };

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn select_system_prompt_tokens(
    tokenizer: &tokenizers::Tokenizer,
    system_token_count: usize,
) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(RUST_BOOK_CONTENT, false)
        .map_err(|e| anyhow!("failed to tokenize Rust Book text: {e}"))?;
    let base_ids = encoding.get_ids();
    if base_ids.is_empty() {
        bail!("embedded Rust Book content tokenized to zero tokens");
    }

    Ok(base_ids
        .iter()
        .copied()
        .cycle()
        .take(system_token_count)
        .collect())
}

fn build_chatml_prompt_tokens(
    tokenizer: &tokenizers::Tokenizer,
    system_tokens: &[u32],
) -> Result<Vec<u32>> {
    const SYSTEM_PREFIX: &str = "<|im_start|>system\n";
    const CHAT_SUFFIX: &str = "\n<|im_end|>\n<|im_start|>user\nWhat is the Rust programming language?\n<|im_end|>\n<|im_start|>assistant\n";

    let mut prompt_tokens = encode_piece(tokenizer, SYSTEM_PREFIX)?;
    prompt_tokens.extend_from_slice(system_tokens);
    prompt_tokens.extend_from_slice(&encode_piece(tokenizer, CHAT_SUFFIX)?);
    Ok(prompt_tokens)
}

fn encode_piece(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    tokenizer
        .encode(text, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|e| anyhow!("failed to tokenize prompt piece: {e}"))
}

fn parse_kv_cache_quant(s: &str) -> Result<KvCacheQuantization> {
    KvCacheQuantization::from_str(s).ok_or_else(|| anyhow!("invalid --kv-cache-quant value '{s}'"))
}

fn tokens_per_second(tokens: usize, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs <= 0.0 {
        0.0
    } else {
        tokens as f64 / secs
    }
}
