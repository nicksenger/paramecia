use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use paramecia_engine::{
    Device, DeviceOffloadMode, KvCacheQuantization, ModelEngineBuilder, ModelInput,
    TokenOutputStream, TrainingConfig,
};
use serde::Serialize;
use std::path::PathBuf;
use std::time::{Duration, Instant};

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
#[command(about = "Benchmark a training-loaded model.")]
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
    #[arg(long, value_enum, default_value_t = OffloadArg::None)]
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

    /// Number of sequences to evaluate in parallel.
    #[arg(long, default_value_t = 32)]
    batch_size: usize,

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

    /// Apply the positive QuZO perturbation before evaluation.
    #[arg(long)]
    perturb_up: bool,

    /// Apply positive then negative QuZO perturbations before evaluation.
    #[arg(long)]
    perturb_down: bool,

    /// Run a complete QuZO perturb-up, perturb-down, and update cycle before evaluation.
    #[arg(long)]
    opt: bool,

    /// Loss at the positive perturbation, used by --opt.
    #[arg(long, default_value_t = 1.0)]
    loss_up: f32,

    /// Loss at the negative perturbation, used by --opt.
    #[arg(long, default_value_t = 0.0)]
    loss_down: f32,

    /// QuZO learning rate.
    #[arg(long, default_value_t = 0.0001)]
    learning_rate: f64,

    /// QuZO perturbation epsilon.
    #[arg(long, default_value_t = 0.001)]
    epsilon: f64,

    /// Tensors to optimize (all, attention, or qk).
    #[arg(long, default_value = "all")]
    optimize_tensors: String,

    /// Generate one perturbation tensor at a time to bound peak host memory.
    #[arg(long)]
    lazy_perturbations: bool,
}

#[derive(Debug, Serialize)]
struct TrainEvalOutput {
    batch_size: usize,
    perturb_up_seconds: Option<f64>,
    perturb_down_seconds: Option<f64>,
    optimization_seconds: Option<f64>,
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
    if args.batch_size == 0 {
        bail!("--batch-size must be greater than 0");
    }

    let mut builder = ModelEngineBuilder::new(&args.model_path)
        .offload_mode(args.offload.to_mode())
        .kv_cache_quant(parse_kv_cache_quant(&args.kv_cache_quant)?)
        .temperature(args.temperature)
        .top_k(args.top_k)
        .tail_samples(args.tail_samples)
        .seed(args.seed)
        .training_config(TrainingConfig {
            lr: args.learning_rate,
            epsilon: args.epsilon,
            optimize_tensors: args.optimize_tensors.clone(),
            lazy_perturbations: args.lazy_perturbations,
            ..TrainingConfig::default()
        });

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

    let (perturb_up_seconds, perturb_down_seconds, optimization_seconds) =
        run_training_phases(&engine, &args).await?;

    let tokenizer = engine.tokenizer().clone();
    let system_tokens = select_system_prompt_tokens(&tokenizer, args.prefill_system_tokens)?;
    let prompt_tokens = build_chatml_prompt_tokens(&tokenizer, &system_tokens)?;
    let inputs = vec![vec![ModelInput::Tokens(prompt_tokens.clone())]; args.batch_size];
    let mut token_streams = (0..args.batch_size)
        .map(|_| TokenOutputStream::new(tokenizer.clone()))
        .collect::<Vec<_>>();
    let mut generated_text = vec![String::new(); args.batch_size];

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
        batch_size: args.batch_size,
        perturb_up_seconds,
        perturb_down_seconds,
        optimization_seconds,
        prefill_tokens_per_second: tokens_per_second(
            prompt_tokens.len() * args.batch_size,
            prefill_elapsed,
        ),
        generation_tokens_per_second: tokens_per_second(
            generation_steps * args.batch_size,
            generation_elapsed,
        ),
        generated_text,
    };

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

async fn run_training_phases(
    engine: &paramecia_engine::ModelEngine,
    args: &Args,
) -> Result<(Option<f64>, Option<f64>, Option<f64>)> {
    let run_up = args.perturb_up || args.perturb_down || args.opt;
    let run_down = args.perturb_down || args.opt;

    let perturb_up_seconds = if run_up {
        let start = Instant::now();
        engine
            .perturb_up(Some(args.seed))
            .await
            .map_err(|e| anyhow!("positive perturbation failed: {e}"))?;
        Some(start.elapsed().as_secs_f64())
    } else {
        None
    };

    let perturb_down_seconds = if run_down {
        let start = Instant::now();
        engine
            .perturb_down()
            .await
            .map_err(|e| anyhow!("negative perturbation failed: {e}"))?;
        Some(start.elapsed().as_secs_f64())
    } else {
        None
    };

    let optimization_seconds = if args.opt {
        let start = Instant::now();
        engine
            .update(args.loss_up, args.loss_down)
            .await
            .map_err(|e| anyhow!("optimization update failed: {e}"))?;
        Some(start.elapsed().as_secs_f64())
    } else {
        None
    };

    Ok((
        perturb_up_seconds,
        perturb_down_seconds,
        optimization_seconds,
    ))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_size_is_configurable() {
        let default_args = Args::try_parse_from(["train-eval", "--model-path", "model.gguf"])
            .expect("default arguments should parse");
        assert_eq!(default_args.batch_size, 32);
        assert!(matches!(default_args.offload, OffloadArg::None));

        let configured_args = Args::try_parse_from([
            "train-eval",
            "--model-path",
            "model.gguf",
            "--batch-size",
            "2",
        ])
        .expect("configured arguments should parse");
        assert_eq!(configured_args.batch_size, 2);
    }

    #[test]
    fn training_phase_flags_parse_with_large_model_options() {
        let args = Args::try_parse_from([
            "train-eval",
            "--model-path",
            "model.gguf",
            "--opt",
            "--lazy-perturbations",
            "--learning-rate",
            "0.0002",
            "--epsilon",
            "0.002",
            "--optimize-tensors",
            "qk",
            "--loss-up",
            "1.5",
            "--loss-down",
            "0.5",
        ])
        .expect("training phase arguments should parse");

        assert!(args.opt);
        assert!(args.lazy_perturbations);
        assert_eq!(args.learning_rate, 0.0002);
        assert_eq!(args.epsilon, 0.002);
        assert_eq!(args.optimize_tensors, "qk");
        assert_eq!(args.loss_up, 1.5);
        assert_eq!(args.loss_down, 0.5);
    }
}
