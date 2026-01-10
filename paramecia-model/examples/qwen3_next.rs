//! Qwen3-Next inference example
//!
//! This example demonstrates inference with the Qwen3-Next model, which features
//! a hybrid architecture combining full attention and linear attention (Gated Delta Net)
//! layers with Mixture-of-Experts (MoE) FFN.

use anyhow::{Error as E, Result};
use candle::{DType, Device, IndexOp, Tensor};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use paramecia_model::qwen3_next::{self, DeviceOffloadMode, KvCacheQuantization};
use paramecia_model::token_output_stream::TokenOutputStream;
use paramecia_model::{generation::LogitsProcessor, utils};
use std::io::Write;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Device offload mode for MoE expert weights.
    /// Options: none (all GPU), up (up on CPU), updown (up+down on CPU), experts (all on CPU)
    #[arg(long, default_value = "experts")]
    offload: String,

    /// Disable KV-cache quantization (enabled by default with Q4K 4-bit).
    #[arg(long)]
    no_kv_quant: bool,

    /// The model repository to use on HuggingFace Hub.
    #[arg(long, default_value = "unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF")]
    model_id: String,

    /// The model file to use.
    #[arg(long, default_value = "Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf")]
    model_file: String,

    /// The tokenizer repository to use.
    #[arg(long, default_value = "Qwen/Qwen3-Next-80B-A3B-Instruct")]
    tokenizer_repo: String,

    /// Path to local model file (overrides HF download).
    #[arg(long)]
    model_path: Option<String>,

    /// Path to local tokenizer file (overrides HF download).
    #[arg(long)]
    tokenizer_path: Option<String>,

    /// The prompt to start generation with.
    #[arg(
        long,
        default_value = "<|im_start|>user\nWhat is Rust?<|im_end|>\n<|im_start|>assistant\n"
    )]
    prompt: String,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    /// The temperature used to generate samples (default: 0.7).
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Nucleus sampling probability cutoff (default: 0.8).
    #[arg(long, default_value_t = 0.8)]
    top_p: f64,

    /// Top-k sampling limit (default: 20).
    #[arg(long, default_value_t = 20)]
    top_k: usize,

    /// Repetition penalty (multiplicative) (default: 1.1).
    /// Divides logits of previously seen tokens by this value.
    /// A value of 1.0 means disabled.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// Presence penalty (additive/flat) (default: 1.0).
    /// Subtracts this value from logits of tokens that have appeared.
    /// A value of 0.0 means disabled.
    #[arg(long, default_value_t = 1.0)]
    presence_penalty: f32,

    /// Thinking budget in tokens (default: 500).
    /// After this many tokens in <think> block, </think> is injected.
    /// Set to 0 to disable.
    #[arg(long, default_value_t = 500)]
    thinking_budget: usize,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Enable prefetch-based pipelining for hiding transfer latency.
    /// Uses a two-stage pipeline: prefetch thread + MoE worker.
    #[arg(long)]
    prefetch_pipeline: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Qwen3-Next Inference Example");
    println!("============================\n");
    println!("Hybrid Architecture: Full Attention + Linear Attention (Gated Delta Net)");
    println!("with Mixture-of-Experts FFN\n");

    // Select device
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)
            .or_else(|_| candle::Device::cuda_if_available(0))
            .unwrap_or(Device::Cpu)
    };
    println!("Device: {:?}", device);

    let offload_mode = if args.cpu {
        DeviceOffloadMode::FullGpu // All on CPU anyway, no offloading needed
    } else {
        match args.offload.as_str() {
            "none" => DeviceOffloadMode::FullGpu,
            "experts" => DeviceOffloadMode::ExpertsOnCpu,
            "up" => DeviceOffloadMode::UpProjectionsOnCpu,
            "updown" => DeviceOffloadMode::UpDownProjectionsOnCpu,
            other => {
                eprintln!("Unknown offload mode '{}', using 'experts'", other);
                DeviceOffloadMode::ExpertsOnCpu
            }
        }
    };

    if !args.cpu {
        match offload_mode {
            DeviceOffloadMode::FullGpu => {
                println!("Offload: none (all weights on GPU)");
            }
            DeviceOffloadMode::ExpertsOnCpu => {
                println!("Offload: experts (all MoE experts on CPU)");
            }
            DeviceOffloadMode::UpProjectionsOnCpu => {
                println!("Offload: up (up projections on CPU)");
            }
            DeviceOffloadMode::UpDownProjectionsOnCpu => {
                println!("Offload: updown (up+down projections on CPU, gate on GPU)");
            }
        }
    }

    let kv_cache_quant = if args.no_kv_quant {
        println!("KV-Cache Quantization: DISABLED (using F16)");
        KvCacheQuantization::F16
    } else {
        println!("KV-Cache Quantization: ENABLED (Q4K)");
        KvCacheQuantization::Q4K
    };

    // Load model
    let model_path = if let Some(path) = &args.model_path {
        std::path::PathBuf::from(path)
    } else {
        println!("Downloading model from HuggingFace...");
        let api = Api::new()?;
        let model_repo = api.repo(Repo::with_revision(
            args.model_id.clone(),
            RepoType::Model,
            "main".to_string(),
        ));
        model_repo.get(&args.model_file)?
    };
    println!("Model path: {:?}", model_path);

    // Load tokenizer
    let tokenizer = if let Some(path) = &args.tokenizer_path {
        Tokenizer::from_file(path).map_err(E::msg)?
    } else {
        println!("Downloading tokenizer from HuggingFace...");
        let api = Api::new()?;
        let tokenizer_repo = api.repo(Repo::with_revision(
            args.tokenizer_repo.clone(),
            RepoType::Model,
            "main".to_string(),
        ));
        let tokenizer_path = tokenizer_repo.get("tokenizer.json")?;
        Tokenizer::from_file(tokenizer_path).map_err(E::msg)?
    };
    println!("Tokenizer loaded\n");

    // Load model
    println!("Loading model...");
    let start = std::time::Instant::now();
    let mut model = qwen3_next::ModelWeights::from_gguf_with_offload_mode(
        &model_path,
        &device,
        offload_mode,
        kv_cache_quant,
    )?;
    println!("Model loaded in {:.1}s", start.elapsed().as_secs_f32());

    // GPU hot expert caching is optional - can add overhead for small cache sizes
    // Disabled by default, enable with --gpu-cache flag if needed
    // if matches!(offload_mode, DeviceOffloadMode::ExpertsOnCpu | DeviceOffloadMode::UpProjectionsOnCpu) {
    //     println!("Enabling GPU hot expert cache...");
    //     model.enable_gpu_hot_cache(32);
    // }

    // Enable prefetch pipeline if requested
    if args.prefetch_pipeline {
        println!("Enabling prefetch-based pipeline for hiding transfer latency...");
        model.enable_prefetch_pipeline()?;
        println!("Prefetch pipeline enabled");
    }

    // Print model info
    let qtensors = model.all_qtensors();
    println!("Model has {} quantized tensors", qtensors.len());

    // Count layer types
    let mut full_attn_count = 0;
    let mut linear_attn_count = 0;
    for (name, _) in &qtensors {
        if name.contains("attn_q") || name.contains("attn_k") {
            full_attn_count += 1;
        } else if name.contains("ssm_in") {
            linear_attn_count += 1;
        }
    }
    // Each full attention layer has q, k, v, o (4 projections)
    // Each linear attention layer has ssm_in, ssm_beta_alpha, ssm_out (3 projections)
    println!(
        "Approximate layer composition: {} full attention, {} linear attention layers\n",
        full_attn_count / 4,
        linear_attn_count
    );

    // Setup generation with recommended sampling: temperature=0.6, top_p=0.95, top_k=20
    let mut token_stream = TokenOutputStream::new(tokenizer);
    let sampling = paramecia_model::generation::Sampling::TopKThenTopP {
        k: args.top_k,
        p: args.top_p,
        temperature: args.temperature,
    };
    let mut logits_processor = LogitsProcessor::from_sampling(args.seed, sampling);
    println!(
        "Sampling: temperature={}, top_p={}, top_k={}, repeat_penalty={}, presence_penalty={}, thinking_budget={}",
        args.temperature, args.top_p, args.top_k, args.repeat_penalty, args.presence_penalty, args.thinking_budget
    );

    let mut tokens = token_stream
        .tokenizer()
        .encode(args.prompt.as_str(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let mut generated_tokens = 0;
    // Qwen stop tokens:
    // - 151643: <|endoftext|> (general EOS)
    // - 151645: <|im_end|> (chat turn end for instruct models)
    let eos_token = 151643u32;
    let im_end_token = 151645u32;

    print!("{}", args.prompt);
    std::io::stdout().flush()?;

    let start_gen = std::time::Instant::now();

    // Lazy state tracking: model state is synced up to (but not including) this position
    let mut state_position: usize = 0;

    let mut index = 0;
    while index < args.sample_len {
        // Use lazy state tracking for consistency
        let start_pos = state_position;
        let ctxt = &tokens[start_pos..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, start_pos)?;

        // State is now synced
        state_position = tokens.len();

        // Get logits for last position
        // model.forward returns [batch, seq, vocab] or [batch, vocab] for seq=1
        let logits = logits.squeeze(0)?; // Remove batch dim
        let logits = if logits.rank() == 2 {
            // [seq, vocab] - get last position
            logits.i(logits.dim(0)? - 1)?.to_dtype(DType::F32)?
        } else {
            // [vocab] - already just one position
            logits.to_dtype(DType::F32)?
        };

        let start_at = tokens.len().saturating_sub(args.repeat_last_n);
        let penalty_context = &tokens[start_at..];
        let logits = utils::apply_repeat_penalty(&logits, args.repeat_penalty, penalty_context)?;
        let logits =
            utils::apply_presence_penalty(&logits, args.presence_penalty, penalty_context)?;

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        generated_tokens += 1;
        index += 1;

        if next_token == eos_token || next_token == im_end_token {
            break;
        }

        if let Some(t) = token_stream.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }

    let elapsed = start_gen.elapsed().as_secs_f64();
    println!(
        "\n\nGenerated {} tokens in {:.2}s ({:.1} tokens/s)",
        generated_tokens,
        elapsed,
        generated_tokens as f64 / elapsed
    );

    Ok(())
}
