//! Qwen3-Next inference example
//!
//! This example demonstrates inference with the Qwen3-Next model, which features
//! a hybrid architecture combining full attention and linear attention (Gated Delta Net)
//! layers with Mixture-of-Experts (MoE) FFN.
//!
//! The model natively supports 256K token context. Use --yarn to enable YARN context
//! extension for 1M token support.

use anyhow::{Error as E, Result};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use paramecia_core::{DType, Device, IndexOp, Tensor};
use paramecia_engine::tensor_trace_agg::tensor_op_aggregation;
use paramecia_model::models::qwen3_next::{
    self, DeviceOffloadMode, KvCacheQuantization, LayerDeviceMap,
};
use paramecia_model::token_output_stream::TokenOutputStream;
use paramecia_model::{generation::LogitsProcessor, utils, YarnConfig};
use std::io::Write;
use tokenizers::Tokenizer;
use tracing_subscriber::{fmt::format::FmtSpan, EnvFilter};

#[cfg(feature = "qwen3next_80b_a3b")]
const DEFAULT_MODEL_ID: &str = "unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF";
#[cfg(not(feature = "qwen3next_80b_a3b"))]
const DEFAULT_MODEL_ID: &str = "unsloth/Qwen3.5-35B-A3B-GGUF";

#[cfg(feature = "qwen3next_80b_a3b")]
const DEFAULT_MODEL_FILE: &str = "Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf";
#[cfg(not(feature = "qwen3next_80b_a3b"))]
const DEFAULT_MODEL_FILE: &str = "Qwen3.5-35B-A3B-Q4_K_M.gguf";

#[cfg(feature = "qwen3next_80b_a3b")]
const DEFAULT_TOKENIZER_REPO: &str = "Qwen/Qwen3-Next-80B-A3B-Instruct";
#[cfg(not(feature = "qwen3next_80b_a3b"))]
const DEFAULT_TOKENIZER_REPO: &str = "Qwen/Qwen3.5-35B-A3B";

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

    /// Disable KV-cache quantization (enabled by default with Q8_0 8-bit).
    #[arg(long)]
    no_kv_quant: bool,

    /// Enable YARN for 1M token context extension.
    /// YARN extends the context window from 256K to 1M using frequency-based RoPE interpolation.
    #[arg(long)]
    yarn: bool,

    /// The model repository to use on HuggingFace Hub.
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    model_id: String,

    /// The model file to use.
    #[arg(long, default_value = DEFAULT_MODEL_FILE)]
    model_file: String,

    /// The tokenizer repository to use.
    #[arg(long, default_value = DEFAULT_TOKENIZER_REPO)]
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

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Batch size for parallel generation (default: 1).
    /// When > 1, generates multiple sequences in parallel from the same prompt.
    /// Only the first sequence is streamed; additional sequences are printed at the end.
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Disable prefetch-based pipelining (enabled by default).
    /// Prefetch pipelining hides transfer latency using a two-stage pipeline.
    #[arg(long)]
    no_prefetch: bool,

    /// Number of speculative tokens to generate via MTP (Multi-Token Prediction).
    /// Default 0 means MTP is disabled and standard autoregressive decoding is used.
    /// Values > 0 enable MTP speculative decoding with the specified number of draft tokens.
    /// Requires a model with MTP weights (e.g., qwen3next-mtp16-*.gguf).
    #[arg(long, default_value_t = 0)]
    n_speculative: usize,

    /// Multi-GPU layer split proportions (e.g., "3,1" = 75% GPU 0, 25% GPU 1).
    #[arg(long)]
    layer_split: Option<String>,

    /// Load snapshot from file and resume generation.
    #[arg(long)]
    snapshot: Option<String>,

    /// Save snapshot to file after generation.
    #[arg(long)]
    save_snapshot: Option<String>,

    /// Save snapshot every N tokens during generation.
    #[arg(long)]
    snapshot_interval: Option<usize>,
}

fn tensor_op_top_k_from_env() -> usize {
    std::env::var("PARAMECIA_TENSOR_OP_TOPK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(20)
}

fn main() -> Result<()> {
    let (tensor_op_agg_guard, init_result) = {
        use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

        let (tensor_layer, tensor_guard) = tensor_op_aggregation(tensor_op_top_k_from_env());
        let init = tracing_subscriber::registry()
            .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
            .with(
                tracing_subscriber::fmt::layer()
                    // Emit span close events so timing for tensor op spans is visible.
                    .with_span_events(FmtSpan::CLOSE)
                    .with_target(false),
            )
            .with(tensor_layer)
            .try_init();
        (tensor_guard, init)
    };

    init_result.map_err(|e| E::msg(e.to_string()))?;
    let _tensor_op_agg_guard = tensor_op_agg_guard;

    let args = Args::parse();

    println!("Qwen3-Next Inference Example");
    println!("============================\n");
    println!("Hybrid Architecture: Full Attention + Linear Attention (Gated Delta Net)");
    println!("with Mixture-of-Experts FFN\n");

    // Select device
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0)
            .or_else(|_| Device::new_vulkan(0))
            .or_else(|_| Device::new_metal(0))
            .unwrap_or(Device::Cpu)
    };
    println!("Device: {:?}", device);

    let offload_mode = if args.cpu {
        DeviceOffloadMode::FullGpu // All on CPU anyway, no offloading needed
    } else {
        match args.offload.as_str() {
            "none" => DeviceOffloadMode::FullGpu,
            "experts" => DeviceOffloadMode::ExpertsOnCpu,
            "down" => DeviceOffloadMode::DownProjectionsOnCpu,
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
            DeviceOffloadMode::DownProjectionsOnCpu => {
                println!("Offload: down (down projections on CPU)");
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
        println!("KV-Cache Quantization: ENABLED (Q8_0)");
        KvCacheQuantization::Q8_0
    };

    // Configure YARN for extended context (256K -> 1M)
    let yarn_config = if args.yarn {
        let yarn = YarnConfig::for_1m_context(262_144); // Qwen3-Next native context is 256K
        println!(
            "YARN Context Extension: ENABLED (256K -> 1M tokens, scale factor: {:.1}x)",
            yarn.scale_factor()
        );
        Some(yarn)
    } else {
        println!("YARN Context Extension: DISABLED (using native 256K context)");
        None
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

    // Load model with YARN configuration
    let layer_split_str = args
        .layer_split
        .or_else(|| std::env::var("PARAMECIA_LAYER_SPLIT").ok());
    if let Some(ref split) = layer_split_str {
        println!("Layer split: {}", split);
    }

    println!("Loading model...");
    let start = std::time::Instant::now();
    let (mut model, device) = if let Some(ref split) = layer_split_str {
        // Multi-GPU layer parallelism
        let num_layers = {
            let mut file = std::fs::File::open(&model_path)?;
            let ct = paramecia_core::quantized::gguf_file::Content::read(&mut file)?;
            let md = &ct.metadata;
            md.get("qwen35moe.block_count")
                .or_else(|| md.get("qwen35.block_count"))
                .or_else(|| md.get("qwen3_5_moe.block_count"))
                .or_else(|| md.get("qwen3_5.block_count"))
                .or_else(|| md.get("qwen3next.block_count"))
                .or_else(|| md.get("qwen3moe.block_count"))
                .or_else(|| md.get("llama.block_count"))
                .and_then(|v| v.to_u32().ok())
                .map(|n| n as usize)
                .ok_or_else(|| anyhow::anyhow!("Could not read num_layers from GGUF"))?
        };
        let layer_device_map = LayerDeviceMap::from_proportions(split, num_layers)?;
        println!(
            "Multi-GPU: {} GPUs, {} layers",
            layer_device_map.num_gpus(),
            num_layers
        );
        // Use the LayerDeviceMap's primary device for input tensors so they share
        // the same CUDA context as the model weights (different CudaDevice instances
        // for the same GPU ordinal have different contexts, causing kernel failures).
        let primary = layer_device_map.primary_device().clone();
        let model = qwen3_next::ModelWeights::from_gguf_with_layer_split(
            &model_path,
            layer_device_map,
            offload_mode,
            kv_cache_quant,
            yarn_config,
        )?;
        (model, primary)
    } else {
        let model = qwen3_next::ModelWeights::from_gguf_with_offload_and_yarn(
            &model_path,
            &device,
            offload_mode,
            kv_cache_quant,
            yarn_config,
        )?;
        (model, device)
    };
    println!("Model loaded in {:.1}s", start.elapsed().as_secs_f32());

    let tokenizer_max_id = tokenizer.get_vocab(true).values().copied().max();
    let model_vocab = model.vocab_size();
    if let Some(max_id) = tokenizer_max_id {
        if (max_id as usize) >= model_vocab {
            anyhow::bail!(
                "Tokenizer token ID range mismatch: tokenizer max token id is {}, but model vocab size is {} (valid ids: 0..{}).",
                max_id,
                model_vocab,
                model_vocab.saturating_sub(1)
            );
        }
    }

    // GPU hot expert caching is optional - can add overhead for small cache sizes
    // Disabled by default, enable with --gpu-cache flag if needed
    // if matches!(offload_mode, DeviceOffloadMode::ExpertsOnCpu | DeviceOffloadMode::DownProjectionsOnCpu) {
    //     println!("Enabling GPU hot expert cache...");
    //     model.enable_gpu_hot_cache(32);
    // }

    // Enable prefetch pipeline by default (can be disabled with --no-prefetch)
    if !args.no_prefetch {
        println!("Enabling prefetch pipeline for hiding transfer latency...");
        model.enable_prefetch_pipeline()?;
    } else {
        println!("Prefetch pipeline: DISABLED");
    }

    // Check MTP support
    let use_mtp = args.n_speculative > 0;
    if use_mtp {
        if model.has_mtp() {
            println!(
                "MTP Speculative Decoding: ENABLED ({} draft tokens per step)",
                args.n_speculative
            );
        } else {
            eprintln!(
                "WARNING: --n-speculative={} requested but model has no MTP weights. Falling back to standard decoding.",
                args.n_speculative
            );
        }
    } else {
        println!("MTP Speculative Decoding: DISABLED (use --n-speculative N to enable)");
    }

    // Print model info
    let qtensors = model.all_qtensors();
    println!("Model has {} quantized tensors", qtensors.len());

    // Count layer types: ssm_out is always present for DeltaNet/linear attention layers
    // (ssm_a is dequantized and won't appear in qtensors; ssm_in may be split as attn_qkv)
    let mut linear_attn_count = 0;
    for (name, _) in &qtensors {
        if name.contains("ssm_out") {
            linear_attn_count += 1;
        }
    }
    let total_layers = model.num_layers();
    let full_attn_count = total_layers.saturating_sub(linear_attn_count);
    println!(
        "Layer composition: {} full attention, {} linear attention layers\n",
        full_attn_count, linear_attn_count
    );

    // Setup generation with recommended sampling: temperature=0.6, top_p=0.95, top_k=20
    // Use ArgMax for temperature=0, otherwise TopKThenTopP
    let mut token_stream = TokenOutputStream::new(tokenizer);
    let sampling = if args.temperature < 1e-7 {
        paramecia_model::generation::Sampling::ArgMax
    } else {
        paramecia_model::generation::Sampling::TopKThenTopP {
            k: args.top_k,
            p: args.top_p,
            temperature: args.temperature,
        }
    };
    let mut logits_processor = LogitsProcessor::from_sampling(args.seed, sampling.clone());
    let mut extra_logits_processors: Vec<LogitsProcessor> = (1..args.batch_size)
        .map(|i| LogitsProcessor::from_sampling(args.seed + i as u64, sampling.clone()))
        .collect();
    println!(
        "Sampling: temperature={}, top_p={}, top_k={}, repeat_penalty={}, presence_penalty={}",
        args.temperature, args.top_p, args.top_k, args.repeat_penalty, args.presence_penalty
    );

    // Load snapshot if specified
    let (mut tokens, snapshot_state_pos) = if let Some(ref ckpt_path) = args.snapshot {
        println!("\nLoading snapshot from {:?}...", ckpt_path);
        let snapshot = model.load_snapshot(ckpt_path)?;
        println!(
            "Resumed from position {} ({} tokens)",
            snapshot.state_position,
            snapshot.tokens.len()
        );

        // Print the last few tokens from snapshot for context
        let preview_len = 50.min(snapshot.tokens.len());
        if preview_len > 0 {
            let preview_tokens = &snapshot.tokens[snapshot.tokens.len() - preview_len..];
            let preview_text = token_stream
                .tokenizer()
                .decode(preview_tokens, true)
                .unwrap_or_else(|_| "[decode error]".to_string());
            println!("Snapshot context: ...{}", preview_text);
        }

        // If user provided a prompt, append it to the snapshot tokens
        // This allows continuing generation with a new instruction/question
        let mut snapshot_tokens = snapshot.tokens;
        if !args.prompt.is_empty() && args.prompt != "What is Rust?" {
            println!(
                "\nContinuing from snapshot with new prompt: \"{}\"",
                args.prompt
            );
            let prompt_tokens = token_stream
                .tokenizer()
                .encode(args.prompt.as_str(), true)
                .map_err(E::msg)?
                .get_ids()
                .to_vec();
            snapshot_tokens.extend_from_slice(&prompt_tokens);
            println!(
                "Added {} prompt tokens to snapshot context",
                prompt_tokens.len()
            );
        }

        (snapshot_tokens, Some(snapshot.state_position))
    } else {
        let toks = token_stream
            .tokenizer()
            .encode(args.prompt.as_str(), true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        (toks, None)
    };

    let mut generated_tokens = 0;
    // Qwen stop tokens:
    // - 151643: <|endoftext|> (general EOS)
    // - 151645: <|im_end|> (chat turn end for instruct models)
    let eos_token = 151643u32;
    let im_end_token = 151645u32;

    // Batch state: extra sequences (1..batch_size) track tokens independently
    let prompt_len = tokens.len();
    let mut extra_tokens: Vec<Vec<u32>> = (1..args.batch_size).map(|_| tokens.clone()).collect();
    let mut all_finished: Vec<bool> = vec![false; args.batch_size];

    // Warn about batch incompatibilities
    if args.batch_size > 1 {
        println!(
            "Batch size: {} (generating {} sequences in parallel)",
            args.batch_size, args.batch_size
        );
        if args.n_speculative > 0 {
            eprintln!("WARNING: MTP speculative decoding is not supported with --batch-size > 1. Disabling MTP.");
        }
        if args.snapshot.is_some() || args.save_snapshot.is_some() {
            eprintln!("WARNING: Snapshots are not supported with --batch-size > 1. Ignoring snapshot options.");
        }
    }

    // If loading from snapshot, skip printing prompt since it's already in the token history
    if args.snapshot.is_none() {
        print!("{}", args.prompt);
        std::io::stdout().flush()?;
    }

    let start_gen = std::time::Instant::now();
    let mut prefill_elapsed: Option<std::time::Duration> = None;
    let mut prefill_tokens = 0usize;

    // Lazy state tracking: model state is synced up to (but not including) this position
    // When loading from snapshot, use the saved state_position (not tokens.len()!)
    let mut state_position: usize = snapshot_state_pos.unwrap_or(0);

    // Track last snapshot position for periodic saving
    let mut last_snapshot_pos = state_position;

    // Threshold for using chunked prefill (matches paramecia-text backend)
    const CHUNKED_PREFILL_THRESHOLD: usize = 512;

    // Stats for MTP
    let mut mtp_accepted_total = 0usize;
    let mut mtp_speculated_total = 0usize;
    let mut mtp_bonus_tokens = 0usize;
    let mut mtp_steps = 0usize;

    let mut index = 0;
    while index < args.sample_len {
        // Use lazy state tracking for consistency
        let start_pos = state_position;
        let ctxt = &tokens[start_pos..];

        if index == 0 && args.snapshot.is_some() {
            println!("DEBUG: First iteration after snapshot load:");
            println!(
                "  state_position: {}, tokens.len(): {}, ctxt.len(): {}",
                state_position,
                tokens.len(),
                ctxt.len()
            );
        }

        // If ctxt is empty (fully synced state), use a dummy token to prime the forward pass
        // This happens when resuming from a snapshot where all tokens were processed
        let dummy_token_holder;
        let actual_ctxt = if ctxt.is_empty() && !tokens.is_empty() {
            if index == 0 {
                println!("DEBUG: Empty ctxt, using BOS token for forward pass");
            }
            // Use BOS token (151643 for Qwen) as dummy input
            dummy_token_holder = vec![151643u32];
            &dummy_token_holder
        } else if ctxt.is_empty() {
            println!("ERROR: Cannot generate from empty token history");
            break;
        } else {
            ctxt
        };

        // Choose between MTP speculative decoding and standard decoding
        if use_mtp && model.has_mtp() && ctxt.len() == 1 && args.batch_size == 1 {
            let input = Tensor::new(actual_ctxt, &device)?.unsqueeze(0)?;
            // MTP Speculative Decoding Path
            // Only use MTP for single-token decode (not prefill)
            let spec_result = model.speculative_step(&input, start_pos, args.n_speculative)?;

            // IMPORTANT: Use the raw argmax token from speculative_step, NOT a re-sampled token.
            // MTP predicted based on this token, so we must use it for verification consistency.
            // Penalties are not applied for MTP speculation (they would cause mismatch).
            let main_token: u32 = spec_result.main_token.to_vec0()?;

            // Check for EOS on main token
            if main_token == eos_token || main_token == im_end_token {
                tokens.push(main_token);
                generated_tokens += 1;
                break;
            }

            // Emit main token
            tokens.push(main_token);
            generated_tokens += 1;
            index += 1;
            if let Some(t) = token_stream.next_token(main_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }

            // If we have speculative tokens, verify them
            if !spec_result.spec_tokens.is_empty() {
                // Build draft tensor from speculative tokens
                // MTP tokens are 1D [1] tensors, extract the single element
                let spec_vec: Vec<u32> = spec_result
                    .spec_tokens
                    .iter()
                    .map(|t| {
                        // Handle both 0D scalar and 1D [1] tensors
                        if t.dims().is_empty() {
                            t.to_vec0::<u32>()
                        } else {
                            t.flatten_all()?.to_vec1::<u32>().map(|v| v[0])
                        }
                    })
                    .collect::<paramecia_core::Result<Vec<_>>>()?;

                // CRITICAL: Draft tensor must include main_token at the beginning!
                // Verification feeds [main_token, spec_0, spec_1, ...] to the model:
                // - Position N+1: see main_token → logits should predict spec_0
                // - Position N+2: see spec_0 → logits should predict spec_1
                // - etc.
                let mut draft_vec = vec![main_token];
                draft_vec.extend_from_slice(&spec_vec);
                let draft_len = spec_vec.len(); // Only count spec tokens for acceptance
                let draft_tensor = Tensor::new(draft_vec.as_slice(), &device)?.unsqueeze(0)?;

                // Verify offset is position after the input that was processed
                // (start_pos + 1 since MTP only processes 1 token)
                let verify_offset = start_pos + 1;

                // Verify draft tokens
                let verify_result =
                    model.verify_and_accept(&draft_tensor, spec_result.snapshots, verify_offset)?;

                mtp_steps += 1;
                mtp_speculated_total += draft_len;
                mtp_accepted_total += verify_result.num_accepted;

                // Accept verified tokens (from spec_vec, not draft_vec which has main_token prepended)
                for i in 0..verify_result.num_accepted {
                    let accepted_token = spec_vec[i];

                    // Check for EOS
                    if accepted_token == eos_token || accepted_token == im_end_token {
                        tokens.push(accepted_token);
                        generated_tokens += 1;
                        index = args.sample_len; // Exit outer loop
                        break;
                    }

                    tokens.push(accepted_token);
                    generated_tokens += 1;
                    index += 1;

                    if let Some(t) = token_stream.next_token(accepted_token)? {
                        print!("{t}");
                        std::io::stdout().flush()?;
                    }
                }

                // Update state position based on what KV cache actually processed
                // After verify_and_accept: KV cache always has main_token + num_accepted specs
                // (even on full rejection, main_token's state is kept)
                // verify_offset is where main_token was processed, so:
                // state_position = verify_offset + 1 + num_accepted
                state_position = verify_offset + verify_result.num_accepted + 1;

                // If we have next_logits, sample the next token (bonus token from verification)
                // next_logits is always returned and usable since we always keep main_token's state
                if let Some(next_logits) = verify_result.next_logits {
                    if index < args.sample_len {
                        let next_logits = next_logits.to_dtype(DType::F32)?;
                        let start_at = tokens.len().saturating_sub(args.repeat_last_n);
                        let penalty_context = &tokens[start_at..];
                        let next_logits = utils::apply_penalties(
                            &next_logits,
                            args.repeat_penalty,
                            args.presence_penalty,
                            penalty_context,
                        )?;
                        let next_token = logits_processor.sample(&next_logits)?;

                        if next_token == eos_token || next_token == im_end_token {
                            tokens.push(next_token);
                            generated_tokens += 1;
                            break;
                        }

                        tokens.push(next_token);
                        generated_tokens += 1;
                        mtp_bonus_tokens += 1;
                        index += 1;
                        // Sampling from next_logits doesn't run a forward pass
                        // state_position was already set correctly above, no change needed

                        if let Some(t) = token_stream.next_token(next_token)? {
                            print!("{t}");
                            std::io::stdout().flush()?;
                        }
                    }
                }
            } else {
                // No speculation, just update state position
                state_position = tokens.len();
            }
        } else {
            // Standard Autoregressive Decoding Path
            // Build input tensor (batched if batch_size > 1)
            let input = if args.batch_size > 1 {
                let mut batch_inputs = vec![Tensor::new(actual_ctxt, &device)?];
                for extra in &extra_tokens {
                    let extra_ctxt = &extra[start_pos..];
                    batch_inputs.push(Tensor::new(extra_ctxt, &device)?);
                }
                Tensor::stack(&batch_inputs, 0)?
            } else {
                Tensor::new(actual_ctxt, &device)?.unsqueeze(0)?
            };

            // Use chunked prefill for large prompts to avoid VRAM spikes with KV-cache quantization
            let t_fwd = std::time::Instant::now();
            let logits = if ctxt.len() > CHUNKED_PREFILL_THRESHOLD {
                model.forward_chunked(&input, start_pos, None)?
            } else {
                model.forward(&input, start_pos)?
            };
            let fwd_ms = t_fwd.elapsed().as_secs_f64() * 1000.0;

            // Capture prefill timing on first iteration (multi-token forward)
            if prefill_elapsed.is_none() && ctxt.len() > 1 {
                prefill_elapsed = Some(t_fwd.elapsed());
                prefill_tokens = ctxt.len();
                let prefill_secs = t_fwd.elapsed().as_secs_f64();
                let prefill_rate = if prefill_secs > 0.0 {
                    prefill_tokens as f64 / prefill_secs
                } else {
                    0.0
                };
                println!(
                    "\nPrefill: {} tokens in {:.2}s ({:.1} tok/s)",
                    prefill_tokens, prefill_secs, prefill_rate
                );
            }

            #[cfg(feature = "vulkan")]
            {
                paramecia_core::vulkan_backend::device::print_and_reset_stats();
                // Enable detailed transfer log for 2nd token only
                if generated_tokens == 1 {
                    paramecia_core::vulkan_backend::device::enable_transfer_log(600);
                }
            }

            // State is now synced
            state_position = tokens.len();

            // Get logits for sequence 0 (primary/streamed sequence)
            // model.forward returns [batch, seq, vocab] or [batch, vocab] for seq=1
            let t_post = std::time::Instant::now();

            if !all_finished[0] {
                let seq0_logits = logits.i(0)?; // Select batch element 0
                let seq0_logits = if seq0_logits.rank() == 2 {
                    // [seq, vocab] - get last position
                    seq0_logits
                        .i(seq0_logits.dim(0)? - 1)?
                        .to_dtype(DType::F32)?
                } else {
                    // [vocab] - already just one position
                    seq0_logits.to_dtype(DType::F32)?
                };

                let start_at = tokens.len().saturating_sub(args.repeat_last_n);
                let penalty_context = &tokens[start_at..];
                let seq0_logits = utils::apply_penalties(
                    &seq0_logits,
                    args.repeat_penalty,
                    args.presence_penalty,
                    penalty_context,
                )?;

                let next_token = logits_processor.sample(&seq0_logits)?;
                if index < 3 && args.snapshot.is_some() {
                    let logits_min = seq0_logits.min(0)?.to_vec0::<f32>()?;
                    let logits_max = seq0_logits.max(0)?.to_vec0::<f32>()?;
                    println!("DEBUG: Generated token {}: {} (decoded: {:?}), logits range: [{:.2}, {:.2}]",
                        index, next_token,
                        token_stream.tokenizer().decode(&[next_token], false).ok(),
                        logits_min, logits_max);
                }
                tokens.push(next_token);
                generated_tokens += 1;

                if next_token == eos_token || next_token == im_end_token {
                    all_finished[0] = true;
                    if args.batch_size == 1 || all_finished.iter().all(|&f| f) {
                        break;
                    }
                } else if let Some(t) = token_stream.next_token(next_token)? {
                    print!("{t}");
                    std::io::stdout().flush()?;
                }
            } else {
                // Seq 0 finished; push padding to keep lengths aligned
                tokens.push(eos_token);
            }

            let post_ms = t_post.elapsed().as_secs_f64() * 1000.0;
            if std::env::var("PARAMECIA_PROFILE").is_ok() {
                eprintln!(
                    "[TIMING] fwd={:.1}ms post={:.1}ms total={:.1}ms",
                    fwd_ms,
                    post_ms,
                    fwd_ms + post_ms
                );
            }
            index += 1;

            // Process extra sequences (batch_size > 1)
            for seq_idx in 0..extra_tokens.len() {
                if all_finished[seq_idx + 1] {
                    // Push padding so all sequences stay the same length
                    extra_tokens[seq_idx].push(eos_token);
                    continue;
                }

                let seq_logits = logits.i(seq_idx + 1)?;
                let seq_logits = if seq_logits.rank() == 2 {
                    seq_logits.i(seq_logits.dim(0)? - 1)?.to_dtype(DType::F32)?
                } else {
                    seq_logits.to_dtype(DType::F32)?
                };

                let start_at = extra_tokens[seq_idx]
                    .len()
                    .saturating_sub(args.repeat_last_n);
                let penalty_context = &extra_tokens[seq_idx][start_at..];
                let seq_logits = utils::apply_penalties(
                    &seq_logits,
                    args.repeat_penalty,
                    args.presence_penalty,
                    penalty_context,
                )?;

                let extra_next = extra_logits_processors[seq_idx].sample(&seq_logits)?;
                extra_tokens[seq_idx].push(extra_next);

                if extra_next == eos_token || extra_next == im_end_token {
                    all_finished[seq_idx + 1] = true;
                }
            }

            // Check if all sequences are done
            if all_finished.iter().all(|&f| f) {
                break;
            }
        }

        // Periodic snapshot saving
        if let Some(interval) = args.snapshot_interval {
            if let Some(ref base_path) = args.save_snapshot {
                if tokens.len() - last_snapshot_pos >= interval {
                    let ckpt_path = format!("{}.step{}", base_path, tokens.len());
                    println!("\n[Saving snapshot: {}]", ckpt_path);
                    model.save_snapshot(&ckpt_path, state_position, &tokens)?;
                    last_snapshot_pos = tokens.len();
                }
            }
        }
    }

    let total_elapsed = start_gen.elapsed().as_secs_f64();
    let gen_elapsed = total_elapsed - prefill_elapsed.map_or(0.0, |d| d.as_secs_f64());
    let gen_tokens = generated_tokens;
    let gen_rate = if gen_elapsed > 0.0 {
        gen_tokens as f64 / gen_elapsed
    } else {
        0.0
    };
    println!("\n\n--- Stats ---");
    if let Some(pf) = prefill_elapsed {
        let pf_secs = pf.as_secs_f64();
        let pf_rate = if pf_secs > 0.0 {
            prefill_tokens as f64 / pf_secs
        } else {
            0.0
        };
        println!(
            "Prefill:    {} tokens in {:.2}s ({:.1} tok/s)",
            prefill_tokens, pf_secs, pf_rate
        );
    }
    println!(
        "Generation: {} tokens in {:.2}s ({:.1} tok/s)",
        gen_tokens, gen_elapsed, gen_rate
    );
    println!("Total:      {:.2}s", total_elapsed);

    // Print MTP statistics if MTP was used
    if use_mtp && mtp_steps > 0 {
        let acceptance_rate = mtp_accepted_total as f64 / mtp_speculated_total as f64 * 100.0;
        // Breakdown: mtp_steps main tokens + mtp_accepted_total spec tokens + mtp_bonus_tokens bonus tokens
        let mtp_total = mtp_steps + mtp_accepted_total + mtp_bonus_tokens;
        println!(
            "MTP Statistics: {} steps, {}/{} spec accepted ({:.1}%), {} bonus, {} total from MTP",
            mtp_steps,
            mtp_accepted_total,
            mtp_speculated_total,
            acceptance_rate,
            mtp_bonus_tokens,
            mtp_total
        );
    }

    // Print extra sequences (batch_size > 1)
    if !extra_tokens.is_empty() {
        println!("\n--- Additional Sequences ---");
        for (i, seq_tokens) in extra_tokens.iter().enumerate() {
            let gen_tokens = &seq_tokens[prompt_len..];
            // Trim trailing EOS/padding
            let end = gen_tokens
                .iter()
                .position(|&t| t == eos_token || t == im_end_token)
                .unwrap_or(gen_tokens.len());
            let text = token_stream
                .tokenizer()
                .decode(&gen_tokens[..end], true)
                .unwrap_or_else(|_| "[decode error]".to_string());
            println!("\n[Sequence {}] ({} tokens)\n{}", i + 1, end, text);
        }
    }

    // Final snapshot saving
    if let Some(ref ckpt_path) = args.save_snapshot {
        println!("\nSaving final snapshot to {:?}...", ckpt_path);
        println!(
            "  state_position: {}, tokens.len(): {}",
            state_position,
            tokens.len()
        );

        // CRITICAL: Save only the tokens that have been fully processed through the model.
        // If state_position < tokens.len(), there are unprocessed tokens that were sampled
        // but not yet run through the model. Don't save them - they'll cause corruption on reload.
        let synced_tokens = &tokens[..state_position];
        model.save_snapshot(ckpt_path, state_position, synced_tokens)?;
        println!(
            "Snapshot saved successfully (saved {} tokens, state at position {})",
            synced_tokens.len(),
            state_position
        );
    }

    Ok(())
}
