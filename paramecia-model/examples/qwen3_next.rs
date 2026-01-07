//! Qwen3-Next inference example
//!
//! This example demonstrates inference with the Qwen3-Next model, which features
//! a hybrid architecture combining full attention and linear attention (Gated Delta Net)
//! layers with Mixture-of-Experts (MoE) FFN.
//!
//! Supports Multi-Token Prediction (MTP) for speculative decoding, which can
//! significantly improve throughput by predicting multiple tokens at once.

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

    /// Enable Multi-Token Prediction (MTP) for speculative decoding.
    /// This predicts multiple tokens at once using the MTP head.
    #[arg(long)]
    mtp: bool,

    /// Number of tokens to speculate with MTP (default: 3).
    /// Higher values may improve throughput but reduce accuracy.
    #[arg(long, default_value_t = 3)]
    num_speculative: usize,

    /// Top-K relaxed verification for MTP speculative decoding.
    /// Accepts draft tokens if they are within the top-K candidates of the main model.
    /// Default is 1 (exact argmax match). Higher values (e.g., 5-10) increase acceptance
    /// rate at the cost of potentially different output from non-speculative decoding.
    #[arg(long, default_value_t = 1)]
    spec_top_k: usize,

    /// Enable verbose MTP debug output.
    #[arg(long)]
    verbose: bool,
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

    // Try to load MTP weights if MTP is enabled
    let mut mtp_enabled = false;
    let effective_num_speculative = args.num_speculative;
    if args.mtp {
        print!("Loading MTP weights... ");
        std::io::stdout().flush()?;
        match model.load_mtp(&model_path) {
            Ok(()) => {
                let num_mtp_layers = model.num_mtp_layers();
                // MTP uses recurrent prediction with a single layer
                // Speculation depth is not limited by number of layers
                println!(
                    "OK ({} MTP layer(s), recurrent speculation depth {})",
                    num_mtp_layers, effective_num_speculative
                );
                mtp_enabled = true;
            }
            Err(e) => {
                println!("not found ({}) - using standard generation", e);
            }
        }
    }
    println!();

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
    let mut mtp_tokens = 0;
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

    // Cache for hidden states from previous round - avoids redundant forward passes
    let mut cached_hidden: Option<Tensor> = None;
    let mut cached_next_token: Option<u32> = None;

    let mut index = 0;
    while index < args.sample_len {
        // MTP speculative decoding
        if mtp_enabled && generated_tokens > 0 && effective_num_speculative > 0 {
            // Check if we have cached hidden states from previous round
            let (hidden_states, next_token) = if let (Some(hidden), Some(tok)) =
                (cached_hidden.take(), cached_next_token.take())
            {
                // Use cached hidden states - no forward needed!
                if args.verbose {
                    eprintln!(
                        "\n[MTP DEBUG] Using cached hidden states, next_token={}",
                        tok
                    );
                }
                (hidden, tok)
            } else {
                // Need to catch-up: forward all tokens from state_position
                let catchup_start = state_position;
                let ctxt = &tokens[catchup_start..];

                if args.verbose {
                    eprintln!("\n[MTP DEBUG] Catch-up forward: state_pos={}, tokens.len()={}, processing {} tokens",
                        state_position, tokens.len(), ctxt.len());
                    let last_n = tokens.len().saturating_sub(10);
                    eprintln!("[MTP DEBUG] tokens[{}..]: {:?}", last_n, &tokens[last_n..]);
                }

                let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
                let (logits, hidden_states) =
                    model.forward_with_hidden_states(&input, catchup_start)?;

                // State is now synced to tokens.len()
                state_position = tokens.len();

                // Get logits for last position and sample next token
                let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
                let start_at = tokens.len().saturating_sub(args.repeat_last_n);
                let penalty_context = &tokens[start_at..];
                let logits =
                    utils::apply_repeat_penalty(&logits, args.repeat_penalty, penalty_context)?;
                let logits =
                    utils::apply_presence_penalty(&logits, args.presence_penalty, penalty_context)?;
                let next_token = logits_processor.sample(&logits)?;

                (hidden_states, next_token)
            };

            // Push next_token to sequence
            tokens.push(next_token);
            generated_tokens += 1;
            index += 1;

            if next_token == eos_token || next_token == im_end_token {
                if let Some(t) = token_stream.next_token(next_token)? {
                    print!("{t}");
                    std::io::stdout().flush()?;
                }
                break;
            }

            if let Some(t) = token_stream.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }

            // Generate draft tokens with MTP using greedy decoding for consistency
            // MTP has its OWN tiny KV cache that starts fresh each speculation round
            model.clear_mtp_cache();
            let mut draft_tokens: Vec<u32> = Vec::new();
            let mut current_hidden = hidden_states.clone();
            let mut current_token = next_token;

            // Track penalty context for draft generation
            let mut draft_penalty_tokens = tokens.clone();

            // MTP base position: the position AFTER the last verified token
            // tokens.len() is the number of tokens including the just-generated next_token
            let mtp_base_pos = tokens.len();

            for mtp_step in 0..effective_num_speculative {
                let mtp_input = Tensor::new(&[current_token], &device)?.unsqueeze(0)?;

                // Actual sequence position for this draft token (for RoPE)
                let mtp_pos = mtp_base_pos + mtp_step;

                if args.verbose {
                    let hid_f32 = current_hidden.to_dtype(DType::F32)?;
                    let mean = hid_f32.mean_all()?.to_scalar::<f32>()?;
                    eprintln!(
                        "[EXAMPLE] MTP step {}: token={}, pos={}, hidden mean={:.4}",
                        mtp_step, current_token, mtp_pos, mean
                    );
                }

                // Use actual sequence position for RoPE, mtp_step for layer selection
                let (mtp_logits, mtp_hidden) = model.forward_mtp_with_hidden(
                    &mtp_input,
                    &current_hidden,
                    mtp_pos,  // Actual sequence position for RoPE
                    mtp_step, // MTP layer selection (typically 0 for single-layer MTP)
                )?;
                let mtp_logits = mtp_logits.squeeze(0)?.to_dtype(DType::F32)?;

                // Apply repeat penalty
                let start_at = draft_penalty_tokens
                    .len()
                    .saturating_sub(args.repeat_last_n);
                let penalty_context = &draft_penalty_tokens[start_at..];
                let mtp_logits_penalized =
                    utils::apply_repeat_penalty(&mtp_logits, args.repeat_penalty, penalty_context)?;
                let mtp_logits_penalized = utils::apply_presence_penalty(
                    &mtp_logits_penalized,
                    args.presence_penalty,
                    penalty_context,
                )?;

                // Use greedy (argmax) for draft generation to match verification
                let draft_token = mtp_logits_penalized.argmax(0)?.to_scalar::<u32>()?;

                draft_tokens.push(draft_token);
                draft_penalty_tokens.push(draft_token);

                if draft_token == eos_token || draft_token == im_end_token {
                    break;
                }

                current_hidden = mtp_hidden;
                current_token = draft_token;
            }

            // JOINT VERIFICATION-GENERATION with O(1) STATE SLICING
            //
            // Key optimizations:
            // 1. PARALLEL verification with intermediate DeltaNet state materialization
            // 2. ALL ACCEPTED: O(1) - use H[last] for MTP, no extra forward
            // 3. PARTIAL REJECT: O(1) state slice to correct position, no re-forward
            // 4. Bonus/correction token becomes anchor of next batch - no separate forward
            //
            if !draft_tokens.is_empty() {
                // Build verification batch: [next_token, d0, d1, d2, ...]
                let mut verify_tokens = vec![next_token];
                verify_tokens.extend_from_slice(&draft_tokens);

                let verify_input = Tensor::new(verify_tokens.as_slice(), &device)?.unsqueeze(0)?;
                let verify_start = tokens.len() - 1;

                // PARALLEL verification with state materialization
                // - all_logits: [batch, seq_len, vocab]
                // - all_hidden: [batch, seq_len, hidden] (pre-norm, for MTP)
                // - intermediate DeltaNet states stored for O(1) slicing
                let (all_logits, all_hidden) =
                    model.verify_with_state_materialization(&verify_input, verify_start)?;
                let all_logits_f32 = all_logits.squeeze(0)?.to_dtype(DType::F32)?;

                // Verify each draft token using rejection sampling
                let mut num_accepted = 0;
                let mut accepted_tokens_batch: Vec<u32> = Vec::new();
                let mut rejection_token: Option<u32> = None;

                // Debug: print verification details
                if args.verbose {
                    eprintln!(
                        "\n[MTP DEBUG] tokens.len()={}, verify_start={}",
                        tokens.len(),
                        verify_start
                    );
                    eprintln!("[MTP DEBUG] verify_tokens: {:?}", verify_tokens);
                }

                // Track tokens for penalty context (starts with current tokens)
                let mut penalty_tokens = tokens.clone();

                for (i, &draft_token) in draft_tokens.iter().enumerate() {
                    if index + num_accepted >= args.sample_len {
                        break;
                    }

                    // logits_i is for position verify_start + i, predicting token at verify_start + i + 1
                    // draft_token was MTP's prediction for position verify_start + i + 1
                    let logits_i = all_logits_f32.i(i)?;

                    // Apply penalties and compute target probabilities
                    let start_at = penalty_tokens.len().saturating_sub(args.repeat_last_n);
                    let penalty_context = &penalty_tokens[start_at..];
                    let logits_penalized = utils::apply_repeat_penalty(
                        &logits_i,
                        args.repeat_penalty,
                        penalty_context,
                    )?;
                    let logits_penalized = utils::apply_presence_penalty(
                        &logits_penalized,
                        args.presence_penalty,
                        penalty_context,
                    )?;

                    // Top-K relaxed verification: accept if draft is in top-K candidates
                    let accept = if args.spec_top_k <= 1 {
                        // Exact argmax match (default)
                        let main_argmax = logits_penalized.argmax(0)?.to_scalar::<u32>()?;
                        if args.verbose {
                            eprintln!(
                                "[MTP DEBUG] i={}: draft={}, main_argmax={}, accept={}",
                                i,
                                draft_token,
                                main_argmax,
                                main_argmax == draft_token
                            );
                        }
                        main_argmax == draft_token
                    } else {
                        // Relaxed top-K matching
                        let k = args.spec_top_k;

                        // Get logits as a vector and find top-K indices on CPU
                        // (arg_sort can be slow on GPU for small vocab, and we only need top-K)
                        let logits_vec: Vec<f32> = logits_penalized.to_vec1()?;

                        // Create (index, logit) pairs and find top-K
                        let mut indexed_logits: Vec<(u32, f32)> = logits_vec
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| (i as u32, v))
                            .collect();

                        // Partial sort to get top-K (more efficient than full sort)
                        indexed_logits.select_nth_unstable_by(k - 1, |a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });

                        // Check if draft token is in top-K
                        let top_k_vec: Vec<u32> =
                            indexed_logits[..k].iter().map(|(i, _)| *i).collect();
                        let in_top_k = top_k_vec.contains(&draft_token);

                        if args.verbose {
                            let main_argmax = logits_penalized.argmax(0)?.to_scalar::<u32>()?;
                            // Sort for display
                            let mut display_vec = indexed_logits[..k].to_vec();
                            display_vec.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let display_indices: Vec<u32> =
                                display_vec.iter().take(5).map(|(i, _)| *i).collect();
                            eprintln!(
                                "[MTP DEBUG] i={}: draft={}, main_argmax={}, top_{}={:?}, accept={}",
                                i, draft_token, main_argmax, k, display_indices, in_top_k
                            );
                        }
                        in_top_k
                    };

                    if accept {
                        // Accept the draft token
                        accepted_tokens_batch.push(draft_token);
                        penalty_tokens.push(draft_token);
                        num_accepted += 1;

                        if draft_token == eos_token || draft_token == im_end_token {
                            break;
                        }
                    } else {
                        // Reject: sample from main model's distribution
                        rejection_token = Some(logits_processor.sample(&logits_penalized)?);
                        break;
                    }
                }

                // JOINT VERIFICATION-GENERATION: Handle based on acceptance
                //
                // Key insight: We use the hidden states from verification directly.
                // The bonus/correction token becomes the "anchor" of the NEXT verification batch.
                //
                if rejection_token.is_none() && num_accepted == draft_tokens.len() {
                    // ALL ACCEPTED - State is already correct from parallel verification!
                    state_position = verify_start + verify_tokens.len();

                    // Get bonus token from last position's logits
                    let last_pos = verify_tokens.len() - 1;
                    let last_logits = all_logits_f32.i(last_pos)?;

                    // Build penalty context including accepted tokens
                    let mut temp_tokens = tokens.clone();
                    for &tok in &accepted_tokens_batch {
                        temp_tokens.push(tok);
                    }
                    let start_at = temp_tokens.len().saturating_sub(args.repeat_last_n);
                    let penalty_context = &temp_tokens[start_at..];
                    let last_logits = utils::apply_repeat_penalty(
                        &last_logits,
                        args.repeat_penalty,
                        penalty_context,
                    )?;
                    let last_logits = utils::apply_presence_penalty(
                        &last_logits,
                        args.presence_penalty,
                        penalty_context,
                    )?;

                    // Sample bonus token
                    let bonus_token = logits_processor.sample(&last_logits)?;

                    // Get hidden states for MTP from verification (NO EXTRA FORWARD!)
                    // H[last_pos] is the hidden state that produced the bonus logits
                    let mtp_hidden = all_hidden.narrow(1, last_pos, 1)?;

                    if args.verbose {
                        eprintln!(
                            "[MTP DEBUG] All accepted! bonus={}, H[{}], NO EXTRA FORWARD",
                            bonus_token, last_pos
                        );
                    }

                    // Add all accepted draft tokens
                    for &tok in &accepted_tokens_batch {
                        tokens.push(tok);
                        generated_tokens += 1;
                        mtp_tokens += 1;
                        index += 1;

                        if let Some(t) = token_stream.next_token(tok)? {
                            print!("{t}");
                            std::io::stdout().flush()?;
                        }

                        if tok == eos_token || tok == im_end_token {
                            break;
                        }
                    }

                    // Cache bonus token for next iteration - NO FORWARD NEEDED!
                    // MTP will use (bonus_token, mtp_hidden) for drafting
                    // Next verification batch: [bonus_token, new_drafts...]
                    if index < args.sample_len
                        && bonus_token != eos_token
                        && bonus_token != im_end_token
                    {
                        cached_hidden = Some(mtp_hidden);
                        cached_next_token = Some(bonus_token);
                    }

                    if args.verbose {
                        eprintln!(
                            "[MTP DEBUG] tokens.len()={}, state_position={}, cached={}",
                            tokens.len(),
                            state_position,
                            cached_hidden.is_some()
                        );
                    }
                } else if let Some(rej) = rejection_token {
                    // PARTIAL REJECT with O(1) STATE SLICING
                    //
                    // The verification already materialized intermediate states.
                    // We just "pick" the state after the last accepted token.
                    //
                    // num_accepted = number of accepted drafts
                    // In verify_tokens: [next_token, d0, d1, d2, ...]
                    // If we accepted d0 (num_accepted=1), we want state after position 1
                    // slice_index = num_accepted (state after processing verify_tokens[num_accepted])
                    //
                    let slice_index = num_accepted;
                    model.restore_to_intermediate_state(slice_index, verify_start);

                    // State is now at verify_start + slice_index + 1
                    state_position = verify_start + slice_index + 1;

                    // Get hidden state from verification (NO EXTRA FORWARD!)
                    // H[slice_index] is the hidden that produced the logits for the rejection
                    let mtp_hidden = all_hidden.narrow(1, slice_index, 1)?;

                    if args.verbose {
                        eprintln!("[MTP DEBUG] Partial reject: accepted={}, rej={}, O(1) slice to index {}",
                            num_accepted, rej, slice_index);
                    }

                    // Add accepted tokens to output
                    for &tok in &accepted_tokens_batch {
                        tokens.push(tok);
                        generated_tokens += 1;
                        mtp_tokens += 1;
                        index += 1;

                        if let Some(t) = token_stream.next_token(tok)? {
                            print!("{t}");
                            std::io::stdout().flush()?;
                        }
                    }

                    // Cache rejection token for next iteration
                    // MTP will use (rej, mtp_hidden) for drafting
                    // Next verification batch: [rej, new_drafts...]
                    if index < args.sample_len {
                        cached_hidden = Some(mtp_hidden);
                        cached_next_token = Some(rej);
                    }

                    // Clear intermediate states - no longer needed
                    model.clear_intermediate_states();

                    if args.verbose {
                        eprintln!(
                            "[MTP DEBUG] tokens.len()={}, state_position={}, cached={}",
                            tokens.len(),
                            state_position,
                            cached_hidden.is_some()
                        );
                    }
                }
            }
        } else {
            // Standard generation without MTP (uses KV cache incrementally)
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
            let logits =
                utils::apply_repeat_penalty(&logits, args.repeat_penalty, penalty_context)?;
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
    }

    let elapsed = start_gen.elapsed().as_secs_f64();
    println!(
        "\n\nGenerated {} tokens in {:.2}s ({:.1} tokens/s)",
        generated_tokens,
        elapsed,
        generated_tokens as f64 / elapsed
    );

    if mtp_enabled && mtp_tokens > 0 {
        println!(
            "MTP: {} tokens predicted via MTP ({:.1}% of total)",
            mtp_tokens,
            100.0 * mtp_tokens as f64 / generated_tokens as f64
        );
    }

    Ok(())
}
