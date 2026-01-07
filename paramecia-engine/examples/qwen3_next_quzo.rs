//! QuZO (Quantized Zeroth-Order) fine-tuning example for Qwen3-Next
//!
//! This example demonstrates using QuZO to directly optimize quantized weights
//! with full model forward passes. Unlike QZO which only optimizes continuous
//! scale factors, QuZO modifies the discrete quantized weights themselves using
//! stochastic rounding.
//!
//! Key features:
//! - Full model forward pass through actual Qwen3-Next model
//! - Directly modifies quantized block weights (not just scales)
//! - Uses two-seed perturbation for unbiased gradient estimation
//! - Weights stored as Arc<RwLock<QTensor>> for mutable access during training
//!
//! Usage:
//!   cargo run -p paramecia-opt --example qwen3_next_quzo --release -- <model.gguf>

use paramecia_core::quantized::{GgmlDType, SharedQTensor};
use paramecia_core::{Device, IndexOp, Result, Tensor};
use paramecia_model::models::qwen3_next::ModelWeights;
use paramecia_opt::{EpsilonConfig, ParamsQuZO, QuZO};
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

#[cfg(feature = "qwen3next_80b_a3b")]
const DEFAULT_TOKENIZER_REPO: &str = "Qwen/Qwen3-Next-80B-A3B-Instruct";
#[cfg(not(feature = "qwen3next_80b_a3b"))]
const DEFAULT_TOKENIZER_REPO: &str = "Qwen/Qwen3.5-35B-A3B";

/// Compute cross-entropy loss for completion tokens only
/// logits: [batch, seq_len, vocab_size] - full sequence logits
/// targets: [batch, completion_len] - only completion tokens
/// prompt_len: number of prompt tokens (skip these positions)
fn compute_completion_loss(logits: &Tensor, targets: &Tensor, prompt_len: usize) -> Result<Tensor> {
    // Extract logits for completion positions only
    // Position i in logits predicts token i+1, so:
    // - Position prompt_len-1 predicts first completion token
    // - We need positions [prompt_len-1, prompt_len, ..., prompt_len+completion_len-2]
    let (batch, _seq_len, vocab_size) = logits.dims3()?;
    let completion_len = targets.dim(1)?;

    // Get logits starting from position (prompt_len - 1) for completion_len positions
    let start_pos = prompt_len - 1;
    let completion_logits = logits.narrow(1, start_pos, completion_len)?;

    let batch_seq = batch * completion_len;
    let logits_flat = completion_logits.reshape((batch_seq, vocab_size))?;
    let targets_flat = targets.reshape(batch_seq)?;
    paramecia_nn::loss::cross_entropy(&logits_flat, &targets_flat)
}

/// Show top token predictions from logits at a specific position
fn show_top_predictions(
    logits: &Tensor,
    prompt: &str,
    tokenizer: &Tokenizer,
    position: usize,
    num_top: usize,
) -> Result<()> {
    let logits_2d = logits.squeeze(0)?; // [seq_len, vocab_size]

    // Get predictions at specified position
    let logits_pos = logits_2d.i(position)?;
    let probs = paramecia_nn::ops::softmax(&logits_pos, paramecia_core::D::Minus1)?;
    let probs_vec = probs.to_vec1::<f32>()?;

    let mut indexed: Vec<_> = probs_vec.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(a.1));

    print!("  \"{}\" -> ", prompt);
    for (i, (token_id, prob)) in indexed.iter().copied().take(num_top).enumerate() {
        if let Ok(decoded) = tokenizer.decode(&[token_id as u32], false) {
            let token_str = decoded.trim();
            let display = if token_str.is_empty() {
                "<space>"
            } else {
                token_str
            };
            print!("'{}' ({:.1}%)", display, prob * 100.0);
            if i < num_top - 1 {
                print!(", ");
            }
        }
    }
    println!();

    Ok(())
}

fn get_best_device() -> Result<Device> {
    if let Ok(device) = Device::new_metal(0) {
        return Ok(device);
    }
    if let Ok(device) = Device::cuda_if_available(0) {
        if let Device::Cuda(_) = device {
            return Ok(device);
        }
    }
    Ok(Device::Cpu)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = if args.len() > 1 && !args[1].starts_with("--") {
        PathBuf::from(&args[1])
    } else {
        eprintln!("Usage: {} <path-to-gguf-model> [options]", args[0]);
        eprintln!("\nOptions:");
        eprintln!("  --lr F           Learning rate (default: 0.0001)");
        eprintln!("  --epsilon F      Perturbation magnitude (default: 0.001)");
        eprintln!("  --steps N        Number of training steps (default: 10)");
        eprintln!("  --samples N      Perturbation samples per step (default: 1)");
        eprintln!("  --lb-loss F      Load balance loss weight (default: 0.0, disabled)");
        eprintln!("  --z-loss F       Z-loss weight (default: 0.0, disabled)");
        eprintln!("  --demo           Use simple demo mode without full model (faster)");
        eprintln!("\nEpsilon multipliers (effective epsilon = base * multiplier):");
        eprintln!("  --eps-embedding F   Embedding layer multiplier (default: 0.1)");
        eprintln!("  --eps-attention F   Attention Q/K/V/O multiplier (default: 1.0)");
        eprintln!("  --eps-moe-gating F  MoE router/gating multiplier (default: 0.5)");
        eprintln!("  --eps-moe-experts F MoE expert layers multiplier (default: 2.0)");
        eprintln!("  --eps-mtp F         MTP heads multiplier (default: 1.5)");
        eprintln!("  --eps-other F       Other layers multiplier (default: 1.0)");
        std::process::exit(1);
    };

    // Parse arguments
    let lr: f64 = args
        .iter()
        .position(|a| a == "--lr")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0001);
    let epsilon: f64 = args
        .iter()
        .position(|a| a == "--epsilon")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.001);
    let num_steps: usize = args
        .iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let num_samples: usize = args
        .iter()
        .position(|a| a == "--samples")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let lb_loss: f64 = args
        .iter()
        .position(|a| a == "--lb-loss")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let z_loss: f64 = args
        .iter()
        .position(|a| a == "--z-loss")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let demo_mode = args.iter().any(|a| a == "--demo");

    // Epsilon multipliers for different model components
    let eps_embedding: f64 = args
        .iter()
        .position(|a| a == "--eps-embedding")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.1);
    let eps_attention: f64 = args
        .iter()
        .position(|a| a == "--eps-attention")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let eps_moe_gating: f64 = args
        .iter()
        .position(|a| a == "--eps-moe-gating")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.5);
    let eps_moe_experts: f64 = args
        .iter()
        .position(|a| a == "--eps-moe-experts")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(2.0);
    let eps_mtp: f64 = args
        .iter()
        .position(|a| a == "--eps-mtp")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.5);
    let eps_other: f64 = args
        .iter()
        .position(|a| a == "--eps-other")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);

    let epsilon_config = EpsilonConfig {
        embedding: eps_embedding,
        attention: eps_attention,
        moe_gating: eps_moe_gating,
        moe_experts: eps_moe_experts,
        mtp: eps_mtp,
        other: eps_other,
        moe_expert_banks: 1.0,
        norms: 1.0,
        ssm: 1.0,
    };

    println!("QuZO Fine-tuning for Qwen3-Next");
    println!("================================");
    println!("Paper: arXiv:2502.12346\n");

    let device = get_best_device()?;
    println!("Using device: {:?}", device);

    if demo_mode {
        return run_demo_mode(
            &model_path,
            &device,
            lr,
            epsilon,
            num_steps,
            num_samples,
            &epsilon_config,
        );
    }

    // Full model training mode
    println!("Loading model in training mode from {:?}...", model_path);
    let start_load = Instant::now();
    let mut model = ModelWeights::from_gguf_for_training(&model_path, &device)?;
    println!(
        "Model loaded in training mode in {:.1}s",
        start_load.elapsed().as_secs_f32()
    );

    // Enable prefetch pipeline for overlapped GPU/CPU execution during training
    // This hides CPU expert processing latency by overlapping with GPU attention
    model.enable_prefetch_pipeline()?;
    if model.has_prefetch_pipeline() {
        println!("Prefetch pipeline enabled for training");
    }

    // Get trainable tensors from the model
    let quzo_tensors = model.quzo_qtensors();
    println!(
        "Found {} trainable tensors (QMatMul weights)",
        quzo_tensors.len()
    );

    if quzo_tensors.is_empty() {
        println!("\nNo QuZO-compatible tensors found.");
        println!("QuZO requires quantized formats: Q4_0, Q8_0, Q2K-Q8K");
        return Ok(());
    }

    // Show sample of tensor info
    println!("\nTrainable tensors from layer 0:");
    let layer0_tensors: Vec<_> = quzo_tensors
        .iter()
        .filter(|(n, _)| n.starts_with("blk.0."))
        .collect();
    if layer0_tensors.is_empty() {
        println!("  (none found - check if model loaded in training mode)");
    } else {
        for (name, qt) in &layer0_tensors {
            let qt_read = qt.read().unwrap();
            println!("  {} - {:?}, {:?}", name, qt_read.dtype(), qt_read.shape());
        }
    }

    // Filter to only QuZO-compatible tensors (exclude F16, BF16, and pruning metadata)
    let selected_tensors: Vec<(String, SharedQTensor)> = quzo_tensors
        .into_iter()
        .filter(|(name, qt)| {
            // Skip expert_mask and expert_remap tensors (pruning metadata, not trainable)
            if name.contains("expert_mask") || name.contains("expert_remap") {
                println!("  Skipping {} (pruning metadata)", name);
                return false;
            }

            let qt_read = qt.read().unwrap();
            let dtype = qt_read.dtype();
            let supported = matches!(
                dtype,
                GgmlDType::F32
                    | GgmlDType::Q2K
                    | GgmlDType::Q3K
                    | GgmlDType::Q4_0
                    | GgmlDType::Q4_1
                    | GgmlDType::Q4K
                    | GgmlDType::Q5_0
                    | GgmlDType::Q5_1
                    | GgmlDType::Q5K
                    | GgmlDType::Q6K
                    | GgmlDType::Q8_0
                    | GgmlDType::Q8_1
                    | GgmlDType::Q8K
            );
            if !supported {
                println!("  Skipping {} ({:?} not supported by QuZO)", name, dtype);
            }
            supported
        })
        .collect();

    if selected_tensors.is_empty() {
        println!("No tensors available for optimization.");
        return Ok(());
    }

    println!(
        "Optimizing {} tensors (all quantized)",
        selected_tensors.len()
    );

    // Compute epsilon multipliers for each tensor based on its name
    let epsilon_multipliers: Vec<f64> = selected_tensors
        .iter()
        .map(|(name, _)| epsilon_config.multiplier_for(name))
        .collect();

    // Extract just the tensors for the optimizer
    let optimizer_tensors: Vec<SharedQTensor> =
        selected_tensors.iter().map(|(_, qt)| qt.clone()).collect();

    // Load tokenizer
    let tokenizer_repo = DEFAULT_TOKENIZER_REPO;
    println!("\nDownloading tokenizer from {}...", tokenizer_repo);
    let api =
        hf_hub::api::sync::Api::new().map_err(|e| paramecia_core::Error::Msg(format!("{e}")))?;
    let repo = api.model(tokenizer_repo.to_string());
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| paramecia_core::Error::Msg(format!("{e}")))?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| paramecia_core::Error::Msg(format!("Tokenizer error: {e}")))?;
    let tokenizer_max_id = tokenizer.get_vocab(true).values().copied().max();
    let model_vocab = model.vocab_size();
    if let Some(max_id) = tokenizer_max_id {
        if (max_id as usize) >= model_vocab {
            return Err(paramecia_core::Error::Msg(format!(
                "Tokenizer token ID range mismatch: tokenizer max token id is {}, but model vocab size is {} (valid ids: 0..{}).",
                max_id,
                model_vocab,
                model_vocab.saturating_sub(1)
            ))
            .bt());
        }
    }

    // Training: given prompt, learn to predict completion
    let prompt = "Rust is a fast,";
    let completion = " safe systems programming language.";
    println!("\nTraining task:");
    println!("  Prompt: \"{}\"", prompt);
    println!("  Target: \"{}\"", completion);

    // Tokenize prompt and completion separately
    let prompt_encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| paramecia_core::Error::Msg(format!("Encoding error: {e}")))?;
    let completion_encoding = tokenizer
        .encode(completion, false) // no special tokens for completion
        .map_err(|e| paramecia_core::Error::Msg(format!("Encoding error: {e}")))?;

    let prompt_tokens: Vec<u32> = prompt_encoding.get_ids().to_vec();
    let completion_tokens: Vec<u32> = completion_encoding.get_ids().to_vec();

    // For next-token prediction:
    // Input: prompt + completion[:-1] (all but last completion token)
    // Target: completion (what we want to predict at each position after prompt)
    let mut input_tokens = prompt_tokens.clone();
    input_tokens.extend(&completion_tokens[..completion_tokens.len() - 1]);

    let input_ids = Tensor::new(input_tokens.as_slice(), &device)?.unsqueeze(0)?;
    let target_ids = Tensor::new(completion_tokens.as_slice(), &device)?.unsqueeze(0)?;
    let prompt_len = prompt_tokens.len();

    println!("Prompt tokens: {:?}", prompt_tokens);
    println!("Completion tokens: {:?}", completion_tokens);

    // Create QuZO optimizer
    let params = ParamsQuZO {
        lr,
        epsilon,
        num_samples,
        clip_threshold: 1.0,
        use_fused: true,
        epsilon_multipliers: Some(epsilon_multipliers),
        lazy_perturbations: false,
        error_feedback: None,
        error_decay: 0.9,
        error_gain: 1.0,
    };

    println!("\nQuZO Parameters:");
    println!("  Learning rate: {}", params.lr);
    println!("  Epsilon: {}", params.epsilon);
    println!("  Samples per step: {}", params.num_samples);
    println!("  Clip threshold: {}", params.clip_threshold);
    println!("  Load balance loss: {}", lb_loss);
    println!("  Z-loss: {}", z_loss);
    println!(
        "  Epsilon multipliers: emb={}, attn={}, gate={}, expert={}, mtp={}, other={}",
        epsilon_config.embedding,
        epsilon_config.attention,
        epsilon_config.moe_gating,
        epsilon_config.moe_experts,
        epsilon_config.mtp,
        epsilon_config.other
    );

    let mut optimizer = QuZO::new_with_seed(optimizer_tensors, params, 42)?;

    // Position to show predictions: last prompt token predicts first completion token
    let predict_position = prompt_len - 1;
    let prompt_display = "Rust is a fast,";

    // Show predictions BEFORE training
    println!("\n=== BEFORE TRAINING ===");
    model.clear_cache();
    let initial_logits = model.forward_all_positions(&input_ids, 0)?;
    let initial_loss =
        compute_completion_loss(&initial_logits, &target_ids, prompt_len)?.to_vec0::<f32>()?;
    println!(
        "Loss: {:.4} (perplexity: {:.2})",
        initial_loss,
        initial_loss.exp()
    );
    show_top_predictions(
        &initial_logits,
        prompt_display,
        &tokenizer,
        predict_position,
        5,
    )?;

    // Training loop with full model forward passes
    println!("\n=== TRAINING ===");
    println!("Running {} QuZO optimization steps...", num_steps);
    println!("Step | Loss     | Perplexity | Time");
    println!("-----|----------|------------|--------");

    let training_start = Instant::now();
    for step in 0..num_steps {
        let step_start = Instant::now();

        // QuZO step: perturb weights, compute loss, update
        // The optimizer calls this closure multiple times (positive/negative perturbations)
        let loss = optimizer.step(|| {
            // Clear cache before each forward pass to avoid KV cache accumulation
            model.clear_cache();

            // Forward pass through the full model (with router stats for aux loss)
            let (logits, router_stats) = model.forward_training(&input_ids, 0)?;

            // Compute loss on completion tokens only
            let ce_loss = compute_completion_loss(&logits, &target_ids, prompt_len)?;

            if lb_loss > 0.0 || z_loss > 0.0 {
                let aux_loss = model.compute_auxiliary_loss(&router_stats, z_loss, lb_loss)?;
                ce_loss + aux_loss
            } else {
                Ok(ce_loss)
            }
        })?;

        // Print progress
        let step_time = step_start.elapsed().as_secs_f32();
        let perplexity = loss.exp();
        println!(
            "{:4} | {:.6} | {:10.2} | {:.2}s",
            step, loss, perplexity, step_time
        );
    }

    let total_time = training_start.elapsed().as_secs_f32();
    println!(
        "\nTotal training time: {:.1}s ({:.2}s/step)",
        total_time,
        total_time / num_steps as f32
    );

    // Show predictions AFTER training
    println!("\n=== AFTER TRAINING ===");
    model.clear_cache();
    let final_logits = model.forward_all_positions(&input_ids, 0)?;
    let final_loss =
        compute_completion_loss(&final_logits, &target_ids, prompt_len)?.to_vec0::<f32>()?;
    println!(
        "Loss: {:.4} (perplexity: {:.2})",
        final_loss,
        final_loss.exp()
    );
    show_top_predictions(
        &final_logits,
        prompt_display,
        &tokenizer,
        predict_position,
        5,
    )?;

    // Summary
    let improvement = (1.0 - final_loss / initial_loss) * 100.0;
    println!("\n=== SUMMARY ===");
    println!("Initial loss: {:.4}", initial_loss);
    println!("Final loss:   {:.4}", final_loss);
    println!("Improvement:  {:.1}%", improvement);

    Ok(())
}

/// Demo mode: runs QuZO on standalone tensors without full model forward
fn run_demo_mode(
    model_path: &PathBuf,
    device: &Device,
    lr: f64,
    epsilon: f64,
    num_steps: usize,
    num_samples: usize,
    epsilon_config: &EpsilonConfig,
) -> Result<()> {
    println!("Running in demo mode (standalone tensor optimization)...\n");

    // Load using MutableVarBuilder for standalone tensor access
    let start_load = Instant::now();
    let vb =
        paramecia_model::quantized_var_builder::MutableVarBuilder::from_gguf(model_path, device)?;
    println!(
        "Model weights loaded in {:.1}s",
        start_load.elapsed().as_secs_f32()
    );

    // Get QuZO-compatible tensors
    let quzo_tensors = vb.quzo_tensors();
    println!(
        "Found {} tensors that support QuZO optimization",
        quzo_tensors.len()
    );

    if quzo_tensors.is_empty() {
        println!("\nNo QuZO-compatible tensors found.");
        return Ok(());
    }

    // Compute epsilon multipliers for each tensor based on its name
    let epsilon_multipliers: Vec<f64> = quzo_tensors
        .iter()
        .map(|(name, _)| epsilon_config.multiplier_for(name))
        .collect();

    // Use all QuZO-compatible tensors
    let selected_tensors: Vec<SharedQTensor> = quzo_tensors.into_iter().map(|(_, qt)| qt).collect();

    println!(
        "\nSelected {} tensors for optimization (all quantized weights)",
        selected_tensors.len()
    );

    // Create optimizer
    let params = ParamsQuZO {
        lr,
        epsilon,
        num_samples,
        clip_threshold: 1.0,
        use_fused: true,
        epsilon_multipliers: Some(epsilon_multipliers),
        lazy_perturbations: false,
        error_feedback: None,
        error_decay: 0.9,
        error_gain: 1.0,
    };

    println!("\nQuZO Parameters:");
    println!("  Learning rate: {}", params.lr);
    println!("  Epsilon: {}", params.epsilon);
    println!(
        "  Epsilon multipliers: emb={}, attn={}, gate={}, expert={}, mtp={}, other={}",
        epsilon_config.embedding,
        epsilon_config.attention,
        epsilon_config.moe_gating,
        epsilon_config.moe_experts,
        epsilon_config.mtp,
        epsilon_config.other
    );

    let mut optimizer = QuZO::new_with_seed(selected_tensors.clone(), params, 42)?;

    // Simple proxy loss: minimize L2 norm of weights
    println!("\nRunning {} QuZO steps (minimizing L2 norm)...", num_steps);
    println!("Step | Loss");
    println!("-----|--------");

    for step in 0..num_steps {
        let loss = optimizer.step(|| {
            let mut total_loss = 0.0f32;
            for qt in &selected_tensors {
                let qt_read = qt.read().unwrap();
                let dequant = qt_read.dequantize(device)?;
                let l2 = dequant.sqr()?.mean_all()?.to_vec0::<f32>()?;
                total_loss += l2;
            }
            Tensor::new(total_loss, device)
        })?;

        if step % 5 == 0 || step == num_steps - 1 {
            println!("{:4} | {:.6}", step, loss);
        }
    }

    println!("\nDemo mode complete!");
    println!("\nRun without --demo for full model training.");

    Ok(())
}
