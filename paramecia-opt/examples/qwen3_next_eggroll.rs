//! EGGROLL fine-tuning example for Qwen3-Next
//!
//! This example demonstrates using EGGROLL to jointly optimize:
//! 1. Quantization scale factors
//! 2. Low-rank adapters (LoRA)
//!
//! Qwen3-Next is a hybrid model featuring:
//! - Full attention layers with gated Q projection
//! - Linear attention layers (Gated Delta Net) for efficient recurrence
//! - Mixture-of-Experts (MoE) FFN with shared experts
//!
//! ## Training Modes
//!
//! - **Loss Mode** (default): Minimizes cross-entropy loss on a target text
//! - **Replicate Mode**: Scores based on cosine similarity + Hirschberg alignment
//!   of generated text against a target completion
//!
//! Usage:
//!   cargo run -p paramecia-opt --example qwen3_next_eggroll --release -- <model.gguf>
//!   cargo run -p paramecia-opt --example qwen3_next_eggroll --release -- <model.gguf> --replicate

use candle::{Device, IndexOp, Result, Tensor};
use hf_hub::api::sync::Api;
use hirschberg::Config as HirschbergConfig;
use paramecia_model::qwen3_next::{DeviceOffloadMode, KvCacheQuantization, ModelWeights};
use paramecia_opt::{Eggroll, EggrollParams, LayerConfig};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

/// Default weight for cosine similarity in Replicate scoring.
const DEFAULT_COSINE_WEIGHT: f32 = 0.6;
/// Default weight for Hirschberg alignment in Replicate scoring.
const DEFAULT_HIRSCHBERG_WEIGHT: f32 = 0.4;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let x = f64::from(x);
        let y = f64::from(y);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (dot / denom) as f32
    }
}

fn hirschberg_alignment_score(
    a: &str,
    b: &str,
    match_score: i32,
    mismatch_score: i32,
    gap_score: i32,
) -> f32 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let max_len = a_chars.len().max(b_chars.len());
    if max_len == 0 {
        return 0.0;
    }

    let config = HirschbergConfig {
        match_score,
        mismatch_score,
        gap_score,
    };
    let score = config.compute(&a_chars, &b_chars).score();

    // Normalize to roughly [-1, 1] by dividing by the best-possible score for the
    // longer sequence (perfect match, no gaps).
    let denom = (match_score.max(1) as f32) * (max_len as f32);
    (score as f32) / denom
}

fn compute_cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<f32> {
    let (batch_size, seq_len, vocab_size) = logits.dims3()?;
    let logits_flat = logits.reshape((batch_size * seq_len, vocab_size))?;
    let targets_flat = targets.reshape(batch_size * seq_len)?;
    let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
    loss.to_scalar::<f32>()
}

/// Compute Replicate score: weighted sum of cosine similarity and Hirschberg alignment.
fn compute_replicate_score(
    generated_text: &str,
    target_text: &str,
    generated_embedding: &[f32],
    target_embedding: &[f32],
    cosine_weight: f32,
    hirschberg_weight: f32,
) -> f32 {
    let cosine_sim = cosine_similarity(generated_embedding, target_embedding);
    let hirschberg = hirschberg_alignment_score(generated_text, target_text, 2, -1, -1);
    cosine_weight * cosine_sim + hirschberg_weight * hirschberg
}

/// Result of text generation.
struct GenerationResult {
    text: String,
    tokens: Vec<u32>,
}

/// Generate text from a prompt using greedy sampling.
fn generate_text(
    model: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    device: &Device,
) -> Result<GenerationResult> {
    let mut generated_tokens = prompt_tokens.to_vec();

    for _ in 0..max_new_tokens {
        let input = Tensor::new(&generated_tokens[..], device)?.unsqueeze(0)?;
        model.clear_kv_cache();
        let logits = model.forward(&input, 0)?;

        // Greedy sampling: take argmax
        let next_token = logits.argmax(candle::D::Minus1)?.to_vec1::<u32>()?[0];

        // Check for EOS (assuming token 151643 is EOS for Qwen)
        if next_token == 151643 {
            break;
        }

        generated_tokens.push(next_token);
    }

    let text = tokenizer
        .decode(&generated_tokens, true)
        .map_err(|e| candle::Error::Msg(format!("Decode error: {}", e)))?;

    Ok(GenerationResult {
        text,
        tokens: generated_tokens,
    })
}

/// Show top predictions from logits
fn show_top_predictions(logits: &Tensor, prompt: &str, tokenizer: &Tokenizer) -> Result<()> {
    let logits_2d = logits.squeeze(0)?; // [seq_len, vocab_size]
    let seq_len = logits_2d.dim(0)?;

    // Get last position predictions
    let logits_last = logits_2d.i(seq_len - 1)?;
    let probs = candle_nn::ops::softmax(&logits_last, candle::D::Minus1)?;
    let probs_vec = probs.to_vec1::<f32>()?;

    let mut indexed: Vec<_> = probs_vec.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(a.1));

    print!("  Input: \"{}\"\n  Top predictions: ", prompt);
    for (i, (token_id, prob)) in indexed.iter().copied().take(3).enumerate() {
        if let Ok(decoded) = tokenizer.decode(&[token_id as u32], false) {
            print!("'{}' ({:.1}%)", decoded.trim(), prob * 100.0);
            if i < 2 {
                print!(", ");
            }
        }
    }
    println!();

    Ok(())
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let model_path = if args.len() > 1 && !args[1].starts_with("--") {
        PathBuf::from(&args[1])
    } else {
        eprintln!("Usage: {} <path-to-gguf-model> [options]", args[0]);
        eprintln!("\nOptions:");
        eprintln!("  --replicate      Use Replicate mode (cosine + hirschberg scoring)");
        eprintln!("  --lora-only      Only optimize LoRA adapters (no scales)");
        eprintln!("  --scales-only    Only optimize scales (no LoRA)");
        eprintln!("  --attn-only      Only optimize attention layers (not FFN/MoE)");
        eprintln!("  --moe-only       Only optimize MoE expert layers");
        eprintln!("  --population N   Population size (default: 8)");
        eprintln!("  --rank N         LoRA rank (default: 1)");
        eprintln!("  --generations N  Number of generations (default: 20)");
        eprintln!("  --sigma F        Perturbation magnitude (default: 0.001)");
        eprintln!("  --lr F           Learning rate (default: 0.0001)");
        eprintln!("  --cosine-weight F  Weight for cosine similarity (default: 0.6)");
        eprintln!("  --hirschberg-weight F  Weight for Hirschberg alignment (default: 0.4)");
        eprintln!("  --no-kv-quant    Disable KV-cache quantization");
        eprintln!(
            "  --offload MODE   Device offload: none, up, updown, experts (default: experts)"
        );
        eprintln!("\nExamples:");
        eprintln!(
            "  {} /path/to/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf",
            args[0]
        );
        eprintln!("  {} /path/to/qwen3-next.gguf --replicate", args[0]);
        eprintln!(
            "  {} /path/to/qwen3-next.gguf --attn-only --rank 4",
            args[0]
        );
        std::process::exit(1);
    };

    let replicate_mode = args.contains(&"--replicate".to_string());
    let optimize_lora = !args.contains(&"--scales-only".to_string());
    let optimize_scales = !args.contains(&"--lora-only".to_string());
    let attn_only = args.contains(&"--attn-only".to_string());
    let moe_only = args.contains(&"--moe-only".to_string());
    let no_kv_quant = args.contains(&"--no-kv-quant".to_string());
    let offload_mode =
        parse_string_arg(&args, "--offload").unwrap_or_else(|| "experts".to_string());

    // Parse optional numeric arguments
    let population_size = parse_arg(&args, "--population").unwrap_or(4);
    let rank = parse_arg(&args, "--rank").unwrap_or(1);
    let num_generations = parse_arg(&args, "--generations").unwrap_or(10);
    // Higher sigma and learning rate for replicate mode
    let default_sigma = if replicate_mode { 0.01 } else { 0.001 };
    let default_lr = if replicate_mode { 0.001 } else { 0.0001 };
    let sigma: f64 = parse_float_arg(&args, "--sigma").unwrap_or(default_sigma);
    let lr: f64 = parse_float_arg(&args, "--lr").unwrap_or(default_lr);
    let cosine_weight: f32 = parse_float_arg(&args, "--cosine-weight")
        .map(|v| v as f32)
        .unwrap_or(DEFAULT_COSINE_WEIGHT);
    let hirschberg_weight: f32 = parse_float_arg(&args, "--hirschberg-weight")
        .map(|v| v as f32)
        .unwrap_or(DEFAULT_HIRSCHBERG_WEIGHT);

    println!("EGGROLL Fine-tuning for Qwen3-Next");
    println!("===================================");
    println!("Hybrid Architecture: Full Attention + Linear Attention (Gated Delta Net)");
    println!("with Mixture-of-Experts FFN\n");
    if replicate_mode {
        println!("Training Mode: REPLICATE (cosine similarity + Hirschberg alignment)");
        println!("  Cosine weight: {:.2}", cosine_weight);
        println!("  Hirschberg weight: {:.2}", hirschberg_weight);
    } else {
        println!("Training Mode: LOSS (cross-entropy minimization)");
    }
    println!("Optimize LoRA: {}", optimize_lora);
    println!("Optimize Scales: {}", optimize_scales);
    println!("Attention only: {}", attn_only);
    println!("MoE only: {}", moe_only);
    println!("Population size: {}", population_size);
    println!("LoRA rank: {}", rank);
    println!("Generations: {}", num_generations);
    println!();

    // Download tokenizer
    let tokenizer_repo = "Qwen/Qwen3-Next-80B-A3B-Instruct";
    println!("Downloading tokenizer from {}...", tokenizer_repo);
    let tokenizer_path = download_file(tokenizer_repo, "tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| candle::Error::Msg(format!("Tokenizer error: {}", e)))?;
    println!("Tokenizer loaded\n");

    // Load model
    let device = get_best_device()?;
    println!("Using device: {:?}", device);
    println!("Loading model from {:?}...", model_path);

    let kv_cache_quant = if no_kv_quant {
        println!("KV-cache quantization: DISABLED (using F16)");
        KvCacheQuantization::F16
    } else {
        println!("KV-cache quantization: Q8_0 (8-bit)");
        KvCacheQuantization::Q8_0
    };

    let offload = match offload_mode.as_str() {
        "none" => {
            println!("Offload: none (all weights on GPU)");
            DeviceOffloadMode::FullGpu
        }
        "up" => {
            println!("Offload: up (up projections on CPU)");
            DeviceOffloadMode::UpProjectionsOnCpu
        }
        "experts" => {
            println!("Offload: experts (all MoE experts on CPU)");
            DeviceOffloadMode::ExpertsOnCpu
        }
        "updown" => {
            println!("Offload: updown (up+down projections on CPU, gate on GPU)");
            DeviceOffloadMode::UpDownProjectionsOnCpu
        }
        _ => {
            println!("Offload: experts (all MoE experts on CPU)");
            DeviceOffloadMode::ExpertsOnCpu
        }
    };

    let start_load = Instant::now();
    let mut model =
        ModelWeights::from_gguf_with_offload_mode(&model_path, &device, offload, kv_cache_quant)?;
    println!(
        "Model loaded in {:.1}s with {} layers\n",
        start_load.elapsed().as_secs_f32(),
        model.num_layers()
    );

    // Training data
    let training_text = "Rust is a fast, safe systems programming language.";

    // For Replicate mode: prompt and target
    let replicate_prompt = "Rust is";
    let replicate_target = "Rust is a fast, safe systems programming language";

    // Tokenize prompt for Replicate mode
    let prompt_encoding = tokenizer
        .encode(replicate_prompt, false)
        .map_err(|e| candle::Error::Msg(format!("Encoding error: {}", e)))?;
    let prompt_tokens = prompt_encoding.get_ids().to_vec();

    // Get target embedding (computed once, outside the loop)
    let target_encoding = tokenizer
        .encode(replicate_target, false)
        .map_err(|e| candle::Error::Msg(format!("Encoding error: {}", e)))?;
    let target_token_ids = target_encoding.get_ids().to_vec();
    let target_input = Tensor::new(&target_token_ids[..], &device)?.unsqueeze(0)?;

    model.clear_kv_cache();
    let target_embedding = model.forward_embeddings_last(&target_input)?;
    let target_embedding_vec: Vec<f32> = target_embedding.squeeze(0)?.to_vec1()?;

    // Get layer configurations from model
    let model_configs = model.eggroll_layer_configs();

    // Convert to EGGROLL layer configs
    // Filter based on command line options
    let layer_configs: Vec<LayerConfig> = model_configs
        .iter()
        .filter(|(name, _, num_blocks)| {
            // For scale optimization, only include layers that support scale modification
            // For LoRA-only mode, we can include all attention layers
            if optimize_scales && !optimize_lora && num_blocks.is_none() {
                return false;
            }

            // Filter by layer type
            if attn_only {
                // Full attention: attn_q, attn_k, attn_v, attn_o
                // Linear attention: ssm_in, ssm_ba, ssm_out
                name.contains("attn") || name.contains("ssm")
            } else if moe_only {
                // MoE layers: ffn_gate_exps, ffn_up_exps, ffn_down_exps
                name.contains("ffn") && name.contains("exps")
            } else {
                // Include all: attention, ssm, router, experts, shared experts, output
                true
            }
        })
        .filter(|(name, _, _)| {
            // For LoRA, only include attention layers (full and linear)
            if optimize_lora && !optimize_scales {
                name.contains("attn") || name.contains("ssm")
            } else {
                true
            }
        })
        .map(|(name, shape, num_scale_blocks)| {
            let is_mtp = LayerConfig::is_mtp_layer(name);
            LayerConfig {
                name: name.clone(),
                shape: *shape,
                num_scale_blocks: if optimize_scales {
                    *num_scale_blocks
                } else {
                    None
                },
                optimize_scales,
                optimize_lora: optimize_lora && (name.contains("attn") || name.contains("ssm")),
                lr_multiplier: 1.0,
                is_mtp,
            }
        })
        .collect();

    if layer_configs.is_empty() {
        println!("No layers available for optimization.");
        println!("Note: Only K-quant formats (Q2K-Q8K) support scale modification.");
        return Ok(());
    }

    println!(
        "Configuring {} layers for EGGROLL optimization:",
        layer_configs.len()
    );
    for config in layer_configs.iter().take(10) {
        println!(
            "  {} - shape {:?}, scales: {:?}, lora: {}",
            config.name, config.shape, config.num_scale_blocks, config.optimize_lora
        );
    }
    if layer_configs.len() > 10 {
        println!("  ... and {} more layers", layer_configs.len() - 10);
    }
    println!();

    // EGGROLL parameters
    let params = EggrollParams {
        lr,
        sigma,
        sigma_min: 1e-5,
        sigma_decay: 0.95,               // Faster decay for quicker convergence
        adaptive_sigma: !replicate_mode, // Disable adaptive sigma for replicate mode
        rank,
        population_size,
        antithetic: true,
        scale_lr: Some(lr * 10.0), // Scales can use higher LR
        momentum: 0.9,
        rank_based_fitness: true,
        base_seed: 0,
    };

    println!("EGGROLL Parameters:");
    println!("  Learning rate: {}", params.lr);
    println!("  Scale LR: {:?}", params.scale_lr);
    println!("  Sigma: {}", params.sigma);
    println!("  LoRA rank: {}", params.rank);
    println!("  Population size: {}", params.population_size);
    println!("  Antithetic sampling: {}", params.antithetic);
    println!("  Momentum: {}", params.momentum);
    println!();

    // Create optimizer
    let mut optimizer = Eggroll::new(params.clone(), layer_configs.clone(), &device)?;

    // Tokenize training data
    let encoding = tokenizer
        .encode(training_text, true)
        .map_err(|e| candle::Error::Msg(format!("Encoding error: {}", e)))?;
    let tokens = encoding.get_ids().to_vec();

    if tokens.len() < 2 {
        return Err(candle::Error::Msg("Training text too short".to_string()));
    }

    let input_ids = Tensor::new(&tokens[..tokens.len() - 1], &device)?.unsqueeze(0)?;
    let target_ids = Tensor::new(&tokens[1..], &device)?.unsqueeze(0)?;

    // Compute initial loss and get predictions BEFORE training
    model.clear_kv_cache();
    let (initial_logits, _router_stats) = model.forward_training(&input_ids, 0)?;
    let initial_loss = compute_cross_entropy_loss(&initial_logits, &target_ids)?;

    println!("=== BEFORE TRAINING ===");
    println!("Loss: {:.5} (PPL: {:.2})", initial_loss, initial_loss.exp());
    show_top_predictions(&initial_logits, "Rust is a fast,", &tokenizer)?;
    println!();

    // Training loop
    if replicate_mode {
        println!(
            "Replicate training: prompt=\"{}\" → target=\"{}\"",
            replicate_prompt, replicate_target
        );
    } else {
        println!("Training on: \"{}\"", training_text);
    }
    println!();

    if replicate_mode {
        println!("Gen  | Mean Fit | Best Fit | Sigma    | Time(s) | Best Text");
        println!("-----|----------|----------|----------|---------|----------");
    } else {
        println!("Gen  | Mean Fit | Best Fit | Sigma    | Time(s)");
        println!("-----|----------|----------|----------|--------");
    }

    let mut best_loss = initial_loss;
    let mut best_score = 0.0f32;
    let mut best_generated_text = String::new();

    for gen in 0..num_generations {
        let gen_start = Instant::now();

        // Sample new population
        optimizer.sample_population()?;

        // Evaluate each population member
        for member_idx in 0..optimizer.population_size() {
            // Get scales for this member
            let scales = optimizer.get_member_scales(member_idx)?;
            model.set_all_custom_scales(scales)?;

            // Get LoRA adapters for this member (mean + perturbation)
            if optimize_lora {
                let mut lora_map = HashMap::new();
                for config in &layer_configs {
                    if config.optimize_lora {
                        if let Some((a, b)) = optimizer.get_member_lora(member_idx, &config.name)? {
                            lora_map.insert(config.name.clone(), (a, b));
                        }
                    }
                }
                // Pass sigma for proper scaling: σ/√r * A @ B^T
                model.set_lora_adapters(&lora_map, optimizer.sigma())?;
            }

            if replicate_mode {
                // Replicate mode: generate text and score
                let max_new_tokens = target_token_ids.len().saturating_sub(prompt_tokens.len()) + 5;
                let gen_result = generate_text(
                    &mut model,
                    &tokenizer,
                    &prompt_tokens,
                    max_new_tokens,
                    &device,
                )?;

                // Get embedding for generated text
                let gen_input = Tensor::new(&gen_result.tokens[..], &device)?.unsqueeze(0)?;
                model.clear_kv_cache();
                let gen_embedding = model.forward_embeddings_last(&gen_input)?;
                let gen_embedding_vec: Vec<f32> = gen_embedding.squeeze(0)?.to_vec1()?;

                // Compute Replicate score
                let score = compute_replicate_score(
                    &gen_result.text,
                    replicate_target,
                    &gen_embedding_vec,
                    &target_embedding_vec,
                    cosine_weight,
                    hirschberg_weight,
                );

                // Set fitness (positive score since we maximize)
                optimizer.population_mut()[member_idx].fitness = Some(score);

                if score > best_score {
                    best_score = score;
                    best_generated_text = gen_result.text.clone();
                }

            } else {
                // Loss mode: compute cross-entropy loss
                model.clear_kv_cache();

                let (logits, _router_stats) = model.forward_training(&input_ids, 0)?;
                let loss = compute_cross_entropy_loss(&logits, &target_ids)?;

                // Set fitness (negative loss since we minimize)
                optimizer.population_mut()[member_idx].fitness = Some(-loss);

                if loss < best_loss {
                    best_loss = loss;
                }
            }
        }

        // Clear perturbations after evaluation
        if optimize_lora {
            model.clear_lora_adapters();
        }

        // Get best fitness before step
        let best_fitness = optimizer
            .population()
            .iter()
            .filter_map(|m| m.fitness)
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0);

        // Perform EGGROLL update
        let mean_fitness = optimizer.step()?;
        let gen_time = gen_start.elapsed().as_secs_f32();

        if replicate_mode {
            // Truncate best text for display
            let display_text = if best_generated_text.len() > 40 {
                format!("{}...", &best_generated_text[..40])
            } else {
                best_generated_text.clone()
            };
            println!(
                "{:4} | {:.5} | {:.5} | {:.6} | {:7.2} | {}",
                gen,
                mean_fitness,
                best_fitness,
                optimizer.sigma(),
                gen_time,
                display_text.replace('\n', " ")
            );
        } else {
            println!(
                "{:4} | {:.5} | {:.5} | {:.6} | {:7.2}",
                gen,
                -mean_fitness,
                -best_fitness,
                optimizer.sigma(),
                gen_time
            );
        }
    }

    // Apply final mean parameters
    model.set_all_custom_scales(optimizer.mean_scales().clone())?;

    // Apply mean LoRA if optimizing
    if optimize_lora {
        let mut lora_map = HashMap::new();
        for (name, lora) in optimizer.mean_lora() {
            lora_map.insert(name.clone(), (lora.a.clone(), lora.b.clone()));
        }
        if !lora_map.is_empty() {
            model.set_lora_adapters(&lora_map, 1.0)?; // sigma=1 for mean LoRA
        }
    }

    // Compute final loss and get predictions AFTER training
    model.clear_kv_cache();
    let (final_logits, _router_stats) = model.forward_training(&input_ids, 0)?;
    let final_loss = compute_cross_entropy_loss(&final_logits, &target_ids)?;

    println!();
    println!("=== AFTER TRAINING ===");

    if replicate_mode {
        // For Replicate mode, generate final text and show score
        let max_new_tokens = target_token_ids.len().saturating_sub(prompt_tokens.len()) + 5;
        let final_result = generate_text(
            &mut model,
            &tokenizer,
            &prompt_tokens,
            max_new_tokens,
            &device,
        )?;

        let gen_input = Tensor::new(&final_result.tokens[..], &device)?.unsqueeze(0)?;
        model.clear_kv_cache();
        let gen_embedding = model.forward_embeddings_last(&gen_input)?;
        let gen_embedding_vec: Vec<f32> = gen_embedding.squeeze(0)?.to_vec1()?;

        let final_score = compute_replicate_score(
            &final_result.text,
            replicate_target,
            &gen_embedding_vec,
            &target_embedding_vec,
            cosine_weight,
            hirschberg_weight,
        );
        let final_cosine = cosine_similarity(&gen_embedding_vec, &target_embedding_vec);
        let final_hirschberg =
            hirschberg_alignment_score(&final_result.text, replicate_target, 2, -1, -1);

        println!("Final generation: \"{}\"", final_result.text);
        println!("Target:           \"{}\"", replicate_target);
        println!();
        println!("Final Replicate Score: {:.5}", final_score);
        println!("  Cosine Similarity:   {:.5}", final_cosine);
        println!("  Hirschberg Alignment: {:.5}", final_hirschberg);
        println!();
        println!("Summary:");
        println!("  Best score:   {:.5}", best_score);
        println!("  Best text:    \"{}\"", best_generated_text);
    } else {
        println!("Loss: {:.5} (PPL: {:.2})", final_loss, final_loss.exp());
        show_top_predictions(&final_logits, "Rust is a fast,", &tokenizer)?;
        println!();

        println!("Summary:");
        println!(
            "  Initial loss: {:.5} (PPL: {:.2})",
            initial_loss,
            initial_loss.exp()
        );
        println!(
            "  Final loss:   {:.5} (PPL: {:.2})",
            final_loss,
            final_loss.exp()
        );
        println!(
            "  Best loss:    {:.5} (PPL: {:.2})",
            best_loss,
            best_loss.exp()
        );
        println!(
            "  Improvement:  {:.2}%",
            (1.0 - final_loss / initial_loss) * 100.0
        );
    }
    println!("  Final sigma:  {:.6}", optimizer.sigma());

    // Print scale statistics
    let mean_scales = optimizer.mean_scales();
    if !mean_scales.is_empty() && optimize_scales {
        println!("\nScale Modification Statistics:");
        let mut total_modified = 0usize;
        let mut total_blocks = 0usize;
        let mut max_deviation = 0.0f32;

        for (name, scales) in mean_scales.iter() {
            let scales_vec = scales.to_vec1::<f32>()?;
            let num_blocks = scales_vec.len();
            total_blocks += num_blocks;

            for &s in &scales_vec {
                let deviation = (s - 1.0).abs();
                if deviation > 0.001 {
                    total_modified += 1;
                }
                if deviation > max_deviation {
                    max_deviation = deviation;
                }
            }

            // Show first layer stats as example
            if name.contains("blk.0") {
                let mean_scale: f32 = scales_vec.iter().sum::<f32>() / num_blocks as f32;
                let min_scale = scales_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_scale = scales_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                println!(
                    "  {} - mean: {:.4}, range: [{:.4}, {:.4}]",
                    name, mean_scale, min_scale, max_scale
                );
            }
        }

        println!(
            "  Total: {} / {} blocks modified (deviation > 0.1%)",
            total_modified, total_blocks
        );
        println!("  Max deviation from 1.0: {:.4}", max_deviation);
    }

    // Print LoRA statistics
    let mean_lora = optimizer.mean_lora();
    if !mean_lora.is_empty() && optimize_lora {
        println!("\nLoRA Adapter Statistics:");
        println!("  {} adapters learned", mean_lora.len());
        for (name, lora) in mean_lora.iter().take(3) {
            let a_norm = lora.a.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            let b_norm = lora.b.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            println!(
                "  {} - rank: {}, ||A||: {:.4}, ||B||: {:.4}",
                name, lora.rank, a_norm, b_norm
            );
        }
        if mean_lora.len() > 3 {
            println!("  ... and {} more adapters", mean_lora.len() - 3);
        }
    }

    Ok(())
}

fn download_file(repo_id: &str, filename: &str) -> Result<PathBuf> {
    let api = Api::new().map_err(|e| candle::Error::Msg(format!("HF API: {}", e)))?;
    let repo = api.model(repo_id.to_string());
    repo.get(filename)
        .map_err(|e| candle::Error::Msg(format!("Download: {}", e)))
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

fn parse_arg(args: &[String], flag: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn parse_float_arg(args: &[String], flag: &str) -> Option<f64> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn parse_string_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
