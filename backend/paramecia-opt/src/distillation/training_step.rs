//! Online training step logic for use by the controller's training actor.
//!
//! Provides `TrainingState` (training session container) and
//! `run_training_step_with_grad_accum` (core QuZO gradient-accumulation loop).

use super::{DistillationLoss, DistillationLossConfig, EpsilonConfig, MtpLossConfig};
use crate::{OptimizeTensors, QuZO, TuningData};

use paramecia_model::models::qwen3_next::ModelWeights;
use std::path::PathBuf;
use tokenizers::Tokenizer;

/// Training state for an online distillation session.
pub struct TrainingState {
    pub model: Option<ModelWeights>,
    pub device: paramecia_core::Device,
    pub tokenizer: Tokenizer,
    pub original_model_path: PathBuf,
    pub checkpoint_dir: PathBuf,
    pub minibatch_size: usize,
    pub n_grad_steps: usize,
    pub lr: f64,
    pub epsilon: f64,
    pub optimize_tensors: OptimizeTensors,
    pub loss_config: DistillationLossConfig,
    pub epsilon_config: EpsilonConfig,
    pub mtp_config: Option<MtpLossConfig>,
    pub step: usize,
    pub best_loss: f64,
}

/// Load a tokenizer — tries local path first, falls back to HF download.
pub fn load_tokenizer_for_training(model_path: &std::path::Path) -> Tokenizer {
    // Try to find tokenizer.json next to the model file
    if let Some(parent) = model_path.parent() {
        let tokenizer_path = parent.join("tokenizer.json");
        if tokenizer_path.exists() {
            if let Ok(t) = Tokenizer::from_file(&tokenizer_path) {
                return t;
            }
        }
    }
    // Try HF download
    if let Ok(api) = hf_hub::api::sync::Api::new() {
        let repo = api.repo(hf_hub::Repo::with_revision(
            "Qwen/Qwen3-Next-80B-A3B-Instruct".to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));
        if let Ok(path) = repo.get("tokenizer.json") {
            if let Ok(t) = Tokenizer::from_file(path) {
                return t;
            }
        }
    }
    tracing::warn!("Could not load tokenizer for training — context segments will be ignored");
    Tokenizer::from_bytes(b"{}").unwrap_or_else(|_| panic!("Failed to create fallback tokenizer"))
}

/// Parse an optimize-tensors mode from a CLI string.
pub fn parse_optimize_tensors(s: &str) -> OptimizeTensors {
    match s {
        "attention" => OptimizeTensors::AttentionOnly,
        "qk" => OptimizeTensors::AttentionQKOnly,
        _ => OptimizeTensors::All,
    }
}

/// Run a training step with gradient accumulation.
///
/// Drains `n_grad_steps * minibatch_size` samples from the buffer (or fewer if
/// the buffer is smaller). Each grad step runs one QuZO perturbation on a
/// minibatch of conversations. Loss is accumulated and averaged.
#[allow(clippy::too_many_arguments)]
pub fn run_training_step_with_grad_accum(
    model: &mut ModelWeights,
    optimizer: &mut Option<QuZO>,
    loss_fn: &DistillationLoss,
    sample_buffer: &mut Vec<TuningData>,
    device: &paramecia_core::Device,
    is_validation: bool,
    minibatch_size: usize,
    n_grad_steps: usize,
    mtp_config: Option<&MtpLossConfig>,
) -> Result<f64, String> {
    let mut accumulated_loss = 0.0f64;
    let mut accumulated_positions = 0usize;

    for _grad_step in 0..n_grad_steps {
        let drain_end = minibatch_size.min(sample_buffer.len());
        if drain_end == 0 {
            break;
        }
        let batch: Vec<TuningData> = sample_buffer.drain(..drain_end).collect();

        let positions_count: usize = batch.iter().map(|c| c.n_assistant_tokens()).sum();
        if positions_count == 0 {
            continue;
        }

        model.clear_cache();

        if is_validation || optimizer.is_none() {
            let loss = compute_batch_loss(model, loss_fn, &batch, device, mtp_config)
                .map_err(|e| format!("Loss computation failed: {e}"))?;
            let loss_val = loss
                .to_scalar::<f32>()
                .map_err(|e| format!("Loss scalar extraction failed: {e}"))?;
            accumulated_loss += loss_val as f64 * positions_count as f64;
            accumulated_positions += positions_count;
        } else {
            let optimizer = optimizer.as_mut().ok_or_else(|| {
                "Optimizer unavailable for non-validation training step".to_string()
            })?;
            let loss_val = optimizer
                .step(|| compute_batch_loss(model, loss_fn, &batch, device, mtp_config))
                .map_err(|e| format!("Optimizer step failed: {e}"))?;
            accumulated_loss += loss_val as f64 * positions_count as f64;
            accumulated_positions += positions_count;
        }
    }

    if accumulated_positions > 0 {
        Ok(accumulated_loss / accumulated_positions as f64)
    } else {
        Ok(0.0)
    }
}

/// Compute aggregate loss over a minibatch of conversations.
fn compute_batch_loss(
    model: &mut ModelWeights,
    loss_fn: &DistillationLoss,
    batch: &[TuningData],
    device: &paramecia_core::Device,
    mtp_config: Option<&MtpLossConfig>,
) -> paramecia_core::Result<paramecia_core::Tensor> {
    use paramecia_core::Tensor;

    if batch.is_empty() {
        return Tensor::new(0.0f32, device);
    }

    model.clear_cache();

    let mut total_loss = Tensor::new(0.0f32, device)?;
    let mut total_positions = 0usize;

    let use_mtp = mtp_config.is_some() && model.has_mtp();
    let embed_weights = if use_mtp {
        Some(model.embedding_weights().clone())
    } else {
        None
    };

    for conversation in batch {
        if conversation.n_assistant_tokens() == 0 {
            continue;
        }

        let assistant_positions: Vec<(usize, usize)> = conversation
            .assistant_indices()
            .iter()
            .enumerate()
            .map(|(i, &pos)| (pos, i))
            .collect();

        let loss = if let (Some(config), Some(weights)) = (mtp_config, embed_weights.as_ref()) {
            let input_ids = conversation
                .input_ids_tensor(device)
                .map_err(|e| paramecia_core::Error::Msg(format!("input_ids_tensor: {e}")))?
                .unsqueeze(0)?;
            let weighted_embeds = compute_weighted_embeddings_for_mtp(
                conversation,
                &assistant_positions,
                config.num_depths,
                weights,
            )?;
            let (main_logits, router_stats, mtp_logits) = model
                .forward_training_with_mtp_weighted(
                    &input_ids,
                    0,
                    &weighted_embeds,
                    config.num_depths,
                )?;
            loss_fn.compute_mtp_loss(
                &main_logits,
                &mtp_logits,
                &router_stats,
                conversation,
                &assistant_positions,
                config,
                device,
            )?
        } else {
            let input_ids = conversation
                .input_ids_tensor(device)
                .map_err(|e| paramecia_core::Error::Msg(format!("input_ids_tensor: {e}")))?
                .unsqueeze(0)?;
            let (logits, router_stats) = model.forward_training(&input_ids, 0)?;
            loss_fn.compute_total_loss(
                &logits,
                &router_stats,
                conversation,
                &assistant_positions,
                device,
            )?
        };

        let n_positions = assistant_positions.len();
        let weighted = (&loss * n_positions as f64)?;
        total_loss = (&total_loss + &weighted)?;
        total_positions += n_positions;

        model.clear_cache();
    }

    if total_positions > 0 {
        total_loss = (&total_loss / total_positions as f64)?;
    }

    Ok(total_loss)
}

fn compute_weighted_embeddings_for_mtp(
    conversation: &TuningData,
    assistant_positions: &[(usize, usize)],
    num_depths: usize,
    embed_weights: &paramecia_core::Tensor,
) -> paramecia_core::Result<Vec<paramecia_core::Tensor>> {
    let mut weighted_embeds_per_depth = Vec::with_capacity(num_depths);

    for depth in 0..num_depths {
        let embed_shift = depth + 1;
        let shifted_indices: Vec<usize> = assistant_positions
            .iter()
            .filter_map(|&(_pos, data_idx)| {
                let shifted_idx = data_idx + embed_shift;
                if shifted_idx < conversation.n_assistant_tokens() {
                    Some(shifted_idx)
                } else {
                    None
                }
            })
            .collect();

        if shifted_indices.is_empty() {
            let hidden_dim = embed_weights.dim(1)?;
            let empty = paramecia_core::Tensor::zeros(
                (0, hidden_dim),
                embed_weights.dtype(),
                embed_weights.device(),
            )?;
            weighted_embeds_per_depth.push(empty);
            continue;
        }

        let embeds =
            conversation.compute_weighted_embeddings_batch(&shifted_indices, embed_weights)?;
        weighted_embeds_per_depth.push(embeds);
    }

    Ok(weighted_embeds_per_depth)
}
