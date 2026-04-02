//! Conversion bridge between engine training types and paramecia-opt TuningData.

use crate::types::*;
use std::path::PathBuf;

/// Convert a training sample's data segments to paramecia-opt TuningData.
///
/// Walks interleaved context/target/raw segments:
/// - context(model-input) → dispatch by input type:
///   - Text(string) → tokenize → extend token_ids with assistant_mask=false
///   - Tokens(list<u32>) → extend token_ids with assistant_mask=false
///   - Soft(_) → not supported in training context (returns error)
///   - Raw(list<f32>) → validate and use argmax token as context (assistant_mask=false)
/// - target(list<predicted>) → each predicted adds one token with assistant_mask=true
/// - raw(list<f32>) → one assistant token with full-vocab teacher probabilities
pub fn sample_to_tuning_data(
    sample: &[TrainingData],
    tokenizer: &tokenizers::Tokenizer,
) -> Result<paramecia_opt::TuningData, String> {
    let mut token_ids = Vec::new();
    let mut assistant_mask = Vec::new();
    let mut top_k_token_ids = Vec::new();
    let mut top_k_log_probs = Vec::new();
    let mut tail_token_ids = Vec::new();
    let mut tail_log_probs = Vec::new();
    let mut tail_mass_vec = Vec::new();
    let mut raw_probs = Vec::new();
    let mut expert_indices = Vec::new();

    for segment in sample {
        match segment {
            TrainingData::Context(input) => match input {
                ModelInput::Text(text) => {
                    let encoding = tokenizer
                        .encode(text.as_str(), false)
                        .map_err(|e| format!("Tokenization failed: {e}"))?;
                    let ids = encoding.get_ids();
                    token_ids.extend_from_slice(ids);
                    assistant_mask.extend(std::iter::repeat_n(false, ids.len()));
                }
                ModelInput::Tokens(ids) => {
                    token_ids.extend_from_slice(ids);
                    assistant_mask.extend(std::iter::repeat_n(false, ids.len()));
                }
                ModelInput::Soft(_) => {
                    return Err(
                        "Soft prompts are not supported in training context segments".to_string(),
                    );
                }
                ModelInput::Raw(raw_dist) => {
                    let (argmax_idx, _) = validate_raw_distribution(
                        raw_dist,
                        "Raw probability input in training context segment",
                    )?;
                    token_ids.push(argmax_idx as u32);
                    assistant_mask.push(false);
                }
            },
            TrainingData::Target(predictions) => {
                for pred in predictions {
                    token_ids.push(pred.token_id);
                    assistant_mask.push(true);

                    let tk_ids: Vec<u32> = pred.top_k.iter().map(|e| e.token_id).collect();
                    let tk_lps: Vec<f32> = pred.top_k.iter().map(|e| e.log_prob).collect();
                    top_k_token_ids.push(tk_ids);
                    top_k_log_probs.push(tk_lps);

                    let tl_ids: Vec<u32> = pred.tail.iter().map(|e| e.token_id).collect();
                    let tl_lps: Vec<f32> = pred.tail.iter().map(|e| e.log_prob).collect();
                    tail_token_ids.push(tl_ids);
                    tail_log_probs.push(tl_lps);

                    tail_mass_vec.push(pred.tail_mass);

                    raw_probs.push(None);
                    expert_indices.push(pred.expert_indices.clone());
                }
            }
            TrainingData::Raw(raw_dist) => {
                let (argmax_idx, sum) =
                    validate_raw_distribution(raw_dist, "Raw training distribution")?;

                token_ids.push(argmax_idx as u32);
                assistant_mask.push(true);

                // Raw targets supervise this position with full-vocab KL.
                // Keep sparse fields aligned with assistant positions.
                top_k_token_ids.push(Vec::new());
                top_k_log_probs.push(Vec::new());
                tail_token_ids.push(Vec::new());
                tail_log_probs.push(Vec::new());
                tail_mass_vec.push(0.0);

                let mut normalized = raw_dist.clone();
                for p in &mut normalized {
                    *p /= sum;
                }
                raw_probs.push(Some(normalized));
                expert_indices.push(Vec::new());
            }
        }
    }

    let n_assistant_tokens = assistant_mask.iter().filter(|&&is_asst| is_asst).count();
    if top_k_token_ids.len() != n_assistant_tokens
        || top_k_log_probs.len() != n_assistant_tokens
        || tail_token_ids.len() != n_assistant_tokens
        || tail_log_probs.len() != n_assistant_tokens
        || tail_mass_vec.len() != n_assistant_tokens
        || raw_probs.len() != n_assistant_tokens
        || expert_indices.len() != n_assistant_tokens
    {
        return Err("Internal error: training data arrays are misaligned".to_string());
    }

    Ok(paramecia_opt::TuningData {
        token_ids,
        assistant_mask,
        top_k_token_ids,
        top_k_log_probs,
        tail_token_ids,
        tail_log_probs,
        tail_mass: tail_mass_vec,
        raw_probs,
        expert_indices,
        vocab_size: 0,
        n_layers: 0,
        n_experts_per_tok: 0,
        source_path: PathBuf::from("<stream>"),
    })
}

fn validate_raw_distribution(raw_dist: &[f32], name: &str) -> Result<(usize, f32), String> {
    if raw_dist.is_empty() {
        return Err(format!("{name} is empty"));
    }

    if raw_dist
        .iter()
        .any(|p| !p.is_finite() || p.is_sign_negative())
    {
        return Err(format!("{name} contains non-finite or negative values"));
    }

    let sum: f32 = raw_dist.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(format!("{name} must have positive finite mass"));
    }

    let (argmax_idx, _) = raw_dist
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| format!("{name} is empty"))?;

    Ok((argmax_idx, sum))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::bpe::BPE;

    fn test_tokenizer() -> tokenizers::Tokenizer {
        tokenizers::Tokenizer::new(BPE::default())
    }

    #[test]
    fn supports_raw_probability_context_by_using_argmax_token() {
        let sample = vec![TrainingData::Context(ModelInput::Raw(vec![
            0.1, 0.7, 0.2,
        ]))];
        let tuning = sample_to_tuning_data(&sample, &test_tokenizer()).expect("conversion failed");

        assert_eq!(tuning.token_ids, vec![1]);
        assert_eq!(tuning.assistant_mask, vec![false]);
        assert!(tuning.top_k_token_ids.is_empty());
        assert!(tuning.top_k_log_probs.is_empty());
        assert!(tuning.tail_token_ids.is_empty());
        assert!(tuning.tail_log_probs.is_empty());
        assert!(tuning.tail_mass.is_empty());
        assert!(tuning.raw_probs.is_empty());
        assert!(tuning.expert_indices.is_empty());
    }

    #[test]
    fn raw_probability_context_rejects_negative_values() {
        let sample = vec![TrainingData::Context(ModelInput::Raw(vec![0.5, -0.1, 0.6]))];
        let err = sample_to_tuning_data(&sample, &test_tokenizer()).expect_err("expected error");
        assert!(err.contains("contains non-finite or negative values"));
    }
}
