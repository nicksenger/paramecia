//! Conversion bridge between engine training types and paramecia-opt TuningData.

use crate::types::*;
use std::path::PathBuf;

/// Convert a training sample's data segments to paramecia-opt TuningData.
///
/// Walks interleaved context/target segments:
/// - context(model-input) → dispatch by input type:
///   - Text(string) → tokenize → extend token_ids with assistant_mask=false
///   - Tokens(list<u32>) → extend token_ids with assistant_mask=false
///   - Soft(_) → not supported in training context (returns error)
/// - target(list<predicted>) → each predicted adds one token with assistant_mask=true
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

                    expert_indices.push(pred.expert_indices.clone());
                }
            }
        }
    }

    Ok(paramecia_opt::TuningData {
        token_ids,
        assistant_mask,
        top_k_token_ids,
        top_k_log_probs,
        tail_token_ids,
        tail_log_probs,
        tail_mass: tail_mass_vec,
        expert_indices,
        vocab_size: 0,
        n_layers: 0,
        n_experts_per_tok: 0,
        source_path: PathBuf::from("<stream>"),
    })
}
