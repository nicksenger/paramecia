//! Apply penalty and repeat_kv

use candle::{Result, Tensor};

/// Apply repetition penalty (multiplicative, frequency-based).
/// Penalizes logits based on how many times each token appears in the context.
/// A penalty of 1.0 means no effect (disabled).
/// Values > 1.0 reduce the probability of repeated tokens, with stronger
/// penalties for tokens that appear more frequently.
pub fn apply_repeat_penalty(logits: &Tensor, penalty: f32, context: &[u32]) -> Result<Tensor> {
    if penalty == 1.0 {
        return Ok(logits.clone());
    }

    let device = logits.device();
    let mut logits = logits.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?;

    // Count token frequencies for frequency-based penalty
    let mut token_counts = std::collections::HashMap::new();
    for &token_id in context {
        *token_counts.entry(token_id).or_insert(0u32) += 1;
    }

    // Apply penalty based on frequency: penalty^count
    for (token_id, count) in token_counts {
        if let Some(logit) = logits.get_mut(token_id as usize) {
            // Compute cumulative penalty: penalty^count
            let cumulative_penalty = penalty.powi(count as i32);
            // Multiplicative penalty: divide positive logits, multiply negative logits
            if *logit >= 0. {
                *logit /= cumulative_penalty;
            } else {
                *logit *= cumulative_penalty;
            }
        }
    }

    let logits_len = logits.len();
    Tensor::from_vec(logits, logits_len, device)
}

/// Apply presence penalty (additive/flat).
/// Subtracts a flat penalty from logits of tokens that have appeared at least once.
/// A penalty of 0.0 means no effect (disabled).
/// Values > 0.0 reduce the probability of tokens that have appeared.
pub fn apply_presence_penalty(logits: &Tensor, penalty: f32, context: &[u32]) -> Result<Tensor> {
    if penalty == 0.0 {
        return Ok(logits.clone());
    }

    let device = logits.device();
    let mut logits = logits.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?;

    // Track unique tokens (each token penalized once regardless of frequency)
    let mut already_seen = std::collections::HashSet::new();
    for &token_id in context {
        if already_seen.contains(&token_id) {
            continue;
        }
        already_seen.insert(token_id);
        if let Some(logit) = logits.get_mut(token_id as usize) {
            // Flat/additive penalty: subtract from the logit
            *logit -= penalty;
        }
    }

    let logits_len = logits.len();
    Tensor::from_vec(logits, logits_len, device)
}

/// Repeats a key or value tensor for grouped query attention
/// The input tensor should have a shape `(batch, num_kv_heads, seq_len, head_dim)`,
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        // Using cat is faster than a broadcast as it avoids going through a potentially
        // strided copy.
        // https://github.com/huggingface/candle/pull/2043
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}
