//! Logits-to-distribution conversion helpers.
//!
//! Converts raw model logits into structured distribution information
//! with top-k entries and stratified tail sampling.

use crate::types::LogitEntry;
use paramecia_core::Tensor;
use rand::prelude::IndexedRandom;

/// Intermediate distribution representation.
pub(crate) struct Distribution {
    pub top_k: Vec<LogitEntry>,
    pub tail: Vec<LogitEntry>,
    pub tail_mass: f32,
}

/// Compute log-softmax of logits.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let log_sum = exp_sum.ln();
    logits.iter().map(|&x| x - max - log_sum).collect()
}

/// Convert raw logits tensor to distribution info with stratified tail sampling.
///
/// Returns top-k log-probability entries plus a stratified sample from the tail
/// (near-miss region + bottom 99% of vocabulary).
pub(crate) fn logits_to_distribution(
    logits: &Tensor,
    top_k: usize,
    tail_samples: usize,
) -> anyhow::Result<Distribution> {
    let logits_f32 = logits.to_vec1::<f32>()?;
    let vocab_size = logits_f32.len();

    let log_probs = log_softmax(&logits_f32);

    // Sort indices by log-prob descending
    let mut indexed: Vec<(usize, f32)> = log_probs.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-K
    let k = top_k.min(vocab_size);
    let top_k_entries: Vec<LogitEntry> = indexed[..k]
        .iter()
        .map(|&(idx, lp)| LogitEntry {
            token_id: idx as u32,
            log_prob: lp,
        })
        .collect();

    // Stratified tail sampling
    let tail_indices: Vec<(usize, f32)> = indexed[k..].to_vec();

    let (tail, tail_mass) = if tail_indices.is_empty() {
        (Vec::new(), 0.0)
    } else {
        let tail_mass: f32 = tail_indices.iter().map(|&(_, lp)| lp.exp()).sum();

        let mut rng = rand::rng();
        let total_samples = tail_samples.min(tail_indices.len());

        // Split: extra sample to near-misses if odd
        let near_miss_count = total_samples.div_ceil(2);
        let bottom_count = total_samples / 2;

        // Near-miss region: top_k*4 positions immediately after top-k
        let near_miss_size = k * 4;
        let near_miss_region: Vec<(usize, f32)> =
            tail_indices.iter().take(near_miss_size).copied().collect();

        // Bottom 99%: skip top 1% of vocab
        let top_1_percent = indexed.len() / 100;
        let bottom_99_start = top_1_percent.saturating_sub(k);
        let bottom_region: Vec<(usize, f32)> =
            tail_indices.iter().skip(bottom_99_start).copied().collect();

        let near_samples: Vec<LogitEntry> = near_miss_region
            .choose_multiple(&mut rng, near_miss_count.min(near_miss_region.len()))
            .map(|&(idx, lp)| LogitEntry {
                token_id: idx as u32,
                log_prob: lp,
            })
            .collect();

        let bottom_samples: Vec<LogitEntry> = bottom_region
            .choose_multiple(&mut rng, bottom_count.min(bottom_region.len()))
            .map(|&(idx, lp)| LogitEntry {
                token_id: idx as u32,
                log_prob: lp,
            })
            .collect();

        let mut tail = near_samples;
        tail.extend(bottom_samples);

        (tail, tail_mass)
    };

    Ok(Distribution {
        top_k: top_k_entries,
        tail,
        tail_mass,
    })
}
