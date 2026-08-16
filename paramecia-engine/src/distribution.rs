//! Logits-to-distribution conversion helpers.
//!
//! Converts raw model logits into structured distribution information
//! with top-k entries and stratified tail sampling.

use crate::types::LogitEntry;
use paramecia_model::Tensor;
use rand::prelude::IndexedRandom;

/// Intermediate distribution representation.
pub(crate) struct Distribution {
    pub top_k: Vec<LogitEntry>,
    pub tail: Vec<LogitEntry>,
    pub tail_mass: f32,
}

/// Compute the scalar normalization term for log-softmax.
fn log_softmax_normalizer(logits: &[f32]) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    max + exp_sum.ln()
}

/// Convert a host logits slice to distribution info with stratified tail sampling.
///
/// Only the top-k entries are sorted. The vocabulary is partitioned in linear
/// time for the near-miss and bottom-99% regions, avoiding a full vocabulary
/// sort on every generated token.
pub(crate) fn logits_slice_to_distribution(
    logits: &[f32],
    top_k: usize,
    tail_samples: usize,
) -> Distribution {
    let vocab_size = logits.len();
    let log_normalizer = log_softmax_normalizer(logits);
    let k = top_k.min(vocab_size);

    let near_end = k.saturating_add(k.saturating_mul(4).min(vocab_size.saturating_sub(k)));
    let bottom_start = (vocab_size / 100).max(k).min(vocab_size);

    let mut indices: Vec<usize> = (0..vocab_size).collect();
    let cmp_desc = |a: &usize, b: &usize| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    };

    // Establish all rank boundaries from largest to smallest. Partitioning only
    // the already-established prefix preserves every larger boundary.
    let mut boundaries = vec![k, near_end, bottom_start];
    boundaries.sort_unstable();
    boundaries.dedup();
    let mut prefix_end = vocab_size;
    for &boundary in boundaries.iter().rev() {
        if boundary < prefix_end {
            indices[..prefix_end].select_nth_unstable_by(boundary, &cmp_desc);
        }
        prefix_end = boundary;
    }
    indices[..k].sort_unstable_by(&cmp_desc);

    let top_k_entries = indices[..k]
        .iter()
        .map(|&idx| LogitEntry {
            token_id: idx as u32,
            log_prob: logits[idx] - log_normalizer,
        })
        .collect();

    if k == vocab_size {
        return Distribution {
            top_k: top_k_entries,
            tail: Vec::new(),
            tail_mass: 0.0,
        };
    }

    let top_mass: f32 = top_k_entries.iter().map(|entry| entry.log_prob.exp()).sum();
    let tail_mass = (1.0 - top_mass).clamp(0.0, 1.0);
    let total_samples = tail_samples.min(vocab_size - k);
    let near_miss_count = total_samples.div_ceil(2);
    let bottom_count = total_samples / 2;
    let mut rng = rand::rng();

    let mut tail: Vec<LogitEntry> = indices[k..near_end]
        .choose_multiple(&mut rng, near_miss_count.min(near_end - k))
        .map(|&idx| LogitEntry {
            token_id: idx as u32,
            log_prob: logits[idx] - log_normalizer,
        })
        .collect();
    tail.extend(
        indices[bottom_start..]
            .choose_multiple(
                &mut rng,
                bottom_count.min(vocab_size.saturating_sub(bottom_start)),
            )
            .map(|&idx| LogitEntry {
                token_id: idx as u32,
                log_prob: logits[idx] - log_normalizer,
            }),
    );

    Distribution {
        top_k: top_k_entries,
        tail,
        tail_mass,
    }
}

/// Convert raw logits tensor to distribution info with stratified tail sampling.
pub(crate) fn logits_to_distribution(
    logits: &Tensor,
    top_k: usize,
    tail_samples: usize,
) -> anyhow::Result<Distribution> {
    let logits_f32 = logits.to_vec1::<f32>()?;
    Ok(logits_slice_to_distribution(
        &logits_f32,
        top_k,
        tail_samples,
    ))
}

#[cfg(test)]
mod tests {
    use super::logits_slice_to_distribution;

    #[test]
    fn slice_distribution_preserves_ranked_top_k_and_tail_mass() {
        let logits = [0.5, -1.0, 4.0, 2.0, -3.0, 1.0];
        let distribution = logits_slice_to_distribution(&logits, 3, 0);

        let ids: Vec<u32> = distribution
            .top_k
            .iter()
            .map(|entry| entry.token_id)
            .collect();
        assert_eq!(ids, vec![2, 3, 5]);
        assert!(distribution.tail.is_empty());

        let top_mass: f32 = distribution
            .top_k
            .iter()
            .map(|entry| entry.log_prob.exp())
            .sum();
        assert!((distribution.tail_mass - (1.0 - top_mass)).abs() < 1e-5);
    }

    #[test]
    fn slice_distribution_handles_top_k_larger_than_vocabulary() {
        let distribution = logits_slice_to_distribution(&[1.0, 3.0, 2.0], 10, 4);
        let ids: Vec<u32> = distribution
            .top_k
            .iter()
            .map(|entry| entry.token_id)
            .collect();
        assert_eq!(ids, vec![1, 2, 0]);
        assert!(distribution.tail.is_empty());
        assert_eq!(distribution.tail_mass, 0.0);
    }
}
