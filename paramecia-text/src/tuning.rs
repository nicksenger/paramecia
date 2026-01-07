//! Tuning output types and writer for distillation.
//!
//! This module provides types and utilities for capturing model logits during
//! generation for subsequent distillation/tuning. The binary format is designed
//! to be portable across platforms and zero-copy friendly.

use std::io::Write;
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use half::f16;
use rand::prelude::IndexedRandom;

/// Magic bytes for tuning output files.
pub const TUNING_MAGIC: [u8; 4] = *b"PGEN";

/// Current format version.
/// Version 1: Initial format with logits only
/// Version 2: Added expert routing data for MoE models (u8 indices)
/// Version 3: Expert indices stored as u16 (supports >255 experts)
pub const TUNING_VERSION: u16 = 3;

/// Default number of top-k tokens to store per position.
pub const DEFAULT_TOP_K: usize = 128;

/// Default number of tail samples to store per position.
pub const DEFAULT_TAIL_SAMPLES: usize = 32;

/// Header for tuning output files.
///
/// This is a fixed 32-byte header that describes the contents of the file.
/// All multi-byte values are stored as little-endian for portability.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TuningHeader {
    /// Magic bytes: "PGEN"
    pub magic: [u8; 4],
    /// Format version
    pub version: u16,
    /// Number of top-k tokens stored per position
    pub n_top_k: u16,
    /// Number of tail samples stored per position
    pub n_tail: u16,
    /// Total number of tokens in the conversation
    pub n_tokens: u32,
    /// Number of assistant tokens (positions with logits)
    pub n_assistant_tokens: u32,
    /// Model vocabulary size
    pub vocab_size: u32,
    /// Number of MoE layers (0 if not an MoE model or expert data not captured)
    pub n_layers: u16,
    /// Number of experts selected per token per layer (typically 8)
    pub n_experts_per_tok: u8,
    /// Reserved for future use
    pub reserved: [u8; 7],
}

impl TuningHeader {
    /// Create a new header with the given parameters.
    #[must_use]
    pub fn new(
        n_top_k: usize,
        n_tail: usize,
        n_tokens: usize,
        n_assistant_tokens: usize,
        vocab_size: usize,
        n_layers: usize,
        n_experts_per_tok: usize,
    ) -> Self {
        Self {
            magic: TUNING_MAGIC,
            version: TUNING_VERSION,
            n_top_k: n_top_k as u16,
            n_tail: n_tail as u16,
            n_tokens: n_tokens as u32,
            n_assistant_tokens: n_assistant_tokens as u32,
            vocab_size: vocab_size as u32,
            n_layers: n_layers as u16,
            n_experts_per_tok: n_experts_per_tok as u8,
            reserved: [0u8; 7],
        }
    }

    /// Validate the header magic and version.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.magic == TUNING_MAGIC && (self.version >= 1 && self.version <= TUNING_VERSION)
    }

    /// Get the number of logit entries per assistant token.
    #[must_use]
    pub fn entries_per_token(&self) -> usize {
        self.n_top_k as usize + self.n_tail as usize
    }

    /// Check if this file contains expert routing data.
    #[must_use]
    pub fn has_expert_data(&self) -> bool {
        self.version >= 2 && self.n_layers > 0 && self.n_experts_per_tok > 0
    }

    /// Get the number of expert indices per assistant token.
    #[must_use]
    pub fn expert_indices_per_token(&self) -> usize {
        self.n_layers as usize * self.n_experts_per_tok as usize
    }
}

/// Single logit entry storing a token ID and its log probability.
///
/// Uses f16 for log probabilities to save space while maintaining
/// sufficient precision for KL divergence calculations.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LogitEntry {
    /// Token ID in the vocabulary
    pub token_id: u32,
    /// Log probability as f16 bits
    pub log_prob_bits: u16,
}

impl LogitEntry {
    /// Create a new logit entry.
    #[must_use]
    pub fn new(token_id: u32, log_prob: f32) -> Self {
        Self {
            token_id,
            log_prob_bits: f16::from_f32(log_prob).to_bits(),
        }
    }

    /// Get the log probability as f32.
    #[must_use]
    pub fn log_prob(&self) -> f32 {
        f16::from_bits(self.log_prob_bits).to_f32()
    }
}

/// Accumulated logits for a single token position.
#[derive(Debug, Clone)]
pub struct PositionLogits {
    /// Top-k logit entries (sorted by probability, descending)
    pub top_k: Vec<LogitEntry>,
    /// Sampled tail entries (randomly sampled from outside top-k)
    pub tail: Vec<LogitEntry>,
    /// Total probability mass of the excluded tail (for importance weighting)
    pub tail_mass: f32,
}

/// Configuration for tuning output.
#[derive(Debug, Clone)]
pub struct TuningConfig {
    /// Number of top-k tokens to capture
    pub top_k: usize,
    /// Number of tail samples to capture
    pub tail_samples: usize,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            top_k: DEFAULT_TOP_K,
            tail_samples: DEFAULT_TAIL_SAMPLES,
        }
    }
}

impl TuningConfig {
    /// Create from provider config options.
    #[must_use]
    pub fn from_options(top_k: Option<usize>, tail_samples: Option<usize>) -> Self {
        Self {
            top_k: top_k
                .or_else(|| {
                    std::env::var("PARAMECIA_TUNING_TOP_K")
                        .ok()
                        .and_then(|s| s.parse().ok())
                })
                .unwrap_or(DEFAULT_TOP_K),
            tail_samples: tail_samples
                .or_else(|| {
                    std::env::var("PARAMECIA_TUNING_TAIL_SAMPLES")
                        .ok()
                        .and_then(|s| s.parse().ok())
                })
                .unwrap_or(DEFAULT_TAIL_SAMPLES),
        }
    }
}

/// Writer for accumulating and saving tuning data.
///
/// This accumulates token IDs, assistant masks, and logits during generation,
/// then writes them to a binary file in a portable format.
pub struct TuningWriter {
    config: TuningConfig,
    vocab_size: usize,
    /// All token IDs in the conversation
    token_ids: Vec<u32>,
    /// Mask indicating assistant-generated tokens (1 = assistant, 0 = other)
    assistant_mask: Vec<u8>,
    /// Logits for each assistant token position
    position_logits: Vec<PositionLogits>,
    /// Expert indices for each assistant token position (flattened: n_layers * n_experts_per_tok per token)
    expert_indices: Vec<Vec<u32>>,
    /// Number of MoE layers (0 if not tracking experts)
    n_layers: usize,
    /// Number of experts selected per token per layer
    n_experts_per_tok: usize,
}

impl TuningWriter {
    /// Create a new tuning writer without expert tracking.
    #[must_use]
    pub fn new(config: TuningConfig, vocab_size: usize) -> Self {
        Self {
            config,
            vocab_size,
            token_ids: Vec::new(),
            assistant_mask: Vec::new(),
            position_logits: Vec::new(),
            expert_indices: Vec::new(),
            n_layers: 0,
            n_experts_per_tok: 0,
        }
    }

    /// Create a new tuning writer with expert tracking for MoE models.
    #[must_use]
    pub fn new_with_experts(
        config: TuningConfig,
        vocab_size: usize,
        n_layers: usize,
        n_experts_per_tok: usize,
    ) -> Self {
        Self {
            config,
            vocab_size,
            token_ids: Vec::new(),
            assistant_mask: Vec::new(),
            position_logits: Vec::new(),
            expert_indices: Vec::new(),
            n_layers,
            n_experts_per_tok,
        }
    }

    /// Add a token that is not an assistant token (user, system, tool).
    pub fn add_non_assistant_token(&mut self, token_id: u32) {
        self.token_ids.push(token_id);
        self.assistant_mask.push(0);
    }

    /// Add multiple non-assistant tokens at once.
    pub fn add_non_assistant_tokens(&mut self, token_ids: &[u32]) {
        for &token_id in token_ids {
            self.add_non_assistant_token(token_id);
        }
    }

    /// Add an assistant-generated token with its logits.
    ///
    /// The logits should be the raw model output (before temperature/sampling).
    /// This function will:
    /// 1. Convert to log probabilities
    /// 2. Extract top-k entries
    /// 3. Randomly sample tail entries
    /// 4. Calculate tail mass for importance weighting
    pub fn add_assistant_token_with_logits(&mut self, token_id: u32, logits: &[f32]) {
        self.add_assistant_token_impl(token_id, logits, None);
    }

    /// Add an assistant-generated token with logits and expert routing data.
    ///
    /// The expert_indices should be a flat array of shape [n_layers, n_experts_per_tok],
    /// containing the indices of experts selected at each layer for this token.
    pub fn add_assistant_token_with_logits_and_experts(
        &mut self,
        token_id: u32,
        logits: &[f32],
        expert_indices: &[u32],
    ) {
        self.add_assistant_token_impl(token_id, logits, Some(expert_indices));
    }

    /// Add an assistant-generated token with pre-computed distribution entries.
    ///
    /// This accepts the distribution already computed by the executor (top_k + tail
    /// as `(token_id, log_prob)` pairs) instead of raw logits. Used when the executor
    /// has already done the log-softmax and stratified sampling.
    pub fn add_assistant_token_from_distribution(
        &mut self,
        token_id: u32,
        top_k: &[(u32, f32)],
        tail: &[(u32, f32)],
        tail_mass: f32,
        expert_indices: Option<&[u32]>,
    ) {
        self.token_ids.push(token_id);
        self.assistant_mask.push(1);

        let top_k_entries = top_k
            .iter()
            .map(|&(tid, lp)| LogitEntry::new(tid, lp))
            .collect();
        let tail_entries = tail
            .iter()
            .map(|&(tid, lp)| LogitEntry::new(tid, lp))
            .collect();

        self.position_logits.push(PositionLogits {
            top_k: top_k_entries,
            tail: tail_entries,
            tail_mass,
        });

        if let Some(indices) = expert_indices {
            self.expert_indices.push(indices.to_vec());
        } else if self.n_layers > 0 {
            self.expert_indices
                .push(vec![0u32; self.n_layers * self.n_experts_per_tok]);
        }
    }

    fn add_assistant_token_impl(
        &mut self,
        token_id: u32,
        logits: &[f32],
        expert_indices: Option<&[u32]>,
    ) {
        self.token_ids.push(token_id);
        self.assistant_mask.push(1);

        // Convert logits to log probabilities using log-softmax
        let log_probs = log_softmax(logits);

        // Create (index, log_prob) pairs and sort by probability (descending)
        let mut indexed: Vec<(usize, f32)> = log_probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Extract top-k
        let top_k: Vec<LogitEntry> = indexed
            .iter()
            .take(self.config.top_k)
            .map(|&(idx, log_prob)| LogitEntry::new(idx as u32, log_prob))
            .collect();

        // Sample from tail using stratified sampling:
        // - Half from "near-misses": top_k*4 positions immediately after top-k
        // - Half from "bottom 99%": the lowest-probability tokens
        // - If tail_samples is odd, extra sample goes to near-misses
        let tail_indices: Vec<(usize, f32)> =
            indexed.iter().skip(self.config.top_k).copied().collect();

        let (tail, tail_mass) = if tail_indices.is_empty() {
            (Vec::new(), 0.0)
        } else {
            // Calculate total tail probability mass (for importance weighting)
            let tail_mass: f32 = tail_indices
                .iter()
                .map(|&(_, log_prob)| log_prob.exp())
                .sum();

            let mut rng = rand::rng();
            let total_samples = self.config.tail_samples.min(tail_indices.len());

            // Split samples: half to near-misses, half to bottom 99%
            // If odd, extra goes to near-misses
            let near_miss_count = total_samples.div_ceil(2);
            let bottom_count = total_samples / 2;

            // Near-miss region: top_k*4 positions immediately after top-k
            let near_miss_size = self.config.top_k * 4;
            let near_miss_region: Vec<(usize, f32)> =
                tail_indices.iter().take(near_miss_size).copied().collect();

            // Bottom 99% region: skip top 1% of vocab
            let top_1_percent = indexed.len() / 100;
            let bottom_99_start = top_1_percent.saturating_sub(self.config.top_k);
            let bottom_region: Vec<(usize, f32)> =
                tail_indices.iter().skip(bottom_99_start).copied().collect();

            // Sample from each region
            let near_miss_samples: Vec<LogitEntry> = near_miss_region
                .choose_multiple(&mut rng, near_miss_count.min(near_miss_region.len()))
                .map(|&(idx, log_prob)| LogitEntry::new(idx as u32, log_prob))
                .collect();

            let bottom_samples: Vec<LogitEntry> = bottom_region
                .choose_multiple(&mut rng, bottom_count.min(bottom_region.len()))
                .map(|&(idx, log_prob)| LogitEntry::new(idx as u32, log_prob))
                .collect();

            // Combine samples
            let mut sampled = near_miss_samples;
            sampled.extend(bottom_samples);

            (sampled, tail_mass)
        };

        self.position_logits.push(PositionLogits {
            top_k,
            tail,
            tail_mass,
        });

        // Store expert indices if provided
        if let Some(indices) = expert_indices {
            self.expert_indices.push(indices.to_vec());
        } else if self.n_layers > 0 {
            // If we're tracking experts but none provided, store zeros
            self.expert_indices
                .push(vec![0u32; self.n_layers * self.n_experts_per_tok]);
        }
    }

    /// Get the number of tokens accumulated.
    #[must_use]
    pub fn n_tokens(&self) -> usize {
        self.token_ids.len()
    }

    /// Get the number of assistant tokens accumulated.
    #[must_use]
    pub fn n_assistant_tokens(&self) -> usize {
        self.position_logits.len()
    }

    /// Get the number of non-assistant tokens accumulated (user prompts, tool results).
    #[must_use]
    pub fn n_non_assistant_tokens(&self) -> usize {
        self.token_ids.len() - self.position_logits.len()
    }

    /// Write the accumulated data to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or written.
    pub fn write_to_file(&self, path: &Path) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        self.write_to(&mut file)
    }

    /// Write the accumulated data to a writer.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Write header
        let header = TuningHeader::new(
            self.config.top_k,
            self.config.tail_samples,
            self.token_ids.len(),
            self.position_logits.len(),
            self.vocab_size,
            self.n_layers,
            self.n_experts_per_tok,
        );
        writer.write_all(bytemuck::bytes_of(&header))?;

        // Write token IDs
        writer.write_all(bytemuck::cast_slice(&self.token_ids))?;

        // Write assistant mask
        writer.write_all(&self.assistant_mask)?;

        // Write logit data for each assistant position
        for pos in &self.position_logits {
            // Write top-k entries
            for entry in &pos.top_k {
                writer.write_all(bytemuck::bytes_of(entry))?;
            }
            // Pad if we have fewer than top_k
            let padding_top_k = self.config.top_k - pos.top_k.len();
            for _ in 0..padding_top_k {
                writer.write_all(bytemuck::bytes_of(&LogitEntry::new(0, f32::NEG_INFINITY)))?;
            }

            // Write tail entries
            for entry in &pos.tail {
                writer.write_all(bytemuck::bytes_of(entry))?;
            }
            // Pad if we have fewer than tail_samples
            let padding_tail = self.config.tail_samples - pos.tail.len();
            for _ in 0..padding_tail {
                writer.write_all(bytemuck::bytes_of(&LogitEntry::new(0, f32::NEG_INFINITY)))?;
            }
        }

        // Write tail mass values
        for pos in &self.position_logits {
            let tail_mass_f16 = f16::from_f32(pos.tail_mass);
            writer.write_all(&tail_mass_f16.to_le_bytes())?;
        }

        // Write expert indices as u16 if tracking experts (version 3+)
        if self.n_layers > 0 && self.n_experts_per_tok > 0 {
            let expected_len = self.n_layers * self.n_experts_per_tok;
            for indices in &self.expert_indices {
                let src = if indices.len() == expected_len {
                    indices.as_slice()
                } else {
                    // Pad or truncate to expected length
                    &[]
                };
                for i in 0..expected_len {
                    let val = src.get(i).copied().unwrap_or(0) as u16;
                    writer.write_all(&val.to_le_bytes())?;
                }
            }
        }

        writer.flush()
    }
}

/// Compute log-softmax of logits.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    // Find max for numerical stability
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let exp_sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let log_sum = exp_sum.ln();

    // log_softmax(x) = x - max - log(sum(exp(x - max)))
    logits.iter().map(|&x| x - max - log_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_size() {
        assert_eq!(std::mem::size_of::<TuningHeader>(), 32);
    }

    #[test]
    fn test_logit_entry_size() {
        assert_eq!(std::mem::size_of::<LogitEntry>(), 6);
    }

    #[test]
    fn test_log_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let log_probs = log_softmax(&logits);

        // Check that exp(log_probs) sums to 1
        let sum: f32 = log_probs.iter().map(|&x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_tuning_writer_basic() {
        let config = TuningConfig {
            top_k: 5,
            tail_samples: 3,
        };
        let mut writer = TuningWriter::new(config, 100);

        // Add some non-assistant tokens
        writer.add_non_assistant_tokens(&[1, 2, 3]);

        // Add an assistant token with fake logits
        let logits: Vec<f32> = (0..100).map(|i| i as f32).collect();
        writer.add_assistant_token_with_logits(50, &logits);

        assert_eq!(writer.n_tokens(), 4);
        assert_eq!(writer.n_assistant_tokens(), 1);
    }
}
