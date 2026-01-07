//! Binary tuning file reader for distillation training.
//!
//! Parses the PGEN format tuning files containing teacher model outputs:
//! - Top-k token probabilities (typically 100)
//! - Randomly sampled tail token probabilities (typically 20)
//! - Tail probability mass for importance weighting
//! - Expert routing indices per layer

use bytemuck::{Pod, Zeroable};
use half::f16;
use paramecia_core::{Device, Result, Tensor};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use tracing::warn;

/// Magic bytes for tuning output files.
pub const TUNING_MAGIC: [u8; 4] = *b"PGEN";

/// Header for tuning output files (32 bytes).
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TuningHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub n_top_k: u16,
    pub n_tail: u16,
    pub n_tokens: u32,
    pub n_assistant_tokens: u32,
    pub vocab_size: u32,
    pub n_layers: u16,
    pub n_experts_per_tok: u8,
    pub reserved: [u8; 7],
}

impl TuningHeader {
    pub fn is_valid(&self) -> bool {
        // Copy packed fields to avoid unaligned access
        let magic = self.magic;
        let version = self.version;
        magic == TUNING_MAGIC && (1..=3).contains(&version)
    }

    pub fn has_expert_data(&self) -> bool {
        // Copy packed fields to avoid unaligned access
        let version = self.version;
        let n_layers = self.n_layers;
        let n_experts_per_tok = self.n_experts_per_tok;
        version >= 2 && n_layers > 0 && n_experts_per_tok > 0
    }

    #[allow(dead_code)] // public API accessor
    pub fn entries_per_token(&self) -> usize {
        // Copy packed fields to avoid unaligned access
        let n_top_k = self.n_top_k;
        let n_tail = self.n_tail;
        n_top_k as usize + n_tail as usize
    }
}

/// Single logit entry storing a token ID and its log probability.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LogitEntry {
    pub token_id: u32,
    pub log_prob_bits: u16,
}

impl LogitEntry {
    pub fn log_prob(&self) -> f32 {
        f16::from_bits(self.log_prob_bits).to_f32()
    }

    #[allow(dead_code)] // public API accessor, used by examples
    pub fn prob(&self) -> f32 {
        self.log_prob().exp()
    }
}

/// Parsed tuning data for a single conversation.
#[derive(Debug, Clone)]
pub struct TuningData {
    /// All token IDs in the conversation (prompt + assistant)
    pub token_ids: Vec<u32>,
    /// Mask indicating assistant tokens (true = assistant generated)
    pub assistant_mask: Vec<bool>,
    /// Top-k log probabilities for each assistant token [n_assistant, n_top_k]
    pub top_k_token_ids: Vec<Vec<u32>>,
    pub top_k_log_probs: Vec<Vec<f32>>,
    /// Tail sample log probabilities for each assistant token [n_assistant, n_tail]
    pub tail_token_ids: Vec<Vec<u32>>,
    pub tail_log_probs: Vec<Vec<f32>>,
    /// Total probability mass of tail (for importance weighting)
    pub tail_mass: Vec<f32>,
    /// Expert indices per assistant token [n_assistant, n_layers * n_experts_per_tok]
    pub expert_indices: Vec<Vec<u32>>,
    /// Metadata
    pub vocab_size: usize,
    pub n_layers: usize,
    pub n_experts_per_tok: usize,
    /// Source file path (for logging/debugging)
    pub source_path: PathBuf,
}

/// A TuningData padded to a target length for batching.
#[derive(Debug, Clone)]
pub struct PaddedTuningData {
    pub original: TuningData,
    pub padded_token_ids: Vec<u32>,
    pub padded_assistant_mask: Vec<bool>,
    pub padding_mask: Vec<bool>, // true = padding position
    pub original_len: usize,
}

impl TuningData {
    /// Get the number of assistant tokens (positions with teacher probabilities).
    pub fn n_assistant_tokens(&self) -> usize {
        self.top_k_log_probs.len()
    }

    /// Get the total sequence length.
    pub fn seq_len(&self) -> usize {
        self.token_ids.len()
    }

    /// Get indices of assistant tokens in the sequence.
    pub fn assistant_indices(&self) -> Vec<usize> {
        self.assistant_mask
            .iter()
            .enumerate()
            .filter(|(_, &is_assistant)| is_assistant)
            .map(|(i, _)| i)
            .collect()
    }

    /// Convert token IDs to tensor for model input.
    pub fn input_ids_tensor(&self, device: &Device) -> Result<Tensor> {
        Tensor::new(self.token_ids.as_slice(), device)
    }

    /// Get a chunk of the conversation for processing.
    /// Returns (token_ids, start_idx, assistant_data_for_chunk).
    pub fn get_chunk(&self, start: usize, chunk_size: usize) -> (Vec<u32>, Vec<(usize, usize)>) {
        let end = (start + chunk_size).min(self.token_ids.len());
        let chunk_tokens = self.token_ids[start..end].to_vec();

        // Map global assistant indices to (chunk_position, assistant_data_index)
        let mut assistant_data_idx = 0;
        let mut chunk_assistant_positions = Vec::new();

        for (global_idx, &is_assistant) in self.assistant_mask.iter().enumerate() {
            if is_assistant {
                if global_idx >= start && global_idx < end {
                    let chunk_pos = global_idx - start;
                    chunk_assistant_positions.push((chunk_pos, assistant_data_idx));
                }
                assistant_data_idx += 1;
            }
        }

        (chunk_tokens, chunk_assistant_positions)
    }

    /// Compute a weighted embedding for a single data position.
    ///
    /// The weighted embedding is the expectation over the teacher distribution:
    /// `weighted_embed = Σ p_teacher[j] * embed[token_j]`
    ///
    /// This provides smoother gradient estimates for ZO optimization compared to
    /// using single ground-truth token embeddings.
    ///
    /// # Arguments
    /// * `data_idx` - Index into assistant token data (top_k/tail arrays)
    /// * `embed_weights` - Token embedding weights [vocab_size, hidden_dim]
    ///
    /// # Returns
    /// Weighted embedding tensor [hidden_dim]
    pub fn compute_weighted_embedding(
        &self,
        data_idx: usize,
        embed_weights: &Tensor,
    ) -> Result<Tensor> {
        // CRITICAL: Use same device as embed_weights to avoid CPU<->GPU sync bottleneck
        let device = embed_weights.device();

        // Gather top-k + tail token IDs and probs
        let top_k_ids = &self.top_k_token_ids[data_idx];
        let top_k_probs: Vec<f32> = self.top_k_log_probs[data_idx]
            .iter()
            .map(|lp| lp.exp())
            .collect();
        let tail_ids = &self.tail_token_ids[data_idx];
        let tail_probs: Vec<f32> = self.tail_log_probs[data_idx]
            .iter()
            .map(|lp| lp.exp())
            .collect();

        // Combine all IDs and probs
        let all_ids: Vec<u32> = top_k_ids.iter().chain(tail_ids.iter()).copied().collect();
        let mut all_probs: Vec<f32> = top_k_probs.into_iter().chain(tail_probs).collect();

        // CRITICAL: Normalize probabilities to sum to 1.0
        // The tail provides a "floor" that prevents perturbations from collapsing the embedding space
        let prob_sum: f32 = all_probs.iter().sum();
        if prob_sum > 0.0 {
            for p in &mut all_probs {
                *p /= prob_sum;
            }
        }

        // Create tensors directly on target device (CUDA/Metal)
        let ids_tensor = Tensor::new(all_ids.as_slice(), device)?;
        let sparse_embeds = embed_weights.index_select(&ids_tensor, 0)?; // [n_sparse, hidden]

        // Weighted sum
        let probs_tensor = Tensor::new(all_probs.as_slice(), device)?.unsqueeze(1)?; // [n_sparse, 1]
        sparse_embeds.broadcast_mul(&probs_tensor)?.sum(0) // [hidden]
    }

    /// Compute weighted embeddings for a batch of data positions.
    ///
    /// # Arguments
    /// * `data_indices` - Indices into assistant token data
    /// * `embed_weights` - Token embedding weights [vocab_size, hidden_dim]
    ///
    /// # Returns
    /// Batched weighted embeddings [batch_size, hidden_dim]
    pub fn compute_weighted_embeddings_batch(
        &self,
        data_indices: &[usize],
        embed_weights: &Tensor,
    ) -> Result<Tensor> {
        if data_indices.is_empty() {
            let hidden_dim = embed_weights.dim(1)?;
            return Tensor::zeros(
                (0, hidden_dim),
                embed_weights.dtype(),
                embed_weights.device(),
            );
        }

        let embeds: Vec<Tensor> = data_indices
            .iter()
            .map(|&idx| self.compute_weighted_embedding(idx, embed_weights))
            .collect::<Result<Vec<_>>>()?;

        Tensor::stack(&embeds, 0)
    }

    /// Truncate to max_length if needed.
    pub fn truncate(&self, max_len: usize) -> TuningData {
        if self.seq_len() <= max_len {
            return self.clone();
        }

        // Truncate all fields
        let mut truncated = self.clone();
        truncated.token_ids.truncate(max_len);
        truncated.assistant_mask.truncate(max_len);

        // Count how many assistant positions we're keeping
        let n_assistant_kept = truncated
            .assistant_mask
            .iter()
            .filter(|&&is_asst| is_asst)
            .count();

        // Truncate teacher data arrays to match
        truncated.top_k_token_ids.truncate(n_assistant_kept);
        truncated.top_k_log_probs.truncate(n_assistant_kept);
        truncated.tail_token_ids.truncate(n_assistant_kept);
        truncated.tail_log_probs.truncate(n_assistant_kept);
        truncated.tail_mass.truncate(n_assistant_kept);
        if !truncated.expert_indices.is_empty() {
            truncated.expert_indices.truncate(n_assistant_kept);
        }

        truncated
    }

    /// Pad to target length for batching.
    pub fn pad_to_length(&self, target_len: usize, pad_token: u32) -> PaddedTuningData {
        if self.seq_len() >= target_len {
            return PaddedTuningData {
                padded_token_ids: self.token_ids.clone(),
                padded_assistant_mask: self.assistant_mask.clone(),
                padding_mask: vec![false; self.seq_len()],
                original_len: self.seq_len(),
                original: self.clone(),
            };
        }

        let pad_len = target_len - self.seq_len();
        let mut padded_token_ids = self.token_ids.clone();
        let mut padded_assistant_mask = self.assistant_mask.clone();
        let mut padding_mask = vec![false; self.seq_len()];

        padded_token_ids.extend(vec![pad_token; pad_len]);
        padded_assistant_mask.extend(vec![false; pad_len]);
        padding_mask.extend(vec![true; pad_len]);

        PaddedTuningData {
            padded_token_ids,
            padded_assistant_mask,
            padding_mask,
            original_len: self.seq_len(),
            original: self.clone(),
        }
    }
}

impl PaddedTuningData {
    /// Get chunk with padding information.
    pub fn get_chunk(
        &self,
        start: usize,
        chunk_size: usize,
    ) -> (Vec<u32>, Vec<(usize, usize)>, Vec<bool>) {
        let end = (start + chunk_size).min(self.padded_token_ids.len());
        let chunk_tokens = self.padded_token_ids[start..end].to_vec();
        let chunk_padding = self.padding_mask[start..end].to_vec();

        // Map assistant positions (skip padding)
        let mut assistant_data_idx = 0;
        let mut chunk_assistant_positions = Vec::new();

        for (global_idx, &is_assistant) in self.padded_assistant_mask.iter().enumerate() {
            if is_assistant {
                if global_idx >= start && global_idx < end {
                    let chunk_pos = global_idx - start;
                    if !self.padding_mask[global_idx] {
                        chunk_assistant_positions.push((chunk_pos, assistant_data_idx));
                    }
                }
                assistant_data_idx += 1;
            }
        }

        (chunk_tokens, chunk_assistant_positions, chunk_padding)
    }
}

/// Reader for individual tuning files.
pub struct TuningReader;

impl TuningReader {
    /// Read a tuning file from disk.
    pub fn read_file(path: &Path) -> Result<TuningData> {
        let file = File::open(path).map_err(|e| {
            paramecia_core::Error::Msg(format!("Failed to open {}: {}", path.display(), e))
        })?;
        let mut reader = BufReader::new(file);

        // Read header
        let mut header_bytes = [0u8; 32];
        reader
            .read_exact(&mut header_bytes)
            .map_err(|e| paramecia_core::Error::Msg(format!("Failed to read header: {}", e)))?;
        let header: TuningHeader = *bytemuck::from_bytes(&header_bytes);

        // Copy packed fields to local variables to avoid unaligned access
        let magic = header.magic;
        let version = header.version;
        let n_tokens = header.n_tokens as usize;
        let n_assistant_tokens = header.n_assistant_tokens as usize;
        let n_top_k = header.n_top_k as usize;
        let n_tail = header.n_tail as usize;
        let vocab_size = header.vocab_size as usize;
        let n_layers = header.n_layers as usize;
        let n_experts_per_tok = header.n_experts_per_tok as usize;

        if !header.is_valid() {
            return Err(paramecia_core::Error::Msg(format!(
                "Invalid tuning file: {:?} (magic: {:?}, version: {})",
                path, magic, version
            )));
        }

        // Read token IDs
        let mut token_ids = vec![0u32; n_tokens];
        reader
            .read_exact(bytemuck::cast_slice_mut(&mut token_ids))
            .map_err(|e| paramecia_core::Error::Msg(format!("Failed to read token IDs: {}", e)))?;

        // Read assistant mask
        let mut assistant_mask_bytes = vec![0u8; n_tokens];
        reader.read_exact(&mut assistant_mask_bytes).map_err(|e| {
            paramecia_core::Error::Msg(format!("Failed to read assistant mask: {}", e))
        })?;
        let assistant_mask: Vec<bool> = assistant_mask_bytes.iter().map(|&b| b != 0).collect();

        // Read logit entries
        let entries_per_token = n_top_k + n_tail;
        let entry_size = std::mem::size_of::<LogitEntry>();
        let mut logit_buffer = vec![0u8; entry_size * entries_per_token];

        let mut top_k_token_ids = Vec::with_capacity(n_assistant_tokens);
        let mut top_k_log_probs = Vec::with_capacity(n_assistant_tokens);
        let mut tail_token_ids = Vec::with_capacity(n_assistant_tokens);
        let mut tail_log_probs = Vec::with_capacity(n_assistant_tokens);

        for _ in 0..n_assistant_tokens {
            reader.read_exact(&mut logit_buffer).map_err(|e| {
                paramecia_core::Error::Msg(format!("Failed to read logit entries: {}", e))
            })?;

            let entries: &[LogitEntry] = bytemuck::cast_slice(&logit_buffer);

            // Split into top-k and tail
            let top_k_entries = &entries[..n_top_k];
            let tail_entries = &entries[n_top_k..];

            top_k_token_ids.push(top_k_entries.iter().map(|e| e.token_id).collect());
            top_k_log_probs.push(top_k_entries.iter().map(|e| e.log_prob()).collect());
            tail_token_ids.push(tail_entries.iter().map(|e| e.token_id).collect());
            tail_log_probs.push(tail_entries.iter().map(|e| e.log_prob()).collect());
        }

        // Read tail mass values (f16)
        let mut tail_mass = Vec::with_capacity(n_assistant_tokens);
        for _ in 0..n_assistant_tokens {
            let mut bytes = [0u8; 2];
            reader.read_exact(&mut bytes).map_err(|e| {
                paramecia_core::Error::Msg(format!("Failed to read tail mass: {}", e))
            })?;
            let f16_val = f16::from_le_bytes(bytes);
            tail_mass.push(f16_val.to_f32());
        }

        // Read expert indices if present
        let version = header.version;
        let expert_indices = if header.has_expert_data() {
            let indices_per_token = n_layers * n_experts_per_tok;
            let mut indices = Vec::with_capacity(n_assistant_tokens);

            if version >= 3 {
                // Version 3+: expert indices stored as u16
                let mut buffer = vec![0u8; indices_per_token * 2];
                for _ in 0..n_assistant_tokens {
                    reader.read_exact(&mut buffer).map_err(|e| {
                        paramecia_core::Error::Msg(format!("Failed to read expert indices: {}", e))
                    })?;
                    let token_indices: Vec<u32> = buffer
                        .chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
                        .collect();
                    indices.push(token_indices);
                }
            } else {
                // Version 2: expert indices stored as u8
                let mut buffer = vec![0u8; indices_per_token];
                for _ in 0..n_assistant_tokens {
                    reader.read_exact(&mut buffer).map_err(|e| {
                        paramecia_core::Error::Msg(format!("Failed to read expert indices: {}", e))
                    })?;
                    indices.push(buffer.iter().map(|&b| b as u32).collect());
                }
            }
            indices
        } else {
            vec![Vec::new(); n_assistant_tokens]
        };

        Ok(TuningData {
            token_ids,
            assistant_mask,
            top_k_token_ids,
            top_k_log_probs,
            tail_token_ids,
            tail_log_probs,
            tail_mass,
            expert_indices,
            vocab_size,
            n_layers,
            n_experts_per_tok,
            source_path: path.to_path_buf(),
        })
    }
}

/// Dataset that manages multiple tuning files and provides batching.
pub struct TuningDataset {
    /// All available tuning file paths
    file_paths: Vec<PathBuf>,
    /// Current position in the file list
    current_idx: usize,
    /// Number of times we've cycled through all files
    epoch: usize,
    /// Shuffle files each epoch
    shuffle: bool,
    /// Random seed for shuffling
    seed: u64,
}

impl TuningDataset {
    /// Create a new dataset from a directory of tuning files.
    pub fn from_directory(path: &Path, shuffle: bool, seed: u64) -> Result<Self> {
        let mut file_paths = Vec::new();

        if path.is_file() {
            if path.extension().map(|e| e == "bin").unwrap_or(false) {
                file_paths.push(path.to_path_buf());
            }
        } else if path.is_dir() {
            Self::collect_files_recursive(path, &mut file_paths)?;
        }

        if file_paths.is_empty() {
            return Err(paramecia_core::Error::Msg(format!(
                "No .bin tuning files found at {:?}",
                path
            )));
        }

        // Sort for deterministic ordering before any shuffling
        file_paths.sort();

        let mut dataset = Self {
            file_paths,
            current_idx: 0,
            epoch: 0,
            shuffle,
            seed,
        };

        if shuffle {
            dataset.shuffle_files();
        }

        Ok(dataset)
    }

    fn collect_files_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| paramecia_core::Error::Msg(format!("Failed to read directory: {}", e)))?;

        for entry in entries {
            let entry = entry
                .map_err(|e| paramecia_core::Error::Msg(format!("Failed to read entry: {}", e)))?;
            let path = entry.path();

            if path.is_dir() {
                Self::collect_files_recursive(&path, files)?;
            } else if path.extension().map(|e| e == "bin").unwrap_or(false) {
                files.push(path);
            }
        }

        Ok(())
    }

    fn shuffle_files(&mut self) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Simple deterministic shuffle based on seed + epoch
        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        self.epoch.hash(&mut hasher);
        let shuffle_seed = hasher.finish();

        // Fisher-Yates shuffle with deterministic RNG
        let n = self.file_paths.len();
        for i in (1..n).rev() {
            // Simple LCG for deterministic randomness
            let mut state = shuffle_seed.wrapping_add(i as u64);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (state as usize) % (i + 1);
            self.file_paths.swap(i, j);
        }
    }

    /// Get the total number of files in the dataset.
    pub fn len(&self) -> usize {
        self.file_paths.len()
    }

    /// Check if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.file_paths.is_empty()
    }

    /// Get the current epoch (number of complete passes through the dataset).
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Get the next batch of tuning data.
    /// Returns None if batch_size files couldn't be loaded (shouldn't happen with cycling).
    pub fn next_batch(&mut self, batch_size: usize) -> Result<Vec<TuningData>> {
        let mut batch = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            // Cycle through files if we reach the end
            if self.current_idx >= self.file_paths.len() {
                self.current_idx = 0;
                self.epoch += 1;
                if self.shuffle {
                    self.shuffle_files();
                }
            }

            let path = &self.file_paths[self.current_idx];
            self.current_idx += 1;

            match TuningReader::read_file(path) {
                Ok(data) => batch.push(data),
                Err(e) => {
                    warn!("Failed to read {:?}: {}", path, e);
                    // Try to continue with next file
                    continue;
                }
            }
        }

        if batch.is_empty() {
            return Err(paramecia_core::Error::Msg(
                "Failed to load any files for batch".to_string(),
            ));
        }

        Ok(batch)
    }

    /// Get next batch, grouping by similar lengths.
    /// If max_seq_len is set, truncate conversations that exceed it.
    pub fn next_batch_grouped(
        &mut self,
        batch_size: usize,
        length_tolerance: f32,
        max_seq_len: Option<usize>,
    ) -> Result<Vec<TuningData>> {
        // Load more data than needed to find similar-length conversations
        let candidate_size = (batch_size as f32 * 2.0) as usize;
        let mut candidates = Vec::with_capacity(candidate_size);

        for _ in 0..candidate_size {
            if self.current_idx >= self.file_paths.len() {
                if candidates.len() >= batch_size {
                    break;
                }
                self.current_idx = 0;
                self.epoch += 1;
                if self.shuffle {
                    self.shuffle_files();
                }
            }

            let path = &self.file_paths[self.current_idx];
            self.current_idx += 1;

            match TuningReader::read_file(path) {
                Ok(mut data) => {
                    // Truncate if exceeds max length
                    if let Some(max_len) = max_seq_len {
                        if data.seq_len() > max_len {
                            data = data.truncate(max_len);
                        }
                    }
                    candidates.push((data.seq_len(), data));
                }
                Err(e) => warn!("Failed to read {:?}: {}", path, e),
            }
        }

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Sort by length
        candidates.sort_by_key(|(len, _)| *len);

        // Find best group of batch_size conversations
        let mut best_batch = Vec::new();
        let mut best_max_len = usize::MAX;

        for i in 0..=candidates.len().saturating_sub(batch_size) {
            let group = &candidates[i..i + batch_size];
            let min_len = group.first().unwrap().0;
            let max_len = group.last().unwrap().0;

            // Check if within tolerance
            if max_len as f32 <= min_len as f32 * (1.0 + length_tolerance) && max_len < best_max_len
            {
                best_max_len = max_len;
                best_batch = group.iter().map(|(_, d)| d.clone()).collect();
            }
        }

        // If no good group found, take first batch_size
        if best_batch.is_empty() {
            best_batch = candidates
                .iter()
                .take(batch_size)
                .map(|(_, d)| d.clone())
                .collect();
        }

        Ok(best_batch)
    }

    /// Reset the dataset to the beginning.
    pub fn reset(&mut self) {
        self.current_idx = 0;
        self.epoch = 0;
        if self.shuffle {
            self.shuffle_files();
        }
    }

    /// Get progress information: (files_processed_this_epoch, total_files, current_epoch)
    pub fn progress(&self) -> (usize, usize, usize) {
        (self.current_idx, self.file_paths.len(), self.epoch)
    }
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
}
