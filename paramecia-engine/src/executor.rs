//! ModelEngine — inference engine with internal actor pattern.
//!
//! `ModelEngineInner` owns the model, tokenizer, and all inference state.
//! `ModelEngine` is a cheap, cloneable handle that forwards async calls
//! to `ModelEngineInner` running on a dedicated OS thread via channels.

use crate::distribution::logits_to_distribution;
use crate::model_actor::{ModelActorHandle, ModelCommand};
use crate::types::*;

use paramecia_model::{
    apply_penalties, DType, GraftComposite, GraftCompositeOptions, GraftLayerSource,
    LogitsProcessor, ModelWeights, PrefixCache, Tensor,
};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
use tokio::sync::{mpsc, oneshot};

/// Threshold for using chunked prefill (matches paramecia-text backend).
const CHUNKED_PREFILL_THRESHOLD: usize = 512;
/// Chunk size for progressive prefill updates.
const PREFILL_PROGRESS_CHUNK_SIZE: usize = 512;

/// In-memory snapshot of model state.
pub(crate) struct MemorySnapshot {
    prefix_cache: PrefixCache,
    tokens: Vec<u32>,
    state_position: usize,
    pending_logits: Option<Tensor>,
}

pub(crate) struct PendingPrediction {
    predicted: Predicted,
    already_appended: bool,
}

// ---------------------------------------------------------------------------
// ModelEngineInner — the actual model-owning state (runs on actor thread)
// ---------------------------------------------------------------------------

/// Core engine that owns inference state. Runs on a dedicated actor thread.
pub(crate) struct ModelEngineInner {
    pub(crate) model: ModelWeights,
    pub(crate) device: paramecia_model::Device,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) tokens: Vec<u32>,
    pub(crate) state_position: usize,
    pub(crate) awaiting_commit: bool,
    pub(crate) pending_logits: Option<Tensor>,
    pub(crate) sampler: LogitsProcessor,
    pub(crate) top_k: usize,
    pub(crate) tail_samples: usize,
    pub(crate) repeat_penalty: f32,
    pub(crate) presence_penalty: f32,
    pub(crate) penalty_last_n: usize,
    pub(crate) snapshot_dir: PathBuf,
    pub(crate) model_path: PathBuf,
    pub(crate) memory_snapshots: HashMap<Uuid, MemorySnapshot>,
    pub(crate) next_snapshot_counter: u64,
    pub(crate) mtp_inference_speculation_depth: Option<usize>,
    pub(crate) mtp_allow_inference_override: bool,
    pub(crate) pending_speculative_predictions: VecDeque<PendingPrediction>,
    pub(crate) pending_speculative_rollback: Option<MemorySnapshot>,
    pub(crate) pending_speculative_commits: Vec<u32>,
    pub(crate) pending_commit_expected_token: Option<u32>,
    pub(crate) pending_commit_preappended: bool,
    pub(crate) pending_metadata: Vec<(String, MetadataValue)>,
}

impl ModelEngineInner {
    fn validate_token_ids(&self, token_ids: &[u32], context: &str) -> Result<(), Error> {
        let vocab_size = self.model.vocab_size();
        if vocab_size == 0 {
            return Err(Error::ModelError(
                "Model vocabulary size is zero; cannot validate token IDs.".to_string(),
            ));
        }

        if let Some((pos, bad_id)) = token_ids
            .iter()
            .copied()
            .enumerate()
            .find(|(_, id)| u64::from(*id) >= vocab_size as u64)
        {
            return Err(Error::ModelError(format!(
                "Token ID {} at position {} is out of range for model vocab size {} while processing {}. \
This usually indicates tokenizer/model mismatch (for example, using a Qwen3.5 tokenizer with Qwen3-Next weights).",
                bad_id, pos, vocab_size, context
            )));
        }
        Ok(())
    }

    fn validate_single_token_id(&self, token_id: u32, context: &str) -> Result<(), Error> {
        self.validate_token_ids(&[token_id], context)
    }

    /// Get a reference to the underlying model weights.
    pub fn model(&self) -> &ModelWeights {
        &self.model
    }

    /// Get a mutable reference to the underlying model weights.
    pub fn model_mut(&mut self) -> &mut ModelWeights {
        &mut self.model
    }

    /// Get the device used by this engine.
    pub fn device(&self) -> &paramecia_model::Device {
        &self.device
    }

    /// Get a reference to the tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get the current token history.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Get the current state position (how far KV cache is synced).
    pub fn state_position(&self) -> usize {
        self.state_position
    }

    /// Get the model path.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Unload inference model, clearing all state.
    pub fn unload_model(&mut self) {
        self.model.clear_kv_cache();
        self.tokens.clear();
        self.state_position = 0;
        self.awaiting_commit = false;
        self.pending_logits = None;
        self.clear_pending_speculative_predictions();
    }

    fn clear_pending_speculative_predictions(&mut self) {
        self.pending_speculative_predictions.clear();
        self.pending_speculative_rollback = None;
        self.pending_speculative_commits.clear();
        self.pending_commit_expected_token = None;
        self.pending_commit_preappended = false;
    }

    /// Generate a new snapshot UUID using the monotonic counter.
    fn next_snapshot_uuid(&mut self) -> Uuid {
        let id = self.next_snapshot_counter;
        self.next_snapshot_counter += 1;
        (0, id)
    }

    async fn maybe_queue_speculative_predictions(&mut self) -> Result<(), Error> {
        let Some(speculation_depth) = self.mtp_inference_speculation_depth else {
            return Ok(());
        };
        if speculation_depth == 0 || self.pending_logits.is_some() || !self.model.has_mtp() {
            return Ok(());
        }
        if self.state_position > self.tokens.len() {
            return Err(Error::InvalidState(
                "Model state position is out of sync with token history.".into(),
            ));
        }
        let ctxt_len = self.tokens.len().saturating_sub(self.state_position);
        if ctxt_len != 1 {
            return Ok(());
        }

        if self.mtp_allow_inference_override {
            let prefix_cache = self
                .model
                .save_prefix_cache(self.tokens[..self.state_position].to_vec());
            self.pending_speculative_rollback = Some(MemorySnapshot {
                prefix_cache,
                tokens: self.tokens.clone(),
                state_position: self.state_position,
                pending_logits: self.pending_logits.clone(),
            });
            self.pending_speculative_commits.clear();
        }

        let predictions = match self.predict_tokens_speculative(speculation_depth).await {
            Ok(predictions) => predictions,
            Err(err) => {
                self.pending_speculative_rollback = None;
                self.pending_speculative_commits.clear();
                return Err(err);
            }
        };
        if predictions.is_empty() {
            self.pending_speculative_rollback = None;
            self.pending_speculative_commits.clear();
            return Err(Error::ModelError(
                "Speculative decoding returned no predictions.".into(),
            ));
        }

        self.pending_speculative_predictions
            .extend(predictions.into_iter().map(|predicted| PendingPrediction {
                predicted,
                already_appended: true,
            }));
        Ok(())
    }

    fn replay_forced_token(&mut self, token: u32) -> Result<(), Error> {
        if self.state_position > self.tokens.len() {
            return Err(Error::InvalidState(
                "Model state position is out of sync with token history.".into(),
            ));
        }
        let start_pos = self.state_position;
        let ctxt: Vec<u32> = self.tokens[start_pos..].to_vec();
        if ctxt.is_empty() {
            return Err(Error::InvalidState(
                "Cannot replay override with empty unprocessed context.".into(),
            ));
        }

        let input = Tensor::new(ctxt.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| Error::ModelError(e.to_string()))?;
        let result = if ctxt.len() > CHUNKED_PREFILL_THRESHOLD {
            self.model.forward_chunked(&input, start_pos, None)
        } else {
            self.model.forward(&input, start_pos)
        };
        result.map_err(|e| Error::ModelError(e.to_string()))?;

        self.state_position = self.tokens.len();
        self.pending_logits = None;
        self.tokens.push(token);
        Ok(())
    }

    fn override_speculative_commit(&mut self, token: u32) -> Result<(), Error> {
        let rollback = self.pending_speculative_rollback.take().ok_or_else(|| {
            Error::InvalidState(
                "Speculative override requested but rollback state is unavailable.".into(),
            )
        })?;

        let mut decisions = self.pending_speculative_commits.clone();
        decisions.push(token);

        self.model
            .restore_prefix_cache(&rollback.prefix_cache)
            .map_err(|e| Error::ModelError(e.to_string()))?;
        self.tokens = rollback.tokens;
        self.state_position = rollback.state_position;
        self.pending_logits = rollback.pending_logits;

        self.pending_speculative_predictions.clear();
        self.pending_speculative_commits.clear();
        self.pending_commit_expected_token = None;
        self.pending_commit_preappended = false;

        for forced in decisions {
            self.replay_forced_token(forced)?;
        }

        Ok(())
    }

    pub async fn fill_context(
        &mut self,
        text: &str,
        progress_tx: Option<mpsc::Sender<Result<u32, Error>>>,
    ) -> Result<u32, Error> {
        let encode_result = self.tokenizer.encode(text, true);
        let encoding = match encode_result {
            Ok(e) => e,
            Err(e) => {
                let err = Error::TokenizeError(e.to_string());
                if let Some(tx) = &progress_tx {
                    let _ = tx.send(Err(err.clone())).await;
                }
                return Err(err);
            }
        };
        let new_tokens = encoding.get_ids().to_vec();
        self.fill_context_tokens(&new_tokens, progress_tx).await
    }

    /// Fill context with a list of ModelInput items.
    ///
    /// Dispatches each input type:
    /// - Text(s) → tokenize then fill_context_tokens
    /// - Tokens(ids) → fill_context_tokens directly
    /// - Soft(soft_tokens) → compute per-position weighted embeddings from dark-knowledge, stack into [1, seq_len, hidden], then forward_with_embeddings
    pub async fn fill_context_inputs(
        &mut self,
        inputs: &[ModelInput],
        progress_tx: Option<mpsc::Sender<Result<u32, Error>>>,
    ) -> Result<u32, Error> {
        let mut total = 0u32;
        for input in inputs {
            match input {
                ModelInput::Text(text) => {
                    let n = self.fill_context(text, progress_tx.clone()).await?;
                    total += n;
                }
                ModelInput::Tokens(ids) => {
                    self.validate_token_ids(ids, "fill_context_inputs:tokens")?;
                    let n = self.fill_context_tokens(ids, progress_tx.clone()).await?;
                    total += n;
                }
                ModelInput::Soft(soft_tokens) => {
                    // Compute a weighted embedding per soft-token position from the
                    // dark-knowledge distribution, then stack into [1, seq_len, hidden_dim].
                    let embed_weights = self.model.embedding_weights();
                    let device = embed_weights.device().clone();

                    if soft_tokens.is_empty() {
                        continue;
                    }

                    // Validate all predicted token IDs and dark-knowledge token IDs
                    let mut all_ids: Vec<u32> = Vec::new();
                    for st in soft_tokens {
                        all_ids.push(st.predicted);
                        for entry in &st.dark_knowledge {
                            all_ids.push(entry.token_id);
                        }
                    }
                    self.validate_token_ids(&all_ids, "fill_context_inputs:soft")?;

                    let seq_len = soft_tokens.len();

                    // Compute one weighted embedding per position
                    let mut position_embeddings: Vec<Tensor> = Vec::with_capacity(seq_len);
                    for st in soft_tokens {
                        if st.dark_knowledge.is_empty() {
                            // No dark knowledge — use the predicted token's embedding directly
                            let id_tensor = Tensor::new(&[st.predicted], &device)
                                .map_err(|e| Error::ModelError(e.to_string()))?;
                            let embed = embed_weights
                                .index_select(&id_tensor, 0)
                                .map_err(|e| Error::ModelError(e.to_string()))?;
                            // Shape: [1, hidden_dim] → squeeze to [hidden_dim]
                            let squeezed = embed.reshape((embed.dims().last().copied().unwrap_or(0),))
                                .map_err(|e| Error::ModelError(e.to_string()))?;
                            position_embeddings.push(squeezed);
                        } else {
                            // Compute weighted embedding from dark-knowledge distribution
                            let ids: Vec<u32> = st.dark_knowledge.iter().map(|e| e.token_id).collect();
                            let mut probs: Vec<f32> =
                                st.dark_knowledge.iter().map(|e| e.log_prob.exp()).collect();

                            // Normalize probabilities
                            let prob_sum: f32 = probs.iter().sum();
                            if prob_sum > 0.0 {
                                for p in &mut probs {
                                    *p /= prob_sum;
                                }
                            }

                            let ids_tensor = Tensor::new(ids.as_slice(), &device)
                                .map_err(|e| Error::ModelError(e.to_string()))?;
                            let sparse_embeds = embed_weights
                                .index_select(&ids_tensor, 0)
                                .map_err(|e| Error::ModelError(e.to_string()))?;
                            let probs_tensor = Tensor::new(probs.as_slice(), &device)
                                .and_then(|t| t.unsqueeze(1))
                                .map_err(|e| Error::ModelError(e.to_string()))?;
                            let weighted = sparse_embeds
                                .broadcast_mul(&probs_tensor)
                                .and_then(|t| t.sum(0))
                                .map_err(|e| Error::ModelError(e.to_string()))?;
                            position_embeddings.push(weighted);
                        }
                    }

                    // Stack positions into [seq_len, hidden_dim], then add batch dim → [1, seq_len, hidden_dim]
                    let stacked = Tensor::stack(&position_embeddings, 0)
                        .map_err(|e| Error::ModelError(e.to_string()))?;
                    let embedding = stacked
                        .unsqueeze(0)
                        .and_then(|t| t.to_dtype(self.model.dtype()))
                        .map_err(|e| Error::ModelError(e.to_string()))?;

                    self.clear_pending_speculative_predictions();
                    self.awaiting_commit = false;

                    let offset = self.state_position;
                    let logits = self
                        .model
                        .forward_with_embeddings(&embedding, offset)
                        .map_err(|e| Error::ModelError(e.to_string()))?;

                    self.state_position += seq_len;
                    self.pending_logits = Some(logits);
                    // Soft tokens don't add to the token history (they're synthetic)
                    total += seq_len as u32;

                    if let Some(tx) = &progress_tx {
                        let _ = tx.send(Ok(total)).await;
                    }
                }
            }
        }
        Ok(total)
    }

    pub async fn predict_token(&mut self) -> Result<Predicted, Error> {
        if self.awaiting_commit {
            return Err(Error::InvalidState(
                "Must call commit_token before calling predict_token again".into(),
            ));
        }

        if self.pending_speculative_predictions.is_empty() {
            self.maybe_queue_speculative_predictions().await?;
        }

        if let Some(pending) = self.pending_speculative_predictions.pop_front() {
            self.pending_commit_expected_token = Some(pending.predicted.token_id);
            self.pending_commit_preappended = pending.already_appended;
            self.awaiting_commit = true;
            return Ok(pending.predicted);
        }

        if self.tokens.is_empty() {
            return Err(Error::InvalidState(
                "No tokens in context. Call fill_context first.".into(),
            ));
        }

        // Use cached logits from fill_context if available
        let logits = if let Some(cached) = self.pending_logits.take() {
            cached
        } else {
            let start_pos = self.state_position;
            let ctxt: Vec<u32> = self.tokens[start_pos..].to_vec();

            if ctxt.is_empty() {
                return Err(Error::InvalidState(
                    "No uncommitted tokens to process. Call commit_token first.".into(),
                ));
            }

            let device = self.device.clone();
            let input = Tensor::new(ctxt.as_slice(), &device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| Error::ModelError(e.to_string()))?;

            let logits = self
                .model
                .forward(&input, start_pos)
                .map_err(|e| Error::ModelError(e.to_string()))?;

            self.state_position = self.tokens.len();
            logits
        };

        // Extract logits for last position
        let logits = logits
            .squeeze(0)
            .map_err(|e| Error::ModelError(e.to_string()))?;
        let logits = if logits.rank() == 2 {
            match logits.dims()[0].checked_sub(1) {
                Some(last_idx) => logits
                    .get(last_idx)
                    .map_err(|e| Error::ModelError(e.to_string()))?,
                None => return Err(Error::ModelError("Empty logits".into())),
            }
        } else {
            logits
        };

        let logits = logits
            .to_dtype(DType::F32)
            .map_err(|e| Error::ModelError(e.to_string()))?;

        // Capture expert indices before sampling
        let expert_indices = self.take_expert_indices();

        // Build distribution info from raw (pre-penalty) logits
        let distribution = logits_to_distribution(&logits, self.top_k, self.tail_samples)
            .map_err(|e| Error::ModelError(e.to_string()))?;

        // Apply penalties for sampling
        let penalty_start = self.tokens.len().saturating_sub(self.penalty_last_n);
        let penalty_context = &self.tokens[penalty_start..];
        let penalized = apply_penalties(
            &logits,
            self.repeat_penalty,
            self.presence_penalty,
            penalty_context,
        )
        .map_err(|e| Error::ModelError(e.to_string()))?;

        // Sample from penalized logits
        let token_id = self
            .sampler
            .sample(&penalized)
            .map_err(|e| Error::ModelError(e.to_string()))?;

        // Decode token text
        let text = self
            .tokenizer
            .decode(&[token_id], false)
            .ok()
            .filter(|s| !s.is_empty());

        self.pending_commit_expected_token = None;
        self.pending_commit_preappended = false;
        self.awaiting_commit = true;
        Ok(Predicted {
            token_id,
            text,
            top_k: distribution.top_k,
            tail: distribution.tail,
            tail_mass: distribution.tail_mass,
            expert_indices,
        })
    }

    pub async fn predict_completion(
        &mut self,
        mut cancel_rx: oneshot::Receiver<()>,
        tx: mpsc::Sender<Result<Predicted, Error>>,
    ) {
        // Get stop token IDs
        let im_end_token = self
            .tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| self.tokenizer.token_to_id("<|endoftext|>"));
        let eos_token = self.tokenizer.token_to_id("<|endoftext|>");

        let mut count = 0;

        loop {
            tokio::select! {
                _ = &mut cancel_rx => {
                    let _ = tx.send(Err(Error::InvalidState(
                        "Completion cancelled by user request".into(),
                    ))).await;
                    break;
                }
                _ = async {} => {
                    let prediction = match self.predict_token().await {
                        Ok(pred) => pred,
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            return;
                        }
                    };

                    let token_id = prediction.token_id;
                    let should_stop = Some(token_id) == im_end_token || Some(token_id) == eos_token;

                    if tx.send(Ok(prediction)).await.is_err() {
                        // Receiver dropped before this token could be consumed.
                        // Clear pending commit state so future generations are not poisoned.
                        self.awaiting_commit = false;
                        self.clear_pending_speculative_predictions();
                        self.prepare_for_generation();
                        return;
                    }

                    count += 1;

                    // Auto-commit the token so executor state remains consistent
                    // even when this token is a stop marker.
                    if let Err(err) = self.commit_token(token_id).await {
                        self.awaiting_commit = false;
                        self.clear_pending_speculative_predictions();
                        self.prepare_for_generation();
                        let _ = tx.send(Err(err)).await;
                        return;
                    }

                    if should_stop {
                        return;
                    }

                    if count >= 4096 {
                        let _ = tx.send(Err(Error::ModelError(
                            "Maximum completion length (4096 tokens) exceeded".into(),
                        ))).await;
                        return;
                    }
                }
            }
        }
    }

    /// Run batched completion for N sequences simultaneously.
    ///
    /// Each input sequence is tokenized, left-padded to the max length, and
    /// processed through the model in a single batched forward pass per step.
    /// Streams `Vec<Predicted>` (one per sequence) per decode step.
    ///
    /// This method is self-contained: it clears existing state, runs to
    /// completion, and restores batch=1 state afterward.
    pub async fn predict_completions_batched(
        &mut self,
        inputs: Vec<Vec<ModelInput>>,
        mut cancel_rx: oneshot::Receiver<()>,
        tx: mpsc::Sender<Result<Vec<Predicted>, Error>>,
    ) {
        // Validate: no pre-existing batch=1 state
        if self.state_position > 0 || !self.tokens.is_empty() {
            let _ = tx
                .send(Err(Error::InvalidState(
                    "Cannot start batched completions with existing model state. \
                     Call reset_state first."
                        .into(),
                )))
                .await;
            return;
        }

        let n = inputs.len();
        if n == 0 {
            let _ = tx
                .send(Err(Error::InvalidState(
                    "No input sequences provided".into(),
                )))
                .await;
            return;
        }

        // Tokenize all sequences
        let mut seq_tokens: Vec<Vec<u32>> = Vec::with_capacity(n);
        for (i, seq_inputs) in inputs.iter().enumerate() {
            let mut tokens = Vec::new();
            for input in seq_inputs {
                match input {
                    ModelInput::Text(text) => match self.tokenizer.encode(text.as_str(), true) {
                        Ok(encoding) => tokens.extend_from_slice(encoding.get_ids()),
                        Err(e) => {
                            let _ = tx
                                .send(Err(Error::TokenizeError(format!("Sequence {}: {}", i, e))))
                                .await;
                            return;
                        }
                    },
                    ModelInput::Tokens(ids) => tokens.extend_from_slice(ids),
                    ModelInput::Soft(_) => {
                        let _ = tx
                            .send(Err(Error::InvalidState(
                                "Soft inputs are not supported in batched mode".into(),
                            )))
                            .await;
                        return;
                    }
                }
            }
            if let Err(e) = self.validate_token_ids(&tokens, &format!("batched sequence {}", i)) {
                let _ = tx.send(Err(e)).await;
                return;
            }
            seq_tokens.push(tokens);
        }

        // Left-pad to max length
        let max_len = seq_tokens.iter().map(|s| s.len()).max().unwrap_or(0);
        if max_len == 0 {
            let _ = tx
                .send(Err(Error::InvalidState(
                    "All input sequences are empty".into(),
                )))
                .await;
            return;
        }

        let pad_token_id: u32 = 0;
        let mut padding_lengths = Vec::with_capacity(n);
        let mut padded_input = Vec::with_capacity(n * max_len);
        for seq in &seq_tokens {
            let pad_len = max_len - seq.len();
            padding_lengths.push(pad_len);
            padded_input.extend(std::iter::repeat_n(pad_token_id, pad_len));
            padded_input.extend_from_slice(seq);
        }

        // Initialize model for batch=N
        if let Err(e) = self.model.reinit_caches(n) {
            let _ = tx
                .send(Err(Error::ModelError(format!(
                    "Failed to reinit caches: {}",
                    e
                ))))
                .await;
            self.cleanup_batched();
            return;
        }

        // Build [N, max_len] input tensor
        let input_tensor = match Tensor::from_slice(&padded_input, (n, max_len), &self.device) {
            Ok(t) => t,
            Err(e) => {
                let _ = tx
                    .send(Err(Error::ModelError(format!(
                        "Failed to create input tensor: {}",
                        e
                    ))))
                    .await;
                self.cleanup_batched();
                return;
            }
        };

        // Prefill
        let logits = match self
            .model
            .forward_batched(&input_tensor, 0, &padding_lengths)
        {
            Ok(l) => l,
            Err(e) => {
                let _ = tx
                    .send(Err(Error::ModelError(format!("Prefill failed: {}", e))))
                    .await;
                self.cleanup_batched();
                return;
            }
        };

        let mut offset = max_len;
        let mut current_logits = logits;

        // Stop tokens
        let im_end_token = self
            .tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| self.tokenizer.token_to_id("<|endoftext|>"));
        let eos_token = self.tokenizer.token_to_id("<|endoftext|>");

        // Per-sequence state
        let mut active = vec![true; n];
        let mut per_seq_tokens: Vec<Vec<u32>> = seq_tokens;

        // Create per-sequence samplers with derived seeds
        let base_seed = 299792458u64; // default seed
        let sampling = self.sampler.sampling().clone();
        let mut per_seq_samplers: Vec<LogitsProcessor> = (0..n)
            .map(|i| {
                LogitsProcessor::from_sampling(base_seed.wrapping_add(i as u64), sampling.clone())
            })
            .collect();

        // Decode loop
        for count in 0..4096 {
            // Check cancellation
            if cancel_rx.try_recv().is_ok() {
                let _ = tx
                    .send(Err(Error::InvalidState(
                        "Batched completion cancelled".into(),
                    )))
                    .await;
                break;
            }

            // Sample N tokens from logits
            let mut predictions = Vec::with_capacity(n);
            let mut next_tokens = Vec::with_capacity(n);

            for i in 0..n {
                if !active[i] {
                    // Inactive: emit padding prediction (EOS)
                    predictions.push(Predicted {
                        token_id: eos_token.unwrap_or(0),
                        text: None,
                        top_k: Vec::new(),
                        tail: Vec::new(),
                        tail_mass: 0.0,
                        expert_indices: Vec::new(),
                    });
                    next_tokens.push(eos_token.unwrap_or(0));
                    continue;
                }

                // Extract logits for batch element i
                let row_logits = match current_logits.get(i) {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = tx
                            .send(Err(Error::ModelError(format!(
                                "Failed to extract logits for seq {}: {}",
                                i, e
                            ))))
                            .await;
                        self.cleanup_batched();
                        return;
                    }
                };

                // Squeeze if needed (may be [1, vocab] or [vocab])
                let row_logits = if row_logits.rank() == 2 {
                    match row_logits.squeeze(0) {
                        Ok(l) => l,
                        Err(e) => {
                            let _ = tx.send(Err(Error::ModelError(e.to_string()))).await;
                            self.cleanup_batched();
                            return;
                        }
                    }
                } else {
                    row_logits
                };

                let row_logits = match row_logits.to_dtype(DType::F32) {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = tx.send(Err(Error::ModelError(e.to_string()))).await;
                        self.cleanup_batched();
                        return;
                    }
                };

                // Build distribution from raw logits
                let distribution =
                    match logits_to_distribution(&row_logits, self.top_k, self.tail_samples) {
                        Ok(d) => d,
                        Err(e) => {
                            let _ = tx.send(Err(Error::ModelError(e.to_string()))).await;
                            self.cleanup_batched();
                            return;
                        }
                    };

                // Apply per-sequence penalties
                let penalty_start = per_seq_tokens[i].len().saturating_sub(self.penalty_last_n);
                let penalty_context = &per_seq_tokens[i][penalty_start..];
                let penalized = match apply_penalties(
                    &row_logits,
                    self.repeat_penalty,
                    self.presence_penalty,
                    penalty_context,
                ) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = tx.send(Err(Error::ModelError(e.to_string()))).await;
                        self.cleanup_batched();
                        return;
                    }
                };

                // Sample
                let token_id = match per_seq_samplers[i].sample(&penalized) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = tx.send(Err(Error::ModelError(e.to_string()))).await;
                        self.cleanup_batched();
                        return;
                    }
                };

                // Check stop condition
                let should_stop = Some(token_id) == im_end_token || Some(token_id) == eos_token;
                if should_stop {
                    active[i] = false;
                }

                per_seq_tokens[i].push(token_id);

                let text = self
                    .tokenizer
                    .decode(&[token_id], false)
                    .ok()
                    .filter(|s| !s.is_empty());

                predictions.push(Predicted {
                    token_id,
                    text,
                    top_k: distribution.top_k,
                    tail: distribution.tail,
                    tail_mass: distribution.tail_mass,
                    expert_indices: Vec::new(),
                });
                next_tokens.push(token_id);
            }

            // Send predictions for this step
            if tx.send(Ok(predictions)).await.is_err() {
                break;
            }

            // Check if all sequences are done
            if active.iter().all(|&a| !a) {
                break;
            }

            // Check max length
            if count >= 4095 {
                let _ = tx
                    .send(Err(Error::ModelError(
                        "Maximum batched completion length (4096 tokens) exceeded".into(),
                    )))
                    .await;
                break;
            }

            // Build next input: [N, 1] tensor
            let next_input = match Tensor::from_slice(&next_tokens, (n, 1), &self.device) {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx
                        .send(Err(Error::ModelError(format!(
                            "Failed to create next input: {}",
                            e
                        ))))
                        .await;
                    break;
                }
            };

            // Decode forward pass
            match self
                .model
                .forward_batched(&next_input, offset, &padding_lengths)
            {
                Ok(l) => {
                    current_logits = l;
                    offset += 1;
                }
                Err(e) => {
                    let _ = tx
                        .send(Err(Error::ModelError(format!(
                            "Decode forward failed: {}",
                            e
                        ))))
                        .await;
                    break;
                }
            }
        }

        // Cleanup: restore batch=1 for subsequent single-sequence use
        self.cleanup_batched();
    }

    /// Restore model to batch=1 state after batched completion.
    fn cleanup_batched(&mut self) {
        let _ = self.model.reinit_caches(1);
        self.tokens.clear();
        self.state_position = 0;
        self.pending_logits = None;
        self.awaiting_commit = false;
        self.clear_pending_speculative_predictions();
    }

    pub async fn commit_token(&mut self, token: u32) -> Result<(), Error> {
        self.validate_single_token_id(token, "commit_token")?;
        if !self.awaiting_commit {
            return Err(Error::InvalidState(
                "No pending predict_token to commit. Call predict_token first.".into(),
            ));
        }

        if self.pending_commit_preappended {
            let expected = self.pending_commit_expected_token.ok_or_else(|| {
                Error::InvalidState("Missing expected token for speculative commit.".into())
            })?;
            if token != expected {
                if self.mtp_allow_inference_override {
                    self.override_speculative_commit(token)?;
                } else {
                    return Err(Error::InvalidState(format!(
                        "Speculative commit mismatch: expected token {}, got {}.",
                        expected, token
                    )));
                }
            } else if self.mtp_allow_inference_override
                && self.pending_speculative_rollback.is_some()
            {
                self.pending_speculative_commits.push(token);
            }
        } else {
            self.tokens.push(token);
        }

        self.pending_commit_expected_token = None;
        self.pending_commit_preappended = false;
        self.awaiting_commit = false;
        if self.pending_speculative_predictions.is_empty() {
            self.pending_speculative_rollback = None;
            self.pending_speculative_commits.clear();
        }
        Ok(())
    }

    /// Take an in-memory snapshot of current state. Returns UUID-keyed Snapshot.
    pub async fn take_snapshot(&mut self) -> Result<Snapshot, Error> {
        let prefix_cache = self
            .model
            .save_prefix_cache(self.tokens[..self.state_position].to_vec());
        let id = self.next_snapshot_uuid();
        self.memory_snapshots.insert(
            id,
            MemorySnapshot {
                prefix_cache,
                tokens: self.tokens.clone(),
                state_position: self.state_position,
                pending_logits: self.pending_logits.clone(),
            },
        );
        Ok(Snapshot {
            id,
            state_position: self.state_position as u64,
            num_tokens: self.tokens.len() as u32,
        })
    }

    /// Persist an in-memory snapshot to disk. Consumes the memory snapshot.
    /// Returns both the PersistedSnapshot handle and the file path.
    pub async fn persist_snapshot(
        &mut self,
        snapshot_id: Uuid,
    ) -> Result<(PersistedSnapshot, PathBuf), Error> {
        let snapshot = self
            .memory_snapshots
            .remove(&snapshot_id)
            .ok_or_else(|| Error::SnapshotError(format!("Snapshot {:?} not found", snapshot_id)))?;

        // Save current state, restore target, write to disk, restore current
        let current_cache = self
            .model
            .save_prefix_cache(self.tokens[..self.state_position].to_vec());

        self.model
            .restore_prefix_cache(&snapshot.prefix_cache)
            .map_err(|e| Error::SnapshotError(e.to_string()))?;

        let filename = format!("{}.bin", uuid::Uuid::new_v4());
        let path = self.snapshot_dir.join(&filename);
        self.model
            .save_snapshot(
                &path,
                snapshot.state_position,
                &snapshot.tokens[..snapshot.state_position],
            )
            .map_err(|e| Error::SnapshotError(e.to_string()))?;

        self.model
            .restore_prefix_cache(&current_cache)
            .map_err(|e| Error::SnapshotError(e.to_string()))?;

        Ok((PersistedSnapshot { id: snapshot_id }, path))
    }

    /// Drop an in-memory snapshot, freeing memory.
    pub fn drop_snapshot(&mut self, snapshot_id: Uuid) -> Result<(), Error> {
        self.memory_snapshots
            .remove(&snapshot_id)
            .ok_or_else(|| Error::SnapshotError(format!("Snapshot {:?} not found", snapshot_id)))?;
        Ok(())
    }

    /// Restore a snapshot by UUID (in-memory lookup).
    pub fn restore_snapshot(&mut self, snapshot_id: &Uuid) -> Result<(), Error> {
        let snap = self
            .memory_snapshots
            .get(snapshot_id)
            .ok_or_else(|| Error::SnapshotError(format!("Snapshot {:?} not found", snapshot_id)))?;
        self.model
            .restore_prefix_cache(&snap.prefix_cache)
            .map_err(|e| Error::SnapshotError(e.to_string()))?;
        self.tokens = snap.tokens.clone();
        self.state_position = snap.state_position;
        self.awaiting_commit = false;
        self.pending_logits = snap.pending_logits.clone();
        self.clear_pending_speculative_predictions();
        Ok(())
    }

    /// Load a persisted snapshot from disk into an in-memory snapshot.
    /// Does not restore model state — use `restore_snapshot` afterwards.
    pub fn load_persisted_snapshot(&mut self, path: &Path) -> Result<Snapshot, Error> {
        // Save current model state
        let current_cache = self
            .model
            .save_prefix_cache(self.tokens[..self.state_position].to_vec());

        // Load disk snapshot into model (restores layer states)
        let disk_snapshot = self
            .model
            .load_snapshot(path)
            .map_err(|e| Error::SnapshotError(e.to_string()))?;

        let state_position = disk_snapshot.state_position;
        let tokens = disk_snapshot.tokens;
        let num_tokens = tokens.len();

        // Capture the now-restored state as a PrefixCache
        let loaded_cache = self
            .model
            .save_prefix_cache(tokens[..state_position].to_vec());

        // Restore original model state
        self.model
            .restore_prefix_cache(&current_cache)
            .map_err(|e| Error::SnapshotError(e.to_string()))?;

        // Store as in-memory snapshot
        let id = self.next_snapshot_uuid();
        self.memory_snapshots.insert(
            id,
            MemorySnapshot {
                prefix_cache: loaded_cache,
                tokens,
                state_position,
                pending_logits: None,
            },
        );
        Ok(Snapshot {
            id,
            state_position: state_position as u64,
            num_tokens: num_tokens as u32,
        })
    }

    /// Get the pending metadata updates.
    pub fn pending_metadata(&self) -> &[(String, MetadataValue)] {
        &self.pending_metadata
    }

    pub async fn reseed(&mut self, seed: u64) -> Result<(), Error> {
        self.sampler.reseed(seed);
        Ok(())
    }

    /// Fill context with pre-tokenized token IDs.
    pub async fn fill_context_tokens(
        &mut self,
        new_tokens: &[u32],
        progress_tx: Option<mpsc::Sender<Result<u32, Error>>>,
    ) -> Result<u32, Error> {
        self.clear_pending_speculative_predictions();
        self.awaiting_commit = false;

        if new_tokens.is_empty() {
            // Still process any unprocessed buffered tokens (e.g., after snapshot restore)
            if self.state_position >= self.tokens.len() {
                if let Some(tx) = &progress_tx {
                    let _ = tx.send(Ok(0)).await;
                }
                return Ok(0);
            }
        } else {
            self.validate_token_ids(new_tokens, "fill_context_tokens:new_tokens")?;
            self.tokens.extend_from_slice(new_tokens);
        }

        // Run forward pass on unprocessed tokens
        let start_pos = self.state_position;
        let ctxt: Vec<u32> = self.tokens[start_pos..].to_vec();
        self.validate_token_ids(&ctxt, "fill_context_tokens:context_window")?;
        let total = ctxt.len() as u32;
        let device = self.device.clone();
        let input = Tensor::new(ctxt.as_slice(), &device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| Error::ModelError(e.to_string()))?;

        if ctxt.len() > CHUNKED_PREFILL_THRESHOLD {
            let mut current_offset = start_pos;
            let mut chunk_start = 0usize;
            let mut processed = 0u32;
            let mut final_logits: Option<Tensor> = None;

            while chunk_start < ctxt.len() {
                let chunk_end = (chunk_start + PREFILL_PROGRESS_CHUNK_SIZE).min(ctxt.len());
                let chunk_len = chunk_end - chunk_start;
                let is_last_chunk = chunk_end == ctxt.len();

                let chunk_input = Tensor::new(&ctxt[chunk_start..chunk_end], &device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| Error::ModelError(e.to_string()))?;

                if is_last_chunk {
                    let logits = self
                        .model
                        .forward(&chunk_input, current_offset)
                        .map_err(|e| Error::ModelError(e.to_string()))?;
                    final_logits = Some(logits);
                } else {
                    self.model
                        .forward_state_update_only(&chunk_input, current_offset)
                        .map_err(|e| Error::ModelError(e.to_string()))?;
                }

                processed += chunk_len as u32;
                if let Some(tx) = &progress_tx {
                    let _ = tx.send(Ok(processed)).await;
                }

                current_offset += chunk_len;
                chunk_start = chunk_end;
            }

            self.state_position = self.tokens.len();
            self.awaiting_commit = false;
            self.pending_logits = final_logits;
            Ok(total)
        } else {
            let result = self.model.forward(&input, start_pos);

            match result {
                Ok(logits) => {
                    self.state_position = self.tokens.len();
                    self.awaiting_commit = false;
                    self.pending_logits = Some(logits);
                    if let Some(tx) = &progress_tx {
                        let _ = tx.send(Ok(total)).await;
                    }
                    Ok(total)
                }
                Err(e) => {
                    let err = Error::ModelError(e.to_string());
                    if let Some(tx) = &progress_tx {
                        let _ = tx.send(Err(err.clone())).await;
                    }
                    Err(err)
                }
            }
        }
    }

    /// Lightweight state clear without reloading model weights.
    pub fn reset_state(&mut self) {
        self.model.clear_kv_cache();
        self.tokens.clear();
        self.state_position = 0;
        self.awaiting_commit = false;
        self.pending_logits = None;
        self.clear_pending_speculative_predictions();
    }

    /// Prepare for generation after snapshot restore with no new tokens.
    pub fn prepare_for_generation(&mut self) {
        if self.pending_logits.is_none()
            && !self.awaiting_commit
            && self.state_position == self.tokens.len()
            && self.state_position > 0
        {
            self.state_position -= 1;
        }
    }

    /// Take and flatten expert indices from the last forward pass.
    fn take_expert_indices(&mut self) -> Vec<u32> {
        self.model
            .take_last_expert_indices()
            .map(|tensors| {
                tensors
                    .iter()
                    .flat_map(|t| t.flatten_all().unwrap().to_vec1::<u32>().unwrap())
                    .collect::<Vec<u32>>()
            })
            .unwrap_or_default()
    }

    pub async fn predict_tokens_speculative(
        &mut self,
        num_speculative: usize,
    ) -> Result<Vec<Predicted>, Error> {
        if !self.model.has_mtp() {
            return Err(Error::InvalidState(
                "Model does not have MTP weights".into(),
            ));
        }

        let start_pos = self.state_position;
        if start_pos > self.tokens.len() {
            return Err(Error::InvalidState(
                "Model state position is out of sync with token history.".into(),
            ));
        }
        let ctxt: Vec<u32> = self.tokens[start_pos..].to_vec();

        if ctxt.is_empty() {
            return Err(Error::InvalidState(
                "No tokens to process for speculative decoding".into(),
            ));
        }
        if ctxt.len() != 1 {
            return Err(Error::InvalidState(
                "MTP speculative decoding requires exactly one unprocessed token.".into(),
            ));
        }

        let device = self.device.clone();
        let input = Tensor::new(ctxt.as_slice(), &device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| Error::ModelError(e.to_string()))?;

        // Run speculative step
        let spec_result = self
            .model
            .speculative_step(&input, start_pos, num_speculative)
            .map_err(|e| Error::ModelError(e.to_string()))?;

        let main_token: u32 = spec_result
            .main_token
            .to_vec0()
            .map_err(|e| Error::ModelError(e.to_string()))?;

        let mut results = Vec::new();

        // Create prediction for main token (no distribution for speculative path)
        let main_text = self
            .tokenizer
            .decode(&[main_token], false)
            .ok()
            .filter(|s| !s.is_empty());
        results.push(Predicted {
            token_id: main_token,
            text: main_text,
            top_k: Vec::new(),
            tail: Vec::new(),
            tail_mass: 0.0,
            expert_indices: Vec::new(),
        });

        self.tokens.push(main_token);

        // Verify speculative tokens if any
        if !spec_result.spec_tokens.is_empty() {
            let spec_vec: Vec<u32> = spec_result
                .spec_tokens
                .iter()
                .map(|t| {
                    if t.dims().is_empty() {
                        t.to_vec0::<u32>()
                    } else {
                        t.flatten_all()?.to_vec1::<u32>().map(|v| v[0])
                    }
                })
                .collect::<paramecia_model::Result<Vec<_>>>()
                .map_err(|e| Error::ModelError(e.to_string()))?;

            // Draft tensor: [main_token, spec_0, spec_1, ...]
            let mut draft_vec = vec![main_token];
            draft_vec.extend_from_slice(&spec_vec);
            let draft_tensor = Tensor::new(draft_vec.as_slice(), &device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| Error::ModelError(e.to_string()))?;

            let verify_offset = start_pos + 1;
            let verify_result = self
                .model
                .verify_and_accept(&draft_tensor, spec_result.snapshots, verify_offset)
                .map_err(|e| Error::ModelError(e.to_string()))?;

            // Accept verified tokens
            for &accepted_token in spec_vec.iter().take(verify_result.num_accepted) {
                let text = self
                    .tokenizer
                    .decode(&[accepted_token], false)
                    .ok()
                    .filter(|s| !s.is_empty());
                results.push(Predicted {
                    token_id: accepted_token,
                    text,
                    top_k: Vec::new(),
                    tail: Vec::new(),
                    tail_mass: 0.0,
                    expert_indices: Vec::new(),
                });
                self.tokens.push(accepted_token);
            }

            // Update state position
            self.state_position = verify_offset + verify_result.num_accepted + 1;

            // Bonus token from next_logits
            if let Some(next_logits) = verify_result.next_logits {
                let next_logits = next_logits
                    .to_dtype(DType::F32)
                    .map_err(|e| Error::ModelError(e.to_string()))?;

                // Apply penalties for bonus token sampling
                let penalty_start = self.tokens.len().saturating_sub(self.penalty_last_n);
                let penalty_context = &self.tokens[penalty_start..];
                let penalized = apply_penalties(
                    &next_logits,
                    self.repeat_penalty,
                    self.presence_penalty,
                    penalty_context,
                )
                .map_err(|e| Error::ModelError(e.to_string()))?;

                let next_token = self
                    .sampler
                    .sample(&penalized)
                    .map_err(|e| Error::ModelError(e.to_string()))?;

                let text = self
                    .tokenizer
                    .decode(&[next_token], false)
                    .ok()
                    .filter(|s| !s.is_empty());

                results.push(Predicted {
                    token_id: next_token,
                    text,
                    top_k: Vec::new(),
                    tail: Vec::new(),
                    tail_mass: 0.0,
                    expert_indices: Vec::new(),
                });
                self.tokens.push(next_token);
            }
        } else {
            self.state_position = self.tokens.len();
        }

        self.awaiting_commit = false;
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// ModelEngine — cheap cloneable handle (public API)
// ---------------------------------------------------------------------------

/// Inference engine handle. Internally communicates with a dedicated
/// actor thread via message passing, so GPU operations never block the caller.
#[derive(Clone)]
pub struct ModelEngine {
    actor: std::sync::Arc<ModelActorHandle>,
    device: paramecia_model::Device,
    model_path: PathBuf,
    tokenizer: Tokenizer,
    has_mtp: bool,
    num_layers: usize,
    num_experts_per_token: usize,
    inference_mtp_speculation_depth: Option<usize>,
    training_mtp_speculation_depth: Option<usize>,
    mtp_allow_inference_override: bool,
}

impl ModelEngine {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        actor: std::sync::Arc<ModelActorHandle>,
        device: paramecia_model::Device,
        model_path: PathBuf,
        tokenizer: Tokenizer,
        has_mtp: bool,
        num_layers: usize,
        num_experts_per_token: usize,
        inference_mtp_speculation_depth: Option<usize>,
        training_mtp_speculation_depth: Option<usize>,
        mtp_allow_inference_override: bool,
    ) -> Self {
        Self {
            actor,
            device,
            model_path,
            tokenizer,
            has_mtp,
            num_layers,
            num_experts_per_token,
            inference_mtp_speculation_depth,
            training_mtp_speculation_depth,
            mtp_allow_inference_override,
        }
    }

    /// Get the device used by this engine.
    pub fn device(&self) -> &paramecia_model::Device {
        &self.device
    }

    /// Get the model path.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Get a reference to the tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Whether the model has MTP (multi-token prediction) weights.
    pub fn has_mtp(&self) -> bool {
        self.has_mtp
    }

    /// Configured speculation depth for inference MTP.
    pub fn inference_mtp_speculation_depth(&self) -> Option<usize> {
        self.inference_mtp_speculation_depth
    }

    /// Configured speculation depth for training MTP.
    pub fn training_mtp_speculation_depth(&self) -> Option<usize> {
        self.training_mtp_speculation_depth
    }

    /// Whether mismatched commits are allowed during speculative MTP decoding.
    pub fn mtp_allow_inference_override(&self) -> bool {
        self.mtp_allow_inference_override
    }

    /// Get the number of layers in the model.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get the number of experts selected per token per layer.
    pub fn num_experts_per_token(&self) -> usize {
        self.num_experts_per_token
    }

    /// Get the current token history.
    pub async fn tokens(&self) -> Result<Vec<u32>, Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::GetTokens { respond: tx })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))
    }

    /// Get the current state position (how far KV cache is synced).
    pub async fn state_position(&self) -> Result<usize, Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::GetStatePosition { respond: tx })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))
    }

    /// Fill context with pre-tokenized token IDs.
    pub async fn fill_context_tokens(
        &self,
        tokens: &[u32],
    ) -> Result<mpsc::Receiver<Result<u32, Error>>, Error> {
        let (progress_tx, progress_rx) = mpsc::channel(64);
        self.actor
            .sender()
            .send(ModelCommand::FillContextTokens {
                tokens: tokens.to_vec(),
                progress: progress_tx,
            })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        Ok(progress_rx)
    }

    /// Fill context with ModelInput items (text, tokens, or soft prompts).
    pub async fn fill_context_inputs(
        &self,
        inputs: &[ModelInput],
    ) -> Result<mpsc::Receiver<Result<u32, Error>>, Error> {
        let (progress_tx, progress_rx) = mpsc::channel(64);
        self.actor
            .sender()
            .send(ModelCommand::FillContextInputs {
                inputs: inputs.to_vec(),
                progress: progress_tx,
            })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        Ok(progress_rx)
    }

    /// Lightweight state clear without reloading model weights.
    pub async fn reset_state(&self) -> Result<(), Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::ResetState { respond: tx })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))?
    }

    /// Prepare for generation after snapshot restore with no new tokens.
    pub async fn prepare_for_generation(&self) -> Result<(), Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::PrepareForGeneration { respond: tx })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))?
    }

    /// Fill context with the given text.
    pub async fn fill_context(
        &self,
        text: &str,
    ) -> Result<mpsc::Receiver<Result<u32, Error>>, Error> {
        let (progress_tx, progress_rx) = mpsc::channel(64);
        self.actor
            .sender()
            .send(ModelCommand::FillContext {
                text: text.to_string(),
                progress: progress_tx,
            })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        Ok(progress_rx)
    }

    /// Predict the next token.
    pub async fn predict_token(&self) -> Result<Predicted, Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::PredictToken { respond: tx })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))?
    }

    /// Start a streaming completion.
    pub async fn predict_completion(
        &self,
    ) -> Result<
        (
            mpsc::Receiver<Result<Predicted, Error>>,
            oneshot::Sender<()>,
        ),
        Error,
    > {
        let (result_tx, result_rx) = mpsc::channel(32);
        let (cancel_tx, cancel_rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::PredictCompletion {
                respond: result_tx,
                cancel_rx,
            })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        Ok((result_rx, cancel_tx))
    }

    /// Start a batched streaming completion for N sequences.
    pub async fn predict_completions_batched(
        &self,
        inputs: &[Vec<ModelInput>],
    ) -> Result<
        (
            mpsc::Receiver<Result<Vec<Predicted>, Error>>,
            oneshot::Sender<()>,
        ),
        Error,
    > {
        let (result_tx, result_rx) = mpsc::channel(32);
        let (cancel_tx, cancel_rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::PredictCompletionsBatched {
                inputs: inputs.to_vec(),
                respond: result_tx,
                cancel_rx,
            })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        Ok((result_rx, cancel_tx))
    }

    /// Commit a predicted token to the context.
    pub async fn commit_token(&self, token: u32) -> Result<(), Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::CommitToken { token, respond: tx })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))?
    }

    /// Take an in-memory snapshot of current state.
    pub async fn take_snapshot(&self) -> Result<Snapshot, Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::TakeSnapshot { respond: tx })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))?
    }

    /// Persist an in-memory snapshot to disk. Returns (PersistedSnapshot, file_path).
    pub async fn persist_snapshot(
        &self,
        snapshot_id: Uuid,
    ) -> Result<(PersistedSnapshot, PathBuf), Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::PersistSnapshot {
                snapshot_id,
                respond: tx,
            })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))?
    }

    /// Restore state from an in-memory snapshot.
    pub async fn restore_snapshot(&self, snapshot_id: Uuid) -> Result<(), Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::RestoreSnapshot {
                snapshot_id,
                respond: tx,
            })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))?
    }

    /// Load a persisted snapshot from disk into an in-memory snapshot.
    /// Does not restore model state — use `restore_snapshot` afterwards.
    pub async fn load_persisted_snapshot(&self, path: PathBuf) -> Result<Snapshot, Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::LoadPersistedSnapshot { path, respond: tx })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))?
    }

    /// Drop an in-memory snapshot.
    pub async fn drop_snapshot(&self, snapshot_id: Uuid) -> Result<(), Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::DropSnapshot {
                snapshot_id,
                respond: tx,
            })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))?
    }

    /// Unload the model.
    pub async fn unload_model(&self) -> Result<(), Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::UnloadModel { respond: tx })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))?
    }

    /// Reseed the sampler.
    pub async fn reseed(&self, seed: u64) -> Result<(), Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::Reseed { seed, respond: tx })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))?
    }

    /// Save a checkpoint, returning the path where it was written.
    pub async fn save_checkpoint(&self) -> Result<PathBuf, Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::SaveCheckpoint { respond: tx })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))?
    }

    /// Send a training command to the unified actor.
    pub async fn train_model(
        &self,
        sample_rx: mpsc::Receiver<TrainingSample>,
        is_validation: bool,
        loss_tx: mpsc::Sender<Result<StepResult, Error>>,
        cancel_rx: oneshot::Receiver<()>,
    ) -> Result<(), Error> {
        self.actor
            .sender()
            .send(ModelCommand::TrainModel {
                sample_rx,
                is_validation,
                loss_tx,
                cancel_rx,
            })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        Ok(())
    }

    /// Set hyperparameters on the unified actor.
    pub async fn set_hyper_parameters(&self, update: HyperParameterUpdate) -> Result<(), Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::SetHyperParameters {
                update: Box::new(update),
                respond: tx,
            })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))?
    }

    /// Perturb model weights in the positive direction.
    pub async fn perturb_up(&self, seed: Option<u64>) -> Result<(), Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::PerturbUp {
                seed,
                respond: tx,
            })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))?
    }

    /// Perturb model weights in the negative direction.
    pub async fn perturb_down(&self) -> Result<(), Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::PerturbDown {
                respond: tx,
            })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))?
    }

    /// Calculate loss difference and update weights.
    pub async fn update(&self, loss_up: f32, loss_down: f32) -> Result<(), Error> {
        let (tx, rx) = oneshot::channel();
        self.actor
            .sender()
            .send(ModelCommand::Update {
                loss_up,
                loss_down,
                respond: tx,
            })
            .await
            .map_err(|_| Error::ModelError("Model actor stopped".into()))?;
        rx.await
            .map_err(|_| Error::ModelError("Model actor dropped response".into()))?
    }
}

// ---------------------------------------------------------------------------
// Standalone fusion helper (operates on GGUF files, not loaded models)
// ---------------------------------------------------------------------------

/// Fuse multiple models using task arithmetic, writing the result to `output`.
///
/// `base` is the path to the base GGUF model.
/// `members` is a list of `(path, contribution)` pairs.
/// `output` is the exact path for the merged GGUF.
/// `strategy` controls how quantization dtype conflicts are resolved.
pub fn fuse_models(
    base: &Path,
    members: &[(PathBuf, f32)],
    output: &Path,
    strategy: QuantConflictStrategy,
) -> Result<(), Error> {
    let opt_strategy = match strategy {
        QuantConflictStrategy::Reject => paramecia_opt::QuantConflictStrategy::Reject,
        QuantConflictStrategy::Highest => paramecia_opt::QuantConflictStrategy::Highest,
        QuantConflictStrategy::Lowest => paramecia_opt::QuantConflictStrategy::Lowest,
    };
    let options = paramecia_opt::FuseOptions {
        base: base.to_path_buf(),
        models: members.to_vec(),
        output: output.to_path_buf(),
        quant_conflict_strategy: opt_strategy,
    };
    paramecia_opt::fuse_models(&options)
        .map_err(|e| Error::CheckpointError(format!("Fusion failed: {e}")))?;
    Ok(())
}

/// Update GGUF metadata entries in-place for an existing checkpoint/model file.
pub fn update_model_metadata(
    checkpoint_path: &Path,
    metadata_updates: &HashMap<String, MetadataValue>,
) -> Result<(), Error> {
    let gguf_updates: HashMap<String, paramecia_model::gguf_file::Value> = metadata_updates
        .iter()
        .map(|(k, v)| (k.clone(), metadata_value_to_gguf(v)))
        .collect();

    paramecia_opt::update_gguf_metadata(checkpoint_path, &gguf_updates)
        .map_err(|e| Error::CheckpointError(format!("Failed to update metadata: {e}")))?;
    Ok(())
}

/// Prune experts from a GGUF model and write the result to `output`.
pub fn prune_experts(
    source: &Path,
    output: &Path,
    retained_indices: &[Vec<u32>],
) -> Result<(), Error> {
    let retained_indices_u16: Vec<Vec<u16>> = retained_indices
        .iter()
        .map(|layer| {
            layer
                .iter()
                .map(|&idx| {
                    u16::try_from(idx).map_err(|_| {
                        Error::CheckpointError(format!(
                            "Expert index {idx} exceeds u16::MAX and cannot be encoded"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    paramecia_opt::prune_experts(&paramecia_opt::PruneExpertsOptions {
        input_model: source.to_path_buf(),
        output_model: output.to_path_buf(),
        retained_indices: retained_indices_u16,
        verbose: false,
    })
    .map_err(|e| Error::CheckpointError(format!("Expert pruning failed: {e}")))?;

    Ok(())
}

/// Prune transformer layers from a GGUF model and write the result to `output`.
pub fn prune_layers(source: &Path, output: &Path, retained_layers: &[u32]) -> Result<(), Error> {
    paramecia_opt::prune_layers(&paramecia_opt::PruneLayersOptions {
        input_model: source.to_path_buf(),
        output_model: output.to_path_buf(),
        retained_layers: retained_layers.to_vec(),
        verbose: false,
    })
    .map_err(|e| Error::CheckpointError(format!("Layer pruning failed: {e}")))?;

    Ok(())
}

/// Graft a composite model from resolved GGUF paths and write it to `output`.
pub fn graft_composite_from_paths(
    embedding: &Path,
    layers: &[(PathBuf, u32)],
    lm_head: &Path,
    mtp_head: &Path,
    output: &Path,
) -> Result<(), Error> {
    let layers = layers
        .iter()
        .map(|(path, layer_idx)| GraftLayerSource {
            path: path.clone(),
            layer_idx: *layer_idx,
        })
        .collect();

    paramecia_model::graft_composite(&GraftCompositeOptions {
        composite: GraftComposite {
            embedding: embedding.to_path_buf(),
            layers,
            lm_head: lm_head.to_path_buf(),
            mtp_head: mtp_head.to_path_buf(),
        },
        output: output.to_path_buf(),
        verbose: false,
    })
    .map_err(|e| Error::CheckpointError(format!("Graft failed: {e}")))?;

    Ok(())
}

/// Describe a model by reading GGUF headers without loading weights.
///
/// Returns layer/expert counts, total/active parameter counts, metadata, and tensor info.
pub fn describe_model(path: &Path) -> Result<ModelDescription, Error> {
    use paramecia_model::gguf_file;

    let mut file = std::fs::File::open(path)
        .map_err(|e| Error::ModelError(format!("Failed to open model: {e}")))?;
    let ct = gguf_file::Content::read(&mut file)
        .map_err(|e| Error::ModelError(format!("Failed to read GGUF: {e}")))?;

    let md = &ct.metadata;

    let n_layers = md
        .get("qwen35moe.block_count")
        .or_else(|| md.get("qwen35.block_count"))
        .or_else(|| md.get("qwen3_5_moe.block_count"))
        .or_else(|| md.get("qwen3_5.block_count"))
        .or_else(|| md.get("qwen3next.block_count"))
        .or_else(|| md.get("qwen3moe.block_count"))
        .or_else(|| md.get("llama.block_count"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0);

    let n_experts_metadata = md
        .get("qwen35moe.expert_count")
        .or_else(|| md.get("qwen35.expert_count"))
        .or_else(|| md.get("qwen3_5_moe.expert_count"))
        .or_else(|| md.get("qwen3_5.expert_count"))
        .or_else(|| md.get("qwen3next.expert_count"))
        .or_else(|| md.get("qwen3moe.expert_count"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0);

    let n_experts_per_tok = md
        .get("qwen35moe.expert_used_count")
        .or_else(|| md.get("qwen35.expert_used_count"))
        .or_else(|| md.get("qwen3_5_moe.expert_used_count"))
        .or_else(|| md.get("qwen3_5.expert_used_count"))
        .or_else(|| md.get("qwen3next.expert_used_count"))
        .or_else(|| md.get("qwen3moe.expert_used_count"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(n_experts_metadata);

    // Determine true active expert count by reading expert_mask tensors (if present).
    // Pruned models store blk.{layer}.expert_mask with 0.0 for kept and -inf for pruned.
    let n_experts = if n_experts_metadata > 0 {
        let device = paramecia_model::Device::Cpu;
        let mut min_active: Option<u32> = None;
        for layer in 0..n_layers {
            let mask_name = format!("blk.{layer}.expert_mask");
            if ct.tensor_infos.contains_key(&mask_name) {
                if let Ok(qt) = ct.tensor(&mut file, &mask_name, &device) {
                    if let Ok(t) = qt.dequantize(&device) {
                        if let Ok(vals) = t.to_vec1::<f32>() {
                            let active = vals.iter().filter(|v| v.is_finite()).count() as u32;
                            min_active = Some(match min_active {
                                Some(prev) => prev.min(active),
                                None => active,
                            });
                        }
                    }
                }
            }
        }
        min_active.unwrap_or(n_experts_metadata)
    } else {
        0
    };

    // Convert tensors and compute parameter counts
    let mut n_total_parameters: u64 = 0;
    let mut inactive_parameters: u64 = 0;
    let mut tensors = Vec::with_capacity(ct.tensor_infos.len());

    for (name, info) in &ct.tensor_infos {
        let elem_count = info.shape.elem_count() as u64;
        n_total_parameters += elem_count;

        // For MoE expert tensors, subtract inactive expert fraction.
        // Active per token = n_experts_per_tok out of n_experts_metadata total slots.
        if n_experts_metadata > 0
            && n_experts_per_tok < n_experts_metadata
            && (name.contains("ffn_gate_exps.weight")
                || name.contains("ffn_up_exps.weight")
                || name.contains("ffn_down_exps.weight"))
        {
            inactive_parameters += elem_count * (n_experts_metadata - n_experts_per_tok) as u64
                / n_experts_metadata as u64;
        }

        tensors.push(GgufTensor {
            name: name.clone(),
            shape: info.shape.dims().iter().map(|&d| d as u32).collect(),
            dtype: convert_ggml_dtype(info.ggml_dtype),
        });
    }

    let n_active_parameters = n_total_parameters - inactive_parameters;

    // Convert metadata
    let metadata: Vec<(String, MetadataValue)> = ct
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), convert_gguf_value(v)))
        .collect();

    Ok(ModelDescription {
        n_experts,
        n_layers,
        n_total_parameters,
        n_active_parameters,
        metadata,
        tensors,
    })
}

fn convert_ggml_dtype(dt: paramecia_model::GgmlDType) -> GgmlDtype {
    use paramecia_model::GgmlDType;
    match dt {
        GgmlDType::F32 => GgmlDtype::F32,
        GgmlDType::F16 => GgmlDtype::F16,
        GgmlDType::BF16 => GgmlDtype::Bf16,
        GgmlDType::Q4_0 => GgmlDtype::Q4_0,
        GgmlDType::Q4_1 => GgmlDtype::Q4_1,
        GgmlDType::Q5_0 => GgmlDtype::Q5_0,
        GgmlDType::Q5_1 => GgmlDtype::Q5_1,
        GgmlDType::Q8_0 => GgmlDtype::Q8_0,
        GgmlDType::Q8_1 => GgmlDtype::Q8_1,
        GgmlDType::Q2K => GgmlDtype::Q2K,
        GgmlDType::Q3K => GgmlDtype::Q3K,
        GgmlDType::Q4K => GgmlDtype::Q4K,
        GgmlDType::Q5K => GgmlDtype::Q5K,
        GgmlDType::Q6K => GgmlDtype::Q6K,
        GgmlDType::Q8K => GgmlDtype::Q8K,
    }
}

fn convert_gguf_value(v: &paramecia_model::gguf_file::Value) -> MetadataValue {
    use paramecia_model::gguf_file::Value;
    match v {
        Value::U8(x) => MetadataValue::U8(*x),
        Value::I8(x) => MetadataValue::I8(*x),
        Value::U16(x) => MetadataValue::U16(*x),
        Value::I16(x) => MetadataValue::I16(*x),
        Value::U32(x) => MetadataValue::U32(*x),
        Value::I32(x) => MetadataValue::I32(*x),
        Value::U64(x) => MetadataValue::U64(*x),
        Value::I64(x) => MetadataValue::I64(*x),
        Value::F32(x) => MetadataValue::F32(*x),
        Value::F64(x) => MetadataValue::F64(*x),
        Value::Bool(x) => MetadataValue::Bool(*x),
        Value::String(x) => MetadataValue::String(x.clone()),
        Value::Array(arr) => MetadataValue::Array(arr.iter().map(convert_gguf_value).collect()),
    }
}

fn metadata_value_to_gguf(v: &MetadataValue) -> paramecia_model::gguf_file::Value {
    use paramecia_model::gguf_file::Value;
    match v {
        MetadataValue::U8(x) => Value::U8(*x),
        MetadataValue::I8(x) => Value::I8(*x),
        MetadataValue::U16(x) => Value::U16(*x),
        MetadataValue::I16(x) => Value::I16(*x),
        MetadataValue::U32(x) => Value::U32(*x),
        MetadataValue::I32(x) => Value::I32(*x),
        MetadataValue::U64(x) => Value::U64(*x),
        MetadataValue::I64(x) => Value::I64(*x),
        MetadataValue::F32(x) => Value::F32(*x),
        MetadataValue::F64(x) => Value::F64(*x),
        MetadataValue::Bool(x) => Value::Bool(*x),
        MetadataValue::String(x) => Value::String(x.clone()),
        MetadataValue::Array(arr) => Value::Array(arr.iter().map(metadata_value_to_gguf).collect()),
    }
}

#[cfg(test)]
mod tests {
    use tokenizers::Tokenizer;

    /// Download the Qwen3-Next tokenizer from HuggingFace.
    fn load_tokenizer() -> Tokenizer {
        let api = hf_hub::api::sync::Api::new().expect("HF API init");
        let repo = api.repo(hf_hub::Repo::with_revision(
            "Qwen/Qwen3-Next-80B-A3B-Instruct".to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));
        let path = repo.get("tokenizer.json").expect("download tokenizer");
        Tokenizer::from_file(path).expect("parse tokenizer")
    }

    #[test]
    #[ignore] // requires network access to download tokenizer from HuggingFace
    fn special_tokens_encode_as_single_ids() {
        let tokenizer = load_tokenizer();

        // Verify token_to_id lookups work for special tokens
        let im_start_id = tokenizer
            .token_to_id("<|im_start|>")
            .expect("<|im_start|> should be in vocab");
        let im_end_id = tokenizer
            .token_to_id("<|im_end|>")
            .expect("<|im_end|> should be in vocab");
        let eos_id = tokenizer
            .token_to_id("<|endoftext|>")
            .expect("<|endoftext|> should be in vocab");

        assert_eq!(im_start_id, 151644);
        assert_eq!(im_end_id, 151645);
        assert_eq!(eos_id, 151643);

        // Encode a typical chat-template string and verify special tokens
        // appear as single token IDs (not split into characters)
        let text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(text, false).expect("encode");
        let ids = encoding.get_ids();

        // <|im_start|> must be the very first token
        assert_eq!(
            ids[0], im_start_id,
            "first token should be <|im_start|>, got {}",
            ids[0]
        );

        // Collect all positions where our special tokens appear
        let im_start_positions: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == im_start_id)
            .map(|(i, _)| i)
            .collect();
        let im_end_positions: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == im_end_id)
            .map(|(i, _)| i)
            .collect();

        // The template has 3 <|im_start|> and 2 <|im_end|>
        assert_eq!(
            im_start_positions.len(),
            3,
            "expected 3 <|im_start|> tokens, found {} at positions {:?}",
            im_start_positions.len(),
            im_start_positions
        );
        assert_eq!(
            im_end_positions.len(),
            2,
            "expected 2 <|im_end|> tokens, found {} at positions {:?}",
            im_end_positions.len(),
            im_end_positions
        );

        // Each <|im_end|> must be a single token (not split across multiple IDs).
        // Verify by round-tripping: decode just that single token position.
        for &pos in &im_end_positions {
            let decoded = tokenizer.decode(&[ids[pos]], false).expect("decode");
            assert_eq!(
                decoded, "<|im_end|>",
                "token at position {} should decode to <|im_end|>, got {:?}",
                pos, decoded
            );
        }
        for &pos in &im_start_positions {
            let decoded = tokenizer.decode(&[ids[pos]], false).expect("decode");
            assert_eq!(
                decoded, "<|im_start|>",
                "token at position {} should decode to <|im_start|>, got {:?}",
                pos, decoded
            );
        }
    }
}
