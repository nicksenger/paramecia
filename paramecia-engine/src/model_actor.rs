//! Unified model actor - owns ModelEngineInner + training state, processes
//! all commands via message passing.
//!
//! Runs on a dedicated OS thread to avoid tokio task migration breaking GPU
//! context thread-locality. Replaces the previous separate inference_actor.rs
//! and training_actor.rs with a single actor per model.

use crate::executor::ModelEngineInner;
use crate::training::sample_to_tuning_data;
use crate::types::*;

use paramecia_model::gguf_file;
use paramecia_model::GgmlDType;
use paramecia_opt::{
    run_training_step_with_grad_accum, save_trained_model, DecomposedZOState, DistillationLoss,
    EpsilonConfig, MtpLossConfig, OptimizeTensors, ParamsQuZO, QuZO,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info};

/// Commands sent to the unified model actor.
pub(crate) enum ModelCommand {
    // -- Inference commands --
    FillContext {
        text: String,
        progress: mpsc::Sender<Result<u32, Error>>,
    },
    FillContextTokens {
        tokens: Vec<u32>,
        progress: mpsc::Sender<Result<u32, Error>>,
    },
    FillContextInputs {
        inputs: Vec<ModelInput>,
        progress: mpsc::Sender<Result<u32, Error>>,
    },
    PredictToken {
        respond: oneshot::Sender<Result<Predicted, Error>>,
    },
    PredictCompletion {
        respond: mpsc::Sender<Result<Predicted, Error>>,
        limit: u32,
        cancel_rx: oneshot::Receiver<()>,
    },
    PredictCompletionsBatched {
        inputs: Vec<Vec<ModelInput>>,
        respond: mpsc::Sender<Result<Vec<Predicted>, Error>>,
        cancel_rx: oneshot::Receiver<()>,
    },
    CommitToken {
        token: u32,
        respond: oneshot::Sender<Result<(), Error>>,
    },
    TakeSnapshot {
        respond: oneshot::Sender<Result<Snapshot, Error>>,
    },
    PersistSnapshot {
        snapshot_id: Uuid,
        respond: oneshot::Sender<Result<(PersistedSnapshot, PathBuf), Error>>,
    },
    DropSnapshot {
        snapshot_id: Uuid,
        respond: oneshot::Sender<Result<(), Error>>,
    },
    RestoreSnapshot {
        snapshot_id: Uuid,
        respond: oneshot::Sender<Result<(), Error>>,
    },
    LoadPersistedSnapshot {
        path: PathBuf,
        respond: oneshot::Sender<Result<Snapshot, Error>>,
    },
    Reseed {
        seed: u64,
        respond: oneshot::Sender<Result<(), Error>>,
    },
    ResetState {
        respond: oneshot::Sender<Result<(), Error>>,
    },
    PrepareForGeneration {
        respond: oneshot::Sender<Result<(), Error>>,
    },
    GetTokens {
        respond: oneshot::Sender<Vec<u32>>,
    },
    GetStatePosition {
        respond: oneshot::Sender<usize>,
    },

    // -- Training commands --
    TrainModel {
        sample_rx: mpsc::Receiver<TrainingSample>,
        is_validation: bool,
        loss_tx: mpsc::Sender<Result<StepResult, Error>>,
        cancel_rx: oneshot::Receiver<()>,
    },
    SetHyperParameters {
        update: Box<HyperParameterUpdate>,
        respond: oneshot::Sender<Result<(), Error>>,
    },

    // -- Training-ext (decomposed ZO) commands --
    PerturbUp {
        seed: Option<u64>,
        respond: oneshot::Sender<Result<(), Error>>,
    },
    PerturbDown {
        respond: oneshot::Sender<Result<(), Error>>,
    },
    Update {
        loss_up: f32,
        loss_down: f32,
        respond: oneshot::Sender<Result<(), Error>>,
    },

    // -- Lifecycle --
    SaveCheckpoint {
        respond: oneshot::Sender<Result<PathBuf, Error>>,
    },
    UnloadModel {
        respond: oneshot::Sender<Result<(), Error>>,
    },
}

/// Configuration for the training subsystem within the unified actor.
#[derive(Clone)]
pub struct TrainingConfig {
    pub minibatch_size: usize,
    pub n_grad_steps: usize,
    pub lr: f64,
    pub epsilon: f64,
    pub optimize_tensors: String,
    /// Always regenerate perturbations one tensor at a time.
    pub lazy_perturbations: bool,
    /// Maximum aggregate size of eagerly retained f32 perturbations.
    ///
    /// The optimizer automatically switches to lazy perturbations when the
    /// selected tensors would exceed this budget. A value of zero therefore
    /// forces the bounded-memory path.
    pub perturbation_memory_budget_bytes: u64,
    pub mtp_speculation_depth: Option<usize>,
    pub mtp_decay_factor: f64,
    pub checkpoint_dir: PathBuf,
    pub clip_threshold: f64,
    pub lb_loss: f64,
    pub z_loss: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            minibatch_size: 2,
            n_grad_steps: 2,
            lr: 0.0001,
            epsilon: 0.001,
            optimize_tensors: "all".to_string(),
            lazy_perturbations: false,
            perturbation_memory_budget_bytes: 4 * 1024 * 1024 * 1024,
            mtp_speculation_depth: None,
            mtp_decay_factor: 0.5,
            checkpoint_dir: PathBuf::from("/tmp/paramecia-checkpoints"),
            clip_threshold: 10.0,
            lb_loss: 0.01,
            z_loss: 0.001,
        }
    }
}

/// Active training session state — stored between commands so the actor loop
/// can interleave inference and training work via `select!`.
struct ActiveTraining {
    sample_rx: mpsc::Receiver<TrainingSample>,
    loss_tx: mpsc::Sender<Result<StepResult, Error>>,
    cancel_rx: oneshot::Receiver<()>,
    optimizer: Option<QuZO>,
    loss_fn: DistillationLoss,
    mtp_config: Option<MtpLossConfig>,
    is_validation: bool,
    sample_buffer: Vec<paramecia_opt::TuningData>,
    id_buffer: Vec<Uuid>,
    minibatch_size: usize,
    n_grad_steps: usize,
}

/// Run the unified model actor - processes commands and training samples
/// concurrently via `tokio::select!` on a dedicated OS thread.
pub(crate) async fn run_model_actor(
    mut rx: mpsc::Receiver<ModelCommand>,
    mut executor: ModelEngineInner,
    training_config: TrainingConfig,
) {
    // Training subsystem config is lazily initialized when first training command arrives.
    let mut training_state: Option<TrainingSubsystem> = None;
    let checkpoint_dir = training_config.checkpoint_dir.clone();

    // Active training session — when Some, the select! loop also polls for
    // training samples, allowing inference commands to be interleaved.
    let mut active_training: Option<ActiveTraining> = None;

    // Decomposed ZO state — persists between perturb_up/perturb_down/update calls.
    let mut decomposed_zo: Option<DecomposedZO> = None;

    loop {
        // Use select! to interleave command processing with training sample ingestion.
        // When no training session is active, we only wait on commands.
        enum Event {
            Command(Option<ModelCommand>),
            TrainingSample(Option<TrainingSample>),
            TrainingCancelled,
        }

        let event = if let Some(ref mut at) = active_training {
            tokio::select! {
                biased;
                // Always check commands first (inference, lifecycle, etc.)
                cmd = rx.recv() => Event::Command(cmd),
                // Check training cancellation
                _ = &mut at.cancel_rx => Event::TrainingCancelled,
                // Ingest training samples when the actor isn't busy with a command
                sample = at.sample_rx.recv() => Event::TrainingSample(sample),
            }
        } else {
            Event::Command(rx.recv().await)
        };

        match event {
            Event::TrainingCancelled => {
                if let Some(at) = active_training.take() {
                    let _ = at
                        .loss_tx
                        .send(Err(Error::InvalidState("Training cancelled".into())))
                        .await;
                }
                continue;
            }
            Event::TrainingSample(sample) => {
                match sample {
                    Some(exec_sample) => {
                        ingest_training_sample(
                            &mut executor,
                            &mut training_state,
                            active_training.as_mut().unwrap(),
                            exec_sample,
                        )
                        .await;
                    }
                    None => {
                        // Sample stream closed — flush remaining buffer then end session
                        flush_training_buffer(
                            &mut executor,
                            &mut training_state,
                            active_training.as_mut().unwrap(),
                        )
                        .await;
                        active_training = None;
                    }
                }
                continue;
            }
            Event::Command(None) => break, // Command channel closed
            Event::Command(Some(cmd)) => {
                // Fall through to command processing below
                // (cmd is used in the match below)
                // We need to assign it for the match
                // Restructure: process the command inline
                process_command(
                    cmd,
                    &mut rx,
                    &mut executor,
                    &mut training_state,
                    &mut active_training,
                    &mut decomposed_zo,
                    &training_config,
                    &checkpoint_dir,
                )
                .await;
            }
        }
    }
}

/// Training subsystem state, lazily created within the unified actor.
struct TrainingSubsystem {
    mtp_config: Option<MtpLossConfig>,
    loss_config: paramecia_opt::DistillationLossConfig,
    epsilon_config: EpsilonConfig,
    optimize_tensors: OptimizeTensors,
    lazy_perturbations: bool,
    perturbation_memory_budget_bytes: u64,
    minibatch_size: usize,
    n_grad_steps: usize,
    lr: f64,
    epsilon: f64,
    clip_threshold: f64,
    error_feedback: Option<ErrorFeedbackMode>,
    error_decay: f64,
    error_gain: f64,
    step: usize,
    best_loss: f64,
}

fn ensure_training_subsystem<'a>(
    state: &'a mut Option<TrainingSubsystem>,
    executor: &mut ModelEngineInner,
    config: &TrainingConfig,
) -> &'a mut TrainingSubsystem {
    if state.is_none() {
        // Initialize training subsystem config.

        let mtp_config = match config.mtp_speculation_depth {
            Some(depth) if depth > 0 && executor.model().has_mtp() => Some(MtpLossConfig {
                num_depths: depth,
                decay_factor: config.mtp_decay_factor,
                normalize_weights: true,
            }),
            _ => None,
        };
        *state = Some(TrainingSubsystem {
            mtp_config,
            loss_config: paramecia_opt::DistillationLossConfig {
                distill_weight: 1.0,
                z_loss_weight: config.z_loss,
                lb_loss_weight: config.lb_loss,
                temperature: 1.0,
                min_prob: 1e-10,
            },
            epsilon_config: EpsilonConfig::default(),
            optimize_tensors: paramecia_opt::parse_optimize_tensors(&config.optimize_tensors),
            lazy_perturbations: config.lazy_perturbations,
            perturbation_memory_budget_bytes: config.perturbation_memory_budget_bytes,
            minibatch_size: config.minibatch_size,
            n_grad_steps: config.n_grad_steps,
            lr: config.lr,
            epsilon: config.epsilon,
            clip_threshold: config.clip_threshold,
            error_feedback: None,
            error_decay: 0.9,
            error_gain: 1.0,
            step: 0,
            best_loss: f64::INFINITY,
        });
    }
    state.as_mut().unwrap()
}

fn apply_hyper_parameters(
    ts: &mut TrainingSubsystem,
    update: &HyperParameterUpdate,
) -> Result<(), Error> {
    if let Some(lr) = update.lr {
        ts.lr = lr;
    }
    if let Some(eps) = update.epsilon {
        ts.epsilon = eps;
    }
    if let Some(temp) = update.temperature {
        ts.loss_config.temperature = temp;
    }
    if let Some(lb) = update.lb_loss {
        ts.loss_config.lb_loss_weight = lb;
    }
    if let Some(z) = update.z_loss {
        ts.loss_config.z_loss_weight = z;
    }
    if let Some(clip) = update.clip_threshold {
        ts.clip_threshold = clip;
    }
    if let Some(ref ef) = update.error_feedback {
        match ef {
            ErrorFeedbackMode::None => {
                ts.error_feedback = Some(ef.clone());
            }
            ErrorFeedbackMode::Persistent(p) => {
                ts.error_feedback = Some(ef.clone());
                ts.error_decay = p.decay;
                ts.error_gain = p.gain;
            }
            ErrorFeedbackMode::Replay(r) => {
                ts.error_feedback = Some(ef.clone());
                ts.error_decay = r.decay;
                ts.error_gain = r.gain;
            }
        }
    }
    if let Some(decay) = update.mtp_decay {
        if let Some(ref mut mtp) = ts.mtp_config {
            mtp.decay_factor = decay;
        }
    }
    if let Some(n) = update.num_speculative {
        if let Some(ref mut mtp) = ts.mtp_config {
            mtp.num_depths = n as usize;
        }
    }
    if let Some(ref multipliers) = update.epsilon_multipliers {
        for &(component, value) in multipliers {
            match component {
                EpsilonComponent::Embedding => ts.epsilon_config.embedding = value,
                EpsilonComponent::Attention => ts.epsilon_config.attention = value,
                EpsilonComponent::MoeGating => ts.epsilon_config.moe_gating = value,
                EpsilonComponent::MoeExperts => ts.epsilon_config.moe_experts = value,
                EpsilonComponent::MoeExpertBanks => ts.epsilon_config.moe_expert_banks = value,
                EpsilonComponent::Mtp => ts.epsilon_config.mtp = value,
                EpsilonComponent::Norms => ts.epsilon_config.norms = value,
                EpsilonComponent::Ssm => ts.epsilon_config.ssm = value,
                EpsilonComponent::Other => ts.epsilon_config.other = value,
            }
        }
    }
    Ok(())
}

/// Set up the QuZO optimizer from the model's shared tensors.
fn setup_optimizer(
    model: &mut paramecia_model::ModelWeights,
    ts: &TrainingSubsystem,
) -> Result<Option<QuZO>, Error> {
    let quzo_tensors = model.quzo_qtensors();
    if quzo_tensors.is_empty() {
        return Err(Error::TrainError("No trainable tensors found".into()));
    }

    let verbose_tensors = std::env::var("PARAMECIA_LOG_TENSORS").is_ok();
    info!("Found {} trainable tensors", quzo_tensors.len());
    if verbose_tensors {
        for (name, qt) in &quzo_tensors {
            let qt_read = qt.read().unwrap();
            debug!(
                "[shared] {} : {:?} {:?}",
                name,
                qt_read.shape(),
                qt_read.dtype()
            );
        }
    }

    let supported_dtypes = [
        GgmlDType::F32,
        GgmlDType::BF16,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4_0,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8_0,
        GgmlDType::Q8K,
    ];

    let filtered: Vec<_> = quzo_tensors
        .iter()
        .filter(|(name, qt)| {
            if name.contains("expert_mask") || name.contains("expert_remap") {
                if verbose_tensors {
                    debug!("[skip] {} (pruning metadata)", name);
                }
                return false;
            }
            let qt_read = qt.read().unwrap();
            let dtype = qt_read.dtype();
            if !supported_dtypes.contains(&dtype) {
                if verbose_tensors {
                    debug!("[skip] {} ({:?} not supported)", name, dtype);
                }
                return false;
            }
            let included = match ts.optimize_tensors {
                OptimizeTensors::All => true,
                OptimizeTensors::AttentionOnly => {
                    name.contains("attn_q")
                        || name.contains("attn_k")
                        || name.contains("attn_v")
                        || name.contains("attn_output")
                }
                OptimizeTensors::AttentionQKOnly => {
                    name.contains("attn_q") || name.contains("attn_k")
                }
            };
            if !included && verbose_tensors {
                debug!("[skip] {} (not in optimize set)", name);
            }
            included
        })
        .collect();

    info!(
        "After filtering: {} tensors for optimizer (mode: {:?})",
        filtered.len(),
        ts.optimize_tensors
    );

    let epsilon_multipliers: Vec<f64> = filtered
        .iter()
        .map(|(name, _)| ts.epsilon_config.multiplier_for(name))
        .collect();

    let perturbation_bytes = filtered.iter().fold(0u64, |total, (_, qt)| {
        let elem_count = qt.read().unwrap().shape().elem_count() as u64;
        total.saturating_add(elem_count.saturating_mul(std::mem::size_of::<f32>() as u64))
    });
    let lazy_perturbations = should_use_lazy_perturbations(
        ts.lazy_perturbations,
        perturbation_bytes,
        ts.perturbation_memory_budget_bytes,
    );
    info!(
        "Perturbation storage: estimated {:.2} GiB, budget {:.2} GiB; using {} mode",
        bytes_to_gib(perturbation_bytes),
        bytes_to_gib(ts.perturbation_memory_budget_bytes),
        if lazy_perturbations { "lazy" } else { "eager" }
    );

    let optimizer_tensors: Vec<_> = filtered.into_iter().map(|(_, qt)| qt.clone()).collect();

    let params = ParamsQuZO {
        lr: ts.lr,
        epsilon: ts.epsilon,
        num_samples: 1,
        clip_threshold: ts.clip_threshold,
        // Scoped fused state limits on-the-fly perturbation to this optimizer set.
        use_fused: true,
        epsilon_multipliers: Some(epsilon_multipliers),
        lazy_perturbations,
        error_feedback: ts.error_feedback.as_ref().and_then(|ef| match ef {
            ErrorFeedbackMode::None => None,
            ErrorFeedbackMode::Persistent(_) => Some(paramecia_opt::ErrorFeedbackMode::Persistent),
            ErrorFeedbackMode::Replay(r) => Some(paramecia_opt::ErrorFeedbackMode::Replay {
                max_history: r.steps as usize,
            }),
        }),
        error_decay: ts.error_decay,
        error_gain: ts.error_gain,
    };

    match QuZO::new(optimizer_tensors, params) {
        Ok(opt) => Ok(Some(opt)),
        Err(e) => Err(Error::TrainError(format!(
            "Failed to create optimizer: {e}"
        ))),
    }
}

fn should_use_lazy_perturbations(forced: bool, estimated_bytes: u64, budget_bytes: u64) -> bool {
    forced || estimated_bytes > budget_bytes
}

fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Process a single training sample: convert, buffer, and run a step when batch is full.
async fn ingest_training_sample(
    executor: &mut ModelEngineInner,
    training_state: &mut Option<TrainingSubsystem>,
    at: &mut ActiveTraining,
    exec_sample: TrainingSample,
) {
    let tokenizer = executor.tokenizer().clone();
    let sample_id = exec_sample.id;

    match sample_to_tuning_data(&exec_sample.data, &tokenizer) {
        Ok(tuning_data) => {
            at.sample_buffer.push(tuning_data);
            at.id_buffer.push(sample_id);
        }
        Err(e) => {
            let _ = at
                .loss_tx
                .send(Err(Error::TrainError(format!(
                    "Failed to convert sample: {e}"
                ))))
                .await;
            return;
        }
    }

    let effective_batch = at.minibatch_size * at.n_grad_steps;
    if at.sample_buffer.len() >= effective_batch {
        run_training_step(executor, training_state, at, effective_batch).await;
    }
}

/// Flush any remaining buffered samples as a partial training step.
async fn flush_training_buffer(
    executor: &mut ModelEngineInner,
    training_state: &mut Option<TrainingSubsystem>,
    at: &mut ActiveTraining,
) {
    if !at.sample_buffer.is_empty() {
        let batch_size = at.sample_buffer.len();
        run_training_step(executor, training_state, at, batch_size).await;
    }
}

/// Run one training step on the buffered samples.
async fn run_training_step(
    executor: &mut ModelEngineInner,
    training_state: &mut Option<TrainingSubsystem>,
    at: &mut ActiveTraining,
    batch_size: usize,
) {
    let device = executor.device().clone();
    let model = executor.model_mut();
    let actual_n_grad = batch_size.div_ceil(at.minibatch_size);

    let step_ids: Vec<Uuid> = at
        .id_buffer
        .drain(..batch_size.min(at.id_buffer.len()))
        .collect();
    let n_tokens: u64 = at
        .sample_buffer
        .iter()
        .take(batch_size)
        .map(|s| s.token_ids.len() as u64)
        .sum();
    let step_loss = run_training_step_with_grad_accum(
        model,
        &mut at.optimizer,
        &at.loss_fn,
        &mut at.sample_buffer,
        &device,
        at.is_validation,
        at.minibatch_size,
        actual_n_grad,
        at.mtp_config.as_ref(),
    );

    match step_loss {
        Ok(loss) => {
            if let Some(ref mut ts) = training_state {
                ts.step += 1;
                if loss < ts.best_loss {
                    ts.best_loss = loss;
                }
            }
            let result = StepResult {
                loss,
                sample_ids: step_ids,
                n_tokens,
            };
            let _ = at.loss_tx.send(Ok(result)).await;
        }
        Err(e) => {
            let _ = at
                .loss_tx
                .send(Err(Error::TrainError(format!("Training step failed: {e}"))))
                .await;
        }
    }
}

/// State for decomposed ZO training (perturb_up → perturb_down → update cycle).
struct DecomposedZO {
    optimizer: QuZO,
    /// State from perturb_up, cleared after update.
    pending_state: Option<DecomposedZOState>,
}

#[derive(Clone, Copy)]
enum ModelLifecycleEvent {
    /// Clears inference state only; an in-progress optimization cycle survives.
    InferenceReset,
    /// Unloading the model aborts any in-progress optimization cycle.
    ModelUnload,
}

impl DecomposedZO {
    fn handle_lifecycle(&mut self, event: ModelLifecycleEvent) {
        match event {
            ModelLifecycleEvent::InferenceReset => {}
            ModelLifecycleEvent::ModelUnload => {
                self.optimizer.clear_decomposed_perturbation();
                self.pending_state = None;
            }
        }
    }
}

/// Process a single command from the actor's command channel.
#[allow(clippy::too_many_arguments)]
async fn process_command(
    cmd: ModelCommand,
    _rx: &mut mpsc::Receiver<ModelCommand>,
    executor: &mut ModelEngineInner,
    training_state: &mut Option<TrainingSubsystem>,
    active_training: &mut Option<ActiveTraining>,
    decomposed_zo: &mut Option<DecomposedZO>,
    training_config: &TrainingConfig,
    checkpoint_dir: &Path,
) {
    match cmd {
        // -- Inference commands --
        ModelCommand::FillContext { text, progress } => {
            let _ = executor.fill_context(&text, Some(progress)).await;
        }
        ModelCommand::FillContextTokens { tokens, progress } => {
            let _ = executor.fill_context_tokens(&tokens, Some(progress)).await;
        }
        ModelCommand::FillContextInputs { inputs, progress } => {
            let _ = executor.fill_context_inputs(&inputs, Some(progress)).await;
        }
        ModelCommand::PredictToken { respond } => {
            let result = executor.predict_token().await;
            let _ = respond.send(result);
        }
        ModelCommand::PredictCompletion { respond, limit, cancel_rx } => {
            executor.predict_completion(cancel_rx, limit, respond).await;
        }
        ModelCommand::PredictCompletionsBatched {
            inputs,
            respond,
            cancel_rx,
        } => {
            executor
                .predict_completions_batched(inputs, cancel_rx, respond)
                .await;
        }
        ModelCommand::CommitToken { token, respond } => {
            let result = executor.commit_token(token).await;
            let _ = respond.send(result);
        }
        ModelCommand::TakeSnapshot { respond } => {
            let result = executor.take_snapshot().await;
            let _ = respond.send(result);
        }
        ModelCommand::PersistSnapshot {
            snapshot_id,
            respond,
        } => {
            let result = executor.persist_snapshot(snapshot_id).await;
            let _ = respond.send(result);
        }
        ModelCommand::DropSnapshot {
            snapshot_id,
            respond,
        } => {
            let result = executor.drop_snapshot(snapshot_id);
            let _ = respond.send(result);
        }
        ModelCommand::RestoreSnapshot {
            snapshot_id,
            respond,
        } => {
            let result = executor.restore_snapshot(&snapshot_id);
            let _ = respond.send(result);
        }
        ModelCommand::LoadPersistedSnapshot { path, respond } => {
            let result = executor.load_persisted_snapshot(&path);
            let _ = respond.send(result);
        }
        ModelCommand::Reseed { seed, respond } => {
            let result = executor.reseed(seed).await;
            let _ = respond.send(result);
        }
        ModelCommand::ResetState { respond } => {
            if let Some(dzo) = decomposed_zo.as_mut() {
                dzo.handle_lifecycle(ModelLifecycleEvent::InferenceReset);
            }
            executor.reset_state();
            let _ = respond.send(Ok(()));
        }
        ModelCommand::PrepareForGeneration { respond } => {
            executor.prepare_for_generation();
            let _ = respond.send(Ok(()));
        }
        ModelCommand::GetTokens { respond } => {
            let _ = respond.send(executor.tokens().to_vec());
        }
        ModelCommand::GetStatePosition { respond } => {
            let _ = respond.send(executor.state_position());
        }

        // -- Training commands --
        ModelCommand::TrainModel {
            sample_rx,
            is_validation,
            loss_tx,
            cancel_rx,
        } => {
            let ts = ensure_training_subsystem(training_state, executor, training_config);

            // Set up optimizer (only if not validation)
            let optimizer = if !is_validation {
                match setup_optimizer(executor.model_mut(), ts) {
                    Ok(opt) => opt,
                    Err(e) => {
                        let _ = loss_tx.send(Err(e)).await;
                        return;
                    }
                }
            } else {
                None
            };

            *active_training = Some(ActiveTraining {
                sample_rx,
                loss_tx,
                cancel_rx,
                optimizer,
                loss_fn: DistillationLoss::new(ts.loss_config.clone()),
                mtp_config: ts.mtp_config.clone(),
                is_validation,
                sample_buffer: Vec::new(),
                id_buffer: Vec::new(),
                minibatch_size: ts.minibatch_size,
                n_grad_steps: ts.n_grad_steps,
            });
        }
        ModelCommand::SetHyperParameters { update, respond } => {
            let ts = ensure_training_subsystem(training_state, executor, training_config);
            let result = apply_hyper_parameters(ts, &update);
            let _ = respond.send(result);
        }
        ModelCommand::PerturbUp { seed, respond } => {
            // Ensure training subsystem + optimizer exist
            let ts = ensure_training_subsystem(training_state, executor, training_config);
            if decomposed_zo.is_none() {
                match setup_optimizer(executor.model_mut(), ts) {
                    Ok(Some(opt)) => {
                        *decomposed_zo = Some(DecomposedZO {
                            optimizer: opt,
                            pending_state: None,
                        });
                    }
                    Ok(None) => {
                        let _ = respond.send(Err(Error::TrainError("No optimizer created".into())));
                        return;
                    }
                    Err(e) => {
                        let _ = respond.send(Err(e));
                        return;
                    }
                }
            }

            let dzo = decomposed_zo.as_mut().unwrap();
            match dzo.optimizer.perturb_up(seed) {
                Ok(state) => {
                    dzo.pending_state = Some(state);
                    let _ = respond.send(Ok(()));
                }
                Err(e) => {
                    let _ = respond.send(Err(Error::TrainError(format!("perturb_up failed: {e}"))));
                }
            }
        }
        ModelCommand::PerturbDown { respond } => match decomposed_zo.as_mut() {
            Some(dzo) => match dzo.pending_state.as_ref() {
                Some(state) => match dzo.optimizer.perturb_down(state) {
                    Ok(()) => {
                        let _ = respond.send(Ok(()));
                    }
                    Err(e) => {
                        let _ = respond
                            .send(Err(Error::TrainError(format!("perturb_down failed: {e}"))));
                    }
                },
                None => {
                    let _ = respond.send(Err(Error::InvalidState(
                        "perturb_down called without prior perturb_up".into(),
                    )));
                }
            },
            None => {
                let _ = respond.send(Err(Error::InvalidState(
                    "perturb_down called without prior perturb_up".into(),
                )));
            }
        },
        ModelCommand::Update { loss_up, loss_down, respond } => match decomposed_zo.as_mut() {
            Some(dzo) => match dzo.pending_state.take() {
                Some(state) => {
                    match dzo.optimizer.update(state, loss_up, loss_down) {
                        Ok(_avg_loss) => {
                            let _ = respond.send(Ok(()));
                        }
                        Err(e) => {
                            let _ =
                                respond.send(Err(Error::TrainError(format!("update failed: {e}"))));
                        }
                    }
                }
                None => {
                    let _ = respond.send(Err(Error::InvalidState(
                        "update called without prior perturb_up/perturb_down".into(),
                    )));
                }
            },
            None => {
                let _ = respond.send(Err(Error::InvalidState(
                    "update called without prior perturb_up".into(),
                )));
            }
        },

        // -- Lifecycle --
        ModelCommand::SaveCheckpoint { respond } => {
            let result = do_save_checkpoint(executor, checkpoint_dir);
            let _ = respond.send(result);
        }
        ModelCommand::UnloadModel { respond } => {
            if let Some(dzo) = decomposed_zo.as_mut() {
                dzo.handle_lifecycle(ModelLifecycleEvent::ModelUnload);
            }
            executor.unload_model();
            let _ = respond.send(Ok(()));
        }
    }
}

fn do_save_checkpoint(
    executor: &ModelEngineInner,
    checkpoint_dir: &Path,
) -> Result<PathBuf, Error> {
    let model = executor.model();
    let model_path = executor.model_path();

    let filename = format!("checkpoint-{}.gguf", uuid::Uuid::new_v4());
    let output_path = checkpoint_dir.join(&filename);

    let extra = if executor.pending_metadata().is_empty() {
        None
    } else {
        let map: HashMap<String, gguf_file::Value> = executor
            .pending_metadata()
            .iter()
            .map(|(k, v)| (k.clone(), metadata_value_to_gguf(v)))
            .collect();
        Some(map)
    };

    save_trained_model(model, model_path, &output_path, extra.as_ref())
        .map_err(|e| Error::CheckpointError(format!("Failed to save checkpoint: {e}")))?;

    Ok(output_path)
}

fn metadata_value_to_gguf(v: &MetadataValue) -> gguf_file::Value {
    match v {
        MetadataValue::U8(x) => gguf_file::Value::U8(*x),
        MetadataValue::I8(x) => gguf_file::Value::I8(*x),
        MetadataValue::U16(x) => gguf_file::Value::U16(*x),
        MetadataValue::I16(x) => gguf_file::Value::I16(*x),
        MetadataValue::U32(x) => gguf_file::Value::U32(*x),
        MetadataValue::I32(x) => gguf_file::Value::I32(*x),
        MetadataValue::U64(x) => gguf_file::Value::U64(*x),
        MetadataValue::I64(x) => gguf_file::Value::I64(*x),
        MetadataValue::F32(x) => gguf_file::Value::F32(*x),
        MetadataValue::F64(x) => gguf_file::Value::F64(*x),
        MetadataValue::Bool(x) => gguf_file::Value::Bool(*x),
        MetadataValue::String(x) => gguf_file::Value::String(x.clone()),
        MetadataValue::Array(arr) => {
            gguf_file::Value::Array(arr.iter().map(metadata_value_to_gguf).collect())
        }
    }
}

/// Handle to the dedicated model actor thread.
///
/// Dropping this handle closes the command channel and joins the actor thread.
pub(crate) struct ModelActorHandle {
    tx: Option<mpsc::Sender<ModelCommand>>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl ModelActorHandle {
    pub(crate) fn sender(&self) -> &mpsc::Sender<ModelCommand> {
        self.tx
            .as_ref()
            .expect("model actor sender missing while handle is alive")
    }
}

impl Drop for ModelActorHandle {
    fn drop(&mut self) {
        let _ = self.tx.take();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

/// Spawn the unified model actor on a dedicated OS thread.
///
/// Returns the command channel handle. The actor runs until the channel is dropped.
pub(crate) fn spawn_model_actor(
    executor: ModelEngineInner,
    training_config: TrainingConfig,
) -> ModelActorHandle {
    let (tx, rx) = mpsc::channel(32);
    let join = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create model actor runtime");
        rt.block_on(run_model_actor(rx, executor, training_config));
    });
    ModelActorHandle {
        tx: Some(tx),
        join: Some(join),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        should_use_lazy_perturbations, DecomposedZO, ModelLifecycleEvent, ParamsQuZO, QuZO,
    };

    #[test]
    fn perturbation_memory_policy_honors_force_and_budget() {
        assert!(should_use_lazy_perturbations(true, 1, u64::MAX));
        assert!(!should_use_lazy_perturbations(false, 1024, 1024));
        assert!(should_use_lazy_perturbations(false, 1025, 1024));
        assert!(should_use_lazy_perturbations(false, 1, 0));
    }

    #[test]
    fn inference_resets_preserve_decomposed_zo_cycle_until_update() {
        // An empty tensor list is sufficient to exercise the actor's cycle state
        // without constructing a full model. Inference itself does not own or
        // mutate this state; only its lifecycle event matters here.
        let optimizer = QuZO::new_with_seed(
            Vec::new(),
            ParamsQuZO {
                use_fused: false,
                ..ParamsQuZO::default()
            },
            42,
        )
        .expect("create optimizer");
        let mut decomposed_zo = DecomposedZO {
            optimizer,
            pending_state: None,
        };

        decomposed_zo.pending_state = Some(
            decomposed_zo
                .optimizer
                .perturb_up(Some(7))
                .expect("perturb up"),
        );

        // perturb_up -> inference -> reset_state -> perturb_down
        decomposed_zo.handle_lifecycle(ModelLifecycleEvent::InferenceReset);
        let state = decomposed_zo
            .pending_state
            .as_ref()
            .expect("first inference reset must preserve the cycle");
        decomposed_zo
            .optimizer
            .perturb_down(state)
            .expect("perturb down");

        // perturb_down -> inference -> reset_state -> update
        decomposed_zo.handle_lifecycle(ModelLifecycleEvent::InferenceReset);
        let state = decomposed_zo
            .pending_state
            .take()
            .expect("second inference reset must preserve the cycle");
        decomposed_zo
            .optimizer
            .update(state, 1.0, 0.5)
            .expect("update");

        assert!(
            decomposed_zo.pending_state.is_none(),
            "update must consume the pending cycle"
        );
    }
}
