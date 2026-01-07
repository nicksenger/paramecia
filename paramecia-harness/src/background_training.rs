//! Background training manager for concurrent training and interaction.
//!
//! This module enables running distributed training in the background while
//! the user interacts with the agent in the foreground. Users can:
//! - Switch between agent modes (Plan, AcceptEdits, AutoApprove, Default)
//! - Test the model during training
//! - Pause/resume training
//! - Save/load checkpoints

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::RwLock;
use tokio::sync::mpsc;

/// Status of background training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundTrainingStatus {
    /// Not started.
    Idle,
    /// Connecting to training network.
    Connecting,
    /// Training is active.
    Running,
    /// Training is paused.
    Paused,
    /// Training completed successfully.
    Completed,
    /// Training failed.
    Failed,
}

/// Events from background training.
#[derive(Debug, Clone)]
pub enum BackgroundTrainingEvent {
    /// Status changed.
    StatusChanged { status: BackgroundTrainingStatus },
    /// Progress update.
    Progress {
        generation: u32,
        completed: u32,
        total: u32,
        mean_fitness: f32,
    },
    /// Generation completed.
    GenerationCompleted {
        generation: u32,
        mean_fitness: f32,
        best_fitness: f32,
    },
    /// Checkpoint saved.
    CheckpointSaved { path: PathBuf },
    /// Training completed.
    TrainingCompleted { final_fitness: f32 },
    /// Error occurred.
    Error { message: String },
}

/// Commands to send to background training.
#[derive(Debug, Clone)]
pub enum BackgroundTrainingCommand {
    /// Pause training.
    Pause,
    /// Resume training.
    Resume,
    /// Save a checkpoint.
    SaveCheckpoint,
    /// Stop training.
    Stop,
}

/// Background training state shared between threads.
pub struct BackgroundTrainingState {
    /// Current status.
    pub status: BackgroundTrainingStatus,
    /// Current generation.
    pub current_generation: u32,
    /// Total generations.
    pub total_generations: u32,
    /// Current mean fitness.
    pub mean_fitness: f32,
    /// Best fitness so far.
    pub best_fitness: f32,
    /// Number of connected workers (host only).
    pub worker_count: usize,
    /// Last checkpoint path.
    pub last_checkpoint: Option<PathBuf>,
}

impl Default for BackgroundTrainingState {
    fn default() -> Self {
        Self {
            status: BackgroundTrainingStatus::Idle,
            current_generation: 0,
            total_generations: 0,
            mean_fitness: 0.0,
            best_fitness: 0.0,
            worker_count: 0,
            last_checkpoint: None,
        }
    }
}

/// Manager for background training.
pub struct BackgroundTrainingManager {
    /// Shared state.
    state: Arc<RwLock<BackgroundTrainingState>>,
    /// Command sender.
    command_tx: Option<mpsc::UnboundedSender<BackgroundTrainingCommand>>,
    /// Event receiver.
    event_rx: Option<mpsc::UnboundedReceiver<BackgroundTrainingEvent>>,
    /// Handle to background task.
    task_handle: Option<tokio::task::JoinHandle<()>>,
}

impl BackgroundTrainingManager {
    /// Create a new background training manager.
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(BackgroundTrainingState::default())),
            command_tx: None,
            event_rx: None,
            task_handle: None,
        }
    }

    /// Get the current status.
    pub fn status(&self) -> BackgroundTrainingStatus {
        self.state.read().status
    }

    /// Get the current state.
    pub fn state(&self) -> BackgroundTrainingState {
        self.state.read().clone()
    }

    /// Check if training is running.
    pub fn is_running(&self) -> bool {
        matches!(
            self.status(),
            BackgroundTrainingStatus::Running | BackgroundTrainingStatus::Paused
        )
    }

    /// Pause training.
    pub fn pause(&self) {
        if let Some(tx) = &self.command_tx {
            let _ = tx.send(BackgroundTrainingCommand::Pause);
        }
    }

    /// Resume training.
    pub fn resume(&self) {
        if let Some(tx) = &self.command_tx {
            let _ = tx.send(BackgroundTrainingCommand::Resume);
        }
    }

    /// Save a checkpoint.
    pub fn save_checkpoint(&self) {
        if let Some(tx) = &self.command_tx {
            let _ = tx.send(BackgroundTrainingCommand::SaveCheckpoint);
        }
    }

    /// Stop training.
    pub fn stop(&mut self) {
        if let Some(tx) = &self.command_tx {
            let _ = tx.send(BackgroundTrainingCommand::Stop);
        }
        if let Some(handle) = self.task_handle.take() {
            handle.abort();
        }
        self.state.write().status = BackgroundTrainingStatus::Idle;
    }

    /// Poll for events (non-blocking).
    pub fn poll_event(&mut self) -> Option<BackgroundTrainingEvent> {
        self.event_rx.as_mut().and_then(|rx| rx.try_recv().ok())
    }

    /// Process pending events and update state.
    pub fn process_events(&mut self) {
        while let Some(event) = self.poll_event() {
            self.handle_event(event);
        }
    }

    fn handle_event(&self, event: BackgroundTrainingEvent) {
        let mut state = self.state.write();

        match &event {
            BackgroundTrainingEvent::StatusChanged { status } => {
                state.status = *status;
            }
            BackgroundTrainingEvent::Progress {
                generation,
                completed: _,
                total: _,
                mean_fitness,
            } => {
                state.current_generation = *generation;
                state.mean_fitness = *mean_fitness;
            }
            BackgroundTrainingEvent::GenerationCompleted {
                generation,
                mean_fitness,
                best_fitness,
            } => {
                state.current_generation = *generation;
                state.mean_fitness = *mean_fitness;
                if *best_fitness > state.best_fitness {
                    state.best_fitness = *best_fitness;
                }
            }
            BackgroundTrainingEvent::CheckpointSaved { path } => {
                state.last_checkpoint = Some(path.clone());
            }
            BackgroundTrainingEvent::TrainingCompleted { final_fitness } => {
                state.status = BackgroundTrainingStatus::Completed;
                state.mean_fitness = *final_fitness;
            }
            BackgroundTrainingEvent::Error { .. } => {
                state.status = BackgroundTrainingStatus::Failed;
            }
        }
    }

    /// Get a clone of the shared state for display.
    pub fn get_state_clone(&self) -> BackgroundTrainingState {
        self.state.read().clone()
    }

    /// Format status for display.
    pub fn format_status(&self) -> String {
        let state = self.state.read();
        match state.status {
            BackgroundTrainingStatus::Idle => "Training: Idle".to_string(),
            BackgroundTrainingStatus::Connecting => "Training: Connecting...".to_string(),
            BackgroundTrainingStatus::Running => {
                format!(
                    "Training: Gen {}/{} | Fitness: {:.3} | Workers: {}",
                    state.current_generation,
                    state.total_generations,
                    state.mean_fitness,
                    state.worker_count
                )
            }
            BackgroundTrainingStatus::Paused => {
                format!(
                    "Training: PAUSED (Gen {}/{} | Fitness: {:.3})",
                    state.current_generation, state.total_generations, state.mean_fitness
                )
            }
            BackgroundTrainingStatus::Completed => {
                format!(
                    "Training: Complete (Final fitness: {:.3})",
                    state.mean_fitness
                )
            }
            BackgroundTrainingStatus::Failed => "Training: FAILED".to_string(),
        }
    }
}

impl Default for BackgroundTrainingManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for BackgroundTrainingState {
    fn clone(&self) -> Self {
        Self {
            status: self.status,
            current_generation: self.current_generation,
            total_generations: self.total_generations,
            mean_fitness: self.mean_fitness,
            best_fitness: self.best_fitness,
            worker_count: self.worker_count,
            last_checkpoint: self.last_checkpoint.clone(),
        }
    }
}
