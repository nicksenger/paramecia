//! Prefetch-Based Pipelining for CPU/GPU Overlap
//!
//! The prefetch pipeline hides transfer latency by:
//! 1. Pre-transferring hidden states to CPU while GPU computes norm/router
//! 2. Starting MoE immediately when transfer completes (no wait)
//! 3. Overlapping CPU->GPU result transfer with next attention
//!
//! ```text
//! Time →
//! GPU:  [Attn N]──[Norm+Router N]────────────────[Attn N+1]──[Norm+Router N+1]──...
//!                      │                              │
//!                      └─ Transfer to CPU ─┐          └─ Transfer to CPU ─┐
//!                                          v                              v
//! CPU:                          [MoE N]─────────────────────[MoE N+1]─────...
//!                                   │                            │
//!                                   └─ Transfer to GPU ──────────┘
//! ```
//!
//! This implementation uses a producer-consumer pattern with double buffering.

use candle::{DType, Device, Result, Tensor};
use crossbeam::channel::{self, Receiver, Sender};
use half::f16;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Data storage that can hold either F16 or F32 for optimized transfers
enum HiddenData {
    F16(Vec<f16>),
    F32(Vec<f32>),
}

impl HiddenData {
    /// Convert to f32 slice for processing
    fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            HiddenData::F16(data) => data.iter().map(|x| x.to_f32()).collect(),
            HiddenData::F32(data) => data.clone(),
        }
    }

    /// Get length
    #[allow(dead_code)]
    fn len(&self) -> usize {
        match self {
            HiddenData::F16(data) => data.len(),
            HiddenData::F32(data) => data.len(),
        }
    }
}

struct PrefetchedData {
    layer_idx: usize,
    /// Hidden states on CPU (F16 or F32 depending on input)
    hidden_cpu: HiddenData,
    /// Expert indices on CPU
    indices_cpu: Vec<u32>,
    /// Expert weights on CPU
    weights_cpu: Vec<f32>,
    /// FFN residual on CPU (F16 or F32 depending on input)
    residual_cpu: HiddenData,
    /// Original shapes for reconstruction
    hidden_shape: (usize, usize),
    residual_shape: (usize, usize, usize),
    indices_shape: (usize, usize),
    /// Whether input was F16 (for result conversion)
    use_f16: bool,
}

/// MoE result with output data
struct MoeOutputData {
    layer_idx: usize,
    /// Output data on CPU (F16 or F32 depending on input)
    output_data: HiddenData,
    /// Shape for reconstruction
    shape: (usize, usize, usize),
}

/// Prefetch-based pipeline coordinator
///
/// This pipeline overlaps CPU MoE computation with GPU work:
/// 1. GPU thread transfers data to CPU (must happen on GPU thread for CUDA context)
/// 2. MoE worker picks up the data and computes on CPU
/// 3. Result is transferred back to GPU
///
/// The key optimization is that MoE computation happens on a separate thread
/// while GPU can continue with other work.
pub struct PrefetchPipelineCoordinator {
    /// Data sender (GPU thread -> MoE worker)
    data_tx: Sender<Option<PrefetchedData>>,
    /// MoE result receiver
    result_rx: Receiver<MoeOutputData>,
    /// MoE worker thread handle
    moe_handle: Option<JoinHandle<()>>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// Pending request count
    pending_count: Arc<AtomicUsize>,
    /// GPU device
    gpu_device: Device,
    /// Number of layers
    num_layers: usize,
}

impl PrefetchPipelineCoordinator {
    /// Create a new prefetch pipeline coordinator
    pub fn new(
        gpu_device: Device,
        num_layers: usize,
        gate_weights: Vec<Arc<candle::quantized::QTensor>>,
        up_weights: Vec<Arc<candle::quantized::QTensor>>,
        down_weights: Vec<Arc<candle::quantized::QTensor>>,
        num_experts: usize,
    ) -> Self {
        // Channel from GPU thread to MoE worker (larger buffer to prevent blocking)
        let (data_tx, data_rx) = channel::bounded::<Option<PrefetchedData>>(8);
        // Channel from MoE worker back to GPU thread
        let (result_tx, result_rx) = channel::bounded::<MoeOutputData>(8);

        let shutdown = Arc::new(AtomicBool::new(false));
        let pending_count = Arc::new(AtomicUsize::new(0));

        let shutdown_moe = Arc::clone(&shutdown);

        // MoE worker thread: processes MoE on CPU data
        let moe_handle = thread::spawn(move || {
            Self::moe_worker(
                data_rx,
                result_tx,
                shutdown_moe,
                gate_weights,
                up_weights,
                down_weights,
                num_experts,
            );
        });

        Self {
            data_tx,
            result_rx,
            moe_handle: Some(moe_handle),
            shutdown,
            pending_count,
            gpu_device,
            num_layers,
        }
    }

    /// MoE worker - processes MoE on prefetched CPU data
    fn moe_worker(
        rx: Receiver<Option<PrefetchedData>>,
        tx: Sender<MoeOutputData>,
        shutdown: Arc<AtomicBool>,
        gate_weights: Vec<Arc<candle::quantized::QTensor>>,
        up_weights: Vec<Arc<candle::quantized::QTensor>>,
        down_weights: Vec<Arc<candle::quantized::QTensor>>,
        num_experts: usize,
    ) {
        use rayon::prelude::*;

        while !shutdown.load(Ordering::Relaxed) {
            match rx.recv() {
                Ok(Some(data)) => {
                    let (n_tokens, hidden_dim) = data.hidden_shape;
                    let (batch_size, seq_len, _) = data.residual_shape;
                    let (_, top_k) = data.indices_shape;

                    // Convert hidden data to f32 for processing (computation needs f32)
                    let hidden_f32 = data.hidden_cpu.to_f32_vec();
                    let residual_f32 = data.residual_cpu.to_f32_vec();

                    // Group tokens by expert for efficient processing
                    let mut expert_groups: std::collections::HashMap<usize, Vec<(usize, usize)>> =
                        std::collections::HashMap::new();

                    for token_idx in 0..n_tokens {
                        for k in 0..top_k {
                            let expert_id = data.indices_cpu[token_idx * top_k + k] as usize;
                            expert_groups
                                .entry(expert_id)
                                .or_default()
                                .push((token_idx, k));
                        }
                    }

                    // Process experts in parallel using tensor operations
                    type ExpertResult = Vec<(usize, f32, Vec<f32>)>;
                    let expert_results: Vec<(usize, ExpertResult)> = expert_groups
                        .into_par_iter()
                        .filter_map(|(expert_id, assignments)| {
                            if expert_id >= num_experts {
                                return None;
                            }

                            let gate = gate_weights[data.layer_idx]
                                .slice_first_dim(expert_id)
                                .ok()?;
                            let up = up_weights[data.layer_idx].slice_first_dim(expert_id).ok()?;
                            let down = down_weights[data.layer_idx]
                                .slice_first_dim(expert_id)
                                .ok()?;

                            // Gather inputs for this expert
                            let segment_size = assignments.len();
                            let mut expert_input = vec![0.0f32; segment_size * hidden_dim];
                            let mut token_info: Vec<(usize, f32)> =
                                Vec::with_capacity(segment_size);

                            for (i, (token_idx, k)) in assignments.iter().enumerate() {
                                let src_start = token_idx * hidden_dim;
                                let dst_start = i * hidden_dim;
                                expert_input[dst_start..dst_start + hidden_dim].copy_from_slice(
                                    &hidden_f32[src_start..src_start + hidden_dim],
                                );

                                let weight = data.weights_cpu[token_idx * top_k + k];
                                token_info.push((*token_idx, weight));
                            }

                            // Create input tensor
                            let input = Tensor::from_vec(
                                expert_input,
                                (segment_size, hidden_dim),
                                &Device::Cpu,
                            )
                            .ok()?;

                            // Compute gate and up projections using quantized matmul
                            let gate_out = input.apply_op1_no_bwd(&gate).ok()?;
                            let up_out = input.apply_op1_no_bwd(&up).ok()?;

                            // SwiGLU activation: silu(gate) * up
                            let activated =
                                candle_nn::ops::silu(&gate_out).ok()?.mul(&up_out).ok()?;

                            // Down projection
                            let output = activated.apply_op1_no_bwd(&down).ok()?;

                            // Extract output data
                            let output_vec: Vec<f32> = output.flatten_all().ok()?.to_vec1().ok()?;

                            // Prepare results
                            let results: Vec<(usize, f32, Vec<f32>)> = token_info
                                .into_iter()
                                .enumerate()
                                .map(|(i, (token_id, weight))| {
                                    let output_row =
                                        output_vec[i * hidden_dim..(i + 1) * hidden_dim].to_vec();
                                    (token_id, weight, output_row)
                                })
                                .collect();

                            Some((expert_id, results))
                        })
                        .collect();

                    // Aggregate results and add residual (in f32)
                    let mut output_f32 = residual_f32;

                    for (_expert_id, results) in expert_results {
                        for (token_id, weight, output_row) in results {
                            for j in 0..hidden_dim {
                                output_f32[token_id * hidden_dim + j] += output_row[j] * weight;
                            }
                        }
                    }

                    // Convert output to F16 if input was F16 (halves return bandwidth)
                    let output_data = if data.use_f16 {
                        HiddenData::F16(output_f32.iter().map(|&x| f16::from_f32(x)).collect())
                    } else {
                        HiddenData::F32(output_f32)
                    };

                    let _ = tx.send(MoeOutputData {
                        layer_idx: data.layer_idx,
                        output_data,
                        shape: (batch_size, seq_len, hidden_dim),
                    });
                }
                Ok(None) | Err(_) => break,
            }
        }
    }

    /// Submit a prefetch request (non-blocking)
    ///
    /// Call this as soon as you have the attention output - the prefetch
    /// will happen in parallel with norm/router computation on GPU.
    /// Submit data for async MoE processing
    ///
    /// This does the GPU->CPU transfer on the calling thread (required for CUDA context),
    /// then sends the CPU data to the MoE worker for processing.
    pub fn submit_prefetch(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        expert_indices: &Tensor,
        expert_weights: &Tensor,
        ffn_residual: &Tensor,
    ) -> Result<()> {
        // Check if input is F16 - if so, transfer as F16 for half the bandwidth
        let use_f16 = hidden_states.dtype() == DType::F16;

        // Transfer data to CPU on this thread (has CUDA context)
        // Use F16 transfer when input is F16 to halve bandwidth
        let hidden_cpu = if use_f16 {
            HiddenData::F16(hidden_states.flatten_all()?.to_vec1::<f16>()?)
        } else {
            HiddenData::F32(
                hidden_states
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            )
        };

        let indices_cpu = expert_indices.flatten_all()?.to_vec1::<u32>()?;

        // Weights are always small, keep as F32
        let weights_cpu = expert_weights
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // Use F16 for residual too
        let residual_cpu = if use_f16 {
            HiddenData::F16(ffn_residual.flatten_all()?.to_vec1::<f16>()?)
        } else {
            HiddenData::F32(
                ffn_residual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            )
        };

        let hidden_shape = hidden_states.dims2()?;
        let residual_shape = ffn_residual.dims3()?;
        let indices_shape = expert_indices.dims2()?;

        let data = PrefetchedData {
            layer_idx,
            hidden_cpu,
            indices_cpu,
            weights_cpu,
            residual_cpu,
            hidden_shape,
            residual_shape,
            indices_shape,
            use_f16,
        };

        self.pending_count.fetch_add(1, Ordering::SeqCst);

        self.data_tx
            .send(Some(data))
            .map_err(|e| candle::Error::Msg(format!("Failed to submit data: {}", e)))
    }

    /// Wait for MoE result for a specific layer
    pub fn wait_for_result(&self, expected_layer: usize) -> Result<Tensor> {
        let result = self
            .result_rx
            .recv()
            .map_err(|e| candle::Error::Msg(format!("Failed to receive result: {}", e)))?;

        self.pending_count.fetch_sub(1, Ordering::SeqCst);

        if result.layer_idx != expected_layer {
            candle::bail!(
                "Layer mismatch: expected {}, got {}",
                expected_layer,
                result.layer_idx
            );
        }

        let (batch_size, seq_len, hidden_dim) = result.shape;

        // Create tensor from HiddenData (F16 or F32)
        let output = match result.output_data {
            HiddenData::F16(data) => {
                Tensor::from_vec(data, (batch_size * seq_len, hidden_dim), &Device::Cpu)?
                    .reshape((batch_size, seq_len, hidden_dim))?
                    .to_device(&self.gpu_device)?
            }
            HiddenData::F32(data) => {
                Tensor::from_vec(data, (batch_size * seq_len, hidden_dim), &Device::Cpu)?
                    .reshape((batch_size, seq_len, hidden_dim))?
                    .to_device(&self.gpu_device)?
            }
        };

        Ok(output)
    }

    /// Check if there are pending results
    pub fn has_pending(&self) -> bool {
        self.pending_count.load(Ordering::SeqCst) > 0
    }

    /// Get GPU device
    pub fn gpu_device(&self) -> &Device {
        &self.gpu_device
    }

    /// Shutdown the pipeline
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.data_tx.send(None);
        if let Some(handle) = self.moe_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for PrefetchPipelineCoordinator {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl std::fmt::Debug for PrefetchPipelineCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefetchPipelineCoordinator")
            .field("num_layers", &self.num_layers)
            .field("pending_count", &self.pending_count.load(Ordering::Relaxed))
            .field("shutdown", &self.shutdown.load(Ordering::Relaxed))
            .finish()
    }
}
