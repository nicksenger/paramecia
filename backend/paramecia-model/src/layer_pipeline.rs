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
//! On Vulkan, GPU→CPU transfers are submitted asynchronously so the GPU can
//! continue recording the next layer while the MoE worker waits for the fence.

use crossbeam::channel::{self, Receiver, Sender};
use half::f16;
use paramecia_core::{DType, Device, Result, Tensor};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
#[cfg(feature = "vulkan")]
use tracing::warn;

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

/// Shapes and metadata needed for MoE processing
struct PrefetchMeta {
    layer_idx: usize,
    hidden_shape: (usize, usize),
    output_shape: (usize, usize, usize),
    indices_shape: (usize, usize),
    use_f16: bool,
}

/// CPU-resident data ready for MoE processing (non-Vulkan path, or after fence wait)
struct PrefetchedData {
    meta: PrefetchMeta,
    hidden_cpu: HiddenData,
    indices_cpu: Vec<u32>,
    weights_cpu: Vec<f32>,
}

/// Vulkan async path: in-flight batch + pending downloads, resolved by worker thread
#[cfg(feature = "vulkan")]
struct AsyncPrefetchData {
    meta: PrefetchMeta,
    in_flight: paramecia_core::vulkan_backend::InFlightBatch,
    h_dl: paramecia_core::vulkan_backend::PendingDownload,
    i_dl: paramecia_core::vulkan_backend::PendingDownload,
    w_dl: paramecia_core::vulkan_backend::PendingDownload,
    h_count: usize,
    i_count: usize,
    w_count: usize,
}

/// Message sent from the GPU thread to the MoE worker
enum WorkerMessage {
    /// Data already on CPU (CUDA/Metal path, or non-GPU)
    Ready(PrefetchedData),
    /// Vulkan async: worker must wait on fence then read staging buffers
    #[cfg(feature = "vulkan")]
    Async(AsyncPrefetchData),
    /// Shutdown signal
    Shutdown,
}

// Safety: WorkerMessage contains InFlightBatch (marked Send) and PendingDownload.
// PendingDownload holds a VulkanBuffer whose Drop uses Arc<Mutex<Allocator>> — thread-safe.
// complete_download only reads from a mapped staging buffer after fence wait — no data races.
unsafe impl Send for WorkerMessage {}

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
/// while GPU can continue with other work. On Vulkan, the GPU→CPU transfer is
/// submitted asynchronously so the main thread can immediately start recording
/// the next layer while the worker waits for the transfer fence.
pub struct PrefetchPipelineCoordinator {
    /// Message sender (GPU thread -> MoE worker)
    msg_tx: Sender<WorkerMessage>,
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
        gate_weights: Vec<paramecia_core::quantized::SharedQTensor>,
        up_weights: Vec<paramecia_core::quantized::SharedQTensor>,
        down_weights: Vec<paramecia_core::quantized::SharedQTensor>,
        num_experts: usize,
        #[cfg(feature = "vulkan")] vk_device: Option<paramecia_core::VulkanDevice>,
    ) -> Self {
        // Channel from GPU thread to MoE worker (larger buffer to prevent blocking)
        let (msg_tx, msg_rx) = channel::bounded::<WorkerMessage>(8);
        // Channel from MoE worker back to GPU thread
        let (result_tx, result_rx) = channel::bounded::<MoeOutputData>(8);

        let shutdown = Arc::new(AtomicBool::new(false));
        let pending_count = Arc::new(AtomicUsize::new(0));

        let shutdown_moe = Arc::clone(&shutdown);

        // MoE worker thread: processes MoE on CPU data
        let moe_handle = thread::spawn(move || {
            Self::moe_worker(
                msg_rx,
                result_tx,
                shutdown_moe,
                gate_weights,
                up_weights,
                down_weights,
                num_experts,
                #[cfg(feature = "vulkan")]
                vk_device,
            );
        });

        Self {
            msg_tx,
            result_rx,
            moe_handle: Some(moe_handle),
            shutdown,
            pending_count,
            gpu_device,
            num_layers,
        }
    }

    /// Resolve a WorkerMessage into PrefetchedData (waits on fence for async path)
    #[cfg(feature = "vulkan")]
    fn resolve_message(
        msg: WorkerMessage,
        vk_device: &Option<paramecia_core::VulkanDevice>,
    ) -> Option<PrefetchedData> {
        match msg {
            WorkerMessage::Ready(data) => Some(data),
            WorkerMessage::Async(async_data) => {
                let vk = vk_device
                    .as_ref()
                    .expect("Vulkan device required for async prefetch");
                // Wait for the GPU transfer to complete
                if let Err(e) = async_data.in_flight.wait() {
                    warn!(error = %e, "Prefetch fence wait failed");
                    return None;
                }
                // Read from staging buffers (no GPU work, just memcpy from mapped memory)
                let hidden_cpu = if async_data.meta.use_f16 {
                    match vk.complete_download::<f16>(async_data.h_dl, async_data.h_count) {
                        Ok(v) => HiddenData::F16(v),
                        Err(e) => {
                            warn!(error = %e, "Prefetch hidden download failed");
                            return None;
                        }
                    }
                } else {
                    match vk.complete_download::<f32>(async_data.h_dl, async_data.h_count) {
                        Ok(v) => HiddenData::F32(v),
                        Err(e) => {
                            warn!(error = %e, "Prefetch hidden download failed");
                            return None;
                        }
                    }
                };
                let indices_cpu: Vec<u32> =
                    match vk.complete_download(async_data.i_dl, async_data.i_count) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!(error = %e, "Prefetch indices download failed");
                            return None;
                        }
                    };
                let weights_cpu: Vec<f32> =
                    match vk.complete_download(async_data.w_dl, async_data.w_count) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!(error = %e, "Prefetch weights download failed");
                            return None;
                        }
                    };
                Some(PrefetchedData {
                    meta: async_data.meta,
                    hidden_cpu,
                    indices_cpu,
                    weights_cpu,
                })
            }
            WorkerMessage::Shutdown => None,
        }
    }

    #[cfg(not(feature = "vulkan"))]
    fn resolve_message(msg: WorkerMessage) -> Option<PrefetchedData> {
        match msg {
            WorkerMessage::Ready(data) => Some(data),
            WorkerMessage::Shutdown => None,
        }
    }

    /// MoE worker - processes MoE on prefetched CPU data
    fn moe_worker(
        rx: Receiver<WorkerMessage>,
        tx: Sender<MoeOutputData>,
        shutdown: Arc<AtomicBool>,
        gate_weights: Vec<paramecia_core::quantized::SharedQTensor>,
        up_weights: Vec<paramecia_core::quantized::SharedQTensor>,
        down_weights: Vec<paramecia_core::quantized::SharedQTensor>,
        num_experts: usize,
        #[cfg(feature = "vulkan")] vk_device: Option<paramecia_core::VulkanDevice>,
    ) {
        use rayon::prelude::*;

        while !shutdown.load(Ordering::Relaxed) {
            let msg = match rx.recv() {
                Ok(msg) => msg,
                Err(_) => break,
            };

            #[cfg(feature = "vulkan")]
            let data = match Self::resolve_message(msg, &vk_device) {
                Some(d) => d,
                None => break,
            };
            #[cfg(not(feature = "vulkan"))]
            let data = match Self::resolve_message(msg) {
                Some(d) => d,
                None => break,
            };

            let (n_tokens, hidden_dim) = data.meta.hidden_shape;
            let (batch_size, seq_len, _) = data.meta.output_shape;
            let (_, top_k) = data.meta.indices_shape;

            // Convert hidden data to f32 for processing (computation needs f32)
            let hidden_f32 = data.hidden_cpu.to_f32_vec();

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

                    let gate = gate_weights[data.meta.layer_idx]
                        .read()
                        .unwrap()
                        .slice_first_dim(expert_id)
                        .ok()?;
                    let up = up_weights[data.meta.layer_idx]
                        .read()
                        .unwrap()
                        .slice_first_dim(expert_id)
                        .ok()?;
                    let down = down_weights[data.meta.layer_idx]
                        .read()
                        .unwrap()
                        .slice_first_dim(expert_id)
                        .ok()?;

                    // Gather inputs for this expert
                    let segment_size = assignments.len();
                    let mut expert_input = vec![0.0f32; segment_size * hidden_dim];
                    let mut token_info: Vec<(usize, f32)> = Vec::with_capacity(segment_size);

                    for (i, (token_idx, k)) in assignments.iter().enumerate() {
                        let src_start = token_idx * hidden_dim;
                        let dst_start = i * hidden_dim;
                        expert_input[dst_start..dst_start + hidden_dim]
                            .copy_from_slice(&hidden_f32[src_start..src_start + hidden_dim]);

                        let weight = data.weights_cpu[token_idx * top_k + k];
                        token_info.push((*token_idx, weight));
                    }

                    // Create input tensor
                    let input =
                        Tensor::from_vec(expert_input, (segment_size, hidden_dim), &Device::Cpu)
                            .ok()?;

                    // Compute gate and up projections using quantized matmul
                    let gate_out = input.apply_op1_no_bwd(&gate).ok()?;
                    let up_out = input.apply_op1_no_bwd(&up).ok()?;

                    // SwiGLU activation: silu(gate) * up
                    let activated = paramecia_nn::ops::silu(&gate_out).ok()?.mul(&up_out).ok()?;

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

            // Aggregate weighted expert outputs (residual is added on GPU)
            let mut output_f32 = vec![0.0f32; n_tokens * hidden_dim];

            for (_expert_id, results) in expert_results {
                for (token_id, weight, output_row) in results {
                    for j in 0..hidden_dim {
                        output_f32[token_id * hidden_dim + j] += output_row[j] * weight;
                    }
                }
            }

            // Convert output to F16 if input was F16 (halves return bandwidth)
            let output_data = if data.meta.use_f16 {
                HiddenData::F16(output_f32.iter().map(|&x| f16::from_f32(x)).collect())
            } else {
                HiddenData::F32(output_f32)
            };

            let _ = tx.send(MoeOutputData {
                layer_idx: data.meta.layer_idx,
                output_data,
                shape: (batch_size, seq_len, hidden_dim),
            });
        }
    }

    /// Submit a prefetch request (non-blocking)
    ///
    /// On Vulkan, this submits the GPU→CPU transfer asynchronously and sends
    /// the in-flight batch to the worker thread. The main thread can immediately
    /// continue recording GPU work for the next layer.
    ///
    /// On CUDA/Metal, this downloads to CPU on the calling thread (required for
    /// CUDA context), then sends the CPU data to the MoE worker.
    pub fn submit_prefetch(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        expert_indices: &Tensor,
        expert_weights: &Tensor,
        output_shape: (usize, usize, usize),
    ) -> Result<()> {
        // Check if input is F16 - if so, transfer as F16 for half the bandwidth
        let use_f16 = hidden_states.dtype() == DType::F16;

        let hidden_shape = hidden_states.dims2()?;
        let indices_shape = expert_indices.dims2()?;

        let meta = PrefetchMeta {
            layer_idx,
            hidden_shape,
            output_shape,
            indices_shape,
            use_f16,
        };

        // Vulkan async path: submit_async + send InFlightBatch to worker
        #[cfg(feature = "vulkan")]
        let msg: WorkerMessage = if let Device::Vulkan(vk) = hidden_states.device() {
            // Prepare tensors on GPU (dtype conversion + flatten are compute ops, batched)
            let h_flat = if use_f16 {
                hidden_states.flatten_all()?
            } else {
                hidden_states.to_dtype(DType::F32)?.flatten_all()?
            };
            let i_flat = expert_indices.flatten_all()?;
            let w_flat = expert_weights.to_dtype(DType::F32)?.flatten_all()?;

            // Get vk::Buffer handles
            let h_buf = match &*h_flat.storage_and_layout().0 {
                paramecia_core::Storage::Vulkan(s) => s.vk_buffer_arc()?,
                _ => unreachable!(),
            };
            let i_buf = match &*i_flat.storage_and_layout().0 {
                paramecia_core::Storage::Vulkan(s) => s.vk_buffer_arc()?,
                _ => unreachable!(),
            };
            let w_buf = match &*w_flat.storage_and_layout().0 {
                paramecia_core::Storage::Vulkan(s) => s.vk_buffer_arc()?,
                _ => unreachable!(),
            };

            let h_bytes = (h_flat.elem_count() * h_flat.dtype().size_in_bytes()) as u64;
            let i_bytes = (i_flat.elem_count() * i_flat.dtype().size_in_bytes()) as u64;
            let w_bytes = (w_flat.elem_count() * w_flat.dtype().size_in_bytes()) as u64;

            // Record all 3 copies into batch (residual stays on GPU)
            let h_dl = vk.prepare_download(h_buf, h_bytes)?;
            let i_dl = vk.prepare_download(i_buf, i_bytes)?;
            let w_dl = vk.prepare_download(w_buf, w_bytes)?;

            // Submit async — GPU starts executing, main thread continues immediately
            let in_flight = vk.submit_async()?.ok_or_else(|| {
                paramecia_core::Error::Msg("submit_async returned None (no batch)".into())
            })?;

            WorkerMessage::Async(AsyncPrefetchData {
                meta,
                in_flight,
                h_dl,
                i_dl,
                w_dl,
                h_count: h_flat.elem_count(),
                i_count: i_flat.elem_count(),
                w_count: w_flat.elem_count(),
            })
        } else {
            // Non-Vulkan GPU (CUDA/Metal) - download on calling thread
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
            let weights_cpu = expert_weights
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            WorkerMessage::Ready(PrefetchedData {
                meta,
                hidden_cpu,
                indices_cpu,
                weights_cpu,
            })
        };

        #[cfg(not(feature = "vulkan"))]
        let msg: WorkerMessage = {
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
            let weights_cpu = expert_weights
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            WorkerMessage::Ready(PrefetchedData {
                meta,
                hidden_cpu,
                indices_cpu,
                weights_cpu,
            })
        };

        self.pending_count.fetch_add(1, Ordering::SeqCst);

        self.msg_tx
            .send(msg)
            .map_err(|e| paramecia_core::Error::Msg(format!("Failed to submit data: {}", e)))
    }

    /// Wait for MoE result for a specific layer.
    ///
    /// `target_device` specifies which GPU to transfer the result to (supports multi-GPU
    /// layer parallelism where different layers live on different GPUs).
    pub fn wait_for_result(&self, expected_layer: usize, target_device: &Device) -> Result<Tensor> {
        let result = self
            .result_rx
            .recv()
            .map_err(|e| paramecia_core::Error::Msg(format!("Failed to receive result: {}", e)))?;

        self.pending_count.fetch_sub(1, Ordering::SeqCst);

        if result.layer_idx != expected_layer {
            paramecia_core::bail!(
                "Layer mismatch: expected {}, got {}",
                expected_layer,
                result.layer_idx
            );
        }

        let (batch_size, seq_len, hidden_dim) = result.shape;

        // Create tensor from HiddenData (F16 or F32), transfer to target device
        let output = match result.output_data {
            HiddenData::F16(data) => {
                Tensor::from_vec(data, (batch_size * seq_len, hidden_dim), &Device::Cpu)?
                    .reshape((batch_size, seq_len, hidden_dim))?
                    .to_device(target_device)?
            }
            HiddenData::F32(data) => {
                Tensor::from_vec(data, (batch_size * seq_len, hidden_dim), &Device::Cpu)?
                    .reshape((batch_size, seq_len, hidden_dim))?
                    .to_device(target_device)?
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
        let _ = self.msg_tx.send(WorkerMessage::Shutdown);
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
