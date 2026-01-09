//! Qwen3-Next implementation with quantization support.
//!
//! Based on Qwen3-Next architecture which is a hybrid model featuring:
//! - Full attention layers with gated Q projection
//! - Linear attention layers (Gated Delta Net) for efficient recurrence
//! - Mixture-of-Experts (MoE) FFN with shared experts
//!
//! References:
//! - llama.cpp qwen3next implementation
//! - Delta Net: https://arxiv.org/abs/2406.06484
//!
use crate::models::with_tracing::QMatMul;
use crate::quantized_nn::RmsNorm;
#[cfg(not(feature = "flash-attn"))]
use crate::utils::repeat_kv;
use std::collections::{HashMap, VecDeque};

/// LoRA adapter (A, B matrices) for fine-tuning
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// A matrix: (out_features, rank)
    pub a: Tensor,
    /// B matrix: (in_features, rank)
    pub b: Tensor,
}
use candle::quantized::{gguf_file, GgmlDType, QStorage, QTensor};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Activation, Embedding, Module};
use std::io::{Read, Seek};
use std::path::Path;

#[cfg(feature = "flash-attn")]
use candle_flash_attn;
use std::sync::Arc;

struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        QMatMul::from_weights(ws.into())
    }

    /// Load standard RmsNorm (GGUF weights already have Gemma +1 adjustment baked in)
    fn rms_norm(&mut self, name: &str, eps: f64, dtype: DType) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)?.to_dtype(dtype)
    }

    /// Alias for rms_norm
    fn rms_norm_gemma(&mut self, name: &str, eps: f64, dtype: DType) -> Result<RmsNorm> {
        self.rms_norm(name, eps, dtype)
    }

    fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }

    fn try_tensor(&mut self, name: &str) -> Result<Option<QTensor>> {
        match self.ct.tensor(&mut self.reader, name, &self.device) {
            Ok(t) => Ok(Some(t)),
            Err(_) => Ok(None),
        }
    }

    fn try_qmatmul(&mut self, name: &str) -> Result<Option<QMatMul>> {
        match self.try_tensor(name)? {
            Some(ws) => Ok(Some(QMatMul::from_weights(ws.into())?)),
            None => Ok(None),
        }
    }
}

/// KV-cache quantization mode for memory efficiency
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheQuantization {
    /// No quantization - store KV-cache in the model's native dtype (f16/bf16).
    /// Provides maximum accuracy.
    #[deprecated(since = "0.2.0", note = "Use F16 or BF16 instead for clarity")]
    None,
    /// Store KV-cache as f16 (16-bit float). Maximum accuracy, higher memory usage.
    F16,
    /// Store KV-cache as bf16 (bfloat16). Similar to F16 but with better dynamic range.
    /// Useful for models trained with bf16.
    BF16,
    /// Quantize to Q8_0 (8-bit, ~2x memory reduction with minimal accuracy loss).
    Q8_0,
    /// Quantize to Q4K (4-bit, ~4x memory reduction).
    /// This is the default setting.
    Q4K,
}

impl Default for KvCacheQuantization {
    fn default() -> Self {
        Self::Q4K
    }
}

impl KvCacheQuantization {
    fn to_ggml_dtype(self) -> Option<GgmlDType> {
        match self {
            #[allow(deprecated)]
            Self::None | Self::F16 | Self::BF16 => None,
            Self::Q8_0 => Some(GgmlDType::Q8_0),
            Self::Q4K => Some(GgmlDType::Q4K),
        }
    }

    fn block_size(&self) -> usize {
        match self {
            #[allow(deprecated)]
            Self::None | Self::F16 | Self::BF16 => 1,
            Self::Q8_0 => 32,
            Self::Q4K => 256,
        }
    }

    /// Returns the preferred DType for non-quantized cache storage.
    /// Returns None if using GGML quantization (Q8_0, Q4K).
    pub fn cache_dtype(&self) -> Option<candle::DType> {
        match self {
            #[allow(deprecated)]
            Self::None | Self::F16 => Some(candle::DType::F16),
            Self::BF16 => Some(candle::DType::BF16),
            Self::Q8_0 | Self::Q4K => None, // Uses GGML quantization
        }
    }

    /// Parse a KV cache quantization mode from a string.
    ///
    /// Accepts: "f16", "bf16", "q8", "q8_0", "q4", "q4k", "none" (case insensitive)
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "f16" | "fp16" => Some(Self::F16),
            "bf16" | "bfloat16" => Some(Self::BF16),
            "q8" | "q8_0" => Some(Self::Q8_0),
            "q4" | "q4k" | "q4_k" => Some(Self::Q4K),
            #[allow(deprecated)]
            "none" => Some(Self::None),
            _ => None,
        }
    }
}

impl std::str::FromStr for KvCacheQuantization {
    type Err = ();

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::from_str(s).ok_or(())
    }
}

/// Device offloading mode for MoE expert weights.
///
/// This enum controls how expert weights are distributed across devices for memory/speed tradeoffs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceOffloadMode {
    /// All expert weights on GPU (maximum speed, requires most VRAM).
    ///
    /// Best for GPUs with sufficient memory (24GB+ for 80B models).
    FullGpu,

    /// All expert weights (gate, up, down projections) offloaded to CPU.
    ///
    /// Significantly reduces VRAM usage but slower inference due to PCIe bandwidth.
    /// Best for GPUs with limited VRAM (8-16GB). Enables parallel CPU expert
    /// processing and is optimal for the async pipeline.
    #[default]
    ExpertsOnCpu,

    /// Only up projections offloaded to CPU, gate and down stay on GPU.
    ///
    /// Balanced approach: up projections are large but less frequently accessed.
    /// Gate projections benefit from GPU for fast routing, down projections
    /// benefit from GPU for fast output accumulation.
    /// Best for GPUs with moderate VRAM (16-24GB).
    UpProjectionsOnCpu,

    /// Up and down projections offloaded to CPU, gate stays on GPU.
    ///
    /// Gate projections benefit from GPU for fast routing decisions.
    /// Up and down projections (the largest weights) are on CPU to save VRAM.
    /// Best balance of VRAM savings and routing performance.
    UpDownProjectionsOnCpu,
}

impl DeviceOffloadMode {
    /// Returns the device placement for (gate, up, down) expert projections.
    pub fn get_expert_devices(&self, gpu_device: &Device) -> (Device, Device, Device) {
        match self {
            Self::FullGpu => (gpu_device.clone(), gpu_device.clone(), gpu_device.clone()),
            Self::ExpertsOnCpu => (Device::Cpu, Device::Cpu, Device::Cpu),
            Self::UpProjectionsOnCpu => (gpu_device.clone(), Device::Cpu, gpu_device.clone()),
            Self::UpDownProjectionsOnCpu => (gpu_device.clone(), Device::Cpu, Device::Cpu),
        }
    }
}

/// Default maximum sequence length for pre-allocated KV cache
const DEFAULT_MAX_SEQ_LEN: usize = 8192;

/// Pre-allocated KV cache with O(1) token insertion.
///
/// Unlike the naive `Tensor::cat` approach which is O(n) per token (O(n²) overall),
/// this implementation pre-allocates buffers and uses slice assignment for O(1) insertion.
/// This matches llama.cpp's approach using `ggml_set_rows` with pre-allocated buffers.
#[derive(Debug)]
struct PreallocatedKvCache {
    /// Pre-allocated K tensor: [batch, num_kv_heads, max_seq_len, head_dim]
    k_cache: Tensor,
    /// Pre-allocated V tensor: [batch, num_kv_heads, max_seq_len, head_dim]
    v_cache: Tensor,
    /// Current number of tokens stored (head position)
    seq_len: usize,
    /// Maximum capacity
    max_seq_len: usize,
    /// Batch size (for validation)
    batch_size: usize,
}

#[allow(dead_code)]
impl PreallocatedKvCache {
    /// Create a new pre-allocated KV cache
    fn new(
        batch_size: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        // Pre-allocate K and V tensors with full capacity
        // Shape: [batch, num_kv_heads, max_seq_len, head_dim]
        let k_cache = Tensor::zeros(
            (batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype,
            device,
        )?;
        let v_cache = Tensor::zeros(
            (batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype,
            device,
        )?;

        Ok(Self {
            k_cache,
            v_cache,
            seq_len: 0,
            max_seq_len,
            batch_size,
        })
    }

    /// Append new K/V tensors to the cache. Returns the full K/V up to current position.
    ///
    /// This is O(1) per call since we use slice assignment instead of concatenation.
    /// new_k, new_v shape: [batch, num_kv_heads, new_seq_len, head_dim]
    fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        let new_seq_len = new_k.dim(2)?;
        let new_end = self.seq_len + new_seq_len;

        if new_end > self.max_seq_len {
            candle::bail!(
                "KV cache overflow: trying to store {} tokens but max is {}",
                new_end,
                self.max_seq_len
            );
        }

        // Use slice_scatter to write new K/V at the current position
        // This avoids the O(n) copy that Tensor::cat would require
        self.k_cache = self.k_cache.slice_scatter(new_k, 2, self.seq_len)?;
        self.v_cache = self.v_cache.slice_scatter(new_v, 2, self.seq_len)?;

        self.seq_len = new_end;

        // Return view of valid portion only
        let k = self.k_cache.narrow(2, 0, self.seq_len)?;
        let v = self.v_cache.narrow(2, 0, self.seq_len)?;

        Ok((k, v))
    }

    /// Get current sequence length
    fn len(&self) -> usize {
        self.seq_len
    }

    /// Truncate the cache to a given length.
    /// This is O(1) - just updates the seq_len pointer.
    fn truncate(&mut self, new_len: usize) {
        self.seq_len = new_len.min(self.seq_len);
    }

    /// Check if cache is empty
    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Clear the cache
    fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Resize cache for a new batch size (clears existing data)
    fn resize_batch(&mut self, batch_size: usize, dtype: DType) -> Result<()> {
        if batch_size != self.batch_size {
            let num_kv_heads = self.k_cache.dim(1)?;
            let head_dim = self.k_cache.dim(3)?;
            let device = self.k_cache.device().clone();

            self.k_cache = Tensor::zeros(
                (batch_size, num_kv_heads, self.max_seq_len, head_dim),
                dtype,
                &device,
            )?;
            self.v_cache = Tensor::zeros(
                (batch_size, num_kv_heads, self.max_seq_len, head_dim),
                dtype,
                &device,
            )?;
            self.batch_size = batch_size;
            self.seq_len = 0;
        }
        Ok(())
    }
}

/// Pre-allocated quantized KV cache with O(1) token insertion.
///
/// Similar to `PreallocatedKvCache` but stores K/V in quantized format.
/// Key insight from llama.cpp: pre-allocate quantized buffer and quantize
/// only new tokens on each append, avoiding O(n) re-quantization.
///
/// Used when `KvCacheQuantization` is set to `Q8_0` or `Q4K` for memory-efficient
/// long-context inference with significant VRAM savings.
#[derive(Debug)]
struct PreallocatedQuantizedKvCache {
    /// Pre-allocated quantized K buffer (stores raw quantized bytes)
    /// Shape conceptually: [batch, num_kv_heads, max_seq_len, head_dim]
    /// But stored as flat quantized data with block structure
    k_cache: Vec<u8>,
    /// Pre-allocated quantized V buffer
    v_cache: Vec<u8>,
    /// GGML dtype for quantization
    ggml_dtype: GgmlDType,
    /// Current number of tokens stored
    seq_len: usize,
    /// Maximum capacity
    max_seq_len: usize,
    /// Batch size
    batch_size: usize,
    /// Number of KV heads
    num_kv_heads: usize,
    /// Head dimension (may be padded for block alignment)
    head_dim: usize,
    /// Padded head dimension (aligned to block size)
    padded_head_dim: usize,
    /// Block size for quantization (stored for potential future use)
    #[allow(dead_code)]
    block_size: usize,
    /// Bytes per row (one row = one head for one token)
    bytes_per_row: usize,
    /// Device for dequantization
    device: Device,
}

impl PreallocatedQuantizedKvCache {
    /// Create a new pre-allocated quantized KV cache
    fn new(
        batch_size: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        ggml_dtype: GgmlDType,
        device: &Device,
    ) -> Result<Self> {
        let block_size = ggml_dtype.block_size();
        let padded_head_dim = head_dim.div_ceil(block_size) * block_size;

        // Calculate bytes per row based on the quantization format
        // Each row is one head for one token: [padded_head_dim] elements
        let type_size = ggml_dtype.type_size();
        let num_blocks = padded_head_dim / block_size;
        let bytes_per_row = num_blocks * type_size;

        // Total rows = batch * num_heads * max_seq_len
        let total_rows = batch_size * num_kv_heads * max_seq_len;
        let total_bytes = total_rows * bytes_per_row;

        // Pre-allocate with zeros
        let k_cache = vec![0u8; total_bytes];
        let v_cache = vec![0u8; total_bytes];

        Ok(Self {
            k_cache,
            v_cache,
            ggml_dtype,
            seq_len: 0,
            max_seq_len,
            batch_size,
            num_kv_heads,
            head_dim,
            padded_head_dim,
            block_size,
            bytes_per_row,
            device: device.clone(),
        })
    }

    /// Append new K/V tensors to the cache.
    ///
    /// This quantizes only the new tokens (O(new_tokens)) and writes them
    /// to the pre-allocated buffer at the current position.
    ///
    /// new_k, new_v shape: [batch, num_kv_heads, new_seq_len, head_dim]
    fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        let new_seq_len = new_k.dim(2)?;
        let new_end = self.seq_len + new_seq_len;

        if new_end > self.max_seq_len {
            candle::bail!(
                "Quantized KV cache overflow: trying to store {} tokens but max is {}",
                new_end,
                self.max_seq_len
            );
        }

        // Pad the new K/V if head_dim isn't aligned to block size
        let (new_k_padded, new_v_padded) = if self.head_dim != self.padded_head_dim {
            let pad_size = self.padded_head_dim - self.head_dim;
            let b = new_k.dim(0)?;
            let zeros_k = Tensor::zeros(
                (b, self.num_kv_heads, new_seq_len, pad_size),
                new_k.dtype(),
                new_k.device(),
            )?;
            let zeros_v = Tensor::zeros(
                (b, self.num_kv_heads, new_seq_len, pad_size),
                new_v.dtype(),
                new_v.device(),
            )?;
            (
                Tensor::cat(&[new_k, &zeros_k], 3)?,
                Tensor::cat(&[new_v, &zeros_v], 3)?,
            )
        } else {
            (new_k.clone(), new_v.clone())
        };

        // Convert to F32 for quantization if needed
        let new_k_f32 = new_k_padded.to_dtype(DType::F32)?.contiguous()?;
        let new_v_f32 = new_v_padded.to_dtype(DType::F32)?.contiguous()?;

        // Quantize only the new tokens
        let new_k_qtensor = QTensor::quantize(&new_k_f32.flatten_all()?, self.ggml_dtype)?;
        let new_v_qtensor = QTensor::quantize(&new_v_f32.flatten_all()?, self.ggml_dtype)?;

        // Get the raw quantized bytes
        let new_k_bytes = new_k_qtensor.data()?;
        let new_v_bytes = new_v_qtensor.data()?;

        // Calculate where to write in the pre-allocated buffer
        // Layout: [batch, num_kv_heads, seq_len, head_dim] but stored as rows
        // Each position in the sequence has batch * num_kv_heads rows
        let rows_per_position = self.batch_size * self.num_kv_heads;
        let start_offset = self.seq_len * rows_per_position * self.bytes_per_row;
        let bytes_to_copy = new_seq_len * rows_per_position * self.bytes_per_row;

        // Copy quantized data to pre-allocated buffer
        self.k_cache[start_offset..start_offset + bytes_to_copy]
            .copy_from_slice(&new_k_bytes.as_ref()[..bytes_to_copy]);
        self.v_cache[start_offset..start_offset + bytes_to_copy]
            .copy_from_slice(&new_v_bytes.as_ref()[..bytes_to_copy]);

        self.seq_len = new_end;

        // Dequantize and return the full cache up to current position
        self.get_kv()
    }

    /// Get the current K/V tensors (dequantized)
    fn get_kv(&self) -> Result<(Tensor, Tensor)> {
        if self.seq_len == 0 {
            // Return empty tensors
            let k = Tensor::zeros(
                (self.batch_size, self.num_kv_heads, 0, self.head_dim),
                DType::F32,
                &self.device,
            )?;
            let v = Tensor::zeros(
                (self.batch_size, self.num_kv_heads, 0, self.head_dim),
                DType::F32,
                &self.device,
            )?;
            return Ok((k, v));
        }

        // Calculate how many bytes represent the current sequence
        let rows_per_position = self.batch_size * self.num_kv_heads;
        let valid_bytes = self.seq_len * rows_per_position * self.bytes_per_row;

        // Create QStorage from the valid portion of the cache
        let k_storage = QStorage::from_data(
            std::borrow::Cow::Borrowed(&self.k_cache[..valid_bytes]),
            &self.device,
            self.ggml_dtype,
        )?;
        let v_storage = QStorage::from_data(
            std::borrow::Cow::Borrowed(&self.v_cache[..valid_bytes]),
            &self.device,
            self.ggml_dtype,
        )?;

        // Create QTensor from the storage
        let k_qtensor = QTensor::new(
            k_storage,
            (
                self.batch_size * self.num_kv_heads * self.seq_len,
                self.padded_head_dim,
            ),
        )?;
        let v_qtensor = QTensor::new(
            v_storage,
            (
                self.batch_size * self.num_kv_heads * self.seq_len,
                self.padded_head_dim,
            ),
        )?;

        // Dequantize
        let k_deq = k_qtensor.dequantize(&self.device)?;
        let v_deq = v_qtensor.dequantize(&self.device)?;

        // Reshape and trim padding
        let k = k_deq
            .reshape((
                self.batch_size,
                self.num_kv_heads,
                self.seq_len,
                self.padded_head_dim,
            ))?
            .narrow(3, 0, self.head_dim)?;
        let v = v_deq
            .reshape((
                self.batch_size,
                self.num_kv_heads,
                self.seq_len,
                self.padded_head_dim,
            ))?
            .narrow(3, 0, self.head_dim)?;

        Ok((k, v))
    }

    /// Clear the cache
    fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Truncate the cache to a given length.
    /// This is O(1) - just updates the seq_len pointer.
    fn truncate(&mut self, new_len: usize) {
        self.seq_len = new_len.min(self.seq_len);
    }

    /// Resize cache for a new batch size
    #[allow(dead_code)]
    fn resize_batch(&mut self, batch_size: usize) -> Result<()> {
        if batch_size != self.batch_size {
            let total_rows = batch_size * self.num_kv_heads * self.max_seq_len;
            let total_bytes = total_rows * self.bytes_per_row;
            self.k_cache = vec![0u8; total_bytes];
            self.v_cache = vec![0u8; total_bytes];
            self.batch_size = batch_size;
            self.seq_len = 0;
        }
        Ok(())
    }
}

/// Legacy storage for quantized or unquantized KV-cache.
///
/// **Deprecated:** This is superseded by `PreallocatedQuantizedKvCache` for quantized
/// storage and `PreallocatedKvCache` for non-quantized storage. Both are now properly
/// wired into the forward pass based on `KvCacheQuantization` settings.
/// Kept for reference and potential future use cases.
#[allow(dead_code)]
#[derive(Debug)]
enum KvCacheStorage {
    Float(Tensor, Tensor),
    Quantized(QTensor, QTensor, Vec<usize>, Vec<usize>),
}

#[allow(dead_code)]
impl KvCacheStorage {
    fn get_kv_ref(&self) -> Result<(&Tensor, &Tensor)> {
        match self {
            Self::Float(k, v) => Ok((k, v)),
            Self::Quantized(_, _, _, _) => {
                candle::bail!("Use get_kv_owned for quantized cache")
            }
        }
    }

    fn get_kv_owned(&self) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Float(k, v) => Ok((k.clone(), v.clone())),
            Self::Quantized(k_qtensor, v_qtensor, k_padded_shape, v_padded_shape) => {
                let k_deq = k_qtensor.dequantize(&k_qtensor.device())?;
                let v_deq = v_qtensor.dequantize(&v_qtensor.device())?;
                let k_padded = k_deq.reshape(k_padded_shape.as_slice())?;
                let v_padded = v_deq.reshape(v_padded_shape.as_slice())?;
                Ok((k_padded, v_padded))
            }
        }
    }
}

/// Recurrent state storage for linear attention layers with double-buffering
/// for efficient checkpoint/restore during speculative decoding.
#[derive(Debug, Clone)]
struct RecurrentState {
    /// SSM/delta-net state: [batch, num_heads, state_dim, state_dim]
    ssm_state: Tensor,
    /// Convolution state: [batch, conv_dim, conv_kernel_size - 1]
    conv_state: Tensor,
    /// Accumulated gate cumsum offset for prefix cache continuation.
    /// Shape: [batch, num_heads]. When continuing from prefix cache, this offset
    /// is added to the new tokens' gate_cumsum to correctly compute decay factors.
    gate_cumsum_offset: Option<Tensor>,
    /// Backup buffer for checkpoint/restore (lazy allocated)
    backup_ssm: Option<Tensor>,
    backup_conv: Option<Tensor>,
    backup_gate_offset: Option<Tensor>,
    /// Intermediate states for verification state-slicing.
    /// Stores SSM state after each position: Vec of [batch, num_heads, state_dim, state_dim]
    intermediate_states: Option<Vec<Tensor>>,
    /// Intermediate conv states for verification state-slicing.
    intermediate_conv_states: Option<Vec<Tensor>>,
}

impl RecurrentState {
    fn new(ssm_state: Tensor, conv_state: Tensor) -> Self {
        Self {
            ssm_state,
            conv_state,
            gate_cumsum_offset: None,
            backup_ssm: None,
            backup_conv: None,
            backup_gate_offset: None,
            intermediate_states: None,
            intermediate_conv_states: None,
        }
    }

    fn with_gate_offset(ssm_state: Tensor, conv_state: Tensor, gate_offset: Tensor) -> Self {
        Self {
            ssm_state,
            conv_state,
            gate_cumsum_offset: Some(gate_offset),
            backup_ssm: None,
            backup_conv: None,
            backup_gate_offset: None,
            intermediate_states: None,
            intermediate_conv_states: None,
        }
    }

    /// Checkpoint: save reference to current state tensors.
    ///
    /// This uses shallow clone (O(1)) which is safe because:
    /// - State updates use field assignment (e.g., `rs.ssm_state = new_tensor`)
    /// - This replaces the tensor reference, not mutating underlying data
    /// - So backup's reference to the old tensor remains valid
    fn checkpoint(&mut self) -> Result<()> {
        self.backup_ssm = Some(self.ssm_state.clone());
        self.backup_conv = Some(self.conv_state.clone());
        self.backup_gate_offset = self.gate_cumsum_offset.clone();
        Ok(())
    }

    /// Restore: swap backup references back to primary.
    /// O(1) operation - just swaps tensor references.
    fn restore(&mut self) -> Result<()> {
        if let Some(backup) = self.backup_ssm.take() {
            self.ssm_state = backup;
        }
        if let Some(backup) = self.backup_conv.take() {
            self.conv_state = backup;
        }
        self.gate_cumsum_offset = self.backup_gate_offset.take();
        Ok(())
    }

    /// Clear intermediate states buffer.
    fn clear_intermediate_states(&mut self) {
        self.intermediate_states = None;
        self.intermediate_conv_states = None;
    }

    /// Initialize intermediate states buffer for a given sequence length.
    /// NOTE: Part of speculative decoding verification - not yet wired up.
    #[allow(dead_code)]
    fn init_intermediate_states(&mut self, seq_len: usize) {
        self.intermediate_states = Some(Vec::with_capacity(seq_len));
        self.intermediate_conv_states = Some(Vec::with_capacity(seq_len));
    }

    /// Save current state as an intermediate state.
    /// NOTE: Part of speculative decoding verification - not yet wired up.
    #[allow(dead_code)]
    fn save_intermediate_state(&mut self) {
        if let Some(ref mut states) = self.intermediate_states {
            states.push(self.ssm_state.clone());
        }
        if let Some(ref mut states) = self.intermediate_conv_states {
            states.push(self.conv_state.clone());
        }
    }

    /// Restore to a specific intermediate state by index.
    /// Returns true if successful, false if index out of bounds.
    fn restore_to_intermediate(&mut self, index: usize) -> bool {
        let ssm_ok = if let Some(ref states) = self.intermediate_states {
            if index < states.len() {
                self.ssm_state = states[index].clone();
                true
            } else {
                false
            }
        } else {
            false
        };

        let conv_ok = if let Some(ref states) = self.intermediate_conv_states {
            if index < states.len() {
                self.conv_state = states[index].clone();
                true
            } else {
                false
            }
        } else {
            false
        };

        ssm_ok && conv_ok
    }

    /// Get the number of stored intermediate states.
    /// NOTE: Part of speculative decoding verification - not yet wired up.
    #[allow(dead_code)]
    fn num_intermediate_states(&self) -> usize {
        self.intermediate_states
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(0)
    }
}

/// Checkpoint marker for a single layer, used for speculative decoding rollback.
/// For full attention: stores the KV cache sequence length.
/// For linear attention: just a marker (state is stored in-place in the layer).
pub enum LayerCheckpoint {
    /// Full attention: just need the KV cache sequence length (O(1) to save/restore)
    FullAttention { seq_len: usize },
    /// Linear attention: marker only - state is stored in layer's backup buffers
    LinearAttention,
}

/// Saved cache state for prefix caching (deep copy, not just length markers).
/// This allows restoring the full KV cache state across different forward calls.
#[derive(Clone)]
pub enum PrefixCacheEntry {
    /// Full attention: cloned K/V tensors
    FullAttention {
        k_cache: Tensor,
        v_cache: Tensor,
        seq_len: usize,
    },
    /// Linear attention: cloned SSM and conv states
    LinearAttention {
        ssm_state: Tensor,
        conv_state: Tensor,
        /// Accumulated gate cumsum at the end of prefix: [batch, num_heads]
        /// Used to offset new tokens' gate_cumsum for correct decay computation.
        gate_cumsum_offset: Tensor,
    },
    /// No cache (layer not yet initialized or cleared)
    Empty,
}

/// Complete prefix cache state for the entire model.
/// Stores the token prefix and all layer cache states.
#[derive(Clone)]
pub struct PrefixCache {
    /// Token IDs that this cache was computed for
    pub prefix_tokens: Vec<u32>,
    /// Saved cache state for each layer
    pub layer_caches: Vec<PrefixCacheEntry>,
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    /// Number of dimensions to rotate (partial rotary embedding).
    /// For Qwen3-Next this is typically 64 (25% of head_dim 256).
    n_rot: usize,
    /// Full head dimension
    head_dim: usize,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        n_rot: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        // Only compute frequencies for the first n_rot dimensions.
        // The frequency exponent uses n_rot as the base (matching llama.cpp).
        let max_seq_len = max_position_embeddings;
        let inv_freq: Vec<_> = (0..n_rot)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / n_rot as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
            n_rot,
            head_dim,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, d) = q.dims4()?;

        // If n_rot == head_dim, rotate everything (standard RoPE)
        if self.n_rot == self.head_dim {
            let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
            let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
            let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            return Ok((q_embed, k_embed));
        }

        // Partial rotary embedding: only rotate first n_rot dimensions
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;

        // Split Q into rotated and pass-through parts
        // Q shape: [batch, heads, seq, head_dim]
        let q_rot = q.narrow(D::Minus1, 0, self.n_rot)?.contiguous()?;
        let q_pass = q.narrow(D::Minus1, self.n_rot, d - self.n_rot)?;

        // Split K into rotated and pass-through parts
        let k_rot = k.narrow(D::Minus1, 0, self.n_rot)?.contiguous()?;
        let k_pass = k.narrow(D::Minus1, self.n_rot, d - self.n_rot)?;

        // Apply RoPE only to the rotated parts
        let q_rot_embed = candle_nn::rotary_emb::rope(&q_rot, &cos, &sin)?;
        let k_rot_embed = candle_nn::rotary_emb::rope(&k_rot, &cos, &sin)?;

        // Concatenate rotated and pass-through parts
        let q_embed = Tensor::cat(&[&q_rot_embed, &q_pass], D::Minus1)?;
        let k_embed = Tensor::cat(&[&k_rot_embed, &k_pass], D::Minus1)?;

        Ok((q_embed, k_embed))
    }
}

// ============================================================================
// MoE Components
// ============================================================================

#[derive(Debug, Clone)]
struct CachedMatmuls {
    gate: Arc<QMatMul>,
    up: Arc<QMatMul>,
    down: Arc<QMatMul>,
    gate_device: Device,
    up_device: Device,
    down_device: Device,
}

/// GPU Hot Expert Cache - keeps frequently used experts on GPU even when
/// the main expert weights are on CPU. This is the key optimization for
/// CPU-offloaded MoE: avoid CPU computation entirely for hot experts.
struct GpuHotExpertCache {
    /// GPU device for caching
    gpu_device: Device,
    /// Cached expert matmuls on GPU: expert_idx -> CachedMatmuls
    entries: HashMap<usize, CachedMatmuls>,
    /// Usage counts for each expert
    usage_counts: Vec<u64>,
    /// LRU tracking
    lru: VecDeque<usize>,
    /// Maximum number of experts to cache on GPU
    capacity: usize,
    /// Total tokens processed (for promotion decisions)
    total_tokens: u64,
    /// Promotion threshold (min usage to promote to GPU)
    promotion_threshold: u64,
}

impl std::fmt::Debug for GpuHotExpertCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuHotExpertCache")
            .field("capacity", &self.capacity)
            .field("cached_count", &self.entries.len())
            .finish()
    }
}

impl GpuHotExpertCache {
    fn new(gpu_device: Device, num_experts: usize, capacity: usize) -> Self {
        Self {
            gpu_device,
            entries: HashMap::new(),
            usage_counts: vec![0u64; num_experts],
            lru: VecDeque::new(),
            capacity,
            total_tokens: 0,
            promotion_threshold: 1, // Promote immediately on first use
        }
    }

    /// Record expert usage and return whether it's now a hot expert
    fn record_usage(&mut self, expert_idx: usize) -> bool {
        if expert_idx < self.usage_counts.len() {
            self.usage_counts[expert_idx] += 1;
            self.total_tokens += 1;
        }
        self.usage_counts.get(expert_idx).copied().unwrap_or(0) >= self.promotion_threshold
    }

    /// Check if expert is cached on GPU
    #[allow(dead_code)]
    fn is_cached(&self, expert_idx: usize) -> bool {
        self.entries.contains_key(&expert_idx)
    }

    /// Get cached expert if available
    fn get(&mut self, expert_idx: usize) -> Option<CachedMatmuls> {
        if let Some(entry) = self.entries.get(&expert_idx).cloned() {
            // Update LRU
            self.lru.retain(|&idx| idx != expert_idx);
            self.lru.push_back(expert_idx);
            Some(entry)
        } else {
            None
        }
    }

    /// Add expert to GPU cache
    fn insert(&mut self, expert_idx: usize, entry: CachedMatmuls) {
        // Evict if at capacity
        while self.entries.len() >= self.capacity {
            if let Some(oldest) = self.lru.pop_front() {
                self.entries.remove(&oldest);
            } else {
                break;
            }
        }

        self.entries.insert(expert_idx, entry);
        self.lru.retain(|&idx| idx != expert_idx);
        self.lru.push_back(expert_idx);
    }

    /// Build GPU-cached entry from CPU source tensors
    fn build_gpu_entry(
        &self,
        expert_idx: usize,
        gate_src: &Arc<QTensor>,
        up_src: &Arc<QTensor>,
        down_src: &Arc<QTensor>,
    ) -> Result<CachedMatmuls> {
        // Slice out the expert
        let gate_slice = gate_src.slice_first_dim(expert_idx)?;
        let up_slice = up_src.slice_first_dim(expert_idx)?;
        let down_slice = down_src.slice_first_dim(expert_idx)?;

        // Copy to GPU
        let gate_gpu = Self::copy_to_gpu(&gate_slice, &self.gpu_device)?;
        let up_gpu = Self::copy_to_gpu(&up_slice, &self.gpu_device)?;
        let down_gpu = Self::copy_to_gpu(&down_slice, &self.gpu_device)?;

        Ok(CachedMatmuls {
            gate: Arc::new(QMatMul::from_weights(Arc::new(gate_gpu))?),
            up: Arc::new(QMatMul::from_weights(Arc::new(up_gpu))?),
            down: Arc::new(QMatMul::from_weights(Arc::new(down_gpu))?),
            gate_device: self.gpu_device.clone(),
            up_device: self.gpu_device.clone(),
            down_device: self.gpu_device.clone(),
        })
    }

    fn copy_to_gpu(src: &QTensor, gpu_device: &Device) -> Result<QTensor> {
        // Always copy to ensure we have a GPU version
        let data = src.data()?;
        let storage = QStorage::from_data(data, gpu_device, src.dtype())?;
        QTensor::new(storage, src.shape().clone())
    }

    /// Get the top-k most used experts
    #[allow(dead_code)]
    fn top_k_experts(&self, k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, u64)> = self
            .usage_counts
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c))
            .collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));
        indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
    }

    /// Get cache hit rate
    #[allow(dead_code)]
    fn hit_rate(&self) -> f64 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        let cached_usage: u64 = self.entries.keys().map(|&idx| self.usage_counts[idx]).sum();
        cached_usage as f64 / self.total_tokens as f64
    }
}

#[derive(Debug)]
struct ExpertCache {
    target_device: Device,
    capacity: usize,
    entries: HashMap<usize, CachedMatmuls>,
    lru: VecDeque<usize>,
}

impl ExpertCache {
    fn new(target_device: Device, capacity: usize) -> Self {
        Self {
            target_device,
            capacity,
            entries: HashMap::new(),
            lru: VecDeque::new(),
        }
    }

    fn enabled(&self) -> bool {
        self.capacity > 0
    }

    fn touch(&mut self, expert_idx: usize) {
        self.lru.retain(|idx| *idx != expert_idx);
        self.lru.push_back(expert_idx);
    }

    fn evict_if_needed(&mut self) {
        while self.entries.len() >= self.capacity {
            if let Some(oldest) = self.lru.pop_front() {
                self.entries.remove(&oldest);
            } else {
                break;
            }
        }
    }

    fn materialize_qtensor(&self, src: &Arc<QTensor>) -> Result<Arc<QTensor>> {
        if src.device().same_device(&self.target_device) {
            return Ok(src.clone());
        }

        let data = src.data()?;
        let storage = QStorage::from_data(data, &self.target_device, src.dtype())?;
        let qtensor = QTensor::new(storage, src.shape().clone())?;
        Ok(Arc::new(qtensor))
    }

    fn build_entry(
        &self,
        expert_idx: usize,
        gate_src: &Arc<QTensor>,
        up_src: &Arc<QTensor>,
        down_src: &Arc<QTensor>,
    ) -> Result<CachedMatmuls> {
        let gate_slice = gate_src.slice_first_dim(expert_idx)?;
        let up_slice = up_src.slice_first_dim(expert_idx)?;
        let down_slice = down_src.slice_first_dim(expert_idx)?;

        let gate_qtensor = self.materialize_qtensor(&Arc::new(gate_slice))?;
        let up_qtensor = self.materialize_qtensor(&Arc::new(up_slice))?;
        let down_qtensor = self.materialize_qtensor(&Arc::new(down_slice))?;

        Ok(CachedMatmuls {
            gate: Arc::new(QMatMul::from_weights(gate_qtensor)?),
            up: Arc::new(QMatMul::from_weights(up_qtensor)?),
            down: Arc::new(QMatMul::from_weights(down_qtensor)?),
            gate_device: self.target_device.clone(),
            up_device: self.target_device.clone(),
            down_device: self.target_device.clone(),
        })
    }

    fn get_or_prepare(
        &mut self,
        expert_idx: usize,
        gate_src: &Arc<QTensor>,
        up_src: &Arc<QTensor>,
        down_src: &Arc<QTensor>,
    ) -> Result<CachedMatmuls> {
        if let Some(entry) = self.entries.get(&expert_idx).cloned() {
            self.touch(expert_idx);
            return Ok(entry);
        }

        self.evict_if_needed();
        let entry = self.build_entry(expert_idx, gate_src, up_src, down_src)?;
        self.entries.insert(expert_idx, entry.clone());
        self.touch(expert_idx);
        Ok(entry)
    }
}

fn should_cache_experts(
    gate_exps: &Arc<QTensor>,
    up_exps: &Arc<QTensor>,
    down_exps: &Arc<QTensor>,
    compute_device: &Device,
) -> bool {
    !matches!(compute_device, Device::Cpu)
        && (!gate_exps.device().same_device(compute_device)
            || !up_exps.device().same_device(compute_device)
            || !down_exps.device().same_device(compute_device))
}

struct MoeExperts {
    gate_exps: Arc<QTensor>,
    up_exps: Arc<QTensor>,
    down_exps: Arc<QTensor>,
    act_fn: Activation,
    span: tracing::Span,
    cache: Option<ExpertCache>,
    training_cache: Option<ExpertCache>,
    compute_device: Device,
    custom_gate_block_mults: Option<Tensor>,
    custom_up_block_mults: Option<Tensor>,
    custom_down_block_mults: Option<Tensor>,
    /// GPU hot expert cache - keeps frequently used experts on GPU
    /// even when main expert weights are on CPU
    gpu_hot_cache: Option<GpuHotExpertCache>,
    /// The GPU device for hot caching (may differ from compute_device)
    gpu_device: Option<Device>,
    /// Whether expert weights have transposed layout
    /// When true, expert weights are [n_exp, in_dim, out_dim] instead of [n_exp, out_dim, in_dim]
    transposed_layout: bool,
}

impl std::fmt::Debug for MoeExperts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeExperts")
            .field("gate_exps", &self.gate_exps)
            .field("up_exps", &self.up_exps)
            .field("down_exps", &self.down_exps)
            .field("act_fn", &self.act_fn)
            .field("compute_device", &self.compute_device)
            .field("gpu_hot_cache", &self.gpu_hot_cache)
            .finish()
    }
}

impl MoeExperts {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        compute_device: &Device,
        cache_capacity: usize,
    ) -> Result<Self> {
        Self::new_with_gpu_cache(gg, prefix, compute_device, cache_capacity, None)
    }

    fn new_with_gpu_cache<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        compute_device: &Device,
        cache_capacity: usize,
        gpu_device: Option<Device>,
    ) -> Result<Self> {
        let gate_exps = Arc::new(gg.tensor(&format!("{}.ffn_gate_exps.weight", prefix))?);
        let up_exps = Arc::new(gg.tensor(&format!("{}.ffn_up_exps.weight", prefix))?);
        let down_exps = Arc::new(gg.tensor(&format!("{}.ffn_down_exps.weight", prefix))?);
        let act_fn = Activation::Silu;
        let span = tracing::span!(tracing::Level::TRACE, "moe-experts");

        // Get number of experts from shape
        let num_experts = gate_exps.shape().dims().first().copied().unwrap_or(64);

        let cache = if cache_capacity == 0 {
            None
        } else if should_cache_experts(&gate_exps, &up_exps, &down_exps, compute_device) {
            Some(ExpertCache::new(compute_device.clone(), cache_capacity))
        } else {
            None
        };

        let training_cache =
            if should_cache_experts(&gate_exps, &up_exps, &down_exps, compute_device) {
                Some(ExpertCache::new(compute_device.clone(), cache_capacity))
            } else {
                None
            };

        // Create GPU hot cache if:
        // 1. We have a GPU device
        // 2. Experts are on CPU (so we benefit from caching hot experts on GPU)
        let gpu_hot_cache = if let Some(ref gpu_dev) = gpu_device {
            if matches!(gpu_dev, Device::Cuda(_)) && !matches!(compute_device, Device::Cuda(_)) {
                // Cache up to 16 hot experts on GPU (configurable)
                Some(GpuHotExpertCache::new(gpu_dev.clone(), num_experts, 16))
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            gate_exps,
            up_exps,
            down_exps,
            act_fn,
            span,
            cache,
            training_cache,
            compute_device: compute_device.clone(),
            custom_gate_block_mults: None,
            custom_up_block_mults: None,
            custom_down_block_mults: None,
            gpu_hot_cache,
            gpu_device,
            transposed_layout: false,
        })
    }

    /// Create MoeExperts for MTP layers using GGUF-style weight names
    /// Uses blk.{idx}.mtp.ffn_{gate|up|down}_exps.weight naming convention
    /// The experts are packed into a single 3D tensor per projection type
    /// Create MoeExperts for MTP layers using GGUF-style weight names.
    ///
    /// With the fixed GGUF converter, MTP expert weights have the same layout as main model:
    /// - gate/up: [n_exp, intermediate, hidden]
    /// - down: [n_exp, hidden, intermediate]
    ///   No transpose needed - weights are loaded directly.
    #[allow(clippy::too_many_arguments)]
    fn new_mtp<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        compute_device: &Device,
        cache_capacity: usize,
    ) -> Result<Self> {
        // MTP uses GGUF naming with packed experts: blk.{idx}.mtp.ffn_gate_exps.weight
        // With fixed converter, layout matches main model - no transpose needed
        let gate_exps = Arc::new(gg.tensor(&format!("{}.ffn_gate_exps.weight", prefix))?);
        let up_exps = Arc::new(gg.tensor(&format!("{}.ffn_up_exps.weight", prefix))?);
        let down_exps = Arc::new(gg.tensor(&format!("{}.ffn_down_exps.weight", prefix))?);

        let act_fn = Activation::Silu;
        let span = tracing::span!(tracing::Level::TRACE, "mtp-experts");

        // Get number of experts from shape
        let _num_experts = gate_exps.shape().dims().first().copied().unwrap_or(512);

        let cache = if cache_capacity == 0 {
            None
        } else if should_cache_experts(&gate_exps, &up_exps, &down_exps, compute_device) {
            Some(ExpertCache::new(compute_device.clone(), cache_capacity))
        } else {
            None
        };

        let training_cache =
            if should_cache_experts(&gate_exps, &up_exps, &down_exps, compute_device) {
                Some(ExpertCache::new(compute_device.clone(), cache_capacity))
            } else {
                None
            };

        Ok(Self {
            gate_exps,
            up_exps,
            down_exps,
            act_fn,
            span,
            cache,
            training_cache,
            compute_device: compute_device.clone(),
            custom_gate_block_mults: None,
            custom_up_block_mults: None,
            custom_down_block_mults: None,
            gpu_hot_cache: None,
            gpu_device: None,
            // Weights are pre-transposed during loading, no runtime transpose needed
            transposed_layout: false,
        })
    }

    /// Enable GPU hot caching for CPU-offloaded experts
    fn enable_gpu_hot_cache(&mut self, gpu_device: Device, capacity: usize) {
        let num_experts = self.gate_exps.shape().dims().first().copied().unwrap_or(64);
        if matches!(gpu_device, Device::Cuda(_)) && !matches!(self.compute_device, Device::Cuda(_))
        {
            self.gpu_hot_cache = Some(GpuHotExpertCache::new(
                gpu_device.clone(),
                num_experts,
                capacity,
            ));
            self.gpu_device = Some(gpu_device);
        }
    }

    /// Expert-parallel forward pass using fused SwiGLU and parallel matmul
    ///
    /// This processes all (token, expert) pairs in parallel:
    /// 1. Gathers only the active expert weights (reduces memory access)
    /// 2. Uses parallel indexed matmul for gate+up projections
    /// 3. Applies fused SwiGLU activation
    /// 4. Uses parallel indexed matmul for down projection
    /// 5. Scatter-adds weighted expert outputs
    ///
    /// hidden_states: [n_tokens, hidden_dim]
    /// expert_indices: [n_tokens, top_k] (u32)
    /// expert_weights: [n_tokens, top_k] (f32)
    /// Returns: [n_tokens, hidden_dim]
    /// Batched forward pass using indexed_moe_forward CUDA kernels
    ///
    /// This uses the optimized quantized matmul kernels that keep weights
    /// in quantized form, avoiding dequantization overhead.
    ///
    /// hidden_states: [n_tokens, hidden_dim]
    /// expert_indices: [n_tokens, top_k] (u32)
    /// expert_weights: [n_tokens, top_k] (f32)
    /// Returns: [n_tokens, hidden_dim]
    fn forward_batched(
        &self,
        hidden_states: &Tensor,
        expert_indices: &Tensor,
        expert_weights: &Tensor,
        top_k: usize,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (n_tokens, hidden_dim) = hidden_states.dims2()?;

        // Reshape hidden states for indexed_moe_forward: [n_tokens, 1, hidden_dim]
        // Note: We expect F32 input (model uses F32 activations to match llama.cpp)
        let hidden_expanded = hidden_states.unsqueeze(1)?;

        // Gate projection: [n_tokens, 1, hidden_dim] @ gate_exps -> [n_tokens, top_k, n_ff]
        let gate_out = self
            .gate_exps
            .indexed_moe_forward(&hidden_expanded, expert_indices)?;

        // Up projection: [n_tokens, 1, hidden_dim] @ up_exps -> [n_tokens, top_k, n_ff]
        let up_out = self
            .up_exps
            .indexed_moe_forward(&hidden_expanded, expert_indices)?;

        // Fused SwiGLU activation: silu(gate) * up
        let activated = crate::ops::fused_swiglu(&gate_out, &up_out)?;

        // For down projection, reshape activated for indexed_moe_forward
        // activated: [n_tokens, top_k, n_ff]
        // Need: [n_tokens * top_k, 1, n_ff] with indices [n_tokens * top_k, 1]
        let n_ff = activated.dims()[2];
        let activated_flat = activated.reshape((n_tokens * top_k, 1, n_ff))?;

        // Expand indices: [n_tokens, top_k] -> [n_tokens * top_k, 1]
        let indices_flat = expert_indices.reshape((n_tokens * top_k,))?;
        let indices_expanded = indices_flat.unsqueeze(1)?;

        // Down projection
        let down_out = self
            .down_exps
            .indexed_moe_forward(&activated_flat, &indices_expanded)?;
        // down_out: [n_tokens * top_k, 1, hidden_dim]
        let down_out = down_out.reshape((n_tokens, top_k, hidden_dim))?;

        // Weight and sum expert outputs
        let weights_expanded = expert_weights.unsqueeze(2)?;
        let weighted = down_out.broadcast_mul(&weights_expanded)?;
        weighted.sum(1)
    }

    /// Check if batched forward is supported (CUDA + supported quant type + ALL experts on GPU)
    fn supports_batched_forward(&self) -> bool {
        use candle::quantized::GgmlDType;

        // Transposed layout requires per-expert transpose, can't use indexed_moe_forward
        if self.transposed_layout {
            return false;
        }

        // Check if we're on CUDA
        if !matches!(self.compute_device, Device::Cuda(_)) {
            return false;
        }

        // Check if weights have supported dtype
        let gate_dtype = self.gate_exps.dtype();
        let supported = matches!(
            gate_dtype,
            GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
                | GgmlDType::Q8_0
        );

        // All expert projections must be on CUDA for indexed_moe_forward
        supported
            && matches!(self.gate_exps.device(), Device::Cuda(_))
            && matches!(self.up_exps.device(), Device::Cuda(_))
            && matches!(self.down_exps.device(), Device::Cuda(_))
    }

    fn forward_expert(&mut self, hidden_states: &Tensor, expert_idx: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let input_device = hidden_states.device();
        let training_mode = self.custom_gate_block_mults.is_some();

        // First, check GPU hot cache for non-training mode
        // This is the fast path: expert is already on GPU
        if !training_mode {
            if let Some(ref mut gpu_cache) = self.gpu_hot_cache {
                // Record usage and check if hot
                let is_hot = gpu_cache.record_usage(expert_idx);

                // Try to get from GPU cache
                if let Some(cached_mats) = gpu_cache.get(expert_idx) {
                    // Fast path: use GPU-cached expert
                    let gpu_device = gpu_cache.gpu_device.clone();
                    let hidden_on_gpu = if !hidden_states.device().same_device(&gpu_device) {
                        hidden_states.to_device(&gpu_device)?
                    } else {
                        hidden_states.clone()
                    };
                    let result =
                        self.forward_with_cached(&hidden_on_gpu, &cached_mats, &gpu_device)?;
                    return if !result.device().same_device(input_device) {
                        result.to_device(input_device)
                    } else {
                        Ok(result)
                    };
                }

                // If hot but not cached, promote to GPU
                if is_hot {
                    if let Ok(entry) = gpu_cache.build_gpu_entry(
                        expert_idx,
                        &self.gate_exps,
                        &self.up_exps,
                        &self.down_exps,
                    ) {
                        let gpu_device = gpu_cache.gpu_device.clone();
                        gpu_cache.insert(expert_idx, entry.clone());

                        // Use the newly cached entry
                        let hidden_on_gpu = if !hidden_states.device().same_device(&gpu_device) {
                            hidden_states.to_device(&gpu_device)?
                        } else {
                            hidden_states.clone()
                        };
                        let result =
                            self.forward_with_cached(&hidden_on_gpu, &entry, &gpu_device)?;
                        return if !result.device().same_device(input_device) {
                            result.to_device(input_device)
                        } else {
                            Ok(result)
                        };
                    }
                }
            }
        }

        // Fall back to original path
        let mats = if training_mode {
            // Training mode: use training_cache if available
            if let Some(cache) = &self.training_cache {
                if cache.enabled() {
                    if let Some(entry) = cache.entries.get(&expert_idx).cloned() {
                        if let Some(cache) = &mut self.training_cache {
                            cache.touch(expert_idx);
                        }
                        entry
                    } else {
                        let entry = self.build_scaled_mats(expert_idx)?;
                        if let Some(cache) = &mut self.training_cache {
                            cache.evict_if_needed();
                            cache.entries.insert(expert_idx, entry.clone());
                            cache.touch(expert_idx);
                        }
                        entry
                    }
                } else {
                    self.build_scaled_mats(expert_idx)?
                }
            } else {
                self.build_scaled_mats(expert_idx)?
            }
        } else if let Some(cache) = &mut self.cache {
            // Skip cache for transposed layout (MTP) since cache doesn't handle transpose
            if cache.enabled() && !self.transposed_layout {
                cache.get_or_prepare(expert_idx, &self.gate_exps, &self.up_exps, &self.down_exps)?
            } else {
                self.build_uncached_mats(expert_idx)?
            }
        } else {
            self.build_uncached_mats(expert_idx)?
        };

        self.forward_with_cached(hidden_states, &mats, input_device)
    }

    fn build_uncached_mats(&self, expert_idx: usize) -> Result<CachedMatmuls> {
        let gate_qtensor = self.gate_exps.slice_first_dim(expert_idx)?;
        let up_qtensor = self.up_exps.slice_first_dim(expert_idx)?;
        let down_qtensor = self.down_exps.slice_first_dim(expert_idx)?;

        if self.transposed_layout {
            // MTP expert weights are transposed: [in_dim, out_dim] instead of [out_dim, in_dim]
            // Need to dequantize, transpose, and re-quantize
            let gate_device = self.compute_device.clone();
            let up_device = self.compute_device.clone();
            let down_device = self.compute_device.clone();

            let gate_dequant = gate_qtensor.dequantize(&gate_device)?;
            let up_dequant = up_qtensor.dequantize(&up_device)?;
            let down_dequant = down_qtensor.dequantize(&down_device)?;

            // Transpose: [in_dim, out_dim] -> [out_dim, in_dim]
            let gate_transposed = gate_dequant.t()?.contiguous()?;
            let up_transposed = up_dequant.t()?.contiguous()?;
            let down_transposed = down_dequant.t()?.contiguous()?;

            // Re-quantize with original dtype
            let gate_requant = QTensor::quantize(&gate_transposed, gate_qtensor.dtype())?;
            let up_requant = QTensor::quantize(&up_transposed, up_qtensor.dtype())?;
            let down_requant = QTensor::quantize(&down_transposed, down_qtensor.dtype())?;

            Ok(CachedMatmuls {
                gate: Arc::new(QMatMul::from_weights(Arc::new(gate_requant))?),
                up: Arc::new(QMatMul::from_weights(Arc::new(up_requant))?),
                down: Arc::new(QMatMul::from_weights(Arc::new(down_requant))?),
                gate_device,
                up_device,
                down_device,
            })
        } else {
            let (gate_tensor, gate_device) = self.materialize_for_compute(gate_qtensor)?;
            let (up_tensor, up_device) = self.materialize_for_compute(up_qtensor)?;
            let (down_tensor, down_device) = self.materialize_for_compute(down_qtensor)?;

            Ok(CachedMatmuls {
                gate: Arc::new(QMatMul::from_weights(gate_tensor)?),
                up: Arc::new(QMatMul::from_weights(up_tensor)?),
                down: Arc::new(QMatMul::from_weights(down_tensor)?),
                gate_device,
                up_device,
                down_device,
            })
        }
    }

    fn materialize_for_compute(&self, qtensor: QTensor) -> Result<(Arc<QTensor>, Device)> {
        if self.compute_device.same_device(&qtensor.device())
            || matches!(self.compute_device, Device::Cpu)
        {
            let device = qtensor.device();
            return Ok((Arc::new(qtensor), device));
        }

        let data = qtensor.data()?;
        let storage = QStorage::from_data(data, &self.compute_device, qtensor.dtype())?;
        let moved = QTensor::new(storage, qtensor.shape().clone())?;
        Ok((Arc::new(moved), self.compute_device.clone()))
    }

    fn build_scaled_mats(&self, expert_idx: usize) -> Result<CachedMatmuls> {
        let gate_qtensor = self.gate_exps.slice_first_dim(expert_idx)?;
        let up_qtensor = self.up_exps.slice_first_dim(expert_idx)?;
        let down_qtensor = self.down_exps.slice_first_dim(expert_idx)?;

        let gate_src_device = self.gate_exps.device();
        let up_src_device = self.up_exps.device();
        let down_src_device = self.down_exps.device();

        let gate_mults = self
            .custom_gate_block_mults
            .as_ref()
            .map(|t| self.extract_expert_scales(t, expert_idx, &gate_qtensor))
            .transpose()?
            .ok_or_else(|| candle::Error::Msg("missing gate multipliers".to_string()))?;
        let up_mults = self
            .custom_up_block_mults
            .as_ref()
            .map(|t| self.extract_expert_scales(t, expert_idx, &up_qtensor))
            .transpose()?
            .ok_or_else(|| candle::Error::Msg("missing up multipliers".to_string()))?;
        let down_mults = self
            .custom_down_block_mults
            .as_ref()
            .map(|t| self.extract_expert_scales(t, expert_idx, &down_qtensor))
            .transpose()?
            .ok_or_else(|| candle::Error::Msg("missing down multipliers".to_string()))?;

        // Move multipliers to the source device for scale modification
        let gate_mults = if !gate_mults.device().same_device(&gate_src_device) {
            gate_mults.to_device(&gate_src_device)?
        } else {
            gate_mults
        };
        let up_mults = if !up_mults.device().same_device(&up_src_device) {
            up_mults.to_device(&up_src_device)?
        } else {
            up_mults
        };
        let down_mults = if !down_mults.device().same_device(&down_src_device) {
            down_mults.to_device(&down_src_device)?
        } else {
            down_mults
        };

        let gate_mults = gate_mults.to_dtype(DType::F32)?;
        let up_mults = up_mults.to_dtype(DType::F32)?;
        let down_mults = down_mults.to_dtype(DType::F32)?;

        // Apply scale modifications on source device
        let gate_modified = gate_qtensor.modify_block_scales(&gate_mults)?;
        let up_modified = up_qtensor.modify_block_scales(&up_mults)?;
        let down_modified = down_qtensor.modify_block_scales(&down_mults)?;

        if self.transposed_layout {
            // MTP expert weights are transposed: [in_dim, out_dim] instead of [out_dim, in_dim]
            // Need to dequantize, transpose, and re-quantize
            let gate_device = self.compute_device.clone();
            let up_device = self.compute_device.clone();
            let down_device = self.compute_device.clone();

            let gate_dequant = gate_modified.dequantize(&gate_device)?;
            let up_dequant = up_modified.dequantize(&up_device)?;
            let down_dequant = down_modified.dequantize(&down_device)?;

            // Transpose: [in_dim, out_dim] -> [out_dim, in_dim]
            let gate_transposed = gate_dequant.t()?.contiguous()?;
            let up_transposed = up_dequant.t()?.contiguous()?;
            let down_transposed = down_dequant.t()?.contiguous()?;

            // Re-quantize with original dtype
            let gate_requant = QTensor::quantize(&gate_transposed, gate_modified.dtype())?;
            let up_requant = QTensor::quantize(&up_transposed, up_modified.dtype())?;
            let down_requant = QTensor::quantize(&down_transposed, down_modified.dtype())?;

            Ok(CachedMatmuls {
                gate: Arc::new(QMatMul::from_weights(Arc::new(gate_requant))?),
                up: Arc::new(QMatMul::from_weights(Arc::new(up_requant))?),
                down: Arc::new(QMatMul::from_weights(Arc::new(down_requant))?),
                gate_device,
                up_device,
                down_device,
            })
        } else {
            // Materialize modified tensors to compute device
            let (gate_tensor, gate_device) = self.materialize_for_compute(gate_modified)?;
            let (up_tensor, up_device) = self.materialize_for_compute(up_modified)?;
            let (down_tensor, down_device) = self.materialize_for_compute(down_modified)?;

            Ok(CachedMatmuls {
                gate: Arc::new(QMatMul::from_weights(gate_tensor)?),
                up: Arc::new(QMatMul::from_weights(up_tensor)?),
                down: Arc::new(QMatMul::from_weights(down_tensor)?),
                gate_device,
                up_device,
                down_device,
            })
        }
    }

    fn forward_with_cached(
        &self,
        hidden_states: &Tensor,
        mats: &CachedMatmuls,
        input_device: &Device,
    ) -> Result<Tensor> {
        self.forward_expert_with_qmatmul(
            hidden_states,
            mats.gate.as_ref(),
            mats.up.as_ref(),
            mats.down.as_ref(),
            input_device,
            &mats.gate_device,
            &mats.up_device,
            &mats.down_device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_expert_with_qmatmul(
        &self,
        hidden_states: &Tensor,
        gate_qmatmul: &QMatMul,
        up_qmatmul: &QMatMul,
        down_qmatmul: &QMatMul,
        input_device: &Device,
        gate_device: &Device,
        up_device: &Device,
        down_device: &Device,
    ) -> Result<Tensor> {
        let xs_for_gate = if !input_device.same_device(gate_device) {
            hidden_states.to_device(gate_device)?
        } else {
            hidden_states.clone()
        };
        let gate = gate_qmatmul
            .forward(&xs_for_gate.contiguous()?)?
            .apply(&self.act_fn)?;

        let xs_for_up = if !input_device.same_device(up_device) {
            hidden_states.to_device(up_device)?
        } else {
            hidden_states.clone()
        };
        let up = up_qmatmul.forward(&xs_for_up.contiguous()?)?;

        let gate_on_up_device = if !gate.device().same_device(up_device) {
            gate.to_device(up_device)?
        } else {
            gate
        };
        let gated = (gate_on_up_device * up)?.contiguous()?;

        let gated_for_down = if !gated.device().same_device(down_device) {
            gated.to_device(down_device)?
        } else {
            gated
        };
        let output = down_qmatmul.forward(&gated_for_down)?;

        if !output.device().same_device(input_device) {
            output.to_device(input_device)
        } else {
            Ok(output)
        }
    }

    fn extract_expert_scales(
        &self,
        all_multipliers: &Tensor,
        expert_idx: usize,
        expert_qtensor: &QTensor,
    ) -> Result<Tensor> {
        let num_blocks = expert_qtensor.shape().elem_count() / expert_qtensor.dtype().block_size();
        let blocks_per_expert = num_blocks;
        let start_block = expert_idx * blocks_per_expert;
        all_multipliers
            .narrow(0, start_block, blocks_per_expert)?
            .contiguous()
    }

    #[allow(dead_code)]
    fn set_block_multipliers(
        &mut self,
        gate_mults: Tensor,
        up_mults: Tensor,
        down_mults: Tensor,
    ) -> Result<()> {
        self.custom_gate_block_mults = Some(gate_mults);
        self.custom_up_block_mults = Some(up_mults);
        self.custom_down_block_mults = Some(down_mults);
        // Clear training cache since mults have changed
        if let Some(cache) = &mut self.training_cache {
            cache.entries.clear();
            cache.lru.clear();
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn clear_block_multipliers(&mut self) {
        self.custom_gate_block_mults = None;
        self.custom_up_block_mults = None;
        self.custom_down_block_mults = None;
        // Clear training cache when exiting training mode
        if let Some(cache) = &mut self.training_cache {
            cache.entries.clear();
            cache.lru.clear();
        }
    }
}

/// Shared expert FFN (dense, not MoE)
#[derive(Debug)]
struct SharedExpert {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    shared_gate: Tensor, // 1D tensor [n_embd] for dot product to get scalar gate per token
    act_fn: Activation,
    custom_gate_scales: Option<Tensor>,
    custom_up_scales: Option<Tensor>,
    custom_down_scales: Option<Tensor>,
}

impl SharedExpert {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str, dtype: DType) -> Result<Option<Self>> {
    #[allow(clippy::too_many_arguments)]
        let up_proj = match gg.try_qmatmul(&format!("{}.ffn_up_shexp.weight", prefix))? {
            Some(p) => p,
            None => return Ok(None),
        };
        let gate_proj = gg.qmatmul(&format!("{}.ffn_gate_shexp.weight", prefix))?;
        let down_proj = gg.qmatmul(&format!("{}.ffn_down_shexp.weight", prefix))?;
        // ffn_gate_inp_shexp is a 1D tensor [n_embd] used for dot product
        let shared_gate = gg
            .tensor(&format!("{}.ffn_gate_inp_shexp.weight", prefix))
            .or_else(|_| gg.tensor(&format!("{}.ffn_gate_inp_shexp", prefix)))?
            .dequantize(&gg.device)?
            .to_dtype(dtype)?;

        Ok(Some(Self {
            gate_proj,
            up_proj,
            down_proj,
            shared_gate,
            act_fn: Activation::Silu,
            custom_gate_scales: None,
            custom_up_scales: None,
            custom_down_scales: None,
        }))
    }

    /// Create SharedExpert for MTP layers using GGUF-style weight names
    /// Uses blk.{idx}.mtp.ffn_*_shexp.weight naming convention
    fn new_mtp<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        dtype: DType,
    ) -> Result<Option<Self>> {
        // MTP uses GGUF naming: blk.{idx}.mtp.ffn_up_shexp.weight
        let up_proj = match gg.try_qmatmul(&format!("{}.ffn_up_shexp.weight", prefix))? {
            Some(p) => p,
            None => return Ok(None),
        };
        let gate_proj = gg.qmatmul(&format!("{}.ffn_gate_shexp.weight", prefix))?;
        let down_proj = gg.qmatmul(&format!("{}.ffn_down_shexp.weight", prefix))?;

        // MTP shared expert mixing gate (1D [hidden_size])
        // Try standard naming first, fall back to alternative naming for compatibility
        let shared_gate = gg
            .tensor(&format!("{}.ffn_gate_inp_shexp.weight", prefix))
            .or_else(|_| gg.tensor(&format!("{}.ffn_gate_inp_shexp", prefix)))
            .or_else(|_| gg.tensor(&format!("{}.shared_expert_gate.weight", prefix)))
            .or_else(|_| gg.tensor(&format!("{}.shared_expert_gate", prefix)))?
            .dequantize(&gg.device)?
            .to_dtype(dtype)?;

        Ok(Some(Self {
            gate_proj,
            up_proj,
            down_proj,
            shared_gate,
            act_fn: Activation::Silu,
            custom_gate_scales: None,
            custom_up_scales: None,
            custom_down_scales: None,
        }))
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = hidden_states.contiguous()?;

        // Gate projection with SiLU
        let gate_proj = if let Some(scales) = &self.custom_gate_scales {
            self.gate_proj.forward_with_scales(&hidden_states, scales)?
        } else {
            self.gate_proj.forward(&hidden_states)?
        };
        let gate = gate_proj.apply(&self.act_fn)?;
        // Up projection
        let up = if let Some(scales) = &self.custom_up_scales {
            self.up_proj.forward_with_scales(&hidden_states, scales)?
        } else {
            self.up_proj.forward(&hidden_states)?
        };
        // Combine gate and up
        let gated = (gate * up)?;
        // Down projection
        let ffn_out = if let Some(scales) = &self.custom_down_scales {
            self.down_proj
                .forward_with_scales(&gated.contiguous()?, scales)?
        } else {
            self.down_proj.forward(&gated.contiguous()?)?
        };

        // Apply shared expert gate (sigmoid)
        // shared_gate is [n_embd], hidden_states is [b, l, n_embd]
        // Compute dot product: sum(hidden_states * shared_gate, dim=-1) -> [b, l]
        // Ensure gate matches hidden_states dtype for mixed precision
        let gate = if self.shared_gate.dtype() != hidden_states.dtype() {
            self.shared_gate.to_dtype(hidden_states.dtype())?
        } else {
            self.shared_gate.clone()
        };
        let gate_value = hidden_states.broadcast_mul(&gate)?.sum(D::Minus1)?;
        let gate_sigmoid = candle_nn::ops::sigmoid(&gate_value)?;

        // Broadcast gate [b, l] to match ffn_out [b, l, hidden_dim]
        let gate_sigmoid = gate_sigmoid.unsqueeze(D::Minus1)?;
        // Ensure ffn_out matches gate dtype
        let ffn_out = if ffn_out.dtype() != gate_sigmoid.dtype() {
            ffn_out.to_dtype(gate_sigmoid.dtype())?
        } else {
            ffn_out
        };
        ffn_out.broadcast_mul(&gate_sigmoid)
    }
}

fn default_cache_capacity(num_experts_per_tok: usize) -> usize {
    let env_override = std::env::var("QWEN3NEXT_MOE_CACHE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    if let Some(value) = env_override {
        return value;
    }

    let scaled = num_experts_per_tok.saturating_mul(3);
    let buffered = num_experts_per_tok.saturating_add(4);
    let base = std::cmp::max(scaled, buffered);
    std::cmp::max(base, 8)
}

struct MoeBlock {
    experts: MoeExperts,
    shared_expert: Option<SharedExpert>,
    gate: QMatMul,
    num_experts: usize,
    num_experts_per_tok: usize,
    span: tracing::Span,
    custom_gate_scales: Option<Tensor>,
    /// Layer index for this MoE block
    layer_idx: usize,
}

impl std::fmt::Debug for MoeBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeBlock")
            .field("num_experts", &self.num_experts)
            .field("num_experts_per_tok", &self.num_experts_per_tok)
            .field("layer_idx", &self.layer_idx)
            .finish()
    }
}

impl MoeBlock {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        num_experts: usize,
        num_experts_per_tok: usize,
        dtype: DType,
        compute_device: &Device,
        cache_capacity: usize,
        layer_idx: usize,
    ) -> Result<Self> {
        let cache_capacity = if cache_capacity == 0 {
            default_cache_capacity(num_experts_per_tok)
        } else {
            cache_capacity
        };

        let experts = MoeExperts::new(gg, prefix, compute_device, cache_capacity)?;
        let shared_expert = SharedExpert::new(gg, prefix, dtype)?;
        let gate = gg.qmatmul(&format!("{}.ffn_gate_inp.weight", prefix))?;

        let span = tracing::span!(tracing::Level::TRACE, "moe-block");
        Ok(Self {
            experts,
            shared_expert,
            gate,
            num_experts,
            num_experts_per_tok,
            span,
            custom_gate_scales: None,
            layer_idx,
        })
    }

    /// Create MoeBlock for MTP layers using GGUF-style weight names
    /// Uses blk.{idx}.mtp.* naming convention
    #[allow(clippy::too_many_arguments)]
    fn new_mtp<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        num_experts: usize,
        num_experts_per_tok: usize,
        dtype: DType,
        compute_device: &Device,
        cache_capacity: usize,
        _layer_idx: usize,
    ) -> Result<Self> {
        let cache_capacity = if cache_capacity == 0 {
            default_cache_capacity(num_experts_per_tok)
        } else {
            cache_capacity
        };

        // MTP uses GGUF naming: blk.{idx}.mtp.*
        let experts = MoeExperts::new_mtp(gg, prefix, compute_device, cache_capacity)?;

        // MTP also has shared expert
        let shared_expert = SharedExpert::new_mtp(gg, prefix, dtype)?;

        // MTP router gate: blk.{idx}.mtp.ffn_gate_inp.weight
        // Fall back to shared_gate.weight for compatibility with different GGUF exports
        // GGML format: [hidden_size, num_experts] which is correct for projecting
        // from hidden_size to num_experts logits (no transpose needed)
        let gate = gg
            .try_qmatmul(&format!("{}.ffn_gate_inp.weight", prefix))?
            .or(gg.try_qmatmul(&format!("{}.shared_gate.weight", prefix))?)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "MTP router gate not found at {}.ffn_gate_inp.weight or {}.shared_gate.weight",
                    prefix, prefix
                ))
            })?;

        let span = tracing::span!(tracing::Level::TRACE, "mtp-moe-block");
        Ok(Self {
            experts,
            shared_expert,
            gate,
            num_experts,
            num_experts_per_tok,
            span,
            custom_gate_scales: None,
            layer_idx: 0,
        })
    }

    #[allow(dead_code)]
    fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let (output, _) = self.forward_with_stats(hidden_states)?;
        Ok(output)
    }

    fn forward_with_stats(&mut self, hidden_states: &Tensor) -> Result<(Tensor, (Tensor, Tensor))> {
        let _enter = self.span.enter();
        let (batch_size, seq_len, hidden_dim) = hidden_states.dims3()?;

        let hidden_flat = hidden_states
            .reshape((batch_size * seq_len, hidden_dim))?
            .contiguous()?;

        let router_logits = if let Some(scales) = &self.custom_gate_scales {
            self.gate
                .forward_with_scales(&hidden_flat, scales)?
                .reshape((batch_size, seq_len, self.num_experts))?
        } else {
            self.gate
                .forward(&hidden_flat)?
                .reshape((batch_size, seq_len, self.num_experts))?
        };

        // Clamp router logits to prevent softmax overflow (exp of large values -> inf -> NaN)
        // This is critical for training stability when scale modifications push logits to extremes
        let router_logits_clamped = router_logits.clamp(-100.0, 100.0)?;
        let routing_weights = candle_nn::ops::softmax(&router_logits_clamped, D::Minus1)?;
        let (top_weights, top_indices) = self.top_k(&routing_weights)?;

        // Route tokens to experts
        drop(_enter);
        let moe_output = self.apply_experts(hidden_states, &top_indices, &top_weights)?;

        // Add shared expert output if present
        let final_output = if let Some(ref shared_exp) = self.shared_expert {
            let shared_out = shared_exp.forward(hidden_states)?;
            (moe_output + shared_out)?
        } else {
            moe_output
        };

        Ok((final_output, (router_logits, top_indices)))
    }

    fn top_k(&self, routing_weights: &Tensor) -> Result<(Tensor, Tensor)> {
        // Use GPU-accelerated top-k selection when available
        // This avoids the per-token GPU->CPU synchronization bottleneck
        crate::ops::topk_moe_routing(routing_weights, self.num_experts_per_tok)
    }

    fn apply_experts(
        &mut self,
        hidden_states: &Tensor,
        expert_indices: &Tensor,
        expert_weights: &Tensor,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_dim) = hidden_states.dims3()?;
        let total_tokens = batch_size * seq_len;

        let hidden_flat = hidden_states
            .reshape((total_tokens, hidden_dim))?
            .contiguous()?;

        // Reshape indices and weights
        let indices_flat = expert_indices
            .reshape((total_tokens, self.num_experts_per_tok))?
            .contiguous()?;
        let weights_flat = expert_weights
            .reshape((total_tokens, self.num_experts_per_tok))?
            .contiguous()?;

        // Check if we can use the batched path
        // The batched path uses indexed_moe_forward CUDA kernels which:
        // 1. Avoid GPU->CPU sync for expert indices (major bottleneck)
        // 2. Process all experts in parallel on GPU
        // 3. Handle input quantization to Q8_1 internally
        //
        // Even for single tokens, batched is faster than sequential due to
        // avoiding the synchronization overhead.
        let training_mode = self.experts.custom_gate_block_mults.is_some();
        let supports_batched = self.experts.supports_batched_forward();

        // Check if all experts are on CPU (can use optimized CPU paths)
        let all_experts_on_cpu = !matches!(self.experts.gate_exps.device(), Device::Cuda(_))
            && !matches!(self.experts.up_exps.device(), Device::Cuda(_))
            && !matches!(self.experts.down_exps.device(), Device::Cuda(_));

        // Check if hidden states are on GPU (can use GPU-side grouping)
        let hidden_on_gpu = matches!(hidden_flat.device(), Device::Cuda(_));

        // Always use batched when supported (avoids GPU->CPU sync)
        let use_batched = !training_mode && supports_batched;

        // Use GPU-grouped path when experts on CPU but hidden states on GPU
        // This keeps expert indices on GPU for sorting, minimizing sync overhead
        let use_gpu_grouped = !training_mode
            && all_experts_on_cpu
            && hidden_on_gpu
            && !self.experts.transposed_layout;

        let result = if use_batched {
            // Use batched indexed_moe_forward CUDA kernels (fastest - all on GPU)
            self.experts.forward_batched(
                &hidden_flat,
                &indices_flat,
                &weights_flat,
                self.num_experts_per_tok,
            )?
        } else if use_gpu_grouped {
            // GPU-side grouping: sort tokens by expert on GPU, then process on CPU
            // This minimizes GPU->CPU sync by doing grouping on GPU
            crate::ops::process_moe_gpu_grouped(
                &hidden_flat,
                &indices_flat,
                &weights_flat,
                &self.experts.gate_exps,
                &self.experts.up_exps,
                &self.experts.down_exps,
                self.num_experts,
            )?
        } else {
            // Sequential processing with Rayon parallelization
            self.apply_experts_sequential(&hidden_flat, &indices_flat, &weights_flat)?
        };

        result.reshape((batch_size, seq_len, hidden_dim))
    }

    /// Sequential expert application (fallback for training mode or unsupported configs)
    fn apply_experts_sequential(
        &mut self,
        hidden_flat: &Tensor,
        indices_flat: &Tensor,
        weights_flat: &Tensor,
    ) -> Result<Tensor> {
        let (_total_tokens, hidden_dim) = hidden_flat.dims2()?;

        let indices_vec = indices_flat.to_vec2::<u32>()?;
        // Convert to F32 for to_vec2 extraction (weights need to be f32 for arithmetic)
        let weights_f32 = weights_flat.to_dtype(DType::F32)?;
        let weights_vec = weights_f32.to_vec2::<f32>()?;

        let mut top_x = vec![vec![]; self.num_experts];
        let mut selected_rws = vec![vec![]; self.num_experts];

        for (token_idx, (token_experts, token_weights)) in
            indices_vec.iter().zip(weights_vec.iter()).enumerate()
        {
            for (k, &expert_id) in token_experts.iter().enumerate() {
                let expert_idx = expert_id as usize;
                if expert_idx < self.num_experts {
                    top_x[expert_idx].push(token_idx as u32);
                    selected_rws[expert_idx].push(token_weights[k]);
                }
            }
        }

        let training_mode = self.experts.custom_gate_block_mults.is_some();
        let inference_prefetch = !training_mode
            && self
                .experts
                .cache
                .as_ref()
                .map(|c| c.enabled())
                .unwrap_or(false);
        let training_prefetch = training_mode
            && self
                .experts
                .training_cache
                .as_ref()
                .map(|c| c.enabled())
                .unwrap_or(false);

        if inference_prefetch {
            if let Some(cache) = self.experts.cache.as_mut() {
                for (idx, assignments) in top_x.iter().enumerate() {
                    if assignments.is_empty() {
                        continue;
                    }
                    let _ = cache.get_or_prepare(
                        idx,
                        &self.experts.gate_exps,
                        &self.experts.up_exps,
                        &self.experts.down_exps,
                    )?;
                }
            }
        } else if training_prefetch {
            // Prefetch for training mode - build scaled mats and cache them
            for (idx, assignments) in top_x.iter().enumerate() {
                if assignments.is_empty() {
                    continue;
                }
                // Check if already in cache
                let in_cache = self
                    .experts
                    .training_cache
                    .as_ref()
                    .map(|c| c.entries.contains_key(&idx))
                    .unwrap_or(false);
                if !in_cache {
                    let entry = self.experts.build_scaled_mats(idx)?;
                    if let Some(cache) = self.experts.training_cache.as_mut() {
                        cache.evict_if_needed();
                        cache.entries.insert(idx, entry);
                        cache.touch(idx);
                    }
                }
            }
        }

        let mut ys = hidden_flat.zeros_like()?;

        // Collect active experts for parallel processing
        let active_experts: Vec<usize> = (0..self.num_experts)
            .filter(|&idx| !top_x[idx].is_empty())
            .collect();

        // First, check if GPU hot cache is available and collect cached experts
        let (gpu_device_opt, cached_entries): (Option<Device>, Vec<(usize, CachedMatmuls)>) = {
            if let Some(ref mut gpu_cache) = self.experts.gpu_hot_cache {
                let gpu_device = gpu_cache.gpu_device.clone();
                let mut cached = Vec::new();

                for &expert_idx in &active_experts {
                    gpu_cache.record_usage(expert_idx);
                    if let Some(mats) = gpu_cache.get(expert_idx) {
                        cached.push((expert_idx, mats));
                    }
                }
                (Some(gpu_device), cached)
            } else {
                (None, Vec::new())
            }
        };

        // Process GPU-cached experts first (fast path)
        let mut gpu_processed = std::collections::HashSet::new();
        if let Some(ref gpu_device) = gpu_device_opt {
            for (expert_idx, cached_mats) in &cached_entries {
                let top_x_expert = &top_x[*expert_idx];
                let top_x_tensor = Tensor::new(top_x_expert.as_slice(), hidden_flat.device())?;
                let selected_rws_tensor =
                    Tensor::new(selected_rws[*expert_idx].as_slice(), hidden_flat.device())?
                        .reshape((top_x_expert.len(), 1))?
                        .to_dtype(hidden_flat.dtype())?;

                let current_state = hidden_flat
                    .index_select(&top_x_tensor, 0)?
                    .reshape((top_x_expert.len(), hidden_dim))?
                    .contiguous()?;

                // Move to GPU, compute, move back
                let hidden_on_gpu = current_state.to_device(gpu_device)?;
                let result =
                    self.experts
                        .forward_with_cached(&hidden_on_gpu, cached_mats, gpu_device)?;
                let result = result.to_device(hidden_flat.device())?;
                let weighted = result.broadcast_mul(&selected_rws_tensor)?;

                ys = ys.index_add(&top_x_tensor, &weighted, 0)?;
                gpu_processed.insert(*expert_idx);
            }
        }

        // Get remaining experts not processed by GPU cache
        let remaining_experts: Vec<usize> = active_experts
            .iter()
            .filter(|&&idx| !gpu_processed.contains(&idx))
            .copied()
            .collect();

        // Check if we should use parallel processing for remaining
        // Parallel is beneficial when we have multiple experts and ALL weights are on CPU
        // (the parallel path processes entire expert on CPU, so mixed GPU/CPU doesn't work)
        // Don't use parallel for transposed layout (MTP) as it doesn't handle the transpose
        let all_experts_on_cpu = !matches!(self.experts.gate_exps.device(), Device::Cuda(_))
            && !matches!(self.experts.up_exps.device(), Device::Cuda(_))
            && !matches!(self.experts.down_exps.device(), Device::Cuda(_));
        let use_parallel = remaining_experts.len() > 1
            && all_experts_on_cpu
            && !training_mode
            && !self.experts.transposed_layout;

        if use_parallel {
            // Parallel expert processing using Tokio's blocking pool
            let expert_assignments: Vec<(usize, Vec<u32>, Vec<f32>)> = remaining_experts
                .iter()
                .map(|&idx| (idx, top_x[idx].clone(), selected_rws[idx].clone()))
                .collect();

            let result = crate::tokio_experts::process_experts_tokio(
                hidden_flat,
                &self.experts.gate_exps,
                &self.experts.up_exps,
                &self.experts.down_exps,
                &expert_assignments,
                hidden_dim,
            )?;

            ys = ys.add(&result)?;
        } else {
            // Sequential processing for remaining experts
            for expert_idx in remaining_experts {
                let top_x_expert = &top_x[expert_idx];
                let top_x_tensor = Tensor::new(top_x_expert.as_slice(), hidden_flat.device())?;
                let selected_rws_tensor =
                    Tensor::new(selected_rws[expert_idx].as_slice(), hidden_flat.device())?
                        .reshape((top_x_expert.len(), 1))?
                        .to_dtype(hidden_flat.dtype())?;

                let current_state = hidden_flat
                    .index_select(&top_x_tensor, 0)?
                    .reshape((top_x_expert.len(), hidden_dim))?
                    .contiguous()?;

                let current_hidden_states =
                    self.experts.forward_expert(&current_state, expert_idx)?;
                let current_hidden_states =
                    current_hidden_states.broadcast_mul(&selected_rws_tensor)?;

                ys = ys.index_add(&top_x_tensor, &current_hidden_states, 0)?;
            }
        }

        Ok(ys)
    }
}

// ============================================================================
// Full Attention (for non-recurrent layers)
// ============================================================================

#[derive(Debug)]
struct FullAttention {
    /// Q projection (may be joint Q+Gate or just Q)
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    /// Number of query heads per KV head (for GQA). Only used without flash-attn.
    #[cfg_attr(feature = "flash-attn", allow(dead_code))]
    num_kv_groups: usize,
    head_dim: usize,
    /// Whether Q projection outputs Q+Gate (true) or just Q (false)
    gated_attention: bool,
    rotary_emb: Arc<RotaryEmbedding>,
    /// Pre-allocated KV cache for O(1) token insertion (used when not quantizing)
    preallocated_cache: Option<PreallocatedKvCache>,
    /// Pre-allocated quantized KV cache for O(1) token insertion with quantization
    quantized_cache: Option<PreallocatedQuantizedKvCache>,
    /// Legacy KV cache storage (no longer used - kept for potential future use)
    #[allow(dead_code)]
    kv_cache: Option<KvCacheStorage>,
    kv_cache_quantization: KvCacheQuantization,
    /// Maximum sequence length for pre-allocated cache
    max_seq_len: usize,
    #[allow(dead_code)]
    span: tracing::Span,
    custom_q_scales: Option<Tensor>,
    custom_k_scales: Option<Tensor>,
    custom_v_scales: Option<Tensor>,
    custom_o_scales: Option<Tensor>,
    // LoRA adapters for fine-tuning
    lora_q: Option<LoraAdapter>,
    lora_k: Option<LoraAdapter>,
    lora_v: Option<LoraAdapter>,
    lora_o: Option<LoraAdapter>,
    lora_sigma: f64,
}

impl FullAttention {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
        dtype: DType,
        kv_cache_quantization: KvCacheQuantization,
        max_seq_len: usize,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;

        let wq = gg.qmatmul(&format!("{}.attn_q.weight", prefix))?;
        let wk = gg.qmatmul(&format!("{}.attn_k.weight", prefix))?;
        let wv = gg.qmatmul(&format!("{}.attn_v.weight", prefix))?;
        let wo = gg.qmatmul(&format!("{}.attn_output.weight", prefix))?;

        // Detect if this is gated attention by checking Q projection output dimension
        // Gated: outputs num_heads * head_dim * 2 (Q + gate)
        // Non-gated: outputs num_heads * head_dim (just Q)
        let q_out_dim = wq
            .qtensor()
            .map(|qt| qt.shape().dims()[0])
            .unwrap_or(num_heads * head_dim);
        let expected_q_dim = num_heads * head_dim;
        let gated_attention = q_out_dim == expected_q_dim * 2;

        let q_norm = gg.rms_norm(
            &format!("{}.attn_q_norm.weight", prefix),
            rms_norm_eps,
            dtype,
        )?;
        let k_norm = gg.rms_norm(
            &format!("{}.attn_k_norm.weight", prefix),
            rms_norm_eps,
            dtype,
        )?;

        let span = tracing::span!(tracing::Level::TRACE, "full-attn");

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            gated_attention,
            rotary_emb,
            preallocated_cache: None, // Initialized lazily on first forward pass
            quantized_cache: None,    // Initialized lazily on first forward pass
            kv_cache: None,
            kv_cache_quantization,
            max_seq_len,
            span,
            custom_q_scales: None,
            custom_k_scales: None,
            custom_v_scales: None,
            custom_o_scales: None,
            lora_q: None,
            lora_k: None,
            lora_v: None,
            lora_o: None,
            lora_sigma: 1.0,
        })
    }

    /// Create FullAttention for MTP layers using GGUF-style weight names
    /// Uses blk.{idx}.mtp.attn_* naming convention
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn new_mtp<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
        dtype: DType,
        kv_cache_quantization: KvCacheQuantization,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;

        // MTP uses GGUF naming: blk.{idx}.mtp.attn_q.weight
        let wq = gg.qmatmul(&format!("{}.attn_q.weight", prefix))?;
        let wk = gg.qmatmul(&format!("{}.attn_k.weight", prefix))?;
        let wv = gg.qmatmul(&format!("{}.attn_v.weight", prefix))?;
        // Try attn_o first, fall back to attn_output for compatibility
        let wo = gg
            .try_qmatmul(&format!("{}.attn_o.weight", prefix))?
            .or(gg.try_qmatmul(&format!("{}.attn_output.weight", prefix))?)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "MTP attn output not found at {}.attn_o.weight or {}.attn_output.weight",
                    prefix, prefix
                ))
            })?;

        // MTP uses gated attention (Q outputs 2x: Q + gate)
        let gated_attention = true;

        let q_norm = gg.rms_norm(
            &format!("{}.attn_q_norm.weight", prefix),
            rms_norm_eps,
            dtype,
        )?;
        let k_norm = gg.rms_norm(
            &format!("{}.attn_k_norm.weight", prefix),
            rms_norm_eps,
            dtype,
        )?;

        let span = tracing::span!(tracing::Level::TRACE, "mtp-attn");

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            gated_attention,
            rotary_emb,
            preallocated_cache: None,
            quantized_cache: None,
            kv_cache: None,
            kv_cache_quantization,
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
            span,
            custom_q_scales: None,
            custom_k_scales: None,
            custom_v_scales: None,
            custom_o_scales: None,
            lora_q: None,
            lora_k: None,
            lora_v: None,
            lora_o: None,
            lora_sigma: 1.0,
        })
    }

    /// Create FullAttention for MTP layers with Gemma-style (1+weight) Q/K norms
    /// Uses blk.{idx}.mtp.attn_* naming convention
    #[allow(clippy::too_many_arguments)]
    fn new_mtp_gemma<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
        dtype: DType,
        kv_cache_quantization: KvCacheQuantization,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;

        // MTP uses GGUF naming: blk.{idx}.mtp.attn_q.weight
        let wq = gg.qmatmul(&format!("{}.attn_q.weight", prefix))?;
        let wk = gg.qmatmul(&format!("{}.attn_k.weight", prefix))?;
        let wv = gg.qmatmul(&format!("{}.attn_v.weight", prefix))?;
        // Try attn_o first, fall back to attn_output for compatibility
        let wo = gg
            .try_qmatmul(&format!("{}.attn_o.weight", prefix))?
            .or(gg.try_qmatmul(&format!("{}.attn_output.weight", prefix))?)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "MTP attn output not found at {}.attn_o.weight or {}.attn_output.weight",
                    prefix, prefix
                ))
            })?;

        // MTP uses gated attention (Q outputs 2x: Q + gate)
        let gated_attention = true;

        // Q/K normalization for MTP attention (standard RmsNorm)
        let q_norm = gg.rms_norm(
            &format!("{}.attn_q_norm.weight", prefix),
            rms_norm_eps,
            dtype,
        )?;
        let k_norm = gg.rms_norm(
            &format!("{}.attn_k_norm.weight", prefix),
            rms_norm_eps,
            dtype,
        )?;

        let span = tracing::span!(tracing::Level::TRACE, "mtp-attn");

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            gated_attention,
            rotary_emb,
            preallocated_cache: None,
            quantized_cache: None,
            kv_cache: None,
            kv_cache_quantization,
            max_seq_len: DEFAULT_MAX_SEQ_LEN,
            span,
            custom_q_scales: None,
            custom_k_scales: None,
            custom_v_scales: None,
            custom_o_scales: None,
            lora_q: None,
            lora_k: None,
            lora_v: None,
            lora_o: None,
            lora_sigma: 1.0,
        })
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let x_contiguous = x.contiguous()?;

        // Q projection - may output Q+gate or just Q (with optional scales and LoRA)
        let q_proj = self.wq.forward_with_scales_and_lora(
            &x_contiguous,
            self.custom_q_scales.as_ref(),
            self.lora_q.as_ref().map(|l| &l.a),
            self.lora_q.as_ref().map(|l| &l.b),
            self.lora_sigma,
        )?;

        // Handle gated vs non-gated attention
        let (q, gate) = if self.gated_attention {
            // Split Q and gate: Q proj outputs [b, l, num_heads * head_dim * 2]
            let q_full = q_proj.reshape((b, l, self.num_heads, self.head_dim * 2))?;
            let q = q_full.narrow(3, 0, self.head_dim)?;
            let gate = q_full.narrow(3, self.head_dim, self.head_dim)?;
            (q, Some(gate))
        } else {
            // Non-gated: Q proj outputs [b, l, num_heads * head_dim]
            let q = q_proj.reshape((b, l, self.num_heads, self.head_dim))?;
            (q, None)
        };

        // Apply Q normalization
        let q = self.q_norm.forward(&q.contiguous()?)?;

        // K, V projections (with optional scales and LoRA)
        let k = self.wk.forward_with_scales_and_lora(
            &x_contiguous,
            self.custom_k_scales.as_ref(),
            self.lora_k.as_ref().map(|l| &l.a),
            self.lora_k.as_ref().map(|l| &l.b),
            self.lora_sigma,
        )?;
        let v = self.wv.forward_with_scales_and_lora(
            &x_contiguous,
            self.custom_v_scales.as_ref(),
            self.lora_v.as_ref().map(|l| &l.a),
            self.lora_v.as_ref().map(|l| &l.b),
            self.lora_sigma,
        )?;

        // Reshape K for normalization
        let k = k.reshape((b, l, self.num_kv_heads, self.head_dim))?;
        let k = self.k_norm.forward(&k)?;

        let v = v.reshape((b, l, self.num_kv_heads, self.head_dim))?;

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Apply RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // KV-cache handling with optional quantization
        // For Q8_0/Q4K: use PreallocatedQuantizedKvCache for memory efficiency
        // For F16/BF16: use PreallocatedKvCache with appropriate dtype
        let (k, v) = if let Some(ggml_dtype) = self.kv_cache_quantization.to_ggml_dtype() {
            // Quantized KV cache mode (Q8_0 or Q4K)
            if self.quantized_cache.is_none() {
                // First call - initialize quantized cache
                self.quantized_cache = Some(PreallocatedQuantizedKvCache::new(
                    b,
                    self.num_kv_heads,
                    self.max_seq_len,
                    self.head_dim,
                    ggml_dtype,
                    k.device(),
                )?);
            }

            // Append new K/V and get dequantized result
            let cache = self.quantized_cache.as_mut().unwrap();
            let (cached_k, cached_v) = cache.append(&k, &v)?;

            // Convert back to model dtype for attention computation
            let cached_k = cached_k.to_dtype(q.dtype())?.contiguous()?;
            let cached_v = cached_v.to_dtype(q.dtype())?.contiguous()?;
            (cached_k, cached_v)
        } else {
            // Non-quantized mode (F16 or BF16)
            // Optionally convert K/V to the cache dtype before storing
            let cache_dtype = self.kv_cache_quantization.cache_dtype();
            let (k_for_cache, v_for_cache) = if let Some(dtype) = cache_dtype {
                if k.dtype() != dtype {
                    (
                        k.to_dtype(dtype)?.contiguous()?,
                        v.to_dtype(dtype)?.contiguous()?,
                    )
                } else {
                    (k, v)
                }
            } else {
                (k, v)
            };

            let (cached_k, cached_v) = if let Some(ref mut cache) = self.preallocated_cache {
                // Have existing cache, concatenate new K/V
                let prev_k = cache.k_cache.narrow(2, 0, cache.seq_len)?;
                let prev_v = cache.v_cache.narrow(2, 0, cache.seq_len)?;
                let new_k = Tensor::cat(&[&prev_k, &k_for_cache], 2)?.contiguous()?;
                let new_v = Tensor::cat(&[&prev_v, &v_for_cache], 2)?.contiguous()?;

                // Update cache with new values
                cache.k_cache = new_k.clone();
                cache.v_cache = new_v.clone();
                cache.seq_len = new_k.dim(2)?;
                cache.max_seq_len = cache.seq_len;

                (new_k, new_v)
            } else {
                // First call - initialize cache with current K/V
                self.preallocated_cache = Some(PreallocatedKvCache {
                    k_cache: k_for_cache.clone(),
                    v_cache: v_for_cache.clone(),
                    seq_len: k_for_cache.dim(2)?,
                    max_seq_len: k_for_cache.dim(2)?,
                    batch_size: b,
                });
                (k_for_cache, v_for_cache)
            };

            // Convert back to query dtype for attention if cache dtype differs
            if cached_k.dtype() != q.dtype() {
                (
                    cached_k.to_dtype(q.dtype())?.contiguous()?,
                    cached_v.to_dtype(q.dtype())?.contiguous()?,
                )
            } else {
                (cached_k, cached_v)
            }
        };

        // Compute attention - use flash attention when available, otherwise standard SDPA
        #[cfg(feature = "flash-attn")]
        let attn_output = {
            let input_dtype = q.dtype();

            // Flash attention only supports F16/BF16, convert if needed
            let (q_fa, k_fa, v_fa, needs_convert) = if input_dtype == DType::F32 {
                (
                    q.to_dtype(DType::F16)?,
                    k.to_dtype(DType::F16)?,
                    v.to_dtype(DType::F16)?,
                    true,
                )
            } else {
                (q, k, v, false)
            };

            // Flash attention expects (batch, seq, heads, head_dim) format
            // Current tensors are (batch, heads, seq, head_dim), so transpose back
            let q_fa = q_fa.transpose(1, 2)?.contiguous()?;
            let k_fa = k_fa.transpose(1, 2)?.contiguous()?;
            let v_fa = v_fa.transpose(1, 2)?.contiguous()?;

            let scale = 1f32 / (self.head_dim as f32).sqrt();
            // Flash attention handles GQA natively and applies causal mask internally
            let is_causal = attn_mask.is_some();
            let attn_out = candle_flash_attn::flash_attn(&q_fa, &k_fa, &v_fa, scale, is_causal)?;

            // Convert back to original dtype if we converted
            let attn_out = if needs_convert {
                attn_out.to_dtype(input_dtype)?
            } else {
                attn_out
            };

            // Output is (batch, seq, heads, head_dim), reshape to (batch, seq, hidden)
            attn_out
                .reshape((b, l, self.num_heads * self.head_dim))?
                .contiguous()?
        };

        #[cfg(not(feature = "flash-attn"))]
        let attn_output = {
            // Repeat KV heads for grouped-query attention
            let k = repeat_kv(k, self.num_kv_groups)?;
            let v = repeat_kv(v, self.num_kv_groups)?;

            // Compute attention
            let scale = 1f64 / (self.head_dim as f64).sqrt();
            let k_t = k.transpose(2, 3)?.contiguous()?;
            let attn_scores = (q.contiguous()?.matmul(&k_t)? * scale)?;
            let attn_scores = match attn_mask {
                None => attn_scores,
                Some(mask) => attn_scores.broadcast_add(mask)?,
            };

            // Clamp attention scores to prevent softmax overflow during training
            let attn_scores_clamped = attn_scores.clamp(-100.0, 100.0)?;
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_scores_clamped)?;
            let attn_output = attn_weights.matmul(&v.contiguous()?)?;

            // Reshape attention output
            attn_output
                .transpose(1, 2)?
                .reshape((b, l, self.num_heads * self.head_dim))?
                .contiguous()?
        };

        // Apply gating if present
        let final_output = if let Some(gate_tensor) = gate {
            // Gate shape is [b, l, num_heads, head_dim], need to flatten for matching
            let gate_flat = gate_tensor.reshape((b, l, self.num_heads * self.head_dim))?;
            let gate_sigmoid = candle_nn::ops::sigmoid(&gate_flat)?;
            (attn_output * gate_sigmoid)?
        } else {
            attn_output
        };

        // Output projection (with optional scales and LoRA)
        self.wo.forward_with_scales_and_lora(
            &final_output.contiguous()?,
            self.custom_o_scales.as_ref(),
            self.lora_o.as_ref().map(|l| &l.a),
            self.lora_o.as_ref().map(|l| &l.b),
            self.lora_sigma,
        )
    }

    /// Legacy method for storing K/V to cache with optional quantization.
    ///
    /// **Deprecated:** This is superseded by `PreallocatedQuantizedKvCache` which is now
    /// used directly in the forward pass when `kv_cache_quantization` is Q8_0 or Q4K.
    /// Kept for reference and potential future use cases.
    #[allow(dead_code)]
    fn store_kv_cache(&mut self, k: &Tensor, v: &Tensor, b: usize) -> Result<()> {
        if let Some(ggml_dtype) = self.kv_cache_quantization.to_ggml_dtype() {
            let block_size = self.kv_cache_quantization.block_size();

            let (k_for_quant, v_for_quant) = if !self.head_dim.is_multiple_of(block_size) {
                let padded_head_dim = self.head_dim.div_ceil(block_size) * block_size;
                let pad_size = padded_head_dim - self.head_dim;

                let k_padded = {
                    let zeros = Tensor::zeros(
                        (b, self.num_kv_heads, k.dim(2)?, pad_size),
                        k.dtype(),
                        k.device(),
                    )?;
                    Tensor::cat(&[k, &zeros], 3)?
                };
                let v_padded = {
                    let zeros = Tensor::zeros(
                        (b, self.num_kv_heads, v.dim(2)?, pad_size),
                        v.dtype(),
                        v.device(),
                    )?;
                    Tensor::cat(&[v, &zeros], 3)?
                };
                (k_padded, v_padded)
            } else {
                (k.clone(), v.clone())
            };

            let k_flat = k_for_quant.flatten_to(k_for_quant.rank() - 1)?;
            let v_flat = v_for_quant.flatten_to(v_for_quant.rank() - 1)?;
            let k_flat = k_flat.to_dtype(DType::F32)?;
            let v_flat = v_flat.to_dtype(DType::F32)?;

            let k_qtensor = QTensor::quantize(&k_flat, ggml_dtype)?;
            let v_qtensor = QTensor::quantize(&v_flat, ggml_dtype)?;

            let k_padded_shape = k_for_quant.dims().to_vec();
            let v_padded_shape = v_for_quant.dims().to_vec();

            self.kv_cache = Some(KvCacheStorage::Quantized(
                k_qtensor,
                v_qtensor,
                k_padded_shape,
                v_padded_shape,
            ));
        } else {
            self.kv_cache = Some(KvCacheStorage::Float(k.clone(), v.clone()));
        }
        Ok(())
    }

    fn clear_kv_cache(&mut self) {
        // Clear pre-allocated cache (reset head position to 0, keeps buffer allocated)
        if let Some(ref mut cache) = self.preallocated_cache {
            cache.clear();
        }
        // Clear pre-allocated quantized cache
        if let Some(ref mut cache) = self.quantized_cache {
            cache.clear();
        }
        // Clear legacy cache (shouldn't be used anymore)
        self.kv_cache = None;
    }

    /// Truncate the KV cache to a given sequence length.
    /// Used for speculative decoding when draft tokens are rejected.
    fn truncate_kv_cache(&mut self, new_len: usize) {
        if let Some(ref mut cache) = self.preallocated_cache {
            cache.truncate(new_len);
        }
        if let Some(ref mut cache) = self.quantized_cache {
            cache.truncate(new_len);
        }
    }

    /// Set the maximum sequence length for the pre-allocated KV cache.
    ///
    /// This should be called before the first forward pass. If called after
    /// the cache is initialized, it will clear the existing cache.
    #[allow(dead_code)]
    fn set_max_seq_len(&mut self, max_seq_len: usize) {
        if self.max_seq_len != max_seq_len {
            self.max_seq_len = max_seq_len;
            // Clear existing caches so they're re-created with new size
            self.preallocated_cache = None;
            self.quantized_cache = None;
        }
    }

    /// Save the current KV cache state for prefix caching.
    /// Returns a deep copy of the K/V tensors that can be restored later.
    fn save_kv_state(&self) -> PrefixCacheEntry {
        // Check quantized cache first (for Q8_0/Q4K modes)
        if let Some(ref cache) = self.quantized_cache {
            if cache.seq_len > 0 {
                // Dequantize the cache for storage
                if let Ok((k, v)) = cache.get_kv() {
                    if let (Ok(k_cloned), Ok(v_cloned)) = (k.contiguous(), v.contiguous()) {
                        return PrefixCacheEntry::FullAttention {
                            k_cache: k_cloned,
                            v_cache: v_cloned,
                            seq_len: cache.seq_len,
                        };
                    }
                }
            }
        }
        // Check non-quantized cache (for F16/BF16 modes)
        if let Some(ref cache) = self.preallocated_cache {
            if cache.seq_len > 0 {
                // Extract the valid portion of the cache and clone it
                if let (Ok(k), Ok(v)) = (
                    cache.k_cache.narrow(2, 0, cache.seq_len),
                    cache.v_cache.narrow(2, 0, cache.seq_len),
                ) {
                    if let (Ok(k_cloned), Ok(v_cloned)) = (k.contiguous(), v.contiguous()) {
                        return PrefixCacheEntry::FullAttention {
                            k_cache: k_cloned,
                            v_cache: v_cloned,
                            seq_len: cache.seq_len,
                        };
                    }
                }
            }
        }
        PrefixCacheEntry::Empty
    }

    /// Restore KV cache state from a prefix cache entry.
    /// This allows resuming generation from a previously cached prefix.
    fn restore_kv_state(&mut self, entry: &PrefixCacheEntry) -> Result<()> {
        match entry {
            PrefixCacheEntry::FullAttention {
                k_cache,
                v_cache,
                seq_len,
            } => {
                let b = k_cache.dim(0)?;
                let device = k_cache.device();

                // Check if we're using quantized caching
                if let Some(ggml_dtype) = self.kv_cache_quantization.to_ggml_dtype() {
                    // Initialize quantized cache if needed
                    if self.quantized_cache.is_none() {
                        self.quantized_cache = Some(PreallocatedQuantizedKvCache::new(
                            b,
                            self.num_kv_heads,
                            self.max_seq_len,
                            self.head_dim,
                            ggml_dtype,
                            device,
                        )?);
                    }

                    // Clear and restore by appending the saved K/V
                    if let Some(ref mut cache) = self.quantized_cache {
                        cache.clear();
                        cache.append(k_cache, v_cache)?;
                    }
                } else {
                    // Non-quantized mode: restore to preallocated cache
                    if self.preallocated_cache.is_none() {
                        let num_kv_heads = k_cache.dim(1)?;
                        let head_dim = k_cache.dim(3)?;
                        let dtype = k_cache.dtype();

                        self.preallocated_cache = Some(PreallocatedKvCache::new(
                            b,
                            num_kv_heads,
                            self.max_seq_len,
                            head_dim,
                            dtype,
                            device,
                        )?);
                    }

                    if let Some(ref mut cache) = self.preallocated_cache {
                        // Copy the saved K/V into the pre-allocated buffer
                        cache.k_cache = cache.k_cache.slice_scatter(k_cache, 2, 0)?;
                        cache.v_cache = cache.v_cache.slice_scatter(v_cache, 2, 0)?;
                        cache.seq_len = *seq_len;
                    }
                }
                Ok(())
            }
            PrefixCacheEntry::Empty => {
                self.clear_kv_cache();
                Ok(())
            }
            PrefixCacheEntry::LinearAttention { .. } => {
                // Wrong entry type for full attention - clear cache
                self.clear_kv_cache();
                Ok(())
            }
        }
    }

    /// Get the current cache sequence length.
    fn cache_seq_len(&self) -> usize {
        // Check quantized cache first
        if let Some(ref cache) = self.quantized_cache {
            return cache.seq_len;
        }
        // Fall back to non-quantized cache
        self.preallocated_cache
            .as_ref()
            .map(|c| c.seq_len)
            .unwrap_or(0)
    }

    /// Get the current K/V tensors from cache (for MTP state sharing).
    /// Returns (K, V) tensors in shape [batch, heads, seq, head_dim].
    fn get_cached_kv(&self) -> Option<(Tensor, Tensor)> {
        // Check quantized cache first (dequantize on access)
        if let Some(ref cache) = self.quantized_cache {
            if cache.seq_len > 0 {
                return cache.get_kv().ok();
            }
        }
        // Fall back to non-quantized cache
        self.preallocated_cache.as_ref().and_then(|cache| {
            if cache.seq_len > 0 {
                // Extract valid portion
                let k = cache.k_cache.narrow(2, 0, cache.seq_len).ok()?;
                let v = cache.v_cache.narrow(2, 0, cache.seq_len).ok()?;
                Some((k, v))
            } else {
                None
            }
        })
    }

    /// Forward pass using SHARED K/V cache from another layer (for MTP state sharing).
    ///
    /// This is used for "native" MTP where the prediction head reads directly from
    /// the main model's KV cache. MTP computes its own Q but uses the provided K/V.
    ///
    /// # Arguments
    /// - `x`: Input tensor [batch, seq, hidden]
    /// - `shared_k`: K tensor from main model's cache [batch, kv_heads, cache_seq, head_dim]
    /// - `shared_v`: V tensor from main model's cache [batch, kv_heads, cache_seq, head_dim]
    /// - `attn_mask`: Optional attention mask
    /// - `offset`: Position offset for RoPE
    fn forward_with_shared_kv(
        &mut self,
        x: &Tensor,
        shared_k: &Tensor,
        shared_v: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let x_contiguous = x.contiguous()?;

        // Q projection (MTP's own weights)
        let q_proj = self.wq.forward_with_scales_and_lora(
            &x_contiguous,
            self.custom_q_scales.as_ref(),
            self.lora_q.as_ref().map(|l| &l.a),
            self.lora_q.as_ref().map(|l| &l.b),
            self.lora_sigma,
        )?;

        // Handle gated vs non-gated attention
        let (q, gate) = if self.gated_attention {
            let q_full = q_proj.reshape((b, l, self.num_heads, self.head_dim * 2))?;
            let q = q_full.narrow(3, 0, self.head_dim)?;
            let gate = q_full.narrow(3, self.head_dim, self.head_dim)?;
            (q, Some(gate))
        } else {
            let q = q_proj.reshape((b, l, self.num_heads, self.head_dim))?;
            (q, None)
        };

        // Apply Q normalization
        let q = self.q_norm.forward(&q.contiguous()?)?;

        // Transpose Q to [batch, heads, seq, head_dim]
        let q = q.transpose(1, 2)?.contiguous()?;

        // Apply RoPE to Q only (K is already rotated in the shared cache)
        // We create a dummy K tensor to use the standard apply method, then discard it.
        // The offset should be the position of the NEW tokens (total cache len before MTP tokens).
        let dummy_k = Tensor::zeros_like(&q)?;
        let (q, _) = self.rotary_emb.apply(&q, &dummy_k, offset)?;

        // Use shared K, V directly (already have RoPE applied from main model)
        let k = shared_k.clone();
        let v = shared_v.clone();

        // Compute attention with shared K/V
        #[cfg(feature = "flash-attn")]
        let attn_output = {
            let input_dtype = q.dtype();

            // Flash attention only supports F16/BF16
            let (q_fa, k_fa, v_fa, needs_convert) = if input_dtype == DType::F32 {
                (
                    q.to_dtype(DType::F16)?,
                    k.to_dtype(DType::F16)?,
                    v.to_dtype(DType::F16)?,
                    true,
                )
            } else {
                (q.clone(), k, v, false)
            };

            // Flash attention expects (batch, seq, heads, head_dim)
            let q_fa = q_fa.transpose(1, 2)?.contiguous()?;
            let k_fa = k_fa.transpose(1, 2)?.contiguous()?;
            let v_fa = v_fa.transpose(1, 2)?.contiguous()?;

            let scale = 1f32 / (self.head_dim as f32).sqrt();
            let is_causal = attn_mask.is_some();
            let attn_out = candle_flash_attn::flash_attn(&q_fa, &k_fa, &v_fa, scale, is_causal)?;

            let attn_out = if needs_convert {
                attn_out.to_dtype(input_dtype)?
            } else {
                attn_out
            };

            attn_out
                .reshape((b, l, self.num_heads * self.head_dim))?
                .contiguous()?
        };

        #[cfg(not(feature = "flash-attn"))]
        let attn_output = {
            // Expand KV for GQA
            let k = crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
            let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let attn_weights = (q.matmul(&k.t()?)? * scale)?;

            let attn_weights = if let Some(mask) = attn_mask {
                let cache_len = k.dim(2)?;
                let mask_len = mask.dim(D::Minus1)?;
                if mask_len < cache_len {
                    let pad_len = cache_len - mask_len;
                    let zeros = Tensor::zeros((b, 1, l, pad_len), mask.dtype(), mask.device())?;
                    let extended_mask = Tensor::cat(&[&zeros, mask], D::Minus1)?;
                    (attn_weights + extended_mask)?
                } else {
                    (attn_weights + mask)?
                }
            } else {
                attn_weights
            };

            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_output = attn_weights.matmul(&v)?;

            attn_output
                .transpose(1, 2)?
                .reshape((b, l, self.num_heads * self.head_dim))?
                .contiguous()?
        };

        // Apply gate if gated attention
        let attn_output = if let Some(gate) = gate {
            let gate = gate
                .transpose(1, 2)?
                .reshape((b, l, self.num_heads * self.head_dim))?;
            let gate = candle_nn::ops::sigmoid(&gate)?;
            (&attn_output * &gate)?
        } else {
            attn_output
        };

        // Output projection
        let output = self.wo.forward_with_scales_and_lora(
            &attn_output,
            self.custom_o_scales.as_ref(),
            self.lora_o.as_ref().map(|l| &l.a),
            self.lora_o.as_ref().map(|l| &l.b),
            self.lora_sigma,
        )?;

        Ok(output)
    }
}

// ============================================================================
// Linear Attention (Gated Delta Net for recurrent layers)
// ============================================================================

/// Chunk size for delta net processing (matches llama.cpp CHUNK_SIZE)
#[allow(dead_code)]
const DELTA_NET_CHUNK_SIZE: usize = 64;

/// Cached masks for a given sequence length
#[derive(Debug)]
struct CachedMasks {
    seq_len: usize,
    causal_mask: Tensor,      // Strictly lower triangular
    causal_diag_mask: Tensor, // Lower triangular including diagonal
}

#[derive(Debug)]
struct LinearAttention {
    /// Input projection for Q, K, V, Z
    ssm_in: QMatMul,
    /// Input projection for beta and alpha
    ssm_beta_alpha: QMatMul,
    /// 1D convolution weights
    ssm_conv1d: Tensor,
    /// Output projection
    ssm_out: QMatMul,
    /// Gated normalization weights
    ssm_norm: RmsNorm,
    /// A log parameter (for decay)
    ssm_a: Tensor,
    /// dt bias
    ssm_dt: Tensor,
    /// Configuration
    d_inner: usize,
    d_state: usize,
    n_groups: usize,
    dt_rank: usize,
    conv_kernel_size: usize,
    #[allow(dead_code)]
    hidden_size: usize,
    rms_norm_eps: f64,
    /// Pre-computed scale factor: 1/sqrt(head_v_dim)
    scale: f64,
    /// Recurrent state
    recurrent_state: Option<RecurrentState>,
    /// Cached masks for reuse
    cached_masks: Option<CachedMasks>,
    #[allow(dead_code)]
    span: tracing::Span,
    custom_ssm_in_scales: Option<Tensor>,
    custom_ssm_ba_scales: Option<Tensor>,
    custom_ssm_out_scales: Option<Tensor>,
    // LoRA adapters for fine-tuning
    lora_ssm_in: Option<LoraAdapter>,
    lora_ssm_ba: Option<LoraAdapter>,
    lora_ssm_out: Option<LoraAdapter>,
    lora_sigma: f64,
}

impl LinearAttention {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        d_inner: usize,
        d_state: usize,
        n_groups: usize,
        dt_rank: usize,
        hidden_size: usize,
        rms_norm_eps: f64,
        dtype: DType,
    ) -> Result<Self> {
        // SSM tensors: llama.cpp uses .weight suffix for most, see tensor loading code
        let ssm_in = gg
            .qmatmul(&format!("{}.ssm_in.weight", prefix))
            .or_else(|_| gg.qmatmul(&format!("{}.ssm_in", prefix)))?;
        // ssm_beta_alpha is called ssm_ba in llama.cpp
        let ssm_beta_alpha = gg
            .qmatmul(&format!("{}.ssm_ba.weight", prefix))
            .or_else(|_| gg.qmatmul(&format!("{}.ssm_ba", prefix)))
            .or_else(|_| gg.qmatmul(&format!("{}.ssm_beta_alpha.weight", prefix)))
            .or_else(|_| gg.qmatmul(&format!("{}.ssm_beta_alpha", prefix)))?;
        let ssm_conv1d = gg
            .tensor(&format!("{}.ssm_conv1d.weight", prefix))
            .or_else(|_| gg.tensor(&format!("{}.ssm_conv1d", prefix)))?
            .dequantize(&gg.device)?
            .to_dtype(dtype)?;
        let ssm_out = gg
            .qmatmul(&format!("{}.ssm_out.weight", prefix))
            .or_else(|_| gg.qmatmul(&format!("{}.ssm_out", prefix)))?;
        let ssm_norm = gg
            .rms_norm(&format!("{}.ssm_norm.weight", prefix), rms_norm_eps, dtype)
            .or_else(|_| gg.rms_norm(&format!("{}.ssm_norm", prefix), rms_norm_eps, dtype))?;
        // ssm_a has no suffix in llama.cpp (LLM_TENSOR_SSM_A_NOSCAN)
        let ssm_a = gg
            .tensor(&format!("{}.ssm_a", prefix))?
            .dequantize(&gg.device)?
            .to_dtype(dtype)?;
        // ssm_dt uses "bias" suffix in llama.cpp, not "weight"
        let ssm_dt = gg
            .tensor(&format!("{}.ssm_dt.bias", prefix))
            .or_else(|_| gg.tensor(&format!("{}.ssm_dt", prefix)))
            .or_else(|_| gg.tensor(&format!("{}.ssm_dt.weight", prefix)))?
            .dequantize(&gg.device)?
            .to_dtype(dtype)?;

        // ssm_conv1d is stored as [channels, kernel_size] in GGUF
        // kernel_size is the smaller dimension (typically 4)
        let conv_kernel_size = ssm_conv1d.dim(1)?;

        let span = tracing::span!(tracing::Level::TRACE, "linear-attn");

        // Pre-compute scale factor: 1/sqrt(head_v_dim) where head_v_dim = d_inner / num_v_heads
        let scale = 1.0 / ((d_inner / dt_rank) as f64).sqrt();

        Ok(Self {
            ssm_in,
            ssm_beta_alpha,
            ssm_conv1d,
            ssm_out,
            ssm_norm,
            ssm_a,
            ssm_dt,
            d_inner,
            d_state,
            n_groups,
            dt_rank,
            conv_kernel_size,
            hidden_size,
            rms_norm_eps,
            scale,
            recurrent_state: None,
            cached_masks: None,
            span,
            custom_ssm_in_scales: None,
            custom_ssm_ba_scales: None,
            custom_ssm_out_scales: None,
            lora_ssm_in: None,
            lora_ssm_ba: None,
            lora_ssm_out: None,
            lora_sigma: 1.0,
        })
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let x_contiguous = x.contiguous()?;

        // Compute dimensions
        let head_k_dim = self.d_state;
        let num_k_heads = self.n_groups;
        let num_v_heads = self.dt_rank;
        let head_v_dim = self.d_inner / num_v_heads;

        if head_k_dim != head_v_dim {
            candle::bail!(
                "qwen3-next expects head_k_dim ({}) to match head_v_dim ({})",
                head_k_dim,
                head_v_dim
            );
        }

        // Input projections (with optional scales and LoRA)
        let mixed_qkvz = self.ssm_in.forward_with_scales_and_lora(
            &x_contiguous,
            self.custom_ssm_in_scales.as_ref(),
            self.lora_ssm_in.as_ref().map(|l| &l.a),
            self.lora_ssm_in.as_ref().map(|l| &l.b),
            self.lora_sigma,
        )?;
        let mixed_ba = self.ssm_beta_alpha.forward_with_scales_and_lora(
            &x_contiguous,
            self.custom_ssm_ba_scales.as_ref(),
            self.lora_ssm_ba.as_ref().map(|l| &l.a),
            self.lora_ssm_ba.as_ref().map(|l| &l.b),
            self.lora_sigma,
        )?;

        // Parse QKVZ
        let qkvz_new_dim = 2 * head_k_dim + 2 * head_v_dim * (num_v_heads / num_k_heads);
        let mixed_qkvz = mixed_qkvz.reshape((b, l, num_k_heads, qkvz_new_dim))?;

        // Parse beta/alpha
        let ba_new_dim = 2 * num_v_heads / num_k_heads;
        let mixed_ba = mixed_ba.reshape((b, l, num_k_heads, ba_new_dim))?;

        // Split beta and alpha
        let beta_size = num_v_heads / num_k_heads;
        let beta = mixed_ba.narrow(3, 0, beta_size)?;
        let alpha = mixed_ba.narrow(3, beta_size, beta_size)?;

        // Reshape beta and alpha to [b, l, num_v_heads] by flattening the last two dims
        let beta = beta.reshape((b, l, num_v_heads))?;
        let alpha = alpha.reshape((b, l, num_v_heads))?;

        // Compute gate from alpha
        // ssm_dt and ssm_a have shape [num_v_heads]
        let alpha_biased = alpha.broadcast_add(&self.ssm_dt)?;
        let alpha_softplus = softplus(&alpha_biased)?;
        // ssm_a contains negative values for decay (it's -exp(A_log))
        let gate = alpha_softplus.broadcast_mul(&self.ssm_a)?;

        // Split QKVZ
        let q = mixed_qkvz.narrow(3, 0, head_k_dim)?;
        let k = mixed_qkvz.narrow(3, head_k_dim, head_k_dim)?;
        let v_size = head_v_dim * num_v_heads / num_k_heads;
        let v = mixed_qkvz.narrow(3, 2 * head_k_dim, v_size)?;
        let z = mixed_qkvz.narrow(3, 2 * head_k_dim + v_size, v_size)?;

        // Flatten for convolution
        let q_flat = q.reshape((b, l, head_k_dim * num_k_heads))?.contiguous()?;
        let k_flat = k.reshape((b, l, head_k_dim * num_k_heads))?.contiguous()?;
        let v_flat = v.reshape((b, l, head_v_dim * num_v_heads))?.contiguous()?;

        // Concatenate for convolution
        let qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
        let qkv_cat = Tensor::cat(&[&q_flat, &k_flat, &v_flat], 2)?;
        let qkv_cat = qkv_cat.transpose(1, 2)?; // [b, qkv_dim, l]

        // Apply convolution with state handling
        let conv_out = self.apply_conv(&qkv_cat, b, l, qkv_dim)?;

        // Apply SiLU activation
        let conv_out = candle_nn::ops::silu(&conv_out)?;

        // Split back Q, K, V after conv
        let conv_out = conv_out.transpose(1, 2)?; // [b, l, qkv_dim]
        let q_conv = conv_out.narrow(2, 0, head_k_dim * num_k_heads)?;
        let k_conv = conv_out.narrow(2, head_k_dim * num_k_heads, head_k_dim * num_k_heads)?;
        let v_conv = conv_out.narrow(2, head_k_dim * num_k_heads * 2, head_v_dim * num_v_heads)?;

        // Reshape for attention
        let q_conv = q_conv.reshape((b, l, num_k_heads, head_k_dim))?;
        let k_conv = k_conv.reshape((b, l, num_k_heads, head_k_dim))?;
        let v_conv = v_conv.reshape((b, l, num_v_heads, head_v_dim))?;

        // beta and gate are already [b, l, num_v_heads] from earlier

        // Repeat Q and K if num_k_heads != num_v_heads (repeat_interleave)
        let (q_conv, k_conv) = if num_k_heads != num_v_heads {
            let repeat_factor = num_v_heads / num_k_heads;
            // Use repeat_interleave: each k_head is repeated repeat_factor times
            let q_repeated = q_conv
                .unsqueeze(3)?
                .expand((b, l, num_k_heads, repeat_factor, head_k_dim))?
                .contiguous()? // Need contiguous before reshape
                .reshape((b, l, num_v_heads, head_k_dim))?;
            let k_repeated = k_conv
                .unsqueeze(3)?
                .expand((b, l, num_k_heads, repeat_factor, head_k_dim))?
                .contiguous()? // Need contiguous before reshape
                .reshape((b, l, num_v_heads, head_k_dim))?;
            (q_repeated, k_repeated)
        } else {
            (q_conv, k_conv)
        };

        // Delta net computation
        // For small batches (MTP verification), use autoregressive loop for correctness
        // For large batches (prefill), use chunked for efficiency
        const AUTOREGRESSIVE_LOOP_THRESHOLD: usize = 16;

        let attn_out = if l == 1 {
            self.delta_net_autoregressive(&q_conv, &k_conv, &v_conv, &gate, &beta)?
        } else if l <= AUTOREGRESSIVE_LOOP_THRESHOLD {
            // Process tokens one at a time to ensure correct state updates
            self.delta_net_autoregressive_loop(&q_conv, &k_conv, &v_conv, &gate, &beta)?
        } else {
            self.delta_net_chunked(&q_conv, &k_conv, &v_conv, &gate, &beta)?
        };

        // Reshape for gated norm (need per-head normalization)
        // ssm_norm has shape [head_v_dim], so we need to reshape to [..., head_v_dim]
        let z = z.reshape((b, l, num_v_heads, head_v_dim))?;
        let attn_out = attn_out.reshape((b, l, num_v_heads, head_v_dim))?;

        // Apply gated normalization (per-head)
        let attn_out_norm = self.gated_norm(&attn_out, &z)?;

        // Reshape back to [b, l, num_v_heads * head_v_dim]
        let attn_out_norm = attn_out_norm.reshape((b, l, num_v_heads * head_v_dim))?;

        // Output projection (with optional scales and LoRA)
        let result = self.ssm_out.forward_with_scales_and_lora(
            &attn_out_norm.contiguous()?,
            self.custom_ssm_out_scales.as_ref(),
            self.lora_ssm_out.as_ref().map(|l| &l.a),
            self.lora_ssm_out.as_ref().map(|l| &l.b),
            self.lora_sigma,
        )?;
        Ok(result)
    }

    /// Forward with PARALLEL intermediate state materialization.
    ///
    /// This method computes outputs AND stores intermediate DeltaNet states
    /// at each position, enabling O(1) state slicing for verification.
    ///
    /// Follows the same structure as forward() but uses delta_net_single_chunk_with_states
    /// which materializes intermediate states during the parallel computation.
    fn forward_with_state_materialization(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        // Compute dimensions (same as forward())
        let head_k_dim = self.d_state;
        let num_k_heads = self.n_groups;
        let num_v_heads = self.dt_rank;
        let head_v_dim = self.d_inner / num_v_heads;

        // Flatten for projections
        let x_contiguous = x.contiguous()?.flatten(0, 1)?;

        // Input projections (with optional scales and LoRA)
        let mixed_qkvz = self.ssm_in.forward_with_scales_and_lora(
            &x_contiguous,
            self.custom_ssm_in_scales.as_ref(),
            self.lora_ssm_in.as_ref().map(|l| &l.a),
            self.lora_ssm_in.as_ref().map(|l| &l.b),
            self.lora_sigma,
        )?;
        let mixed_ba = self.ssm_beta_alpha.forward_with_scales_and_lora(
            &x_contiguous,
            self.custom_ssm_ba_scales.as_ref(),
            self.lora_ssm_ba.as_ref().map(|l| &l.a),
            self.lora_ssm_ba.as_ref().map(|l| &l.b),
            self.lora_sigma,
        )?;

        // Parse QKVZ
        let qkvz_new_dim = 2 * head_k_dim + 2 * head_v_dim * (num_v_heads / num_k_heads);
        let mixed_qkvz = mixed_qkvz.reshape((b, l, num_k_heads, qkvz_new_dim))?;

        // Parse beta/alpha
        let ba_new_dim = 2 * num_v_heads / num_k_heads;
        let mixed_ba = mixed_ba.reshape((b, l, num_k_heads, ba_new_dim))?;

        // Split beta and alpha
        let beta_size = num_v_heads / num_k_heads;
        let beta = mixed_ba.narrow(3, 0, beta_size)?;
        let alpha = mixed_ba.narrow(3, beta_size, beta_size)?;

        // Reshape beta and alpha to [b, l, num_v_heads]
        let beta = beta.reshape((b, l, num_v_heads))?;
        let alpha = alpha.reshape((b, l, num_v_heads))?;

        // Compute gate from alpha
        let alpha_biased = alpha.broadcast_add(&self.ssm_dt)?;
        let alpha_softplus = softplus(&alpha_biased)?;
        let gate = alpha_softplus.broadcast_mul(&self.ssm_a)?;

        // Split QKVZ
        let q = mixed_qkvz.narrow(3, 0, head_k_dim)?;
        let k = mixed_qkvz.narrow(3, head_k_dim, head_k_dim)?;
        let v_size = head_v_dim * num_v_heads / num_k_heads;
        let v = mixed_qkvz.narrow(3, 2 * head_k_dim, v_size)?;
        let z = mixed_qkvz.narrow(3, 2 * head_k_dim + v_size, v_size)?;

        // Flatten for convolution
        let q_flat = q.reshape((b, l, head_k_dim * num_k_heads))?.contiguous()?;
        let k_flat = k.reshape((b, l, head_k_dim * num_k_heads))?.contiguous()?;
        let v_flat = v.reshape((b, l, head_v_dim * num_v_heads))?.contiguous()?;

        // Concatenate for convolution
        let qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
        let qkv_cat = Tensor::cat(&[&q_flat, &k_flat, &v_flat], 2)?;
        let qkv_cat = qkv_cat.transpose(1, 2)?; // [b, qkv_dim, l]

        // Apply convolution with state handling
        let conv_out = self.apply_conv(&qkv_cat, b, l, qkv_dim)?;

        // Apply SiLU activation
        let conv_out = candle_nn::ops::silu(&conv_out)?;

        // Split back Q, K, V after conv
        let conv_out = conv_out.transpose(1, 2)?; // [b, l, qkv_dim]
        let q_conv = conv_out.narrow(2, 0, head_k_dim * num_k_heads)?;
        let k_conv = conv_out.narrow(2, head_k_dim * num_k_heads, head_k_dim * num_k_heads)?;
        let v_conv = conv_out.narrow(2, head_k_dim * num_k_heads * 2, head_v_dim * num_v_heads)?;

        // Reshape for attention
        let q_conv = q_conv.reshape((b, l, num_k_heads, head_k_dim))?;
        let k_conv = k_conv.reshape((b, l, num_k_heads, head_k_dim))?;
        let v_conv = v_conv.reshape((b, l, num_v_heads, head_v_dim))?;

        // Repeat Q and K if num_k_heads != num_v_heads
        let (q_conv, k_conv) = if num_k_heads != num_v_heads {
            let repeat_factor = num_v_heads / num_k_heads;
            let q_repeated = q_conv
                .unsqueeze(3)?
                .expand((b, l, num_k_heads, repeat_factor, head_k_dim))?
                .contiguous()?
                .reshape((b, l, num_v_heads, head_k_dim))?;
            let k_repeated = k_conv
                .unsqueeze(3)?
                .expand((b, l, num_k_heads, repeat_factor, head_k_dim))?
                .contiguous()?
                .reshape((b, l, num_v_heads, head_k_dim))?;
            (q_repeated, k_repeated)
        } else {
            (q_conv, k_conv)
        };

        // Use the STATE-MATERIALIZING kernel for verification
        // This computes outputs AND stores intermediate states in O(1) parallel time
        let attn_out =
            self.delta_net_single_chunk_with_states(&q_conv, &k_conv, &v_conv, &gate, &beta, l)?;

        // Reshape for gated norm
        let z = z.reshape((b, l, num_v_heads, head_v_dim))?;
        let attn_out = attn_out.reshape((b, l, num_v_heads, head_v_dim))?;

        // Apply gated normalization
        let attn_out_norm = self.gated_norm(&attn_out, &z)?;
        let attn_out_norm = attn_out_norm.reshape((b, l, num_v_heads * head_v_dim))?;

        // Output projection
        let result = self.ssm_out.forward_with_scales_and_lora(
            &attn_out_norm.contiguous()?,
            self.custom_ssm_out_scales.as_ref(),
            self.lora_ssm_out.as_ref().map(|l| &l.a),
            self.lora_ssm_out.as_ref().map(|l| &l.b),
            self.lora_sigma,
        )?;
        Ok(result)
    }

    fn apply_conv(&mut self, qkv: &Tensor, b: usize, l: usize, dim: usize) -> Result<Tensor> {
        // Handle convolution state
        let conv_state_len = self.conv_kernel_size - 1;

        let (conv_input, new_conv_state) = if let Some(ref state) = self.recurrent_state {
            // Concatenate previous state with current input
            let prev_state = &state.conv_state;
            let input = Tensor::cat(&[prev_state, qkv], 2)?;

            // Extract new state (last conv_state_len elements)
            let new_state = if l >= conv_state_len {
                qkv.narrow(2, l - conv_state_len, conv_state_len)?
            } else {
                // Need to combine old state with new input
                let old_part = prev_state.narrow(2, l, conv_state_len - l)?;
                Tensor::cat(&[&old_part, qkv], 2)?
            };
            (input, new_state)
        } else {
            // No previous state - pad with zeros
            let zeros = Tensor::zeros((b, dim, conv_state_len), qkv.dtype(), qkv.device())?;
            let input = Tensor::cat(&[&zeros, qkv], 2)?;

            // Extract new state
            let new_state = if l >= conv_state_len {
                qkv.narrow(2, l - conv_state_len, conv_state_len)?
            } else {
                let old_zeros =
                    Tensor::zeros((b, dim, conv_state_len - l), qkv.dtype(), qkv.device())?;
                Tensor::cat(&[&old_zeros, qkv], 2)?
            };
            (input, new_state)
        };

        // Apply 1D convolution (ssm_conv style)
        // conv1d weight is [kernel_size, channels]
        // We need to do depthwise conv
        let conv_out = self.depthwise_conv1d(&conv_input, l)?;

        // Update state - don't initialize SSM state here, let delta_net functions handle it
        // Just update conv_state. IMPORTANT: preserve backup tensors for checkpoint/restore!
        if let Some(ref mut state) = self.recurrent_state {
            // Just update the conv_state in place, preserving ssm_state and backups
            state.conv_state = new_conv_state;
        } else {
            // Initialize SSM state with correct shape: [b, num_heads, state_dim, state_dim]
            let num_heads = self.dt_rank; // num_v_heads
            let state_dim = self.d_inner / self.dt_rank;
            let ssm_state = Tensor::zeros(
                (b, num_heads, state_dim, state_dim),
                qkv.dtype(),
                qkv.device(),
            )?;
            self.recurrent_state = Some(RecurrentState::new(ssm_state, new_conv_state));
        }

        Ok(conv_out)
    }

    fn depthwise_conv1d(&self, input: &Tensor, _output_len: usize) -> Result<Tensor> {
        // input: [b, channels, input_len]
        // ssm_conv1d: [channels, kernel_size] (GGUF stores it this way)
        // Use the optimized CUDA kernel implementation
        crate::ops::depthwise_conv1d(input, &self.ssm_conv1d)
    }
    fn delta_net_autoregressive(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        gate: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor> {
        // Autoregressive (single token) delta net computation using fused CUDA kernel
        // Input shapes: q, k: [b, 1, num_heads, head_k_dim], v: [b, 1, num_heads, head_v_dim]
        // gate, beta: [b, 1, num_heads]
        let (b, _l, num_heads, head_k_dim) = q.dims4()?;
        let (_, _, _, head_v_dim) = v.dims4()?;

        if head_k_dim != head_v_dim {
            candle::bail!(
                "delta_net_autoregressive expects matching head dims ({} vs {})",
                head_k_dim,
                head_v_dim
            );
        }

        // Squeeze the sequence dimension: [b, num_heads, head_dim]
        let q_sq = q.squeeze(1)?.contiguous()?;
        let k_sq = k.squeeze(1)?.contiguous()?;
        let v_sq = v.squeeze(1)?.contiguous()?;
        let gate_sq = gate.squeeze(1)?.contiguous()?; // [b, num_heads]
        let beta_sq = beta.squeeze(1)?.contiguous()?; // [b, num_heads] (pre-sigmoid)

        // Get or initialize state: [b, num_heads, head_v_dim, head_v_dim]
        let state = if let Some(ref rs) = self.recurrent_state {
            rs.ssm_state.contiguous()?
        } else {
            Tensor::zeros(
                (b, num_heads, head_v_dim, head_v_dim),
                q.dtype(),
                q.device(),
            )?
        };

        // Use fused CUDA kernel for autoregressive step
        let (output, new_state) = crate::ops::delta_net_autoregressive_step(
            &q_sq,
            &k_sq,
            &v_sq,
            &gate_sq,
            &beta_sq,
            &state,
            self.scale as f32,
            self.rms_norm_eps as f32,
        )?;

        // Store updated state and update gate offset
        if let Some(ref mut rs) = self.recurrent_state {
            rs.ssm_state = new_state;
            // Update gate offset: add current gate to accumulated offset
            rs.gate_cumsum_offset = Some(if let Some(ref offset) = rs.gate_cumsum_offset {
                (offset + &gate_sq)?
            } else {
                gate_sq.clone()
            });
        } else {
            let conv_state = Tensor::zeros(
                (
                    b,
                    self.d_inner + 2 * self.n_groups * self.d_state,
                    self.conv_kernel_size - 1,
                ),
                q.dtype(),
                q.device(),
            )?;
            let mut rs = RecurrentState::new(new_state, conv_state);
            rs.gate_cumsum_offset = Some(gate_sq.clone());
            self.recurrent_state = Some(rs);
        }

        // Reshape output: [b, num_heads, head_v_dim] -> [b, 1, num_heads, head_v_dim]
        output.unsqueeze(1)
    }

    /// Process multiple tokens using autoregressive loop.
    /// This ensures correct state updates for MTP verification.
    fn delta_net_autoregressive_loop(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        gate: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor> {
        let (_b, l, _num_heads, _head_dim) = q.dims4()?;
        let (_, _, _, _head_v_dim) = v.dims4()?;

        // Process each token sequentially
        let mut outputs = Vec::with_capacity(l);

        for t in 0..l {
            // Extract single token for this step
            let q_t = q.narrow(1, t, 1)?;
            let k_t = k.narrow(1, t, 1)?;
            let v_t = v.narrow(1, t, 1)?;
            let gate_t = gate.narrow(1, t, 1)?;
            let beta_t = beta.narrow(1, t, 1)?;

            // Run autoregressive step (updates state)
            let out_t = self.delta_net_autoregressive(&q_t, &k_t, &v_t, &gate_t, &beta_t)?;
            outputs.push(out_t);
        }

        // Concatenate outputs along sequence dimension
        let result = Tensor::cat(&outputs, 1)?;
        Ok(result)
    }

    fn delta_net_chunked(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        gate: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor> {
        let (b, l, num_heads, head_k_dim) = q.dims4()?;
        let (_, _, _, head_v_dim) = v.dims4()?;

        if head_k_dim != head_v_dim {
            candle::bail!(
                "delta_net_chunked expects matching head dims ({} vs {})",
                head_k_dim,
                head_v_dim
            );
        }

        // For short sequences, use non-chunked path (avoids chunking overhead)
        if l <= DELTA_NET_CHUNK_SIZE {
            return self.delta_net_single_chunk(q, k, v, gate, beta, l);
        }

        // === CHUNKED PATH for longer sequences ===
        // Following llama.cpp: pad to multiples of CHUNK_SIZE and process chunk by chunk

        let chunk_size = DELTA_NET_CHUNK_SIZE;
        let pad = (chunk_size - l % chunk_size) % chunk_size;
        let padded_len = l + pad;
        let n_chunks = padded_len / chunk_size;

        // L2 normalize Q and K, with Q also scaled by 1/sqrt(head_dim)
        let q = crate::ops::l2_normalize_scale(q, self.scale as f32, self.rms_norm_eps)?;
        let k = l2_normalize(k, self.rms_norm_eps)?;

        // Apply sigmoid to beta: [b, l, num_heads]
        let beta_sig = candle_nn::ops::sigmoid(beta)?;

        // Transpose to [b, num_heads, l, head_dim] for batch matmul
        let q_t = q.transpose(1, 2)?.contiguous()?;
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;

        // beta_sig: [b, l, num_heads] -> [b, num_heads, l, 1]
        let beta_expanded = beta_sig.transpose(1, 2)?.unsqueeze(3)?;

        // v_beta = v * beta, k_beta = k * beta
        let v_beta = v_t.broadcast_mul(&beta_expanded)?;
        let k_beta = k_t.broadcast_mul(&beta_expanded)?;

        // Pad tensors to padded_len if needed
        // Note: gate operations use F32 for cumsum/exp to avoid overflow; model uses F32 activations
        let (q_t, k_t, v_t, v_beta, k_beta, gate_t) = if pad > 0 {
            let q_t = pad_tensor(&q_t, 2, pad)?;
            let k_t = pad_tensor(&k_t, 2, pad)?;
            let v_t = pad_tensor(&v_t, 2, pad)?;
            let v_beta = pad_tensor(&v_beta, 2, pad)?;
            let k_beta = pad_tensor(&k_beta, 2, pad)?;
            let gate_t = gate.transpose(1, 2)?.contiguous()?;
            let gate_t = pad_tensor(&gate_t, 2, pad)?;
            (q_t, k_t, v_t, v_beta, k_beta, gate_t)
        } else {
            let gate_t = gate.transpose(1, 2)?.contiguous()?;
            (q_t, k_t, v_t, v_beta, k_beta, gate_t)
        };

        // Create chunk-size masks in F32 for decay computation (matches model dtype)
        let chunk_causal_mask = create_causal_mask(chunk_size, q.device(), DType::F32)?;
        let chunk_identity = create_identity_mask(chunk_size, q.device(), DType::F32)?;
        let chunk_causal_diag_mask = (&chunk_causal_mask + &chunk_identity)?;

        // Initialize or get state and gate offset
        let (mut state, mut running_gate_offset) = if let Some(ref rs) = self.recurrent_state {
            let offset = rs.gate_cumsum_offset.clone();
            (rs.ssm_state.clone(), offset)
        } else {
            (
                Tensor::zeros(
                    (b, num_heads, head_v_dim, head_v_dim),
                    q.dtype(),
                    q.device(),
                )?,
                None,
            )
        };

        // Cumulative sum of gate for decay computation (per chunk)
        // NOTE: This uses per-chunk cumsum to avoid exp() overflow with very long sequences.
        // The chunk-local cumsum is correct for within-chunk decay computation.
        // For prefix cache continuation, the state already encodes the prefix decay.
        // gate_t is F32 to avoid overflow in cumsum/exp
        let gate_cumsum = {
            let gate_reshaped = gate_t.reshape((b, num_heads, n_chunks, chunk_size))?;
            let gate_flat = gate_reshaped.reshape((b * num_heads * n_chunks, chunk_size))?;
            let gate_cumsum_flat = gate_flat.cumsum(1)?;
            gate_cumsum_flat
                .reshape((b, num_heads, n_chunks, chunk_size))?
                .reshape((b, num_heads, padded_len))?
        };

        // Process chunks and collect outputs
        let mut chunk_outputs: Vec<Tensor> = Vec::with_capacity(n_chunks);

        for chunk_idx in 0..n_chunks {
            let start = chunk_idx * chunk_size;

            // Extract chunk tensors
            let q_chunk = q_t.narrow(2, start, chunk_size)?;
            let k_chunk = k_t.narrow(2, start, chunk_size)?;
            let _v_chunk = v_t.narrow(2, start, chunk_size)?;
            let v_beta_chunk = v_beta.narrow(2, start, chunk_size)?;
            let k_beta_chunk = k_beta.narrow(2, start, chunk_size)?;
            let gate_cumsum_chunk = gate_cumsum.narrow(2, start, chunk_size)?;

            // Compute decay mask in F32, then convert to input dtype for matmul
            let gate_i = gate_cumsum_chunk.unsqueeze(3)?;
            let gate_j = gate_cumsum_chunk.unsqueeze(2)?;
            let decay_diff = gate_i.broadcast_sub(&gate_j)?;
            let decay_mask = decay_diff.broadcast_mul(&chunk_causal_diag_mask)?;
            let decay_mask = decay_mask.exp()?;
            let decay_mask = decay_mask.broadcast_mul(&chunk_causal_diag_mask)?;

            // === Intra-chunk attention ===
            // kmulkbeta = k_beta @ k^T
            let kmulkbeta = k_beta_chunk.matmul(&k_chunk.transpose(2, 3)?)?;
            let k_decay = kmulkbeta.broadcast_mul(&decay_mask)?;
            let attn1 = k_decay.broadcast_mul(&chunk_causal_mask)?.neg()?;

            // Triangular solve
            let attn1_solved = solve_lower_triangular(&attn1, &chunk_causal_mask)?;

            // value = attn1_solved @ v_beta
            let value = attn1_solved.matmul(&v_beta_chunk)?;

            // For state interaction, use LOCAL cumsum (not total cumsum with offset).
            // The state already encodes decay from all previous chunks, so we only need
            // the decay from the START of this chunk to each position within the chunk.
            // Adding the offset would double-count the decay from previous chunks,
            // causing the state contribution to be incorrectly scaled down (context forgetting).
            // Reference: llama.cpp build_delta_net_chunking uses local cumsum for gexp_chunk.

            // k_cumdecay = attn1_solved @ (k_beta * exp(gate_cumsum))
            let gate_cumsum_exp = gate_cumsum_chunk.unsqueeze(3)?.exp()?;
            let kbeta_gexp = k_beta_chunk.broadcast_mul(&gate_cumsum_exp)?;
            let k_cumdecay = attn1_solved.matmul(&kbeta_gexp)?;

            // q @ k^T attention (strictly lower triangular)
            let attn2 = q_chunk.matmul(&k_chunk.transpose(2, 3)?)?;
            let attn2 = attn2.broadcast_mul(&decay_mask)?;
            let attn2 = attn2.broadcast_mul(&chunk_causal_diag_mask)?;

            // === Inter-chunk (state) contribution ===
            // v_prime[t, i] = sum_j(k_cumdecay[t, j] * state[i, j])
            // In llama.cpp: v_prime = state^T @ k_cumdecay (via ggml_mul_mat(state_t, k_cumdecay))
            // We need to use transposed state for correct matrix-vector multiply
            let state_t = state.transpose(2, 3)?;
            let v_prime = k_cumdecay.matmul(&state_t)?;

            // v_new = value - v_prime
            let v_new = (value - v_prime)?;

            // attn_inter[t, i] = sum_j(q[t, j] * g[t] * state[i, j])
            // In llama.cpp: attn_inter = state^T @ q_g_exp
            let q_g_exp = q_chunk.broadcast_mul(&gate_cumsum_exp)?;
            let attn_inter = q_g_exp.matmul(&state_t)?;

            // core_attn_out = attn_inter + attn2 @ v_new
            let v_attn = attn2.matmul(&v_new)?;
            let core_attn_out_chunk = (attn_inter + v_attn)?;

            chunk_outputs.push(core_attn_out_chunk);

            // === Update state for next chunk ===
            let gate_last = gate_cumsum_chunk.narrow(2, chunk_size - 1, 1)?;
            let g_diff = gate_last.broadcast_sub(&gate_cumsum_chunk)?;
            let g_diff_exp = g_diff.unsqueeze(3)?.exp()?;

            // kgdmulvnew[i, j] = sum_t(v_new[t, i] * k[t, j] * g_diff[t])
            // v_new^T @ key_gdiff: [b,h,d,seq] @ [b,h,seq,d] = [b,h,d,d]
            let key_gdiff = k_chunk.broadcast_mul(&g_diff_exp)?;
            let kgdmulvnew = v_new.transpose(2, 3)?.matmul(&key_gdiff)?;

            let g_last_exp = gate_last.unsqueeze(3)?.exp()?;
            state = (state.broadcast_mul(&g_last_exp)? + kgdmulvnew)?;

            // Update running gate offset: add this chunk's total gate
            let chunk_gate_sum = gate_last.squeeze(2)?; // [b, num_heads]
            running_gate_offset = Some(if let Some(offset) = running_gate_offset {
                (&offset + &chunk_gate_sum)?
            } else {
                chunk_gate_sum
            });
        }

        // Concatenate all chunk outputs
        let core_attn_out = Tensor::cat(&chunk_outputs, 2)?;

        // Remove padding if we added any
        let core_attn_out = if pad > 0 {
            core_attn_out.narrow(2, 0, l)?
        } else {
            core_attn_out
        };

        // Store final state for autoregressive phase, preserving backup tensors
        if let Some(ref mut rs) = self.recurrent_state {
            rs.ssm_state = state;
            rs.gate_cumsum_offset = running_gate_offset;
            // conv_state is already correct from apply_conv, don't overwrite
        } else {
            let conv_dim = self.d_inner + 2 * self.n_groups * self.d_state;
            let conv_state = Tensor::zeros(
                (b, conv_dim, self.conv_kernel_size - 1),
                state.dtype(),
                state.device(),
            )?;
            let mut rs = RecurrentState::new(state, conv_state);
            rs.gate_cumsum_offset = running_gate_offset;
            self.recurrent_state = Some(rs);
        }

        // Transpose back to [b, l, num_heads, head_v_dim]
        // Output is already in input dtype
        core_attn_out.transpose(1, 2)
    }

    /// Single-chunk delta net computation (for sequences <= CHUNK_SIZE)
    /// This avoids chunking overhead for short sequences
    fn delta_net_single_chunk(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        gate: &Tensor,
        beta: &Tensor,
        l: usize,
    ) -> Result<Tensor> {
        let (b, _, num_heads, head_k_dim) = q.dims4()?;
        let (_, _, _, head_v_dim) = v.dims4()?;

        if head_k_dim != head_v_dim {
            candle::bail!(
                "delta_net_single_chunk expects matching head dims ({} vs {})",
                head_k_dim,
                head_v_dim
            );
        }

        // L2 normalize Q and K, with Q also scaled by 1/sqrt(head_dim)
        let q = crate::ops::l2_normalize_scale(q, self.scale as f32, self.rms_norm_eps)?;
        let k = l2_normalize(k, self.rms_norm_eps)?;

        // Apply sigmoid to beta: [b, l, num_heads]
        let beta_sig = candle_nn::ops::sigmoid(beta)?;

        // Transpose to [b, num_heads, l, head_dim] for batch matmul
        let q_t = q.transpose(1, 2)?.contiguous()?;
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;

        // beta_sig: [b, l, num_heads] -> [b, num_heads, l, 1]
        let beta_expanded = beta_sig.transpose(1, 2)?.unsqueeze(3)?;

        // v_beta = v * beta, k_beta = k * beta
        let v_beta = v_t.broadcast_mul(&beta_expanded)?;
        let k_beta = k_t.broadcast_mul(&beta_expanded)?;

        // Get or create cached masks (F32 matches model dtype)
        let (causal_mask, causal_diag_mask) = {
            let needs_new = match &self.cached_masks {
                Some(cached) => cached.seq_len != l,
                None => true,
            };

            if needs_new {
                let cm = create_causal_mask(l, q.device(), DType::F32)?;
                let identity = create_identity_mask(l, q.device(), DType::F32)?;
                let cdm = (&cm + &identity)?;

                self.cached_masks = Some(CachedMasks {
                    seq_len: l,
                    causal_mask: cm.clone(),
                    causal_diag_mask: cdm.clone(),
                });
                (cm, cdm)
            } else {
                let cached = self.cached_masks.as_ref().unwrap();
                (cached.causal_mask.clone(), cached.causal_diag_mask.clone())
            }
        };

        // gate: [b, l, num_heads] -> [b, num_heads, l]
        let gate_t = gate.transpose(1, 2)?.contiguous()?;
        let gate_cumsum = gate_t.cumsum(2)?;

        // Track gate offset for state updates, but do NOT add it to gate_cumsum.
        // The state already encodes decay from all previous tokens, so we only need
        // the LOCAL cumsum (decay within this chunk) for state interaction.
        // Adding the offset would double-count the decay, causing context forgetting.
        // Reference: llama.cpp build_delta_net_chunking uses local cumsum for gexp_chunk.
        let gate_offset_for_state = {
            let local_last = gate_cumsum.narrow(2, l - 1, 1)?.squeeze(2)?;
            if let Some(ref rs) = self.recurrent_state {
                if let Some(ref offset) = rs.gate_cumsum_offset {
                    // Accumulate offset for next continuation
                    Some((offset + &local_last)?)
                } else {
                    Some(local_last)
                }
            } else {
                Some(local_last)
            }
        };

        // Compute decay mask (within-chunk decay is still based on relative positions)
        let gate_i = gate_cumsum.unsqueeze(3)?;
        let gate_j = gate_cumsum.unsqueeze(2)?;
        let decay_diff = gate_i.broadcast_sub(&gate_j)?;
        let decay_mask = decay_diff.broadcast_mul(&causal_diag_mask)?;
        let decay_mask = decay_mask.exp()?;
        let decay_mask = decay_mask.broadcast_mul(&causal_diag_mask)?;

        // kmulkbeta = k_beta @ k^T
        let kmulkbeta = k_beta.matmul(&k_t.transpose(2, 3)?)?;
        let k_decay = kmulkbeta.broadcast_mul(&decay_mask)?;
        let attn1 = k_decay.broadcast_mul(&causal_mask)?.neg()?;

        // Triangular solve
        let attn1_solved = solve_lower_triangular(&attn1, &causal_mask)?;

        // value = attn1_solved @ v_beta
        let value = attn1_solved.matmul(&v_beta)?;

        // k_cumdecay = attn1_solved @ (k_beta * exp(gate_cumsum))
        let gate_cumsum_exp = gate_cumsum.unsqueeze(3)?.exp()?;
        let kbeta_gexp = k_beta.broadcast_mul(&gate_cumsum_exp)?;
        let k_cumdecay = attn1_solved.matmul(&kbeta_gexp)?;

        // q @ k^T attention (strictly lower triangular)
        let attn2 = q_t.matmul(&k_t.transpose(2, 3)?)?;
        let attn2 = attn2.broadcast_mul(&decay_mask)?;
        let attn2 = attn2.broadcast_mul(&causal_diag_mask)?;

        // Get or initialize state (in input dtype)
        let state = if let Some(ref rs) = self.recurrent_state {
            rs.ssm_state.clone()
        } else {
            Tensor::zeros(
                (b, num_heads, head_v_dim, head_v_dim),
                q.dtype(),
                q.device(),
            )?
        };

        // v_prime[t, i] = sum_j(k_cumdecay[t, j] * state[i, j])
        // Need transposed state for correct matrix-vector multiply
        let state_t = state.transpose(2, 3)?;
        let v_prime = k_cumdecay.matmul(&state_t)?;
        let v_new = (value - v_prime)?;

        // attn_inter[t, i] = sum_j(q[t, j] * g[t] * state[i, j])
        let q_g_exp = q_t.broadcast_mul(&gate_cumsum_exp)?;
        let attn_inter = q_g_exp.matmul(&state_t)?;

        // core_attn_out = attn_inter + attn2 @ v_new
        let v_attn = attn2.matmul(&v_new)?;
        let core_attn_out = (attn_inter + v_attn)?;

        // Update state
        let gate_last = gate_cumsum.narrow(2, l - 1, 1)?;
        let g_diff = gate_last.broadcast_sub(&gate_cumsum)?;
        let g_diff_exp = g_diff.unsqueeze(3)?.exp()?;

        // kgdmulvnew[i, j] = sum_t(v_new[t, i] * k[t, j] * g_diff[t])
        // v_new^T @ key_gdiff: [b,h,d,seq] @ [b,h,seq,d] = [b,h,d,d]
        let key_gdiff = k_t.broadcast_mul(&g_diff_exp)?;
        let kgdmulvnew = v_new.transpose(2, 3)?.matmul(&key_gdiff)?;

        let g_last_exp = gate_last.unsqueeze(3)?.exp()?;
        let new_state = (state.broadcast_mul(&g_last_exp)? + kgdmulvnew)?;

        // Store state while preserving backup tensors
        if let Some(ref mut rs) = self.recurrent_state {
            rs.ssm_state = new_state;
            rs.gate_cumsum_offset = gate_offset_for_state;
            // conv_state is already correct from apply_conv, don't overwrite
        } else {
            let conv_dim = self.d_inner + 2 * self.n_groups * self.d_state;
            let conv_state = Tensor::zeros(
                (b, conv_dim, self.conv_kernel_size - 1),
                new_state.dtype(),
                new_state.device(),
            )?;
            let mut rs = RecurrentState::new(new_state, conv_state);
            rs.gate_cumsum_offset = gate_offset_for_state;
            self.recurrent_state = Some(rs);
        }

        // Output is already in input dtype
        core_attn_out.transpose(1, 2)
    }

    /// Single-chunk delta net with PARALLEL intermediate state materialization.
    ///
    /// Uses a fused CUDA kernel that computes DeltaNet outputs AND materializes
    /// intermediate states at each position in a single kernel launch, enabling
    /// O(1) state slicing for speculative decoding verification.
    fn delta_net_single_chunk_with_states(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        gate: &Tensor,
        beta: &Tensor,
        l: usize,
    ) -> Result<Tensor> {
        let (b, _, num_heads, head_k_dim) = q.dims4()?;
        let (_, _, _, head_v_dim) = v.dims4()?;

        if head_k_dim != head_v_dim {
            candle::bail!(
                "delta_net_single_chunk_with_states expects matching head dims ({} vs {})",
                head_k_dim,
                head_v_dim
            );
        }

        // L2 normalize Q and K, with Q also scaled by 1/sqrt(head_dim)
        let q = crate::ops::l2_normalize_scale(q, self.scale as f32, self.rms_norm_eps)?;
        let k = l2_normalize(k, self.rms_norm_eps)?;

        // Apply sigmoid to beta: [b, l, num_heads]
        let beta_sig = candle_nn::ops::sigmoid(beta)?;

        // Transpose to [b, num_heads, l, head_dim] for the CUDA kernel
        let q_t = q.transpose(1, 2)?.contiguous()?;
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;

        // gate and beta_sig: [b, l, num_heads] -> [b, num_heads, l]
        let gate_t = gate.transpose(1, 2)?.contiguous()?;
        let beta_t = beta_sig.transpose(1, 2)?.contiguous()?;

        // Get or initialize state
        let state = if let Some(ref rs) = self.recurrent_state {
            rs.ssm_state.clone()
        } else {
            Tensor::zeros(
                (b, num_heads, head_v_dim, head_v_dim),
                q.dtype(),
                q.device(),
            )?
        };

        // Call the fused CUDA kernel that computes outputs AND intermediate states
        let (output, final_state, all_states) =
            crate::ops::delta_net_parallel_with_states(&q_t, &k_t, &v_t, &gate_t, &beta_t, &state)?;

        // Store final state and intermediate states
        // all_states: [b, h, l, d, d]
        if let Some(ref mut rs) = self.recurrent_state {
            rs.ssm_state = final_state;
            // Extract individual states for O(1) access
            let mut intermediate_states = Vec::with_capacity(l);
            for t in 0..l {
                let state_t = all_states.narrow(2, t, 1)?.squeeze(2)?;
                intermediate_states.push(state_t);
            }
            rs.intermediate_states = Some(intermediate_states);
        } else {
            let conv_dim = self.d_inner + 2 * self.n_groups * self.d_state;
            let conv_state = Tensor::zeros(
                (b, conv_dim, self.conv_kernel_size - 1),
                final_state.dtype(),
                final_state.device(),
            )?;
            let mut rs = RecurrentState::new(final_state, conv_state);
            let mut intermediate_states = Vec::with_capacity(l);
            for t in 0..l {
                let state_t = all_states.narrow(2, t, 1)?.squeeze(2)?;
                intermediate_states.push(state_t);
            }
            rs.intermediate_states = Some(intermediate_states);
            self.recurrent_state = Some(rs);
        }

        // output is [b, h, l, d], transpose to [b, l, h, d]
        output.transpose(1, 2)
    }

    fn gated_norm(&self, input: &Tensor, gate: &Tensor) -> Result<Tensor> {
        // Apply RMS norm to input, then multiply by SiLU(gate)
        let normalized = self.ssm_norm.forward(input)?;
        let gate_silu = candle_nn::ops::silu(gate)?;
        normalized.broadcast_mul(&gate_silu)
    }

    fn clear_state(&mut self) {
        self.recurrent_state = None;
    }

    /// Save a checkpoint of the current recurrent state for later rollback.
    /// Uses in-place copy to backup buffers (no allocation after first call).
    fn checkpoint_state(&mut self) -> Result<()> {
        if let Some(ref mut state) = self.recurrent_state {
            state.checkpoint()?;
        }
        Ok(())
    }

    /// Restore the recurrent state from the saved checkpoint.
    /// Uses deep copy to ensure state independence.
    fn restore_state(&mut self) {
        if let Some(ref mut state) = self.recurrent_state {
            if let Err(e) = state.restore() {
                eprintln!("Warning: failed to restore LinearAttention state: {}", e);
            }
        }
    }

    /// Initialize intermediate states buffer for verification with state slicing.
    /// NOTE: Part of speculative decoding verification - not yet wired up.
    #[allow(dead_code)]
    fn init_intermediate_states(&mut self, seq_len: usize) {
        if let Some(ref mut state) = self.recurrent_state {
            state.init_intermediate_states(seq_len);
        }
    }

    /// Clear intermediate states buffer.
    fn clear_intermediate_states(&mut self) {
        if let Some(ref mut state) = self.recurrent_state {
            state.clear_intermediate_states();
        }
    }

    /// Restore to a specific intermediate state by index.
    /// This is O(1) - just swaps tensor references.
    fn restore_to_intermediate_state(&mut self, index: usize) -> bool {
        if let Some(ref mut state) = self.recurrent_state {
            state.restore_to_intermediate(index)
        } else {
            false
        }
    }

    /// Get the number of stored intermediate states.
    /// NOTE: Part of speculative decoding verification - not yet wired up.
    #[allow(dead_code)]
    fn num_intermediate_states(&self) -> usize {
        self.recurrent_state
            .as_ref()
            .map(|s| s.num_intermediate_states())
            .unwrap_or(0)
    }

    /// Save the current recurrent state for prefix caching.
    /// Returns a deep copy of the SSM state, conv state, and gate offset.
    fn save_state_for_prefix(&self) -> PrefixCacheEntry {
        if let Some(ref state) = self.recurrent_state {
            if let (Ok(ssm), Ok(conv)) =
                (state.ssm_state.contiguous(), state.conv_state.contiguous())
            {
                // Get gate offset, defaulting to zeros if not present
                let gate_offset = state
                    .gate_cumsum_offset
                    .as_ref()
                    .and_then(|t| t.contiguous().ok())
                    .unwrap_or_else(|| {
                        // Create zero offset with correct shape: [batch, num_heads]
                        let (b, num_heads, _, _) = ssm.dims4().unwrap_or((1, 1, 1, 1));
                        Tensor::zeros((b, num_heads), ssm.dtype(), ssm.device())
                            .expect("Failed to create zero gate offset")
                    });
                return PrefixCacheEntry::LinearAttention {
                    ssm_state: ssm,
                    conv_state: conv,
                    gate_cumsum_offset: gate_offset,
                };
            }
        }
        PrefixCacheEntry::Empty
    }

    /// Restore recurrent state from a prefix cache entry.
    fn restore_state_from_prefix(&mut self, entry: &PrefixCacheEntry) -> Result<()> {
        match entry {
            PrefixCacheEntry::LinearAttention {
                ssm_state,
                conv_state,
                gate_cumsum_offset,
            } => {
                // Restore full state including gate offset for correct prefix cache continuation
                self.recurrent_state = Some(RecurrentState::with_gate_offset(
                    ssm_state.contiguous()?,
                    conv_state.contiguous()?,
                    gate_cumsum_offset.contiguous()?,
                ));
                Ok(())
            }
            PrefixCacheEntry::Empty => {
                self.clear_state();
                Ok(())
            }
            PrefixCacheEntry::FullAttention { .. } => {
                // Wrong entry type - clear state
                self.clear_state();
                Ok(())
            }
        }
    }

    /// Fused multi-token delta net computation for small batches.
    ///
    /// Uses a fused CUDA kernel that processes K tokens with the same algorithm
    /// as K sequential autoregressive steps, ensuring identical state updates.
    /// This is used for MTP verification where correctness is critical.
    ///
    /// NOTE: Alternative kernel for performance optimization - not yet wired up.
    #[allow(dead_code)]
    fn delta_net_fused_batch(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        gate: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor> {
        let (b, _l, num_heads, head_k_dim) = q.dims4()?;
        let (_, _, _, head_v_dim) = v.dims4()?;

        if head_k_dim != head_v_dim {
            candle::bail!(
                "delta_net_fused_batch expects matching head dims ({} vs {})",
                head_k_dim,
                head_v_dim
            );
        }

        // L2 normalize Q (with scale) and K
        let q_norm = crate::ops::l2_normalize_scale(q, self.scale as f32, self.rms_norm_eps)?;
        let k_norm = l2_normalize(k, self.rms_norm_eps)?;

        // Apply sigmoid to beta: [b, l, num_heads]
        let beta_sig = candle_nn::ops::sigmoid(beta)?;

        // Transpose to [b, num_heads, l, head_dim] for the fused kernel
        let q_t = q_norm.transpose(1, 2)?.contiguous()?;
        let k_t = k_norm.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;
        let gate_t = gate.transpose(1, 2)?.contiguous()?;
        let beta_t = beta_sig.transpose(1, 2)?.contiguous()?;

        // Get or initialize state
        let state = if let Some(ref rs) = self.recurrent_state {
            rs.ssm_state.clone()
        } else {
            Tensor::zeros(
                (b, num_heads, head_v_dim, head_v_dim),
                q.dtype(),
                q.device(),
            )?
        };

        // Use fused multi-token update kernel
        // Note: q_t is already L2-normalized with scale applied by l2_normalize_scale above
        let (output, new_state) =
            crate::ops::delta_net_multi_token_update(&q_t, &k_t, &v_t, &gate_t, &beta_t, &state)?;

        // Store final state, preserving backup tensors
        if let Some(ref mut rs) = self.recurrent_state {
            rs.ssm_state = new_state;
            // conv_state is already correct from apply_conv, don't overwrite
        } else {
            let conv_dim = self.d_inner + 2 * self.n_groups * self.d_state;
            let conv_state = Tensor::zeros(
                (b, conv_dim, self.conv_kernel_size - 1),
                new_state.dtype(),
                new_state.device(),
            )?;
            self.recurrent_state = Some(RecurrentState::new(new_state, conv_state));
        }

        // Transpose output back to [b, l, num_heads, head_v_dim]
        output.transpose(1, 2)
    }
}

// ============================================================================
// Layer implementation
// ============================================================================

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
enum AttentionLayer {
    Full(FullAttention),
    Linear(LinearAttention),
}

#[derive(Debug)]
struct LayerWeights {
    attn: AttentionLayer,
    moe_block: MoeBlock,
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

impl LayerWeights {
    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (output, _) = self.forward_with_stats(x, mask, offset)?;
        Ok(output)
    }

    fn forward_with_stats(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        let input_dtype = x.dtype();
        let inp_sa = x;

        // Pre-attention norm
        // Note: The CUDA RmsNorm kernel already uses F32 accumulation internally
        // for F16 inputs, so no manual dtype conversion needed
        let h = self.attn_norm.forward(x)?;

        // Attention (full or linear)
        // Full attention handles its own dtype (uses F16 for flash attn internally)
        // Linear attention (DeltaNet) stays in F32 for numerical stability
        let h = match &mut self.attn {
            AttentionLayer::Full(attn) => {
                let out = attn.forward(&h, mask, offset)?;
                // Ensure output matches input dtype
                if out.dtype() != input_dtype {
                    out.to_dtype(input_dtype)?
                } else {
                    out
                }
            }
            AttentionLayer::Linear(attn) => attn.forward(&h)?,
        };

        // Residual connection after attention (stays in input dtype)
        let h = (h + inp_sa)?;

        // Save for FFN residual
        let ffn_residual = &h;

        // Pre-FFN norm
        // Note: The CUDA RmsNorm kernel already uses F32 accumulation internally
        let h = self.ffn_norm.forward(&h)?;

        // FFN (MoE) - quantized weights handle precision internally
        let (h, router_stats) = self.moe_block.forward_with_stats(&h)?;

        // Ensure MoE output matches residual dtype for addition
        let h = if h.dtype() != ffn_residual.dtype() {
            h.to_dtype(ffn_residual.dtype())?
        } else {
            h
        };

        // FFN residual connection
        let output = (h + ffn_residual)?;

        Ok((output, router_stats))
    }

    fn clear_cache(&mut self) {
        match &mut self.attn {
            AttentionLayer::Full(attn) => attn.clear_kv_cache(),
            AttentionLayer::Linear(attn) => attn.clear_state(),
        }
    }

    /// Truncate KV cache to a given sequence length.
    /// For full attention layers, this truncates the cache.
    /// For linear attention layers, this clears the state (can't truncate recurrent state).
    fn truncate_cache(&mut self, new_len: usize) {
        match &mut self.attn {
            AttentionLayer::Full(attn) => attn.truncate_kv_cache(new_len),
            AttentionLayer::Linear(attn) => attn.clear_state(), // Can't truncate recurrent state
        }
    }

    /// Create a checkpoint of the current cache/state for speculative decoding rollback.
    /// For full attention: just records the current cache length (O(1)).
    /// For linear attention: copies state to backup buffers (reuses allocation).
    fn checkpoint_cache(&mut self) -> Result<LayerCheckpoint> {
        match &mut self.attn {
            AttentionLayer::Full(attn) => {
                let len = attn
                    .preallocated_cache
                    .as_ref()
                    .map(|c| c.seq_len)
                    .unwrap_or(0);
                Ok(LayerCheckpoint::FullAttention { seq_len: len })
            }
            AttentionLayer::Linear(attn) => {
                attn.checkpoint_state()?;
                Ok(LayerCheckpoint::LinearAttention)
            }
        }
    }

    /// Restore cache/state from a checkpoint after speculative decoding rejection.
    /// For full attention: just updates the seq_len pointer (O(1)).
    /// For linear attention: swaps backup to primary (O(1) pointer swap).
    fn restore_cache(&mut self, checkpoint: LayerCheckpoint) {
        match (&mut self.attn, checkpoint) {
            (AttentionLayer::Full(attn), LayerCheckpoint::FullAttention { seq_len }) => {
                if let Some(ref mut cache) = attn.preallocated_cache {
                    cache.seq_len = seq_len;
                }
            }
            (AttentionLayer::Linear(attn), LayerCheckpoint::LinearAttention) => {
                attn.restore_state();
            }
            _ => {} // Mismatched types - shouldn't happen
        }
    }

    /// Initialize intermediate states buffer for verification with state slicing.
    /// Only applies to linear attention layers.
    /// NOTE: Part of speculative decoding verification - not yet wired up.
    #[allow(dead_code)]
    fn init_intermediate_states(&mut self, seq_len: usize) {
        if let AttentionLayer::Linear(attn) = &mut self.attn {
            attn.init_intermediate_states(seq_len);
        }
    }

    /// Clear intermediate states buffer.
    fn clear_intermediate_states(&mut self) {
        if let AttentionLayer::Linear(attn) = &mut self.attn {
            attn.clear_intermediate_states();
        }
    }

    /// Restore to a specific intermediate state by index.
    /// For full attention: truncates KV cache to (base_len + index + 1).
    /// For linear attention: restores to the saved intermediate state.
    fn restore_to_intermediate_state(&mut self, index: usize, base_kv_len: usize) -> bool {
        match &mut self.attn {
            AttentionLayer::Full(attn) => {
                // For full attention, truncate KV cache to the position after index
                let new_len = base_kv_len + index + 1;
                attn.truncate_kv_cache(new_len);
                true
            }
            AttentionLayer::Linear(attn) => attn.restore_to_intermediate_state(index),
        }
    }

    /// Forward with PARALLEL intermediate state materialization for linear attention layers.
    ///
    /// For full attention: regular forward (KV cache naturally supports truncation)
    /// For linear attention: uses parallel kernel that materializes states in O(1)
    fn forward_with_state_materialization(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let input_dtype = x.dtype();
        let inp_sa = x;

        // Pre-attention norm
        let h = self.attn_norm.forward(x)?;

        // Attention with parallel state materialization for linear attention
        let h = match &mut self.attn {
            AttentionLayer::Full(attn) => {
                // Full attention uses regular forward - KV cache naturally supports truncation
                let out = attn.forward(&h, mask, offset)?;
                if out.dtype() != input_dtype {
                    out.to_dtype(input_dtype)?
                } else {
                    out
                }
            }
            AttentionLayer::Linear(attn) => attn.forward_with_state_materialization(&h)?,
        };

        // Residual connection after attention
        let h = (h + inp_sa)?;

        // Pre-FFN norm
        let ffn_residual = &h;
        let h = self.ffn_norm.forward(&h)?;

        // FFN (MoE)
        let (h, _router_stats) = self.moe_block.forward_with_stats(&h)?;

        // Ensure output dtype matches
        let h = if h.dtype() != ffn_residual.dtype() {
            h.to_dtype(ffn_residual.dtype())?
        } else {
            h
        };

        // FFN residual connection
        let output = (h + ffn_residual)?;

        Ok(output)
    }

    /// Get the shared K/V cache for MTP state sharing.
    /// Returns None if this is a linear attention layer (MTP uses DeltaNet state instead).
    fn get_shared_kv(&self) -> Option<(Tensor, Tensor)> {
        match &self.attn {
            AttentionLayer::Full(attn) => attn.get_cached_kv(),
            AttentionLayer::Linear(_) => None,
        }
    }

    /// Check if this is a full attention layer.
    fn is_full_attention(&self) -> bool {
        matches!(&self.attn, AttentionLayer::Full(_))
    }

    /// Save the current cache/state for prefix caching.
    /// Returns a deep copy that can be restored later.
    fn save_cache_for_prefix(&self) -> PrefixCacheEntry {
        match &self.attn {
            AttentionLayer::Full(attn) => attn.save_kv_state(),
            AttentionLayer::Linear(attn) => attn.save_state_for_prefix(),
        }
    }

    /// Restore cache/state from a prefix cache entry.
    fn restore_cache_from_prefix(&mut self, entry: &PrefixCacheEntry) -> Result<()> {
        match &mut self.attn {
            AttentionLayer::Full(attn) => attn.restore_kv_state(entry),
            AttentionLayer::Linear(attn) => attn.restore_state_from_prefix(entry),
        }
    }
}

// ============================================================================
// Multi-Token Prediction (MTP) Module
// ============================================================================

/// MTP Decoder Layer - Uses shared state with main model
///
/// CRITICAL: Qwen3-Next MTP uses "state sharing" design:
/// - For full attention: MTP READs from the main model's KV cache (does not maintain its own)
/// - For linear attention: MTP reads from the main model's DeltaNet hidden state
///
/// The MTP layer index corresponds to the main model layer it shares state with.
/// Typically this is the last layer (layer 47 for a 48-layer model like Qwen3-Next).
#[derive(Debug)]
struct MtpDecoderLayer {
    /// Full attention with shared cache (NOTE: cache is borrowed from main model at runtime)
    attn: FullAttention,
    moe_block: MoeBlock,
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    /// The main model layer index this MTP layer shares state with
    #[allow(dead_code)]
    shared_layer_idx: usize,
}

impl MtpDecoderLayer {
    /// Create MTP decoder layer using GGUF-style weight names
    /// Prefix should be like "blk.{idx}.mtp"
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        config: &ModelConfig,
        rotary: Arc<RotaryEmbedding>,
        compute_device: &Device,
        layer_idx: usize,
    ) -> Result<Self> {
        // GGUF naming: blk.{idx}.mtp.in_norm.weight
        // Use Gemma-style norms (1+weight) for MTP
        let attn_norm = gg.rms_norm_gemma(
            &format!("{}.in_norm.weight", prefix),
            config.rms_norm_eps,
            config.dtype,
        )?;
        let ffn_norm = gg.rms_norm_gemma(
            &format!("{}.post_attn_norm.weight", prefix),
            config.rms_norm_eps,
            config.dtype,
        )?;

        // MTP uses full attention (not linear attention) with Gemma-style Q/K norms
        let attn = FullAttention::new_mtp_gemma(
            gg,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rms_norm_eps,
            rotary,
            prefix,
            config.dtype,
            KvCacheQuantization::F16, // MTP doesn't need KV cache quantization
        )?;

        // MTP uses MoE block
        let cache_capacity = default_cache_capacity(config.num_experts_per_tok);
        let moe_block = MoeBlock::new_mtp(
            gg,
            prefix,
            config.num_experts,
            config.num_experts_per_tok,
            config.dtype,
            compute_device,
            cache_capacity,
            layer_idx,
        )?;

        // The shared layer index is the main model layer that MTP shares state with.
        // For MTP at layer_idx 0, it shares with main model's last layer (typically 47 for Qwen3-Next).
        // The caller should configure this appropriately.
        let shared_layer_idx = layer_idx;

        Ok(Self {
            attn,
            moe_block,
            attn_norm,
            ffn_norm,
            shared_layer_idx,
        })
    }

    /// Forward pass returning (hidden_states, residual) like vLLM's Qwen3NextDecoderLayer.
    ///
    /// The vLLM layer returns:
    /// - `hidden_states`: MoE output (NOT normalized after MoE)
    /// - `residual`: attn_output + input (NOT updated after MoE)
    ///
    /// The caller (model.forward()) then does: norm(hidden_states + residual) = norm(moe + attn + input)
    #[allow(dead_code)]
    fn forward(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        // Input residual (like vLLM: residual = hidden_states when residual is None)
        let residual = x.clone();

        // Pre-attention norm (input_layernorm)
        let h = self.attn_norm.forward(x)?;

        // Full attention
        let h = self.attn.forward(&h, mask, offset)?;

        // Fused add+norm after attention (post_attention_layernorm)
        // residual = attn_output + old_residual = attn + input
        // hidden_states = norm(residual)
        let residual = (h + &residual)?;
        let h = self.ffn_norm.forward(&residual)?;

        // FFN (MoE)
        let (h, _router_stats) = self.moe_block.forward_with_stats(&h)?;

        // Return (moe_output, attn + input) matching vLLM pattern
        // Note: residual is NOT updated after MoE - only after attention
        Ok((h, residual))
    }

    /// Forward pass using SHARED K/V cache from main model (native MTP design).
    ///
    /// This is the correct way to run MTP for Qwen3-Next:
    /// - MTP computes its own Q from the input
    /// - MTP reads K/V from the main model's cache (does NOT maintain its own)
    /// - This enables "state sharing" where MTP sees the same context as the main model
    ///
    /// # Arguments
    /// - `x`: Input tensor [batch, seq, hidden]
    /// - `shared_k`: K tensor from main model's last layer cache
    /// - `shared_v`: V tensor from main model's last layer cache
    /// - `mask`: Optional attention mask
    /// - `offset`: Position offset for RoPE
    fn forward_with_shared_kv(
        &mut self,
        x: &Tensor,
        shared_k: &Tensor,
        shared_v: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        // Input residual
        let residual = x.clone();

        // Pre-attention norm
        let h = self.attn_norm.forward(x)?;

        // Full attention with SHARED K/V from main model
        let h = self
            .attn
            .forward_with_shared_kv(&h, shared_k, shared_v, mask, offset)?;

        // Fused add+norm after attention
        let residual = (h + &residual)?;
        let h = self.ffn_norm.forward(&residual)?;

        // FFN (MoE)
        let (h, _router_stats) = self.moe_block.forward_with_stats(&h)?;

        Ok((h, residual))
    }

    fn clear_cache(&mut self) {
        self.attn.clear_kv_cache();
    }
}

/// Multi-Token Prediction weights
///
/// MTP is a form of speculative decoding where we predict multiple tokens
/// using a lightweight prediction head. The MTP head combines the hidden
/// state from the main model with the embedding of the predicted token
/// to predict the next token.
#[derive(Debug)]
pub struct MtpWeights {
    /// FC layer to combine hidden state and embedding: [hidden_size * 2] -> [hidden_size]
    fc: QMatMul,
    /// RMS norm applied to embeddings before FC (zero-centered)
    pre_fc_norm_embedding: RmsNorm,
    /// RMS norm applied to hidden states before FC (zero-centered)
    pre_fc_norm_hidden: RmsNorm,
    /// MTP decoder layers (typically 1)
    layers: Vec<MtpDecoderLayer>,
    /// Final RMS norm (zero-centered)
    pub norm: RmsNorm,
    /// Hidden size
    #[allow(dead_code)]
    hidden_size: usize,
    /// Number of MTP layers
    num_mtp_layers: usize,
}

impl MtpWeights {
    /// Forward pass through MTP
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs to predict next token for [batch, 1]
    /// * `embed_tokens` - Embedding layer (shared with main model)
    /// * `hidden_states` - Hidden states from main model [batch, 1, hidden_size]
    /// * `lm_head` - LM head for producing logits (shared with main model)
    /// * `offset` - Position offset for attention
    /// * `dtype` - Data type for computation
    /// * `spec_step_idx` - Speculative step index (determines which layer to use)
    ///
    /// # Returns
    /// Logits for next token prediction [batch, vocab_size]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        embed_tokens: &Embedding,
        hidden_states: &Tensor,
        lm_head: &QMatMul,
        offset: usize,
        dtype: DType,
        spec_step_idx: usize,
    ) -> Result<Tensor> {
        let debug = std::env::var("MTP_DEBUG").is_ok();

        // Embed the input tokens
        let inputs_embeds = embed_tokens
            .forward(input_ids)?
            .to_dtype(dtype)?
            .contiguous()?;

        // Ensure hidden_states is contiguous (it might be a narrow view)
        let hidden_states = hidden_states.contiguous()?;

        if debug {
            let emb_f32 = inputs_embeds.to_dtype(candle::DType::F32)?;
            let hid_f32 = hidden_states.to_dtype(candle::DType::F32)?;
            eprintln!(
                "[MTP] input_embeds shape={:?}, mean={:.4}",
                inputs_embeds.dims(),
                emb_f32.mean_all()?.to_scalar::<f32>()?
            );
            eprintln!(
                "[MTP] hidden_states (input) shape={:?}, mean={:.4}",
                hidden_states.dims(),
                hid_f32.mean_all()?.to_scalar::<f32>()?
            );
        }

        // Normalize embeddings and hidden states
        let inputs_embeds_norm = self.pre_fc_norm_embedding.forward(&inputs_embeds)?;
        let hidden_states_norm = self.pre_fc_norm_hidden.forward(&hidden_states)?;

        if debug {
            let emb_f32 = inputs_embeds_norm.to_dtype(candle::DType::F32)?;
            let hid_f32 = hidden_states_norm.to_dtype(candle::DType::F32)?;
            eprintln!(
                "[MTP] after pre_fc_norm_embedding: mean={:.4}",
                emb_f32.mean_all()?.to_scalar::<f32>()?
            );
            eprintln!(
                "[MTP] after pre_fc_norm_hidden: mean={:.4}",
                hid_f32.mean_all()?.to_scalar::<f32>()?
            );
        }

        // Concatenate embeddings and hidden states along the last dimension
        // inputs_embeds: [batch, seq, hidden_size]
        // hidden_states: [batch, seq, hidden_size]
        // result: [batch, seq, hidden_size * 2]
        let combined = Tensor::cat(&[&inputs_embeds_norm, &hidden_states_norm], 2)?.contiguous()?;

        if debug {
            eprintln!("[MTP] combined shape={:?}", combined.dims());
        }

        // Apply FC layer to reduce back to hidden_size
        let h = self.fc.forward(&combined)?.contiguous()?;

        if debug {
            let h_f32 = h.to_dtype(candle::DType::F32)?;
            eprintln!(
                "[MTP] after FC: shape={:?}, mean={:.4}",
                h.dims(),
                h_f32.mean_all()?.to_scalar::<f32>()?
            );
        }

        // Forward through ONE MTP decoder layer based on spec_step_idx
        // This matches vLLM's implementation: current_step_idx = spec_step_idx % self.num_mtp_layers
        let layer_idx = spec_step_idx % self.num_mtp_layers;
        let (hidden_states, residual) = self.layers[layer_idx].forward(&h, None, offset)?;

        if debug {
            let hs_f32 = hidden_states.to_dtype(candle::DType::F32)?;
            let res_f32 = residual.to_dtype(candle::DType::F32)?;
            eprintln!(
                "[MTP] layer output: hidden mean={:.4}, residual mean={:.4}",
                hs_f32.mean_all()?.to_scalar::<f32>()?,
                res_f32.mean_all()?.to_scalar::<f32>()?
            );
        }

        // Apply fused add+norm matching vLLM's model.forward():
        // norm(hidden_states + residual) = norm(moe_output + attn + input)
        let h = self.norm.forward(&(&hidden_states + &residual)?)?;

        if debug {
            let h_f32 = h.to_dtype(candle::DType::F32)?;
            eprintln!(
                "[MTP] after final norm: mean={:.4}",
                h_f32.mean_all()?.to_scalar::<f32>()?
            );
        }

        // Apply lm_head to get logits
        let logits = lm_head.forward(&h)?.squeeze(1)?;

        if debug {
            let logits_f32 = logits.to_dtype(candle::DType::F32)?;
            let argmax = logits_f32.argmax(0)?.to_scalar::<u32>()?;
            eprintln!("[MTP] logits shape={:?}, argmax={}", logits.dims(), argmax);
        }

        Ok(logits)
    }

    /// Forward pass through MTP for a single step of speculative decoding.
    /// Returns NORMALIZED hidden states for chaining (matching vLLM Qwen3NextMTP).
    ///
    /// This matches vLLM's Qwen3NextMultiTokenPredictor.forward() which does:
    /// 1. Layer returns (hidden_states, residual) where hidden_states = norm(sum), residual = sum
    /// 2. Final norm does fused add+norm: norm(hidden_states + residual) = norm(norm(sum) + sum)
    ///
    /// Since our layer doesn't have mlp_layernorm, we achieve the same by:
    /// 1. Layer returns (sum, sum)
    /// 2. We do: norm(norm(sum) + sum) using self.norm twice
    pub fn forward_hidden(
        &mut self,
        input_ids: &Tensor,
        embed_tokens: &Embedding,
        hidden_states: &Tensor,
        offset: usize,
        dtype: DType,
        spec_step_idx: usize,
    ) -> Result<Tensor> {
        let debug = std::env::var("MTP_DEBUG").is_ok();

        // Embed the input tokens
        let inputs_embeds = embed_tokens
            .forward(input_ids)?
            .to_dtype(dtype)?
            .contiguous()?;

        // Ensure hidden_states is contiguous (it might be a narrow view)
        let hidden_states = hidden_states.contiguous()?;

        if debug {
            let emb_f32 = inputs_embeds.to_dtype(candle::DType::F32)?;
            let hid_f32 = hidden_states.to_dtype(candle::DType::F32)?;
            eprintln!(
                "[MTP_H] input_embeds shape={:?}, mean={:.4}",
                inputs_embeds.dims(),
                emb_f32.mean_all()?.to_scalar::<f32>()?
            );
            eprintln!(
                "[MTP_H] hidden_states (input) shape={:?}, mean={:.4}",
                hidden_states.dims(),
                hid_f32.mean_all()?.to_scalar::<f32>()?
            );
        }

        // Normalize embeddings and hidden states before projection
        let inputs_embeds_norm = self.pre_fc_norm_embedding.forward(&inputs_embeds)?;
        let hidden_states_norm = self.pre_fc_norm_hidden.forward(&hidden_states)?;

        if debug {
            let emb_f32 = inputs_embeds_norm.to_dtype(candle::DType::F32)?;
            let hid_f32 = hidden_states_norm.to_dtype(candle::DType::F32)?;
            eprintln!(
                "[MTP_H] after norms: embed mean={:.4}, hidden mean={:.4}",
                emb_f32.mean_all()?.to_scalar::<f32>()?,
                hid_f32.mean_all()?.to_scalar::<f32>()?
            );
        }

        // Concatenate embeddings and hidden states along the last dimension
        let combined = Tensor::cat(&[&inputs_embeds_norm, &hidden_states_norm], 2)?.contiguous()?;

        // Apply FC layer
        let h = self.fc.forward(&combined)?.contiguous()?;

        if debug {
            let h_f32 = h.to_dtype(candle::DType::F32)?;
            eprintln!(
                "[MTP_H] after FC: shape={:?}, mean={:.4}",
                h.dims(),
                h_f32.mean_all()?.to_scalar::<f32>()?
            );
        }

        // Forward through MTP decoder layer
        // Layer returns (moe_output, attn + input) matching vLLM pattern
        let layer_idx = spec_step_idx % self.num_mtp_layers;
        let (hidden_states, residual) = self.layers[layer_idx].forward(&h, None, offset)?;

        let output = (&hidden_states + &residual)?;

        if debug {
            let out_f32 = output.to_dtype(candle::DType::F32)?;
            eprintln!(
                "[MTP_H] output (pre-norm) shape={:?}, mean={:.4}",
                output.dims(),
                out_f32.mean_all()?.to_scalar::<f32>()?
            );
        }

        // Return PRE-NORM hidden states for chaining
        // Next MTP step will apply pre_fc_norm_hidden
        Ok(output)
    }

    /// Forward pass using SHARED K/V cache from main model (native MTP design).
    ///
    /// This is the correct way to run MTP for Qwen3-Next:
    /// - MTP computes its own Q from the input
    /// - MTP reads K/V from the main model's cache (does NOT maintain its own)
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs to predict next token for [batch, 1]
    /// * `embed_tokens` - Embedding layer (shared with main model)
    /// * `hidden_states` - Hidden states from main model [batch, 1, hidden_size]
    /// * `lm_head` - LM head for producing logits (shared with main model)
    /// * `shared_k` - K tensor from main model's last layer cache
    /// * `shared_v` - V tensor from main model's last layer cache
    /// * `offset` - Position offset for attention
    /// * `dtype` - Data type for computation
    /// * `spec_step_idx` - Speculative step index
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_shared_kv(
        &mut self,
        input_ids: &Tensor,
        embed_tokens: &Embedding,
        hidden_states: &Tensor,
        lm_head: &QMatMul,
        shared_k: &Tensor,
        shared_v: &Tensor,
        offset: usize,
        dtype: DType,
        spec_step_idx: usize,
    ) -> Result<Tensor> {
        let debug = std::env::var("MTP_DEBUG").is_ok();

        // Embed the input tokens
        let inputs_embeds = embed_tokens
            .forward(input_ids)?
            .to_dtype(dtype)?
            .contiguous()?;

        let hidden_states = hidden_states.contiguous()?;

        if debug {
            eprintln!(
                "[MTP-SHARED] Using shared KV cache: K={:?}, V={:?}",
                shared_k.dims(),
                shared_v.dims()
            );
        }

        // Normalize embeddings and hidden states
        let inputs_embeds_norm = self.pre_fc_norm_embedding.forward(&inputs_embeds)?;
        let hidden_states_norm = self.pre_fc_norm_hidden.forward(&hidden_states)?;

        // Concatenate and project
        let combined = Tensor::cat(&[&inputs_embeds_norm, &hidden_states_norm], 2)?.contiguous()?;
        let h = self.fc.forward(&combined)?.contiguous()?;

        // Forward through MTP decoder layer using SHARED K/V
        let layer_idx = spec_step_idx % self.num_mtp_layers;
        let (hidden_states, residual) =
            self.layers[layer_idx].forward_with_shared_kv(&h, shared_k, shared_v, None, offset)?;

        // Apply fused add+norm
        let h = self.norm.forward(&(&hidden_states + &residual)?)?;

        // Apply lm_head to get logits
        let logits = lm_head.forward(&h)?.squeeze(1)?;

        if debug {
            let logits_f32 = logits.to_dtype(candle::DType::F32)?;
            let argmax = logits_f32.argmax(0)?.to_scalar::<u32>()?;
            eprintln!("[MTP-SHARED] output logits: argmax={}", argmax);
        }

        Ok(logits)
    }

    /// Forward pass using SHARED K/V returning hidden states for chaining.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_hidden_with_shared_kv(
        &mut self,
        input_ids: &Tensor,
        embed_tokens: &Embedding,
        hidden_states: &Tensor,
        shared_k: &Tensor,
        shared_v: &Tensor,
        offset: usize,
        dtype: DType,
        spec_step_idx: usize,
    ) -> Result<Tensor> {
        // Embed the input tokens
        let inputs_embeds = embed_tokens
            .forward(input_ids)?
            .to_dtype(dtype)?
            .contiguous()?;

        let hidden_states = hidden_states.contiguous()?;

        // Normalize embeddings and hidden states
        let inputs_embeds_norm = self.pre_fc_norm_embedding.forward(&inputs_embeds)?;
        let hidden_states_norm = self.pre_fc_norm_hidden.forward(&hidden_states)?;

        // Concatenate and project
        let combined = Tensor::cat(&[&inputs_embeds_norm, &hidden_states_norm], 2)?.contiguous()?;
        let h = self.fc.forward(&combined)?.contiguous()?;

        // Forward through MTP decoder layer using SHARED K/V
        let layer_idx = spec_step_idx % self.num_mtp_layers;
        let (hidden_states, residual) =
            self.layers[layer_idx].forward_with_shared_kv(&h, shared_k, shared_v, None, offset)?;

        // Return PRE-NORM hidden states for chaining
        &hidden_states + &residual
    }

    /// Clear all KV caches in MTP layers
    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }
}

// ============================================================================
// Model
// ============================================================================

#[derive(Debug, Clone)]
struct ModelConfig {
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    num_layers: usize,
    hidden_size: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    rope_freq_base: f64,
    /// Number of dimensions to rotate with RoPE (partial rotary embedding).
    /// For Qwen3-Next this is typically head_dim / 4 = 64 (out of 256).
    /// Only the first n_rot dimensions are rotated, the rest pass through unchanged.
    n_rot: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    ssm_d_inner: usize,
    ssm_d_state: usize,
    ssm_n_groups: usize,
    ssm_dt_rank: usize,
    recurrent_layers: Vec<bool>,
    dtype: DType,
}

#[derive(Debug)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    dtype: DType,
    span: tracing::Span,
    span_output: tracing::Span,
    custom_lm_head_scales: Option<Tensor>,
    /// Multi-Token Prediction module for speculative decoding (optional)
    mtp: Option<MtpWeights>,
    /// Model config for MTP loading
    config: ModelConfig,
    /// Prefetch pipeline coordinator for hiding transfer latency
    prefetch_pipeline: Option<crate::layer_pipeline::PrefetchPipelineCoordinator>,
}

impl ModelWeights {
    fn read_model_config<R: Read + Seek>(gg: &Gguf<R>) -> Result<ModelConfig> {
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let num_attention_heads = md_get("qwen3next.attention.head_count")
            .or_else(|_| md_get("qwen3moe.attention.head_count"))
            .or_else(|_| md_get("llama.attention.head_count"))?
            .to_u32()? as usize;
        let num_key_value_heads = md_get("qwen3next.attention.head_count_kv")
            .or_else(|_| md_get("qwen3moe.attention.head_count_kv"))
            .or_else(|_| md_get("llama.attention.head_count_kv"))?
            .to_u32()? as usize;
        let head_dim = md_get("qwen3next.attention.key_length")
            .or_else(|_| md_get("qwen3moe.attention.key_length"))
            .or_else(|_| md_get("llama.attention.key_length"))?
            .to_u32()? as usize;
        let num_layers = md_get("qwen3next.block_count")
            .or_else(|_| md_get("qwen3moe.block_count"))
            .or_else(|_| md_get("llama.block_count"))?
            .to_u32()? as usize;
        let hidden_size = md_get("qwen3next.embedding_length")
            .or_else(|_| md_get("qwen3moe.embedding_length"))
            .or_else(|_| md_get("llama.embedding_length"))?
            .to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3next.context_length")
            .or_else(|_| md_get("qwen3moe.context_length"))
            .or_else(|_| md_get("llama.context_length"))?
            .to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3next.attention.layer_norm_rms_epsilon")
            .or_else(|_| md_get("qwen3moe.attention.layer_norm_rms_epsilon"))
            .or_else(|_| md_get("llama.attention.layer_norm_rms_epsilon"))?
            .to_f32()? as f64;
        let rope_freq_base = md_get("qwen3next.rope.freq_base")
            .or_else(|_| md_get("qwen3moe.rope.freq_base"))
            .or_else(|_| md_get("llama.rope.freq_base"))?
            .to_f32()? as f64;

        // Read n_rot (rope dimension count) - only these dimensions get rotated.
        // For Qwen3-Next, this is typically 25% of head_dim (64 out of 256).
        // The remaining dimensions pass through unchanged.
        let n_rot = md_get("qwen3next.rope.dimension_count")
            .or_else(|_| md_get("qwen3moe.rope.dimension_count"))
            .or_else(|_| md_get("llama.rope.dimension_count"))
            .map(|v| v.to_u32().unwrap_or((head_dim / 4) as u32) as usize)
            .unwrap_or(head_dim / 4); // Default to 25% for Qwen3-Next

        let num_experts = md_get("qwen3next.expert_count")
            .or_else(|_| md_get("qwen3moe.expert_count"))
            .map(|v| v.to_u32().unwrap_or(64) as usize)
            .unwrap_or(64);
        let num_experts_per_tok = md_get("qwen3next.expert_used_count")
            .or_else(|_| md_get("qwen3moe.expert_used_count"))
            .map(|v| v.to_u32().unwrap_or(8) as usize)
            .unwrap_or(8);

        let ssm_d_inner = md_get("qwen3next.ssm.inner_size")
            .or_else(|_| md_get("qwen3next.ssm.d_inner"))
            .or_else(|_| md_get("qwen3moe.ssm.inner_size"))
            .map(|v| v.to_u32().unwrap_or(0) as usize)
            .unwrap_or(hidden_size * 2);
        let ssm_d_state = md_get("qwen3next.ssm.state_size")
            .or_else(|_| md_get("qwen3next.ssm.d_state"))
            .or_else(|_| md_get("qwen3moe.ssm.state_size"))
            .map(|v| v.to_u32().unwrap_or(128) as usize)
            .unwrap_or(128);
        let ssm_n_groups = md_get("qwen3next.ssm.group_count")
            .or_else(|_| md_get("qwen3next.ssm.n_groups"))
            .or_else(|_| md_get("qwen3moe.ssm.group_count"))
            .map(|v| v.to_u32().unwrap_or(16) as usize)
            .unwrap_or(16);
        let ssm_dt_rank = md_get("qwen3next.ssm.time_step_rank")
            .or_else(|_| md_get("qwen3next.ssm.dt_rank"))
            .or_else(|_| md_get("qwen3moe.ssm.time_step_rank"))
            .map(|v| v.to_u32().unwrap_or(32) as usize)
            .unwrap_or(32);

        // Determine recurrent layers (linear attention without RoPE).
        // In Qwen3-Next, every 4th layer (indices 3, 7, 11, ...) is full attention with RoPE,
        // while the other 3 out of 4 layers are linear attention (recurrent) without RoPE.
        // This matches llama.cpp: hparams.recurrent_layer_arr[i] = ((i + 1) % 4 != 0)
        let recurrent_layers: Vec<bool> = (0..num_layers)
            .map(|i| {
                md_get(&format!("qwen3next.layer.{}.is_recurrent", i))
                    .or_else(|_| md_get(&format!("qwen3moe.layer.{}.is_recurrent", i)))
                    .map(|v| v.to_bool().unwrap_or(false))
                    // Default: use Qwen3-Next pattern where only every 4th layer is full attention
                    // (i + 1) % 4 != 0 means layers 0,1,2 are recurrent, layer 3 is full, etc.
                    .unwrap_or((i + 1) % 4 != 0)
            })
            .collect();

        Ok(ModelConfig {
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            num_layers,
            hidden_size,
            max_position_embeddings,
            rms_norm_eps,
            rope_freq_base,
            n_rot,
            num_experts,
            num_experts_per_tok,
            ssm_d_inner,
            ssm_d_state,
            ssm_n_groups,
            ssm_dt_rank,
            recurrent_layers,
            // Use F32 for activations to match llama.cpp and avoid F16↔F32 conversion overhead.
            // The quantized weights (Q4K, etc.) handle memory efficiency, while F32 activations
            // avoid the costly dtype conversions that kill performance.
            dtype: DType::F32,
        })
    }

    fn build_layers<R: Read + Seek, F>(
        gg: &mut Gguf<R>,
        config: &ModelConfig,
        rotary: Arc<RotaryEmbedding>,
        kv_cache_quantization: KvCacheQuantization,
        compute_device: &Device,
        mut load_moe_block: F,
    ) -> Result<Vec<LayerWeights>>
    where
        F: FnMut(&mut Gguf<R>, &str, &Device, usize) -> Result<MoeBlock>,
    {
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let prefix = format!("blk.{}", i);

            let attn_norm = gg.rms_norm(
                &format!("{}.attn_norm.weight", prefix),
                config.rms_norm_eps,
                config.dtype,
            )?;
            let ffn_norm = gg
                .rms_norm(
                    &format!("{}.ffn_norm.weight", prefix),
                    config.rms_norm_eps,
                    config.dtype,
                )
                .or_else(|_| {
                    gg.rms_norm(
                        &format!("{}.post_attention_norm.weight", prefix),
                        config.rms_norm_eps,
                        config.dtype,
                    )
                })?;

            let is_recurrent = config.recurrent_layers.get(i).copied().unwrap_or(false)
                || gg
                    .try_tensor(&format!("{}.ssm_in.weight", prefix))?
                    .is_some()
                || gg.try_tensor(&format!("{}.ssm_in", prefix))?.is_some();

            let attn = if is_recurrent {
                AttentionLayer::Linear(LinearAttention::new(
                    gg,
                    &prefix,
                    config.ssm_d_inner,
                    config.ssm_d_state,
                    config.ssm_n_groups,
                    config.ssm_dt_rank,
                    config.hidden_size,
                    config.rms_norm_eps,
                    config.dtype,
                )?)
            } else {
                AttentionLayer::Full(FullAttention::new(
                    gg,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    config.rms_norm_eps,
                    rotary.clone(),
                    &prefix,
                    config.dtype,
                    kv_cache_quantization,
                    config.max_position_embeddings,
                )?)
            };

            let moe_block = load_moe_block(gg, &prefix, compute_device, i)?;

            layers.push(LayerWeights {
                attn,
                moe_block,
                attn_norm,
                ffn_norm,
            });
        }

        Ok(layers)
    }

    fn load_expert_tensor(path: &Path, tensor_name: &str, device: &Device) -> Result<QTensor> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut gg = Gguf::new(content, file, device.clone());
        gg.tensor(tensor_name)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_moe_block_with_devices<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        config: &ModelConfig,
        path: &Path,
        gate_device: &Device,
        up_device: &Device,
        down_device: &Device,
        compute_device: &Device,
        layer_idx: usize,
    ) -> Result<MoeBlock> {
        let gate_exps = Self::load_expert_tensor(
            path,
            &format!("{}.ffn_gate_exps.weight", prefix),
            gate_device,
        )?;
        let up_exps =
            Self::load_expert_tensor(path, &format!("{}.ffn_up_exps.weight", prefix), up_device)?;
        let down_exps = Self::load_expert_tensor(
            path,
            &format!("{}.ffn_down_exps.weight", prefix),
            down_device,
        )?;
        let cache_capacity = default_cache_capacity(config.num_experts_per_tok);

        let gate = gg.qmatmul(&format!("{}.ffn_gate_inp.weight", prefix))?;
        let shared_expert = SharedExpert::new(gg, prefix, config.dtype)?;

        let gate_exps = Arc::new(gate_exps);
        let up_exps = Arc::new(up_exps);
        let down_exps = Arc::new(down_exps);

        let cache = if should_cache_experts(&gate_exps, &up_exps, &down_exps, compute_device) {
            Some(ExpertCache::new(compute_device.clone(), cache_capacity))
        } else {
            None
        };

        let training_cache =
            if should_cache_experts(&gate_exps, &up_exps, &down_exps, compute_device) {
                Some(ExpertCache::new(compute_device.clone(), cache_capacity))
            } else {
                None
            };

        // GPU hot cache is initialized later via enable_gpu_hot_cache
        let experts = MoeExperts {
            gate_exps,
            up_exps,
            down_exps,
            act_fn: Activation::Silu,
            span: tracing::span!(tracing::Level::TRACE, "moe-experts"),
            cache,
            training_cache,
            compute_device: compute_device.clone(),
            custom_gate_block_mults: None,
            custom_up_block_mults: None,
            custom_down_block_mults: None,
            gpu_hot_cache: None,
            gpu_device: None,
            transposed_layout: false,
        };

        Ok(MoeBlock {
            experts,
            shared_expert,
            gate,
            num_experts: config.num_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            span: tracing::span!(tracing::Level::TRACE, "moe-block"),
            custom_gate_scales: None,
            layer_idx,
        })
    }

    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let config = Self::read_model_config(&gg)?;
        let embed_tensor = gg
            .tensor("token_embd.weight")
            .or_else(|_| gg.tensor("token_embdd.weight"))?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, config.hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            config.dtype,
            config.head_dim,
            config.n_rot,
            config.max_position_embeddings,
            config.rope_freq_base,
            device,
        )?);

        let layers = Self::build_layers(
            &mut gg,
            &config,
            rotary.clone(),
            KvCacheQuantization::F16,
            device,
            |gg, prefix, compute_device, layer_idx| {
                MoeBlock::new(
                    gg,
                    prefix,
                    config.num_experts,
                    config.num_experts_per_tok,
                    config.dtype,
                    compute_device,
                    0,
                    layer_idx,
                )
            },
        )?;

        let norm = gg.rms_norm("output_norm.weight", config.rms_norm_eps, config.dtype)?;
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg
                .tensor("token_embd.weight")
                .or_else(|_| gg.tensor("token_embdd.weight"))?,
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype: config.dtype,
            span,
            span_output,
            custom_lm_head_scales: None,
            mtp: None,
            config,
            prefetch_pipeline: None,
        })
    }

    pub fn from_gguf_with_device_map<P: AsRef<Path>>(
        path: P,
        device: &Device,
        expert_device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_with_kv_cache_config(
            path,
            device,
            expert_device,
            expert_device,
            expert_device,
            KvCacheQuantization::F16,
        )
    }

    pub fn from_gguf_with_expert_device_map<P: AsRef<Path>>(
        path: P,
        device: &Device,
        gate_device: &Device,
        up_device: &Device,
        down_device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_with_kv_cache_config(
            path,
            device,
            gate_device,
            up_device,
            down_device,
            KvCacheQuantization::F16,
        )
    }

    /// Load model from GGUF with explicit device placement for expert weights and KV-cache settings.
    pub fn from_gguf_with_kv_cache_config<P: AsRef<Path>>(
        path: P,
        device: &Device,
        gate_device: &Device,
        up_device: &Device,
        down_device: &Device,
        kv_cache_quantization: KvCacheQuantization,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(path.as_ref())?;
        let content = gguf_file::Content::read(&mut file)?;
        if device.same_device(gate_device)
            && device.same_device(up_device)
            && device.same_device(down_device)
            && matches!(kv_cache_quantization, KvCacheQuantization::F16)
        {
            return Self::from_gguf(content, &mut file, device);
        }
        Self::from_gguf_with_devices(
            content,
            &mut file,
            path.as_ref(),
            device,
            gate_device,
            up_device,
            down_device,
            kv_cache_quantization,
        )
    }

    /// Load model from GGUF with device offload mode.
    ///
    /// This provides a simpler API for common offloading strategies:
    /// - [`DeviceOffloadMode::FullGpu`]: All weights on GPU (fastest, most VRAM)
    /// - [`DeviceOffloadMode::ExpertsOnCpu`]: All MoE expert weights on CPU (default, lowest VRAM)
    /// - [`DeviceOffloadMode::UpProjectionsOnCpu`]: Only up projections on CPU
    /// - [`DeviceOffloadMode::UpDownProjectionsOnCpu`]: Up and down on CPU, gate on GPU
    ///
    /// # Example
    ///
    /// ```ignore
    /// use paramecia_model::qwen3_next::{ModelWeights, DeviceOffloadMode, KvCacheQuantization};
    ///
    /// let model = ModelWeights::from_gguf_with_offload_mode(
    ///     "model.gguf",
    ///     &device,
    ///     DeviceOffloadMode::UpDownProjectionsOnCpu,
    ///     KvCacheQuantization::Q4K,
    /// )?;
    /// ```
    pub fn from_gguf_with_offload_mode<P: AsRef<Path>>(
        path: P,
        device: &Device,
        offload_mode: DeviceOffloadMode,
        kv_cache_quantization: KvCacheQuantization,
    ) -> Result<Self> {
        let (gate_device, up_device, down_device) = offload_mode.get_expert_devices(device);
        Self::from_gguf_with_kv_cache_config(
            path,
            device,
            &gate_device,
            &up_device,
            &down_device,
            kv_cache_quantization,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_gguf_with_devices<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        path: &Path,
        device: &Device,
        gate_device: &Device,
        up_device: &Device,
        down_device: &Device,
        kv_cache_quantization: KvCacheQuantization,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let config = Self::read_model_config(&gg)?;
        let embed_tensor = gg
            .tensor("token_embd.weight")
            .or_else(|_| gg.tensor("token_embdd.weight"))?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, config.hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            config.dtype,
            config.head_dim,
            config.n_rot,
            config.max_position_embeddings,
            config.rope_freq_base,
            device,
        )?);

        let layers = Self::build_layers(
            &mut gg,
            &config,
            rotary.clone(),
            kv_cache_quantization,
            device,
            |gg, prefix, compute_device, layer_idx| {
                Self::build_moe_block_with_devices(
                    gg,
                    prefix,
                    &config,
                    path,
                    gate_device,
                    up_device,
                    down_device,
                    compute_device,
                    layer_idx,
                )
            },
        )?;

        let norm = gg.rms_norm("output_norm.weight", config.rms_norm_eps, config.dtype)?;
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg
                .tensor("token_embd.weight")
                .or_else(|_| gg.tensor("token_embdd.weight"))?,
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype: config.dtype,
            span,
            span_output,
            custom_lm_head_scales: None,
            mtp: None,
            config,
            prefetch_pipeline: None,
        })
    }

    pub fn from_gguf_file<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        Self::from_gguf_with_kv_cache_config(
            path,
            device,
            device,
            device,
            device,
            KvCacheQuantization::F16,
        )
    }

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| (0..(tgt + offset)).map(move |j| if j <= i + offset { 0. } else { minf }))
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    /// Enable GPU hot expert caching for CPU-offloaded experts.
    ///
    /// This keeps the most frequently used experts cached on GPU,
    /// avoiding CPU computation entirely for hot experts. This is
    /// the key optimization for CPU-offloaded MoE.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of hot experts to cache on GPU per layer
    pub fn enable_gpu_hot_cache(&mut self, capacity: usize) {
        if !matches!(self.device, Device::Cuda(_)) {
            return; // No GPU available
        }

        for layer in &mut self.layers {
            layer
                .moe_block
                .experts
                .enable_gpu_hot_cache(self.device.clone(), capacity);
        }
    }

    /// Enable prefetch-based pipelining for hiding transfer latency
    ///
    /// This creates a two-stage pipeline:
    /// 1. Prefetch thread: transfers data from GPU to CPU in parallel with GPU work
    /// 2. MoE worker thread: processes MoE on prefetched data
    ///
    /// Best used when GPU->CPU transfer is a bottleneck.
    pub fn enable_prefetch_pipeline(&mut self) -> Result<()> {
        // Collect expert weights from all layers
        let mut gate_weights = Vec::with_capacity(self.layers.len());
        let mut up_weights = Vec::with_capacity(self.layers.len());
        let mut down_weights = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            gate_weights.push(Arc::clone(&layer.moe_block.experts.gate_exps));
            up_weights.push(Arc::clone(&layer.moe_block.experts.up_exps));
            down_weights.push(Arc::clone(&layer.moe_block.experts.down_exps));
        }

        let num_experts = self.layers[0].moe_block.num_experts;

        // Create the prefetch pipeline coordinator
        let pipeline = crate::layer_pipeline::PrefetchPipelineCoordinator::new(
            self.device.clone(),
            self.layers.len(),
            gate_weights,
            up_weights,
            down_weights,
            num_experts,
        );

        self.prefetch_pipeline = Some(pipeline);
        Ok(())
    }

    /// Check if prefetch pipeline is enabled
    pub fn has_prefetch_pipeline(&self) -> bool {
        self.prefetch_pipeline.is_some()
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let span = self.span.clone();
        let _enter = span.enter();
        let dims = input.dims();
        let b = dims.first().copied().unwrap_or(0);
        let l = dims.get(1).copied().unwrap_or(0);
        let mut h = self.embed_tokens.forward(input)?.to_dtype(self.dtype)?;

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };

        // Use prefetch pipeline if enabled, else standard sequential
        if self.prefetch_pipeline.is_some() {
            h = self.forward_prefetch_pipelined(&h, causal_mask.as_ref(), offset)?;
        } else {
            for layer in self.layers.iter_mut() {
                h = layer.forward(&h, causal_mask.as_ref(), offset)?;
            }
        }

        // Final normalization and language model head
        // Norm may output F32 for stability, ensure proper casting for lm_head
        let h = self.norm.forward(&h)?;
        let span_output = self.span_output.clone();
        let _enter_output = span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?;
        // Convert to F32 BEFORE lm_head to avoid F16 overflow in logits
        // (F16 max is ~65504, but logits for large vocabs can exceed this)
        let last_hidden = if last_hidden.dtype() != DType::F32 {
            last_hidden.to_dtype(DType::F32)?
        } else {
            last_hidden
        };
        // LM head now operates in F32, producing F32 logits without overflow
        let logits = if let Some(scales) = &self.custom_lm_head_scales {
            self.lm_head
                .forward_with_scales(&last_hidden, scales)?
                .squeeze(1)?
        } else {
            self.lm_head.forward(&last_hidden)?.squeeze(1)?
        };
        Ok(logits)
    }

    /// Prefetch-based pipelined forward pass
    ///
    /// This version hides GPU->CPU transfer latency by:
    /// 1. Starting data transfer immediately after attention
    /// 2. Processing router/norm on GPU while transfer happens
    /// 3. MoE worker starts as soon as data arrives
    fn forward_prefetch_pipelined(
        &mut self,
        initial_hidden: &Tensor,
        causal_mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let pipeline = self
            .prefetch_pipeline
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("Prefetch pipeline not enabled".to_string()))?;

        let num_layers = self.layers.len();
        let (batch_size, seq_len, hidden_dim) = initial_hidden.dims3()?;
        let activation_dtype = initial_hidden.dtype();
        let mut h = initial_hidden.clone();
        let mut pending_moe: Option<usize> = None;

        for layer_idx in 0..num_layers {
            let is_last = layer_idx == num_layers - 1;
            let layer = &mut self.layers[layer_idx];

            // Wait for previous layer's MoE result if pending
            if let Some(prev_layer) = pending_moe.take() {
                let result = pipeline.wait_for_result(prev_layer)?;
                // Pipeline returns F32, cast to activation dtype
                h = if result.dtype() != activation_dtype {
                    result.to_dtype(activation_dtype)?
                } else {
                    result
                };
            }

            // === STAGE 1: Attention (GPU) ===
            let inp_sa = &h;
            let h_normed = layer.attn_norm.forward(&h)?;
            // Ensure norm output matches activation dtype
            let h_normed = if h_normed.dtype() != activation_dtype {
                h_normed.to_dtype(activation_dtype)?
            } else {
                h_normed
            };

            let attn_out = match &mut layer.attn {
                AttentionLayer::Full(attn) => attn.forward(&h_normed, causal_mask, offset)?,
                AttentionLayer::Linear(attn) => attn.forward(&h_normed)?,
            };

            // Ensure attention output matches input dtype for residual
            let attn_out = if attn_out.dtype() != activation_dtype {
                attn_out.to_dtype(activation_dtype)?
            } else {
                attn_out
            };

            // Residual after attention
            let h_after_attn = (&attn_out + inp_sa)?;

            // === STAGE 2: FFN Norm + Router (GPU) ===
            let ffn_residual = h_after_attn.clone();
            let h_ffn_normed = layer.ffn_norm.forward(&h_after_attn)?;
            // Ensure norm output matches activation dtype
            let h_ffn_normed = if h_ffn_normed.dtype() != activation_dtype {
                h_ffn_normed.to_dtype(activation_dtype)?
            } else {
                h_ffn_normed
            };

            if is_last {
                // Last layer - run synchronously
                let (moe_out, _stats) = layer.moe_block.forward_with_stats(&h_ffn_normed)?;
                // Ensure MoE output matches residual dtype
                let moe_out = if moe_out.dtype() != activation_dtype {
                    moe_out.to_dtype(activation_dtype)?
                } else {
                    moe_out
                };
                h = (&moe_out + &ffn_residual)?;
            } else {
                // Get router outputs (GPU)
                let hidden_flat = h_ffn_normed
                    .reshape((batch_size * seq_len, hidden_dim))?
                    .contiguous()?;

                // Router forward (GPU) - must match MoeBlock::forward_with_stats behavior
                // including custom_gate_scales handling and logit clamping
                let routing_logits = if let Some(ref scales) = layer.moe_block.custom_gate_scales {
                    layer
                        .moe_block
                        .gate
                        .forward_with_scales(&hidden_flat, scales)?
                } else {
                    layer.moe_block.gate.forward(&hidden_flat)?
                };

                // Clamp router logits to prevent softmax overflow (exp of large values -> inf -> NaN)
                // This is critical for numerical stability - without it, extreme logits cause
                // degenerate routing where one expert gets all weight, leading to repetitions
                let routing_logits_clamped = routing_logits.clamp(-100.0, 100.0)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&routing_logits_clamped)?;

                // Top-k selection (GPU) - reshape to [batch, seq, n_experts] as expected
                let routing_3d =
                    routing_weights.reshape((batch_size, seq_len, routing_weights.dim(1)?))?;
                let (weights, indices) =
                    crate::ops::topk_moe_routing(&routing_3d, layer.moe_block.num_experts_per_tok)?;

                let weights_flat =
                    weights.reshape((batch_size * seq_len, layer.moe_block.num_experts_per_tok))?;
                let indices_flat =
                    indices.reshape((batch_size * seq_len, layer.moe_block.num_experts_per_tok))?;

                // Compute shared expert output on GPU (if present)
                // This must be included in the residual since the async worker only handles routed experts
                let adjusted_residual = if let Some(ref shared_exp) = layer.moe_block.shared_expert
                {
                    let shared_out = shared_exp.forward(&h_ffn_normed)?;
                    (&ffn_residual + &shared_out)?
                } else {
                    ffn_residual.clone()
                };

                // Submit prefetch request - this starts the GPU->CPU transfer
                // while we continue to the next iteration
                pipeline.submit_prefetch(
                    layer_idx,
                    &hidden_flat,
                    &indices_flat,
                    &weights_flat,
                    &adjusted_residual,
                )?;

                pending_moe = Some(layer_idx);

                // Use post-attention state as placeholder
                h = h_after_attn;
            }
        }

        Ok(h)
    }

    /// Forward pass that returns logits for ALL positions (for speculative decoding verification)
    pub fn forward_all_positions(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let dims = input.dims();
        let b = dims.first().copied().unwrap_or(0);
        let l = dims.get(1).copied().unwrap_or(0);
        let mut h = self.embed_tokens.forward(input)?.to_dtype(self.dtype)?;

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal_mask.as_ref(), offset)?;
        }

        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();

        // Return logits for ALL positions, not just last
        // Shape: [batch, seq_len, vocab_size]
        if let Some(scales) = &self.custom_lm_head_scales {
            self.lm_head.forward_with_scales(&h, scales)
        } else {
            self.lm_head.forward(&h)
        }
    }

    /// Forward pass that returns logits for ALL positions AND hidden states for the last position.
    /// This is optimal for MTP verification + next-round preparation in a single pass.
    ///
    /// Returns: (all_logits [batch, seq_len, vocab], last_hidden_states [batch, 1, hidden])
    pub fn forward_all_positions_with_hidden(
        &mut self,
        input: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();
        let dims = input.dims();
        let b = dims.first().copied().unwrap_or(0);
        let l = dims.get(1).copied().unwrap_or(0);
        let mut h = self.embed_tokens.forward(input)?.to_dtype(self.dtype)?;

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal_mask.as_ref(), offset)?;
        }

        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();

        // Get hidden states for the last position (for MTP)
        let last_hidden = h.narrow(1, l - 1, 1)?;

        // Return logits for ALL positions
        let logits = if let Some(scales) = &self.custom_lm_head_scales {
            self.lm_head.forward_with_scales(&h, scales)?
        } else {
            self.lm_head.forward(&h)?
        };

        Ok((logits, last_hidden))
    }

    /// Verification forward pass with PARALLEL intermediate state materialization.
    ///
    /// This method:
    /// 1. Processes all tokens in ONE parallel forward pass
    /// 2. Materializes intermediate DeltaNet states at each position in O(1)
    /// 3. Returns logits and hidden states for all positions
    ///
    /// On partial rejection, call `restore_to_intermediate_state(index)` for O(1) state slicing.
    ///
    /// # Arguments
    /// * `input` - Token IDs [batch, seq_len]
    /// * `offset` - Starting position offset
    ///
    /// # Returns
    /// * `all_logits` - Logits for each position [batch, seq_len, vocab]
    /// * `all_hidden` - Hidden states at each position [batch, seq_len, hidden_size] (pre-norm)
    pub fn verify_with_state_materialization(
        &mut self,
        input: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();
        let dims = input.dims();
        let b = dims.first().copied().unwrap_or(0);
        let l = dims.get(1).copied().unwrap_or(0);

        let mut h = self.embed_tokens.forward(input)?.to_dtype(self.dtype)?;

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };

        // PARALLEL forward with state materialization for LinearAttention layers
        for layer in &mut self.layers {
            h = layer.forward_with_state_materialization(&h, causal_mask.as_ref(), offset)?;
        }

        // Save pre-norm hidden states for MTP
        let all_hidden = h.clone();

        // Apply final norm and get logits
        let h_normed = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();

        let logits = if let Some(scales) = &self.custom_lm_head_scales {
            self.lm_head.forward_with_scales(&h_normed, scales)?
        } else {
            self.lm_head.forward(&h_normed)?
        };

        Ok((logits, all_hidden))
    }

    /// Restore all layers to a specific intermediate state index.
    ///
    /// This is O(1) for both layer types:
    /// - FullAttention: truncates KV cache to (offset + index + 1)
    /// - LinearAttention: picks from materialized intermediate states
    ///
    /// Call this after `verify_with_state_materialization` when partial rejection occurs.
    pub fn restore_to_intermediate_state(&mut self, index: usize, offset: usize) {
        for layer in &mut self.layers {
            layer.restore_to_intermediate_state(index, offset);
        }
    }

    /// Clear intermediate states buffer for all layers.
    /// Call this after verification is complete to free memory.
    pub fn clear_intermediate_states(&mut self) {
        for layer in &mut self.layers {
            layer.clear_intermediate_states();
        }
    }

    /// Forward pass that returns both logits and hidden states (for MTP speculative decoding)
    ///
    /// Returns: (logits, hidden_states) where hidden_states are PRE-NORM
    /// (before the final RMS norm). MTP expects pre-norm hidden states because
    /// it applies its own pre_fc_norm_hidden normalization.
    pub fn forward_with_hidden_states(
        &mut self,
        input: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let debug = std::env::var("MTP_DEBUG").is_ok();

        // Debug: check cache before forward
        if debug {
            let cache_before = self.get_last_full_attention_kv();
            if let Some((k, _)) = cache_before {
                eprintln!(
                    "[forward_with_hidden_states] Cache BEFORE: seq_len={}",
                    k.dim(2)?
                );
            } else {
                eprintln!("[forward_with_hidden_states] Cache BEFORE: None");
            }
        }

        let _enter = self.span.enter();
        let dims = input.dims();
        let b = dims.first().copied().unwrap_or(0);
        let l = dims.get(1).copied().unwrap_or(0);
        let mut h = self.embed_tokens.forward(input)?.to_dtype(self.dtype)?;

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal_mask.as_ref(), offset)?;
        }

        // Debug: check cache after forward
        if debug {
            let cache_after = self.get_last_full_attention_kv();
            if let Some((k, _)) = cache_after {
                eprintln!(
                    "[forward_with_hidden_states] Cache AFTER: seq_len={}, offset={}, input_len={}",
                    k.dim(2)?,
                    offset,
                    l
                );
            } else {
                eprintln!("[forward_with_hidden_states] Cache AFTER: None");
            }
        }

        // Save PRE-norm hidden states for MTP
        // MTP applies its own pre_fc_norm, so we give it raw hidden states
        let pre_norm_hidden = h.narrow(1, l - 1, 1)?;

        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?;

        // Debug: print hidden state statistics
        if std::env::var("MTP_DEBUG").is_ok() {
            let pre_f32 = pre_norm_hidden.to_dtype(candle::DType::F32)?;
            let post_f32 = last_hidden.to_dtype(candle::DType::F32)?;
            eprintln!(
                "[MAIN] pre_norm mean={:.4}, post_norm mean={:.4}",
                pre_f32.mean_all()?.to_scalar::<f32>()?,
                post_f32.mean_all()?.to_scalar::<f32>()?
            );
        }

        let logits = if let Some(scales) = &self.custom_lm_head_scales {
            self.lm_head
                .forward_with_scales(&last_hidden, scales)?
                .squeeze(1)?
        } else {
            self.lm_head.forward(&last_hidden)?.squeeze(1)?
        };

        // Return PRE-norm hidden states for MTP
        // MTP applies its own pre_fc_norm_hidden normalization
        Ok((logits, pre_norm_hidden))
    }

    /// Extract text embeddings by mean-pooling hidden states.
    ///
    /// This method runs a forward pass and returns the mean of all hidden states,
    /// which can be used for text similarity comparisons (e.g., cosine similarity).
    ///
    /// # Arguments
    /// * `input` - Token IDs tensor of shape [batch, seq_len]
    ///
    /// # Returns
    /// * Embedding tensor of shape [batch, hidden_size]
    pub fn forward_embeddings(&mut self, input: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let dims = input.dims();
        let b = dims.first().copied().unwrap_or(0);
        let l = dims.get(1).copied().unwrap_or(0);
        let mut h = self.embed_tokens.forward(input)?.to_dtype(self.dtype)?;

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, 0)?)
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal_mask.as_ref(), 0)?;
        }

        // Apply final normalization
        let h = self.norm.forward(&h)?;

        // Mean pooling: average all token hidden states
        // h shape: [batch, seq_len, hidden_size]
        let embeddings = h.mean(1)?; // [batch, hidden_size]

        Ok(embeddings)
    }

    /// Extract text embeddings using the last token's hidden state.
    ///
    /// This is more efficient than mean pooling and often works well for
    /// causal LLMs where information accumulates at the end.
    ///
    /// # Arguments
    /// * `input` - Token IDs tensor of shape [batch, seq_len]
    ///
    /// # Returns
    /// * Embedding tensor of shape [batch, hidden_size]
    pub fn forward_embeddings_last(&mut self, input: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let dims = input.dims();
        let b = dims.first().copied().unwrap_or(0);
        let l = dims.get(1).copied().unwrap_or(0);
        let mut h = self.embed_tokens.forward(input)?.to_dtype(self.dtype)?;

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, 0)?)
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal_mask.as_ref(), 0)?;
        }

        // Apply final normalization
        let h = self.norm.forward(&h)?;

        // Take last token hidden state
        // h shape: [batch, seq_len, hidden_size]
        let embeddings = h.narrow(1, l - 1, 1)?.squeeze(1)?; // [batch, hidden_size]

        Ok(embeddings)
    }

    /// Evaluate MTP prediction accuracy on a sequence.
    ///
    /// This method evaluates how accurate the MTP head is at predicting future tokens.
    /// For each position in the input, it:
    /// 1. Gets the main model's prediction for the next token
    /// 2. Gets the MTP head's prediction for the next N tokens
    /// 3. Compares MTP predictions against actual next tokens (if available)
    ///
    /// # Arguments
    /// * `input` - Token IDs tensor of shape [batch, seq_len]
    /// * `num_mtp_steps` - Number of MTP prediction steps to evaluate (typically 1-4)
    ///
    /// # Returns
    /// * (accuracy, total_predictions, correct_predictions)
    ///   - accuracy: Percentage of correct MTP predictions (0.0 - 1.0)
    ///   - total_predictions: Total MTP predictions made
    ///   - correct_predictions: Number of correct predictions
    pub fn evaluate_mtp_accuracy(
        &mut self,
        input: &Tensor,
        num_mtp_steps: usize,
    ) -> Result<(f32, usize, usize)> {
        if self.mtp.is_none() {
            // No MTP head loaded, return 0 accuracy
            return Ok((0.0, 0, 0));
        }

        let seq_len = input.dims()[1];
        if seq_len < num_mtp_steps + 2 {
            // Need at least num_mtp_steps + 2 tokens to evaluate
            return Ok((0.0, 0, 0));
        }

        let mut total_predictions = 0usize;
        let mut correct_predictions = 0usize;

        // Get the input tokens as a vector
        let input_tokens: Vec<u32> = input.squeeze(0)?.to_vec1()?;

        // For each starting position where we can evaluate MTP
        let max_start = seq_len.saturating_sub(num_mtp_steps + 1);

        for start_pos in 0..max_start {
            // Get prefix up to and including start_pos
            let prefix_len = start_pos + 1;
            let prefix = input.narrow(1, 0, prefix_len)?;

            // Clear caches
            self.clear_kv_cache();
            if let Some(ref mut mtp) = self.mtp {
                mtp.clear_cache();
            }

            // Run main model to get hidden states at start_pos
            let (main_logits, hidden_states) = self.forward_with_hidden_states(&prefix, 0)?;

            // Get the main model's predicted token (greedy)
            let main_pred = main_logits.argmax(candle::D::Minus1)?.to_vec1::<u32>()?[0];

            // Now run MTP for num_mtp_steps
            let mut current_hidden = hidden_states;
            let mut current_token = main_pred;
            let mut current_offset = prefix_len;

            for mtp_step in 0..num_mtp_steps {
                // Check if we have ground truth for this position
                let target_pos = start_pos + 1 + mtp_step;
                if target_pos >= seq_len {
                    break;
                }

                let target_token = input_tokens[target_pos];

                // Create input tensor for current token
                let token_input = Tensor::new(&[current_token], &self.device)?.unsqueeze(0)?;

                // Get MTP prediction
                let mtp = self.mtp.as_mut().unwrap();
                let mtp_hidden = mtp.forward_hidden(
                    &token_input,
                    &self.embed_tokens,
                    &current_hidden,
                    current_offset,
                    self.dtype,
                    mtp_step,
                )?;

                // CRITICAL: Must normalize before lm_head - forward_hidden returns pre-norm states
                let mtp_normalized = mtp.norm.forward(&mtp_hidden)?;
                let mtp_logits = self.lm_head.forward(&mtp_normalized)?.squeeze(1)?;
                let mtp_pred = mtp_logits.argmax(candle::D::Minus1)?.to_vec1::<u32>()?[0];

                // Compare prediction to ground truth
                total_predictions += 1;
                if mtp_pred == target_token {
                    correct_predictions += 1;
                }

                // Update for next step
                current_hidden = mtp_hidden;
                current_token = target_token; // Use ground truth for next step evaluation
                current_offset += 1;
            }
        }

        let accuracy = if total_predictions > 0 {
            correct_predictions as f32 / total_predictions as f32
        } else {
            0.0
        };

        Ok((accuracy, total_predictions, correct_predictions))
    }

    /// Quick MTP accuracy evaluation on a single sequence position.
    ///
    /// This is a lighter-weight version that evaluates MTP from a single position.
    ///
    /// # Arguments
    /// * `input` - Token IDs tensor of shape [batch, seq_len]  
    /// * `eval_position` - Position to evaluate from (must have num_mtp_steps tokens after it)
    /// * `num_mtp_steps` - Number of MTP steps to evaluate
    ///
    /// # Returns
    /// * Accuracy as f32 (0.0 - 1.0)
    pub fn evaluate_mtp_accuracy_at_position(
        &mut self,
        input: &Tensor,
        eval_position: usize,
        num_mtp_steps: usize,
    ) -> Result<f32> {
        if self.mtp.is_none() {
            return Ok(0.0);
        }

        let seq_len = input.dims()[1];
        if eval_position + num_mtp_steps >= seq_len {
            return Ok(0.0);
        }

        let input_tokens: Vec<u32> = input.squeeze(0)?.to_vec1()?;

        // Clear caches
        self.clear_kv_cache();
        if let Some(ref mut mtp) = self.mtp {
            mtp.clear_cache();
        }

        // Get prefix up to eval_position
        let prefix_len = eval_position + 1;
        let prefix = input.narrow(1, 0, prefix_len)?;

        // Run main model
        let (main_logits, hidden_states) = self.forward_with_hidden_states(&prefix, 0)?;
        let main_pred = main_logits.argmax(candle::D::Minus1)?.to_vec1::<u32>()?[0];

        let mut correct = 0usize;
        let mut current_hidden = hidden_states;
        let mut current_token = main_pred;
        let mut current_offset = prefix_len;

        for mtp_step in 0..num_mtp_steps {
            let target_pos = eval_position + 1 + mtp_step;
            let target_token = input_tokens[target_pos];

            let token_input = Tensor::new(&[current_token], &self.device)?.unsqueeze(0)?;

            let mtp = self.mtp.as_mut().unwrap();
            let mtp_hidden = mtp.forward_hidden(
                &token_input,
                &self.embed_tokens,
                &current_hidden,
                current_offset,
                self.dtype,
                mtp_step,
            )?;

            // CRITICAL: Must normalize before lm_head - forward_hidden returns pre-norm states
            let mtp_normalized = mtp.norm.forward(&mtp_hidden)?;
            let mtp_logits = self.lm_head.forward(&mtp_normalized)?.squeeze(1)?;
            let mtp_pred = mtp_logits.argmax(candle::D::Minus1)?.to_vec1::<u32>()?[0];

            if mtp_pred == target_token {
                correct += 1;
            }

            current_hidden = mtp_hidden;
            current_token = target_token;
            current_offset += 1;
        }

        Ok(correct as f32 / num_mtp_steps as f32)
    }

    /// Load MTP weights for multi-token prediction speculative decoding.
    ///
    /// This must be called after model loading if MTP weights are available in the GGUF.
    /// MTP enables speculative decoding where multiple tokens are predicted in parallel.
    ///
    /// # GPU Placement for Maximum Performance
    ///
    /// MTP tensors are always placed on the model's primary device (`self.device`).
    /// For maximum speculative decoding performance, this should be a CUDA GPU.
    ///
    /// When using [`DeviceOffloadMode`] to offload main model expert weights to CPU,
    /// the model's device remains GPU, so MTP will correctly stay on GPU. This ensures
    /// fast speculative token generation even when memory constraints require CPU
    /// offloading of the larger expert weights.
    ///
    /// A warning is logged if MTP is loaded to CPU when CUDA is available, as this
    /// may indicate suboptimal configuration.
    ///
    /// # MTP Weight Naming Convention (GGUF)
    ///
    /// - Top-level: `mtp.fc.weight`, `mtp.pre_fc_norm_embedding.weight`, `mtp.norm.weight`
    /// - Layers: `blk.{idx}.mtp.attn_q.weight`, `blk.{idx}.mtp.gate.weight`, etc.
    /// - Experts are packed: `blk.{idx}.mtp.ffn_gate_exps.weight` (3D tensor)
    pub fn load_mtp<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let mut file = std::fs::File::open(path.as_ref())?;
        let content = gguf_file::Content::read(&mut file)?;

        // MTP tensors must be on the same device as the model for compatibility with
        // embed_tokens and lm_head during forward passes.
        //
        // For maximum speculative decoding performance, MTP should be on GPU.
        // This is automatically ensured when:
        // - Model is loaded with a CUDA device (recommended)
        // - DeviceOffloadMode is used (self.device remains GPU, only experts go to CPU)
        //
        // Note: If the model is on CPU, MTP must also be on CPU for correctness.
        // A warning is logged in this case as it may indicate suboptimal configuration.
        let mtp_device = self.device.clone();

        if matches!(mtp_device, Device::Cuda(_)) {
            tracing::debug!(
                "Loading MTP weights to GPU ({:?}) for maximum performance",
                mtp_device
            );
        } else {
            // Check if CUDA is available but not being used
            if Device::cuda_if_available(0)
                .map(|d| matches!(d, Device::Cuda(_)))
                .unwrap_or(false)
            {
                tracing::warn!(
                    "MTP weights are being loaded to CPU but CUDA is available. \
                     For maximum speculative decoding performance, load the model with a CUDA device. \
                     Note: DeviceOffloadMode can be used to offload experts to CPU while keeping \
                     MTP on GPU."
                );
            } else {
                tracing::debug!("Loading MTP weights to CPU (no CUDA available)");
            }
        }

        let mut gg = Gguf::new(content, &mut file, mtp_device.clone());

        // Check if MTP weights exist (top-level FC layer)
        let fc = match gg.try_qmatmul("mtp.fc.weight")? {
            Some(fc) => fc,
            None => {
                return Err(candle::Error::Msg(
                    "MTP weights not found in GGUF file".to_string(),
                ));
            }
        };

        // Load pre-FC norms (top-level MTP weights) with Gemma-style +1 correction
        let pre_fc_norm_embedding = gg.rms_norm_gemma(
            "mtp.pre_fc_norm_embedding.weight",
            self.config.rms_norm_eps,
            self.dtype,
        )?;
        let pre_fc_norm_hidden = gg.rms_norm_gemma(
            "mtp.pre_fc_norm_hidden.weight",
            self.config.rms_norm_eps,
            self.dtype,
        )?;

        // Load final norm (GGUF naming: mtp.norm.weight) with Gemma-style +1 correction
        let norm = gg.rms_norm_gemma("mtp.norm.weight", self.config.rms_norm_eps, self.dtype)?;

        // Create rotary embedding for MTP on the MTP device (GPU)
        let rotary = Arc::new(RotaryEmbedding::new(
            self.dtype,
            self.config.head_dim,
            self.config.n_rot,
            self.config.max_position_embeddings,
            self.config.rope_freq_base,
            &mtp_device,
        )?);

        // Load MTP layers using GGUF naming: blk.{idx}.mtp.*
        // All MTP layers are placed on GPU for maximum speculative decoding performance
        let mut layers = Vec::new();
        for layer_idx in 0..4 {
            // Check up to 4 MTP layers
            let prefix = format!("blk.{}.mtp", layer_idx);

            // Check if this MTP layer exists by trying to read its input norm
            if gg
                .try_tensor(&format!("{}.in_norm.weight", prefix))?
                .is_none()
            {
                break;
            }

            let layer = MtpDecoderLayer::new(
                &mut gg,
                &prefix,
                &self.config,
                rotary.clone(),
                &mtp_device,
                layer_idx,
            )?;
            layers.push(layer);
        }

        if layers.is_empty() {
            return Err(candle::Error::Msg(
                "No MTP layers found in GGUF file".to_string(),
            ));
        }

        let num_mtp_layers = layers.len();

        self.mtp = Some(MtpWeights {
            fc,
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            layers,
            norm,
            hidden_size: self.config.hidden_size,
            num_mtp_layers,
        });

        Ok(())
    }

    /// Check if MTP weights are loaded
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Get the number of MTP layers (for determining max speculation depth)
    pub fn num_mtp_layers(&self) -> usize {
        self.mtp.as_ref().map(|m| m.num_mtp_layers).unwrap_or(0)
    }

    /// Clear all KV caches in the model (main model + MTP)
    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
        if let Some(ref mut mtp) = self.mtp {
            mtp.clear_cache();
        }
    }

    /// Truncate all KV caches in the model to a given sequence length.
    /// This is used for speculative decoding when draft tokens are rejected.
    /// For full attention layers, this truncates the cache to the specified length.
    /// For linear attention layers, this clears the recurrent state (can't truncate).
    pub fn truncate_cache(&mut self, new_len: usize) {
        for layer in &mut self.layers {
            layer.truncate_cache(new_len);
        }
    }

    /// Create a checkpoint of all layer caches/states for speculative decoding.
    /// This should be called BEFORE batched verification.
    /// For full attention layers: O(1) - just records seq_len pointers.
    /// For linear attention layers: copies to backup buffers (reuses allocation).
    pub fn checkpoint_cache(&mut self) -> Result<Vec<LayerCheckpoint>> {
        self.layers
            .iter_mut()
            .map(|layer| layer.checkpoint_cache())
            .collect()
    }

    /// Restore all layer caches/states from a checkpoint after rejection.
    /// This should be called when speculative tokens are rejected.
    /// For full attention layers: O(1) - just updates seq_len pointers.
    /// For linear attention layers: O(1) pointer swap to restored state.
    pub fn restore_cache(&mut self, checkpoints: Vec<LayerCheckpoint>) {
        for (layer, checkpoint) in self.layers.iter_mut().zip(checkpoints) {
            layer.restore_cache(checkpoint);
        }
    }

    /// Clear only the MTP cache (keeps main model's KV cache).
    /// Should be called before each MTP speculation to ensure clean state.
    pub fn clear_mtp_cache(&mut self) {
        if let Some(ref mut mtp) = self.mtp {
            mtp.clear_cache();
        }
    }

    /// Get the shared K/V cache from the last full attention layer.
    /// This is used for native MTP state sharing.
    fn get_last_full_attention_kv(&self) -> Option<(Tensor, Tensor)> {
        let debug = std::env::var("MTP_DEBUG").is_ok();
        let total_layers = self.layers.len();

        // Find the last full attention layer (iterate backwards)
        for (rev_idx, layer) in self.layers.iter().rev().enumerate() {
            let actual_idx = total_layers - 1 - rev_idx;
            if layer.is_full_attention() {
                if debug {
                    eprintln!(
                        "[get_last_full_attention_kv] Using layer {} (of {})",
                        actual_idx, total_layers
                    );
                }
                return layer.get_shared_kv();
            }
        }
        if debug {
            eprintln!("[get_last_full_attention_kv] No full attention layer found!");
        }
        None
    }

    /// Forward pass through MTP for speculative token prediction.
    ///
    /// Native MTP design:
    /// - Hidden states from main model are passed as input (this is the "state sharing")
    /// - MTP has its OWN tiny KV cache for its internal attention layer
    /// - MTP builds its cache incrementally for speculative tokens
    ///
    /// # Arguments
    /// * `input_id` - Single token ID to predict next token for [batch, 1]
    /// * `hidden_states` - Hidden states from main model [batch, 1, hidden_size]
    /// * `offset` - Position offset for MTP's internal attention
    /// * `spec_step_idx` - Speculative step index (determines which MTP layer to use)
    ///
    /// # Returns
    /// Logits for next token prediction [batch, vocab_size]
    pub fn forward_mtp(
        &mut self,
        input_id: &Tensor,
        hidden_states: &Tensor,
        offset: usize,
        spec_step_idx: usize,
    ) -> Result<Tensor> {
        let mtp = self
            .mtp
            .as_mut()
            .ok_or_else(|| candle::Error::Msg("MTP weights not loaded".to_string()))?;

        // MTP uses its own KV cache, hidden states are the shared input
        mtp.forward(
            input_id,
            &self.embed_tokens,
            hidden_states,
            &self.lm_head,
            offset,
            self.dtype,
            spec_step_idx,
        )
    }

    /// Forward pass through MTP returning both logits and hidden states.
    ///
    /// Native MTP design:
    /// - Hidden states from main model are passed as input (this is the "state sharing")
    /// - MTP has its OWN tiny KV cache for its internal attention layer
    /// - Returns hidden states for chaining multiple MTP steps
    ///
    /// # Arguments
    /// * `input_id` - Single token ID to predict next token for [batch, 1]
    /// * `hidden_states` - Hidden states from main model [batch, 1, hidden_size]
    /// * `offset` - Position offset for MTP's internal attention
    /// * `spec_step_idx` - Speculative step index (determines which MTP layer to use)
    ///
    /// # Returns
    /// (logits, hidden_states) - Logits for sampling and hidden states for next step
    pub fn forward_mtp_with_hidden(
        &mut self,
        input_id: &Tensor,
        hidden_states: &Tensor,
        offset: usize,
        spec_step_idx: usize,
    ) -> Result<(Tensor, Tensor)> {
        let mtp = self
            .mtp
            .as_mut()
            .ok_or_else(|| candle::Error::Msg("MTP weights not loaded".to_string()))?;

        // MTP uses its own KV cache, hidden states are the shared input
        let mtp_hidden = mtp.forward_hidden(
            input_id,
            &self.embed_tokens,
            hidden_states,
            offset,
            self.dtype,
            spec_step_idx,
        )?;

        // Normalize before applying lm_head for logits
        let normalized = mtp.norm.forward(&mtp_hidden)?;
        let logits = self.lm_head.forward(&normalized)?.squeeze(1)?;

        // Return logits and PRE-NORM hidden states for next MTP step
        Ok((logits, mtp_hidden))
    }

    /// Run speculative decoding with MTP.
    ///
    /// This generates `num_speculative` tokens speculatively using MTP, then verifies
    /// them against the main model. Accepted tokens are returned along with their count.
    ///
    /// # Arguments
    /// * `input` - Input token IDs [batch, seq_len]
    /// * `offset` - Position offset for attention
    /// * `num_speculative` - Number of tokens to speculate (typically 1-4)
    /// * `logits_processor` - Function to sample from logits
    ///
    /// # Returns
    /// (all_logits, accepted_tokens, num_accepted)
    /// - all_logits: Logits from both speculation and verification [num_tokens, vocab_size]
    /// - accepted_tokens: Vector of accepted token IDs
    /// - num_accepted: Number of accepted speculative tokens (0 to num_speculative)
    pub fn forward_speculative<F>(
        &mut self,
        input: &Tensor,
        offset: usize,
        num_speculative: usize,
        mut sample_fn: F,
    ) -> Result<(Tensor, Vec<u32>, usize)>
    where
        F: FnMut(&Tensor) -> Result<u32>,
    {
        if self.mtp.is_none() {
            // No MTP, fall back to regular forward
            let logits = self.forward(input, offset)?;
            let token = sample_fn(&logits)?;
            return Ok((logits, vec![token], 0));
        }

        // Step 1: Run main model to get first logits and hidden states
        let (main_logits, hidden_states) = self.forward_with_hidden_states(input, offset)?;
        let first_token = sample_fn(&main_logits)?;

        // Step 2: Use MTP to speculatively predict next tokens
        let mut speculative_tokens = vec![first_token];
        let mut speculative_logits = vec![main_logits.clone()];
        let mut current_hidden = hidden_states;
        let mut current_token = first_token;
        let mut current_offset = offset + input.dims()[1];

        for spec_step_idx in 0..num_speculative {
            // Create input tensor for the current token
            let token_input = Tensor::new(&[current_token], &self.device)?.unsqueeze(0)?;

            // Get MTP prediction (pass spec_step_idx to select the correct layer)
            let mtp = self.mtp.as_mut().unwrap();
            let mtp_hidden = mtp.forward_hidden(
                &token_input,
                &self.embed_tokens,
                &current_hidden,
                current_offset,
                self.dtype,
                spec_step_idx,
            )?;

            // Get logits from MTP hidden states
            // CRITICAL: Must normalize before lm_head - forward_hidden returns pre-norm states
            let mtp_normalized = mtp.norm.forward(&mtp_hidden)?;
            let mtp_logits = self.lm_head.forward(&mtp_normalized)?.squeeze(1)?;
            let next_token = sample_fn(&mtp_logits)?;

            speculative_tokens.push(next_token);
            speculative_logits.push(mtp_logits);
            current_hidden = mtp_hidden;
            current_token = next_token;
            current_offset += 1;
        }

        // Step 3: Verify speculative tokens with main model
        // Concatenate speculative tokens and run through main model
        let spec_tokens: Vec<u32> = speculative_tokens.clone();
        let verify_input = Tensor::new(spec_tokens.as_slice(), &self.device)?.unsqueeze(0)?;

        // Clear caches to restart from the original offset
        self.clear_cache();
        if let Some(ref mut mtp) = self.mtp {
            mtp.clear_cache();
        }

        // Run verification through main model with all tokens
        let full_input = if input.dims()[1] > 0 {
            // Concatenate original input with speculative tokens
            let input_2d = input.squeeze(0)?;
            let verify_2d = verify_input.squeeze(0)?;
            Tensor::cat(&[&input_2d, &verify_2d], 0)?.unsqueeze(0)?
        } else {
            verify_input
        };

        let verify_logits = self.forward_all_positions(&full_input, offset)?;

        // Step 4: Verify speculative tokens against main model predictions
        // For each position, check if the speculative token matches what the main model would predict
        let mut num_accepted = 0;
        let mut accepted_tokens = Vec::new();

        // The verify_logits has shape [batch, full_seq_len, vocab]
        // We need to check positions starting from the end of the original input
        let original_len = input.dims()[1];
        let verify_logits_2d = verify_logits.squeeze(0)?; // [full_seq_len, vocab]

        for i in 0..num_speculative {
            // Position in verify_logits where we check the prediction for speculative_tokens[i]
            // At position (original_len + i - 1), the model predicts what should be at (original_len + i)
            let check_pos = original_len + i;
            if check_pos >= verify_logits_2d.dim(0)? {
                break;
            }

            // Get the main model's prediction at this position (greedy)
            let logits_at_pos = verify_logits_2d.i(check_pos - 1)?;
            let main_pred = logits_at_pos.argmax(0)?.to_scalar::<u32>()?;

            // Check if MTP's speculative token matches
            let spec_token = speculative_tokens[i + 1]; // +1 because speculative_tokens[0] is the first token from main model
            if main_pred == spec_token {
                num_accepted += 1;
                accepted_tokens.push(spec_token);
            } else {
                // First mismatch - stop accepting
                break;
            }
        }

        // Include the first token (from main model) in accepted tokens
        let mut final_accepted = vec![first_token];
        final_accepted.extend(accepted_tokens);

        // Stack all logits
        let all_logits = Tensor::stack(&speculative_logits, 0)?;

        Ok((all_logits, final_accepted, num_accepted))
    }

    pub fn forward_training(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let _enter = self.span.enter();
        let (b, l) = input_ids.dims2()?;
        let mut h = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;

        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, seqlen_offset)?)
        };

        let mut all_router_stats = Vec::new();
        for layer in &mut self.layers {
            let (output, router_stats) =
                layer.forward_with_stats(&h, causal_mask.as_ref(), seqlen_offset)?;
            h = output;
            all_router_stats.push(router_stats);
        }

        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        let logits = if let Some(scales) = &self.custom_lm_head_scales {
            self.lm_head.forward_with_scales(&h, scales)?
        } else {
            self.lm_head.forward(&h)?
        };

        Ok((logits, all_router_stats))
    }

    /// Compute Z-loss from router statistics for MoE load balancing.
    ///
    /// Z-loss penalizes large router logits to prevent router collapse and
    /// encourage more uniform expert utilization. This is especially important
    /// during fine-tuning when scale perturbations can push logits to extremes.
    ///
    /// # Arguments
    /// * `router_stats` - Vector of (router_logits, selected_experts) tuples from forward_training
    /// * `z_loss_weight` - Weight for Z-loss term (typically 0.001)
    ///
    /// # Returns
    /// Scalar Z-loss tensor
    pub fn compute_z_loss(
        &self,
        router_stats: &[(Tensor, Tensor)],
        z_loss_weight: f64,
    ) -> Result<Tensor> {
        if router_stats.is_empty() {
            return Tensor::new(0.0f32, &self.device);
        }

        let mut total_z_loss: Option<Tensor> = None;

        for (router_logits, _) in router_stats {
            // Flatten to [total_tokens, num_experts]
            let router_logits = router_logits.flatten(0, 1)?;

            // Compute log-sum-exp for each token
            let max_val = router_logits.max(D::Minus1)?;
            let max_val_keepdim = max_val.unsqueeze(D::Minus1)?;
            let exp_shifted = router_logits.broadcast_sub(&max_val_keepdim)?.exp()?;
            let sum_exp = exp_shifted.sum(D::Minus1)?;
            let log_sum_exp = (max_val + sum_exp.log()?)?;

            // Z-loss = mean(log_sum_exp^2)
            let layer_z_loss = (&log_sum_exp * &log_sum_exp)?.mean_all()?;

            total_z_loss = Some(match total_z_loss {
                Some(acc) => (acc + layer_z_loss)?,
                None => layer_z_loss,
            });
        }

        let z_loss = total_z_loss.unwrap_or_else(|| Tensor::new(0.0f32, &self.device).unwrap());
        let num_layers = router_stats.len() as f64;
        (z_loss / num_layers)? * z_loss_weight
    }

    /// Compute load balance loss from router statistics for MoE.
    ///
    /// Load balance loss encourages uniform expert utilization across tokens.
    /// It penalizes routing patterns where some experts are used much more
    /// frequently than others.
    ///
    /// # Arguments
    /// * `router_stats` - Vector of (router_logits, selected_experts) tuples from forward_training
    /// * `lb_loss_weight` - Weight for load balance loss term (typically 0.01)
    ///
    /// # Returns
    /// Scalar load balance loss tensor
    pub fn compute_load_balance_loss(
        &self,
        router_stats: &[(Tensor, Tensor)],
        lb_loss_weight: f64,
    ) -> Result<Tensor> {
        if router_stats.is_empty() {
            return Tensor::new(0.0f32, &self.device);
        }

        let num_experts = self.num_experts();
        let mut total_lb_loss: Option<Tensor> = None;

        for (router_logits, selected_experts) in router_stats {
            // Flatten to [total_tokens, num_experts] and [total_tokens, k]
            let router_logits = router_logits.flatten(0, 1)?;
            let selected_experts = selected_experts.flatten(0, 1)?;

            // Compute routing probabilities
            let router_probs = candle_nn::ops::softmax_last_dim(&router_logits)?;

            // f_i = fraction of tokens routed to expert i
            let selected_flat = selected_experts.to_vec2::<u32>()?;
            let mut expert_counts = vec![0.0f32; num_experts];
            for row in &selected_flat {
                for &expert_idx in row {
                    if (expert_idx as usize) < num_experts {
                        expert_counts[expert_idx as usize] += 1.0;
                    }
                }
            }
            let total_selections =
                selected_flat.len() * selected_flat.first().map_or(1, |r| r.len());
            let f_i = Tensor::from_vec(
                expert_counts
                    .iter()
                    .map(|c| c / total_selections as f32)
                    .collect::<Vec<_>>(),
                num_experts,
                &self.device,
            )?;

            // p_i = mean probability assigned to expert i
            let p_i = router_probs.mean(0)?;

            // Load balance loss = num_experts * sum(f_i * p_i)
            let lb_loss = (&f_i * &p_i)?.sum_all()?;
            let lb_loss = (lb_loss * (num_experts as f64))?;

            total_lb_loss = Some(match total_lb_loss {
                Some(acc) => (acc + lb_loss)?,
                None => lb_loss,
            });
        }

        let lb_loss = total_lb_loss.unwrap_or_else(|| Tensor::new(0.0f32, &self.device).unwrap());
        let num_layers = router_stats.len() as f64;
        (lb_loss / num_layers)? * lb_loss_weight
    }

    /// Compute combined auxiliary loss for MoE training.
    ///
    /// This combines Z-loss and load balance loss into a single auxiliary loss term
    /// that can be added to the main training objective.
    ///
    /// # Arguments
    /// * `router_stats` - Vector of (router_logits, selected_experts) tuples from forward_training
    /// * `z_loss_weight` - Weight for Z-loss term (typically 0.001)
    /// * `lb_loss_weight` - Weight for load balance loss term (typically 0.01)
    ///
    /// # Returns
    /// Scalar auxiliary loss tensor (z_loss + lb_loss)
    pub fn compute_auxiliary_loss(
        &self,
        router_stats: &[(Tensor, Tensor)],
        z_loss_weight: f64,
        lb_loss_weight: f64,
    ) -> Result<Tensor> {
        let z_loss = self.compute_z_loss(router_stats, z_loss_weight)?;
        let lb_loss = self.compute_load_balance_loss(router_stats, lb_loss_weight)?;
        z_loss + lb_loss
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }

    /// Save the current KV cache state for prefix caching.
    ///
    /// This creates a deep copy of all layer caches that can be restored later
    /// to resume generation from a previously computed prefix, avoiding redundant
    /// recomputation of the KV cache for shared conversation context.
    ///
    /// # Arguments
    /// * `prefix_tokens` - The token IDs that this cache was computed for
    ///
    /// # Returns
    /// A `PrefixCache` containing the token prefix and all layer cache states.
    pub fn save_prefix_cache(&self, prefix_tokens: Vec<u32>) -> PrefixCache {
        let layer_caches = self
            .layers
            .iter()
            .map(|layer| layer.save_cache_for_prefix())
            .collect();

        PrefixCache {
            prefix_tokens,
            layer_caches,
        }
    }

    /// Restore KV cache state from a prefix cache.
    ///
    /// This restores the cached KV state from a previous generation, allowing
    /// the model to continue from where the prefix left off without recomputing
    /// the attention keys/values for the shared prefix tokens.
    ///
    /// # Arguments
    /// * `cache` - The prefix cache to restore from
    ///
    /// # Returns
    /// The number of tokens in the restored prefix (cache sequence length).
    pub fn restore_prefix_cache(&mut self, cache: &PrefixCache) -> Result<usize> {
        if cache.layer_caches.len() != self.layers.len() {
            candle::bail!(
                "Prefix cache layer count mismatch: {} vs {}",
                cache.layer_caches.len(),
                self.layers.len()
            );
        }

        for (layer, entry) in self.layers.iter_mut().zip(cache.layer_caches.iter()) {
            layer.restore_cache_from_prefix(entry)?;
        }

        Ok(cache.prefix_tokens.len())
    }

    /// Get the current cache sequence length (from the first full attention layer).
    pub fn cache_seq_len(&self) -> usize {
        for layer in &self.layers {
            if let AttentionLayer::Full(attn) = &layer.attn {
                return attn.cache_seq_len();
            }
        }
        0
    }

    pub fn all_qtensors(&self) -> Vec<(String, &QTensor)> {
        let mut tensors = Vec::new();

        if let Some(qt) = self.lm_head.qtensor() {
            tensors.push(("output.weight".to_string(), qt));
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let prefix = format!("blk.{}", layer_idx);

            match &layer.attn {
                AttentionLayer::Full(attn) => {
                    if let Some(qt) = attn.wq.qtensor() {
                        tensors.push((format!("{}.attn_q.weight", prefix), qt));
                    }
                    if let Some(qt) = attn.wk.qtensor() {
                        tensors.push((format!("{}.attn_k.weight", prefix), qt));
                    }
                    if let Some(qt) = attn.wv.qtensor() {
                        tensors.push((format!("{}.attn_v.weight", prefix), qt));
                    }
                    if let Some(qt) = attn.wo.qtensor() {
                        tensors.push((format!("{}.attn_output.weight", prefix), qt));
                    }
                }
                AttentionLayer::Linear(attn) => {
                    if let Some(qt) = attn.ssm_in.qtensor() {
                        tensors.push((format!("{}.ssm_in", prefix), qt));
                    }
                    if let Some(qt) = attn.ssm_beta_alpha.qtensor() {
                        tensors.push((format!("{}.ssm_ba", prefix), qt));
                    }
                    if let Some(qt) = attn.ssm_out.qtensor() {
                        tensors.push((format!("{}.ssm_out", prefix), qt));
                    }
                }
            }

            if let Some(qt) = layer.moe_block.gate.qtensor() {
                tensors.push((format!("{}.ffn_gate_inp.weight", prefix), qt));
            }

            tensors.push((
                format!("{}.ffn_gate_exps.weight", prefix),
                &layer.moe_block.experts.gate_exps,
            ));
            tensors.push((
                format!("{}.ffn_up_exps.weight", prefix),
                &layer.moe_block.experts.up_exps,
            ));
            tensors.push((
                format!("{}.ffn_down_exps.weight", prefix),
                &layer.moe_block.experts.down_exps,
            ));

            if let Some(shared) = &layer.moe_block.shared_expert {
                if let Some(qt) = shared.gate_proj.qtensor() {
                    tensors.push((format!("{}.ffn_gate_shexp.weight", prefix), qt));
                }
                if let Some(qt) = shared.up_proj.qtensor() {
                    tensors.push((format!("{}.ffn_up_shexp.weight", prefix), qt));
                }
                if let Some(qt) = shared.down_proj.qtensor() {
                    tensors.push((format!("{}.ffn_down_shexp.weight", prefix), qt));
                }
            }
        }

        tensors
    }

    fn block_count(qtensor: &QTensor) -> usize {
        qtensor.shape().elem_count() / qtensor.dtype().block_size()
    }

    pub fn all_qtensors_with_info(&self) -> Vec<(String, &QTensor, usize)> {
        let mut tensors = Vec::new();

        if let Some(qt) = self.lm_head.qtensor() {
            tensors.push(("lm_head".to_string(), qt, Self::block_count(qt)));
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layer_{}", layer_idx);

            match &layer.attn {
                AttentionLayer::Full(attn) => {
                    if let Some(qt) = attn.wq.qtensor() {
                        tensors.push((format!("{}_attn_q", prefix), qt, Self::block_count(qt)));
                    }
                    if let Some(qt) = attn.wk.qtensor() {
                        tensors.push((format!("{}_attn_k", prefix), qt, Self::block_count(qt)));
                    }
                    if let Some(qt) = attn.wv.qtensor() {
                        tensors.push((format!("{}_attn_v", prefix), qt, Self::block_count(qt)));
                    }
                    if let Some(qt) = attn.wo.qtensor() {
                        tensors.push((format!("{}_attn_o", prefix), qt, Self::block_count(qt)));
                    }
                }
                AttentionLayer::Linear(attn) => {
                    if let Some(qt) = attn.ssm_in.qtensor() {
                        tensors.push((format!("{}_ssm_in", prefix), qt, Self::block_count(qt)));
                    }
                    if let Some(qt) = attn.ssm_beta_alpha.qtensor() {
                        tensors.push((format!("{}_ssm_ba", prefix), qt, Self::block_count(qt)));
                    }
                    if let Some(qt) = attn.ssm_out.qtensor() {
                        tensors.push((format!("{}_ssm_out", prefix), qt, Self::block_count(qt)));
                    }
                }
            }

            if let Some(qt) = layer.moe_block.gate.qtensor() {
                tensors.push((format!("{}_router_gate", prefix), qt, Self::block_count(qt)));
            }

            let gate_exps = &layer.moe_block.experts.gate_exps;
            let up_exps = &layer.moe_block.experts.up_exps;
            let down_exps = &layer.moe_block.experts.down_exps;

            tensors.push((
                format!("{}_gate_exps", prefix),
                gate_exps,
                Self::block_count(gate_exps),
            ));
            tensors.push((
                format!("{}_up_exps", prefix),
                up_exps,
                Self::block_count(up_exps),
            ));
            tensors.push((
                format!("{}_down_exps", prefix),
                down_exps,
                Self::block_count(down_exps),
            ));

            if let Some(shared) = &layer.moe_block.shared_expert {
                if let Some(qt) = shared.gate_proj.qtensor() {
                    tensors.push((format!("{}_shared_gate", prefix), qt, Self::block_count(qt)));
                }
                if let Some(qt) = shared.up_proj.qtensor() {
                    tensors.push((format!("{}_shared_up", prefix), qt, Self::block_count(qt)));
                }
                if let Some(qt) = shared.down_proj.qtensor() {
                    tensors.push((format!("{}_shared_down", prefix), qt, Self::block_count(qt)));
                }
            }
        }

        tensors
    }

    pub fn set_all_custom_scales(
        &mut self,
        scales_map: std::collections::HashMap<String, Tensor>,
    ) -> Result<()> {
        if let Some(scales) = scales_map.get("lm_head") {
            self.custom_lm_head_scales = Some(scales.clone());
        }

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("layer_{}", layer_idx);

            match &mut layer.attn {
                AttentionLayer::Full(attn) => {
                    if let Some(scales) = scales_map.get(&format!("{}_attn_q", prefix)) {
                        attn.custom_q_scales = Some(scales.clone());
                    }
                    if let Some(scales) = scales_map.get(&format!("{}_attn_k", prefix)) {
                        attn.custom_k_scales = Some(scales.clone());
                    }
                    if let Some(scales) = scales_map.get(&format!("{}_attn_v", prefix)) {
                        attn.custom_v_scales = Some(scales.clone());
                    }
                    if let Some(scales) = scales_map.get(&format!("{}_attn_o", prefix)) {
                        attn.custom_o_scales = Some(scales.clone());
                    }
                }
                AttentionLayer::Linear(attn) => {
                    if let Some(scales) = scales_map.get(&format!("{}_ssm_in", prefix)) {
                        attn.custom_ssm_in_scales = Some(scales.clone());
                    }
                    if let Some(scales) = scales_map.get(&format!("{}_ssm_ba", prefix)) {
                        attn.custom_ssm_ba_scales = Some(scales.clone());
                    }
                    if let Some(scales) = scales_map.get(&format!("{}_ssm_out", prefix)) {
                        attn.custom_ssm_out_scales = Some(scales.clone());
                    }
                }
            }

            if let Some(scales) = scales_map.get(&format!("{}_router_gate", prefix)) {
                layer.moe_block.custom_gate_scales = Some(scales.clone());
            }

            if let Some(gate_scales) = scales_map.get(&format!("{}_gate_exps", prefix)) {
                if let (Some(up_scales), Some(down_scales)) = (
                    scales_map.get(&format!("{}_up_exps", prefix)),
                    scales_map.get(&format!("{}_down_exps", prefix)),
                ) {
                    layer.moe_block.experts.set_block_multipliers(
                        gate_scales.clone(),
                        up_scales.clone(),
                        down_scales.clone(),
                    )?;
                }
            }

            if let Some(shared) = &mut layer.moe_block.shared_expert {
                if let Some(scales) = scales_map.get(&format!("{}_shared_gate", prefix)) {
                    shared.custom_gate_scales = Some(scales.clone());
                }
                if let Some(scales) = scales_map.get(&format!("{}_shared_up", prefix)) {
                    shared.custom_up_scales = Some(scales.clone());
                }
                if let Some(scales) = scales_map.get(&format!("{}_shared_down", prefix)) {
                    shared.custom_down_scales = Some(scales.clone());
                }
            }
        }

        Ok(())
    }

    pub fn custom_scale_map_cpu(&self) -> Result<std::collections::HashMap<String, Tensor>> {
        let mut scales = std::collections::HashMap::new();

        if let Some(ref lm_scales) = self.custom_lm_head_scales {
            scales.insert(
                "output.weight".to_string(),
                lm_scales.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
            );
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let gguf_prefix = format!("blk.{}", layer_idx);

            match &layer.attn {
                AttentionLayer::Full(attn) => {
                    if let Some(ref s) = attn.custom_q_scales {
                        scales.insert(
                            format!("{}.attn_q.weight", gguf_prefix),
                            s.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
                        );
                    }
                    if let Some(ref s) = attn.custom_k_scales {
                        scales.insert(
                            format!("{}.attn_k.weight", gguf_prefix),
                            s.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
                        );
                    }
                    if let Some(ref s) = attn.custom_v_scales {
                        scales.insert(
                            format!("{}.attn_v.weight", gguf_prefix),
                            s.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
                        );
                    }
                    if let Some(ref s) = attn.custom_o_scales {
                        scales.insert(
                            format!("{}.attn_output.weight", gguf_prefix),
                            s.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
                        );
                    }
                }
                AttentionLayer::Linear(attn) => {
                    if let Some(ref s) = attn.custom_ssm_in_scales {
                        scales.insert(
                            format!("{}.ssm_in", gguf_prefix),
                            s.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
                        );
                    }
                    if let Some(ref s) = attn.custom_ssm_ba_scales {
                        scales.insert(
                            format!("{}.ssm_ba", gguf_prefix),
                            s.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
                        );
                    }
                    if let Some(ref s) = attn.custom_ssm_out_scales {
                        scales.insert(
                            format!("{}.ssm_out", gguf_prefix),
                            s.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
                        );
                    }
                }
            }

            if let Some(ref s) = layer.moe_block.custom_gate_scales {
                scales.insert(
                    format!("{}.ffn_gate_inp.weight", gguf_prefix),
                    s.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
                );
            }

            if let Some(ref s) = layer.moe_block.experts.custom_gate_block_mults {
                scales.insert(
                    format!("{}.ffn_gate_exps.weight", gguf_prefix),
                    s.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
                );
            }
            if let Some(ref s) = layer.moe_block.experts.custom_up_block_mults {
                scales.insert(
                    format!("{}.ffn_up_exps.weight", gguf_prefix),
                    s.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
                );
            }
            if let Some(ref s) = layer.moe_block.experts.custom_down_block_mults {
                scales.insert(
                    format!("{}.ffn_down_exps.weight", gguf_prefix),
                    s.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
                );
            }

            if let Some(shared) = &layer.moe_block.shared_expert {
                if let Some(ref s) = shared.custom_gate_scales {
                    scales.insert(
                        format!("{}.ffn_gate_shexp.weight", gguf_prefix),
                        s.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
                    );
                }
                if let Some(ref s) = shared.custom_up_scales {
                    scales.insert(
                        format!("{}.ffn_up_shexp.weight", gguf_prefix),
                        s.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
                    );
                }
                if let Some(ref s) = shared.custom_down_scales {
                    scales.insert(
                        format!("{}.ffn_down_shexp.weight", gguf_prefix),
                        s.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
                    );
                }
            }
        }

        Ok(scales)
    }

    pub fn num_experts(&self) -> usize {
        self.layers
            .first()
            .map_or(0, |layer| layer.moe_block.num_experts)
    }

    pub fn num_experts_per_token(&self) -> usize {
        self.layers
            .first()
            .map_or(0, |layer| layer.moe_block.num_experts_per_tok)
    }

    /// Get a reference to the embedding layer
    pub fn embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }

    /// Get a reference to the LM head
    pub fn lm_head(&self) -> &QMatMul {
        &self.lm_head
    }

    /// Get the model's data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get layer configurations for EGGROLL optimization
    /// Returns: Vec<(name, (out_features, in_features), num_scale_blocks)>
    pub fn eggroll_layer_configs(&self) -> Vec<(String, (usize, usize), Option<usize>)> {
        let mut configs = Vec::new();

        // Output layer (lm_head)
        if let Some(qt) = self.lm_head.qtensor() {
            let shape = qt.shape();
            let (out_features, in_features) = (shape.dims()[0], shape.dims()[1]);
            let num_blocks = if supports_scale_modification(qt.dtype()) {
                Some(shape.elem_count() / qt.dtype().block_size())
            } else {
                None
            };
            configs.push((
                "output.weight".to_string(),
                (out_features, in_features),
                num_blocks,
            ));
        }

        // Layer weights
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let prefix = format!("blk.{}", layer_idx);

            // Attention layers (full or linear)
            match &layer.attn {
                AttentionLayer::Full(attn) => {
                    // Q projection
                    if let Some(qt) = attn.wq.qtensor() {
                        let shape = qt.shape();
                        let (out_features, in_features) = (shape.dims()[0], shape.dims()[1]);
                        let num_blocks = if supports_scale_modification(qt.dtype()) {
                            Some(shape.elem_count() / qt.dtype().block_size())
                        } else {
                            None
                        };
                        configs.push((
                            format!("{}.attn_q", prefix),
                            (out_features, in_features),
                            num_blocks,
                        ));
                    }
                    // K projection
                    if let Some(qt) = attn.wk.qtensor() {
                        let shape = qt.shape();
                        let (out_features, in_features) = (shape.dims()[0], shape.dims()[1]);
                        let num_blocks = if supports_scale_modification(qt.dtype()) {
                            Some(shape.elem_count() / qt.dtype().block_size())
                        } else {
                            None
                        };
                        configs.push((
                            format!("{}.attn_k", prefix),
                            (out_features, in_features),
                            num_blocks,
                        ));
                    }
                    // V projection
                    if let Some(qt) = attn.wv.qtensor() {
                        let shape = qt.shape();
                        let (out_features, in_features) = (shape.dims()[0], shape.dims()[1]);
                        let num_blocks = if supports_scale_modification(qt.dtype()) {
                            Some(shape.elem_count() / qt.dtype().block_size())
                        } else {
                            None
                        };
                        configs.push((
                            format!("{}.attn_v", prefix),
                            (out_features, in_features),
                            num_blocks,
                        ));
                    }
                    // O projection
                    if let Some(qt) = attn.wo.qtensor() {
                        let shape = qt.shape();
                        let (out_features, in_features) = (shape.dims()[0], shape.dims()[1]);
                        let num_blocks = if supports_scale_modification(qt.dtype()) {
                            Some(shape.elem_count() / qt.dtype().block_size())
                        } else {
                            None
                        };
                        configs.push((
                            format!("{}.attn_o", prefix),
                            (out_features, in_features),
                            num_blocks,
                        ));
                    }
                }
                AttentionLayer::Linear(attn) => {
                    // SSM input projection
                    if let Some(qt) = attn.ssm_in.qtensor() {
                        let shape = qt.shape();
                        let (out_features, in_features) = (shape.dims()[0], shape.dims()[1]);
                        let num_blocks = if supports_scale_modification(qt.dtype()) {
                            Some(shape.elem_count() / qt.dtype().block_size())
                        } else {
                            None
                        };
                        configs.push((
                            format!("{}.ssm_in", prefix),
                            (out_features, in_features),
                            num_blocks,
                        ));
                    }
                    // SSM beta/alpha projection
                    if let Some(qt) = attn.ssm_beta_alpha.qtensor() {
                        let shape = qt.shape();
                        let (out_features, in_features) = (shape.dims()[0], shape.dims()[1]);
                        let num_blocks = if supports_scale_modification(qt.dtype()) {
                            Some(shape.elem_count() / qt.dtype().block_size())
                        } else {
                            None
                        };
                        configs.push((
                            format!("{}.ssm_ba", prefix),
                            (out_features, in_features),
                            num_blocks,
                        ));
                    }
                    // SSM output projection
                    if let Some(qt) = attn.ssm_out.qtensor() {
                        let shape = qt.shape();
                        let (out_features, in_features) = (shape.dims()[0], shape.dims()[1]);
                        let num_blocks = if supports_scale_modification(qt.dtype()) {
                            Some(shape.elem_count() / qt.dtype().block_size())
                        } else {
                            None
                        };
                        configs.push((
                            format!("{}.ssm_out", prefix),
                            (out_features, in_features),
                            num_blocks,
                        ));
                    }
                }
            }

            // Router gate
            if let Some(qt) = layer.moe_block.gate.qtensor() {
                let shape = qt.shape();
                let (out_features, in_features) = (shape.dims()[0], shape.dims()[1]);
                let num_blocks = if supports_scale_modification(qt.dtype()) {
                    Some(shape.elem_count() / qt.dtype().block_size())
                } else {
                    None
                };
                configs.push((
                    format!("{}.ffn_gate_inp", prefix),
                    (out_features, in_features),
                    num_blocks,
                ));
            }

            // MoE expert weights (batched tensors)
            {
                let gate_shape = layer.moe_block.experts.gate_exps.shape();
                let num_experts = gate_shape.dims()[0];
                let out_features = gate_shape.dims()[1];
                let in_features = gate_shape.dims()[2];
                let dtype = layer.moe_block.experts.gate_exps.dtype();
                let num_blocks = if supports_scale_modification(dtype) {
                    Some(gate_shape.elem_count() / dtype.block_size())
                } else {
                    None
                };
                configs.push((
                    format!("{}.ffn_gate_exps", prefix),
                    (num_experts * out_features, in_features),
                    num_blocks,
                ));
            }
            {
                let up_shape = layer.moe_block.experts.up_exps.shape();
                let num_experts = up_shape.dims()[0];
                let out_features = up_shape.dims()[1];
                let in_features = up_shape.dims()[2];
                let dtype = layer.moe_block.experts.up_exps.dtype();
                let num_blocks = if supports_scale_modification(dtype) {
                    Some(up_shape.elem_count() / dtype.block_size())
                } else {
                    None
                };
                configs.push((
                    format!("{}.ffn_up_exps", prefix),
                    (num_experts * out_features, in_features),
                    num_blocks,
                ));
            }
            {
                let down_shape = layer.moe_block.experts.down_exps.shape();
                let num_experts = down_shape.dims()[0];
                let out_features = down_shape.dims()[1];
                let in_features = down_shape.dims()[2];
                let dtype = layer.moe_block.experts.down_exps.dtype();
                let num_blocks = if supports_scale_modification(dtype) {
                    Some(down_shape.elem_count() / dtype.block_size())
                } else {
                    None
                };
                configs.push((
                    format!("{}.ffn_down_exps", prefix),
                    (num_experts * out_features, in_features),
                    num_blocks,
                ));
            }

            // Shared expert (if present)
            if let Some(shared) = &layer.moe_block.shared_expert {
                if let Some(qt) = shared.gate_proj.qtensor() {
                    let shape = qt.shape();
                    let (out_features, in_features) = (shape.dims()[0], shape.dims()[1]);
                    let num_blocks = if supports_scale_modification(qt.dtype()) {
                        Some(shape.elem_count() / qt.dtype().block_size())
                    } else {
                        None
                    };
                    configs.push((
                        format!("{}.ffn_gate_shexp", prefix),
                        (out_features, in_features),
                        num_blocks,
                    ));
                }
                if let Some(qt) = shared.up_proj.qtensor() {
                    let shape = qt.shape();
                    let (out_features, in_features) = (shape.dims()[0], shape.dims()[1]);
                    let num_blocks = if supports_scale_modification(qt.dtype()) {
                        Some(shape.elem_count() / qt.dtype().block_size())
                    } else {
                        None
                    };
                    configs.push((
                        format!("{}.ffn_up_shexp", prefix),
                        (out_features, in_features),
                        num_blocks,
                    ));
                }
                if let Some(qt) = shared.down_proj.qtensor() {
                    let shape = qt.shape();
                    let (out_features, in_features) = (shape.dims()[0], shape.dims()[1]);
                    let num_blocks = if supports_scale_modification(qt.dtype()) {
                        Some(shape.elem_count() / qt.dtype().block_size())
                    } else {
                        None
                    };
                    configs.push((
                        format!("{}.ffn_down_shexp", prefix),
                        (out_features, in_features),
                        num_blocks,
                    ));
                }
            }
        }

        configs
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Set LoRA adapters for attention layers
    /// lora_map: layer name -> (A tensor, B tensor)
    /// sigma: perturbation magnitude (σ/√r scaling is applied internally)
    pub fn set_lora_adapters(
        &mut self,
        lora_map: &HashMap<String, (Tensor, Tensor)>,
        sigma: f64,
    ) -> Result<()> {
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("blk.{}", layer_idx);

            match &mut layer.attn {
                AttentionLayer::Full(attn) => {
                    attn.lora_sigma = sigma;

                    if let Some((a, b)) = lora_map.get(&format!("{}.attn_q", prefix)) {
                        attn.lora_q = Some(LoraAdapter {
                            a: a.clone(),
                            b: b.clone(),
                        });
                    }
                    if let Some((a, b)) = lora_map.get(&format!("{}.attn_k", prefix)) {
                        attn.lora_k = Some(LoraAdapter {
                            a: a.clone(),
                            b: b.clone(),
                        });
                    }
                    if let Some((a, b)) = lora_map.get(&format!("{}.attn_v", prefix)) {
                        attn.lora_v = Some(LoraAdapter {
                            a: a.clone(),
                            b: b.clone(),
                        });
                    }
                    if let Some((a, b)) = lora_map.get(&format!("{}.attn_o", prefix)) {
                        attn.lora_o = Some(LoraAdapter {
                            a: a.clone(),
                            b: b.clone(),
                        });
                    }
                }
                AttentionLayer::Linear(attn) => {
                    attn.lora_sigma = sigma;

                    if let Some((a, b)) = lora_map.get(&format!("{}.ssm_in", prefix)) {
                        attn.lora_ssm_in = Some(LoraAdapter {
                            a: a.clone(),
                            b: b.clone(),
                        });
                    }
                    if let Some((a, b)) = lora_map.get(&format!("{}.ssm_ba", prefix)) {
                        attn.lora_ssm_ba = Some(LoraAdapter {
                            a: a.clone(),
                            b: b.clone(),
                        });
                    }
                    if let Some((a, b)) = lora_map.get(&format!("{}.ssm_out", prefix)) {
                        attn.lora_ssm_out = Some(LoraAdapter {
                            a: a.clone(),
                            b: b.clone(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Clear all LoRA adapters
    pub fn clear_lora_adapters(&mut self) {
        for layer in self.layers.iter_mut() {
            match &mut layer.attn {
                AttentionLayer::Full(attn) => {
                    attn.lora_q = None;
                    attn.lora_k = None;
                    attn.lora_v = None;
                    attn.lora_o = None;
                }
                AttentionLayer::Linear(attn) => {
                    attn.lora_ssm_in = None;
                    attn.lora_ssm_ba = None;
                    attn.lora_ssm_out = None;
                }
            }
        }
    }

    /// Clear all custom scales
    pub fn clear_custom_scales(&mut self) {
        self.custom_lm_head_scales = None;
        for layer in self.layers.iter_mut() {
            match &mut layer.attn {
                AttentionLayer::Full(attn) => {
                    attn.custom_q_scales = None;
                    attn.custom_k_scales = None;
                    attn.custom_v_scales = None;
                    attn.custom_o_scales = None;
                }
                AttentionLayer::Linear(attn) => {
                    attn.custom_ssm_in_scales = None;
                    attn.custom_ssm_ba_scales = None;
                    attn.custom_ssm_out_scales = None;
                }
            }
            layer.moe_block.custom_gate_scales = None;
            layer.moe_block.experts.custom_gate_block_mults = None;
            layer.moe_block.experts.custom_up_block_mults = None;
            layer.moe_block.experts.custom_down_block_mults = None;
            if let Some(ref mut shared) = layer.moe_block.shared_expert {
                shared.custom_gate_scales = None;
                shared.custom_up_scales = None;
                shared.custom_down_scales = None;
            }
        }
    }

    /// Merge LoRA adapters into the underlying quantized weights
    /// Returns a list of (internal_name, QTensor) pairs for the modified weights
    /// lora_map: layer name -> (A tensor, B tensor)
    /// sigma: perturbation magnitude (σ/√r scaling is applied internally)
    pub fn merge_lora_into_weights(
        &self,
        lora_map: &std::collections::HashMap<String, (Tensor, Tensor)>,
        sigma: f64,
    ) -> Result<Vec<(String, QTensor)>> {
        let mut merged = Vec::new();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let prefix = format!("blk.{}", layer_idx);

            match &layer.attn {
                AttentionLayer::Full(attn) => {
                    if let Some((a, b)) = lora_map.get(&format!("{}.attn_q", prefix)) {
                        let qt = attn.wq.merge_lora(a, b, sigma)?;
                        merged.push((format!("{}.attn_q", prefix), qt));
                    }
                    if let Some((a, b)) = lora_map.get(&format!("{}.attn_k", prefix)) {
                        let qt = attn.wk.merge_lora(a, b, sigma)?;
                        merged.push((format!("{}.attn_k", prefix), qt));
                    }
                    if let Some((a, b)) = lora_map.get(&format!("{}.attn_v", prefix)) {
                        let qt = attn.wv.merge_lora(a, b, sigma)?;
                        merged.push((format!("{}.attn_v", prefix), qt));
                    }
                    if let Some((a, b)) = lora_map.get(&format!("{}.attn_o", prefix)) {
                        let qt = attn.wo.merge_lora(a, b, sigma)?;
                        merged.push((format!("{}.attn_o", prefix), qt));
                    }
                }
                AttentionLayer::Linear(attn) => {
                    if let Some((a, b)) = lora_map.get(&format!("{}.ssm_in", prefix)) {
                        let qt = attn.ssm_in.merge_lora(a, b, sigma)?;
                        merged.push((format!("{}.ssm_in", prefix), qt));
                    }
                    if let Some((a, b)) = lora_map.get(&format!("{}.ssm_ba", prefix)) {
                        let qt = attn.ssm_beta_alpha.merge_lora(a, b, sigma)?;
                        merged.push((format!("{}.ssm_ba", prefix), qt));
                    }
                    if let Some((a, b)) = lora_map.get(&format!("{}.ssm_out", prefix)) {
                        let qt = attn.ssm_out.merge_lora(a, b, sigma)?;
                        merged.push((format!("{}.ssm_out", prefix), qt));
                    }
                }
            }
        }

        Ok(merged)
    }
}

/// Check if a dtype supports block scale modification for EGGROLL fine-tuning
fn supports_scale_modification(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
            | GgmlDType::Q8K
    )
}

// ============================================================================
// Helper functions
// ============================================================================

fn softplus(x: &Tensor) -> Result<Tensor> {
    // Numerically stable softplus: softplus(x) = log(1 + exp(x))
    // Compute in F32 to avoid F16 overflow (exp(11) > F16 max)
    // For large x: softplus(x) ≈ x (avoids exp overflow)
    // For small x: use standard formula
    // Threshold of 20 chosen because exp(20) ≈ 4.8e8 which is safe for f32
    // Note: Model uses F32 activations, no dtype conversion needed
    let threshold = 20.0f64;
    let x_clamped = x.clamp(-threshold, threshold)?;
    let exp_x = x_clamped.exp()?;
    let one_plus_exp = (exp_x + 1.0)?;
    let log_result = one_plus_exp.log()?;
    // For x > threshold, use x directly (softplus(x) ≈ x for large x)
    let mask = x.ge(threshold)?;
    mask.where_cond(x, &log_result)
}

fn l2_normalize(x: &Tensor, eps: f64) -> Result<Tensor> {
    // Use fused L2 normalize kernel when available (scale=1.0 for just normalization)
    // The fused kernel avoids multiple kernel launches for sqr, sum, sqrt, div
    crate::ops::l2_normalize_scale(x, 1.0, eps)
}

/// Pad a tensor with zeros along a specific dimension
/// Supports 3D and 4D tensors
fn pad_tensor(x: &Tensor, dim: usize, pad_size: usize) -> Result<Tensor> {
    if pad_size == 0 {
        return Ok(x.clone());
    }

    let dims = x.dims();
    let zeros = match dims.len() {
        3 => {
            let mut pad_dims = [dims[0], dims[1], dims[2]];
            pad_dims[dim] = pad_size;
            Tensor::zeros(&pad_dims[..], x.dtype(), x.device())?
        }
        4 => {
            let mut pad_dims = [dims[0], dims[1], dims[2], dims[3]];
            pad_dims[dim] = pad_size;
            Tensor::zeros(&pad_dims[..], x.dtype(), x.device())?
        }
        _ => {
            return Err(candle::Error::Msg(
                "pad_tensor only supports 3D and 4D tensors".to_string(),
            ));
        }
    };

    Tensor::cat(&[x, &zeros], dim)
}

/// Create strictly lower triangular mask (1s below diagonal, 0s on and above)
/// This is the causal_mask in llama.cpp
/// Optimized version using candle's comparison operations
fn create_causal_mask(size: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let indices = Tensor::arange(0u32, size as u32, device)?;
    // After broadcast: row_indices[i][j] = j (column index at each position)
    let col_idx = indices.reshape((1, size))?.broadcast_as((size, size))?;
    // After broadcast: col_indices[i][j] = i (row index at each position)
    let row_idx = indices.reshape((size, 1))?.broadcast_as((size, size))?;

    // Create mask where col < row (strictly lower triangular): j < i
    let mask = col_idx.lt(&row_idx)?;
    mask.to_dtype(dtype)?.reshape((1, 1, size, size))
}

/// Solve (I - L) * X = B where L is strictly lower triangular
/// Uses forward substitution for numerical stability
/// Returns the solved X matrix masked and with identity added
fn solve_lower_triangular(attn: &Tensor, causal_mask: &Tensor) -> Result<Tensor> {
    let (_, _, seq_len, seq_len2) = attn.dims4()?;

    if seq_len != seq_len2 {
        candle::bail!("solve_lower_triangular expects square matrix at dim 2,3");
    }

    if seq_len <= 1 {
        let identity = create_identity_mask(seq_len, attn.device(), attn.dtype())?;
        return attn.broadcast_mul(causal_mask)?.broadcast_add(&identity);
    }

    // Extract strictly lower triangular part (L matrix)
    let l_matrix = attn.broadcast_mul(causal_mask)?;

    // Use batched triangular solve with forward substitution
    crate::ops::solve_lower_triangular_batched(&l_matrix, attn, causal_mask)
}

/// Create identity mask (1s on diagonal)
/// Optimized version using candle's comparison operations
fn create_identity_mask(size: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let indices = Tensor::arange(0u32, size as u32, device)?;
    // After broadcast: col_idx[i][j] = j (column index at each position)
    let col_idx = indices.reshape((1, size))?.broadcast_as((size, size))?;
    // After broadcast: row_idx[i][j] = i (row index at each position)
    let row_idx = indices.reshape((size, 1))?.broadcast_as((size, size))?;

    // Diagonal is where row == col: i == j
    let mask = row_idx.eq(&col_idx)?;
    mask.to_dtype(dtype)?.reshape((1, 1, size, size))
}
