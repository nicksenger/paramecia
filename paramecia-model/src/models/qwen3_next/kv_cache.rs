use paramecia_core::quantized::{GgmlDType, QStorage, QTensor};
use paramecia_core::{DType, Device, Result, Tensor};
use paramecia_tensor::glowstick::{Shape2, Shape3, Shape4};
use paramecia_tensor::Tensor as TypedTensor;

use super::shape::{Ch, DState, DtRank, QkvDim, B};

type TLinearSsmState = TypedTensor<Shape4<B, DtRank, DState, DState>>;
type TLinearConvState = TypedTensor<Shape3<B, QkvDim, Ch>>;
type TLinearGateOffset = TypedTensor<Shape2<B, DtRank>>;
type TFullKvCache = TypedTensor<Shape4<B, super::shape::K, super::shape::Lk, super::shape::H>>;

/// Default maximum sequence length for pre-allocated KV cache
/// Pre-allocated KV cache with O(1) token insertion.
///
/// Unlike the naive `Tensor::cat` approach which is O(n) per token (O(n²) overall),
/// this implementation pre-allocates buffers and uses slice assignment for O(1) insertion.
/// This matches llama.cpp's approach using `ggml_set_rows` with pre-allocated buffers.
pub(super) struct PreallocatedKvCache {
    /// Pre-allocated K tensor: [batch, num_kv_heads, max_seq_len, head_dim]
    pub(super) k_cache: TFullKvCache,
    /// Pre-allocated V tensor: [batch, num_kv_heads, max_seq_len, head_dim]
    pub(super) v_cache: TFullKvCache,
    /// Current number of tokens stored (head position)
    pub(super) seq_len: usize,
    /// Maximum capacity
    pub(super) max_seq_len: usize,
    /// Batch size (for validation)
    pub(super) batch_size: usize,
}

impl std::fmt::Debug for PreallocatedKvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreallocatedKvCache")
            .field("seq_len", &self.seq_len)
            .field("max_seq_len", &self.max_seq_len)
            .field("batch_size", &self.batch_size)
            .finish()
    }
}

#[allow(dead_code)]
impl PreallocatedKvCache {
    /// Create a new pre-allocated KV cache
    pub(super) fn new(
        batch_size: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        use paramecia_tensor::glowstick::num::Unsigned;
        if num_kv_heads != <super::shape::K as Unsigned>::USIZE {
            paramecia_core::bail!(
                "shape/config mismatch for num_kv_heads: runtime={} type-level={}",
                num_kv_heads,
                <super::shape::K as Unsigned>::USIZE
            );
        }
        if head_dim != <super::shape::H as Unsigned>::USIZE {
            paramecia_core::bail!(
                "shape/config mismatch for head_dim: runtime={} type-level={}",
                head_dim,
                <super::shape::H as Unsigned>::USIZE
            );
        }

        // Pre-allocate K and V tensors with full capacity
        // Shape: [batch, num_kv_heads, max_seq_len, head_dim]
        let k_cache: TFullKvCache = Tensor::zeros(
            (batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype,
            device,
        )?
        .try_into()?;
        let v_cache: TFullKvCache = Tensor::zeros(
            (batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype,
            device,
        )?
        .try_into()?;

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
    pub(super) fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        let _new_k_typed: TFullKvCache = new_k.clone().try_into()?;
        let _new_v_typed: TFullKvCache = new_v.clone().try_into()?;
        let new_seq_len = new_k.dim(2)?;
        let new_end = self.seq_len + new_seq_len;

        if new_end > self.max_seq_len {
            paramecia_core::bail!(
                "KV cache overflow: trying to store {} tokens but max is {}",
                new_end,
                self.max_seq_len
            );
        }

        // Use slice_scatter to write new K/V at the current position
        // This avoids the O(n) copy that Tensor::cat would require
        self.k_cache = self
            .k_cache
            .inner()
            .slice_scatter(new_k, 2, self.seq_len)?
            .try_into()?;
        self.v_cache = self
            .v_cache
            .inner()
            .slice_scatter(new_v, 2, self.seq_len)?
            .try_into()?;

        self.seq_len = new_end;

        // Return view of valid portion only
        let k = self.k_cache.inner().narrow(2, 0, self.seq_len)?;
        let v = self.v_cache.inner().narrow(2, 0, self.seq_len)?;

        Ok((k, v))
    }

    /// Get current sequence length
    pub(super) fn len(&self) -> usize {
        self.seq_len
    }

    /// Truncate the cache to a given length.
    /// This is O(1) - just updates the seq_len pointer.
    pub(super) fn truncate(&mut self, new_len: usize) {
        self.seq_len = new_len.min(self.seq_len);
    }

    /// Check if cache is empty
    #[allow(dead_code)]
    pub(super) fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Clear the cache
    pub(super) fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Resize cache for a new batch size (clears existing data)
    pub(super) fn resize_batch(&mut self, batch_size: usize, dtype: DType) -> Result<()> {
        if batch_size != self.batch_size {
            let num_kv_heads = self.k_cache.inner().dim(1)?;
            let head_dim = self.k_cache.inner().dim(3)?;
            let device = self.k_cache.inner().device().clone();

            self.k_cache = Tensor::zeros(
                (batch_size, num_kv_heads, self.max_seq_len, head_dim),
                dtype,
                &device,
            )?
            .try_into()?;
            self.v_cache = Tensor::zeros(
                (batch_size, num_kv_heads, self.max_seq_len, head_dim),
                dtype,
                &device,
            )?
            .try_into()?;
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
pub(super) struct PreallocatedQuantizedKvCache {
    /// Pre-allocated quantized K buffer (stores raw quantized bytes)
    /// Shape conceptually: [batch, num_kv_heads, max_seq_len, head_dim]
    /// But stored as flat quantized data with block structure
    pub(super) k_cache: Vec<u8>,
    /// Pre-allocated quantized V buffer
    pub(super) v_cache: Vec<u8>,
    /// GGML dtype for quantization
    pub(super) ggml_dtype: GgmlDType,
    /// Current number of tokens stored
    pub(super) seq_len: usize,
    /// Maximum capacity
    pub(super) max_seq_len: usize,
    /// Batch size
    pub(super) batch_size: usize,
    /// Number of KV heads
    pub(super) num_kv_heads: usize,
    /// Head dimension (may be padded for block alignment)
    pub(super) head_dim: usize,
    /// Padded head dimension (aligned to block size)
    pub(super) padded_head_dim: usize,
    /// Block size for quantization (stored for potential future use)
    #[allow(dead_code)]
    pub(super) block_size: usize,
    /// Bytes per row (one row = one head for one token)
    pub(super) bytes_per_row: usize,
    /// Device for dequantization
    pub(super) device: Device,
    /// Persistent GPU storage for Vulkan Q8_0 fast-path.
    #[cfg(feature = "vulkan")]
    pub(super) k_gpu_storage: Option<QStorage>,
    #[cfg(feature = "vulkan")]
    pub(super) v_gpu_storage: Option<QStorage>,
    /// Persistent GPU storage for CUDA — avoids re-uploading entire KV cache every step.
    #[cfg(feature = "cuda")]
    pub(super) k_gpu_storage: Option<QStorage>,
    #[cfg(feature = "cuda")]
    pub(super) v_gpu_storage: Option<QStorage>,
}

impl std::fmt::Debug for PreallocatedQuantizedKvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("PreallocatedQuantizedKvCache");
        ds.field("ggml_dtype", &self.ggml_dtype)
            .field("seq_len", &self.seq_len)
            .field("max_seq_len", &self.max_seq_len)
            .field("batch_size", &self.batch_size)
            .field("num_kv_heads", &self.num_kv_heads)
            .field("head_dim", &self.head_dim)
            .field("padded_head_dim", &self.padded_head_dim)
            .field("bytes_per_row", &self.bytes_per_row);
        #[cfg(feature = "vulkan")]
        {
            ds.field("has_k_gpu_storage", &self.k_gpu_storage.is_some())
                .field("has_v_gpu_storage", &self.v_gpu_storage.is_some());
        }
        #[cfg(feature = "cuda")]
        {
            ds.field("has_k_gpu_storage", &self.k_gpu_storage.is_some())
                .field("has_v_gpu_storage", &self.v_gpu_storage.is_some());
        }
        ds.finish()
    }
}

impl PreallocatedQuantizedKvCache {
    /// Create a new pre-allocated quantized KV cache
    pub(super) fn new(
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

        let cache = Self {
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
            #[cfg(feature = "vulkan")]
            k_gpu_storage: None,
            #[cfg(feature = "vulkan")]
            v_gpu_storage: None,
            #[cfg(feature = "cuda")]
            k_gpu_storage: None,
            #[cfg(feature = "cuda")]
            v_gpu_storage: None,
        };
        // Note: GPU storage is allocated lazily on first append, not here,
        // to avoid pre-allocating max_seq_len worth of VRAM per layer.
        Ok(cache)
    }

    #[inline]
    pub(super) fn rows_per_position(&self) -> usize {
        self.batch_size * self.num_kv_heads
    }

    #[inline]
    pub(super) fn valid_bytes(&self) -> usize {
        self.seq_len * self.rows_per_position() * self.bytes_per_row
    }

    #[cfg(feature = "vulkan")]
    pub(super) fn maybe_init_gpu_storage(&mut self) -> Result<()> {
        if self.ggml_dtype != GgmlDType::Q8_0 {
            self.k_gpu_storage = None;
            self.v_gpu_storage = None;
            return Ok(());
        }
        if !matches!(self.device, Device::Vulkan(_)) {
            self.k_gpu_storage = None;
            self.v_gpu_storage = None;
            return Ok(());
        }
        if self.k_gpu_storage.is_none() || self.v_gpu_storage.is_none() {
            self.k_gpu_storage = Some(QStorage::from_data(
                std::borrow::Cow::Borrowed(&self.k_cache),
                &self.device,
                self.ggml_dtype,
            )?);
            self.v_gpu_storage = Some(QStorage::from_data(
                std::borrow::Cow::Borrowed(&self.v_cache),
                &self.device,
                self.ggml_dtype,
            )?);
        }
        Ok(())
    }

    #[cfg(not(feature = "vulkan"))]
    #[allow(dead_code)]
    pub(super) fn maybe_init_gpu_storage(&mut self) -> Result<()> {
        Ok(()) // CUDA uses lazy growth-based allocation via ensure_gpu_capacity
    }

    /// Ensure CUDA GPU buffers have at least `needed_bytes` capacity.
    /// Returns true if (re)allocation occurred (caller must re-upload existing data).
    #[cfg(feature = "cuda")]
    pub(super) fn ensure_gpu_capacity(&mut self, needed_bytes: usize) -> Result<bool> {
        if !matches!(self.device, Device::Cuda(_)) {
            return Ok(false);
        }
        let has_capacity = match &self.k_gpu_storage {
            Some(s) => paramecia_core::quantized::cuda::kv_buffer_capacity(s) >= needed_bytes,
            None => false,
        };
        if has_capacity {
            return Ok(false);
        }
        // Allocate with 2x growth, capped at max capacity
        let alloc_bytes = (needed_bytes * 2).min(self.k_cache.len());
        let cuda_dev = self.device.as_cuda_device()?;
        self.k_gpu_storage = Some(paramecia_core::quantized::cuda::alloc_kv_buffer(
            cuda_dev,
            alloc_bytes,
            self.ggml_dtype,
        )?);
        self.v_gpu_storage = Some(paramecia_core::quantized::cuda::alloc_kv_buffer(
            cuda_dev,
            alloc_bytes,
            self.ggml_dtype,
        )?);
        Ok(true)
    }

    #[cfg(feature = "vulkan")]
    pub(super) fn rebuild_gpu_storage_from_host(&mut self) -> Result<()> {
        if self.ggml_dtype != GgmlDType::Q8_0 || !matches!(self.device, Device::Vulkan(_)) {
            self.k_gpu_storage = None;
            self.v_gpu_storage = None;
            return Ok(());
        }
        self.k_gpu_storage = Some(QStorage::from_data(
            std::borrow::Cow::Borrowed(&self.k_cache),
            &self.device,
            self.ggml_dtype,
        )?);
        self.v_gpu_storage = Some(QStorage::from_data(
            std::borrow::Cow::Borrowed(&self.v_cache),
            &self.device,
            self.ggml_dtype,
        )?);
        Ok(())
    }

    #[cfg(all(not(feature = "vulkan"), feature = "cuda"))]
    pub(super) fn rebuild_gpu_storage_from_host(&mut self) -> Result<()> {
        self.k_gpu_storage = None;
        self.v_gpu_storage = None;
        if !matches!(self.device, Device::Cuda(_)) || self.seq_len == 0 {
            return Ok(());
        }
        let valid = self.valid_bytes();
        self.ensure_gpu_capacity(valid)?;
        paramecia_core::quantized::cuda::kv_buffer_append(
            self.k_gpu_storage.as_mut().unwrap(),
            &self.k_cache[..valid],
            0,
        )?;
        paramecia_core::quantized::cuda::kv_buffer_append(
            self.v_gpu_storage.as_mut().unwrap(),
            &self.v_cache[..valid],
            0,
        )?;
        Ok(())
    }

    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    pub(super) fn rebuild_gpu_storage_from_host(&mut self) -> Result<()> {
        Ok(())
    }

    pub(super) fn storage_views(&self) -> Result<(QStorage, QStorage)> {
        let valid_bytes = self.valid_bytes();
        #[cfg(feature = "vulkan")]
        {
            if let (Some(QStorage::Vulkan(k_storage)), Some(QStorage::Vulkan(v_storage))) =
                (&self.k_gpu_storage, &self.v_gpu_storage)
            {
                let k_buf = k_storage.gpu_buffer_arc().ok_or_else(|| {
                    paramecia_core::Error::Msg(
                        "quantized K cache is missing GPU buffer".to_string(),
                    )
                })?;
                let v_buf = v_storage.gpu_buffer_arc().ok_or_else(|| {
                    paramecia_core::Error::Msg(
                        "quantized V cache is missing GPU buffer".to_string(),
                    )
                })?;

                let k_view = paramecia_core::quantized::vulkan::QVulkanStorage::from_gpu_buffer(
                    self.ggml_dtype,
                    k_storage.device(),
                    k_buf,
                    valid_bytes,
                );
                let v_view = paramecia_core::quantized::vulkan::QVulkanStorage::from_gpu_buffer(
                    self.ggml_dtype,
                    v_storage.device(),
                    v_buf,
                    valid_bytes,
                );
                return Ok((QStorage::Vulkan(k_view), QStorage::Vulkan(v_view)));
            }
        }
        #[cfg(feature = "cuda")]
        {
            if let (Some(k_store), Some(v_store)) = (&self.k_gpu_storage, &self.v_gpu_storage) {
                let k_view = paramecia_core::quantized::cuda::kv_buffer_view(k_store, valid_bytes)?;
                let v_view = paramecia_core::quantized::cuda::kv_buffer_view(v_store, valid_bytes)?;
                return Ok((k_view, v_view));
            }
        }

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
        Ok((k_storage, v_storage))
    }

    #[cfg(feature = "vulkan")]
    pub(super) fn try_append_quantized_vulkan_q8(
        &mut self,
        new_k_padded: &Tensor,
        new_v_padded: &Tensor,
        new_seq_len: usize,
    ) -> Result<bool> {
        if self.ggml_dtype != GgmlDType::Q8_0 {
            return Ok(false);
        }
        self.maybe_init_gpu_storage()?;

        let (k_dst_storage, v_dst_storage) = match (&self.k_gpu_storage, &self.v_gpu_storage) {
            (Some(QStorage::Vulkan(k_dst)), Some(QStorage::Vulkan(v_dst))) => (k_dst, v_dst),
            _ => return Ok(false),
        };
        let k_dst_buf = k_dst_storage.gpu_buffer().ok_or_else(|| {
            paramecia_core::Error::Msg("quantized K cache Vulkan buffer is missing".to_string())
        })?;
        let v_dst_buf = v_dst_storage.gpu_buffer().ok_or_else(|| {
            paramecia_core::Error::Msg("quantized V cache Vulkan buffer is missing".to_string())
        })?;

        let new_k_seq = new_k_padded
            .permute((2, 0, 1, 3))?
            .contiguous()?
            .flatten_all()?;
        let new_v_seq = new_v_padded
            .permute((2, 0, 1, 3))?
            .contiguous()?
            .flatten_all()?;

        let (k_src_storage, _) = new_k_seq.storage_and_layout();
        let (v_src_storage, _) = new_v_seq.storage_and_layout();
        let (k_src_vk, v_src_vk) = match (&*k_src_storage, &*v_src_storage) {
            (paramecia_core::Storage::Vulkan(k), paramecia_core::Storage::Vulkan(v)) => (k, v),
            _ => return Ok(false),
        };

        let start_offset = self.seq_len * self.rows_per_position() * self.bytes_per_row;
        let elem_count = new_seq_len * self.rows_per_position() * self.padded_head_dim;

        paramecia_core::quantized::vulkan::QVulkanStorage::quantize_into_buffer(
            k_dst_storage.device(),
            k_src_vk,
            k_dst_buf,
            start_offset,
            elem_count,
        )?;
        paramecia_core::quantized::vulkan::QVulkanStorage::quantize_into_buffer(
            v_dst_storage.device(),
            v_src_vk,
            v_dst_buf,
            start_offset,
            elem_count,
        )?;
        Ok(true)
    }

    #[cfg(not(feature = "vulkan"))]
    pub(super) fn try_append_quantized_vulkan_q8(
        &mut self,
        _new_k_padded: &Tensor,
        _new_v_padded: &Tensor,
        _new_seq_len: usize,
    ) -> Result<bool> {
        Ok(false)
    }

    pub(super) fn snapshot_quantized_bytes(&self) -> Result<(Vec<u8>, Vec<u8>)> {
        if self.seq_len == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let valid_bytes = self.valid_bytes();
        let (k_storage, v_storage) = self.storage_views()?;
        let shape = (
            self.seq_len,
            self.batch_size,
            self.num_kv_heads,
            self.padded_head_dim,
        );
        let k_qtensor = QTensor::new(k_storage, shape)?;
        let v_qtensor = QTensor::new(v_storage, shape)?;
        let k_data = k_qtensor.data()?.into_owned();
        let v_data = v_qtensor.data()?.into_owned();
        if k_data.len() < valid_bytes || v_data.len() < valid_bytes {
            paramecia_core::bail!(
                "snapshot quantized cache size mismatch: k={} v={} expected at least {}",
                k_data.len(),
                v_data.len(),
                valid_bytes
            );
        }
        Ok((
            k_data[..valid_bytes].to_vec(),
            v_data[..valid_bytes].to_vec(),
        ))
    }

    /// Append new K/V tensors to the cache.
    ///
    /// This quantizes only the new tokens (O(new_tokens)) and writes them
    /// to the pre-allocated buffer at the current position.
    ///
    /// new_k, new_v shape: [batch, num_kv_heads, new_seq_len, head_dim]
    pub(super) fn append_quantized(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
        let new_seq_len = new_k.dim(2)?;
        let new_end = self.seq_len + new_seq_len;

        if new_end > self.max_seq_len {
            paramecia_core::bail!(
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

        if self.try_append_quantized_vulkan_q8(&new_k_padded, &new_v_padded, new_seq_len)? {
            self.seq_len = new_end;
            return Ok(());
        }

        // Calculate where to write in the pre-allocated buffer
        // Layout: [batch, num_kv_heads, seq_len, head_dim] but stored as rows
        // Each position in the sequence has batch * num_kv_heads rows
        let rows_per_position = self.rows_per_position();
        let start_offset = self.seq_len * rows_per_position * self.bytes_per_row;
        let bytes_to_copy = new_seq_len * rows_per_position * self.bytes_per_row;

        #[cfg(feature = "cuda")]
        if matches!(self.device, Device::Cuda(_)) && self.ggml_dtype == GgmlDType::Q8_0 {
            let new_valid_bytes = start_offset + bytes_to_copy;
            let reallocated = self.ensure_gpu_capacity(new_valid_bytes)?;
            // On reallocation, re-upload existing prefix from CPU mirror.
            if reallocated && start_offset > 0 {
                paramecia_core::quantized::cuda::kv_buffer_append(
                    self.k_gpu_storage.as_mut().unwrap(),
                    &self.k_cache[..start_offset],
                    0,
                )?;
                paramecia_core::quantized::cuda::kv_buffer_append(
                    self.v_gpu_storage.as_mut().unwrap(),
                    &self.v_cache[..start_offset],
                    0,
                )?;
            }

            // Prepare seq-major contiguous f32 source on CUDA: [seq, batch, head, d] flattened.
            let new_k_seq = new_k_padded
                .permute((2, 0, 1, 3))?
                .contiguous()?
                .to_dtype(DType::F32)?
                .flatten_all()?;
            let new_v_seq = new_v_padded
                .permute((2, 0, 1, 3))?
                .contiguous()?
                .to_dtype(DType::F32)?
                .flatten_all()?;

            let (new_k_storage, _) = new_k_seq.storage_and_layout();
            let (new_v_storage, _) = new_v_seq.storage_and_layout();
            let (new_k_cuda, new_v_cuda) = match (&*new_k_storage, &*new_v_storage) {
                (paramecia_core::Storage::Cuda(k), paramecia_core::Storage::Cuda(v)) => (k, v),
                _ => {
                    paramecia_core::bail!("expected CUDA storage for Q8_0 KV append fast-path")
                }
            };

            let new_k_bytes = paramecia_core::quantized::cuda::kv_buffer_append_quantized_q8_0_f32(
                self.k_gpu_storage.as_mut().unwrap(),
                new_k_cuda,
                self.padded_head_dim,
                new_seq_len * rows_per_position,
                start_offset,
            )?;
            let new_v_bytes = paramecia_core::quantized::cuda::kv_buffer_append_quantized_q8_0_f32(
                self.v_gpu_storage.as_mut().unwrap(),
                new_v_cuda,
                self.padded_head_dim,
                new_seq_len * rows_per_position,
                start_offset,
            )?;

            // Keep host mirrors in sync for snapshot/restore/reallocation paths.
            self.k_cache[start_offset..start_offset + bytes_to_copy].copy_from_slice(&new_k_bytes);
            self.v_cache[start_offset..start_offset + bytes_to_copy].copy_from_slice(&new_v_bytes);

            self.seq_len = new_end;
            return Ok(());
        }

        // IMPORTANT: This cache stores quantized bytes in seq-major layout:
        //   [seq, batch, head, d]
        // so that the first `seq_len` tokens are a single contiguous prefix in memory.
        //
        // This makes appends O(new_tokens) (single memcpy) and enables `valid_bytes`
        // to be a contiguous slice.
        // Download to CPU first, then permute to seq-major on CPU.
        // This avoids potential issues with Vulkan strided copy for permuted tensors.
        let (new_k_cpu, new_v_cpu) = {
            let k = new_k_padded
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .permute((2, 0, 1, 3))?
                .contiguous()?
                .flatten_all()?; // Flatten to 1D for quantization
            let v = new_v_padded
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .permute((2, 0, 1, 3))?
                .contiguous()?
                .flatten_all()?; // Flatten to 1D for quantization
            (k, v)
        };

        // Quantize only the new tokens
        let new_k_qtensor = QTensor::quantize(&new_k_cpu, self.ggml_dtype)?;
        let new_v_qtensor = QTensor::quantize(&new_v_cpu, self.ggml_dtype)?;

        // Get the raw quantized bytes
        let new_k_bytes = new_k_qtensor.data()?;
        let new_v_bytes = new_v_qtensor.data()?;

        // Copy quantized data to pre-allocated buffer
        self.k_cache[start_offset..start_offset + bytes_to_copy]
            .copy_from_slice(&new_k_bytes.as_ref()[..bytes_to_copy]);
        self.v_cache[start_offset..start_offset + bytes_to_copy]
            .copy_from_slice(&new_v_bytes.as_ref()[..bytes_to_copy]);

        // Sync to persistent GPU buffer (CUDA path)
        #[cfg(feature = "cuda")]
        {
            if matches!(self.device, Device::Cuda(_)) {
                let new_valid_bytes = start_offset + bytes_to_copy;
                let reallocated = self.ensure_gpu_capacity(new_valid_bytes)?;
                // On reallocation, re-upload all existing data from CPU cache
                if reallocated && start_offset > 0 {
                    paramecia_core::quantized::cuda::kv_buffer_append(
                        self.k_gpu_storage.as_mut().unwrap(),
                        &self.k_cache[..start_offset],
                        0,
                    )?;
                    paramecia_core::quantized::cuda::kv_buffer_append(
                        self.v_gpu_storage.as_mut().unwrap(),
                        &self.v_cache[..start_offset],
                        0,
                    )?;
                }
                // Upload only the new bytes
                paramecia_core::quantized::cuda::kv_buffer_append(
                    self.k_gpu_storage.as_mut().unwrap(),
                    &new_k_bytes.as_ref()[..bytes_to_copy],
                    start_offset,
                )?;
                paramecia_core::quantized::cuda::kv_buffer_append(
                    self.v_gpu_storage.as_mut().unwrap(),
                    &new_v_bytes.as_ref()[..bytes_to_copy],
                    start_offset,
                )?;
            }
        }

        self.seq_len = new_end;

        Ok(())
    }

    /// Backwards-compatible helper: append then dequantize the full cache.
    ///
    /// This is expensive (O(seq_len)) and should be avoided on the hot path.
    pub(super) fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        self.append_quantized(new_k, new_v)?;
        self.get_kv()
    }

    /// Get the current K/V tensors (dequantized)
    pub(super) fn get_kv(&self) -> Result<(Tensor, Tensor)> {
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
        let (k_storage, v_storage) = self.storage_views()?;

        // Storage is seq-major: [seq, batch, head, d]
        let k_qtensor = QTensor::new(
            k_storage,
            (
                self.seq_len,
                self.batch_size,
                self.num_kv_heads,
                self.padded_head_dim,
            ),
        )?;
        let v_qtensor = QTensor::new(
            v_storage,
            (
                self.seq_len,
                self.batch_size,
                self.num_kv_heads,
                self.padded_head_dim,
            ),
        )?;

        // Dequantize
        let k_deq = k_qtensor.dequantize(&self.device)?;
        let v_deq = v_qtensor.dequantize(&self.device)?;

        // Convert to [batch, head, seq, d] and trim padding.
        let k = k_deq.permute((1, 2, 0, 3))?.narrow(3, 0, self.head_dim)?;
        let v = v_deq.permute((1, 2, 0, 3))?.narrow(3, 0, self.head_dim)?;

        Ok((k, v))
    }

    /// Get K/V tensors as quantized storage for Q8_0 flash attention
    ///
    /// Returns (k_storage, v_storage, k_layout, v_layout) for direct
    /// use with Q8_0 flash attention kernel, avoiding dequantization.
    #[allow(dead_code)]
    pub(super) fn get_kv_storage(
        &self,
    ) -> Result<(
        QStorage,
        QStorage,
        paramecia_core::Layout,
        paramecia_core::Layout,
    )> {
        if self.seq_len == 0 {
            paramecia_core::bail!("Cannot get storage from empty cache");
        }

        // Calculate how many bytes represent the current sequence
        let (k_storage, v_storage) = self.storage_views()?;

        // Storage is seq-major: [seq, batch, head, d], exposed as [batch, seq, head, d].
        let k_shape = paramecia_core::Shape::from((
            self.batch_size,
            self.seq_len,
            self.num_kv_heads,
            self.head_dim,
        ));
        let v_shape = k_shape.clone();

        // Strides are in bytes (packed rows); `*_stride_d` is unused by the kernel.
        let stride_d = 1;
        let stride_h = self.bytes_per_row;
        let stride_b = self.num_kv_heads * self.bytes_per_row;
        let stride_seq = self.batch_size * self.num_kv_heads * self.bytes_per_row;

        let k_layout =
            paramecia_core::Layout::new(k_shape, vec![stride_b, stride_seq, stride_h, stride_d], 0);
        let v_layout =
            paramecia_core::Layout::new(v_shape, vec![stride_b, stride_seq, stride_h, stride_d], 0);

        Ok((k_storage, v_storage, k_layout, v_layout))
    }

    /// Clear the cache
    pub(super) fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Truncate the cache to a given length.
    /// This is O(1) - just updates the seq_len pointer.
    pub(super) fn truncate(&mut self, new_len: usize) {
        self.seq_len = new_len.min(self.seq_len);
    }

    /// Resize cache for a new batch size
    #[allow(dead_code)]
    pub(super) fn resize_batch(&mut self, batch_size: usize) -> Result<()> {
        if batch_size != self.batch_size {
            let total_rows = batch_size * self.num_kv_heads * self.max_seq_len;
            let total_bytes = total_rows * self.bytes_per_row;
            self.k_cache = vec![0u8; total_bytes];
            self.v_cache = vec![0u8; total_bytes];
            self.batch_size = batch_size;
            self.seq_len = 0;
            self.rebuild_gpu_storage_from_host()?;
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
pub(super) enum KvCacheStorage {
    Float(TFullKvCache, TFullKvCache),
    Quantized(QTensor, QTensor, Vec<usize>, Vec<usize>),
}

impl std::fmt::Debug for KvCacheStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float(_, _) => f.debug_tuple("Float").field(&"...").field(&"...").finish(),
            Self::Quantized(_, _, k_shape, v_shape) => f
                .debug_tuple("Quantized")
                .field(&"...")
                .field(&"...")
                .field(k_shape)
                .field(v_shape)
                .finish(),
        }
    }
}

#[allow(dead_code)]
impl KvCacheStorage {
    pub(super) fn get_kv_ref(&self) -> Result<(&Tensor, &Tensor)> {
        match self {
            Self::Float(k, v) => Ok((k.inner(), v.inner())),
            Self::Quantized(_, _, _, _) => {
                paramecia_core::bail!("Use get_kv_owned for quantized cache")
            }
        }
    }

    pub(super) fn get_kv_owned(&self) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Float(k, v) => Ok((k.inner().clone(), v.inner().clone())),
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
/// for efficient snapshot/restore during speculative decoding.
pub(super) struct RecurrentState {
    /// SSM/delta-net state: [batch, num_heads, state_dim, state_dim]
    pub(super) ssm_state: TLinearSsmState,
    /// Convolution state: [batch, conv_dim, conv_kernel_size - 1]
    pub(super) conv_state: TLinearConvState,
    /// Accumulated gate cumsum offset for prefix cache continuation.
    /// Shape: [batch, num_heads]. When continuing from prefix cache, this offset
    /// is added to the new tokens' gate_cumsum to correctly compute decay factors.
    pub(super) gate_cumsum_offset: Option<TLinearGateOffset>,
    /// Backup buffer for snapshot/restore (lazy allocated)
    pub(super) backup_ssm: Option<TLinearSsmState>,
    pub(super) backup_conv: Option<TLinearConvState>,
    pub(super) backup_gate_offset: Option<TLinearGateOffset>,
    /// Intermediate states for verification state-slicing.
    /// Stores SSM state after each position: Vec of [batch, num_heads, state_dim, state_dim]
    pub(super) intermediate_states: Option<Vec<TLinearSsmState>>,
    /// Intermediate conv states for verification state-slicing.
    pub(super) intermediate_conv_states: Option<Vec<TLinearConvState>>,
}

impl std::fmt::Debug for RecurrentState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecurrentState")
            .field("has_gate_offset", &self.gate_cumsum_offset.is_some())
            .field("has_backup_ssm", &self.backup_ssm.is_some())
            .field("has_backup_conv", &self.backup_conv.is_some())
            .field("has_backup_gate_offset", &self.backup_gate_offset.is_some())
            .field(
                "num_intermediate_states",
                &self.intermediate_states.as_ref().map(|v| v.len()),
            )
            .field(
                "num_intermediate_conv_states",
                &self.intermediate_conv_states.as_ref().map(|v| v.len()),
            )
            .finish()
    }
}

impl RecurrentState {
    pub(super) fn new(ssm_state: Tensor, conv_state: Tensor) -> Result<Self> {
        Ok(Self {
            ssm_state: ssm_state.try_into()?,
            conv_state: conv_state.try_into()?,
            gate_cumsum_offset: None,
            backup_ssm: None,
            backup_conv: None,
            backup_gate_offset: None,
            intermediate_states: None,
            intermediate_conv_states: None,
        })
    }

    pub(super) fn with_gate_offset(
        ssm_state: Tensor,
        conv_state: Tensor,
        gate_offset: Tensor,
    ) -> Result<Self> {
        Ok(Self {
            ssm_state: ssm_state.try_into()?,
            conv_state: conv_state.try_into()?,
            gate_cumsum_offset: Some(gate_offset.try_into()?),
            backup_ssm: None,
            backup_conv: None,
            backup_gate_offset: None,
            intermediate_states: None,
            intermediate_conv_states: None,
        })
    }

    pub(super) fn ssm_state_ref(&self) -> &Tensor {
        self.ssm_state.inner()
    }

    pub(super) fn conv_state_ref(&self) -> &Tensor {
        self.conv_state.inner()
    }

    pub(super) fn gate_offset_ref(&self) -> Option<&Tensor> {
        self.gate_cumsum_offset.as_ref().map(|t| t.inner())
    }

    pub(super) fn set_ssm_state(&mut self, state: Tensor) -> Result<()> {
        self.ssm_state = state.try_into()?;
        Ok(())
    }

    pub(super) fn set_conv_state(&mut self, state: Tensor) -> Result<()> {
        self.conv_state = state.try_into()?;
        Ok(())
    }

    pub(super) fn set_gate_offset(&mut self, offset: Option<Tensor>) -> Result<()> {
        self.gate_cumsum_offset = match offset {
            Some(t) => Some(t.try_into()?),
            None => None,
        };
        Ok(())
    }

    /// Snapshot: save reference to current state tensors.
    ///
    /// This uses shallow clone (O(1)) which is safe because:
    /// - State updates use field assignment (e.g., `rs.ssm_state = new_tensor`)
    /// - This replaces the tensor reference, not mutating underlying data
    /// - So backup's reference to the old tensor remains valid
    pub(super) fn snapshot(&mut self) -> Result<()> {
        self.backup_ssm = Some(self.ssm_state.clone());
        self.backup_conv = Some(self.conv_state.clone());
        self.backup_gate_offset = self.gate_cumsum_offset.clone();
        Ok(())
    }

    /// Restore: swap backup references back to primary.
    /// O(1) operation - just swaps tensor references.
    pub(super) fn restore(&mut self) -> Result<()> {
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
    pub(super) fn clear_intermediate_states(&mut self) {
        self.intermediate_states = None;
        self.intermediate_conv_states = None;
    }

    /// Initialize intermediate states buffer for a given sequence length.
    /// NOTE: Part of speculative decoding verification - not yet wired up.
    #[allow(dead_code)]
    pub(super) fn init_intermediate_states(&mut self, seq_len: usize) {
        self.intermediate_states = Some(Vec::with_capacity(seq_len));
        self.intermediate_conv_states = Some(Vec::with_capacity(seq_len));
    }

    /// Save current state as an intermediate state.
    /// NOTE: Part of speculative decoding verification - not yet wired up.
    #[allow(dead_code)]
    pub(super) fn save_intermediate_state(&mut self) {
        if let Some(ref mut states) = self.intermediate_states {
            states.push(self.ssm_state.clone());
        }
        if let Some(ref mut states) = self.intermediate_conv_states {
            states.push(self.conv_state.clone());
        }
    }

    /// Restore to a specific intermediate state by index.
    /// Returns true if successful, false if index out of bounds.
    pub(super) fn restore_to_intermediate(&mut self, index: usize) -> bool {
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
    pub(super) fn num_intermediate_states(&self) -> usize {
        self.intermediate_states
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(0)
    }
}

/// Snapshot marker for a single layer, used for speculative decoding rollback.
/// For full attention: stores the KV cache sequence length.
/// For linear attention: just a marker (state is stored in-place in the layer).
#[derive(Debug)]
pub enum LayerSnapshot {
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
        k_cache: TFullKvCache,
        v_cache: TFullKvCache,
        seq_len: usize,
    },
    /// Linear attention: cloned SSM and conv states
    LinearAttention {
        ssm_state: TLinearSsmState,
        conv_state: TLinearConvState,
        /// Accumulated gate cumsum at the end of prefix: [batch, num_heads]
        /// Used to offset new tokens' gate_cumsum for correct decay computation.
        gate_cumsum_offset: TLinearGateOffset,
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
