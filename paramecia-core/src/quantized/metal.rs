use super::{GgmlDType, QStorage};
use crate::backend::BackendStorage;
use crate::{DType, MetalDevice, MetalStorage, Result, Shape, D};
use paramecia_metal::metal::Buffer;
use std::sync::Arc;

#[derive(Clone)]
pub struct QMetalStorage {
    dtype: GgmlDType,
    device: MetalDevice,
    buffer: Arc<Buffer>,
}

impl QMetalStorage {
    pub fn zeros(device: &MetalDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size = elem_count * dtype.type_size() / dtype.block_size();
        let buffer = device.allocate_zeros(size)?;
        Ok(Self {
            buffer,
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    /// Returns the per-tensor ID used by the fused matmul kernel for RNG seeding.
    /// Must match the `tensor_id` passed to `call_fused_matmul_vec_metal`.
    pub fn fused_tensor_id(&self) -> u64 {
        Arc::as_ptr(&self.buffer) as u64
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<MetalStorage> {
        use crate::quantized::k_quants::GgmlType;

        let buffer = self.device.allocate_buffer(self.buffer.length())?;
        let blit = self.device.blit_command_encoder()?;
        blit.set_label("blit_to_cpu");
        blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
        blit.end_encoding();
        self.device.wait_until_completed()?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => {
                let vec: Vec<f32> = read_to_vec(&buffer, block_len);
                f32::to_float(&vec, &mut out);
            }
            GgmlDType::F16 => {
                let vec: Vec<half::f16> = read_to_vec(&buffer, block_len);
                half::f16::to_float(&vec, &mut out);
            }
            GgmlDType::BF16 => {
                let vec: Vec<half::bf16> = read_to_vec(&buffer, block_len);
                half::bf16::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_0 => {
                let vec: Vec<crate::quantized::BlockQ4_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_1 => {
                let vec: Vec<crate::quantized::BlockQ4_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_0 => {
                let vec: Vec<crate::quantized::BlockQ5_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_1 => {
                let vec: Vec<crate::quantized::BlockQ5_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_0 => {
                let vec: Vec<crate::quantized::BlockQ8_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_1 => {
                let vec: Vec<crate::quantized::BlockQ8_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q2K => {
                let vec: Vec<crate::quantized::BlockQ2K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ2K::to_float(&vec, &mut out);
            }
            GgmlDType::Q3K => {
                let vec: Vec<crate::quantized::BlockQ3K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ3K::to_float(&vec, &mut out);
            }
            GgmlDType::Q4K => {
                let vec: Vec<crate::quantized::BlockQ4K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4K::to_float(&vec, &mut out);
            }
            GgmlDType::Q5K => {
                let vec: Vec<crate::quantized::BlockQ5K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5K::to_float(&vec, &mut out);
            }
            GgmlDType::Q6K => {
                let vec: Vec<crate::quantized::BlockQ6K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ6K::to_float(&vec, &mut out);
            }
            GgmlDType::Q8K => {
                let vec: Vec<crate::quantized::BlockQ8K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8K::to_float(&vec, &mut out);
            }
        }

        let buffer = self.device.new_buffer_with_data(&out)?;
        Ok(MetalStorage::new(
            buffer,
            self.device.clone(),
            elem_count,
            DType::F32,
        ))
    }

    pub fn quantize(&mut self, src: &MetalStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &MetalStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.buffer.length()
    }

    pub fn slice(&self, offset: usize, size: usize) -> Result<Self> {
        if offset + size > self.buffer.length() {
            crate::bail!(
                "slice range {}..{} exceeds storage size {}",
                offset,
                offset + size,
                self.buffer.length()
            )
        }
        // Metal buffers don't support slicing directly like CUDA, but we can create a new buffer view
        // by using the contents() method to get a pointer and creating a buffer from that slice
        // For now, we'll copy the slice to a new buffer
        let slice_contents = unsafe {
            std::slice::from_raw_parts((self.buffer.contents() as *const u8).add(offset), size)
        };
        let new_buffer = self.device.new_buffer_with_data(slice_contents)?;
        Ok(Self {
            dtype: self.dtype,
            device: self.device.clone(),
            buffer: new_buffer,
        })
    }

    fn fwd_mv(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();

        // We always use a single batch dimension and stack all the tensors in the batch on the
        // second dimension as the implementation in candle-metal-kernels doesn't handle batch
        // properly.
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            n => crate::bail!("Invalid rank {n} for quantized matmul metal"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul")?;
        let encoder = device.command_encoder()?;
        // In some cases it would be better to use the mm variant, though it has its drawbacks
        // around memory alignment.
        for batch_id in 0..m {
            paramecia_metal::call_quantized_matmul_mv_t(
                device.device(),
                &encoder,
                device.kernels(),
                self.dtype.into(),
                (1, 1, n, k),
                storage.buffer(),
                (layout.start_offset() + batch_id * k) * storage.dtype().size_in_bytes(),
                &self.buffer,
                batch_id * n * DType::F32.size_in_bytes(),
                &dst,
            )
            .map_err(MetalError::from)?;
        }
        let dst_storage = crate::MetalStorage::new(dst, device, dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let n = self_shape.dim(D::Minus2)?;
        let k = self_shape.dim(D::Minus1)?;
        let mut dst_shape = src_shape.dims().to_vec();

        if src_shape.rank() < self_shape.rank() {
            crate::bail!(
                "input rank ({}) must be >= weight rank ({})",
                src_shape.rank(),
                self_shape.rank()
            )
        }

        if src_shape.dim(D::Minus2)? == 1 {
            return self.fwd_mv(self_shape, storage, layout);
        }

        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), DType::F32, "qmatmul")?;
        let encoder = device.command_encoder()?;

        assert_eq!(storage.dtype(), DType::F32);

        if self_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", self_shape.rank())
        }
        let src0_l = crate::Layout::contiguous(
            [vec![1; 4 - self_shape.rank()], self_shape.dims().to_vec()].concat(),
        );
        let src0_stride = src0_l
            .stride()
            .iter()
            .map(|x| {
                (*x as f32 * (self.dtype.type_size() as f32 / self.dtype.block_size() as f32))
                    as usize
            })
            .collect::<Vec<_>>();

        if src_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", src_shape.rank())
        }
        let src1_l = crate::Layout::contiguous(
            [vec![1; 4 - src_shape.rank()], src_shape.dims().to_vec()].concat(),
        );

        paramecia_metal::call_quantized_matmul_mm_t(
            device.device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            src0_l.dims(),
            &src0_stride,
            &self.buffer,
            src1_l.dims(),
            &src1_l
                .stride()
                .iter()
                .map(|x| x * DType::F32.size_in_bytes())
                .collect::<Vec<_>>(),
            storage.buffer(),
            src1_l.start_offset() * storage.dtype().size_in_bytes(),
            dst_shape.dims(),
            0,
            &dst,
        )
        .map_err(MetalError::from)?;

        let dst_storage = crate::MetalStorage::new(dst, device, dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let buffer = self.device.allocate_buffer(self.buffer.length())?;
        {
            let blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
            blit.end_encoding();
        }
        self.device.wait_until_completed()?;
        Ok(read_to_vec::<u8>(&buffer, self.storage_size_in_bytes()))
    }

    /// Extract scaling factors from quantized blocks (QZO support).
    ///
    /// Uses GPU kernels for supported types (Q8_0, Q4K, Q2K, Q3K, Q5K, Q6K),
    /// falls back to CPU for others.
    pub fn extract_scales(&self, elem_count: usize) -> Result<MetalStorage> {
        use crate::quantized::k_quants::GgmlType;

        let block_len = elem_count / self.dtype.block_size();

        // Try GPU path for supported types
        match self.dtype {
            GgmlDType::Q8_0
            | GgmlDType::Q4K
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q5K
            | GgmlDType::Q6K => {
                let scales_buffer =
                    self.device
                        .new_buffer(block_len, DType::F32, "extract_scales")?;
                let encoder = self.device.command_encoder()?;
                paramecia_metal::call_extract_scales_metal(
                    self.device.device(),
                    &encoder,
                    self.device.kernels(),
                    self.dtype.into(),
                    &self.buffer,
                    &scales_buffer,
                    block_len,
                )
                .map_err(crate::MetalError::from)?;
                return Ok(MetalStorage::new(
                    scales_buffer,
                    self.device.clone(),
                    block_len,
                    DType::F32,
                ));
            }
            _ => {}
        }

        // CPU fallback for unsupported types
        let buffer = self.device.allocate_buffer(self.buffer.length())?;
        let blit = self.device.blit_command_encoder()?;
        blit.set_label("extract_scales_blit_to_cpu");
        blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
        blit.end_encoding();
        self.device.wait_until_completed()?;

        let scales = match self.dtype {
            GgmlDType::Q4_0 => {
                let vec: Vec<crate::quantized::BlockQ4_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_0::extract_scales(&vec)
            }
            GgmlDType::Q4_1 => {
                let vec: Vec<crate::quantized::BlockQ4_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_1::extract_scales(&vec)
            }
            GgmlDType::Q5_0 => {
                let vec: Vec<crate::quantized::BlockQ5_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_0::extract_scales(&vec)
            }
            GgmlDType::Q5_1 => {
                let vec: Vec<crate::quantized::BlockQ5_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_1::extract_scales(&vec)
            }
            GgmlDType::Q8_1 => {
                let vec: Vec<crate::quantized::BlockQ8_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_1::extract_scales(&vec)
            }
            GgmlDType::Q8K => {
                let vec: Vec<crate::quantized::BlockQ8K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8K::extract_scales(&vec)
            }
            _ => {
                crate::bail!("extract_scales not supported for dtype {:?}", self.dtype)
            }
        };

        let scales_buffer = self.device.new_buffer_with_data(&scales)?;
        Ok(MetalStorage::new(
            scales_buffer,
            self.device.clone(),
            scales.len(),
            DType::F32,
        ))
    }

    /// GPU-accelerated in-place perturbation of quantized weights.
    pub fn perturb_weights_gpu(
        &mut self,
        perturbation: &MetalStorage,
        epsilon: f32,
        seed: u64,
        add: bool,
    ) -> Result<()> {
        use crate::MetalError;

        let num_blocks = self.buffer.length() as usize / self.dtype.type_size();

        let encoder = self.device.command_encoder()?;
        paramecia_metal::call_perturb_weights_metal(
            self.device.device(),
            &encoder,
            self.device.kernels(),
            self.dtype.into(),
            &self.buffer,
            perturbation.buffer(),
            num_blocks,
            epsilon,
            seed,
            add,
        )
        .map_err(MetalError::from)?;
        Ok(())
    }

    /// GPU-accelerated combined restore and update operation.
    pub fn restore_and_update_gpu(
        &mut self,
        perturbation: &MetalStorage,
        restore_epsilon: f32,
        update_scale: f32,
        restore_seed: u64,
        update_seed: u64,
    ) -> Result<()> {
        use crate::MetalError;

        let num_blocks = self.buffer.length() as usize / self.dtype.type_size();

        let encoder = self.device.command_encoder()?;
        paramecia_metal::call_restore_and_update_metal(
            self.device.device(),
            &encoder,
            self.device.kernels(),
            self.dtype.into(),
            &self.buffer,
            perturbation.buffer(),
            num_blocks,
            restore_epsilon,
            update_scale,
            restore_seed,
            update_seed,
        )
        .map_err(MetalError::from)?;
        Ok(())
    }

    /// Modify block scales by per-block multipliers (GPU).
    pub fn modify_block_scales(&self, multipliers: &MetalStorage) -> Result<Self> {
        use crate::MetalError;

        let num_blocks = self.buffer.length() as usize / self.dtype.type_size();

        // Clone the buffer for modification
        let new_buffer = self.device.allocate_buffer(self.buffer.length())?;
        {
            let blit = self.device.blit_command_encoder()?;
            blit.set_label("modify_block_scales_copy");
            blit.copy_from_buffer(&self.buffer, 0, &new_buffer, 0, self.buffer.length());
            blit.end_encoding();
        }

        let encoder = self.device.command_encoder()?;
        paramecia_metal::call_modify_block_scales_metal(
            self.device.device(),
            &encoder,
            self.device.kernels(),
            self.dtype.into(),
            &new_buffer,
            multipliers.buffer(),
            num_blocks,
        )
        .map_err(MetalError::from)?;

        Ok(Self {
            dtype: self.dtype,
            device: self.device.clone(),
            buffer: new_buffer,
        })
    }

    /// Fused dequantize + perturb + matmul (GPU).
    pub fn fused_fwd(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
        seed: u64,
        epsilon: f32,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();

        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            n => crate::bail!("Invalid rank {n} for quantized fused matmul metal"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), DType::F32, "fused_qmatmul")?;

        // Use per-tensor ID from buffer address (Arc pointer value as unique ID)
        let tensor_id = Arc::as_ptr(&self.buffer) as u64;

        // Dispatch one vector-matrix multiply per batch element
        for batch_id in 0..m {
            let encoder = device.command_encoder()?;
            paramecia_metal::call_fused_matmul_vec_metal(
                device.device(),
                &encoder,
                device.kernels(),
                self.dtype.into(),
                &self.buffer,
                storage.buffer(),
                (layout.start_offset() + batch_id * k) * storage.dtype().size_in_bytes(),
                &dst,
                batch_id * n * DType::F32.size_in_bytes(),
                n,
                k,
                seed,
                epsilon,
                tensor_id,
            )
            .map_err(MetalError::from)?;
        }

        let dst_storage = MetalStorage::new(dst, device, dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &MetalDevice,
    data: &[T],
) -> Result<QStorage> {
    let buffer = device.new_buffer_with_data(data)?;
    let device = device.clone();
    Ok(QStorage::Metal(QMetalStorage {
        dtype: T::DTYPE,
        device,
        buffer,
    }))
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

impl QMetalStorage {
    /// Indexed MoE forward pass for quantized expert weights.
    ///
    /// Performs batched matrix-vector multiplication with dynamic expert selection.
    ///
    /// # Arguments
    /// * `self_shape` - Shape of weight tensor [num_experts, n, k]
    /// * `input` - Input tensor storage [batch, input_dim1, k]
    /// * `input_l` - Input tensor layout
    /// * `ids` - Expert indices storage [batch, topk]
    /// * `ids_l` - Expert indices layout
    ///
    /// # Returns
    /// Output tensor [batch, topk, n] and its shape
    pub fn indexed_moe_forward(
        &self,
        self_shape: &Shape,
        input: &MetalStorage,
        input_l: &crate::Layout,
        ids: &MetalStorage,
        ids_l: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !input_l.is_contiguous() {
            crate::bail!("input tensor is not contiguous {:?}", input_l)
        }
        if !ids_l.is_contiguous() {
            crate::bail!("ids tensor is not contiguous {:?}", ids_l)
        }

        // Weight shape: [num_experts, n, k]
        let w_dims = self_shape.dims();
        if w_dims.len() != 3 {
            crate::bail!(
                "indexed_moe_forward expects 3D weight tensor [num_experts, n, k], got {:?}",
                w_dims
            );
        }
        let _num_experts = w_dims[0];
        let n = w_dims[1];
        let k = w_dims[2];

        // Input shape: [batch, input_dim1, k] or [batch, k]
        let in_shape = input_l.shape();
        let in_dims = in_shape.dims();
        let (batch, input_dim1) = match in_dims.len() {
            2 => (in_dims[0], 1usize),
            3 => (in_dims[0], in_dims[1]),
            _ => crate::bail!(
                "indexed_moe_forward expects 2D or 3D input tensor, got {:?}",
                in_dims
            ),
        };
        let in_k = in_dims[in_dims.len() - 1];
        if in_k != k {
            crate::bail!(
                "input dimension {} doesn't match weight dimension {}",
                in_k,
                k
            );
        }

        // IDs shape: [batch, topk]
        let ids_shape = ids_l.shape();
        let ids_dims = ids_shape.dims();
        if ids_dims.len() != 2 {
            crate::bail!(
                "indexed_moe_forward expects 2D ids tensor [batch, topk], got {:?}",
                ids_dims
            );
        }
        if ids_dims[0] != batch {
            crate::bail!(
                "ids batch dimension {} doesn't match input batch dimension {}",
                ids_dims[0],
                batch
            );
        }
        let topk = ids_dims[1];

        // Output shape: [batch, topk, n]
        let out_shape = Shape::from(vec![batch, topk, n]);
        let out_elem_count = out_shape.elem_count();

        let device = self.device.clone();
        let dst = device.new_buffer(out_elem_count, DType::F32, "indexed_moe_forward")?;
        let encoder = device.command_encoder()?;

        paramecia_metal::call_indexed_moe_forward(
            device.device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            &self.buffer,
            input.buffer(),
            input_l.start_offset() * input.dtype().size_in_bytes(),
            ids.buffer(),
            ids_l.start_offset() * std::mem::size_of::<u32>(),
            &dst,
            n,
            k,
            batch,
            topk,
            input_dim1,
        )
        .map_err(MetalError::from)?;

        let dst_storage = MetalStorage::new(dst, device, out_elem_count, DType::F32);
        Ok((dst_storage, out_shape))
    }

    /// Fused gate+up indexed MoE forward pass.
    ///
    /// Computes both gate and up projections in a single kernel pass,
    /// reading the input only once. Returns (gate_output, up_output).
    pub fn indexed_moe_gate_up(
        gate_weights: &Self,
        up_weights: &Self,
        self_shape: &Shape,
        input: &MetalStorage,
        input_l: &crate::Layout,
        ids: &MetalStorage,
        ids_l: &crate::Layout,
    ) -> Result<((MetalStorage, Shape), (MetalStorage, Shape))> {
        use crate::MetalError;

        if !input_l.is_contiguous() {
            crate::bail!("input tensor is not contiguous {:?}", input_l)
        }
        if !ids_l.is_contiguous() {
            crate::bail!("ids tensor is not contiguous {:?}", ids_l)
        }

        // Both weight tensors must have same shape [num_experts, n, k]
        let w_dims = self_shape.dims();
        if w_dims.len() != 3 {
            crate::bail!(
                "indexed_moe_gate_up expects 3D weight tensor [num_experts, n, k], got {:?}",
                w_dims
            );
        }
        let n = w_dims[1];
        let k = w_dims[2];

        // Input shape: [batch, 1, k]
        let in_shape = input_l.shape();
        let in_dims = in_shape.dims();
        let batch = in_dims[0];
        let in_k = in_dims[in_dims.len() - 1];
        if in_k != k {
            crate::bail!(
                "input dimension {} doesn't match weight dimension {}",
                in_k,
                k
            );
        }

        // IDs shape: [batch, topk]
        let ids_shape = ids_l.shape();
        let ids_dims = ids_shape.dims();
        if ids_dims.len() != 2 || ids_dims[0] != batch {
            crate::bail!(
                "indexed_moe_gate_up expects 2D ids tensor [batch, topk], got {:?}",
                ids_dims
            );
        }
        let topk = ids_dims[1];

        // Output shape: [batch, topk, n]
        let out_shape = Shape::from(vec![batch, topk, n]);
        let out_elem_count = out_shape.elem_count();

        let device = gate_weights.device.clone();
        let gate_dst = device.new_buffer(out_elem_count, DType::F32, "indexed_moe_gate_up_gate")?;
        let up_dst = device.new_buffer(out_elem_count, DType::F32, "indexed_moe_gate_up_up")?;
        let encoder = device.command_encoder()?;

        paramecia_metal::call_indexed_moe_gate_up(
            device.device(),
            &encoder,
            device.kernels(),
            gate_weights.dtype.into(),
            &gate_weights.buffer,
            &up_weights.buffer,
            input.buffer(),
            input_l.start_offset() * input.dtype().size_in_bytes(),
            ids.buffer(),
            ids_l.start_offset() * std::mem::size_of::<u32>(),
            &gate_dst,
            &up_dst,
            n,
            k,
            batch,
            topk,
        )
        .map_err(MetalError::from)?;

        let gate_storage = MetalStorage::new(gate_dst, device.clone(), out_elem_count, DType::F32);
        let up_storage = MetalStorage::new(up_dst, device, out_elem_count, DType::F32);
        Ok(((gate_storage, out_shape.clone()), (up_storage, out_shape)))
    }
}

impl From<GgmlDType> for paramecia_metal::GgmlDType {
    fn from(value: GgmlDType) -> Self {
        match value {
            GgmlDType::Q4_0 => paramecia_metal::GgmlDType::Q4_0,
            GgmlDType::Q4_1 => paramecia_metal::GgmlDType::Q4_1,
            GgmlDType::Q5_0 => paramecia_metal::GgmlDType::Q5_0,
            GgmlDType::Q5_1 => paramecia_metal::GgmlDType::Q5_1,
            GgmlDType::Q8_0 => paramecia_metal::GgmlDType::Q8_0,
            GgmlDType::Q8_1 => paramecia_metal::GgmlDType::Q8_1,
            GgmlDType::Q2K => paramecia_metal::GgmlDType::Q2K,
            GgmlDType::Q3K => paramecia_metal::GgmlDType::Q3K,
            GgmlDType::Q4K => paramecia_metal::GgmlDType::Q4K,
            GgmlDType::Q5K => paramecia_metal::GgmlDType::Q5K,
            GgmlDType::Q6K => paramecia_metal::GgmlDType::Q6K,
            GgmlDType::Q8K => paramecia_metal::GgmlDType::Q8K,
            GgmlDType::F16 => paramecia_metal::GgmlDType::F16,
            GgmlDType::F32 => paramecia_metal::GgmlDType::F32,
            GgmlDType::BF16 => paramecia_metal::GgmlDType::F16,
        }
    }
}
