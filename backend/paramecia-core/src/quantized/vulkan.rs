#![allow(unused)]
use super::{GgmlDType, GgmlType, QStorage};
use crate::backend::BackendDevice;
use crate::{DType, Result, VulkanDevice, VulkanError, VulkanStorage};
use ash::vk;
use std::sync::Arc;
use tracing::{debug, warn};

use crate::vulkan_backend::device::VulkanBuffer;

const VENDOR_ID_NVIDIA: u32 = 0x10DE;
const VENDOR_ID_AMD: u32 = 0x1002;
const VENDOR_ID_INTEL: u32 = 0x8086;

/// Push constants for QuZO perturbation/restore_update shaders (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PerturbPushConstants {
    count: u32,
    param1: f32,
    param2: f32,
    seed1_lo: u32,
    seed1_hi: u32,
    seed2_lo: u32,
    seed2_hi: u32,
    flags: u32,
}

#[derive(Clone)]
pub struct QVulkanStorage {
    dtype: GgmlDType,
    device: VulkanDevice,
    /// GPU buffer holding the raw quantized bytes
    buffer: Option<Arc<VulkanBuffer>>,
    byte_count: usize,
    /// CPU-side data for fallback dequantize/matmul
    cpu_data: Vec<u8>,
}

impl QVulkanStorage {
    pub fn zeros(device: &VulkanDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let block_size = dtype.block_size();
        let type_size = dtype.type_size();
        let num_blocks = (elem_count + block_size - 1) / block_size;
        let byte_count = num_blocks * type_size;

        Ok(Self {
            dtype,
            device: device.clone(),
            buffer: None,
            byte_count,
            cpu_data: Vec::new(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    /// Get the raw vk::Buffer handle for the GPU buffer (if present).
    pub fn gpu_buffer(&self) -> Option<vk::Buffer> {
        self.buffer.as_ref().map(|b| b.buffer)
    }

    /// Get an Arc reference to the GPU buffer (for keeping it alive across async dispatch).
    pub fn gpu_buffer_arc(&self) -> Option<Arc<VulkanBuffer>> {
        self.buffer.clone()
    }

    /// Create a QVulkanStorage that wraps an existing GPU buffer.
    /// Does not own the buffer (it's shared via Arc). cpu_data is empty.
    pub fn from_gpu_buffer(
        dtype: GgmlDType,
        device: &VulkanDevice,
        buffer: Arc<VulkanBuffer>,
        byte_count: usize,
    ) -> Self {
        Self {
            dtype,
            device: device.clone(),
            buffer: Some(buffer),
            byte_count,
            cpu_data: Vec::new(),
        }
    }

    /// Get CPU-side data, downloading from GPU if not already cached.
    fn get_cpu_data(&self) -> Result<Vec<u8>> {
        if !self.cpu_data.is_empty() {
            return Ok(self.cpu_data.clone());
        }
        if let Some(buf) = &self.buffer {
            self.device.flush()?;
            self.device.download_buffer::<u8>(buf, self.byte_count)
        } else {
            Ok(vec![0u8; self.byte_count])
        }
    }

    /// Create a CPU-side QuantizedType from cpu_data bytes, downloading from GPU if needed.
    pub fn cpu_quantized_type(&self) -> Result<Box<dyn super::QuantizedType>> {
        use std::borrow::Cow;
        let data = self.get_cpu_data()?;
        Ok(self.dtype.from_data(Cow::Owned(data)))
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<VulkanStorage> {
        let cpu_storage = self.dequantize_to_cpu(elem_count)?;
        self.device.storage_from_cpu_storage(&cpu_storage)
    }

    fn dequantize_to_cpu(&self, elem_count: usize) -> Result<crate::CpuStorage> {
        use super::k_quants::*;
        let data = self.get_cpu_data()?;
        let mut f32_data = vec![0f32; elem_count];
        match self.dtype {
            GgmlDType::F32 => {
                let src: &[f32] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
                };
                f32_data[..elem_count.min(src.len())]
                    .copy_from_slice(&src[..elem_count.min(src.len())]);
            }
            GgmlDType::F16 => {
                let src: &[half::f16] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const half::f16, data.len() / 2)
                };
                for (i, v) in src.iter().enumerate().take(elem_count) {
                    f32_data[i] = v.to_f32();
                }
            }
            GgmlDType::BF16 => {
                let src: &[half::bf16] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, data.len() / 2)
                };
                for (i, v) in src.iter().enumerate().take(elem_count) {
                    f32_data[i] = v.to_f32();
                }
            }
            _ => {
                macro_rules! dequant {
                    ($ty:ty) => {{
                        let blocks: &[$ty] = unsafe {
                            std::slice::from_raw_parts(
                                data.as_ptr() as *const $ty,
                                data.len() / std::mem::size_of::<$ty>(),
                            )
                        };
                        <$ty as GgmlType>::to_float(blocks, &mut f32_data);
                    }};
                }
                match self.dtype {
                    GgmlDType::Q4_0 => dequant!(BlockQ4_0),
                    GgmlDType::Q4_1 => dequant!(BlockQ4_1),
                    GgmlDType::Q5_0 => dequant!(BlockQ5_0),
                    GgmlDType::Q5_1 => dequant!(BlockQ5_1),
                    GgmlDType::Q8_0 => dequant!(BlockQ8_0),
                    GgmlDType::Q8_1 => dequant!(BlockQ8_1),
                    GgmlDType::Q2K => dequant!(BlockQ2K),
                    GgmlDType::Q3K => dequant!(BlockQ3K),
                    GgmlDType::Q4K => dequant!(BlockQ4K),
                    GgmlDType::Q5K => dequant!(BlockQ5K),
                    GgmlDType::Q6K => dequant!(BlockQ6K),
                    GgmlDType::Q8K => dequant!(BlockQ8K),
                    other => crate::bail!("Vulkan dequantize: unsupported dtype {:?}", other),
                }
            }
        }
        Ok(crate::CpuStorage::F32(f32_data))
    }

    pub fn quantize(&mut self, src: &VulkanStorage) -> Result<()> {
        use crate::backend::BackendStorage;

        if crate::vulkan_backend::debug::force_cpu_quantize() {
            let cpu_src = src.to_cpu_storage()?;
            return self.quantize_onto(&cpu_src);
        }

        match self.dtype {
            GgmlDType::Q8_0 => {}
            other => {
                // Fall back to CPU for non-Q8_0 formats
                let cpu_src = src.to_cpu_storage()?;
                return self.quantize_onto(&cpu_src);
            }
        }

        let src_buf = src.vk_buffer()?;

        let block_size = self.dtype.block_size(); // 32 for Q8_0
        let type_size = self.dtype.type_size(); // 34 for Q8_0
        let num_blocks = self.byte_count / type_size;

        // Allocate output buffer (byte_count bytes, rounded up to 4-byte alignment)
        let out_size = ((self.byte_count + 3) / 4) * 4;
        let out_buf = self.device.allocate_buffer(out_size as u64)?;

        // Load pipeline (2 SSBOs: input, output; 8 bytes push constants).
        // Prefer native activation dtype variants to avoid extra cast dispatches.
        let preferred_kernel = match src.dtype() {
            DType::F16 => "quantize_q8_0_act_f16",
            DType::BF16 => "quantize_q8_0_act_bf16",
            _ => "quantize_q8_0",
        };
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(
                self.device.device(),
                preferred_kernel,
                None,
                8,
                2,
                None,
                false,
            )
            .or_else(|e| {
                if preferred_kernel == "quantize_q8_0" {
                    Err(e)
                } else {
                    self.device.kernels().load_pipeline(
                        self.device.device(),
                        "quantize_q8_0",
                        None,
                        8,
                        2,
                        None,
                        false,
                    )
                }
            })
            .map_err(|e| crate::Error::Vulkan(VulkanError::from(e)))?;

        // Dispatch: ceil(num_blocks/2) workgroups (2 blocks per workgroup)
        let dispatch = [((num_blocks as u32) + 1) / 2, 1, 1];
        let pc: [u32; 2] = [num_blocks as u32, 0]; // output_byte_offset = 0

        // Record compute with push constants
        let tmp_storage = VulkanStorage::new(
            Some(Arc::new(out_buf)),
            self.device.clone(),
            out_size / 4,
            DType::U32,
        );
        self.device.record_compute_with_write_mask(
            &pipeline,
            &[src_buf, tmp_storage.vk_buffer()?],
            Some(0b10),
            Some(bytemuck::cast_slice(&pc)),
            dispatch,
        )?;

        // Download quantized bytes from GPU
        self.device.flush()?;
        let out_vk_buf = tmp_storage
            .buffer()
            .ok_or_else(|| crate::Error::Msg("quantize output buffer missing".into()))?;
        let out_bytes = self.device.download_buffer::<u8>(out_vk_buf, out_size)?;
        self.cpu_data = out_bytes[..self.byte_count].to_vec();

        // Upload quantized data to a properly sized GPU buffer
        let gpu_buf = self.device.allocate_buffer(self.byte_count as u64)?;
        self.device.upload_to_buffer(&self.cpu_data, &gpu_buf)?;
        self.buffer = Some(Arc::new(gpu_buf));

        Ok(())
    }

    /// Quantize src (f32/f16/bf16 GPU tensor) into this storage, keeping the result on GPU only.
    /// Unlike `quantize()`, this does NOT download to CPU or re-upload.
    /// The `cpu_data` field is left empty. Use this for GPU-resident KV caches.
    pub fn quantize_gpu_only(&mut self, src: &VulkanStorage) -> Result<()> {
        use crate::backend::BackendStorage;

        match self.dtype {
            GgmlDType::Q8_0 => {}
            other => crate::bail!("quantize_gpu_only: only Q8_0 supported, got {:?}", other),
        }

        let src_buf = src.vk_buffer()?;
        let type_size = self.dtype.type_size(); // 34 for Q8_0
        let num_blocks = self.byte_count / type_size;

        let out_size = ((self.byte_count + 3) / 4) * 4;
        let out_buf = Arc::new(self.device.allocate_buffer(out_size as u64)?);

        let preferred_kernel = match src.dtype() {
            DType::F16 => "quantize_q8_0_act_f16",
            DType::BF16 => "quantize_q8_0_act_bf16",
            _ => "quantize_q8_0",
        };
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(
                self.device.device(),
                preferred_kernel,
                None,
                8,
                2,
                None,
                false,
            )
            .or_else(|e| {
                if preferred_kernel == "quantize_q8_0" {
                    Err(e)
                } else {
                    self.device.kernels().load_pipeline(
                        self.device.device(),
                        "quantize_q8_0",
                        None,
                        8,
                        2,
                        None,
                        false,
                    )
                }
            })
            .map_err(|e| crate::Error::Vulkan(VulkanError::from(e)))?;

        let dispatch = [((num_blocks as u32) + 1) / 2, 1, 1];
        let pc: [u32; 2] = [num_blocks as u32, 0]; // output_byte_offset = 0

        self.device.record_compute_with_write_mask(
            &pipeline,
            &[src_buf, out_buf.buffer],
            Some(0b10),
            Some(bytemuck::cast_slice(&pc)),
            dispatch,
        )?;

        // Keep result on GPU — no download, no re-upload
        self.buffer = Some(out_buf);
        // cpu_data left empty (not synced)
        self.cpu_data.clear();
        Ok(())
    }

    /// Quantize f32/f16/bf16 GPU data directly into an existing GPU buffer at a byte offset.
    /// No temporary buffers are allocated — the quantize shader writes directly to `dst_buffer`.
    /// `dst_byte_offset` must be 4-byte aligned.
    /// `elem_count` is the number of input elements (must be multiple of block_size=32).
    pub fn quantize_into_buffer(
        device: &VulkanDevice,
        src: &VulkanStorage,
        dst_buffer: vk::Buffer,
        dst_byte_offset: usize,
        elem_count: usize,
    ) -> Result<()> {
        use crate::backend::BackendStorage;

        let block_size = GgmlDType::Q8_0.block_size(); // 32
        let type_size = GgmlDType::Q8_0.type_size(); // 34
        let num_blocks = elem_count / block_size;

        if elem_count % block_size != 0 {
            crate::bail!(
                "quantize_into_buffer: elem_count {} not multiple of block_size {}",
                elem_count,
                block_size
            );
        }
        if dst_byte_offset % 4 != 0 {
            crate::bail!(
                "quantize_into_buffer: dst_byte_offset {} not 4-byte aligned",
                dst_byte_offset
            );
        }

        let src_buf = src.vk_buffer()?;

        let preferred_kernel = match src.dtype() {
            DType::F16 => "quantize_q8_0_act_f16",
            DType::BF16 => "quantize_q8_0_act_bf16",
            _ => "quantize_q8_0",
        };
        let pipeline = device
            .kernels()
            .load_pipeline(device.device(), preferred_kernel, None, 8, 2, None, false)
            .or_else(|e| {
                if preferred_kernel == "quantize_q8_0" {
                    Err(e)
                } else {
                    device.kernels().load_pipeline(
                        device.device(),
                        "quantize_q8_0",
                        None,
                        8,
                        2,
                        None,
                        false,
                    )
                }
            })
            .map_err(|e| crate::Error::Vulkan(VulkanError::from(e)))?;

        let dispatch = [((num_blocks as u32) + 1) / 2, 1, 1];
        let pc: [u32; 2] = [num_blocks as u32, dst_byte_offset as u32];

        device.record_compute_with_write_mask(
            &pipeline,
            &[src_buf, dst_buffer],
            Some(0b10),
            Some(bytemuck::cast_slice(&pc)),
            dispatch,
        )?;

        Ok(())
    }

    /// Dequantize Q8_0 data on GPU, returning a f32 VulkanStorage.
    /// `input_byte_offset` specifies where in the quantized buffer to start reading.
    /// `elem_count` is the number of f32 elements to produce.
    /// Requires the quantized data to be on GPU (buffer must be Some).
    pub fn dequantize_gpu(
        &self,
        elem_count: usize,
        input_byte_offset: usize,
    ) -> Result<VulkanStorage> {
        match self.dtype {
            GgmlDType::Q8_0 => {}
            other => crate::bail!("dequantize_gpu: only Q8_0 supported, got {:?}", other),
        }

        let src_buf = self
            .buffer
            .as_ref()
            .ok_or_else(|| crate::Error::Msg("dequantize_gpu: no GPU buffer".into()))?;

        let out_elems = elem_count;
        let out_buf = self.device.allocate_buffer((out_elems * 4) as u64)?;

        // Push constants: 8 bytes (num_elements, input_byte_offset)
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(
                self.device.device(),
                "dequantize_q8_0",
                None,
                8,
                2,
                None,
                false,
            )
            .map_err(|e| crate::Error::Vulkan(VulkanError::from(e)))?;

        let dispatch = [((elem_count as u32) + 255) / 256, 1, 1];
        let pc: [u32; 2] = [elem_count as u32, input_byte_offset as u32];

        let out_storage = VulkanStorage::new(
            Some(Arc::new(out_buf)),
            self.device.clone(),
            out_elems,
            DType::F32,
        );
        self.device.record_compute_with_write_mask(
            &pipeline,
            &[src_buf.buffer, out_storage.vk_buffer()?],
            Some(0b10),
            Some(bytemuck::cast_slice(&pc)),
            dispatch,
        )?;

        Ok(out_storage)
    }

    pub fn quantize_imatrix(
        &mut self,
        _src: &VulkanStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        crate::bail!("Vulkan quantize_imatrix not yet implemented")
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        _src: &crate::CpuStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        crate::bail!("Vulkan quantize_imatrix_onto not yet implemented")
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Keep Vulkan behavior aligned with CPU/CUDA/Metal by quantizing through
        // the CPU quantized storage implementation, then uploading raw bytes.
        let src_f32 = src.as_slice::<f32>()?;
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_f32.len(), self.dtype)?;

        match &mut qcpu_storage {
            QStorage::Cpu(storage) => storage.from_float(src_f32),
            _ => crate::bail!("Vulkan quantize_onto: internal error creating CPU qstorage"),
        }

        let data = qcpu_storage.data()?;
        self.byte_count = data.len();
        self.cpu_data = data.into_owned();

        if self.cpu_data.is_empty() {
            self.buffer = None;
            return Ok(());
        }

        let gpu_buf = self.device.allocate_buffer(self.cpu_data.len() as u64)?;
        self.device.upload_to_buffer(&self.cpu_data, &gpu_buf)?;
        self.buffer = Some(Arc::new(gpu_buf));
        Ok(())
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        crate::bail!("Vulkan device_ptr not applicable")
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.byte_count
    }

    pub fn slice(&self, offset: usize, size: usize) -> Result<Self> {
        let data = self.get_cpu_data()?;
        if offset + size > data.len() {
            crate::bail!(
                "QVulkanStorage slice out of bounds: offset={offset} size={size} total={}",
                data.len()
            );
        }
        let sliced = data[offset..offset + size].to_vec();

        // Eagerly upload sliced data to GPU so that fwd() can use GPU shaders.
        // Without this, fwd() falls back to CPU (sync download + CPU matmul + upload),
        // which causes pipeline stalls and crashes with mixed-device expert offloading.
        let buffer = if sliced.is_empty() {
            None
        } else {
            let buf = self.device.allocate_buffer(sliced.len() as u64)?;
            self.device.upload_to_buffer(&sliced, &buf)?;
            Some(Arc::new(buf))
        };

        Ok(Self {
            dtype: self.dtype,
            device: self.device.clone(),
            buffer,
            byte_count: size,
            cpu_data: sliced,
        })
    }

    /// Select fused matmul+perturb kernel name for this quantization format.
    fn fused_kernel_name(&self) -> Option<&'static str> {
        match self.dtype {
            GgmlDType::Q8_0 => Some("fused_matmul_q8_0"),
            GgmlDType::Q4K => Some("fused_matmul_q4_k"),
            GgmlDType::Q2K => Some("fused_matmul_q2_k"),
            GgmlDType::Q3K => Some("fused_matmul_q3_k"),
            GgmlDType::Q5K => Some("fused_matmul_q5_k"),
            GgmlDType::Q6K => Some("fused_matmul_q6_k"),
            GgmlDType::BF16 => Some("fused_matmul_bf16"),
            _ => None,
        }
    }

    /// Get a unique per-tensor ID for fused perturbation seeding.
    /// Uses the raw Vulkan buffer handle as a stable unique identifier.
    pub fn fused_tensor_id(&self) -> u64 {
        use ash::vk::Handle;
        self.buffer.as_ref().map(|b| b.buffer.as_raw()).unwrap_or(0)
    }

    /// Fused dequantize + perturb + matmul on GPU.
    ///
    /// Like fwd() but adds on-the-fly perturbation during dequantization:
    /// w_perturbed = w_dequant + epsilon * philox_gaussian(seed ^ tensor_id, weight_index)
    pub fn fused_fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &VulkanStorage,
        layout: &crate::Layout,
        seed: u64,
        epsilon: f32,
    ) -> Result<(VulkanStorage, crate::Shape)> {
        let kernel_name = match self.fused_kernel_name() {
            Some(name) => name,
            None => crate::bail!("fused_fwd not supported for {:?}", self.dtype),
        };

        let weight_buf = match &self.buffer {
            Some(b) => b.buffer,
            None => crate::bail!("fused_fwd: no GPU buffer for weights"),
        };

        // Weight shape: [n, k]
        let n = self_shape.dim(0)?;
        let k = self_shape.dim(1)?;

        // Parse activation dimensions
        let x_dims = layout.shape().dims();
        let (batch, m) = match x_dims.len() {
            2 => (1usize, x_dims[0]),
            3 => (x_dims[0], x_dims[1]),
            _ => crate::bail!("Expected 2D or 3D input for fused_fwd, got {:?}", x_dims),
        };
        let x_k = x_dims[x_dims.len() - 1];
        if x_k != k {
            crate::bail!("fused_fwd dim mismatch: input k={x_k} vs weight k={k}");
        }

        // Ensure activations are f32
        use crate::backend::BackendStorage;
        let (f32_storage, f32_layout) = if storage.dtype() != DType::F32 {
            let cast = storage.to_dtype(layout, DType::F32)?;
            let cast_layout = crate::Layout::contiguous(layout.shape());
            (Some(cast), cast_layout)
        } else {
            (None, layout.clone())
        };
        let act_storage = f32_storage.as_ref().unwrap_or(storage);
        let act_buf = act_storage.vk_buffer()?;

        // Compute activation strides
        let strides = f32_layout.stride();
        let (a_stride_batch, a_stride_m, a_stride_k) = match strides.len() {
            2 => (0u32, strides[0] as u32, strides[1] as u32),
            3 => (strides[0] as u32, strides[1] as u32, strides[2] as u32),
            _ => (0, k as u32, 1),
        };
        let a_offset = f32_layout.start_offset() as u32;

        // Build push constants: 12 x u32 = 48 bytes
        let tensor_id = self.fused_tensor_id();
        let params: [u32; 12] = [
            m as u32,                 // [0] m
            k as u32,                 // [1] k
            n as u32,                 // [2] n
            a_stride_batch,           // [3] a_stride_batch
            a_offset,                 // [4] a_offset
            a_stride_k,               // [5] a_stride_k
            a_stride_m,               // [6] a_stride_m
            seed as u32,              // [7] seed_lo
            (seed >> 32) as u32,      // [8] seed_hi
            epsilon.to_bits(),        // [9] epsilon as u32 bits
            tensor_id as u32,         // [10] tensor_id_lo
            (tensor_id >> 32) as u32, // [11] tensor_id_hi
        ];

        // Allocate output buffer
        let out_elems = batch * m * n;
        let out_buf = self.device.allocate_buffer((out_elems * 4) as u64)?;

        // Get specialization constants if using cooperative matrix
        let specialization = if kernel_name.ends_with("_coopmat") {
            let (tile_m, tile_n, tile_k) =
                self.device.coop_matrix_tile_size().unwrap_or((16, 16, 16));
            Some(vec![tile_m, tile_n, tile_k])
        } else {
            None
        };

        // Load pipeline: 3 SSBOs, 48 bytes push constants
        // If cooperative matrix shader fails to compile, fall back to standard shader
        let pipeline = match self.device.kernels().load_pipeline(
            self.device.device(),
            kernel_name,
            None,
            48,
            3,
            specialization.as_deref(),
            false,
        ) {
            Ok(p) => p,
            Err(e) if kernel_name.ends_with("_coopmat") => {
                // Cooperative matrix shader failed, try fallback
                let fallback_name = kernel_name.trim_end_matches("_coopmat");
                let fallback_name = if fallback_name == "matmul_q8_0" {
                    if std::env::var("PARAMECIA_DISABLE_DP4A").is_ok() {
                        "matmul_q8_0"
                    } else {
                        "matmul_q8_0_dp4a"
                    }
                } else {
                    fallback_name
                };
                warn!(
                    "[VK] Cooperative matrix shader '{}' failed to compile, falling back to '{}'. Error: {:?}",
                    kernel_name, fallback_name, e
                );
                self.device
                    .kernels()
                    .load_pipeline(
                        self.device.device(),
                        fallback_name,
                        None,
                        48,
                        3,
                        None,
                        false,
                    )
                    .map_err(|e2| crate::Error::Vulkan(VulkanError::from(e2)))?
            }
            Err(e) => return Err(crate::Error::Vulkan(VulkanError::from(e))),
        };

        // Dispatch: 4 columns per workgroup (NWARPS=4)
        let dispatch = [(n as u32 + 3) / 4, m as u32, batch as u32];

        let out_storage = VulkanStorage::new(
            Some(Arc::new(out_buf)),
            self.device.clone(),
            out_elems,
            DType::F32,
        );
        self.device.record_compute_with_write_mask(
            &pipeline,
            &[out_storage.vk_buffer()?, act_buf, weight_buf],
            Some(0b001),
            Some(bytemuck::cast_slice(&params)),
            dispatch,
        )?;

        let result_shape = if x_dims.len() == 2 {
            crate::Shape::from_dims(&[m, n])
        } else {
            crate::Shape::from_dims(&[batch, m, n])
        };

        // Cast output back to input dtype if needed
        let input_dtype = storage.dtype();
        if input_dtype != DType::F32 {
            let out_layout = crate::Layout::contiguous(&result_shape);
            let cast_out = out_storage.to_dtype(&out_layout, input_dtype)?;
            Ok((cast_out, result_shape))
        } else {
            Ok((out_storage, result_shape))
        }
    }

    /// Check if coopmat should be disabled for a specific dtype and operation.
    ///
    /// Coopmat is capability-driven by default when the device supports it.
    /// `MOE_GATE_UP` remains opt-in until the coopmat fused SwiGLU path is numerically stable.
    /// Checks in order of specificity:
    /// 1. Global disable flag: PARAMECIA_DISABLE_COOPMAT
    /// 2. Operation-wide (all dtypes): PARAMECIA_DISABLE_COOPMAT_MATMUL
    /// 3. Operation+dtype specific: PARAMECIA_DISABLE_COOPMAT_Q2_K_MATMUL
    /// 4. Dtype-wide: PARAMECIA_DISABLE_COOPMAT_Q2_K
    fn is_coopmat_disabled(&self, operation: &str) -> bool {
        // Global opt-out for all coopmat kernels.
        if std::env::var("PARAMECIA_DISABLE_COOPMAT").is_ok() {
            return true;
        }

        // Keep fused gate+up SwiGLU coopmat kernels opt-in for now.
        if operation == "MOE_GATE_UP"
            && std::env::var("PARAMECIA_ENABLE_COOPMAT_MOE_GATE_UP").is_err()
        {
            return true;
        }

        // Check operation-wide env var first (e.g., PARAMECIA_DISABLE_COOPMAT_MATMUL)
        // This disables coopmat for the operation across ALL dtypes
        let op_wide = format!("PARAMECIA_DISABLE_COOPMAT_{}", operation);
        if std::env::var(&op_wide).is_ok() {
            return true;
        }

        let dtype_suffix = match self.dtype {
            GgmlDType::Q8_0 => "Q8_0",
            GgmlDType::Q4K => "Q4_K",
            GgmlDType::Q5K => "Q5_K",
            GgmlDType::Q6K => "Q6_K",
            GgmlDType::Q3K => "Q3_K",
            GgmlDType::Q2K => "Q2_K",
            _ => return true, // Disable for unsupported types
        };

        // Check operation+dtype specific env var (e.g., PARAMECIA_DISABLE_COOPMAT_Q2_K_MATMUL)
        let op_dtype_specific = format!("PARAMECIA_DISABLE_COOPMAT_{}_{}", dtype_suffix, operation);
        if std::env::var(&op_dtype_specific).is_ok() {
            return true;
        }

        // Fall back to dtype-wide env var (e.g., PARAMECIA_DISABLE_COOPMAT_Q2_K)
        let dtype_wide = format!("PARAMECIA_DISABLE_COOPMAT_{}", dtype_suffix);
        std::env::var(&dtype_wide).is_ok()
    }

    /// Select the matmul prefill threshold (minimum `m`) for tiled prefill kernels.
    /// Uses simple shape/device heuristics so larger GPUs and larger reductions switch
    /// earlier, while smaller devices keep decode-style kernels for longer.
    fn tiled_prefill_threshold(&self, n: usize, k: usize, batch: usize) -> usize {
        let cores = self.device.shader_core_count();
        let mut threshold = if cores >= 80 {
            8usize
        } else if cores >= 40 {
            12usize
        } else if cores > 0 {
            16usize
        } else {
            14usize
        };

        if k >= 4096 {
            threshold = threshold.saturating_sub(2);
        }
        if n >= 4096 {
            threshold = threshold.saturating_sub(2);
        }
        if batch > 1 {
            threshold = threshold.saturating_sub(1);
        }

        threshold.clamp(4, 24)
    }

    /// Decide if tiled prefill coopmat should be used for this shape/device.
    fn can_use_tiled_prefill_coopmat(
        &self,
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
        force: bool,
    ) -> bool {
        if self.is_coopmat_disabled("MATMUL") || !self.device.has_cooperative_matrix() {
            return false;
        }
        if self.device.subgroup_size() != 32 {
            return false;
        }

        let (tile_m, tile_n, tile_k) = match self.device.coop_matrix_tile_size() {
            Some(tile) => tile,
            None => return false,
        };

        // Tiled coopmat shaders are hard-tuned for 16x16x16 tensor-core operations.
        if tile_m != 16 || tile_n != 16 || tile_k != 16 {
            return false;
        }

        // Current tiled coopmat kernels require full COOP_K chunks to avoid tail loss.
        if tile_k == 0 || (k as u32) % tile_k != 0 {
            return false;
        }

        if force {
            return true;
        }

        let cores = self.device.shader_core_count();
        let (mut min_m, min_n, min_k) = match self.device.vendor_id() {
            VENDOR_ID_NVIDIA => {
                if cores >= 80 {
                    (16usize, 2048usize, 2048usize)
                } else if cores >= 40 {
                    (24usize, 3072usize, 3072usize)
                } else {
                    (32usize, 4096usize, 4096usize)
                }
            }
            VENDOR_ID_AMD | VENDOR_ID_INTEL => (32usize, 4096usize, 4096usize),
            _ => return false,
        };

        if batch > 1 {
            min_m = min_m.saturating_sub(8).max(8);
        }

        m >= min_m && n >= min_n && k >= min_k
    }

    /// Select GPU kernel name for this quantization format.
    /// Returns None for formats without a dedicated GPU shader.
    ///
    /// Kernel selection priority (for m>1):
    /// 1. Hierarchical tiled kernels (PARAMECIA_ENABLE_TILED) - Phase 1-3 implementation
    ///    - Decode-optimized (m<=8): BM=16, 2 warps
    ///    - Prefill-optimized (m>8): BM=64, 4 warps
    /// 2. Cooperative matrix kernels (if supported and enabled)
    /// 3. Standard kernels (fallback)
    fn kernel_name(
        &self,
        is_matvec: bool,
        m: usize,
        n: usize,
        k: usize,
        batch: usize,
    ) -> Option<&'static str> {
        // Check if mat-vec kernels are disabled via environment variable
        let disable_matvec_global = std::env::var("PARAMECIA_DISABLE_MATVEC").is_ok();

        // Check per-dtype disable flags
        let disable_matvec_dtype = match self.dtype {
            GgmlDType::Q8_0 => std::env::var("PARAMECIA_DISABLE_MATVEC_Q8_0").is_ok(),
            GgmlDType::Q4K => std::env::var("PARAMECIA_DISABLE_MATVEC_Q4K").is_ok(),
            GgmlDType::Q5K => std::env::var("PARAMECIA_DISABLE_MATVEC_Q5K").is_ok(),
            GgmlDType::Q6K => std::env::var("PARAMECIA_DISABLE_MATVEC_Q6K").is_ok(),
            GgmlDType::Q3K => std::env::var("PARAMECIA_DISABLE_MATVEC_Q3K").is_ok(),
            GgmlDType::Q2K => std::env::var("PARAMECIA_DISABLE_MATVEC_Q2K").is_ok(),
            _ => false,
        };

        let disable_matvec = disable_matvec_global || disable_matvec_dtype;

        // Use specialized mat-vec kernels for decode phase (m=1) unless disabled
        if is_matvec && !disable_matvec {
            let use_coop = std::env::var("PARAMECIA_DISABLE_COOP_MATVEC").is_err();
            // IDP path is best on larger reductions, but has extra quantization overhead.
            let force_idp = std::env::var("PARAMECIA_FORCE_IDP").is_ok();
            let auto_idp = std::env::var("PARAMECIA_DISABLE_IDP_AUTO").is_err();
            let idp_supported = self.device.has_integer_dot_product();
            let idp_enabled = std::env::var("PARAMECIA_DISABLE_IDP").is_err();
            let idp_beneficial = k >= 1024 || n >= 2048 || self.device.shader_core_count() >= 64;
            let use_idp = use_coop
                && idp_supported
                && idp_enabled
                && (force_idp || (auto_idp && idp_beneficial));

            return match self.dtype {
                GgmlDType::Q8_0 => {
                    if use_idp {
                        // Packed i8 dot product: quantize activations on-the-fly,
                        // avoid per-element i8→f32 dequant of weights
                        Some("matmul_vec_q8_0_idp")
                    } else if use_coop {
                        Some("matmul_vec_q8_0_coop")
                    } else {
                        Some("matmul_vec_q8_0")
                    }
                }
                GgmlDType::Q4K => Some(if use_coop {
                    "matmul_vec_q4_k_coop"
                } else {
                    "matmul_vec_q4_k"
                }),
                GgmlDType::Q5K => Some(if use_coop {
                    "matmul_vec_q5_k_coop"
                } else {
                    "matmul_vec_q5_k"
                }),
                GgmlDType::Q6K => Some(if use_coop {
                    "matmul_vec_q6_k_coop"
                } else {
                    "matmul_vec_q6_k"
                }),
                GgmlDType::Q3K => Some(if use_coop {
                    "matmul_vec_q3_k_coop"
                } else {
                    "matmul_vec_q3_k"
                }),
                GgmlDType::Q2K => Some(if use_coop {
                    "matmul_vec_q2_k_coop"
                } else {
                    "matmul_vec_q2_k"
                }),
                _ => None,
            };
        }

        // Matrix-matrix path (prefill or matvec disabled)
        // Check if hierarchical tiled kernels are enabled (Phase 1: opt-in)
        let enable_tiled = std::env::var("PARAMECIA_ENABLE_TILED").is_ok();
        let disable_tiled = std::env::var("PARAMECIA_DISABLE_TILED").is_ok();
        let use_tiled = enable_tiled && !disable_tiled;

        if use_tiled {
            // Hierarchical tiled kernels (Phase 1-3: Q8_0 and K-quants)
            // Select decode vs prefill variant with shape/device-aware threshold.
            let use_prefill = m >= self.tiled_prefill_threshold(n, k, batch);

            // Phase 4 & 6: Use optimized variants by default, allow override
            let disable_opt = std::env::var("PARAMECIA_DISABLE_OPT").is_ok();
            let use_vec = std::env::var("PARAMECIA_ENABLE_VEC").is_ok();
            let use_db = std::env::var("PARAMECIA_ENABLE_DB").is_ok();
            let force_coopmat = std::env::var("PARAMECIA_FORCE_COOPMAT").is_ok();
            let enable_tiled_coopmat = std::env::var("PARAMECIA_ENABLE_TILED_COOPMAT").is_ok();
            let disable_tiled_coopmat = std::env::var("PARAMECIA_DISABLE_TILED_COOPMAT").is_ok();
            let allow_tiled_coopmat = !disable_tiled_coopmat
                && (force_coopmat
                    || enable_tiled_coopmat
                    || std::env::var("PARAMECIA_DISABLE_COOPMAT").is_err());
            let use_tiled_coopmat = allow_tiled_coopmat
                && self.can_use_tiled_prefill_coopmat(m, n, k, batch, force_coopmat);
            let cores = self.device.shader_core_count();

            return match self.dtype {
                GgmlDType::Q8_0 => {
                    if use_prefill {
                        let auto_db =
                            !disable_opt && cores >= 64 && m >= 32 && n >= 4096 && k >= 4096;
                        let auto_vec = !disable_opt && !auto_db && n >= 2048 && k >= 2048;
                        if use_tiled_coopmat {
                            // Phase 6: Hierarchical tiled coopmat (tensor cores)
                            Some("matmul_tiled_q8_0_prefill_coopmat")
                        } else if use_db || auto_db {
                            Some("matmul_tiled_q8_0_prefill_db")
                        } else if use_vec || auto_vec {
                            Some("matmul_tiled_q8_0_prefill_vec")
                        } else if disable_opt {
                            Some("matmul_tiled_q8_0_prefill")
                        } else {
                            // Phase 4: Use K-loop unrolled variant by default (17-26% faster)
                            Some("matmul_tiled_q8_0_prefill_opt")
                        }
                    } else {
                        Some("matmul_tiled_q8_0")
                    }
                }
                GgmlDType::Q2K => {
                    if use_prefill {
                        if use_tiled_coopmat {
                            // Phase 6: Hierarchical tiled coopmat (tensor cores)
                            Some("matmul_tiled_q2_k_prefill_coopmat")
                        } else {
                            Some("matmul_tiled_q2_k_prefill")
                        }
                    } else {
                        Some("matmul_tiled_q2_k")
                    }
                }
                GgmlDType::Q3K => {
                    if use_prefill {
                        if use_tiled_coopmat {
                            // Phase 6: Hierarchical tiled coopmat (tensor cores)
                            Some("matmul_tiled_q3_k_prefill_coopmat")
                        } else {
                            Some("matmul_tiled_q3_k_prefill")
                        }
                    } else {
                        Some("matmul_tiled_q3_k")
                    }
                }
                GgmlDType::Q4K => {
                    if use_prefill {
                        if use_tiled_coopmat {
                            // Phase 6: Hierarchical tiled coopmat (tensor cores)
                            Some("matmul_tiled_q4_k_prefill_coopmat")
                        } else {
                            Some("matmul_tiled_q4_k_prefill")
                        }
                    } else {
                        Some("matmul_tiled_q4_k")
                    }
                }
                GgmlDType::Q5K => {
                    if use_prefill {
                        if use_tiled_coopmat {
                            // Phase 6: Hierarchical tiled coopmat (tensor cores)
                            Some("matmul_tiled_q5_k_prefill_coopmat")
                        } else {
                            Some("matmul_tiled_q5_k_prefill")
                        }
                    } else {
                        Some("matmul_tiled_q5_k")
                    }
                }
                GgmlDType::Q6K => {
                    if use_prefill {
                        if use_tiled_coopmat {
                            // Phase 6: Hierarchical tiled coopmat (tensor cores)
                            Some("matmul_tiled_q6_k_prefill_coopmat")
                        } else {
                            Some("matmul_tiled_q6_k_prefill")
                        }
                    } else {
                        Some("matmul_tiled_q6_k")
                    }
                }
                _ => None,
            };
        }

        let has_coopmat = self.device.has_cooperative_matrix();

        match self.dtype {
            GgmlDType::Q8_0 => {
                if has_coopmat && !self.is_coopmat_disabled("MATMUL") {
                    Some("matmul_q8_0_coopmat")
                } else if std::env::var("PARAMECIA_DISABLE_DP4A").is_ok() {
                    Some("matmul_q8_0")
                } else {
                    Some("matmul_q8_0_dp4a")
                }
            }
            GgmlDType::Q4K => Some(if has_coopmat && !self.is_coopmat_disabled("MATMUL") {
                "matmul_q4_k_coopmat"
            } else {
                "matmul_q4_k"
            }),
            GgmlDType::Q5K => Some(if has_coopmat && !self.is_coopmat_disabled("MATMUL") {
                "matmul_q5_k_coopmat"
            } else {
                "matmul_q5_k"
            }),
            GgmlDType::Q6K => Some(if has_coopmat && !self.is_coopmat_disabled("MATMUL") {
                "matmul_q6_k_coopmat"
            } else {
                "matmul_q6_k"
            }),
            GgmlDType::Q3K => Some(if has_coopmat && !self.is_coopmat_disabled("MATMUL") {
                "matmul_q3_k_coopmat"
            } else {
                "matmul_q3_k"
            }),
            GgmlDType::Q2K => Some(if has_coopmat && !self.is_coopmat_disabled("MATMUL") {
                "matmul_q2_k_coopmat"
            } else {
                "matmul_q2_k"
            }),
            _ => None,
        }
    }

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &VulkanStorage,
        layout: &crate::Layout,
    ) -> Result<(VulkanStorage, crate::Shape)> {
        // Weight shape: [n, k] where n=output_dim, k=input_dim
        let n = self_shape.dim(0)?;
        let k = self_shape.dim(1)?;

        // Parse activation dimensions
        let x_dims = layout.shape().dims();
        let (batch, m) = match x_dims.len() {
            2 => (1usize, x_dims[0]),
            3 => (x_dims[0], x_dims[1]),
            _ => crate::bail!(
                "Expected 2D or 3D input for quantized fwd, got {:?}",
                x_dims
            ),
        };
        let x_k = x_dims[x_dims.len() - 1];
        if x_k != k {
            crate::bail!("Quantized fwd dim mismatch: input k={x_k} vs weight k={k}");
        }

        // Detect decode phase: m=1 (single row) -> use specialized mat-vec kernels
        let is_matvec = m == 1;

        // Try GPU dispatch first, fall back to CPU for unsupported formats
        let kernel_name = match self.kernel_name(is_matvec, m, n, k, batch) {
            Some(name) => name,
            None => return self.fwd_cpu_fallback(self_shape, storage, layout),
        };
        let mut selected_kernel = kernel_name.to_string();

        debug!(
            "[DEBUG] QTensor::fwd: dtype={:?}, m={}, n={}, k={}, batch={}, is_matvec={}, kernel={}",
            self.dtype, m, n, k, batch, is_matvec, kernel_name
        );
        // Check env var override for this specific dtype
        if crate::vulkan_backend::debug::force_cpu_qmatmul(kernel_name) {
            return self.fwd_cpu_fallback(self_shape, storage, layout);
        }
        let weight_buf = match &self.buffer {
            Some(b) => b.buffer,
            None => return self.fwd_cpu_fallback(self_shape, storage, layout),
        };

        // Prefer native f16/bf16 activation paths for default Q8_0 kernels to avoid
        // an extra cast dispatch + bandwidth overhead.
        let supports_native_act = matches!(
            kernel_name,
            "matmul_q8_0_dp4a" | "matmul_vec_q8_0_idp" | "matmul_vec_q8_0_coop"
        );
        let use_native_act = supports_native_act
            && std::env::var("PARAMECIA_DISABLE_NATIVE_Q8_ACT").is_err()
            && match storage.dtype() {
                DType::F16 => {
                    selected_kernel.push_str("_act_f16");
                    true
                }
                DType::BF16 => {
                    selected_kernel.push_str("_act_bf16");
                    true
                }
                _ => false,
            };

        // Ensure activations are f32 for kernels without native activation support.
        use crate::backend::BackendStorage;
        let (f32_storage, f32_layout) = if storage.dtype() != DType::F32 && !use_native_act {
            let cast = storage.to_dtype(layout, DType::F32)?;
            let cast_layout = crate::Layout::contiguous(layout.shape());
            (Some(cast), cast_layout)
        } else {
            (None, layout.clone())
        };
        let act_storage = f32_storage.as_ref().unwrap_or(storage);
        let act_buf = act_storage.vk_buffer()?;

        // Compute activation strides
        let strides = f32_layout.stride();
        let (a_stride_batch, a_stride_m, a_stride_k) = match strides.len() {
            2 => (0u32, strides[0] as u32, strides[1] as u32),
            3 => (strides[0] as u32, strides[1] as u32, strides[2] as u32),
            _ => (0, k as u32, 1),
        };
        let a_offset = f32_layout.start_offset() as u32;

        // Push constants: 8 x u32 = 32 bytes
        let params: [u32; 8] = [
            m as u32,       // [0] m (output rows)
            k as u32,       // [1] k (reduction dimension)
            n as u32,       // [2] n (output columns)
            a_stride_batch, // [3] a_stride_batch
            a_offset,       // [4] a_offset
            a_stride_k,     // [5] a_stride_k
            a_stride_m,     // [6] a_stride_m
            0,              // [7] unused
        ];

        // Allocate output buffer (batch * m * n * sizeof(f32))
        let out_elems = batch * m * n;
        let out_buf = self.device.allocate_buffer((out_elems * 4) as u64)?;

        // Specialization constants for cooperative matvec variants
        let coop_block_size = 32u32;
        let coop_num_rows = 2u32;
        let coop_q8_spec = [coop_block_size, coop_num_rows];
        let coop_kquant_spec = [coop_num_rows];
        let mut coopmat_spec: Option<Vec<u32>> = None;
        let specialization = if selected_kernel == "matmul_vec_q8_0_coop"
            || selected_kernel == "matmul_vec_q8_0_idp"
        {
            // Q8_0 coop/IDP: spec constants [BLOCK_SIZE, NUM_ROWS]
            Some(coop_q8_spec.as_slice())
        } else if selected_kernel.starts_with("matmul_vec_q") && selected_kernel.ends_with("_coop")
        {
            // K-quant coop: spec constant [NUM_ROWS]
            Some(coop_kquant_spec.as_slice())
        } else if selected_kernel.ends_with("_coopmat") {
            if let Some((tile_m, tile_n, tile_k)) = self.device.coop_matrix_tile_size() {
                coopmat_spec = Some(vec![tile_m, tile_n, tile_k]);
                coopmat_spec.as_deref()
            } else {
                None
            }
        } else {
            None
        };

        // Cooperative matvec kernels require full subgroups for correctness of subgroupAdd
        let mut is_coop_kernel = selected_kernel == "matmul_vec_q8_0_coop"
            || selected_kernel == "matmul_vec_q8_0_idp"
            || (selected_kernel.starts_with("matmul_vec_q") && selected_kernel.ends_with("_coop"));
        let mut require_full_subgroups = is_coop_kernel || selected_kernel.ends_with("_coopmat");

        // Load pipeline (3 SSBOs: output, inputA, inputB; 32 bytes push constants)
        let pipeline = self
            .device
            .kernels()
            .load_pipeline(
                self.device.device(),
                &selected_kernel,
                None,
                32,
                3,
                specialization,
                require_full_subgroups,
            )
            .or_else(|e| {
                if selected_kernel == "matmul_vec_q8_0_idp" {
                    let failed = selected_kernel.clone();
                    let fallback = "matmul_vec_q8_0_coop";
                    selected_kernel = fallback.to_string();
                    is_coop_kernel = true;
                    require_full_subgroups = true;
                    self.device
                        .kernels()
                        .load_pipeline(
                            self.device.device(),
                            fallback,
                            None,
                            32,
                            3,
                            specialization,
                            require_full_subgroups,
                        )
                        .map_err(|e2| {
                            crate::Error::Msg(format!(
                                "Failed to load '{}' (err: {}) and fallback '{}' (err: {})",
                                failed, e, fallback, e2
                            ))
                        })
                } else if selected_kernel.ends_with("_coopmat") {
                    let failed = selected_kernel.clone();
                    let fallback = if failed == "matmul_q8_0_coopmat" {
                        if std::env::var("PARAMECIA_DISABLE_DP4A").is_ok() {
                            "matmul_q8_0"
                        } else {
                            "matmul_q8_0_dp4a"
                        }
                    } else {
                        failed.trim_end_matches("_coopmat")
                    };
                    selected_kernel = fallback.to_string();
                    is_coop_kernel = false;
                    require_full_subgroups = false;
                    self.device
                        .kernels()
                        .load_pipeline(self.device.device(), fallback, None, 32, 3, None, false)
                        .map_err(|e2| {
                            crate::Error::Msg(format!(
                                "Failed to load '{}' (err: {}) and fallback '{}' (err: {})",
                                failed, e, fallback, e2
                            ))
                        })
                } else {
                    Err(crate::Error::Vulkan(VulkanError::from(e)))
                }
            })?;

        let dispatch = if is_coop_kernel {
            // Cooperative matvec: NUM_ROWS outputs per workgroup
            [
                (n as u32 + coop_num_rows - 1) / coop_num_rows,
                batch as u32,
                1,
            ]
        } else if is_matvec && selected_kernel.starts_with("matmul_vec_") {
            // Non-cooperative mat-vec kernels
            let outputs_per_workgroup = if selected_kernel == "matmul_vec_q8_0" {
                128
            } else {
                4 // Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (all use 4 warps, 1 output per warp)
            };
            [
                (n as u32 + outputs_per_workgroup - 1) / outputs_per_workgroup,
                batch as u32,
                1,
            ]
        } else if selected_kernel.starts_with("matmul_tiled_") {
            // Hierarchical tiled kernels use BM×BN tiles
            // Prefill variants all use BM=64 (including _prefill_opt/_vec/_db/_coopmat).
            let is_prefill_kernel = selected_kernel.contains("_prefill");
            let (tile_m, tile_n) = if is_prefill_kernel {
                // Prefill-optimized: BM=64, BN=64 (workgroup: 128 threads = 4 warps)
                (64u32, 64u32)
            } else {
                // Decode-optimized: BM=16, BN=64 (workgroup: 64 threads = 2 warps)
                (16u32, 64u32)
            };
            // Dispatch: [ceil(n/BN), ceil(m/BM), batch]
            [
                (n as u32 + tile_n - 1) / tile_n, // N dimension (workgroup.x)
                (m as u32 + tile_m - 1) / tile_m, // M dimension (workgroup.y)
                batch as u32,                     // Batch dimension (workgroup.z)
            ]
        } else if selected_kernel.ends_with("_coopmat") {
            let (tile_m, tile_n, _) = self.device.coop_matrix_tile_size().unwrap_or((16, 16, 16));
            // Coopmat kernels use runtime tile sizes: dispatch [ceil(m/tile_m), ceil(n/tile_n), batch]
            [
                (m as u32 + tile_m - 1) / tile_m, // M dimension (workgroup.x)
                (n as u32 + tile_n - 1) / tile_n, // N dimension (workgroup.y)
                batch as u32,                     // Batch dimension (workgroup.z)
            ]
        } else {
            // Standard kernels: DP4A processes 16 cols/workgroup, others 4
            let cols_per_wg: u32 = if selected_kernel.starts_with("matmul_q8_0_dp4a") {
                16
            } else {
                4
            };
            [
                (n as u32 + cols_per_wg - 1) / cols_per_wg,
                m as u32,
                batch as u32,
            ]
        };

        debug!(
            "[DEBUG] Dispatch: [{}, {}, {}], params: m={}, k={}, n={}, strides=[{}, {}, {}], offset={}",
            dispatch[0],
            dispatch[1],
            dispatch[2],
            params[0],
            params[1],
            params[2],
            params[3],
            params[5],
            params[6],
            params[4]
        );

        // Record compute with push constants (no temp buffer, no flush needed)
        let out_storage = VulkanStorage::new(
            Some(Arc::new(out_buf)),
            self.device.clone(),
            out_elems,
            DType::F32,
        );
        self.device.record_compute_with_write_mask(
            &pipeline,
            &[out_storage.vk_buffer()?, act_buf, weight_buf],
            Some(0b001),
            Some(bytemuck::cast_slice(&params)),
            dispatch,
        )?;

        let result_shape = if x_dims.len() == 2 {
            crate::Shape::from_dims(&[m, n])
        } else {
            crate::Shape::from_dims(&[batch, m, n])
        };

        // Cast output back to input dtype if needed (shader always produces f32)
        let input_dtype = storage.dtype();
        if input_dtype != DType::F32 {
            let out_layout = crate::Layout::contiguous(&result_shape);
            let cast_out = out_storage.to_dtype(&out_layout, input_dtype)?;
            Ok((cast_out, result_shape))
        } else {
            Ok((out_storage, result_shape))
        }
    }

    /// CPU fallback for quantized matmul when no GPU shader is available.
    fn fwd_cpu_fallback(
        &self,
        self_shape: &crate::Shape,
        storage: &VulkanStorage,
        layout: &crate::Layout,
    ) -> Result<(VulkanStorage, crate::Shape)> {
        use crate::backend::BackendStorage;

        {
            use std::sync::atomic::{AtomicU64, Ordering};
            static QMATMUL_FALLBACK: AtomicU64 = AtomicU64::new(0);
            let n = QMATMUL_FALLBACK.fetch_add(1, Ordering::Relaxed);
            if n < 50 {
                warn!(
                    "[VK FALLBACK] QMatMul fwd_cpu dtype={:?} shape={:?} x_shape={:?} has_buf={}",
                    self.dtype,
                    self_shape.dims(),
                    layout.shape().dims(),
                    self.buffer.is_some()
                );
            }
        }

        let cpu_x = storage.to_cpu_storage()?;

        let nrows = self_shape.dim(0)?;
        let ncols = self_shape.dim(1)?;
        let elem_count = nrows * ncols;
        let cpu_weights = self.dequantize_to_cpu(elem_count)?;

        let weight_shape = crate::Shape::from_dims(&[nrows, ncols]);
        let weight_layout = crate::Layout::contiguous(&weight_shape);

        let x_dims = layout.shape().dims();
        let (b, m, k) = if x_dims.len() == 2 {
            (1, x_dims[0], x_dims[1])
        } else if x_dims.len() == 3 {
            (x_dims[0], x_dims[1], x_dims[2])
        } else {
            crate::bail!(
                "Expected 2D or 3D input for quantized fwd, got {:?}",
                x_dims
            );
        };
        let n = nrows;
        if k != ncols {
            crate::bail!("Quantized fwd dim mismatch: input k={k} vs weight k={ncols}");
        }

        let cpu_result = cpu_weights.matmul(&cpu_x, (b, m, n, k), &weight_layout, layout)?;
        let result_shape = if x_dims.len() == 2 {
            crate::Shape::from_dims(&[m, n])
        } else {
            crate::Shape::from_dims(&[b, m, n])
        };

        let vk_result = self.device.storage_from_cpu_storage(&cpu_result)?;
        Ok((vk_result, result_shape))
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        self.get_cpu_data()
    }

    /// GPU-accelerated perturbation of quantized weights.
    ///
    /// Dispatches a Vulkan compute shader that:
    /// 1. Dequantizes each element
    /// 2. Adds or subtracts epsilon * perturbation[i]
    /// 3. Re-quantizes with stochastic rounding
    ///
    /// Modifies the GPU buffer in-place. Clears cpu_data (must be re-downloaded if needed).
    pub fn perturb_weights_gpu(
        &mut self,
        perturbation: &VulkanStorage,
        epsilon: f32,
        seed: u64,
        add: bool,
    ) -> Result<()> {
        let perturb_buf = perturbation.vk_buffer()?;

        // Select kernel name and dispatch parameters based on dtype
        let (kernel_name, num_buffers, push_constant_size) = match self.dtype {
            GgmlDType::Q8_0 => ("perturb_q8_0", 2u32, 32u32),
            GgmlDType::Q4K => ("perturb_q4_k", 2u32, 32u32),
            GgmlDType::Q2K => ("perturb_q2_k", 2u32, 32u32),
            GgmlDType::Q3K => ("perturb_q3_k", 2u32, 32u32),
            GgmlDType::Q5K => ("perturb_q5_k", 2u32, 32u32),
            GgmlDType::Q6K => ("perturb_q6_k", 2u32, 32u32),
            GgmlDType::BF16 => ("perturb_bf16", 2u32, 32u32),
            _ => crate::bail!("perturb_weights_gpu not supported for {:?}", self.dtype),
        };

        // Compute count and dispatch dimensions
        let (count, dispatch_x) = self.compute_perturb_dispatch();

        // Make a copy of the GPU buffer (perturb kernels work in-place)
        let old_buf = self
            .buffer
            .as_ref()
            .ok_or_else(|| crate::Error::Msg("perturb_weights_gpu: no GPU buffer".into()))?
            .clone();
        let new_buf = self.device.allocate_buffer(self.byte_count as u64)?;
        self.device
            .record_copy(old_buf.buffer, new_buf.buffer, self.byte_count as u64)?;
        // Keep old buffer alive until the batch is submitted (record_copy references it)
        self.device.keep_buffer_alive(old_buf)?;

        // Build push constants: count, param1(epsilon), param2(unused), seed1_lo, seed1_hi, seed2_lo, seed2_hi, flags
        let flags: u32 = if add { 1 } else { 0 };
        let pc = PerturbPushConstants {
            count,
            param1: epsilon,
            param2: 0.0,
            seed1_lo: seed as u32,
            seed1_hi: (seed >> 32) as u32,
            seed2_lo: 0,
            seed2_hi: 0,
            flags,
        };

        let pipeline = self
            .device
            .kernels()
            .load_pipeline(
                self.device.device(),
                kernel_name,
                None,
                push_constant_size,
                num_buffers,
                None,
                false,
            )
            .map_err(|e| crate::Error::Vulkan(VulkanError::from(e)))?;

        self.device.record_compute_with_write_mask(
            &pipeline,
            &[new_buf.buffer, perturb_buf],
            Some(0b01),
            Some(bytemuck::bytes_of(&pc)),
            [dispatch_x, 1, 1],
        )?;

        // Update buffer, invalidate cpu_data
        self.buffer = Some(Arc::new(new_buf));
        self.cpu_data.clear();

        Ok(())
    }

    /// GPU-accelerated combined restore and update operation.
    ///
    /// Fuses restore (from -epsilon to 0) and gradient update into a single kernel pass.
    /// val = dequant(w) + restore_epsilon * perturbation - update_scale * perturbation
    /// then re-quantize with stochastic rounding using update_seed.
    pub fn restore_and_update_gpu(
        &mut self,
        perturbation: &VulkanStorage,
        restore_epsilon: f32,
        update_scale: f32,
        restore_seed: u64,
        update_seed: u64,
    ) -> Result<()> {
        let perturb_buf = perturbation.vk_buffer()?;

        let (kernel_name, num_buffers, push_constant_size) = match self.dtype {
            GgmlDType::Q8_0 => ("restore_update_q8_0", 2u32, 32u32),
            GgmlDType::Q4K => ("restore_update_q4_k", 2u32, 32u32),
            GgmlDType::Q2K => ("restore_update_q2_k", 2u32, 32u32),
            GgmlDType::Q3K => ("restore_update_q3_k", 2u32, 32u32),
            GgmlDType::Q5K => ("restore_update_q5_k", 2u32, 32u32),
            GgmlDType::Q6K => ("restore_update_q6_k", 2u32, 32u32),
            GgmlDType::BF16 => ("restore_update_bf16", 2u32, 32u32),
            _ => crate::bail!("restore_and_update_gpu not supported for {:?}", self.dtype),
        };

        let (count, dispatch_x) = self.compute_perturb_dispatch();

        // Copy buffer
        let old_buf = self
            .buffer
            .as_ref()
            .ok_or_else(|| crate::Error::Msg("restore_and_update_gpu: no GPU buffer".into()))?
            .clone();
        let new_buf = self.device.allocate_buffer(self.byte_count as u64)?;
        self.device
            .record_copy(old_buf.buffer, new_buf.buffer, self.byte_count as u64)?;
        // Keep old buffer alive until the batch is submitted (record_copy references it)
        self.device.keep_buffer_alive(old_buf)?;

        // Push constants: count, param1(restore_epsilon), param2(update_scale),
        // seed1_lo/hi (restore_seed), seed2_lo/hi (update_seed), flags(unused)
        let pc = PerturbPushConstants {
            count,
            param1: restore_epsilon,
            param2: update_scale,
            seed1_lo: restore_seed as u32,
            seed1_hi: (restore_seed >> 32) as u32,
            seed2_lo: update_seed as u32,
            seed2_hi: (update_seed >> 32) as u32,
            flags: 0,
        };

        let pipeline = self
            .device
            .kernels()
            .load_pipeline(
                self.device.device(),
                kernel_name,
                None,
                push_constant_size,
                num_buffers,
                None,
                false,
            )
            .map_err(|e| crate::Error::Vulkan(VulkanError::from(e)))?;

        self.device.record_compute_with_write_mask(
            &pipeline,
            &[new_buf.buffer, perturb_buf],
            Some(0b01),
            Some(bytemuck::bytes_of(&pc)),
            [dispatch_x, 1, 1],
        )?;

        self.buffer = Some(Arc::new(new_buf));
        self.cpu_data.clear();

        Ok(())
    }

    /// Compute the count (number of blocks or elements) and dispatch X dimension
    /// for perturbation kernels based on dtype.
    fn compute_perturb_dispatch(&self) -> (u32, u32) {
        match self.dtype {
            GgmlDType::BF16 => {
                // BF16: count = num_elements (each element is 2 bytes)
                // Each thread handles 2 elements (1 u32 word), local_size_x = 128
                let num_elements = (self.byte_count / 2) as u32;
                let num_words = (num_elements + 1) / 2;
                (num_words, (num_words + 127) / 128)
            }
            GgmlDType::Q8_0 => {
                // Q8_0: 34 bytes/block, 2 blocks per workgroup, local_size_x = 64
                let num_blocks = (self.byte_count / 34) as u32;
                (num_blocks, (num_blocks + 1) / 2)
            }
            GgmlDType::Q4K => {
                // Q4K: 144 bytes/block, 1 block per workgroup, local_size_x = 128
                let num_blocks = (self.byte_count / 144) as u32;
                (num_blocks, num_blocks)
            }
            GgmlDType::Q2K => {
                // Q2K: 84 bytes/block, 1 block per workgroup, local_size_x = 64
                let num_blocks = (self.byte_count / 84) as u32;
                (num_blocks, num_blocks)
            }
            GgmlDType::Q3K => {
                // Q3K: 110 bytes/block, 2 blocks per workgroup, local_size_x = 128
                let num_blocks = (self.byte_count / 110) as u32;
                (num_blocks, (num_blocks + 1) / 2)
            }
            GgmlDType::Q5K => {
                // Q5K: 176 bytes/block, 1 block per workgroup, local_size_x = 64
                let num_blocks = (self.byte_count / 176) as u32;
                (num_blocks, num_blocks)
            }
            GgmlDType::Q6K => {
                // Q6K: 210 bytes/block, 2 blocks per workgroup, local_size_x = 128
                let num_blocks = (self.byte_count / 210) as u32;
                (num_blocks, (num_blocks + 1) / 2)
            }
            _ => (0, 0),
        }
    }

    /// Select MoE kernel name and block parameters for this quantization format.
    /// Returns (kernel_name, block_size, bytes_per_block, cols_per_wg) or None if unsupported.
    fn moe_kernel_params(&self) -> Option<(&'static str, usize, usize, u32)> {
        let has_coopmat = self.device.has_cooperative_matrix();

        match self.dtype {
            GgmlDType::Q8_0 => Some((
                if has_coopmat && !self.is_coopmat_disabled("MOE_FORWARD") {
                    "indexed_moe_forward_q8_0_coopmat"
                } else {
                    "indexed_moe_forward_q8_0"
                },
                32,
                34,
                16,
            )),
            GgmlDType::Q4K => Some((
                if has_coopmat && !self.is_coopmat_disabled("MOE_FORWARD") {
                    "indexed_moe_forward_q4_k_coopmat"
                } else {
                    "indexed_moe_forward_q4_k"
                },
                256,
                144,
                4,
            )),
            GgmlDType::Q5K => Some((
                if has_coopmat && !self.is_coopmat_disabled("MOE_FORWARD") {
                    "indexed_moe_forward_q5_k_coopmat"
                } else {
                    "indexed_moe_forward_q5_k"
                },
                256,
                176,
                4,
            )),
            GgmlDType::Q6K => Some((
                if has_coopmat && !self.is_coopmat_disabled("MOE_FORWARD") {
                    "indexed_moe_forward_q6_k_coopmat"
                } else {
                    "indexed_moe_forward_q6_k"
                },
                256,
                210,
                4,
            )),
            GgmlDType::Q3K => Some((
                if has_coopmat && !self.is_coopmat_disabled("MOE_FORWARD") {
                    "indexed_moe_forward_q3_k_coopmat"
                } else {
                    "indexed_moe_forward_q3_k"
                },
                256,
                110,
                4,
            )),
            GgmlDType::Q2K => Some((
                if has_coopmat && !self.is_coopmat_disabled("MOE_FORWARD") {
                    "indexed_moe_forward_q2_k_coopmat"
                } else {
                    "indexed_moe_forward_q2_k"
                },
                256,
                84,
                4,
            )),
            _ => None,
        }
    }

    /// Select fused MoE gate+up+swiglu kernel parameters.
    /// Returns (kernel_name, block_size, bytes_per_block, cols_per_wg) or None if unsupported.
    fn fused_moe_kernel_params(&self) -> Option<(&'static str, usize, usize, u32)> {
        let has_coopmat = self.device.has_cooperative_matrix();

        match self.dtype {
            GgmlDType::Q8_0 => Some((
                if has_coopmat && !self.is_coopmat_disabled("MOE_GATE_UP") {
                    "indexed_moe_gate_up_swiglu_q8_0_coopmat"
                } else {
                    "indexed_moe_gate_up_swiglu_q8_0"
                },
                32,
                34,
                16,
            )),
            GgmlDType::Q4K => Some((
                if has_coopmat && !self.is_coopmat_disabled("MOE_GATE_UP") {
                    "indexed_moe_gate_up_swiglu_q4_k_coopmat"
                } else {
                    "indexed_moe_gate_up_swiglu_q4_k"
                },
                256,
                144,
                4,
            )),
            GgmlDType::Q5K => Some((
                if has_coopmat && !self.is_coopmat_disabled("MOE_GATE_UP") {
                    "indexed_moe_gate_up_swiglu_q5_k_coopmat"
                } else {
                    "indexed_moe_gate_up_swiglu_q5_k"
                },
                256,
                176,
                4,
            )),
            GgmlDType::Q6K => Some((
                if has_coopmat && !self.is_coopmat_disabled("MOE_GATE_UP") {
                    "indexed_moe_gate_up_swiglu_q6_k_coopmat"
                } else {
                    "indexed_moe_gate_up_swiglu_q6_k"
                },
                256,
                210,
                4,
            )),
            GgmlDType::Q3K => Some((
                if has_coopmat && !self.is_coopmat_disabled("MOE_GATE_UP") {
                    "indexed_moe_gate_up_swiglu_q3_k_coopmat"
                } else {
                    "indexed_moe_gate_up_swiglu_q3_k"
                },
                256,
                110,
                4,
            )),
            GgmlDType::Q2K => Some((
                if has_coopmat && !self.is_coopmat_disabled("MOE_GATE_UP") {
                    "indexed_moe_gate_up_swiglu_q2_k_coopmat"
                } else {
                    "indexed_moe_gate_up_swiglu_q2_k"
                },
                256,
                84,
                4,
            )),
            _ => None,
        }
    }

    pub fn indexed_moe_forward(
        &self,
        self_shape: &crate::Shape,
        x_storage: &VulkanStorage,
        x_layout: &crate::Layout,
        ids_storage: &VulkanStorage,
        ids_layout: &crate::Layout,
    ) -> Result<(VulkanStorage, crate::Shape)> {
        let (kernel_name, block_size, bytes_per_block, cols_per_wg) = match self.moe_kernel_params()
        {
            Some(p) => p,
            None => crate::bail!(
                "Vulkan indexed_moe_forward: unsupported dtype {:?}",
                self.dtype
            ),
        };

        let weight_buf = match &self.buffer {
            Some(b) => b.buffer,
            None => crate::bail!("Vulkan indexed_moe_forward: no GPU buffer for weights"),
        };

        // Weight shape: [num_experts, n, k]
        let weight_dims = self_shape.dims();
        if weight_dims.len() != 3 {
            crate::bail!(
                "indexed_moe_forward expects 3D weights [num_experts, out_dim, in_dim], got {:?}",
                weight_dims
            );
        }
        let n = weight_dims[1]; // out_dim
        let k = weight_dims[2]; // in_dim

        // Parse input X: [batch, topk_or_1, k] or [batch, k]
        let x_dims = x_layout.shape().dims();
        let (batch, x_inner_dim, in_dim_x) = match x_dims.len() {
            2 => (x_dims[0], 1usize, x_dims[1]),
            3 => (x_dims[0], x_dims[1], x_dims[2]),
            _ => crate::bail!(
                "indexed_moe_forward expects 2D or 3D input, got {:?}",
                x_dims
            ),
        };
        if in_dim_x != k {
            crate::bail!("indexed_moe_forward dim mismatch: weight k={k} vs input k={in_dim_x}");
        }

        // Parse IDs: [batch, topk]
        let ids_dims = ids_layout.shape().dims();
        if ids_dims.len() != 2 {
            crate::bail!(
                "indexed_moe_forward expects 2D ids [batch, topk], got {:?}",
                ids_dims
            );
        }
        let topk = ids_dims[1];

        // Ensure activations are f32
        use crate::backend::BackendStorage;
        let (f32_storage, f32_layout) = if x_storage.dtype() != DType::F32 {
            let cast = x_storage.to_dtype(x_layout, DType::F32)?;
            let cast_layout = crate::Layout::contiguous(x_layout.shape());
            (Some(cast), cast_layout)
        } else {
            (None, x_layout.clone())
        };
        let act_storage = f32_storage.as_ref().unwrap_or(x_storage);
        let act_buf = act_storage.vk_buffer()?;

        // IDs buffer
        let ids_buf = ids_storage.vk_buffer()?;

        // Expert stride in u32 words: each expert has n * num_blocks * bytes_per_block bytes
        let num_blocks = k / block_size;
        let expert_bytes = n * num_blocks * bytes_per_block;
        let expert_stride_words = (expert_bytes + 3) / 4; // ceil to u32

        // X strides (in f32 elements)
        let x_strides = f32_layout.stride();
        let (x_stride_batch, x_stride_inner) = match x_strides.len() {
            2 => (x_strides[0] as u32, k as u32),
            3 => (x_strides[0] as u32, x_strides[1] as u32),
            _ => (0u32, k as u32),
        };

        // Push constants: 9 x u32 = 36 bytes
        let params: [u32; 9] = [
            n as u32,                   // [0] out_dim
            k as u32,                   // [1] in_dim
            batch as u32,               // [2] batch
            topk as u32,                // [3] topk
            x_inner_dim as u32,         // [4] x_inner_dim
            expert_stride_words as u32, // [5] expert_stride (in u32 words)
            x_stride_batch,             // [6] x_stride_batch
            x_stride_inner,             // [7] x_stride_inner
            0,                          // [8] unused
        ];

        // Allocate output: [batch, topk, n]
        let out_elems = batch * topk * n;
        let out_buf = self.device.allocate_buffer((out_elems * 4) as u64)?;

        // Get specialization constants if using cooperative matrix
        let specialization = if kernel_name.ends_with("_coopmat") {
            let (tile_m, tile_n, tile_k) =
                self.device.coop_matrix_tile_size().unwrap_or((16, 16, 16));
            Some(vec![tile_m, tile_n, tile_k])
        } else {
            None
        };

        // Load pipeline: 4 SSBOs (output, inputX, weights, ids), 36 bytes push constants
        // If cooperative matrix shader fails, fall back to standard shader
        let mut selected_kernel = kernel_name;
        let pipeline = match self.device.kernels().load_pipeline(
            self.device.device(),
            selected_kernel,
            None,
            36,
            4,
            specialization.as_deref(),
            selected_kernel.ends_with("_coopmat"),
        ) {
            Ok(p) => p,
            Err(e) if selected_kernel.ends_with("_coopmat") => {
                let fallback_name = selected_kernel.trim_end_matches("_coopmat");
                warn!(
                    "[VK] Cooperative matrix MoE shader '{}' failed, falling back to '{}'. Error: {:?}",
                    selected_kernel, fallback_name, e
                );
                selected_kernel = fallback_name;
                self.device
                    .kernels()
                    .load_pipeline(
                        self.device.device(),
                        selected_kernel,
                        None,
                        36,
                        4,
                        None,
                        false,
                    )
                    .map_err(|e2| crate::Error::Vulkan(VulkanError::from(e2)))?
            }
            Err(e) => return Err(crate::Error::Vulkan(VulkanError::from(e))),
        };

        // Dispatch depends on kernel type
        let dispatch = if selected_kernel.ends_with("_coopmat") {
            let (_, tile_n, _) = self.device.coop_matrix_tile_size().unwrap_or((16, 16, 16));
            // indexed_moe_forward processes single input vectors (M=1), so x dimension is always 1.
            [1, (n as u32 + tile_n - 1) / tile_n, (batch * topk) as u32]
        } else {
            // Standard kernels: (ceil(n/cols_per_wg), batch, topk)
            [
                (n as u32 + cols_per_wg - 1) / cols_per_wg,
                batch as u32,
                topk as u32,
            ]
        };

        let out_storage = VulkanStorage::new(
            Some(Arc::new(out_buf)),
            self.device.clone(),
            out_elems,
            DType::F32,
        );
        self.device.record_compute_with_write_mask(
            &pipeline,
            &[out_storage.vk_buffer()?, act_buf, weight_buf, ids_buf],
            Some(0b0001),
            Some(bytemuck::cast_slice(&params)),
            dispatch,
        )?;

        let result_shape = crate::Shape::from_dims(&[batch, topk, n]);

        // Cast output back to input dtype if needed
        let input_dtype = x_storage.dtype();
        if input_dtype != DType::F32 {
            let out_layout = crate::Layout::contiguous(&result_shape);
            let cast_out = out_storage.to_dtype(&out_layout, input_dtype)?;
            Ok((cast_out, result_shape))
        } else {
            Ok((out_storage, result_shape))
        }
    }

    /// Fused gate+up+SwiGLU MoE kernel.
    /// Computes: output = silu(gate_weights[expert] @ x) * (up_weights[expert] @ x)
    /// in a single GPU dispatch, eliminating intermediate buffers and barriers.
    pub fn indexed_moe_gate_up_swiglu(
        &self, // gate weights
        gate_shape: &crate::Shape,
        up_storage: &QVulkanStorage,
        up_shape: &crate::Shape,
        x_storage: &VulkanStorage,
        x_layout: &crate::Layout,
        ids_storage: &VulkanStorage,
        ids_layout: &crate::Layout,
    ) -> Result<(VulkanStorage, crate::Shape)> {
        // Both gate and up must have the same dtype for the fused kernel
        if self.dtype != up_storage.dtype {
            crate::bail!(
                "indexed_moe_gate_up_swiglu: gate dtype {:?} != up dtype {:?}",
                self.dtype,
                up_storage.dtype
            );
        }

        let (kernel_name, block_size, bytes_per_block, cols_per_wg) =
            match self.fused_moe_kernel_params() {
                Some(p) => p,
                None => crate::bail!(
                    "Vulkan indexed_moe_gate_up_swiglu: unsupported dtype {:?}",
                    self.dtype
                ),
            };

        let gate_buf = match &self.buffer {
            Some(b) => b.buffer,
            None => crate::bail!("indexed_moe_gate_up_swiglu: no GPU buffer for gate weights"),
        };
        let up_buf = match &up_storage.buffer {
            Some(b) => b.buffer,
            None => crate::bail!("indexed_moe_gate_up_swiglu: no GPU buffer for up weights"),
        };

        // Gate shape: [num_experts, n_ff, k]
        let gate_dims = gate_shape.dims();
        if gate_dims.len() != 3 {
            crate::bail!(
                "indexed_moe_gate_up_swiglu: gate weights must be 3D, got {:?}",
                gate_dims
            );
        }
        let n_ff = gate_dims[1];
        let k = gate_dims[2];

        // Up shape must match
        let up_dims = up_shape.dims();
        if up_dims.len() != 3 || up_dims[1] != n_ff || up_dims[2] != k {
            crate::bail!(
                "indexed_moe_gate_up_swiglu: shape mismatch gate={:?} up={:?}",
                gate_dims,
                up_dims
            );
        }

        // Parse input X: [batch, topk_or_1, k] or [batch, k]
        let x_dims = x_layout.shape().dims();
        let (batch, x_inner_dim, in_dim_x) = match x_dims.len() {
            2 => (x_dims[0], 1usize, x_dims[1]),
            3 => (x_dims[0], x_dims[1], x_dims[2]),
            _ => crate::bail!(
                "indexed_moe_gate_up_swiglu: input must be 2D or 3D, got {:?}",
                x_dims
            ),
        };
        if in_dim_x != k {
            crate::bail!(
                "indexed_moe_gate_up_swiglu: dim mismatch weight k={k} vs input k={in_dim_x}"
            );
        }

        // Parse IDs: [batch, topk]
        let ids_dims = ids_layout.shape().dims();
        if ids_dims.len() != 2 {
            crate::bail!(
                "indexed_moe_gate_up_swiglu: ids must be 2D [batch, topk], got {:?}",
                ids_dims
            );
        }
        let topk = ids_dims[1];

        // Ensure activations are f32
        use crate::backend::BackendStorage;
        let (f32_storage, f32_layout) = if x_storage.dtype() != DType::F32 {
            let cast = x_storage.to_dtype(x_layout, DType::F32)?;
            let cast_layout = crate::Layout::contiguous(x_layout.shape());
            (Some(cast), cast_layout)
        } else {
            (None, x_layout.clone())
        };
        let act_storage = f32_storage.as_ref().unwrap_or(x_storage);
        let act_buf = act_storage.vk_buffer()?;
        let ids_buf = ids_storage.vk_buffer()?;

        // Expert strides (in u32 words)
        let num_blocks = k / block_size;
        let gate_expert_bytes = n_ff * num_blocks * bytes_per_block;
        let gate_expert_stride = (gate_expert_bytes + 3) / 4;
        let up_expert_bytes = n_ff * num_blocks * bytes_per_block;
        let up_expert_stride = (up_expert_bytes + 3) / 4;

        // X strides
        let x_strides = f32_layout.stride();
        let (x_stride_batch, x_stride_inner) = match x_strides.len() {
            2 => (x_strides[0] as u32, k as u32),
            3 => (x_strides[0] as u32, x_strides[1] as u32),
            _ => (0u32, k as u32),
        };

        // Push constants: 12 x u32 = 48 bytes
        let params: [u32; 12] = [
            n_ff as u32,               // [0] n_ff (out_dim)
            k as u32,                  // [1] k (in_dim)
            batch as u32,              // [2] batch
            topk as u32,               // [3] topk
            x_inner_dim as u32,        // [4] x_inner_dim
            gate_expert_stride as u32, // [5] gate_expert_stride
            up_expert_stride as u32,   // [6] up_expert_stride
            x_stride_batch,            // [7] x_stride_batch
            x_stride_inner,            // [8] x_stride_inner
            0,
            0,
            0, // [9-11] unused
        ];

        // Allocate output: [batch, topk, n_ff]
        let out_elems = batch * topk * n_ff;
        let out_buf = self.device.allocate_buffer((out_elems * 4) as u64)?;

        // Get specialization constants if using cooperative matrix
        let specialization = if kernel_name.ends_with("_coopmat") {
            let (tile_m, tile_n, tile_k) =
                self.device.coop_matrix_tile_size().unwrap_or((16, 16, 16));
            Some(vec![tile_m, tile_n, tile_k])
        } else {
            None
        };

        // Load pipeline: 5 SSBOs (output, inputX, gate_weights, up_weights, ids), 48 bytes PC
        // If cooperative matrix shader fails, fall back to standard shader
        let mut selected_kernel = kernel_name;
        let pipeline = match self.device.kernels().load_pipeline(
            self.device.device(),
            selected_kernel,
            None,
            48,
            5,
            specialization.as_deref(),
            selected_kernel.ends_with("_coopmat"),
        ) {
            Ok(p) => p,
            Err(e) if selected_kernel.ends_with("_coopmat") => {
                let fallback_name = selected_kernel.trim_end_matches("_coopmat");
                warn!(
                    "[VK] Cooperative matrix MoE gate+up shader '{}' failed, falling back to '{}'. Error: {:?}",
                    selected_kernel, fallback_name, e
                );
                selected_kernel = fallback_name;
                self.device
                    .kernels()
                    .load_pipeline(
                        self.device.device(),
                        selected_kernel,
                        None,
                        48,
                        5,
                        None,
                        false,
                    )
                    .map_err(|e2| crate::Error::Vulkan(VulkanError::from(e2)))?
            }
            Err(e) => return Err(crate::Error::Vulkan(VulkanError::from(e))),
        };

        // Dispatch depends on kernel type
        let dispatch = if selected_kernel.ends_with("_coopmat") {
            let (_, tile_n, _) = self.device.coop_matrix_tile_size().unwrap_or((16, 16, 16));
            // indexed_moe_gate_up_swiglu processes single vectors with coopmat tile width on N.
            [
                (n_ff as u32 + tile_n - 1) / tile_n,
                batch as u32,
                topk as u32,
            ]
        } else {
            // Standard kernels
            [
                (n_ff as u32 + cols_per_wg - 1) / cols_per_wg,
                batch as u32,
                topk as u32,
            ]
        };

        let out_storage = VulkanStorage::new(
            Some(Arc::new(out_buf)),
            self.device.clone(),
            out_elems,
            DType::F32,
        );
        self.device.record_compute_with_write_mask(
            &pipeline,
            &[out_storage.vk_buffer()?, act_buf, gate_buf, up_buf, ids_buf],
            Some(0b00001),
            Some(bytemuck::cast_slice(&params)),
            dispatch,
        )?;

        let result_shape = crate::Shape::from_dims(&[batch, topk, n_ff]);

        let input_dtype = x_storage.dtype();
        if input_dtype != DType::F32 {
            let out_layout = crate::Layout::contiguous(&result_shape);
            let cast_out = out_storage.to_dtype(&out_layout, input_dtype)?;
            Ok((cast_out, result_shape))
        } else {
            Ok((out_storage, result_shape))
        }
    }

    pub fn extract_scales(&self, elem_count: usize) -> Result<VulkanStorage> {
        use super::k_quants::*;
        let data = self.get_cpu_data()?;

        macro_rules! extract {
            ($ty:ty) => {{
                let blocks: &[$ty] = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const $ty,
                        data.len() / std::mem::size_of::<$ty>(),
                    )
                };
                <$ty as GgmlType>::extract_scales(blocks)
            }};
        }

        let scales = match self.dtype {
            GgmlDType::Q4_0 => extract!(BlockQ4_0),
            GgmlDType::Q4_1 => extract!(BlockQ4_1),
            GgmlDType::Q5_0 => extract!(BlockQ5_0),
            GgmlDType::Q5_1 => extract!(BlockQ5_1),
            GgmlDType::Q8_0 => extract!(BlockQ8_0),
            GgmlDType::Q8_1 => extract!(BlockQ8_1),
            GgmlDType::Q2K => extract!(BlockQ2K),
            GgmlDType::Q3K => extract!(BlockQ3K),
            GgmlDType::Q4K => extract!(BlockQ4K),
            GgmlDType::Q5K => extract!(BlockQ5K),
            GgmlDType::Q6K => extract!(BlockQ6K),
            GgmlDType::Q8K => extract!(BlockQ8K),
            other => crate::bail!("Vulkan extract_scales: unsupported dtype {:?}", other),
        };

        let cpu_storage = crate::CpuStorage::F32(scales);
        self.device.storage_from_cpu_storage(&cpu_storage)
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &VulkanDevice,
    data: &[T],
) -> Result<QStorage> {
    let dtype = T::DTYPE;
    let raw_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };

    let buffer = if raw_bytes.is_empty() {
        None
    } else {
        let buf = device.allocate_buffer(raw_bytes.len() as u64)?;
        device.upload_to_buffer(raw_bytes, &buf)?;
        Some(Arc::new(buf))
    };

    let storage = QVulkanStorage {
        dtype,
        device: device.clone(),
        buffer,
        byte_count: raw_bytes.len(),
        cpu_data: Vec::new(),
    };

    Ok(QStorage::Vulkan(storage))
}
