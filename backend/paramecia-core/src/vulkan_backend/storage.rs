#![allow(dead_code)]

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::scalar::Scalar;
use crate::{CpuStorage, DType, Layout, Result, Shape, VulkanDevice, VulkanError};
use ash::vk;
use half::{bf16, f16};
use paramecia_vulkan::CachedPipeline;
use std::sync::Arc;

use super::device::VulkanBuffer;
use tracing::{trace, warn};

const MAX_RANK: usize = 8;

#[derive(Clone, Debug)]
pub struct VulkanStorage {
    buffer: Option<Arc<VulkanBuffer>>,
    device: VulkanDevice,
    count: usize,
    dtype: DType,
}

impl VulkanStorage {
    pub(crate) fn new(
        buffer: Option<Arc<VulkanBuffer>>,
        device: VulkanDevice,
        count: usize,
        dtype: DType,
    ) -> Self {
        if let Some(ref b) = buffer {
            device.register_buffer(b);
        }
        Self {
            buffer,
            device,
            count,
            dtype,
        }
    }

    pub fn buffer(&self) -> Option<&VulkanBuffer> {
        self.buffer.as_deref()
    }

    pub fn vk_buffer(&self) -> Result<vk::Buffer> {
        self.buffer
            .as_ref()
            .map(|b| b.buffer)
            .ok_or_else(|| crate::Error::Msg("Vulkan storage has no buffer (zero-sized)".into()))
    }

    pub fn vk_buffer_arc(&self) -> Result<Arc<VulkanBuffer>> {
        self.buffer
            .as_ref()
            .cloned()
            .ok_or_else(|| crate::Error::Msg("Vulkan storage has no buffer (zero-sized)".into()))
    }

    fn dtype_suffix(&self) -> Result<&'static str> {
        Ok(match self.dtype {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::U32 => "u32",
            DType::U8 => "u8",
            DType::I64 => "i64",
            _ => crate::bail!("Vulkan does not support {:?}", self.dtype),
        })
    }

    /// Check if cooperative matrix is disabled for GEMM via environment variables.
    /// Checks in order:
    /// 1. PARAMECIA_DISABLE_COOPMAT_GEMM (all dtypes)
    /// 2. PARAMECIA_DISABLE_COOPMAT_<DTYPE>_GEMM (specific dtype)
    fn is_gemm_coopmat_disabled(&self, dtype_suffix: &str) -> bool {
        // Check global enable flag first - coopmat is opt-in
        if std::env::var("PARAMECIA_ENABLE_COOPMAT").is_err() {
            return true; // Disabled by default
        }

        // Check operation-wide env var (e.g., PARAMECIA_DISABLE_COOPMAT_GEMM)
        if std::env::var("PARAMECIA_DISABLE_COOPMAT_GEMM").is_ok() {
            return true;
        }

        // Check dtype-specific env var (e.g., PARAMECIA_DISABLE_COOPMAT_F32_GEMM)
        let dtype_upper = dtype_suffix.to_uppercase();
        let dtype_specific = format!("PARAMECIA_DISABLE_COOPMAT_{}_GEMM", dtype_upper);
        std::env::var(&dtype_specific).is_ok()
    }

    /// Fill shape and stride arrays (MAX_RANK=8) from a Layout.
    /// Returns (shape, stride, base_offset, rank).
    fn layout_to_arrays(layout: &Layout) -> ([u32; MAX_RANK], [u32; MAX_RANK], u32, u32) {
        let shape_slice = layout.shape();
        let stride_slice = layout.stride();
        let rank = shape_slice.rank();
        let mut shape_arr = [1u32; MAX_RANK];
        let mut stride_arr = [1u32; MAX_RANK];
        for i in 0..rank.min(MAX_RANK) {
            shape_arr[i] = shape_slice.dim(i).unwrap_or(1) as u32;
        }
        for i in 0..stride_slice.len().min(MAX_RANK) {
            stride_arr[i] = stride_slice[i] as u32;
        }
        (
            shape_arr,
            stride_arr,
            layout.start_offset() as u32,
            rank as u32,
        )
    }

    /// Fill shape and stride as uvec4 (for shaders using uvec4 instead of uint[MAX_RANK]).
    /// Returns (shape, stride, base_offset, rank).
    fn layout_to_uvec4(layout: &Layout) -> ([u32; 4], [u32; 4], u32, u32) {
        let shape_slice = layout.shape();
        let stride_slice = layout.stride();
        let rank = shape_slice.rank();
        let mut shape_arr = [1u32; 4];
        let mut stride_arr = [1u32; 4];
        for i in 0..rank.min(4) {
            shape_arr[i] = shape_slice.dim(i).unwrap_or(1) as u32;
        }
        for i in 0..stride_slice.len().min(4) {
            stride_arr[i] = stride_slice[i] as u32;
        }
        (
            shape_arr,
            stride_arr,
            layout.start_offset() as u32,
            rank as u32,
        )
    }

    /// Execute a compute dispatch with the given pipeline, bindings, push constants, and dispatch dimensions.
    /// Records into the current command batch for deferred submission.
    fn execute_compute<PC: bytemuck::Pod>(
        &self,
        pipeline: &CachedPipeline,
        buffers: &[vk::Buffer],
        push_constants: &PC,
        dispatch: [u32; 3],
    ) -> Result<()> {
        let pc_bytes = bytemuck::bytes_of(push_constants);
        let write_mask = if buffers.is_empty() {
            Some(0u64)
        } else if buffers.len() < 64 {
            Some(1u64 << (buffers.len() - 1))
        } else {
            None
        };
        self.device.record_compute_with_write_mask(
            pipeline,
            buffers,
            write_mask,
            Some(pc_bytes),
            dispatch,
        )
    }

    fn load_pipeline(
        &self,
        name: &str,
        push_constant_size: u32,
        num_buffers: u32,
    ) -> Result<CachedPipeline> {
        self.device
            .kernels()
            .load_pipeline(
                self.device.device(),
                name,
                None,
                push_constant_size,
                num_buffers,
                None,
                false,
            )
            .map_err(|e| crate::Error::Vulkan(VulkanError::from(e)))
    }

    #[allow(dead_code)]
    fn load_pipeline_with_defines(
        &self,
        name: &str,
        defines: &[(&str, &str)],
        push_constant_size: u32,
        num_buffers: u32,
    ) -> Result<CachedPipeline> {
        self.device
            .kernels()
            .load_pipeline(
                self.device.device(),
                name,
                Some(defines),
                push_constant_size,
                num_buffers,
                None,
                false,
            )
            .map_err(|e| crate::Error::Vulkan(VulkanError::from(e)))
    }

    /// Execute a compute dispatch without push constants (for shaders using only SSBOs).
    /// Records into the current command batch for deferred submission.
    pub(crate) fn execute_compute_no_pc(
        &self,
        pipeline: &CachedPipeline,
        buffers: &[vk::Buffer],
        dispatch: [u32; 3],
    ) -> Result<()> {
        let write_mask = if buffers.is_empty() {
            Some(0u64)
        } else if buffers.len() < 64 {
            Some(1u64 << (buffers.len() - 1))
        } else {
            None
        };
        self.device
            .record_compute_with_write_mask(pipeline, buffers, write_mask, None, dispatch)
    }

    /// Helper: compute workgroup dispatch for N elements with local_size=256.
    fn dispatch_1d(n: usize) -> [u32; 3] {
        let groups = ((n as u32) + 255) / 256;
        [groups, 1, 1]
    }

    fn copy_strided_src_gpu(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        layout: &Layout,
    ) -> Result<()> {
        let suffix = self.dtype_suffix()?;
        let key = format!("copy_strided_src_{}", suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            base: u32,
            rank: u32,
            shape: [u32; MAX_RANK],
            stride: [u32; MAX_RANK],
            dst_offset: u32,
            _pad: [u32; 3],
        }

        let (shape, stride, base, rank) = Self::layout_to_arrays(layout);
        let pc = PC {
            base,
            rank,
            shape,
            stride,
            dst_offset: dst_offset as u32,
            _pad: [0; 3],
        };

        let total_elements = layout.shape().elem_count();
        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 2)?;

        // Keep buffers alive until the batch is submitted
        // (the compute command records buffer handles, but if the Arc is dropped
        // before the batch executes, the buffers could be freed)
        if let Some(src_buf) = &self.buffer {
            self.device.keep_buffer_alive(src_buf.clone())?;
        }
        if let Some(dst_buf) = &dst.buffer {
            self.device.keep_buffer_alive(dst_buf.clone())?;
        }

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            Self::dispatch_1d(total_elements),
        )?;

        Ok(())
    }

    fn copy2d_gpu(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        let suffix = self.dtype_suffix()?;
        let key = format!("copy2d_{}", suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            src_offset: u32,
            dst_offset: u32,
            rows: u32,
            cols: u32,
            src_stride: u32,
            dst_stride: u32,
        }

        let pc = PC {
            src_offset: src_offset as u32,
            dst_offset: dst_offset as u32,
            rows: d1 as u32,
            cols: d2 as u32,
            src_stride: src_stride1 as u32,
            dst_stride: dst_stride1 as u32,
        };

        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 2)?;

        let dispatch_x = ((d2 as u32) + 15) / 16;
        let dispatch_y = ((d1 as u32) + 15) / 16;

        // Keep buffers alive until the batch is submitted
        if let Some(src_buf) = &self.buffer {
            self.device.keep_buffer_alive(src_buf.clone())?;
        }
        if let Some(dst_buf) = &dst.buffer {
            self.device.keep_buffer_alive(dst_buf.clone())?;
        }

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            [dispatch_x, dispatch_y, 1],
        )?;

        Ok(())
    }

    /// Dispatch rope_i (interleaved rotary embeddings) on Vulkan.
    /// src shape: (B, H, T, D), cos/sin shape: (T, D/2) or (B, T, D/2)
    pub fn rope_i(
        &self,
        l_src: &Layout,
        cos: &Self,
        l_cos: &Layout,
        sin: &Self,
        l_sin: &Layout,
    ) -> Result<(Self, Shape)> {
        let suffix = self.dtype_suffix()?;
        let key = format!("rope_i_{}", suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            shape: [u32; 4],
            strides: [u32; 4],
            base: u32,
            stride_b: u32,
            _pad: [u32; 2],
        }

        let (b, h, t, d) = l_src.shape().dims4()?;
        let strides = l_src.stride();
        let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
            (h * t * d) as u32
        } else {
            0u32
        };

        let pc = PC {
            shape: [b as u32, h as u32, t as u32, d as u32],
            strides: [
                strides[0] as u32,
                strides[1] as u32,
                strides[2] as u32,
                strides[3] as u32,
            ],
            base: l_src.start_offset() as u32,
            stride_b,
            _pad: [0; 2],
        };

        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 4)?;

        let el = b * h * t * d;
        let out_shape = Shape::from((b, h, t, d));
        let dst = self.device.zeros_impl(&out_shape, self.dtype)?;

        let total_pairs = (el / 2) as u32;
        let groups = (total_pairs + 255) / 256;

        self.execute_compute(
            &pipeline,
            &[
                self.vk_buffer()?,
                cos.vk_buffer()?,
                sin.vk_buffer()?,
                dst.vk_buffer()?,
            ],
            &pc,
            [groups, 1, 1],
        )?;

        Ok((dst, out_shape))
    }

    /// Dispatch rope (contiguous rotary embeddings) on Vulkan.
    /// src shape: (B, H, T, D), cos/sin shape: (T, D/2) or (B, T, D/2)
    pub fn rope(
        &self,
        l_src: &Layout,
        cos: &Self,
        l_cos: &Layout,
        sin: &Self,
        l_sin: &Layout,
    ) -> Result<(Self, Shape)> {
        let suffix = self.dtype_suffix()?;
        let key = format!("rope_{}", suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            shape: [u32; 4],
            strides: [u32; 4],
            base: u32,
            stride_b: u32,
            _pad: [u32; 2],
        }

        let (b, h, t, d) = l_src.shape().dims4()?;
        let strides = l_src.stride();
        let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
            (h * t * d) as u32
        } else {
            0u32
        };

        let pc = PC {
            shape: [b as u32, h as u32, t as u32, d as u32],
            strides: [
                strides[0] as u32,
                strides[1] as u32,
                strides[2] as u32,
                strides[3] as u32,
            ],
            base: l_src.start_offset() as u32,
            stride_b,
            _pad: [0; 2],
        };

        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 4)?;

        let el = b * h * t * d;
        let out_shape = Shape::from((b, h, t, d));
        let dst = self.device.zeros_impl(&out_shape, self.dtype)?;

        let groups = ((el as u32) + 255) / 256;

        self.execute_compute(
            &pipeline,
            &[
                self.vk_buffer()?,
                cos.vk_buffer()?,
                sin.vk_buffer()?,
                dst.vk_buffer()?,
            ],
            &pc,
            [groups, 1, 1],
        )?;

        Ok((dst, out_shape))
    }

    /// Dispatch rope_thd (T/H/D layout rotary embeddings) on Vulkan.
    /// src shape: (B, T, H, D), cos/sin shape: (T, D/2) or (B, T, D/2)
    pub fn rope_thd(
        &self,
        l_src: &Layout,
        cos: &Self,
        l_cos: &Layout,
        sin: &Self,
        l_sin: &Layout,
    ) -> Result<(Self, Shape)> {
        let suffix = self.dtype_suffix()?;
        let key = format!("rope_thd_{}", suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            shape: [u32; 4],
            strides: [u32; 4],
            base: u32,
            stride_b: u32,
            _pad: [u32; 2],
        }

        let (b, t, h, d) = l_src.shape().dims4()?;
        let strides = l_src.stride();
        let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
            (h * t * d) as u32
        } else {
            0u32
        };

        // For THD=1 shader: shape is (B, T, H, D) but shader interprets
        // shape.x=B, shape.y=T, shape.z=H, shape.w=D (same order as dims)
        let pc = PC {
            shape: [b as u32, t as u32, h as u32, d as u32],
            strides: [
                strides[0] as u32,
                strides[1] as u32,
                strides[2] as u32,
                strides[3] as u32,
            ],
            base: l_src.start_offset() as u32,
            stride_b,
            _pad: [0; 2],
        };

        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 4)?;

        let el = b * t * h * d;
        let out_shape = Shape::from((b, t, h, d));
        let dst = self.device.zeros_impl(&out_shape, self.dtype)?;

        let groups = ((el as u32) + 255) / 256;

        self.execute_compute(
            &pipeline,
            &[
                self.vk_buffer()?,
                cos.vk_buffer()?,
                sin.vk_buffer()?,
                dst.vk_buffer()?,
            ],
            &pc,
            [groups, 1, 1],
        )?;

        Ok((dst, out_shape))
    }
}

impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        if super::debug::force_cpu_const() {
            let cpu = self.to_cpu_storage()?;
            let cpu_result = cpu.try_clone(layout)?;
            return self.device.storage_from_cpu_storage(&cpu_result);
        }
        let src = match &self.buffer {
            Some(b) => b,
            None => return Ok(Self::new(None, self.device.clone(), 0, self.dtype)),
        };
        let byte_size = src.size;
        let dst = self.device.allocate_buffer(byte_size)?;

        // Record copy into batch command buffer (async, no fence wait).
        // The copy will execute when the batch is next flushed.
        self.device.record_copy(src.buffer, dst.buffer, byte_size)?;

        Ok(Self::new(
            Some(Arc::new(dst)),
            self.device.clone(),
            self.count,
            self.dtype,
        ))
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &VulkanDevice {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        // No explicit flush needed: download_buffer records the copy into the
        // batch with proper barriers and flushes once, avoiding double-flush.
        let buffer = match &self.buffer {
            Some(b) => b,
            None => {
                return Ok(match self.dtype {
                    DType::U8 => CpuStorage::U8(vec![]),
                    DType::U32 => CpuStorage::U32(vec![]),
                    DType::I64 => CpuStorage::I64(vec![]),
                    DType::I16 => CpuStorage::I16(vec![]),
                    DType::I32 => CpuStorage::I32(vec![]),
                    DType::BF16 => CpuStorage::BF16(vec![]),
                    DType::F16 => CpuStorage::F16(vec![]),
                    DType::F32 => CpuStorage::F32(vec![]),
                    DType::F64 => CpuStorage::F64(vec![]),
                    DType::F8E4M3 => CpuStorage::F8E4M3(vec![]),
                });
            }
        };

        Ok(match self.dtype {
            DType::U8 => CpuStorage::U8(self.device.download_buffer::<u8>(buffer, self.count)?),
            DType::U32 => CpuStorage::U32(self.device.download_buffer::<u32>(buffer, self.count)?),
            DType::I64 => CpuStorage::I64(self.device.download_buffer::<i64>(buffer, self.count)?),
            DType::I16 => CpuStorage::I16(self.device.download_buffer::<i16>(buffer, self.count)?),
            DType::I32 => CpuStorage::I32(self.device.download_buffer::<i32>(buffer, self.count)?),
            DType::BF16 => {
                CpuStorage::BF16(self.device.download_buffer::<bf16>(buffer, self.count)?)
            }
            DType::F16 => CpuStorage::F16(self.device.download_buffer::<f16>(buffer, self.count)?),
            DType::F32 => CpuStorage::F32(self.device.download_buffer::<f32>(buffer, self.count)?),
            DType::F64 => CpuStorage::F64(self.device.download_buffer::<f64>(buffer, self.count)?),
            _ => crate::bail!("Vulkan to_cpu_storage: unsupported dtype {:?}", self.dtype),
        })
    }

    // =========================================================================
    // affine / powf / elu — shader: affine_elu.comp
    // Push constants: { uint base, uint rank, _pad[2], uvec4 shape, uvec4 stride, float mul, add, alpha }
    // Buffers: 2 (input, output)
    // =========================================================================

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        if super::debug::force_cpu_affine() {
            let cpu = self.to_cpu_storage()?;
            let cpu_result = cpu.affine(layout, mul, add)?;
            return self.device.storage_from_cpu_storage(&cpu_result);
        }
        let suffix = self.dtype_suffix()?;
        let key = format!("affine_{}", suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            base: u32,
            rank: u32,
            _pad: [u32; 2],
            shape: [u32; 4],
            stride: [u32; 4],
            mul: f32,
            add: f32,
            alpha: f32,
            _pad2: u32,
        }

        let (shape, stride, base, rank) = Self::layout_to_uvec4(layout);
        let pc = PC {
            base,
            rank,
            _pad: [0; 2],
            shape,
            stride,
            mul: mul as f32,
            add: add as f32,
            alpha: 0.0,
            _pad2: 0,
        };

        let num_elements = layout.shape().elem_count();
        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 2)?;
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            Self::dispatch_1d(num_elements),
        )?;

        Ok(dst)
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        if super::debug::force_cpu_affine() {
            let cpu = self.to_cpu_storage()?;
            let cpu_result = cpu.powf(layout, e)?;
            return self.device.storage_from_cpu_storage(&cpu_result);
        }
        let suffix = self.dtype_suffix()?;
        let key = format!("powf_{}", suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            base: u32,
            rank: u32,
            _pad: [u32; 2],
            shape: [u32; 4],
            stride: [u32; 4],
            mul: f32,
            add: f32,
            alpha: f32,
            _pad2: u32,
        }

        let (shape, stride, base, rank) = Self::layout_to_uvec4(layout);
        let pc = PC {
            base,
            rank,
            _pad: [0; 2],
            shape,
            stride,
            mul: e as f32,
            add: 0.0,
            alpha: 0.0,
            _pad2: 0,
        };

        let num_elements = layout.shape().elem_count();
        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 2)?;
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            Self::dispatch_1d(num_elements),
        )?;

        Ok(dst)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        if super::debug::force_cpu_affine() {
            let cpu = self.to_cpu_storage()?;
            let cpu_result = cpu.elu(layout, alpha)?;
            return self.device.storage_from_cpu_storage(&cpu_result);
        }
        let suffix = self.dtype_suffix()?;
        let key = format!("elu_{}", suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            base: u32,
            rank: u32,
            _pad: [u32; 2],
            shape: [u32; 4],
            stride: [u32; 4],
            mul: f32,
            add: f32,
            alpha: f32,
            _pad2: u32,
        }

        let (shape, stride, base, rank) = Self::layout_to_uvec4(layout);
        let pc = PC {
            base,
            rank,
            _pad: [0; 2],
            shape,
            stride,
            mul: 1.0,
            add: 0.0,
            alpha: alpha as f32,
            _pad2: 0,
        };

        let num_elements = layout.shape().elem_count();
        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 2)?;
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            Self::dispatch_1d(num_elements),
        )?;

        Ok(dst)
    }

    // =========================================================================
    // reduce_op — shaders: reduce_partial.comp + reduce_combine.comp (2-pass)
    // =========================================================================

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, reduce_dims: &[usize]) -> Result<Self> {
        if super::debug::force_cpu_reduce() {
            let cpu = self.to_cpu_storage()?;
            let cpu_result = cpu.reduce_op(op, layout, reduce_dims)?;
            return self.device.storage_from_cpu_storage(&cpu_result);
        }
        let suffix = self.dtype_suffix()?;
        let (partial_key, combine_key) = match op {
            ReduceOp::Sum => (
                format!("sum_partial_{}", suffix),
                format!("sum_combine_{}", suffix),
            ),
            ReduceOp::Max => (
                format!("max_partial_{}", suffix),
                format!("max_combine_{}", suffix),
            ),
            ReduceOp::Min => (
                format!("min_partial_{}", suffix),
                format!("min_combine_{}", suffix),
            ),
            ReduceOp::ArgMax => (
                format!("argmax_partial_{}", suffix),
                format!("argmax_combine_{}", suffix),
            ),
            ReduceOp::ArgMin => (
                format!("argmin_partial_{}", suffix),
                format!("argmin_combine_{}", suffix),
            ),
        };

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct ReducePC {
            base: u32,
            rank: u32,
            shape: [u32; MAX_RANK],
            stride: [u32; MAX_RANK],
            reduce_axes: [u32; MAX_RANK],
        }

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct CombinePC {
            num_partials: u32,
        }

        let (shape_arr, stride_arr, base, rank) = Self::layout_to_arrays(layout);

        // Build reduce_axes array: pack reduced axis indices at positions 0, 1, 2, ...
        // The shader iterates from index 0 and breaks at the first 0xFFFFFFFF sentinel.
        let mut reduce_axes_arr = [u32::MAX; MAX_RANK];
        for (i, &ax) in reduce_dims.iter().enumerate() {
            if i < MAX_RANK {
                reduce_axes_arr[i] = ax as u32;
            }
        }

        // Compute flat reduction size (product of reduced dimensions)
        let mut flat_reduction_size = 1u32;
        for &ax in reduce_dims {
            if ax < rank as usize {
                flat_reduction_size *= shape_arr[ax];
            }
        }

        // Compute number of batches (product of non-reduced dimensions)
        let mut num_batches = 1u32;
        for d in 0..(rank as usize) {
            if !reduce_dims.contains(&d) {
                num_batches *= shape_arr[d];
            }
        }

        let wg_size = 256u32;
        let segments_per_batch = (flat_reduction_size + wg_size - 1) / wg_size;
        let total_workgroups = num_batches * segments_per_batch;

        let result_dtype = match op {
            ReduceOp::ArgMax | ReduceOp::ArgMin => DType::U32,
            _ => self.dtype,
        };

        let pc = ReducePC {
            base,
            rank,
            shape: shape_arr,
            stride: stride_arr,
            reduce_axes: reduce_axes_arr,
        };

        // Pass 1: partial reduction
        let partial_shape = Shape::from(&[num_batches as usize * segments_per_batch as usize]);
        let partial_values = unsafe { self.device.alloc_uninit(&partial_shape, result_dtype)? };
        let partial_indices = unsafe { self.device.alloc_uninit(&partial_shape, DType::U32)? };

        let partial_pipeline =
            self.load_pipeline(&partial_key, std::mem::size_of::<ReducePC>() as u32, 3)?;

        self.execute_compute(
            &partial_pipeline,
            &[
                self.vk_buffer()?,
                partial_values.vk_buffer()?,
                partial_indices.vk_buffer()?,
            ],
            &pc,
            [total_workgroups, 1, 1],
        )?;

        // Pass 2: combine (if needed)
        if segments_per_batch > 1 {
            let combine_pc = CombinePC {
                num_partials: segments_per_batch,
            };

            let final_shape = Shape::from(&[num_batches as usize]);
            let final_values = unsafe { self.device.alloc_uninit(&final_shape, result_dtype)? };
            let final_indices = unsafe { self.device.alloc_uninit(&final_shape, DType::U32)? };

            let combine_pipeline =
                self.load_pipeline(&combine_key, std::mem::size_of::<CombinePC>() as u32, 4)?;

            self.execute_compute(
                &combine_pipeline,
                &[
                    partial_values.vk_buffer()?,
                    partial_indices.vk_buffer()?,
                    final_values.vk_buffer()?,
                    final_indices.vk_buffer()?,
                ],
                &combine_pc,
                [num_batches, 1, 1],
            )?;

            match op {
                ReduceOp::ArgMax | ReduceOp::ArgMin => Ok(final_indices),
                _ => Ok(final_values),
            }
        } else {
            match op {
                ReduceOp::ArgMax | ReduceOp::ArgMin => Ok(partial_indices),
                _ => Ok(partial_values),
            }
        }
    }

    // =========================================================================
    // cmp — shader: cmp.comp
    // Push constants: { uint a_base, a_rank, _pad[2], uvec4 a_shape, a_stride, uint b_base, b_rank, _pad[2], uvec4 b_shape, b_stride }
    // Buffers: 3 (lhs, rhs, output)
    // =========================================================================

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if super::debug::force_cpu_binary() {
            let cpu_lhs = self.to_cpu_storage()?;
            let cpu_rhs = rhs.to_cpu_storage()?;
            let cpu_result = cpu_lhs.cmp(op, &cpu_rhs, lhs_l, rhs_l)?;
            return self.device.storage_from_cpu_storage(&cpu_result);
        }
        let suffix = self.dtype_suffix()?;
        let op_name = match op {
            CmpOp::Eq => "eq",
            CmpOp::Ne => "ne",
            CmpOp::Lt => "lt",
            CmpOp::Gt => "gt",
            CmpOp::Le => "le",
            CmpOp::Ge => "ge",
        };
        let key = format!("{}_{}", op_name, suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            a_base: u32,
            a_rank: u32,
            _pad0: [u32; 2],
            a_shape: [u32; 4],
            a_stride: [u32; 4],
            b_base: u32,
            b_rank: u32,
            _pad1: [u32; 2],
            b_shape: [u32; 4],
            b_stride: [u32; 4],
        }

        let (a_shape, a_stride, a_base, a_rank) = Self::layout_to_uvec4(lhs_l);
        let (b_shape, b_stride, b_base, b_rank) = Self::layout_to_uvec4(rhs_l);

        let pc = PC {
            a_base,
            a_rank,
            _pad0: [0; 2],
            a_shape,
            a_stride,
            b_base,
            b_rank,
            _pad1: [0; 2],
            b_shape,
            b_stride,
        };

        let num_elements = lhs_l.shape().elem_count();
        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 3)?;
        let dst = unsafe { self.device.alloc_uninit(lhs_l.shape(), DType::U8)? };

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, rhs.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            Self::dispatch_1d(num_elements),
        )?;

        Ok(dst)
    }

    // =========================================================================
    // to_dtype — shader: cast.comp
    // Push constants: { uint rank, uint base, uint shape[MAX_RANK], uint stride[MAX_RANK] }
    // Buffers: 2 (input, output)
    // =========================================================================

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        if self.dtype == dtype {
            return self.try_clone(layout);
        }
        if super::debug::force_cpu_cast() {
            let cpu = self.to_cpu_storage()?;
            let cpu_result = cpu.to_dtype(layout, dtype)?;
            return self.device.storage_from_cpu_storage(&cpu_result);
        }
        if super::debug::trace_ops() {
            use std::sync::atomic::{AtomicU32, Ordering};
            static CAST_COUNT: AtomicU32 = AtomicU32::new(0);
            let n = CAST_COUNT.fetch_add(1, Ordering::Relaxed);
            trace!(
                "[vk cast #{}] {:?}→{:?} shape={:?} stride={:?} off={} buf={}",
                n,
                self.dtype,
                dtype,
                layout.shape().dims(),
                layout.stride(),
                layout.start_offset(),
                self.count
            );
        }
        let src_suffix = self.dtype_suffix()?;
        let dst_suffix = match dtype {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::U32 => "u32",
            DType::U8 => "u8",
            DType::I64 => "i64",
            DType::F64 => crate::bail!("Vulkan does not support F64"),
            _ => crate::bail!("Vulkan to_dtype: unsupported target dtype {:?}", dtype),
        };
        let key = format!("cast_{}_{}", src_suffix, dst_suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            rank: u32,
            base: u32,
            shape: [u32; MAX_RANK],
            stride: [u32; MAX_RANK],
        }

        let (shape, stride, base, rank) = Self::layout_to_arrays(layout);
        let pc = PC {
            rank,
            base,
            shape,
            stride,
        };

        let num_elements = layout.shape().elem_count();
        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 2)?;
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), dtype)? };

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            Self::dispatch_1d(num_elements),
        )?;

        // Validation: compare GPU cast with CPU cast
        {
            use std::sync::atomic::{AtomicU32, Ordering};
            static CAST_VALIDATE_COUNT: AtomicU32 = AtomicU32::new(0);
            let max_validate = super::debug::validate_count();
            if max_validate > 0 {
                let count = CAST_VALIDATE_COUNT.fetch_add(1, Ordering::Relaxed);
                if count < max_validate {
                    let cpu_src = self.to_cpu_storage()?;
                    let cpu_expected = cpu_src.to_dtype(layout, dtype)?;
                    let gpu_result = dst.to_cpu_storage()?;
                    let gpu_f32 = gpu_result
                        .to_dtype(&crate::Layout::contiguous(layout.shape()), DType::F32)?;
                    let cpu_f32 = cpu_expected
                        .to_dtype(&crate::Layout::contiguous(layout.shape()), DType::F32)?;
                    if let (CpuStorage::F32(gpu_v), CpuStorage::F32(cpu_v)) = (&gpu_f32, &cpu_f32) {
                        let mut max_diff = 0.0f32;
                        let mut nan_count = 0;
                        for (g, c) in gpu_v.iter().zip(cpu_v.iter()) {
                            if g.is_nan() {
                                nan_count += 1;
                            }
                            let diff = (g - c).abs();
                            if diff > max_diff {
                                max_diff = diff;
                            }
                        }
                        let status = if max_diff < 0.01 && nan_count == 0 {
                            "OK"
                        } else {
                            "MISMATCH"
                        };
                        if status == "MISMATCH" {
                            warn!(
                                "[vk validate cast #{}] {:?}→{:?} shape={:?} stride={:?} off={} elems={} max_diff={:.6} nans={} → {}",
                                count,
                                self.dtype,
                                dtype,
                                layout.shape().dims(),
                                layout.stride(),
                                layout.start_offset(),
                                num_elements,
                                max_diff,
                                nan_count,
                                status
                            );
                            let n = 10.min(gpu_v.len());
                            warn!("  gpu[:{}]={:?}", n, &gpu_v[..n]);
                            warn!("  cpu[:{}]={:?}", n, &cpu_v[..n]);
                        } else {
                            trace!(
                                "[vk validate cast #{}] {:?}→{:?} shape={:?} stride={:?} off={} elems={} max_diff={:.6} nans={} → {}",
                                count,
                                self.dtype,
                                dtype,
                                layout.shape().dims(),
                                layout.stride(),
                                layout.start_offset(),
                                num_elements,
                                max_diff,
                                nan_count,
                                status
                            );
                        }
                    }
                }
            }
        }

        Ok(dst)
    }

    // =========================================================================
    // unary_impl — shader: unary.comp
    // Push constants: { uint rank, uint base, uint shape[MAX_RANK], uint stride[MAX_RANK] }
    // Buffers: 2 (input, output)
    // =========================================================================

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        if super::debug::force_cpu_unary() {
            let cpu = self.to_cpu_storage()?;
            let cpu_result = cpu.unary_impl::<B>(layout)?;
            return self.device.storage_from_cpu_storage(&cpu_result);
        }
        let suffix = self.dtype_suffix()?;
        let op_name = B::NAME;
        let key = format!("{}_{}", op_name, suffix);

        if super::debug::trace_ops() {
            trace!(
                "[vk unary] op={} dtype={:?} shape={:?} stride={:?} offset={} count={} buf_size={}",
                key,
                self.dtype,
                layout.shape().dims(),
                layout.stride(),
                layout.start_offset(),
                self.count,
                self.buffer.as_ref().map(|b| b.size).unwrap_or(0)
            );
        }

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            rank: u32,
            base: u32,
            shape: [u32; MAX_RANK],
            stride: [u32; MAX_RANK],
        }

        let (shape, stride, base, rank) = Self::layout_to_arrays(layout);
        let pc = PC {
            rank,
            base,
            shape,
            stride,
        };

        let num_elements = layout.shape().elem_count();
        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 2)?;
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), self.dtype)? };

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            Self::dispatch_1d(num_elements),
        )?;

        Ok(dst)
    }

    // =========================================================================
    // binary_impl — shader: binary.comp
    // Push constants: { uint a_rank, a_base, a_shape[MAX_RANK], a_stride[MAX_RANK],
    //                    uint b_rank, b_base, b_shape[MAX_RANK], b_stride[MAX_RANK] }
    // Buffers: 3 (lhs, rhs, output)
    // =========================================================================

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if super::debug::force_cpu_binary() {
            let cpu_lhs = self.to_cpu_storage()?;
            let cpu_rhs = rhs.to_cpu_storage()?;
            let cpu_result = cpu_lhs.binary_impl::<B>(&cpu_rhs, lhs_l, rhs_l)?;
            return self.device.storage_from_cpu_storage(&cpu_result);
        }
        if super::debug::trace_ops() {
            use std::sync::atomic::{AtomicU32, Ordering};
            static BIN_COUNT: AtomicU32 = AtomicU32::new(0);
            let n = BIN_COUNT.fetch_add(1, Ordering::Relaxed);
            trace!(
                "[vk binary #{}] op={} dtype={:?} lhs_shape={:?} lhs_stride={:?} lhs_off={} lhs_buf={} rhs_shape={:?} rhs_stride={:?} rhs_off={} rhs_buf={}",
                n,
                B::NAME,
                self.dtype,
                lhs_l.shape().dims(),
                lhs_l.stride(),
                lhs_l.start_offset(),
                self.count,
                rhs_l.shape().dims(),
                rhs_l.stride(),
                rhs_l.start_offset(),
                rhs.count
            );
        }
        let suffix = self.dtype_suffix()?;
        let op_name = B::NAME;
        let key = format!("{}_{}", op_name, suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            a_rank: u32,
            a_base: u32,
            a_shape: [u32; MAX_RANK],
            a_stride: [u32; MAX_RANK],
            b_rank: u32,
            b_base: u32,
            b_shape: [u32; MAX_RANK],
            b_stride: [u32; MAX_RANK],
        }

        let (a_shape, a_stride, a_base, a_rank) = Self::layout_to_arrays(lhs_l);
        let (b_shape, b_stride, b_base, b_rank) = Self::layout_to_arrays(rhs_l);

        let pc = PC {
            a_rank,
            a_base,
            a_shape,
            a_stride,
            b_rank,
            b_base,
            b_shape,
            b_stride,
        };

        let num_elements = lhs_l.shape().elem_count();
        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 3)?;
        let dst = unsafe { self.device.alloc_uninit(lhs_l.shape(), self.dtype)? };

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, rhs.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            Self::dispatch_1d(num_elements),
        )?;

        // Validation: compare GPU result with CPU result
        {
            use std::sync::atomic::{AtomicU32, Ordering};
            static BINARY_VALIDATE_COUNT: AtomicU32 = AtomicU32::new(0);
            let max_validate = super::debug::validate_count();
            if max_validate > 0 {
                let count = BINARY_VALIDATE_COUNT.fetch_add(1, Ordering::Relaxed);
                if count < max_validate {
                    let cpu_lhs = self.to_cpu_storage()?;
                    let cpu_rhs = rhs.to_cpu_storage()?;
                    let cpu_expected = cpu_lhs.binary_impl::<B>(&cpu_rhs, lhs_l, rhs_l)?;
                    let gpu_result = dst.to_cpu_storage()?;
                    // Compare as f32
                    let gpu_f32 = gpu_result
                        .to_dtype(&crate::Layout::contiguous(lhs_l.shape()), DType::F32)?;
                    let cpu_f32 = cpu_expected
                        .to_dtype(&crate::Layout::contiguous(lhs_l.shape()), DType::F32)?;
                    if let (CpuStorage::F32(gpu_v), CpuStorage::F32(cpu_v)) = (&gpu_f32, &cpu_f32) {
                        let mut max_diff = 0.0f32;
                        let mut max_idx = 0;
                        let mut nan_count = 0;
                        for (i, (g, c)) in gpu_v.iter().zip(cpu_v.iter()).enumerate() {
                            if g.is_nan() {
                                nan_count += 1;
                            }
                            let diff = (g - c).abs();
                            if diff > max_diff || diff.is_nan() {
                                max_diff = diff;
                                max_idx = i;
                            }
                        }
                        let status = if max_diff < 0.01 && nan_count == 0 {
                            "OK"
                        } else {
                            "MISMATCH"
                        };
                        if status == "MISMATCH" {
                            warn!(
                                "[vk validate binary #{}] op={} dtype={:?} shape={:?} lhs_stride={:?} lhs_off={} rhs_stride={:?} rhs_off={} elems={} max_diff={:.6} at idx {} nans={} lhs_buf={} rhs_buf={} → {}",
                                count,
                                key,
                                self.dtype,
                                lhs_l.shape().dims(),
                                lhs_l.stride(),
                                lhs_l.start_offset(),
                                rhs_l.stride(),
                                rhs_l.start_offset(),
                                num_elements,
                                max_diff,
                                max_idx,
                                nan_count,
                                self.count,
                                rhs.count,
                                status
                            );
                            let n = 10.min(gpu_v.len());
                            warn!("  gpu[:{}]={:?}", n, &gpu_v[..n]);
                            warn!("  cpu[:{}]={:?}", n, &cpu_v[..n]);
                            if max_idx > 0 {
                                let lo = max_idx.saturating_sub(2);
                                let hi = (max_idx + 3).min(gpu_v.len());
                                warn!("  gpu[{}..{}]={:?}", lo, hi, &gpu_v[lo..hi]);
                                warn!("  cpu[{}..{}]={:?}", lo, hi, &cpu_v[lo..hi]);
                            }
                        } else {
                            trace!(
                                "[vk validate binary #{}] op={} dtype={:?} shape={:?} lhs_stride={:?} lhs_off={} rhs_stride={:?} rhs_off={} elems={} max_diff={:.6} at idx {} nans={} lhs_buf={} rhs_buf={} → {}",
                                count,
                                key,
                                self.dtype,
                                lhs_l.shape().dims(),
                                lhs_l.stride(),
                                lhs_l.start_offset(),
                                rhs_l.stride(),
                                rhs_l.start_offset(),
                                num_elements,
                                max_diff,
                                max_idx,
                                nan_count,
                                self.count,
                                rhs.count,
                                status
                            );
                        }
                    }
                }
            }
        }

        Ok(dst)
    }

    // =========================================================================
    // where_cond — shader: where.comp
    // Push constants: { uint elem_count, uint base }
    // Buffers: 4 (cond, true, false, output)
    // =========================================================================

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        if super::debug::force_cpu_where() {
            let cpu_cond = self.to_cpu_storage()?;
            let cpu_t = t.to_cpu_storage()?;
            let cpu_f = f.to_cpu_storage()?;
            let cpu_result = cpu_cond.where_cond(layout, &cpu_t, t_l, &cpu_f, f_l)?;
            return self.device.storage_from_cpu_storage(&cpu_result);
        }
        let arg_suffix = match t.dtype {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I64 => "i64",
            DType::U32 => "u32",
            DType::U8 => "u8",
            _ => crate::bail!("Vulkan where_cond unsupported arg dtype {:?}", t.dtype),
        };
        let cond_suffix = self.dtype_suffix()?;
        let key = format!("where_{}_{}", cond_suffix, arg_suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            elem_count: u32,
            base: u32,
        }

        let num_elements = layout.shape().elem_count();
        let pc = PC {
            elem_count: num_elements as u32,
            base: layout.start_offset() as u32,
        };

        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 4)?;
        let dst = unsafe { self.device.alloc_uninit(layout.shape(), t.dtype)? };

        self.execute_compute(
            &pipeline,
            &[
                self.vk_buffer()?,
                t.vk_buffer()?,
                f.vk_buffer()?,
                dst.vk_buffer()?,
            ],
            &pc,
            Self::dispatch_1d(num_elements),
        )?;

        Ok(dst)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        let suffix = self.dtype_suffix()?;
        let key = format!("conv1d_{}", suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            in_base: u32,
            in_rank: u32,
            ker_base: u32,
            ker_rank: u32,
            in_stride: [u32; 4],
            ker_stride: [u32; 4],
            elem_count: u32,
            b_size: u32,
            l_in: u32,
            c_out: u32,
            c_in: u32,
            k_size: u32,
            l_out: u32,
            padding: u32,
            stride: u32,
            dilation: u32,
        }

        let l_out = params.l_out();
        let elem_count = params.b_size * params.c_out * l_out;

        let in_stride = l.stride();
        let ker_stride = kernel_l.stride();
        let mut in_s = [0u32; 4];
        let mut ker_s = [0u32; 4];
        for i in 0..in_stride.len().min(4) {
            in_s[i] = in_stride[i] as u32;
        }
        for i in 0..ker_stride.len().min(4) {
            ker_s[i] = ker_stride[i] as u32;
        }

        let pc = PC {
            in_base: l.start_offset() as u32,
            in_rank: l.shape().rank() as u32,
            ker_base: kernel_l.start_offset() as u32,
            ker_rank: kernel_l.shape().rank() as u32,
            in_stride: in_s,
            ker_stride: ker_s,
            elem_count: elem_count as u32,
            b_size: params.b_size as u32,
            l_in: params.l_in as u32,
            c_out: params.c_out as u32,
            c_in: params.c_in as u32,
            k_size: params.k_size as u32,
            l_out: l_out as u32,
            padding: params.padding as u32,
            stride: params.stride as u32,
            dilation: params.dilation as u32,
        };

        let out_shape = Shape::from(&[params.b_size, params.c_out, l_out]);
        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 3)?;
        let dst = self.device.zeros_impl(&out_shape, self.dtype)?;

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, kernel.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            Self::dispatch_1d(elem_count),
        )?;

        Ok(dst)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let suffix = self.dtype_suffix()?;
        let key = format!("conv_transpose1d_{}", suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            in_base: u32,
            in_rank: u32,
            _pad_in: [u32; 2],
            in_shape: [u32; 4],
            in_stride: [u32; 4],
            ker_base: u32,
            ker_rank: u32,
            _pad_ker: [u32; 2],
            ker_shape: [u32; 4],
            ker_stride: [u32; 4],
            b_size: u32,
            c_in: u32,
            l_in: u32,
            c_out: u32,
            k_size: u32,
            l_out: u32,
            padding: u32,
            stride: u32,
            dilation: u32,
            output_padding: u32,
        }

        let l_out = params.l_out();
        let elem_count = params.b_size * params.c_out * l_out;

        let (in_shape, in_stride, in_base, in_rank) = Self::layout_to_uvec4(l);
        let (ker_shape, ker_stride, ker_base, ker_rank) = Self::layout_to_uvec4(kernel_l);

        let pc = PC {
            in_base,
            in_rank,
            _pad_in: [0; 2],
            in_shape,
            in_stride,
            ker_base,
            ker_rank,
            _pad_ker: [0; 2],
            ker_shape,
            ker_stride,
            b_size: params.b_size as u32,
            c_in: params.c_in as u32,
            l_in: params.l_in as u32,
            c_out: params.c_out as u32,
            k_size: params.k_size as u32,
            l_out: l_out as u32,
            padding: params.padding as u32,
            stride: params.stride as u32,
            dilation: params.dilation as u32,
            output_padding: params.output_padding as u32,
        };

        let out_shape = Shape::from(&[params.b_size, params.c_out, l_out]);
        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 3)?;
        let dst = self.device.zeros_impl(&out_shape, self.dtype)?;

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, kernel.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            Self::dispatch_1d(elem_count),
        )?;

        Ok(dst)
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let suffix = self.dtype_suffix()?;
        let key = format!("conv2d_{}", suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            in_base: u32,
            in_rank: u32,
            _pad_in: [u32; 2],
            in_shape: [u32; 4],
            in_stride: [u32; 4],
            ker_base: u32,
            ker_rank: u32,
            _pad_ker: [u32; 2],
            ker_shape: [u32; 4],
            ker_stride: [u32; 4],
            b_size: u32,
            c_in: u32,
            h_in: u32,
            w_in: u32,
            c_out: u32,
            k_h: u32,
            k_w: u32,
            h_out: u32,
            w_out: u32,
            padding: u32,
            stride: u32,
            dilation: u32,
        }

        let h_out = params.out_h();
        let w_out = params.out_w();
        let elem_count = params.b_size * params.c_out * h_out * w_out;

        let (in_shape, in_stride, in_base, in_rank) = Self::layout_to_uvec4(l);
        let (ker_shape, ker_stride, ker_base, ker_rank) = Self::layout_to_uvec4(kernel_l);

        let pc = PC {
            in_base,
            in_rank,
            _pad_in: [0; 2],
            in_shape,
            in_stride,
            ker_base,
            ker_rank,
            _pad_ker: [0; 2],
            ker_shape,
            ker_stride,
            b_size: params.b_size as u32,
            c_in: params.c_in as u32,
            h_in: params.i_h as u32,
            w_in: params.i_w as u32,
            c_out: params.c_out as u32,
            k_h: params.k_h as u32,
            k_w: params.k_w as u32,
            h_out: h_out as u32,
            w_out: w_out as u32,
            padding: params.padding as u32,
            stride: params.stride as u32,
            dilation: params.dilation as u32,
        };

        let out_shape = Shape::from(&[params.b_size, params.c_out, h_out, w_out]);
        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 3)?;
        let dst = self.device.zeros_impl(&out_shape, self.dtype)?;

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, kernel.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            Self::dispatch_1d(elem_count),
        )?;

        Ok(dst)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let suffix = self.dtype_suffix()?;
        let key = format!("conv_transpose2d_{}", suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            in_base: u32,
            in_rank: u32,
            _pad_in: [u32; 2],
            in_shape: [u32; 4],
            in_stride: [u32; 4],
            ker_base: u32,
            ker_rank: u32,
            _pad_ker: [u32; 2],
            ker_shape: [u32; 4],
            ker_stride: [u32; 4],
            b_size: u32,
            c_in_t: u32,
            h_in: u32,
            w_in: u32,
            c_out_t: u32,
            k_h: u32,
            k_w: u32,
            h_out: u32,
            w_out: u32,
            padding: u32,
            stride: u32,
            dilation: u32,
        }

        let h_out = params.out_h();
        let w_out = params.out_w();
        let elem_count = params.b_size * params.c_out * h_out * w_out;

        let (in_shape, in_stride, in_base, in_rank) = Self::layout_to_uvec4(l);
        let (ker_shape, ker_stride, ker_base, ker_rank) = Self::layout_to_uvec4(kernel_l);

        let pc = PC {
            in_base,
            in_rank,
            _pad_in: [0; 2],
            in_shape,
            in_stride,
            ker_base,
            ker_rank,
            _pad_ker: [0; 2],
            ker_shape,
            ker_stride,
            b_size: params.b_size as u32,
            c_in_t: params.c_in as u32,
            h_in: params.i_h as u32,
            w_in: params.i_w as u32,
            c_out_t: params.c_out as u32,
            k_h: params.k_h as u32,
            k_w: params.k_w as u32,
            h_out: h_out as u32,
            w_out: w_out as u32,
            padding: params.padding as u32,
            stride: params.stride as u32,
            dilation: params.dilation as u32,
        };

        let out_shape = Shape::from(&[params.b_size, params.c_out, h_out, w_out]);
        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 3)?;
        let dst = self.device.zeros_impl(&out_shape, self.dtype)?;

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, kernel.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            Self::dispatch_1d(elem_count),
        )?;

        Ok(dst)
    }

    // =========================================================================
    // index_select — shader: index_select.comp
    // Push constants: { uint total_out_elems, rank, base, selected_dim, uvec4 input_strides, output_strides }
    // Buffers: 3 (input, indices, output)
    // =========================================================================

    fn index_select(&self, ids: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        if super::debug::force_cpu_index() {
            let cpu_src = self.to_cpu_storage()?;
            let cpu_ids = ids.to_cpu_storage()?;
            let cpu_result = cpu_src.index_select(&cpu_ids, src_l, ids_l, dim)?;
            return self.device.storage_from_cpu_storage(&cpu_result);
        }
        let idx_suffix = ids.dtype_suffix()?;
        let val_suffix = self.dtype_suffix()?;
        let key = format!("index_select_{}_{}", idx_suffix, val_suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            total_out_elems: u32,
            rank: u32,
            base: u32,
            selected_dim: u32,
            input_strides: [u32; 4],
            output_strides: [u32; 4],
        }

        let src_shape = src_l.shape();
        let rank = src_shape.rank();
        let ids_count = ids_l.shape().elem_count();

        // Build output shape: replace dim with ids count
        let mut out_dims: Vec<usize> = src_shape.dims().to_vec();
        out_dims[dim] = ids_count;
        let out_shape = Shape::from(out_dims);
        let total_out = out_shape.elem_count();

        // Input strides from layout
        let src_stride = src_l.stride();
        let mut input_strides = [0u32; 4];
        for i in 0..rank.min(4) {
            input_strides[i] = src_stride[i] as u32;
        }

        // Output strides: contiguous row-major
        let mut output_strides = [0u32; 4];
        if rank > 0 {
            output_strides[rank - 1] = 1;
            for i in (0..rank - 1).rev() {
                output_strides[i] =
                    output_strides[i + 1] * out_shape.dim(i + 1).unwrap_or(1) as u32;
            }
        }

        let pc = PC {
            total_out_elems: total_out as u32,
            rank: rank as u32,
            base: src_l.start_offset() as u32,
            selected_dim: dim as u32,
            input_strides,
            output_strides,
        };

        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 3)?;
        let dst = unsafe { self.device.alloc_uninit(&out_shape, self.dtype)? };

        let dispatch_x = (total_out as u32 + 255) / 256;
        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, ids.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            [dispatch_x, 1, 1],
        )?;

        Ok(dst)
    }

    fn gather(&self, src_l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        if super::debug::force_cpu_index() {
            let cpu_src = self.to_cpu_storage()?;
            let cpu_ids = ids.to_cpu_storage()?;
            let cpu_result = cpu_src.gather(src_l, &cpu_ids, ids_l, dim)?;
            return self.device.storage_from_cpu_storage(&cpu_result);
        }
        let idx_suffix = ids.dtype_suffix()?;
        let val_suffix = self.dtype_suffix()?;
        let key = format!("gather_{}_{}", idx_suffix, val_suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            total_out_elems: u32,
            base: u32,
            rank: u32,
            selected_dim: u32,
            input_strides: [u32; 4],
            output_strides: [u32; 4],
        }

        let ids_shape = ids_l.shape();
        let total_out = ids_shape.elem_count();
        let rank = src_l.shape().rank();

        let src_stride = src_l.stride();
        let mut input_strides = [0u32; 4];
        for i in 0..rank.min(4) {
            input_strides[i] = src_stride[i] as u32;
        }

        // Output strides: contiguous over ids shape
        let mut output_strides = [0u32; 4];
        if rank > 0 {
            output_strides[rank - 1] = 1;
            for i in (0..rank - 1).rev() {
                output_strides[i] =
                    output_strides[i + 1] * ids_shape.dim(i + 1).unwrap_or(1) as u32;
            }
        }

        let pc = PC {
            total_out_elems: total_out as u32,
            base: src_l.start_offset() as u32,
            rank: rank as u32,
            selected_dim: dim as u32,
            input_strides,
            output_strides,
        };

        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 3)?;
        let dst = unsafe { self.device.alloc_uninit(ids_shape, self.dtype)? };

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, ids.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            Self::dispatch_1d(total_out),
        )?;

        Ok(dst)
    }

    fn scatter_set(
        &mut self,
        layout: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        // CPU fallback - scatter requires read-modify-write semantics
        let mut cpu_self = self.to_cpu_storage()?;
        let cpu_ids = ids.to_cpu_storage()?;
        let cpu_src = src.to_cpu_storage()?;
        cpu_self.scatter_set(layout, &cpu_ids, ids_l, &cpu_src, src_l, dim)?;
        let new_storage = self.device.storage_from_cpu_storage(&cpu_self)?;
        *self = new_storage;
        Ok(())
    }

    fn scatter_add_set(
        &mut self,
        layout: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        // CPU fallback - scatter_add requires atomics
        let mut cpu_self = self.to_cpu_storage()?;
        let cpu_ids = ids.to_cpu_storage()?;
        let cpu_src = src.to_cpu_storage()?;
        cpu_self.scatter_add_set(layout, &cpu_ids, ids_l, &cpu_src, src_l, dim)?;
        let new_storage = self.device.storage_from_cpu_storage(&cpu_self)?;
        *self = new_storage;
        Ok(())
    }

    fn index_add(
        &self,
        layout: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        // CPU fallback - index_add requires atomics
        let cpu_self = self.to_cpu_storage()?;
        let cpu_ids = ids.to_cpu_storage()?;
        let cpu_src = src.to_cpu_storage()?;
        let cpu_result = cpu_self.index_add(layout, &cpu_ids, ids_l, &cpu_src, src_l, dim)?;
        self.device.storage_from_cpu_storage(&cpu_result)
    }

    // =========================================================================
    // matmul — shader: gemm.comp
    // Push constants: { uint m, n, k, a_stride_bh, a_stride_m, a_stride_k,
    //                    b_stride_bh, b_stride_k, b_stride_n, a_base, b_base,
    //                    ldc, c_stride_bh, float alpha, beta }
    // Buffers: 3 (A, B, C)
    // =========================================================================

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if super::debug::force_cpu_gemm() {
            let cpu_lhs = self.to_cpu_storage()?;
            let cpu_rhs = rhs.to_cpu_storage()?;
            let cpu_result = cpu_lhs.matmul(&cpu_rhs, (b, m, n, k), lhs_l, rhs_l)?;
            return self.device.storage_from_cpu_storage(&cpu_result);
        }

        let suffix = self.dtype_suffix()?;

        // Check if cooperative matrix is available and not disabled
        let has_coopmat = self.device.has_cooperative_matrix();
        let use_coopmat = has_coopmat && !self.is_gemm_coopmat_disabled(&suffix);

        let use_f16acc = suffix == "f16"
            && !use_coopmat
            && self.device.has_fp16_compute()
            && std::env::var("PARAMECIA_DISABLE_F16ACC").is_err();

        let key = if use_coopmat {
            format!("gemm_{}_coopmat", suffix)
        } else if use_f16acc {
            "gemm_f16_f16acc".to_string()
        } else {
            format!("gemm_{}", suffix)
        };

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            m: u32,
            n: u32,
            k: u32,
            a_stride_bh: u32,
            a_stride_m: u32,
            a_stride_k: u32,
            b_stride_bh: u32,
            b_stride_k: u32,
            b_stride_n: u32,
            a_base: u32,
            b_base: u32,
            ldc: u32,
            c_stride_bh: u32,
            alpha: f32,
            beta: f32,
            k_split: u32, // 0 = no split, >0 = K-slice size for split-K
        }

        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();
        let lhs_rank = lhs_stride.len();
        let rhs_rank = rhs_stride.len();

        let pc = PC {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            a_stride_bh: if b > 1 && lhs_rank > 2 {
                lhs_stride[lhs_rank - 3] as u32
            } else {
                0
            },
            a_stride_m: lhs_stride[lhs_rank - 2] as u32,
            a_stride_k: lhs_stride[lhs_rank - 1] as u32,
            b_stride_bh: if b > 1 && rhs_rank > 2 {
                rhs_stride[rhs_rank - 3] as u32
            } else {
                0
            },
            b_stride_k: rhs_stride[rhs_rank - 2] as u32,
            b_stride_n: rhs_stride[rhs_rank - 1] as u32,
            a_base: lhs_l.start_offset() as u32,
            b_base: rhs_l.start_offset() as u32,
            ldc: n as u32,
            c_stride_bh: (m * n) as u32,
            alpha: 1.0,
            beta: 0.0,
            k_split: 0,
        };

        // Get tile sizes and specialization constants for coopmat
        let ((tile_m, tile_n), specialization_constants) = if use_coopmat {
            let (tile_m, tile_n, tile_k) =
                self.device.coop_matrix_tile_size().unwrap_or((16, 16, 16));
            ((tile_m, tile_n), Some(vec![tile_m, tile_n, tile_k]))
        } else {
            ((16, 16), None)
        };

        // Split-K heuristic: if GPU is underutilized, split K across workgroups
        let core_count = self.device.shader_core_count();
        let m_tiles = ((m as u32) + tile_m - 1) / tile_m;
        let n_tiles = ((n as u32) + tile_n - 1) / tile_n;
        let total_tiles = m_tiles * n_tiles;
        let split_k_disabled = std::env::var("PARAMECIA_DISABLE_SPLIT_K").is_ok();
        let split_k = if !split_k_disabled
            && !use_coopmat
            && !use_f16acc
            && core_count > 0
            && k >= 2048
            && total_tiles <= core_count / 2
        {
            let sk = (core_count / total_tiles).max(1).min(8);
            // Ensure split doesn't create empty slices
            let k_per_split = ((k as u32 + sk - 1) / sk + 255) & !255; // align to 256
            let effective_sk = ((k as u32) + k_per_split - 1) / k_per_split;
            effective_sk
        } else {
            1
        };

        // For split-K with non-f32 types, use the _splitk variant that writes f32 partials
        let gemm_key = if split_k > 1 && self.dtype != DType::F32 {
            format!("gemm_{}_splitk", suffix)
        } else {
            key.clone()
        };

        // Try to load the coopmat pipeline, fall back to standard on failure
        let pipeline = if use_coopmat {
            match self
                .device
                .kernels()
                .load_pipeline(
                    self.device.device(),
                    &gemm_key,
                    None,
                    std::mem::size_of::<PC>() as u32,
                    3,
                    specialization_constants.as_ref().map(|v| v.as_slice()),
                    false,
                )
                .map_err(|e| crate::Error::Vulkan(VulkanError::from(e)))
            {
                Ok(p) => p,
                Err(_) => {
                    // Fall back to standard GEMM
                    let fallback_key = format!("gemm_{}", suffix);
                    self.load_pipeline(&fallback_key, std::mem::size_of::<PC>() as u32, 3)?
                }
            }
        } else {
            self.load_pipeline(&gemm_key, std::mem::size_of::<PC>() as u32, 3)?
        };

        let shape = Shape::from(&[b, m, n]);

        if split_k > 1 {
            // Split-K mode: dispatch GEMM writing partials, then reduce
            let k_per_split = ((k as u32 + split_k - 1) / split_k + 255) & !255;
            let mut pc_split = pc;
            pc_split.k_split = k_per_split;

            // Acquire partials buffer from scratch pool: split_k * batch * m * n floats
            let partials_bytes = ((split_k as usize) * b * m * n * 4) as u64;
            let partials_buf = self.device.acquire_scratch_buffer(partials_bytes)?;
            let partials_vk = partials_buf.buffer;

            // Dispatch GEMM: y-dimension packs m_tiles * split_k
            let dispatch_x = n_tiles;
            let dispatch_y = m_tiles * split_k;
            let dispatch_z = b as u32;

            self.execute_compute(
                &pipeline,
                &[self.vk_buffer()?, rhs.vk_buffer()?, partials_vk],
                &pc_split,
                [dispatch_x, dispatch_y, dispatch_z],
            )?;

            // Dispatch reduce kernel (typed to match output dtype)
            let dst = self.device.zeros_impl(&shape, self.dtype)?;
            let total_elems = (b * m * n) as u32;
            let reduce_pc: [u32; 2] = [total_elems, split_k];
            let reduce_name = if self.dtype == DType::F32 {
                "matmul_split_k_reduce".to_string()
            } else {
                format!("matmul_split_k_reduce_{}", suffix)
            };
            let reduce_pipeline = self
                .device
                .kernels()
                .load_pipeline(
                    self.device.device(),
                    &reduce_name,
                    None,
                    8, // 2 * u32
                    2, // partials + output
                    None,
                    false,
                )
                .map_err(|e| crate::Error::Vulkan(VulkanError::from(e)))?;

            self.execute_compute(
                &reduce_pipeline,
                &[partials_vk, dst.vk_buffer()?],
                &reduce_pc,
                [(total_elems + 255) / 256, 1, 1],
            )?;

            // Return scratch buffer to pool for reuse
            self.device.release_scratch_buffer(partials_buf);

            Ok(dst)
        } else {
            // Normal single-pass GEMM
            let dst = self.device.zeros_impl(&shape, self.dtype)?;

            let dispatch_x = n_tiles;
            let dispatch_y = m_tiles;
            let dispatch_z = b as u32;

            self.execute_compute(
                &pipeline,
                &[self.vk_buffer()?, rhs.vk_buffer()?, dst.vk_buffer()?],
                &pc,
                [dispatch_x, dispatch_y, dispatch_z],
            )?;

            Ok(dst)
        }
    }

    // =========================================================================
    // copy_strided_src — shader: copy_strided_src.comp
    // Push constants: { uint base, rank, _pad[2], uvec4 shape, uvec4 stride, uint dst_offset }
    // Buffers: 2 (src, dst)
    // =========================================================================

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, layout: &Layout) -> Result<()> {
        if super::debug::force_cpu_copy() {
            let cpu_src = self.to_cpu_storage()?;
            let mut cpu_dst = dst.to_cpu_storage()?;
            cpu_src.copy_strided_src(&mut cpu_dst, dst_offset, layout)?;
            let new_dst = self.device.storage_from_cpu_storage(&cpu_dst)?;
            *dst = new_dst;
            return Ok(());
        }

        // Validation: compare GPU result with CPU result
        if super::debug::trace_ops() {
            use std::sync::atomic::{AtomicU32, Ordering};
            static COPY_COUNT: AtomicU32 = AtomicU32::new(0);
            let n = COPY_COUNT.fetch_add(1, Ordering::Relaxed);
            trace!(
                "[vk copy_strided #{}] dtype={:?} shape={:?} stride={:?} off={} dst_off={} src_buf={} dst_buf={}",
                n,
                self.dtype,
                layout.shape().dims(),
                layout.stride(),
                layout.start_offset(),
                dst_offset,
                self.count,
                dst.count
            );

            // Do CPU reference
            let cpu_src = self.to_cpu_storage()?;
            let mut cpu_dst = dst.to_cpu_storage()?;
            cpu_src.copy_strided_src(&mut cpu_dst, dst_offset, layout)?;

            // Do GPU
            self.copy_strided_src_gpu(dst, dst_offset, layout)?;

            // Compare
            let gpu_dst = dst.to_cpu_storage()?;
            let total = layout.shape().elem_count();
            match (&cpu_dst, &gpu_dst) {
                (CpuStorage::F32(cpu), CpuStorage::F32(gpu)) => {
                    let mut max_diff = 0.0f32;
                    let mut first_bad = None;
                    for i in dst_offset..(dst_offset + total).min(cpu.len()).min(gpu.len()) {
                        let d = (cpu[i] - gpu[i]).abs();
                        if d > max_diff {
                            max_diff = d;
                        }
                        if d > 1e-6 && first_bad.is_none() {
                            first_bad = Some((i, cpu[i], gpu[i]));
                        }
                    }
                    if let Some((i, c, g)) = first_bad {
                        warn!(
                            "[vk copy_strided #{}] MISMATCH! first_bad[{}]: cpu={} gpu={} max_diff={}",
                            n, i, c, g, max_diff
                        );
                    } else {
                        trace!("[vk copy_strided #{}] OK max_diff={}", n, max_diff);
                    }
                }
                (CpuStorage::BF16(cpu), CpuStorage::BF16(gpu)) => {
                    let mut first_bad = None;
                    for i in dst_offset..(dst_offset + total).min(cpu.len()).min(gpu.len()) {
                        if cpu[i] != gpu[i] && first_bad.is_none() {
                            first_bad = Some((i, cpu[i], gpu[i]));
                        }
                    }
                    if let Some((i, c, g)) = first_bad {
                        warn!(
                            "[vk copy_strided #{}] BF16 MISMATCH! first_bad[{}]: cpu={:?} gpu={:?}",
                            n, i, c, g
                        );
                    } else {
                        trace!("[vk copy_strided #{}] BF16 OK", n);
                    }
                }
                _ => {
                    trace!(
                        "[vk copy_strided #{}] (skip validation for {:?})",
                        n,
                        self.dtype
                    );
                }
            }
            // Use CPU result (known correct) so validation doesn't corrupt model
            let new_dst = self.device.storage_from_cpu_storage(&cpu_dst)?;
            *dst = new_dst;
            return Ok(());
        }

        self.copy_strided_src_gpu(dst, dst_offset, layout)
    }

    // =========================================================================
    // copy2d — shader: copy2d.comp
    // Push constants: { uint src_offset, dst_offset, rows, cols, src_stride, dst_stride }
    // Buffers: 2 (src, dst)
    // =========================================================================

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        if super::debug::force_cpu_copy() {
            let cpu_src = self.to_cpu_storage()?;
            let mut cpu_dst = dst.to_cpu_storage()?;
            cpu_src.copy2d(
                &mut cpu_dst,
                d1,
                d2,
                src_stride1,
                dst_stride1,
                src_offset,
                dst_offset,
            )?;
            let new_dst = self.device.storage_from_cpu_storage(&cpu_dst)?;
            *dst = new_dst;
            return Ok(());
        }

        if super::debug::trace_ops() {
            use std::sync::atomic::{AtomicU32, Ordering};
            static COPY2D_COUNT: AtomicU32 = AtomicU32::new(0);
            let n = COPY2D_COUNT.fetch_add(1, Ordering::Relaxed);
            trace!(
                "[vk copy2d #{}] dtype={:?} rows={} cols={} src_stride={} dst_stride={} src_off={} dst_off={} src_buf={} dst_buf={}",
                n,
                self.dtype,
                d1,
                d2,
                src_stride1,
                dst_stride1,
                src_offset,
                dst_offset,
                self.count,
                dst.count
            );

            // CPU reference
            let cpu_src = self.to_cpu_storage()?;
            let mut cpu_dst = dst.to_cpu_storage()?;
            cpu_src.copy2d(
                &mut cpu_dst,
                d1,
                d2,
                src_stride1,
                dst_stride1,
                src_offset,
                dst_offset,
            )?;

            // GPU
            self.copy2d_gpu(
                dst,
                d1,
                d2,
                src_stride1,
                dst_stride1,
                src_offset,
                dst_offset,
            )?;

            // Compare
            let gpu_dst = dst.to_cpu_storage()?;
            let total = d1 * d2;
            match (&cpu_dst, &gpu_dst) {
                (CpuStorage::F32(cpu), CpuStorage::F32(gpu)) => {
                    let mut max_diff = 0.0f32;
                    let mut first_bad = None;
                    for row in 0..d1 {
                        for col in 0..d2 {
                            let i = dst_offset + row * dst_stride1 + col;
                            if i < cpu.len() && i < gpu.len() {
                                let d = (cpu[i] - gpu[i]).abs();
                                if d > max_diff {
                                    max_diff = d;
                                }
                                if d > 1e-6 && first_bad.is_none() {
                                    first_bad = Some((row, col, i, cpu[i], gpu[i]));
                                }
                            }
                        }
                    }
                    if let Some((r, c, i, cv, gv)) = first_bad {
                        warn!(
                            "[vk copy2d #{}] MISMATCH! [{},{}] idx={}: cpu={} gpu={} max_diff={}",
                            n, r, c, i, cv, gv, max_diff
                        );
                    } else {
                        trace!(
                            "[vk copy2d #{}] OK max_diff={} ({} elems)",
                            n,
                            max_diff,
                            total
                        );
                    }
                }
                (CpuStorage::BF16(cpu), CpuStorage::BF16(gpu)) => {
                    let mut first_bad = None;
                    for row in 0..d1 {
                        for col in 0..d2 {
                            let i = dst_offset + row * dst_stride1 + col;
                            if i < cpu.len()
                                && i < gpu.len()
                                && cpu[i] != gpu[i]
                                && first_bad.is_none()
                            {
                                first_bad = Some((row, col, i, cpu[i], gpu[i]));
                            }
                        }
                    }
                    if let Some((r, c, i, cv, gv)) = first_bad {
                        warn!(
                            "[vk copy2d #{}] BF16 MISMATCH! [{},{}] idx={}: cpu={:?} gpu={:?}",
                            n, r, c, i, cv, gv
                        );
                    } else {
                        trace!("[vk copy2d #{}] BF16 OK ({} elems)", n, total);
                    }
                }
                _ => trace!("[vk copy2d #{}] (skip validation for {:?})", n, self.dtype),
            }
            let new_dst = self.device.storage_from_cpu_storage(&cpu_dst)?;
            *dst = new_dst;
            return Ok(());
        }

        self.copy2d_gpu(
            dst,
            d1,
            d2,
            src_stride1,
            dst_stride1,
            src_offset,
            dst_offset,
        )
    }

    // =========================================================================
    // const_set — shader: const_set.comp
    // Push constants: { uint rank, base, shape[MAX_RANK], stride[MAX_RANK], lo, hi, count }
    // Buffers: 1 (target)
    // =========================================================================

    fn const_set(&mut self, scalar: Scalar, layout: &Layout) -> Result<()> {
        if super::debug::force_cpu_const() {
            let mut cpu = self.to_cpu_storage()?;
            cpu.const_set(scalar, layout)?;
            let new_self = self.device.storage_from_cpu_storage(&cpu)?;
            *self = new_self;
            return Ok(());
        }
        let suffix = match self.dtype {
            DType::U8 => 8,
            DType::F16 | DType::BF16 => 16,
            DType::U32 | DType::F32 => 32,
            DType::I64 | DType::F64 => 0,
            _ => crate::bail!("Vulkan const_set unsupported dtype {:?}", self.dtype),
        };
        let key = format!("const_set_{}", suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            rank: u32,
            base: u32,
            shape: [u32; MAX_RANK],
            stride: [u32; MAX_RANK],
            lo: u32,
            hi: u32,
            count: u32,
            _pad: u32,
        }

        let (lo, hi) = match self.dtype {
            DType::U8 => (scalar.to_f64() as u8 as u32, 0u32),
            DType::F16 => (f16::from_f64(scalar.to_f64()).to_bits() as u32, 0u32),
            DType::BF16 => (bf16::from_f64(scalar.to_f64()).to_bits() as u32, 0u32),
            DType::U32 => (scalar.to_f64() as u32, 0u32),
            DType::F32 => (f32::to_bits(scalar.to_f64() as f32), 0u32),
            DType::I64 => {
                let v = scalar.to_f64() as i64 as u64;
                (v as u32, (v >> 32) as u32)
            }
            DType::F64 => {
                let v = scalar.to_f64().to_bits();
                (v as u32, (v >> 32) as u32)
            }
            _ => crate::bail!("Vulkan const_set unsupported dtype {:?}", self.dtype),
        };

        let (shape, stride, base, rank) = Self::layout_to_arrays(layout);
        let count = layout.shape().elem_count() as u32;

        let pc = PC {
            rank,
            base,
            shape,
            stride,
            lo,
            hi,
            count,
            _pad: 0,
        };

        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 1)?;

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?],
            &pc,
            Self::dispatch_1d(count as usize),
        )?;

        Ok(())
    }

    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.pool2d_impl(layout, kernel_size, stride, "avg")
    }

    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.pool2d_impl(layout, kernel_size, stride, "max")
    }

    fn upsample_nearest1d(&self, layout: &Layout, sz: usize) -> Result<Self> {
        let suffix = self.dtype_suffix()?;
        let key = format!("upsample_nearest1d_{}", suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            in_base: u32,
            in_rank: u32,
            _pad_in: [u32; 2],
            in_shape: [u32; 4],
            in_stride: [u32; 4],
            b_size: u32,
            c_size: u32,
            l_out: u32,
            scale_l: u32,
            total_out_elems: u32,
        }

        let dims = layout.shape().dims();
        let (b, c, l_in) = (dims[0], dims[1], dims[2]);
        let scale_l = if l_in > 0 { sz / l_in } else { 1 };
        let total = b * c * sz;

        let (in_shape, in_stride, in_base, in_rank) = Self::layout_to_uvec4(layout);
        let pc = PC {
            in_base,
            in_rank,
            _pad_in: [0; 2],
            in_shape,
            in_stride,
            b_size: b as u32,
            c_size: c as u32,
            l_out: sz as u32,
            scale_l: scale_l as u32,
            total_out_elems: total as u32,
        };

        let out_shape = Shape::from(&[b, c, sz]);
        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 2)?;
        let dst = unsafe { self.device.alloc_uninit(&out_shape, self.dtype)? };

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            Self::dispatch_1d(total),
        )?;

        Ok(dst)
    }

    fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self> {
        let suffix = self.dtype_suffix()?;
        let key = format!("upsample_nearest2d_{}", suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            in_base: u32,
            in_rank: u32,
            _pad_in: [u32; 2],
            in_shape: [u32; 4],
            in_stride: [u32; 4],
            b_size: u32,
            c_size: u32,
            h_out: u32,
            w_out: u32,
            h_in: u32,
            w_in: u32,
            total_out_elems: u32,
        }

        let dims = layout.shape().dims();
        let (b, c, h_in, w_in) = (dims[0], dims[1], dims[2], dims[3]);
        let total = b * c * h * w;

        let (in_shape, in_stride, in_base, in_rank) = Self::layout_to_uvec4(layout);
        let pc = PC {
            in_base,
            in_rank,
            _pad_in: [0; 2],
            in_shape,
            in_stride,
            b_size: b as u32,
            c_size: c as u32,
            h_out: h as u32,
            w_out: w as u32,
            h_in: h_in as u32,
            w_in: w_in as u32,
            total_out_elems: total as u32,
        };

        let out_shape = Shape::from(&[b, c, h, w]);
        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 2)?;
        let dst = unsafe { self.device.alloc_uninit(&out_shape, self.dtype)? };

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            Self::dispatch_1d(total),
        )?;

        Ok(dst)
    }

    fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        h: usize,
        w: usize,
        _align: bool,
        _scale_h: Option<f64>,
        _scale_w: Option<f64>,
    ) -> Result<Self> {
        // CPU fallback - no bilinear shader available
        let cpu_self = self.to_cpu_storage()?;
        let cpu_result = cpu_self.upsample_bilinear2d(layout, h, w, _align, _scale_h, _scale_w)?;
        self.device.storage_from_cpu_storage(&cpu_result)
    }
}

impl VulkanStorage {
    fn pool2d_impl(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        mode: &str,
    ) -> Result<Self> {
        let suffix = self.dtype_suffix()?;
        let key = format!("pool2d_{}_{}", mode, suffix);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            in_base: u32,
            in_rank: u32,
            in_shape: [u32; MAX_RANK],
            in_stride: [u32; MAX_RANK],
            out_rank: u32,
            out_shape: [u32; MAX_RANK],
            out_stride: [u32; MAX_RANK],
            k_h: u32,
            k_w: u32,
            s_h: u32,
            s_w: u32,
            h_in: u32,
            w_in: u32,
            h_out: u32,
            w_out: u32,
            total_out_elems: u32,
        }

        let dims = layout.shape().dims();
        let (b, c, h_in, w_in) = (dims[0], dims[1], dims[2], dims[3]);
        let h_out = (h_in - kernel_size.0) / stride.0 + 1;
        let w_out = (w_in - kernel_size.1) / stride.1 + 1;
        let total = b * c * h_out * w_out;

        let (in_shape, in_stride, in_base, in_rank) = Self::layout_to_arrays(layout);

        let out_shape_dims = Shape::from(&[b, c, h_out, w_out]);
        let out_layout = Layout::contiguous(&out_shape_dims);
        let (out_shape_arr, out_stride_arr, _, out_rank) = Self::layout_to_arrays(&out_layout);

        let pc = PC {
            in_base,
            in_rank,
            in_shape,
            in_stride,
            out_rank,
            out_shape: out_shape_arr,
            out_stride: out_stride_arr,
            k_h: kernel_size.0 as u32,
            k_w: kernel_size.1 as u32,
            s_h: stride.0 as u32,
            s_w: stride.1 as u32,
            h_in: h_in as u32,
            w_in: w_in as u32,
            h_out: h_out as u32,
            w_out: w_out as u32,
            total_out_elems: total as u32,
        };

        let pipeline = self.load_pipeline(&key, std::mem::size_of::<PC>() as u32, 2)?;
        let dst = unsafe { self.device.alloc_uninit(&out_shape_dims, self.dtype)? };

        self.execute_compute(
            &pipeline,
            &[self.vk_buffer()?, dst.vk_buffer()?],
            &pc,
            Self::dispatch_1d(total),
        )?;

        Ok(dst)
    }
}
