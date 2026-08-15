use super::{GgmlDType, QStorage};
use crate::quantized::k_quants::GgmlType;
use crate::{backend::BackendDevice, backend::BackendStorage, cuda_backend::WrapErr};
use crate::{builder_arg as barg, CudaDevice, CudaStorage, Result};
use half::f16;
use tracing::debug;

use cudarc::driver::{CudaSlice, CudaView, PushKernelArg};

#[derive(Clone, Debug)]
struct PaddedCudaSlice {
    inner: CudaSlice<u8>,
    len: usize,
}

#[derive(Clone, Debug)]
pub struct QCudaStorage {
    data: PaddedCudaSlice,
    dtype: GgmlDType,
    device: CudaDevice,
}

static FORCE_DMMV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_force_dmmv(f: bool) {
    FORCE_DMMV.store(f, std::sync::atomic::Ordering::Relaxed)
}

/// Check if the MMA tensor core kernels are available.
/// MMA m16n8k16 requires sm_80+ (Ampere). Uses build-time CUDA_COMPUTE_CAP env var
/// with a runtime env var override (PARAMECIA_DISABLE_MMA=1 to force off).
pub fn has_mma_support() -> bool {
    use std::sync::OnceLock;
    static MMA_OK: OnceLock<bool> = OnceLock::new();
    *MMA_OK.get_or_init(|| {
        if std::env::var("PARAMECIA_DISABLE_MMA").as_deref() == Ok("1") {
            return false;
        }
        // The MMA kernels are always compiled at sm_80+ in build.rs.
        // If CUDA_COMPUTE_CAP was set < 80 at build time, they still
        // compiled at sm_80. The binary will fail at runtime on pre-Ampere
        // GPUs, so we conservatively check the build env var.
        match std::env::var("CUDA_COMPUTE_CAP") {
            Ok(cap) => cap.parse::<u32>().unwrap_or(80) >= 80,
            Err(_) => true, // default build targets sm_80+
        }
    })
}

pub const WARP_SIZE: usize = 32;
pub const MMQ_X_Q4_0_AMPERE: usize = 4;
pub const MMQ_Y_Q4_0_AMPERE: usize = 32;
pub const NWARPS_Q4_0_AMPERE: usize = 4;
pub const GGML_CUDA_MMV_X: usize = 32;
pub const GGML_CUDA_MMV_Y: usize = 1;
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const CUDA_DEQUANTIZE_BLOCK_SIZE: usize = 256;
pub const MATRIX_ROW_PADDING: usize = 512;

fn ceil_div(p: usize, q: usize) -> usize {
    p.div_ceil(q)
}

fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

fn quantize_q8_1(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);

    let total_rows = ky;
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = kx_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;

    const CHUNK_SIZE: usize = 65535; // gridDim.y limit
    let func = dev.get_or_load_func("quantize_q8_1", &paramecia_cuda::QUANTIZED)?;

    let mut rows_processed = 0;
    while rows_processed < total_rows {
        // --- calculate the number of rows for this chunk ---
        let remaining_rows = total_rows - rows_processed;
        // This is our gridDim.y, now <= 65535
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        // --- slice the source (f32) tensor by elements ---
        let src_start_elem = rows_processed * k;
        let src_num_elems = rows_in_chunk * k;
        let src_chunk = src.slice(src_start_elem..(src_start_elem + src_num_elems));

        // --- slice the destination (u8) tensor by bytes ---
        let dst_start_byte = rows_processed * dst_row_size_bytes;
        let dst_num_bytes = rows_in_chunk * dst_row_size_bytes;
        let dst_chunk = dst.slice(dst_start_byte..(dst_start_byte + dst_num_bytes));

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, rows_in_chunk as u32, 1),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&src_chunk);
        builder.arg(&dst_chunk);
        barg!(builder, k as i32, kx_padded as i32);
        unsafe { builder.launch(cfg) }.w()?;

        rows_processed += rows_in_chunk;
    }

    Ok(())
}

fn quantize_q8_0_rowwise(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    dst_byte_offset: usize,
    k: usize,
    rows: usize,
    dev: &CudaDevice,
) -> Result<()> {
    if k % GgmlDType::Q8_0.block_size() != 0 {
        crate::bail!(
            "quantize_q8_0_rowwise requires k divisible by {} (got {})",
            GgmlDType::Q8_0.block_size(),
            k
        );
    }

    let blocks_per_row = k / GgmlDType::Q8_0.block_size();
    let func = dev.get_or_load_func("quantize_q8_0_rowwise", &paramecia_cuda::QUANTIZED)?;

    const CHUNK_SIZE: usize = 65535; // gridDim.y limit
    let mut rows_processed = 0;
    while rows_processed < rows {
        let remaining_rows = rows - rows_processed;
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        let src_start_elem = rows_processed * k;
        let src_num_elems = rows_in_chunk * k;
        let src_chunk = src.slice(src_start_elem..(src_start_elem + src_num_elems));

        let dst_row_size_bytes = blocks_per_row * GgmlDType::Q8_0.type_size();
        let dst_start_byte = dst_byte_offset + rows_processed * dst_row_size_bytes;
        let dst_num_bytes = rows_in_chunk * dst_row_size_bytes;
        let dst_chunk = dst.slice(dst_start_byte..(dst_start_byte + dst_num_bytes));

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (blocks_per_row as u32, rows_in_chunk as u32, 1),
            block_dim: (GgmlDType::Q8_0.block_size() as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&src_chunk);
        builder.arg(&dst_chunk);
        barg!(builder, k as i32);
        unsafe { builder.launch(cfg) }.w()?;

        rows_processed += rows_in_chunk;
    }

    Ok(())
}

fn dequantize_f32(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f32", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f32", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f32", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f32", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f32", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f32", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f32", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f32", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f32", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &paramecia_cuda::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_f16(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f16", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f16", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f16", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f16", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f16", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f16", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f16", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f16", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f16", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &paramecia_cuda::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f16>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_mul_mat_vec(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "dequantize_mul_mat_vec_q4_0_cuda",
        GgmlDType::Q4_1 => "dequantize_mul_mat_vec_q4_1_cuda",
        GgmlDType::Q5_0 => "dequantize_mul_mat_vec_q5_0_cuda",
        GgmlDType::Q5_1 => "dequantize_mul_mat_vec_q5_1_cuda",
        GgmlDType::Q8_0 => "dequantize_mul_mat_vec_q8_0_cuda",
        GgmlDType::Q2K => "dequantize_mul_mat_vec_q2_k",
        GgmlDType::Q3K => "dequantize_mul_mat_vec_q3_k",
        GgmlDType::Q4K => "dequantize_mul_mat_vec_q4_k",
        GgmlDType::Q5K => "dequantize_mul_mat_vec_q5_k",
        GgmlDType::Q6K => "dequantize_mul_mat_vec_q6_k",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &paramecia_cuda::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(nrows)? };
    let block_num_y = ceil_div(nrows, GGML_CUDA_MMV_Y);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (block_num_y as u32, 1, 1),
        block_dim: (WARP_SIZE as u32, GGML_CUDA_MMV_Y as u32, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(y);
    builder.arg(&dst);
    barg!(builder, ncols as i32, nrows as i32);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn mul_mat_vec_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols * b_size {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    if b_size == 0 || b_size > 8 {
        crate::bail!("only bsize between 1 and 8 are supported, got {b_size}")
    }
    // Start by quantizing y
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        b_size * ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
    quantize_q8_1(y, &mut y_q8_1, ncols, b_size, dev)?;

    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "mul_mat_vec_q4_0_q8_1_cuda",
        GgmlDType::Q4_1 => "mul_mat_vec_q4_1_q8_1_cuda",
        GgmlDType::Q5_0 => "mul_mat_vec_q5_0_q8_1_cuda",
        GgmlDType::Q5_1 => "mul_mat_vec_q5_1_q8_1_cuda",
        GgmlDType::Q8_0 => "mul_mat_vec_q8_0_q8_1_cuda",
        GgmlDType::Q2K => "mul_mat_vec_q2_K_q8_1_cuda",
        GgmlDType::Q3K => "mul_mat_vec_q3_K_q8_1_cuda",
        GgmlDType::Q4K => "mul_mat_vec_q4_K_q8_1_cuda",
        GgmlDType::Q5K => "mul_mat_vec_q5_K_q8_1_cuda",
        GgmlDType::Q6K => "mul_mat_vec_q6_K_q8_1_cuda",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let kernel_name = format!("{kernel_name}{b_size}");
    let func = dev.get_or_load_func(&kernel_name, &paramecia_cuda::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(nrows * b_size)? };
    // https://github.com/ggerganov/llama.cpp/blob/facb8b56f8fd3bb10a693bf0943ae9d69d0828ef/ggml-cuda/mmvq.cu#L98
    let (nblocks, nwarps) = match b_size {
        1 => (nrows as u32, 1),
        2..=4 => ((nrows as u32).div_ceil(2), 4),
        5..=8 => ((nrows as u32).div_ceil(2), 2),
        _ => crate::bail!("unexpected bsize {b_size}"),
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nblocks, 1, 1),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(&y_q8_1);
    builder.arg(&dst);
    barg!(
        builder,
        /* ncols_x */ ncols as i32,
        /* nrows_x */ nrows as i32,
        /* nrows_y */ ncols_padded as i32,
        /* nrows_dst */ nrows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Fused SwiGLU MMVQ: computes silu(gate_weights @ x) * (up_weights @ x) in one kernel.
/// Both weight matrices must have the same dtype, shape [nrows, ncols].
/// Only supports batch_size=1 (single-token decode).
#[allow(clippy::too_many_arguments)]
fn mul_mat_vec_swiglu_via_q8_1(
    gate_data: &PaddedCudaSlice,
    up_data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let gate_elems = gate_data.len / dtype.type_size() * dtype.block_size();
    if gate_elems < ncols * nrows {
        crate::bail!(
            "unexpected gate data size {}, ncols {ncols} {nrows}",
            gate_elems
        )
    }
    let up_elems = up_data.len / dtype.type_size() * dtype.block_size();
    if up_elems < ncols * nrows {
        crate::bail!(
            "unexpected up data size {}, ncols {ncols} {nrows}",
            up_elems
        )
    }
    if y.len() != ncols {
        crate::bail!("unexpected y size {}, ncols {ncols}", y.len())
    }

    // Quantize y to Q8_1
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let y_size_in_bytes = ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
    quantize_q8_1(y, &mut y_q8_1, ncols, 1, dev)?;

    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "mul_mat_vec_q4_0_q8_1_swiglu",
        GgmlDType::Q4_1 => "mul_mat_vec_q4_1_q8_1_swiglu",
        GgmlDType::Q5_0 => "mul_mat_vec_q5_0_q8_1_swiglu",
        GgmlDType::Q5_1 => "mul_mat_vec_q5_1_q8_1_swiglu",
        GgmlDType::Q8_0 => "mul_mat_vec_q8_0_q8_1_swiglu",
        GgmlDType::Q2K => "mul_mat_vec_q2_K_q8_1_swiglu",
        GgmlDType::Q3K => "mul_mat_vec_q3_K_q8_1_swiglu",
        GgmlDType::Q4K => "mul_mat_vec_q4_K_q8_1_swiglu",
        GgmlDType::Q5K => "mul_mat_vec_q5_K_q8_1_swiglu",
        GgmlDType::Q6K => "mul_mat_vec_q6_K_q8_1_swiglu",
        _ => crate::bail!("unsupported dtype for fused swiglu matmul {dtype:?}"),
    };

    let func = dev.get_or_load_func(kernel_name, &paramecia_cuda::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(nrows)? };

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nrows as u32, 1, 1),
        block_dim: (WARP_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&gate_data.inner);
    builder.arg(&up_data.inner);
    builder.arg(&y_q8_1);
    builder.arg(&dst);
    barg!(
        builder,
        /* ncols_x */ ncols as i32,
        /* nrows_x */ nrows as i32,
        /* nrows_y */ ncols_padded as i32,
        /* nrows_dst */ nrows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

#[allow(clippy::too_many_arguments)]
fn mul_mat_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_rows: usize,
    y_cols: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < x_rows * x_cols {
        crate::bail!("unexpected lhs size {}, {x_rows} {x_cols}", data_elems)
    }
    if y.len() != y_rows * y_cols {
        crate::bail!("unexpected y size {}, {y_rows} {y_cols}", y.len())
    }
    if x_cols != y_rows {
        crate::bail!("unexpected x/y size {x_rows} {x_cols} {y_rows} {y_cols}")
    }
    // For narrow RHS (common in decode / small micro-batches), MMVQ kernels
    // have better memory behavior than tiled MMQ by avoiding broad output tiles.
    if (1..=8).contains(&y_cols) {
        return mul_mat_vec_via_q8_1(data, y, dtype, x_cols, x_rows, y_cols, dev);
    }
    let k = x_cols;
    // Start by quantizing y
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        k_padded * y_cols * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
    quantize_q8_1(y, &mut y_q8_1, k, y_cols, dev)?;

    // MMA tensor core path for supported quant types (sm_80+ / Ampere+)
    // Only beneficial for compute-bound cases (prefill/batch with multiple columns).
    // Single-token decode is memory-bound — DP4A is already optimal there.
    // PARAMECIA_MMA_VERIFY=1 runs both MMA and DP4A, compares output
    // MMA needs enough work to saturate tensor cores. Require enough output
    // elements that we get meaningful block-level parallelism (>= ~64 blocks).
    let mma_blocks = ((x_rows + 63) / 64) * ((y_cols + 63) / 64);
    if has_mma_support()
        && mma_blocks >= 64
        && matches!(dtype, GgmlDType::Q8_0 | GgmlDType::Q4K | GgmlDType::Q6K)
    {
        if std::env::var("PARAMECIA_MMA_VERIFY").as_deref() == Ok("1") {
            return mul_mat_mma_verify(data, &y_q8_1, dtype, x_rows, x_cols, y_cols, k_padded, dev);
        }
        return mul_mat_mma(data, &y_q8_1, dtype, x_rows, x_cols, y_cols, k_padded, dev);
    }

    let (kernel_name, mmq_x, mmq_y) = match dtype {
        GgmlDType::Q4_0 => ("mul_mat_q4_0", 64, 128),
        GgmlDType::Q4_1 => ("mul_mat_q4_1", 64, 128),
        GgmlDType::Q5_0 => ("mul_mat_q5_0", 128, 64),
        GgmlDType::Q5_1 => ("mul_mat_q5_1", 128, 64),
        GgmlDType::Q8_0 => ("mul_mat_q8_0", 128, 64),
        GgmlDType::Q2K => ("mul_mat_q2_K", 64, 128),
        GgmlDType::Q3K => ("mul_mat_q3_K", 128, 128),
        GgmlDType::Q4K => ("mul_mat_q4_K", 64, 128),
        GgmlDType::Q5K => ("mul_mat_q5_K", 64, 128),
        GgmlDType::Q6K => ("mul_mat_q6_K", 64, 64),
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &paramecia_cuda::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(x_rows * y_cols)? };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (
            ceil_div(x_rows, mmq_y) as u32,
            ceil_div(y_cols, mmq_x) as u32,
            1,
        ),
        block_dim: (WARP_SIZE as u32, 4, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(/* vx */ &data.inner);
    builder.arg(/* vy */ &y_q8_1);
    builder.arg(/* dst */ &dst);
    barg!(
        builder,
        /* ncols_x */ x_cols as i32,
        /* nrows_x */ x_rows as i32,
        /* ncols_y */ y_cols as i32,
        /* nrows_y */ k_padded as i32,
        /* nrows_dst */ x_rows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// MMA tensor core matrix multiply dispatch for Q8_0, Q4_K, Q6_K.
/// Called from mul_mat_via_q8_1 when MMA is available.
#[allow(clippy::too_many_arguments)]
fn mul_mat_mma(
    data: &PaddedCudaSlice,
    y_q8_1: &CudaSlice<u8>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_cols: usize,
    k_padded: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    use core::ffi::c_void;
    use cudarc::driver::DevicePtr;

    let stream = dev.cuda_stream();
    let stream_i64 = stream.cu_stream() as i64;

    let (vx_ptr, _vx_sync) = data.inner.device_ptr(stream.as_ref());
    let (vy_ptr, _vy_sync) = y_q8_1.device_ptr(stream.as_ref());

    let dst = unsafe { dev.alloc::<f32>(x_rows * y_cols)? };

    {
        let (dst_ptr, _dst_sync) = dst.device_ptr(stream.as_ref());

        unsafe {
            match dtype {
                GgmlDType::Q8_0 => paramecia_cuda::ffi::mul_mat_q8_0_mma_launch(
                    vx_ptr as *const c_void,
                    vy_ptr as *const c_void,
                    dst_ptr as *mut f32,
                    x_cols as i32,
                    x_rows as i32,
                    y_cols as i32,
                    k_padded as i32,
                    x_rows as i32,
                    stream_i64,
                ),
                GgmlDType::Q4K => paramecia_cuda::ffi::mul_mat_q4_K_mma_launch(
                    vx_ptr as *const c_void,
                    vy_ptr as *const c_void,
                    dst_ptr as *mut f32,
                    x_cols as i32,
                    x_rows as i32,
                    y_cols as i32,
                    k_padded as i32,
                    x_rows as i32,
                    stream_i64,
                ),
                GgmlDType::Q6K => paramecia_cuda::ffi::mul_mat_q6_K_mma_launch(
                    vx_ptr as *const c_void,
                    vy_ptr as *const c_void,
                    dst_ptr as *mut f32,
                    x_cols as i32,
                    x_rows as i32,
                    y_cols as i32,
                    k_padded as i32,
                    x_rows as i32,
                    stream_i64,
                ),
                _ => unreachable!("mul_mat_mma called with unsupported dtype"),
            }
        }
    } // _dst_sync dropped here, releasing borrow on dst

    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Verification mode: run both MMA and DP4A, compare outputs.
#[allow(clippy::too_many_arguments)]
fn mul_mat_mma_verify(
    data: &PaddedCudaSlice,
    y_q8_1: &CudaSlice<u8>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_cols: usize,
    k_padded: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    // Run MMA kernel
    let mma_result = mul_mat_mma(data, y_q8_1, dtype, x_rows, x_cols, y_cols, k_padded, dev)?;

    // Run DP4A kernel
    let (kernel_name, mmq_x, mmq_y) = match dtype {
        GgmlDType::Q8_0 => ("mul_mat_q8_0", 128, 64),
        GgmlDType::Q4K => ("mul_mat_q4_K", 64, 128),
        GgmlDType::Q6K => ("mul_mat_q6_K", 64, 64),
        _ => unreachable!(),
    };
    let func = dev.get_or_load_func(kernel_name, &paramecia_cuda::QUANTIZED)?;
    let dp4a_dst = unsafe { dev.alloc::<f32>(x_rows * y_cols)? };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (
            ceil_div(x_rows, mmq_y) as u32,
            ceil_div(y_cols, mmq_x) as u32,
            1,
        ),
        block_dim: (WARP_SIZE as u32, 4, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(y_q8_1);
    builder.arg(&dp4a_dst);
    barg!(
        builder,
        x_cols as i32,
        x_rows as i32,
        y_cols as i32,
        k_padded as i32,
        x_rows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;

    // Download both results and compare
    let mma_slice = mma_result.as_cuda_slice::<f32>()?;
    let mut mma_host = vec![0f32; x_rows * y_cols];
    let mut dp4a_host = vec![0f32; x_rows * y_cols];
    dev.memcpy_dtoh(mma_slice, &mut mma_host)?;
    dev.memcpy_dtoh(&dp4a_dst, &mut dp4a_host)?;

    let mut max_diff: f32 = 0.0;
    let mut max_diff_idx = 0usize;
    let mut num_big_diffs = 0usize;
    let mut mma_sum: f64 = 0.0;
    let mut dp4a_sum: f64 = 0.0;
    for i in 0..mma_host.len() {
        let diff = (mma_host[i] - dp4a_host[i]).abs();
        mma_sum += mma_host[i] as f64;
        dp4a_sum += dp4a_host[i] as f64;
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
        if diff > 1.0 {
            num_big_diffs += 1;
        }
    }

    let row = max_diff_idx % x_rows; // column-major: index = col * nrows + row
    let col = max_diff_idx / x_rows;
    debug!(
        "MMA VERIFY {:?} [{x_rows}x{y_cols}]: max_diff={max_diff:.4} at ({row},{col}) \
         mma={:.4} dp4a={:.4} | big_diffs={num_big_diffs}/{} | mma_sum={mma_sum:.2} dp4a_sum={dp4a_sum:.2}",
        dtype,
        mma_host[max_diff_idx],
        dp4a_host[max_diff_idx],
        mma_host.len(),
    );
    // Print first few mismatches
    if max_diff > 0.1 {
        let mut printed = 0;
        for i in 0..mma_host.len().min(x_rows * 8) {
            let diff = (mma_host[i] - dp4a_host[i]).abs();
            if diff > 0.1 && printed < 5 {
                let r = i % x_rows;
                let c = i / x_rows;
                debug!(
                    "  diff at ({r},{c}): mma={:.6} dp4a={:.6} diff={diff:.6}",
                    mma_host[i], dp4a_host[i]
                );
                printed += 1;
            }
        }
    }

    // Return DP4A result for correctness (MMA is the one under test)
    Ok(CudaStorage::wrap_cuda_slice(dp4a_dst, dev.clone()))
}

fn indexed_moe_forward_fused_q8_1_input(
    weight: &CudaView<u8>,
    w_shape: &crate::Shape, //[num_experts, n, k]
    w_dtype: GgmlDType,
    input: &CudaSlice<f32>,
    in_shape: &crate::Shape, //[batch, topk or 1, k]
    ids: &CudaView<u32>,
    idx_shape: &crate::Shape, //[batch, topk]
    dev: &CudaDevice,
) -> Result<(CudaStorage, crate::Shape)> {
    let (_, n, k) = w_shape.dims3()?;
    let batch = in_shape.dims()[0];
    let input_dim1 = in_shape.dims()[1];

    let topk = idx_shape.dims()[1];
    assert!(batch == idx_shape.dims()[0], "batch dim not match!");

    // Quantize input into q8_1.
    let total_rows = batch * input_dim1;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = k_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;
    let y_size_in_bytes = total_rows * dst_row_size_bytes;
    let mut input_quant = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };

    let input_view = input.slice(0..);
    quantize_q8_1(&input_view, &mut input_quant, k, total_rows, dev)?;

    // output buffer
    let outsize = batch * topk * n;
    let out = unsafe { dev.alloc::<f32>(outsize)? };

    let kernel_name = match w_dtype {
        GgmlDType::Q2K => "indexed_moe_forward_q2k_q8_1",
        GgmlDType::Q3K => "indexed_moe_forward_q3k_q8_1",
        GgmlDType::Q4K => "indexed_moe_forward_q4k_q8_1",
        GgmlDType::Q5K => "indexed_moe_forward_q5k_q8_1",
        GgmlDType::Q6K => "indexed_moe_forward_q6k_q8_1",
        GgmlDType::Q8_0 => "indexed_moe_forward_q8_0_q8_1",
        _ => crate::bail!("unsupported dtype for indexed_moe_forward {w_dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &paramecia_cuda::QUANTIZED)?;
    let (nblocks, nwarps) = (n as u32, 4);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nblocks, batch as u32, topk as u32),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(weight);
    builder.arg(&input_quant);
    builder.arg(ids);
    builder.arg(&out);

    barg!(
        builder,
        n as i32,
        k as i32,
        batch as i32,
        topk as i32,
        k_padded as i32,
        input_dim1 as i32
    );
    unsafe { builder.launch(cfg) }.w()?;

    let mut out_shape = in_shape.dims().to_vec();
    out_shape.pop();
    out_shape.push(n);
    out_shape[1] = topk;
    Ok((
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        out_shape.into(),
    ))
}

impl QCudaStorage {
    /// Get the device pointer for this storage's data as u64.
    /// Used for per-tensor seeding in fused QuZO operations.
    pub fn device_ptr_u64(&self) -> u64 {
        use cudarc::driver::DevicePtr;
        self.data.inner.device_ptr(self.data.inner.stream()).0 as u64
    }

    /// Modify quantization block scales in-place on GPU.
    /// This is used for QZO fine-tuning where scales are trainable parameters.
    pub fn modify_block_scales(&self, multipliers: &CudaStorage) -> Result<Self> {
        // self.data.len is in bytes, so we divide by type_size (bytes per block)
        // not block_size (elements per block)
        let num_blocks = self.data.len / self.dtype.type_size();

        // Get multipliers as f32 slice
        let multipliers_slice = multipliers.as_cuda_slice::<f32>()?;

        // Handle case where storage might have more elements (use only first num_blocks)
        // This is expected when multipliers come from a narrowed tensor that was made contiguous
        if multipliers_slice.len() < num_blocks {
            crate::bail!(
                "Multipliers storage too small: expected at least {} blocks, got {}",
                num_blocks,
                multipliers_slice.len()
            );
        }

        // Slice to use only the first num_blocks elements that match this QTensor
        let multipliers_slice = multipliers_slice.slice(..num_blocks);

        // Create a copy of the data for modification
        let modified_data_len = self.data.len;
        let mut modified_data = unsafe { self.device.alloc::<u8>(modified_data_len)? };

        // Copy original data
        self.device.memcpy_dtod(
            &self.data.inner.slice(..self.data.len),
            &mut modified_data.slice_mut(..self.data.len),
        )?;

        // Select appropriate kernel
        let kernel_name = match self.dtype {
            GgmlDType::Q4_0 => "modify_scales_q4_0",
            GgmlDType::Q4_1 => "modify_scales_q4_1",
            GgmlDType::Q5_0 => "modify_scales_q5_0",
            GgmlDType::Q5_1 => "modify_scales_q5_1",
            GgmlDType::Q8_0 => "modify_scales_q8_0",
            GgmlDType::Q2K => "modify_scales_q2_K",
            GgmlDType::Q3K => "modify_scales_q3_K",
            GgmlDType::Q4K => "modify_scales_q4_K",
            GgmlDType::Q5K => "modify_scales_q5_K",
            GgmlDType::Q6K => "modify_scales_q6_K",
            GgmlDType::Q8K => "modify_scales_q8_K",
            _ => crate::bail!(
                "modify_block_scales not supported for dtype {:?}",
                self.dtype
            ),
        };

        // Launch kernel to modify scales in-place
        let func = self
            .device
            .get_or_load_func(kernel_name, &paramecia_cuda::QUANTIZED)?;

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (((num_blocks + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&modified_data);
        builder.arg(&multipliers_slice);
        let num_blocks_i32 = num_blocks as i32;
        builder.arg(&num_blocks_i32);
        unsafe { builder.launch(cfg) }.w()?;

        // Return new QCudaStorage with modified data
        Ok(Self {
            data: PaddedCudaSlice {
                inner: modified_data,
                len: self.data.len,
            },
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }
    pub fn indexed_moe_forward(
        &self,
        self_shape: &crate::Shape, //[num_experts, n, k]
        input: &CudaStorage,       //[batch, topk or 1, k]
        input_l: &crate::Layout,
        ids: &CudaStorage, //[batch, topk]
        ids_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;

        if matches!(
            self.dtype(),
            GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        ) {
            let input_dtype = input.dtype();

            // For F16 input, use F16-native kernels that output directly to F16
            if input_dtype == crate::DType::F16 {
                return self.indexed_moe_forward_f16(self_shape, input, input_l, ids, ids_l);
            }

            // Convert to F32 if needed for the kernel
            let input_owned;
            let input_f32 = if input_dtype != crate::DType::F32 {
                input_owned = input.to_dtype(input_l, crate::DType::F32)?;
                &input_owned
            } else {
                input
            };

            let input_storage = input_f32.as_cuda_slice::<f32>()?;
            let (num_experts, n, k) = self_shape.dims3()?;
            let (batch, input_dim1, input_k) = input_l.shape().dims3()?;
            let (ids_batch, topk) = ids_l.shape().dims2()?;
            if input_k != k || ids_batch != batch {
                crate::bail!(
                    "indexed_moe_forward shape mismatch weights={self_shape:?} input={:?} ids={:?}",
                    input_l.shape(),
                    ids_l.shape(),
                )
            }

            // Dense Qwen models are represented as a one-expert MoE. The
            // generic indexed kernel launches an independent matvec for every
            // batch item, causing each item to reread the same expert weights.
            // Route this case through the regular quantized matrix kernels so
            // a block computes several batch columns while loading each
            // weight block only once. MMVQ handles narrow batches and MMQ/MMA
            // handles larger flattened prefill batches. Mid-sized decode
            // batches stay on the indexed kernel, which has lower latency.
            if num_experts == 1
                && input_dim1 == 1
                && topk == 1
                && (batch > 32 || ((1..=8).contains(&batch) && (batch == 1 || n % 2 == 0)))
            {
                let input_view = if input_dtype == crate::DType::F32 {
                    match input_l.contiguous_offsets() {
                        Some((start, end)) => input_storage.slice(start..end),
                        None => {
                            return Err(crate::Error::RequiresContiguous {
                                op: "indexed-moe-dense-mmvq",
                            }
                            .bt())
                        }
                    }
                } else {
                    // `to_dtype` creates compact storage at offset zero; the
                    // original view's offset no longer applies.
                    input_storage.slice(..input_l.shape().elem_count())
                };
                let out = if batch <= 8 {
                    mul_mat_vec_via_q8_1(
                        &self.data,
                        &input_view,
                        self.dtype(),
                        k,
                        n,
                        batch,
                        &self.device,
                    )?
                } else {
                    mul_mat_via_q8_1(
                        &self.data,
                        &input_view,
                        self.dtype(),
                        n,
                        k,
                        k,
                        batch,
                        &self.device,
                    )?
                };
                let out_shape: crate::Shape = (batch, topk, n).into();
                let out = if input_dtype != crate::DType::F32 {
                    let out_layout = crate::Layout::contiguous(&out_shape);
                    out.to_dtype(&out_layout, input_dtype)?
                } else {
                    out
                };
                return Ok((out, out_shape));
            }

            let ids_storage = ids.as_cuda_slice::<u32>()?;
            let (out, out_shape) = indexed_moe_forward_fused_q8_1_input(
                &self.data.inner.slice(0..),
                self_shape, //[num_experts, n, k]
                self.dtype(),
                &input_storage,
                input_l.shape(), //[batch, topk or 1, k]
                &ids_storage.slice(0..),
                ids_l.shape(), //[batch, topk]
                &self.device,
            )?;

            // Convert output back to input dtype if needed
            let out = if input_dtype != crate::DType::F32 {
                let out_layout = crate::Layout::contiguous(&out_shape);
                out.to_dtype(&out_layout, input_dtype)?
            } else {
                out
            };

            Ok((out, out_shape))
        } else {
            crate::bail!(
                "The given quantized dtype {:?} is not supported for indexed_moe_forward!",
                self.dtype()
            );
        }
    }

    /// F16-native indexed MoE forward that stays in F16 throughout
    /// This avoids the F16->F32->F16 conversion overhead
    fn indexed_moe_forward_f16(
        &self,
        self_shape: &crate::Shape,
        input: &CudaStorage,
        input_l: &crate::Layout,
        ids: &CudaStorage,
        ids_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;

        let (_, n, k) = self_shape.dims3()?;
        let batch = input_l.shape().dims()[0];
        let input_dim1 = input_l.shape().dims()[1];
        let topk = ids_l.shape().dims()[1];

        // Convert F16 input to F32 for Q8_1 quantization (required by kernel)
        let input_f32 = input.to_dtype(input_l, crate::DType::F32)?;
        let input_storage = input_f32.as_cuda_slice::<f32>()?;
        let ids_storage = ids.as_cuda_slice::<u32>()?;

        // Quantize input to Q8_1
        let total_rows = batch * input_dim1;
        let k_padded = pad(k, MATRIX_ROW_PADDING);
        let q8_1_type_size = GgmlDType::Q8_1.type_size();
        let q8_1_block_size = GgmlDType::Q8_1.block_size();
        let num_blocks_per_row = k_padded / q8_1_block_size;
        let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;
        let y_size_in_bytes = total_rows * dst_row_size_bytes;
        let mut input_quant = unsafe { self.device.alloc::<u8>(y_size_in_bytes)? };

        quantize_q8_1(
            &input_storage.slice(0..),
            &mut input_quant,
            k,
            total_rows,
            &self.device,
        )?;

        // Allocate F16 output directly
        let outsize = batch * topk * n;
        let out = unsafe { self.device.alloc::<f16>(outsize)? };

        // Select F16 kernel
        let kernel_name = match self.dtype() {
            GgmlDType::Q2K => "indexed_moe_forward_q2k_q8_1_f16",
            GgmlDType::Q3K => "indexed_moe_forward_q3k_q8_1_f16",
            GgmlDType::Q4K => "indexed_moe_forward_q4k_q8_1_f16",
            GgmlDType::Q5K => "indexed_moe_forward_q5k_q8_1_f16",
            GgmlDType::Q6K => "indexed_moe_forward_q6k_q8_1_f16",
            GgmlDType::Q8_0 => "indexed_moe_forward_q8_0_q8_1_f16",
            _ => crate::bail!(
                "unsupported dtype for indexed_moe_forward_f16 {:?}",
                self.dtype()
            ),
        };

        let func = self
            .device
            .get_or_load_func(kernel_name, &paramecia_cuda::QUANTIZED)?;
        let (nblocks, nwarps) = (n as u32, 4);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (nblocks, batch as u32, topk as u32),
            block_dim: (WARP_SIZE as u32, nwarps, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        let weights_slice = self.data.inner.slice(0..);
        let ids_slice = ids_storage.slice(0..);
        builder.arg(&weights_slice);
        builder.arg(&input_quant);
        builder.arg(&ids_slice);
        builder.arg(&out);
        barg!(
            builder,
            n as i32,
            k as i32,
            batch as i32,
            topk as i32,
            k_padded as i32,
            input_dim1 as i32
        );
        unsafe { builder.launch(cfg) }.w()?;

        let mut out_shape = input_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        out_shape[1] = topk;

        Ok((
            CudaStorage::wrap_cuda_slice(out, self.device.clone()),
            out_shape.into(),
        ))
    }

    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let padded_size_in_bytes =
            ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let inner = device.alloc_zeros::<u8>(padded_size_in_bytes)?;
        Ok(QCudaStorage {
            data: PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            },
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<CudaStorage> {
        fn deq<T: GgmlType>(buffer: &[u8], n: usize, dst: &mut [f32]) {
            let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, n) };
            let vec = slice.to_vec();
            T::to_float(&vec, dst)
        }

        let fast_kernel = matches!(
            self.dtype,
            GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
                | GgmlDType::Q8K
        );
        if fast_kernel {
            return dequantize_f32(&self.data, self.dtype, elem_count, self.device());
        }
        // Run the dequantization on cpu.

        let buffer = self
            .device
            .clone_dtoh(&self.data.inner.slice(..self.data.len))?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => deq::<f32>(&buffer, block_len, &mut out),
            GgmlDType::F16 => deq::<half::f16>(&buffer, block_len, &mut out),
            GgmlDType::BF16 => deq::<half::bf16>(&buffer, block_len, &mut out),
            GgmlDType::Q4_0 => deq::<crate::quantized::BlockQ4_0>(&buffer, block_len, &mut out),
            GgmlDType::Q4_1 => deq::<crate::quantized::BlockQ4_1>(&buffer, block_len, &mut out),
            GgmlDType::Q5_0 => deq::<crate::quantized::BlockQ5_0>(&buffer, block_len, &mut out),
            GgmlDType::Q5_1 => deq::<crate::quantized::BlockQ5_1>(&buffer, block_len, &mut out),
            GgmlDType::Q8_0 => deq::<crate::quantized::BlockQ8_0>(&buffer, block_len, &mut out),
            GgmlDType::Q8_1 => deq::<crate::quantized::BlockQ8_1>(&buffer, block_len, &mut out),
            GgmlDType::Q2K => deq::<crate::quantized::BlockQ2K>(&buffer, block_len, &mut out),
            GgmlDType::Q3K => deq::<crate::quantized::BlockQ3K>(&buffer, block_len, &mut out),
            GgmlDType::Q4K => deq::<crate::quantized::BlockQ4K>(&buffer, block_len, &mut out),
            GgmlDType::Q5K => deq::<crate::quantized::BlockQ5K>(&buffer, block_len, &mut out),
            GgmlDType::Q6K => deq::<crate::quantized::BlockQ6K>(&buffer, block_len, &mut out),
            GgmlDType::Q8K => deq::<crate::quantized::BlockQ8K>(&buffer, block_len, &mut out),
        }

        self.device
            .storage_from_cpu_storage(&crate::CpuStorage::F32(out))
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<CudaStorage> {
        // Non-quantized storage dtypes don't have dedicated dequantize-to-f16 kernels.
        // Fall back to f32 dequantization and cast to f16 on device.
        if matches!(
            self.dtype,
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
        ) {
            let deq_f32 = self.dequantize(elem_count)?;
            let layout = crate::Layout::contiguous(&crate::Shape::from(elem_count));
            return deq_f32.to_dtype(&layout, crate::DType::F16);
        }
        dequantize_f16(&self.data, self.dtype, elem_count, self.device())
    }

    pub fn quantize(&mut self, src: &CudaStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(data.as_ref(), &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &CudaStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(data.as_ref(), &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(data.as_ref(), &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(data.as_ref(), &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len
    }

    pub fn slice(&self, offset: usize, size: usize) -> Result<Self> {
        if offset + size > self.data.len {
            crate::bail!(
                "slice range {}..{} exceeds storage size {}",
                offset,
                offset + size,
                self.data.len
            )
        }

        // CUDA kernels require MATRIX_ROW_PADDING at the end of EACH matrix.
        // The source 3D tensor only has padding at the end of the entire buffer.
        // So we must copy each slice into its own padded allocation.
        let padded_len =
            size + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut new_inner = self.device.alloc_zeros::<u8>(padded_len)?;

        // Copy the actual quantized data (without padding)
        let src_slice = self.data.inner.slice(offset..offset + size);
        self.device
            .memcpy_dtod(&src_slice, &mut new_inner.slice_mut(..size))?;

        Ok(Self {
            data: PaddedCudaSlice {
                inner: new_inner,
                len: size,
            },
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        // Non-quantized dtypes (F32, F16, BF16) don't have specialized CUDA
        // dequantize-matmul kernels. Dequantize to a standard tensor and use cuBLAS.
        if matches!(
            self.dtype,
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
        ) {
            return self.fwd_via_dequantize(self_shape, storage, layout);
        }

        let max_bm = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            1
        } else {
            8
        };
        let use_vec_kernel = match layout.shape().dims() {
            [b, m, _k] => b * m <= max_bm,
            [b, _k] => *b <= max_bm,
            _ => false,
        };
        if use_vec_kernel {
            self.dequantize_matmul_vec(self_shape, storage, layout)
        } else {
            self.dequantize_matmul(self_shape, storage, layout)
        }
    }

    /// Forward pass for non-quantized dtypes (F32, F16, BF16).
    /// Dequantizes to a standard tensor and uses cuBLAS matmul.
    fn fwd_via_dequantize(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;

        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }

        let input_dtype = storage.dtype();

        // Dequantize weight data to f32
        let data_f32 = self.dequantize(n * k)?;
        let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;

        // Convert input to f32 if needed. to_dtype produces contiguous output
        // starting at offset 0, so we must use a contiguous layout for matmul.
        let storage_owned;
        let lhs_layout;
        let storage_f32: &CudaStorage = if input_dtype != crate::DType::F32 {
            storage_owned = storage.to_dtype(layout, crate::DType::F32)?;
            lhs_layout = crate::Layout::contiguous(layout.shape());
            &storage_owned
        } else {
            lhs_layout = layout.clone();
            storage
        };
        let out = storage_f32.matmul(&data_f32, (b, m, n, k), &lhs_layout, &rhs_l)?;

        // Build output shape
        let mut out_shape: Vec<usize> = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        let out_shape: crate::Shape = out_shape.into();

        // Convert back to input dtype
        let out = if input_dtype != crate::DType::F32 {
            let out_layout = crate::Layout::contiguous(&out_shape);
            out.to_dtype(&out_layout, input_dtype)?
        } else {
            out
        };

        Ok((out, out_shape))
    }

    /// Fused SwiGLU forward: computes silu(gate @ x) * (up @ x) in one kernel.
    /// Only used for batch_size=1 (single-token decode). Falls back to separate
    /// ops for larger batches or unsupported dtypes.
    pub fn gate_up_swiglu_fwd(
        gate: &Self,
        up: &Self,
        gate_shape: &crate::Shape,
        up_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;

        if gate.dtype != up.dtype {
            crate::bail!(
                "gate_up_swiglu: dtype mismatch gate={:?} up={:?}",
                gate.dtype,
                up.dtype
            )
        }

        // Non-quantized dtypes don't have fused SwiGLU CUDA kernels.
        // Fall back to separate dequantize-matmul ops.
        if matches!(
            gate.dtype,
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
        ) {
            let (gate_out, gate_out_shape) = gate.fwd_via_dequantize(gate_shape, rhs, rhs_l)?;
            let (up_out, up_out_shape) = up.fwd_via_dequantize(up_shape, rhs, rhs_l)?;
            let gate_layout = crate::Layout::contiguous(&gate_out_shape);
            let up_layout = crate::Layout::contiguous(&up_out_shape);
            // silu(gate) * up
            let activated = gate_out.unary_impl::<crate::op::Silu>(&gate_layout)?;
            let out = activated.binary_impl::<crate::op::Mul>(&up_out, &gate_layout, &up_layout)?;
            return Ok((out, gate_out_shape));
        }

        let (gate_nrows, gate_ncols) = gate_shape.dims2()?;
        let (up_nrows, up_ncols) = up_shape.dims2()?;
        if gate_nrows != up_nrows || gate_ncols != up_ncols {
            crate::bail!(
                "gate_up_swiglu: shape mismatch gate={:?} up={:?}",
                gate_shape,
                up_shape
            )
        }
        let nrows = gate_nrows;
        let ncols = gate_ncols;

        let input_dtype = rhs.dtype();

        let rhs_owned;
        let rhs_f32 = if input_dtype != crate::DType::F32 {
            rhs_owned = rhs.to_dtype(rhs_l, crate::DType::F32)?;
            &rhs_owned
        } else {
            rhs
        };

        let rhs_slice = rhs_f32.as_cuda_slice::<f32>()?;
        let rhs_slice = match rhs_l.contiguous_offsets() {
            Some((o1, o2)) => rhs_slice.slice(o1..o2),
            None => {
                return Err(crate::Error::RequiresContiguous {
                    op: "gate_up_swiglu",
                }
                .bt())
            }
        };

        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!("unexpected rhs shape in gate_up_swiglu {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!(
                "mismatch on matmul dim gate={:?} rhs={:?}",
                gate_shape,
                rhs_l.shape()
            )
        }
        if b_size != 1 {
            crate::bail!("gate_up_swiglu only supports batch_size=1, got {b_size}")
        }

        let out = mul_mat_vec_swiglu_via_q8_1(
            &gate.data,
            &up.data,
            &rhs_slice,
            gate.dtype,
            ncols,
            nrows,
            gate.device(),
        )?;

        // Preserve batch dimensions from input, replacing last dim with nrows
        let mut out_dims: Vec<usize> = rhs_l.shape().dims().to_vec();
        out_dims.pop();
        out_dims.push(nrows);
        let out_shape: crate::Shape = out_dims.into();

        let out = if input_dtype != crate::DType::F32 {
            let out_layout = crate::Layout::contiguous(&out_shape);
            out.to_dtype(&out_layout, input_dtype)?
        } else {
            out
        };

        Ok((out, out_shape))
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.data.len];
        self.device
            .memcpy_dtoh(&self.data.inner.slice(..self.data.len), &mut out)?;
        Ok(out)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        use cudarc::driver::DevicePtr;
        Ok(self.data.inner.device_ptr(self.data.inner.stream()).0 as *const u8)
    }

    /// In-place perturbation of quantized weights on GPU.
    /// Implements: w' = Q(w + sign * ε * z) where Q is stochastic rounding.
    ///
    /// # Arguments
    /// * `perturbation` - f32 tensor of perturbation values (z)
    /// * `epsilon` - perturbation magnitude (ε)
    /// * `seed` - random seed for stochastic rounding
    /// * `add` - true to add (w + εz), false to subtract (w - εz)
    pub fn perturb_weights_gpu(
        &mut self,
        perturbation: &CudaStorage,
        epsilon: f32,
        seed: u64,
        add: bool,
    ) -> Result<()> {
        let num_blocks = self.data.len / self.dtype.type_size();
        let num_elements = num_blocks * self.dtype.block_size();

        // Get perturbation as f32 slice
        let perturb_cuda_slice = perturbation.as_cuda_slice::<f32>()?;
        if perturb_cuda_slice.len() < num_elements {
            crate::bail!(
                "perturbation tensor size {} < required {}",
                perturb_cuda_slice.len(),
                num_elements
            );
        }
        let perturb_slice = perturb_cuda_slice.slice(..num_elements);

        // Select kernel and block size based on dtype
        // Q8_0: 32 elements per block, 32 threads
        // Q4K: 256 elements per block, 128 threads (each handles 2 packed nibbles)
        // BF16: 1 element per block, 256 threads for coalesced access
        let (kernel_name, block_dim_x, use_element_launch) = match self.dtype {
            GgmlDType::Q8_0 => ("perturb_q8_0", 32u32, false),
            GgmlDType::Q4K => ("perturb_q4_K", 128u32, false),
            GgmlDType::Q2K => ("perturb_q2_K", 64u32, false),
            GgmlDType::Q3K => ("perturb_q3_K", 64u32, false),
            GgmlDType::Q5K => ("perturb_q5_K", 64u32, false),
            GgmlDType::Q6K => ("perturb_q6_K", 64u32, false),
            GgmlDType::BF16 => ("perturb_bf16", 256u32, true),
            _ => crate::bail!(
                "perturb_weights_gpu not supported for dtype {:?}",
                self.dtype
            ),
        };

        let func = self
            .device
            .get_or_load_func(kernel_name, &paramecia_cuda::QUZO_PERTURB)?;

        // For BF16, launch one thread per element; for quantized, one block per quant block
        let grid_dim = if use_element_launch {
            ((num_elements as u32 + block_dim_x - 1) / block_dim_x, 1, 1)
        } else {
            (num_blocks as u32, 1, 1)
        };

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim,
            block_dim: (block_dim_x, 1, 1),
            shared_mem_bytes: 0,
        };

        let add_i32: i32 = if add { 1 } else { 0 };
        // BF16 kernel takes num_elements, quantized kernels take num_blocks
        let count_arg = if use_element_launch {
            num_elements as i32
        } else {
            num_blocks as i32
        };

        let mut builder = func.builder();
        builder.arg(&self.data.inner);
        builder.arg(&perturb_slice);
        builder.arg(&count_arg);
        builder.arg(&epsilon);
        builder.arg(&seed);
        builder.arg(&add_i32);
        unsafe { builder.launch(cfg) }.w()?;

        Ok(())
    }

    /// Combined restore and update operation for QuZO optimization.
    /// This fuses the restore (from -ε to 0) and gradient update (apply -η*μ*z) into one kernel.
    ///
    /// # Arguments
    /// * `perturbation` - f32 tensor of perturbation values (z)
    /// * `restore_epsilon` - ε to restore from (adds ε*z to go from -ε to 0)
    /// * `update_scale` - η * μ / n (learning_rate * gradient_estimate / num_samples)
    /// * `restore_seed` - random seed for restore stochastic rounding
    /// * `update_seed` - random seed for update stochastic rounding
    pub fn restore_and_update_gpu(
        &mut self,
        perturbation: &CudaStorage,
        restore_epsilon: f32,
        update_scale: f32,
        restore_seed: u64,
        update_seed: u64,
    ) -> Result<()> {
        let num_blocks = self.data.len / self.dtype.type_size();
        let num_elements = num_blocks * self.dtype.block_size();

        let perturb_cuda_slice = perturbation.as_cuda_slice::<f32>()?;
        if perturb_cuda_slice.len() < num_elements {
            crate::bail!(
                "perturbation tensor size {} < required {}",
                perturb_cuda_slice.len(),
                num_elements
            );
        }
        let perturb_slice = perturb_cuda_slice.slice(..num_elements);

        let (kernel_name, block_dim_x, use_element_launch) = match self.dtype {
            GgmlDType::Q8_0 => ("restore_and_update_q8_0", 32u32, false),
            GgmlDType::Q4K => ("restore_and_update_q4_K", 128u32, false),
            GgmlDType::Q2K => ("restore_and_update_q2_K", 64u32, false),
            GgmlDType::Q3K => ("restore_and_update_q3_K", 64u32, false),
            GgmlDType::Q5K => ("restore_and_update_q5_K", 64u32, false),
            GgmlDType::Q6K => ("restore_and_update_q6_K", 64u32, false),
            GgmlDType::BF16 => ("restore_and_update_bf16", 256u32, true),
            _ => crate::bail!(
                "restore_and_update_gpu not supported for dtype {:?}",
                self.dtype
            ),
        };

        let func = self
            .device
            .get_or_load_func(kernel_name, &paramecia_cuda::QUZO_PERTURB)?;

        // For BF16, launch one thread per element; for quantized, one block per quant block
        let grid_dim = if use_element_launch {
            ((num_elements as u32 + block_dim_x - 1) / block_dim_x, 1, 1)
        } else {
            (num_blocks as u32, 1, 1)
        };

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim,
            block_dim: (block_dim_x, 1, 1),
            shared_mem_bytes: 0,
        };

        // BF16 kernel takes num_elements, quantized kernels take num_blocks
        let count_arg = if use_element_launch {
            num_elements as i32
        } else {
            num_blocks as i32
        };

        let mut builder = func.builder();
        builder.arg(&self.data.inner);
        builder.arg(&perturb_slice);
        builder.arg(&count_arg);
        builder.arg(&restore_epsilon);
        builder.arg(&update_scale);
        builder.arg(&restore_seed);
        builder.arg(&update_seed);
        unsafe { builder.launch(cfg) }.w()?;

        Ok(())
    }

    /// Fused dequantize + perturb + matmul operation.
    /// Computes: y = (W + ε*z) @ x where z is generated on-the-fly from seed.
    /// This avoids storing perturbed weights, ideal for CPU-offloaded experts.
    ///
    /// # Arguments
    /// * `self_shape` - Shape of the weight matrix [nrows, ncols]
    /// * `storage` - Input tensor to multiply with
    /// * `layout` - Layout of the input tensor
    /// * `seed` - Random seed for generating perturbation z
    /// * `epsilon` - Perturbation magnitude ε
    pub fn fused_fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
        seed: u64,
        epsilon: f32,
    ) -> Result<(CudaStorage, crate::Shape)> {
        // For now, only support vector-matrix multiply (batch size 1)
        // which is the common case for QuZO training
        self.fused_matmul_vec(self_shape, storage, layout, seed, epsilon)
    }

    fn fused_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
        seed: u64,
        epsilon: f32,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;

        let (nrows, ncols) = self_shape.dims2()?;

        // Get input dtype for potential conversion
        let input_dtype = rhs.dtype();

        // Convert to F32 if needed for the fused kernel
        let rhs_owned;
        let rhs_f32 = if input_dtype != crate::DType::F32 {
            rhs_owned = rhs.to_dtype(rhs_l, crate::DType::F32)?;
            &rhs_owned
        } else {
            rhs
        };

        let rhs_slice = rhs_f32.as_cuda_slice::<f32>()?;
        let rhs_slice = match rhs_l.contiguous_offsets() {
            Some((o1, o2)) => rhs_slice.slice(o1..o2),
            None => return Err(crate::Error::RequiresContiguous { op: "fused_dmmv" }.bt()),
        };

        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!("unexpected rhs shape in fused_dmmv {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }

        // Select appropriate kernel based on dtype
        let kernel_name = match self.dtype {
            GgmlDType::Q8_0 => "fused_mul_mat_vec_q8_0_f32",
            GgmlDType::Q4K => "fused_mul_mat_vec_q4_K_f32",
            GgmlDType::Q2K => "fused_mul_mat_vec_q2_K_f32",
            GgmlDType::Q3K => "fused_mul_mat_vec_q3_K_f32",
            GgmlDType::Q5K => "fused_mul_mat_vec_q5_K_f32",
            GgmlDType::Q6K => "fused_mul_mat_vec_q6_K_f32",
            GgmlDType::BF16 => "fused_mul_mat_vec_bf16_f32",
            _ => crate::bail!("fused_fwd not supported for dtype {:?}", self.dtype),
        };

        let func = self
            .device
            .get_or_load_func(kernel_name, &paramecia_cuda::QUZO_FUSED)?;

        let dst = unsafe { self.device.alloc::<f32>(nrows * b_size)? };

        // For vector case (b_size=1), each block handles one row
        // blockDim = (32, 1, 1) - one warp per row
        // gridDim = (nrows, 1, 1)
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (nrows as u32, 1, 1),
            block_dim: (WARP_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let ncols_i32 = ncols as i32;
        let nrows_i32 = nrows as i32;

        let mut builder = func.builder();
        let data_slice = self.data.inner.slice(..);
        // Get weight pointer for per-tensor uniqueness
        // The device pointer is XORed with seed in the kernel
        let weight_ptr = self.device_ptr_u64();
        builder.arg(&data_slice);
        builder.arg(&rhs_slice);
        builder.arg(&dst);
        builder.arg(&ncols_i32);
        builder.arg(&nrows_i32);
        builder.arg(&seed);
        builder.arg(&epsilon);
        builder.arg(&weight_ptr);
        unsafe { builder.launch(cfg) }.w()?;

        let out = CudaStorage::wrap_cuda_slice(dst, self.device.clone());

        // Build output shape
        let mut out_shape: Vec<usize> = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        let out_shape: crate::Shape = out_shape.into();

        // Convert output back to input dtype if needed
        let out = if input_dtype != crate::DType::F32 {
            let out_layout = crate::Layout::contiguous(&out_shape);
            out.to_dtype(&out_layout, input_dtype)?
        } else {
            out
        };

        Ok((out, out_shape))
    }

    /// Extract scaling factors from all quantization blocks to a GPU tensor.
    /// This is used for QZO (Quantized Zeroth-Order Optimization) where only
    /// the scaling factors are trainable.
    pub fn extract_scales(&self) -> Result<CudaStorage> {
        let _block_size = self.dtype.block_size();
        let num_blocks = self.data.len / self.dtype.type_size();

        // Allocate output buffer for scales
        let scales = unsafe { self.device.alloc::<f32>(num_blocks)? };

        // Select appropriate kernel based on dtype
        let kernel_name = match self.dtype {
            GgmlDType::Q4_0 => "extract_scales_q4_0",
            GgmlDType::Q4_1 => "extract_scales_q4_1",
            GgmlDType::Q5_0 => "extract_scales_q5_0",
            GgmlDType::Q5_1 => "extract_scales_q5_1",
            GgmlDType::Q8_0 => "extract_scales_q8_0",
            GgmlDType::Q2K => "extract_scales_q2_K",
            GgmlDType::Q3K => "extract_scales_q3_K",
            GgmlDType::Q4K => "extract_scales_q4_K",
            GgmlDType::Q5K => "extract_scales_q5_K",
            GgmlDType::Q6K => "extract_scales_q6_K",
            _ => crate::bail!("extract_scales not supported for dtype {:?}", self.dtype),
        };

        let func = self
            .device
            .get_or_load_func(kernel_name, &paramecia_cuda::QUANTIZED)?;

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (((num_blocks + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&self.data.inner);
        builder.arg(&scales);
        let num_blocks_i32 = num_blocks as i32;
        builder.arg(&num_blocks_i32);
        unsafe { builder.launch(cfg) }.w()?;

        Ok(CudaStorage::wrap_cuda_slice(scales, self.device.clone()))
    }
}

impl QCudaStorage {
    fn dequantize_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;

        let (nrows, ncols) = self_shape.dims2()?;

        // Get input dtype for potential conversion
        let input_dtype = rhs.dtype();

        // Convert to F32 if needed for the quantized matmul kernel
        let rhs_owned;
        let rhs_f32 = if input_dtype != crate::DType::F32 {
            rhs_owned = rhs.to_dtype(rhs_l, crate::DType::F32)?;
            &rhs_owned
        } else {
            rhs
        };

        let rhs_slice = rhs_f32.as_cuda_slice::<f32>()?;
        let rhs_slice = match rhs_l.contiguous_offsets() {
            Some((o1, o2)) => rhs_slice.slice(o1..o2),
            None => return Err(crate::Error::RequiresContiguous { op: "dmmv" }.bt()),
        };

        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!("unexpected rhs shape in dmmv {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            dequantize_mul_mat_vec(
                &self.data,
                &rhs_slice,
                self.dtype,
                ncols,
                nrows,
                self.device(),
            )?
        } else {
            mul_mat_vec_via_q8_1(
                &self.data,
                &rhs_slice,
                self.dtype,
                ncols,
                nrows,
                b_size,
                self.device(),
            )?
        };

        // Convert output back to input dtype if needed
        let mut out_shape: Vec<usize> = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        let out_shape: crate::Shape = out_shape.into();

        let out = if input_dtype != crate::DType::F32 {
            let out_layout = crate::Layout::contiguous(&out_shape);
            out.to_dtype(&out_layout, input_dtype)?
        } else {
            out
        };

        Ok((out, out_shape))
    }

    fn dequantize_matmul(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }

        // Get input dtype for potential conversion
        let input_dtype = storage.dtype();

        // For F16 input with FORCE_DMMV, stay in F16 throughout for better performance
        let use_f16_path = input_dtype == crate::DType::F16
            && FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed);

        let out = if use_f16_path {
            // F16-native path: dequantize to F16, do F16 matmul
            let data_f16 = self.dequantize_f16(n * k)?;
            let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
            // Input is already F16, do matmul directly
            storage.matmul(&data_f16, (b, m, n, k), layout, &rhs_l)?
        } else if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            let data_f32 = self.dequantize(n * k)?;
            let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
            // Convert storage to F32 if needed for matmul with dequantized F32 weights
            let storage_owned;
            let storage_f32: &CudaStorage = if input_dtype != crate::DType::F32 {
                storage_owned = storage.to_dtype(layout, crate::DType::F32)?;
                &storage_owned
            } else {
                storage
            };
            storage_f32.matmul(&data_f32, (b, m, n, k), layout, &rhs_l)?
        } else {
            // Convert to F32 if needed for the quantized matmul kernel
            let storage_owned;
            let storage_f32 = if input_dtype != crate::DType::F32 {
                storage_owned = storage.to_dtype(layout, crate::DType::F32)?;
                &storage_owned
            } else {
                storage
            };

            let storage_slice = storage_f32.as_cuda_slice::<f32>()?;
            let storage_slice = match layout.contiguous_offsets() {
                Some((o1, o2)) => storage_slice.slice(o1..o2),
                None => {
                    return Err(crate::Error::RequiresContiguous {
                        op: "quantized-matmul",
                    }
                    .bt())
                }
            };

            mul_mat_via_q8_1(
                &self.data,
                &storage_slice,
                self.dtype,
                /* x_rows */ n,
                /* x_cols */ k,
                /* y_rows */ k,
                /* y_cols */ b * m,
                self.device(),
            )?
        };

        // Build output shape
        let mut out_shape: Vec<usize> = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        let out_shape: crate::Shape = out_shape.into();

        // Convert output back to input dtype if needed (skip if we used F16 path)
        let out = if !use_f16_path && input_dtype != crate::DType::F32 {
            let out_layout = crate::Layout::contiguous(&out_shape);
            out.to_dtype(&out_layout, input_dtype)?
        } else {
            out
        };

        Ok((out, out_shape))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &CudaDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let dtype = T::DTYPE;
    let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let mut inner = unsafe { device.alloc::<u8>(padded_len)? };
    // GGUF data is commonly backed by a temporary pageable Vec. Wait for each
    // upload before that buffer is dropped so CUDA cannot accumulate a
    // model-sized queue of driver-owned pinned staging allocations.
    device.memcpy_htod_synchronized(data, &mut inner.slice_mut(..data.len()))?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: PaddedCudaSlice {
            inner,
            len: data.len(),
        },
        device: device.clone(),
        dtype,
    }))
}

/// Allocate a pre-sized GPU buffer for KV cache storage.
/// The buffer is allocated with `total_bytes` capacity but initially has 0 valid bytes.
pub fn alloc_kv_buffer(
    device: &CudaDevice,
    total_bytes: usize,
    dtype: GgmlDType,
) -> Result<QStorage> {
    let inner = unsafe { device.alloc::<u8>(total_bytes)? };
    Ok(QStorage::Cuda(QCudaStorage {
        data: PaddedCudaSlice {
            inner,
            len: total_bytes,
        },
        device: device.clone(),
        dtype,
    }))
}

/// Copy host bytes into a GPU KV cache buffer at the given byte offset.
pub fn kv_buffer_append(
    storage: &mut QStorage,
    host_data: &[u8],
    byte_offset: usize,
) -> Result<()> {
    match storage {
        QStorage::Cuda(s) => {
            let end = byte_offset + host_data.len();
            s.device
                .memcpy_htod(host_data, &mut s.data.inner.slice_mut(byte_offset..end))?;
            Ok(())
        }
        _ => crate::bail!("kv_buffer_append requires CUDA storage"),
    }
}

/// Copy a prefix between CUDA KV buffers without staging through host memory.
pub fn kv_buffer_copy_prefix(
    src: &QStorage,
    dst: &mut QStorage,
    bytes_to_copy: usize,
) -> Result<()> {
    match (src, dst) {
        (QStorage::Cuda(src), QStorage::Cuda(dst)) => {
            if bytes_to_copy > src.data.inner.len() || bytes_to_copy > dst.data.inner.len() {
                crate::bail!(
                    "kv_buffer_copy_prefix overflow: bytes={} src={} dst={}",
                    bytes_to_copy,
                    src.data.inner.len(),
                    dst.data.inner.len()
                );
            }
            dst.device.memcpy_dtod(
                &src.data.inner.slice(..bytes_to_copy),
                &mut dst.data.inner.slice_mut(..bytes_to_copy),
            )?;
            Ok(())
        }
        _ => crate::bail!("kv_buffer_copy_prefix requires CUDA storage"),
    }
}

/// Download a byte range from a CUDA KV buffer.
///
/// This is intended for explicit snapshot/debug paths, not decode hot paths.
pub fn kv_buffer_download_range(
    storage: &QStorage,
    byte_offset: usize,
    bytes_to_copy: usize,
) -> Result<Vec<u8>> {
    match storage {
        QStorage::Cuda(storage) => {
            let end = byte_offset + bytes_to_copy;
            if end > storage.data.inner.len() {
                crate::bail!(
                    "kv_buffer_download_range overflow: end={} capacity={}",
                    end,
                    storage.data.inner.len()
                );
            }
            let mut host = vec![0u8; bytes_to_copy];
            storage
                .device
                .memcpy_dtoh(&storage.data.inner.slice(byte_offset..end), &mut host)?;
            Ok(host)
        }
        _ => crate::bail!("kv_buffer_download_range requires CUDA storage"),
    }
}

/// Dequantize a CUDA KV buffer without cloning its device allocation.
pub fn kv_buffer_dequantize(storage: &QStorage, shape: &crate::Shape) -> Result<crate::Tensor> {
    match storage {
        QStorage::Cuda(storage) => {
            let dequantized = storage.dequantize(shape.elem_count())?;
            Ok(crate::Tensor::from_storage(
                crate::Storage::Cuda(dequantized),
                shape.clone(),
            ))
        }
        _ => crate::bail!("kv_buffer_dequantize requires CUDA storage"),
    }
}

/// Returns the total allocated byte capacity of a CUDA KV buffer.
pub fn kv_buffer_capacity(storage: &QStorage) -> usize {
    match storage {
        QStorage::Cuda(s) => s.data.inner.len(),
        _ => 0,
    }
}

/// Quantize a contiguous CUDA f32 matrix `[rows, k]` to Q8_0 on-device and append into
/// a CUDA KV buffer at `byte_offset`.
pub fn kv_buffer_append_quantized_q8_0_f32(
    storage: &mut QStorage,
    src: &CudaStorage,
    k: usize,
    rows: usize,
    byte_offset: usize,
) -> Result<()> {
    let (dst, src_f32) = match (storage, &src.slice) {
        (QStorage::Cuda(dst), crate::cuda_backend::CudaStorageSlice::F32(src_f32)) => {
            (dst, src_f32)
        }
        (QStorage::Cuda(_), _) => {
            crate::bail!("kv_buffer_append_quantized_q8_0_f32 requires f32 CUDA source")
        }
        _ => crate::bail!("kv_buffer_append_quantized_q8_0_f32 requires CUDA destination"),
    };

    if dst.dtype != GgmlDType::Q8_0 {
        crate::bail!(
            "kv_buffer_append_quantized_q8_0_f32 requires Q8_0 destination (got {:?})",
            dst.dtype
        );
    }
    if k % GgmlDType::Q8_0.block_size() != 0 {
        crate::bail!(
            "kv_buffer_append_quantized_q8_0_f32 requires k divisible by {} (got {})",
            GgmlDType::Q8_0.block_size(),
            k
        );
    }
    if rows == 0 {
        return Ok(());
    }

    let src_len = src_f32.len();
    let expected_src_len = rows * k;
    if src_len < expected_src_len {
        crate::bail!(
            "kv_buffer_append_quantized_q8_0_f32 source too small: {} < {}",
            src_len,
            expected_src_len
        );
    }

    let bytes_per_row = (k / GgmlDType::Q8_0.block_size()) * GgmlDType::Q8_0.type_size();
    let bytes_to_copy = rows * bytes_per_row;
    let end = byte_offset + bytes_to_copy;
    if end > dst.data.inner.len() {
        crate::bail!(
            "kv_buffer_append_quantized_q8_0_f32 overflow: {} > {}",
            end,
            dst.data.inner.len()
        );
    }

    quantize_q8_0_rowwise(
        &src_f32.as_view(),
        &mut dst.data.inner,
        byte_offset,
        k,
        rows,
        &dst.device,
    )
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn cuda_load_quantized_from_temporary_pageable_data() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let source = vec![1.25f32, -2.5, 3.75, -4.0];
        let expected = source
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();

        let storage = load_quantized(&dev, &source)?;
        drop(source);

        let QStorage::Cuda(storage) = storage else {
            unreachable!("CUDA load must produce CUDA storage")
        };
        let actual = dev.clone_dtoh(&storage.data.inner.slice(..storage.data.len))?;
        dev.synchronize()?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn cuda_quantize_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let el = 256;
        let el_padded = pad(el, MATRIX_ROW_PADDING);
        let y_size_in_bytes =
            el_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
        let vs: Vec<f32> = (0..el).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        quantize_q8_1(&y.as_view(), &mut y_q8_1, el, 1, &dev)?;
        Ok(())
    }

    #[test]
    fn cuda_mmv_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_vec_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            /* b_size */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        // for n = 255, n.(n+1).(2n+1) / 6 = 5559680
        // Q8 means 1/256 precision.
        assert_eq!(vs[0], 5561664.5);

        let cuda_storage = dequantize_mul_mat_vec(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0], 5561851.0);
        Ok(())
    }

    #[test]
    fn cuda_mm_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols * 4).map(|v| v as f32 / 4.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * 4, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ 4,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ 4,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;

        /*
           x = torch.tensor([float(v) for v in range(1024)]).reshape(4, 256)
           x @ x.t() / 16
        tensor([[  347480.0000,   869720.0000,  1391960.0000,  1914200.0000],
                [  869720.0000,  2440536.0000,  4011352.0000,  5582166.5000],
                [ 1391960.0000,  4011352.0000,  6630742.0000,  9250132.0000],
                [ 1914200.0000,  5582166.5000,  9250132.0000, 12918099.0000]])
                */
        assert_eq!(vs.len(), 16);
        assert_eq!(vs[0], 347604.0);
        assert_eq!(vs[1], 888153.06);
        assert_eq!(vs[4], 869780.7);
        assert_eq!(vs[5], 2483145.0);
        assert_eq!(vs[11], 9407368.0);
        assert_eq!(vs[14], 9470856.0);
        assert_eq!(vs[15], 13138824.0);
        Ok(())
    }

    // The following test used to fail under compute-sanitizer until #2526.
    #[test]
    fn cuda_mm_q8_1_pad() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (x_rows, ncols, y_cols) = (4, 16, 2048);
        let vs: Vec<f32> = (0..ncols * y_cols).map(|v| v as f32 / 256.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ x_rows,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ y_cols,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let _vs = dev.clone_dtoh(&vs.as_view())?;
        Ok(())
    }
}
