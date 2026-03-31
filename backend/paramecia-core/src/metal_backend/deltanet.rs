//! Metal backend implementations for Delta Net / Linear Attention operations.

use super::{buffer_o, MetalError, MetalStorage};
use crate::backend::BackendStorage;
use crate::{DType, Layout, Result};

/// L2 normalize and scale in one fused operation.
pub fn l2_normalize_scale(
    x: &MetalStorage,
    x_layout: &Layout,
    scale: f32,
    eps: f32,
) -> Result<MetalStorage> {
    let device = x.device();
    let dtype = x.dtype();

    let shape = x_layout.shape();
    let dims = shape.dims();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let dim = dims[dims.len() - 1];
    let el_count = shape.elem_count();

    let kernel_name = match dtype {
        DType::F32 => "l2_normalize_scale_f32",
        DType::F16 => "l2_normalize_scale_f16",
        dtype => crate::bail!("Metal l2_normalize_scale not implemented for {dtype:?}"),
    };

    let output = device.new_buffer(el_count, dtype, "l2_normalize_scale")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("l2_normalize_scale");

    let src = buffer_o(x.buffer(), x_layout, dtype);
    paramecia_metal::call_l2_normalize_scale(
        &device.device,
        &encoder,
        &device.kernels,
        kernel_name,
        batch,
        dim,
        scale,
        eps,
        src,
        &output,
    )
    .map_err(MetalError::from)?;

    Ok(MetalStorage::new(output, device.clone(), el_count, dtype))
}

/// Depthwise 1D convolution for SSM/Mamba-style models.
pub fn depthwise_conv1d(
    input: &MetalStorage,
    input_layout: &Layout,
    weight: &MetalStorage,
    weight_layout: &Layout,
    output_len: usize,
) -> Result<MetalStorage> {
    let device = input.device();
    let dtype = input.dtype();

    let (batch, channels, input_len) = input_layout.shape().dims3()?;
    let (_, kernel_size) = weight_layout.shape().dims2()?;
    let el_count = batch * channels * output_len;

    let kernel_name = match dtype {
        DType::F32 => "depthwise_conv1d_f32",
        DType::F16 => "depthwise_conv1d_f16",
        dtype => crate::bail!("Metal depthwise_conv1d not implemented for {dtype:?}"),
    };

    let output = device.new_buffer(el_count, dtype, "depthwise_conv1d")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("depthwise_conv1d");

    let input_src = buffer_o(input.buffer(), input_layout, dtype);
    let weight_src = buffer_o(weight.buffer(), weight_layout, dtype);
    paramecia_metal::call_depthwise_conv1d(
        &device.device,
        &encoder,
        &device.kernels,
        kernel_name,
        batch,
        channels,
        input_len,
        output_len,
        kernel_size,
        input_src,
        weight_src,
        &output,
    )
    .map_err(MetalError::from)?;

    Ok(MetalStorage::new(output, device.clone(), el_count, dtype))
}

/// Fused SwiGLU activation: silu(gate) * up
pub fn fused_swiglu(
    gate: &MetalStorage,
    gate_layout: &Layout,
    up: &MetalStorage,
    up_layout: &Layout,
) -> Result<MetalStorage> {
    let device = gate.device();
    let dtype = gate.dtype();

    let el_count = gate_layout.shape().elem_count();

    let kernel_name = match dtype {
        DType::F32 => "fused_swiglu_f32",
        DType::F16 => "fused_swiglu_f16",
        dtype => crate::bail!("Metal fused_swiglu not implemented for {dtype:?}"),
    };

    let output = device.new_buffer(el_count, dtype, "fused_swiglu")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("fused_swiglu");

    let gate_src = buffer_o(gate.buffer(), gate_layout, dtype);
    let up_src = buffer_o(up.buffer(), up_layout, dtype);
    paramecia_metal::call_fused_swiglu(
        &device.device,
        &encoder,
        &device.kernels,
        kernel_name,
        el_count,
        gate_src,
        up_src,
        &output,
    )
    .map_err(MetalError::from)?;

    Ok(MetalStorage::new(output, device.clone(), el_count, dtype))
}

/// Gated RMS norm: silu(z) * rms_norm(x, weight)
pub fn gated_rms_norm(
    x: &MetalStorage,
    x_layout: &Layout,
    z: &MetalStorage,
    z_layout: &Layout,
    weight: &MetalStorage,
    weight_layout: &Layout,
    eps: f32,
) -> Result<MetalStorage> {
    let device = x.device();
    let dtype = x.dtype();

    let dims = x_layout.shape().dims();
    let batch: usize = dims[..dims.len() - 1].iter().product();
    let dim = dims[dims.len() - 1];
    let el_count = x_layout.shape().elem_count();

    let kernel_name = match dtype {
        DType::F32 => "gated_rms_norm_f32",
        dtype => crate::bail!("Metal gated_rms_norm not implemented for {dtype:?}"),
    };

    let output = device.new_buffer(el_count, dtype, "gated_rms_norm")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("gated_rms_norm");

    let x_src = buffer_o(x.buffer(), x_layout, dtype);
    let z_src = buffer_o(z.buffer(), z_layout, dtype);
    let weight_src = buffer_o(weight.buffer(), weight_layout, dtype);
    paramecia_metal::call_gated_rms_norm(
        &device.device,
        &encoder,
        &device.kernels,
        kernel_name,
        batch,
        dim,
        eps,
        x_src,
        z_src,
        weight_src,
        &output,
    )
    .map_err(MetalError::from)?;

    Ok(MetalStorage::new(output, device.clone(), el_count, dtype))
}

/// Lower triangular solve for (I - L) @ X = B
pub fn solve_i_minus_lower_triangular(
    l_matrix: &MetalStorage,
    l_layout: &Layout,
    b_matrix: &MetalStorage,
    b_layout: &Layout,
) -> Result<MetalStorage> {
    let device = l_matrix.device();
    let dtype = l_matrix.dtype();

    let l_dims = l_layout.shape().dims();
    let b_dims = b_layout.shape().dims();

    let n = l_dims[l_dims.len() - 1];
    let k = b_dims[b_dims.len() - 1];
    let batch: usize = l_dims[..l_dims.len() - 2].iter().product();
    let batch = batch.max(1);
    let el_count = b_layout.shape().elem_count();

    let kernel_name = match dtype {
        DType::F32 => "solve_i_minus_lower_tri_f32",
        dtype => crate::bail!("Metal solve_i_minus_lower_triangular not implemented for {dtype:?}"),
    };

    let output = device.new_buffer(el_count, dtype, "solve_lower_tri")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("solve_lower_tri");

    let l_src = buffer_o(l_matrix.buffer(), l_layout, dtype);
    let b_src = buffer_o(b_matrix.buffer(), b_layout, dtype);
    paramecia_metal::call_solve_i_minus_lower_tri(
        &device.device,
        &encoder,
        &device.kernels,
        kernel_name,
        batch,
        n,
        k,
        l_src,
        b_src,
        &output,
    )
    .map_err(MetalError::from)?;

    Ok(MetalStorage::new(output, device.clone(), el_count, dtype))
}

/// Fused Delta Net autoregressive step
#[allow(clippy::too_many_arguments)]
pub fn delta_net_autoregressive_step(
    q: &MetalStorage,
    q_layout: &Layout,
    k: &MetalStorage,
    k_layout: &Layout,
    v: &MetalStorage,
    v_layout: &Layout,
    gate: &MetalStorage,
    gate_layout: &Layout,
    beta: &MetalStorage,
    beta_layout: &Layout,
    state: &MetalStorage,
    state_layout: &Layout,
    scale: f32,
    eps: f32,
) -> Result<(MetalStorage, MetalStorage)> {
    let device = q.device();
    let dtype = q.dtype();

    let q_dims = q_layout.shape().dims3()?;
    let batch = q_dims.0;
    let num_heads = q_dims.1;
    let head_dim = q_dims.2;

    let output_el_count = batch * num_heads * head_dim;
    let state_el_count = batch * num_heads * head_dim * head_dim;

    let kernel_name = match dtype {
        DType::F32 => "delta_net_autoregressive_step_f32",
        dtype => crate::bail!("Metal delta_net_autoregressive_step not implemented for {dtype:?}"),
    };

    let output = device.new_buffer(output_el_count, dtype, "delta_net_step_output")?;
    let new_state = device.new_buffer(state_el_count, dtype, "delta_net_step_state")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("delta_net_autoregressive_step");

    let q_src = buffer_o(q.buffer(), q_layout, dtype);
    let k_src = buffer_o(k.buffer(), k_layout, dtype);
    let v_src = buffer_o(v.buffer(), v_layout, dtype);
    let gate_src = buffer_o(gate.buffer(), gate_layout, dtype);
    let beta_src = buffer_o(beta.buffer(), beta_layout, dtype);
    let state_src = buffer_o(state.buffer(), state_layout, dtype);

    paramecia_metal::call_delta_net_autoregressive_step(
        &device.device,
        &encoder,
        &device.kernels,
        kernel_name,
        batch,
        num_heads,
        head_dim,
        scale,
        eps,
        q_src,
        k_src,
        v_src,
        gate_src,
        beta_src,
        state_src,
        &output,
        &new_state,
    )
    .map_err(MetalError::from)?;

    Ok((
        MetalStorage::new(output, device.clone(), output_el_count, dtype),
        MetalStorage::new(new_state, device.clone(), state_el_count, dtype),
    ))
}

/// Multi-token Delta Net state update
#[allow(clippy::too_many_arguments)]
pub fn delta_net_multi_token_update(
    q: &MetalStorage,
    q_layout: &Layout,
    k: &MetalStorage,
    k_layout: &Layout,
    v: &MetalStorage,
    v_layout: &Layout,
    gate: &MetalStorage,
    gate_layout: &Layout,
    beta: &MetalStorage,
    beta_layout: &Layout,
    state: &MetalStorage,
    state_layout: &Layout,
) -> Result<(MetalStorage, MetalStorage)> {
    let device = q.device();
    let dtype = q.dtype();

    let q_dims = q_layout.shape().dims();
    if q_dims.len() != 4 {
        crate::bail!(
            "delta_net_multi_token_update: q must be 4D, got {:?}",
            q_dims
        );
    }
    let batch = q_dims[0];
    let num_heads = q_dims[1];
    let num_tokens = q_dims[2];
    let head_dim = q_dims[3];

    let output_el_count = batch * num_heads * num_tokens * head_dim;
    let state_el_count = batch * num_heads * head_dim * head_dim;

    let kernel_name = match dtype {
        DType::F32 => "delta_net_multi_token_update_f32",
        dtype => crate::bail!("Metal delta_net_multi_token_update not implemented for {dtype:?}"),
    };

    let output = device.new_buffer(output_el_count, dtype, "delta_net_mtu_output")?;
    let new_state = device.new_buffer(state_el_count, dtype, "delta_net_mtu_state")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("delta_net_multi_token_update");

    let q_src = buffer_o(q.buffer(), q_layout, dtype);
    let k_src = buffer_o(k.buffer(), k_layout, dtype);
    let v_src = buffer_o(v.buffer(), v_layout, dtype);
    let gate_src = buffer_o(gate.buffer(), gate_layout, dtype);
    let beta_src = buffer_o(beta.buffer(), beta_layout, dtype);
    let state_src = buffer_o(state.buffer(), state_layout, dtype);

    paramecia_metal::call_delta_net_multi_token_update(
        &device.device,
        &encoder,
        &device.kernels,
        kernel_name,
        batch,
        num_heads,
        num_tokens,
        head_dim,
        q_src,
        k_src,
        v_src,
        gate_src,
        beta_src,
        state_src,
        &output,
        &new_state,
    )
    .map_err(MetalError::from)?;

    Ok((
        MetalStorage::new(output, device.clone(), output_el_count, dtype),
        MetalStorage::new(new_state, device.clone(), state_el_count, dtype),
    ))
}

/// Delta Net parallel with intermediate state materialization
#[allow(clippy::too_many_arguments)]
pub fn delta_net_parallel_with_states(
    q: &MetalStorage,
    q_layout: &Layout,
    k: &MetalStorage,
    k_layout: &Layout,
    v: &MetalStorage,
    v_layout: &Layout,
    gate: &MetalStorage,
    gate_layout: &Layout,
    beta: &MetalStorage,
    beta_layout: &Layout,
    state: &MetalStorage,
    state_layout: &Layout,
) -> Result<(MetalStorage, MetalStorage, MetalStorage)> {
    let device = q.device();
    let dtype = q.dtype();

    let q_dims = q_layout.shape().dims();
    if q_dims.len() != 4 {
        crate::bail!(
            "delta_net_parallel_with_states: q must be 4D, got {:?}",
            q_dims
        );
    }
    let batch = q_dims[0];
    let num_heads = q_dims[1];
    let num_tokens = q_dims[2];
    let head_dim = q_dims[3];

    let output_el_count = batch * num_heads * num_tokens * head_dim;
    let state_el_count = batch * num_heads * head_dim * head_dim;
    let inter_state_el_count = batch * num_heads * num_tokens * head_dim * head_dim;

    let kernel_name = match dtype {
        DType::F32 => "delta_net_parallel_with_states_f32",
        dtype => crate::bail!("Metal delta_net_parallel_with_states not implemented for {dtype:?}"),
    };

    let output = device.new_buffer(output_el_count, dtype, "delta_net_pws_output")?;
    let new_state = device.new_buffer(state_el_count, dtype, "delta_net_pws_state")?;
    let intermediate_states =
        device.new_buffer(inter_state_el_count, dtype, "delta_net_pws_inter")?;
    let encoder = device.command_encoder()?;
    encoder.set_label("delta_net_parallel_with_states");

    let q_src = buffer_o(q.buffer(), q_layout, dtype);
    let k_src = buffer_o(k.buffer(), k_layout, dtype);
    let v_src = buffer_o(v.buffer(), v_layout, dtype);
    let gate_src = buffer_o(gate.buffer(), gate_layout, dtype);
    let beta_src = buffer_o(beta.buffer(), beta_layout, dtype);
    let state_src = buffer_o(state.buffer(), state_layout, dtype);

    paramecia_metal::call_delta_net_parallel_with_states(
        &device.device,
        &encoder,
        &device.kernels,
        kernel_name,
        batch,
        num_heads,
        num_tokens,
        head_dim,
        q_src,
        k_src,
        v_src,
        gate_src,
        beta_src,
        state_src,
        &output,
        &new_state,
        &intermediate_states,
    )
    .map_err(MetalError::from)?;

    Ok((
        MetalStorage::new(output, device.clone(), output_el_count, dtype),
        MetalStorage::new(new_state, device.clone(), state_el_count, dtype),
        MetalStorage::new(
            intermediate_states,
            device.clone(),
            inter_state_el_count,
            dtype,
        ),
    ))
}

/// Flash attention with Q8_0 quantized KV cache for Metal.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_q8_metal(
    q: &crate::Tensor,
    k_storage: &crate::quantized::QStorage,
    v_storage: &crate::quantized::QStorage,
    q_l: &Layout,
    k_l: &Layout,
    v_l: &Layout,
    scale: f32,
    b: usize,
    h: usize,
    _h_k: usize,
    d: usize,
    seq_q: usize,
    seq_k: usize,
    _q_offset: usize,
    _causal: bool,
) -> Result<crate::Tensor> {
    use crate::quantized::QStorage;

    // Get Metal device from Q tensor
    let device = match q.device() {
        crate::Device::Metal(dev) => dev.clone(),
        _ => crate::bail!("flash_attn_q8_metal requires Metal device"),
    };

    // Convert Q to F16 if needed
    let q_f16 = match q.dtype() {
        DType::F16 => q.clone(),
        DType::BF16 | DType::F32 => q.to_dtype(DType::F16)?,
        dt => crate::bail!("flash_attn_q8: unsupported Q dtype {dt:?}"),
    }
    .contiguous()?;

    let q_storage_guard = q_f16.storage();
    let q_metal = match &*q_storage_guard {
        crate::Storage::Metal(s) => s,
        _ => crate::bail!("flash_attn_q8_metal requires Metal storage"),
    };

    let q_buf = q_metal.buffer();

    // Get K and V Metal buffers from QStorage
    let (k_buf, v_buf) = match (k_storage, v_storage) {
        (QStorage::Metal(k_metal), QStorage::Metal(v_metal)) => {
            (k_metal.buffer().clone(), v_metal.buffer().clone())
        }
        _ => crate::bail!("flash_attn_q8_metal requires Metal quantized storage"),
    };

    // Allocate output buffer (F16)
    let out_shape = crate::Shape::from((b, seq_q, h, d));
    let out_el_count = out_shape.elem_count();
    let out_buf = device.new_buffer(out_el_count, DType::F16, "flash_attn_q8_output")?;

    let encoder = device.command_encoder()?;
    encoder.set_label("flash_attn_q8");

    // Compute strides
    let q_stride = q_l.stride();
    let k_stride = k_l.stride();
    let v_stride = v_l.stride();
    let out_layout = crate::Layout::contiguous(&out_shape);
    let o_stride = out_layout.stride();

    paramecia_metal::call_flash_attn_q8_metal(
        &device.device,
        &encoder,
        &device.kernels,
        d,
        q_buf,
        &k_buf,
        &v_buf,
        &out_buf,
        scale,
        seq_k,
        seq_q,
        q_stride[0],
        q_stride[2],
        q_stride[1],
        q_stride[3],
        k_stride[0],
        k_stride[2],
        k_stride[1],
        v_stride[0],
        v_stride[2],
        v_stride[1],
        v_stride[3],
        o_stride[0],
        o_stride[2],
        o_stride[1],
        b,
        h,
    )
    .map_err(MetalError::from)?;

    let storage = MetalStorage::new(out_buf, device.clone(), out_el_count, DType::F16);
    let output = crate::Tensor::from_storage(crate::Storage::Metal(storage), out_shape);
    Ok(output)
}
