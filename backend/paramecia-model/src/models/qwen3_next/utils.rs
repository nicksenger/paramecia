use paramecia_core::{DType, Device, Result, Tensor};

/// Log tensor shape at a checkpoint (enabled by PARAMECIA_SHAPE_LOG=1)
pub(super) fn log_shape(name: &str, t: &Tensor) {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if *ENABLED.get_or_init(|| std::env::var("PARAMECIA_SHAPE_LOG").is_ok()) {
        tracing::info!("{}: {:?} (dtype={:?})", name, t.dims(), t.dtype());
    }
}

/// Inner logging helper shared by typed and untyped QMatMul shape loggers.
fn log_qmatmul_shape_inner(
    name: &str,
    qtensor: Option<&paramecia_core::quantized::QTensor>,
    shared: Option<paramecia_core::quantized::SharedQTensor>,
    out_dim: Option<usize>,
) {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("PARAMECIA_SHAPE_LOG").is_ok()) {
        return;
    }
    if let Some(qt) = qtensor {
        tracing::info!(
            "qmatmul {}: {:?} (ggml_dtype={:?})",
            name,
            qt.shape().dims(),
            qt.dtype()
        );
    } else if let Some(sq) = shared {
        let dims = sq.dims();
        tracing::info!("qmatmul {}: {:?} (shared)", name, dims);
    } else if let Some(od) = out_dim {
        tracing::info!("qmatmul {}: out_dim={} (dequantized)", name, od);
    }
}

/// Log typed QMatMul weight shape at initialization (enabled by PARAMECIA_SHAPE_LOG=1)
pub(super) fn log_typed_qmatmul_shape<S: paramecia_tensor::glowstick::Shape>(
    name: &str,
    qmm: &paramecia_tensor::QMatMul<S>,
) {
    log_qmatmul_shape_inner(name, qmm.qtensor(), qmm.shared_qtensor(), qmm.out_dim());
}

pub(super) fn softplus(x: &Tensor) -> Result<Tensor> {
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

pub(super) fn l2_normalize(x: &Tensor, eps: f64) -> Result<Tensor> {
    // Use fused L2 normalize kernel when available (scale=1.0 for just normalization)
    // The fused kernel avoids multiple kernel launches for sqr, sum, sqrt, div
    crate::ops::l2_normalize_scale(x, 1.0, eps)
}

/// Pad a tensor with zeros along a specific dimension
/// Supports 3D and 4D tensors
pub(super) fn pad_tensor(x: &Tensor, dim: usize, pad_size: usize) -> Result<Tensor> {
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
            return Err(paramecia_core::Error::Msg(
                "pad_tensor only supports 3D and 4D tensors".to_string(),
            ));
        }
    };

    Tensor::cat(&[x, &zeros], dim)
}

/// Create strictly lower triangular mask (1s below diagonal, 0s on and above)
/// This is the causal_mask in llama.cpp
/// Optimized version using candle's comparison operations
pub(super) fn create_causal_mask(size: usize, device: &Device, dtype: DType) -> Result<Tensor> {
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
pub(super) fn solve_lower_triangular(attn: &Tensor, causal_mask: &Tensor) -> Result<Tensor> {
    let (_, _, seq_len, seq_len2) = attn.dims4()?;

    if seq_len != seq_len2 {
        paramecia_core::bail!("solve_lower_triangular expects square matrix at dim 2,3");
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
pub(super) fn create_identity_mask(size: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let indices = Tensor::arange(0u32, size as u32, device)?;
    // After broadcast: col_idx[i][j] = j (column index at each position)
    let col_idx = indices.reshape((1, size))?.broadcast_as((size, size))?;
    // After broadcast: row_idx[i][j] = i (row index at each position)
    let row_idx = indices.reshape((size, 1))?.broadcast_as((size, size))?;

    // Diagonal is where row == col: i == j
    let mask = row_idx.eq(&col_idx)?;
    mask.to_dtype(dtype)?.reshape((1, 1, size, size))
}
