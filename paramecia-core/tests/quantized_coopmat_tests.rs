// Comprehensive tests comparing coopmat vs standard implementations
// for all quantization types and operations
#![cfg(feature = "vulkan")]

use paramecia_core::quantized::GgmlDType;
use paramecia_core::{quantized, Device, Module, Result, Tensor};

// Helper function to get Vulkan device ordinal from environment variable
// Defaults to 0 if not set. Set PARAMECIA_VK_DEVICE=N to use device N.
fn get_vulkan_device() -> Result<usize> {
    match std::env::var("PARAMECIA_VK_DEVICE") {
        Ok(val) => val.parse::<usize>().map_err(|e| {
            paramecia_core::Error::Msg(format!("Invalid PARAMECIA_VK_DEVICE '{}': {}", val, e))
        }),
        Err(_) => Ok(0),
    }
}

// Helper macro to test matmul operation
macro_rules! test_matmul_coopmat {
    ($dtype:expr, $dtype_name:expr, $env_var:expr, $tolerance:expr) => {{
        use std::env;

        let device_ordinal = get_vulkan_device()?;
        let device = Device::new_vulkan(device_ordinal)?;
        let (m, k, n) = (16, 256, 16);

        let lhs_data: Vec<f32> = (0..(m * k))
            .map(|v| (v as f32 / (m * k) as f32) * 2.0 - 1.0)
            .collect();
        let rhs_data: Vec<f32> = (0..(k * n))
            .map(|v| (v as f32 / (k * n) as f32) * 2.0 - 1.0)
            .collect();

        let lhs = Tensor::from_slice(&lhs_data, (m, k), &device)?;
        let rhs = Tensor::from_slice(&rhs_data, (k, n), &device)?;
        let rhs_t = rhs.t()?;

        // Test with coopmat DISABLED
        env::set_var($env_var, "1");
        let qtensor_std = quantized::QTensor::quantize(&rhs_t, $dtype)?;
        let matmul_std = quantized::QMatMul::from_qtensor(qtensor_std)?;
        let result_std = matmul_std.forward(&lhs)?;
        env::remove_var($env_var);

        // Test with coopmat ENABLED
        env::set_var("PARAMECIA_ENABLE_COOPMAT", "1");
        let qtensor_coop = quantized::QTensor::quantize(&rhs_t, $dtype)?;
        let matmul_coop = quantized::QMatMul::from_qtensor(qtensor_coop)?;
        let result_coop = matmul_coop.forward(&lhs)?;
        env::remove_var("PARAMECIA_ENABLE_COOPMAT");

        // Compare results
        let diff = (&result_std - &result_coop)?.abs()?;
        let max_diff: f32 = diff.flatten_all()?.max(0)?.to_vec0()?;
        let mean_diff: f32 = diff.flatten_all()?.mean(0)?.to_vec0()?;
        let std_abs = result_std.abs()?;
        let rel_diff = (&diff / &std_abs)?;
        let rel_error: f32 = rel_diff.flatten_all()?.mean(0)?.to_vec0()?;

        println!("{} Matmul Coopmat vs Standard:", $dtype_name);
        println!("  Max absolute diff: {}", max_diff);
        println!("  Mean absolute diff: {}", mean_diff);
        println!("  Relative error: {}", rel_error);
        println!(
            "  Standard (first 8): {:?}",
            &result_std.flatten_all()?.to_vec1::<f32>()?[..8]
        );
        println!(
            "  Coopmat  (first 8): {:?}",
            &result_coop.flatten_all()?.to_vec1::<f32>()?[..8]
        );

        if max_diff >= $tolerance {
            println!(
                "  ⚠ WARNING: Exceeds tolerance ({} >= {})",
                max_diff, $tolerance
            );
        } else {
            println!("  ✓ PASS");
        }
        println!();

        assert!(
            max_diff < $tolerance,
            "{} matmul coopmat differs too much: max_diff={}, tolerance={}",
            $dtype_name,
            max_diff,
            $tolerance
        );

        Ok::<_, paramecia_core::Error>(())
    }};
}

// Helper macro to test indexed_moe_forward operation
macro_rules! test_moe_forward_coopmat {
    ($dtype:expr, $dtype_name:expr, $env_var:expr, $tolerance:expr) => {{
        use std::env;

        let device_ordinal = get_vulkan_device()?;
        let device = Device::new_vulkan(device_ordinal)?;

        // Expert weights: [num_experts=4, n=64, k=256]
        let (num_experts, n, k) = (4usize, 64usize, 256usize);
        let (batch, topk) = (2usize, 2usize);

        // Create expert weights
        let expert_data: Vec<f32> = (0..(num_experts * n * k))
            .map(|v| (v as f32 / (num_experts * n * k) as f32) * 2.0 - 1.0)
            .collect();
        let expert_weights = Tensor::from_slice(&expert_data, (num_experts, n, k), &device)?;

        // Create input: [batch=2, topk=2, k=256]
        let x_data: Vec<f32> = (0..(batch * topk * k))
            .map(|v| (v as f32 / (batch * topk * k) as f32) * 2.0 - 1.0)
            .collect();
        let x = Tensor::from_slice(&x_data, (batch, topk, k), &device)?;

        // Expert IDs: [batch=2, topk=2]
        let ids = Tensor::from_slice(&[0u32, 1, 2, 3], (batch, topk), &device)?;

        // Test with coopmat DISABLED
        env::set_var($env_var, "1");
        let qexperts_std = quantized::QTensor::quantize(&expert_weights, $dtype)?;
        let result_std = qexperts_std.indexed_moe_forward(&x, &ids)?;
        env::remove_var($env_var);

        // Test with coopmat ENABLED
        let qexperts_coop = quantized::QTensor::quantize(&expert_weights, $dtype)?;
        let result_coop = qexperts_coop.indexed_moe_forward(&x, &ids)?;

        // Compare results
        let diff = (&result_std - &result_coop)?.abs()?;
        let max_diff: f32 = diff.flatten_all()?.max(0)?.to_vec0()?;
        let mean_diff: f32 = diff.flatten_all()?.mean(0)?.to_vec0()?;
        let std_abs = result_std.abs()?;
        let rel_diff = (&diff / &std_abs)?;
        let rel_error: f32 = rel_diff.flatten_all()?.mean(0)?.to_vec0()?;

        println!("{} MoE Forward Coopmat vs Standard:", $dtype_name);
        println!("  Max absolute diff: {}", max_diff);
        println!("  Mean absolute diff: {}", mean_diff);
        println!("  Relative error: {}", rel_error);
        println!(
            "  Standard (first 8): {:?}",
            &result_std.flatten_all()?.to_vec1::<f32>()?[..8]
        );
        println!(
            "  Coopmat  (first 8): {:?}",
            &result_coop.flatten_all()?.to_vec1::<f32>()?[..8]
        );

        if max_diff >= $tolerance {
            println!(
                "  ⚠ WARNING: Exceeds tolerance ({} >= {})",
                max_diff, $tolerance
            );
        } else {
            println!("  ✓ PASS");
        }
        println!();

        assert!(
            max_diff < $tolerance,
            "{} moe_forward coopmat differs too much: max_diff={}, tolerance={}",
            $dtype_name,
            max_diff,
            $tolerance
        );

        Ok::<_, paramecia_core::Error>(())
    }};
}

// Helper macro to test indexed_moe_gate_up_swiglu operation
macro_rules! test_moe_gate_up_swiglu_coopmat {
    ($dtype:expr, $dtype_name:expr, $env_var:expr, $tolerance:expr) => {{
        use std::env;

        let device_ordinal = get_vulkan_device()?;
        let device = Device::new_vulkan(device_ordinal)?;

        // Gate/Up weights: [num_experts=4, n_ff=128, k=256]
        let (num_experts, n_ff, k) = (4usize, 128usize, 256usize);
        let (batch, topk) = (2usize, 2usize);

        // Create gate and up weights
        let gate_data: Vec<f32> = (0..(num_experts * n_ff * k))
            .map(|v| (v as f32 / (num_experts * n_ff * k) as f32) * 2.0 - 1.0)
            .collect();
        let up_data: Vec<f32> = (0..(num_experts * n_ff * k))
            .map(|v| ((v + 1000) as f32 / (num_experts * n_ff * k) as f32) * 2.0 - 1.0)
            .collect();

        let gate_weights = Tensor::from_slice(&gate_data, (num_experts, n_ff, k), &device)?;
        let up_weights = Tensor::from_slice(&up_data, (num_experts, n_ff, k), &device)?;

        // Create input: [batch=2, topk=2, k=256]
        let x_data: Vec<f32> = (0..(batch * topk * k))
            .map(|v| (v as f32 / (batch * topk * k) as f32) * 2.0 - 1.0)
            .collect();
        let x = Tensor::from_slice(&x_data, (batch, topk, k), &device)?;

        // Expert IDs: [batch=2, topk=2]
        let ids = Tensor::from_slice(&[0u32, 1, 2, 3], (batch, topk), &device)?;

        // Test with coopmat DISABLED
        env::set_var($env_var, "1");
        let qgate_std = quantized::QTensor::quantize(&gate_weights, $dtype)?;
        let qup_std = quantized::QTensor::quantize(&up_weights, $dtype)?;
        let result_std =
            quantized::QTensor::indexed_moe_gate_up_swiglu(&qgate_std, &qup_std, &x, &ids)?;
        env::remove_var($env_var);

        // Test with coopmat ENABLED
        let qgate_coop = quantized::QTensor::quantize(&gate_weights, $dtype)?;
        let qup_coop = quantized::QTensor::quantize(&up_weights, $dtype)?;
        let result_coop =
            quantized::QTensor::indexed_moe_gate_up_swiglu(&qgate_coop, &qup_coop, &x, &ids)?;

        // Compare results
        let diff = (&result_std - &result_coop)?.abs()?;
        let max_diff: f32 = diff.flatten_all()?.max(0)?.to_vec0()?;
        let mean_diff: f32 = diff.flatten_all()?.mean(0)?.to_vec0()?;
        let std_abs = result_std.abs()?;
        let rel_diff = (&diff / &std_abs)?;
        let rel_error: f32 = rel_diff.flatten_all()?.mean(0)?.to_vec0()?;

        println!("{} MoE Gate+Up+SwiGLU Coopmat vs Standard:", $dtype_name);
        println!("  Max absolute diff: {}", max_diff);
        println!("  Mean absolute diff: {}", mean_diff);
        println!("  Relative error: {}", rel_error);
        println!(
            "  Standard (first 8): {:?}",
            &result_std.flatten_all()?.to_vec1::<f32>()?[..8]
        );
        println!(
            "  Coopmat  (first 8): {:?}",
            &result_coop.flatten_all()?.to_vec1::<f32>()?[..8]
        );

        if max_diff >= $tolerance {
            println!(
                "  ⚠ WARNING: Exceeds tolerance ({} >= {})",
                max_diff, $tolerance
            );
        } else {
            println!("  ✓ PASS");
        }
        println!();

        assert!(
            max_diff < $tolerance,
            "{} gate_up_swiglu coopmat differs too much: max_diff={}, tolerance={}",
            $dtype_name,
            max_diff,
            $tolerance
        );

        Ok::<_, paramecia_core::Error>(())
    }};
}

// ============================================================================
// Matmul Tests
// ============================================================================

#[cfg(feature = "vulkan")]
#[test]
fn test_q8_0_matmul_coopmat() -> Result<()> {
    test_matmul_coopmat!(
        GgmlDType::Q8_0,
        "Q8_0",
        "PARAMECIA_DISABLE_COOPMAT_Q8_0_MATMUL",
        0.5
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_q6k_matmul_coopmat() -> Result<()> {
    test_matmul_coopmat!(
        GgmlDType::Q6K,
        "Q6K",
        "PARAMECIA_DISABLE_COOPMAT_Q6_K_MATMUL",
        0.5
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_q5k_matmul_coopmat() -> Result<()> {
    test_matmul_coopmat!(
        GgmlDType::Q5K,
        "Q5K",
        "PARAMECIA_DISABLE_COOPMAT_Q5_K_MATMUL",
        0.1
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_q4k_matmul_coopmat() -> Result<()> {
    test_matmul_coopmat!(
        GgmlDType::Q4K,
        "Q4K",
        "PARAMECIA_DISABLE_COOPMAT_Q4_K_MATMUL",
        0.15
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_q3k_matmul_coopmat() -> Result<()> {
    test_matmul_coopmat!(
        GgmlDType::Q3K,
        "Q3K",
        "PARAMECIA_DISABLE_COOPMAT_Q3_K_MATMUL",
        0.25
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_q2k_matmul_coopmat() -> Result<()> {
    test_matmul_coopmat!(
        GgmlDType::Q2K,
        "Q2K",
        "PARAMECIA_DISABLE_COOPMAT_Q2_K_MATMUL",
        0.3
    )
}

// ============================================================================
// MoE Forward Tests
// ============================================================================

#[cfg(feature = "vulkan")]
#[test]
fn test_q8_0_moe_forward_coopmat() -> Result<()> {
    test_moe_forward_coopmat!(
        GgmlDType::Q8_0,
        "Q8_0",
        "PARAMECIA_DISABLE_COOPMAT_Q8_0_MOE_FORWARD",
        0.05
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_q6k_moe_forward_coopmat() -> Result<()> {
    test_moe_forward_coopmat!(
        GgmlDType::Q6K,
        "Q6K",
        "PARAMECIA_DISABLE_COOPMAT_Q6_K_MOE_FORWARD",
        0.1
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_q5k_moe_forward_coopmat() -> Result<()> {
    test_moe_forward_coopmat!(
        GgmlDType::Q5K,
        "Q5K",
        "PARAMECIA_DISABLE_COOPMAT_Q5_K_MOE_FORWARD",
        0.1
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_q4k_moe_forward_coopmat() -> Result<()> {
    test_moe_forward_coopmat!(
        GgmlDType::Q4K,
        "Q4K",
        "PARAMECIA_DISABLE_COOPMAT_Q4_K_MOE_FORWARD",
        0.15
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_q3k_moe_forward_coopmat() -> Result<()> {
    test_moe_forward_coopmat!(
        GgmlDType::Q3K,
        "Q3K",
        "PARAMECIA_DISABLE_COOPMAT_Q3_K_MOE_FORWARD",
        0.2
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_q2k_moe_forward_coopmat() -> Result<()> {
    test_moe_forward_coopmat!(
        GgmlDType::Q2K,
        "Q2K",
        "PARAMECIA_DISABLE_COOPMAT_Q2_K_MOE_FORWARD",
        0.3
    )
}

// ============================================================================
// MoE Gate+Up+SwiGLU Tests
// ============================================================================

#[cfg(feature = "vulkan")]
#[test]
fn test_q8_0_moe_gate_up_swiglu_coopmat() -> Result<()> {
    test_moe_gate_up_swiglu_coopmat!(
        GgmlDType::Q8_0,
        "Q8_0",
        "PARAMECIA_DISABLE_COOPMAT_Q8_0_MOE_GATE_UP",
        0.05
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_q6k_moe_gate_up_swiglu_coopmat() -> Result<()> {
    test_moe_gate_up_swiglu_coopmat!(
        GgmlDType::Q6K,
        "Q6K",
        "PARAMECIA_DISABLE_COOPMAT_Q6_K_MOE_GATE_UP",
        0.1
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_q5k_moe_gate_up_swiglu_coopmat() -> Result<()> {
    test_moe_gate_up_swiglu_coopmat!(
        GgmlDType::Q5K,
        "Q5K",
        "PARAMECIA_DISABLE_COOPMAT_Q5_K_MOE_GATE_UP",
        0.1
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_q4k_moe_gate_up_swiglu_coopmat() -> Result<()> {
    test_moe_gate_up_swiglu_coopmat!(
        GgmlDType::Q4K,
        "Q4K",
        "PARAMECIA_DISABLE_COOPMAT_Q4_K_MOE_GATE_UP",
        0.15
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_q3k_moe_gate_up_swiglu_coopmat() -> Result<()> {
    test_moe_gate_up_swiglu_coopmat!(
        GgmlDType::Q3K,
        "Q3K",
        "PARAMECIA_DISABLE_COOPMAT_Q3_K_MOE_GATE_UP",
        0.2
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_q2k_moe_gate_up_swiglu_coopmat() -> Result<()> {
    test_moe_gate_up_swiglu_coopmat!(
        GgmlDType::Q2K,
        "Q2K",
        "PARAMECIA_DISABLE_COOPMAT_Q2_K_MOE_GATE_UP",
        0.3
    )
}
