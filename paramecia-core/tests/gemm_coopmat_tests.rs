// Comprehensive tests comparing coopmat vs standard implementations for GEMM

use paramecia_core::{DType, Device, Result, Tensor};

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

// Helper macro to test GEMM operation with coopmat
macro_rules! test_gemm_coopmat {
    ($dtype:expr, $dtype_name:expr, $env_var:expr, $tolerance:expr) => {{
        use std::env;

        let device_ordinal = get_vulkan_device()?;
        let device = Device::new_vulkan(device_ordinal)?;
        let (m, k, n) = (64, 256, 64);

        // Create test matrices
        let lhs_data: Vec<f32> = (0..(m * k))
            .map(|v| (v as f32 / (m * k) as f32) * 2.0 - 1.0)
            .collect();
        let rhs_data: Vec<f32> = (0..(k * n))
            .map(|v| (v as f32 / (k * n) as f32) * 2.0 - 1.0)
            .collect();

        // Create tensors in the target dtype
        let lhs = Tensor::from_slice(&lhs_data, (m, k), &device)?;
        let rhs = Tensor::from_slice(&rhs_data, (k, n), &device)?;

        let lhs = lhs.to_dtype($dtype)?;
        let rhs = rhs.to_dtype($dtype)?;

        // Test with coopmat DISABLED (standard GEMM)
        env::set_var($env_var, "1");
        let result_std = lhs.matmul(&rhs)?;
        env::remove_var($env_var);

        // Test with coopmat ENABLED
        let result_coop = lhs.matmul(&rhs)?;

        // Convert both results to F32 for comparison
        let result_std_f32 = result_std.to_dtype(DType::F32)?;
        let result_coop_f32 = result_coop.to_dtype(DType::F32)?;

        // Compare results
        let diff = (&result_std_f32 - &result_coop_f32)?.abs()?;
        let max_diff: f32 = diff.flatten_all()?.max(0)?.to_vec0()?;
        let mean_diff: f32 = diff.flatten_all()?.mean(0)?.to_vec0()?;
        let std_abs = result_std_f32.abs()?;
        let rel_diff = (&diff / &std_abs)?;
        let rel_error: f32 = rel_diff.flatten_all()?.mean(0)?.to_vec0()?;

        println!("{} GEMM Coopmat vs Standard:", $dtype_name);
        println!("  Matrix dimensions: M={}, K={}, N={}", m, k, n);
        println!("  Max absolute diff: {}", max_diff);
        println!("  Mean absolute diff: {}", mean_diff);
        println!("  Relative error: {}", rel_error);
        println!(
            "  Standard (first 8): {:?}",
            &result_std_f32.flatten_all()?.to_vec1::<f32>()?[..8]
        );
        println!(
            "  Coopmat  (first 8): {:?}",
            &result_coop_f32.flatten_all()?.to_vec1::<f32>()?[..8]
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
            "{} GEMM coopmat differs too much: max_diff={}, tolerance={}",
            $dtype_name,
            max_diff,
            $tolerance
        );

        Ok::<_, paramecia_core::Error>(())
    }};
}

// Helper macro to test batched GEMM operation
macro_rules! test_gemm_batched_coopmat {
    ($dtype:expr, $dtype_name:expr, $env_var:expr, $tolerance:expr) => {{
        use std::env;

        let device_ordinal = get_vulkan_device()?;
        let device = Device::new_vulkan(device_ordinal)?;
        let (batch, m, k, n) = (4, 32, 128, 32);

        // Create test matrices
        let lhs_data: Vec<f32> = (0..(batch * m * k))
            .map(|v| (v as f32 / (batch * m * k) as f32) * 2.0 - 1.0)
            .collect();
        let rhs_data: Vec<f32> = (0..(batch * k * n))
            .map(|v| (v as f32 / (batch * k * n) as f32) * 2.0 - 1.0)
            .collect();

        // Create tensors in the target dtype
        let lhs = Tensor::from_slice(&lhs_data, (batch, m, k), &device)?;
        let rhs = Tensor::from_slice(&rhs_data, (batch, k, n), &device)?;

        let lhs = lhs.to_dtype($dtype)?;
        let rhs = rhs.to_dtype($dtype)?;

        // Test with coopmat DISABLED (standard GEMM)
        env::set_var($env_var, "1");
        let result_std = lhs.matmul(&rhs)?;
        env::remove_var($env_var);

        // Test with coopmat ENABLED
        let result_coop = lhs.matmul(&rhs)?;

        // Convert both results to F32 for comparison
        let result_std_f32 = result_std.to_dtype(DType::F32)?;
        let result_coop_f32 = result_coop.to_dtype(DType::F32)?;

        // Compare results
        let diff = (&result_std_f32 - &result_coop_f32)?.abs()?;
        let max_diff: f32 = diff.flatten_all()?.max(0)?.to_vec0()?;
        let mean_diff: f32 = diff.flatten_all()?.mean(0)?.to_vec0()?;

        println!("{} Batched GEMM Coopmat vs Standard:", $dtype_name);
        println!(
            "  Matrix dimensions: B={}, M={}, K={}, N={}",
            batch, m, k, n
        );
        println!("  Max absolute diff: {}", max_diff);
        println!("  Mean absolute diff: {}", mean_diff);

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
            "{} Batched GEMM coopmat differs too much: max_diff={}, tolerance={}",
            $dtype_name,
            max_diff,
            $tolerance
        );

        Ok::<_, paramecia_core::Error>(())
    }};
}

// ============================================================================
// F32 GEMM Tests
// ============================================================================

#[cfg(feature = "vulkan")]
#[test]
fn test_f32_gemm_coopmat() -> Result<()> {
    test_gemm_coopmat!(
        DType::F32,
        "F32",
        "PARAMECIA_DISABLE_COOPMAT_F32_GEMM",
        0.05
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_f32_gemm_batched_coopmat() -> Result<()> {
    test_gemm_batched_coopmat!(
        DType::F32,
        "F32",
        "PARAMECIA_DISABLE_COOPMAT_F32_GEMM",
        0.05
    )
}

// ============================================================================
// F16 GEMM Tests
// ============================================================================

#[cfg(feature = "vulkan")]
#[test]
fn test_f16_gemm_coopmat() -> Result<()> {
    test_gemm_coopmat!(
        DType::F16,
        "F16",
        "PARAMECIA_DISABLE_COOPMAT_F16_GEMM",
        0.05
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_f16_gemm_batched_coopmat() -> Result<()> {
    test_gemm_batched_coopmat!(
        DType::F16,
        "F16",
        "PARAMECIA_DISABLE_COOPMAT_F16_GEMM",
        0.05
    )
}

// ============================================================================
// BF16 GEMM Tests
// ============================================================================

#[cfg(feature = "vulkan")]
#[test]
fn test_bf16_gemm_coopmat() -> Result<()> {
    test_gemm_coopmat!(
        DType::BF16,
        "BF16",
        "PARAMECIA_DISABLE_COOPMAT_BF16_GEMM",
        0.1
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_bf16_gemm_batched_coopmat() -> Result<()> {
    test_gemm_batched_coopmat!(
        DType::BF16,
        "BF16",
        "PARAMECIA_DISABLE_COOPMAT_BF16_GEMM",
        0.1
    )
}
