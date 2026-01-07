// Hierarchical Tiled MatMul Tests
// Tests for verifying correctness of tiled matmul kernels vs non-tiled implementations

use paramecia_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Module, Result, Tensor,
};

/// Helper to get a Vulkan device, or skip test if not available
fn get_vulkan_device() -> Result<Device> {
    match Device::new_vulkan(0) {
        Ok(dev) => Ok(dev),
        Err(_) => {
            eprintln!("Vulkan not available, skipping tiled matmul test");
            std::process::exit(0);
        }
    }
}

/// Helper to compare two tensors with relative error tolerance
/// Uses relative error for large values, absolute error for small values
fn assert_close(
    actual: &Tensor,
    expected: &Tensor,
    max_rel_error: f32,
    test_name: &str,
) -> Result<()> {
    let actual_flat: Vec<f32> = actual.flatten_all()?.to_vec1()?;
    let expected_flat: Vec<f32> = expected.flatten_all()?.to_vec1()?;

    assert_eq!(
        actual_flat.len(),
        expected_flat.len(),
        "Tensor size mismatch in {}",
        test_name
    );

    let mut max_rel_err = 0.0f32;
    let mut max_abs_err = 0.0f32;
    let mut sum_rel_err = 0.0f32;
    let mut max_err_idx = 0;

    const ABS_ERR_THRESHOLD: f32 = 0.01; // Use absolute error below this threshold

    for (i, (&a, &e)) in actual_flat.iter().zip(expected_flat.iter()).enumerate() {
        let abs_err = (a - e).abs();
        max_abs_err = max_abs_err.max(abs_err);

        // Use relative error for larger values, absolute error for small values
        let err = if e.abs() > ABS_ERR_THRESHOLD {
            abs_err / e.abs()
        } else {
            abs_err // Absolute error for values near zero
        };

        if err > max_rel_err {
            max_rel_err = err;
            max_err_idx = i;
        }
        sum_rel_err += err;
    }

    let mean_rel_err = sum_rel_err / actual_flat.len() as f32;

    if max_rel_err > max_rel_error {
        eprintln!("\n=== MISMATCH in {} ===", test_name);
        eprintln!(
            "Max relative error: {} (threshold: {})",
            max_rel_err, max_rel_error
        );
        eprintln!("Max absolute error: {}", max_abs_err);
        eprintln!("Mean relative error: {}", mean_rel_err);
        eprintln!("\nWorst mismatch at index {}:", max_err_idx);
        eprintln!("  Actual:   {}", actual_flat[max_err_idx]);
        eprintln!("  Expected: {}", expected_flat[max_err_idx]);
        eprintln!("\nFirst 16 values:");
        eprintln!("Actual:   {:?}", &actual_flat[..16.min(actual_flat.len())]);
        eprintln!(
            "Expected: {:?}",
            &expected_flat[..16.min(expected_flat.len())]
        );

        panic!(
            "Relative error {} exceeds threshold {}",
            max_rel_err, max_rel_error
        );
    }

    Ok(())
}

/// Test tiled Q8_0 matmul against non-tiled implementation
fn test_tiled_vs_nontiled_q8_0(m: usize, n: usize, k: usize, max_rel_error: f32) -> Result<()> {
    let device = get_vulkan_device()?;

    println!(
        "\n=== Testing Q8_0 Tiled vs Non-Tiled: m={}, n={}, k={} ===",
        m, n, k
    );

    // Create activation tensor (m×k)
    let act_data: Vec<f32> = (0..(m * k))
        .map(|i| (i as f32 / (m * k) as f32) * 2.0 - 1.0) // Range: [-1, 1]
        .collect();
    let activations = Tensor::from_slice(&act_data, (m, k), &device)?;

    // Create weight tensor (n×k) and quantize to Q8_0
    let weight_data: Vec<f32> = (0..(n * k))
        .map(|i| ((i * 7 + 13) as f32 / (n * k) as f32) * 2.0 - 1.0) // Different pattern
        .collect();
    let weights_f32 = Tensor::from_slice(&weight_data, (n, k), &device)?;

    // Quantize weights to Q8_0
    let qtensor = QTensor::quantize(&weights_f32, GgmlDType::Q8_0)?;
    let qmatmul = QMatMul::from_qtensor(qtensor)?;

    // Run with NON-TILED kernel (disable tiled via env var)
    std::env::set_var("PARAMECIA_DISABLE_TILED", "1");
    let result_nontiled = qmatmul.forward(&activations)?;
    std::env::remove_var("PARAMECIA_DISABLE_TILED");

    // Run with TILED kernel (enable tiled via env var)
    std::env::set_var("PARAMECIA_ENABLE_TILED", "1");
    let result_tiled = qmatmul.forward(&activations)?;
    std::env::remove_var("PARAMECIA_ENABLE_TILED");

    // Compare results
    let test_name = format!("Q8_0_{}x{}x{}", m, n, k);
    assert_close(&result_tiled, &result_nontiled, max_rel_error, &test_name)?;

    println!(
        "✓ Test passed: Q8_0 tiled matches non-tiled (m={}, n={}, k={})",
        m, n, k
    );

    Ok(())
}

/// Test various matrix sizes including edge cases
#[test]
#[cfg(feature = "vulkan")]
fn test_tiled_q8_0_various_sizes() -> Result<()> {
    // Maximum relative error tolerance for quantized operations
    // Higher tolerance for coopmat (fp16 precision)
    let using_coopmat = std::env::var("PARAMECIA_FORCE_COOPMAT").is_ok();
    let max_error = if using_coopmat { 0.15 } else { 0.01 }; // 15% for fp16, 1% for fp32

    // Test 1: Decode workload (m=1)
    // Note: Tiled kernel may not be optimal here, but should still be correct
    test_tiled_vs_nontiled_q8_0(1, 4096, 4096, max_error)?;

    // Test 2: Small batch (m=4) - decode-optimized tile size
    test_tiled_vs_nontiled_q8_0(4, 4096, 4096, max_error)?;

    // Test 3: Small prefill (m=16)
    test_tiled_vs_nontiled_q8_0(16, 4096, 4096, max_error)?;

    // Test 4: Medium prefill (m=32)
    test_tiled_vs_nontiled_q8_0(32, 2048, 2048, max_error)?;

    // Test 5: Large prefill (m=64)
    test_tiled_vs_nontiled_q8_0(64, 1024, 1024, max_error)?;

    // Test 6: Non-multiples of tile size (edge case)
    // BM=16, BN=64, BK=32, so test sizes that don't divide evenly
    // Note: k must be multiple of 32 (Q8_0 block size)
    test_tiled_vs_nontiled_q8_0(17, 100, 128, max_error)?;

    // Test 7: Very small matrices
    test_tiled_vs_nontiled_q8_0(8, 64, 64, max_error)?;

    // Test 8: Rectangular matrices (tall)
    test_tiled_vs_nontiled_q8_0(128, 256, 512, max_error)?;

    // Test 9: Rectangular matrices (wide)
    test_tiled_vs_nontiled_q8_0(16, 8192, 512, max_error)?;

    println!("\n✅ All tiled Q8_0 matmul tests passed!");

    Ok(())
}

/// Test with different activation patterns to ensure robustness
#[test]
#[cfg(feature = "vulkan")]
fn test_tiled_q8_0_activation_patterns() -> Result<()> {
    let device = get_vulkan_device()?;
    let max_error = 0.01;

    let (m, n, k) = (32, 1024, 1024);

    println!("\n=== Testing Q8_0 Tiled with Different Activation Patterns ===");

    // Pattern 1: All zeros
    let act_zeros = Tensor::zeros((m, k), DType::F32, &device)?;

    // Pattern 2: All ones
    let act_ones = Tensor::ones((m, k), DType::F32, &device)?;

    // Pattern 3: Random values
    let act_random: Vec<f32> = (0..(m * k))
        .map(|i| {
            let x = (i as f32 * 0.1).sin();
            x * 2.0 // Range: [-2, 2]
        })
        .collect();
    let act_random = Tensor::from_slice(&act_random, (m, k), &device)?;

    // Create and quantize weights
    let weight_data: Vec<f32> = (0..(n * k))
        .map(|i| ((i as f32 / (n * k) as f32) * 2.0 - 1.0))
        .collect();
    let weights_f32 = Tensor::from_slice(&weight_data, (n, k), &device)?;
    let qtensor = QTensor::quantize(&weights_f32, GgmlDType::Q8_0)?;
    let qmatmul = QMatMul::from_qtensor(qtensor)?;

    // Test each activation pattern
    for (pattern_name, activations) in [
        ("zeros", &act_zeros),
        ("ones", &act_ones),
        ("random", &act_random),
    ] {
        println!("  Testing pattern: {}", pattern_name);

        // Non-tiled
        std::env::set_var("PARAMECIA_DISABLE_TILED", "1");
        let result_nontiled = qmatmul.forward(activations)?;
        std::env::remove_var("PARAMECIA_DISABLE_TILED");

        // Tiled
        std::env::set_var("PARAMECIA_ENABLE_TILED", "1");
        let result_tiled = qmatmul.forward(activations)?;
        std::env::remove_var("PARAMECIA_ENABLE_TILED");

        // For zeros, results should be exactly zero
        if pattern_name == "zeros" {
            let sum: f32 = result_tiled.abs()?.sum_all()?.to_scalar()?;
            assert_eq!(sum, 0.0, "Zeros pattern should produce zero output");
        }

        // Compare
        let test_name = format!("Q8_0_pattern_{}", pattern_name);
        assert_close(&result_tiled, &result_nontiled, max_error, &test_name)?;
    }

    println!("✓ All activation pattern tests passed!");

    Ok(())
}

/// Benchmark-style test to show performance characteristics
/// (This doesn't fail, just reports timing)
#[test]
#[cfg(feature = "vulkan")]
#[ignore] // Run with: cargo test --features vulkan -- --ignored
fn benchmark_tiled_vs_nontiled_q8_0() -> Result<()> {
    let device = get_vulkan_device()?;

    println!("\n=== Performance Comparison: Tiled vs Non-Tiled Q8_0 ===");
    println!("(This is informational only, not a pass/fail test)\n");

    for (m, n, k) in [
        (1, 4096, 4096),  // Decode
        (4, 4096, 4096),  // Small batch
        (16, 4096, 4096), // Small prefill
        (32, 4096, 4096), // Medium prefill
        (64, 4096, 4096), // Large prefill
    ] {
        let act_data: Vec<f32> = (0..(m * k))
            .map(|i| (i as f32 / (m * k) as f32) * 2.0 - 1.0)
            .collect();
        let activations = Tensor::from_slice(&act_data, (m, k), &device)?;

        let weight_data: Vec<f32> = (0..(n * k))
            .map(|i| ((i as f32 / (n * k) as f32) * 2.0 - 1.0))
            .collect();
        let weights_f32 = Tensor::from_slice(&weight_data, (n, k), &device)?;
        let qtensor = QTensor::quantize(&weights_f32, GgmlDType::Q8_0)?;
        let qmatmul = QMatMul::from_qtensor(qtensor)?;

        // Warmup
        for _ in 0..3 {
            let _ = qmatmul.forward(&activations)?;
        }

        // Benchmark non-tiled
        std::env::set_var("PARAMECIA_DISABLE_TILED", "1");
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = qmatmul.forward(&activations)?;
        }
        // Force GPU completion (flush)
        if let Device::Vulkan(vdev) = &device {
            vdev.flush()?;
        }
        let nontiled_time = start.elapsed().as_micros() as f64 / 10.0;
        std::env::remove_var("PARAMECIA_DISABLE_TILED");

        // Benchmark tiled
        std::env::set_var("PARAMECIA_ENABLE_TILED", "1");
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = qmatmul.forward(&activations)?;
        }
        if let Device::Vulkan(vdev) = &device {
            vdev.flush()?;
        }
        let tiled_time = start.elapsed().as_micros() as f64 / 10.0;
        std::env::remove_var("PARAMECIA_ENABLE_TILED");

        let speedup = nontiled_time / tiled_time;
        println!(
            "m={:3} n={:4} k={:4}  |  Non-tiled: {:7.1}µs  |  Tiled: {:7.1}µs  |  Speedup: {:.2}×",
            m, n, k, nontiled_time, tiled_time, speedup
        );
    }

    println!("\n(Note: Actual speedup depends on GPU, driver, and workload)");

    Ok(())
}

/// Test batched operations (3D tensors)
#[test]
#[cfg(feature = "vulkan")]
fn test_tiled_q8_0_batched() -> Result<()> {
    let device = get_vulkan_device()?;
    let max_error = 0.01;

    let (batch, m, n, k) = (4, 16, 512, 512);

    println!(
        "\n=== Testing Q8_0 Tiled Batched: batch={}, m={}, n={}, k={} ===",
        batch, m, n, k
    );

    // Create batched activations (batch×m×k)
    let act_data: Vec<f32> = (0..(batch * m * k))
        .map(|i| (i as f32 / (batch * m * k) as f32) * 2.0 - 1.0)
        .collect();
    let activations = Tensor::from_slice(&act_data, (batch, m, k), &device)?;

    // Create weights (n×k)
    let weight_data: Vec<f32> = (0..(n * k))
        .map(|i| ((i as f32 / (n * k) as f32) * 2.0 - 1.0))
        .collect();
    let weights_f32 = Tensor::from_slice(&weight_data, (n, k), &device)?;
    let qtensor = QTensor::quantize(&weights_f32, GgmlDType::Q8_0)?;
    let qmatmul = QMatMul::from_qtensor(qtensor)?;

    // Non-tiled
    std::env::set_var("PARAMECIA_DISABLE_TILED", "1");
    let result_nontiled = qmatmul.forward(&activations)?;
    std::env::remove_var("PARAMECIA_DISABLE_TILED");

    // Tiled
    std::env::set_var("PARAMECIA_ENABLE_TILED", "1");
    let result_tiled = qmatmul.forward(&activations)?;
    std::env::remove_var("PARAMECIA_ENABLE_TILED");

    // Compare - need to reshape for comparison
    let nontiled_flat = result_nontiled.flatten_all()?;
    let tiled_flat = result_tiled.flatten_all()?;

    let diff = (&tiled_flat - &nontiled_flat)?;
    let max_abs_diff: f32 = diff.abs()?.max(0)?.to_scalar()?;
    let max_nontiled: f32 = nontiled_flat.abs()?.max(0)?.to_scalar()?;
    let rel_error = max_abs_diff / max_nontiled.max(1e-8);

    assert!(
        rel_error < max_error,
        "Batched test failed: relative error {} exceeds threshold {}",
        rel_error,
        max_error
    );

    println!(
        "✓ Batched test passed: Q8_0 tiled matches non-tiled (batch={}, m={}, n={}, k={})",
        batch, m, n, k
    );

    Ok(())
}

/// Generic helper to test tiled matmul for any K-quant format
fn test_tiled_vs_nontiled_kquant(
    dtype: GgmlDType,
    m: usize,
    n: usize,
    k: usize,
    max_rel_error: f32,
) -> Result<()> {
    let device = get_vulkan_device()?;

    let dtype_name = format!("{:?}", dtype);
    println!(
        "\n=== Testing {} Tiled vs Non-Tiled: m={}, n={}, k={} ===",
        dtype_name, m, n, k
    );

    // Create activation tensor (m×k)
    let act_data: Vec<f32> = (0..(m * k))
        .map(|i| (i as f32 / (m * k) as f32) * 2.0 - 1.0) // Range: [-1, 1]
        .collect();
    let activations = Tensor::from_slice(&act_data, (m, k), &device)?;

    // Create weight tensor (n×k) and quantize
    let weight_data: Vec<f32> = (0..(n * k))
        .map(|i| ((i * 7 + 13) as f32 / (n * k) as f32) * 2.0 - 1.0) // Different pattern
        .collect();
    let weights_f32 = Tensor::from_slice(&weight_data, (n, k), &device)?;

    // Quantize weights to specified dtype
    let qtensor = QTensor::quantize(&weights_f32, dtype)?;
    let qmatmul = QMatMul::from_qtensor(qtensor)?;

    // Run with NON-TILED kernel (disable tiled via env var)
    std::env::set_var("PARAMECIA_DISABLE_TILED", "1");
    let result_nontiled = qmatmul.forward(&activations)?;
    std::env::remove_var("PARAMECIA_DISABLE_TILED");

    // Run with TILED kernel (enable tiled via env var)
    std::env::set_var("PARAMECIA_ENABLE_TILED", "1");
    let result_tiled = qmatmul.forward(&activations)?;
    std::env::remove_var("PARAMECIA_ENABLE_TILED");

    // Compare results
    let test_name = format!("{}_{}x{}x{}", dtype_name, m, n, k);
    assert_close(&result_tiled, &result_nontiled, max_rel_error, &test_name)?;

    println!(
        "✓ Test passed: {} tiled matches non-tiled (m={}, n={}, k={})",
        dtype_name, m, n, k
    );

    Ok(())
}

/// Test tiled Q4_K matmul (256 elements per block, 144 bytes)
#[test]
#[cfg(feature = "vulkan")]
fn test_tiled_q4_k_various_sizes() -> Result<()> {
    // Q4_K has 256 elements per block, so k must be multiple of 256
    let max_error = 0.02; // K-quants may have slightly higher quantization error

    println!("\n=== Testing Q4_K Tiled Kernels ===");

    // Decode workload
    test_tiled_vs_nontiled_kquant(GgmlDType::Q4K, 1, 4096, 4096, max_error)?;

    // Small batch
    test_tiled_vs_nontiled_kquant(GgmlDType::Q4K, 4, 4096, 4096, max_error)?;

    // Medium prefill
    test_tiled_vs_nontiled_kquant(GgmlDType::Q4K, 16, 2048, 2048, max_error)?;

    // Large prefill
    test_tiled_vs_nontiled_kquant(GgmlDType::Q4K, 32, 1024, 1024, max_error)?;

    // Edge case: non-multiples of tile size (k must be multiple of 256)
    test_tiled_vs_nontiled_kquant(GgmlDType::Q4K, 17, 100, 512, max_error)?;

    println!("\n✅ All tiled Q4_K matmul tests passed!");

    Ok(())
}

/// Test tiled Q6_K matmul (256 elements per block, 210 bytes)
#[test]
#[cfg(feature = "vulkan")]
fn test_tiled_q6_k_various_sizes() -> Result<()> {
    let max_error = 0.02;

    println!("\n=== Testing Q6_K Tiled Kernels ===");

    test_tiled_vs_nontiled_kquant(GgmlDType::Q6K, 1, 4096, 4096, max_error)?;
    test_tiled_vs_nontiled_kquant(GgmlDType::Q6K, 4, 4096, 4096, max_error)?;
    test_tiled_vs_nontiled_kquant(GgmlDType::Q6K, 16, 2048, 2048, max_error)?;
    test_tiled_vs_nontiled_kquant(GgmlDType::Q6K, 32, 1024, 1024, max_error)?;

    println!("\n✅ All tiled Q6_K matmul tests passed!");

    Ok(())
}

/// Test tiled Q5_K matmul (256 elements per block, 176 bytes)
#[test]
#[cfg(feature = "vulkan")]
fn test_tiled_q5_k_various_sizes() -> Result<()> {
    let max_error = 0.02;

    println!("\n=== Testing Q5_K Tiled Kernels ===");

    test_tiled_vs_nontiled_kquant(GgmlDType::Q5K, 1, 4096, 4096, max_error)?;
    test_tiled_vs_nontiled_kquant(GgmlDType::Q5K, 4, 4096, 4096, max_error)?;
    test_tiled_vs_nontiled_kquant(GgmlDType::Q5K, 16, 2048, 2048, max_error)?;

    println!("\n✅ All tiled Q5_K matmul tests passed!");

    Ok(())
}

/// Test tiled Q3_K matmul (256 elements per block, 110 bytes)
#[test]
#[cfg(feature = "vulkan")]
fn test_tiled_q3_k_various_sizes() -> Result<()> {
    let max_error = 0.03; // Q3_K may have higher quantization error (3-bit)

    println!("\n=== Testing Q3_K Tiled Kernels ===");

    test_tiled_vs_nontiled_kquant(GgmlDType::Q3K, 1, 4096, 4096, max_error)?;
    test_tiled_vs_nontiled_kquant(GgmlDType::Q3K, 4, 4096, 4096, max_error)?;
    test_tiled_vs_nontiled_kquant(GgmlDType::Q3K, 16, 2048, 2048, max_error)?;

    println!("\n✅ All tiled Q3_K matmul tests passed!");

    Ok(())
}

/// Test tiled Q2_K matmul (256 elements per block, 84 bytes)
#[test]
#[cfg(feature = "vulkan")]
fn test_tiled_q2_k_various_sizes() -> Result<()> {
    let max_error = 0.05; // Q2_K has highest quantization error (2-bit)

    println!("\n=== Testing Q2_K Tiled Kernels ===");

    test_tiled_vs_nontiled_kquant(GgmlDType::Q2K, 1, 4096, 4096, max_error)?;
    test_tiled_vs_nontiled_kquant(GgmlDType::Q2K, 4, 4096, 4096, max_error)?;
    test_tiled_vs_nontiled_kquant(GgmlDType::Q2K, 16, 2048, 2048, max_error)?;

    println!("\n✅ All tiled Q2_K matmul tests passed!");

    Ok(())
}

/// Test that prefill-optimized variant is selected and works correctly
#[test]
#[cfg(feature = "vulkan")]
fn test_prefill_variant_selection() -> Result<()> {
    let device = get_vulkan_device()?;
    let max_error = 0.01;

    println!("\n=== Testing Prefill Variant Selection (m > 8) ===");

    // Test with m=16 (should trigger prefill variant)
    let (m, n, k) = (16, 4096, 4096);

    let act_data: Vec<f32> = (0..(m * k))
        .map(|i| (i as f32 / (m * k) as f32) * 2.0 - 1.0)
        .collect();
    let activations = Tensor::from_slice(&act_data, (m, k), &device)?;

    let weight_data: Vec<f32> = (0..(n * k))
        .map(|i| ((i * 7 + 13) as f32 / (n * k) as f32) * 2.0 - 1.0)
        .collect();
    let weights_f32 = Tensor::from_slice(&weight_data, (n, k), &device)?;

    let qtensor = QTensor::quantize(&weights_f32, GgmlDType::Q8_0)?;
    let qmatmul = QMatMul::from_qtensor(qtensor)?;

    // Run with non-tiled (reference)
    std::env::set_var("PARAMECIA_DISABLE_TILED", "1");
    let result_nontiled = qmatmul.forward(&activations)?;
    std::env::remove_var("PARAMECIA_DISABLE_TILED");

    // Run with tiled (should use prefill variant for m=16)
    std::env::set_var("PARAMECIA_ENABLE_TILED", "1");
    let result_tiled = qmatmul.forward(&activations)?;
    std::env::remove_var("PARAMECIA_ENABLE_TILED");

    assert_close(
        &result_tiled,
        &result_nontiled,
        max_error,
        "Q8_0_prefill_16x4096x4096",
    )?;

    println!("✓ Prefill variant (m=16) matches reference");

    Ok(())
}

/// Benchmark decode vs prefill variants
#[test]
#[cfg(feature = "vulkan")]
#[ignore] // Run with: cargo test --features vulkan -- --ignored
fn benchmark_decode_vs_prefill() -> Result<()> {
    let device = get_vulkan_device()?;

    println!("\n=== Benchmark: Decode vs Prefill Variants ===\n");
    println!(
        "{:<6} {:<6} {:<6} | {:>12} | {:>12} | {:>10}",
        "m", "n", "k", "Decode (µs)", "Prefill (µs)", "Speedup"
    );
    println!("{}", "-".repeat(75));

    for (m, n, k) in [
        (4, 4096, 4096),  // Small: should use decode
        (8, 4096, 4096),  // Threshold
        (16, 4096, 4096), // Medium prefill
        (32, 4096, 4096), // Large prefill
        (64, 4096, 4096), // Very large prefill
    ] {
        let act_data: Vec<f32> = (0..(m * k))
            .map(|i| (i as f32 / (m * k) as f32) * 2.0 - 1.0)
            .collect();
        let activations = Tensor::from_slice(&act_data, (m, k), &device)?;

        let weight_data: Vec<f32> = (0..(n * k))
            .map(|i| ((i as f32 / (n * k) as f32) * 2.0 - 1.0))
            .collect();
        let weights_f32 = Tensor::from_slice(&weight_data, (n, k), &device)?;
        let qtensor = QTensor::quantize(&weights_f32, GgmlDType::Q8_0)?;
        let qmatmul = QMatMul::from_qtensor(qtensor)?;

        // Warmup
        std::env::set_var("PARAMECIA_ENABLE_TILED", "1");
        for _ in 0..3 {
            let _ = qmatmul.forward(&activations)?;
        }

        // Benchmark with automatic selection (will use decode for m<=8, prefill for m>8)
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = qmatmul.forward(&activations)?;
        }
        if let Device::Vulkan(vdev) = &device {
            vdev.flush()?;
        }
        let auto_time = start.elapsed().as_micros() as f64 / 10.0;
        std::env::remove_var("PARAMECIA_ENABLE_TILED");

        // For comparison, show which variant was auto-selected
        let variant = if m > 8 { "Prefill" } else { "Decode" };

        println!(
            "{:<6} {:<6} {:<6} | {:>12.1} | {:>12} | {:>10}",
            m, n, k, auto_time, variant, "-"
        );
    }

    println!("\n(Automatic variant selection based on m dimension)");

    std::env::remove_var("PARAMECIA_ENABLE_TILED");
    Ok(())
}

/// Benchmark Phase 4 optimizations: baseline vs all optimized variants
#[test]
#[ignore]
#[cfg(feature = "vulkan")]
fn benchmark_phase4_optimizations() -> Result<()> {
    let device = get_vulkan_device()?;

    println!("\n=== Phase 4 Benchmark: Baseline vs Optimized Variants ===\n");
    println!(
        "{:<6} {:<6} {:<6} | {:>11} | {:>11} | {:>11} | {:>11} | {:>7} | {:>7} | {:>7}",
        "m",
        "n",
        "k",
        "Baseline(µs)",
        "K-unrl(µs)",
        "Vec-ld(µs)",
        "DblBuf(µs)",
        "UnrlX",
        "VecX",
        "DblBufX"
    );
    println!("{}", "-".repeat(115));

    for (m, n, k) in [
        (16, 4096, 4096),  // Medium prefill
        (32, 4096, 4096),  // Large prefill
        (64, 4096, 4096),  // Very large prefill
        (128, 2048, 2048), // Huge prefill
    ] {
        let act_data: Vec<f32> = (0..(m * k))
            .map(|i| (i as f32 / (m * k) as f32) * 2.0 - 1.0)
            .collect();
        let activations = Tensor::from_slice(&act_data, (m, k), &device)?;

        let weight_data: Vec<f32> = (0..(n * k))
            .map(|i| (i as f32 / (n * k) as f32) * 2.0 - 1.0)
            .collect();
        let weights_f32 = Tensor::from_slice(&weight_data, (n, k), &device)?;
        let qtensor = QTensor::quantize(&weights_f32, GgmlDType::Q8_0)?;
        let qmatmul = QMatMul::from_qtensor(qtensor)?;

        // Benchmark baseline prefill kernel
        std::env::set_var("PARAMECIA_ENABLE_TILED", "1");
        std::env::remove_var("PARAMECIA_ENABLE_OPT");

        // Warmup
        for _ in 0..3 {
            let _ = qmatmul.forward(&activations)?;
        }

        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = qmatmul.forward(&activations)?;
        }
        if let Device::Vulkan(vdev) = &device {
            vdev.flush()?;
        }
        let baseline_time = start.elapsed().as_micros() as f64 / 10.0;

        // Benchmark K-loop unrolled variant
        std::env::set_var("PARAMECIA_ENABLE_OPT", "1");
        std::env::remove_var("PARAMECIA_ENABLE_VEC");

        // Warmup
        for _ in 0..3 {
            let _ = qmatmul.forward(&activations)?;
        }

        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = qmatmul.forward(&activations)?;
        }
        if let Device::Vulkan(vdev) = &device {
            vdev.flush()?;
        }
        let unroll_time = start.elapsed().as_micros() as f64 / 10.0;

        // Benchmark vectorized loading variant
        std::env::remove_var("PARAMECIA_ENABLE_OPT");
        std::env::set_var("PARAMECIA_ENABLE_VEC", "1");

        // Warmup
        for _ in 0..3 {
            let _ = qmatmul.forward(&activations)?;
        }

        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = qmatmul.forward(&activations)?;
        }
        if let Device::Vulkan(vdev) = &device {
            vdev.flush()?;
        }
        let vec_time = start.elapsed().as_micros() as f64 / 10.0;

        // Benchmark double buffering variant
        std::env::remove_var("PARAMECIA_ENABLE_VEC");
        std::env::set_var("PARAMECIA_ENABLE_DB", "1");

        // Warmup
        for _ in 0..3 {
            let _ = qmatmul.forward(&activations)?;
        }

        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = qmatmul.forward(&activations)?;
        }
        if let Device::Vulkan(vdev) = &device {
            vdev.flush()?;
        }
        let db_time = start.elapsed().as_micros() as f64 / 10.0;

        let unroll_speedup = baseline_time / unroll_time;
        let vec_speedup = baseline_time / vec_time;
        let db_speedup = baseline_time / db_time;

        println!("{:<6} {:<6} {:<6} | {:>11.1} | {:>11.1} | {:>11.1} | {:>11.1} | {:>6.2}x | {:>6.2}x | {:>6.2}x",
                 m, n, k, baseline_time, unroll_time, vec_time, db_time, unroll_speedup, vec_speedup, db_speedup);
    }

    println!("\nPhase 4 optimizations:");
    println!("  K-unroll: K-loop unrolling (2× per iteration)");
    println!("  Vec-load: Vectorized loading (vec4) for A matrix");
    println!("  DblBuf:   Double buffering (overlap load + compute)");

    std::env::remove_var("PARAMECIA_ENABLE_TILED");
    std::env::remove_var("PARAMECIA_ENABLE_OPT");
    std::env::remove_var("PARAMECIA_ENABLE_VEC");
    std::env::remove_var("PARAMECIA_ENABLE_DB");
    Ok(())
}

/// Benchmark Phase 6: Hierarchical tiled coopmat (tensor cores)
#[test]
#[ignore]
#[cfg(feature = "vulkan")]
fn benchmark_phase6_coopmat() -> Result<()> {
    let device = get_vulkan_device()?;

    println!("\n=== Phase 6 Benchmark: Hierarchical Tiled Coopmat (Tensor Cores) ===\n");
    println!(
        "{:<6} {:<6} {:<6} | {:>12} | {:>12} | {:>12} | {:>8}",
        "m", "n", "k", "Baseline(µs)", "K-unrl(µs)", "Coopmat(µs)", "CoopX"
    );
    println!("{}", "-".repeat(85));

    for (m, n, k) in [
        (16, 4096, 4096),  // Medium prefill
        (32, 4096, 4096),  // Large prefill
        (64, 4096, 4096),  // Very large prefill
        (128, 2048, 2048), // Huge prefill
        (256, 2048, 2048), // Extra large
    ] {
        let act_data: Vec<f32> = (0..(m * k))
            .map(|i| (i as f32 / (m * k) as f32) * 2.0 - 1.0)
            .collect();
        let activations = Tensor::from_slice(&act_data, (m, k), &device)?;

        let weight_data: Vec<f32> = (0..(n * k))
            .map(|i| (i as f32 / (n * k) as f32) * 2.0 - 1.0)
            .collect();
        let weights_f32 = Tensor::from_slice(&weight_data, (n, k), &device)?;
        let qtensor = QTensor::quantize(&weights_f32, GgmlDType::Q8_0)?;
        let qmatmul = QMatMul::from_qtensor(qtensor)?;

        // Benchmark baseline prefill kernel
        std::env::set_var("PARAMECIA_ENABLE_TILED", "1");
        std::env::set_var("PARAMECIA_DISABLE_OPT", "1");

        // Warmup
        for _ in 0..3 {
            let _ = qmatmul.forward(&activations)?;
        }

        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = qmatmul.forward(&activations)?;
        }
        if let Device::Vulkan(vdev) = &device {
            vdev.flush()?;
        }
        let baseline_time = start.elapsed().as_micros() as f64 / 10.0;

        // Benchmark K-loop unrolled variant
        std::env::remove_var("PARAMECIA_DISABLE_OPT");

        // Warmup
        for _ in 0..3 {
            let _ = qmatmul.forward(&activations)?;
        }

        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = qmatmul.forward(&activations)?;
        }
        if let Device::Vulkan(vdev) = &device {
            vdev.flush()?;
        }
        let unroll_time = start.elapsed().as_micros() as f64 / 10.0;

        // Benchmark coopmat variant (if available)
        std::env::set_var("PARAMECIA_FORCE_COOPMAT", "1");

        let coopmat_time = match (|| -> Result<f64> {
            // Warmup
            for _ in 0..3 {
                let _ = qmatmul.forward(&activations)?;
            }

            let start = std::time::Instant::now();
            for _ in 0..10 {
                let _ = qmatmul.forward(&activations)?;
            }
            if let Device::Vulkan(vdev) = &device {
                vdev.flush()?;
            }
            Ok(start.elapsed().as_micros() as f64 / 10.0)
        })() {
            Ok(time) => time,
            Err(_) => {
                println!(
                    "{:<6} {:<6} {:<6} | {:>12.1} | {:>12.1} | {:>12} | {:>8}",
                    m, n, k, baseline_time, unroll_time, "N/A", "-"
                );
                std::env::remove_var("PARAMECIA_FORCE_COOPMAT");
                continue;
            }
        };

        let coopmat_speedup = baseline_time / coopmat_time;

        println!(
            "{:<6} {:<6} {:<6} | {:>12.1} | {:>12.1} | {:>12.1} | {:>7.2}x",
            m, n, k, baseline_time, unroll_time, coopmat_time, coopmat_speedup
        );

        std::env::remove_var("PARAMECIA_FORCE_COOPMAT");
    }

    println!("\nPhase 6: Hierarchical Tiled Coopmat");
    println!("  - Combines Phase 3 memory hierarchy with tensor core acceleration");
    println!("  - Uses fp16 precision for tensor core operations");
    println!("  - Expected speedup: 2-4× on large prefill workloads");

    std::env::remove_var("PARAMECIA_ENABLE_TILED");
    std::env::remove_var("PARAMECIA_DISABLE_OPT");
    Ok(())
}

/// Test Phase 6: Hierarchical tiled coopmat for all K-quants
#[test]
#[ignore]
#[cfg(feature = "vulkan")]
fn test_kquant_coopmat_correctness() -> Result<()> {
    let device = get_vulkan_device()?;

    // Check if cooperative matrix is supported
    let has_coopmat = if let Device::Vulkan(vdev) = &device {
        vdev.has_cooperative_matrix()
    } else {
        return Ok(());
    };

    if !has_coopmat {
        println!("Skipping K-quant coopmat test: cooperative matrix not supported");
        return Ok(());
    }

    println!("\n=== Phase 6 K-Quant Coopmat Correctness Test ===\n");

    std::env::set_var("PARAMECIA_ENABLE_TILED", "1");
    std::env::set_var("PARAMECIA_FORCE_COOPMAT", "1");

    let (m, n, k) = (64, 4096, 4096);

    let act_data: Vec<f32> = (0..(m * k))
        .map(|i| (i as f32 / (m * k) as f32) * 2.0 - 1.0)
        .collect();
    let activations = Tensor::from_slice(&act_data, (m, k), &device)?;

    let weight_data: Vec<f32> = (0..(n * k))
        .map(|i| (i as f32 / (n * k) as f32) * 2.0 - 1.0)
        .collect();
    let weights_f32 = Tensor::from_slice(&weight_data, (n, k), &device)?;

    // Test all K-quant formats
    for dtype in [
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ] {
        println!("Testing {:?} coopmat variant...", dtype);

        // Get baseline result without coopmat
        std::env::remove_var("PARAMECIA_FORCE_COOPMAT");
        let qtensor_baseline = QTensor::quantize(&weights_f32, dtype)?;
        let qmatmul_baseline = QMatMul::from_qtensor(qtensor_baseline)?;
        let baseline = qmatmul_baseline.forward(&activations)?;

        // Get coopmat result
        std::env::set_var("PARAMECIA_FORCE_COOPMAT", "1");
        let qtensor_coopmat = QTensor::quantize(&weights_f32, dtype)?;
        let qmatmul_coopmat = QMatMul::from_qtensor(qtensor_coopmat)?;
        let coopmat = qmatmul_coopmat.forward(&activations)?;

        // Compare results (fp16 precision, allow 15% error)
        let baseline_data = baseline.flatten_all()?.to_vec1::<f32>()?;
        let coopmat_data = coopmat.flatten_all()?.to_vec1::<f32>()?;

        let mut max_rel_error = 0.0f32;
        let mut max_abs_error = 0.0f32;
        for (i, (&b, &c)) in baseline_data.iter().zip(coopmat_data.iter()).enumerate() {
            let abs_err = (b - c).abs();
            let rel_err = if b.abs() > 1e-6 {
                abs_err / b.abs()
            } else {
                abs_err
            };

            max_rel_error = max_rel_error.max(rel_err);
            max_abs_error = max_abs_error.max(abs_err);

            if rel_err > 0.15 {
                println!(
                    "  Large error at index {}: baseline={}, coopmat={}, rel_err={:.4}",
                    i, b, c, rel_err
                );
            }
        }

        println!(
            "  {:?}: max_rel_error={:.4}, max_abs_error={:.4}",
            dtype, max_rel_error, max_abs_error
        );

        // Relaxed threshold for fp16 precision
        assert!(
            max_rel_error < 0.15,
            "{:?} coopmat: Relative error {:.6} exceeds threshold 0.15",
            dtype,
            max_rel_error
        );

        println!("  {:?} coopmat: ✓ PASSED\n", dtype);
    }

    std::env::remove_var("PARAMECIA_ENABLE_TILED");
    std::env::remove_var("PARAMECIA_FORCE_COOPMAT");
    Ok(())
}

/// Benchmark Phase 6: Hierarchical tiled coopmat for all K-quants
#[test]
#[ignore]
#[cfg(feature = "vulkan")]
fn benchmark_kquant_coopmat() -> Result<()> {
    let device = get_vulkan_device()?;

    // Check if cooperative matrix is supported
    let has_coopmat = if let Device::Vulkan(vdev) = &device {
        vdev.has_cooperative_matrix()
    } else {
        return Ok(());
    };

    if !has_coopmat {
        println!("Skipping K-quant coopmat benchmark: cooperative matrix not supported");
        return Ok(());
    }

    println!("\n=== Phase 6 K-Quant Coopmat Performance Benchmark ===\n");
    println!(
        "{:<6} {:<6} | {:>12} | {:>12} | {:>12} | {:>8}",
        "DType", "m", "Baseline(µs)", "K-unrl(µs)", "Coopmat(µs)", "CoopX"
    );
    println!("{}", "-".repeat(75));

    std::env::set_var("PARAMECIA_ENABLE_TILED", "1");

    let (n, k) = (4096, 4096);

    for dtype in [
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ] {
        for m in [16, 64, 128] {
            let act_data: Vec<f32> = (0..(m * k))
                .map(|i| (i as f32 / (m * k) as f32) * 2.0 - 1.0)
                .collect();
            let activations = Tensor::from_slice(&act_data, (m, k), &device)?;

            let weight_data: Vec<f32> = (0..(n * k))
                .map(|i| (i as f32 / (n * k) as f32) * 2.0 - 1.0)
                .collect();
            let weights_f32 = Tensor::from_slice(&weight_data, (n, k), &device)?;
            let qtensor = QTensor::quantize(&weights_f32, dtype)?;
            let qmatmul = QMatMul::from_qtensor(qtensor)?;

            // Benchmark baseline (Phase 3)
            std::env::set_var("PARAMECIA_DISABLE_OPT", "1");
            std::env::remove_var("PARAMECIA_FORCE_COOPMAT");
            for _ in 0..3 {
                let _ = qmatmul.forward(&activations)?;
            }
            let start = std::time::Instant::now();
            for _ in 0..10 {
                let _ = qmatmul.forward(&activations)?;
            }
            if let Device::Vulkan(vdev) = &device {
                vdev.flush()?;
            }
            let baseline_time = start.elapsed().as_micros() as f64 / 10.0;

            // Benchmark K-loop unrolled (Phase 4)
            std::env::remove_var("PARAMECIA_DISABLE_OPT");
            for _ in 0..3 {
                let _ = qmatmul.forward(&activations)?;
            }
            let start = std::time::Instant::now();
            for _ in 0..10 {
                let _ = qmatmul.forward(&activations)?;
            }
            if let Device::Vulkan(vdev) = &device {
                vdev.flush()?;
            }
            let unroll_time = start.elapsed().as_micros() as f64 / 10.0;

            // Benchmark coopmat (Phase 6)
            std::env::set_var("PARAMECIA_FORCE_COOPMAT", "1");
            for _ in 0..3 {
                let _ = qmatmul.forward(&activations)?;
            }
            let start = std::time::Instant::now();
            for _ in 0..10 {
                let _ = qmatmul.forward(&activations)?;
            }
            if let Device::Vulkan(vdev) = &device {
                vdev.flush()?;
            }
            let coopmat_time = start.elapsed().as_micros() as f64 / 10.0;

            let coopmat_speedup = baseline_time / coopmat_time;

            println!(
                "{:<6} {:<6} | {:>12.1} | {:>12.1} | {:>12.1} | {:>7.2}x",
                format!("{:?}", dtype),
                m,
                baseline_time,
                unroll_time,
                coopmat_time,
                coopmat_speedup
            );

            std::env::remove_var("PARAMECIA_FORCE_COOPMAT");
        }
    }

    println!("\nPhase 6 K-Quant Coopmat Summary:");
    println!("  - Hierarchical tiling + tensor core acceleration for all K-quants");
    println!("  - Uses fp16 precision for tensor core operations");
    println!("  - Expected speedup: 1.5-2× on prefill workloads");

    std::env::remove_var("PARAMECIA_ENABLE_TILED");
    std::env::remove_var("PARAMECIA_DISABLE_OPT");
    Ok(())
}
