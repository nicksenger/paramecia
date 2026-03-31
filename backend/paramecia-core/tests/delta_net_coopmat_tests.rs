// Tests comparing coopmat vs standard implementations for DeltaNet operations
#![cfg(feature = "vulkan")]

use paramecia_core::deltanet_ops::{delta_net_autoregressive_step, delta_net_parallel_with_states};
use paramecia_core::{Device, Result, Tensor};

// Helper function to get Vulkan device ordinal from environment variable
fn get_vulkan_device() -> Result<usize> {
    match std::env::var("PARAMECIA_VK_DEVICE") {
        Ok(val) => val.parse::<usize>().map_err(|e| {
            paramecia_core::Error::Msg(format!("Invalid PARAMECIA_VK_DEVICE '{}': {}", val, e))
        }),
        Err(_) => Ok(0),
    }
}

// Helper macro to test delta_net_parallel operation
macro_rules! test_delta_net_parallel_coopmat {
    ($head_dim:expr, $env_var:expr, $tolerance:expr) => {{
        use std::env;

        let device_ordinal = get_vulkan_device()?;
        let device = Device::new_vulkan(device_ordinal)?;

        let (batch, num_heads, num_tokens) = (2, 4, 8);
        let head_dim = $head_dim;

        // Create test inputs
        let q_data: Vec<f32> = (0..(batch * num_heads * num_tokens * head_dim))
            .map(|i| (i as f32 / 1000.0).sin())
            .collect();
        let k_data: Vec<f32> = (0..(batch * num_heads * num_tokens * head_dim))
            .map(|i| (i as f32 / 1000.0 + 0.5).cos())
            .collect();
        let v_data: Vec<f32> = (0..(batch * num_heads * num_tokens * head_dim))
            .map(|i| (i as f32 / 1000.0 + 1.0).sin())
            .collect();
        let gate_data: Vec<f32> = (0..(batch * num_heads * num_tokens))
            .map(|i| -0.1 * (i as f32 / 10.0))
            .collect();
        let beta_data: Vec<f32> = (0..(batch * num_heads * num_tokens))
            .map(|i| 0.5 + 0.3 * ((i as f32) / 10.0).sin())
            .collect();
        let state_data: Vec<f32> = (0..(batch * num_heads * head_dim * head_dim))
            .map(|i| 0.01 * ((i as f32) / 100.0).sin())
            .collect();

        let q = Tensor::from_slice(&q_data, (batch, num_heads, num_tokens, head_dim), &device)?;
        let k = Tensor::from_slice(&k_data, (batch, num_heads, num_tokens, head_dim), &device)?;
        let v = Tensor::from_slice(&v_data, (batch, num_heads, num_tokens, head_dim), &device)?;
        let gate = Tensor::from_slice(&gate_data, (batch, num_heads, num_tokens), &device)?;
        let beta = Tensor::from_slice(&beta_data, (batch, num_heads, num_tokens), &device)?;
        let state =
            Tensor::from_slice(&state_data, (batch, num_heads, head_dim, head_dim), &device)?;

        // Test with coopmat DISABLED (standard path)
        env::set_var($env_var, "1");
        let (output_std, new_state_std, _inter_std) =
            delta_net_parallel_with_states(&q, &k, &v, &gate, &beta, &state)?;
        env::remove_var($env_var);

        // Test with coopmat ENABLED
        let (output_coop, new_state_coop, _inter_coop) =
            delta_net_parallel_with_states(&q, &k, &v, &gate, &beta, &state)?;

        // Compare outputs
        let output_diff = (&output_std - &output_coop)?.abs()?;
        let output_max_diff: f32 = output_diff.flatten_all()?.max(0)?.to_vec0()?;
        let output_mean_diff: f32 = output_diff.flatten_all()?.mean(0)?.to_vec0()?;

        // Compare states
        let state_diff = (&new_state_std - &new_state_coop)?.abs()?;
        let state_max_diff: f32 = state_diff.flatten_all()?.max(0)?.to_vec0()?;
        let state_mean_diff: f32 = state_diff.flatten_all()?.mean(0)?.to_vec0()?;

        println!("DeltaNet Parallel d{} Coopmat vs Standard:", head_dim);
        println!("  Output max diff: {}", output_max_diff);
        println!("  Output mean diff: {}", output_mean_diff);
        println!("  State max diff: {}", state_max_diff);
        println!("  State mean diff: {}", state_mean_diff);
        println!(
            "  Output std (first 8): {:?}",
            &output_std.flatten_all()?.to_vec1::<f32>()?[..8]
        );
        println!(
            "  Output coop (first 8): {:?}",
            &output_coop.flatten_all()?.to_vec1::<f32>()?[..8]
        );

        if output_max_diff >= $tolerance {
            println!(
                "  ⚠ WARNING: Output exceeds tolerance ({} >= {})",
                output_max_diff, $tolerance
            );
        } else {
            println!("  ✓ PASS");
        }
        println!();

        assert!(
            output_max_diff < $tolerance,
            "d{} delta_net_parallel coopmat output differs too much: max_diff={}, tolerance={}",
            head_dim,
            output_max_diff,
            $tolerance
        );

        assert!(
            state_max_diff < $tolerance,
            "d{} delta_net_parallel coopmat state differs too much: max_diff={}, tolerance={}",
            head_dim,
            state_max_diff,
            $tolerance
        );

        Ok::<_, paramecia_core::Error>(())
    }};
}

// ============================================================================
// DeltaNet Parallel Tests
// ============================================================================

#[cfg(feature = "vulkan")]
#[test]
fn test_delta_net_parallel_d64_coopmat() -> Result<()> {
    test_delta_net_parallel_coopmat!(64, "PARAMECIA_DISABLE_COOPMAT_D64_DELTA_NET_PARALLEL", 0.05)
}

#[cfg(feature = "vulkan")]
#[test]
fn test_delta_net_parallel_d128_coopmat() -> Result<()> {
    test_delta_net_parallel_coopmat!(
        128,
        "PARAMECIA_DISABLE_COOPMAT_D128_DELTA_NET_PARALLEL",
        0.05
    )
}

#[cfg(feature = "vulkan")]
#[test]
fn test_delta_net_parallel_d256_coopmat() -> Result<()> {
    test_delta_net_parallel_coopmat!(
        256,
        "PARAMECIA_DISABLE_COOPMAT_D256_DELTA_NET_PARALLEL",
        0.05
    )
}

// ============================================================================
// GLA Step Tests
// ============================================================================

// Helper macro to test gla_step operation
macro_rules! test_gla_step_coopmat {
    ($head_dim:expr, $env_var:expr, $tolerance:expr) => {{
        use std::env;

        let device_ordinal = get_vulkan_device()?;
        let device = Device::new_vulkan(device_ordinal)?;

        let (batch, num_heads) = (2, 4);
        let head_dim = $head_dim;
        let batch_heads = batch * num_heads;

        // Create test inputs
        let q_data: Vec<f32> = (0..(batch_heads * head_dim))
            .map(|i| (i as f32 / 100.0).sin())
            .collect();
        let k_data: Vec<f32> = (0..(batch_heads * head_dim))
            .map(|i| (i as f32 / 100.0 + 0.5).cos())
            .collect();
        let v_data: Vec<f32> = (0..(batch_heads * head_dim))
            .map(|i| (i as f32 / 100.0 + 1.0).sin())
            .collect();
        let gate_data: Vec<f32> = (0..batch_heads).map(|i| -0.1 * (i as f32 / 10.0)).collect();
        let beta_data: Vec<f32> = (0..batch_heads)
            .map(|i| 0.5 * ((i as f32) / 5.0).sin())
            .collect();
        let state_data: Vec<f32> = (0..(batch_heads * head_dim * head_dim))
            .map(|i| 0.01 * ((i as f32) / 100.0).sin())
            .collect();

        let q = Tensor::from_slice(&q_data, (batch, num_heads, head_dim), &device)?;
        let k = Tensor::from_slice(&k_data, (batch, num_heads, head_dim), &device)?;
        let v = Tensor::from_slice(&v_data, (batch, num_heads, head_dim), &device)?;
        let gate = Tensor::from_slice(&gate_data, (batch, num_heads), &device)?;
        let beta = Tensor::from_slice(&beta_data, (batch, num_heads), &device)?;
        let state =
            Tensor::from_slice(&state_data, (batch, num_heads, head_dim, head_dim), &device)?;

        let scale = 1.0f32;
        let eps = 1e-6f32;

        // Test with coopmat DISABLED (standard path)
        env::set_var($env_var, "1");
        let (output_std, new_state_std) =
            delta_net_autoregressive_step(&q, &k, &v, &gate, &beta, &state, scale, eps)?;
        env::remove_var($env_var);

        // Test with coopmat ENABLED
        let (output_coop, new_state_coop) =
            delta_net_autoregressive_step(&q, &k, &v, &gate, &beta, &state, scale, eps)?;

        // Compare outputs
        let output_diff = (&output_std - &output_coop)?.abs()?;
        let output_max_diff: f32 = output_diff.flatten_all()?.max(0)?.to_vec0()?;
        let output_mean_diff: f32 = output_diff.flatten_all()?.mean(0)?.to_vec0()?;

        // Compare states
        let state_diff = (&new_state_std - &new_state_coop)?.abs()?;
        let state_max_diff: f32 = state_diff.flatten_all()?.max(0)?.to_vec0()?;
        let state_mean_diff: f32 = state_diff.flatten_all()?.mean(0)?.to_vec0()?;

        println!("GLA Step d{} Coopmat vs Standard:", head_dim);
        println!("  Output max diff: {}", output_max_diff);
        println!("  Output mean diff: {}", output_mean_diff);
        println!("  State max diff: {}", state_max_diff);
        println!("  State mean diff: {}", state_mean_diff);

        if output_max_diff >= $tolerance {
            println!(
                "  ⚠ WARNING: Output exceeds tolerance ({} >= {})",
                output_max_diff, $tolerance
            );
        } else {
            println!("  ✓ PASS");
        }
        println!();

        assert!(
            output_max_diff < $tolerance,
            "d{} gla_step coopmat output differs too much: max_diff={}, tolerance={}",
            head_dim,
            output_max_diff,
            $tolerance
        );

        assert!(
            state_max_diff < $tolerance,
            "d{} gla_step coopmat state differs too much: max_diff={}, tolerance={}",
            head_dim,
            state_max_diff,
            $tolerance
        );

        Ok::<_, paramecia_core::Error>(())
    }};
}

#[cfg(feature = "vulkan")]
#[test]
fn test_gla_step_d64_coopmat() -> Result<()> {
    test_gla_step_coopmat!(64, "PARAMECIA_DISABLE_COOPMAT_D64_GLA_STEP", 0.05)
}

#[cfg(feature = "vulkan")]
#[test]
fn test_gla_step_d128_coopmat() -> Result<()> {
    test_gla_step_coopmat!(128, "PARAMECIA_DISABLE_COOPMAT_D128_GLA_STEP", 0.05)
}

#[cfg(feature = "vulkan")]
#[test]
fn test_gla_step_d256_coopmat() -> Result<()> {
    test_gla_step_coopmat!(256, "PARAMECIA_DISABLE_COOPMAT_D256_GLA_STEP", 0.05)
}
