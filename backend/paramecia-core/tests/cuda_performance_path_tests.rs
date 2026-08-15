#![cfg(feature = "cuda")]

use paramecia_core::deltanet_ops::delta_net_autoregressive_step;
use paramecia_core::quantized::{GgmlDType, QTensor};
use paramecia_core::{utils, Device, Result, Tensor};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
    let diff = (lhs.to_device(&Device::Cpu)? - rhs.to_device(&Device::Cpu)?)?;
    diff.abs()?.flatten_all()?.max(0)?.to_vec0()
}

#[test]
fn delta_net_autoregressive_step_matches_cpu() -> Result<()> {
    if !utils::cuda_is_available() {
        return Ok(());
    }

    let cuda = Device::new_cuda(0)?;
    let (batch, heads, dim) = (2, 2, 128);
    let vector_len = batch * heads * dim;
    let state_len = vector_len * dim;

    let q: Vec<f32> = (0..vector_len)
        .map(|i| ((i as f32 + 1.0) * 0.013).sin() * 0.2)
        .collect();
    let k: Vec<f32> = (0..vector_len)
        .map(|i| ((i as f32 + 3.0) * 0.017).cos() * 0.2)
        .collect();
    let v: Vec<f32> = (0..vector_len)
        .map(|i| ((i as f32 + 5.0) * 0.019).sin() * 0.2)
        .collect();
    let gate: Vec<f32> = (0..batch * heads).map(|i| -0.1 - i as f32 * 0.02).collect();
    let beta: Vec<f32> = (0..batch * heads).map(|i| -0.3 + i as f32 * 0.1).collect();
    let state: Vec<f32> = (0..state_len)
        .map(|i| ((i as f32 + 7.0) * 0.003).sin() * 0.01)
        .collect();

    let run = |device: &Device| -> Result<(Tensor, Tensor)> {
        delta_net_autoregressive_step(
            &Tensor::from_slice(&q, (batch, heads, dim), device)?,
            &Tensor::from_slice(&k, (batch, heads, dim), device)?,
            &Tensor::from_slice(&v, (batch, heads, dim), device)?,
            &Tensor::from_slice(&gate, (batch, heads), device)?,
            &Tensor::from_slice(&beta, (batch, heads), device)?,
            &Tensor::from_slice(&state, (batch, heads, dim, dim), device)?,
            1.0,
            1e-6,
        )
    };

    let (cpu_output, cpu_state) = run(&Device::Cpu)?;
    let (cuda_output, cuda_state) = run(&cuda)?;
    assert!(max_abs_diff(&cpu_output, &cuda_output)? < 2e-4);
    assert!(max_abs_diff(&cpu_state, &cuda_state)? < 2e-4);
    Ok(())
}

#[test]
fn dense_indexed_moe_large_batch_matches_cpu() -> Result<()> {
    if !utils::cuda_is_available() {
        return Ok(());
    }

    let cuda = Device::new_cuda(0)?;
    let (batch, n, k) = (64, 256, 256);
    let weights: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32 + 1.0) * 0.007).sin() * 0.05)
        .collect();
    let input: Vec<f32> = (0..batch * k)
        .map(|i| ((i as f32 + 2.0) * 0.011).cos() * 0.1)
        .collect();
    let ids = vec![0u32; batch];

    let run = |device: &Device| -> Result<Tensor> {
        let weights = Tensor::from_slice(&weights, (1, n, k), device)?;
        let weights = QTensor::quantize(&weights, GgmlDType::Q4K)?;
        let input = Tensor::from_slice(&input, (batch, 1, k), device)?;
        let ids = Tensor::from_slice(&ids, (batch, 1), device)?;
        weights.indexed_moe_forward(&input, &ids)
    };

    let cpu_output = run(&Device::Cpu)?;
    let cuda_output = run(&cuda)?;
    assert!(max_abs_diff(&cpu_output, &cuda_output)? < 5e-3);
    Ok(())
}
