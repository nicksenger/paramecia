use paramecia_core::quantized::GgmlDType;
use paramecia_core::{Device, Result};
use tracing::warn;

/// KV-cache quantization mode for memory efficiency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvCacheQuantization {
    /// No quantization - store KV-cache in the model's native dtype (f16/bf16).
    /// Provides maximum accuracy.
    #[deprecated(since = "0.2.0", note = "Use F16 or BF16 instead for clarity")]
    None,
    /// Store KV-cache as f16 (16-bit float). Maximum accuracy, higher memory usage.
    F16,
    /// Store KV-cache as bf16 (bfloat16). Similar to F16 but with better dynamic range.
    /// Useful for models trained with bf16.
    BF16,
    /// Quantize to Q8_0 (8-bit, ~2x memory reduction with minimal accuracy loss).
    Q8_0,
    /// Quantize to Q4K (4-bit, ~4x memory reduction).
    /// This is the default setting.
    #[default]
    Q4K,
}

impl KvCacheQuantization {
    pub(super) fn to_ggml_dtype(self) -> Option<GgmlDType> {
        match self {
            #[allow(deprecated)]
            Self::None | Self::F16 | Self::BF16 => None,
            Self::Q8_0 => Some(GgmlDType::Q8_0),
            Self::Q4K => Some(GgmlDType::Q4K),
        }
    }

    pub(super) fn block_size(&self) -> usize {
        match self {
            #[allow(deprecated)]
            Self::None | Self::F16 | Self::BF16 => 1,
            Self::Q8_0 => 32,
            Self::Q4K => 256,
        }
    }

    /// Returns the preferred DType for non-quantized cache storage.
    /// Returns None if using GGML quantization (Q8_0, Q4K).
    pub fn cache_dtype(&self) -> Option<paramecia_core::DType> {
        match self {
            #[allow(deprecated)]
            Self::None | Self::F16 => Some(paramecia_core::DType::F16),
            Self::BF16 => Some(paramecia_core::DType::BF16),
            Self::Q8_0 | Self::Q4K => None, // Uses GGML quantization
        }
    }

    /// Parse a KV cache quantization mode from a string.
    ///
    /// Accepts: "f16", "bf16", "q8", "q8_0", "q4", "q4k", "none" (case insensitive)
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "f16" | "fp16" => Some(Self::F16),
            "bf16" | "bfloat16" => Some(Self::BF16),
            "q8" | "q8_0" => Some(Self::Q8_0),
            "q4" | "q4k" | "q4_k" => Some(Self::Q4K),
            #[allow(deprecated)]
            "none" => Some(Self::None),
            _ => None,
        }
    }
}

impl std::str::FromStr for KvCacheQuantization {
    type Err = ();

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Self::from_str(s).ok_or(())
    }
}

/// Device offloading mode for MoE expert weights.
///
/// This enum controls how expert weights are distributed across devices for memory/speed tradeoffs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceOffloadMode {
    /// All expert weights on GPU (maximum speed, requires most VRAM).
    ///
    /// Best for GPUs with sufficient memory (24GB+ for 80B models).
    FullGpu,

    /// All expert weights (gate, up, down projections) offloaded to CPU.
    ///
    /// Significantly reduces VRAM usage but slower inference due to PCIe bandwidth.
    /// Best for GPUs with limited VRAM (8-16GB). Enables parallel CPU expert
    /// processing and is optimal for the async pipeline.
    ExpertsOnCpu,

    /// Only down projections offloaded to CPU, gate and up stay on GPU.
    ///
    /// Balanced approach: down projections are large but less frequently accessed.
    /// Gate projections benefit from GPU for fast routing, up projections
    /// benefit from GPU for fast activation computation.
    /// Best for GPUs with moderate VRAM (16-24GB).
    DownProjectionsOnCpu,

    /// Up and down projections offloaded to CPU, gate stays on GPU.
    ///
    /// Gate projections benefit from GPU for fast routing decisions.
    /// Up and down projections (the largest weights) are on CPU to save VRAM.
    /// Best balance of VRAM savings and routing performance.
    UpDownProjectionsOnCpu,
}

impl Default for DeviceOffloadMode {
    fn default() -> Self {
        // The dense 0.8B model is only about 0.5 GiB at Q4_K_M. Treating its
        // FFN as an offloaded "expert" defeats GPU inference and makes PCIe
        // transfers dominate every layer.
        let default_architecture_is_0p8b = cfg!(not(any(
            feature = "qwen3next_80b_a3b",
            feature = "qwen35moe_35b_a3b",
            feature = "qwen35moe_122b_a10b",
            feature = "qwen35moe_397b_a17b",
            feature = "qwen35_0p8b",
            feature = "qwen35_2b",
            feature = "qwen35_4b",
            feature = "qwen35_9b",
            feature = "qwen35_27b",
        )));
        if cfg!(feature = "qwen35_0p8b") || default_architecture_is_0p8b {
            Self::FullGpu
        } else {
            Self::ExpertsOnCpu
        }
    }
}

impl DeviceOffloadMode {
    /// Returns the device placement for (gate, up, down) expert projections.
    pub fn get_expert_devices(&self, gpu_device: &Device) -> (Device, Device, Device) {
        match self {
            Self::FullGpu => (gpu_device.clone(), gpu_device.clone(), gpu_device.clone()),
            Self::ExpertsOnCpu => (Device::Cpu, Device::Cpu, Device::Cpu),
            Self::DownProjectionsOnCpu => (gpu_device.clone(), gpu_device.clone(), Device::Cpu),
            Self::UpDownProjectionsOnCpu => (gpu_device.clone(), Device::Cpu, Device::Cpu),
        }
    }

    /// Parse an offload mode from a CLI string.
    ///
    /// If `cpu` is true, returns `FullGpu` (everything on one device when running on CPU).
    /// Accepts: "auto", "none", "experts", "down", "updown" (case sensitive).
    pub fn parse(offload: &str, cpu: bool) -> Self {
        if cpu {
            Self::FullGpu
        } else {
            match offload {
                "auto" => Self::default(),
                "none" => Self::FullGpu,
                "experts" => Self::ExpertsOnCpu,
                "down" => Self::DownProjectionsOnCpu,
                "updown" => Self::UpDownProjectionsOnCpu,
                other => {
                    let fallback = Self::default();
                    warn!(mode = %other, ?fallback, "Unknown offload mode, using model default");
                    fallback
                }
            }
        }
    }
}

impl KvCacheQuantization {
    /// Parse a KV cache quantization hint from a CLI argument.
    ///
    /// Returns `Q8_0` when `hint` is `None`. Falls back to `Q8_0` on unrecognized input.
    pub fn parse(hint: Option<&str>) -> Self {
        match hint {
            None => Self::Q8_0,
            Some(s) => Self::from_str(s).unwrap_or_else(|| {
                warn!(value = %s, "Unknown KV cache quantization, using 'q8_0'");
                Self::Q8_0
            }),
        }
    }
}

/// Select the best available compute device (CUDA → Vulkan → Metal → CPU).
pub fn select_best_device() -> Device {
    Device::new_cuda(0)
        .or_else(|_| Device::new_vulkan(0))
        .or_else(|_| Device::new_metal(0))
        .unwrap_or(Device::Cpu)
}

/// Create a GPU device by ordinal, trying CUDA first, then Vulkan, then Metal as fallback.
pub(super) fn create_gpu_device(ordinal: usize) -> Result<Device> {
    Device::new_cuda(ordinal)
        .or_else(|_| Device::new_vulkan(ordinal))
        .or_else(|_| Device::new_metal(ordinal))
        .map_err(|e| {
            paramecia_core::Error::Msg(format!(
                "Failed to create GPU device {}: {}. Ensure {} GPUs are available.",
                ordinal,
                e,
                ordinal + 1
            ))
        })
}

/// Maps transformer layer indices to GPU devices for multi-GPU layer parallelism.
///
/// When `layer_devices` is empty, the model runs in single-GPU mode with zero overhead.
/// Otherwise, each layer is assigned to a specific GPU based on proportional splits.
#[derive(Debug, Clone)]
pub struct LayerDeviceMap {
    /// Device for each layer (empty = single-GPU mode).
    layer_devices: Vec<Device>,
    /// Primary device (GPU 0) for embedding/norm/lm_head.
    primary_device: Device,
}

impl LayerDeviceMap {
    /// Create a device map from proportional split string.
    ///
    /// Each position corresponds to a GPU device ordinal. Zero means skip that device.
    /// Examples:
    /// - "3,1" = 75% GPU 0, 25% GPU 1
    /// - "0,1,0" = 100% GPU 1 (GPUs 0 and 2 are skipped)
    /// - "0,1,3" = 25% GPU 1, 75% GPU 2 (GPU 0 is skipped)
    pub fn from_proportions(proportions: &str, num_layers: usize) -> Result<Self> {
        let parts: Vec<f64> = proportions
            .split(',')
            .map(|s| {
                s.trim().parse::<f64>().map_err(|e| {
                    paramecia_core::Error::Msg(format!("Invalid proportion '{}': {}", s.trim(), e))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        if parts.is_empty() {
            paramecia_core::bail!("Layer split proportions cannot be empty");
        }
        if parts.iter().any(|&p| p < 0.0) {
            paramecia_core::bail!("Layer split proportions must be non-negative");
        }

        let total: f64 = parts.iter().sum();
        if total <= 0.0 {
            paramecia_core::bail!("Layer split proportions must have at least one positive value");
        }

        // Collect (ordinal, proportion) for non-zero entries
        let active: Vec<(usize, f64)> = parts
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, &p)| (i, p))
            .collect();

        // Create GPU devices only for active ordinals
        let mut gpu_devices = Vec::with_capacity(active.len());
        for &(ordinal, _) in &active {
            gpu_devices.push(create_gpu_device(ordinal)?);
        }

        // Assign layers proportionally across active devices
        let mut layer_devices = Vec::with_capacity(num_layers);
        let mut assigned = 0usize;
        let num_active = active.len();

        for (active_idx, &(_, proportion)) in active.iter().enumerate() {
            let count = if active_idx == num_active - 1 {
                // Last active GPU gets remaining layers to avoid rounding issues
                num_layers - assigned
            } else {
                ((proportion / total) * num_layers as f64).round() as usize
            };
            for _ in 0..count {
                layer_devices.push(gpu_devices[active_idx].clone());
            }
            assigned += count;
        }

        let primary_device = gpu_devices[0].clone();
        Ok(Self {
            layer_devices,
            primary_device,
        })
    }

    /// Single-GPU mode (no per-layer mapping).
    pub fn single(device: Device) -> Self {
        Self {
            layer_devices: Vec::new(),
            primary_device: device,
        }
    }

    /// Get the device for a specific layer, falling back to primary when in single-GPU mode.
    pub fn device_for_layer(&self, idx: usize) -> &Device {
        if self.layer_devices.is_empty() {
            &self.primary_device
        } else {
            &self.layer_devices[idx]
        }
    }

    /// Get the primary device (GPU 0, used for embedding/norm/lm_head).
    pub fn primary_device(&self) -> &Device {
        &self.primary_device
    }

    /// Check if multi-GPU mode is active.
    pub fn is_multi_gpu(&self) -> bool {
        !self.layer_devices.is_empty()
    }

    /// Number of unique GPU devices.
    pub fn num_gpus(&self) -> usize {
        if self.layer_devices.is_empty() {
            1
        } else {
            use std::collections::HashSet;
            let mut seen = HashSet::new();
            for d in &self.layer_devices {
                seen.insert(format!("{:?}", d));
            }
            seen.len()
        }
    }
}
