use paramecia_core::quantized::{QStorage, QTensor, SharedQTensor};
use paramecia_core::{Device, Result};
use paramecia_tensor::glowstick::Shape2;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use super::shape::{S, SI};

type TQMatMul<Sh> = paramecia_tensor::QMatMul<Sh>;

#[derive(Debug, Clone)]
pub(super) struct CachedMatmuls {
    pub(super) gate: Arc<TQMatMul<Shape2<SI, S>>>,
    pub(super) up: Arc<TQMatMul<Shape2<SI, S>>>,
    pub(super) down: Arc<TQMatMul<Shape2<S, SI>>>,
    pub(super) gate_device: Device,
    pub(super) up_device: Device,
    pub(super) down_device: Device,
}

/// GPU Hot Expert Cache - keeps frequently used experts on GPU even when
/// the main expert weights are on CPU. This is the key optimization for
/// CPU-offloaded MoE: avoid CPU computation entirely for hot experts.
pub(super) struct GpuHotExpertCache {
    /// GPU device for caching
    pub(super) gpu_device: Device,
    /// Cached expert matmuls on GPU: expert_idx -> CachedMatmuls
    pub(super) entries: HashMap<usize, CachedMatmuls>,
    /// Usage counts for each expert
    pub(super) usage_counts: Vec<u64>,
    /// LRU tracking
    pub(super) lru: VecDeque<usize>,
    /// Maximum number of experts to cache on GPU
    pub(super) capacity: usize,
    /// Total tokens processed (for promotion decisions)
    pub(super) total_tokens: u64,
    /// Promotion threshold (min usage to promote to GPU)
    pub(super) promotion_threshold: u64,
}

impl std::fmt::Debug for GpuHotExpertCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuHotExpertCache")
            .field("capacity", &self.capacity)
            .field("cached_count", &self.entries.len())
            .finish()
    }
}

impl GpuHotExpertCache {
    pub(super) fn new(gpu_device: Device, num_experts: usize, capacity: usize) -> Self {
        Self {
            gpu_device,
            entries: HashMap::new(),
            usage_counts: vec![0u64; num_experts],
            lru: VecDeque::new(),
            capacity,
            total_tokens: 0,
            promotion_threshold: 1, // Promote immediately on first use
        }
    }

    /// Record expert usage and return whether it's now a hot expert
    pub(super) fn record_usage(&mut self, expert_idx: usize) -> bool {
        if expert_idx < self.usage_counts.len() {
            self.usage_counts[expert_idx] += 1;
            self.total_tokens += 1;
        }
        self.usage_counts.get(expert_idx).copied().unwrap_or(0) >= self.promotion_threshold
    }

    /// Check if expert is cached on GPU
    #[allow(dead_code)]
    pub(super) fn is_cached(&self, expert_idx: usize) -> bool {
        self.entries.contains_key(&expert_idx)
    }

    /// Get cached expert if available
    pub(super) fn get(&mut self, expert_idx: usize) -> Option<CachedMatmuls> {
        if let Some(entry) = self.entries.get(&expert_idx).cloned() {
            // Update LRU
            self.lru.retain(|&idx| idx != expert_idx);
            self.lru.push_back(expert_idx);
            Some(entry)
        } else {
            None
        }
    }

    /// Add expert to GPU cache
    pub(super) fn insert(&mut self, expert_idx: usize, entry: CachedMatmuls) {
        // Evict if at capacity
        while self.entries.len() >= self.capacity {
            if let Some(oldest) = self.lru.pop_front() {
                self.entries.remove(&oldest);
            } else {
                break;
            }
        }

        self.entries.insert(expert_idx, entry);
        self.lru.retain(|&idx| idx != expert_idx);
        self.lru.push_back(expert_idx);
    }

    /// Build GPU-cached entry from CPU source tensors
    pub(super) fn build_gpu_entry(
        &self,
        expert_idx: usize,
        gate_src: &SharedQTensor,
        up_src: &SharedQTensor,
        down_src: &SharedQTensor,
    ) -> Result<CachedMatmuls> {
        // Slice out the expert
        let gate_slice = gate_src.read().unwrap().slice_first_dim(expert_idx)?;
        let up_slice = up_src.read().unwrap().slice_first_dim(expert_idx)?;
        let down_slice = down_src.read().unwrap().slice_first_dim(expert_idx)?;

        // Copy to GPU
        let gate_gpu = Self::copy_to_gpu(&gate_slice, &self.gpu_device)?;
        let up_gpu = Self::copy_to_gpu(&up_slice, &self.gpu_device)?;
        let down_gpu = Self::copy_to_gpu(&down_slice, &self.gpu_device)?;

        Ok(CachedMatmuls {
            gate: Arc::new(
                paramecia_core::quantized::QMatMul::from_arc(Arc::new(gate_gpu))?.try_into()?,
            ),
            up: Arc::new(
                paramecia_core::quantized::QMatMul::from_arc(Arc::new(up_gpu))?.try_into()?,
            ),
            down: Arc::new(
                paramecia_core::quantized::QMatMul::from_arc(Arc::new(down_gpu))?.try_into()?,
            ),
            gate_device: self.gpu_device.clone(),
            up_device: self.gpu_device.clone(),
            down_device: self.gpu_device.clone(),
        })
    }

    pub(super) fn copy_to_gpu(src: &QTensor, gpu_device: &Device) -> Result<QTensor> {
        // Always copy to ensure we have a GPU version
        let data = src.data()?;
        let storage = QStorage::from_data(data, gpu_device, src.dtype())?;
        QTensor::new(storage, src.shape().clone())
    }

    /// Get the top-k most used experts
    #[allow(dead_code)]
    pub(super) fn top_k_experts(&self, k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, u64)> = self
            .usage_counts
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c))
            .collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));
        indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
    }

    /// Get cache hit rate
    #[allow(dead_code)]
    pub(super) fn hit_rate(&self) -> f64 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        let cached_usage: u64 = self.entries.keys().map(|&idx| self.usage_counts[idx]).sum();
        cached_usage as f64 / self.total_tokens as f64
    }
}

#[derive(Debug)]
pub(super) struct ExpertCache {
    pub(super) target_device: Device,
    pub(super) capacity: usize,
    pub(super) entries: HashMap<usize, CachedMatmuls>,
    pub(super) lru: VecDeque<usize>,
}

impl ExpertCache {
    pub(super) fn new(target_device: Device, capacity: usize) -> Self {
        Self {
            target_device,
            capacity,
            entries: HashMap::new(),
            lru: VecDeque::new(),
        }
    }

    pub(super) fn enabled(&self) -> bool {
        self.capacity > 0
    }

    pub(super) fn touch(&mut self, expert_idx: usize) {
        self.lru.retain(|idx| *idx != expert_idx);
        self.lru.push_back(expert_idx);
    }

    pub(super) fn evict_if_needed(&mut self) {
        while self.entries.len() >= self.capacity {
            if let Some(oldest) = self.lru.pop_front() {
                self.entries.remove(&oldest);
            } else {
                break;
            }
        }
    }

    pub(super) fn materialize_qtensor(&self, src: &Arc<QTensor>) -> Result<Arc<QTensor>> {
        if src.device().same_device(&self.target_device) {
            return Ok(src.clone());
        }

        let data = src.data()?;
        let storage = QStorage::from_data(data, &self.target_device, src.dtype())?;
        let qtensor = QTensor::new(storage, src.shape().clone())?;
        Ok(Arc::new(qtensor))
    }

    pub(super) fn build_entry(
        &self,
        expert_idx: usize,
        gate_src: &SharedQTensor,
        up_src: &SharedQTensor,
        down_src: &SharedQTensor,
    ) -> Result<CachedMatmuls> {
        let gate_slice = gate_src.read().unwrap().slice_first_dim(expert_idx)?;
        let up_slice = up_src.read().unwrap().slice_first_dim(expert_idx)?;
        let down_slice = down_src.read().unwrap().slice_first_dim(expert_idx)?;

        let gate_qtensor = self.materialize_qtensor(&Arc::new(gate_slice))?;
        let up_qtensor = self.materialize_qtensor(&Arc::new(up_slice))?;
        let down_qtensor = self.materialize_qtensor(&Arc::new(down_slice))?;

        Ok(CachedMatmuls {
            gate: Arc::new(paramecia_core::quantized::QMatMul::from_arc(gate_qtensor)?.try_into()?),
            up: Arc::new(paramecia_core::quantized::QMatMul::from_arc(up_qtensor)?.try_into()?),
            down: Arc::new(paramecia_core::quantized::QMatMul::from_arc(down_qtensor)?.try_into()?),
            gate_device: self.target_device.clone(),
            up_device: self.target_device.clone(),
            down_device: self.target_device.clone(),
        })
    }

    pub(super) fn get_or_prepare(
        &mut self,
        expert_idx: usize,
        gate_src: &SharedQTensor,
        up_src: &SharedQTensor,
        down_src: &SharedQTensor,
    ) -> Result<CachedMatmuls> {
        if let Some(entry) = self.entries.get(&expert_idx).cloned() {
            self.touch(expert_idx);
            return Ok(entry);
        }

        self.evict_if_needed();
        let entry = self.build_entry(expert_idx, gate_src, up_src, down_src)?;
        self.entries.insert(expert_idx, entry.clone());
        self.touch(expert_idx);
        Ok(entry)
    }
}

pub(super) fn should_cache_experts(
    gate_exps: &SharedQTensor,
    up_exps: &SharedQTensor,
    down_exps: &SharedQTensor,
    compute_device: &Device,
) -> bool {
    let gate = gate_exps.read().unwrap();
    let up = up_exps.read().unwrap();
    let down = down_exps.read().unwrap();
    !matches!(compute_device, Device::Cpu)
        && (!gate.device().same_device(compute_device)
            || !up.device().same_device(compute_device)
            || !down.device().same_device(compute_device))
}
