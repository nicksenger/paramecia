use crate::{
    backend::BackendStorage, CpuStorage, DType, Device, Layout, Result, Shape, Storage, Tensor, D,
};
use k_quants::*;
use std::borrow::Cow;

#[cfg(target_feature = "avx2")]
pub mod avx;
mod dummy_cuda;
mod dummy_metal;
pub mod ggml_file;
pub mod gguf_file;
pub mod imatrix_file;
pub mod k_quants;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(not(feature = "metal"))]
mod metal {
    pub use super::dummy_metal::*;
}
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(not(feature = "cuda"))]
mod cuda {
    pub use super::dummy_cuda::*;
}
mod dummy_vulkan;
#[cfg(feature = "vulkan")]
pub mod vulkan;
#[cfg(not(feature = "vulkan"))]
mod vulkan {
    pub use super::dummy_vulkan::*;
}

#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(target_feature = "simd128")]
pub mod simd128;
pub mod utils;
use half::{bf16, f16};

pub use k_quants::GgmlType;

// ============================================================================
// Fused Perturbation State (for QuZO training)
// ============================================================================
//
// Thread-local state that enables fused perturbation during forward passes.
// When set, QMatMul::forward will use fused_forward instead of regular forward,
// applying on-the-fly perturbation using the specified seed and epsilon.

use std::cell::RefCell;
use std::collections::HashMap;

/// Perturbation state for fused forward passes.
#[derive(Debug, Clone, Copy)]
pub struct PerturbationState {
    /// Random seed for perturbation generation
    pub seed: u64,
    /// Perturbation magnitude (can be positive or negative)
    pub epsilon: f32,
}

thread_local! {
    static PERTURBATION_STATE: RefCell<Option<PerturbationState>> = const { RefCell::new(None) };
    // Maps the current storage identity to a stable tensor ordinal. Keeping this
    // separate preserves the small Copy perturbation state used by hot forwards.
    static PERTURBATION_TENSOR_ORDINALS: RefCell<Option<HashMap<u64, u64>>> = const { RefCell::new(None) };
}

/// Set the perturbation state for the current thread.
/// When set, QMatMul::forward will use fused perturbation.
pub fn set_perturbation_state(seed: u64, epsilon: f32) {
    PERTURBATION_STATE.with(|state| {
        *state.borrow_mut() = Some(PerturbationState { seed, epsilon });
    });
    PERTURBATION_TENSOR_ORDINALS.with(|ordinals| *ordinals.borrow_mut() = None);
}

/// Set perturbation state for a model-scoped set of tensors.
///
/// Tensor storage identities are used only for lookup. The Philox stream is
/// derived from the stable ordinal, so perturbations remain deterministic when
/// device allocations change across model reloads.
pub fn set_scoped_perturbation_state(seed: u64, epsilon: f32, tensor_ordinals: &[(u64, u64)]) {
    PERTURBATION_STATE.with(|state| {
        *state.borrow_mut() = Some(PerturbationState { seed, epsilon });
    });
    PERTURBATION_TENSOR_ORDINALS.with(|ordinals| {
        *ordinals.borrow_mut() = Some(tensor_ordinals.iter().copied().collect());
    });
}

/// Clear the perturbation state for the current thread.
/// After clearing, QMatMul::forward will use regular (non-perturbed) forward.
pub fn clear_perturbation_state() {
    PERTURBATION_STATE.with(|state| {
        *state.borrow_mut() = None;
    });
    PERTURBATION_TENSOR_ORDINALS.with(|ordinals| *ordinals.borrow_mut() = None);
}

/// Get the current perturbation state for the current thread.
pub fn get_perturbation_state() -> Option<PerturbationState> {
    PERTURBATION_STATE.with(|state| *state.borrow())
}

/// Get perturbation state for a particular tensor storage identity.
///
/// In model-scoped mode tensors outside the registered optimizer set return
/// `None`. The adjusted seed cancels the backend kernel's storage-pointer XOR
/// and replaces it with a stable ordinal-derived stream.
pub fn get_perturbation_state_for_tensor(tensor_id: u64) -> Option<PerturbationState> {
    let state = get_perturbation_state()?;
    PERTURBATION_TENSOR_ORDINALS.with(|ordinals| match ordinals.borrow().as_ref() {
        None => Some(state),
        Some(ordinals) => ordinals.get(&tensor_id).map(|ordinal| PerturbationState {
            seed: state.seed ^ ordinal ^ tensor_id,
            epsilon: state.epsilon,
        }),
    })
}

/// RAII guard that sets perturbation state and clears it on drop.
pub struct PerturbationGuard {
    _private: (),
}

impl PerturbationGuard {
    /// Create a new guard that sets perturbation state.
    pub fn new(seed: u64, epsilon: f32) -> Self {
        set_perturbation_state(seed, epsilon);
        Self { _private: () }
    }
}

impl Drop for PerturbationGuard {
    fn drop(&mut self) {
        clear_perturbation_state();
    }
}

#[cfg(test)]
mod perturbation_state_tests {
    use super::*;

    #[test]
    fn scoped_state_uses_stable_ordinals_and_filters_other_tensors() {
        set_scoped_perturbation_state(7, 0.25, &[(100, 3)]);

        let state = get_perturbation_state_for_tensor(100).expect("registered tensor state");
        assert_eq!(state.seed, 7 ^ 3 ^ 100);
        assert_eq!(state.epsilon, 0.25);
        assert!(get_perturbation_state_for_tensor(101).is_none());

        clear_perturbation_state();
        assert!(get_perturbation_state().is_none());
    }
}

/// Converts a byte slice to a typed slice.
/// SAFETY: Only valid when `data` is Borrowed - the caller must ensure the data outlives the returned slice.
fn as_t_slice_borrowed<T>(data: &[u8]) -> &[T] {
    let size = std::mem::size_of::<T>();
    assert_eq!(
        data.len() % size,
        0,
        "Data length must be a multiple of T's size"
    );
    let ptr = data.as_ptr();
    assert_eq!(
        (ptr as usize) % std::mem::align_of::<T>(),
        0,
        "Data pointer must be aligned to T's alignment"
    );
    unsafe { std::slice::from_raw_parts(ptr as *const T, data.len() / size) }
}

/// Converts owned bytes to an owned Vec of T.
/// This properly handles alignment and copies the data.
fn bytes_to_vec<T: Clone>(data: Vec<u8>) -> Vec<T> {
    let size = std::mem::size_of::<T>();
    assert_eq!(
        data.len() % size,
        0,
        "Data length must be a multiple of T's size"
    );
    let num_elements = data.len() / size;

    // Check alignment - if not aligned, we need to copy byte-by-byte
    let ptr = data.as_ptr();
    if (ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
        // Data is properly aligned, we can reinterpret
        let mut data = std::mem::ManuallyDrop::new(data);
        unsafe { Vec::from_raw_parts(data.as_mut_ptr() as *mut T, num_elements, num_elements) }
    } else {
        // Data is not aligned, copy through a properly aligned buffer
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const T, num_elements) };
        slice.to_vec()
    }
}

pub struct QTensor {
    storage: std::sync::Arc<QStorage>,
    layout: Layout,
}

impl Device {
    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<QStorage> {
        match self {
            Device::Cpu => {
                let storage = dtype.cpu_zeros(elem_count);
                Ok(QStorage::Cpu(storage))
            }
            Device::Metal(metal) => {
                let storage = metal::QMetalStorage::zeros(metal, elem_count, dtype)?;
                Ok(QStorage::Metal(storage))
            }
            Device::Cuda(cuda) => {
                let storage = cuda::QCudaStorage::zeros(cuda, elem_count, dtype)?;
                Ok(QStorage::Cuda(storage))
            }
            Device::Vulkan(vk) => {
                let storage = vulkan::QVulkanStorage::zeros(vk, elem_count, dtype)?;
                Ok(QStorage::Vulkan(storage))
            }
        }
    }
}

pub enum QStorage {
    Cpu(Box<dyn QuantizedType>),
    Metal(metal::QMetalStorage),
    Cuda(cuda::QCudaStorage),
    Vulkan(vulkan::QVulkanStorage),
}

impl QStorage {
    pub fn from_data(data: Cow<'_, [u8]>, device: &Device, dtype: GgmlDType) -> Result<Self> {
        match device {
            Device::Cpu => Ok(Self::Cpu(dtype.from_data(data))),
            Device::Metal(d) => {
                // For Metal/CUDA, we need to get a reference to the bytes.
                // The reference is valid for the duration of load_quantized which copies to GPU.
                let bytes: &[u8] = &data;
                match dtype {
                    GgmlDType::F32 => metal::load_quantized(d, as_t_slice_borrowed::<f32>(bytes)),
                    GgmlDType::F16 => metal::load_quantized(d, as_t_slice_borrowed::<f16>(bytes)),
                    GgmlDType::Q4_0 => {
                        metal::load_quantized(d, as_t_slice_borrowed::<BlockQ4_0>(bytes))
                    }
                    GgmlDType::Q4_1 => {
                        metal::load_quantized(d, as_t_slice_borrowed::<BlockQ4_1>(bytes))
                    }
                    GgmlDType::Q5_0 => {
                        metal::load_quantized(d, as_t_slice_borrowed::<BlockQ5_0>(bytes))
                    }
                    GgmlDType::Q5_1 => {
                        metal::load_quantized(d, as_t_slice_borrowed::<BlockQ5_1>(bytes))
                    }
                    GgmlDType::Q8_0 => {
                        metal::load_quantized(d, as_t_slice_borrowed::<BlockQ8_0>(bytes))
                    }
                    GgmlDType::Q8_1 => {
                        metal::load_quantized(d, as_t_slice_borrowed::<BlockQ8_1>(bytes))
                    }
                    GgmlDType::Q2K => {
                        metal::load_quantized(d, as_t_slice_borrowed::<BlockQ2K>(bytes))
                    }
                    GgmlDType::Q3K => {
                        metal::load_quantized(d, as_t_slice_borrowed::<BlockQ3K>(bytes))
                    }
                    GgmlDType::Q4K => {
                        metal::load_quantized(d, as_t_slice_borrowed::<BlockQ4K>(bytes))
                    }
                    GgmlDType::Q5K => {
                        metal::load_quantized(d, as_t_slice_borrowed::<BlockQ5K>(bytes))
                    }
                    GgmlDType::Q6K => {
                        metal::load_quantized(d, as_t_slice_borrowed::<BlockQ6K>(bytes))
                    }
                    GgmlDType::Q8K => {
                        metal::load_quantized(d, as_t_slice_borrowed::<BlockQ8K>(bytes))
                    }
                    GgmlDType::BF16 => metal::load_quantized(d, as_t_slice_borrowed::<bf16>(bytes)),
                }
            }
            Device::Cuda(d) => {
                // For Metal/CUDA, we need to get a reference to the bytes.
                // The reference is valid for the duration of load_quantized which copies to GPU.
                let bytes: &[u8] = &data;
                match dtype {
                    GgmlDType::F32 => cuda::load_quantized(d, as_t_slice_borrowed::<f32>(bytes)),
                    GgmlDType::F16 => cuda::load_quantized(d, as_t_slice_borrowed::<f16>(bytes)),
                    GgmlDType::Q4_0 => {
                        cuda::load_quantized(d, as_t_slice_borrowed::<BlockQ4_0>(bytes))
                    }
                    GgmlDType::Q4_1 => {
                        cuda::load_quantized(d, as_t_slice_borrowed::<BlockQ4_1>(bytes))
                    }
                    GgmlDType::Q5_0 => {
                        cuda::load_quantized(d, as_t_slice_borrowed::<BlockQ5_0>(bytes))
                    }
                    GgmlDType::Q5_1 => {
                        cuda::load_quantized(d, as_t_slice_borrowed::<BlockQ5_1>(bytes))
                    }
                    GgmlDType::Q8_0 => {
                        cuda::load_quantized(d, as_t_slice_borrowed::<BlockQ8_0>(bytes))
                    }
                    GgmlDType::Q8_1 => {
                        cuda::load_quantized(d, as_t_slice_borrowed::<BlockQ8_1>(bytes))
                    }
                    GgmlDType::Q2K => {
                        cuda::load_quantized(d, as_t_slice_borrowed::<BlockQ2K>(bytes))
                    }
                    GgmlDType::Q3K => {
                        cuda::load_quantized(d, as_t_slice_borrowed::<BlockQ3K>(bytes))
                    }
                    GgmlDType::Q4K => {
                        cuda::load_quantized(d, as_t_slice_borrowed::<BlockQ4K>(bytes))
                    }
                    GgmlDType::Q5K => {
                        cuda::load_quantized(d, as_t_slice_borrowed::<BlockQ5K>(bytes))
                    }
                    GgmlDType::Q6K => {
                        cuda::load_quantized(d, as_t_slice_borrowed::<BlockQ6K>(bytes))
                    }
                    GgmlDType::Q8K => {
                        cuda::load_quantized(d, as_t_slice_borrowed::<BlockQ8K>(bytes))
                    }
                    GgmlDType::BF16 => cuda::load_quantized(d, as_t_slice_borrowed::<bf16>(bytes)),
                }
            }
            Device::Vulkan(d) => {
                let bytes: &[u8] = &data;
                match dtype {
                    GgmlDType::F32 => vulkan::load_quantized(d, as_t_slice_borrowed::<f32>(bytes)),
                    GgmlDType::F16 => vulkan::load_quantized(d, as_t_slice_borrowed::<f16>(bytes)),
                    GgmlDType::Q4_0 => {
                        vulkan::load_quantized(d, as_t_slice_borrowed::<BlockQ4_0>(bytes))
                    }
                    GgmlDType::Q4_1 => {
                        vulkan::load_quantized(d, as_t_slice_borrowed::<BlockQ4_1>(bytes))
                    }
                    GgmlDType::Q5_0 => {
                        vulkan::load_quantized(d, as_t_slice_borrowed::<BlockQ5_0>(bytes))
                    }
                    GgmlDType::Q5_1 => {
                        vulkan::load_quantized(d, as_t_slice_borrowed::<BlockQ5_1>(bytes))
                    }
                    GgmlDType::Q8_0 => {
                        vulkan::load_quantized(d, as_t_slice_borrowed::<BlockQ8_0>(bytes))
                    }
                    GgmlDType::Q8_1 => {
                        vulkan::load_quantized(d, as_t_slice_borrowed::<BlockQ8_1>(bytes))
                    }
                    GgmlDType::Q2K => {
                        vulkan::load_quantized(d, as_t_slice_borrowed::<BlockQ2K>(bytes))
                    }
                    GgmlDType::Q3K => {
                        vulkan::load_quantized(d, as_t_slice_borrowed::<BlockQ3K>(bytes))
                    }
                    GgmlDType::Q4K => {
                        vulkan::load_quantized(d, as_t_slice_borrowed::<BlockQ4K>(bytes))
                    }
                    GgmlDType::Q5K => {
                        vulkan::load_quantized(d, as_t_slice_borrowed::<BlockQ5K>(bytes))
                    }
                    GgmlDType::Q6K => {
                        vulkan::load_quantized(d, as_t_slice_borrowed::<BlockQ6K>(bytes))
                    }
                    GgmlDType::Q8K => {
                        vulkan::load_quantized(d, as_t_slice_borrowed::<BlockQ8K>(bytes))
                    }
                    GgmlDType::BF16 => {
                        vulkan::load_quantized(d, as_t_slice_borrowed::<bf16>(bytes))
                    }
                }
            }
        }
    }

    fn block_size(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.block_size(),
            QStorage::Metal(storage) => storage.dtype().block_size(),
            QStorage::Cuda(storage) => storage.dtype().block_size(),
            QStorage::Vulkan(storage) => storage.dtype().block_size(),
        }
    }

    fn dtype(&self) -> GgmlDType {
        match self {
            QStorage::Cpu(storage) => storage.dtype(),
            QStorage::Metal(storage) => storage.dtype(),
            QStorage::Cuda(storage) => storage.dtype(),
            QStorage::Vulkan(storage) => storage.dtype(),
        }
    }

    fn device(&self) -> Device {
        match self {
            QStorage::Cpu(_storage) => Device::Cpu,
            QStorage::Metal(storage) => Device::Metal(storage.device().clone()),
            QStorage::Cuda(storage) => Device::Cuda(storage.device().clone()),
            QStorage::Vulkan(storage) => Device::Vulkan(storage.device().clone()),
        }
    }

    fn size_in_bytes(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.storage_size_in_bytes(),
            QStorage::Metal(storage) => storage.storage_size_in_bytes(),
            QStorage::Cuda(storage) => storage.storage_size_in_bytes(),
            QStorage::Vulkan(storage) => storage.storage_size_in_bytes(),
        }
    }

    fn quantize(&mut self, src: &Storage) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float(src.as_slice::<f32>()?);
            }
            (QStorage::Metal(storage), Storage::Metal(src)) => storage.quantize(src)?,
            (QStorage::Cuda(storage), Storage::Cuda(src)) => storage.quantize(src)?,
            (QStorage::Vulkan(storage), Storage::Vulkan(src)) => storage.quantize(src)?,
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn quantize_imatrix(
        &mut self,
        src: &Storage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
            }
            (QStorage::Metal(storage), Storage::Metal(src)) => {
                storage.quantize_imatrix(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Cuda(storage), Storage::Cuda(src)) => {
                storage.quantize_imatrix(src, imatrix_weights, n_per_row)?
            }
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn quantize_onto(&mut self, src: &Storage) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float(src.as_slice::<f32>()?);
            }
            (QStorage::Metal(storage), Storage::Cpu(src)) => storage.quantize_onto(src)?,
            (QStorage::Cuda(storage), Storage::Cpu(src)) => storage.quantize_onto(src)?,
            (QStorage::Vulkan(storage), Storage::Cpu(src)) => storage.quantize_onto(src)?,
            _ => crate::bail!("Invalid quantize source storage locations: not on cpu"),
        }
        Ok(())
    }

    fn quantize_imatrix_onto(
        &mut self,
        src: &Storage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
            }
            (QStorage::Metal(storage), Storage::Cpu(src)) => {
                storage.quantize_imatrix_onto(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Cuda(storage), Storage::Cpu(src)) => {
                storage.quantize_imatrix_onto(src, imatrix_weights, n_per_row)?
            }
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    pub(crate) fn dequantize(&self, elem_count: usize) -> Result<Storage> {
        match self {
            QStorage::Cpu(storage) => Ok(Storage::Cpu(storage.dequantize(elem_count)?)),
            QStorage::Metal(storage) => Ok(Storage::Metal(storage.dequantize(elem_count)?)),
            QStorage::Cuda(storage) => Ok(Storage::Cuda(storage.dequantize(elem_count)?)),
            QStorage::Vulkan(storage) => Ok(Storage::Vulkan(storage.dequantize(elem_count)?)),
        }
    }

    fn data(&self) -> Result<Cow<'_, [u8]>> {
        match self {
            QStorage::Cpu(storage) => {
                let data_ptr = storage.as_ptr();
                let size_in_bytes = storage.storage_size_in_bytes();
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
                Ok(Cow::from(data))
            }
            QStorage::Cuda(storage) => Ok(Cow::from(storage.data()?)),
            QStorage::Metal(storage) => Ok(Cow::from(storage.data()?)),
            QStorage::Vulkan(storage) => Ok(Cow::from(storage.data()?)),
        }
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        match self {
            QStorage::Cuda(storage) => storage.device_ptr(),
            QStorage::Metal(_) | QStorage::Cpu(_) | QStorage::Vulkan(_) => {
                crate::bail!("not implemented");
            }
        }
    }

    /// Create a new QStorage that references a slice of the current storage.
    /// The offset and size are in bytes.
    fn slice(&self, offset: usize, size: usize) -> Result<Self> {
        match self {
            QStorage::Cpu(storage) => Ok(QStorage::Cpu(storage.slice(offset, size)?)),
            QStorage::Metal(storage) => Ok(QStorage::Metal(storage.slice(offset, size)?)),
            QStorage::Cuda(storage) => Ok(QStorage::Cuda(storage.slice(offset, size)?)),
            QStorage::Vulkan(storage) => Ok(QStorage::Vulkan(storage.slice(offset, size)?)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl GgmlDType {
    pub(crate) fn from_u32(u: u32) -> Result<Self> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => Self::BF16,
            _ => crate::bail!("unknown dtype for tensor {u}"),
        };
        Ok(dtype)
    }

    pub(crate) fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            Self::BF16 => 30,
        }
    }

    /// The block dtype
    pub fn cpu_zeros(&self, elem_count: usize) -> Box<dyn QuantizedType> {
        match self {
            Self::F32 => Box::new(vec![f32::zeros(); elem_count]),
            Self::F16 => Box::new(vec![f16::zeros(); elem_count]),
            Self::Q4_0 => Box::new(vec![BlockQ4_0::zeros(); elem_count / BlockQ4_0::BLCK_SIZE]),
            Self::Q4_1 => Box::new(vec![BlockQ4_1::zeros(); elem_count / BlockQ4_1::BLCK_SIZE]),
            Self::Q5_0 => Box::new(vec![BlockQ5_0::zeros(); elem_count / BlockQ5_0::BLCK_SIZE]),
            Self::Q5_1 => Box::new(vec![BlockQ5_1::zeros(); elem_count / BlockQ5_1::BLCK_SIZE]),
            Self::Q8_0 => Box::new(vec![BlockQ8_0::zeros(); elem_count / BlockQ8_0::BLCK_SIZE]),
            Self::Q8_1 => Box::new(vec![BlockQ8_1::zeros(); elem_count / BlockQ8_1::BLCK_SIZE]),
            Self::Q2K => Box::new(vec![BlockQ2K::zeros(); elem_count / BlockQ2K::BLCK_SIZE]),
            Self::Q3K => Box::new(vec![BlockQ3K::zeros(); elem_count / BlockQ3K::BLCK_SIZE]),
            Self::Q4K => Box::new(vec![BlockQ4K::zeros(); elem_count / BlockQ4K::BLCK_SIZE]),
            Self::Q5K => Box::new(vec![BlockQ5K::zeros(); elem_count / BlockQ5K::BLCK_SIZE]),
            Self::Q6K => Box::new(vec![BlockQ6K::zeros(); elem_count / BlockQ6K::BLCK_SIZE]),
            Self::Q8K => Box::new(vec![BlockQ8K::zeros(); elem_count / BlockQ8K::BLCK_SIZE]),
            Self::BF16 => Box::new(vec![bf16::zeros(); elem_count]),
        }
    }

    pub fn from_data(&self, data: Cow<'_, [u8]>) -> Box<dyn QuantizedType> {
        // Handle Borrowed vs Owned separately to avoid use-after-free
        match data {
            Cow::Borrowed(bytes) => {
                // Safe: the borrowed data outlives this function call
                match self {
                    Self::F32 => Box::new(as_t_slice_borrowed::<f32>(bytes).to_vec()),
                    Self::F16 => Box::new(as_t_slice_borrowed::<f16>(bytes).to_vec()),
                    Self::Q4_0 => Box::new(as_t_slice_borrowed::<BlockQ4_0>(bytes).to_vec()),
                    Self::Q4_1 => Box::new(as_t_slice_borrowed::<BlockQ4_1>(bytes).to_vec()),
                    Self::Q5_0 => Box::new(as_t_slice_borrowed::<BlockQ5_0>(bytes).to_vec()),
                    Self::Q5_1 => Box::new(as_t_slice_borrowed::<BlockQ5_1>(bytes).to_vec()),
                    Self::Q8_0 => Box::new(as_t_slice_borrowed::<BlockQ8_0>(bytes).to_vec()),
                    Self::Q8_1 => Box::new(as_t_slice_borrowed::<BlockQ8_1>(bytes).to_vec()),
                    Self::Q2K => Box::new(as_t_slice_borrowed::<BlockQ2K>(bytes).to_vec()),
                    Self::Q3K => Box::new(as_t_slice_borrowed::<BlockQ3K>(bytes).to_vec()),
                    Self::Q4K => Box::new(as_t_slice_borrowed::<BlockQ4K>(bytes).to_vec()),
                    Self::Q5K => Box::new(as_t_slice_borrowed::<BlockQ5K>(bytes).to_vec()),
                    Self::Q6K => Box::new(as_t_slice_borrowed::<BlockQ6K>(bytes).to_vec()),
                    Self::Q8K => Box::new(as_t_slice_borrowed::<BlockQ8K>(bytes).to_vec()),
                    Self::BF16 => Box::new(as_t_slice_borrowed::<bf16>(bytes).to_vec()),
                }
            }
            Cow::Owned(bytes) => {
                // Convert owned bytes directly to owned Vec<T> without dangling references
                match self {
                    Self::F32 => Box::new(bytes_to_vec::<f32>(bytes)),
                    Self::F16 => Box::new(bytes_to_vec::<f16>(bytes)),
                    Self::Q4_0 => Box::new(bytes_to_vec::<BlockQ4_0>(bytes)),
                    Self::Q4_1 => Box::new(bytes_to_vec::<BlockQ4_1>(bytes)),
                    Self::Q5_0 => Box::new(bytes_to_vec::<BlockQ5_0>(bytes)),
                    Self::Q5_1 => Box::new(bytes_to_vec::<BlockQ5_1>(bytes)),
                    Self::Q8_0 => Box::new(bytes_to_vec::<BlockQ8_0>(bytes)),
                    Self::Q8_1 => Box::new(bytes_to_vec::<BlockQ8_1>(bytes)),
                    Self::Q2K => Box::new(bytes_to_vec::<BlockQ2K>(bytes)),
                    Self::Q3K => Box::new(bytes_to_vec::<BlockQ3K>(bytes)),
                    Self::Q4K => Box::new(bytes_to_vec::<BlockQ4K>(bytes)),
                    Self::Q5K => Box::new(bytes_to_vec::<BlockQ5K>(bytes)),
                    Self::Q6K => Box::new(bytes_to_vec::<BlockQ6K>(bytes)),
                    Self::Q8K => Box::new(bytes_to_vec::<BlockQ8K>(bytes)),
                    Self::BF16 => Box::new(bytes_to_vec::<bf16>(bytes)),
                }
            }
        }
    }

    /// The type size for blocks in bytes.
    pub fn type_size(&self) -> usize {
        use k_quants::*;
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6K => std::mem::size_of::<BlockQ6K>(),
            Self::Q8K => std::mem::size_of::<BlockQ8K>(),
        }
    }

    /// The block size, i.e. the number of elements stored in each block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 | Self::BF16 => 1,
            Self::Q4_0 => k_quants::QK4_0,
            Self::Q4_1 => k_quants::QK4_1,
            Self::Q5_0 => k_quants::QK5_0,
            Self::Q5_1 => k_quants::QK5_1,
            Self::Q8_0 => k_quants::QK8_0,
            Self::Q8_1 => k_quants::QK8_1,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => k_quants::QK_K,
        }
    }
}

// A version of GgmlType without `vec_dot` so that it can be dyn boxed.
pub trait QuantizedType: Send + Sync {
    fn dtype(&self) -> GgmlDType;
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()>;
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()>;
    /// Fused perturbed matmul for QuZO training.
    ///
    /// Computes: dst = (weights + ε*z) @ lhs
    /// This preserves SIMD benefits by computing: vec_dot(w, x) + ε * dot(z, x)
    fn matmul_t_perturbed(
        &self,
        mkn: (usize, usize, usize),
        lhs: &[f32],
        dst: &mut [f32],
        seed: u64,
        epsilon: f32,
    ) -> Result<()>;
    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage>;
    /// Dequantize with on-the-fly perturbation for fused QuZO training.
    fn dequantize_perturbed(
        &self,
        elem_count: usize,
        seed: u64,
        epsilon: f32,
        dst: &mut [f32],
    ) -> Result<()>;
    fn storage_size_in_bytes(&self) -> usize;
    fn as_ptr(&self) -> *const u8;
    fn block_size(&self) -> usize;
    #[allow(clippy::wrong_self_convention)]
    fn from_float(&mut self, xs: &[f32]);
    #[allow(clippy::wrong_self_convention)]
    fn from_float_imatrix(&mut self, xs: &[f32], imatrix_weights: &[f32], n_per_row: usize);
    fn size(&self) -> usize;
    /// Create a slice of this quantized storage at the given byte offset and size
    fn slice(&self, offset: usize, size: usize) -> Result<Box<dyn QuantizedType>>;

    /// Extract the scaling factors from all blocks as f32 values.
    /// For block types with multiple scales, this extracts the primary scale (d).
    fn extract_scales(&self) -> Vec<f32>;
}

impl<T: k_quants::GgmlType + Send + Sync + 'static> QuantizedType for Vec<T> {
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        k_quants::matmul(mkn, lhs, self.as_slice(), dst)
    }
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()> {
        k_quants::matmul_f16(mkn, lhs, self.as_slice(), dst)
    }
    fn matmul_t_perturbed(
        &self,
        mkn: (usize, usize, usize),
        lhs: &[f32],
        dst: &mut [f32],
        seed: u64,
        epsilon: f32,
    ) -> Result<()> {
        k_quants::matmul_perturbed(mkn, lhs, self.as_slice(), dst, seed, epsilon)
    }

    fn size(&self) -> usize {
        self.len() * core::mem::size_of::<T>()
    }

    fn from_float(&mut self, xs: &[f32]) {
        T::from_float(xs, self)
    }

    fn from_float_imatrix(&mut self, xs: &[f32], imatrix_weights: &[f32], n_per_row: usize) {
        T::from_float_imatrix(xs, self, imatrix_weights, n_per_row)
    }

    fn dtype(&self) -> GgmlDType {
        T::DTYPE
    }

    fn block_size(&self) -> usize {
        T::BLCK_SIZE
    }

    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage> {
        let mut ys = vec![0.0f32; elem_count];
        T::to_float(self.as_slice(), &mut ys);
        Ok(CpuStorage::F32(ys))
    }

    fn dequantize_perturbed(
        &self,
        elem_count: usize,
        seed: u64,
        epsilon: f32,
        dst: &mut [f32],
    ) -> Result<()> {
        if dst.len() < elem_count {
            crate::bail!(
                "dequantize_perturbed: dst buffer too small ({} < {})",
                dst.len(),
                elem_count
            );
        }
        T::to_float_perturbed(self.as_slice(), &mut dst[..elem_count], seed, epsilon);
        Ok(())
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    fn as_ptr(&self) -> *const u8 {
        self.as_ptr() as *const u8
    }

    fn extract_scales(&self) -> Vec<f32> {
        T::extract_scales(self.as_slice())
    }

    fn slice(&self, offset: usize, size: usize) -> Result<Box<dyn QuantizedType>> {
        let block_size = std::mem::size_of::<T>();
        if !offset.is_multiple_of(block_size) {
            crate::bail!(
                "offset {} is not aligned to block size {}",
                offset,
                block_size
            )
        }
        if !size.is_multiple_of(block_size) {
            crate::bail!("size {} is not aligned to block size {}", size, block_size)
        }

        let start_block = offset / block_size;
        let num_blocks = size / block_size;
        let end_block = start_block + num_blocks;

        if end_block > self.len() {
            crate::bail!(
                "slice range {}..{} exceeds storage length {}",
                start_block,
                end_block,
                self.len()
            )
        }

        let sliced = self[start_block..end_block].to_vec();
        Ok(Box::new(sliced))
    }
}

impl Clone for QTensor {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            layout: self.layout.clone(),
        }
    }
}

impl std::fmt::Debug for QTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "QTensor[{:?}; {:?}]", self.layout.shape(), self.dtype())
    }
}

fn check_shape(shape: &Shape, block_size: usize) -> Result<()> {
    let dims = shape.dims();
    if dims.is_empty() {
        crate::bail!("scalar tensor cannot be quantized {shape:?}")
    }
    if !dims[dims.len() - 1].is_multiple_of(block_size) {
        crate::bail!(
            "quantized tensor must have their last dim divisible by block size {shape:?} {}",
            block_size
        )
    }
    Ok(())
}

impl QTensor {
    pub fn new<S: Into<Shape>>(storage: QStorage, shape: S) -> Result<Self> {
        let shape = shape.into();
        check_shape(&shape, storage.block_size())?;
        Ok(Self {
            storage: std::sync::Arc::new(storage),
            layout: Layout::contiguous(shape),
        })
    }

    pub fn quantize(src: &Tensor, dtype: GgmlDType) -> Result<Self> {
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize(&src.storage())?;
        Ok(Self {
            storage: std::sync::Arc::new(storage),
            layout: Layout::contiguous(shape.clone()),
        })
    }

    pub fn quantize_imatrix(
        src: &Tensor,
        imatrix_weights: &[f32],
        dtype: GgmlDType,
    ) -> Result<Self> {
        // (n_per_row/QK_K-1)*QK_K+(QK_K/32-1)*32+32=n_per_row
        // Size of imatrix == last dim of tensor
        let n_per_row = src.dim(D::Minus1)?;
        if imatrix_weights.len() != n_per_row {
            crate::bail!(
                "imatrix weights must have the same length {} as the last dim of src {}",
                imatrix_weights.len(),
                src.dim(D::Minus1)?
            );
        }

        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            );
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize_imatrix(&src.storage(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage: std::sync::Arc::new(storage),
            layout: Layout::contiguous(shape.clone()),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`
    pub fn quantize_imatrix_onto(
        src: &Tensor,
        imatrix_weights: &[f32],
        dtype: GgmlDType,
        dev: &Device,
    ) -> Result<Self> {
        if !src.device().is_cpu() {
            crate::bail!(
                "`quantize_onto` expects a `src` to be on the cpu, got {:?}.",
                src.device()
            )
        }
        // (n_per_row/QK_K-1)*QK_K+(QK_K/32-1)*32+32=n_per_row
        // Size of imatrix == last dim of tensor
        let n_per_row = src.dim(D::Minus1)?;
        if imatrix_weights.len() != n_per_row {
            crate::bail!(
                "imatrix weights must have the same length {} as the last dim of src {}",
                imatrix_weights.len(),
                src.dim(D::Minus1)?
            );
        }
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        // storage is on the `dev`, src is on `cpu`
        let mut storage = dev.qzeros(elem_count, dtype)?;
        storage.quantize_imatrix_onto(&src.storage(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage: std::sync::Arc::new(storage),
            layout: Layout::contiguous(shape.clone()),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`
    pub fn quantize_onto(src: &Tensor, dtype: GgmlDType, dev: &Device) -> Result<Self> {
        if !src.device().is_cpu() {
            crate::bail!(
                "`quantize_onto` expects a `src` to be on the cpu, got {:?}.",
                src.device()
            )
        }
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        // storage is on the `dev`, src is on `cpu`
        let mut storage = dev.qzeros(elem_count, dtype)?;
        storage.quantize_onto(&src.storage())?;
        Ok(Self {
            storage: std::sync::Arc::new(storage),
            layout: Layout::contiguous(shape.clone()),
        })
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    pub fn dtype(&self) -> GgmlDType {
        self.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// Get the CUDA device pointer for this tensor's data as u64.
    /// Returns None if the tensor is not on a CUDA device.
    /// This is used for per-tensor seeding in fused QuZO operations.
    #[cfg(feature = "cuda")]
    pub fn cuda_device_ptr(&self) -> Option<u64> {
        match self.storage.as_ref() {
            QStorage::Cuda(s) => Some(s.device_ptr_u64()),
            _ => None,
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn cuda_device_ptr(&self) -> Option<u64> {
        None
    }

    /// Get a unique per-tensor ID for fused perturbation seeding.
    ///
    /// This must match the tensor_id used by the fused matmul kernel on each backend:
    /// - CUDA: the GPU device pointer
    /// - Metal: `Arc::as_ptr` of the Metal buffer
    /// - CPU: pointer to the raw data
    #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
    pub fn fused_tensor_id(&self) -> u64 {
        match self.storage.as_ref() {
            QStorage::Cpu(s) => s.as_ptr() as u64,
            #[cfg(feature = "cuda")]
            QStorage::Cuda(s) => s.device_ptr_u64(),
            #[cfg(feature = "metal")]
            QStorage::Metal(s) => s.fused_tensor_id(),
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(s) => s.fused_tensor_id(),
            // When compiling with only one GPU feature, the other variant
            // is still present in the enum but unreachable
            #[allow(unreachable_patterns)]
            _ => 0,
        }
    }

    pub fn rank(&self) -> usize {
        self.layout.shape().rank()
    }

    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    pub fn storage(&self) -> &QStorage {
        &self.storage
    }

    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let storage = self.storage.dequantize(self.shape().elem_count())?;
        crate::tensor::from_storage(storage, self.shape().clone()).to_device(device)
    }

    pub fn dequantize_f16(&self, device: &Device) -> Result<Tensor> {
        // In the CUDA case, we have a specialized kernel as this can be useful for volta
        // architectures. https://github.com/huggingface/candle/issues/2136
        match self.storage.as_ref() {
            QStorage::Cuda(s) => {
                let s = s.dequantize_f16(self.shape().elem_count())?;
                crate::tensor::from_storage(Storage::Cuda(s), self.shape().clone())
                    .to_device(device)
            }
            _ => {
                let s = self.dequantize(device)?.to_dtype(crate::DType::F16)?;
                Ok(s)
            }
        }
    }

    /// Extract the primary scaling factors from all quantization blocks.
    /// Returns a 1D tensor containing one f32 scale value per block.
    pub fn extract_scales(&self) -> Result<Tensor> {
        match self.storage.as_ref() {
            QStorage::Cpu(storage) => {
                let scales = storage.extract_scales();
                let len = scales.len();
                Tensor::from_vec(scales, len, &Device::Cpu)
            }
            QStorage::Cuda(storage) => {
                let cuda_storage = storage.extract_scales()?;
                let num_blocks = self.shape().elem_count() / self.storage.block_size();
                let shape = Shape::from(num_blocks);
                Ok(crate::tensor::from_storage(
                    Storage::Cuda(cuda_storage),
                    shape,
                ))
            }
            QStorage::Metal(storage) => {
                let metal_storage = storage.extract_scales(self.shape().elem_count())?;
                let num_blocks = self.shape().elem_count() / self.storage.block_size();
                let shape = Shape::from(num_blocks);
                Ok(crate::tensor::from_storage(
                    Storage::Metal(metal_storage),
                    shape,
                ))
            }
            QStorage::Vulkan(storage) => {
                let vk_storage = storage.extract_scales(self.shape().elem_count())?;
                let num_blocks = self.shape().elem_count() / self.storage.block_size();
                let shape = Shape::from(num_blocks);
                Ok(crate::tensor::from_storage(
                    Storage::Vulkan(vk_storage),
                    shape,
                ))
            }
        }
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.storage.size_in_bytes()
    }

    pub fn data(&self) -> Result<Cow<'_, [u8]>> {
        self.storage.data()
    }

    // ========================================================================
    // View ops — zero-copy, share Arc<QStorage>, return new QTensor with
    // modified layout.
    // ========================================================================

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        Ok(Self {
            storage: self.storage.clone(),
            layout: self.layout.narrow(dim, start, len)?,
        })
    }

    pub fn squeeze(&self, dim: usize) -> Result<Self> {
        let dims = self.layout.dims();
        if dim >= dims.len() {
            crate::bail!("squeeze: dim {dim} out of range for rank {}", dims.len())
        }
        if dims[dim] != 1 {
            return Ok(self.clone());
        }
        let mut new_dims = dims.to_vec();
        let mut strides = self.layout.stride().to_vec();
        new_dims.remove(dim);
        strides.remove(dim);
        Ok(Self {
            storage: self.storage.clone(),
            layout: Layout::new(new_dims.into(), strides, self.layout.start_offset()),
        })
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let mut new_dims = self.layout.dims().to_vec();
        let mut strides = self.layout.stride().to_vec();
        let stride = if dim < strides.len() { strides[dim] } else { 1 };
        new_dims.insert(dim, 1);
        strides.insert(dim, stride);
        Ok(Self {
            storage: self.storage.clone(),
            layout: Layout::new(new_dims.into(), strides, self.layout.start_offset()),
        })
    }

    pub fn transpose(&self, d1: usize, d2: usize) -> Result<Self> {
        Ok(Self {
            storage: self.storage.clone(),
            layout: self.layout.transpose(d1, d2)?,
        })
    }

    pub fn reshape<S: crate::shape::ShapeWithOneHole>(&self, s: S) -> Result<Self> {
        let shape = s.into_shape(self.layout.shape().elem_count())?;
        if !self.is_contiguous() {
            crate::bail!("cannot reshape non-contiguous quantized tensor")
        }
        Ok(Self {
            storage: self.storage.clone(),
            layout: Layout::contiguous_with_offset(shape, self.layout.start_offset()),
        })
    }

    pub fn expand(&self, shape: &Shape) -> Result<Self> {
        Ok(Self {
            storage: self.storage.clone(),
            layout: self.layout.broadcast_as(shape.clone())?,
        })
    }

    pub fn flatten(&self, start_dim: usize, end_dim: usize) -> Result<Self> {
        let dims = self.layout.dims();
        let mut new_dims = Vec::new();
        new_dims.extend_from_slice(&dims[..start_dim]);
        let flat: usize = dims[start_dim..=end_dim].iter().product();
        new_dims.push(flat);
        if end_dim + 1 < dims.len() {
            new_dims.extend_from_slice(&dims[end_dim + 1..]);
        }
        self.reshape(new_dims)
    }

    /// Modify quantization block scales by multiplying each block's scale by a corresponding multiplier.
    /// This enables per-block fine-tuning without dequantization.
    ///
    /// # Arguments
    /// * `multipliers` - Tensor of shape [num_blocks] containing scale multipliers (one per block)
    ///
    /// # Returns
    /// A new QTensor with modified block scales (quantized data unchanged, only scales modified)
    ///
    /// # Example
    /// ```ignore
    /// let qtensor = ...; // Q4K quantized tensor
    /// let multipliers = Tensor::ones(num_blocks, DType::F32, &device)?;
    /// let modified = qtensor.modify_block_scales(&multipliers)?;
    /// ```
    pub fn modify_block_scales(&self, multipliers: &Tensor) -> Result<Self> {
        use half::f16;

        let num_blocks = self.shape().elem_count() / self.storage.block_size();

        // Verify multipliers shape
        if multipliers.elem_count() != num_blocks {
            crate::bail!(
                "Multipliers count mismatch: expected {} blocks, got {}",
                num_blocks,
                multipliers.elem_count()
            );
        }

        // Dispatch to device-specific implementation if available
        match self.storage.as_ref() {
            #[cfg(feature = "cuda")]
            QStorage::Cuda(cuda_storage) => {
                // Ensure multipliers tensor is contiguous to get proper storage layout
                let multipliers_contiguous = multipliers.contiguous()?;
                let storage_guard = multipliers_contiguous.storage_and_layout().0;
                if let Storage::Cuda(mult_storage) = &*storage_guard {
                    let modified_storage = cuda_storage.modify_block_scales(mult_storage)?;
                    return QTensor::new(QStorage::Cuda(modified_storage), self.shape().clone());
                } else {
                    crate::bail!("Multipliers must be on CUDA device when QTensor is on CUDA");
                }
            }
            #[cfg(feature = "metal")]
            QStorage::Metal(metal_storage) => {
                let multipliers_contiguous = multipliers.contiguous()?;
                let storage_guard = multipliers_contiguous.storage_and_layout().0;
                if let Storage::Metal(mult_storage) = &*storage_guard {
                    let modified_storage = metal_storage.modify_block_scales(mult_storage)?;
                    return QTensor::new(QStorage::Metal(modified_storage), self.shape().clone());
                } else {
                    crate::bail!("Multipliers must be on Metal device when QTensor is on Metal");
                }
            }
            _ => {
                // Use CPU path below
            }
        }

        // CPU path: Get multipliers as f32 vec
        let mults = multipliers.to_vec1::<f32>()?;

        // Get raw data
        let data = self.data()?;
        let mut modified_data = data.to_vec();

        // Modify scales based on quantization type
        match self.dtype() {
            GgmlDType::Q4K => {
                // BlockQ4K: scale 'd' (f16) at offset 0
                let block_size_bytes = std::mem::size_of::<BlockQ4K>();
                const SCALE_OFFSET: usize = 0;

                for (block_idx, multiplier) in mults.iter().enumerate() {
                    let offset = block_idx * block_size_bytes + SCALE_OFFSET;
                    if offset + 2 > modified_data.len() {
                        crate::bail!("Block index {} out of bounds", block_idx);
                    }

                    // Read current scale
                    let current =
                        f16::from_le_bytes([modified_data[offset], modified_data[offset + 1]]);

                    // Apply multiplier
                    let new_scale = f16::from_f32(current.to_f32() * multiplier);
                    let new_bytes = new_scale.to_le_bytes();

                    // Write back
                    modified_data[offset] = new_bytes[0];
                    modified_data[offset + 1] = new_bytes[1];
                }
            }
            GgmlDType::Q5K => {
                // BlockQ5K: 176 bytes, scale 'd' (f16) at offset 0
                let block_size_bytes = std::mem::size_of::<BlockQ5K>();
                const SCALE_OFFSET: usize = 0;

                for (block_idx, multiplier) in mults.iter().enumerate() {
                    let offset = block_idx * block_size_bytes + SCALE_OFFSET;
                    if offset + 2 > modified_data.len() {
                        crate::bail!("Block index {} out of bounds", block_idx);
                    }

                    // Read current scale
                    let current =
                        f16::from_le_bytes([modified_data[offset], modified_data[offset + 1]]);

                    // Apply multiplier
                    let new_scale = f16::from_f32(current.to_f32() * multiplier);
                    let new_bytes = new_scale.to_le_bytes();

                    // Write back
                    modified_data[offset] = new_bytes[0];
                    modified_data[offset + 1] = new_bytes[1];
                }
            }
            GgmlDType::Q6K => {
                // BlockQ6K: 210 bytes, scale 'd' (f16) at offset 208 (at the end)
                let block_size_bytes = std::mem::size_of::<BlockQ6K>();
                let scale_offset = block_size_bytes
                    .checked_sub(std::mem::size_of::<f16>())
                    .ok_or_else(|| crate::error::Error::Msg("invalid BlockQ6K size".into()))?;

                for (block_idx, multiplier) in mults.iter().enumerate() {
                    let offset = block_idx * block_size_bytes + scale_offset;
                    if offset + 2 > modified_data.len() {
                        crate::bail!("Block index {} out of bounds", block_idx);
                    }

                    // Read current scale
                    let current =
                        f16::from_le_bytes([modified_data[offset], modified_data[offset + 1]]);

                    // Apply multiplier
                    let new_scale = f16::from_f32(current.to_f32() * multiplier);
                    let new_bytes = new_scale.to_le_bytes();

                    // Write back
                    modified_data[offset] = new_bytes[0];
                    modified_data[offset + 1] = new_bytes[1];
                }
            }
            GgmlDType::Q8K => {
                // BlockQ8K: 292 bytes, scale 'd' (f32) at offset 0
                let block_size_bytes = std::mem::size_of::<BlockQ8K>();
                const SCALE_OFFSET: usize = 0;

                for (block_idx, multiplier) in mults.iter().enumerate() {
                    let offset = block_idx * block_size_bytes + SCALE_OFFSET;
                    if offset + 4 > modified_data.len() {
                        crate::bail!("Block index {} out of bounds", block_idx);
                    }

                    // Read current scale (f32)
                    let current = f32::from_le_bytes([
                        modified_data[offset],
                        modified_data[offset + 1],
                        modified_data[offset + 2],
                        modified_data[offset + 3],
                    ]);

                    // Apply multiplier
                    let new_scale = current * multiplier;
                    let new_bytes = new_scale.to_le_bytes();

                    // Write back
                    modified_data[offset..(4 + offset)].copy_from_slice(&new_bytes);
                }
            }
            GgmlDType::Q2K => {
                // BlockQ2K: 84 bytes
                // Layout: scales[16], qs[64], d(f16), dmin(f16)
                // d is at offset 80 (only modify d, not dmin, to match CUDA kernel behavior)
                let block_size_bytes = std::mem::size_of::<BlockQ2K>();
                const D_OFFSET: usize = 80; // QK_K/16 + QK_K/4 = 16 + 64 = 80

                for (block_idx, multiplier) in mults.iter().enumerate() {
                    let offset = block_idx * block_size_bytes + D_OFFSET;
                    if offset + 2 > modified_data.len() {
                        crate::bail!("Block index {} out of bounds", block_idx);
                    }

                    // Read current d scale
                    let current =
                        f16::from_le_bytes([modified_data[offset], modified_data[offset + 1]]);

                    // Apply multiplier
                    let new_scale = f16::from_f32(current.to_f32() * multiplier);
                    let new_bytes = new_scale.to_le_bytes();

                    // Write back
                    modified_data[offset] = new_bytes[0];
                    modified_data[offset + 1] = new_bytes[1];
                }
            }
            GgmlDType::Q3K => {
                // BlockQ3K: 110 bytes
                // Layout: hmask[32], qs[64], scales[12], d(f16)
                // d is at offset 108
                let block_size_bytes = std::mem::size_of::<BlockQ3K>();
                const D_OFFSET: usize = 108; // QK_K/8 + QK_K/4 + 12 = 32 + 64 + 12 = 108

                for (block_idx, multiplier) in mults.iter().enumerate() {
                    let offset = block_idx * block_size_bytes + D_OFFSET;
                    if offset + 2 > modified_data.len() {
                        crate::bail!("Block index {} out of bounds", block_idx);
                    }

                    // Read current scale
                    let current =
                        f16::from_le_bytes([modified_data[offset], modified_data[offset + 1]]);

                    // Apply multiplier
                    let new_scale = f16::from_f32(current.to_f32() * multiplier);
                    let new_bytes = new_scale.to_le_bytes();

                    // Write back
                    modified_data[offset] = new_bytes[0];
                    modified_data[offset + 1] = new_bytes[1];
                }
            }
            dtype => {
                crate::bail!(
                    "modify_block_scales not yet implemented for {:?}. Supported: Q2K, Q3K, Q4K, Q5K, Q6K, Q8K",
                    dtype
                );
            }
        }

        // Create new QStorage from modified data
        let new_storage =
            QStorage::from_data(Cow::Owned(modified_data), &self.device(), self.dtype())?;

        // Return new QTensor with same shape
        QTensor::new(new_storage, self.shape().clone())
    }

    /// Check if this quantization type supports QuZO-style discrete weight perturbation.
    pub fn supports_quzo(&self) -> bool {
        matches!(
            self.dtype(),
            GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4_0
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
                | GgmlDType::Q8_0
                | GgmlDType::Q8K
        )
    }

    /// Apply stochastic perturbation to discrete quantized weights (QuZO).
    ///
    /// This implements the forward perturbation step of the QuZO algorithm,
    /// directly modifying discrete quantized values using stochastic rounding.
    ///
    /// # Arguments
    /// * `perturbation` - Continuous perturbation tensor (same element count as weights)
    /// * `epsilon` - Perturbation magnitude
    /// * `seed` - Random seed for stochastic rounding
    /// * `add` - If true, add perturbation; if false, subtract
    ///
    /// # Returns
    /// A new QTensor with perturbed weights
    pub fn perturb_weights(
        &self,
        perturbation: &Tensor,
        epsilon: f32,
        seed: u64,
        add: bool,
    ) -> Result<Self> {
        // Verify perturbation size matches
        let elem_count = self.shape().elem_count();
        if perturbation.elem_count() != elem_count {
            crate::bail!(
                "Perturbation size mismatch: expected {}, got {}",
                elem_count,
                perturbation.elem_count()
            );
        }

        // Get perturbation as f32 vec
        let perturb_data = perturbation.flatten_all()?.to_vec1::<f32>()?;

        // Get raw data and make a mutable copy
        let data = self.data()?;
        let mut modified_data = data.to_vec();

        // Apply perturbation based on quantization type
        match self.dtype() {
            GgmlDType::Q4_0 => {
                let block_size = std::mem::size_of::<BlockQ4_0>();
                let num_blocks = modified_data.len() / block_size;

                // Safety: BlockQ4_0 is repr(C) and we control the layout
                let blocks: &mut [BlockQ4_0] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut BlockQ4_0,
                        num_blocks,
                    )
                };

                BlockQ4_0::apply_perturbation_stochastic(blocks, &perturb_data, epsilon, seed, add);
            }
            GgmlDType::Q4K => {
                let block_size = std::mem::size_of::<BlockQ4K>();
                let num_blocks = modified_data.len() / block_size;

                let blocks: &mut [BlockQ4K] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut BlockQ4K,
                        num_blocks,
                    )
                };

                BlockQ4K::apply_perturbation_stochastic(blocks, &perturb_data, epsilon, seed, add);
            }
            GgmlDType::Q8_0 => {
                let block_size = std::mem::size_of::<BlockQ8_0>();
                let num_blocks = modified_data.len() / block_size;

                let blocks: &mut [BlockQ8_0] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut BlockQ8_0,
                        num_blocks,
                    )
                };

                BlockQ8_0::apply_perturbation_stochastic(blocks, &perturb_data, epsilon, seed, add);
            }
            GgmlDType::Q2K => {
                let block_size = std::mem::size_of::<BlockQ2K>();
                let num_blocks = modified_data.len() / block_size;

                let blocks: &mut [BlockQ2K] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut BlockQ2K,
                        num_blocks,
                    )
                };

                BlockQ2K::apply_perturbation_stochastic(blocks, &perturb_data, epsilon, seed, add);
            }
            GgmlDType::Q3K => {
                let block_size = std::mem::size_of::<BlockQ3K>();
                let num_blocks = modified_data.len() / block_size;

                let blocks: &mut [BlockQ3K] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut BlockQ3K,
                        num_blocks,
                    )
                };

                BlockQ3K::apply_perturbation_stochastic(blocks, &perturb_data, epsilon, seed, add);
            }
            GgmlDType::Q5K => {
                let block_size = std::mem::size_of::<BlockQ5K>();
                let num_blocks = modified_data.len() / block_size;

                let blocks: &mut [BlockQ5K] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut BlockQ5K,
                        num_blocks,
                    )
                };

                BlockQ5K::apply_perturbation_stochastic(blocks, &perturb_data, epsilon, seed, add);
            }
            GgmlDType::Q6K => {
                let block_size = std::mem::size_of::<BlockQ6K>();
                let num_blocks = modified_data.len() / block_size;

                let blocks: &mut [BlockQ6K] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut BlockQ6K,
                        num_blocks,
                    )
                };

                BlockQ6K::apply_perturbation_stochastic(blocks, &perturb_data, epsilon, seed, add);
            }
            GgmlDType::Q8K => {
                let block_size = std::mem::size_of::<BlockQ8K>();
                let num_blocks = modified_data.len() / block_size;

                let blocks: &mut [BlockQ8K] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut BlockQ8K,
                        num_blocks,
                    )
                };

                BlockQ8K::apply_perturbation_stochastic(blocks, &perturb_data, epsilon, seed, add);
            }
            GgmlDType::F32 => {
                // F32 is unquantized - just add epsilon * perturbation directly
                // Cast raw bytes to f32 slice
                let floats: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut f32,
                        elem_count,
                    )
                };

                let sign = if add { 1.0f32 } else { -1.0f32 };
                for (val, &perturb) in floats.iter_mut().zip(perturb_data.iter()) {
                    *val += sign * epsilon * perturb;
                }
            }
            GgmlDType::BF16 => {
                let halfs: &mut [half::bf16] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut half::bf16,
                        elem_count,
                    )
                };

                let sign = if add { 1.0f32 } else { -1.0f32 };
                for (val, &perturb) in halfs.iter_mut().zip(perturb_data.iter()) {
                    let f = val.to_f32() + sign * epsilon * perturb;
                    *val = half::bf16::from_f32(f);
                }
            }
            dtype => {
                crate::bail!(
                    "perturb_weights not yet implemented for {:?}. Supported: F32, BF16, Q2K, Q3K, Q4_0, Q4K, Q5K, Q6K, Q8_0, Q8K",
                    dtype
                );
            }
        }

        // Create new QStorage from modified data
        let new_storage =
            QStorage::from_data(Cow::Owned(modified_data), &self.device(), self.dtype())?;

        QTensor::new(new_storage, self.shape().clone())
    }

    /// GPU-accelerated in-place perturbation of quantized weights.
    ///
    /// This is much faster than `perturb_weights` for CUDA tensors as it:
    /// - Performs all operations directly on GPU
    /// - Uses efficient CUDA kernels for stochastic rounding
    /// - Avoids CPU-GPU memory transfers
    ///
    /// # Arguments
    /// * `perturbation` - f32 tensor on GPU (will be moved to GPU if on CPU)
    /// * `epsilon` - Perturbation magnitude
    /// * `seed` - Random seed for stochastic rounding
    /// * `add` - If true, add perturbation; if false, subtract
    ///
    /// # Returns
    /// A new QTensor with perturbed weights (on GPU)
    #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
    pub fn perturb_weights_gpu(
        &self,
        perturbation: &Tensor,
        epsilon: f32,
        seed: u64,
        add: bool,
    ) -> Result<Self> {
        use crate::Storage;

        // Check supported dtypes
        match self.dtype() {
            GgmlDType::Q8_0 | GgmlDType::Q4K | GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q5K | GgmlDType::Q6K | GgmlDType::BF16 => {}
            dtype => crate::bail!(
                "perturb_weights_gpu not supported for {:?}. Supported: Q8_0, Q4K, Q2K, Q3K, Q5K, Q6K, BF16",
                dtype
            ),
        }

        // Verify perturbation size
        let elem_count = self.shape().elem_count();
        if perturbation.elem_count() != elem_count {
            crate::bail!(
                "Perturbation size mismatch: expected {}, got {}",
                elem_count,
                perturbation.elem_count()
            );
        }

        match self.storage.as_ref() {
            #[cfg(feature = "cuda")]
            QStorage::Cuda(cuda_storage) => {
                let perturb_contiguous = perturbation.to_device(&self.device())?.contiguous()?;
                let perturb_storage_guard = perturb_contiguous.storage_and_layout().0;
                let perturb_cuda = match &*perturb_storage_guard {
                    Storage::Cuda(s) => s,
                    _ => crate::bail!("Failed to move perturbation to CUDA"),
                };
                let mut modified_storage = cuda_storage.clone();
                modified_storage.perturb_weights_gpu(perturb_cuda, epsilon, seed, add)?;
                QTensor::new(QStorage::Cuda(modified_storage), self.shape().clone())
            }
            #[cfg(feature = "metal")]
            QStorage::Metal(metal_storage) => {
                let perturb_contiguous = perturbation.to_device(&self.device())?.contiguous()?;
                let perturb_storage_guard = perturb_contiguous.storage_and_layout().0;
                let perturb_metal = match &*perturb_storage_guard {
                    Storage::Metal(s) => s,
                    _ => crate::bail!("Failed to move perturbation to Metal"),
                };
                let mut modified_storage = metal_storage.clone();
                modified_storage.perturb_weights_gpu(perturb_metal, epsilon, seed, add)?;
                QTensor::new(QStorage::Metal(modified_storage), self.shape().clone())
            }
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(vulkan_storage) => {
                let perturb_contiguous = perturbation.to_device(&self.device())?.contiguous()?;
                let perturb_storage_guard = perturb_contiguous.storage_and_layout().0;
                let perturb_vulkan = match &*perturb_storage_guard {
                    Storage::Vulkan(s) => s,
                    _ => crate::bail!("Failed to move perturbation to Vulkan"),
                };
                let mut modified_storage = vulkan_storage.clone();
                modified_storage.perturb_weights_gpu(perturb_vulkan, epsilon, seed, add)?;
                QTensor::new(QStorage::Vulkan(modified_storage), self.shape().clone())
            }
            _ => crate::bail!("perturb_weights_gpu requires CUDA, Metal, or Vulkan storage"),
        }
    }

    /// GPU-accelerated combined restore and update operation.
    ///
    /// Fuses restore (from -ε to 0) and gradient update into a single kernel pass.
    ///
    /// # Arguments
    /// * `perturbation` - f32 tensor on GPU
    /// * `restore_epsilon` - Epsilon for restore step
    /// * `update_scale` - Scale for gradient update (η·μ/n)
    /// * `restore_seed` - Seed for restore perturbation
    /// * `update_seed` - Seed for update (must differ from restore_seed!)
    #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
    pub fn restore_and_update_gpu(
        &self,
        perturbation: &Tensor,
        restore_epsilon: f32,
        update_scale: f32,
        restore_seed: u64,
        update_seed: u64,
    ) -> Result<Self> {
        use crate::Storage;

        // Check supported dtypes
        match self.dtype() {
            GgmlDType::Q8_0 | GgmlDType::Q4K | GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q5K | GgmlDType::Q6K | GgmlDType::BF16 => {}
            dtype => crate::bail!(
                "restore_and_update_gpu not supported for {:?}. Supported: Q8_0, Q4K, Q2K, Q3K, Q5K, Q6K, BF16",
                dtype
            ),
        }

        let elem_count = self.shape().elem_count();
        if perturbation.elem_count() != elem_count {
            crate::bail!(
                "Perturbation size mismatch: expected {}, got {}",
                elem_count,
                perturbation.elem_count()
            );
        }

        match self.storage.as_ref() {
            #[cfg(feature = "cuda")]
            QStorage::Cuda(cuda_storage) => {
                let perturb_contiguous = perturbation.to_device(&self.device())?.contiguous()?;
                let perturb_storage_guard = perturb_contiguous.storage_and_layout().0;
                let perturb_cuda = match &*perturb_storage_guard {
                    Storage::Cuda(s) => s,
                    _ => crate::bail!("Failed to move perturbation to CUDA"),
                };
                let mut modified_storage = cuda_storage.clone();
                modified_storage.restore_and_update_gpu(
                    perturb_cuda,
                    restore_epsilon,
                    update_scale,
                    restore_seed,
                    update_seed,
                )?;
                QTensor::new(QStorage::Cuda(modified_storage), self.shape().clone())
            }
            #[cfg(feature = "metal")]
            QStorage::Metal(metal_storage) => {
                let perturb_contiguous = perturbation.to_device(&self.device())?.contiguous()?;
                let perturb_storage_guard = perturb_contiguous.storage_and_layout().0;
                let perturb_metal = match &*perturb_storage_guard {
                    Storage::Metal(s) => s,
                    _ => crate::bail!("Failed to move perturbation to Metal"),
                };
                let mut modified_storage = metal_storage.clone();
                modified_storage.restore_and_update_gpu(
                    perturb_metal,
                    restore_epsilon,
                    update_scale,
                    restore_seed,
                    update_seed,
                )?;
                QTensor::new(QStorage::Metal(modified_storage), self.shape().clone())
            }
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(vulkan_storage) => {
                let perturb_contiguous = perturbation.to_device(&self.device())?.contiguous()?;
                let perturb_storage_guard = perturb_contiguous.storage_and_layout().0;
                let perturb_vulkan = match &*perturb_storage_guard {
                    Storage::Vulkan(s) => s,
                    _ => crate::bail!("Failed to move perturbation to Vulkan"),
                };
                let mut modified_storage = vulkan_storage.clone();
                modified_storage.restore_and_update_gpu(
                    perturb_vulkan,
                    restore_epsilon,
                    update_scale,
                    restore_seed,
                    update_seed,
                )?;
                QTensor::new(QStorage::Vulkan(modified_storage), self.shape().clone())
            }
            _ => crate::bail!("restore_and_update_gpu requires CUDA, Metal, or Vulkan storage"),
        }
    }

    /// Apply a quantized gradient update to discrete weights (QuZO update step).
    ///
    /// This implements: w̄ ← w̄ - Q(η·μ/n · u_{i,2})
    /// Uses a different seed than the forward perturbation to maintain unbiased gradients.
    ///
    /// # Arguments
    /// * `perturbation` - Same continuous perturbation u_i used in forward pass
    /// * `scale` - Update scale (η·μ/n where η=lr, μ=directional derivative)
    /// * `seed` - Random seed (MUST differ from forward pass seed!)
    ///
    /// # Returns
    /// A new QTensor with updated weights
    pub fn apply_quantized_update(
        &self,
        perturbation: &Tensor,
        scale: f32,
        seed: u64,
    ) -> Result<Self> {
        // This is equivalent to perturb_weights with negated scale
        // w̄ ← w̄ - Q(scale · u) = w̄ + Q(-scale · u)
        self.perturb_weights(perturbation, -scale, seed, true)
    }

    /// Apply a quantized update with accumulated error feedback.
    ///
    /// Returns a new QTensor with updated weights. Residuals are modified in-place.
    pub fn apply_quantized_update_with_residual(
        &self,
        perturbation: &Tensor,
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) -> Result<Self> {
        // Unquantized tensors have no rounding error, so residual feedback is not needed.
        if matches!(self.dtype(), GgmlDType::F32 | GgmlDType::BF16) {
            return self.apply_quantized_update(perturbation, scale, seed);
        }

        let elem_count = self.shape().elem_count();
        if perturbation.elem_count() != elem_count {
            crate::bail!(
                "Perturbation size mismatch: expected {}, got {}",
                elem_count,
                perturbation.elem_count()
            );
        }

        let perturb_data = perturbation.flatten_all()?.to_vec1::<f32>()?;
        let data = self.data()?;
        let mut modified_data = data.to_vec();

        macro_rules! dispatch_update_residual {
            ($block_ty:ty) => {{
                let block_size = std::mem::size_of::<$block_ty>();
                let num_blocks = modified_data.len() / block_size;
                let blocks: &mut [$block_ty] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut $block_ty,
                        num_blocks,
                    )
                };
                <$block_ty>::apply_quantized_update_with_residual(
                    blocks,
                    &perturb_data,
                    scale,
                    seed,
                    residuals,
                    gain,
                );
            }};
        }

        match self.dtype() {
            GgmlDType::Q4_0 => dispatch_update_residual!(BlockQ4_0),
            GgmlDType::Q4K => dispatch_update_residual!(BlockQ4K),
            GgmlDType::Q8_0 => dispatch_update_residual!(BlockQ8_0),
            GgmlDType::Q2K => dispatch_update_residual!(BlockQ2K),
            GgmlDType::Q3K => dispatch_update_residual!(BlockQ3K),
            GgmlDType::Q5K => dispatch_update_residual!(BlockQ5K),
            GgmlDType::Q6K => dispatch_update_residual!(BlockQ6K),
            GgmlDType::Q8K => dispatch_update_residual!(BlockQ8K),
            dtype => {
                crate::bail!(
                    "apply_quantized_update_with_residual not supported for {:?}",
                    dtype
                );
            }
        }

        let new_storage =
            QStorage::from_data(Cow::Owned(modified_data), &self.device(), self.dtype())?;
        QTensor::new(new_storage, self.shape().clone())
    }

    /// Combined restore-and-update operation for QuZO (more efficient than separate calls).
    ///
    /// This performs two operations in a single pass over the data:
    /// 1. Restore: w̄ ← w̄ + ε·Q(u) (go from -ε to 0)
    /// 2. Update: w̄ ← w̄ - Q(scale·u) (apply gradient update)
    ///
    /// This is equivalent to calling `perturb_weights(+ε)` followed by `apply_quantized_update(scale)`,
    /// but does both in a single pass over the quantized data, halving the work.
    ///
    /// # Arguments
    /// * `perturbation` - Continuous perturbation tensor
    /// * `restore_epsilon` - Epsilon for restore step (positive to add back)
    /// * `update_scale` - Scale for gradient update (η·μ/n)
    /// * `restore_seed` - Seed for restore perturbation (seed_forward from QuZO)
    /// * `update_seed` - Seed for update (seed_update from QuZO, must differ from restore_seed!)
    ///
    /// # Returns
    /// A new QTensor with restored and updated weights
    pub fn restore_and_update(
        &self,
        perturbation: &Tensor,
        restore_epsilon: f32,
        update_scale: f32,
        restore_seed: u64,
        update_seed: u64,
    ) -> Result<Self> {
        let elem_count = self.shape().elem_count();
        if perturbation.elem_count() != elem_count {
            crate::bail!(
                "Perturbation size mismatch: expected {}, got {}",
                elem_count,
                perturbation.elem_count()
            );
        }

        // Get perturbation as f32 vec
        let perturb_data = perturbation.flatten_all()?.to_vec1::<f32>()?;

        // Get raw data and make a mutable copy
        let data = self.data()?;
        let mut modified_data = data.to_vec();

        // Apply combined restore+update based on quantization type
        match self.dtype() {
            GgmlDType::Q4_0 => {
                let block_size = std::mem::size_of::<BlockQ4_0>();
                let num_blocks = modified_data.len() / block_size;
                let blocks: &mut [BlockQ4_0] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut BlockQ4_0,
                        num_blocks,
                    )
                };
                BlockQ4_0::apply_restore_and_update(
                    blocks,
                    &perturb_data,
                    restore_epsilon,
                    update_scale,
                    restore_seed,
                    update_seed,
                );
            }
            GgmlDType::Q4K => {
                let block_size = std::mem::size_of::<BlockQ4K>();
                let num_blocks = modified_data.len() / block_size;
                let blocks: &mut [BlockQ4K] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut BlockQ4K,
                        num_blocks,
                    )
                };
                BlockQ4K::apply_restore_and_update(
                    blocks,
                    &perturb_data,
                    restore_epsilon,
                    update_scale,
                    restore_seed,
                    update_seed,
                );
            }
            GgmlDType::Q8_0 => {
                let block_size = std::mem::size_of::<BlockQ8_0>();
                let num_blocks = modified_data.len() / block_size;
                let blocks: &mut [BlockQ8_0] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut BlockQ8_0,
                        num_blocks,
                    )
                };
                BlockQ8_0::apply_restore_and_update(
                    blocks,
                    &perturb_data,
                    restore_epsilon,
                    update_scale,
                    restore_seed,
                    update_seed,
                );
            }
            GgmlDType::Q2K => {
                let block_size = std::mem::size_of::<BlockQ2K>();
                let num_blocks = modified_data.len() / block_size;
                let blocks: &mut [BlockQ2K] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut BlockQ2K,
                        num_blocks,
                    )
                };
                BlockQ2K::apply_restore_and_update(
                    blocks,
                    &perturb_data,
                    restore_epsilon,
                    update_scale,
                    restore_seed,
                    update_seed,
                );
            }
            GgmlDType::Q3K => {
                let block_size = std::mem::size_of::<BlockQ3K>();
                let num_blocks = modified_data.len() / block_size;
                let blocks: &mut [BlockQ3K] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut BlockQ3K,
                        num_blocks,
                    )
                };
                BlockQ3K::apply_restore_and_update(
                    blocks,
                    &perturb_data,
                    restore_epsilon,
                    update_scale,
                    restore_seed,
                    update_seed,
                );
            }
            GgmlDType::Q5K => {
                let block_size = std::mem::size_of::<BlockQ5K>();
                let num_blocks = modified_data.len() / block_size;
                let blocks: &mut [BlockQ5K] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut BlockQ5K,
                        num_blocks,
                    )
                };
                BlockQ5K::apply_restore_and_update(
                    blocks,
                    &perturb_data,
                    restore_epsilon,
                    update_scale,
                    restore_seed,
                    update_seed,
                );
            }
            GgmlDType::Q6K => {
                let block_size = std::mem::size_of::<BlockQ6K>();
                let num_blocks = modified_data.len() / block_size;
                let blocks: &mut [BlockQ6K] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut BlockQ6K,
                        num_blocks,
                    )
                };
                BlockQ6K::apply_restore_and_update(
                    blocks,
                    &perturb_data,
                    restore_epsilon,
                    update_scale,
                    restore_seed,
                    update_seed,
                );
            }
            GgmlDType::Q8K => {
                let block_size = std::mem::size_of::<BlockQ8K>();
                let num_blocks = modified_data.len() / block_size;
                let blocks: &mut [BlockQ8K] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut BlockQ8K,
                        num_blocks,
                    )
                };
                BlockQ8K::apply_restore_and_update(
                    blocks,
                    &perturb_data,
                    restore_epsilon,
                    update_scale,
                    restore_seed,
                    update_seed,
                );
            }
            GgmlDType::F32 => {
                // F32 is unquantized - directly apply restore and update
                // restore: w += restore_epsilon * perturbation
                // update:  w -= update_scale * perturbation
                // Combined: w += (restore_epsilon - update_scale) * perturbation
                let floats: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut f32,
                        elem_count,
                    )
                };

                let combined_scale = restore_epsilon - update_scale;
                for (val, &perturb) in floats.iter_mut().zip(perturb_data.iter()) {
                    *val += combined_scale * perturb;
                }
            }
            GgmlDType::BF16 => {
                let halfs: &mut [half::bf16] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut half::bf16,
                        elem_count,
                    )
                };

                let combined_scale = restore_epsilon - update_scale;
                for (val, &perturb) in halfs.iter_mut().zip(perturb_data.iter()) {
                    let f = val.to_f32() + combined_scale * perturb;
                    *val = half::bf16::from_f32(f);
                }
            }
            dtype => {
                crate::bail!(
                    "restore_and_update not yet implemented for {:?}. Supported: F32, BF16, Q2K, Q3K, Q4_0, Q4K, Q5K, Q6K, Q8_0, Q8K",
                    dtype
                );
            }
        }

        let new_storage =
            QStorage::from_data(Cow::Owned(modified_data), &self.device(), self.dtype())?;
        QTensor::new(new_storage, self.shape().clone())
    }

    /// Combined restore-and-update with accumulated error feedback.
    ///
    /// Same as `restore_and_update` but uses residual-biased stochastic rounding.
    /// The `gain` parameter controls residual influence (0=pure stochastic, 1=full feedback).
    /// Always uses CPU path (residuals live on CPU).
    #[allow(clippy::too_many_arguments)]
    pub fn restore_and_update_with_residual(
        &self,
        perturbation: &Tensor,
        restore_epsilon: f32,
        update_scale: f32,
        restore_seed: u64,
        update_seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) -> Result<Self> {
        // Unquantized tensors have no rounding error, so residual feedback is not needed.
        if matches!(self.dtype(), GgmlDType::F32 | GgmlDType::BF16) {
            return self.restore_and_update(
                perturbation,
                restore_epsilon,
                update_scale,
                restore_seed,
                update_seed,
            );
        }

        let elem_count = self.shape().elem_count();
        if perturbation.elem_count() != elem_count {
            crate::bail!(
                "Perturbation size mismatch: expected {}, got {}",
                elem_count,
                perturbation.elem_count()
            );
        }

        let perturb_data = perturbation.flatten_all()?.to_vec1::<f32>()?;
        let data = self.data()?;
        let mut modified_data = data.to_vec();

        macro_rules! dispatch_restore_update_residual {
            ($block_ty:ty, $dtype_variant:ident) => {{
                let block_size = std::mem::size_of::<$block_ty>();
                let num_blocks = modified_data.len() / block_size;
                let blocks: &mut [$block_ty] = unsafe {
                    std::slice::from_raw_parts_mut(
                        modified_data.as_mut_ptr() as *mut $block_ty,
                        num_blocks,
                    )
                };
                <$block_ty>::apply_restore_and_update_with_residual(
                    blocks,
                    &perturb_data,
                    restore_epsilon,
                    update_scale,
                    restore_seed,
                    update_seed,
                    residuals,
                    gain,
                );
            }};
        }

        match self.dtype() {
            GgmlDType::Q4_0 => dispatch_restore_update_residual!(BlockQ4_0, Q4_0),
            GgmlDType::Q4K => dispatch_restore_update_residual!(BlockQ4K, Q4K),
            GgmlDType::Q8_0 => dispatch_restore_update_residual!(BlockQ8_0, Q8_0),
            GgmlDType::Q2K => dispatch_restore_update_residual!(BlockQ2K, Q2K),
            GgmlDType::Q3K => dispatch_restore_update_residual!(BlockQ3K, Q3K),
            GgmlDType::Q5K => dispatch_restore_update_residual!(BlockQ5K, Q5K),
            GgmlDType::Q6K => dispatch_restore_update_residual!(BlockQ6K, Q6K),
            GgmlDType::Q8K => dispatch_restore_update_residual!(BlockQ8K, Q8K),
            dtype => {
                //crate::bail!(
                //    "restore_and_update_with_residual not supported for {:?}",
                //    dtype
                //);
            }
        }

        let new_storage =
            QStorage::from_data(Cow::Owned(modified_data), &self.device(), self.dtype())?;
        QTensor::new(new_storage, self.shape().clone())
    }

    /// Simulate a quantized update with residual tracking (read-only weights).
    ///
    /// Used for replay reconstruction. Does not modify weights, only updates residuals.
    pub fn simulate_update_with_residual(
        &self,
        perturbation: &[f32],
        scale: f32,
        seed: u64,
        residuals: &mut [f16],
        gain: f32,
    ) -> Result<()> {
        // Unquantized tensors have no quantization residual to track.
        if matches!(self.dtype(), GgmlDType::F32 | GgmlDType::BF16) {
            return Ok(());
        }

        let data = self.data()?;

        macro_rules! dispatch_simulate_residual {
            ($block_ty:ty) => {{
                let block_size = std::mem::size_of::<$block_ty>();
                let num_blocks = data.len() / block_size;
                let blocks: &[$block_ty] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const $block_ty, num_blocks)
                };
                <$block_ty>::simulate_update_with_residual(
                    blocks,
                    perturbation,
                    scale,
                    seed,
                    residuals,
                    gain,
                );
            }};
        }

        match self.dtype() {
            GgmlDType::Q4_0 => dispatch_simulate_residual!(BlockQ4_0),
            GgmlDType::Q4K => dispatch_simulate_residual!(BlockQ4K),
            GgmlDType::Q8_0 => dispatch_simulate_residual!(BlockQ8_0),
            GgmlDType::Q2K => dispatch_simulate_residual!(BlockQ2K),
            GgmlDType::Q3K => dispatch_simulate_residual!(BlockQ3K),
            GgmlDType::Q5K => dispatch_simulate_residual!(BlockQ5K),
            GgmlDType::Q6K => dispatch_simulate_residual!(BlockQ6K),
            GgmlDType::Q8K => dispatch_simulate_residual!(BlockQ8K),
            dtype => {
                crate::bail!(
                    "simulate_update_with_residual not supported for {:?}",
                    dtype
                );
            }
        }

        Ok(())
    }

    /// Fused dequantize + perturb + matmul operation (GPU only).
    ///
    /// Computes: y = (W + ε*z) @ x where z is generated on-the-fly from seed.
    /// This avoids storing perturbed weights, which is ideal for:
    /// - CPU-offloaded experts where we don't want to copy weights twice
    /// - Memory-constrained training where we can't store both original and perturbed weights
    ///
    /// The perturbation z is generated deterministically using Philox RNG from the seed
    /// and weight index, so the same (seed, weight_index) always produces the same z value.
    ///
    /// # Arguments
    /// * `x` - Input tensor to multiply with (will be broadcast/transposed as needed)
    /// * `seed` - Random seed for generating perturbation z
    /// * `epsilon` - Perturbation magnitude ε
    ///
    /// # Returns
    /// The result of the fused operation as a regular Tensor
    #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
    pub fn fused_fwd(&self, x: &Tensor, seed: u64, epsilon: f32) -> Result<Tensor> {
        use crate::Storage;

        #[cfg(feature = "cuda")]
        if let QStorage::Cuda(cuda_storage) = self.storage.as_ref() {
            // Check supported dtypes
            match self.dtype() {
                GgmlDType::Q8_0 | GgmlDType::Q4K | GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q5K | GgmlDType::Q6K | GgmlDType::BF16 => {}
                dtype => crate::bail!(
                    "fused_fwd not supported for {:?}. Supported: Q8_0, Q4K, Q2K, Q3K, Q5K, Q6K, BF16",
                    dtype
                ),
            }

            let x_contiguous = x.to_device(&self.device())?.contiguous()?;
            let x_storage_guard = x_contiguous.storage_and_layout();
            let (x_storage, x_layout) = (&*x_storage_guard.0, x_storage_guard.1);
            let x_cuda = match x_storage {
                Storage::Cuda(s) => s,
                _ => crate::bail!("Failed to move input to CUDA"),
            };

            let (out_storage, out_shape) =
                cuda_storage.fused_fwd(self.shape(), x_cuda, x_layout, seed, epsilon)?;

            return Ok(crate::tensor::from_storage(
                Storage::Cuda(out_storage),
                out_shape,
            ));
        }

        #[cfg(feature = "metal")]
        if let QStorage::Metal(metal_storage) = self.storage.as_ref() {
            let x_contiguous = x.to_device(&self.device())?.contiguous()?;
            let x_storage_guard = x_contiguous.storage_and_layout();
            let (x_storage, x_layout) = (&*x_storage_guard.0, x_storage_guard.1);
            let x_metal = match x_storage {
                Storage::Metal(s) => s,
                _ => crate::bail!("Failed to move input to Metal"),
            };

            let (out_storage, out_shape) =
                metal_storage.fused_fwd(self.shape(), x_metal, x_layout, seed, epsilon)?;

            return Ok(crate::tensor::from_storage(
                Storage::Metal(out_storage),
                out_shape,
            ));
        }

        #[cfg(feature = "vulkan")]
        if let QStorage::Vulkan(vulkan_storage) = self.storage.as_ref() {
            let x_contiguous = x.to_device(&self.device())?.contiguous()?;
            let x_storage_guard = x_contiguous.storage_and_layout();
            let (x_storage, x_layout) = (&*x_storage_guard.0, x_storage_guard.1);
            let x_vulkan = match x_storage {
                Storage::Vulkan(s) => s,
                _ => crate::bail!("Failed to move input to Vulkan"),
            };

            let (out_storage, out_shape) =
                vulkan_storage.fused_fwd(self.shape(), x_vulkan, x_layout, seed, epsilon)?;

            return Ok(crate::tensor::from_storage(
                Storage::Vulkan(out_storage),
                out_shape,
            ));
        }

        crate::bail!("fused_fwd requires CUDA, Metal, or Vulkan storage")
    }

    /// CPU fused dequantize + perturb + matmul operation.
    ///
    /// This dequantizes the weights with on-the-fly perturbation, then performs
    /// a regular matmul. Less efficient than CUDA fused kernel but allows
    /// fused QuZO training for CPU-offloaded weights.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `seed` - Philox seed for perturbation
    /// * `epsilon` - Perturbation magnitude
    pub fn fused_cpu_forward(&self, x: &Tensor, seed: u64, epsilon: f32) -> Result<Tensor> {
        let cpu_storage = match self.storage.as_ref() {
            QStorage::Cpu(s) => s,
            _ => crate::bail!("fused_cpu_forward requires CPU storage"),
        };

        // Get tensor pointer for per-tensor seeding
        let tensor_ptr = cpu_storage.as_ptr() as u64;
        let effective_seed = seed ^ tensor_ptr;

        // Get dimensions: weight is (out_dim, in_dim), x is (..., in_dim)
        let w_dims = self.shape().dims();
        if w_dims.len() != 2 {
            crate::bail!(
                "fused_cpu_forward expects 2D weight tensor, got {:?}",
                w_dims
            );
        }
        let (out_dim, in_dim) = (w_dims[0], w_dims[1]);

        // Flatten x to 2D: (batch_size, in_dim)
        let x_dims = x.dims();
        let x_cpu = x.to_device(&Device::Cpu)?;
        let x_f32 = x_cpu.to_dtype(DType::F32)?;
        let in_dtype = x.dtype();
        let original_device = x.device().clone();

        let (batch_size, x_2d) = if x_dims.len() == 1 {
            // (in_dim,) -> (1, in_dim)
            (1, x_f32.reshape((1, in_dim))?)
        } else if x_dims.len() == 2 {
            // (m, in_dim) -> (m, in_dim)
            (x_dims[0], x_f32.clone())
        } else {
            // (..., in_dim) -> (prod(...), in_dim)
            let batch: usize = x_dims[..x_dims.len() - 1].iter().product();
            (batch, x_f32.reshape((batch, in_dim))?)
        };

        // Get lhs as contiguous f32 slice
        let x_storage = x_2d.storage_and_layout().0;
        let lhs = match &*x_storage {
            crate::Storage::Cpu(cpu) => cpu.as_slice::<f32>()?,
            _ => crate::bail!("expected CPU storage for lhs"),
        };

        // Allocate output: (batch_size, out_dim)
        let mut dst = vec![0f32; batch_size * out_dim];

        // Use fused perturbed matmul: preserves SIMD benefits
        // mkn: m = batch_size (rows of lhs), k = in_dim, n = out_dim (rows of transposed weights)
        cpu_storage.matmul_t_perturbed(
            (batch_size, in_dim, out_dim),
            lhs,
            &mut dst,
            effective_seed,
            epsilon,
        )?;

        // Reshape back to original batch dimensions
        let result = Tensor::from_vec(dst, (batch_size, out_dim), &Device::Cpu)?;
        let result = if x_dims.len() == 1 {
            result.reshape(out_dim)?
        } else if x_dims.len() == 2 {
            result
        } else {
            let mut new_shape: Vec<usize> = x_dims[..x_dims.len() - 1].to_vec();
            new_shape.push(out_dim);
            result.reshape(new_shape)?
        };

        // Convert back to original dtype and device
        result.to_dtype(in_dtype)?.to_device(&original_device)
    }

    /// Slice the first dimension of the quantized tensor, keeping it quantized.
    /// For a tensor with shape [n, ...], this extracts the i-th slice with shape [...].
    /// This only works when the slice aligns with quantization block boundaries.
    pub fn slice_first_dim(&self, index: usize) -> Result<Self> {
        let dims = self.shape().dims();
        if dims.is_empty() {
            crate::bail!("cannot slice a scalar qtensor")
        }
        if index >= dims[0] {
            crate::bail!(
                "index {} is out of bounds for dimension 0 with size {}",
                index,
                dims[0]
            )
        }

        // Calculate the number of elements per slice
        let elems_per_slice: usize = dims[1..].iter().product();
        let block_size = self.storage.block_size();

        // Check that the slice size is divisible by block size
        if !elems_per_slice.is_multiple_of(block_size) {
            crate::bail!(
                "slice size {} is not divisible by block size {}",
                elems_per_slice,
                block_size
            )
        }

        // Calculate byte offset and size for this slice
        let blocks_per_slice = elems_per_slice / block_size;
        let bytes_per_block = self.dtype().type_size();
        let bytes_per_slice = blocks_per_slice * bytes_per_block;
        let byte_offset = index * bytes_per_slice;

        // Create new storage with the sliced data
        let new_storage = self.storage.slice(byte_offset, bytes_per_slice)?;

        // Create new shape without the first dimension
        let new_shape = Shape::from(&dims[1..]);

        Ok(Self {
            storage: std::sync::Arc::new(new_storage),
            layout: Layout::contiguous(new_shape),
        })
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match self.storage.as_ref() {
            QStorage::Vulkan(s) => {
                #[cfg(feature = "vulkan")]
                if crate::vulkan_backend::debug::force_cpu_moe(s.dtype()) {
                    return self.indexed_moe_forward_cpu(x, ids);
                }
                match (&*x.storage(), &*ids.storage()) {
                    (Storage::Vulkan(x_s), Storage::Vulkan(ids_s)) => {
                        let (storage, shape) = s.indexed_moe_forward(
                            self.shape(),
                            x_s,
                            x.layout(),
                            ids_s,
                            ids.layout(),
                        )?;
                        Ok(crate::tensor::from_storage(Storage::Vulkan(storage), shape))
                    }
                    _ => self.indexed_moe_forward_cpu(x, ids),
                }
            }
            QStorage::Cuda(s) => {
                // Non-quantized dtypes (F32, F16, BF16) don't have indexed MoE CUDA
                // kernels — fall back to CPU
                if matches!(s.dtype(), GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16) {
                    return self.indexed_moe_forward_cpu(x, ids);
                }
                match (&*x.storage(), &*ids.storage()) {
                    (Storage::Cuda(x_storage), Storage::Cuda(ids_storage)) => {
                        let (storage, out_shape) = s.indexed_moe_forward(
                            self.shape(),
                            x_storage,
                            x.layout(),
                            ids_storage,
                            ids.layout(),
                        )?;
                        Ok(crate::tensor::from_storage(
                            Storage::Cuda(storage),
                            out_shape,
                        ))
                    }
                    _ => {
                        // Fall back to CPU implementation
                        self.indexed_moe_forward_cpu(x, ids)
                    }
                }
            }
            #[cfg(feature = "metal")]
            QStorage::Metal(s) => match (&*x.storage(), &*ids.storage()) {
                (Storage::Metal(x_storage), Storage::Metal(ids_storage)) => {
                    let (storage, out_shape) = s.indexed_moe_forward(
                        self.shape(),
                        x_storage,
                        x.layout(),
                        ids_storage,
                        ids.layout(),
                    )?;
                    Ok(crate::tensor::from_storage(
                        Storage::Metal(storage),
                        out_shape,
                    ))
                }
                _ => {
                    // Fall back to CPU implementation
                    self.indexed_moe_forward_cpu(x, ids)
                }
            },
            #[cfg(not(feature = "metal"))]
            QStorage::Metal(_) => {
                crate::bail!("indexed_moe_forward is not implemented for Metal (feature disabled)");
            }
            QStorage::Cpu(_) => self.indexed_moe_forward_cpu(x, ids),
        }
    }

    /// Fused gate+up indexed MoE forward pass.
    ///
    /// Computes both gate and up projections in a single kernel pass,
    /// reading the input only once. Returns (gate_output, up_output).
    ///
    /// This is an optimization for Metal that halves memory bandwidth
    /// compared to calling indexed_moe_forward twice.
    pub fn indexed_moe_gate_up(
        gate_weights: &Self,
        up_weights: &Self,
        x: &Tensor,
        ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // TODO: Debug - temporarily disabled fused kernel to test original kernels
        // #[cfg(feature = "metal")]
        // match (gate_weights.storage.as_ref(), up_weights.storage.as_ref()) {
        //     (QStorage::Metal(gate_s), QStorage::Metal(up_s)) => {
        //         match (&*x.storage(), &*ids.storage()) {
        //             (Storage::Metal(x_storage), Storage::Metal(ids_storage)) => {
        //                 let ((gate_storage, gate_shape), (up_storage, up_shape)) =
        //                     metal::QMetalStorage::indexed_moe_gate_up(
        //                         gate_s,
        //                         up_s,
        //                         gate_weights.shape(),
        //                         x_storage,
        //                         x.layout(),
        //                         ids_storage,
        //                         ids.layout(),
        //                     )?;
        //                 let gate_out = crate::tensor::from_storage(
        //                     Storage::Metal(gate_storage),
        //                     gate_shape,
        //                 );
        //                 let up_out = crate::tensor::from_storage(
        //                     Storage::Metal(up_storage),
        //                     up_shape,
        //                 );
        //                 return Ok((gate_out, up_out));
        //             }
        //             _ => {}
        //         }
        //     }
        //     _ => {}
        // }

        // Fallback: call indexed_moe_forward twice
        let gate_out = gate_weights.indexed_moe_forward(x, ids)?;
        let up_out = up_weights.indexed_moe_forward(x, ids)?;
        Ok((gate_out, up_out))
    }

    /// Fused gate+up+SwiGLU: computes silu(gate @ x) * (up @ x) in a single GPU dispatch.
    /// Falls back to indexed_moe_gate_up + fused_swiglu if the backend doesn't support fusion.
    pub fn indexed_moe_gate_up_swiglu(
        gate_weights: &Self,
        up_weights: &Self,
        x: &Tensor,
        ids: &Tensor,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            use crate::Storage;

            if let (QStorage::Cuda(gate_s), QStorage::Cuda(up_s)) =
                (gate_weights.storage.as_ref(), up_weights.storage.as_ref())
            {
                let (gate_experts, gate_rows, gate_cols) = gate_weights.shape().dims3()?;
                let (up_experts, up_rows, up_cols) = up_weights.shape().dims3()?;
                let batch_size = match x.dims() {
                    [batch, tokens, _] => batch * tokens,
                    [batch, _] => *batch,
                    _ => usize::MAX,
                };
                let (ids_batch, top_k) = ids.dims2()?;

                // Dense models are represented as one-expert MoEs. During
                // single-token decode the expert index is necessarily zero,
                // so use the non-indexed fused CUDA kernel and avoid a second
                // input quantization plus separate SwiGLU elementwise ops.
                if gate_experts == 1
                    && up_experts == 1
                    && gate_rows == up_rows
                    && gate_cols == up_cols
                    && batch_size == 1
                    && ids_batch == 1
                    && top_k == 1
                {
                    let dense_shape: crate::Shape = (gate_rows, gate_cols).into();
                    if let Storage::Cuda(x_storage) = &*x.storage() {
                        let (storage, shape) = cuda::QCudaStorage::gate_up_swiglu_fwd(
                            gate_s,
                            up_s,
                            &dense_shape,
                            &dense_shape,
                            x_storage,
                            x.layout(),
                        )?;
                        return Ok(crate::tensor::from_storage(Storage::Cuda(storage), shape));
                    }
                }
            }
        }

        #[cfg(feature = "vulkan")]
        {
            use crate::Storage;
            match (gate_weights.storage.as_ref(), up_weights.storage.as_ref()) {
                (QStorage::Vulkan(gate_s), QStorage::Vulkan(up_s)) => {
                    if !crate::vulkan_backend::debug::force_cpu_moe(gate_s.dtype()) {
                        match (&*x.storage(), &*ids.storage()) {
                            (Storage::Vulkan(x_storage), Storage::Vulkan(ids_storage)) => {
                                let (storage, shape) = gate_s.indexed_moe_gate_up_swiglu(
                                    gate_weights.shape(),
                                    up_s,
                                    up_weights.shape(),
                                    x_storage,
                                    x.layout(),
                                    ids_storage,
                                    ids.layout(),
                                )?;
                                return Ok(crate::tensor::from_storage(
                                    Storage::Vulkan(storage),
                                    shape,
                                ));
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        // Fallback: separate gate+up then SwiGLU via basic tensor ops
        let (gate_out, up_out) = Self::indexed_moe_gate_up(gate_weights, up_weights, x, ids)?;
        gate_out.silu()?.mul(&up_out)
    }

    /// Fused gate+up SwiGLU for non-MoE FFN: computes silu(gate @ x) * (up @ x).
    /// Uses a single kernel for CUDA batch_size=1 decode. Falls back to separate ops otherwise.
    pub fn gate_up_swiglu(gate_weights: &Self, up_weights: &Self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            use crate::Storage;
            if let (QStorage::Cuda(gate_s), QStorage::Cuda(up_s)) =
                (gate_weights.storage.as_ref(), up_weights.storage.as_ref())
            {
                // Only fuse for batch_size=1 (single-token decode)
                let b_size = match x.layout().shape().dims() {
                    [b, m, _k] => b * m,
                    [b, _k] => *b,
                    _ => usize::MAX,
                };
                if b_size == 1 {
                    if let Storage::Cuda(x_storage) = &*x.storage() {
                        let (storage, shape) = cuda::QCudaStorage::gate_up_swiglu_fwd(
                            gate_s,
                            up_s,
                            gate_weights.shape(),
                            up_weights.shape(),
                            x_storage,
                            x.layout(),
                        )?;
                        return Ok(crate::tensor::from_storage(Storage::Cuda(storage), shape));
                    }
                }
            }
        }

        // Fallback: separate gate, up, then SwiGLU
        let gate_out = x.apply_op1_no_bwd(gate_weights)?;
        let up_out = x.apply_op1_no_bwd(up_weights)?;
        gate_out.silu()?.mul(&up_out)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        match self.storage.as_ref() {
            QStorage::Cuda(storage) => storage.device_ptr(),
            QStorage::Metal(_) | QStorage::Cpu(_) | QStorage::Vulkan(_) => {
                crate::bail!("not implemented");
            }
        }
    }

    /// CPU implementation of indexed MoE forward using quantized matmul
    ///
    /// This follows llama.cpp's approach:
    /// 1. Group rows by expert
    /// 2. Batch all rows for each expert into a single matmul
    /// 3. Use quantized matmul_t which handles Q8K quantization internally
    ///
    /// - weights: [num_experts, out_dim, in_dim]
    /// - x: [batch, topk_or_1, in_dim]
    /// - ids: [batch, topk]
    /// - output: [batch, topk, out_dim]
    fn indexed_moe_forward_cpu(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        use rayon::prelude::*;

        let weight_dims = self.shape().dims();
        if weight_dims.len() != 3 {
            crate::bail!("indexed_moe_forward expects 3D weight tensor [num_experts, out_dim, in_dim], got {:?}", weight_dims);
        }
        let num_experts = weight_dims[0];
        let out_dim = weight_dims[1];
        let in_dim = weight_dims[2];

        let x_dims = x.dims();
        let ids_dims = ids.dims();

        // x can be [batch, in_dim] or [batch, topk_or_1, in_dim]
        let (batch_size, x_inner_dim, in_dim_x) = if x_dims.len() == 2 {
            (x_dims[0], 1usize, x_dims[1])
        } else if x_dims.len() == 3 {
            (x_dims[0], x_dims[1], x_dims[2])
        } else {
            crate::bail!(
                "indexed_moe_forward expects 2D or 3D input tensor, got {:?}",
                x_dims
            );
        };

        if in_dim_x != in_dim {
            crate::bail!(
                "Input dimension mismatch: weight has {}, input has {}",
                in_dim,
                in_dim_x
            );
        }

        let topk = if ids_dims.len() == 2 {
            ids_dims[1]
        } else {
            crate::bail!(
                "indexed_moe_forward expects 2D ids tensor [batch, topk], got {:?}",
                ids_dims
            );
        };

        // Move tensors to CPU if needed
        let x_cpu = x.to_device(&Device::Cpu)?;
        let ids_cpu = ids.to_device(&Device::Cpu)?;

        // Convert to f32 for computation
        let x_f32 = x_cpu.to_dtype(DType::F32)?;
        let x_data: Vec<f32> = x_f32.flatten_all()?.to_vec1()?;
        let ids_data: Vec<u32> = ids_cpu.flatten_all()?.to_vec1()?;

        // Create output buffer
        let total_outputs = batch_size * topk;
        let mut output_data = vec![0.0f32; total_outputs * out_dim];

        // Group by expert for batched processing
        // Structure: expert_id -> [(output_idx, input_row_offset)]
        let mut expert_groups: std::collections::HashMap<usize, Vec<(usize, usize)>> =
            std::collections::HashMap::new();

        for batch_idx in 0..batch_size {
            for k in 0..topk {
                let expert_id = ids_data[batch_idx * topk + k] as usize;
                if expert_id < num_experts {
                    let x_idx = if x_inner_dim > 1 {
                        k.min(x_inner_dim - 1)
                    } else {
                        0
                    };
                    let x_offset = batch_idx * x_inner_dim * in_dim + x_idx * in_dim;
                    let output_idx = batch_idx * topk + k;
                    expert_groups
                        .entry(expert_id)
                        .or_default()
                        .push((output_idx, x_offset));
                }
            }
        }

        // Process each expert's assignments in parallel
        // Each expert processes all its assigned rows in a batched matmul
        #[allow(clippy::type_complexity)]
        let expert_results: Vec<(usize, Vec<(usize, Vec<f32>)>)> = expert_groups
            .into_par_iter()
            .filter_map(|(expert_id, assignments)| {
                // Get expert weights slice
                let expert_weights = self.slice_first_dim(expert_id).ok()?;

                // Get CPU-side QuantizedType for matmul.
                // For Vulkan storage, use cached cpu_data (no GPU-CPU sync needed).
                let vk_qt: Option<Box<dyn QuantizedType>> = match expert_weights.storage.as_ref() {
                    QStorage::Vulkan(vk) => vk.cpu_quantized_type().ok(),
                    _ => None,
                };
                let storage: &dyn QuantizedType = match expert_weights.storage.as_ref() {
                    QStorage::Cpu(s) => s.as_ref(),
                    QStorage::Vulkan(_) => match vk_qt.as_deref() {
                        Some(qt) => qt,
                        None => return None,
                    },
                    _ => return None,
                };

                {
                    // Batch all input rows for this expert
                    let num_rows = assignments.len();
                    let mut batched_input = vec![0.0f32; num_rows * in_dim];

                    for (i, (_, x_offset)) in assignments.iter().enumerate() {
                        batched_input[i * in_dim..(i + 1) * in_dim]
                            .copy_from_slice(&x_data[*x_offset..*x_offset + in_dim]);
                    }

                    // Batched matmul: (num_rows, in_dim, out_dim)
                    let mut batched_output = vec![0.0f32; num_rows * out_dim];
                    if storage
                        .matmul_t(
                            (num_rows, in_dim, out_dim),
                            &batched_input,
                            &mut batched_output,
                        )
                        .is_ok()
                    {
                        // Map outputs back to their positions
                        let results: Vec<(usize, Vec<f32>)> = assignments
                            .iter()
                            .enumerate()
                            .map(|(i, (output_idx, _))| {
                                let row = batched_output[i * out_dim..(i + 1) * out_dim].to_vec();
                                (*output_idx, row)
                            })
                            .collect();
                        return Some((expert_id, results));
                    }
                }
                None
            })
            .collect();

        // Scatter results back to output buffer
        for (_expert_id, results) in expert_results {
            for (output_idx, row) in results {
                let offset = output_idx * out_dim;
                output_data[offset..offset + out_dim].copy_from_slice(&row);
            }
        }

        // Create output tensor
        let output = Tensor::from_vec(output_data, (batch_size, topk, out_dim), &Device::Cpu)?;

        // Move to original device and dtype
        let output = output.to_dtype(x.dtype())?.to_device(x.device())?;

        Ok(output)
    }
}

/// Shared QTensor for mutable access during training (e.g., QuZO).
///
/// Wraps a QTensor with generation-counted caching: F32/F16/BF16 weights
/// are lazily dequantized and cached until `replace()` bumps the generation.
/// Quantized types bypass the cache and use quantized kernels directly.
#[derive(Clone)]
pub struct SharedQTensor {
    inner: std::sync::Arc<SharedQTensorInner>,
}

struct SharedQTensorInner {
    qtensor: std::sync::RwLock<QTensor>,
    generation: std::sync::atomic::AtomicU64,
    cache: std::sync::Mutex<Option<(crate::Tensor, u64)>>,
}

impl SharedQTensor {
    /// Wrap a QTensor in shared storage with generation=0 and empty cache.
    pub fn new(qtensor: QTensor) -> Self {
        Self {
            inner: std::sync::Arc::new(SharedQTensorInner {
                qtensor: std::sync::RwLock::new(qtensor),
                generation: std::sync::atomic::AtomicU64::new(0),
                cache: std::sync::Mutex::new(None),
            }),
        }
    }

    /// Read access to the underlying QTensor.
    pub fn read(&self) -> std::sync::LockResult<std::sync::RwLockReadGuard<'_, QTensor>> {
        self.inner.qtensor.read()
    }

    /// Replace the underlying QTensor and bump the generation counter,
    /// invalidating any cached dequantization.
    pub fn replace(&self, qtensor: QTensor) {
        let mut guard = self.inner.qtensor.write().unwrap();
        *guard = qtensor;
        self.inner
            .generation
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    /// Current generation counter.
    pub fn generation(&self) -> u64 {
        self.inner
            .generation
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// For F32/F16/BF16 dtypes, returns a cached dequantized tensor.
    /// For quantized types, returns None (use quantized kernels instead).
    ///
    /// The cache is invalidated when `replace()` bumps the generation.
    pub fn cached_dequantize(&self) -> Result<Option<crate::Tensor>> {
        let gen = self
            .inner
            .generation
            .load(std::sync::atomic::Ordering::Acquire);

        // Check cache
        {
            let cache = self.inner.cache.lock().unwrap();
            if let Some((ref tensor, cached_gen)) = *cache {
                if cached_gen == gen {
                    return Ok(Some(tensor.clone()));
                }
            }
        }

        // Cache miss — read-lock the QTensor
        let qt = self
            .inner
            .qtensor
            .read()
            .map_err(|e| crate::Error::Msg(format!("SharedQTensor read lock failed: {e}")))?;

        match qt.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {
                let tensor = qt.dequantize(&qt.device())?;
                drop(qt);
                let mut cache = self.inner.cache.lock().unwrap();
                *cache = Some((tensor.clone(), gen));
                Ok(Some(tensor))
            }
            _ => Ok(None),
        }
    }

    /// Shape of the underlying QTensor.
    pub fn shape(&self) -> Shape {
        self.inner.qtensor.read().unwrap().shape().clone()
    }

    pub fn rank(&self) -> usize {
        self.inner.qtensor.read().unwrap().rank()
    }

    pub fn dims(&self) -> Vec<usize> {
        self.inner.qtensor.read().unwrap().shape().dims().to_vec()
    }

    /// Dtype of the underlying QTensor.
    pub fn dtype(&self) -> GgmlDType {
        self.inner.qtensor.read().unwrap().dtype()
    }

    /// Device of the underlying QTensor.
    pub fn device(&self) -> crate::Device {
        self.inner.qtensor.read().unwrap().device()
    }

    /// Whether this tensor supports QuZO optimization.
    pub fn supports_quzo(&self) -> bool {
        self.inner
            .qtensor
            .read()
            .map(|qt| qt.supports_quzo())
            .unwrap_or(false)
    }

    /// Fused tensor ID for QuZO fused path.
    #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
    pub fn fused_tensor_id(&self) -> u64 {
        self.inner.qtensor.read().unwrap().fused_tensor_id()
    }

    // ========================================================================
    // View ops — zero-copy, stay quantized, delegate to QTensor
    // ========================================================================

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let qt = self.inner.qtensor.read().unwrap();
        Ok(Self::new(qt.narrow(dim, start, len)?))
    }

    pub fn squeeze(&self, dim: usize) -> Result<Self> {
        let qt = self.inner.qtensor.read().unwrap();
        Ok(Self::new(qt.squeeze(dim)?))
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let qt = self.inner.qtensor.read().unwrap();
        Ok(Self::new(qt.unsqueeze(dim)?))
    }

    pub fn transpose(&self, d1: usize, d2: usize) -> Result<Self> {
        let qt = self.inner.qtensor.read().unwrap();
        Ok(Self::new(qt.transpose(d1, d2)?))
    }

    pub fn reshape<S: crate::shape::ShapeWithOneHole>(&self, s: S) -> Result<Self> {
        let qt = self.inner.qtensor.read().unwrap();
        Ok(Self::new(qt.reshape(s)?))
    }

    pub fn expand(&self, shape: &Shape) -> Result<Self> {
        let qt = self.inner.qtensor.read().unwrap();
        Ok(Self::new(qt.expand(shape)?))
    }

    pub fn flatten(&self, start_dim: usize, end_dim: usize) -> Result<Self> {
        let qt = self.inner.qtensor.read().unwrap();
        Ok(Self::new(qt.flatten(start_dim, end_dim)?))
    }

    // ========================================================================
    // Compute ops — dequantize internally, return new SharedQTensor
    // ========================================================================

    /// Dequantize the underlying QTensor to a core::Tensor.
    pub fn dequant(&self) -> Result<crate::Tensor> {
        let qt = self
            .inner
            .qtensor
            .read()
            .map_err(|e| crate::Error::Msg(format!("SharedQTensor read lock failed: {e}")))?;
        qt.dequantize(&qt.device())
    }

    /// Wrap a core::Tensor into a SharedQTensor (stored as F32).
    pub fn from_core_tensor(t: crate::Tensor) -> Result<Self> {
        Ok(Self::new(QTensor::quantize(&t, GgmlDType::F32)?))
    }

    pub fn broadcast_add(&self, rhs: &Self) -> Result<Self> {
        Self::from_core_tensor(self.dequant()?.broadcast_add(&rhs.dequant()?)?)
    }

    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        Self::from_core_tensor(self.dequant()?.matmul(&rhs.dequant()?)?)
    }

    pub fn mean_keepdim(&self, dim: usize) -> Result<Self> {
        Self::from_core_tensor(self.dequant()?.mean_keepdim(dim)?)
    }

    pub fn sum_keepdim(&self, dim: usize) -> Result<Self> {
        Self::from_core_tensor(self.dequant()?.sum_keepdim(dim)?)
    }

    pub fn var_keepdim(&self, dim: usize) -> Result<Self> {
        Self::from_core_tensor(self.dequant()?.var_keepdim(dim)?)
    }

    pub fn max_keepdim(&self, dim: usize) -> Result<Self> {
        Self::from_core_tensor(self.dequant()?.max_keepdim(dim)?)
    }

    pub fn min_keepdim(&self, dim: usize) -> Result<Self> {
        Self::from_core_tensor(self.dequant()?.min_keepdim(dim)?)
    }

    pub fn argmax_keepdim(&self, dim: usize) -> Result<Self> {
        Self::from_core_tensor(self.dequant()?.argmax_keepdim(dim)?)
    }

    pub fn argmin_keepdim(&self, dim: usize) -> Result<Self> {
        Self::from_core_tensor(self.dequant()?.argmin_keepdim(dim)?)
    }

    pub fn gather(&self, indices: &Self, dim: usize) -> Result<Self> {
        // Indices must be U32/I64 — dequantize them and cast to U32
        let idx = indices.dequant()?.to_dtype(crate::DType::U32)?;
        Self::from_core_tensor(self.dequant()?.gather(&idx, dim)?)
    }

    pub fn conv2d(
        &self,
        kernel: &Self,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        Self::from_core_tensor(self.dequant()?.conv2d(
            &kernel.dequant()?,
            padding,
            stride,
            dilation,
            groups,
        )?)
    }

    // ========================================================================
    // Utility methods
    // ========================================================================

    pub fn to_vec1<T: crate::WithDType>(&self) -> Result<Vec<T>> {
        self.dequant()?.to_vec1()
    }

    pub fn to_vec2<T: crate::WithDType>(&self) -> Result<Vec<Vec<T>>> {
        self.dequant()?.to_vec2()
    }

    pub fn to_vec3<T: crate::WithDType>(&self) -> Result<Vec<Vec<Vec<T>>>> {
        self.dequant()?.to_vec3()
    }
}

impl std::fmt::Debug for SharedQTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SharedQTensor({:?})", self.shape())
    }
}

#[derive(Clone)]
pub enum QMatMul {
    /// Immutable quantized tensor (standard inference path)
    QTensor(std::sync::Arc<QTensor>),
    /// Mutable quantized tensor for training (QuZO, etc.)
    /// Takes read lock during forward, write lock during optimization
    Shared(SharedQTensor),
    /// Dequantized F32 tensor
    Tensor(Tensor),
    /// Dequantized F16 tensor
    TensorF16(Tensor),
}

impl std::fmt::Debug for QMatMul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QTensor(t) => write!(f, "QMatMul::QTensor({:?})", t.shape()),
            Self::Shared(t) => {
                write!(f, "QMatMul::Shared({:?})", t.shape())
            }
            Self::Tensor(t) => write!(f, "QMatMul::Tensor({:?})", t.shape()),
            Self::TensorF16(t) => write!(f, "QMatMul::TensorF16({:?})", t.shape()),
        }
    }
}

thread_local! {
    static DEQUANTIZE_ALL: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

thread_local! {
    static DEQUANTIZE_ALL_F16: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL_F16") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

impl QMatMul {
    pub fn from_arc(qtensor: std::sync::Arc<QTensor>) -> Result<Self> {
        let dequantize = match qtensor.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => true,
            _ => DEQUANTIZE_ALL.with(|b| *b),
        };
        let t = if dequantize {
            let tensor = qtensor.dequantize(&qtensor.device())?;
            Self::Tensor(tensor)
        } else if DEQUANTIZE_ALL_F16.with(|b| *b) {
            let tensor = qtensor.dequantize_f16(&qtensor.device())?;
            Self::TensorF16(tensor)
        } else {
            Self::QTensor(qtensor)
        };
        Ok(t)
    }

    pub fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        Self::from_arc(std::sync::Arc::new(qtensor))
    }

    /// Create a QMatMul with shared storage for training.
    ///
    /// Always returns `Shared` — F32/F16/BF16 dequantization is handled lazily
    /// via `SharedQTensor::cached_dequantize()` during forward pass.
    pub fn from_shared(qtensor: SharedQTensor) -> Result<Self> {
        Ok(Self::Shared(qtensor))
    }

    /// Create a QMatMul with mutable/shared storage from an owned QTensor.
    pub fn into_shared(qtensor: QTensor) -> Self {
        Self::Shared(SharedQTensor::new(qtensor))
    }

    /// Get the SharedQTensor if this is a Shared variant.
    /// Returns None for other variants.
    pub fn shared_qtensor(&self) -> Option<SharedQTensor> {
        match self {
            Self::Shared(t) => Some(t.clone()),
            _ => None,
        }
    }

    /// Convert a QTensor variant to Shared variant in-place, enabling training.
    ///
    /// Returns the SharedQTensor if conversion succeeded or was already shared.
    /// Returns None for Tensor/TensorF16 variants (not quantized).
    /// The Arc<QTensor> must have exactly one strong reference (panics otherwise).
    pub fn make_shared(&mut self) -> Option<SharedQTensor> {
        match self {
            Self::Shared(s) => return Some(s.clone()),
            Self::Tensor(_) | Self::TensorF16(_) => return None,
            Self::QTensor(_) => {}
        }
        // Take the QTensor out via swap with a dummy
        let dummy = Self::Tensor(
            crate::Tensor::zeros(&[1], crate::DType::F32, &crate::Device::Cpu).unwrap(),
        );
        let old = std::mem::replace(self, dummy);
        match old {
            Self::QTensor(arc) => {
                let qt = std::sync::Arc::try_unwrap(arc)
                    .expect("QMatMul::make_shared: Arc<QTensor> has multiple strong references");
                let shared = SharedQTensor::new(qt);
                *self = Self::Shared(shared.clone());
                Some(shared)
            }
            _ => unreachable!(),
        }
    }

    /// Check if this QMatMul supports QuZO optimization.
    pub fn supports_quzo(&self) -> bool {
        match self {
            Self::Shared(t) => t.supports_quzo(),
            Self::QTensor(t) => t.supports_quzo(),
            _ => false,
        }
    }

    pub fn dequantize_f16(&self) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.dequantize_f16(&t.device()),
            Self::Shared(t) => {
                let qt = t.read().map_err(|e| {
                    crate::Error::Msg(format!("QMatMul::Shared read lock failed: {e}"))
                })?;
                qt.dequantize_f16(&qt.device())
            }
            Self::Tensor(t) => t.to_dtype(DType::F16),
            Self::TensorF16(t) => Ok(t.clone()),
        }
    }

    pub fn forward_via_f16(&self, xs: &Tensor) -> Result<Tensor> {
        let w = self.dequantize_f16()?;
        let in_dtype = xs.dtype();
        let w = match *xs.dims() {
            [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
            _ => w.t()?,
        };
        xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.indexed_moe_forward(x, ids),
            Self::Shared(t) => {
                let qt = t.read().map_err(|e| {
                    crate::Error::Msg(format!("QMatMul::Shared read lock failed: {e}"))
                })?;
                qt.indexed_moe_forward(x, ids)
            }
            _ => {
                panic!("indexed_moe_forward not implemented for dequantized types")
            }
        }
    }

    /// Fused gate+up indexed MoE forward pass.
    ///
    /// Computes both gate and up projections in a single kernel pass.
    pub fn indexed_moe_gate_up(
        gate_weights: &Self,
        up_weights: &Self,
        x: &Tensor,
        ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        match (gate_weights, up_weights) {
            (Self::QTensor(gate_t), Self::QTensor(up_t)) => {
                QTensor::indexed_moe_gate_up(gate_t, up_t, x, ids)
            }
            (Self::Shared(gate_t), Self::Shared(up_t)) => {
                let gate_qt = gate_t
                    .read()
                    .map_err(|e| crate::Error::Msg(format!("gate read lock failed: {e}")))?;
                let up_qt = up_t
                    .read()
                    .map_err(|e| crate::Error::Msg(format!("up read lock failed: {e}")))?;
                QTensor::indexed_moe_gate_up(&gate_qt, &up_qt, x, ids)
            }
            _ => {
                panic!("indexed_moe_gate_up requires matching QTensor or Shared types")
            }
        }
    }

    /// Fused gate+up+SwiGLU: computes silu(gate @ x) * (up @ x) in a single GPU dispatch.
    pub fn indexed_moe_gate_up_swiglu(
        gate_weights: &Self,
        up_weights: &Self,
        x: &Tensor,
        ids: &Tensor,
    ) -> Result<Tensor> {
        match (gate_weights, up_weights) {
            (Self::QTensor(gate_t), Self::QTensor(up_t)) => {
                QTensor::indexed_moe_gate_up_swiglu(gate_t, up_t, x, ids)
            }
            (Self::Shared(gate_t), Self::Shared(up_t)) => {
                let gate_qt = gate_t
                    .read()
                    .map_err(|e| crate::Error::Msg(format!("gate read lock failed: {e}")))?;
                let up_qt = up_t
                    .read()
                    .map_err(|e| crate::Error::Msg(format!("up read lock failed: {e}")))?;
                QTensor::indexed_moe_gate_up_swiglu(&gate_qt, &up_qt, x, ids)
            }
            _ => {
                panic!("indexed_moe_gate_up_swiglu requires matching QTensor or Shared types")
            }
        }
    }

    /// Fused gate+up SwiGLU for non-MoE FFN: computes silu(gate @ x) * (up @ x).
    /// Uses a single CUDA kernel for batch_size=1. Falls back to separate ops otherwise.
    pub fn gate_up_swiglu(gate: &Self, up: &Self, x: &Tensor) -> Result<Tensor> {
        match (gate, up) {
            (Self::QTensor(gate_t), Self::QTensor(up_t)) => {
                QTensor::gate_up_swiglu(gate_t, up_t, x)
            }
            (Self::Shared(gate_t), Self::Shared(up_t)) => {
                let gate_qt = gate_t
                    .read()
                    .map_err(|e| crate::Error::Msg(format!("gate read lock failed: {e}")))?;
                let up_qt = up_t
                    .read()
                    .map_err(|e| crate::Error::Msg(format!("up read lock failed: {e}")))?;
                QTensor::gate_up_swiglu(&gate_qt, &up_qt, x)
            }
            (Self::Tensor(gate_w), Self::Tensor(up_w))
            | (Self::TensorF16(gate_w), Self::TensorF16(up_w)) => {
                let in_dtype = x.dtype();
                let gate_w = match *x.dims() {
                    [b1, b2, _, _] => gate_w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => gate_w.broadcast_left(bsize)?.t()?,
                    _ => gate_w.t()?,
                };
                let up_w = match *x.dims() {
                    [b1, b2, _, _] => up_w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => up_w.broadcast_left(bsize)?.t()?,
                    _ => up_w.t()?,
                };
                let x_cast = if gate_w.dtype() != in_dtype {
                    x.to_dtype(gate_w.dtype())?
                } else {
                    x.clone()
                };
                let gate_out = x_cast.matmul(&gate_w)?;
                let up_out = x_cast.matmul(&up_w)?;
                gate_out.silu()?.mul(&up_out)?.to_dtype(in_dtype)
            }
            _ => {
                // Mixed types: fall back to separate forward calls
                let gate_out = crate::Module::forward(gate, x)?;
                let up_out = crate::Module::forward(up, x)?;
                gate_out.silu()?.mul(&up_out)
            }
        }
    }

    /// Fused forward pass with on-the-fly perturbation (GPU only).
    ///
    /// Computes: y = (W + ε*z) @ x where z is generated on-the-fly from seed.
    /// This is used for QuZO training when we want to avoid storing perturbed weights.
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `seed` - Random seed for perturbation generation
    /// * `epsilon` - Perturbation magnitude
    #[cfg(feature = "cuda")]
    pub fn fused_forward(&self, x: &Tensor, seed: u64, epsilon: f32) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.fused_fwd(x, seed, epsilon),
            Self::Shared(t) => {
                let qt = t.read().map_err(|e| {
                    crate::Error::Msg(format!("QMatMul::Shared read lock failed: {e}"))
                })?;
                qt.fused_fwd(x, seed, epsilon)
            }
            Self::Tensor(_) | Self::TensorF16(_) => {
                crate::bail!("fused_forward only supported for quantized weights (QTensor/Shared)")
            }
        }
    }
}

impl crate::CustomOp1 for QTensor {
    fn name(&self) -> &'static str {
        "qmatmul"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        let (n, k) = self.shape().dims2()?;
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!(
                "input tensor {layout:?} incompatible with {:?}",
                self.shape()
            )
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        #[allow(clippy::infallible_destructuring_match)]
        let self_storage = match self.storage.as_ref() {
            QStorage::Cpu(storage) => storage,
            QStorage::Metal(_) | QStorage::Cuda(_) | QStorage::Vulkan(_) => {
                crate::bail!("Invalid storage")
            }
        };
        match storage.dtype() {
            DType::F32 => {
                let slice = storage.as_slice::<f32>()?;
                let slice =
                    &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                let mut dst_storage = vec![0f32; dst_shape.elem_count()];
                self_storage.matmul_t(
                    (dst_shape.elem_count() / n, k, n),
                    slice,
                    &mut dst_storage,
                )?;
                Ok((crate::CpuStorage::F32(dst_storage), dst_shape))
            }
            DType::F16 => {
                let slice = storage.as_slice::<f16>()?;
                let slice =
                    &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                let mut dst_storage = vec![f16::ZERO; dst_shape.elem_count()];
                self_storage.matmul_t_f16(
                    (dst_shape.elem_count() / n, k, n),
                    slice,
                    &mut dst_storage,
                )?;
                Ok((crate::CpuStorage::F16(dst_storage), dst_shape))
            }
            _ => crate::bail!("Expected f32/f16"),
        }
    }

    fn vulkan_fwd(
        &self,
        storage: &crate::VulkanStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::VulkanStorage, Shape)> {
        let self_storage = match self.storage.as_ref() {
            QStorage::Vulkan(vk) => vk,
            _ => unreachable!("Cannot call vulkan matmul on non vulkan QTensor"),
        };
        self_storage.fwd(self.shape(), storage, layout)
    }

    fn metal_fwd(
        &self,
        storage: &crate::MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, Shape)> {
        let self_storage = match self.storage.as_ref() {
            QStorage::Metal(metal) => metal,
            _ => unreachable!("Cannot call metal matmul on non metal QTensor"),
        };
        self_storage.fwd(self.shape(), storage, layout)
    }

    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        let self_storage = match self.storage.as_ref() {
            QStorage::Cuda(cuda) => cuda,
            _ => unreachable!("Cannot call cuda matmul on non cuda QTensor"),
        };
        self_storage.fwd(self.shape(), storage, layout)
    }
}

impl crate::Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Check for fused perturbation state (QuZO training)
        if let Some(_base_perturb) = get_perturbation_state() {
            match self {
                Self::QTensor(t) => {
                    #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
                    let perturb = get_perturbation_state_for_tensor(t.fused_tensor_id());
                    #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
                    let perturb = Some(_base_perturb);
                    if let Some(perturb) = perturb {
                        // CUDA fused path
                        #[cfg(feature = "cuda")]
                        if t.device().is_cuda()
                            && matches!(
                                t.dtype(),
                                GgmlDType::Q8_0
                                    | GgmlDType::Q4K
                                    | GgmlDType::Q2K
                                    | GgmlDType::Q3K
                                    | GgmlDType::Q5K
                                    | GgmlDType::Q6K
                                    | GgmlDType::BF16
                            )
                        {
                            return t.fused_fwd(xs, perturb.seed, perturb.epsilon);
                        }
                        // Metal fused path
                        #[cfg(feature = "metal")]
                        if t.device().is_metal()
                            && matches!(
                                t.dtype(),
                                GgmlDType::Q8_0
                                    | GgmlDType::Q4K
                                    | GgmlDType::Q2K
                                    | GgmlDType::Q3K
                                    | GgmlDType::Q5K
                                    | GgmlDType::Q6K
                                    | GgmlDType::BF16
                            )
                        {
                            return t.fused_fwd(xs, perturb.seed, perturb.epsilon);
                        }
                        // Vulkan fused path
                        #[cfg(feature = "vulkan")]
                        if t.device().is_vulkan()
                            && matches!(
                                t.dtype(),
                                GgmlDType::Q8_0
                                    | GgmlDType::Q4K
                                    | GgmlDType::Q2K
                                    | GgmlDType::Q3K
                                    | GgmlDType::Q5K
                                    | GgmlDType::Q6K
                                    | GgmlDType::BF16
                            )
                        {
                            return t.fused_fwd(xs, perturb.seed, perturb.epsilon);
                        }
                        // CPU fused path - dequantize with perturbation then matmul
                        if t.device().is_cpu() {
                            return t.fused_cpu_forward(xs, perturb.seed, perturb.epsilon);
                        }
                    }
                }
                Self::Shared(t) => {
                    let qt = t.read().map_err(|e| {
                        crate::Error::Msg(format!("QMatMul::Shared read lock failed: {e}"))
                    })?;
                    #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
                    let perturb = get_perturbation_state_for_tensor(qt.fused_tensor_id());
                    #[cfg(not(any(feature = "cuda", feature = "metal", feature = "vulkan")))]
                    let perturb = Some(_base_perturb);
                    if let Some(perturb) = perturb {
                        // CUDA fused path
                        #[cfg(feature = "cuda")]
                        if qt.device().is_cuda()
                            && matches!(
                                qt.dtype(),
                                GgmlDType::Q8_0
                                    | GgmlDType::Q4K
                                    | GgmlDType::Q2K
                                    | GgmlDType::Q3K
                                    | GgmlDType::Q5K
                                    | GgmlDType::Q6K
                                    | GgmlDType::BF16
                            )
                        {
                            return qt.fused_fwd(xs, perturb.seed, perturb.epsilon);
                        }
                        // Metal fused path
                        #[cfg(feature = "metal")]
                        if qt.device().is_metal()
                            && matches!(
                                qt.dtype(),
                                GgmlDType::Q8_0
                                    | GgmlDType::Q4K
                                    | GgmlDType::Q2K
                                    | GgmlDType::Q3K
                                    | GgmlDType::Q5K
                                    | GgmlDType::Q6K
                                    | GgmlDType::BF16
                            )
                        {
                            return qt.fused_fwd(xs, perturb.seed, perturb.epsilon);
                        }
                        // Vulkan fused path
                        #[cfg(feature = "vulkan")]
                        if qt.device().is_vulkan()
                            && matches!(
                                qt.dtype(),
                                GgmlDType::Q8_0
                                    | GgmlDType::Q4K
                                    | GgmlDType::Q2K
                                    | GgmlDType::Q3K
                                    | GgmlDType::Q5K
                                    | GgmlDType::Q6K
                                    | GgmlDType::BF16
                            )
                        {
                            return qt.fused_fwd(xs, perturb.seed, perturb.epsilon);
                        }
                        // CPU fused path
                        if qt.device().is_cpu() {
                            return qt.fused_cpu_forward(xs, perturb.seed, perturb.epsilon);
                        }
                    }
                }
                _ => {}
            }
        }

        // Regular forward (no perturbation or unsupported dtype/device)
        match self {
            Self::QTensor(t) => xs.apply_op1_no_bwd(t.as_ref()),
            Self::Shared(t) => {
                if let Some(w) = t.cached_dequantize()? {
                    // F32/F16/BF16 — cached dequantized tensor
                    let in_dtype = xs.dtype();
                    let w = match *xs.dims() {
                        [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                        [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                        _ => w.t()?,
                    };
                    let w = if w.dtype() != in_dtype {
                        w.to_dtype(in_dtype)?
                    } else {
                        w
                    };
                    xs.matmul(&w)
                } else {
                    // Quantized — use quantized kernel
                    let qt = t.read().map_err(|e| {
                        crate::Error::Msg(format!("QMatMul::Shared read lock failed: {e}"))
                    })?;
                    xs.apply_op1_no_bwd(&*qt)
                }
            }
            Self::Tensor(w) => {
                let in_dtype = xs.dtype();
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                // Convert weight to input dtype to avoid matmul mismatch
                let w = if w.dtype() != in_dtype {
                    w.to_dtype(in_dtype)?
                } else {
                    w
                };
                xs.matmul(&w)
            }
            Self::TensorF16(w) => {
                let in_dtype = xs.dtype();
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
            }
        }
    }
}
