use crate::quantized_nn::RmsNorm;
use paramecia_core::quantized::{gguf_file, QTensor, SharedQTensor};
use paramecia_core::{DType, Device, Result};
use paramecia_tensor::glowstick::Shape;
use std::io::{Read, Seek};
use tracing::{debug, info};

/// Unified GGUF loader — creates SharedQTensor (mutable) for all QMatMul weights.
/// Used for both inference and training paths. The shared wrapper has negligible
/// overhead for inference (single-reader, no contention).
pub(super) struct Gguf<R: Read + Seek> {
    pub(super) ct: gguf_file::Content,
    pub(super) reader: R,
    pub(super) device: Device,
    /// Device for expert weights (may be CPU for memory efficiency)
    pub(super) expert_device: Device,
    /// All SharedQTensors created during loading (for optimizer access)
    pub(super) shared_tensors: Vec<(String, SharedQTensor)>,
}

impl<R: Read + Seek> Gguf<R> {
    fn log_loaded_tensor(&self, kind: &'static str, name: &str, qt: &QTensor) {
        if std::env::var("PARAMECIA_LOG_GGUF_SHAPES").is_ok() {
            info!(
                kind,
                tensor = %name,
                shape = ?qt.shape(),
                dtype = ?qt.dtype(),
                "GGUF tensor load"
            );
        } else {
            debug!(
                kind,
                tensor = %name,
                shape = ?qt.shape(),
                dtype = ?qt.dtype(),
                "GGUF tensor load"
            );
        }
    }

    pub(super) fn new(
        ct: gguf_file::Content,
        reader: R,
        device: Device,
        expert_device: Device,
    ) -> Self {
        Self {
            ct,
            reader,
            device,
            expert_device,
            shared_tensors: Vec::new(),
        }
    }

    /// Load standard RmsNorm (GGUF weights already have Gemma +1 adjustment baked in)
    pub(super) fn rms_norm(&mut self, name: &str, eps: f64, dtype: DType) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        self.log_loaded_tensor("rms_norm", name, &ws);
        RmsNorm::from_qtensor(ws, eps)?.to_dtype(dtype)
    }

    pub(super) fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    pub(super) fn tensor(&mut self, name: &str) -> Result<QTensor> {
        let qt = self.ct.tensor(&mut self.reader, name, &self.device)?;
        self.log_loaded_tensor("tensor", name, &qt);
        Ok(qt)
    }

    /// Load a tensor to the expert device (may be CPU)
    #[allow(dead_code)]
    pub(super) fn expert_tensor(&mut self, name: &str) -> Result<QTensor> {
        let qt = self
            .ct
            .tensor(&mut self.reader, name, &self.expert_device)?;
        self.log_loaded_tensor("expert_tensor", name, &qt);
        Ok(qt)
    }

    /// Load a tensor as SharedQTensor (mutable) for training
    pub(super) fn shared_tensor(&mut self, name: &str) -> Result<SharedQTensor> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        self.log_loaded_tensor("shared_tensor", name, &ws);
        let shared: SharedQTensor = SharedQTensor::new(ws);
        self.shared_tensors.push((name.to_string(), shared.clone()));
        Ok(shared)
    }

    /// Load an expert tensor as SharedQTensor (mutable) for training
    pub(super) fn shared_expert_tensor(&mut self, name: &str) -> Result<SharedQTensor> {
        let ws = self
            .ct
            .tensor(&mut self.reader, name, &self.expert_device)?;
        self.log_loaded_tensor("shared_expert_tensor", name, &ws);
        let shared = SharedQTensor::new(ws);
        self.shared_tensors.push((name.to_string(), shared.clone()));
        Ok(shared)
    }

    /// Load a dense expert tensor and expose it as a one-expert 3-D tensor.
    ///
    /// The view is installed on the existing shared wrapper so the optimizer
    /// and the dense forward path retain the same mutable identity. Creating a
    /// second `SharedQTensor` from `unsqueeze()` would share only the initial
    /// storage; a later optimizer `replace()` would not reach the forward view.
    pub(super) fn shared_dense_expert_tensor(&mut self, name: &str) -> Result<SharedQTensor> {
        let shared = self.shared_expert_tensor(name)?;
        shared.unsqueeze_in_place(0)?;
        Ok(shared)
    }

    /// Load RmsNorm with shared (mutable) weight for training
    pub(super) fn shared_rms_norm(
        &mut self,
        name: &str,
        eps: f64,
        dtype: DType,
    ) -> Result<RmsNorm> {
        let shared = self.shared_tensor(name)?;
        let weight = shared
            .read()
            .unwrap()
            .dequantize(&self.device)?
            .to_dtype(dtype)?;
        RmsNorm::from_shared(weight, eps, shared)
    }

    pub(super) fn try_tensor(&mut self, name: &str) -> Result<Option<QTensor>> {
        match self.ct.tensor(&mut self.reader, name, &self.device) {
            Ok(t) => {
                self.log_loaded_tensor("try_tensor", name, &t);
                Ok(Some(t))
            }
            Err(_) => {
                debug!(tensor = %name, "GGUF tensor missing (optional)");
                Ok(None)
            }
        }
    }

    /// Get all SharedQTensors that were loaded (for optimizer access)
    pub(super) fn take_shared_tensors(&mut self) -> Vec<(String, SharedQTensor)> {
        let tensors = std::mem::take(&mut self.shared_tensors);
        info!(
            count = tensors.len(),
            "Collected shared tensors for training"
        );
        if std::env::var("PARAMECIA_LOG_TENSORS").is_ok() {
            for (name, qt) in &tensors {
                let qt_read = qt.read().unwrap();
                debug!(
                    tensor = %name,
                    shape = ?qt_read.shape(),
                    dtype = ?qt_read.dtype(),
                    "Collected tensor"
                );
            }
        }
        tensors
    }

    /// Swap the loading device, returning the previous device.
    pub(super) fn set_device(&mut self, device: Device) -> Device {
        std::mem::replace(&mut self.device, device)
    }

    /// Swap the expert loading device, returning the previous device.
    pub(super) fn set_expert_device(&mut self, device: Device) -> Device {
        std::mem::replace(&mut self.expert_device, device)
    }

    /// Load a SharedQTensor with compile-time shape validation against `S`.
    ///
    /// Internally calls `shared_tensor()` (which pushes to the collection for the
    /// optimizer), then wraps the result in a typed `SharedQTensor<S>`.
    pub(super) fn shared_tensor_typed<S: Shape>(
        &mut self,
        name: &str,
    ) -> Result<paramecia_tensor::SharedQTensor<S>> {
        let shared = self.shared_tensor(name)?;
        paramecia_tensor::SharedQTensor::<S>::new(shared)
            .map_err(|e| paramecia_core::Error::Msg(format!("shared_tensor_typed {name}: {e}")))
    }

    /// Load a typed QMatMul with shape validation against `S`.
    ///
    /// Creates a SharedQTensor and validates that the weight dimensions
    /// match the compile-time shape `S`. On mismatch, the error message
    /// includes the tensor name and expected vs actual dimensions.
    pub(super) fn typed_qmatmul<S: Shape>(
        &mut self,
        name: &str,
    ) -> Result<paramecia_tensor::QMatMul<S>> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        self.log_loaded_tensor("typed_qmatmul", name, &ws);
        let shared = SharedQTensor::new(ws);
        self.shared_tensors.push((name.to_string(), shared.clone()));
        paramecia_tensor::QMatMul::<S>::from_shared(shared)
            .map_err(|e| paramecia_core::Error::Msg(format!("typed_qmatmul {name}: {e}")))
    }

    /// Like `typed_qmatmul` but returns `None` if the tensor is missing.
    pub(super) fn try_typed_qmatmul<S: Shape>(
        &mut self,
        name: &str,
    ) -> Result<Option<paramecia_tensor::QMatMul<S>>> {
        match self.try_tensor(name)? {
            Some(ws) => {
                let shared = SharedQTensor::new(ws);
                self.shared_tensors.push((name.to_string(), shared.clone()));
                let qmm = paramecia_tensor::QMatMul::<S>::from_shared(shared).map_err(|e| {
                    paramecia_core::Error::Msg(format!("try_typed_qmatmul {name}: {e}"))
                })?;
                Ok(Some(qmm))
            }
            None => Ok(None),
        }
    }
}
