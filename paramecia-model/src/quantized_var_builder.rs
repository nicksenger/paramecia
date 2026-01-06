//! Varbuilder for Loading gguf files
//!
//! VarBuilder is a utility to store quantized tensors from a [GGUF model file](https://huggingface.co/docs/hub/gguf).
//! These tensors can be loaded from disk using `from_gguf` or from an in-memory
//! buffer using `from_gguf_buffer`.
//!
//! ## Mutable Access for Training
//!
//! For training scenarios like QuZO (Quantized Zeroth-Order optimization), use
//! `MutableVarBuilder` which wraps tensors in `Arc<RwLock<QTensor>>` for both
//! read (forward pass) and write (optimization) access.

use paramecia_core::quantized::QTensor;
use paramecia_core::{Device, Result, Shape};
use std::sync::Arc;

// Re-export SharedQTensor from candle-core for convenience
pub use paramecia_core::quantized::SharedQTensor;

// VarBuilder specialized for QTensors (immutable after loading)
#[derive(Clone)]
pub struct VarBuilder {
    data: Arc<std::collections::HashMap<String, Arc<QTensor>>>,
    path: Vec<String>,
    device: Device,
}

impl VarBuilder {
    pub fn from_gguf<P: AsRef<std::path::Path>>(p: P, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(p)?;
        let content = paramecia_core::quantized::gguf_file::Content::read(&mut file)?;
        let mut data = std::collections::HashMap::new();
        for tensor_name in content.tensor_infos.keys() {
            let tensor = content.tensor(&mut file, tensor_name, device)?;
            data.insert(tensor_name.to_string(), Arc::new(tensor));
        }
        Ok(Self {
            data: Arc::new(data),
            path: Vec::new(),
            device: device.clone(),
        })
    }

    pub fn from_gguf_buffer(buffer: &[u8], device: &Device) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(buffer);
        let content = paramecia_core::quantized::gguf_file::Content::read(&mut cursor)?;
        let mut data = std::collections::HashMap::new();
        for tensor_name in content.tensor_infos.keys() {
            let tensor = content.tensor(&mut cursor, tensor_name, device)?;
            data.insert(tensor_name.to_string(), Arc::new(tensor));
        }
        Ok(Self {
            data: Arc::new(data),
            path: Vec::new(),
            device: device.clone(),
        })
    }

    pub fn pp<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            data: self.data.clone(),
            path,
            device: self.device.clone(),
        }
    }

    fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    pub fn get<S: Into<Shape>>(&self, s: S, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);
        match self.data.get(&path) {
            None => {
                paramecia_core::bail!("cannot find tensor {path}")
            }
            Some(qtensor) => {
                let shape = s.into();
                if qtensor.shape() != &shape {
                    paramecia_core::bail!(
                        "shape mismatch for {name}, got {:?}, expected {shape:?}",
                        qtensor.shape()
                    )
                }
                Ok(qtensor.clone())
            }
        }
    }

    pub fn get_no_shape(&self, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);
        match self.data.get(&path) {
            None => {
                paramecia_core::bail!("cannot find tensor {name}")
            }
            Some(qtensor) => Ok(qtensor.clone()),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Get all tensor names stored in this VarBuilder.
    pub fn tensor_names(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }
}

/// MutableVarBuilder for training scenarios like QuZO.
///
/// Unlike VarBuilder which stores immutable Arc<QTensor>, this stores
/// Arc<RwLock<QTensor>> allowing both read and write access to the
/// quantized weights during training.
#[derive(Clone)]
pub struct MutableVarBuilder {
    data: Arc<std::collections::HashMap<String, SharedQTensor>>,
    path: Vec<String>,
    device: Device,
}

impl MutableVarBuilder {
    /// Load tensors from a GGUF file with mutable access.
    pub fn from_gguf<P: AsRef<std::path::Path>>(p: P, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(p)?;
        let content = paramecia_core::quantized::gguf_file::Content::read(&mut file)?;
        let mut data = std::collections::HashMap::new();
        for tensor_name in content.tensor_infos.keys() {
            let tensor = content.tensor(&mut file, tensor_name, device)?;
            data.insert(tensor_name.to_string(), SharedQTensor::new(tensor));
        }
        Ok(Self {
            data: Arc::new(data),
            path: Vec::new(),
            device: device.clone(),
        })
    }

    /// Load tensors from a buffer with mutable access.
    pub fn from_gguf_buffer(buffer: &[u8], device: &Device) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(buffer);
        let content = paramecia_core::quantized::gguf_file::Content::read(&mut cursor)?;
        let mut data = std::collections::HashMap::new();
        for tensor_name in content.tensor_infos.keys() {
            let tensor = content.tensor(&mut cursor, tensor_name, device)?;
            data.insert(tensor_name.to_string(), SharedQTensor::new(tensor));
        }
        Ok(Self {
            data: Arc::new(data),
            path: Vec::new(),
            device: device.clone(),
        })
    }

    pub fn pp<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            data: self.data.clone(),
            path,
            device: self.device.clone(),
        }
    }

    fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    /// Get a shared (mutable) reference to a QTensor.
    pub fn get<S: Into<Shape>>(&self, s: S, name: &str) -> Result<SharedQTensor> {
        let path = self.path(name);
        match self.data.get(&path) {
            None => {
                paramecia_core::bail!("cannot find tensor {path}")
            }
            Some(qtensor) => {
                let shape = s.into();
                if qtensor.shape() != shape {
                    paramecia_core::bail!(
                        "shape mismatch for {name}, got {:?}, expected {shape:?}",
                        qtensor.shape()
                    )
                }
                Ok(qtensor.clone())
            }
        }
    }

    /// Get a shared reference without shape validation.
    pub fn get_no_shape(&self, name: &str) -> Result<SharedQTensor> {
        let path = self.path(name);
        match self.data.get(&path) {
            None => {
                paramecia_core::bail!("cannot find tensor {name}")
            }
            Some(qtensor) => Ok(qtensor.clone()),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Get all tensor names stored in this builder.
    pub fn tensor_names(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }

    /// Get all SharedQTensors that support QuZO optimization.
    /// Returns (name, SharedQTensor) pairs for tensors with supported quant types.
    pub fn quzo_tensors(&self) -> Vec<(String, SharedQTensor)> {
        self.data
            .iter()
            .filter(|(_, qt)| qt.supports_quzo())
            .map(|(name, qt)| (name.clone(), qt.clone()))
            .collect()
    }

    /// Get all SharedQTensors (regardless of QuZO support).
    pub fn all_tensors(&self) -> Vec<(String, SharedQTensor)> {
        self.data
            .iter()
            .map(|(name, qt)| (name.clone(), qt.clone()))
            .collect()
    }
}
