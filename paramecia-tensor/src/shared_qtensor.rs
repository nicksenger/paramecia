use std::marker::PhantomData;

use glowstick::Shape;
use paramecia_core::quantized::QTensor;
use paramecia_core::{DType, Device};

use crate::{Error, Tensor};

/// Typed wrapper around `paramecia_core::quantized::SharedQTensor`.
///
/// `S` encodes the expected tensor shape at the type level (e.g. `Shape1<DtRank>`).
/// Construction validates that runtime dims match `S` (0 in type-level shape = dynamic).
///
/// The typed and untyped wrappers share the same `Arc` interior — the optimizer gets
/// an untyped clone via `inner()` for its heterogeneous collection, while struct fields
/// hold typed wrappers. When QuZO calls `replace()`, mutations propagate through the
/// shared Arc to both.
pub struct SharedQTensor<S: Shape> {
    inner: paramecia_core::quantized::SharedQTensor,
    _phantom: PhantomData<S>,
}

impl<S: Shape> std::fmt::Debug for SharedQTensor<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SharedQTensor<{}>({:?})",
            std::any::type_name::<S>(),
            self.inner.shape()
        )
    }
}

impl<S: Shape> Clone for SharedQTensor<S> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<S: Shape> SharedQTensor<S> {
    /// Wrap an untyped `SharedQTensor`, validating that its dims match `S`.
    #[track_caller]
    pub fn new(inner: paramecia_core::quantized::SharedQTensor) -> Result<Self, Error> {
        let caller = std::panic::Location::caller().to_string();
        let runtime_shape = inner.dims();
        let type_level_shape = S::iter().collect::<Vec<_>>();

        if S::RANK != runtime_shape.len() {
            return Err(Error::RankMismatch {
                context: "Typed SharedQTensor construction",
                typed_shape: std::any::type_name::<S>(),
                caller: caller.into_boxed_str(),
                runtime_rank: runtime_shape.len(),
                type_level_rank: S::RANK,
                runtime_shape: runtime_shape.into_boxed_slice(),
                type_level_shape: type_level_shape.into_boxed_slice(),
            });
        }

        for (dim, (a, b)) in runtime_shape
            .iter()
            .copied()
            .zip(type_level_shape.iter().copied())
            .enumerate()
        {
            if b != 0 && a != b {
                return Err(Error::DimensionMismatch {
                    context: "Typed SharedQTensor construction",
                    typed_shape: std::any::type_name::<S>(),
                    caller: caller.clone().into_boxed_str(),
                    dim,
                    runtime: a,
                    type_level: b,
                    runtime_shape: runtime_shape.clone().into_boxed_slice(),
                    type_level_shape: type_level_shape.clone().into_boxed_slice(),
                });
            }
        }

        Ok(Self {
            inner,
            _phantom: PhantomData,
        })
    }

    /// Reference to the untyped inner `SharedQTensor`.
    pub fn inner(&self) -> &paramecia_core::quantized::SharedQTensor {
        &self.inner
    }

    /// Consume and return the untyped inner `SharedQTensor`.
    pub fn into_inner(self) -> paramecia_core::quantized::SharedQTensor {
        self.inner
    }

    /// Dequantize to a typed `Tensor<S>` in the tensor's native dtype/device.
    pub fn dequant(&self) -> Result<Tensor<S>, Error> {
        let t = self.inner.dequant()?;
        // Shape was validated at construction; skip re-validation.
        Ok(Tensor(t, PhantomData))
    }

    /// Dequantize to a typed `Tensor<S>`, converting to `dtype` on `device`.
    pub fn dequant_to(&self, dtype: DType, device: &Device) -> Result<Tensor<S>, Error> {
        let t = self
            .inner
            .read()
            .unwrap()
            .dequantize(device)?
            .to_dtype(dtype)?;
        // Shape was validated at construction; skip re-validation.
        Ok(Tensor(t, PhantomData))
    }

    /// Read access to the underlying `QTensor`.
    pub fn read(&self) -> std::sync::LockResult<std::sync::RwLockReadGuard<'_, QTensor>> {
        self.inner.read()
    }

    /// Replace the underlying `QTensor` and bump the generation counter.
    pub fn replace(&self, qtensor: QTensor) {
        self.inner.replace(qtensor);
    }

    /// Current generation counter.
    pub fn generation(&self) -> u64 {
        self.inner.generation()
    }
}
