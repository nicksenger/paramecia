use std::marker::PhantomData;

use glowstick::Shape;
use paramecia_core::quantized::{QStorage, SharedQTensor};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(
        "{context}: rank mismatch for typed shape {typed_shape}; runtime rank {runtime_rank} shape {runtime_shape:?}, type-level rank {type_level_rank} shape {type_level_shape:?} (0 in type-level shape means dynamic), callsite {caller}"
    )]
    RankMismatch {
        context: &'static str,
        typed_shape: &'static str,
        caller: Box<str>,
        runtime_rank: usize,
        type_level_rank: usize,
        runtime_shape: Box<[usize]>,
        type_level_shape: Box<[usize]>,
    },

    #[error(
        "{context}: dimension mismatch for typed shape {typed_shape} at dim {dim}; runtime {runtime}, type-level {type_level}. runtime shape {runtime_shape:?}, type-level shape {type_level_shape:?} (0 in type-level shape means dynamic), callsite {caller}"
    )]
    DimensionMismatch {
        context: &'static str,
        typed_shape: &'static str,
        caller: Box<str>,
        dim: usize,
        runtime: usize,
        type_level: usize,
        runtime_shape: Box<[usize]>,
        type_level_shape: Box<[usize]>,
    },

    #[error("{0}")]
    Core(#[from] paramecia_core::Error),

    #[error("{0}")]
    Msg(String),
}

impl From<Error> for paramecia_core::Error {
    fn from(e: Error) -> Self {
        match e {
            Error::Core(core_err) => core_err,
            other => paramecia_core::Error::Msg(other.to_string()),
        }
    }
}
impl Clone for Error {
    fn clone(&self) -> Self {
        Error::Msg(self.to_string())
    }
}

#[allow(unused)]
#[derive(Debug)]
pub struct Tensor<S: Shape>(pub(crate) paramecia_core::Tensor, pub(crate) PhantomData<S>);

impl<S: Shape> Clone for Tensor<S> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<S> glowstick::Tensor for Tensor<S>
where
    S: Shape,
{
    type Shape = S;
}
impl<S> glowstick::Tensor for &Tensor<S>
where
    S: Shape,
{
    type Shape = S;
}

/// Convert from SharedQTensor (for weight loading): dequant then validate shape.
impl<S> TryFrom<SharedQTensor> for Tensor<S>
where
    S: Shape,
{
    type Error = Error;
    fn try_from(x: SharedQTensor) -> Result<Self, Self::Error> {
        let t = x.dequant()?;
        t.try_into()
    }
}

/// Convert from paramecia_core::Tensor: zero-cost validate rank+dims and wrap.
impl<S> TryFrom<paramecia_core::Tensor> for Tensor<S>
where
    S: Shape,
{
    type Error = Error;
    #[track_caller]
    fn try_from(t: paramecia_core::Tensor) -> Result<Self, Self::Error> {
        let caller = std::panic::Location::caller().to_string();
        let runtime_shape = t.dims().to_vec();
        let type_level_shape = S::iter().collect::<Vec<_>>();

        if S::RANK != t.rank() {
            return Err(Error::RankMismatch {
                context: "Typed tensor conversion (core Tensor -> Tensor<S>)",
                typed_shape: std::any::type_name::<S>(),
                caller: caller.into_boxed_str(),
                runtime_rank: t.rank(),
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
            // b == 0 means dynamic dimension (any value accepted)
            if b != 0 && a != b {
                return Err(Error::DimensionMismatch {
                    context: "Typed tensor conversion (core Tensor -> Tensor<S>)",
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

        Ok(Self(t, PhantomData))
    }
}

impl<S> Tensor<S>
where
    S: Shape,
{
    pub fn contiguous(&self) -> Result<Self, Error> {
        self.inner().contiguous()?.try_into()
    }

    pub fn to_dtype(&self, dtype: paramecia_core::DType) -> Result<Self, Error> {
        self.inner().to_dtype(dtype)?.try_into()
    }

    pub fn to_device(&self, device: &paramecia_core::Device) -> Result<Self, Error> {
        self.inner().to_device(device)?.try_into()
    }

    pub fn inner(&self) -> &paramecia_core::Tensor {
        &self.0
    }

    pub fn shape() -> impl Into<paramecia_core::Shape> {
        <S as Shape>::iter().collect::<Vec<_>>()
    }

    pub fn dims(&self) -> Vec<usize> {
        self.0.dims().to_vec()
    }

    pub fn new(storage: QStorage) -> Result<Self, Error> {
        let qt = paramecia_core::quantized::QTensor::new(storage, Self::shape())?;
        let sqt = SharedQTensor::new(qt);
        let t = sqt.dequant()?;
        t.try_into()
    }

    /// Return the core tensor, discarding type information
    pub fn into_inner(self) -> paramecia_core::Tensor {
        self.0
    }

    /// Create a typed tensor from a Vec on the given device.
    pub fn from_vec<D: paramecia_core::WithDType>(
        v: Vec<D>,
        device: &paramecia_core::Device,
    ) -> Result<Self, Error> {
        let shape_vec: Vec<usize> = <S as Shape>::iter().collect();
        let core_tensor = paramecia_core::Tensor::from_vec(v, shape_vec, device)?;
        core_tensor.try_into()
    }

    /// Create a typed tensor filled with ones.
    pub fn ones(
        dtype: paramecia_core::DType,
        device: &paramecia_core::Device,
    ) -> Result<Self, Error> {
        let shape_vec: Vec<usize> = <S as Shape>::iter().collect();
        let core_tensor = paramecia_core::Tensor::ones(shape_vec, dtype, device)?;
        core_tensor.try_into()
    }

    /// Create a typed tensor filled with zeros.
    pub fn zeros(
        dtype: paramecia_core::DType,
        device: &paramecia_core::Device,
    ) -> Result<Self, Error> {
        let shape_vec: Vec<usize> = <S as Shape>::iter().collect();
        let core_tensor = paramecia_core::Tensor::zeros(shape_vec, dtype, device)?;
        core_tensor.try_into()
    }
}
