use std::marker::PhantomData;

use glowstick::Shape;
use inception::primitive;
use paramecia_arrow::{
    vis::{Graph, Vis, Visualize},
    Arrow, Combinator,
};
use paramecia_core::{Device, WithDType};

use crate::{Error, Tensor};

/// Creates a rank-1 typed tensor from host data on the specified device.
pub struct FromVec1OnDeviceOp<S, T>(PhantomData<(S, T)>);
impl<S, T> Default for FromVec1OnDeviceOp<S, T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S, T> Combinator for FromVec1OnDeviceOp<S, T>
where
    S: Shape + glowstick::ShapeDiagnostic,
    T: WithDType,
{
    type In = (Vec<T>, Device);
    type Out = Result<Tensor<S>, Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_output!("from_vec_on_device", S);
        let (data, device) = input;
        let core = paramecia_core::Tensor::new(data.as_slice(), &device)?;
        core.try_into()
    }
}
#[primitive(property = Visualize)]
impl<S, T> Vis for FromVec1OnDeviceOp<S, T>
where
    S: Shape,
    T: WithDType,
{
    fn visualize() -> Graph {
        Graph::leaf("FromVec1OnDevice")
    }
}

/// Creates a rank-2 typed column tensor `[len, 1]` from host data on the specified device.
pub struct FromVecColOnDeviceOp<S, T>(PhantomData<(S, T)>);
impl<S, T> Default for FromVecColOnDeviceOp<S, T> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S, T> Combinator for FromVecColOnDeviceOp<S, T>
where
    S: Shape + glowstick::ShapeDiagnostic,
    T: WithDType,
{
    type In = (Vec<T>, Device);
    type Out = Result<Tensor<S>, Error>;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_output!("from_vec_on_device", S);
        let (data, device) = input;
        let len = data.len();
        let core = paramecia_core::Tensor::new(data.as_slice(), &device)?.reshape((len, 1))?;
        core.try_into()
    }
}
#[primitive(property = Visualize)]
impl<S, T> Vis for FromVecColOnDeviceOp<S, T>
where
    S: Shape,
    T: WithDType,
{
    fn visualize() -> Graph {
        Graph::leaf("FromVecColOnDevice")
    }
}
