use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Moves a tensor to a target device.
/// Shape-preserving: input and output have the same shape.
pub struct ToDeviceOp<S> {
    device: paramecia_core::Device,
    _phantom: PhantomData<S>,
}

impl<S> ToDeviceOp<S> {
    pub fn new(device: paramecia_core::Device) -> Self {
        Self {
            device,
            _phantom: PhantomData,
        }
    }
}

#[primitive(property = Arrow)]
impl<S> Combinator for ToDeviceOp<S>
where
    S: Shape + glowstick::ShapeDiagnostic,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<S>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("to_device", S);
        input.to_device(&self.device)
    }
}

#[primitive(property = Visualize)]
impl<S> Vis for ToDeviceOp<S>
where
    S: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "ToDevice",
            Some(&pretty_shape(std::any::type_name::<
                <S as glowstick::ShapeDiagnostic>::Out,
            >())),
        )
    }
}

#[cfg(test)]
mod test {
    use glowstick::{
        assert_shape_eq,
        num::{U2, U3},
        Shape2,
    };

    use super::*;

    #[test]
    fn to_device_op() -> Result<(), crate::Error> {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape2<U2, U3>>;
        let a = A::ones(paramecia_core::DType::F32, &device)?;

        let mut op = ToDeviceOp::<Shape2<U2, U3>>::new(paramecia_core::Device::Cpu);
        let res = op.forward(&mut (), a)?;
        assert_shape_eq!(res, Shape2<U2, U3>);
        Ok(())
    }
}
