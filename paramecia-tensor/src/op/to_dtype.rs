use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Casts a tensor to a different dtype.
/// Shape-preserving: input and output have the same shape.
pub struct ToDtypeOp<S> {
    dtype: paramecia_core::DType,
    _phantom: PhantomData<S>,
}

impl<S> ToDtypeOp<S> {
    pub fn new(dtype: paramecia_core::DType) -> Self {
        Self {
            dtype,
            _phantom: PhantomData,
        }
    }
}

#[primitive(property = Arrow)]
impl<S> Combinator for ToDtypeOp<S>
where
    S: Shape + glowstick::ShapeDiagnostic,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<S>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("to_dtype", S);
        let result = input.0.to_dtype(self.dtype)?;
        result.try_into()
    }
}
#[primitive(property = Visualize)]
impl<S> Vis for ToDtypeOp<S>
where
    S: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "ToDtype",
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
    fn to_dtype_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape2<U2, U3>>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        let mut op = ToDtypeOp::<Shape2<U2, U3>>::new(paramecia_core::DType::F64);
        let res = op.forward(&mut (), a).unwrap();
        assert_shape_eq!(res, Shape2<U2, U3>);
    }
}
