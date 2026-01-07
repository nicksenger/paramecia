use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Clamps tensor values into the range `[min, max]`.
/// Shape-preserving: input and output have the same shape.
#[macro_export]
macro_rules! clamp {
    ($t:expr, $min:expr, $max:expr) => {{
        use $crate::op::clamp::Clamp;
        ($t, $min as f64, $max as f64).clamp_typed()
    }};
}

pub trait Clamp {
    type Out;
    fn clamp_typed(self) -> Self::Out;
}
impl<S> Clamp for (Tensor<S>, f64, f64)
where
    S: Shape,
{
    type Out = Result<Tensor<S>, crate::Error>;
    fn clamp_typed(self) -> Self::Out {
        let result = self.0 .0.clamp(self.1, self.2)?;
        result.try_into()
    }
}

pub struct ClampOp<S>(f64, f64, PhantomData<S>);
impl<S> ClampOp<S> {
    pub fn new(min: f64, max: f64) -> Self {
        Self(min, max, PhantomData)
    }
}
impl<S> Default for ClampOp<S> {
    fn default() -> Self {
        Self(0.0, 1.0, PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S> Combinator for ClampOp<S>
where
    S: Shape + glowstick::ShapeDiagnostic,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<S>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("clamp", S);
        clamp!(input, self.0, self.1)
    }
}
#[primitive(property = Visualize)]
impl<S> Vis for ClampOp<S>
where
    S: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "Clamp",
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
    fn clamp_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape2<U2, U3>>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = ClampOp<Shape2<U2, U3>>;
        let res = MyOp::forward(&mut ClampOp::new(-1.0, 1.0), &mut (), a).unwrap();
        assert_shape_eq!(res, Shape2<U2, U3>);
    }
}
