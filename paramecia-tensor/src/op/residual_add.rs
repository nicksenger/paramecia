use std::marker::PhantomData;

use glowstick::{Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Element-wise addition of two tensors with the same shape (residual connection).
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{residual_add, Tensor};
/// use glowstick::{Shape2, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U2, U3>>::ones(DType::F32, &device)?;
/// let b = Tensor::<Shape2<U2, U3>>::ones(DType::F32, &device)?;
/// let c = residual_add!(a, b)?;
///
/// assert_eq!(c.dims(), &[2, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! residual_add {
    ($t1:expr,$t2:expr) => {{
        use $crate::op::residual_add::ResidualAdd;
        ($t1, $t2).residual_add()
    }};
}

pub trait ResidualAdd {
    type Out;
    fn residual_add(self) -> Self::Out;
}
impl<S> ResidualAdd for (Tensor<S>, Tensor<S>)
where
    S: Shape,
{
    type Out = Result<Tensor<S>, crate::Error>;
    fn residual_add(self) -> Self::Out {
        let result = (&self.0 .0 + &self.1 .0)?;
        result.try_into()
    }
}
impl<S> ResidualAdd for (&Tensor<S>, &Tensor<S>)
where
    S: Shape,
{
    type Out = Result<Tensor<S>, crate::Error>;
    fn residual_add(self) -> Self::Out {
        let result = (&self.0 .0 + &self.1 .0)?;
        result.try_into()
    }
}

pub struct ResidualAddOp<S>(PhantomData<S>);
impl<S> Default for ResidualAddOp<S> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S> Combinator for ResidualAddOp<S>
where
    S: Shape + glowstick::ShapeDiagnostic,
{
    type In = (Tensor<S>, Tensor<S>);
    type Out = Result<Tensor<S>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: (Tensor<S>, Tensor<S>)) -> Self::Out {
        let _span = crate::op::trace::forward_2!("residual_add", S, S);
        residual_add!(input.0, input.1)
    }
}
#[primitive(property = Visualize)]
impl<S> Vis for ResidualAddOp<S>
where
    S: Shape + ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "ResidualAdd",
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
    fn residual_add_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape2<U2, U3>>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();
        let b = A::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = ResidualAddOp<Shape2<U2, U3>>;
        let res = MyOp::forward(&mut ResidualAddOp(PhantomData), &mut (), (a, b)).unwrap();
        assert_shape_eq!(res, Shape2<U2, U3>);

        // Verify values are 2.0 (1.0 + 1.0)
        let vals = res.inner().to_vec2::<f32>().unwrap();
        assert_eq!(vals, vec![vec![2.0, 2.0, 2.0], vec![2.0, 2.0, 2.0]]);
    }
}
