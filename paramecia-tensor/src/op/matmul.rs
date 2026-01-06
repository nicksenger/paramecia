use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{op::matmul, Shape, ShapeDiagnostic, TensorShape};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Performs matrix multiplication of the lefthand tensor and righthand tensor(s).
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{matmul, Tensor};
/// use glowstick::{Shape2, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U2, U1>>::from_vec(vec![4f32, 5.], &device)?;
/// let b = Tensor::<Shape2<U1, U2>>::from_vec(vec![5f32, 4.], &device)?;
/// let c = matmul!(a, b)?;
///
/// assert_eq!(
///     c.inner().to_vec2::<f32>()?,
///     vec![
///         vec![20., 16.],
///         vec![25., 20.]
///     ]
/// );
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! matmul {
    ($t1:expr,$t2:expr) => {{
        use $crate::op::matmul::Matmul;
        ($t1, $t2, std::marker::PhantomData).matmul()
    }};
    ($t1:expr,$t2:expr,$($t2s:expr),+) => {{
        use $crate::op::matmul::Matmul;
        ($t1, $t2, std::marker::PhantomData)
            .matmul()
            .and_then(|t| $crate::matmul!(&t, $t2s))
    }};
}

pub trait Matmul {
    type Out;
    fn matmul(self) -> Self::Out;
}
impl<S1, U, S2> Matmul for (Tensor<S1>, U, PhantomData<S2>)
where
    U: Borrow<Tensor<S2>>,
    S1: Shape + matmul::Operand,
    S2: Shape + matmul::Operand,
    (S1, S2): matmul::Compatible,
{
    type Out = Result<Tensor<TensorShape<<(S1, S2) as matmul::Compatible>::Out>>, Error>;
    fn matmul(self) -> Self::Out {
        self.0
            .into_inner()
            .matmul(self.1.borrow().inner())?
            .try_into()
    }
}
impl<S1, U, S2> Matmul for (&Tensor<S1>, U, PhantomData<S2>)
where
    U: Borrow<Tensor<S2>>,
    S1: Shape + matmul::Operand,
    S2: Shape + matmul::Operand,
    (S1, S2): matmul::Compatible,
{
    type Out = Result<Tensor<TensorShape<<(S1, S2) as matmul::Compatible>::Out>>, Error>;
    fn matmul(self) -> Self::Out {
        self.0.inner().matmul(self.1.borrow().inner())?.try_into()
    }
}

pub struct MatmulOp<T, U>(PhantomData<T>, PhantomData<U>);
impl<T, U> Default for MatmulOp<T, U> {
    fn default() -> Self {
        Self(PhantomData, PhantomData)
    }
}

#[primitive(property = Arrow)]
impl<S1, S2> Combinator for MatmulOp<S1, S2>
where
    S1: Shape + glowstick::ShapeDiagnostic + matmul::Operand,
    S2: Shape + glowstick::ShapeDiagnostic + matmul::Operand,
    (S1, S2): matmul::Compatible,
{
    type In = (Tensor<S1>, Tensor<S2>);
    type Out = Result<Tensor<TensorShape<<(S1, S2) as matmul::Compatible>::Out>>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: (Tensor<S1>, Tensor<S2>)) -> Self::Out {
        let _span = crate::op::trace::forward_2!("matmul", S1, S2);
        matmul!(input.0, input.1)
    }
}
#[primitive(property = Visualize)]
impl<S1, S2> Vis for MatmulOp<S1, S2>
where
    S1: Shape + ShapeDiagnostic + matmul::Operand,
    S2: Shape + ShapeDiagnostic + matmul::Operand,
    (S1, S2): matmul::Compatible,
    TensorShape<<(S1, S2) as matmul::Compatible>::Out>: ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "Matmul",
            Some(&pretty_shape(std::any::type_name::<
                <TensorShape<<(S1, S2) as matmul::Compatible>::Out> as ShapeDiagnostic>::Out,
            >())),
        )
    }
}

#[cfg(test)]
mod test {
    use glowstick::{
        assert_shape_eq,
        num::{U1, U2},
        Shape2,
    };

    use super::*;

    #[test]
    fn matmul_op() {
        let device = paramecia_core::Device::Cpu;

        type A = Tensor<Shape2<U2, U1>>;
        type B = Tensor<Shape2<U1, U2>>;
        let a = A::from_vec(vec![4f32, 5.], &device).unwrap();
        let b = B::from_vec(vec![5f32, 4.], &device).unwrap();

        type MyOp = MatmulOp<Shape2<U2, U1>, Shape2<U1, U2>>;
        let res = MyOp::forward(&mut MatmulOp(PhantomData, PhantomData), &mut (), (a, b)).unwrap();
        assert_shape_eq!(res, Shape2<U2, U2>);
    }
}
