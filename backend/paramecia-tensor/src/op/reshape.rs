use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{op::reshape, Shape, ShapeDiagnostic};
use inception::primitive;
use paramecia_core::shape::ShapeWithOneHole;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Reshapes a tensor to the specified dimensions.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{reshape, Tensor};
/// use glowstick::{Shape2, Shape4, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U2, U3>>::ones(DType::F32, &device)?;
/// let reshaped = reshape!(a, [U1, U6])?;
///
/// assert_eq!(reshaped.dims(), &[1, 6]);
/// # Ok(())
/// # }
/// ```
///
/// When using dynamic dimensions, the resulting tensor's shape will be determined by the provided expressions.
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{reshape, Tensor};
/// use glowstick::{Shape2, num::*, dyndims};
///
/// dyndims! {
///     A: Rows,
///     B: Cols
/// }
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape2<U1, U4>>::ones(DType::F32, &device)?;
/// let [rows, cols] = [2, 2];
/// let reshaped = reshape!(a, [{ rows } => A, { cols } => B])?;
///
/// assert_eq!(reshaped.dims(), &[2, 2]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! reshape {
    ($t:expr,[$($ds:tt)+]) => {{
        type TS = $crate::glowstick::TensorShape<$crate::reshape_tys!($($ds)+)>;
        $crate::glowstick::op::reshape::check::<_, _, TS>(&$t);
        use $crate::op::reshape::Reshape;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<TS>,
        )
            .reshape($crate::reshape_val!($($ds)+).tuplify())
    }};
}

pub trait Reshape {
    type Out;
    fn reshape<Args: ShapeWithOneHole>(&self, args: Args) -> Self::Out;
}

impl<T, S1, S2> Reshape for (T, PhantomData<S1>, PhantomData<S2>)
where
    T: Borrow<Tensor<S1>>,
    S1: Shape,
    S2: Shape,
    (S1, S2): reshape::Compatible,
{
    type Out = Result<Tensor<<(S1, S2) as reshape::Compatible>::Out>, Error>;
    fn reshape<Args: ShapeWithOneHole>(&self, args: Args) -> Self::Out {
        self.0.borrow().inner().reshape(args)?.try_into()
    }
}

#[macro_export]
macro_rules! reshape_tys {
    ($e:expr => $d:ty) => {
        $crate::glowstick::Shp<(<$d as $crate::glowstick::dynamic::Dim>::Id, $crate::glowstick::Empty)>
    };
    ($e:expr => $d:ty,$($ds:tt)+) => {
        $crate::glowstick::Shp<(<$d as $crate::glowstick::dynamic::Dim>::Id, $crate::reshape_tys!($($ds)+))>
    };
    ($d:ty) => {
        $crate::glowstick::Shp<($d, $crate::glowstick::Empty)>
    };
    ($d:ty,$($ds:tt)+) => {
        $crate::glowstick::Shp<($d, $crate::reshape_tys!($($ds)+))>
    };
}
#[macro_export]
macro_rules! reshape_val {
    ($e:expr => $d:ty) => {
        $crate::glowstick::ValueList(($e, $crate::glowstick::ValueList(())))
    };
    ($d:ty) => {
        $crate::glowstick::ValueList((<$d as $crate::glowstick::num::Unsigned>::USIZE,$crate::glowstick::ValueList(())))
    };
    ($e:expr => $d:ty,$($ds:tt)+) => {
        $crate::glowstick::ValueList(($e,$crate::reshape_val!($($ds)+)))
    };
    ($d:ty,$($ds:tt)+) => {
        $crate::glowstick::ValueList((<$d as $crate::glowstick::num::Unsigned>::USIZE,$crate::reshape_val!($($ds)+)))
    };
}

pub struct ReshapeOp<S, S1, S2> {
    _phantom: PhantomData<(S1, S2)>,
    shape: S,
}
#[primitive(property = Arrow)]
impl<S1, S2, S> Combinator for ReshapeOp<S, S1, S2>
where
    S1: Shape + glowstick::ShapeDiagnostic,
    S2: Shape + glowstick::ShapeDiagnostic,
    S: ShapeWithOneHole + Clone,
    (S1, S2): reshape::Compatible,
{
    type In = Tensor<S1>;
    type Out = Result<Tensor<<(S1, S2) as reshape::Compatible>::Out>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S1>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("reshape", S1);
        input.inner().reshape(self.shape.clone())?.try_into()
    }
}
#[primitive(property = Visualize)]
impl<S1, S2, S> Vis for ReshapeOp<S, S1, S2>
where
    S1: Shape + ShapeDiagnostic,
    S2: Shape + ShapeDiagnostic,
    S: ShapeWithOneHole + Clone,
    (S1, S2): reshape::Compatible,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            "Reshape",
            Some(&pretty_shape(std::any::type_name::<
                <S2 as glowstick::ShapeDiagnostic>::Out,
            >())),
        )
    }
}
impl<S1, S2, S> ReshapeOp<S, S1, S2>
where
    S1: Shape,
{
    pub fn new(shape: S) -> Self {
        Self {
            _phantom: PhantomData,
            shape,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use glowstick::{
        assert_shape_eq,
        num::{U1, U2, U3, U6},
        Shape2,
    };

    use crate::Tensor;

    #[test]
    fn reshape_op() {
        let device = paramecia_core::Device::Cpu;

        type S1 = Shape2<U2, U3>;
        type S2 = Shape2<U1, U6>;
        type A = Tensor<S1>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = ReshapeOp<(usize, usize), S1, S2>;
        let res = MyOp::forward(&mut ReshapeOp::new((1, 6)), &mut (), a).unwrap();
        assert_shape_eq!(res, Shape2<U1, U6>);
    }
}
