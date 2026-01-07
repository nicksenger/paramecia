use std::{borrow::Borrow, marker::PhantomData};

use glowstick::{num::Unsigned, op::narrow, Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Narrows a tensor at the specified dimension from start index to length.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{narrow, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U3, U4>>::ones(DType::F32, &device)?;
/// let narrowed = narrow!(a, U0: [U1, U1])?;
///
/// assert_eq!(narrowed.dims(), &[1, 3, 4]);
/// # Ok(())
/// # }
/// ```
///
/// When using dynamic start and length, the resulting tensor's shape will be determined by the provided expressions.
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{narrow, Tensor};
/// use glowstick::{Shape3, num::{U0, U1, U2, U3, U4}, dyndims};
///
/// dyndims! {
///     N: SequenceLength
/// }
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U3, U4>>::ones(DType::F32, &device)?;
/// let [start, len] = [1, 2];
/// let narrowed = narrow!(a, U1: [{ start }, { len }] => N)?;
///
/// assert_eq!(narrowed.dims(), &[2, 2, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! narrow {
    ($t:expr,$d:ty:[$s:ty,$l:ty]) => {{
        $crate::glowstick::op::narrow::check::<_, _, $d, $s, $l>(&$t);
        use $crate::op::narrow::Narrow;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$s>,
            std::marker::PhantomData::<$l>
        ).narrow()
    }};
    ($t:expr,$d:ty:[$s:expr,$l:ty]) => {{
        $crate::glowstick::op::narrow::check::<_, _, $d, $crate::glowstick::num::U0, $l>(&$t);
        use $crate::op::narrow_dyn_start::NarrowDynStart;
        (
            $t,
            std::marker::PhantomData,
            std::marker::PhantomData::<$d>,
            $s,
            std::marker::PhantomData::<$l>,
        )
            .narrow_dyn_start()
    }};
    ($t:expr,$d:ty:[$s:expr,$l:expr] => $y:ty) => {{
        $crate::glowstick::op::narrow::check::<_, _, $d, $crate::glowstick::num::U0, $y>(&$t);
        use $crate::op::narrow_dyn::NarrowDyn;
        (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$y>,
            $s,
            $l,
        )
            .narrow_dyn()
    }};
    ($t:expr,$d:ty:[$s:ty,$l:ty],$($ds:tt)+) => {{
        $crate::glowstick::op::narrow::check::<_, _, $d, $s, $l>(&$t);
        use $crate::op::narrow::Narrow;
        (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$s>,
            std::marker::PhantomData::<$l>,
        )
            .narrow().and_then(|t| $crate::narrow!(&t,$($ds)+))
    }};
    ($t:expr,$d:ty:[$s:ty,$l:ty],$($ds:tt)+) => {{
        $crate::glowstick::op::narrow::check::<_, _, $d, $crate::glowstick::num::U0, $l>(&$t);
        use $crate::op::narrow_dyn_start::NarrowDynStart;
        (
            $t,
            std::marker::PhantomData::<$d>,
            $s,
            std::marker::PhantomData::<$l>,
        )
            .narrow_dyn_start().and_then(|t| $crate::narrow!(&t,$($ds)+))
    }};
    ($t:expr,$d:ty:[$s:expr,$l:expr] => $y:ty,$($ds:tt)+) => {{
        $crate::glowstick::op::narrow::check::<_, _, $d, $crate::glowstick::num::U0, $y>(&$t);
        use $crate::op::narrow_dyn::NarrowDyn;
        (
            $t,
            std::marker::PhantomData::<$d>,
            std::marker::PhantomData::<$y>,
            $s,
            $l,
        )
            .narrow_dyn().and_then(|t| $crate::narrow!(&t,$($ds)+))
    }};
}

#[allow(unused)]
pub trait Narrow {
    type Out;
    fn narrow(&self) -> Self::Out;
}
impl<T, S, Dim, Start, Len> Narrow
    for (
        T,
        PhantomData<S>,
        PhantomData<Dim>,
        PhantomData<Start>,
        PhantomData<Len>,
    )
where
    T: Borrow<Tensor<S>>,
    S: Shape,
    Dim: Unsigned,
    Start: Unsigned,
    Len: Unsigned,
    (S, Dim, Start, Len): narrow::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, Start, Len) as narrow::Compatible>::Out>, Error>;
    fn narrow(&self) -> Self::Out {
        self.0
            .borrow()
            .inner()
            .narrow(
                <Dim as Unsigned>::USIZE,
                <Start as Unsigned>::USIZE,
                <Len as Unsigned>::USIZE,
            )?
            .try_into()
    }
}

pub struct NarrowOp<S, Dim, Start, Len>(
    PhantomData<S>,
    PhantomData<Dim>,
    PhantomData<Start>,
    PhantomData<Len>,
);
impl<A, B, C, D> Default for NarrowOp<A, B, C, D> {
    fn default() -> Self {
        Self(PhantomData, PhantomData, PhantomData, PhantomData)
    }
}

#[primitive(property = Arrow)]
impl<S, Dim, Start, Len> Combinator for NarrowOp<S, Dim, Start, Len>
where
    S: Shape + glowstick::ShapeDiagnostic,
    Dim: Unsigned,
    Start: Unsigned,
    Len: Unsigned,
    (S, Dim, Start, Len): narrow::Compatible,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<<(S, Dim, Start, Len) as narrow::Compatible>::Out>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("narrow", S);
        Ok(crate::Tensor(
            input.inner().narrow(
                <Dim as Unsigned>::USIZE,
                <Start as Unsigned>::USIZE,
                <Len as Unsigned>::USIZE,
            )?,
            PhantomData,
        ))
    }
}
#[primitive(property = Visualize)]
impl<S, Dim, Start, Len> Vis for NarrowOp<S, Dim, Start, Len>
where
    S: Shape + ShapeDiagnostic,
    Dim: Unsigned,
    Start: Unsigned,
    Len: Unsigned,
    (S, Dim, Start, Len): narrow::Compatible,
    <(S, Dim, Start, Len) as narrow::Compatible>::Out: ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            &format!(
                "Narrow(dim={}, start={}, len={})",
                <Dim as Unsigned>::USIZE,
                <Start as Unsigned>::USIZE,
                <Len as Unsigned>::USIZE
            ),
            Some(&pretty_shape(std::any::type_name::<
                <<(S, Dim, Start, Len) as narrow::Compatible>::Out as ShapeDiagnostic>::Out,
            >())),
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use glowstick::{
        assert_shape_eq,
        num::{U0, U1, U2, U3, U4},
        Shape3,
    };

    use crate::Tensor;

    #[test]
    fn narrow_op() {
        let device = paramecia_core::Device::Cpu;

        type S = Shape3<U2, U3, U4>;
        type A = Tensor<S>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        let res = NarrowOp::<S, U0, U1, U1>::forward(
            &mut NarrowOp(PhantomData, PhantomData, PhantomData, PhantomData),
            &mut (),
            a,
        )
        .unwrap();
        assert_shape_eq!(res, Shape3<U1, U3, U4>);
    }
}
