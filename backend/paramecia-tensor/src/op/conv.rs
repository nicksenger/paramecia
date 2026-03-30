use std::marker::PhantomData;

use crate::{Error, Tensor};
use glowstick::op::convolution::IsCompatible;
use glowstick::{
    num::{Unsigned, U0},
    op::convolution,
    Indexed, Shape, ShapeDiagnostic, ShapeFragment,
};
use inception::primitive;
use paramecia_arrow::{
    vis::{Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Applies a 2D convolution over the input tensor with the provided kernel, padding,
/// dilation, stride and groups.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{conv2d, Tensor};
/// use glowstick::{Shape4, num::*};
///
/// let device = Device::Cpu;
/// let input = Tensor::<Shape4<U2, U2, U5, U5>>::ones(DType::F32, &device)?;
/// let kernel = Tensor::<Shape4<U4, U2, U3, U3>>::ones(DType::F32, &device)?;
/// let convolved = conv2d!(input, kernel, U0, U1, U1, 1)?;
///
/// assert_eq!(convolved.dims(), &[2, 4, 3, 3]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! conv2d {
    ($t:expr,$kernel:expr,$padding:ty,$dilation:ty,$stride:ty,$groups:expr) => {{
        use std::marker::PhantomData;
        use $crate::op::conv::Conv2d;
        type Pad = glowstick::list![$padding, $padding];
        type Dilation = glowstick::list![$dilation, $dilation];
        type Stride = glowstick::list![$stride, $stride];
        (
            $t,
            $kernel,
            PhantomData::<Pad>,
            PhantomData::<Pad>,
            PhantomData::<Dilation>,
            PhantomData::<Stride>,
            $groups,
        )
            .conv2d()
    }};
}

pub trait Conv2d {
    type Out;
    fn conv2d(self) -> Self::Out;
}

use convolution::Kernel;
use glowstick::num::U1;
use glowstick::num::{Sub, ZipSubOneMul};
use glowstick::{Container, Empty, List, Map, Mappend, TakeFragment};

impl<T, K, P1, P2, S, D> Conv2d
    for (
        Tensor<T>,
        Tensor<K>,
        PhantomData<P1>,
        PhantomData<P2>,
        PhantomData<D>,
        PhantomData<S>,
        usize,
    )
where
    (T, K, P1, P2, S, D): convolution::IsCompatible,
    (P1, U0): Indexed,
    (S, U0): Indexed,
    (D, U0): Indexed,
    <(P1, U0) as Indexed>::Out: Unsigned,
    <(S, U0) as Indexed>::Out: Unsigned,
    <(D, U0) as Indexed>::Out: Unsigned,
    T: Shape + ShapeDiagnostic,
    K: Kernel<D> + ShapeDiagnostic,
    (T, K, P1, P2, S, D): IsCompatible,
    (
        <T as Shape>::Rank,
        <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): Sub,
    (
        <(
            <T as Shape>::Rank,
            <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as Sub>::Out,
        U1,
    ): Sub,
    <K as Kernel<D>>::DilateZipped: Container,
    (<K as Kernel<D>>::DilateZipped, ZipSubOneMul):
        Map<<<K as Kernel<D>>::DilateZipped as Container>::Content, ZipSubOneMul>,
    (
        T,
        <(
            <(
                <T as Shape>::Rank,
                <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as Sub>::Out,
            U1,
        ) as Sub>::Out,
    ): TakeFragment,
    (
        <(
            T,
            <(
                <(
                    <T as Shape>::Rank,
                    <<K as Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as Sub>::Out,
                U1,
            ) as Sub>::Out,
        ) as TakeFragment>::Out,
        List<(<K as Kernel<D>>::M, Empty)>,
    ): Mappend,
    (T, K, P1, P2, S, D): convolution::Compatible,
{
    type Out = Result<Tensor<<(T, K, P1, P2, S, D) as convolution::Compatible>::Out>, Error>;

    fn conv2d(self) -> Self::Out {
        let p = <<(P1, U0) as Indexed>::Out as glowstick::num::Unsigned>::USIZE;
        let s = <<(S, U0) as Indexed>::Out as glowstick::num::Unsigned>::USIZE;
        let d = <<(D, U0) as Indexed>::Out as glowstick::num::Unsigned>::USIZE;
        self.0
            .inner()
            .conv2d(self.1.inner(), p, s, d, self.6)?
            .try_into()
    }
}

pub struct Conv2dOp<T, K, P1, P2, S, D> {
    _phantom: PhantomData<(T, K, P1, P2, S, D)>,
    groups: usize,
}
impl<T, K, P1, P2, S, D> Conv2dOp<T, K, P1, P2, S, D> {
    pub fn new(groups: usize) -> Self {
        Self {
            _phantom: PhantomData,
            groups,
        }
    }
}
#[primitive(property = Arrow)]
impl<T, K, P1, P2, S, D> Combinator for Conv2dOp<T, K, P1, P2, S, D>
where
    (T, K, P1, P2, S, D): convolution::IsCompatible,
    (P1, U0): Indexed,
    (S, U0): Indexed,
    (D, U0): Indexed,
    <(P1, U0) as Indexed>::Out: Unsigned,
    <(S, U0) as Indexed>::Out: Unsigned,
    <(D, U0) as Indexed>::Out: Unsigned,
    T: Shape + glowstick::ShapeDiagnostic + ShapeDiagnostic,
    K: convolution::Kernel<D> + ShapeDiagnostic,
    (
        <T as Shape>::Rank,
        <<K as convolution::Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): glowstick::num::Sub,
    (
        <(
            <T as Shape>::Rank,
            <<K as convolution::Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as glowstick::num::Sub>::Out,
        glowstick::num::U1,
    ): glowstick::num::Sub,
    <K as convolution::Kernel<D>>::DilateZipped: glowstick::Container,
    (
        <K as convolution::Kernel<D>>::DilateZipped,
        glowstick::num::ZipSubOneMul,
    ): glowstick::Map<
        <<K as convolution::Kernel<D>>::DilateZipped as glowstick::Container>::Content,
        glowstick::num::ZipSubOneMul,
    >,
    (
        T,
        <(
            <(
                <T as Shape>::Rank,
                <<K as convolution::Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as glowstick::num::Sub>::Out,
            glowstick::num::U1,
        ) as glowstick::num::Sub>::Out,
    ): glowstick::TakeFragment,
    (
        <(
            T,
            <(
                <(
                    <T as Shape>::Rank,
                    <<K as convolution::Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as glowstick::num::Sub>::Out,
                glowstick::num::U1,
            ) as glowstick::num::Sub>::Out,
        ) as glowstick::TakeFragment>::Out,
        glowstick::List<(<K as convolution::Kernel<D>>::M, glowstick::Empty)>,
    ): glowstick::Mappend,
    (T, K, P1, P2, S, D): convolution::Compatible,
{
    type In = (Tensor<T>, Tensor<K>);
    type Out = Result<Tensor<<(T, K, P1, P2, S, D) as convolution::Compatible>::Out>, Error>;

    fn forward(&mut self, _ctx: &mut (), input: (Tensor<T>, Tensor<K>)) -> Self::Out {
        let _span = crate::op::trace::forward_2!("conv", T, K);
        let p = <<(P1, U0) as Indexed>::Out as Unsigned>::USIZE;
        let s = <<(S, U0) as Indexed>::Out as Unsigned>::USIZE;
        let d = <<(D, U0) as Indexed>::Out as Unsigned>::USIZE;
        input
            .0
            .inner()
            .conv2d(input.1.inner(), p, s, d, self.groups)?
            .try_into()
    }
}
#[primitive(property = Visualize)]
impl<T, K, P1, P2, S, D> Vis for Conv2dOp<T, K, P1, P2, S, D>
where
    (T, K, P1, P2, S, D): convolution::IsCompatible,
    (P1, U0): Indexed,
    (S, U0): Indexed,
    (D, U0): Indexed,
    <(P1, U0) as Indexed>::Out: Unsigned,
    <(S, U0) as Indexed>::Out: Unsigned,
    <(D, U0) as Indexed>::Out: Unsigned,
    T: Shape + ShapeDiagnostic,
    K: convolution::Kernel<D> + ShapeDiagnostic,
    (
        <T as Shape>::Rank,
        <<K as convolution::Kernel<D>>::Sp as ShapeFragment>::Rank,
    ): glowstick::num::Sub,
    (
        <(
            <T as Shape>::Rank,
            <<K as convolution::Kernel<D>>::Sp as ShapeFragment>::Rank,
        ) as glowstick::num::Sub>::Out,
        glowstick::num::U1,
    ): glowstick::num::Sub,
    <K as convolution::Kernel<D>>::DilateZipped: glowstick::Container,
    (
        <K as convolution::Kernel<D>>::DilateZipped,
        glowstick::num::ZipSubOneMul,
    ): glowstick::Map<
        <<K as convolution::Kernel<D>>::DilateZipped as glowstick::Container>::Content,
        glowstick::num::ZipSubOneMul,
    >,
    (
        T,
        <(
            <(
                <T as Shape>::Rank,
                <<K as convolution::Kernel<D>>::Sp as ShapeFragment>::Rank,
            ) as glowstick::num::Sub>::Out,
            glowstick::num::U1,
        ) as glowstick::num::Sub>::Out,
    ): glowstick::TakeFragment,
    (
        <(
            T,
            <(
                <(
                    <T as Shape>::Rank,
                    <<K as convolution::Kernel<D>>::Sp as ShapeFragment>::Rank,
                ) as glowstick::num::Sub>::Out,
                glowstick::num::U1,
            ) as glowstick::num::Sub>::Out,
        ) as glowstick::TakeFragment>::Out,
        glowstick::List<(<K as convolution::Kernel<D>>::M, glowstick::Empty)>,
    ): glowstick::Mappend,
    (T, K, P1, P2, S, D): convolution::Compatible,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            &format!(
                "Conv2d(pad={}, stride={}, dilation={})",
                <<(P1, U0) as Indexed>::Out as Unsigned>::USIZE,
                <<(S, U0) as Indexed>::Out as Unsigned>::USIZE,
                <<(D, U0) as Indexed>::Out as Unsigned>::USIZE
            ),
            None,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use glowstick::{
        num::{U0, U1, U2, U3, U4, U5},
        Shape4,
    };

    use crate::Tensor;

    #[test]
    fn conv2d_op() {
        let device = paramecia_core::Device::Cpu;

        type T = Shape4<U2, U2, U5, U5>;
        type K = Shape4<U4, U2, U3, U3>;
        type Pad = glowstick::list![U0, U0];
        type Dilation = glowstick::list![U1, U1];
        type Stride = glowstick::list![U1, U1];
        let input = Tensor::<T>::ones(paramecia_core::DType::F32, &device).unwrap();
        let kernel = Tensor::<K>::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = Conv2dOp<T, K, Pad, Pad, Stride, Dilation>;
        let res = MyOp::forward(
            &mut Conv2dOp {
                _phantom: PhantomData,
                groups: 1,
            },
            &mut (),
            (input, kernel),
        )
        .unwrap();
        assert_eq!(res.dims(), &[2, 4, 3, 3]);
    }
}
