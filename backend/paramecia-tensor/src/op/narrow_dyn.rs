use std::marker::PhantomData;

use glowstick::{num::Unsigned, op::narrow_dyn, Shape, ShapeDiagnostic};
use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

#[allow(unused)]
pub trait NarrowDyn {
    type Out;
    fn narrow_dyn(&self) -> Self::Out;
}
impl<S, Dim, DynDim> NarrowDyn
    for (
        Tensor<S>,
        PhantomData<Dim>,
        PhantomData<DynDim>,
        usize,
        usize,
    )
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim, DynDim): narrow_dyn::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, DynDim) as narrow_dyn::Compatible>::Out>, Error>;
    fn narrow_dyn(&self) -> Self::Out {
        self.0
            .inner()
            .narrow(<Dim as Unsigned>::USIZE, self.3, self.4)?
            .try_into()
    }
}
impl<S, Dim, DynDim> NarrowDyn
    for (
        &Tensor<S>,
        PhantomData<Dim>,
        PhantomData<DynDim>,
        usize,
        usize,
    )
where
    S: Shape,
    Dim: Unsigned,
    (S, Dim, DynDim): narrow_dyn::Compatible,
{
    type Out = Result<Tensor<<(S, Dim, DynDim) as narrow_dyn::Compatible>::Out>, Error>;
    fn narrow_dyn(&self) -> Self::Out {
        self.0
            .inner()
            .narrow(<Dim as Unsigned>::USIZE, self.3, self.4)?
            .try_into()
    }
}

pub struct NarrowDynOp<S, Dim, DynDim> {
    _phantom: PhantomData<(S, Dim, DynDim)>,
    start: usize,
    len: usize,
}
impl<S, Dim, DynDim> NarrowDynOp<S, Dim, DynDim> {
    pub fn new(start: usize, len: usize) -> Self {
        Self {
            _phantom: PhantomData,
            start,
            len,
        }
    }
}
#[primitive(property = Arrow)]
impl<S, Dim, DynDim> Combinator for NarrowDynOp<S, Dim, DynDim>
where
    S: Shape + glowstick::ShapeDiagnostic,
    Dim: Unsigned,
    (S, Dim, DynDim): narrow_dyn::Compatible,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<<(S, Dim, DynDim) as narrow_dyn::Compatible>::Out>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("narrow_dyn", S);
        input
            .inner()
            .narrow(<Dim as Unsigned>::USIZE, self.start, self.len)?
            .try_into()
    }
}
#[primitive(property = Visualize)]
impl<S, Dim, DynDim> Vis for NarrowDynOp<S, Dim, DynDim>
where
    S: Shape + ShapeDiagnostic,
    Dim: Unsigned,
    (S, Dim, DynDim): narrow_dyn::Compatible,
    <(S, Dim, DynDim) as narrow_dyn::Compatible>::Out: ShapeDiagnostic,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            &format!("NarrowDyn(dim={})", <Dim as Unsigned>::USIZE),
            Some(&pretty_shape(std::any::type_name::<
                <<(S, Dim, DynDim) as narrow_dyn::Compatible>::Out as ShapeDiagnostic>::Out,
            >())),
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use glowstick::{
        num::{U0, U2, U3, U4},
        Shape3,
    };

    use crate::Tensor;

    #[test]
    fn narrow_dyn_op() {
        glowstick::dyndims! {
            N: SequenceLength
        }

        let device = paramecia_core::Device::Cpu;

        type S = Shape3<U2, U3, U4>;
        type A = Tensor<S>;
        let a = A::ones(paramecia_core::DType::F32, &device).unwrap();

        type MyOp = NarrowDynOp<S, U0, glowstick::Dyn<N>>;
        let res = MyOp::forward(
            &mut NarrowDynOp {
                _phantom: PhantomData,
                start: 0,
                len: 1,
            },
            &mut (),
            a,
        )
        .unwrap();
        assert_eq!(res.dims(), &[1, 3, 4]);
    }
}
