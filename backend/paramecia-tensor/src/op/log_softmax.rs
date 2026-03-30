use std::marker::PhantomData;

use glowstick::cmp::Greater;
use glowstick::num::Unsigned;
use glowstick::{Shape, ShapeDiagnostic};

use inception::primitive;

use crate::{Error, Tensor};
use paramecia_arrow::{
    vis::{pretty_shape, Graph, Vis, Visualize},
    Arrow, Combinator,
};

/// Applies the log softmax function to a tensor along the specified dimension.
///
/// # Example
///
/// ```rust
/// # fn main() -> Result<(), paramecia_tensor::Error> {
/// use paramecia_core::{Device, DType};
/// use paramecia_tensor::{log_softmax, Tensor};
/// use glowstick::{Shape3, num::*};
///
/// let device = Device::Cpu;
/// let a = Tensor::<Shape3<U2, U3, U4>>::ones(DType::F32, &device)?;
/// let logsoftmaxed = log_softmax!(a, U1)?;
///
/// assert_eq!(logsoftmaxed.dims(), &[2, 3, 4]);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! log_softmax {
    ($t:expr,$i:ty) => {{
        use $crate::op::log_softmax::LogSoftmax;
        ($t, std::marker::PhantomData::<$i>).log_softmax()
    }};
    ($t:expr,$i:ty,$($is:ty),+) => {{
        $crate::log_softmax!($crate::log_softmax!($t,$i),$($is),+)
    }};
}

pub trait LogSoftmax {
    type Out;
    fn log_softmax(self) -> Self::Out;
}
impl<S, Dim> LogSoftmax for (Tensor<S>, PhantomData<Dim>)
where
    S: Shape,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type Out = Result<Tensor<S>, crate::Error>;
    fn log_softmax(self) -> Self::Out {
        let result = paramecia_nn::ops::log_softmax(&self.0 .0, <Dim as Unsigned>::USIZE)?;
        result.try_into()
    }
}

pub struct LogSoftmaxOp<S, Dim>(PhantomData<S>, PhantomData<Dim>);
impl<S, Dim> Default for LogSoftmaxOp<S, Dim> {
    fn default() -> Self {
        Self(PhantomData, PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S, Dim> Combinator for LogSoftmaxOp<S, Dim>
where
    S: Shape + glowstick::ShapeDiagnostic,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    type In = Tensor<S>;
    type Out = Result<Tensor<S>, Error>;
    fn forward(&mut self, _ctx: &mut (), input: Tensor<S>) -> Self::Out {
        let _span = crate::op::trace::forward_1!("log_softmax", S);
        log_softmax!(input, Dim)
    }
}
#[primitive(property = Visualize)]
impl<S, Dim> Vis for LogSoftmaxOp<S, Dim>
where
    S: Shape + ShapeDiagnostic,
    Dim: Unsigned,
    (<S as Shape>::Rank, Dim): Greater,
{
    fn visualize() -> Graph {
        Graph::leaf_with_edges(
            &format!("LogSoftmax(dim={})", <Dim as Unsigned>::USIZE),
            Some(&pretty_shape(std::any::type_name::<
                <S as glowstick::ShapeDiagnostic>::Out,
            >())),
        )
    }
}

#[cfg(test)]
mod test_logsoft {
    #[test]
    fn logsoft() {
        use crate::Tensor;
        use glowstick::num::{U0, U4};
        type TestShape = glowstick::shape![U4];
        let ct = paramecia_core::Tensor::from_vec(
            vec![0f32, 1., 2., 3.],
            4,
            &paramecia_core::Device::Cpu,
        )
        .unwrap();
        let gt: Tensor<TestShape> = ct.clone().try_into().unwrap();
        let c_softmaxed: Vec<f32> = paramecia_nn::ops::log_softmax(&ct, 0)
            .unwrap()
            .to_vec1()
            .unwrap();
        let g_softmaxed: Vec<f32> = log_softmax!(gt, U0)
            .unwrap()
            .into_inner()
            .to_vec1()
            .unwrap();
        assert_eq!(c_softmaxed, g_softmaxed);
    }
}
