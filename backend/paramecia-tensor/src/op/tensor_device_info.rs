use std::marker::PhantomData;

use glowstick::Shape;
use inception::primitive;
use paramecia_arrow::{
    vis::{Graph, Vis, Visualize},
    Arrow, Combinator,
};
use paramecia_core::Device;

use crate::Tensor;

#[derive(Debug, Clone)]
pub struct TensorDeviceInfo {
    pub device: Device,
    pub is_gpu: bool,
    pub is_vulkan: bool,
}

/// Extracts runtime device metadata from a typed tensor.
pub struct TensorDeviceInfoOp<S>(PhantomData<S>);
impl<S> Default for TensorDeviceInfoOp<S> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
#[primitive(property = Arrow)]
impl<S> Combinator for TensorDeviceInfoOp<S>
where
    S: Shape + glowstick::ShapeDiagnostic,
{
    type In = Tensor<S>;
    type Out = TensorDeviceInfo;

    fn forward(&mut self, _ctx: &mut (), input: Self::In) -> Self::Out {
        let _span = crate::op::trace::forward_1!("tensor_device_info", S);
        let device = input.inner().device().clone();
        let is_vulkan = device.is_vulkan();
        let is_gpu = !device.is_cpu();
        TensorDeviceInfo {
            device,
            is_gpu,
            is_vulkan,
        }
    }
}
#[primitive(property = Visualize)]
impl<S> Vis for TensorDeviceInfoOp<S>
where
    S: Shape,
{
    fn visualize() -> Graph {
        Graph::leaf("TensorDeviceInfo")
    }
}
