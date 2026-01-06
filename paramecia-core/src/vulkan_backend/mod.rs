#![allow(dead_code)]

pub mod debug;
pub mod device;
pub use device::{InFlightBatch, PendingDownload, VulkanDevice};

mod storage;
pub use storage::VulkanStorage;

use paramecia_vulkan::VulkanKernelError;
use std::sync::{PoisonError, TryLockError};

#[derive(thiserror::Error, Debug)]
pub enum LockError {
    #[error("{0}")]
    Poisoned(String),
    #[error("Would block")]
    WouldBlock,
}

impl<T> From<TryLockError<T>> for VulkanError {
    fn from(value: TryLockError<T>) -> Self {
        match value {
            TryLockError::Poisoned(p) => VulkanError::LockError(LockError::Poisoned(p.to_string())),
            TryLockError::WouldBlock => VulkanError::LockError(LockError::WouldBlock),
        }
    }
}

impl<T> From<PoisonError<T>> for VulkanError {
    fn from(p: PoisonError<T>) -> Self {
        VulkanError::LockError(LockError::Poisoned(p.to_string()))
    }
}

#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("{0}")]
    Message(String),
    #[error("{0:?}")]
    LockError(#[from] LockError),
    #[error("Vulkan API error: {0}")]
    VkResult(#[from] ash::vk::Result),
    #[error("GPU allocator error: {0}")]
    AllocatorError(#[from] gpu_allocator::AllocationError),
}

impl From<String> for VulkanError {
    fn from(e: String) -> Self {
        VulkanError::Message(e)
    }
}

impl From<&str> for VulkanError {
    fn from(e: &str) -> Self {
        VulkanError::Message(e.to_string())
    }
}

impl From<VulkanKernelError> for VulkanError {
    fn from(e: VulkanKernelError) -> Self {
        VulkanError::Message(e.to_string())
    }
}
