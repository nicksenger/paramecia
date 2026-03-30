#![allow(dead_code)]

use crate::backend::BackendStorage;
use crate::{CpuStorage, CpuStorageRef, DType, Layout, Result, Shape, VulkanError, VulkanStorage};
use ash::vk;
use bytemuck::Pod;
use crossbeam::queue::SegQueue;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use half::bf16;
use paramecia_vulkan::Kernels;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use tracing::{debug, info, trace, warn};

static UPLOAD_COUNT: AtomicU64 = AtomicU64::new(0);
static DOWNLOAD_COUNT: AtomicU64 = AtomicU64::new(0);
static UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);
static DOWNLOAD_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static FREE_COUNT: AtomicU64 = AtomicU64::new(0);
static LAST_OP_SIZE: AtomicU64 = AtomicU64::new(0);
static FLUSH_COUNT: AtomicU64 = AtomicU64::new(0);
static FENCE_WAIT_US: AtomicU64 = AtomicU64::new(0);
static EXECUTE_ONE_TIME_COUNT: AtomicU64 = AtomicU64::new(0);
static RECORD_COMPUTE_COUNT: AtomicU64 = AtomicU64::new(0);
/// Counter for first N transfer logs (backtrace sampling)
static TRANSFER_LOG_REMAINING: AtomicU64 = AtomicU64::new(0);

fn profiling_enabled() -> bool {
    use std::sync::atomic::AtomicU8;
    // 0 = unchecked, 1 = disabled, 2 = enabled
    static CACHE: AtomicU8 = AtomicU8::new(0);
    let v = CACHE.load(Ordering::Relaxed);
    if v != 0 {
        return v == 2;
    }
    let enabled = std::env::var_os("PARAMECIA_PROFILE").is_some();
    CACHE.store(if enabled { 2 } else { 1 }, Ordering::Relaxed);
    enabled
}

thread_local! {
    static TRANSFER_LABEL: std::cell::RefCell<&'static str> = const { std::cell::RefCell::new("") };
}

/// Set a label for the next transfer(s) to identify their source.
pub fn set_transfer_label(label: &'static str) {
    TRANSFER_LABEL.with(|l| *l.borrow_mut() = label);
}

fn get_transfer_label() -> &'static str {
    TRANSFER_LABEL.with(|l| *l.borrow())
}

/// Print Vulkan profiling stats and reset counters. Call between tokens.
pub fn print_and_reset_stats() {
    let uploads = UPLOAD_COUNT.swap(0, Ordering::Relaxed);
    let downloads = DOWNLOAD_COUNT.swap(0, Ordering::Relaxed);
    let upload_kb = UPLOAD_BYTES.swap(0, Ordering::Relaxed) / 1024;
    let download_kb = DOWNLOAD_BYTES.swap(0, Ordering::Relaxed) / 1024;
    let flushes = FLUSH_COUNT.swap(0, Ordering::Relaxed);
    let fence_us = FENCE_WAIT_US.swap(0, Ordering::Relaxed);
    let eot = EXECUTE_ONE_TIME_COUNT.swap(0, Ordering::Relaxed);
    let dispatches = RECORD_COMPUTE_COUNT.swap(0, Ordering::Relaxed);
    if profiling_enabled() && uploads + downloads + flushes + eot + dispatches > 0 {
        debug!(
            "[VK PROF] up={}({} KB) down={}({} KB) flushes={} exec1t={} dispatches={} fence={:.1}ms",
            uploads,
            upload_kb,
            downloads,
            download_kb,
            flushes,
            eot,
            dispatches,
            fence_us as f64 / 1000.0
        );
    }
}

/// Enable detailed transfer logging for the next N transfers.
pub fn enable_transfer_log(count: u64) {
    TRANSFER_LOG_REMAINING.store(count, Ordering::Relaxed);
}

/// Install a signal handler that prints Vulkan transfer stats on SIGSEGV.
pub fn install_crash_handler() {
    unsafe {
        libc::signal(
            libc::SIGSEGV,
            crash_handler as *const () as libc::sighandler_t,
        );
    }
}

extern "C" fn crash_handler(sig: libc::c_int) {
    // Use write() syscall directly — signal-safe
    let msg = format!(
        "\n[VULKAN CRASH] Signal {}\n  uploads: {}, downloads: {}, allocs: {}, frees: {}, last_op_bytes: {}\n",
        sig,
        UPLOAD_COUNT.load(Ordering::Relaxed),
        DOWNLOAD_COUNT.load(Ordering::Relaxed),
        ALLOC_COUNT.load(Ordering::Relaxed),
        FREE_COUNT.load(Ordering::Relaxed),
        LAST_OP_SIZE.load(Ordering::Relaxed),
    );
    unsafe {
        libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
        // Re-raise so the OS gets the core dump
        libc::signal(libc::SIGSEGV, libc::SIG_DFL);
        libc::raise(libc::SIGSEGV);
    }
}

/// A GPU buffer with its allocation metadata.
/// Automatically frees the buffer and allocation on drop.
pub struct VulkanBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: vk::DeviceSize,
    device: Arc<ash::Device>,
    allocator: Arc<Mutex<Allocator>>,
    device_api_lock: Arc<Mutex<()>>,
}

impl std::fmt::Debug for VulkanBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBuffer")
            .field("buffer", &self.buffer)
            .field("size", &self.size)
            .finish()
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        FREE_COUNT.fetch_add(1, Ordering::Relaxed);
        if let Ok(_guard) = self.device_api_lock.lock() {
            unsafe { self.device.destroy_buffer(self.buffer, None) };
            if let Some(allocation) = self.allocation.take() {
                if let Ok(mut alloc) = self.allocator.lock() {
                    let _ = alloc.free(allocation);
                }
            }
        }
    }
}

impl VulkanBuffer {
    pub fn device_address(&self) -> vk::DeviceSize {
        self.allocation.as_ref().map(|a| a.offset()).unwrap_or(0)
    }
}

/// Batched command recording for reducing per-operation overhead.
/// Instead of creating a new command buffer, fence, and descriptor set for each operation,
/// we record multiple dispatches into a single command buffer with pipeline barriers.
#[derive(Clone, Copy)]
struct DescriptorSetRecord {
    layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
}

struct CommandBatch {
    cmd: vk::CommandBuffer,
    /// Descriptor sets used by this batch (returned to cache after completion).
    desc_sets: Vec<DescriptorSetRecord>,
    /// Staging buffers kept alive until the batch is flushed
    staging_buffers: Vec<VulkanBuffer>,
    /// Shared buffers kept alive until the batch completes (for use-after-record safety)
    keep_alive: Vec<Arc<VulkanBuffer>>,
    /// Buffers that may have been written by prior compute dispatches in this batch.
    /// Used to inject compute-to-compute barriers only for overlapping resources.
    shader_written_buffers: HashSet<vk::Buffer>,
    /// Buffers written by prior transfer operations (copies/uploads) in this batch.
    /// Used to inject transfer-to-compute or transfer-to-transfer barriers on demand.
    transfer_written_buffers: HashSet<vk::Buffer>,
}

/// Resources from a completed batch, queued for recycling on the main thread.
struct DeferredCleanup {
    cmd: vk::CommandBuffer,
    desc_sets: Vec<DescriptorSetRecord>,
    staging_buffers: Vec<VulkanBuffer>,
    keep_alive: Vec<Arc<VulkanBuffer>>,
}

// Safety: DeferredCleanup only holds Vulkan handles (u64 IDs) and VulkanBuffer
// whose cleanup uses Arc<Mutex<Allocator>> which is thread-safe.
unsafe impl Send for DeferredCleanup {}

/// A submitted batch whose GPU work may still be in-flight.
/// Owns all resources that must stay alive until the timeline semaphore signals.
/// Call `wait()` to block until completion, or let `Drop` handle it.
pub struct InFlightBatch {
    timeline_semaphore: vk::Semaphore,
    timeline_value: u64,
    cmd: vk::CommandBuffer,
    desc_sets: Vec<DescriptorSetRecord>,
    staging_buffers: Vec<VulkanBuffer>,
    keep_alive: Vec<Arc<VulkanBuffer>>,
    device: Arc<ash::Device>,
    cleanup_queue: Arc<SegQueue<DeferredCleanup>>,
    waited: bool,
}

// Safety: InFlightBatch only touches Vulkan handles via ash::Device (which is Send+Sync),
// and VulkanBuffer cleanup uses Arc<Mutex<Allocator>>. The semaphore wait is thread-safe per Vulkan spec.
unsafe impl Send for InFlightBatch {}

impl InFlightBatch {
    /// Block until this batch's GPU work completes, then defer resource recycling.
    pub fn wait(mut self) -> crate::Result<()> {
        self.wait_inner()
    }

    fn wait_inner(&mut self) -> crate::Result<()> {
        if self.waited {
            return Ok(());
        }
        self.waited = true;

        let t0 = std::time::Instant::now();
        let semaphores = [self.timeline_semaphore];
        let values = [self.timeline_value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe { self.device.wait_semaphores(&wait_info, u64::MAX) }.map_err(VulkanError::from)?;
        FENCE_WAIT_US.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);

        // Push spent resources to deferred cleanup queue (lock-free)
        self.cleanup_queue.push(DeferredCleanup {
            cmd: self.cmd,
            desc_sets: std::mem::take(&mut self.desc_sets),
            staging_buffers: std::mem::take(&mut self.staging_buffers),
            keep_alive: std::mem::take(&mut self.keep_alive),
        });
        Ok(())
    }
}

impl Drop for InFlightBatch {
    fn drop(&mut self) {
        if !self.waited {
            // Must wait before freeing resources
            let semaphores = [self.timeline_semaphore];
            let values = [self.timeline_value];
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            let _ = unsafe { self.device.wait_semaphores(&wait_info, u64::MAX) };
            self.cleanup_queue.push(DeferredCleanup {
                cmd: self.cmd,
                desc_sets: std::mem::take(&mut self.desc_sets),
                staging_buffers: std::mem::take(&mut self.staging_buffers),
                keep_alive: std::mem::take(&mut self.keep_alive),
            });
        }
    }
}

#[derive(Clone, Debug)]
pub struct CoopMatrixCapabilities {
    pub best_f32_tile: Option<(u32, u32, u32)>, // (M, N, K)
    pub best_i8_tile: Option<(u32, u32, u32)>,
}

const VENDOR_ID_NVIDIA: u32 = 0x10DE;
const VENDOR_ID_AMD: u32 = 0x1002;
const VENDOR_ID_INTEL: u32 = 0x8086;

#[derive(Clone)]
pub struct VulkanDevice {
    ordinal: usize,
    pub(crate) vendor_id: u32,
    instance: Arc<ash::Instance>,
    physical_device: vk::PhysicalDevice,
    pub(crate) device: Arc<ash::Device>,
    pub(crate) queue: vk::Queue,
    pub(crate) queue_family_index: u32,
    pub(crate) allocator: Arc<Mutex<Allocator>>,
    pub(crate) kernels: Arc<Kernels>,
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) descriptor_pool: vk::DescriptorPool,
    descriptor_pool_lock: Arc<Mutex<()>>,
    descriptor_set_cache: Arc<Mutex<HashMap<vk::DescriptorSetLayout, Vec<vk::DescriptorSet>>>>,
    pub(crate) global_seed: Arc<Mutex<u64>>,
    /// Batched command recording to reduce per-op overhead
    batch: Arc<Mutex<Option<CommandBatch>>>,
    /// Serialize Vulkan command recording on this logical device.
    /// Vulkan command pools/command buffers require external synchronization.
    record_lock: Arc<Mutex<()>>,
    /// Serialize VkQueue submissions (required by Vulkan spec for external synchronization).
    queue_lock: Arc<Mutex<()>>,
    /// Cache for scalar constant buffers (key = (dtype, raw_bytes_as_u64))
    /// Avoids repeated upload_to_buffer for the same scalar values
    scalar_cache: Arc<Mutex<std::collections::HashMap<(DType, u64), Arc<VulkanBuffer>>>>,
    /// Pool of reusable command buffers (reset before returning to pool)
    cmd_pool: Arc<Mutex<Vec<vk::CommandBuffer>>>,
    /// Lock-free queue for deferred cleanup of completed batch resources
    cleanup_queue: Arc<SegQueue<DeferredCleanup>>,
    /// Timeline semaphore for batch synchronization (replaces per-batch fences)
    timeline_semaphore: vk::Semaphore,
    /// Monotonically increasing counter for timeline semaphore signal values
    timeline_value: Arc<AtomicU64>,
    /// Cooperative matrix capabilities (Tensor Cores, WMMA, XMX)
    pub(crate) coop_matrix_props: Option<CoopMatrixCapabilities>,
    /// Integer dot product acceleration (VK_KHR_shader_integer_dot_product / Vulkan 1.3)
    pub(crate) has_integer_dot_product: bool,
    /// Native FP16 compute (shaderFloat16)
    pub(crate) has_fp16_compute: bool,
    /// Shader core count (SMs on NVIDIA, CUs on AMD), 0 = unknown
    pub(crate) shader_core_count: u32,
    /// Subgroup size properties
    pub(crate) subgroup_size: u32,
    pub(crate) subgroup_min_size: u32,
    pub(crate) subgroup_max_size: u32,
    /// Pool of reusable scratch buffers (e.g. split-K partials)
    scratch_pool: Arc<Mutex<Vec<Arc<VulkanBuffer>>>>,
    /// Pool of reusable download staging buffers (GpuToCpu)
    download_staging_pool: Arc<Mutex<Vec<VulkanBuffer>>>,
    /// Pool of reusable upload staging buffers (CpuToGpu)
    upload_staging_pool: Arc<Mutex<Vec<VulkanBuffer>>>,
    /// Map raw vk::Buffer handles to owning Arc so recorded commands can keep buffers alive.
    buffer_registry: Arc<Mutex<HashMap<vk::Buffer, Weak<VulkanBuffer>>>>,
    /// Serialize device-level Vulkan API teardown/creation calls used from multiple threads.
    device_api_lock: Arc<Mutex<()>>,
    /// Push descriptor function loader (eliminates descriptor set allocation overhead)
    push_descriptor_fn: Option<Arc<ash::khr::push_descriptor::Device>>,
    /// Dedicated transfer queue (DMA engine or second queue from compute family)
    transfer_queue: Option<vk::Queue>,
    transfer_queue_family_index: Option<u32>,
    /// Whether the transfer queue is from the same family as compute (no QFOTs needed)
    transfer_same_family: bool,
    transfer_command_pool: Option<vk::CommandPool>,
    transfer_queue_lock: Arc<Mutex<()>>,
    transfer_timeline_semaphore: Option<vk::Semaphore>,
    transfer_timeline_value: Arc<AtomicU64>,
    _entry: Arc<ash::Entry>,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("ordinal", &self.ordinal)
            .finish()
    }
}

/// A pending download handle. Records the copy into the batch but doesn't flush.
/// Call `device.flush()` then `device.complete_download(handle)` to read the result.
pub struct PendingDownload {
    staging: VulkanBuffer,
    byte_size: usize,
}

// Safety: PendingDownload only reads from a mapped staging buffer (via NonNull pointer).
// After fence wait, the mapped memory is stable and thread-safe to read from any thread.
// The VulkanBuffer cleanup uses Arc<Mutex<Allocator>> which is thread-safe.
unsafe impl Send for PendingDownload {}

impl VulkanDevice {
    pub(crate) fn register_buffer(&self, buf: &Arc<VulkanBuffer>) {
        if let Ok(mut registry) = self.buffer_registry.lock() {
            registry.insert(buf.buffer, Arc::downgrade(buf));
        }
    }

    fn acquire_descriptor_set(&self, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet> {
        {
            let mut cache = self
                .descriptor_set_cache
                .lock()
                .map_err(|e| VulkanError::Message(format!("descriptor set cache lock: {}", e)))?;
            if let Some(sets) = cache.get_mut(&layout) {
                if let Some(set) = sets.pop() {
                    return Ok(set);
                }
            }
        }

        let set_layouts = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&set_layouts);
        let _descriptor_pool_guard = self
            .descriptor_pool_lock
            .lock()
            .map_err(|e| VulkanError::Message(format!("descriptor pool lock: {}", e)))?;
        let desc_sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info) }
            .map_err(VulkanError::from)?;
        Ok(desc_sets[0])
    }

    fn keep_registered_buffers_alive(&self, handles: &[vk::Buffer]) -> Result<()> {
        let mut to_keep: Vec<Arc<VulkanBuffer>> = Vec::new();
        {
            let mut registry = self
                .buffer_registry
                .lock()
                .map_err(|e| VulkanError::Message(format!("buffer registry lock: {}", e)))?;
            for &h in handles {
                if let Some(weak) = registry.get(&h) {
                    if let Some(arc) = weak.upgrade() {
                        to_keep.push(arc);
                    } else {
                        registry.remove(&h);
                    }
                }
            }
        }

        if to_keep.is_empty() {
            return Ok(());
        }

        let mut batch_guard = self
            .batch
            .lock()
            .map_err(|e| VulkanError::Message(format!("batch lock: {}", e)))?;
        if let Some(batch) = batch_guard.as_mut() {
            batch.keep_alive.extend(to_keep);
        }
        Ok(())
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Whether a dedicated transfer queue is available for async DMA transfers.
    pub fn has_transfer_queue(&self) -> bool {
        self.transfer_queue.is_some()
    }

    /// Execute a one-time transfer on the dedicated transfer queue (if available).
    /// Falls back to the compute queue if no transfer queue exists.
    /// Waits for the compute timeline to reach `wait_value` before starting,
    /// and signals the transfer timeline when complete.
    /// Returns the transfer timeline value that was signaled.
    pub fn submit_transfer<F: FnOnce(vk::CommandBuffer)>(
        &self,
        record: F,
        wait_compute_value: Option<u64>,
    ) -> Result<u64> {
        let (queue, cmd_pool, tl_sem, queue_lock) = if let (Some(tq), Some(tcp), Some(ts)) = (
            self.transfer_queue,
            self.transfer_command_pool,
            self.transfer_timeline_semaphore,
        ) {
            (tq, tcp, ts, &self.transfer_queue_lock)
        } else {
            // Fall back to compute queue
            return self.flush().map(|_| 0);
        };

        let _record_guard = self
            .record_lock
            .lock()
            .map_err(|e| VulkanError::Message(format!("record lock: {}", e)))?;

        // Allocate a one-time command buffer from the transfer pool
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { self.device.allocate_command_buffers(&alloc_info) }
            .map_err(VulkanError::from)?[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(cmd, &begin_info) }.map_err(VulkanError::from)?;

        record(cmd);

        unsafe { self.device.end_command_buffer(cmd) }.map_err(VulkanError::from)?;

        // Signal transfer timeline
        let signal_value = self.transfer_timeline_value.fetch_add(1, Ordering::Relaxed) + 1;
        let signal_semaphores = [tl_sem];
        let signal_values = [signal_value];

        // Optionally wait on compute timeline before executing transfer
        let wait_semaphores = [self.timeline_semaphore];
        let wait_stages = [vk::PipelineStageFlags::TRANSFER];
        // wait_values[0] is consumed by wait_semaphore_values when wait is used;
        // wait_values[1] is unused but kept for layout consistency.
        let wait_values = [wait_compute_value.unwrap_or(0), signal_value];
        let mut timeline_info = if wait_compute_value.is_some() {
            vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_values[..1])
                .signal_semaphore_values(&signal_values)
        } else {
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values)
        };

        let cmds = [cmd];
        let mut submit_info = vk::SubmitInfo::default()
            .command_buffers(&cmds)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut timeline_info);

        if wait_compute_value.is_some() {
            submit_info = submit_info
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages);
        }

        {
            let _queue_guard = queue_lock
                .lock()
                .map_err(|e| VulkanError::Message(format!("transfer queue lock: {}", e)))?;
            unsafe {
                self.device
                    .queue_submit(queue, &[submit_info], vk::Fence::null())
            }
            .map_err(VulkanError::from)?;
        }

        Ok(signal_value)
    }

    /// Wait for the transfer timeline to reach a specific value.
    pub fn wait_transfer(&self, value: u64) -> Result<()> {
        if let Some(tl_sem) = self.transfer_timeline_semaphore {
            let semaphores = [tl_sem];
            let values = [value];
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            unsafe { self.device.wait_semaphores(&wait_info, u64::MAX) }
                .map_err(VulkanError::from)?;
        }
        Ok(())
    }

    /// Drain deferred cleanups — recycle command buffers, descriptor sets, and staging buffers.
    fn drain_deferred_cleanups(&self) {
        while let Some(cleanup) = self.cleanup_queue.pop() {
            // Reset and recycle command buffer
            if let Ok(_guard) = self.device_api_lock.lock() {
                unsafe {
                    let _ = self
                        .device
                        .reset_command_buffer(cleanup.cmd, vk::CommandBufferResetFlags::empty());
                }
            }
            if let Ok(mut pool) = self.cmd_pool.lock() {
                pool.push(cleanup.cmd);
            }
            // Recycle descriptor sets
            if !cleanup.desc_sets.is_empty() {
                if let Ok(mut cache) = self.descriptor_set_cache.lock() {
                    for rec in cleanup.desc_sets {
                        cache.entry(rec.layout).or_default().push(rec.set);
                    }
                }
            }
            // Recycle upload staging buffers instead of dropping (cap at 8)
            if !cleanup.staging_buffers.is_empty() {
                if let Ok(mut pool) = self.upload_staging_pool.lock() {
                    for buf in cleanup.staging_buffers {
                        if pool.len() < 8 {
                            pool.push(buf);
                        }
                        // else: drop naturally (over cap)
                    }
                }
            }
            // keep_alive dropped here
        }
    }

    /// Ensure the current batch is started; returns the command buffer.
    fn ensure_batch(&self) -> Result<vk::CommandBuffer> {
        // Recycle resources from completed batches before allocating new ones
        self.drain_deferred_cleanups();

        let mut batch_guard = self
            .batch
            .lock()
            .map_err(|e| VulkanError::Message(format!("batch lock: {}", e)))?;
        if let Some(ref b) = *batch_guard {
            return Ok(b.cmd);
        }
        // Try to reuse a command buffer from the pool, otherwise allocate new
        let cmd = if let Ok(mut pool) = self.cmd_pool.lock() {
            pool.pop()
        } else {
            None
        };
        let cmd = match cmd {
            Some(c) => c,
            None => {
                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                unsafe { self.device.allocate_command_buffers(&alloc_info) }
                    .map_err(VulkanError::from)?[0]
            }
        };
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(cmd, &begin_info) }.map_err(VulkanError::from)?;
        *batch_guard = Some(CommandBatch {
            cmd,
            desc_sets: Vec::new(),
            staging_buffers: Vec::new(),
            keep_alive: Vec::new(),
            shader_written_buffers: HashSet::new(),
            transfer_written_buffers: HashSet::new(),
        });
        Ok(cmd)
    }

    /// Record a compute dispatch into the current batch.
    ///
    /// This conservative variant assumes all bound buffers may be written.
    pub fn record_compute(
        &self,
        pipeline: &paramecia_vulkan::CachedPipeline,
        buffers: &[vk::Buffer],
        push_constants: Option<&[u8]>,
        dispatch: [u32; 3],
    ) -> Result<()> {
        self.record_compute_with_write_mask(pipeline, buffers, None, push_constants, dispatch)
    }

    /// Record a compute dispatch into the current batch with an explicit write mask.
    ///
    /// `write_mask` maps descriptor binding index -> writable buffer.
    /// If `None`, all buffers are treated as writable (conservative fallback).
    pub fn record_compute_with_write_mask(
        &self,
        pipeline: &paramecia_vulkan::CachedPipeline,
        buffers: &[vk::Buffer],
        write_mask: Option<u64>,
        push_constants: Option<&[u8]>,
        dispatch: [u32; 3],
    ) -> Result<()> {
        {
            let _record_guard = self
                .record_lock
                .lock()
                .map_err(|e| VulkanError::Message(format!("record lock: {}", e)))?;
            RECORD_COMPUTE_COUNT.fetch_add(1, Ordering::Relaxed);
            let cmd = self.ensure_batch()?;
            let dev = &self.device;

            if let Some(mask) = write_mask {
                if buffers.len() >= 64 {
                    crate::bail!(
                        "record_compute_with_write_mask: {} bindings exceed 64-bit write mask",
                        buffers.len()
                    );
                }
                if (mask >> buffers.len()) != 0 {
                    crate::bail!(
                        "record_compute_with_write_mask: write mask 0x{mask:016x} exceeds {} bindings",
                        buffers.len()
                    );
                }
            }

            // Use push descriptors if available, otherwise allocate from pool
            let use_push_desc = pipeline.push_descriptors && self.push_descriptor_fn.is_some();
            let desc_set = if use_push_desc {
                vk::DescriptorSet::null()
            } else {
                self.acquire_descriptor_set(pipeline.desc_set_layout)?
            };

            // Write buffer descriptors
            let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
                .iter()
                .map(|&b| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(b)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(desc_set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            if !use_push_desc {
                unsafe { dev.update_descriptor_sets(&writes, &[]) };
            }

            // Inject barriers only for buffers that overlap with prior writes in this batch.
            // Separate compute-written and transfer-written hazards (different source stages).
            let (compute_hazards, transfer_hazards): (Vec<vk::Buffer>, Vec<vk::Buffer>) = {
                let batch_guard = self
                    .batch
                    .lock()
                    .map_err(|e| VulkanError::Message(format!("batch lock: {}", e)))?;
                let batch = batch_guard.as_ref().ok_or_else(|| {
                    VulkanError::Message("record_compute: missing active batch".into())
                })?;
                let mut c_hazards = Vec::new();
                let mut t_hazards = Vec::new();
                for &buf in buffers {
                    if batch.shader_written_buffers.contains(&buf) {
                        c_hazards.push(buf);
                    } else if batch.transfer_written_buffers.contains(&buf) {
                        t_hazards.push(buf);
                    }
                }
                (c_hazards, t_hazards)
            };

            unsafe {
                // Compute-to-compute barriers
                if !compute_hazards.is_empty() {
                    let buffer_barriers: Vec<vk::BufferMemoryBarrier> = compute_hazards
                        .iter()
                        .map(|&buffer| {
                            vk::BufferMemoryBarrier::default()
                                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                                .dst_access_mask(
                                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                                )
                                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                .buffer(buffer)
                                .offset(0)
                                .size(vk::WHOLE_SIZE)
                        })
                        .collect();
                    dev.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &buffer_barriers,
                        &[],
                    );
                }

                // Transfer-to-compute barriers
                if !transfer_hazards.is_empty() {
                    let buffer_barriers: Vec<vk::BufferMemoryBarrier> = transfer_hazards
                        .iter()
                        .map(|&buffer| {
                            vk::BufferMemoryBarrier::default()
                                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                                .dst_access_mask(
                                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                                )
                                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                .buffer(buffer)
                                .offset(0)
                                .size(vk::WHOLE_SIZE)
                        })
                        .collect();
                    dev.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &buffer_barriers,
                        &[],
                    );
                }

                dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
                if use_push_desc {
                    // Push descriptors: write + bind in one call, no descriptor set allocation
                    self.push_descriptor_fn
                        .as_ref()
                        .unwrap()
                        .cmd_push_descriptor_set(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline.layout,
                            0,
                            &writes,
                        );
                } else {
                    dev.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline.layout,
                        0,
                        &[desc_set],
                        &[],
                    );
                }
                if let Some(pc_bytes) = push_constants {
                    dev.cmd_push_constants(
                        cmd,
                        pipeline.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        pc_bytes,
                    );
                }
                dev.cmd_dispatch(cmd, dispatch[0], dispatch[1], dispatch[2]);
            }

            let mut written_buffers = Vec::new();
            if let Some(mask) = write_mask {
                for (idx, &buf) in buffers.iter().enumerate() {
                    if (mask & (1u64 << idx)) != 0 {
                        written_buffers.push(buf);
                    }
                }
            } else {
                written_buffers.extend_from_slice(buffers);
            }

            // Track descriptor set (if allocated) and write-set for subsequent hazard checks.
            let mut batch_guard = self
                .batch
                .lock()
                .map_err(|e| VulkanError::Message(format!("batch lock: {}", e)))?;
            let batch = batch_guard.as_mut().ok_or_else(|| {
                VulkanError::Message("record_compute: missing active batch".into())
            })?;
            if !use_push_desc {
                batch.desc_sets.push(DescriptorSetRecord {
                    layout: pipeline.desc_set_layout,
                    set: desc_set,
                });
            }
            for &buf in &written_buffers {
                batch.shader_written_buffers.insert(buf);
            }
            drop(batch_guard);

            self.keep_registered_buffers_alive(buffers)?;
        }

        // When batch disabling is active, flush after every dispatch (sync execution)
        if super::debug::disable_batch() {
            self.flush()?;
        }

        Ok(())
    }

    /// Record a buffer copy with offsets into the current batch command buffer.
    /// This is async (no fence wait) - the copy will be executed on next flush.
    /// Uses hazard tracking to insert barriers only when src was previously written
    /// by compute or transfer operations in this batch.
    pub fn record_copy_offset(
        &self,
        src: vk::Buffer,
        src_offset: u64,
        dst: vk::Buffer,
        dst_offset: u64,
        size: u64,
    ) -> Result<()> {
        {
            let _record_guard = self
                .record_lock
                .lock()
                .map_err(|e| VulkanError::Message(format!("record lock: {}", e)))?;
            let cmd = self.ensure_batch()?;
            let dev = &self.device;

            // Check which prior write sources overlap with src
            let (src_compute_hazard, src_transfer_hazard) = {
                let batch_guard = self
                    .batch
                    .lock()
                    .map_err(|e| VulkanError::Message(format!("batch lock: {}", e)))?;
                let batch = batch_guard.as_ref().ok_or_else(|| {
                    VulkanError::Message("record_copy: missing active batch".into())
                })?;
                (
                    batch.shader_written_buffers.contains(&src),
                    batch.transfer_written_buffers.contains(&src),
                )
            };

            unsafe {
                // Barrier only if src was written by prior compute in this batch
                if src_compute_hazard {
                    let barrier = vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(src)
                        .offset(0)
                        .size(vk::WHOLE_SIZE);
                    dev.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[barrier],
                        &[],
                    );
                } else if src_transfer_hazard {
                    // Barrier if src was written by a prior transfer in this batch
                    let barrier = vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(src)
                        .offset(0)
                        .size(vk::WHOLE_SIZE);
                    dev.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[barrier],
                        &[],
                    );
                }

                let region = vk::BufferCopy {
                    src_offset,
                    dst_offset,
                    size,
                };
                dev.cmd_copy_buffer(cmd, src, dst, &[region]);
            }

            // Track dst as transfer-written (deferred barrier for future consumers)
            {
                let mut batch_guard = self
                    .batch
                    .lock()
                    .map_err(|e| VulkanError::Message(format!("batch lock: {}", e)))?;
                let batch = batch_guard.as_mut().ok_or_else(|| {
                    VulkanError::Message("record_copy: missing active batch".into())
                })?;
                batch.transfer_written_buffers.insert(dst);
            }

            self.keep_registered_buffers_alive(&[src, dst])?;
        }

        // When batch disabling is active, flush after every operation (sync execution)
        if super::debug::disable_batch() {
            self.flush()?;
        }

        Ok(())
    }

    /// Record a buffer copy into the current batch command buffer (from offset 0 to offset 0).
    /// This is async (no fence wait) - the copy will be executed on next flush.
    pub fn record_copy(&self, src: vk::Buffer, dst: vk::Buffer, size: u64) -> Result<()> {
        self.record_copy_offset(src, 0, dst, 0, size)
    }

    /// Submit the current batch without waiting. Returns an `InFlightBatch` handle
    /// that can be waited on later (possibly from another thread).
    /// Immediately starts a new empty batch for recording.
    /// Returns `None` if no batch was in progress.
    pub fn submit_async(&self) -> Result<Option<InFlightBatch>> {
        // Phase 1: Hold record_lock only to take the batch, then release.
        // This lets the main thread start recording the next batch immediately.
        let batch = {
            let _record_guard = self
                .record_lock
                .lock()
                .map_err(|e| VulkanError::Message(format!("record lock: {}", e)))?;
            let mut guard = self
                .batch
                .lock()
                .map_err(|e| VulkanError::Message(format!("batch lock: {}", e)))?;
            guard.take()
        };
        // record_lock released — main thread can record next batch immediately

        let batch = match batch {
            Some(b) => b,
            None => return Ok(None),
        };

        FLUSH_COUNT.fetch_add(1, Ordering::Relaxed);

        unsafe { self.device.end_command_buffer(batch.cmd) }.map_err(VulkanError::from)?;

        // Phase 3: Timeline semaphore — atomically increment and signal
        let signal_value = self.timeline_value.fetch_add(1, Ordering::Relaxed) + 1;
        let signal_semaphores = [self.timeline_semaphore];
        let signal_values = [signal_value];
        let mut timeline_info =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);

        let cmds = [batch.cmd];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&cmds)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut timeline_info);

        // Phase 1: Use queue_lock for queue submission serialization
        {
            let _queue_guard = self
                .queue_lock
                .lock()
                .map_err(|e| VulkanError::Message(format!("queue lock: {}", e)))?;
            unsafe {
                self.device
                    .queue_submit(self.queue, &[submit_info], vk::Fence::null())
            }
            .map_err(VulkanError::from)?;
        }

        Ok(Some(InFlightBatch {
            timeline_semaphore: self.timeline_semaphore,
            timeline_value: signal_value,
            cmd: batch.cmd,
            desc_sets: batch.desc_sets,
            staging_buffers: batch.staging_buffers,
            keep_alive: batch.keep_alive,
            device: self.device.clone(),
            cleanup_queue: self.cleanup_queue.clone(),
            waited: false,
        }))
    }

    /// Submit the current batch and wait for completion.
    /// No-op if no batch is in progress.
    pub fn flush(&self) -> Result<()> {
        if let Some(in_flight) = self.submit_async()? {
            in_flight.wait()
        } else {
            Ok(())
        }
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Allocate a GPU-local buffer of `size` bytes.
    pub fn allocate_buffer(&self, size: vk::DeviceSize) -> Result<VulkanBuffer> {
        if size == 0 {
            crate::bail!("Cannot allocate zero-size Vulkan buffer");
        }
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);

        // Use CONCURRENT sharing if transfer queue is on a different family,
        // allowing buffers to be used on both compute and transfer queues without QFOTs.
        let queue_family_indices: Vec<u32>;
        let mut buffer_info = vk::BufferCreateInfo::default().size(size).usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        );
        if let Some(tf_family) = self.transfer_queue_family_index {
            if !self.transfer_same_family {
                queue_family_indices = vec![self.queue_family_index, tf_family];
                buffer_info = buffer_info
                    .sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&queue_family_indices);
            } else {
                buffer_info = buffer_info.sharing_mode(vk::SharingMode::EXCLUSIVE);
            }
        } else {
            buffer_info = buffer_info.sharing_mode(vk::SharingMode::EXCLUSIVE);
        }

        let buffer =
            unsafe { self.device.create_buffer(&buffer_info, None) }.map_err(VulkanError::from)?;

        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocation = self
            .allocator
            .lock()
            .map_err(|e| VulkanError::Message(format!("allocator lock: {}", e)))?
            .allocate(&AllocationCreateDesc {
                name: "vulkan_buffer",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(VulkanError::from)?;

        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }
        .map_err(VulkanError::from)?;

        Ok(VulkanBuffer {
            buffer,
            allocation: Some(allocation),
            size,
            device: self.device.clone(),
            allocator: self.allocator.clone(),
            device_api_lock: self.device_api_lock.clone(),
        })
    }

    /// Allocate a CPU-visible staging buffer.
    /// Checks the upload staging pool first; falls back to fresh allocation.
    fn allocate_staging_buffer(&self, size: vk::DeviceSize) -> Result<VulkanBuffer> {
        // Try to reuse a pooled buffer with sufficient size
        if let Ok(mut pool) = self.upload_staging_pool.lock() {
            if let Some(idx) = pool.iter().position(|b| b.size >= size) {
                return Ok(pool.swap_remove(idx));
            }
        }

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer =
            unsafe { self.device.create_buffer(&buffer_info, None) }.map_err(VulkanError::from)?;

        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocation = self
            .allocator
            .lock()
            .map_err(|e| VulkanError::Message(format!("allocator lock: {}", e)))?
            .allocate(&AllocationCreateDesc {
                name: "staging_buffer",
                requirements,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(VulkanError::from)?;

        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }
        .map_err(VulkanError::from)?;

        Ok(VulkanBuffer {
            buffer,
            allocation: Some(allocation),
            size,
            device: self.device.clone(),
            allocator: self.allocator.clone(),
            device_api_lock: self.device_api_lock.clone(),
        })
    }

    /// Allocate a CPU-readable download buffer.
    /// Checks the download staging pool first; falls back to fresh allocation.
    fn allocate_download_buffer(&self, size: vk::DeviceSize) -> Result<VulkanBuffer> {
        // Try to reuse a pooled buffer with sufficient size
        if let Ok(mut pool) = self.download_staging_pool.lock() {
            if let Some(idx) = pool.iter().position(|b| b.size >= size) {
                return Ok(pool.swap_remove(idx));
            }
        }

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer =
            unsafe { self.device.create_buffer(&buffer_info, None) }.map_err(VulkanError::from)?;

        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocation = self
            .allocator
            .lock()
            .map_err(|e| VulkanError::Message(format!("allocator lock: {}", e)))?
            .allocate(&AllocationCreateDesc {
                name: "download_buffer",
                requirements,
                location: MemoryLocation::GpuToCpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(VulkanError::from)?;

        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }
        .map_err(VulkanError::from)?;

        Ok(VulkanBuffer {
            buffer,
            allocation: Some(allocation),
            size,
            device: self.device.clone(),
            allocator: self.allocator.clone(),
            device_api_lock: self.device_api_lock.clone(),
        })
    }

    /// Free a VulkanBuffer (Drop handles the actual cleanup).
    pub fn free_buffer(&self, _vbuf: VulkanBuffer) -> Result<()> {
        // VulkanBuffer::drop handles destroy_buffer + allocator.free
        Ok(())
    }

    /// Validate upload/download roundtrip. Call at startup to verify data transfer.
    pub fn validate_transfer(&self) -> Result<()> {
        // Test with f32
        let test_f32: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        let storage = self.allocate_with_data(DType::F32, &test_f32)?;
        let buf = storage
            .buffer()
            .ok_or_else(|| crate::Error::Msg("no buffer".into()))?;
        let roundtrip: Vec<f32> = self.download_buffer(buf, 256)?;
        for (i, (a, b)) in test_f32.iter().zip(roundtrip.iter()).enumerate() {
            if a != b {
                crate::bail!(
                    "Vulkan f32 transfer validation failed at index {}: uploaded {} but downloaded {}",
                    i,
                    a,
                    b
                );
            }
        }
        // Test with bf16
        let test_bf16: Vec<bf16> = (0..256).map(|i| bf16::from_f32(i as f32 * 0.1)).collect();
        let storage = self.allocate_with_data(DType::BF16, &test_bf16)?;
        let buf = storage
            .buffer()
            .ok_or_else(|| crate::Error::Msg("no buffer".into()))?;
        let roundtrip: Vec<bf16> = self.download_buffer(buf, 256)?;
        for (i, (a, b)) in test_bf16.iter().zip(roundtrip.iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                crate::bail!(
                    "Vulkan bf16 transfer validation failed at index {}: uploaded {:?} but downloaded {:?}",
                    i,
                    a,
                    b
                );
            }
        }
        //eprintln!("[vulkan] Transfer validation passed (f32 + bf16 roundtrip OK)");
        Ok(())
    }

    /// Upload data to a GPU buffer via staging. Flushes any pending command batch first.
    pub fn upload_to_buffer<T: Pod>(&self, data: &[T], dst: &VulkanBuffer) -> Result<()> {
        let byte_size = std::mem::size_of_val(data) as vk::DeviceSize;
        if byte_size == 0 {
            return Ok(());
        }
        UPLOAD_COUNT.fetch_add(1, Ordering::Relaxed);
        UPLOAD_BYTES.fetch_add(byte_size, Ordering::Relaxed);
        LAST_OP_SIZE.store(byte_size, Ordering::Relaxed);
        if profiling_enabled() && TRANSFER_LOG_REMAINING.load(Ordering::Relaxed) > 0 {
            TRANSFER_LOG_REMAINING.fetch_sub(1, Ordering::Relaxed);
            let label = get_transfer_label();
            trace!("[VK XFER] UPLOAD {} bytes [{}]", byte_size, label);
        }

        let staging = self.allocate_staging_buffer(byte_size)?;

        // Copy data to staging buffer (CPU-side memcpy)
        let mapped = staging.allocation.as_ref().and_then(|a| a.mapped_ptr());
        if let Some(mapped) = mapped {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    mapped.as_ptr() as *mut u8,
                    byte_size as usize,
                );
            }
        } else {
            self.free_buffer(staging)?;
            crate::bail!("Failed to map staging buffer");
        }

        {
            let _record_guard = self
                .record_lock
                .lock()
                .map_err(|e| VulkanError::Message(format!("record lock: {}", e)))?;

            // Record copy into the batch command buffer (async)
            let cmd = self.ensure_batch()?;

            // Check if dst was previously written by compute or transfer in this batch
            let (dst_compute_hazard, dst_transfer_hazard) = {
                let batch_guard = self
                    .batch
                    .lock()
                    .map_err(|e| VulkanError::Message(format!("batch lock: {}", e)))?;
                let batch = batch_guard.as_ref().ok_or_else(|| {
                    VulkanError::Message("upload_to_buffer: missing active batch".into())
                })?;
                (
                    batch.shader_written_buffers.contains(&dst.buffer),
                    batch.transfer_written_buffers.contains(&dst.buffer),
                )
            };

            unsafe {
                // Barrier only if dst has prior writes in this batch
                if dst_compute_hazard {
                    let barrier = vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(dst.buffer)
                        .offset(0)
                        .size(vk::WHOLE_SIZE);
                    self.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[barrier],
                        &[],
                    );
                } else if dst_transfer_hazard {
                    let barrier = vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(dst.buffer)
                        .offset(0)
                        .size(vk::WHOLE_SIZE);
                    self.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[barrier],
                        &[],
                    );
                }

                let region = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: byte_size,
                };
                self.device
                    .cmd_copy_buffer(cmd, staging.buffer, dst.buffer, &[region]);
            }

            // Track dst as transfer-written and keep staging alive
            {
                let mut batch_guard = self
                    .batch
                    .lock()
                    .map_err(|e| VulkanError::Message(format!("batch lock: {}", e)))?;
                let batch = batch_guard.as_mut().unwrap();
                batch.staging_buffers.push(staging);
                batch.transfer_written_buffers.insert(dst.buffer);
            }
        }

        // When batch disabling is active, flush after every operation (sync execution)
        if super::debug::disable_batch() {
            self.flush()?;
        }

        Ok(())
    }

    /// Keep a buffer alive until the current batch completes.
    /// Used to prevent use-after-free when a buffer is recorded into a dispatch
    /// but owned elsewhere (e.g. temporary QStorage from get_kv_storage).
    pub fn keep_buffer_alive(&self, buf: Arc<VulkanBuffer>) -> Result<()> {
        let mut batch_guard = self
            .batch
            .lock()
            .map_err(|e| VulkanError::Message(format!("batch lock: {}", e)))?;
        if let Some(batch) = batch_guard.as_mut() {
            batch.keep_alive.push(buf);
        }
        // If no batch is active, the buffer will be kept alive by the caller's Arc
        Ok(())
    }

    /// Acquire a scratch buffer of at least `min_bytes` from the pool, or allocate a new one.
    /// The returned buffer is removed from the pool. Call `release_scratch_buffer` to return it.
    /// Also adds the buffer to the current batch's keep_alive list.
    pub fn acquire_scratch_buffer(&self, min_bytes: u64) -> Result<Arc<VulkanBuffer>> {
        let mut pool = self
            .scratch_pool
            .lock()
            .map_err(|e| VulkanError::Message(format!("scratch pool lock: {}", e)))?;

        // Find the smallest buffer that fits (best-fit to reduce waste)
        let best_idx = pool
            .iter()
            .enumerate()
            .filter(|(_, b)| b.size >= min_bytes)
            .min_by_key(|(_, b)| b.size)
            .map(|(i, _)| i);

        let buf = if let Some(idx) = best_idx {
            pool.swap_remove(idx)
        } else {
            drop(pool); // release lock before allocating
            Arc::new(self.allocate_buffer(min_bytes)?)
        };

        // Keep alive in the current batch so it survives until GPU completes
        self.keep_buffer_alive(buf.clone())?;

        Ok(buf)
    }

    /// Return a scratch buffer to the pool for reuse.
    pub fn release_scratch_buffer(&self, buf: Arc<VulkanBuffer>) {
        if let Ok(mut pool) = self.scratch_pool.lock() {
            // Cap pool size to prevent unbounded growth
            if pool.len() < 16 {
                pool.push(buf);
            }
            // else: drop the buffer (Arc refcount decreases)
        }
    }

    /// Download data from a GPU buffer. Records copy into batch and flushes.
    pub fn download_buffer<T: Pod + Default + Clone>(
        &self,
        src: &VulkanBuffer,
        count: usize,
    ) -> Result<Vec<T>> {
        let byte_size = (count * std::mem::size_of::<T>()) as vk::DeviceSize;
        if byte_size == 0 {
            return Ok(vec![]);
        }
        DOWNLOAD_COUNT.fetch_add(1, Ordering::Relaxed);
        DOWNLOAD_BYTES.fetch_add(byte_size, Ordering::Relaxed);
        LAST_OP_SIZE.store(byte_size, Ordering::Relaxed);
        if profiling_enabled() && TRANSFER_LOG_REMAINING.load(Ordering::Relaxed) > 0 {
            TRANSFER_LOG_REMAINING.fetch_sub(1, Ordering::Relaxed);
            let label = get_transfer_label();
            trace!("[VK XFER] DOWNLOAD {} bytes [{}]", byte_size, label);
        }

        // Validate: requested size must not exceed source buffer
        if byte_size > src.size {
            crate::bail!(
                "Vulkan download_buffer: requested {} bytes (count={} * {}B) but src buffer only has {} bytes",
                byte_size,
                count,
                std::mem::size_of::<T>(),
                src.size
            );
        }

        let download = self.allocate_download_buffer(byte_size)?;

        {
            let _record_guard = self
                .record_lock
                .lock()
                .map_err(|e| VulkanError::Message(format!("record lock: {}", e)))?;

            // Record copy into the batch (instead of separate execute_one_time)
            let cmd = self.ensure_batch()?;

            // Check if src was written by prior compute or transfer in this batch
            let (src_compute_hazard, src_transfer_hazard) = {
                let batch_guard = self
                    .batch
                    .lock()
                    .map_err(|e| VulkanError::Message(format!("batch lock: {}", e)))?;
                let batch = batch_guard.as_ref().ok_or_else(|| {
                    VulkanError::Message("download_buffer: missing active batch".into())
                })?;
                (
                    batch.shader_written_buffers.contains(&src.buffer),
                    batch.transfer_written_buffers.contains(&src.buffer),
                )
            };

            unsafe {
                if src_compute_hazard {
                    let barrier = vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(src.buffer)
                        .offset(0)
                        .size(vk::WHOLE_SIZE);
                    self.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[barrier],
                        &[],
                    );
                } else if src_transfer_hazard {
                    let barrier = vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(src.buffer)
                        .offset(0)
                        .size(vk::WHOLE_SIZE);
                    self.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[barrier],
                        &[],
                    );
                }

                let region = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: byte_size,
                };
                self.device
                    .cmd_copy_buffer(cmd, src.buffer, download.buffer, &[region]);
            }
        }

        // Flush to execute all pending operations including this copy
        self.flush()?;

        let mut result = vec![T::default(); count];
        let mapped = download.allocation.as_ref().and_then(|a| a.mapped_ptr());
        if let Some(mapped) = mapped {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mapped.as_ptr() as *const u8,
                    result.as_mut_ptr() as *mut u8,
                    byte_size as usize,
                );
            }
        } else {
            self.free_buffer(download)?;
            crate::bail!("Failed to map download buffer");
        }

        self.free_buffer(download)?;
        Ok(result)
    }

    /// Record a download copy into the batch without flushing.
    /// Returns a handle that can be read after a subsequent `flush()`.
    pub fn prepare_download(
        &self,
        src: Arc<VulkanBuffer>,
        byte_size: u64,
    ) -> Result<PendingDownload> {
        if byte_size == 0 {
            crate::bail!("Cannot prepare zero-size download");
        }
        DOWNLOAD_COUNT.fetch_add(1, Ordering::Relaxed);
        DOWNLOAD_BYTES.fetch_add(byte_size, Ordering::Relaxed);
        LAST_OP_SIZE.store(byte_size, Ordering::Relaxed);
        if profiling_enabled() && TRANSFER_LOG_REMAINING.load(Ordering::Relaxed) > 0 {
            TRANSFER_LOG_REMAINING.fetch_sub(1, Ordering::Relaxed);
            let label = get_transfer_label();
            trace!("[VK XFER] DOWNLOAD(prep) {} bytes [{}]", byte_size, label);
        }

        let _record_guard = self
            .record_lock
            .lock()
            .map_err(|e| VulkanError::Message(format!("record lock: {}", e)))?;

        let download = self.allocate_download_buffer(byte_size)?;

        let cmd = self.ensure_batch()?;

        // Check if src was written by prior compute or transfer in this batch
        let (src_compute_hazard, src_transfer_hazard) = {
            let batch_guard = self
                .batch
                .lock()
                .map_err(|e| VulkanError::Message(format!("batch lock: {}", e)))?;
            let batch = batch_guard.as_ref().ok_or_else(|| {
                VulkanError::Message("prepare_download: missing active batch".into())
            })?;
            (
                batch.shader_written_buffers.contains(&src.buffer),
                batch.transfer_written_buffers.contains(&src.buffer),
            )
        };

        unsafe {
            // Barrier only if src has prior writes in this batch
            if src_compute_hazard {
                let barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(src.buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[barrier],
                    &[],
                );
            } else if src_transfer_hazard {
                let barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(src.buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[barrier],
                    &[],
                );
            }

            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: byte_size,
            };
            self.device
                .cmd_copy_buffer(cmd, src.buffer, download.buffer, &[region]);
        }

        self.batch
            .lock()
            .map_err(|e| VulkanError::Message(format!("batch lock: {}", e)))?
            .as_mut()
            .ok_or_else(|| VulkanError::Message("prepare_download: missing active batch".into()))?
            .keep_alive
            .push(src);

        Ok(PendingDownload {
            staging: download,
            byte_size: byte_size as usize,
        })
    }

    /// Read data from a completed pending download. Must call `flush()` first.
    pub fn complete_download<T: Pod + Default + Clone>(
        &self,
        pending: PendingDownload,
        count: usize,
    ) -> Result<Vec<T>> {
        let expected_bytes = count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| VulkanError::Message("complete_download size overflow".into()))?;
        if expected_bytes != pending.byte_size {
            crate::bail!(
                "complete_download size mismatch: expected {} bytes (count={} * {}B) but pending has {} bytes",
                expected_bytes,
                count,
                std::mem::size_of::<T>(),
                pending.byte_size
            );
        }

        let mut result = vec![T::default(); count];
        let mapped = pending
            .staging
            .allocation
            .as_ref()
            .and_then(|a| a.mapped_ptr());
        if let Some(mapped) = mapped {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mapped.as_ptr() as *const u8,
                    result.as_mut_ptr() as *mut u8,
                    pending.byte_size,
                );
            }
        } else {
            crate::bail!("Failed to map download buffer");
        }
        // Return staging buffer to pool instead of dropping (cap at 8)
        if let Ok(mut pool) = self.download_staging_pool.lock() {
            if pool.len() < 8 {
                pool.push(pending.staging);
            }
            // else: drop naturally (over cap)
        }
        Ok(result)
    }

    /// Execute a one-time command buffer synchronously.
    /// Flushes any pending command batch first to ensure correct ordering.
    pub fn execute_one_time<F: FnOnce(vk::CommandBuffer)>(&self, record: F) -> Result<()> {
        EXECUTE_ONE_TIME_COUNT.fetch_add(1, Ordering::Relaxed);
        self.flush()?;
        let _record_guard = self
            .record_lock
            .lock()
            .map_err(|e| VulkanError::Message(format!("record lock: {}", e)))?;
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = unsafe { self.device.allocate_command_buffers(&alloc_info) }
            .map_err(VulkanError::from)?[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { self.device.begin_command_buffer(cmd, &begin_info) }.map_err(VulkanError::from)?;

        record(cmd);

        unsafe { self.device.end_command_buffer(cmd) }.map_err(VulkanError::from)?;

        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmds);
        let fence_info = vk::FenceCreateInfo::default();
        let fence =
            unsafe { self.device.create_fence(&fence_info, None) }.map_err(VulkanError::from)?;

        {
            let _queue_guard = self
                .queue_lock
                .lock()
                .map_err(|e| VulkanError::Message(format!("queue lock: {}", e)))?;
            unsafe { self.device.queue_submit(self.queue, &[submit_info], fence) }
                .map_err(VulkanError::from)?;
        }

        unsafe { self.device.wait_for_fences(&[fence], true, u64::MAX) }
            .map_err(VulkanError::from)?;

        unsafe {
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &cmds);
        }

        Ok(())
    }

    /// Allocate and upload data, returning a VulkanStorage.
    fn allocate_with_data<T: Pod + Default + Clone + 'static>(
        &self,
        dtype: DType,
        data: &[T],
    ) -> Result<VulkanStorage> {
        let count = data.len();
        let byte_size = (count * std::mem::size_of::<T>()) as vk::DeviceSize;

        if byte_size == 0 {
            return Ok(VulkanStorage::new(None, self.clone(), 0, dtype));
        }

        // For single-element (scalar) data, check the cache to avoid repeated uploads
        if count == 1 && std::mem::size_of::<T>() <= 8 {
            let mut key_bytes = 0u64;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    &mut key_bytes as *mut u64 as *mut u8,
                    std::mem::size_of::<T>(),
                );
            }
            let cache_key = (dtype, key_bytes);
            if let Ok(cache) = self.scalar_cache.lock() {
                if let Some(buf) = cache.get(&cache_key) {
                    return Ok(VulkanStorage::new(
                        Some(buf.clone()),
                        self.clone(),
                        count,
                        dtype,
                    ));
                }
            }
            // Not cached, allocate and upload, then cache
            let buffer = self.allocate_buffer(byte_size)?;
            self.upload_to_buffer(data, &buffer)?;
            let arc_buf = Arc::new(buffer);
            if let Ok(mut cache) = self.scalar_cache.lock() {
                cache.insert(cache_key, arc_buf.clone());
            }
            return Ok(VulkanStorage::new(
                Some(arc_buf),
                self.clone(),
                count,
                dtype,
            ));
        }

        let buffer = self.allocate_buffer(byte_size)?;
        self.upload_to_buffer(data, &buffer)?;

        Ok(VulkanStorage::new(
            Some(Arc::new(buffer)),
            self.clone(),
            count,
            dtype,
        ))
    }

    /// Allocate and fill with a value, returning a VulkanStorage.
    fn allocate_filled<T: Pod + Default + Clone + 'static>(
        &self,
        count: usize,
        dtype: DType,
        value: T,
    ) -> Result<VulkanStorage> {
        let data: Vec<T> = vec![value; count];
        self.allocate_with_data(dtype, &data)
    }

    /// Check if cooperative matrix extension is available
    pub fn has_cooperative_matrix(&self) -> bool {
        self.coop_matrix_props.is_some()
    }

    /// Get the best tile size for cooperative matrix operations (M, N, K)
    pub fn coop_matrix_tile_size(&self) -> Option<(u32, u32, u32)> {
        self.coop_matrix_props
            .as_ref()
            .and_then(|c| c.best_f32_tile)
    }

    /// Check if integer dot product extension is available and accelerated
    pub fn has_integer_dot_product(&self) -> bool {
        self.has_integer_dot_product
    }

    /// PCI vendor ID for the selected physical device.
    pub fn vendor_id(&self) -> u32 {
        self.vendor_id
    }

    /// Check if native FP16 compute (shaderFloat16) is available
    pub fn has_fp16_compute(&self) -> bool {
        self.has_fp16_compute
    }

    /// Get the shader core count (SMs on NVIDIA, CUs on AMD), 0 = unknown
    pub fn shader_core_count(&self) -> u32 {
        self.shader_core_count
    }

    /// Get the device's subgroup size
    pub fn subgroup_size(&self) -> u32 {
        self.subgroup_size
    }

    fn vendor_supports_coop_matrix(vendor_id: u32) -> bool {
        matches!(
            vendor_id,
            VENDOR_ID_NVIDIA | VENDOR_ID_AMD | VENDOR_ID_INTEL
        )
    }

    fn vendor_name(vendor_id: u32) -> &'static str {
        match vendor_id {
            VENDOR_ID_NVIDIA => "NVIDIA",
            VENDOR_ID_AMD => "AMD",
            VENDOR_ID_INTEL => "Intel",
            _ => "Unknown",
        }
    }

    /// Query cooperative matrix capabilities from a physical device
    fn query_coop_matrix_capabilities(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<CoopMatrixCapabilities> {
        let coop_matrix_fn = ash::khr::cooperative_matrix::Instance::new(entry, instance);

        // Query properties using the ash API
        let properties = unsafe {
            coop_matrix_fn
                .get_physical_device_cooperative_matrix_properties(physical_device)
                .map_err(VulkanError::from)?
        };

        // Find best tile sizes that match shader expectations.
        let best_f32_tile = Self::find_best_tile(&properties, vk::ComponentTypeKHR::FLOAT32);
        let best_i8_tile = Self::find_best_i8_tile(&properties);

        Ok(CoopMatrixCapabilities {
            best_f32_tile,
            best_i8_tile,
        })
    }

    /// Find the best cooperative matrix tile size for a given accumulator type
    fn find_best_tile(
        props: &[vk::CooperativeMatrixPropertiesKHR],
        acc_type: vk::ComponentTypeKHR,
    ) -> Option<(u32, u32, u32)> {
        props
            .iter()
            .filter(|p| p.scope == vk::ScopeKHR::SUBGROUP)
            .filter(|p| p.a_type == vk::ComponentTypeKHR::FLOAT16)
            .filter(|p| p.b_type == vk::ComponentTypeKHR::FLOAT16)
            .filter(|p| p.c_type == acc_type && p.result_type == acc_type)
            .filter(|p| p.saturating_accumulation == vk::FALSE)
            .filter(|p| p.m_size > 0 && p.n_size > 0 && p.k_size > 0)
            .filter(|p| p.m_size <= 32 && p.n_size <= 32 && p.k_size <= 32)
            .max_by_key(|p| {
                let is_16x16x16 = if p.m_size == 16 && p.n_size == 16 && p.k_size == 16 {
                    1u32
                } else {
                    0u32
                };
                let k_is_16_multiple = if p.k_size % 16 == 0 { 1u32 } else { 0u32 };
                let is_square = if p.m_size == p.n_size { 1u32 } else { 0u32 };
                (is_16x16x16 * 1_000_000)
                    + (k_is_16_multiple * 100_000)
                    + (is_square * 10_000)
                    + (p.m_size * p.n_size * p.k_size)
            })
            .map(|p| (p.m_size, p.n_size, p.k_size))
    }

    fn find_best_i8_tile(props: &[vk::CooperativeMatrixPropertiesKHR]) -> Option<(u32, u32, u32)> {
        props
            .iter()
            .filter(|p| p.scope == vk::ScopeKHR::SUBGROUP)
            .filter(|p| p.a_type == vk::ComponentTypeKHR::SINT8)
            .filter(|p| p.b_type == vk::ComponentTypeKHR::SINT8)
            .filter(|p| p.c_type == vk::ComponentTypeKHR::SINT32)
            .filter(|p| p.result_type == vk::ComponentTypeKHR::SINT32)
            .filter(|p| p.saturating_accumulation == vk::FALSE)
            .filter(|p| p.m_size > 0 && p.n_size > 0 && p.k_size > 0)
            .max_by_key(|p| p.m_size * p.n_size * p.k_size)
            .map(|p| (p.m_size, p.n_size, p.k_size))
    }
}

impl crate::backend::BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let entry = Arc::new(
            unsafe { ash::Entry::load() }
                .map_err(|e| VulkanError::Message(format!("Failed to load Vulkan: {}", e)))?,
        );

        let app_info = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));
        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = Arc::new(
            unsafe { entry.create_instance(&instance_info, None) }.map_err(VulkanError::from)?,
        );

        let physical_devices =
            unsafe { instance.enumerate_physical_devices() }.map_err(VulkanError::from)?;
        let physical_device = *physical_devices.get(ordinal).ok_or_else(|| {
            VulkanError::Message(format!(
                "Vulkan GPU ordinal {} out of range (found {})",
                ordinal,
                physical_devices.len()
            ))
        })?;

        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let vendor_id = props.vendor_id;
        info!(
            "Vulkan device {}: {} (vendor=0x{:04x} {}, API {}.{}.{})",
            ordinal,
            unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }.to_string_lossy(),
            vendor_id,
            Self::vendor_name(vendor_id),
            vk::api_version_major(props.api_version),
            vk::api_version_minor(props.api_version),
            vk::api_version_patch(props.api_version),
        );

        // Find compute queue family
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let queue_family_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| VulkanError::Message("No compute queue family".to_string()))?;

        // Look for a transfer queue: prefer a second queue from the same family
        // (avoids queue family ownership transfers), fall back to a dedicated
        // transfer-only family (DMA engine).
        let compute_family_queue_count = queue_families[queue_family_index as usize].queue_count;
        let transfer_queue_config = if compute_family_queue_count >= 2 {
            // Same family, second queue index — no QFOTs needed
            Some((queue_family_index, 1u32, true))
        } else {
            // Look for a dedicated transfer family (TRANSFER but not COMPUTE/GRAPHICS)
            queue_families
                .iter()
                .enumerate()
                .find(|(i, props)| {
                    *i as u32 != queue_family_index
                        && props.queue_flags.contains(vk::QueueFlags::TRANSFER)
                        && !props.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && !props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                })
                .map(|(i, _)| (i as u32, 0u32, false))
        };

        let queue_priorities_two = [1.0f32, 0.5f32];
        let queue_priority = [1.0f32];

        // Build queue create infos based on whether we have a transfer queue
        let queue_create_info = if let Some((_tf_family, _, same_family)) = transfer_queue_config {
            if same_family {
                // Request 2 queues from the same family
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .queue_priorities(&queue_priorities_two)
            } else {
                // Just the compute queue; transfer queue from different family added separately
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .queue_priorities(&queue_priority)
            }
        } else {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priority)
        };

        // If transfer queue is from a different family, create a separate queue create info
        let transfer_queue_create_info = transfer_queue_config
            .filter(|(_, _, same_family)| !same_family)
            .map(|(tf_family, _, _)| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(tf_family)
                    .queue_priorities(&queue_priority)
            });

        // Check optional device extensions
        let coop_matrix_ext = c"VK_KHR_cooperative_matrix";
        let idp_ext = c"VK_KHR_shader_integer_dot_product";
        let ext_properties = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .map_err(VulkanError::from)?
        };
        let has_coop_matrix = ext_properties.iter().any(|ext| {
            let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
            name == coop_matrix_ext
        });
        let has_idp_extension = ext_properties.iter().any(|ext| {
            let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
            name == idp_ext
        });
        let push_desc_ext = c"VK_KHR_push_descriptor";
        let has_push_descriptor = ext_properties.iter().any(|ext| {
            let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
            name == push_desc_ext
        }) && std::env::var("PARAMECIA_DISABLE_PUSH_DESCRIPTOR").is_err();

        let disabled = std::env::var("PARAMECIA_DISABLE_COOP_MATRIX").is_ok();

        // ---- Query Vulkan 1.3 properties (IDP, subgroup size control) ----
        let mut vk13_props = vk::PhysicalDeviceVulkan13Properties::default();
        let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut vk13_props)
            .push_next(&mut subgroup_props);
        unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };

        // Integer dot product: check if 4x8-bit packed signed is hardware-accelerated
        let idp_accelerated =
            vk13_props.integer_dot_product4x8_bit_packed_signed_accelerated == vk::TRUE;

        // Subgroup size properties
        let subgroup_size = subgroup_props.subgroup_size;
        let subgroup_min_size = vk13_props.min_subgroup_size;
        let subgroup_max_size = vk13_props.max_subgroup_size;

        // Feature query: FP16 compute + shader integer dot product
        let mut query_f16_int8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut query_idp = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
        let mut query_features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut query_f16_int8)
            .push_next(&mut query_idp);
        unsafe { instance.get_physical_device_features2(physical_device, &mut query_features2) };
        let fp16_supported = query_f16_int8.shader_float16 == vk::TRUE;
        let has_fp16_compute =
            fp16_supported && std::env::var("PARAMECIA_DISABLE_FP16_COMPUTE").is_err();
        let idp_feature_supported = query_idp.shader_integer_dot_product == vk::TRUE;
        let has_integer_dot_product = idp_accelerated
            && idp_feature_supported
            && std::env::var("PARAMECIA_DISABLE_IDP").is_err();

        // ---- Shader core count (vendor-specific extensions) ----
        let has_nv_sm = ext_properties.iter().any(|ext| {
            let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
            name == c"VK_NV_shader_sm_builtins"
        });
        let has_amd_core2 = ext_properties.iter().any(|ext| {
            let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
            name == c"VK_AMD_shader_core_properties2"
        });
        let shader_core_count =
            if has_nv_sm && std::env::var("PARAMECIA_DISABLE_CORE_COUNT").is_err() {
                let mut sm_props = vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV::default();
                let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sm_props);
                unsafe { instance.get_physical_device_properties2(physical_device, &mut p2) };
                sm_props.shader_sm_count
            } else if has_amd_core2 && std::env::var("PARAMECIA_DISABLE_CORE_COUNT").is_err() {
                let mut amd_props = vk::PhysicalDeviceShaderCoreProperties2AMD::default();
                let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut amd_props);
                unsafe { instance.get_physical_device_properties2(physical_device, &mut p2) };
                amd_props.active_compute_unit_count
            } else {
                0
            };

        //eprintln!(
        //    "[vulkan] Capabilities: IDP={}, FP16_compute={}, cores={}, subgroup={}(min={},max={})",
        //    has_integer_dot_product,
        //    has_fp16_compute,
        //    shader_core_count,
        //    subgroup_size,
        //    subgroup_min_size,
        //    subgroup_max_size,
        //);
        if subgroup_size != 32 {
            //eprintln!(
            //    "[vulkan] WARNING: subgroup size is {} (expected 32), some shaders may produce incorrect results",
            //    subgroup_size
            //);
        }

        let vendor_allows_coopmat = Self::vendor_supports_coop_matrix(vendor_id);
        let subgroup_allows_coopmat = subgroup_size == 32;
        let enable_coop_matrix =
            has_coop_matrix && !disabled && vendor_allows_coopmat && subgroup_allows_coopmat;

        // Enable required features (consolidated into Vulkan 1.2 feature struct)
        let mut features_v12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_int8(true)
            .shader_float16(has_fp16_compute)
            .uniform_and_storage_buffer8_bit_access(true)
            .storage_buffer8_bit_access(true)
            .timeline_semaphore(true);
        let mut features_idp = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
            .shader_integer_dot_product(has_integer_dot_product);

        // Enable cooperative matrix feature if extension is available
        let mut coop_matrix_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(enable_coop_matrix);

        let mut features2 =
            vk::PhysicalDeviceFeatures2::default().features(vk::PhysicalDeviceFeatures {
                shader_int16: vk::TRUE,
                shader_int64: vk::TRUE,
                shader_float64: vk::FALSE,
                ..Default::default()
            });
        features2 = features2.push_next(&mut features_v12);
        features2 = features2.push_next(&mut features_idp);
        if enable_coop_matrix {
            features2 = features2.push_next(&mut coop_matrix_features);
        }

        // Build enabled extensions list
        let mut enabled_extensions = vec![];
        if has_integer_dot_product && has_idp_extension {
            enabled_extensions.push(idp_ext.as_ptr());
        }
        if enable_coop_matrix {
            enabled_extensions.push(coop_matrix_ext.as_ptr());
        }
        if has_push_descriptor {
            enabled_extensions.push(push_desc_ext.as_ptr());
        }

        let queue_create_infos_vec: Vec<vk::DeviceQueueCreateInfo> =
            if let Some(tf_qci) = transfer_queue_create_info {
                vec![queue_create_info, tf_qci]
            } else {
                vec![queue_create_info]
            };
        let mut device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos_vec)
            .push_next(&mut features2);

        if !enabled_extensions.is_empty() {
            device_create_info = device_create_info.enabled_extension_names(&enabled_extensions);
        }

        let device = Arc::new(
            unsafe { instance.create_device(physical_device, &device_create_info, None) }
                .map_err(VulkanError::from)?,
        );

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // Retrieve transfer queue if available
        let (transfer_queue, transfer_queue_family_index, transfer_same_family) =
            if let Some((tf_family, tf_index, same_family)) = transfer_queue_config {
                let tq = unsafe { device.get_device_queue(tf_family, tf_index) };
                info!(
                    "Transfer queue: family={}, index={}, same_family={}",
                    tf_family, tf_index, same_family
                );
                (Some(tq), Some(tf_family), same_family)
            } else {
                debug!("No dedicated transfer queue available");
                (None, None, false)
            };

        // Create memory allocator
        let allocator = Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: (*instance).clone(),
            device: (*device).clone(),
            physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
            buffer_device_address: false,
            allocation_sizes: gpu_allocator::AllocationSizes::default(),
        })
        .map_err(|e| VulkanError::Message(format!("Failed to create allocator: {}", e)))?;

        // Create timeline semaphore for batch synchronization (core in Vulkan 1.3)
        let mut timeline_type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_type_info);
        let timeline_semaphore =
            unsafe { device.create_semaphore(&sem_info, None) }.map_err(VulkanError::from)?;

        // Create command pool
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool =
            unsafe { device.create_command_pool(&pool_info, None) }.map_err(VulkanError::from)?;

        // Create transfer command pool and timeline semaphore if transfer queue is available
        let transfer_command_pool = if let Some(tf_family) = transfer_queue_family_index {
            let pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(tf_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            Some(
                unsafe { device.create_command_pool(&pool_info, None) }
                    .map_err(VulkanError::from)?,
            )
        } else {
            None
        };
        let transfer_timeline_semaphore = if transfer_queue.is_some() {
            let mut tl_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut tl_info);
            Some(unsafe { device.create_semaphore(&sem_info, None) }.map_err(VulkanError::from)?)
        } else {
            None
        };

        // Create descriptor pool (generous limits for compute)
        // Increased for distillation training which does many operations per batch
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 131072, // Was 65536
        }];
        let desc_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(16384) // Was 8192
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&desc_pool_info, None) }
            .map_err(VulkanError::from)?;
        let descriptor_pool_lock = Arc::new(Mutex::new(()));

        let kernels = Arc::new(Kernels::new().map_err(VulkanError::from)?);

        // Create push descriptor function loader if extension is available
        let push_descriptor_fn = if has_push_descriptor {
            kernels.enable_push_descriptors();
            info!("Push descriptors enabled (VK_KHR_push_descriptor)");
            Some(Arc::new(ash::khr::push_descriptor::Device::new(
                &instance, &device,
            )))
        } else {
            debug!("Push descriptors not available");
            None
        };

        // Query cooperative matrix capabilities if extension is enabled
        let coop_matrix_props = if enable_coop_matrix {
            match Self::query_coop_matrix_capabilities(&entry, &instance, physical_device) {
                Ok(caps) => {
                    if caps.best_f32_tile.is_none() {
                        warn!(
                            "[VK_KHR_cooperative_matrix] Disabled: no supported fp16xfp16->fp32 subgroup tile found"
                        );
                        None
                    } else {
                        debug!(
                            "[VK_KHR_cooperative_matrix] Enabled - F32 tile: {:?}, I8 tile: {:?}",
                            caps.best_f32_tile, caps.best_i8_tile
                        );
                        Some(caps)
                    }
                }
                Err(e) => {
                    warn!("[VK_KHR_cooperative_matrix] Failed to query: {}", e);
                    None
                }
            }
        } else {
            if !has_coop_matrix {
                debug!("[VK_KHR_cooperative_matrix] Not available on this device");
            } else if disabled {
                debug!("[VK_KHR_cooperative_matrix] Disabled via PARAMECIA_DISABLE_COOP_MATRIX");
            } else if !vendor_allows_coopmat {
                debug!(
                    "[VK_KHR_cooperative_matrix] Disabled on vendor 0x{:04x} ({}): unsupported tuning profile",
                    vendor_id,
                    Self::vendor_name(vendor_id),
                );
            } else if !subgroup_allows_coopmat {
                warn!(
                    "[VK_KHR_cooperative_matrix] Disabled: subgroup size {} (required 32)",
                    subgroup_size
                );
            }
            None
        };

        let dev = Self {
            ordinal,
            vendor_id,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            allocator: Arc::new(Mutex::new(allocator)),
            kernels,
            command_pool,
            descriptor_pool,
            descriptor_pool_lock,
            descriptor_set_cache: Arc::new(Mutex::new(HashMap::new())),
            global_seed: Arc::new(Mutex::new(0)),
            batch: Arc::new(Mutex::new(None)),
            record_lock: Arc::new(Mutex::new(())),
            queue_lock: Arc::new(Mutex::new(())),
            scalar_cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
            cmd_pool: Arc::new(Mutex::new(Vec::new())),
            cleanup_queue: Arc::new(SegQueue::new()),
            timeline_semaphore,
            timeline_value: Arc::new(AtomicU64::new(0)),
            coop_matrix_props,
            has_integer_dot_product,
            has_fp16_compute,
            shader_core_count,
            subgroup_size,
            subgroup_min_size,
            subgroup_max_size,
            scratch_pool: Arc::new(Mutex::new(Vec::new())),
            download_staging_pool: Arc::new(Mutex::new(Vec::new())),
            upload_staging_pool: Arc::new(Mutex::new(Vec::new())),
            buffer_registry: Arc::new(Mutex::new(HashMap::new())),
            device_api_lock: Arc::new(Mutex::new(())),
            push_descriptor_fn,
            transfer_queue,
            transfer_queue_family_index,
            transfer_same_family,
            transfer_command_pool,
            transfer_queue_lock: Arc::new(Mutex::new(())),
            transfer_timeline_semaphore,
            transfer_timeline_value: Arc::new(AtomicU64::new(0)),
            _entry: entry,
        };
        //eprintln!("[vulkan] Device created, checking VK_VALIDATE...");
        install_crash_handler();
        if std::env::var("VK_VALIDATE")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            dev.validate_transfer()?;
        }
        Ok(dev)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Vulkan {
            gpu_id: self.ordinal,
        }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.ordinal == other.ordinal
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let count = shape.elem_count();
        let byte_size = count * dtype.size_in_bytes();
        if byte_size == 0 {
            return Ok(VulkanStorage::new(None, self.clone(), 0, dtype));
        }
        let zeros = vec![0u8; byte_size];
        let buffer = self.allocate_buffer(byte_size as vk::DeviceSize)?;
        self.upload_to_buffer(&zeros, &buffer)?;
        Ok(VulkanStorage::new(
            Some(Arc::new(buffer)),
            self.clone(),
            count,
            dtype,
        ))
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let count = shape.elem_count();
        let byte_size = count * dtype.size_in_bytes();
        if byte_size == 0 {
            return Ok(VulkanStorage::new(None, self.clone(), 0, dtype));
        }
        let buffer = self.allocate_buffer(byte_size as vk::DeviceSize)?;
        Ok(VulkanStorage::new(
            Some(Arc::new(buffer)),
            self.clone(),
            count,
            dtype,
        ))
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(storage) => self.allocate_with_data(T::DTYPE, storage),
            CpuStorageRef::U32(storage) => self.allocate_with_data(T::DTYPE, storage),
            CpuStorageRef::I64(storage) => self.allocate_with_data(T::DTYPE, storage),
            CpuStorageRef::BF16(storage) => self.allocate_with_data(T::DTYPE, storage),
            CpuStorageRef::F16(storage) => self.allocate_with_data(T::DTYPE, storage),
            CpuStorageRef::F32(storage) => self.allocate_with_data(T::DTYPE, storage),
            CpuStorageRef::F64(storage) => self.allocate_with_data(T::DTYPE, storage),
            CpuStorageRef::I16(storage) => self.allocate_with_data(T::DTYPE, storage),
            CpuStorageRef::I32(storage) => self.allocate_with_data(T::DTYPE, storage),
            CpuStorageRef::F8E4M3(storage) => {
                // F8E4M3 doesn't implement Pod, convert to f32 then upload
                let f32_data: Vec<f32> = storage.iter().map(|v| v.to_f64() as f32).collect();
                self.allocate_with_data(DType::F32, &f32_data)
            }
        }
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        match storage {
            CpuStorage::U8(s) => self.allocate_with_data(storage.dtype(), s),
            CpuStorage::U32(s) => self.allocate_with_data(storage.dtype(), s),
            CpuStorage::I64(s) => self.allocate_with_data(storage.dtype(), s),
            CpuStorage::BF16(s) => self.allocate_with_data(storage.dtype(), s),
            CpuStorage::F16(s) => self.allocate_with_data(storage.dtype(), s),
            CpuStorage::F32(s) => self.allocate_with_data(storage.dtype(), s),
            CpuStorage::F64(s) => self.allocate_with_data(storage.dtype(), s),
            CpuStorage::I16(s) => self.allocate_with_data(storage.dtype(), s),
            CpuStorage::I32(s) => self.allocate_with_data(storage.dtype(), s),
            CpuStorage::F8E4M3(s) => {
                let f32_data: Vec<f32> = s.iter().map(|v| v.to_f64() as f32).collect();
                self.allocate_with_data(DType::F32, &f32_data)
            }
        }
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        low: f64,
        high: f64,
    ) -> Result<Self::Storage> {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            element_count: u32,
            global_seed: u32,
            low: f32,
            high: f32,
        }

        let elem_count = shape.elem_count();
        let seed = {
            let mut s = self
                .global_seed
                .lock()
                .map_err(|e| VulkanError::Message(format!("seed lock: {}", e)))?;
            let cur = *s;
            *s = s.wrapping_add(elem_count as u64);
            cur
        };

        let pc = PC {
            element_count: elem_count as u32,
            global_seed: seed as u32,
            low: low as f32,
            high: high as f32,
        };

        let pipeline = self
            .kernels()
            .load_pipeline(
                self.device(),
                "rand_uniform_f32",
                None,
                std::mem::size_of::<PC>() as u32,
                1,
                None,
                false,
            )
            .map_err(|e| crate::Error::Vulkan(VulkanError::from(e)))?;

        let dst = self.zeros_impl(shape, DType::F32)?;
        let dst_buf = dst.vk_buffer()?;

        // Allocate descriptor set
        let set_layouts = [pipeline.desc_set_layout];
        let alloc_info = ash::vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&set_layouts);
        let _descriptor_pool_guard = self
            .descriptor_pool_lock
            .lock()
            .map_err(|e| VulkanError::Message(format!("descriptor pool lock: {}", e)))?;
        let desc_sets = unsafe { self.device().allocate_descriptor_sets(&alloc_info) }
            .map_err(VulkanError::from)?;
        let desc_set = desc_sets[0];

        let buf_info = ash::vk::DescriptorBufferInfo::default()
            .buffer(dst_buf)
            .offset(0)
            .range(ash::vk::WHOLE_SIZE);
        let write = ash::vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(0)
            .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buf_info));
        unsafe { self.device().update_descriptor_sets(&[write], &[]) };

        self.execute_one_time(|cmd| unsafe {
            self.device().cmd_bind_pipeline(
                cmd,
                ash::vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            );
            self.device().cmd_bind_descriptor_sets(
                cmd,
                ash::vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[desc_set],
                &[],
            );
            let pc_bytes = bytemuck::bytes_of(&pc);
            self.device().cmd_push_constants(
                cmd,
                pipeline.layout,
                ash::vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );
            let groups = ((elem_count as u32) + 255) / 256;
            self.device().cmd_dispatch(cmd, groups, 1, 1);
        })?;

        let _descriptor_pool_guard = self
            .descriptor_pool_lock
            .lock()
            .map_err(|e| VulkanError::Message(format!("descriptor pool lock: {}", e)))?;
        unsafe {
            self.device()
                .free_descriptor_sets(self.descriptor_pool, &[desc_set])
        }
        .map_err(VulkanError::from)?;

        if dtype == DType::F32 {
            Ok(dst)
        } else {
            use crate::backend::BackendStorage;
            let layout = Layout::contiguous(shape);
            dst.to_dtype(&layout, dtype)
        }
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        stddev: f64,
    ) -> Result<Self::Storage> {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PC {
            element_count: u32,
            global_seed: u32,
            mean: f32,
            stddev: f32,
        }

        let elem_count = shape.elem_count();
        let seed = {
            let mut s = self
                .global_seed
                .lock()
                .map_err(|e| VulkanError::Message(format!("seed lock: {}", e)))?;
            let cur = *s;
            *s = s.wrapping_add(elem_count as u64);
            cur
        };

        let pc = PC {
            element_count: elem_count as u32,
            global_seed: seed as u32,
            mean: mean as f32,
            stddev: stddev as f32,
        };

        let pipeline = self
            .kernels()
            .load_pipeline(
                self.device(),
                "rand_normal_f32",
                None,
                std::mem::size_of::<PC>() as u32,
                1,
                None,
                false,
            )
            .map_err(|e| crate::Error::Vulkan(VulkanError::from(e)))?;

        let dst = self.zeros_impl(shape, DType::F32)?;
        let dst_buf = dst.vk_buffer()?;

        let set_layouts = [pipeline.desc_set_layout];
        let alloc_info = ash::vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&set_layouts);
        let _descriptor_pool_guard = self
            .descriptor_pool_lock
            .lock()
            .map_err(|e| VulkanError::Message(format!("descriptor pool lock: {}", e)))?;
        let desc_sets = unsafe { self.device().allocate_descriptor_sets(&alloc_info) }
            .map_err(VulkanError::from)?;
        let desc_set = desc_sets[0];

        let buf_info = ash::vk::DescriptorBufferInfo::default()
            .buffer(dst_buf)
            .offset(0)
            .range(ash::vk::WHOLE_SIZE);
        let write = ash::vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(0)
            .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buf_info));
        unsafe { self.device().update_descriptor_sets(&[write], &[]) };

        self.execute_one_time(|cmd| unsafe {
            self.device().cmd_bind_pipeline(
                cmd,
                ash::vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            );
            self.device().cmd_bind_descriptor_sets(
                cmd,
                ash::vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[desc_set],
                &[],
            );
            let pc_bytes = bytemuck::bytes_of(&pc);
            self.device().cmd_push_constants(
                cmd,
                pipeline.layout,
                ash::vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );
            let groups = ((elem_count as u32) + 255) / 256;
            self.device().cmd_dispatch(cmd, groups, 1, 1);
        })?;

        let _descriptor_pool_guard = self
            .descriptor_pool_lock
            .lock()
            .map_err(|e| VulkanError::Message(format!("descriptor pool lock: {}", e)))?;
        unsafe {
            self.device()
                .free_descriptor_sets(self.descriptor_pool, &[desc_set])
        }
        .map_err(VulkanError::from)?;

        if dtype == DType::F32 {
            Ok(dst)
        } else {
            use crate::backend::BackendStorage;
            let layout = Layout::contiguous(shape);
            dst.to_dtype(&layout, dtype)
        }
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        let mut global_seed = self
            .global_seed
            .lock()
            .map_err(|e| VulkanError::Message(format!("seed lock: {}", e)))?;
        *global_seed = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        let seed = self
            .global_seed
            .lock()
            .map_err(|e| VulkanError::Message(format!("seed lock: {}", e)))?;
        Ok(*seed)
    }

    fn synchronize(&self) -> Result<()> {
        unsafe { self.device.device_wait_idle() }.map_err(VulkanError::from)?;
        Ok(())
    }
}
