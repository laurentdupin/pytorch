# Vulkan stream/event retirement notes

This backend uses Vulkan timeline semaphores as the first-class completion
primitive for logical Vulkan streams.

Version one keeps the existing per-context command buffer model. When the
current stream changes, the context submits any pending command buffer before
switching streams. This preserves correctness without requiring a broad command
recording rewrite. Multiple logical streams may still submit to the same Vulkan
queue, but each stream owns a distinct timeline semaphore and stream id.

Normal resource cleanup is attached to a `VulkanSubmission` token and is polled
through `RetireQueue`. Buffers and images registered for cleanup are destroyed
only after the stream timeline value for the submission has completed.

The remaining `vkQueueWaitIdle` call is in GPU timestamp query reset. That path
is profiling/debug-only and increments `queue_wait_idle_count`. Normal command
submission, event wait, stream synchronization, and `vulkan_prepack::synchronize`
use timeline waits instead of queue idle.

## Follow-up Audits

`recordDataPtrOnStream` is not wired as a Vulkan allocator hook because Vulkan
tensors in this backend are `VulkanOpaqueTensorImpl<vTensor>` objects, not
ordinary caching-allocator `DataPtr` allocations. Stream lifetime is currently
tracked through submitted `VulkanBuffer` and `VulkanImage` resources and the
retire queue. A future allocator rewrite should add per-allocation stream-use
tracking at the `vTensor` storage/resource layer, not by inventing a fake
`DataPtr` mapping.

Local linear/matmul hot paths were checked for naive row-padding experiments.
Remaining padding state is in compiled-session/executable-region layout
planning, where alignment belongs; no per-op linear row-padding materialization
path was added for this stream/event cleanup.
