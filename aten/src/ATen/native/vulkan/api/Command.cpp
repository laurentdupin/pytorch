#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Command.h>

#include <mutex>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace api {

namespace {

VkPipelineStageFlags2 to_stage2(const VkPipelineStageFlags stages) {
  VkPipelineStageFlags2 out = 0u;
  if (stages & VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT) {
    out |= VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
  }
  if (stages & VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT) {
    out |= VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
  }
  if (stages & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
    out |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  }
  if (stages & VK_PIPELINE_STAGE_TRANSFER_BIT) {
    out |= VK_PIPELINE_STAGE_2_TRANSFER_BIT;
  }
  if (stages & VK_PIPELINE_STAGE_HOST_BIT) {
    out |= VK_PIPELINE_STAGE_2_HOST_BIT;
  }
  return out != 0u ? out : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
}

VkAccessFlags2 to_access2(const VkAccessFlags access) {
  VkAccessFlags2 out = 0u;
  if (access & VK_ACCESS_SHADER_READ_BIT) {
    out |= VK_ACCESS_2_SHADER_READ_BIT;
  }
  if (access & VK_ACCESS_SHADER_WRITE_BIT) {
    out |= VK_ACCESS_2_SHADER_WRITE_BIT;
  }
  if (access & VK_ACCESS_TRANSFER_READ_BIT) {
    out |= VK_ACCESS_2_TRANSFER_READ_BIT;
  }
  if (access & VK_ACCESS_TRANSFER_WRITE_BIT) {
    out |= VK_ACCESS_2_TRANSFER_WRITE_BIT;
  }
  if (access & VK_ACCESS_HOST_READ_BIT) {
    out |= VK_ACCESS_2_HOST_READ_BIT;
  }
  if (access & VK_ACCESS_HOST_WRITE_BIT) {
    out |= VK_ACCESS_2_HOST_WRITE_BIT;
  }
  if (access & VK_ACCESS_MEMORY_READ_BIT) {
    out |= VK_ACCESS_2_MEMORY_READ_BIT;
  }
  if (access & VK_ACCESS_MEMORY_WRITE_BIT) {
    out |= VK_ACCESS_2_MEMORY_WRITE_BIT;
  }
  return out;
}

} // namespace

//
// CommandBuffer
//

CommandBuffer::CommandBuffer(
    VkCommandBuffer handle,
    const VkCommandBufferUsageFlags flags,
    PFN_vkCmdPipelineBarrier2 cmd_pipeline_barrier2,
    const VkCommandBufferLevel level)
    : handle_(handle),
      flags_(flags),
      cmd_pipeline_barrier2_(cmd_pipeline_barrier2),
      level_(level),
      state_(CommandBuffer::State::NEW),
      bound_{} {}

CommandBuffer::CommandBuffer(CommandBuffer&& other) noexcept
    : handle_(other.handle_),
      flags_(other.flags_),
      cmd_pipeline_barrier2_(other.cmd_pipeline_barrier2_),
      level_(other.level_),
      state_(other.state_),
      bound_(other.bound_) {
  other.handle_ = VK_NULL_HANDLE;
  other.cmd_pipeline_barrier2_ = nullptr;
  other.bound_.reset();
  other.state_ = CommandBuffer::State::INVALID;
}

CommandBuffer& CommandBuffer::operator=(CommandBuffer&& other) noexcept {
  handle_ = other.handle_;
  flags_ = other.flags_;
  cmd_pipeline_barrier2_ = other.cmd_pipeline_barrier2_;
  level_ = other.level_;
  state_ = other.state_;
  bound_ = other.bound_;

  other.handle_ = VK_NULL_HANDLE;
  other.cmd_pipeline_barrier2_ = nullptr;
  other.bound_.reset();
  other.state_ = CommandBuffer::State::INVALID;

  return *this;
}

void CommandBuffer::begin() {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::NEW,
      "Vulkan CommandBuffer: called begin() on a command buffer whose state "
      "is not NEW.");

  const VkCommandBufferInheritanceInfo inheritance_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
      nullptr,
      VK_NULL_HANDLE,
      0u,
      VK_NULL_HANDLE,
      false,
      0u,
      0u,
  };
  const VkCommandBufferBeginInfo begin_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      nullptr,
      flags_,
      is_secondary() ? &inheritance_info : nullptr,
  };

  VK_CHECK(vkBeginCommandBuffer(handle_, &begin_info));
  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::execute_commands(const CommandBuffer& secondary) {
  VK_CHECK_COND(
      !is_secondary() &&
          state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: execute_commands() requires a primary command "
      "buffer with its entry barriers inserted.");
  const VkCommandBuffer secondary_handle = secondary.get_execute_handle();
  vkCmdExecuteCommands(handle_, 1u, &secondary_handle);
  bound_.reset();
  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::end() {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::RECORDING ||
          state_ == CommandBuffer::State::READY ||
          state_ == CommandBuffer::State::SUBMITTED,
      "Vulkan CommandBuffer: called end() on a command buffer whose state "
      "is not RECORDING, READY, or SUBMITTED.");

  if (state_ == CommandBuffer::State::RECORDING) {
    VK_CHECK(vkEndCommandBuffer(handle_));
  }
  state_ = CommandBuffer::State::READY;
}

void CommandBuffer::bind_pipeline(
    VkPipeline pipeline,
    VkPipelineLayout pipeline_layout,
    const utils::uvec3 local_workgroup_size) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::RECORDING,
      "Vulkan CommandBuffer: called bind_pipeline() on a command buffer whose state "
      "is not RECORDING.");

  if (pipeline != bound_.pipeline) {
    vkCmdBindPipeline(handle_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    bound_.pipeline = pipeline;
  }

  bound_.pipeline_layout = pipeline_layout;
  bound_.local_workgroup_size = local_workgroup_size;

  state_ = CommandBuffer::State::PIPELINE_BOUND;
}

void CommandBuffer::bind_descriptors(VkDescriptorSet descriptors) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::PIPELINE_BOUND,
      "Vulkan CommandBuffer: called bind_descriptors() on a command buffer whose state "
      "is not PIPELINE_BOUND.");

  if (descriptors != bound_.descriptors) {
    vkCmdBindDescriptorSets(
        handle_, // commandBuffer
        VK_PIPELINE_BIND_POINT_COMPUTE, // pipelineBindPoint
        bound_.pipeline_layout, // layout
        0u, // firstSet
        1u, // descriptorSetCount
        &descriptors, // pDescriptorSets
        0u, // dynamicOffsetCount
        nullptr); // pDynamicOffsets
  }

  bound_.descriptors = descriptors;

  state_ = CommandBuffer::State::DESCRIPTORS_BOUND;
}

void CommandBuffer::insert_barrier(PipelineBarrier& pipeline_barrier) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::DESCRIPTORS_BOUND ||
          state_ == CommandBuffer::State::RECORDING,
      "Vulkan CommandBuffer: called insert_barrier() on a command buffer whose state "
      "is not DESCRIPTORS_BOUND or RECORDING.");

  if (pipeline_barrier) {
    std::vector<VkMemoryBarrier2> memory_barriers;
    if (pipeline_barrier.buffers.empty() && pipeline_barrier.images.empty()) {
      memory_barriers.push_back({
          VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, // sType
          nullptr, // pNext
          to_stage2(pipeline_barrier.stage.src), // srcStageMask
          0u, // srcAccessMask
          to_stage2(pipeline_barrier.stage.dst), // dstStageMask
          0u, // dstAccessMask
      });
    }

    std::vector<VkBufferMemoryBarrier2> buffer_barriers;
    buffer_barriers.reserve(pipeline_barrier.buffers.size());
    for (const api::BufferMemoryBarrier& memory_barrier :
         pipeline_barrier.buffers) {
      const VkBufferMemoryBarrier& legacy = memory_barrier.handle;
      buffer_barriers.push_back({
          VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, // sType
          nullptr, // pNext
          to_stage2(pipeline_barrier.stage.src), // srcStageMask
          to_access2(legacy.srcAccessMask), // srcAccessMask
          to_stage2(pipeline_barrier.stage.dst), // dstStageMask
          to_access2(legacy.dstAccessMask), // dstAccessMask
          legacy.srcQueueFamilyIndex, // srcQueueFamilyIndex
          legacy.dstQueueFamilyIndex, // dstQueueFamilyIndex
          legacy.buffer, // buffer
          legacy.offset, // offset
          legacy.size, // size
      });
    }

    std::vector<VkImageMemoryBarrier2> image_barriers;
    image_barriers.reserve(pipeline_barrier.images.size());
    for (const api::ImageMemoryBarrier& memory_barrier :
         pipeline_barrier.images) {
      const VkImageMemoryBarrier& legacy = memory_barrier.handle;
      image_barriers.push_back({
          VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, // sType
          nullptr, // pNext
          to_stage2(pipeline_barrier.stage.src), // srcStageMask
          to_access2(legacy.srcAccessMask), // srcAccessMask
          to_stage2(pipeline_barrier.stage.dst), // dstStageMask
          to_access2(legacy.dstAccessMask), // dstAccessMask
          legacy.oldLayout, // oldLayout
          legacy.newLayout, // newLayout
          legacy.srcQueueFamilyIndex, // srcQueueFamilyIndex
          legacy.dstQueueFamilyIndex, // dstQueueFamilyIndex
          legacy.image, // image
          legacy.subresourceRange, // subresourceRange
      });
    }

    const VkDependencyInfo dependency_info{
        VK_STRUCTURE_TYPE_DEPENDENCY_INFO, // sType
        nullptr, // pNext
        0u, // dependencyFlags
        static_cast<uint32_t>(memory_barriers.size()), // memoryBarrierCount
        !memory_barriers.empty() ? memory_barriers.data()
                                 : nullptr, // pMemoryBarriers
        static_cast<uint32_t>(buffer_barriers.size()), // bufferMemoryBarrierCount
        !buffer_barriers.empty() ? buffer_barriers.data()
                                 : nullptr, // pBufferMemoryBarriers
        static_cast<uint32_t>(image_barriers.size()), // imageMemoryBarrierCount
        !image_barriers.empty() ? image_barriers.data()
                                : nullptr, // pImageMemoryBarriers
    };

    VK_CHECK_COND(
        cmd_pipeline_barrier2_,
        "Vulkan synchronization2 command vkCmdPipelineBarrier2 was not loaded "
        "from the logical device.");
    cmd_pipeline_barrier2_(handle_, &dependency_info);
  }

  state_ = CommandBuffer::State::BARRIERS_INSERTED;
}

void CommandBuffer::dispatch(const utils::uvec3& global_workgroup_size) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called dispatch() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  vkCmdDispatch(
      handle_,
      utils::div_up(
          global_workgroup_size.data[0u], bound_.local_workgroup_size.data[0u]),
      utils::div_up(
          global_workgroup_size.data[1u], bound_.local_workgroup_size.data[1u]),
      utils::div_up(
          global_workgroup_size.data[2u],
          bound_.local_workgroup_size.data[2u]));

  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::copy_buffer_to_buffer(
    const api::VulkanBuffer& source,
    const api::VulkanBuffer& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called copy_buffer_to_buffer() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  const VkBufferCopy copy_details{
      src_offset.data[0u], // srcOffset
      dst_offset.data[0u], // dstOffset
      copy_range.data[0u], // size
  };

  vkCmdCopyBuffer(
      handle_, source.handle(), destination.handle(), 1u, &copy_details);

  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::copy_texture_to_texture(
    const api::VulkanImage& source,
    const api::VulkanImage& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called copy_texture_to_texture() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  const VkImageSubresourceLayers src_subresource_layers{
      VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
      0u, // mipLevel
      0u, // baseArrayLayer
      1u, // layerCount
  };

  const VkImageSubresourceLayers dst_subresource_layers{
      VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
      0u, // mipLevel
      0u, // baseArrayLayer
      1u, // layerCount
  };

  const VkImageCopy copy_details{
      src_subresource_layers, // srcSubresource
      create_offset3d(src_offset), // srcOffset
      dst_subresource_layers, // dstSubresource
      create_offset3d(dst_offset), // dstOffset
      create_extent3d(copy_range), // extent
  };

  vkCmdCopyImage(
      handle_,
      source.handle(),
      source.layout(),
      destination.handle(),
      destination.layout(),
      1u,
      &copy_details);

  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::copy_texture_to_buffer(
    const api::VulkanImage& source,
    const api::VulkanBuffer& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called copy_texture_to_buffer() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  const VkImageSubresourceLayers src_subresource_layers{
      VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
      0u, // mipLevel
      0u, // baseArrayLayer
      1u, // layerCount
  };

  const VkBufferImageCopy copy_details{
      dst_offset.data[0u], // bufferOffset
      dst_offset.data[1u], // bufferRowLength
      dst_offset.data[2u], // bufferImageHeight
      src_subresource_layers, // imageSubresource
      create_offset3d(src_offset), // imageOffset
      create_extent3d(copy_range), // imageExtent
  };

  vkCmdCopyImageToBuffer(
      handle_,
      source.handle(),
      source.layout(),
      destination.handle(),
      1u,
      &copy_details);

  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::copy_buffer_to_texture(
    const api::VulkanBuffer& source,
    const api::VulkanImage& destination,
    const api::utils::uvec3& copy_range,
    const api::utils::uvec3& src_offset,
    const api::utils::uvec3& dst_offset) {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::BARRIERS_INSERTED,
      "Vulkan CommandBuffer: called copy_buffer_to_texture() on a command buffer whose state "
      "is not BARRIERS_INSERTED.");

  const VkImageSubresourceLayers dst_subresource_layers{
      VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
      0u, // mipLevel
      0u, // baseArrayLayer
      1u, // layerCount
  };

  const VkBufferImageCopy copy_details{
      src_offset.data[0u], // bufferOffset
      src_offset.data[1u], // bufferRowLength
      src_offset.data[2u], // bufferImageHeight
      dst_subresource_layers, // imageSubresource
      create_offset3d(dst_offset), // imageOffset
      create_extent3d(copy_range), // imageExtent
  };

  vkCmdCopyBufferToImage(
      handle_,
      source.handle(),
      destination.handle(),
      destination.layout(),
      1u,
      &copy_details);

  state_ = CommandBuffer::State::RECORDING;
}

void CommandBuffer::write_timestamp(
    VkQueryPool querypool,
    const uint32_t idx,
    const VkPipelineStageFlagBits stage) const {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::RECORDING,
      "Vulkan CommandBuffer: called write_timestamp() on a command buffer whose state "
      "is not RECORDING.");

  vkCmdWriteTimestamp(handle_, stage, querypool, idx);
}

void CommandBuffer::reset_querypool(
    VkQueryPool querypool,
    const uint32_t first_idx,
    const uint32_t count) const {
  VK_CHECK_COND(
      state_ == CommandBuffer::State::RECORDING,
      "Vulkan CommandBuffer: called reset_querypool() on a command buffer whose state "
      "is not RECORDING.");

  vkCmdResetQueryPool(handle_, querypool, first_idx, count);
}

VkCommandBuffer CommandBuffer::get_submit_handle(const bool final_use) {
  VK_CHECK_COND(
      !is_secondary(),
      "Vulkan CommandBuffer: secondary command buffers cannot be submitted "
      "directly.");
  VK_CHECK_COND(
      state_ == CommandBuffer::State::READY ||
          (state_ == CommandBuffer::State::SUBMITTED && is_reusable()),
      "Vulkan CommandBuffer: called begin() on a command buffer whose state "
      "is not READY.");

  VkCommandBuffer handle = handle_;

  if (!is_reusable() || final_use) {
    invalidate();
  }
  state_ = CommandBuffer::State::SUBMITTED;

  return handle;
}

VkCommandBuffer CommandBuffer::get_execute_handle() const {
  VK_CHECK_COND(
      is_secondary() && state_ == CommandBuffer::State::READY,
      "Vulkan CommandBuffer: execute handle requires a ready secondary "
      "command buffer.");
  return handle_;
}

//
// CommandPool
//

CommandPool::CommandPool(
    VkDevice device,
    const uint32_t queue_family_idx,
    const CommandPoolConfig& config)
    : device_(device),
      queue_family_idx_(queue_family_idx),
      cmd_pipeline_barrier2_(reinterpret_cast<PFN_vkCmdPipelineBarrier2>(
          vkGetDeviceProcAddr(device_, "vkCmdPipelineBarrier2"))),
      pool_(VK_NULL_HANDLE),
      config_(config),
      mutex_{},
      primary_buffers_{},
      secondary_buffers_{},
      primary_in_use_(0u),
      secondary_in_use_(0u) {
  VK_CHECK_COND(
      cmd_pipeline_barrier2_,
      "Vulkan synchronization2 command vkCmdPipelineBarrier2 was not loaded "
      "from the logical device.");

  const VkCommandPoolCreateInfo create_info{
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      nullptr,
      VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      queue_family_idx_,
  };

  VK_CHECK(vkCreateCommandPool(device_, &create_info, nullptr, &pool_));

  // Pre-allocate some command buffers
  if (config_.cmdPoolInitialSize > 0u) {
    allocate_new_batch(
        config_.cmdPoolInitialSize, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
  }
}

CommandPool::~CommandPool() {
  if (VK_NULL_HANDLE == pool_) {
    return;
  }
  vkDestroyCommandPool(device_, pool_, nullptr);
}

CommandBuffer CommandPool::get_new_cmd(
    const bool reusable,
    const VkCommandBufferLevel level) {
  std::lock_guard<std::mutex> lock(mutex_);

  // No-ops if there are command buffers available
  allocate_new_batch(config_.cmdPoolBatchSize, level);

  std::vector<VkCommandBuffer>& buffers =
      level == VK_COMMAND_BUFFER_LEVEL_SECONDARY ? secondary_buffers_
                                                 : primary_buffers_;
  size_t& in_use = level == VK_COMMAND_BUFFER_LEVEL_SECONDARY
      ? secondary_in_use_
      : primary_in_use_;
  VkCommandBuffer handle = buffers[in_use];

  VkCommandBufferUsageFlags cmd_flags = 0u;
  if (!reusable) {
    cmd_flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  } else {
    cmd_flags |= VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  }

  in_use++;
  return CommandBuffer(handle, cmd_flags, cmd_pipeline_barrier2_, level);
}

void CommandPool::flush() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (primary_in_use_ == 0u && secondary_in_use_ == 0u) {
    return;
  }
  VK_CHECK(vkResetCommandPool(device_, pool_, 0u));
  primary_in_use_ = 0u;
  secondary_in_use_ = 0u;
}

void CommandPool::allocate_new_batch(
    const uint32_t count,
    const VkCommandBufferLevel level) {
  std::vector<VkCommandBuffer>& buffers =
      level == VK_COMMAND_BUFFER_LEVEL_SECONDARY ? secondary_buffers_
                                                 : primary_buffers_;
  const size_t in_use = level == VK_COMMAND_BUFFER_LEVEL_SECONDARY
      ? secondary_in_use_
      : primary_in_use_;
  // No-ops if there are still command buffers available
  if (in_use < buffers.size()) {
    return;
  }

  const size_t allocation_offset = buffers.size();
  buffers.resize(buffers.size() + count);

  const VkCommandBufferAllocateInfo allocate_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, // sType
      nullptr, // pNext
      pool_, // commandPool
      level, // level
      count, // commandBufferCount
  };

  VK_CHECK(vkAllocateCommandBuffers(
      device_, &allocate_info, buffers.data() + allocation_offset));
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
