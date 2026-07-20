#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/vk_api.h>

#include <ATen/native/vulkan/api/Descriptor.h>
#include <ATen/native/vulkan/api/Pipeline.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Shader.h>
#include <ATen/native/vulkan/api/Utils.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

class CommandBuffer final {
 public:
  explicit CommandBuffer(
      VkCommandBuffer,
      const VkCommandBufferUsageFlags,
      PFN_vkCmdPipelineBarrier2,
      VkCommandBufferLevel = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

  CommandBuffer(const CommandBuffer&) = delete;
  CommandBuffer& operator=(const CommandBuffer&) = delete;

  CommandBuffer(CommandBuffer&&) noexcept;
  CommandBuffer& operator=(CommandBuffer&&) noexcept;

  ~CommandBuffer() = default;

  // The lifecycle of a command buffer is as follows:
  enum State {
    INVALID, // Used to indicate the command buffer is moved from
    NEW, // Set during constructor
    RECORDING, // Set during call to begin(), dispatch(), and
               // copy_*_to_*()
    PIPELINE_BOUND, // Set during call to  bind_pipeline()
    DESCRIPTORS_BOUND, // Set during call to bind_descriptors()
    BARRIERS_INSERTED, // Set during call to insert_barrier()
    READY, //  Set during call to end()
    SUBMITTED, // Set during call to get_submit_handle()
  };

  struct Bound {
    VkPipeline pipeline;
    VkPipelineLayout pipeline_layout;
    utils::uvec3 local_workgroup_size;
    VkDescriptorSet descriptors;

    explicit Bound()
        : pipeline{VK_NULL_HANDLE},
          pipeline_layout{VK_NULL_HANDLE},
          local_workgroup_size{0u, 0u, 0u},
          descriptors{VK_NULL_HANDLE} {}

    inline void reset() {
      pipeline = VK_NULL_HANDLE;
      pipeline_layout = VK_NULL_HANDLE;
      local_workgroup_size = {0u, 0u, 0u};
      descriptors = VK_NULL_HANDLE;
    }
  };

 private:
  VkCommandBuffer handle_;
  VkCommandBufferUsageFlags flags_;
  PFN_vkCmdPipelineBarrier2 cmd_pipeline_barrier2_;
  VkCommandBufferLevel level_;
  State state_;
  Bound bound_;

 public:
  inline bool is_reusable() {
    return !(flags_ & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
  }

  inline bool is_secondary() const {
    return level_ == VK_COMMAND_BUFFER_LEVEL_SECONDARY;
  }

  inline void invalidate() {
    handle_ = VK_NULL_HANDLE;
    bound_.reset();
  }

  void begin();
  void end();

  void bind_pipeline(VkPipeline, VkPipelineLayout, const utils::uvec3);
  void bind_descriptors(VkDescriptorSet);

  void insert_barrier(PipelineBarrier& pipeline_barrier);
  void dispatch(const utils::uvec3&);
  void execute_commands(const CommandBuffer&);

  void copy_buffer_to_buffer(
      const api::VulkanBuffer&,
      const api::VulkanBuffer&,
      const api::utils::uvec3&,
      const api::utils::uvec3&,
      const api::utils::uvec3&);

  void copy_texture_to_texture(
      const api::VulkanImage&,
      const api::VulkanImage&,
      const api::utils::uvec3&,
      const api::utils::uvec3&,
      const api::utils::uvec3&);

  void copy_texture_to_buffer(
      const api::VulkanImage&,
      const api::VulkanBuffer&,
      const api::utils::uvec3&,
      const api::utils::uvec3&,
      const api::utils::uvec3&);

  void copy_buffer_to_texture(
      const api::VulkanBuffer&,
      const api::VulkanImage&,
      const api::utils::uvec3&,
      const api::utils::uvec3&,
      const api::utils::uvec3&);

  void write_timestamp(
      VkQueryPool,
      const uint32_t,
      VkPipelineStageFlagBits = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT) const;
  void reset_querypool(VkQueryPool, const uint32_t, const uint32_t) const;

  VkCommandBuffer get_submit_handle(const bool final_use = false);
  VkCommandBuffer get_execute_handle() const;

  inline operator bool() const {
    return VK_NULL_HANDLE != handle_;
  }
};

struct CommandPoolConfig final {
  uint32_t cmdPoolInitialSize;
  uint32_t cmdPoolBatchSize;
};

class CommandPool final {
 public:
  explicit CommandPool(VkDevice, const uint32_t, const CommandPoolConfig&);

  CommandPool(const CommandPool&) = delete;
  CommandPool& operator=(const CommandPool&) = delete;

  CommandPool(CommandPool&&) = delete;
  CommandPool& operator=(CommandPool&&) = delete;

  ~CommandPool();

 private:
  VkDevice device_;
  uint32_t queue_family_idx_;
  PFN_vkCmdPipelineBarrier2 cmd_pipeline_barrier2_;
  VkCommandPool pool_;
  CommandPoolConfig config_;
  // New Buffers
  std::mutex mutex_;
  std::vector<VkCommandBuffer> primary_buffers_;
  std::vector<VkCommandBuffer> secondary_buffers_;
  size_t primary_in_use_;
  size_t secondary_in_use_;

 public:
  CommandBuffer get_new_cmd(
      bool reusable = false,
      VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

  void flush();

 private:
  void allocate_new_batch(const uint32_t, VkCommandBufferLevel);
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
