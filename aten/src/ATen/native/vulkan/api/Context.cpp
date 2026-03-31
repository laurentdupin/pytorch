#include <ATen/native/vulkan/api/Context.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <memory>
#include <sstream>
#include <unordered_map>

#ifndef VULKAN_DESCRIPTOR_POOL_SIZE
#define VULKAN_DESCRIPTOR_POOL_SIZE 1024u
#endif

#ifndef VULKAN_QUERY_POOL_SIZE
#define VULKAN_QUERY_POOL_SIZE 4096u
#endif

namespace at {
namespace native {
namespace vulkan {
namespace api {

namespace {

constexpr uint64_t kCleanupSoftThresholdBytes = 512ull * 1024ull * 1024ull;
constexpr uint64_t kCleanupHardThresholdBytes = 1024ull * 1024ull * 1024ull;
constexpr uint32_t kCleanupSubmissionThreshold = 8u;
constexpr uint32_t kCleanupMaxSubmissionThreshold = 16u;
constexpr uint32_t kSoftReclaimsPerPoolFlush = 8u;

const std::string& sync_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_SYNC_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool sync_logging_enabled() {
  return !sync_log_path().empty();
}

std::string format_sync_bytes(const uint64_t bytes) {
  std::ostringstream stream;
  const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
  stream.setf(std::ios::fixed);
  stream.precision(2);
  stream << mib << " MiB";
  return stream.str();
}

template <typename Resource>
void accumulate_cleanup_bytes_by_label(
    const std::vector<Resource>& resources,
    std::unordered_map<std::string, uint64_t>& bytes_by_label) {
  for (const auto& resource : resources) {
    if (!resource.owns_memory()) {
      continue;
    }
    bytes_by_label[resource.allocation_label()] +=
        static_cast<uint64_t>(resource.allocated_size());
  }
}

std::string cleanup_signature(const VulkanBuffer& buffer) {
  std::ostringstream stream;
  stream << "buffer(size=" << format_sync_bytes(static_cast<uint64_t>(buffer.mem_size()))
         << ",alloc=" << format_sync_bytes(static_cast<uint64_t>(buffer.allocated_size()))
         << ")";
  return stream.str();
}

std::string cleanup_signature(const VulkanImage& image) {
  const VkExtent3D extents = image.extents();
  std::ostringstream stream;
  stream << "image(extents=" << extents.width << "x" << extents.height << "x"
         << extents.depth << ",alloc="
         << format_sync_bytes(static_cast<uint64_t>(image.allocated_size())) << ")";
  return stream.str();
}

template <typename Resource>
void accumulate_unlabeled_cleanup_signatures(
    const std::vector<Resource>& resources,
    std::unordered_map<std::string, uint64_t>& bytes_by_signature) {
  for (const auto& resource : resources) {
    if (!resource.owns_memory() || resource.allocation_label() != "unlabeled") {
      continue;
    }
    bytes_by_signature[cleanup_signature(resource)] +=
        static_cast<uint64_t>(resource.allocated_size());
  }
}

std::string top_cleanup_label_summary(
    const std::unordered_map<std::string, uint64_t>& bytes_by_label,
    const size_t limit = 16u) {
  if (bytes_by_label.empty()) {
    return {};
  }

  std::vector<std::pair<std::string, uint64_t>> entries(
      bytes_by_label.begin(), bytes_by_label.end());
  std::sort(
      entries.begin(),
      entries.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

  std::ostringstream stream;
  const size_t count = std::min(limit, entries.size());
  for (size_t idx = 0u; idx < count; ++idx) {
    if (idx > 0u) {
      stream << ", ";
    }
    stream << entries[idx].first << ":" << format_sync_bytes(entries[idx].second);
  }
  return stream.str();
}

std::string top_cleanup_signature_summary(
    const std::unordered_map<std::string, uint64_t>& bytes_by_signature,
    const size_t limit = 8u) {
  if (bytes_by_signature.empty()) {
    return {};
  }

  std::vector<std::pair<std::string, uint64_t>> entries(
      bytes_by_signature.begin(), bytes_by_signature.end());
  std::sort(
      entries.begin(),
      entries.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

  std::ostringstream stream;
  const size_t count = std::min(limit, entries.size());
  for (size_t idx = 0u; idx < count; ++idx) {
    if (idx > 0u) {
      stream << ", ";
    }
    stream << entries[idx].first << ":" << format_sync_bytes(entries[idx].second);
  }
  return stream.str();
}

void append_sync_log_line(const std::string& line) {
  if (!sync_logging_enabled()) {
    return;
  }

  std::ofstream out(sync_log_path(), std::ios::app);
  out << line << '\n';
}

} // namespace

Context::Context(size_t adapter_i, const ContextConfig& config)
    : config_(config),
      // Important handles
      adapter_p_(runtime()->get_adapter_p(adapter_i)),
      device_(adapter_p_->device_handle()),
      queue_(adapter_p_->request_queue()),
      // Resource pools
      command_pool_(device_, queue_.family_index, config_.cmdPoolConfig),
      descriptor_pool_(device_, config_.descriptorPoolConfig),
      fences_(device_),
// Diagnostics
#ifdef USE_VULKAN_GPU_DIAGNOSTICS
      querypool_(config_.queryPoolConfig, adapter_p_),
#endif /* USE_VULKAN_GPU_DIAGNOSTICS */
      // Command buffer submission
      cmd_mutex_{},
      cmd_(VK_NULL_HANDLE, 0u),
      submit_count_{0u},
      // Memory Management
      buffer_clearlist_mutex_{},
      buffers_to_clear_{},
      image_clearlist_mutex_{},
      images_to_clear_{},
      pending_cleanup_bytes_{0u},
      submissions_since_reclaim_{0u},
      reclaims_since_pool_flush_{0u} {
}

Context::~Context() {
  try {
    flush();
    // Let the device know the context is done with the queue
    adapter_p_->return_queue(queue_);
  } catch (...) {
  }
}

DescriptorSet Context::get_descriptor_set(
    const ShaderInfo& shader_descriptor,
    const utils::uvec3& local_workgroup_size) {
  VkDescriptorSetLayout shader_layout =
      shader_layout_cache().retrieve(shader_descriptor.kernel_layout);

  VkPipelineLayout pipeline_layout =
      pipeline_layout_cache().retrieve(shader_layout);

  VkPipeline pipeline = pipeline_cache().retrieve(
      {pipeline_layout_cache().retrieve(shader_layout),
       shader_cache().retrieve(shader_descriptor),
       local_workgroup_size});

  cmd_.bind_pipeline(pipeline, pipeline_layout, local_workgroup_size);

  return descriptor_pool().get_descriptor_set(
      shader_layout, shader_descriptor.kernel_layout);
}

void Context::register_shader_dispatch(
    const DescriptorSet& descriptors,
    PipelineBarrier& pipeline_barrier,
    const ShaderInfo& shader_descriptor,
    const utils::uvec3& global_workgroup_size) {
  // Adjust the global workgroup size based on the output tile size
  const utils::uvec3 effective_global_wg = {
      utils::div_up(
          global_workgroup_size.data[0u],
          shader_descriptor.out_tile_size.data[0u]),
      utils::div_up(
          global_workgroup_size.data[1u],
          shader_descriptor.out_tile_size.data[1u]),
      utils::div_up(
          global_workgroup_size.data[2u],
          shader_descriptor.out_tile_size.data[2u]),
  };

  cmd_.bind_descriptors(descriptors.get_bind_handle());
  cmd_.insert_barrier(pipeline_barrier);

  cmd_.dispatch(effective_global_wg);
}

void Context::submit_cmd_to_gpu(VkFence fence_handle, const bool final_use) {
  if (cmd_) {
    cmd_.end();
    adapter_p_->submit_cmd(
        queue_, cmd_.get_submit_handle(final_use), fence_handle);

    submit_count_ = 0u;
    submissions_since_reclaim_.fetch_add(1u, std::memory_order_relaxed);
  }
}

void Context::clear_deferred_cleanup_locked() {
  std::lock_guard<std::mutex> bufferlist_lock(buffer_clearlist_mutex_);
  std::lock_guard<std::mutex> imagelist_lock(image_clearlist_mutex_);
  buffers_to_clear_.clear();
  images_to_clear_.clear();
  pending_cleanup_bytes_.store(0u, std::memory_order_relaxed);
}

bool Context::should_sync_and_reclaim() {
  const uint64_t pending_cleanup = pending_cleanup_bytes();
  const uint32_t submitted_work = submissions_since_reclaim();

  return pending_cleanup >= kCleanupHardThresholdBytes ||
      (pending_cleanup >= kCleanupSoftThresholdBytes &&
       submitted_work >= kCleanupSubmissionThreshold) ||
      submitted_work >= kCleanupMaxSubmissionThreshold;
}

void Context::sync_and_reclaim() {
  const uint64_t pending_cleanup = pending_cleanup_bytes();
  const uint32_t submitted_work = submissions_since_reclaim();
  if (pending_cleanup == 0u && submitted_work == 0u && submit_count_ == 0u) {
    return;
  }

  const bool full_pool_flush =
      pending_cleanup >= kCleanupHardThresholdBytes ||
      reclaims_since_pool_flush_ >= kSoftReclaimsPerPoolFlush;

  if (sync_logging_enabled()) {
    std::unordered_map<std::string, uint64_t> cleanup_bytes_by_label;
    std::unordered_map<std::string, uint64_t> unlabeled_cleanup_by_signature;
    {
      std::lock_guard<std::mutex> bufferlist_lock(buffer_clearlist_mutex_);
      accumulate_cleanup_bytes_by_label(buffers_to_clear_, cleanup_bytes_by_label);
      accumulate_unlabeled_cleanup_signatures(
          buffers_to_clear_, unlabeled_cleanup_by_signature);
    }
    {
      std::lock_guard<std::mutex> imagelist_lock(image_clearlist_mutex_);
      accumulate_cleanup_bytes_by_label(images_to_clear_, cleanup_bytes_by_label);
      accumulate_unlabeled_cleanup_signatures(
          images_to_clear_, unlabeled_cleanup_by_signature);
    }

    std::ostringstream stream;
    stream << "sync_and_reclaim: pending=" << format_sync_bytes(pending_cleanup)
           << " submitted=" << submitted_work
           << " submit_count=" << submit_count_
           << " caller=" << current_allocation_label()
           << " full_pool_flush=" << (full_pool_flush ? "1" : "0")
           << " reclaims_since_pool_flush=" << reclaims_since_pool_flush_;
    const std::string top_labels =
        top_cleanup_label_summary(cleanup_bytes_by_label);
    if (!top_labels.empty()) {
      stream << " cleanup_labels={" << top_labels << "}";
    }
    const std::string top_unlabeled =
        top_cleanup_signature_summary(unlabeled_cleanup_by_signature);
    if (!top_unlabeled.empty()) {
      stream << " unlabeled_signatures={" << top_unlabeled << "}";
    }
    append_sync_log_line(stream.str());
  }

  std::unique_lock<std::mutex> context_lock(dispatch_lock());

  if (cmd_) {
    VulkanFence fence = fences_.get_fence();
    submit_cmd_to_gpu(fence.get_submit_handle(), full_pool_flush);
    fence.wait();
    fences_.return_fence(fence);
  } else if (submitted_work > 0u) {
    VK_CHECK(vkQueueWaitIdle(queue()));
  } else {
    return;
  }

  descriptor_pool_.flush();

  if (full_pool_flush) {
    command_pool_.flush();
    if (cmd_) {
      cmd_.invalidate();
    }
    reclaims_since_pool_flush_ = 0u;
  } else {
    reclaims_since_pool_flush_++;
  }
  submit_count_ = 0u;
  submissions_since_reclaim_.store(0u, std::memory_order_relaxed);
  clear_deferred_cleanup_locked();
}

void Context::flush() {
  if (sync_logging_enabled()) {
    std::ostringstream stream;
    stream << "flush: pending=" << format_sync_bytes(pending_cleanup_bytes())
           << " submitted=" << submissions_since_reclaim()
           << " submit_count=" << submit_count_
           << " reclaims_since_pool_flush=" << reclaims_since_pool_flush_;
    append_sync_log_line(stream.str());
  }

  VK_CHECK(vkQueueWaitIdle(queue()));

  command_pool_.flush();
  descriptor_pool_.flush();

  // If there is an existing command buffer, invalidate it
  if (cmd_) {
    cmd_.invalidate();
  }

  reclaims_since_pool_flush_ = 0u;
  submit_count_ = 0u;
  submissions_since_reclaim_.store(0u, std::memory_order_relaxed);
  clear_deferred_cleanup_locked();
}

void Context::flush_after_fence_wait() {
  if (sync_logging_enabled()) {
    std::ostringstream stream;
    stream << "flush_after_fence_wait: pending="
           << format_sync_bytes(pending_cleanup_bytes())
           << " submitted=" << submissions_since_reclaim()
           << " submit_count=" << submit_count_
           << " caller=" << current_allocation_label()
           << " reclaims_since_pool_flush=" << reclaims_since_pool_flush_;
    append_sync_log_line(stream.str());
  }

  command_pool_.flush();
  descriptor_pool_.flush();

  if (cmd_) {
    cmd_.invalidate();
  }

  reclaims_since_pool_flush_ = 0u;
  submit_count_ = 0u;
  submissions_since_reclaim_.store(0u, std::memory_order_relaxed);
  clear_deferred_cleanup_locked();
}

bool available() {
  return context();
}

Context* context() {
  static const std::unique_ptr<Context> context([]() -> Context* {
    try {
      const uint32_t submit_frequency = 16u;

      const CommandPoolConfig cmd_config{
          32u, // cmdPoolInitialSize
          8u, // cmdPoolBatchSize
      };

      const DescriptorPoolConfig descriptor_pool_config{
          VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorPoolMaxSets
          VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorUniformBufferCount
          VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorStorageBufferCount
          VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorCombinedSamplerCount
          VULKAN_DESCRIPTOR_POOL_SIZE, // descriptorStorageImageCount
          32u, // descriptorPileSizes
      };

      const QueryPoolConfig query_pool_config{
          VULKAN_QUERY_POOL_SIZE, // maxQueryCount
          256u, // initialReserveSize
      };

      const ContextConfig config{
          submit_frequency, // cmdSubmitFrequency
          cmd_config, // cmdPoolConfig
          descriptor_pool_config, // descriptorPoolConfig
          query_pool_config, // queryPoolConfig
      };

      return new Context(runtime()->default_adapter_i(), config);
    } catch (...) {
    }

    return nullptr;
  }());

  return context.get();
}

//
// UniformParamsBuffer
//

namespace {

void memcpy_to_buffer(const VulkanBuffer& src, VulkanBuffer& dst) {
  MemoryMap dst_mapping(dst, MemoryAccessType::WRITE);

  MemoryMap src_mapping(src, MemoryAccessType::READ);
  src_mapping.invalidate();

  void* dst_ptr = dst_mapping.template data<void>();
  void* src_ptr = src_mapping.template data<void>();

  // @lint-ignore CLANGTIDY facebook-security-vulnerable-memcpy
  memcpy(dst_ptr, src_ptr, src.mem_size());
}

} // namespace

UniformParamsBuffer::UniformParamsBuffer(const UniformParamsBuffer& other)
    : context_p_(other.context_p_), vulkan_buffer_{} {
  if (other.vulkan_buffer_) {
    vulkan_buffer_ = context_p_->adapter_ptr()->vma().create_uniform_buffer(
        other.vulkan_buffer_.mem_size());

    memcpy_to_buffer(other.vulkan_buffer_, vulkan_buffer_);
  }
}

UniformParamsBuffer& UniformParamsBuffer::operator=(
    const UniformParamsBuffer& other) {
  if (&other != this) {
    context_p_ = other.context_p_;

    // Move vulkan_buffer_ to another VulkanBuffer for cleanup
    if (vulkan_buffer_) {
      VulkanBuffer temp_buffer(std::move(vulkan_buffer_));
      context_p_->register_buffer_cleanup(temp_buffer);
    }
    // vulkan_buffer_ should now be empty

    if (other.vulkan_buffer_) {
      vulkan_buffer_ = context_p_->adapter_ptr()->vma().create_uniform_buffer(
          other.vulkan_buffer_.mem_size());

      memcpy_to_buffer(other.vulkan_buffer_, vulkan_buffer_);
    }
  }

  return *this;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
