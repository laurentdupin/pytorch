#include <ATen/native/vulkan/impl/Packing.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/zeros.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace utils {

using namespace api::utils;
namespace {

bool can_native_buffer_cast_input(const vTensor& v_input) {
  return (v_input.dtype() == api::kFloat || v_input.dtype() == api::kInt ||
          v_input.dtype() == api::kBFloat16) &&
      v_input.storage_type() == api::StorageType::BUFFER &&
      v_input.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
      !v_input.is_quantized();
}

Tensor cast_vulkan_tensor_dtype_cpu_fallback(
    const Tensor& input,
    const ScalarType dtype) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);
  return input.cpu().to(dtype).vulkan();
}

std::vector<int64_t> calc_logical_contiguous_strides(
    const std::vector<int64_t>& sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (int idx = safe_downcast<int>(sizes.size()) - 2; idx >= 0; --idx) {
    strides[idx] = strides[idx + 1] * std::max<int64_t>(sizes[idx + 1], 1);
  }
  return strides;
}

bool can_make_buffer_metadata_view_impl(
    const vTensor& v_input,
    IntArrayRef sizes,
    IntArrayRef logical_strides,
    IntArrayRef physical_strides,
    const int64_t storage_offset) {
  if (
      v_input.storage_type() != api::StorageType::BUFFER ||
      !supports_buffer_view_fast_path(v_input) ||
      sizes.size() != logical_strides.size() ||
      sizes.size() != physical_strides.size() || sizes.size() > 4 ||
      storage_offset < 0) {
    return false;
  }

  int64_t max_offset = storage_offset;
  for (const auto idx : c10::irange(sizes.size())) {
    if (
        sizes[idx] < 0 || logical_strides[idx] < 0 ||
        physical_strides[idx] < 0) {
      return false;
    }
    if (sizes[idx] == 0) {
      continue;
    }
    max_offset += (sizes[idx] - 1) * physical_strides[idx];
    if (max_offset < 0) {
      return false;
    }
  }

  return max_offset < v_input.buffer_length();
}

struct PackedWeightResidencyEntry final {
  Tensor weight_ref;
  std::optional<Tensor> bias_ref;
  int64_t weight_version;
  int64_t bias_version;
  std::vector<int64_t> logical_weight_sizes;
  PackedWeightKind kind;
  PackedWeightResidencyClass residency_class;
  bool quantized;
  uint64_t options_key;
  PackedWeightHandle handle;
};

constexpr size_t kPackedWeightResidencyMaxEntries = 256u;

size_t packed_weight_cache_limit_bytes() {
  static const size_t limit_bytes = []() {
    constexpr size_t kDefaultLimitBytes = size_t{2} * 1024u * 1024u * 1024u;
    const char* env =
        std::getenv("PYTORCH_VULKAN_PACKED_WEIGHT_CACHE_LIMIT_MB");
    if (!env || *env == '\0') {
      return kDefaultLimitBytes;
    }

    std::istringstream stream(env);
    size_t limit_mb = 0u;
    stream >> limit_mb;
    if (!stream || limit_mb == 0u) {
      return kDefaultLimitBytes;
    }
    return limit_mb * 1024u * 1024u;
  }();
  return limit_bytes;
}

const std::string& packed_weight_cache_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_PACKED_WEIGHT_CACHE_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool packed_weight_cache_logging_enabled() {
  return !packed_weight_cache_log_path().empty();
}

size_t packed_weight_tensor_nbytes(const Tensor& tensor) {
  if (!tensor.defined() || !tensor.is_vulkan()) {
    return 0u;
  }
  return static_cast<size_t>(convert(tensor).gpu_nbytes());
}

size_t packed_weight_handle_nbytes(const Tensor& weight, const Tensor& bias) {
  return packed_weight_tensor_nbytes(weight) + packed_weight_tensor_nbytes(bias);
}

api::ExecutionLayout resolve_buffer_execution_layout(const vTensor& v_tensor) {
  return v_tensor.has_direct_buffer_layout() ? api::ExecutionLayout::BUFFER_DIRECT
                                             : api::ExecutionLayout::BUFFER_VIEW;
}

struct PackedWeightResidencyLogState final {
  std::atomic<uint64_t> lookups{0u};
  std::atomic<uint64_t> hits{0u};
  std::atomic<uint64_t> stores{0u};
  std::atomic<uint64_t> evictions{0u};
  std::atomic<uint64_t> cache_bytes{0u};
  std::atomic<uint64_t> peak_cache_bytes{0u};
  std::atomic<uint64_t> persistent_cache_bytes{0u};
  std::atomic<uint64_t> peak_persistent_cache_bytes{0u};

  ~PackedWeightResidencyLogState() {
    if (!packed_weight_cache_logging_enabled()) {
      return;
    }

    std::ofstream out(packed_weight_cache_log_path(), std::ios::app);
    out << "packed_weight_residency: lookups="
        << lookups.load(std::memory_order_relaxed)
        << " hits=" << hits.load(std::memory_order_relaxed)
        << " stores=" << stores.load(std::memory_order_relaxed)
        << " evictions=" << evictions.load(std::memory_order_relaxed)
        << " cache_bytes=" << cache_bytes.load(std::memory_order_relaxed)
        << " peak_cache_bytes="
        << peak_cache_bytes.load(std::memory_order_relaxed)
        << " persistent_cache_bytes="
        << persistent_cache_bytes.load(std::memory_order_relaxed)
        << " peak_persistent_cache_bytes="
        << peak_persistent_cache_bytes.load(std::memory_order_relaxed)
        << " cache_limit_bytes=" << packed_weight_cache_limit_bytes() << '\n';
  }
};

PackedWeightResidencyLogState& packed_weight_cache_log_state() {
  static PackedWeightResidencyLogState state;
  return state;
}

class PackedWeightResidencyManager final {
 private:
  std::mutex mutex_;
  std::deque<PackedWeightResidencyEntry> cache_;
  size_t cache_bytes_{0u};
  size_t persistent_cache_bytes_{0u};

  static bool matches_entry(
      const PackedWeightResidencyEntry& entry,
      const Tensor& source_weight,
      const std::optional<Tensor>& normalized_bias,
      const int64_t weight_version,
      const int64_t bias_version,
      IntArrayRef logical_weight_sizes,
      const PackedWeightKind kind,
      const bool quantized,
      const uint64_t options_key) {
    return entry.weight_ref.unsafeGetTensorImpl() ==
            source_weight.unsafeGetTensorImpl() &&
        entry.weight_version == weight_version &&
        same_optional_tensor(entry.bias_ref, normalized_bias) &&
        entry.bias_version == bias_version &&
        entry.logical_weight_sizes.size() == logical_weight_sizes.size() &&
        std::equal(
            logical_weight_sizes.begin(),
            logical_weight_sizes.end(),
            entry.logical_weight_sizes.begin()) &&
        entry.kind == kind && entry.quantized == quantized &&
        entry.options_key == options_key;
  }

  void update_log_snapshot_locked() const {
    if (!packed_weight_cache_logging_enabled()) {
      return;
    }
    auto& log_state = packed_weight_cache_log_state();
    const auto cache_bytes = static_cast<uint64_t>(cache_bytes_);
    const auto persistent_cache_bytes =
        static_cast<uint64_t>(persistent_cache_bytes_);
    log_state.cache_bytes.store(cache_bytes, std::memory_order_relaxed);
    log_state.persistent_cache_bytes.store(
        persistent_cache_bytes, std::memory_order_relaxed);

    uint64_t observed_peak_cache =
        log_state.peak_cache_bytes.load(std::memory_order_relaxed);
    while (
        cache_bytes > observed_peak_cache &&
        !log_state.peak_cache_bytes.compare_exchange_weak(
            observed_peak_cache,
            cache_bytes,
            std::memory_order_relaxed,
            std::memory_order_relaxed)) {
    }

    uint64_t observed_peak_persistent =
        log_state.peak_persistent_cache_bytes.load(std::memory_order_relaxed);
    while (
        persistent_cache_bytes > observed_peak_persistent &&
        !log_state.peak_persistent_cache_bytes.compare_exchange_weak(
            observed_peak_persistent,
            persistent_cache_bytes,
            std::memory_order_relaxed,
            std::memory_order_relaxed)) {
    }
  }

  void erase_entry_locked(
      std::deque<PackedWeightResidencyEntry>::iterator entry_it,
      const bool count_eviction) {
    cache_bytes_ -= entry_it->handle.resident_nbytes();
    if (
        entry_it->residency_class ==
        PackedWeightResidencyClass::PersistentInference) {
      persistent_cache_bytes_ -= entry_it->handle.resident_nbytes();
    }
    cache_.erase(entry_it);
    if (count_eviction && packed_weight_cache_logging_enabled()) {
      packed_weight_cache_log_state().evictions.fetch_add(
          1u, std::memory_order_relaxed);
    }
  }

  std::deque<PackedWeightResidencyEntry>::iterator
  select_eviction_candidate_locked() {
    auto transient_it = cache_.end();
    for (auto it = cache_.end(); it != cache_.begin();) {
      --it;
      if (
          it->residency_class == PackedWeightResidencyClass::Transient &&
          it->handle.defined()) {
        transient_it = it;
        break;
      }
    }
    if (transient_it != cache_.end()) {
      return transient_it;
    }
    return cache_.empty() ? cache_.end() : std::prev(cache_.end());
  }

  void trim_locked() {
    while (
        cache_.size() > kPackedWeightResidencyMaxEntries ||
        cache_bytes_ > packed_weight_cache_limit_bytes()) {
      auto victim = select_eviction_candidate_locked();
      if (victim == cache_.end()) {
        break;
      }
      erase_entry_locked(victim, true);
    }
    update_log_snapshot_locked();
  }

 public:
  std::optional<PackedWeightHandle> lookup(
      const Tensor& source_weight,
      const std::optional<Tensor>& normalized_bias,
      IntArrayRef logical_weight_sizes,
      const PackedWeightKind kind,
      const bool quantized,
      const uint64_t options_key) {
    if (!source_weight.defined()) {
      return std::nullopt;
    }

    if (packed_weight_cache_logging_enabled()) {
      packed_weight_cache_log_state().lookups.fetch_add(
          1u, std::memory_order_relaxed);
    }

    const int64_t weight_version = tensor_version_or_zero(source_weight);
    const int64_t bias_version =
        normalized_bias ? tensor_version_or_zero(*normalized_bias) : 0u;

    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
      if (!matches_entry(
              *it,
              source_weight,
              normalized_bias,
              weight_version,
              bias_version,
              logical_weight_sizes,
              kind,
              quantized,
              options_key)) {
        continue;
      }

      PackedWeightHandle handle = it->handle;
      if (it != cache_.begin()) {
        PackedWeightResidencyEntry entry = std::move(*it);
        cache_.erase(it);
        cache_.emplace_front(std::move(entry));
        handle = cache_.front().handle;
      }

      if (packed_weight_cache_logging_enabled()) {
        packed_weight_cache_log_state().hits.fetch_add(
            1u, std::memory_order_relaxed);
      }
      update_log_snapshot_locked();
      return handle;
    }

    update_log_snapshot_locked();
    return std::nullopt;
  }

  void store(
      const Tensor& source_weight,
      const std::optional<Tensor>& normalized_bias,
      IntArrayRef logical_weight_sizes,
      const PackedWeightKind kind,
      const PackedWeightHandle& handle,
      const bool quantized,
      const uint64_t options_key) {
    if (!source_weight.defined() || !handle.defined()) {
      return;
    }

    PackedWeightResidencyEntry entry;
    entry.weight_ref = source_weight;
    entry.bias_ref = normalized_bias;
    entry.weight_version = tensor_version_or_zero(source_weight);
    entry.bias_version =
        normalized_bias ? tensor_version_or_zero(*normalized_bias) : 0u;
    entry.logical_weight_sizes = std::vector<int64_t>(
        logical_weight_sizes.begin(), logical_weight_sizes.end());
    entry.kind = kind;
    entry.residency_class = handle.residency_class();
    entry.quantized = quantized;
    entry.options_key = options_key;
    entry.handle = handle;

    if (packed_weight_cache_logging_enabled()) {
      packed_weight_cache_log_state().stores.fetch_add(
          1u, std::memory_order_relaxed);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
      if (!matches_entry(
              *it,
              source_weight,
              normalized_bias,
              entry.weight_version,
              entry.bias_version,
              logical_weight_sizes,
              kind,
              quantized,
              options_key)) {
        continue;
      }
      erase_entry_locked(it, false);
      break;
    }

    cache_bytes_ += handle.resident_nbytes();
    if (
        handle.residency_class() ==
        PackedWeightResidencyClass::PersistentInference) {
      persistent_cache_bytes_ += handle.resident_nbytes();
    }
    cache_.emplace_front(std::move(entry));
    trim_locked();
  }
};

PackedWeightResidencyManager& packed_weight_residency_manager() {
  static PackedWeightResidencyManager manager;
  return manager;
}

Tensor cast_vulkan_tensor_dtype_buffer_native(
    const Tensor& input_arg,
    const ScalarType dtype,
    const api::ShaderInfo& shader_descriptor) {
  api::AllocationScope allocation_scope("cast.buffer");
  api::Context* const context = api::context();

  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  vTensor v_input = convert(input);

  TORCH_CHECK(
      can_native_buffer_cast_input(v_input),
      "Native Vulkan buffer cast requires a float or int32-compatible buffer tensor");

  vTensor v_out{
      context,
      v_input.sizes(),
      convert_dtype(dtype),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(v_out.numel()),
      1u,
      1u,
  };
  api::UniformParamsBuffer out_meta =
      make_buffer_compute_metadata_ubo(context, v_out);
  api::UniformParamsBuffer in_meta =
      make_buffer_compute_metadata_ubo(context, v_input);

  context->submit_compute_job(
      shader_descriptor,
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_out.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer());

  return convert(v_out);
}

const std::string& materialize_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_MATERIALIZE_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool materialize_logging_enabled() {
  return !materialize_log_path().empty();
}

const char* storage_type_name(const api::StorageType storage_type) {
  switch (storage_type) {
    case api::StorageType::TEXTURE_3D:
      return "TEXTURE_3D";
    case api::StorageType::TEXTURE_2D:
      return "TEXTURE_2D";
    case api::StorageType::BUFFER:
      return "BUFFER";
    case api::StorageType::UNKNOWN:
      return "UNKNOWN";
  }
  return "UNKNOWN";
}

const char* memory_layout_name(const api::GPUMemoryLayout memory_layout) {
  switch (memory_layout) {
    case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
      return "TENSOR_WIDTH_PACKED";
    case api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED:
      return "TENSOR_HEIGHT_PACKED";
    case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
      return "TENSOR_CHANNELS_PACKED";
  }
  return "UNKNOWN";
}

std::string format_sizes(const std::vector<int64_t>& sizes) {
  std::ostringstream stream;
  stream << "[";
  for (size_t idx = 0; idx < sizes.size(); ++idx) {
    if (idx > 0) {
      stream << ",";
    }
    stream << sizes[idx];
  }
  stream << "]";
  return stream.str();
}

void append_materialize_log_line(const std::string& line) {
  if (!materialize_logging_enabled()) {
    return;
  }

  std::ofstream out(materialize_log_path(), std::ios::app);
  out << line << '\n';
}

void log_materialize_event(
    const char* kind,
    const vTensor& v_in,
    const api::StorageType dst_storage_type,
    const api::GPUMemoryLayout dst_memory_layout,
    const char* path) {
  if (!materialize_logging_enabled()) {
    return;
  }

  std::ostringstream stream;
  stream << "kind=" << kind
         << " caller=" << api::current_allocation_label()
         << " path=" << path
         << " exec_layout=" << execution_layout_name(v_in.execution_layout())
         << " src_storage=" << storage_type_name(v_in.storage_type())
         << " src_layout=" << memory_layout_name(v_in.gpu_memory_layout())
         << " dst_storage=" << storage_type_name(dst_storage_type)
         << " dst_layout=" << memory_layout_name(dst_memory_layout)
         << " direct_buffer=" << (v_in.has_direct_buffer_layout() ? 1 : 0)
         << " storage_offset=" << v_in.storage_offset()
         << " logical_bytes=" << v_in.nbytes()
         << " gpu_bytes=" << v_in.gpu_nbytes()
         << " sizes=" << format_sizes(v_in.sizes());
  append_materialize_log_line(stream.str());
}

Tensor materialize_inference_vulkan_matrix_arg(const Tensor& tensor) {
  if (
      c10::InferenceMode::is_enabled() &&
      tensor.is_vulkan() &&
      tensor.dim() == 2 &&
      !tensor.is_contiguous_or_false()) {
    return tensor.t().clone().t();
  }
  return tensor;
}

constexpr size_t kVulkanExecutionPlanKindCount =
    static_cast<size_t>(VulkanExecutionPlanKind::NumKinds);

size_t execution_plan_kind_index(const VulkanExecutionPlanKind kind) {
  const size_t idx = static_cast<size_t>(kind);
  TORCH_INTERNAL_ASSERT(
      idx < kVulkanExecutionPlanKindCount,
      "Invalid VulkanExecutionPlanKind");
  return idx;
}

const std::array<VulkanExecutionPlanPolicy, kVulkanExecutionPlanKindCount>&
execution_plan_policies() {
  static const std::array<VulkanExecutionPlanPolicy, kVulkanExecutionPlanKindCount>
      policies{{
          {"Generic",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::Never,
           VulkanExecutionPolicyMemoryRule::Fixed,
           true,
           false,
           false,
           false,
           false},
          {"TextureComputeInput",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::Never,
           VulkanExecutionPolicyMemoryRule::Fixed,
           true,
           false,
           false,
           false,
           false},
          {"ElementwiseInput",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::PreferElementwiseBuffer,
           VulkanExecutionPolicyMemoryRule::Fixed,
           true,
           false,
           false,
           false,
           false},
          {"ElementwiseBufferInput",
           api::ExecutionLayout::BUFFER_DIRECT,
           api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
           api::StorageType::BUFFER,
           VulkanExecutionPolicyBufferRule::RequireElementwiseBuffer,
           VulkanExecutionPolicyMemoryRule::Fixed,
           true,
           false,
           false,
           false,
           false},
          {"ReductionAllInput",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::PreferReductionBuffer,
           VulkanExecutionPolicyMemoryRule::Fixed,
           true,
           false,
           false,
           false,
           false},
          {"ReductionDimInput",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::PreferReductionBuffer,
           VulkanExecutionPolicyMemoryRule::Fixed,
           true,
           false,
           false,
           false,
           false},
          {"LinearInputSource",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::Never,
           VulkanExecutionPolicyMemoryRule::LinearInputSource,
           false,
           true,
           true,
           false,
           false},
          {"LinearWeightSource",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::Never,
           VulkanExecutionPolicyMemoryRule::Fixed,
           false,
           true,
           true,
           true,
           false},
          {"LinearBiasSource",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::Never,
           VulkanExecutionPolicyMemoryRule::Fixed,
           false,
           true,
           true,
           false,
           false},
          {"LinearPackedBias",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::Never,
           VulkanExecutionPolicyMemoryRule::Fixed,
           true,
           false,
           false,
           false,
           false},
          {"LinearPackedInput",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::Never,
           VulkanExecutionPolicyMemoryRule::Fixed,
           true,
           false,
           false,
           false,
           false},
          {"LinearPackedWeight",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::Never,
           VulkanExecutionPolicyMemoryRule::Fixed,
           true,
           false,
           false,
           false,
           false},
          {"Conv2dWeightSource",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::Never,
           VulkanExecutionPolicyMemoryRule::Fixed,
           false,
           true,
           true,
           false,
           false},
          {"Conv2dBiasSource",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::Never,
           VulkanExecutionPolicyMemoryRule::Fixed,
           false,
           true,
           true,
           false,
           false},
          {"Conv2dRuntimeInput",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::Never,
           VulkanExecutionPolicyMemoryRule::Fixed,
           true,
           false,
           true,
           false,
           false},
          {"Conv1dPrepackWeight",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::Never,
           VulkanExecutionPolicyMemoryRule::Fixed,
           true,
           false,
           true,
           false,
           false},
          {"Conv1dPrepackBias",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::Never,
           VulkanExecutionPolicyMemoryRule::Fixed,
           true,
           false,
           true,
           false,
           false},
          {"Conv1dRuntimeInput",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::Never,
           VulkanExecutionPolicyMemoryRule::Fixed,
           true,
           false,
           true,
           false,
           false},
          {"Conv1dRuntimeWeight",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::Never,
           VulkanExecutionPolicyMemoryRule::Fixed,
           true,
           false,
           true,
           false,
           false},
          {"Conv1dRuntimeBias",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::Never,
           VulkanExecutionPolicyMemoryRule::Fixed,
           true,
           false,
           true,
           false,
           false},
      }};
  return policies;
}

const std::string& execution_plan_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_EXECUTION_PLAN_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool execution_plan_logging_enabled() {
  return !execution_plan_log_path().empty();
}

struct ExecutionPlanLogState final {
  std::array<std::atomic<uint64_t>, kVulkanExecutionPlanKindCount> builds{};
  std::array<std::atomic<uint64_t>, kVulkanExecutionPlanKindCount> executes{};
  std::array<std::atomic<uint64_t>, kVulkanExecutionPlanKindCount> passthrough{};
  std::array<std::atomic<uint64_t>, kVulkanExecutionPlanKindCount> texture{};
  std::array<std::atomic<uint64_t>, kVulkanExecutionPlanKindCount> buffer_direct{};
  std::array<std::atomic<uint64_t>, kVulkanExecutionPlanKindCount> buffer_view{};
  std::array<std::atomic<uint64_t>, kVulkanExecutionPlanKindCount> packed_weight{};
  std::array<std::atomic<uint64_t>, kVulkanExecutionPlanKindCount>
      widened_bfloat16{};
  std::array<std::atomic<uint64_t>, kVulkanExecutionPlanKindCount>
      inference_materializations{};
  std::array<std::atomic<uint64_t>, kVulkanExecutionPlanKindCount>
      buffer_materializations{};
  std::array<std::atomic<uint64_t>, kVulkanExecutionPlanKindCount>
      texture_materializations{};

  ~ExecutionPlanLogState() {
    if (!execution_plan_logging_enabled()) {
      return;
    }

    std::ofstream out(execution_plan_log_path(), std::ios::app);
    uint64_t total_builds = 0u;
    uint64_t total_executes = 0u;
    for (const auto idx : c10::irange(kVulkanExecutionPlanKindCount)) {
      const auto build_count = builds[idx].load(std::memory_order_relaxed);
      const auto execute_count = executes[idx].load(std::memory_order_relaxed);
      const auto passthrough_count =
          passthrough[idx].load(std::memory_order_relaxed);
      const auto texture_count = texture[idx].load(std::memory_order_relaxed);
      const auto buffer_direct_count =
          buffer_direct[idx].load(std::memory_order_relaxed);
      const auto buffer_view_count =
          buffer_view[idx].load(std::memory_order_relaxed);
      const auto packed_weight_count =
          packed_weight[idx].load(std::memory_order_relaxed);
      const auto widened_count =
          widened_bfloat16[idx].load(std::memory_order_relaxed);
      const auto inference_materialize_count =
          inference_materializations[idx].load(std::memory_order_relaxed);
      const auto buffer_materialize_count =
          buffer_materializations[idx].load(std::memory_order_relaxed);
      const auto texture_materialize_count =
          texture_materializations[idx].load(std::memory_order_relaxed);
      if (
          build_count == 0u && execute_count == 0u && passthrough_count == 0u &&
          texture_count == 0u && buffer_direct_count == 0u &&
          buffer_view_count == 0u && packed_weight_count == 0u &&
          widened_count == 0u && inference_materialize_count == 0u &&
          buffer_materialize_count == 0u &&
          texture_materialize_count == 0u) {
        continue;
      }

      const auto kind =
          static_cast<VulkanExecutionPlanKind>(safe_downcast<uint8_t>(idx));
      out << "execution_plan kind=" << execution_plan_kind_name(kind)
          << " builds=" << build_count << " executes=" << execute_count
          << " passthrough=" << passthrough_count
          << " texture=" << texture_count
          << " buffer_direct=" << buffer_direct_count
          << " buffer_view=" << buffer_view_count
          << " packed_weight=" << packed_weight_count
          << " widened_bfloat16=" << widened_count
          << " inference_materializations=" << inference_materialize_count
          << " buffer_materializations=" << buffer_materialize_count
          << " texture_materializations=" << texture_materialize_count
          << '\n';
      total_builds += build_count;
      total_executes += execute_count;
    }

    out << "execution_plan_summary builds=" << total_builds
        << " executes=" << total_executes << '\n';
  }
};

ExecutionPlanLogState& execution_plan_log_state() {
  static ExecutionPlanLogState state;
  return state;
}

void log_execution_plan_build(const VulkanExecutionPlanKind kind) {
  if (!execution_plan_logging_enabled()) {
    return;
  }
  execution_plan_log_state().builds[execution_plan_kind_index(kind)].fetch_add(
      1u, std::memory_order_relaxed);
}

void log_execution_plan_execute(
    const VulkanExecutionPlan& plan,
    const bool passthrough,
    const bool widened_bfloat16,
    const bool materialized_inference_matrix,
    const bool materialized_buffer,
    const bool materialized_texture,
    const std::optional<api::ExecutionLayout>& actual_layout) {
  if (!execution_plan_logging_enabled()) {
    return;
  }

  auto& state = execution_plan_log_state();
  const size_t idx = execution_plan_kind_index(plan.kind);
  state.executes[idx].fetch_add(1u, std::memory_order_relaxed);
  if (passthrough) {
    state.passthrough[idx].fetch_add(1u, std::memory_order_relaxed);
  }
  if (widened_bfloat16) {
    state.widened_bfloat16[idx].fetch_add(1u, std::memory_order_relaxed);
  }
  if (materialized_inference_matrix) {
    state.inference_materializations[idx].fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (materialized_buffer) {
    state.buffer_materializations[idx].fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (materialized_texture) {
    state.texture_materializations[idx].fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (!actual_layout.has_value()) {
    return;
  }

  switch (*actual_layout) {
    case api::ExecutionLayout::TEXTURE:
      state.texture[idx].fetch_add(1u, std::memory_order_relaxed);
      break;
    case api::ExecutionLayout::BUFFER_DIRECT:
      state.buffer_direct[idx].fetch_add(1u, std::memory_order_relaxed);
      break;
    case api::ExecutionLayout::BUFFER_VIEW:
      state.buffer_view[idx].fetch_add(1u, std::memory_order_relaxed);
      break;
    case api::ExecutionLayout::PACKED_WEIGHT:
      state.packed_weight[idx].fetch_add(1u, std::memory_order_relaxed);
      break;
  }
}

api::GPUMemoryLayout resolve_execution_plan_memory_layout(
    const Tensor& tensor,
    const VulkanExecutionPlanPolicy& policy) {
  switch (policy.memory_rule) {
    case VulkanExecutionPolicyMemoryRule::Fixed:
      return policy.memory_layout;
    case VulkanExecutionPolicyMemoryRule::LinearInputSource:
      return tensor.dim() == 2 ? api::GPUMemoryLayout::TENSOR_WIDTH_PACKED
                               : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;
  }

  TORCH_CHECK(false, "Unsupported Vulkan execution plan memory rule");
}

std::optional<api::ExecutionLayout> select_buffer_execution_layout(
    const Tensor& tensor,
    const VulkanExecutionPlanPolicy& policy) {
  if (!tensor.is_vulkan()) {
    return std::nullopt;
  }

  const vTensor& v_tensor = convert(tensor);
  if (v_tensor.storage_type() != api::StorageType::BUFFER) {
    return std::nullopt;
  }

  switch (policy.buffer_rule) {
    case VulkanExecutionPolicyBufferRule::Never:
      return std::nullopt;
    case VulkanExecutionPolicyBufferRule::PreferElementwiseBuffer:
      if (tensor.scalar_type() != c10::ScalarType::Float) {
        return std::nullopt;
      }
      [[fallthrough]];
    case VulkanExecutionPolicyBufferRule::RequireElementwiseBuffer:
      if (supports_buffer_elementwise_compute(v_tensor)) {
        return resolve_buffer_execution_layout(v_tensor);
      }
      return std::nullopt;
    case VulkanExecutionPolicyBufferRule::PreferReductionBuffer:
      if (supports_buffer_reduction_compute(v_tensor)) {
        return resolve_buffer_execution_layout(v_tensor);
      }
      return std::nullopt;
  }

  TORCH_CHECK(false, "Unsupported Vulkan execution plan buffer rule");
}

bool needs_buffer_storage_transition(
    const Tensor& input,
    const api::GPUMemoryLayout memory_layout) {
  if (!input.is_vulkan()) {
    return true;
  }

  const vTensor& v_input = convert(input);
  return !(
      v_input.storage_type() == api::StorageType::BUFFER &&
      v_input.gpu_memory_layout() == memory_layout &&
      v_input.has_direct_buffer_layout());
}

bool needs_texture_storage_transition(
    const Tensor& input,
    const api::GPUMemoryLayout memory_layout,
    const api::StorageType storage_type) {
  if (!input.is_vulkan()) {
    return true;
  }

  const vTensor& v_input = convert(input);
  return !(
      v_input.storage_type() == storage_type &&
      v_input.gpu_memory_layout() == memory_layout);
}

} // namespace

const char* execution_layout_name(const api::ExecutionLayout execution_layout) {
  return api::to_string(execution_layout);
}

const char* execution_plan_kind_name(const VulkanExecutionPlanKind kind) {
  return execution_plan_policy(kind).name;
}

const VulkanExecutionPlanPolicy& execution_plan_policy(
    const VulkanExecutionPlanKind kind) {
  return execution_plan_policies()[execution_plan_kind_index(kind)];
}

std::optional<Tensor> normalized_optional_tensor(
    const std::optional<Tensor>& tensor) {
  if (tensor && tensor->defined()) {
    return tensor;
  }
  return std::nullopt;
}

bool same_optional_tensor(
    const std::optional<Tensor>& lhs,
    const std::optional<Tensor>& rhs) {
  if (lhs.has_value() != rhs.has_value()) {
    return false;
  }
  if (!lhs.has_value()) {
    return true;
  }
  return lhs->unsafeGetTensorImpl() == rhs->unsafeGetTensorImpl();
}

int64_t tensor_version_or_zero(const Tensor& tensor) {
  return tensor.is_inference() ? 0 : tensor._version();
}

bool has_inference_tensor(
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  const auto normalized_bias = normalized_optional_tensor(bias);
  return weight.is_inference() ||
      (normalized_bias && normalized_bias->is_inference());
}

VulkanExecutionPlan build_vulkan_execution_plan(
    const Tensor& tensor,
    const VulkanExecutionPlanKind kind) {
  const auto& policy = execution_plan_policy(kind);
  VulkanExecutionPlan plan;
  plan.kind = kind;
  plan.execution_layout = policy.execution_layout;
  plan.memory_layout = resolve_execution_plan_memory_layout(tensor, policy);
  plan.storage_type = policy.storage_type;
  plan.force_storage =
      policy.force_storage ||
      (policy.force_storage_if_widen_bfloat16 &&
       tensor.scalar_type() == c10::ScalarType::BFloat16);
  plan.widen_bfloat16 =
      policy.widen_bfloat16 && tensor.scalar_type() == c10::ScalarType::BFloat16;
  plan.materialize_inference_matrix = policy.materialize_inference_matrix;
  plan.persistent = policy.persistent;

  if (
      const auto buffer_execution_layout =
          select_buffer_execution_layout(tensor, policy)) {
    plan.execution_layout = *buffer_execution_layout;
    plan.storage_type = api::StorageType::BUFFER;
    plan.memory_layout = api::GPUMemoryLayout::TENSOR_WIDTH_PACKED;
  }

  log_execution_plan_build(kind);
  return plan;
}

Tensor execute_vulkan_execution_plan(
    const Tensor& input_arg,
    const VulkanExecutionPlan& plan) {
  Tensor input = input_arg;
  const bool should_materialize_inference_matrix =
      plan.materialize_inference_matrix &&
      c10::InferenceMode::is_enabled() && input.is_vulkan() && input.dim() == 2 &&
      !input.is_contiguous_or_false();
  if (plan.materialize_inference_matrix) {
    input = materialize_inference_vulkan_matrix_arg(input);
  }

  const bool should_widen_bfloat16 =
      plan.widen_bfloat16 && input.scalar_type() == kBFloat16;
  if (plan.widen_bfloat16 && input.scalar_type() == kBFloat16) {
    if (input.is_vulkan()) {
      input = convert(input).storage_type() == api::StorageType::BUFFER
          ? upcast_bfloat16_buffer_to_float(input)
          : input.cpu().to(kFloat).vulkan();
    } else {
      input = input.to(kFloat);
    }
  }

  if (!plan.force_storage) {
    log_execution_plan_execute(
        plan,
        true,
        should_widen_bfloat16,
        should_materialize_inference_matrix,
        false,
        false,
        input.is_vulkan()
            ? std::optional<api::ExecutionLayout>(convert(input).execution_layout())
            : std::nullopt);
    return input;
  }

  if (!input.is_vulkan()) {
    input = input.vulkan();
  }

  switch (plan.execution_layout) {
    case api::ExecutionLayout::TEXTURE: {
      const bool materialized_texture = needs_texture_storage_transition(
          input, plan.memory_layout, plan.storage_type);
      Tensor output = mark_tensor_execution(
          ensure_texture_storage(input, plan.memory_layout, plan.storage_type),
          api::ExecutionLayout::TEXTURE,
          plan.persistent);
      log_execution_plan_execute(
          plan,
          false,
          should_widen_bfloat16,
          should_materialize_inference_matrix,
          false,
          materialized_texture,
          api::ExecutionLayout::TEXTURE);
      return output;
    }
    case api::ExecutionLayout::BUFFER_DIRECT: {
      const bool materialized_buffer =
          needs_buffer_storage_transition(input, plan.memory_layout);
      Tensor output = mark_tensor_execution(
          ensure_buffer_storage(input, plan.memory_layout),
          api::ExecutionLayout::BUFFER_DIRECT,
          plan.persistent);
      log_execution_plan_execute(
          plan,
          false,
          should_widen_bfloat16,
          should_materialize_inference_matrix,
          materialized_buffer,
          false,
          api::ExecutionLayout::BUFFER_DIRECT);
      return output;
    }
    case api::ExecutionLayout::BUFFER_VIEW:
      if (!input.is_vulkan()) {
        input = input.vulkan();
      }
      if (input.is_vulkan()) {
        const vTensor& v_input = convert(input);
        if (
            v_input.storage_type() == api::StorageType::BUFFER &&
            v_input.gpu_memory_layout() == plan.memory_layout &&
            supports_buffer_view_fast_path(v_input)) {
          Tensor output = mark_tensor_execution(
              input,
              resolve_buffer_execution_layout(v_input),
              plan.persistent);
          log_execution_plan_execute(
              plan,
              false,
              should_widen_bfloat16,
              should_materialize_inference_matrix,
              false,
              false,
              convert(output).execution_layout());
          return output;
        }
      }
      input = mark_tensor_execution(
          ensure_buffer_storage(input, plan.memory_layout),
          api::ExecutionLayout::BUFFER_DIRECT,
          plan.persistent);
      log_execution_plan_execute(
          plan,
          false,
          should_widen_bfloat16,
          should_materialize_inference_matrix,
          true,
          false,
          api::ExecutionLayout::BUFFER_DIRECT);
      return input;
    case api::ExecutionLayout::PACKED_WEIGHT: {
      const bool materialized_texture = needs_texture_storage_transition(
          input, plan.memory_layout, plan.storage_type);
      input = mark_tensor_execution(
          ensure_texture_storage(input, plan.memory_layout, plan.storage_type),
          api::ExecutionLayout::PACKED_WEIGHT,
          plan.persistent);
      log_execution_plan_execute(
          plan,
          false,
          should_widen_bfloat16,
          should_materialize_inference_matrix,
          false,
          materialized_texture,
          api::ExecutionLayout::PACKED_WEIGHT);
      return input;
    }
  }

  TORCH_CHECK(false, "Unsupported Vulkan execution layout");
}

Tensor prepare_vulkan_direct_buffer_execution_tensor(
    const Tensor& input,
    const VulkanExecutionPlan& plan) {
  TORCH_CHECK(
      api::uses_buffer_execution(plan.execution_layout),
      "Vulkan direct buffer execution requires a buffer execution plan");

  Tensor prepared = execute_vulkan_execution_plan(input, plan);
  const vTensor& v_prepared = convert(prepared);
  if (
      v_prepared.storage_type() == api::StorageType::BUFFER &&
      v_prepared.gpu_memory_layout() == plan.memory_layout &&
      v_prepared.has_direct_buffer_layout()) {
    return mark_tensor_execution(
        prepared, api::ExecutionLayout::BUFFER_DIRECT, plan.persistent);
  }

  return mark_tensor_execution(
      ensure_buffer_storage(prepared, plan.memory_layout),
      api::ExecutionLayout::BUFFER_DIRECT,
      plan.persistent);
}

Tensor prepare_vulkan_direct_buffer_execution_tensor(
    const Tensor& input,
    const VulkanExecutionPlanKind kind) {
  const VulkanExecutionPlan plan = build_vulkan_execution_plan(input, kind);
  return prepare_vulkan_direct_buffer_execution_tensor(input, plan);
}

Tensor prepare_vulkan_execution_tensor(
    const Tensor& input,
    const VulkanExecutionPlanKind kind) {
  return execute_vulkan_execution_plan(
      input, build_vulkan_execution_plan(input, kind));
}

std::optional<Tensor> prepare_optional_vulkan_execution_tensor(
    const std::optional<Tensor>& input,
    const VulkanExecutionPlanKind kind) {
  if (!input || !input->defined()) {
    return std::nullopt;
  }

  return prepare_vulkan_execution_tensor(*input, kind);
}

LogicalBufferMetadata make_buffer_compute_metadata(const vTensor& tensor) {
  return {
      api::utils::make_whcn_uvec4(tensor.logical_sizes()),
      api::utils::make_whcn_uvec4(tensor.logical_strides()),
      api::utils::make_whcn_uvec4(tensor.physical_strides()),
      {
          api::utils::safe_downcast<uint32_t>(tensor.logical_sizes().size()),
          api::utils::safe_downcast<uint32_t>(tensor.numel()),
          api::utils::safe_downcast<uint32_t>(tensor.buffer_length()),
          api::utils::safe_downcast<uint32_t>(tensor.storage_offset()),
      },
  };
}

api::UniformParamsBuffer make_buffer_compute_metadata_ubo(
    api::Context* const context,
    const vTensor& tensor) {
  return api::UniformParamsBuffer(context, make_buffer_compute_metadata(tensor));
}

/*
 * This function formats an input tensor in NCHW layout to NC4HW layout such
 * that the buffer of the formatted tensor can be directly copied into a GPU
 * texture. Conceptually, the formatting can be achieved via the following
 * steps:
 *
 * 1. Given that the src tensor has size {N,C,H,W}
 *
 * 2. Combine the batch and channel dims by reshaping to {N*C, H, W}
 *
 * 3. Determine the amount of padding to add: determine how many channels to add
 *    in order to align N*C to the next multiple of 4
 *
 * 4. Add padding to the tensor so that the batch-channel dimension is a
 *    multiple of four; the shape of the tensor is now {NC_aligned, H, W}
 *
 * 5. Split the batch-channel dimension into groups of 4 by reshaping the tensor
 *    to size {NC_aligned/4, 4, H, W}
 *
 * 6. The groups of 4 channels (dim 1) should be contiguous. Therefore, permute
 *    the dims of the tensor in the order {0, 2, 3, 1}
 *
 * 7. Finally, return a contiguous version of the tensor. The final shape of the
 *    tensor would be {NC_aligned/4, H, W, 4}
 */
Tensor nchw_to_nc4hw(const Tensor& src) {
  uint32_t N = get_dim<Dim4D::Batch>(src.sizes());
  uint32_t C = get_dim<Dim4D::Channel>(src.sizes());
  uint32_t H = get_dim<Dim4D::Height>(src.sizes());
  uint32_t W = get_dim<Dim4D::Width>(src.sizes());

  uint32_t C_aligned = api::utils::align_up(C, 4u);
  uint32_t NC4 = (N * C_aligned) / 4;

  // Add padding to the tensor so that the channel dim is a multiple of 4
  Tensor padding = at::zeros({N, C_aligned - C, H, W}, src.options());
  Tensor src_padded = at::cat({src.reshape({N, C, H, W}), padding}, 1);
  // Reshape to group channels into groups of 4 and permute so that the groups
  // are in the first dimension so that they are contiguous
  Tensor src_NC4HW = src_padded.reshape({NC4, 4, H, W}).permute({0, 2, 3, 1});

  // Return a contiguous version of the tensor
  return src_NC4HW.contiguous();
}

/*
 * Creates a staging tensor into which texture data, which will be in NC4HW
 * format, can be copied directly. The shape of the staging tensor will be the
 * same as the tensor produced by a call to format_src_tensor().
 */
Tensor create_staging_tensor(const vTensor& v_in) {
  uint32_t N = get_dim<Dim4D::Batch>(v_in.sizes());
  uint32_t C = get_dim<Dim4D::Channel>(v_in.sizes());
  uint32_t H = get_dim<Dim4D::Height>(v_in.sizes());
  uint32_t W = get_dim<Dim4D::Width>(v_in.sizes());

  uint32_t NC4 = N * api::utils::div_up(C, 4u);

  // Note that the dtype corresponding with the texture format of the vTensor is
  // used instead of options().dtype(). This is to ensure the number of bytes in
  // the staging tensor matches the number of bytes in the image texture. Refer
  // to comments for api::vk_format()
  return at::empty(
      {NC4, H, W, 4},
      at::device(at::kCPU).dtype(convert_dtype(v_in.texture_dtype())));
}

/*
 * After copying texture data, which will be in NC4HW format, to a staging
 * tensor created in create_staging_tensor(), this function reformats the tensor
 * to NCHW format. It essentially reverses the transformations made by
 * format_src_tensor().
 *
 * Note that the sizes of the original tensor must be passed in to fully restore
 * the properties of the original tensor.
 */
Tensor nc4hw_to_nchw(const Tensor& t_in, IntArrayRef sizes) {
  uint32_t N = get_dim<Dim4D::Batch>(sizes);
  uint32_t C = get_dim<Dim4D::Channel>(sizes);
  uint32_t H = get_dim<Dim4D::Height>(sizes);
  uint32_t W = get_dim<Dim4D::Width>(sizes);

  uint32_t C_aligned = api::utils::align_up(C, 4u);

  // Undo the permute step and channel grouping step
  Tensor t_in_padded = t_in.permute({0, 3, 1, 2}).reshape({N, C_aligned, H, W});
  // Remove the padding channels
  Tensor t_in_shaved =
      at::narrow(t_in_padded, /*dim=*/1, /*start*/ 0, /*end*/ C);

  // Reshape to original sizing and dtype and return a contiguous Tensor
  return t_in_shaved.reshape(sizes).contiguous();
}

bool supports_buffer_view_fast_path(const vTensor& v_in) {
  return api::supports_generic_buffer_view_ops(
      v_in.dtype(), v_in.sizes().size(), v_in.is_quantized());
}

bool uses_buffer_execution(const vTensor& v_in) {
  return v_in.uses_buffer_execution();
}

bool uses_texture_execution(const vTensor& v_in) {
  return !v_in.uses_buffer_execution();
}

bool supports_buffer_elementwise_compute(const vTensor& v_in) {
  return supports_buffer_view_fast_path(v_in);
}

bool supports_buffer_reduction_compute(const vTensor& v_in) {
  return supports_buffer_view_fast_path(v_in) &&
      (v_in.dtype() == api::kFloat || v_in.dtype() == api::kBFloat16);
}

bool scalar_fits_vulkan_int32(const Scalar& scalar) {
  if (!scalar.isIntegral(true)) {
    return false;
  }
  const int64_t value = scalar.to<int64_t>();
  return value >= static_cast<int64_t>(std::numeric_limits<int32_t>::min()) &&
      value <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

int32_t scalar_to_vulkan_int32(const Scalar& scalar) {
  return safe_downcast<int32_t>(scalar.to<int64_t>());
}

bool last_dim_is_width_aligned(const Tensor& tensor) {
  return tensor.dim() == 0 || tensor.sizes().back() % 4 == 0;
}

bool supports_native_integral_buffer_compute_dtype(const api::ScalarType dtype) {
  switch (dtype) {
    case api::kInt:
      return true;
    case api::kByte:
    case api::kChar:
      return api::context()->adapter_ptr()->supports_int8_buffer_arithmetic();
    default:
      return false;
  }
}

bool supports_native_integral_buffer_compute(const Tensor& tensor) {
  if (!tensor.is_vulkan()) {
    return false;
  }
  const vTensor& v_tensor = convert(tensor);
  return supports_native_integral_buffer_compute_dtype(v_tensor.dtype()) &&
      v_tensor.storage_type() == api::StorageType::BUFFER &&
      v_tensor.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
      supports_buffer_elementwise_compute(v_tensor) && !v_tensor.is_quantized();
}

bool supports_native_bool_buffer_compute(const Tensor& tensor) {
  if (!tensor.is_vulkan()) {
    return false;
  }
  const vTensor& v_tensor = convert(tensor);
  return api::context()->adapter_ptr()->supports_int8_buffer_arithmetic() &&
      v_tensor.dtype() == api::kBool &&
      supports_buffer_elementwise_compute(v_tensor) && !v_tensor.is_quantized();
}

bool can_make_buffer_metadata_view(
    const vTensor& v_in,
    IntArrayRef sizes,
    IntArrayRef logical_strides,
    IntArrayRef physical_strides,
    int64_t storage_offset) {
  return can_make_buffer_metadata_view_impl(
      v_in, sizes, logical_strides, physical_strides, storage_offset);
}

Tensor make_buffer_metadata_view(
    const Tensor& input_arg,
    IntArrayRef sizes,
    IntArrayRef logical_strides,
    IntArrayRef physical_strides,
    int64_t storage_offset) {
  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  TORCH_CHECK(
      can_make_buffer_metadata_view_impl(
          v_input,
          sizes,
          logical_strides,
          physical_strides,
          storage_offset),
      "Vulkan buffer metadata view requires a float buffer-backed tensor with "
      "supported logical sizes/strides and in-range storage_offset");

  log_materialize_event(
      "make_buffer_metadata_view",
      v_input,
      api::StorageType::BUFFER,
      v_input.gpu_memory_layout(),
      "metadata_view");

  return convert(vTensor{
      v_input,
      sizes.vec(),
      logical_strides.vec(),
      physical_strides.vec(),
      storage_offset,
  });
}

std::string describe_buffer_view_fast_path_failure(const vTensor& v_in) {
  std::ostringstream stream;
  stream
      << "Vulkan texture materialization from buffer views currently only "
      << "supports non-quantized float tensors with up to 4 dimensions"
      << " (caller=" << api::current_allocation_label()
      << ", sizes=" << format_sizes(v_in.sizes())
      << ", ndim=" << v_in.sizes().size()
      << ", dtype=" << api::to_string(v_in.dtype())
      << ", quantized=" << (v_in.is_quantized() ? 1 : 0)
      << ", exec_layout=" << execution_layout_name(v_in.execution_layout())
      << ", storage=" << storage_type_name(v_in.storage_type())
      << ", layout=" << memory_layout_name(v_in.gpu_memory_layout())
      << ", direct_buffer=" << (v_in.has_direct_buffer_layout() ? 1 : 0)
      << ")";
  return stream.str();
}

vTensor materialize_to_contiguous_buffer(
    const vTensor& v_in,
    api::GPUMemoryLayout memory_layout) {
  TORCH_CHECK(
      supports_buffer_view_fast_path(v_in),
      describe_buffer_view_fast_path_failure(v_in));

  if (
      v_in.storage_type() == api::StorageType::BUFFER &&
      v_in.gpu_memory_layout() == memory_layout &&
      v_in.has_direct_buffer_layout()) {
    return v_in;
  }

  log_materialize_event(
      "materialize_to_contiguous_buffer",
      v_in,
      api::StorageType::BUFFER,
      memory_layout,
      "materialize");

  api::Context* const context = api::context();
  vTensor v_out{
      context,
      v_in.sizes(),
      v_in.dtype(),
      api::StorageType::BUFFER,
      memory_layout,
  };

  if (
      v_in.storage_type() == api::StorageType::BUFFER &&
      v_in.dtype() == api::kFloat) {
    vTensor v_src = v_in;
    api::PipelineBarrier pipeline_barrier{};
    const api::utils::uvec3 global_size = {
        api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_out.numel(), 1)),
        1u,
        1u,
    };
    context->submit_compute_job(
        VK_KERNEL(buffer_to_buffer),
        pipeline_barrier,
        global_size,
        adaptive_work_group_size(global_size),
        VK_NULL_HANDLE,
        v_out.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        make_buffer_compute_metadata_ubo(context, v_out).buffer(),
        v_src.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        make_buffer_compute_metadata_ubo(context, v_src).buffer());
    return v_out;
  }

  api::StorageBuffer staging(context, v_in.dtype(), v_in.numel());
  vTensor v_src = v_in;
  pack_vtensor_to_staging(v_src, staging.buffer());

  api::PipelineBarrier pipeline_barrier{};
  add_buffer_barrier(
      pipeline_barrier,
      staging.buffer(),
      api::PipelineStage::COMPUTE | api::PipelineStage::TRANSFER,
      api::MemoryAccessType::WRITE,
      api::PipelineStage::COMPUTE | api::PipelineStage::TRANSFER,
      api::MemoryAccessType::READ);
  pack_buffer_to_vtensor(staging.buffer(), v_out, pipeline_barrier);
  return v_out;
}

Tensor ensure_buffer_storage(
    const Tensor& input_arg,
    api::GPUMemoryLayout memory_layout) {
  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  vTensor v_input = convert(input);

  if (
      v_input.storage_type() == api::StorageType::BUFFER &&
      v_input.gpu_memory_layout() == memory_layout &&
      v_input.has_direct_buffer_layout()) {
    return input;
  }

  TORCH_CHECK(
      supports_buffer_view_fast_path(v_input),
      describe_buffer_view_fast_path_failure(v_input));

  log_materialize_event(
      "ensure_buffer_storage",
      v_input,
      api::StorageType::BUFFER,
      memory_layout,
      v_input.storage_type() == api::StorageType::BUFFER
          ? "buffer_relayout"
          : "texture_to_buffer");

  return convert(materialize_to_contiguous_buffer(v_input, memory_layout));
}

Tensor ensure_texture_storage(
    const Tensor& input_arg,
    api::GPUMemoryLayout memory_layout,
    api::StorageType storage_type) {
  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  vTensor v_input = convert(input);

  if (
      v_input.storage_type() == storage_type &&
      v_input.gpu_memory_layout() == memory_layout) {
    return input;
  }

  if (
      v_input.storage_type() != api::StorageType::BUFFER &&
      v_input.storage_type() == api::StorageType::TEXTURE_3D) {
    if (
        v_input.gpu_memory_layout() ==
        api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED) {
      if (memory_layout == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED) {
        log_materialize_event(
            "ensure_texture_storage",
            v_input,
            storage_type,
            memory_layout,
            "image_layout_convert_width");
        return convert(
            packing::convert_image_channels_packed_to_width_packed(v_input));
      }
      if (memory_layout == api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED) {
        log_materialize_event(
            "ensure_texture_storage",
            v_input,
            storage_type,
            memory_layout,
            "image_layout_convert_height");
        return convert(
            packing::convert_image_channels_packed_to_height_packed(v_input));
      }
    }
  }

  TORCH_CHECK(
      supports_buffer_view_fast_path(v_input),
      describe_buffer_view_fast_path_failure(v_input));

  api::Context* const context = api::context();
  log_materialize_event(
      "ensure_texture_storage",
      v_input,
      storage_type,
      memory_layout,
      "buffer_to_texture_via_staging");

  api::StorageBuffer staging(context, v_input.dtype(), v_input.numel());
  vTensor v_src = v_input;
  pack_vtensor_to_staging(v_src, staging.buffer());

  vTensor v_out{
      context,
      v_input.sizes(),
      v_input.dtype(),
      storage_type,
      memory_layout,
  };
  api::PipelineBarrier pipeline_barrier{};
  add_buffer_barrier(
      pipeline_barrier,
      staging.buffer(),
      api::PipelineStage::COMPUTE | api::PipelineStage::TRANSFER,
      api::MemoryAccessType::WRITE,
      api::PipelineStage::COMPUTE | api::PipelineStage::TRANSFER,
      api::MemoryAccessType::READ);
  pack_buffer_to_vtensor(staging.buffer(), v_out, pipeline_barrier);
  return convert(v_out);
}

Tensor upcast_bfloat16_buffer_to_float(const Tensor& input) {
  TORCH_CHECK(input.is_vulkan(), "Input must be a Vulkan tensor");
  vTensor v_input = convert(input);
  TORCH_CHECK(
      v_input.storage_type() == api::StorageType::BUFFER,
      "BF16 buffer upcast requires a buffer-backed Vulkan tensor");
  TORCH_CHECK(
      v_input.dtype() == api::kBFloat16,
      "BF16 buffer upcast requires a BFloat16 Vulkan tensor");
  TORCH_CHECK(
      v_input.sizes().size() <= 4,
      "BF16 buffer upcast currently only supports tensors with up to 4 dimensions");

  api::AllocationScope allocation_scope("bf16.buffer_to_float");
  api::Context* const context = api::context();
  vTensor v_out{
      context,
      v_input.sizes(),
      api::kFloat,
      api::StorageType::BUFFER,
      v_input.gpu_memory_layout(),
  };

  api::PipelineBarrier pipeline_barrier{};
  api::utils::uvec3 global_size = {
      api::utils::safe_downcast<uint32_t>(v_out.numel()),
      1u,
      1u,
  };
  api::utils::uvec3 local_size = {32u, 1u, 1u};

  context->submit_compute_job(
      VK_KERNEL(buffer_to_buffer_bfloat16_to_float),
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      v_out.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      make_buffer_compute_metadata_ubo(context, v_out).buffer(),
      v_input.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ),
      make_buffer_compute_metadata_ubo(context, v_input).buffer());

  return convert(v_out);
}

Tensor mark_tensor_execution(
    const Tensor& input,
    const api::ExecutionLayout execution_layout,
    const bool persistent) {
  if (!input.is_vulkan()) {
    return input;
  }

  vTensor& v_input = convert(input);
  v_input.set_execution_layout(execution_layout);
  v_input.set_execution_persistent(persistent);
  return input;
}

PackedWeightHandle make_packed_weight_handle(
    Tensor weight,
    Tensor bias,
    std::vector<int64_t> logical_weight_sizes,
    const PackedWeightKind kind,
    const bool bias_defined,
    const bool quantized,
    const PackedWeightResidencyClass residency_class) {
  const bool persistent =
      residency_class == PackedWeightResidencyClass::PersistentInference;
  const size_t resident_nbytes = packed_weight_handle_nbytes(weight, bias);
  return PackedWeightHandle(
      mark_tensor_execution(
          weight, api::ExecutionLayout::PACKED_WEIGHT, persistent),
      mark_tensor_execution(
          bias, api::ExecutionLayout::PACKED_WEIGHT, persistent),
      std::move(logical_weight_sizes),
      kind,
      bias_defined,
      residency_class,
      quantized,
      api::ExecutionLayout::PACKED_WEIGHT,
      resident_nbytes);
}

std::optional<PackedWeightHandle> lookup_packed_weight_handle(
    const Tensor& source_weight,
    const std::optional<Tensor>& source_bias,
    IntArrayRef logical_weight_sizes,
    const PackedWeightKind kind,
    const bool quantized,
    const uint64_t options_key) {
  return packed_weight_residency_manager().lookup(
      source_weight,
      normalized_optional_tensor(source_bias),
      logical_weight_sizes,
      kind,
      quantized,
      options_key);
}

void store_packed_weight_handle(
    const Tensor& source_weight,
    const std::optional<Tensor>& source_bias,
    IntArrayRef logical_weight_sizes,
    const PackedWeightKind kind,
    const PackedWeightHandle& handle,
    const bool quantized,
    const uint64_t options_key) {
  packed_weight_residency_manager().store(
      source_weight,
      normalized_optional_tensor(source_bias),
      logical_weight_sizes,
      kind,
      handle,
      quantized,
      options_key);
}

Tensor cast_vulkan_tensor_dtype(const Tensor& input_arg, ScalarType dtype) {
  if (input_arg.scalar_type() == dtype) {
    return input_arg;
  }

  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  vTensor v_input = convert(input);

  switch (resolve_vulkan_cast_method(input.scalar_type(), dtype)) {
    case VulkanCastMethod::Identity:
      return input;
    case VulkanCastMethod::NativeBufferFloatToInt:
      if (!can_native_buffer_cast_input(v_input)) {
        return cast_vulkan_tensor_dtype_cpu_fallback(input, dtype);
      }
      return cast_vulkan_tensor_dtype_buffer_native(
          input, dtype, VK_KERNEL(buffer_cast_float_to_int));
    case VulkanCastMethod::NativeBufferIntToFloat:
      if (!can_native_buffer_cast_input(v_input)) {
        return cast_vulkan_tensor_dtype_cpu_fallback(input, dtype);
      }
      return cast_vulkan_tensor_dtype_buffer_native(
          input, dtype, VK_KERNEL(buffer_cast_int_to_float));
    case VulkanCastMethod::NativeBufferBFloat16ToFloat:
      if (!can_native_buffer_cast_input(v_input)) {
        return cast_vulkan_tensor_dtype_cpu_fallback(input, dtype);
      }
      return upcast_bfloat16_buffer_to_float(input);
    case VulkanCastMethod::CpuFallback:
      return cast_vulkan_tensor_dtype_cpu_fallback(input, dtype);
    case VulkanCastMethod::Unsupported:
      TORCH_CHECK(
          false,
          "Unsupported Vulkan cast from ",
          input.scalar_type(),
          " to ",
          dtype);
  }

  TORCH_CHECK(false, "Invalid Vulkan cast dispatch state");
}

void copy_buffer_to_vtensor(
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst,
    api::PipelineBarrier& pipeline_barrier) {
  api::Context* const context = api::context();

  TORCH_CHECK(
      src_buffer.mem_size() == v_dst.gpu_nbytes(),
      "Vulkan copy_buffer_to_vtensor: source buffer and destination texture "
      "do not have the same number of bytes");

  context->submit_copy<api::VulkanBuffer, api::VulkanImage>(
      // pipeline barrier
      pipeline_barrier,
      // resources
      src_buffer,
      v_dst.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::WRITE),
      // copy details
      v_dst.extents(),
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      VK_NULL_HANDLE);
}

void copy_buffer_to_buffer(
    api::Context* const context,
    api::StorageBuffer& src,
    api::StorageBuffer& dst,
    VkFence fence_handle) {
  api::PipelineBarrier pipeline_barrier{};

  context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
      // pipeline barrier
      pipeline_barrier,
      // resources
      src.buffer(),
      dst.buffer(),
      // copy details
      {static_cast<uint32_t>(src.buffer().mem_size()), 0u, 0u},
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      fence_handle);
}

void copy_vtensor_to_buffer(
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier& pipeline_barrier,
    const VkFence fence_handle) {
  api::Context* const context = api::context();

  TORCH_CHECK(
      v_src.gpu_nbytes() == dst_buffer.mem_size(),
      "Vulkan copy_vtensor_to_buffer: source texture and destination buffer "
      "do not have the same number of bytes");

  context->submit_copy<api::VulkanImage, api::VulkanBuffer>(
      // pipeline barrier
      pipeline_barrier,
      // resources
      v_src.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::READ),
      dst_buffer,
      // copy details
      v_src.extents(),
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      fence_handle);
}

void pack_buffer_to_vtensor(
    api::VulkanBuffer& buffer,
    vTensor& v_self,
    api::PipelineBarrier& pipeline_barrier) {
  api::Context* const context = api::context();

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    // The generic helper is used for both CPU-style logical staging buffers and
    // already packed GPU buffers. Only the former is safe to assume here, so
    // always repack through the metadata-aware shader path.
    packing::record_nchw_to_buffer_op(
        context, buffer, v_self, pipeline_barrier, VK_NULL_HANDLE);
  } else {
    api::ShaderInfo compute_shader = packing::get_nchw_to_image_shader(v_self);
    packing::record_nchw_to_image_op(
        context,
        compute_shader,
        buffer,
        v_self,
        pipeline_barrier,
        VK_NULL_HANDLE);
  }
}

void pack_staging_to_vtensor(api::VulkanBuffer& staging, vTensor& v_self) {
  api::PipelineBarrier pipeline_barrier{};
  pack_buffer_to_vtensor(staging, v_self, pipeline_barrier);
}

bool pack_vtensor_to_staging(
    vTensor& v_self,
    api::VulkanBuffer& staging,
    const VkFence fence_handle) {
  api::Context* const context = api::context();
  api::PipelineBarrier pipeline_barrier{};

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    // Compute-written direct buffer tensors such as large embedding/index_select
    // gathers can read back incorrectly through the raw transfer-copy fast
    // path. Transfer-written buffers, such as large weights uploaded from CPU,
    // still rely on the direct path for exact roundtrips. Use the direct path
    // only when the buffer was not last written by a compute shader.
    if (
        v_self.has_direct_buffer_layout() &&
        !v_self.last_write_was_compute() &&
        v_self.gpu_nbytes() == staging.mem_size()) {
      return context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
          pipeline_barrier,
          v_self.buffer(pipeline_barrier, api::PipelineStage::TRANSFER),
          staging,
          {api::utils::safe_downcast<uint32_t>(staging.mem_size()), 0u, 0u},
          {0u, 0u, 0u},
          {0u, 0u, 0u},
          fence_handle);
    }
    return packing::record_buffer_to_nchw_op(
        context, v_self, staging, pipeline_barrier, fence_handle);
  } else {
    api::ShaderInfo compute_shader = packing::get_image_to_nchw_shader(v_self);
    return packing::record_image_to_nchw_op(
        context,
        compute_shader,
        v_self,
        staging,
        pipeline_barrier,
        fence_handle);
  }
}

/*
 * Broadcasting Utils
 */

// check if two tensors are broadcastable
void is_broadcastable(const Tensor& input1, const Tensor& input2) {
  TORCH_CHECK(
      input1.dim() <= 4 && input2.dim() <= 4,
      "Vulkan only supports tensors <= 4 dimensions");

  // check if the shapes of input tensors are broadcastable
  // see https://pytorch.org/docs/stable/notes/broadcasting.html
  // for broadcasting semantics
  const std::string broadcast_error_msg = "Tensors are not broadcastable!";

  if (get_dim<Dim4D::Batch>(input1) != get_dim<Dim4D::Batch>(input2)) {
    TORCH_CHECK(
        get_dim<Dim4D::Batch>(input1) == 1 ||
            get_dim<Dim4D::Batch>(input2) == 1,
        broadcast_error_msg);
  }
  if (get_dim<Dim4D::Channel>(input1) != get_dim<Dim4D::Channel>(input2)) {
    TORCH_CHECK(
        get_dim<Dim4D::Channel>(input1) == 1 ||
            get_dim<Dim4D::Channel>(input2) == 1,
        broadcast_error_msg);
  }
  if (get_dim<Dim4D::Height>(input1) != get_dim<Dim4D::Height>(input2)) {
    TORCH_CHECK(
        get_dim<Dim4D::Height>(input1) == 1 ||
            get_dim<Dim4D::Height>(input2) == 1,
        broadcast_error_msg);
  }
  if (get_dim<Dim4D::Width>(input1) != get_dim<Dim4D::Width>(input2)) {
    TORCH_CHECK(
        get_dim<Dim4D::Width>(input1) == 1 ||
            get_dim<Dim4D::Width>(input2) == 1,
        broadcast_error_msg);
  }
}

// compute the output shape by broadcasting the shapes of t1 and t2
std::vector<int64_t> broadcast_size(const Tensor& t1, const Tensor& t2) {
  int64_t t1_size = t1.dim();
  int64_t t2_size = t2.dim();

  std::vector<int64_t> out;
  if (t1_size > t2_size) {
    for (int64_t i = 0; i < t1_size; i++) {
      out.push_back(t1.sizes()[i]);
    }
  } else {
    for (int64_t i = 0; i < t2_size; i++) {
      out.push_back(t2.sizes()[i]);
    }
  }

  if (!out.empty()) {
    out[out.size() - 1] =
        std::max(get_dim<Dim4D::Width>(t1), get_dim<Dim4D::Width>(t2));
  }
  if (out.size() > 1) {
    out[out.size() - 2] =
        std::max(get_dim<Dim4D::Height>(t1), get_dim<Dim4D::Height>(t2));
  }
  if (out.size() > 2) {
    out[out.size() - 3] =
        std::max(get_dim<Dim4D::Channel>(t1), get_dim<Dim4D::Channel>(t2));
  }
  if (out.size() > 3) {
    out[out.size() - 4] =
        std::max(get_dim<Dim4D::Batch>(t1), get_dim<Dim4D::Batch>(t2));
  }

  return out;
}

api::utils::vec4 extract_texel(const Tensor& input, const ivec3& pos) {
  api::Context* const context = api::context();

  TORCH_CHECK(input.is_vulkan());
  const vTensor& v_input = convert(input);

  api::PipelineBarrier pipeline_barrier{};

  std::vector<int64_t> output_size{1, 1, 1};

  // x, y, z, w all using a single element tensor. We intend to pull
  // (0, 0, 0).x from each tensor. This allows us to isolate the effect
  // of most packing mechanism.
  api::ScalarType dtype = convert_dtype(input.scalar_type());
  vTensor v_outputs_x{context, output_size, dtype};
  vTensor v_outputs_y{context, output_size, dtype};
  vTensor v_outputs_z{context, output_size, dtype};
  vTensor v_outputs_w{context, output_size, dtype};

  const struct Block final {
    ivec3 pos;
  } block{
      pos,
  };

  api::UniformParamsBuffer params(context, block);

  context->submit_compute_job(
      VK_KERNEL(extract_texel),
      pipeline_barrier,
      {1, 1, 1},
      {1, 1, 1},
      VK_NULL_HANDLE,
      v_outputs_x.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_outputs_y.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_outputs_z.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_outputs_w.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  vec4 rv = {
      convert(v_outputs_x).cpu().data_ptr<float>()[0],
      convert(v_outputs_y).cpu().data_ptr<float>()[0],
      convert(v_outputs_z).cpu().data_ptr<float>()[0],
      convert(v_outputs_w).cpu().data_ptr<float>()[0],
  };

  return rv;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
