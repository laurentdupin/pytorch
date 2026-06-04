#include <ATen/ATen.h>
#include <ATen/native/vulkan/api/Diagnostics.h>
#include <ATen/native/vulkan/impl/Packing.h>
#include <ATen/native/vulkan/ops/BinaryOp.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/LayoutTransitions.h>
#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/Softmax.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/planning/ExecutionObjects.h>
#include <ATen/vulkan/Context.h>
#include <c10/util/irange.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace {

constexpr int64_t kLargeFloatingMatrixNumelThreshold = 1 << 20;

struct VulkanBufferCopyCounters final {
  std::atomic<uint64_t> total{0u};
  std::atomic<uint64_t> total_bytes{0u};
  std::atomic<uint64_t> explicit_copy{0u};
  std::atomic<uint64_t> contiguous{0u};
  std::atomic<uint64_t> view_materialization{0u};
  std::atomic<uint64_t> reshape_materialization{0u};
  std::atomic<uint64_t> permute_materialization{0u};
  std::atomic<uint64_t> transpose_materialization{0u};
  std::atomic<uint64_t> layout_conversion{0u};
  std::atomic<uint64_t> attention_materialization{0u};
  std::atomic<uint64_t> linear_materialization{0u};
  std::atomic<uint64_t> conv_materialization{0u};
  std::atomic<uint64_t> decoder_materialization{0u};
  std::atomic<uint64_t> backbone_materialization{0u};
  std::atomic<uint64_t> logical_noop_copy{0u};
};

struct VulkanBufferCopyDecision final {
  VulkanBufferCopyReason reason = VulkanBufferCopyReason::Unknown;
  int64_t numel = 0;
  int64_t bytes = 0;
  std::vector<int64_t> src_sizes;
  std::vector<int64_t> dst_sizes;
  std::vector<int64_t> src_strides;
  std::vector<int64_t> dst_strides;
  int64_t src_storage_offset = 0;
  int64_t dst_storage_offset = 0;
  ScalarType dtype = ScalarType::Undefined;
  bool src_direct_buffer = false;
  bool dst_direct_buffer = false;
  bool src_contiguous = false;
  bool dst_contiguous = false;
  bool logical_noop = false;
  std::string producer_label;
  std::string consumer_label;
  std::string producer_role;
  std::string consumer_role;
};

struct VulkanBufferCopyAggregateValue final {
  uint64_t count = 0u;
  uint64_t bytes = 0u;
  uint64_t logical_noop = 0u;
};

enum class VulkanCloneRequirementReason : uint8_t {
  Unknown = 0,
  UserVisibleAtenClone,
  InternalDirectBufferPreparation,
  LinearInputPreparation,
  Fc2InputPreparation,
  TensorToContiguous,
  StrideNormalization,
  StorageOffsetNormalization,
  LayoutConversion,
  AliasBreakRequired,
  DistinctOutputRequired,
  ConsumerDoesNotAcceptProducerLayout,
};

struct VulkanCloneRequirementAggregateValue final {
  uint64_t count = 0u;
  uint64_t bytes = 0u;
};

c10::MemoryFormat memory_format_for_buffer_layout(
    const api::GPUMemoryLayout memory_layout);

void pack_logical_tensor_to_buffer_mapping_dispatch(
    const Tensor& src,
    api::MemoryMap& dst_mapping,
    const IntArrayRef physical_strides,
    const int64_t physical_numel,
    const int64_t storage_offset,
    const bool clear_destination);

std::string format_sizes(const std::vector<int64_t>& sizes);

const std::string& copy_sync_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_COPY_SYNC_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool copy_sync_logging_enabled() {
  return !copy_sync_log_path().empty();
}

VulkanBufferCopyCounters& buffer_copy_counters() {
  static VulkanBufferCopyCounters counters;
  return counters;
}

const std::string& buffer_copy_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_BUFFER_COPY_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool buffer_copy_logging_enabled() {
  return !buffer_copy_log_path().empty();
}

bool buffer_copy_aggregate_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_BUFFER_COPY_AGGREGATE");
    return env != nullptr && std::string(env) != "0" &&
        std::string(env) != "false" && std::string(env) != "FALSE";
  }();
  return enabled;
}

const char* clone_requirement_reason_name(
    const VulkanCloneRequirementReason reason) {
  switch (reason) {
    case VulkanCloneRequirementReason::Unknown:
      return "unknown";
    case VulkanCloneRequirementReason::UserVisibleAtenClone:
      return "user_visible_aten_clone";
    case VulkanCloneRequirementReason::InternalDirectBufferPreparation:
      return "internal_direct_buffer_preparation";
    case VulkanCloneRequirementReason::LinearInputPreparation:
      return "linear_input_preparation";
    case VulkanCloneRequirementReason::Fc2InputPreparation:
      return "fc2_input_preparation";
    case VulkanCloneRequirementReason::TensorToContiguous:
      return "tensor_to_contiguous";
    case VulkanCloneRequirementReason::StrideNormalization:
      return "stride_normalization";
    case VulkanCloneRequirementReason::StorageOffsetNormalization:
      return "storage_offset_normalization";
    case VulkanCloneRequirementReason::LayoutConversion:
      return "layout_conversion";
    case VulkanCloneRequirementReason::AliasBreakRequired:
      return "alias_break_required";
    case VulkanCloneRequirementReason::DistinctOutputRequired:
      return "distinct_output_required";
    case VulkanCloneRequirementReason::ConsumerDoesNotAcceptProducerLayout:
      return "consumer_does_not_accept_producer_layout";
  }
  return "unknown";
}

const char* buffer_copy_reason_name(const VulkanBufferCopyReason reason) {
  switch (reason) {
    case VulkanBufferCopyReason::Unknown:
      return "unknown";
    case VulkanBufferCopyReason::ExplicitCopy:
      return "explicit_copy";
    case VulkanBufferCopyReason::TensorToContiguous:
      return "tensor_to_contiguous";
    case VulkanBufferCopyReason::ViewMaterialization:
      return "view_materialization";
    case VulkanBufferCopyReason::ReshapeMaterialization:
      return "reshape_materialization";
    case VulkanBufferCopyReason::PermuteMaterialization:
      return "permute_materialization";
    case VulkanBufferCopyReason::TransposeMaterialization:
      return "transpose_materialization";
    case VulkanBufferCopyReason::LayoutConversion:
      return "layout_conversion";
    case VulkanBufferCopyReason::DirectBufferConversion:
      return "direct_buffer_conversion";
    case VulkanBufferCopyReason::PackedLayoutConversion:
      return "packed_layout_conversion";
    case VulkanBufferCopyReason::StorageOffsetNormalization:
      return "storage_offset_normalization";
    case VulkanBufferCopyReason::StrideNormalization:
      return "stride_normalization";
    case VulkanBufferCopyReason::AttentionInputMaterialization:
      return "attention_input_materialization";
    case VulkanBufferCopyReason::AttentionOutputMaterialization:
      return "attention_output_materialization";
    case VulkanBufferCopyReason::LinearInputMaterialization:
      return "linear_input_materialization";
    case VulkanBufferCopyReason::LinearOutputMaterialization:
      return "linear_output_materialization";
    case VulkanBufferCopyReason::ConvInputMaterialization:
      return "conv_input_materialization";
    case VulkanBufferCopyReason::ConvOutputMaterialization:
      return "conv_output_materialization";
    case VulkanBufferCopyReason::DecoderFeatureMaterialization:
      return "decoder_feature_materialization";
    case VulkanBufferCopyReason::BackboneTokenMaterialization:
      return "backbone_token_materialization";
    case VulkanBufferCopyReason::ReadbackPreparation:
      return "readback_preparation";
  }
  return "unknown";
}

const char* scalar_type_name(const ScalarType dtype) {
  switch (dtype) {
    case ScalarType::Float:
      return "float";
    case ScalarType::BFloat16:
      return "bfloat16";
    case ScalarType::Half:
      return "half";
    case ScalarType::Byte:
      return "byte";
    case ScalarType::Char:
      return "char";
    case ScalarType::Int:
      return "int";
    case ScalarType::Long:
      return "long";
    case ScalarType::Bool:
      return "bool";
    case ScalarType::Undefined:
      return "undefined";
    default:
      return "other";
  }
}

bool has_token(const std::string& text, const char* token) {
  return text.find(token) != std::string::npos;
}

bool is_contiguous_desc(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides) {
  if (sizes.size() != strides.size()) {
    return false;
  }
  int64_t expected = 1;
  for (int64_t dim = static_cast<int64_t>(sizes.size()) - 1; dim >= 0; --dim) {
    if (sizes[dim] <= 1) {
      continue;
    }
    if (strides[dim] != expected) {
      return false;
    }
    expected *= sizes[dim];
  }
  return true;
}

std::array<int64_t, 5> compact_desc(const std::vector<int64_t>& values) {
  std::array<int64_t, 5> out{0, 0, 0, 0, 0};
  out[0] = static_cast<int64_t>(std::min<size_t>(values.size(), 4));
  for (size_t idx = 0; idx < values.size() && idx < 4; ++idx) {
    out[idx + 1] = values[idx];
  }
  return out;
}

std::string format_compact_desc(const std::array<int64_t, 5>& values) {
  std::ostringstream stream;
  stream << "[";
  const int64_t rank = std::max<int64_t>(0, std::min<int64_t>(values[0], 4));
  for (int64_t idx = 0; idx < rank; ++idx) {
    if (idx > 0) {
      stream << ",";
    }
    stream << values[idx + 1];
  }
  stream << "]";
  return stream.str();
}

VulkanBufferCopyReason refine_buffer_copy_reason(
    const VulkanBufferCopyReason reason,
    const std::string& producer,
    const std::string& consumer) {
  const std::string labels = producer + " " + consumer;
  if (has_token(labels, "attention")) {
    return has_token(labels, "output")
        ? VulkanBufferCopyReason::AttentionOutputMaterialization
        : VulkanBufferCopyReason::AttentionInputMaterialization;
  }
  if (has_token(labels, "linear") || has_token(labels, "mm")) {
    return has_token(labels, "output")
        ? VulkanBufferCopyReason::LinearOutputMaterialization
        : VulkanBufferCopyReason::LinearInputMaterialization;
  }
  if (has_token(labels, "conv")) {
    return has_token(labels, "output")
        ? VulkanBufferCopyReason::ConvOutputMaterialization
        : VulkanBufferCopyReason::ConvInputMaterialization;
  }
  if (has_token(labels, "decoder")) {
    return VulkanBufferCopyReason::DecoderFeatureMaterialization;
  }
  if (has_token(labels, "backbone") || has_token(labels, "token")) {
    return VulkanBufferCopyReason::BackboneTokenMaterialization;
  }
  if (has_token(labels, "contiguous")) {
    return VulkanBufferCopyReason::TensorToContiguous;
  }
  if (has_token(labels, "reshape")) {
    return VulkanBufferCopyReason::ReshapeMaterialization;
  }
  if (has_token(labels, "permute")) {
    return VulkanBufferCopyReason::PermuteMaterialization;
  }
  if (has_token(labels, "transpose")) {
    return VulkanBufferCopyReason::TransposeMaterialization;
  }
  if (has_token(labels, "materialize")) {
    return VulkanBufferCopyReason::ViewMaterialization;
  }
  return reason;
}

VulkanBufferCopyDecision make_buffer_copy_decision(
    VulkanBufferCopyReason reason,
    const vTensor& src,
    const vTensor& dst,
    const char* producer_label,
    const char* consumer_label,
    const char* producer_role,
    const char* consumer_role) {
  const std::string producer = producer_label != nullptr
      ? std::string(producer_label)
      : api::current_runtime_label();
  const std::string consumer = consumer_label != nullptr
      ? std::string(consumer_label)
      : api::current_allocation_label();
  reason = refine_buffer_copy_reason(reason, producer, consumer);

  VulkanBufferCopyDecision decision;
  decision.reason = reason;
  decision.numel = src.numel();
  decision.bytes = src.gpu_nbytes();
  decision.src_sizes = src.sizes();
  decision.dst_sizes = dst.sizes();
  decision.src_strides = src.logical_strides();
  decision.dst_strides = dst.logical_strides();
  decision.src_storage_offset = src.storage_offset();
  decision.dst_storage_offset = dst.storage_offset();
  decision.dtype = convert_dtype(src.dtype());
  decision.src_direct_buffer = src.has_direct_buffer_layout();
  decision.dst_direct_buffer = dst.has_direct_buffer_layout();
  decision.src_contiguous =
      is_contiguous_desc(decision.src_sizes, decision.src_strides);
  decision.dst_contiguous =
      is_contiguous_desc(decision.dst_sizes, decision.dst_strides);
  const bool same_buffer_storage =
      src.storage_type() == api::StorageType::BUFFER &&
      dst.storage_type() == api::StorageType::BUFFER &&
      &src.buffer() == &dst.buffer();
  decision.logical_noop =
      same_buffer_storage && src.dtype() == dst.dtype() &&
      src.sizes() == dst.sizes() &&
      src.logical_strides() == dst.logical_strides() &&
      src.storage_offset() == dst.storage_offset() &&
      src.has_direct_buffer_layout() == dst.has_direct_buffer_layout();
  decision.producer_label = producer.empty() ? "unknown" : producer;
  decision.consumer_label = consumer.empty() ? "unknown" : consumer;
  decision.producer_role =
      producer_role && producer_role[0] != '\0' ? producer_role : "unknown";
  decision.consumer_role =
      consumer_role && consumer_role[0] != '\0' ? consumer_role : "unknown";
  return decision;
}

std::string make_buffer_copy_aggregate_key(
    const VulkanBufferCopyDecision& decision) {
  std::ostringstream stream;
  stream << "reason=" << buffer_copy_reason_name(decision.reason)
         << " producer=" << decision.producer_label
         << " producer_role=" << decision.producer_role
         << " consumer=" << decision.consumer_label
         << " consumer_role=" << decision.consumer_role
         << " dtype=" << scalar_type_name(decision.dtype)
         << " src_sizes=" << format_compact_desc(compact_desc(decision.src_sizes))
         << " dst_sizes=" << format_compact_desc(compact_desc(decision.dst_sizes))
         << " src_strides="
         << format_compact_desc(compact_desc(decision.src_strides))
         << " dst_strides="
         << format_compact_desc(compact_desc(decision.dst_strides))
         << " src_direct=" << (decision.src_direct_buffer ? 1 : 0)
         << " dst_direct=" << (decision.dst_direct_buffer ? 1 : 0);
  return stream.str();
}

class VulkanBufferCopyAggregateProfiler final {
 public:
  void record(const VulkanBufferCopyDecision& decision) {
    if (!buffer_copy_aggregate_enabled()) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto& value = entries_[make_buffer_copy_aggregate_key(decision)];
    value.count += 1u;
    value.bytes += static_cast<uint64_t>(std::max<int64_t>(decision.bytes, 0));
    value.logical_noop += decision.logical_noop ? 1u : 0u;
  }

  std::vector<std::string> snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (const auto& entry : entries_) {
      std::ostringstream stream;
      stream << "buffer_copy_aggregate " << entry.first
             << " count=" << entry.second.count
             << " bytes=" << entry.second.bytes
             << " logical_noop=" << entry.second.logical_noop;
      out.emplace_back(stream.str());
    }
    std::sort(out.begin(), out.end());
    return out;
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, VulkanBufferCopyAggregateValue> entries_;
};

VulkanBufferCopyAggregateProfiler& buffer_copy_aggregate_profiler() {
  static VulkanBufferCopyAggregateProfiler profiler;
  return profiler;
}

VulkanCloneRequirementReason classify_clone_requirement(
    const VulkanBufferCopyDecision& decision) {
  if (!has_token(decision.consumer_label, "clone")) {
    return VulkanCloneRequirementReason::Unknown;
  }
  if (decision.src_storage_offset != 0 || decision.dst_storage_offset != 0) {
    return VulkanCloneRequirementReason::StorageOffsetNormalization;
  }
  if (decision.src_strides != decision.dst_strides) {
    return VulkanCloneRequirementReason::StrideNormalization;
  }
  if (decision.src_direct_buffer != decision.dst_direct_buffer) {
    return VulkanCloneRequirementReason::InternalDirectBufferPreparation;
  }
  if (
      decision.producer_label == "aten::gelu" &&
      decision.src_sizes.size() == 3 && decision.src_sizes[0] == 1 &&
      decision.src_sizes[2] == 1536) {
    return VulkanCloneRequirementReason::Fc2InputPreparation;
  }
  if (
      has_token(decision.producer_label, "linear") ||
      has_token(decision.producer_role, "linear")) {
    return VulkanCloneRequirementReason::LinearInputPreparation;
  }
  return VulkanCloneRequirementReason::UserVisibleAtenClone;
}

std::string make_clone_requirement_key(
    const VulkanBufferCopyDecision& decision,
    const VulkanCloneRequirementReason reason) {
  std::ostringstream stream;
  stream << "reason=" << clone_requirement_reason_name(reason)
         << " producer=" << decision.producer_label
         << " producer_role=" << decision.producer_role
         << " consumer=" << decision.consumer_label
         << " consumer_role=" << decision.consumer_role
         << " dtype=" << scalar_type_name(decision.dtype)
         << " sizes=" << format_compact_desc(compact_desc(decision.src_sizes))
         << " src_strides="
         << format_compact_desc(compact_desc(decision.src_strides))
         << " dst_strides="
         << format_compact_desc(compact_desc(decision.dst_strides))
         << " src_direct=" << (decision.src_direct_buffer ? 1 : 0)
         << " dst_direct=" << (decision.dst_direct_buffer ? 1 : 0)
         << " src_offset=" << decision.src_storage_offset
         << " dst_offset=" << decision.dst_storage_offset;
  return stream.str();
}

class VulkanCloneRequirementProfiler final {
 public:
  void record(const VulkanBufferCopyDecision& decision) {
    const VulkanCloneRequirementReason reason =
        classify_clone_requirement(decision);
    if (reason == VulkanCloneRequirementReason::Unknown) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto& value = entries_[make_clone_requirement_key(decision, reason)];
    value.count += 1u;
    value.bytes += static_cast<uint64_t>(std::max<int64_t>(decision.bytes, 0));
  }

  std::vector<std::string> snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (const auto& entry : entries_) {
      std::ostringstream stream;
      stream << "clone_requirement " << entry.first
             << " count=" << entry.second.count
             << " bytes=" << entry.second.bytes;
      out.emplace_back(stream.str());
    }
    std::sort(out.begin(), out.end());
    return out;
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, VulkanCloneRequirementAggregateValue>
      entries_;
};

VulkanCloneRequirementProfiler& clone_requirement_profiler() {
  static VulkanCloneRequirementProfiler profiler;
  return profiler;
}

void append_buffer_copy_log(const VulkanBufferCopyDecision& decision) {
  if (!buffer_copy_logging_enabled()) {
    return;
  }

  std::ofstream out(buffer_copy_log_path(), std::ios::app);
  out << "buffer_copy"
      << " reason=" << buffer_copy_reason_name(decision.reason)
      << " numel=" << decision.numel
      << " bytes=" << decision.bytes
      << " dtype=" << scalar_type_name(decision.dtype)
      << " src_sizes=" << format_sizes(decision.src_sizes)
      << " dst_sizes=" << format_sizes(decision.dst_sizes)
      << " src_strides=" << format_sizes(decision.src_strides)
      << " dst_strides=" << format_sizes(decision.dst_strides)
      << " src_offset=" << decision.src_storage_offset
      << " dst_offset=" << decision.dst_storage_offset
      << " src_direct=" << (decision.src_direct_buffer ? 1 : 0)
      << " dst_direct=" << (decision.dst_direct_buffer ? 1 : 0)
      << " src_contig=" << (decision.src_contiguous ? 1 : 0)
      << " dst_contig=" << (decision.dst_contiguous ? 1 : 0)
      << " logical_noop=" << (decision.logical_noop ? 1 : 0)
      << " producer=" << decision.producer_label
      << " producer_role=" << decision.producer_role
      << " consumer=" << decision.consumer_label
      << " consumer_role=" << decision.consumer_role
      << '\n';
}

void note_buffer_copy_decision(const VulkanBufferCopyDecision& decision) {
  VulkanBufferCopyCounters& counters = buffer_copy_counters();
  counters.total.fetch_add(1u, std::memory_order_relaxed);
  counters.total_bytes.fetch_add(
      static_cast<uint64_t>(std::max<int64_t>(decision.bytes, 0)),
      std::memory_order_relaxed);
  if (decision.logical_noop) {
    counters.logical_noop_copy.fetch_add(1u, std::memory_order_relaxed);
  }
  switch (decision.reason) {
    case VulkanBufferCopyReason::ExplicitCopy:
      counters.explicit_copy.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanBufferCopyReason::TensorToContiguous:
      counters.contiguous.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanBufferCopyReason::ViewMaterialization:
      counters.view_materialization.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanBufferCopyReason::ReshapeMaterialization:
      counters.reshape_materialization.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanBufferCopyReason::PermuteMaterialization:
      counters.permute_materialization.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanBufferCopyReason::TransposeMaterialization:
      counters.transpose_materialization.fetch_add(
          1u, std::memory_order_relaxed);
      break;
    case VulkanBufferCopyReason::LayoutConversion:
    case VulkanBufferCopyReason::DirectBufferConversion:
    case VulkanBufferCopyReason::PackedLayoutConversion:
    case VulkanBufferCopyReason::StorageOffsetNormalization:
    case VulkanBufferCopyReason::StrideNormalization:
      counters.layout_conversion.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanBufferCopyReason::AttentionInputMaterialization:
    case VulkanBufferCopyReason::AttentionOutputMaterialization:
      counters.attention_materialization.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanBufferCopyReason::LinearInputMaterialization:
    case VulkanBufferCopyReason::LinearOutputMaterialization:
      counters.linear_materialization.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanBufferCopyReason::ConvInputMaterialization:
    case VulkanBufferCopyReason::ConvOutputMaterialization:
      counters.conv_materialization.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanBufferCopyReason::DecoderFeatureMaterialization:
      counters.decoder_materialization.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanBufferCopyReason::BackboneTokenMaterialization:
      counters.backbone_materialization.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanBufferCopyReason::ReadbackPreparation:
    case VulkanBufferCopyReason::Unknown:
      break;
  }
  buffer_copy_aggregate_profiler().record(decision);
  clone_requirement_profiler().record(decision);
  append_buffer_copy_log(decision);
}

std::string readback_buffer_label(const char* suffix) {
  const std::string& runtime_label = api::current_runtime_label();
  if (!runtime_label.empty()) {
    return runtime_label + "." + suffix;
  }

  const std::string& allocation_label = api::current_allocation_label();
  if (!allocation_label.empty() && allocation_label != "unlabeled") {
    return allocation_label + "." + suffix;
  }

  return std::string("cpu_readback.") + suffix;
}

utils::ReadbackBufferObject lookup_or_create_readback_buffer(
    const char* suffix,
    const size_t size_bytes) {
  return utils::lookup_or_create_labeled_readback_buffer_object(
      readback_buffer_label(suffix),
      utils::VulkanReadbackBufferSpec{
          size_bytes,
          true,
      });
}

bool should_force_buffer_storage_for_to_vulkan(const Tensor& src) {
  // Low-rank floating tensors are typically model weights, biases,
  // normalization parameters, token-space tables, or 4D activations. Keep them
  // buffer-backed so buffer-native model paths do not begin life as textures
  // and immediately materialize back.
  return c10::isFloatingType(src.scalar_type()) && src.dim() >= 1 &&
      src.dim() <= 4 && src.numel() > 0;
}

bool should_use_host_visible_buffer_for_to_vulkan(const Tensor& src) {
  // Desktop vision input packets arrive as large uint8 tensors that are copied
  // to Vulkan every frame. Keeping those buffer-backed tensors host-visible
  // avoids an extra staging allocation and transfer before the first buffer
  // kernels consume them.
  return src.scalar_type() == at::kByte && src.dim() >= 1 && src.dim() <= 4 &&
      src.numel() > 0;
}

bool should_preserve_compact_cpu_buffer_view_for_to_vulkan(const Tensor& src) {
  // Deep Desktop image packets arrive as contiguous HWC/NHWC uint8 tensors.
  // Uploading them directly into a flat contiguous buffer and then wrapping a
  // metadata view avoids the expensive HWC -> width-packed scatter on every
  // frame while keeping the Python path unchanged.
  return src.scalar_type() == at::kByte && src.is_contiguous() &&
      src.dim() >= 2 && src.dim() <= 4 && src.size(src.dim() - 1) > 0 &&
      src.size(src.dim() - 1) < 4 && src.numel() % 4 == 0;
}

bool is_large_floating_matrix(const Tensor& src) {
  return c10::isFloatingType(src.scalar_type()) && src.dim() == 2 &&
      src.numel() >= kLargeFloatingMatrixNumelThreshold;
}

bool should_flush_after_labeled_to_vulkan(const Tensor& src) {
  // Model placement uses to_vulkan_labeled() heavily. Large weight transfers
  // otherwise keep both the staging allocation and the destination allocation
  // live until the command stream is flushed, which doubles residency and can
  // OOM medium models during module.to("vulkan"). Flush eagerly for large 2D
  // floating matrices to retire staging allocations as we go.
  return is_large_floating_matrix(src);
}

bool buffer_allocation_is_host_visible(const vTensor& tensor) {
  return tensor.buffer_uses_host_visible_allocation();
}

void pack_cpu_to_host_visible_vulkan_buffer(const Tensor& src, vTensor& dst);

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

void log_buffer_to_buffer_copy_submit(
    const char* path,
    const vTensor& src,
    const vTensor& dst) {
  std::ostringstream stream;
  stream << "aten::copy_.buffer_to_buffer_submit"
         << " path=" << path
         << " src_sizes=" << format_sizes(src.sizes())
         << " dst_sizes=" << format_sizes(dst.sizes())
         << " src_direct=" << (src.has_direct_buffer_layout() ? 1 : 0)
         << " dst_direct=" << (dst.has_direct_buffer_layout() ? 1 : 0)
         << " src_offset=" << src.storage_offset()
         << " dst_offset=" << dst.storage_offset()
         << " src_logical_bytes=" << src.nbytes()
         << " dst_logical_bytes=" << dst.nbytes()
         << " src_gpu_bytes=" << src.gpu_nbytes()
         << " dst_gpu_bytes=" << dst.gpu_nbytes();
  utils::log_vulkan_op_hit(stream.str());
}

void log_copy_sync_event(
    const char* kind,
    const vTensor& tensor,
    const bool direct_buffer_layout) {
  if (!copy_sync_logging_enabled()) {
    return;
  }

  std::ofstream out(copy_sync_log_path(), std::ios::app);
  out << "kind=" << kind
      << " caller=" << api::current_allocation_label()
      << " storage=" << storage_type_name(tensor.storage_type())
      << " direct_buffer=" << (direct_buffer_layout ? 1 : 0)
      << " logical_bytes=" << tensor.nbytes()
      << " gpu_bytes=" << tensor.gpu_nbytes()
      << " sizes=" << format_sizes(tensor.sizes()) << '\n';
}

void log_cpu_timeline_copy_event(
    const char* event,
    const uint64_t duration_us,
    const vTensor& tensor,
    const bool direct_buffer_layout) {
  if (!api::cpu_timeline_logging_enabled()) {
    return;
  }

  std::ostringstream stream;
  stream << "event=" << event
         << " duration_us=" << duration_us
         << " caller=" << api::current_allocation_label()
         << " storage=" << storage_type_name(tensor.storage_type())
         << " direct_buffer=" << (direct_buffer_layout ? 1 : 0)
         << " logical_bytes=" << tensor.nbytes()
         << " gpu_bytes=" << tensor.gpu_nbytes()
         << " sizes=" << format_sizes(tensor.sizes());
  api::append_cpu_timeline_log_line(stream.str());
}

void retire_command_resources_after_fence_wait(api::Context* const context) {
  utils::log_vulkan_op_hit("aten::copy_.retire_after_fence_begin");
  context->retire_after_fence_wait();
  utils::log_vulkan_op_hit("aten::copy_.retire_after_fence_end");
}

void release_retired_objects_after_context_unlock(api::Context* const context) {
  utils::log_vulkan_op_hit("aten::copy_.release_retired_contexts_begin");
  const bool released_retired_objects =
      utils::release_retired_packed_weight_entries() |
      utils::release_retired_linear_contexts();
  utils::log_vulkan_op_hit("aten::copy_.release_retired_contexts_end");
  if (released_retired_objects) {
    utils::log_vulkan_op_hit("aten::copy_.retire_after_release_begin");
    context->retire_after_fence_wait();
    utils::log_vulkan_op_hit("aten::copy_.retire_after_release_end");
  }
}

void retire_after_fence_wait_and_release(api::Context* const context) {
  retire_command_resources_after_fence_wait(context);
  release_retired_objects_after_context_unlock(context);
}

bool can_copy_vulkan_buffer_to_buffer_on_device(
    const vTensor& src,
    const vTensor& dst) {
  if (
      src.storage_type() != api::StorageType::BUFFER ||
      dst.storage_type() != api::StorageType::BUFFER ||
      src.dtype() != dst.dtype() ||
      src.sizes() != dst.sizes() ||
      src.sizes().size() > 4 ||
      dst.sizes().size() > 4) {
    return false;
  }

  return src.dtype() == api::kFloat || src.dtype() == api::kByte;
}

void copy_vulkan_buffer_to_buffer_on_device(
    vTensor& src,
    vTensor& dst,
    const char* producer_label = nullptr,
    const char* consumer_label = nullptr,
    const char* producer_role = nullptr,
    const char* consumer_role = nullptr) {
  TORCH_CHECK(
      can_copy_vulkan_buffer_to_buffer_on_device(src, dst),
      "Unsupported Vulkan buffer-to-buffer device copy");

  if (src.numel() == 0u) {
    return;
  }

  utils::log_vulkan_op_hit("aten::copy_.buffer_to_buffer");
  note_vulkan_buffer_copy(
      VulkanBufferCopyReason::ExplicitCopy,
      src,
      dst,
      producer_label ? producer_label : "aten::copy_",
      consumer_label ? consumer_label : "aten::copy_.buffer_to_buffer",
      producer_role,
      consumer_role);

  api::Context* const context = dst.context();
  if (
      src.has_direct_buffer_layout() && dst.has_direct_buffer_layout() &&
      src.storage_offset() == 0 && dst.storage_offset() == 0 &&
      src.gpu_nbytes() == dst.gpu_nbytes() &&
      !src.last_write_was_compute()) {
    log_buffer_to_buffer_copy_submit("direct_transfer", src, dst);
    api::PipelineBarrier pipeline_barrier{};
    context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
        pipeline_barrier,
        src.buffer(
            pipeline_barrier,
            api::PipelineStage::TRANSFER,
            api::MemoryAccessType::READ),
        dst.buffer(
            pipeline_barrier,
            api::PipelineStage::TRANSFER,
            api::MemoryAccessType::WRITE),
        {api::utils::safe_downcast<uint32_t>(src.gpu_nbytes()), 0u, 0u},
        {0u, 0u, 0u},
        {0u, 0u, 0u},
        VK_NULL_HANDLE);
    log_copy_sync_event(
        "copy_vulkan_to_vulkan_buffer_direct",
        dst,
        dst.has_direct_buffer_layout());
    return;
  }

  log_buffer_to_buffer_copy_submit("staging_pack", src, dst);
  api::StorageBuffer staging(context, src.dtype(), src.numel());
  api::PipelineBarrier read_barrier{};
  packing::record_buffer_to_nchw_op(
      context, src, staging.buffer(), read_barrier, VK_NULL_HANDLE);

  api::PipelineBarrier write_barrier{};
  add_buffer_barrier(
      write_barrier,
      staging.buffer(),
      api::PipelineStage::COMPUTE | api::PipelineStage::TRANSFER,
      api::MemoryAccessType::WRITE,
      api::PipelineStage::COMPUTE | api::PipelineStage::TRANSFER,
      api::MemoryAccessType::READ);
  packing::record_nchw_to_buffer_op(
      context, staging.buffer(), dst, write_barrier, VK_NULL_HANDLE);
  log_copy_sync_event(
      "copy_vulkan_to_vulkan_buffer_staging",
      dst,
      dst.has_direct_buffer_layout());
}

c10::MemoryFormat memory_format_for_buffer_layout(
    const api::GPUMemoryLayout memory_layout) {
  switch (memory_layout) {
    case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
      return c10::MemoryFormat::Contiguous;
    case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
      return c10::MemoryFormat::ChannelsLast;
    default:
      VK_THROW("Unsupported buffer memory layout");
  }
}

std::vector<int64_t> calc_logical_contiguous_strides(
    const IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (int idx = static_cast<int>(sizes.size()) - 2; idx >= 0; --idx) {
    strides[idx] = strides[idx + 1] * std::max<int64_t>(sizes[idx + 1], 1);
  }
  return strides;
}

bool has_last_dim_padded_contiguous_source_layout(
    const Tensor& src,
    const int64_t padded_last) {
  if (
      src.dim() < 1 || src.dim() > 4 || src.storage_offset() != 0 ||
      src.size(src.dim() - 1) <= 0 || src.stride(src.dim() - 1) != 1) {
    return false;
  }

  std::vector<int64_t> physical_sizes = src.sizes().vec();
  physical_sizes.back() = padded_last;
  const std::vector<int64_t> expected_strides =
      calc_logical_contiguous_strides(physical_sizes);
  for (const auto dim : c10::irange(src.dim())) {
    if (src.stride(dim) != expected_strides[dim]) {
      return false;
    }
  }
  return true;
}

bool can_pack_last_dim_padded_width_packed_contiguous_buffer(
    const Tensor& src,
    const vTensor& dst) {
  if (
      !src.is_contiguous() || dst.storage_type() != api::StorageType::BUFFER ||
      dst.gpu_memory_layout() != api::GPUMemoryLayout::TENSOR_WIDTH_PACKED ||
      src.dim() < 1 || src.dim() > 4 || dst.storage_offset() != 0 ||
      src.sizes().vec() != dst.sizes()) {
    return false;
  }

  const int64_t logical_last = src.size(src.dim() - 1);
  if (logical_last <= 0 || logical_last >= 4) {
    return false;
  }

  const int64_t physical_last = dst.gpu_sizes().back();
  if (physical_last != api::utils::align_up(logical_last, int64_t(4))) {
    return false;
  }

  const int64_t outer = src.numel() / logical_last;
  return api::utils::safe_downcast<int64_t>(dst.gpu_numel()) ==
      outer * physical_last;
}

bool can_pack_last_dim_padded_width_packed_strided_buffer(
    const Tensor& src,
    const vTensor& dst) {
  if (
      dst.storage_type() != api::StorageType::BUFFER ||
      dst.gpu_memory_layout() != api::GPUMemoryLayout::TENSOR_WIDTH_PACKED ||
      src.dim() < 1 || src.dim() > 4 || dst.storage_offset() != 0 ||
      src.sizes().vec() != dst.sizes()) {
    return false;
  }

  const int64_t logical_last = src.size(src.dim() - 1);
  if (logical_last <= 0 || logical_last >= 4) {
    return false;
  }

  const int64_t physical_last = dst.gpu_sizes().back();
  if (physical_last != api::utils::align_up(logical_last, int64_t(4))) {
    return false;
  }

  if (!has_last_dim_padded_contiguous_source_layout(src, physical_last)) {
    return false;
  }

  const int64_t outer = src.numel() / logical_last;
  return api::utils::safe_downcast<int64_t>(dst.gpu_numel()) ==
      outer * physical_last;
}

void pack_last_dim_padded_width_packed_contiguous_uint8_buffer(
    const Tensor& src,
    api::MemoryMap& dst_mapping,
    const vTensor& dst) {
  TORCH_CHECK(
      can_pack_last_dim_padded_width_packed_contiguous_buffer(src, dst),
      "Expected contiguous tensor with last-dim width padding");
  TORCH_CHECK(
      src.scalar_type() == at::kByte,
      "Expected uint8 tensor for padded width-packed upload fast path");

  const int64_t logical_last = src.size(src.dim() - 1);
  const int64_t outer = src.numel() / logical_last;

  uint32_t* const dst_words = dst_mapping.data<uint32_t>();
  const uint8_t* const src_ptr = src.const_data_ptr<uint8_t>();

  for (int64_t idx = 0; idx < outer; ++idx) {
    const int64_t src_offset = idx * logical_last;
    uint32_t packed = static_cast<uint32_t>(src_ptr[src_offset]);
    if (logical_last >= 2) {
      packed |= static_cast<uint32_t>(src_ptr[src_offset + 1]) << 8;
    }
    if (logical_last >= 3) {
      packed |= static_cast<uint32_t>(src_ptr[src_offset + 2]) << 16;
    }
    dst_words[idx] = packed;
  }
}

void pack_last_dim_padded_width_packed_strided_uint8_buffer(
    const Tensor& src,
    api::MemoryMap& dst_mapping,
    const vTensor& dst) {
  TORCH_CHECK(
      can_pack_last_dim_padded_width_packed_strided_buffer(src, dst),
      "Expected last-dim padded uint8 source view for width-packed upload");
  TORCH_CHECK(
      src.scalar_type() == at::kByte,
      "Expected uint8 tensor for padded width-packed strided upload fast path");

  const int64_t logical_last = src.size(src.dim() - 1);
  const int64_t physical_last = dst.gpu_sizes().back();
  const int64_t outer = src.numel() / logical_last;

  uint32_t* const dst_words = dst_mapping.data<uint32_t>();
  const uint8_t* const src_ptr = src.const_data_ptr<uint8_t>();

  for (int64_t idx = 0; idx < outer; ++idx) {
    const int64_t src_offset = idx * physical_last;
    uint32_t packed = static_cast<uint32_t>(src_ptr[src_offset]);
    if (logical_last >= 2) {
      packed |= static_cast<uint32_t>(src_ptr[src_offset + 1]) << 8;
    }
    if (logical_last >= 3) {
      packed |= static_cast<uint32_t>(src_ptr[src_offset + 2]) << 16;
    }
    dst_words[idx] = packed;
  }
}

template <typename T>
void pack_logical_tensor_to_buffer_mapping(
    const Tensor& src,
    api::MemoryMap& dst_mapping,
    const IntArrayRef physical_strides,
    const int64_t physical_numel,
    const int64_t storage_offset,
    const bool clear_destination) {
  T* const dst = dst_mapping.data<T>();
  if (clear_destination) {
    std::fill(dst, dst + physical_numel, T{});
  }

  const T* const src_ptr = src.const_data_ptr<T>();
  const std::vector<int64_t> logical_strides =
      calc_logical_contiguous_strides(src.sizes());

  for (int64_t logical_idx = 0; logical_idx < src.numel(); ++logical_idx) {
    int64_t remaining = logical_idx;
    int64_t physical_idx = storage_offset;
    for (int64_t dim = 0; dim < src.dim(); ++dim) {
      const int64_t stride = logical_strides[dim];
      const int64_t coord = stride == 0 ? 0 : remaining / stride;
      remaining = stride == 0 ? 0 : remaining % stride;
      physical_idx += coord * physical_strides[dim];
    }
    dst[physical_idx] = src_ptr[logical_idx];
  }
}

template <typename T>
void unpack_buffer_mapping_to_logical_tensor(
    api::MemoryMap& src_mapping,
    Tensor& dst,
    const IntArrayRef physical_strides,
    const int64_t storage_offset) {
  const T* const src_ptr = src_mapping.data<T>();
  T* const dst_ptr = dst.mutable_data_ptr<T>();
  const std::vector<int64_t> logical_strides =
      calc_logical_contiguous_strides(dst.sizes());

  for (int64_t logical_idx = 0; logical_idx < dst.numel(); ++logical_idx) {
    int64_t remaining = logical_idx;
    int64_t physical_idx = storage_offset;
    for (int64_t dim = 0; dim < dst.dim(); ++dim) {
      const int64_t stride = logical_strides[dim];
      const int64_t coord = stride == 0 ? 0 : remaining / stride;
      remaining = stride == 0 ? 0 : remaining % stride;
      physical_idx += coord * physical_strides[dim];
    }
    dst_ptr[logical_idx] = src_ptr[physical_idx];
  }
}

void pack_logical_tensor_to_buffer_mapping_dispatch(
    const Tensor& src,
    api::MemoryMap& dst_mapping,
    const IntArrayRef physical_strides,
    const int64_t physical_numel,
    const int64_t storage_offset,
    const bool clear_destination) {
  switch (src.scalar_type()) {
    case at::kFloat:
      pack_logical_tensor_to_buffer_mapping<float>(
          src,
          dst_mapping,
          physical_strides,
          physical_numel,
          storage_offset,
          clear_destination);
      return;
    case at::kHalf:
      pack_logical_tensor_to_buffer_mapping<c10::Half>(
          src,
          dst_mapping,
          physical_strides,
          physical_numel,
          storage_offset,
          clear_destination);
      return;
    case at::kBFloat16:
      pack_logical_tensor_to_buffer_mapping<c10::BFloat16>(
          src,
          dst_mapping,
          physical_strides,
          physical_numel,
          storage_offset,
          clear_destination);
      return;
    case at::kByte:
      pack_logical_tensor_to_buffer_mapping<uint8_t>(
          src,
          dst_mapping,
          physical_strides,
          physical_numel,
          storage_offset,
          clear_destination);
      return;
    case at::kChar:
      pack_logical_tensor_to_buffer_mapping<int8_t>(
          src,
          dst_mapping,
          physical_strides,
          physical_numel,
          storage_offset,
          clear_destination);
      return;
    case at::kInt:
      pack_logical_tensor_to_buffer_mapping<int32_t>(
          src,
          dst_mapping,
          physical_strides,
          physical_numel,
          storage_offset,
          clear_destination);
      return;
    case at::kLong:
      pack_logical_tensor_to_buffer_mapping<int64_t>(
          src,
          dst_mapping,
          physical_strides,
          physical_numel,
          storage_offset,
          clear_destination);
      return;
    case at::kBool:
      pack_logical_tensor_to_buffer_mapping<bool>(
          src,
          dst_mapping,
          physical_strides,
          physical_numel,
          storage_offset,
          clear_destination);
      return;
    default:
      TORCH_CHECK(
          false,
          "Unsupported scalar type for Vulkan buffer pack: ",
          src.scalar_type());
  }
}

void pack_cpu_to_host_visible_vulkan_buffer(const Tensor& src, vTensor& dst) {
  const bool cpu_timeline = api::cpu_timeline_logging_enabled();
  const uint64_t start_us = cpu_timeline ? api::cpu_timeline_now_us() : 0u;
  api::MemoryMap mapping(dst.buffer(), api::MemoryAccessType::WRITE);
  if (dst.has_direct_buffer_layout()) {
    const c10::MemoryFormat target_memory_format =
        memory_format_for_buffer_layout(dst.gpu_memory_layout());
    Tensor src_contig = src.contiguous(target_memory_format);
    memcpy_to_mapping(src_contig, mapping);
  } else if (
      src.scalar_type() == at::kByte &&
      can_pack_last_dim_padded_width_packed_strided_buffer(src, dst)) {
    pack_last_dim_padded_width_packed_strided_uint8_buffer(src, mapping, dst);
  } else if (
      src.scalar_type() == at::kByte) {
    const c10::MemoryFormat target_memory_format =
        memory_format_for_buffer_layout(dst.gpu_memory_layout());
    Tensor src_contig = src.contiguous(target_memory_format);
    if (can_pack_last_dim_padded_width_packed_contiguous_buffer(
            src_contig, dst)) {
      pack_last_dim_padded_width_packed_contiguous_uint8_buffer(
          src_contig, mapping, dst);
    } else {
      pack_logical_tensor_to_buffer_mapping_dispatch(
          src_contig,
          mapping,
          dst.gpu_strides(),
          dst.gpu_numel(),
          dst.storage_offset(),
          dst.storage_offset() == 0 &&
              api::utils::safe_downcast<int64_t>(dst.gpu_numel()) ==
                  dst.buffer_length());
    }
  } else {
    const c10::MemoryFormat target_memory_format =
        memory_format_for_buffer_layout(dst.gpu_memory_layout());
    Tensor src_contig = src.contiguous(target_memory_format);
    pack_logical_tensor_to_buffer_mapping_dispatch(
        src_contig,
        mapping,
        dst.gpu_strides(),
        dst.gpu_numel(),
        dst.storage_offset(),
        dst.storage_offset() == 0 &&
            api::utils::safe_downcast<int64_t>(dst.gpu_numel()) ==
                dst.buffer_length());
  }
  dst.mark_host_write();
  if (cpu_timeline) {
    log_cpu_timeline_copy_event(
        "copy_cpu_to_host_visible_vulkan_buffer",
        api::cpu_timeline_now_us() - start_us,
        dst,
        dst.has_direct_buffer_layout());
  }
}

void unpack_buffer_mapping_to_logical_tensor_dispatch(
    api::MemoryMap& src_mapping,
    Tensor& dst,
    const IntArrayRef physical_strides,
    const int64_t storage_offset) {
  switch (dst.scalar_type()) {
    case at::kFloat:
      unpack_buffer_mapping_to_logical_tensor<float>(
          src_mapping, dst, physical_strides, storage_offset);
      return;
    case at::kHalf:
      unpack_buffer_mapping_to_logical_tensor<c10::Half>(
          src_mapping, dst, physical_strides, storage_offset);
      return;
    case at::kBFloat16:
      unpack_buffer_mapping_to_logical_tensor<c10::BFloat16>(
          src_mapping, dst, physical_strides, storage_offset);
      return;
    case at::kByte:
      unpack_buffer_mapping_to_logical_tensor<uint8_t>(
          src_mapping, dst, physical_strides, storage_offset);
      return;
    case at::kChar:
      unpack_buffer_mapping_to_logical_tensor<int8_t>(
          src_mapping, dst, physical_strides, storage_offset);
      return;
    case at::kInt:
      unpack_buffer_mapping_to_logical_tensor<int32_t>(
          src_mapping, dst, physical_strides, storage_offset);
      return;
    case at::kLong:
      unpack_buffer_mapping_to_logical_tensor<int64_t>(
          src_mapping, dst, physical_strides, storage_offset);
      return;
    case at::kBool:
      unpack_buffer_mapping_to_logical_tensor<bool>(
          src_mapping, dst, physical_strides, storage_offset);
      return;
    default:
      TORCH_CHECK(
          false,
          "Unsupported scalar type for Vulkan buffer unpack: ",
          dst.scalar_type());
  }
}

void copy_staging_buffer_to_vtensor_buffer(
    api::Context* const context,
    api::VulkanBuffer& staging,
    vTensor& dst,
    const VkFence fence_handle,
    const api::VulkanSubmitOrigin fenced_submit_origin =
        api::VulkanSubmitOrigin::TensorCpuReadback) {
  api::PipelineBarrier pipeline_barrier{};
  context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
      pipeline_barrier,
      staging,
      dst.buffer(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::WRITE),
      {api::utils::safe_downcast<uint32_t>(staging.mem_size()), 0u, 0u},
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      fence_handle,
      fenced_submit_origin);
}

bool copy_vtensor_buffer_to_staging(
    api::Context* const context,
    vTensor& src,
    api::VulkanBuffer& staging,
    const VkFence fence_handle) {
  const bool raw_buffer_copy_legal =
      is_raw_buffer_readback_legal(src, staging.mem_size());
  const bool snapshot_readback_legal =
      is_buffer_snapshot_readback_legal(src, staging.mem_size());
  const bool use_packed_buffer_shader =
      requires_logical_pack_shader_for_readback(src);
  if (use_packed_buffer_shader) {
    utils::log_vulkan_op_hit(
        "aten::copy_.vulkan_to_cpu_buffer_pack_shader_required");
    return utils::pack_vtensor_to_staging(src, staging, fence_handle);
  }

  if (!raw_buffer_copy_legal && !snapshot_readback_legal) {
    std::ostringstream detail;
    detail << "storage=" << storage_type_name(src.storage_type())
           << " sizes=" << format_sizes(src.sizes())
           << " storage_offset=" << src.storage_offset()
           << " direct_buffer=" << (src.has_direct_buffer_layout() ? 1 : 0)
           << " last_write_was_compute="
           << (src.last_write_was_compute() ? 1 : 0)
           << " logical_bytes=" << src.nbytes()
           << " gpu_bytes=" << src.gpu_nbytes()
           << " buffer_length=" << src.buffer_length()
           << " staging_bytes=" << staging.mem_size();
    api::fail_vulkan(
        api::VulkanFailureClass::RawCopyIllegal,
        "aten::copy_.vulkan_to_cpu",
        "BufferReadbackIllegal",
        detail.str());
  }
  if (snapshot_readback_legal && !raw_buffer_copy_legal) {
    utils::log_vulkan_op_hit(
        "aten::copy_.vulkan_to_cpu_buffer_snapshot_metadata_readback");
  }

  api::PipelineBarrier pipeline_barrier{};
  return context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
      pipeline_barrier,
      src.buffer(pipeline_barrier, api::PipelineStage::TRANSFER),
      staging,
      {api::utils::safe_downcast<uint32_t>(staging.mem_size()), 0u, 0u},
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      fence_handle);
}

} // namespace

//
// Utility functions for memcpy
//

void note_vulkan_buffer_copy(
    VulkanBufferCopyReason reason,
    const vTensor& src,
    const vTensor& dst,
    const char* producer_label,
    const char* consumer_label) {
  note_buffer_copy_decision(make_buffer_copy_decision(
      reason,
      src,
      dst,
      producer_label,
      consumer_label,
      nullptr,
      nullptr));
}

void note_vulkan_buffer_copy(
    VulkanBufferCopyReason reason,
    const vTensor& src,
    const vTensor& dst,
    const char* producer_label,
    const char* consumer_label,
    const char* producer_role,
    const char* consumer_role) {
  note_buffer_copy_decision(make_buffer_copy_decision(
      reason,
      src,
      dst,
      producer_label,
      consumer_label,
      producer_role,
      consumer_role));
}

std::vector<int64_t> buffer_copy_counters_snapshot() {
  const VulkanBufferCopyCounters& counters = buffer_copy_counters();
  return {
      static_cast<int64_t>(counters.total.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.total_bytes.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.explicit_copy.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.contiguous.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.view_materialization.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reshape_materialization.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.permute_materialization.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.transpose_materialization.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.layout_conversion.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.attention_materialization.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.linear_materialization.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.conv_materialization.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.decoder_materialization.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.backbone_materialization.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.logical_noop_copy.load(std::memory_order_relaxed)),
  };
}

void reset_buffer_copy_counters() {
  VulkanBufferCopyCounters& counters = buffer_copy_counters();
  counters.total.store(0u, std::memory_order_relaxed);
  counters.total_bytes.store(0u, std::memory_order_relaxed);
  counters.explicit_copy.store(0u, std::memory_order_relaxed);
  counters.contiguous.store(0u, std::memory_order_relaxed);
  counters.view_materialization.store(0u, std::memory_order_relaxed);
  counters.reshape_materialization.store(0u, std::memory_order_relaxed);
  counters.permute_materialization.store(0u, std::memory_order_relaxed);
  counters.transpose_materialization.store(0u, std::memory_order_relaxed);
  counters.layout_conversion.store(0u, std::memory_order_relaxed);
  counters.attention_materialization.store(0u, std::memory_order_relaxed);
  counters.linear_materialization.store(0u, std::memory_order_relaxed);
  counters.conv_materialization.store(0u, std::memory_order_relaxed);
  counters.decoder_materialization.store(0u, std::memory_order_relaxed);
  counters.backbone_materialization.store(0u, std::memory_order_relaxed);
  counters.logical_noop_copy.store(0u, std::memory_order_relaxed);
}

std::vector<std::string> buffer_copy_aggregate_snapshot() {
  return buffer_copy_aggregate_profiler().snapshot();
}

void reset_buffer_copy_aggregate() {
  buffer_copy_aggregate_profiler().reset();
}

std::vector<std::string> clone_requirement_snapshot() {
  return clone_requirement_profiler().snapshot();
}

void reset_clone_requirement_snapshot() {
  clone_requirement_profiler().reset();
}

void memcpy_to_mapping(const Tensor& src, api::MemoryMap& dst_mapping) {
  if (src.dtype() == at::kFloat) {
    memcpy_to_mapping_impl<float>(src, dst_mapping);
  } else if (src.dtype() == at::kHalf) {
    memcpy_to_mapping_impl<c10::Half>(src, dst_mapping);
  } else if (src.dtype() == at::kBFloat16) {
    memcpy_to_mapping_impl<c10::BFloat16>(src, dst_mapping);
  } else if (src.dtype() == at::kByte) {
    memcpy_to_mapping_impl<uint8_t>(src, dst_mapping);
  } else if (src.dtype() == at::kChar) {
    memcpy_to_mapping_impl<int8_t>(src, dst_mapping);
  } else if (src.dtype() == at::kInt) {
    memcpy_to_mapping_impl<int32_t>(src, dst_mapping);
  } else if (src.dtype() == at::kLong) {
    memcpy_to_mapping_impl<int64_t>(src, dst_mapping);
  } else if (src.dtype() == c10::kQUInt8) {
    memcpy_to_mapping_impl<c10::quint8>(src, dst_mapping);
  } else if (src.dtype() == c10::kQInt8) {
    memcpy_to_mapping_impl<c10::qint8>(src, dst_mapping);
  } else if (src.dtype() == c10::kQInt32) {
    memcpy_to_mapping_impl<c10::qint32>(src, dst_mapping);
  } else if (src.dtype() == c10::kBool) {
    memcpy_to_mapping_uint8(src, dst_mapping);
  } else {
    TORCH_CHECK(
        false,
        "Invalid Data Type: expected c10::kQInt32, c10::kQInt8, c10::kQUInt8,",
        " c10::kBool, at::kByte, at::kChar, at::kInt, at::kLong, at::kHalf,",
        " at::kBFloat16, or at::kFloat but got ",
        src.dtype());
  }
}

void memcpy_from_mapping(api::MemoryMap& src_mapping, Tensor& dst) {
  if (dst.dtype() == at::kFloat) {
    memcpy_from_mapping_impl<float>(src_mapping, dst);
  } else if (dst.dtype() == at::kHalf) {
    memcpy_from_mapping_impl<c10::Half>(src_mapping, dst);
  } else if (dst.dtype() == at::kBFloat16) {
    memcpy_from_mapping_impl<c10::BFloat16>(src_mapping, dst);
  } else if (dst.dtype() == at::kByte) {
    memcpy_from_mapping_impl<uint8_t>(src_mapping, dst);
  } else if (dst.dtype() == at::kChar) {
    memcpy_from_mapping_impl<int8_t>(src_mapping, dst);
  } else if (dst.dtype() == at::kInt) {
    memcpy_from_mapping_impl<int32_t>(src_mapping, dst);
  } else if (dst.dtype() == at::kLong) {
    memcpy_from_mapping_impl<int64_t>(src_mapping, dst);
  } else if (dst.dtype() == c10::kQUInt8) {
    memcpy_from_mapping_impl<c10::quint8>(src_mapping, dst);
  } else if (dst.dtype() == c10::kQInt8) {
    memcpy_from_mapping_impl<c10::qint8>(src_mapping, dst);
  } else if (dst.dtype() == c10::kQInt32) {
    memcpy_from_mapping_impl<c10::qint32>(src_mapping, dst);
  } else if (dst.dtype() == c10::kBool) {
    memcpy_from_mapping_bool(src_mapping, dst);
  } else {
    TORCH_CHECK(
        false,
        "Invalid Data Type: expected c10::kQInt32, c10::kQInt8, c10::kQUInt8,",
        " c10::kBool, at::kByte, at::kChar, at::kInt, at::kLong, at::kHalf,",
        " at::kBFloat16, or at::kFloat but got ",
        dst.dtype());
  }
}

//
// CPU <-> GPU copy implementations (these functions use Transfer commands)
//

void transfer_cpu_to_vulkan(const Tensor& src, vTensor& v_dst) {
  api::Context* const context = v_dst.context();

  // Convert to dtype corresponding to the image format of the texture to
  // ensure that byte alignment is consistent when copying. In some cases
  // a 16 bit format will be used for at::kFloat.
  Tensor src_nc4hw =
      utils::nchw_to_nc4hw(src).to(convert_dtype(v_dst.texture_dtype()));

  api::StorageBuffer staging(context, v_dst.texture_dtype(), v_dst.gpu_numel());
  // Copy data into the staging buffer
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
    mapping.invalidate();

    memcpy_to_mapping(src_nc4hw, mapping);
  }

  api::PipelineBarrier pipeline_barrier{};
  utils::copy_buffer_to_vtensor(staging.buffer(), v_dst, pipeline_barrier);
}

void transfer_vulkan_to_cpu(vTensor& v_src, Tensor& dst) {
  api::Context* const context = v_src.context();

  // Temporary tensor to receive copied NC4HW data
  at::Tensor dst_tmp = utils::create_staging_tensor(v_src);
  const size_t staging_bytes = api::element_size(v_src.texture_dtype()) *
      static_cast<size_t>(v_src.gpu_numel());

  api::VulkanFence fence = context->fences().get_fence();

  if (staging_bytes > 0u) {
    auto staging =
        lookup_or_create_readback_buffer("texture_transfer", staging_bytes);
    std::unique_lock<std::mutex> staging_lock(staging.mutex());

    {
      // Refer to comment in submit_compute_job. When syncing with the GPU, the
      // context must not allow other threads to record dispatches into it between
      // between calling vkQueueSubmit and retiring the context state. Therefore,
      // cmd_mutex_ must be manually managed by the calling thread.
      std::unique_lock<std::mutex> context_lock(context->dispatch_lock());

      api::PipelineBarrier pipeline_barrier{};
      utils::copy_vtensor_to_buffer(
          v_src,
          staging.buffer(),
          pipeline_barrier,
          fence.get_submit_handle());

      fence.wait();

      log_copy_sync_event("transfer_vulkan_to_cpu", v_src, false);
      retire_after_fence_wait_and_release(context);
      // cmd_mutex_ will be released when exiting this scope.
    }

    // Copy data from buffer back to CPU tensor.
    {
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
      mapping.invalidate();

      memcpy_from_mapping(mapping, dst_tmp);
    }
  }

  context->fences().return_fence(fence);

  dst = utils::nc4hw_to_nchw(dst_tmp, v_src.sizes())
            .to(convert_dtype(v_src.dtype()));
}

static void transfer_vulkan_to_vulkan(vTensor& src, vTensor& dst) {
  TORCH_CHECK(
      src.context()->device_index() == dst.context()->device_index(),
      "Cross-device Vulkan copy is not supported yet: source is on vulkan:",
      src.context()->device_index(),
      " while destination is on vulkan:",
      dst.context()->device_index());
  api::Context* const context = dst.context();

  api::PipelineBarrier pipeline_barrier{};

  context->submit_copy<api::VulkanImage, api::VulkanImage>(
      // pipeline barrier
      pipeline_barrier,
      // images
      src.image(pipeline_barrier, api::PipelineStage::TRANSFER),
      dst.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::WRITE),
      // copy details
      src.extents(),
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      VK_NULL_HANDLE);
}

//
// CPU <-> GPU copy implementations (these functions use compute shaders)
//

void pack_cpu_to_vulkan(
    const Tensor& src,
    vTensor& dst,
    const api::VulkanSubmitOrigin fenced_submit_origin) {
  api::Context* const context = dst.context();

  if (dst.storage_type() == api::StorageType::BUFFER) {
    if (buffer_allocation_is_host_visible(dst)) {
      pack_cpu_to_host_visible_vulkan_buffer(src, dst);
      return;
    }

    const c10::MemoryFormat target_memory_format =
        memory_format_for_buffer_layout(dst.gpu_memory_layout());
    const bool copy_covers_full_buffer =
        dst.has_direct_buffer_layout() ||
        (dst.storage_offset() == 0 &&
         api::utils::safe_downcast<int64_t>(dst.gpu_numel()) ==
             dst.buffer_length());
    const int64_t staging_numel = copy_covers_full_buffer
        ? api::utils::safe_downcast<int64_t>(dst.gpu_numel())
        : dst.buffer_length();
    api::StorageBuffer staging(
        context,
        convert_dtype(src.scalar_type()),
        staging_numel);
    if (
        !copy_covers_full_buffer &&
        staging.buffer().mem_size() > 0u) {
      api::VulkanFence fence = context->fences().get_fence();
      {
        std::unique_lock<std::mutex> context_lock(context->dispatch_lock());
        api::PipelineBarrier pipeline_barrier{};
        context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
            pipeline_barrier,
            dst.buffer(
                pipeline_barrier,
                api::PipelineStage::TRANSFER,
                api::MemoryAccessType::READ),
            staging.buffer(),
            {api::utils::safe_downcast<uint32_t>(staging.buffer().mem_size()),
             0u,
             0u},
            {0u, 0u, 0u},
            {0u, 0u, 0u},
            fence.get_submit_handle());
        const bool cpu_timeline = api::cpu_timeline_logging_enabled();
        const uint64_t wait_start_us =
            cpu_timeline ? api::cpu_timeline_now_us() : 0u;
        fence.wait();
        if (cpu_timeline) {
          log_cpu_timeline_copy_event(
              "copy_preserve_vulkan_buffer_view_fence_wait",
              api::cpu_timeline_now_us() - wait_start_us,
              dst,
              false);
        }
        log_copy_sync_event("preserve_vulkan_buffer_view", dst, false);
        retire_after_fence_wait_and_release(context);
      }
      context->fences().return_fence(fence);
    }
    {
      const bool cpu_timeline = api::cpu_timeline_logging_enabled();
      const uint64_t start_us =
          cpu_timeline ? api::cpu_timeline_now_us() : 0u;
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
      if (dst.has_direct_buffer_layout()) {
        Tensor src_contig = src.contiguous(target_memory_format);
        memcpy_to_mapping(src_contig, mapping);
      } else if (
          src.scalar_type() == at::kByte &&
          copy_covers_full_buffer &&
          can_pack_last_dim_padded_width_packed_strided_buffer(src, dst)) {
        pack_last_dim_padded_width_packed_strided_uint8_buffer(
            src, mapping, dst);
      } else if (
          src.scalar_type() == at::kByte &&
          copy_covers_full_buffer &&
          can_pack_last_dim_padded_width_packed_contiguous_buffer(
              src.contiguous(target_memory_format), dst)) {
        Tensor src_contig = src.contiguous(target_memory_format);
        pack_last_dim_padded_width_packed_contiguous_uint8_buffer(
            src_contig, mapping, dst);
      } else {
        Tensor src_contig = src.contiguous(target_memory_format);
        pack_logical_tensor_to_buffer_mapping_dispatch(
            src_contig,
            mapping,
            dst.gpu_strides(),
            staging_numel,
            dst.storage_offset(),
            copy_covers_full_buffer);
      }
      if (cpu_timeline) {
        log_cpu_timeline_copy_event(
            "copy_cpu_to_vulkan_buffer_staging_pack",
            api::cpu_timeline_now_us() - start_us,
            dst,
            dst.has_direct_buffer_layout());
      }
    }
    if (staging.buffer().mem_size() > 0u) {
      api::VulkanFence fence = context->fences().get_fence();
      {
        std::unique_lock<std::mutex> context_lock(context->dispatch_lock());
        copy_staging_buffer_to_vtensor_buffer(
            context,
            staging.buffer(),
            dst,
            fence.get_submit_handle(),
            fenced_submit_origin);
        const bool cpu_timeline = api::cpu_timeline_logging_enabled();
        const uint64_t wait_start_us =
            cpu_timeline ? api::cpu_timeline_now_us() : 0u;
        fence.wait();
        if (cpu_timeline) {
          log_cpu_timeline_copy_event(
              "copy_cpu_to_vulkan_buffer_fence_wait",
              api::cpu_timeline_now_us() - wait_start_us,
              dst,
              dst.has_direct_buffer_layout());
        }
        log_copy_sync_event(
            "pack_cpu_to_vulkan_buffer",
            dst,
            dst.has_direct_buffer_layout());
        retire_after_fence_wait_and_release(context);
      }
      context->fences().return_fence(fence);
    }
    return;
  }

  // Ensure that src is contiguous in its memory format
  Tensor src_contig = src.contiguous(src.suggest_memory_format());

  // Note that the float data type has been enforced for the storage buffer
  // below. The reason for this is that the nchw_to_image and image_to_nchw
  // shaders which perform the transfer to/from an image texture expect a buffer
  // of floats as input. GLSL/Vulkan does not natively support 16 bit arithmetic
  // types, so for now storage buffers created for compute shaders must define
  // floats as their base data type.
  api::StorageBuffer staging(context, api::kFloat, dst.gpu_numel());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
    std::memset(mapping.data<uint8_t>(), 0, mapping.nbytes());

    // If the dtype() of src is at::kHalf, then first convert it to 32 bit
    // float. This is required since the nchw_to_image shader uses a float
    // buffer as input (note that at::kFloat is used to create the StorageBuffer
    // above).
    if (src.dtype() == at::kHalf) {
      memcpy_to_mapping(src_contig.to(at::kFloat), mapping);
    } else {
      memcpy_to_mapping(src_contig, mapping);
    }
  }
  utils::pack_staging_to_vtensor(staging.buffer(), dst);
}

void pack_vulkan_to_cpu(vTensor& src, Tensor& dst) {
  TORCH_CHECK(
      !src.is_quantized(),
      "Copy of vulkan quantized tensors to cpu is currently disabled!");
  api::Context* const context = src.context();

  if (src.storage_type() == api::StorageType::BUFFER) {
    const bool raw_buffer_readback_legal = is_raw_buffer_readback_legal(src);
    const bool shader_packed_buffer =
        requires_logical_pack_shader_for_readback(src);
    const int64_t staging_length =
        shader_packed_buffer
        ? api::utils::safe_downcast<int64_t>(src.numel())
        : src.has_direct_buffer_layout()
        ? api::utils::safe_downcast<int64_t>(src.gpu_numel())
        : src.buffer_length();
    api::VulkanFence fence = context->fences().get_fence();
    const size_t staging_bytes = api::element_size(src.dtype()) *
        static_cast<size_t>(staging_length);

    Tensor dst_tmp = at::empty(
        src.sizes(),
        at::device(at::kCPU).dtype(convert_dtype(src.dtype())));
    if (src.has_direct_buffer_layout()) {
      dst_tmp = dst_tmp.to(
          memory_format_for_buffer_layout(src.gpu_memory_layout()));
    }

    if (staging_bytes > 0u) {
      auto submit_to_staging = [&](api::VulkanBuffer& staging_buffer) {
        bool submitted_to_gpu = false;
        std::unique_lock<std::mutex> context_lock(context->dispatch_lock());
        submitted_to_gpu = copy_vtensor_buffer_to_staging(
            context, src, staging_buffer, fence.get_submit_handle());
        if (submitted_to_gpu) {
          const bool cpu_timeline = api::cpu_timeline_logging_enabled();
          const uint64_t wait_start_us =
              cpu_timeline ? api::cpu_timeline_now_us() : 0u;
          fence.wait();
          if (cpu_timeline) {
            log_cpu_timeline_copy_event(
                "copy_vulkan_to_cpu_buffer_fence_wait",
                api::cpu_timeline_now_us() - wait_start_us,
                src,
                src.has_direct_buffer_layout());
          }
          log_copy_sync_event(
              "pack_vulkan_to_cpu_buffer",
              src,
              src.has_direct_buffer_layout());
          retire_command_resources_after_fence_wait(context);
        }
        context_lock.unlock();
        if (submitted_to_gpu) {
          release_retired_objects_after_context_unlock(context);
        }
      };
      auto copy_from_staging = [&](api::VulkanBuffer& staging_buffer) {
        const bool cpu_timeline = api::cpu_timeline_logging_enabled();
        const uint64_t start_us =
            cpu_timeline ? api::cpu_timeline_now_us() : 0u;
        api::MemoryMap mapping(staging_buffer, api::MemoryAccessType::READ);
        TORCH_CHECK(
            mapping.nbytes() >= staging_bytes,
            "Vulkan CPU readback staging buffer is smaller than requested: ",
            mapping.nbytes(),
            " < ",
            staging_bytes);
        mapping.invalidate();
        utils::log_vulkan_op_hit("aten::copy_.vulkan_to_cpu_buffer_map_begin");
        if (shader_packed_buffer) {
          utils::log_vulkan_op_hit(
              "aten::copy_.vulkan_to_cpu_buffer_packed_shader");
          memcpy_from_mapping(mapping, dst_tmp);
        } else {
          if (src.has_direct_buffer_layout()) {
            utils::log_vulkan_op_hit(
                "aten::copy_.vulkan_to_cpu_buffer_direct_metadata_unpack");
          }
          utils::log_vulkan_op_hit(
              "aten::copy_.vulkan_to_cpu_buffer_metadata_unpack");
          unpack_buffer_mapping_to_logical_tensor_dispatch(
              mapping, dst_tmp, src.gpu_strides(), src.storage_offset());
        }
        utils::log_vulkan_op_hit("aten::copy_.vulkan_to_cpu_buffer_map_end");
        if (cpu_timeline) {
          log_cpu_timeline_copy_event(
              "copy_vulkan_to_cpu_buffer_map_unpack",
              api::cpu_timeline_now_us() - start_us,
              src,
              src.has_direct_buffer_layout());
        }
      };

      auto staging = lookup_or_create_readback_buffer("buffer_pack", staging_bytes);
      std::unique_lock<std::mutex> staging_lock(staging.mutex());
      submit_to_staging(staging.buffer());
      copy_from_staging(staging.buffer());
    }

    context->fences().return_fence(fence);
    utils::log_vulkan_op_hit("aten::copy_.vulkan_to_cpu_buffer_dst_copy_begin");
    const bool cpu_timeline = api::cpu_timeline_logging_enabled();
    const uint64_t dst_copy_start_us =
        cpu_timeline ? api::cpu_timeline_now_us() : 0u;
    dst.copy_(dst_tmp);
    if (cpu_timeline) {
      log_cpu_timeline_copy_event(
          "copy_vulkan_to_cpu_buffer_dst_copy",
          api::cpu_timeline_now_us() - dst_copy_start_us,
          src,
          src.has_direct_buffer_layout());
    }
    utils::log_vulkan_op_hit("aten::copy_.vulkan_to_cpu_buffer_dst_copy_end");
    return;
  }

  // Refer to the comment in pack_cpu_to_vulkan for why at::kFloat is specified
  // for the storage buffer below.
  api::VulkanFence fence = context->fences().get_fence();
  const size_t staging_bytes = api::element_size(api::kFloat) *
      static_cast<size_t>(src.gpu_numel());

  if (staging_bytes > 0u) {
    auto staging = lookup_or_create_readback_buffer("texture_pack", staging_bytes);
    std::unique_lock<std::mutex> staging_lock(staging.mutex());

    {
      // Refer to comment in submit_compute_job. When syncing with the GPU, the
      // context must not allow other threads to record dispatches into it between
      // between calling vkQueueSubmit and retiring the context state. Therefore,
      // cmd_mutex_ must be manually managed by the calling thread.
      std::unique_lock<std::mutex> context_lock(context->dispatch_lock());

        const bool submitted_to_gpu = utils::pack_vtensor_to_staging(
            src, staging.buffer(), fence.get_submit_handle());

      // Only wait on the fence if work was actually submitted to the GPU.
      // Otherwise, it will hang indefinitely.
      if (submitted_to_gpu) {
        const bool cpu_timeline = api::cpu_timeline_logging_enabled();
        const uint64_t wait_start_us =
            cpu_timeline ? api::cpu_timeline_now_us() : 0u;
        fence.wait();
        if (cpu_timeline) {
          log_cpu_timeline_copy_event(
              "copy_texture_to_cpu_fence_wait",
              api::cpu_timeline_now_us() - wait_start_us,
              src,
              false);
        }
        log_copy_sync_event("pack_vulkan_to_cpu_texture", src, false);
        retire_after_fence_wait_and_release(context);
      }
      // cmd_mutex_ will be released when exiting this scope.
    }

    // Copy data from buffer back to CPU tensor.
    {
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
      mapping.invalidate();

      // If the dtype() of dst is at::kHalf, then copy the data into a float
      // version of it first, similar to pack_cpu_to_vulkan().
      if (dst.dtype() == at::kHalf) {
        Tensor dst_float = dst.to(at::kFloat);
        memcpy_from_mapping(mapping, dst_float);
        dst = dst_float.to(at::kHalf);
      } else {
        memcpy_from_mapping(mapping, dst);
      }
    }
  }

  context->fences().return_fence(fence);
}

//
// Copy op implementations
//

Tensor& copy_(Tensor& dst, const Tensor& src) {
  // Check that sizes are equal
  TORCH_CHECK(
      dst.sizes() == src.sizes(), "Vulkan copy_: Tensor sizes are mismatched!");
  Tensor src_to_copy = src.is_vulkan()
      ? materialize_decomposed_attention_candidate_if_needed(src)
      : src;
  src_to_copy = src_to_copy.is_vulkan()
      ? materialize_deferred_attention_query_scale_candidate_if_needed(src_to_copy)
      : src_to_copy;
  src_to_copy = src_to_copy.is_vulkan()
      ? materialize_deferred_linear_gelu_candidate_if_needed(src_to_copy)
      : src_to_copy;
  src_to_copy = src_to_copy.is_vulkan()
      ? materialize_deferred_add_layer_norm_candidate_if_needed(src_to_copy)
      : src_to_copy;
  src_to_copy = src_to_copy.is_vulkan()
      ? materialize_deferred_layer_scale_candidate_if_needed(src_to_copy)
      : src_to_copy;
  src_to_copy = src_to_copy.is_vulkan()
      ? materialize_deferred_image_normalize_candidate_if_needed(src_to_copy)
      : src_to_copy;

  // X -> Vulkan
  if (at::kVulkan == dst.device().type()) {
    vTensor& v_self = convert(dst);
    api::set_current_device(v_self.context()->device_index());

    // Vulkan -> Vulkan
    if (at::kVulkan == src_to_copy.device().type()) {
      Tensor src_casted = src_to_copy;
      if (src_to_copy.scalar_type() != dst.scalar_type()) {
        src_casted =
            utils::cast_vulkan_tensor_dtype(src_to_copy, dst.scalar_type());
      }

      vTensor& v_src = convert(src_casted);
      TORCH_CHECK(
          v_src.context()->device_index() == v_self.context()->device_index(),
          "Cross-device Vulkan copy is not supported yet: source is on vulkan:",
          v_src.context()->device_index(),
          " while destination is on vulkan:",
          v_self.context()->device_index());
      api::set_current_device(v_self.context()->device_index());
      const bool can_direct_copy =
          v_src.dtype() == v_self.dtype() &&
          v_src.storage_type() != api::StorageType::BUFFER &&
          v_self.storage_type() != api::StorageType::BUFFER;
      if (can_direct_copy) {
        transfer_vulkan_to_vulkan(v_src, v_self);
      } else if (can_copy_vulkan_buffer_to_buffer_on_device(v_src, v_self)) {
        const std::string producer = tensor_provenance_writer(src_casted);
        const std::string producer_role = tensor_provenance_route(src_casted);
        const std::string allocation_label = api::current_allocation_label();
        const std::string consumer =
            (allocation_label.empty() || allocation_label == "unlabeled")
            ? "aten::copy_.buffer_to_buffer"
            : allocation_label + ".buffer_to_buffer";
        copy_vulkan_buffer_to_buffer_on_device(
            v_src,
            v_self,
            producer.c_str(),
            consumer.c_str(),
            producer_role.c_str(),
            "raw_copy_destination");
      } else {
        c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
        c10::InferenceMode inference_mode_guard(false);
        Tensor cpu_src = from_vulkan(v_src);
        pack_cpu_to_vulkan(cpu_src, v_self);
      }
    }
    // CPU -> Vulkan
    else {
      Tensor cpu_src = src_to_copy;
      if (cpu_src.scalar_type() != dst.scalar_type()) {
        cpu_src = cpu_src.to(dst.scalar_type());
      }
      pack_cpu_to_vulkan(cpu_src, v_self);
    }
  }
  // Vulkan -> X
  else if (at::kVulkan == src_to_copy.device().type()) {
    vTensor& v_src = convert(src_to_copy);
    api::set_current_device(v_src.context()->device_index());

    // Vulkan -> CPU
    if (dst.device().is_cpu()) {
      pack_vulkan_to_cpu(v_src, dst);
      v_src.context()->submit_pending_work_and_poll_retire();
    } else {
      api::fail_vulkan(
          api::VulkanFailureClass::Unsupported,
          "aten::copy_",
          "UnsupportedVulkanDestination",
          "Vulkan copy supports Vulkan->CPU or CPU->Vulkan for this path");
    }
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "Invalid code path taken! Either the source or the destination tensor "
        "was expected to be Vulkan a tensor!  Incorrect dispatch?");
  }

  return dst;
}

vTensor to_vulkan(
    at::Tensor& src,
    const api::StorageType storage_type,
    const std::optional<c10::DeviceIndex> device_index) {
  TORCH_CHECK(
      src.device().type() == at::kCPU,
      "Vulkan to_vulkan(): input tensor must be a CPU tensor!")

  const c10::DeviceIndex resolved_device_index =
      device_index.has_value() ? *device_index : api::current_device();
  api::set_current_device(resolved_device_index);

  const api::StorageType resolved_storage_type =
      (api::requires_buffer_storage(convert_dtype(src.scalar_type()), src.dim()) ||
       should_force_buffer_storage_for_to_vulkan(src))
      ? api::StorageType::BUFFER
      : storage_type;
  const bool use_host_visible_buffer =
      resolved_storage_type == api::StorageType::BUFFER &&
      should_use_host_visible_buffer_for_to_vulkan(src);

  if (
      resolved_storage_type == api::StorageType::BUFFER &&
      should_preserve_compact_cpu_buffer_view_for_to_vulkan(src)) {
    vTensor flat_storage{
        api::context(resolved_device_index),
        {src.numel()},
        convert_dtype(src.scalar_type()),
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
        /*allocate_memory=*/true,
        /*buffer_gpu_only=*/!use_host_visible_buffer,
    };

    Tensor flat_src = src.reshape({src.numel()});
    ops::pack_cpu_to_vulkan(flat_src, flat_storage);

    const std::vector<int64_t> strides =
        calc_logical_contiguous_strides(src.sizes());
    return vTensor(
        flat_storage,
        src.sizes().vec(),
        strides,
        strides,
        /*storage_offset=*/0);
  }

  vTensor v_ret{
      api::context(resolved_device_index),
      src.sizes().vec(),
      convert_dtype(src.scalar_type()),
      resolved_storage_type,
      get_gpu_memory_layout(resolved_storage_type, src.suggest_memory_format()),
      /*allocate_memory=*/true,
      /*buffer_gpu_only=*/!use_host_visible_buffer,
  };

  ops::pack_cpu_to_vulkan(src, v_ret);

  return v_ret;
}

at::Tensor to_vulkan_labeled(at::Tensor src, std::string label) {
  if (src.is_vulkan()) {
    return src;
  }
  TORCH_CHECK(
      src.device().type() == at::kCPU,
      "Vulkan to_vulkan_labeled(): input tensor must be a CPU or Vulkan tensor!");
  (void)label;

  const c10::Device vulkan_device(at::kVulkan, api::current_device());
  Tensor result = at::empty(src.sizes(), src.options().device(vulkan_device));
  ops::copy_(result, src);
  if (should_flush_after_labeled_to_vulkan(src)) {
    vTensor& v_result = convert(result);
    v_result.context()->flush();
  }
  record_tensor_write(
      result,
      "vulkan_prepack::to_vulkan_labeled",
      label.empty() ? "upload" : label.c_str(),
      {src});
  return result;
}

at::Tensor from_vulkan(vTensor& v_src) {
  at::TensorOptions opt(at::kCPU);
  opt = opt.dtype(convert_dtype(v_src.dtype()));

  c10::MemoryFormat v_src_memory_format = c10::MemoryFormat::Contiguous;

  switch (v_src.gpu_memory_layout()) {
    case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
      v_src_memory_format = c10::MemoryFormat::Contiguous;
      break;
    case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
      // ChannelsLast is only valid for rank-4 CPU tensors. Lower-rank Vulkan
      // tensors can still use channels-packed GPU layout, but the CPU staging
      // tensor for readback/conversion must stay contiguous.
      v_src_memory_format =
          v_src.sizes().size() == 4 ? c10::MemoryFormat::ChannelsLast
                                    : c10::MemoryFormat::Contiguous;
      break;
    default:
      api::fail_vulkan(
          api::VulkanFailureClass::Unsupported,
          "aten::copy_.vulkan_to_cpu",
          "UnsupportedMemoryLayoutForReadback",
          "No corresponding CPU memory format");
  }

  at::Tensor ret = at::empty(v_src.sizes(), opt).to(v_src_memory_format);
  ops::pack_vulkan_to_cpu(v_src, ret);
  return ret;
}

//
// VulkanImpl
//

struct VulkanImpl final : public at::vulkan::VulkanImplInterface {
  bool is_vulkan_available() const override {
    return api::available();
  }

  Tensor& vulkan_copy_(Tensor& self, const Tensor& src) const override {
    return vulkan::ops::copy_(self, src);
  }
};
static at::vulkan::VulkanImplRegistrar g_vulkan_impl(new VulkanImpl());

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
