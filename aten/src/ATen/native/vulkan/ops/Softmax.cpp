#include <ATen/native/vulkan/ops/Softmax.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/LayoutTransitions.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/TensorState.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/planning/Capabilities.h>
#include <ATen/native/vulkan/planning/InferenceGraphs.h>
#include <ATen/native/vulkan/planning/ExecutionObjects.h>
#include <ATen/native/vulkan/planning/ExecutionPrograms.h>
#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/ReplayTensorState.h>
#include <ATen/native/vulkan/planning/RoutePolicy.h>
#include <ATen/Functions.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/InferenceMode.h>
#include <ATen/ops/scaled_dot_product_attention_ops.h>
#include <torch/library.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace {

using namespace api::utils;

constexpr int32_t kTiledSdpaLocalSizeX = 16;
constexpr int32_t kTiledSdpaMaxOutputsPerThread = 32;
constexpr int64_t kTiledSdpaMaxValueDim =
    static_cast<int64_t>(kTiledSdpaLocalSizeX) *
    static_cast<int64_t>(kTiledSdpaMaxOutputsPerThread);
constexpr int32_t kTiledSdpaBufferMaxQueryValuesPerThread = 8;
constexpr int64_t kTiledSdpaBufferMaxHeadDim =
    static_cast<int64_t>(kTiledSdpaLocalSizeX) *
    static_cast<int64_t>(kTiledSdpaBufferMaxQueryValuesPerThread);
constexpr int64_t kTiledSdpaBufferDefaultFastPathMaxSequence = 512;
constexpr int64_t kTiledSdpaBufferVisionFastPathMaxSequence = 512;
constexpr uint32_t kBufferSoftmaxLastDimLocalSizeX = 128u;
constexpr uint32_t kBufferSoftmaxLastDimMaxWorkGroupsX = 65535u;
constexpr uint32_t kBufferSoftmaxDimLocalSizeX = 128u;
constexpr int32_t kRuntimeProgramSdpaWideLocalSizeX = 32;
constexpr int32_t kRuntimeProgramSdpaWideMaxOutputsPerThread = 16;
constexpr int32_t kRuntimeProgramSdpaWideMaxQueryValuesPerThread = 8;
constexpr int32_t kRuntimeProgramSdpaHead64LocalSizeX = 64;
constexpr int32_t kRuntimeProgramSdpaHead64MaxOutputsPerThread = 1;
constexpr int32_t kRuntimeProgramSdpaHead64MaxQueryValuesPerThread = 1;
constexpr int32_t kRuntimeProgramSdpaHead64QueryRowsPerWorkgroupQ4 = 4;
constexpr int64_t kRuntimeProgramSdpaWideMaxHeadDim =
    static_cast<int64_t>(kRuntimeProgramSdpaWideLocalSizeX) *
    static_cast<int64_t>(kRuntimeProgramSdpaWideMaxQueryValuesPerThread);
constexpr int64_t kRuntimeProgramSdpaWideMaxValueDim =
    static_cast<int64_t>(kRuntimeProgramSdpaWideLocalSizeX) *
    static_cast<int64_t>(kRuntimeProgramSdpaWideMaxOutputsPerThread);
constexpr int64_t kRuntimeProgramSdpaWideLongSequenceMin = 1024;
constexpr int64_t kRuntimeProgramSdpaWideLongSequenceMinHeadDim = 64;

bool supports_effective_qtile_q4_subgroup_kernel() {
  const auto capabilities = utils::query_vulkan_runtime_capability_profile();
  constexpr uint32_t kRequiredSubgroupSize =
      static_cast<uint32_t>(kRuntimeProgramSdpaHead64LocalSizeX);
  const bool supports_required_compute_stage =
      (capabilities.required_subgroup_size_stages &
       VK_SHADER_STAGE_COMPUTE_BIT) != 0u;
  return capabilities.has_compute_full_subgroups &&
      capabilities.has_subgroup_size_control &&
      supports_required_compute_stage &&
      capabilities.min_subgroup_size <= kRequiredSubgroupSize &&
      capabilities.max_subgroup_size >= kRequiredSubgroupSize;
}

enum class RuntimeProgramBufferFusedKernelVariant : uint8_t {
  Narrow16 = 0u,
  Wide32 = 1u,
  Head64 = 2u,
  Head64Query4 = 3u,
};

enum class VulkanAttentionFastPath : uint8_t {
  Unknown = 0u,
  ScoresValueFloatSingleQuery = 1u,
  ScoresValueFloatQueryTile = 2u,
  Fallback = 3u,
};

enum class VulkanAttentionRejectReason : uint8_t {
  None = 0u,
  InputNotVulkan = 1u,
  UnsupportedDType = 2u,
  UnsupportedRank = 3u,
  UnsupportedLayout = 4u,
  UnsupportedHeadDim = 5u,
  UnsupportedValueDim = 6u,
  MaskPresent = 7u,
  DropoutNonZero = 8u,
  Causal = 9u,
  ShapeUnsupported = 10u,
  Unknown = 11u,
};

struct VulkanAttentionPlanDecision final {
  VulkanAttentionFastPath selected{VulkanAttentionFastPath::Unknown};
  VulkanAttentionRejectReason reject{VulkanAttentionRejectReason::None};
  int64_t batch_heads{0};
  int64_t target_len{0};
  int64_t source_len{0};
  int64_t head_dim{0};
  int64_t value_dim{0};
  bool query_vulkan{false};
  bool key_vulkan{false};
  bool value_vulkan{false};
  bool query_direct_buffer{false};
  bool key_direct_buffer{false};
  bool value_direct_buffer{false};
  bool output_direct_buffer{false};
  bool dtype_float{false};
  bool self_attention_shape{false};
  bool mask_present{false};
  bool dropout_nonzero{false};
  bool causal{false};
  int64_t query_tile{1};
};

struct VulkanAttentionPlanCounters final {
  std::atomic<uint64_t> total{0u};
  std::atomic<uint64_t> single_query_hit{0u};
  std::atomic<uint64_t> qtile_hit{0u};
  std::atomic<uint64_t> reject_dtype{0u};
  std::atomic<uint64_t> reject_layout{0u};
  std::atomic<uint64_t> reject_mask{0u};
  std::atomic<uint64_t> reject_dropout{0u};
  std::atomic<uint64_t> reject_causal{0u};
  std::atomic<uint64_t> reject_head_dim{0u};
  std::atomic<uint64_t> reject_shape{0u};
  std::atomic<uint64_t> qtile_q4_hit{0u};
  std::atomic<uint64_t> qtile_q4_shared_hit{0u};
  std::atomic<uint64_t> qtile_q4_subgroup_hit{0u};
};

enum class DecomposedAttentionStage : uint8_t {
  Scores,
  Probs,
};

struct DecomposedAttentionCandidate {
  Tensor query;
  Tensor key;
  Tensor key_t;
  DecomposedAttentionStage stage;
  float query_scale{1.0f};
  uint64_t producer_storage_id{0u};
  uint64_t producer_generation{0u};
  uint64_t producer_logical_desc_hash{0u};
};

struct DeferredAttentionQueryScaleCandidate {
  Tensor query;
  float scale{1.0f};
  uint64_t producer_storage_id{0u};
  uint64_t producer_generation{0u};
  uint64_t producer_logical_desc_hash{0u};
};

VulkanAttentionPlanCounters& attention_plan_counters() {
  static VulkanAttentionPlanCounters counters;
  return counters;
}

const std::string& vulkan_attention_plan_log_path() {
  static const std::string path = []() {
    const char* const env = std::getenv("PYTORCH_VULKAN_ATTENTION_PLAN_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

void append_vulkan_attention_plan_log(
    const VulkanAttentionPlanDecision& decision,
    const char* label) {
  const auto& path = vulkan_attention_plan_log_path();
  if (path.empty()) {
    return;
  }

  std::ofstream out(path, std::ios::app);
  out << "attention_plan"
      << " label=" << (label ? label : "unknown")
      << " selected=" << static_cast<int>(decision.selected)
      << " reject=" << static_cast<int>(decision.reject)
      << " batch_heads=" << decision.batch_heads
      << " target_len=" << decision.target_len
      << " source_len=" << decision.source_len
      << " head_dim=" << decision.head_dim
      << " value_dim=" << decision.value_dim
      << " query_tile=" << decision.query_tile
      << " q_buffer=" << (decision.query_direct_buffer ? 1 : 0)
      << " k_buffer=" << (decision.key_direct_buffer ? 1 : 0)
      << " v_buffer=" << (decision.value_direct_buffer ? 1 : 0)
      << " out_buffer=" << (decision.output_direct_buffer ? 1 : 0)
      << " dtype_float=" << (decision.dtype_float ? 1 : 0)
      << " self_attention=" << (decision.self_attention_shape ? 1 : 0)
      << " mask=" << (decision.mask_present ? 1 : 0)
      << " dropout=" << (decision.dropout_nonzero ? 1 : 0)
      << " causal=" << (decision.causal ? 1 : 0)
      << '\n';
}

void note_attention_plan_decision(
    const VulkanAttentionPlanDecision& decision,
    const char* label) {
  VulkanAttentionPlanCounters& counters = attention_plan_counters();
  counters.total.fetch_add(1u, std::memory_order_relaxed);
  switch (decision.selected) {
    case VulkanAttentionFastPath::ScoresValueFloatSingleQuery:
      counters.single_query_hit.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanAttentionFastPath::ScoresValueFloatQueryTile:
      counters.qtile_hit.fetch_add(1u, std::memory_order_relaxed);
      if (decision.query_tile ==
          kRuntimeProgramSdpaHead64QueryRowsPerWorkgroupQ4) {
        counters.qtile_q4_hit.fetch_add(1u, std::memory_order_relaxed);
      }
      break;
    case VulkanAttentionFastPath::Fallback:
    case VulkanAttentionFastPath::Unknown:
      break;
  }

  switch (decision.reject) {
    case VulkanAttentionRejectReason::UnsupportedDType:
      counters.reject_dtype.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanAttentionRejectReason::UnsupportedLayout:
      counters.reject_layout.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanAttentionRejectReason::MaskPresent:
      counters.reject_mask.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanAttentionRejectReason::DropoutNonZero:
      counters.reject_dropout.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanAttentionRejectReason::Causal:
      counters.reject_causal.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanAttentionRejectReason::UnsupportedHeadDim:
    case VulkanAttentionRejectReason::UnsupportedValueDim:
      counters.reject_head_dim.fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanAttentionRejectReason::ShapeUnsupported:
      counters.reject_shape.fetch_add(1u, std::memory_order_relaxed);
      break;
    default:
      break;
  }
  append_vulkan_attention_plan_log(decision, label);
}

constexpr size_t kMaxDecomposedAttentionCandidates = 128;
constexpr size_t kMaxDeferredAttentionQueryScaleCandidates = 32;
thread_local bool g_materializing_deferred_attention_query_scale = false;

const void* decomposed_attention_key(const Tensor& tensor) {
  if (tensor.is_vulkan()) {
    const vTensor& v_tensor = convert(tensor);
    if (v_tensor.storage_type() == api::StorageType::BUFFER) {
      return static_cast<const void*>(&v_tensor.buffer());
    }
  }
  return static_cast<const void*>(tensor.unsafeGetTensorImpl());
}

std::mutex& decomposed_attention_candidate_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<const void*, DecomposedAttentionCandidate>&
decomposed_attention_candidates() {
  static std::unordered_map<const void*, DecomposedAttentionCandidate>
      candidates;
  return candidates;
}

std::mutex& deferred_attention_query_scale_candidate_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<const void*, DeferredAttentionQueryScaleCandidate>&
deferred_attention_query_scale_candidates() {
  static std::unordered_map<
      const void*,
      DeferredAttentionQueryScaleCandidate>
      candidates;
  return candidates;
}

const void* deferred_attention_query_scale_key(const Tensor& tensor) {
  if (tensor.is_vulkan()) {
    const vTensor& v_tensor = convert(tensor);
    if (v_tensor.storage_type() == api::StorageType::BUFFER) {
      return static_cast<const void*>(&v_tensor.buffer());
    }
  }
  return static_cast<const void*>(tensor.unsafeGetTensorImpl());
}

void stamp_decomposed_attention_candidate(
    const Tensor& tensor,
    DecomposedAttentionCandidate& candidate) {
  const VulkanTensorStateDesc state = inspect_tensor_state(tensor);
  candidate.producer_storage_id = state.storage_id;
  candidate.producer_generation = state.generation;
  candidate.producer_logical_desc_hash = state.logical_desc_hash;
}

void stamp_deferred_attention_query_scale_candidate(
    const Tensor& tensor,
    DeferredAttentionQueryScaleCandidate& candidate) {
  const VulkanTensorStateDesc state = inspect_tensor_state(tensor);
  candidate.producer_storage_id = state.storage_id;
  candidate.producer_generation = state.generation;
  candidate.producer_logical_desc_hash = state.logical_desc_hash;
}

bool can_retarget_decomposed_attention_candidate(
    const Tensor& tensor,
    const DecomposedAttentionCandidate& candidate) {
  if (!tensor.is_vulkan() || tensor.scalar_type() != kFloat) {
    return false;
  }
  const int64_t batch_heads = candidate.query.size(0);
  const int64_t target_length = candidate.query.size(1);
  const int64_t source_length = candidate.key.size(1);
  if (
      tensor.dim() == 3 &&
      tensor.size(0) == batch_heads &&
      tensor.size(1) == target_length &&
      tensor.size(2) == source_length) {
    return true;
  }
  if (
      tensor.dim() == 4 &&
      tensor.size(0) == 1 &&
      tensor.size(1) == batch_heads &&
      tensor.size(2) == target_length &&
      tensor.size(3) == source_length) {
    return true;
  }
  return false;
}

bool matches_decomposed_attention_candidate_stamp(
    const Tensor& tensor,
    const DecomposedAttentionCandidate& candidate) {
  if (!tensor.is_vulkan()) {
    return false;
  }
  const VulkanTensorStateDesc state = inspect_tensor_state(tensor);
  return state.storage_id == candidate.producer_storage_id &&
      state.generation == candidate.producer_generation &&
      (state.logical_desc_hash == candidate.producer_logical_desc_hash ||
       can_retarget_decomposed_attention_candidate(tensor, candidate));
}

bool can_retarget_deferred_attention_query_scale_candidate(
    const Tensor& tensor,
    const DeferredAttentionQueryScaleCandidate& candidate);

bool matches_deferred_attention_query_scale_candidate_stamp(
    const Tensor& tensor,
    const DeferredAttentionQueryScaleCandidate& candidate) {
  if (!tensor.is_vulkan()) {
    return false;
  }
  const VulkanTensorStateDesc state = inspect_tensor_state(tensor);
  return state.storage_id == candidate.producer_storage_id &&
      state.generation == candidate.producer_generation &&
      (state.logical_desc_hash == candidate.producer_logical_desc_hash ||
       can_retarget_deferred_attention_query_scale_candidate(tensor, candidate));
}

bool can_retarget_deferred_attention_query_scale_candidate(
    const Tensor& tensor,
    const DeferredAttentionQueryScaleCandidate& candidate) {
  if (!tensor.is_vulkan() || tensor.scalar_type() != kFloat) {
    return false;
  }
  if (tensor.sizes().equals(candidate.query.sizes())) {
    return true;
  }
  if (
      candidate.query.dim() == 4 && tensor.dim() == 3 &&
      candidate.query.size(0) == 1 &&
      candidate.query.size(1) == tensor.size(0) &&
      candidate.query.size(2) == tensor.size(1) &&
      candidate.query.size(3) == tensor.size(2) &&
      tensor.size(0) == 6 &&
      tensor.size(2) == 64) {
    return true;
  }
  if (
      candidate.query.dim() == 3 && tensor.dim() == 4 &&
      tensor.size(0) == 1 &&
      tensor.size(1) == candidate.query.size(0) &&
      tensor.size(2) == candidate.query.size(1) &&
      tensor.size(3) == candidate.query.size(2) &&
      candidate.query.size(0) == 6 &&
      candidate.query.size(2) == 64) {
    return true;
  }
  return false;
}

DeferredAttentionQueryScaleCandidate retarget_deferred_attention_query_scale(
    const Tensor& tensor,
    DeferredAttentionQueryScaleCandidate candidate) {
  if (!candidate.query.sizes().equals(tensor.sizes())) {
    candidate.query = candidate.query.reshape(tensor.sizes());
  }
  return candidate;
}

Tensor detached_attention_tensor(const Tensor& tensor) {
  return tensor.requires_grad() ? tensor.detach() : tensor;
}

bool can_start_deferred_attention_query_scale_candidate(
    const Tensor& query,
    const float scale) {
  if (
      g_materializing_deferred_attention_query_scale ||
      !std::isfinite(scale) ||
      scale <= 0.0f ||
      scale > 1.0f ||
      !query.is_vulkan() ||
      query.scalar_type() != kFloat ||
      ((query.dim() != 3 ||
        query.size(0) != 6 ||
        query.size(1) < 512 ||
        query.size(2) != 64) &&
       (query.dim() != 4 ||
        query.size(0) != 1 ||
        query.size(1) != 6 ||
        query.size(2) < 512 ||
        query.size(3) != 64))) {
    return false;
  }

  const vTensor& v_query = convert(query);
  return v_query.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_view_fast_path(v_query);
}

bool can_start_decomposed_attention_candidate(
    const Tensor& query,
    const Tensor& key_t) {
  if (
      !query.is_vulkan() ||
      !key_t.is_vulkan() ||
      query.scalar_type() != kFloat ||
      key_t.scalar_type() != kFloat ||
      query.dim() != 3 ||
      key_t.dim() != 3 ||
      query.size(0) != key_t.size(0) ||
      query.size(2) != key_t.size(1) ||
      query.size(1) != key_t.size(2) ||
      query.size(1) < 512 ||
      query.size(2) != 64) {
    return false;
  }

  const vTensor& v_query = convert(query);
  const vTensor& v_key_t = convert(key_t);
  return v_query.storage_type() == api::StorageType::BUFFER &&
      v_key_t.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_view_fast_path(v_query) &&
      utils::supports_buffer_view_fast_path(v_key_t);
}

bool can_consume_decomposed_attention_candidate(
    const DecomposedAttentionCandidate& candidate,
    const Tensor& value) {
  if (
      !value.is_vulkan() ||
      value.scalar_type() != kFloat ||
      value.dim() != 3 ||
      candidate.stage != DecomposedAttentionStage::Probs ||
      candidate.query.size(0) != value.size(0) ||
      candidate.key.size(1) != value.size(1) ||
      value.size(2) != 64) {
    return false;
  }

  const vTensor& v_value = convert(value);
  return v_value.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_view_fast_path(v_value);
}

std::optional<DecomposedAttentionCandidate>
lookup_decomposed_attention_candidate(const Tensor& tensor) {
  std::lock_guard<std::mutex> lock(decomposed_attention_candidate_mutex());
  auto& candidates = decomposed_attention_candidates();
  const auto it = candidates.find(decomposed_attention_key(tensor));
  if (it == candidates.end()) {
    return std::nullopt;
  }
  if (!matches_decomposed_attention_candidate_stamp(tensor, it->second)) {
    utils::log_vulkan_op_hit(
        "aten::decomposed_attention_bridge.stale_candidate");
    candidates.erase(it);
    return std::nullopt;
  }
  return it->second;
}

std::optional<DecomposedAttentionCandidate>
take_decomposed_attention_candidate(const Tensor& tensor) {
  std::lock_guard<std::mutex> lock(decomposed_attention_candidate_mutex());
  auto& candidates = decomposed_attention_candidates();
  const auto it = candidates.find(decomposed_attention_key(tensor));
  if (it == candidates.end()) {
    return std::nullopt;
  }
  if (!matches_decomposed_attention_candidate_stamp(tensor, it->second)) {
    utils::log_vulkan_op_hit(
        "aten::decomposed_attention_bridge.stale_candidate");
    candidates.erase(it);
    return std::nullopt;
  }
  DecomposedAttentionCandidate candidate = it->second;
  candidates.erase(it);
  return candidate;
}

void register_decomposed_attention_candidate(
    const Tensor& tensor,
    DecomposedAttentionCandidate candidate) {
  std::lock_guard<std::mutex> lock(decomposed_attention_candidate_mutex());
  auto& candidates = decomposed_attention_candidates();
  if (candidates.size() >= kMaxDecomposedAttentionCandidates) {
    utils::log_vulkan_op_hit("aten::decomposed_attention_bridge.registry_clear");
    candidates.clear();
  }
  stamp_decomposed_attention_candidate(tensor, candidate);
  candidates[decomposed_attention_key(tensor)] = std::move(candidate);
}

std::optional<DeferredAttentionQueryScaleCandidate>
lookup_deferred_attention_query_scale_candidate(const Tensor& tensor) {
  DeferredAttentionQueryScaleCandidate candidate;
  {
    std::lock_guard<std::mutex> lock(
        deferred_attention_query_scale_candidate_mutex());
    auto& candidates = deferred_attention_query_scale_candidates();
    const auto it = candidates.find(deferred_attention_query_scale_key(tensor));
    if (it == candidates.end()) {
      return std::nullopt;
    }
    candidate = it->second;
    if (!can_retarget_deferred_attention_query_scale_candidate(
            tensor, candidate) ||
        !matches_deferred_attention_query_scale_candidate_stamp(
            tensor, candidate)) {
      utils::log_vulkan_op_hit(
          "aten::attention_query_scale_bridge.stale_candidate");
      candidates.erase(it);
      return std::nullopt;
    }
  }
  return retarget_deferred_attention_query_scale(tensor, std::move(candidate));
}

std::optional<DeferredAttentionQueryScaleCandidate>
take_deferred_attention_query_scale_candidate(const Tensor& tensor) {
  DeferredAttentionQueryScaleCandidate candidate;
  {
    std::lock_guard<std::mutex> lock(
        deferred_attention_query_scale_candidate_mutex());
    auto& candidates = deferred_attention_query_scale_candidates();
    const auto it = candidates.find(deferred_attention_query_scale_key(tensor));
    if (it == candidates.end()) {
      return std::nullopt;
    }
    candidate = it->second;
    if (!can_retarget_deferred_attention_query_scale_candidate(
            tensor, candidate) ||
        !matches_deferred_attention_query_scale_candidate_stamp(
            tensor, candidate)) {
      utils::log_vulkan_op_hit(
          "aten::attention_query_scale_bridge.stale_candidate");
      candidates.erase(it);
      return std::nullopt;
    }
    candidates.erase(it);
  }
  return retarget_deferred_attention_query_scale(tensor, std::move(candidate));
}

void register_deferred_attention_query_scale_candidate(
    const Tensor& tensor,
    DeferredAttentionQueryScaleCandidate candidate) {
  std::lock_guard<std::mutex> lock(
      deferred_attention_query_scale_candidate_mutex());
  auto& candidates = deferred_attention_query_scale_candidates();
  if (candidates.size() >= kMaxDeferredAttentionQueryScaleCandidates) {
    utils::log_vulkan_op_hit(
        "aten::attention_query_scale_bridge.registry_clear");
    candidates.clear();
  }
  stamp_deferred_attention_query_scale_candidate(tensor, candidate);
  candidates[deferred_attention_query_scale_key(tensor)] = std::move(candidate);
}

class DeferredAttentionQueryScaleMaterializeGuard final {
 public:
  DeferredAttentionQueryScaleMaterializeGuard() {
    previous_ = g_materializing_deferred_attention_query_scale;
    g_materializing_deferred_attention_query_scale = true;
  }

  ~DeferredAttentionQueryScaleMaterializeGuard() {
    g_materializing_deferred_attention_query_scale = previous_;
  }

 private:
  bool previous_{false};
};

Tensor materialize_deferred_attention_query_scale_candidate_impl(
    const Tensor& tensor) {
  if (!tensor.is_vulkan()) {
    return tensor;
  }
  auto candidate = take_deferred_attention_query_scale_candidate(tensor);
  if (!candidate.has_value()) {
    return tensor;
  }

  utils::log_vulkan_op_hit("aten::attention_query_scale_bridge.materialize");
  DeferredAttentionQueryScaleMaterializeGuard guard;
  return at::mul(candidate->query, candidate->scale);
}

Tensor scaled_decomposed_attention_query(
    const DecomposedAttentionCandidate& candidate) {
  if (candidate.query_scale == 1.0f) {
    return candidate.query;
  }
  DeferredAttentionQueryScaleMaterializeGuard guard;
  return at::mul(candidate.query, candidate.query_scale);
}

Tensor materialize_decomposed_attention_candidate(
    const Tensor& tensor,
    DecomposedAttentionCandidate candidate) {
  const std::vector<int64_t> scores_sizes{
      candidate.query.size(0),
      candidate.query.size(1),
      candidate.key.size(1),
  };
  const auto restore_public_shape = [&tensor](Tensor materialized) {
    if (materialized.sizes().equals(tensor.sizes())) {
      return materialized;
    }
    TORCH_CHECK(
        materialized.numel() == tensor.numel(),
        "Vulkan decomposed attention materialization cannot restore alias "
        "shape: materialized sizes=",
        materialized.sizes(),
        " alias sizes=",
        tensor.sizes());
    return materialized.reshape(tensor.sizes());
  };

  if (candidate.stage == DecomposedAttentionStage::Scores) {
    utils::log_vulkan_op_hit(
        "aten::decomposed_attention_bridge.materialize_scores");
    Tensor output = tensor.sizes().equals(scores_sizes)
        ? tensor
        : utils::create_buffer_tensor(
              scores_sizes,
              kFloat,
              /*persistent=*/false);
    output = bmm_buffer_out_vulkan(
        scaled_decomposed_attention_query(candidate),
        candidate.key_t,
        output);
    return restore_public_shape(output);
  }

  utils::log_vulkan_op_hit(
      "aten::decomposed_attention_bridge.materialize_probs");
  Tensor scores = utils::create_buffer_tensor(
      scores_sizes,
      kFloat,
      /*persistent=*/false);
  Tensor probs = tensor.sizes().equals(scores_sizes)
      ? tensor
      : utils::create_buffer_tensor(
            scores_sizes,
            kFloat,
            /*persistent=*/false);
  bmm_buffer_out_vulkan(
      scaled_decomposed_attention_query(candidate),
      candidate.key_t,
      scores);
  probs = softmax_buffer_lastdim_out_vulkan(scores, probs);
  return restore_public_shape(probs);
}

std::optional<Tensor> make_decomposed_attention_merge_friendly_output(
    const Tensor& query,
    const Tensor& value) {
  if (
      query.dim() == 3 && value.dim() == 3 &&
      query.scalar_type() == kFloat && value.scalar_type() == kFloat &&
      query.size(0) == 6 && query.size(1) == 601 && query.size(2) == 64 &&
      value.size(0) == 6 && value.size(1) == 601 && value.size(2) == 64) {
    // DAv2 vits immediately consumes attention as:
    //   [B,H,N,D].transpose(1,2).reshape(B,N,H*D)
    // Store the 3D bmm result in token-major physical order so that the
    // subsequent transpose+reshape can become a direct [N,H*D] buffer view.
    Tensor base = utils::mark_tensor_execution(
        convert(vTensor{
            api::context(),
            {1, query.size(1), query.size(0), value.size(2)},
            api::kFloat,
            api::StorageType::BUFFER,
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
        }),
        api::ExecutionLayout::BUFFER_DIRECT);
    const std::vector<int64_t> sizes{
        query.size(0),
        query.size(1),
        value.size(2),
    };
    const std::vector<int64_t> token_major_strides{
        value.size(2),
        query.size(0) * value.size(2),
        1,
    };
    Tensor output = make_buffer_metadata_view_checked(
        base,
        sizes,
        token_major_strides,
        token_major_strides,
        0,
        "aten::decomposed_attention_bridge.merge_friendly_output");
    utils::log_vulkan_op_hit(
        "aten::decomposed_attention_bridge.merge_friendly_output");
    return output;
  }
  utils::log_vulkan_op_hit(
      "aten::decomposed_attention_bridge.merge_friendly_output_disabled");
  return std::nullopt;
}

bool can_use_runtime_program_buffer_fused_fast_path(
    const vTensor& v_query,
    const vTensor& v_key,
    const vTensor& v_value) {
  return v_query.storage_type() == api::StorageType::BUFFER &&
      v_key.storage_type() == api::StorageType::BUFFER &&
      v_value.storage_type() == api::StorageType::BUFFER &&
      v_query.dtype() == api::kFloat &&
      v_key.dtype() == api::kFloat &&
      v_value.dtype() == api::kFloat &&
      v_query.sizes().size() == 3 &&
      v_key.sizes().size() == 3 &&
      v_value.sizes().size() == 3 &&
      v_query.sizes()[2] <= kRuntimeProgramSdpaWideMaxHeadDim &&
      v_value.sizes()[2] <= kRuntimeProgramSdpaWideMaxValueDim &&
      utils::supports_buffer_reduction_compute(v_query) &&
      utils::supports_buffer_reduction_compute(v_key) &&
      utils::supports_buffer_reduction_compute(v_value);
}

bool can_use_runtime_program_buffer_fused_fast_path(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value) {
  return can_use_runtime_program_buffer_fused_fast_path(
      convert(query), convert(key), convert(value));
}

bool can_use_head64_query_tile_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value) {
  if (query.dim() != 3 || key.dim() != 3 || value.dim() != 3) {
    return false;
  }
  if (query.scalar_type() != kFloat || key.scalar_type() != kFloat ||
      value.scalar_type() != kFloat) {
    return false;
  }
  if (query.size(0) <= 0 || query.size(1) < 128 || key.size(1) < 128) {
    return false;
  }
  return query.size(0) == key.size(0) && query.size(0) == value.size(0) &&
      query.size(2) == 64 && key.size(2) == 64 && value.size(2) == 64 &&
      key.size(1) == value.size(1);
}

Tensor scaled_dot_product_attention_tiled_3d_buffer_out_vulkan(
    const Tensor& query_arg,
    const Tensor& key_arg,
    const Tensor& value_arg,
    Tensor& output_arg);

RuntimeProgramBufferFusedKernelVariant select_runtime_program_buffer_fused_variant(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value) {
  if (can_use_head64_query_tile_attention(query, key, value)) {
    return RuntimeProgramBufferFusedKernelVariant::Head64Query4;
  }

  const bool requires_wide_head_dim =
      query.size(2) > kTiledSdpaBufferMaxHeadDim ||
      key.size(2) > kTiledSdpaBufferMaxHeadDim;
  const bool is_long_sequence =
      std::max(query.size(1), key.size(1)) >=
      kRuntimeProgramSdpaWideLongSequenceMin;
  const bool long_sequence_head64 =
      is_long_sequence && query.size(2) == 64 && key.size(2) == 64 &&
      value.size(2) == 64;
  const bool exact_head64 =
      query.size(2) == 64 && key.size(2) == 64 && value.size(2) == 64;
  const bool long_sequence_head_dim64_or_larger =
      is_long_sequence &&
      std::max(query.size(2), key.size(2)) >=
          kRuntimeProgramSdpaWideLongSequenceMinHeadDim;
  // The head64-specialized runtime variants currently produce invalid values on
  // diffusion-style SDPA. Keep exact head64 on the generic fused kernel until
  // that kernel family has targeted cross-device correctness coverage.
  if (long_sequence_head64 || exact_head64) {
    return RuntimeProgramBufferFusedKernelVariant::Narrow16;
  }
  return requires_wide_head_dim || long_sequence_head_dim64_or_larger
      ? RuntimeProgramBufferFusedKernelVariant::Wide32
      : RuntimeProgramBufferFusedKernelVariant::Narrow16;
}

const char* runtime_program_buffer_fused_variant_log_name(
    const RuntimeProgramBufferFusedKernelVariant variant) {
  switch (variant) {
    case RuntimeProgramBufferFusedKernelVariant::Narrow16:
      return "aten::scaled_dot_product_attention.runtime_program_buffer_fused_narrow";
    case RuntimeProgramBufferFusedKernelVariant::Wide32:
      return "aten::scaled_dot_product_attention.runtime_program_buffer_fused_wide";
    case RuntimeProgramBufferFusedKernelVariant::Head64:
      return "aten::scaled_dot_product_attention.runtime_program_buffer_fused_head64";
    case RuntimeProgramBufferFusedKernelVariant::Head64Query4:
      return "aten::scaled_dot_product_attention.runtime_program_buffer_fused_head64_q4";
  }
  return "aten::scaled_dot_product_attention.runtime_program_buffer_fused_unknown";
}

Tensor scaled_dot_product_attention_runtime_fused_3d_buffer_out_vulkan(
    const Tensor& query_arg,
    const Tensor& key_arg,
    const Tensor& value_arg,
    Tensor& output_arg,
    float query_scale = 1.0f);

bool is_vision_backbone_attention_policy(
    const utils::VulkanRuntimePolicy& runtime_policy) {
  return runtime_policy.attention_kernel_family ==
      utils::VulkanAttentionKernelFamily::BufferMath &&
      (runtime_policy.request.workload_class ==
           utils::VulkanWorkloadClass::VisionBackbone ||
       (runtime_policy.request.model_domain == utils::VulkanModelDomain::Vision &&
        runtime_policy.request.execution_phase ==
            utils::VulkanExecutionPhase::Backbone));
}

bool is_generic_attention_policy(
    const utils::VulkanRuntimePolicy& runtime_policy) {
  return runtime_policy.request.workload_class ==
      utils::VulkanWorkloadClass::Attention &&
      runtime_policy.request.model_domain == utils::VulkanModelDomain::Generic;
}

int64_t tiled_sdpa_buffer_fast_path_max_sequence(
    const utils::VulkanRuntimePolicy& runtime_policy) {
  return is_vision_backbone_attention_policy(runtime_policy)
      ? kTiledSdpaBufferVisionFastPathMaxSequence
      : kTiledSdpaBufferDefaultFastPathMaxSequence;
}

Tensor prepare_buffer_math_input_direct(const Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_vulkan(),
      "Vulkan SDPA buffer math expects Vulkan tensors");
  const vTensor& v_tensor = convert(tensor);
  if (
      v_tensor.storage_type() == api::StorageType::BUFFER &&
      v_tensor.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
      v_tensor.has_direct_buffer_layout() &&
      utils::supports_buffer_view_fast_path(v_tensor)) {
    return utils::mark_tensor_execution(
        tensor, utils::resolve_buffer_execution_layout(v_tensor));
  }

  if (
      v_tensor.storage_type() == api::StorageType::BUFFER &&
      !v_tensor.has_direct_buffer_layout()) {
    utils::log_vulkan_op_hit(
        "aten::scaled_dot_product_attention.materialize_metadata_view");
  }
  Tensor buffer_tensor =
      utils::ensure_buffer_storage(tensor, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  TORCH_CHECK(
      convert(buffer_tensor).has_direct_buffer_layout(),
      "Vulkan SDPA buffer math expects materialized direct buffers");
  return utils::mark_tensor_execution(
      buffer_tensor,
      utils::resolve_buffer_execution_layout(convert(buffer_tensor)));
}

Tensor materialize_buffer_attention_output_view(const Tensor& tensor) {
  if (!tensor.is_vulkan()) {
    return tensor;
  }
  const vTensor& v_tensor = convert(tensor);
  if (
      v_tensor.storage_type() == api::StorageType::BUFFER &&
      !v_tensor.has_direct_buffer_layout()) {
    utils::log_vulkan_op_hit(
        "aten::scaled_dot_product_attention.materialize_output_view");
    Tensor output = utils::ensure_buffer_storage(
        tensor, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
    return utils::mark_tensor_execution(
        output, api::ExecutionLayout::BUFFER_DIRECT);
  }
  return tensor;
}

bool can_use_attention_buffer_math_ops(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value) {
  return convert(query).storage_type() == api::StorageType::BUFFER &&
      convert(key).storage_type() == api::StorageType::BUFFER &&
      convert(value).storage_type() == api::StorageType::BUFFER;
}

bool has_float_attention_inputs(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value) {
  return query.scalar_type() == kFloat && key.scalar_type() == kFloat &&
      value.scalar_type() == kFloat;
}

bool can_use_attention_runtime_buffer_math_replay(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value) {
  return has_float_attention_inputs(query, key, value) &&
      can_use_attention_buffer_math_ops(query, key, value);
}

utils::VulkanKVCacheSpec make_attention_kv_cache_spec(
    const Tensor& tensor,
    const utils::VulkanKVCachePlanningDesc& desc) {
  return utils::VulkanKVCacheSpec{
      tensor.scalar_type(),
      std::vector<int64_t>(tensor.sizes().begin(), tensor.sizes().end()),
      1,
      desc.prefer_buffer_storage ? api::ExecutionLayout::BUFFER_DIRECT
                                 : api::ExecutionLayout::TEXTURE,
      desc.prefer_buffer_storage
          ? api::GPUMemoryLayout::TENSOR_WIDTH_PACKED
          : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
      desc.prefer_buffer_storage ? api::StorageType::BUFFER
                                 : api::StorageType::TEXTURE_3D,
      desc.prefer_persistent_object,
  };
}

size_t attention_runtime_scratch_bytes(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value) {
  const auto packed_buffer_bytes =
      [](std::initializer_list<int64_t> logical_sizes) -> size_t {
    std::vector<int64_t> physical_sizes(logical_sizes.begin(), logical_sizes.end());
    if (!physical_sizes.empty()) {
      physical_sizes.back() =
          api::utils::align_up(physical_sizes.back(), INT64_C(4));
    }
    return static_cast<size_t>(
        c10::elementSize(kFloat) * api::utils::multiply_integers(physical_sizes));
  };

  const size_t scores_bytes = packed_buffer_bytes(
      {query.size(0), query.size(1), key.size(1)});
  const size_t output_bytes = packed_buffer_bytes(
      {query.size(0), query.size(1), value.size(2)});
  return scores_bytes + scores_bytes + output_bytes;
}

std::optional<utils::AttentionRuntimeProgram>
lookup_attention_runtime_program_for_inputs(
    const utils::VulkanRuntimePolicy& input_policy,
    const utils::VulkanAttentionPolicy& attention_policy,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value) {
  if (
      !input_policy.execution_program_plan.has_value() ||
      input_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::AttentionRuntime) {
    return std::nullopt;
  }

  const auto cache_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_attention_request(
          attention_policy,
          query,
          key,
          value,
          utils::VulkanTensorRole::Cache));
  const auto scratch_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_attention_request(
          attention_policy,
          query,
          key,
          value,
          utils::VulkanTensorRole::Scratch));
  const std::optional<utils::VulkanKVCacheSpec> key_cache_spec =
      cache_policy.kv_cache_plan.has_value()
      ? std::optional<utils::VulkanKVCacheSpec>(
            make_attention_kv_cache_spec(key, *cache_policy.kv_cache_plan))
      : std::nullopt;
  const std::optional<utils::VulkanKVCacheSpec> value_cache_spec =
      cache_policy.kv_cache_plan.has_value()
      ? std::optional<utils::VulkanKVCacheSpec>(
            make_attention_kv_cache_spec(value, *cache_policy.kv_cache_plan))
      : std::nullopt;
  const std::optional<utils::VulkanScratchArenaSpec> scratch_spec =
      scratch_policy.scratch_arena_plan.has_value() &&
          !can_use_runtime_program_buffer_fused_fast_path(query, key, value)
      ? std::optional<utils::VulkanScratchArenaSpec>(utils::VulkanScratchArenaSpec{
            kByte,
            std::max<size_t>(
                attention_runtime_scratch_bytes(query, key, value),
                scratch_policy.scratch_arena_plan->min_arena_bytes),
            scratch_policy.scratch_arena_plan->alignment,
            scratch_policy.scratch_arena_plan->prefer_buffer_storage
                ? api::ExecutionLayout::BUFFER_DIRECT
                : api::ExecutionLayout::TEXTURE,
            scratch_policy.scratch_arena_plan->prefer_buffer_storage
                ? api::GPUMemoryLayout::TENSOR_WIDTH_PACKED
                : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
            scratch_policy.scratch_arena_plan->prefer_buffer_storage
                ? api::StorageType::BUFFER
                : api::StorageType::TEXTURE_3D,
            scratch_policy.scratch_arena_plan->prefer_reusable_arena,
        })
      : std::nullopt;

  return utils::lookup_or_create_labeled_attention_runtime_program(
      utils::make_vulkan_runtime_object_label(input_policy.request, "program"),
      input_policy.attention_kernel_family,
      key_cache_spec,
      value_cache_spec,
      scratch_spec,
      key.size(1),
      value.size(1),
      *input_policy.execution_program_plan);
}

void prime_attention_runtime_objects(
    const utils::VulkanRuntimePolicy& input_policy,
    const utils::VulkanAttentionPolicy& attention_policy,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value) {
  log_attention_kernel_family_choice(input_policy);
  log_attention_execution_strategy_choice(input_policy);
  (void)lookup_attention_runtime_program_for_inputs(
      input_policy, attention_policy, query, key, value);
}

std::vector<int64_t> calc_attention_width_packed_buffer_sizes(IntArrayRef sizes) {
  std::vector<int64_t> physical_sizes(sizes.begin(), sizes.end());
  if (!physical_sizes.empty()) {
    physical_sizes.back() =
        api::utils::align_up(physical_sizes.back(), INT64_C(4));
  }
  return physical_sizes;
}

size_t attention_buffer_descriptor_nbytes(
    IntArrayRef sizes,
    const ScalarType dtype) {
  return static_cast<size_t>(
      api::element_size(convert_dtype(dtype)) *
      api::utils::multiply_integers(
          calc_attention_width_packed_buffer_sizes(sizes)));
}

std::vector<int64_t> calc_attention_contiguous_strides(IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (int64_t idx = static_cast<int64_t>(sizes.size()) - 2; idx >= 0; --idx) {
    strides[idx] = strides[idx + 1] * std::max<int64_t>(sizes[idx + 1], 1);
  }
  return strides;
}

std::vector<int64_t> calc_attention_width_packed_buffer_strides(
    IntArrayRef sizes) {
  return calc_attention_contiguous_strides(
      calc_attention_width_packed_buffer_sizes(sizes));
}

Tensor make_attention_scratch_buffer_alias(
    const utils::ScratchArena& arena,
    const utils::VulkanScratchSlice& slice,
    IntArrayRef sizes,
    const ScalarType dtype) {
  const size_t required_bytes = attention_buffer_descriptor_nbytes(sizes, dtype);
  TORCH_CHECK(
      required_bytes <= slice.size_bytes,
      "Attention-runtime scratch alias requested ",
      required_bytes,
      " bytes from a slice sized for ",
      slice.size_bytes,
      " bytes");

  const int64_t element_size =
      static_cast<int64_t>(c10::elementSize(dtype));
  TORCH_CHECK(
      element_size > 0,
      "Attention-runtime scratch alias requires a concrete element size");
  TORCH_CHECK(
      slice.offset_bytes % static_cast<size_t>(element_size) == 0u &&
          arena.size_bytes() % static_cast<size_t>(element_size) == 0u,
      "Attention-runtime scratch alias requires byte-aligned offsets for dtype ",
      dtype);

  const int64_t storage_offset =
      static_cast<int64_t>(slice.offset_bytes / static_cast<size_t>(element_size));
  const int64_t buffer_length_override =
      static_cast<int64_t>(arena.size_bytes() / static_cast<size_t>(element_size));
  const api::ExecutionLayout execution_layout =
      slice.offset_bytes == 0u ? api::ExecutionLayout::BUFFER_DIRECT
                               : api::ExecutionLayout::BUFFER_VIEW;
  return make_typed_buffer_metadata_view_checked(
      arena.storage(),
      dtype,
      sizes,
      calc_attention_contiguous_strides(sizes),
      calc_attention_width_packed_buffer_strides(sizes),
      storage_offset,
      buffer_length_override,
      execution_layout,
      "aten::scaled_dot_product_attention.scratch");
}

Tensor reserve_attention_scratch_tensor(
    utils::ScratchArena& arena,
    IntArrayRef sizes,
    const ScalarType dtype) {
  const size_t required_bytes = attention_buffer_descriptor_nbytes(sizes, dtype);
  const utils::VulkanScratchSlice slice = arena.reserve(
      required_bytes,
      std::max<uint32_t>(
          arena.alignment(),
          static_cast<uint32_t>(std::max<int64_t>(
              1, static_cast<int64_t>(c10::elementSize(dtype))))));
  return make_attention_scratch_buffer_alias(arena, slice, sizes, dtype);
}

Tensor ensure_attention_runtime_direct_buffer(const Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_vulkan(),
      "Attention-runtime replay expects Vulkan tensors");
  const vTensor& v_tensor = convert(tensor);
  if (
      v_tensor.storage_type() == api::StorageType::BUFFER &&
      v_tensor.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
      v_tensor.has_direct_buffer_layout() &&
      utils::supports_buffer_view_fast_path(v_tensor)) {
    return utils::mark_tensor_execution(
        tensor, utils::resolve_buffer_execution_layout(v_tensor));
  }

  if (
      v_tensor.storage_type() == api::StorageType::BUFFER &&
      !v_tensor.has_direct_buffer_layout()) {
    utils::log_vulkan_op_hit(
        "aten::scaled_dot_product_attention.materialize_metadata_view");
  }
  Tensor output = utils::ensure_buffer_storage(
      tensor, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  const vTensor& v_output = convert(output);
  TORCH_CHECK(
      v_output.storage_type() == api::StorageType::BUFFER &&
          v_output.has_direct_buffer_layout() &&
          utils::supports_buffer_view_fast_path(v_output),
      "Attention-runtime replay expects buffer-backed tensors with supported "
      "view semantics");
  return utils::mark_tensor_execution(
      output, utils::resolve_buffer_execution_layout(v_output));
}

void copy_tensor_for_attention_replay(Tensor& dst, const Tensor& src) {
  if (dst.is_vulkan() && src.is_vulkan()) {
    const vTensor& v_dst = convert(dst);
    const vTensor& v_src = convert(src);
    if (
        v_dst.storage_type() == api::StorageType::BUFFER &&
        v_src.storage_type() == api::StorageType::BUFFER &&
        v_dst.has_direct_buffer_layout() && v_src.has_direct_buffer_layout()) {
      utils::copy_buffer_tensor_direct_(dst, src);
      return;
    }
  }
  dst.copy_(src);
}

Tensor run_attention_runtime_buffer_math_program_impl(
    const Tensor& query_arg,
    const Tensor& key_arg,
    const Tensor& value_arg,
    utils::AttentionRuntimeProgram* const runtime_program,
    Tensor* const output_override = nullptr,
    const float query_scale = 1.0f) {
  Tensor query = ensure_attention_runtime_direct_buffer(query_arg);
  Tensor key = ensure_attention_runtime_direct_buffer(key_arg);
  Tensor value = ensure_attention_runtime_direct_buffer(value_arg);
  TORCH_CHECK(
      query.scalar_type() == kFloat && key.scalar_type() == kFloat &&
          value.scalar_type() == kFloat,
      "Attention-runtime replay currently expects float tensors");
  TORCH_CHECK(
      query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
      "Attention-runtime replay currently expects rank-3 tensors");
  TORCH_CHECK(
      query.size(0) == key.size(0) && query.size(0) == value.size(0) &&
          query.size(2) == key.size(2) && key.size(1) == value.size(1),
      "Attention-runtime replay expects matching [B, T, K] / [B, S, K] / [B, S, V] shapes");

  utils::ScratchArena* scratch_arena = nullptr;
  if (runtime_program && runtime_program->defined() &&
      runtime_program->scratch_arena().has_value()) {
    runtime_program->scratch_arena()->reset();
    scratch_arena = &(*runtime_program->scratch_arena());
  }

  const std::vector<int64_t> scores_sizes{
      query.size(0),
      query.size(1),
      key.size(1),
  };
  const std::vector<int64_t> output_sizes{
      query.size(0),
      query.size(1),
      value.size(2),
  };

  Tensor output = output_override ? *output_override
                                  : utils::create_buffer_tensor(
                                        output_sizes, kFloat, /*persistent=*/false);
  if (can_use_runtime_program_buffer_fused_fast_path(query, key, value)) {
    utils::log_vulkan_op_hit(
        "aten::scaled_dot_product_attention.runtime_program_buffer_fused");
    const auto variant =
        select_runtime_program_buffer_fused_variant(query, key, value);
    utils::log_vulkan_op_hit(
        runtime_program_buffer_fused_variant_log_name(variant));
    return scaled_dot_product_attention_runtime_fused_3d_buffer_out_vulkan(
        query, key, value, output, query_scale);
  }

  Tensor scores_output;
  Tensor probs_output;

  if (scratch_arena) {
    scores_output = reserve_attention_scratch_tensor(
        *scratch_arena, scores_sizes, kFloat);
    probs_output = reserve_attention_scratch_tensor(
        *scratch_arena, scores_sizes, kFloat);
    output = output_override ? *output_override
                             : reserve_attention_scratch_tensor(
                                   *scratch_arena, output_sizes, kFloat);
  } else if (output_override) {
    output = *output_override;
    scores_output = utils::create_buffer_tensor(
        scores_sizes, kFloat, /*persistent=*/false);
    probs_output = utils::create_buffer_tensor(
        scores_sizes, kFloat, /*persistent=*/false);
  } else {
    scores_output = utils::create_buffer_tensor(
        scores_sizes, kFloat, /*persistent=*/false);
    probs_output = utils::create_buffer_tensor(
        scores_sizes, kFloat, /*persistent=*/false);
  }

  utils::log_vulkan_op_hit(
      "aten::scaled_dot_product_attention.runtime_program_buffer_materialized");
  Tensor query_for_scores = query;
  if (query_scale != 1.0f) {
    DeferredAttentionQueryScaleMaterializeGuard guard;
    query_for_scores = at::mul(query, query_scale);
  }
  Tensor key_t = ensure_attention_runtime_direct_buffer(key.transpose(1, 2));
  Tensor scores = bmm_buffer_out_vulkan(query_for_scores, key_t, scores_output);
  Tensor probs = softmax_buffer_lastdim_out_vulkan(scores, probs_output);
  return bmm_buffer_out_vulkan(probs, value, output);
}

Tensor run_attention_runtime_buffer_math_replay_impl(
    const Tensor& query_arg,
    const Tensor& key_arg,
    const Tensor& value_arg,
    const std::string& allocation_label) {
  Tensor query = ensure_attention_runtime_direct_buffer(query_arg);
  Tensor key = ensure_attention_runtime_direct_buffer(key_arg);
  Tensor value = ensure_attention_runtime_direct_buffer(value_arg);
  TORCH_CHECK(
      can_use_attention_runtime_buffer_math_replay(query, key, value),
      "Attention-runtime replay currently expects float buffer-backed tensors");

  const utils::VulkanExecutionProgramPlanningDesc program_plan{
      utils::VulkanExecutionProgramKind::AttentionRuntime,
      true,
  };
  const std::optional<utils::VulkanScratchArenaSpec> scratch_spec =
      can_use_runtime_program_buffer_fused_fast_path(query, key, value)
      ? std::nullopt
      : std::optional<utils::VulkanScratchArenaSpec>(utils::VulkanScratchArenaSpec{
            kByte,
            std::max<size_t>(
                attention_runtime_scratch_bytes(query, key, value),
                1u),
            256u,
            api::ExecutionLayout::BUFFER_DIRECT,
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
            api::StorageType::BUFFER,
            true,
        });

  auto attention_graph =
      utils::lookup_or_create_labeled_attention_runtime_inference_graph(
          allocation_label, kFloat, /*persistent=*/true);
  auto attention_replay = attention_graph.lookup_or_create_replay(
      allocation_label + ".buffer_math",
      query.sizes(),
      key.sizes(),
      value.sizes(),
      utils::VulkanAttentionKernelFamily::BufferMath,
      std::nullopt,
      std::nullopt,
      scratch_spec,
      key.size(1),
      value.size(1),
      program_plan);
  copy_tensor_for_attention_replay(attention_replay.query_slot(), query);
  copy_tensor_for_attention_replay(attention_replay.key_slot(), key);
  copy_tensor_for_attention_replay(attention_replay.value_slot(), value);
  api::context()->flush_pending_cmds();

  if (!attention_replay.recorded()) {
    Tensor warmup_output = utils::create_buffer_tensor(
        attention_replay.output_slot().sizes(),
        attention_replay.output_slot().scalar_type(),
        /*persistent=*/false);
    (void)run_attention_runtime_buffer_math_program_impl(
        attention_replay.query_slot(),
        attention_replay.key_slot(),
        attention_replay.value_slot(),
        &attention_replay.program(),
        &attention_replay.output_slot());
    copy_tensor_for_attention_replay(
        warmup_output, attention_replay.output_slot());
    api::context()->flush_pending_cmds();
    attention_replay.replay().record([&]() {
      (void)run_attention_runtime_buffer_math_program_impl(
          attention_replay.query_slot(),
          attention_replay.key_slot(),
          attention_replay.value_slot(),
          &attention_replay.program(),
          &attention_replay.output_slot());
    });
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_attention_runtime_buffer_math_replay_bridge.warmup");
    return warmup_output;
  }

  attention_replay.replay().submit();
  Tensor output = utils::create_buffer_tensor(
      attention_replay.output_slot().sizes(),
      attention_replay.output_slot().scalar_type(),
      /*persistent=*/false);
  copy_tensor_for_attention_replay(output, attention_replay.output_slot());
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_attention_runtime_buffer_math_replay_bridge.replay");
  return output;
}

const std::string& sdpa_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_SDPA_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool sdpa_logging_enabled() {
  return !sdpa_log_path().empty();
}

const char* sdpa_storage_type_name(const api::StorageType storage_type) {
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

const char* sdpa_memory_layout_name(const api::GPUMemoryLayout memory_layout) {
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

std::string format_sdpa_sizes(IntArrayRef sizes) {
  std::ostringstream stream;
  stream << "[";
  for (const auto idx : c10::irange(sizes.size())) {
    if (idx > 0) {
      stream << ",";
    }
    stream << sizes[idx];
  }
  stream << "]";
  return stream.str();
}

void append_sdpa_log_line(const std::string& line) {
  if (!sdpa_logging_enabled()) {
    return;
  }

  std::ofstream out(sdpa_log_path(), std::ios::app);
  out << line << '\n';
}

void append_sdpa_tensor_log_details(
    std::ostringstream& stream,
    const char* prefix,
    const Tensor& tensor) {
  stream << " " << prefix << "_sizes=" << format_sdpa_sizes(tensor.sizes())
         << " " << prefix << "_dtype=" << tensor.scalar_type();
  if (!tensor.is_vulkan()) {
    stream << " " << prefix << "_device=" << tensor.device();
    return;
  }

  const vTensor& v_tensor = convert(tensor);
  stream << " " << prefix << "_exec="
         << utils::execution_layout_name(v_tensor.execution_layout())
         << " " << prefix << "_storage="
         << sdpa_storage_type_name(v_tensor.storage_type())
         << " " << prefix << "_layout="
         << sdpa_memory_layout_name(v_tensor.gpu_memory_layout())
         << " " << prefix << "_direct_buffer="
         << (v_tensor.has_direct_buffer_layout() ? 1 : 0);
}

void log_sdpa_event(
    const char* event,
    const char* result,
    const char* reason,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double>& scale,
    const bool enable_gqa) {
  if (!sdpa_logging_enabled()) {
    return;
  }

  std::ostringstream stream;
  stream << "sdpa event=" << event << " result=" << result
         << " reason=" << reason
         << " caller=" << api::current_allocation_label()
         << " dropout_p=" << dropout_p
         << " is_causal=" << (is_causal ? 1 : 0)
         << " enable_gqa=" << (enable_gqa ? 1 : 0)
         << " has_mask="
         << ((attn_mask && attn_mask->defined()) ? 1 : 0)
         << " scale=";
  if (scale.has_value()) {
    stream << *scale;
  } else {
    stream << "none";
  }

  append_sdpa_tensor_log_details(stream, "query", query);
  append_sdpa_tensor_log_details(stream, "key", key);
  append_sdpa_tensor_log_details(stream, "value", value);
  append_sdpa_log_line(stream.str());
}

Tensor finalize_public_sdpa_output(Tensor output) {
  if (!c10::InferenceMode::is_enabled()) {
    utils::log_vulkan_op_hit(
      "aten::scaled_dot_product_attention.non_inference_sync");
    api::context()->synchronize_device();
  }
  return output;
}

Tensor maybe_scale_query(const Tensor& query, const double query_scale) {
  if (query_scale == 1.0) {
    return query;
  }
  return query.mul(query_scale);
}

bool can_run_buffer_softmax(const Tensor& input, const int64_t dim) {
  if (
      !input.is_vulkan() ||
      input.scalar_type() != kFloat ||
      input.dim() < 1 ||
      input.dim() > 4 ||
      dim < 0 ||
      dim >= input.dim()) {
    return false;
  }
  if (dim != input.dim() - 1) {
    return input.numel() > 0;
  }
  if (
      input.dim() == 3 && dim == input.dim() - 1 && input.size(dim) >= 64 &&
      !utils::matches_sdpa_buffer_softmax_score_contract(
          input.sizes(), input.scalar_type(), dim)) {
    return false;
  }

  const vTensor& v_input = convert(input);
  return v_input.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_reduction_compute(v_input);
}

Tensor ensure_softmax_buffer_output_tensor(
    Tensor& output,
    IntArrayRef sizes,
    const c10::ScalarType dtype) {
  bool needs_allocation = !output.defined() || !output.is_vulkan() ||
      output.scalar_type() != dtype || !output.sizes().equals(sizes);
  if (!needs_allocation) {
    const vTensor& v_output = convert(output);
    needs_allocation =
        v_output.storage_type() != api::StorageType::BUFFER ||
        v_output.gpu_memory_layout() !=
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED ||
        !utils::supports_buffer_view_fast_path(v_output);
  }
  if (needs_allocation) {
    output = utils::mark_tensor_execution(
        convert(vTensor{
            api::context(),
            sizes.vec(),
            convert_dtype(dtype),
            api::StorageType::BUFFER,
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
        }),
        api::ExecutionLayout::BUFFER_DIRECT);
  } else {
    output = utils::mark_tensor_execution(
        output,
        utils::resolve_buffer_execution_layout(convert(output)));
  }
  return output;
}

bool can_use_vision_score_softmax_padded_buffer_input(
    const Tensor& input,
    const int64_t dim) {
  const utils::SDPAScoreSoftmaxMatch score_contract =
      utils::match_sdpa_buffer_softmax_score_contract(
          input.sizes(), input.scalar_type(), dim);
  if (
      !score_contract.matched ||
      score_contract.family !=
          utils::SDPAScoreSoftmaxFamily::VisionSelfAttentionScores) {
    return false;
  }

  const vTensor& v_input = convert(input);
  return v_input.storage_type() == api::StorageType::BUFFER &&
      v_input.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
      utils::supports_buffer_reduction_compute(v_input);
}

Tensor prepare_softmax_buffer_lastdim_input(const Tensor& input) {
  const int64_t dim = input.dim() - 1;
  if (can_use_vision_score_softmax_padded_buffer_input(input, dim)) {
    utils::log_vulkan_op_hit(
        "aten::_softmax.buffer_lastdim_vision_score_padded_input");
    return utils::mark_tensor_execution(
        input, utils::resolve_buffer_execution_layout(convert(input)));
  }

  const auto plan = utils::build_vulkan_execution_plan(
      input, utils::VulkanExecutionPlanKind::ReductionDimInput);
  return utils::prepare_vulkan_direct_buffer_execution_tensor(input, plan);
}

Tensor softmax_buffer_lastdim_impl(const Tensor& input, Tensor* output_opt) {
  api::AllocationScope allocation_scope("softmax.buffer_lastdim");
  utils::log_vulkan_op_hit("aten::_softmax.buffer_lastdim");
  utils::validate_replay_tensor_not_stale(
      input, "aten::_softmax.buffer_lastdim");

  Tensor resolved_input = prepare_softmax_buffer_lastdim_input(input);

  api::Context* const context = api::context();
  vTensor& v_input = convert(resolved_input);
  const uint32_t reduce_size =
      safe_downcast<uint32_t>(
          std::max<int64_t>(resolved_input.size(resolved_input.dim() - 1), 1));
  const uint32_t row_count =
      safe_downcast<uint32_t>(v_input.numel() / reduce_size);

  Tensor output_tensor = output_opt
      ? ensure_softmax_buffer_output_tensor(
            *output_opt, resolved_input.sizes(), convert_dtype(v_input.dtype()))
      : utils::mark_tensor_execution(
            convert(vTensor{
                context,
                v_input.sizes(),
                v_input.dtype(),
                api::StorageType::BUFFER,
                api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
            }),
            api::ExecutionLayout::BUFFER_DIRECT);
  vTensor& v_output = convert(output_tensor);

  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);

  api::PipelineBarrier pipeline_barrier{};
  const uint32_t rows_per_grid_x =
      std::min(row_count, kBufferSoftmaxLastDimMaxWorkGroupsX);
  const uint32_t grid_y =
      api::utils::div_up(row_count, rows_per_grid_x);
  const struct {
    uint32_t row_count;
    uint32_t rows_per_grid_x;
    uint32_t reduce_size;
    uint32_t reserved;
  } block{
      row_count,
      rows_per_grid_x,
      reduce_size,
      0u,
  };
  api::UniformParamsBuffer params(context, block);
  const api::utils::uvec3 global_size{
      safe_downcast<uint32_t>(
          static_cast<uint64_t>(rows_per_grid_x) *
          kBufferSoftmaxLastDimLocalSizeX),
      grid_y,
      1u};
  context->submit_compute_job(
      VK_KERNEL(buffer_softmax_lastdim_float),
      pipeline_barrier,
      global_size,
      {kBufferSoftmaxLastDimLocalSizeX, 1u, 1u},
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer(),
      params.buffer());

  return record_tensor_write_and_return(
      output_tensor,
      "aten::_softmax",
      "buffer_lastdim",
      {resolved_input});
}

Tensor softmax_buffer_lastdim(const Tensor& input) {
  return softmax_buffer_lastdim_impl(input, nullptr);
}

Tensor softmax_buffer_dim_impl(const Tensor& input_arg, const int64_t dim) {
  api::AllocationScope allocation_scope("softmax.buffer_dim");
  utils::log_vulkan_op_hit("aten::_softmax.buffer_dim");
  utils::validate_replay_tensor_not_stale(
      input_arg, "aten::_softmax.buffer_dim");

  const auto plan = utils::build_vulkan_execution_plan(
      input_arg, utils::VulkanExecutionPlanKind::ReductionDimInput);
  Tensor input =
      utils::prepare_vulkan_direct_buffer_execution_tensor(input_arg, plan);

  api::Context* const context = api::context();
  context->submit_pending_work_and_poll_retire();
  vTensor& v_input = convert(input);
  Tensor output = utils::mark_tensor_execution(
      convert(vTensor{
          context,
          v_input.sizes(),
          v_input.dtype(),
          api::StorageType::BUFFER,
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      }),
      api::ExecutionLayout::BUFFER_DIRECT);
  vTensor& v_output = convert(output);

  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);

  const uint32_t out_numel =
      safe_downcast<uint32_t>(std::max<int64_t>(input.numel(), 0));
  const uint32_t reduce_axis =
      safe_downcast<uint32_t>(input.dim() - 1 - dim);
  const uint32_t reduce_size =
      safe_downcast<uint32_t>(std::max<int64_t>(input.size(dim), 1));
  const struct {
    uint32_t out_numel;
    uint32_t reduce_axis;
    uint32_t reduce_size;
    uint32_t reserved;
  } block{
      out_numel,
      reduce_axis,
      reduce_size,
      0u,
  };
  api::UniformParamsBuffer params(context, block);

  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::div_up(out_numel, kBufferSoftmaxDimLocalSizeX) *
          kBufferSoftmaxDimLocalSizeX,
      1u,
      1u};
  context->submit_compute_job(
      VK_KERNEL(buffer_softmax_dim_float),
      pipeline_barrier,
      global_size,
      {kBufferSoftmaxDimLocalSizeX, 1u, 1u},
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer(),
      params.buffer());

  return record_tensor_write_and_return(
      output,
      "aten::_softmax",
      "buffer_dim",
      {input});
}

Tensor softmax_buffer(
    const Tensor& input_arg,
    const int64_t dim) {
  if (dim == input_arg.dim() - 1 && input_arg.size(dim) > 0 &&
      input_arg.numel() > 0) {
    return softmax_buffer_lastdim(input_arg);
  }
  return softmax_buffer_dim_impl(input_arg, dim);
}

std::tuple<Tensor, Tensor, Tensor> transform_bias_rescale_qkv_vulkan_out_impl(
    const Tensor& qkv_arg,
    const Tensor& qkv_bias_arg,
    const int64_t num_head,
    const Tensor& q_out_arg,
    const Tensor& k_out_arg,
    const Tensor& v_out_arg) {
  TORCH_CHECK(
      qkv_arg.is_vulkan() && q_out_arg.is_vulkan() && k_out_arg.is_vulkan() &&
          v_out_arg.is_vulkan(),
      "Vulkan _transform_bias_rescale_qkv_out expects Vulkan tensors");
  TORCH_CHECK(
      qkv_arg.dim() == 2 && qkv_bias_arg.dim() == 1,
      "Vulkan _transform_bias_rescale_qkv_out expects a 2D qkv tensor and 1D bias tensor");
  TORCH_CHECK(
      qkv_arg.scalar_type() == kFloat && qkv_bias_arg.scalar_type() == kFloat,
      "Vulkan _transform_bias_rescale_qkv_out currently supports float tensors");

  const int64_t token_count = qkv_arg.size(0);
  const int64_t embed_dim = qkv_arg.size(1) / 3;
  TORCH_CHECK(
      qkv_arg.size(1) == qkv_bias_arg.size(0) && qkv_arg.size(1) % 3 == 0,
      "Vulkan _transform_bias_rescale_qkv_out expects matching qkv and bias widths");
  TORCH_CHECK(
      embed_dim % num_head == 0,
      "Vulkan _transform_bias_rescale_qkv_out expects embed_dim divisible by num_heads");
  const int64_t head_dim = embed_dim / num_head;
  const float q_scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));

  const auto prepare_buffer_output =
      [](const Tensor& output_arg) -> Tensor {
    Tensor output = output_arg.is_vulkan() ? output_arg : output_arg.vulkan();
    const vTensor& v_output = convert(output);
    if (
        v_output.storage_type() == api::StorageType::BUFFER &&
        v_output.gpu_memory_layout() ==
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
        utils::supports_buffer_view_fast_path(v_output)) {
      return utils::mark_tensor_execution(
          output, utils::resolve_buffer_execution_layout(v_output));
    }
    return utils::mark_tensor_execution(
        utils::ensure_buffer_storage(
            output, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
        api::ExecutionLayout::BUFFER_DIRECT);
  };

  Tensor qkv_buffer = utils::mark_tensor_execution(
      utils::ensure_buffer_storage(
          qkv_arg, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
      api::ExecutionLayout::BUFFER_DIRECT);
  Tensor bias_buffer = utils::mark_tensor_execution(
      utils::ensure_buffer_storage(
          qkv_bias_arg, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
      api::ExecutionLayout::BUFFER_DIRECT);
  Tensor q = prepare_buffer_output(q_out_arg);
  Tensor k = prepare_buffer_output(k_out_arg);
  Tensor v = prepare_buffer_output(v_out_arg);

  vTensor& v_qkv_buffer = convert(qkv_buffer);
  vTensor& v_bias_buffer = convert(bias_buffer);
  vTensor& v_q = convert(q);
  vTensor& v_k = convert(k);
  vTensor& v_v = convert(v);
  TORCH_CHECK(
      v_qkv_buffer.storage_type() == api::StorageType::BUFFER &&
          v_qkv_buffer.has_direct_buffer_layout(),
      "Vulkan buffer _transform_bias_rescale_qkv_out expects direct-buffer qkv");
  TORCH_CHECK(
      v_bias_buffer.storage_type() == api::StorageType::BUFFER &&
          v_bias_buffer.has_direct_buffer_layout(),
      "Vulkan buffer _transform_bias_rescale_qkv_out expects direct-buffer bias");
  TORCH_CHECK(
      v_q.storage_type() == api::StorageType::BUFFER &&
          utils::supports_buffer_view_fast_path(v_q),
      "Vulkan buffer _transform_bias_rescale_qkv_out expects buffer-backed q output");
  TORCH_CHECK(
      v_k.storage_type() == api::StorageType::BUFFER &&
          utils::supports_buffer_view_fast_path(v_k),
      "Vulkan buffer _transform_bias_rescale_qkv_out expects buffer-backed k output");
  TORCH_CHECK(
      v_v.storage_type() == api::StorageType::BUFFER &&
          utils::supports_buffer_view_fast_path(v_v),
      "Vulkan buffer _transform_bias_rescale_qkv_out expects buffer-backed v output");
  TORCH_CHECK(
      q.sizes().equals({num_head, token_count, head_dim}) &&
          k.sizes().equals({num_head, token_count, head_dim}) &&
          v.sizes().equals({num_head, token_count, head_dim}),
      "Vulkan _transform_bias_rescale_qkv_out received unexpected q/k/v output sizes");

  api::Context* const context = api::context();
  const struct Block final {
    ivec4 sizes;
    vec4 scale;
  } block{
      {
          safe_downcast<int32_t>(head_dim),
          safe_downcast<int32_t>(token_count),
          safe_downcast<int32_t>(num_head),
          safe_downcast<int32_t>(embed_dim),
      },
      {q_scale, 0.0f, 0.0f, 0.0f},
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      safe_downcast<uint32_t>(head_dim),
      safe_downcast<uint32_t>(token_count),
      safe_downcast<uint32_t>(num_head),
  };

  context->submit_compute_job(
      VK_KERNEL(transform_bias_rescale_qkv_buffer),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_q.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_q.buffer_metadata(),
      v_k.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_k.buffer_metadata(),
      v_v.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_v.buffer_metadata(),
      v_qkv_buffer.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_qkv_buffer.buffer_metadata(),
      v_bias_buffer.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_bias_buffer.buffer_metadata(),
      params.buffer());

  utils::log_vulkan_op_hit("aten::_transform_bias_rescale_qkv.buffer_native");
  return std::make_tuple(std::move(q), std::move(k), std::move(v));
}

std::tuple<Tensor, Tensor, Tensor> transform_bias_rescale_qkv_vulkan(
    const Tensor& qkv_arg,
    const Tensor& qkv_bias_arg,
    const int64_t num_head) {
  api::AllocationScope allocation_scope("transform_bias_rescale_qkv");
  utils::log_vulkan_op_hit("aten::_transform_bias_rescale_qkv");

  TORCH_CHECK(
      qkv_arg.is_vulkan(),
      "Vulkan _transform_bias_rescale_qkv expects qkv on Vulkan");
  TORCH_CHECK(
      qkv_arg.dim() == 2,
      "Vulkan _transform_bias_rescale_qkv currently expects a 2D [T, 3D] tensor");
  TORCH_CHECK(
      qkv_bias_arg.dim() == 1,
      "Vulkan _transform_bias_rescale_qkv currently expects a 1D bias tensor");
  TORCH_CHECK(
      qkv_arg.scalar_type() == qkv_bias_arg.scalar_type(),
      "Vulkan _transform_bias_rescale_qkv expects qkv and bias to share a dtype");
  TORCH_CHECK(
      qkv_arg.scalar_type() == kFloat || qkv_arg.scalar_type() == kHalf,
      "Vulkan _transform_bias_rescale_qkv currently supports float and half");
  TORCH_CHECK(
      qkv_arg.size(1) == qkv_bias_arg.size(0),
      "Vulkan _transform_bias_rescale_qkv expects matching qkv and bias widths");
  TORCH_CHECK(
      qkv_arg.size(1) % 3 == 0,
      "Vulkan _transform_bias_rescale_qkv expects the last qkv dim to be divisible by 3");

  const int64_t token_count = qkv_arg.size(0);
  const int64_t embed_dim = qkv_arg.size(1) / 3;
  TORCH_CHECK(
      embed_dim % num_head == 0,
      "Vulkan _transform_bias_rescale_qkv expects embed_dim divisible by num_heads");
  const int64_t head_dim = embed_dim / num_head;
  const float q_scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));

  const auto buffer_qkv_transform = [&]() -> std::tuple<Tensor, Tensor, Tensor> {
    api::Context* const context = api::context();
    Tensor q = utils::mark_tensor_execution(
        convert(vTensor{
            context,
            {num_head, token_count, head_dim},
            api::kFloat,
            api::StorageType::BUFFER,
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
        }),
        api::ExecutionLayout::BUFFER_DIRECT);
    Tensor k = utils::mark_tensor_execution(
        convert(vTensor{
            context,
            {num_head, token_count, head_dim},
            api::kFloat,
            api::StorageType::BUFFER,
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
        }),
        api::ExecutionLayout::BUFFER_DIRECT);
    Tensor v = utils::mark_tensor_execution(
        convert(vTensor{
            context,
            {num_head, token_count, head_dim},
            api::kFloat,
            api::StorageType::BUFFER,
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
        }),
        api::ExecutionLayout::BUFFER_DIRECT);
    return transform_bias_rescale_qkv_vulkan_out_impl(
        qkv_arg, qkv_bias_arg, num_head, q, k, v);
  };

  const auto generic_qkv_transform = [&]() -> std::tuple<Tensor, Tensor, Tensor> {
    Tensor qkv = qkv_arg;
    std::vector<Tensor> qkv_chunks = at::chunk(qkv, 3, 1);
    Tensor q = qkv_chunks[0].add(qkv_bias_arg.slice(0, 0, embed_dim));
    Tensor k =
        qkv_chunks[1].add(qkv_bias_arg.slice(0, embed_dim, 2 * embed_dim));
    Tensor v =
        qkv_chunks[2].add(qkv_bias_arg.slice(0, 2 * embed_dim, 3 * embed_dim));
    q = q.reshape({token_count, num_head, head_dim})
            .permute({1, 0, 2})
            .mul(q_scale);
    k = k.reshape({token_count, num_head, head_dim}).permute({1, 0, 2});
    v = v.reshape({token_count, num_head, head_dim}).permute({1, 0, 2});
    return std::make_tuple(std::move(q), std::move(k), std::move(v));
  };

  const vTensor& v_qkv_arg = convert(qkv_arg);
  const vTensor& v_qkv_bias_arg = convert(qkv_bias_arg);
  const bool qkv_arg_is_texture =
      v_qkv_arg.storage_type() == api::StorageType::TEXTURE_3D &&
      v_qkv_arg.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;
  const bool bias_arg_is_texture =
      v_qkv_bias_arg.storage_type() == api::StorageType::TEXTURE_3D &&
      v_qkv_bias_arg.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;

  if (!qkv_arg_is_texture || !bias_arg_is_texture) {
    return buffer_qkv_transform();
  }

  const Tensor qkv = utils::prepare_vulkan_execution_tensor(
      qkv_arg, utils::VulkanExecutionPlanKind::TextureComputeInput);
  const Tensor qkv_bias = utils::prepare_vulkan_execution_tensor(
      qkv_bias_arg, utils::VulkanExecutionPlanKind::TextureComputeInput);
  const vTensor& v_qkv = convert(qkv);
  const vTensor& v_qkv_bias = convert(qkv_bias);

  const bool qkv_is_texture =
      v_qkv.storage_type() == api::StorageType::TEXTURE_3D &&
      v_qkv.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;
  const bool bias_is_texture =
      v_qkv_bias.storage_type() == api::StorageType::TEXTURE_3D &&
      v_qkv_bias.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;
  TORCH_INTERNAL_ASSERT(
      qkv_is_texture && bias_is_texture,
      "Expected texture-backed qkv and bias after texture preparation");

  api::Context* const context = api::context();
  vTensor v_q{
      context,
      {num_head, token_count, head_dim},
      convert_dtype(qkv_arg.scalar_type()),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
  };
  vTensor v_k{
      context,
      {num_head, token_count, head_dim},
      convert_dtype(qkv_arg.scalar_type()),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
  };
  vTensor v_v{
      context,
      {num_head, token_count, head_dim},
      convert_dtype(qkv_arg.scalar_type()),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
  };

  const struct Block final {
    ivec4 sizes;
    vec4 scale;
  } block{
      {
          safe_downcast<int32_t>(head_dim),
          safe_downcast<int32_t>(token_count),
          safe_downcast<int32_t>(num_head),
          safe_downcast<int32_t>(embed_dim),
      },
      {q_scale, 0.0f, 0.0f, 0.0f},
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(transform_bias_rescale_qkv),
      pipeline_barrier,
      v_q.extents(),
      adaptive_work_group_size(v_q.extents()),
      VK_NULL_HANDLE,
      v_q.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_k.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_v.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_qkv.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_qkv_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return std::make_tuple(convert(v_q), convert(v_k), convert(v_v));
}

Tensor flatten_attention_batch_heads(
    const Tensor& tensor,
    const int64_t batch_heads,
    const int64_t sequence_length,
    const int64_t feature_size) {
  if (tensor.dim() == 3) {
    return tensor;
  }
  return tensor.reshape({batch_heads, sequence_length, feature_size});
}

utils::GQARepeatMatch match_gqa_repeat_materialization_contract(
    const Tensor& tensor,
    const int64_t repeat_factor) {
  const bool has_buffer_storage =
      tensor.is_vulkan() &&
      convert(tensor).storage_type() == api::StorageType::BUFFER;
  return utils::match_gqa_repeat_contract(
      tensor.sizes(),
      tensor.scalar_type(),
      tensor.is_vulkan(),
      has_buffer_storage,
      repeat_factor);
}

Tensor materialize_bounded_decode_gqa_repeat(
    const Tensor& tensor,
    const int64_t repeat_factor) {
  api::AllocationScope allocation_scope("sdpa.gqa_repeat");
  TORCH_INTERNAL_ASSERT(
      match_gqa_repeat_materialization_contract(tensor, repeat_factor).matched);

  const int64_t batch = tensor.size(0);
  const int64_t heads = tensor.size(1);
  const int64_t sequence_length = tensor.size(2);
  const int64_t head_dim = tensor.size(3);
  Tensor output = utils::create_buffer_tensor(
      {batch, heads * repeat_factor, sequence_length, head_dim},
      tensor.scalar_type(),
      /*persistent=*/false);
  output = utils::mark_tensor_execution(
      output,
      utils::resolve_buffer_execution_layout(convert(output)),
      false);

  const vTensor& v_input = convert(tensor);
  vTensor& v_output = convert(output);
  api::Context* const context = api::context();
  const struct Block final {
    ivec4 sizes;
    ivec4 repeat_info;
  } block{
      {
          safe_downcast<int32_t>(batch),
          safe_downcast<int32_t>(heads),
          safe_downcast<int32_t>(sequence_length),
          safe_downcast<int32_t>(head_dim),
      },
      {
          safe_downcast<int32_t>(repeat_factor),
          0,
          0,
          0,
      },
  };

  api::UniformParamsBuffer params(context, block);
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer input_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);
  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size{
      safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
      1u,
      1u,
  };

  utils::log_vulkan_op_hit(
      "aten::scaled_dot_product_attention.bounded_gqa_repeat_materialize");
  context->submit_compute_job(
      VK_KERNEL(gqa_repeat_buffer_float),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      input_meta.buffer(),
      params.buffer());

  output = utils::mark_tensor_execution(
      output, api::ExecutionLayout::BUFFER_DIRECT);
  return record_tensor_write_and_return(
      output,
      "aten::scaled_dot_product_attention",
      "gqa_repeat_materialize",
      {tensor});
}

Tensor repeat_attention_heads_for_gqa(
    const Tensor& tensor,
    const int64_t repeat_factor) {
  if (repeat_factor == 1) {
    return tensor;
  }

  TORCH_CHECK(
      tensor.dim() == 4,
      "Vulkan SDPA GQA expects 4D [B, H, T, D] key/value tensors");
  const int64_t batch = tensor.size(0);
  const int64_t heads = tensor.size(1);
  const int64_t sequence_length = tensor.size(2);
  const int64_t head_dim = tensor.size(3);

  if (match_gqa_repeat_materialization_contract(tensor, repeat_factor).matched) {
    return materialize_bounded_decode_gqa_repeat(tensor, repeat_factor);
  }

  return tensor.unsqueeze(2)
      .expand({batch, heads, repeat_factor, sequence_length, head_dim})
      .reshape({batch, heads * repeat_factor, sequence_length, head_dim});
}

bool can_use_direct_gqa_sdpa_buffer_path(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    const bool is_causal) {
  if (
      attn_mask.has_value() || query.dim() != 4 || key.dim() != 4 ||
      value.dim() != 4 || query.scalar_type() != kFloat ||
      key.scalar_type() != kFloat || value.scalar_type() != kFloat ||
      query.size(0) != 1 || key.size(0) != 1 || value.size(0) != 1 ||
      key.size(1) != value.size(1) || key.size(1) <= 0 ||
      query.size(1) % key.size(1) != 0 || query.size(3) != key.size(3) ||
      key.size(2) != value.size(2)) {
    return false;
  }
  if (is_causal && query.size(2) != key.size(2)) {
    return false;
  }
  if (!is_causal) {
    return false;
  }
  if (
      query.size(3) > kTiledSdpaBufferMaxHeadDim ||
      value.size(3) > kTiledSdpaMaxValueDim) {
    return false;
  }
  return query.is_vulkan() && key.is_vulkan() && value.is_vulkan() &&
      convert(query).storage_type() == api::StorageType::BUFFER &&
      convert(key).storage_type() == api::StorageType::BUFFER &&
      convert(value).storage_type() == api::StorageType::BUFFER;
}

Tensor scaled_dot_product_attention_direct_gqa_4d_buffer_vulkan(
    const Tensor& query_arg,
    const Tensor& key_arg,
    const Tensor& value_arg,
    const double query_scale,
    const bool is_causal) {
  api::AllocationScope allocation_scope("sdpa.direct_gqa_buffer");
  Tensor query = prepare_buffer_math_input_direct(query_arg);
  Tensor key = prepare_buffer_math_input_direct(key_arg);
  Tensor value = prepare_buffer_math_input_direct(value_arg);

  const int64_t batch = query.size(0);
  const int64_t query_heads = query.size(1);
  const int64_t key_value_heads = key.size(1);
  const int64_t target_len = query.size(2);
  const int64_t source_len = key.size(2);
  const int64_t head_dim = query.size(3);
  const int64_t value_dim = value.size(3);
  const int64_t repeat_factor = query_heads / key_value_heads;

  Tensor output = utils::create_buffer_tensor(
      {batch * query_heads, target_len, value_dim},
      query.scalar_type(),
      /*persistent=*/false);
  output = utils::mark_tensor_execution(
      output,
      utils::resolve_buffer_execution_layout(convert(output)),
      false);

  const vTensor& v_query = convert(query);
  const vTensor& v_key = convert(key);
  const vTensor& v_value = convert(value);
  vTensor& v_output = convert(output);

  api::Context* const context = api::context();
  const struct Block final {
    ivec4 sizes;
    ivec4 tiled_info;
    vec4 params;
  } block{
      {
          safe_downcast<int32_t>(batch),
          safe_downcast<int32_t>(query_heads),
          safe_downcast<int32_t>(target_len),
          safe_downcast<int32_t>(source_len),
      },
      {
          safe_downcast<int32_t>(head_dim),
          safe_downcast<int32_t>(value_dim),
          safe_downcast<int32_t>(repeat_factor),
          is_causal ? 1 : 0,
      },
      {static_cast<float>(query_scale), 0.0f, 0.0f, 0.0f},
  };

  api::UniformParamsBuffer params(context, block);
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer query_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_query);
  api::UniformParamsBuffer key_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_key);
  api::UniformParamsBuffer value_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_value);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(scaled_dot_product_scores_value_gqa_buffer_float),
      pipeline_barrier,
      {
          static_cast<uint32_t>(kTiledSdpaLocalSizeX),
          safe_downcast<uint32_t>(target_len),
          safe_downcast<uint32_t>(batch * query_heads),
      },
      {
          static_cast<uint32_t>(kTiledSdpaLocalSizeX),
          1u,
          1u,
      },
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_query.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      query_meta.buffer(),
      v_key.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      key_meta.buffer(),
      v_value.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      value_meta.buffer(),
      params.buffer());

  utils::log_vulkan_op_hit(
      "aten::scaled_dot_product_attention.direct_gqa_buffer");
  output = utils::mark_tensor_execution(
      output, api::ExecutionLayout::BUFFER_DIRECT);
  return record_tensor_write_and_return(
      output,
      "aten::scaled_dot_product_attention",
      "direct_gqa_buffer",
      {query, key, value});
}

Tensor expand_attention_mask_3d(
    const Tensor& attn_mask,
    const int64_t batch,
    const int64_t heads,
    const int64_t target_len,
    const int64_t source_len) {
  TORCH_CHECK(
      attn_mask.dim() >= 2 && attn_mask.dim() <= 4,
      "Vulkan SDPA expects 2D, 3D, or 4D attention masks");

  if (attn_mask.dim() == 2) {
    TORCH_CHECK(
        attn_mask.size(0) == target_len && attn_mask.size(1) == source_len,
        "Vulkan SDPA 2D attention mask must match [T, S]");
    return attn_mask.unsqueeze(0).expand({batch * heads, target_len, source_len});
  }

  if (attn_mask.dim() == 3) {
    TORCH_CHECK(
        attn_mask.size(1) == target_len && attn_mask.size(2) == source_len,
        "Vulkan SDPA 3D attention mask must match [N, T, S]");
    if (attn_mask.size(0) == batch * heads) {
      return attn_mask;
    }
    TORCH_CHECK(
        attn_mask.size(0) == batch || attn_mask.size(0) == 1,
        "Vulkan SDPA 3D attention mask batch dimension must be 1, batch, or batch*heads");
    return attn_mask.unsqueeze(1)
        .expand({attn_mask.size(0), heads, target_len, source_len})
        .reshape({attn_mask.size(0) * heads, target_len, source_len})
        .expand({batch * heads, target_len, source_len});
  }

  TORCH_CHECK(
      attn_mask.size(2) == target_len && attn_mask.size(3) == source_len,
      "Vulkan SDPA 4D attention mask must match [B, H, T, S]");
  TORCH_CHECK(
      (attn_mask.size(0) == batch || attn_mask.size(0) == 1) &&
          (attn_mask.size(1) == heads || attn_mask.size(1) == 1),
      "Vulkan SDPA 4D attention mask batch/head dimensions must be 1 or match the input");
  if (attn_mask.size(0) == 1 && attn_mask.size(1) == 1) {
    return attn_mask.reshape({1, target_len, source_len});
  }
  return attn_mask.expand({batch, heads, target_len, source_len})
      .reshape({batch * heads, target_len, source_len});
}

Tensor make_attention_mask_additive(
    const Tensor& attn_mask,
    const Tensor& query,
    const int64_t batch,
    const int64_t heads,
    const int64_t target_len,
    const int64_t source_len) {
  if (attn_mask.scalar_type() == kBool) {
    report_vulkan_cpu_fallback(
        "aten::scaled_dot_product_attention",
        "bool_attention_mask_cpu_materialization",
        {attn_mask, query});
    Tensor mask_cpu = expand_attention_mask_3d(
                           attn_mask.is_vulkan() ? attn_mask.cpu() : attn_mask,
                           batch,
                           heads,
                           target_len,
                           source_len)
                          .to(kBool);
    Tensor additive_mask = at::zeros(
        mask_cpu.sizes(), query.options().device(at::kCPU).dtype(kFloat));
    additive_mask.masked_fill_(mask_cpu.logical_not(), -std::numeric_limits<float>::infinity());
    return additive_mask.to(query.scalar_type());
  }
  Tensor mask = expand_attention_mask_3d(
      attn_mask, batch, heads, target_len, source_len);
  return mask.to(query.scalar_type());
}

Tensor make_causal_attention_bias(
    const Tensor& query,
    const int64_t batch_heads,
    const int64_t target_len,
    const int64_t source_len) {
  report_vulkan_cpu_fallback(
      "aten::scaled_dot_product_attention",
      "causal_attention_bias_cpu_materialization",
      {query});
  Tensor causal_mask = at::ones(
      {target_len, source_len},
      query.options().device(at::kCPU).dtype(kBool));
  causal_mask = at::triu(causal_mask, 1);

  Tensor causal_bias = at::zeros(
      {target_len, source_len},
      query.options().device(at::kCPU).dtype(kFloat));
  causal_bias.masked_fill_(causal_mask, -std::numeric_limits<float>::infinity());
  return causal_bias.to(query.scalar_type())
      .unsqueeze(0)
      .expand({batch_heads, target_len, source_len});
}

Tensor prepare_attention_bias(
    const std::optional<Tensor>& attn_mask,
    const utils::VulkanAttentionPolicy& attention_policy,
    const Tensor& query,
    const int64_t batch,
    const int64_t heads,
    const int64_t target_len,
    const int64_t source_len) {
  const int64_t batch_heads = batch * heads;
  Tensor additive_bias;
  if (attn_mask && attn_mask->defined()) {
    additive_bias = make_attention_mask_additive(
        *attn_mask, query, batch, heads, target_len, source_len);
  }

  if (attention_policy.is_causal) {
    Tensor causal_bias =
        make_causal_attention_bias(query, batch_heads, target_len, source_len);
    if (additive_bias.defined()) {
      if (!additive_bias.is_vulkan()) {
        additive_bias = additive_bias.vulkan();
      }
      if (!causal_bias.is_vulkan()) {
        causal_bias = causal_bias.vulkan();
      }
      additive_bias = at::add(additive_bias, causal_bias);
    } else {
      additive_bias = causal_bias;
    }
  }

  if (!additive_bias.defined()) {
    return additive_bias;
  }

  return utils::prepare_vulkan_execution_tensor(
      additive_bias,
      attention_policy.mask_plan_kind,
      utils::make_vulkan_attention_request(
          attention_policy, utils::VulkanTensorRole::Mask));
}

bool can_use_tiled_sdpa_fast_path(
    const vTensor& v_query,
    const vTensor& v_key,
    const vTensor& v_value) {
  return v_query.storage_type() == api::StorageType::TEXTURE_3D &&
      v_key.storage_type() == api::StorageType::TEXTURE_3D &&
      v_value.storage_type() == api::StorageType::TEXTURE_3D &&
      v_query.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED &&
      v_key.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED &&
      v_value.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED &&
      v_value.sizes().size() == 3 &&
      v_value.sizes()[2] <= kTiledSdpaMaxValueDim;
}

bool can_use_tiled_sdpa_buffer_fast_path(
    const vTensor& v_query,
    const vTensor& v_key,
    const vTensor& v_value,
    const int64_t max_sequence) {
  return v_query.storage_type() == api::StorageType::BUFFER &&
      v_key.storage_type() == api::StorageType::BUFFER &&
      v_value.storage_type() == api::StorageType::BUFFER &&
      v_query.dtype() == api::kFloat &&
      v_key.dtype() == api::kFloat &&
      v_value.dtype() == api::kFloat &&
      v_query.sizes().size() == 3 &&
      v_key.sizes().size() == 3 &&
      v_value.sizes().size() == 3 &&
      v_query.sizes()[1] <= max_sequence &&
      v_key.sizes()[1] <= max_sequence &&
      v_query.sizes()[2] <= kTiledSdpaBufferMaxHeadDim &&
      v_value.sizes()[2] <= kTiledSdpaMaxValueDim &&
      utils::supports_buffer_reduction_compute(v_query) &&
      utils::supports_buffer_reduction_compute(v_key) &&
      utils::supports_buffer_reduction_compute(v_value);
}

Tensor scaled_dot_product_attention_tiled_3d_vulkan(
    const Tensor& query_arg,
    const Tensor& key_arg,
    const Tensor& value_arg) {
  api::AllocationScope allocation_scope("sdpa");
  TORCH_CHECK(
      query_arg.is_vulkan() && key_arg.is_vulkan() && value_arg.is_vulkan(),
      "Vulkan tiled SDPA expects Vulkan tensors");
  TORCH_CHECK(
      query_arg.dim() == 3 && key_arg.dim() == 3 && value_arg.dim() == 3,
      "Vulkan tiled SDPA expects 3D tensors");
  TORCH_CHECK(
      query_arg.size(0) == key_arg.size(0) &&
          query_arg.size(0) == value_arg.size(0) &&
          query_arg.size(2) == key_arg.size(2) &&
          key_arg.size(1) == value_arg.size(1),
      "Vulkan tiled SDPA expects matching [B, T, K] / [B, S, K] / [B, S, V] shapes");

  const Tensor query =
      query_arg.is_contiguous_or_false() ? query_arg : query_arg.contiguous();
  const Tensor key =
      key_arg.is_contiguous_or_false() ? key_arg : key_arg.contiguous();
  const Tensor value =
      value_arg.is_contiguous_or_false() ? value_arg : value_arg.contiguous();

  const Tensor query_texture = utils::prepare_vulkan_execution_tensor(
      query,
      utils::VulkanExecutionPlanKind::TextureComputeInput,
      utils::make_vulkan_planning_request(
          utils::VulkanWorkloadClass::Attention,
          utils::VulkanTensorRole::Input));
  const Tensor key_texture = utils::prepare_vulkan_execution_tensor(
      key,
      utils::VulkanExecutionPlanKind::TextureComputeInput,
      utils::make_vulkan_planning_request(
          utils::VulkanWorkloadClass::Attention,
          utils::VulkanTensorRole::Input));
  const Tensor value_texture = utils::prepare_vulkan_execution_tensor(
      value,
      utils::VulkanExecutionPlanKind::TextureComputeInput,
      utils::make_vulkan_planning_request(
          utils::VulkanWorkloadClass::Attention,
          utils::VulkanTensorRole::Input));

  const vTensor& v_query = convert(query_texture);
  const vTensor& v_key = convert(key_texture);
  const vTensor& v_value = convert(value_texture);

  TORCH_CHECK(
      can_use_tiled_sdpa_fast_path(v_query, v_key, v_value),
      "Vulkan tiled SDPA expects channels-packed TEXTURE_3D inputs with value dim <= ",
      kTiledSdpaMaxValueDim);

  api::Context* const context = api::context();

  vTensor v_output{
      context,
      {query.size(0), query.size(1), value.size(2)},
      v_value.dtype(),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
  };

  const struct Block final {
    ivec4 sizes;
    ivec4 tiled_info;
  } block{
      {
          safe_downcast<int32_t>(query.size(0)),
          safe_downcast<int32_t>(query.size(1)),
          safe_downcast<int32_t>(key.size(1)),
          safe_downcast<int32_t>(query.size(2)),
      },
      {
          safe_downcast<int32_t>(value.size(2)),
          kTiledSdpaLocalSizeX,
          kTiledSdpaMaxOutputsPerThread,
          safe_downcast<int32_t>(v_output.extents().data[2u]),
      },
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(scaled_dot_product_scores_value),
      pipeline_barrier,
      {
          static_cast<uint32_t>(kTiledSdpaLocalSizeX),
          v_output.extents().data[1u],
          v_output.extents().data[2u],
      },
      {
          static_cast<uint32_t>(kTiledSdpaLocalSizeX),
          1u,
          1u,
      },
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_query.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_key.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_value.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

Tensor scaled_dot_product_attention_tiled_3d_buffer_vulkan(
    const Tensor& query_arg,
    const Tensor& key_arg,
    const Tensor& value_arg,
    const int64_t max_sequence) {
  api::AllocationScope allocation_scope("sdpa.buffer_tiled");
  TORCH_CHECK(
      query_arg.is_vulkan() && key_arg.is_vulkan() && value_arg.is_vulkan(),
      "Vulkan buffer tiled SDPA expects Vulkan tensors");
  TORCH_CHECK(
      query_arg.dim() == 3 && key_arg.dim() == 3 && value_arg.dim() == 3,
      "Vulkan buffer tiled SDPA expects 3D tensors");
  TORCH_CHECK(
      query_arg.size(0) == key_arg.size(0) &&
          query_arg.size(0) == value_arg.size(0) &&
          query_arg.size(2) == key_arg.size(2) &&
          key_arg.size(1) == value_arg.size(1),
      "Vulkan buffer tiled SDPA expects matching [B, T, K] / [B, S, K] / [B, S, V] shapes");

  const vTensor& v_query = convert(query_arg);
  const vTensor& v_key = convert(key_arg);
  const vTensor& v_value = convert(value_arg);
  TORCH_CHECK(
      can_use_tiled_sdpa_buffer_fast_path(
          v_query, v_key, v_value, max_sequence),
      "Vulkan buffer tiled SDPA expects float BUFFER inputs with value dim <= ",
      kTiledSdpaMaxValueDim,
      " and head dim <= ",
      kTiledSdpaBufferMaxHeadDim);

  api::Context* const context = api::context();
  vTensor v_output{
      context,
      {query_arg.size(0), query_arg.size(1), value_arg.size(2)},
      v_value.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };
  Tensor output = convert(v_output);
  (void)scaled_dot_product_attention_tiled_3d_buffer_out_vulkan(
      query_arg, key_arg, value_arg, output);
  utils::log_vulkan_op_hit("aten::scaled_dot_product_attention.buffer_tiled");
  return utils::mark_tensor_execution(output, api::ExecutionLayout::BUFFER_DIRECT);
}

Tensor scaled_dot_product_attention_runtime_fused_3d_buffer_out_vulkan(
    const Tensor& query_arg,
    const Tensor& key_arg,
    const Tensor& value_arg,
    Tensor& output_arg,
    const float query_scale) {
  const RuntimeProgramBufferFusedKernelVariant variant =
      select_runtime_program_buffer_fused_variant(query_arg, key_arg, value_arg);
  if (variant ==
      RuntimeProgramBufferFusedKernelVariant::Narrow16) {
    VulkanAttentionPlanDecision decision;
    decision.selected = VulkanAttentionFastPath::ScoresValueFloatSingleQuery;
    decision.reject = VulkanAttentionRejectReason::ShapeUnsupported;
    decision.batch_heads = query_arg.size(0);
    decision.target_len = query_arg.size(1);
    decision.source_len = key_arg.size(1);
    decision.head_dim = query_arg.size(2);
    decision.value_dim = value_arg.size(2);
    decision.query_vulkan = query_arg.is_vulkan();
    decision.key_vulkan = key_arg.is_vulkan();
    decision.value_vulkan = value_arg.is_vulkan();
    decision.dtype_float = query_arg.scalar_type() == kFloat &&
        key_arg.scalar_type() == kFloat && value_arg.scalar_type() == kFloat;
    decision.self_attention_shape =
        query_arg.size(1) == key_arg.size(1) &&
        query_arg.size(1) == value_arg.size(1);
    note_attention_plan_decision(
        decision,
        "aten::scaled_dot_product_attention.buffer_float_single_query");
    return scaled_dot_product_attention_tiled_3d_buffer_out_vulkan(
        query_arg, key_arg, value_arg, output_arg);
  }

  TORCH_CHECK(
      query_arg.is_vulkan() && key_arg.is_vulkan() && value_arg.is_vulkan() &&
          output_arg.is_vulkan(),
      "Vulkan wide buffer fused SDPA expects Vulkan tensors");
  TORCH_CHECK(
      query_arg.dim() == 3 && key_arg.dim() == 3 && value_arg.dim() == 3 &&
          output_arg.dim() == 3,
      "Vulkan wide buffer fused SDPA expects 3D tensors");
  TORCH_CHECK(
      query_arg.size(0) == key_arg.size(0) &&
          query_arg.size(0) == value_arg.size(0) &&
          query_arg.size(0) == output_arg.size(0) &&
          query_arg.size(1) == output_arg.size(1) &&
          query_arg.size(2) == key_arg.size(2) &&
          key_arg.size(1) == value_arg.size(1) &&
          value_arg.size(2) == output_arg.size(2),
      "Vulkan wide buffer fused SDPA expects matching [B, T, K] / [B, S, K] / [B, S, V] shapes");

  const vTensor& v_query = convert(query_arg);
  const vTensor& v_key = convert(key_arg);
  const vTensor& v_value = convert(value_arg);
  vTensor& v_output = convert(output_arg);
  TORCH_CHECK(
      can_use_runtime_program_buffer_fused_fast_path(v_query, v_key, v_value),
      "Vulkan wide buffer fused SDPA expects float BUFFER inputs with value dim <= ",
      kRuntimeProgramSdpaWideMaxValueDim,
      " and head dim <= ",
      kRuntimeProgramSdpaWideMaxHeadDim);
  TORCH_CHECK(
      v_output.storage_type() == api::StorageType::BUFFER &&
          v_output.gpu_memory_layout() ==
              api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
          v_output.dtype() == v_value.dtype(),
      "Vulkan wide buffer fused SDPA expects a width-packed float buffer output");

  VulkanAttentionPlanDecision decision;
  decision.batch_heads = query_arg.size(0);
  decision.target_len = query_arg.size(1);
  decision.source_len = key_arg.size(1);
  decision.head_dim = query_arg.size(2);
  decision.value_dim = value_arg.size(2);
  decision.query_vulkan = query_arg.is_vulkan();
  decision.key_vulkan = key_arg.is_vulkan();
  decision.value_vulkan = value_arg.is_vulkan();
  decision.query_direct_buffer = v_query.has_direct_buffer_layout();
  decision.key_direct_buffer = v_key.has_direct_buffer_layout();
  decision.value_direct_buffer = v_value.has_direct_buffer_layout();
  decision.output_direct_buffer = v_output.has_direct_buffer_layout();
  decision.dtype_float = query_arg.scalar_type() == kFloat &&
      key_arg.scalar_type() == kFloat && value_arg.scalar_type() == kFloat;
  decision.self_attention_shape =
      query_arg.size(1) == key_arg.size(1) &&
      query_arg.size(1) == value_arg.size(1);
  const int head64_query_tile =
      variant == RuntimeProgramBufferFusedKernelVariant::Head64Query4
      ? kRuntimeProgramSdpaHead64QueryRowsPerWorkgroupQ4
      : kRuntimeProgramSdpaHead64MaxQueryValuesPerThread;
  const bool head64_query_tile_variant =
      variant == RuntimeProgramBufferFusedKernelVariant::Head64Query4;
  decision.query_tile =
      head64_query_tile_variant ? head64_query_tile : 1;
  decision.selected =
      head64_query_tile_variant
      ? VulkanAttentionFastPath::ScoresValueFloatQueryTile
      : VulkanAttentionFastPath::ScoresValueFloatSingleQuery;
  decision.reject =
      can_use_head64_query_tile_attention(query_arg, key_arg, value_arg)
      ? VulkanAttentionRejectReason::None
      : VulkanAttentionRejectReason::ShapeUnsupported;
  note_attention_plan_decision(
      decision,
      head64_query_tile_variant
          ? "aten::scaled_dot_product_attention.buffer_float_qtile"
          : "aten::scaled_dot_product_attention.buffer_float_single_query");

  api::Context* const context = api::context();
  if (variant == RuntimeProgramBufferFusedKernelVariant::Head64 ||
      head64_query_tile_variant) {
    TORCH_CHECK(
        query_arg.size(2) == 64 && key_arg.size(2) == 64 &&
            value_arg.size(2) == 64,
        "Vulkan head64 buffer fused SDPA expects head_dim=value_dim=64");
    const struct Block final {
      ivec4 sizes;
      ivec4 tiled_info;
      vec4 params;
    } block{
        {
            safe_downcast<int32_t>(query_arg.size(0)),
            safe_downcast<int32_t>(query_arg.size(1)),
            safe_downcast<int32_t>(key_arg.size(1)),
            safe_downcast<int32_t>(query_arg.size(2)),
        },
        {
            safe_downcast<int32_t>(value_arg.size(2)),
            kRuntimeProgramSdpaHead64LocalSizeX,
            kRuntimeProgramSdpaHead64MaxOutputsPerThread,
            head64_query_tile_variant
                ? head64_query_tile
                : kRuntimeProgramSdpaHead64MaxQueryValuesPerThread,
        },
        {query_scale, 0.0f, 0.0f, 0.0f},
    };

    api::UniformParamsBuffer params(context, block);
    api::UniformParamsBuffer out_meta =
        utils::make_buffer_compute_metadata_ubo(context, v_output);
    api::UniformParamsBuffer query_meta =
        utils::make_buffer_compute_metadata_ubo(context, v_query);
    api::UniformParamsBuffer key_meta =
        utils::make_buffer_compute_metadata_ubo(context, v_key);
    api::UniformParamsBuffer value_meta =
        utils::make_buffer_compute_metadata_ubo(context, v_value);
    api::PipelineBarrier pipeline_barrier{};

    if (head64_query_tile_variant) {
      api::ShaderInfo shader =
          VK_KERNEL(scaled_dot_product_scores_value_buffer_float_head64_q4);
      bool subgroup_q4 = false;
      if (supports_effective_qtile_q4_subgroup_kernel()) {
        shader = VK_KERNEL(
            scaled_dot_product_scores_value_buffer_float_head64_q4_subgroup);
        shader.required_subgroup_size = 64u;
        shader.require_full_subgroups = true;
        subgroup_q4 = true;
      }
      VulkanAttentionPlanCounters& counters = attention_plan_counters();
      if (subgroup_q4) {
        counters.qtile_q4_subgroup_hit.fetch_add(
            1u, std::memory_order_relaxed);
      } else {
        counters.qtile_q4_shared_hit.fetch_add(1u, std::memory_order_relaxed);
      }
      context->submit_compute_job(
          shader,
          pipeline_barrier,
          {
              static_cast<uint32_t>(kRuntimeProgramSdpaHead64LocalSizeX),
              api::utils::div_up(
                  safe_downcast<uint32_t>(query_arg.size(1)),
                  static_cast<uint32_t>(head64_query_tile)),
              safe_downcast<uint32_t>(query_arg.size(0)),
          },
          {
              static_cast<uint32_t>(kRuntimeProgramSdpaHead64LocalSizeX),
              1u,
              1u,
          },
          VK_NULL_HANDLE,
          v_output.buffer(
              pipeline_barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::WRITE),
          out_meta.buffer(),
          v_query.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
          query_meta.buffer(),
          v_key.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
          key_meta.buffer(),
          v_value.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
          value_meta.buffer(),
          params.buffer());

      return utils::mark_tensor_execution(
          output_arg, api::ExecutionLayout::BUFFER_DIRECT);
    }

    context->submit_compute_job(
        VK_KERNEL(scaled_dot_product_scores_value_buffer_float_head64),
        pipeline_barrier,
        {
            static_cast<uint32_t>(kRuntimeProgramSdpaHead64LocalSizeX),
            safe_downcast<uint32_t>(query_arg.size(1)),
            safe_downcast<uint32_t>(query_arg.size(0)),
        },
        {
            static_cast<uint32_t>(kRuntimeProgramSdpaHead64LocalSizeX),
            1u,
            1u,
        },
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        out_meta.buffer(),
        v_query.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        query_meta.buffer(),
        v_key.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        key_meta.buffer(),
        v_value.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        value_meta.buffer(),
        params.buffer());

    return utils::mark_tensor_execution(
        output_arg, api::ExecutionLayout::BUFFER_DIRECT);
  }

  const struct Block final {
    ivec4 sizes;
    ivec4 tiled_info;
  } block{
      {
          safe_downcast<int32_t>(query_arg.size(0)),
          safe_downcast<int32_t>(query_arg.size(1)),
          safe_downcast<int32_t>(key_arg.size(1)),
          safe_downcast<int32_t>(query_arg.size(2)),
      },
      {
          safe_downcast<int32_t>(value_arg.size(2)),
          kRuntimeProgramSdpaWideLocalSizeX,
          kRuntimeProgramSdpaWideMaxOutputsPerThread,
          0,
      },
  };

  api::UniformParamsBuffer params(context, block);
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer query_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_query);
  api::UniformParamsBuffer key_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_key);
  api::UniformParamsBuffer value_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_value);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(scaled_dot_product_scores_value_buffer_float_wide),
      pipeline_barrier,
      {
          static_cast<uint32_t>(kRuntimeProgramSdpaWideLocalSizeX),
          safe_downcast<uint32_t>(query_arg.size(1)),
          safe_downcast<uint32_t>(query_arg.size(0)),
      },
      {
          static_cast<uint32_t>(kRuntimeProgramSdpaWideLocalSizeX),
          1u,
          1u,
      },
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_query.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      query_meta.buffer(),
      v_key.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      key_meta.buffer(),
      v_value.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      value_meta.buffer(),
      params.buffer());

  return utils::mark_tensor_execution(
      output_arg, utils::resolve_buffer_execution_layout(convert(output_arg)));
}

Tensor scaled_dot_product_attention_tiled_3d_buffer_out_vulkan(
    const Tensor& query_arg,
    const Tensor& key_arg,
    const Tensor& value_arg,
    Tensor& output_arg) {
  TORCH_CHECK(
      query_arg.is_vulkan() && key_arg.is_vulkan() && value_arg.is_vulkan() &&
          output_arg.is_vulkan(),
      "Vulkan buffer fused SDPA expects Vulkan tensors");
  TORCH_CHECK(
      query_arg.dim() == 3 && key_arg.dim() == 3 && value_arg.dim() == 3 &&
          output_arg.dim() == 3,
      "Vulkan buffer fused SDPA expects 3D tensors");
  TORCH_CHECK(
      query_arg.size(0) == key_arg.size(0) &&
          query_arg.size(0) == value_arg.size(0) &&
          query_arg.size(0) == output_arg.size(0) &&
          query_arg.size(1) == output_arg.size(1) &&
          query_arg.size(2) == key_arg.size(2) &&
          key_arg.size(1) == value_arg.size(1) &&
          value_arg.size(2) == output_arg.size(2),
      "Vulkan buffer fused SDPA expects matching [B, T, K] / [B, S, K] / [B, S, V] shapes");

  const vTensor& v_query = convert(query_arg);
  const vTensor& v_key = convert(key_arg);
  const vTensor& v_value = convert(value_arg);
  vTensor& v_output = convert(output_arg);
  TORCH_CHECK(
      can_use_runtime_program_buffer_fused_fast_path(v_query, v_key, v_value),
      "Vulkan buffer fused SDPA expects float BUFFER inputs with value dim <= ",
      kTiledSdpaMaxValueDim,
      " and head dim <= ",
      kTiledSdpaBufferMaxHeadDim);
  TORCH_CHECK(
      v_output.storage_type() == api::StorageType::BUFFER &&
          v_output.gpu_memory_layout() ==
              api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
          v_output.dtype() == v_value.dtype(),
      "Vulkan buffer fused SDPA expects a width-packed float buffer output");

  api::Context* const context = api::context();
  const struct Block final {
    ivec4 sizes;
    ivec4 tiled_info;
  } block{
      {
          safe_downcast<int32_t>(query_arg.size(0)),
          safe_downcast<int32_t>(query_arg.size(1)),
          safe_downcast<int32_t>(key_arg.size(1)),
          safe_downcast<int32_t>(query_arg.size(2)),
      },
      {
          safe_downcast<int32_t>(value_arg.size(2)),
          kTiledSdpaLocalSizeX,
          kTiledSdpaMaxOutputsPerThread,
          0,
      },
  };

  api::UniformParamsBuffer params(context, block);
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer query_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_query);
  api::UniformParamsBuffer key_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_key);
  api::UniformParamsBuffer value_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_value);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(scaled_dot_product_scores_value_buffer_float),
      pipeline_barrier,
      {
          static_cast<uint32_t>(kTiledSdpaLocalSizeX),
          safe_downcast<uint32_t>(query_arg.size(1)),
          safe_downcast<uint32_t>(query_arg.size(0)),
      },
      {
          static_cast<uint32_t>(kTiledSdpaLocalSizeX),
          1u,
          1u,
      },
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_query.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      query_meta.buffer(),
      v_key.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      key_meta.buffer(),
      v_value.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      value_meta.buffer(),
      params.buffer());

  return utils::mark_tensor_execution(output_arg, api::ExecutionLayout::BUFFER_DIRECT);
}

std::optional<Tensor> try_scaled_dot_product_attention_tiled_fast_path(
    const utils::VulkanRuntimePolicy& runtime_policy,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  if (
      runtime_policy.attention_execution_strategy !=
      utils::VulkanAttentionExecutionStrategy::TextureTiled) {
    log_sdpa_event(
        "tiled_fast_path",
        "reject",
        "non_texture_tiled_strategy",
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa);
    return std::nullopt;
  }
  const auto normalized_attn_mask =
      (attn_mask && attn_mask->defined()) ? attn_mask : std::nullopt;
  if (normalized_attn_mask.has_value()) {
    log_sdpa_event(
        "tiled_fast_path", "reject", "explicit_mask", query, key, value,
        attn_mask, dropout_p, is_causal, scale, enable_gqa);
    return std::nullopt;
  }
  if (dropout_p != 0.0) {
    log_sdpa_event(
        "tiled_fast_path", "reject", "dropout", query, key, value, attn_mask,
        dropout_p, is_causal, scale, enable_gqa);
    return std::nullopt;
  }
  if (is_causal) {
    log_sdpa_event(
        "tiled_fast_path", "reject", "causal", query, key, value, attn_mask,
        dropout_p, is_causal, scale, enable_gqa);
    return std::nullopt;
  }
  if (enable_gqa) {
    log_sdpa_event(
        "tiled_fast_path", "reject", "gqa", query, key, value, attn_mask,
        dropout_p, is_causal, scale, enable_gqa);
    return std::nullopt;
  }
  if (query.dim() != 3 && query.dim() != 4) {
    log_sdpa_event(
        "tiled_fast_path", "reject", "unsupported_query_rank", query, key,
        value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
    return std::nullopt;
  }
  if (key.dim() != query.dim() || value.dim() != query.dim()) {
    log_sdpa_event(
        "tiled_fast_path", "reject", "rank_mismatch", query, key, value,
        attn_mask, dropout_p, is_causal, scale, enable_gqa);
    return std::nullopt;
  }

  const int64_t target_len = query.size(query.dim() - 2);
  const int64_t source_len = key.size(key.dim() - 2);
  const int64_t head_dim = query.size(query.dim() - 1);
  const int64_t value_dim = value.size(value.dim() - 1);
  const int64_t batch = query.dim() == 4 ? query.size(0) : query.size(0);
  const int64_t heads = query.dim() == 4 ? query.size(1) : 1;
  const int64_t batch_heads = batch * heads;

  const double sdpa_scale =
      scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim)));
  Tensor query_3d = maybe_scale_query(
      flatten_attention_batch_heads(query, batch_heads, target_len, head_dim),
      sdpa_scale);
  Tensor key_3d =
      flatten_attention_batch_heads(key, batch_heads, source_len, head_dim);
  Tensor value_3d = flatten_attention_batch_heads(
      value, batch_heads, source_len, value_dim);

  if (
      (query_3d.is_vulkan() &&
       convert(query_3d).storage_type() == api::StorageType::BUFFER) ||
      (key_3d.is_vulkan() &&
       convert(key_3d).storage_type() == api::StorageType::BUFFER) ||
      (value_3d.is_vulkan() &&
       convert(value_3d).storage_type() == api::StorageType::BUFFER)) {
    log_sdpa_event(
        "tiled_fast_path", "reject", "buffer_backed_inputs", query_3d, key_3d,
        value_3d, std::nullopt, dropout_p, is_causal, scale, enable_gqa);
    return std::nullopt;
  }

  if (
      query_3d.size(0) != key_3d.size(0) ||
      query_3d.size(0) != value_3d.size(0) ||
      query_3d.size(2) != key_3d.size(2) ||
      key_3d.size(1) != value_3d.size(1)) {
    log_sdpa_event(
        "tiled_fast_path", "reject", "shape_mismatch", query_3d, key_3d,
        value_3d, std::nullopt, dropout_p, is_causal, scale, enable_gqa);
    return std::nullopt;
  }

  Tensor output =
      scaled_dot_product_attention_tiled_3d_vulkan(query_3d, key_3d, value_3d);
  log_sdpa_event(
      "tiled_fast_path", "hit", "ok", query_3d, key_3d, value_3d,
      std::nullopt, dropout_p, is_causal, scale, enable_gqa);
  if (query.dim() == 4) {
    return output.reshape({batch, heads, target_len, value_dim});
  }
  return output;
}

std::tuple<Tensor, Tensor> scaled_dot_product_attention_math_vulkan_impl(
    const Tensor& query_arg,
    const Tensor& key_arg,
    const Tensor& value_arg,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const std::optional<Tensor>& dropout_mask,
    std::optional<double> scale,
    bool enable_gqa,
    const utils::VulkanAttentionPolicy& attention_policy,
    const utils::VulkanRuntimePolicy& input_runtime_policy) {
  api::AllocationScope allocation_scope("sdpa");
  log_sdpa_event(
      "math_vulkan_entry",
      "enter",
      "ok",
      query_arg,
      key_arg,
      value_arg,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      enable_gqa);
  TORCH_CHECK(
      query_arg.is_vulkan() && key_arg.is_vulkan() && value_arg.is_vulkan(),
      "Vulkan SDPA expects query, key, and value to already be Vulkan tensors");
  TORCH_CHECK(
      (query_arg.dim() == 3 || query_arg.dim() == 4) &&
          key_arg.dim() == query_arg.dim() &&
          value_arg.dim() == query_arg.dim(),
      "Vulkan SDPA currently supports matching 3D or 4D tensors");
  TORCH_CHECK(
      dropout_p == 0.0,
      "Vulkan SDPA currently supports inference-only dropout_p=0");
  TORCH_CHECK(
      !dropout_mask.has_value(),
      "Vulkan SDPA does not support explicit dropout masks");
  TORCH_CHECK(
      query_arg.dim() == 3
          ? (query_arg.size(0) == key_arg.size(0) &&
             query_arg.size(0) == value_arg.size(0) &&
             query_arg.size(2) == key_arg.size(2) &&
             key_arg.size(1) == value_arg.size(1))
          : (query_arg.size(0) == key_arg.size(0) &&
             query_arg.size(0) == value_arg.size(0) &&
             query_arg.size(3) == key_arg.size(3) &&
             key_arg.size(2) == value_arg.size(2) &&
             (enable_gqa
                  ? (key_arg.size(1) == value_arg.size(1) &&
                     key_arg.size(1) > 0 &&
                     query_arg.size(1) % key_arg.size(1) == 0)
                  : (query_arg.size(1) == key_arg.size(1) &&
                     query_arg.size(1) == value_arg.size(1)))),
      "Vulkan SDPA expects matching 3D [B, T, K] / [B, S, K] / [B, S, V] "
      "or 4D [B, H, T, K] / [B, H, S, K] / [B, H, S, V] shapes");

  const Tensor query =
      query_arg.is_contiguous_or_false() ? query_arg : query_arg.contiguous();
  Tensor key = key_arg.is_contiguous_or_false() ? key_arg : key_arg.contiguous();
  Tensor value =
      value_arg.is_contiguous_or_false() ? value_arg : value_arg.contiguous();

  const utils::SDPAExecutionPolicyMatch sdpa_execution_policy =
      utils::match_sdpa_execution_policy_contract(
          query.sizes(),
          key.sizes(),
          value.sizes(),
          query.scalar_type(),
          key.scalar_type(),
          value.scalar_type(),
          attn_mask && attn_mask->defined(),
          dropout_p,
          is_causal,
          scale,
          enable_gqa);
  const bool materialized_diffusion_input =
      sdpa_execution_policy.requires_materialized_math_path;

  if (!materialized_diffusion_input) {
    if (const auto fast_output = try_scaled_dot_product_attention_tiled_fast_path(
            input_runtime_policy,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa)) {
      return std::make_tuple(*fast_output, Tensor());
    }
  }

  log_sdpa_event(
      "math_vulkan_entry",
      "fallback",
      "math_path",
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      enable_gqa);

  if (enable_gqa) {
    TORCH_CHECK(
        query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
        "Vulkan SDPA GQA currently supports 4D tensors only");
    TORCH_CHECK(
        key.size(1) == value.size(1) &&
            query.size(1) % key.size(1) == 0,
        "Vulkan SDPA GQA expects query heads to be divisible by key/value heads");
    if (can_use_direct_gqa_sdpa_buffer_path(
            query, key, value, attn_mask, is_causal)) {
      const int64_t head_dim = query.size(3);
      const double sdpa_scale =
          scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim)));
      Tensor output = scaled_dot_product_attention_direct_gqa_4d_buffer_vulkan(
          query, key, value, sdpa_scale, is_causal);
      Tensor output_4d =
          output.reshape({query.size(0), query.size(1), query.size(2), value.size(3)});
      Tensor materialized_output = utils::ensure_buffer_storage(
          output_4d, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
      return std::make_tuple(
          utils::mark_tensor_execution(
              materialized_output, api::ExecutionLayout::BUFFER_DIRECT),
          Tensor());
    }
    const int64_t repeat_factor = query.size(1) / key.size(1);
    key = repeat_attention_heads_for_gqa(key, repeat_factor);
    value = repeat_attention_heads_for_gqa(value, repeat_factor);
  }

  const int64_t target_len = query.size(query.dim() - 2);
  const int64_t source_len = key.size(key.dim() - 2);
  const int64_t head_dim = query.size(query.dim() - 1);
  const int64_t value_dim = value.size(value.dim() - 1);

  const double sdpa_scale =
      scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim)));
  const double query_scale = sdpa_scale;

  const int64_t batch = query.dim() == 4 ? query.size(0) : query.size(0);
  const int64_t heads = query.dim() == 4 ? query.size(1) : 1;
  const int64_t batch_heads = batch * heads;

  Tensor query_3d = maybe_scale_query(
      flatten_attention_batch_heads(query, batch_heads, target_len, head_dim),
      query_scale);
  Tensor key_3d =
      flatten_attention_batch_heads(key, batch_heads, source_len, head_dim);
  Tensor value_3d =
      flatten_attention_batch_heads(value, batch_heads, source_len, value_dim);

  prime_attention_runtime_objects(
      input_runtime_policy, attention_policy, query_3d, key_3d, value_3d);

  const auto query_request =
      utils::make_vulkan_attention_request(
          attention_policy,
          query_3d,
          key_3d,
          value_3d,
          utils::VulkanTensorRole::Input);
  const auto key_value_request = utils::make_vulkan_attention_request(
      attention_policy,
      query_3d,
      key_3d,
      value_3d,
      attention_policy.cache_mode == utils::VulkanAttentionCacheMode::Disabled
          ? utils::VulkanTensorRole::Input
          : utils::VulkanTensorRole::Cache);
  query_3d = utils::prepare_vulkan_execution_tensor(
      query_3d, attention_policy.query_plan_kind, query_request);
  key_3d = utils::prepare_vulkan_execution_tensor(
      key_3d, attention_policy.key_value_plan_kind, key_value_request);
  value_3d = utils::prepare_vulkan_execution_tensor(
      value_3d, attention_policy.key_value_plan_kind, key_value_request);

  query_3d = prepare_buffer_math_input_direct(query_3d);
  key_3d = prepare_buffer_math_input_direct(key_3d);
  value_3d = prepare_buffer_math_input_direct(value_3d);

  const bool uses_buffer_math =
      can_use_attention_buffer_math_ops(query_3d, key_3d, value_3d);
  if (uses_buffer_math) {
    utils::log_vulkan_op_hit("aten::scaled_dot_product_attention.buffer_math_ops");
  }

  const bool has_explicit_mask = attn_mask && attn_mask->defined();
  const int64_t buffer_tiled_max_sequence =
      tiled_sdpa_buffer_fast_path_max_sequence(input_runtime_policy);
  if (
      uses_buffer_math &&
      !is_generic_attention_policy(input_runtime_policy) &&
      input_runtime_policy.attention_execution_strategy ==
          utils::VulkanAttentionExecutionStrategy::BufferTiled &&
      !has_explicit_mask &&
      !is_causal &&
      !enable_gqa &&
      can_use_tiled_sdpa_buffer_fast_path(
          convert(query_3d),
          convert(key_3d),
          convert(value_3d),
          buffer_tiled_max_sequence)) {
    Tensor output = scaled_dot_product_attention_tiled_3d_buffer_vulkan(
        query_3d, key_3d, value_3d, buffer_tiled_max_sequence);
    log_sdpa_event(
        "buffer_tiled_fast_path",
        "hit",
        "ok",
        query_3d,
        key_3d,
        value_3d,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa);
    if (query.dim() == 3) {
      return std::make_tuple(output, Tensor());
    }

    return std::make_tuple(
        materialize_buffer_attention_output_view(
            output.reshape({batch, heads, target_len, value_dim})),
        Tensor());
  }

  const utils::SDPAExecutionPolicyMatch attention_probability_policy =
      utils::match_sdpa_execution_policy_contract(
          query.sizes(),
          key.sizes(),
          value.sizes(),
          query.scalar_type(),
          key.scalar_type(),
          value.scalar_type(),
          has_explicit_mask,
          dropout_p,
          is_causal,
          scale,
          enable_gqa);

  Tensor attn = at::bmm(query_3d, key_3d.transpose(1, 2));
  Tensor additive_bias = prepare_attention_bias(
      attn_mask,
      attention_policy,
      query,
      batch,
      heads,
      target_len,
      source_len);
  if (additive_bias.defined()) {
    attn = at::add(attn, additive_bias);
  }
  if (attention_probability_policy.requires_score_pre_materialization) {
    attn = prepare_buffer_math_input_direct(attn);
  }
  attn = attn.softmax(-1);
  if (attention_probability_policy.requires_post_softmax_clone) {
    attn = attn.clone();
  }
  Tensor output = at::bmm(attn, value_3d);

  if (query.dim() == 3) {
    return std::make_tuple(output, attn);
  }

  return std::make_tuple(
      materialize_buffer_attention_output_view(
          output.reshape({batch, heads, target_len, value_dim})),
      attn.reshape({batch, heads, target_len, source_len}));
}

std::tuple<Tensor, Tensor> scaled_dot_product_attention_math_vulkan(
    const Tensor& query_arg,
    const Tensor& key_arg,
    const Tensor& value_arg,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const std::optional<Tensor>& dropout_mask,
    std::optional<double> scale,
    bool enable_gqa) {
  utils::log_vulkan_op_hit("aten::_scaled_dot_product_attention_math");
  const auto attention_policy = utils::build_vulkan_attention_policy(
      attn_mask, is_causal, enable_gqa, false, false);
  const auto input_runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_attention_request(
          attention_policy,
          query_arg,
          key_arg,
          value_arg,
          utils::VulkanTensorRole::Input,
          dropout_p != 0.0));
  const auto route_decision = utils::select_sdpa_route(
      query_arg,
      key_arg,
      value_arg,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      enable_gqa,
      input_runtime_policy.request,
      utils::current_vulkan_device_policy());
  if (route_decision.hard_fail) {
    utils::fail_hard_fail(
        "aten::_scaled_dot_product_attention_math", route_decision);
  }
  log_attention_kernel_family_choice(input_runtime_policy);
  log_attention_execution_strategy_choice(input_runtime_policy);
  return scaled_dot_product_attention_math_vulkan_impl(
      query_arg,
      key_arg,
      value_arg,
      attn_mask,
      dropout_p,
      is_causal,
      dropout_mask,
      scale,
      enable_gqa,
      attention_policy,
      input_runtime_policy);
}

Tensor scaled_dot_product_attention_vulkan_impl(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  api::AllocationScope allocation_scope("sdpa");
  const auto attention_policy = utils::build_vulkan_attention_policy(
      attn_mask, is_causal, enable_gqa, false, false);
  const auto input_runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_attention_request(
          attention_policy,
          query,
          key,
          value,
          utils::VulkanTensorRole::Input,
          dropout_p != 0.0));
  const auto route_decision = utils::select_sdpa_route(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      enable_gqa,
      input_runtime_policy.request,
      utils::current_vulkan_device_policy());
  if (route_decision.hard_fail) {
    utils::fail_hard_fail(
        "aten::scaled_dot_product_attention", route_decision);
  }
  log_attention_kernel_family_choice(input_runtime_policy);
  log_attention_execution_strategy_choice(input_runtime_policy);
  log_sdpa_event(
      "public_vulkan_entry",
      "enter",
      "ok",
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      enable_gqa);
  const utils::SDPAExecutionPolicyMatch sdpa_execution_policy =
      utils::match_sdpa_execution_policy_contract(
          query.sizes(),
          key.sizes(),
          value.sizes(),
          query.scalar_type(),
          key.scalar_type(),
          value.scalar_type(),
          attn_mask && attn_mask->defined(),
          dropout_p,
          is_causal,
          scale,
          enable_gqa);
  const bool materialized_diffusion_input =
      sdpa_execution_policy.requires_materialized_math_path;
  if (!materialized_diffusion_input) {
    if (const auto fast_output = try_scaled_dot_product_attention_tiled_fast_path(
            input_runtime_policy,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa)) {
      return finalize_public_sdpa_output(*fast_output);
    }
  }
  log_sdpa_event(
      "public_vulkan_entry",
      "fallback",
      "math_path",
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      enable_gqa);
  if (
      input_runtime_policy.attention_execution_strategy ==
          utils::VulkanAttentionExecutionStrategy::RuntimeProgram &&
      !materialized_diffusion_input &&
      !is_generic_attention_policy(input_runtime_policy) &&
      dropout_p == 0.0 && !attn_mask.has_value() && !is_causal &&
      !enable_gqa && (query.dim() == 3 || query.dim() == 4)) {
    const Tensor query_contig =
        query.is_contiguous_or_false() ? query : query.contiguous();
    const Tensor key_contig = key.is_contiguous_or_false() ? key : key.contiguous();
    const Tensor value_contig =
        value.is_contiguous_or_false() ? value : value.contiguous();
    const int64_t target_len = query_contig.size(query_contig.dim() - 2);
    const int64_t source_len = key_contig.size(key_contig.dim() - 2);
    const int64_t head_dim = query_contig.size(query_contig.dim() - 1);
    const int64_t value_dim = value_contig.size(value_contig.dim() - 1);
    const int64_t batch =
        query_contig.dim() == 4 ? query_contig.size(0) : query_contig.size(0);
    const int64_t heads = query_contig.dim() == 4 ? query_contig.size(1) : 1;
    const int64_t batch_heads = batch * heads;
    const double sdpa_scale =
        scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim)));

    Tensor query_3d = maybe_scale_query(
        flatten_attention_batch_heads(
            query_contig, batch_heads, target_len, head_dim),
        sdpa_scale);
    Tensor key_3d = flatten_attention_batch_heads(
        key_contig, batch_heads, source_len, head_dim);
    Tensor value_3d = flatten_attention_batch_heads(
        value_contig, batch_heads, source_len, value_dim);

    prime_attention_runtime_objects(
        input_runtime_policy,
        attention_policy,
        query_3d,
        key_3d,
        value_3d);
    if (
        input_runtime_policy.execution_program_plan.has_value() &&
        input_runtime_policy.execution_program_plan->kind ==
            utils::VulkanExecutionProgramKind::AttentionRuntime) {
      const auto query_request = utils::make_vulkan_attention_request(
          attention_policy,
          query_3d,
          key_3d,
          value_3d,
          utils::VulkanTensorRole::Input);
      const auto key_value_request = utils::make_vulkan_attention_request(
          attention_policy,
          query_3d,
          key_3d,
          value_3d,
          attention_policy.cache_mode ==
                  utils::VulkanAttentionCacheMode::Disabled
              ? utils::VulkanTensorRole::Input
              : utils::VulkanTensorRole::Cache);
      query_3d = utils::prepare_vulkan_execution_tensor(
          query_3d, attention_policy.query_plan_kind, query_request);
      key_3d = utils::prepare_vulkan_execution_tensor(
          key_3d, attention_policy.key_value_plan_kind, key_value_request);
      value_3d = utils::prepare_vulkan_execution_tensor(
          value_3d, attention_policy.key_value_plan_kind, key_value_request);

      query_3d = prepare_buffer_math_input_direct(query_3d);
      key_3d = prepare_buffer_math_input_direct(key_3d);
      value_3d = prepare_buffer_math_input_direct(value_3d);
      if (can_use_attention_runtime_buffer_math_replay(
              query_3d, key_3d, value_3d)) {
        Tensor output = run_attention_runtime_buffer_math_replay_impl(
            query_3d,
            key_3d,
            value_3d,
            utils::make_vulkan_runtime_object_label(
                input_runtime_policy.request, "attention_runtime_graph"));
        log_sdpa_event(
            "public_vulkan_entry",
            "replay",
            "attention_runtime",
            query_3d,
            key_3d,
            value_3d,
            std::nullopt,
            dropout_p,
            is_causal,
            scale,
            enable_gqa);
        Tensor public_output = query.dim() == 4
            ? materialize_buffer_attention_output_view(
                  output.reshape({batch, heads, target_len, value_dim}))
            : output;
        return finalize_public_sdpa_output(public_output);
      }
    }
  }
  return finalize_public_sdpa_output(
      std::get<0>(scaled_dot_product_attention_math_vulkan_impl(
          query,
          key,
          value,
          attn_mask,
          dropout_p,
          is_causal,
          std::nullopt,
          scale,
          enable_gqa,
          attention_policy,
          input_runtime_policy)));
}

void set_softmax_kernel_params(
    const long long num_dims,
    const long long softmax_dim,
    const IntArrayRef v_input_sizes,
    api::ShaderInfo& shader_descriptor,
    api::utils::ivec4& input_shader_extents,
    api::utils::ivec4& early_exit,
    api::utils::ivec4& input_dim_stride,
    api::utils::ivec4& input_tensor_dims) {
  if (num_dims == 1) {
    early_exit.data[0u] = 1;
    input_dim_stride.data[0u] = 1;
    shader_descriptor = VK_KERNEL(softmax_batch_height_width);
  } else if (num_dims == 2) {
    // for height, width dim case, we can reuse a single shader
    // with vectorized parameters
    if (softmax_dim == 0) {
      early_exit.data[1u] = 1;
      input_dim_stride.data[1u] = 1;
      shader_descriptor = VK_KERNEL(softmax_batch_height_width);
    } else { // dim == 1
      early_exit.data[0u] = 1;
      input_dim_stride.data[0u] = 1;
      shader_descriptor = VK_KERNEL(softmax_batch_height_width);
    }
  } else if (num_dims == 3) {
    // for height, width dim case, we can reuse a single shader
    // with vectorized parameters
    for (uint32_t i = 0; i < num_dims; i++) {
      input_tensor_dims.data[i + 1] = safe_downcast<int32_t>(v_input_sizes[i]);
    }
    if (softmax_dim == 0) {
      early_exit.data[2u] = 1;
      input_dim_stride.data[2u] = 1;
      shader_descriptor = VK_KERNEL(softmax_channel);
    } else if (softmax_dim == 1) {
      early_exit.data[1u] = 1;
      input_dim_stride.data[1u] = 1;
      shader_descriptor = VK_KERNEL(softmax_batch_height_width);
    } else { // dim == 2
      early_exit.data[0u] = 1;
      input_dim_stride.data[0u] = 1;
      shader_descriptor = VK_KERNEL(softmax_batch_height_width);
    }
  } else {
    // assume num_dims is 4
    // for batch, height, width dim case, we can reuse a single shader
    // with vectorized parameters
    for (uint32_t i = 0; i < num_dims; i++) {
      input_tensor_dims.data[i] = safe_downcast<int32_t>(v_input_sizes[i]);
    }
    if (softmax_dim == 1) {
      // for 4-rank Tensor, softmax along channel dim case, the memory layout
      // forces a different shader algorithm than other dims
      input_shader_extents.data[2u] =
          v_input_sizes[Layout::Activation4D::batch];
      shader_descriptor = VK_KERNEL(softmax_channel);
    } else {
      if (softmax_dim == 0) {
        early_exit.data[2u] = safe_downcast<int32_t>(
            std::ceil(v_input_sizes[Layout::Activation4D::channels] / 4.0));
        input_dim_stride.data[2u] = safe_downcast<int32_t>(
            std::ceil(v_input_sizes[Layout::Activation4D::channels] / 4.0));
      } else if (softmax_dim == 2) {
        early_exit.data[1u] = 1;
        input_dim_stride.data[1u] = 1;
      } else { // dim == 3
        early_exit.data[0u] = 1;
        input_dim_stride.data[0u] = 1;
      }
      shader_descriptor = VK_KERNEL(softmax_batch_height_width);
    }
  }
}

Tensor softmax_internal(
    const at::Tensor& input_arg,
    const int64_t dim_arg,
    const bool half_to_float) {
  TORCH_CHECK(
      input_arg.dim() >= 1 && input_arg.dim() <= 4,
      "Vulkan softmax expects 1,2,3 or 4-dimensional input!");
  int64_t dim = utils::normalize(dim_arg, input_arg.dim());
  TORCH_CHECK(
      dim >= 0 && dim < input_arg.dim(),
      "Softmax dim input was ",
      dim,
      " out of range for Tensor input with dimensions ",
      input_arg.dim());

  if (auto propagated =
          try_propagate_decomposed_attention_softmax(input_arg, dim)) {
    return *propagated;
  }
  const Tensor input_for_compute =
      materialize_deferred_linear_gelu_candidate_if_needed(
          materialize_decomposed_attention_candidate_if_needed(input_arg));

  if (!half_to_float) {
    if (can_run_buffer_softmax(input_for_compute, dim)) {
      return softmax_buffer(input_for_compute, dim);
    }
    if (
        input_for_compute.is_vulkan() && input_for_compute.scalar_type() == kFloat &&
        input_for_compute.dim() == 3 && dim == input_for_compute.dim() - 1 &&
        input_for_compute.size(dim) >= 64) {
      utils::log_vulkan_op_hit(
          "aten::_softmax.buffer_lastdim_known_bad_texture_fallback");
    }
  }
  api::Context* const context = api::context();

  Tensor input = utils::prepare_vulkan_execution_tensor(
      input_for_compute,
      utils::VulkanExecutionPlanKind::TextureComputeInput,
      utils::make_vulkan_execution_request(
          utils::VulkanExecutionPlanKind::TextureComputeInput));
  const vTensor& v_input = convert(input);

  vTensor v_output{
      context,
      v_input.sizes(),
      v_input.dtype(),
  };
  const api::utils::uvec3 global_workgroup_extents = v_output.extents();
  api::utils::ivec4 input_shader_extents = {
      safe_downcast<int32_t>(v_input.extents().data[0u]),
      safe_downcast<int32_t>(v_input.extents().data[1u]),
      safe_downcast<int32_t>(v_input.extents().data[2u]),
      0 // zero pad
  };
  // early_exit is the global workgroup position-based condition for
  // unnecessary invocations to exit.
  api::utils::ivec4 early_exit = {
      safe_downcast<int32_t>(v_input.extents().data[0u]),
      safe_downcast<int32_t>(v_input.extents().data[1u]),
      safe_downcast<int32_t>(v_input.extents().data[2u]),
      0 // zero pad
  };
  // for batch/height/width, they share the same shader
  // vectorized by input_dim_stride for each dimension case
  api::utils::ivec4 input_dim_stride = {
      0,
      0,
      0,
      0, // zero pad
  };
  api::utils::ivec4 input_tensor_dims = {
      0,
      0,
      0,
      0,
  };
  api::ShaderInfo shader_descriptor;
  set_softmax_kernel_params(
      input_for_compute.dim(),
      dim,
      v_input.sizes(),
      shader_descriptor,
      input_shader_extents,
      early_exit,
      input_dim_stride,
      input_tensor_dims);

  const struct Block final {
    ivec4 input_shader_extents;
    ivec4 input_tensor_dims;
    ivec4 input_dim_stride;
    ivec4 early_exit;
  } block{
      input_shader_extents, input_tensor_dims, input_dim_stride, early_exit};
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_workgroup_extents,
      // local work group size
      adaptive_work_group_size(global_workgroup_extents),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor softmax(
    const at::Tensor& input_arg,
    const int64_t dim,
    const bool half_to_float) {
  utils::log_vulkan_op_hit("aten::_softmax");
  return softmax_internal(input_arg, dim, half_to_float);
}

Tensor log_softmax(
    const at::Tensor& input_arg,
    const int64_t dim,
    const bool half_to_float) {
  utils::log_vulkan_op_hit("aten::_log_softmax");
  // After computing softmax, some values are so small that they are below the
  // float16 precision. These values are represented as 0 in float16 and result
  // in -inf when log is applied. According to Wikipedia:
  // https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Exponent_encoding,
  // the minimum strictly positive (subnormal) value is 2^−24 ≈ 5.9605 × 10^−8.
  // Therefore, we add 6 x 10^-8 to the output of softmax to avoid the numerical
  // issue.
  float epsilon = 6e-8;
  return softmax_internal(input_arg, dim, half_to_float).add(epsilon).log();
}

} // namespace

std::vector<int64_t> attention_plan_counters_snapshot() {
  const VulkanAttentionPlanCounters& counters = attention_plan_counters();
  return {
      static_cast<int64_t>(counters.total.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.single_query_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.qtile_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.reject_dtype.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.reject_layout.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.reject_mask.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_dropout.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.reject_causal.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_head_dim.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.reject_shape.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.qtile_q4_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.qtile_q4_shared_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.qtile_q4_subgroup_hit.load(std::memory_order_relaxed)),
  };
}

std::vector<int64_t> attention_subgroup_capabilities_snapshot() {
  api::Context* const context = api::context();
  const api::Adapter* const adapter = context ? context->adapter_ptr() : nullptr;
  if (adapter == nullptr) {
    return {0, 0, 0, 0, 0};
  }

  return {
      adapter->has_compute_full_subgroups() ? 1 : 0,
      static_cast<int64_t>(adapter->min_subgroup_size()),
      static_cast<int64_t>(adapter->max_subgroup_size()),
      static_cast<int64_t>(adapter->required_subgroup_size_stages()),
      adapter->supports_required_subgroup_size(
          VK_SHADER_STAGE_COMPUTE_BIT, 64u) ? 1 : 0,
      static_cast<int64_t>(adapter->subgroup_size()),
      static_cast<int64_t>(adapter->subgroup_supported_stages()),
      static_cast<int64_t>(adapter->subgroup_supported_operations()),
      adapter->supports_compute_subgroup_operations(
          VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
          VK_SUBGROUP_FEATURE_BALLOT_BIT) ? 1 : 0,
  };
}

void reset_attention_plan_counters() {
  VulkanAttentionPlanCounters& counters = attention_plan_counters();
  counters.total.store(0u, std::memory_order_relaxed);
  counters.single_query_hit.store(0u, std::memory_order_relaxed);
  counters.qtile_hit.store(0u, std::memory_order_relaxed);
  counters.reject_dtype.store(0u, std::memory_order_relaxed);
  counters.reject_layout.store(0u, std::memory_order_relaxed);
  counters.reject_mask.store(0u, std::memory_order_relaxed);
  counters.reject_dropout.store(0u, std::memory_order_relaxed);
  counters.reject_causal.store(0u, std::memory_order_relaxed);
  counters.reject_head_dim.store(0u, std::memory_order_relaxed);
  counters.reject_shape.store(0u, std::memory_order_relaxed);
  counters.qtile_q4_hit.store(0u, std::memory_order_relaxed);
  counters.qtile_q4_shared_hit.store(0u, std::memory_order_relaxed);
  counters.qtile_q4_subgroup_hit.store(0u, std::memory_order_relaxed);
}

std::tuple<Tensor, Tensor, Tensor> transform_bias_rescale_qkv_vulkan_out(
    const Tensor& qkv,
    const Tensor& qkv_bias,
    const int64_t num_head,
    const Tensor& q_out,
    const Tensor& k_out,
    const Tensor& v_out) {
  return transform_bias_rescale_qkv_vulkan_out_impl(
      qkv, qkv_bias, num_head, q_out, k_out, v_out);
}

Tensor softmax_buffer_lastdim_out_vulkan(
    const Tensor& input,
    Tensor& output) {
  const auto route_request = utils::make_vulkan_planning_request(
      utils::VulkanWorkloadClass::Attention,
      utils::VulkanTensorRole::Input,
      utils::VulkanModelDomain::Vision,
      utils::VulkanExecutionPhase::Backbone);
  const auto route_decision = utils::select_softmax_route(
      input,
      input.dim() == 0 ? 0 : input.dim() - 1,
      route_request,
      utils::current_vulkan_device_policy());
  if (route_decision.hard_fail) {
    utils::fail_hard_fail("aten::_softmax", route_decision);
  }
  TORCH_CHECK(
      can_run_buffer_softmax(input, input.dim() - 1),
      "Vulkan softmax_buffer_lastdim_out expects float buffer-backed tensors");
  return softmax_buffer_lastdim_impl(input, &output);
}

std::optional<Tensor> try_start_decomposed_attention_scores(
    const Tensor& query_arg,
    const Tensor& key_t_arg) {
  Tensor query_source = query_arg;
  float query_scale = 1.0f;
  const auto scaled_query =
      lookup_deferred_attention_query_scale_candidate(query_arg);
  if (scaled_query.has_value()) {
    query_source = scaled_query->query;
    query_scale = scaled_query->scale;
  }

  if (!can_start_decomposed_attention_candidate(query_source, key_t_arg)) {
    return std::nullopt;
  }

  if (scaled_query.has_value()) {
    utils::log_vulkan_op_hit(
        "aten::decomposed_attention_bridge.consume_query_scale");
  }

  Tensor query = detached_attention_tensor(query_source);
  Tensor key_t = detached_attention_tensor(key_t_arg);
  Tensor key = key_t.transpose(1, 2);
  const std::vector<int64_t> scores_sizes{
      query.size(0),
      query.size(1),
      key.size(1),
  };
  Tensor scores = utils::create_buffer_tensor(
      scores_sizes, kFloat, /*persistent=*/false);
  register_decomposed_attention_candidate(
      scores,
      DecomposedAttentionCandidate{
          std::move(query),
          std::move(key),
          std::move(key_t),
          DecomposedAttentionStage::Scores,
          query_scale,
      });
  utils::log_vulkan_op_hit("aten::decomposed_attention_bridge.scores");
  return scores;
}

std::optional<Tensor> try_propagate_decomposed_attention_softmax(
    const Tensor& input,
    const int64_t dim) {
  const auto candidate = lookup_decomposed_attention_candidate(input);
  if (
      !candidate.has_value() ||
      candidate->stage != DecomposedAttentionStage::Scores ||
      dim != input.dim() - 1) {
    return std::nullopt;
  }

  auto taken = take_decomposed_attention_candidate(input);
  if (!taken.has_value()) {
    return std::nullopt;
  }
  taken->stage = DecomposedAttentionStage::Probs;
  Tensor probs = utils::create_buffer_tensor(
      input.sizes(), input.scalar_type(), /*persistent=*/false);
  register_decomposed_attention_candidate(probs, std::move(*taken));
  utils::log_vulkan_op_hit("aten::decomposed_attention_bridge.softmax");
  return probs;
}

std::optional<Tensor> try_consume_decomposed_attention_probs(
    const Tensor& probs,
    const Tensor& value_arg) {
  const auto candidate = lookup_decomposed_attention_candidate(probs);
  if (
      !candidate.has_value() ||
      !can_consume_decomposed_attention_candidate(*candidate, value_arg)) {
    return std::nullopt;
  }

  auto taken = take_decomposed_attention_candidate(probs);
  if (!taken.has_value()) {
    return std::nullopt;
  }
  Tensor value = detached_attention_tensor(value_arg);
  Tensor query = scaled_decomposed_attention_query(*taken);
  utils::log_vulkan_op_hit("aten::decomposed_attention_bridge.hit");
  const auto attention_policy = utils::build_vulkan_attention_policy(
      std::nullopt,
      /*is_causal=*/false,
      /*enable_gqa=*/false,
      /*use_kv_cache=*/false,
      /*cache_has_previous_state=*/false);
  const auto input_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_attention_request(
          attention_policy,
          query,
          taken->key,
          value,
          utils::VulkanTensorRole::Input));
  auto runtime_program = lookup_attention_runtime_program_for_inputs(
      input_policy,
      attention_policy,
      query,
      taken->key,
      value);
  std::optional<Tensor> output_override =
      make_decomposed_attention_merge_friendly_output(query, value);
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_attention_runtime_buffer_math_program_bridge");
  return run_attention_runtime_buffer_math_program_impl(
      query,
      taken->key,
      value,
      runtime_program.has_value() ? &(*runtime_program) : nullptr,
      output_override.has_value() ? &(*output_override) : nullptr,
      1.0f);
}

Tensor materialize_decomposed_attention_candidate_if_needed(
    const Tensor& tensor) {
  if (!tensor.is_vulkan()) {
    return tensor;
  }
  auto candidate = take_decomposed_attention_candidate(tensor);
  if (!candidate.has_value()) {
    return tensor;
  }
  return materialize_decomposed_attention_candidate(tensor, std::move(*candidate));
}

std::optional<Tensor> try_start_deferred_attention_query_scale(
    const Tensor& query,
    const Scalar& scale_arg) {
  const float scale = scale_arg.to<float>();
  if (!can_start_deferred_attention_query_scale_candidate(query, scale)) {
    return std::nullopt;
  }

  Tensor output = utils::create_buffer_tensor(
      query.sizes(), query.scalar_type(), /*persistent=*/false);
  register_deferred_attention_query_scale_candidate(
      output,
      DeferredAttentionQueryScaleCandidate{
          detached_attention_tensor(query),
          scale,
      });
  utils::log_vulkan_op_hit("aten::attention_query_scale_bridge.defer");
  return output;
}

Tensor materialize_deferred_attention_query_scale_candidate_if_needed(
    const Tensor& tensor) {
  return materialize_deferred_attention_query_scale_candidate_impl(tensor);
}

void move_decomposed_attention_candidate_to_alias(
    const Tensor& source,
    const Tensor& alias) {
  if (!source.is_vulkan() || !alias.is_vulkan()) {
    return;
  }
  auto candidate = take_decomposed_attention_candidate(source);
  if (!candidate.has_value()) {
    return;
  }
  register_decomposed_attention_candidate(alias, std::move(*candidate));
  utils::log_vulkan_op_hit("aten::decomposed_attention_bridge.alias");
}

void move_deferred_attention_query_scale_candidate_to_alias(
    const Tensor& source,
    const Tensor& alias) {
  if (!source.is_vulkan() || !alias.is_vulkan()) {
    return;
  }
  auto candidate = lookup_deferred_attention_query_scale_candidate(source);
  if (!candidate.has_value()) {
    return;
  }
  if (!candidate->query.sizes().equals(alias.sizes())) {
    candidate->query = candidate->query.reshape(alias.sizes());
  }
  register_deferred_attention_query_scale_candidate(alias, std::move(*candidate));
  utils::log_vulkan_op_hit("aten::attention_query_scale_bridge.alias");
}

Tensor scaled_dot_product_attention_vulkan(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  utils::log_vulkan_op_hit("aten::scaled_dot_product_attention");
  return scaled_dot_product_attention_vulkan_impl(
      query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
}

Tensor scaled_dot_product_attention_autograd_other(
    c10::DispatchKeySet ks,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  (void)ks;
  return scaled_dot_product_attention_vulkan(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      enable_gqa);
}

Tensor run_attention_runtime_buffer_math_replay_bridge(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value) {
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_attention_runtime_buffer_math_replay_bridge");
  return run_attention_runtime_buffer_math_replay_impl(
      query,
      key,
      value,
      utils::make_vulkan_runtime_object_label(
          utils::make_vulkan_planning_request(
              utils::VulkanWorkloadClass::Attention,
              utils::VulkanTensorRole::Input),
          "attention_runtime_bridge"));
}

Tensor run_attention_runtime_buffer_math_program_bridge(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value) {
  const auto attention_policy = utils::build_vulkan_attention_policy(
      std::nullopt,
      /*is_causal=*/false,
      /*enable_gqa=*/false,
      /*use_kv_cache=*/false,
      /*cache_has_previous_state=*/false);
  const auto input_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_attention_request(
          attention_policy,
          query,
          key,
          value,
          utils::VulkanTensorRole::Input));
  auto runtime_program = lookup_attention_runtime_program_for_inputs(
      input_policy,
      attention_policy,
      query,
      key,
      value);
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_attention_runtime_buffer_math_program_bridge");
  return run_attention_runtime_buffer_math_program_impl(
      query,
      key,
      value,
      runtime_program.has_value() ? &(*runtime_program) : nullptr);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("_softmax", TORCH_FN(softmax));
  m.impl("_log_softmax", TORCH_FN(log_softmax));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_transform_bias_rescale_qkv"),
      TORCH_FN(transform_bias_rescale_qkv_vulkan));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::scaled_dot_product_attention"),
      TORCH_FN(scaled_dot_product_attention_vulkan));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_scaled_dot_product_attention_math"),
      TORCH_FN(scaled_dot_product_attention_math_vulkan));
}

TORCH_LIBRARY_IMPL(aten, AutogradOther, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::scaled_dot_product_attention"),
      TORCH_FN(scaled_dot_product_attention_autograd_other));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
