#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/planning/Persistence.h>

#include <c10/core/InferenceMode.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <fstream>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

using namespace api::utils;

constexpr size_t kVulkanExecutionPlanKindCount =
    static_cast<size_t>(VulkanExecutionPlanKind::NumKinds);

VulkanTensorRole execution_plan_tensor_role(const VulkanExecutionPlanKind kind) {
  switch (kind) {
    case VulkanExecutionPlanKind::LinearWeightSource:
    case VulkanExecutionPlanKind::LinearPackedWeight:
    case VulkanExecutionPlanKind::Conv2dWeightSource:
    case VulkanExecutionPlanKind::Conv1dPrepackWeight:
    case VulkanExecutionPlanKind::Conv1dRuntimeWeight:
      return VulkanTensorRole::Weight;
    case VulkanExecutionPlanKind::LinearBiasSource:
    case VulkanExecutionPlanKind::LinearPackedBias:
    case VulkanExecutionPlanKind::Conv2dBiasSource:
    case VulkanExecutionPlanKind::Conv1dPrepackBias:
    case VulkanExecutionPlanKind::Conv1dRuntimeBias:
      return VulkanTensorRole::Bias;
    case VulkanExecutionPlanKind::AttentionMaskInput:
      return VulkanTensorRole::Mask;
    case VulkanExecutionPlanKind::AttentionCacheInput:
    case VulkanExecutionPlanKind::AttentionCacheAppendInput:
      return VulkanTensorRole::Cache;
    default:
      return VulkanTensorRole::Input;
  }
}

size_t execution_plan_kind_index(const VulkanExecutionPlanKind kind) {
  const size_t idx = static_cast<size_t>(kind);
  TORCH_INTERNAL_ASSERT(
      idx < kVulkanExecutionPlanKindCount,
      "Invalid VulkanExecutionPlanKind");
  return idx;
}

Tensor materialize_inference_vulkan_matrix_arg(const Tensor& tensor) {
  if (
      c10::InferenceMode::is_enabled() &&
      tensor.is_vulkan() &&
      tensor.dim() == 2 &&
      !tensor.is_contiguous_or_false()) {
    return tensor.contiguous(c10::MemoryFormat::Contiguous);
  }
  return tensor;
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
          {"NormInput",
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
          {"AttentionInput",
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
          {"AttentionMaskInput",
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
          {"AttentionCacheInput",
           api::ExecutionLayout::TEXTURE,
           api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
           api::StorageType::TEXTURE_3D,
           VulkanExecutionPolicyBufferRule::Never,
           VulkanExecutionPolicyMemoryRule::Fixed,
           true,
           false,
           false,
           false,
           true},
          {"AttentionCacheAppendInput",
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

void apply_runtime_family_overrides(
    const VulkanExecutionPlanKind kind,
    const VulkanRuntimePolicy& runtime_policy,
    VulkanExecutionPlan& plan) {
  switch (kind) {
    case VulkanExecutionPlanKind::LinearInputSource:
    case VulkanExecutionPlanKind::LinearWeightSource:
    case VulkanExecutionPlanKind::LinearBiasSource:
    case VulkanExecutionPlanKind::LinearPackedBias:
    case VulkanExecutionPlanKind::LinearPackedInput:
    case VulkanExecutionPlanKind::LinearPackedWeight:
      if (
          runtime_policy.linear_kernel_family ==
          VulkanLinearKernelFamily::PersistentPackedTexture) {
        plan.persistent = true;
      }
      break;
    case VulkanExecutionPlanKind::NormInput:
      if (
          runtime_policy.norm_kernel_family !=
          VulkanNormKernelFamily::TextureWidth) {
        plan.persistent = true;
      }
      break;
    case VulkanExecutionPlanKind::AttentionInput:
    case VulkanExecutionPlanKind::AttentionMaskInput:
    case VulkanExecutionPlanKind::AttentionCacheInput:
    case VulkanExecutionPlanKind::AttentionCacheAppendInput:
      if (
          runtime_policy.attention_kernel_family !=
          VulkanAttentionKernelFamily::TextureMath) {
        plan.persistent = true;
      }
      break;
    default:
      break;
  }
}

} // namespace

const char* execution_layout_name(const api::ExecutionLayout execution_layout) {
  return api::to_string(execution_layout);
}

const char* execution_plan_kind_name(const VulkanExecutionPlanKind kind) {
  return execution_plan_policy(kind).name;
}

const char* attention_mask_kind_name(const VulkanAttentionMaskKind kind) {
  switch (kind) {
    case VulkanAttentionMaskKind::None:
      return "None";
    case VulkanAttentionMaskKind::Additive:
      return "Additive";
    case VulkanAttentionMaskKind::Boolean:
      return "Boolean";
  }
  return "Unknown";
}

const char* attention_cache_mode_name(const VulkanAttentionCacheMode mode) {
  switch (mode) {
    case VulkanAttentionCacheMode::Disabled:
      return "Disabled";
    case VulkanAttentionCacheMode::Prefill:
      return "Prefill";
    case VulkanAttentionCacheMode::DecodeAppend:
      return "DecodeAppend";
  }
  return "Unknown";
}

const VulkanExecutionPlanPolicy& execution_plan_policy(
    const VulkanExecutionPlanKind kind) {
  return execution_plan_policies()[execution_plan_kind_index(kind)];
}

VulkanAttentionPolicy build_vulkan_attention_policy(
    const std::optional<Tensor>& attn_mask,
    const bool is_causal,
    const bool enable_gqa,
    const bool use_kv_cache,
    const bool cache_has_previous_state) {
  VulkanAttentionPolicy policy;
  policy.is_causal = is_causal;
  policy.enable_gqa = enable_gqa;
  policy.mask_kind =
      (!attn_mask || !attn_mask->defined())
      ? VulkanAttentionMaskKind::None
      : (attn_mask->scalar_type() == kBool ? VulkanAttentionMaskKind::Boolean
                                           : VulkanAttentionMaskKind::Additive);
  policy.cache_mode = !use_kv_cache
      ? VulkanAttentionCacheMode::Disabled
      : (cache_has_previous_state ? VulkanAttentionCacheMode::DecodeAppend
                                  : VulkanAttentionCacheMode::Prefill);
  policy.key_value_plan_kind =
      policy.cache_mode == VulkanAttentionCacheMode::Disabled
      ? VulkanExecutionPlanKind::AttentionInput
      : VulkanExecutionPlanKind::AttentionCacheInput;
  return policy;
}

VulkanPlanningRequest make_vulkan_attention_request(
    const VulkanAttentionPolicy& attention_policy,
    const VulkanTensorRole tensor_role) {
  const bool uses_cache =
      attention_policy.cache_mode != VulkanAttentionCacheMode::Disabled;
  const VulkanExecutionPhase execution_phase =
      attention_policy.cache_mode == VulkanAttentionCacheMode::Prefill
      ? VulkanExecutionPhase::Prefill
      : (attention_policy.cache_mode == VulkanAttentionCacheMode::DecodeAppend
             ? VulkanExecutionPhase::Decode
             : VulkanExecutionPhase::None);
  return make_vulkan_planning_request(
      uses_cache ? VulkanWorkloadClass::AttentionCache
                 : VulkanWorkloadClass::Attention,
      tensor_role,
      uses_cache ? VulkanModelDomain::LLM
                 : VulkanModelDomain::Generic,
      execution_phase);
}

VulkanPlanningRequest make_vulkan_execution_request(
    const VulkanExecutionPlanKind kind,
    const VulkanTensorRole tensor_role,
    const VulkanModelDomain model_domain,
    const VulkanExecutionPhase execution_phase) {
  return make_vulkan_planning_request(
      execution_plan_workload_class(kind),
      tensor_role,
      model_domain,
      execution_phase);
}

VulkanPlanningRequest make_vulkan_execution_request(
    const VulkanExecutionPlanKind kind,
    const VulkanModelDomain model_domain,
    const VulkanExecutionPhase execution_phase) {
  return make_vulkan_execution_request(
      kind, execution_plan_tensor_role(kind), model_domain, execution_phase);
}

VulkanExecutionPlan build_vulkan_execution_plan(
    const Tensor& tensor,
    const VulkanExecutionPlanKind kind) {
  const auto request = specialize_vulkan_planning_request_for_tensor(
      tensor, make_vulkan_execution_request(kind));
  return build_vulkan_execution_plan(tensor, kind, request);
}

VulkanExecutionPlan build_vulkan_execution_plan(
    const Tensor& tensor,
    const VulkanExecutionPlanKind kind,
    const VulkanPlanningRequest& request) {
  const auto& policy = execution_plan_policy(kind);
  const auto runtime_policy = build_vulkan_runtime_policy(request);
  const auto persistence_hints = build_vulkan_persistence_hints(request);
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
  plan.persistent = policy.persistent ||
      persistence_hints.prefer_persistent_contexts ||
      persistence_hints.prefer_persistent_weights;
  apply_runtime_family_overrides(kind, runtime_policy, plan);

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

Tensor prepare_vulkan_direct_buffer_execution_tensor(
    const Tensor& input,
    const VulkanExecutionPlanKind kind,
    const VulkanPlanningRequest& request) {
  const VulkanExecutionPlan plan =
      build_vulkan_execution_plan(input, kind, request);
  return prepare_vulkan_direct_buffer_execution_tensor(input, plan);
}

Tensor prepare_vulkan_execution_tensor(
    const Tensor& input,
    const VulkanExecutionPlanKind kind) {
  return execute_vulkan_execution_plan(
      input, build_vulkan_execution_plan(input, kind));
}

Tensor prepare_vulkan_execution_tensor(
    const Tensor& input,
    const VulkanExecutionPlanKind kind,
    const VulkanPlanningRequest& request) {
  return execute_vulkan_execution_plan(
      input, build_vulkan_execution_plan(input, kind, request));
}

std::optional<Tensor> prepare_optional_vulkan_execution_tensor(
    const std::optional<Tensor>& input,
    const VulkanExecutionPlanKind kind) {
  if (!input || !input->defined()) {
    return std::nullopt;
  }

  return prepare_vulkan_execution_tensor(*input, kind);
}

std::optional<Tensor> prepare_optional_vulkan_execution_tensor(
    const std::optional<Tensor>& input,
    const VulkanExecutionPlanKind kind,
    const VulkanPlanningRequest& request) {
  if (!input || !input->defined()) {
    return std::nullopt;
  }

  return prepare_vulkan_execution_tensor(*input, kind, request);
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
