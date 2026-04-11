#include <ATen/native/vulkan/planning/Capabilities.h>
#include <ATen/native/vulkan/planning/Execution.h>
#include <ATen/native/vulkan/planning/Persistence.h>
#include <ATen/native/vulkan/planning\Request.h>
#include <ATen/native/vulkan/planning/Runtime.h>
#include <ATen/native/vulkan/planning/Scheduler.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

using namespace api::utils;

constexpr size_t kVulkanWorkloadClassCount =
    static_cast<size_t>(VulkanWorkloadClass::LLMDecode) + 1u;

size_t workload_class_index(const VulkanWorkloadClass workload_class) {
  const size_t idx = static_cast<size_t>(workload_class);
  TORCH_INTERNAL_ASSERT(
      idx < kVulkanWorkloadClassCount,
      "Invalid VulkanWorkloadClass");
  return idx;
}

VulkanLinearKernelFamily select_linear_kernel_family(
    const VulkanPlanningRequest& request,
    const VulkanRuntimeCapabilityProfile& capabilities,
    const VulkanPersistenceHints& persistence_hints) {
  if (
      request.model_domain == VulkanModelDomain::LLM &&
      request.execution_phase != VulkanExecutionPhase::None) {
    return capabilities.has_unified_memory
        ? VulkanLinearKernelFamily::UnifiedBufferView
        : VulkanLinearKernelFamily::PersistentPackedTexture;
  }
  if (
      request.workload_class == VulkanWorkloadClass::VisionBackbone ||
      (request.model_domain == VulkanModelDomain::Vision &&
       request.execution_phase == VulkanExecutionPhase::Backbone)) {
    // Vision backbone execution now prefers the buffer-native compatibility
    // path. Performance tuning can continue independently, but the canonical
    // planner family for ViT-style blocks should be buffer based.
    return VulkanLinearKernelFamily::UnifiedBufferView;
  }
  if (
      persistence_hints.prefer_persistent_weights ||
      persistence_hints.prefer_persistent_contexts) {
    return VulkanLinearKernelFamily::PersistentPackedTexture;
  }
  return capabilities.has_unified_memory
      ? VulkanLinearKernelFamily::UnifiedBufferView
      : VulkanLinearKernelFamily::TexturePacked;
}

VulkanNormKernelFamily select_norm_kernel_family(
    const VulkanPlanningRequest& request,
    const VulkanRuntimeCapabilityProfile& capabilities) {
  (void)capabilities;
  if (
      request.workload_class == VulkanWorkloadClass::VisionBackbone ||
      (request.model_domain == VulkanModelDomain::Vision &&
       request.execution_phase == VulkanExecutionPhase::Backbone)) {
    return VulkanNormKernelFamily::UnifiedBufferView;
  }
  return VulkanNormKernelFamily::TextureWidth;
}

VulkanAttentionKernelFamily select_attention_kernel_family(
    const VulkanPlanningRequest& request,
    const VulkanSchedulerDecision& scheduler_decision) {
  if (scheduler_decision.boundary_plan.has_value()) {
    return VulkanAttentionKernelFamily::SplitCoordinator;
  }
  if (
      request.workload_class == VulkanWorkloadClass::AttentionCache ||
      request.tensor_role == VulkanTensorRole::Cache) {
    return VulkanAttentionKernelFamily::CacheAwareTexture;
  }
  if (
      request.workload_class == VulkanWorkloadClass::VisionBackbone ||
      (request.model_domain == VulkanModelDomain::Vision &&
       request.execution_phase == VulkanExecutionPhase::Backbone)) {
    return VulkanAttentionKernelFamily::BufferMath;
  }
  return VulkanAttentionKernelFamily::TextureMath;
}

std::optional<VulkanExecutionProgramPlanningDesc> select_execution_program_plan(
    const VulkanPlanningRequest& request,
    const VulkanSchedulerDecision& scheduler_decision,
    const VulkanAttentionKernelFamily attention_kernel_family) {
  if (
      scheduler_decision.boundary_plan.has_value() &&
      scheduler_decision.boundary_plan->kind ==
          VulkanBoundaryKind::LLMLinearAttentionSplit &&
      request.workload_class == VulkanWorkloadClass::LLMDecode) {
    return VulkanExecutionProgramPlanningDesc{
        VulkanExecutionProgramKind::GatedDeltaSplit, true};
  }

  if (
      request.workload_class == VulkanWorkloadClass::AttentionCache &&
      (scheduler_decision.kv_cache_plan.has_value() ||
       scheduler_decision.scratch_arena_plan.has_value() ||
       attention_kernel_family != VulkanAttentionKernelFamily::TextureMath)) {
    return VulkanExecutionProgramPlanningDesc{
        VulkanExecutionProgramKind::AttentionRuntime, true};
  }

  if (
      request.workload_class == VulkanWorkloadClass::VisionBackbone &&
      request.model_domain == VulkanModelDomain::Vision &&
      request.execution_phase == VulkanExecutionPhase::Backbone) {
    return VulkanExecutionProgramPlanningDesc{
        VulkanExecutionProgramKind::VisionBackbone, true};
  }

  return std::nullopt;
}

const std::string& runtime_policy_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_RUNTIME_POLICY_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool runtime_policy_logging_enabled() {
  return !runtime_policy_log_path().empty();
}

struct RuntimePolicyLogState final {
  std::array<std::atomic<uint64_t>, kVulkanWorkloadClassCount> builds{};
  std::array<std::atomic<uint64_t>, kVulkanWorkloadClassCount> vulkan_only{};
  std::array<std::atomic<uint64_t>, kVulkanWorkloadClassCount> prefer_split{};
  std::array<std::atomic<uint64_t>, kVulkanWorkloadClassCount>
      prefer_cpu_fallback{};
  std::atomic<uint64_t> has_unified_memory{0u};
  std::atomic<uint64_t> has_timestamps{0u};
  std::atomic<uint64_t> has_shader_bfloat16{0u};
  std::atomic<uint64_t> has_shader_int8{0u};
  std::atomic<uint64_t> has_storage_buffer_8bit{0u};
  std::atomic<uint64_t> supports_int8_buffer_arithmetic{0u};
  std::atomic<uint64_t> num_compute_queues{0u};
  std::atomic<uint64_t> max_compute_workgroup_invocations{0u};
  std::atomic<uint64_t> max_compute_shared_memory_size{0u};

  ~RuntimePolicyLogState() {
    if (!runtime_policy_logging_enabled()) {
      return;
    }

    std::ofstream out(runtime_policy_log_path(), std::ios::app);
    out << "runtime_capabilities has_unified_memory="
        << has_unified_memory.load(std::memory_order_relaxed)
        << " has_timestamps=" << has_timestamps.load(std::memory_order_relaxed)
        << " has_shader_bfloat16="
        << has_shader_bfloat16.load(std::memory_order_relaxed)
        << " has_shader_int8=" << has_shader_int8.load(std::memory_order_relaxed)
        << " has_storage_buffer_8bit="
        << has_storage_buffer_8bit.load(std::memory_order_relaxed)
        << " supports_int8_buffer_arithmetic="
        << supports_int8_buffer_arithmetic.load(std::memory_order_relaxed)
        << " num_compute_queues="
        << num_compute_queues.load(std::memory_order_relaxed)
        << " max_compute_workgroup_invocations="
        << max_compute_workgroup_invocations.load(std::memory_order_relaxed)
        << " max_compute_shared_memory_size="
        << max_compute_shared_memory_size.load(std::memory_order_relaxed)
        << '\n';

    for (const auto idx : c10::irange(kVulkanWorkloadClassCount)) {
      const auto build_count = builds[idx].load(std::memory_order_relaxed);
      const auto vulkan_only_count =
          vulkan_only[idx].load(std::memory_order_relaxed);
      const auto prefer_split_count =
          prefer_split[idx].load(std::memory_order_relaxed);
      const auto prefer_cpu_fallback_count =
          prefer_cpu_fallback[idx].load(std::memory_order_relaxed);
      if (
          build_count == 0u && vulkan_only_count == 0u &&
          prefer_split_count == 0u && prefer_cpu_fallback_count == 0u) {
        continue;
      }

      const auto workload_class =
          static_cast<VulkanWorkloadClass>(safe_downcast<uint8_t>(idx));
      out << "runtime_policy workload=" << workload_class_name(workload_class)
          << " builds=" << build_count
          << " vulkan_only=" << vulkan_only_count
          << " prefer_split=" << prefer_split_count
          << " prefer_cpu_fallback=" << prefer_cpu_fallback_count << '\n';
    }
  }
};

RuntimePolicyLogState& runtime_policy_log_state() {
  static RuntimePolicyLogState state;
  return state;
}

void log_runtime_policy_build(const VulkanRuntimePolicy& policy) {
  if (!runtime_policy_logging_enabled()) {
    return;
  }

  auto& state = runtime_policy_log_state();
  const auto capabilities = query_vulkan_runtime_capability_profile();
  state.has_unified_memory.store(
      capabilities.has_unified_memory ? 1u : 0u, std::memory_order_relaxed);
  state.has_timestamps.store(
      capabilities.has_timestamps ? 1u : 0u, std::memory_order_relaxed);
  state.has_shader_bfloat16.store(
      capabilities.has_shader_bfloat16 ? 1u : 0u,
      std::memory_order_relaxed);
  state.has_shader_int8.store(
      capabilities.has_shader_int8 ? 1u : 0u, std::memory_order_relaxed);
  state.has_storage_buffer_8bit.store(
      capabilities.has_storage_buffer_8bit ? 1u : 0u,
      std::memory_order_relaxed);
  state.supports_int8_buffer_arithmetic.store(
      capabilities.supports_int8_buffer_arithmetic ? 1u : 0u,
      std::memory_order_relaxed);
  state.num_compute_queues.store(
      capabilities.num_compute_queues, std::memory_order_relaxed);
  state.max_compute_workgroup_invocations.store(
      capabilities.max_compute_workgroup_invocations,
      std::memory_order_relaxed);
  state.max_compute_shared_memory_size.store(
      capabilities.max_compute_shared_memory_size,
      std::memory_order_relaxed);
  const size_t idx = workload_class_index(policy.request.workload_class);
  state.builds[idx].fetch_add(1u, std::memory_order_relaxed);
  switch (policy.backend_route) {
    case VulkanBackendRoute::Vulkan:
      state.vulkan_only[idx].fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanBackendRoute::Split:
      state.prefer_split[idx].fetch_add(1u, std::memory_order_relaxed);
      break;
    case VulkanBackendRoute::CPU:
      state.prefer_cpu_fallback[idx].fetch_add(1u, std::memory_order_relaxed);
      break;
  }

  std::ofstream out(runtime_policy_log_path(), std::ios::app);
  out << "runtime_capabilities has_unified_memory="
      << (capabilities.has_unified_memory ? 1u : 0u)
      << " has_timestamps=" << (capabilities.has_timestamps ? 1u : 0u)
      << " has_shader_bfloat16=" << (capabilities.has_shader_bfloat16 ? 1u : 0u)
      << " has_shader_int8=" << (capabilities.has_shader_int8 ? 1u : 0u)
      << " has_storage_buffer_8bit="
      << (capabilities.has_storage_buffer_8bit ? 1u : 0u)
      << " supports_int8_buffer_arithmetic="
      << (capabilities.supports_int8_buffer_arithmetic ? 1u : 0u)
      << " num_compute_queues=" << capabilities.num_compute_queues
      << " max_compute_workgroup_invocations="
      << capabilities.max_compute_workgroup_invocations
      << " max_compute_shared_memory_size="
      << capabilities.max_compute_shared_memory_size << '\n';
  out << "runtime_policy workload="
      << workload_class_name(policy.request.workload_class)
      << " model_domain=" << model_domain_name(policy.request.model_domain)
      << " execution_phase="
      << execution_phase_name(policy.request.execution_phase)
      << " tensor_role=" << tensor_role_name(policy.request.tensor_role)
      << " backend_route=" << backend_route_name(policy.backend_route)
      << " linear_kernel_family="
      << linear_kernel_family_name(policy.linear_kernel_family)
      << " norm_kernel_family="
      << norm_kernel_family_name(policy.norm_kernel_family)
      << " attention_kernel_family="
      << attention_kernel_family_name(policy.attention_kernel_family)
      << " has_execution_program_plan="
      << (policy.execution_program_plan.has_value() ? 1u : 0u);
  if (policy.execution_program_plan.has_value()) {
    out << " execution_program_kind="
        << execution_program_kind_name(policy.execution_program_plan->kind)
        << " execution_program_persistent="
        << (policy.execution_program_plan->persistent ? 1u : 0u);
  }
  out << " has_boundary_plan="
      << (policy.boundary_plan.has_value() ? 1u : 0u);
  if (policy.boundary_plan.has_value()) {
    out << " boundary_kind="
        << boundary_kind_name(policy.boundary_plan->kind)
        << " boundary_input_layout="
        << boundary_transfer_layout_name(
               policy.boundary_plan->input_transfer_layout)
        << " boundary_output_layout="
        << boundary_transfer_layout_name(
               policy.boundary_plan->output_transfer_layout)
        << " boundary_backend_owned_execution="
        << (policy.boundary_plan->prefer_backend_owned_execution ? 1u : 0u)
        << " boundary_requires_scratch="
        << (policy.boundary_plan->requires_scratch_arena ? 1u : 0u)
        << " boundary_preferred_cpu_threads="
        << policy.boundary_plan->preferred_cpu_threads;
  }
  out
      << " has_kv_cache_plan=" << (policy.kv_cache_plan.has_value() ? 1u : 0u)
      << " has_scratch_arena_plan="
      << (policy.scratch_arena_plan.has_value() ? 1u : 0u)
      << " inferred_from_label="
      << (policy.request.inferred_from_label ? 1u : 0u) << '\n';
}

} // namespace

const char* linear_kernel_family_name(
    const VulkanLinearKernelFamily family) {
  switch (family) {
    case VulkanLinearKernelFamily::TexturePacked:
      return "TexturePacked";
    case VulkanLinearKernelFamily::UnifiedBufferView:
      return "UnifiedBufferView";
    case VulkanLinearKernelFamily::PersistentPackedTexture:
      return "PersistentPackedTexture";
  }
  return "TexturePacked";
}

const char* norm_kernel_family_name(const VulkanNormKernelFamily family) {
  switch (family) {
    case VulkanNormKernelFamily::TextureWidth:
      return "TextureWidth";
    case VulkanNormKernelFamily::SharedMemoryWidth:
      return "SharedMemoryWidth";
    case VulkanNormKernelFamily::UnifiedBufferView:
      return "UnifiedBufferView";
  }
  return "TextureWidth";
}

const char* attention_kernel_family_name(
    const VulkanAttentionKernelFamily family) {
  switch (family) {
    case VulkanAttentionKernelFamily::TextureMath:
      return "TextureMath";
    case VulkanAttentionKernelFamily::BufferMath:
      return "BufferMath";
    case VulkanAttentionKernelFamily::CacheAwareTexture:
      return "CacheAwareTexture";
    case VulkanAttentionKernelFamily::SplitCoordinator:
      return "SplitCoordinator";
  }
  return "TextureMath";
}

const char* execution_program_kind_name(
    const VulkanExecutionProgramKind kind) {
  switch (kind) {
    case VulkanExecutionProgramKind::AttentionRuntime:
      return "AttentionRuntime";
    case VulkanExecutionProgramKind::GatedDeltaSplit:
      return "GatedDeltaSplit";
    case VulkanExecutionProgramKind::VisionBackbone:
      return "VisionBackbone";
  }
  return "AttentionRuntime";
}

VulkanWorkloadClass execution_plan_workload_class(
    const VulkanExecutionPlanKind kind) {
  VulkanWorkloadClass workload_class = VulkanWorkloadClass::Generic;
  switch (kind) {
    case VulkanExecutionPlanKind::Generic:
    case VulkanExecutionPlanKind::TextureComputeInput:
      workload_class = VulkanWorkloadClass::Generic;
      break;
    case VulkanExecutionPlanKind::NormInput:
      workload_class = VulkanWorkloadClass::Norm;
      break;
    case VulkanExecutionPlanKind::AttentionInput:
    case VulkanExecutionPlanKind::AttentionMaskInput:
      workload_class = VulkanWorkloadClass::Attention;
      break;
    case VulkanExecutionPlanKind::AttentionCacheInput:
    case VulkanExecutionPlanKind::AttentionCacheAppendInput:
      workload_class = VulkanWorkloadClass::AttentionCache;
      break;
    case VulkanExecutionPlanKind::ElementwiseInput:
    case VulkanExecutionPlanKind::ElementwiseBufferInput:
      workload_class = VulkanWorkloadClass::Elementwise;
      break;
    case VulkanExecutionPlanKind::ReductionAllInput:
    case VulkanExecutionPlanKind::ReductionDimInput:
      workload_class = VulkanWorkloadClass::Reduction;
      break;
    case VulkanExecutionPlanKind::LinearInputSource:
    case VulkanExecutionPlanKind::LinearWeightSource:
    case VulkanExecutionPlanKind::LinearBiasSource:
    case VulkanExecutionPlanKind::LinearPackedBias:
    case VulkanExecutionPlanKind::LinearPackedInput:
    case VulkanExecutionPlanKind::LinearPackedWeight:
      workload_class = VulkanWorkloadClass::LinearMatmul;
      break;
    case VulkanExecutionPlanKind::Conv2dWeightSource:
    case VulkanExecutionPlanKind::Conv2dBiasSource:
    case VulkanExecutionPlanKind::Conv2dRuntimeInput:
    case VulkanExecutionPlanKind::Conv1dPrepackWeight:
    case VulkanExecutionPlanKind::Conv1dPrepackBias:
    case VulkanExecutionPlanKind::Conv1dRuntimeInput:
    case VulkanExecutionPlanKind::Conv1dRuntimeWeight:
    case VulkanExecutionPlanKind::Conv1dRuntimeBias:
      workload_class = VulkanWorkloadClass::Convolution;
      break;
    case VulkanExecutionPlanKind::NumKinds:
      workload_class = VulkanWorkloadClass::Generic;
      break;
  }
  return workload_class;
}

VulkanRuntimePolicy build_vulkan_runtime_policy(
    const VulkanPlanningRequest& planning_request) {
  VulkanRuntimePolicy policy;
  const auto inferred_request = infer_vulkan_planning_request(planning_request);
  policy.request = inferred_request;

  const auto capabilities = query_vulkan_runtime_capability_profile();
  const auto persistence_hints = build_vulkan_persistence_hints(inferred_request);
  const auto scheduler_decision =
      build_vulkan_scheduler_decision(inferred_request, capabilities);

  policy.backend_route = scheduler_decision.backend_route;
  policy.boundary_plan = scheduler_decision.boundary_plan;
  policy.kv_cache_plan = scheduler_decision.kv_cache_plan;
  policy.scratch_arena_plan = scheduler_decision.scratch_arena_plan;
  policy.linear_kernel_family = select_linear_kernel_family(
      inferred_request, capabilities, persistence_hints);
  policy.norm_kernel_family =
      select_norm_kernel_family(inferred_request, capabilities);
  policy.attention_kernel_family =
      select_attention_kernel_family(inferred_request, scheduler_decision);
  policy.execution_program_plan = select_execution_program_plan(
      inferred_request,
      scheduler_decision,
      policy.attention_kernel_family);

  log_runtime_policy_build(policy);
  return policy;
}

VulkanRuntimePolicy build_vulkan_runtime_policy(
    const VulkanWorkloadClass workload_class) {
  return build_vulkan_runtime_policy(make_vulkan_planning_request(workload_class));
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
