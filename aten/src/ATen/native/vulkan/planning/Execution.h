#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/planning/Runtime.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

enum class VulkanExecutionPlanKind : uint8_t {
  Generic = 0u,
  TextureComputeInput,
  NormInput,
  AttentionInput,
  AttentionMaskInput,
  AttentionCacheInput,
  AttentionCacheAppendInput,
  ElementwiseInput,
  ElementwiseBufferInput,
  ReductionAllInput,
  ReductionDimInput,
  LinearInputSource,
  LinearWeightSource,
  LinearBiasSource,
  LinearPackedBias,
  LinearPackedInput,
  LinearPackedWeight,
  Conv2dWeightSource,
  Conv2dBiasSource,
  Conv2dRuntimeInput,
  Conv1dPrepackWeight,
  Conv1dPrepackBias,
  Conv1dRuntimeInput,
  Conv1dRuntimeWeight,
  Conv1dRuntimeBias,
  NumKinds,
};

enum class VulkanAttentionMaskKind : uint8_t {
  None = 0u,
  Additive,
  Boolean,
};

enum class VulkanAttentionCacheMode : uint8_t {
  Disabled = 0u,
  Prefill,
  DecodeAppend,
};

struct VulkanAttentionPolicy final {
  VulkanExecutionPlanKind query_plan_kind{
      VulkanExecutionPlanKind::AttentionInput};
  VulkanExecutionPlanKind key_value_plan_kind{
      VulkanExecutionPlanKind::AttentionInput};
  VulkanExecutionPlanKind mask_plan_kind{
      VulkanExecutionPlanKind::AttentionMaskInput};
  VulkanExecutionPlanKind cache_plan_kind{
      VulkanExecutionPlanKind::AttentionCacheInput};
  VulkanExecutionPlanKind cache_append_plan_kind{
      VulkanExecutionPlanKind::AttentionCacheAppendInput};
  VulkanAttentionMaskKind mask_kind{VulkanAttentionMaskKind::None};
  VulkanAttentionCacheMode cache_mode{VulkanAttentionCacheMode::Disabled};
  bool is_causal{false};
  bool enable_gqa{false};
};

enum class VulkanExecutionPolicyBufferRule : uint8_t {
  Never = 0u,
  PreferElementwiseBuffer,
  RequireElementwiseBuffer,
  PreferReductionBuffer,
};

enum class VulkanExecutionPolicyMemoryRule : uint8_t {
  Fixed = 0u,
  LinearInputSource,
};

struct VulkanExecutionPlanPolicy final {
  const char* name{"Generic"};
  api::ExecutionLayout execution_layout{api::ExecutionLayout::TEXTURE};
  api::GPUMemoryLayout memory_layout{
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED};
  api::StorageType storage_type{api::StorageType::TEXTURE_3D};
  VulkanExecutionPolicyBufferRule buffer_rule{
      VulkanExecutionPolicyBufferRule::Never};
  VulkanExecutionPolicyMemoryRule memory_rule{
      VulkanExecutionPolicyMemoryRule::Fixed};
  bool force_storage{true};
  bool force_storage_if_widen_bfloat16{false};
  bool widen_bfloat16{false};
  bool materialize_inference_matrix{false};
  bool persistent{false};
};

struct VulkanExecutionPlan final {
  VulkanExecutionPlanKind kind{VulkanExecutionPlanKind::Generic};
  api::ExecutionLayout execution_layout{api::ExecutionLayout::TEXTURE};
  api::GPUMemoryLayout memory_layout{
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED};
  api::StorageType storage_type{api::StorageType::TEXTURE_3D};
  bool force_storage{true};
  bool widen_bfloat16{false};
  bool materialize_inference_matrix{false};
  bool persistent{false};
};

const char* execution_layout_name(api::ExecutionLayout);

const char* execution_plan_kind_name(VulkanExecutionPlanKind);

const char* attention_mask_kind_name(VulkanAttentionMaskKind);

const char* attention_cache_mode_name(VulkanAttentionCacheMode);

const VulkanExecutionPlanPolicy& execution_plan_policy(VulkanExecutionPlanKind);

VulkanWorkloadClass execution_plan_workload_class(VulkanExecutionPlanKind);

VulkanAttentionPolicy build_vulkan_attention_policy(
    const std::optional<Tensor>& attn_mask,
    bool is_causal,
    bool enable_gqa,
    bool use_kv_cache,
    bool cache_has_previous_state);

VulkanPlanningRequest make_vulkan_execution_request(
    VulkanExecutionPlanKind,
    VulkanTensorRole,
    VulkanModelDomain model_domain = VulkanModelDomain::Generic,
    VulkanExecutionPhase execution_phase = VulkanExecutionPhase::None);

VulkanPlanningRequest make_vulkan_execution_request(
    VulkanExecutionPlanKind,
    VulkanModelDomain model_domain = VulkanModelDomain::Generic,
    VulkanExecutionPhase execution_phase = VulkanExecutionPhase::None);

VulkanExecutionPlan build_vulkan_execution_plan(
    const Tensor&,
    VulkanExecutionPlanKind);

VulkanExecutionPlan build_vulkan_execution_plan(
    const Tensor&,
    VulkanExecutionPlanKind,
    const VulkanPlanningRequest&);

Tensor execute_vulkan_execution_plan(
    const Tensor&,
    const VulkanExecutionPlan&);

Tensor prepare_vulkan_direct_buffer_execution_tensor(
    const Tensor&,
    const VulkanExecutionPlan&);

Tensor prepare_vulkan_direct_buffer_execution_tensor(
    const Tensor&,
    VulkanExecutionPlanKind);

Tensor prepare_vulkan_direct_buffer_execution_tensor(
    const Tensor&,
    VulkanExecutionPlanKind,
    const VulkanPlanningRequest&);

Tensor prepare_vulkan_execution_tensor(
    const Tensor&,
    VulkanExecutionPlanKind);

Tensor prepare_vulkan_execution_tensor(
    const Tensor&,
    VulkanExecutionPlanKind,
    const VulkanPlanningRequest&);

std::optional<Tensor> prepare_optional_vulkan_execution_tensor(
    const std::optional<Tensor>&,
    VulkanExecutionPlanKind);

std::optional<Tensor> prepare_optional_vulkan_execution_tensor(
    const std::optional<Tensor>&,
    VulkanExecutionPlanKind,
    const VulkanPlanningRequest&);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
