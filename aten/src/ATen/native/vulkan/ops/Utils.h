#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/PackedWeight.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace utils {

struct LogicalBufferMetadata final {
  api::utils::uvec4 logical_sizes;
  api::utils::uvec4 logical_strides;
  api::utils::uvec4 physical_strides;
  api::utils::uvec4 info;
};

enum class VulkanExecutionPlanKind : uint8_t {
  Generic = 0u,
  TextureComputeInput,
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

VulkanAttentionPolicy build_vulkan_attention_policy(
    const std::optional<Tensor>& attn_mask,
    bool is_causal,
    bool enable_gqa,
    bool use_kv_cache,
    bool cache_has_previous_state);

bool uses_buffer_execution(const vTensor&);

bool uses_texture_execution(const vTensor&);

Tensor nchw_to_nc4hw(const Tensor&);

Tensor create_staging_tensor(const vTensor&);

Tensor nc4hw_to_nchw(const Tensor&, IntArrayRef);

std::optional<Tensor> normalized_optional_tensor(const std::optional<Tensor>&);

bool same_optional_tensor(
    const std::optional<Tensor>&,
    const std::optional<Tensor>&);

int64_t tensor_version_or_zero(const Tensor&);

bool has_inference_tensor(const Tensor&, const std::optional<Tensor>&);

bool supports_buffer_view_fast_path(const vTensor&);

bool supports_buffer_elementwise_compute(const vTensor&);

bool supports_buffer_reduction_compute(const vTensor&);

bool scalar_fits_vulkan_int32(const Scalar&);

int32_t scalar_to_vulkan_int32(const Scalar&);

bool last_dim_is_width_aligned(const Tensor&);

bool supports_native_integral_buffer_compute_dtype(api::ScalarType);

bool supports_native_integral_buffer_compute(const Tensor&);

bool supports_native_bool_buffer_compute(const Tensor&);

bool can_make_buffer_metadata_view(
    const vTensor&,
    IntArrayRef sizes,
    IntArrayRef logical_strides,
    IntArrayRef physical_strides,
    int64_t storage_offset);

Tensor make_buffer_metadata_view(
    const Tensor&,
    IntArrayRef sizes,
    IntArrayRef logical_strides,
    IntArrayRef physical_strides,
    int64_t storage_offset);

LogicalBufferMetadata make_buffer_compute_metadata(const vTensor&);

api::UniformParamsBuffer make_buffer_compute_metadata_ubo(
    api::Context* const,
    const vTensor&);

vTensor materialize_to_contiguous_buffer(
    const vTensor&,
    api::GPUMemoryLayout memory_layout =
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);

Tensor ensure_buffer_storage(
    const Tensor&,
    api::GPUMemoryLayout memory_layout =
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);

Tensor ensure_texture_storage(
    const Tensor&,
    api::GPUMemoryLayout memory_layout =
        api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
    api::StorageType storage_type = api::StorageType::TEXTURE_3D);

Tensor upcast_bfloat16_buffer_to_float(const Tensor&);

Tensor cast_vulkan_tensor_dtype(const Tensor&, ScalarType);

Tensor mark_tensor_execution(
    const Tensor&,
    api::ExecutionLayout,
    bool persistent = false);

VulkanExecutionPlan build_vulkan_execution_plan(
    const Tensor&,
    VulkanExecutionPlanKind);

Tensor execute_vulkan_execution_plan(
    const Tensor&,
    const VulkanExecutionPlan&);

Tensor prepare_vulkan_direct_buffer_execution_tensor(
    const Tensor&,
    const VulkanExecutionPlan&);

Tensor prepare_vulkan_direct_buffer_execution_tensor(
    const Tensor&,
    VulkanExecutionPlanKind);

Tensor prepare_vulkan_execution_tensor(
    const Tensor&,
    VulkanExecutionPlanKind);

std::optional<Tensor> prepare_optional_vulkan_execution_tensor(
    const std::optional<Tensor>&,
    VulkanExecutionPlanKind);

PackedWeightHandle make_packed_weight_handle(
    Tensor,
    Tensor,
    std::vector<int64_t>,
    PackedWeightKind,
    bool bias_defined,
    bool quantized = false,
    PackedWeightResidencyClass residency_class =
        PackedWeightResidencyClass::PersistentInference);

std::optional<PackedWeightHandle> lookup_packed_weight_handle(
    const Tensor& source_weight,
    const std::optional<Tensor>& source_bias,
    IntArrayRef logical_weight_sizes,
    PackedWeightKind kind,
    bool quantized = false,
    uint64_t options_key = 0u);

void store_packed_weight_handle(
    const Tensor& source_weight,
    const std::optional<Tensor>& source_bias,
    IntArrayRef logical_weight_sizes,
    PackedWeightKind kind,
    const PackedWeightHandle& handle,
    bool quantized = false,
    uint64_t options_key = 0u);

void copy_buffer_to_buffer(
    api::Context* const context,
    api::StorageBuffer& src,
    api::StorageBuffer& dst,
    VkFence fence_handle);

void copy_buffer_to_vtensor(
    api::VulkanBuffer&,
    vTensor&,
    api::PipelineBarrier&);

void copy_vtensor_to_buffer(
    vTensor&,
    api::VulkanBuffer&,
    api::PipelineBarrier&,
    const VkFence fence_handle = VK_NULL_HANDLE);

inline int64_t normalize(const int64_t dimension, const int64_t n) {
  return (dimension % n + n) % n;
}

void pack_buffer_to_vtensor(
    api::VulkanBuffer&,
    vTensor&,
    api::PipelineBarrier&);

void pack_staging_to_vtensor(api::VulkanBuffer&, vTensor&);

bool pack_vtensor_to_staging(
    vTensor&,
    api::VulkanBuffer&,
    const VkFence fence_handle = VK_NULL_HANDLE);

// Broadcasting Utils
void is_broadcastable(const Tensor& input1, const Tensor& input2);
std::vector<int64_t> broadcast_size(const Tensor& t1, const Tensor& t2);

// This function returns the value of the underlying texel at pos of the given
// tensor. It is useful for debugging and unit test at which we want to verify
// the actual tensor layout. This function is very slow as it involves a fench
// to extract just one value.
api::utils::vec4 extract_texel(
    const Tensor& tensor,
    const api::utils::ivec3& pos);

inline api::utils::ivec2 make_ivec2(
    const IntArrayRef ints,
    bool reverse = false) {
  VK_CHECK_COND(ints.size() == 2);
  if (reverse) {
    return {
        api::utils::safe_downcast<int32_t>(ints[1]),
        api::utils::safe_downcast<int32_t>(ints[0])};
  } else {
    return {
        api::utils::safe_downcast<int32_t>(ints[0]),
        api::utils::safe_downcast<int32_t>(ints[1])};
  }
}

inline api::utils::ivec4 make_ivec4(
    const IntArrayRef ints,
    bool reverse = false) {
  VK_CHECK_COND(ints.size() == 4);
  if (reverse) {
    return {
        api::utils::safe_downcast<int32_t>(ints[3]),
        api::utils::safe_downcast<int32_t>(ints[2]),
        api::utils::safe_downcast<int32_t>(ints[1]),
        api::utils::safe_downcast<int32_t>(ints[0]),
    };
  } else {
    return {
        api::utils::safe_downcast<int32_t>(ints[0]),
        api::utils::safe_downcast<int32_t>(ints[1]),
        api::utils::safe_downcast<int32_t>(ints[2]),
        api::utils::safe_downcast<int32_t>(ints[3]),
    };
  }
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
