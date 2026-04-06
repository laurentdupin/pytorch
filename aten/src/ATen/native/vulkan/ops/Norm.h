#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/planning/Runtime.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

struct FusedNormWidthSpec final {
  const char* allocation_scope;
  const char* shader_name;
  const char* op_hit_name;
  bool has_bias;
};

bool supports_fused_norm_last_dim(
    const at::Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    bool require_bias);

void maybe_synchronize_after_norm();

Tensor fused_norm_width_impl(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps,
    const FusedNormWidthSpec& spec);

void log_norm_kernel_family_choice(
    const utils::VulkanRuntimePolicy& runtime_policy);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
