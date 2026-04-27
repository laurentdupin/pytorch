#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/LayoutTransitions.h>
#include <ATen/native/vulkan/planning/DevicePolicy.h>
#include <ATen/native/vulkan/planning/ModelLanePolicy.h>
#include <ATen/native/vulkan/planning/Runtime.h>

#include <cstdint>
#include <string>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

enum class VulkanRouteKind : uint8_t {
  VulkanTextureKernel = 0u,
  VulkanBufferDirectKernel,
  VulkanBufferViewKernel,
  VulkanCompiledReplay,
  VulkanMaterializeThenRun,
  SmallCpuFallback,
  HardFail,
  NotSupported,
};

enum class VulkanRouteRejectReason : uint8_t {
  None = 0u,
  UnsupportedDType,
  UnsupportedRank,
  UnsupportedLayout,
  MetadataViewInvalid,
  RequiresLargeCpuFallback,
  KnownBadConv3x3Stride1Pad1,
  KnownBadLargeBufferConv3x3,
  KnownBadLargePointwiseConv,
  KnownBadDiffusion4dSdpa,
  KnownBadGenericSdpa,
  KnownBadSdpaMaskOrCausal,
  KnownBadSdpaExplicitScale,
  KnownBadBufferLastDimSoftmax,
  KnownBadGenericTiledDiffusionLinear,
  DeviceQuirkDenied,
  ReplayViewStale,
  ReplayOutputAliasUnsafe,
  OutputAliasUnsafe,
};

struct VulkanRouteDecision final {
  VulkanRouteKind kind{VulkanRouteKind::VulkanTextureKernel};
  VulkanRouteRejectReason reject_reason{VulkanRouteRejectReason::None};
  VulkanRuntimePolicy runtime_policy{};
  ::at::native::vulkan::ops::VulkanLayoutTarget input_target{};
  ::at::native::vulkan::ops::VulkanLayoutTarget output_target{};
  VulkanModelLane lane{VulkanModelLane::Generic};
  std::string kernel_family;
  std::string telemetry_label;
  std::string shape_summary;
  std::string device_summary;
  bool hard_fail{false};
};

const char* route_kind_name(VulkanRouteKind kind);
const char* route_reject_reason_name(VulkanRouteRejectReason reason);

VulkanRouteDecision make_hard_fail_route(
    const char* op_name,
    VulkanRouteRejectReason reason,
    const std::string& shape_summary,
    const VulkanPlanningRequest& request,
    const VulkanDevicePolicy& device_policy);

void log_route_decision(
    const char* op_name,
    const VulkanRouteDecision& decision);

std::string format_hard_fail(
    const char* op_name,
    const VulkanRouteDecision& decision);

VulkanRouteDecision select_softmax_route(
    const Tensor& input,
    int64_t dim,
    const VulkanPlanningRequest& request,
    const VulkanDevicePolicy& device_policy);

VulkanRouteDecision select_conv2d_route(
    IntArrayRef input_sizes,
    IntArrayRef weight_sizes,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    ScalarType dtype,
    bool input_requires_grad,
    const VulkanPlanningRequest& request,
    const VulkanDevicePolicy& device_policy);

VulkanRouteDecision select_sdpa_route(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa,
    const VulkanPlanningRequest& request,
    const VulkanDevicePolicy& device_policy);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
