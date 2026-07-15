#include <ATen/native/vulkan/planning/RoutePolicy.h>

#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/Diagnostics.h>
#include <ATen/native/vulkan/ops/TensorState.h>
#include <ATen/native/vulkan/planning/ExecutionContracts.h>

#include <cstdlib>
#include <cmath>
#include <fstream>
#include <mutex>
#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

std::string route_log_path() {
  const char* env = std::getenv("PYTORCH_VULKAN_ROUTE_LOG");
  return env ? std::string(env) : std::string();
}

bool route_logging_enabled() {
  return !route_log_path().empty();
}

std::mutex& route_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::string softmax_shape_summary(const Tensor& input, const int64_t dim) {
  std::ostringstream out;
  out << "rank=" << input.dim() << " dim=" << dim << " sizes="
      << input.sizes();
  if (input.is_vulkan()) {
    out << " state={" << describe_tensor_state(input) << "}";
  }
  return out.str();
}

std::string conv2d_shape_summary(
    IntArrayRef input_sizes,
    IntArrayRef weight_sizes,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    const int64_t groups,
    const ScalarType dtype,
    const bool input_requires_grad) {
  std::ostringstream out;
  out << "input=" << input_sizes << " weight=" << weight_sizes
      << " stride=" << stride << " padding=" << padding
      << " dilation=" << dilation << " groups=" << groups
      << " dtype=" << dtype
      << " input_requires_grad=" << (input_requires_grad ? 1 : 0);
  return out.str();
}

std::string sdpa_shape_summary(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa) {
  std::ostringstream out;
  out << "query=" << query.sizes()
      << " key=" << key.sizes()
      << " value=" << value.sizes()
      << " dtype=" << query.scalar_type()
      << " mask=" << (attn_mask && attn_mask->defined() ? 1 : 0)
      << " dropout=" << dropout_p
      << " causal=" << (is_causal ? 1 : 0)
      << " scale=";
  if (scale.has_value()) {
    out << *scale;
  } else {
    out << "default";
  }
  out << " gqa=" << (enable_gqa ? 1 : 0);
  return out.str();
}

std::string hard_fail_detail(const VulkanRouteDecision& decision) {
  std::ostringstream out;
  out << "lane=" << model_lane_name(decision.lane);
  if (!decision.shape_summary.empty()) {
    out << " shape={" << decision.shape_summary << "}";
  }
  if (!decision.device_summary.empty()) {
    out << " device={" << decision.device_summary << "}";
  }
  return out.str();
}

} // namespace

const char* route_kind_name(const VulkanRouteKind kind) {
  switch (kind) {
    case VulkanRouteKind::VulkanTextureKernel:
      return "VulkanTextureKernel";
    case VulkanRouteKind::VulkanBufferDirectKernel:
      return "VulkanBufferDirectKernel";
    case VulkanRouteKind::VulkanBufferViewKernel:
      return "VulkanBufferViewKernel";
    case VulkanRouteKind::VulkanCompiledReplay:
      return "VulkanCompiledReplay";
    case VulkanRouteKind::VulkanMaterializeThenRun:
      return "VulkanMaterializeThenRun";
    case VulkanRouteKind::SmallCpuFallback:
      return "SmallCpuFallback";
    case VulkanRouteKind::HardFail:
      return "HardFail";
    case VulkanRouteKind::NotSupported:
      return "NotSupported";
  }
  return "NotSupported";
}

const char* route_reject_reason_name(
    const VulkanRouteRejectReason reason) {
  switch (reason) {
    case VulkanRouteRejectReason::None:
      return "None";
    case VulkanRouteRejectReason::UnsupportedDType:
      return "UnsupportedDType";
    case VulkanRouteRejectReason::UnsupportedRank:
      return "UnsupportedRank";
    case VulkanRouteRejectReason::UnsupportedLayout:
      return "UnsupportedLayout";
    case VulkanRouteRejectReason::MetadataViewInvalid:
      return "MetadataViewInvalid";
    case VulkanRouteRejectReason::RequiresLargeCpuFallback:
      return "RequiresLargeCpuFallback";
    case VulkanRouteRejectReason::KnownBadConv3x3Stride1Pad1:
      return "KnownBadConv3x3Stride1Pad1";
    case VulkanRouteRejectReason::KnownBadLargeBufferConv3x3:
      return "KnownBadLargeBufferConv3x3";
    case VulkanRouteRejectReason::KnownBadLargePointwiseConv:
      return "KnownBadLargePointwiseConv";
    case VulkanRouteRejectReason::KnownBadDiffusion4dSdpa:
      return "KnownBadDiffusion4dSdpa";
    case VulkanRouteRejectReason::KnownBadGenericSdpa:
      return "KnownBadGenericSdpa";
    case VulkanRouteRejectReason::KnownBadSdpaMaskOrCausal:
      return "KnownBadSdpaMaskOrCausal";
    case VulkanRouteRejectReason::KnownBadSdpaExplicitScale:
      return "KnownBadSdpaExplicitScale";
    case VulkanRouteRejectReason::KnownBadBufferLastDimSoftmax:
      return "KnownBadBufferLastDimSoftmax";
    case VulkanRouteRejectReason::KnownBadGenericTiledDiffusionLinear:
      return "KnownBadGenericTiledDiffusionLinear";
    case VulkanRouteRejectReason::DeviceQuirkDenied:
      return "DeviceQuirkDenied";
    case VulkanRouteRejectReason::ReplayViewStale:
      return "ReplayViewStale";
    case VulkanRouteRejectReason::ReplayOutputAliasUnsafe:
      return "ReplayOutputAliasUnsafe";
    case VulkanRouteRejectReason::OutputAliasUnsafe:
      return "OutputAliasUnsafe";
  }
  return "None";
}

VulkanRouteDecision make_hard_fail_route(
    const char* op_name,
    const VulkanRouteRejectReason reason,
    const std::string& shape_summary,
    const VulkanPlanningRequest& request,
    const VulkanDevicePolicy& device_policy) {
  VulkanRouteDecision decision;
  decision.kind = VulkanRouteKind::HardFail;
  decision.reject_reason = reason;
  decision.runtime_policy = build_vulkan_runtime_policy(request);
  decision.lane = infer_model_lane(decision.runtime_policy.request);
  decision.kernel_family = "none";
  decision.telemetry_label = route_reject_reason_name(reason);
  decision.shape_summary = shape_summary;
  decision.device_summary = describe_device_policy(device_policy);
  decision.hard_fail = true;
  log_route_decision(op_name, decision);
  return decision;
}

void log_route_decision(
    const char* op_name,
    const VulkanRouteDecision& decision) {
  if (!route_logging_enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(route_log_mutex());
  std::ofstream out(route_log_path(), std::ios::app);
  out << "vulkan_route";
  if (op_name && op_name[0] != '\0') {
    out << " op=" << op_name;
  }
  out << " lane=" << model_lane_name(decision.lane)
      << " decision=" << route_kind_name(decision.kind)
      << " reason=" << route_reject_reason_name(decision.reject_reason)
      << " family=" << decision.kernel_family
      << " telemetry=" << decision.telemetry_label
      << " hard_fail=" << (decision.hard_fail ? 1 : 0);
  if (!decision.shape_summary.empty()) {
    out << " shape={" << decision.shape_summary << "}";
  }
  if (!decision.device_summary.empty()) {
    out << " device={" << decision.device_summary << "}";
  }
  out << '\n';
}

std::string format_hard_fail(
    const char* op_name,
    const VulkanRouteDecision& decision) {
  return api::format_vulkan_failure(
      api::VulkanFailureClass::RouteHardFail,
      op_name,
      route_reject_reason_name(decision.reject_reason),
      hard_fail_detail(decision));
}

[[noreturn]] void fail_hard_fail(
    const char* op_name,
    const VulkanRouteDecision& decision) {
  api::context()->synchronize_device();
  api::fail_vulkan(
      api::VulkanFailureClass::RouteHardFail,
      op_name,
      route_reject_reason_name(decision.reject_reason),
      hard_fail_detail(decision));
}

VulkanRouteDecision select_softmax_route(
    const Tensor& input,
    const int64_t dim,
    const VulkanPlanningRequest& request,
    const VulkanDevicePolicy& device_policy) {
  if (
      input.dim() == 3 && dim == input.dim() - 1 && input.size(dim) >= 64 &&
      !matches_sdpa_buffer_softmax_score_contract(
          input.sizes(), input.scalar_type(), dim)) {
    return make_hard_fail_route(
        "aten::_softmax",
        VulkanRouteRejectReason::KnownBadBufferLastDimSoftmax,
        softmax_shape_summary(input, dim),
        request,
        device_policy);
  }

  VulkanRouteDecision decision;
  decision.kind = VulkanRouteKind::VulkanBufferDirectKernel;
  decision.reject_reason = VulkanRouteRejectReason::None;
  decision.runtime_policy = build_vulkan_runtime_policy(request);
  decision.lane = infer_model_lane(decision.runtime_policy.request);
  decision.kernel_family = "buffer_softmax_lastdim_float";
  decision.telemetry_label = "SelectedBufferLastDimSoftmax";
  decision.shape_summary = softmax_shape_summary(input, dim);
  decision.device_summary = describe_device_policy(device_policy);
  log_route_decision("aten::_softmax", decision);
  return decision;
}

VulkanRouteDecision select_sdpa_route(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa,
    const VulkanPlanningRequest& request,
    const VulkanDevicePolicy& device_policy) {
  const std::string shape_summary = sdpa_shape_summary(
      query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
  const VulkanRuntimePolicy runtime_policy = build_vulkan_runtime_policy(request);
  const VulkanModelLane lane = infer_model_lane(runtime_policy.request);
  const bool has_attn_mask = attn_mask && attn_mask->defined();
  const IntArrayRef attn_mask_sizes =
      has_attn_mask ? attn_mask->sizes() : IntArrayRef{};
  const ScalarType attn_mask_dtype =
      has_attn_mask ? attn_mask->scalar_type() : ScalarType::Undefined;
  const bool attn_mask_storage_supported =
      !has_attn_mask || attn_mask->is_vulkan();
  const TransformerGQASDPAMatch transformer_gqa_sdpa_match =
      match_transformer_gqa_sdpa_contract(
          query.sizes(),
          key.sizes(),
          value.sizes(),
          query.scalar_type(),
          key.scalar_type(),
          value.scalar_type(),
          has_attn_mask,
          dropout_p,
          is_causal,
          scale,
          enable_gqa);
  const bool allow_transformer_gqa_sdpa =
      transformer_gqa_sdpa_match.matched;
  const MaskedTinySDPAMatch masked_tiny_sdpa_match =
      match_masked_tiny_sdpa_contract(
          query.sizes(),
          key.sizes(),
          value.sizes(),
          attn_mask_sizes,
          query.scalar_type(),
          key.scalar_type(),
          value.scalar_type(),
          attn_mask_dtype,
          has_attn_mask,
          dropout_p,
          is_causal,
          scale,
          enable_gqa);
  const bool allow_masked_tiny_sdpa =
      masked_tiny_sdpa_match.matched && attn_mask_storage_supported;
  const DiffusionSDPAMatch diffusion_sdpa_match =
      match_diffusion_sdpa_contract(
          query.sizes(),
          key.sizes(),
          value.sizes(),
          query.scalar_type(),
          key.scalar_type(),
          value.scalar_type(),
          has_attn_mask,
          dropout_p,
          is_causal,
          scale,
          enable_gqa);
  const bool allow_diffusion_sdpa = diffusion_sdpa_match.matched;
  const VisionSelfAttentionSDPAMatch vision_self_attention_sdpa_match =
      match_vision_self_attention_sdpa_contract(
          query.sizes(),
          key.sizes(),
          value.sizes(),
          query.scalar_type(),
          key.scalar_type(),
          value.scalar_type(),
          has_attn_mask,
          dropout_p,
          is_causal,
          scale,
          enable_gqa);
  const bool allow_vision_self_attention_sdpa =
      vision_self_attention_sdpa_match.matched;
  const SDPAExecutionPolicyMatch sdpa_execution_policy_match =
      match_sdpa_execution_policy_contract(
          query.sizes(),
          key.sizes(),
          value.sizes(),
          query.scalar_type(),
          key.scalar_type(),
          value.scalar_type(),
          has_attn_mask,
          dropout_p,
          is_causal,
          scale,
          enable_gqa);
  const bool allow_sdpa_execution_policy =
      sdpa_execution_policy_match.matched;
  const bool can_check_gqa_repeat_materialization =
      enable_gqa && !has_attn_mask && !is_causal && dropout_p == 0.0 &&
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4 &&
      query.scalar_type() == key.scalar_type() &&
      key.scalar_type() == value.scalar_type() &&
      query.size(0) == key.size(0) && key.size(0) == value.size(0) &&
      key.size(1) == value.size(1) && key.size(1) > 0 &&
      query.size(1) % key.size(1) == 0 && query.size(3) == key.size(3) &&
      key.size(3) == value.size(3) && key.size(2) == value.size(2);
  const int64_t gqa_repeat_factor = can_check_gqa_repeat_materialization
      ? query.size(1) / key.size(1)
      : 0;
  const bool key_has_buffer_storage =
      key.is_vulkan() &&
      convert(key).storage_type() == api::StorageType::BUFFER;
  const bool value_has_buffer_storage =
      value.is_vulkan() &&
      convert(value).storage_type() == api::StorageType::BUFFER;
  const GQARepeatMatch key_gqa_repeat_match =
      can_check_gqa_repeat_materialization
      ? match_gqa_repeat_contract(
            key.sizes(),
            key.scalar_type(),
            key.is_vulkan(),
            key_has_buffer_storage,
            gqa_repeat_factor)
      : GQARepeatMatch{};
  const GQARepeatMatch value_gqa_repeat_match =
      can_check_gqa_repeat_materialization
      ? match_gqa_repeat_contract(
            value.sizes(),
            value.scalar_type(),
            value.is_vulkan(),
            value_has_buffer_storage,
            gqa_repeat_factor)
      : GQARepeatMatch{};
  const bool allow_gqa_repeat_materialization =
      key_gqa_repeat_match.matched && value_gqa_repeat_match.matched;
  const bool gqa_repeat_materialization_semantics_matched =
      key_gqa_repeat_match.matched && value_gqa_repeat_match.matched;

  if (
      (has_attn_mask || is_causal || enable_gqa) &&
      !allow_transformer_gqa_sdpa && !allow_masked_tiny_sdpa &&
      !allow_sdpa_execution_policy &&
      !gqa_repeat_materialization_semantics_matched) {
    return make_hard_fail_route(
        "aten::scaled_dot_product_attention",
        VulkanRouteRejectReason::KnownBadSdpaMaskOrCausal,
        shape_summary,
        request,
        device_policy);
  }

  if (dropout_p != 0.0) {
    return make_hard_fail_route(
        "aten::scaled_dot_product_attention",
        VulkanRouteRejectReason::UnsupportedDType,
        shape_summary,
        request,
        device_policy);
  }

  if (
      scale.has_value() && std::abs(*scale - 1.0) > 1.0e-9 &&
      !allow_transformer_gqa_sdpa && !allow_masked_tiny_sdpa &&
      !allow_diffusion_sdpa && !allow_sdpa_execution_policy &&
      !gqa_repeat_materialization_semantics_matched) {
    return make_hard_fail_route(
        "aten::scaled_dot_product_attention",
        VulkanRouteRejectReason::KnownBadSdpaExplicitScale,
        shape_summary,
        request,
        device_policy);
  }

  if (
      device_policy.disable_generic_4d_sdpa &&
      lane != VulkanModelLane::LLM &&
      !allow_transformer_gqa_sdpa &&
      !allow_masked_tiny_sdpa &&
      !allow_diffusion_sdpa &&
      !allow_vision_self_attention_sdpa &&
      !allow_sdpa_execution_policy &&
      !allow_gqa_repeat_materialization &&
      (query.dim() == 3 || query.dim() == 4)) {
    const int64_t target_len = query.size(query.dim() - 2);
    const int64_t source_len = key.size(key.dim() - 2);
    const int64_t head_dim = query.size(query.dim() - 1);
    if (
        (query.dim() == 4 &&
         (target_len >= 64 || source_len >= 64 || head_dim >= 64)) ||
        (query.dim() == 3 &&
         target_len >= 32 && source_len >= 29 && head_dim >= 64)) {
      return make_hard_fail_route(
          "aten::scaled_dot_product_attention",
          query.dim() == 4
              ? VulkanRouteRejectReason::KnownBadDiffusion4dSdpa
              : VulkanRouteRejectReason::KnownBadGenericSdpa,
          shape_summary,
          request,
          device_policy);
    }
  }

  VulkanRouteDecision decision;
  decision.kind = VulkanRouteKind::VulkanBufferDirectKernel;
  decision.reject_reason = VulkanRouteRejectReason::None;
  decision.runtime_policy = runtime_policy;
  decision.lane = lane;
  if (allow_transformer_gqa_sdpa) {
    decision.kernel_family = "transformer_gqa_sdpa";
    decision.telemetry_label =
        transformer_gqa_sdpa_route_label(transformer_gqa_sdpa_match.family);
  } else if (allow_masked_tiny_sdpa) {
    decision.kernel_family = "masked_tiny_sdpa";
    decision.telemetry_label =
        masked_tiny_sdpa_route_label(masked_tiny_sdpa_match.family);
  } else if (allow_diffusion_sdpa) {
    decision.kernel_family = "diffusion_sdpa";
    decision.telemetry_label =
        diffusion_sdpa_route_label(diffusion_sdpa_match.family);
  } else if (allow_vision_self_attention_sdpa) {
    decision.kernel_family = "vision_self_attention_sdpa";
    decision.telemetry_label = vision_self_attention_sdpa_route_label(
        vision_self_attention_sdpa_match.family);
  } else if (allow_sdpa_execution_policy) {
    decision.kernel_family = "sdpa_execution_policy";
    decision.telemetry_label =
        sdpa_execution_policy_family_name(sdpa_execution_policy_match.family);
  } else if (allow_gqa_repeat_materialization) {
    decision.kernel_family = "gqa_repeat_materialized_sdpa";
    decision.telemetry_label = "SelectedGQARepeatMaterializedRuntimeShape";
  } else {
    decision.kernel_family = "sdpa";
    decision.telemetry_label = "SelectedSdpa";
  }
  decision.shape_summary = shape_summary;
  decision.device_summary = describe_device_policy(device_policy);
  log_route_decision("aten::scaled_dot_product_attention", decision);
  return decision;
}

VulkanRouteDecision select_conv2d_route(
    IntArrayRef input_sizes,
    IntArrayRef weight_sizes,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    const int64_t groups,
    const ScalarType dtype,
    const bool input_requires_grad,
    const VulkanPlanningRequest& request,
    const VulkanDevicePolicy& device_policy) {
  const std::string shape_summary = conv2d_shape_summary(
      input_sizes,
      weight_sizes,
      stride,
      padding,
      dilation,
      groups,
      dtype,
      input_requires_grad);
  const VulkanRuntimePolicy runtime_policy = build_vulkan_runtime_policy(request);
  const VulkanModelLane lane = infer_model_lane(runtime_policy.request);

  if (
      device_policy.disable_large_buffer_conv_3x3 &&
      dtype == kFloat &&
      input_sizes.size() == 4 &&
      weight_sizes.size() == 4 &&
      stride.size() == 2 &&
      padding.size() == 2 &&
      dilation.size() == 2 &&
      groups == 1 &&
      weight_sizes[2] == 3 &&
      weight_sizes[3] == 3 &&
      stride[0] == 1 &&
      stride[1] == 1 &&
      padding[0] == 1 &&
      padding[1] == 1 &&
      dilation[0] == 1 &&
      dilation[1] == 1 &&
      (input_sizes[1] >= 64 || input_requires_grad) &&
      input_sizes[2] * input_sizes[3] >= 18 * 18) {
    VulkanRouteDecision decision;
    decision.kind = VulkanRouteKind::VulkanTextureKernel;
    decision.reject_reason =
        VulkanRouteRejectReason::KnownBadConv3x3Stride1Pad1;
    decision.runtime_policy = runtime_policy;
    decision.lane = lane;
    decision.kernel_family = "legacy_image_conv2d";
    decision.telemetry_label = "SelectedTextureConv2dForKnownBadLarge3x3";
    decision.shape_summary = shape_summary;
    decision.device_summary = describe_device_policy(device_policy);
    decision.hard_fail = false;
    log_route_decision("aten::convolution", decision);
    return decision;
  }

  if (
      dtype == kFloat &&
      input_sizes.size() == 4 &&
      weight_sizes.size() == 4 &&
      stride.size() == 2 &&
      padding.size() == 2 &&
      dilation.size() == 2 &&
      groups == 1 &&
      weight_sizes[2] == 1 &&
      weight_sizes[3] == 1 &&
      stride[0] == 1 &&
      stride[1] == 1 &&
      padding[0] == 0 &&
      padding[1] == 0 &&
      dilation[0] == 1 &&
      dilation[1] == 1 &&
      input_sizes[1] >= 384 &&
      weight_sizes[0] >= 192 &&
      (input_sizes[3] % 4 != 0 || input_sizes[2] * input_sizes[3] < 512)) {
    const DynamicPointwiseConv1x1DirectBufferMatch dynamic_pointwise_contract =
        match_dynamic_pointwise_conv1x1_direct_buffer_contract(
            input_sizes,
            weight_sizes,
            stride,
            padding,
            dilation,
            groups,
            dtype);
    if (dynamic_pointwise_contract.matched) {
      VulkanRouteDecision decision;
      decision.kind = VulkanRouteKind::VulkanBufferDirectKernel;
      decision.reject_reason = VulkanRouteRejectReason::None;
      decision.runtime_policy = runtime_policy;
      decision.lane = lane;
      decision.kernel_family = "dynamic_pointwise_conv1x1_direct_buffer";
      decision.telemetry_label =
          dynamic_pointwise_conv1x1_direct_buffer_route_label(
              dynamic_pointwise_contract.family);
      decision.shape_summary = shape_summary;
      decision.device_summary = describe_device_policy(device_policy);
      decision.hard_fail = false;
      log_route_decision("aten::convolution", decision);
      return decision;
    }
    return make_hard_fail_route(
        "aten::convolution",
        VulkanRouteRejectReason::KnownBadLargePointwiseConv,
        shape_summary,
        request,
        device_policy);
  }

  VulkanRouteDecision decision;
  decision.kind = VulkanRouteKind::VulkanBufferDirectKernel;
  decision.reject_reason = VulkanRouteRejectReason::None;
  decision.runtime_policy = runtime_policy;
  decision.lane = lane;
  decision.kernel_family = "buffer_float_conv2d";
  decision.telemetry_label = "SelectedBufferFloatConv2d";
  decision.shape_summary = shape_summary;
  decision.device_summary = describe_device_policy(device_policy);
  log_route_decision("aten::convolution", decision);
  return decision;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
