#include <ATen/native/vulkan/planning/RoutePolicy.h>

#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/Diagnostics.h>
#include <ATen/native/vulkan/ops/TensorState.h>

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

bool is_known_dav2_decoder_project_pointwise_shape(
    IntArrayRef input_sizes,
    IntArrayRef weight_sizes) {
  if (input_sizes.size() != 4 || weight_sizes.size() != 4) {
    return false;
  }
  if (
      input_sizes[0] != 1 ||
      input_sizes[1] != 384 ||
      weight_sizes[1] != 384 ||
      weight_sizes[2] != 1 ||
      weight_sizes[3] != 1 ||
      (weight_sizes[0] != 192 && weight_sizes[0] != 384)) {
    return false;
  }
  const int64_t height = input_sizes[2];
  const int64_t width = input_sizes[3];
  return (height == 15 && width == 10) ||
      (height == 20 && width == 13) ||
      (height == 30 && width == 20) ||
      (height == 37 && width == 57) ||
      (height == 45 && width == 30);
}

bool is_known_paddleocr_small_spatial_pointwise_shape(
    IntArrayRef input_sizes,
    IntArrayRef weight_sizes) {
  if (input_sizes.size() != 4 || weight_sizes.size() != 4) {
    return false;
  }
  if (
      input_sizes[0] != 1 || weight_sizes[2] != 1 ||
      weight_sizes[3] != 1) {
    return false;
  }
  return (input_sizes[1] == 384 && weight_sizes[1] == 384 &&
          input_sizes[2] == 7 && input_sizes[3] == 7 &&
          weight_sizes[0] == 384) ||
      (input_sizes[1] == 512 && weight_sizes[1] == 512 &&
       input_sizes[2] == 7 && input_sizes[3] == 7 &&
       weight_sizes[0] == 512) ||
      (input_sizes[1] == 512 && weight_sizes[1] == 512 &&
       input_sizes[2] == 14 && input_sizes[3] == 14 &&
       (weight_sizes[0] == 192 || weight_sizes[0] == 1024)) ||
      (input_sizes[1] == 512 && weight_sizes[1] == 512 &&
       input_sizes[2] == 1 && input_sizes[3] == 1 &&
       weight_sizes[0] == 1280) ||
      (input_sizes[1] == 1024 && weight_sizes[1] == 1024 &&
       input_sizes[2] == 7 && input_sizes[3] == 7 &&
       (weight_sizes[0] == 384 || weight_sizes[0] == 2048)) ||
      (input_sizes[1] == 1024 && weight_sizes[1] == 1024 &&
       input_sizes[2] == 14 && input_sizes[3] == 14 &&
       (weight_sizes[0] == 192 || weight_sizes[0] == 256)) ||
      (input_sizes[1] == 1664 && weight_sizes[1] == 1664 &&
       input_sizes[2] == 14 && input_sizes[3] == 14 &&
       weight_sizes[0] == 512) ||
      (input_sizes[1] == 2048 && weight_sizes[1] == 2048 &&
       input_sizes[2] == 7 && input_sizes[3] == 7 &&
       weight_sizes[0] == 256) ||
      (input_sizes[1] == 2176 && weight_sizes[1] == 2176 &&
       input_sizes[2] == 14 && input_sizes[3] == 14 &&
       weight_sizes[0] == 512) ||
      (input_sizes[1] == 3328 && weight_sizes[1] == 3328 &&
       input_sizes[2] == 7 && input_sizes[3] == 7 &&
       weight_sizes[0] == 1024);
}

bool is_known_diffusion_small_spatial_pointwise_shape(
    IntArrayRef input_sizes,
    IntArrayRef weight_sizes) {
  if (input_sizes.size() != 4 || weight_sizes.size() != 4) {
    return false;
  }
  if (
      input_sizes[0] != 1 ||
      weight_sizes[2] != 1 ||
      weight_sizes[3] != 1) {
    return false;
  }
  return (
      input_sizes[1] == 640 &&
      input_sizes[2] == 5 &&
      input_sizes[3] == 7 &&
      weight_sizes[0] == 1280 &&
      weight_sizes[1] == 640) ||
      (
          input_sizes[1] == 2560 &&
          input_sizes[2] == 3 &&
          input_sizes[3] == 4 &&
          weight_sizes[0] == 1280 &&
          weight_sizes[1] == 2560) ||
      (
          input_sizes[1] == 2560 &&
          input_sizes[2] == 5 &&
          input_sizes[3] == 7 &&
          weight_sizes[0] == 1280 &&
          weight_sizes[1] == 2560) ||
      (
          input_sizes[1] == 1920 &&
          input_sizes[2] == 5 &&
          input_sizes[3] == 7 &&
          weight_sizes[0] == 1280 &&
          weight_sizes[1] == 1920) ||
      (
          input_sizes[1] == 1920 &&
          input_sizes[2] == 9 &&
          input_sizes[3] == 14 &&
          weight_sizes[0] == 640 &&
          weight_sizes[1] == 1920);
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

bool is_known_hymt_small_causal_gqa_sdpa_shape(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa) {
  constexpr double kHymtHeadDim128Scale = 0.08838834764831845;
  if (
      (attn_mask && attn_mask->defined()) || dropout_p != 0.0 ||
      (!is_causal && !enable_gqa) || query.scalar_type() != kFloat ||
      key.scalar_type() != kFloat || value.scalar_type() != kFloat ||
      query.dim() != 4 || key.dim() != 4 || value.dim() != 4) {
    return false;
  }
  if (
      scale.has_value() &&
      std::abs(*scale - kHymtHeadDim128Scale) > 1.0e-6) {
    return false;
  }
  if (
      query.size(0) != 1 || key.size(0) != 1 || value.size(0) != 1 ||
      query.size(1) != 16 ||
      query.size(2) < 1 || query.size(2) > 128 ||
      query.size(3) != 128 || key.size(2) < query.size(2) ||
      key.size(3) != 128 ||
      value.size(2) != key.size(2) || value.size(3) != 128 ||
      key.size(1) != value.size(1)) {
    return false;
  }
  if (is_causal) {
    if (query.size(2) != key.size(2) || key.size(2) > 128) {
      return false;
    }
  } else if (query.size(2) > 14 || key.size(2) > 64) {
    return false;
  }
  return enable_gqa ? key.size(1) == 4 : key.size(1) == 16;
}

bool is_supported_tiny_float_mask_sdpa_shape(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa) {
  constexpr double kHeadDim64Scale = 0.125;
  if (
      !attn_mask || !attn_mask->defined() || dropout_p != 0.0 ||
      is_causal || enable_gqa || query.scalar_type() != kFloat ||
      key.scalar_type() != kFloat || value.scalar_type() != kFloat ||
      attn_mask->scalar_type() != kFloat || query.dim() != 4 ||
      key.dim() != 4 || value.dim() != 4 || attn_mask->dim() != 4) {
    return false;
  }
  if (scale.has_value() && std::abs(*scale - kHeadDim64Scale) > 1.0e-6) {
    return false;
  }
  return query.size(0) == 1 && key.size(0) == 1 && value.size(0) == 1 &&
      query.size(1) == 16 && key.size(1) == 16 && value.size(1) == 16 &&
      query.size(2) == 2 && key.size(2) == 2 && value.size(2) == 2 &&
      query.size(3) == 64 && key.size(3) == 64 && value.size(3) == 64 &&
      attn_mask->size(0) == 1 && attn_mask->size(1) == 1 &&
      attn_mask->size(2) == 2 && attn_mask->size(3) == 2;
}

bool is_supported_materialized_diffusion_sdpa_shape(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa) {
  constexpr double kHeadDim512Scale = 0.04419417382415922;
  constexpr double kHeadDim64Scale = 0.125;
  if (
      (attn_mask && attn_mask->defined()) || dropout_p != 0.0 ||
      is_causal || enable_gqa || query.scalar_type() != kFloat ||
      key.scalar_type() != kFloat || value.scalar_type() != kFloat ||
      query.dim() != 4 || key.dim() != 4 || value.dim() != 4) {
    return false;
  }
  if (
      query.size(0) != 1 || key.size(0) != 1 || value.size(0) != 1 ||
      query.size(1) != key.size(1) || query.size(1) != value.size(1) ||
      query.size(2) != key.size(2) || query.size(2) != value.size(2) ||
      query.size(3) != key.size(3) || query.size(3) != value.size(3)) {
    return false;
  }
  const int64_t heads = query.size(1);
  const int64_t sequence = query.size(2);
  const int64_t head_dim = query.size(3);
  const bool supported =
      (heads == 1 && sequence == 640 && head_dim == 512) ||
      (heads == 5 && sequence == 640 && head_dim == 64) ||
      (heads == 1 && sequence == 504 && head_dim == 512) ||
      (heads == 5 && sequence == 504 && head_dim == 64) ||
      (heads == 10 && sequence == 126 && head_dim == 64) ||
      (heads == 20 && sequence == 35 && head_dim == 64) ||
      (heads == 20 && sequence == 12 && head_dim == 64);
  if (!supported) {
    return false;
  }
  if (scale.has_value()) {
    const double expected_scale =
        head_dim == 512 ? kHeadDim512Scale : kHeadDim64Scale;
    if (std::abs(*scale - expected_scale) > 1.0e-6) {
      return false;
    }
  }
  return true;
}

bool is_supported_diffusion_cross_sdpa_shape(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa) {
  constexpr double kHeadDim64Scale = 0.125;
  if (
      (attn_mask && attn_mask->defined()) || dropout_p != 0.0 ||
      is_causal || enable_gqa || query.scalar_type() != kFloat ||
      key.scalar_type() != kFloat || value.scalar_type() != kFloat ||
      query.dim() != 4 || key.dim() != 4 || value.dim() != 4) {
    return false;
  }
  if (scale.has_value() && std::abs(*scale - kHeadDim64Scale) > 1.0e-6) {
    return false;
  }
  if (
      query.size(0) != 1 || key.size(0) != 1 || value.size(0) != 1 ||
      query.size(1) != key.size(1) || query.size(1) != value.size(1) ||
      query.size(3) != 64 || key.size(3) != 64 || value.size(3) != 64 ||
      key.size(2) != 2 || value.size(2) != 2) {
    return false;
  }
  const int64_t heads = query.size(1);
  const int64_t query_sequence = query.size(2);
  return (heads == 5 && query_sequence == 504) ||
      (heads == 10 && query_sequence == 126) ||
      (heads == 20 && (query_sequence == 12 || query_sequence == 35));
}

bool is_supported_diffusion_sdpa_score_softmax_shape(
    const Tensor& input,
    const int64_t dim) {
  if (
      input.scalar_type() != kFloat || input.dim() != 3 ||
      dim != input.dim() - 1 || input.size(1) != input.size(2)) {
    return false;
  }
  const int64_t heads = input.size(0);
  const int64_t sequence = input.size(1);
  return (heads == 1 && (sequence == 504 || sequence == 640)) ||
      (heads == 5 && (sequence == 504 || sequence == 640));
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
  decision.lane = infer_model_lane(request);
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
      !is_supported_diffusion_sdpa_score_softmax_shape(input, dim)) {
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
  decision.lane = infer_model_lane(request);
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
  const VulkanModelLane lane = infer_model_lane(request);
  const bool allow_hymt_small_causal_gqa =
      is_known_hymt_small_causal_gqa_sdpa_shape(
          query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
  const bool allow_tiny_float_mask =
      is_supported_tiny_float_mask_sdpa_shape(
          query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
  const bool allow_materialized_diffusion =
      is_supported_materialized_diffusion_sdpa_shape(
          query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
  const bool allow_diffusion_cross =
      is_supported_diffusion_cross_sdpa_shape(
          query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);

  if (
      (attn_mask && attn_mask->defined() || is_causal || enable_gqa) &&
      !allow_hymt_small_causal_gqa && !allow_tiny_float_mask) {
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
      !allow_hymt_small_causal_gqa && !allow_tiny_float_mask &&
      !allow_materialized_diffusion && !allow_diffusion_cross) {
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
      !allow_hymt_small_causal_gqa &&
      !allow_tiny_float_mask &&
      !allow_materialized_diffusion &&
      !allow_diffusion_cross &&
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
  decision.kernel_family = "sdpa";
  decision.telemetry_label = "SelectedSdpa";
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
  const VulkanModelLane lane = infer_model_lane(request);

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
    if (is_known_dav2_decoder_project_pointwise_shape(
            input_sizes, weight_sizes)) {
      VulkanRouteDecision decision;
      decision.kind = VulkanRouteKind::VulkanBufferDirectKernel;
      decision.reject_reason =
          VulkanRouteRejectReason::KnownBadLargePointwiseConv;
      decision.runtime_policy = runtime_policy;
      decision.lane = lane;
      decision.kernel_family = "buffer_float_conv2d_generic";
      decision.telemetry_label =
          "SelectedGenericBufferConv2dForDav2DecoderProjectPointwise";
      decision.shape_summary = shape_summary;
      decision.device_summary = describe_device_policy(device_policy);
      decision.hard_fail = false;
      log_route_decision("aten::convolution", decision);
      return decision;
    }
    if (is_known_paddleocr_small_spatial_pointwise_shape(
            input_sizes, weight_sizes)) {
      VulkanRouteDecision decision;
      decision.kind = VulkanRouteKind::VulkanBufferDirectKernel;
      decision.reject_reason =
          VulkanRouteRejectReason::KnownBadLargePointwiseConv;
      decision.runtime_policy = runtime_policy;
      decision.lane = lane;
      decision.kernel_family = "buffer_float_conv2d_generic";
      decision.telemetry_label =
          "SelectedGenericBufferConv2dForPaddleOCRSmallSpatialPointwise";
      decision.shape_summary = shape_summary;
      decision.device_summary = describe_device_policy(device_policy);
      decision.hard_fail = false;
      log_route_decision("aten::convolution", decision);
      return decision;
    }
    if (is_known_diffusion_small_spatial_pointwise_shape(
            input_sizes, weight_sizes)) {
      VulkanRouteDecision decision;
      decision.kind = VulkanRouteKind::VulkanBufferDirectKernel;
      decision.reject_reason =
          VulkanRouteRejectReason::KnownBadLargePointwiseConv;
      decision.runtime_policy = runtime_policy;
      decision.lane = lane;
      decision.kernel_family = "buffer_float_conv2d_generic";
      decision.telemetry_label =
          "SelectedGenericBufferConv2dForDiffusionSmallSpatialPointwise";
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
