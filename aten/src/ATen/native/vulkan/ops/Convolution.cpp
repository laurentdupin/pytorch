
#include <ATen/Context.h>

#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/vulkan/api/Diagnostics.h>
#include <ATen/native/vulkan/api/Utils.h>
#include <ATen/native/vulkan/impl/Packing.h>
#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/ExecutionObjects.h>
#include <ATen/native/vulkan/planning/RoutePolicy.h>
#include <ATen/native/vulkan/ops/BinaryOp.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/conv2d.h>
#include <ATen/ops/convolution.h>
#include <ATen/ops/dequantize.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/permute.h>
#include <ATen/ops/quantize_per_tensor.h>
#include <ATen/ops/zeros.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace {

bool output_padding_is_zero(const IntArrayRef output_padding);

enum class VulkanConvPlanSelected : uint8_t {
  Unknown = 0,
  TextureConv,
  FloatBufferConv,
  FloatBufferPointwise1x1,
  FloatBufferPointwise1x1AsLinear,
  CpuFallback,
  HardFailKnownBad,
};

enum class VulkanConvRejectReason : uint8_t {
  None = 0,
  InputNotVulkan,
  UnsupportedDType,
  UnsupportedRank,
  WeightNotPacked,
  UnsupportedLayout,
  UnsupportedGroups,
  UnsupportedKernel,
  UnsupportedStridePaddingDilation,
  KnownBadLargePointwiseConv,
  ShapeUnsupported,
  Unknown,
};

struct VulkanConvPlanDecision final {
  VulkanConvPlanSelected selected{VulkanConvPlanSelected::Unknown};
  VulkanConvRejectReason reject{VulkanConvRejectReason::None};
  int64_t n{0};
  int64_t cin{0};
  int64_t h{0};
  int64_t w{0};
  int64_t cout{0};
  int64_t kh{0};
  int64_t kw{0};
  int64_t groups{0};
  bool input_vulkan{false};
  bool input_buffer{false};
  bool weight_packed{false};
  bool bias_present{false};
  bool transposed{false};
  bool pointwise{false};
  bool large{false};
};

struct VulkanConvPlanCounters final {
  std::atomic<uint64_t> total{0u};
  std::atomic<uint64_t> pointwise_1x1_hit{0u};
  std::atomic<uint64_t> pointwise_1x1_as_linear_hit{0u};
  std::atomic<uint64_t> known_bad_large_pointwise{0u};
  std::atomic<uint64_t> cpu_fallback{0u};
  std::atomic<uint64_t> reject_layout{0u};
  std::atomic<uint64_t> reject_dtype{0u};
};

struct VulkanPointwiseConvRouteCounters final {
  std::atomic<uint64_t> total_1x1{0u};
  std::atomic<uint64_t> specialized_1x1_hit{0u};
  std::atomic<uint64_t> generic_1x1_hit{0u};
  std::atomic<uint64_t> reject_not_direct_buffer{0u};
  std::atomic<uint64_t> reject_input_not_buffer{0u};
  std::atomic<uint64_t> reject_input_not_direct_buffer{0u};
  std::atomic<uint64_t> reject_output_not_direct_buffer{0u};
  std::atomic<uint64_t> reject_storage_offset{0u};
  std::atomic<uint64_t> reject_dtype{0u};
  std::atomic<uint64_t> reject_groups{0u};
  std::atomic<uint64_t> reject_stride_padding_dilation{0u};
  std::atomic<uint64_t> reject_weight_layout{0u};
  std::atomic<uint64_t> reject_bias{0u};
  std::atomic<uint64_t> reject_shape{0u};
};

struct VulkanConvAggregateKey final {
  VulkanConvPlanSelected selected{VulkanConvPlanSelected::Unknown};
  VulkanConvRejectReason reject{VulkanConvRejectReason::None};
  std::string kernel_name;
  std::string role;
  int64_t n{0};
  int64_t cin{0};
  int64_t h{0};
  int64_t w{0};
  int64_t cout{0};
  int64_t kh{0};
  int64_t kw{0};
  int64_t groups{0};
  int64_t stride_h{0};
  int64_t stride_w{0};
  int64_t pad_h{0};
  int64_t pad_w{0};
  int64_t dilation_h{0};
  int64_t dilation_w{0};
  bool input_direct{false};
  bool output_direct{false};
  bool weight_packed{false};
  bool bias{false};
  bool pointwise{false};
  bool depthwise{false};
  bool sliding_window{false};

  bool operator==(const VulkanConvAggregateKey& other) const {
    return selected == other.selected && reject == other.reject &&
        kernel_name == other.kernel_name && role == other.role &&
        n == other.n && cin == other.cin && h == other.h && w == other.w &&
        cout == other.cout && kh == other.kh && kw == other.kw &&
        groups == other.groups && stride_h == other.stride_h &&
        stride_w == other.stride_w && pad_h == other.pad_h &&
        pad_w == other.pad_w && dilation_h == other.dilation_h &&
        dilation_w == other.dilation_w && input_direct == other.input_direct &&
        output_direct == other.output_direct &&
        weight_packed == other.weight_packed && bias == other.bias &&
        pointwise == other.pointwise && depthwise == other.depthwise &&
        sliding_window == other.sliding_window;
  }
};

struct VulkanConvAggregateKeyHash final {
  size_t operator()(const VulkanConvAggregateKey& key) const {
    size_t seed = 0;
    auto combine = [&seed](const auto& value) {
      seed ^= std::hash<std::decay_t<decltype(value)>>{}(value) + 0x9e3779b9 +
          (seed << 6) + (seed >> 2);
    };
    combine(static_cast<uint8_t>(key.selected));
    combine(static_cast<uint8_t>(key.reject));
    combine(key.kernel_name);
    combine(key.role);
    combine(key.n);
    combine(key.cin);
    combine(key.h);
    combine(key.w);
    combine(key.cout);
    combine(key.kh);
    combine(key.kw);
    combine(key.groups);
    combine(key.stride_h);
    combine(key.stride_w);
    combine(key.pad_h);
    combine(key.pad_w);
    combine(key.dilation_h);
    combine(key.dilation_w);
    combine(key.input_direct);
    combine(key.output_direct);
    combine(key.weight_packed);
    combine(key.bias);
    combine(key.pointwise);
    combine(key.depthwise);
    combine(key.sliding_window);
    return seed;
  }
};

struct VulkanConvAggregateValue final {
  uint64_t count{0};
  uint64_t input_bytes{0};
  uint64_t output_bytes{0};
  uint64_t weight_bytes{0};
};

class VulkanConvAggregateProfiler final {
 public:
  void record(
      const VulkanConvAggregateKey& key,
      const uint64_t input_bytes,
      const uint64_t output_bytes,
      const uint64_t weight_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    VulkanConvAggregateValue& value = entries_[key];
    value.count += 1u;
    value.input_bytes += input_bytes;
    value.output_bytes += output_bytes;
    value.weight_bytes += weight_bytes;
  }

  std::vector<std::pair<VulkanConvAggregateKey, VulkanConvAggregateValue>>
  snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<VulkanConvAggregateKey, VulkanConvAggregateValue>> out;
    out.reserve(entries_.size());
    for (const auto& entry : entries_) {
      out.emplace_back(entry.first, entry.second);
    }
    return out;
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<
      VulkanConvAggregateKey,
      VulkanConvAggregateValue,
      VulkanConvAggregateKeyHash>
      entries_;
};

VulkanConvPlanCounters& conv_plan_counters() {
  static VulkanConvPlanCounters counters;
  return counters;
}

VulkanPointwiseConvRouteCounters& pointwise_conv_route_counters() {
  static VulkanPointwiseConvRouteCounters counters;
  return counters;
}

VulkanConvAggregateProfiler& conv_aggregate_profiler() {
  static VulkanConvAggregateProfiler profiler;
  return profiler;
}

const char* conv_plan_selected_name(const VulkanConvPlanSelected selected) {
  switch (selected) {
    case VulkanConvPlanSelected::TextureConv:
      return "TextureConv";
    case VulkanConvPlanSelected::FloatBufferConv:
      return "FloatBufferConv";
    case VulkanConvPlanSelected::FloatBufferPointwise1x1:
      return "FloatBufferPointwise1x1";
    case VulkanConvPlanSelected::FloatBufferPointwise1x1AsLinear:
      return "FloatBufferPointwise1x1AsLinear";
    case VulkanConvPlanSelected::CpuFallback:
      return "CpuFallback";
    case VulkanConvPlanSelected::HardFailKnownBad:
      return "HardFailKnownBad";
    case VulkanConvPlanSelected::Unknown:
      return "Unknown";
  }
  return "Unknown";
}

const char* conv_reject_reason_name(const VulkanConvRejectReason reject) {
  switch (reject) {
    case VulkanConvRejectReason::None:
      return "None";
    case VulkanConvRejectReason::InputNotVulkan:
      return "InputNotVulkan";
    case VulkanConvRejectReason::UnsupportedDType:
      return "UnsupportedDType";
    case VulkanConvRejectReason::UnsupportedRank:
      return "UnsupportedRank";
    case VulkanConvRejectReason::WeightNotPacked:
      return "WeightNotPacked";
    case VulkanConvRejectReason::UnsupportedLayout:
      return "UnsupportedLayout";
    case VulkanConvRejectReason::UnsupportedGroups:
      return "UnsupportedGroups";
    case VulkanConvRejectReason::UnsupportedKernel:
      return "UnsupportedKernel";
    case VulkanConvRejectReason::UnsupportedStridePaddingDilation:
      return "UnsupportedStridePaddingDilation";
    case VulkanConvRejectReason::KnownBadLargePointwiseConv:
      return "KnownBadLargePointwiseConv";
    case VulkanConvRejectReason::ShapeUnsupported:
      return "ShapeUnsupported";
    case VulkanConvRejectReason::Unknown:
      return "Unknown";
  }
  return "Unknown";
}

const std::string& vulkan_conv_plan_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_CONV_PLAN_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

void append_vulkan_conv_plan_log(
    const VulkanConvPlanDecision& decision,
    const char* label) {
  const std::string& path = vulkan_conv_plan_log_path();
  if (path.empty()) {
    return;
  }
  std::ofstream out(path, std::ios::app);
  out << "conv_plan"
      << " label=" << (label ? label : "unknown")
      << " selected=" << static_cast<int>(decision.selected)
      << " reject=" << static_cast<int>(decision.reject)
      << " n=" << decision.n
      << " cin=" << decision.cin
      << " h=" << decision.h
      << " w=" << decision.w
      << " cout=" << decision.cout
      << " kh=" << decision.kh
      << " kw=" << decision.kw
      << " groups=" << decision.groups
      << " input_vulkan=" << (decision.input_vulkan ? 1 : 0)
      << " input_buffer=" << (decision.input_buffer ? 1 : 0)
      << " weight_packed=" << (decision.weight_packed ? 1 : 0)
      << " bias=" << (decision.bias_present ? 1 : 0)
      << " transposed=" << (decision.transposed ? 1 : 0)
      << " pointwise=" << (decision.pointwise ? 1 : 0)
      << " large=" << (decision.large ? 1 : 0)
      << '\n';
}

void update_conv_plan_counters(const VulkanConvPlanDecision& decision) {
  VulkanConvPlanCounters& counters = conv_plan_counters();
  counters.total.fetch_add(1u, std::memory_order_relaxed);
  if (decision.selected == VulkanConvPlanSelected::FloatBufferPointwise1x1) {
    counters.pointwise_1x1_hit.fetch_add(1u, std::memory_order_relaxed);
  }
  if (
      decision.selected ==
      VulkanConvPlanSelected::FloatBufferPointwise1x1AsLinear) {
    counters.pointwise_1x1_as_linear_hit.fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (decision.reject == VulkanConvRejectReason::KnownBadLargePointwiseConv) {
    counters.known_bad_large_pointwise.fetch_add(
        1u, std::memory_order_relaxed);
  }
  if (decision.selected == VulkanConvPlanSelected::CpuFallback) {
    counters.cpu_fallback.fetch_add(1u, std::memory_order_relaxed);
  }
  if (decision.reject == VulkanConvRejectReason::UnsupportedLayout) {
    counters.reject_layout.fetch_add(1u, std::memory_order_relaxed);
  }
  if (decision.reject == VulkanConvRejectReason::UnsupportedDType) {
    counters.reject_dtype.fetch_add(1u, std::memory_order_relaxed);
  }
}

void record_conv_plan_decision(
    const VulkanConvPlanDecision& decision,
    const char* label) {
  update_conv_plan_counters(decision);
  append_vulkan_conv_plan_log(decision, label);
}

utils::VulkanPlanningRequest convolution_request(
    const utils::VulkanTensorRole role) {
  return utils::make_vulkan_planning_request(
      utils::VulkanWorkloadClass::Convolution, role);
}

PackedWeightKind packed_weight_kind_for_conv2d_method(
    const Conv2dMethod method) {
  switch (method) {
    case Conv2dDepthwise:
      return PackedWeightKind::Conv2dDepthwise;
    case Conv2dPointwise:
      return PackedWeightKind::Conv2dPointwise;
    case Conv2dSlidingWindow:
      return PackedWeightKind::Conv2dSlidingWindow;
  }
  return PackedWeightKind::Unknown;
}

} // namespace

namespace conv2d {

inline bool has_bias(const std::optional<Tensor>& bias) {
  return bias && bias->defined();
}

std::string format_conv_sizes(IntArrayRef sizes) {
  std::ostringstream stream;
  stream << '[';
  for (const auto idx : c10::irange(sizes.size())) {
    if (idx > 0) {
      stream << 'x';
    }
    stream << sizes[idx];
  }
  stream << ']';
  return stream.str();
}

std::string classify_conv_role(
    const IntArrayRef input_sizes,
    const IntArrayRef weight_sizes,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups) {
  const int64_t cin = input_sizes.size() > 1 ? input_sizes[1] : 0;
  const int64_t cout = weight_sizes.size() > 0 ? weight_sizes[0] : 0;
  const int64_t kh = weight_sizes.size() > 2 ? weight_sizes[2] : 0;
  const int64_t kw = weight_sizes.size() > 3 ? weight_sizes[3] : 0;
  const int64_t n = input_sizes.size() > 0 ? input_sizes[0] : 0;
  const int64_t h = input_sizes.size() > 2 ? input_sizes[2] : 0;
  const int64_t w = input_sizes.size() > 3 ? input_sizes[3] : 0;
  const int64_t stride_h = stride.size() > 0 ? stride[0] : 0;
  const int64_t stride_w = stride.size() > 1 ? stride[1] : 0;
  const int64_t pad_h = padding.size() > 0 ? padding[0] : 0;
  const int64_t pad_w = padding.size() > 1 ? padding[1] : 0;
  const int64_t dilation_h = dilation.size() > 0 ? dilation[0] : 0;
  const int64_t dilation_w = dilation.size() > 1 ? dilation[1] : 0;
  if (cin == 3 && cout == 384 && kh == 14 && kw == 14 &&
      stride.size() >= 2 && stride[0] == 14 && stride[1] == 14 &&
      groups == 1) {
    return "patch_embed";
  }

  const bool high_channel_384 =
      n == 1 && cin == 384 && cout == 384 && groups == 1;
  const bool small_spatial_decoder =
      h >= 16 && h <= 80 && w >= 16 && w <= 96;
  if (high_channel_384 && small_spatial_decoder && kh == 1 && kw == 1 &&
      stride_h == 1 && stride_w == 1 && pad_h == 0 && pad_w == 0 &&
      dilation_h == 1 && dilation_w == 1) {
    return "decoder_head_pointwise_1x1";
  }
  if (high_channel_384 && small_spatial_decoder && kh == 3 && kw == 3 &&
      stride_h == 2 && stride_w == 2 && pad_h == 1 && pad_w == 1 &&
      dilation_h == 1 && dilation_w == 1) {
    return "decoder_head_3x3_s2p1";
  }
  if (high_channel_384 && small_spatial_decoder && kh == 3 && kw == 3 &&
      stride_h == 1 && stride_w == 1 && pad_h == 1 && pad_w == 1 &&
      dilation_h == 1 && dilation_w == 1) {
    return "decoder_head_3x3_s1p1";
  }
  if (kh == 3 && kw == 3 && stride.size() >= 2 && padding.size() >= 2 &&
      dilation.size() >= 2 && stride[0] == 1 && stride[1] == 1 &&
      padding[0] == 1 && padding[1] == 1 && dilation[0] == 1 &&
      dilation[1] == 1 && groups == 1) {
    return "other_3x3_s1p1";
  }
  if (kh == 1 && kw == 1 && groups == 1) {
    return "other_pointwise_1x1";
  }
  if (groups == cin && groups == cout && groups > 1) {
    return "depthwise";
  }
  if (input_sizes.size() >= 4 && input_sizes[2] <= 74 && input_sizes[3] <= 114) {
    return "decoder_head_generic";
  }
  if (kh == 3 && kw == 3 && groups == 1) {
    return "other_3x3";
  }
  return "other_generic";
}

VulkanConvPlanSelected selected_from_conv_kernel_name(
    const std::string& kernel_name) {
  if (kernel_name == "conv2d_buffer_float_1x1") {
    return VulkanConvPlanSelected::FloatBufferPointwise1x1;
  }
  return VulkanConvPlanSelected::FloatBufferConv;
}

void record_float_buffer_conv2d_aggregate(
    const char* kernel_name,
    const vTensor& v_input,
    const vTensor& v_output,
    const PackedWeightHandle& packed_weight,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups) {
  const IntArrayRef input_sizes = v_input.sizes();
  const IntArrayRef output_sizes = v_output.sizes();
  const IntArrayRef weight_sizes = packed_weight.logical_weight_sizes();
  VulkanConvAggregateKey key;
  key.kernel_name = kernel_name ? kernel_name : "unknown";
  key.selected = selected_from_conv_kernel_name(key.kernel_name);
  key.reject = VulkanConvRejectReason::None;
  key.role = classify_conv_role(
      input_sizes, weight_sizes, stride, padding, dilation, groups);
  key.n = input_sizes.size() > 0 ? input_sizes[0] : 0;
  key.cin = input_sizes.size() > 1 ? input_sizes[1] : 0;
  key.h = input_sizes.size() > 2 ? input_sizes[2] : 0;
  key.w = input_sizes.size() > 3 ? input_sizes[3] : 0;
  key.cout = weight_sizes.size() > 0 ? weight_sizes[0] : 0;
  key.kh = weight_sizes.size() > 2 ? weight_sizes[2] : 0;
  key.kw = weight_sizes.size() > 3 ? weight_sizes[3] : 0;
  key.groups = groups;
  key.stride_h = stride.size() > 0 ? stride[0] : 0;
  key.stride_w = stride.size() > 1 ? stride[1] : 0;
  key.pad_h = padding.size() > 0 ? padding[0] : 0;
  key.pad_w = padding.size() > 1 ? padding[1] : 0;
  key.dilation_h = dilation.size() > 0 ? dilation[0] : 0;
  key.dilation_w = dilation.size() > 1 ? dilation[1] : 0;
  key.input_direct = v_input.has_direct_buffer_layout();
  key.output_direct = v_output.has_direct_buffer_layout();
  key.weight_packed = true;
  key.bias = packed_weight.has_bias();
  key.pointwise = key.kh == 1 && key.kw == 1;
  key.depthwise = groups == key.cin && groups == key.cout && groups > 1;
  key.sliding_window = !key.pointwise && !key.depthwise;

  const vTensor v_weight = packed_weight.weight_vtensor();
  conv_aggregate_profiler().record(
      key,
      static_cast<uint64_t>(v_input.gpu_nbytes()),
      static_cast<uint64_t>(v_output.gpu_nbytes()),
      static_cast<uint64_t>(v_weight.gpu_nbytes()));
}

void log_float_buffer_conv2d_submit(
    const char* kernel_name,
    const vTensor& v_input,
    const vTensor& v_output,
    const PackedWeightHandle& packed_weight,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups,
    const api::utils::uvec3& global_size,
    const api::utils::uvec3& local_size) {
  std::ostringstream stream;
  stream << "aten::convolution.submit"
         << " kernel=" << kernel_name
         << " input=" << format_conv_sizes(v_input.sizes())
         << " output=" << format_conv_sizes(v_output.sizes())
         << " weight=" << format_conv_sizes(packed_weight.logical_weight_sizes())
         << " bias=" << (packed_weight.has_bias() ? 1 : 0)
         << " stride=" << format_conv_sizes(stride)
         << " padding=" << format_conv_sizes(padding)
         << " dilation=" << format_conv_sizes(dilation)
         << " groups=" << groups
         << " input_direct=" << (v_input.has_direct_buffer_layout() ? 1 : 0)
         << " output_direct=" << (v_output.has_direct_buffer_layout() ? 1 : 0)
         << " input_offset=" << v_input.storage_offset()
         << " output_offset=" << v_output.storage_offset()
         << " global=[" << global_size.data[0] << 'x' << global_size.data[1]
         << 'x' << global_size.data[2] << ']'
         << " local=[" << local_size.data[0] << 'x' << local_size.data[1]
         << 'x' << local_size.data[2] << ']';
  utils::log_vulkan_op_hit(stream.str());
  record_float_buffer_conv2d_aggregate(
      kernel_name,
      v_input,
      v_output,
      packed_weight,
      stride,
      padding,
      dilation,
      groups);
}

//
// Convolution type classification
//

inline bool is_depthwise(const IntArrayRef weight_size, const int64_t groups) {
  uint32_t groups_uint = api::utils::safe_downcast<uint32_t>(groups);
  if (get_dim<DimConv2DKernel::OutChannels>(weight_size) != groups_uint) {
    return false;
  }
  if (get_dim<DimConv2DKernel::InChannels>(weight_size) != 1) {
    return false;
  }
  return true;
}

inline bool is_pointwise(const IntArrayRef weight_size) {
  if (get_dim<DimConv2DKernel::Width>(weight_size) != 1) {
    return false;
  }
  if (get_dim<DimConv2DKernel::Height>(weight_size) != 1) {
    return false;
  }
  return true;
}

static Conv2dMethod determine_method(
    const IntArrayRef weight_size,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups,
    const bool transposed,
    const bool quantized) {
  if (transposed) {
    return Conv2dSlidingWindow;
  }
  if (is_depthwise(weight_size, groups)) {
    return Conv2dDepthwise;
  }
  if (is_pointwise(weight_size)) {
    return Conv2dPointwise;
  }
  return Conv2dSlidingWindow;
}

//
// Rearrangement functions for pre-packing
//

/*
 * Rearranges a convolution weight tensor to a layout that can be used by
 * convolution compute shaders. The goal of this packing is to arrange the data
 * such that data access in the compute shader is as linear as possible. The
 * reasoning behind the packing pattern will be described in the shader kernel
 * code.
 *
 * To understand the transformations performed by this function, consider an
 * example input of size {11, 1, 3, 3}. The following transformations will
 * applied to this weight tensor:
 *
 * 1. First, apply padding to the N dims so that it is a multiple of 4.
 * In this case, 1 batch is added, producing a tensor of size {12,1,3,3}.
 *
 * 2. Next, flatten the last two dims of the tensor. This is done by reshaping
 * the tensor to size {12,1,9}.
 *
 * 3. Finally, we want to "fold" the batch dim into the channel dim. We start by
 * splitting the tensor along the N dim so that each split has 4 batches. This
 * is done by reshaping the tensor to size {3,4,1,9}.
 *
 * 4. Normally, we would be done, but we want to stack each back vertically.
 * This is done by permuting the N and C dims and reshaping the tensor to size
 * {4,3,9}.
 */
at::Tensor rearrange_weights_dw(const Tensor& weight_in) {
  at::Tensor weight = weight_in.clone();

  uint32_t N = ops::get_dim<DimConv2DKernel::OutChannels>(weight);
  uint32_t C = ops::get_dim<DimConv2DKernel::InChannels>(weight);
  uint32_t H = ops::get_dim<DimConv2DKernel::Height>(weight);
  uint32_t W = ops::get_dim<DimConv2DKernel::Width>(weight);

  uint32_t N_aligned = api::utils::align_up(N, 4u);

  // Add padding to the N dimension so that it's a multiple of 4
  uint32_t N_padding_needed = N_aligned - N;
  weight =
      at::pad(weight, {0, 0, 0, 0, 0, 0, 0, N_padding_needed}, "constant", 0);

  // Flatten so the H and W dim are on one row
  weight = weight.reshape({N_aligned, C, H * W});

  // Split batch dim to make groups of 4
  uint32_t N4 = N_aligned / 4u;
  weight = weight.reshape({N4, 4, C, H * W});

  // Permute the groups of 4 so they are arranged along the channel dim, then
  // reshape to stack the resulting batches vertically
  weight = weight.permute({1, 0, 2, 3}).reshape({4, N4 * C, H * W});

  return weight.contiguous();
}

/*
 * Rearranges a convolution weight tensor to a layout that can be used by
 * convolution compute shaders. The goal of this packing is to arrange the data
 * such that data access in the compute shader is as linear as possible. The
 * reasoning behind the packing pattern will be described in the shader kernel
 * code.
 *
 * To understand the transformations performed by this function, consider an
 * example input of size {10, 7, 3, 3}. The following transformations will
 * applied to this weight tensor:
 *
 * 1. First, apply padding to the N and C dims so that both are a multiple of 4.
 * In this case, 2 batches and 1 channel of padding are added, producing a
 * tensor of size {12,8,3,3}.
 *
 * 2. Next, split the tensor along the C dim so that each split has 4 channels.
 * This is done by reshaping the channel to have the size {12,2,(4,3,3)}. ()
 * brackets denote the size of the split.
 *
 * 3. For each split, we want to "fold" the C dim into the W dim. So suppose the
 * first rows at H=0 of the split has values
 *
 *    0,1,2 | 10,11,12 | 20,21,22 | 30,31,32
 *
 *    where | denotes a channel boundary, then the goal is to combine those rows
 * into one row with the values
 *
 *    0, 10, 20, 30, 1, 11, 21, 31, 2, 12, 22, 32
 *
 *    This is done in code by permuting and reshaping the tensor, producing a
 * tensor of size {12,2,(3,12)}.
 *
 * 4. Next, we want to stack the splits belonging to the same batch horizontally
 * which is done by swapping the C and H dims of the intermediate tensor and
 * reshaping to produce a tensor of size {12,3,24}.
 *
 * 5. Now we will repeat a similar process of "folding" the N dim into the C
 * dim. We start by splitting along the N dim so that each split has 4 batches.
 * To do this the tensor is reshaped to {3,4,3,24}.
 *
 * 6. Normally, we would be done but we also want to stack each batch on each
 * other vertically. Therefore final step is another permute swapping the N and
 * C dims and reshaping to the output shape of {4, 9, 24}.
 *
 * For transposed convolutions, there are some slight differences to reflect the
 * data access pattern in the shader. The first major difference is that the
 * weight tensor is flipped along the H and W dims. The second major difference
 * is that steps 3 and 4 are slightly different so that the splits are
 * interleaved.
 */
at::Tensor rearrange_weights_2d(const Tensor& weight_in, bool tconv) {
  at::Tensor weight = weight_in.clone();

  // Flip values along the H and W axes for transposed convolutions
  if (tconv) {
    weight = weight.flip(3).flip(2);
  }

  uint32_t N = get_dim<DimConv2DKernel::OutChannels>(weight);
  uint32_t C = get_dim<DimConv2DKernel::InChannels>(weight);
  uint32_t H = get_dim<DimConv2DKernel::Height>(weight);
  uint32_t W = get_dim<DimConv2DKernel::Width>(weight);

  uint32_t N_aligned = api::utils::align_up(N, 4u);
  uint32_t C_aligned = api::utils::align_up(C, 4u);

  // Add padding to the N and C dimensions so that it's a multiple of 4
  uint32_t C_padding_needed = C_aligned - C;
  uint32_t N_padding_needed = N_aligned - N;
  weight = at::pad(
      weight,
      {0, 0, 0, 0, 0, C_padding_needed, 0, N_padding_needed},
      "constant",
      0);

  // Split the C dim into groups of 4
  uint32_t C4 = C_aligned / 4u;
  weight = weight.reshape({N_aligned, C4, 4, H, W});

  if (!tconv) {
    // Collapse each group of 4 channels onto the width axis
    weight = weight.permute({0, 1, 3, 4, 2}).reshape({N_aligned, C4, H, 4 * W});
    // Next collapse each group of four onto the width axis
    weight =
        weight.permute({0, 2, 1, 3}).reshape({N_aligned, H, C_aligned * W});
  } else {
    // For tconv, do the same thing as above but we want to interleave batches
    // of 4 from each of the channels
    weight = weight.permute({0, 3, 4, 1, 2}).reshape({N_aligned, H, W, 4 * C4});
    // Next reshape to combine the last two dims into a single row
    weight = weight.reshape({N_aligned, H, C_aligned * W});
  }

  // Split the N dim into groups of 4
  uint32_t N4 = N_aligned / 4u;
  weight = weight.reshape({N4, 4, H, C_aligned * W});

  // Collapse the outermost dim so that each group of 4 is stacked vertically
  weight = weight.permute({1, 0, 2, 3}).reshape({4, N4 * H, C_aligned * W});

  return weight.contiguous();
}

/*
 * Rearranges a convolution weight tensor to a layout that can be used by
 * convolution compute shaders. The goal of this packing is to arrange the data
 * such that data access in the compute shader is as linear as possible. The
 * reasoning behind the packing pattern will be described in the shader kernel
 * code.
 *
 * The rearrangement structure is quite straightforward. Essentially we are
 * taking each texel and arranging them along the x axis.
 */
at::Tensor rearrange_bias(
    const std::optional<Tensor>& bias_in,
    const at::Tensor& weight_in,
    bool tconv) {
  const auto cpu_options = weight_in.options().device(c10::Device(c10::DeviceType::CPU));

  // If optional is empty, just return zeros
  if (!has_bias(bias_in)) {
    uint32_t L = tconv ? get_dim<DimTConv2DKernel::OutChannels>(weight_in)
                       : get_dim<DimConv2DKernel::OutChannels>(weight_in);
    const uint32_t L4 = api::utils::div_up(L, 4u);

    at::Tensor bias = at::zeros({4, 1, L4}, cpu_options);
    return bias;
  }

  at::Tensor bias = bias_in->is_vulkan() ? bias_in->cpu() : bias_in->clone();

  // Bias should just be a 1D tensor
  uint32_t L = get_dim<Dim1D::Length>(bias);

  uint32_t L_aligned = api::utils::align_up(L, 4u);

  // Add padding so that the length is a multiple of 4
  uint32_t padding_needed = L_aligned - L;
  bias = at::pad(bias, {0, padding_needed}, "constant", 0);

  // Reshape + permute to group every 4 consecutive elements along the same
  // channel
  uint32_t L4 = L_aligned / 4u;
  bias = bias.reshape({L4, 4}).permute({1, 0});
  bias = bias.reshape({4, 1, L4});

  return bias.contiguous();
}

//
// Shader and Workgroup size determination
//

static api::ShaderInfo get_shader(
    const IntArrayRef kernel_size,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const Conv2dMethod method,
    const bool transposed,
    const bool quantized) {
  api::ShaderInfo shader;

  if (quantized) {
    if (transposed) {
      shader = VK_KERNEL(quantized_conv_transpose2d);
      return shader;
    }

    switch (method) {
      case Conv2dSlidingWindow:
        shader = VK_KERNEL(quantized_conv2d);
        break;
      case Conv2dDepthwise:
        shader = VK_KERNEL(quantized_conv2d_dw);
        break;
      case Conv2dPointwise:
        shader = VK_KERNEL(quantized_conv2d_pw_2x2);
        break;
        // todo fail for quantized transposed conv
    }
    return shader;
  }

  if (transposed) {
    shader = VK_KERNEL(conv_transpose2d);
    return shader;
  }

  switch (method) {
    case Conv2dSlidingWindow:
      shader = VK_KERNEL(conv2d);
      break;
    case Conv2dDepthwise:
      shader = VK_KERNEL(conv2d_dw);
      if (kernel_size.size() == 4 && kernel_size[2] == 3 &&
          kernel_size[3] == 3) {
        // 1x1 refers to the output tile size
        shader = VK_KERNEL(conv2d_dw_output_tile_3x3);
      }
      if (kernel_size.size() == 4 && kernel_size[2] == 5 &&
          kernel_size[3] == 5) {
        // 1x1 refers to the output tile size
        shader = VK_KERNEL(conv2d_dw_output_tile_5x5);
      }
      break;
    case Conv2dPointwise:
      shader = VK_KERNEL(conv2d_pw_output_tile_2x2);
      break;
  }
  return shader;
}

//
// Op Recording
//

struct Params final {
  api::utils::ivec3 out_extents;
  int32_t fill0;
  api::utils::ivec3 in_extents;
  int32_t fill1;
  api::utils::ivec4 overlay_region;
  api::utils::ivec2 kernel_size;
  api::utils::ivec2 stride;
  api::utils::ivec2 padding;
  api::utils::ivec2 dilate;
  api::utils::vec2 clamp;
};

static void record_op(
    api::Context* const context,
    api::ShaderInfo& compute_shader,
    vTensor& v_output,
    const vTensor& v_input,
    const vTensor& v_weight,
    const vTensor& v_bias,
    const IntArrayRef overlay_region,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const float output_min,
    const float output_max,
    const IntArrayRef kernel_size,
    const Conv2dMethod method,
    const bool transposed) {
  api::PipelineBarrier pipeline_barrier{};

  api::utils::uvec3 global_size = v_output.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  Params block{
      api::utils::make_ivec3(v_output.extents()),
      0u,
      api::utils::make_ivec3(v_input.extents()),
      0u,
      utils::make_ivec4(overlay_region, /*reverse=*/true),
      utils::make_ivec2({kernel_size[3], kernel_size[2]}),
      utils::make_ivec2(stride, /*reverse=*/true),
      utils::make_ivec2(padding, /*reverse=*/true),
      utils::make_ivec2(dilation, /*reverse=*/true),
      {output_min, output_max},
  };
  api::UniformParamsBuffer params(context, block);

  context->submit_compute_job(
      // shader descriptor
      compute_shader,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_size,
      // local work group size
      local_size,
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());
}

struct QParams final {
  api::utils::vec4 scales;
  api::utils::ivec4 zero_points;
  api::utils::ivec3 out_extents;
  int32_t fill0;
  api::utils::ivec3 in_extents;
  int32_t fill1;
  api::utils::ivec4 overlay_region;
  api::utils::ivec2 kernel_size;
  api::utils::ivec2 stride;
  api::utils::ivec2 padding;
  api::utils::ivec2 dilate;
  api::utils::vec2 clamp;
};

static void record_quantized_op(
    api::Context* const context,
    api::ShaderInfo& compute_shader,
    vTensor& v_output,
    const vTensor& v_input,
    const vTensor& v_weight,
    const vTensor& v_bias,
    const IntArrayRef overlay_region,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const float output_min,
    const float output_max,
    const IntArrayRef kernel_size,
    const Conv2dMethod method,
    const bool transposed) {
  api::PipelineBarrier pipeline_barrier{};

  api::utils::uvec3 global_size = v_output.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  QParams block{
      {
          v_output.get_scale_float(),
          v_input.get_scale_float(),
          v_weight.get_scale_float(),
          v_bias.get_scale_float(),
      },
      {
          v_output.get_zero_point_int32(),
          v_input.get_zero_point_int32(),
          v_weight.get_zero_point_int32(),
          v_bias.get_zero_point_int32(),
      },
      api::utils::make_ivec3(v_output.extents()),
      0u,
      api::utils::make_ivec3(v_input.extents()),
      0u,
      utils::make_ivec4(overlay_region, /*reverse=*/true),
      utils::make_ivec2({kernel_size[3], kernel_size[2]}),
      utils::make_ivec2(stride, /*reverse=*/true),
      utils::make_ivec2(padding, /*reverse=*/true),
      utils::make_ivec2(dilation, /*reverse=*/true),
      {output_min, output_max},
  };
  api::UniformParamsBuffer params(context, block);

  context->submit_compute_job(
      // shader descriptor
      compute_shader,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_size,
      // local work group size
      local_size,
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());
}

} // namespace conv2d

namespace {

using namespace api::utils;

const std::string& conv_pack_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_CONV_CACHE_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool conv_pack_logging_enabled() {
  return !conv_pack_log_path().empty();
}

struct ConvPackLogState final {
  std::atomic<uint64_t> vulkan_pack_weights{0u};
  std::atomic<uint64_t> vulkan_to_cpu_copies{0u};

  ~ConvPackLogState() {
    if (!conv_pack_logging_enabled()) {
      return;
    }

    std::ofstream out(conv_pack_log_path(), std::ios::app);
    out << "conv_pack: vulkan_pack_weights="
        << vulkan_pack_weights.load(std::memory_order_relaxed)
        << " vulkan_to_cpu_copies="
        << vulkan_to_cpu_copies.load(std::memory_order_relaxed) << '\n';
  }
};

ConvPackLogState& conv_pack_log_state() {
  static ConvPackLogState state;
  return state;
}

Tensor copy_vulkan_tensor_to_cpu(const Tensor& src) {
  if (!src.is_vulkan()) {
    return src;
  }

  if (conv_pack_logging_enabled()) {
    conv_pack_log_state().vulkan_to_cpu_copies.fetch_add(
        1u, std::memory_order_relaxed);
  }
  report_vulkan_cpu_fallback(
      "vulkan_prepack::conv2d_context",
      "vulkan_weight_cpu_materialization",
      {src},
      VulkanCpuFallbackKind::SyncReadback);

  if (convert(src).storage_type() == api::StorageType::BUFFER) {
    return src.cpu();
  }

  Tensor dst;
  transfer_vulkan_to_cpu(convert(src), dst);
  return dst;
}

vTensor pack_weights(
    const Tensor& weight_inp,
    const bool transposed,
    const bool quantized,
    const Conv2dMethod conv_method) {
  if (conv_pack_logging_enabled() && weight_inp.is_vulkan()) {
    conv_pack_log_state().vulkan_pack_weights.fetch_add(
        1u, std::memory_order_relaxed);
  }

  // Raw Vulkan module weights are not in the shader-packed layout that the
  // convolution kernels expect. Re-materialize them on CPU first so they go
  // through the same rearrangement path as CPU-resident weights.
  const Tensor weight_source = copy_vulkan_tensor_to_cpu(weight_inp);
  Tensor weight_arg =
      quantized ? at::dequantize(weight_source) : weight_source;
  if (
      !quantized &&
      (weight_arg.scalar_type() == kBFloat16 ||
       weight_arg.scalar_type() == kHalf)) {
    weight_arg = weight_arg.to(kFloat);
  }

  const Tensor weight = transposed
      ? at::permute(weight_arg, {1, 0, 2, 3}).contiguous()
      : weight_arg.contiguous();

  at::Tensor weight_rearranged;
  if (conv_method == Conv2dDepthwise) {
    weight_rearranged = conv2d::rearrange_weights_dw(weight);
  } else {
    weight_rearranged = conv2d::rearrange_weights_2d(weight, transposed);
  }

  vTensor v_weight{
      api::context(),
      weight_rearranged.sizes().vec(),
      convert_dtype(weight_rearranged.scalar_type()),
      api::StorageType::TEXTURE_2D,
  };

  pack_cpu_to_vulkan(weight_rearranged, v_weight);

  return v_weight;
}

vTensor pack_biases(
    const std::optional<Tensor>& bias,
    const Tensor& weight,
    const bool transposed,
    const bool quantized) {
  at::Tensor bias_arg = conv2d::rearrange_bias(bias, weight, transposed);
  at::Tensor bias_rearranged =
      (quantized &&
       (bias_arg.scalar_type() == kQUInt8 || bias_arg.scalar_type() == kQInt8 ||
        bias_arg.scalar_type() == kQInt32))
      ? at::dequantize(bias_arg)
      : bias_arg;
  if (
      !quantized &&
      (bias_rearranged.scalar_type() == kBFloat16 ||
       bias_rearranged.scalar_type() == kHalf)) {
    bias_rearranged = bias_rearranged.to(kFloat);
  }

  vTensor v_bias{
      api::context(),
      bias_rearranged.sizes().vec(),
      convert_dtype(bias_rearranged.scalar_type()),
      api::StorageType::TEXTURE_2D,
  };

  pack_cpu_to_vulkan(bias_rearranged, v_bias);

  return v_bias;
}

/*
 * Computes the size of the overlay region when computing a convolution output.
 */
std::array<int64_t, 4> compute_overlay_region(
    const Tensor& weight,
    const IntArrayRef dilation,
    const bool transposed) {
  const IntArrayRef filter = weight.sizes();

  const auto overlay_length = [](const int64_t k, const int64_t d) {
    return k + (k - 1) * (d - 1);
  };

  return {
      align_up(
          transposed ? filter[Layout::TransposedFilter::output]
                     : filter[Layout::Filter::output],
          INT64_C(4)),
      align_up(
          transposed ? filter[Layout::TransposedFilter::input]
                     : filter[Layout::Filter::input],
          INT64_C(4)),
      overlay_length(
          filter[Layout::Filter::height], dilation[Layout::Parameter::height]),
      overlay_length(
          filter[Layout::Filter::width], dilation[Layout::Parameter::width]),
  };
}

std::array<int64_t, 2> pack_params(const std::vector<int64_t>& vector) {
  TORCH_INTERNAL_ASSERT(2u == vector.size(), "Invalid usage!");

  return {
      vector[0],
      vector[1],
  };
}

bool weight_valid(const Tensor& weight, const bool quantized) {
  if (4 != weight.ndimension()) {
    return false;
  }
  if (get_dim<DimConv2DKernel::Height>(weight) == 0) {
    return false;
  }
  if (get_dim<DimConv2DKernel::Width>(weight) == 0) {
    return false;
  }
  if (!weight.device().is_cpu() &&
      weight.device().type() != c10::DeviceType::Vulkan) {
    return false;
  }
  if (quantized &&
      (weight.scalar_type() != c10::kQUInt8 &&
       weight.scalar_type() != c10::kQInt8)) {
    return false;
  }

  return true;
}

bool bias_valid(
    const std::optional<Tensor>& bias,
    const Tensor& weight,
    const bool transposed,
    const bool quantized) {
  if (!conv2d::has_bias(bias)) {
    return true;
  }

  if (bias->ndimension() != 1) {
    return false;
  }
  if (!bias->device().is_cpu() &&
      bias->device().type() != c10::DeviceType::Vulkan) {
    return false;
  }
  uint32_t L = get_dim<Dim1D::Length>(*bias);
  uint32_t OC = transposed ? get_dim<DimTConv2DKernel::OutChannels>(weight)
                           : get_dim<DimConv2DKernel::OutChannels>(weight);
  if (L != OC) {
    return false;
  }

  return true;
}

bool available(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const bool transposed,
    const bool quantized,
    const IntArrayRef /* output_padding */,
    const int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  if (!weight_valid(weight, quantized)) {
    return false;
  }
  if (!bias_valid(bias, weight, transposed, quantized)) {
    return false;
  }
  if (get_dim<Dim4D::Height>(stride) == 0 ||
      get_dim<Dim4D::Width>(stride) == 0) {
    return false;
  }
  if (transposed) {
    if (get_dim<Dim4D::Height>(dilation) != 1 ||
        get_dim<Dim4D::Width>(dilation) != 1) {
      return false;
    }
  } else {
    if (get_dim<Dim4D::Height>(dilation) == 0 ||
        get_dim<Dim4D::Width>(dilation) == 0) {
      return false;
    }
  }
  if (groups <= 0) {
    return false;
  }
  if (transposed) {
    if ((get_dim<DimTConv2DKernel::OutChannels>(weight) % groups) != 0) {
      return false;
    }
  } else {
    if ((get_dim<DimConv2DKernel::OutChannels>(weight) % groups) != 0) {
      return false;
    }
  }
  if (get_dim<DimConv2DKernel::InChannels>(weight) == 0 ||
      get_dim<DimConv2DKernel::OutChannels>(weight) == 0) {
    return false;
  }
  if (output_min && !output_min->isFloatingPoint()) {
    return false;
  }
  if (output_max && !output_max->isFloatingPoint()) {
    return false;
  }
  return true;
}

bool usable(const Tensor& input, const bool quantized) {
  if (input.ndimension() != 4) {
    return false;
  }
  if (input.device().type() != c10::DeviceType::Vulkan) {
    return false;
  }
  if (!quantized && input.scalar_type() != at::kFloat) {
    return false;
  }
  if (quantized && input.scalar_type() != c10::kQUInt8) {
    return false;
  }
  if (get_dim<Dim4D::Batch>(input) == 0) {
    return false;
  }
  if (get_dim<Dim4D::Channel>(input) == 0) {
    return false;
  }
  if (get_dim<Dim4D::Height>(input) == 0) {
    return false;
  }
  if (get_dim<Dim4D::Width>(input) == 0) {
    return false;
  }
  if (input.requires_grad()) {
    return false;
  }

  return true;
}

static inline std::vector<int64_t> get_conv_transpose_output_size(
    IntArrayRef input_size,
    IntArrayRef weight_size,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation = IntArrayRef()) {
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_input_channels_dim];
  for (const auto d : c10::irange(2, dim)) {
    output_size[d] = stride[d - 2] * (input_size[d] - 1) + weight_size[d] -
        2 * padding[d - 2] + output_padding[d - 2];
  }
  return output_size;
}

bool output_padding_is_zero(const IntArrayRef output_padding) {
  for (const auto value : output_padding) {
    if (value != 0) {
      return false;
    }
  }
  return true;
}

bool is_float_or_half_conv_tensor(const Tensor& tensor) {
  return tensor.scalar_type() == kFloat || tensor.scalar_type() == kHalf;
}

Tensor upcast_half_conv_tensor_for_packing(const Tensor& tensor) {
  const Tensor source = tensor.requires_grad() ? tensor.detach() : tensor;
  if (source.scalar_type() == kFloat) {
    return source;
  }

  TORCH_CHECK(
      source.scalar_type() == kHalf,
      "Vulkan float buffer conv prepack expects float or half tensors");

  if (source.is_vulkan()) {
    return utils::cast_vulkan_tensor_dtype(source, kFloat);
  }

  return source.to(kFloat);
}

std::optional<Tensor> upcast_half_conv_tensor_for_packing(
    const std::optional<Tensor>& tensor) {
  if (!tensor || !tensor->defined()) {
    return tensor;
  }
  return upcast_half_conv_tensor_for_packing(*tensor);
}

Tensor upload_conv_tensor_to_buffer(
    const Tensor& tensor,
    const api::GPUMemoryLayout memory_layout) {
  const Tensor source = tensor.requires_grad() ? tensor.detach() : tensor;

  if (source.is_vulkan()) {
    const vTensor& v_source = convert(source);
    Tensor buffer_source =
        v_source.storage_type() == api::StorageType::BUFFER &&
            v_source.gpu_memory_layout() == memory_layout
        ? source
        : utils::ensure_buffer_storage(source, memory_layout);
    return utils::mark_tensor_execution(
        buffer_source, api::ExecutionLayout::BUFFER_DIRECT, true);
  }

  TORCH_CHECK(
      source.device().is_cpu(),
      "Vulkan float buffer conv prepack expects CPU or Vulkan tensors");
  vTensor v_buffer{
      api::context(),
      source.sizes().vec(),
      convert_dtype(source.scalar_type()),
      api::StorageType::BUFFER,
      memory_layout,
  };
  pack_cpu_to_vulkan(source, v_buffer);
  return utils::mark_tensor_execution(
      convert(v_buffer), api::ExecutionLayout::BUFFER_DIRECT, true);
}

bool can_use_float_buffer_conv2d_prepack(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const bool transposed,
    const bool quantized,
    const IntArrayRef output_padding) {
  if (
      quantized ||
      weight.dim() != 4 ||
      !is_float_or_half_conv_tensor(weight)) {
    return false;
  }

  if (!transposed && !output_padding_is_zero(output_padding)) {
    return false;
  }

  if (bias && bias->defined()) {
    if (bias->dim() > 2 || !is_float_or_half_conv_tensor(*bias)) {
      return false;
    }
  }

  return true;
}

bool should_force_image_conv_for_small_metadata_input(const Tensor& input) {
  if (
      !input.is_vulkan() || input.scalar_type() != kFloat || input.dim() != 4 ||
      input.size(1) <= 1 || input.size(1) >= 20) {
    return false;
  }
  const vTensor& v_input = convert(input);
  return v_input.storage_type() == api::StorageType::BUFFER &&
      v_input.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
      !v_input.has_direct_buffer_layout();
}

utils::SmallMetadataPaddedConv2DTensorInfo
small_metadata_padded_conv2d_tensor_info(const Tensor& input) {
  utils::SmallMetadataPaddedConv2DTensorInfo info;
  info.is_vulkan = input.is_vulkan();
  info.dtype = input.scalar_type();
  info.rank = input.dim();
  if (input.dim() == 4) {
    info.batch = input.size(0);
    info.channels = input.size(1);
    info.height = input.size(2);
    info.width = input.size(3);
  }
  if (input.is_vulkan()) {
    const vTensor& v_input = convert(input);
    info.has_buffer_storage =
        v_input.storage_type() == api::StorageType::BUFFER;
    info.is_width_packed =
        v_input.gpu_memory_layout() ==
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED;
    info.has_direct_buffer_layout = v_input.has_direct_buffer_layout();
    info.supports_buffer_compute =
        utils::supports_buffer_elementwise_compute(v_input);
  }
  return info;
}

utils::SmallMetadataPaddedConv2DWeightInfo
small_metadata_padded_conv2d_weight_info(const Tensor& weight) {
  utils::SmallMetadataPaddedConv2DWeightInfo info;
  info.defined = weight.defined();
  info.dtype = weight.defined() ? weight.scalar_type() : kFloat;
  info.rank = weight.defined() ? weight.dim() : 0;
  if (weight.defined() && weight.dim() == 4) {
    info.output_channels = weight.size(0);
    info.input_channels = weight.size(1);
    info.kernel_h = weight.size(2);
    info.kernel_w = weight.size(3);
  }
  return info;
}

utils::SmallMetadataPaddedConv2DOptions small_metadata_padded_conv2d_options(
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const bool transposed,
    const IntArrayRef output_padding,
    const int64_t groups) {
  utils::SmallMetadataPaddedConv2DOptions options;
  options.transposed = transposed;
  options.quantized = false;
  options.groups = groups;
  options.stride_h = stride.size() == 2 ? stride[0] : -1;
  options.stride_w = stride.size() == 2 ? stride[1] : -1;
  options.padding_h = padding.size() == 2 ? padding[0] : -1;
  options.padding_w = padding.size() == 2 ? padding[1] : -1;
  options.dilation_h = dilation.size() == 2 ? dilation[0] : -1;
  options.dilation_w = dilation.size() == 2 ? dilation[1] : -1;
  options.output_padding_is_zero = output_padding_is_zero(output_padding);
  return options;
}

bool should_force_image_conv_for_known_bad_large_buffer_conv(
    const Tensor& input,
    const Tensor& weight,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups) {
  const utils::VulkanDevicePolicy device_policy =
      utils::current_vulkan_device_policy();
  if (
      !device_policy.disable_large_buffer_conv_3x3 ||
      !input.is_vulkan() ||
      input.scalar_type() != kFloat ||
      input.dim() != 4 ||
      weight.dim() != 4 ||
      stride.size() != 2 ||
      padding.size() != 2 ||
      dilation.size() != 2 ||
      groups != 1 ||
      weight.size(2) != 3 ||
      weight.size(3) != 3 ||
      stride[0] != 1 ||
      stride[1] != 1 ||
      padding[0] != 1 ||
      padding[1] != 1 ||
      dilation[0] != 1 ||
      dilation[1] != 1 ||
      input.size(2) * input.size(3) < 18 * 18 ||
      (input.size(1) < 64 && !input.requires_grad())) {
    return false;
  }
  return true;
}

bool can_run_bfloat16_buffer_conv2d(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const bool transposed,
    const bool quantized,
    const IntArrayRef output_padding) {
  if (
      transposed ||
      quantized ||
      !output_padding_is_zero(output_padding) ||
      input.device().type() != c10::DeviceType::Vulkan ||
      weight.device().type() != c10::DeviceType::Vulkan ||
      input.scalar_type() != kBFloat16 ||
      weight.scalar_type() != kBFloat16 ||
      input.dim() != 4 ||
      weight.dim() != 4 ||
      input.requires_grad() ||
      weight.requires_grad()) {
    return false;
  }

  if (
      convert(input).storage_type() != api::StorageType::BUFFER ||
      convert(weight).storage_type() != api::StorageType::BUFFER) {
    return false;
  }

  if (bias && bias->defined()) {
    if (
        bias->dim() > 2 ||
        bias->requires_grad() ||
        (bias->scalar_type() != kBFloat16 && bias->scalar_type() != kFloat)) {
      return false;
    }
  }

  return true;
}

Tensor prepare_float_bias_buffer_for_conv2d(
    const std::optional<Tensor>& bias,
    const int64_t out_channels) {
  if (!bias || !bias->defined()) {
    return upload_conv_tensor_to_buffer(
        at::zeros({out_channels}, at::device(at::kCPU).dtype(at::kFloat)),
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  }

  Tensor prepared_bias = *bias;
  if (prepared_bias.is_vulkan()) {
    if (
        prepared_bias.scalar_type() == kHalf ||
        prepared_bias.scalar_type() == kBFloat16) {
      prepared_bias = utils::cast_vulkan_tensor_dtype(prepared_bias, kFloat);
    }
    return utils::mark_tensor_execution(
        utils::ensure_buffer_storage(
            prepared_bias, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
        api::ExecutionLayout::BUFFER_DIRECT,
        true);
  }

  if (
      prepared_bias.scalar_type() == kHalf ||
      prepared_bias.scalar_type() == kBFloat16) {
    prepared_bias = prepared_bias.to(kFloat);
  }
  return upload_conv_tensor_to_buffer(
      prepared_bias, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
}

PackedWeightHandle make_float_buffer_conv2d_handle(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const std::vector<int64_t>& logical_weight_sizes,
    const PackedWeightKind packed_weight_kind,
    const int64_t bias_channels) {
  api::Context* const context = api::context();
  context->submit_pending_work_and_poll_retire(
      api::PendingWorkRetireDrainPolicy::DeferTinyOldPathPending);

  const Tensor pack_source_weight = upcast_half_conv_tensor_for_packing(weight);
  const std::optional<Tensor> pack_source_bias =
      upcast_half_conv_tensor_for_packing(bias);
  Tensor buffer_weight = upload_conv_tensor_to_buffer(
      pack_source_weight, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  Tensor buffer_bias = prepare_float_bias_buffer_for_conv2d(
      pack_source_bias, bias_channels);

  const size_t resident_nbytes =
      convert(buffer_weight).gpu_nbytes() + convert(buffer_bias).gpu_nbytes();
  return PackedWeightHandle(
      std::move(buffer_weight),
      std::move(buffer_bias),
      logical_weight_sizes,
      packed_weight_kind,
      bias && bias->defined(),
      PackedWeightResidencyClass::PersistentInference,
      false,
      api::ExecutionLayout::BUFFER_DIRECT,
      resident_nbytes);
}

bool should_cache_float_buffer_conv2d_handle(
    const PackedWeightHandle& handle,
    const PackedWeightKind packed_weight_kind) {
  const char* const device_name =
      api::context()->adapter_ptr()->physical_device().properties.deviceName;
  if (
      device_name != nullptr && std::strstr(device_name, "GTX") != nullptr &&
      handle.resident_nbytes() > 64u * 1024u) {
    utils::log_vulkan_op_hit(
        std::string(
            "aten::convolution.packed_weight_cache_skip.gtx bytes=") +
        std::to_string(handle.resident_nbytes()) + " weight=" +
        conv2d::format_conv_sizes(handle.logical_weight_sizes()));
    return false;
  }
  // Large eager 3x3 diffusion/decoder weights are often touched once per frame.
  // Keeping all of them in the persistent packed-weight cache creates live
  // memory pressure without producing hits; let normal Vulkan deferred cleanup
  // own their in-flight lifetime instead.
  constexpr size_t kLargeSlidingWindowConvCacheLimitBytes =
      size_t{2} * 1024u * 1024u;
  if (
      packed_weight_kind == PackedWeightKind::Conv2dSlidingWindow &&
      handle.resident_nbytes() > kLargeSlidingWindowConvCacheLimitBytes) {
    utils::log_vulkan_op_hit(
        std::string("aten::convolution.packed_weight_cache_skip.large bytes=") +
        std::to_string(handle.resident_nbytes()) + " weight=" +
        conv2d::format_conv_sizes(handle.logical_weight_sizes()));
    return false;
  }
  return true;
}

bool is_gtx_class_runtime_device() {
  const char* const device_name =
      api::context()->adapter_ptr()->physical_device().properties.deviceName;
  return device_name != nullptr && std::strstr(device_name, "GTX") != nullptr;
}

void maybe_sync_after_gtx_large_buffer_conv(
    api::Context* const context,
    const vTensor& v_output) {
  constexpr size_t kGtxLargeConvSyncBytes = 128u * 1024u * 1024u;
  if (
      is_gtx_class_runtime_device() &&
    v_output.gpu_nbytes() >= kGtxLargeConvSyncBytes) {
    utils::log_vulkan_op_hit("aten::convolution.gtx_large_buffer_sync");
    context->synchronize_device();
  }
}

bool can_run_float_buffer_conv2d(
    const Tensor& input,
    const PackedWeightHandle& packed_weight,
    const bool transposed,
    const bool quantized,
    const IntArrayRef output_padding) {
  if (
      transposed ||
      quantized ||
      !output_padding_is_zero(output_padding) ||
      input.device().type() != c10::DeviceType::Vulkan ||
      input.scalar_type() != kFloat ||
      input.dim() != 4 ||
      !packed_weight.defined() ||
      packed_weight.execution_layout() != api::ExecutionLayout::BUFFER_DIRECT ||
      packed_weight.quantized()) {
    return false;
  }

  const vTensor& v_input = convert(input);
  if (v_input.storage_type() != api::StorageType::BUFFER) {
    return false;
  }

  const vTensor& v_weight = packed_weight.weight_vtensor();
  if (
      v_weight.storage_type() != api::StorageType::BUFFER ||
      v_weight.dtype() != api::kFloat) {
    return false;
  }

  const vTensor& v_bias = packed_weight.bias_vtensor();
  if (
      v_bias.storage_type() != api::StorageType::BUFFER ||
      v_bias.dtype() != api::kFloat) {
    return false;
  }

  return true;
}

bool can_run_float_buffer_conv_transpose2d(
    const Tensor& input,
    const PackedWeightHandle& packed_weight,
    const bool transposed,
    const bool quantized) {
  if (
      !transposed ||
      quantized ||
      input.device().type() != c10::DeviceType::Vulkan ||
      input.scalar_type() != kFloat ||
      input.dim() != 4 ||
      !packed_weight.defined() ||
      packed_weight.execution_layout() != api::ExecutionLayout::BUFFER_DIRECT ||
      packed_weight.quantized()) {
    return false;
  }

  const vTensor& v_input = convert(input);
  if (v_input.storage_type() != api::StorageType::BUFFER) {
    return false;
  }

  const vTensor& v_weight = packed_weight.weight_vtensor();
  if (
      v_weight.storage_type() != api::StorageType::BUFFER ||
      v_weight.dtype() != api::kFloat ||
      packed_weight.logical_weight_sizes().size() != 4) {
    return false;
  }

  const vTensor& v_bias = packed_weight.bias_vtensor();
  if (
      v_bias.storage_type() != api::StorageType::BUFFER ||
      v_bias.dtype() != api::kFloat) {
    return false;
  }

  return true;
}

const char* float_buffer_conv_transpose2d_skip_reason(
    const Tensor& input,
    const PackedWeightHandle& packed_weight,
    const bool transposed,
    const bool quantized) {
  if (!transposed) {
    return "aten::convolution.buffer_float_transpose_skip.not_transposed";
  }
  if (quantized) {
    return "aten::convolution.buffer_float_transpose_skip.quantized";
  }
  if (input.device().type() != c10::DeviceType::Vulkan) {
    return "aten::convolution.buffer_float_transpose_skip.input_not_vulkan";
  }
  if (input.scalar_type() != kFloat) {
    return "aten::convolution.buffer_float_transpose_skip.input_not_float";
  }
  if (input.dim() != 4) {
    return "aten::convolution.buffer_float_transpose_skip.input_not_4d";
  }
  if (!packed_weight.defined()) {
    return "aten::convolution.buffer_float_transpose_skip.no_packed_weight";
  }
  if (packed_weight.execution_layout() != api::ExecutionLayout::BUFFER_DIRECT) {
    return "aten::convolution.buffer_float_transpose_skip.weight_not_buffer_direct";
  }
  if (packed_weight.quantized()) {
    return "aten::convolution.buffer_float_transpose_skip.weight_quantized";
  }

  const vTensor& v_input = convert(input);
  if (v_input.storage_type() != api::StorageType::BUFFER) {
    return "aten::convolution.buffer_float_transpose_skip.input_not_buffer";
  }

  const vTensor& v_weight = packed_weight.weight_vtensor();
  if (v_weight.storage_type() != api::StorageType::BUFFER) {
    return "aten::convolution.buffer_float_transpose_skip.weight_not_buffer";
  }
  if (v_weight.dtype() != api::kFloat) {
    return "aten::convolution.buffer_float_transpose_skip.weight_not_float";
  }
  if (packed_weight.logical_weight_sizes().size() != 4) {
    return "aten::convolution.buffer_float_transpose_skip.weight_bad_rank";
  }

  const vTensor& v_bias = packed_weight.bias_vtensor();
  if (v_bias.storage_type() != api::StorageType::BUFFER) {
    return "aten::convolution.buffer_float_transpose_skip.bias_not_buffer";
  }
  if (v_bias.dtype() != api::kFloat) {
    return "aten::convolution.buffer_float_transpose_skip.bias_not_float";
  }

  return nullptr;
}

bool can_use_float_buffer_nonoverlap_conv_transpose2d(
    const PackedWeightHandle& packed_weight,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const IntArrayRef output_padding) {
  if (
      stride.size() != 2 || padding.size() != 2 || dilation.size() != 2 ||
      !output_padding_is_zero(output_padding)) {
    return false;
  }

  if (
      padding[0] != 0 || padding[1] != 0 || dilation[0] != 1 ||
      dilation[1] != 1) {
    return false;
  }

  const auto& logical_weight_sizes = packed_weight.logical_weight_sizes();
  return get_dim<DimTConv2DKernel::Height>(logical_weight_sizes) == stride[0] &&
      get_dim<DimTConv2DKernel::Width>(logical_weight_sizes) == stride[1];
}

bool might_match_no_overlap_conv_transpose2d_contract(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context) {
  if (
      input.device().type() != c10::DeviceType::Vulkan ||
      input.scalar_type() != kFloat || input.dim() != 4 ||
      input.size(0) != 1 || input.size(1) < 64 ||
      !conv_context->transposed() || conv_context->quantized() ||
      conv_context->groups() != 1) {
    return false;
  }

  const auto& stride = conv_context->stride();
  const auto& padding = conv_context->padding();
  const auto& dilation = conv_context->dilation();
  const auto& output_padding = conv_context->output_padding();
  if (
      stride.size() != 2 || padding.size() != 2 || dilation.size() != 2 ||
      stride[0] != 2 || stride[1] != 2 ||
      padding[0] != 0 || padding[1] != 0 ||
      dilation[0] != 1 || dilation[1] != 1 ||
      !output_padding_is_zero(output_padding)) {
    return false;
  }

  const PackedWeightHandle& packed_weight = conv_context->packed_weight();
  if (
      !packed_weight.defined() ||
      packed_weight.execution_layout() != api::ExecutionLayout::BUFFER_DIRECT ||
      packed_weight.quantized()) {
    return false;
  }

  const auto& logical_weight_sizes = packed_weight.logical_weight_sizes();
  return logical_weight_sizes.size() == 4 &&
      get_dim<DimTConv2DKernel::InChannels>(logical_weight_sizes) ==
      input.size(1) &&
      get_dim<DimTConv2DKernel::Height>(logical_weight_sizes) == 2 &&
      get_dim<DimTConv2DKernel::Width>(logical_weight_sizes) == 2;
}

utils::NoOverlapConvTranspose2DTensorInfo
no_overlap_conv_transpose2d_tensor_info(const Tensor& input) {
  utils::NoOverlapConvTranspose2DTensorInfo info;
  info.is_vulkan = input.is_vulkan();
  info.dtype = input.scalar_type();
  info.rank = input.dim();
  if (input.dim() == 4) {
    info.batch = input.size(0);
    info.channels = input.size(1);
  }
  if (input.is_vulkan()) {
    const vTensor& v_input = convert(input);
    info.has_buffer_storage =
        v_input.storage_type() == api::StorageType::BUFFER;
    info.supports_buffer_compute =
        utils::supports_buffer_elementwise_compute(v_input);
  }
  return info;
}

utils::NoOverlapConvTranspose2DPackedInfo
no_overlap_conv_transpose2d_packed_info(
    const PackedWeightHandle& packed_weight) {
  utils::NoOverlapConvTranspose2DPackedInfo info;
  info.defined = packed_weight.defined();
  if (!packed_weight.defined()) {
    return info;
  }

  info.execution_is_buffer_direct =
      packed_weight.execution_layout() == api::ExecutionLayout::BUFFER_DIRECT;
  info.quantized = packed_weight.quantized();
  const auto& logical_weight_sizes = packed_weight.logical_weight_sizes();
  info.weight_rank = logical_weight_sizes.size();
  if (logical_weight_sizes.size() == 4) {
    info.input_channels =
        get_dim<DimTConv2DKernel::InChannels>(logical_weight_sizes);
    info.output_channels =
        get_dim<DimTConv2DKernel::OutChannels>(logical_weight_sizes);
    info.kernel_h = get_dim<DimTConv2DKernel::Height>(logical_weight_sizes);
    info.kernel_w = get_dim<DimTConv2DKernel::Width>(logical_weight_sizes);
  }

  const vTensor& v_weight = packed_weight.weight_vtensor();
  info.weight_dtype =
      v_weight.dtype() == api::kFloat ? kFloat : ScalarType::Undefined;
  info.weight_has_buffer_storage =
      v_weight.storage_type() == api::StorageType::BUFFER;
  const vTensor& v_bias = packed_weight.bias_vtensor();
  info.bias_has_buffer_storage =
      v_bias.storage_type() == api::StorageType::BUFFER;
  info.bias_is_float = v_bias.dtype() == api::kFloat;
  return info;
}

utils::NoOverlapConvTranspose2DOptions no_overlap_conv_transpose2d_options(
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context) {
  utils::NoOverlapConvTranspose2DOptions options;
  options.transposed = conv_context->transposed();
  options.quantized = conv_context->quantized();
  options.groups = conv_context->groups();
  const auto& stride = conv_context->stride();
  const auto& padding = conv_context->padding();
  const auto& dilation = conv_context->dilation();
  if (stride.size() == 2) {
    options.stride_h = stride[0];
    options.stride_w = stride[1];
  }
  if (padding.size() == 2) {
    options.padding_h = padding[0];
    options.padding_w = padding[1];
  }
  if (dilation.size() == 2) {
    options.dilation_h = dilation[0];
    options.dilation_w = dilation[1];
  }
  options.output_padding_is_zero =
      output_padding_is_zero(conv_context->output_padding());
  return options;
}

bool can_run_exact_pointwise_nooverlap_conv_transpose2d(
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context) {
  if (
      !conv_context->transposed() || conv_context->quantized() ||
      conv_context->groups() != 1) {
    return false;
  }

  const auto& stride = conv_context->stride();
  const auto& padding = conv_context->padding();
  const auto& dilation = conv_context->dilation();
  const auto& output_padding = conv_context->output_padding();
  if (
      stride.size() != 2 || padding.size() != 2 || dilation.size() != 2 ||
      !output_padding_is_zero(output_padding)) {
    return false;
  }

  if (
      padding[0] != 0 || padding[1] != 0 || dilation[0] != 1 ||
      dilation[1] != 1) {
    return false;
  }

  const auto& logical_weight_sizes =
      conv_context->packed_weight().logical_weight_sizes();
  if (logical_weight_sizes.size() != 4) {
    return false;
  }

  // The exact rearrange path rebuilds a synthetic pointwise weight on every
  // invocation. Keep that route for smaller transposed convolutions, but hand
  // larger decoder-style shapes to the prepacked nonoverlap shader instead.
  const int64_t out_channels =
      get_dim<DimTConv2DKernel::OutChannels>(logical_weight_sizes);
  const int64_t kernel_h = get_dim<DimTConv2DKernel::Height>(logical_weight_sizes);
  const int64_t kernel_w = get_dim<DimTConv2DKernel::Width>(logical_weight_sizes);
  const int64_t expanded_pointwise_channels = out_channels * kernel_h * kernel_w;
  constexpr int64_t kExactRearrangeMaxExpandedChannels = 256;
  if (expanded_pointwise_channels > kExactRearrangeMaxExpandedChannels) {
    return false;
  }

  return
      get_dim<DimTConv2DKernel::Height>(logical_weight_sizes) == stride[0] &&
      get_dim<DimTConv2DKernel::Width>(logical_weight_sizes) == stride[1];
}

Tensor run_exact_pointwise_nooverlap_conv_transpose2d(
    const Tensor& input_arg,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context,
    const float output_min,
    const float output_max,
    Tensor* output_arg) {
  utils::log_vulkan_op_hit(
      "aten::convolution.buffer_float_transpose_exact_rearrange");

  const c10::impl::GenericList unpacked = conv_context->unpack();
  const Tensor weight =
      unpacked.get(Conv2dPackedContext::Unpacked::Weight).toTensor();
  const std::optional<Tensor> bias =
      get_optional_tensor(unpacked, Conv2dPackedContext::Unpacked::Bias);

  const int64_t out_channels = weight.size(1);
  const int64_t kernel_h = weight.size(2);
  const int64_t kernel_w = weight.size(3);

  const Tensor pointwise_weight =
      weight.permute({1, 2, 3, 0})
          .reshape(
              {out_channels * kernel_h * kernel_w, weight.size(0), 1, 1})
          .contiguous();
  const std::optional<Tensor> no_bias = std::nullopt;
  Tensor patches = at::conv2d(
      input_arg,
      pointwise_weight,
      no_bias,
      IntArrayRef{1, 1},
      IntArrayRef{0, 0},
      IntArrayRef{1, 1},
      1);

  Tensor output = patches.view(
      {patches.size(0),
       out_channels,
       kernel_h,
       kernel_w,
       patches.size(2),
       patches.size(3)});
  output = output.permute({0, 1, 4, 2, 5, 3}).reshape(
      {patches.size(0),
       out_channels,
       patches.size(2) * kernel_h,
       patches.size(3) * kernel_w});

  if (bias && bias->defined()) {
    Tensor bias_term = bias->is_vulkan() ? *bias : bias->to(input_arg.device());
    output = output.add(bias_term.view({1, out_channels, 1, 1}));
  }

  output = output.clamp(output_min, output_max);
  if (output_arg != nullptr) {
    copy_(*output_arg, output);
    return *output_arg;
  }
  return output;
}

enum class FloatBufferConv2dShaderKind {
  Generic,
  Pointwise1x1,
  Kernel3x3Stride1Pad1,
  Kernel3x3Stride2Pad0,
  Kernel3x3Stride2Pad1,
};

FloatBufferConv2dShaderKind select_float_buffer_conv2d_shader_kind(
    const PackedWeightHandle& packed_weight,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups) {
  if (
      groups != 1 || stride.size() != 2 || padding.size() != 2 ||
      dilation.size() != 2 || dilation[0] != 1 || dilation[1] != 1) {
    return FloatBufferConv2dShaderKind::Generic;
  }

  const auto& logical_weight_sizes = packed_weight.logical_weight_sizes();
  if (logical_weight_sizes.size() != 4) {
    return FloatBufferConv2dShaderKind::Generic;
  }

  const int64_t kernel_h = get_dim<DimConv2DKernel::Height>(logical_weight_sizes);
  const int64_t kernel_w = get_dim<DimConv2DKernel::Width>(logical_weight_sizes);
  const int64_t out_channels =
      get_dim<DimConv2DKernel::OutChannels>(logical_weight_sizes);
  const int64_t in_channels =
      get_dim<DimConv2DKernel::InChannels>(logical_weight_sizes);
  if (
      kernel_h == 1 && kernel_w == 1 && stride[0] == 1 && stride[1] == 1 &&
      padding[0] == 0 && padding[1] == 0) {
    return FloatBufferConv2dShaderKind::Pointwise1x1;
  }

  if (kernel_h == 3 && kernel_w == 3 && out_channels >= 1280 &&
      in_channels >= 1280) {
    return FloatBufferConv2dShaderKind::Generic;
  }

  if (kernel_h == 3 && kernel_w == 3) {
    if (
        padding[0] == 0 && padding[1] == 0 && stride[0] == 2 &&
        stride[1] == 2) {
      return FloatBufferConv2dShaderKind::Kernel3x3Stride2Pad0;
    }
    if (
        padding[0] == 1 && padding[1] == 1 && stride[0] == 2 &&
        stride[1] == 2 && in_channels == 384 && out_channels == 384) {
      return FloatBufferConv2dShaderKind::Kernel3x3Stride2Pad1;
    }
    if (
        padding[0] == 1 && padding[1] == 1 && stride[0] == 1 &&
        stride[1] == 1) {
      return FloatBufferConv2dShaderKind::Kernel3x3Stride1Pad1;
    }
  }

  return FloatBufferConv2dShaderKind::Generic;
}

api::utils::uvec3 select_float_buffer_conv2d_work_group_size(
    const FloatBufferConv2dShaderKind shader_kind,
    const api::utils::uvec3& global_size) {
  if (global_size.data[2u] <= 1u) {
    return adaptive_work_group_size(global_size);
  }

  // The specialized float buffer conv kernels do not share work across
  // adjacent output channels, so keeping the z dimension at 1 tends to map
  // better to the large spatial tiles used by the decoder-head hot path.
  switch (shader_kind) {
    case FloatBufferConv2dShaderKind::Pointwise1x1:
      return {16u, 4u, 1u};
    case FloatBufferConv2dShaderKind::Kernel3x3Stride1Pad1:
    case FloatBufferConv2dShaderKind::Kernel3x3Stride2Pad0:
    case FloatBufferConv2dShaderKind::Kernel3x3Stride2Pad1:
      return {8u, 8u, 1u};
    case FloatBufferConv2dShaderKind::Generic:
      return {4u, 4u, 1u};
  }

  return adaptive_work_group_size(global_size);
}

void record_pointwise_conv_route(
    const VulkanConvPlanDecision& decision,
    const FloatBufferConv2dShaderKind shader_kind,
    const vTensor& v_input,
    const vTensor& v_output) {
  if (!decision.pointwise) {
    return;
  }

  VulkanPointwiseConvRouteCounters& counters = pointwise_conv_route_counters();
  counters.total_1x1.fetch_add(1u, std::memory_order_relaxed);

  const bool specialized =
      shader_kind == FloatBufferConv2dShaderKind::Pointwise1x1;
  if (specialized) {
    counters.specialized_1x1_hit.fetch_add(1u, std::memory_order_relaxed);
  } else {
    counters.generic_1x1_hit.fetch_add(1u, std::memory_order_relaxed);
    if (!decision.input_buffer) {
      counters.reject_input_not_buffer.fetch_add(
          1u, std::memory_order_relaxed);
    } else if (!v_input.has_direct_buffer_layout()) {
      counters.reject_not_direct_buffer.fetch_add(
          1u, std::memory_order_relaxed);
      counters.reject_input_not_direct_buffer.fetch_add(
          1u, std::memory_order_relaxed);
    } else if (!v_output.has_direct_buffer_layout()) {
      counters.reject_not_direct_buffer.fetch_add(
          1u, std::memory_order_relaxed);
      counters.reject_output_not_direct_buffer.fetch_add(
          1u, std::memory_order_relaxed);
    } else if (v_input.storage_offset() != 0 || v_output.storage_offset() != 0) {
      counters.reject_storage_offset.fetch_add(1u, std::memory_order_relaxed);
    } else if (decision.groups != 1) {
      counters.reject_groups.fetch_add(1u, std::memory_order_relaxed);
    } else if (decision.reject ==
               VulkanConvRejectReason::KnownBadLargePointwiseConv) {
      counters.reject_shape.fetch_add(1u, std::memory_order_relaxed);
    } else {
      counters.reject_stride_padding_dilation.fetch_add(
          1u, std::memory_order_relaxed);
    }
  }

  std::ostringstream stream;
  const IntArrayRef output_sizes = v_output.sizes();
  stream << "pointwise_route"
         << " selected=" << (specialized ? "specialized_1x1" : "generic")
         << " reject=" << conv_reject_reason_name(decision.reject)
         << " input=[" << decision.n << ',' << decision.cin << ','
         << decision.h << ',' << decision.w << ']'
         << " output=["
         << (output_sizes.size() > 0 ? output_sizes[0] : 0) << ','
         << (output_sizes.size() > 1 ? output_sizes[1] : 0) << ','
         << (output_sizes.size() > 2 ? output_sizes[2] : 0) << ','
         << (output_sizes.size() > 3 ? output_sizes[3] : 0) << ']'
         << " weight=[" << decision.cout << ',' << decision.cin << ",1,1]"
         << " input_storage=" << static_cast<int>(v_input.storage_type())
         << " output_storage=" << static_cast<int>(v_output.storage_type())
         << " input_layout=" << static_cast<int>(v_input.gpu_memory_layout())
         << " output_layout=" << static_cast<int>(v_output.gpu_memory_layout())
         << " input_direct=" << (v_input.has_direct_buffer_layout() ? 1 : 0)
         << " output_direct=" << (v_output.has_direct_buffer_layout() ? 1 : 0)
         << " input_offset=" << v_input.storage_offset()
         << " output_offset=" << v_output.storage_offset()
         << " weight_packed=" << (decision.weight_packed ? 1 : 0)
         << " bias=" << (decision.bias_present ? 1 : 0)
         << " stride=[1,1] padding=[0,0] dilation=[1,1]"
         << " groups=" << decision.groups;
  utils::log_vulkan_op_hit(stream.str());
}

bool can_run_float_buffer_conv2d_add(
    const Tensor& input,
    const PackedWeightHandle& packed_weight,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups,
    const Tensor& residual) {
  if (
      !can_run_float_buffer_conv2d(
          input,
          packed_weight,
          /*transposed=*/false,
          /*quantized=*/false,
          /*output_padding=*/{}) ||
      residual.device().type() != c10::DeviceType::Vulkan ||
      residual.scalar_type() != kFloat || residual.dim() != 4 ||
      residual.requires_grad()) {
    return false;
  }

  const vTensor& v_residual = convert(residual);
  if (
      v_residual.storage_type() != api::StorageType::BUFFER ||
      v_residual.dtype() != api::kFloat ||
      !utils::supports_buffer_view_fast_path(v_residual)) {
    return false;
  }

  if (
      select_float_buffer_conv2d_shader_kind(
          packed_weight, stride, padding, dilation, groups) !=
      FloatBufferConv2dShaderKind::Kernel3x3Stride1Pad1) {
    return false;
  }

  const std::vector<int64_t> output_size = conv_output_size(
      input.sizes(),
      packed_weight.logical_weight_sizes(),
      padding,
      stride,
      dilation);
  const utils::VulkanRouteDecision route_decision = utils::select_conv2d_route(
      input.sizes(),
      packed_weight.logical_weight_sizes(),
      stride,
      padding,
      dilation,
      groups,
      input.scalar_type(),
      input.requires_grad(),
      convolution_request(utils::VulkanTensorRole::Input),
      utils::current_vulkan_device_policy());
  if (route_decision.hard_fail) {
    api::context()->flush();
    utils::fail_hard_fail("aten::convolution", route_decision);
  }
  return output_size == residual.sizes().vec();
}

Tensor prepare_runtime_float_buffer_conv_input(const Tensor& input_arg) {
  Tensor input = input_arg.is_vulkan()
      ? materialize_deferred_image_normalize_candidate_if_needed(input_arg)
      : input_arg.vulkan();
  if (input.scalar_type() == kHalf) {
    input = utils::cast_vulkan_tensor_dtype(input, kFloat);
  }
  if (input.is_vulkan()) {
    const vTensor& v_input = convert(input);
    if (
        v_input.storage_type() == api::StorageType::BUFFER &&
        v_input.gpu_memory_layout() ==
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
        utils::supports_buffer_elementwise_compute(v_input)) {
      return utils::mark_tensor_execution(
          input, utils::resolve_buffer_execution_layout(v_input), false);
    }
  }
  return utils::mark_tensor_execution(
      utils::ensure_buffer_storage(
          input, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
      api::ExecutionLayout::BUFFER_DIRECT,
      false);
}

Tensor prepare_runtime_float_buffer_conv_output(
    Tensor output,
    IntArrayRef expected_sizes) {
  output = output.is_vulkan() ? output : output.vulkan();
  output = utils::mark_tensor_execution(
      output,
      utils::resolve_buffer_execution_layout(convert(output)),
      false);
  const vTensor& v_output = convert(output);
  TORCH_CHECK(
      v_output.storage_type() == api::StorageType::BUFFER &&
          v_output.dtype() == api::kFloat &&
          utils::supports_buffer_view_fast_path(v_output),
      "Vulkan float buffer convolution out expects float buffer-backed output");
  TORCH_CHECK(
      output.sizes().vec() == expected_sizes.vec(),
      "Vulkan float buffer convolution out received mismatched output shape");
  return output;
}

Tensor run_float_buffer_conv2d_impl(
    const Tensor& input,
    const PackedWeightHandle& packed_weight,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups,
    const float output_min,
    const float output_max,
    Tensor* output_arg) {
  FloatBufferConv2dShaderKind shader_kind =
      select_float_buffer_conv2d_shader_kind(
          packed_weight, stride, padding, dilation, groups);
  api::AllocationScope allocation_scope("conv.float_buffer");
  api::Context* const context = api::context();

  vTensor v_input = convert(input);
  vTensor v_weight = packed_weight.weight_vtensor();
  vTensor v_bias = packed_weight.bias_vtensor();

  const std::vector<int64_t> output_size = conv_output_size(
      v_input.sizes(), packed_weight.logical_weight_sizes(), padding, stride, dilation);
  const utils::VulkanRouteDecision route_decision = utils::select_conv2d_route(
      v_input.sizes(),
      packed_weight.logical_weight_sizes(),
      stride,
      padding,
      dilation,
      groups,
      input.scalar_type(),
      input.requires_grad(),
      convolution_request(utils::VulkanTensorRole::Input),
      utils::current_vulkan_device_policy());
  VulkanConvPlanDecision plan_decision;
  plan_decision.n = v_input.sizes()[0];
  plan_decision.cin = v_input.sizes()[1];
  plan_decision.h = v_input.sizes()[2];
  plan_decision.w = v_input.sizes()[3];
  plan_decision.cout = packed_weight.logical_weight_sizes()[0];
  plan_decision.kh = packed_weight.logical_weight_sizes()[2];
  plan_decision.kw = packed_weight.logical_weight_sizes()[3];
  plan_decision.groups = groups;
  plan_decision.input_vulkan = input.is_vulkan();
  plan_decision.input_buffer =
      v_input.storage_type() == api::StorageType::BUFFER;
  plan_decision.weight_packed = true;
  plan_decision.bias_present = packed_weight.has_bias();
  plan_decision.transposed = false;
  plan_decision.pointwise = plan_decision.kh == 1 && plan_decision.kw == 1;
  plan_decision.large =
      plan_decision.cin >= 384 && plan_decision.cout >= 192;
  if (route_decision.hard_fail) {
    plan_decision.selected = VulkanConvPlanSelected::HardFailKnownBad;
    plan_decision.reject =
        route_decision.reject_reason ==
            utils::VulkanRouteRejectReason::KnownBadLargePointwiseConv
        ? VulkanConvRejectReason::KnownBadLargePointwiseConv
        : VulkanConvRejectReason::ShapeUnsupported;
    record_conv_plan_decision(plan_decision, "aten::convolution");
    context->flush();
    utils::fail_hard_fail("aten::convolution", route_decision);
  }
  const utils::SmallSpatialPointwiseConvMatch pointwise_contract =
      utils::match_small_spatial_pointwise_conv_contract(
          input.sizes(),
          packed_weight.logical_weight_sizes(),
          stride,
          padding,
          dilation,
          groups,
          input.scalar_type());
  if (pointwise_contract.matched) {
    if (
        pointwise_contract.family ==
            utils::SmallSpatialPointwiseConvFamily::DepthVisionProjection ||
        pointwise_contract.family ==
            utils::SmallSpatialPointwiseConvFamily::OCRProjection) {
      shader_kind = FloatBufferConv2dShaderKind::Generic;
      plan_decision.selected = VulkanConvPlanSelected::FloatBufferPointwise1x1;
      plan_decision.reject = VulkanConvRejectReason::KnownBadLargePointwiseConv;
      utils::log_vulkan_op_hit(
          utils::small_spatial_pointwise_conv_op_hit_label(
              pointwise_contract.family));
    } else {
      plan_decision.selected =
          shader_kind == FloatBufferConv2dShaderKind::Pointwise1x1
          ? VulkanConvPlanSelected::FloatBufferPointwise1x1
          : VulkanConvPlanSelected::FloatBufferConv;
      plan_decision.reject = VulkanConvRejectReason::None;
    }
  } else {
    plan_decision.selected =
        shader_kind == FloatBufferConv2dShaderKind::Pointwise1x1
        ? VulkanConvPlanSelected::FloatBufferPointwise1x1
        : VulkanConvPlanSelected::FloatBufferConv;
    plan_decision.reject = VulkanConvRejectReason::None;
  }
  record_conv_plan_decision(plan_decision, "aten::convolution");

  switch (shader_kind) {
    case FloatBufferConv2dShaderKind::Pointwise1x1:
      utils::log_vulkan_op_hit("aten::convolution.buffer_float_1x1");
      break;
    case FloatBufferConv2dShaderKind::Kernel3x3Stride1Pad1:
      utils::log_vulkan_op_hit("aten::convolution.buffer_float_3x3_s1p1");
      break;
    case FloatBufferConv2dShaderKind::Kernel3x3Stride2Pad0:
      utils::log_vulkan_op_hit("aten::convolution.buffer_float_3x3_s2p0");
      break;
    case FloatBufferConv2dShaderKind::Kernel3x3Stride2Pad1:
      utils::log_vulkan_op_hit("aten::convolution.buffer_float_3x3_s2p1");
      break;
    case FloatBufferConv2dShaderKind::Generic:
      utils::log_vulkan_op_hit("aten::convolution.buffer_float");
      break;
  }
  Tensor output_tensor;
  vTensor* v_output_ptr = nullptr;
  vTensor owned_output;
  if (output_arg != nullptr) {
    output_tensor =
        prepare_runtime_float_buffer_conv_output(*output_arg, output_size);
    v_output_ptr = &convert(output_tensor);
  } else {
    owned_output = vTensor{
        context,
        output_size,
        api::kFloat,
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
    };
    v_output_ptr = &owned_output;
  }
  vTensor& v_output = *v_output_ptr;
  record_pointwise_conv_route(plan_decision, shader_kind, v_input, v_output);

  const struct {
    int32_t stride_w;
    int32_t stride_h;
    int32_t pad_w;
    int32_t pad_h;
    int32_t dil_w;
    int32_t dil_h;
    int32_t groups;
    int32_t has_bias;
    float output_min;
    float output_max;
    float output_minmax_pad0;
    float output_minmax_pad1;
  } block{
      api::utils::safe_downcast<int32_t>(stride[1]),
      api::utils::safe_downcast<int32_t>(stride[0]),
      api::utils::safe_downcast<int32_t>(padding[1]),
      api::utils::safe_downcast<int32_t>(padding[0]),
      api::utils::safe_downcast<int32_t>(dilation[1]),
      api::utils::safe_downcast<int32_t>(dilation[0]),
      api::utils::safe_downcast<int32_t>(groups),
      packed_weight.has_bias() ? 1 : 0,
      output_min,
      output_max,
      0.0f,
      0.0f,
  };

  api::UniformParamsBuffer params(context, block);
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);
  api::UniformParamsBuffer weight_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_weight);
  api::UniformParamsBuffer bias_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_bias);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(output_size[3]),
      api::utils::safe_downcast<uint32_t>(output_size[2]),
      api::utils::safe_downcast<uint32_t>(output_size[0] * output_size[1]),
  };
  const api::utils::uvec3 local_size =
      select_float_buffer_conv2d_work_group_size(shader_kind, global_size);
  api::ShaderInfo shader = VK_KERNEL(conv2d_buffer_float);
  const char* kernel_name = "conv2d_buffer_float";
  switch (shader_kind) {
    case FloatBufferConv2dShaderKind::Pointwise1x1:
      shader = VK_KERNEL(conv2d_buffer_float_1x1);
      kernel_name = "conv2d_buffer_float_1x1";
      break;
    case FloatBufferConv2dShaderKind::Kernel3x3Stride1Pad1:
      shader = VK_KERNEL(conv2d_buffer_float_3x3_s1p1);
      kernel_name = "conv2d_buffer_float_3x3_s1p1";
      break;
    case FloatBufferConv2dShaderKind::Kernel3x3Stride2Pad0:
      shader = VK_KERNEL(conv2d_buffer_float_3x3_s2p0);
      kernel_name = "conv2d_buffer_float_3x3_s2p0";
      break;
    case FloatBufferConv2dShaderKind::Kernel3x3Stride2Pad1:
      shader = VK_KERNEL(conv2d_buffer_float_3x3_s2p1);
      kernel_name = "conv2d_buffer_float_3x3_s2p1";
      break;
    case FloatBufferConv2dShaderKind::Generic:
      break;
  }

  conv2d::log_float_buffer_conv2d_submit(
      kernel_name,
      v_input,
      v_output,
      packed_weight,
      stride,
      padding,
      dilation,
      groups,
      global_size,
      local_size);
  context->submit_compute_job(
      shader,
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer(),
      v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      weight_meta.buffer(),
      v_bias.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      bias_meta.buffer(),
      params.buffer());

  maybe_sync_after_gtx_large_buffer_conv(context, v_output);
  return record_tensor_write_and_return(
      output_arg != nullptr ? output_tensor : convert(v_output),
      "aten::convolution",
      "buffer_float",
      {input, packed_weight.weight(), packed_weight.bias()});
}

Tensor run_float_buffer_conv2d_add_impl(
    const Tensor& input,
    const PackedWeightHandle& packed_weight,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups,
    const float output_min,
    const float output_max,
    const Tensor& residual,
    Tensor& output_arg) {
  const FloatBufferConv2dShaderKind shader_kind =
      select_float_buffer_conv2d_shader_kind(
          packed_weight, stride, padding, dilation, groups);
  TORCH_CHECK(
      shader_kind == FloatBufferConv2dShaderKind::Kernel3x3Stride1Pad1,
      "Vulkan float buffer conv2d add fusion only supports 3x3 stride-1 pad-1");
  api::AllocationScope allocation_scope("conv.float_buffer_add");
  utils::log_vulkan_op_hit("aten::convolution.buffer_float_3x3_s1p1_add");
  api::Context* const context = api::context();

  const vTensor& v_input = convert(input);
  const vTensor& v_weight = packed_weight.weight_vtensor();
  const vTensor& v_bias = packed_weight.bias_vtensor();
  const vTensor& v_residual = convert(residual);

  const std::vector<int64_t> output_size = conv_output_size(
      v_input.sizes(),
      packed_weight.logical_weight_sizes(),
      padding,
      stride,
      dilation);
  Tensor output_tensor =
      prepare_runtime_float_buffer_conv_output(output_arg, output_size);
  vTensor& v_output = convert(output_tensor);

  const struct {
    int32_t stride_w;
    int32_t stride_h;
    int32_t pad_w;
    int32_t pad_h;
    int32_t dil_w;
    int32_t dil_h;
    int32_t groups;
    int32_t has_bias;
    float output_min;
    float output_max;
    float output_minmax_pad0;
    float output_minmax_pad1;
  } block{
      api::utils::safe_downcast<int32_t>(stride[1]),
      api::utils::safe_downcast<int32_t>(stride[0]),
      api::utils::safe_downcast<int32_t>(padding[1]),
      api::utils::safe_downcast<int32_t>(padding[0]),
      api::utils::safe_downcast<int32_t>(dilation[1]),
      api::utils::safe_downcast<int32_t>(dilation[0]),
      api::utils::safe_downcast<int32_t>(groups),
      packed_weight.has_bias() ? 1 : 0,
      output_min,
      output_max,
      0.0f,
      0.0f,
  };

  api::UniformParamsBuffer params(context, block);
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);
  api::UniformParamsBuffer weight_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_weight);
  api::UniformParamsBuffer bias_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_bias);
  api::UniformParamsBuffer residual_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_residual);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(output_size[3]),
      api::utils::safe_downcast<uint32_t>(output_size[2]),
      api::utils::safe_downcast<uint32_t>(output_size[0] * output_size[1]),
  };

  context->submit_compute_job(
      VK_KERNEL(conv2d_buffer_float_3x3_s1p1_add),
      pipeline_barrier,
      global_size,
      select_float_buffer_conv2d_work_group_size(shader_kind, global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer(),
      v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      weight_meta.buffer(),
      v_bias.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      bias_meta.buffer(),
      v_residual.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      residual_meta.buffer(),
      params.buffer());

  return record_tensor_write_and_return(
      output_tensor,
      "aten::convolution",
      "buffer_float_3x3_s1p1_add",
      {input, packed_weight.weight(), packed_weight.bias(), residual});
}

Tensor run_float_buffer_conv2d(
    const Tensor& input,
    const PackedWeightHandle& packed_weight,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups,
    const float output_min,
    const float output_max) {
  return run_float_buffer_conv2d_impl(
      input,
      packed_weight,
      stride,
      padding,
      dilation,
      groups,
      output_min,
      output_max,
      nullptr);
}

Tensor run_float_buffer_conv_transpose2d_impl(
    const Tensor& input,
    const PackedWeightHandle& packed_weight,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const IntArrayRef output_padding,
    const int64_t groups,
    const float output_min,
    const float output_max,
    Tensor* output_arg) {
  const bool use_nonoverlap_kernel =
      can_use_float_buffer_nonoverlap_conv_transpose2d(
          packed_weight, stride, padding, dilation, output_padding);
  utils::log_vulkan_op_hit(
      use_nonoverlap_kernel
          ? "aten::convolution.buffer_float_transpose_nonoverlap"
          : "aten::convolution.buffer_float_transpose");
  api::AllocationScope allocation_scope("conv_transpose.float_buffer");
  api::Context* const context = api::context();

  vTensor v_input = convert(input);
  vTensor v_weight = packed_weight.weight_vtensor();
  vTensor v_bias = packed_weight.bias_vtensor();

  const std::vector<int64_t> output_size = get_conv_transpose_output_size(
      v_input.sizes(),
      packed_weight.logical_weight_sizes(),
      padding,
      output_padding,
      stride,
      dilation);
  Tensor output_tensor;
  vTensor* v_output_ptr = nullptr;
  vTensor owned_output;
  if (output_arg != nullptr) {
    output_tensor =
        prepare_runtime_float_buffer_conv_output(*output_arg, output_size);
    v_output_ptr = &convert(output_tensor);
  } else {
    owned_output = vTensor{
        context,
        output_size,
        api::kFloat,
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
    };
    v_output_ptr = &owned_output;
  }
  vTensor& v_output = *v_output_ptr;

  const struct {
    int32_t stride_w;
    int32_t stride_h;
    int32_t pad_w;
    int32_t pad_h;
    int32_t dil_w;
    int32_t dil_h;
    int32_t groups;
    int32_t has_bias;
    float output_min;
    float output_max;
    float output_minmax_pad0;
    float output_minmax_pad1;
  } block{
      api::utils::safe_downcast<int32_t>(stride[1]),
      api::utils::safe_downcast<int32_t>(stride[0]),
      api::utils::safe_downcast<int32_t>(padding[1]),
      api::utils::safe_downcast<int32_t>(padding[0]),
      api::utils::safe_downcast<int32_t>(dilation[1]),
      api::utils::safe_downcast<int32_t>(dilation[0]),
      api::utils::safe_downcast<int32_t>(groups),
      packed_weight.has_bias() ? 1 : 0,
      output_min,
      output_max,
      0.0f,
      0.0f,
  };

  api::UniformParamsBuffer params(context, block);
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);
  api::UniformParamsBuffer weight_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_weight);
  api::UniformParamsBuffer bias_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_bias);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(output_size[3]),
      api::utils::safe_downcast<uint32_t>(output_size[2]),
      api::utils::safe_downcast<uint32_t>(output_size[0] * output_size[1]),
  };
  const api::ShaderInfo shader = use_nonoverlap_kernel
      ? VK_KERNEL(conv_transpose2d_buffer_float_nonoverlap)
      : VK_KERNEL(conv_transpose2d_buffer_float);

  context->submit_compute_job(
      shader,
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
      in_meta.buffer(),
      v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      weight_meta.buffer(),
      v_bias.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      bias_meta.buffer(),
      params.buffer());

  return record_tensor_write_and_return(
      output_arg != nullptr ? output_tensor : convert(v_output),
      "aten::convolution",
      use_nonoverlap_kernel ? "buffer_float_transpose_nonoverlap"
                            : "buffer_float_transpose",
      {input, packed_weight.weight(), packed_weight.bias()});
}

Tensor run_float_buffer_conv_transpose2d(
    const Tensor& input,
    const PackedWeightHandle& packed_weight,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const IntArrayRef output_padding,
    const int64_t groups,
    const float output_min,
    const float output_max) {
  return run_float_buffer_conv_transpose2d_impl(
      input,
      packed_weight,
      stride,
      padding,
      dilation,
      output_padding,
      groups,
      output_min,
      output_max,
      nullptr);
}

std::optional<Tensor> try_run_no_overlap_conv_transpose2d_contract(
    const Tensor& input_arg,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context,
    const float output_min,
    const float output_max,
    Tensor* output_arg) {
  if (!might_match_no_overlap_conv_transpose2d_contract(
          input_arg, conv_context)) {
    return std::nullopt;
  }

  Tensor buffer_input = prepare_runtime_float_buffer_conv_input(input_arg);
  const PackedWeightHandle& packed_weight = conv_context->packed_weight();
  const utils::NoOverlapConvTranspose2DMatch match =
      utils::match_no_overlap_conv_transpose2d_contract(
          no_overlap_conv_transpose2d_tensor_info(buffer_input),
          no_overlap_conv_transpose2d_packed_info(packed_weight),
          no_overlap_conv_transpose2d_options(conv_context));
  if (!match.matched) {
    return std::nullopt;
  }

  return run_float_buffer_conv_transpose2d_impl(
      buffer_input,
      packed_weight,
      conv_context->stride(),
      conv_context->padding(),
      conv_context->dilation(),
      conv_context->output_padding(),
      conv_context->groups(),
      output_min,
      output_max,
      output_arg);
}

Tensor run_bfloat16_buffer_conv2d(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups) {
  api::AllocationScope allocation_scope("conv.bf16_buffer");
  api::Context* const context = api::context();

  vTensor v_input = convert(input);
  vTensor v_weight = convert(weight);
  Tensor bias_buffer =
      prepare_float_bias_buffer_for_conv2d(bias, weight.size(0));
  vTensor v_bias = convert(bias_buffer);

  const std::vector<int64_t> output_size =
      conv_output_size(input.sizes(), weight.sizes(), padding, stride, dilation);
  vTensor v_output{
      context,
      output_size,
      api::kFloat,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct {
    int32_t stride_w;
    int32_t stride_h;
    int32_t pad_w;
    int32_t pad_h;
    int32_t dil_w;
    int32_t dil_h;
    int32_t groups;
    int32_t has_bias;
  } block{
      api::utils::safe_downcast<int32_t>(stride[1]),
      api::utils::safe_downcast<int32_t>(stride[0]),
      api::utils::safe_downcast<int32_t>(padding[1]),
      api::utils::safe_downcast<int32_t>(padding[0]),
      api::utils::safe_downcast<int32_t>(dilation[1]),
      api::utils::safe_downcast<int32_t>(dilation[0]),
      api::utils::safe_downcast<int32_t>(groups),
      (bias && bias->defined()) ? 1 : 0,
  };

  api::UniformParamsBuffer params(context, block);
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);
  api::UniformParamsBuffer weight_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_weight);
  api::UniformParamsBuffer bias_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_bias);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(output_size[3]),
      api::utils::safe_downcast<uint32_t>(output_size[2]),
      api::utils::safe_downcast<uint32_t>(output_size[0] * output_size[1]),
  };

  context->submit_compute_job(
      VK_KERNEL(conv2d_buffer_bfloat16),
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
      in_meta.buffer(),
      v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      weight_meta.buffer(),
      v_bias.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      bias_meta.buffer(),
      params.buffer());

  return record_tensor_write_and_return(
      convert(v_output),
      "aten::convolution",
      "bf16_buffer_float_output",
      {input, weight, bias_buffer});
}

  Tensor convolution(
      const Tensor& input,
      const Tensor& weight,
      const std::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const bool transposed,
    const IntArrayRef output_padding,
    const int64_t groups) {
      Tensor compute_input = input.is_vulkan()
          ? materialize_deferred_image_normalize_candidate_if_needed(input)
          : input;
      if (can_run_bfloat16_buffer_conv2d(
              compute_input, weight, bias, transposed, false, output_padding)) {
        return run_bfloat16_buffer_conv2d(
            compute_input, weight, bias, stride, padding, dilation, groups);
      }
      const Tensor compute_weight = utils::prepare_vulkan_execution_tensor(
          weight,
          utils::VulkanExecutionPlanKind::Conv2dWeightSource,
          convolution_request(utils::VulkanTensorRole::Weight));
  const std::optional<Tensor> compute_bias =
      utils::prepare_optional_vulkan_execution_tensor(
          bias,
          utils::VulkanExecutionPlanKind::Conv2dBiasSource,
          convolution_request(utils::VulkanTensorRole::Bias));
  const bool avoid_large_buffer_conv_3x3 =
      should_force_image_conv_for_known_bad_large_buffer_conv(
          compute_input,
          weight,
          stride,
          padding,
          dilation,
          groups);
  const auto small_metadata_padded_conv2d_match =
      utils::match_small_metadata_padded_conv2d_contract(
          small_metadata_padded_conv2d_tensor_info(compute_input),
          small_metadata_padded_conv2d_weight_info(compute_weight),
          small_metadata_padded_conv2d_options(
              stride, padding, dilation, transposed, output_padding, groups));
  if (
      small_metadata_padded_conv2d_match.matched &&
      small_metadata_padded_conv2d_match.requires_input_materialization) {
    compute_input = utils::mark_tensor_execution(
        utils::ensure_buffer_storage(
            compute_input, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
        api::ExecutionLayout::BUFFER_DIRECT,
        true);
    utils::log_vulkan_op_hit(
        "aten::convolution.small_metadata_padded_conv2d.materialize_input");
  }
  if (
      avoid_large_buffer_conv_3x3 ||
      utils::match_small_spatial_pointwise_conv_contract(
          compute_input.sizes(),
          weight.sizes(),
          stride,
          padding,
          dilation,
          groups,
          compute_input.scalar_type())
              .family ==
          utils::SmallSpatialPointwiseConvFamily::DepthVisionProjection) {
    utils::select_conv2d_route(
        compute_input.sizes(),
        weight.sizes(),
        stride,
        padding,
        dilation,
        groups,
        compute_input.scalar_type(),
        compute_input.requires_grad(),
        convolution_request(utils::VulkanTensorRole::Input),
        utils::current_vulkan_device_policy());
  }
  const bool force_small_metadata_image_pack =
      should_force_image_conv_for_small_metadata_input(compute_input) &&
      !small_metadata_padded_conv2d_match.matched;
  const bool force_legacy_image_pack =
      force_small_metadata_image_pack || avoid_large_buffer_conv_3x3;
  if (force_legacy_image_pack) {
    utils::log_vulkan_op_hit(
        force_small_metadata_image_pack
            ? "aten::convolution.buffer_float_skip.small_metadata_input"
            : "aten::convolution.buffer_float_skip.known_bad_large_3x3");
  }
  if (utils::has_inference_tensor(compute_weight, compute_bias)) {
    auto conv_context = c10::make_intrusive<Conv2dPackedContext>(
        compute_weight,
        compute_bias,
        stride,
        padding,
        dilation,
        transposed,
        false,
        output_padding,
        groups,
        std::nullopt,
        std::nullopt,
        weight,
        bias,
        force_legacy_image_pack);
    return run_conv2d_context(compute_input, conv_context);
  }
  auto conv_context = c10::make_intrusive<Conv2dPackedContext>(
      compute_weight,
      compute_bias,
      stride,
      padding,
      dilation,
      transposed,
      false,
      output_padding,
      groups,
      std::nullopt,
      std::nullopt,
      weight,
      bias,
      force_legacy_image_pack);

  return run_conv2d_context(compute_input, conv_context);
}

} // namespace

namespace conv1d {

static Tensor upload_tensor_to_buffer(
    const Tensor& tensor,
    const api::GPUMemoryLayout memory_layout) {
  Tensor source = tensor.requires_grad() ? tensor.detach() : tensor;
  if (source.scalar_type() == kBFloat16 || source.scalar_type() == kHalf) {
    source = source.to(kFloat);
  }

  if (source.is_vulkan()) {
    return utils::mark_tensor_execution(
        utils::ensure_buffer_storage(source, memory_layout),
        api::ExecutionLayout::BUFFER_DIRECT,
        true);
  }

  TORCH_CHECK(
      source.device().is_cpu(),
      "Vulkan conv1d buffer prepack expects CPU or Vulkan tensors");
  source = source.contiguous();
  vTensor v_buffer{
      api::context(),
      source.sizes().vec(),
      convert_dtype(source.scalar_type()),
      api::StorageType::BUFFER,
      memory_layout,
  };
  pack_cpu_to_vulkan(source, v_buffer);
  return utils::mark_tensor_execution(
      convert(v_buffer), api::ExecutionLayout::BUFFER_DIRECT, true);
}

static vTensor pack_weights_using_width_packing(const Tensor& weight_arg) {
  Tensor weight = weight_arg;

  if (weight.is_cpu()) {
    weight = weight.vulkan();
  }

  TORCH_CHECK(weight.is_vulkan(), "Weight must be on Vulkan device!");

  vTensor v_weight = convert(weight);
  if (v_weight.gpu_memory_layout() ==
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED) {
    v_weight = packing::convert_image_channels_packed_to_width_packed(v_weight);
  }

  TORCH_CHECK(
      v_weight.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      "After packing, the v_weight must be in TENSOR_WIDTH_PACKED format");

  return v_weight;
}

/*
 * This is a full implementation. For algorithm details, refer to the shader
 * kernel code.
 */
static Tensor run_conv1d_context_impl(
    const Tensor& input_arg,
    const Tensor& weight_arg,
    const std::optional<Tensor>& bias_arg_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  api::Context* const context = api::context();
  Tensor input = utils::prepare_vulkan_execution_tensor(
      input_arg,
      utils::VulkanExecutionPlanKind::Conv1dRuntimeInput,
      convolution_request(utils::VulkanTensorRole::Input));
  if (input.scalar_type() == kBFloat16 || input.scalar_type() == kHalf) {
    input = utils::cast_vulkan_tensor_dtype(input, kFloat);
  }

  Tensor weight = utils::prepare_vulkan_execution_tensor(
      weight_arg,
      utils::VulkanExecutionPlanKind::Conv1dRuntimeWeight,
      convolution_request(utils::VulkanTensorRole::Weight));
  if (weight.scalar_type() == kBFloat16 || weight.scalar_type() == kHalf) {
    weight = utils::cast_vulkan_tensor_dtype(weight, kFloat);
  }

  const IntArrayRef& input_sizes = input.sizes();
  const IntArrayRef& weight_sizes = weight.sizes();

  int32_t in_channels = static_cast<int32_t>(input_sizes[1]);
  int32_t out_channels = static_cast<int32_t>(weight_sizes[0]);
  int32_t kernel_size = static_cast<int32_t>(weight_sizes[2]);

  Tensor bias;
  if (bias_arg_opt) {
    bias = utils::prepare_vulkan_execution_tensor(
        *bias_arg_opt,
        utils::VulkanExecutionPlanKind::Conv1dRuntimeBias,
        convolution_request(utils::VulkanTensorRole::Bias));
  } else {
    bias = utils::prepare_vulkan_execution_tensor(
        at::zeros({out_channels}, at::device(at::kCPU).dtype(at::kFloat)),
        utils::VulkanExecutionPlanKind::Conv1dRuntimeBias,
        convolution_request(utils::VulkanTensorRole::Bias));
  }
  if (bias.scalar_type() == kBFloat16 || bias.scalar_type() == kHalf) {
    bias = utils::cast_vulkan_tensor_dtype(bias, kFloat);
  }

  TORCH_CHECK(input.dim() == 3, "input must be a 3-dim tensor");
  TORCH_CHECK(weight.dim() == 3, "weight must be a 3-dim tensor");
  TORCH_CHECK(
      in_channels % groups == 0, "in_channels must be divisible by groups");
  TORCH_CHECK(
      out_channels % groups == 0, "out_channels must be divisible by groups");

  const vTensor& v_input = convert(input);
  const vTensor& v_weight = convert(weight);
  const vTensor& v_bias = convert(bias);

  vTensor v_output{
      context,
      conv_output_size(input_sizes, weight_sizes, padding, stride, dilation),
      v_input.dtype(),
  };

  const struct Block final {
    int32_t in_length;
    int32_t kernel_size;
    int32_t stride;
    int32_t padding;
    int32_t dilation;
    int32_t in_group_size;
    int32_t out_group_size;
    int32_t batch_size;
  } block{
      static_cast<int32_t>(input_sizes[2]),
      kernel_size,
      static_cast<int32_t>(stride[0]),
      static_cast<int32_t>(padding[0]),
      static_cast<int32_t>(dilation[0]),
      static_cast<int32_t>(in_channels / groups),
      static_cast<int32_t>(out_channels / groups),
      static_cast<int32_t>(input_sizes[0]),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(conv1d),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      {1, static_cast<uint32_t>(out_channels), 1},
      // local work group size
      {1, 1, 1},
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return record_tensor_write_and_return(
      convert(v_output),
      "aten::convolution",
      "conv1d_texture",
      {input, weight, bias});
}

static Tensor run_conv1d_buffer_context_impl(
    const Tensor& input_arg,
    const Tensor& weight_arg,
    const Tensor& bias_arg,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  api::Context* const context = api::context();
  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  if (input.scalar_type() == kBFloat16 || input.scalar_type() == kHalf) {
    input = utils::cast_vulkan_tensor_dtype(input, kFloat);
  }
  input = utils::ensure_buffer_storage(
      input, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);

  Tensor weight = weight_arg;
  if (weight.scalar_type() == kBFloat16 || weight.scalar_type() == kHalf) {
    weight = utils::cast_vulkan_tensor_dtype(weight, kFloat);
  }
  weight = utils::ensure_buffer_storage(
      weight, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);

  Tensor bias = bias_arg;
  if (bias.scalar_type() == kBFloat16 || bias.scalar_type() == kHalf) {
    bias = utils::cast_vulkan_tensor_dtype(bias, kFloat);
  }
  bias = utils::ensure_buffer_storage(
      bias, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);

  const IntArrayRef input_sizes = input.sizes();
  const IntArrayRef weight_sizes = weight.sizes();
  const int32_t in_channels = static_cast<int32_t>(input_sizes[1]);
  const int32_t out_channels = static_cast<int32_t>(weight_sizes[0]);
  const int32_t kernel_size = static_cast<int32_t>(weight_sizes[2]);

  TORCH_CHECK(input.dim() == 3, "input must be a 3-dim tensor");
  TORCH_CHECK(weight.dim() == 3, "weight must be a 3-dim tensor");
  TORCH_CHECK(bias.dim() == 1, "bias must be a 1-dim tensor");
  TORCH_CHECK(
      in_channels % groups == 0, "in_channels must be divisible by groups");
  TORCH_CHECK(
      out_channels % groups == 0, "out_channels must be divisible by groups");

  vTensor v_output{
      context,
      conv_output_size(input_sizes, weight_sizes, padding, stride, dilation),
      api::kFloat,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };
  const vTensor& v_input = convert(input);
  const vTensor& v_weight = convert(weight);
  const vTensor& v_bias = convert(bias);

  const struct Block final {
    ivec4 size0;
    ivec4 size1;
  } block{
      {
          static_cast<int32_t>(input_sizes[2]),
          kernel_size,
          static_cast<int32_t>(stride[0]),
          static_cast<int32_t>(padding[0]),
      },
      {
          static_cast<int32_t>(dilation[0]),
          static_cast<int32_t>(in_channels / groups),
          static_cast<int32_t>(out_channels / groups),
          static_cast<int32_t>(input_sizes[0]),
      },
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size = {
      api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
      1u,
      1u,
  };
  context->submit_compute_job(
      VK_KERNEL(conv1d_buffer_float),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      utils::make_buffer_compute_metadata_ubo(context, v_output).buffer(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      utils::make_buffer_compute_metadata_ubo(context, v_input).buffer(),
      v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      utils::make_buffer_compute_metadata_ubo(context, v_weight).buffer(),
      v_bias.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      utils::make_buffer_compute_metadata_ubo(context, v_bias).buffer(),
      params.buffer());

  return record_tensor_write_and_return(
      convert(v_output),
      "aten::convolution",
      "conv1d_buffer",
      {input, weight, bias});
}

} // namespace conv1d

Conv2dPackedContext::Conv2dPackedContext(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const IntArrayRef stride_arg,
    const IntArrayRef padding_arg,
    const IntArrayRef dilation_arg,
    const bool transposed,
    const bool quantized,
    const IntArrayRef output_padding_arg,
    const int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max,
    const Tensor& cache_weight_arg,
    const std::optional<Tensor>& cache_bias_arg,
    const bool force_legacy_image_pack)
    : unpacked_{c10::AnyType::get()} {
  const auto stride = expand_param_if_needed(stride_arg, "stride", 2);
  const auto padding = expand_param_if_needed(padding_arg, "padding", 2);
  const auto dilation = expand_param_if_needed(dilation_arg, "dilation", 2);
  const auto output_padding =
      expand_param_if_needed(output_padding_arg, "output_padding", 2);

  TORCH_CHECK(
      available(
          weight,
          bias,
          stride,
          padding,
          dilation,
          transposed,
          quantized,
          output_padding,
          groups,
          output_min,
          output_max),
      "Vulkan::convolution not available! "
      "Reason: The provided (weight, bias, stride, padding, dilation, groups, "
      "transposed, output_padding, output_min, output_max) parameters are either "
      "invalid individually or their combination is not supported by Vulkan impl.");

  const auto method = conv2d::determine_method(
      weight.sizes(), stride, padding, dilation, groups, transposed, quantized);

  const auto normalized_bias = utils::normalized_optional_tensor(bias);
  const Tensor& cache_weight =
      cache_weight_arg.defined() ? cache_weight_arg : weight;
  const std::optional<Tensor> normalized_cache_bias =
      cache_bias_arg.has_value()
      ? utils::normalized_optional_tensor(cache_bias_arg)
      : normalized_bias;
  const std::vector<int64_t> logical_weight_sizes = weight.sizes().vec();
  constexpr uint64_t kConvTransposedPackOption = 1u;
  constexpr uint64_t kConvBufferPackOption = 1u << 1;
  const PackedWeightKind packed_weight_kind =
      packed_weight_kind_for_conv2d_method(method);
  const bool use_float_buffer_packing = !force_legacy_image_pack &&
      can_use_float_buffer_conv2d_prepack(
          weight, bias, transposed, quantized, output_padding);
  const uint64_t pack_options =
      (transposed ? kConvTransposedPackOption : 0u) |
      (use_float_buffer_packing ? kConvBufferPackOption : 0u);
  if (const auto cached_packed_weight = utils::lookup_packed_weight_handle(
          cache_weight,
          normalized_cache_bias,
          logical_weight_sizes,
          packed_weight_kind,
          quantized,
          pack_options)) {
    packed_weight_ = *cached_packed_weight;
  } else {
    if (use_float_buffer_packing) {
      utils::log_vulkan_op_hit("aten::convolution.buffer_float_prepack");
      const int64_t buffer_bias_channels =
          transposed ? logical_weight_sizes[1] * groups : logical_weight_sizes[0];
      packed_weight_ = make_float_buffer_conv2d_handle(
          weight,
          bias,
          logical_weight_sizes,
          packed_weight_kind,
          buffer_bias_channels);
    } else {
      packed_weight_ = utils::make_packed_weight_handle(
          convert(pack_weights(weight, transposed, quantized, method)),
          convert(pack_biases(bias, weight, transposed, quantized)),
          logical_weight_sizes,
          packed_weight_kind,
          bias && bias->defined(),
          quantized);
    }
    if (should_cache_float_buffer_conv2d_handle(
            packed_weight_, packed_weight_kind)) {
      utils::store_packed_weight_handle(
          cache_weight,
          normalized_cache_bias,
          logical_weight_sizes,
          packed_weight_kind,
          packed_weight_,
          quantized,
          pack_options);
    }
  }
  overlay_region_ = compute_overlay_region(weight, dilation, transposed);
  const auto packed_stride = pack_params(stride);
  const auto packed_padding = pack_params(padding);
  const auto packed_dilation = pack_params(dilation);
  stride_ = {packed_stride.begin(), packed_stride.end()};
  padding_ = {packed_padding.begin(), packed_padding.end()};
  output_padding_ = output_padding;
  dilation_ = {packed_dilation.begin(), packed_dilation.end()};
  transposed_ = transposed;
  quantized_ = quantized;
  groups_ = safe_downcast<int32_t>(groups);
  output_min_ = output_min ? output_min->template to<float>()
                           : -std::numeric_limits<float>::infinity();
  output_max_ = output_max ? output_max->template to<float>()
                           : +std::numeric_limits<float>::infinity();
  conv_method_ = method;

  compute_shader_ = conv2d::get_shader(
      weight.sizes(), stride, padding, dilation, method, transposed, quantized);

  if (!at::globalContext().releaseWeightsWhenPrepacking()) {
    unpacked_.reserve(Unpacked::NumArgs);
    unpacked_.emplace_back(weight);
    unpacked_.emplace_back(normalized_bias);
    unpacked_.emplace_back(stride_arg.vec());
    unpacked_.emplace_back(padding_arg.vec());
    unpacked_.emplace_back(dilation_arg.vec());
    unpacked_.emplace_back(transposed);
    unpacked_.emplace_back(quantized);
    unpacked_.emplace_back(output_padding_arg.vec());
    unpacked_.emplace_back(groups);
    unpacked_.emplace_back(output_min);
    unpacked_.emplace_back(output_max);
  }
}

Conv2dPackedContext Conv2dPackedContext::pack(c10::impl::GenericList unpacked) {
  return Conv2dPackedContext(
      unpacked.get(Unpacked::Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Bias),
      unpacked.get(Unpacked::Stride).toIntVector(),
      unpacked.get(Unpacked::Padding).toIntVector(),
      unpacked.get(Unpacked::Dilation).toIntVector(),
      unpacked.get(Unpacked::isTransposed).toBool(),
      unpacked.get(Unpacked::isQuantized).toBool(),
      unpacked.get(Unpacked::OutputPadding).toIntVector(),
      unpacked.get(Unpacked::Groups).toInt(),
      get_optional_scalar(unpacked, Unpacked::OutputMin),
      get_optional_scalar(unpacked, Unpacked::OutputMax));
}

c10::intrusive_ptr<Conv2dPackedContext> create_conv2d_context(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  return c10::make_intrusive<Conv2dPackedContext>(Conv2dPackedContext(
      weight,
      bias,
      stride,
      padding,
      dilation,
      /* transposed = */ false,
      /* quantized = */ false,
      /* output_padding_arg = */ {0},
      groups,
      output_min,
      output_max));
}

c10::intrusive_ptr<Conv2dPackedContext> create_tconv2d_context(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& output_padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  return c10::make_intrusive<Conv2dPackedContext>(Conv2dPackedContext(
      weight,
      bias,
      stride,
      padding,
      dilation,
      /* transposed = */ true,
      /* quantized = */ false,
      output_padding,
      groups,
      output_min,
      output_max));
}

c10::intrusive_ptr<Conv2dPackedContext> create_qconv2d_context(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  return c10::make_intrusive<Conv2dPackedContext>(Conv2dPackedContext(
      weight,
      bias,
      stride,
      padding,
      dilation,
      /* transposed = */ false,
      /* quantized = */ true,
      /* output_padding_arg = */ {0},
      groups,
      output_min,
      output_max));
}

c10::intrusive_ptr<Conv2dPackedContext> create_qtconv2d_context(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& output_padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  return c10::make_intrusive<Conv2dPackedContext>(Conv2dPackedContext(
      weight,
      bias,
      stride,
      padding,
      dilation,
      /* transposed = */ true,
      /* quantized = */ true,
      output_padding,
      groups,
      output_min,
      output_max));
}

static Tensor run_conv2d_context_impl(
    const Tensor& input_arg,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context,
    double scale,
    int64_t zero_point,
    Tensor* output_arg = nullptr,
    const bool fuse_relu = false) {
  const PackedWeightHandle& packed_weight = conv_context->packed_weight();
  const auto quantized = conv_context->quantized();
  const auto& stride = conv_context->stride();
  const auto& padding = conv_context->padding();
  const auto& output_padding = conv_context->output_padding();
  const auto& dilation = conv_context->dilation();
  const auto transposed = conv_context->transposed();
  float output_min = conv_context->output_min();
  float output_max = conv_context->output_max();
  if (fuse_relu) {
    output_min = output_min > 0.0f ? output_min : 0.0f;
    output_max = output_max > 0.0f ? output_max : 0.0f;
  }

  if (
      input_arg.device().type() == c10::DeviceType::Vulkan &&
      input_arg.scalar_type() == kFloat &&
      can_run_exact_pointwise_nooverlap_conv_transpose2d(conv_context)) {
    if (auto no_overlap_output = try_run_no_overlap_conv_transpose2d_contract(
            input_arg,
            conv_context,
            output_min,
            output_max,
            output_arg)) {
      return *no_overlap_output;
    }
    return run_exact_pointwise_nooverlap_conv_transpose2d(
        input_arg,
        conv_context,
        output_min,
        output_max,
        output_arg);
  }

  if (!quantized && packed_weight.execution_layout() ==
          api::ExecutionLayout::BUFFER_DIRECT) {
    Tensor buffer_input = prepare_runtime_float_buffer_conv_input(input_arg);
    const char* const buffer_transpose_skip_reason =
        float_buffer_conv_transpose2d_skip_reason(
            buffer_input, packed_weight, transposed, quantized);
    if (buffer_transpose_skip_reason == nullptr) {
      return run_float_buffer_conv_transpose2d_impl(
          buffer_input,
          packed_weight,
          stride,
          padding,
          dilation,
          output_padding,
          conv_context->groups(),
          output_min,
          output_max,
          output_arg);
    }
    if (transposed) {
      utils::log_vulkan_op_hit(buffer_transpose_skip_reason);
    }
    if (can_run_float_buffer_conv2d(
            buffer_input, packed_weight, transposed, quantized, output_padding)) {
      return run_float_buffer_conv2d_impl(
          buffer_input,
          packed_weight,
          stride,
          padding,
          dilation,
          conv_context->groups(),
          output_min,
          output_max,
          output_arg);
    }
  }

  TORCH_CHECK(
      output_arg == nullptr,
      "Vulkan convolution out is only supported for float buffer-backed contexts");

  api::Context* const context = api::context();
  const Tensor runtime_input_arg =
      input_arg.requires_grad() ? input_arg.detach() : input_arg;
  Tensor input = utils::prepare_vulkan_execution_tensor(
      runtime_input_arg,
      utils::VulkanExecutionPlanKind::Conv2dRuntimeInput,
      convolution_request(utils::VulkanTensorRole::Input));
  if (
      !quantized &&
      (input.scalar_type() == kBFloat16 || input.scalar_type() == kHalf)) {
    input = utils::cast_vulkan_tensor_dtype(input, kFloat);
  }
  TORCH_CHECK(input.is_vulkan(), "Input tensor must be Vulkan!");
  const vTensor& v_input = convert(input);
  const vTensor& v_weight = packed_weight.weight_vtensor();
  const vTensor& v_bias = packed_weight.bias_vtensor();

  api::AllocationScope allocation_scope(quantized ? "qconv" : "conv");
  const auto& overlay_region = conv_context->overlay_region();
  const Conv2dMethod method_ = conv_context->conv_method();
  const auto& kernel_size = packed_weight.logical_weight_sizes();

  TORCH_CHECK(
      usable(input, quantized),
      "Input tensor not usable for convolution! state={",
      describe_tensor_state(input),
      "} provenance={",
      describe_tensor_provenance(input),
      "}");

  std::vector<int64_t> output_size;
  if (transposed) {
    output_size = get_conv_transpose_output_size(
        v_input.sizes(),
        kernel_size,
        padding,
        output_padding,
        stride,
        dilation);
  } else {
    output_size = conv_output_size(
        v_input.sizes(), kernel_size, padding, stride, dilation);
  }

  vTensor v_output{
      context,
      output_size,
      v_input.dtype(),
  };

  if (quantized) {
    v_output.set_is_quantized();
    v_output.set_scale(scale);
    v_output.set_zero_point(zero_point);
  }

  if (quantized) {
    conv2d::record_quantized_op(
        context,
        conv_context->compute_shader(),
        v_output,
        v_input,
        v_weight,
        v_bias,
        overlay_region,
        stride,
        padding,
        dilation,
        output_min,
        output_max,
        kernel_size,
        method_,
        transposed);
  } else {
    conv2d::record_op(
        context,
        conv_context->compute_shader(),
        v_output,
        v_input,
        v_weight,
        v_bias,
        overlay_region,
        stride,
        padding,
        dilation,
        output_min,
        output_max,
        kernel_size,
        method_,
        transposed);
  }

  return record_tensor_write_and_return(
      convert(v_output),
      "aten::convolution",
      "packed_context",
      {input_arg});
}

Tensor run_conv2d_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context) {
  return run_conv2d_context_impl(input_arg, conv_context, 1.0f, 0u, nullptr);
}

Tensor run_conv2d_context_out(
    const Tensor& input_arg,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context,
    Tensor& output) {
  return run_conv2d_context_impl(input_arg, conv_context, 1.0f, 0u, &output);
}

Tensor run_conv2d_context_relu_out(
    const Tensor& input_arg,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context,
    Tensor& output) {
  return run_conv2d_context_impl(
      input_arg, conv_context, 1.0f, 0u, &output, /*fuse_relu=*/true);
}

std::optional<Tensor> try_run_conv2d_context_add_out(
    const Tensor& input_arg,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context,
    const Tensor& residual_arg,
    Tensor& output) {
  const PackedWeightHandle& packed_weight = conv_context->packed_weight();
  if (
      conv_context->quantized() || conv_context->transposed() ||
      packed_weight.execution_layout() != api::ExecutionLayout::BUFFER_DIRECT) {
    return std::nullopt;
  }

  Tensor input = prepare_runtime_float_buffer_conv_input(input_arg);
  Tensor residual = prepare_runtime_float_buffer_conv_input(residual_arg);
  if (!can_run_float_buffer_conv2d_add(
          input,
          packed_weight,
          conv_context->stride(),
          conv_context->padding(),
          conv_context->dilation(),
          conv_context->groups(),
          residual)) {
    return std::nullopt;
  }

  return run_float_buffer_conv2d_add_impl(
      input,
      packed_weight,
      conv_context->stride(),
      conv_context->padding(),
      conv_context->dilation(),
      conv_context->groups(),
      conv_context->output_min(),
      conv_context->output_max(),
      residual,
      output);
}

Tensor run_tconv2d_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context) {
  return run_conv2d_context_impl(input_arg, conv_context, 1.0f, 0u, nullptr);
}

Tensor run_tconv2d_context_out(
    const Tensor& input_arg,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context,
    Tensor& output) {
  return run_conv2d_context_impl(input_arg, conv_context, 1.0f, 0u, &output);
}

Tensor run_qconv2d_context(
    const Tensor& input_arg,
    double scale,
    int64_t zero_point,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context) {
  return run_conv2d_context_impl(
      input_arg, conv_context, scale, zero_point, nullptr);
}

/* Backwards compatibility */
Conv2dOpContext::Conv2dOpContext(Conv2dPackedContext conv_context)
    : conv_context_{std::move(conv_context)} {}

Conv2dOpContext Conv2dOpContext::create(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const IntArrayRef stride_arg,
    const IntArrayRef padding_arg,
    const IntArrayRef dilation_arg,
    const bool transposed,
    const IntArrayRef output_padding_arg,
    const int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  return Conv2dOpContext{Conv2dPackedContext(
      weight,
      bias,
      stride_arg,
      padding_arg,
      dilation_arg,
      transposed,
      /* quantized = */ false,
      output_padding_arg,
      groups,
      output_min,
      output_max)};
}

Tensor Conv2dOpContext::run(const Tensor& input_arg) const {
  return run_conv2d_context(
      input_arg, c10::make_intrusive<Conv2dPackedContext>(conv_context_));
}

Conv2dOpContext::State Conv2dOpContext::unpack() const {
  const c10::impl::GenericList unpacked_ = conv_context_.unpack();

  TORCH_CHECK(!unpacked_.empty(), "unpacked_ does not have any elements!");

  return Conv2dOpContext::State(
      unpacked_.get(Conv2dPackedContext::Unpacked::Weight).toTensor(),
      get_optional_tensor(unpacked_, Conv2dPackedContext::Unpacked::Bias),
      unpacked_.get(Conv2dPackedContext::Unpacked::Stride).toIntVector(),
      unpacked_.get(Conv2dPackedContext::Unpacked::Padding).toIntVector(),
      unpacked_.get(Conv2dPackedContext::Unpacked::Dilation).toIntVector(),
      unpacked_.get(Conv2dPackedContext::Unpacked::Groups).toInt(),
      get_optional_scalar(unpacked_, Conv2dPackedContext::Unpacked::OutputMin),
      get_optional_scalar(unpacked_, Conv2dPackedContext::Unpacked::OutputMax));
}

c10::intrusive_ptr<Conv2dOpContext> conv2d_clamp_prepack(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  return c10::make_intrusive<Conv2dOpContext>(Conv2dOpContext::create(
      std::move(weight),
      std::move(bias),
      std::move(stride),
      std::move(padding),
      std::move(dilation),
      /* transposed = */ false,
      /* output_padding = */ {0},
      groups,
      output_min,
      output_max));
}

Tensor conv2d_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dOpContext>& context) {
  return context->run(input);
}

std::vector<int64_t> conv_plan_counters_snapshot() {
  const VulkanConvPlanCounters& counters = conv_plan_counters();
  return {
      static_cast<int64_t>(counters.total.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.pointwise_1x1_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.pointwise_1x1_as_linear_hit.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.known_bad_large_pointwise.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.cpu_fallback.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_layout.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_dtype.load(std::memory_order_relaxed)),
  };
}

std::vector<int64_t> pointwise_conv_route_counters_snapshot() {
  const VulkanPointwiseConvRouteCounters& counters =
      pointwise_conv_route_counters();
  return {
      static_cast<int64_t>(
          counters.total_1x1.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.specialized_1x1_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.generic_1x1_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_not_direct_buffer.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_input_not_buffer.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_input_not_direct_buffer.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_output_not_direct_buffer.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_storage_offset.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_dtype.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_groups.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_stride_padding_dilation.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_weight_layout.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_bias.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_shape.load(std::memory_order_relaxed)),
  };
}

std::vector<std::string> conv_aggregate_snapshot() {
  std::vector<std::pair<VulkanConvAggregateKey, VulkanConvAggregateValue>>
      entries = conv_aggregate_profiler().snapshot();
  std::sort(
      entries.begin(),
      entries.end(),
      [](const auto& lhs, const auto& rhs) {
        const uint64_t lhs_bytes =
            lhs.second.input_bytes + lhs.second.output_bytes +
            lhs.second.weight_bytes;
        const uint64_t rhs_bytes =
            rhs.second.input_bytes + rhs.second.output_bytes +
            rhs.second.weight_bytes;
        if (lhs_bytes != rhs_bytes) {
          return lhs_bytes > rhs_bytes;
        }
        return lhs.second.count > rhs.second.count;
      });

  std::vector<std::string> out;
  out.reserve(entries.size());
  for (const auto& entry : entries) {
    const VulkanConvAggregateKey& key = entry.first;
    const VulkanConvAggregateValue& value = entry.second;
    std::ostringstream stream;
    stream << "conv_aggregate"
           << " selected=" << conv_plan_selected_name(key.selected)
           << " reject=" << conv_reject_reason_name(key.reject)
           << " kernel=" << key.kernel_name
           << " role=" << key.role
           << " count=" << value.count
           << " input=[" << key.n << ',' << key.cin << ',' << key.h << ','
           << key.w << ']'
           << " output_channels=" << key.cout
           << " weight=[" << key.cout << ',' << key.cin << ',' << key.kh
           << ',' << key.kw << ']'
           << " stride=[" << key.stride_h << ',' << key.stride_w << ']'
           << " padding=[" << key.pad_h << ',' << key.pad_w << ']'
           << " dilation=[" << key.dilation_h << ',' << key.dilation_w << ']'
           << " groups=" << key.groups
           << " input_direct=" << (key.input_direct ? 1 : 0)
           << " output_direct=" << (key.output_direct ? 1 : 0)
           << " weight_packed=" << (key.weight_packed ? 1 : 0)
           << " bias=" << (key.bias ? 1 : 0)
           << " pointwise=" << (key.pointwise ? 1 : 0)
           << " depthwise=" << (key.depthwise ? 1 : 0)
           << " sliding_window=" << (key.sliding_window ? 1 : 0)
           << " input_bytes=" << value.input_bytes
           << " output_bytes=" << value.output_bytes
           << " weight_bytes=" << value.weight_bytes;
    out.emplace_back(stream.str());
  }
  return out;
}

void reset_conv_aggregate() {
  conv_aggregate_profiler().reset();
}

void reset_conv_plan_counters() {
  VulkanConvPlanCounters& counters = conv_plan_counters();
  counters.total.store(0u, std::memory_order_relaxed);
  counters.pointwise_1x1_hit.store(0u, std::memory_order_relaxed);
  counters.pointwise_1x1_as_linear_hit.store(0u, std::memory_order_relaxed);
  counters.known_bad_large_pointwise.store(0u, std::memory_order_relaxed);
  counters.cpu_fallback.store(0u, std::memory_order_relaxed);
  counters.reject_layout.store(0u, std::memory_order_relaxed);
  counters.reject_dtype.store(0u, std::memory_order_relaxed);
}

void reset_pointwise_conv_route_counters() {
  VulkanPointwiseConvRouteCounters& counters = pointwise_conv_route_counters();
  counters.total_1x1.store(0u, std::memory_order_relaxed);
  counters.specialized_1x1_hit.store(0u, std::memory_order_relaxed);
  counters.generic_1x1_hit.store(0u, std::memory_order_relaxed);
  counters.reject_not_direct_buffer.store(0u, std::memory_order_relaxed);
  counters.reject_input_not_buffer.store(0u, std::memory_order_relaxed);
  counters.reject_input_not_direct_buffer.store(0u, std::memory_order_relaxed);
  counters.reject_output_not_direct_buffer.store(0u, std::memory_order_relaxed);
  counters.reject_storage_offset.store(0u, std::memory_order_relaxed);
  counters.reject_dtype.store(0u, std::memory_order_relaxed);
  counters.reject_groups.store(0u, std::memory_order_relaxed);
  counters.reject_stride_padding_dilation.store(0u, std::memory_order_relaxed);
  counters.reject_weight_layout.store(0u, std::memory_order_relaxed);
  counters.reject_bias.store(0u, std::memory_order_relaxed);
  counters.reject_shape.store(0u, std::memory_order_relaxed);
}

Conv1dPackedContext::Conv1dPackedContext(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const IntArrayRef stride_arg,
    const IntArrayRef padding_arg,
    const IntArrayRef dilation_arg,
    const int64_t groups)
    : unpacked_{c10::AnyType::get()} {
  const auto normalized_bias = utils::normalized_optional_tensor(bias);
  const std::vector<int64_t> logical_weight_sizes = weight.sizes().vec();
  Tensor buffer_weight = conv1d::upload_tensor_to_buffer(
      weight, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  Tensor buffer_bias = bias && bias->defined()
      ? conv1d::upload_tensor_to_buffer(
            *bias, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED)
      : conv1d::upload_tensor_to_buffer(
            at::zeros({weight.size(0)}, at::device(at::kCPU).dtype(at::kFloat)),
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  const size_t buffer_resident_nbytes =
      convert(buffer_weight).gpu_nbytes() + convert(buffer_bias).gpu_nbytes();
  buffer_weight_ = PackedWeightHandle(
      std::move(buffer_weight),
      std::move(buffer_bias),
      logical_weight_sizes,
      PackedWeightKind::Conv1d,
      bias && bias->defined(),
      PackedWeightResidencyClass::PersistentInference,
      false,
      api::ExecutionLayout::BUFFER_DIRECT,
      buffer_resident_nbytes);

  if (const auto cached_packed_weight = utils::lookup_packed_weight_handle(
          weight,
          normalized_bias,
          logical_weight_sizes,
          PackedWeightKind::Conv1d)) {
    packed_weight_ = *cached_packed_weight;
  } else {
    Tensor prepared_weight = utils::prepare_vulkan_execution_tensor(
        weight,
        utils::VulkanExecutionPlanKind::Conv1dPrepackWeight,
        convolution_request(utils::VulkanTensorRole::Weight));
    if (
        prepared_weight.scalar_type() == kBFloat16 ||
        prepared_weight.scalar_type() == kHalf) {
      prepared_weight = utils::cast_vulkan_tensor_dtype(prepared_weight, kFloat);
    }
    Tensor packed_bias = bias && bias->defined()
        ? utils::prepare_vulkan_execution_tensor(
              *bias,
              utils::VulkanExecutionPlanKind::Conv1dPrepackBias,
              convolution_request(utils::VulkanTensorRole::Bias))
        : utils::prepare_vulkan_execution_tensor(
              at::zeros(
                  {weight.size(0)},
                  at::device(at::kCPU).dtype(at::kFloat)),
              utils::VulkanExecutionPlanKind::Conv1dPrepackBias,
              convolution_request(utils::VulkanTensorRole::Bias));
    if (packed_bias.scalar_type() == kBFloat16 || packed_bias.scalar_type() == kHalf) {
      packed_bias = utils::cast_vulkan_tensor_dtype(packed_bias, kFloat);
    }
    packed_weight_ = utils::make_packed_weight_handle(
        convert(conv1d::pack_weights_using_width_packing(prepared_weight)),
        std::move(packed_bias),
        logical_weight_sizes,
        PackedWeightKind::Conv1d,
        bias && bias->defined());
    utils::store_packed_weight_handle(
        weight,
        normalized_bias,
        logical_weight_sizes,
        PackedWeightKind::Conv1d,
        packed_weight_);
  }
  stride_ = stride_arg.vec();
  padding_ = padding_arg.vec();
  dilation_ = dilation_arg.vec();
  groups_ = safe_downcast<int32_t>(groups);

  compute_shader_ = VK_KERNEL(conv1d);

  if (!at::globalContext().releaseWeightsWhenPrepacking()) {
    unpacked_.reserve(Unpacked::NumArgs);
    unpacked_.emplace_back(weight);
    unpacked_.emplace_back(normalized_bias);
    unpacked_.emplace_back(stride_arg.vec());
    unpacked_.emplace_back(padding_arg.vec());
    unpacked_.emplace_back(dilation_arg.vec());
    unpacked_.emplace_back(safe_downcast<int32_t>(groups));
  }
}

Conv1dPackedContext Conv1dPackedContext::pack(c10::impl::GenericList unpacked) {
  return Conv1dPackedContext(
      unpacked.get(Unpacked::Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Bias),
      unpacked.get(Unpacked::Stride).toIntVector(),
      unpacked.get(Unpacked::Padding).toIntVector(),
      unpacked.get(Unpacked::Dilation).toIntVector(),
      unpacked.get(Unpacked::Groups).toInt());
}

c10::intrusive_ptr<Conv1dPackedContext> create_conv1d_context(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups) {
  return c10::make_intrusive<Conv1dPackedContext>(
      Conv1dPackedContext(weight, bias, stride, padding, dilation, groups));
}

static Tensor convolution1d(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups) {
  Conv1dPackedContext conv1d_context =
      Conv1dPackedContext(weight, bias, stride, padding, dilation, groups);

  return run_conv1d_context(
      input, c10::make_intrusive<Conv1dPackedContext>(conv1d_context));
}

Tensor run_conv1d_context(
    const Tensor& input,
    const c10::intrusive_ptr<Conv1dPackedContext>& context) {
  const PackedWeightHandle& buffer_weight = context->buffer_weight();
  return conv1d::run_conv1d_buffer_context_impl(
      input,
      buffer_weight.weight(),
      buffer_weight.bias(),
      context->stride(),
      context->padding(),
      context->dilation(),
      context->groups());
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("convolution_overrideable", convolution);
  m.impl(TORCH_SELECTIVE_NAME("aten::conv1d"), TORCH_FN(convolution1d));
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
