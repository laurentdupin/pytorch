
#include <ATen/Context.h>

#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/vulkan/api/Utils.h>
#include <ATen/native/vulkan/impl/Packing.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/irange.h>

#include <atomic>
#include <array>
#include <deque>
#include <fstream>
#include <sstream>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
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

Tensor upcast_bfloat16_weight_for_vulkan_convolution(const Tensor& tensor) {
  if (tensor.scalar_type() != kBFloat16) {
    return tensor;
  }
  if (tensor.is_vulkan()) {
    return convert(tensor).storage_type() == api::StorageType::BUFFER
        ? utils::upcast_bfloat16_buffer_to_float(tensor)
        : tensor.cpu().to(kFloat).vulkan();
  }
  return tensor.to(kFloat);
}

std::optional<Tensor> upcast_bfloat16_bias_for_vulkan_convolution(
    const std::optional<Tensor>& tensor) {
  if (!tensor || !tensor->defined()) {
    return std::nullopt;
  }
  if (tensor->scalar_type() != kBFloat16) {
    return tensor;
  }
  if (tensor->is_vulkan()) {
    return convert(*tensor).storage_type() == api::StorageType::BUFFER
        ? utils::upcast_bfloat16_buffer_to_float(*tensor)
        : tensor->cpu().to(kFloat).vulkan();
  }
  return tensor->to(kFloat);
}

Tensor upcast_bfloat16_input_for_vulkan_convolution(const Tensor& tensor) {
  if (tensor.scalar_type() != kBFloat16) {
    return tensor;
  }
  if (tensor.is_vulkan()) {
    return convert(tensor).storage_type() == api::StorageType::BUFFER
        ? utils::upcast_bfloat16_buffer_to_float(tensor)
        : tensor.cpu().to(kFloat).vulkan();
  }
  return tensor.to(kFloat);
}

} // namespace

namespace conv2d {

inline bool has_bias(const std::optional<Tensor>& bias) {
  return bias && bias->defined();
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

constexpr size_t kPackedConv2dContextCacheSize = 128u;

const std::string& conv_cache_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_CONV_CACHE_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool conv_cache_logging_enabled() {
  return !conv_cache_log_path().empty();
}

struct ConvCacheLogState final {
  std::atomic<uint64_t> lookups{0u};
  std::atomic<uint64_t> hits{0u};
  std::atomic<uint64_t> stores{0u};
  std::atomic<uint64_t> vulkan_pack_weights{0u};
  std::atomic<uint64_t> vulkan_to_cpu_copies{0u};

  ~ConvCacheLogState() {
    if (!conv_cache_logging_enabled()) {
      return;
    }

    std::ofstream out(conv_cache_log_path(), std::ios::app);
    out << "conv_cache: lookups=" << lookups.load(std::memory_order_relaxed)
        << " hits=" << hits.load(std::memory_order_relaxed)
        << " stores=" << stores.load(std::memory_order_relaxed)
        << " vulkan_pack_weights="
        << vulkan_pack_weights.load(std::memory_order_relaxed)
        << " vulkan_to_cpu_copies="
        << vulkan_to_cpu_copies.load(std::memory_order_relaxed) << '\n';
  }
};

ConvCacheLogState& conv_cache_log_state() {
  static ConvCacheLogState state;
  return state;
}

struct PackedConv2dCacheEntry final {
  Tensor weight_ref;
  std::optional<Tensor> bias_ref;
  int64_t weight_version;
  int64_t bias_version;
  std::array<int64_t, 2> stride;
  std::array<int64_t, 2> padding;
  std::array<int64_t, 2> dilation;
  std::array<int64_t, 2> output_padding;
  int64_t groups;
  bool transposed;
  bool quantized;
  bool has_output_min;
  float output_min;
  bool has_output_max;
  float output_max;
  c10::intrusive_ptr<Conv2dPackedContext> context;
};

thread_local std::deque<PackedConv2dCacheEntry> packed_conv2d_context_cache;

std::array<int64_t, 2> make_param_key(const IntArrayRef values) {
  TORCH_INTERNAL_ASSERT(values.size() == 2, "Invalid convolution parameter");
  return {values[0], values[1]};
}

std::optional<Tensor> normalized_optional_tensor(
    const std::optional<Tensor>& tensor) {
  if (tensor && tensor->defined()) {
    return tensor;
  }
  return std::nullopt;
}

bool same_optional_tensor_impl(
    const std::optional<Tensor>& lhs,
    const std::optional<Tensor>& rhs) {
  if (lhs.has_value() != rhs.has_value()) {
    return false;
  }
  if (!lhs.has_value()) {
    return true;
  }
  return lhs->unsafeGetTensorImpl() == rhs->unsafeGetTensorImpl();
}

int64_t tensor_version_or_zero(const Tensor& tensor) {
  return tensor.is_inference() ? 0 : tensor._version();
}

bool has_inference_tensor(
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  const auto normalized_bias = normalized_optional_tensor(bias);
  return weight.is_inference() ||
      (normalized_bias && normalized_bias->is_inference());
}

std::optional<c10::intrusive_ptr<Conv2dPackedContext>> lookup_packed_conv2d_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const bool transposed,
    const bool quantized,
    const IntArrayRef output_padding,
    const int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  const auto normalized_bias = normalized_optional_tensor(bias);
  if (!weight.is_vulkan() || (normalized_bias && !normalized_bias->is_vulkan())) {
    return std::nullopt;
  }

  if (conv_cache_logging_enabled()) {
    conv_cache_log_state().lookups.fetch_add(1u, std::memory_order_relaxed);
  }

  const int64_t weight_version = tensor_version_or_zero(weight);
  const int64_t bias_version = normalized_bias
      ? tensor_version_or_zero(*normalized_bias)
      : 0u;
  const auto stride_key = make_param_key(stride);
  const auto padding_key = make_param_key(padding);
  const auto dilation_key = make_param_key(dilation);
  const auto output_padding_key = make_param_key(output_padding);
  const bool has_output_min = output_min.has_value();
  const float output_min_value =
      has_output_min ? output_min->to<float>() : 0.0f;
  const bool has_output_max = output_max.has_value();
  const float output_max_value =
      has_output_max ? output_max->to<float>() : 0.0f;

  for (auto it = packed_conv2d_context_cache.begin();
       it != packed_conv2d_context_cache.end();
       ++it) {
    if (it->weight_ref.unsafeGetTensorImpl() != weight.unsafeGetTensorImpl() ||
        it->weight_version != weight_version ||
        !same_optional_tensor_impl(it->bias_ref, normalized_bias) ||
        it->bias_version != bias_version || it->stride != stride_key ||
        it->padding != padding_key || it->dilation != dilation_key ||
        it->output_padding != output_padding_key || it->groups != groups ||
        it->transposed != transposed || it->quantized != quantized ||
        it->has_output_min != has_output_min ||
        (has_output_min && it->output_min != output_min_value) ||
        it->has_output_max != has_output_max ||
        (has_output_max && it->output_max != output_max_value)) {
      continue;
    }

    auto context = it->context;
    if (it != packed_conv2d_context_cache.begin()) {
      PackedConv2dCacheEntry entry = std::move(*it);
      packed_conv2d_context_cache.erase(it);
      packed_conv2d_context_cache.emplace_front(std::move(entry));
      context = packed_conv2d_context_cache.front().context;
    }
    if (conv_cache_logging_enabled()) {
      conv_cache_log_state().hits.fetch_add(1u, std::memory_order_relaxed);
    }
    return context;
  }

  return std::nullopt;
}

void store_packed_conv2d_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const bool transposed,
    const bool quantized,
    const IntArrayRef output_padding,
    const int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max,
    const c10::intrusive_ptr<Conv2dPackedContext>& context) {
  const auto normalized_bias = normalized_optional_tensor(bias);
  if (!weight.is_vulkan() || (normalized_bias && !normalized_bias->is_vulkan())) {
    return;
  }

  if (conv_cache_logging_enabled()) {
    conv_cache_log_state().stores.fetch_add(1u, std::memory_order_relaxed);
  }

  PackedConv2dCacheEntry entry;
  entry.weight_ref = weight;
  entry.bias_ref = normalized_bias;
  entry.weight_version = tensor_version_or_zero(weight);
  entry.bias_version =
      normalized_bias ? tensor_version_or_zero(*normalized_bias) : 0u;
  entry.stride = make_param_key(stride);
  entry.padding = make_param_key(padding);
  entry.dilation = make_param_key(dilation);
  entry.output_padding = make_param_key(output_padding);
  entry.groups = groups;
  entry.transposed = transposed;
  entry.quantized = quantized;
  entry.has_output_min = output_min.has_value();
  entry.output_min = output_min ? output_min->to<float>() : 0.0f;
  entry.has_output_max = output_max.has_value();
  entry.output_max = output_max ? output_max->to<float>() : 0.0f;
  entry.context = context;

  packed_conv2d_context_cache.emplace_front(std::move(entry));
  if (packed_conv2d_context_cache.size() > kPackedConv2dContextCacheSize) {
    packed_conv2d_context_cache.pop_back();
  }
}

Tensor copy_vulkan_tensor_to_cpu(const Tensor& src) {
  if (!src.is_vulkan()) {
    return src;
  }

  if (conv_cache_logging_enabled()) {
    conv_cache_log_state().vulkan_to_cpu_copies.fetch_add(
        1u, std::memory_order_relaxed);
  }

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
  if (conv_cache_logging_enabled() && weight_inp.is_vulkan()) {
    conv_cache_log_state().vulkan_pack_weights.fetch_add(
        1u, std::memory_order_relaxed);
  }

  // Raw Vulkan module weights are not in the shader-packed layout that the
  // convolution kernels expect. Re-materialize them on CPU first so they go
  // through the same rearrangement path as CPU-resident weights.
  const Tensor weight_source = copy_vulkan_tensor_to_cpu(weight_inp);
  Tensor weight_arg =
      quantized ? at::dequantize(weight_source) : weight_source;
  if (!quantized && weight_arg.scalar_type() == kBFloat16) {
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
  if (!quantized && bias_rearranged.scalar_type() == kBFloat16) {
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

bool can_run_bfloat16_buffer_conv2d(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const bool transposed,
    const bool quantized,
    const IntArrayRef output_padding) {
  // Disabled for now. The first buffer-native BF16 convolution path still
  // needs more backend bring-up, so BF16 conv currently widens through the
  // established float execution path.
  return false;

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

Tensor prepare_float_bias_buffer_for_bfloat16_conv2d(
    const std::optional<Tensor>& bias,
    const int64_t out_channels) {
  if (!bias || !bias->defined()) {
    Tensor cpu_zero_bias =
        at::zeros({out_channels}, at::device(at::kCPU).dtype(at::kFloat));
    return convert(ops::to_vulkan(cpu_zero_bias, api::StorageType::BUFFER));
  }

  std::optional<Tensor> float_bias =
      upcast_bfloat16_bias_for_vulkan_convolution(bias);
  Tensor prepared_bias = float_bias.value();
  if (prepared_bias.is_vulkan()) {
    vTensor v_bias = convert(prepared_bias);
    if (
        v_bias.storage_type() != api::StorageType::BUFFER ||
        !v_bias.has_direct_buffer_layout() ||
        v_bias.gpu_memory_layout() != api::GPUMemoryLayout::TENSOR_WIDTH_PACKED) {
      v_bias = utils::materialize_to_contiguous_buffer(
          v_bias, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
      prepared_bias = convert(v_bias);
    }
    return prepared_bias;
  }

  return convert(ops::to_vulkan(prepared_bias, api::StorageType::BUFFER));
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
      prepare_float_bias_buffer_for_bfloat16_conv2d(bias, weight.size(0));
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
      v_output.buffer_metadata(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_input.buffer_metadata(),
      v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.buffer_metadata(),
      v_bias.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_bias.buffer_metadata(),
      params.buffer());

  return convert(v_output);
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
      if (can_run_bfloat16_buffer_conv2d(
              input, weight, bias, transposed, false, output_padding)) {
        return run_bfloat16_buffer_conv2d(
            input, weight, bias, stride, padding, dilation, groups);
      }
      const Tensor compute_weight =
          upcast_bfloat16_weight_for_vulkan_convolution(weight);
      const std::optional<Tensor> compute_bias =
          upcast_bfloat16_bias_for_vulkan_convolution(bias);
  if (has_inference_tensor(compute_weight, compute_bias)) {
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
        std::nullopt);
    return run_conv2d_context(input, conv_context);
  }
  if (const auto cached_context = lookup_packed_conv2d_context(
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
          std::nullopt)) {
    return run_conv2d_context(input, *cached_context);
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
      groups);
  store_packed_conv2d_context(
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
      conv_context);

  return run_conv2d_context(input, conv_context);
}

} // namespace

namespace conv1d {

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
  Tensor input = upcast_bfloat16_input_for_vulkan_convolution(input_arg);
  if (!input.is_vulkan()) {
    input = input.vulkan();
  }
  const Tensor compute_weight =
      upcast_bfloat16_weight_for_vulkan_convolution(weight_arg);
  const Tensor weight =
      compute_weight.is_vulkan() ? compute_weight : compute_weight.vulkan();

  const IntArrayRef& input_sizes = input.sizes();
  const IntArrayRef& weight_sizes = weight.sizes();

  int32_t in_channels = static_cast<int32_t>(input_sizes[1]);
  int32_t out_channels = static_cast<int32_t>(weight_sizes[0]);
  int32_t kernel_size = static_cast<int32_t>(weight_sizes[2]);

  Tensor bias;
  if (bias_arg_opt) {
    const Tensor compute_bias =
        upcast_bfloat16_bias_for_vulkan_convolution(bias_arg_opt).value();
    if (compute_bias.is_vulkan()) {
      bias = compute_bias;
    } else {
      bias = compute_bias.vulkan();
    }
  } else {
    bias = at::zeros({out_channels}).vulkan();
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

  return convert(v_output);
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
    const std::optional<Scalar>& output_max)
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

  packed_.reserve(Packed::NumArgs);
  packed_.emplace_back(
      convert(pack_weights(weight, transposed, quantized, method)));
  packed_.emplace_back(
      convert(pack_biases(bias, weight, transposed, quantized)));
  packed_.emplace_back(compute_overlay_region(weight, dilation, transposed));
  packed_.emplace_back(pack_params(stride));
  packed_.emplace_back(pack_params(padding));
  packed_.emplace_back(output_padding);
  packed_.emplace_back(pack_params(dilation));
  packed_.emplace_back(transposed);
  packed_.emplace_back(quantized);
  packed_.emplace_back(safe_downcast<int32_t>(groups));
  packed_.emplace_back(
      output_min ? output_min->template to<float>()
                 : -std::numeric_limits<float>::infinity());
  packed_.emplace_back(
      output_max ? output_max->template to<float>()
                 : +std::numeric_limits<float>::infinity());
  packed_.emplace_back(method);
  packed_.emplace_back(weight.sizes().vec());

  compute_shader_ = conv2d::get_shader(
      weight.sizes(), stride, padding, dilation, method, transposed, quantized);

  if (!at::globalContext().releaseWeightsWhenPrepacking()) {
    unpacked_.reserve(Unpacked::NumArgs);
    unpacked_.emplace_back(weight);
    unpacked_.emplace_back(bias);
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
    int64_t zero_point) {
  api::Context* const context = api::context();
  Tensor input = upcast_bfloat16_input_for_vulkan_convolution(input_arg);
  TORCH_CHECK(input.is_vulkan(), "Input tensor must be Vulkan!");

  if (convert(input).storage_type() == api::StorageType::BUFFER) {
    input = input.add(0.0);
    if (convert(input).storage_type() == api::StorageType::BUFFER) {
      input = utils::ensure_texture_storage(input);
    }
  }
  if (convert(input).storage_type() == api::StorageType::BUFFER) {
    input = utils::ensure_texture_storage(input);
  }
  const vTensor& v_input = convert(input);

  // Extract everything from the PackedContext
  const Tensor weight =
      conv_context->get_val(Conv2dPackedContext::Packed::Weight).toTensor();
  const vTensor& v_weight = convert(weight);

  const auto quantized =
      conv_context->get_val(Conv2dPackedContext::Packed::isQuantized).toBool();
  api::AllocationScope allocation_scope(quantized ? "qconv" : "conv");

  Tensor bias =
      conv_context->get_val(Conv2dPackedContext::Packed::Bias).toTensor();

  const vTensor& v_bias = convert(bias);

  const auto overlay_region =
      conv_context->get_val(Conv2dPackedContext::Packed::OverlayRegion)
          .toIntVector();

  const auto stride =
      conv_context->get_val(Conv2dPackedContext::Packed::Stride).toIntVector();
  const auto padding =
      conv_context->get_val(Conv2dPackedContext::Packed::Padding).toIntVector();
  const auto output_padding =
      conv_context->get_val(Conv2dPackedContext::Packed::OutputPadding)
          .toIntVector();
  const auto dilation =
      conv_context->get_val(Conv2dPackedContext::Packed::Dilation)
          .toIntVector();

  const auto transposed =
      conv_context->get_val(Conv2dPackedContext::Packed::isTransposed).toBool();

  const float output_min = safe_downcast<float>(
      conv_context->get_val(Conv2dPackedContext::Packed::OutputMin).toDouble());
  const float output_max = safe_downcast<float>(
      conv_context->get_val(Conv2dPackedContext::Packed::OutputMax).toDouble());

  const Conv2dMethod method_ = static_cast<Conv2dMethod>(
      conv_context->get_val(Conv2dPackedContext::Packed::ConvMethod).toInt());

  const auto kernel_size =
      conv_context->get_val(Conv2dPackedContext::Packed::WeightSizes)
          .toIntVector();

  TORCH_CHECK(
      usable(input, quantized), "Input tensor not usable for convolution!");

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

  return convert(v_output);
}

Tensor run_conv2d_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context) {
  return run_conv2d_context_impl(input_arg, conv_context, 1.0f, 0u);
}

Tensor run_tconv2d_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context) {
  return run_conv2d_context_impl(input_arg, conv_context, 1.0f, 0u);
}

Tensor run_qconv2d_context(
    const Tensor& input_arg,
    double scale,
    int64_t zero_point,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context) {
  return run_conv2d_context_impl(input_arg, conv_context, scale, zero_point);
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

Conv1dPackedContext::Conv1dPackedContext(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const IntArrayRef stride_arg,
    const IntArrayRef padding_arg,
    const IntArrayRef dilation_arg,
    const int64_t groups)
    : unpacked_{c10::AnyType::get()} {
  packed_.reserve(Packed::NumArgs);
  packed_.emplace_back(
      convert(conv1d::pack_weights_using_width_packing(weight.vulkan())));
  packed_.emplace_back(bias->vulkan());
  packed_.emplace_back(stride_arg);
  packed_.emplace_back(padding_arg);
  packed_.emplace_back(dilation_arg);
  packed_.emplace_back(safe_downcast<int32_t>(groups));

  compute_shader_ = VK_KERNEL(conv1d);

  if (!at::globalContext().releaseWeightsWhenPrepacking()) {
    unpacked_.reserve(Unpacked::NumArgs);
    unpacked_.emplace_back(weight);
    unpacked_.emplace_back(bias);
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
  const Tensor weight =
      context->get_val(Conv1dPackedContext::Packed::Weight).toTensor();
  const std::optional<Tensor>& bias_opt =
      context->get_val(Conv1dPackedContext::Packed::Bias).toTensor();
  const auto stride =
      context->get_val(Conv1dPackedContext::Packed::Stride).toIntVector();
  const auto padding =
      context->get_val(Conv1dPackedContext::Packed::Padding).toIntVector();
  const auto dilation =
      context->get_val(Conv1dPackedContext::Packed::Dilation).toIntVector();
  const auto groups =
      context->get_val(Conv1dPackedContext::Packed::Groups).toInt();
  return conv1d::run_conv1d_context_impl(
      input, weight, bias_opt, stride, padding, dilation, groups);
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("convolution_overrideable", convolution);
  m.impl(TORCH_SELECTIVE_NAME("aten::conv1d"), TORCH_FN(convolution1d));
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
