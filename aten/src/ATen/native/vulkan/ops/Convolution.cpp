
#include <ATen/Context.h>

#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/vulkan/api/Utils.h>
#include <ATen/native/vulkan/impl/Packing.h>
#include <ATen/native/vulkan/planning/ExecutionObjects.h>
#include <ATen/native/vulkan/ops/BinaryOp.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/irange.h>

#include <atomic>
#include <array>
#include <cstdlib>
#include <fstream>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/conv2d.h>
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
  if (context->should_sync_and_reclaim()) {
    context->sync_and_reclaim();
  }

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
  if (
      kernel_h == 1 && kernel_w == 1 && stride[0] == 1 && stride[1] == 1 &&
      padding[0] == 0 && padding[1] == 0) {
    return FloatBufferConv2dShaderKind::Pointwise1x1;
  }

  if (kernel_h == 3 && kernel_w == 3 && padding[0] == 1 && padding[1] == 1) {
    if (stride[0] == 1 && stride[1] == 1) {
      return FloatBufferConv2dShaderKind::Kernel3x3Stride1Pad1;
    }
    if (stride[0] == 2 && stride[1] == 2) {
      return FloatBufferConv2dShaderKind::Kernel3x3Stride2Pad1;
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
    case FloatBufferConv2dShaderKind::Kernel3x3Stride2Pad1:
      return {8u, 8u, 1u};
    case FloatBufferConv2dShaderKind::Generic:
      return adaptive_work_group_size(global_size);
  }

  return adaptive_work_group_size(global_size);
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
  switch (shader_kind) {
    case FloatBufferConv2dShaderKind::Pointwise1x1:
      utils::log_vulkan_op_hit("aten::convolution.buffer_float_1x1");
      break;
    case FloatBufferConv2dShaderKind::Kernel3x3Stride1Pad1:
      utils::log_vulkan_op_hit("aten::convolution.buffer_float_3x3_s1p1");
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
  switch (shader_kind) {
    case FloatBufferConv2dShaderKind::Pointwise1x1:
      shader = VK_KERNEL(conv2d_buffer_float_1x1);
      break;
    case FloatBufferConv2dShaderKind::Kernel3x3Stride1Pad1:
      shader = VK_KERNEL(conv2d_buffer_float_3x3_s1p1);
      break;
    case FloatBufferConv2dShaderKind::Kernel3x3Stride2Pad1:
      shader = VK_KERNEL(conv2d_buffer_float_3x3_s2p1);
      break;
    case FloatBufferConv2dShaderKind::Generic:
      break;
  }

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

  return output_arg != nullptr ? output_tensor : convert(v_output);
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

  return output_tensor;
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

  return output_arg != nullptr ? output_tensor : convert(v_output);
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
      const Tensor compute_weight = utils::prepare_vulkan_execution_tensor(
          weight,
          utils::VulkanExecutionPlanKind::Conv2dWeightSource,
          convolution_request(utils::VulkanTensorRole::Weight));
      const std::optional<Tensor> compute_bias =
          utils::prepare_optional_vulkan_execution_tensor(
              bias,
              utils::VulkanExecutionPlanKind::Conv2dBiasSource,
              convolution_request(utils::VulkanTensorRole::Bias));
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
        std::nullopt);
    return run_conv2d_context(input, conv_context);
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

  const auto normalized_bias = utils::normalized_optional_tensor(bias);
  const std::vector<int64_t> logical_weight_sizes = weight.sizes().vec();
  constexpr uint64_t kConvTransposedPackOption = 1u;
  constexpr uint64_t kConvBufferPackOption = 1u << 1;
  const PackedWeightKind packed_weight_kind =
      packed_weight_kind_for_conv2d_method(method);
  const bool use_float_buffer_packing = can_use_float_buffer_conv2d_prepack(
      weight, bias, transposed, quantized, output_padding);
  const uint64_t pack_options =
      (transposed ? kConvTransposedPackOption : 0u) |
      (use_float_buffer_packing ? kConvBufferPackOption : 0u);
  if (const auto cached_packed_weight = utils::lookup_packed_weight_handle(
          weight,
          normalized_bias,
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
    utils::store_packed_weight_handle(
        weight,
        normalized_bias,
        logical_weight_sizes,
        packed_weight_kind,
        packed_weight_,
        quantized,
        pack_options);
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
  Tensor input = utils::prepare_vulkan_execution_tensor(
      input_arg,
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
  const PackedWeightHandle& packed_weight = context->packed_weight();
  return conv1d::run_conv1d_context_impl(
      input,
      packed_weight.weight(),
      std::optional<Tensor>(packed_weight.bias()),
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
