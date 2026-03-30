#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/Functions.h>
#include <torch/library.h>
#include <cmath>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

constexpr int32_t kTiledSdpaLocalSizeX = 16;
constexpr int32_t kTiledSdpaMaxOutputsPerThread = 32;
constexpr int64_t kTiledSdpaMaxValueDim =
    static_cast<int64_t>(kTiledSdpaLocalSizeX) *
    static_cast<int64_t>(kTiledSdpaMaxOutputsPerThread);

Tensor maybe_scale_query(const Tensor& query, const double query_scale) {
  if (query_scale == 1.0) {
    return query;
  }
  return query.mul(query_scale);
}

bool can_use_tiled_sdpa_fast_path(
    const vTensor& v_query,
    const vTensor& v_key,
    const vTensor& v_value) {
  return v_query.storage_type() == api::StorageType::TEXTURE_3D &&
      v_key.storage_type() == api::StorageType::TEXTURE_3D &&
      v_value.storage_type() == api::StorageType::TEXTURE_3D &&
      v_query.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED &&
      v_key.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED &&
      v_value.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED &&
      v_value.sizes().size() == 3 &&
      v_value.sizes()[2] <= kTiledSdpaMaxValueDim;
}

Tensor scaled_dot_product_attention_tiled_3d_vulkan(
    const Tensor& query_arg,
    const Tensor& key_arg,
    const Tensor& value_arg) {
  api::AllocationScope allocation_scope("sdpa");
  TORCH_CHECK(
      query_arg.is_vulkan() && key_arg.is_vulkan() && value_arg.is_vulkan(),
      "Vulkan tiled SDPA expects Vulkan tensors");
  TORCH_CHECK(
      query_arg.dim() == 3 && key_arg.dim() == 3 && value_arg.dim() == 3,
      "Vulkan tiled SDPA expects 3D tensors");
  TORCH_CHECK(
      query_arg.size(0) == key_arg.size(0) &&
          query_arg.size(0) == value_arg.size(0) &&
          query_arg.size(2) == key_arg.size(2) &&
          key_arg.size(1) == value_arg.size(1),
      "Vulkan tiled SDPA expects matching [B, T, K] / [B, S, K] / [B, S, V] shapes");

  const Tensor query =
      query_arg.is_contiguous_or_false() ? query_arg : query_arg.contiguous();
  const Tensor key =
      key_arg.is_contiguous_or_false() ? key_arg : key_arg.contiguous();
  const Tensor value =
      value_arg.is_contiguous_or_false() ? value_arg : value_arg.contiguous();

  const vTensor& v_query = convert(query);
  const vTensor& v_key = convert(key);
  const vTensor& v_value = convert(value);

  TORCH_CHECK(
      can_use_tiled_sdpa_fast_path(v_query, v_key, v_value),
      "Vulkan tiled SDPA expects channels-packed TEXTURE_3D inputs with value dim <= ",
      kTiledSdpaMaxValueDim);

  api::Context* const context = api::context();

  vTensor v_output{
      context,
      {query.size(0), query.size(1), value.size(2)},
      v_value.dtype(),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
  };

  const struct Block final {
    ivec4 sizes;
    ivec4 tiled_info;
  } block{
      {
          safe_downcast<int32_t>(query.size(0)),
          safe_downcast<int32_t>(query.size(1)),
          safe_downcast<int32_t>(key.size(1)),
          safe_downcast<int32_t>(query.size(2)),
      },
      {
          safe_downcast<int32_t>(value.size(2)),
          kTiledSdpaLocalSizeX,
          kTiledSdpaMaxOutputsPerThread,
          safe_downcast<int32_t>(v_output.extents().data[2u]),
      },
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(scaled_dot_product_scores_value),
      pipeline_barrier,
      {
          static_cast<uint32_t>(kTiledSdpaLocalSizeX),
          v_output.extents().data[1u],
          v_output.extents().data[2u],
      },
      {
          static_cast<uint32_t>(kTiledSdpaLocalSizeX),
          1u,
          1u,
      },
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_query.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_key.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_value.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return convert(v_output);
}

std::tuple<Tensor, Tensor> scaled_dot_product_attention_math_vulkan(
    const Tensor& query_arg,
    const Tensor& key_arg,
    const Tensor& value_arg,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const std::optional<Tensor>& dropout_mask,
    std::optional<double> scale,
    bool enable_gqa) {
  api::AllocationScope allocation_scope("sdpa");
  TORCH_CHECK(
      query_arg.is_vulkan() && key_arg.is_vulkan() && value_arg.is_vulkan(),
      "Vulkan SDPA expects query, key, and value to already be Vulkan tensors");
  TORCH_CHECK(
      (query_arg.dim() == 3 || query_arg.dim() == 4) &&
          key_arg.dim() == query_arg.dim() &&
          value_arg.dim() == query_arg.dim(),
      "Vulkan SDPA currently supports matching 3D or 4D tensors");
  TORCH_CHECK(
      dropout_p == 0.0,
      "Vulkan SDPA currently supports inference-only dropout_p=0");
  TORCH_CHECK(
      !dropout_mask.has_value(),
      "Vulkan SDPA does not support explicit dropout masks");
  TORCH_CHECK(
      !attn_mask.has_value(),
      "Vulkan SDPA does not support attention masks yet");
  TORCH_CHECK(
      !is_causal,
      "Vulkan SDPA does not support causal masking yet");
  TORCH_CHECK(
      !enable_gqa,
      "Vulkan SDPA does not support GQA yet");
  TORCH_CHECK(
      query_arg.dim() == 3
          ? (query_arg.size(0) == key_arg.size(0) &&
             query_arg.size(0) == value_arg.size(0) &&
             query_arg.size(2) == key_arg.size(2) &&
             key_arg.size(1) == value_arg.size(1))
          : (query_arg.size(0) == key_arg.size(0) &&
             query_arg.size(0) == value_arg.size(0) &&
             query_arg.size(1) == key_arg.size(1) &&
             query_arg.size(1) == value_arg.size(1) &&
             query_arg.size(3) == key_arg.size(3) &&
             key_arg.size(2) == value_arg.size(2)),
      "Vulkan SDPA expects matching 3D [B, T, K] / [B, S, K] / [B, S, V] "
      "or 4D [B, H, T, K] / [B, H, S, K] / [B, H, S, V] shapes");

  const Tensor query =
      query_arg.is_contiguous_or_false() ? query_arg : query_arg.contiguous();
  const Tensor key =
      key_arg.is_contiguous_or_false() ? key_arg : key_arg.contiguous();
  const Tensor value =
      value_arg.is_contiguous_or_false() ? value_arg : value_arg.contiguous();

  const int64_t target_len = query.size(query.dim() - 2);
  const int64_t source_len = key.size(key.dim() - 2);
  const int64_t head_dim = query.size(query.dim() - 1);
  const int64_t value_dim = value.size(value.dim() - 1);

  const double sdpa_scale =
      scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim)));
  const double query_scale = sdpa_scale;

  Tensor query_3d;
  Tensor key_3d;
  Tensor value_3d;
  std::vector<int64_t> output_shape;
  std::vector<int64_t> attn_shape;

  if (query.dim() == 3) {
    query_3d = maybe_scale_query(query, query_scale);
    key_3d = key;
    value_3d = value;
    output_shape = {query.size(0), target_len, value_dim};
    attn_shape = {query.size(0), target_len, source_len};
  } else {
    const int64_t batch = query.size(0);
    const int64_t heads = query.size(1);
    query_3d = maybe_scale_query(
        query.reshape({batch * heads, target_len, head_dim}),
        query_scale);
    key_3d = key.reshape({batch * heads, source_len, head_dim});
    value_3d = value.reshape({batch * heads, source_len, value_dim});
    output_shape = {batch, heads, target_len, value_dim};
    attn_shape = {batch, heads, target_len, source_len};
  }

  Tensor attn = at::bmm(query_3d, key_3d.transpose(1, 2));
  attn = attn.softmax(-1);
  Tensor output = at::bmm(attn, value_3d);

  return std::make_tuple(
      output.reshape(output_shape),
      attn.reshape(attn_shape));
}

Tensor scaled_dot_product_attention_vulkan(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  api::AllocationScope allocation_scope("sdpa");
  TORCH_CHECK(
      query.is_vulkan() && key.is_vulkan() && value.is_vulkan(),
      "Vulkan SDPA expects query, key, and value to already be Vulkan tensors");
  TORCH_CHECK(
      (query.dim() == 3 || query.dim() == 4) &&
          key.dim() == query.dim() &&
          value.dim() == query.dim(),
      "Vulkan SDPA currently supports matching 3D or 4D tensors");
  TORCH_CHECK(
      dropout_p == 0.0,
      "Vulkan SDPA currently supports inference-only dropout_p=0");
  TORCH_CHECK(
      !attn_mask.has_value(),
      "Vulkan SDPA does not support attention masks yet");
  TORCH_CHECK(
      !is_causal,
      "Vulkan SDPA does not support causal masking yet");
  TORCH_CHECK(
      !enable_gqa,
      "Vulkan SDPA does not support GQA yet");

  const Tensor q =
      query.is_contiguous_or_false() ? query : query.contiguous();
  const Tensor k =
      key.is_contiguous_or_false() ? key : key.contiguous();
  const Tensor v =
      value.is_contiguous_or_false() ? value : value.contiguous();
  TORCH_CHECK(
      q.dim() == 3
          ? (q.size(0) == k.size(0) &&
             q.size(0) == v.size(0) &&
             q.size(2) == k.size(2) &&
             k.size(1) == v.size(1))
          : (q.size(0) == k.size(0) &&
             q.size(0) == v.size(0) &&
             q.size(1) == k.size(1) &&
             q.size(1) == v.size(1) &&
             q.size(3) == k.size(3) &&
             k.size(2) == v.size(2)),
      "Vulkan SDPA expects matching 3D [B, T, K] / [B, S, K] / [B, S, V] "
      "or 4D [B, H, T, K] / [B, H, S, K] / [B, H, S, V] shapes");

  const int64_t target_len = q.size(q.dim() - 2);
  const int64_t source_len = k.size(k.dim() - 2);
  const int64_t head_dim = q.size(q.dim() - 1);
  const int64_t value_dim = v.size(v.dim() - 1);

  const double sdpa_scale =
      scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim)));
  const double query_scale = sdpa_scale;

  Tensor q3;
  Tensor k3;
  Tensor v3;
  std::vector<int64_t> output_shape;

  if (q.dim() == 3) {
    q3 = maybe_scale_query(q, query_scale);
    k3 = k;
    v3 = v;
    output_shape = {q.size(0), target_len, value_dim};
  } else {
    const int64_t batch = q.size(0);
    const int64_t heads = q.size(1);
    q3 = maybe_scale_query(
        q.reshape({batch * heads, target_len, head_dim}),
        query_scale);
    k3 = k.reshape({batch * heads, source_len, head_dim});
    v3 = v.reshape({batch * heads, source_len, value_dim});
    output_shape = {batch, heads, target_len, value_dim};
  }

  Tensor output;
  if (can_use_tiled_sdpa_fast_path(convert(q3), convert(k3), convert(v3))) {
    output = scaled_dot_product_attention_tiled_3d_vulkan(q3, k3, v3);
  } else {
    return std::get<0>(scaled_dot_product_attention_math_vulkan(
        q,
        k,
        v,
        attn_mask,
        dropout_p,
        is_causal,
        std::nullopt,
        scale,
        enable_gqa));
  }
  return output.reshape(output_shape);
}

void set_softmax_kernel_params(
    const long long num_dims,
    const long long softmax_dim,
    const IntArrayRef v_input_sizes,
    api::ShaderInfo& shader_descriptor,
    api::utils::ivec4& input_shader_extents,
    api::utils::ivec4& early_exit,
    api::utils::ivec4& input_dim_stride,
    api::utils::ivec4& input_tensor_dims) {
  if (num_dims == 1) {
    early_exit.data[0u] = 1;
    input_dim_stride.data[0u] = 1;
    shader_descriptor = VK_KERNEL(softmax_batch_height_width);
  } else if (num_dims == 2) {
    // for height, width dim case, we can reuse a single shader
    // with vectorized parameters
    if (softmax_dim == 0) {
      early_exit.data[1u] = 1;
      input_dim_stride.data[1u] = 1;
      shader_descriptor = VK_KERNEL(softmax_batch_height_width);
    } else { // dim == 1
      early_exit.data[0u] = 1;
      input_dim_stride.data[0u] = 1;
      shader_descriptor = VK_KERNEL(softmax_batch_height_width);
    }
  } else if (num_dims == 3) {
    // for height, width dim case, we can reuse a single shader
    // with vectorized parameters
    for (uint32_t i = 0; i < num_dims; i++) {
      input_tensor_dims.data[i + 1] = safe_downcast<int32_t>(v_input_sizes[i]);
    }
    if (softmax_dim == 0) {
      early_exit.data[2u] = 1;
      input_dim_stride.data[2u] = 1;
      shader_descriptor = VK_KERNEL(softmax_channel);
    } else if (softmax_dim == 1) {
      early_exit.data[1u] = 1;
      input_dim_stride.data[1u] = 1;
      shader_descriptor = VK_KERNEL(softmax_batch_height_width);
    } else { // dim == 2
      early_exit.data[0u] = 1;
      input_dim_stride.data[0u] = 1;
      shader_descriptor = VK_KERNEL(softmax_batch_height_width);
    }
  } else {
    // assume num_dims is 4
    // for batch, height, width dim case, we can reuse a single shader
    // with vectorized parameters
    for (uint32_t i = 0; i < num_dims; i++) {
      input_tensor_dims.data[i] = safe_downcast<int32_t>(v_input_sizes[i]);
    }
    if (softmax_dim == 1) {
      // for 4-rank Tensor, softmax along channel dim case, the memory layout
      // forces a different shader algorithm than other dims
      input_shader_extents.data[2u] =
          v_input_sizes[Layout::Activation4D::batch];
      shader_descriptor = VK_KERNEL(softmax_channel);
    } else {
      if (softmax_dim == 0) {
        early_exit.data[2u] = safe_downcast<int32_t>(
            std::ceil(v_input_sizes[Layout::Activation4D::channels] / 4.0));
        input_dim_stride.data[2u] = safe_downcast<int32_t>(
            std::ceil(v_input_sizes[Layout::Activation4D::channels] / 4.0));
      } else if (softmax_dim == 2) {
        early_exit.data[1u] = 1;
        input_dim_stride.data[1u] = 1;
      } else { // dim == 3
        early_exit.data[0u] = 1;
        input_dim_stride.data[0u] = 1;
      }
      shader_descriptor = VK_KERNEL(softmax_batch_height_width);
    }
  }
}

Tensor softmax_internal(
    const at::Tensor& input_arg,
    const int64_t dim_arg,
    const bool half_to_float) {
  TORCH_CHECK(
      input_arg.dim() >= 1 && input_arg.dim() <= 4,
      "Vulkan softmax expects 1,2,3 or 4-dimensional input!");
  int64_t dim = utils::normalize(dim_arg, input_arg.dim());
  TORCH_CHECK(
      dim >= 0 && dim < input_arg.dim(),
      "Softmax dim input was ",
      dim,
      " out of range for Tensor input with dimensions ",
      input_arg.dim());
  api::Context* const context = api::context();

  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const vTensor& v_input = convert(input);

  vTensor v_output{
      context,
      v_input.sizes(),
      v_input.dtype(),
  };
  const api::utils::uvec3 global_workgroup_extents = v_output.extents();
  api::utils::ivec4 input_shader_extents = {
      safe_downcast<int32_t>(v_input.extents().data[0u]),
      safe_downcast<int32_t>(v_input.extents().data[1u]),
      safe_downcast<int32_t>(v_input.extents().data[2u]),
      0 // zero pad
  };
  // early_exit is the global workgroup position-based condition for
  // unnecessary invocations to exit.
  api::utils::ivec4 early_exit = {
      safe_downcast<int32_t>(v_input.extents().data[0u]),
      safe_downcast<int32_t>(v_input.extents().data[1u]),
      safe_downcast<int32_t>(v_input.extents().data[2u]),
      0 // zero pad
  };
  // for batch/height/width, they share the same shader
  // vectorized by input_dim_stride for each dimension case
  api::utils::ivec4 input_dim_stride = {
      0,
      0,
      0,
      0, // zero pad
  };
  api::utils::ivec4 input_tensor_dims = {
      0,
      0,
      0,
      0,
  };
  api::ShaderInfo shader_descriptor;
  set_softmax_kernel_params(
      input_arg.dim(),
      dim,
      v_input.sizes(),
      shader_descriptor,
      input_shader_extents,
      early_exit,
      input_dim_stride,
      input_tensor_dims);

  const struct Block final {
    ivec4 input_shader_extents;
    ivec4 input_tensor_dims;
    ivec4 input_dim_stride;
    ivec4 early_exit;
  } block{
      input_shader_extents, input_tensor_dims, input_dim_stride, early_exit};
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_workgroup_extents,
      // local work group size
      adaptive_work_group_size(global_workgroup_extents),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor softmax(
    const at::Tensor& input_arg,
    const int64_t dim,
    const bool half_to_float) {
  return softmax_internal(input_arg, dim, half_to_float);
}

Tensor log_softmax(
    const at::Tensor& input_arg,
    const int64_t dim,
    const bool half_to_float) {
  // After computing softmax, some values are so small that they are below the
  // float16 precision. These values are represented as 0 in float16 and result
  // in -inf when log is applied. According to Wikipedia:
  // https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Exponent_encoding,
  // the minimum strictly positive (subnormal) value is 2^−24 ≈ 5.9605 × 10^−8.
  // Therefore, we add 6 x 10^-8 to the output of softmax to avoid the numerical
  // issue.
  float epsilon = 6e-8;
  return softmax_internal(input_arg, dim, half_to_float).add(epsilon).log();
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl("_softmax", TORCH_FN(softmax));
  m.impl("_log_softmax", TORCH_FN(log_softmax));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::scaled_dot_product_attention"),
      TORCH_FN(scaled_dot_product_attention_vulkan));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_scaled_dot_product_attention_math"),
      TORCH_FN(scaled_dot_product_attention_math_vulkan));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
