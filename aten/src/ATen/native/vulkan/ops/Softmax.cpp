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

Tensor flatten_attention_batch_heads(
    const Tensor& tensor,
    const int64_t batch_heads,
    const int64_t sequence_length,
    const int64_t feature_size) {
  if (tensor.dim() == 3) {
    return tensor;
  }
  return tensor.reshape({batch_heads, sequence_length, feature_size});
}

Tensor repeat_attention_heads_for_gqa(
    const Tensor& tensor,
    const int64_t repeat_factor) {
  if (repeat_factor == 1) {
    return tensor;
  }

  TORCH_CHECK(
      tensor.dim() == 4,
      "Vulkan SDPA GQA expects 4D [B, H, T, D] key/value tensors");

  std::vector<Tensor> repeated_heads;
  repeated_heads.reserve(
      safe_downcast<size_t>(tensor.size(1) * repeat_factor));
  for (const auto head_idx : c10::irange(tensor.size(1))) {
    Tensor head = at::narrow(tensor, 1, head_idx, 1);
    for (const auto _ : c10::irange(repeat_factor)) {
      repeated_heads.push_back(head);
    }
  }
  return at::cat(repeated_heads, 1);
}

Tensor expand_attention_mask_3d(
    const Tensor& attn_mask,
    const int64_t batch,
    const int64_t heads,
    const int64_t target_len,
    const int64_t source_len) {
  TORCH_CHECK(
      attn_mask.dim() >= 2 && attn_mask.dim() <= 4,
      "Vulkan SDPA expects 2D, 3D, or 4D attention masks");

  if (attn_mask.dim() == 2) {
    TORCH_CHECK(
        attn_mask.size(0) == target_len && attn_mask.size(1) == source_len,
        "Vulkan SDPA 2D attention mask must match [T, S]");
    return attn_mask.unsqueeze(0).expand({batch * heads, target_len, source_len});
  }

  if (attn_mask.dim() == 3) {
    TORCH_CHECK(
        attn_mask.size(1) == target_len && attn_mask.size(2) == source_len,
        "Vulkan SDPA 3D attention mask must match [N, T, S]");
    if (attn_mask.size(0) == batch * heads) {
      return attn_mask;
    }
    TORCH_CHECK(
        attn_mask.size(0) == batch || attn_mask.size(0) == 1,
        "Vulkan SDPA 3D attention mask batch dimension must be 1, batch, or batch*heads");
    return attn_mask.unsqueeze(1)
        .expand({attn_mask.size(0), heads, target_len, source_len})
        .reshape({attn_mask.size(0) * heads, target_len, source_len})
        .expand({batch * heads, target_len, source_len});
  }

  TORCH_CHECK(
      attn_mask.size(2) == target_len && attn_mask.size(3) == source_len,
      "Vulkan SDPA 4D attention mask must match [B, H, T, S]");
  TORCH_CHECK(
      (attn_mask.size(0) == batch || attn_mask.size(0) == 1) &&
          (attn_mask.size(1) == heads || attn_mask.size(1) == 1),
      "Vulkan SDPA 4D attention mask batch/head dimensions must be 1 or match the input");
  return attn_mask.expand({batch, heads, target_len, source_len})
      .reshape({batch * heads, target_len, source_len});
}

Tensor make_attention_mask_additive(
    const Tensor& attn_mask,
    const Tensor& query,
    const int64_t batch,
    const int64_t heads,
    const int64_t target_len,
    const int64_t source_len) {
  if (attn_mask.scalar_type() == kBool) {
    Tensor mask_cpu = expand_attention_mask_3d(
                           attn_mask.is_vulkan() ? attn_mask.cpu() : attn_mask,
                           batch,
                           heads,
                           target_len,
                           source_len)
                          .to(kBool);
    Tensor additive_mask = at::zeros(
        mask_cpu.sizes(), query.options().device(at::kCPU).dtype(kFloat));
    additive_mask.masked_fill_(mask_cpu.logical_not(), -std::numeric_limits<float>::infinity());
    return additive_mask.to(query.scalar_type());
  }
  Tensor mask = expand_attention_mask_3d(
      attn_mask, batch, heads, target_len, source_len);
  return mask.to(query.scalar_type());
}

Tensor make_causal_attention_bias(
    const Tensor& query,
    const int64_t batch_heads,
    const int64_t target_len,
    const int64_t source_len) {
  Tensor causal_mask = at::ones(
      {target_len, source_len},
      query.options().device(at::kCPU).dtype(kBool));
  causal_mask = at::triu(causal_mask, 1);

  Tensor causal_bias = at::zeros(
      {target_len, source_len},
      query.options().device(at::kCPU).dtype(kFloat));
  causal_bias.masked_fill_(causal_mask, -std::numeric_limits<float>::infinity());
  return causal_bias.to(query.scalar_type())
      .unsqueeze(0)
      .expand({batch_heads, target_len, source_len});
}

Tensor prepare_attention_bias(
    const std::optional<Tensor>& attn_mask,
    const utils::VulkanAttentionPolicy& attention_policy,
    const Tensor& query,
    const int64_t batch,
    const int64_t heads,
    const int64_t target_len,
    const int64_t source_len) {
  const int64_t batch_heads = batch * heads;
  Tensor additive_bias;
  if (attn_mask && attn_mask->defined()) {
    additive_bias = make_attention_mask_additive(
        *attn_mask, query, batch, heads, target_len, source_len);
  }

  if (attention_policy.is_causal) {
    Tensor causal_bias =
        make_causal_attention_bias(query, batch_heads, target_len, source_len);
    if (additive_bias.defined()) {
      if (!additive_bias.is_vulkan()) {
        additive_bias = additive_bias.vulkan();
      }
      if (!causal_bias.is_vulkan()) {
        causal_bias = causal_bias.vulkan();
      }
      additive_bias = at::add(additive_bias, causal_bias);
    } else {
      additive_bias = causal_bias;
    }
  }

  if (!additive_bias.defined()) {
    return additive_bias;
  }

  return utils::prepare_vulkan_execution_tensor(
      additive_bias, attention_policy.mask_plan_kind);
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

  const Tensor query_texture = utils::prepare_vulkan_execution_tensor(
      query, utils::VulkanExecutionPlanKind::TextureComputeInput);
  const Tensor key_texture = utils::prepare_vulkan_execution_tensor(
      key, utils::VulkanExecutionPlanKind::TextureComputeInput);
  const Tensor value_texture = utils::prepare_vulkan_execution_tensor(
      value, utils::VulkanExecutionPlanKind::TextureComputeInput);

  const vTensor& v_query = convert(query_texture);
  const vTensor& v_key = convert(key_texture);
  const vTensor& v_value = convert(value_texture);

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

std::tuple<Tensor, Tensor> scaled_dot_product_attention_math_vulkan_impl(
    const Tensor& query_arg,
    const Tensor& key_arg,
    const Tensor& value_arg,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const std::optional<Tensor>& dropout_mask,
    std::optional<double> scale,
    bool enable_gqa,
    const utils::VulkanAttentionPolicy& attention_policy) {
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
      query_arg.dim() == 3
          ? (query_arg.size(0) == key_arg.size(0) &&
             query_arg.size(0) == value_arg.size(0) &&
             query_arg.size(2) == key_arg.size(2) &&
             key_arg.size(1) == value_arg.size(1))
          : (query_arg.size(0) == key_arg.size(0) &&
             query_arg.size(0) == value_arg.size(0) &&
             query_arg.size(3) == key_arg.size(3) &&
             key_arg.size(2) == value_arg.size(2) &&
             (enable_gqa
                  ? (key_arg.size(1) == value_arg.size(1) &&
                     key_arg.size(1) > 0 &&
                     query_arg.size(1) % key_arg.size(1) == 0)
                  : (query_arg.size(1) == key_arg.size(1) &&
                     query_arg.size(1) == value_arg.size(1)))),
      "Vulkan SDPA expects matching 3D [B, T, K] / [B, S, K] / [B, S, V] "
      "or 4D [B, H, T, K] / [B, H, S, K] / [B, H, S, V] shapes");

  const Tensor query =
      query_arg.is_contiguous_or_false() ? query_arg : query_arg.contiguous();
  Tensor key = key_arg.is_contiguous_or_false() ? key_arg : key_arg.contiguous();
  Tensor value =
      value_arg.is_contiguous_or_false() ? value_arg : value_arg.contiguous();

  if (enable_gqa) {
    TORCH_CHECK(
        query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
        "Vulkan SDPA GQA currently supports 4D tensors only");
    TORCH_CHECK(
        key.size(1) == value.size(1) &&
            query.size(1) % key.size(1) == 0,
        "Vulkan SDPA GQA expects query heads to be divisible by key/value heads");
    const int64_t repeat_factor = query.size(1) / key.size(1);
    key = repeat_attention_heads_for_gqa(key, repeat_factor);
    value = repeat_attention_heads_for_gqa(value, repeat_factor);
  }

  const int64_t target_len = query.size(query.dim() - 2);
  const int64_t source_len = key.size(key.dim() - 2);
  const int64_t head_dim = query.size(query.dim() - 1);
  const int64_t value_dim = value.size(value.dim() - 1);

  const double sdpa_scale =
      scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim)));
  const double query_scale = sdpa_scale;

  const int64_t batch = query.dim() == 4 ? query.size(0) : query.size(0);
  const int64_t heads = query.dim() == 4 ? query.size(1) : 1;
  const int64_t batch_heads = batch * heads;

  Tensor query_3d = maybe_scale_query(
      flatten_attention_batch_heads(query, batch_heads, target_len, head_dim),
      query_scale);
  Tensor key_3d =
      flatten_attention_batch_heads(key, batch_heads, source_len, head_dim);
  Tensor value_3d =
      flatten_attention_batch_heads(value, batch_heads, source_len, value_dim);

  query_3d = utils::prepare_vulkan_execution_tensor(
      query_3d, attention_policy.query_plan_kind);
  key_3d = utils::prepare_vulkan_execution_tensor(
      key_3d, attention_policy.key_value_plan_kind);
  value_3d = utils::prepare_vulkan_execution_tensor(
      value_3d, attention_policy.key_value_plan_kind);

  Tensor attn = at::bmm(query_3d, key_3d.transpose(1, 2));
  Tensor additive_bias = prepare_attention_bias(
      attn_mask,
      attention_policy,
      query,
      batch,
      heads,
      target_len,
      source_len);
  if (additive_bias.defined()) {
    attn = at::add(attn, additive_bias);
  }
  attn = attn.softmax(-1);
  Tensor output = at::bmm(attn, value_3d);

  if (query.dim() == 3) {
    return std::make_tuple(output, attn);
  }

  return std::make_tuple(
      output.reshape({batch, heads, target_len, value_dim}),
      attn.reshape({batch, heads, target_len, source_len}));
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
  return scaled_dot_product_attention_math_vulkan_impl(
      query_arg,
      key_arg,
      value_arg,
      attn_mask,
      dropout_p,
      is_causal,
      dropout_mask,
      scale,
      enable_gqa,
      utils::build_vulkan_attention_policy(
          attn_mask, is_causal, enable_gqa, false, false));
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
  return std::get<0>(scaled_dot_product_attention_math_vulkan(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      std::nullopt,
      scale,
      enable_gqa));
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

  Tensor input = utils::prepare_vulkan_execution_tensor(
      input_arg, utils::VulkanExecutionPlanKind::TextureComputeInput);
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
      TORCH_SELECTIVE_NAME("aten::_scaled_dot_product_attention_math"),
      TORCH_FN(scaled_dot_product_attention_math_vulkan));
}

#endif /* USE_VULKAN_API */

} // namespace

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
