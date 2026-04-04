#include <algorithm>
#include <ATen/Functions.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Reduction.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <c10/util/irange.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor mean_dim_buffer_chunk(
    const Tensor& prepared_input,
    const std::vector<int64_t>& output_sizes) {
  api::Context* const context = api::context();
  vTensor& v_input = convert(prepared_input);

  vTensor v_output{
      context,
      output_sizes,
      api::kFloat,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);

  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
      1u,
      1u,
  };
  context->submit_compute_job(
      VK_KERNEL(buffer_mean_dim),
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
      in_meta.buffer());

  return convert(v_output);
}

Tensor finalize_bfloat16_mean_output(
    const Tensor& output,
    const std::optional<ScalarType> dtype) {
  const ScalarType target_dtype =
      dtype.has_value() ? *dtype : c10::ScalarType::BFloat16;
  if (target_dtype == c10::ScalarType::Float) {
    return output;
  }
  return utils::cast_vulkan_tensor_dtype(output, target_dtype);
}

Tensor mean_cpu_fallback(
    const Tensor& self_arg,
    const std::optional<ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
  return at::mean(self_cpu, dtype).vulkan();
}

Tensor mean_dim_cpu_fallback(
    const Tensor& self_arg,
    int64_t dim,
    bool keepdim,
    const std::optional<ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
  return at::mean(self_cpu, dim, keepdim, dtype).vulkan();
}

void check_group_norm_inputs(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const int64_t channels,
    const int64_t num_groups) {
  TORCH_CHECK(
      num_groups > 0, "Expected num_groups to be greater than 0, got ", num_groups);
  TORCH_CHECK(
      input.dim() >= 2,
      "Expected group_norm input to have at least 2 dimensions, got ",
      input.dim());
  TORCH_CHECK(
      channels % num_groups == 0,
      "Expected number of channels in input to be divisible by num_groups, got input of shape ",
      input.sizes(),
      " and num_groups=",
      num_groups);
  TORCH_CHECK(
      !weight.defined() || (weight.dim() == 1 && weight.numel() == channels),
      "Expected weight to be a vector of size equal to the number of channels in input, but got weight of shape ",
      weight.sizes(),
      " and input of shape ",
      input.sizes());
  TORCH_CHECK(
      !bias.defined() || (bias.dim() == 1 && bias.numel() == channels),
      "Expected bias to be a vector of size equal to the number of channels in input, but got bias of shape ",
      bias.sizes(),
      " and input of shape ",
      input.sizes());
}

Tensor maybe_to_vulkan(const Tensor& tensor) {
  return tensor.is_vulkan() ? tensor : tensor.vulkan();
}

Tensor maybe_to_compute_dtype(
    const Tensor& tensor,
    const ScalarType compute_dtype) {
  if (!tensor.defined()) {
    return tensor;
  }
  Tensor out = maybe_to_vulkan(tensor);
  if (out.scalar_type() != compute_dtype) {
    out = utils::cast_vulkan_tensor_dtype(out, compute_dtype);
  }
  return out;
}

Tensor group_norm(
    const Tensor& input_arg,
    int64_t num_groups,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps,
    bool /* cudnn_enabled */) {
  Tensor input = maybe_to_vulkan(input_arg).contiguous();
  const Tensor weight = weight_opt.value_or(Tensor());
  const Tensor bias = bias_opt.value_or(Tensor());

  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  check_group_norm_inputs(input, weight, bias, C, num_groups);

  int64_t HxW = 1;
  for (const auto dim : c10::irange(2, input.dim())) {
    HxW *= input.size(dim);
  }
  const int64_t group_size = (C / num_groups) * HxW;
  const ScalarType output_dtype = input.scalar_type();
  const ScalarType compute_dtype = c10::ScalarType::Float;

  Tensor compute_input = maybe_to_compute_dtype(input, compute_dtype);
  Tensor compute_weight = maybe_to_compute_dtype(weight, compute_dtype);
  Tensor compute_bias = maybe_to_compute_dtype(bias, compute_dtype);

  Tensor reshaped = compute_input.reshape({1, N * num_groups, N ? group_size : 1});
  Tensor group_mean =
      at::mean(reshaped, /*dim=*/2, /*keepdim=*/true, c10::ScalarType::Float);
  Tensor centered = at::sub(reshaped, group_mean);
  Tensor group_var = at::mean(
      at::mul(centered, centered),
      /*dim=*/2,
      /*keepdim=*/true,
      c10::ScalarType::Float);
  Tensor group_rstd = at::rsqrt(at::add(group_var, eps));
  Tensor normalized =
      at::mul(centered, group_rstd).reshape(compute_input.sizes());

  std::vector<int64_t> affine_param_shape(input.dim(), 1);
  affine_param_shape[1] = C;
  if (compute_weight.defined()) {
    normalized =
        at::mul(normalized, compute_weight.reshape(affine_param_shape));
  }
  if (compute_bias.defined()) {
    normalized =
        at::add(normalized, compute_bias.reshape(affine_param_shape));
  }

  if (normalized.scalar_type() != output_dtype) {
    normalized = utils::cast_vulkan_tensor_dtype(normalized, output_dtype);
  }

  return normalized;
}

Tensor mean_all_buffer(
    const Tensor& prepared_input_arg,
    const std::optional<ScalarType> dtype) {
  api::AllocationScope allocation_scope("mean.buffer_all");
  api::Context* const context = api::context();

  const ScalarType target_dtype =
      resolve_vulkan_mean_dtype(prepared_input_arg.scalar_type(), dtype);
  Tensor prepared = prepared_input_arg;
  bool is_bfloat16_input = prepared.scalar_type() == c10::ScalarType::BFloat16;
  if (!is_bfloat16_input && prepared.scalar_type() != c10::ScalarType::Float) {
    prepared = utils::cast_vulkan_tensor_dtype(prepared, c10::ScalarType::Float);
  }

  if (is_bfloat16_input) {
    prepared = utils::cast_vulkan_tensor_dtype(prepared, c10::ScalarType::Float);
    is_bfloat16_input = false;
  }
  vTensor& v_input = convert(prepared);

  vTensor v_output{
      context,
      {},
      api::kFloat,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);

  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      is_bfloat16_input ? VK_KERNEL(buffer_mean_all_bfloat16)
                        : VK_KERNEL(buffer_mean_all),
      pipeline_barrier,
      {1u, 1u, 1u},
      {1u, 1u, 1u},
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ),
      in_meta.buffer());

  Tensor output = convert(v_output);
  if (target_dtype != c10::ScalarType::Float) {
    output = utils::cast_vulkan_tensor_dtype(output, target_dtype);
  }
  return output;
}

Tensor mean_dim_buffer(
    const Tensor& prepared_input_arg,
    int64_t dim,
    bool keepdim,
    const std::optional<ScalarType> dtype) {
  api::AllocationScope allocation_scope("mean.buffer_dim");

  const ScalarType target_dtype =
      resolve_vulkan_mean_dtype(prepared_input_arg.scalar_type(), dtype);
  Tensor prepared = prepared_input_arg;
  if (prepared.scalar_type() == c10::ScalarType::BFloat16) {
    prepared = utils::cast_vulkan_tensor_dtype(prepared, c10::ScalarType::Float);
  }

  if (prepared.scalar_type() != c10::ScalarType::Float) {
    prepared = utils::cast_vulkan_tensor_dtype(prepared, c10::ScalarType::Float);
  }

  Tensor canonical = dim == safe_downcast<int64_t>(prepared.dim()) - 1
      ? prepared
      : reduction::canonicalize_buffer_reduction_input(prepared, dim);
  const vTensor& v_input = convert(canonical);
  const std::vector<int64_t> output_sizes =
      reduction::reduced_output_sizes(
          v_input.sizes(),
          safe_downcast<int64_t>(v_input.sizes().size()) - 1,
          keepdim);
  Tensor output = mean_dim_buffer_chunk(canonical, output_sizes);
  output = reduction::restore_buffer_reduction_output_layout(
      output, prepared.sizes(), dim, keepdim);

  if (target_dtype != c10::ScalarType::Float) {
    output = utils::cast_vulkan_tensor_dtype(output, target_dtype);
  }
  return output;
}

Tensor mean_dim(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    const std::optional<ScalarType> dtype) {
  if (self.scalar_type() == c10::ScalarType::BFloat16) {
    return finalize_bfloat16_mean_output(
        at::mean(
            utils::cast_vulkan_tensor_dtype(self, c10::ScalarType::Float),
            dim,
            keepdim,
            c10::ScalarType::Float),
        dtype);
  }

  TORCH_CHECK(
      self.dim() >= 2 && self.dim() <= 4,
      "Vulkan mean_dim supports 2d, 3d, 4d tensors as input!");
  TORCH_CHECK(
      dim >= -self.dim() && dim < self.dim(),
      "Vulkan mean.dim dimension out of range expected to be in range of [",
      -self.dim(),
      ",",
      self.dim() - 1,
      "], but got ",
      dim);

  const auto plan = utils::build_vulkan_execution_plan(
      self, utils::VulkanExecutionPlanKind::ReductionDimInput);
  if (api::uses_buffer_execution(plan.execution_layout)) {
    dim = utils::normalize(dim, self.dim());
    return mean_dim_buffer(
        utils::prepare_vulkan_direct_buffer_execution_tensor(self, plan),
        dim,
        keepdim,
        dtype);
  }

  // Get the global Vulkan context
  api::Context* const context = api::context();

  // Cast the input Tensor to a vTensor
  Tensor input = utils::execute_vulkan_execution_plan(self, plan);
  const vTensor& v_input = convert(input);

  // Normalize dim into range [0, self.dim()]
  dim = utils::normalize(dim, self.dim());

  // Create the output texture
  std::vector<int64_t> output_size = v_input.sizes();
  uint32_t dim_size = output_size[dim];
  if (keepdim) {
    output_size[dim] = 1;
  } else {
    output_size.erase(output_size.begin() + dim);
  }

  const ScalarType type = resolve_vulkan_mean_dtype(self.scalar_type(), dtype);

  vTensor v_output{
      context,
      output_size,
      convert_dtype(type),
  };

  // Required to determine how to insert memory barriers in the command buffer
  api::PipelineBarrier pipeline_barrier{};

  // Shift dim into 4d range
  if (self.dim() < 4) {
    dim += (4 - self.dim());
  }

  // Create the params buffer
  const struct Block final {
    uvec2 dim_info;
    int32_t channel;
  } block{
      {static_cast<uint32_t>(dim), dim_size},
      static_cast<int32_t>(get_dim<Dim4D::Channel>(v_input)),
  };

  api::UniformParamsBuffer params(context, block);

  context->submit_compute_job(
      // shader descriptor
      keepdim ? VK_KERNEL(mean_dim_keepdim) : VK_KERNEL(mean_dim),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
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

Tensor mean_dim_IntList(
    const at::Tensor& self,
    const OptionalIntArrayRef opt_dim,
    bool keepdim,
    const std::optional<ScalarType> dtype) {
  if (
      !self.is_vulkan() ||
      (!is_vulkan_float_dtype(self.scalar_type()) &&
       self.scalar_type() != c10::ScalarType::BFloat16)) {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    const Tensor self_cpu = self.is_vulkan() ? self.cpu() : self;
    return at::mean(self_cpu, opt_dim, keepdim, dtype).vulkan();
  }

  TORCH_CHECK(
      opt_dim.has_value(), "Vulkan mean without a dim arg is not implemented");

  std::set<int64_t> dims_set;

  if (opt_dim.has_value()) {
    auto dims = opt_dim.value();
    for (const auto& d : dims) {
      TORCH_CHECK(
          d >= -self.dim() && d < self.dim(),
          "Vulkan mean.dim_IntList dimension out of range expected to be in range of [",
          -self.dim(),
          ",",
          self.dim() - 1,
          "], but got ",
          d);
      int64_t dim_normalized = utils::normalize(d, self.dim());
      if (dims_set.find(dim_normalized) != dims_set.end()) {
        TORCH_CHECK(
            false,
            "dim ",
            dim_normalized,
            " appears multiple times in the list of dims")
      }
      dims_set.insert(dim_normalized);
    }
    Tensor output = self;
    for (auto it = dims_set.rbegin(); it != dims_set.rend(); ++it) {
      output = mean_dim(output, *it, keepdim, dtype);
    }
    return output;
  }
  return self;
}

Tensor mean(const Tensor& self, const std::optional<ScalarType> dtype) {
  if (self.scalar_type() == c10::ScalarType::BFloat16) {
    return finalize_bfloat16_mean_output(
        at::mean(
            utils::cast_vulkan_tensor_dtype(self, c10::ScalarType::Float),
            c10::ScalarType::Float),
        dtype);
  }

  const auto plan = utils::build_vulkan_execution_plan(
      self, utils::VulkanExecutionPlanKind::ReductionAllInput);
  if (api::uses_buffer_execution(plan.execution_layout)) {
    return mean_all_buffer(
        utils::prepare_vulkan_direct_buffer_execution_tensor(self, plan), dtype);
  }

  return mean_cpu_fallback(self, dtype);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::mean.dim"), TORCH_FN(mean_dim_IntList));
  m.impl(TORCH_SELECTIVE_NAME("aten::mean"), TORCH_FN(mean));
  m.impl(TORCH_SELECTIVE_NAME("aten::group_norm"), TORCH_FN(group_norm));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
