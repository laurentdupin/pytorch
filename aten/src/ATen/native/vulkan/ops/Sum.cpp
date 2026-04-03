#include <algorithm>
#include <ATen/Functions.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

uvec4 make_logical_buffer_sizes(const std::vector<int64_t>& sizes) {
  return api::utils::make_whcn_uvec4(sizes);
}

std::vector<int64_t> calc_logical_contiguous_strides(
    const std::vector<int64_t>& sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (int idx = safe_downcast<int>(sizes.size()) - 2; idx >= 0; --idx) {
    strides[idx] = strides[idx + 1] * std::max<int64_t>(sizes[idx + 1], 1);
  }
  return strides;
}

uvec4 make_logical_buffer_strides(const std::vector<int64_t>& sizes) {
  return api::utils::make_whcn_uvec4(calc_logical_contiguous_strides(sizes));
}

uint32_t to_whcn_dim(const int64_t dim, const int64_t ndim) {
  return safe_downcast<uint32_t>(ndim - 1 - dim);
}

struct BufferDimReduceBlock final {
  uvec4 map_out_sizes;
  uvec4 map_out_strides;
  uvec4 write_out_sizes;
  uvec4 write_out_strides;
  uvec4 info;
};

std::vector<int64_t> reduced_output_sizes(
    const std::vector<int64_t>& input_sizes,
    const int64_t dim,
    const bool keepdim) {
  std::vector<int64_t> output_sizes = input_sizes;
  if (keepdim) {
    output_sizes.at(dim) = 1;
  } else {
    output_sizes.erase(output_sizes.begin() + dim);
  }
  return output_sizes;
}

BufferDimReduceBlock make_buffer_dim_reduce_block(
    const std::vector<int64_t>& input_sizes,
    const std::vector<int64_t>& output_sizes,
    const int64_t dim,
    const int64_t out_numel,
    const uint32_t reduce_offset,
    const uint32_t reduce_size) {
  std::vector<int64_t> map_out_sizes = input_sizes;
  map_out_sizes.at(dim) = 1;

  return {
      make_logical_buffer_sizes(map_out_sizes),
      make_logical_buffer_strides(map_out_sizes),
      make_logical_buffer_sizes(output_sizes),
      make_logical_buffer_strides(output_sizes),
      {
          to_whcn_dim(dim, safe_downcast<int64_t>(input_sizes.size())),
          reduce_size,
          safe_downcast<uint32_t>(out_numel),
          reduce_offset,
      },
  };
}

Tensor sum_dim_buffer_chunk(
    const Tensor& prepared_input,
    const std::vector<int64_t>& output_sizes,
    const int64_t dim,
    const uint32_t reduce_offset,
    const uint32_t reduce_size) {
  api::Context* const context = api::context();
  vTensor& v_input = convert(prepared_input);

  vTensor v_output{
      context,
      output_sizes,
      api::kFloat,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const BufferDimReduceBlock block = make_buffer_dim_reduce_block(
      v_input.sizes(),
      output_sizes,
      dim,
      v_output.numel(),
      reduce_offset,
      reduce_size);
  api::UniformParamsBuffer params(context, block);
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
  const uvec3 local_size = {64u, 1u, 1u};
  context->submit_compute_job(
      VK_KERNEL(buffer_sum_dim),
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
      params.buffer());

  return convert(v_output);
}

Tensor sum_cpu_fallback(
    const Tensor& self_arg,
    const std::optional<ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
  return at::sum(self_cpu, dtype).vulkan();
}

Tensor sum_dim_cpu_fallback(
    const Tensor& self_arg,
    int64_t dim,
    bool keepdim,
    const std::optional<ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
  return at::sum(self_cpu, {dim}, keepdim, dtype).vulkan();
}

Tensor finalize_bfloat16_sum_output(
    const Tensor& output,
    const std::optional<ScalarType> dtype) {
  const ScalarType target_dtype =
      resolve_vulkan_sum_dtype(c10::ScalarType::BFloat16, dtype);
  if (target_dtype == c10::ScalarType::Float) {
    return output;
  }
  return utils::cast_vulkan_tensor_dtype(output, target_dtype);
}

bool should_run_buffer_sum_all(const Tensor& self_arg) {
  if (!self_arg.is_vulkan()) {
    return false;
  }

  const vTensor& v_input = convert(self_arg);
  return utils::supports_buffer_reduction_compute(v_input) &&
      v_input.storage_type() == api::StorageType::BUFFER;
}

bool should_run_buffer_sum_dim(const Tensor& self_arg) {
  return false;
}

Tensor sum_all_buffer(
    const Tensor& self_arg,
    const std::optional<ScalarType> dtype) {
  api::AllocationScope allocation_scope("sum.buffer_all");
  api::Context* const context = api::context();

  const ScalarType target_dtype =
      resolve_vulkan_sum_dtype(self_arg.scalar_type(), dtype);
  Tensor prepared = self_arg;
  bool is_bfloat16_input = prepared.scalar_type() == c10::ScalarType::BFloat16;
  TORCH_CHECK(
      prepared.scalar_type() == c10::ScalarType::Float || is_bfloat16_input,
      "Vulkan buffer full sum currently only supports float and bfloat16 inputs");
  prepared = utils::ensure_buffer_storage(prepared);
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
      is_bfloat16_input ? VK_KERNEL(buffer_sum_all_bfloat16)
                        : VK_KERNEL(buffer_sum_all),
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

Tensor sum_dim_buffer(
    const Tensor& self_arg,
    int64_t dim,
    bool keepdim,
    const std::optional<ScalarType> dtype) {
  api::AllocationScope allocation_scope("sum.buffer_dim");

  const ScalarType target_dtype =
      resolve_vulkan_sum_dtype(self_arg.scalar_type(), dtype);
  Tensor prepared = self_arg;
  if (prepared.scalar_type() == c10::ScalarType::BFloat16) {
    prepared = utils::cast_vulkan_tensor_dtype(prepared, c10::ScalarType::Float);
  }

  TORCH_CHECK(
      prepared.scalar_type() == c10::ScalarType::Float,
      "Vulkan buffer dim sum currently only supports floating-point inputs");

  prepared = utils::ensure_buffer_storage(prepared);
  const vTensor& v_input = convert(prepared);
  const std::vector<int64_t> output_sizes =
      reduced_output_sizes(v_input.sizes(), dim, keepdim);
  Tensor output = sum_dim_buffer_chunk(
      prepared,
      output_sizes,
      dim,
      0u,
      safe_downcast<uint32_t>(v_input.sizes().at(dim)));

  if (target_dtype != c10::ScalarType::Float) {
    output = utils::cast_vulkan_tensor_dtype(output, target_dtype);
  }
  return output;
}

Tensor sum_dim(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    const std::optional<ScalarType> dtype) {
  if (self.scalar_type() == c10::ScalarType::BFloat16) {
    return finalize_bfloat16_sum_output(
        at::sum(
            utils::cast_vulkan_tensor_dtype(self, c10::ScalarType::Float),
            {dim},
            keepdim,
            c10::ScalarType::Float),
        dtype);
  }

  TORCH_CHECK(
      self.dim() >= 1 && self.dim() <= 4,
      "Vulkan sum.dim_IntList supports 1d, 2d, 3d, 4d tensors as input!");

  if (should_run_buffer_sum_dim(self)) {
    dim = utils::normalize(dim, self.dim());
    return sum_dim_buffer(self, dim, keepdim, dtype);
  }

  // Get the global Vulkan context
  api::Context* const context = api::context();

  // Cast the input Tensor to a vTensor
  Tensor input = self.is_vulkan() ? self : self.vulkan();
  if (convert(input).storage_type() == api::StorageType::BUFFER) {
    input = utils::ensure_texture_storage(input);
  }
  const vTensor& v_input = convert(input);

  // Create the output texture
  std::vector<int64_t> output_size = v_input.sizes();
  uint32_t dim_size = output_size[dim];
  if (keepdim) {
    output_size[dim] = 1;
  } else {
    output_size.erase(output_size.begin() + dim);
  }

  const ScalarType type = resolve_vulkan_sum_dtype(self.scalar_type(), dtype);

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
      keepdim ? VK_KERNEL(sum_dim_keepdim) : VK_KERNEL(sum_dim),
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

Tensor sum_dim_IntList(
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
    return at::sum(self_cpu, opt_dim, keepdim, dtype).vulkan();
  }

  TORCH_CHECK(
      opt_dim.has_value(),
      "Vulkan sum.dim_IntList without a dim arg is not implemented");

  std::set<int64_t> dims_set;
  if (opt_dim.has_value()) {
    auto dims = opt_dim.value();
    for (const auto& dim : dims) {
      // Do dim check before normalization to report to specified wrong dim
      // value to user
      TORCH_CHECK(
          dim >= -self.dim() && dim <= self.dim() - 1,
          "Vulkan sum.dim_IntList dimension out of range expected to be in range of [",
          -self.dim(),
          ",",
          self.dim() - 1,
          "], but got ",
          dim);
      // Normalize dim into range [0, self.dim() - 1]
      int64_t dim_normalized = utils::normalize(dim, self.dim());
      if (dims_set.find(dim_normalized) != dims_set.end()) {
        TORCH_CHECK(
            false,
            "dim ",
            dim_normalized,
            " appears multiple times in the list of dims")
      }
      dims_set.insert(dim_normalized);
    }
    Tensor result = self;
    // Reduce the higher dimensionalities first, otherwise when keepdim is
    // false, it will be reducing the wrong dimension.
    for (auto it = dims_set.rbegin(); it != dims_set.rend(); ++it) {
      result = sum_dim(result, *it, keepdim, dtype);
    }
    return result;
  }
  return self;
}

Tensor sum(const Tensor& self, const std::optional<ScalarType> dtype) {
  if (self.scalar_type() == c10::ScalarType::BFloat16) {
    return finalize_bfloat16_sum_output(
        at::sum(
            utils::cast_vulkan_tensor_dtype(self, c10::ScalarType::Float),
            c10::ScalarType::Float),
        dtype);
  }

  if (!is_vulkan_float_dtype(self.scalar_type())) {
    return sum_cpu_fallback(self, dtype);
  }

  if (should_run_buffer_sum_all(self)) {
    return sum_all_buffer(self, dtype);
  }

  std::vector<int64_t> dims;
  for (int64_t d = 0; d < self.dim(); d++) {
    // If any dimension has zero elements, we will shortcut to a zero-dim.
    if (self.size(d) == 0) {
      return self.new_zeros(
          {},
          at::device(at::kVulkan)
              .dtype(resolve_vulkan_sum_dtype(self.scalar_type(), dtype)));
    }

    dims.push_back(d);
  }

  return sum_dim_IntList(self, dims, false, dtype);
}

Tensor all(const Tensor& self) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  return at::all(self.cpu()).vulkan();
}

Tensor& all_out(const Tensor& self, Tensor& out) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  Tensor cpu_result = at::empty({0}, out.options().device(at::kCPU));
  at::all_out(cpu_result, self.cpu());

  TORCH_CHECK(
      out.sizes() == cpu_result.sizes(),
      "Vulkan all.out requires a pre-sized output tensor; resizing Vulkan outputs is not supported");
  ops::copy_(out, cpu_result);
  return out;
}

Tensor argmax(
    const Tensor& self,
    const std::optional<int64_t> dim,
    bool keepdim) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  return at::argmax(self.cpu(), dim, keepdim).vulkan();
}

Tensor& argmax_out(
    const Tensor& self,
    const std::optional<int64_t> dim,
    bool keepdim,
    Tensor& out) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  Tensor cpu_result = at::empty({0}, out.options().device(at::kCPU));
  at::argmax_out(cpu_result, self.cpu(), dim, keepdim);

  TORCH_CHECK(
      out.sizes() == cpu_result.sizes(),
      "Vulkan argmax.out requires a pre-sized output tensor; resizing Vulkan outputs is not supported");
  ops::copy_(out, cpu_result);
  return out;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::sum.dim_IntList"), TORCH_FN(sum_dim_IntList));
  m.impl(TORCH_SELECTIVE_NAME("aten::sum"), TORCH_FN(sum));
  m.impl(TORCH_SELECTIVE_NAME("aten::all"), TORCH_FN(all));
  m.impl(TORCH_SELECTIVE_NAME("aten::all.all_out"), TORCH_FN(all_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::argmax"), TORCH_FN(argmax));
  m.impl(TORCH_SELECTIVE_NAME("aten::argmax.out"), TORCH_FN(argmax_out));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
