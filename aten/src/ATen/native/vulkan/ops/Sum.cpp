#include <algorithm>
#include <ATen/Functions.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/Reduction.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <limits>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

constexpr int64_t kParallelReduceAllChunkSize = 1024;
constexpr int64_t kParallelReduceAllMinNumel = 4096;

Device vulkan_output_device(const Tensor& tensor) {
  return tensor.is_vulkan() ? tensor.device()
                            : Device(at::kVulkan, api::current_device());
}

Tensor sum_dim_buffer_chunk(
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
      VK_KERNEL(buffer_sum_dim),
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

  return record_tensor_write_and_return(
      convert(v_output), "aten::sum", "buffer_dim_chunk", {prepared_input});
}

Tensor max_dim_buffer_chunk(
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
      VK_KERNEL(buffer_max_dim),
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

  return record_tensor_write_and_return(
      convert(v_output), "aten::amax", "buffer_dim_chunk", {prepared_input});
}

Tensor sum_cpu_fallback(
    const Tensor& self_arg,
    const std::optional<ScalarType> dtype) {
  report_vulkan_cpu_fallback("aten::sum", "cpu_fallback", {self_arg});
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
  return record_tensor_write_and_return(
      at::sum(self_cpu, dtype).to(vulkan_output_device(self_arg)),
      "aten::sum",
      "cpu_fallback",
      {self_arg});
}

Tensor sum_dim_cpu_fallback(
    const Tensor& self_arg,
    int64_t dim,
    bool keepdim,
    const std::optional<ScalarType> dtype) {
  report_vulkan_cpu_fallback("aten::sum", "dim_cpu_fallback", {self_arg});
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
  return record_tensor_write_and_return(
      at::sum(self_cpu, {dim}, keepdim, dtype).to(vulkan_output_device(self_arg)),
      "aten::sum",
      "dim_cpu_fallback",
      {self_arg});
}

Tensor amax_cpu_fallback(
    const Tensor& self_arg,
    IntArrayRef dim,
    bool keepdim) {
  report_vulkan_cpu_fallback("aten::amax", "cpu_fallback", {self_arg});
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
  return record_tensor_write_and_return(
      at::amax(self_cpu, dim, keepdim).to(vulkan_output_device(self_arg)),
      "aten::amax",
      "cpu_fallback",
      {self_arg});
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

Tensor finalize_bfloat16_max_output(
    const Tensor& output,
    const ScalarType target_dtype) {
  if (target_dtype == c10::ScalarType::Float) {
    return output;
  }
  return utils::cast_vulkan_tensor_dtype(output, target_dtype);
}

Tensor reduce_all_buffer_chunk(
    const Tensor& prepared_input_arg,
    const api::ShaderInfo& shader) {
  api::Context* const context = api::context();
  vTensor& v_input = convert(prepared_input_arg);

  const int64_t output_numel = api::utils::div_up(
      api::utils::safe_downcast<int64_t>(v_input.numel()),
      kParallelReduceAllChunkSize);
  vTensor v_output{
      context,
      {output_numel},
      api::kFloat,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct {
    int32_t chunk_size;
    int32_t output_numel;
    int32_t reserved0;
    int32_t reserved1;
  } block{
      api::utils::safe_downcast<int32_t>(kParallelReduceAllChunkSize),
      api::utils::safe_downcast<int32_t>(output_numel),
      0,
      0,
  };

  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);
  api::UniformParamsBuffer params(context, block);

  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(std::max<int64_t>(output_numel, 1)),
      1u,
      1u,
  };
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
      v_input.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ),
      in_meta.buffer(),
      params.buffer());

  return record_tensor_write_and_return(
      convert(v_output), "aten::reduction", "buffer_all_chunk", {prepared_input_arg});
}

Tensor reduce_all_buffer_parallel(
    const Tensor& prepared_input_arg,
    const api::ShaderInfo& shader) {
  Tensor current = prepared_input_arg;
  while (current.numel() > 1) {
    current = reduce_all_buffer_chunk(current, shader);
  }
  return current.dim() == 0 ? current : current.reshape({});
}

Tensor sum_all_buffer(
    const Tensor& prepared_input_arg,
    const std::optional<ScalarType> dtype) {
  api::AllocationScope allocation_scope("sum.buffer_all");
  api::Context* const context = api::context();

  const ScalarType target_dtype =
      resolve_vulkan_sum_dtype(prepared_input_arg.scalar_type(), dtype);
  Tensor prepared = prepared_input_arg;
  bool is_bfloat16_input = prepared.scalar_type() == c10::ScalarType::BFloat16;
  TORCH_CHECK(
      prepared.scalar_type() == c10::ScalarType::Float || is_bfloat16_input,
      "Vulkan buffer full sum currently only supports float and bfloat16 inputs");
  vTensor& v_input = convert(prepared);

  if (!is_bfloat16_input && prepared.numel() >= kParallelReduceAllMinNumel) {
    std::vector<int64_t> dims;
    dims.reserve(prepared.dim());
    for (int64_t d = 0; d < prepared.dim(); ++d) {
      dims.push_back(d);
    }
    Tensor output = at::sum(prepared, dims, false, c10::ScalarType::Float);
    if (target_dtype != c10::ScalarType::Float) {
      output = utils::cast_vulkan_tensor_dtype(output, target_dtype);
    }
    return output;
  }

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
  return record_tensor_write_and_return(
      output, "aten::sum", "buffer_all", {prepared_input_arg});
}

Tensor max_all_buffer(const Tensor& prepared_input_arg) {
  api::AllocationScope allocation_scope("amax.buffer_all");
  api::Context* const context = api::context();

  const ScalarType target_dtype = prepared_input_arg.scalar_type();
  Tensor prepared = prepared_input_arg;
  if (prepared.scalar_type() == c10::ScalarType::BFloat16) {
    prepared = utils::cast_vulkan_tensor_dtype(prepared, c10::ScalarType::Float);
  }

  TORCH_CHECK(
      prepared.scalar_type() == c10::ScalarType::Float,
      "Vulkan buffer full max currently only supports float and bfloat16 inputs");

  if (prepared.numel() >= kParallelReduceAllMinNumel) {
    return finalize_bfloat16_max_output(
        reduce_all_buffer_parallel(prepared, VK_KERNEL(buffer_max_all_chunk)),
        target_dtype);
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
      VK_KERNEL(buffer_max_all),
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

  return record_tensor_write_and_return(
      finalize_bfloat16_max_output(convert(v_output), target_dtype),
      "aten::amax",
      "buffer_all",
      {prepared_input_arg});
}

Tensor min_all_buffer(const Tensor& prepared_input_arg) {
  api::AllocationScope allocation_scope("amin.buffer_all");
  api::Context* const context = api::context();

  const ScalarType target_dtype = prepared_input_arg.scalar_type();
  Tensor prepared = prepared_input_arg;
  if (prepared.scalar_type() == c10::ScalarType::BFloat16) {
    prepared = utils::cast_vulkan_tensor_dtype(prepared, c10::ScalarType::Float);
  }

  TORCH_CHECK(
      prepared.scalar_type() == c10::ScalarType::Float,
      "Vulkan buffer full min currently only supports float and bfloat16 inputs");

  if (prepared.numel() > 1) {
    return finalize_bfloat16_max_output(
        reduce_all_buffer_parallel(prepared, VK_KERNEL(buffer_min_all_chunk)),
        target_dtype);
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
      VK_KERNEL(buffer_min_all),
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

  return record_tensor_write_and_return(
      finalize_bfloat16_max_output(convert(v_output), target_dtype),
      "aten::amin",
      "buffer_all",
      {prepared_input_arg});
}

Tensor sum_dim_buffer(
    const Tensor& prepared_input_arg,
    int64_t dim,
    bool keepdim,
    const std::optional<ScalarType> dtype) {
  api::AllocationScope allocation_scope("sum.buffer_dim");

  const ScalarType target_dtype =
      resolve_vulkan_sum_dtype(prepared_input_arg.scalar_type(), dtype);
  Tensor prepared = prepared_input_arg;
  if (prepared.scalar_type() == c10::ScalarType::BFloat16) {
    prepared = utils::cast_vulkan_tensor_dtype(prepared, c10::ScalarType::Float);
  }

  TORCH_CHECK(
      prepared.scalar_type() == c10::ScalarType::Float,
      "Vulkan buffer dim sum currently only supports floating-point inputs");

  Tensor canonical = dim == safe_downcast<int64_t>(prepared.dim()) - 1
      ? prepared
      : reduction::canonicalize_buffer_reduction_input(prepared, dim);
  const vTensor& v_input = convert(canonical);
  const std::vector<int64_t> output_sizes =
      reduction::reduced_output_sizes(
          v_input.sizes(),
          safe_downcast<int64_t>(v_input.sizes().size()) - 1,
          keepdim);
  Tensor output = sum_dim_buffer_chunk(canonical, output_sizes);
  output = reduction::restore_buffer_reduction_output_layout(
      output, prepared.sizes(), dim, keepdim);

  if (target_dtype != c10::ScalarType::Float) {
    output = utils::cast_vulkan_tensor_dtype(output, target_dtype);
  }
  return record_tensor_write_and_return(
      output, "aten::sum", "buffer_dim", {prepared_input_arg});
}

Tensor max_dim_buffer(
    const Tensor& prepared_input_arg,
    int64_t dim,
    bool keepdim) {
  api::AllocationScope allocation_scope("amax.buffer_dim");

  const ScalarType target_dtype = prepared_input_arg.scalar_type();
  Tensor prepared = prepared_input_arg;
  if (prepared.scalar_type() == c10::ScalarType::BFloat16) {
    prepared = utils::cast_vulkan_tensor_dtype(prepared, c10::ScalarType::Float);
  }

  TORCH_CHECK(
      prepared.scalar_type() == c10::ScalarType::Float,
      "Vulkan buffer dim max currently only supports float and bfloat16 inputs");

  Tensor canonical = dim == safe_downcast<int64_t>(prepared.dim()) - 1
      ? prepared
      : reduction::canonicalize_buffer_reduction_input(prepared, dim);
  const vTensor& v_input = convert(canonical);
  const std::vector<int64_t> output_sizes =
      reduction::reduced_output_sizes(
          v_input.sizes(),
          safe_downcast<int64_t>(v_input.sizes().size()) - 1,
          keepdim);
  Tensor output = max_dim_buffer_chunk(canonical, output_sizes);
  output = reduction::restore_buffer_reduction_output_layout(
      output, prepared.sizes(), dim, keepdim);

  return record_tensor_write_and_return(
      finalize_bfloat16_max_output(output, target_dtype),
      "aten::amax",
      "buffer_dim",
      {prepared_input_arg});
}

Tensor sum_dim(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    const std::optional<ScalarType> dtype) {
  if (self.dim() > 4) {
    return sum_dim_cpu_fallback(self, dim, keepdim, dtype);
  }

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

  const auto plan = utils::build_vulkan_execution_plan(
      self, utils::VulkanExecutionPlanKind::ReductionDimInput);
  if (api::uses_buffer_execution(plan.execution_layout)) {
    dim = utils::normalize(dim, self.dim());
    return sum_dim_buffer(
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
  return record_tensor_write_and_return(
      convert(v_output), "aten::sum", "texture_dim", {self});
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
    report_vulkan_cpu_fallback(
        "aten::sum", "dim_IntList_dtype_cpu_fallback", {self});
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
    return sum_cpu_fallback(self, dtype);
  }

  if (!is_vulkan_float_dtype(self.scalar_type())) {
    return sum_cpu_fallback(self, dtype);
  }

  const auto plan = utils::build_vulkan_execution_plan(
      self, utils::VulkanExecutionPlanKind::ReductionAllInput);
  if (api::uses_buffer_execution(plan.execution_layout)) {
    return sum_all_buffer(
        utils::prepare_vulkan_direct_buffer_execution_tensor(self, plan), dtype);
  }

  std::vector<int64_t> dims;
  for (int64_t d = 0; d < self.dim(); d++) {
    // If any dimension has zero elements, we will shortcut to a zero-dim.
    if (self.size(d) == 0) {
      return self.new_zeros(
          {},
          self.options().dtype(resolve_vulkan_sum_dtype(self.scalar_type(), dtype)));
    }

    dims.push_back(d);
  }

  return sum_dim_IntList(self, dims, false, dtype);
}

Tensor amax_vulkan(const Tensor& self, IntArrayRef dim, bool keepdim) {
  if (
      !self.is_vulkan() ||
      (!is_vulkan_float_dtype(self.scalar_type()) &&
       self.scalar_type() != c10::ScalarType::BFloat16) ||
      self.dim() > 4) {
    return amax_cpu_fallback(self, dim, keepdim);
  }

  for (const auto d : c10::irange(self.dim())) {
    if (self.size(d) == 0) {
      return amax_cpu_fallback(self, dim, keepdim);
    }
  }

  const auto plan = utils::build_vulkan_execution_plan(
      self,
      dim.empty() ? utils::VulkanExecutionPlanKind::ReductionAllInput
                  : utils::VulkanExecutionPlanKind::ReductionDimInput);
  if (!api::uses_buffer_execution(plan.execution_layout)) {
    return amax_cpu_fallback(self, dim, keepdim);
  }

  Tensor prepared = utils::execute_vulkan_execution_plan(self, plan);

  if (dim.empty()) {
    return max_all_buffer(prepared);
  }

  std::set<int64_t> dims_set;
  for (const auto d : dim) {
    TORCH_CHECK(
        d >= -self.dim() && d < self.dim(),
        "Vulkan amax dimension out of range expected to be in range of [",
        -self.dim(),
        ",",
        self.dim() - 1,
        "], but got ",
        d);
    const int64_t dim_normalized = utils::normalize(d, self.dim());
    if (dims_set.find(dim_normalized) != dims_set.end()) {
      TORCH_CHECK(
          false,
          "dim ",
          dim_normalized,
          " appears multiple times in the list of dims");
    }
    dims_set.insert(dim_normalized);
  }

  Tensor result = prepared;
  for (auto it = dims_set.rbegin(); it != dims_set.rend(); ++it) {
    result = max_dim_buffer(result, *it, keepdim);
  }
  return result;
}

Tensor all_vulkan(const Tensor& self) {
  report_vulkan_cpu_fallback("aten::all", "cpu_fallback", {self});
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  return record_tensor_write_and_return(
      at::all(self.cpu()).vulkan(), "aten::all", "cpu_fallback", {self});
}

Tensor any_vulkan(const Tensor& self) {
  report_vulkan_cpu_fallback("aten::any", "cpu_fallback", {self});
  const Device output_device = vulkan_output_device(self);
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    const Tensor self_cpu = self.is_vulkan() ? self.cpu() : self;
    cpu_result = at::any(self_cpu);
  }
  return record_tensor_write_and_return(
      cpu_result.to(output_device), "aten::any", "cpu_fallback", {self});
}

Tensor any_dim(const Tensor& self, int64_t dim, bool keepdim) {
  report_vulkan_cpu_fallback("aten::any", "dim_cpu_fallback", {self});
  const Device output_device = vulkan_output_device(self);
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    const Tensor self_cpu = self.is_vulkan() ? self.cpu() : self;
    cpu_result = at::any(self_cpu, dim, keepdim);
  }
  return record_tensor_write_and_return(
      cpu_result.to(output_device), "aten::any", "dim_cpu_fallback", {self});
}

Tensor& all_out(const Tensor& self, Tensor& out) {
  report_vulkan_cpu_fallback("aten::all.out", "cpu_fallback", {self, out});
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  Tensor cpu_result = at::empty({0}, out.options().device(at::kCPU));
  at::all_out(cpu_result, self.cpu());

  Tensor vulkan_result = at::empty(cpu_result.sizes(), out.options());
  ops::copy_(vulkan_result, cpu_result);
  return rebind_vulkan_output(out, vulkan_result);
}

Tensor& any_all_out(const Tensor& self, Tensor& out) {
  report_vulkan_cpu_fallback("aten::any.out", "cpu_fallback", {self, out});
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    cpu_result = at::empty({0}, out.options().device(at::kCPU));
    const Tensor self_cpu = self.is_vulkan() ? self.cpu() : self;
    at::any_out(cpu_result, self_cpu);
  }

  Tensor vulkan_result = at::empty(cpu_result.sizes(), out.options());
  ops::copy_(vulkan_result, cpu_result);
  return rebind_vulkan_output(out, vulkan_result);
}

Tensor& any_dim_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& out) {
  report_vulkan_cpu_fallback("aten::any.out", "dim_cpu_fallback", {self, out});
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    cpu_result = at::empty({0}, out.options().device(at::kCPU));
    const Tensor self_cpu = self.is_vulkan() ? self.cpu() : self;
    at::any_out(cpu_result, self_cpu, dim, keepdim);
  }

  Tensor vulkan_result = at::empty(cpu_result.sizes(), out.options());
  ops::copy_(vulkan_result, cpu_result);
  return rebind_vulkan_output(out, vulkan_result);
}

Tensor argmax(
    const Tensor& self,
    const std::optional<int64_t> dim,
    bool keepdim) {
  report_vulkan_cpu_fallback("aten::argmax", "cpu_fallback", {self});
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  return record_tensor_write_and_return(
      at::argmax(self.cpu(), dim, keepdim).vulkan(),
      "aten::argmax",
      "cpu_fallback",
      {self});
}

Tensor max_all(const Tensor& self) {
  if (
      self.is_vulkan() &&
      (is_vulkan_float_dtype(self.scalar_type()) ||
       self.scalar_type() == c10::ScalarType::BFloat16)) {
    return amax_vulkan(self, {}, false);
  }
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  report_vulkan_cpu_fallback("aten::max", "cpu_fallback", {self});
  c10::InferenceMode inference_mode_guard(false);

  return record_tensor_write_and_return(
      at::max(self.cpu()).vulkan(), "aten::max", "cpu_fallback", {self});
}

Tensor min_all(const Tensor& self) {
  if (
      self.is_vulkan() &&
      (is_vulkan_float_dtype(self.scalar_type()) ||
       self.scalar_type() == c10::ScalarType::BFloat16)) {
    const auto plan = utils::build_vulkan_execution_plan(
        self, utils::VulkanExecutionPlanKind::ReductionAllInput);
    if (api::uses_buffer_execution(plan.execution_layout)) {
      return min_all_buffer(utils::execute_vulkan_execution_plan(self, plan));
    }
  }
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  report_vulkan_cpu_fallback("aten::min", "cpu_fallback", {self});
  c10::InferenceMode inference_mode_guard(false);

  return record_tensor_write_and_return(
      at::min(self.cpu()).vulkan(), "aten::min", "cpu_fallback", {self});
}

Tensor& argmax_out(
    const Tensor& self,
    const std::optional<int64_t> dim,
    bool keepdim,
    Tensor& out) {
  report_vulkan_cpu_fallback("aten::argmax.out", "cpu_fallback", {self, out});
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  Tensor cpu_result = at::empty({0}, out.options().device(at::kCPU));
  at::argmax_out(cpu_result, self.cpu(), dim, keepdim);

  Tensor vulkan_result = at::empty(cpu_result.sizes(), out.options());
  ops::copy_(vulkan_result, cpu_result);
  return rebind_vulkan_output(out, vulkan_result);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::sum.dim_IntList"), TORCH_FN(sum_dim_IntList));
  m.impl(TORCH_SELECTIVE_NAME("aten::sum"), TORCH_FN(sum));
  m.impl(TORCH_SELECTIVE_NAME("aten::amax"), TORCH_FN(amax_vulkan));
  m.impl(TORCH_SELECTIVE_NAME("aten::all"), TORCH_FN(all_vulkan));
  m.impl(TORCH_SELECTIVE_NAME("aten::all.all_out"), TORCH_FN(all_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::any"), TORCH_FN(any_vulkan));
  m.impl(TORCH_SELECTIVE_NAME("aten::any.all_out"), TORCH_FN(any_all_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::any.dim"), TORCH_FN(any_dim));
  m.impl(TORCH_SELECTIVE_NAME("aten::any.out"), TORCH_FN(any_dim_out));
  m.impl("max", TORCH_FN(max_all));
  m.impl("min", TORCH_FN(min_all));
  m.impl(TORCH_SELECTIVE_NAME("aten::argmax"), TORCH_FN(argmax));
  m.impl(TORCH_SELECTIVE_NAME("aten::argmax.out"), TORCH_FN(argmax_out));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
