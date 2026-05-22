#include <ATen/native/GridSamplerUtils.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <sstream>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/grid_sampler_2d.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

bool is_bilinear_zeros_grid_sampler_2d(
    const int64_t interpolation_mode,
    const int64_t padding_mode) {
  return static_cast<GridSamplerInterpolation>(interpolation_mode) ==
      GridSamplerInterpolation::Bilinear &&
      static_cast<GridSamplerPadding>(padding_mode) == GridSamplerPadding::Zeros;
}

bool can_run_grid_sampler_2d_buffer_float(
    const Tensor& input,
    const Tensor& grid,
    const int64_t interpolation_mode,
    const int64_t padding_mode) {
  return input.is_vulkan() && grid.is_vulkan() && input.scalar_type() == kFloat &&
      grid.scalar_type() == kFloat && input.dim() == 4 && grid.dim() == 4 &&
      grid.size(-1) == 2 && input.size(0) == grid.size(0) &&
      is_bilinear_zeros_grid_sampler_2d(interpolation_mode, padding_mode);
}

Tensor upload_grid_sampler_cpu_result_to_vulkan(
    const Tensor& cpu_result,
    const Tensor& prototype) {
  Tensor output = at::empty(
      cpu_result.sizes(),
      prototype.options()
          .device(prototype.device())
          .dtype(cpu_result.scalar_type()));
  ops::copy_(output, cpu_result.contiguous());
  api::context()->submit_pending_work_and_poll_retire();
  return record_tensor_write_and_return(
      output, "aten::grid_sampler_2d", "cpu_upload", {prototype});
}

Tensor grid_sampler_2d_cpu_fallback(
    const Tensor& input,
    const Tensor& grid,
    const int64_t interpolation_mode,
    const int64_t padding_mode,
    const bool align_corners) {
  report_vulkan_cpu_fallback(
      "aten::grid_sampler_2d",
      "unsupported_grid_sampler_2d_cpu_fallback",
      {input, grid},
      VulkanCpuFallbackKind::SyncReadback);
  utils::log_vulkan_op_hit("aten::grid_sampler_2d.cpu_fallback");

  Tensor result_cpu;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    const Tensor input_cpu =
        input.is_vulkan() ? input.detach().cpu() : input.detach();
    const Tensor grid_cpu =
        grid.is_vulkan() ? grid.detach().cpu() : grid.detach();
    result_cpu = at::grid_sampler_2d(
        input_cpu,
        grid_cpu,
        interpolation_mode,
        padding_mode,
        align_corners);
  }
  return upload_grid_sampler_cpu_result_to_vulkan(result_cpu, input);
}

api::UniformParamsBuffer make_grid_sampler_2d_params(
    api::Context* const context,
    const bool align_corners) {
  const struct Params final {
    uint32_t align_corners;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
  } params{
      static_cast<uint32_t>(align_corners),
      0u,
      0u,
      0u,
  };
  return api::UniformParamsBuffer(context, params);
}

std::string grid_sampler_2d_op_hit_label(
    const Tensor& input,
    const Tensor& grid,
    const int64_t interpolation_mode,
    const int64_t padding_mode,
    const bool align_corners) {
  std::ostringstream stream;
  stream << "aten::grid_sampler_2d.buffer_float input=" << input.sizes()
         << " grid=" << grid.sizes()
         << " interpolation_mode=" << interpolation_mode
         << " padding_mode=" << padding_mode
         << " align_corners=" << (align_corners ? 1 : 0);
  return stream.str();
}

Tensor grid_sampler_2d(
    const Tensor& input_arg,
    const Tensor& grid_arg,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  if (!can_run_grid_sampler_2d_buffer_float(
          input_arg, grid_arg, interpolation_mode, padding_mode)) {
    return grid_sampler_2d_cpu_fallback(
        input_arg, grid_arg, interpolation_mode, padding_mode, align_corners);
  }

  api::Context* const context = api::context();
  Tensor input = utils::ensure_buffer_storage(input_arg);
  Tensor grid = utils::ensure_buffer_storage(grid_arg);
  const vTensor& v_input = convert(input);
  const vTensor& v_grid = convert(grid);
  TORCH_CHECK(
      v_input.storage_type() == api::StorageType::BUFFER &&
          v_input.dtype() == api::kFloat &&
          utils::supports_buffer_elementwise_compute(v_input),
      "Vulkan grid_sampler_2d expects float buffer-backed input");
  TORCH_CHECK(
      v_grid.storage_type() == api::StorageType::BUFFER &&
          v_grid.dtype() == api::kFloat &&
          utils::supports_buffer_elementwise_compute(v_grid),
      "Vulkan grid_sampler_2d expects float buffer-backed grid");

  const std::vector<int64_t> output_sizes{
      input_arg.size(0),
      input_arg.size(1),
      grid_arg.size(1),
      grid_arg.size(2),
  };
  Tensor output = utils::create_buffer_tensor(output_sizes, input_arg.scalar_type());
  vTensor& v_output = convert(output);

  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(v_output.numel()),
      1u,
      1u,
  };
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);
  api::UniformParamsBuffer grid_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_grid);
  api::UniformParamsBuffer params =
      make_grid_sampler_2d_params(context, align_corners);

  utils::log_vulkan_op_hit(grid_sampler_2d_op_hit_label(
      input_arg, grid_arg, interpolation_mode, padding_mode, align_corners));
  context->submit_compute_job(
      VK_KERNEL(grid_sampler_2d_bilinear_zeros_buffer_float),
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
      v_grid.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      grid_meta.buffer(),
      params.buffer());

  return record_tensor_write_and_return(
      output,
      "aten::grid_sampler_2d",
      "buffer_float_bilinear_zeros",
      {input_arg, grid_arg});
}

} // namespace

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::grid_sampler_2d"), TORCH_FN(grid_sampler_2d));
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
