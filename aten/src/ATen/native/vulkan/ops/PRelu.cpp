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
#include <ATen/ops/_prelu_kernel.h>
#include <ATen/ops/empty.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

bool can_run_prelu_buffer_float(const Tensor& self, const Tensor& weight) {
  if (!self.is_vulkan() || self.scalar_type() != at::kFloat ||
      weight.scalar_type() != at::kFloat || self.dim() < 1 || self.dim() > 4) {
    return false;
  }
  const int64_t weight_numel = weight.numel();
  if (weight_numel == 1) {
    return true;
  }
  return self.dim() > 1 && weight_numel == self.size(1);
}

Tensor upload_prelu_cpu_result_to_vulkan(
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
      output, "aten::_prelu_kernel", "cpu_upload", {prototype});
}

Tensor prelu_cpu_fallback(const Tensor& self, const Tensor& weight) {
  report_vulkan_cpu_fallback(
      "aten::_prelu_kernel",
      "unsupported_prelu_cpu_fallback",
      {self, weight},
      VulkanCpuFallbackKind::SyncReadback);
  utils::log_vulkan_op_hit("aten::_prelu_kernel.cpu_fallback");

  Tensor result_cpu;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    const Tensor self_cpu =
        self.is_vulkan() ? self.detach().cpu() : self.detach();
    const Tensor weight_cpu =
        weight.is_vulkan() ? weight.detach().cpu() : weight.detach();
    result_cpu = at::_prelu_kernel(self_cpu, weight_cpu);
  }
  return upload_prelu_cpu_result_to_vulkan(
      result_cpu, self.is_vulkan() ? self : weight);
}

api::UniformParamsBuffer make_prelu_params(
    api::Context* const context,
    const Tensor& self,
    const Tensor& weight) {
  const struct Params final {
    uint32_t weight_is_scalar;
    uint32_t input_dim;
    uint32_t reserved0;
    uint32_t reserved1;
  } params{
      static_cast<uint32_t>(weight.numel() == 1),
      safe_downcast<uint32_t>(self.dim()),
      0u,
      0u,
  };
  return api::UniformParamsBuffer(context, params);
}

std::string prelu_op_hit_label(const Tensor& self, const Tensor& weight) {
  std::ostringstream stream;
  stream << "aten::_prelu_kernel.buffer_float input=" << self.sizes()
         << " weight=" << weight.sizes()
         << " weight_numel=" << weight.numel()
         << " per_channel=" << (weight.numel() == 1 ? 0 : 1);
  return stream.str();
}

Tensor prelu_kernel(const Tensor& self_arg, const Tensor& weight_arg) {
  if (!can_run_prelu_buffer_float(self_arg, weight_arg)) {
    return prelu_cpu_fallback(self_arg, weight_arg);
  }

  api::Context* const context = api::context();
  Tensor self = utils::ensure_buffer_storage(self_arg);
  Tensor weight = weight_arg.is_vulkan() ? weight_arg : weight_arg.vulkan();
  weight = weight.reshape({weight.numel()});
  weight = utils::ensure_buffer_storage(weight);

  const vTensor& v_self = convert(self);
  const vTensor& v_weight = convert(weight);
  TORCH_CHECK(
      v_self.storage_type() == api::StorageType::BUFFER &&
          v_self.dtype() == api::kFloat &&
          utils::supports_buffer_elementwise_compute(v_self),
      "Vulkan PReLU expects float buffer-backed input");
  TORCH_CHECK(
      v_weight.storage_type() == api::StorageType::BUFFER &&
          v_weight.dtype() == api::kFloat &&
          utils::supports_buffer_elementwise_compute(v_weight),
      "Vulkan PReLU expects float buffer-backed weight");

  Tensor output = utils::create_buffer_tensor(self_arg.sizes(), self_arg.scalar_type());
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
      utils::make_buffer_compute_metadata_ubo(context, v_self);
  api::UniformParamsBuffer weight_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_weight);
  api::UniformParamsBuffer params =
      make_prelu_params(context, self_arg, weight_arg);

  utils::log_vulkan_op_hit(prelu_op_hit_label(self_arg, weight_arg));
  context->submit_compute_job(
      VK_KERNEL(prelu_buffer_float),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer(),
      v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      weight_meta.buffer(),
      params.buffer());

  return record_tensor_write_and_return(
      output, "aten::_prelu_kernel", "buffer_float", {self_arg, weight_arg});
}

} // namespace

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::_prelu_kernel"), TORCH_FN(prelu_kernel));
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
