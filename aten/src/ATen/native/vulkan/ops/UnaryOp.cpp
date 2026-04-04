#include <ATen/ArrayRef.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/cos.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/log.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/sqrt.h>
#endif
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {
using namespace api::utils;

enum class UnaryOpKind : uint8_t {
  Exp,
  Sqrt,
  Log,
  Sin,
  Cos,
  Neg,
  Rsqrt,
};

bool needs_unary_cpu_fallback(const Tensor& tensor) {
  if (!tensor.is_vulkan()) {
    return false;
  }

  const vTensor& v_tensor = convert(tensor);
  return v_tensor.storage_type() == api::StorageType::BUFFER &&
      !utils::supports_buffer_elementwise_compute(v_tensor);
}

Tensor unary_op_cpu_fallback(const Tensor& self_arg, const UnaryOpKind op_kind) {
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
    switch (op_kind) {
      case UnaryOpKind::Exp:
        cpu_result = at::exp(self_cpu);
        break;
      case UnaryOpKind::Sqrt:
        cpu_result = at::sqrt(self_cpu);
        break;
      case UnaryOpKind::Log:
        cpu_result = at::log(self_cpu);
        break;
      case UnaryOpKind::Sin:
        cpu_result = at::sin(self_cpu);
        break;
      case UnaryOpKind::Cos:
        cpu_result = at::cos(self_cpu);
        break;
      case UnaryOpKind::Neg:
        cpu_result = at::neg(self_cpu);
        break;
      case UnaryOpKind::Rsqrt:
        cpu_result = at::rsqrt(self_cpu);
        break;
    }
  }
  return cpu_result.vulkan();
}

Tensor unary_op_buffer(
    const Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::AllocationScope allocation_scope("unary_op.buffer");
  api::Context* const context = api::context();

  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

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

  context->submit_compute_job(
      shader_descriptor,
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
      in_meta.buffer());

  return convert(v_output);
}

Tensor unary_op(
    const Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor,
    const api::ShaderInfo& buffer_shader_descriptor,
    const UnaryOpKind op_kind) {
  api::Context* const context = api::context();

  if (needs_unary_cpu_fallback(self_arg)) {
    return unary_op_cpu_fallback(self_arg, op_kind);
  }

  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const auto plan = utils::build_vulkan_execution_plan(
      self, utils::VulkanExecutionPlanKind::ElementwiseInput);
  if (api::uses_buffer_execution(plan.execution_layout)) {
    self = utils::prepare_vulkan_direct_buffer_execution_tensor(self, plan);
    return unary_op_buffer(self, buffer_shader_descriptor);
  }

  self = utils::prepare_vulkan_execution_tensor(
      self, utils::VulkanExecutionPlanKind::TextureComputeInput);

  const vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  const struct Block final {
    uvec3 extents;
    uint32_t fill0;
  } block{
      v_self.extents(),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
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
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor& unary_op_(Tensor& self_arg, const api::ShaderInfo& shader_descriptor) {
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);

  const struct Block final {
    uvec3 extents;
    uint32_t fill0;
  } block{
      v_self.extents(),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      // params buffer
      params.buffer());

  return self_arg;
}

Tensor exp(const Tensor& self_arg) {
  return unary_op(
      self_arg, VK_KERNEL(exp), VK_KERNEL(buffer_exp), UnaryOpKind::Exp);
}

Tensor& exp_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(exp_inplace));
}

Tensor sqrt(const Tensor& self_arg) {
  return unary_op(
      self_arg, VK_KERNEL(sqrt), VK_KERNEL(buffer_sqrt), UnaryOpKind::Sqrt);
}

Tensor& sqrt_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(sqrt_inplace));
}

Tensor log(const Tensor& self_arg) {
  return unary_op(
      self_arg, VK_KERNEL(log), VK_KERNEL(buffer_log), UnaryOpKind::Log);
}

Tensor& log_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(log_inplace));
}

Tensor sin(const Tensor& self_arg) {
  return unary_op(
      self_arg, VK_KERNEL(sin), VK_KERNEL(buffer_sin), UnaryOpKind::Sin);
}

Tensor& sin_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(sin_inplace));
}

Tensor cos(const Tensor& self_arg) {
  return unary_op(
      self_arg, VK_KERNEL(cos), VK_KERNEL(buffer_cos), UnaryOpKind::Cos);
}

Tensor& cos_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(cos_inplace));
}

Tensor neg(const Tensor& self_arg) {
  return unary_op(
      self_arg, VK_KERNEL(neg), VK_KERNEL(buffer_neg), UnaryOpKind::Neg);
}

Tensor& neg_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(neg_inplace));
}

Tensor rsqrt(const Tensor& self_arg) {
  return unary_op(
      self_arg, VK_KERNEL(rsqrt), VK_KERNEL(buffer_rsqrt), UnaryOpKind::Rsqrt);
}

Tensor& rsqrt_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(rsqrt_inplace));
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::exp"), TORCH_FN(exp));
  m.impl(TORCH_SELECTIVE_NAME("aten::exp_"), TORCH_FN(exp_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sqrt"), TORCH_FN(sqrt));
  m.impl(TORCH_SELECTIVE_NAME("aten::sqrt_"), TORCH_FN(sqrt_));
  m.impl(TORCH_SELECTIVE_NAME("aten::log"), TORCH_FN(log));
  m.impl(TORCH_SELECTIVE_NAME("aten::log_"), TORCH_FN(log_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sin"), TORCH_FN(sin));
  m.impl(TORCH_SELECTIVE_NAME("aten::sin_"), TORCH_FN(sin_));
  m.impl(TORCH_SELECTIVE_NAME("aten::cos"), TORCH_FN(cos));
  m.impl(TORCH_SELECTIVE_NAME("aten::cos_"), TORCH_FN(cos_));
  m.impl(TORCH_SELECTIVE_NAME("aten::neg"), TORCH_FN(neg));
  m.impl(TORCH_SELECTIVE_NAME("aten::neg_"), TORCH_FN(neg_));
  m.impl(TORCH_SELECTIVE_NAME("aten::rsqrt"), TORCH_FN(rsqrt));
  m.impl(TORCH_SELECTIVE_NAME("aten::rsqrt_"), TORCH_FN(rsqrt_));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
