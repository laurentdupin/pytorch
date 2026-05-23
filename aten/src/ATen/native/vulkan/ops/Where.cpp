#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/Functions.h>
#include <torch/library.h>

#include <algorithm>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

const Tensor& where_vulkan_prototype(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  if (self.is_vulkan()) {
    return self;
  }
  if (other.is_vulkan()) {
    return other;
  }
  return condition;
}

Tensor where_self_cpu_fallback(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other,
    const char* reason) {
  report_vulkan_cpu_fallback(
      "aten::where.self",
      reason,
      {condition, self, other},
      VulkanCpuFallbackKind::SyncReadback);

  Tensor result_cpu;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    const Tensor condition_cpu =
        condition.is_vulkan() ? condition.detach().cpu() : condition.detach();
    const Tensor self_cpu = self.is_vulkan() ? self.detach().cpu() : self.detach();
    const Tensor other_cpu =
        other.is_vulkan() ? other.detach().cpu() : other.detach();
    result_cpu = at::where(condition_cpu, self_cpu, other_cpu);
  }

  const Tensor& prototype = where_vulkan_prototype(condition, self, other);
  Tensor out = at::empty(
      result_cpu.sizes(),
      prototype.options()
          .device(prototype.device())
          .dtype(result_cpu.scalar_type()));
  ops::copy_(out, result_cpu);
  api::context()->submit_pending_work_and_poll_retire();
  return record_tensor_write_and_return(
      out, "aten::where", "cpu_upload", {condition, self, other});
}

bool same_shape(const Tensor& a, const Tensor& b) {
  return a.sizes().equals(b.sizes());
}

bool supports_where_self_buffer_float_shape(const Tensor& self) {
  return self.dim() == 1 || (self.dim() == 2 && self.size(0) == 1);
}

Tensor where_self_buffer_float(
    const Tensor& condition_arg,
    const Tensor& self_arg,
    const Tensor& other_arg) {
  TORCH_CHECK(
      condition_arg.scalar_type() == at::kBool &&
          self_arg.scalar_type() == at::kFloat &&
          other_arg.scalar_type() == at::kFloat,
      "Vulkan where.self buffer path expects bool condition and float branches");

  api::Context* const context = api::context();
  Tensor condition = utils::ensure_buffer_storage(
      condition_arg, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  Tensor self = utils::ensure_buffer_storage(
      self_arg, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  Tensor other = utils::ensure_buffer_storage(
      other_arg, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  Tensor out = utils::create_buffer_tensor(self_arg.sizes(), self_arg.scalar_type());

  const vTensor& v_condition = convert(condition);
  const vTensor& v_self = convert(self);
  const vTensor& v_other = convert(other);
  vTensor& v_out = convert(out);
  TORCH_CHECK(
      utils::supports_buffer_elementwise_compute(v_condition) &&
          utils::supports_buffer_elementwise_compute(v_self) &&
          utils::supports_buffer_elementwise_compute(v_other),
      "Vulkan where.self buffer path requires buffer-compatible tensors");

  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_out);
  api::UniformParamsBuffer condition_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_condition);
  api::UniformParamsBuffer self_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_self);
  api::UniformParamsBuffer other_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_other);
  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size{
      safe_downcast<uint32_t>(std::max<int64_t>(v_out.numel(), 1)),
      1u,
      1u,
  };

  utils::log_vulkan_op_hit("aten::where.self.buffer_float");
  context->submit_compute_job(
      VK_KERNEL(where_self_buffer_float),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_out.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_condition.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      condition_meta.buffer(),
      v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      self_meta.buffer(),
      v_other.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      other_meta.buffer());

  return record_tensor_write_and_return(
      out, "aten::where", "buffer_float", {condition_arg, self_arg, other_arg});
}

Tensor where_self(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  api::AllocationScope allocation_scope("where_self");
  if (!condition.is_vulkan() || !self.is_vulkan() || !other.is_vulkan()) {
    return where_self_cpu_fallback(
        condition, self, other, "mixed_device_cpu_fallback");
  }
  if (condition.dim() > 4 || self.dim() > 4 || other.dim() > 4) {
    return where_self_cpu_fallback(
        condition, self, other, "rank_gt_4_cpu_fallback");
  }
  if (!same_shape(condition, self) || !same_shape(self, other)) {
    return where_self_cpu_fallback(
        condition, self, other, "broadcast_cpu_fallback");
  }
  if (!supports_where_self_buffer_float_shape(self)) {
    return where_self_cpu_fallback(
        condition, self, other, "shape_cpu_fallback");
  }
  if (
      condition.scalar_type() != at::kBool || self.scalar_type() != at::kFloat ||
      other.scalar_type() != at::kFloat) {
    return where_self_cpu_fallback(
        condition, self, other, "dtype_cpu_fallback");
  }

  return where_self_buffer_float(condition, self, other);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::where.self"), TORCH_FN(where_self));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
