#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/ops/Zero.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <c10/core/DefaultDtype.h>
#include <torch/library.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

struct VulkanZeroCounters final {
  std::atomic<uint64_t> total{0u};
  std::atomic<uint64_t> buffer_float_hit{0u};
  std::atomic<uint64_t> buffer_byte_hit{0u};
  std::atomic<uint64_t> buffer_byte_shader_hit{0u};
  std::atomic<uint64_t> buffer_cmd_fill_hit{0u};
  std::atomic<uint64_t> texture_hit{0u};
  std::atomic<uint64_t> reject_dim_gt_4{0u};
  std::atomic<uint64_t> reject_storage{0u};
  std::atomic<uint64_t> reject_dtype{0u};
  std::atomic<uint64_t> reject_int8_feature{0u};
  std::atomic<uint64_t> cpu_fallback{0u};
};

enum class VulkanZeroPath : uint8_t {
  Unknown = 0,
  BufferFloat,
  BufferByteShader,
  BufferCmdFill,
  Texture,
  CpuFallback,
};

VulkanZeroCounters& vulkan_zero_counters() {
  static VulkanZeroCounters counters;
  return counters;
}

const char* zero_path_name(const VulkanZeroPath path) {
  switch (path) {
    case VulkanZeroPath::BufferFloat:
      return "buffer_float";
    case VulkanZeroPath::BufferByteShader:
      return "buffer_byte_shader";
    case VulkanZeroPath::BufferCmdFill:
      return "buffer_cmd_fill";
    case VulkanZeroPath::Texture:
      return "texture";
    case VulkanZeroPath::CpuFallback:
      return "cpu_fallback";
    case VulkanZeroPath::Unknown:
      return "unknown";
  }
  return "unknown";
}

void append_vulkan_zero_plan_log(
    const Tensor& self,
    const VulkanZeroPath path,
    const char* reason) {
  const char* env = std::getenv("PYTORCH_VULKAN_ZERO_PLAN_LOG");
  if (!env || !*env) {
    return;
  }

  const vTensor& v_self = convert(self);
  std::ofstream out(env, std::ios::app);
  out << "zero_plan"
      << " path=" << zero_path_name(path)
      << " reason=" << (reason ? reason : "none")
      << " dtype=" << static_cast<int>(self.scalar_type())
      << " rank=" << self.dim()
      << " numel=" << self.numel()
      << " nbytes=" << self.nbytes()
      << " storage=" << static_cast<int>(v_self.storage_type())
      << " gpu_layout=" << static_cast<int>(v_self.gpu_memory_layout())
      << " direct=" << (v_self.has_direct_buffer_layout() ? 1 : 0)
      << " storage_offset=" << v_self.storage_offset()
      << " caller=" << api::current_allocation_label()
      << '\n';
}

Tensor& zero_cpu_fallback(Tensor& self) {
  report_vulkan_cpu_fallback(
      "aten::zero_", "unsupported_shape_storage_or_dtype", {self});
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);
  Tensor cpu_zeros = at::zeros(self.sizes(), self.options().device(at::kCPU));
  ops::copy_(self, cpu_zeros);
  return self;
}

Tensor& zero_buffer_uint8_(Tensor& self) {
  TORCH_CHECK(self.is_vulkan(), "Vulkan uint8 zero expects a Vulkan tensor");
  TORCH_CHECK(self.scalar_type() == at::kByte, "Expected Byte tensor");

  vTensor& v_self = convert(self);
  TORCH_CHECK(
      v_self.storage_type() == api::StorageType::BUFFER,
      "Byte zero expects buffer storage");

  api::Context* const context = api::context();
  api::UniformParamsBuffer self_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_self);
  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_self.numel(), 1)),
      1u,
      1u,
  };

  utils::log_vulkan_op_hit("aten::zero_.buffer_uint8");
  context->submit_compute_job(
      VK_KERNEL(zero_buffer_uint8),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_self.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      self_meta.buffer());

  return self;
}

Tensor& zero_(at::Tensor& self) {
  auto& counters = vulkan_zero_counters();
  counters.total.fetch_add(1u, std::memory_order_relaxed);
  vTensor& v_self = convert(self);
  if (self.dim() > 4) {
    counters.reject_dim_gt_4.fetch_add(1u, std::memory_order_relaxed);
    counters.cpu_fallback.fetch_add(1u, std::memory_order_relaxed);
    append_vulkan_zero_plan_log(self, VulkanZeroPath::CpuFallback, "dim_gt_4");
    return zero_cpu_fallback(self);
  }
  if (v_self.storage_type() == api::StorageType::BUFFER) {
    if (self.scalar_type() == at::kFloat) {
      counters.buffer_float_hit.fetch_add(1u, std::memory_order_relaxed);
      append_vulkan_zero_plan_log(self, VulkanZeroPath::BufferFloat, "none");
      return utils::fill_buffer_float_(self, 0.0f, "aten::zero_");
    }
    if (self.scalar_type() == at::kByte) {
      if (!api::context()->adapter_ptr()->supports_int8_buffer_arithmetic()) {
        counters.reject_int8_feature.fetch_add(1u, std::memory_order_relaxed);
        counters.cpu_fallback.fetch_add(1u, std::memory_order_relaxed);
        append_vulkan_zero_plan_log(
            self,
            VulkanZeroPath::CpuFallback,
            "missing_int8_buffer_arithmetic");
        return zero_cpu_fallback(self);
      }
      counters.buffer_byte_hit.fetch_add(1u, std::memory_order_relaxed);
      counters.buffer_byte_shader_hit.fetch_add(1u, std::memory_order_relaxed);
      append_vulkan_zero_plan_log(
          self, VulkanZeroPath::BufferByteShader, "none");
      return zero_buffer_uint8_(self);
    }
    counters.reject_dtype.fetch_add(1u, std::memory_order_relaxed);
    counters.cpu_fallback.fetch_add(1u, std::memory_order_relaxed);
    append_vulkan_zero_plan_log(
        self, VulkanZeroPath::CpuFallback, "unsupported_buffer_dtype");
    return zero_cpu_fallback(self);
  }
  if (!api::supports_texture_storage(v_self.dtype())) {
    counters.reject_storage.fetch_add(1u, std::memory_order_relaxed);
    counters.cpu_fallback.fetch_add(1u, std::memory_order_relaxed);
    append_vulkan_zero_plan_log(
        self, VulkanZeroPath::CpuFallback, "unsupported_texture_storage");
    return zero_cpu_fallback(self);
  }
  counters.texture_hit.fetch_add(1u, std::memory_order_relaxed);
  append_vulkan_zero_plan_log(self, VulkanZeroPath::Texture, "none");

  // Get the global Vulkan context
  api::Context* const context = api::context();

  // Required to determine how to insert memory barriers in the command buffer
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(zero),
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
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE));

  return self;
}

Tensor zeros(
    const IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  const ScalarType target_dtype =
      dtype.value_or(c10::get_default_dtype_as_scalartype());
  const Device resolved_device =
      device.value_or(Device(at::kVulkan, api::current_device()));
  Tensor out = at::empty(
      size,
      at::TensorOptions().device(resolved_device).dtype(target_dtype));
  zero_(out);
  return out;
}

} // namespace

std::vector<int64_t> zero_counters_snapshot() {
  const auto& counters = vulkan_zero_counters();
  return {
      static_cast<int64_t>(counters.total.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.buffer_float_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.buffer_byte_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.buffer_byte_shader_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.buffer_cmd_fill_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.texture_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_dim_gt_4.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_storage.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_dtype.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_int8_feature.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.cpu_fallback.load(std::memory_order_relaxed)),
  };
}

void reset_zero_counters() {
  auto& counters = vulkan_zero_counters();
  counters.total.store(0u, std::memory_order_relaxed);
  counters.buffer_float_hit.store(0u, std::memory_order_relaxed);
  counters.buffer_byte_hit.store(0u, std::memory_order_relaxed);
  counters.buffer_byte_shader_hit.store(0u, std::memory_order_relaxed);
  counters.buffer_cmd_fill_hit.store(0u, std::memory_order_relaxed);
  counters.texture_hit.store(0u, std::memory_order_relaxed);
  counters.reject_dim_gt_4.store(0u, std::memory_order_relaxed);
  counters.reject_storage.store(0u, std::memory_order_relaxed);
  counters.reject_dtype.store(0u, std::memory_order_relaxed);
  counters.reject_int8_feature.store(0u, std::memory_order_relaxed);
  counters.cpu_fallback.store(0u, std::memory_order_relaxed);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::zero_"), TORCH_FN(zero_));
  m.impl(TORCH_SELECTIVE_NAME("aten::zeros"), TORCH_FN(zeros));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
