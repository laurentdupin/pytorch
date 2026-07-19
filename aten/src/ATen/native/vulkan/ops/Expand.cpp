#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/LayoutTransitions.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/Functions.h>
#include <ATen/ExpandUtils.h>
#include <c10/core/DispatchKeySet.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/expand_as_ops.h>
#include <ATen/ops/repeat.h>
#endif

#include <ATen/native/vulkan/ops/Utils.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor expand(
    const at::Tensor& self,
    const IntArrayRef output_size,
    bool implicit = false) {
  TORCH_CHECK(self.dim() > 0, "Vulkan expand only supports tensors with at least 1 dimension");
  TORCH_CHECK(
      static_cast<size_t>(self.dim()) <= output_size.size(),
      "Vulkan expand: the number of sizes provided (",
      output_size.size(),
      ") must be greater or equal to the number of dimensions in the tensor (",
      self.dim(),
      ").");

  if (self.is_vulkan() && self.dim() <= 5 && output_size.size() <= 5) {
    const vTensor& v_self = convert(self);
    if (utils::supports_buffer_metadata_view_fast_path(v_self)) {
      const auto logical_geometry = inferExpandGeometry_dimvector(
          v_self.sizes(), v_self.logical_strides(), output_size);
      const auto physical_geometry = inferExpandGeometry_dimvector(
          v_self.sizes(), v_self.gpu_strides(), output_size);

      if (utils::can_make_buffer_metadata_view(
              v_self,
              logical_geometry.sizes,
              logical_geometry.strides,
              physical_geometry.strides,
              v_self.storage_offset())) {
        return make_buffer_metadata_view_checked(
            self,
            logical_geometry.sizes,
            logical_geometry.strides,
            physical_geometry.strides,
            v_self.storage_offset(),
            "aten::expand");
      }
    }
  }

  report_vulkan_cpu_fallback("aten::expand", "cpu_fallback", {self});
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);
  Tensor cpu = self.cpu();
  Tensor cpu_expanded = cpu.expand(output_size.vec(), implicit);
  Tensor out = at::empty(
      cpu_expanded.sizes(),
      self.options().device(self.device()));
  ops::copy_(out, cpu_expanded);
  return out;
}

Tensor expand_as(const at::Tensor& self, const at::Tensor& other) {
  return expand(self, other.sizes());
}

Tensor expand_as_autograd_other(
    c10::DispatchKeySet ks,
    const at::Tensor& self,
    const at::Tensor& other) {
  return at::_ops::expand_as::redispatch(
      ks & c10::after_autograd_keyset,
      self,
      other);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::expand"), TORCH_FN(expand));
  m.impl(TORCH_SELECTIVE_NAME("aten::expand_as"), TORCH_FN(expand_as));
}

TORCH_LIBRARY_IMPL(aten, AutogradOther, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::expand_as"),
      TORCH_FN(expand_as_autograd_other));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
