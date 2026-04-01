#include <ATen/native/vulkan/ops/Common.h>

#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Scalar _local_scalar_dense(const Tensor& self) {
  switch (self.scalar_type()) {
    case c10::ScalarType::Float:
      return Scalar(self.cpu().item<float>());
    case c10::ScalarType::Half:
      return Scalar(self.cpu().item<c10::Half>());
    case c10::ScalarType::BFloat16:
      return Scalar(self.cpu().item<c10::BFloat16>());
    case c10::ScalarType::Bool:
      return Scalar(self.cpu().item<bool>());
    case c10::ScalarType::Byte:
      return Scalar(self.cpu().item<uint8_t>());
    case c10::ScalarType::Char:
      return Scalar(self.cpu().item<int8_t>());
    case c10::ScalarType::Int:
      return Scalar(self.cpu().item<int32_t>());
    case c10::ScalarType::Long:
      return Scalar(self.cpu().item<int64_t>());
    default:
      TORCH_CHECK(false, "Unsupported Vulkan dtype for _local_scalar_dense");
  }
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_local_scalar_dense"),
      TORCH_FN(_local_scalar_dense));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
