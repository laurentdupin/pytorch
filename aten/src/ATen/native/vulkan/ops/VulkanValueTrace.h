#pragma once

#ifdef USE_VULKAN_API

#include <ATen/ArrayRef.h>
#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

bool vulkan_value_trace_enabled();

void record_tensor_value_write(
    const Tensor& output,
    const char* op_name,
    const char* route_name = nullptr,
    ArrayRef<Tensor> inputs = {});

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
