#pragma once

#ifdef USE_VULKAN_API

#include <cstdint>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

std::vector<int64_t> zero_counters_snapshot();

void reset_zero_counters();

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
