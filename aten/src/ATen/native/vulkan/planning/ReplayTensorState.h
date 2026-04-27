#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>

#include <cstdint>
#include <string>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

struct ReplayEpoch final {
  const void* session_identity{nullptr};
  uint64_t run_id{0u};
};

struct ReplayViewStamp final {
  const void* session_identity{nullptr};
  uint32_t slot_id{0u};
  uint64_t generation{0u};
  uint64_t logical_desc_hash{0u};
  uint64_t storage_id{0u};
};

bool replay_logging_enabled();
bool replay_materializes_escaping_outputs();

ReplayEpoch begin_replay_epoch(
    const void* session_identity,
    const char* allocation_label);

ReplayEpoch current_replay_epoch(const void* session_identity);

ReplayViewStamp stamp_replay_export(
    const Tensor& tensor,
    const void* session_identity,
    uint32_t slot_id,
    const char* producer_op);

void validate_replay_view_not_stale(
    const Tensor& tensor,
    const ReplayViewStamp& stamp,
    const char* consumer_op);

void validate_replay_tensor_not_stale(
    const Tensor& tensor,
    const char* consumer_op);

void clear_replay_tensor_stamp(const Tensor& tensor);

void log_replay_event(
    const char* event,
    const void* session_identity,
    uint64_t generation,
    const char* allocation_label,
    const std::string& detail = std::string());

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
