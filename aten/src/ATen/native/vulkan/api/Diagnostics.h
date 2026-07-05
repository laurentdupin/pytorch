#pragma once

#ifdef USE_VULKAN_API

#include <cstdint>
#include <string>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace api {

enum class VulkanFailureClass {
  TensorStateInvalid,
  MetadataViewInvalid,
  RawCopyIllegal,
  ReplayViewStale,
  RouteHardFail,
  KernelIncorrect,
  DeviceLost,
  Unsupported,
  ReplayHangRisk,
  Unknown,
};

const char* vulkan_failure_class_name(VulkanFailureClass failure_class);

bool vulkan_failure_logging_enabled();

void mark_vulkan_post_failure_recovery_required();
bool vulkan_post_failure_recovery_required();
void clear_vulkan_post_failure_recovery_required();

void note_vulkan_lazy_chain_op(const char* op_name);
void flush_vulkan_lazy_chain_boundary(
    const char* boundary_kind,
    const char* reason);
bool vulkan_deferred_region_plan_logging_enabled();
void note_vulkan_deferred_region_tensor_write(
    const char* op_name,
    const char* route_name,
    uint64_t output_storage_id,
    uint64_t output_view_id,
    uint64_t output_generation,
    uint64_t output_logical_desc_hash,
    int64_t output_storage_offset,
    int64_t output_buffer_length,
    bool output_is_view,
    const std::string& output_state,
    const std::vector<std::string>& input_states,
    uint64_t vulkan_input_count,
    uint64_t missing_input_lease_count);
void note_vulkan_deferred_region_value_access_boundary(
    const char* boundary_kind,
    const char* reason,
    const char* access_kind,
    const std::string& source_state,
    const std::string& destination_state,
    uint64_t vulkan_source_count,
    uint64_t cpu_destination_count);

void log_vulkan_failure(
    VulkanFailureClass failure_class,
    const char* op_name,
    const char* reason,
    const std::string& detail = std::string());

std::string format_vulkan_failure(
    VulkanFailureClass failure_class,
    const char* op_name,
    const char* reason,
    const std::string& detail = std::string());

std::string report_vulkan_failure(
    VulkanFailureClass failure_class,
    const char* op_name,
    const char* reason,
    const std::string& detail = std::string());

[[noreturn]] void fail_vulkan(
    VulkanFailureClass failure_class,
    const char* op_name,
    const char* reason,
    const std::string& detail = std::string());

void check_vulkan(
    bool condition,
    VulkanFailureClass failure_class,
    const char* op_name,
    const char* reason,
    const std::string& detail = std::string());

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
