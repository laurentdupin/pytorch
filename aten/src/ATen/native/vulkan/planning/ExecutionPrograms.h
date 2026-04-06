#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/planning/ExecutionObjects.h>
#include <ATen/native/vulkan/planning/Runtime.h>

#include <memory>
#include <optional>
#include <string>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

class AttentionRuntimeProgram final {
 public:
  struct State;

 private:
  std::shared_ptr<State> state_;

 public:
  AttentionRuntimeProgram() = default;
  explicit AttentionRuntimeProgram(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  VulkanAttentionKernelFamily kernel_family() const;
  const std::optional<KVCacheObject>& key_cache() const;
  const std::optional<KVCacheObject>& value_cache() const;
  const std::optional<ScratchArena>& scratch_arena() const;
  bool persistent() const;
  void set_sequence_lengths(
      int64_t key_sequence_length,
      int64_t value_sequence_length) const;
  const void* identity() const;
};

class GatedDeltaSplitProgram final {
 public:
  struct State;

 private:
  std::shared_ptr<State> state_;

 public:
  GatedDeltaSplitProgram() = default;
  explicit GatedDeltaSplitProgram(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  const VulkanBoundaryPlan& boundary_plan() const;
  const std::optional<ScratchArena>& scratch_arena() const;
  bool persistent() const;
  const void* identity() const;
};

AttentionRuntimeProgram lookup_or_create_labeled_attention_runtime_program(
    const std::string& allocation_label,
    VulkanAttentionKernelFamily kernel_family,
    const std::optional<VulkanKVCacheSpec>& key_cache_spec,
    const std::optional<VulkanKVCacheSpec>& value_cache_spec,
    const std::optional<VulkanScratchArenaSpec>& scratch_spec,
    int64_t key_sequence_length,
    int64_t value_sequence_length,
    const VulkanExecutionProgramPlanningDesc& program_plan);

std::optional<GatedDeltaSplitProgram>
lookup_or_create_labeled_gated_delta_split_program(
    const std::string& allocation_label,
    const VulkanBoundaryPlan& boundary_plan,
    const std::optional<VulkanScratchArenaSpec>& scratch_spec,
    const VulkanExecutionProgramPlanningDesc& program_plan);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
