#pragma once

#ifdef USE_VULKAN_API

#include <ATen/core/Tensor.h>
#include <ATen/native/vulkan/planning/ExecutionObjects.h>
#include <ATen/native/vulkan/planning/Runtime.h>

#include <cstddef>
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
  std::optional<ScratchArena>& scratch_arena();
  const std::optional<ScratchArena>& scratch_arena() const;
  size_t resident_nbytes() const;
  const void* identity() const;
};

class VisionBackboneProgram final {
 public:
  struct State;

 private:
  std::shared_ptr<State> state_;

 public:
  VisionBackboneProgram() = default;
  explicit VisionBackboneProgram(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  int64_t num_heads() const;
  std::optional<ScratchArena>& scratch_arena();
  const std::optional<ScratchArena>& scratch_arena() const;
  Tensor& norm1_output();
  Tensor& qkv_output();
  Tensor& merge_output();
  Tensor& proj_output();
  Tensor& norm2_output();
  Tensor& fc1_output();
  Tensor& fc2_output();
  bool persistent() const;
  size_t resident_nbytes() const;
  const void* identity() const;
};

class VisionDecoderProgram final {
 public:
  struct State;

 private:
  std::shared_ptr<State> state_;

 public:
  VisionDecoderProgram() = default;
  explicit VisionDecoderProgram(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  std::optional<ScratchArena>& scratch_arena();
  const std::optional<ScratchArena>& scratch_arena() const;
  Tensor& skip_relu_output();
  Tensor& skip_conv1_output();
  Tensor& skip_conv2_output();
  Tensor& skip_res_output();
  Tensor& main_input_output();
  Tensor& main_relu_output();
  Tensor& main_conv1_output();
  Tensor& main_conv2_output();
  Tensor& main_res_output();
  Tensor& upsample_output();
  Tensor& out_conv_output();
  size_t resident_nbytes() const;
  const void* identity() const;
};

AttentionRuntimeProgram lookup_or_create_labeled_attention_runtime_program(
    const std::string& allocation_label,
    VulkanAttentionKernelFamily kernel_family,
    const std::optional<VulkanScratchArenaSpec>& scratch_spec,
    const VulkanExecutionProgramPlanningDesc& program_plan);

VisionBackboneProgram lookup_or_create_labeled_vision_backbone_program(
    const std::string& allocation_label,
    ScalarType dtype,
    int64_t batch_size,
    int64_t token_count,
    int64_t embed_dim,
    int64_t hidden_dim,
    int64_t num_heads,
    const std::optional<VulkanScratchArenaSpec>& scratch_spec,
    const VulkanExecutionProgramPlanningDesc& program_plan);

VisionDecoderProgram lookup_or_create_labeled_vision_decoder_program(
    const std::string& allocation_label,
    IntArrayRef input_sizes,
    const std::optional<std::vector<int64_t>>& skip_sizes,
    IntArrayRef target_sizes,
    int64_t out_channels,
    const std::optional<VulkanScratchArenaSpec>& scratch_spec,
    const VulkanExecutionProgramPlanningDesc& program_plan,
    bool allocate_intermediate_outputs = true);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
