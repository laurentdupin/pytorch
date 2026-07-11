#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/ops/Mm.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <torch/custom_class.h>
#include <tuple>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

enum class VulkanGraphRegionFamily : uint8_t {
  LinearGeluTanh,
  Conv2dReluConv2d,
};

enum class VulkanGraphRegionOpcode : uint8_t {
  LinearContext,
  GeluTanh,
  Conv2dReluContext,
  Conv2dContext,
};

enum class VulkanGraphRegionValueKind : uint8_t {
  Input,
  Temporary,
  Output,
};

enum class VulkanGraphRegionStaticContextKind : uint8_t {
  Linear,
  Conv2d,
};

enum class VulkanGraphRegionTransition : uint8_t {
  Direct,
  RequireDirectBuffer,
};

struct VulkanGraphRegionValueSchema final {
  uint32_t id{0u};
  VulkanGraphRegionValueKind kind{VulkanGraphRegionValueKind::Input};
  uint32_t use_count{0u};
  uint32_t last_use_instruction{0u};
  bool escapes_region{false};
};

struct VulkanGraphRegionInstructionSchema final {
  VulkanGraphRegionOpcode opcode{VulkanGraphRegionOpcode::LinearContext};
  uint32_t input_value{0u};
  uint32_t output_value{0u};
  int32_t static_context_slot{-1};
  VulkanGraphRegionTransition transition{VulkanGraphRegionTransition::Direct};
};

struct VulkanGraphRegionReplayState final {
  bool persistent_command_buffer{false};
  bool persistent_descriptor_pool{false};
  bool reusable_outputs{false};
  uint32_t recorded_dispatches{0u};
};

struct VulkanGraphRegionPlanSchema final {
  const char* program_name{"VulkanGraphRegionPlan"};
  const char* version{"v1"};
  VulkanGraphRegionFamily family{VulkanGraphRegionFamily::LinearGeluTanh};
  uint32_t input_count{1u};
  uint32_t output_count{1u};
  uint32_t instruction_count{0u};
  bool bounded_submission_owned{false};
  bool program_private_scratch{false};
  uint32_t scratch_ring_capacity{0u};
  bool timeline_gated_release{false};
  bool direct_transition_only{true};
  VulkanGraphRegionReplayState replay_state{};
};

struct VulkanGraphRegionStaticContext final {
  VulkanGraphRegionStaticContextKind kind{
      VulkanGraphRegionStaticContextKind::Linear};
  c10::intrusive_ptr<LinearPackedContext> linear_context;
  c10::intrusive_ptr<Conv2dPackedContext> conv2d_context;
};

struct VulkanGraphRegionScratchDescriptor final {
  std::vector<int64_t> sizes;
  ScalarType dtype{ScalarType::Undefined};
  api::StorageType storage_type{api::StorageType::UNKNOWN};
  api::GPUMemoryLayout memory_layout{
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED};
  api::ExecutionLayout execution_layout{api::ExecutionLayout::TEXTURE};
  bool direct_buffer{false};

  bool matches(const VulkanGraphRegionScratchDescriptor&) const;
};

struct VulkanGraphRegionScratchSlot final {
  Tensor tensor;
  VulkanGraphRegionScratchDescriptor descriptor;
  api::VulkanSubmission submission;
};

class VulkanGraphRegionPlan final : public torch::jit::CustomClassHolder {
 private:
  VulkanGraphRegionPlanSchema schema_{};
  std::vector<VulkanGraphRegionValueSchema> values_;
  std::vector<VulkanGraphRegionInstructionSchema> instructions_;
  std::vector<VulkanGraphRegionStaticContext> static_contexts_;
  std::array<VulkanGraphRegionScratchSlot, 2u> scratch_slots_{};
  std::atomic_flag invocation_active_ = ATOMIC_FLAG_INIT;

 public:
  VulkanGraphRegionPlan(
      VulkanGraphRegionPlanSchema schema,
      std::vector<VulkanGraphRegionValueSchema> values,
      std::vector<VulkanGraphRegionInstructionSchema> instructions,
      std::vector<VulkanGraphRegionStaticContext> static_contexts);
  ~VulkanGraphRegionPlan() noexcept;

  const VulkanGraphRegionPlanSchema& schema() const;
  const std::vector<VulkanGraphRegionValueSchema>& values() const;
  const std::vector<VulkanGraphRegionInstructionSchema>& instructions() const;
  const c10::intrusive_ptr<LinearPackedContext>& linear_context(size_t) const;
  const c10::intrusive_ptr<Conv2dPackedContext>& conv2d_context(size_t) const;
  bool valid() const;
  int64_t find_reusable_scratch_slot(
      const VulkanGraphRegionScratchDescriptor&,
      api::Context&);
  int64_t find_capture_scratch_slot(api::Context&);
  Tensor& scratch_tensor(size_t);
  void adopt_scratch_tensor(
      size_t,
      Tensor,
      VulkanGraphRegionScratchDescriptor,
      api::VulkanSubmission);
  void mark_scratch_submission(size_t, api::VulkanSubmission);
  bool try_begin_invocation();
  void end_invocation();
};

c10::intrusive_ptr<VulkanGraphRegionPlan>
create_vulkan_graph_region_plan_linear_gelu(
    const c10::intrusive_ptr<LinearPackedContext>& linear_context);

c10::intrusive_ptr<VulkanGraphRegionPlan>
create_vulkan_graph_region_plan_conv2d_relu_conv2d(
    const c10::intrusive_ptr<Conv2dPackedContext>& first_conv_context,
    const c10::intrusive_ptr<Conv2dPackedContext>& second_conv_context);

std::vector<Tensor> run_vulkan_graph_region_plan(
    const std::vector<Tensor>& inputs,
    const c10::intrusive_ptr<VulkanGraphRegionPlan>& plan);

struct StaticAddLayernormPlanSchema final {
  const char* program_name{"StaticAddLayernormRegion"};
  const char* version{"v1"};
  const char* fused_instruction{"add_layernorm_fused_or_composed_vulkan"};
  uint32_t instruction_count{1u};
  uint32_t residual_input_ssa{0u};
  uint32_t addend_input_ssa{1u};
  uint32_t residual_output_ssa{2u};
  uint32_t normalized_output_ssa{3u};
  uint32_t residual_input_use_count{1u};
  uint32_t residual_input_last_use{0u};
  uint32_t addend_input_use_count{1u};
  uint32_t addend_input_last_use{0u};
  uint32_t static_context_slot{0u};
  bool direct_transition_only{true};
  bool replay_state_empty{true};
  bool persistent_output_state{false};
};

class GraphAddLayernormPlan final : public torch::jit::CustomClassHolder {
 private:
  StaticAddLayernormPlanSchema schema_{};
  c10::intrusive_ptr<LayernormPackedContext> layernorm_context_;
  std::vector<int64_t> normalized_shape_;
  std::atomic_flag invocation_active_ = ATOMIC_FLAG_INIT;

 public:
  GraphAddLayernormPlan(
      c10::intrusive_ptr<LayernormPackedContext> layernorm_context,
      std::vector<int64_t> normalized_shape);

  const StaticAddLayernormPlanSchema& schema() const;
  const c10::intrusive_ptr<LayernormPackedContext>& layernorm_context() const;
  const std::vector<int64_t>& normalized_shape() const;
  bool try_begin_invocation();
  void end_invocation();
};

c10::intrusive_ptr<GraphAddLayernormPlan> create_graph_add_layernorm_plan(
    const c10::intrusive_ptr<LayernormPackedContext>& layernorm_context,
    IntArrayRef normalized_shape);

std::tuple<Tensor, Tensor> run_graph_add_layernorm_plan(
    const Tensor& residual,
    const Tensor& addend,
    const c10::intrusive_ptr<GraphAddLayernormPlan>& plan);

struct StaticConv2dReluPlanSchema final {
  const char* program_name{"StaticConv2dReluRegion"};
  const char* version{"v1"};
  uint32_t instruction_count{1u};
  uint32_t input_ssa{0u};
  uint32_t output_ssa{1u};
  uint32_t input_use_count{1u};
  uint32_t input_last_use{0u};
  uint32_t static_context_slot{0u};
  bool direct_transition_only{true};
  bool replay_state_empty{true};
};

class GraphConv2dReluPlan final : public torch::jit::CustomClassHolder {
 private:
  StaticConv2dReluPlanSchema schema_{};
  c10::intrusive_ptr<Conv2dPackedContext> conv_context_;
  std::atomic_flag invocation_active_ = ATOMIC_FLAG_INIT;

 public:
  explicit GraphConv2dReluPlan(
      c10::intrusive_ptr<Conv2dPackedContext> conv_context);

  const StaticConv2dReluPlanSchema& schema() const;
  const c10::intrusive_ptr<Conv2dPackedContext>& conv_context() const;
  bool try_begin_invocation();
  void end_invocation();
};

c10::intrusive_ptr<GraphConv2dReluPlan> create_graph_conv2d_relu_plan(
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context);

Tensor run_graph_conv2d_relu_plan(
    const Tensor& input,
    const c10::intrusive_ptr<GraphConv2dReluPlan>& plan);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif // USE_VULKAN_API
