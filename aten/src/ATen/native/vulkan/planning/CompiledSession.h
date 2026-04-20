#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/planning/Capabilities.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

using VulkanValueId = uint32_t;

enum class VulkanCompiledSessionKind : uint8_t {
  DepthAnythingV2 = 0u,
  DepthAnythingV2Image,
  DepthAnythingV2BackboneStack,
  DepthAnythingV2DecoderPreprocessHead,
};

enum class VulkanIROpKind : uint8_t {
  InputImage = 0u,
  PatchEmbed,
  FeatureMapToTokens,
  ElementwiseAdd,
  Concat,
  PatchTokenInput,
  BackboneBlock,
  CaptureNormedPatchTokens,
  TokensToFeatureMap,
  DecoderProject,
  DecoderResize,
  DecoderPreprocess,
  DecoderHead,
  OutputAlias,
};

enum class VulkanIRTensorRole : uint8_t {
  Input = 0u,
  Constant,
  Intermediate,
  Output,
  Scratch,
};

struct VulkanIRTensorSpec final {
  ScalarType dtype{kFloat};
  std::vector<int64_t> logical_sizes;
  std::vector<int64_t> padded_sizes;
  api::ExecutionLayout execution_layout{api::ExecutionLayout::BUFFER_DIRECT};
  api::GPUMemoryLayout memory_layout{api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  api::StorageType storage_type{api::StorageType::BUFFER};
  VulkanIRTensorRole role{VulkanIRTensorRole::Intermediate};
  bool persistent{false};
  bool external{false};
};

struct VulkanIRValue final {
  VulkanValueId id{0u};
  std::string name;
  VulkanIRTensorSpec spec;
};

struct VulkanIROpNode final {
  VulkanIROpKind kind{VulkanIROpKind::BackboneBlock};
  std::string name;
  std::vector<VulkanValueId> inputs;
  std::vector<VulkanValueId> outputs;
  std::vector<VulkanValueId> constants;
  std::string attributes_key;
};

struct VulkanIRLifetime final {
  VulkanValueId id{0u};
  size_t first_op{0u};
  size_t last_op{0u};
  bool may_alias{false};
};

struct VulkanIROutputAlias final {
  VulkanValueId output{0u};
  VulkanValueId source{0u};
};

class VulkanBackendIR final {
 public:
  VulkanValueId add_value(std::string name, VulkanIRTensorSpec spec);
  void add_op(VulkanIROpNode op);
  void add_output_alias(VulkanValueId output, VulkanValueId source);
  void recompute_lifetimes();

  const std::vector<VulkanIRValue>& values() const;
  std::vector<VulkanIRValue>& mutable_values();
  const std::vector<VulkanIROpNode>& ops() const;
  const std::vector<VulkanIRLifetime>& lifetimes() const;
  const std::vector<VulkanIROutputAlias>& output_aliases() const;

 private:
  std::vector<VulkanIRValue> values_;
  std::vector<VulkanIROpNode> ops_;
  std::vector<VulkanIRLifetime> lifetimes_;
  std::vector<VulkanIROutputAlias> output_aliases_;
};

struct VulkanCompiledSessionKey final {
  VulkanCompiledSessionKind kind{VulkanCompiledSessionKind::DepthAnythingV2};
  std::string model_key;
  std::string configuration_key;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  ScalarType dtype{kFloat};
  std::string capability_key;
  bool persistent{true};
};

bool operator==(
    const VulkanCompiledSessionKey& lhs,
    const VulkanCompiledSessionKey& rhs);

struct VulkanCompiledSessionKeyHash final {
  size_t operator()(const VulkanCompiledSessionKey& key) const;
};

struct VulkanGlobalLayoutPlan final {
  api::ExecutionLayout execution_layout{api::ExecutionLayout::BUFFER_DIRECT};
  api::GPUMemoryLayout memory_layout{api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  api::StorageType storage_type{api::StorageType::BUFFER};
  int64_t width_alignment{1};
  bool pad_width{false};
  bool apply_to_constants{true};
  std::string reason;
};

struct VulkanIRAllocationSlot final {
  size_t slot_id{0u};
  size_t bytes{0u};
  size_t first_op{0u};
  size_t last_op{0u};
  bool dedicated{false};
  std::vector<VulkanValueId> values;
};

struct VulkanIRMemoryPlan final {
  std::vector<VulkanIRAllocationSlot> slots;
  size_t reusable_bytes{0u};
  size_t dedicated_bytes{0u};
  size_t external_bytes{0u};
  size_t planned_bytes{0u};
};

struct VulkanCompiledSessionTensorBindings final {
  std::vector<std::optional<size_t>> value_tensor_slots;
  std::vector<VulkanValueId> slot_values;
  std::vector<VulkanValueId> input_values;

  size_t tensor_slot_count() const {
    return slot_values.size();
  }
};

class VulkanCompiledSession final {
 public:
  struct State;

 private:
  std::shared_ptr<State> state_;

 public:
  VulkanCompiledSession() = default;
  explicit VulkanCompiledSession(std::shared_ptr<State> state)
      : state_(std::move(state)) {}

  bool defined() const;
  const VulkanCompiledSessionKey& key() const;
  const VulkanBackendIR& ir() const;
  const VulkanGlobalLayoutPlan& layout_plan() const;
  const VulkanIRMemoryPlan& memory_plan() const;
  bool executable() const;
  const void* identity() const;
};

std::optional<VulkanCompiledSessionTensorBindings>
make_compiled_session_tensor_bindings(const VulkanCompiledSession& session);

struct DepthAnythingV2BackboneStackSessionDesc final {
  std::string model_key;
  std::vector<int64_t> patch_token_sizes;
  ScalarType dtype{kFloat};
  int64_t backbone_block_count{0};
  std::vector<int64_t> capture_indices;
  std::vector<int64_t> block_hidden_dims;
  std::vector<int64_t> block_num_heads;
  std::optional<std::vector<int64_t>> normalized_shape;
  bool persistent{true};
};

struct DepthAnythingV2SessionDesc final {
  std::string model_key;
  std::vector<int64_t> patch_token_sizes;
  ScalarType dtype{kFloat};
  int64_t backbone_block_count{0};
  std::vector<int64_t> capture_indices;
  std::vector<int64_t> block_hidden_dims;
  std::vector<int64_t> block_num_heads;
  std::vector<int64_t> normalized_shape;
  std::array<std::vector<int64_t>, 4u> layer_feature_sizes;
  std::array<std::vector<int64_t>, 4u> project_layer_sizes;
  std::array<std::vector<int64_t>, 4u> resize_layer_sizes;
  std::array<bool, 4u> apply_resize{{true, true, false, true}};
  std::array<std::vector<int64_t>, 4u> decoder_layer_sizes;
  std::vector<int64_t> output_sizes;
  int64_t patch_h{0};
  int64_t patch_w{0};
  bool persistent{true};
};

struct DepthAnythingV2ImageSessionDesc final {
  std::string model_key;
  std::vector<int64_t> image_sizes;
  std::vector<int64_t> patch_token_sizes;
  std::vector<int64_t> prefix_token_sizes;
  std::vector<int64_t> patch_pos_encoding_sizes;
  ScalarType dtype{kFloat};
  int64_t backbone_block_count{0};
  std::vector<int64_t> capture_indices;
  std::vector<int64_t> block_hidden_dims;
  std::vector<int64_t> block_num_heads;
  std::vector<int64_t> normalized_shape;
  std::array<std::vector<int64_t>, 4u> layer_feature_sizes;
  std::array<std::vector<int64_t>, 4u> project_layer_sizes;
  std::array<std::vector<int64_t>, 4u> resize_layer_sizes;
  std::array<bool, 4u> apply_resize{{true, true, false, true}};
  std::array<std::vector<int64_t>, 4u> decoder_layer_sizes;
  std::vector<int64_t> output_sizes;
  int64_t patch_h{0};
  int64_t patch_w{0};
  bool persistent{true};
};

struct DepthAnythingV2DecoderPreprocessHeadSessionDesc final {
  std::string model_key;
  std::array<std::vector<int64_t>, 4u> layer_token_sizes;
  std::array<std::vector<int64_t>, 4u> layer_feature_sizes;
  std::array<std::vector<int64_t>, 4u> project_layer_sizes;
  std::array<std::vector<int64_t>, 4u> resize_layer_sizes;
  std::array<bool, 4u> apply_resize{{true, true, false, true}};
  std::array<std::vector<int64_t>, 4u> decoder_layer_sizes;
  std::vector<int64_t> output_sizes;
  ScalarType dtype{kFloat};
  int64_t patch_h{0};
  int64_t patch_w{0};
  bool persistent{true};
};

const char* compiled_session_kind_name(VulkanCompiledSessionKind kind);
const char* ir_op_kind_name(VulkanIROpKind kind);

std::string make_vulkan_compiled_session_capability_key(
    const VulkanRuntimeCapabilityProfile& profile);

VulkanGlobalLayoutPlan make_buffer_first_width_packed_layout_plan(
    const VulkanCompiledSessionKey& key,
    const VulkanRuntimeCapabilityProfile& profile);

void apply_global_layout_plan(
    VulkanBackendIR& ir,
    const VulkanGlobalLayoutPlan& plan);

VulkanCompiledSession lookup_or_create_vulkan_compiled_session(
    const VulkanCompiledSessionKey& key,
    const std::function<VulkanCompiledSession()>& builder);

VulkanCompiledSession lookup_or_create_depth_anything_v2_session(
    const DepthAnythingV2SessionDesc& desc);

VulkanCompiledSession lookup_or_create_depth_anything_v2_image_session(
    const DepthAnythingV2ImageSessionDesc& desc);

VulkanCompiledSession lookup_or_create_depth_anything_v2_backbone_stack_session(
    const DepthAnythingV2BackboneStackSessionDesc& desc);

VulkanCompiledSession
lookup_or_create_depth_anything_v2_decoder_preprocess_head_session(
    const DepthAnythingV2DecoderPreprocessHeadSessionDesc& desc);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
