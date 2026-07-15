#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>

#include <optional>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

enum class VulkanWorkloadClass : uint8_t {
  Generic = 0u,
  LinearMatmul,
  Attention,
  AttentionCache,
  Norm,
  Convolution,
  ShapeView,
  Elementwise,
  Reduction,
  VisionBackbone,
  VisionDecoder,
  LLMDecode,
};

enum class VulkanModelDomain : uint8_t {
  Generic = 0u,
  Vision,
  LLM,
};

enum class VulkanExecutionPhase : uint8_t {
  None = 0u,
  Prefill,
  Decode,
  Backbone,
  Decoder,
};

enum class VulkanTensorRole : uint8_t {
  Input = 0u,
  Weight,
  Bias,
  Cache,
  Scratch,
  Mask,
};

struct VulkanAttentionShapeDesc final {
  ScalarType dtype{kFloat};
  int64_t batch_heads{0};
  int64_t target_length{0};
  int64_t source_length{0};
  int64_t head_dim{0};
  int64_t value_dim{0};
  bool has_explicit_mask{false};
  bool has_dropout{false};
  bool is_causal{false};
  bool enable_gqa{false};
};

struct VulkanPlanningRequest final {
  VulkanWorkloadClass workload_class{VulkanWorkloadClass::Generic};
  VulkanWorkloadClass source_workload_class{VulkanWorkloadClass::Generic};
  VulkanModelDomain model_domain{VulkanModelDomain::Generic};
  VulkanExecutionPhase execution_phase{VulkanExecutionPhase::None};
  VulkanTensorRole tensor_role{VulkanTensorRole::Input};
  bool inferred_from_label{false};
  bool prefer_packed_layout_propagation{false};
  std::optional<std::vector<int64_t>> fixed_shape_graph_input_sizes;
  std::optional<VulkanAttentionShapeDesc> attention_shape;
};

class VulkanPlanningRequestScope final {
 public:
  explicit VulkanPlanningRequestScope(const VulkanPlanningRequest& request);

  VulkanPlanningRequestScope(const VulkanPlanningRequestScope&) = delete;
  VulkanPlanningRequestScope& operator=(const VulkanPlanningRequestScope&) =
      delete;

  VulkanPlanningRequestScope(VulkanPlanningRequestScope&&) = delete;
  VulkanPlanningRequestScope& operator=(VulkanPlanningRequestScope&&) = delete;

  ~VulkanPlanningRequestScope();

 private:
  std::optional<VulkanPlanningRequest> previous_;
};

int64_t begin_vulkan_planning_request_scope(
    const VulkanPlanningRequest& request);

void end_vulkan_planning_request_scope(int64_t token);

const char* workload_class_name(VulkanWorkloadClass);

const char* model_domain_name(VulkanModelDomain);

const char* execution_phase_name(VulkanExecutionPhase);

const char* tensor_role_name(VulkanTensorRole);

bool is_valid_vulkan_planning_context(
    VulkanModelDomain model_domain,
    VulkanExecutionPhase execution_phase);

VulkanPlanningRequest make_vulkan_planning_request(
    VulkanWorkloadClass workload_class,
    VulkanTensorRole tensor_role = VulkanTensorRole::Input,
    VulkanModelDomain model_domain = VulkanModelDomain::Generic,
    VulkanExecutionPhase execution_phase = VulkanExecutionPhase::None);

VulkanPlanningRequest make_vulkan_linear_request(
    VulkanTensorRole tensor_role = VulkanTensorRole::Input);

VulkanPlanningRequest make_vulkan_tensor_linear_request(
    const Tensor& tensor,
    VulkanTensorRole tensor_role = VulkanTensorRole::Input);

VulkanPlanningRequest make_vulkan_tensor_norm_request(
    const Tensor& tensor,
    VulkanTensorRole tensor_role = VulkanTensorRole::Input);

VulkanPlanningRequest make_vulkan_llm_runtime_request(
    VulkanExecutionPhase execution_phase,
    VulkanTensorRole tensor_role = VulkanTensorRole::Input);

VulkanPlanningRequest make_vulkan_vision_backbone_request(
    VulkanTensorRole tensor_role = VulkanTensorRole::Input);

VulkanPlanningRequest make_vulkan_vision_decoder_request(
    VulkanTensorRole tensor_role = VulkanTensorRole::Input);

VulkanPlanningRequest make_vulkan_tensor_planning_request(
    const Tensor& tensor,
    VulkanWorkloadClass workload_class,
    VulkanTensorRole tensor_role = VulkanTensorRole::Input,
    VulkanModelDomain model_domain = VulkanModelDomain::Generic,
    VulkanExecutionPhase execution_phase = VulkanExecutionPhase::None);

VulkanPlanningRequest infer_vulkan_planning_request(
    const VulkanPlanningRequest&);

VulkanPlanningRequest infer_vulkan_planning_request(VulkanWorkloadClass);

VulkanPlanningRequest specialize_vulkan_planning_request_for_tensor(
    const Tensor&,
    const VulkanPlanningRequest&);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
