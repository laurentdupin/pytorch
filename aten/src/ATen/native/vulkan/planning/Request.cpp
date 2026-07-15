#include <ATen/native/vulkan/planning/Request.h>

#include <ATen/native/vulkan/planning/LegacyPlanningInference.h>

#include <c10/util/Exception.h>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

std::optional<VulkanPlanningRequest>& mutable_scoped_planning_request() {
  static thread_local std::optional<VulkanPlanningRequest> request;
  return request;
}

struct VulkanPlanningRequestScopeFrame final {
  int64_t token{0};
  std::optional<VulkanPlanningRequest> previous;
};

std::vector<VulkanPlanningRequestScopeFrame>& mutable_planning_scope_stack() {
  static thread_local std::vector<VulkanPlanningRequestScopeFrame> stack;
  return stack;
}

int64_t& mutable_next_planning_scope_token() {
  static thread_local int64_t token = 0;
  return token;
}

bool has_explicit_planning_context(const VulkanPlanningRequest& request) {
  return request.workload_class == VulkanWorkloadClass::LLMDecode ||
      request.workload_class == VulkanWorkloadClass::VisionBackbone ||
      request.workload_class == VulkanWorkloadClass::VisionDecoder ||
      request.model_domain != VulkanModelDomain::Generic ||
      request.execution_phase != VulkanExecutionPhase::None;
}

VulkanExecutionPhase default_vision_execution_phase(
    const VulkanWorkloadClass workload_class) {
  return workload_class == VulkanWorkloadClass::Convolution
      ? VulkanExecutionPhase::Decoder
      : VulkanExecutionPhase::Backbone;
}

void apply_llm_planning_context(
    VulkanPlanningRequest& request,
    const VulkanExecutionPhase preferred_phase) {
  request.model_domain = VulkanModelDomain::LLM;
  if (request.execution_phase == VulkanExecutionPhase::None) {
    request.execution_phase = preferred_phase == VulkanExecutionPhase::None
        ? VulkanExecutionPhase::Decode
        : preferred_phase;
  }
  if (
      request.workload_class == VulkanWorkloadClass::Attention ||
      request.workload_class == VulkanWorkloadClass::AttentionCache ||
      request.workload_class == VulkanWorkloadClass::LinearMatmul ||
      request.workload_class == VulkanWorkloadClass::Norm) {
    request.workload_class = VulkanWorkloadClass::LLMDecode;
  }
}

void apply_vision_planning_context(
    VulkanPlanningRequest& request,
    const VulkanExecutionPhase preferred_phase) {
  request.model_domain = VulkanModelDomain::Vision;
  if (request.execution_phase == VulkanExecutionPhase::None) {
    request.execution_phase = preferred_phase == VulkanExecutionPhase::None
        ? default_vision_execution_phase(request.workload_class)
        : preferred_phase;
  }
  if (
      request.workload_class == VulkanWorkloadClass::Attention ||
      request.workload_class == VulkanWorkloadClass::LinearMatmul ||
      request.workload_class == VulkanWorkloadClass::Norm ||
      request.workload_class == VulkanWorkloadClass::ShapeView ||
      request.workload_class == VulkanWorkloadClass::Elementwise ||
      request.workload_class == VulkanWorkloadClass::Reduction) {
    request.workload_class = VulkanWorkloadClass::VisionBackbone;
  } else if (request.workload_class == VulkanWorkloadClass::Convolution) {
    request.workload_class = VulkanWorkloadClass::VisionDecoder;
  }
}

VulkanPlanningRequest apply_scoped_planning_request(
    const VulkanPlanningRequest& fallback_request,
    const VulkanPlanningRequest& scope_request) {
  VulkanPlanningRequest request = fallback_request;
  request.inferred_from_label = false;
  if (
      !request.fixed_shape_graph_input_sizes.has_value() &&
      scope_request.fixed_shape_graph_input_sizes.has_value()) {
    request.fixed_shape_graph_input_sizes =
        scope_request.fixed_shape_graph_input_sizes;
  }
  request.prefer_packed_layout_propagation =
      request.prefer_packed_layout_propagation ||
      scope_request.prefer_packed_layout_propagation;
  if (has_explicit_planning_context(request)) {
    return request;
  }

  if (scope_request.model_domain == VulkanModelDomain::LLM) {
    apply_llm_planning_context(request, scope_request.execution_phase);
  } else if (scope_request.model_domain == VulkanModelDomain::Vision) {
    apply_vision_planning_context(request, scope_request.execution_phase);
  }
  return request;
}

} // namespace

VulkanPlanningRequestScope::VulkanPlanningRequestScope(
    const VulkanPlanningRequest& request)
    : previous_(mutable_scoped_planning_request()) {
  mutable_scoped_planning_request() = request;
}

VulkanPlanningRequestScope::~VulkanPlanningRequestScope() {
  mutable_scoped_planning_request() = previous_;
}

int64_t begin_vulkan_planning_request_scope(
    const VulkanPlanningRequest& request) {
  TORCH_CHECK(
      is_valid_vulkan_planning_context(
          request.model_domain, request.execution_phase),
      "Vulkan planning request has incompatible model domain and execution "
      "phase");
  TORCH_CHECK(
      !request.fixed_shape_graph_input_sizes.has_value() ||
          (!request.fixed_shape_graph_input_sizes->empty() &&
           std::all_of(
               request.fixed_shape_graph_input_sizes->begin(),
               request.fixed_shape_graph_input_sizes->end(),
               [](const int64_t size) { return size > 0; })),
      "Vulkan planning request fixed graph input sizes must be positive");
  int64_t& next_token = mutable_next_planning_scope_token();
  TORCH_CHECK(
      next_token < std::numeric_limits<int64_t>::max(),
      "Vulkan planning request scope token overflow");
  ++next_token;
  mutable_planning_scope_stack().push_back(
      VulkanPlanningRequestScopeFrame{
          next_token, mutable_scoped_planning_request()});
  mutable_scoped_planning_request() = request;
  return next_token;
}

void end_vulkan_planning_request_scope(const int64_t token) {
  auto& stack = mutable_planning_scope_stack();
  TORCH_CHECK(
      !stack.empty() && stack.back().token == token,
      "Vulkan planning request scopes must end in LIFO order");
  mutable_scoped_planning_request() = std::move(stack.back().previous);
  stack.pop_back();
}

const char* workload_class_name(const VulkanWorkloadClass workload_class) {
  switch (workload_class) {
    case VulkanWorkloadClass::Generic:
      return "Generic";
    case VulkanWorkloadClass::LinearMatmul:
      return "LinearMatmul";
    case VulkanWorkloadClass::Attention:
      return "Attention";
    case VulkanWorkloadClass::AttentionCache:
      return "AttentionCache";
    case VulkanWorkloadClass::Norm:
      return "Norm";
    case VulkanWorkloadClass::Convolution:
      return "Convolution";
    case VulkanWorkloadClass::ShapeView:
      return "ShapeView";
    case VulkanWorkloadClass::Elementwise:
      return "Elementwise";
    case VulkanWorkloadClass::Reduction:
      return "Reduction";
    case VulkanWorkloadClass::VisionBackbone:
      return "VisionBackbone";
    case VulkanWorkloadClass::VisionDecoder:
      return "VisionDecoder";
    case VulkanWorkloadClass::LLMDecode:
      return "LLMDecode";
  }
  return "Generic";
}

const char* model_domain_name(const VulkanModelDomain model_domain) {
  switch (model_domain) {
    case VulkanModelDomain::Generic:
      return "Generic";
    case VulkanModelDomain::Vision:
      return "Vision";
    case VulkanModelDomain::LLM:
      return "LLM";
  }
  return "Generic";
}

const char* execution_phase_name(const VulkanExecutionPhase execution_phase) {
  switch (execution_phase) {
    case VulkanExecutionPhase::None:
      return "None";
    case VulkanExecutionPhase::Prefill:
      return "Prefill";
    case VulkanExecutionPhase::Decode:
      return "Decode";
    case VulkanExecutionPhase::Backbone:
      return "Backbone";
    case VulkanExecutionPhase::Decoder:
      return "Decoder";
  }
  return "None";
}

bool is_valid_vulkan_planning_context(
    const VulkanModelDomain model_domain,
    const VulkanExecutionPhase execution_phase) {
  switch (model_domain) {
    case VulkanModelDomain::Generic:
      return execution_phase == VulkanExecutionPhase::None;
    case VulkanModelDomain::Vision:
      return execution_phase == VulkanExecutionPhase::None ||
          execution_phase == VulkanExecutionPhase::Backbone ||
          execution_phase == VulkanExecutionPhase::Decoder;
    case VulkanModelDomain::LLM:
      return execution_phase == VulkanExecutionPhase::Prefill ||
          execution_phase == VulkanExecutionPhase::Decode;
  }
  return false;
}

const char* tensor_role_name(const VulkanTensorRole tensor_role) {
  switch (tensor_role) {
    case VulkanTensorRole::Input:
      return "Input";
    case VulkanTensorRole::Weight:
      return "Weight";
    case VulkanTensorRole::Bias:
      return "Bias";
    case VulkanTensorRole::Cache:
      return "Cache";
    case VulkanTensorRole::Scratch:
      return "Scratch";
    case VulkanTensorRole::Mask:
      return "Mask";
  }
  return "Input";
}

VulkanPlanningRequest make_vulkan_planning_request(
    const VulkanWorkloadClass workload_class,
    const VulkanTensorRole tensor_role,
    const VulkanModelDomain model_domain,
    const VulkanExecutionPhase execution_phase) {
  VulkanPlanningRequest request;
  request.workload_class = workload_class;
  request.source_workload_class = workload_class;
  request.model_domain = model_domain;
  request.execution_phase = execution_phase;
  request.tensor_role = tensor_role;
  return request;
}

VulkanPlanningRequest make_vulkan_linear_request(
    const VulkanTensorRole tensor_role) {
  return make_vulkan_planning_request(
      VulkanWorkloadClass::LinearMatmul, tensor_role);
}

VulkanPlanningRequest make_vulkan_tensor_linear_request(
    const Tensor& tensor,
    const VulkanTensorRole tensor_role) {
  return make_vulkan_tensor_planning_request(
      tensor, VulkanWorkloadClass::LinearMatmul, tensor_role);
}

VulkanPlanningRequest make_vulkan_tensor_norm_request(
    const Tensor& tensor,
    const VulkanTensorRole tensor_role) {
  return make_vulkan_tensor_planning_request(
      tensor, VulkanWorkloadClass::Norm, tensor_role);
}

VulkanPlanningRequest make_vulkan_llm_runtime_request(
    const VulkanExecutionPhase execution_phase,
    const VulkanTensorRole tensor_role) {
  return make_vulkan_planning_request(
      VulkanWorkloadClass::LLMDecode,
      tensor_role,
      VulkanModelDomain::LLM,
      execution_phase);
}

VulkanPlanningRequest make_vulkan_vision_backbone_request(
    const VulkanTensorRole tensor_role) {
  return make_vulkan_planning_request(
      VulkanWorkloadClass::VisionBackbone,
      tensor_role,
      VulkanModelDomain::Vision,
      VulkanExecutionPhase::Backbone);
}

VulkanPlanningRequest make_vulkan_vision_decoder_request(
    const VulkanTensorRole tensor_role) {
  return make_vulkan_planning_request(
      VulkanWorkloadClass::VisionDecoder,
      tensor_role,
      VulkanModelDomain::Vision,
      VulkanExecutionPhase::Decoder);
}

VulkanPlanningRequest make_vulkan_tensor_planning_request(
    const Tensor& tensor,
    const VulkanWorkloadClass workload_class,
    const VulkanTensorRole tensor_role,
    const VulkanModelDomain model_domain,
    const VulkanExecutionPhase execution_phase) {
  return specialize_vulkan_planning_request_for_tensor(
      tensor,
      infer_vulkan_planning_request(make_vulkan_planning_request(
          workload_class, tensor_role, model_domain, execution_phase)));
}

VulkanPlanningRequest infer_vulkan_planning_request(
    const VulkanPlanningRequest& fallback_request) {
  VulkanPlanningRequest request = fallback_request;
  if (mutable_scoped_planning_request().has_value()) {
    request = apply_scoped_planning_request(
        request, *mutable_scoped_planning_request());
    return request;
  }

  switch (legacy::infer_model_domain_from_planning_label()) {
    case VulkanModelDomain::LLM:
      apply_llm_planning_context(request, VulkanExecutionPhase::Decode);
      request.inferred_from_label = true;
      break;
    case VulkanModelDomain::Vision:
      apply_vision_planning_context(request, VulkanExecutionPhase::Backbone);
      request.inferred_from_label = true;
      break;
    case VulkanModelDomain::Generic:
      break;
  }
  return request;
}

VulkanPlanningRequest infer_vulkan_planning_request(
    const VulkanWorkloadClass workload_class) {
  return infer_vulkan_planning_request(
      make_vulkan_planning_request(workload_class));
}

VulkanPlanningRequest specialize_vulkan_planning_request_for_tensor(
    const Tensor& tensor,
    const VulkanPlanningRequest& fallback_request) {
  VulkanPlanningRequest request = fallback_request;

  if (mutable_scoped_planning_request().has_value()) {
    return apply_scoped_planning_request(
        request, *mutable_scoped_planning_request());
  }

  if (
      request.workload_class == VulkanWorkloadClass::LLMDecode ||
      request.model_domain != VulkanModelDomain::Generic ||
      request.execution_phase != VulkanExecutionPhase::None) {
    return request;
  }

  if (!legacy::planning_label_allows_llm_tensor_inference()) {
    return request;
  }

  const auto inferred_phase = legacy::infer_llm_phase_from_tensor_shape(tensor);
  if (!inferred_phase.has_value()) {
    return request;
  }

  request.model_domain = VulkanModelDomain::LLM;
  request.execution_phase = *inferred_phase;
  switch (request.workload_class) {
    case VulkanWorkloadClass::LinearMatmul:
    case VulkanWorkloadClass::Attention:
    case VulkanWorkloadClass::AttentionCache:
    case VulkanWorkloadClass::Norm:
      if (*inferred_phase == VulkanExecutionPhase::Decode) {
        request.workload_class = VulkanWorkloadClass::LLMDecode;
      }
      break;
    default:
      break;
  }
  return request;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
