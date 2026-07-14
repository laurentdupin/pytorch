#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/planning/Request.h>

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

constexpr int64_t kLlmlikeHiddenSizeThreshold = 64;
constexpr int64_t kLlmlikeMaxSequenceExtent = 64;
constexpr int64_t kLlmlikeMaxPrefixExtent = 64;

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

bool allocation_label_contains(
    const std::string& allocation_label,
    std::initializer_list<const char*> needles) {
  for (const char* needle : needles) {
    if (allocation_label.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool is_runtime_llm_operator_label(const std::string& allocation_label) {
  return allocation_label_contains(
      allocation_label,
      {
          "llama",
          "decoder",
          "lm_head",
          "self_attn",
          "linear",
          "bmm",
          "layer_norm",
          "rms_norm",
          "sdpa",
          "softmax",
      });
}

const std::string& current_planning_label() {
  const std::string& runtime_label = api::current_runtime_label();
  if (!runtime_label.empty()) {
    return runtime_label;
  }
  return api::current_allocation_label();
}

std::optional<VulkanExecutionPhase> infer_llm_runtime_phase_from_tensor(
    const Tensor& tensor) {
  if (!tensor.defined() || !tensor.is_vulkan() || tensor.dim() < 2) {
    return std::nullopt;
  }

  if (tensor.size(-1) < kLlmlikeHiddenSizeThreshold) {
    return std::nullopt;
  }

  int64_t sequence_extent = 0;
  int64_t prefix_extent = 1;
  if (tensor.dim() == 2) {
    sequence_extent = tensor.size(0);
  } else {
    sequence_extent = tensor.size(tensor.dim() - 2);
    for (const auto dim : c10::irange(std::max<int64_t>(0, tensor.dim() - 2))) {
      prefix_extent *= std::max<int64_t>(1, tensor.size(dim));
      if (prefix_extent > kLlmlikeMaxPrefixExtent) {
        return std::nullopt;
      }
    }
  }

  if (
      sequence_extent < 1 ||
      sequence_extent > kLlmlikeMaxSequenceExtent) {
    return std::nullopt;
  }

  return sequence_extent == 1 ? VulkanExecutionPhase::Decode
                              : VulkanExecutionPhase::Prefill;
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
    if (has_explicit_planning_context(request)) {
      return request;
    }
  }

  const std::string& planning_label = current_planning_label();
  if (planning_label.empty() || planning_label == "unlabeled") {
    return request;
  }

  const bool is_llm = allocation_label_contains(
      planning_label,
      {"llama", "decoder", "lm_head", "self_attn"});
  const bool is_vision = allocation_label_contains(
      planning_label,
      {"depth", "dino", "beit", "zoe", "midas", "patch_embed", "refinenet"});

  if (is_llm) {
    apply_llm_planning_context(request, VulkanExecutionPhase::Decode);
    request.inferred_from_label = true;
    return request;
  }

  if (is_vision) {
    apply_vision_planning_context(request, VulkanExecutionPhase::Backbone);
    request.inferred_from_label = true;
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

  if (
      request.workload_class == VulkanWorkloadClass::LLMDecode ||
      request.model_domain != VulkanModelDomain::Generic ||
      request.execution_phase != VulkanExecutionPhase::None) {
    return request;
  }

  const std::string& allocation_label = current_planning_label();
  if (
      !allocation_label.empty() && allocation_label != "unlabeled" &&
      !is_runtime_llm_operator_label(allocation_label)) {
    return request;
  }

  const auto inferred_phase = infer_llm_runtime_phase_from_tensor(tensor);
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
