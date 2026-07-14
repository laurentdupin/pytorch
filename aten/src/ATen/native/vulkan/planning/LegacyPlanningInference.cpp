#include <ATen/native/vulkan/planning/LegacyPlanningInference.h>

#include <ATen/native/vulkan/api/Resource.h>

#include <algorithm>
#include <initializer_list>
#include <string>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {
namespace legacy {

namespace {

constexpr int64_t kLlmlikeHiddenSizeThreshold = 64;
constexpr int64_t kLlmlikeMaxSequenceExtent = 64;
constexpr int64_t kLlmlikeMaxPrefixExtent = 64;

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

} // namespace

VulkanModelDomain infer_model_domain_from_planning_label() {
  const std::string& planning_label = current_planning_label();
  if (planning_label.empty() || planning_label == "unlabeled") {
    return VulkanModelDomain::Generic;
  }

  if (allocation_label_contains(
          planning_label, {"llama", "decoder", "lm_head", "self_attn"})) {
    return VulkanModelDomain::LLM;
  }
  if (allocation_label_contains(
          planning_label,
          {"depth", "dino", "beit", "zoe", "midas", "patch_embed", "refinenet"})) {
    return VulkanModelDomain::Vision;
  }
  return VulkanModelDomain::Generic;
}

bool planning_label_allows_llm_tensor_inference() {
  const std::string& planning_label = current_planning_label();
  return planning_label.empty() || planning_label == "unlabeled" ||
      is_runtime_llm_operator_label(planning_label);
}

std::optional<VulkanExecutionPhase> infer_llm_phase_from_tensor_shape(
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

} // namespace legacy
} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
