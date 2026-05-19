#include <ATen/Functions.h>
#include <ATen/native/vulkan/api/Diagnostics.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Sync.h>
#include <ATen/native/vulkan/ops/BinaryOp.h>
#include <ATen/native/vulkan/ops/Clamp.h>
#include <ATen/native/vulkan/ops/Concat.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/LayoutTransitions.h>
#include <ATen/native/vulkan/ops/Softmax.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Upsample.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/ops/VisionBlocks.h>
#include <ATen/native/vulkan/planning/CompiledSession.h>
#include <ATen/native/vulkan/planning/ExecutableRegions.h>
#include <ATen/native/vulkan/planning/ExecutionObjects.h>
#include <ATen/native/vulkan/planning/ExecutionPrograms.h>
#include <ATen/native/vulkan/planning/InferenceGraphs.h>
#include <ATen/native/vulkan/planning/ReplayTensorState.h>
#include <ATen/native/vulkan/planning/Request.h>
#include <ATen/native/vulkan/planning/Runtime.h>

#include <c10/util/ScopeExit.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

bool operator==(
    const VulkanVisionStackShapeKey& lhs,
    const VulkanVisionStackShapeKey& rhs) {
  return lhs.tokens == rhs.tokens && lhs.hidden == rhs.hidden &&
      lhs.num_heads == rhs.num_heads && lhs.head_dim == rhs.head_dim &&
      lhs.mlp_hidden == rhs.mlp_hidden &&
      lhs.num_blocks == rhs.num_blocks && lhs.dtype == rhs.dtype &&
      lhs.device_capability_key == rhs.device_capability_key &&
      lhs.layout_policy_version == rhs.layout_policy_version &&
      lhs.attention_policy_version == rhs.attention_policy_version &&
      lhs.owner_program_version == rhs.owner_program_version &&
      lhs.requested_intermediate_mask == rhs.requested_intermediate_mask &&
      lhs.direct_attention == rhs.direct_attention &&
      lhs.q4_subgroup_available == rhs.q4_subgroup_available;
}

size_t VulkanVisionStackShapeKeyHash::operator()(
    const VulkanVisionStackShapeKey& key) const {
  size_t seed = 0u;
  const auto mix = [&seed](const uint64_t value) {
    seed ^= std::hash<uint64_t>{}(value) + 0x9e3779b97f4a7c15ULL +
        (seed << 6) + (seed >> 2);
  };
  mix(static_cast<uint64_t>(key.tokens));
  mix(static_cast<uint64_t>(key.hidden));
  mix(static_cast<uint64_t>(key.num_heads));
  mix(static_cast<uint64_t>(key.head_dim));
  mix(static_cast<uint64_t>(key.mlp_hidden));
  mix(static_cast<uint64_t>(key.num_blocks));
  mix(static_cast<uint64_t>(key.dtype));
  mix(key.device_capability_key);
  mix(key.layout_policy_version);
  mix(key.attention_policy_version);
  mix(key.owner_program_version);
  mix(key.requested_intermediate_mask);
  mix(key.direct_attention ? 1u : 0u);
  mix(key.q4_subgroup_available ? 1u : 0u);
  return seed;
}

std::string format_stack_shape_key(const VulkanVisionStackShapeKey& key) {
  std::ostringstream out;
  out << "tokens=" << key.tokens
      << ",hidden=" << key.hidden
      << ",heads=" << key.num_heads
      << ",head_dim=" << key.head_dim
      << ",mlp_hidden=" << key.mlp_hidden
      << ",blocks=" << key.num_blocks
      << ",dtype=" << c10::toString(key.dtype)
      << ",capability=" << key.device_capability_key
      << ",layout_policy=" << key.layout_policy_version
      << ",attention_policy=" << key.attention_policy_version
      << ",owner_program=" << key.owner_program_version
      << ",requested_mask=" << key.requested_intermediate_mask
      << ",direct_attention=" << (key.direct_attention ? 1 : 0)
      << ",q4_subgroup=" << (key.q4_subgroup_available ? 1 : 0);
  return out.str();
}

namespace {

std::atomic<uint64_t> g_next_vision_backbone_context_cache_id{1u};

struct VulkanVisionOwnerCounters final {
  std::atomic<uint64_t> total_attempts{0u};
  std::atomic<uint64_t> block_owner_hit{0u};
  std::atomic<uint64_t> stack_owner_hit{0u};
  std::atomic<uint64_t> compiled_session_hit{0u};
  std::atomic<uint64_t> reject_gate_disabled{0u};
  std::atomic<uint64_t> reject_missing_context{0u};
  std::atomic<uint64_t> reject_dtype{0u};
  std::atomic<uint64_t> reject_shape{0u};
  std::atomic<uint64_t> reject_layout{0u};
  std::atomic<uint64_t> reject_route_policy{0u};
  std::atomic<uint64_t> reject_python_bridge{0u};
};

struct VulkanVisionOwnerContextCounters final {
  std::atomic<uint64_t> create_count{0u};
  std::atomic<uint64_t> cache_hit_count{0u};
  std::atomic<uint64_t> unpack_readback_count{0u};
};

struct VulkanVisionOwnerMlpCounters final {
  std::atomic<uint64_t> total{0u};
  std::atomic<uint64_t> linear_gelu_hit{0u};
  std::atomic<uint64_t> fc2_after_linear_gelu_hit{0u};
  std::atomic<uint64_t> reject_no_owner{0u};
  std::atomic<uint64_t> reject_dtype{0u};
  std::atomic<uint64_t> reject_shape{0u};
  std::atomic<uint64_t> reject_context{0u};
};

struct VulkanVisionStackOwnerCounters final {
  std::atomic<uint64_t> total_attempts{0u};
  std::atomic<uint64_t> stack_owner_hit{0u};
  std::atomic<uint64_t> block_context_count{0u};
  std::atomic<uint64_t> block_execute_count{0u};
  std::atomic<uint64_t> reject_missing_context{0u};
  std::atomic<uint64_t> reject_shape{0u};
  std::atomic<uint64_t> reject_dtype{0u};
  std::atomic<uint64_t> reject_layout{0u};
  std::atomic<uint64_t> reject_unsafe_replay{0u};
};

struct VulkanStackAttentionCounters final {
  std::atomic<uint64_t> total{0u};
  std::atomic<uint64_t> direct_hit{0u};
  std::atomic<uint64_t> decomposed_placeholder_bypass{0u};
  std::atomic<uint64_t> reject_shape{0u};
  std::atomic<uint64_t> reject_dtype{0u};
  std::atomic<uint64_t> reject_layout{0u};
};

struct VulkanStackExecutionManifestRow final {
  uint64_t ordinal = 0u;
  int64_t block_index = -1;
  api::VulkanVisionStackPhase phase = api::VulkanVisionStackPhase::Unknown;
  std::string op_label;
  std::string kernel_name;
  std::string input_shapes;
  std::string output_shapes;
  std::string dtype;
  bool uses_dynamic_shape = false;
  bool allocates_output = false;
  bool writes_preexisting_output = false;
  bool escapes_stack = false;
  bool requested_intermediate = false;
  bool requires_cpu_data = false;
  bool uses_fallback = false;
  bool submits_command_buffer = false;
  bool requires_host_sync = false;
  bool uses_runtime_capture = false;
  bool uses_replay = false;
  bool safe_to_capture = false;
};

struct VulkanStackShapePlanCounters final {
  std::atomic<uint64_t> total_attempts{0u};
  std::atomic<uint64_t> plan_build_count{0u};
  std::atomic<uint64_t> plan_cache_hit_count{0u};
  std::atomic<uint64_t> plan_reject_count{0u};
  std::atomic<uint64_t> binding_valid_count{0u};
  std::atomic<uint64_t> binding_invalid_count{0u};
  std::atomic<uint64_t> invalid_tokens{0u};
  std::atomic<uint64_t> invalid_dtype{0u};
  std::atomic<uint64_t> invalid_capability{0u};
  std::atomic<uint64_t> invalid_requested_intermediates{0u};
  std::atomic<uint64_t> invalid_context_identity{0u};
};

struct VulkanStackReplayCounters final {
  std::atomic<uint64_t> total_attempts{0u};
  std::atomic<uint64_t> capture_build_count{0u};
  std::atomic<uint64_t> replay_hit_count{0u};
  std::atomic<uint64_t> descriptor_rebind_count{0u};
  std::atomic<uint64_t> reject_readiness{0u};
  std::atomic<uint64_t> reject_binding_mode{0u};
  std::atomic<uint64_t> reject_binding_validation{0u};
  std::atomic<uint64_t> reject_context_identity{0u};
  std::atomic<uint64_t> reject_capability{0u};
  std::atomic<uint64_t> reject_runtime_capture_active{0u};
};

struct VulkanStackPlannedRecordingCounters final {
  std::atomic<uint64_t> total_attempts{0u};
  std::atomic<uint64_t> planned_record_hit{0u};
  std::atomic<uint64_t> recording_scope_begin_count{0u};
  std::atomic<uint64_t> recording_scope_submit_count{0u};
  std::atomic<uint64_t> recording_scope_reject_count{0u};
  std::atomic<uint64_t> reject_readiness{0u};
  std::atomic<uint64_t> reject_active_capture{0u};
  std::atomic<uint64_t> reject_nested_replay{0u};
  std::atomic<uint64_t> reject_barrier{0u};
  std::atomic<uint64_t> reject_descriptor{0u};
  std::atomic<uint64_t> reject_lifetime{0u};
};

enum class VulkanReplayBindingMode : uint8_t {
  Unknown = 0,
  PersistentResourcesOnly,
  RebindDescriptorSetsPerForward,
  ReRecordCommandBufferPerForward,
  UnsafeStaleDescriptors,
};

std::mutex& stack_execution_manifest_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<VulkanStackExecutionManifestRow>& stack_execution_manifest_rows() {
  static std::vector<VulkanStackExecutionManifestRow> rows;
  return rows;
}

std::mutex& stack_shape_plan_summary_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::string, std::string>& stack_shape_plan_readiness_rows() {
  static std::unordered_map<std::string, std::string> rows;
  return rows;
}

std::vector<std::string>& stack_shape_plan_manifest_rows() {
  static std::vector<std::string> rows;
  return rows;
}

std::mutex& stack_resource_binding_manifest_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<std::string>& stack_resource_binding_manifest_rows() {
  static std::vector<std::string> rows;
  return rows;
}

std::unordered_map<std::string, std::string>& stack_replay_binding_mode_rows() {
  static std::unordered_map<std::string, std::string> rows;
  return rows;
}

std::vector<std::string>& stack_descriptor_binding_table_rows() {
  static std::vector<std::string> rows;
  return rows;
}

std::unordered_map<std::string, std::string>&
stack_descriptor_binding_validation_rows() {
  static std::unordered_map<std::string, std::string> rows;
  return rows;
}

VulkanVisionOwnerCounters& vulkan_vision_owner_counters() {
  static VulkanVisionOwnerCounters counters;
  return counters;
}

VulkanVisionOwnerContextCounters& vulkan_vision_owner_context_counters() {
  static VulkanVisionOwnerContextCounters counters;
  return counters;
}

VulkanVisionOwnerMlpCounters& vulkan_vision_owner_mlp_counters() {
  static VulkanVisionOwnerMlpCounters counters;
  return counters;
}

VulkanVisionStackOwnerCounters& vulkan_vision_stack_owner_counters() {
  static VulkanVisionStackOwnerCounters counters;
  return counters;
}

VulkanStackAttentionCounters& vulkan_stack_attention_counters() {
  static VulkanStackAttentionCounters counters;
  return counters;
}

VulkanStackShapePlanCounters& vulkan_stack_shape_plan_counters() {
  static VulkanStackShapePlanCounters counters;
  return counters;
}

VulkanStackReplayCounters& vulkan_stack_replay_counters() {
  static VulkanStackReplayCounters counters;
  return counters;
}

VulkanStackPlannedRecordingCounters&
vulkan_stack_planned_recording_counters() {
  static VulkanStackPlannedRecordingCounters counters;
  return counters;
}

bool has_explicit_runtime_capture_label();

const std::string& vulkan_vision_owner_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_VISION_OWNER_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

void append_vulkan_vision_owner_log(
    const char* kind,
    const bool selected,
    const char* reject,
    const Tensor& input,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context) {
  const auto& path = vulkan_vision_owner_log_path();
  if (path.empty()) {
    return;
  }

  std::ofstream out(path, std::ios::app);
  out << "vision_owner_attempt"
      << " kind=" << (kind ? kind : "unknown")
      << " selected=" << (selected ? 1 : 0)
      << " reject=" << (reject ? reject : "none")
      << " rank=" << input.dim()
      << " batch=" << (input.dim() == 3 ? input.size(0) : 1)
      << " tokens="
      << (input.dim() == 3 ? input.size(1) : (input.dim() == 2 ? input.size(0) : 0))
      << " hidden=" << (input.dim() >= 1 ? input.size(input.dim() - 1) : 0)
      << " heads=" << (context ? context->num_heads() : 0)
      << " dtype=" << static_cast<int>(input.scalar_type())
      << " input_vulkan=" << (input.is_vulkan() ? 1 : 0)
      << " qkv_context=" << (context && context->qkv_context() ? 1 : 0)
      << " proj_context=" << (context && context->proj_context() ? 1 : 0)
      << " fc1_context=" << (context && context->fc1_context() ? 1 : 0)
      << " fc2_context=" << (context && context->fc2_context() ? 1 : 0)
      << " norm_context="
      << (context && context->norm1_context() && context->norm2_context() ? 1 : 0)
      << '\n';
}

std::string stack_manifest_shape_string(const Tensor& tensor) {
  if (!tensor.defined()) {
    return "[]";
  }
  std::ostringstream out;
  out << '[';
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    if (i > 0) {
      out << ',';
    }
    out << tensor.size(i);
  }
  out << ']';
  return out.str();
}

std::string stack_manifest_shapes_string(
    std::initializer_list<std::reference_wrapper<const Tensor>> tensors) {
  std::ostringstream out;
  bool first = true;
  for (const Tensor& tensor : tensors) {
    if (!tensor.defined()) {
      continue;
    }
    if (!first) {
      out << ';';
    }
    out << stack_manifest_shape_string(tensor);
    first = false;
  }
  return first ? "[]" : out.str();
}

std::string stack_manifest_dtype_string(
    std::initializer_list<std::reference_wrapper<const Tensor>> tensors) {
  for (const Tensor& tensor : tensors) {
    if (tensor.defined()) {
      return tensor.scalar_type() == kFloat
          ? "Float"
          : c10::toString(tensor.scalar_type());
    }
  }
  return "Undefined";
}

std::string stack_plan_shape_string(const std::vector<int64_t>& shape) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      out << ',';
    }
    out << shape[i];
  }
  out << ']';
  return out.str();
}

const char* stack_plan_step_kind_name(const VulkanStackPlanStepKind kind) {
  switch (kind) {
    case VulkanStackPlanStepKind::Norm1:
      return "norm1";
    case VulkanStackPlanStepKind::QkvLinear:
      return "qkv_linear";
    case VulkanStackPlanStepKind::QkvTransform:
      return "qkv_transform";
    case VulkanStackPlanStepKind::Attention:
      return "attention";
    case VulkanStackPlanStepKind::ProjLinear:
      return "proj_linear";
    case VulkanStackPlanStepKind::Residual1:
      return "residual1";
    case VulkanStackPlanStepKind::Norm2:
      return "norm2";
    case VulkanStackPlanStepKind::Fc1Gelu:
      return "fc1_gelu";
    case VulkanStackPlanStepKind::Fc2:
      return "fc2";
    case VulkanStackPlanStepKind::Residual2:
      return "residual2";
    case VulkanStackPlanStepKind::IntermediateCapture:
      return "intermediate_capture";
  }
  return "unknown";
}

const char* replay_binding_mode_name(const VulkanReplayBindingMode mode) {
  switch (mode) {
    case VulkanReplayBindingMode::Unknown:
      return "unknown";
    case VulkanReplayBindingMode::PersistentResourcesOnly:
      return "persistent_resources_only";
    case VulkanReplayBindingMode::RebindDescriptorSetsPerForward:
      return "rebind_descriptor_sets_per_forward";
    case VulkanReplayBindingMode::ReRecordCommandBufferPerForward:
      return "re_record_command_buffer_per_forward";
    case VulkanReplayBindingMode::UnsafeStaleDescriptors:
      return "unsafe_stale_descriptors";
  }
  return "unknown";
}

const char* stack_resource_kind_name(const VulkanStackResourceKind kind) {
  switch (kind) {
    case VulkanStackResourceKind::Unknown:
      return "unknown";
    case VulkanStackResourceKind::StorageBuffer:
      return "storage_buffer";
    case VulkanStackResourceKind::UniformBuffer:
      return "uniform_buffer";
    case VulkanStackResourceKind::Image:
      return "image";
    case VulkanStackResourceKind::Sampler:
      return "sampler";
  }
  return "unknown";
}

const char* stack_resource_lifetime_name(
    const VulkanStackResourceLifetime lifetime) {
  switch (lifetime) {
    case VulkanStackResourceLifetime::Unknown:
      return "unknown";
    case VulkanStackResourceLifetime::PersistentWeight:
      return "persistent_weight";
    case VulkanStackResourceLifetime::PersistentBias:
      return "persistent_bias";
    case VulkanStackResourceLifetime::PersistentNormParam:
      return "persistent_norm_param";
    case VulkanStackResourceLifetime::RuntimeInput:
      return "runtime_input";
    case VulkanStackResourceLifetime::RuntimeOutput:
      return "runtime_output";
    case VulkanStackResourceLifetime::RequestedIntermediateOutput:
      return "requested_intermediate_output";
    case VulkanStackResourceLifetime::InternalTemp:
      return "internal_temp";
    case VulkanStackResourceLifetime::UniformMetadata:
      return "uniform_metadata";
  }
  return "unknown";
}

const char* stack_descriptor_binding_mode_name(
    const VulkanStackDescriptorBindingMode mode) {
  switch (mode) {
    case VulkanStackDescriptorBindingMode::Unknown:
      return "unknown";
    case VulkanStackDescriptorBindingMode::Persistent:
      return "persistent";
    case VulkanStackDescriptorBindingMode::RuntimeRebind:
      return "runtime_rebind";
    case VulkanStackDescriptorBindingMode::ProgramOwnedTemp:
      return "program_owned_temp";
    case VulkanStackDescriptorBindingMode::Unsupported:
      return "unsupported";
  }
  return "unknown";
}

const char* stack_descriptor_type_name(const VkDescriptorType type) {
  switch (type) {
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      return "storage_buffer";
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      return "uniform_buffer";
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      return "storage_image";
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      return "combined_image_sampler";
    default:
      return "unknown";
  }
}

uint64_t requested_intermediate_mask(IntArrayRef capture_indices) {
  uint64_t mask = 0u;
  for (const int64_t index : capture_indices) {
    if (index >= 0 && index < 64) {
      mask |= (1ULL << static_cast<uint64_t>(index));
    }
  }
  return mask;
}

bool q4_subgroup_attention_available() {
  const auto caps = attention_subgroup_capabilities_snapshot();
  return caps.size() >= 9 && caps[0] != 0 && caps[4] != 0 && caps[8] != 0;
}

VulkanVisionStackShapeKey make_stack_shape_key(
    const VisionBackboneStackContext& context,
    const Tensor& input,
    IntArrayRef capture_indices) {
  VulkanVisionStackShapeKey key;
  key.tokens = input.dim() == 2 ? input.size(0) : input.size(1);
  key.hidden = input.size(input.dim() - 1);
  key.num_heads = context.num_heads();
  key.head_dim = context.head_dim();
  key.mlp_hidden = context.mlp_hidden();
  key.num_blocks = static_cast<int64_t>(context.blocks().size());
  key.dtype = input.scalar_type();
  key.q4_subgroup_available = q4_subgroup_attention_available();
  key.device_capability_key = key.q4_subgroup_available ? 2u : 1u;
  key.layout_policy_version = 1u;
  key.attention_policy_version = 1u;
  key.owner_program_version = 1u;
  key.requested_intermediate_mask = requested_intermediate_mask(capture_indices);
  key.direct_attention = true;
  return key;
}

void add_stack_plan_step(
    VulkanVisionStackShapePlan& plan,
    const int64_t block_index,
    const VulkanStackPlanStepKind kind,
    const char* op_label,
    const char* kernel_label,
    std::vector<int64_t> input_shape,
    std::vector<int64_t> output_shape,
    const bool escapes_stack = false,
    const bool requested_intermediate = false) {
  VulkanStackPlanStep step;
  step.ordinal = static_cast<int64_t>(plan.steps.size()) + 1;
  step.block_index = block_index;
  step.kind = kind;
  step.op_label = op_label ? op_label : "unknown";
  step.kernel_label = kernel_label ? kernel_label : "unknown";
  step.input_shape = std::move(input_shape);
  step.output_shape = std::move(output_shape);
  step.dtype = plan.key.dtype;
  step.allocates_output = kind != VulkanStackPlanStepKind::IntermediateCapture;
  step.writes_preexisting_output = false;
  step.escapes_stack = escapes_stack;
  step.requested_intermediate = requested_intermediate;
  plan.steps.emplace_back(std::move(step));
}

VulkanStackDescriptorBindingMode choose_stack_descriptor_binding_mode(
    const VulkanStackResourceLifetime lifetime) {
  switch (lifetime) {
    case VulkanStackResourceLifetime::PersistentWeight:
    case VulkanStackResourceLifetime::PersistentBias:
    case VulkanStackResourceLifetime::PersistentNormParam:
      return VulkanStackDescriptorBindingMode::Persistent;
    case VulkanStackResourceLifetime::RuntimeInput:
    case VulkanStackResourceLifetime::RuntimeOutput:
    case VulkanStackResourceLifetime::RequestedIntermediateOutput:
    case VulkanStackResourceLifetime::UniformMetadata:
      return VulkanStackDescriptorBindingMode::RuntimeRebind;
    case VulkanStackResourceLifetime::InternalTemp:
      return VulkanStackDescriptorBindingMode::ProgramOwnedTemp;
    case VulkanStackResourceLifetime::Unknown:
      return VulkanStackDescriptorBindingMode::Unsupported;
  }
  return VulkanStackDescriptorBindingMode::Unsupported;
}

void add_stack_descriptor_binding(
    VulkanVisionStackShapePlan& plan,
    const VulkanStackPlanStep& step,
    const char* role,
    const VulkanStackResourceKind kind,
    const VulkanStackResourceLifetime lifetime,
    const uint32_t binding_index,
    const VkDescriptorType descriptor_type,
    std::vector<int64_t> shape) {
  VulkanStackDescriptorBinding binding;
  binding.ordinal = step.ordinal;
  binding.block_index = step.block_index;
  binding.phase = step.kind;
  binding.op_label = step.op_label;
  binding.kernel_label = step.kernel_label;
  binding.resource_role = role ? role : "unknown";
  binding.resource_kind = kind;
  binding.lifetime = lifetime;
  binding.binding_mode = choose_stack_descriptor_binding_mode(lifetime);
  binding.descriptor_set_index = 0u;
  binding.binding_index = binding_index;
  binding.descriptor_type = descriptor_type;
  binding.tensor_shape = std::move(shape);
  binding.dtype = step.dtype;
  binding.is_runtime_varying =
      binding.binding_mode != VulkanStackDescriptorBindingMode::Persistent;
  binding.requires_descriptor_update = binding.is_runtime_varying;
  binding.is_persistent =
      binding.binding_mode == VulkanStackDescriptorBindingMode::Persistent;
  binding.escapes_stack = step.escapes_stack ||
      lifetime == VulkanStackResourceLifetime::RuntimeOutput ||
      lifetime == VulkanStackResourceLifetime::RequestedIntermediateOutput;
  binding.descriptor_indices_known =
      binding.binding_mode != VulkanStackDescriptorBindingMode::Unsupported;
  binding.safe_to_rebind =
      binding.binding_mode != VulkanStackDescriptorBindingMode::Unsupported;
  plan.descriptor_bindings.emplace_back(std::move(binding));
}

void append_linear_descriptor_bindings(
    VulkanVisionStackShapePlan& plan,
    const VulkanStackPlanStep& step) {
  add_stack_descriptor_binding(
      plan,
      step,
      step.escapes_stack ? "runtime_output" : "internal_output",
      VulkanStackResourceKind::StorageBuffer,
      step.escapes_stack ? VulkanStackResourceLifetime::RuntimeOutput
                         : VulkanStackResourceLifetime::InternalTemp,
      0u,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      step.output_shape);
  add_stack_descriptor_binding(
      plan,
      step,
      "output_metadata",
      VulkanStackResourceKind::UniformBuffer,
      VulkanStackResourceLifetime::UniformMetadata,
      1u,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      step.output_shape);
  add_stack_descriptor_binding(
      plan,
      step,
      step.block_index == 0 && step.kind == VulkanStackPlanStepKind::QkvLinear
          ? "runtime_input"
          : "activation_input",
      VulkanStackResourceKind::StorageBuffer,
      step.block_index == 0 && step.kind == VulkanStackPlanStepKind::QkvLinear
          ? VulkanStackResourceLifetime::RuntimeInput
          : VulkanStackResourceLifetime::InternalTemp,
      2u,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      step.input_shape);
  add_stack_descriptor_binding(
      plan,
      step,
      "input_metadata",
      VulkanStackResourceKind::UniformBuffer,
      VulkanStackResourceLifetime::UniformMetadata,
      3u,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      step.input_shape);
  add_stack_descriptor_binding(
      plan,
      step,
      "packed_weight",
      VulkanStackResourceKind::StorageBuffer,
      VulkanStackResourceLifetime::PersistentWeight,
      4u,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      {});
  add_stack_descriptor_binding(
      plan,
      step,
      "weight_metadata",
      VulkanStackResourceKind::UniformBuffer,
      VulkanStackResourceLifetime::PersistentWeight,
      5u,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      {});
  add_stack_descriptor_binding(
      plan,
      step,
      "bias",
      VulkanStackResourceKind::StorageBuffer,
      VulkanStackResourceLifetime::PersistentBias,
      6u,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      {step.output_shape.empty() ? 0 : step.output_shape.back()});
  add_stack_descriptor_binding(
      plan,
      step,
      "bias_metadata",
      VulkanStackResourceKind::UniformBuffer,
      VulkanStackResourceLifetime::PersistentBias,
      7u,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      {step.output_shape.empty() ? 0 : step.output_shape.back()});
  add_stack_descriptor_binding(
      plan,
      step,
      "params",
      VulkanStackResourceKind::UniformBuffer,
      VulkanStackResourceLifetime::UniformMetadata,
      8u,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      {});
}

void append_stack_descriptor_bindings_for_step(
    VulkanVisionStackShapePlan& plan,
    const VulkanStackPlanStep& step) {
  if (step.kind == VulkanStackPlanStepKind::IntermediateCapture) {
    add_stack_descriptor_binding(
        plan,
        step,
        "requested_intermediate_output",
        VulkanStackResourceKind::StorageBuffer,
        VulkanStackResourceLifetime::RequestedIntermediateOutput,
        0u,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        step.output_shape);
    return;
  }

  switch (step.kind) {
    case VulkanStackPlanStepKind::QkvLinear:
    case VulkanStackPlanStepKind::ProjLinear:
    case VulkanStackPlanStepKind::Fc1Gelu:
    case VulkanStackPlanStepKind::Fc2:
      append_linear_descriptor_bindings(plan, step);
      return;
    case VulkanStackPlanStepKind::Attention:
      add_stack_descriptor_binding(
          plan,
          step,
          "attention_output",
          VulkanStackResourceKind::StorageBuffer,
          VulkanStackResourceLifetime::InternalTemp,
          0u,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          step.output_shape);
      add_stack_descriptor_binding(
          plan,
          step,
          "output_metadata",
          VulkanStackResourceKind::UniformBuffer,
          VulkanStackResourceLifetime::UniformMetadata,
          1u,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          step.output_shape);
      for (const auto& role_binding : {
               std::pair<const char*, uint32_t>{"query", 2u},
               std::pair<const char*, uint32_t>{"key", 4u},
               std::pair<const char*, uint32_t>{"value", 6u},
           }) {
        add_stack_descriptor_binding(
            plan,
            step,
            role_binding.first,
            VulkanStackResourceKind::StorageBuffer,
            VulkanStackResourceLifetime::InternalTemp,
            role_binding.second,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            step.input_shape);
        add_stack_descriptor_binding(
            plan,
            step,
            (std::string(role_binding.first) + "_metadata").c_str(),
            VulkanStackResourceKind::UniformBuffer,
            VulkanStackResourceLifetime::UniformMetadata,
            role_binding.second + 1u,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            step.input_shape);
      }
      add_stack_descriptor_binding(
          plan,
          step,
          "params",
          VulkanStackResourceKind::UniformBuffer,
          VulkanStackResourceLifetime::UniformMetadata,
          8u,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          {});
      return;
    case VulkanStackPlanStepKind::Norm1:
    case VulkanStackPlanStepKind::Norm2:
    case VulkanStackPlanStepKind::QkvTransform:
    case VulkanStackPlanStepKind::Residual1:
    case VulkanStackPlanStepKind::Residual2:
      add_stack_descriptor_binding(
          plan,
          step,
          step.escapes_stack ? "runtime_output" : "internal_output",
          VulkanStackResourceKind::StorageBuffer,
          step.escapes_stack ? VulkanStackResourceLifetime::RuntimeOutput
                             : VulkanStackResourceLifetime::InternalTemp,
          0u,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          step.output_shape);
      add_stack_descriptor_binding(
          plan,
          step,
          "activation_input",
          VulkanStackResourceKind::StorageBuffer,
          step.block_index == 0 && step.kind == VulkanStackPlanStepKind::Norm1
              ? VulkanStackResourceLifetime::RuntimeInput
              : VulkanStackResourceLifetime::InternalTemp,
          1u,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          step.input_shape);
      if (step.kind == VulkanStackPlanStepKind::Norm1 ||
          step.kind == VulkanStackPlanStepKind::Norm2) {
        add_stack_descriptor_binding(
            plan,
            step,
            "norm_weight",
            VulkanStackResourceKind::StorageBuffer,
            VulkanStackResourceLifetime::PersistentNormParam,
            2u,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            {plan.key.hidden});
        add_stack_descriptor_binding(
            plan,
            step,
            "norm_bias",
            VulkanStackResourceKind::StorageBuffer,
            VulkanStackResourceLifetime::PersistentNormParam,
            3u,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            {plan.key.hidden});
      }
      return;
  }
}

void build_stack_descriptor_binding_table(VulkanVisionStackShapePlan& plan) {
  plan.descriptor_bindings.clear();
  for (const auto& step : plan.steps) {
    append_stack_descriptor_bindings_for_step(plan, step);
  }
  plan.descriptor_table_complete = std::all_of(
      plan.descriptor_bindings.begin(),
      plan.descriptor_bindings.end(),
      [](const VulkanStackDescriptorBinding& binding) {
        return binding.descriptor_indices_known &&
            binding.binding_mode != VulkanStackDescriptorBindingMode::Unknown;
      });
  plan.descriptors_rebindable = std::all_of(
      plan.descriptor_bindings.begin(),
      plan.descriptor_bindings.end(),
      [](const VulkanStackDescriptorBinding& binding) {
        return binding.safe_to_rebind ||
            binding.binding_mode == VulkanStackDescriptorBindingMode::Persistent;
      });
  const bool internal_temps_rebindable = std::all_of(
      plan.descriptor_bindings.begin(),
      plan.descriptor_bindings.end(),
      [](const VulkanStackDescriptorBinding& binding) {
        return binding.lifetime != VulkanStackResourceLifetime::InternalTemp ||
            binding.binding_mode ==
                VulkanStackDescriptorBindingMode::ProgramOwnedTemp;
      });
  plan.descriptor_re_record_ready =
      plan.descriptor_table_complete && plan.descriptors_rebindable &&
      internal_temps_rebindable;
  plan.descriptor_replay_ready = false;
}

std::unique_ptr<VulkanVisionStackShapePlan> build_stack_shape_plan(
    const VisionBackboneStackContext& context,
    const VulkanVisionStackShapeKey& key) {
  auto plan = std::make_unique<VulkanVisionStackShapePlan>();
  plan->key = key;

  const std::vector<int64_t> hidden_shape{1, key.tokens, key.hidden};
  const std::vector<int64_t> qkv_shape{1, key.tokens, key.hidden * 3};
  const std::vector<int64_t> qkv_head_shape{
      key.num_heads, key.tokens, key.head_dim};
  const std::vector<int64_t> mlp_shape{1, key.tokens, key.mlp_hidden};

  for (int64_t block = 0; block < key.num_blocks; ++block) {
    add_stack_plan_step(
        *plan,
        block,
        VulkanStackPlanStepKind::Norm1,
        "vision_block.norm1",
        "layernorm_buffer",
        hidden_shape,
        hidden_shape);
    add_stack_plan_step(
        *plan,
        block,
        VulkanStackPlanStepKind::QkvLinear,
        "vision_block.qkv_linear",
        "mm_buffer_float_bias",
        hidden_shape,
        qkv_shape);
    add_stack_plan_step(
        *plan,
        block,
        VulkanStackPlanStepKind::QkvTransform,
        "vision_block.qkv_transform",
        "transform_bias_rescale_qkv",
        qkv_shape,
        qkv_head_shape);
    add_stack_plan_step(
        *plan,
        block,
        VulkanStackPlanStepKind::Attention,
        "vulkan_prepack::vision_stack_attention_direct",
        key.q4_subgroup_available
            ? "scaled_dot_product_scores_value_buffer_float_head64_q4_subgroup"
            : "scaled_dot_product_scores_value_buffer_float_head64_q4_shared",
        qkv_head_shape,
        qkv_head_shape);
    add_stack_plan_step(
        *plan,
        block,
        VulkanStackPlanStepKind::ProjLinear,
        "vision_block.proj_linear",
        "mm_buffer_float_bias",
        hidden_shape,
        hidden_shape);
    add_stack_plan_step(
        *plan,
        block,
        VulkanStackPlanStepKind::Residual1,
        "vision_block.residual1",
        "add_buffer",
        hidden_shape,
        hidden_shape);
    add_stack_plan_step(
        *plan,
        block,
        VulkanStackPlanStepKind::Norm2,
        "vision_block.norm2",
        "layernorm_buffer",
        hidden_shape,
        hidden_shape);
    add_stack_plan_step(
        *plan,
        block,
        VulkanStackPlanStepKind::Fc1Gelu,
        "vision_block.fc1_gelu",
        "mm_buffer_float_gelu",
        hidden_shape,
        mlp_shape);
    add_stack_plan_step(
        *plan,
        block,
        VulkanStackPlanStepKind::Fc2,
        "vision_block.fc2",
        "mm_buffer_float_bias",
        mlp_shape,
        hidden_shape);
    add_stack_plan_step(
        *plan,
        block,
        VulkanStackPlanStepKind::Residual2,
        "vision_block.residual2",
        "add_buffer",
        hidden_shape,
        hidden_shape);
    if ((key.requested_intermediate_mask & (1ULL << static_cast<uint64_t>(block))) !=
        0u) {
      add_stack_plan_step(
          *plan,
          block,
          VulkanStackPlanStepKind::IntermediateCapture,
          "vision_stack.intermediate_capture",
          "none",
          hidden_shape,
          hidden_shape,
          true,
          true);
    }
  }

  plan->fixed_shapes = key.tokens > 0 && key.hidden > 0 &&
      key.hidden == context.hidden() && key.num_blocks > 0;
  plan->no_cpu_fallback = true;
  plan->no_host_sync = true;
  plan->no_nested_replay = true;
  plan->requested_intermediates_marked =
      key.requested_intermediate_mask == 0u ||
      std::any_of(
          plan->steps.begin(),
          plan->steps.end(),
          [](const VulkanStackPlanStep& step) {
            return step.requested_intermediate && step.escapes_stack;
          });
  plan->internal_outputs_owned = true;
  plan->known_lifetimes = std::all_of(
      plan->steps.begin(),
      plan->steps.end(),
      [](const VulkanStackPlanStep& step) {
        return step.escapes_stack || !step.requested_intermediate;
      });
  build_stack_descriptor_binding_table(*plan);
  return plan;
}

void add_stack_resource_binding_row(
    std::vector<std::string>& rows,
    const VulkanVisionStackShapePlan& plan,
    const VulkanStackPlanStep& step,
    const char* role,
    const char* kind,
    const std::vector<int64_t>& shape,
    const bool is_input,
    const bool is_output,
    const bool is_weight,
    const bool is_bias,
    const bool is_intermediate,
    const bool persistent_across_forwards,
    const bool runtime_rebound_each_forward,
    const int64_t descriptor_set_index,
    const int64_t descriptor_binding_index,
    const int64_t storage_offset,
    const bool requires_descriptor_update,
    const bool safe_for_replay) {
  std::ostringstream out;
  out << "stack_resource_binding"
      << " plan_key=" << format_stack_shape_key(plan.key)
      << " tokens=" << plan.key.tokens
      << " ordinal=" << step.ordinal
      << " block=" << step.block_index
      << " phase=" << stack_plan_step_kind_name(step.kind)
      << " op=" << step.op_label
      << " role=" << (role ? role : "unknown")
      << " kind=" << (kind ? kind : "unknown")
      << " shape=" << stack_plan_shape_string(shape)
      << " dtype=" << c10::toString(step.dtype)
      << " is_input=" << (is_input ? 1 : 0)
      << " is_output=" << (is_output ? 1 : 0)
      << " is_weight=" << (is_weight ? 1 : 0)
      << " is_bias=" << (is_bias ? 1 : 0)
      << " is_intermediate=" << (is_intermediate ? 1 : 0)
      << " escapes_stack=" << (step.escapes_stack ? 1 : 0)
      << " persistent_across_forwards="
      << (persistent_across_forwards ? 1 : 0)
      << " runtime_rebound_each_forward="
      << (runtime_rebound_each_forward ? 1 : 0)
      << " descriptor_set=" << descriptor_set_index
      << " binding=" << descriptor_binding_index
      << " buffer_handle_known=0"
      << " storage_offset=" << storage_offset
      << " requires_descriptor_update="
      << (requires_descriptor_update ? 1 : 0)
      << " safe_for_replay=" << (safe_for_replay ? 1 : 0);
  rows.emplace_back(out.str());
}

void add_stack_step_resource_bindings(
    std::vector<std::string>& rows,
    const VulkanVisionStackShapePlan& plan,
    const VulkanStackPlanStep& step) {
  const std::vector<int64_t> scalar_shape{
      step.output_shape.empty() ? 0 : step.output_shape.back()};
  const bool first_block_input =
      step.block_index == 0 && step.kind == VulkanStackPlanStepKind::Norm1;
  add_stack_resource_binding_row(
      rows,
      plan,
      step,
      first_block_input ? "runtime_input" : "activation_input",
      "buffer",
      step.input_shape,
      true,
      false,
      false,
      false,
      !first_block_input,
      false,
      first_block_input,
      0,
      -1,
      0,
      first_block_input,
      false);

  if (step.kind == VulkanStackPlanStepKind::Norm1 ||
      step.kind == VulkanStackPlanStepKind::Norm2) {
    add_stack_resource_binding_row(
        rows,
        plan,
        step,
        "norm_weight",
        "buffer",
        scalar_shape,
        false,
        false,
        true,
        false,
        false,
        true,
        false,
        0,
        -1,
        0,
        false,
        true);
    add_stack_resource_binding_row(
        rows,
        plan,
        step,
        "norm_bias",
        "buffer",
        scalar_shape,
        false,
        false,
        false,
        true,
        false,
        true,
        false,
        0,
        -1,
        0,
        false,
        true);
  } else if (
      step.kind == VulkanStackPlanStepKind::QkvLinear ||
      step.kind == VulkanStackPlanStepKind::ProjLinear ||
      step.kind == VulkanStackPlanStepKind::Fc1Gelu ||
      step.kind == VulkanStackPlanStepKind::Fc2) {
    add_stack_resource_binding_row(
        rows,
        plan,
        step,
        "packed_weight",
        "buffer",
        {},
        false,
        false,
        true,
        false,
        false,
        true,
        false,
        0,
        -1,
        0,
        false,
        true);
    add_stack_resource_binding_row(
        rows,
        plan,
        step,
        "bias",
        "buffer",
        scalar_shape,
        false,
        false,
        false,
        true,
        false,
        true,
        false,
        0,
        -1,
        0,
        false,
        true);
  } else if (step.kind == VulkanStackPlanStepKind::Attention) {
    for (const char* role : {"query", "key", "value"}) {
      add_stack_resource_binding_row(
          rows,
          plan,
          step,
          role,
          "buffer",
          step.input_shape,
          true,
          false,
          false,
          false,
          true,
          false,
          true,
          0,
          -1,
          0,
          true,
          false);
    }
  }

  add_stack_resource_binding_row(
      rows,
      plan,
      step,
      step.escapes_stack ? "escaping_output" : "internal_output",
      "buffer",
      step.output_shape,
      false,
      true,
      false,
      false,
      !step.escapes_stack,
      false,
      true,
      0,
      -1,
      0,
      true,
      false);
}

VulkanReplayBindingMode determine_stack_replay_binding_mode(
    const VulkanVisionStackShapePlan& plan) {
  if (!plan.descriptor_table_complete) {
    return VulkanReplayBindingMode::UnsafeStaleDescriptors;
  }
  if (plan.descriptor_replay_ready) {
    return VulkanReplayBindingMode::RebindDescriptorSetsPerForward;
  }
  return VulkanReplayBindingMode::ReRecordCommandBufferPerForward;
}

bool stack_plan_ready_for_planned_recording(
    const VulkanVisionStackShapePlan& plan) {
  (void)plan;
  return false;
}

std::string format_stack_descriptor_binding(
    const VulkanVisionStackShapePlan& plan,
    const VulkanStackDescriptorBinding& binding) {
  std::ostringstream out;
  out << "stack_descriptor_binding"
      << " plan_key=" << format_stack_shape_key(plan.key)
      << " tokens=" << plan.key.tokens
      << " ordinal=" << binding.ordinal
      << " block=" << binding.block_index
      << " phase=" << stack_plan_step_kind_name(binding.phase)
      << " op=" << binding.op_label
      << " kernel=" << binding.kernel_label
      << " role=" << binding.resource_role
      << " kind=" << stack_resource_kind_name(binding.resource_kind)
      << " lifetime=" << stack_resource_lifetime_name(binding.lifetime)
      << " mode=" << stack_descriptor_binding_mode_name(binding.binding_mode)
      << " set=" << binding.descriptor_set_index
      << " binding=" << binding.binding_index
      << " descriptor_type="
      << stack_descriptor_type_name(binding.descriptor_type)
      << " shape=" << stack_plan_shape_string(binding.tensor_shape)
      << " dtype=" << c10::toString(binding.dtype)
      << " runtime_varying=" << (binding.is_runtime_varying ? 1 : 0)
      << " requires_update=" << (binding.requires_descriptor_update ? 1 : 0)
      << " persistent=" << (binding.is_persistent ? 1 : 0)
      << " escapes_stack=" << (binding.escapes_stack ? 1 : 0)
      << " indices_known=" << (binding.descriptor_indices_known ? 1 : 0)
      << " safe_to_rebind=" << (binding.safe_to_rebind ? 1 : 0);
  return out.str();
}

std::string format_stack_descriptor_validation(
    const VulkanVisionStackShapePlan& plan) {
  const bool all_runtime_resources_rebindable = std::all_of(
      plan.descriptor_bindings.begin(),
      plan.descriptor_bindings.end(),
      [](const VulkanStackDescriptorBinding& binding) {
        return !binding.is_runtime_varying || binding.safe_to_rebind;
      });
  const bool all_persistent_resources_stable = std::all_of(
      plan.descriptor_bindings.begin(),
      plan.descriptor_bindings.end(),
      [](const VulkanStackDescriptorBinding& binding) {
        return !binding.is_persistent ||
            binding.binding_mode == VulkanStackDescriptorBindingMode::Persistent;
      });
  const bool all_internal_temps_owned_or_rebindable = std::all_of(
      plan.descriptor_bindings.begin(),
      plan.descriptor_bindings.end(),
      [](const VulkanStackDescriptorBinding& binding) {
        return binding.lifetime != VulkanStackResourceLifetime::InternalTemp ||
            binding.binding_mode ==
                VulkanStackDescriptorBindingMode::ProgramOwnedTemp ||
            binding.safe_to_rebind;
      });
  std::ostringstream out;
  out << "stack_descriptor_binding_validation"
      << " plan_key=" << format_stack_shape_key(plan.key)
      << " tokens=" << plan.key.tokens
      << " table_complete=" << (plan.descriptor_table_complete ? 1 : 0)
      << " all_descriptor_indices_known="
      << (plan.descriptor_table_complete ? 1 : 0)
      << " all_runtime_resources_rebindable="
      << (all_runtime_resources_rebindable ? 1 : 0)
      << " all_persistent_resources_stable="
      << (all_persistent_resources_stable ? 1 : 0)
      << " all_internal_temps_owned_or_rebindable="
      << (all_internal_temps_owned_or_rebindable ? 1 : 0)
      << " ready_for_re_record_per_forward="
      << (plan.descriptor_re_record_ready ? 1 : 0)
      << " ready_for_command_replay="
      << (plan.descriptor_replay_ready ? 1 : 0)
      << " rows=" << plan.descriptor_bindings.size();
  return out.str();
}

void record_stack_resource_binding_manifest(
    const VulkanVisionStackShapePlan& plan) {
  std::vector<std::string> new_rows;
  for (const auto& step : plan.steps) {
    if (step.kind == VulkanStackPlanStepKind::IntermediateCapture) {
      add_stack_resource_binding_row(
          new_rows,
          plan,
          step,
          "requested_intermediate_output",
          "buffer",
          step.output_shape,
          false,
          true,
          false,
          false,
          false,
          false,
          true,
          0,
          -1,
          0,
          true,
          false);
      continue;
    }
    add_stack_step_resource_bindings(new_rows, plan, step);
  }

  const std::string key = format_stack_shape_key(plan.key);
  {
    std::lock_guard<std::mutex> lock(stack_resource_binding_manifest_mutex());
    auto& rows = stack_resource_binding_manifest_rows();
    rows.erase(
        std::remove_if(
            rows.begin(),
            rows.end(),
            [&key](const std::string& row) {
              return row.find(" plan_key=" + key + " ") != std::string::npos;
            }),
        rows.end());
    rows.insert(rows.end(), new_rows.begin(), new_rows.end());

    auto& descriptor_rows = stack_descriptor_binding_table_rows();
    descriptor_rows.erase(
        std::remove_if(
            descriptor_rows.begin(),
            descriptor_rows.end(),
            [&key](const std::string& row) {
              return row.find(" plan_key=" + key + " ") != std::string::npos;
            }),
        descriptor_rows.end());
    for (const auto& binding : plan.descriptor_bindings) {
      descriptor_rows.emplace_back(
          format_stack_descriptor_binding(plan, binding));
    }

    stack_descriptor_binding_validation_rows()[key] =
        format_stack_descriptor_validation(plan);

    auto mode = determine_stack_replay_binding_mode(plan);
    std::ostringstream out;
    out << "stack_replay_binding_mode"
        << " plan_key=" << key
        << " tokens=" << plan.key.tokens
        << " mode=" << replay_binding_mode_name(mode)
        << " descriptor_binding_indices_known="
        << (plan.descriptor_table_complete ? 1 : 0)
        << " ready_for_re_record_per_forward="
        << (plan.descriptor_re_record_ready ? 1 : 0)
        << " ready_for_command_replay="
        << (plan.descriptor_replay_ready ? 1 : 0)
        << " reason="
        << (plan.descriptor_replay_ready
                ? "descriptor_sets_rebindable_without_rerecord"
                : "program_owned_temps_not_stable_for_command_replay");
    stack_replay_binding_mode_rows()[key] = out.str();
  }
}

std::string format_stack_shape_plan_readiness(
    const VulkanVisionStackShapePlan& plan) {
  std::ostringstream out;
  out << "stack_shape_plan"
      << " key=" << format_stack_shape_key(plan.key)
      << " tokens=" << plan.key.tokens
      << " fixed_shapes=" << (plan.fixed_shapes ? 1 : 0)
      << " no_cpu_fallback=" << (plan.no_cpu_fallback ? 1 : 0)
      << " no_host_sync=" << (plan.no_host_sync ? 1 : 0)
      << " no_nested_replay=" << (plan.no_nested_replay ? 1 : 0)
      << " requested_intermediates_marked="
      << (plan.requested_intermediates_marked ? 1 : 0)
      << " internal_outputs_owned=" << (plan.internal_outputs_owned ? 1 : 0)
      << " known_lifetimes=" << (plan.known_lifetimes ? 1 : 0)
      << " safe_to_program="
      << (plan.ready_for_programmed_sequence() ? 1 : 0)
      << " steps=" << plan.steps.size();
  return out.str();
}

void record_stack_shape_plan_summary(
    const VulkanVisionStackShapePlan& plan) {
  std::lock_guard<std::mutex> lock(stack_shape_plan_summary_mutex());
  const std::string key = format_stack_shape_key(plan.key);
  stack_shape_plan_readiness_rows()[key] =
      format_stack_shape_plan_readiness(plan);

  auto& manifest_rows = stack_shape_plan_manifest_rows();
  manifest_rows.erase(
      std::remove_if(
          manifest_rows.begin(),
          manifest_rows.end(),
          [&key](const std::string& row) {
            return row.find(" plan_key=" + key + " ") != std::string::npos;
          }),
      manifest_rows.end());
  for (const auto& step : plan.steps) {
    std::ostringstream out;
    out << "stack_shape_plan_manifest"
        << " plan_key=" << key
        << " tokens=" << plan.key.tokens
        << " ordinal=" << step.ordinal
        << " block=" << step.block_index
        << " phase=" << stack_plan_step_kind_name(step.kind)
        << " op=" << step.op_label
        << " kernel=" << step.kernel_label
        << " input_shapes=" << stack_plan_shape_string(step.input_shape)
        << " output_shapes=" << stack_plan_shape_string(step.output_shape)
        << " dtype=" << c10::toString(step.dtype)
        << " uses_dynamic_shape=0"
        << " fixed_shapes=" << (plan.fixed_shapes ? 1 : 0)
        << " allocates_output=" << (step.allocates_output ? 1 : 0)
        << " writes_preexisting_output="
        << (step.writes_preexisting_output ? 1 : 0)
        << " escapes_stack=" << (step.escapes_stack ? 1 : 0)
        << " requested_intermediate="
        << (step.requested_intermediate ? 1 : 0)
        << " safe_to_capture="
        << (plan.ready_for_programmed_sequence() ? 1 : 0);
    manifest_rows.emplace_back(out.str());
  }
  record_stack_resource_binding_manifest(plan);
}

struct VulkanStackPlanRuntimeBinding final {
  int64_t tokens = 0;
  int64_t hidden = 0;
  c10::ScalarType dtype = c10::ScalarType::Undefined;
  uint64_t requested_intermediate_mask = 0u;
};

VulkanStackPlanRuntimeBinding make_stack_plan_runtime_binding(
    const Tensor& input,
    IntArrayRef capture_indices) {
  VulkanStackPlanRuntimeBinding binding;
  binding.tokens = input.dim() == 2 ? input.size(0) : input.size(1);
  binding.hidden = input.size(input.dim() - 1);
  binding.dtype = input.scalar_type();
  binding.requested_intermediate_mask =
      requested_intermediate_mask(capture_indices);
  return binding;
}

bool validate_stack_plan_binding_impl(
    const VulkanVisionStackShapePlan& plan,
    const VulkanStackPlanRuntimeBinding& binding,
    std::string* reason) {
  if (binding.tokens != plan.key.tokens) {
    if (reason) {
      *reason = "tokens_mismatch";
    }
    return false;
  }
  if (binding.hidden != plan.key.hidden) {
    if (reason) {
      *reason = "hidden_mismatch";
    }
    return false;
  }
  if (binding.dtype != plan.key.dtype) {
    if (reason) {
      *reason = "dtype_mismatch";
    }
    return false;
  }
  if (binding.requested_intermediate_mask !=
      plan.key.requested_intermediate_mask) {
    if (reason) {
      *reason = "requested_intermediates_mismatch";
    }
    return false;
  }
  if (reason) {
    *reason = "ok";
  }
  return true;
}

void note_stack_plan_binding_invalid(const std::string& reason) {
  auto& counters = vulkan_stack_shape_plan_counters();
  counters.binding_invalid_count.fetch_add(1u, std::memory_order_relaxed);
  if (reason == "tokens_mismatch") {
    counters.invalid_tokens.fetch_add(1u, std::memory_order_relaxed);
  } else if (reason == "dtype_mismatch") {
    counters.invalid_dtype.fetch_add(1u, std::memory_order_relaxed);
  } else if (reason == "requested_intermediates_mismatch") {
    counters.invalid_requested_intermediates.fetch_add(
        1u,
        std::memory_order_relaxed);
  } else {
    counters.invalid_context_identity.fetch_add(1u, std::memory_order_relaxed);
  }
}

VulkanVisionStackShapePlan& get_or_create_stack_shape_plan(
    VisionBackboneStackContext& context,
    const Tensor& input,
    IntArrayRef capture_indices) {
  auto& counters = vulkan_stack_shape_plan_counters();
  counters.total_attempts.fetch_add(1u, std::memory_order_relaxed);
  const VulkanVisionStackShapeKey key =
      make_stack_shape_key(context, input, capture_indices);

  std::lock_guard<std::mutex> lock(context.shape_plan_mutex());
  auto& plans = context.shape_plans();
  auto it = plans.find(key);
  if (it != plans.end()) {
    counters.plan_cache_hit_count.fetch_add(1u, std::memory_order_relaxed);
    record_stack_shape_plan_summary(*it->second);
    return *it->second;
  }

  auto plan = build_stack_shape_plan(context, key);
  if (!plan->ready_for_programmed_sequence()) {
    counters.plan_reject_count.fetch_add(1u, std::memory_order_relaxed);
  }
  auto& plan_ref = *plan;
  plans.emplace(key, std::move(plan));
  counters.plan_build_count.fetch_add(1u, std::memory_order_relaxed);
  record_stack_shape_plan_summary(plan_ref);
  return plan_ref;
}

void note_stack_execution_manifest_row(
    const char* op_label,
    const char* kernel_name,
    std::initializer_list<std::reference_wrapper<const Tensor>> inputs,
    std::initializer_list<std::reference_wrapper<const Tensor>> outputs,
    const bool allocates_output,
    const bool writes_preexisting_output,
    const bool escapes_stack,
    const bool requested_intermediate,
    const bool submits_command_buffer,
    const bool requires_host_sync = false,
    const bool uses_runtime_capture = false,
    const bool uses_replay = false,
    const bool uses_fallback = false) {
  if (!api::inside_vision_stack_phase()) {
    return;
  }

  VulkanStackExecutionManifestRow row;
  row.block_index = api::current_vision_stack_block_index();
  row.phase = api::current_vision_stack_phase();
  row.op_label = op_label ? op_label : "unknown";
  row.kernel_name = kernel_name ? kernel_name : "unknown";
  row.input_shapes = stack_manifest_shapes_string(inputs);
  row.output_shapes = stack_manifest_shapes_string(outputs);
  row.dtype = stack_manifest_dtype_string(outputs.size() > 0 ? outputs : inputs);
  row.uses_dynamic_shape = true;
  row.allocates_output = allocates_output;
  row.writes_preexisting_output = writes_preexisting_output;
  row.escapes_stack = escapes_stack;
  row.requested_intermediate = requested_intermediate;
  row.requires_cpu_data = false;
  row.uses_fallback = uses_fallback;
  row.submits_command_buffer = submits_command_buffer;
  row.requires_host_sync = requires_host_sync;
  row.uses_runtime_capture = uses_runtime_capture;
  row.uses_replay = uses_replay;
  row.safe_to_capture = !row.uses_dynamic_shape && !row.requires_cpu_data &&
      !row.uses_fallback && !row.requires_host_sync && !row.uses_replay &&
      !row.uses_runtime_capture;

  std::lock_guard<std::mutex> lock(stack_execution_manifest_mutex());
  auto& rows = stack_execution_manifest_rows();
  row.ordinal = static_cast<uint64_t>(rows.size()) + 1u;
  rows.emplace_back(std::move(row));
}

void append_vulkan_vision_stack_owner_log(
    const bool selected,
    const char* reject,
    const Tensor& input,
    const c10::intrusive_ptr<VisionBackboneStackContext>& context) {
  const auto& path = vulkan_vision_owner_log_path();
  if (path.empty()) {
    return;
  }

  const int64_t blocks =
      context ? static_cast<int64_t>(context->blocks().size()) : 0;
  std::ofstream out(path, std::ios::app);
  out << "vision_stack_owner"
      << " selected=" << (selected ? 1 : 0)
      << " reject=" << (reject ? reject : "none")
      << " blocks=" << blocks
      << " tokens="
      << (input.dim() == 3 ? input.size(1) : (input.dim() == 2 ? input.size(0) : 0))
      << " hidden=" << (input.dim() >= 1 ? input.size(input.dim() - 1) : 0)
      << " heads=" << (context ? context->num_heads() : 0)
      << " head_dim=" << (context ? context->head_dim() : 0)
      << " mlp_hidden=" << (context ? context->mlp_hidden() : 0)
      << " owner_forward_fallback=0"
      << " stack_contexts=" << blocks
      << " uses_program=0"
      << " uses_replay=0"
      << " unsafe_nested_replay=0"
      << " dtype=" << static_cast<int>(input.scalar_type())
      << " input_vulkan=" << (input.is_vulkan() ? 1 : 0)
      << '\n';
}

int64_t parse_vision_block_index(const std::string& label) {
  const std::string marker = "block";
  const auto pos = label.rfind(marker);
  if (pos == std::string::npos) {
    return -1;
  }
  int64_t value = 0;
  bool found_digit = false;
  for (size_t i = pos + marker.size(); i < label.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(label[i]))) {
      break;
    }
    found_digit = true;
    value = value * 10 + static_cast<int64_t>(label[i] - '0');
  }
  return found_digit ? value : -1;
}

void append_vulkan_vision_owner_block_log(
    const Tensor& input,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context,
    const bool linear_gelu_context,
    const bool fc2_context) {
  const auto& path = vulkan_vision_owner_log_path();
  if (path.empty() || !context) {
    return;
  }

  const int64_t hidden = input.dim() >= 1 ? input.size(input.dim() - 1) : 0;
  const int64_t heads = context->num_heads();
  const int64_t head_dim = heads > 0 ? hidden / heads : 0;
  const int64_t mlp_hidden = hidden * 4;
  std::ofstream out(path, std::ios::app);
  out << "vision_owner_block"
      << " block_index=" << parse_vision_block_index(context->allocation_label())
      << " label=" << context->allocation_label()
      << " tokens="
      << (input.dim() == 3 ? input.size(1) : (input.dim() == 2 ? input.size(0) : 0))
      << " hidden=" << hidden
      << " heads=" << heads
      << " head_dim=" << head_dim
      << " mlp_hidden=" << mlp_hidden
      << " owner_forward_fallback=0"
      << " linear_gelu_context=" << (linear_gelu_context ? 1 : 0)
      << " fc2_context=" << (fc2_context ? 1 : 0)
      << '\n';
}

struct VisionReplayBundleIdentity final {
  std::string key;
  std::string label_suffix;
};

std::string child_label(const std::string& label, const char* suffix) {
  if (label.empty()) {
    return std::string(suffix);
  }
  return label + "." + suffix;
}

Tensor move_optional_to_vulkan_buffer(const std::optional<Tensor>& tensor) {
  if (!tensor.has_value() || !tensor->defined()) {
    return Tensor();
  }
  Tensor vulkan_tensor = tensor->is_vulkan() ? *tensor : tensor->vulkan();
  return utils::mark_tensor_execution(
      utils::ensure_buffer_storage(
          vulkan_tensor, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
      api::ExecutionLayout::BUFFER_DIRECT,
      true);
}

void recover_after_vulkan_failure_if_needed() {
  if (!api::available()) {
    return;
  }
  api::context()->flush();
  utils::release_retired_packed_weight_entries();
  utils::release_retired_linear_contexts();
  api::clear_vulkan_post_failure_recovery_required();
}

Tensor maybe_restore_tensor(
    const Tensor& tensor,
    const Device& device,
    const ScalarType scalar_type) {
  Tensor restored = tensor;
  if (device.type() != kVulkan) {
    if (tensor.is_vulkan()) {
      report_vulkan_cpu_fallback(
          "vulkan_prepack::vision_context",
          "restore_tensor_cpu_readback",
          {tensor},
          VulkanCpuFallbackKind::SyncReadback);
    }
    restored = tensor.cpu();
  }
  if (restored.scalar_type() != scalar_type) {
    restored = restored.to(scalar_type);
  }
  return restored;
}

Tensor cpu_snapshot_for_unpack(const Tensor& tensor, const char* reason) {
  if (tensor.is_vulkan()) {
    vulkan_vision_owner_context_counters().unpack_readback_count.fetch_add(
        1u,
        std::memory_order_relaxed);
    report_vulkan_cpu_fallback(
        "vulkan_prepack::vision_context",
        reason,
        {tensor},
        VulkanCpuFallbackKind::SyncReadback);
  }
  return tensor.cpu();
}

c10::intrusive_ptr<LayernormPackedContext> make_layernorm_context(
    const Tensor& weight,
    const Tensor& bias,
    const double eps,
    const std::string& label) {
  std::optional<Tensor> owned_weight(weight.clone());
  std::optional<Tensor> owned_bias(bias.clone());
  return create_layernorm_context_labeled(
      std::move(owned_weight), std::move(owned_bias), eps, label);
}

c10::intrusive_ptr<LinearPackedContext> make_linear_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const std::string& label) {
  Tensor owned_weight = weight.clone();
  std::optional<Tensor> owned_bias =
      bias.has_value() ? std::optional<Tensor>(bias->clone()) : std::nullopt;
  return create_linear_context_labeled(
      std::move(owned_weight), std::move(owned_bias), label);
}

c10::intrusive_ptr<LinearPackedContext> make_qkv_context(
    const Tensor& weight,
    const std::string& label) {
  Tensor owned_weight = weight.clone();
  std::optional<Tensor> no_bias = std::nullopt;
  return create_linear_context_labeled(
      std::move(owned_weight), std::move(no_bias), label);
}

c10::intrusive_ptr<Conv2dPackedContext> make_conv2d_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding) {
  Tensor owned_weight = weight.clone();
  std::optional<Tensor> owned_bias =
      bias.has_value() ? std::optional<Tensor>(bias->clone()) : std::nullopt;
  std::vector<int64_t> dilation{1, 1};
  return create_conv2d_context(
      std::move(owned_weight),
      std::move(owned_bias),
      std::move(stride),
      std::move(padding),
      std::move(dilation),
      1);
}

std::vector<int64_t> conv2d_context_output_sizes(
    IntArrayRef input_sizes,
    const c10::intrusive_ptr<Conv2dPackedContext>& context) {
  TORCH_INTERNAL_ASSERT(
      context && input_sizes.size() == 4,
      "Conv2dPackedContext output size computation expects a defined context "
      "and rank-4 input");

  const Tensor weight =
      context->unpack().get(Conv2dPackedContext::Unpacked::Weight).toTensor();
  TORCH_INTERNAL_ASSERT(
      weight.dim() == 4,
      "Conv2dPackedContext output size computation expects rank-4 weight");

  const auto value_or_default =
      [](const std::vector<int64_t>& values,
         const size_t idx,
         const int64_t default_value) -> int64_t {
    return idx < values.size() ? values[idx] : default_value;
  };
  const auto compute_output_extent =
      [&](const int64_t input_extent,
          const int64_t kernel_extent,
          const int64_t stride_extent,
          const int64_t padding_extent,
          const int64_t dilation_extent,
          const int64_t output_padding_extent) -> int64_t {
    if (context->transposed()) {
      return (input_extent - 1) * stride_extent - 2 * padding_extent +
          dilation_extent * (kernel_extent - 1) + output_padding_extent + 1;
    }
    return (input_extent + 2 * padding_extent -
            dilation_extent * (kernel_extent - 1) - 1) /
        stride_extent +
        1;
  };

  const int64_t out_channels = context->transposed()
      ? weight.size(1) * context->groups()
      : weight.size(0);
  const int64_t kernel_h = weight.size(2);
  const int64_t kernel_w = weight.size(3);
  const int64_t out_h = compute_output_extent(
      input_sizes[2],
      kernel_h,
      value_or_default(context->stride(), 0u, 1),
      value_or_default(context->padding(), 0u, 0),
      value_or_default(context->dilation(), 0u, 1),
      value_or_default(context->output_padding(), 0u, 0));
  const int64_t out_w = compute_output_extent(
      input_sizes[3],
      kernel_w,
      value_or_default(context->stride(), 1u, 1),
      value_or_default(context->padding(), 1u, 0),
      value_or_default(context->dilation(), 1u, 1),
      value_or_default(context->output_padding(), 1u, 0));
  return {input_sizes[0], out_channels, out_h, out_w};
}

std::vector<int64_t> conv2d_context_output_sizes(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dPackedContext>& context) {
  TORCH_INTERNAL_ASSERT(
      input.dim() == 4,
      "Conv2dPackedContext output size computation expects rank-4 input");
  return conv2d_context_output_sizes(input.sizes(), context);
}

std::vector<int64_t> tokens_to_feature_map_output_sizes(
    IntArrayRef token_sizes,
    const int64_t patch_h,
    const int64_t patch_w) {
  TORCH_INTERNAL_ASSERT(
      token_sizes.size() == 2 || token_sizes.size() == 3,
      "Token feature-map size computation expects rank-2 or rank-3 tokens");
  const int64_t batch_size = token_sizes.size() == 2 ? 1 : token_sizes[0];
  const int64_t channels = token_sizes[token_sizes.size() - 1u];
  return {batch_size, channels, patch_h, patch_w};
}

std::vector<int64_t> decoder_preprocess_layer_output_sizes(
    IntArrayRef feature_sizes,
    const c10::intrusive_ptr<Conv2dPackedContext>& project_context,
    const c10::intrusive_ptr<Conv2dPackedContext>& resize_context,
    const bool apply_resize,
    const c10::intrusive_ptr<Conv2dPackedContext>& rn_context) {
  std::vector<int64_t> project_sizes =
      conv2d_context_output_sizes(feature_sizes, project_context);
  std::vector<int64_t> resized_sizes = project_sizes;
  if (apply_resize) {
    resized_sizes = conv2d_context_output_sizes(resized_sizes, resize_context);
  }
  return conv2d_context_output_sizes(resized_sizes, rn_context);
}

Tensor run_conv2d_context_any_out(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dPackedContext>& context,
    Tensor& output) {
  return context->transposed() ? run_tconv2d_context_out(input, context, output)
                               : run_conv2d_context_out(input, context, output);
}

Tensor maybe_apply_layerscale(const Tensor& input, const Tensor& gamma) {
  if (!gamma.defined()) {
    return input;
  }
  return at::mul(input, gamma);
}

int64_t vision_block_hidden_dim(
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context) {
  return context->unpack()
      .get(VisionBackboneBlockContext::Unpacked::Fc1Weight)
      .toTensor()
      .size(0);
}

std::string vision_backbone_program_base_label(const std::string& label) {
  if (label.empty()) {
    return "depth.dino.backbone.block";
  }

  constexpr const char* kDynamicBlockMarker = ".block.";
  const auto marker_pos = label.find(kDynamicBlockMarker);
  if (marker_pos != std::string::npos) {
    return label.substr(0, marker_pos + 6u);
  }

  return label;
}

std::string append_context_identity_suffix(
    const std::string& label,
    const void* identity) {
  if (identity == nullptr) {
    return label;
  }
  return label + ".ctx." +
      std::to_string(static_cast<unsigned long long>(
          reinterpret_cast<uintptr_t>(identity)));
}

std::string append_context_cache_id_suffix(
    const std::string& label,
    const uint64_t cache_id) {
  if (cache_id == 0u) {
    return label;
  }
  return label + ".ctx." +
      std::to_string(static_cast<unsigned long long>(cache_id));
}

std::string context_identity_key(const void* identity) {
  if (identity == nullptr) {
    return "null";
  }
  return std::to_string(static_cast<unsigned long long>(
      reinterpret_cast<uintptr_t>(identity)));
}

std::string vision_backbone_context_identity_key(
    const VisionBackboneBlockContext* context) {
  if (context == nullptr || context->cache_id() == 0u) {
    return "null";
  }
  return std::to_string(static_cast<unsigned long long>(context->cache_id()));
}

std::string sizes_key(IntArrayRef sizes) {
  if (sizes.empty()) {
    return "scalar";
  }
  std::ostringstream key;
  for (size_t idx = 0u; idx < sizes.size(); ++idx) {
    if (idx > 0u) {
      key << 'x';
    }
    key << sizes[idx];
  }
  return key.str();
}

std::string optional_tensor_sizes_key(const std::optional<Tensor>& tensor) {
  if (!tensor.has_value() || !tensor->defined()) {
    return "none";
  }
  return sizes_key(tensor->sizes());
}

std::string vision_backbone_program_label(
    const std::string& label,
    const VisionBackboneBlockContext* identity) {
  return append_context_cache_id_suffix(
             vision_backbone_program_base_label(label),
             identity ? identity->cache_id() : 0u) +
      ".program";
}

std::string vision_backbone_execution_label(
    const std::string& label,
    const VisionBackboneBlockContext* identity) {
  return vision_backbone_program_label(label, identity) + ".exec";
}

std::string vision_decoder_program_label(
    const std::string& label,
    const void* identity) {
  if (label.empty()) {
    return append_context_identity_suffix("depth.decoder.fusion", identity) +
        ".program";
  }
  return append_context_identity_suffix(label, identity) + ".program";
}

std::string vision_decoder_program_base_label(const std::string& label) {
  if (label.empty()) {
    return "depth.decoder";
  }

  constexpr const char* kDynamicFusionMarker = ".fusion.";
  const auto marker_pos = label.find(kDynamicFusionMarker);
  if (marker_pos != std::string::npos) {
    return label.substr(0, marker_pos + 7u);
  }

  return label;
}

std::string vision_decoder_head_program_label(
    const std::string& label,
    const void* identity) {
  return append_context_identity_suffix(
             vision_decoder_program_base_label(label), identity) +
      ".head";
}

bool has_explicit_runtime_capture_label() {
  const std::string& runtime_label = api::current_runtime_label();
  return !runtime_label.empty() && runtime_label != "unlabeled";
}

std::string compose_runtime_capture_label(const std::string& base_label) {
  const std::string& runtime_label = api::current_runtime_label();
  if (!runtime_label.empty() && runtime_label != "unlabeled") {
    return runtime_label + "|inner=" + base_label;
  }
  return base_label;
}

std::string current_graph_capture_label(
    const std::string& fallback_base_label,
    const char* default_label) {
  const std::string& runtime_label = api::current_runtime_label();
  if (!runtime_label.empty() && runtime_label != "unlabeled") {
    return runtime_label + ".graph";
  }
  if (!fallback_base_label.empty()) {
    return fallback_base_label + ".graph";
  }
  return std::string(default_label);
}

std::string current_phase_graph_capture_label(
    const std::string& phase_base_label,
    const char* default_label) {
  const std::string& runtime_label = api::current_runtime_label();
  if (!runtime_label.empty() && runtime_label != "unlabeled") {
    if (!phase_base_label.empty()) {
      return runtime_label + "." + phase_base_label + ".graph";
    }
    return runtime_label + ".graph";
  }
  if (!phase_base_label.empty()) {
    return phase_base_label + ".graph";
  }
  return std::string(default_label);
}

VisionReplayBundleIdentity make_vision_backbone_decoder_bundle_identity(
    const c10::intrusive_ptr<VisionBackboneBlockContext>& backbone_context,
    const Tensor& backbone_input,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& decoder_context,
    const Tensor& decoder_input,
    const std::optional<Tensor>& decoder_skip,
    IntArrayRef decoder_target_sizes) {
  std::ostringstream key;
  key << "vision.backbone_decoder"
      << "|backbone_ctx=" << vision_backbone_context_identity_key(backbone_context.get())
      << "|decoder_ctx=" << context_identity_key(decoder_context.get())
      << "|backbone_input=" << sizes_key(backbone_input.sizes())
      << "|decoder_input=" << sizes_key(decoder_input.sizes())
      << "|decoder_skip=" << optional_tensor_sizes_key(decoder_skip)
      << "|decoder_target=" << sizes_key(decoder_target_sizes);

  std::ostringstream suffix;
  suffix << ".bbctx." << vision_backbone_context_identity_key(backbone_context.get())
         << ".decctx." << context_identity_key(decoder_context.get())
         << ".bin." << sizes_key(backbone_input.sizes())
         << ".din." << sizes_key(decoder_input.sizes())
         << ".dskip." << optional_tensor_sizes_key(decoder_skip)
         << ".dtarget." << sizes_key(decoder_target_sizes);
  return VisionReplayBundleIdentity{key.str(), suffix.str()};
}

VisionReplayBundleIdentity make_vision_backbone_stack_bundle_identity(
    const std::vector<c10::intrusive_ptr<VisionBackboneBlockContext>>& contexts,
    const std::vector<int64_t>& capture_indices,
    const std::optional<std::vector<int64_t>>& normalized_shape = std::nullopt,
    const LayernormPackedContext* norm_context = nullptr) {
  std::string key =
      "vision.backbone_stack|count=" + std::to_string(contexts.size()) +
      "|capture=";
  for (size_t idx = 0u; idx < capture_indices.size(); ++idx) {
    if (idx > 0u) {
      key += ",";
    }
    key += std::to_string(capture_indices[idx]);
  }
  key += "|contexts=";
  for (size_t idx = 0u; idx < contexts.size(); ++idx) {
    if (idx > 0u) {
      key += ",";
    }
    key += vision_backbone_context_identity_key(contexts[idx].get());
  }
  if (norm_context != nullptr && normalized_shape.has_value()) {
    key += "|norm_ctx=";
    key += context_identity_key(norm_context);
    key += "|norm_shape=";
    key += sizes_key(*normalized_shape);
  } else {
    key += "|norm=none";
  }

  std::string suffix = ".count." + std::to_string(contexts.size()) + ".capture.";
  for (size_t idx = 0u; idx < capture_indices.size(); ++idx) {
    if (idx > 0u) {
      suffix += "x";
    }
    suffix += std::to_string(capture_indices[idx]);
  }
  suffix += ".contexts.";
  for (size_t idx = 0u; idx < contexts.size(); ++idx) {
    if (idx > 0u) {
      suffix += ".";
    }
    suffix += vision_backbone_context_identity_key(contexts[idx].get());
  }
  if (norm_context != nullptr && normalized_shape.has_value()) {
    suffix += ".normctx.";
    suffix += context_identity_key(norm_context);
    suffix += ".normshape.";
    suffix += sizes_key(*normalized_shape);
  }

  return VisionReplayBundleIdentity{std::move(key), std::move(suffix)};
}

VisionReplayBundleIdentity make_vision_decoder_preprocess_head_bundle_identity(
    const std::array<Tensor, 4u>& layer_tokens,
    const int64_t patch_h,
    const int64_t patch_w,
    IntArrayRef output_size,
    const VisionDecoderPreprocessHeadContext* context) {
  std::ostringstream key;
  key << "vision.decoder_preprocess_head"
      << "|ctx=" << context_identity_key(context)
      << "|patch=" << patch_h << "x" << patch_w
      << "|output=" << sizes_key(output_size);
  for (size_t idx = 0u; idx < layer_tokens.size(); ++idx) {
    key << "|layer" << (idx + 1u) << "=" << sizes_key(layer_tokens[idx].sizes());
  }

  std::ostringstream suffix;
  suffix << ".dphctx." << context_identity_key(context)
         << ".patch." << patch_h << "x" << patch_w
         << ".out." << sizes_key(output_size);
  for (size_t idx = 0u; idx < layer_tokens.size(); ++idx) {
    suffix << ".l" << (idx + 1u) << "." << sizes_key(layer_tokens[idx].sizes());
  }
  return VisionReplayBundleIdentity{key.str(), suffix.str()};
}

std::string vision_backbone_graph_label(
    const std::string& label,
    const VisionBackboneBlockContext* identity) {
  return current_phase_graph_capture_label(
      append_context_cache_id_suffix(
          vision_backbone_program_base_label(label),
          identity ? identity->cache_id() : 0u),
      "depth.dino.backbone.graph");
}

std::string vision_decoder_graph_label(const std::string& label) {
  return current_phase_graph_capture_label(
      vision_decoder_program_base_label(label),
      "depth.decoder.graph");
}

std::vector<int64_t> calc_contiguous_strides(IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (int64_t idx = static_cast<int64_t>(sizes.size()) - 2; idx >= 0; --idx) {
    strides[idx] = strides[idx + 1] * std::max<int64_t>(sizes[idx + 1], 1);
  }
  return strides;
}

std::vector<int64_t> calc_width_packed_buffer_sizes(IntArrayRef sizes) {
  std::vector<int64_t> physical_sizes(sizes.begin(), sizes.end());
  if (!physical_sizes.empty()) {
    physical_sizes.back() =
        api::utils::align_up(physical_sizes.back(), INT64_C(4));
  }
  return physical_sizes;
}

size_t buffer_descriptor_nbytes(IntArrayRef sizes, const ScalarType dtype) {
  return static_cast<size_t>(
      api::element_size(convert_dtype(dtype)) *
      api::utils::multiply_integers(calc_width_packed_buffer_sizes(sizes)));
}

std::vector<int64_t> calc_width_packed_buffer_strides(IntArrayRef sizes) {
  return calc_contiguous_strides(calc_width_packed_buffer_sizes(sizes));
}

size_t align_up_size(const size_t value, const size_t alignment) {
  if (alignment <= 1u) {
    return value;
  }
  const size_t remainder = value % alignment;
  return remainder == 0u ? value : (value + alignment - remainder);
}

size_t vision_attention_scratch_bytes(
    const int64_t batch_size,
    const int64_t token_count,
    const int64_t embed_dim,
    const int64_t num_heads,
    const ScalarType dtype,
    const bool has_qkv_bias,
    const uint32_t alignment) {
  if (
      dtype != kFloat || num_heads <= 0 || embed_dim <= 0 || token_count <= 0 ||
      embed_dim % num_heads != 0 || batch_size <= 0) {
    return 0u;
  }

  const int64_t head_dim = embed_dim / num_heads;
  const int64_t batch_heads = batch_size * num_heads;

  size_t total_bytes = 0u;
  const auto append_slice = [&](const size_t slice_bytes) {
    total_bytes = align_up_size(total_bytes, alignment);
    total_bytes += slice_bytes;
  };

  if (batch_size == 1 && dtype == kFloat && has_qkv_bias) {
    const size_t mixed_qkv_bytes =
        buffer_descriptor_nbytes({token_count, 3 * embed_dim}, dtype);
    const size_t qkv_projection_bytes =
        buffer_descriptor_nbytes({num_heads, token_count, head_dim}, dtype);
    append_slice(mixed_qkv_bytes);
    append_slice(qkv_projection_bytes);
    append_slice(qkv_projection_bytes);
    append_slice(qkv_projection_bytes);
  }

  if (dtype == kFloat) {
    const size_t attention_scores_bytes = buffer_descriptor_nbytes(
        {batch_heads, token_count, token_count}, dtype);
    const size_t attention_context_bytes = buffer_descriptor_nbytes(
        {batch_heads, token_count, head_dim}, dtype);
    const size_t merge_output_bytes = buffer_descriptor_nbytes(
        {batch_size * token_count, embed_dim}, dtype);
    append_slice(attention_scores_bytes);
    append_slice(attention_scores_bytes);
    append_slice(attention_context_bytes);
    append_slice(merge_output_bytes);
  }

  return total_bytes;
}

std::vector<int64_t> resolve_decoder_target_sizes(
    const Tensor& input,
    const std::optional<std::vector<int64_t>>& size) {
  if (size.has_value()) {
    TORCH_CHECK(
        size->size() == 2u,
        "Vision decoder fusion block expects size=[height, width]");
    return {size->at(0), size->at(1)};
  }
  TORCH_CHECK(
      input.dim() == 4,
      "Vision decoder fusion block expects rank-4 input for scale_factor=2");
  return {input.size(2) * 2, input.size(3) * 2};
}

size_t vision_decoder_fusion_block_scratch_bytes(
    const Tensor& input,
    const std::optional<Tensor>& skip,
    const std::vector<int64_t>& target_sizes) {
  if (
      !input.defined() || input.scalar_type() != kFloat || input.dim() != 4 ||
      target_sizes.size() != 2u) {
    return 0u;
  }

  size_t total_bytes = 0u;
  const auto append_slice = [&](IntArrayRef sizes) {
    total_bytes = align_up_size(total_bytes, 256u);
    total_bytes += buffer_descriptor_nbytes(sizes, kFloat);
  };

  if (skip.has_value() && skip->defined()) {
    append_slice(skip->sizes());
    append_slice(skip->sizes());
    append_slice(skip->sizes());
    append_slice(skip->sizes());
    append_slice(input.sizes());
  }

  append_slice(input.sizes());
  append_slice(input.sizes());
  append_slice(input.sizes());
  append_slice(input.sizes());
  append_slice(
      {input.size(0), input.size(1), target_sizes[0], target_sizes[1]});
  return total_bytes;
}

Tensor make_scratch_buffer_alias(
    const utils::ScratchArena& arena,
    const utils::VulkanScratchSlice& slice,
    IntArrayRef sizes,
    const ScalarType dtype) {
  const size_t required_bytes = buffer_descriptor_nbytes(sizes, dtype);
  TORCH_CHECK(
      required_bytes <= slice.size_bytes,
      "Scratch buffer alias requested ",
      required_bytes,
      " bytes from a slice sized for ",
      slice.size_bytes,
      " bytes");

  const int64_t element_size =
      static_cast<int64_t>(c10::elementSize(dtype));
  TORCH_CHECK(
      element_size > 0,
      "Scratch buffer alias requires a concrete element size");
  TORCH_CHECK(
      slice.offset_bytes % static_cast<size_t>(element_size) == 0u &&
          arena.size_bytes() % static_cast<size_t>(element_size) == 0u,
      "Scratch buffer alias requires byte-aligned offsets for dtype ",
      dtype);

  const int64_t storage_offset =
      static_cast<int64_t>(slice.offset_bytes / static_cast<size_t>(element_size));
  const int64_t buffer_length_override =
      static_cast<int64_t>(arena.size_bytes() / static_cast<size_t>(element_size));
  const api::ExecutionLayout execution_layout =
      slice.offset_bytes == 0u ? api::ExecutionLayout::BUFFER_DIRECT
                               : api::ExecutionLayout::BUFFER_VIEW;
  return make_typed_buffer_metadata_view_checked(
      arena.storage(),
      dtype,
      sizes,
      calc_contiguous_strides(sizes),
      calc_width_packed_buffer_strides(sizes),
      storage_offset,
      buffer_length_override,
      execution_layout,
      "vulkan_prepack::scratch_tensor");
}

std::pair<utils::VulkanScratchSlice, Tensor> reserve_scratch_buffer_tensor(
    utils::ScratchArena& arena,
    IntArrayRef sizes,
    const ScalarType dtype) {
  const size_t required_bytes = buffer_descriptor_nbytes(sizes, dtype);
  const utils::VulkanScratchSlice slice = arena.reserve(
      required_bytes,
      std::max<uint32_t>(
          arena.alignment(),
          static_cast<uint32_t>(std::max<int64_t>(
              1, static_cast<int64_t>(c10::elementSize(dtype))))));
  return {slice, make_scratch_buffer_alias(arena, slice, sizes, dtype)};
}

Tensor prepare_buffer_attention_tensor(const Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_vulkan(),
      "Vision attention workspace expects Vulkan tensors");
  const vTensor& v_tensor = convert(tensor);
  if (
      v_tensor.storage_type() == api::StorageType::BUFFER &&
      v_tensor.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
      utils::supports_buffer_view_fast_path(v_tensor)) {
    return utils::mark_tensor_execution(
        tensor, utils::resolve_buffer_execution_layout(v_tensor));
  }

  Tensor buffer_tensor = utils::ensure_buffer_storage(
      tensor, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  return utils::mark_tensor_execution(
      buffer_tensor,
      utils::resolve_buffer_execution_layout(convert(buffer_tensor)));
}

Tensor prepare_decoder_buffer_tensor(const Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_vulkan(),
      "Vision decoder fusion block expects Vulkan tensors");
  const vTensor& v_tensor = convert(tensor);
  if (
      v_tensor.storage_type() == api::StorageType::BUFFER &&
      v_tensor.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
      utils::supports_buffer_view_fast_path(v_tensor)) {
    return utils::mark_tensor_execution(
        tensor, utils::resolve_buffer_execution_layout(v_tensor), false);
  }

  Tensor buffer_tensor = utils::ensure_buffer_storage(
      tensor, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  return utils::mark_tensor_execution(
      buffer_tensor,
      utils::resolve_buffer_execution_layout(convert(buffer_tensor)),
      false);
}

int64_t vision_decoder_out_channels(
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& context) {
  const auto& logical_weight_sizes =
      context->out_conv_context()->packed_weight().logical_weight_sizes();
  TORCH_CHECK(
      logical_weight_sizes.size() == 4u,
      "Vision decoder fusion block expects rank-4 out_conv weights");
  return logical_weight_sizes[0];
}

bool all_values_are(const std::vector<int64_t>& values, const int64_t target) {
  return std::all_of(
      values.begin(), values.end(), [target](const int64_t value) {
        return value == target;
      });
}

bool can_run_decoder_out_conv_before_upsample(
    const Tensor& input,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& context,
    const Tensor& conv_output) {
  const auto& out_conv = context->out_conv_context();
  const auto& weight_sizes = out_conv->packed_weight().logical_weight_sizes();
  if (
      !input.defined() || !conv_output.defined() || !input.is_vulkan() ||
      !conv_output.is_vulkan() || input.dim() != 4 || conv_output.dim() != 4 ||
      weight_sizes.size() != 4u || weight_sizes[2] != 1 ||
      weight_sizes[3] != 1 || out_conv->transposed() ||
      out_conv->quantized() || out_conv->groups() != 1 ||
      !all_values_are(out_conv->stride(), 1) ||
      !all_values_are(out_conv->padding(), 0) ||
      !all_values_are(out_conv->dilation(), 1) ||
      !all_values_are(out_conv->output_padding(), 0) ||
      !std::isinf(out_conv->output_min()) ||
      !std::isinf(out_conv->output_max()) ||
      out_conv->output_min() > 0.0f || out_conv->output_max() < 0.0f) {
    return false;
  }

  const std::vector<int64_t> conv_output_sizes{
      input.size(0), weight_sizes[0], input.size(2), input.size(3)};
  return conv_output.sizes().vec() == conv_output_sizes;
}

constexpr int64_t kDaV2HeadOutputConv1Channels = 32;
constexpr int64_t kDaV2HeadHiddenChannels = 32;
constexpr int64_t kDaV2HeadFinalChannels = 1;
constexpr int64_t kDaV2HeadUpsampleNumerator = 7;
constexpr int64_t kDaV2HeadUpsampleDenominator = 4;
constexpr uint32_t kDaV2HeadWorkGroupSizeX = 16u;
constexpr uint32_t kDaV2HeadWorkGroupSizeY = 16u;
constexpr uint32_t kDaV2HeadWorkGroupSizeZ = 1u;
constexpr uint32_t kDaV2HeadOutputsPerThreadX = 2u;
constexpr uint32_t kDaV2HeadOutputsPerThreadY = 1u;

bool has_identity_output_range(
    const c10::intrusive_ptr<Conv2dPackedContext>& context) {
  return context && std::isinf(context->output_min()) &&
      std::isinf(context->output_max()) && context->output_min() < 0.0f &&
      context->output_max() > 0.0f;
}

bool is_plain_float_conv2d(
    const c10::intrusive_ptr<Conv2dPackedContext>& context,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t padding_h,
    const int64_t padding_w) {
  if (
      !context || context->transposed() || context->quantized() ||
      context->groups() != 1 || !all_values_are(context->stride(), 1) ||
      !all_values_are(context->padding(), padding_h) ||
      !all_values_are(context->dilation(), 1) ||
      !all_values_are(context->output_padding(), 0) ||
      !has_identity_output_range(context)) {
    return false;
  }

  const auto& weight_sizes = context->packed_weight().logical_weight_sizes();
  return weight_sizes.size() == 4u && weight_sizes[2] == kernel_h &&
      weight_sizes[3] == kernel_w &&
      context->padding().size() >= 2u && context->padding()[0] == padding_h &&
      context->padding()[1] == padding_w;
}

bool can_run_depth_anything_v2_head_fusion_shape(
    IntArrayRef path1_sizes,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderHeadContext>& context) {
  // The fused DA-v2 head is not numerically aligned with the reference DPT
  // head across the supported decoder shapes. Keep the generic Vulkan
  // conv/upsample path as the main path until the shader has a correctness
  // matrix behind it.
  (void)path1_sizes;
  (void)output_size;
  (void)context;
  return false;

  if (!context || path1_sizes.size() != 4 || output_size.size() != 2) {
    return false;
  }

  const auto& output_conv1 = context->output_conv1_context();
  const auto& output_conv2_conv1 = context->output_conv2_conv1_context();
  const auto& output_conv2_conv2 = context->output_conv2_conv2_context();
  if (
      !is_plain_float_conv2d(output_conv1, 3, 3, 1, 1) ||
      !is_plain_float_conv2d(output_conv2_conv1, 3, 3, 1, 1) ||
      !is_plain_float_conv2d(output_conv2_conv2, 1, 1, 0, 0)) {
    return false;
  }

  const auto& conv1_weight_sizes =
      output_conv1->packed_weight().logical_weight_sizes();
  const auto& conv2_weight_sizes =
      output_conv2_conv1->packed_weight().logical_weight_sizes();
  const auto& conv3_weight_sizes =
      output_conv2_conv2->packed_weight().logical_weight_sizes();
  return conv1_weight_sizes[0] == kDaV2HeadOutputConv1Channels &&
      conv2_weight_sizes[0] == kDaV2HeadHiddenChannels &&
      conv2_weight_sizes[2] == 3 && conv2_weight_sizes[3] == 3 &&
      conv2_weight_sizes[1] == conv1_weight_sizes[0] &&
      conv3_weight_sizes[0] == kDaV2HeadFinalChannels &&
      conv3_weight_sizes[1] == kDaV2HeadHiddenChannels &&
      conv3_weight_sizes[2] == 1 && conv3_weight_sizes[3] == 1 &&
      output_size[0] * kDaV2HeadUpsampleDenominator ==
          path1_sizes[2] * kDaV2HeadUpsampleNumerator &&
      output_size[1] * kDaV2HeadUpsampleDenominator ==
          path1_sizes[3] * kDaV2HeadUpsampleNumerator;
}

bool can_run_depth_anything_v2_head_fusion(
    const Tensor& path1,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderHeadContext>& context) {
  if (!path1.defined() || !path1.is_vulkan()) {
    return false;
  }
  const vTensor& v_path1 = convert(path1);
  if (
      path1.dim() != 4 || v_path1.storage_type() != api::StorageType::BUFFER ||
      !v_path1.has_direct_buffer_layout()) {
    return false;
  }
  return can_run_depth_anything_v2_head_fusion_shape(
      path1.sizes(), output_size, context);
}

Tensor prepare_depth_anything_v2_head_output(
    Tensor output,
    IntArrayRef expected_sizes) {
  output = output.is_vulkan() ? output : output.vulkan();
  output = utils::mark_tensor_execution(
      output, utils::resolve_buffer_execution_layout(convert(output)), false);
  const vTensor& v_output = convert(output);
  TORCH_CHECK(
      v_output.storage_type() == api::StorageType::BUFFER &&
          v_output.dtype() == api::kFloat &&
          utils::supports_buffer_view_fast_path(v_output),
      "Depth Anything v2 fused head expects float buffer-backed output");
  TORCH_CHECK(
      output.sizes().equals(expected_sizes),
      "Depth Anything v2 fused head received mismatched output shape");
  return output;
}

Tensor run_depth_anything_v2_head_fusion_out(
    const Tensor& path1,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderHeadContext>& context,
    Tensor* output_opt = nullptr) {
  TORCH_INTERNAL_ASSERT(
      can_run_depth_anything_v2_head_fusion(path1, output_size, context),
      "Depth Anything v2 fused head expects DA v2-compatible buffer inputs");
  api::AllocationScope allocation_scope("vision_decoder_head.da_v2");
  utils::log_vulkan_op_hit("aten::vision_decoder_head.da_v2_fused_head");

  const auto& output_conv1 = context->output_conv1_context();
  const auto& output_conv2_conv1 = context->output_conv2_conv1_context();
  const auto& output_conv2_conv2 = context->output_conv2_conv2_context();
  const std::vector<int64_t> expected_output_sizes{
      path1.size(0),
      kDaV2HeadFinalChannels,
      output_size[0],
      output_size[1],
  };

  Tensor output = output_opt != nullptr
      ? prepare_depth_anything_v2_head_output(*output_opt, expected_output_sizes)
      : utils::create_buffer_tensor(expected_output_sizes, kFloat);

  api::Context* const context_vk = api::context();
  vTensor v_output = convert(output);
  vTensor v_input = convert(path1);
  vTensor v_weight1 = output_conv1->packed_weight().weight_vtensor();
  vTensor v_bias1 = output_conv1->packed_weight().bias_vtensor();
  vTensor v_weight2 = output_conv2_conv1->packed_weight().weight_vtensor();
  vTensor v_bias2 = output_conv2_conv1->packed_weight().bias_vtensor();
  vTensor v_weight3 = output_conv2_conv2->packed_weight().weight_vtensor();
  vTensor v_bias3 = output_conv2_conv2->packed_weight().bias_vtensor();

  const struct {
    int32_t align_corners;
    int32_t has_bias1;
    int32_t has_bias2;
    int32_t has_bias3;
  } block{
      context->align_corners() ? 1 : 0,
      output_conv1->packed_weight().has_bias() ? 1 : 0,
      output_conv2_conv1->packed_weight().has_bias() ? 1 : 0,
      output_conv2_conv2->packed_weight().has_bias() ? 1 : 0,
  };

  api::UniformParamsBuffer params(context_vk, block);
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context_vk, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context_vk, v_input);
  api::UniformParamsBuffer weight1_meta =
      utils::make_buffer_compute_metadata_ubo(context_vk, v_weight1);
  api::UniformParamsBuffer bias1_meta =
      utils::make_buffer_compute_metadata_ubo(context_vk, v_bias1);
  api::UniformParamsBuffer weight2_meta =
      utils::make_buffer_compute_metadata_ubo(context_vk, v_weight2);
  api::UniformParamsBuffer bias2_meta =
      utils::make_buffer_compute_metadata_ubo(context_vk, v_bias2);
  api::UniformParamsBuffer weight3_meta =
      utils::make_buffer_compute_metadata_ubo(context_vk, v_weight3);
  api::UniformParamsBuffer bias3_meta =
      utils::make_buffer_compute_metadata_ubo(context_vk, v_bias3);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(
          (output_size[1] + kDaV2HeadOutputsPerThreadX - 1) /
          kDaV2HeadOutputsPerThreadX),
      api::utils::safe_downcast<uint32_t>(
          (output_size[0] + kDaV2HeadOutputsPerThreadY - 1) /
          kDaV2HeadOutputsPerThreadY),
      api::utils::safe_downcast<uint32_t>(path1.size(0) * kDaV2HeadFinalChannels),
  };
  const api::utils::uvec3 local_size{
      kDaV2HeadWorkGroupSizeX,
      kDaV2HeadWorkGroupSizeY,
      kDaV2HeadWorkGroupSizeZ,
  };

  context_vk->submit_compute_job(
      VK_KERNEL(depth_anything_v2_head_buffer_float),
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer(),
      v_weight1.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      weight1_meta.buffer(),
      v_bias1.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      bias1_meta.buffer(),
      v_weight2.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      weight2_meta.buffer(),
      v_bias2.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      bias2_meta.buffer(),
      v_weight3.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      weight3_meta.buffer(),
      v_bias3.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      bias3_meta.buffer(),
      params.buffer());
  return output;
}

Tensor run_vision_decoder_head_tail_context(
    const Tensor& path1,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderHeadContext>& context,
    Tensor* output_opt = nullptr) {
  if (can_run_depth_anything_v2_head_fusion(path1, output_size, context)) {
    return run_depth_anything_v2_head_fusion_out(
        path1, output_size, context, output_opt);
  }

  Tensor output = run_conv2d_context(path1, context->output_conv1_context());
  output = at::upsample_bilinear2d(
      output,
      output_size.vec(),
      context->align_corners(),
      std::nullopt,
      std::nullopt);
  output = run_conv2d_context(output, context->output_conv2_conv1_context());
  output = at::relu(output);
  output = output_opt != nullptr
      ? run_conv2d_context_out(
            output, context->output_conv2_conv2_context(), *output_opt)
      : run_conv2d_context(output, context->output_conv2_conv2_context());
  return at::relu_(output);
}

struct VisionDecoderRunOutputs final {
  Tensor skip_relu_output;
  Tensor skip_conv1_output;
  Tensor skip_conv2_output;
  Tensor skip_res_output;
  Tensor main_input_output;
  Tensor main_relu_output;
  Tensor main_conv1_output;
  Tensor main_conv2_output;
  Tensor main_res_output;
  Tensor upsample_output;
  Tensor out_conv_output;
};

utils::VisionDecoderInferenceGraph prime_vision_decoder_graph(
    const Tensor& input,
    const utils::VulkanRuntimePolicy& runtime_policy,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& context) {
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionDecoder) {
    return {};
  }

  return utils::lookup_or_create_labeled_vision_decoder_inference_graph(
      vision_decoder_graph_label(context->allocation_label()),
      input.scalar_type(),
      runtime_policy.execution_program_plan->persistent);
}

utils::VisionBackboneInferenceGraph prime_vision_backbone_graph(
    const Tensor& input,
    const utils::VulkanRuntimePolicy& runtime_policy,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context) {
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionBackbone) {
    return {};
  }

  return utils::lookup_or_create_labeled_vision_backbone_inference_graph(
      vision_backbone_graph_label(context->allocation_label(), context.get()),
      input.scalar_type(),
      runtime_policy.execution_program_plan->persistent);
}

utils::VisionDecoderProgram prime_vision_decoder_program(
    const Tensor& input,
    const std::optional<Tensor>& skip,
    IntArrayRef target_sizes,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& context,
    const utils::VulkanRuntimePolicy& runtime_policy,
    const bool use_external_scratch,
    const bool allocate_intermediate_outputs = true) {
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionDecoder) {
    return {};
  }

  const std::optional<utils::VulkanScratchArenaSpec> scratch_spec =
      !use_external_scratch && runtime_policy.scratch_arena_plan.has_value()
          ? [&]() -> std::optional<utils::VulkanScratchArenaSpec> {
              const auto requested_bytes = vision_decoder_fusion_block_scratch_bytes(
                  input, skip, target_sizes.vec());
              if (
                  requested_bytes == 0u ||
                  !runtime_policy.scratch_arena_plan->prefer_buffer_storage) {
                return std::nullopt;
              }
              return utils::VulkanScratchArenaSpec{
                  kByte,
                  std::max(
                      requested_bytes,
                      runtime_policy.scratch_arena_plan->min_arena_bytes),
                  runtime_policy.scratch_arena_plan->alignment,
                  api::ExecutionLayout::BUFFER_DIRECT,
                  api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
                  api::StorageType::BUFFER,
                  runtime_policy.scratch_arena_plan->prefer_reusable_arena,
              };
            }()
          : std::nullopt;

  return utils::lookup_or_create_labeled_vision_decoder_program(
      vision_decoder_program_label(context->allocation_label(), context.get()),
      input.sizes(),
      skip.has_value() ? std::optional<std::vector<int64_t>>(skip->sizes().vec())
                       : std::nullopt,
      target_sizes,
      vision_decoder_out_channels(context),
      scratch_spec,
      *runtime_policy.execution_program_plan,
      allocate_intermediate_outputs);
}

Tensor run_attention_with_workspace_fallback(
    const Tensor& query_arg,
    const Tensor& key_arg,
    const Tensor& value_arg,
    const std::optional<Tensor>& attn_bias_arg,
    utils::VisionBackboneProgram* const vision_program,
    utils::ScratchArena* const scratch_override = nullptr) {
  const auto fallback = [&](const Tensor& query,
                            const Tensor& key,
                            const Tensor& value,
                            const std::optional<Tensor>& attn_bias) -> Tensor {
    return at::scaled_dot_product_attention(
        query,
        key,
        value,
        attn_bias,
        0.0,
        false,
        std::optional<double>(1.0),
        false);
  };

  const bool has_attention_bias =
      attn_bias_arg.has_value() && attn_bias_arg->defined();

  const auto attention_policy = utils::build_vulkan_attention_policy(
      std::nullopt,
      /*is_causal=*/false,
      /*enable_gqa=*/false,
      /*use_kv_cache=*/false,
      /*cache_has_previous_state=*/false);
  const auto attention_runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_attention_request(
          attention_policy,
          query_arg,
          key_arg,
          value_arg,
          utils::VulkanTensorRole::Input));
  if (
      attention_runtime_policy.attention_execution_strategy ==
          utils::VulkanAttentionExecutionStrategy::RuntimeProgram &&
      attention_runtime_policy.execution_program_plan.has_value() &&
      attention_runtime_policy.execution_program_plan->kind ==
          utils::VulkanExecutionProgramKind::AttentionRuntime &&
      !has_attention_bias) {
    if (api::current_vision_stack_phase() ==
        api::VulkanVisionStackPhase::Attention) {
      auto& counters = vulkan_stack_attention_counters();
      counters.total.fetch_add(1u, std::memory_order_relaxed);
      if (query_arg.scalar_type() != kFloat ||
          key_arg.scalar_type() != kFloat ||
          value_arg.scalar_type() != kFloat) {
        counters.reject_dtype.fetch_add(1u, std::memory_order_relaxed);
      } else if (query_arg.dim() != 3 || key_arg.dim() != 3 ||
          value_arg.dim() != 3 ||
          query_arg.size(0) != key_arg.size(0) ||
          query_arg.size(0) != value_arg.size(0) ||
          query_arg.size(2) != 64 || key_arg.size(2) != 64 ||
          value_arg.size(2) != 64 || key_arg.size(1) != value_arg.size(1)) {
        counters.reject_shape.fetch_add(1u, std::memory_order_relaxed);
      } else {
        Tensor query = prepare_buffer_attention_tensor(query_arg);
        Tensor key = prepare_buffer_attention_tensor(key_arg);
        Tensor value = prepare_buffer_attention_tensor(value_arg);
        const vTensor& v_query = convert(query);
        const vTensor& v_key = convert(key);
        const vTensor& v_value = convert(value);
        if (v_query.storage_type() == api::StorageType::BUFFER &&
            v_key.storage_type() == api::StorageType::BUFFER &&
            v_value.storage_type() == api::StorageType::BUFFER &&
            utils::supports_buffer_view_fast_path(v_query) &&
            utils::supports_buffer_view_fast_path(v_key) &&
            utils::supports_buffer_view_fast_path(v_value)) {
          counters.direct_hit.fetch_add(1u, std::memory_order_relaxed);
          counters.decomposed_placeholder_bypass.fetch_add(
              1u,
              std::memory_order_relaxed);
          utils::log_vulkan_op_hit(
              "vulkan_prepack::vision_stack_attention_direct");
          return run_attention_runtime_buffer_math_program_bridge(
              query, key, value);
        }
        counters.reject_layout.fetch_add(1u, std::memory_order_relaxed);
      }
    }
    utils::log_vulkan_op_hit(
        "aten::vision_attention.runtime_program_dispatch");
    Tensor key_t = prepare_buffer_attention_tensor(key_arg.transpose(1, 2));
    Tensor scores = at::bmm(query_arg, key_t);
    Tensor probs = at::softmax(scores, -1);
    return at::bmm(probs, value_arg);
  }
  utils::ScratchArena* scratch_arena = scratch_override;
  if (
      !scratch_arena && vision_program && vision_program->defined() &&
      vision_program->scratch_arena().has_value()) {
    scratch_arena = &(*vision_program->scratch_arena());
  }
  if (!scratch_arena && !has_attention_bias) {
    return fallback(query_arg, key_arg, value_arg, attn_bias_arg);
  }

  Tensor query = prepare_buffer_attention_tensor(query_arg);
  Tensor key = prepare_buffer_attention_tensor(key_arg);
  Tensor value = prepare_buffer_attention_tensor(value_arg);
  if (
      query.scalar_type() != kFloat || key.scalar_type() != kFloat ||
      value.scalar_type() != kFloat || query.dim() != 3 || key.dim() != 3 ||
      value.dim() != 3 || query.size(0) != key.size(0) ||
      query.size(0) != value.size(0) || query.size(2) != key.size(2) ||
      key.size(1) != value.size(1)) {
    return fallback(query, key, value, attn_bias_arg);
  }

  const vTensor& v_query = convert(query);
  const vTensor& v_key = convert(key);
  const vTensor& v_value = convert(value);
  if (
      v_query.storage_type() != api::StorageType::BUFFER ||
      v_key.storage_type() != api::StorageType::BUFFER ||
      v_value.storage_type() != api::StorageType::BUFFER ||
      !utils::supports_buffer_view_fast_path(v_query) ||
      !utils::supports_buffer_view_fast_path(v_key) ||
      !utils::supports_buffer_view_fast_path(v_value)) {
    return fallback(query, key, value, attn_bias_arg);
  }

  std::optional<Tensor> attention_bias;
  if (has_attention_bias) {
    Tensor bias = attn_bias_arg->is_vulkan() ? *attn_bias_arg : attn_bias_arg->vulkan();
    if (bias.scalar_type() != kFloat) {
      bias = bias.to(kFloat);
    }
    bias = prepare_buffer_attention_tensor(bias);
    TORCH_CHECK(
        bias.dim() == 3,
        "Vision attention workspace expects attention bias with rank-3 "
        "[B*H, T, T] shape");
    TORCH_CHECK(
        bias.size(0) == query.size(0) && bias.size(1) == query.size(1) &&
            bias.size(2) == key.size(1),
        "Vision attention workspace received unexpected attention bias sizes");

    const vTensor& v_bias = convert(bias);
    if (
        v_bias.storage_type() != api::StorageType::BUFFER ||
        !utils::supports_buffer_view_fast_path(v_bias)) {
      return fallback(query, key, value, attn_bias_arg);
    }
    attention_bias = std::move(bias);
  }

  if (attention_bias.has_value()) {
    utils::log_vulkan_op_hit(
        "aten::vision_attention.attention_bias_composed_vulkan");
    const auto materialize_attention_operand = [](const Tensor& operand) {
      Tensor materialized =
          utils::create_buffer_tensor(operand.sizes(), operand.scalar_type());
      materialized.copy_(operand);
      record_tensor_write(
          materialized,
          "aten::vision_attention",
          "attention_bias_operand_materialize",
          {operand});
      return materialized;
    };
    Tensor materialized_query = materialize_attention_operand(query);
    Tensor materialized_key = materialize_attention_operand(key);
    Tensor materialized_value = materialize_attention_operand(value);
    Tensor materialized_bias = materialize_attention_operand(*attention_bias);
    Tensor key_t =
        prepare_buffer_attention_tensor(materialized_key.transpose(1, 2));
    Tensor scores = at::bmm(materialized_query, key_t);
    Tensor biased_scores = at::add(scores, materialized_bias);
    Tensor probs = at::softmax(biased_scores, -1);
    return at::bmm(probs, materialized_value);
  }

  const std::vector<int64_t> scores_sizes{
      query.size(0),
      query.size(1),
      key.size(1),
  };
  const std::vector<int64_t> output_sizes{
      query.size(0),
      query.size(1),
      value.size(2),
  };
  Tensor scores_output;
  Tensor probs_output;
  Tensor context_output;
  if (has_attention_bias) {
    utils::log_vulkan_op_hit(
        "aten::vision_attention.attention_bias_materialized_workspace");
    scores_output = utils::create_buffer_tensor(scores_sizes, kFloat);
    probs_output = utils::create_buffer_tensor(scores_sizes, kFloat);
    context_output = utils::create_buffer_tensor(output_sizes, kFloat);
  } else {
    auto [scores_slice, scratch_scores_output] =
        reserve_scratch_buffer_tensor(*scratch_arena, scores_sizes, kFloat);
    auto [probs_slice, scratch_probs_output] =
        reserve_scratch_buffer_tensor(*scratch_arena, scores_sizes, kFloat);
    auto [context_slice, scratch_context_output] =
        reserve_scratch_buffer_tensor(*scratch_arena, output_sizes, kFloat);
    (void)scores_slice;
    (void)probs_slice;
    (void)context_slice;
    scores_output = std::move(scratch_scores_output);
    probs_output = std::move(scratch_probs_output);
    context_output = std::move(scratch_context_output);
  }

  Tensor key_t = prepare_buffer_attention_tensor(key.transpose(1, 2));
  Tensor scores = bmm_buffer_out_vulkan(query, key_t, scores_output);
  Tensor probs_input = scores;
  Tensor probs_output_tensor = probs_output;
  if (attention_bias.has_value()) {
    probs_input = add_buffer_out_vulkan(scores, *attention_bias, probs_output);
    probs_output_tensor = scores_output;
  }
  Tensor probs;
  if (
      probs_input.dim() == 3 &&
      probs_input.size(probs_input.dim() - 1) >= 64) {
    utils::log_vulkan_op_hit(
        "aten::vision_backbone_attention.softmax_texture_materialize");
    probs_output_tensor.copy_(at::softmax(probs_input, -1));
    probs = probs_output_tensor;
  } else {
    probs =
        softmax_buffer_lastdim_out_vulkan(probs_input, probs_output_tensor);
  }
  return bmm_buffer_out_vulkan(probs, value, context_output);
}

utils::VisionBackboneProgram prime_vision_backbone_program(
    const Tensor& input,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context,
    const utils::VulkanRuntimePolicy& runtime_policy,
    const bool use_external_scratch) {
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionBackbone) {
    return {};
  }

  const int64_t batch_size = input.dim() == 2 ? 1 : input.size(0);
  const int64_t token_count = input.dim() == 2 ? input.size(0) : input.size(1);
  const int64_t embed_dim = input.size(-1);
  const int64_t hidden_dim = vision_block_hidden_dim(context);
  const std::optional<utils::VulkanScratchArenaSpec> scratch_spec =
      !use_external_scratch && runtime_policy.scratch_arena_plan.has_value()
          ? [&]() -> std::optional<utils::VulkanScratchArenaSpec> {
              const auto requested_bytes = vision_attention_scratch_bytes(
                  batch_size,
                  token_count,
                  embed_dim,
                  context->num_heads(),
                  input.scalar_type(),
                  context->qkv_bias().defined(),
                  std::max<uint32_t>(
                      runtime_policy.scratch_arena_plan->alignment,
                      static_cast<uint32_t>(std::max<int64_t>(
                          1, static_cast<int64_t>(c10::elementSize(kFloat))))));
              if (
                  requested_bytes == 0u ||
                  !runtime_policy.scratch_arena_plan->prefer_buffer_storage) {
                return std::nullopt;
              }
              return utils::VulkanScratchArenaSpec{
                  kByte,
                  std::max(
                      requested_bytes,
                      runtime_policy.scratch_arena_plan->min_arena_bytes),
                  runtime_policy.scratch_arena_plan->alignment,
                  api::ExecutionLayout::BUFFER_DIRECT,
                  api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
                  api::StorageType::BUFFER,
                  runtime_policy.scratch_arena_plan->prefer_reusable_arena,
              };
            }()
          : std::nullopt;

  return utils::lookup_or_create_labeled_vision_backbone_program(
      vision_backbone_program_label(context->allocation_label(), context.get()),
      input.scalar_type(),
      batch_size,
      token_count,
      embed_dim,
      hidden_dim,
      context->num_heads(),
      scratch_spec,
      *runtime_policy.execution_program_plan);
}

VisionDecoderRunOutputs reserve_vision_decoder_graph_outputs(
    utils::ScratchArena& scratch_arena,
    const Tensor& input,
    const std::optional<Tensor>& skip,
    IntArrayRef target_sizes,
    const Tensor& out_conv_output) {
  VisionDecoderRunOutputs outputs;
  if (skip.has_value() && skip->defined()) {
    outputs.skip_relu_output =
        reserve_scratch_buffer_tensor(scratch_arena, skip->sizes(), kFloat).second;
    outputs.skip_conv1_output =
        reserve_scratch_buffer_tensor(scratch_arena, skip->sizes(), kFloat).second;
    outputs.skip_conv2_output =
        reserve_scratch_buffer_tensor(scratch_arena, skip->sizes(), kFloat).second;
    outputs.skip_res_output =
        reserve_scratch_buffer_tensor(scratch_arena, skip->sizes(), kFloat).second;
    outputs.main_input_output =
        reserve_scratch_buffer_tensor(scratch_arena, input.sizes(), kFloat).second;
  }

  outputs.main_relu_output =
      reserve_scratch_buffer_tensor(scratch_arena, input.sizes(), kFloat).second;
  outputs.main_conv1_output =
      reserve_scratch_buffer_tensor(scratch_arena, input.sizes(), kFloat).second;
  outputs.main_conv2_output =
      reserve_scratch_buffer_tensor(scratch_arena, input.sizes(), kFloat).second;
  outputs.main_res_output =
      reserve_scratch_buffer_tensor(scratch_arena, input.sizes(), kFloat).second;
  outputs.upsample_output = reserve_scratch_buffer_tensor(
                                scratch_arena,
                                {input.size(0), input.size(1), target_sizes[0], target_sizes[1]},
                                kFloat)
                                .second;
  outputs.out_conv_output = out_conv_output;
  return outputs;
}

bool can_use_decoder_replay(
    const Tensor& input,
    const std::optional<Tensor>& skip) {
  if (!input.defined() || !input.is_vulkan()) {
    return false;
  }
  const vTensor& v_input = convert(input);
  if (
      v_input.storage_type() != api::StorageType::BUFFER ||
      !v_input.has_direct_buffer_layout()) {
    return false;
  }
  if (skip.has_value() && skip->defined()) {
    const vTensor& v_skip = convert(*skip);
    if (
        v_skip.storage_type() != api::StorageType::BUFFER ||
        !v_skip.has_direct_buffer_layout()) {
      return false;
    }
  }
  return true;
}

Tensor run_vision_decoder_fusion_block_program(
    Tensor main_input,
    const std::optional<Tensor>& skip_tensor,
    IntArrayRef target_sizes,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& context,
    VisionDecoderRunOutputs outputs) {
  Tensor output;
  if (skip_tensor.has_value() && skip_tensor->defined()) {
    Tensor residual =
        relu_buffer_out_vulkan(*skip_tensor, outputs.skip_relu_output);
    residual = run_conv2d_context_relu_out(
        residual,
        context->res1_conv1_context(),
        outputs.skip_conv1_output);
    auto fused_skip_residual = try_run_conv2d_context_add_out(
        residual,
        context->res1_conv2_context(),
        *skip_tensor,
        outputs.skip_res_output);
    if (fused_skip_residual.has_value()) {
      residual = std::move(*fused_skip_residual);
    } else {
      residual =
          run_conv2d_context_out(
              residual,
              context->res1_conv2_context(),
              outputs.skip_conv2_output);
      residual = add_buffer_out_vulkan(
          residual, *skip_tensor, outputs.skip_res_output);
    }
    auto fused_main_input = try_add_relu_buffer_out_vulkan(
        main_input,
        residual,
        outputs.main_input_output,
        outputs.main_relu_output);
    if (fused_main_input.has_value()) {
      main_input = std::move(fused_main_input->first);
      output = std::move(fused_main_input->second);
    } else {
      main_input = add_buffer_out_vulkan(
          main_input, residual, outputs.main_input_output);
    }
  }

  if (!output.defined()) {
    output = relu_buffer_out_vulkan(main_input, outputs.main_relu_output);
  }
  output = run_conv2d_context_relu_out(
      output,
      context->res2_conv1_context(),
      outputs.main_conv1_output);
  auto fused_main_residual = try_run_conv2d_context_add_out(
      output,
      context->res2_conv2_context(),
      main_input,
      outputs.main_res_output);
  if (fused_main_residual.has_value()) {
    output = std::move(*fused_main_residual);
  } else {
    output = run_conv2d_context_out(
        output,
        context->res2_conv2_context(),
        outputs.main_conv2_output);
    output = add_buffer_out_vulkan(
        output, main_input, outputs.main_res_output);
  }
  if (can_run_decoder_out_conv_before_upsample(
          output, context, outputs.main_conv2_output)) {
    utils::log_vulkan_op_hit(
        "aten::vision_decoder.out_conv_before_upsample");
    output = run_conv2d_context_out(
        output,
        context->out_conv_context(),
        outputs.main_conv2_output);
    return upsample_bilinear2d_buffer_out_vulkan(
        output,
        target_sizes,
        context->align_corners(),
        std::nullopt,
        std::nullopt,
        outputs.out_conv_output);
  }
  output = upsample_bilinear2d_buffer_out_vulkan(
      output,
      target_sizes,
      context->align_corners(),
      std::nullopt,
      std::nullopt,
      outputs.upsample_output);
  return run_conv2d_context_out(
      output,
      context->out_conv_context(),
      outputs.out_conv_output);
}

VisionDecoderRunOutputs program_decoder_outputs(
    utils::VisionDecoderProgram& program) {
  return VisionDecoderRunOutputs{
      program.skip_relu_output(),
      program.skip_conv1_output(),
      program.skip_conv2_output(),
      program.skip_res_output(),
      program.main_input_output(),
      program.main_relu_output(),
      program.main_conv1_output(),
      program.main_conv2_output(),
      program.main_res_output(),
      program.upsample_output(),
      program.out_conv_output(),
  };
}

bool can_use_decoder_head_replay(
    const Tensor& layer1,
    const Tensor& layer2,
    const Tensor& layer3,
    const Tensor& layer4) {
  const auto can_use_tensor = [](const Tensor& tensor) {
    if (!tensor.defined() || !tensor.is_vulkan()) {
      return false;
    }
    const vTensor& v_tensor = convert(tensor);
    return v_tensor.storage_type() == api::StorageType::BUFFER &&
        v_tensor.has_direct_buffer_layout();
  };
  return can_use_tensor(layer1) && can_use_tensor(layer2) &&
      can_use_tensor(layer3) && can_use_tensor(layer4);
}

bool conv_context_has_buffer_direct_weight(
    const c10::intrusive_ptr<Conv2dPackedContext>& context) {
  return context &&
      context->packed_weight().execution_layout() ==
      api::ExecutionLayout::BUFFER_DIRECT;
}

bool can_use_decoder_fusion_program_context(
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& context) {
  return context &&
      conv_context_has_buffer_direct_weight(context->res1_conv1_context()) &&
      conv_context_has_buffer_direct_weight(context->res1_conv2_context()) &&
      conv_context_has_buffer_direct_weight(context->res2_conv1_context()) &&
      conv_context_has_buffer_direct_weight(context->res2_conv2_context()) &&
      conv_context_has_buffer_direct_weight(context->out_conv_context());
}

void copy_tensor_for_replay(Tensor& dst, const Tensor& src) {
  if (
      dst.defined() && src.defined() &&
      dst.unsafeGetTensorImpl() == src.unsafeGetTensorImpl()) {
    return;
  }
  dst.copy_(src);
  record_tensor_write(
      dst, "vulkan_prepack::copy_tensor_for_replay", "copy", {src});
}

Tensor materialize_escaping_vulkan_output(
    const Tensor& output,
    const bool persistent) {
  if (!output.defined() || !output.is_vulkan()) {
    return output;
  }
  Tensor materialized = utils::create_buffer_tensor(
      output.sizes(), output.scalar_type(), persistent);
  copy_tensor_for_replay(materialized, output);
  record_tensor_write(
      materialized,
      "vulkan_prepack::materialize_escaping_vulkan_output",
      "materialize_export",
      {output});
  utils::log_replay_event(
      "materialize_export",
      nullptr,
      0u,
      "vision_program_output",
      "action=materialize_escaping_output");
  return materialized;
}

std::optional<utils::VisionDecoderHeadInferenceReplay>
maybe_lookup_vision_decoder_head_replay(
    IntArrayRef layer1_sizes,
    IntArrayRef layer2_sizes,
    IntArrayRef layer3_sizes,
    IntArrayRef layer4_sizes,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderHeadContext>& context) {
  TORCH_INTERNAL_ASSERT(
      context,
      "Vision decoder head replay lookup expects a defined context");
  TORCH_INTERNAL_ASSERT(
      output_size.size() == 2,
      "Vision decoder head replay lookup expects a rank-1 output size with 2 "
      "entries");

  const auto runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_vision_decoder_request());
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionDecoder ||
      !has_explicit_runtime_capture_label()) {
    return std::nullopt;
  }

  const std::vector<int64_t> path1_sizes{
      layer1_sizes[0],
      layer1_sizes[1],
      layer1_sizes[2] * 2,
      layer1_sizes[3] * 2,
  };
  if (!can_run_depth_anything_v2_head_fusion_shape(
          path1_sizes, output_size, context)) {
    return std::nullopt;
  }

  const int64_t output_conv1_channels =
      context->output_conv1_context()
          ->unpack()
          .get(Conv2dPackedContext::Unpacked::Weight)
          .toTensor()
          .size(0);
  const int64_t output_conv2_channels =
      context->output_conv2_conv1_context()
          ->unpack()
          .get(Conv2dPackedContext::Unpacked::Weight)
          .toTensor()
          .size(0);
  const int64_t final_channels =
      context->output_conv2_conv2_context()
          ->unpack()
          .get(Conv2dPackedContext::Unpacked::Weight)
          .toTensor()
          .size(0);
  const std::vector<int64_t> output_sizes{
      layer1_sizes[0],
      final_channels,
      output_size[0],
      output_size[1],
  };

  auto vision_graph = utils::lookup_or_create_labeled_vision_decoder_inference_graph(
      vision_decoder_graph_label(context->allocation_label()),
      kFloat,
      runtime_policy.execution_program_plan->persistent);
  auto vision_replay = vision_graph.lookup_or_create_head_replay(
      vision_decoder_head_program_label(
          context->allocation_label(), context.get()),
      layer1_sizes,
      layer2_sizes,
      layer3_sizes,
      layer4_sizes,
      output_sizes,
      output_conv1_channels,
      output_conv2_channels,
      final_channels,
      *runtime_policy.execution_program_plan);
  if (!vision_replay.defined()) {
    return std::nullopt;
  }
  return vision_replay;
}

Tensor run_vision_decoder_head_program(
    const Tensor& layer1,
    const Tensor& layer2,
    const Tensor& layer3,
    const Tensor& layer4,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderHeadContext>& context,
    utils::VisionDecoderProgram& refinenet4_program,
    utils::VisionDecoderProgram& refinenet3_program,
    utils::VisionDecoderProgram& refinenet2_program,
    utils::VisionDecoderProgram& refinenet1_program,
    Tensor& output_slot) {
  const std::vector<int64_t> layer3_target{layer3.size(2), layer3.size(3)};
  const std::vector<int64_t> layer2_target{layer2.size(2), layer2.size(3)};
  const std::vector<int64_t> layer1_target{layer1.size(2), layer1.size(3)};
  const std::vector<int64_t> path1_target{
      layer1.size(2) * 2, layer1.size(3) * 2};

  Tensor path4 = run_vision_decoder_fusion_block_program(
      layer4,
      std::nullopt,
      layer3_target,
      context->refinenet4_context(),
      program_decoder_outputs(refinenet4_program));
  Tensor path3 = run_vision_decoder_fusion_block_program(
      path4,
      layer3,
      layer2_target,
      context->refinenet3_context(),
      program_decoder_outputs(refinenet3_program));
  Tensor path2 = run_vision_decoder_fusion_block_program(
      path3,
      layer2,
      layer1_target,
      context->refinenet2_context(),
      program_decoder_outputs(refinenet2_program));
  Tensor path1 = run_vision_decoder_fusion_block_program(
      path2,
      layer1,
      path1_target,
      context->refinenet1_context(),
      program_decoder_outputs(refinenet1_program));
  TORCH_INTERNAL_ASSERT(
      can_run_depth_anything_v2_head_fusion(path1, output_size, context),
      "Vision decoder head program expects a DA v2-compatible fused head");
  return run_vision_decoder_head_tail_context(
      path1, output_size, context, &output_slot);
}

std::tuple<Tensor, Tensor, Tensor> reshape_qkv_for_attention(
    const Tensor& mixed_qkv,
    const int64_t batch_size,
    const int64_t token_count,
    const int64_t num_heads,
    const int64_t head_dim) {
  std::vector<Tensor> qkv = at::chunk(mixed_qkv, 3, 2);
  Tensor q =
      qkv[0].reshape({batch_size, token_count, num_heads, head_dim})
          .permute({0, 2, 1, 3})
          .reshape({batch_size * num_heads, token_count, head_dim});
  Tensor k =
      qkv[1].reshape({batch_size, token_count, num_heads, head_dim})
          .permute({0, 2, 1, 3})
          .reshape({batch_size * num_heads, token_count, head_dim});
  Tensor v =
      qkv[2].reshape({batch_size, token_count, num_heads, head_dim})
          .permute({0, 2, 1, 3})
          .reshape({batch_size * num_heads, token_count, head_dim});
  return std::make_tuple(std::move(q), std::move(k), std::move(v));
}

std::optional<Tensor> normalize_attention_bias_for_batch_heads(
    const Tensor& attention_bias,
    const int64_t batch_size,
    const int64_t num_heads,
    const int64_t token_count) {
  if (!attention_bias.defined()) {
    return std::nullopt;
  }
  Tensor bias = attention_bias;
  TORCH_CHECK(
      bias.scalar_type() == kFloat || bias.scalar_type() == kHalf,
      "Vision backbone attention bias expects float or half dtype");
  if (bias.dim() == 4) {
    TORCH_CHECK(
        bias.size(2) == token_count && bias.size(3) == token_count,
        "Vision backbone attention bias expects square token dimensions");
    TORCH_CHECK(
        (bias.size(0) == batch_size || bias.size(0) == 1) &&
            (bias.size(1) == num_heads || bias.size(1) == 1),
        "Vision backbone attention bias expects [B|1, H|1, T, T] shape");
    if (
        batch_size == 1 && bias.size(0) == 1 &&
        (bias.size(1) == num_heads || bias.size(1) == 1)) {
      return bias.squeeze(0)
          .expand({num_heads, token_count, token_count})
          .contiguous();
    }
    if (
        num_heads == 1 && bias.size(1) == 1 &&
        (bias.size(0) == batch_size || bias.size(0) == 1)) {
      return bias.squeeze(1)
          .expand({batch_size, token_count, token_count})
          .contiguous();
    }
    bias = bias.expand({batch_size, num_heads, token_count, token_count})
               .reshape({batch_size * num_heads, token_count, token_count});
    return bias.contiguous();
  }
  if (bias.dim() == 3) {
    TORCH_CHECK(
        bias.size(1) == token_count && bias.size(2) == token_count,
        "Vision backbone attention bias expects square token dimensions");
    TORCH_CHECK(
        bias.size(0) == batch_size * num_heads || bias.size(0) == num_heads ||
            bias.size(0) == batch_size || bias.size(0) == 1,
        "Vision backbone attention bias expects [BH|H|B|1, T, T] shape");
    if (bias.size(0) == batch_size * num_heads) {
      return bias;
    }
    if (batch_size == 1 && (bias.size(0) == num_heads || bias.size(0) == 1)) {
      return bias.expand({num_heads, token_count, token_count}).contiguous();
    }
    if (num_heads == 1 && (bias.size(0) == batch_size || bias.size(0) == 1)) {
      return bias.expand({batch_size, token_count, token_count}).contiguous();
    }
    if (bias.size(0) == num_heads) {
      return bias.unsqueeze(0)
          .expand({batch_size, num_heads, token_count, token_count})
          .reshape({batch_size * num_heads, token_count, token_count})
          .contiguous();
    }
    return bias.unsqueeze(1)
        .expand({batch_size, num_heads, token_count, token_count})
        .reshape({batch_size * num_heads, token_count, token_count})
        .contiguous();
  }
  TORCH_CHECK(
      false,
      "Vision backbone attention bias expects rank-3 or rank-4 tensor");
}

Tensor ensure_attention_merge_output_tensor(
    Tensor& output,
    const int64_t batch_size,
    const int64_t token_count,
    const int64_t embed_dim,
    const ScalarType dtype) {
  const std::vector<int64_t> output_sizes{
      batch_size * token_count,
      embed_dim,
  };
  bool needs_allocation = !output.defined() || !output.is_vulkan() ||
      output.scalar_type() != dtype ||
      !output.sizes().equals(IntArrayRef(output_sizes));
  if (!needs_allocation) {
    const vTensor& v_output = convert(output);
    needs_allocation =
        v_output.storage_type() != api::StorageType::BUFFER ||
        v_output.gpu_memory_layout() !=
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED ||
        !utils::supports_buffer_view_fast_path(v_output);
  }
  if (needs_allocation) {
    output = utils::mark_tensor_execution(
        convert(vTensor{
            api::context(),
            output_sizes,
            convert_dtype(dtype),
            api::StorageType::BUFFER,
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
        }),
        api::ExecutionLayout::BUFFER_DIRECT);
  } else {
    output = utils::mark_tensor_execution(
        output,
        utils::resolve_buffer_execution_layout(convert(output)));
  }
  return output;
}

Tensor merge_attention_heads_for_projection(
    const Tensor& attention_output_arg,
    const int64_t batch_size,
    const int64_t token_count,
    const int64_t num_heads,
    const int64_t head_dim,
    Tensor* output_opt = nullptr) {
  api::AllocationScope allocation_scope("attention_merge_heads");
  const int64_t batch_heads = batch_size * num_heads;
  const int64_t embed_dim = num_heads * head_dim;

  Tensor attention_output = attention_output_arg.is_vulkan()
      ? attention_output_arg
      : attention_output_arg.vulkan();
  {
    const vTensor& v_attention_output = convert(attention_output);
    if (
        v_attention_output.storage_type() == api::StorageType::BUFFER &&
        v_attention_output.gpu_memory_layout() ==
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
        utils::supports_buffer_view_fast_path(v_attention_output)) {
      attention_output = utils::mark_tensor_execution(
          attention_output,
          utils::resolve_buffer_execution_layout(v_attention_output));
    } else {
      attention_output = utils::mark_tensor_execution(
          utils::ensure_buffer_storage(
              attention_output, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
          api::ExecutionLayout::BUFFER_DIRECT);
    }
  }

  TORCH_CHECK(
      attention_output.dim() == 3,
      "Vulkan attention head merge expects a rank-3 [B*H, T, D] tensor");
  TORCH_CHECK(
      attention_output.size(0) == batch_heads &&
          attention_output.size(1) == token_count &&
          attention_output.size(2) == head_dim,
      "Vulkan attention head merge received unexpected attention output sizes");

  vTensor& v_input = convert(attention_output);
  TORCH_CHECK(
      v_input.storage_type() == api::StorageType::BUFFER &&
          utils::supports_buffer_view_fast_path(v_input),
      "Vulkan attention head merge expects buffer-backed attention output");

  Tensor output_tensor = output_opt
      ? ensure_attention_merge_output_tensor(
            *output_opt, batch_size, token_count, embed_dim, attention_output.scalar_type())
      : utils::mark_tensor_execution(
            convert(vTensor{
                api::context(),
                {batch_size * token_count, embed_dim},
                convert_dtype(attention_output.scalar_type()),
                api::StorageType::BUFFER,
                api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
            }),
            api::ExecutionLayout::BUFFER_DIRECT);
  vTensor& v_output = convert(output_tensor);

  const struct Block final {
    int32_t head_dim;
    int32_t token_count;
    int32_t num_heads;
    int32_t batch_size;
  } block{
      api::utils::safe_downcast<int32_t>(head_dim),
      api::utils::safe_downcast<int32_t>(token_count),
      api::utils::safe_downcast<int32_t>(num_heads),
      api::utils::safe_downcast<int32_t>(batch_size),
  };

  api::UniformParamsBuffer params(api::context(), block);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(head_dim),
      api::utils::safe_downcast<uint32_t>(token_count),
      api::utils::safe_downcast<uint32_t>(batch_heads),
  };

  api::context()->submit_compute_job(
      VK_KERNEL(merge_attention_heads_buffer),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_output.buffer_metadata(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_input.buffer_metadata(),
      params.buffer());

  utils::log_vulkan_op_hit("aten::attention_merge_heads.buffer_native");
  return output_tensor;
}

Tensor run_attention_projection(
    const Tensor& input_2d,
    const int64_t batch_size,
    const int64_t token_count,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context,
    utils::VisionBackboneProgram* vision_program = nullptr,
    utils::ScratchArena* scratch_override = nullptr) {
  TORCH_CHECK(
      input_2d.dim() == 2,
      "Vision backbone attention projection expects flattened rank-2 input");

  const int64_t embed_dim = input_2d.size(-1);
  TORCH_CHECK(
      embed_dim % context->num_heads() == 0,
      "Vision backbone block context expects embed_dim divisible by num_heads");
  const int64_t head_dim = embed_dim / context->num_heads();
  utils::ScratchArena* scratch_arena = scratch_override;
  if (
      !scratch_arena && vision_program && vision_program->defined() &&
      vision_program->scratch_arena().has_value()) {
    scratch_arena = &(*vision_program->scratch_arena());
  }
  const bool use_program_scratch = scratch_arena != nullptr;
  Tensor attention_output;
  const std::optional<Tensor> attention_bias =
      context->attention_bias().defined()
      ? normalize_attention_bias_for_batch_heads(
            context->attention_bias(),
            batch_size,
            context->num_heads(),
            token_count)
      : std::nullopt;
  if (batch_size == 1) {
    const bool use_scratch_qkv_projection =
        use_program_scratch &&
        input_2d.scalar_type() == kFloat && context->qkv_bias().defined();

    std::optional<utils::VulkanScratchSlice> mixed_qkv_slice;
    Tensor mixed_qkv_output;
    if (use_scratch_qkv_projection) {
      auto scratch_qkv_output = reserve_scratch_buffer_tensor(
          *scratch_arena,
          {token_count, 3 * embed_dim},
          input_2d.scalar_type());
      mixed_qkv_slice = scratch_qkv_output.first;
      mixed_qkv_output = std::move(scratch_qkv_output.second);
    }

    Tensor mixed_qkv;
    {
      api::VulkanVisionStackPhaseScope scope(
          api::VulkanVisionStackPhase::QkvLinear);
      if (vision_program && vision_program->defined()) {
        mixed_qkv = use_scratch_qkv_projection
            ? run_linear_context_out(
                  input_2d, context->qkv_context(), mixed_qkv_output)
            : run_linear_context_out(
                  input_2d, context->qkv_context(), vision_program->qkv_output());
      } else {
        mixed_qkv = run_linear_context(input_2d, context->qkv_context());
      }
      note_stack_execution_manifest_row(
          "vision_block.qkv_linear",
          "mm_buffer_float_bias",
          {std::cref(input_2d)},
          {std::cref(mixed_qkv)},
          !vision_program,
          vision_program && vision_program->defined(),
          false,
          false,
          true);
    }
    Tensor q;
    Tensor k;
    Tensor v;
    bool q_is_scaled = false;
    {
      api::VulkanVisionStackPhaseScope scope(
          api::VulkanVisionStackPhase::QkvTransform);
      if (context->qkv_bias().defined()) {
        if (use_scratch_qkv_projection) {
          auto [q_slice, q_output] = reserve_scratch_buffer_tensor(
              *scratch_arena,
              {context->num_heads(), token_count, head_dim},
              kFloat);
          auto [k_slice, k_output] = reserve_scratch_buffer_tensor(
              *scratch_arena,
              {context->num_heads(), token_count, head_dim},
              kFloat);
          auto [v_slice, v_output] = reserve_scratch_buffer_tensor(
              *scratch_arena,
              {context->num_heads(), token_count, head_dim},
              kFloat);
          (void)q_slice;
          (void)k_slice;
          (void)v_slice;
          std::tie(q, k, v) = transform_bias_rescale_qkv_vulkan_out(
              mixed_qkv,
              context->qkv_bias(),
              context->num_heads(),
              q_output,
              k_output,
              v_output);
        } else {
          std::tie(q, k, v) = at::_transform_bias_rescale_qkv(
              mixed_qkv, context->qkv_bias(), context->num_heads());
        }
        q_is_scaled = true;
      } else {
        std::vector<Tensor> qkv = at::chunk(mixed_qkv, 3, 1);
        q = qkv[0].reshape({token_count, context->num_heads(), head_dim})
                .permute({1, 0, 2});
        k = qkv[1].reshape({token_count, context->num_heads(), head_dim})
                .permute({1, 0, 2});
        v = qkv[2].reshape({token_count, context->num_heads(), head_dim})
                .permute({1, 0, 2});
      }
      note_stack_execution_manifest_row(
          "vision_block.qkv_transform",
          context->qkv_bias().defined() ? "transform_bias_rescale_qkv"
                                        : "reshape_qkv",
          {std::cref(mixed_qkv)},
          {std::cref(q), std::cref(k), std::cref(v)},
          true,
          false,
          false,
          false,
          context->qkv_bias().defined());
    }
    if (!q_is_scaled) {
      q = at::mul(
          q,
          static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim))));
    }
    {
      api::VulkanVisionStackPhaseScope scope(
          api::VulkanVisionStackPhase::Attention);
      attention_output = run_attention_with_workspace_fallback(
          q, k, v, attention_bias, vision_program, scratch_arena);
      note_stack_execution_manifest_row(
          "vulkan_prepack::vision_stack_attention_direct",
          "scaled_dot_product_scores_value_buffer_float_head64_q4_subgroup",
          {std::cref(q), std::cref(k), std::cref(v)},
          {std::cref(attention_output)},
          true,
          false,
          false,
          false,
          true);
    }
    Tensor scratch_merge_output;
    Tensor* merge_output_opt = nullptr;
    if (use_scratch_qkv_projection && mixed_qkv_slice.has_value()) {
      scratch_merge_output = make_scratch_buffer_alias(
          *scratch_arena,
          *mixed_qkv_slice,
          {batch_size * token_count, embed_dim},
          attention_output.scalar_type());
      merge_output_opt = &scratch_merge_output;
    } else if (use_program_scratch) {
      auto [merge_slice, merge_output] = reserve_scratch_buffer_tensor(
          *scratch_arena,
          {batch_size * token_count, embed_dim},
          attention_output.scalar_type());
      (void)merge_slice;
      scratch_merge_output = std::move(merge_output);
      merge_output_opt = &scratch_merge_output;
    } else if (vision_program && vision_program->defined()) {
      merge_output_opt = &vision_program->merge_output();
    }
    {
      api::VulkanVisionStackPhaseScope scope(
          api::VulkanVisionStackPhase::Attention);
      attention_output = merge_attention_heads_for_projection(
          attention_output,
          batch_size,
          token_count,
          context->num_heads(),
          head_dim,
          merge_output_opt);
      note_stack_execution_manifest_row(
          "vision_block.attention_merge_heads",
          "merge_attention_heads_buffer",
          {std::cref(attention_output)},
          {std::cref(attention_output)},
          merge_output_opt == nullptr,
          merge_output_opt != nullptr,
          false,
          false,
          true);
    }
    api::VulkanVisionStackPhaseScope scope(
        api::VulkanVisionStackPhase::ProjLinear);
    Tensor proj_output = vision_program && vision_program->defined()
        ? run_linear_context_out(
              attention_output,
              context->proj_context(),
              vision_program->proj_output())
        : run_linear_context(attention_output, context->proj_context());
    note_stack_execution_manifest_row(
        "vision_block.proj_linear",
        "mm_buffer_float_bias",
        {std::cref(attention_output)},
        {std::cref(proj_output)},
        !(vision_program && vision_program->defined()),
        vision_program && vision_program->defined(),
        false,
        false,
        true);
    return proj_output;
  }

  Tensor mixed_qkv;
  {
    api::VulkanVisionStackPhaseScope scope(
        api::VulkanVisionStackPhase::QkvLinear);
    mixed_qkv = vision_program && vision_program->defined()
        ? run_linear_context_out(
              input_2d, context->qkv_context(), vision_program->qkv_output())
        : run_linear_context(input_2d, context->qkv_context());
    note_stack_execution_manifest_row(
        "vision_block.qkv_linear",
        "mm_buffer_float_bias",
        {std::cref(input_2d)},
        {std::cref(mixed_qkv)},
        !(vision_program && vision_program->defined()),
        vision_program && vision_program->defined(),
        false,
        false,
        true);
  }
  if (context->qkv_bias().defined()) {
    api::VulkanVisionStackPhaseScope scope(
        api::VulkanVisionStackPhase::QkvTransform);
    mixed_qkv = mixed_qkv.add(context->qkv_bias());
  }
  {
    api::VulkanVisionStackPhaseScope scope(
        api::VulkanVisionStackPhase::QkvTransform);
    mixed_qkv = mixed_qkv.reshape({batch_size, token_count, 3 * embed_dim});
  }
  Tensor q;
  Tensor k;
  Tensor v;
  {
    api::VulkanVisionStackPhaseScope scope(
        api::VulkanVisionStackPhase::QkvTransform);
    std::tie(q, k, v) = reshape_qkv_for_attention(
        mixed_qkv, batch_size, token_count, context->num_heads(), head_dim);
    note_stack_execution_manifest_row(
        "vision_block.qkv_transform",
        "reshape_qkv",
        {std::cref(mixed_qkv)},
        {std::cref(q), std::cref(k), std::cref(v)},
        true,
        false,
        false,
        false,
        false);
  }
  q = at::mul(
      q,
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim))));
  {
    api::VulkanVisionStackPhaseScope scope(
        api::VulkanVisionStackPhase::Attention);
    attention_output = run_attention_with_workspace_fallback(
        q, k, v, attention_bias, vision_program, scratch_arena);
    note_stack_execution_manifest_row(
        "vulkan_prepack::vision_stack_attention_direct",
        "scaled_dot_product_scores_value_buffer_float_head64_q4_subgroup",
        {std::cref(q), std::cref(k), std::cref(v)},
        {std::cref(attention_output)},
        true,
        false,
        false,
        false,
        true);
  }
  Tensor scratch_merge_output;
  Tensor* merge_output_opt = nullptr;
  if (use_program_scratch) {
    auto [merge_slice, merge_output] = reserve_scratch_buffer_tensor(
        *scratch_arena,
        {batch_size * token_count, embed_dim},
        attention_output.scalar_type());
    (void)merge_slice;
    scratch_merge_output = std::move(merge_output);
    merge_output_opt = &scratch_merge_output;
  } else if (vision_program && vision_program->defined()) {
    merge_output_opt = &vision_program->merge_output();
  }
  {
    api::VulkanVisionStackPhaseScope scope(
        api::VulkanVisionStackPhase::Attention);
    attention_output = merge_attention_heads_for_projection(
        attention_output,
        batch_size,
        token_count,
        context->num_heads(),
        head_dim,
        merge_output_opt);
    note_stack_execution_manifest_row(
        "vision_block.attention_merge_heads",
        "merge_attention_heads_buffer",
        {std::cref(attention_output)},
        {std::cref(attention_output)},
        merge_output_opt == nullptr,
        merge_output_opt != nullptr,
        false,
        false,
        true);
  }
  api::VulkanVisionStackPhaseScope scope(
      api::VulkanVisionStackPhase::ProjLinear);
  Tensor proj_output = vision_program && vision_program->defined()
      ? run_linear_context_out(
            attention_output,
            context->proj_context(),
            vision_program->proj_output())
      : run_linear_context(attention_output, context->proj_context());
  note_stack_execution_manifest_row(
      "vision_block.proj_linear",
      "mm_buffer_float_bias",
      {std::cref(attention_output)},
      {std::cref(proj_output)},
      !(vision_program && vision_program->defined()),
      vision_program && vision_program->defined(),
      false,
      false,
      true);
  return proj_output;
}

Tensor tokens_to_feature_map_fallback(
    const Tensor& input_arg,
    const int64_t height,
    const int64_t width) {
  Tensor input = input_arg;
  if (input.dim() == 2) {
    input = input.unsqueeze(0);
  }

  TORCH_CHECK(
      input.dim() == 3,
      "Vulkan tokens_to_feature_map expects a [N, C] or [B, N, C] tensor");
  TORCH_CHECK(
      input.size(1) == height * width,
      "Vulkan tokens_to_feature_map expected token count ",
      height * width,
      " but received ",
      input.size(1));

  Tensor output;
  {
    if (input_arg.is_vulkan()) {
      report_vulkan_cpu_fallback(
          "aten::tokens_to_feature_map",
          "vision_helper_cpu_materialization",
          {input_arg});
    }
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);

    Tensor cpu_input = input.is_vulkan() ? input.cpu() : input;
    output = cpu_input.reshape(
        {cpu_input.size(0), height, width, cpu_input.size(2)});
    output = output.permute({0, 3, 1, 2}).contiguous();
  }

  if (input_arg.is_vulkan()) {
    return record_tensor_write_and_return(
        output.vulkan(),
        "aten::tokens_to_feature_map",
        "cpu_fallback",
        {input_arg});
  }
  return output;
}

Tensor feature_map_to_tokens_fallback(const Tensor& input_arg) {
  TORCH_CHECK(
      input_arg.dim() == 4,
      "Vulkan feature_map_to_tokens expects a [B, C, H, W] tensor");

  Tensor output;
  {
    if (input_arg.is_vulkan()) {
      report_vulkan_cpu_fallback(
          "aten::feature_map_to_tokens",
          "vision_helper_cpu_materialization",
          {input_arg});
    }
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);

    Tensor cpu_input = input_arg.is_vulkan() ? input_arg.cpu() : input_arg;
    output = cpu_input.permute({0, 2, 3, 1})
                 .reshape(
                     {cpu_input.size(0),
                      cpu_input.size(2) * cpu_input.size(3),
                      cpu_input.size(1)})
                 .contiguous();
  }

  if (input_arg.is_vulkan()) {
    return record_tensor_write_and_return(
        output.vulkan(),
        "aten::feature_map_to_tokens",
        "cpu_fallback",
        {input_arg});
  }
  return output;
}

Tensor run_vision_backbone_block_program(
    const Tensor& input,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context,
    utils::VisionBackboneProgram* const vision_program,
    utils::ScratchArena* const graph_scratch,
    Tensor* const output_slot = nullptr) {
  const bool use_2d_input = input.dim() == 2;
  const int64_t batch_size = use_2d_input ? 1 : input.size(0);
  const int64_t token_count = use_2d_input ? input.size(0) : input.size(1);
  const int64_t embed_dim = input.size(-1);
  const int64_t hidden_rows = batch_size * token_count;
  Tensor input_2d = use_2d_input ? input : input.reshape({hidden_rows, embed_dim});

  const std::array<int64_t, 1> normalized_shape = {embed_dim};
  Tensor attention_input;
  {
    api::VulkanVisionStackPhaseScope scope(api::VulkanVisionStackPhase::Norm1);
    attention_input = vision_program
        ? run_layernorm_context_out(
              input_2d,
              normalized_shape,
              context->norm1_context(),
              vision_program->norm1_output())
        : run_layernorm_context(
              input_2d, normalized_shape, context->norm1_context());
    note_stack_execution_manifest_row(
        "vision_block.norm1",
        "native_layer_norm",
        {std::cref(input_2d)},
        {std::cref(attention_input)},
        !vision_program,
        vision_program != nullptr,
        false,
        false,
        true);
  }
  Tensor attention_output = run_attention_projection(
      attention_input,
      batch_size,
      token_count,
      context,
      vision_program,
      graph_scratch);

  Tensor hidden_states;
  Tensor mlp_input;
  Tensor attention_addend = attention_output;
  if (vision_program && context->ls1_gamma().defined()) {
    api::VulkanVisionStackPhaseScope scope(
        api::VulkanVisionStackPhase::Residual1);
    // norm1_output is no longer needed after attention, so use it as the
    // retained residual scratch for the fused residual-add + norm2 pass.
    auto fused_residual_norm = try_run_add_scaled_layernorm_context_out(
        input_2d,
        attention_output,
        context->ls1_gamma(),
        normalized_shape,
        context->norm2_context(),
        vision_program->norm1_output(),
        vision_program->norm2_output());
    if (fused_residual_norm.has_value()) {
      hidden_states = std::move(fused_residual_norm->first);
      mlp_input = std::move(fused_residual_norm->second);
    }
  }
  if (!mlp_input.defined()) {
    api::VulkanVisionStackPhaseScope scope(
        api::VulkanVisionStackPhase::Residual1);
    attention_addend =
        maybe_apply_layerscale(attention_output, context->ls1_gamma());
  }
  if (vision_program && !mlp_input.defined()) {
    api::VulkanVisionStackPhaseScope scope(
        api::VulkanVisionStackPhase::Residual1);
    auto fused_residual_norm = try_run_add_layernorm_context_out(
        input_2d,
        attention_addend,
        normalized_shape,
        context->norm2_context(),
        vision_program->norm1_output(),
        vision_program->norm2_output());
    if (fused_residual_norm.has_value()) {
      hidden_states = std::move(fused_residual_norm->first);
      mlp_input = std::move(fused_residual_norm->second);
    }
  }
  if (!mlp_input.defined()) {
    {
      api::VulkanVisionStackPhaseScope scope(
          api::VulkanVisionStackPhase::Residual1);
      hidden_states = at::add(input_2d, attention_addend);
      note_stack_execution_manifest_row(
          "vision_block.residual1",
          "buffer_add",
          {std::cref(input_2d), std::cref(attention_addend)},
          {std::cref(hidden_states)},
          true,
          false,
          false,
          false,
          true);
    }
    {
      api::VulkanVisionStackPhaseScope scope(api::VulkanVisionStackPhase::Norm2);
      mlp_input = vision_program
          ? run_layernorm_context_out(
                hidden_states,
                normalized_shape,
                context->norm2_context(),
                vision_program->norm2_output())
          : run_layernorm_context(
                hidden_states, normalized_shape, context->norm2_context());
      note_stack_execution_manifest_row(
          "vision_block.norm2",
          "native_layer_norm",
          {std::cref(hidden_states)},
          {std::cref(mlp_input)},
          !vision_program,
          vision_program != nullptr,
          false,
          false,
          true);
    }
  }
  auto& mlp_counters = vulkan_vision_owner_mlp_counters();
  mlp_counters.total.fetch_add(1u, std::memory_order_relaxed);
  if (!context->fc1_context() || !context->fc2_context()) {
    mlp_counters.reject_context.fetch_add(1u, std::memory_order_relaxed);
  } else if (mlp_input.scalar_type() != kFloat) {
    mlp_counters.reject_dtype.fetch_add(1u, std::memory_order_relaxed);
  } else if (mlp_input.dim() != 2 && mlp_input.dim() != 3) {
    mlp_counters.reject_shape.fetch_add(1u, std::memory_order_relaxed);
  }
  Tensor mlp_output;
  {
    api::VulkanVisionStackPhaseScope scope(api::VulkanVisionStackPhase::Fc1Gelu);
    mlp_output = vision_program
        ? run_linear_gelu_context_out(
              mlp_input, context->fc1_context(), vision_program->fc1_output())
        : run_linear_gelu_context(mlp_input, context->fc1_context());
    note_stack_execution_manifest_row(
        "vision_block.fc1_gelu",
        "mm_buffer_float_bias_gelu",
        {std::cref(mlp_input)},
        {std::cref(mlp_output)},
        !vision_program,
        vision_program != nullptr,
        false,
        false,
        true);
  }
  mlp_counters.linear_gelu_hit.fetch_add(1u, std::memory_order_relaxed);

  {
    api::VulkanVisionStackPhaseScope scope(api::VulkanVisionStackPhase::Fc2);
    Tensor fc2_input = mlp_output;
    mlp_output = vision_program
        ? run_linear_context_out(
              fc2_input, context->fc2_context(), vision_program->fc2_output())
        : run_linear_context(fc2_input, context->fc2_context());
    note_stack_execution_manifest_row(
        "vision_block.fc2",
        "mm_buffer_float_bias",
        {std::cref(fc2_input)},
        {std::cref(mlp_output)},
        !vision_program,
        vision_program != nullptr,
        false,
        false,
        true);
  }
  mlp_counters.fc2_after_linear_gelu_hit.fetch_add(
      1u,
      std::memory_order_relaxed);
  append_vulkan_vision_owner_block_log(input, context, true, true);
  Tensor mlp_addend = mlp_output;

  if (output_slot && output_slot->defined() && hidden_states.scalar_type() == kFloat &&
      mlp_output.scalar_type() == kFloat) {
    api::VulkanVisionStackPhaseScope scope(
        api::VulkanVisionStackPhase::Residual2);
    Tensor add_output = use_2d_input
        ? *output_slot
        : output_slot->reshape({hidden_rows, embed_dim});
    if (context->ls2_gamma().defined()) {
      auto scaled_add = try_add_scaled_buffer_out_vulkan(
          hidden_states, mlp_output, context->ls2_gamma(), add_output);
      if (scaled_add.has_value()) {
        return *output_slot;
      }
    }
    mlp_addend = maybe_apply_layerscale(mlp_output, context->ls2_gamma());
    (void)add_buffer_out_vulkan(hidden_states, mlp_addend, add_output);
    note_stack_execution_manifest_row(
        "vision_block.residual2",
        "buffer_add",
        {std::cref(hidden_states), std::cref(mlp_addend)},
        {std::cref(*output_slot)},
        false,
        true,
        false,
        false,
        true);
    return *output_slot;
  }

  Tensor output;
  {
    api::VulkanVisionStackPhaseScope scope(
        api::VulkanVisionStackPhase::Residual2);
    mlp_addend = maybe_apply_layerscale(mlp_output, context->ls2_gamma());
    output = at::add(hidden_states, mlp_addend);
    note_stack_execution_manifest_row(
        "vision_block.residual2",
        "buffer_add",
        {std::cref(hidden_states), std::cref(mlp_addend)},
        {std::cref(output)},
        true,
        false,
        false,
        false,
        true);
  }
  if (!use_2d_input) {
    output = output.reshape({batch_size, token_count, embed_dim});
  }
  return output;
}

utils::ExecutionGraphReplayStep make_vision_backbone_replay_step(
    utils::VisionBackboneInferenceReplay backbone_replay,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& backbone_context,
    std::optional<utils::ScratchArena> backbone_graph_scratch) {
  return backbone_replay.phase_step(
      [backbone_replay,
       backbone_context,
       backbone_graph_scratch]() mutable {
        if (backbone_graph_scratch.has_value()) {
          backbone_graph_scratch->reset();
        }
        (void)run_vision_backbone_block_program(
            backbone_replay.input_slot(),
            backbone_context,
            &backbone_replay.program(),
            backbone_graph_scratch.has_value() ? &(*backbone_graph_scratch)
                                              : nullptr,
            &backbone_replay.output_slot());
      });
}

utils::ExecutionGraphReplayStep make_chained_vision_backbone_replay_step(
    utils::VisionBackboneInferenceReplay previous_replay,
    utils::VisionBackboneInferenceReplay backbone_replay,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& backbone_context,
    std::optional<utils::ScratchArena> backbone_graph_scratch) {
  return backbone_replay.phase_step(
      [previous_replay,
       backbone_replay,
       backbone_context,
       backbone_graph_scratch]() mutable {
        if (backbone_graph_scratch.has_value()) {
          backbone_graph_scratch->reset();
        }
        (void)run_vision_backbone_block_program(
            previous_replay.output_slot(),
            backbone_context,
            &backbone_replay.program(),
            backbone_graph_scratch.has_value() ? &(*backbone_graph_scratch)
                                              : nullptr,
            &backbone_replay.output_slot());
      });
}

size_t compiled_session_tensor_slot(
    const utils::VulkanCompiledSessionTensorBindings& bindings,
    const utils::VulkanValueId value_id) {
  TORCH_INTERNAL_ASSERT(
      value_id < bindings.value_tensor_slots.size() &&
          bindings.value_tensor_slots[value_id].has_value(),
      "Compiled session value does not have a tensor slot binding");
  return *bindings.value_tensor_slots[value_id];
}

std::shared_ptr<std::vector<Tensor>> make_compiled_session_tensor_slots(
    const utils::VulkanCompiledSession& session,
    const utils::VulkanCompiledSessionTensorBindings& bindings,
    const bool persistent) {
  const auto& values = session.ir().values();
  auto tensor_slots =
      std::make_shared<std::vector<Tensor>>(bindings.tensor_slot_count());
  for (size_t slot_idx = 0u; slot_idx < bindings.slot_values.size(); ++slot_idx) {
    const utils::VulkanValueId value_id = bindings.slot_values[slot_idx];
    TORCH_CHECK(
        value_id < values.size(),
        "Compiled session tensor slot references an invalid value");
    const auto& spec = values[value_id].spec;
    TORCH_CHECK(
        spec.storage_type == api::StorageType::BUFFER,
        "Compiled session tensor slots currently require buffer storage");
    TORCH_CHECK(
        spec.execution_layout == api::ExecutionLayout::BUFFER_DIRECT,
        "Compiled session tensor slots currently require direct buffer execution");
    tensor_slots->at(slot_idx) = utils::create_buffer_tensor(
        spec.logical_sizes,
        spec.dtype,
        persistent);
  }
  return tensor_slots;
}

struct CompiledBackboneExecutionPlan final {
  utils::VulkanValueId input_value{0u};
  utils::VulkanValueId backbone_input_value{0u};
  std::vector<utils::VulkanValueId> block_output_values;
  std::vector<std::optional<utils::VulkanValueId>> capture_values_by_block;
};

struct CompiledImageEntryExecutionPlan final {
  utils::VulkanValueId image_input_value{0u};
  utils::VulkanValueId patch_feature_map_value{0u};
  utils::VulkanValueId patch_feature_tokens_value{0u};
  utils::VulkanValueId positioned_patch_tokens_value{0u};
  utils::VulkanValueId patch_token_value{0u};
  utils::VulkanValueId prefix_token_value{0u};
  utils::VulkanValueId patch_pos_encoding_value{0u};
};

struct CompiledDecoderExecutionPlan final {
  std::array<utils::VulkanValueId, 4u> head_input_values{};
  utils::VulkanValueId head_output_value{0u};
  utils::VulkanValueId final_output_value{0u};
};

struct CompiledDecoderHeadPrograms final {
  std::array<utils::VisionDecoderProgram, 4u> programs;

  bool defined() const {
    return std::all_of(
        programs.cbegin(), programs.cend(), [](const auto& program) {
          return program.defined();
        });
  }
};

std::optional<CompiledBackboneExecutionPlan> make_compiled_backbone_execution_plan(
    const utils::VulkanCompiledSession& session,
    const size_t block_count) {
  if (!session.defined() || !session.executable()) {
    return std::nullopt;
  }

  const auto& values = session.ir().values();
  CompiledBackboneExecutionPlan plan;
  bool found_input = false;
  for (const auto& value : values) {
    if (
        value.spec.role == utils::VulkanIRTensorRole::Input &&
        value.spec.external && !value.spec.logical_sizes.empty()) {
      plan.input_value = value.id;
      found_input = true;
      break;
    }
  }
  if (!found_input) {
    return std::nullopt;
  }
  plan.backbone_input_value = plan.input_value;

  plan.block_output_values.reserve(block_count);
  plan.capture_values_by_block.resize(block_count);

  std::optional<size_t> current_block = std::nullopt;
  for (const auto& op : session.ir().ops()) {
    switch (op.kind) {
      case utils::VulkanIROpKind::FeatureMapToTokens:
      case utils::VulkanIROpKind::ElementwiseAdd:
      case utils::VulkanIROpKind::Concat:
      case utils::VulkanIROpKind::PatchTokenInput: {
        if (op.outputs.size() != 1u) {
          return std::nullopt;
        }
        plan.backbone_input_value = op.outputs[0];
        break;
      }
      case utils::VulkanIROpKind::BackboneBlock: {
        if (op.outputs.size() != 1u) {
          return std::nullopt;
        }
        current_block = plan.block_output_values.size();
        plan.block_output_values.push_back(op.outputs[0]);
        break;
      }
      case utils::VulkanIROpKind::CapturePatchTokens:
      case utils::VulkanIROpKind::CaptureNormedPatchTokens: {
        if (
            !current_block.has_value() ||
            *current_block >= plan.capture_values_by_block.size() ||
            op.outputs.size() != 1u) {
          return std::nullopt;
        }
        plan.capture_values_by_block[*current_block] = op.outputs[0];
        break;
      }
      default:
        break;
    }
  }

  if (plan.block_output_values.size() != block_count) {
    return std::nullopt;
  }
  return plan;
}

std::optional<CompiledImageEntryExecutionPlan>
make_compiled_image_entry_execution_plan(
    const utils::VulkanCompiledSession& session) {
  if (!session.defined() || !session.executable()) {
    return std::nullopt;
  }

  CompiledImageEntryExecutionPlan plan;
  bool found_patch_embed = false;
  bool found_feature_map_tokens = false;
  bool found_positioned_tokens = false;
  bool found_patch_token_input = false;
  bool found_concat_tokens = false;

  for (const auto& op : session.ir().ops()) {
    switch (op.kind) {
      case utils::VulkanIROpKind::PatchEmbed: {
        if (
            found_patch_embed || op.inputs.size() != 1u ||
            op.outputs.size() != 1u) {
          return std::nullopt;
        }
        plan.image_input_value = op.inputs[0];
        plan.patch_feature_map_value = op.outputs[0];
        found_patch_embed = true;
        break;
      }
      case utils::VulkanIROpKind::FeatureMapToTokens: {
        if (
            !found_patch_embed || found_feature_map_tokens ||
            op.inputs.size() != 1u || op.outputs.size() != 1u) {
          return std::nullopt;
        }
        plan.patch_feature_map_value = op.inputs[0];
        plan.patch_feature_tokens_value = op.outputs[0];
        found_feature_map_tokens = true;
        break;
      }
      case utils::VulkanIROpKind::ElementwiseAdd: {
        if (
            !found_feature_map_tokens || found_positioned_tokens ||
            op.inputs.size() != 2u || op.outputs.size() != 1u) {
          return std::nullopt;
        }
        plan.patch_feature_tokens_value = op.inputs[0];
        plan.patch_pos_encoding_value = op.inputs[1];
        plan.positioned_patch_tokens_value = op.outputs[0];
        found_positioned_tokens = true;
        break;
      }
      case utils::VulkanIROpKind::Concat: {
        if (
            !found_positioned_tokens || found_concat_tokens ||
            op.inputs.size() != 2u || op.outputs.size() != 1u) {
          return std::nullopt;
        }
        plan.prefix_token_value = op.inputs[0];
        plan.positioned_patch_tokens_value = op.inputs[1];
        plan.patch_token_value = op.outputs[0];
        found_concat_tokens = true;
        break;
      }
      case utils::VulkanIROpKind::PatchTokenInput: {
        if (
            found_patch_token_input || op.inputs.size() != 1u ||
            op.outputs.size() != 1u || op.constants.size() != 2u) {
          return std::nullopt;
        }
        plan.patch_feature_map_value = op.inputs[0];
        plan.patch_token_value = op.outputs[0];
        plan.prefix_token_value = op.constants[0];
        plan.patch_pos_encoding_value = op.constants[1];
        found_patch_token_input = true;
        break;
      }
      default:
        break;
    }
  }

  const bool found_token_path = found_concat_tokens || found_patch_token_input;
  if (!found_patch_embed || !found_token_path) {
    return std::nullopt;
  }
  return plan;
}

std::optional<CompiledDecoderExecutionPlan> make_compiled_decoder_execution_plan(
    const utils::VulkanCompiledSession& session) {
  if (!session.defined() || !session.executable()) {
    return std::nullopt;
  }

  CompiledDecoderExecutionPlan plan;
  bool found_head = false;
  bool found_output = false;

  for (const auto& op : session.ir().ops()) {
    if (op.kind != utils::VulkanIROpKind::DecoderHead) {
      continue;
    }
    if (found_head || op.inputs.size() != 4u || op.outputs.size() != 1u) {
      return std::nullopt;
    }
    std::copy(
        op.inputs.cbegin(), op.inputs.cend(), plan.head_input_values.begin());
    plan.head_output_value = op.outputs[0];
    found_head = true;
  }

  if (!found_head) {
    return std::nullopt;
  }

  for (const auto& value : session.ir().values()) {
    if (
        value.spec.role == utils::VulkanIRTensorRole::Output &&
        value.spec.external && !value.spec.logical_sizes.empty()) {
      plan.final_output_value = value.id;
      found_output = true;
      break;
    }
  }

  if (!found_output) {
    plan.final_output_value = plan.head_output_value;
  }
  return plan;
}

bool has_compiled_decoder_head_shape(
    const utils::VulkanCompiledSession& session,
    const CompiledDecoderExecutionPlan& plan,
    const c10::intrusive_ptr<VisionDecoderHeadContext>& head_context) {
  if (!session.defined() || !session.executable() || !head_context) {
    return false;
  }

  const auto& values = session.ir().values();
  const auto get_sizes = [&](const utils::VulkanValueId value_id)
      -> std::optional<std::vector<int64_t>> {
    if (value_id >= values.size()) {
      return std::nullopt;
    }
    const auto& sizes = values[value_id].spec.logical_sizes;
    if (sizes.size() != 4u) {
      return std::nullopt;
    }
    return sizes;
  };

  const auto layer1_sizes = get_sizes(plan.head_input_values[0]);
  const auto output_sizes = get_sizes(plan.final_output_value);
  if (!layer1_sizes.has_value() || !output_sizes.has_value()) {
    return false;
  }

  return can_run_depth_anything_v2_head_fusion_shape(
      std::vector<int64_t>{
          layer1_sizes->at(0),
          layer1_sizes->at(1),
          layer1_sizes->at(2) * 2,
          layer1_sizes->at(3) * 2},
      std::vector<int64_t>{output_sizes->at(2), output_sizes->at(3)},
      head_context);
}

std::optional<std::shared_ptr<CompiledDecoderHeadPrograms>>
make_compiled_decoder_head_programs(
    const utils::VulkanCompiledSession& session,
    const CompiledDecoderExecutionPlan& plan,
    const c10::intrusive_ptr<VisionDecoderHeadContext>& head_context,
    const utils::VulkanExecutionProgramPlanningDesc& program_plan) {
  if (!session.defined() || !session.executable() || !head_context) {
    return std::nullopt;
  }
  if (!has_compiled_decoder_head_shape(session, plan, head_context)) {
    return std::nullopt;
  }

  const auto& values = session.ir().values();
  const auto get_sizes = [&](const utils::VulkanValueId value_id)
      -> std::optional<std::vector<int64_t>> {
    if (value_id >= values.size()) {
      return std::nullopt;
    }
    const auto& sizes = values[value_id].spec.logical_sizes;
    if (sizes.size() != 4u) {
      return std::nullopt;
    }
    return sizes;
  };

  const auto layer1_sizes = get_sizes(plan.head_input_values[0]);
  const auto layer2_sizes = get_sizes(plan.head_input_values[1]);
  const auto layer3_sizes = get_sizes(plan.head_input_values[2]);
  const auto layer4_sizes = get_sizes(plan.head_input_values[3]);
  if (
      !layer1_sizes.has_value() || !layer2_sizes.has_value() ||
      !layer3_sizes.has_value() || !layer4_sizes.has_value()) {
    return std::nullopt;
  }

  auto decoder_graph = utils::lookup_or_create_labeled_vision_decoder_inference_graph(
      vision_decoder_graph_label(head_context->allocation_label()),
      kFloat,
      program_plan.persistent);
  if (!decoder_graph.defined()) {
    return std::nullopt;
  }
  auto programs = std::make_shared<CompiledDecoderHeadPrograms>();
  programs->programs[0] = decoder_graph.lookup_or_create_program(
      vision_decoder_head_program_label(
          head_context->allocation_label(), head_context.get()) +
          ".refinenet4.program",
      *layer4_sizes,
      std::nullopt,
      std::vector<int64_t>{layer3_sizes->at(2), layer3_sizes->at(3)},
      layer3_sizes->at(1),
      /*allocate_intermediate_outputs=*/true,
      program_plan);
  programs->programs[1] = decoder_graph.lookup_or_create_program(
      vision_decoder_head_program_label(
          head_context->allocation_label(), head_context.get()) +
          ".refinenet3.program",
      *layer3_sizes,
      *layer3_sizes,
      std::vector<int64_t>{layer2_sizes->at(2), layer2_sizes->at(3)},
      layer2_sizes->at(1),
      /*allocate_intermediate_outputs=*/true,
      program_plan);
  programs->programs[2] = decoder_graph.lookup_or_create_program(
      vision_decoder_head_program_label(
          head_context->allocation_label(), head_context.get()) +
          ".refinenet2.program",
      *layer2_sizes,
      *layer2_sizes,
      std::vector<int64_t>{layer1_sizes->at(2), layer1_sizes->at(3)},
      layer1_sizes->at(1),
      /*allocate_intermediate_outputs=*/true,
      program_plan);
  programs->programs[3] = decoder_graph.lookup_or_create_program(
      vision_decoder_head_program_label(
          head_context->allocation_label(), head_context.get()) +
          ".refinenet1.program",
      *layer1_sizes,
      *layer1_sizes,
      std::vector<int64_t>{layer1_sizes->at(2) * 2, layer1_sizes->at(3) * 2},
      layer1_sizes->at(1),
      /*allocate_intermediate_outputs=*/true,
      program_plan);
  if (!programs->defined()) {
    return std::nullopt;
  }
  return programs;
}

struct CompiledExecutableBackboneRegionContext final {
  const utils::VulkanExecutableRegion* executable_region{nullptr};
  std::vector<Tensor>& tensor_slots;
  const utils::VulkanCompiledSessionTensorBindings& bindings;
  std::shared_ptr<std::vector<Tensor>> capture_output_slots;
  size_t capture_output_slot_base{0u};
  std::vector<std::optional<size_t>> capture_output_slots_by_value;
  c10::intrusive_ptr<Conv2dPackedContext> patch_embed_context;
  const std::vector<c10::intrusive_ptr<VisionBackboneBlockContext>>&
      backbone_contexts;
  std::vector<utils::VisionBackboneProgram>& backbone_programs;
  std::vector<std::optional<utils::ScratchArena>>& graph_scratches;
  std::vector<int64_t> normalized_shape;
  c10::intrusive_ptr<LayernormPackedContext> output_norm_context;
  std::array<c10::intrusive_ptr<Conv2dPackedContext>, 4u> project_contexts{};
  std::array<c10::intrusive_ptr<Conv2dPackedContext>, 4u> resize_contexts{};
  std::array<c10::intrusive_ptr<Conv2dPackedContext>, 4u> rn_contexts{};
  c10::intrusive_ptr<VisionDecoderHeadContext> head_context;
  std::shared_ptr<CompiledDecoderHeadPrograms> head_programs;
  std::vector<int64_t> output_size;
  int64_t patch_h{0};
  int64_t patch_w{0};
  int64_t special_token_count{0};
  size_t decoder_project_index{0u};
  size_t decoder_layer_index{0u};
  size_t backbone_step_index{0u};
  size_t capture_step_index{0u};
  bool ran_decoder_head{false};
};

bool run_compiled_executable_region(
    CompiledExecutableBackboneRegionContext& context,
    const utils::VulkanExecutableRegion& executable_region,
    bool include_image_entry);

void run_recorded_compiled_replay_or_direct_steps(
    const utils::ExecutionGraphReplayBundle& replay_bundle,
    const char* op_name,
    const char* allocation_label);

Tensor copy_compiled_session_output(
    const utils::ExecutionGraphReplayBundle& replay_bundle,
    const size_t output_slot_idx) {
  const Tensor& output_slot = replay_bundle.tensor_slot(output_slot_idx);
  (void)utils::stamp_replay_export(
      output_slot,
      replay_bundle.identity(),
      static_cast<uint32_t>(output_slot_idx),
      "copy_compiled_session_output");
  Tensor output = utils::create_buffer_tensor(
      output_slot.sizes(), output_slot.scalar_type(), /*persistent=*/false);
  copy_tensor_for_replay(output, output_slot);
  record_tensor_write(
      output,
      "copy_compiled_session_output",
      "materialized_replay_export",
      {output_slot});
  utils::log_replay_event(
      "materialize_export",
      replay_bundle.identity(),
      utils::current_replay_epoch(replay_bundle.identity()).run_id,
      "compiled_session_output",
      "action=materialize_escaping_output");
  return output;
}

Tensor run_vision_decoder_preprocess_head_fallback(
    const std::array<Tensor, 4u>& layer_tokens,
    const Device& output_device,
    const ScalarType output_dtype,
    const int64_t patch_h,
    const int64_t patch_w,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderPreprocessHeadContext>& context) {
  Tensor layer1 = tokens_to_feature_map(layer_tokens[0], patch_h, patch_w);
  layer1 = run_conv2d_context(layer1, context->project1_context());
  layer1 = run_tconv2d_context(layer1, context->resize1_context());
  layer1 = run_conv2d_context(layer1, context->layer1_rn_context());

  Tensor layer2 = tokens_to_feature_map(layer_tokens[1], patch_h, patch_w);
  layer2 = run_conv2d_context(layer2, context->project2_context());
  layer2 = run_tconv2d_context(layer2, context->resize2_context());
  layer2 = run_conv2d_context(layer2, context->layer2_rn_context());

  Tensor layer3 = tokens_to_feature_map(layer_tokens[2], patch_h, patch_w);
  layer3 = run_conv2d_context(layer3, context->project3_context());
  layer3 = run_conv2d_context(layer3, context->layer3_rn_context());

  Tensor layer4 = tokens_to_feature_map(layer_tokens[3], patch_h, patch_w);
  layer4 = run_conv2d_context(layer4, context->project4_context());
  layer4 = run_tconv2d_context(layer4, context->resize4_context());
  layer4 = run_conv2d_context(layer4, context->layer4_rn_context());

  Tensor output = run_vision_decoder_head_context(
      layer1, layer2, layer3, layer4, output_size, context->head_context());
  return maybe_restore_tensor(output, output_device, output_dtype);
}

std::optional<std::vector<Tensor>> try_run_vision_backbone_stack_compiled_session(
    const Tensor& input,
    const Device& output_device,
    const ScalarType output_dtype,
    const std::vector<c10::intrusive_ptr<VisionBackboneBlockContext>>&
        backbone_contexts,
    const std::vector<int64_t>& capture_indices_vec,
    const std::optional<std::vector<int64_t>>& output_norm_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& output_norm_context,
    const utils::VulkanRuntimePolicy& runtime_policy,
    const VisionReplayBundleIdentity& bundle_identity,
    const std::string& root_label,
    const utils::VulkanCompiledSession& compiled_session) {
  const bool apply_output_norm =
      output_norm_shape.has_value() && static_cast<bool>(output_norm_context);
  const auto execution_plan = make_compiled_backbone_execution_plan(
      compiled_session,
      backbone_contexts.size());
  const auto* executable_region = compiled_session.executable_region();
  if (!execution_plan.has_value() || !executable_region ||
      !executable_region->defined()) {
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_backbone_stack_compiled_session.skip.no_plan");
    return std::nullopt;
  }
  const auto bindings = utils::make_compiled_executable_region_tensor_bindings(
      compiled_session, *executable_region);
  if (
      !bindings.has_value() ||
      !execution_plan.has_value() || !executable_region ||
      !executable_region->defined()) {
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_backbone_stack_compiled_session.skip.no_bindings");
    return std::nullopt;
  }
  const auto compiled_bindings = *bindings;
  const auto compiled_plan = *execution_plan;

  const auto& program_plan = *runtime_policy.execution_program_plan;
  const int64_t batch_size = input.dim() == 2 ? 1 : input.size(0);
  const int64_t token_count = input.dim() == 2 ? input.size(0) : input.size(1);
  const int64_t embed_dim = input.size(-1);
  const uint32_t scratch_alignment =
      runtime_policy.scratch_arena_plan.has_value()
      ? std::max<uint32_t>(
            runtime_policy.scratch_arena_plan->alignment,
            static_cast<uint32_t>(std::max<int64_t>(
                1, static_cast<int64_t>(c10::elementSize(kFloat)))))
      : 1u;

  auto root = utils::lookup_or_create_labeled_execution_graph_root(
      root_label,
      kFloat,
      program_plan.persistent);
  const std::string compiled_bundle_key =
      bundle_identity.key + "|compiled_executable_region_v1";
  auto replay_bundle = root.lookup_or_create_replay_bundle(
      compiled_bundle_key,
      [&]() -> utils::ExecutionGraphReplayBundle {
        auto programs =
            std::make_shared<std::vector<utils::VisionBackboneProgram>>();
        auto graph_scratches =
            std::make_shared<std::vector<std::optional<utils::ScratchArena>>>();
        programs->reserve(backbone_contexts.size());
        graph_scratches->reserve(backbone_contexts.size());
        for (const auto& context : backbone_contexts) {
          auto vision_graph =
              prime_vision_backbone_graph(input, runtime_policy, context);
          if (!vision_graph.defined()) {
            utils::log_vulkan_op_hit(
                "vulkan_prepack::run_vision_backbone_stack_compiled_session.skip.no_graph");
            return {};
          }

          std::optional<utils::ScratchArena> graph_scratch = std::nullopt;
          if (
              runtime_policy.scratch_arena_plan.has_value() &&
              runtime_policy.scratch_arena_plan->prefer_buffer_storage) {
            const size_t requested_bytes = vision_attention_scratch_bytes(
                batch_size,
                token_count,
                embed_dim,
                context->num_heads(),
                input.scalar_type(),
                context->qkv_bias().defined(),
                scratch_alignment);
            if (requested_bytes > 0u) {
              graph_scratch = vision_graph.ensure_shared_scratch(
                  std::max(
                      requested_bytes,
                      runtime_policy.scratch_arena_plan->min_arena_bytes),
                  scratch_alignment,
                  program_plan.persistent);
            }
          }

          const int64_t hidden_dim = vision_block_hidden_dim(context);
          utils::VisionBackboneProgram program =
              vision_graph.lookup_or_create_program(
                  vision_backbone_program_label(
                      context->allocation_label(), context.get()) +
                      ".compiled_session",
                  input.scalar_type(),
                  batch_size,
                  token_count,
                  embed_dim,
                  hidden_dim,
                  context->num_heads(),
                  program_plan);
          if (!program.defined()) {
            utils::log_vulkan_op_hit(
                "vulkan_prepack::run_vision_backbone_stack_compiled_session.skip.no_program");
            return {};
          }
          programs->push_back(std::move(program));
          graph_scratches->push_back(std::move(graph_scratch));
        }

        auto tensor_slots = make_compiled_session_tensor_slots(
            compiled_session,
            compiled_bindings,
            program_plan.persistent);
        const size_t capture_output_slot_base = tensor_slots->size();
        std::vector<std::optional<size_t>> capture_output_slots_by_value(
            compiled_bindings.value_tensor_slots.size());
        for (size_t capture_pos = 0u; capture_pos < capture_indices_vec.size();
             ++capture_pos) {
          tensor_slots->push_back(utils::create_buffer_tensor(
              input.sizes(),
              output_dtype,
              /*persistent=*/true));
          const int64_t replay_idx = capture_indices_vec[capture_pos];
          TORCH_INTERNAL_ASSERT(
              replay_idx >= 0 &&
                  replay_idx <
                      static_cast<int64_t>(compiled_plan.capture_values_by_block.size()) &&
                  compiled_plan.capture_values_by_block
                      .at(static_cast<size_t>(replay_idx))
                      .has_value(),
              "Compiled backbone stack expected a captured output for every "
              "requested capture index");
          capture_output_slots_by_value[*compiled_plan.capture_values_by_block.at(
              static_cast<size_t>(replay_idx))] =
              capture_output_slot_base + capture_pos;
        }
        std::vector<utils::ExecutionGraphReplayStep> steps;
        steps.reserve(1u);
        auto executable_step_replay = utils::make_execution_graph_replay(
            root.allocation_label() +
                ".vision.backbone_stack.compiled.executable_region.step" +
                bundle_identity.label_suffix,
            utils::VulkanInferenceGraphKind::VisionBackbone,
            kFloat,
            program_plan.persistent,
            std::vector<Tensor>{},
            std::vector<std::optional<Tensor>>{},
            std::vector<utils::ExecutionGraphProgramHandle>{});
        steps.push_back(utils::make_execution_graph_replay_step(
            std::move(executable_step_replay),
            [tensor_slots,
             compiled_bindings,
             programs,
             graph_scratches,
             backbone_contexts,
             capture_indices_vec,
             compiled_plan,
             capture_output_slot_base,
             normalized_shape_vec =
                 output_norm_shape.value_or(std::vector<int64_t>{}),
             output_norm_context]() mutable {
              const size_t input_slot_idx = compiled_session_tensor_slot(
                  compiled_bindings, compiled_plan.input_value);
              Tensor current = tensor_slots->at(input_slot_idx);
              for (size_t idx = 0u; idx < backbone_contexts.size(); ++idx) {
                auto& graph_scratch = (*graph_scratches)[idx];
                if (graph_scratch.has_value()) {
                  graph_scratch->reset();
                }
                api::RuntimeLabelScope runtime_scope(
                    compose_runtime_capture_label(
                        vision_backbone_execution_label(
                            backbone_contexts[idx]->allocation_label(),
                            backbone_contexts[idx].get()) +
                        ".compiled_session"));
                Tensor& output_slot = tensor_slots->at(compiled_session_tensor_slot(
                    compiled_bindings,
                    compiled_plan.block_output_values[idx]));
                (void)run_vision_backbone_block_program(
                    current,
                    backbone_contexts[idx],
                    &(*programs)[idx],
                    graph_scratch.has_value() ? &(*graph_scratch) : nullptr,
                    &output_slot);
                for (size_t capture_pos = 0u; capture_pos < capture_indices_vec.size();
                     ++capture_pos) {
                  if (capture_indices_vec[capture_pos] != static_cast<int64_t>(idx)) {
                    continue;
                  }
                  Tensor& capture_slot =
                      tensor_slots->at(capture_output_slot_base + capture_pos);
                  if (!normalized_shape_vec.empty() && output_norm_context) {
                    (void)run_layernorm_context_out(
                        output_slot,
                        normalized_shape_vec,
                        output_norm_context,
                        capture_slot);
                  } else {
                    copy_tensor_for_replay(capture_slot, output_slot);
                  }
                }
                current = output_slot;
              }
            }));
        return utils::make_execution_graph_replay_bundle(
            root.allocation_label() +
                ".vision.backbone_stack.compiled_session.replay" +
                bundle_identity.label_suffix,
            kFloat,
            program_plan.persistent,
            std::move(steps),
            std::move(tensor_slots));
      });

  if (!replay_bundle.defined() ||
      replay_bundle.tensor_slot_count() <
          compiled_bindings.tensor_slot_count() + capture_indices_vec.size()) {
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_backbone_stack_compiled_session.skip.no_replay_bundle");
    return std::nullopt;
  }

  copy_tensor_for_replay(
      replay_bundle.tensor_slot(
          compiled_session_tensor_slot(compiled_bindings, compiled_plan.input_value)),
      input);
  api::context()->flush_pending_cmds();

  const bool first_record = !replay_bundle.recorded();
  if (first_record) {
    replay_bundle.warmup();
    api::context()->flush_pending_cmds();
    replay_bundle.record_steps_individually();
  }
  run_recorded_compiled_replay_or_direct_steps(
      replay_bundle,
      apply_output_norm
          ? "vulkan_prepack::run_vision_backbone_stack_norm_compiled_session"
          : "vulkan_prepack::run_vision_backbone_stack_compiled_session",
      "vision.backbone_stack.compiled_session.replay");

  std::vector<Tensor> outputs(capture_indices_vec.size());
  const size_t capture_output_slot_base = compiled_bindings.tensor_slot_count();
  for (size_t capture_pos = 0u; capture_pos < capture_indices_vec.size();
       ++capture_pos) {
    Tensor output = copy_compiled_session_output(
        replay_bundle, capture_output_slot_base + capture_pos);
    outputs[capture_pos] =
        maybe_restore_tensor(output, output_device, output_dtype);
  }

  utils::log_vulkan_op_hit(
      apply_output_norm
          ? (first_record
                 ? "vulkan_prepack::run_vision_backbone_stack_norm_compiled_session.replay_warmup"
                 : "vulkan_prepack::run_vision_backbone_stack_norm_compiled_session.replay")
          : (first_record
                 ? "vulkan_prepack::run_vision_backbone_stack_compiled_session.replay_warmup"
                 : "vulkan_prepack::run_vision_backbone_stack_compiled_session.replay"));
  utils::log_vulkan_op_hit(
      apply_output_norm
          ? "vulkan_prepack::run_vision_backbone_stack_norm_compiled_session"
          : "vulkan_prepack::run_vision_backbone_stack_compiled_session");
  return outputs;
}

bool run_tokens_to_feature_map_direct_out(
    const Tensor& input_arg,
    const int64_t height,
    const int64_t width,
    Tensor& output) {
  if (
      !input_arg.is_vulkan() || input_arg.scalar_type() != kFloat ||
      !output.defined() || !output.is_vulkan() || output.scalar_type() != kFloat) {
    return false;
  }

  api::AllocationScope allocation_scope("tokens_to_feature_map");

  Tensor input = input_arg;
  const bool use_2d_input = input.dim() == 2;
  const int64_t batch_size = use_2d_input ? 1 : input.size(0);
  const int64_t token_count = use_2d_input ? input.size(0) : input.size(1);
  const int64_t channels = input.size(-1);

  TORCH_CHECK(
      input.dim() == 2 || input.dim() == 3,
      "Vulkan tokens_to_feature_map expects a [N, C] or [B, N, C] tensor");
  TORCH_CHECK(
      token_count == height * width,
      "Vulkan tokens_to_feature_map expected token count ",
      height * width,
      " but received ",
      token_count);

  const std::vector<int64_t> output_sizes{
      batch_size,
      channels,
      height,
      width,
  };
  TORCH_CHECK(
      output.sizes().vec() == output_sizes,
      "Vulkan tokens_to_feature_map_out expected output shape [",
      batch_size,
      ", ",
      channels,
      ", ",
      height,
      ", ",
      width,
      "]");

  utils::log_vulkan_op_hit("aten::tokens_to_feature_map");

  const vTensor& v_input_probe = convert(input);
  vTensor& v_output = convert(output);
  TORCH_CHECK(
      v_output.storage_type() == api::StorageType::BUFFER,
      "Vulkan tokens_to_feature_map_out expects a buffer output tensor");

  if (
      v_input_probe.storage_type() == api::StorageType::TEXTURE_3D &&
      v_input_probe.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED &&
      batch_size == 1) {
    api::PipelineBarrier pipeline_barrier{};
    const api::utils::uvec3 global_size{
        api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
        1u,
        1u,
    };
    api::UniformParamsBuffer out_meta =
        utils::make_buffer_compute_metadata_ubo(api::context(), v_output);

    api::context()->submit_compute_job(
        VK_KERNEL(tokens_to_feature_map_texture_to_buffer),
        pipeline_barrier,
        global_size,
        adaptive_work_group_size(global_size),
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        out_meta.buffer(),
        v_input_probe.image(pipeline_barrier, api::PipelineStage::COMPUTE));

    utils::log_vulkan_op_hit("aten::tokens_to_feature_map.texture_to_buffer");
    return true;
  }

  if (
      v_input_probe.storage_type() != api::StorageType::BUFFER ||
      !utils::supports_buffer_elementwise_compute(v_input_probe)) {
    return false;
  }

  api::UniformParamsBuffer input_meta =
      utils::make_buffer_compute_metadata_ubo(api::context(), v_input_probe);
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(api::context(), v_output);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
      1u,
      1u,
  };

  api::context()->submit_compute_job(
      VK_KERNEL(tokens_to_feature_map_buffer),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_input_probe.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      input_meta.buffer());

  utils::log_vulkan_op_hit("aten::tokens_to_feature_map.buffer_to_buffer");
  return true;
}

bool run_feature_map_to_tokens_direct_out(
    const Tensor& input_arg,
    Tensor& output) {
  if (
      !input_arg.is_vulkan() || input_arg.scalar_type() != kFloat ||
      !output.defined() || !output.is_vulkan() || output.scalar_type() != kFloat) {
    return false;
  }

  TORCH_CHECK(
      input_arg.dim() == 4,
      "Vulkan feature_map_to_tokens expects a [B, C, H, W] tensor");

  api::AllocationScope allocation_scope("feature_map_to_tokens");
  utils::log_vulkan_op_hit("aten::feature_map_to_tokens");

  const std::vector<int64_t> output_sizes{
      input_arg.size(0),
      input_arg.size(2) * input_arg.size(3),
      input_arg.size(1),
  };
  TORCH_CHECK(
      output.sizes().vec() == output_sizes,
      "Vulkan feature_map_to_tokens_out expected output shape [",
      output_sizes[0],
      ", ",
      output_sizes[1],
      ", ",
      output_sizes[2],
      "]");

  const vTensor& v_input = convert(input_arg);
  vTensor& v_output = convert(output);
  TORCH_CHECK(
      v_output.storage_type() == api::StorageType::BUFFER,
      "Vulkan feature_map_to_tokens_out expects a buffer output tensor");

  if (
      v_input.storage_type() == api::StorageType::TEXTURE_3D &&
      v_input.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED &&
      input_arg.size(0) == 1) {
    const struct Block final {
      api::utils::ivec4 info;
    } block{
        {
            api::utils::safe_downcast<int32_t>(input_arg.size(3)),
            api::utils::safe_downcast<int32_t>(input_arg.size(2)),
            api::utils::safe_downcast<int32_t>(input_arg.size(1)),
            api::utils::safe_downcast<int32_t>(input_arg.size(0)),
        },
    };

    api::UniformParamsBuffer params(api::context(), block);
    api::PipelineBarrier pipeline_barrier{};
    const api::utils::uvec3 global_size{
        api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
        1u,
        1u,
    };
    api::UniformParamsBuffer out_meta =
        utils::make_buffer_compute_metadata_ubo(api::context(), v_output);

    api::context()->submit_compute_job(
        VK_KERNEL(feature_map_to_tokens_texture_to_buffer),
        pipeline_barrier,
        global_size,
        adaptive_work_group_size(global_size),
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        out_meta.buffer(),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());

    utils::log_vulkan_op_hit("aten::feature_map_to_tokens.texture_to_buffer");
    return true;
  }

  if (
      v_input.storage_type() != api::StorageType::BUFFER ||
      !utils::supports_buffer_elementwise_compute(v_input)) {
    return false;
  }

  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
      1u,
      1u,
  };
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(api::context(), v_output);
  api::UniformParamsBuffer input_meta =
      utils::make_buffer_compute_metadata_ubo(api::context(), v_input);

  api::context()->submit_compute_job(
      VK_KERNEL(feature_map_to_tokens_buffer),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      input_meta.buffer());

  utils::log_vulkan_op_hit("aten::feature_map_to_tokens.buffer_to_buffer");
  return true;
}

Tensor assemble_depth_anything_v2_tokens_from_feature_map(
    const Tensor& patch_feature_map_arg,
    const Tensor& prefix_token_arg,
    const Tensor& patch_pos_encoding_arg,
    const bool flatten_batch1_tokens) {
  TORCH_CHECK(
      patch_feature_map_arg.defined() && patch_feature_map_arg.is_vulkan() &&
          prefix_token_arg.defined() && patch_pos_encoding_arg.defined(),
      "Depth Anything image entry expects defined Vulkan patch features and "
      "defined token constants");

  Tensor patch_feature_map = prepare_decoder_buffer_tensor(patch_feature_map_arg);
  Tensor patch_tokens = feature_map_to_tokens(patch_feature_map);
  TORCH_CHECK(
      patch_tokens.is_vulkan() && patch_tokens.dim() == 3,
      "Depth Anything image entry expects rank-3 Vulkan patch tokens");
  patch_tokens = prepare_buffer_attention_tensor(patch_tokens);

  Tensor prefix_token =
      prefix_token_arg.is_vulkan() ? prefix_token_arg : prefix_token_arg.vulkan();
  Tensor patch_pos_encoding = patch_pos_encoding_arg.is_vulkan()
      ? patch_pos_encoding_arg
      : patch_pos_encoding_arg.vulkan();
  if (prefix_token.scalar_type() != patch_tokens.scalar_type()) {
    prefix_token = prefix_token.to(patch_tokens.scalar_type());
  }
  if (patch_pos_encoding.scalar_type() != patch_tokens.scalar_type()) {
    patch_pos_encoding = patch_pos_encoding.to(patch_tokens.scalar_type());
  }
  prefix_token = prepare_buffer_attention_tensor(prefix_token);
  patch_pos_encoding = prepare_buffer_attention_tensor(patch_pos_encoding);

  TORCH_CHECK(
      prefix_token.dim() == 3 && patch_pos_encoding.dim() == 3,
      "Depth Anything image entry expects rank-3 prefix and positional "
      "encoding tensors");
  TORCH_CHECK(
      prefix_token.size(2) == patch_tokens.size(2) &&
          patch_pos_encoding.size(1) == patch_tokens.size(1) &&
          patch_pos_encoding.size(2) == patch_tokens.size(2),
      "Depth Anything image entry received mismatched token constant shapes");

  Tensor positioned_tokens = at::add(patch_tokens, patch_pos_encoding);
  TORCH_CHECK(
      positioned_tokens.dim() == 3,
      "Depth Anything image entry expected rank-3 positioned patch tokens");

  Tensor prefix = prefix_token;
  if (prefix.size(0) == 1 && positioned_tokens.size(0) != 1) {
    prefix = prefix.expand(
        {positioned_tokens.size(0), prefix.size(1), prefix.size(2)});
  }
  TORCH_CHECK(
      prefix.size(0) == positioned_tokens.size(0),
      "Depth Anything image entry received incompatible prefix batch size");

  Tensor full_tokens = at::cat({prefix, positioned_tokens}, 1);
  full_tokens = prepare_buffer_attention_tensor(full_tokens);
  if (flatten_batch1_tokens && full_tokens.dim() == 3 && full_tokens.size(0) == 1) {
    full_tokens = full_tokens.reshape({full_tokens.size(1), full_tokens.size(2)});
    full_tokens = prepare_buffer_attention_tensor(full_tokens);
  }
  return full_tokens;
}

Tensor make_depth_anything_v2_tokens_from_image(
    const Tensor& image_arg,
    const c10::intrusive_ptr<Conv2dPackedContext>& patch_embed_context,
    const Tensor& prefix_token_arg,
    const Tensor& patch_pos_encoding_arg,
    const bool flatten_batch1_tokens) {
  TORCH_CHECK(
      static_cast<bool>(patch_embed_context),
      "Depth Anything image entry expects a defined patch embed context");
  TORCH_CHECK(
      image_arg.dim() == 4,
      "Depth Anything image entry expects a rank-4 image tensor");
  Tensor image = image_arg.is_vulkan() ? image_arg : image_arg.vulkan();
  Tensor patch_feature_map = run_conv2d_context(image, patch_embed_context);
  return assemble_depth_anything_v2_tokens_from_feature_map(
      patch_feature_map,
      prefix_token_arg,
      patch_pos_encoding_arg,
      flatten_batch1_tokens);
}

Tensor& compiled_executable_tensor_slot(
    std::vector<Tensor>& tensor_slots,
    const utils::VulkanCompiledSessionTensorBindings& bindings,
    const utils::VulkanValueId value_id) {
  return tensor_slots.at(compiled_session_tensor_slot(bindings, value_id));
}

const Tensor& compiled_executable_tensor_slot(
    const std::vector<Tensor>& tensor_slots,
    const utils::VulkanCompiledSessionTensorBindings& bindings,
    const utils::VulkanValueId value_id) {
  return tensor_slots.at(compiled_session_tensor_slot(bindings, value_id));
}

Tensor maybe_expand_compiled_region_batch(
    Tensor tensor,
    const Tensor& like_tensor) {
  if (
      tensor.dim() == 3 && like_tensor.dim() == 3 && tensor.size(0) == 1 &&
      like_tensor.size(0) != 1) {
    tensor = tensor.expand(
        {like_tensor.size(0), tensor.size(1), tensor.size(2)});
  }
  return tensor;
}

Tensor prepare_compiled_executable_decoder_tokens(
    const Tensor& tokens,
    const int64_t special_token_count) {
  Tensor prepared = tokens;
  if (special_token_count > 0) {
    prepared = prepared.dim() == 2
        ? prepared.slice(0, special_token_count, prepared.size(0))
        : prepared.slice(1, special_token_count, prepared.size(1));
  }
  return prepare_buffer_attention_tensor(prepared);
}

size_t compiled_executable_expected_decoder_layers(
    const CompiledExecutableBackboneRegionContext& context) {
  return static_cast<size_t>(std::count_if(
      context.rn_contexts.cbegin(),
      context.rn_contexts.cend(),
      [](const auto& value) {
        return static_cast<bool>(value);
      }));
}

const utils::LoweredValue& compiled_executable_lowered_value(
    const CompiledExecutableBackboneRegionContext& context,
    const utils::VulkanValueId value_id) {
  TORCH_INTERNAL_ASSERT(
      context.executable_region &&
          value_id < context.executable_region->values.size(),
      "Executable region value lookup references an invalid value id");
  return context.executable_region->values[value_id];
}

Tensor create_compiled_executable_virtual_tensor(
    const CompiledExecutableBackboneRegionContext& context,
    const utils::VulkanValueId value_id) {
  const auto& value = compiled_executable_lowered_value(context, value_id);
  TORCH_INTERNAL_ASSERT(
      value.realization == utils::RealizationKind::Virtual,
      "Executable region temporary tensor expects a virtual lowered value");
  return utils::create_buffer_tensor(
      value.view.logical_sizes, value.view.logical_dtype, /*persistent=*/false);
}

const char* executable_region_dispatch_kind_name(
    const utils::DispatchKind kind) {
  return utils::dispatch_kind_name(kind);
}

std::string executable_region_dispatch_diagnostic_name(
    const utils::DispatchStep& step) {
  if (step.dispatch_kind != utils::DispatchKind::Unknown) {
    return std::string(executable_region_dispatch_kind_name(step.dispatch_kind));
  }
  if (!step.program_key.empty()) {
    return step.program_key;
  }
  return "Dispatch";
}

std::string executable_region_step_diagnostic_name(
    const utils::DispatchStep& step) {
  if (!step.name.empty()) {
    return step.name;
  }
  if (!step.program_key.empty()) {
    return step.program_key;
  }
  return executable_region_dispatch_diagnostic_name(step);
}

bool warn_invalid_executable_region_dispatch_step(
    const utils::DispatchStep& step,
    const char* reason) {
  TORCH_WARN(
      "Executable region dispatch metadata is invalid: dispatch=",
      executable_region_dispatch_kind_name(step.dispatch_kind),
      " name=",
      executable_region_step_diagnostic_name(step),
      " program=",
      step.program_key,
      " ir_op_index=",
      step.ir_op_index,
      " reads=",
      step.reads.size(),
      " constants=",
      step.constants.size(),
      " temporaries=",
      step.temporaries.size(),
      " writes=",
      step.writes.size(),
      " reason=",
      reason);
  return false;
}

std::string sanitize_executable_gpu_profile_value(std::string value) {
  for (char& ch : value) {
    const bool is_alpha =
        (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
    const bool is_digit = ch >= '0' && ch <= '9';
    if (!(is_alpha || is_digit || ch == '.' || ch == '-' || ch == '_')) {
      ch = '_';
    }
  }
  if (value.empty()) {
    return "unlabeled";
  }
  return value;
}

std::string executable_region_profile_kernel_name(
    const utils::DispatchStep& step) {
  const std::string base_name = executable_region_dispatch_diagnostic_name(step);
  return std::string("exec_step.") +
      sanitize_executable_gpu_profile_value(base_name);
}

std::string executable_region_runtime_profile_label(
    const utils::VulkanExecutableRegion& executable_region,
    const utils::StageKind stage_kind,
    const uint32_t step_idx,
    const utils::DispatchStep& step) {
  const std::string region_name =
      executable_region.contract.debug_name.empty()
      ? executable_region.key
      : executable_region.contract.debug_name;
  const std::string step_name = executable_region_step_diagnostic_name(step);
  std::ostringstream stream;
  stream << "exec_region"
         << "|region="
         << sanitize_executable_gpu_profile_value(region_name)
         << "|stage="
         << sanitize_executable_gpu_profile_value(
                utils::stage_kind_name(stage_kind))
         << "|step=" << step_idx
         << "|dispatch="
         << sanitize_executable_gpu_profile_value(
                executable_region_dispatch_kind_name(step.dispatch_kind))
         << "|name="
         << sanitize_executable_gpu_profile_value(step_name);
  return stream.str();
}

bool run_compiled_executable_dispatch_step(
    CompiledExecutableBackboneRegionContext& context,
    const utils::DispatchStep& step) {
  const auto dispatch_kind = step.dispatch_kind;
  if (dispatch_kind == utils::DispatchKind::Unknown) {
    return warn_invalid_executable_region_dispatch_step(
        step, "missing authoritative dispatch_kind metadata");
  }

  if (dispatch_kind == utils::DispatchKind::PatchEmbed) {
    if (
        !context.patch_embed_context || step.reads.size() != 1u ||
        step.constants.size() != 1u ||
        step.writes.size() != 1u) {
      return false;
    }
    Tensor& output_slot = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.writes[0]);
    output_slot = run_conv2d_context_out(
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, step.reads[0]),
        context.patch_embed_context,
        output_slot);
    output_slot = prepare_decoder_buffer_tensor(output_slot);
    return true;
  }

  if (dispatch_kind == utils::DispatchKind::ImagePatchTokenInput) {
    if (
        !context.patch_embed_context || step.reads.size() != 1u ||
        step.constants.size() != 2u ||
        step.writes.size() != 1u) {
      return false;
    }
    Tensor& output_slot = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.writes[0]);
    output_slot = make_depth_anything_v2_tokens_from_image(
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, step.reads[0]),
        context.patch_embed_context,
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, step.constants[0]),
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, step.constants[1]),
        /*flatten_batch1_tokens=*/false);
    return true;
  }

  if (dispatch_kind == utils::DispatchKind::FeatureMapToTokens) {
    if (step.reads.size() != 1u || step.writes.size() != 1u) {
      return false;
    }
    Tensor& output_slot = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.writes[0]);
    return run_feature_map_to_tokens_direct_out(
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, step.reads[0]),
        output_slot);
  }

  if (dispatch_kind == utils::DispatchKind::ElementwiseAdd) {
    if (step.reads.size() != 2u || step.writes.size() != 1u) {
      return false;
    }
    Tensor positioned_input = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.reads[0]);
    Tensor pos_encoding = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.reads[1]);
    pos_encoding =
        maybe_expand_compiled_region_batch(pos_encoding, positioned_input);
    Tensor& output_slot = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.writes[0]);
    output_slot =
        add_buffer_out_vulkan(positioned_input, pos_encoding, output_slot);
    return true;
  }

  if (dispatch_kind == utils::DispatchKind::Concat) {
    if (step.reads.size() != 2u || step.writes.size() != 1u) {
      return false;
    }
    Tensor prefix = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.reads[0]);
    const Tensor& positioned_tokens = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.reads[1]);
    prefix = maybe_expand_compiled_region_batch(prefix, positioned_tokens);
    Tensor& output_slot = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.writes[0]);
    const std::array<Tensor, 2u> concat_inputs{prefix, positioned_tokens};
    return cat_buffer_out_vulkan(concat_inputs, 1, output_slot);
  }

  if (dispatch_kind == utils::DispatchKind::PatchTokenInput) {
    const bool legacy_layout =
        step.reads.size() == 1u && step.constants.size() == 2u;
    const bool flattened_layout =
        step.reads.size() == 3u && step.constants.empty();
    if ((!legacy_layout && !flattened_layout) || step.writes.size() != 1u) {
      return false;
    }
    const utils::VulkanValueId feature_map_value = step.reads[0];
    const utils::VulkanValueId prefix_value =
        legacy_layout ? step.constants[0] : step.reads[1];
    const utils::VulkanValueId pos_encoding_value =
        legacy_layout ? step.constants[1] : step.reads[2];
    Tensor& output_slot = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.writes[0]);
    output_slot = assemble_depth_anything_v2_tokens_from_feature_map(
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, feature_map_value),
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, prefix_value),
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, pos_encoding_value),
        /*flatten_batch1_tokens=*/false);
    return true;
  }

  if (dispatch_kind == utils::DispatchKind::BackboneBlock) {
    if (
        step.reads.size() != 1u || step.constants.size() != 1u ||
        step.writes.size() != 1u ||
        context.backbone_step_index >= context.backbone_contexts.size() ||
        context.backbone_step_index >= context.backbone_programs.size() ||
        context.backbone_step_index >= context.graph_scratches.size()) {
      return false;
    }

    const size_t idx = context.backbone_step_index;
    api::RuntimeLabelScope runtime_scope(
        compose_runtime_capture_label(
            vision_backbone_execution_label(
                context.backbone_contexts[idx]->allocation_label(),
                context.backbone_contexts[idx].get()) +
            ".compiled_session.executable_region"));
    auto& graph_scratch = context.graph_scratches[idx];
    if (graph_scratch.has_value()) {
      graph_scratch->reset();
    }
    Tensor& output_slot = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.writes[0]);
    (void)run_vision_backbone_block_program(
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, step.reads[0]),
        context.backbone_contexts[idx],
        &context.backbone_programs[idx],
        graph_scratch.has_value() ? &(*graph_scratch) : nullptr,
        &output_slot);
    ++context.backbone_step_index;
    return true;
  }

  if (dispatch_kind == utils::DispatchKind::CapturePatchTokens) {
    if (step.reads.size() != 1u || !step.constants.empty() ||
        step.writes.size() != 1u) {
      return warn_invalid_executable_region_dispatch_step(
          step,
          "CapturePatchTokens expects exactly one read, zero constants, and "
          "one write");
    }
    Tensor* output_slot = nullptr;
    if (
        context.capture_output_slots &&
        step.writes[0] < context.capture_output_slots_by_value.size() &&
        context.capture_output_slots_by_value[step.writes[0]].has_value()) {
      output_slot = &context.capture_output_slots->at(
          *context.capture_output_slots_by_value[step.writes[0]]);
    } else {
      output_slot = &compiled_executable_tensor_slot(
          context.tensor_slots, context.bindings, step.writes[0]);
    }
    Tensor& source_slot = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.reads[0]);
    copy_tensor_for_replay(*output_slot, source_slot);
    return true;
  }

  if (dispatch_kind == utils::DispatchKind::CaptureNormedPatchTokens) {
    if (
        !context.output_norm_context || context.normalized_shape.empty() ||
        step.constants.size() != 1u ||
        step.reads.size() != 1u || step.writes.size() != 1u) {
      return warn_invalid_executable_region_dispatch_step(
          step,
          "CaptureNormedPatchTokens expects one read, one norm constant, one "
          "write, and a defined output_norm_context with normalized_shape");
    }
    Tensor* output_slot = nullptr;
    if (
        context.capture_output_slots &&
        step.writes[0] < context.capture_output_slots_by_value.size() &&
        context.capture_output_slots_by_value[step.writes[0]].has_value()) {
      output_slot = &context.capture_output_slots->at(
          *context.capture_output_slots_by_value[step.writes[0]]);
    } else {
      output_slot = &compiled_executable_tensor_slot(
          context.tensor_slots, context.bindings, step.writes[0]);
    }
    (void)run_layernorm_context_out(
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, step.reads[0]),
        context.normalized_shape,
        context.output_norm_context,
        *output_slot);
    return true;
  }

  if (dispatch_kind == utils::DispatchKind::CaptureDecoderLayerPreprocess) {
    const bool apply_resize =
        step.constants.size() == 4u && step.temporaries.size() == 4u;
    if (
        !context.output_norm_context || context.normalized_shape.empty() ||
        step.reads.size() != 1u ||
        (step.constants.size() != 3u && step.constants.size() != 4u) ||
        (step.temporaries.size() != 3u && step.temporaries.size() != 4u) ||
        step.writes.size() != 1u ||
        context.decoder_project_index >= context.project_contexts.size() ||
        context.decoder_layer_index >= context.rn_contexts.size() ||
        !context.project_contexts[context.decoder_project_index] ||
        !context.rn_contexts[context.decoder_layer_index] ||
        (apply_resize &&
         (context.decoder_layer_index >= context.resize_contexts.size() ||
          !context.resize_contexts[context.decoder_layer_index]))) {
      return warn_invalid_executable_region_dispatch_step(
          step,
          "CaptureDecoderLayerPreprocess expects one read, 3-4 constants, "
          "3-4 temporaries, one write, output_norm metadata, and valid "
          "decoder project/resize/preprocess contexts");
    }

    Tensor capture_tokens = create_compiled_executable_virtual_tensor(
        context, step.temporaries[0]);
    capture_tokens = run_layernorm_context_out(
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, step.reads[0]),
        context.normalized_shape,
        context.output_norm_context,
        capture_tokens);

    Tensor layer_tokens = prepare_compiled_executable_decoder_tokens(
        capture_tokens, context.special_token_count);
    Tensor feature_output = create_compiled_executable_virtual_tensor(
        context, step.temporaries[1]);
    if (!run_tokens_to_feature_map_direct_out(
            layer_tokens, context.patch_h, context.patch_w, feature_output)) {
      return false;
    }
    feature_output = prepare_decoder_buffer_tensor(feature_output);
    if (!feature_output.defined() || feature_output.dim() != 4) {
      return false;
    }

    Tensor project_output = create_compiled_executable_virtual_tensor(
        context, step.temporaries[2]);
    project_output = run_conv2d_context_out(
        feature_output,
        context.project_contexts[context.decoder_project_index],
        project_output);

    Tensor preprocess_input = project_output;
    if (apply_resize) {
      Tensor resize_output = create_compiled_executable_virtual_tensor(
          context, step.temporaries[3]);
      resize_output = run_conv2d_context_any_out(
          project_output,
          context.resize_contexts[context.decoder_layer_index],
          resize_output);
      preprocess_input = resize_output;
    }

    Tensor& output_slot = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.writes[0]);
    output_slot = run_conv2d_context_out(
        preprocess_input,
        context.rn_contexts[context.decoder_layer_index],
        output_slot);
    ++context.decoder_project_index;
    ++context.decoder_layer_index;
    return true;
  }

  if (dispatch_kind == utils::DispatchKind::DecoderLayerPreprocess) {
    const bool apply_resize =
        step.constants.size() == 3u && step.temporaries.size() == 3u;
    if (
        step.reads.size() != 1u ||
        (step.constants.size() != 2u && step.constants.size() != 3u) ||
        (step.temporaries.size() != 2u && step.temporaries.size() != 3u) ||
        step.writes.size() != 1u ||
        context.decoder_project_index >= context.project_contexts.size() ||
        context.decoder_layer_index >= context.rn_contexts.size() ||
        !context.project_contexts[context.decoder_project_index] ||
        !context.rn_contexts[context.decoder_layer_index] ||
        (apply_resize &&
         (context.decoder_layer_index >= context.resize_contexts.size() ||
          !context.resize_contexts[context.decoder_layer_index]))) {
      return warn_invalid_executable_region_dispatch_step(
          step,
          "DecoderLayerPreprocess expects one read, 2-3 constants, 2-3 "
          "temporaries, one write, and valid decoder project/resize/"
          "preprocess contexts");
    }

    Tensor layer_tokens = prepare_compiled_executable_decoder_tokens(
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, step.reads[0]),
        context.special_token_count);
    Tensor feature_output = create_compiled_executable_virtual_tensor(
        context, step.temporaries[0]);
    if (!run_tokens_to_feature_map_direct_out(
            layer_tokens, context.patch_h, context.patch_w, feature_output)) {
      return false;
    }
    feature_output = prepare_decoder_buffer_tensor(feature_output);
    if (!feature_output.defined() || feature_output.dim() != 4) {
      return false;
    }

    Tensor project_output = create_compiled_executable_virtual_tensor(
        context, step.temporaries[1]);
    project_output = run_conv2d_context_out(
        feature_output,
        context.project_contexts[context.decoder_project_index],
        project_output);

    Tensor preprocess_input = project_output;
    if (apply_resize) {
      Tensor resize_output = create_compiled_executable_virtual_tensor(
          context, step.temporaries[2]);
      resize_output = run_conv2d_context_any_out(
          project_output,
          context.resize_contexts[context.decoder_layer_index],
          resize_output);
      preprocess_input = resize_output;
    }

    Tensor& output_slot = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.writes[0]);
    output_slot = run_conv2d_context_out(
        preprocess_input,
        context.rn_contexts[context.decoder_layer_index],
        output_slot);
    ++context.decoder_project_index;
    ++context.decoder_layer_index;
    return true;
  }

  if (dispatch_kind == utils::DispatchKind::TokensToFeatureMap) {
    if (
        step.reads.size() != 1u || !step.constants.empty() ||
        step.writes.size() != 1u || context.patch_h <= 0 || context.patch_w <= 0) {
      return false;
    }
    Tensor layer_tokens = prepare_compiled_executable_decoder_tokens(
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, step.reads[0]),
        context.special_token_count);
    Tensor& feature_slot = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.writes[0]);
    if (!run_tokens_to_feature_map_direct_out(
            layer_tokens, context.patch_h, context.patch_w, feature_slot)) {
      return false;
    }
    Tensor feature_buffer = prepare_decoder_buffer_tensor(feature_slot);
    if (!feature_buffer.defined() || feature_buffer.dim() != 4) {
      return false;
    }
    feature_slot = feature_buffer;
    return true;
  }

  if (dispatch_kind == utils::DispatchKind::DecoderProject) {
    if (
        step.reads.size() != 1u || step.constants.size() != 1u ||
        step.writes.size() != 1u ||
        context.decoder_project_index >= context.project_contexts.size() ||
        !context.project_contexts[context.decoder_project_index]) {
      return false;
    }
    Tensor& output_slot = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.writes[0]);
    output_slot = run_conv2d_context_out(
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, step.reads[0]),
        context.project_contexts[context.decoder_project_index],
        output_slot);
    ++context.decoder_project_index;
    return true;
  }

  if (dispatch_kind == utils::DispatchKind::DecoderResize) {
    if (
        step.reads.size() != 1u || step.constants.size() != 1u ||
        step.writes.size() != 1u ||
        context.decoder_layer_index >= context.resize_contexts.size() ||
        !context.resize_contexts[context.decoder_layer_index]) {
      return false;
    }
    Tensor& output_slot = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.writes[0]);
    output_slot = run_conv2d_context_any_out(
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, step.reads[0]),
        context.resize_contexts[context.decoder_layer_index],
        output_slot);
    return true;
  }

  if (dispatch_kind == utils::DispatchKind::DecoderPreprocess) {
    if (
        step.reads.size() != 1u || step.constants.size() != 1u ||
        step.writes.size() != 1u ||
        context.decoder_layer_index >= context.rn_contexts.size() ||
        !context.rn_contexts[context.decoder_layer_index]) {
      return false;
    }
    Tensor& output_slot = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.writes[0]);
    output_slot = run_conv2d_context_out(
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, step.reads[0]),
        context.rn_contexts[context.decoder_layer_index],
        output_slot);
    ++context.decoder_layer_index;
    return true;
  }

  if (dispatch_kind == utils::DispatchKind::DecoderHead) {
    if (
        step.reads.size() != 4u || step.constants.size() != 1u ||
        step.writes.size() != 1u || !context.head_context ||
        !context.head_programs || !context.head_programs->defined() ||
        context.output_size.size() != 2u) {
      return false;
    }
    Tensor& output_slot = compiled_executable_tensor_slot(
        context.tensor_slots, context.bindings, step.writes[0]);
    output_slot = run_vision_decoder_head_program(
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, step.reads[0]),
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, step.reads[1]),
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, step.reads[2]),
        compiled_executable_tensor_slot(
            context.tensor_slots, context.bindings, step.reads[3]),
        context.output_size,
        context.head_context,
        context.head_programs->programs[0],
        context.head_programs->programs[1],
        context.head_programs->programs[2],
        context.head_programs->programs[3],
        output_slot);
    context.ran_decoder_head = true;
    return true;
  }

  return false;
}

bool run_compiled_executable_region_step(
    CompiledExecutableBackboneRegionContext& context,
    const utils::StageKind stage_kind,
    const uint32_t step_idx,
    const utils::ExecStep& step) {
  switch (step.opcode) {
    case utils::ExecOpcode::Dispatch: {
      const auto* dispatch = std::get_if<utils::DispatchStep>(&step.payload);
      if (dispatch == nullptr) {
        return false;
      }
      api::Context* const context_vk = api::context();
      api::RuntimeLabelScope runtime_scope(
          context.executable_region
              ? executable_region_runtime_profile_label(
                    *context.executable_region, stage_kind, step_idx, *dispatch)
              : std::string());
      const VkExtent3D empty_extent{0u, 0u, 0u};
      const uint32_t log_idx = context_vk
          ? context_vk->begin_external_gpu_profile(
                executable_region_profile_kernel_name(*dispatch),
                empty_extent,
                empty_extent)
          : UINT32_MAX;
      auto finalize_profile = c10::make_scope_exit([&]() {
        if (context_vk) {
          context_vk->end_external_gpu_profile(log_idx);
        }
      });
      return run_compiled_executable_dispatch_step(context, *dispatch);
    }
    case utils::ExecOpcode::Barrier:
    case utils::ExecOpcode::Export:
      return true;
    case utils::ExecOpcode::Copy:
    case utils::ExecOpcode::Fill:
      return false;
  }
  return false;
}

bool run_compiled_executable_region(
    CompiledExecutableBackboneRegionContext& context,
    const utils::VulkanExecutableRegion& executable_region,
    const bool include_image_entry) {
  if (!executable_region.defined()) {
    return false;
  }

  bool image_entry_ready = !include_image_entry;
  for (const auto& stage : executable_region.stages) {
    switch (stage.kind) {
      case utils::StageKind::ImageEntry:
        if (!include_image_entry) {
          continue;
        }
        image_entry_ready = true;
        break;
      case utils::StageKind::Backbone:
      case utils::StageKind::Capture:
      case utils::StageKind::Decoder:
      case utils::StageKind::Export:
        break;
      case utils::StageKind::Unknown:
        continue;
    }

    if (
        stage.begin_step > stage.end_step ||
        stage.end_step > executable_region.steps.size()) {
      return false;
    }
    for (uint32_t step_idx = stage.begin_step; step_idx < stage.end_step;
         ++step_idx) {
      if (!run_compiled_executable_region_step(
              context, stage.kind, step_idx, executable_region.steps[step_idx])) {
        if (const auto* dispatch =
                std::get_if<utils::DispatchStep>(
                    &executable_region.steps[step_idx].payload)) {
          TORCH_WARN(
              "Executable region step failed: stage=",
              utils::stage_kind_name(stage.kind),
              " step=",
              step_idx,
              " dispatch=",
              executable_region_dispatch_kind_name(dispatch->dispatch_kind),
              " name=",
              executable_region_step_diagnostic_name(*dispatch),
              " program=",
              dispatch->program_key,
              " reads=",
              dispatch->reads.size(),
              " constants=",
              dispatch->constants.size(),
              " writes=",
              dispatch->writes.size());
        } else {
          TORCH_WARN(
              "Executable region non-dispatch step failed: stage=",
              utils::stage_kind_name(stage.kind),
              " step=",
              step_idx,
              " opcode=",
              static_cast<int>(executable_region.steps[step_idx].opcode));
        }
        return false;
      }
    }
  }

  const size_t expected_decoder_layers =
      compiled_executable_expected_decoder_layers(context);
  if (!image_entry_ready) {
    TORCH_WARN(
        "Executable region completed without satisfying image-entry "
        "requirements");
    return false;
  }
  if (context.backbone_step_index != context.backbone_contexts.size()) {
    TORCH_WARN(
        "Executable region completed without satisfying backbone execution: "
        "backbone_step_index=",
        context.backbone_step_index,
        " backbone_contexts=",
        context.backbone_contexts.size());
    return false;
  }
  if (context.decoder_project_index != expected_decoder_layers) {
    TORCH_WARN(
        "Executable region completed without satisfying decoder project "
        "execution: decoder_project_index=",
        context.decoder_project_index,
        " expected_decoder_layers=",
        expected_decoder_layers);
    return false;
  }
  if (context.decoder_layer_index != expected_decoder_layers) {
    TORCH_WARN(
        "Executable region completed without satisfying decoder layer "
        "execution: decoder_layer_index=",
        context.decoder_layer_index,
        " expected_decoder_layers=",
        expected_decoder_layers);
    return false;
  }
  if (context.head_programs && !context.ran_decoder_head) {
    TORCH_WARN("Executable region completed without running the decoder head");
    return false;
  }
  return true;
}

void run_recorded_compiled_replay_or_direct_steps(
    const utils::ExecutionGraphReplayBundle& replay_bundle,
    const char* op_name,
    const char* allocation_label) {
  const utils::ReplayEpoch epoch =
      utils::begin_replay_epoch(replay_bundle.identity(), allocation_label);
  const std::string detail =
      "action=run_steps_direct reason=compiled_replay_submit_guard "
      "failure_class=ReplayHangRisk";
  utils::log_replay_event(
      "compiled_replay_submit_guard",
      replay_bundle.identity(),
      epoch.run_id,
      allocation_label,
      detail);
  api::report_vulkan_failure(
      api::VulkanFailureClass::ReplayHangRisk,
      op_name ? op_name : "compiled_replay",
      "CompiledReplaySubmitGuard",
      detail);
  replay_bundle.warmup();
  api::context()->flush_pending_cmds();
}

bool run_compiled_session_image_entry_region(
    std::vector<Tensor>& tensor_slots,
    const utils::VulkanCompiledSession& compiled_session,
    const utils::VulkanCompiledSessionTensorBindings& bindings,
    const c10::intrusive_ptr<Conv2dPackedContext>& patch_embed_context) {
  bool ran_patch_embed = false;
  bool ran_patch_tokens = false;

  for (const auto& op : compiled_session.ir().ops()) {
    switch (op.kind) {
      case utils::VulkanIROpKind::PatchEmbed: {
        if (
            !patch_embed_context || op.inputs.size() != 1u ||
            op.outputs.size() != 1u) {
          return false;
        }
        Tensor& output_slot =
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.outputs[0]));
        output_slot = run_conv2d_context_out(
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.inputs[0])),
            patch_embed_context,
            output_slot);
        output_slot = prepare_decoder_buffer_tensor(output_slot);
        ran_patch_embed = true;
        break;
      }
      case utils::VulkanIROpKind::FeatureMapToTokens: {
        if (
            !ran_patch_embed || op.inputs.size() != 1u ||
            op.outputs.size() != 1u) {
          return false;
        }
        Tensor& output_slot =
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.outputs[0]));
        if (!run_feature_map_to_tokens_direct_out(
                tensor_slots.at(compiled_session_tensor_slot(bindings, op.inputs[0])),
                output_slot)) {
          return false;
        }
        break;
      }
      case utils::VulkanIROpKind::ElementwiseAdd: {
        if (
            !ran_patch_embed || op.inputs.size() != 2u ||
            op.outputs.size() != 1u) {
          return false;
        }
        Tensor positioned_input =
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.inputs[0]));
        Tensor pos_encoding =
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.inputs[1]));
        if (pos_encoding.dim() == 3 && positioned_input.dim() == 3 &&
            pos_encoding.size(0) == 1 && positioned_input.size(0) != 1) {
          pos_encoding = pos_encoding.expand(
              {positioned_input.size(0), pos_encoding.size(1), pos_encoding.size(2)});
        }
        Tensor& output_slot =
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.outputs[0]));
        output_slot =
            add_buffer_out_vulkan(positioned_input, pos_encoding, output_slot);
        break;
      }
      case utils::VulkanIROpKind::Concat: {
        if (op.inputs.size() != 2u || op.outputs.size() != 1u) {
          return false;
        }
        Tensor prefix =
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.inputs[0]));
        const Tensor& positioned_tokens =
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.inputs[1]));
        if (prefix.dim() == 3 && positioned_tokens.dim() == 3 &&
            prefix.size(0) == 1 && positioned_tokens.size(0) != 1) {
          prefix = prefix.expand(
              {positioned_tokens.size(0), prefix.size(1), prefix.size(2)});
        }
        Tensor& output_slot =
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.outputs[0]));
        const std::array<Tensor, 2u> concat_inputs{prefix, positioned_tokens};
        if (!cat_buffer_out_vulkan(concat_inputs, 1, output_slot)) {
          return false;
        }
        ran_patch_tokens = true;
        break;
      }
      case utils::VulkanIROpKind::PatchTokenInput: {
        if (
            !ran_patch_embed || op.inputs.size() != 1u ||
            op.outputs.size() != 1u || op.constants.size() != 2u) {
          return false;
        }
        Tensor& output_slot =
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.outputs[0]));
        output_slot = assemble_depth_anything_v2_tokens_from_feature_map(
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.inputs[0])),
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.constants[0])),
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.constants[1])),
            /*flatten_batch1_tokens=*/false);
        ran_patch_tokens = true;
        break;
      }
      case utils::VulkanIROpKind::BackboneBlock:
        return ran_patch_tokens;
      case utils::VulkanIROpKind::InputImage:
      case utils::VulkanIROpKind::CapturePatchTokens:
      case utils::VulkanIROpKind::CaptureNormedPatchTokens:
      case utils::VulkanIROpKind::TokensToFeatureMap:
      case utils::VulkanIROpKind::DecoderProject:
      case utils::VulkanIROpKind::DecoderResize:
      case utils::VulkanIROpKind::DecoderPreprocess:
      case utils::VulkanIROpKind::DecoderHead:
      case utils::VulkanIROpKind::OutputAlias:
        break;
    }
  }

  return ran_patch_embed && ran_patch_tokens;
}

bool run_compiled_session_decoder_region(
    std::vector<Tensor>& tensor_slots,
    const utils::VulkanCompiledSession& compiled_session,
    const utils::VulkanCompiledSessionTensorBindings& bindings,
    const int64_t patch_h,
    const int64_t patch_w,
    IntArrayRef output_size,
    const std::array<c10::intrusive_ptr<Conv2dPackedContext>, 4u>&
        project_contexts,
    const std::array<c10::intrusive_ptr<Conv2dPackedContext>, 4u>&
        resize_contexts,
    const std::array<c10::intrusive_ptr<Conv2dPackedContext>, 4u>& rn_contexts,
    const c10::intrusive_ptr<VisionDecoderHeadContext>& head_context,
    CompiledDecoderHeadPrograms& head_programs,
    const std::function<Tensor(const Tensor&)>& prepare_tokens) {
  size_t project_idx = 0u;
  size_t decoder_layer_idx = 0u;
  bool ran_head = false;

  for (const auto& op : compiled_session.ir().ops()) {
    switch (op.kind) {
      case utils::VulkanIROpKind::TokensToFeatureMap: {
        if (op.inputs.size() != 1u || op.outputs.size() != 1u) {
          return false;
        }
        Tensor layer_tokens =
            prepare_tokens(tensor_slots.at(
                compiled_session_tensor_slot(bindings, op.inputs[0])));
        Tensor& feature_slot =
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.outputs[0]));
        if (!run_tokens_to_feature_map_direct_out(
                layer_tokens,
                patch_h,
                patch_w,
                feature_slot)) {
          return false;
        }
        Tensor feature_buffer = prepare_decoder_buffer_tensor(feature_slot);
        if (!feature_buffer.defined() || feature_buffer.dim() != 4) {
          return false;
        }
        feature_slot = feature_buffer;
        break;
      }
      case utils::VulkanIROpKind::DecoderProject: {
        if (
            op.inputs.size() != 1u || op.outputs.size() != 1u ||
            project_idx >= project_contexts.size()) {
          return false;
        }
        Tensor& output_slot =
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.outputs[0]));
        output_slot = run_conv2d_context_out(
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.inputs[0])),
            project_contexts[project_idx],
            output_slot);
        ++project_idx;
        break;
      }
      case utils::VulkanIROpKind::DecoderResize: {
        if (
            op.inputs.size() != 1u || op.outputs.size() != 1u ||
            decoder_layer_idx >= resize_contexts.size() ||
            !resize_contexts[decoder_layer_idx]) {
          return false;
        }
        Tensor& output_slot =
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.outputs[0]));
        output_slot = run_conv2d_context_any_out(
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.inputs[0])),
            resize_contexts[decoder_layer_idx],
            output_slot);
        break;
      }
      case utils::VulkanIROpKind::DecoderPreprocess: {
        if (
            op.inputs.size() != 1u || op.outputs.size() != 1u ||
            decoder_layer_idx >= rn_contexts.size()) {
          return false;
        }
        Tensor& output_slot =
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.outputs[0]));
        output_slot = run_conv2d_context_out(
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.inputs[0])),
            rn_contexts[decoder_layer_idx],
            output_slot);
        ++decoder_layer_idx;
        break;
      }
      case utils::VulkanIROpKind::DecoderHead: {
        if (op.inputs.size() != 4u || op.outputs.size() != 1u) {
          return false;
        }
        Tensor& output_slot =
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.outputs[0]));
        output_slot = run_vision_decoder_head_program(
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.inputs[0])),
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.inputs[1])),
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.inputs[2])),
            tensor_slots.at(compiled_session_tensor_slot(bindings, op.inputs[3])),
            output_size,
            head_context,
            head_programs.programs[0],
            head_programs.programs[1],
            head_programs.programs[2],
            head_programs.programs[3],
            output_slot);
        ran_head = true;
        break;
      }
      case utils::VulkanIROpKind::OutputAlias:
      case utils::VulkanIROpKind::InputImage:
      case utils::VulkanIROpKind::PatchEmbed:
      case utils::VulkanIROpKind::FeatureMapToTokens:
      case utils::VulkanIROpKind::ElementwiseAdd:
      case utils::VulkanIROpKind::Concat:
      case utils::VulkanIROpKind::PatchTokenInput:
      case utils::VulkanIROpKind::BackboneBlock:
      case utils::VulkanIROpKind::CapturePatchTokens:
      case utils::VulkanIROpKind::CaptureNormedPatchTokens:
        break;
    }
  }

  if (!ran_head || decoder_layer_idx != rn_contexts.size()) {
    return false;
  }
  return true;
}

std::optional<Tensor> try_run_vision_decoder_preprocess_head_compiled_session(
    const std::array<Tensor, 4u>& layer_tokens,
    const Device& output_device,
    const ScalarType output_dtype,
    const int64_t patch_h,
    const int64_t patch_w,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderPreprocessHeadContext>& context,
    const utils::VulkanRuntimePolicy& runtime_policy,
    const VisionReplayBundleIdentity& bundle_identity,
    const std::string& root_label,
    const utils::VulkanCompiledSession& compiled_session) {
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionDecoder ||
      !compiled_session.defined() || !compiled_session.executable()) {
    return std::nullopt;
  }
  const std::string detail =
      "action=fallback_without_executable_region_compiled_session "
      "reason=compiled_executable_region_guard "
      "failure_class=ReplayHangRisk";
  api::report_vulkan_failure(
      api::VulkanFailureClass::ReplayHangRisk,
      "vulkan_prepack::run_vision_decoder_preprocess_head_compiled_session",
      "CompiledExecutableRegionGuard",
      detail);
  utils::log_replay_event(
      "compiled_executable_region_guard",
      compiled_session.identity(),
      utils::current_replay_epoch(compiled_session.identity()).run_id,
      "vision.decoder_preprocess_head.compiled_session",
      detail);
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_vision_decoder_preprocess_head_compiled_session.guard.executable_region_disabled");
  return std::nullopt;
  const auto decoder_plan =
      make_compiled_decoder_execution_plan(compiled_session);
  if (
      !decoder_plan.has_value()) {
    return std::nullopt;
  }
  const auto* executable_region = compiled_session.executable_region();
  if (
      !executable_region ||
      !executable_region->defined()) {
    return std::nullopt;
  }
  const auto bindings = utils::make_compiled_executable_region_tensor_bindings(
      compiled_session, *executable_region);
  if (!bindings.has_value() || bindings->input_values.size() != layer_tokens.size()) {
    return std::nullopt;
  }
  const auto compiled_bindings = *bindings;

  const std::array<c10::intrusive_ptr<Conv2dPackedContext>, 4u>
      project_contexts{
          context->project1_context(),
          context->project2_context(),
          context->project3_context(),
          context->project4_context(),
      };
  const std::array<c10::intrusive_ptr<Conv2dPackedContext>, 4u>
      resize_contexts{
          context->resize1_context(),
          context->resize2_context(),
          c10::intrusive_ptr<Conv2dPackedContext>{},
          context->resize4_context(),
      };
  const std::array<c10::intrusive_ptr<Conv2dPackedContext>, 4u> rn_contexts{
      context->layer1_rn_context(),
      context->layer2_rn_context(),
      context->layer3_rn_context(),
      context->layer4_rn_context(),
  };

  const auto& program_plan = *runtime_policy.execution_program_plan;
  auto root = utils::lookup_or_create_labeled_execution_graph_root(
      root_label,
      kFloat,
      program_plan.persistent);
  const std::string compiled_bundle_key =
      bundle_identity.key + "|compiled_executable_region_v1";
  auto replay_bundle = root.lookup_or_create_replay_bundle(
      compiled_bundle_key,
      [&]() -> utils::ExecutionGraphReplayBundle {
        auto tensor_slots = make_compiled_session_tensor_slots(
            compiled_session,
            compiled_bindings,
            program_plan.persistent);
        if (!has_compiled_decoder_head_shape(
                compiled_session, *decoder_plan, context->head_context())) {
          utils::log_vulkan_op_hit(
              "vulkan_prepack::run_vision_decoder_preprocess_head_compiled_session.skip.invalid_head_shape");
          return {};
        }
        auto head_programs = make_compiled_decoder_head_programs(
            compiled_session,
            *decoder_plan,
            context->head_context(),
            program_plan);
        if (!(head_programs.has_value() && (*head_programs)->defined())) {
          utils::log_vulkan_op_hit(
              "vulkan_prepack::run_vision_decoder_preprocess_head_compiled_session.skip.no_head_programs");
          return {};
        }
        auto decoder_replay = utils::make_execution_graph_replay(
            root.allocation_label() +
                ".vision.decoder_preprocess_head.compiled.decoder.step" +
                bundle_identity.label_suffix,
            utils::VulkanInferenceGraphKind::VisionDecoder,
            kFloat,
            program_plan.persistent,
            std::vector<Tensor>{},
            std::vector<std::optional<Tensor>>{},
            std::vector<utils::ExecutionGraphProgramHandle>{});
        std::vector<utils::ExecutionGraphReplayStep> steps;
        steps.push_back(utils::make_execution_graph_replay_step(
            std::move(decoder_replay),
            [tensor_slots,
             compiled_session,
             compiled_bindings,
             patch_h,
             patch_w,
             output_size = output_size.vec(),
             project_contexts,
             resize_contexts,
             rn_contexts,
             head_context = context->head_context(),
             head_programs = *head_programs]() mutable {
              const auto* region = compiled_session.executable_region();
              TORCH_INTERNAL_ASSERT(
                  region && region->defined(),
                  "Compiled decoder preprocess/head session expected a lowered "
                  "executable region");
              std::vector<c10::intrusive_ptr<VisionBackboneBlockContext>>
                  backbone_contexts;
              std::vector<utils::VisionBackboneProgram> backbone_programs;
              std::vector<std::optional<utils::ScratchArena>> graph_scratches;
              CompiledExecutableBackboneRegionContext context{
                  region,
                  *tensor_slots,
                  compiled_bindings,
                  std::shared_ptr<std::vector<Tensor>>{},
                  0u,
                  std::vector<std::optional<size_t>>{},
                  c10::intrusive_ptr<Conv2dPackedContext>{},
                  backbone_contexts,
                  backbone_programs,
                  graph_scratches,
                  std::vector<int64_t>{},
                  c10::intrusive_ptr<LayernormPackedContext>{},
                  project_contexts,
                  resize_contexts,
                  rn_contexts,
                  head_context,
                  head_programs,
                  output_size,
                  patch_h,
                  patch_w,
                  0};
              TORCH_INTERNAL_ASSERT(
                  run_compiled_executable_region(
                      context,
                      *region,
                      /*include_image_entry=*/false),
                  "Compiled decoder preprocess/head session failed to execute "
                  "the executable region");
            }));

        return utils::make_execution_graph_replay_bundle(
            root.allocation_label() +
                ".vision.decoder_preprocess_head.compiled_session.replay" +
                bundle_identity.label_suffix,
            kFloat,
            program_plan.persistent,
            std::move(steps),
            std::move(tensor_slots));
      });

  if (
      !replay_bundle.defined() || replay_bundle.size() == 0u ||
      replay_bundle.tensor_slot_count() < compiled_bindings.tensor_slot_count()) {
    return std::nullopt;
  }

  const bool first_run =
      replay_bundle.size() > 0u && !replay_bundle.recorded();
  const std::string previous_runtime_label =
      api::swap_runtime_label(std::string());
  auto restore_runtime_label = c10::make_scope_exit([&]() {
    api::swap_runtime_label(previous_runtime_label);
  });
  std::optional<Tensor> warmup_output = std::nullopt;
  if (first_run) {
    warmup_output = run_vision_decoder_preprocess_head_fallback(
        layer_tokens,
        output_device,
        output_dtype,
        patch_h,
        patch_w,
        output_size,
        context);
  }
  for (size_t idx = 0u; idx < layer_tokens.size(); ++idx) {
    copy_tensor_for_replay(
        replay_bundle.tensor_slot(compiled_session_tensor_slot(
            compiled_bindings,
            compiled_bindings.input_values[idx])),
        layer_tokens[idx]);
  }
  api::context()->flush_pending_cmds();
  const size_t output_slot_idx = compiled_session_tensor_slot(
      compiled_bindings,
      decoder_plan->final_output_value);
  if (first_run) {
    replay_bundle.warmup();
    api::context()->flush_pending_cmds();
    replay_bundle.record_steps_individually();
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_decoder_preprocess_head_compiled_session.replay_warmup");
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_decoder_preprocess_head_compiled_session");
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_decoder_preprocess_head_context");
    TORCH_INTERNAL_ASSERT(
        warmup_output.has_value(),
        "Compiled decoder preprocess/head session expected a warmup output");
    return *warmup_output;
  }

  run_recorded_compiled_replay_or_direct_steps(
      replay_bundle,
      "vulkan_prepack::run_vision_decoder_preprocess_head_compiled_session",
      "vision.decoder_preprocess_head.compiled_session.replay");
  Tensor output = copy_compiled_session_output(replay_bundle, output_slot_idx);

  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_vision_decoder_preprocess_head_compiled_session.replay");
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_vision_decoder_preprocess_head_compiled_session");
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_vision_decoder_preprocess_head_context");
  return maybe_restore_tensor(output, output_device, output_dtype);
}

utils::ExecutionGraphReplayStep make_vision_decoder_replay_step(
    utils::VisionDecoderInferenceReplay decoder_replay,
    std::vector<int64_t> decoder_target_sizes,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& decoder_context) {
  const VisionDecoderRunOutputs decoder_replay_outputs{
      decoder_replay.program().skip_relu_output(),
      decoder_replay.program().skip_conv1_output(),
      decoder_replay.program().skip_conv2_output(),
      decoder_replay.program().skip_res_output(),
      decoder_replay.program().main_input_output(),
      decoder_replay.program().main_relu_output(),
      decoder_replay.program().main_conv1_output(),
      decoder_replay.program().main_conv2_output(),
      decoder_replay.program().main_res_output(),
      decoder_replay.program().upsample_output(),
      decoder_replay.program().out_conv_output(),
  };
  return decoder_replay.phase_step(
      [decoder_replay,
       decoder_target_sizes = std::move(decoder_target_sizes),
       decoder_context,
       decoder_replay_outputs]() mutable {
        (void)run_vision_decoder_fusion_block_program(
            decoder_replay.input_slot(),
            decoder_replay.skip_slot(),
            decoder_target_sizes,
            decoder_context,
            decoder_replay_outputs);
      });
}

} // namespace

std::vector<int64_t> vision_owner_counters_snapshot() {
  const auto& counters = vulkan_vision_owner_counters();
  return {
      static_cast<int64_t>(
          counters.total_attempts.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.block_owner_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.stack_owner_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.compiled_session_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_gate_disabled.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_missing_context.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_dtype.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_shape.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_layout.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_route_policy.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_python_bridge.load(std::memory_order_relaxed)),
  };
}

void reset_vision_owner_counters() {
  auto& counters = vulkan_vision_owner_counters();
  counters.total_attempts.store(0u, std::memory_order_relaxed);
  counters.block_owner_hit.store(0u, std::memory_order_relaxed);
  counters.stack_owner_hit.store(0u, std::memory_order_relaxed);
  counters.compiled_session_hit.store(0u, std::memory_order_relaxed);
  counters.reject_gate_disabled.store(0u, std::memory_order_relaxed);
  counters.reject_missing_context.store(0u, std::memory_order_relaxed);
  counters.reject_dtype.store(0u, std::memory_order_relaxed);
  counters.reject_shape.store(0u, std::memory_order_relaxed);
  counters.reject_layout.store(0u, std::memory_order_relaxed);
  counters.reject_route_policy.store(0u, std::memory_order_relaxed);
  counters.reject_python_bridge.store(0u, std::memory_order_relaxed);
}

std::vector<int64_t> vision_owner_context_counters_snapshot() {
  const auto& counters = vulkan_vision_owner_context_counters();
  return {
      static_cast<int64_t>(counters.create_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.cache_hit_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.unpack_readback_count.load(std::memory_order_relaxed)),
  };
}

void reset_vision_owner_context_counters() {
  auto& counters = vulkan_vision_owner_context_counters();
  counters.create_count.store(0u, std::memory_order_relaxed);
  counters.cache_hit_count.store(0u, std::memory_order_relaxed);
  counters.unpack_readback_count.store(0u, std::memory_order_relaxed);
}

void record_vision_owner_context_cache_hit() {
  vulkan_vision_owner_context_counters().cache_hit_count.fetch_add(
      1u,
      std::memory_order_relaxed);
}

std::vector<int64_t> vision_owner_mlp_counters_snapshot() {
  const auto& counters = vulkan_vision_owner_mlp_counters();
  return {
      static_cast<int64_t>(counters.total.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.linear_gelu_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.fc2_after_linear_gelu_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.reject_no_owner.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.reject_dtype.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.reject_shape.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.reject_context.load(std::memory_order_relaxed)),
  };
}

void reset_vision_owner_mlp_counters() {
  auto& counters = vulkan_vision_owner_mlp_counters();
  counters.total.store(0u, std::memory_order_relaxed);
  counters.linear_gelu_hit.store(0u, std::memory_order_relaxed);
  counters.fc2_after_linear_gelu_hit.store(0u, std::memory_order_relaxed);
  counters.reject_no_owner.store(0u, std::memory_order_relaxed);
  counters.reject_dtype.store(0u, std::memory_order_relaxed);
  counters.reject_shape.store(0u, std::memory_order_relaxed);
  counters.reject_context.store(0u, std::memory_order_relaxed);
}

std::vector<int64_t> vision_stack_owner_counters_snapshot() {
  const auto& counters = vulkan_vision_stack_owner_counters();
  return {
      static_cast<int64_t>(
          counters.total_attempts.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.stack_owner_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.block_context_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.block_execute_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_missing_context.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.reject_shape.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.reject_dtype.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.reject_layout.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_unsafe_replay.load(std::memory_order_relaxed)),
  };
}

void reset_vision_stack_owner_counters() {
  auto& counters = vulkan_vision_stack_owner_counters();
  counters.total_attempts.store(0u, std::memory_order_relaxed);
  counters.stack_owner_hit.store(0u, std::memory_order_relaxed);
  counters.block_context_count.store(0u, std::memory_order_relaxed);
  counters.block_execute_count.store(0u, std::memory_order_relaxed);
  counters.reject_missing_context.store(0u, std::memory_order_relaxed);
  counters.reject_shape.store(0u, std::memory_order_relaxed);
  counters.reject_dtype.store(0u, std::memory_order_relaxed);
  counters.reject_layout.store(0u, std::memory_order_relaxed);
  counters.reject_unsafe_replay.store(0u, std::memory_order_relaxed);
}

std::vector<int64_t> stack_attention_counters_snapshot() {
  const auto& counters = vulkan_stack_attention_counters();
  return {
      static_cast<int64_t>(counters.total.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.direct_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.decomposed_placeholder_bypass.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.reject_shape.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.reject_dtype.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.reject_layout.load(std::memory_order_relaxed)),
  };
}

void reset_stack_attention_counters() {
  auto& counters = vulkan_stack_attention_counters();
  counters.total.store(0u, std::memory_order_relaxed);
  counters.direct_hit.store(0u, std::memory_order_relaxed);
  counters.decomposed_placeholder_bypass.store(0u, std::memory_order_relaxed);
  counters.reject_shape.store(0u, std::memory_order_relaxed);
  counters.reject_dtype.store(0u, std::memory_order_relaxed);
  counters.reject_layout.store(0u, std::memory_order_relaxed);
}

std::vector<std::string> stack_execution_manifest_snapshot() {
  std::lock_guard<std::mutex> lock(stack_execution_manifest_mutex());
  std::vector<std::string> snapshot;
  {
    std::lock_guard<std::mutex> plan_lock(stack_shape_plan_summary_mutex());
    snapshot.reserve(
        stack_execution_manifest_rows().size() +
        stack_shape_plan_manifest_rows().size());
  }
  for (const auto& row : stack_execution_manifest_rows()) {
    std::ostringstream out;
    out << "stack_manifest"
        << " ordinal=" << row.ordinal
        << " block=" << row.block_index
        << " phase=" << api::vision_stack_phase_name(row.phase)
        << " op=" << row.op_label
        << " kernel=" << row.kernel_name
        << " input_shapes=" << row.input_shapes
        << " output_shapes=" << row.output_shapes
        << " dtype=" << row.dtype
        << " uses_dynamic_shape=" << (row.uses_dynamic_shape ? 1 : 0)
        << " allocates_output=" << (row.allocates_output ? 1 : 0)
        << " writes_preexisting_output="
        << (row.writes_preexisting_output ? 1 : 0)
        << " escapes_stack=" << (row.escapes_stack ? 1 : 0)
        << " requested_intermediate="
        << (row.requested_intermediate ? 1 : 0)
        << " requires_cpu_data=" << (row.requires_cpu_data ? 1 : 0)
        << " uses_fallback=" << (row.uses_fallback ? 1 : 0)
        << " submits_command_buffer="
        << (row.submits_command_buffer ? 1 : 0)
        << " requires_host_sync=" << (row.requires_host_sync ? 1 : 0)
        << " uses_runtime_capture=" << (row.uses_runtime_capture ? 1 : 0)
        << " uses_replay=" << (row.uses_replay ? 1 : 0)
        << " safe_to_capture=" << (row.safe_to_capture ? 1 : 0);
    snapshot.emplace_back(out.str());
  }
  {
    std::lock_guard<std::mutex> plan_lock(stack_shape_plan_summary_mutex());
    for (const auto& row : stack_shape_plan_manifest_rows()) {
      snapshot.emplace_back(row);
    }
  }
  return snapshot;
}

void reset_stack_execution_manifest() {
  std::lock_guard<std::mutex> lock(stack_execution_manifest_mutex());
  stack_execution_manifest_rows().clear();
}

std::vector<int64_t> stack_capture_readiness_snapshot() {
  std::lock_guard<std::mutex> lock(stack_execution_manifest_mutex());
  const auto& rows = stack_execution_manifest_rows();
  bool fixed_shapes = !rows.empty();
  bool no_cpu_fallback = !rows.empty();
  bool no_host_sync = !rows.empty();
  bool no_nested_replay = !rows.empty();
  bool no_runtime_capture_active = !rows.empty();
  bool requested_intermediates_marked = false;
  bool all_requested_intermediates_escape = true;
  bool all_internal_outputs_owned = !rows.empty();
  bool all_outputs_have_known_lifetime = !rows.empty();

  for (const auto& row : rows) {
    fixed_shapes = fixed_shapes && !row.uses_dynamic_shape;
    no_cpu_fallback = no_cpu_fallback && !row.uses_fallback;
    no_host_sync = no_host_sync && !row.requires_host_sync;
    no_nested_replay = no_nested_replay && !row.uses_replay;
    no_runtime_capture_active =
        no_runtime_capture_active && !row.uses_runtime_capture;
    if (row.requested_intermediate) {
      requested_intermediates_marked = true;
      all_requested_intermediates_escape =
          all_requested_intermediates_escape && row.escapes_stack;
    }
    if (!row.escapes_stack) {
      all_internal_outputs_owned =
          all_internal_outputs_owned && !row.requires_cpu_data;
    }
    all_outputs_have_known_lifetime =
        all_outputs_have_known_lifetime &&
        (row.escapes_stack || !row.requested_intermediate);
  }

  requested_intermediates_marked =
      requested_intermediates_marked && all_requested_intermediates_escape;
  const bool safe_to_capture = fixed_shapes && no_cpu_fallback &&
      no_host_sync && no_nested_replay && no_runtime_capture_active &&
      requested_intermediates_marked && all_internal_outputs_owned &&
      all_outputs_have_known_lifetime;

  return {
      fixed_shapes ? 1 : 0,
      no_cpu_fallback ? 1 : 0,
      no_host_sync ? 1 : 0,
      no_nested_replay ? 1 : 0,
      no_runtime_capture_active ? 1 : 0,
      requested_intermediates_marked ? 1 : 0,
      all_internal_outputs_owned ? 1 : 0,
      all_outputs_have_known_lifetime ? 1 : 0,
      safe_to_capture ? 1 : 0,
  };
}

std::vector<std::string> stack_shape_plan_keys_snapshot() {
  std::lock_guard<std::mutex> lock(stack_shape_plan_summary_mutex());
  std::vector<std::string> keys;
  keys.reserve(stack_shape_plan_readiness_rows().size());
  for (const auto& entry : stack_shape_plan_readiness_rows()) {
    keys.emplace_back(entry.first);
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

std::vector<std::string> stack_shape_plan_readiness_snapshot() {
  std::lock_guard<std::mutex> lock(stack_shape_plan_summary_mutex());
  std::vector<std::string> rows;
  rows.reserve(stack_shape_plan_readiness_rows().size());
  for (const auto& entry : stack_shape_plan_readiness_rows()) {
    rows.emplace_back(entry.second);
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

std::vector<int64_t> stack_shape_plan_counters_snapshot() {
  const auto& counters = vulkan_stack_shape_plan_counters();
  return {
      static_cast<int64_t>(
          counters.total_attempts.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.plan_build_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.plan_cache_hit_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.plan_reject_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.binding_valid_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.binding_invalid_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.invalid_tokens.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.invalid_dtype.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.invalid_capability.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.invalid_requested_intermediates.load(
              std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.invalid_context_identity.load(std::memory_order_relaxed)),
  };
}

void reset_stack_shape_plan_counters() {
  auto& counters = vulkan_stack_shape_plan_counters();
  counters.total_attempts.store(0u, std::memory_order_relaxed);
  counters.plan_build_count.store(0u, std::memory_order_relaxed);
  counters.plan_cache_hit_count.store(0u, std::memory_order_relaxed);
  counters.plan_reject_count.store(0u, std::memory_order_relaxed);
  counters.binding_valid_count.store(0u, std::memory_order_relaxed);
  counters.binding_invalid_count.store(0u, std::memory_order_relaxed);
  counters.invalid_tokens.store(0u, std::memory_order_relaxed);
  counters.invalid_dtype.store(0u, std::memory_order_relaxed);
  counters.invalid_capability.store(0u, std::memory_order_relaxed);
  counters.invalid_requested_intermediates.store(0u, std::memory_order_relaxed);
  counters.invalid_context_identity.store(0u, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(stack_shape_plan_summary_mutex());
  stack_shape_plan_readiness_rows().clear();
  stack_shape_plan_manifest_rows().clear();
}

std::vector<std::string> stack_resource_binding_manifest_snapshot() {
  std::lock_guard<std::mutex> lock(stack_resource_binding_manifest_mutex());
  return stack_resource_binding_manifest_rows();
}

void reset_stack_resource_binding_manifest() {
  std::lock_guard<std::mutex> lock(stack_resource_binding_manifest_mutex());
  stack_resource_binding_manifest_rows().clear();
  stack_replay_binding_mode_rows().clear();
  stack_descriptor_binding_table_rows().clear();
  stack_descriptor_binding_validation_rows().clear();
}

std::vector<std::string> stack_descriptor_binding_table_snapshot() {
  std::lock_guard<std::mutex> lock(stack_resource_binding_manifest_mutex());
  return stack_descriptor_binding_table_rows();
}

std::vector<std::string> stack_descriptor_binding_validation_snapshot() {
  std::lock_guard<std::mutex> lock(stack_resource_binding_manifest_mutex());
  std::vector<std::string> rows;
  rows.reserve(stack_descriptor_binding_validation_rows().size());
  for (const auto& entry : stack_descriptor_binding_validation_rows()) {
    rows.emplace_back(entry.second);
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

void reset_stack_descriptor_binding_table() {
  reset_stack_resource_binding_manifest();
}

std::vector<int64_t> stack_planned_recording_readiness_snapshot() {
  const auto shape_readiness = stack_shape_plan_readiness_snapshot();
  const bool shape_plan_ready = std::any_of(
      shape_readiness.begin(),
      shape_readiness.end(),
      [](const std::string& row) {
        return row.find("fixed_shapes=1") != std::string::npos &&
            row.find("safe_to_program=1") != std::string::npos;
      });
  const bool descriptor_table_complete = []() {
    std::lock_guard<std::mutex> lock(stack_resource_binding_manifest_mutex());
    return !stack_descriptor_binding_validation_rows().empty() &&
        std::all_of(
            stack_descriptor_binding_validation_rows().begin(),
            stack_descriptor_binding_validation_rows().end(),
            [](const auto& entry) {
              return entry.second.find("table_complete=1") != std::string::npos;
            });
  }();
  const bool ready_for_re_record_per_forward = []() {
    std::lock_guard<std::mutex> lock(stack_resource_binding_manifest_mutex());
    return !stack_descriptor_binding_validation_rows().empty() &&
        std::all_of(
            stack_descriptor_binding_validation_rows().begin(),
            stack_descriptor_binding_validation_rows().end(),
            [](const auto& entry) {
              return entry.second.find(
                         "ready_for_re_record_per_forward=1") !=
                  std::string::npos;
            });
  }();
  const auto capture_readiness = stack_capture_readiness_snapshot();
  const bool no_cpu_fallback =
      capture_readiness.size() > 1 && capture_readiness[1] != 0;
  const bool no_host_sync =
      capture_readiness.size() > 2 && capture_readiness[2] != 0;
  const bool no_nested_replay =
      capture_readiness.size() > 3 && capture_readiness[3] != 0;
  const bool no_active_capture = !has_explicit_runtime_capture_label();
  const bool command_recording_scope_available = false;
  const bool barriers_recordable = false;
  const bool descriptors_recordable =
      descriptor_table_complete && ready_for_re_record_per_forward;
  const bool resources_lifetime_tracked = true;
  const bool safe_to_record_stack_per_forward = shape_plan_ready &&
      descriptor_table_complete && ready_for_re_record_per_forward &&
      no_cpu_fallback && no_host_sync && no_nested_replay &&
      no_active_capture && command_recording_scope_available &&
      barriers_recordable && descriptors_recordable &&
      resources_lifetime_tracked;
  return {
      shape_plan_ready ? 1 : 0,
      descriptor_table_complete ? 1 : 0,
      ready_for_re_record_per_forward ? 1 : 0,
      no_cpu_fallback ? 1 : 0,
      no_host_sync ? 1 : 0,
      no_nested_replay ? 1 : 0,
      no_active_capture ? 1 : 0,
      command_recording_scope_available ? 1 : 0,
      barriers_recordable ? 1 : 0,
      descriptors_recordable ? 1 : 0,
      resources_lifetime_tracked ? 1 : 0,
      safe_to_record_stack_per_forward ? 1 : 0,
  };
}

std::vector<int64_t> stack_planned_recording_counters_snapshot() {
  const auto& counters = vulkan_stack_planned_recording_counters();
  return {
      static_cast<int64_t>(
          counters.total_attempts.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.planned_record_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.recording_scope_begin_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.recording_scope_submit_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.recording_scope_reject_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_readiness.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_active_capture.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_nested_replay.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_barrier.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_descriptor.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_lifetime.load(std::memory_order_relaxed)),
  };
}

void reset_stack_planned_recording_counters() {
  auto& counters = vulkan_stack_planned_recording_counters();
  counters.total_attempts.store(0u, std::memory_order_relaxed);
  counters.planned_record_hit.store(0u, std::memory_order_relaxed);
  counters.recording_scope_begin_count.store(0u, std::memory_order_relaxed);
  counters.recording_scope_submit_count.store(0u, std::memory_order_relaxed);
  counters.recording_scope_reject_count.store(0u, std::memory_order_relaxed);
  counters.reject_readiness.store(0u, std::memory_order_relaxed);
  counters.reject_active_capture.store(0u, std::memory_order_relaxed);
  counters.reject_nested_replay.store(0u, std::memory_order_relaxed);
  counters.reject_barrier.store(0u, std::memory_order_relaxed);
  counters.reject_descriptor.store(0u, std::memory_order_relaxed);
  counters.reject_lifetime.store(0u, std::memory_order_relaxed);
}

std::vector<int64_t> stack_replay_readiness_snapshot() {
  const auto shape_readiness = stack_shape_plan_readiness_snapshot();
  const bool fixed_shape_plan = std::any_of(
      shape_readiness.begin(),
      shape_readiness.end(),
      [](const std::string& row) {
        return row.find("fixed_shapes=1") != std::string::npos &&
            row.find("safe_to_program=1") != std::string::npos;
      });
  const bool resources_classified = []() {
    std::lock_guard<std::mutex> lock(stack_resource_binding_manifest_mutex());
    return !stack_resource_binding_manifest_rows().empty();
  }();
  const bool descriptor_table_complete = []() {
    std::lock_guard<std::mutex> lock(stack_resource_binding_manifest_mutex());
    return !stack_descriptor_binding_validation_rows().empty() &&
        std::all_of(
            stack_descriptor_binding_validation_rows().begin(),
            stack_descriptor_binding_validation_rows().end(),
            [](const auto& entry) {
              return entry.second.find("table_complete=1") != std::string::npos;
            });
  }();
  const bool descriptor_indices_known = []() {
    std::lock_guard<std::mutex> lock(stack_resource_binding_manifest_mutex());
    return !stack_descriptor_binding_validation_rows().empty() &&
        std::all_of(
            stack_descriptor_binding_validation_rows().begin(),
            stack_descriptor_binding_validation_rows().end(),
            [](const auto& entry) {
              return entry.second.find("all_descriptor_indices_known=1") !=
                  std::string::npos;
            });
  }();
  const auto& shape_counters = vulkan_stack_shape_plan_counters();
  const bool runtime_bindings_validated =
      shape_counters.binding_valid_count.load(std::memory_order_relaxed) > 0u &&
      shape_counters.binding_invalid_count.load(std::memory_order_relaxed) == 0u;
  const bool descriptors_rebindable = descriptor_table_complete &&
      descriptor_indices_known &&
      []() {
        std::lock_guard<std::mutex> lock(stack_resource_binding_manifest_mutex());
        return std::all_of(
            stack_descriptor_binding_validation_rows().begin(),
            stack_descriptor_binding_validation_rows().end(),
            [](const auto& entry) {
              return entry.second.find(
                         "ready_for_re_record_per_forward=1") !=
                  std::string::npos;
            });
      }();
  const bool persistent_resources_stable = resources_classified;
  const bool internal_temps_owned = descriptors_rebindable;
  const bool escaping_outputs_marked = resources_classified;
  const auto capture_readiness = stack_capture_readiness_snapshot();
  const bool no_cpu_fallback =
      capture_readiness.size() > 1 && capture_readiness[1] != 0;
  const bool no_host_sync = capture_readiness.size() > 2 && capture_readiness[2] != 0;
  const bool no_nested_replay =
      capture_readiness.size() > 3 && capture_readiness[3] != 0;
  const bool no_queue_idle =
      api::vulkan_sync_counters().queue_wait_idle_count.load(
          std::memory_order_relaxed) == 0u;
  const bool command_capture_safe = fixed_shape_plan && resources_classified &&
      runtime_bindings_validated && descriptor_table_complete &&
      descriptor_indices_known && descriptors_rebindable &&
      persistent_resources_stable && internal_temps_owned &&
      escaping_outputs_marked && no_cpu_fallback && no_host_sync &&
      no_nested_replay && no_queue_idle &&
      []() {
        std::lock_guard<std::mutex> lock(stack_resource_binding_manifest_mutex());
        return !stack_descriptor_binding_validation_rows().empty() &&
            std::all_of(
                stack_descriptor_binding_validation_rows().begin(),
                stack_descriptor_binding_validation_rows().end(),
                [](const auto& entry) {
                  return entry.second.find("ready_for_command_replay=1") !=
                      std::string::npos;
                });
      }();

  auto& replay_counters = vulkan_stack_replay_counters();
  replay_counters.total_attempts.fetch_add(1u, std::memory_order_relaxed);
  if (!command_capture_safe) {
    replay_counters.reject_readiness.fetch_add(1u, std::memory_order_relaxed);
    if (!descriptors_rebindable) {
      replay_counters.reject_binding_mode.fetch_add(
          1u,
          std::memory_order_relaxed);
    }
  }

  return {
      fixed_shape_plan ? 1 : 0,
      resources_classified ? 1 : 0,
      runtime_bindings_validated ? 1 : 0,
      descriptor_table_complete ? 1 : 0,
      descriptor_indices_known ? 1 : 0,
      descriptors_rebindable ? 1 : 0,
      persistent_resources_stable ? 1 : 0,
      internal_temps_owned ? 1 : 0,
      escaping_outputs_marked ? 1 : 0,
      no_cpu_fallback ? 1 : 0,
      no_host_sync ? 1 : 0,
      no_nested_replay ? 1 : 0,
      no_queue_idle ? 1 : 0,
      command_capture_safe ? 1 : 0,
  };
}

std::vector<std::string> stack_replay_binding_mode_snapshot() {
  std::lock_guard<std::mutex> lock(stack_resource_binding_manifest_mutex());
  std::vector<std::string> rows;
  rows.reserve(stack_replay_binding_mode_rows().size());
  for (const auto& entry : stack_replay_binding_mode_rows()) {
    rows.emplace_back(entry.second);
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

std::vector<int64_t> stack_replay_counters_snapshot() {
  const auto& counters = vulkan_stack_replay_counters();
  return {
      static_cast<int64_t>(
          counters.total_attempts.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.capture_build_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.replay_hit_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.descriptor_rebind_count.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_readiness.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_binding_mode.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_binding_validation.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_context_identity.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_capability.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_runtime_capture_active.load(std::memory_order_relaxed)),
  };
}

void reset_stack_replay_counters() {
  auto& counters = vulkan_stack_replay_counters();
  counters.total_attempts.store(0u, std::memory_order_relaxed);
  counters.capture_build_count.store(0u, std::memory_order_relaxed);
  counters.replay_hit_count.store(0u, std::memory_order_relaxed);
  counters.descriptor_rebind_count.store(0u, std::memory_order_relaxed);
  counters.reject_readiness.store(0u, std::memory_order_relaxed);
  counters.reject_binding_mode.store(0u, std::memory_order_relaxed);
  counters.reject_binding_validation.store(0u, std::memory_order_relaxed);
  counters.reject_context_identity.store(0u, std::memory_order_relaxed);
  counters.reject_capability.store(0u, std::memory_order_relaxed);
  counters.reject_runtime_capture_active.store(0u, std::memory_order_relaxed);
}

std::string validate_stack_shape_plan_binding(
    const c10::intrusive_ptr<VisionBackboneStackContext>& context,
    const int64_t planned_tokens,
    const Tensor& input,
    IntArrayRef capture_indices) {
  TORCH_CHECK(context, "Vision stack shape plan validation expects a context");
  const VulkanStackPlanRuntimeBinding binding =
      make_stack_plan_runtime_binding(input, capture_indices);
  std::lock_guard<std::mutex> lock(context->shape_plan_mutex());
  for (const auto& entry : context->shape_plans()) {
    const auto& plan = *entry.second;
    if (plan.key.tokens != planned_tokens) {
      continue;
    }
    std::string reason;
    const bool valid = validate_stack_plan_binding_impl(plan, binding, &reason);
    if (valid) {
      vulkan_stack_shape_plan_counters().binding_valid_count.fetch_add(
          1u,
          std::memory_order_relaxed);
    } else {
      note_stack_plan_binding_invalid(reason);
    }
    return reason;
  }
  vulkan_stack_shape_plan_counters().binding_invalid_count.fetch_add(
      1u,
      std::memory_order_relaxed);
  vulkan_stack_shape_plan_counters().invalid_context_identity.fetch_add(
      1u,
      std::memory_order_relaxed);
  return "plan_not_found";
}

VisionBackboneStackContext::VisionBackboneStackContext(
    std::vector<c10::intrusive_ptr<VisionBackboneBlockContext>> blocks,
    const int64_t num_heads,
    const int64_t head_dim,
    const int64_t hidden,
    const int64_t mlp_hidden)
    : blocks_(std::move(blocks)),
      num_heads_(num_heads),
      head_dim_(head_dim),
      hidden_(hidden),
      mlp_hidden_(mlp_hidden) {}

VisionBackboneBlockContext::VisionBackboneBlockContext(
    const Tensor& norm1_weight,
    const Tensor& norm1_bias,
    const double norm1_eps,
    const Tensor& qkv_weight,
    const std::optional<Tensor>& qkv_bias,
    const std::optional<Tensor>& attention_bias,
    const int64_t num_heads,
    const Tensor& proj_weight,
    const std::optional<Tensor>& proj_bias,
    const std::optional<Tensor>& ls1_gamma,
    const Tensor& norm2_weight,
    const Tensor& norm2_bias,
    const double norm2_eps,
    const Tensor& fc1_weight,
    const std::optional<Tensor>& fc1_bias,
    const Tensor& fc2_weight,
    const std::optional<Tensor>& fc2_bias,
    const std::optional<Tensor>& ls2_gamma,
    std::string allocation_label)
    : cache_id_(
          g_next_vision_backbone_context_cache_id.fetch_add(
              1u,
              std::memory_order_relaxed)),
      allocation_label_(std::move(allocation_label)),
      norm1_context_(make_layernorm_context(
          norm1_weight,
          norm1_bias,
          norm1_eps,
          child_label(allocation_label_, "norm1"))),
      qkv_context_(
          make_qkv_context(qkv_weight, child_label(allocation_label_, "qkv"))),
      qkv_bias_(move_optional_to_vulkan_buffer(qkv_bias)),
      attention_bias_(move_optional_to_vulkan_buffer(attention_bias)),
      num_heads_(num_heads),
      proj_context_(make_linear_context(
          proj_weight,
          proj_bias,
          child_label(allocation_label_, "proj"))),
      ls1_gamma_(move_optional_to_vulkan_buffer(ls1_gamma)),
      norm2_context_(make_layernorm_context(
          norm2_weight,
          norm2_bias,
          norm2_eps,
          child_label(allocation_label_, "norm2"))),
      fc1_context_(
          make_linear_context(fc1_weight, fc1_bias, child_label(allocation_label_, "fc1"))),
      fc2_context_(
          make_linear_context(fc2_weight, fc2_bias, child_label(allocation_label_, "fc2"))),
      ls2_gamma_(move_optional_to_vulkan_buffer(ls2_gamma)) {
  unpacked_.reserve(Unpacked::NumArgs);
  unpacked_.emplace_back(
      cpu_snapshot_for_unpack(norm1_weight, "unpack_norm1_weight_readback"));
  unpacked_.emplace_back(
      cpu_snapshot_for_unpack(norm1_bias, "unpack_norm1_bias_readback"));
  unpacked_.emplace_back(norm1_eps);
  unpacked_.emplace_back(
      cpu_snapshot_for_unpack(qkv_weight, "unpack_qkv_weight_readback"));
  if (qkv_bias.has_value()) {
    unpacked_.emplace_back(
        cpu_snapshot_for_unpack(*qkv_bias, "unpack_qkv_bias_readback"));
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  if (attention_bias.has_value()) {
    unpacked_.emplace_back(cpu_snapshot_for_unpack(
        *attention_bias, "unpack_attention_bias_readback"));
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(num_heads_);
  unpacked_.emplace_back(
      cpu_snapshot_for_unpack(proj_weight, "unpack_proj_weight_readback"));
  if (proj_bias.has_value()) {
    unpacked_.emplace_back(
        cpu_snapshot_for_unpack(*proj_bias, "unpack_proj_bias_readback"));
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  if (ls1_gamma.has_value()) {
    unpacked_.emplace_back(
        cpu_snapshot_for_unpack(*ls1_gamma, "unpack_ls1_gamma_readback"));
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(
      cpu_snapshot_for_unpack(norm2_weight, "unpack_norm2_weight_readback"));
  unpacked_.emplace_back(
      cpu_snapshot_for_unpack(norm2_bias, "unpack_norm2_bias_readback"));
  unpacked_.emplace_back(norm2_eps);
  unpacked_.emplace_back(
      cpu_snapshot_for_unpack(fc1_weight, "unpack_fc1_weight_readback"));
  if (fc1_bias.has_value()) {
    unpacked_.emplace_back(
        cpu_snapshot_for_unpack(*fc1_bias, "unpack_fc1_bias_readback"));
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(
      cpu_snapshot_for_unpack(fc2_weight, "unpack_fc2_weight_readback"));
  if (fc2_bias.has_value()) {
    unpacked_.emplace_back(
        cpu_snapshot_for_unpack(*fc2_bias, "unpack_fc2_bias_readback"));
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  if (ls2_gamma.has_value()) {
    unpacked_.emplace_back(
        cpu_snapshot_for_unpack(*ls2_gamma, "unpack_ls2_gamma_readback"));
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(allocation_label_);
}

VisionBackboneBlockContext VisionBackboneBlockContext::pack(
    c10::impl::GenericList unpacked) {
  const uint32_t unpacked_size =
      api::utils::safe_downcast<uint32_t>(unpacked.size());
  const bool has_attention_bias =
      unpacked_size >= VisionBackboneBlockContext::Unpacked::NumArgs;
  constexpr uint32_t legacy_num_heads = 5u;
  constexpr uint32_t legacy_proj_weight = 6u;
  constexpr uint32_t legacy_proj_bias = 7u;
  constexpr uint32_t legacy_ls1_gamma = 8u;
  constexpr uint32_t legacy_norm2_weight = 9u;
  constexpr uint32_t legacy_norm2_bias = 10u;
  constexpr uint32_t legacy_norm2_eps = 11u;
  constexpr uint32_t legacy_fc1_weight = 12u;
  constexpr uint32_t legacy_fc1_bias = 13u;
  constexpr uint32_t legacy_fc2_weight = 14u;
  constexpr uint32_t legacy_fc2_bias = 15u;
  constexpr uint32_t legacy_ls2_gamma = 16u;
  constexpr uint32_t legacy_label = 17u;
  return VisionBackboneBlockContext(
      unpacked.get(Unpacked::Norm1Weight).toTensor(),
      unpacked.get(Unpacked::Norm1Bias).toTensor(),
      unpacked.get(Unpacked::Norm1Eps).toDouble(),
      unpacked.get(Unpacked::QkvWeight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::QkvBias),
      has_attention_bias ? get_optional_tensor(unpacked, Unpacked::AttentionBias)
                         : std::optional<Tensor>{},
      unpacked.get(has_attention_bias ? Unpacked::NumHeads : legacy_num_heads)
          .toInt(),
      unpacked
          .get(has_attention_bias ? Unpacked::ProjWeight : legacy_proj_weight)
          .toTensor(),
      get_optional_tensor(
          unpacked, has_attention_bias ? Unpacked::ProjBias : legacy_proj_bias),
      get_optional_tensor(
          unpacked, has_attention_bias ? Unpacked::Ls1Gamma : legacy_ls1_gamma),
      unpacked
          .get(has_attention_bias ? Unpacked::Norm2Weight : legacy_norm2_weight)
          .toTensor(),
      unpacked
          .get(has_attention_bias ? Unpacked::Norm2Bias : legacy_norm2_bias)
          .toTensor(),
      unpacked
          .get(has_attention_bias ? Unpacked::Norm2Eps : legacy_norm2_eps)
          .toDouble(),
      unpacked
          .get(has_attention_bias ? Unpacked::Fc1Weight : legacy_fc1_weight)
          .toTensor(),
      get_optional_tensor(
          unpacked, has_attention_bias ? Unpacked::Fc1Bias : legacy_fc1_bias),
      unpacked
          .get(has_attention_bias ? Unpacked::Fc2Weight : legacy_fc2_weight)
          .toTensor(),
      get_optional_tensor(
          unpacked, has_attention_bias ? Unpacked::Fc2Bias : legacy_fc2_bias),
      get_optional_tensor(
          unpacked, has_attention_bias ? Unpacked::Ls2Gamma : legacy_ls2_gamma),
      unpacked
          .get(has_attention_bias ? Unpacked::Label : legacy_label)
          .toStringRef());
}

c10::intrusive_ptr<VisionBackboneBlockContext>
create_vision_backbone_block_context(
    Tensor&& norm1_weight,
    Tensor&& norm1_bias,
    const double norm1_eps,
    Tensor&& qkv_weight,
    std::optional<Tensor>&& qkv_bias,
    const int64_t num_heads,
    Tensor&& proj_weight,
    std::optional<Tensor>&& proj_bias,
    std::optional<Tensor>&& ls1_gamma,
    Tensor&& norm2_weight,
    Tensor&& norm2_bias,
    const double norm2_eps,
    Tensor&& fc1_weight,
    std::optional<Tensor>&& fc1_bias,
    Tensor&& fc2_weight,
    std::optional<Tensor>&& fc2_bias,
    std::optional<Tensor>&& ls2_gamma,
    std::string label) {
  recover_after_vulkan_failure_if_needed();
  vulkan_vision_owner_context_counters().create_count.fetch_add(
      1u,
      std::memory_order_relaxed);
  return c10::make_intrusive<VisionBackboneBlockContext>(
      norm1_weight,
      norm1_bias,
      norm1_eps,
      qkv_weight,
      qkv_bias,
      std::nullopt,
      num_heads,
      proj_weight,
      proj_bias,
      ls1_gamma,
      norm2_weight,
      norm2_bias,
      norm2_eps,
      fc1_weight,
      fc1_bias,
      fc2_weight,
      fc2_bias,
      ls2_gamma,
      std::move(label));
}

c10::intrusive_ptr<VisionBackboneStackContext>
create_vision_backbone_stack_context(
    const c10::List<c10::intrusive_ptr<VisionBackboneBlockContext>>& blocks,
    const int64_t num_heads,
    const int64_t head_dim,
    const int64_t hidden,
    const int64_t mlp_hidden) {
  recover_after_vulkan_failure_if_needed();
  TORCH_CHECK(
      blocks.size() > 0,
      "Vision backbone stack context expects at least one block context");

  std::vector<c10::intrusive_ptr<VisionBackboneBlockContext>> block_contexts;
  block_contexts.reserve(blocks.size());
  for (const auto& context_ref : blocks) {
    c10::intrusive_ptr<VisionBackboneBlockContext> context = context_ref;
    TORCH_CHECK(
        static_cast<bool>(context),
        "Vision backbone stack context expects defined block contexts");
    block_contexts.push_back(std::move(context));
  }

  return c10::make_intrusive<VisionBackboneStackContext>(
      std::move(block_contexts), num_heads, head_dim, hidden, mlp_hidden);
}

c10::intrusive_ptr<VisionBackboneBlockContext>
create_vision_backbone_block_context_with_attention_bias(
    Tensor&& norm1_weight,
    Tensor&& norm1_bias,
    const double norm1_eps,
    Tensor&& qkv_weight,
    std::optional<Tensor>&& qkv_bias,
    std::optional<Tensor>&& attention_bias,
    const int64_t num_heads,
    Tensor&& proj_weight,
    std::optional<Tensor>&& proj_bias,
    std::optional<Tensor>&& ls1_gamma,
    Tensor&& norm2_weight,
    Tensor&& norm2_bias,
    const double norm2_eps,
    Tensor&& fc1_weight,
    std::optional<Tensor>&& fc1_bias,
    Tensor&& fc2_weight,
    std::optional<Tensor>&& fc2_bias,
    std::optional<Tensor>&& ls2_gamma,
    std::string label) {
  recover_after_vulkan_failure_if_needed();
  vulkan_vision_owner_context_counters().create_count.fetch_add(
      1u,
      std::memory_order_relaxed);
  return c10::make_intrusive<VisionBackboneBlockContext>(
      norm1_weight,
      norm1_bias,
      norm1_eps,
      qkv_weight,
      qkv_bias,
      attention_bias,
      num_heads,
      proj_weight,
      proj_bias,
      ls1_gamma,
      norm2_weight,
      norm2_bias,
      norm2_eps,
      fc1_weight,
      fc1_bias,
      fc2_weight,
      fc2_bias,
      ls2_gamma,
      std::move(label));
}

Tensor run_vision_backbone_block_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context) {
  recover_after_vulkan_failure_if_needed();
  auto& owner_counters = vulkan_vision_owner_counters();
  owner_counters.total_attempts.fetch_add(1u, std::memory_order_relaxed);
  if (!context) {
    owner_counters.reject_missing_context.fetch_add(
        1u,
        std::memory_order_relaxed);
    append_vulkan_vision_owner_log(
        "block", false, "missing_context", input_arg, context);
  }
  TORCH_CHECK(context, "Vision backbone block context is required");
  api::AllocationScope allocation_scope(context->allocation_label());
  TORCH_CHECK(
      input_arg.dim() == 2 || input_arg.dim() == 3,
      "Vision backbone block context expects rank-2 or rank-3 input");
  append_vulkan_vision_owner_log("block", true, "none", input_arg, context);
  owner_counters.block_owner_hit.fetch_add(1u, std::memory_order_relaxed);
  utils::validate_replay_tensor_not_stale(
      input_arg, "vulkan_prepack::run_vision_backbone_block_context");

  const Device output_device = input_arg.device();
  const ScalarType output_dtype = input_arg.scalar_type();
  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const bool has_context_attention_bias = context->attention_bias().defined();
  const bool has_runtime_capture_label = has_explicit_runtime_capture_label();
  std::optional<utils::VulkanPlanningRequestScope> planning_scope;
  if (has_runtime_capture_label) {
    planning_scope.emplace(utils::make_vulkan_vision_backbone_request());
  }
  const auto runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_vision_backbone_request());
  auto vision_graph = (has_context_attention_bias || !has_runtime_capture_label)
      ? utils::VisionBackboneInferenceGraph{}
      : prime_vision_backbone_graph(input, runtime_policy, context);
  std::optional<utils::ScratchArena> graph_scratch = std::nullopt;
  if (vision_graph.defined() && runtime_policy.scratch_arena_plan.has_value()) {
    const int64_t batch_size = input.dim() == 2 ? 1 : input.size(0);
    const int64_t token_count = input.dim() == 2 ? input.size(0) : input.size(1);
    const int64_t embed_dim = input.size(-1);
    const uint32_t scratch_alignment = std::max<uint32_t>(
        runtime_policy.scratch_arena_plan->alignment,
        static_cast<uint32_t>(
            std::max<int64_t>(1, static_cast<int64_t>(c10::elementSize(kFloat)))));
    const size_t requested_bytes = vision_attention_scratch_bytes(
        batch_size,
        token_count,
        embed_dim,
        context->num_heads(),
        input.scalar_type(),
        context->qkv_bias().defined(),
        scratch_alignment);
    if (
        requested_bytes > 0u &&
        runtime_policy.scratch_arena_plan->prefer_buffer_storage) {
      graph_scratch = vision_graph.ensure_shared_scratch(
          std::max(
              requested_bytes,
              runtime_policy.scratch_arena_plan->min_arena_bytes),
          scratch_alignment,
          runtime_policy.execution_program_plan.has_value() &&
              runtime_policy.execution_program_plan->persistent);
    }
  }

  if (graph_scratch.has_value()) {
    graph_scratch->reset();
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_backbone_block_context.graph");
  }
  const int64_t batch_size = input.dim() == 2 ? 1 : input.size(0);
  const int64_t token_count = input.dim() == 2 ? input.size(0) : input.size(1);
  const int64_t embed_dim = input.size(-1);
  const int64_t hidden_dim = vision_block_hidden_dim(context);
  const bool allow_vision_replay = has_runtime_capture_label;
  const std::string backbone_program_label =
      vision_backbone_program_label(context->allocation_label(), context.get());
  std::optional<api::RuntimeLabelScope> execution_runtime_scope;
  if (has_explicit_runtime_capture_label()) {
    execution_runtime_scope.emplace(
        compose_runtime_capture_label(
            vision_backbone_execution_label(
                context->allocation_label(), context.get())));
  }

  if (
      !has_context_attention_bias &&
      allow_vision_replay &&
      vision_graph.defined() &&
      runtime_policy.execution_program_plan.has_value() &&
      input.scalar_type() == kFloat) {
    auto vision_replay = vision_graph.lookup_or_create_replay(
        backbone_program_label,
        input.sizes(),
        token_count,
        embed_dim,
        hidden_dim,
        context->num_heads(),
        *runtime_policy.execution_program_plan);
    if (vision_replay.defined()) {
      copy_tensor_for_replay(vision_replay.input_slot(), input);
      api::context()->flush_pending_cmds();

      if (!vision_replay.recorded()) {
        Tensor warmup_output = utils::create_buffer_tensor(
            vision_replay.output_slot().sizes(),
            vision_replay.output_slot().scalar_type(),
            /*persistent=*/output_device.type() == kVulkan);
        if (graph_scratch.has_value()) {
          graph_scratch->reset();
        }
        (void)run_vision_backbone_block_program(
            vision_replay.input_slot(),
            context,
            &vision_replay.program(),
            graph_scratch.has_value() ? &(*graph_scratch) : nullptr,
            &vision_replay.output_slot());
        copy_tensor_for_replay(warmup_output, vision_replay.output_slot());
        api::context()->flush_pending_cmds();
        vision_replay.replay().record([&]() {
          if (graph_scratch.has_value()) {
            graph_scratch->reset();
          }
          (void)run_vision_backbone_block_program(
              vision_replay.input_slot(),
              context,
              &vision_replay.program(),
              graph_scratch.has_value() ? &(*graph_scratch) : nullptr,
              &vision_replay.output_slot());
        });
        utils::log_vulkan_op_hit(
            "vulkan_prepack::run_vision_backbone_block_context.replay_warmup");
        utils::log_vulkan_op_hit(
            "vulkan_prepack::run_vision_backbone_block_context");
        Tensor restored =
            maybe_restore_tensor(warmup_output, output_device, output_dtype);
        record_tensor_write(
            restored,
            "vulkan_prepack::run_vision_backbone_block_context",
            "replay_warmup_materialized",
            {input_arg});
        return restored;
      }

      const utils::ReplayEpoch epoch = utils::begin_replay_epoch(
          vision_replay.identity(), "vision.backbone_block.replay");
      const std::string detail =
          "action=run_backbone_block_direct "
          "reason=vision_replay_submit_guard "
          "failure_class=ReplayHangRisk";
      utils::log_replay_event(
          "vision_replay_submit_guard",
          vision_replay.identity(),
          epoch.run_id,
          "vision.backbone_block.replay",
          detail);
      api::report_vulkan_failure(
          api::VulkanFailureClass::ReplayHangRisk,
          "vulkan_prepack::run_vision_backbone_block_context",
          "VisionReplaySubmitGuard",
          detail);
      utils::log_inference_replay_lifecycle_event(
          vision_replay.replay(), "direct");
      if (graph_scratch.has_value()) {
        graph_scratch->reset();
      }
      (void)run_vision_backbone_block_program(
          vision_replay.input_slot(),
          context,
          &vision_replay.program(),
          graph_scratch.has_value() ? &(*graph_scratch) : nullptr,
          &vision_replay.output_slot());
      api::context()->flush_pending_cmds();
      Tensor output = utils::create_buffer_tensor(
          vision_replay.output_slot().sizes(),
          vision_replay.output_slot().scalar_type(),
          /*persistent=*/output_device.type() == kVulkan);
      copy_tensor_for_replay(output, vision_replay.output_slot());
      utils::log_vulkan_op_hit(
          "vulkan_prepack::run_vision_backbone_block_context.replay");
      utils::log_vulkan_op_hit(
          "vulkan_prepack::run_vision_backbone_block_context");
      Tensor restored = maybe_restore_tensor(output, output_device, output_dtype);
      record_tensor_write(
          restored,
          "vulkan_prepack::run_vision_backbone_block_context",
          "replay_materialized",
          {input_arg});
      return restored;
    }
  }

  auto vision_program = (has_context_attention_bias || !has_runtime_capture_label)
      ? utils::VisionBackboneProgram{}
      : (vision_graph.defined()
             ? vision_graph.lookup_or_create_program(
                   backbone_program_label,
                   input.scalar_type(),
                   batch_size,
                   token_count,
                   embed_dim,
                   hidden_dim,
                   context->num_heads(),
                   *runtime_policy.execution_program_plan)
             : prime_vision_backbone_program(
                   input, context, runtime_policy, graph_scratch.has_value()));
  if (vision_program.defined()) {
    if (!graph_scratch.has_value() && vision_program.scratch_arena().has_value()) {
      vision_program.scratch_arena()->reset();
    }
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_backbone_block_context.program");
  }
  utils::VisionBackboneProgram* const vision_program_ptr =
      vision_program.defined() ? &vision_program : nullptr;

  Tensor output = run_vision_backbone_block_program(
      input,
      context,
      vision_program_ptr,
      graph_scratch.has_value() ? &(*graph_scratch) : nullptr);
  if (vision_program_ptr != nullptr) {
    output = materialize_escaping_vulkan_output(
        output, output_device.type() == kVulkan);
    api::context()->flush_pending_cmds();
  }
  utils::log_vulkan_op_hit("vulkan_prepack::run_vision_backbone_block_context");
  Tensor restored = maybe_restore_tensor(output, output_device, output_dtype);
  record_tensor_write(
      restored,
      "vulkan_prepack::run_vision_backbone_block_context",
      vision_program_ptr != nullptr ? "program_materialized" : "direct",
      {input_arg});
  return restored;
}

std::vector<Tensor> run_vision_backbone_stack_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<VisionBackboneStackContext>& context,
    IntArrayRef capture_indices) {
  recover_after_vulkan_failure_if_needed();
  api::VulkanVisionStackPhaseScope stack_entry_scope(
      api::VulkanVisionStackPhase::StackEntry);
  auto& stack_counters = vulkan_vision_stack_owner_counters();
  stack_counters.total_attempts.fetch_add(1u, std::memory_order_relaxed);

  if (!context) {
    stack_counters.reject_missing_context.fetch_add(
        1u,
        std::memory_order_relaxed);
    append_vulkan_vision_stack_owner_log(
        false, "missing_context", input_arg, context);
  }
  TORCH_CHECK(context, "Vision backbone stack context is required");
  if (context->blocks().empty()) {
    stack_counters.reject_missing_context.fetch_add(
        1u,
        std::memory_order_relaxed);
    append_vulkan_vision_stack_owner_log(
        false, "empty_context", input_arg, context);
  }
  TORCH_CHECK(
      !context->blocks().empty(),
      "Vision backbone stack context expects at least one block context");

  if (input_arg.scalar_type() != kFloat) {
    stack_counters.reject_dtype.fetch_add(1u, std::memory_order_relaxed);
    append_vulkan_vision_stack_owner_log(false, "dtype", input_arg, context);
  }
  TORCH_CHECK(
      input_arg.scalar_type() == kFloat,
      "Vision backbone stack context is FP32 for now");
  if (input_arg.dim() != 2 && input_arg.dim() != 3) {
    stack_counters.reject_shape.fetch_add(1u, std::memory_order_relaxed);
    append_vulkan_vision_stack_owner_log(false, "shape", input_arg, context);
  }
  TORCH_CHECK(
      input_arg.dim() == 2 || input_arg.dim() == 3,
      "Vision backbone stack context expects rank-2 or rank-3 input");
  if (context->hidden() > 0 && input_arg.size(input_arg.dim() - 1) != context->hidden()) {
    stack_counters.reject_shape.fetch_add(1u, std::memory_order_relaxed);
    append_vulkan_vision_stack_owner_log(
        false, "hidden_mismatch", input_arg, context);
  }
  TORCH_CHECK(
      context->hidden() <= 0 ||
          input_arg.size(input_arg.dim() - 1) == context->hidden(),
      "Vision backbone stack hidden dimension mismatch: expected ",
      context->hidden(),
      " got ",
      input_arg.size(input_arg.dim() - 1));
  if (!input_arg.is_vulkan()) {
    stack_counters.reject_layout.fetch_add(1u, std::memory_order_relaxed);
    append_vulkan_vision_stack_owner_log(
        false, "not_vulkan", input_arg, context);
  }
  TORCH_CHECK(input_arg.is_vulkan(), "Vision backbone stack expects Vulkan input");
  if (has_explicit_runtime_capture_label()) {
    stack_counters.reject_unsafe_replay.fetch_add(
        1u,
        std::memory_order_relaxed);
    append_vulkan_vision_stack_owner_log(
        false, "unsafe_nested_replay", input_arg, context);
  }
  TORCH_CHECK(
      !has_explicit_runtime_capture_label(),
      "Vision backbone stack owner does not run under runtime capture labels");

  auto& owner_counters = vulkan_vision_owner_counters();
  owner_counters.total_attempts.fetch_add(1u, std::memory_order_relaxed);
  owner_counters.stack_owner_hit.fetch_add(1u, std::memory_order_relaxed);
  stack_counters.stack_owner_hit.fetch_add(1u, std::memory_order_relaxed);
  stack_counters.block_context_count.fetch_add(
      context->blocks().size(),
      std::memory_order_relaxed);
  append_vulkan_vision_stack_owner_log(true, "none", input_arg, context);

  std::vector<int64_t> capture_indices_vec = capture_indices.vec();
  if (capture_indices_vec.empty()) {
    capture_indices_vec.push_back(
        static_cast<int64_t>(context->blocks().size()) - 1);
  }
  for (const int64_t capture_idx : capture_indices_vec) {
    TORCH_CHECK(
        capture_idx >= 0 &&
            capture_idx < static_cast<int64_t>(context->blocks().size()),
        "Vision backbone stack capture index ",
        capture_idx,
        " is out of range for ",
        context->blocks().size(),
        " contexts");
  }

  VulkanVisionStackShapePlan* stack_shape_plan = nullptr;
  {
    VulkanVisionStackShapePlan& plan =
        get_or_create_stack_shape_plan(*context, input_arg, capture_indices_vec);
    stack_shape_plan = &plan;
    const VulkanStackPlanRuntimeBinding binding =
        make_stack_plan_runtime_binding(input_arg, capture_indices_vec);
    std::string reason;
    if (validate_stack_plan_binding_impl(plan, binding, &reason)) {
      vulkan_stack_shape_plan_counters().binding_valid_count.fetch_add(
          1u,
          std::memory_order_relaxed);
    } else {
      note_stack_plan_binding_invalid(reason);
    }
  }

  auto& planned_counters = vulkan_stack_planned_recording_counters();
  planned_counters.total_attempts.fetch_add(1u, std::memory_order_relaxed);
  if (stack_shape_plan &&
      stack_plan_ready_for_planned_recording(*stack_shape_plan)) {
    planned_counters.planned_record_hit.fetch_add(
        1u,
        std::memory_order_relaxed);
  } else {
    planned_counters.recording_scope_reject_count.fetch_add(
        1u,
        std::memory_order_relaxed);
    planned_counters.reject_readiness.fetch_add(
        1u,
        std::memory_order_relaxed);
    planned_counters.reject_barrier.fetch_add(
        1u,
        std::memory_order_relaxed);
  }

  Tensor current = input_arg;
  std::vector<Tensor> outputs(capture_indices_vec.size());
  for (size_t block_idx = 0u; block_idx < context->blocks().size(); ++block_idx) {
    const auto& block_context = context->blocks()[block_idx];
    TORCH_CHECK(
        static_cast<bool>(block_context),
        "Vision backbone stack context contains an undefined block context");
    {
      api::VulkanVisionStackBlockScope block_scope(
          static_cast<int64_t>(block_idx));
      api::VulkanVisionStackPhaseScope phase_scope(
          api::VulkanVisionStackPhase::BlockEntry);
      current = run_vision_backbone_block_context(current, block_context);
    }
    stack_counters.block_execute_count.fetch_add(
        1u,
        std::memory_order_relaxed);
    for (size_t capture_pos = 0u; capture_pos < capture_indices_vec.size();
         ++capture_pos) {
      if (capture_indices_vec[capture_pos] == static_cast<int64_t>(block_idx)) {
        api::VulkanVisionStackBlockScope block_scope(
            static_cast<int64_t>(block_idx));
        api::VulkanVisionStackPhaseScope phase_scope(
            api::VulkanVisionStackPhase::IntermediateCapture);
        const uint64_t bytes = static_cast<uint64_t>(current.numel()) *
            static_cast<uint64_t>(current.element_size());
        api::note_vulkan_stack_allocation(
            "vision_stack_capture",
            block_idx + 1u == context->blocks().size()
                ? api::VulkanStackTensorLifetimeClass::FinalStackOutput
                : api::VulkanStackTensorLifetimeClass::
                      RequestedIntermediateOutput,
            current.sizes().vec(),
            current.strides().vec(),
            static_cast<int64_t>(current.scalar_type()),
            current.is_vulkan(),
            current.is_vulkan(),
            false,
            true,
            true,
            bytes);
        note_stack_execution_manifest_row(
            "vision_stack.intermediate_capture",
            "none",
            {std::cref(current)},
            {std::cref(current)},
            false,
            false,
            true,
            true,
            false);
        outputs[capture_pos] = record_tensor_write_and_return(
            current,
            "vulkan_prepack::run_vision_backbone_stack_context",
            "vision_stack_capture",
            {input_arg});
      }
    }
  }

  api::VulkanVisionStackPhaseScope stack_exit_scope(
      api::VulkanVisionStackPhase::StackExit);
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_vision_backbone_stack_context");
  return outputs;
}

void prime_vision_backbone_block_context_graph(
    const Tensor& input_arg,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context) {
  if (!input_arg.defined() || !input_arg.is_vulkan() || !context) {
    return;
  }

  TORCH_CHECK(
      input_arg.dim() == 2 || input_arg.dim() == 3,
      "Vision backbone block graph priming expects rank-2 or rank-3 input");

  utils::VulkanPlanningRequestScope planning_scope(
      utils::make_vulkan_vision_backbone_request());
  const auto runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_vision_backbone_request());
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionBackbone) {
    return;
  }
  if (input_arg.scalar_type() != kFloat) {
    return;
  }

  auto vision_graph = prime_vision_backbone_graph(input_arg, runtime_policy, context);
  if (!vision_graph.defined()) {
    return;
  }

  if (runtime_policy.scratch_arena_plan.has_value()) {
    const int64_t batch_size = input_arg.dim() == 2 ? 1 : input_arg.size(0);
    const int64_t token_count =
        input_arg.dim() == 2 ? input_arg.size(0) : input_arg.size(1);
    const int64_t embed_dim = input_arg.size(-1);
    const uint32_t scratch_alignment = std::max<uint32_t>(
        runtime_policy.scratch_arena_plan->alignment,
        static_cast<uint32_t>(std::max<int64_t>(
            1, static_cast<int64_t>(c10::elementSize(kFloat)))));
    const size_t requested_bytes = vision_attention_scratch_bytes(
        batch_size,
        token_count,
        embed_dim,
        context->num_heads(),
        input_arg.scalar_type(),
        context->qkv_bias().defined(),
        scratch_alignment);
    if (
        requested_bytes > 0u &&
        runtime_policy.scratch_arena_plan->prefer_buffer_storage) {
      vision_graph.note_shared_scratch_requirement(
          std::max(
              requested_bytes,
              runtime_policy.scratch_arena_plan->min_arena_bytes),
          scratch_alignment,
          runtime_policy.execution_program_plan->persistent);
    }
  }

  (void)vision_graph.lookup_or_create_replay(
      vision_backbone_program_label(
          context->allocation_label(), context.get()),
      input_arg.sizes(),
      input_arg.dim() == 2 ? input_arg.size(0) : input_arg.size(1),
      input_arg.size(-1),
      vision_block_hidden_dim(context),
      context->num_heads(),
      *runtime_policy.execution_program_plan);
  utils::log_vulkan_op_hit(
      "vulkan_prepack::prime_vision_backbone_block_context_graph");
}

VisionDecoderFusionBlockContext::VisionDecoderFusionBlockContext(
    const Tensor& res1_conv1_weight,
    const std::optional<Tensor>& res1_conv1_bias,
    const Tensor& res1_conv2_weight,
    const std::optional<Tensor>& res1_conv2_bias,
    const Tensor& res2_conv1_weight,
    const std::optional<Tensor>& res2_conv1_bias,
    const Tensor& res2_conv2_weight,
    const std::optional<Tensor>& res2_conv2_bias,
    const Tensor& out_conv_weight,
    const std::optional<Tensor>& out_conv_bias,
    const bool align_corners,
    std::string allocation_label)
    : allocation_label_(std::move(allocation_label)),
      align_corners_(align_corners),
      res1_conv1_context_(make_conv2d_context(
          res1_conv1_weight,
          res1_conv1_bias,
          {1, 1},
          {1, 1})),
      res1_conv2_context_(make_conv2d_context(
          res1_conv2_weight,
          res1_conv2_bias,
          {1, 1},
          {1, 1})),
      res2_conv1_context_(make_conv2d_context(
          res2_conv1_weight,
          res2_conv1_bias,
          {1, 1},
          {1, 1})),
      res2_conv2_context_(make_conv2d_context(
          res2_conv2_weight,
          res2_conv2_bias,
          {1, 1},
          {1, 1})),
      out_conv_context_(make_conv2d_context(
          out_conv_weight,
          out_conv_bias,
          {1, 1},
          {0, 0})) {
  unpacked_.reserve(Unpacked::NumArgs);
  unpacked_.emplace_back(cpu_snapshot_for_unpack(
      res1_conv1_weight, "unpack_res1_conv1_weight_readback"));
  if (res1_conv1_bias.has_value()) {
    unpacked_.emplace_back(cpu_snapshot_for_unpack(
        *res1_conv1_bias, "unpack_res1_conv1_bias_readback"));
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(cpu_snapshot_for_unpack(
      res1_conv2_weight, "unpack_res1_conv2_weight_readback"));
  if (res1_conv2_bias.has_value()) {
    unpacked_.emplace_back(cpu_snapshot_for_unpack(
        *res1_conv2_bias, "unpack_res1_conv2_bias_readback"));
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(cpu_snapshot_for_unpack(
      res2_conv1_weight, "unpack_res2_conv1_weight_readback"));
  if (res2_conv1_bias.has_value()) {
    unpacked_.emplace_back(cpu_snapshot_for_unpack(
        *res2_conv1_bias, "unpack_res2_conv1_bias_readback"));
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(cpu_snapshot_for_unpack(
      res2_conv2_weight, "unpack_res2_conv2_weight_readback"));
  if (res2_conv2_bias.has_value()) {
    unpacked_.emplace_back(cpu_snapshot_for_unpack(
        *res2_conv2_bias, "unpack_res2_conv2_bias_readback"));
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(
      cpu_snapshot_for_unpack(out_conv_weight, "unpack_out_conv_weight_readback"));
  if (out_conv_bias.has_value()) {
    unpacked_.emplace_back(
        cpu_snapshot_for_unpack(*out_conv_bias, "unpack_out_conv_bias_readback"));
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(align_corners_);
  unpacked_.emplace_back(allocation_label_);
}

VisionDecoderFusionBlockContext VisionDecoderFusionBlockContext::pack(
    c10::impl::GenericList unpacked) {
  return VisionDecoderFusionBlockContext(
      unpacked.get(Unpacked::Res1Conv1Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Res1Conv1Bias),
      unpacked.get(Unpacked::Res1Conv2Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Res1Conv2Bias),
      unpacked.get(Unpacked::Res2Conv1Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Res2Conv1Bias),
      unpacked.get(Unpacked::Res2Conv2Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Res2Conv2Bias),
      unpacked.get(Unpacked::OutConvWeight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::OutConvBias),
      unpacked.get(Unpacked::AlignCorners).toBool(),
      unpacked.get(Unpacked::Label).toStringRef());
}

c10::intrusive_ptr<VisionDecoderFusionBlockContext>
create_vision_decoder_fusion_block_context(
    Tensor&& res1_conv1_weight,
    std::optional<Tensor>&& res1_conv1_bias,
    Tensor&& res1_conv2_weight,
    std::optional<Tensor>&& res1_conv2_bias,
    Tensor&& res2_conv1_weight,
    std::optional<Tensor>&& res2_conv1_bias,
    Tensor&& res2_conv2_weight,
    std::optional<Tensor>&& res2_conv2_bias,
    Tensor&& out_conv_weight,
    std::optional<Tensor>&& out_conv_bias,
    const bool align_corners,
    std::string label) {
  return c10::make_intrusive<VisionDecoderFusionBlockContext>(
      res1_conv1_weight,
      res1_conv1_bias,
      res1_conv2_weight,
      res1_conv2_bias,
      res2_conv1_weight,
      res2_conv1_bias,
      res2_conv2_weight,
      res2_conv2_bias,
      out_conv_weight,
      out_conv_bias,
      align_corners,
      std::move(label));
}

Tensor run_vision_decoder_fusion_block_context(
    const Tensor& input_arg,
    const std::optional<Tensor>& skip_arg,
    const std::optional<std::vector<int64_t>>& size,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& context) {
  TORCH_CHECK(
      input_arg.dim() == 4,
      "Vision decoder fusion block context expects rank-4 input");

  const Device output_device = input_arg.device();
  const ScalarType output_dtype = input_arg.scalar_type();
  const auto fallback =
      [&](const Tensor& input_tensor,
          const std::optional<Tensor>& skip_tensor) -> Tensor {
    Tensor main_input = input_tensor;
    if (skip_tensor.has_value() && skip_tensor->defined()) {
      Tensor residual = at::relu(*skip_tensor);
      residual = run_conv2d_context(residual, context->res1_conv1_context());
      residual = at::relu(residual);
      residual = run_conv2d_context(residual, context->res1_conv2_context());
      main_input = at::add(input_tensor, at::add(residual, *skip_tensor));
    }

    Tensor output = at::relu(main_input);
    output = run_conv2d_context(output, context->res2_conv1_context());
    output = at::relu(output);
    output = run_conv2d_context(output, context->res2_conv2_context());
    output = at::add(output, main_input);
    output = at::upsample_bilinear2d(
        output,
        resolve_decoder_target_sizes(input_tensor, size),
        context->align_corners(),
        std::nullopt,
        std::nullopt);
    output = run_conv2d_context(output, context->out_conv_context());
    return maybe_restore_tensor(output, output_device, output_dtype);
  };

  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  std::optional<Tensor> skip =
      (skip_arg.has_value() && skip_arg->defined())
      ? std::optional<Tensor>(skip_arg->is_vulkan() ? *skip_arg : skip_arg->vulkan())
      : std::nullopt;

  const std::vector<int64_t> target_sizes =
      resolve_decoder_target_sizes(input, size);
  utils::VulkanPlanningRequestScope planning_scope(
      utils::make_vulkan_vision_decoder_request());

  if (input.scalar_type() != kFloat || input.device().type() != kVulkan) {
    return fallback(input, skip);
  }

  if (!can_use_decoder_fusion_program_context(context)) {
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_decoder_fusion_block_context.program_skip.non_buffer_direct_context");
    return fallback(input, skip);
  }

  Tensor main_input = prepare_decoder_buffer_tensor(input);
  if (main_input.dim() != 4) {
    return fallback(input, skip);
  }

  std::optional<Tensor> skip_tensor = std::nullopt;
  if (skip.has_value() && skip->defined()) {
    skip_tensor = prepare_decoder_buffer_tensor(*skip);
  }

  const auto runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_vision_decoder_request());
  // The implicit scale_factor=2 path (size omitted) is still unreliable under
  // the shared graph/replay path after shape transitions. Keep it on the
  // standalone Vulkan program path until the graph slot lifecycle is tightened.
  const bool allow_decoder_graph = size.has_value();
  auto vision_graph = allow_decoder_graph
      ? prime_vision_decoder_graph(main_input, runtime_policy, context)
      : utils::VisionDecoderInferenceGraph{};
  if (allow_decoder_graph) {
    const std::string detail =
        "action=use_decoder_program_path "
        "reason=decoder_replay_guard "
        "failure_class=ReplayHangRisk";
    utils::log_replay_event(
        "decoder_replay_guard",
        vision_graph.identity(),
        utils::current_replay_epoch(vision_graph.identity()).run_id,
        "vision.decoder_fusion.replay",
        detail);
    api::report_vulkan_failure(
        api::VulkanFailureClass::ReplayHangRisk,
        "vulkan_prepack::run_vision_decoder_fusion_block_context",
        "DecoderReplayGuard",
        detail);
  }
  const bool allow_decoder_replay = false;
  if (
      allow_decoder_replay &&
      vision_graph.defined() &&
      runtime_policy.execution_program_plan.has_value() &&
      can_use_decoder_replay(main_input, skip_tensor)) {
    auto vision_replay = vision_graph.lookup_or_create_replay(
        vision_decoder_program_label(
            context->allocation_label(), context.get()),
        main_input.sizes(),
        skip_tensor.has_value()
            ? std::optional<std::vector<int64_t>>(skip_tensor->sizes().vec())
            : std::nullopt,
        target_sizes,
        vision_decoder_out_channels(context),
        *runtime_policy.execution_program_plan);
    if (vision_replay.defined()) {
      auto& replay_program = vision_replay.program();
      const VisionDecoderRunOutputs replay_outputs{
          replay_program.skip_relu_output(),
          replay_program.skip_conv1_output(),
          replay_program.skip_conv2_output(),
          replay_program.skip_res_output(),
          replay_program.main_input_output(),
          replay_program.main_relu_output(),
          replay_program.main_conv1_output(),
          replay_program.main_conv2_output(),
          replay_program.main_res_output(),
          replay_program.upsample_output(),
          replay_program.out_conv_output(),
      };
      utils::copy_buffer_tensor_direct_(
          vision_replay.input_slot(), main_input);
      if (skip_tensor.has_value() && skip_tensor->defined()) {
        TORCH_INTERNAL_ASSERT(
            vision_replay.skip_slot().has_value(),
            "Vision decoder replay expected a skip slot");
        utils::copy_buffer_tensor_direct_(
            *vision_replay.skip_slot(), *skip_tensor);
      }
      api::context()->flush_pending_cmds();

      if (!vision_replay.recorded()) {
        Tensor warmup_output = utils::create_buffer_tensor(
            vision_replay.output_slot().sizes(),
            vision_replay.output_slot().scalar_type(),
            /*persistent=*/false);
        utils::copy_buffer_tensor_direct_(
            warmup_output,
            run_vision_decoder_fusion_block_program(
                vision_replay.input_slot(),
                vision_replay.skip_slot(),
                target_sizes,
                context,
                replay_outputs));
        api::context()->flush_pending_cmds();
        vision_replay.replay().record([&]() {
          (void)run_vision_decoder_fusion_block_program(
              vision_replay.input_slot(),
              vision_replay.skip_slot(),
              target_sizes,
              context,
              replay_outputs);
        });
        utils::log_vulkan_op_hit(
            "vulkan_prepack::run_vision_decoder_fusion_block_context.replay_warmup");
        utils::log_vulkan_op_hit(
            "vulkan_prepack::run_vision_decoder_fusion_block_context");
        return maybe_restore_tensor(warmup_output, output_device, output_dtype);
      }

      const utils::ReplayEpoch epoch = utils::begin_replay_epoch(
          vision_replay.identity(), "vision.decoder_fusion.replay");
      const std::string detail =
          "action=run_decoder_fusion_direct "
          "reason=vision_replay_submit_guard "
          "failure_class=ReplayHangRisk";
      utils::log_replay_event(
          "vision_replay_submit_guard",
          vision_replay.identity(),
          epoch.run_id,
          "vision.decoder_fusion.replay",
          detail);
      api::report_vulkan_failure(
          api::VulkanFailureClass::ReplayHangRisk,
          "vulkan_prepack::run_vision_decoder_fusion_block_context",
          "VisionReplaySubmitGuard",
          detail);
      (void)run_vision_decoder_fusion_block_program(
          vision_replay.input_slot(),
          vision_replay.skip_slot(),
          target_sizes,
          context,
          replay_outputs);
      api::context()->flush_pending_cmds();
      Tensor output = utils::create_buffer_tensor(
          vision_replay.output_slot().sizes(),
          vision_replay.output_slot().scalar_type(),
          /*persistent=*/false);
      utils::copy_buffer_tensor_direct_(
          output, vision_replay.output_slot());
      utils::log_vulkan_op_hit(
          "vulkan_prepack::run_vision_decoder_fusion_block_context.replay");
      utils::log_vulkan_op_hit(
          "vulkan_prepack::run_vision_decoder_fusion_block_context");
      return maybe_restore_tensor(output, output_device, output_dtype);
    }
  }
  std::optional<utils::ScratchArena> graph_scratch = std::nullopt;
  if (vision_graph.defined() && runtime_policy.scratch_arena_plan.has_value()) {
    const uint32_t scratch_alignment = std::max<uint32_t>(
        runtime_policy.scratch_arena_plan->alignment,
        static_cast<uint32_t>(std::max<int64_t>(
            1, static_cast<int64_t>(c10::elementSize(kFloat)))));
    const size_t requested_bytes = vision_decoder_fusion_block_scratch_bytes(
        main_input, skip_tensor, target_sizes);
    if (
        requested_bytes > 0u &&
        runtime_policy.scratch_arena_plan->prefer_buffer_storage) {
      graph_scratch = vision_graph.ensure_shared_scratch(
          std::max(
              requested_bytes,
              runtime_policy.scratch_arena_plan->min_arena_bytes),
          scratch_alignment,
          runtime_policy.execution_program_plan.has_value() &&
              runtime_policy.execution_program_plan->persistent);
    }
  }

  auto vision_program = vision_graph.defined()
      ? vision_graph.lookup_or_create_program(
            vision_decoder_program_label(
                context->allocation_label(), context.get()),
            main_input.sizes(),
            skip_tensor.has_value()
                ? std::optional<std::vector<int64_t>>(skip_tensor->sizes().vec())
                : std::nullopt,
            target_sizes,
            vision_decoder_out_channels(context),
            !graph_scratch.has_value(),
            *runtime_policy.execution_program_plan)
      : prime_vision_decoder_program(
            main_input,
            skip_tensor,
            target_sizes,
            context,
            runtime_policy,
            graph_scratch.has_value(),
            !graph_scratch.has_value());
  auto& program_ref = vision_program;
  if (!program_ref.defined()) {
    return fallback(main_input, skip_tensor);
  }
  if (graph_scratch.has_value()) {
    graph_scratch->reset();
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_decoder_fusion_block_context.graph");
  } else if (program_ref.scratch_arena().has_value()) {
    program_ref.scratch_arena()->reset();
  }

  VisionDecoderRunOutputs outputs = graph_scratch.has_value()
      ? reserve_vision_decoder_graph_outputs(
            *graph_scratch,
            main_input,
            skip_tensor,
            target_sizes,
            program_ref.out_conv_output())
      : VisionDecoderRunOutputs{
            program_ref.skip_relu_output(),
            program_ref.skip_conv1_output(),
            program_ref.skip_conv2_output(),
            program_ref.skip_res_output(),
            program_ref.main_input_output(),
            program_ref.main_relu_output(),
            program_ref.main_conv1_output(),
            program_ref.main_conv2_output(),
            program_ref.main_res_output(),
            program_ref.upsample_output(),
            program_ref.out_conv_output(),
        };
  Tensor output = run_vision_decoder_fusion_block_program(
      main_input,
      skip_tensor,
      target_sizes,
      context,
      outputs);
  output = materialize_escaping_vulkan_output(
      output, output_device.type() == kVulkan);
  utils::log_vulkan_op_hit("vulkan_prepack::run_vision_decoder_fusion_block_context");
  return maybe_restore_tensor(output, output_device, output_dtype);
}

void prime_vision_decoder_fusion_block_context_graph(
    const Tensor& input_arg,
    const std::optional<Tensor>& skip_arg,
    const std::optional<std::vector<int64_t>>& size,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& context) {
  if (!input_arg.defined() || !input_arg.is_vulkan() || !context) {
    return;
  }

  TORCH_CHECK(
      input_arg.dim() == 4,
      "Vision decoder fusion block graph priming expects rank-4 input");
  if (input_arg.scalar_type() != kFloat) {
    return;
  }

  std::optional<Tensor> skip =
      (skip_arg.has_value() && skip_arg->defined()) ? skip_arg : std::nullopt;
  const std::vector<int64_t> target_sizes =
      resolve_decoder_target_sizes(input_arg, size);

  utils::VulkanPlanningRequestScope planning_scope(
      utils::make_vulkan_vision_decoder_request());
  const auto runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_vision_decoder_request());
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionDecoder) {
    return;
  }

  if (!size.has_value()) {
    return;
  }

  auto vision_graph =
      prime_vision_decoder_graph(input_arg, runtime_policy, context);
  if (!vision_graph.defined()) {
    return;
  }

  bool use_graph_scratch = false;
  if (runtime_policy.scratch_arena_plan.has_value()) {
    const uint32_t scratch_alignment = std::max<uint32_t>(
        runtime_policy.scratch_arena_plan->alignment,
        static_cast<uint32_t>(std::max<int64_t>(
            1, static_cast<int64_t>(c10::elementSize(kFloat)))));
    const size_t requested_bytes = vision_decoder_fusion_block_scratch_bytes(
        input_arg, skip, target_sizes);
    if (
        requested_bytes > 0u &&
        runtime_policy.scratch_arena_plan->prefer_buffer_storage) {
      vision_graph.note_shared_scratch_requirement(
          std::max(
              requested_bytes,
              runtime_policy.scratch_arena_plan->min_arena_bytes),
          scratch_alignment,
          runtime_policy.execution_program_plan->persistent);
      use_graph_scratch = true;
    }
  }

  (void)vision_graph.lookup_or_create_program(
      vision_decoder_program_label(
          context->allocation_label(), context.get()),
      input_arg.sizes(),
      skip.has_value() ? std::optional<std::vector<int64_t>>(skip->sizes().vec())
                       : std::nullopt,
      target_sizes,
      vision_decoder_out_channels(context),
      !use_graph_scratch,
      *runtime_policy.execution_program_plan);
  utils::log_vulkan_op_hit(
      "vulkan_prepack::prime_vision_decoder_fusion_block_context_graph");
}

VisionDecoderHeadContext::VisionDecoderHeadContext(
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet4_context,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet3_context,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet2_context,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet1_context,
    c10::intrusive_ptr<Conv2dPackedContext> output_conv1_context,
    c10::intrusive_ptr<Conv2dPackedContext> output_conv2_conv1_context,
    c10::intrusive_ptr<Conv2dPackedContext> output_conv2_conv2_context,
    const bool align_corners,
    std::string allocation_label)
    : allocation_label_(std::move(allocation_label)),
      align_corners_(align_corners),
      refinenet4_context_(std::move(refinenet4_context)),
      refinenet3_context_(std::move(refinenet3_context)),
      refinenet2_context_(std::move(refinenet2_context)),
      refinenet1_context_(std::move(refinenet1_context)),
      output_conv1_context_(std::move(output_conv1_context)),
      output_conv2_conv1_context_(std::move(output_conv2_conv1_context)),
      output_conv2_conv2_context_(std::move(output_conv2_conv2_context)) {
  TORCH_CHECK(
      refinenet4_context_ && refinenet3_context_ && refinenet2_context_ &&
          refinenet1_context_ && output_conv1_context_ &&
          output_conv2_conv1_context_ && output_conv2_conv2_context_,
      "Vision decoder head context requires all sub-contexts to be defined");

  unpacked_.reserve(Unpacked::NumArgs);
  unpacked_.emplace_back(refinenet4_context_);
  unpacked_.emplace_back(refinenet3_context_);
  unpacked_.emplace_back(refinenet2_context_);
  unpacked_.emplace_back(refinenet1_context_);
  unpacked_.emplace_back(output_conv1_context_);
  unpacked_.emplace_back(output_conv2_conv1_context_);
  unpacked_.emplace_back(output_conv2_conv2_context_);
  unpacked_.emplace_back(align_corners_);
  unpacked_.emplace_back(allocation_label_);
}

VisionDecoderHeadContext VisionDecoderHeadContext::pack(
    c10::impl::GenericList unpacked) {
  return VisionDecoderHeadContext(
      unpacked.get(Unpacked::Refinenet4Context)
          .toCustomClass<VisionDecoderFusionBlockContext>(),
      unpacked.get(Unpacked::Refinenet3Context)
          .toCustomClass<VisionDecoderFusionBlockContext>(),
      unpacked.get(Unpacked::Refinenet2Context)
          .toCustomClass<VisionDecoderFusionBlockContext>(),
      unpacked.get(Unpacked::Refinenet1Context)
          .toCustomClass<VisionDecoderFusionBlockContext>(),
      unpacked.get(Unpacked::OutputConv1Context)
          .toCustomClass<Conv2dPackedContext>(),
      unpacked.get(Unpacked::OutputConv2Conv1Context)
          .toCustomClass<Conv2dPackedContext>(),
      unpacked.get(Unpacked::OutputConv2Conv2Context)
          .toCustomClass<Conv2dPackedContext>(),
      unpacked.get(Unpacked::AlignCorners).toBool(),
      unpacked.get(Unpacked::Label).toStringRef());
}

c10::intrusive_ptr<VisionDecoderHeadContext> create_vision_decoder_head_context(
    const Tensor& prototype,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet4_context,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet3_context,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet2_context,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet1_context,
    c10::intrusive_ptr<Conv2dPackedContext> output_conv1_context,
    c10::intrusive_ptr<Conv2dPackedContext> output_conv2_conv1_context,
    c10::intrusive_ptr<Conv2dPackedContext> output_conv2_conv2_context,
    const bool align_corners,
    std::string label) {
  (void)prototype;
  return c10::make_intrusive<VisionDecoderHeadContext>(
      std::move(refinenet4_context),
      std::move(refinenet3_context),
      std::move(refinenet2_context),
      std::move(refinenet1_context),
      std::move(output_conv1_context),
      std::move(output_conv2_conv1_context),
      std::move(output_conv2_conv2_context),
      align_corners,
      std::move(label));
}

VisionDecoderPreprocessHeadContext::VisionDecoderPreprocessHeadContext(
    c10::intrusive_ptr<Conv2dPackedContext> project1_context,
    c10::intrusive_ptr<Conv2dPackedContext> project2_context,
    c10::intrusive_ptr<Conv2dPackedContext> project3_context,
    c10::intrusive_ptr<Conv2dPackedContext> project4_context,
    c10::intrusive_ptr<Conv2dPackedContext> resize1_context,
    c10::intrusive_ptr<Conv2dPackedContext> resize2_context,
    c10::intrusive_ptr<Conv2dPackedContext> resize4_context,
    c10::intrusive_ptr<Conv2dPackedContext> layer1_rn_context,
    c10::intrusive_ptr<Conv2dPackedContext> layer2_rn_context,
    c10::intrusive_ptr<Conv2dPackedContext> layer3_rn_context,
    c10::intrusive_ptr<Conv2dPackedContext> layer4_rn_context,
    c10::intrusive_ptr<VisionDecoderHeadContext> head_context,
    std::string allocation_label)
    : allocation_label_(std::move(allocation_label)),
      project1_context_(std::move(project1_context)),
      project2_context_(std::move(project2_context)),
      project3_context_(std::move(project3_context)),
      project4_context_(std::move(project4_context)),
      resize1_context_(std::move(resize1_context)),
      resize2_context_(std::move(resize2_context)),
      resize4_context_(std::move(resize4_context)),
      layer1_rn_context_(std::move(layer1_rn_context)),
      layer2_rn_context_(std::move(layer2_rn_context)),
      layer3_rn_context_(std::move(layer3_rn_context)),
      layer4_rn_context_(std::move(layer4_rn_context)),
      head_context_(std::move(head_context)) {
  TORCH_CHECK(
      project1_context_ && project2_context_ && project3_context_ &&
          project4_context_ && resize1_context_ && resize2_context_ &&
          resize4_context_ && layer1_rn_context_ && layer2_rn_context_ &&
          layer3_rn_context_ && layer4_rn_context_ && head_context_,
      "Vision decoder preprocess head context requires all sub-contexts to be "
      "defined");

  unpacked_.reserve(Unpacked::NumArgs);
  unpacked_.emplace_back(project1_context_);
  unpacked_.emplace_back(project2_context_);
  unpacked_.emplace_back(project3_context_);
  unpacked_.emplace_back(project4_context_);
  unpacked_.emplace_back(resize1_context_);
  unpacked_.emplace_back(resize2_context_);
  unpacked_.emplace_back(resize4_context_);
  unpacked_.emplace_back(layer1_rn_context_);
  unpacked_.emplace_back(layer2_rn_context_);
  unpacked_.emplace_back(layer3_rn_context_);
  unpacked_.emplace_back(layer4_rn_context_);
  unpacked_.emplace_back(head_context_);
  unpacked_.emplace_back(allocation_label_);
}

VisionDecoderPreprocessHeadContext VisionDecoderPreprocessHeadContext::pack(
    c10::impl::GenericList unpacked) {
  return VisionDecoderPreprocessHeadContext(
      unpacked.get(Unpacked::Project1Context)
          .toCustomClass<Conv2dPackedContext>(),
      unpacked.get(Unpacked::Project2Context)
          .toCustomClass<Conv2dPackedContext>(),
      unpacked.get(Unpacked::Project3Context)
          .toCustomClass<Conv2dPackedContext>(),
      unpacked.get(Unpacked::Project4Context)
          .toCustomClass<Conv2dPackedContext>(),
      unpacked.get(Unpacked::Resize1Context)
          .toCustomClass<Conv2dPackedContext>(),
      unpacked.get(Unpacked::Resize2Context)
          .toCustomClass<Conv2dPackedContext>(),
      unpacked.get(Unpacked::Resize4Context)
          .toCustomClass<Conv2dPackedContext>(),
      unpacked.get(Unpacked::Layer1RnContext)
          .toCustomClass<Conv2dPackedContext>(),
      unpacked.get(Unpacked::Layer2RnContext)
          .toCustomClass<Conv2dPackedContext>(),
      unpacked.get(Unpacked::Layer3RnContext)
          .toCustomClass<Conv2dPackedContext>(),
      unpacked.get(Unpacked::Layer4RnContext)
          .toCustomClass<Conv2dPackedContext>(),
      unpacked.get(Unpacked::HeadContext)
          .toCustomClass<VisionDecoderHeadContext>(),
      unpacked.get(Unpacked::Label).toStringRef());
}

c10::intrusive_ptr<VisionDecoderPreprocessHeadContext>
create_vision_decoder_preprocess_head_context(
    const Tensor& prototype,
    c10::intrusive_ptr<Conv2dPackedContext> project1_context,
    c10::intrusive_ptr<Conv2dPackedContext> project2_context,
    c10::intrusive_ptr<Conv2dPackedContext> project3_context,
    c10::intrusive_ptr<Conv2dPackedContext> project4_context,
    c10::intrusive_ptr<Conv2dPackedContext> resize1_context,
    c10::intrusive_ptr<Conv2dPackedContext> resize2_context,
    c10::intrusive_ptr<Conv2dPackedContext> resize4_context,
    c10::intrusive_ptr<Conv2dPackedContext> layer1_rn_context,
    c10::intrusive_ptr<Conv2dPackedContext> layer2_rn_context,
    c10::intrusive_ptr<Conv2dPackedContext> layer3_rn_context,
    c10::intrusive_ptr<Conv2dPackedContext> layer4_rn_context,
    c10::intrusive_ptr<VisionDecoderHeadContext> head_context,
    std::string label) {
  (void)prototype;
  return c10::make_intrusive<VisionDecoderPreprocessHeadContext>(
      std::move(project1_context),
      std::move(project2_context),
      std::move(project3_context),
      std::move(project4_context),
      std::move(resize1_context),
      std::move(resize2_context),
      std::move(resize4_context),
      std::move(layer1_rn_context),
      std::move(layer2_rn_context),
      std::move(layer3_rn_context),
      std::move(layer4_rn_context),
      std::move(head_context),
      std::move(label));
}

Tensor run_vision_decoder_preprocess_head_context(
    const Tensor& layer1_tokens_arg,
    const Tensor& layer2_tokens_arg,
    const Tensor& layer3_tokens_arg,
    const Tensor& layer4_tokens_arg,
    const int64_t patch_h,
    const int64_t patch_w,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderPreprocessHeadContext>& context) {
  TORCH_CHECK(
      context,
      "Vision decoder preprocess head context must be defined");
  TORCH_CHECK(
      patch_h > 0 && patch_w > 0,
      "Vision decoder preprocess head context expects positive patch sizes");
  TORCH_CHECK(
      output_size.size() == 2,
      "Vision decoder preprocess head context expects a rank-1 output size "
      "with 2 entries");
  TORCH_CHECK(
      (layer1_tokens_arg.dim() == 2 || layer1_tokens_arg.dim() == 3) &&
          (layer2_tokens_arg.dim() == 2 || layer2_tokens_arg.dim() == 3) &&
          (layer3_tokens_arg.dim() == 2 || layer3_tokens_arg.dim() == 3) &&
          (layer4_tokens_arg.dim() == 2 || layer4_tokens_arg.dim() == 3),
      "Vision decoder preprocess head context expects rank-2 or rank-3 token "
      "inputs");
  utils::validate_replay_tensor_not_stale(
      layer1_tokens_arg,
      "vulkan_prepack::run_vision_decoder_preprocess_head_context.layer1");
  utils::validate_replay_tensor_not_stale(
      layer2_tokens_arg,
      "vulkan_prepack::run_vision_decoder_preprocess_head_context.layer2");
  utils::validate_replay_tensor_not_stale(
      layer3_tokens_arg,
      "vulkan_prepack::run_vision_decoder_preprocess_head_context.layer3");
  utils::validate_replay_tensor_not_stale(
      layer4_tokens_arg,
      "vulkan_prepack::run_vision_decoder_preprocess_head_context.layer4");

  const Device output_device = layer1_tokens_arg.device();
  const ScalarType output_dtype = layer1_tokens_arg.scalar_type();

  const auto fallback = [&]() -> Tensor {
    Tensor layer1 = tokens_to_feature_map(layer1_tokens_arg, patch_h, patch_w);
    layer1 = run_conv2d_context(layer1, context->project1_context());
    layer1 = run_tconv2d_context(layer1, context->resize1_context());
    layer1 = run_conv2d_context(layer1, context->layer1_rn_context());

    Tensor layer2 = tokens_to_feature_map(layer2_tokens_arg, patch_h, patch_w);
    layer2 = run_conv2d_context(layer2, context->project2_context());
    layer2 = run_tconv2d_context(layer2, context->resize2_context());
    layer2 = run_conv2d_context(layer2, context->layer2_rn_context());

    Tensor layer3 = tokens_to_feature_map(layer3_tokens_arg, patch_h, patch_w);
    layer3 = run_conv2d_context(layer3, context->project3_context());
    layer3 = run_conv2d_context(layer3, context->layer3_rn_context());

    Tensor layer4 = tokens_to_feature_map(layer4_tokens_arg, patch_h, patch_w);
    layer4 = run_conv2d_context(layer4, context->project4_context());
    layer4 = run_conv2d_context(layer4, context->resize4_context());
    layer4 = run_conv2d_context(layer4, context->layer4_rn_context());

    Tensor output = run_vision_decoder_head_context(
        layer1,
        layer2,
        layer3,
        layer4,
        output_size,
        context->head_context());
    Tensor restored = maybe_restore_tensor(output, output_device, output_dtype);
    record_tensor_write(
        restored,
        "vulkan_prepack::run_vision_decoder_preprocess_head_context",
        "fallback",
        {layer1_tokens_arg, layer2_tokens_arg, layer3_tokens_arg, layer4_tokens_arg});
    return restored;
  };

  {
    const std::string detail =
        "action=use_decoder_preprocess_explicit_path "
        "reason=decoder_preprocess_program_guard "
        "failure_class=KernelIncorrect";
    api::report_vulkan_failure(
        api::VulkanFailureClass::KernelIncorrect,
        "vulkan_prepack::run_vision_decoder_preprocess_head_context",
        "DecoderPreprocessProgramGuard",
        detail);
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_decoder_preprocess_head_context.guard.program_disabled");
    return fallback();
  }

  Tensor layer1_tokens = layer1_tokens_arg.is_vulkan() ? layer1_tokens_arg
                                                       : layer1_tokens_arg.vulkan();
  Tensor layer2_tokens = layer2_tokens_arg.is_vulkan() ? layer2_tokens_arg
                                                       : layer2_tokens_arg.vulkan();
  Tensor layer3_tokens = layer3_tokens_arg.is_vulkan() ? layer3_tokens_arg
                                                       : layer3_tokens_arg.vulkan();
  Tensor layer4_tokens = layer4_tokens_arg.is_vulkan() ? layer4_tokens_arg
                                                       : layer4_tokens_arg.vulkan();
  if (
      layer1_tokens.scalar_type() != kFloat ||
      layer2_tokens.scalar_type() != kFloat ||
      layer3_tokens.scalar_type() != kFloat ||
      layer4_tokens.scalar_type() != kFloat) {
    return fallback();
  }
  const std::array<Tensor, 4u> compiled_layer_tokens{{
      prepare_buffer_attention_tensor(layer1_tokens),
      prepare_buffer_attention_tensor(layer2_tokens),
      prepare_buffer_attention_tensor(layer3_tokens),
      prepare_buffer_attention_tensor(layer4_tokens),
  }};

  const std::array<std::vector<int64_t>, 4u> layer_token_sizes{{
      compiled_layer_tokens[0].sizes().vec(),
      compiled_layer_tokens[1].sizes().vec(),
      compiled_layer_tokens[2].sizes().vec(),
      compiled_layer_tokens[3].sizes().vec(),
  }};
  const std::array<std::vector<int64_t>, 4u> layer_feature_sizes{{
      tokens_to_feature_map_output_sizes(compiled_layer_tokens[0].sizes(), patch_h, patch_w),
      tokens_to_feature_map_output_sizes(compiled_layer_tokens[1].sizes(), patch_h, patch_w),
      tokens_to_feature_map_output_sizes(compiled_layer_tokens[2].sizes(), patch_h, patch_w),
      tokens_to_feature_map_output_sizes(compiled_layer_tokens[3].sizes(), patch_h, patch_w),
  }};
  const std::array<std::vector<int64_t>, 4u> decoder_layer_sizes{{
      decoder_preprocess_layer_output_sizes(
          layer_feature_sizes[0],
          context->project1_context(),
          context->resize1_context(),
          /*apply_resize=*/true,
          context->layer1_rn_context()),
      decoder_preprocess_layer_output_sizes(
          layer_feature_sizes[1],
          context->project2_context(),
          context->resize2_context(),
          /*apply_resize=*/true,
          context->layer2_rn_context()),
      decoder_preprocess_layer_output_sizes(
          layer_feature_sizes[2],
          context->project3_context(),
          c10::intrusive_ptr<Conv2dPackedContext>{},
          /*apply_resize=*/false,
          context->layer3_rn_context()),
      decoder_preprocess_layer_output_sizes(
          layer_feature_sizes[3],
          context->project4_context(),
          context->resize4_context(),
          /*apply_resize=*/true,
          context->layer4_rn_context()),
  }};
  const std::array<std::vector<int64_t>, 4u> project_layer_sizes{{
      conv2d_context_output_sizes(layer_feature_sizes[0], context->project1_context()),
      conv2d_context_output_sizes(layer_feature_sizes[1], context->project2_context()),
      conv2d_context_output_sizes(layer_feature_sizes[2], context->project3_context()),
      conv2d_context_output_sizes(layer_feature_sizes[3], context->project4_context()),
  }};
  const std::array<std::vector<int64_t>, 4u> resize_layer_sizes{{
      conv2d_context_output_sizes(project_layer_sizes[0], context->resize1_context()),
      conv2d_context_output_sizes(project_layer_sizes[1], context->resize2_context()),
      project_layer_sizes[2],
      conv2d_context_output_sizes(project_layer_sizes[3], context->resize4_context()),
  }};
  const std::array<bool, 4u> apply_resize{{true, true, false, true}};
  const std::vector<int64_t> final_output_sizes{
      layer_feature_sizes[0][0],
      kDaV2HeadFinalChannels,
      output_size[0],
      output_size[1],
  };
  auto decoder_request = utils::make_vulkan_vision_decoder_request();
  decoder_request.fixed_shape_graph_input_sizes = decoder_layer_sizes[3];
  utils::VulkanPlanningRequestScope decoder_planning_scope(decoder_request);
  const auto runtime_policy =
      utils::build_vulkan_runtime_policy(decoder_request);
  const auto compiled_session =
      utils::lookup_or_create_vision_transformer_depth_decoder_session(
          utils::VisionTransformerDepthDecoderSessionDesc{
              current_graph_capture_label("depth.vision", "depth.vision.graph") +
                  ".decoder_preprocess_head.ctx." +
                  context_identity_key(context.get()),
              layer_token_sizes,
              layer_feature_sizes,
              project_layer_sizes,
              resize_layer_sizes,
              apply_resize,
              decoder_layer_sizes,
              final_output_sizes,
              layer1_tokens.scalar_type(),
              patch_h,
              patch_w,
              /*persistent=*/true});
  const VisionReplayBundleIdentity bundle_identity =
      make_vision_decoder_preprocess_head_bundle_identity(
          compiled_layer_tokens,
          patch_h,
          patch_w,
          output_size,
          context.get());
  const std::string root_label =
      current_graph_capture_label("depth.vision", "depth.vision.graph");
  if (auto compiled_output =
          try_run_vision_decoder_preprocess_head_compiled_session(
              compiled_layer_tokens,
              output_device,
              output_dtype,
              patch_h,
              patch_w,
              output_size,
              context,
              runtime_policy,
              bundle_identity,
              root_label,
              compiled_session)) {
    return *compiled_output;
  }

  const auto run_layer =
      [&](const Tensor& tokens,
          const c10::intrusive_ptr<Conv2dPackedContext>& project_context,
          const c10::intrusive_ptr<Conv2dPackedContext>& resize_context,
          const bool apply_resize,
          const c10::intrusive_ptr<Conv2dPackedContext>& rn_context) -> Tensor {
    Tensor feature_map = tokens_to_feature_map(tokens, patch_h, patch_w);
    if (!feature_map.defined() || !feature_map.is_vulkan()) {
      return Tensor();
    }

    Tensor feature_buffer = prepare_decoder_buffer_tensor(feature_map);
    if (!feature_buffer.defined() || feature_buffer.dim() != 4) {
      return Tensor();
    }

    Tensor project_output = utils::create_buffer_tensor(
        conv2d_context_output_sizes(feature_buffer, project_context),
        feature_buffer.scalar_type(),
        /*persistent=*/false);
    (void)run_conv2d_context_out(feature_buffer, project_context, project_output);

    Tensor resized = project_output;
    Tensor resize_output;
    if (apply_resize) {
      resize_output = utils::create_buffer_tensor(
          conv2d_context_output_sizes(project_output, resize_context),
          project_output.scalar_type(),
          /*persistent=*/false);
      (void)run_conv2d_context_any_out(
          project_output,
          resize_context,
          resize_output);
      resized = resize_output;
    }

    Tensor rn_output = utils::create_buffer_tensor(
        conv2d_context_output_sizes(resized, rn_context),
        resized.scalar_type(),
        /*persistent=*/false);
    (void)run_conv2d_context_out(resized, rn_context, rn_output);
    return rn_output;
  };

  Tensor layer1 = run_layer(
      layer1_tokens,
      context->project1_context(),
      context->resize1_context(),
      /*apply_resize=*/true,
      context->layer1_rn_context());
  Tensor layer2 = run_layer(
      layer2_tokens,
      context->project2_context(),
      context->resize2_context(),
      /*apply_resize=*/true,
      context->layer2_rn_context());
  Tensor layer3 = run_layer(
      layer3_tokens,
      context->project3_context(),
      c10::intrusive_ptr<Conv2dPackedContext>{},
      /*apply_resize=*/false,
      context->layer3_rn_context());
  Tensor layer4 = run_layer(
      layer4_tokens,
      context->project4_context(),
      context->resize4_context(),
      /*apply_resize=*/true,
      context->layer4_rn_context());
  if (
      !layer1.defined() || !layer2.defined() || !layer3.defined() ||
      !layer4.defined()) {
    return fallback();
  }

  Tensor output = run_vision_decoder_head_context(
      layer1,
      layer2,
      layer3,
      layer4,
      output_size,
      context->head_context());
  output = materialize_escaping_vulkan_output(
      output, output_device.type() == kVulkan);
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_vision_decoder_preprocess_head_context");
  return maybe_restore_tensor(output, output_device, output_dtype);
}

Tensor run_vision_decoder_head_context(
    const Tensor& layer1_arg,
    const Tensor& layer2_arg,
    const Tensor& layer3_arg,
    const Tensor& layer4_arg,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderHeadContext>& context) {
  TORCH_CHECK(context, "Vision decoder head context must be defined");
  TORCH_CHECK(
      layer1_arg.dim() == 4 && layer2_arg.dim() == 4 && layer3_arg.dim() == 4 &&
          layer4_arg.dim() == 4,
      "Vision decoder head context expects rank-4 layer inputs");
  TORCH_CHECK(
      output_size.size() == 2,
      "Vision decoder head context expects a rank-1 output size with 2 entries");

  const Device output_device = layer1_arg.device();
  const ScalarType output_dtype = layer1_arg.scalar_type();

  Tensor layer1 = layer1_arg.is_vulkan() ? layer1_arg : layer1_arg.vulkan();
  Tensor layer2 = layer2_arg.is_vulkan() ? layer2_arg : layer2_arg.vulkan();
  Tensor layer3 = layer3_arg.is_vulkan() ? layer3_arg : layer3_arg.vulkan();
  Tensor layer4 = layer4_arg.is_vulkan() ? layer4_arg : layer4_arg.vulkan();
  if (layer1.scalar_type() != kFloat) {
    layer1 = layer1.to(kFloat);
  }
  if (layer2.scalar_type() != kFloat) {
    layer2 = layer2.to(kFloat);
  }
  if (layer3.scalar_type() != kFloat) {
    layer3 = layer3.to(kFloat);
  }
  if (layer4.scalar_type() != kFloat) {
    layer4 = layer4.to(kFloat);
  }

  const auto fallback = [&]() -> Tensor {
    Tensor path4 = run_vision_decoder_fusion_block_context(
        layer4,
        std::nullopt,
        std::optional<std::vector<int64_t>>({layer3.size(2), layer3.size(3)}),
        context->refinenet4_context());
    Tensor path3 = run_vision_decoder_fusion_block_context(
        path4,
        layer3,
        std::optional<std::vector<int64_t>>({layer2.size(2), layer2.size(3)}),
        context->refinenet3_context());
    Tensor path2 = run_vision_decoder_fusion_block_context(
        path3,
        layer2,
        std::optional<std::vector<int64_t>>({layer1.size(2), layer1.size(3)}),
        context->refinenet2_context());
    Tensor path1 = run_vision_decoder_fusion_block_context(
        path2, layer1, std::nullopt, context->refinenet1_context());
    Tensor output = run_vision_decoder_head_tail_context(
        prepare_decoder_buffer_tensor(path1), output_size, context);
    return maybe_restore_tensor(output, output_device, output_dtype);
  };

  Tensor layer1_buffer = prepare_decoder_buffer_tensor(layer1);
  Tensor layer2_buffer = prepare_decoder_buffer_tensor(layer2);
  Tensor layer3_buffer = prepare_decoder_buffer_tensor(layer3);
  Tensor layer4_buffer = prepare_decoder_buffer_tensor(layer4);
  if (
      layer1_buffer.dim() != 4 || layer2_buffer.dim() != 4 ||
      layer3_buffer.dim() != 4 || layer4_buffer.dim() != 4) {
    return fallback();
  }

  const std::vector<int64_t> path1_sizes{
      layer1_buffer.size(0),
      layer1_buffer.size(1),
      layer1_buffer.size(2) * 2,
      layer1_buffer.size(3) * 2,
  };
  if (!can_run_depth_anything_v2_head_fusion_shape(
          path1_sizes, output_size, context)) {
    return fallback();
  }

  if (!can_use_decoder_head_replay(
          layer1_buffer, layer2_buffer, layer3_buffer, layer4_buffer)) {
    return fallback();
  }

  {
    const std::string detail =
        "action=use_decoder_head_program_path "
        "reason=decoder_head_replay_guard "
        "failure_class=ReplayHangRisk";
    utils::log_replay_event(
        "decoder_head_replay_guard",
        nullptr,
        0u,
        "vision.decoder_head.replay",
        detail);
    api::report_vulkan_failure(
        api::VulkanFailureClass::ReplayHangRisk,
        "vulkan_prepack::run_vision_decoder_head_context",
        "DecoderHeadReplayGuard",
        detail);
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_decoder_head_context.guard.replay_disabled");
    return fallback();
  }

  auto vision_replay = maybe_lookup_vision_decoder_head_replay(
      layer1_buffer.sizes(),
      layer2_buffer.sizes(),
      layer3_buffer.sizes(),
      layer4_buffer.sizes(),
      output_size,
      context);
  if (!vision_replay.has_value()) {
    return fallback();
  }

  copy_tensor_for_replay(vision_replay->layer1_slot(), layer1_buffer);
  copy_tensor_for_replay(vision_replay->layer2_slot(), layer2_buffer);
  copy_tensor_for_replay(vision_replay->layer3_slot(), layer3_buffer);
  copy_tensor_for_replay(vision_replay->layer4_slot(), layer4_buffer);
  api::context()->flush_pending_cmds();

  if (!vision_replay->recorded()) {
    Tensor warmup_output = utils::create_buffer_tensor(
        vision_replay->output_slot().sizes(),
        vision_replay->output_slot().scalar_type(),
        /*persistent=*/false);
    copy_tensor_for_replay(
        warmup_output,
        run_vision_decoder_head_program(
            vision_replay->layer1_slot(),
            vision_replay->layer2_slot(),
            vision_replay->layer3_slot(),
            vision_replay->layer4_slot(),
            output_size,
            context,
            vision_replay->refinenet4_program(),
            vision_replay->refinenet3_program(),
            vision_replay->refinenet2_program(),
            vision_replay->refinenet1_program(),
            vision_replay->output_slot()));
    api::context()->flush_pending_cmds();
    vision_replay->replay().record([&]() {
      (void)run_vision_decoder_head_program(
          vision_replay->layer1_slot(),
          vision_replay->layer2_slot(),
          vision_replay->layer3_slot(),
          vision_replay->layer4_slot(),
          output_size,
          context,
          vision_replay->refinenet4_program(),
          vision_replay->refinenet3_program(),
          vision_replay->refinenet2_program(),
          vision_replay->refinenet1_program(),
          vision_replay->output_slot());
    });
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_decoder_head_context.replay_warmup");
    utils::log_vulkan_op_hit("vulkan_prepack::run_vision_decoder_head_context");
    return maybe_restore_tensor(warmup_output, output_device, output_dtype);
  }

  const utils::ReplayEpoch epoch = utils::begin_replay_epoch(
      vision_replay->identity(), "vision.decoder_head.replay");
  const std::string detail =
      "action=run_decoder_head_direct reason=decoder_head_replay_submit_guard "
      "failure_class=ReplayHangRisk";
  utils::log_replay_event(
      "decoder_head_replay_submit_guard",
      vision_replay->identity(),
      epoch.run_id,
      "vision.decoder_head.replay",
      detail);
  api::report_vulkan_failure(
      api::VulkanFailureClass::ReplayHangRisk,
      "vulkan_prepack::run_vision_decoder_head_context",
      "DecoderHeadReplaySubmitGuard",
      detail);
  (void)run_vision_decoder_head_program(
      vision_replay->layer1_slot(),
      vision_replay->layer2_slot(),
      vision_replay->layer3_slot(),
      vision_replay->layer4_slot(),
      output_size,
      context,
      vision_replay->refinenet4_program(),
      vision_replay->refinenet3_program(),
      vision_replay->refinenet2_program(),
      vision_replay->refinenet1_program(),
      vision_replay->output_slot());
  api::context()->flush_pending_cmds();
  Tensor output = utils::create_buffer_tensor(
      vision_replay->output_slot().sizes(),
      vision_replay->output_slot().scalar_type(),
      /*persistent=*/false);
  copy_tensor_for_replay(output, vision_replay->output_slot());
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_vision_decoder_head_context.replay");
  utils::log_vulkan_op_hit("vulkan_prepack::run_vision_decoder_head_context");
  return maybe_restore_tensor(output, output_device, output_dtype);
}

void prime_vision_decoder_head_context_graph(
    const Tensor& layer1,
    const Tensor& layer2,
    const Tensor& layer3,
    const Tensor& layer4,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderHeadContext>& context) {
  if (
      !context || !layer1.defined() || !layer1.is_vulkan() ||
      !layer2.defined() || !layer2.is_vulkan() || !layer3.defined() ||
      !layer3.is_vulkan() || !layer4.defined() || !layer4.is_vulkan()) {
    return;
  }

  TORCH_CHECK(
      layer1.dim() == 4 && layer2.dim() == 4 && layer3.dim() == 4 &&
          layer4.dim() == 4,
      "Vision decoder head graph priming expects rank-4 layer inputs");
  TORCH_CHECK(
      output_size.size() == 2,
      "Vision decoder head graph priming expects a rank-1 output size with 2 entries");
  if (
      layer1.scalar_type() != kFloat || layer2.scalar_type() != kFloat ||
      layer3.scalar_type() != kFloat || layer4.scalar_type() != kFloat) {
    return;
  }
  const std::vector<int64_t> path1_sizes{
      layer1.size(0),
      layer1.size(1),
      layer1.size(2) * 2,
      layer1.size(3) * 2,
  };
  if (!can_run_depth_anything_v2_head_fusion_shape(
          path1_sizes, output_size, context)) {
    return;
  }

  utils::VulkanPlanningRequestScope planning_scope(
      utils::make_vulkan_vision_decoder_request());
  const auto runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_vision_decoder_request());
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionDecoder ||
      !has_explicit_runtime_capture_label()) {
    return;
  }

  const int64_t output_conv1_channels =
      context->output_conv1_context()
          ->unpack()
          .get(Conv2dPackedContext::Unpacked::Weight)
          .toTensor()
          .size(0);
  const int64_t output_conv2_channels =
      context->output_conv2_conv1_context()
          ->unpack()
          .get(Conv2dPackedContext::Unpacked::Weight)
          .toTensor()
          .size(0);
  const int64_t final_channels =
      context->output_conv2_conv2_context()
          ->unpack()
          .get(Conv2dPackedContext::Unpacked::Weight)
          .toTensor()
          .size(0);
  const std::vector<int64_t> output_sizes{
      layer1.size(0), final_channels, output_size[0], output_size[1]};

  auto vision_graph = utils::lookup_or_create_labeled_vision_decoder_inference_graph(
      vision_decoder_graph_label(context->allocation_label()),
      kFloat,
      runtime_policy.execution_program_plan->persistent);
  (void)vision_graph.lookup_or_create_head_replay(
      vision_decoder_head_program_label(
          context->allocation_label(), context.get()),
      layer1.sizes(),
      layer2.sizes(),
      layer3.sizes(),
      layer4.sizes(),
      output_sizes,
      output_conv1_channels,
      output_conv2_channels,
      final_channels,
      *runtime_policy.execution_program_plan);
  utils::log_vulkan_op_hit(
      "vulkan_prepack::prime_vision_decoder_head_context_graph");
}

std::tuple<Tensor, Tensor> run_vision_backbone_decoder_replay_bundle_bridge(
    const Tensor& backbone_input_arg,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& backbone_context,
    const Tensor& decoder_input_arg,
    const std::optional<Tensor>& decoder_skip_arg,
    const std::optional<std::vector<int64_t>>& decoder_size,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& decoder_context) {
  TORCH_CHECK(
      backbone_context && decoder_context,
      "Vision backbone/decoder replay bundle bridge expects defined contexts");

  const Device backbone_output_device = backbone_input_arg.device();
  const ScalarType backbone_output_dtype = backbone_input_arg.scalar_type();
  const Device decoder_output_device = decoder_input_arg.device();
  const ScalarType decoder_output_dtype = decoder_input_arg.scalar_type();

  Tensor backbone_input =
      backbone_input_arg.is_vulkan() ? backbone_input_arg : backbone_input_arg.vulkan();
  Tensor decoder_input =
      decoder_input_arg.is_vulkan() ? decoder_input_arg : decoder_input_arg.vulkan();
  std::optional<Tensor> decoder_skip =
      (decoder_skip_arg.has_value() && decoder_skip_arg->defined())
      ? std::optional<Tensor>(
            decoder_skip_arg->is_vulkan() ? *decoder_skip_arg : decoder_skip_arg->vulkan())
      : std::nullopt;

  TORCH_CHECK(
      backbone_input.dim() == 2 || backbone_input.dim() == 3,
      "Vision backbone/decoder replay bundle bridge expects rank-2 or rank-3 "
      "backbone input");
  TORCH_CHECK(
      decoder_input.dim() == 4,
      "Vision backbone/decoder replay bundle bridge expects rank-4 decoder input");
  TORCH_CHECK(
      backbone_input.scalar_type() == kFloat &&
          decoder_input.scalar_type() == kFloat &&
          (!decoder_skip.has_value() || decoder_skip->scalar_type() == kFloat),
      "Vision backbone/decoder replay bundle bridge currently expects float inputs");

  auto backbone_request = utils::make_vulkan_vision_backbone_request();
  backbone_request.fixed_shape_graph_input_sizes = backbone_input.sizes().vec();
  backbone_request.prefer_packed_layout_propagation = true;
  utils::VulkanPlanningRequestScope backbone_scope(backbone_request);
  const auto backbone_runtime_policy =
      utils::build_vulkan_runtime_policy(backbone_request);
  if (
      !backbone_runtime_policy.execution_program_plan.has_value() ||
      backbone_runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionBackbone) {
    return std::make_tuple(
        run_vision_backbone_block_context(backbone_input_arg, backbone_context),
        run_vision_decoder_fusion_block_context(
            decoder_input_arg, decoder_skip_arg, decoder_size, decoder_context));
  }

  auto decoder_request = utils::make_vulkan_vision_decoder_request();
  decoder_request.fixed_shape_graph_input_sizes = decoder_input.sizes().vec();
  utils::VulkanPlanningRequestScope decoder_scope(decoder_request);
  const auto decoder_runtime_policy =
      utils::build_vulkan_runtime_policy(decoder_request);
  if (
      !decoder_runtime_policy.execution_program_plan.has_value() ||
      decoder_runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionDecoder) {
    return std::make_tuple(
        run_vision_backbone_block_context(backbone_input_arg, backbone_context),
        run_vision_decoder_fusion_block_context(
            decoder_input_arg, decoder_skip_arg, decoder_size, decoder_context));
  }

  auto backbone_graph =
      prime_vision_backbone_graph(backbone_input, backbone_runtime_policy, backbone_context);
  Tensor decoder_input_buffer = prepare_decoder_buffer_tensor(decoder_input);
  std::optional<Tensor> decoder_skip_buffer =
      (decoder_skip.has_value() && decoder_skip->defined())
      ? std::optional<Tensor>(prepare_decoder_buffer_tensor(*decoder_skip))
      : std::nullopt;
  auto decoder_graph =
      prime_vision_decoder_graph(decoder_input_buffer, decoder_runtime_policy, decoder_context);
  if (!backbone_graph.defined() || !decoder_graph.defined()) {
    return std::make_tuple(
        run_vision_backbone_block_context(backbone_input_arg, backbone_context),
        run_vision_decoder_fusion_block_context(
            decoder_input_arg, decoder_skip_arg, decoder_size, decoder_context));
  }

  std::optional<utils::ScratchArena> backbone_graph_scratch = std::nullopt;
  if (backbone_runtime_policy.scratch_arena_plan.has_value()) {
    const int64_t batch_size =
        backbone_input.dim() == 2 ? 1 : backbone_input.size(0);
    const int64_t token_count =
        backbone_input.dim() == 2 ? backbone_input.size(0) : backbone_input.size(1);
    const int64_t embed_dim = backbone_input.size(-1);
    const uint32_t scratch_alignment = std::max<uint32_t>(
        backbone_runtime_policy.scratch_arena_plan->alignment,
        static_cast<uint32_t>(std::max<int64_t>(
            1, static_cast<int64_t>(c10::elementSize(kFloat)))));
    const size_t requested_bytes = vision_attention_scratch_bytes(
        batch_size,
        token_count,
        embed_dim,
        backbone_context->num_heads(),
        backbone_input.scalar_type(),
        backbone_context->qkv_bias().defined(),
        scratch_alignment);
    if (
        requested_bytes > 0u &&
        backbone_runtime_policy.scratch_arena_plan->prefer_buffer_storage) {
      backbone_graph_scratch = backbone_graph.ensure_shared_scratch(
          std::max(
              requested_bytes,
              backbone_runtime_policy.scratch_arena_plan->min_arena_bytes),
          scratch_alignment,
          backbone_runtime_policy.execution_program_plan->persistent);
    }
  }

  const int64_t backbone_token_count =
      backbone_input.dim() == 2 ? backbone_input.size(0) : backbone_input.size(1);
  const int64_t backbone_embed_dim = backbone_input.size(-1);
  const int64_t backbone_hidden_dim = vision_block_hidden_dim(backbone_context);
  auto backbone_replay = backbone_graph.lookup_or_create_replay(
      vision_backbone_program_label(
          backbone_context->allocation_label(), backbone_context.get()),
      backbone_input.sizes(),
      backbone_token_count,
      backbone_embed_dim,
      backbone_hidden_dim,
      backbone_context->num_heads(),
      *backbone_runtime_policy.execution_program_plan);

  const std::vector<int64_t> decoder_target_sizes =
      resolve_decoder_target_sizes(decoder_input_buffer, decoder_size);
  auto decoder_replay = decoder_graph.lookup_or_create_replay(
      vision_decoder_program_label(
          decoder_context->allocation_label(), decoder_context.get()),
      decoder_input_buffer.sizes(),
      decoder_skip_buffer.has_value()
          ? std::optional<std::vector<int64_t>>(decoder_skip_buffer->sizes().vec())
          : std::nullopt,
      decoder_target_sizes,
      vision_decoder_out_channels(decoder_context),
      *decoder_runtime_policy.execution_program_plan);

  TORCH_CHECK(
      backbone_replay.defined() && decoder_replay.defined(),
      "Vision backbone/decoder replay bundle bridge expected defined replays");

  copy_tensor_for_replay(backbone_replay.input_slot(), backbone_input);
  utils::copy_buffer_tensor_direct_(
      decoder_replay.input_slot(), decoder_input_buffer);
  if (decoder_skip_buffer.has_value() && decoder_skip_buffer->defined()) {
    TORCH_INTERNAL_ASSERT(
        decoder_replay.skip_slot().has_value(),
        "Vision backbone/decoder replay bundle bridge expected decoder skip slot");
    utils::copy_buffer_tensor_direct_(
        *decoder_replay.skip_slot(), *decoder_skip_buffer);
  }
  api::context()->flush_pending_cmds();

  const std::string root_label =
      current_graph_capture_label("depth.vision", "depth.vision.graph");
  const VisionReplayBundleIdentity bundle_identity =
      make_vision_backbone_decoder_bundle_identity(
          backbone_context,
          backbone_input,
          decoder_context,
          decoder_input_buffer,
          decoder_skip_buffer,
          decoder_target_sizes);
  auto root = utils::lookup_or_create_labeled_execution_graph_root(
      root_label,
      kFloat,
      backbone_runtime_policy.execution_program_plan->persistent &&
          decoder_runtime_policy.execution_program_plan->persistent);
  auto replay_bundle = root.lookup_or_create_replay_bundle(
      bundle_identity.key,
      [&]() -> utils::ExecutionGraphReplayBundle {
        std::vector<utils::ExecutionGraphReplayStep> steps;
        steps.reserve(2u);
        steps.push_back(make_vision_backbone_replay_step(
            backbone_replay, backbone_context, backbone_graph_scratch));
        steps.push_back(make_vision_decoder_replay_step(
            decoder_replay, decoder_target_sizes, decoder_context));
        return utils::make_execution_graph_replay_bundle(
            root.allocation_label() + ".vision.backbone_decoder.replay" +
                bundle_identity.label_suffix,
            kFloat,
            backbone_runtime_policy.execution_program_plan->persistent &&
                decoder_runtime_policy.execution_program_plan->persistent,
            std::move(steps));
      });
  TORCH_CHECK(
      replay_bundle.defined() && replay_bundle.size() == 2u,
      "Vision backbone/decoder replay bundle bridge expected a 2-phase bundle");

  if (!replay_bundle.recorded()) {
    Tensor warmup_backbone_output = utils::create_buffer_tensor(
        backbone_replay.output_slot().sizes(),
        backbone_replay.output_slot().scalar_type(),
        /*persistent=*/false);
    Tensor warmup_decoder_output = utils::create_buffer_tensor(
        decoder_replay.output_slot().sizes(),
        decoder_replay.output_slot().scalar_type(),
        /*persistent=*/false);
    if (backbone_graph_scratch.has_value()) {
      backbone_graph_scratch->reset();
    }
    api::RuntimeLabelScope backbone_runtime_scope(compose_runtime_capture_label(
        vision_backbone_execution_label(
            backbone_context->allocation_label(), backbone_context.get())));
    (void)run_vision_backbone_block_program(
        backbone_replay.input_slot(),
        backbone_context,
        &backbone_replay.program(),
        backbone_graph_scratch.has_value() ? &(*backbone_graph_scratch) : nullptr,
        &backbone_replay.output_slot());
    copy_tensor_for_replay(warmup_backbone_output, backbone_replay.output_slot());
    const VisionDecoderRunOutputs replay_outputs{
        decoder_replay.program().skip_relu_output(),
        decoder_replay.program().skip_conv1_output(),
        decoder_replay.program().skip_conv2_output(),
        decoder_replay.program().skip_res_output(),
        decoder_replay.program().main_input_output(),
        decoder_replay.program().main_relu_output(),
        decoder_replay.program().main_conv1_output(),
        decoder_replay.program().main_conv2_output(),
        decoder_replay.program().main_res_output(),
        decoder_replay.program().upsample_output(),
        decoder_replay.program().out_conv_output(),
    };
    utils::copy_buffer_tensor_direct_(
        warmup_decoder_output,
        run_vision_decoder_fusion_block_program(
            decoder_replay.input_slot(),
            decoder_replay.skip_slot(),
            decoder_target_sizes,
            decoder_context,
            replay_outputs));
    api::context()->flush_pending_cmds();
    replay_bundle.record();
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_backbone_decoder_replay_bundle_bridge.replay_warmup");
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_backbone_decoder_replay_bundle_bridge");
    return std::make_tuple(
        maybe_restore_tensor(
            warmup_backbone_output, backbone_output_device, backbone_output_dtype),
        maybe_restore_tensor(
            warmup_decoder_output, decoder_output_device, decoder_output_dtype));
  }

  run_recorded_compiled_replay_or_direct_steps(
      replay_bundle,
      "vulkan_prepack::run_vision_backbone_decoder_replay_bundle_bridge",
      "vision.backbone_decoder.replay");

  Tensor backbone_output = utils::create_buffer_tensor(
      backbone_replay.output_slot().sizes(),
      backbone_replay.output_slot().scalar_type(),
      /*persistent=*/false);
  copy_tensor_for_replay(backbone_output, backbone_replay.output_slot());
  Tensor decoder_output = utils::create_buffer_tensor(
      decoder_replay.output_slot().sizes(),
      decoder_replay.output_slot().scalar_type(),
      /*persistent=*/false);
  utils::copy_buffer_tensor_direct_(
      decoder_output, decoder_replay.output_slot());
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_vision_backbone_decoder_replay_bundle_bridge");
  return std::make_tuple(
      maybe_restore_tensor(
          backbone_output, backbone_output_device, backbone_output_dtype),
      maybe_restore_tensor(
          decoder_output, decoder_output_device, decoder_output_dtype));
}

std::vector<Tensor> run_vision_backbone_stack_replay_bundle_bridge_impl(
    const Tensor& input_arg,
    const c10::List<c10::intrusive_ptr<VisionBackboneBlockContext>>& contexts,
    IntArrayRef capture_indices,
    const std::optional<std::vector<int64_t>>& output_norm_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& output_norm_context) {
  TORCH_CHECK(
      contexts.size() > 0,
      "Vision backbone stack replay bundle bridge expects at least one context");
  const bool apply_output_norm =
      output_norm_shape.has_value() && static_cast<bool>(output_norm_context);

  const std::vector<int64_t> capture_indices_vec = capture_indices.vec();
  for (const int64_t capture_idx : capture_indices_vec) {
    TORCH_CHECK(
        capture_idx >= 0 &&
            capture_idx < static_cast<int64_t>(contexts.size()),
        "Vision backbone stack replay bundle bridge capture index ",
        capture_idx,
        " is out of range for ",
        contexts.size(),
        " contexts");
  }
  if (capture_indices_vec.empty()) {
    return {};
  }

  std::vector<c10::intrusive_ptr<VisionBackboneBlockContext>> backbone_contexts;
  backbone_contexts.reserve(contexts.size());
  for (const auto& context_ref : contexts) {
    c10::intrusive_ptr<VisionBackboneBlockContext> context = context_ref;
    TORCH_CHECK(
        static_cast<bool>(context),
        "Vision backbone stack replay bundle bridge expects defined contexts");
    backbone_contexts.push_back(std::move(context));
  }

  const auto sequential_fallback =
      [&]() -> std::vector<Tensor> {
    Tensor current = input_arg;
    std::vector<Tensor> outputs(capture_indices_vec.size());
    for (size_t idx = 0u; idx < backbone_contexts.size(); ++idx) {
      current = run_vision_backbone_block_context(current, backbone_contexts[idx]);
      for (size_t capture_pos = 0u; capture_pos < capture_indices_vec.size();
           ++capture_pos) {
        if (capture_indices_vec[capture_pos] == static_cast<int64_t>(idx)) {
          outputs[capture_pos] = apply_output_norm
              ? run_layernorm_context(
                    current, *output_norm_shape, output_norm_context)
              : current;
        }
      }
    }
    return outputs;
  };

  TORCH_CHECK(
      input_arg.dim() == 2 || input_arg.dim() == 3,
      "Vision backbone stack replay bundle bridge expects rank-2 or rank-3 input");

  const Device output_device = input_arg.device();
  const ScalarType output_dtype = input_arg.scalar_type();
  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();

  auto backbone_request = utils::make_vulkan_vision_backbone_request();
  backbone_request.fixed_shape_graph_input_sizes = input.sizes().vec();
  backbone_request.prefer_packed_layout_propagation = true;
  utils::VulkanPlanningRequestScope planning_scope(backbone_request);
  const auto runtime_policy =
      utils::build_vulkan_runtime_policy(backbone_request);
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionBackbone ||
      input.scalar_type() != kFloat) {
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_backbone_stack_replay_bundle_bridge.skip.no_runtime_program");
    return sequential_fallback();
  }

  const VisionReplayBundleIdentity bundle_identity =
      make_vision_backbone_stack_bundle_identity(
          backbone_contexts,
          capture_indices_vec,
          output_norm_shape,
          apply_output_norm ? output_norm_context.get() : nullptr);
  const std::string root_label =
      current_graph_capture_label("depth.vision", "depth.vision.graph");
  std::vector<int64_t> block_hidden_dims;
  std::vector<int64_t> block_num_heads;
  block_hidden_dims.reserve(backbone_contexts.size());
  block_num_heads.reserve(backbone_contexts.size());
  for (const auto& context : backbone_contexts) {
    block_hidden_dims.push_back(vision_block_hidden_dim(context));
    block_num_heads.push_back(context->num_heads());
  }
  const auto compiled_session =
      utils::lookup_or_create_vision_transformer_depth_backbone_session(
          utils::VisionTransformerDepthBackboneSessionDesc{
              root_label + ".backbone_stack" + bundle_identity.label_suffix,
              input.sizes().vec(),
              input.scalar_type(),
              static_cast<int64_t>(backbone_contexts.size()),
              capture_indices_vec,
              std::move(block_hidden_dims),
              std::move(block_num_heads),
              output_norm_shape,
              runtime_policy.execution_program_plan->persistent});

  const int64_t batch_size = input.dim() == 2 ? 1 : input.size(0);
  const int64_t token_count = input.dim() == 2 ? input.size(0) : input.size(1);
  const int64_t embed_dim = input.size(-1);
  const uint32_t scratch_alignment =
      runtime_policy.scratch_arena_plan.has_value()
      ? std::max<uint32_t>(
            runtime_policy.scratch_arena_plan->alignment,
            static_cast<uint32_t>(std::max<int64_t>(
                1, static_cast<int64_t>(c10::elementSize(kFloat)))))
      : 1u;

  std::vector<std::optional<utils::ScratchArena>> graph_scratches;
  graph_scratches.reserve(backbone_contexts.size());
  std::vector<utils::VisionBackboneInferenceReplay> replays;
  replays.reserve(backbone_contexts.size());
  for (const auto& context : backbone_contexts) {
    auto vision_graph = prime_vision_backbone_graph(input, runtime_policy, context);
    if (!vision_graph.defined()) {
      utils::log_vulkan_op_hit(
          "vulkan_prepack::run_vision_backbone_stack_replay_bundle_bridge.skip.no_graph");
      return sequential_fallback();
    }
    std::optional<utils::ScratchArena> graph_scratch = std::nullopt;
    if (
        runtime_policy.scratch_arena_plan.has_value() &&
        runtime_policy.scratch_arena_plan->prefer_buffer_storage) {
      const size_t requested_bytes = vision_attention_scratch_bytes(
          batch_size,
          token_count,
          embed_dim,
          context->num_heads(),
          input.scalar_type(),
          context->qkv_bias().defined(),
          scratch_alignment);
      if (requested_bytes > 0u) {
        graph_scratch = vision_graph.ensure_shared_scratch(
            std::max(
                requested_bytes,
                runtime_policy.scratch_arena_plan->min_arena_bytes),
            scratch_alignment,
            runtime_policy.execution_program_plan->persistent);
      }
    }
    const int64_t hidden_dim = vision_block_hidden_dim(context);
    auto replay = vision_graph.lookup_or_create_replay(
        vision_backbone_program_label(
            context->allocation_label(), context.get()),
        input.sizes(),
        token_count,
        embed_dim,
        hidden_dim,
        context->num_heads(),
        *runtime_policy.execution_program_plan);
    if (!replay.defined()) {
      utils::log_vulkan_op_hit(
          "vulkan_prepack::run_vision_backbone_stack_replay_bundle_bridge.skip.no_replay");
      return sequential_fallback();
    }
    graph_scratches.push_back(std::move(graph_scratch));
    replays.push_back(std::move(replay));
  }

  auto root = utils::lookup_or_create_labeled_execution_graph_root(
      root_label,
      kFloat,
      runtime_policy.execution_program_plan->persistent);
  auto replay_bundle = root.lookup_or_create_replay_bundle(
      bundle_identity.key,
      [&]() -> utils::ExecutionGraphReplayBundle {
        std::vector<utils::ExecutionGraphReplayStep> steps;
        steps.reserve(replays.size());
        std::shared_ptr<std::vector<Tensor>> bundle_tensor_slots;

        if (!apply_output_norm) {
          steps.push_back(make_vision_backbone_replay_step(
              replays[0], backbone_contexts[0], graph_scratches[0]));
          for (size_t idx = 1u; idx < replays.size(); ++idx) {
            steps.push_back(make_chained_vision_backbone_replay_step(
                replays[idx - 1u],
                replays[idx],
                backbone_contexts[idx],
                graph_scratches[idx]));
          }
        } else {
          auto output_norm_slots =
              std::make_shared<std::vector<Tensor>>();
          output_norm_slots->reserve(capture_indices_vec.size());
          for (size_t capture_pos = 0u;
               capture_pos < capture_indices_vec.size();
               ++capture_pos) {
            output_norm_slots->push_back(utils::create_buffer_tensor(
                input.sizes(),
                kFloat,
                /*persistent=*/true));
          }

          const std::vector<int64_t> norm_shape = *output_norm_shape;
          bundle_tensor_slots = output_norm_slots;
          for (size_t idx = 0u; idx < replays.size(); ++idx) {
            std::optional<size_t> capture_pos = std::nullopt;
            for (size_t pos = 0u; pos < capture_indices_vec.size(); ++pos) {
              if (capture_indices_vec[pos] == static_cast<int64_t>(idx)) {
                capture_pos = pos;
                break;
              }
            }

            auto previous_replay =
                idx == 0u ? utils::VisionBackboneInferenceReplay{}
                          : replays[idx - 1u];
            auto backbone_replay = replays[idx];
            auto backbone_context = backbone_contexts[idx];
            auto graph_scratch = graph_scratches[idx];
            steps.push_back(backbone_replay.phase_step(
                [previous_replay,
                 backbone_replay,
                 backbone_context,
                 graph_scratch,
                 capture_pos,
                 output_norm_slots,
                 norm_shape,
                 output_norm_context]() mutable {
                  if (graph_scratch.has_value()) {
                    graph_scratch->reset();
                  }
                  const Tensor& replay_input = previous_replay.defined()
                      ? previous_replay.output_slot()
                      : backbone_replay.input_slot();
                  (void)run_vision_backbone_block_program(
                      replay_input,
                      backbone_context,
                      &backbone_replay.program(),
                      graph_scratch.has_value() ? &(*graph_scratch) : nullptr,
                      &backbone_replay.output_slot());
                  if (capture_pos.has_value()) {
                    (void)run_layernorm_context_out(
                        backbone_replay.output_slot(),
                        norm_shape,
                        output_norm_context,
                        output_norm_slots->at(*capture_pos));
                  }
                }));
          }
        }
        return utils::make_execution_graph_replay_bundle(
            root.allocation_label() + ".vision.backbone_stack.replay" +
                bundle_identity.label_suffix,
            kFloat,
            runtime_policy.execution_program_plan->persistent,
            std::move(steps),
            std::move(bundle_tensor_slots));
      });
  TORCH_CHECK(
      replay_bundle.defined() && replay_bundle.size() == replays.size(),
      "Vision backbone stack replay bundle bridge expected a replay bundle "
      "matching the number of contexts");
  if (apply_output_norm) {
    TORCH_CHECK(
        replay_bundle.tensor_slot_count() == capture_indices_vec.size(),
        "Vision backbone stack replay bundle bridge expected one norm output "
        "slot per captured output");
  }
  const char* replay_warmup_log_name = apply_output_norm
      ? "vulkan_prepack::run_vision_backbone_stack_norm_replay_bundle_bridge.replay_warmup"
      : "vulkan_prepack::run_vision_backbone_stack_replay_bundle_bridge.replay_warmup";
  const char* replay_log_name = apply_output_norm
      ? "vulkan_prepack::run_vision_backbone_stack_norm_replay_bundle_bridge.replay"
      : "vulkan_prepack::run_vision_backbone_stack_replay_bundle_bridge.replay";
  const char* bridge_log_name = apply_output_norm
      ? "vulkan_prepack::run_vision_backbone_stack_norm_replay_bundle_bridge"
      : "vulkan_prepack::run_vision_backbone_stack_replay_bundle_bridge";

  copy_tensor_for_replay(replays[0].input_slot(), input);
  api::context()->flush_pending_cmds();

  const bool first_record = !replay_bundle.recorded();
  for (size_t idx = 0u; idx < replays.size(); ++idx) {
    if (graph_scratches[idx].has_value()) {
      graph_scratches[idx]->reset();
    }
    api::RuntimeLabelScope runtime_scope(compose_runtime_capture_label(
        vision_backbone_execution_label(
            backbone_contexts[idx]->allocation_label(),
            backbone_contexts[idx].get())));
    const Tensor& replay_input =
        idx == 0u ? replays[idx].input_slot() : replays[idx - 1u].output_slot();
    (void)run_vision_backbone_block_program(
        replay_input,
        backbone_contexts[idx],
        &replays[idx].program(),
        graph_scratches[idx].has_value() ? &(*graph_scratches[idx]) : nullptr,
        &replays[idx].output_slot());
    for (size_t capture_pos = 0u; capture_pos < capture_indices_vec.size();
         ++capture_pos) {
      if (capture_indices_vec[capture_pos] != static_cast<int64_t>(idx)) {
        continue;
      }
      if (apply_output_norm) {
        Tensor& norm_slot = replay_bundle.tensor_slot(capture_pos);
        (void)run_layernorm_context_out(
            replays[idx].output_slot(),
            *output_norm_shape,
            output_norm_context,
            norm_slot);
      }
    }
  }
  api::context()->flush_pending_cmds();
  if (first_record) {
    replay_bundle.record_empty();
  }
  run_recorded_compiled_replay_or_direct_steps(
      replay_bundle,
      apply_output_norm
          ? "vulkan_prepack::run_vision_backbone_stack_norm_replay_bundle_bridge"
          : "vulkan_prepack::run_vision_backbone_stack_replay_bundle_bridge",
      "vision.backbone_stack.replay");
  std::vector<Tensor> outputs(capture_indices_vec.size());
  for (size_t capture_pos = 0u; capture_pos < capture_indices_vec.size();
       ++capture_pos) {
    const int64_t replay_idx = capture_indices_vec[capture_pos];
    const Tensor& replay_output = apply_output_norm
        ? replay_bundle.tensor_slot(capture_pos)
        : replays[replay_idx].output_slot();
    (void)utils::stamp_replay_export(
        replay_output,
        apply_output_norm ? replay_bundle.identity()
                          : replays[replay_idx].identity(),
        static_cast<uint32_t>(capture_pos),
        apply_output_norm
            ? "run_vision_backbone_stack_norm_replay_bundle_bridge"
            : "run_vision_backbone_stack_replay_bundle_bridge");
    Tensor output = utils::create_buffer_tensor(
        replay_output.sizes(),
        replay_output.scalar_type(),
        /*persistent=*/true);
    copy_tensor_for_replay(output, replay_output);
    outputs[capture_pos] =
        maybe_restore_tensor(output, output_device, output_dtype);
    record_tensor_write(
        outputs[capture_pos],
        bridge_log_name,
        "materialized_replay_export",
        {replay_output});
  }
  utils::log_vulkan_op_hit(first_record ? replay_warmup_log_name : replay_log_name);
  utils::log_vulkan_op_hit(bridge_log_name);
  return outputs;
}

std::vector<Tensor> run_vision_backbone_stack_compiled_session_bridge_impl(
    const Tensor& input_arg,
    const c10::List<c10::intrusive_ptr<VisionBackboneBlockContext>>& contexts,
    IntArrayRef capture_indices,
    const std::optional<std::vector<int64_t>>& output_norm_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& output_norm_context) {
  TORCH_CHECK(
      contexts.size() > 0,
      "Vision backbone stack compiled session bridge expects at least one context");
  const bool apply_output_norm =
      output_norm_shape.has_value() && static_cast<bool>(output_norm_context);

  const std::vector<int64_t> capture_indices_vec = capture_indices.vec();
  for (const int64_t capture_idx : capture_indices_vec) {
    TORCH_CHECK(
        capture_idx >= 0 &&
            capture_idx < static_cast<int64_t>(contexts.size()),
        "Vision backbone stack compiled session bridge capture index ",
        capture_idx,
        " is out of range for ",
        contexts.size(),
        " contexts");
  }
  if (capture_indices_vec.empty()) {
    return {};
  }

  std::vector<c10::intrusive_ptr<VisionBackboneBlockContext>> backbone_contexts;
  backbone_contexts.reserve(contexts.size());
  for (const auto& context_ref : contexts) {
    c10::intrusive_ptr<VisionBackboneBlockContext> context = context_ref;
    TORCH_CHECK(
        static_cast<bool>(context),
        "Vision backbone stack compiled session bridge expects defined contexts");
    backbone_contexts.push_back(std::move(context));
  }

  const Device output_device = input_arg.device();
  const ScalarType output_dtype = input_arg.scalar_type();
  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();

  auto backbone_request = utils::make_vulkan_vision_backbone_request();
  backbone_request.fixed_shape_graph_input_sizes = input.sizes().vec();
  backbone_request.prefer_packed_layout_propagation = true;
  utils::VulkanPlanningRequestScope planning_scope(backbone_request);
  const auto runtime_policy =
      utils::build_vulkan_runtime_policy(backbone_request);

  const VisionReplayBundleIdentity bundle_identity =
      make_vision_backbone_stack_bundle_identity(
          backbone_contexts,
          capture_indices_vec,
          output_norm_shape,
          apply_output_norm ? output_norm_context.get() : nullptr);
  const std::string root_label =
      current_graph_capture_label("depth.vision", "depth.vision.graph");

  std::vector<int64_t> block_hidden_dims;
  std::vector<int64_t> block_num_heads;
  block_hidden_dims.reserve(backbone_contexts.size());
  block_num_heads.reserve(backbone_contexts.size());
  for (const auto& context : backbone_contexts) {
    block_hidden_dims.push_back(vision_block_hidden_dim(context));
    block_num_heads.push_back(context->num_heads());
  }

  const auto compiled_session =
      utils::lookup_or_create_vision_transformer_depth_backbone_session(
          utils::VisionTransformerDepthBackboneSessionDesc{
              root_label + ".backbone_stack.compiled" + bundle_identity.label_suffix,
              input.sizes().vec(),
              input.scalar_type(),
              static_cast<int64_t>(backbone_contexts.size()),
              capture_indices_vec,
              std::move(block_hidden_dims),
              std::move(block_num_heads),
              output_norm_shape,
              runtime_policy.execution_program_plan.has_value()
                  ? runtime_policy.execution_program_plan->persistent
                  : true});

  if (auto compiled_outputs = try_run_vision_backbone_stack_compiled_session(
          input,
          output_device,
          output_dtype,
          backbone_contexts,
          capture_indices_vec,
          output_norm_shape,
          output_norm_context,
          runtime_policy,
          bundle_identity,
          root_label,
          compiled_session)) {
    return *compiled_outputs;
  }

  return run_vision_backbone_stack_replay_bundle_bridge_impl(
      input_arg,
      contexts,
      capture_indices,
      output_norm_shape,
      output_norm_context);
}

std::vector<Tensor> run_vision_backbone_stack_replay_bundle_bridge(
    const Tensor& input,
    const c10::List<c10::intrusive_ptr<VisionBackboneBlockContext>>& contexts,
    IntArrayRef capture_indices) {
  return run_vision_backbone_stack_replay_bundle_bridge_impl(
      input,
      contexts,
      capture_indices,
      std::nullopt,
      c10::intrusive_ptr<LayernormPackedContext>());
}

std::vector<Tensor> run_vision_backbone_stack_compiled_session_bridge(
    const Tensor& input,
    const c10::List<c10::intrusive_ptr<VisionBackboneBlockContext>>& contexts,
    IntArrayRef capture_indices) {
  return run_vision_backbone_stack_compiled_session_bridge_impl(
      input,
      contexts,
      capture_indices,
      std::nullopt,
      c10::intrusive_ptr<LayernormPackedContext>());
}

std::vector<Tensor> run_vision_backbone_stack_norm_replay_bundle_bridge(
    const Tensor& input,
    const c10::List<c10::intrusive_ptr<VisionBackboneBlockContext>>& contexts,
    IntArrayRef capture_indices,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& norm_context) {
  TORCH_CHECK(
      static_cast<bool>(norm_context),
      "Vision backbone stack norm replay bundle bridge expects a defined "
      "LayerNorm context");
  return run_vision_backbone_stack_replay_bundle_bridge_impl(
      input,
      contexts,
      capture_indices,
      normalized_shape.vec(),
      norm_context);
}

std::vector<Tensor> run_vision_backbone_stack_norm_compiled_session_bridge(
    const Tensor& input,
    const c10::List<c10::intrusive_ptr<VisionBackboneBlockContext>>& contexts,
    IntArrayRef capture_indices,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& norm_context) {
  TORCH_CHECK(
      static_cast<bool>(norm_context),
      "Vision backbone stack norm compiled session bridge expects a defined "
      "LayerNorm context");
  return run_vision_backbone_stack_compiled_session_bridge_impl(
      input,
      contexts,
      capture_indices,
      normalized_shape.vec(),
      norm_context);
}

Tensor run_depth_anything_v2_compiled_session_bridge(
    const Tensor& input_arg,
    const c10::List<c10::intrusive_ptr<VisionBackboneBlockContext>>& contexts,
    IntArrayRef capture_indices,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& norm_context,
    const int64_t patch_h,
    const int64_t patch_w,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderPreprocessHeadContext>&
        decoder_context) {
  TORCH_CHECK(
      contexts.size() > 0,
      "Depth Anything compiled session bridge expects at least one backbone "
      "context");
  TORCH_CHECK(
      capture_indices.size() == 4,
      "Depth Anything compiled session bridge expects exactly four capture "
      "indices");
  TORCH_CHECK(
      !normalized_shape.empty() && norm_context,
      "Depth Anything compiled session bridge expects a defined output norm "
      "context and normalized shape");
  TORCH_CHECK(
      patch_h > 0 && patch_w > 0,
      "Depth Anything compiled session bridge expects positive patch sizes");
  TORCH_CHECK(
      output_size.size() == 2 && decoder_context,
      "Depth Anything compiled session bridge expects a decoder context and a "
      "rank-1 output size with 2 entries");
  TORCH_CHECK(
      input_arg.dim() == 2 || input_arg.dim() == 3,
      "Depth Anything compiled session bridge expects rank-2 or rank-3 patch "
      "tokens");

  for (const auto& context_ref : contexts) {
    c10::intrusive_ptr<VisionBackboneBlockContext> context = context_ref;
    TORCH_CHECK(
        static_cast<bool>(context),
        "Depth Anything compiled session bridge expects defined backbone "
        "contexts");
  }
  const int64_t input_token_count =
      input_arg.dim() == 2 ? input_arg.size(0) : input_arg.size(1);
  const int64_t special_token_count = input_token_count - (patch_h * patch_w);
  TORCH_CHECK(
      special_token_count >= 0,
      "Depth Anything compiled session bridge expected at least ",
      patch_h * patch_w,
      " patch tokens but received ",
      input_token_count,
      " total tokens");
  const Device output_device = input_arg.device();
  const ScalarType output_dtype = input_arg.scalar_type();

  const std::string detail =
      "action=use_replay_bundle_backbone "
      "reason=depth_anything_compiled_backbone_guard "
      "failure_class=ReplayHangRisk";
  api::report_vulkan_failure(
      api::VulkanFailureClass::ReplayHangRisk,
      "vulkan_prepack::run_depth_anything_v2_compiled_session_bridge",
      "DepthAnythingCompiledBackboneGuard",
      detail);
  utils::log_replay_event(
      "depth_anything_compiled_backbone_guard",
      nullptr,
      0u,
      "vision.depth_anything_v2.compiled_session",
      detail);
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_depth_anything_v2_compiled_session_bridge.guard.compiled_backbone_disabled");
  std::vector<Tensor> captured =
      run_vision_backbone_stack_norm_replay_bundle_bridge(
          input_arg,
          contexts,
          capture_indices,
          normalized_shape,
          norm_context);
  TORCH_CHECK(
      captured.size() == 4u,
      "Depth Anything compiled session bridge expected four captured tensors");
  const auto strip_special_tokens = [&](const Tensor& tensor) {
    if (tensor.dim() == 2) {
      return tensor.slice(0, special_token_count, tensor.size(0));
    }
    return tensor.slice(1, special_token_count, tensor.size(1));
  };
  Tensor output = run_vision_decoder_preprocess_head_context(
      strip_special_tokens(captured[0]),
      strip_special_tokens(captured[1]),
      strip_special_tokens(captured[2]),
      strip_special_tokens(captured[3]),
      patch_h,
      patch_w,
      output_size,
      decoder_context);
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_depth_anything_v2_compiled_session_bridge");
  Tensor restored = maybe_restore_tensor(output, output_device, output_dtype);
  record_tensor_write(
      restored,
      "vulkan_prepack::run_depth_anything_v2_compiled_session_bridge",
      "decoder_output",
      {captured[0], captured[1], captured[2], captured[3]});
  return restored;
}

Tensor run_depth_anything_v2_image_compiled_session_bridge(
    const Tensor& input_arg,
    const c10::intrusive_ptr<Conv2dPackedContext>& patch_embed_context,
    const Tensor& prefix_token_arg,
    const Tensor& patch_pos_encoding_arg,
    const c10::List<c10::intrusive_ptr<VisionBackboneBlockContext>>& contexts,
    IntArrayRef capture_indices,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& norm_context,
    const int64_t patch_h,
    const int64_t patch_w,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderPreprocessHeadContext>&
        decoder_context) {
  TORCH_CHECK(
      contexts.size() > 0,
      "Depth Anything image compiled session bridge expects at least one "
      "backbone context");
  TORCH_CHECK(
      capture_indices.size() == 4,
      "Depth Anything image compiled session bridge expects exactly four "
      "capture indices");
  TORCH_CHECK(
      !normalized_shape.empty() && norm_context,
      "Depth Anything image compiled session bridge expects a defined output "
      "norm context and normalized shape");
  TORCH_CHECK(
      patch_h > 0 && patch_w > 0,
      "Depth Anything image compiled session bridge expects positive patch "
      "sizes");
  TORCH_CHECK(
      output_size.size() == 2 && decoder_context,
      "Depth Anything image compiled session bridge expects a decoder context "
      "and a rank-1 output size with 2 entries");
  TORCH_CHECK(
      static_cast<bool>(patch_embed_context),
      "Depth Anything image compiled session bridge expects a defined patch "
      "embed context");
  TORCH_CHECK(
      input_arg.dim() == 4,
      "Depth Anything image compiled session bridge expects a rank-4 image "
      "tensor");
  TORCH_CHECK(
      prefix_token_arg.dim() == 3 && patch_pos_encoding_arg.dim() == 3,
      "Depth Anything image compiled session bridge expects rank-3 prefix "
      "and positional encoding tensors");

  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const bool flatten_batch1_tokens = input.size(0) == 1;
  const int64_t special_token_count = prefix_token_arg.size(1);
  TORCH_CHECK(
      special_token_count >= 0,
      "Depth Anything image compiled session bridge expects a non-negative "
      "special token count");

  const Device output_device = input_arg.device();
  const ScalarType output_dtype = input_arg.scalar_type();
  const auto supported_route = [&]() -> Tensor {
    Tensor tokens = make_depth_anything_v2_tokens_from_image(
        input,
        patch_embed_context,
        prefix_token_arg,
        patch_pos_encoding_arg,
        flatten_batch1_tokens);
    return run_depth_anything_v2_compiled_session_bridge(
        tokens,
        contexts,
        capture_indices,
        normalized_shape,
        norm_context,
        patch_h,
        patch_w,
        output_size,
        decoder_context);
  };

  TORCH_CHECK(
      patch_pos_encoding_arg.size(1) == patch_h * patch_w,
      "Depth Anything image compiled session bridge expected ",
      patch_h * patch_w,
      " positional patch tokens but received ",
      patch_pos_encoding_arg.size(1));
  TORCH_CHECK(
      prefix_token_arg.size(2) == patch_pos_encoding_arg.size(2),
      "Depth Anything image compiled session bridge received mismatched "
      "token embedding dimensions");
  Tensor output = supported_route();
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_depth_anything_v2_image_compiled_session_bridge");
  Tensor restored = maybe_restore_tensor(output, output_device, output_dtype);
  record_tensor_write(
      restored,
      "vulkan_prepack::run_depth_anything_v2_image_compiled_session_bridge",
      "image_depth_output",
      {input_arg});
  return restored;
}

Tensor tokens_to_feature_map(
    const Tensor& input_arg,
    const int64_t height,
    const int64_t width) {
  if (!input_arg.is_vulkan() || input_arg.scalar_type() != kFloat) {
    return tokens_to_feature_map_fallback(input_arg, height, width);
  }

  Tensor output = utils::create_buffer_tensor(
      tokens_to_feature_map_output_sizes(input_arg.sizes(), height, width),
      input_arg.scalar_type(),
      /*persistent=*/false);
  if (::at::native::vulkan::ops::run_tokens_to_feature_map_direct_out(
          input_arg, height, width, output)) {
    return output;
  }

  const bool use_2d_input = input_arg.dim() == 2;
  const int64_t batch_size = use_2d_input ? 1 : input_arg.size(0);
  const int64_t channels = input_arg.size(-1);
  utils::log_vulkan_op_hit("aten::tokens_to_feature_map.texture_view_fallback");
  if (use_2d_input) {
    return tokens_to_feature_map_fallback(input_arg, height, width);
  }
  return input_arg.permute({0, 2, 1})
      .reshape({batch_size, channels, height, width});
}

Tensor feature_map_to_tokens(const Tensor& input_arg) {
  if (!input_arg.is_vulkan() || input_arg.scalar_type() != kFloat) {
    return feature_map_to_tokens_fallback(input_arg);
  }

  TORCH_CHECK(
      input_arg.dim() == 4,
      "Vulkan feature_map_to_tokens expects a [B, C, H, W] tensor");

  Tensor output = utils::create_buffer_tensor(
      {
          input_arg.size(0),
          input_arg.size(2) * input_arg.size(3),
          input_arg.size(1),
      },
      input_arg.scalar_type(),
      /*persistent=*/false);
  if (run_feature_map_to_tokens_direct_out(input_arg, output)) {
    return output;
  }

  utils::log_vulkan_op_hit("aten::feature_map_to_tokens.fallback");
  return feature_map_to_tokens_fallback(input_arg);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
