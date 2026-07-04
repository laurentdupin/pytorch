#include <ATen/native/vulkan/planning/DynamicProgramRuntime.h>

#include <algorithm>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

bool positive(const int64_t value) {
  return value > 0;
}

bool complete_contract_metadata(
    const ExecutionContractMetadata* const metadata) {
  return has_complete_execution_contract_metadata(metadata);
}

void fill_contract_key(
    DynamicProgramKey& key,
    const ExecutionContractMetadata* const metadata) {
  if (metadata == nullptr) {
    return;
  }
  key.contract_name = metadata->contract_name;
  key.contract_family = metadata->family_name;
  key.contract_tuple_id = metadata->tuple_id;
}

bool is_pointwise_conv1x1_semantics(
    const DynamicProgramRequest& request) {
  const auto& shape = request.shape;
  return request.dtype == kFloat && request.rank == 4 &&
      request.input_direct_buffer && request.weight_direct_buffer &&
      request.output_direct_buffer && positive(shape.batch) &&
      positive(shape.input_channels) && positive(shape.output_channels) &&
      positive(shape.height) && positive(shape.width) &&
      shape.kernel_h == 1 && shape.kernel_w == 1 &&
      shape.stride_h == 1 && shape.stride_w == 1 &&
      shape.padding_h == 0 && shape.padding_w == 0 &&
      shape.dilation_h == 1 && shape.dilation_w == 1 &&
      shape.groups == 1;
}

bool is_conv2d_direct_buffer_semantics(const DynamicProgramRequest& request) {
  const auto& shape = request.shape;
  return request.dtype == kFloat && request.rank == 4 &&
      request.input_direct_buffer && request.weight_direct_buffer &&
      request.output_direct_buffer && positive(shape.batch) &&
      positive(shape.input_channels) &&
      positive(shape.weight_input_channels) &&
      positive(shape.output_channels) && positive(shape.height) &&
      positive(shape.width) && positive(shape.kernel_h) &&
      positive(shape.kernel_w) && positive(shape.stride_h) &&
      positive(shape.stride_w) && shape.padding_h >= 0 &&
      shape.padding_w >= 0 && positive(shape.dilation_h) &&
      positive(shape.dilation_w) && positive(shape.groups) &&
      shape.input_channels == shape.weight_input_channels * shape.groups &&
      shape.output_channels % shape.groups == 0;
}

int64_t numel_or_zero(const IntArrayRef sizes) {
  if (sizes.empty()) {
    return 0;
  }
  int64_t numel = 1;
  for (const int64_t size : sizes) {
    if (!positive(size)) {
      return 0;
    }
    numel *= size;
  }
  return numel;
}

bool broadcast_compatible(const IntArrayRef self, const IntArrayRef other) {
  const int64_t self_rank = static_cast<int64_t>(self.size());
  const int64_t other_rank = static_cast<int64_t>(other.size());
  const int64_t rank = std::max(self_rank, other_rank);
  for (int64_t i = 0; i < rank; ++i) {
    const int64_t self_index = self_rank - 1 - i;
    const int64_t other_index = other_rank - 1 - i;
    const int64_t self_dim = self_index >= 0 ? self[self_index] : 1;
    const int64_t other_dim = other_index >= 0 ? other[other_index] : 1;
    if (self_dim != other_dim && self_dim != 1 && other_dim != 1) {
      return false;
    }
  }
  return true;
}

int64_t broadcast_output_numel_or_zero(
    const IntArrayRef self,
    const IntArrayRef other) {
  if (!broadcast_compatible(self, other)) {
    return 0;
  }
  const int64_t self_rank = static_cast<int64_t>(self.size());
  const int64_t other_rank = static_cast<int64_t>(other.size());
  const int64_t rank = std::max(self_rank, other_rank);
  int64_t numel = 1;
  for (int64_t i = 0; i < rank; ++i) {
    const int64_t self_index = self_rank - 1 - i;
    const int64_t other_index = other_rank - 1 - i;
    const int64_t self_dim = self_index >= 0 ? self[self_index] : 1;
    const int64_t other_dim = other_index >= 0 ? other[other_index] : 1;
    const int64_t dim = std::max(self_dim, other_dim);
    if (!positive(dim)) {
      return 0;
    }
    numel *= dim;
  }
  return numel;
}

bool is_elementwise_broadcast_semantics(
    const DynamicProgramRequest& request) {
  return request.dtype == kFloat && request.other_dtype == kFloat &&
      request.output_dtype == kFloat && request.shape.self_rank >= 1 &&
      request.shape.self_rank <= 4 && request.shape.other_rank >= 1 &&
      request.shape.other_rank <= 4 && request.shape.output_rank >= 1 &&
      request.shape.output_rank <= 4 && request.input_direct_buffer &&
      request.weight_direct_buffer && request.output_direct_buffer &&
      request.elementwise_op != ElementwiseBroadcastOp::Unsupported &&
      request.alpha_is_one && !request.has_output && !request.inplace &&
      request.broadcast_compatible && positive(request.shape.self_numel) &&
      positive(request.shape.other_numel) &&
      positive(request.shape.output_numel);
}

bool is_sequence_cat_direct_buffer_semantics(
    const DynamicProgramRequest& request) {
  const auto& shape = request.shape;
  return request.dtype == kFloat && request.other_dtype == kFloat &&
      request.output_dtype == kFloat && request.rank == 4 &&
      request.input_direct_buffer && request.weight_direct_buffer &&
      request.output_direct_buffer && shape.self_rank == 4 &&
      shape.other_rank == 4 && shape.output_rank == 4 && shape.cat_dim == 2 &&
      positive(shape.batch) && positive(shape.heads) &&
      positive(shape.left_sequence) && positive(shape.right_sequence) &&
      shape.output_sequence == shape.left_sequence + shape.right_sequence &&
      positive(shape.head_dim);
}

bool is_linear_or_matmul_semantics(const DynamicProgramRequest& request) {
  const auto& shape = request.shape;
  return request.dtype == kFloat && (request.rank == 2 || request.rank == 3) &&
      request.input_direct_buffer && request.weight_direct_buffer &&
      request.output_direct_buffer && request.post_op_none &&
      positive(shape.m) && positive(shape.k) && positive(shape.n) &&
      shape.k == shape.rhs_k && shape.lhs_rank == request.rank &&
      shape.rhs_rank == 2 && shape.output_rank == request.rank;
}

DynamicProgramCommandPlan pointwise_conv1x1_static_shader_plan() {
  DynamicProgramCommandPlan plan;
  plan.shader_policy = DynamicProgramShaderSelectionPolicy::ExistingStaticShader;
  plan.command_plan = DynamicProgramCommandPlanKind::SingleDispatch;
  plan.cache_policy = DynamicProgramCachePolicy::CapabilityProfileProgramKey;
  plan.shader_family = "conv2d_buffer_float_1x1";
  plan.command_list_label = "pointwise_conv1x1_single_dispatch";
  plan.requires_runtime_shader_compile = false;
  plan.requires_custom_command_list = false;
  return plan;
}

DynamicProgramCommandPlan conv2d_direct_buffer_static_shader_plan() {
  DynamicProgramCommandPlan plan;
  plan.shader_policy = DynamicProgramShaderSelectionPolicy::ExistingStaticShader;
  plan.command_plan = DynamicProgramCommandPlanKind::SingleDispatch;
  plan.cache_policy = DynamicProgramCachePolicy::CapabilityProfileProgramKey;
  plan.shader_family = "conv2d_buffer_float";
  plan.command_list_label = "conv2d_direct_buffer_single_dispatch";
  plan.requires_runtime_shader_compile = false;
  plan.requires_custom_command_list = false;
  return plan;
}

DynamicProgramCommandPlan sequence_cat_direct_buffer_static_shader_plan() {
  DynamicProgramCommandPlan plan;
  plan.shader_policy = DynamicProgramShaderSelectionPolicy::ExistingStaticShader;
  plan.command_plan = DynamicProgramCommandPlanKind::SingleDispatch;
  plan.cache_policy = DynamicProgramCachePolicy::CapabilityProfileProgramKey;
  plan.shader_family = "cat_dim2_4d_buffer_float";
  plan.command_list_label = "sequence_cat_dim2_4d_single_dispatch";
  plan.requires_runtime_shader_compile = false;
  plan.requires_custom_command_list = false;
  return plan;
}

DynamicProgramCommandPlan elementwise_broadcast_static_shader_plan() {
  DynamicProgramCommandPlan plan;
  plan.shader_policy = DynamicProgramShaderSelectionPolicy::ExistingStaticShader;
  plan.command_plan = DynamicProgramCommandPlanKind::SingleDispatch;
  plan.cache_policy = DynamicProgramCachePolicy::CapabilityProfileProgramKey;
  plan.shader_family = "binary_op_buffer_float";
  plan.command_list_label = "elementwise_broadcast_single_dispatch";
  plan.requires_runtime_shader_compile = false;
  plan.requires_custom_command_list = false;
  return plan;
}

DynamicProgramCommandPlan linear_or_matmul_static_shader_plan(
    const bool has_bias) {
  DynamicProgramCommandPlan plan;
  plan.shader_policy = DynamicProgramShaderSelectionPolicy::ExistingStaticShader;
  plan.command_plan = DynamicProgramCommandPlanKind::SingleDispatch;
  plan.cache_policy = DynamicProgramCachePolicy::CapabilityProfileProgramKey;
  plan.shader_family = has_bias ? "mm_buffer_float_bias" : "mm_buffer_float";
  plan.command_list_label = "linear_or_matmul_single_dispatch";
  plan.requires_runtime_shader_compile = false;
  plan.requires_custom_command_list = false;
  return plan;
}

const char* status_for_reject(const DynamicProgramRejectReason reason) {
  switch (reason) {
    case DynamicProgramRejectReason::None:
      return "dynamic_program_runtime_selection_authorized";
    case DynamicProgramRejectReason::MissingContract:
      return "dynamic_program_runtime_rejected_missing_contract";
    case DynamicProgramRejectReason::IncompleteProgramKey:
      return "dynamic_program_runtime_rejected_incomplete_program_key";
    case DynamicProgramRejectReason::UnsupportedSemanticFamily:
      return "dynamic_program_runtime_rejected_unsupported_semantic_family";
    case DynamicProgramRejectReason::UnsupportedDType:
      return "dynamic_program_runtime_rejected_unsupported_dtype";
    case DynamicProgramRejectReason::UnsupportedRank:
      return "dynamic_program_runtime_rejected_unsupported_rank";
    case DynamicProgramRejectReason::UnsupportedLayout:
      return "dynamic_program_runtime_rejected_unsupported_layout";
    case DynamicProgramRejectReason::UnsupportedKernelSemantics:
      return "dynamic_program_runtime_rejected_unsupported_kernel_semantics";
    case DynamicProgramRejectReason::MissingPipelinePolicy:
      return "dynamic_program_runtime_rejected_missing_pipeline_policy";
    case DynamicProgramRejectReason::MissingCommandPlan:
      return "dynamic_program_runtime_rejected_missing_command_plan";
    case DynamicProgramRejectReason::RuntimeCompilationUnavailable:
      return "dynamic_program_runtime_rejected_runtime_compilation_unavailable";
    case DynamicProgramRejectReason::BehaviorDisabled:
      return "dynamic_program_runtime_scaffold_present_behavior_disabled";
  }
  return "dynamic_program_runtime_rejected_incomplete_program_key";
}

} // namespace

const char* dynamic_program_semantic_family_name(
    const DynamicProgramSemanticFamily family) {
  switch (family) {
    case DynamicProgramSemanticFamily::PointwiseConv1x1DirectBuffer:
      return "PointwiseConv1x1DirectBuffer";
    case DynamicProgramSemanticFamily::Conv2DDirectBuffer:
      return "Conv2DDirectBuffer";
    case DynamicProgramSemanticFamily::SequenceCatDirectBuffer:
      return "SequenceCatDirectBuffer";
    case DynamicProgramSemanticFamily::ElementwiseBroadcastDirectBuffer:
      return "ElementwiseBroadcastDirectBuffer";
    case DynamicProgramSemanticFamily::LinearOrMatmulDirectBuffer:
      return "LinearOrMatmulDirectBuffer";
    case DynamicProgramSemanticFamily::StackRegionCommandReplay:
      return "StackRegionCommandReplay";
    case DynamicProgramSemanticFamily::None:
      return "None";
  }
  return "None";
}

const char* dynamic_program_shader_selection_policy_name(
    const DynamicProgramShaderSelectionPolicy policy) {
  switch (policy) {
    case DynamicProgramShaderSelectionPolicy::ExistingStaticShader:
      return "ExistingStaticShader";
    case DynamicProgramShaderSelectionPolicy::RuntimeSpecializedShader:
      return "RuntimeSpecializedShader";
    case DynamicProgramShaderSelectionPolicy::RuntimeGeneratedShader:
      return "RuntimeGeneratedShader";
    case DynamicProgramShaderSelectionPolicy::CachedCompiledPipeline:
      return "CachedCompiledPipeline";
    case DynamicProgramShaderSelectionPolicy::None:
      return "None";
  }
  return "None";
}

const char* dynamic_program_command_plan_kind_name(
    const DynamicProgramCommandPlanKind kind) {
  switch (kind) {
    case DynamicProgramCommandPlanKind::SingleDispatch:
      return "SingleDispatch";
    case DynamicProgramCommandPlanKind::MultiDispatch:
      return "MultiDispatch";
    case DynamicProgramCommandPlanKind::CustomCommandList:
      return "CustomCommandList";
    case DynamicProgramCommandPlanKind::RegionCommandList:
      return "RegionCommandList";
    case DynamicProgramCommandPlanKind::None:
      return "None";
  }
  return "None";
}

const char* dynamic_program_cache_policy_name(
    const DynamicProgramCachePolicy policy) {
  switch (policy) {
    case DynamicProgramCachePolicy::EvidenceOnly:
      return "EvidenceOnly";
    case DynamicProgramCachePolicy::ProgramKeyLocal:
      return "ProgramKeyLocal";
    case DynamicProgramCachePolicy::CapabilityProfileProgramKey:
      return "CapabilityProfileProgramKey";
    case DynamicProgramCachePolicy::PersistentPipelineCache:
      return "PersistentPipelineCache";
    case DynamicProgramCachePolicy::None:
      return "None";
  }
  return "None";
}

const char* dynamic_program_reject_reason_name(
    const DynamicProgramRejectReason reason) {
  switch (reason) {
    case DynamicProgramRejectReason::None:
      return "None";
    case DynamicProgramRejectReason::MissingContract:
      return "MissingContract";
    case DynamicProgramRejectReason::IncompleteProgramKey:
      return "IncompleteProgramKey";
    case DynamicProgramRejectReason::UnsupportedSemanticFamily:
      return "UnsupportedSemanticFamily";
    case DynamicProgramRejectReason::UnsupportedDType:
      return "UnsupportedDType";
    case DynamicProgramRejectReason::UnsupportedRank:
      return "UnsupportedRank";
    case DynamicProgramRejectReason::UnsupportedLayout:
      return "UnsupportedLayout";
    case DynamicProgramRejectReason::UnsupportedKernelSemantics:
      return "UnsupportedKernelSemantics";
    case DynamicProgramRejectReason::MissingPipelinePolicy:
      return "MissingPipelinePolicy";
    case DynamicProgramRejectReason::MissingCommandPlan:
      return "MissingCommandPlan";
    case DynamicProgramRejectReason::RuntimeCompilationUnavailable:
      return "RuntimeCompilationUnavailable";
    case DynamicProgramRejectReason::BehaviorDisabled:
      return "BehaviorDisabled";
  }
  return "IncompleteProgramKey";
}

DynamicProgramDecision build_dynamic_program_runtime_plan(
    const DynamicProgramRequest& request) {
  DynamicProgramDecision decision;
  decision.key.semantic_family = request.semantic_family;
  decision.key.dtype = request.dtype;
  decision.key.other_dtype = request.other_dtype;
  decision.key.output_dtype = request.output_dtype;
  decision.key.rank = request.rank;
  decision.key.shape = request.shape;
  fill_contract_key(decision.key, request.contract_metadata);
  decision.behavior_enabled = request.behavior_enabled;

  switch (request.semantic_family) {
    case DynamicProgramSemanticFamily::PointwiseConv1x1DirectBuffer:
      if (!is_pointwise_conv1x1_semantics(request)) {
        decision.reject_reason =
            (request.dtype != kFloat || request.other_dtype != kFloat ||
             request.output_dtype != kFloat)
            ? DynamicProgramRejectReason::UnsupportedDType
            : request.rank != 4
            ? DynamicProgramRejectReason::UnsupportedRank
            : !(request.input_direct_buffer && request.weight_direct_buffer &&
                request.output_direct_buffer)
            ? DynamicProgramRejectReason::UnsupportedLayout
            : DynamicProgramRejectReason::UnsupportedKernelSemantics;
        decision.status = status_for_reject(decision.reject_reason);
        return decision;
      }
      break;
    case DynamicProgramSemanticFamily::Conv2DDirectBuffer:
      if (!is_conv2d_direct_buffer_semantics(request)) {
        decision.reject_reason =
            request.dtype != kFloat
            ? DynamicProgramRejectReason::UnsupportedDType
            : request.rank != 4
            ? DynamicProgramRejectReason::UnsupportedRank
            : !(request.input_direct_buffer && request.weight_direct_buffer &&
                request.output_direct_buffer)
            ? DynamicProgramRejectReason::UnsupportedLayout
            : DynamicProgramRejectReason::UnsupportedKernelSemantics;
        decision.status = status_for_reject(decision.reject_reason);
        return decision;
      }
      break;
    case DynamicProgramSemanticFamily::SequenceCatDirectBuffer:
      if (!is_sequence_cat_direct_buffer_semantics(request)) {
        decision.reject_reason =
            (request.dtype != kFloat || request.other_dtype != kFloat ||
             request.output_dtype != kFloat)
            ? DynamicProgramRejectReason::UnsupportedDType
            : (request.rank != 4 || request.shape.self_rank != 4 ||
               request.shape.other_rank != 4 || request.shape.output_rank != 4)
            ? DynamicProgramRejectReason::UnsupportedRank
            : !(request.input_direct_buffer && request.weight_direct_buffer &&
                request.output_direct_buffer)
            ? DynamicProgramRejectReason::UnsupportedLayout
            : DynamicProgramRejectReason::UnsupportedKernelSemantics;
        decision.status = status_for_reject(decision.reject_reason);
        return decision;
      }
      break;
    case DynamicProgramSemanticFamily::ElementwiseBroadcastDirectBuffer:
      if (!is_elementwise_broadcast_semantics(request)) {
        decision.reject_reason =
            request.dtype != kFloat
            ? DynamicProgramRejectReason::UnsupportedDType
            : (request.shape.self_rank < 1 || request.shape.self_rank > 4 ||
               request.shape.other_rank < 1 || request.shape.other_rank > 4 ||
               request.shape.output_rank < 1 || request.shape.output_rank > 4)
            ? DynamicProgramRejectReason::UnsupportedRank
            : !(request.input_direct_buffer && request.weight_direct_buffer &&
                request.output_direct_buffer)
            ? DynamicProgramRejectReason::UnsupportedLayout
            : DynamicProgramRejectReason::UnsupportedKernelSemantics;
        decision.status = status_for_reject(decision.reject_reason);
        return decision;
      }
      break;
    case DynamicProgramSemanticFamily::LinearOrMatmulDirectBuffer:
      if (!is_linear_or_matmul_semantics(request)) {
        decision.reject_reason =
            request.dtype != kFloat
            ? DynamicProgramRejectReason::UnsupportedDType
            : (request.rank != 2 && request.rank != 3)
            ? DynamicProgramRejectReason::UnsupportedRank
            : !(request.input_direct_buffer && request.weight_direct_buffer &&
                request.output_direct_buffer)
            ? DynamicProgramRejectReason::UnsupportedLayout
            : DynamicProgramRejectReason::UnsupportedKernelSemantics;
        decision.status = status_for_reject(decision.reject_reason);
        return decision;
      }
      break;
    case DynamicProgramSemanticFamily::StackRegionCommandReplay:
    case DynamicProgramSemanticFamily::None:
      decision.reject_reason =
          DynamicProgramRejectReason::UnsupportedSemanticFamily;
      decision.status = status_for_reject(decision.reject_reason);
      return decision;
  }
  decision.semantic_validation_passed = true;

  decision.command_plan =
      request.semantic_family ==
          DynamicProgramSemanticFamily::ElementwiseBroadcastDirectBuffer
      ? elementwise_broadcast_static_shader_plan()
      : request.semantic_family ==
              DynamicProgramSemanticFamily::SequenceCatDirectBuffer
      ? sequence_cat_direct_buffer_static_shader_plan()
      : request.semantic_family ==
              DynamicProgramSemanticFamily::Conv2DDirectBuffer
      ? conv2d_direct_buffer_static_shader_plan()
      : request.semantic_family ==
              DynamicProgramSemanticFamily::LinearOrMatmulDirectBuffer
      ? linear_or_matmul_static_shader_plan(request.has_bias)
      : pointwise_conv1x1_static_shader_plan();
  decision.command_plan_available = true;

  if (!complete_contract_metadata(request.contract_metadata)) {
    decision.reject_reason = DynamicProgramRejectReason::MissingContract;
    decision.status = status_for_reject(decision.reject_reason);
    return decision;
  }
  decision.program_key_complete = true;

  if (!request.behavior_enabled) {
    decision.reject_reason = DynamicProgramRejectReason::BehaviorDisabled;
    decision.status = status_for_reject(decision.reject_reason);
    decision.runtime_selection_authorized = false;
    return decision;
  }

  decision.reject_reason = DynamicProgramRejectReason::None;
  decision.status = status_for_reject(decision.reject_reason);
  decision.runtime_selection_authorized = true;
  return decision;
}

DynamicProgramAdmission admit_dynamic_program(
    const DynamicProgramRequest& request) {
  const DynamicProgramDecision decision =
      build_dynamic_program_runtime_plan(request);
  DynamicProgramAdmission admission;
  admission.reject_reason = decision.reject_reason;
  admission.status = decision.status;
  admission.accepted =
      decision.semantic_validation_passed && decision.command_plan_available &&
      decision.reject_reason != DynamicProgramRejectReason::UnsupportedDType &&
      decision.reject_reason != DynamicProgramRejectReason::UnsupportedRank &&
      decision.reject_reason != DynamicProgramRejectReason::UnsupportedLayout &&
      decision.reject_reason !=
          DynamicProgramRejectReason::UnsupportedKernelSemantics &&
      decision.reject_reason !=
          DynamicProgramRejectReason::UnsupportedSemanticFamily;
  return admission;
}

DynamicProgramRequest make_pointwise_conv1x1_direct_buffer_program_request(
    const IntArrayRef input_sizes,
    const IntArrayRef weight_sizes,
    const bool has_bias,
    const ScalarType dtype) {
  const int64_t stride_values[] = {1, 1};
  const int64_t padding_values[] = {0, 0};
  const int64_t dilation_values[] = {1, 1};
  DynamicProgramRequest request =
      make_pointwise_conv1x1_direct_buffer_dynamic_program(
          input_sizes,
          weight_sizes,
          IntArrayRef(stride_values, 2),
          IntArrayRef(padding_values, 2),
          IntArrayRef(dilation_values, 2),
          1,
          dtype,
          nullptr,
          false);
  request.has_bias = has_bias;
  return request;
}

DynamicProgramRequest make_pointwise_conv1x1_direct_buffer_dynamic_program(
    const IntArrayRef input_sizes,
    const IntArrayRef weight_sizes,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups,
    const ScalarType dtype,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request;
  request.semantic_family =
      DynamicProgramSemanticFamily::PointwiseConv1x1DirectBuffer;
  request.dtype = dtype;
  request.other_dtype = dtype;
  request.output_dtype = dtype;
  request.rank = input_sizes.size();
  if (
      input_sizes.size() == 4 && weight_sizes.size() == 4 &&
      stride.size() == 2 && padding.size() == 2 && dilation.size() == 2) {
    request.shape.batch = input_sizes[0];
    request.shape.input_channels = input_sizes[1];
    request.shape.height = input_sizes[2];
    request.shape.width = input_sizes[3];
    request.shape.output_channels = weight_sizes[0];
    request.shape.kernel_h = weight_sizes[2];
    request.shape.kernel_w = weight_sizes[3];
    request.shape.stride_h = stride[0];
    request.shape.stride_w = stride[1];
    request.shape.padding_h = padding[0];
    request.shape.padding_w = padding[1];
    request.shape.dilation_h = dilation[0];
    request.shape.dilation_w = dilation[1];
    request.shape.groups = groups;
  }
  request.capabilities.has_pipeline_cache = true;
  request.contract_metadata = contract_metadata;
  request.input_direct_buffer = true;
  request.weight_direct_buffer = true;
  request.output_direct_buffer = true;
  request.has_bias = false;
  request.behavior_enabled = behavior_enabled;
  return request;
}

DynamicProgramRequest make_conv2d_direct_buffer_dynamic_program(
    const IntArrayRef input_sizes,
    const IntArrayRef weight_sizes,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups,
    const ScalarType dtype,
    const bool input_direct_buffer,
    const bool weight_direct_buffer,
    const bool output_direct_buffer,
    const bool has_bias,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request;
  request.semantic_family = DynamicProgramSemanticFamily::Conv2DDirectBuffer;
  request.dtype = dtype;
  request.other_dtype = dtype;
  request.output_dtype = dtype;
  request.rank = static_cast<int64_t>(input_sizes.size());
  if (
      input_sizes.size() == 4 && weight_sizes.size() == 4 &&
      stride.size() == 2 && padding.size() == 2 && dilation.size() == 2) {
    request.shape.batch = input_sizes[0];
    request.shape.input_channels = input_sizes[1];
    request.shape.weight_input_channels = weight_sizes[1];
    request.shape.height = input_sizes[2];
    request.shape.width = input_sizes[3];
    request.shape.output_channels = weight_sizes[0];
    request.shape.kernel_h = weight_sizes[2];
    request.shape.kernel_w = weight_sizes[3];
    request.shape.stride_h = stride[0];
    request.shape.stride_w = stride[1];
    request.shape.padding_h = padding[0];
    request.shape.padding_w = padding[1];
    request.shape.dilation_h = dilation[0];
    request.shape.dilation_w = dilation[1];
    request.shape.groups = groups;
  }
  request.capabilities.has_pipeline_cache = true;
  request.contract_metadata = contract_metadata;
  request.input_direct_buffer = input_direct_buffer;
  request.weight_direct_buffer = weight_direct_buffer;
  request.output_direct_buffer = output_direct_buffer;
  request.has_bias = has_bias;
  request.behavior_enabled = behavior_enabled;
  return request;
}

DynamicProgramRequest make_elementwise_broadcast_direct_buffer_dynamic_program(
    const IntArrayRef self_sizes,
    const IntArrayRef other_sizes,
    const ScalarType self_dtype,
    const ScalarType other_dtype,
    const ScalarType output_dtype,
    const bool self_direct_buffer,
    const bool other_direct_buffer,
    const bool output_direct_buffer,
    const ElementwiseBroadcastOp op,
    const bool alpha_is_one,
    const bool has_output,
    const bool inplace,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request;
  request.semantic_family =
      DynamicProgramSemanticFamily::ElementwiseBroadcastDirectBuffer;
  request.dtype = self_dtype;
  request.other_dtype = other_dtype;
  request.output_dtype = output_dtype;
  request.rank =
      static_cast<int64_t>(std::max(self_sizes.size(), other_sizes.size()));
  request.shape.self_rank = static_cast<int64_t>(self_sizes.size());
  request.shape.other_rank = static_cast<int64_t>(other_sizes.size());
  request.shape.output_rank = request.rank;
  request.shape.self_numel = numel_or_zero(self_sizes);
  request.shape.other_numel = numel_or_zero(other_sizes);
  request.shape.output_numel =
      broadcast_output_numel_or_zero(self_sizes, other_sizes);
  request.contract_metadata = contract_metadata;
  request.input_direct_buffer = self_direct_buffer;
  request.weight_direct_buffer = other_direct_buffer;
  request.output_direct_buffer = output_direct_buffer;
  request.elementwise_op = op;
  request.alpha_is_one = alpha_is_one;
  request.has_output = has_output;
  request.inplace = inplace;
  request.broadcast_compatible =
      broadcast_compatible(self_sizes, other_sizes);
  request.behavior_enabled = behavior_enabled;
  return request;
}

DynamicProgramRequest make_sequence_cat_direct_buffer_dynamic_program(
    const IntArrayRef left_sizes,
    const IntArrayRef right_sizes,
    const ScalarType left_dtype,
    const ScalarType right_dtype,
    const ScalarType output_dtype,
    const bool left_direct_buffer,
    const bool right_direct_buffer,
    const bool output_direct_buffer,
    const int64_t dim,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request;
  request.semantic_family = DynamicProgramSemanticFamily::SequenceCatDirectBuffer;
  request.dtype = left_dtype;
  request.other_dtype = right_dtype;
  request.output_dtype = output_dtype;
  request.rank = left_sizes.size();
  request.shape.self_rank = static_cast<int64_t>(left_sizes.size());
  request.shape.other_rank = static_cast<int64_t>(right_sizes.size());
  request.shape.output_rank = request.rank;
  request.shape.cat_dim = dim;
  if (left_sizes.size() == 4 && right_sizes.size() == 4) {
    request.shape.batch = left_sizes[0] == right_sizes[0] ? left_sizes[0] : 0;
    request.shape.heads = left_sizes[1] == right_sizes[1] ? left_sizes[1] : 0;
    request.shape.left_sequence = left_sizes[2];
    request.shape.right_sequence = right_sizes[2];
    request.shape.output_sequence = left_sizes[2] + right_sizes[2];
    request.shape.head_dim =
        left_sizes[3] == right_sizes[3] ? left_sizes[3] : 0;
  }
  request.contract_metadata = contract_metadata;
  request.input_direct_buffer = left_direct_buffer;
  request.weight_direct_buffer = right_direct_buffer;
  request.output_direct_buffer = output_direct_buffer;
  request.has_output = false;
  request.inplace = false;
  request.behavior_enabled = behavior_enabled;
  return request;
}

DynamicProgramRequest make_linear_or_matmul_direct_buffer_program_request(
    const IntArrayRef input_sizes,
    const IntArrayRef weight_sizes,
    const bool has_bias,
    const ScalarType dtype) {
  return make_linear_or_matmul_direct_buffer_dynamic_program(
      input_sizes, weight_sizes, has_bias, dtype, nullptr, false);
}

DynamicProgramRequest make_linear_or_matmul_direct_buffer_dynamic_program(
    const IntArrayRef input_sizes,
    const IntArrayRef weight_sizes,
    const bool has_bias,
    const ScalarType dtype,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request;
  request.semantic_family =
      DynamicProgramSemanticFamily::LinearOrMatmulDirectBuffer;
  request.dtype = dtype;
  request.other_dtype = dtype;
  request.output_dtype = dtype;
  request.rank = input_sizes.size();
  if (
      (input_sizes.size() == 2 || input_sizes.size() == 3) &&
      weight_sizes.size() == 2) {
    int64_t m = 1;
    for (int64_t i = 0; i + 1 < static_cast<int64_t>(input_sizes.size());
         ++i) {
      m *= input_sizes[i];
    }
    request.shape.m = m;
    request.shape.k = input_sizes[input_sizes.size() - 1];
    request.shape.rhs_k = weight_sizes[0];
    request.shape.n = weight_sizes[1];
    request.shape.lhs_rank = static_cast<int64_t>(input_sizes.size());
    request.shape.rhs_rank = static_cast<int64_t>(weight_sizes.size());
    request.shape.output_rank = request.rank;
  }
  request.capabilities.has_pipeline_cache = true;
  request.contract_metadata = contract_metadata;
  request.input_direct_buffer = true;
  request.weight_direct_buffer = true;
  request.output_direct_buffer = true;
  request.has_bias = has_bias;
  request.post_op_none = true;
  request.behavior_enabled = behavior_enabled;
  return request;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
