#include <ATen/native/vulkan/planning/DynamicProgramRuntime.h>

#include <algorithm>
#include <cmath>

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
  if (!(request.dtype == kFloat && request.rank == 4 &&
      request.input_direct_buffer && request.weight_direct_buffer &&
      request.output_direct_buffer && positive(shape.batch) &&
      positive(shape.input_channels) &&
      positive(shape.weight_input_channels) &&
      positive(shape.output_channels) && positive(shape.height) &&
      positive(shape.width) && positive(shape.kernel_h) &&
      positive(shape.kernel_w) && positive(shape.stride_h) &&
      positive(shape.stride_w) && shape.padding_h >= 0 &&
      shape.padding_w >= 0 && shape.dilation_h == 1 &&
      shape.dilation_w == 1 && shape.groups == 1 &&
      shape.input_channels == shape.weight_input_channels * shape.groups &&
      shape.output_channels % shape.groups == 0)) {
    return false;
  }
  const int64_t output_h =
      (shape.height + 2 * shape.padding_h - shape.kernel_h) / shape.stride_h +
      1;
  const int64_t output_w =
      (shape.width + 2 * shape.padding_w - shape.kernel_w) / shape.stride_w +
      1;
  return positive(output_h) && positive(output_w);
}

bool is_packed_buffer_conv2d_semantics(const DynamicProgramRequest& request) {
  const auto& shape = request.shape;
  if (!(request.dtype == kFloat && request.rank == 4 &&
      request.input_buffer_storage && request.weight_buffer_storage &&
      request.output_buffer_storage && shape.batch == 1 &&
      positive(shape.input_channels) &&
      positive(shape.weight_input_channels) &&
      positive(shape.output_channels) && positive(shape.height) &&
      positive(shape.width) && positive(shape.kernel_h) &&
      positive(shape.kernel_w) && positive(shape.stride_h) &&
      positive(shape.stride_w) && shape.padding_h >= 0 &&
      shape.padding_w >= 0 && shape.dilation_h == 1 &&
      shape.dilation_w == 1 && shape.groups == 1 &&
      shape.input_channels == shape.weight_input_channels * shape.groups &&
      shape.output_channels % shape.groups == 0)) {
    return false;
  }
  const int64_t output_h =
      (shape.height + 2 * shape.padding_h - shape.kernel_h) / shape.stride_h +
      1;
  const int64_t output_w =
      (shape.width + 2 * shape.padding_w - shape.kernel_w) / shape.stride_w +
      1;
  return positive(output_h) && positive(output_w);
}

bool is_patch_embed_float_buffer_conv_route_semantics(
    const DynamicProgramRequest& request) {
  const auto& shape = request.shape;
  if (
      request.dtype != kFloat || request.rank != 4 ||
      !request.input_direct_buffer || !request.weight_direct_buffer ||
      !request.output_direct_buffer || shape.batch != 1 ||
      shape.input_channels != 3 || shape.weight_input_channels != 3 ||
      !positive(shape.output_channels) || shape.kernel_h != 14 ||
      shape.kernel_w != 14 || shape.stride_h != 14 ||
      shape.stride_w != 14 || shape.padding_h != 0 ||
      shape.padding_w != 0 || shape.dilation_h != 1 ||
      shape.dilation_w != 1 || shape.groups != 1 ||
      shape.height < 14 || shape.width < 14) {
    return false;
  }
  const int64_t output_h =
      (shape.height - shape.kernel_h) / shape.stride_h + 1;
  const int64_t output_w =
      (shape.width - shape.kernel_w) / shape.stride_w + 1;
  return positive(output_h) && positive(output_w);
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
  const bool supported_dtype =
      (request.dtype == kFloat && request.other_dtype == kFloat &&
       request.output_dtype == kFloat) ||
      (request.dtype == kBFloat16 && request.other_dtype == kBFloat16 &&
       request.output_dtype == kBFloat16);
  return supported_dtype && request.rank == 4 &&
      request.input_direct_buffer && request.weight_direct_buffer &&
      request.output_direct_buffer && shape.self_rank == 4 &&
      shape.other_rank == 4 && shape.output_rank == 4 && shape.cat_dim == 2 &&
      positive(shape.batch) && positive(shape.heads) &&
      positive(shape.left_sequence) && positive(shape.right_sequence) &&
      shape.output_sequence == shape.left_sequence + shape.right_sequence &&
      positive(shape.head_dim);
}

bool is_initial_sequence_cat_direct_buffer_semantics(
    const DynamicProgramRequest& request) {
  const auto& shape = request.shape;
  const bool supported_dtype =
      (request.dtype == kFloat && request.other_dtype == kFloat &&
       request.output_dtype == kFloat) ||
      (request.dtype == kBFloat16 && request.other_dtype == kBFloat16 &&
       request.output_dtype == kBFloat16);
  return supported_dtype && request.rank == 4 &&
      request.input_direct_buffer && request.weight_direct_buffer &&
      request.output_direct_buffer && shape.self_rank == 1 &&
      shape.other_rank == 4 && shape.output_rank == 4 && shape.cat_dim == 2 &&
      shape.self_numel == 0 && positive(shape.batch) &&
      positive(shape.heads) && positive(shape.right_sequence) &&
      shape.output_sequence == shape.right_sequence &&
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

bool is_embedding_lookup_semantics(const DynamicProgramRequest& request) {
  const auto& shape = request.shape;
  return request.dtype == kFloat &&
      request.other_dtype == kLong && request.output_dtype == kFloat &&
      request.rank == 2 &&
      request.input_direct_buffer && request.weight_direct_buffer &&
      request.output_direct_buffer && request.index_bounds_proven &&
      request.padding_idx_has_hint && !request.scale_grad_by_freq &&
      !request.sparse && shape.index_rank >= 1 && shape.index_rank <= 2 &&
      positive(shape.num_embeddings) && positive(shape.embedding_dim) &&
      positive(shape.num_indices);
}

bool is_feature_map_to_tokens_semantics(
    const DynamicProgramRequest& request) {
  const auto& shape = request.shape;
  return request.dtype == kFloat && request.output_dtype == kFloat &&
      request.rank == 4 && request.input_direct_buffer &&
      request.output_direct_buffer && positive(shape.batch) &&
      positive(shape.input_channels) && positive(shape.height) &&
      positive(shape.width) &&
      shape.output_sequence == shape.height * shape.width;
}

bool is_cat_axis_direct_buffer_semantics(
    const DynamicProgramRequest& request) {
  const auto& shape = request.shape;
  return request.dtype == kFloat && request.output_dtype == kFloat &&
      request.rank == 4 && shape.self_rank == 4 && shape.output_rank == 4 &&
      request.input_direct_buffer && request.output_direct_buffer &&
      shape.cat_dim == 1 && positive(shape.batch) &&
      positive(shape.height) && positive(shape.width) &&
      positive(shape.input_count) && positive(shape.total_cat_dim) &&
      shape.total_cat_dim == shape.output_channels;
}

bool is_batch_norm_inference_direct_buffer_semantics(
    const DynamicProgramRequest& request) {
  const auto& shape = request.shape;
  return request.dtype == kFloat && request.output_dtype == kFloat &&
      request.rank == 4 && request.input_direct_buffer &&
      request.weight_direct_buffer && request.output_direct_buffer &&
      !request.training && positive(shape.batch) &&
      positive(shape.input_channels) && positive(shape.height) &&
      positive(shape.width) && shape.output_channels == shape.input_channels;
}

bool is_gqa_repeat_direct_buffer_semantics(
    const DynamicProgramRequest& request) {
  const auto& shape = request.shape;
  return request.dtype == kFloat && request.output_dtype == kFloat &&
      request.rank == 4 && request.input_direct_buffer &&
      request.output_direct_buffer && positive(shape.batch) &&
      positive(shape.heads) && positive(shape.left_sequence) &&
      positive(shape.head_dim) && shape.repeat_factor > 1 &&
      shape.output_channels == shape.heads * shape.repeat_factor;
}

bool is_direct_decode_gqa_sdpa_direct_buffer_semantics(
    const DynamicProgramRequest& request) {
  constexpr int64_t kDirectGQAMaxHeadDim = 128;
  constexpr int64_t kDirectGQAMaxValueDim = 512;
  const auto& shape = request.shape;
  return request.dtype == kFloat && request.other_dtype == kFloat &&
      request.output_dtype == kFloat && request.rank == 4 &&
      request.input_direct_buffer && request.weight_direct_buffer &&
      request.output_direct_buffer && !request.has_attn_mask &&
      request.dropout_is_zero && !request.is_causal && request.enable_gqa &&
      request.scale_is_default_or_head_dim && shape.batch == 1 &&
      positive(shape.query_heads) && positive(shape.key_value_heads) &&
      shape.query_heads % shape.key_value_heads == 0 &&
      shape.query_sequence == 1 && positive(shape.key_value_sequence) &&
      positive(shape.head_dim) && positive(shape.value_dim) &&
      shape.head_dim <= kDirectGQAMaxHeadDim &&
      shape.value_dim <= kDirectGQAMaxValueDim;
}

bool is_small_non_causal_gqa_sdpa_direct_buffer_semantics(
    const DynamicProgramRequest& request) {
  constexpr int64_t kDirectGQAMaxHeadDim = 128;
  constexpr int64_t kDirectGQAMaxValueDim = 512;
  constexpr int64_t kSmallGQAMaxSequence = 64;
  const auto& shape = request.shape;
  return request.dtype == kFloat && request.other_dtype == kFloat &&
      request.output_dtype == kFloat && request.rank == 4 &&
      request.input_direct_buffer && request.weight_direct_buffer &&
      request.output_direct_buffer && !request.has_attn_mask &&
      request.dropout_is_zero && !request.is_causal && request.enable_gqa &&
      request.scale_is_default_or_head_dim && shape.batch == 1 &&
      positive(shape.query_heads) && positive(shape.key_value_heads) &&
      shape.query_heads % shape.key_value_heads == 0 &&
      positive(shape.query_sequence) &&
      shape.query_sequence <= kSmallGQAMaxSequence &&
      positive(shape.key_value_sequence) &&
      shape.key_value_sequence <= kSmallGQAMaxSequence &&
      positive(shape.head_dim) && positive(shape.value_dim) &&
      shape.head_dim <= kDirectGQAMaxHeadDim &&
      shape.value_dim <= kDirectGQAMaxValueDim;
}

bool is_direct_non_causal_mha_sdpa_direct_buffer_semantics(
    const DynamicProgramRequest& request) {
  constexpr int64_t kDirectGQAMaxHeadDim = 128;
  constexpr int64_t kDirectGQAMaxValueDim = 512;
  const auto& shape = request.shape;
  return request.dtype == kFloat && request.other_dtype == kFloat &&
      request.output_dtype == kFloat && request.rank == 4 &&
      request.input_direct_buffer && request.weight_direct_buffer &&
      request.output_direct_buffer && !request.has_attn_mask &&
      request.dropout_is_zero && !request.is_causal && !request.enable_gqa &&
      request.scale_is_default_or_head_dim && shape.batch == 1 &&
      positive(shape.query_heads) &&
      shape.query_heads == shape.key_value_heads &&
      positive(shape.query_sequence) && positive(shape.key_value_sequence) &&
      positive(shape.head_dim) && positive(shape.value_dim) &&
      shape.head_dim % 16 == 0 && shape.value_dim % 16 == 0 &&
      shape.head_dim <= kDirectGQAMaxHeadDim &&
      shape.value_dim <= kDirectGQAMaxValueDim;
}

bool is_direct_causal_prefill_gqa_sdpa_direct_buffer_semantics(
    const DynamicProgramRequest& request) {
  constexpr int64_t kDirectGQAMaxHeadDim = 128;
  constexpr int64_t kDirectGQAMaxValueDim = 512;
  const auto& shape = request.shape;
  return request.dtype == kFloat && request.other_dtype == kFloat &&
      request.output_dtype == kFloat && request.rank == 4 &&
      request.input_direct_buffer && request.weight_direct_buffer &&
      request.output_direct_buffer && !request.has_attn_mask &&
      request.dropout_is_zero && request.is_causal &&
      (request.enable_gqa || shape.query_heads == shape.key_value_heads) &&
      request.scale_is_default_or_head_dim && shape.batch == 1 &&
      positive(shape.query_heads) && positive(shape.key_value_heads) &&
      shape.query_heads % shape.key_value_heads == 0 &&
      positive(shape.query_sequence) &&
      shape.query_sequence == shape.key_value_sequence &&
      positive(shape.head_dim) && positive(shape.value_dim) &&
      shape.head_dim <= kDirectGQAMaxHeadDim &&
      shape.value_dim <= kDirectGQAMaxValueDim;
}

bool is_token_prefix_cat_add_direct_buffer_semantics(
    const DynamicProgramRequest& request) {
  const auto& shape = request.shape;
  return request.dtype == kFloat && request.other_dtype == kFloat &&
      request.output_dtype == kFloat && request.rank == 3 &&
      request.input_direct_buffer && request.weight_direct_buffer &&
      request.output_direct_buffer && !request.inplace && !request.has_output &&
      shape.cat_dim == 1 && positive(shape.batch) &&
      shape.left_sequence == 1 && positive(shape.right_sequence) &&
      shape.output_sequence == shape.left_sequence + shape.right_sequence &&
      positive(shape.input_channels);
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

DynamicProgramCommandPlan patch_embed_float_buffer_conv_route_plan() {
  DynamicProgramCommandPlan plan;
  plan.shader_policy = DynamicProgramShaderSelectionPolicy::ExistingStaticShader;
  plan.command_plan = DynamicProgramCommandPlanKind::SingleDispatch;
  plan.cache_policy = DynamicProgramCachePolicy::CapabilityProfileProgramKey;
  plan.shader_family = "conv2d_buffer_float";
  plan.command_list_label = "patch_embed_float_buffer_conv_single_dispatch";
  plan.requires_runtime_shader_compile = false;
  plan.requires_custom_command_list = false;
  return plan;
}

DynamicProgramCommandPlan sequence_cat_direct_buffer_static_shader_plan(
    const ScalarType dtype) {
  DynamicProgramCommandPlan plan;
  plan.shader_policy = DynamicProgramShaderSelectionPolicy::ExistingStaticShader;
  plan.command_plan = DynamicProgramCommandPlanKind::SingleDispatch;
  plan.cache_policy = DynamicProgramCachePolicy::CapabilityProfileProgramKey;
  plan.shader_family = dtype == kBFloat16
      ? "cat_dim2_4d_buffer_bfloat16"
      : "cat_dim2_4d_buffer_float";
  plan.command_list_label = "sequence_cat_dim2_4d_single_dispatch";
  plan.requires_runtime_shader_compile = false;
  plan.requires_custom_command_list = false;
  return plan;
}

DynamicProgramCommandPlan initial_sequence_cat_direct_buffer_plan(
    const ScalarType dtype) {
  DynamicProgramCommandPlan plan;
  plan.shader_policy = DynamicProgramShaderSelectionPolicy::ExistingStaticShader;
  plan.command_plan = DynamicProgramCommandPlanKind::MultiDispatch;
  plan.cache_policy = DynamicProgramCachePolicy::CapabilityProfileProgramKey;
  plan.shader_family =
      dtype == kBFloat16 ? "raw_buffer_copy" : "buffer_to_buffer";
  plan.command_list_label = "initial_sequence_cat_direct_buffer_copy";
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

DynamicProgramCommandPlan embedding_lookup_static_shader_plan() {
  DynamicProgramCommandPlan plan;
  plan.shader_policy = DynamicProgramShaderSelectionPolicy::ExistingStaticShader;
  plan.command_plan = DynamicProgramCommandPlanKind::SingleDispatch;
  plan.cache_policy = DynamicProgramCachePolicy::CapabilityProfileProgramKey;
  plan.shader_family = "embedding_2d_buffer_float_long";
  plan.command_list_label = "embedding_lookup_direct_buffer_single_dispatch";
  plan.requires_runtime_shader_compile = false;
  plan.requires_custom_command_list = false;
  return plan;
}

DynamicProgramCommandPlan feature_map_to_tokens_static_shader_plan() {
  DynamicProgramCommandPlan plan;
  plan.shader_policy = DynamicProgramShaderSelectionPolicy::ExistingStaticShader;
  plan.command_plan = DynamicProgramCommandPlanKind::SingleDispatch;
  plan.cache_policy = DynamicProgramCachePolicy::CapabilityProfileProgramKey;
  plan.shader_family = "feature_map_to_tokens_buffer";
  plan.command_list_label = "feature_map_to_tokens_single_dispatch";
  plan.requires_runtime_shader_compile = false;
  plan.requires_custom_command_list = false;
  return plan;
}

DynamicProgramCommandPlan cat_axis_direct_buffer_plan() {
  DynamicProgramCommandPlan plan;
  plan.shader_policy = DynamicProgramShaderSelectionPolicy::ExistingStaticShader;
  plan.command_plan = DynamicProgramCommandPlanKind::MultiDispatch;
  plan.cache_policy = DynamicProgramCachePolicy::CapabilityProfileProgramKey;
  plan.shader_family = "buffer_to_buffer";
  plan.command_list_label = "cat_axis_direct_buffer_multi_dispatch";
  plan.requires_runtime_shader_compile = false;
  plan.requires_custom_command_list = false;
  return plan;
}

DynamicProgramCommandPlan batch_norm_inference_direct_buffer_plan() {
  DynamicProgramCommandPlan plan;
  plan.shader_policy = DynamicProgramShaderSelectionPolicy::ExistingStaticShader;
  plan.command_plan = DynamicProgramCommandPlanKind::SingleDispatch;
  plan.cache_policy = DynamicProgramCachePolicy::CapabilityProfileProgramKey;
  plan.shader_family = "batchnorm_4d_buffer_float";
  plan.command_list_label = "batch_norm_inference_direct_buffer_single_dispatch";
  plan.requires_runtime_shader_compile = false;
  plan.requires_custom_command_list = false;
  return plan;
}

DynamicProgramCommandPlan gqa_repeat_direct_buffer_plan() {
  DynamicProgramCommandPlan plan;
  plan.shader_policy = DynamicProgramShaderSelectionPolicy::ExistingStaticShader;
  plan.command_plan = DynamicProgramCommandPlanKind::SingleDispatch;
  plan.cache_policy = DynamicProgramCachePolicy::CapabilityProfileProgramKey;
  plan.shader_family = "gqa_repeat_buffer_float";
  plan.command_list_label = "gqa_repeat_direct_buffer_single_dispatch";
  plan.requires_runtime_shader_compile = false;
  plan.requires_custom_command_list = false;
  return plan;
}

DynamicProgramCommandPlan packed_buffer_conv2d_plan() {
  DynamicProgramCommandPlan plan;
  plan.shader_policy = DynamicProgramShaderSelectionPolicy::ExistingStaticShader;
  plan.command_plan = DynamicProgramCommandPlanKind::SingleDispatch;
  plan.cache_policy = DynamicProgramCachePolicy::CapabilityProfileProgramKey;
  plan.shader_family = "conv2d_buffer_float";
  plan.command_list_label = "packed_buffer_conv2d_single_dispatch";
  plan.requires_runtime_shader_compile = false;
  plan.requires_custom_command_list = false;
  return plan;
}

DynamicProgramCommandPlan direct_decode_gqa_sdpa_direct_buffer_plan() {
  DynamicProgramCommandPlan plan;
  plan.shader_policy = DynamicProgramShaderSelectionPolicy::ExistingStaticShader;
  plan.command_plan = DynamicProgramCommandPlanKind::SingleDispatch;
  plan.cache_policy = DynamicProgramCachePolicy::CapabilityProfileProgramKey;
  plan.shader_family = "scaled_dot_product_scores_value_gqa_buffer_float";
  plan.command_list_label = "direct_decode_gqa_sdpa_single_dispatch";
  plan.requires_runtime_shader_compile = false;
  plan.requires_custom_command_list = false;
  return plan;
}

DynamicProgramCommandPlan token_prefix_cat_add_direct_buffer_plan() {
  DynamicProgramCommandPlan plan;
  plan.shader_policy = DynamicProgramShaderSelectionPolicy::ExistingStaticShader;
  plan.command_plan = DynamicProgramCommandPlanKind::MultiDispatch;
  plan.cache_policy = DynamicProgramCachePolicy::CapabilityProfileProgramKey;
  plan.shader_family = "binary_op_buffer_float";
  plan.command_list_label = "token_prefix_cat_add_two_add_dispatches";
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
    case DynamicProgramRejectReason::MissingIndexBoundsProof:
      return "dynamic_program_runtime_rejected_missing_index_bounds_proof";
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
    case DynamicProgramSemanticFamily::PackedBufferConv2D:
      return "PackedBufferConv2D";
    case DynamicProgramSemanticFamily::PatchEmbedFloatBufferConvRoute:
      return "PatchEmbedFloatBufferConvRoute";
    case DynamicProgramSemanticFamily::SequenceCatDirectBuffer:
      return "SequenceCatDirectBuffer";
    case DynamicProgramSemanticFamily::InitialSequenceCatDirectBuffer:
      return "InitialSequenceCatDirectBuffer";
    case DynamicProgramSemanticFamily::ElementwiseBroadcastDirectBuffer:
      return "ElementwiseBroadcastDirectBuffer";
    case DynamicProgramSemanticFamily::LinearOrMatmulDirectBuffer:
      return "LinearOrMatmulDirectBuffer";
    case DynamicProgramSemanticFamily::EmbeddingLookupDirectBuffer:
      return "EmbeddingLookupDirectBuffer";
    case DynamicProgramSemanticFamily::FeatureMapToTokensDirectBuffer:
      return "FeatureMapToTokensDirectBuffer";
    case DynamicProgramSemanticFamily::CatAxisDirectBuffer:
      return "CatAxisDirectBuffer";
    case DynamicProgramSemanticFamily::BatchNormInferenceDirectBuffer:
      return "BatchNormInferenceDirectBuffer";
    case DynamicProgramSemanticFamily::GQARepeatDirectBuffer:
      return "GQARepeatDirectBuffer";
    case DynamicProgramSemanticFamily::DirectDecodeGQASDPADirectBuffer:
      return "DirectDecodeGQASDPADirectBuffer";
    case DynamicProgramSemanticFamily::SmallNonCausalGQASDPADirectBuffer:
      return "SmallNonCausalGQASDPADirectBuffer";
    case DynamicProgramSemanticFamily::DirectNonCausalMHASDPADirectBuffer:
      return "DirectNonCausalMHASDPADirectBuffer";
    case DynamicProgramSemanticFamily::DirectCausalPrefillGQASDPADirectBuffer:
      return "DirectCausalPrefillGQASDPADirectBuffer";
    case DynamicProgramSemanticFamily::TokenPrefixCatAddDirectBuffer:
      return "TokenPrefixCatAddDirectBuffer";
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
    case DynamicProgramRejectReason::MissingIndexBoundsProof:
      return "MissingIndexBoundsProof";
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
    case DynamicProgramSemanticFamily::PackedBufferConv2D:
      if (!is_packed_buffer_conv2d_semantics(request)) {
        decision.reject_reason =
            request.dtype != kFloat
            ? DynamicProgramRejectReason::UnsupportedDType
            : request.rank != 4
            ? DynamicProgramRejectReason::UnsupportedRank
            : !(request.input_buffer_storage && request.weight_buffer_storage &&
                request.output_buffer_storage)
            ? DynamicProgramRejectReason::UnsupportedLayout
            : DynamicProgramRejectReason::UnsupportedKernelSemantics;
        decision.status = status_for_reject(decision.reject_reason);
        return decision;
      }
      break;
    case DynamicProgramSemanticFamily::PatchEmbedFloatBufferConvRoute:
      if (!is_patch_embed_float_buffer_conv_route_semantics(request)) {
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
        const bool supported_dtype =
            (request.dtype == kFloat && request.other_dtype == kFloat &&
             request.output_dtype == kFloat) ||
            (request.dtype == kBFloat16 &&
             request.other_dtype == kBFloat16 &&
             request.output_dtype == kBFloat16);
        decision.reject_reason =
            !supported_dtype
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
    case DynamicProgramSemanticFamily::InitialSequenceCatDirectBuffer:
      if (!is_initial_sequence_cat_direct_buffer_semantics(request)) {
        const bool supported_dtype =
            (request.dtype == kFloat && request.other_dtype == kFloat &&
             request.output_dtype == kFloat) ||
            (request.dtype == kBFloat16 &&
             request.other_dtype == kBFloat16 &&
             request.output_dtype == kBFloat16);
        decision.reject_reason =
            !supported_dtype
            ? DynamicProgramRejectReason::UnsupportedDType
            : request.rank != 4 || request.shape.self_rank != 1 ||
                    request.shape.other_rank != 4 ||
                    request.shape.output_rank != 4
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
    case DynamicProgramSemanticFamily::EmbeddingLookupDirectBuffer:
      if (!is_embedding_lookup_semantics(request)) {
        decision.reject_reason =
            (request.dtype != kFloat || request.other_dtype != kLong ||
             request.output_dtype != kFloat)
            ? DynamicProgramRejectReason::UnsupportedDType
            : request.rank != 2 ||
                    (request.shape.index_rank != 1 &&
                     request.shape.index_rank != 2)
            ? DynamicProgramRejectReason::UnsupportedRank
            : !(request.input_direct_buffer && request.weight_direct_buffer &&
                request.output_direct_buffer)
            ? DynamicProgramRejectReason::UnsupportedLayout
            : !request.index_bounds_proven
            ? DynamicProgramRejectReason::MissingIndexBoundsProof
            : DynamicProgramRejectReason::UnsupportedKernelSemantics;
        decision.status = status_for_reject(decision.reject_reason);
        return decision;
      }
      break;
    case DynamicProgramSemanticFamily::FeatureMapToTokensDirectBuffer:
      if (!is_feature_map_to_tokens_semantics(request)) {
        decision.reject_reason =
            (request.dtype != kFloat || request.output_dtype != kFloat)
            ? DynamicProgramRejectReason::UnsupportedDType
            : request.rank != 4
            ? DynamicProgramRejectReason::UnsupportedRank
            : !(request.input_direct_buffer && request.output_direct_buffer)
            ? DynamicProgramRejectReason::UnsupportedLayout
            : DynamicProgramRejectReason::UnsupportedKernelSemantics;
        decision.status = status_for_reject(decision.reject_reason);
        return decision;
      }
      break;
    case DynamicProgramSemanticFamily::CatAxisDirectBuffer:
      if (!is_cat_axis_direct_buffer_semantics(request)) {
        decision.reject_reason =
            (request.dtype != kFloat || request.output_dtype != kFloat)
            ? DynamicProgramRejectReason::UnsupportedDType
            : request.rank != 4
            ? DynamicProgramRejectReason::UnsupportedRank
            : !(request.input_direct_buffer && request.output_direct_buffer)
            ? DynamicProgramRejectReason::UnsupportedLayout
            : DynamicProgramRejectReason::UnsupportedKernelSemantics;
        decision.status = status_for_reject(decision.reject_reason);
        return decision;
      }
      break;
    case DynamicProgramSemanticFamily::BatchNormInferenceDirectBuffer:
      if (!is_batch_norm_inference_direct_buffer_semantics(request)) {
        decision.reject_reason =
            (request.dtype != kFloat || request.output_dtype != kFloat)
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
    case DynamicProgramSemanticFamily::GQARepeatDirectBuffer:
      if (!is_gqa_repeat_direct_buffer_semantics(request)) {
        decision.reject_reason =
            (request.dtype != kFloat || request.output_dtype != kFloat)
            ? DynamicProgramRejectReason::UnsupportedDType
            : request.rank != 4
            ? DynamicProgramRejectReason::UnsupportedRank
            : !(request.input_direct_buffer && request.output_direct_buffer)
            ? DynamicProgramRejectReason::UnsupportedLayout
            : DynamicProgramRejectReason::UnsupportedKernelSemantics;
        decision.status = status_for_reject(decision.reject_reason);
        return decision;
      }
      break;
    case DynamicProgramSemanticFamily::DirectDecodeGQASDPADirectBuffer:
      if (!is_direct_decode_gqa_sdpa_direct_buffer_semantics(request)) {
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
    case DynamicProgramSemanticFamily::SmallNonCausalGQASDPADirectBuffer:
      if (!is_small_non_causal_gqa_sdpa_direct_buffer_semantics(request)) {
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
    case DynamicProgramSemanticFamily::DirectNonCausalMHASDPADirectBuffer:
      if (!is_direct_non_causal_mha_sdpa_direct_buffer_semantics(request)) {
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
    case DynamicProgramSemanticFamily::DirectCausalPrefillGQASDPADirectBuffer:
      if (!is_direct_causal_prefill_gqa_sdpa_direct_buffer_semantics(request)) {
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
    case DynamicProgramSemanticFamily::TokenPrefixCatAddDirectBuffer:
      if (!is_token_prefix_cat_add_direct_buffer_semantics(request)) {
        decision.reject_reason =
            (request.dtype != kFloat || request.other_dtype != kFloat ||
             request.output_dtype != kFloat)
            ? DynamicProgramRejectReason::UnsupportedDType
            : request.rank != 3
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
      ? sequence_cat_direct_buffer_static_shader_plan(request.dtype)
      : request.semantic_family ==
              DynamicProgramSemanticFamily::InitialSequenceCatDirectBuffer
      ? initial_sequence_cat_direct_buffer_plan(request.dtype)
      : request.semantic_family ==
              DynamicProgramSemanticFamily::Conv2DDirectBuffer
      ? conv2d_direct_buffer_static_shader_plan()
      : request.semantic_family ==
              DynamicProgramSemanticFamily::PackedBufferConv2D
      ? packed_buffer_conv2d_plan()
      : request.semantic_family ==
              DynamicProgramSemanticFamily::PatchEmbedFloatBufferConvRoute
      ? patch_embed_float_buffer_conv_route_plan()
      : request.semantic_family ==
              DynamicProgramSemanticFamily::LinearOrMatmulDirectBuffer
      ? linear_or_matmul_static_shader_plan(request.has_bias)
      : request.semantic_family ==
              DynamicProgramSemanticFamily::EmbeddingLookupDirectBuffer
      ? embedding_lookup_static_shader_plan()
      : request.semantic_family ==
              DynamicProgramSemanticFamily::FeatureMapToTokensDirectBuffer
      ? feature_map_to_tokens_static_shader_plan()
      : request.semantic_family ==
              DynamicProgramSemanticFamily::CatAxisDirectBuffer
      ? cat_axis_direct_buffer_plan()
      : request.semantic_family ==
              DynamicProgramSemanticFamily::BatchNormInferenceDirectBuffer
      ? batch_norm_inference_direct_buffer_plan()
      : request.semantic_family ==
              DynamicProgramSemanticFamily::GQARepeatDirectBuffer
      ? gqa_repeat_direct_buffer_plan()
      : request.semantic_family ==
              DynamicProgramSemanticFamily::DirectDecodeGQASDPADirectBuffer
      ? direct_decode_gqa_sdpa_direct_buffer_plan()
      : request.semantic_family ==
              DynamicProgramSemanticFamily::SmallNonCausalGQASDPADirectBuffer
      ? direct_decode_gqa_sdpa_direct_buffer_plan()
      : request.semantic_family ==
              DynamicProgramSemanticFamily::DirectNonCausalMHASDPADirectBuffer
      ? direct_decode_gqa_sdpa_direct_buffer_plan()
      : request.semantic_family ==
              DynamicProgramSemanticFamily::
                  DirectCausalPrefillGQASDPADirectBuffer
      ? direct_decode_gqa_sdpa_direct_buffer_plan()
      : request.semantic_family ==
              DynamicProgramSemanticFamily::TokenPrefixCatAddDirectBuffer
      ? token_prefix_cat_add_direct_buffer_plan()
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
          DynamicProgramRejectReason::UnsupportedSemanticFamily &&
      decision.reject_reason !=
          DynamicProgramRejectReason::MissingIndexBoundsProof;
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
  request.input_buffer_storage = input_direct_buffer;
  request.weight_buffer_storage = weight_direct_buffer;
  request.output_buffer_storage = output_direct_buffer;
  request.has_bias = has_bias;
  request.behavior_enabled = behavior_enabled;
  return request;
}

DynamicProgramRequest make_packed_buffer_conv2d_dynamic_program(
    const IntArrayRef input_sizes,
    const IntArrayRef weight_sizes,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups,
    const ScalarType dtype,
    const bool input_buffer_storage,
    const bool weight_buffer_storage,
    const bool output_buffer_storage,
    const bool has_bias,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request =
      make_conv2d_direct_buffer_dynamic_program(
          input_sizes,
          weight_sizes,
          stride,
          padding,
          dilation,
          groups,
          dtype,
          /*input_direct_buffer=*/false,
          /*weight_direct_buffer=*/false,
          /*output_direct_buffer=*/false,
          has_bias,
          contract_metadata,
          behavior_enabled);
  request.semantic_family = DynamicProgramSemanticFamily::PackedBufferConv2D;
  request.input_buffer_storage = input_buffer_storage;
  request.weight_buffer_storage = weight_buffer_storage;
  request.output_buffer_storage = output_buffer_storage;
  return request;
}

DynamicProgramRequest make_patch_embed_float_buffer_conv_route_dynamic_program(
    const IntArrayRef input_sizes,
    const IntArrayRef weight_sizes,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t groups,
    const ScalarType dtype,
    const bool input_buffer_storage,
    const bool weight_buffer_storage,
    const bool output_buffer_storage,
    const bool has_bias,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request;
  request.semantic_family =
      DynamicProgramSemanticFamily::PatchEmbedFloatBufferConvRoute;
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
  request.input_direct_buffer = input_buffer_storage;
  request.weight_direct_buffer = weight_buffer_storage;
  request.output_direct_buffer = output_buffer_storage;
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

DynamicProgramRequest make_initial_sequence_cat_direct_buffer_dynamic_program(
    const IntArrayRef left_sizes,
    const IntArrayRef right_sizes,
    const ScalarType left_dtype,
    const ScalarType right_dtype,
    const ScalarType output_dtype,
    const bool left_is_vulkan,
    const bool right_buffer_storage,
    const bool output_direct_buffer,
    const int64_t dim,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request;
  request.semantic_family =
      DynamicProgramSemanticFamily::InitialSequenceCatDirectBuffer;
  request.dtype = left_dtype;
  request.other_dtype = right_dtype;
  request.output_dtype = output_dtype;
  request.rank = 4;
  request.shape.self_rank = static_cast<int64_t>(left_sizes.size());
  request.shape.other_rank = static_cast<int64_t>(right_sizes.size());
  request.shape.output_rank = 4;
  request.shape.cat_dim = dim;
  request.shape.self_numel = numel_or_zero(left_sizes);
  if (left_sizes.size() == 1 && right_sizes.size() == 4) {
    request.shape.batch = right_sizes[0];
    request.shape.heads = right_sizes[1];
    request.shape.right_sequence = right_sizes[2];
    request.shape.output_sequence = right_sizes[2];
    request.shape.head_dim = right_sizes[3];
  }
  request.contract_metadata = contract_metadata;
  request.input_direct_buffer = left_is_vulkan;
  request.weight_direct_buffer = right_buffer_storage;
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

DynamicProgramRequest make_embedding_lookup_direct_buffer_dynamic_program(
    const IntArrayRef weight_sizes,
    const IntArrayRef indices_sizes,
    const ScalarType weight_dtype,
    const ScalarType indices_dtype,
    const bool weight_direct_buffer,
    const bool indices_direct_buffer,
    const bool output_direct_buffer,
    const bool index_bounds_proven,
    const bool padding_idx_has_hint,
    const bool scale_grad_by_freq,
    const bool sparse,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request;
  request.semantic_family =
      DynamicProgramSemanticFamily::EmbeddingLookupDirectBuffer;
  request.dtype = weight_dtype;
  request.other_dtype = indices_dtype;
  request.output_dtype = weight_dtype;
  request.rank = static_cast<int64_t>(weight_sizes.size());
  if (weight_sizes.size() == 2) {
    request.shape.num_embeddings = weight_sizes[0];
    request.shape.embedding_dim = weight_sizes[1];
  }
  request.shape.index_rank = static_cast<int64_t>(indices_sizes.size());
  request.shape.num_indices = numel_or_zero(indices_sizes);
  request.contract_metadata = contract_metadata;
  request.input_direct_buffer = weight_direct_buffer;
  request.weight_direct_buffer = indices_direct_buffer;
  request.output_direct_buffer = output_direct_buffer;
  request.index_bounds_proven = index_bounds_proven;
  request.padding_idx_has_hint = padding_idx_has_hint;
  request.scale_grad_by_freq = scale_grad_by_freq;
  request.sparse = sparse;
  request.behavior_enabled = behavior_enabled;
  return request;
}

DynamicProgramRequest make_feature_map_to_tokens_direct_buffer_dynamic_program(
    const IntArrayRef input_sizes,
    const ScalarType dtype,
    const bool input_direct_buffer,
    const bool output_direct_buffer,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request;
  request.semantic_family =
      DynamicProgramSemanticFamily::FeatureMapToTokensDirectBuffer;
  request.dtype = dtype;
  request.other_dtype = dtype;
  request.output_dtype = dtype;
  request.rank = static_cast<int64_t>(input_sizes.size());
  if (input_sizes.size() == 4) {
    request.shape.batch = input_sizes[0];
    request.shape.input_channels = input_sizes[1];
    request.shape.height = input_sizes[2];
    request.shape.width = input_sizes[3];
    request.shape.output_sequence = input_sizes[2] * input_sizes[3];
    request.shape.output_rank = 3;
    request.shape.output_numel =
        input_sizes[0] * input_sizes[1] * input_sizes[2] * input_sizes[3];
  }
  request.capabilities.has_pipeline_cache = true;
  request.contract_metadata = contract_metadata;
  request.input_direct_buffer = input_direct_buffer;
  request.output_direct_buffer = output_direct_buffer;
  request.behavior_enabled = behavior_enabled;
  return request;
}

DynamicProgramRequest make_cat_axis_direct_buffer_dynamic_program(
    const ArrayRef<ChannelCatTensorInfo> tensors,
    const int64_t dim,
    const ScalarType output_dtype,
    const bool output_direct_buffer,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request;
  request.semantic_family = DynamicProgramSemanticFamily::CatAxisDirectBuffer;
  request.output_dtype = output_dtype;
  request.rank = tensors.empty() ? 0 : tensors[0].rank;
  request.dtype = tensors.empty() ? output_dtype : tensors[0].dtype;
  request.other_dtype = request.dtype;
  request.shape.self_rank = request.rank;
  request.shape.output_rank = request.rank;
  request.shape.cat_dim = dim;
  request.shape.input_count = static_cast<int64_t>(tensors.size());
  request.input_direct_buffer = !tensors.empty();
  for (const ChannelCatTensorInfo& tensor : tensors) {
    request.input_direct_buffer = request.input_direct_buffer &&
        tensor.is_vulkan && tensor.has_buffer_storage &&
        tensor.supports_buffer_compute && tensor.dtype == request.dtype;
  }

  if (!tensors.empty() && tensors[0].rank == 4 && dim == 1) {
    const ChannelCatTensorInfo& reference = tensors[0];
    request.shape.batch = reference.batch;
    request.shape.height = reference.height;
    request.shape.width = reference.width;
    bool same_non_cat_dims = true;
    int64_t total_channels = 0;
    for (const ChannelCatTensorInfo& tensor : tensors) {
      same_non_cat_dims = same_non_cat_dims && tensor.rank == 4 &&
          tensor.batch == reference.batch && tensor.height == reference.height &&
          tensor.width == reference.width;
      total_channels += tensor.channels;
    }
    if (same_non_cat_dims) {
      request.shape.total_cat_dim = total_channels;
      request.shape.output_channels = total_channels;
      request.shape.output_numel =
          reference.batch * total_channels * reference.height *
          reference.width;
    }
  }

  request.capabilities.has_pipeline_cache = true;
  request.contract_metadata = contract_metadata;
  request.output_direct_buffer = output_direct_buffer;
  request.behavior_enabled = behavior_enabled;
  return request;
}

DynamicProgramRequest make_batch_norm_inference_direct_buffer_dynamic_program(
    const BatchNormInferenceTensorInfo& input,
    const BatchNormInferenceTensorInfo& weight,
    const BatchNormInferenceTensorInfo& bias,
    const BatchNormInferenceTensorInfo& running_mean,
    const BatchNormInferenceTensorInfo& running_var,
    const bool training,
    const bool output_direct_buffer,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request;
  request.semantic_family =
      DynamicProgramSemanticFamily::BatchNormInferenceDirectBuffer;
  request.dtype = input.dtype;
  request.output_dtype = input.dtype;
  request.rank = input.dim;
  request.shape.self_rank = input.dim;
  request.shape.output_rank = input.dim;
  request.shape.batch = input.batch;
  request.shape.input_channels = input.channels;
  request.shape.output_channels =
      input.channels > 0 && running_mean.numel == input.channels &&
          running_var.numel == input.channels &&
          (!weight.has_value || weight.numel == input.channels) &&
          (!bias.has_value || bias.numel == input.channels)
      ? input.channels
      : 0;
  request.shape.height = input.height;
  request.shape.width = input.width;
  request.contract_metadata = contract_metadata;
  request.input_direct_buffer =
      input.is_vulkan && input.has_buffer_storage &&
      input.supports_buffer_compute;
  request.weight_direct_buffer =
      running_mean.has_value && running_mean.defined &&
      running_mean.has_buffer_storage && running_var.has_value &&
      running_var.defined && running_var.has_buffer_storage &&
      (!weight.has_value || weight.has_buffer_storage) &&
      (!bias.has_value || bias.has_buffer_storage);
  request.output_direct_buffer = output_direct_buffer;
  request.training = training;
  request.behavior_enabled = behavior_enabled;
  return request;
}

DynamicProgramRequest make_gqa_repeat_direct_buffer_dynamic_program(
    const IntArrayRef tensor_sizes,
    const ScalarType dtype,
    const bool input_direct_buffer,
    const int64_t repeat_factor,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request;
  request.semantic_family = DynamicProgramSemanticFamily::GQARepeatDirectBuffer;
  request.dtype = dtype;
  request.output_dtype = dtype;
  request.rank = static_cast<int64_t>(tensor_sizes.size());
  if (tensor_sizes.size() == 4) {
    request.shape.batch = tensor_sizes[0];
    request.shape.heads = tensor_sizes[1];
    request.shape.left_sequence = tensor_sizes[2];
    request.shape.head_dim = tensor_sizes[3];
    request.shape.repeat_factor = repeat_factor;
    request.shape.output_channels = tensor_sizes[1] * repeat_factor;
    request.shape.output_sequence = tensor_sizes[2];
    request.shape.output_numel =
        tensor_sizes[0] * tensor_sizes[1] * repeat_factor *
        tensor_sizes[2] * tensor_sizes[3];
  }
  request.capabilities.has_pipeline_cache = true;
  request.contract_metadata = contract_metadata;
  request.input_direct_buffer = input_direct_buffer;
  request.output_direct_buffer = true;
  request.behavior_enabled = behavior_enabled;
  return request;
}

DynamicProgramRequest make_direct_decode_gqa_sdpa_direct_buffer_dynamic_program(
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes,
    const IntArrayRef value_sizes,
    const ScalarType query_dtype,
    const ScalarType key_dtype,
    const ScalarType value_dtype,
    const bool query_direct_buffer,
    const bool key_direct_buffer,
    const bool value_direct_buffer,
    const bool has_attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request;
  request.semantic_family =
      DynamicProgramSemanticFamily::DirectDecodeGQASDPADirectBuffer;
  request.dtype = query_dtype;
  request.other_dtype = key_dtype;
  request.output_dtype = value_dtype;
  request.rank = static_cast<int64_t>(query_sizes.size());
  if (
      query_sizes.size() == 4 && key_sizes.size() == 4 &&
      value_sizes.size() == 4) {
    request.shape.batch = query_sizes[0];
    request.shape.query_heads = query_sizes[1];
    request.shape.key_value_heads = key_sizes[1];
    request.shape.query_sequence = query_sizes[2];
    request.shape.key_value_sequence = key_sizes[2];
    request.shape.head_dim = query_sizes[3];
    request.shape.value_dim = value_sizes[3];
  }
  request.contract_metadata = contract_metadata;
  request.input_direct_buffer = query_direct_buffer;
  request.weight_direct_buffer = key_direct_buffer;
  request.output_direct_buffer = value_direct_buffer;
  request.has_attn_mask = has_attn_mask;
  request.dropout_is_zero = dropout_p == 0.0;
  request.is_causal = is_causal;
  request.enable_gqa = enable_gqa;
  request.scale_is_default_or_head_dim =
      !scale.has_value() ||
      (query_sizes.size() == 4 && query_sizes[3] > 0 &&
       std::abs(
           *scale -
           (1.0 / std::sqrt(static_cast<double>(query_sizes[3])))) <= 1.0e-6);
  request.behavior_enabled = behavior_enabled;
  return request;
}

DynamicProgramRequest
make_direct_causal_prefill_gqa_sdpa_direct_buffer_dynamic_program(
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes,
    const IntArrayRef value_sizes,
    const ScalarType query_dtype,
    const ScalarType key_dtype,
    const ScalarType value_dtype,
    const bool query_direct_buffer,
    const bool key_direct_buffer,
    const bool value_direct_buffer,
    const bool has_attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request =
      make_direct_decode_gqa_sdpa_direct_buffer_dynamic_program(
          query_sizes,
          key_sizes,
          value_sizes,
          query_dtype,
          key_dtype,
          value_dtype,
          query_direct_buffer,
          key_direct_buffer,
          value_direct_buffer,
          has_attn_mask,
          dropout_p,
          is_causal,
          scale,
          enable_gqa,
          contract_metadata,
          behavior_enabled);
  request.semantic_family =
      DynamicProgramSemanticFamily::DirectCausalPrefillGQASDPADirectBuffer;
  return request;
}

DynamicProgramRequest
make_small_non_causal_gqa_sdpa_direct_buffer_dynamic_program(
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes,
    const IntArrayRef value_sizes,
    const ScalarType query_dtype,
    const ScalarType key_dtype,
    const ScalarType value_dtype,
    const bool query_direct_buffer,
    const bool key_direct_buffer,
    const bool value_direct_buffer,
    const bool has_attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request =
      make_direct_decode_gqa_sdpa_direct_buffer_dynamic_program(
          query_sizes,
          key_sizes,
          value_sizes,
          query_dtype,
          key_dtype,
          value_dtype,
          query_direct_buffer,
          key_direct_buffer,
          value_direct_buffer,
          has_attn_mask,
          dropout_p,
          is_causal,
          scale,
          enable_gqa,
          contract_metadata,
          behavior_enabled);
  request.semantic_family =
      DynamicProgramSemanticFamily::SmallNonCausalGQASDPADirectBuffer;
  return request;
}

DynamicProgramRequest
make_direct_non_causal_mha_sdpa_direct_buffer_dynamic_program(
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes,
    const IntArrayRef value_sizes,
    const ScalarType query_dtype,
    const ScalarType key_dtype,
    const ScalarType value_dtype,
    const bool query_direct_buffer,
    const bool key_direct_buffer,
    const bool value_direct_buffer,
    const bool has_attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request =
      make_direct_decode_gqa_sdpa_direct_buffer_dynamic_program(
          query_sizes,
          key_sizes,
          value_sizes,
          query_dtype,
          key_dtype,
          value_dtype,
          query_direct_buffer,
          key_direct_buffer,
          value_direct_buffer,
          has_attn_mask,
          dropout_p,
          is_causal,
          scale,
          enable_gqa,
          contract_metadata,
          behavior_enabled);
  request.semantic_family =
      DynamicProgramSemanticFamily::DirectNonCausalMHASDPADirectBuffer;
  return request;
}

DynamicProgramRequest make_token_prefix_cat_add_direct_buffer_dynamic_program(
    const IntArrayRef prefix_sizes,
    const IntArrayRef token_sizes,
    const IntArrayRef pos_sizes,
    const ScalarType prefix_dtype,
    const ScalarType token_dtype,
    const ScalarType pos_dtype,
    const bool prefix_buffer_storage,
    const bool token_buffer_storage,
    const bool pos_buffer_storage,
    const int64_t dim,
    const bool inplace,
    const bool alias_output,
    const ExecutionContractMetadata* const contract_metadata,
    const bool behavior_enabled) {
  DynamicProgramRequest request;
  request.semantic_family =
      DynamicProgramSemanticFamily::TokenPrefixCatAddDirectBuffer;
  request.dtype = prefix_dtype;
  request.other_dtype = token_dtype;
  request.output_dtype = pos_dtype;
  request.rank = static_cast<int64_t>(prefix_sizes.size());
  request.shape.self_rank = static_cast<int64_t>(prefix_sizes.size());
  request.shape.other_rank = static_cast<int64_t>(token_sizes.size());
  request.shape.output_rank = static_cast<int64_t>(pos_sizes.size());
  request.shape.cat_dim = dim;
  if (
      prefix_sizes.size() == 3 && token_sizes.size() == 3 &&
      pos_sizes.size() == 3) {
    request.shape.batch =
        prefix_sizes[0] == token_sizes[0] && prefix_sizes[0] == pos_sizes[0]
        ? prefix_sizes[0]
        : 0;
    request.shape.left_sequence = prefix_sizes[1];
    request.shape.right_sequence = token_sizes[1];
    request.shape.output_sequence = pos_sizes[1];
    request.shape.input_channels =
        prefix_sizes[2] == token_sizes[2] && prefix_sizes[2] == pos_sizes[2]
        ? prefix_sizes[2]
        : 0;
  }
  request.contract_metadata = contract_metadata;
  request.input_direct_buffer = prefix_buffer_storage;
  request.weight_direct_buffer = token_buffer_storage;
  request.output_direct_buffer = pos_buffer_storage;
  request.inplace = inplace;
  request.has_output = alias_output;
  request.behavior_enabled = behavior_enabled;
  return request;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
