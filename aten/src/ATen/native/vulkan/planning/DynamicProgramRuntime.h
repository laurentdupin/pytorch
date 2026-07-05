#pragma once

#ifdef USE_VULKAN_API

#include <ATen/ArrayRef.h>
#include <ATen/core/ScalarType.h>
#include <ATen/native/vulkan/planning/ExecutionContracts.h>

#include <cstdint>
#include <optional>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

enum class DynamicProgramSemanticFamily : uint8_t {
  None = 0u,
  PointwiseConv1x1DirectBuffer,
  Conv2DDirectBuffer,
  PackedBufferConv2D,
  PatchEmbedFloatBufferConvRoute,
  SequenceCatDirectBuffer,
  InitialSequenceCatDirectBuffer,
  ElementwiseBroadcastDirectBuffer,
  LinearOrMatmulDirectBuffer,
  EmbeddingLookupDirectBuffer,
  FeatureMapToTokensDirectBuffer,
  CatAxisDirectBuffer,
  BatchNormInferenceDirectBuffer,
  GQARepeatDirectBuffer,
  DirectDecodeGQASDPADirectBuffer,
  SmallNonCausalGQASDPADirectBuffer,
  DirectNonCausalMHASDPADirectBuffer,
  DirectCausalPrefillGQASDPADirectBuffer,
  TokenPrefixCatAddDirectBuffer,
  StackRegionCommandReplay,
};

enum class DynamicProgramShaderSelectionPolicy : uint8_t {
  None = 0u,
  ExistingStaticShader,
  RuntimeSpecializedShader,
  RuntimeGeneratedShader,
  CachedCompiledPipeline,
};

enum class DynamicProgramCommandPlanKind : uint8_t {
  None = 0u,
  SingleDispatch,
  MultiDispatch,
  CustomCommandList,
  RegionCommandList,
};

enum class DynamicProgramCachePolicy : uint8_t {
  None = 0u,
  EvidenceOnly,
  ProgramKeyLocal,
  CapabilityProfileProgramKey,
  PersistentPipelineCache,
};

enum class DynamicProgramRejectReason : uint8_t {
  None = 0u,
  MissingContract,
  IncompleteProgramKey,
  UnsupportedSemanticFamily,
  UnsupportedDType,
  UnsupportedRank,
  UnsupportedLayout,
  UnsupportedKernelSemantics,
  MissingPipelinePolicy,
  MissingCommandPlan,
  RuntimeCompilationUnavailable,
  BehaviorDisabled,
  MissingIndexBoundsProof,
};

struct DynamicProgramShapeDesc final {
  int64_t batch{0};
  int64_t input_channels{0};
  int64_t weight_input_channels{0};
  int64_t output_channels{0};
  int64_t height{0};
  int64_t width{0};
  int64_t kernel_h{0};
  int64_t kernel_w{0};
  int64_t stride_h{0};
  int64_t stride_w{0};
  int64_t padding_h{0};
  int64_t padding_w{0};
  int64_t dilation_h{0};
  int64_t dilation_w{0};
  int64_t groups{0};
  int64_t self_rank{0};
  int64_t other_rank{0};
  int64_t output_rank{0};
  int64_t self_numel{0};
  int64_t other_numel{0};
  int64_t output_numel{0};
  int64_t cat_dim{0};
  int64_t left_sequence{0};
  int64_t right_sequence{0};
  int64_t output_sequence{0};
  int64_t heads{0};
  int64_t head_dim{0};
  int64_t m{0};
  int64_t k{0};
  int64_t n{0};
  int64_t rhs_k{0};
  int64_t lhs_rank{0};
  int64_t rhs_rank{0};
  int64_t num_embeddings{0};
  int64_t embedding_dim{0};
  int64_t num_indices{0};
  int64_t index_rank{0};
  int64_t input_count{0};
  int64_t total_cat_dim{0};
  int64_t repeat_factor{0};
  int64_t query_heads{0};
  int64_t key_value_heads{0};
  int64_t query_sequence{0};
  int64_t key_value_sequence{0};
  int64_t value_dim{0};
};

struct DynamicProgramCapabilityDesc final {
  bool has_runtime_shader_compile{false};
  bool has_pipeline_cache{false};
  bool has_custom_command_list{false};
};

struct DynamicProgramRequest final {
  DynamicProgramSemanticFamily semantic_family{
      DynamicProgramSemanticFamily::None};
  ScalarType dtype{kFloat};
  ScalarType other_dtype{kFloat};
  ScalarType output_dtype{kFloat};
  int64_t rank{0};
  DynamicProgramShapeDesc shape{};
  DynamicProgramCapabilityDesc capabilities{};
  const ExecutionContractMetadata* contract_metadata{nullptr};
  bool input_direct_buffer{false};
  bool weight_direct_buffer{false};
  bool output_direct_buffer{false};
  bool input_buffer_storage{false};
  bool weight_buffer_storage{false};
  bool output_buffer_storage{false};
  bool has_bias{false};
  ElementwiseBroadcastOp elementwise_op{ElementwiseBroadcastOp::Unsupported};
  bool alpha_is_one{false};
  bool has_output{false};
  bool inplace{false};
  bool broadcast_compatible{false};
  bool post_op_none{true};
  bool index_bounds_proven{false};
  bool padding_idx_has_hint{false};
  bool scale_grad_by_freq{false};
  bool sparse{false};
  bool training{false};
  bool has_attn_mask{false};
  bool is_causal{false};
  bool enable_gqa{false};
  bool dropout_is_zero{true};
  bool scale_is_default_or_head_dim{false};
  bool behavior_enabled{false};
};

struct DynamicProgramKey final {
  const char* schema{"DynamicProgramRuntime.v0"};
  DynamicProgramSemanticFamily semantic_family{
      DynamicProgramSemanticFamily::None};
  ScalarType dtype{kFloat};
  ScalarType other_dtype{kFloat};
  ScalarType output_dtype{kFloat};
  int64_t rank{0};
  DynamicProgramShapeDesc shape{};
  const char* contract_name{nullptr};
  const char* contract_family{nullptr};
  const char* contract_tuple_id{nullptr};
};

struct DynamicProgramCommandPlan final {
  DynamicProgramShaderSelectionPolicy shader_policy{
      DynamicProgramShaderSelectionPolicy::None};
  DynamicProgramCommandPlanKind command_plan{
      DynamicProgramCommandPlanKind::None};
  DynamicProgramCachePolicy cache_policy{DynamicProgramCachePolicy::None};
  const char* shader_family{nullptr};
  const char* command_list_label{nullptr};
  bool requires_runtime_shader_compile{false};
  bool requires_custom_command_list{false};
};

struct DynamicProgramDecision final {
  DynamicProgramKey key{};
  DynamicProgramCommandPlan command_plan{};
  DynamicProgramRejectReason reject_reason{
      DynamicProgramRejectReason::IncompleteProgramKey};
  const char* status{"dynamic_program_runtime_rejected_incomplete_program_key"};
  bool semantic_validation_passed{false};
  bool program_key_complete{false};
  bool command_plan_available{false};
  bool behavior_enabled{false};
  bool runtime_selection_authorized{false};
};

struct DynamicProgramAdmission final {
  bool accepted{false};
  DynamicProgramRejectReason reject_reason{
      DynamicProgramRejectReason::IncompleteProgramKey};
  const char* status{"dynamic_program_runtime_rejected_incomplete_program_key"};
};

const char* dynamic_program_semantic_family_name(
    DynamicProgramSemanticFamily family);

const char* dynamic_program_shader_selection_policy_name(
    DynamicProgramShaderSelectionPolicy policy);

const char* dynamic_program_command_plan_kind_name(
    DynamicProgramCommandPlanKind kind);

const char* dynamic_program_cache_policy_name(DynamicProgramCachePolicy policy);

const char* dynamic_program_reject_reason_name(
    DynamicProgramRejectReason reason);

DynamicProgramDecision build_dynamic_program_runtime_plan(
    const DynamicProgramRequest& request);

DynamicProgramAdmission admit_dynamic_program(
    const DynamicProgramRequest& request);

DynamicProgramRequest make_pointwise_conv1x1_direct_buffer_program_request(
    IntArrayRef input_sizes,
    IntArrayRef weight_sizes,
    bool has_bias,
    ScalarType dtype);

DynamicProgramRequest make_pointwise_conv1x1_direct_buffer_dynamic_program(
    IntArrayRef input_sizes,
    IntArrayRef weight_sizes,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    ScalarType dtype,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

DynamicProgramRequest make_conv2d_direct_buffer_dynamic_program(
    IntArrayRef input_sizes,
    IntArrayRef weight_sizes,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    ScalarType dtype,
    bool input_direct_buffer,
    bool weight_direct_buffer,
    bool output_direct_buffer,
    bool has_bias,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

DynamicProgramRequest make_packed_buffer_conv2d_dynamic_program(
    IntArrayRef input_sizes,
    IntArrayRef weight_sizes,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    ScalarType dtype,
    bool input_buffer_storage,
    bool weight_buffer_storage,
    bool output_buffer_storage,
    bool has_bias,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

DynamicProgramRequest make_patch_embed_float_buffer_conv_route_dynamic_program(
    IntArrayRef input_sizes,
    IntArrayRef weight_sizes,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    ScalarType dtype,
    bool input_buffer_storage,
    bool weight_buffer_storage,
    bool output_buffer_storage,
    bool has_bias,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

DynamicProgramRequest make_elementwise_broadcast_direct_buffer_dynamic_program(
    IntArrayRef self_sizes,
    IntArrayRef other_sizes,
    ScalarType self_dtype,
    ScalarType other_dtype,
    ScalarType output_dtype,
    bool self_direct_buffer,
    bool other_direct_buffer,
    bool output_direct_buffer,
    ElementwiseBroadcastOp op,
    bool alpha_is_one,
    bool has_output,
    bool inplace,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

DynamicProgramRequest make_sequence_cat_direct_buffer_dynamic_program(
    IntArrayRef left_sizes,
    IntArrayRef right_sizes,
    ScalarType left_dtype,
    ScalarType right_dtype,
    ScalarType output_dtype,
    bool left_direct_buffer,
    bool right_direct_buffer,
    bool output_direct_buffer,
    int64_t dim,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

DynamicProgramRequest make_initial_sequence_cat_direct_buffer_dynamic_program(
    IntArrayRef left_sizes,
    IntArrayRef right_sizes,
    ScalarType left_dtype,
    ScalarType right_dtype,
    ScalarType output_dtype,
    bool left_is_vulkan,
    bool right_buffer_storage,
    bool output_direct_buffer,
    int64_t dim,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

DynamicProgramRequest make_linear_or_matmul_direct_buffer_program_request(
    IntArrayRef input_sizes,
    IntArrayRef weight_sizes,
    bool has_bias,
    ScalarType dtype);

DynamicProgramRequest make_linear_or_matmul_direct_buffer_dynamic_program(
    IntArrayRef input_sizes,
    IntArrayRef weight_sizes,
    bool has_bias,
    ScalarType dtype,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

DynamicProgramRequest make_embedding_lookup_direct_buffer_dynamic_program(
    IntArrayRef weight_sizes,
    IntArrayRef indices_sizes,
    ScalarType weight_dtype,
    ScalarType indices_dtype,
    bool weight_direct_buffer,
    bool indices_direct_buffer,
    bool output_direct_buffer,
    bool index_bounds_proven,
    bool padding_idx_has_hint,
    bool scale_grad_by_freq,
    bool sparse,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

DynamicProgramRequest make_feature_map_to_tokens_direct_buffer_dynamic_program(
    IntArrayRef input_sizes,
    ScalarType dtype,
    bool input_direct_buffer,
    bool output_direct_buffer,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

DynamicProgramRequest make_cat_axis_direct_buffer_dynamic_program(
    ArrayRef<ChannelCatTensorInfo> tensors,
    int64_t dim,
    ScalarType output_dtype,
    bool output_direct_buffer,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

DynamicProgramRequest make_batch_norm_inference_direct_buffer_dynamic_program(
    const BatchNormInferenceTensorInfo& input,
    const BatchNormInferenceTensorInfo& weight,
    const BatchNormInferenceTensorInfo& bias,
    const BatchNormInferenceTensorInfo& running_mean,
    const BatchNormInferenceTensorInfo& running_var,
    bool training,
    bool output_direct_buffer,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

DynamicProgramRequest make_gqa_repeat_direct_buffer_dynamic_program(
    IntArrayRef tensor_sizes,
    ScalarType dtype,
    bool input_direct_buffer,
    int64_t repeat_factor,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

DynamicProgramRequest make_direct_decode_gqa_sdpa_direct_buffer_dynamic_program(
    IntArrayRef query_sizes,
    IntArrayRef key_sizes,
    IntArrayRef value_sizes,
    ScalarType query_dtype,
    ScalarType key_dtype,
    ScalarType value_dtype,
    bool query_direct_buffer,
    bool key_direct_buffer,
    bool value_direct_buffer,
    bool has_attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

DynamicProgramRequest make_small_non_causal_gqa_sdpa_direct_buffer_dynamic_program(
    IntArrayRef query_sizes,
    IntArrayRef key_sizes,
    IntArrayRef value_sizes,
    ScalarType query_dtype,
    ScalarType key_dtype,
    ScalarType value_dtype,
    bool query_direct_buffer,
    bool key_direct_buffer,
    bool value_direct_buffer,
    bool has_attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

DynamicProgramRequest make_direct_non_causal_mha_sdpa_direct_buffer_dynamic_program(
    IntArrayRef query_sizes,
    IntArrayRef key_sizes,
    IntArrayRef value_sizes,
    ScalarType query_dtype,
    ScalarType key_dtype,
    ScalarType value_dtype,
    bool query_direct_buffer,
    bool key_direct_buffer,
    bool value_direct_buffer,
    bool has_attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

DynamicProgramRequest make_direct_causal_prefill_gqa_sdpa_direct_buffer_dynamic_program(
    IntArrayRef query_sizes,
    IntArrayRef key_sizes,
    IntArrayRef value_sizes,
    ScalarType query_dtype,
    ScalarType key_dtype,
    ScalarType value_dtype,
    bool query_direct_buffer,
    bool key_direct_buffer,
    bool value_direct_buffer,
    bool has_attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

DynamicProgramRequest make_token_prefix_cat_add_direct_buffer_dynamic_program(
    IntArrayRef prefix_sizes,
    IntArrayRef token_sizes,
    IntArrayRef pos_sizes,
    ScalarType prefix_dtype,
    ScalarType token_dtype,
    ScalarType pos_dtype,
    bool prefix_buffer_storage,
    bool token_buffer_storage,
    bool pos_buffer_storage,
    int64_t dim,
    bool inplace,
    bool alias_output,
    const ExecutionContractMetadata* contract_metadata,
    bool behavior_enabled = false);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
