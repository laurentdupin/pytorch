#pragma once

#ifdef USE_VULKAN_API

#include <ATen/ArrayRef.h>
#include <ATen/core/ScalarType.h>
#include <ATen/native/vulkan/planning/ExecutionContracts.h>

#include <cstdint>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

enum class DynamicProgramSemanticFamily : uint8_t {
  None = 0u,
  PointwiseConv1x1DirectBuffer,
  Conv2DDirectBuffer,
  SequenceCatDirectBuffer,
  ElementwiseBroadcastDirectBuffer,
  LinearOrMatmulDirectBuffer,
  EmbeddingLookupDirectBuffer,
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

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
