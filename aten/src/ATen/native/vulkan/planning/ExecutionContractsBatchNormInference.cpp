#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsBatchNormInferenceMaterializedSpec.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsBatchNormInferenceSpec.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

constexpr ExecutionContractMetadata make_execution_contract_metadata(
    const char* contract_name,
    const char* family_name,
    const char* tuple_id,
    const char* evidence_id,
    const char* guard_id,
    const char* fallback_policy,
    const char* materialization_policy) {
  return ExecutionContractMetadata{
      contract_name,
      family_name,
      tuple_id,
      evidence_id,
      guard_id,
      fallback_policy,
      materialization_policy};
}

constexpr ExecutionContractMetadata kBatchNormInferenceBufferFloat4DMetadata =
    make_execution_contract_metadata(
        generated::kBatchNormInferenceBufferFloat4DSpec.contract_name,
        generated::kBatchNormInferenceBufferFloat4DSpec.family_name,
        generated::kBatchNormInferenceBufferFloat4DSpec.tuple_id,
        generated::kBatchNormInferenceBufferFloat4DSpec.evidence_id,
        generated::kBatchNormInferenceBufferFloat4DSpec.guard_id,
        generated::kBatchNormInferenceBufferFloat4DSpec.fallback_policy,
        generated::kBatchNormInferenceBufferFloat4DSpec.materialization_policy);
constexpr ExecutionContractMetadata
    kBatchNormInferenceMaterializedBufferFloat4DMetadata =
        make_execution_contract_metadata(
            generated::kBatchNormInferenceMaterializedBufferFloat4DSpec
                .contract_name,
            generated::kBatchNormInferenceMaterializedBufferFloat4DSpec
                .family_name,
            generated::kBatchNormInferenceMaterializedBufferFloat4DSpec
                .tuple_id,
            generated::kBatchNormInferenceMaterializedBufferFloat4DSpec
                .evidence_id,
            generated::kBatchNormInferenceMaterializedBufferFloat4DSpec
                .guard_id,
            generated::kBatchNormInferenceMaterializedBufferFloat4DSpec
                .fallback_policy,
            generated::kBatchNormInferenceMaterializedBufferFloat4DSpec
                .materialization_policy);

bool batch_norm_float_1d_matches(
    const BatchNormInferenceTensorInfo& tensor,
    const int64_t num_features) {
  return tensor.has_value && tensor.defined && tensor.is_vulkan &&
      tensor.dtype == kFloat && tensor.dim == 1 &&
      tensor.numel == num_features && tensor.is_contiguous;
}

bool batch_norm_float_1d_buffer_matches(
    const BatchNormInferenceTensorInfo& tensor,
    const int64_t num_features) {
  return batch_norm_float_1d_matches(tensor, num_features) &&
      tensor.has_buffer_storage;
}

bool batch_norm_float_1d_materializable_matches(
    const BatchNormInferenceTensorInfo& tensor,
    const int64_t num_features) {
  return batch_norm_float_1d_matches(tensor, num_features) &&
      tensor.supports_buffer_compute;
}

bool batch_norm_optional_float_1d_matches(
    const BatchNormInferenceTensorInfo& tensor,
    const int64_t num_features) {
  return !tensor.has_value ||
      batch_norm_float_1d_matches(tensor, num_features);
}

bool batch_norm_optional_float_1d_materializable_matches(
    const BatchNormInferenceTensorInfo& tensor,
    const int64_t num_features) {
  return !tensor.has_value ||
      batch_norm_float_1d_materializable_matches(tensor, num_features);
}

bool batch_norm_effective_affine_has_buffer_storage(
    const BatchNormInferenceTensorInfo& tensor,
    const BatchNormInferenceTensorInfo& running_mean) {
  return tensor.has_value ? tensor.has_buffer_storage
                          : running_mean.has_buffer_storage;
}

bool batch_norm_effective_affine_supports_buffer_compute(
    const BatchNormInferenceTensorInfo& tensor,
    const BatchNormInferenceTensorInfo& running_mean) {
  return tensor.has_value ? tensor.supports_buffer_compute
                          : running_mean.supports_buffer_compute;
}

} // namespace

const char* batch_norm_inference_family_name(
    const BatchNormInferenceFamily family) {
  switch (family) {
    case BatchNormInferenceFamily::BufferFloat4D:
      return "BatchNormInferenceBufferFloat4D";
    case BatchNormInferenceFamily::MaterializedBufferFloat4D:
      return "BatchNormInferenceMaterializedBufferFloat4D";
    case BatchNormInferenceFamily::None:
      return "BatchNormInferenceNone";
  }
  return "BatchNormInferenceNone";
}

BatchNormInferenceMatch match_batch_norm_inference_contract(
    const BatchNormInferenceTensorInfo& input,
    const BatchNormInferenceTensorInfo& weight,
    const BatchNormInferenceTensorInfo& bias,
    const BatchNormInferenceTensorInfo& running_mean,
    const BatchNormInferenceTensorInfo& running_var,
    const bool training) {
  BatchNormInferenceMatch result;
  const auto& buffer_spec =
      generated::kBatchNormInferenceBufferFloat4DSpec;
  const auto& materialized_spec =
      generated::kBatchNormInferenceMaterializedBufferFloat4DSpec;
  if (
      !generated::batch_norm_inference_buffer_float_4_d_options_match(
          buffer_spec,
          input.dtype,
          buffer_spec.parameter_dtype,
          input.dim,
          buffer_spec.parameter_rank,
          training,
          buffer_spec.weight_optional,
          buffer_spec.bias_optional,
          buffer_spec.requires_vulkan,
          buffer_spec.requires_contiguous,
          buffer_spec.requires_buffer_storage,
          buffer_spec.requires_buffer_compute) ||
      !input.has_value || !input.defined ||
      (buffer_spec.requires_vulkan && !input.is_vulkan) ||
      (buffer_spec.requires_contiguous && !input.is_contiguous) ||
      (buffer_spec.requires_buffer_compute &&
       !input.supports_buffer_compute)) {
    return result;
  }

  const int64_t num_features = input.channels;
  const bool buffer_match =
      input.has_buffer_storage &&
      batch_norm_float_1d_buffer_matches(running_mean, num_features) &&
      batch_norm_float_1d_buffer_matches(running_var, num_features) &&
      batch_norm_optional_float_1d_matches(weight, num_features) &&
      batch_norm_optional_float_1d_matches(bias, num_features) &&
      batch_norm_effective_affine_has_buffer_storage(weight, running_mean) &&
      batch_norm_effective_affine_has_buffer_storage(bias, running_mean);
  if (buffer_match) {
    result.matched = true;
    result.family = BatchNormInferenceFamily::BufferFloat4D;
    result.tuple_id = buffer_spec.tuple_id;
    result.metadata = &kBatchNormInferenceBufferFloat4DMetadata;
    return result;
  }

  if (
      !generated::batch_norm_inference_materialized_buffer_float_4_d_options_match(
          materialized_spec,
          input.dtype,
          materialized_spec.parameter_dtype,
          input.dim,
          materialized_spec.parameter_rank,
          training,
          materialized_spec.weight_optional,
          materialized_spec.bias_optional,
          materialized_spec.requires_vulkan,
          materialized_spec.requires_contiguous,
          materialized_spec.requires_buffer_storage,
          materialized_spec.requires_buffer_compute,
          materialized_spec.requires_materialization) ||
      !batch_norm_float_1d_materializable_matches(
          running_mean, num_features) ||
      !batch_norm_float_1d_materializable_matches(running_var, num_features) ||
      !batch_norm_optional_float_1d_materializable_matches(
          weight, num_features) ||
      !batch_norm_optional_float_1d_materializable_matches(
          bias, num_features) ||
      !batch_norm_effective_affine_supports_buffer_compute(
          weight, running_mean) ||
      !batch_norm_effective_affine_supports_buffer_compute(
          bias, running_mean)) {
    return result;
  }

  result.matched = true;
  result.family = BatchNormInferenceFamily::MaterializedBufferFloat4D;
  result.tuple_id = materialized_spec.tuple_id;
  result.metadata = &kBatchNormInferenceMaterializedBufferFloat4DMetadata;
  result.requires_materialization = materialized_spec.requires_materialization;
  return result;
}

bool matches_batch_norm_inference_contract(
    const BatchNormInferenceTensorInfo& input,
    const BatchNormInferenceTensorInfo& weight,
    const BatchNormInferenceTensorInfo& bias,
    const BatchNormInferenceTensorInfo& running_mean,
    const BatchNormInferenceTensorInfo& running_var,
    const bool training) {
  return match_batch_norm_inference_contract(
             input, weight, bias, running_mean, running_var, training)
      .matched;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
