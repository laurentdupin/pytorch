#include <ATen/native/vulkan/planning/ExecutionContracts.h>

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

constexpr const char* kFallbackUnsupportedShapesDoNotMatch =
    "unsupported_shapes_do_not_match";
constexpr const char* kMaterializationBatchNormInferenceBuffer =
    "batch_norm_inference_buffer_kernel";
constexpr const char* kMaterializationBatchNormInferenceMaterializedBuffer =
    "materialize_to_buffer_then_batch_norm_inference_buffer_kernel";

constexpr const char* kBatchNormInferenceBufferFloat4DTupleId =
    "buffer_inference_4d_float";
constexpr ExecutionContractMetadata kBatchNormInferenceBufferFloat4DMetadata =
    make_execution_contract_metadata(
        "BatchNormInferenceContract",
        "BufferFloat4D",
        kBatchNormInferenceBufferFloat4DTupleId,
        "batch_norm_inference_focused_tests",
        "batch_norm_inference_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationBatchNormInferenceBuffer);
constexpr const char* kBatchNormInferenceMaterializedBufferFloat4DTupleId =
    "materialized_buffer_inference_4d_float";
constexpr ExecutionContractMetadata
    kBatchNormInferenceMaterializedBufferFloat4DMetadata =
        make_execution_contract_metadata(
            "BatchNormInferenceContract",
            "MaterializedBufferFloat4D",
            kBatchNormInferenceMaterializedBufferFloat4DTupleId,
            "batch_norm_inference_materialized_buffer_focused_tests",
            "batch_norm_inference_adjacent_guards",
            kFallbackUnsupportedShapesDoNotMatch,
            kMaterializationBatchNormInferenceMaterializedBuffer);

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
  if (
      training || !input.has_value || !input.defined || !input.is_vulkan ||
      input.dtype != kFloat || input.dim != 4 || !input.is_contiguous ||
      !input.supports_buffer_compute) {
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
    result.tuple_id = kBatchNormInferenceBufferFloat4DTupleId;
    result.metadata = &kBatchNormInferenceBufferFloat4DMetadata;
    return result;
  }

  if (
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
  result.tuple_id = kBatchNormInferenceMaterializedBufferFloat4DTupleId;
  result.metadata = &kBatchNormInferenceMaterializedBufferFloat4DMetadata;
  result.requires_materialization = true;
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
