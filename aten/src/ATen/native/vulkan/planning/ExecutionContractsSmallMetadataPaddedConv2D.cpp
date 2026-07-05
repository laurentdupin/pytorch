#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsSmallMetadataPaddedConv2DSpec.h>

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

constexpr ExecutionContractMetadata
    kSmallMetadataPaddedConv2DMaterializedBufferInput2x2Metadata =
        make_execution_contract_metadata(
            generated::kSmallMetadataPaddedConv2DMaterializedBufferInput2x2Spec
                .contract_name,
            generated::kSmallMetadataPaddedConv2DMaterializedBufferInput2x2Spec
                .family_name,
            generated::kSmallMetadataPaddedConv2DMaterializedBufferInput2x2Spec
                .tuple_id,
            generated::kSmallMetadataPaddedConv2DMaterializedBufferInput2x2Spec
                .evidence_id,
            generated::kSmallMetadataPaddedConv2DMaterializedBufferInput2x2Spec
                .guard_id,
            generated::kSmallMetadataPaddedConv2DMaterializedBufferInput2x2Spec
                .fallback_policy,
            generated::kSmallMetadataPaddedConv2DMaterializedBufferInput2x2Spec
                .materialization_policy);
constexpr ExecutionContractMetadata
    kSmallMetadataPaddedConv2DRuntimeMaterializedBufferInput2x2Metadata =
        make_execution_contract_metadata(
            "SmallMetadataPaddedConv2DContract",
            "RuntimeMaterializedBufferInput2x2",
            "small_metadata_padded_conv2d_runtime_materialized_2x2",
            "small_metadata_padded_conv2d_dynamic_random_shape_tests",
            "small_metadata_padded_conv2d_semantic_layout_guards",
            "unsupported_semantics_keep_legacy_image_pack",
            "materialize_small_metadata_input_then_conv2d_buffer_float");

} // namespace

const char* small_metadata_padded_conv2d_family_name(
    const SmallMetadataPaddedConv2DFamily family) {
  switch (family) {
    case SmallMetadataPaddedConv2DFamily::MaterializedBufferInput2x2:
      return "SmallMetadataPaddedConv2DMaterializedBufferInput2x2";
    case SmallMetadataPaddedConv2DFamily::RuntimeMaterializedBufferInput2x2:
      return "SmallMetadataPaddedConv2DRuntimeMaterializedBufferInput2x2";
    case SmallMetadataPaddedConv2DFamily::None:
      return "SmallMetadataPaddedConv2DNone";
  }
  return "SmallMetadataPaddedConv2DNone";
}

SmallMetadataPaddedConv2DMatch match_small_metadata_padded_conv2d_contract(
    const SmallMetadataPaddedConv2DTensorInfo& input,
    const SmallMetadataPaddedConv2DWeightInfo& weight,
    const SmallMetadataPaddedConv2DOptions& options) {
  SmallMetadataPaddedConv2DMatch result;
  if (
      input.dtype != kFloat || weight.dtype != kFloat || input.rank != 4 ||
      weight.rank != 4 || input.batch != 1 || input.channels <= 1 ||
      input.channels >= 20 || input.height <= 0 || input.width <= 0 ||
      weight.output_channels <= 0 || weight.input_channels != input.channels ||
      weight.kernel_h != 2 || weight.kernel_w != 2 || options.groups != 1 ||
      options.stride_h != 1 || options.stride_w != 1 ||
      options.padding_h != 0 || options.padding_w != 0 ||
      options.dilation_h != 1 || options.dilation_w != 1 ||
      options.transposed || options.quantized ||
      !options.output_padding_is_zero || !input.is_vulkan ||
      !input.has_buffer_storage || !input.is_width_packed ||
      input.has_direct_buffer_layout || !input.supports_buffer_compute ||
      !weight.defined) {
    return result;
  }
  const int64_t output_h = input.height - weight.kernel_h + 1;
  const int64_t output_w = input.width - weight.kernel_w + 1;
  if (output_h <= 0 || output_w <= 0) {
    return result;
  }

  result.matched = true;
  result.family =
      SmallMetadataPaddedConv2DFamily::RuntimeMaterializedBufferInput2x2;
  result.tuple_id =
      kSmallMetadataPaddedConv2DRuntimeMaterializedBufferInput2x2Metadata
          .tuple_id;
  result.metadata =
      &kSmallMetadataPaddedConv2DRuntimeMaterializedBufferInput2x2Metadata;
  result.requires_input_materialization = true;
  return result;
}

bool matches_small_metadata_padded_conv2d_contract(
    const SmallMetadataPaddedConv2DTensorInfo& input,
    const SmallMetadataPaddedConv2DWeightInfo& weight,
    const SmallMetadataPaddedConv2DOptions& options) {
  return match_small_metadata_padded_conv2d_contract(input, weight, options)
      .matched;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
