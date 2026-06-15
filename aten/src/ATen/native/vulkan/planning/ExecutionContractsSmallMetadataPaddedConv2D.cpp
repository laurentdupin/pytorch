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

} // namespace

const char* small_metadata_padded_conv2d_family_name(
    const SmallMetadataPaddedConv2DFamily family) {
  switch (family) {
    case SmallMetadataPaddedConv2DFamily::MaterializedBufferInput2x2:
      return "SmallMetadataPaddedConv2DMaterializedBufferInput2x2";
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
  const auto& spec =
      generated::kSmallMetadataPaddedConv2DMaterializedBufferInput2x2Spec;
  if (
      !generated::small_metadata_padded_conv_2_d_materialized_buffer_input_2_x_2_options_match(
          spec,
          input.dtype,
          weight.dtype,
          input.rank,
          weight.rank,
          input.batch,
          input.channels,
          input.height,
          input.width,
          weight.output_channels,
          options.groups,
          weight.kernel_h,
          weight.kernel_w,
          options.stride_h,
          options.stride_w,
          options.padding_h,
          options.padding_w,
          options.dilation_h,
          options.dilation_w,
          options.transposed,
          options.quantized,
          options.output_padding_is_zero,
          input.is_vulkan,
          input.has_buffer_storage,
          input.is_width_packed,
          input.has_direct_buffer_layout,
          input.supports_buffer_compute,
          weight.defined) ||
      !generated::small_metadata_padded_conv_2_d_materialized_buffer_input_2_x_2_in_bounds(
          spec) ||
      !generated::small_metadata_padded_conv_2_d_materialized_buffer_input_2_x_2_input_weight_channels_equal(
          input.channels, weight.input_channels)) {
    return result;
  }

  result.matched = true;
  result.family =
      SmallMetadataPaddedConv2DFamily::MaterializedBufferInput2x2;
  result.tuple_id = spec.tuple_id;
  result.metadata =
      &kSmallMetadataPaddedConv2DMaterializedBufferInput2x2Metadata;
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
