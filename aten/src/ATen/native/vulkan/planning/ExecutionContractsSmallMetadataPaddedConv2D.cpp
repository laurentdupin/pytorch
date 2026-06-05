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
constexpr const char* kMaterializationSmallMetadataPaddedConv2DInput =
    "materialize_small_metadata_input_then_conv2d_buffer_float";

constexpr int64_t kSmallMetadataPaddedConv2DBatch = 1;
constexpr int64_t kSmallMetadataPaddedConv2DInputChannels = 16;
constexpr int64_t kSmallMetadataPaddedConv2DInputHeight = 721;
constexpr int64_t kSmallMetadataPaddedConv2DInputWidth = 1281;
constexpr int64_t kSmallMetadataPaddedConv2DOutputChannels = 32;
constexpr int64_t kSmallMetadataPaddedConv2DKernel = 2;
constexpr const char* kSmallMetadataPaddedConv2DTupleId =
    "input_1x16x721x1281_weight_32x16x2x2_stride1";
constexpr ExecutionContractMetadata
    kSmallMetadataPaddedConv2DMaterializedBufferInput2x2Metadata =
        make_execution_contract_metadata(
            "SmallMetadataPaddedConv2DContract",
            "MaterializedBufferInput2x2",
            kSmallMetadataPaddedConv2DTupleId,
            "task028_paddleocr_conv2d_pressure_classification",
            "small_metadata_padded_conv2d_adjacent_guards",
            kFallbackUnsupportedShapesDoNotMatch,
            kMaterializationSmallMetadataPaddedConv2DInput);

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
  if (
      options.transposed || options.quantized || options.groups != 1 ||
      options.stride_h != 1 || options.stride_w != 1 ||
      options.padding_h != 0 || options.padding_w != 0 ||
      options.dilation_h != 1 || options.dilation_w != 1 ||
      !options.output_padding_is_zero || !input.is_vulkan ||
      input.dtype != kFloat || input.rank != 4 ||
      input.batch != kSmallMetadataPaddedConv2DBatch ||
      input.channels != kSmallMetadataPaddedConv2DInputChannels ||
      input.height != kSmallMetadataPaddedConv2DInputHeight ||
      input.width != kSmallMetadataPaddedConv2DInputWidth ||
      !input.has_buffer_storage || !input.is_width_packed ||
      input.has_direct_buffer_layout || !input.supports_buffer_compute ||
      !weight.defined || weight.dtype != kFloat || weight.rank != 4 ||
      weight.output_channels != kSmallMetadataPaddedConv2DOutputChannels ||
      weight.input_channels != kSmallMetadataPaddedConv2DInputChannels ||
      weight.kernel_h != kSmallMetadataPaddedConv2DKernel ||
      weight.kernel_w != kSmallMetadataPaddedConv2DKernel) {
    return result;
  }

  result.matched = true;
  result.family =
      SmallMetadataPaddedConv2DFamily::MaterializedBufferInput2x2;
  result.tuple_id = kSmallMetadataPaddedConv2DTupleId;
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
