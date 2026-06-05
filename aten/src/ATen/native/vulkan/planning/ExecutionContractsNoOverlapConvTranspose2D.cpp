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
constexpr const char* kMaterializationConvTransposeNoOverlapBuffer =
    "conv_transpose2d_no_overlap_buffer_kernel";

constexpr int64_t kNoOverlapConvTranspose2DBatch = 1;
constexpr int64_t kNoOverlapConvTranspose2DMinInputChannels = 64;
constexpr int64_t kNoOverlapConvTranspose2DKernel = 2;
constexpr int64_t kNoOverlapConvTranspose2DStride = 2;
constexpr const char* kNoOverlapConvTranspose2DTupleId =
    "batch1_cin_ge64_kernel2_stride2_float_buffer";
constexpr ExecutionContractMetadata
    kNoOverlapConvTranspose2DKernel2Stride2FloatBufferMetadata =
        make_execution_contract_metadata(
            "NoOverlapConvTranspose2DContract",
            "Kernel2Stride2FloatBuffer",
            kNoOverlapConvTranspose2DTupleId,
            "conv_transpose2d_no_overlap_2x2_stride2_buffer_float",
            "conv_transpose2d_no_overlap_adjacent_guards",
            kFallbackUnsupportedShapesDoNotMatch,
            kMaterializationConvTransposeNoOverlapBuffer);

} // namespace

const char* no_overlap_conv_transpose2d_family_name(
    const NoOverlapConvTranspose2DFamily family) {
  switch (family) {
    case NoOverlapConvTranspose2DFamily::Kernel2Stride2FloatBuffer:
      return "NoOverlapConvTranspose2DKernel2Stride2FloatBuffer";
    case NoOverlapConvTranspose2DFamily::None:
      return "NoOverlapConvTranspose2DNone";
  }
  return "NoOverlapConvTranspose2DNone";
}

NoOverlapConvTranspose2DMatch match_no_overlap_conv_transpose2d_contract(
    const NoOverlapConvTranspose2DTensorInfo& input,
    const NoOverlapConvTranspose2DPackedInfo& packed,
    const NoOverlapConvTranspose2DOptions& options) {
  NoOverlapConvTranspose2DMatch result;
  if (
      !options.transposed || options.quantized || options.groups != 1 ||
      options.stride_h != kNoOverlapConvTranspose2DStride ||
      options.stride_w != kNoOverlapConvTranspose2DStride ||
      options.padding_h != 0 || options.padding_w != 0 ||
      options.dilation_h != 1 || options.dilation_w != 1 ||
      !options.output_padding_is_zero || !input.is_vulkan ||
      input.dtype != kFloat || input.rank != 4 ||
      input.batch != kNoOverlapConvTranspose2DBatch ||
      input.channels < kNoOverlapConvTranspose2DMinInputChannels ||
      !input.has_buffer_storage || !input.supports_buffer_compute ||
      !packed.defined || !packed.execution_is_buffer_direct ||
      packed.quantized || packed.weight_dtype != kFloat ||
      packed.weight_rank != 4 ||
      packed.input_channels != input.channels ||
      packed.kernel_h != kNoOverlapConvTranspose2DKernel ||
      packed.kernel_w != kNoOverlapConvTranspose2DKernel ||
      !packed.weight_has_buffer_storage || !packed.bias_has_buffer_storage ||
      !packed.bias_is_float) {
    return result;
  }

  result.matched = true;
  result.family = NoOverlapConvTranspose2DFamily::Kernel2Stride2FloatBuffer;
  result.tuple_id = kNoOverlapConvTranspose2DTupleId;
  result.metadata = &kNoOverlapConvTranspose2DKernel2Stride2FloatBufferMetadata;
  return result;
}

bool matches_no_overlap_conv_transpose2d_contract(
    const NoOverlapConvTranspose2DTensorInfo& input,
    const NoOverlapConvTranspose2DPackedInfo& packed,
    const NoOverlapConvTranspose2DOptions& options) {
  return match_no_overlap_conv_transpose2d_contract(input, packed, options)
      .matched;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
