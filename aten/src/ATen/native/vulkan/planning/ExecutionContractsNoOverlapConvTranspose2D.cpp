#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsNoOverlapConvTranspose2DSpec.h>

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
    kNoOverlapConvTranspose2DKernel2Stride2FloatBufferMetadata =
        make_execution_contract_metadata(
            generated::kNoOverlapConvTranspose2DKernel2Stride2FloatBufferSpec
                .contract_name,
            generated::kNoOverlapConvTranspose2DKernel2Stride2FloatBufferSpec
                .family_name,
            generated::kNoOverlapConvTranspose2DKernel2Stride2FloatBufferSpec
                .tuple_id,
            generated::kNoOverlapConvTranspose2DKernel2Stride2FloatBufferSpec
                .evidence_id,
            generated::kNoOverlapConvTranspose2DKernel2Stride2FloatBufferSpec
                .guard_id,
            generated::kNoOverlapConvTranspose2DKernel2Stride2FloatBufferSpec
                .fallback_policy,
            generated::kNoOverlapConvTranspose2DKernel2Stride2FloatBufferSpec
                .materialization_policy);

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
  const auto& spec =
      generated::kNoOverlapConvTranspose2DKernel2Stride2FloatBufferSpec;
  if (
      !generated::no_overlap_conv_transpose_2_d_kernel_2_stride_2_float_buffer_options_match(
          spec,
          input.dtype,
          packed.weight_dtype,
          input.rank,
          packed.weight_rank,
          input.batch,
          options.groups,
          packed.kernel_h,
          packed.kernel_w,
          options.stride_h,
          options.stride_w,
          options.padding_h,
          options.padding_w,
          options.dilation_h,
          options.dilation_w,
          options.transposed,
          options.quantized,
          packed.quantized,
          options.output_padding_is_zero,
          input.is_vulkan,
          input.has_buffer_storage,
          input.supports_buffer_compute,
          packed.defined,
          packed.execution_is_buffer_direct,
          packed.weight_has_buffer_storage,
          packed.bias_has_buffer_storage,
          packed.bias_is_float) ||
      !generated::no_overlap_conv_transpose_2_d_kernel_2_stride_2_float_buffer_in_bounds(
          spec, input.channels) ||
      packed.input_channels != input.channels) {
    return result;
  }

  result.matched = true;
  result.family = NoOverlapConvTranspose2DFamily::Kernel2Stride2FloatBuffer;
  result.tuple_id = spec.tuple_id;
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
