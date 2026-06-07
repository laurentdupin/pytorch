#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsChannelCatSpec.h>

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
constexpr const char* kMaterializationChannelCatBufferView =
    "channel_cat_buffer_view_copy_kernel";

constexpr ExecutionContractMetadata kChannelCatRank4Dim1BufferViewMetadata =
    make_execution_contract_metadata(
        generated::kChannelCatContractName,
        generated::kChannelCatRank4Dim1BufferViewFamilyName,
        generated::kChannelCatRank4Dim1BufferViewTupleId,
        "channel_cat_buffer_view_focused_tests",
        "channel_cat_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationChannelCatBufferView);

} // namespace

const char* channel_cat_family_name(const ChannelCatFamily family) {
  switch (family) {
    case ChannelCatFamily::Rank4Dim1BufferView:
      return "ChannelCatRank4Dim1BufferView";
    case ChannelCatFamily::None:
      return "ChannelCatNone";
  }
  return "ChannelCatNone";
}

const char* channel_cat_op_hit_label(const ChannelCatFamily family) {
  switch (family) {
    case ChannelCatFamily::Rank4Dim1BufferView:
      return generated::kChannelCatRank4Dim1BufferViewRouteLabel;
    case ChannelCatFamily::None:
      return "aten::cat.channel_cat.none";
  }
  return "aten::cat.channel_cat.none";
}

ChannelCatMatch match_channel_cat_contract(
    ArrayRef<ChannelCatTensorInfo> tensors,
    const int64_t dim) {
  ChannelCatMatch result;
  if (
      tensors.size() < generated::kChannelCatRank4Dim1MinInputs ||
      tensors.size() > generated::kChannelCatRank4Dim1MaxInputs ||
      dim != generated::kChannelCatRank4Dim1Dim) {
    return result;
  }

  const ChannelCatTensorInfo& reference = tensors[0];
  if (
      (generated::kChannelCatRank4Dim1RequiresVulkan &&
       !reference.is_vulkan) ||
      reference.dtype != kFloat ||
      reference.rank != generated::kChannelCatRank4Dim1Rank ||
      reference.batch != generated::kChannelCatRank4Dim1Batch ||
      (generated::kChannelCatRank4Dim1RequiresContiguous &&
       !reference.is_contiguous) ||
      reference.height < generated::kChannelCatRank4Dim1MinHeight ||
      reference.height > generated::kChannelCatRank4Dim1MaxHeight ||
      reference.width < generated::kChannelCatRank4Dim1MinWidth ||
      reference.width > generated::kChannelCatRank4Dim1MaxWidth) {
    return result;
  }

  int64_t total_channels = 0;
  for (const ChannelCatTensorInfo& tensor : tensors) {
    if (
        !tensor.is_vulkan || tensor.dtype != reference.dtype ||
        tensor.rank != reference.rank || tensor.batch != reference.batch ||
        tensor.height != reference.height || tensor.width != reference.width ||
        (generated::kChannelCatRank4Dim1RequiresContiguous &&
         !tensor.is_contiguous) ||
        (generated::kChannelCatRank4Dim1RequiresBufferStorage &&
         !tensor.has_buffer_storage) ||
        (generated::kChannelCatRank4Dim1RequiresBufferCompute &&
         !tensor.supports_buffer_compute) ||
        tensor.channels < generated::kChannelCatRank4Dim1MinInputChannels ||
        tensor.channels > generated::kChannelCatRank4Dim1MaxInputChannels ||
        tensor.channels % generated::kChannelCatRank4Dim1ChannelMultiple != 0) {
      return result;
    }
    total_channels += tensor.channels;
  }

  if (
      total_channels <= 0 ||
      total_channels > generated::kChannelCatRank4Dim1MaxTotalChannels ||
      total_channels % generated::kChannelCatRank4Dim1ChannelMultiple != 0) {
    return result;
  }

  result.matched = true;
  result.family = ChannelCatFamily::Rank4Dim1BufferView;
  result.tuple_id = generated::kChannelCatRank4Dim1BufferViewTupleId;
  result.metadata = &kChannelCatRank4Dim1BufferViewMetadata;
  result.input_count = static_cast<int64_t>(tensors.size());
  result.total_channels = total_channels;
  return result;
}

bool matches_channel_cat_contract(
    ArrayRef<ChannelCatTensorInfo> tensors,
    const int64_t dim) {
  return match_channel_cat_contract(tensors, dim).matched;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
