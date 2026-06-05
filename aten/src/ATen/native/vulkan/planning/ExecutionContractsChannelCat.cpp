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
constexpr const char* kMaterializationChannelCatBufferView =
    "channel_cat_buffer_view_copy_kernel";

constexpr int64_t kChannelCatRank4Dim1MinInputs = 3;
constexpr int64_t kChannelCatRank4Dim1MaxInputs = 8;
constexpr int64_t kChannelCatRank4Dim1Batch = 1;
constexpr int64_t kChannelCatRank4Dim1MaxInputChannels = 256;
constexpr int64_t kChannelCatRank4Dim1MaxTotalChannels = 1024;
constexpr int64_t kChannelCatRank4Dim1MaxHeight = 128;
constexpr int64_t kChannelCatRank4Dim1MaxWidth = 128;
constexpr const char* kChannelCatRank4Dim1BufferViewTupleId =
    "rank4_dim1_inputs3_to_8_c_mult4_spatial_le128_total_c_le1024";
constexpr ExecutionContractMetadata kChannelCatRank4Dim1BufferViewMetadata =
    make_execution_contract_metadata(
        "ChannelCatContract",
        "Rank4Dim1BufferView",
        kChannelCatRank4Dim1BufferViewTupleId,
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
      return "aten::cat.buffer_channel_view";
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
      tensors.size() < kChannelCatRank4Dim1MinInputs ||
      tensors.size() > kChannelCatRank4Dim1MaxInputs || dim != 1) {
    return result;
  }

  const ChannelCatTensorInfo& reference = tensors[0];
  if (
      !reference.is_vulkan || reference.dtype != kFloat ||
      reference.rank != 4 || reference.batch != kChannelCatRank4Dim1Batch ||
      !reference.is_contiguous || reference.height <= 0 ||
      reference.height > kChannelCatRank4Dim1MaxHeight ||
      reference.width <= 0 || reference.width > kChannelCatRank4Dim1MaxWidth) {
    return result;
  }

  int64_t total_channels = 0;
  for (const ChannelCatTensorInfo& tensor : tensors) {
    if (
        !tensor.is_vulkan || tensor.dtype != reference.dtype ||
        tensor.rank != reference.rank || tensor.batch != reference.batch ||
        tensor.height != reference.height || tensor.width != reference.width ||
        !tensor.is_contiguous || !tensor.has_buffer_storage ||
        !tensor.supports_buffer_compute || tensor.channels <= 0 ||
        tensor.channels > kChannelCatRank4Dim1MaxInputChannels ||
        tensor.channels % 4 != 0) {
      return result;
    }
    total_channels += tensor.channels;
  }

  if (
      total_channels <= 0 ||
      total_channels > kChannelCatRank4Dim1MaxTotalChannels ||
      total_channels % 4 != 0) {
    return result;
  }

  result.matched = true;
  result.family = ChannelCatFamily::Rank4Dim1BufferView;
  result.tuple_id = kChannelCatRank4Dim1BufferViewTupleId;
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
