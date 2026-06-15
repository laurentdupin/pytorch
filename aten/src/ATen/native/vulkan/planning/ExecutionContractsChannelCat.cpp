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

constexpr ExecutionContractMetadata kChannelCatRank4Dim1BufferViewMetadata =
    make_execution_contract_metadata(
        generated::kChannelCatRank4Dim1BufferViewSpec.contract_name,
        generated::kChannelCatRank4Dim1BufferViewSpec.family_name,
        generated::kChannelCatRank4Dim1BufferViewSpec.tuple_id,
        generated::kChannelCatRank4Dim1BufferViewSpec.evidence_id,
        generated::kChannelCatRank4Dim1BufferViewSpec.guard_id,
        generated::kChannelCatRank4Dim1BufferViewSpec.fallback_policy,
        generated::kChannelCatRank4Dim1BufferViewSpec.materialization_policy);

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
      return generated::kChannelCatRank4Dim1BufferViewSpec.route_label;
    case ChannelCatFamily::None:
      return "aten::cat.channel_cat.none";
  }
  return "aten::cat.channel_cat.none";
}

ChannelCatMatch match_channel_cat_contract(
    ArrayRef<ChannelCatTensorInfo> tensors,
    const int64_t dim) {
  const auto& spec = generated::kChannelCatRank4Dim1BufferViewSpec;
  ChannelCatMatch result;
  if (
      !generated::channel_cat_input_count_in_bounds(
          spec, static_cast<int64_t>(tensors.size())) ||
      dim != spec.dim) {
    return result;
  }

  const ChannelCatTensorInfo& reference = tensors[0];
  if (!generated::channel_cat_reference_in_bounds(spec, reference)) {
    return result;
  }

  for (const ChannelCatTensorInfo& tensor : tensors) {
    if (!generated::channel_cat_input_in_bounds(spec, reference, tensor)) {
      return result;
    }
  }

  const int64_t total_channels =
      generated::channel_cat_total_channels_sum(spec, tensors);
  if (!generated::channel_cat_total_channels_in_bounds(spec, total_channels)) {
    return result;
  }

  result.matched = true;
  result.family = ChannelCatFamily::Rank4Dim1BufferView;
  result.tuple_id = spec.tuple_id;
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
