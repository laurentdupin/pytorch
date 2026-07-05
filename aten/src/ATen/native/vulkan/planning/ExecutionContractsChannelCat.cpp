#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/DynamicProgramRuntime.h>
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

constexpr ExecutionContractMetadata kCatAxisDirectBufferMetadata =
    make_execution_contract_metadata(
        "ChannelCatContract",
        "Rank4Dim1RuntimeShape",
        "cat_axis_rank4_dim1_direct_buffer_runtime_shape",
        "dynamic_cat_axis_rank4_dim1_random_shape_tests",
        "cat_axis_rank4_dim1_semantic_guards",
        "unsupported_semantics_hard_fail",
        "device_channel_cat_materialization");

bool supports_dynamic_rank4_dim1_cat(ArrayRef<ChannelCatTensorInfo> tensors) {
  if (tensors.size() < 2) {
    return false;
  }
  const ChannelCatTensorInfo& reference = tensors[0];
  if (
      !reference.is_vulkan || reference.dtype != kFloat ||
      reference.rank != 4 || !reference.has_buffer_storage ||
      !reference.supports_buffer_compute || reference.batch <= 0 ||
      reference.channels <= 0 || reference.height <= 0 ||
      reference.width <= 0 || reference.channels % 4 != 0) {
    return false;
  }
  int64_t total_channels = 0;
  for (const ChannelCatTensorInfo& tensor : tensors) {
    if (
        !tensor.is_vulkan || tensor.dtype != reference.dtype ||
        tensor.rank != 4 || !tensor.has_buffer_storage ||
        !tensor.supports_buffer_compute ||
        tensor.batch != reference.batch ||
        tensor.height != reference.height ||
        tensor.width != reference.width || tensor.channels <= 0 ||
        tensor.channels % 4 != 0) {
      return false;
    }
    total_channels += tensor.channels;
  }
  return total_channels > 0 && total_channels % 4 == 0;
}

} // namespace

const char* channel_cat_family_name(const ChannelCatFamily family) {
  switch (family) {
    case ChannelCatFamily::Rank4Dim1BufferView:
      return "ChannelCatRank4Dim1BufferView";
    case ChannelCatFamily::GenericRank4Dim1DirectBuffer:
      return "CatAxisDirectBuffer";
    case ChannelCatFamily::None:
      return "ChannelCatNone";
  }
  return "ChannelCatNone";
}

const char* channel_cat_op_hit_label(const ChannelCatFamily family) {
  switch (family) {
    case ChannelCatFamily::Rank4Dim1BufferView:
      return generated::kChannelCatRank4Dim1BufferViewSpec.route_label;
    case ChannelCatFamily::GenericRank4Dim1DirectBuffer:
      return "aten::cat.buffer_channel_view";
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
      generated::channel_cat_input_count_in_bounds(
          spec, static_cast<int64_t>(tensors.size())) &&
      dim == spec.dim) {
    const ChannelCatTensorInfo& reference = tensors[0];
    bool generated_match =
        generated::channel_cat_reference_in_bounds(spec, reference);
    if (generated_match) {
      for (const ChannelCatTensorInfo& tensor : tensors) {
        if (!generated::channel_cat_input_in_bounds(spec, reference, tensor)) {
          generated_match = false;
          break;
        }
      }
    }

    if (generated_match) {
      const int64_t total_channels =
          generated::channel_cat_total_channels_sum(spec, tensors);
      if (generated::channel_cat_total_channels_in_bounds(
              spec, total_channels)) {
        result.matched = true;
        result.family = ChannelCatFamily::Rank4Dim1BufferView;
        result.tuple_id = spec.tuple_id;
        result.metadata = &kChannelCatRank4Dim1BufferViewMetadata;
        result.input_count = static_cast<int64_t>(tensors.size());
        result.total_channels = total_channels;
        return result;
      }
    }
  }

  if (dim != 1 || !supports_dynamic_rank4_dim1_cat(tensors)) {
    return result;
  }
  const DynamicProgramDecision decision = build_dynamic_program_runtime_plan(
      make_cat_axis_direct_buffer_dynamic_program(
          tensors,
          dim,
          kFloat,
          /*output_direct_buffer=*/true,
          &kCatAxisDirectBufferMetadata,
          /*behavior_enabled=*/true));
  if (!decision.runtime_selection_authorized) {
    return result;
  }
  int64_t total_channels = 0;
  for (const ChannelCatTensorInfo& tensor : tensors) {
    total_channels += tensor.channels;
  }
  result.matched = true;
  result.family = ChannelCatFamily::GenericRank4Dim1DirectBuffer;
  result.tuple_id = kCatAxisDirectBufferMetadata.tuple_id;
  result.metadata = &kCatAxisDirectBufferMetadata;
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
