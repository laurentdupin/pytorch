#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/ExecutionContractDiagnostics.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsSafeViewReshapeAliasSpec.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsSafeViewReshapeSpec.h>

#include <c10/util/Exception.h>
#include <c10/util/strides.h>

#include <algorithm>
#include <vector>

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
    kSafeViewReshapeViewMaterializedDirectBufferMetadata =
        make_execution_contract_metadata(
            generated::kSafeViewReshapeViewMaterializedDirectBufferSpec
                .contract_name,
            generated::kSafeViewReshapeViewMaterializedDirectBufferSpec
                .family_name,
            generated::kSafeViewReshapeViewMaterializedDirectBufferSpec
                .tuple_id,
            generated::kSafeViewReshapeViewMaterializedDirectBufferSpec
                .evidence_id,
            generated::kSafeViewReshapeViewMaterializedDirectBufferSpec
                .guard_id,
            generated::kSafeViewReshapeViewMaterializedDirectBufferSpec
                .fallback_policy,
            generated::kSafeViewReshapeViewMaterializedDirectBufferSpec
                .materialization_policy);
constexpr ExecutionContractMetadata
    kSafeViewReshapeAliasDenseBufferDirectMetadata =
        make_execution_contract_metadata(
            generated::kSafeViewReshapeReshapeAliasDenseBufferDirectSpec
                .contract_name,
            generated::kSafeViewReshapeReshapeAliasDenseBufferDirectSpec
                .family_name,
            generated::kSafeViewReshapeReshapeAliasDenseBufferDirectSpec
                .tuple_id,
            generated::kSafeViewReshapeReshapeAliasDenseBufferDirectSpec
                .evidence_id,
            generated::kSafeViewReshapeReshapeAliasDenseBufferDirectSpec
                .guard_id,
            generated::kSafeViewReshapeReshapeAliasDenseBufferDirectSpec
                .fallback_policy,
            generated::kSafeViewReshapeReshapeAliasDenseBufferDirectSpec
                .materialization_policy);

bool is_contiguous_stride(
    const IntArrayRef sizes,
    const IntArrayRef strides) {
  return strides.equals(c10::contiguous_strides(sizes));
}

bool is_non_overlapping_dense_stride(
    const IntArrayRef sizes,
    const IntArrayRef strides) {
  TORCH_INTERNAL_ASSERT(sizes.size() == strides.size());
  std::vector<size_t> dims;
  dims.reserve(sizes.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] > 1) {
      dims.push_back(i);
    }
  }
  std::sort(dims.begin(), dims.end(), [&](const size_t lhs, const size_t rhs) {
    return strides[lhs] < strides[rhs];
  });
  int64_t expected_stride = 1;
  for (const size_t dim : dims) {
    if (strides[dim] != expected_stride) {
      return false;
    }
    expected_stride *= sizes[dim];
  }
  return true;
}

} // namespace

const char* safe_view_reshape_family_name(
    const SafeViewReshapeFamily family) {
  switch (family) {
    case SafeViewReshapeFamily::ViewMaterializedDirectBuffer:
      return "SafeViewReshapeViewMaterializedDirectBuffer";
    case SafeViewReshapeFamily::ReshapeAliasDenseBufferDirect:
      return "SafeViewReshapeReshapeAliasDenseBufferDirect";
    case SafeViewReshapeFamily::None:
      return "SafeViewReshapeNone";
  }
  return "SafeViewReshapeNone";
}

SafeViewReshapeMatch
match_safe_view_reshape_materialized_direct_buffer_contract(
    const IntArrayRef input_sizes,
    const IntArrayRef output_sizes,
    const IntArrayRef output_strides,
    const int64_t storage_offset) {
  SafeViewReshapeMatch result;
  const auto& spec =
      generated::kSafeViewReshapeViewMaterializedDirectBufferSpec;
  if (!generated::safe_view_materialized_direct_buffer_input_rank_in_bounds(
          spec, static_cast<int64_t>(input_sizes.size()))) {
    log_contract_reject(
        &kSafeViewReshapeViewMaterializedDirectBufferMetadata,
        ContractAdmissionPhase::GeneratedBounds,
        "safe_view_materialized_direct_buffer_input_rank_in_bounds",
        "view_input_rank_out_of_bounds",
        "generated");
    return result;
  }
  if (!generated::safe_view_materialized_direct_buffer_output_rank_in_bounds(
          spec, static_cast<int64_t>(output_sizes.size()))) {
    log_contract_reject(
        &kSafeViewReshapeViewMaterializedDirectBufferMetadata,
        ContractAdmissionPhase::GeneratedBounds,
        "safe_view_materialized_direct_buffer_output_rank_in_bounds",
        "view_output_rank_out_of_bounds",
        "generated");
    return result;
  }
  if (!generated::safe_view_materialized_direct_buffer_storage_offset_matches(
          spec, storage_offset)) {
    log_contract_reject(
        &kSafeViewReshapeViewMaterializedDirectBufferMetadata,
        ContractAdmissionPhase::GeneratedOptions,
        "safe_view_materialized_direct_buffer_storage_offset_matches",
        "view_storage_offset_mismatch",
        "generated");
    return result;
  }
  if (!is_contiguous_stride(output_sizes, output_strides)) {
    log_contract_reject(
        &kSafeViewReshapeViewMaterializedDirectBufferMetadata,
        ContractAdmissionPhase::MaterializationPolicy,
        "is_contiguous_stride",
        "view_output_stride_not_contiguous",
        "handwritten");
    return result;
  }

  if (!generated::safe_view_materialized_direct_buffer_product_equal(
          spec, input_sizes, output_sizes)) {
    log_contract_reject(
        &kSafeViewReshapeViewMaterializedDirectBufferMetadata,
        ContractAdmissionPhase::GeneratedRelationship,
        "safe_view_materialized_direct_buffer_product_equal",
        "view_product_mismatch",
        "generated");
    return result;
  }

  if (!generated::safe_view_materialized_direct_buffer_output_last_dim_multiple_matches(
          spec,
          !output_sizes.empty(),
          output_sizes.empty() ? 0 : output_sizes.back())) {
    log_contract_reject(
        &kSafeViewReshapeViewMaterializedDirectBufferMetadata,
        ContractAdmissionPhase::GeneratedBounds,
        "safe_view_materialized_direct_buffer_output_last_dim_multiple_matches",
        "view_output_last_dim_multiple_mismatch",
        "generated");
    return result;
  }

  result.matched = true;
  result.family = SafeViewReshapeFamily::ViewMaterializedDirectBuffer;
  result.tuple_id = spec.tuple_id;
  result.metadata = &kSafeViewReshapeViewMaterializedDirectBufferMetadata;
  log_contract_accept(
      result.metadata,
      "match_safe_view_reshape_materialized_direct_buffer_contract");
  return result;
}

bool matches_safe_view_reshape_materialized_direct_buffer_contract(
    const IntArrayRef input_sizes,
    const IntArrayRef output_sizes,
    const IntArrayRef output_strides,
    const int64_t storage_offset) {
  return match_safe_view_reshape_materialized_direct_buffer_contract(
             input_sizes, output_sizes, output_strides, storage_offset)
      .matched;
}

SafeViewReshapeMatch match_safe_view_reshape_contract(
    const IntArrayRef input_sizes,
    const IntArrayRef input_logical_strides,
    const IntArrayRef output_sizes,
    const IntArrayRef output_strides,
    const bool input_is_float,
    const bool input_has_buffer_storage,
    const int64_t storage_offset) {
  SafeViewReshapeMatch result;
  const auto& spec =
      generated::kSafeViewReshapeReshapeAliasDenseBufferDirectSpec;
  if (
      !input_is_float || !input_has_buffer_storage ||
      !generated::safe_reshape_alias_dense_buffer_direct_input_rank_in_bounds(
          spec, static_cast<int64_t>(input_sizes.size())) ||
      !generated::safe_reshape_alias_dense_buffer_direct_output_rank_in_bounds(
          spec, static_cast<int64_t>(output_sizes.size())) ||
      !generated::safe_reshape_alias_dense_buffer_direct_storage_offset_matches(
          spec, storage_offset) ||
      !is_non_overlapping_dense_stride(input_sizes, input_logical_strides) ||
      !is_non_overlapping_dense_stride(output_sizes, output_strides)) {
    return result;
  }

  if (!generated::safe_reshape_alias_dense_buffer_direct_product_equal(
          spec, input_sizes, output_sizes)) {
    return result;
  }

  if (!generated::safe_reshape_alias_dense_buffer_direct_output_last_dim_multiple_matches(
          spec,
          !output_sizes.empty(),
          output_sizes.empty() ? 0 : output_sizes.back())) {
    return result;
  }

  result.matched = true;
  result.family = SafeViewReshapeFamily::ReshapeAliasDenseBufferDirect;
  result.tuple_id = spec.tuple_id;
  result.metadata = &kSafeViewReshapeAliasDenseBufferDirectMetadata;
  return result;
}

bool matches_safe_view_reshape_contract(
    const IntArrayRef input_sizes,
    const IntArrayRef input_logical_strides,
    const IntArrayRef output_sizes,
    const IntArrayRef output_strides,
    const bool input_is_float,
    const bool input_has_buffer_storage,
    const int64_t storage_offset) {
  return match_safe_view_reshape_contract(
             input_sizes,
             input_logical_strides,
             output_sizes,
             output_strides,
             input_is_float,
             input_has_buffer_storage,
             storage_offset)
      .matched;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
