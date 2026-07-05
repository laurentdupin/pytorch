#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsKVCacheAppendInitialSpec.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsKVCacheAppendSpec.h>

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

constexpr ExecutionContractMetadata kKVCacheAppendInitialMetadata =
    make_execution_contract_metadata(
        generated::kKVCacheAppendInitialCacheSpec.contract_name,
        generated::kKVCacheAppendInitialCacheSpec.family_name,
        generated::kKVCacheAppendInitialCacheSpec.tuple_id,
        generated::kKVCacheAppendInitialCacheSpec.evidence_id,
        generated::kKVCacheAppendInitialCacheSpec.guard_id,
        generated::kKVCacheAppendInitialCacheSpec.fallback_policy,
        generated::kKVCacheAppendInitialCacheSpec.materialization_policy);
constexpr ExecutionContractMetadata kKVCacheAppendInitialDynamicMetadata =
    make_execution_contract_metadata(
        "KVCacheAppendContract",
        "InitialCacheDirectBuffer",
        "dynamic_initial_cache_direct_buffer",
        "kv_cache_initial_dynamic_random_shape_tests",
        "kv_cache_initial_dynamic_semantic_guards",
        "fallback_on_unsupported_layout_or_semantics",
        "kv_cache_initial_buffer_copy");
constexpr ExecutionContractMetadata kKVCacheAppendSequenceMetadata =
    make_execution_contract_metadata(
        generated::kKVCacheAppendSequenceAppendSpec.contract_name,
        generated::kKVCacheAppendSequenceAppendSpec.family_name,
        generated::kKVCacheAppendSequenceAppendSpec.tuple_id,
        generated::kKVCacheAppendSequenceAppendSpec.evidence_id,
        generated::kKVCacheAppendSequenceAppendSpec.guard_id,
        generated::kKVCacheAppendSequenceAppendSpec.fallback_policy,
        generated::kKVCacheAppendSequenceAppendSpec.materialization_policy);

int64_t size_at_or(const IntArrayRef sizes, const int64_t index) {
  if (index < 0 || index >= static_cast<int64_t>(sizes.size())) {
    return -1;
  }
  return sizes[index];
}

} // namespace

const char* kv_cache_append_family_name(const KVCacheAppendFamily family) {
  switch (family) {
    case KVCacheAppendFamily::InitialCache:
      return "KVCacheAppendInitialCache";
    case KVCacheAppendFamily::InitialCacheDirectBuffer:
      return "KVCacheAppendInitialCacheDirectBuffer";
    case KVCacheAppendFamily::SequenceAppend:
      return "KVCacheAppendSequenceAppend";
    case KVCacheAppendFamily::None:
      return "KVCacheAppendNone";
  }
  return "KVCacheAppendNone";
}

const char* kv_cache_append_op_hit_label(const KVCacheAppendFamily family) {
  switch (family) {
    case KVCacheAppendFamily::InitialCache:
    case KVCacheAppendFamily::InitialCacheDirectBuffer:
      return generated::kKVCacheAppendInitialCacheSpec.route_label;
    case KVCacheAppendFamily::SequenceAppend:
      return generated::kKVCacheAppendSequenceAppendSpec.route_label;
    case KVCacheAppendFamily::None:
      return "aten::cat.kv_cache_append.none";
  }
  return "aten::cat.kv_cache_append.none";
}

const ExecutionContractMetadata* kv_cache_initial_dynamic_metadata() {
  return &kKVCacheAppendInitialDynamicMetadata;
}

KVCacheAppendMatch match_kv_cache_append_contract(
    const IntArrayRef left_sizes,
    const IntArrayRef right_sizes,
    const ScalarType left_dtype,
    const ScalarType right_dtype,
    const bool left_is_vulkan,
    const bool right_is_vulkan,
    const int64_t dim) {
  KVCacheAppendMatch result;
  const auto& initial_spec = generated::kKVCacheAppendInitialCacheSpec;
  const auto& sequence_spec = generated::kKVCacheAppendSequenceAppendSpec;
  const int64_t left_rank = static_cast<int64_t>(left_sizes.size());
  const int64_t right_rank = static_cast<int64_t>(right_sizes.size());
  if (
      generated::kv_cache_append_initial_cache_options_match(
          initial_spec,
          right_dtype,
          left_rank,
          right_rank,
          size_at_or(left_sizes, 0),
          dim,
          size_at_or(right_sizes, 0),
          size_at_or(right_sizes, 1),
          size_at_or(right_sizes, 3),
          left_is_vulkan,
          right_is_vulkan) &&
      size_at_or(right_sizes, 2) >= initial_spec.min_sequence &&
      generated::kv_cache_append_initial_cache_in_bounds(
          initial_spec, size_at_or(right_sizes, 2))) {
    result.matched = true;
    result.family = KVCacheAppendFamily::InitialCache;
    result.tuple_id = initial_spec.tuple_id;
    result.metadata = &kKVCacheAppendInitialMetadata;
    result.sequence_length = right_sizes[2];
    return result;
  }
  if (
      generated::kv_cache_append_sequence_append_options_match(
          sequence_spec,
          left_dtype,
          right_dtype,
          left_rank,
          right_rank,
          dim,
          size_at_or(left_sizes, 0),
          size_at_or(left_sizes, 1),
          size_at_or(right_sizes, 2),
          size_at_or(left_sizes, 3),
          left_is_vulkan,
          right_is_vulkan) &&
      size_at_or(left_sizes, 2) >= sequence_spec.min_source_sequence &&
      generated::kv_cache_append_sequence_append_in_bounds(
          sequence_spec, size_at_or(left_sizes, 2)) &&
      generated::kv_cache_append_sequence_append_batch_equal(
          size_at_or(left_sizes, 0), size_at_or(right_sizes, 0)) &&
      generated::kv_cache_append_sequence_append_heads_equal(
          size_at_or(left_sizes, 1), size_at_or(right_sizes, 1)) &&
      generated::kv_cache_append_sequence_append_head_dim_equal(
          size_at_or(left_sizes, 3), size_at_or(right_sizes, 3))) {
    result.matched = true;
    result.family = KVCacheAppendFamily::SequenceAppend;
    result.tuple_id = sequence_spec.tuple_id;
    result.metadata = &kKVCacheAppendSequenceMetadata;
    result.sequence_length = left_sizes[2];
    return result;
  }
  return result;
}

bool matches_kv_cache_append_contract(
    const IntArrayRef left_sizes,
    const IntArrayRef right_sizes,
    const ScalarType left_dtype,
    const ScalarType right_dtype,
    const bool left_is_vulkan,
    const bool right_is_vulkan,
    const int64_t dim) {
  return match_kv_cache_append_contract(
             left_sizes,
             right_sizes,
             left_dtype,
             right_dtype,
             left_is_vulkan,
             right_is_vulkan,
             dim)
      .matched;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
