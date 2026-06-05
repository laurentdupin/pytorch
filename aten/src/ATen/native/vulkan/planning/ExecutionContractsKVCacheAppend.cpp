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
constexpr const char* kMaterializationKVCacheAppendBuffer =
    "kv_cache_append_buffer_kernel";

constexpr int64_t kKVCacheAppendBatch = 1;
constexpr int64_t kKVCacheAppendHeads = 4;
constexpr int64_t kKVCacheAppendMinSequence = 99;
constexpr int64_t kKVCacheAppendMaxSequence = 116;
constexpr int64_t kKVCacheAppendMaxSourceSequence = 115;
constexpr int64_t kKVCacheAppendTokenSequence = 1;
constexpr int64_t kKVCacheAppendHeadDim = 128;
constexpr const char* kKVCacheAppendInitialTupleId =
    "initial_empty_s99_to_s116_heads4_dim128";
constexpr const char* kKVCacheAppendSequenceTupleId =
    "sequence_append_s99_to_s115_token1_heads4_dim128";
constexpr ExecutionContractMetadata kKVCacheAppendInitialMetadata =
    make_execution_contract_metadata(
        "KVCacheAppendContract",
        "InitialCache",
        kKVCacheAppendInitialTupleId,
        "kv_cache_append_focused_tests",
        "kv_cache_append_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationKVCacheAppendBuffer);
constexpr ExecutionContractMetadata kKVCacheAppendSequenceMetadata =
    make_execution_contract_metadata(
        "KVCacheAppendContract",
        "SequenceAppend",
        kKVCacheAppendSequenceTupleId,
        "kv_cache_append_focused_tests",
        "kv_cache_append_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationKVCacheAppendBuffer);

bool matches_kv_cache_state_shape(const IntArrayRef sizes) {
  return sizes.size() == 4 && sizes[0] == kKVCacheAppendBatch &&
      sizes[1] == kKVCacheAppendHeads &&
      sizes[2] >= kKVCacheAppendMinSequence &&
      sizes[2] <= kKVCacheAppendMaxSequence &&
      sizes[3] == kKVCacheAppendHeadDim;
}

bool matches_kv_cache_token_shape(const IntArrayRef sizes) {
  return sizes.size() == 4 && sizes[0] == kKVCacheAppendBatch &&
      sizes[1] == kKVCacheAppendHeads &&
      sizes[2] == kKVCacheAppendTokenSequence &&
      sizes[3] == kKVCacheAppendHeadDim;
}

bool matches_empty_initial_cache_shape(const IntArrayRef sizes) {
  return sizes.size() == 1 && sizes[0] == 0;
}

} // namespace

const char* kv_cache_append_family_name(const KVCacheAppendFamily family) {
  switch (family) {
    case KVCacheAppendFamily::InitialCache:
      return "KVCacheAppendInitialCache";
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
      return "aten::cat.kv_cache_initial_dim2_buffer";
    case KVCacheAppendFamily::SequenceAppend:
      return "aten::cat.kv_cache_append_dim2_buffer";
    case KVCacheAppendFamily::None:
      return "aten::cat.kv_cache_append.none";
  }
  return "aten::cat.kv_cache_append.none";
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
  if (!left_is_vulkan || !right_is_vulkan || dim != 2) {
    return result;
  }
  if (
      matches_empty_initial_cache_shape(left_sizes) &&
      right_dtype == kFloat && matches_kv_cache_state_shape(right_sizes)) {
    result.matched = true;
    result.family = KVCacheAppendFamily::InitialCache;
    result.tuple_id = kKVCacheAppendInitialTupleId;
    result.metadata = &kKVCacheAppendInitialMetadata;
    result.sequence_length = right_sizes[2];
    return result;
  }
  if (
      left_dtype == kFloat && right_dtype == kFloat &&
      matches_kv_cache_state_shape(left_sizes) &&
      matches_kv_cache_token_shape(right_sizes) &&
      left_sizes[2] <= kKVCacheAppendMaxSourceSequence) {
    result.matched = true;
    result.family = KVCacheAppendFamily::SequenceAppend;
    result.tuple_id = kKVCacheAppendSequenceTupleId;
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
