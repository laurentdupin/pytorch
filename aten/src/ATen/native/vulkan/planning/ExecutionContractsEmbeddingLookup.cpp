#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/generated/ExecutionContractsEmbeddingLookupSpec.h>

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
constexpr const char* kMaterializationEmbeddingLookupBuffer =
    "embedding_lookup_buffer_kernel";

constexpr int64_t kEmbeddingLookupTokenNumEmbeddings = 120818;
constexpr int64_t kEmbeddingLookupTokenEmbeddingDim = 2048;
constexpr int64_t kEmbeddingLookupTokenBatch = 1;
constexpr int64_t kEmbeddingLookupTokenMinIndices = 1;
constexpr int64_t kEmbeddingLookupTokenMaxIndices = 116;
constexpr const char* kEmbeddingLookupTokenBatch1TupleId =
    "token_batch1_vocab120818_dim2048_indices1_to_116";
constexpr ExecutionContractMetadata kEmbeddingLookupTokenBatch1Metadata =
    make_execution_contract_metadata(
        "EmbeddingLookupContract",
        "TokenBatch1",
        kEmbeddingLookupTokenBatch1TupleId,
        "embedding_lookup_focused_tests",
        "embedding_lookup_adjacent_guards",
        kFallbackUnsupportedShapesDoNotMatch,
        kMaterializationEmbeddingLookupBuffer);
constexpr ExecutionContractMetadata kEmbeddingLookupSmallBoundedMetadata =
    make_execution_contract_metadata(
        generated::kEmbeddingLookupSmallBoundedLookupSpec.contract_name,
        generated::kEmbeddingLookupSmallBoundedLookupSpec.family_name,
        generated::kEmbeddingLookupSmallBoundedLookupSpec.tuple_id,
        generated::kEmbeddingLookupSmallBoundedLookupSpec.evidence_id,
        generated::kEmbeddingLookupSmallBoundedLookupSpec.guard_id,
        generated::kEmbeddingLookupSmallBoundedLookupSpec.fallback_policy,
        generated::kEmbeddingLookupSmallBoundedLookupSpec.materialization_policy);

int64_t product_of_sizes(const IntArrayRef sizes) {
  int64_t product = 1;
  for (const int64_t size : sizes) {
    product *= size;
  }
  return product;
}

} // namespace

const char* embedding_lookup_family_name(const EmbeddingLookupFamily family) {
  switch (family) {
    case EmbeddingLookupFamily::SmallBoundedLookup:
      return "EmbeddingLookupSmallBoundedLookup";
    case EmbeddingLookupFamily::TokenBatch1:
      return "EmbeddingLookupTokenBatch1";
    case EmbeddingLookupFamily::None:
      return "EmbeddingLookupNone";
  }
  return "EmbeddingLookupNone";
}

const char* embedding_lookup_write_label(const EmbeddingLookupFamily family) {
  switch (family) {
    case EmbeddingLookupFamily::SmallBoundedLookup:
      return generated::kEmbeddingLookupSmallBoundedLookupSpec.route_label;
    case EmbeddingLookupFamily::TokenBatch1:
      return "buffer_float_long.token_batch1";
    case EmbeddingLookupFamily::None:
      return "buffer_float_long.none";
  }
  return "buffer_float_long.none";
}

EmbeddingLookupMatch match_embedding_lookup_contract(
    const IntArrayRef weight_sizes,
    const IntArrayRef indices_sizes,
    const ScalarType weight_dtype,
    const ScalarType indices_dtype,
    const bool weight_is_vulkan,
    const bool indices_is_vulkan,
    const bool padding_idx_has_hint,
    const bool scale_grad_by_freq,
    const bool sparse) {
  EmbeddingLookupMatch result;
  if (
      !weight_is_vulkan || !indices_is_vulkan ||
      weight_dtype != kFloat || indices_dtype != kLong ||
      weight_sizes.size() != 2 ||
      (indices_sizes.size() != 1 && indices_sizes.size() != 2) ||
      !padding_idx_has_hint || scale_grad_by_freq || sparse) {
    return result;
  }

  const int64_t num_embeddings = weight_sizes[0];
  const int64_t embedding_dim = weight_sizes[1];
  const int64_t num_indices = product_of_sizes(indices_sizes);
  const auto& small_spec = generated::kEmbeddingLookupSmallBoundedLookupSpec;
  result.num_embeddings = num_embeddings;
  result.embedding_dim = embedding_dim;
  result.num_indices = num_indices;

  if (
      num_embeddings == kEmbeddingLookupTokenNumEmbeddings &&
      embedding_dim == kEmbeddingLookupTokenEmbeddingDim &&
      indices_sizes.size() == 2 &&
      indices_sizes[0] == kEmbeddingLookupTokenBatch &&
      indices_sizes[1] >= kEmbeddingLookupTokenMinIndices &&
      indices_sizes[1] <= kEmbeddingLookupTokenMaxIndices) {
    result.matched = true;
    result.family = EmbeddingLookupFamily::TokenBatch1;
    result.tuple_id = kEmbeddingLookupTokenBatch1TupleId;
    result.metadata = &kEmbeddingLookupTokenBatch1Metadata;
    return result;
  }

  if (
      embedding_dim <= small_spec.max_embedding_dim &&
      num_indices <= small_spec.max_num_indices &&
      num_embeddings <= small_spec.max_num_embeddings) {
    result.matched = true;
    result.family = EmbeddingLookupFamily::SmallBoundedLookup;
    result.tuple_id = small_spec.tuple_id;
    result.metadata = &kEmbeddingLookupSmallBoundedMetadata;
    return result;
  }

  return result;
}

bool matches_embedding_lookup_contract(
    const IntArrayRef weight_sizes,
    const IntArrayRef indices_sizes,
    const ScalarType weight_dtype,
    const ScalarType indices_dtype,
    const bool weight_is_vulkan,
    const bool indices_is_vulkan,
    const bool padding_idx_has_hint,
    const bool scale_grad_by_freq,
    const bool sparse) {
  return match_embedding_lookup_contract(
             weight_sizes,
             indices_sizes,
             weight_dtype,
             indices_dtype,
             weight_is_vulkan,
             indices_is_vulkan,
             padding_idx_has_hint,
             scale_grad_by_freq,
             sparse)
      .matched;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
