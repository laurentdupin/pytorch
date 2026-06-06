#include <ATen/native/vulkan/planning/ExecutionContracts.h>

#include <cmath>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

struct DiffusionSDPATuple final {
  DiffusionSDPAFamily family;
  int64_t heads;
  int64_t query_sequence;
  int64_t key_value_sequence;
  int64_t head_dim;
  const char* tuple_id;
  ExecutionContractMetadata metadata;
};

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
constexpr const char* kMaterializationDelegatedToSDPAExecutionPolicy =
    "delegated_to_sdpa_execution_policy";
constexpr double kHeadDim64Scale = 0.125;
constexpr double kHeadDim512Scale = 0.04419417382415922;

#define DIFFUSION_SDPA_TUPLE(                                      \
    FAMILY, HEADS, QUERY_SEQUENCE, KEY_VALUE_SEQUENCE, DIM, TUPLE_ID) \
  {                                                                \
      DiffusionSDPAFamily::FAMILY,                                 \
      HEADS,                                                       \
      QUERY_SEQUENCE,                                              \
      KEY_VALUE_SEQUENCE,                                          \
      DIM,                                                         \
      TUPLE_ID,                                                    \
      make_execution_contract_metadata(                            \
          "DiffusionSDPAContract",                                 \
          #FAMILY,                                                 \
          TUPLE_ID,                                                \
          "diffusion_sdpa_focused_tests",                          \
          "diffusion_sdpa_adjacent_guards",                        \
          kFallbackUnsupportedShapesDoNotMatch,                    \
          kMaterializationDelegatedToSDPAExecutionPolicy)}

constexpr DiffusionSDPATuple kDiffusionSDPATuples[] = {
    DIFFUSION_SDPA_TUPLE(
        SquareSelfAttention, 1, 640, 640, 512, "square_heads1_sequence640_dim512"),
    DIFFUSION_SDPA_TUPLE(
        SquareSelfAttention, 5, 640, 640, 64, "square_heads5_sequence640_dim64"),
    DIFFUSION_SDPA_TUPLE(
        SquareSelfAttention, 1, 504, 504, 512, "square_heads1_sequence504_dim512"),
    DIFFUSION_SDPA_TUPLE(
        SquareSelfAttention, 5, 504, 504, 64, "square_heads5_sequence504_dim64"),
    DIFFUSION_SDPA_TUPLE(
        SquareSelfAttention, 10, 126, 126, 64, "square_heads10_sequence126_dim64"),
    DIFFUSION_SDPA_TUPLE(
        SquareSelfAttention, 20, 35, 35, 64, "square_heads20_sequence35_dim64"),
    DIFFUSION_SDPA_TUPLE(
        SquareSelfAttention, 20, 12, 12, 64, "square_heads20_sequence12_dim64"),
    DIFFUSION_SDPA_TUPLE(
        CrossAttention, 5, 504, 2, 64, "cross_heads5_query504_kv2_dim64"),
    DIFFUSION_SDPA_TUPLE(
        CrossAttention, 10, 126, 2, 64, "cross_heads10_query126_kv2_dim64"),
    DIFFUSION_SDPA_TUPLE(
        CrossAttention, 20, 35, 2, 64, "cross_heads20_query35_kv2_dim64"),
    DIFFUSION_SDPA_TUPLE(
        CrossAttention, 20, 12, 2, 64, "cross_heads20_query12_kv2_dim64"),
};

#undef DIFFUSION_SDPA_TUPLE

} // namespace

const char* diffusion_sdpa_route_label(const DiffusionSDPAFamily family) {
  switch (family) {
    case DiffusionSDPAFamily::SquareSelfAttention:
      return "SelectedDiffusionSDPASquareSelfAttention";
    case DiffusionSDPAFamily::CrossAttention:
      return "SelectedDiffusionSDPACrossAttention";
    case DiffusionSDPAFamily::None:
      return "SelectedDiffusionSDPANone";
  }
  return "SelectedDiffusionSDPANone";
}

DiffusionSDPAMatch match_diffusion_sdpa_contract(
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes,
    const IntArrayRef value_sizes,
    const ScalarType query_dtype,
    const ScalarType key_dtype,
    const ScalarType value_dtype,
    const bool has_attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa) {
  DiffusionSDPAMatch result;
  if (
      has_attn_mask || dropout_p != 0.0 || is_causal || enable_gqa ||
      query_dtype != kFloat || key_dtype != kFloat || value_dtype != kFloat ||
      query_sizes.size() != 4 || key_sizes.size() != 4 ||
      value_sizes.size() != 4) {
    return result;
  }
  if (
      query_sizes[0] != 1 || key_sizes[0] != 1 || value_sizes[0] != 1 ||
      query_sizes[1] != key_sizes[1] || query_sizes[1] != value_sizes[1] ||
      key_sizes[2] != value_sizes[2] || query_sizes[3] != key_sizes[3] ||
      query_sizes[3] != value_sizes[3]) {
    return result;
  }

  const int64_t heads = query_sizes[1];
  const int64_t query_sequence = query_sizes[2];
  const int64_t key_value_sequence = key_sizes[2];
  const int64_t head_dim = query_sizes[3];
  for (const DiffusionSDPATuple& tuple : kDiffusionSDPATuples) {
    if (
        heads != tuple.heads ||
        query_sequence != tuple.query_sequence ||
        key_value_sequence != tuple.key_value_sequence ||
        head_dim != tuple.head_dim) {
      continue;
    }
    if (scale.has_value()) {
      const double expected_scale =
          head_dim == 512 ? kHeadDim512Scale : kHeadDim64Scale;
      if (std::abs(*scale - expected_scale) > 1.0e-6) {
        return result;
      }
    }
    result.matched = true;
    result.family = tuple.family;
    result.tuple_id = tuple.tuple_id;
    result.metadata = &tuple.metadata;
    return result;
  }
  return result;
}

bool matches_diffusion_sdpa_contract(
    const IntArrayRef query_sizes,
    const IntArrayRef key_sizes,
    const IntArrayRef value_sizes,
    const ScalarType query_dtype,
    const ScalarType key_dtype,
    const ScalarType value_dtype,
    const bool has_attn_mask,
    const double dropout_p,
    const bool is_causal,
    const std::optional<double> scale,
    const bool enable_gqa) {
  return match_diffusion_sdpa_contract(
             query_sizes,
             key_sizes,
             value_sizes,
             query_dtype,
             key_dtype,
             value_dtype,
             has_attn_mask,
             dropout_p,
             is_causal,
             scale,
             enable_gqa)
      .matched;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
