#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <optional>
#include <tuple>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

std::tuple<Tensor, Tensor, Tensor> transform_bias_rescale_qkv_vulkan_out(
    const Tensor& qkv,
    const Tensor& qkv_bias,
    int64_t num_head,
    const Tensor& q_out,
    const Tensor& k_out,
    const Tensor& v_out);

Tensor softmax_buffer_lastdim_out_vulkan(
    const Tensor& input,
    Tensor& output);

Tensor scaled_dot_product_attention_vulkan(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa);

Tensor run_attention_runtime_buffer_math_program_bridge(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value);

Tensor run_attention_runtime_buffer_math_replay_bridge(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value);

std::optional<Tensor> try_start_decomposed_attention_scores(
    const Tensor& query,
    const Tensor& key_t);

std::optional<Tensor> try_propagate_decomposed_attention_softmax(
    const Tensor& input,
    int64_t dim);

std::optional<Tensor> try_consume_decomposed_attention_probs(
    const Tensor& probs,
    const Tensor& value);

Tensor materialize_decomposed_attention_candidate_if_needed(
    const Tensor& tensor);

std::optional<Tensor> try_start_deferred_attention_query_scale(
    const Tensor& query,
    const Scalar& scale);

Tensor materialize_deferred_attention_query_scale_candidate_if_needed(
    const Tensor& tensor);

void move_decomposed_attention_candidate_to_alias(
    const Tensor& source,
    const Tensor& alias);

void move_deferred_attention_query_scale_candidate_to_alias(
    const Tensor& source,
    const Tensor& alias);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
