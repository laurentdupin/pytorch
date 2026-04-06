#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

std::tuple<Tensor, std::optional<Tensor>> run_gated_delta_rule_chunk_fallback(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    int64_t chunk_size = 64,
    const std::optional<Tensor>& initial_state = std::nullopt,
    bool output_final_state = false,
    bool use_qk_l2norm_in_kernel = false);

std::tuple<Tensor, std::optional<Tensor>>
run_gated_delta_rule_recurrent_fallback(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const std::optional<Tensor>& initial_state = std::nullopt,
    bool output_final_state = false,
    bool use_qk_l2norm_in_kernel = false);

std::tuple<Tensor, std::optional<Tensor>> run_scheduled_gated_delta_rule_chunk(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    int64_t chunk_size = 64,
    const std::optional<Tensor>& initial_state = std::nullopt,
    bool output_final_state = false,
    bool use_qk_l2norm_in_kernel = false);

std::tuple<Tensor, std::optional<Tensor>>
run_scheduled_gated_delta_rule_recurrent(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const std::optional<Tensor>& initial_state = std::nullopt,
    bool output_final_state = false,
    bool use_qk_l2norm_in_kernel = false);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
