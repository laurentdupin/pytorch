#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

#include <optional>
#include <string_view>
#include <utility>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class LayernormPackedContext final : public torch::jit::CustomClassHolder {
 private:
 Tensor weight_;
  Tensor bias_;
  double eps_{0.0};
  std::string allocation_label_;

 public:
  LayernormPackedContext(
      const std::optional<Tensor>& weight,
      const std::optional<Tensor>& bias,
      double eps,
      std::string allocation_label = "");

  /*
   * Assigns a name to each index in the unpacked list.
   */
  struct ListArgs final {
    static constexpr uint32_t kWeight = 0u;
    static constexpr uint32_t kBias = 1u;
    static constexpr uint32_t kEps = 2u;
    static constexpr uint32_t kLabel = 3u;

    static constexpr uint32_t kNumArgs = 4u;
  };

  static LayernormPackedContext pack(const c10::impl::GenericList);

  const c10::impl::GenericList unpack() const;

  const Tensor& weight() const {
    return weight_;
  }

  const Tensor& bias() const {
    return bias_;
  }

  double eps() const {
    return eps_;
  }

  const std::string& allocation_label() const {
    return allocation_label_;
  }
};

c10::intrusive_ptr<LayernormPackedContext> create_layernorm_context(
    std::optional<Tensor>&& weight,
    std::optional<Tensor>&& bias,
    double eps);

c10::intrusive_ptr<LayernormPackedContext> create_layernorm_context_labeled(
    std::optional<Tensor>&& weight,
    std::optional<Tensor>&& bias,
    double eps,
    std::string label);

Tensor layer_norm_impl(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    double eps);

Tensor run_layernorm_context(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& context);

Tensor run_layernorm_context_out(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& context,
    Tensor& output);

std::optional<std::pair<Tensor, Tensor>> try_run_add_layernorm_context_out(
    const Tensor& residual,
    const Tensor& addend,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& context,
    Tensor& residual_output,
    Tensor& norm_output);

std::optional<std::pair<Tensor, Tensor>>
try_run_add_scaled_layernorm_context_out(
    const Tensor& residual,
    const Tensor& addend,
    const Tensor& scale,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& context,
    Tensor& residual_output,
    Tensor& norm_output);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
