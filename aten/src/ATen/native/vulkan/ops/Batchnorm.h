#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class BatchNormPackedContext final : public torch::jit::CustomClassHolder {
 private:
  Tensor weight_;
  Tensor bias_;
  Tensor running_mean_;
  Tensor running_var_;
  double eps_{0.0};

 public:
  BatchNormPackedContext(
      const std::optional<Tensor>& weight_opt,
      const std::optional<Tensor>& bias_opt,
      const std::optional<Tensor>& running_mean_opt,
      const std::optional<Tensor>& running_var_opt,
      double eps);

  /*
   * Assigns a name to each index in the packed/unpacked list.
   */
  struct ListArgs final {
    static constexpr uint32_t kWeight = 0u;
    static constexpr uint32_t kBias = 1u;
    static constexpr uint32_t kRunningMean = 2u;
    static constexpr uint32_t kRunningVar = 3u;
    static constexpr uint32_t kEps = 4u;

    static constexpr uint32_t kNumArgs = 5u;
  };

  static BatchNormPackedContext pack(c10::impl::GenericList);

  const c10::impl::GenericList unpack() const;

  const Tensor& weight() const {
    return weight_;
  }

  const Tensor& bias() const {
    return bias_;
  }

  const Tensor& running_mean() const {
    return running_mean_;
  }

  const Tensor& running_var() const {
    return running_var_;
  }

  double eps() const {
    return eps_;
  }
};

c10::intrusive_ptr<BatchNormPackedContext> create_batchnorm_context(
    std::optional<Tensor>&& weight_opt,
    std::optional<Tensor>&& bias_opt,
    std::optional<Tensor>&& running_mean_opt,
    std::optional<Tensor>&& running_var_opt,
    bool training,
    double /* momentum */,
    double eps,
    bool /* cudnn_enable, deprecated */);

Tensor run_batchnorm_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<BatchNormPackedContext>& context);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
