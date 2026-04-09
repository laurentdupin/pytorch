#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <torch/library.h>

#include <optional>
#include <string>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class VisionBackboneBlockContext final : public torch::jit::CustomClassHolder {
 private:
  c10::impl::GenericList unpacked_{c10::AnyType::get()};
  std::string allocation_label_;
  c10::intrusive_ptr<LayernormPackedContext> norm1_context_;
  c10::intrusive_ptr<LinearPackedContext> qkv_context_;
  Tensor qkv_bias_;
  int64_t num_heads_{0};
  c10::intrusive_ptr<LinearPackedContext> proj_context_;
  Tensor ls1_gamma_;
  c10::intrusive_ptr<LayernormPackedContext> norm2_context_;
  c10::intrusive_ptr<LinearPackedContext> fc1_context_;
  c10::intrusive_ptr<LinearPackedContext> fc2_context_;
  Tensor ls2_gamma_;

 public:
  VisionBackboneBlockContext(
      const Tensor& norm1_weight,
      const Tensor& norm1_bias,
      double norm1_eps,
      const Tensor& qkv_weight,
      const std::optional<Tensor>& qkv_bias,
      int64_t num_heads,
      const Tensor& proj_weight,
      const std::optional<Tensor>& proj_bias,
      const std::optional<Tensor>& ls1_gamma,
      const Tensor& norm2_weight,
      const Tensor& norm2_bias,
      double norm2_eps,
      const Tensor& fc1_weight,
      const std::optional<Tensor>& fc1_bias,
      const Tensor& fc2_weight,
      const std::optional<Tensor>& fc2_bias,
      const std::optional<Tensor>& ls2_gamma,
      std::string allocation_label = std::string());

  struct Unpacked final {
    static constexpr uint32_t Norm1Weight = 0u;
    static constexpr uint32_t Norm1Bias = 1u;
    static constexpr uint32_t Norm1Eps = 2u;
    static constexpr uint32_t QkvWeight = 3u;
    static constexpr uint32_t QkvBias = 4u;
    static constexpr uint32_t NumHeads = 5u;
    static constexpr uint32_t ProjWeight = 6u;
    static constexpr uint32_t ProjBias = 7u;
    static constexpr uint32_t Ls1Gamma = 8u;
    static constexpr uint32_t Norm2Weight = 9u;
    static constexpr uint32_t Norm2Bias = 10u;
    static constexpr uint32_t Norm2Eps = 11u;
    static constexpr uint32_t Fc1Weight = 12u;
    static constexpr uint32_t Fc1Bias = 13u;
    static constexpr uint32_t Fc2Weight = 14u;
    static constexpr uint32_t Fc2Bias = 15u;
    static constexpr uint32_t Ls2Gamma = 16u;
    static constexpr uint32_t Label = 17u;
    static constexpr uint32_t NumArgs = 18u;
  };

  static VisionBackboneBlockContext pack(c10::impl::GenericList unpacked);

  const c10::impl::GenericList unpack() const {
    return unpacked_;
  }

  const std::string& allocation_label() const {
    return allocation_label_;
  }

  const c10::intrusive_ptr<LayernormPackedContext>& norm1_context() const {
    return norm1_context_;
  }

  const c10::intrusive_ptr<LinearPackedContext>& qkv_context() const {
    return qkv_context_;
  }

  const Tensor& qkv_bias() const {
    return qkv_bias_;
  }

  int64_t num_heads() const {
    return num_heads_;
  }

  const c10::intrusive_ptr<LinearPackedContext>& proj_context() const {
    return proj_context_;
  }

  const Tensor& ls1_gamma() const {
    return ls1_gamma_;
  }

  const c10::intrusive_ptr<LayernormPackedContext>& norm2_context() const {
    return norm2_context_;
  }

  const c10::intrusive_ptr<LinearPackedContext>& fc1_context() const {
    return fc1_context_;
  }

  const c10::intrusive_ptr<LinearPackedContext>& fc2_context() const {
    return fc2_context_;
  }

  const Tensor& ls2_gamma() const {
    return ls2_gamma_;
  }
};

c10::intrusive_ptr<VisionBackboneBlockContext>
create_vision_backbone_block_context(
    Tensor&& norm1_weight,
    Tensor&& norm1_bias,
    double norm1_eps,
    Tensor&& qkv_weight,
    std::optional<Tensor>&& qkv_bias,
    int64_t num_heads,
    Tensor&& proj_weight,
    std::optional<Tensor>&& proj_bias,
    std::optional<Tensor>&& ls1_gamma,
    Tensor&& norm2_weight,
    Tensor&& norm2_bias,
    double norm2_eps,
    Tensor&& fc1_weight,
    std::optional<Tensor>&& fc1_bias,
    Tensor&& fc2_weight,
    std::optional<Tensor>&& fc2_bias,
    std::optional<Tensor>&& ls2_gamma,
    std::string label);

Tensor run_vision_backbone_block_context(
    const Tensor& input,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
