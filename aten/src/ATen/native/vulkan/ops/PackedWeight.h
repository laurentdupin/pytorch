#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <cstdint>
#include <memory>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

enum class PackedWeightKind : uint8_t {
  Unknown = 0u,
  Linear,
  Conv1d,
  Conv2dDepthwise,
  Conv2dPointwise,
  Conv2dSlidingWindow,
};

enum class PackedWeightResidencyClass : uint8_t {
  Transient = 0u,
  PersistentInference,
};

inline const char* to_string(const PackedWeightKind kind) {
  switch (kind) {
    case PackedWeightKind::Unknown:
      return "Unknown";
    case PackedWeightKind::Linear:
      return "Linear";
    case PackedWeightKind::Conv1d:
      return "Conv1d";
    case PackedWeightKind::Conv2dDepthwise:
      return "Conv2dDepthwise";
    case PackedWeightKind::Conv2dPointwise:
      return "Conv2dPointwise";
    case PackedWeightKind::Conv2dSlidingWindow:
      return "Conv2dSlidingWindow";
  }
  return "Unknown";
}

inline const char* to_string(const PackedWeightResidencyClass residency_class) {
  switch (residency_class) {
    case PackedWeightResidencyClass::Transient:
      return "Transient";
    case PackedWeightResidencyClass::PersistentInference:
      return "PersistentInference";
  }
  return "Transient";
}

class PackedWeightHandle final {
 private:
  struct State final {
    Tensor weight_;
    Tensor bias_;
    std::vector<int64_t> logical_weight_sizes_;
    PackedWeightKind kind_{PackedWeightKind::Unknown};
    PackedWeightResidencyClass residency_class_{
        PackedWeightResidencyClass::PersistentInference};
    api::ExecutionLayout execution_layout_{
        api::ExecutionLayout::PACKED_WEIGHT};
    size_t resident_nbytes_{0u};
    bool bias_defined_{false};
    bool quantized_{false};

    State(
        Tensor weight,
        Tensor bias,
        std::vector<int64_t> logical_weight_sizes,
        const PackedWeightKind kind,
        const bool bias_defined,
        const PackedWeightResidencyClass residency_class,
        const bool quantized,
        const api::ExecutionLayout execution_layout,
        const size_t resident_nbytes)
        : weight_(std::move(weight)),
          bias_(std::move(bias)),
          logical_weight_sizes_(std::move(logical_weight_sizes)),
          kind_(kind),
          residency_class_(residency_class),
          execution_layout_(execution_layout),
          resident_nbytes_(resident_nbytes),
          bias_defined_(bias_defined),
          quantized_(quantized) {}
  };

  std::shared_ptr<const State> state_;

 public:
  PackedWeightHandle() = default;

  PackedWeightHandle(
      Tensor weight,
      Tensor bias,
      std::vector<int64_t> logical_weight_sizes,
      const PackedWeightKind kind,
      const bool bias_defined,
      const PackedWeightResidencyClass residency_class =
          PackedWeightResidencyClass::PersistentInference,
      const bool quantized = false,
      const api::ExecutionLayout execution_layout =
          api::ExecutionLayout::PACKED_WEIGHT,
      const size_t resident_nbytes = 0u)
      : state_(std::make_shared<const State>(
            std::move(weight),
            std::move(bias),
            std::move(logical_weight_sizes),
            kind,
            bias_defined,
            residency_class,
            quantized,
            execution_layout,
            resident_nbytes)) {}

  bool defined() const {
    return state_ && state_->weight_.defined();
  }

  const Tensor& weight() const {
    TORCH_CHECK(state_, "Packed weight handle is not initialized");
    return state_->weight_;
  }

  const Tensor& bias() const {
    TORCH_CHECK(state_, "Packed weight handle is not initialized");
    return state_->bias_;
  }

  const std::vector<int64_t>& logical_weight_sizes() const {
    TORCH_CHECK(state_, "Packed weight handle is not initialized");
    return state_->logical_weight_sizes_;
  }

  PackedWeightKind kind() const {
    TORCH_CHECK(state_, "Packed weight handle is not initialized");
    return state_->kind_;
  }

  PackedWeightResidencyClass residency_class() const {
    TORCH_CHECK(state_, "Packed weight handle is not initialized");
    return state_->residency_class_;
  }

  api::ExecutionLayout execution_layout() const {
    TORCH_CHECK(state_, "Packed weight handle is not initialized");
    return state_->execution_layout_;
  }

  size_t resident_nbytes() const {
    TORCH_CHECK(state_, "Packed weight handle is not initialized");
    return state_->resident_nbytes_;
  }

  bool has_bias() const {
    return state_ && state_->bias_defined_;
  }

  bool quantized() const {
    return state_ && state_->quantized_;
  }

  const void* identity() const {
    return state_.get();
  }

  const vTensor& weight_vtensor() const {
    TORCH_CHECK(
        weight().defined() && weight().is_vulkan(),
        "Packed Vulkan weight tensor must be defined and resident on Vulkan");
    return convert(weight());
  }

  const vTensor& bias_vtensor() const {
    TORCH_CHECK(
        bias().defined() && bias().is_vulkan(),
        "Packed Vulkan bias tensor must be defined and resident on Vulkan");
    return convert(bias());
  }
};

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
