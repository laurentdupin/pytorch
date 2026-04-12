#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <torch/library.h>
#include <string>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class QwenLinearAttentionPrefillPackedContext final
    : public torch::jit::CustomClassHolder {
 private:
  c10::impl::GenericList unpacked_{c10::AnyType::get()};
  std::string allocation_label_;
  c10::intrusive_ptr<LinearPackedContext> qkv_context_;
  c10::intrusive_ptr<LinearPackedContext> z_context_;
  c10::intrusive_ptr<LinearPackedContext> a_context_;
  c10::intrusive_ptr<LinearPackedContext> b_context_;
  c10::intrusive_ptr<LinearPackedContext> out_context_;
  c10::intrusive_ptr<Conv1dPackedContext> conv_context_;
  c10::intrusive_ptr<Conv1dPackedContext> conv_update_context_;
  Tensor norm_weight_cpu_;
  Tensor A_log_cpu_;
  Tensor dt_bias_cpu_;
  Tensor norm_weight_;
  Tensor A_log_;
  Tensor dt_bias_;
  int64_t key_dim_{0};
  int64_t value_dim_{0};
  int64_t head_k_dim_{0};
  int64_t head_v_dim_{0};
  int64_t num_k_heads_{0};
  int64_t num_v_heads_{0};
  int64_t chunk_size_{64};
  double norm_eps_{1.0e-6};

 public:
  QwenLinearAttentionPrefillPackedContext(
      const Tensor& qkv_weight,
      const Tensor& z_weight,
      const Tensor& a_weight,
      const Tensor& b_weight,
      const Tensor& out_weight,
      const Tensor& conv_weight,
      const std::optional<Tensor>& conv_bias,
      const Tensor& norm_weight,
      const Tensor& A_log,
      const Tensor& dt_bias,
      int64_t key_dim,
      int64_t value_dim,
      int64_t head_k_dim,
      int64_t head_v_dim,
      int64_t num_k_heads,
      int64_t num_v_heads,
      int64_t chunk_size,
      double norm_eps,
      std::string allocation_label = std::string());

  struct Unpacked final {
    static constexpr uint32_t QkvWeight = 0u;
    static constexpr uint32_t ZWeight = 1u;
    static constexpr uint32_t AWeight = 2u;
    static constexpr uint32_t BWeight = 3u;
    static constexpr uint32_t OutWeight = 4u;
    static constexpr uint32_t ConvWeight = 5u;
    static constexpr uint32_t ConvBias = 6u;
    static constexpr uint32_t NormWeight = 7u;
    static constexpr uint32_t ALog = 8u;
    static constexpr uint32_t DtBias = 9u;
    static constexpr uint32_t KeyDim = 10u;
    static constexpr uint32_t ValueDim = 11u;
    static constexpr uint32_t HeadKDim = 12u;
    static constexpr uint32_t HeadVDim = 13u;
    static constexpr uint32_t NumKHeads = 14u;
    static constexpr uint32_t NumVHeads = 15u;
    static constexpr uint32_t ChunkSize = 16u;
    static constexpr uint32_t NormEps = 17u;
    static constexpr uint32_t Label = 18u;
    static constexpr uint32_t NumArgs = 19u;
  };

  static QwenLinearAttentionPrefillPackedContext pack(
      c10::impl::GenericList unpacked);

  const c10::impl::GenericList unpack() const {
    return unpacked_;
  }

  const std::string& allocation_label() const {
    return allocation_label_;
  }

  const c10::intrusive_ptr<LinearPackedContext>& qkv_context() const {
    return qkv_context_;
  }

  const c10::intrusive_ptr<LinearPackedContext>& z_context() const {
    return z_context_;
  }

  const c10::intrusive_ptr<LinearPackedContext>& a_context() const {
    return a_context_;
  }

  const c10::intrusive_ptr<LinearPackedContext>& b_context() const {
    return b_context_;
  }

  const c10::intrusive_ptr<LinearPackedContext>& out_context() const {
    return out_context_;
  }

  const c10::intrusive_ptr<Conv1dPackedContext>& conv_context() const {
    return conv_context_;
  }

  const c10::intrusive_ptr<Conv1dPackedContext>& conv_update_context() const {
    return conv_update_context_;
  }

  const Tensor& norm_weight() const {
    return norm_weight_;
  }

  const Tensor& norm_weight_cpu() const {
    return norm_weight_cpu_;
  }

  const Tensor& A_log() const {
    return A_log_;
  }

  const Tensor& A_log_cpu() const {
    return A_log_cpu_;
  }

  const Tensor& dt_bias() const {
    return dt_bias_;
  }

  const Tensor& dt_bias_cpu() const {
    return dt_bias_cpu_;
  }

  int64_t key_dim() const {
    return key_dim_;
  }

  int64_t value_dim() const {
    return value_dim_;
  }

  int64_t head_k_dim() const {
    return head_k_dim_;
  }

  int64_t head_v_dim() const {
    return head_v_dim_;
  }

  int64_t num_k_heads() const {
    return num_k_heads_;
  }

  int64_t num_v_heads() const {
    return num_v_heads_;
  }

  int64_t chunk_size() const {
    return chunk_size_;
  }

  double norm_eps() const {
    return norm_eps_;
  }
};

c10::intrusive_ptr<QwenLinearAttentionPrefillPackedContext>
create_qwen_linear_attention_prefill_context(
    Tensor&& qkv_weight,
    Tensor&& z_weight,
    Tensor&& a_weight,
    Tensor&& b_weight,
    Tensor&& out_weight,
    Tensor&& conv_weight,
    std::optional<Tensor>&& conv_bias,
    Tensor&& norm_weight,
    Tensor&& A_log,
    Tensor&& dt_bias,
    int64_t key_dim,
    int64_t value_dim,
    int64_t head_k_dim,
    int64_t head_v_dim,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t chunk_size,
    double norm_eps,
    std::string label);

Tensor run_qwen_linear_attention_prefill_context(
    const Tensor& input,
    const c10::intrusive_ptr<QwenLinearAttentionPrefillPackedContext>& context);

std::tuple<Tensor, Tensor, Tensor> run_qwen_linear_attention_decode_context(
    const Tensor& input,
    const Tensor& conv_state,
    const Tensor& recurrent_state,
    const c10::intrusive_ptr<QwenLinearAttentionPrefillPackedContext>& context);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
