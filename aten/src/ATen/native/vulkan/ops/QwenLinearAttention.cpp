#include <ATen/Functions.h>
#include <ATen/native/vulkan/ops/GatedDelta.h>
#include <ATen/native/vulkan/ops/QwenLinearAttention.h>
#include <ATen/native/vulkan/ops/Utils.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

std::string child_label(const std::string& label, const char* suffix) {
  if (label.empty()) {
    return std::string(suffix);
  }
  return label + "." + suffix;
}

Tensor move_to_vulkan_float(const Tensor& tensor) {
  return tensor.is_vulkan() ? tensor.to(kFloat) : tensor.vulkan().to(kFloat);
}

c10::intrusive_ptr<LinearPackedContext> make_labeled_linear_context(
    const Tensor& weight,
    const std::string& label) {
  Tensor owned_weight = weight;
  std::optional<Tensor> no_bias = std::nullopt;
  return create_linear_context_labeled(
      std::move(owned_weight), std::move(no_bias), label);
}

c10::intrusive_ptr<Conv1dPackedContext> make_conv1d_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  Tensor owned_weight = weight;
  std::optional<Tensor> owned_bias =
      bias.has_value() ? std::optional<Tensor>(*bias) : std::nullopt;
  return create_conv1d_context(
      std::move(owned_weight),
      std::move(owned_bias),
      {1, 1},
      {static_cast<int64_t>(weight.size(2) - 1),
       static_cast<int64_t>(weight.size(2) - 1)},
      {1, 1},
      weight.size(0));
}

c10::intrusive_ptr<Conv1dPackedContext> make_conv1d_update_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  Tensor owned_weight = weight;
  std::optional<Tensor> owned_bias =
      bias.has_value() ? std::optional<Tensor>(*bias) : std::nullopt;
  return create_conv1d_context(
      std::move(owned_weight),
      std::move(owned_bias),
      {1, 1},
      {0, 0},
      {1, 1},
      weight.size(0));
}

Tensor maybe_restore_tensor(
    const Tensor& tensor,
    const Device& device,
    const ScalarType scalar_type) {
  Tensor restored = device.type() == kVulkan ? tensor : tensor.cpu();
  if (restored.scalar_type() != scalar_type) {
    restored = restored.to(scalar_type);
  }
  return restored;
}

Tensor run_gated_rms_norm(
    const Tensor& hidden_states,
    const Tensor& gate,
    const Tensor& weight,
    const double eps) {
  const Tensor variance =
      at::mean(at::mul(hidden_states, hidden_states), {-1}, true);
  Tensor output =
      at::mul(hidden_states, at::rsqrt(at::add(variance, eps)));
  output = at::mul(weight, output);
  output = at::mul(output, at::silu(gate));
  return output;
}

} // namespace

QwenLinearAttentionPrefillPackedContext::QwenLinearAttentionPrefillPackedContext(
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
    const int64_t key_dim,
    const int64_t value_dim,
    const int64_t head_k_dim,
    const int64_t head_v_dim,
    const int64_t num_k_heads,
    const int64_t num_v_heads,
    const int64_t chunk_size,
    const double norm_eps,
    std::string allocation_label)
    : allocation_label_(std::move(allocation_label)),
      qkv_context_(make_labeled_linear_context(
          qkv_weight, child_label(allocation_label_, "qkv"))),
      z_context_(
          make_labeled_linear_context(z_weight, child_label(allocation_label_, "z"))),
      a_context_(
          make_labeled_linear_context(a_weight, child_label(allocation_label_, "a"))),
      b_context_(
          make_labeled_linear_context(b_weight, child_label(allocation_label_, "b"))),
      out_context_(make_labeled_linear_context(
          out_weight, child_label(allocation_label_, "out"))),
      conv_context_(make_conv1d_context(conv_weight, conv_bias)),
      conv_update_context_(make_conv1d_update_context(conv_weight, conv_bias)),
      norm_weight_(move_to_vulkan_float(norm_weight)),
      A_log_(move_to_vulkan_float(A_log)),
      dt_bias_(move_to_vulkan_float(dt_bias)),
      key_dim_(key_dim),
      value_dim_(value_dim),
      head_k_dim_(head_k_dim),
      head_v_dim_(head_v_dim),
      num_k_heads_(num_k_heads),
      num_v_heads_(num_v_heads),
      chunk_size_(chunk_size),
      norm_eps_(norm_eps) {
  unpacked_.reserve(Unpacked::NumArgs);
  unpacked_.emplace_back(qkv_weight.cpu());
  unpacked_.emplace_back(z_weight.cpu());
  unpacked_.emplace_back(a_weight.cpu());
  unpacked_.emplace_back(b_weight.cpu());
  unpacked_.emplace_back(out_weight.cpu());
  unpacked_.emplace_back(conv_weight.cpu());
  if (conv_bias.has_value()) {
    unpacked_.emplace_back(conv_bias->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(norm_weight.cpu());
  unpacked_.emplace_back(A_log.cpu());
  unpacked_.emplace_back(dt_bias.cpu());
  unpacked_.emplace_back(key_dim_);
  unpacked_.emplace_back(value_dim_);
  unpacked_.emplace_back(head_k_dim_);
  unpacked_.emplace_back(head_v_dim_);
  unpacked_.emplace_back(num_k_heads_);
  unpacked_.emplace_back(num_v_heads_);
  unpacked_.emplace_back(chunk_size_);
  unpacked_.emplace_back(norm_eps_);
  unpacked_.emplace_back(allocation_label_);
}

QwenLinearAttentionPrefillPackedContext
QwenLinearAttentionPrefillPackedContext::pack(c10::impl::GenericList unpacked) {
  return QwenLinearAttentionPrefillPackedContext(
      unpacked.get(Unpacked::QkvWeight).toTensor(),
      unpacked.get(Unpacked::ZWeight).toTensor(),
      unpacked.get(Unpacked::AWeight).toTensor(),
      unpacked.get(Unpacked::BWeight).toTensor(),
      unpacked.get(Unpacked::OutWeight).toTensor(),
      unpacked.get(Unpacked::ConvWeight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::ConvBias),
      unpacked.get(Unpacked::NormWeight).toTensor(),
      unpacked.get(Unpacked::ALog).toTensor(),
      unpacked.get(Unpacked::DtBias).toTensor(),
      unpacked.get(Unpacked::KeyDim).toInt(),
      unpacked.get(Unpacked::ValueDim).toInt(),
      unpacked.get(Unpacked::HeadKDim).toInt(),
      unpacked.get(Unpacked::HeadVDim).toInt(),
      unpacked.get(Unpacked::NumKHeads).toInt(),
      unpacked.get(Unpacked::NumVHeads).toInt(),
      unpacked.get(Unpacked::ChunkSize).toInt(),
      unpacked.get(Unpacked::NormEps).toDouble(),
      unpacked.get(Unpacked::Label).toStringRef());
}

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
    const int64_t key_dim,
    const int64_t value_dim,
    const int64_t head_k_dim,
    const int64_t head_v_dim,
    const int64_t num_k_heads,
    const int64_t num_v_heads,
    const int64_t chunk_size,
    const double norm_eps,
    std::string label) {
  return c10::make_intrusive<QwenLinearAttentionPrefillPackedContext>(
      qkv_weight,
      z_weight,
      a_weight,
      b_weight,
      out_weight,
      conv_weight,
      conv_bias,
      norm_weight,
      A_log,
      dt_bias,
      key_dim,
      value_dim,
      head_k_dim,
      head_v_dim,
      num_k_heads,
      num_v_heads,
      chunk_size,
      norm_eps,
      std::move(label));
}

Tensor run_qwen_linear_attention_prefill_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<QwenLinearAttentionPrefillPackedContext>& context) {
  TORCH_CHECK(
      input_arg.dim() == 3,
      "Qwen linear attention prefill context expects a rank-3 input tensor");

  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  if (input.scalar_type() != kFloat) {
    input = input.to(kFloat);
  }

  const int64_t batch_size = input.size(0);
  const int64_t seq_len = input.size(1);

  Tensor mixed_qkv = run_linear_context(input, context->qkv_context());
  mixed_qkv = at::transpose(mixed_qkv, 1, 2);
  mixed_qkv = run_conv1d_context(mixed_qkv, context->conv_context());
  mixed_qkv = at::silu(at::slice(mixed_qkv, 2, 0, seq_len, 1));
  mixed_qkv = at::transpose(mixed_qkv, 1, 2).contiguous();

  Tensor z = run_linear_context(input, context->z_context()).contiguous();
  Tensor a = run_linear_context(input, context->a_context()).contiguous();
  Tensor b = run_linear_context(input, context->b_context()).contiguous();

  std::vector<Tensor> qkv = at::split_with_sizes(
      mixed_qkv,
      {context->key_dim(), context->key_dim(), context->value_dim()},
      -1);
  Tensor query = qkv[0].reshape(
      {batch_size, seq_len, -1, context->head_k_dim()});
  Tensor key = qkv[1].reshape({batch_size, seq_len, -1, context->head_k_dim()});
  Tensor value =
      qkv[2].reshape({batch_size, seq_len, -1, context->head_v_dim()});

  Tensor beta = at::sigmoid(b);
  Tensor g = at::mul(
      at::neg(at::exp(context->A_log())),
      at::softplus(at::add(a, context->dt_bias())));

  if (context->num_v_heads() / context->num_k_heads() > 1) {
    const int64_t repeat_factor = context->num_v_heads() / context->num_k_heads();
    query = query.unsqueeze(3)
                .expand(
                    {batch_size,
                     seq_len,
                     context->num_k_heads(),
                     repeat_factor,
                     context->head_k_dim()})
                .reshape(
                    {batch_size,
                     seq_len,
                     context->num_v_heads(),
                     context->head_k_dim()});
    key = key.unsqueeze(3)
              .expand(
                  {batch_size,
                   seq_len,
                   context->num_k_heads(),
                   repeat_factor,
                   context->head_k_dim()})
              .reshape(
                  {batch_size,
                   seq_len,
                   context->num_v_heads(),
                   context->head_k_dim()});
  }

  Tensor core_attn_out = std::get<0>(run_scheduled_gated_delta_rule_chunk(
      query,
      key,
      value,
      g,
      beta,
      context->chunk_size(),
      std::nullopt,
      false,
      true));

  const int64_t num_value_heads = core_attn_out.size(2);
  core_attn_out =
      core_attn_out.reshape({batch_size * seq_len * num_value_heads, context->head_v_dim()});
  Tensor gate =
      z.reshape({batch_size * seq_len * num_value_heads, context->head_v_dim()});
  core_attn_out = run_gated_rms_norm(
      core_attn_out, gate, context->norm_weight(), context->norm_eps());
  core_attn_out =
      core_attn_out.reshape({batch_size, seq_len, context->value_dim()});

  Tensor output = run_linear_context(core_attn_out, context->out_context());
  if (!input_arg.is_vulkan()) {
    output = output.cpu();
  }
  if (output.scalar_type() != input_arg.scalar_type()) {
    output = output.to(input_arg.scalar_type());
  }
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_qwen_linear_attention_prefill_context");
  return output;
}

std::tuple<Tensor, Tensor, Tensor> run_qwen_linear_attention_decode_context(
    const Tensor& input_arg,
    const Tensor& conv_state_arg,
    const Tensor& recurrent_state_arg,
    const c10::intrusive_ptr<QwenLinearAttentionPrefillPackedContext>& context) {
  TORCH_CHECK(
      input_arg.dim() == 3 && input_arg.size(1) == 1,
      "Qwen linear attention decode context expects a [B, 1, C] input tensor");
  TORCH_CHECK(
      conv_state_arg.dim() == 3,
      "Qwen linear attention decode context expects a rank-3 conv_state tensor");
  TORCH_CHECK(
      recurrent_state_arg.dim() == 4,
      "Qwen linear attention decode context expects a rank-4 recurrent_state tensor");

  const Device output_device = input_arg.device();
  const ScalarType output_dtype = input_arg.scalar_type();

  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  if (input.scalar_type() != kFloat) {
    input = input.to(kFloat);
  }

  Tensor conv_state = conv_state_arg.is_vulkan() ? conv_state_arg : conv_state_arg.vulkan();
  if (conv_state.scalar_type() != kFloat) {
    conv_state = conv_state.to(kFloat);
  }

  Tensor recurrent_state =
      recurrent_state_arg.is_vulkan() ? recurrent_state_arg : recurrent_state_arg.vulkan();
  if (recurrent_state.scalar_type() != kFloat) {
    recurrent_state = recurrent_state.to(kFloat);
  }

  const int64_t batch_size = input.size(0);
  const int64_t seq_len = input.size(1);
  const int64_t conv_state_len = conv_state.size(2);

  Tensor mixed_qkv = run_linear_context(input, context->qkv_context());
  mixed_qkv = at::transpose(mixed_qkv, 1, 2);
  Tensor conv_input = at::cat({conv_state, mixed_qkv}, 2);
  Tensor next_conv_state =
      at::slice(conv_input, 2, conv_input.size(2) - conv_state_len, conv_input.size(2), 1)
          .contiguous();
  mixed_qkv = run_conv1d_context(conv_input, context->conv_update_context());
  mixed_qkv = at::silu(
      at::slice(mixed_qkv, 2, mixed_qkv.size(2) - seq_len, mixed_qkv.size(2), 1));
  mixed_qkv = at::transpose(mixed_qkv, 1, 2).contiguous();

  Tensor z = run_linear_context(input, context->z_context()).contiguous();
  Tensor a = run_linear_context(input, context->a_context()).contiguous();
  Tensor b = run_linear_context(input, context->b_context()).contiguous();

  std::vector<Tensor> qkv = at::split_with_sizes(
      mixed_qkv,
      {context->key_dim(), context->key_dim(), context->value_dim()},
      -1);
  Tensor query = qkv[0].reshape(
      {batch_size, seq_len, -1, context->head_k_dim()});
  Tensor key = qkv[1].reshape({batch_size, seq_len, -1, context->head_k_dim()});
  Tensor value =
      qkv[2].reshape({batch_size, seq_len, -1, context->head_v_dim()});

  Tensor beta = at::sigmoid(b);
  Tensor g = at::mul(
      at::neg(at::exp(context->A_log())),
      at::softplus(at::add(a, context->dt_bias())));

  if (context->num_v_heads() / context->num_k_heads() > 1) {
    const int64_t repeat_factor = context->num_v_heads() / context->num_k_heads();
    query = query.unsqueeze(3)
                .expand(
                    {batch_size,
                     seq_len,
                     context->num_k_heads(),
                     repeat_factor,
                     context->head_k_dim()})
                .reshape(
                    {batch_size,
                     seq_len,
                     context->num_v_heads(),
                     context->head_k_dim()});
    key = key.unsqueeze(3)
              .expand(
                  {batch_size,
                   seq_len,
                   context->num_k_heads(),
                   repeat_factor,
                   context->head_k_dim()})
              .reshape(
                  {batch_size,
                   seq_len,
                   context->num_v_heads(),
                   context->head_k_dim()});
  }

  auto recurrent_out = run_scheduled_gated_delta_rule_recurrent(
      query,
      key,
      value,
      g,
      beta,
      recurrent_state,
      true,
      true);
  Tensor core_attn_out = std::get<0>(recurrent_out);
  Tensor next_recurrent_state = *std::get<1>(recurrent_out);

  const int64_t num_value_heads = core_attn_out.size(2);
  core_attn_out =
      core_attn_out.reshape({batch_size * seq_len * num_value_heads, context->head_v_dim()});
  Tensor gate =
      z.reshape({batch_size * seq_len * num_value_heads, context->head_v_dim()});
  core_attn_out = run_gated_rms_norm(
      core_attn_out, gate, context->norm_weight(), context->norm_eps());
  core_attn_out =
      core_attn_out.reshape({batch_size, seq_len, context->value_dim()});

  Tensor output = run_linear_context(core_attn_out, context->out_context());
  output = maybe_restore_tensor(output, output_device, output_dtype);
  next_conv_state = maybe_restore_tensor(next_conv_state, output_device, kFloat);
  next_recurrent_state =
      maybe_restore_tensor(next_recurrent_state, output_device, kFloat);
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_qwen_linear_attention_decode_context");
  return {output, next_conv_state, next_recurrent_state};
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
