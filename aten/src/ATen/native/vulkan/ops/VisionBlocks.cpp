#include <ATen/Functions.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/ops/VisionBlocks.h>

#include <cmath>
#include <optional>

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

Tensor move_optional_to_vulkan(const std::optional<Tensor>& tensor) {
  if (!tensor.has_value() || !tensor->defined()) {
    return Tensor();
  }
  return tensor->is_vulkan() ? *tensor : tensor->vulkan();
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

c10::intrusive_ptr<LayernormPackedContext> make_layernorm_context(
    const Tensor& weight,
    const Tensor& bias,
    const double eps) {
  std::optional<Tensor> owned_weight(weight);
  std::optional<Tensor> owned_bias(bias);
  return create_layernorm_context(
      std::move(owned_weight), std::move(owned_bias), eps);
}

c10::intrusive_ptr<LinearPackedContext> make_linear_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const std::string& label) {
  Tensor owned_weight = weight;
  std::optional<Tensor> owned_bias =
      bias.has_value() ? std::optional<Tensor>(*bias) : std::nullopt;
  return create_linear_context_labeled(
      std::move(owned_weight), std::move(owned_bias), label);
}

c10::intrusive_ptr<LinearPackedContext> make_qkv_context(
    const Tensor& weight,
    const std::string& label) {
  Tensor owned_weight = weight;
  std::optional<Tensor> no_bias = std::nullopt;
  return create_linear_context_labeled(
      std::move(owned_weight), std::move(no_bias), label);
}

Tensor maybe_apply_layerscale(const Tensor& input, const Tensor& gamma) {
  if (!gamma.defined()) {
    return input;
  }
  return at::mul(input, gamma);
}

std::tuple<Tensor, Tensor, Tensor> reshape_qkv_for_attention(
    const Tensor& mixed_qkv,
    const int64_t batch_size,
    const int64_t token_count,
    const int64_t num_heads,
    const int64_t head_dim) {
  std::vector<Tensor> qkv = at::chunk(mixed_qkv, 3, 2);
  Tensor q =
      qkv[0].reshape({batch_size, token_count, num_heads, head_dim})
          .permute({0, 2, 1, 3})
          .reshape({batch_size * num_heads, token_count, head_dim});
  Tensor k =
      qkv[1].reshape({batch_size, token_count, num_heads, head_dim})
          .permute({0, 2, 1, 3})
          .reshape({batch_size * num_heads, token_count, head_dim});
  Tensor v =
      qkv[2].reshape({batch_size, token_count, num_heads, head_dim})
          .permute({0, 2, 1, 3})
          .reshape({batch_size * num_heads, token_count, head_dim});
  return std::make_tuple(std::move(q), std::move(k), std::move(v));
}

Tensor run_attention_projection(
    const Tensor& input,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context) {
  TORCH_CHECK(
      input.dim() == 2 || input.dim() == 3,
      "Vision backbone block context expects rank-2 or rank-3 input");

  const bool use_2d_input = input.dim() == 2;
  const int64_t batch_size = use_2d_input ? 1 : input.size(0);
  const int64_t token_count = use_2d_input ? input.size(0) : input.size(1);
  const int64_t embed_dim = input.size(-1);
  TORCH_CHECK(
      embed_dim % context->num_heads() == 0,
      "Vision backbone block context expects embed_dim divisible by num_heads");
  const int64_t head_dim = embed_dim / context->num_heads();

  Tensor attention_output;
  if (batch_size == 1) {
    const Tensor linear_input =
        use_2d_input ? input : input.reshape({token_count, embed_dim});
    Tensor mixed_qkv = run_linear_context(linear_input, context->qkv_context());
    Tensor q;
    Tensor k;
    Tensor v;
    bool q_is_scaled = false;
    if (context->qkv_bias().defined()) {
      std::tie(q, k, v) = at::_transform_bias_rescale_qkv(
          mixed_qkv, context->qkv_bias(), context->num_heads());
      q_is_scaled = true;
    } else {
      std::vector<Tensor> qkv = at::chunk(mixed_qkv, 3, 1);
      q = qkv[0].reshape({token_count, context->num_heads(), head_dim})
              .permute({1, 0, 2});
      k = qkv[1].reshape({token_count, context->num_heads(), head_dim})
              .permute({1, 0, 2});
      v = qkv[2].reshape({token_count, context->num_heads(), head_dim})
              .permute({1, 0, 2});
    }
    if (!q_is_scaled) {
      q = at::mul(
          q,
          static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim))));
    }
    attention_output = at::scaled_dot_product_attention(
        q,
        k,
        v,
        std::nullopt,
        0.0,
        false,
        std::optional<double>(1.0),
        false);
    attention_output = attention_output.permute({1, 0, 2}).reshape(
        {token_count, embed_dim});
    attention_output =
        run_linear_context(attention_output, context->proj_context());
    if (!use_2d_input) {
      attention_output = attention_output.reshape({1, token_count, embed_dim});
    }
    return attention_output;
  }

  Tensor mixed_qkv = run_linear_context(input, context->qkv_context());
  Tensor q;
  Tensor k;
  Tensor v;
  std::tie(q, k, v) = reshape_qkv_for_attention(
      mixed_qkv, batch_size, token_count, context->num_heads(), head_dim);
  q = at::mul(
      q,
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim))));
  attention_output = at::scaled_dot_product_attention(
      q,
      k,
      v,
      std::nullopt,
      0.0,
      false,
      std::optional<double>(1.0),
      false);
  attention_output =
      attention_output.reshape({batch_size, context->num_heads(), token_count, head_dim})
          .permute({0, 2, 1, 3})
          .reshape({batch_size, token_count, embed_dim});
  return run_linear_context(attention_output, context->proj_context());
}

} // namespace

VisionBackboneBlockContext::VisionBackboneBlockContext(
    const Tensor& norm1_weight,
    const Tensor& norm1_bias,
    const double norm1_eps,
    const Tensor& qkv_weight,
    const std::optional<Tensor>& qkv_bias,
    const int64_t num_heads,
    const Tensor& proj_weight,
    const std::optional<Tensor>& proj_bias,
    const std::optional<Tensor>& ls1_gamma,
    const Tensor& norm2_weight,
    const Tensor& norm2_bias,
    const double norm2_eps,
    const Tensor& fc1_weight,
    const std::optional<Tensor>& fc1_bias,
    const Tensor& fc2_weight,
    const std::optional<Tensor>& fc2_bias,
    const std::optional<Tensor>& ls2_gamma,
    std::string allocation_label)
    : allocation_label_(std::move(allocation_label)),
      norm1_context_(make_layernorm_context(norm1_weight, norm1_bias, norm1_eps)),
      qkv_context_(
          make_qkv_context(qkv_weight, child_label(allocation_label_, "qkv"))),
      qkv_bias_(move_optional_to_vulkan(qkv_bias)),
      num_heads_(num_heads),
      proj_context_(make_linear_context(
          proj_weight,
          proj_bias,
          child_label(allocation_label_, "proj"))),
      ls1_gamma_(move_optional_to_vulkan(ls1_gamma)),
      norm2_context_(make_layernorm_context(norm2_weight, norm2_bias, norm2_eps)),
      fc1_context_(
          make_linear_context(fc1_weight, fc1_bias, child_label(allocation_label_, "fc1"))),
      fc2_context_(
          make_linear_context(fc2_weight, fc2_bias, child_label(allocation_label_, "fc2"))),
      ls2_gamma_(move_optional_to_vulkan(ls2_gamma)) {
  unpacked_.reserve(Unpacked::NumArgs);
  unpacked_.emplace_back(norm1_weight.cpu());
  unpacked_.emplace_back(norm1_bias.cpu());
  unpacked_.emplace_back(norm1_eps);
  unpacked_.emplace_back(qkv_weight.cpu());
  if (qkv_bias.has_value()) {
    unpacked_.emplace_back(qkv_bias->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(num_heads_);
  unpacked_.emplace_back(proj_weight.cpu());
  if (proj_bias.has_value()) {
    unpacked_.emplace_back(proj_bias->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  if (ls1_gamma.has_value()) {
    unpacked_.emplace_back(ls1_gamma->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(norm2_weight.cpu());
  unpacked_.emplace_back(norm2_bias.cpu());
  unpacked_.emplace_back(norm2_eps);
  unpacked_.emplace_back(fc1_weight.cpu());
  if (fc1_bias.has_value()) {
    unpacked_.emplace_back(fc1_bias->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(fc2_weight.cpu());
  if (fc2_bias.has_value()) {
    unpacked_.emplace_back(fc2_bias->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  if (ls2_gamma.has_value()) {
    unpacked_.emplace_back(ls2_gamma->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(allocation_label_);
}

VisionBackboneBlockContext VisionBackboneBlockContext::pack(
    c10::impl::GenericList unpacked) {
  return VisionBackboneBlockContext(
      unpacked.get(Unpacked::Norm1Weight).toTensor(),
      unpacked.get(Unpacked::Norm1Bias).toTensor(),
      unpacked.get(Unpacked::Norm1Eps).toDouble(),
      unpacked.get(Unpacked::QkvWeight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::QkvBias),
      unpacked.get(Unpacked::NumHeads).toInt(),
      unpacked.get(Unpacked::ProjWeight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::ProjBias),
      get_optional_tensor(unpacked, Unpacked::Ls1Gamma),
      unpacked.get(Unpacked::Norm2Weight).toTensor(),
      unpacked.get(Unpacked::Norm2Bias).toTensor(),
      unpacked.get(Unpacked::Norm2Eps).toDouble(),
      unpacked.get(Unpacked::Fc1Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Fc1Bias),
      unpacked.get(Unpacked::Fc2Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Fc2Bias),
      get_optional_tensor(unpacked, Unpacked::Ls2Gamma),
      unpacked.get(Unpacked::Label).toStringRef());
}

c10::intrusive_ptr<VisionBackboneBlockContext>
create_vision_backbone_block_context(
    Tensor&& norm1_weight,
    Tensor&& norm1_bias,
    const double norm1_eps,
    Tensor&& qkv_weight,
    std::optional<Tensor>&& qkv_bias,
    const int64_t num_heads,
    Tensor&& proj_weight,
    std::optional<Tensor>&& proj_bias,
    std::optional<Tensor>&& ls1_gamma,
    Tensor&& norm2_weight,
    Tensor&& norm2_bias,
    const double norm2_eps,
    Tensor&& fc1_weight,
    std::optional<Tensor>&& fc1_bias,
    Tensor&& fc2_weight,
    std::optional<Tensor>&& fc2_bias,
    std::optional<Tensor>&& ls2_gamma,
    std::string label) {
  return c10::make_intrusive<VisionBackboneBlockContext>(
      norm1_weight,
      norm1_bias,
      norm1_eps,
      qkv_weight,
      qkv_bias,
      num_heads,
      proj_weight,
      proj_bias,
      ls1_gamma,
      norm2_weight,
      norm2_bias,
      norm2_eps,
      fc1_weight,
      fc1_bias,
      fc2_weight,
      fc2_bias,
      ls2_gamma,
      std::move(label));
}

Tensor run_vision_backbone_block_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context) {
  TORCH_CHECK(
      input_arg.dim() == 2 || input_arg.dim() == 3,
      "Vision backbone block context expects rank-2 or rank-3 input");

  const Device output_device = input_arg.device();
  const ScalarType output_dtype = input_arg.scalar_type();
  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const std::string runtime_label = context->allocation_label().empty()
      ? std::string("depth.dino.backbone.block")
      : context->allocation_label();
  api::AllocationScope allocation_scope(runtime_label);
  api::RuntimeLabelScope runtime_scope(runtime_label);

  const std::array<int64_t, 1> normalized_shape = {input.size(-1)};
  Tensor attention_input =
      run_layernorm_context(input, normalized_shape, context->norm1_context());
  Tensor attention_output = run_attention_projection(attention_input, context);
  attention_output = maybe_apply_layerscale(attention_output, context->ls1_gamma());
  Tensor hidden_states = at::add(input, attention_output);

  Tensor mlp_input =
      run_layernorm_context(hidden_states, normalized_shape, context->norm2_context());
  Tensor mlp_output = run_linear_gelu_context(mlp_input, context->fc1_context());
  mlp_output = run_linear_context(mlp_output, context->fc2_context());
  mlp_output = maybe_apply_layerscale(mlp_output, context->ls2_gamma());

  Tensor output = at::add(hidden_states, mlp_output);
  utils::log_vulkan_op_hit("vulkan_prepack::run_vision_backbone_block_context");
  return maybe_restore_tensor(output, output_device, output_dtype);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
