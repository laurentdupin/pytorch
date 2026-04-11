#include <ATen/Functions.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/ops/VisionBlocks.h>
#include <ATen/native/vulkan/planning/ExecutionPrograms.h>
#include <ATen/native/vulkan/planning/Runtime.h>

#include <algorithm>
#include <array>
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

Tensor move_optional_to_vulkan_buffer(const std::optional<Tensor>& tensor) {
  if (!tensor.has_value() || !tensor->defined()) {
    return Tensor();
  }
  Tensor vulkan_tensor = tensor->is_vulkan() ? *tensor : tensor->vulkan();
  return utils::mark_tensor_execution(
      utils::ensure_buffer_storage(
          vulkan_tensor, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
      api::ExecutionLayout::BUFFER_DIRECT,
      true);
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
    const double eps,
    const std::string& label) {
  std::optional<Tensor> owned_weight(weight);
  std::optional<Tensor> owned_bias(bias);
  return create_layernorm_context_labeled(
      std::move(owned_weight), std::move(owned_bias), eps, label);
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

int64_t vision_block_hidden_dim(
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context) {
  return context->unpack()
      .get(VisionBackboneBlockContext::Unpacked::Fc1Weight)
      .toTensor()
      .size(0);
}

std::string vision_backbone_program_base_label(const std::string& label) {
  if (label.empty()) {
    return "depth.dino.backbone.block";
  }

  constexpr const char* kDynamicBlockMarker = ".block.";
  const auto marker_pos = label.find(kDynamicBlockMarker);
  if (marker_pos != std::string::npos) {
    return label.substr(0, marker_pos + 6u);
  }

  return label;
}

std::string vision_backbone_program_label(const std::string& label) {
  return vision_backbone_program_base_label(label) + ".program";
}

utils::VisionBackboneProgram prime_vision_backbone_program(
    const Tensor& input,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context) {
  const auto request = utils::make_vulkan_planning_request(
      utils::VulkanWorkloadClass::VisionBackbone,
      utils::VulkanTensorRole::Input,
      utils::VulkanModelDomain::Vision,
      utils::VulkanExecutionPhase::Backbone);
  const auto runtime_policy = utils::build_vulkan_runtime_policy(request);
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionBackbone) {
    return {};
  }

  const int64_t batch_size = input.dim() == 2 ? 1 : input.size(0);
  const int64_t token_count = input.dim() == 2 ? input.size(0) : input.size(1);
  const int64_t embed_dim = input.size(-1);
  const int64_t hidden_dim = vision_block_hidden_dim(context);
  (void)batch_size;
  (void)token_count;
  (void)embed_dim;
  (void)hidden_dim;
  const std::optional<utils::VulkanScratchArenaSpec> scratch_spec =
      std::nullopt;

  return utils::lookup_or_create_labeled_vision_backbone_program(
      vision_backbone_program_label(context->allocation_label()),
      batch_size,
      token_count,
      embed_dim,
      hidden_dim,
      context->num_heads(),
      scratch_spec,
      *runtime_policy.execution_program_plan);
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

Tensor ensure_attention_merge_output_tensor(
    Tensor& output,
    const int64_t batch_size,
    const int64_t token_count,
    const int64_t embed_dim,
    const ScalarType dtype) {
  const std::vector<int64_t> output_sizes{
      batch_size * token_count,
      embed_dim,
  };
  bool needs_allocation = !output.defined() || !output.is_vulkan() ||
      output.scalar_type() != dtype ||
      !output.sizes().equals(IntArrayRef(output_sizes));
  if (!needs_allocation) {
    const vTensor& v_output = convert(output);
    needs_allocation =
        v_output.storage_type() != api::StorageType::BUFFER ||
        v_output.gpu_memory_layout() !=
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED ||
        !v_output.has_direct_buffer_layout();
  }
  if (needs_allocation) {
    output = utils::mark_tensor_execution(
        convert(vTensor{
            api::context(),
            output_sizes,
            convert_dtype(dtype),
            api::StorageType::BUFFER,
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
        }),
        api::ExecutionLayout::BUFFER_DIRECT);
  }
  return output;
}

Tensor merge_attention_heads_for_projection(
    const Tensor& attention_output_arg,
    const int64_t batch_size,
    const int64_t token_count,
    const int64_t num_heads,
    const int64_t head_dim,
    Tensor* output_opt = nullptr) {
  api::AllocationScope allocation_scope("attention_merge_heads");
  const int64_t batch_heads = batch_size * num_heads;
  const int64_t embed_dim = num_heads * head_dim;

  Tensor attention_output = attention_output_arg.is_vulkan()
      ? attention_output_arg
      : attention_output_arg.vulkan();
  attention_output = utils::mark_tensor_execution(
      utils::ensure_buffer_storage(
          attention_output, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
      api::ExecutionLayout::BUFFER_DIRECT);

  TORCH_CHECK(
      attention_output.dim() == 3,
      "Vulkan attention head merge expects a rank-3 [B*H, T, D] tensor");
  TORCH_CHECK(
      attention_output.size(0) == batch_heads &&
          attention_output.size(1) == token_count &&
          attention_output.size(2) == head_dim,
      "Vulkan attention head merge received unexpected attention output sizes");

  vTensor& v_input = convert(attention_output);
  TORCH_CHECK(
      v_input.storage_type() == api::StorageType::BUFFER &&
          v_input.has_direct_buffer_layout(),
      "Vulkan attention head merge expects direct-buffer attention output");

  Tensor output_tensor = output_opt
      ? ensure_attention_merge_output_tensor(
            *output_opt, batch_size, token_count, embed_dim, attention_output.scalar_type())
      : utils::mark_tensor_execution(
            convert(vTensor{
                api::context(),
                {batch_size * token_count, embed_dim},
                convert_dtype(attention_output.scalar_type()),
                api::StorageType::BUFFER,
                api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
            }),
            api::ExecutionLayout::BUFFER_DIRECT);
  vTensor& v_output = convert(output_tensor);

  const struct Block final {
    int32_t head_dim;
    int32_t token_count;
    int32_t num_heads;
    int32_t batch_size;
  } block{
      api::utils::safe_downcast<int32_t>(head_dim),
      api::utils::safe_downcast<int32_t>(token_count),
      api::utils::safe_downcast<int32_t>(num_heads),
      api::utils::safe_downcast<int32_t>(batch_size),
  };

  api::UniformParamsBuffer params(api::context(), block);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(head_dim),
      api::utils::safe_downcast<uint32_t>(token_count),
      api::utils::safe_downcast<uint32_t>(batch_heads),
  };

  api::context()->submit_compute_job(
      VK_KERNEL(merge_attention_heads_buffer),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_output.buffer_metadata(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_input.buffer_metadata(),
      params.buffer());

  utils::log_vulkan_op_hit("aten::attention_merge_heads.buffer_native");
  return output_tensor;
}

Tensor run_attention_projection(
    const Tensor& input_2d,
    const int64_t batch_size,
    const int64_t token_count,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context,
    utils::VisionBackboneProgram* vision_program = nullptr) {
  TORCH_CHECK(
      input_2d.dim() == 2,
      "Vision backbone attention projection expects flattened rank-2 input");

  const int64_t embed_dim = input_2d.size(-1);
  TORCH_CHECK(
      embed_dim % context->num_heads() == 0,
      "Vision backbone block context expects embed_dim divisible by num_heads");
  const int64_t head_dim = embed_dim / context->num_heads();

  Tensor attention_output;
  if (batch_size == 1) {
    Tensor mixed_qkv = vision_program && vision_program->defined()
        ? run_linear_context_out(
              input_2d, context->qkv_context(), vision_program->qkv_output())
        : run_linear_context(input_2d, context->qkv_context());
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
    attention_output = merge_attention_heads_for_projection(
        attention_output,
        batch_size,
        token_count,
        context->num_heads(),
        head_dim,
        vision_program && vision_program->defined() ? &vision_program->qkv_output()
                                                    : nullptr);
    return vision_program && vision_program->defined()
        ? run_linear_context_out(
              attention_output,
              context->proj_context(),
              vision_program->proj_output())
        : run_linear_context(attention_output, context->proj_context());
  }

  Tensor mixed_qkv = vision_program && vision_program->defined()
      ? run_linear_context_out(
            input_2d, context->qkv_context(), vision_program->qkv_output())
      : run_linear_context(input_2d, context->qkv_context());
  if (context->qkv_bias().defined()) {
    mixed_qkv = mixed_qkv.add(context->qkv_bias());
  }
  mixed_qkv = mixed_qkv.reshape({batch_size, token_count, 3 * embed_dim});
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
  attention_output = merge_attention_heads_for_projection(
      attention_output,
      batch_size,
      token_count,
      context->num_heads(),
      head_dim,
      vision_program && vision_program->defined() ? &vision_program->qkv_output()
                                                  : nullptr);
  return vision_program && vision_program->defined()
      ? run_linear_context_out(
            attention_output,
            context->proj_context(),
            vision_program->proj_output())
      : run_linear_context(attention_output, context->proj_context());
}

Tensor tokens_to_feature_map_fallback(
    const Tensor& input_arg,
    const int64_t height,
    const int64_t width) {
  Tensor input = input_arg;
  if (input.dim() == 2) {
    input = input.unsqueeze(0);
  }

  TORCH_CHECK(
      input.dim() == 3,
      "Vulkan tokens_to_feature_map expects a [N, C] or [B, N, C] tensor");
  TORCH_CHECK(
      input.size(1) == height * width,
      "Vulkan tokens_to_feature_map expected token count ",
      height * width,
      " but received ",
      input.size(1));

  Tensor output;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);

    Tensor cpu_input = input.is_vulkan() ? input.cpu() : input;
    output = cpu_input.reshape(
        {cpu_input.size(0), height, width, cpu_input.size(2)});
    output = output.permute({0, 3, 1, 2}).contiguous();
  }

  if (input_arg.is_vulkan()) {
    return output.vulkan();
  }
  return output;
}

Tensor feature_map_to_tokens_fallback(const Tensor& input_arg) {
  TORCH_CHECK(
      input_arg.dim() == 4,
      "Vulkan feature_map_to_tokens expects a [B, C, H, W] tensor");

  Tensor output;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);

    Tensor cpu_input = input_arg.is_vulkan() ? input_arg.cpu() : input_arg;
    output = cpu_input.permute({0, 2, 3, 1})
                 .reshape(
                     {cpu_input.size(0),
                      cpu_input.size(2) * cpu_input.size(3),
                      cpu_input.size(1)})
                 .contiguous();
  }

  if (input_arg.is_vulkan()) {
    return output.vulkan();
  }
  return output;
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
      norm1_context_(make_layernorm_context(
          norm1_weight,
          norm1_bias,
          norm1_eps,
          child_label(allocation_label_, "norm1"))),
      qkv_context_(
          make_qkv_context(qkv_weight, child_label(allocation_label_, "qkv"))),
      qkv_bias_(move_optional_to_vulkan_buffer(qkv_bias)),
      num_heads_(num_heads),
      proj_context_(make_linear_context(
          proj_weight,
          proj_bias,
          child_label(allocation_label_, "proj"))),
      ls1_gamma_(move_optional_to_vulkan_buffer(ls1_gamma)),
      norm2_context_(make_layernorm_context(
          norm2_weight,
          norm2_bias,
          norm2_eps,
          child_label(allocation_label_, "norm2"))),
      fc1_context_(
          make_linear_context(fc1_weight, fc1_bias, child_label(allocation_label_, "fc1"))),
      fc2_context_(
          make_linear_context(fc2_weight, fc2_bias, child_label(allocation_label_, "fc2"))),
      ls2_gamma_(move_optional_to_vulkan_buffer(ls2_gamma)) {
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
  auto vision_program = prime_vision_backbone_program(input, context);
  if (vision_program.defined()) {
    if (vision_program.scratch_arena().has_value()) {
      vision_program.scratch_arena()->reset();
    }
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_backbone_block_context.program");
  }

  const bool use_2d_input = input.dim() == 2;
  const int64_t batch_size = use_2d_input ? 1 : input.size(0);
  const int64_t token_count = use_2d_input ? input.size(0) : input.size(1);
  const int64_t embed_dim = input.size(-1);
  const int64_t hidden_rows = batch_size * token_count;
  Tensor input_2d = use_2d_input ? input : input.reshape({hidden_rows, embed_dim});
  utils::VisionBackboneProgram* const vision_program_ptr =
      vision_program.defined() ? &vision_program : nullptr;

  const std::array<int64_t, 1> normalized_shape = {embed_dim};
  Tensor attention_input = vision_program_ptr
      ? run_layernorm_context_out(
            input_2d,
            normalized_shape,
            context->norm1_context(),
            vision_program.norm1_output())
      : run_layernorm_context(input_2d, normalized_shape, context->norm1_context());
  Tensor attention_output = run_attention_projection(
      attention_input,
      batch_size,
      token_count,
      context,
      vision_program_ptr);
  attention_output = maybe_apply_layerscale(attention_output, context->ls1_gamma());
  Tensor hidden_states = at::add(input_2d, attention_output);

  Tensor mlp_input = vision_program_ptr
      ? run_layernorm_context_out(
            hidden_states,
            normalized_shape,
            context->norm2_context(),
            vision_program.norm2_output())
      : run_layernorm_context(
            hidden_states, normalized_shape, context->norm2_context());
  Tensor mlp_output = vision_program_ptr
      ? run_linear_gelu_context_out(
            mlp_input, context->fc1_context(), vision_program.fc1_output())
      : run_linear_gelu_context(mlp_input, context->fc1_context());
  mlp_output = vision_program_ptr
      ? run_linear_context_out(
            mlp_output, context->fc2_context(), vision_program.fc2_output())
      : run_linear_context(mlp_output, context->fc2_context());
  mlp_output = maybe_apply_layerscale(mlp_output, context->ls2_gamma());

  Tensor output = at::add(hidden_states, mlp_output);
  if (!use_2d_input) {
    output = output.reshape({batch_size, token_count, embed_dim});
  }
  utils::log_vulkan_op_hit("vulkan_prepack::run_vision_backbone_block_context");
  return maybe_restore_tensor(output, output_device, output_dtype);
}

Tensor tokens_to_feature_map(
    const Tensor& input_arg,
    const int64_t height,
    const int64_t width) {
  if (!input_arg.is_vulkan() || input_arg.scalar_type() != kFloat) {
    return tokens_to_feature_map_fallback(input_arg, height, width);
  }

  api::AllocationScope allocation_scope("tokens_to_feature_map");

  Tensor input = input_arg;
  const bool use_2d_input = input.dim() == 2;
  const int64_t batch_size = use_2d_input ? 1 : input.size(0);
  const int64_t token_count = use_2d_input ? input.size(0) : input.size(1);
  const int64_t channels = input.size(-1);

  TORCH_CHECK(
      input.dim() == 2 || input.dim() == 3,
      "Vulkan tokens_to_feature_map expects a [N, C] or [B, N, C] tensor");
  TORCH_CHECK(
      token_count == height * width,
      "Vulkan tokens_to_feature_map expected token count ",
      height * width,
      " but received ",
      token_count);

  utils::log_vulkan_op_hit("aten::tokens_to_feature_map");

  const vTensor& v_input_probe = convert(input);
  if (
      v_input_probe.storage_type() == api::StorageType::TEXTURE_3D &&
      v_input_probe.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED &&
      batch_size == 1) {
    const std::vector<int64_t> output_sizes{
        batch_size,
        channels,
        height,
        width,
    };

    vTensor v_output{
        api::context(),
        output_sizes,
        convert_dtype(input.scalar_type()),
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
    };

    api::PipelineBarrier pipeline_barrier{};
    const api::utils::uvec3 global_size{
        api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
        1u,
        1u,
    };
    api::UniformParamsBuffer out_meta =
        utils::make_buffer_compute_metadata_ubo(api::context(), v_output);

    api::context()->submit_compute_job(
        VK_KERNEL(tokens_to_feature_map_texture_to_buffer),
        pipeline_barrier,
        global_size,
        adaptive_work_group_size(global_size),
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        out_meta.buffer(),
        v_input_probe.image(pipeline_barrier, api::PipelineStage::COMPUTE));

    utils::log_vulkan_op_hit("aten::tokens_to_feature_map.texture_to_buffer");
    return convert(v_output);
  }

  if (
      v_input_probe.storage_type() != api::StorageType::BUFFER ||
      !utils::supports_buffer_elementwise_compute(v_input_probe)) {
    utils::log_vulkan_op_hit("aten::tokens_to_feature_map.texture_view_fallback");
    if (use_2d_input) {
      return tokens_to_feature_map_fallback(input_arg, height, width);
    }
    return input.permute({0, 2, 1})
        .reshape({batch_size, channels, height, width});
  }

  const vTensor& v_input = v_input_probe;

  const std::vector<int64_t> output_sizes{
      batch_size,
      channels,
      height,
      width,
  };

  vTensor v_output{
      api::context(),
      output_sizes,
      convert_dtype(input.scalar_type()),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  api::UniformParamsBuffer input_meta =
      utils::make_buffer_compute_metadata_ubo(api::context(), v_input);
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(api::context(), v_output);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
      1u,
      1u,
  };

  api::context()->submit_compute_job(
      VK_KERNEL(tokens_to_feature_map_buffer),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      input_meta.buffer());

  utils::log_vulkan_op_hit("aten::tokens_to_feature_map.buffer_to_buffer");
  return convert(v_output);
}

Tensor feature_map_to_tokens(const Tensor& input_arg) {
  if (!input_arg.is_vulkan() || input_arg.scalar_type() != kFloat) {
    return feature_map_to_tokens_fallback(input_arg);
  }

  TORCH_CHECK(
      input_arg.dim() == 4,
      "Vulkan feature_map_to_tokens expects a [B, C, H, W] tensor");

  api::AllocationScope allocation_scope("feature_map_to_tokens");
  utils::log_vulkan_op_hit("aten::feature_map_to_tokens");

  const vTensor& v_input = convert(input_arg);
  if (
      v_input.storage_type() == api::StorageType::TEXTURE_3D &&
      v_input.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED &&
      input_arg.size(0) == 1) {
    const std::vector<int64_t> output_sizes{
        1,
        input_arg.size(2) * input_arg.size(3),
        input_arg.size(1),
    };

    vTensor v_output{
        api::context(),
        output_sizes,
        convert_dtype(input_arg.scalar_type()),
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
    };

    const struct Block final {
      api::utils::ivec4 info;
    } block{
        {
            api::utils::safe_downcast<int32_t>(input_arg.size(3)),
            api::utils::safe_downcast<int32_t>(input_arg.size(2)),
            api::utils::safe_downcast<int32_t>(input_arg.size(1)),
            api::utils::safe_downcast<int32_t>(input_arg.size(0)),
        },
    };

    api::UniformParamsBuffer params(api::context(), block);
    api::PipelineBarrier pipeline_barrier{};
    const api::utils::uvec3 global_size{
        api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
        1u,
        1u,
    };
    api::UniformParamsBuffer out_meta =
        utils::make_buffer_compute_metadata_ubo(api::context(), v_output);

    api::context()->submit_compute_job(
        VK_KERNEL(feature_map_to_tokens_texture_to_buffer),
        pipeline_barrier,
        global_size,
        adaptive_work_group_size(global_size),
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        out_meta.buffer(),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());

    utils::log_vulkan_op_hit("aten::feature_map_to_tokens.texture_to_buffer");
    return convert(v_output);
  }

  if (
      v_input.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_elementwise_compute(v_input)) {
    const std::vector<int64_t> output_sizes{
        input_arg.size(0),
        input_arg.size(2) * input_arg.size(3),
        input_arg.size(1),
    };

    vTensor v_output{
        api::context(),
        output_sizes,
        convert_dtype(input_arg.scalar_type()),
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
    };

    api::PipelineBarrier pipeline_barrier{};
    const api::utils::uvec3 global_size{
        api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
        1u,
        1u,
    };
    api::UniformParamsBuffer out_meta =
        utils::make_buffer_compute_metadata_ubo(api::context(), v_output);
    api::UniformParamsBuffer input_meta =
        utils::make_buffer_compute_metadata_ubo(api::context(), v_input);

    api::context()->submit_compute_job(
        VK_KERNEL(feature_map_to_tokens_buffer),
        pipeline_barrier,
        global_size,
        adaptive_work_group_size(global_size),
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        out_meta.buffer(),
        v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        input_meta.buffer());

    utils::log_vulkan_op_hit("aten::feature_map_to_tokens.buffer_to_buffer");
    return convert(v_output);
  }

  utils::log_vulkan_op_hit("aten::feature_map_to_tokens.fallback");
  return feature_map_to_tokens_fallback(input_arg);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
