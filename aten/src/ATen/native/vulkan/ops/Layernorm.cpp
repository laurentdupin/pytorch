#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/NativeLayerNorm.h>
#include <ATen/native/vulkan/ops/Norm.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>

#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <optional>
#include <utility>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

bool prefer_buffer_layer_norm(
    const Tensor& input_arg,
    IntArrayRef normalized_shape) {
  if (!input_arg.is_vulkan()) {
    return false;
  }
  const bool last_dim_width_norm = normalized_shape.size() == 1u &&
      normalized_shape.front() == input_arg.size(-1);
  if (
      last_dim_width_norm && input_arg.scalar_type() == kFloat &&
      input_arg.dim() >= 2 && input_arg.dim() <= 4) {
    const vTensor& v_input = convert(input_arg);
    // The unlabeled eager vision path already carries buffer-native residuals
    // into layer_norm. Keeping that path on the native buffer family avoids an
    // immediate buffer->texture->buffer roundtrip before the following linear.
    if (
        v_input.storage_type() == api::StorageType::BUFFER &&
        utils::supports_buffer_reduction_compute(v_input)) {
      return true;
    }
  }
  const auto request = utils::make_vulkan_tensor_norm_request(
      input_arg, utils::VulkanTensorRole::Input);
  const auto runtime_policy = utils::build_vulkan_runtime_policy(request);
  return runtime_policy.norm_kernel_family ==
          utils::VulkanNormKernelFamily::UnifiedBufferView &&
      last_dim_width_norm;
}

Tensor layer_norm_fused_width(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps) {
  static constexpr FusedNormWidthSpec kSpec{
      "layer_norm.output_only",
      "layer_norm_width",
      "aten::layer_norm.fused_width",
      true,
  };
  Tensor output = fused_norm_width_impl(
      input_arg, normalized_shape, weight_opt, bias_opt, eps, kSpec);
  if (c10::InferenceMode::is_enabled()) {
    maybe_synchronize_after_norm();
  }
  return output;
}

Tensor layer_norm_fused_width_out(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps,
    Tensor& output) {
  static constexpr FusedNormWidthSpec kSpec{
      "layer_norm.output_only",
      "layer_norm_width",
      "aten::layer_norm.fused_width",
      true,
  };
  Tensor result = fused_norm_width_impl(
      input_arg, normalized_shape, weight_opt, bias_opt, eps, kSpec, output);
  if (c10::InferenceMode::is_enabled()) {
    maybe_synchronize_after_norm();
  }
  return result;
}

Tensor layer_norm_context_parameter_to_buffer(const Tensor& tensor) {
  Tensor vulkan_tensor = tensor.is_vulkan() ? tensor : tensor.vulkan();
  return utils::mark_tensor_execution(
      utils::ensure_buffer_storage(
          vulkan_tensor, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
      api::ExecutionLayout::BUFFER_DIRECT,
      true);
}

bool can_run_add_layer_norm_buffer_width(
    const Tensor& residual_arg,
    const Tensor& addend_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    const Tensor& residual_output,
    const Tensor& norm_output) {
  if (
      !residual_arg.is_vulkan() || !addend_arg.is_vulkan() ||
      !residual_output.defined() || !norm_output.defined() ||
      !residual_output.is_vulkan() || !norm_output.is_vulkan() ||
      residual_arg.scalar_type() != kFloat ||
      addend_arg.scalar_type() != kFloat ||
      residual_output.scalar_type() != kFloat ||
      norm_output.scalar_type() != kFloat ||
      !residual_arg.sizes().equals(addend_arg.sizes()) ||
      !residual_arg.sizes().equals(residual_output.sizes()) ||
      !residual_arg.sizes().equals(norm_output.sizes()) ||
      residual_arg.dim() < 2 || residual_arg.dim() > 4 ||
      normalized_shape.size() != 1u ||
      normalized_shape.front() != residual_arg.size(-1) ||
      !weight_opt.has_value() || !bias_opt.has_value() ||
      !weight_opt->defined() || !bias_opt->defined() ||
      weight_opt->scalar_type() != kFloat ||
      bias_opt->scalar_type() != kFloat ||
      !weight_opt->sizes().equals(normalized_shape) ||
      !bias_opt->sizes().equals(normalized_shape)) {
    return false;
  }

  const vTensor& v_residual = convert(residual_arg);
  const vTensor& v_addend = convert(addend_arg);
  const vTensor& v_residual_output = convert(residual_output);
  const vTensor& v_norm_output = convert(norm_output);
  return v_residual.storage_type() == api::StorageType::BUFFER &&
      v_addend.storage_type() == api::StorageType::BUFFER &&
      v_residual_output.storage_type() == api::StorageType::BUFFER &&
      v_norm_output.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_reduction_compute(v_residual) &&
      utils::supports_buffer_reduction_compute(v_addend) &&
      utils::supports_buffer_reduction_compute(v_residual_output) &&
      utils::supports_buffer_reduction_compute(v_norm_output);
}

bool can_run_layer_scale_buffer_width(
    const Tensor& scale_arg,
    IntArrayRef normalized_shape) {
  if (
      !scale_arg.defined() || !scale_arg.is_vulkan() ||
      scale_arg.scalar_type() != kFloat || scale_arg.dim() != 1 ||
      normalized_shape.size() != 1u ||
      scale_arg.size(0) != normalized_shape.front()) {
    return false;
  }

  const vTensor& v_scale = convert(scale_arg);
  return v_scale.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_elementwise_compute(v_scale);
}

std::optional<std::pair<Tensor, Tensor>> try_run_add_layer_norm_eager_out(
    const Tensor& residual_arg,
    const Tensor& addend_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps,
    Tensor& residual_output_arg,
    Tensor& norm_output_arg) {
  const Tensor residual =
      residual_arg.is_vulkan() ? residual_arg : residual_arg.vulkan();
  const Tensor addend =
      addend_arg.is_vulkan() ? addend_arg : addend_arg.vulkan();

  if (!can_run_add_layer_norm_buffer_width(
          residual,
          addend,
          normalized_shape,
          weight_opt,
          bias_opt,
          residual_output_arg,
          norm_output_arg)) {
    return std::nullopt;
  }

  api::AllocationScope allocation_scope("add_layer_norm.buffer_width");
  api::Context* const context = api::context();
  utils::log_vulkan_op_hit("aten::add_layer_norm.buffer_width");

  Tensor weight = utils::prepare_vulkan_execution_tensor(
      *weight_opt, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  Tensor bias = utils::prepare_vulkan_execution_tensor(
      *bias_opt, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  Tensor residual_output = utils::mark_tensor_execution(
      residual_output_arg,
      utils::resolve_buffer_execution_layout(convert(residual_output_arg)),
      false);
  Tensor norm_output = utils::mark_tensor_execution(
      norm_output_arg,
      utils::resolve_buffer_execution_layout(convert(norm_output_arg)),
      false);

  const vTensor& v_residual = convert(residual);
  const vTensor& v_addend = convert(addend);
  vTensor& v_residual_output = convert(residual_output);
  vTensor& v_norm_output = convert(norm_output);
  const vTensor& v_weight = convert(weight);
  const vTensor& v_bias = convert(bias);

  const struct Block final {
    float eps;
    float fill0;
    float fill1;
    float fill2;
  } block{
      api::utils::safe_downcast<float>(eps),
      0.0f,
      0.0f,
      0.0f,
  };

  api::UniformParamsBuffer params(context, block);
  api::UniformParamsBuffer residual_out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_residual_output);
  api::UniformParamsBuffer norm_out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_norm_output);
  api::UniformParamsBuffer residual_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_residual);
  api::UniformParamsBuffer addend_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_addend);
  api::UniformParamsBuffer weight_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_weight);
  api::UniformParamsBuffer bias_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_bias);

  const uint32_t normalized_size = api::utils::safe_downcast<uint32_t>(
      std::max<int64_t>(normalized_shape.front(), 1));
  const uint32_t row_count = api::utils::safe_downcast<uint32_t>(
      v_norm_output.numel() / normalized_size);
  const bool use_width384_kernel =
      normalized_size == 384u && row_count >= 512u;
  const api::utils::uvec3 global_size{
      use_width384_kernel ? row_count * 256u : row_count,
      1u,
      1u,
  };
  const api::utils::uvec3 local_size =
      use_width384_kernel ? api::utils::uvec3{256u, 1u, 1u}
                          : adaptive_work_group_size(global_size);
  if (use_width384_kernel) {
    utils::log_vulkan_op_hit("aten::add_layer_norm.buffer_width384");
  }

  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      use_width384_kernel ? VK_KERNEL(add_layer_norm_width384_buffer_float)
                          : VK_KERNEL(add_layer_norm_width_buffer_float),
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      v_residual_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      residual_out_meta.buffer(),
      v_norm_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      norm_out_meta.buffer(),
      v_residual.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      residual_meta.buffer(),
      v_addend.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      addend_meta.buffer(),
      v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      weight_meta.buffer(),
      v_bias.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      bias_meta.buffer(),
      params.buffer());

  if (c10::InferenceMode::is_enabled()) {
    maybe_synchronize_after_norm();
  }
  record_tensor_write(
      residual_output,
      "aten::add_layer_norm",
      "buffer_width_residual",
      {residual, addend, weight, bias});
  record_tensor_write(
      norm_output,
      "aten::add_layer_norm",
      "buffer_width_norm",
      {residual, addend, weight, bias});

  return std::make_pair(residual_output, norm_output);
}

std::optional<std::pair<Tensor, Tensor>> try_run_add_scaled_layer_norm_eager_out(
    const Tensor& residual_arg,
    const Tensor& addend_arg,
    const Tensor& scale_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps,
    Tensor& residual_output_arg,
    Tensor& norm_output_arg) {
  const Tensor residual =
      residual_arg.is_vulkan() ? residual_arg : residual_arg.vulkan();
  const Tensor addend =
      addend_arg.is_vulkan() ? addend_arg : addend_arg.vulkan();
  const Tensor scale =
      scale_arg.is_vulkan() ? scale_arg : scale_arg.vulkan();

  if (
      !can_run_add_layer_norm_buffer_width(
          residual,
          addend,
          normalized_shape,
          weight_opt,
          bias_opt,
          residual_output_arg,
          norm_output_arg) ||
      !can_run_layer_scale_buffer_width(scale, normalized_shape)) {
    return std::nullopt;
  }

  api::AllocationScope allocation_scope("add_scaled_layer_norm.buffer_width");
  api::Context* const context = api::context();
  utils::log_vulkan_op_hit("aten::add_scaled_layer_norm.buffer_width");

  Tensor weight = utils::prepare_vulkan_execution_tensor(
      *weight_opt, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  Tensor bias = utils::prepare_vulkan_execution_tensor(
      *bias_opt, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  Tensor scale_input = utils::prepare_vulkan_execution_tensor(
      scale, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  Tensor residual_output = utils::mark_tensor_execution(
      residual_output_arg,
      utils::resolve_buffer_execution_layout(convert(residual_output_arg)),
      false);
  Tensor norm_output = utils::mark_tensor_execution(
      norm_output_arg,
      utils::resolve_buffer_execution_layout(convert(norm_output_arg)),
      false);

  const vTensor& v_residual = convert(residual);
  const vTensor& v_addend = convert(addend);
  const vTensor& v_scale = convert(scale_input);
  vTensor& v_residual_output = convert(residual_output);
  vTensor& v_norm_output = convert(norm_output);
  const vTensor& v_weight = convert(weight);
  const vTensor& v_bias = convert(bias);

  const struct Block final {
    float eps;
    float fill0;
    float fill1;
    float fill2;
  } block{
      api::utils::safe_downcast<float>(eps),
      0.0f,
      0.0f,
      0.0f,
  };

  api::UniformParamsBuffer params(context, block);
  api::UniformParamsBuffer residual_out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_residual_output);
  api::UniformParamsBuffer norm_out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_norm_output);
  api::UniformParamsBuffer residual_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_residual);
  api::UniformParamsBuffer addend_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_addend);
  api::UniformParamsBuffer scale_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_scale);
  api::UniformParamsBuffer weight_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_weight);
  api::UniformParamsBuffer bias_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_bias);

  const uint32_t normalized_size = api::utils::safe_downcast<uint32_t>(
      std::max<int64_t>(normalized_shape.front(), 1));
  const uint32_t row_count = api::utils::safe_downcast<uint32_t>(
      v_norm_output.numel() / normalized_size);
  const bool use_width384_kernel =
      normalized_size == 384u && row_count >= 512u;
  const api::utils::uvec3 global_size{
      use_width384_kernel ? row_count * 256u : row_count,
      1u,
      1u,
  };
  const api::utils::uvec3 local_size =
      use_width384_kernel ? api::utils::uvec3{256u, 1u, 1u}
                          : adaptive_work_group_size(global_size);
  if (use_width384_kernel) {
    utils::log_vulkan_op_hit("aten::add_scaled_layer_norm.buffer_width384");
  }

  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      use_width384_kernel ? VK_KERNEL(add_scaled_layer_norm_width384_buffer_float)
                          : VK_KERNEL(add_scaled_layer_norm_width_buffer_float),
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      v_residual_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      residual_out_meta.buffer(),
      v_norm_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      norm_out_meta.buffer(),
      v_residual.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      residual_meta.buffer(),
      v_addend.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      addend_meta.buffer(),
      v_scale.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      scale_meta.buffer(),
      v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      weight_meta.buffer(),
      v_bias.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      bias_meta.buffer(),
      params.buffer());

  if (c10::InferenceMode::is_enabled()) {
    maybe_synchronize_after_norm();
  }
  record_tensor_write(
      residual_output,
      "aten::add_scaled_layer_norm",
      "buffer_width_residual",
      {residual, addend, scale_input, weight, bias});
  record_tensor_write(
      norm_output,
      "aten::add_scaled_layer_norm",
      "buffer_width_norm",
      {residual, addend, scale_input, weight, bias});

  return std::make_pair(residual_output, norm_output);
}

} // namespace

Tensor layer_norm_impl(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps) {
  utils::log_vulkan_op_hit("aten::layer_norm");
  check_layer_norm_inputs(input_arg, normalized_shape, weight_opt, bias_opt);

  TORCH_CHECK(
      weight_opt->defined() && bias_opt->defined(),
      "Vulkan layer_norm expects weight and bias arguments");

  if (
      !prefer_buffer_layer_norm(input_arg, normalized_shape) &&
      supports_fused_layer_norm_last_dim(
          input_arg, normalized_shape, weight_opt, bias_opt)) {
    return layer_norm_fused_width(
        input_arg, normalized_shape, weight_opt, bias_opt, eps);
  }

  return std::get<0>(native_layer_norm_impl(
      input_arg, normalized_shape, weight_opt, bias_opt, eps));
}

LayernormPackedContext::LayernormPackedContext(
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    double eps,
    std::string allocation_label) {
  TORCH_CHECK(weight, "Weight must be provided!");
  weight_ = layer_norm_context_parameter_to_buffer(*weight);
  TORCH_CHECK(bias, "Bias must be provided!");
  bias_ = layer_norm_context_parameter_to_buffer(*bias);
  eps_ = eps;
  allocation_label_ = std::move(allocation_label);
}

LayernormPackedContext LayernormPackedContext::pack(
    c10::impl::GenericList unpacked) {
  return LayernormPackedContext(
      get_optional_tensor(unpacked, ListArgs::kWeight),
      get_optional_tensor(unpacked, ListArgs::kBias),
      unpacked.get(ListArgs::kEps).toDouble(),
      unpacked.get(ListArgs::kLabel).toStringRef());
}

const c10::impl::GenericList LayernormPackedContext::unpack() const {
  c10::impl::GenericList unpacked{c10::AnyType::get()};
  unpacked.reserve(ListArgs::kNumArgs);
  report_vulkan_cpu_fallback(
      "vulkan_prepack::layernorm_context",
      "unpack_cpu_readback",
      {weight_, bias_},
      VulkanCpuFallbackKind::SyncReadback);
  unpacked.emplace_back(weight_.cpu());
  unpacked.emplace_back(bias_.cpu());
  unpacked.emplace_back(eps_);
  unpacked.emplace_back(allocation_label_);
  return unpacked;
}

c10::intrusive_ptr<LayernormPackedContext> create_layernorm_context(
    std::optional<Tensor>&& weight,
    std::optional<Tensor>&& bias,
    double eps) {
  return c10::make_intrusive<LayernormPackedContext>(
      LayernormPackedContext(weight, bias, eps));
}

c10::intrusive_ptr<LayernormPackedContext> create_layernorm_context_labeled(
    std::optional<Tensor>&& weight,
    std::optional<Tensor>&& bias,
    double eps,
    std::string label) {
  return c10::make_intrusive<LayernormPackedContext>(
      LayernormPackedContext(weight, bias, eps, std::move(label)));
}

Tensor run_layernorm_context(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& layernorm_context) {
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const std::optional<Tensor> weight_opt = layernorm_context->weight();
  const std::optional<Tensor> bias_opt = layernorm_context->bias();
  const float eps = api::utils::safe_downcast<float>(layernorm_context->eps());
  std::optional<api::RuntimeLabelScope> runtime_scope;
  if (!layernorm_context->allocation_label().empty()) {
    runtime_scope.emplace(layernorm_context->allocation_label());
  }

  return layer_norm_impl(input, normalized_shape, weight_opt, bias_opt, eps);
}

Tensor run_layernorm_context_out(
    const Tensor& input_arg,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& layernorm_context,
    Tensor& output) {
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const std::optional<Tensor> weight_opt = layernorm_context->weight();
  const std::optional<Tensor> bias_opt = layernorm_context->bias();
  const float eps = api::utils::safe_downcast<float>(layernorm_context->eps());
  std::optional<api::RuntimeLabelScope> runtime_scope;
  if (!layernorm_context->allocation_label().empty()) {
    runtime_scope.emplace(layernorm_context->allocation_label());
  }
  const bool prefer_buffer_path =
      prefer_buffer_layer_norm(input, normalized_shape);

  if (prefer_buffer_path) {
    if (auto result = try_run_native_layer_norm_buffer_width_out(
            input,
            normalized_shape,
            weight_opt,
            bias_opt,
            eps,
            output)) {
      return *result;
    }
  }

  if (
      !prefer_buffer_path &&
      supports_fused_layer_norm_last_dim(
          input, normalized_shape, weight_opt, bias_opt)) {
    return layer_norm_fused_width_out(
        input, normalized_shape, weight_opt, bias_opt, eps, output);
  }

  Tensor result =
      layer_norm_impl(input, normalized_shape, weight_opt, bias_opt, eps);
  if (output.defined() && output.is_vulkan()) {
    if (prefer_buffer_path) {
      output = result;
      return output;
    }
    return rebind_vulkan_output(output, result);
  }
  output = result;
  return output;
}

std::optional<std::pair<Tensor, Tensor>> try_run_add_layernorm_context_out(
    const Tensor& residual_arg,
    const Tensor& addend_arg,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& layernorm_context,
    Tensor& residual_output_arg,
    Tensor& norm_output_arg) {
  const Tensor residual =
      residual_arg.is_vulkan() ? residual_arg : residual_arg.vulkan();
  const Tensor addend =
      addend_arg.is_vulkan() ? addend_arg : addend_arg.vulkan();
  const std::optional<Tensor> weight_opt = layernorm_context->weight();
  const std::optional<Tensor> bias_opt = layernorm_context->bias();

  if (!can_run_add_layer_norm_buffer_width(
          residual,
          addend,
          normalized_shape,
          weight_opt,
          bias_opt,
          residual_output_arg,
          norm_output_arg)) {
    return std::nullopt;
  }

  std::optional<api::RuntimeLabelScope> runtime_scope;
  if (!layernorm_context->allocation_label().empty()) {
    runtime_scope.emplace(layernorm_context->allocation_label());
  }

  api::AllocationScope allocation_scope("add_layer_norm.buffer_width");
  api::Context* const context = api::context();
  utils::log_vulkan_op_hit("aten::add_layer_norm.buffer_width");

  Tensor weight = utils::prepare_vulkan_execution_tensor(
      *weight_opt, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  Tensor bias = utils::prepare_vulkan_execution_tensor(
      *bias_opt, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  Tensor residual_output = utils::mark_tensor_execution(
      residual_output_arg,
      utils::resolve_buffer_execution_layout(convert(residual_output_arg)),
      false);
  Tensor norm_output = utils::mark_tensor_execution(
      norm_output_arg,
      utils::resolve_buffer_execution_layout(convert(norm_output_arg)),
      false);

  const vTensor& v_residual = convert(residual);
  const vTensor& v_addend = convert(addend);
  vTensor& v_residual_output = convert(residual_output);
  vTensor& v_norm_output = convert(norm_output);
  const vTensor& v_weight = convert(weight);
  const vTensor& v_bias = convert(bias);

  const struct Block final {
    float eps;
    float fill0;
    float fill1;
    float fill2;
  } block{
      api::utils::safe_downcast<float>(layernorm_context->eps()),
      0.0f,
      0.0f,
      0.0f,
  };

  api::UniformParamsBuffer params(context, block);
  api::UniformParamsBuffer residual_out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_residual_output);
  api::UniformParamsBuffer norm_out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_norm_output);
  api::UniformParamsBuffer residual_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_residual);
  api::UniformParamsBuffer addend_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_addend);
  api::UniformParamsBuffer weight_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_weight);
  api::UniformParamsBuffer bias_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_bias);

  const uint32_t normalized_size = api::utils::safe_downcast<uint32_t>(
      std::max<int64_t>(normalized_shape.front(), 1));
  const uint32_t row_count = api::utils::safe_downcast<uint32_t>(
      v_norm_output.numel() / normalized_size);
  const bool use_width384_kernel =
      normalized_size == 384u && row_count >= 512u;
  const api::utils::uvec3 global_size{
      use_width384_kernel ? row_count * 256u : row_count,
      1u,
      1u,
  };
  const api::utils::uvec3 local_size =
      use_width384_kernel ? api::utils::uvec3{256u, 1u, 1u}
                          : adaptive_work_group_size(global_size);
  if (use_width384_kernel) {
    utils::log_vulkan_op_hit("aten::add_layer_norm.buffer_width384");
  }

  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      use_width384_kernel ? VK_KERNEL(add_layer_norm_width384_buffer_float)
                          : VK_KERNEL(add_layer_norm_width_buffer_float),
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      v_residual_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      residual_out_meta.buffer(),
      v_norm_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      norm_out_meta.buffer(),
      v_residual.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      residual_meta.buffer(),
      v_addend.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      addend_meta.buffer(),
      v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      weight_meta.buffer(),
      v_bias.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      bias_meta.buffer(),
      params.buffer());

  if (c10::InferenceMode::is_enabled()) {
    maybe_synchronize_after_norm();
  }
  record_tensor_write(
      residual_output,
      "vulkan_prepack::run_add_layernorm_context",
      "buffer_width_residual",
      {residual, addend, weight, bias});
  record_tensor_write(
      norm_output,
      "vulkan_prepack::run_add_layernorm_context",
      "buffer_width_norm",
      {residual, addend, weight, bias});

  return std::make_pair(residual_output, norm_output);
}

std::optional<std::pair<Tensor, Tensor>>
try_run_add_scaled_layernorm_context_out(
    const Tensor& residual_arg,
    const Tensor& addend_arg,
    const Tensor& scale_arg,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& layernorm_context,
    Tensor& residual_output_arg,
    Tensor& norm_output_arg) {
  if (!scale_arg.defined()) {
    return std::nullopt;
  }

  const Tensor residual =
      residual_arg.is_vulkan() ? residual_arg : residual_arg.vulkan();
  const Tensor addend =
      addend_arg.is_vulkan() ? addend_arg : addend_arg.vulkan();
  const Tensor scale = scale_arg.is_vulkan() ? scale_arg : scale_arg.vulkan();
  const std::optional<Tensor> weight_opt = layernorm_context->weight();
  const std::optional<Tensor> bias_opt = layernorm_context->bias();

  if (
      !can_run_add_layer_norm_buffer_width(
          residual,
          addend,
          normalized_shape,
          weight_opt,
          bias_opt,
          residual_output_arg,
          norm_output_arg) ||
      !can_run_layer_scale_buffer_width(scale, normalized_shape)) {
    return std::nullopt;
  }

  std::optional<api::RuntimeLabelScope> runtime_scope;
  if (!layernorm_context->allocation_label().empty()) {
    runtime_scope.emplace(layernorm_context->allocation_label());
  }

  api::AllocationScope allocation_scope("add_scaled_layer_norm.buffer_width");
  api::Context* const context = api::context();
  utils::log_vulkan_op_hit("aten::add_scaled_layer_norm.buffer_width");

  Tensor weight = utils::prepare_vulkan_execution_tensor(
      *weight_opt, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  Tensor bias = utils::prepare_vulkan_execution_tensor(
      *bias_opt, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  Tensor scale_input = utils::prepare_vulkan_execution_tensor(
      scale, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  Tensor residual_output = utils::mark_tensor_execution(
      residual_output_arg,
      utils::resolve_buffer_execution_layout(convert(residual_output_arg)),
      false);
  Tensor norm_output = utils::mark_tensor_execution(
      norm_output_arg,
      utils::resolve_buffer_execution_layout(convert(norm_output_arg)),
      false);

  const vTensor& v_residual = convert(residual);
  const vTensor& v_addend = convert(addend);
  const vTensor& v_scale = convert(scale_input);
  vTensor& v_residual_output = convert(residual_output);
  vTensor& v_norm_output = convert(norm_output);
  const vTensor& v_weight = convert(weight);
  const vTensor& v_bias = convert(bias);

  const struct Block final {
    float eps;
    float fill0;
    float fill1;
    float fill2;
  } block{
      api::utils::safe_downcast<float>(layernorm_context->eps()),
      0.0f,
      0.0f,
      0.0f,
  };

  api::UniformParamsBuffer params(context, block);
  api::UniformParamsBuffer residual_out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_residual_output);
  api::UniformParamsBuffer norm_out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_norm_output);
  api::UniformParamsBuffer residual_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_residual);
  api::UniformParamsBuffer addend_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_addend);
  api::UniformParamsBuffer scale_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_scale);
  api::UniformParamsBuffer weight_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_weight);
  api::UniformParamsBuffer bias_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_bias);

  const uint32_t normalized_size = api::utils::safe_downcast<uint32_t>(
      std::max<int64_t>(normalized_shape.front(), 1));
  const uint32_t row_count = api::utils::safe_downcast<uint32_t>(
      v_norm_output.numel() / normalized_size);
  const bool use_width384_kernel =
      normalized_size == 384u && row_count >= 512u;
  const api::utils::uvec3 global_size{
      use_width384_kernel ? row_count * 256u : row_count,
      1u,
      1u,
  };
  const api::utils::uvec3 local_size =
      use_width384_kernel ? api::utils::uvec3{256u, 1u, 1u}
                          : adaptive_work_group_size(global_size);
  if (use_width384_kernel) {
    utils::log_vulkan_op_hit("aten::add_scaled_layer_norm.buffer_width384");
  }

  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      use_width384_kernel ? VK_KERNEL(add_scaled_layer_norm_width384_buffer_float)
                          : VK_KERNEL(add_scaled_layer_norm_width_buffer_float),
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      v_residual_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      residual_out_meta.buffer(),
      v_norm_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      norm_out_meta.buffer(),
      v_residual.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      residual_meta.buffer(),
      v_addend.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      addend_meta.buffer(),
      v_scale.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      scale_meta.buffer(),
      v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      weight_meta.buffer(),
      v_bias.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      bias_meta.buffer(),
      params.buffer());

  if (c10::InferenceMode::is_enabled()) {
    maybe_synchronize_after_norm();
  }
  record_tensor_write(
      residual_output,
      "vulkan_prepack::run_add_scaled_layernorm_context",
      "buffer_width_residual",
      {residual, addend, scale_input, weight, bias});
  record_tensor_write(
      norm_output,
      "vulkan_prepack::run_add_scaled_layernorm_context",
      "buffer_width_norm",
      {residual, addend, scale_input, weight, bias});

  return std::make_pair(residual_output, norm_output);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
