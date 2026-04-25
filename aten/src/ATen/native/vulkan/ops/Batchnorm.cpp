#include <c10/core/DispatchKeySet.h>
#include <c10/core/InferenceMode.h>
#include <ATen/ops/batch_norm_ops.h>
#include <ATen/native/vulkan/ops/Batchnorm.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace batchnorm {

struct Params final {
  api::utils::ivec3 out_extents;
  int32_t c4;
  float eps;
};

static void record_op(
    api::Context* const context,
    vTensor& v_output,
    const vTensor& v_input,
    const vTensor& v_weight,
    const vTensor& v_bias,
    const vTensor& v_running_mean,
    const vTensor& v_running_var,
    const float eps) {
  api::PipelineBarrier pipeline_barrier{};

  api::utils::uvec3 global_size = v_output.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  uint32_t num_features = get_dim<Dim4D::Channel>(v_input.sizes());
  uint32_t channels_ext = api::utils::div_up(num_features, 4u);

  Params block{
      api::utils::make_ivec3(v_output.extents()),
      api::utils::safe_downcast<int32_t>(channels_ext),
      eps,
  };

  api::UniformParamsBuffer params(context, block);

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(batchnorm),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_size,
      // local work group size
      local_size,
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_running_mean.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_running_var.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());
}

} // namespace batchnorm

namespace {

using namespace api::utils;

Tensor batch_norm(
    const at::Tensor& input_arg,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    const std::optional<Tensor>& running_mean_opt /* optional */,
    const std::optional<Tensor>& running_var_opt /* optional */,
    bool training,
    double /* momentum, not used in eval mode */,
    double eps,
    bool /* cudnn_enable, deprecated */) {
  TORCH_CHECK(!training, "Only evaluation mode is supported!");

  const Device output_device = input_arg.device();
  Tensor result_cpu;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    const Tensor input_cpu = input_arg.is_vulkan() ? input_arg.cpu() : input_arg;
    const std::optional<Tensor> weight_cpu =
        weight_opt && weight_opt->is_vulkan()
        ? std::optional<Tensor>(weight_opt->cpu())
        : weight_opt;
    const std::optional<Tensor> bias_cpu =
        bias_opt && bias_opt->is_vulkan()
        ? std::optional<Tensor>(bias_opt->cpu())
        : bias_opt;
    const std::optional<Tensor> running_mean_cpu =
        running_mean_opt && running_mean_opt->is_vulkan()
        ? std::optional<Tensor>(running_mean_opt->cpu())
        : running_mean_opt;
    const std::optional<Tensor> running_var_cpu =
        running_var_opt && running_var_opt->is_vulkan()
        ? std::optional<Tensor>(running_var_opt->cpu())
        : running_var_opt;

    result_cpu = at::_ops::batch_norm::call(
        input_cpu,
        weight_cpu,
        bias_cpu,
        running_mean_cpu,
        running_var_cpu,
        training,
        0.0,
        eps,
        false);
  }
  return result_cpu.to(output_device);
}

Tensor batch_norm_autograd_other(
    c10::DispatchKeySet ks,
    const at::Tensor& input_arg,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    bool training,
    double momentum,
    double eps,
    bool cudnn_enable) {
  return at::_ops::batch_norm::redispatch(
      ks & c10::after_autograd_keyset,
      input_arg,
      weight_opt,
      bias_opt,
      running_mean_opt,
      running_var_opt,
      training,
      momentum,
      eps,
      cudnn_enable);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::batch_norm"), TORCH_FN(batch_norm));
}

TORCH_LIBRARY_IMPL(aten, AutogradOther, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::batch_norm"),
      TORCH_FN(batch_norm_autograd_other));
}

#endif /* USE_VULKAN_API */

} // namespace

BatchNormPackedContext::BatchNormPackedContext(
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    double eps) {
  // Each optional tensor arg, if provided should be a 1 dimensional tensor. To
  // achieve more efficient packing as a texture, they are first reshaped to {N,
  // 1, 1}. Eventually this rearrangement should happen automatically in vTensor
  // itself.

  // Weight
  TORCH_CHECK(weight_opt, "Weight must be provided!");
  TORCH_CHECK(weight_opt->dim() == 1, "Weight must have ndim == 1!");

  const int64_t num_features =
      api::utils::safe_downcast<int64_t>(weight_opt->numel());
  const Tensor weight_3d = weight_opt->reshape({num_features, 1, 1});
  weight_ = weight_3d.vulkan();

  // Bias
  TORCH_CHECK(bias_opt, "Bias must be provided!");
  TORCH_CHECK(bias_opt->dim() == 1, "Bias must have ndim == 1!");
  TORCH_CHECK(
      bias_opt->numel() == num_features,
      "Bias must have the same numel as weight!");

  const Tensor bias_3d = bias_opt->reshape({num_features, 1, 1});
  bias_ = bias_3d.vulkan();

  // Running Mean
  TORCH_CHECK(running_mean_opt, "Running mean must be provided!");
  TORCH_CHECK(running_mean_opt->dim() == 1, "Running mean must have ndim == 1");
  TORCH_CHECK(
      running_mean_opt->numel() == num_features,
      "Running mean must have the same numel as weight!");

  const Tensor running_mean_3d =
      running_mean_opt->reshape({num_features, 1, 1});
  running_mean_ = running_mean_3d.vulkan();

  // Running var
  TORCH_CHECK(running_var_opt, "Running var must be provided!");
  TORCH_CHECK(running_var_opt->dim() == 1, "Running var must have ndim == 1");
  TORCH_CHECK(
      running_var_opt->numel() == num_features,
      "Running var must have the same numel as weight!");

  const Tensor running_var_3d = running_var_opt->reshape({num_features, 1, 1});
  running_var_ = running_var_3d.vulkan();

  // Epsilon
  eps_ = eps;
}

BatchNormPackedContext BatchNormPackedContext::pack(
    c10::impl::GenericList unpacked) {
  return BatchNormPackedContext(
      get_optional_tensor(unpacked, ListArgs::kWeight),
      get_optional_tensor(unpacked, ListArgs::kBias),
      get_optional_tensor(unpacked, ListArgs::kRunningMean),
      get_optional_tensor(unpacked, ListArgs::kRunningVar),
      unpacked.get(ListArgs::kEps).toDouble());
}

const c10::impl::GenericList BatchNormPackedContext::unpack() const {
  c10::impl::GenericList unpacked{c10::AnyType::get()};
  unpacked.reserve(ListArgs::kNumArgs);
  unpacked.emplace_back(weight_.cpu().reshape({weight_.numel()}));
  unpacked.emplace_back(bias_.cpu().reshape({bias_.numel()}));
  unpacked.emplace_back(running_mean_.cpu().reshape({running_mean_.numel()}));
  unpacked.emplace_back(running_var_.cpu().reshape({running_var_.numel()}));
  unpacked.emplace_back(eps_);
  return unpacked;
}

c10::intrusive_ptr<BatchNormPackedContext> create_batchnorm_context(
    std::optional<Tensor>&& weight_opt,
    std::optional<Tensor>&& bias_opt,
    std::optional<Tensor>&& running_mean_opt,
    std::optional<Tensor>&& running_var_opt,
    bool training,
    double /* momentum */,
    double eps,
    bool /* cudnn_enable, deprecated */) {
  return c10::make_intrusive<BatchNormPackedContext>(BatchNormPackedContext(
      weight_opt, bias_opt, running_mean_opt, running_var_opt, eps));
}

Tensor run_batchnorm_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<BatchNormPackedContext>& batchnorm_context) {
  api::Context* const context = api::context();

  const vTensor& v_input = convert(input_arg);
  const vTensor& v_weight = convert(batchnorm_context->weight());
  const vTensor& v_bias = convert(batchnorm_context->bias());
  const vTensor& v_running_mean = convert(batchnorm_context->running_mean());
  const vTensor& v_running_var = convert(batchnorm_context->running_var());
  const float eps = api::utils::safe_downcast<float>(batchnorm_context->eps());

  vTensor v_output{
      context,
      v_input.sizes(),
      v_input.dtype(),
  };

  batchnorm::record_op(
      context,
      v_output,
      v_input,
      v_weight,
      v_bias,
      v_running_mean,
      v_running_var,
      eps);

  return convert(v_output);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
