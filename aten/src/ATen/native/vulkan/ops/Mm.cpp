#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/planning/ExecutionObjects.h>

#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <ATen/native/vulkan/api/Tensor.h>
#include <ATen/native/vulkan/api/Types.h>
#include <ATen/native/vulkan/impl/Packing.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/irange.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;
using namespace at::native::vulkan::ops;

constexpr float kGeluBeta =
    static_cast<float>(M_SQRT2 * M_2_SQRTPI * 0.5);

enum class LinearPostOp : uint8_t {
  None,
  Gelu,
};

size_t linear_runtime_scratch_bytes(const Tensor& input) {
  return std::max<size_t>(
      128u * 1024u,
      static_cast<size_t>(std::max<int64_t>(1, input.numel())) *
          sizeof(float) * 4u);
}

Tensor upcast_half_linear_tensor_for_packing(const Tensor& tensor) {
  if (tensor.scalar_type() != kHalf && tensor.scalar_type() != kBFloat16) {
    return tensor;
  }

  if (!tensor.is_vulkan()) {
    return tensor.to(kFloat);
  }

  Tensor cpu_float;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    cpu_float = tensor.cpu().to(kFloat);
  }
  return cpu_float.vulkan();
}

std::optional<Tensor> upcast_half_linear_tensor_for_packing(
    const std::optional<Tensor>& tensor) {
  if (!tensor || !tensor->defined()) {
    return tensor;
  }
  return upcast_half_linear_tensor_for_packing(*tensor);
}

Tensor upload_linear_tensor_to_buffer(
    const Tensor& tensor,
    const api::GPUMemoryLayout memory_layout) {
  const Tensor source = tensor.requires_grad() ? tensor.detach() : tensor;

  if (source.is_vulkan()) {
    return utils::mark_tensor_execution(
        utils::ensure_buffer_storage(source, memory_layout),
        api::ExecutionLayout::BUFFER_DIRECT,
        true);
  }

  TORCH_CHECK(
      source.device().is_cpu(),
      "Vulkan linear buffer prepack expects CPU or Vulkan tensors");
  vTensor v_buffer{
      api::context(),
      source.sizes().vec(),
      convert_dtype(source.scalar_type()),
      api::StorageType::BUFFER,
      memory_layout,
  };
  pack_cpu_to_vulkan(source, v_buffer);
  return utils::mark_tensor_execution(
      convert(v_buffer), api::ExecutionLayout::BUFFER_DIRECT, true);
}

bool is_float_or_half_tensor(const Tensor& tensor) {
  return tensor.scalar_type() == kFloat || tensor.scalar_type() == kHalf ||
      tensor.scalar_type() == kBFloat16;
}

bool can_run_half_buffer_linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  if (
      input.scalar_type() != kHalf ||
      weight.scalar_type() != kHalf ||
      input.dim() < 1 ||
      weight.dim() != 2 ||
      input.requires_grad() ||
      weight.requires_grad() ||
      input.size(-1) != weight.size(1)) {
    return false;
  }

  if (bias && bias->defined()) {
    if (
        bias->requires_grad() ||
        (bias->scalar_type() != kHalf && bias->scalar_type() != kFloat)) {
      return false;
    }
  }

  return true;
}

c10::intrusive_ptr<LinearPackedContext> get_or_create_linear_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  if (utils::has_inference_tensor(weight, bias)) {
    const Tensor prepared_weight =
        (weight.is_vulkan() && weight.dim() == 2) ? weight.cpu().t().contiguous()
                                                  : weight.t();
    return c10::make_intrusive<LinearPackedContext>(
        LinearPackedContext(
            prepared_weight,
            bias,
            false,
            std::string(),
            false));
  }

  if (const auto cached_context = utils::lookup_linear_context(weight, bias)) {
    return *cached_context;
  }

  const Tensor prepared_weight =
      (c10::InferenceMode::is_enabled() && weight.is_vulkan() &&
       weight.dim() == 2)
      ? weight.cpu().t().contiguous()
      : weight.t();
  const auto context = c10::make_intrusive<LinearPackedContext>(
      LinearPackedContext(
          prepared_weight,
          bias,
          false,
          std::string(),
          false));
  utils::store_linear_context(weight, bias, context);
  return context;
}

inline bool has_bias(const std::optional<Tensor>& bias) {
  return bias && bias->defined();
}

struct LinearPackedRunState final {
  const PackedWeightHandle& packed_weight;
  const vTensor& packed_v_weight;
  const vTensor& packed_v_bias;
  const std::vector<int64_t>& logical_weight_sizes;
  bool bias_defined;
};

LinearPackedRunState get_linear_packed_run_state(
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  const PackedWeightHandle& packed_weight = linear_context->packed_weight();
  return {
      packed_weight,
      packed_weight.weight_vtensor(),
      packed_weight.bias_vtensor(),
      packed_weight.logical_weight_sizes(),
      packed_weight.has_bias(),
  };
}

Tensor ensure_linear_buffer_output_tensor(
    Tensor& output,
    IntArrayRef sizes,
    const c10::ScalarType dtype) {
  bool needs_allocation = !output.defined() || !output.is_vulkan() ||
      output.scalar_type() != dtype || !output.sizes().equals(sizes);
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
            sizes.vec(),
            convert_dtype(dtype),
            api::StorageType::BUFFER,
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
        }),
        api::ExecutionLayout::BUFFER_DIRECT);
  }
  return output;
}

bool can_fuse_linear_bias(
    const vTensor& v_output,
    const vTensor& v_bias,
    const std::vector<int64_t>& weight_sizes) {
  if (
      v_bias.storage_type() != api::StorageType::TEXTURE_3D ||
      v_bias.gpu_memory_layout() !=
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED) {
    return false;
  }

  const IntArrayRef bias_sizes = v_bias.sizes();
  if (bias_sizes.empty() || bias_sizes.size() > 2) {
    return false;
  }

  const int64_t output_width = weight_sizes[Layout::Parameter::width];
  const int64_t output_height = v_output.sizes()[Layout::Parameter::height];
  const int64_t bias_width = bias_sizes.back();
  const int64_t bias_height =
      bias_sizes.size() == 2 ? bias_sizes.front() : 1;

  return bias_width == output_width &&
      (bias_height == 1 || bias_height == output_height);
}

bool can_use_channel_packed_linear_input(
    const vTensor& v_input,
    const vTensor& packed_v_weight) {
  return v_input.dtype() == api::kFloat &&
      v_input.storage_type() == api::StorageType::TEXTURE_3D &&
      v_input.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED &&
      v_input.sizes().size() == 2 &&
      !v_input.is_quantized() &&
      packed_v_weight.dtype() == api::kFloat &&
      packed_v_weight.storage_type() == api::StorageType::TEXTURE_3D &&
      packed_v_weight.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED &&
      !packed_v_weight.is_quantized();
}

bool linear_kernel_family_allows_channel_packed_input(
    const utils::VulkanRuntimePolicy& runtime_policy) {
  if (
      runtime_policy.request.model_domain == utils::VulkanModelDomain::Generic &&
      runtime_policy.request.execution_phase ==
          utils::VulkanExecutionPhase::None) {
    return true;
  }

  switch (runtime_policy.linear_kernel_family) {
    case utils::VulkanLinearKernelFamily::TexturePacked:
      return false;
    case utils::VulkanLinearKernelFamily::UnifiedBufferView:
    case utils::VulkanLinearKernelFamily::PersistentPackedTexture:
      return true;
  }
  return true;
}

Tensor reshape_linear_output_if_needed(
    const Tensor& output,
    const Tensor& input_arg) {
  if (input_arg.dim() == 2) {
    return output;
  }

  std::vector<int64_t> shape;
  shape.reserve(static_cast<size_t>(std::max<int64_t>(0, input_arg.dim())));
  for (const auto i : c10::irange(input_arg.dim() - 1)) {
    shape.emplace_back(input_arg.size(i));
  }
  shape.emplace_back(output.size(-1));
  Tensor reshaped_output = utils::reshape_inference(output, shape);
  if (c10::InferenceMode::is_enabled() && reshaped_output.is_vulkan()) {
    const vTensor& v_reshaped_output = convert(reshaped_output);
    const bool needs_materialization =
        v_reshaped_output.storage_type() == api::StorageType::BUFFER &&
        !v_reshaped_output.has_direct_buffer_layout();
    if (needs_materialization) {
      reshaped_output = reshaped_output.clone();
    }
  } else if (c10::InferenceMode::is_enabled()) {
    reshaped_output = reshaped_output.clone();
  }
  return reshaped_output;
}

Tensor& ensure_linear_output_tensor(
    Tensor& output,
    IntArrayRef sizes,
    const c10::ScalarType dtype) {
  bool needs_allocation = !output.defined() || !output.is_vulkan() ||
      output.scalar_type() != dtype || !output.sizes().equals(sizes);
  if (!needs_allocation) {
    const vTensor& v_output = convert(output);
    needs_allocation =
        v_output.storage_type() != api::StorageType::TEXTURE_3D;
  }
  if (needs_allocation) {
    output = convert(vTensor{
        api::context(),
        sizes.vec(),
        convert_dtype(dtype),
    });
  }
  return output;
}

bool can_run_float_buffer_linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  if (
      input.device().type() != c10::DeviceType::Vulkan ||
      weight.device().type() != c10::DeviceType::Vulkan ||
      input.scalar_type() != kFloat ||
      weight.scalar_type() != kFloat ||
      input.dim() != 2 ||
      weight.dim() != 2 ||
      input.requires_grad() ||
      weight.requires_grad() ||
      input.size(Layout::Parameter::width) !=
          weight.size(Layout::Parameter::height)) {
    return false;
  }

  const vTensor& v_input = convert(input);
  const vTensor& v_weight = convert(weight);
  if (
      v_input.storage_type() != api::StorageType::BUFFER ||
      v_weight.storage_type() != api::StorageType::BUFFER ||
      v_input.gpu_memory_layout() != api::GPUMemoryLayout::TENSOR_WIDTH_PACKED ||
      v_weight.gpu_memory_layout() !=
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED ||
      !utils::supports_buffer_view_fast_path(v_input) ||
      !utils::supports_buffer_view_fast_path(v_weight)) {
    return false;
  }

  if (bias && bias->defined()) {
    if (
        bias->device().type() != c10::DeviceType::Vulkan ||
        bias->dim() > 2 ||
        bias->requires_grad() ||
        bias->scalar_type() != kFloat) {
      return false;
    }

    const vTensor& v_bias = convert(*bias);
    if (
        v_bias.storage_type() != api::StorageType::BUFFER ||
        !utils::supports_buffer_view_fast_path(v_bias)) {
      return false;
    }
  }

  return true;
}

bool can_run_float_buffer_bmm(const Tensor& mat1, const Tensor& mat2) {
  if (
      mat1.device().type() != c10::DeviceType::Vulkan ||
      mat2.device().type() != c10::DeviceType::Vulkan ||
      mat1.scalar_type() != kFloat ||
      mat2.scalar_type() != kFloat ||
      mat1.dim() != 3 ||
      mat2.dim() != 3 ||
      mat1.requires_grad() ||
      mat2.requires_grad() ||
      mat1.size(Layout::BatchMatrices::batch) !=
          mat2.size(Layout::BatchMatrices::batch) ||
      mat1.size(Layout::BatchMatrices::width) !=
          mat2.size(Layout::BatchMatrices::height)) {
    return false;
  }

  const vTensor& v_mat1 = convert(mat1);
  const vTensor& v_mat2 = convert(mat2);
  return v_mat1.storage_type() == api::StorageType::BUFFER &&
      v_mat2.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_view_fast_path(v_mat1) &&
      utils::supports_buffer_view_fast_path(v_mat2);
}

Tensor run_float_buffer_linear(
    const Tensor& input_arg,
    const Tensor& input_arg_2d,
    const LinearPackedRunState& packed_state,
    const float alpha,
    const float beta,
    const LinearPostOp post_op,
    Tensor* output_opt = nullptr) {
  api::Context* const context = api::context();

  const Tensor& packed_weight_tensor = packed_state.packed_weight.weight();
  const std::optional<Tensor> packed_bias_tensor = packed_state.bias_defined
      ? std::optional<Tensor>(packed_state.packed_weight.bias())
      : std::nullopt;

  TORCH_INTERNAL_ASSERT(
      can_run_float_buffer_linear(
          input_arg_2d, packed_weight_tensor, packed_bias_tensor));

  Tensor input_tensor = input_arg_2d;
  Tensor weight_tensor = packed_weight_tensor;
  vTensor& v_input = convert(input_tensor);
  vTensor& v_weight = convert(weight_tensor);
  const std::vector<int64_t> output_sizes{
      input_arg_2d.sizes()[Layout::Parameter::height],
      packed_state.logical_weight_sizes[Layout::Parameter::width],
  };
  Tensor output_tensor = output_opt
      ? ensure_linear_buffer_output_tensor(
            *output_opt, output_sizes, input_arg_2d.scalar_type())
      : utils::mark_tensor_execution(
            convert(vTensor{
                context,
                output_sizes,
                api::kFloat,
                api::StorageType::BUFFER,
                api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
            }),
            api::ExecutionLayout::BUFFER_DIRECT);
  vTensor& v_output = convert(output_tensor);

  const struct {
    int32_t out_width;
    int32_t out_height;
    int32_t inner_dim;
    int32_t reserved;
  } block{
      api::utils::safe_downcast<int32_t>(
          packed_state.logical_weight_sizes[Layout::Parameter::width]),
      api::utils::safe_downcast<int32_t>(
          input_arg_2d.size(Layout::Parameter::height)),
      api::utils::safe_downcast<int32_t>(
          input_arg_2d.size(Layout::Parameter::width)),
      0,
  };
  Tensor fused_bias_tensor;
  bool fuse_buffer_bias_gelu = false;
  if (
      post_op == LinearPostOp::Gelu && packed_state.bias_defined &&
      alpha == 1.0f && beta == 1.0f) {
    fused_bias_tensor = packed_state.packed_weight.bias();
    const vTensor& v_bias = convert(fused_bias_tensor);
    fuse_buffer_bias_gelu =
        v_bias.storage_type() == api::StorageType::BUFFER &&
        v_bias.gpu_memory_layout() ==
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
        v_bias.has_direct_buffer_layout() && v_bias.sizes().size() == 1 &&
        v_bias.sizes()[0] == output_sizes[Layout::Parameter::width];
  }

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(
          packed_state.logical_weight_sizes[Layout::Parameter::width]),
      api::utils::safe_downcast<uint32_t>(
          input_arg_2d.size(Layout::Parameter::height)),
      1u,
  };

  if (fuse_buffer_bias_gelu) {
    vTensor& v_bias = convert(fused_bias_tensor);
    utils::log_vulkan_op_hit("aten::linear.buffer_float_bias_gelu");
    context->submit_compute_job(
        VK_KERNEL(mm_buffer_float_bias_gelu),
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
        v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_weight.buffer_metadata(),
        v_bias.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_bias.buffer_metadata(),
        params.buffer());
  } else {
    context->submit_compute_job(
        VK_KERNEL(mm_buffer_float),
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
        v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_weight.buffer_metadata(),
        params.buffer());
  }

  Tensor output = output_tensor;
  if (!fuse_buffer_bias_gelu && alpha != 1.0f) {
    output = output.mul(alpha);
  }
  if (!fuse_buffer_bias_gelu && packed_state.bias_defined) {
    Tensor bias = packed_state.packed_weight.bias();
    if (beta != 1.0f) {
      bias = bias.mul(beta);
    }
    output = output.add(bias);
  }
  if (!fuse_buffer_bias_gelu && post_op == LinearPostOp::Gelu) {
    output = at::gelu(output, "none");
  }
  if (output_opt && output.unsafeGetTensorImpl() != output_tensor.unsafeGetTensorImpl()) {
    *output_opt = output;
    output = *output_opt;
  }

  return reshape_linear_output_if_needed(output, input_arg);
}

Tensor run_float_buffer_bmm(
    const Tensor& mat1_arg,
    const Tensor& mat2_arg,
    const float alpha,
    const float beta,
    const std::optional<Tensor>& bias = std::nullopt) {
  api::Context* const context = api::context();
  TORCH_INTERNAL_ASSERT(can_run_float_buffer_bmm(mat1_arg, mat2_arg));

  Tensor mat1 = mat1_arg;
  Tensor mat2 = mat2_arg;
  vTensor& v_mat1 = convert(mat1);
  vTensor& v_mat2 = convert(mat2);

  const std::vector<int64_t> output_sizes{
      mat1.size(Layout::BatchMatrices::batch),
      mat1.size(Layout::BatchMatrices::height),
      mat2.size(Layout::BatchMatrices::width),
  };
  Tensor output = utils::mark_tensor_execution(
      convert(vTensor{
          context,
          output_sizes,
          api::kFloat,
          api::StorageType::BUFFER,
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      }),
      api::ExecutionLayout::BUFFER_DIRECT);
  vTensor& v_output = convert(output);

  const struct {
    int32_t out_width;
    int32_t out_height;
    int32_t inner_dim;
    int32_t batch;
  } block{
      api::utils::safe_downcast<int32_t>(
          mat2.size(Layout::BatchMatrices::width)),
      api::utils::safe_downcast<int32_t>(
          mat1.size(Layout::BatchMatrices::height)),
      api::utils::safe_downcast<int32_t>(
          mat1.size(Layout::BatchMatrices::width)),
      api::utils::safe_downcast<int32_t>(
          mat1.size(Layout::BatchMatrices::batch)),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(
          mat2.size(Layout::BatchMatrices::width)),
      api::utils::safe_downcast<uint32_t>(
          mat1.size(Layout::BatchMatrices::height)),
      api::utils::safe_downcast<uint32_t>(
          mat1.size(Layout::BatchMatrices::batch)),
  };

  context->submit_compute_job(
      VK_KERNEL(bmm_buffer_float),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_output.buffer_metadata(),
      v_mat1.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_mat1.buffer_metadata(),
      v_mat2.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_mat2.buffer_metadata(),
      params.buffer());

  if (alpha != 1.0f) {
    output = output.mul(alpha);
  }
  if (bias && bias->defined()) {
    Tensor bias_tensor = bias->is_vulkan() ? *bias : bias->vulkan();
    if (beta != 1.0f) {
      bias_tensor = bias_tensor.mul(beta);
    }
    output = output.add(bias_tensor);
  }
  return output;
}

bool can_run_half_buffer_bmm(const Tensor& mat1, const Tensor& mat2) {
  return mat1.scalar_type() == kHalf && mat2.scalar_type() == kHalf &&
      mat1.dim() == 3 && mat2.dim() == 3 && !mat1.requires_grad() &&
      !mat2.requires_grad() &&
      mat1.size(Layout::BatchMatrices::batch) ==
          mat2.size(Layout::BatchMatrices::batch) &&
      mat1.size(Layout::BatchMatrices::width) ==
          mat2.size(Layout::BatchMatrices::height);
}

Tensor widen_half_bmm_tensor_to_float_buffer(const Tensor& tensor) {
  Tensor widened = upcast_half_linear_tensor_for_packing(tensor);
  Tensor vulkan_widened = widened.is_vulkan() ? widened : widened.vulkan();
  return utils::mark_tensor_execution(
      utils::ensure_buffer_storage(
          vulkan_widened, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
      api::ExecutionLayout::BUFFER_DIRECT);
}

Tensor run_half_buffer_bmm(
    const Tensor& mat1,
    const Tensor& mat2,
    const float alpha,
    const float beta,
    const std::optional<Tensor>& bias = std::nullopt) {
  const Tensor float_mat1 = widen_half_bmm_tensor_to_float_buffer(mat1);
  const Tensor float_mat2 = widen_half_bmm_tensor_to_float_buffer(mat2);
  const std::optional<Tensor> float_bias =
      upcast_half_linear_tensor_for_packing(bias);
  return run_float_buffer_bmm(
      float_mat1, float_mat2, alpha, beta, float_bias);
}

Tensor run_addmm_context_channel_packed_input(
    const Tensor& input_arg,
    const Tensor& input_2d,
    const vTensor& v_input,
    const LinearPackedRunState& packed_state,
    const float alpha,
    const float beta,
    const LinearPostOp post_op,
    Tensor* output_opt = nullptr) {
  api::Context* const context = api::context();
  const vTensor& packed_v_weight = packed_state.packed_v_weight;
  const vTensor& packed_v_bias = packed_state.packed_v_bias;
  const std::vector<int64_t>& unpacked_weight_sizes =
      packed_state.logical_weight_sizes;
  const bool bias_defined = packed_state.bias_defined;
  const std::vector<int64_t> output_sizes{
      input_2d.sizes()[Layout::Parameter::height],
      unpacked_weight_sizes[Layout::Parameter::width],
  };
  Tensor output_tensor = output_opt
      ? ensure_linear_output_tensor(
            *output_opt, output_sizes, convert_dtype(v_input.dtype()))
      : convert(vTensor{context, output_sizes, v_input.dtype()});
  vTensor& v_output = convert(output_tensor);

  api::UniformParamsBuffer params;
  api::ShaderInfo compute_shader;
  const int step_size =
      div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(4));
  const bool fuse_bias =
      bias_defined &&
      can_fuse_linear_bias(v_output, packed_v_bias, unpacked_weight_sizes);
  const bool fuse_gelu = fuse_bias && post_op == LinearPostOp::Gelu;
  const api::utils::ivec4 input_sizes =
      api::utils::make_ivec4_prepadded1(v_input.sizes());

  if (fuse_gelu) {
    const struct {
      uvec4 shader_extents_and_step;
      ivec4 input_sizes;
      uvec4 bias_extents;
      vec4 multipliers_and_gelu;
    } block_with_bias_gelu{
        {
            v_output.extents().data[0u],
            v_output.extents().data[1u],
            v_output.extents().data[2u],
            safe_downcast<uint32_t>(step_size),
        },
        input_sizes,
        {
            packed_v_bias.extents().data[0u],
            packed_v_bias.extents().data[1u],
            packed_v_bias.extents().data[2u],
            0u,
        },
        {alpha, beta, kGeluBeta, 0.0f},
    };
    params = api::UniformParamsBuffer(context, block_with_bias_gelu);
    compute_shader = VK_KERNEL(mm_bias_gelu_channel_packed_input);
  } else if (fuse_bias) {
    const struct {
      uvec4 shader_extents_and_step;
      ivec4 input_sizes;
      uvec4 bias_extents;
      vec2 multipliers;
    } block_with_bias{
        {
            v_output.extents().data[0u],
            v_output.extents().data[1u],
            v_output.extents().data[2u],
            safe_downcast<uint32_t>(step_size),
        },
        input_sizes,
        {
            packed_v_bias.extents().data[0u],
            packed_v_bias.extents().data[1u],
            packed_v_bias.extents().data[2u],
            0u,
        },
        {alpha, beta},
    };
    params = api::UniformParamsBuffer(context, block_with_bias);
    compute_shader = VK_KERNEL(mm_bias_channel_packed_input);
  } else {
    const struct {
      uvec4 shader_extents_and_step;
      ivec4 input_sizes;
    } block_no_bias{
        {
            v_output.extents().data[0u],
            v_output.extents().data[1u],
            v_output.extents().data[2u],
            safe_downcast<uint32_t>(step_size),
        },
        input_sizes,
    };
    params = api::UniformParamsBuffer(context, block_no_bias);
    compute_shader = VK_KERNEL(mm_channel_packed_input);
  }

  api::PipelineBarrier pipeline_barrier{};
  if (fuse_bias) {
    context->submit_compute_job(
        compute_shader,
        pipeline_barrier,
        {
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::width], INT64_C(4))),
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::height], INT64_C(4))),
            1,
        },
        {8, 8, 1},
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  } else {
    context->submit_compute_job(
        compute_shader,
        pipeline_barrier,
        {
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::width], INT64_C(4))),
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::height], INT64_C(4))),
            1,
        },
        {8, 8, 1},
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  }

  Tensor output = output_tensor;
  if (!fuse_bias && alpha != 1.0f) {
    output = output.mul(alpha);
  }
  if (!fuse_bias && bias_defined) {
    output = output.add(convert(packed_v_bias).mul(beta));
  }
  if (post_op == LinearPostOp::Gelu && !fuse_gelu) {
    output = at::gelu(output, "none");
  }
  if (output_opt && output.unsafeGetTensorImpl() != output_tensor.unsafeGetTensorImpl()) {
    output = rebind_vulkan_output(*output_opt, output);
  }

  return reshape_linear_output_if_needed(output, input_arg);
}

vTensor pack_cpu_float_weight_using_height_packing(const Tensor& weight_arg) {
  TORCH_INTERNAL_ASSERT(weight_arg.is_cpu());
  TORCH_INTERNAL_ASSERT(weight_arg.scalar_type() == kFloat);
  TORCH_INTERNAL_ASSERT(weight_arg.dim() == 2);

  api::Context* const context = api::context();
  const Tensor weight = weight_arg.contiguous();
  const int64_t height = weight.size(Layout::Parameter::height);
  const int64_t width = weight.size(Layout::Parameter::width);

  vTensor v_weight{
      context,
      weight.sizes().vec(),
      convert_dtype(weight.scalar_type()),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
  };

  api::StorageBuffer staging(context, api::kFloat, v_weight.gpu_numel());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
    float* const dst = mapping.template data<float>();
    const float* const src = weight.const_data_ptr<float>();
    std::fill_n(dst, v_weight.gpu_numel(), 0.0f);

    const api::utils::uvec3 extents = v_weight.extents();
    const int64_t texel_width =
        static_cast<int64_t>(extents.data[0u]);
    const int64_t texel_height =
        static_cast<int64_t>(extents.data[1u]);
    const int64_t texel_depth =
        static_cast<int64_t>(extents.data[2u]);

    for (const auto z : c10::irange(texel_depth)) {
      for (const auto y : c10::irange(texel_height)) {
        const int64_t src_base_h = y * 4;
        for (const auto x : c10::irange(texel_width)) {
          const int64_t texel_base =
              (((z * texel_height) + y) * texel_width + x) * 4;
          for (const auto c : c10::irange(int64_t{4})) {
            const int64_t src_h = src_base_h + c;
            if (src_h < height && x < width) {
              dst[texel_base + c] = src[src_h * width + x];
            }
          }
        }
      }
    }
  }

  api::PipelineBarrier pipeline_barrier{};
  utils::copy_buffer_to_vtensor(staging.buffer(), v_weight, pipeline_barrier);
  return v_weight;
}

vTensor pack_inputs_using_width_packing(
    const Tensor& input_arg,
    const utils::VulkanPlanningRequest& input_request) {
  TORCH_INTERNAL_ASSERT(
      !input_arg.is_quantized(),
      "Vulkan Linear not usable! "
      "Reason: Input packing only supports non-quantized tensors.");
  TORCH_INTERNAL_ASSERT(
      input_arg.dim() == 2 || input_arg.dim() == 3,
      "Vulkan Linear not usable! "
      "Reason: Input packing only supports 2D or 3D tensors.");

  const Tensor input = utils::prepare_vulkan_execution_tensor(
      input_arg,
      utils::VulkanExecutionPlanKind::LinearPackedInput,
      input_request);

  vTensor v_input = convert(input);

  TORCH_CHECK(
      v_input.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      "After packing, the v_input must be in TENSOR_WIDTH_PACKED format");

  return v_input;
}

vTensor pack_inputs_using_width_packing(const Tensor& input_arg) {
  return pack_inputs_using_width_packing(
      input_arg,
      utils::make_vulkan_tensor_linear_request(
          input_arg, utils::VulkanTensorRole::Input));
}

vTensor pack_weights_using_height_packing(const Tensor& weight_arg) {
  // Only non-batch, non-quantized tensors are supported
  TORCH_INTERNAL_ASSERT(
      !weight_arg.is_quantized(),
      "Vulkan Linear not usable! "
      "Reason: Weight packing only supports non-quantized tensors.");
  TORCH_INTERNAL_ASSERT(
      weight_arg.dim() == 2 || weight_arg.dim() == 3,
      "Vulkan Linear not usable! "
      "Reason: Weight packing only supports 2D or 3D tensors.");

  if (weight_arg.is_cpu() && weight_arg.scalar_type() == kFloat &&
      weight_arg.dim() == 2) {
    return pack_cpu_float_weight_using_height_packing(weight_arg);
  }

  const Tensor weight = utils::prepare_vulkan_execution_tensor(
      weight_arg,
      utils::VulkanExecutionPlanKind::LinearPackedWeight,
      utils::make_vulkan_linear_request(utils::VulkanTensorRole::Weight));

  vTensor v_weight = convert(weight);

  TORCH_CHECK(
      v_weight.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
      "After packing, the v_weight must be in TENSOR_HEIGHT_PACKED format");

  return v_weight;
}

vTensor pack_weights(const Tensor& weight_arg, const bool use_batch = false) {
  if (!weight_arg.is_quantized()) {
    return pack_weights_using_height_packing(weight_arg);
  }

  TORCH_CHECK(
      weight_arg.is_quantized(), "Only quantized weights logic after here");

  // Rest of the logic are either quantized or batched.

  api::Context* const context = api::context();

  const Tensor weight = weight_arg.contiguous();
  const IntArrayRef w_sizes = weight.sizes();
  if (use_batch) {
    TORCH_CHECK(
        w_sizes.size() == 3,
        "Vulkan Linear not usable! "
        "Reason: Unable to perform weight packing with batch; the input tensor of a batch of matrices should contain 3 dimensions: batch, height, width.");
  }
  /* Source */
  int64_t src_kb_sz = 0;
  int64_t src_kw_sz = 0;
  int64_t src_kh_sz = 0;
  /* Destination */
  int64_t dst_kb_sz = 0;
  int64_t dst_kw_sz = 0;
  int64_t dst_kh_sz = 0;
  std::vector<int64_t> dst_vtensor_sizes;
  /* Source */
  src_kb_sz = use_batch ? w_sizes[Layout::BatchMatrices::batch] : 1;
  src_kw_sz = use_batch ? w_sizes[Layout::BatchMatrices::width]
                        : w_sizes[Layout::Parameter::width];
  src_kh_sz = use_batch ? w_sizes[Layout::BatchMatrices::height]
                        : w_sizes[Layout::Parameter::height];

  /* Destination */
  dst_kb_sz = src_kb_sz;
  dst_kw_sz = div_up(src_kw_sz, INT64_C(2));
  dst_kh_sz = div_up(src_kh_sz, INT64_C(2));
  dst_vtensor_sizes = {
      dst_kb_sz,
      4,
      dst_kh_sz,
      dst_kw_sz,
  };

  vTensor v_weight{
      context, dst_vtensor_sizes, convert_dtype(weight_arg.scalar_type())};

  v_weight.set_is_quantized();
  v_weight.set_scale(weight_arg.q_scale());
  v_weight.set_zero_point(weight_arg.q_zero_point());

  stage_pack_weights<int8_t>(
      context,
      v_weight,
      weight,
      src_kb_sz,
      src_kh_sz,
      src_kw_sz,
      dst_kh_sz,
      dst_kw_sz);
  return v_weight;
}

vTensor pack_biases(
    const Tensor& weight_arg,
    const std::optional<Tensor>& bias_arg,
    const bool use_batch = false) {
  if (has_bias(bias_arg)) {
    Tensor bias = utils::prepare_vulkan_execution_tensor(
        *bias_arg,
        utils::VulkanExecutionPlanKind::LinearPackedBias,
        utils::make_vulkan_linear_request(utils::VulkanTensorRole::Bias));
    return convert(bias);
  } else {
    return convert(at::zeros({1}, at::device(at::kVulkan).dtype(at::kFloat)));
  }
}

// Old version of pack_biases that fixes issues with quantization and to be
// removed in the future.
vTensor pack_biases_quantized_weights(
    const Tensor& weight_arg,
    const std::optional<Tensor>& bias_arg,
    const bool use_batch = false) {
  TORCH_CHECK(
      weight_arg.is_quantized(),
      "pack_biases_quantized to be used only when using quantized linear ops");

  if (has_bias(bias_arg) && bias_arg->is_vulkan()) {
    Tensor bias = utils::prepare_vulkan_execution_tensor(
        *bias_arg,
        utils::VulkanExecutionPlanKind::TextureComputeInput,
        utils::make_vulkan_linear_request(utils::VulkanTensorRole::Bias));
    return convert(bias);
  }

  api::Context* const context = api::context();

  if (has_bias(bias_arg)) {
    const Tensor bias = bias_arg->contiguous();
    const IntArrayRef b_sizes = bias.sizes();
    const float* const src_bias_ptr = bias.const_data_ptr<float>();

    /* Source */
    int64_t src_kb_sz = 0;
    int64_t src_kw_sz = 0;
    int64_t src_kh_sz = 0;
    if (use_batch) {
      if (bias.sizes().size() == 3) {
        src_kb_sz = b_sizes[Layout::BatchMatrices::batch];
        src_kw_sz = b_sizes[Layout::BatchMatrices::width];
        src_kh_sz = b_sizes[Layout::BatchMatrices::height];
      } else if (bias.sizes().size() == 2) {
        // skip batch dim for broadcasting; index -1
        src_kb_sz = 1;
        src_kw_sz = b_sizes[Layout::BatchMatrices::height];
        src_kh_sz = b_sizes[Layout::BatchMatrices::batch];
      } else {
        // skip batch & height dim for broadcasting; index -2
        src_kb_sz = 1;
        src_kw_sz = b_sizes[Layout::BatchMatrices::batch];
        src_kh_sz = 1;
      }
    } else {
      src_kb_sz = 1;
      if (bias.sizes().size() == 2) {
        src_kw_sz = b_sizes[Layout::Parameter::width];
        src_kh_sz = b_sizes[Layout::Parameter::height];
      } else {
        src_kw_sz = b_sizes[Layout::Parameter::height];
        src_kh_sz = 1;
      }
    }
    const int64_t src_matrix_sz = src_kw_sz * src_kh_sz;

    /* Destination */
    const int64_t dst_kw_sz = div_up(src_kw_sz, INT64_C(2));
    const int64_t dst_kh_sz = div_up(src_kh_sz, INT64_C(2));
    const int64_t dst_plane_sz = dst_kw_sz * dst_kh_sz;
    const int64_t dst_matrix_sz = dst_plane_sz * 4;

    vTensor v_bias{
        context,
        {
            src_kb_sz,
            4,
            dst_kh_sz,
            dst_kw_sz,
        },
        convert_dtype(bias_arg->scalar_type()),
    };

    api::StorageBuffer staging(
        context, api::ScalarType::Float, v_bias.gpu_numel());
    {
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

      float* dst_bias_ptr = mapping.template data<float>();

      memset(dst_bias_ptr, 0, v_bias.nbytes());

      for (const auto src_b : c10::irange(src_kb_sz)) {
        for (const auto src_h : c10::irange(src_kh_sz == 1 ? 2 : src_kh_sz)) {
          for (const auto src_w :
               c10::irange((use_batch && src_kw_sz == 1) ? 2 : src_kw_sz)) {
            int64_t dst_plane = 2 * (src_h % 2) + (src_w % 2);
            int64_t dst_index = (src_h / 2) * dst_kw_sz + (src_w / 2);
            memcpy(
                dst_bias_ptr + src_b * dst_matrix_sz +
                    dst_plane * dst_plane_sz + dst_index,
                src_bias_ptr + src_b * src_matrix_sz +
                    (src_kh_sz == 1 ? 0 : src_h * src_kw_sz) +
                    ((use_batch && src_kw_sz == 1) ? 0 : src_w),
                sizeof(float));
          }
        }
      }
    }
    utils::pack_staging_to_vtensor(staging.buffer(), v_bias);

    return v_bias;
  } else {
    vTensor v_bias{
        api::context(),
        {1},
        convert_dtype(weight_arg.scalar_type()),
    };

    api::StorageBuffer staging(
        context, api::ScalarType::Float, v_bias.gpu_numel());
    {
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

      float* data_ptr = mapping.template data<float>();

      memset(
          data_ptr,
          // 2's complement integers and IEEE-754 floating point numbers both
          // have identical bit representations for 0, so can use memset which
          // only accepts uint8_t parameter.
          0,
          v_bias.nbytes());
    }
    utils::pack_staging_to_vtensor(staging.buffer(), v_bias);

    return v_bias;
  }
}

bool available_check_with_batch(
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  const bool weight_available = (3 == weight.ndimension()) &&
      (weight.size(Layout::BatchMatrices::batch) > 0) &&
      (weight.size(Layout::BatchMatrices::height) > 0) &&
      (weight.size(Layout::BatchMatrices::width) > 0) &&
      ((weight.device().is_cpu()) ||
       (c10::DeviceType::Vulkan == weight.device().type())) &&
      (kFloat == weight.scalar_type() || kHalf == weight.scalar_type());
  if (!weight_available) {
    return false;
  }

  if (!bias || !bias->defined()) {
    // no need to check bias since it is not used.
    return true;
  }

  bool bias_available = true;
  bias_available &= (bias->ndimension() > 0);
  bias_available &=
      ((bias->device().is_cpu()) ||
       (c10::DeviceType::Vulkan == bias->device().type()));
  bias_available &=
      (kFloat == bias->scalar_type() || kHalf == bias->scalar_type());
  // Only check the consistency of batch and width dimension. The height
  // dimension consistency is unchecked, due to the 2nd input which determines
  // the height is not passed into LinearPackedContext.
  if (bias->ndimension() == 3) {
    bias_available &=
        (bias->size(Layout::BatchMatrices::width) ==
             weight.size(Layout::BatchMatrices::width) ||
         bias->size(Layout::BatchMatrices::width) == 1);
    bias_available &=
        (bias->size(Layout::BatchMatrices::batch) ==
             weight.size(Layout::BatchMatrices::batch) ||
         bias->size(Layout::BatchMatrices::batch) == 1);
  } else if (bias->ndimension() == 2) {
    // skip batch dim for broadcasting; index -1
    bias_available &=
        (bias->size(Layout::BatchMatrices::height) ==
             weight.size(Layout::BatchMatrices::width) ||
         bias->size(Layout::BatchMatrices::height) == 1);
  } else {
    // skip batch & height dim for broadcasting; index -2
    bias_available &=
        (bias->size(Layout::BatchMatrices::batch) ==
             weight.size(Layout::BatchMatrices::width) ||
         bias->size(Layout::BatchMatrices::batch) == 1);
  }
  return bias_available;
}

bool available(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const bool use_batch = false) {
  if (!api::available()) {
    return false;
  }

  if (use_batch) {
    return available_check_with_batch(weight, bias);
  }

  const bool weight_available = (2 == weight.ndimension()) &&
      (weight.size(Layout::Parameter::height) > 0) &&
      (weight.size(Layout::Parameter::width) > 0) &&
      ((weight.device().is_cpu()) ||
       (c10::DeviceType::Vulkan == weight.device().type())) &&
      (kFloat == weight.scalar_type() || kHalf == weight.scalar_type() ||
       kBFloat16 == weight.scalar_type() ||
       kQInt8 == weight.scalar_type());
  if (!weight_available) {
    return false;
  }

  const bool bias_available =
      ((bias && bias.has_value() && bias->defined())
           ? ((bias->ndimension() > 0) &&
              ((bias->device().is_cpu()) ||
               (c10::DeviceType::Vulkan == bias->device().type())) &&
              (kFloat == bias->scalar_type() ||
               kHalf == bias->scalar_type() ||
               kBFloat16 == bias->scalar_type()) &&
              ((bias->ndimension() > 1)
                   ? (bias->size(Layout::Parameter::width) ==
                      weight.size(Layout::Parameter::width))
                   : true))
           : true);
  return bias_available;
}

bool usable_check_with_batch(
    const Tensor& input,
    const IntArrayRef unpacked_weight_sizes) {
  return (3 == input.ndimension()) &&
      (c10::DeviceType::Vulkan == input.device().type()) &&
      (kFloat == input.scalar_type() || kHalf == input.scalar_type()) &&
      (input.size(Layout::BatchMatrices::width) ==
       unpacked_weight_sizes[Layout::BatchMatrices::height]) &&
      (input.size(Layout::BatchMatrices::batch) ==
       unpacked_weight_sizes[Layout::BatchMatrices::batch]) &&
      !input.requires_grad() && true;
}

bool usable(
    const Tensor& input,
    const IntArrayRef unpacked_weight_sizes,
    const bool use_batch = false) {
  if (use_batch) {
    return usable_check_with_batch(input, unpacked_weight_sizes);
  }
  const auto v_input = convert(input);
  return (2 == input.ndimension()) &&
      (c10::DeviceType::Vulkan == input.device().type()) &&
      ((kFloat == input.scalar_type()) || (kHalf == input.scalar_type()) ||
       (v_input.is_quantized() &&
        (kQUInt8 == input.scalar_type() || kQInt8 == input.scalar_type()))) &&
      (input.size(Layout::Parameter::width) ==
       unpacked_weight_sizes[Layout::Parameter::height]) &&
      !input.requires_grad() && true;
}

static Tensor reshape_to_2d(const Tensor& input_arg) {
  TORCH_CHECK(
      input_arg.dim() >= 1,
      "Vulkan Linear op only supports input tensor with dim >= 1");

  Tensor reshape_input = input_arg;
  if (input_arg.is_vulkan() && c10::InferenceMode::is_enabled()) {
    const vTensor& v_input = convert(input_arg);
    const bool needs_materialization =
        v_input.storage_type() == api::StorageType::BUFFER &&
        !v_input.has_direct_buffer_layout();
    if (needs_materialization) {
      reshape_input =
          utils::contiguous_inference(input_arg, c10::MemoryFormat::Contiguous);
    }
  }

  if (reshape_input.dim() == 1) {
    return reshape_input.unsqueeze(0);
  }
  const IntArrayRef input_sizes = reshape_input.sizes();
  const auto d =
      c10::multiply_integers(input_sizes.cbegin(), input_sizes.end() - 1);
  return utils::reshape_inference(
      reshape_input, {d, reshape_input.size(-1)});
}

bool can_run_bfloat16_buffer_linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  if (
      input.device().type() != c10::DeviceType::Vulkan ||
      weight.device().type() != c10::DeviceType::Vulkan ||
      input.scalar_type() != kBFloat16 ||
      weight.scalar_type() != kBFloat16 ||
      input.dim() != 2 ||
      weight.dim() != 2 ||
      input.requires_grad() ||
      weight.requires_grad() ||
      input.size(Layout::Parameter::width) !=
          weight.size(Layout::Parameter::width)) {
    return false;
  }

  vTensor v_input = convert(input);
  vTensor v_weight = convert(weight);
  if (
      v_input.storage_type() != api::StorageType::BUFFER ||
      v_weight.storage_type() != api::StorageType::BUFFER ||
      v_input.gpu_memory_layout() != api::GPUMemoryLayout::TENSOR_WIDTH_PACKED ||
      v_weight.gpu_memory_layout() !=
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED ||
      !utils::supports_buffer_view_fast_path(v_input) ||
      !utils::supports_buffer_view_fast_path(v_weight)) {
    return false;
  }

  if (bias && bias->defined()) {
    if (
        bias->device().type() != c10::DeviceType::Vulkan ||
        bias->dim() > 2 ||
        bias->requires_grad()) {
      return false;
    }

    if (convert(*bias).storage_type() != api::StorageType::BUFFER) {
      return false;
    }

    if (bias->scalar_type() != kBFloat16 && bias->scalar_type() != kFloat) {
      return false;
    }

    if (!utils::supports_buffer_view_fast_path(convert(*bias))) {
      return false;
    }
  }

  return true;
}

Tensor run_bfloat16_buffer_linear(
    const Tensor& input_arg,
    const Tensor& weight_arg,
    const std::optional<Tensor>& bias_arg) {
  api::AllocationScope allocation_scope("linear.bf16_buffer");
  api::Context* const context = api::context();

  const Tensor input_arg_2d =
      input_arg.dim() == 2 ? input_arg : reshape_to_2d(input_arg);
  const Tensor input =
      input_arg_2d.is_vulkan() ? input_arg_2d : input_arg_2d.vulkan();
  const Tensor weight = weight_arg.is_vulkan() ? weight_arg : weight_arg.vulkan();

  TORCH_INTERNAL_ASSERT(can_run_bfloat16_buffer_linear(input, weight, bias_arg));

  vTensor v_input = convert(input);
  vTensor v_weight = convert(weight);

  vTensor v_output{
      context,
      {
          input_arg_2d.sizes()[Layout::Parameter::height],
          weight.sizes()[Layout::Parameter::height],
      },
      api::kFloat,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct {
    int32_t out_width;
    int32_t out_height;
    int32_t inner_dim;
    int32_t reserved;
  } block{
      api::utils::safe_downcast<int32_t>(weight.size(Layout::Parameter::height)),
      api::utils::safe_downcast<int32_t>(
          input_arg_2d.size(Layout::Parameter::height)),
      api::utils::safe_downcast<int32_t>(
          input_arg_2d.size(Layout::Parameter::width)),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(weight.size(Layout::Parameter::height)),
      api::utils::safe_downcast<uint32_t>(
          input_arg_2d.size(Layout::Parameter::height)),
      1u,
  };

  context->submit_compute_job(
      VK_KERNEL(mm_buffer_bfloat16),
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
      v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.buffer_metadata(),
      params.buffer());

  Tensor output = convert(v_output);

  std::optional<Tensor> bias = utils::prepare_optional_vulkan_execution_tensor(
      bias_arg,
      utils::VulkanExecutionPlanKind::LinearBiasSource,
      utils::make_vulkan_linear_request(utils::VulkanTensorRole::Bias));
  if (bias && bias->defined()) {
    if (!bias->is_vulkan()) {
      *bias = bias->vulkan();
    }
    output = output.add(*bias);
  }

  if (input_arg.dim() == 2) {
    return output;
  }

  std::vector<int64_t> shape;
  shape.reserve(static_cast<size_t>(std::max<int64_t>(0, input_arg.dim())));
  for (const auto i : c10::irange(input_arg.dim() - 1)) {
    shape.emplace_back(input_arg.size(i));
  }
  shape.emplace_back(output.size(-1));
  return utils::reshape_inference(output, shape);
}

Tensor run_quantized_addmm_context(
    const Tensor& input_arg,
    const float alpha,
    const float beta,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context,
    double output_scale,
    int64_t output_zero_point) {
  api::Context* const context = api::context();

  const Tensor input_arg_2d =
      input_arg.dim() == 2 ? input_arg : reshape_to_2d(input_arg);
  const Tensor input =
      input_arg_2d.is_vulkan() ? input_arg_2d : input_arg_2d.vulkan();
  const vTensor& v_input = convert(input);
  const LinearPackedRunState packed_state =
      get_linear_packed_run_state(linear_context);
  const vTensor& packed_v_weight = packed_state.packed_v_weight;
  const vTensor& packed_v_bias = packed_state.packed_v_bias;
  const std::vector<int64_t>& unpacked_weight_sizes =
      packed_state.logical_weight_sizes;
  const bool bias_defined = packed_state.bias_defined;

  TORCH_CHECK(
      usable(input, unpacked_weight_sizes),
      "Vulkan Linear not usable! "
      "Reason: The provided input tensor is either invalid on its own, or its "
      "combination with the provided weight and bias tensors are unsupported by "
      "Vulkan impl.");

  TORCH_CHECK(
      (packed_v_weight.is_quantized() && v_input.is_quantized()),
      "run_quantized_addmm_context called for quantized version with unquantized input");

  vTensor v_output{
      context,
      {
          input_arg_2d.sizes()[Layout::Parameter::height],
          unpacked_weight_sizes[Layout::Parameter::width],
      },
      v_input.dtype(),
  };

  v_output.set_is_quantized();
  v_output.set_scale(output_scale);
  v_output.set_zero_point(output_zero_point);

  if (bias_defined) {
    api::UniformParamsBuffer params;
    api::ShaderInfo compute_shader;
    compute_shader = (kQInt8 == input_arg.scalar_type())
        ? VK_KERNEL(quantized_addmm_qint8)
        : VK_KERNEL(quantized_addmm_quint8);
    const struct {
      uvec3 size;
      int32_t K;
      uvec3 um1_size;
      int32_t K1;
      uvec3 um2_size;
      int32_t K2;
      uvec3 ut_size;
      int32_t K3;
      vec2 multiplier;
      vec2 input_scales;
      float out_scale;
      float _1;
      ivec2 input_zero_points;
      int32_t out_zero_point;
      int32_t _2;
    } block{
        v_output.extents(),
        safe_downcast<int32_t>(
            div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(2))),
        v_input.extents(),
        0u,
        packed_v_weight.extents(),
        0u,
        packed_v_bias.extents(),
        0u,
        {
            alpha,
            beta,
        },
        {
            safe_downcast<float>(v_input.get_scale()),
            safe_downcast<float>(packed_v_weight.get_scale()),
        },
        safe_downcast<float>(output_scale),
        0.0f,
        {
            safe_downcast<int32_t>(v_input.get_zero_point()),
            safe_downcast<int32_t>(packed_v_weight.get_zero_point()),
        },
        safe_downcast<int32_t>(output_zero_point),
        0u,
    };
    params = api::UniformParamsBuffer(context, block);

    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        // shader descriptor
        compute_shader,
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {
            safe_downcast<uint32_t>(
                div_up(v_output.sizes()[Layout::Parameter::width], INT64_C(2))),
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::height], INT64_C(2))),
            1,
        },
        // local work group size
        {8, 8, 1},
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());

  } else { // no bias
    api::UniformParamsBuffer params;
    api::ShaderInfo compute_shader;
    const struct {
      uvec3 size;
      int32_t K;
      uvec3 um1_size;
      int32_t K1;
      uvec3 um2_size;
      int32_t K2;
      vec2 input_scales;
      float out_scale;
      float _1;
      ivec2 input_zero_points;
      int32_t out_zero_point;
      int32_t _2;
    } block_no_bias{
        v_output.extents(),
        safe_downcast<int32_t>(
            div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(2))),
        v_input.extents(),
        0u,
        packed_v_weight.extents(),
        0u,
        {
            safe_downcast<float>(v_input.get_scale()),
            safe_downcast<float>(packed_v_weight.get_scale()),
        },
        safe_downcast<float>(output_scale),
        0.0f,
        {
            safe_downcast<int32_t>(v_input.get_zero_point()),
            safe_downcast<int32_t>(packed_v_weight.get_zero_point()),
        },
        safe_downcast<int32_t>(output_zero_point),
        0u,
    };
    params = api::UniformParamsBuffer(context, block_no_bias);
    compute_shader = (kQInt8 == input_arg.scalar_type())
        ? VK_KERNEL(quantized_mm_qint8)
        : VK_KERNEL(quantized_mm_quint8);

    api::PipelineBarrier pipeline_barrier{};

    context->submit_compute_job(
        // shader descriptor
        compute_shader,
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {
            safe_downcast<uint32_t>(
                div_up(v_output.sizes()[Layout::Parameter::width], INT64_C(2))),
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::height], INT64_C(2))),
            1,
        },
        // local work group size
        {8, 8, 1},
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());
  }
  Tensor output = convert(v_output);
  if (input_arg.dim() == 2) {
    return output;
  } else {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(std::max<int64_t>(0, input_arg.dim())));
    for (const auto i : c10::irange(input_arg.dim() - 1)) {
      shape.emplace_back(input_arg.size(i));
    }
    shape.emplace_back(output.size(-1));
    Tensor reshaped_output = utils::reshape_inference(output, shape);
    if (c10::InferenceMode::is_enabled()) {
      reshaped_output = reshaped_output.clone();
    }
    return reshaped_output;
  }
}

Tensor run_addmm_context(
    const Tensor& input_arg,
    const float alpha,
    const float beta,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context,
    bool quantized,
    double output_scale,
    int64_t output_zero_point,
  const LinearPostOp post_op = LinearPostOp::None,
  Tensor* output_opt = nullptr) {
  const auto input_request =
      utils::make_vulkan_tensor_linear_request(
          input_arg, utils::VulkanTensorRole::Input);
  api::AllocationScope allocation_scope(
      utils::resolve_vulkan_linear_runtime_label(
          linear_context ? linear_context->allocation_label() : std::string(),
          "linear"));
  if (quantized) {
    return run_quantized_addmm_context(
        input_arg,
        alpha,
        beta,
        linear_context,
        output_scale,
        output_zero_point);
  }

  api::Context* const context = api::context();
  utils::prime_labeled_scratch_arena_for_request(
      input_arg,
      input_request,
      linear_runtime_scratch_bytes(input_arg),
      "linear_decode");
  const auto runtime_policy = utils::build_vulkan_runtime_policy(input_request);
  if (
      runtime_policy.request.model_domain != utils::VulkanModelDomain::Generic ||
      runtime_policy.request.execution_phase !=
          utils::VulkanExecutionPhase::None) {
    log_linear_kernel_family_choice(runtime_policy);
  }

  const Tensor source_input_arg =
      input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  const Tensor compute_input_arg = utils::prepare_vulkan_execution_tensor(
      source_input_arg,
      utils::VulkanExecutionPlanKind::LinearInputSource,
      input_request);
  const Tensor input_arg_2d =
      compute_input_arg.dim() == 2 ? compute_input_arg
                                   : reshape_to_2d(compute_input_arg);
  const Tensor input =
      input_arg_2d.is_vulkan() ? input_arg_2d : input_arg_2d.vulkan();
  const LinearPackedRunState packed_state =
      get_linear_packed_run_state(linear_context);
  const vTensor& packed_v_weight = packed_state.packed_v_weight;
  const vTensor& packed_v_bias = packed_state.packed_v_bias;
  const std::vector<int64_t>& unpacked_weight_sizes =
      packed_state.logical_weight_sizes;
  const bool bias_defined = packed_state.bias_defined;
  const vTensor& source_v_input = convert(input);

  TORCH_CHECK(
      usable(input, unpacked_weight_sizes),
      "Vulkan Linear not usable! "
      "Reason: The provided input tensor is either invalid on its own, or its "
      "combination with the provided weight and bias tensors are unsupported by "
      "Vulkan impl.");

  if (
      packed_v_weight.storage_type() == api::StorageType::BUFFER &&
      packed_state.packed_weight.execution_layout() ==
          api::ExecutionLayout::BUFFER_DIRECT) {
    Tensor buffer_input = utils::mark_tensor_execution(
        utils::ensure_buffer_storage(
            input, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
        api::ExecutionLayout::BUFFER_DIRECT);
    const std::optional<Tensor> packed_bias_tensor = bias_defined
        ? std::optional<Tensor>(packed_state.packed_weight.bias())
        : std::nullopt;
    if (can_run_float_buffer_linear(
            buffer_input,
            packed_state.packed_weight.weight(),
            packed_bias_tensor)) {
      utils::log_vulkan_op_hit("aten::linear.buffer_float");
      return run_float_buffer_linear(
          input_arg,
          buffer_input,
          packed_state,
          alpha,
          beta,
          post_op,
          output_opt);
    }
  }

  if (
      linear_kernel_family_allows_channel_packed_input(runtime_policy) &&
      can_use_channel_packed_linear_input(source_v_input, packed_v_weight)) {
    utils::log_vulkan_op_hit("aten::linear.channel_packed_family");
    return run_addmm_context_channel_packed_input(
        input_arg,
        input_arg_2d,
        source_v_input,
        packed_state,
        alpha,
        beta,
        post_op,
        output_opt);
  }

  const vTensor& v_input = pack_inputs_using_width_packing(input, input_request);

  TORCH_CHECK(
      v_input.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      "run_addmm_context must have width packed input");

  TORCH_CHECK(
      packed_v_weight.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
      "run_addmm_context must have height packed weight");

  const std::vector<int64_t> output_sizes{
      input_arg_2d.sizes()[Layout::Parameter::height],
      unpacked_weight_sizes[Layout::Parameter::width],
  };
  Tensor output_tensor = output_opt
      ? ensure_linear_output_tensor(
            *output_opt, output_sizes, convert_dtype(v_input.dtype()))
      : convert(vTensor{context, output_sizes, v_input.dtype()});
  vTensor& v_output = convert(output_tensor);

  api::UniformParamsBuffer params;
  api::ShaderInfo compute_shader;
  // Step size is the 2d input's w dimension / 4.
  int step_size = div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(4));
  const bool fuse_bias =
      bias_defined &&
      can_fuse_linear_bias(v_output, packed_v_bias, unpacked_weight_sizes);
  const bool fuse_gelu = fuse_bias && post_op == LinearPostOp::Gelu;

  if (fuse_gelu) {
    const struct {
      uvec4 shader_extents_and_step;
      uvec4 bias_extents;
      vec4 multipliers_and_gelu;
    } block_with_bias_gelu{
        {
            v_output.extents().data[0u],
            v_output.extents().data[1u],
            v_output.extents().data[2u],
            safe_downcast<uint32_t>(step_size),
        },
        {
            packed_v_bias.extents().data[0u],
            packed_v_bias.extents().data[1u],
            packed_v_bias.extents().data[2u],
            0u,
        },
        {alpha, beta, kGeluBeta, 0.0f},
    };
    params = api::UniformParamsBuffer(context, block_with_bias_gelu);
    compute_shader = VK_KERNEL(mm_bias_gelu);
  } else if (fuse_bias) {
    const struct {
      uvec4 shader_extents_and_step;
      uvec4 bias_extents;
      vec2 multipliers;
    } block_with_bias{
        {
            v_output.extents().data[0u],
            v_output.extents().data[1u],
            v_output.extents().data[2u],
            safe_downcast<uint32_t>(step_size),
        },
        {
            packed_v_bias.extents().data[0u],
            packed_v_bias.extents().data[1u],
            packed_v_bias.extents().data[2u],
            0u,
        },
        {alpha, beta},
    };
    params = api::UniformParamsBuffer(context, block_with_bias);
    compute_shader = VK_KERNEL(mm_bias);
  } else {
    const struct {
      uvec3 shader_extents;
      uint32_t mm_step_size;
    } block_no_bias{
        v_output.extents(),
        safe_downcast<uint32_t>(step_size),
    };
    params = api::UniformParamsBuffer(context, block_no_bias);
    compute_shader = VK_KERNEL(mm);
  }

  api::PipelineBarrier pipeline_barrier{};

  if (fuse_bias) {
    context->submit_compute_job(
        // shader descriptor
        compute_shader,
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::width],
                INT64_C(4))),
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::height],
                INT64_C(4))),
            1,
        },
        // local work group size
        {8, 8, 1},
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());
  } else {
    context->submit_compute_job(
        // shader descriptor
        compute_shader,
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::width],
                INT64_C(4))),
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::height],
                INT64_C(4))),
            1,
        },
        // local work group size
        {8, 8, 1},
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());
  }

  Tensor output = output_tensor;

  // addmm/linear operation, multiplying by alpha and adding bias when present.
  if (!fuse_bias && alpha != 1.0f) {
    output = output.mul(alpha);
  }
  if (!fuse_bias && bias_defined) {
    output = output.add(convert(packed_v_bias).mul(beta));
  }
  if (post_op == LinearPostOp::Gelu && !fuse_gelu) {
    output = at::gelu(output, "none");
  }
  if (output_opt && output.unsafeGetTensorImpl() != output_tensor.unsafeGetTensorImpl()) {
    output = rebind_vulkan_output(*output_opt, output);
  }
  return reshape_linear_output_if_needed(output, input_arg);
}

Tensor run_baddbmm_context(
    const Tensor& input_arg,
    const float alpha,
    const float beta,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  const auto input_request =
      utils::make_vulkan_tensor_linear_request(
          input_arg, utils::VulkanTensorRole::Input);
  api::AllocationScope allocation_scope("bmm");
  // TODO: Refactor run_baddbmm_context and run_addmm_context into one.
  api::Context* const context = api::context();

  TORCH_CHECK(
      input_arg.dim() == 3,
      "Vulkan Linear not usable! "
      "Reason: The input has the wrong dimension; the tensor of a batch of matrices should contain 3 dimensions: batch, height, width.");

  utils::prime_labeled_scratch_arena_for_request(
      input_arg,
      input_request,
      linear_runtime_scratch_bytes(input_arg),
      "bmm_decode");
  const Tensor compute_input_arg = utils::prepare_vulkan_execution_tensor(
      input_arg,
      utils::VulkanExecutionPlanKind::LinearInputSource,
      input_request);
  Tensor input =
      compute_input_arg.is_vulkan() ? compute_input_arg
                                    : compute_input_arg.vulkan();
  if (input.scalar_type() == kHalf) {
    // The current batched matmul path backing Vulkan SDPA is much more stable
    // when half inputs are widened before packing. Keep the model path running
    // on Vulkan until a true native half batch-matmul family exists.
    input = input.to(kFloat);
  }
  vTensor packed_v_input = pack_inputs_using_width_packing(input, input_request);

  const LinearPackedRunState packed_state =
      get_linear_packed_run_state(linear_context);
  const vTensor& packed_v_weight = packed_state.packed_v_weight;
  const vTensor& packed_v_bias = packed_state.packed_v_bias;
  const std::vector<int64_t>& unpacked_weight_sizes =
      packed_state.logical_weight_sizes;

  TORCH_CHECK(
      usable(input, unpacked_weight_sizes, true /*use batch*/),
      "Vulkan Linear not usable! "
      "Reason: The provided input tensor is either invalid on its own, or its "
      "combination with the provided weight and bias tensors are unsupported by "
      "Vulkan impl.");

  TORCH_CHECK(
      packed_v_input.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      "run_addmm_context called for non-quantized version with unpacked weight");

  TORCH_CHECK(
      packed_v_weight.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
      "run_addmm_context called for non-quantized version with unpacked weight");

  int64_t input_batch = packed_v_input.sizes()[Layout::BatchMatrices::batch];

  // Step size is the input's w dimension / 4.
  int64_t input_width = packed_v_input.sizes()[Layout::BatchMatrices::width];
  int64_t mm_step_size = div_up(input_width, INT64_C(4));

  vTensor v_output{
      context,
      {
          input_batch,
          packed_v_input.sizes()[Layout::BatchMatrices::height],
          unpacked_weight_sizes.back(), // "w" dimension in weight matrix
      },
      packed_v_input.dtype(),
  };

  const struct {
    uvec4 shader_extents_and_step;
    uvec4 batch_info;
  } block_no_bias{
      {
          v_output.extents().data[0u],
          v_output.extents().data[1u],
          v_output.extents().data[2u],
          safe_downcast<uint32_t>(mm_step_size),
      },
      {
          safe_downcast<uint32_t>(input_batch),
          0u,
          0u,
          0u,
      },
  };

  api::UniformParamsBuffer params(context, block_no_bias);

  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(bmm_channel_packed),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      {
          safe_downcast<uint32_t>(div_up(
              v_output.sizes()[Layout::BatchMatrices::width], INT64_C(4))),
          safe_downcast<uint32_t>(div_up(
              v_output.sizes()[Layout::BatchMatrices::height], INT64_C(4))),
          v_output.extents().data[2u],
      },
      // local work group size
      {8, 8, 1},
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      packed_v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  // The dedicated batched kernel writes up to four batch results directly into
  // each channel-packed output texel, so no post-slice is needed here.
  return convert(v_output).mul(alpha).add(convert(packed_v_bias).mul(beta));
}

Tensor addmm(
    const Tensor& bias,
    const Tensor& input,
    const Tensor& weight,
    const Scalar& beta,
    const Scalar& alpha) {
  return run_addmm_context(
      input,
      alpha.to<float>(),
      beta.to<float>(),
      c10::make_intrusive<LinearPackedContext>(
          LinearPackedContext(weight, bias)),
      false,
      0,
      0);
}

Tensor run_half_buffer_linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  const Tensor float_input = upcast_half_linear_tensor_for_packing(input);
  const Tensor float_weight = upcast_half_linear_tensor_for_packing(weight);
  const std::optional<Tensor> float_bias =
      upcast_half_linear_tensor_for_packing(bias);

  Tensor output = run_addmm_context(
      float_input,
      1.0f,
      1.0f,
      get_or_create_linear_context(float_weight, float_bias),
      false,
      0,
      0);
  return output.to(kHalf);
}

Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  utils::log_vulkan_op_hit("aten::linear");
  const Tensor effective_weight =
      weight.requires_grad() ? weight.detach() : weight;
  const std::optional<Tensor> effective_bias =
      (bias && bias->defined() && bias->requires_grad())
      ? std::optional<Tensor>(bias->detach())
      : bias;
  const Tensor linear_input = input.dim() == 2 ? input : reshape_to_2d(input);
  const Tensor linear_weight = effective_weight.is_vulkan()
      ? effective_weight
      : effective_weight.vulkan();
  const std::optional<Tensor> linear_bias =
      (effective_bias && effective_bias->defined() && !effective_bias->is_vulkan())
      ? effective_bias->vulkan()
      : effective_bias;

  if (can_run_bfloat16_buffer_linear(
          linear_input, linear_weight, linear_bias)) {
    return run_bfloat16_buffer_linear(input, linear_weight, linear_bias);
  }

  if (can_run_half_buffer_linear(input, effective_weight, effective_bias)) {
    return run_half_buffer_linear(input, effective_weight, effective_bias);
  }

  return run_addmm_context(
      input,
      1.0f,
      1.0f,
      get_or_create_linear_context(effective_weight, effective_bias),
      false,
      0,
      0);
}

Tensor linear_gelu(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  return run_addmm_context(
      input,
      1.0f,
      1.0f,
      get_or_create_linear_context(weight, bias),
      false,
      0,
      0,
      LinearPostOp::Gelu);
}

Tensor mm(const Tensor& mat1_arg, const Tensor& mat2_arg) {
  utils::log_vulkan_op_hit("aten::mm");
  return run_addmm_context(
      mat1_arg,
      1.0f,
      1.0f,
      c10::make_intrusive<LinearPackedContext>(
          LinearPackedContext(mat2_arg, std::optional<Tensor>())),
      false,
      0,
      0);
}

Tensor bmm(const Tensor& mat1_arg, const Tensor& mat2_arg) {
  utils::log_vulkan_op_hit("aten::bmm");
  const Tensor mat1 = mat1_arg.is_vulkan() ? mat1_arg : mat1_arg.vulkan();
  const Tensor mat2 = mat2_arg.is_vulkan() ? mat2_arg : mat2_arg.vulkan();
  if (can_run_half_buffer_bmm(mat1, mat2)) {
    utils::log_vulkan_op_hit("aten::bmm.buffer_float");
    return run_half_buffer_bmm(mat1, mat2, 1.0f, 1.0f);
  }
  if (can_run_float_buffer_bmm(mat1, mat2)) {
    utils::log_vulkan_op_hit("aten::bmm.buffer_float");
    return run_float_buffer_bmm(mat1, mat2, 1.0f, 1.0f);
  }
  return run_baddbmm_context(
      mat1_arg,
      1.0f,
      1.0f,
      c10::make_intrusive<LinearPackedContext>(LinearPackedContext(
          mat2_arg, std::optional<Tensor>(), true /*use batch*/)));
}

Tensor baddbmm(
    const Tensor& bias,
    const Tensor& input,
    const Tensor& weight,
    const Scalar& beta,
    const Scalar& alpha) {
  const Tensor mat1 = input.is_vulkan() ? input : input.vulkan();
  const Tensor mat2 = weight.is_vulkan() ? weight : weight.vulkan();
  if (can_run_half_buffer_bmm(mat1, mat2)) {
    utils::log_vulkan_op_hit("aten::bmm.buffer_float");
    return run_half_buffer_bmm(
        mat1, mat2, alpha.to<float>(), beta.to<float>(), bias);
  }
  if (can_run_float_buffer_bmm(mat1, mat2)) {
    utils::log_vulkan_op_hit("aten::bmm.buffer_float");
    return run_float_buffer_bmm(
        mat1, mat2, alpha.to<float>(), beta.to<float>(), bias);
  }
  return run_baddbmm_context(
      input,
      alpha.to<float>(),
      beta.to<float>(),
      c10::make_intrusive<LinearPackedContext>(
          LinearPackedContext(weight, bias, true /*use batch*/)));
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::addmm"), TORCH_FN(addmm));
  m.impl(TORCH_SELECTIVE_NAME("aten::linear"), TORCH_FN(linear));
  m.impl(TORCH_SELECTIVE_NAME("aten::mm"), TORCH_FN(mm));
  m.impl(TORCH_SELECTIVE_NAME("aten::bmm"), TORCH_FN(bmm));
  m.impl(TORCH_SELECTIVE_NAME("aten::baddbmm"), TORCH_FN(baddbmm));
}

#endif /* USE_VULKAN_API */

} // namespace

LinearPackedContext::LinearPackedContext(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const bool use_batch,
    std::string allocation_label,
    const bool retain_unpacked)
    : unpacked_{c10::AnyType::get()} {
  allocation_label_ = std::move(allocation_label);
  api::AllocationScope allocation_scope(
      utils::make_vulkan_linear_pack_label(
          allocation_label_, use_batch ? "bmm.pack" : "linear.pack"));
  const auto normalized_bias = utils::normalized_optional_tensor(bias);
  const std::vector<int64_t> logical_weight_sizes = weight.sizes().vec();
  constexpr uint64_t kLinearBatchPackOption = 1u;
  constexpr uint64_t kLinearBufferPackOption = 2u;
  const bool use_buffer_packed_weights =
      !use_batch &&
      !weight.is_quantized() &&
      is_float_or_half_tensor(weight) &&
      (!bias || !bias->defined() || is_float_or_half_tensor(*bias));
  const uint64_t pack_options = (use_batch ? kLinearBatchPackOption : 0u) |
      (use_buffer_packed_weights ? kLinearBufferPackOption : 0u);
  if (const auto cached_packed_weight = utils::lookup_packed_weight_handle(
          weight,
          normalized_bias,
          logical_weight_sizes,
          PackedWeightKind::Linear,
          weight.is_quantized(),
          pack_options)) {
    packed_weight_ = *cached_packed_weight;
  } else {
    const Tensor pack_source_weight =
        upcast_half_linear_tensor_for_packing(weight);
    const std::optional<Tensor> pack_source_bias =
        upcast_half_linear_tensor_for_packing(bias);
    const Tensor compute_weight =
        upcast_half_linear_tensor_for_packing(pack_source_weight);
    const std::optional<Tensor> compute_bias =
        upcast_half_linear_tensor_for_packing(pack_source_bias);
    TORCH_CHECK(
        available(compute_weight, compute_bias, use_batch),
        "Vulkan Linear not available! "
        "Reason: The provided (weight, bias) parameters are either invalid "
        "individually or their combination is not supported by Vulkan Impl.");

    if (use_buffer_packed_weights) {
      Tensor buffer_weight = upload_linear_tensor_to_buffer(
          compute_weight, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);

      Tensor buffer_bias_tensor;
      if (compute_bias && compute_bias->defined()) {
        buffer_bias_tensor = upload_linear_tensor_to_buffer(
            *compute_bias, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
      } else {
        buffer_bias_tensor = upload_linear_tensor_to_buffer(
            at::zeros({1}, at::device(at::kCPU).dtype(at::kFloat)),
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
      }

      const size_t resident_nbytes =
          convert(buffer_weight).gpu_nbytes() +
          (buffer_bias_tensor.defined() ? convert(buffer_bias_tensor).gpu_nbytes()
                                        : 0u);
      packed_weight_ = PackedWeightHandle(
          std::move(buffer_weight),
          std::move(buffer_bias_tensor),
          logical_weight_sizes,
          PackedWeightKind::Linear,
          compute_bias && compute_bias->defined(),
          PackedWeightResidencyClass::PersistentInference,
          false,
          api::ExecutionLayout::BUFFER_DIRECT,
          resident_nbytes);
    } else {
      const Tensor packed_weight = utils::prepare_vulkan_execution_tensor(
          pack_source_weight,
          utils::VulkanExecutionPlanKind::LinearWeightSource,
          utils::make_vulkan_linear_request(utils::VulkanTensorRole::Weight));
      const std::optional<Tensor> packed_bias =
          utils::prepare_optional_vulkan_execution_tensor(
              pack_source_bias,
              utils::VulkanExecutionPlanKind::LinearBiasSource,
              utils::make_vulkan_linear_request(utils::VulkanTensorRole::Bias));
      const Tensor texture_compute_weight =
          upcast_half_linear_tensor_for_packing(packed_weight);
      const std::optional<Tensor> texture_compute_bias =
          upcast_half_linear_tensor_for_packing(packed_bias);

      Tensor packed_bias_tensor = packed_weight.is_quantized()
          ? convert(pack_biases_quantized_weights(
                texture_compute_weight, texture_compute_bias, use_batch))
          : convert(
                pack_biases(texture_compute_weight, texture_compute_bias, use_batch));

      packed_weight_ = utils::make_packed_weight_handle(
          convert(pack_weights(texture_compute_weight, use_batch)),
          std::move(packed_bias_tensor),
          packed_weight.sizes().vec(),
          PackedWeightKind::Linear,
          texture_compute_bias && texture_compute_bias->defined(),
          packed_weight.is_quantized());
    }
    utils::store_packed_weight_handle(
        weight,
        normalized_bias,
        logical_weight_sizes,
        PackedWeightKind::Linear,
        packed_weight_,
        weight.is_quantized(),
        pack_options);
  }

  if (retain_unpacked && !at::globalContext().releaseWeightsWhenPrepacking()) {
    unpacked_.reserve(Unpacked::NumArgs);
    unpacked_.emplace_back(weight);
    unpacked_.emplace_back(normalized_bias);
  }
}

LinearPackedContext LinearPackedContext::pack(c10::impl::GenericList unpacked) {
  return LinearPackedContext(
      unpacked.get(Unpacked::Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Bias));
}

c10::intrusive_ptr<LinearPackedContext> create_linear_context(
    Tensor&& weight,
    std::optional<Tensor>&& bias) {
  return c10::make_intrusive<LinearPackedContext>(
      LinearPackedContext(weight, bias));
}

c10::intrusive_ptr<LinearPackedContext> create_linear_context_labeled(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    std::string label) {
  if (const auto cached_context =
          utils::lookup_labeled_linear_context(weight, bias, label)) {
    return *cached_context;
  }

  const Tensor prepared_weight =
      (c10::InferenceMode::is_enabled() && weight.is_vulkan() &&
       weight.dim() == 2)
      ? weight.cpu().t().contiguous()
      : weight.t();
  const auto context = c10::make_intrusive<LinearPackedContext>(
      LinearPackedContext(
          prepared_weight,
          bias,
          false,
          std::move(label),
          true));
  utils::store_labeled_linear_context(
      weight, bias, context->allocation_label(), context);
  return context;
}

Tensor run_linear_context(
    const Tensor& input,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  utils::log_vulkan_op_hit("vulkan_prepack::run_linear_context");
  return run_addmm_context(input, 1.0f, 1.0f, linear_context, false, 0, 0);
}

Tensor run_linear_context_out(
    const Tensor& input,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context,
    Tensor& output) {
  utils::log_vulkan_op_hit("vulkan_prepack::run_linear_context");
  return run_addmm_context(
      input,
      1.0f,
      1.0f,
      linear_context,
      false,
      0,
      0,
      LinearPostOp::None,
      &output);
}

Tensor run_linear_gelu_context(
    const Tensor& input,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  return run_addmm_context(
      input,
      1.0f,
      1.0f,
      linear_context,
      false,
      0,
      0,
      LinearPostOp::Gelu);
}

Tensor run_linear_gelu_context_out(
    const Tensor& input,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context,
    Tensor& output) {
  return run_addmm_context(
      input,
      1.0f,
      1.0f,
      linear_context,
      false,
      0,
      0,
      LinearPostOp::Gelu,
      &output);
}

Tensor run_qlinear_context(
    const Tensor& input_arg,
    double output_scale,
    int64_t output_zero_point,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  return run_addmm_context(
      input_arg,
      1.0f,
      1.0f,
      linear_context,
      true,
      output_scale,
      output_zero_point);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
