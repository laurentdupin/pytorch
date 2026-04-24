#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#endif // _WIN32

#include <ATen/native/vulkan/ops/Clamp.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

bool can_run_float_buffer_clamp(const vTensor& v_self) {
  return v_self.storage_type() == api::StorageType::BUFFER &&
      v_self.dtype() == api::kFloat &&
      !v_self.is_quantized() &&
      utils::supports_buffer_elementwise_compute(v_self);
}

api::UniformParamsBuffer make_buffer_clamp_params(
    api::Context* const context,
    const std::optional<Scalar>& min,
    const std::optional<Scalar>& max) {
  const struct Block final {
    vec2 clamp;
  } block{
      {
          min ? min->to<float>() : -std::numeric_limits<float>::infinity(),
          max ? max->to<float>() : std::numeric_limits<float>::infinity(),
      },
  };
  return api::UniformParamsBuffer(context, block);
}

Tensor prepare_runtime_float_buffer_clamp_output(
    Tensor output,
    IntArrayRef expected_sizes) {
  output = output.is_vulkan() ? output : output.vulkan();
  output = utils::mark_tensor_execution(
      output,
      utils::resolve_buffer_execution_layout(convert(output)),
      false);
  const vTensor& v_output = convert(output);
  TORCH_CHECK(
      v_output.storage_type() == api::StorageType::BUFFER &&
          v_output.dtype() == api::kFloat &&
          utils::supports_buffer_elementwise_compute(v_output),
      "Vulkan float buffer clamp out expects float buffer-backed output");
  TORCH_CHECK(
      output.sizes().vec() == expected_sizes.vec(),
      "Vulkan float buffer clamp out received mismatched output shape");
  return output;
}

Tensor clamp_buffer_impl(
    const Tensor& self_arg,
    const std::optional<Scalar>& min,
    const std::optional<Scalar>& max,
    Tensor* output_arg) {
  api::Context* const context = api::context();
  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  vTensor& v_self = convert(self);
  TORCH_CHECK(
      can_run_float_buffer_clamp(v_self),
      "Vulkan buffer clamp expects float buffer-backed input with supported metadata");

  Tensor output_tensor;
  vTensor* v_output_ptr = nullptr;
  vTensor owned_output;
  if (output_arg != nullptr) {
    output_tensor =
        prepare_runtime_float_buffer_clamp_output(*output_arg, v_self.sizes());
    v_output_ptr = &convert(output_tensor);
  } else {
    owned_output = vTensor{
        context,
        v_self.sizes(),
        v_self.dtype(),
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
    };
    v_output_ptr = &owned_output;
  }
  vTensor& v_output = *v_output_ptr;

  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(v_output.numel()),
      1u,
      1u,
  };
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_self);
  api::UniformParamsBuffer params = make_buffer_clamp_params(context, min, max);

  context->submit_compute_job(
      VK_KERNEL(buffer_clamp),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer(),
      params.buffer());

  return output_arg != nullptr ? output_tensor : convert(v_output);
}

Tensor clamp_buffer(
    const Tensor& self_arg,
    const std::optional<Scalar>& min,
    const std::optional<Scalar>& max) {
  return clamp_buffer_impl(self_arg, min, max, nullptr);
}

Tensor& clamp_buffer_(
    Tensor& self_arg,
    const std::optional<Scalar>& min,
    const std::optional<Scalar>& max) {
  api::Context* const context = api::context();
  vTensor& v_self = convert(self_arg);
  TORCH_CHECK(
      can_run_float_buffer_clamp(v_self),
      "Vulkan in-place buffer clamp expects float buffer-backed input with supported metadata");

  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(v_self.numel()),
      1u,
      1u,
  };
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_self);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_self);
  api::UniformParamsBuffer params = make_buffer_clamp_params(context, min, max);

  context->submit_compute_job(
      VK_KERNEL(buffer_clamp),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_self.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_self.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      in_meta.buffer(),
      params.buffer());

  return self_arg;
}

Tensor _clamp(
    const Tensor& self_arg,
    const std::optional<Scalar>& min,
    const std::optional<Scalar>& max,
    const api::ShaderInfo& shader_descriptor) {
  TORCH_CHECK(min || max, "At least one of 'min' or 'max' must not be None");

  if (self_arg.is_vulkan() && self_arg.scalar_type() == at::kFloat) {
    const vTensor& v_self = convert(self_arg);
    if (can_run_float_buffer_clamp(v_self)) {
      return clamp_buffer(self_arg, min, max);
    }
  }

  api::Context* const context = api::context();

  Tensor self = utils::prepare_vulkan_execution_tensor(
      self_arg, utils::VulkanExecutionPlanKind::TextureComputeInput);
  const vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };
  if (v_self.is_quantized()) {
    v_output.set_is_quantized();
    v_output.set_scale(v_self.get_scale());
    v_output.set_zero_point(v_self.get_zero_point());
  }

  api::UniformParamsBuffer params;

  if (v_self.is_quantized()) {
    float mini = min
        ? roundevenf(min->to<float>() / float(v_self.get_scale())) +
            float(v_self.get_zero_point())
        : -std::numeric_limits<float>::infinity();
    float maxi = max
        ? roundevenf(max->to<float>() / float(v_self.get_scale())) +
            float(v_self.get_zero_point())
        : std::numeric_limits<float>::infinity();
    const struct Block final {
      uvec3 extents;
      uint32_t align;
      vec2 clamp;
    } block{
        v_output.extents(),
        0u,
        {mini, maxi},
    };
    params = api::UniformParamsBuffer(context, block);
  } else {
    const struct Block final {
      uvec3 extents;
      uint32_t align;
      vec2 clamp;
    } block{
        v_output.extents(),
        0u,
        {
            min ? min->to<float>() : -std::numeric_limits<float>::infinity(),
            max ? max->to<float>() : std::numeric_limits<float>::infinity(),
        },
    };
    params = api::UniformParamsBuffer(context, block);
  }

  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor clamp(
    const Tensor& self_arg,
    const std::optional<Scalar>& min,
    const std::optional<Scalar>& max) {
  return _clamp(self_arg, min, max, VK_KERNEL(clamp));
}

Tensor& _clamp_(
    Tensor& self_arg,
    const std::optional<Scalar>& min,
    const std::optional<Scalar>& max,
    const api::ShaderInfo& shader_descriptor) {
  TORCH_CHECK(min || max, "At least one of 'min' or 'max' must not be None");

  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place clamp is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  vTensor& v_self = convert(self);
  if (can_run_float_buffer_clamp(v_self)) {
    return clamp_buffer_(self_arg, min, max);
  }
  TORCH_CHECK(
      v_self.storage_type() != api::StorageType::BUFFER,
      "In-place Vulkan clamp is not yet supported on buffer-backed logical views");

  api::UniformParamsBuffer params;

  if (v_self.is_quantized()) {
    float mini = min
        ? roundevenf(min->to<float>() / float(v_self.get_scale())) +
            float(v_self.get_zero_point())
        : -std::numeric_limits<float>::infinity();
    float maxi = max
        ? roundevenf(max->to<float>() / float(v_self.get_scale())) +
            float(v_self.get_zero_point())
        : std::numeric_limits<float>::infinity();
    const struct Block final {
      uvec3 extents;
      uint32_t align;
      vec2 clamp;
    } block{
        v_self.extents(),
        0u,
        {mini, maxi},
    };
    params = api::UniformParamsBuffer(context, block);
  } else {
    const struct Block final {
      uvec3 extents;
      uint32_t align;
      vec2 clamp;
    } block{
        v_self.extents(),
        0u,
        {
            min ? min->to<float>() : -std::numeric_limits<float>::infinity(),
            max ? max->to<float>() : std::numeric_limits<float>::infinity(),
        },
    };
    params = api::UniformParamsBuffer(context, block);
  }
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      // params buffer
      params.buffer());

  return self_arg;
}

Tensor threshold(
    const Tensor& self,
    const Scalar& threshold,
    const Scalar& value) {
  return _clamp(self, threshold, value, VK_KERNEL(threshold));
}

Tensor& clamp_(
    Tensor& self,
    const std::optional<Scalar>& min,
    const std::optional<Scalar>& max) {
  return _clamp_(self, min, max, VK_KERNEL(clamp_));
}

Tensor activation(
    const Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::Context* const context = api::context();

  Tensor self = utils::prepare_vulkan_execution_tensor(
      self_arg, utils::VulkanExecutionPlanKind::TextureComputeInput);
  const vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  const struct Block final {
    uvec3 extents;
    uint32_t _;
  } block{
      v_output.extents(),
      0u,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor activation_buffer(
    const Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::Context* const context = api::context();

  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  vTensor& v_self = convert(self);

  TORCH_CHECK(
      v_self.storage_type() == api::StorageType::BUFFER,
      "Vulkan buffer activation expects buffer-backed input");
  TORCH_CHECK(
      utils::supports_buffer_elementwise_compute(v_self),
      "Vulkan buffer activation requires supported buffer elementwise compute input");

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(v_output.numel()),
      1u,
      1u,
  };
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_self);

  context->submit_compute_job(
      shader_descriptor,
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer());

  return convert(v_output);
}

Tensor& activation_(
    Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor) {
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);
  TORCH_CHECK(
      v_self.storage_type() != api::StorageType::BUFFER,
      "In-place Vulkan activation is not yet supported on buffer-backed logical views");

  const struct Block final {
    uvec3 extents;
    uint32_t _;
  } block{
      v_self.extents(),
      0u,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      // params buffer
      params.buffer());

  return self_arg;
}

Tensor hardtanh(const Tensor& self, const Scalar& min, const Scalar& max) {
  return ops::_clamp(self, min, max, VK_KERNEL(clamp));
}

Tensor& hardtanh_(Tensor& self, const Scalar& min, const Scalar& max) {
  return ops::_clamp_(self, min, max, VK_KERNEL(clamp_));
}

Tensor relu(const Tensor& self) {
  return (
      (self.scalar_type() == at::kQUInt8)
          ? ops::_clamp(
                self, 0, std::nullopt, VK_KERNEL(quantized_clamp_quint8))
          : ((self.scalar_type() == at::kQInt8)
                 ? ops::_clamp(
                       self, 0, std::nullopt, VK_KERNEL(quantized_clamp_qint8))
                 : ops::_clamp(self, 0, std::nullopt, VK_KERNEL(clamp))));
}

Tensor& relu_(Tensor& self) {
  return (
      (self.scalar_type() == at::kQUInt8)
          ? ops::_clamp_(
                self, 0, std::nullopt, VK_KERNEL(quantized_clamp_quint8_))
          : ((self.scalar_type() == at::kQInt8)
                 ? ops::_clamp_(
                       self, 0, std::nullopt, VK_KERNEL(quantized_clamp_qint8_))
                 : ops::_clamp_(self, 0, std::nullopt, VK_KERNEL(clamp_))));
}

Tensor hardswish(const Tensor& self) {
  return ops::activation(self, VK_KERNEL(hardswish));
}

Tensor& hardswish_(Tensor& self) {
  return ops::activation_(self, VK_KERNEL(hardswish_));
}

Tensor hardsigmoid(const Tensor& self) {
  return ops::activation(self, VK_KERNEL(hardsigmoid));
}

Tensor& hardsigmoid_(Tensor& self) {
  return ops::activation_(self, VK_KERNEL(hardsigmoid_));
}

Tensor activation_scalar(
    const Tensor& self_arg,
    const std::vector<Scalar>& scalar_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::Context* const context = api::context();

  Tensor self = utils::prepare_vulkan_execution_tensor(
      self_arg, utils::VulkanExecutionPlanKind::TextureComputeInput);
  const vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  api::UniformParamsBuffer params;

  if (v_self.is_quantized()) {
    v_output.set_is_quantized();
    v_output.set_scale(v_self.get_scale());
    v_output.set_zero_point(v_self.get_zero_point());
  }

  if (scalar_arg.size() == 1) {
    if (v_self.is_quantized()) {
      const struct Block final {
        uvec3 extents;
        uint32_t _;
        float scalar_value;
        float scale;
        int zero_point;
      } block{
          v_output.extents(),
          0u,
          scalar_arg[0].to<float>(),
          safe_downcast<float>(v_self.get_scale()),
          safe_downcast<int32_t>(v_self.get_zero_point()),
      };
      params = api::UniformParamsBuffer(context, block);
    } else {
      const struct Block final {
        uvec3 extents;
        uint32_t _;
        float scalar_value;
      } block{
          v_output.extents(),
          0u,
          scalar_arg[0].to<float>(),
      };
      params = api::UniformParamsBuffer(context, block);
    }
  } else {
    const struct Block final {
      uvec3 extents;
      uint32_t _;
      float scalar_value1;
      float scalar_value2;
    } block{
        v_output.extents(),
        0u,
        scalar_arg[0].to<float>(),
        scalar_arg[1].to<float>(),
    };
    params = api::UniformParamsBuffer(context, block);
  }

  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor& activation_scalar_(
    Tensor& self_arg,
    const std::vector<Scalar>& scalar_arg,
    const api::ShaderInfo& shader_descriptor) {
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);
  TORCH_CHECK(
      v_self.storage_type() != api::StorageType::BUFFER,
      "In-place Vulkan scalar activation is not yet supported on buffer-backed logical views");

  api::UniformParamsBuffer params;

  if (scalar_arg.size() == 1) {
    if (v_self.is_quantized()) {
      const struct Block final {
        uvec3 extents;
        uint32_t _;
        float scalar_value;
        float scale;
        int zero_point;
      } block{
          v_self.extents(),
          0u,
          scalar_arg[0].to<float>(),
          safe_downcast<float>(v_self.get_scale()),
          safe_downcast<int32_t>(v_self.get_zero_point()),
      };
      params = api::UniformParamsBuffer(context, block);
    } else {
      const struct Block final {
        uvec3 extents;
        uint32_t _;
        float scalar_value;
      } block{
          v_self.extents(),
          0u,
          scalar_arg[0].to<float>(),
      };
      params = api::UniformParamsBuffer(context, block);
    }
  } else {
    const struct Block final {
      uvec3 extents;
      uint32_t _;
      float scalar_value1;
      float scalar_value2;
    } block{
        v_self.extents(),
        0u,
        scalar_arg[0].to<float>(),
        scalar_arg[1].to<float>(),
    };
    params = api::UniformParamsBuffer(context, block);
  }

  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      // params buffer
      params.buffer());

  return self_arg;
}

Tensor gelu(const Tensor& self, std::string_view approximate) {
  api::AllocationScope allocation_scope("gelu");
  TORCH_CHECK(
      approximate == "none" || approximate == "tanh",
      "Vulkan: gelu only supported for none or tanh type");
  if (auto fused = try_consume_deferred_linear_gelu(self, approximate)) {
    return *fused;
  }
  const Tensor gelu_input =
      materialize_deferred_linear_gelu_candidate_if_needed(self);
  // The Vulkan backend only has the tanh GELU kernel today, so route the
  // default eager GELU call through the same implementation for inference.
  if (gelu_input.is_vulkan() && gelu_input.scalar_type() == at::kFloat) {
    const vTensor& v_self = convert(gelu_input);
    if (
        v_self.storage_type() == api::StorageType::BUFFER &&
        utils::supports_buffer_elementwise_compute(v_self)) {
      utils::log_vulkan_op_hit("aten::gelu.buffer_float");
      return ops::activation_buffer(gelu_input, VK_KERNEL(buffer_gelu_tanh));
    }

    const auto plan = utils::build_vulkan_execution_plan(
        gelu_input, utils::VulkanExecutionPlanKind::ElementwiseInput);
    if (api::uses_buffer_execution(plan.execution_layout)) {
      Tensor prepared =
          utils::prepare_vulkan_direct_buffer_execution_tensor(gelu_input, plan);
      utils::log_vulkan_op_hit("aten::gelu.buffer_float");
      return ops::activation_buffer(prepared, VK_KERNEL(buffer_gelu_tanh));
    }
  }

  Scalar kBetaVec = M_SQRT2 * M_2_SQRTPI * 0.5;
  std::vector<Scalar> scalar;
  scalar.push_back(kBetaVec);

  if (gelu_input.scalar_type() == at::kQUInt8) {
    return ops::activation_scalar(
        gelu_input, scalar, VK_KERNEL(quantized_gelu_tanh_quint8));
  }

  if (gelu_input.scalar_type() == at::kQInt8) {
    return ops::activation_scalar(
        gelu_input, scalar, VK_KERNEL(quantized_gelu_tanh_qint8));
  }

  return ops::activation_scalar(gelu_input, scalar, VK_KERNEL(gelu_tanh));
}

Tensor& gelu_(Tensor& self, std::string_view approximate) {
  api::AllocationScope allocation_scope("gelu");
  TORCH_CHECK(
      approximate == "none" || approximate == "tanh",
      "Vulkan: gelu only supported for none or tanh type");
  // The Vulkan backend only has the tanh GELU kernel today, so route the
  // default eager GELU call through the same implementation for inference.
  Scalar kBetaVec = M_SQRT2 * M_2_SQRTPI * 0.5;
  std::vector<Scalar> scalar;
  scalar.push_back(kBetaVec);

  if (self.scalar_type() == at::kQUInt8) {
    return ops::activation_scalar_(
        self, scalar, VK_KERNEL(quantized_gelu_tanh_quint8_));
  }

  if (self.scalar_type() == at::kQInt8) {
    return ops::activation_scalar_(
        self, scalar, VK_KERNEL(quantized_gelu_tanh_qint8_));
  }

  return ops::activation_scalar_(self, scalar, VK_KERNEL(gelu_tanh_));
}

Tensor hardshrink(const Tensor& self_arg, const Scalar& lambd) {
  float abs_lambd = std::abs(lambd.to<float>());
  std::vector<Scalar> scalar;
  scalar.push_back(abs_lambd);
  return ops::activation_scalar(self_arg, scalar, VK_KERNEL(hardshrink));
}

Tensor& hardshrink_(Tensor& self, const Scalar& lambd) {
  float abs_lambd = std::abs(lambd.to<float>());
  std::vector<Scalar> scalar;
  scalar.push_back(abs_lambd);
  return ops::activation_scalar_(self, scalar, VK_KERNEL(hardshrink_));
}

Tensor leaky_relu(const Tensor& self_arg, const Scalar& negative_slope) {
  std::vector<Scalar> scalar;
  scalar.push_back(negative_slope);
  return ops::activation_scalar(self_arg, scalar, VK_KERNEL(leaky_relu));
}

Tensor& leaky_relu_(Tensor& self, const Scalar& negative_slope) {
  std::vector<Scalar> scalar;
  scalar.push_back(negative_slope);
  return ops::activation_scalar_(self, scalar, VK_KERNEL(leaky_relu_));
}

Tensor softplus(
    const Tensor& self_arg,
    const Scalar& beta,
    const Scalar& threshold) {
  api::AllocationScope allocation_scope("softplus");
  std::vector<Scalar> scalar;
  scalar.push_back(beta);
  scalar.push_back(threshold);
  return ops::activation_scalar(self_arg, scalar, VK_KERNEL(softplus));
}

Tensor sigmoid(const Tensor& self) {
  return ops::activation(self, VK_KERNEL(sigmoid));
}

Tensor& sigmoid_(Tensor& self) {
  return ops::activation_(self, VK_KERNEL(sigmoid_));
}

Tensor tanh(const Tensor& self) {
  return ops::activation(self, VK_KERNEL(tanh));
}

Tensor& tanh_(Tensor& self) {
  return ops::activation_(self, VK_KERNEL(tanh_));
}

Tensor abs(const Tensor& self) {
  return ops::activation(self, VK_KERNEL(abs));
}

Tensor& abs_(Tensor& self) {
  return ops::activation_(self, VK_KERNEL(abs_));
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::clamp"), TORCH_FN(clamp));
  m.impl(TORCH_SELECTIVE_NAME("aten::clamp_"), TORCH_FN(clamp_));
  m.impl(TORCH_SELECTIVE_NAME("aten::gelu"), gelu);
  m.impl(TORCH_SELECTIVE_NAME("aten::gelu_"), gelu_);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardsigmoid"), hardsigmoid);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardsigmoid_"), hardsigmoid_);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardshrink"), hardshrink);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardshrink_"), hardshrink_);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardswish"), hardswish);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardswish_"), hardswish_);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardtanh"), hardtanh);
  m.impl(TORCH_SELECTIVE_NAME("aten::hardtanh_"), hardtanh_);
  m.impl(TORCH_SELECTIVE_NAME("aten::leaky_relu"), leaky_relu);
  m.impl(TORCH_SELECTIVE_NAME("aten::leaky_relu_"), leaky_relu_);
  m.impl(TORCH_SELECTIVE_NAME("aten::sigmoid"), sigmoid);
  m.impl(TORCH_SELECTIVE_NAME("aten::sigmoid_"), sigmoid_);
  m.impl(TORCH_SELECTIVE_NAME("aten::softplus"), softplus);
  m.impl(TORCH_SELECTIVE_NAME("aten::tanh"), tanh);
  m.impl(TORCH_SELECTIVE_NAME("aten::tanh_"), tanh_);
  m.impl(TORCH_SELECTIVE_NAME("aten::abs"), abs);
  m.impl(TORCH_SELECTIVE_NAME("aten::abs_"), abs_);
  m.impl(TORCH_SELECTIVE_NAME("aten::relu"), relu);
  m.impl(TORCH_SELECTIVE_NAME("aten::relu_"), relu_);
  m.impl(TORCH_SELECTIVE_NAME("aten::threshold"), threshold);
}

#endif /* USE_VULKAN_API */

} // namespace

Tensor relu_buffer_out_vulkan(
    const Tensor& input,
    Tensor& output) {
  TORCH_CHECK(
      input.is_vulkan() && input.scalar_type() == at::kFloat &&
          can_run_float_buffer_clamp(convert(input)),
      "Vulkan relu_buffer_out expects float buffer-backed tensors");
  return clamp_buffer_impl(input, 0, std::nullopt, &output);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
