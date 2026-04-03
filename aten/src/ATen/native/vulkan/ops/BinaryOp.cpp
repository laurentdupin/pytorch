#ifdef USE_VULKAN_API
#include <ATen/ArrayRef.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/add.h>
#include <ATen/ops/div.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/sub.h>
#endif
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <cmath>
#include <limits>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

using namespace api::utils;

namespace {

enum class BinaryOpKind : uint8_t {
  Add,
  Sub,
  Mul,
  Div,
  FloorDivide,
  Pow,
};

bool supports_int8_buffer_arithmetic() {
  return api::context()->adapter_ptr()->supports_int8_buffer_arithmetic();
}

bool scalar_fits_int32(const Scalar& scalar) {
  if (!scalar.isIntegral(true)) {
    return false;
  }
  const int64_t value = scalar.to<int64_t>();
  return value >= static_cast<int64_t>(std::numeric_limits<int32_t>::min()) &&
      value <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

int32_t scalar_to_int32(const Scalar& scalar) {
  return safe_downcast<int32_t>(scalar.to<int64_t>());
}

bool is_integral_buffer_dtype(const api::ScalarType dtype) {
  return dtype == api::kInt || dtype == api::kByte || dtype == api::kChar;
}

bool supports_native_integral_buffer_compute_dtype(const api::ScalarType dtype) {
  switch (dtype) {
    case api::kInt:
      return true;
    case api::kByte:
    case api::kChar:
      return supports_int8_buffer_arithmetic();
    default:
      return false;
  }
}

bool last_dim_is_width_aligned(const Tensor& tensor) {
  return tensor.dim() == 0 || tensor.sizes().back() % 4 == 0;
}

const api::ShaderInfo& integral_buffer_scalar_shader(
    const api::ScalarType dtype,
    const BinaryOpKind op_kind) {
  switch (dtype) {
    case api::kInt:
      switch (op_kind) {
        case BinaryOpKind::Add:
          return VK_KERNEL(buffer_int_add_scalar);
        case BinaryOpKind::Sub:
          return VK_KERNEL(buffer_int_sub_scalar);
        case BinaryOpKind::Mul:
          return VK_KERNEL(buffer_int_mul_scalar);
        default:
          break;
      }
      break;
    case api::kChar:
      switch (op_kind) {
        case BinaryOpKind::Add:
          return VK_KERNEL(buffer_char_add_scalar);
        case BinaryOpKind::Sub:
          return VK_KERNEL(buffer_char_sub_scalar);
        case BinaryOpKind::Mul:
          return VK_KERNEL(buffer_char_mul_scalar);
        default:
          break;
      }
      break;
    case api::kByte:
      switch (op_kind) {
        case BinaryOpKind::Add:
          return VK_KERNEL(buffer_byte_add_scalar);
        case BinaryOpKind::Sub:
          return VK_KERNEL(buffer_byte_sub_scalar);
        case BinaryOpKind::Mul:
          return VK_KERNEL(buffer_byte_mul_scalar);
        default:
          break;
      }
      break;
    default:
      break;
  }
  VK_THROW("Unsupported integral buffer scalar binary op");
}

const api::ShaderInfo& integral_buffer_tensor_shader(
    const api::ScalarType dtype,
    const BinaryOpKind op_kind) {
  switch (dtype) {
    case api::kInt:
      switch (op_kind) {
        case BinaryOpKind::Add:
          return VK_KERNEL(buffer_int_add);
        case BinaryOpKind::Sub:
          return VK_KERNEL(buffer_int_sub);
        case BinaryOpKind::Mul:
          return VK_KERNEL(buffer_int_mul);
        default:
          break;
      }
      break;
    case api::kChar:
      switch (op_kind) {
        case BinaryOpKind::Add:
          return VK_KERNEL(buffer_char_add);
        case BinaryOpKind::Sub:
          return VK_KERNEL(buffer_char_sub);
        case BinaryOpKind::Mul:
          return VK_KERNEL(buffer_char_mul);
        default:
          break;
      }
      break;
    case api::kByte:
      switch (op_kind) {
        case BinaryOpKind::Add:
          return VK_KERNEL(buffer_byte_add);
        case BinaryOpKind::Sub:
          return VK_KERNEL(buffer_byte_sub);
        case BinaryOpKind::Mul:
          return VK_KERNEL(buffer_byte_mul);
        default:
          break;
      }
      break;
    default:
      break;
  }
  VK_THROW("Unsupported integral buffer tensor binary op");
}

const api::ShaderInfo& bool_buffer_scalar_shader(const BinaryOpKind op_kind) {
  switch (op_kind) {
    case BinaryOpKind::Add:
      return VK_KERNEL(buffer_bool_add_scalar);
    case BinaryOpKind::Mul:
      return VK_KERNEL(buffer_bool_mul_scalar);
    default:
      VK_THROW("Unsupported bool buffer scalar binary op");
  }
}

const api::ShaderInfo& bool_buffer_tensor_shader(const BinaryOpKind op_kind) {
  switch (op_kind) {
    case BinaryOpKind::Add:
      return VK_KERNEL(buffer_bool_add);
    case BinaryOpKind::Mul:
      return VK_KERNEL(buffer_bool_mul);
    default:
      VK_THROW("Unsupported bool buffer tensor binary op");
  }
}

bool needs_binary_cpu_fallback(const Tensor& tensor) {
  return tensor.is_vulkan() && convert(tensor).dtype() != api::kFloat;
}

bool is_buffer_elementwise_candidate(const Tensor& tensor) {
  return tensor.is_vulkan() &&
      utils::supports_buffer_elementwise_compute(convert(tensor));
}

bool should_run_buffer_binary_scalar(const Tensor& tensor) {
  // Disabled for now. The first scalar-uniform buffer variant still needs
  // dedicated shader validation; keep scalar math on the proven texture path
  // while the generic tensor-tensor and unary buffer lanes are brought up.
  return false;
}

bool is_integral_buffer_compute_candidate(const Tensor& tensor) {
  if (!tensor.is_vulkan()) {
    return false;
  }
  const vTensor& v_tensor = convert(tensor);
  return is_integral_buffer_dtype(v_tensor.dtype()) &&
      supports_native_integral_buffer_compute_dtype(v_tensor.dtype()) &&
      v_tensor.storage_type() == api::StorageType::BUFFER &&
      v_tensor.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
      utils::supports_buffer_elementwise_compute(v_tensor) &&
      !v_tensor.is_quantized();
}

bool should_run_buffer_binary_tensor(const Tensor& self, const Tensor& other) {
  const ScalarType promoted_dtype =
      promote_for_vulkan_binary(self.scalar_type(), other.scalar_type());
  return is_buffer_elementwise_candidate(self) &&
      is_buffer_elementwise_candidate(other) &&
      promoted_dtype == c10::ScalarType::Float &&
      self.scalar_type() == promoted_dtype &&
      other.scalar_type() == promoted_dtype &&
      self.sizes().vec() == other.sizes().vec() &&
      (convert(self).storage_type() == api::StorageType::BUFFER ||
       convert(other).storage_type() == api::StorageType::BUFFER);
}

bool should_run_buffer_binary_tensor_integral(
    const Tensor& self,
    const Tensor& other,
    const std::optional<Scalar>& alpha_arg,
    const BinaryOpKind op_kind) {
  if (self.scalar_type() != other.scalar_type()) {
    return false;
  }
  if (self.is_vulkan() && !is_integral_buffer_compute_candidate(self)) {
    return false;
  }
  if (other.is_vulkan() && !is_integral_buffer_compute_candidate(other)) {
    return false;
  }
  if (self.sizes().vec() != other.sizes().vec()) {
    return false;
  }
  if (
      (self.scalar_type() == kChar || self.scalar_type() == kByte) &&
      !last_dim_is_width_aligned(self)) {
    return false;
  }
  if (
      op_kind != BinaryOpKind::Add && op_kind != BinaryOpKind::Sub &&
      op_kind != BinaryOpKind::Mul) {
    return false;
  }
  if (!alpha_arg.has_value()) {
    return true;
  }
  return scalar_fits_int32(*alpha_arg);
}

bool should_run_buffer_binary_scalar_integral(
    const Tensor& self,
    const Scalar& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const BinaryOpKind op_kind) {
  if (!is_integral_buffer_compute_candidate(self)) {
    return false;
  }
  if (
      (self.scalar_type() == kChar || self.scalar_type() == kByte) &&
      !last_dim_is_width_aligned(self)) {
    return false;
  }
  if (
      op_kind != BinaryOpKind::Add && op_kind != BinaryOpKind::Sub &&
      op_kind != BinaryOpKind::Mul) {
    return false;
  }
  if (!scalar_fits_int32(other_arg)) {
    return false;
  }

  if (!alpha_arg.has_value()) {
    return true;
  }
  return scalar_fits_int32(*alpha_arg);
}

bool is_bool_buffer_compute_candidate(const Tensor& tensor) {
  if (!tensor.is_vulkan()) {
    return false;
  }
  const vTensor& v_tensor = convert(tensor);
  return supports_int8_buffer_arithmetic() && v_tensor.dtype() == api::kBool &&
      utils::supports_buffer_elementwise_compute(v_tensor) &&
      !v_tensor.is_quantized();
}

bool should_run_buffer_binary_tensor_bool(
    const Tensor& self,
    const Tensor& other,
    const std::optional<Scalar>& alpha_arg,
    const BinaryOpKind op_kind) {
  if (self.scalar_type() != c10::kBool || other.scalar_type() != c10::kBool) {
    return false;
  }
  if (self.is_vulkan() && !is_bool_buffer_compute_candidate(self)) {
    return false;
  }
  if (other.is_vulkan() && !is_bool_buffer_compute_candidate(other)) {
    return false;
  }
  if (self.sizes().vec() != other.sizes().vec()) {
    return false;
  }
  if (!last_dim_is_width_aligned(self)) {
    return false;
  }
  if (op_kind != BinaryOpKind::Add && op_kind != BinaryOpKind::Mul) {
    return false;
  }
  return !alpha_arg.has_value() || scalar_fits_int32(*alpha_arg);
}

bool should_run_buffer_binary_scalar_bool(
    const Tensor& self,
    const Scalar& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const BinaryOpKind op_kind) {
  if (self.scalar_type() != c10::kBool || !other_arg.isBoolean()) {
    return false;
  }
  if (!is_bool_buffer_compute_candidate(self)) {
    return false;
  }
  if (!last_dim_is_width_aligned(self)) {
    return false;
  }
  if (op_kind != BinaryOpKind::Add && op_kind != BinaryOpKind::Mul) {
    return false;
  }
  return !alpha_arg.has_value() || scalar_fits_int32(*alpha_arg);
}

Tensor binary_op_scalar_cpu_fallback(
    const Tensor& self_arg,
    const Scalar& other,
    const std::optional<Scalar>& alpha_arg,
    const BinaryOpKind op_kind) {
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
    switch (op_kind) {
      case BinaryOpKind::Add:
        cpu_result = at::add(self_cpu, other, alpha_arg.value_or(Scalar(1)));
        break;
      case BinaryOpKind::Sub:
        cpu_result = at::sub(self_cpu, other, alpha_arg.value_or(Scalar(1)));
        break;
      case BinaryOpKind::Mul:
        cpu_result = at::mul(self_cpu, other);
        break;
      case BinaryOpKind::Div:
        cpu_result = at::div(self_cpu, other);
        break;
      case BinaryOpKind::FloorDivide:
        cpu_result = at::floor_divide(self_cpu, other);
        break;
      case BinaryOpKind::Pow:
        cpu_result = at::pow(self_cpu, other);
        break;
    }
  }
  return cpu_result.vulkan();
}

Tensor binary_op_tensor_cpu_fallback(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const BinaryOpKind op_kind) {
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
    const Tensor other_cpu = other_arg.is_vulkan() ? other_arg.cpu() : other_arg;
    switch (op_kind) {
      case BinaryOpKind::Add:
        cpu_result =
            at::add(self_cpu, other_cpu, alpha_arg.value_or(Scalar(1)));
        break;
      case BinaryOpKind::Sub:
        cpu_result =
            at::sub(self_cpu, other_cpu, alpha_arg.value_or(Scalar(1)));
        break;
      case BinaryOpKind::Mul:
        cpu_result = at::mul(self_cpu, other_cpu);
        break;
      case BinaryOpKind::Div:
        cpu_result = at::div(self_cpu, other_cpu);
        break;
      case BinaryOpKind::FloorDivide:
        cpu_result = at::floor_divide(self_cpu, other_cpu);
        break;
      case BinaryOpKind::Pow:
        cpu_result = at::pow(self_cpu, other_cpu);
        break;
    }
  }
  return cpu_result.vulkan();
}

Tensor prepare_native_integral_buffer_input(const Tensor& input_arg) {
  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  vTensor v_input = convert(input);
  if (
      v_input.dtype() != api::kByte && v_input.dtype() != api::kChar &&
      v_input.dtype() != api::kBool) {
    return utils::ensure_buffer_storage(input);
  }

  input = utils::ensure_buffer_storage(input);
  v_input = convert(input);
  if (
      !v_input.has_direct_buffer_layout() ||
      v_input.storage_type() != api::StorageType::BUFFER ||
      v_input.gpu_memory_layout() !=
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED) {
    return input;
  }

  api::AllocationScope allocation_scope("binary_op.narrow_buffer_materialize");
  api::Context* const context = api::context();
  api::StorageBuffer staging(context, v_input.dtype(), v_input.numel());
  vTensor v_src = v_input;
  utils::pack_vtensor_to_staging(v_src, staging.buffer());

  vTensor v_out{
      context,
      v_input.sizes(),
      v_input.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };
  api::PipelineBarrier pipeline_barrier{};
  add_buffer_barrier(
      pipeline_barrier,
      staging.buffer(),
      api::PipelineStage::COMPUTE | api::PipelineStage::TRANSFER,
      api::MemoryAccessType::WRITE,
      api::PipelineStage::COMPUTE | api::PipelineStage::TRANSFER,
      api::MemoryAccessType::READ);
  utils::pack_buffer_to_vtensor(staging.buffer(), v_out, pipeline_barrier);
  return convert(v_out);
}

} // namespace

static Tensor binary_op_tensor_buffer_integral(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::AllocationScope allocation_scope("binary_op.integral_buffer");
  utils::is_broadcastable(self_arg, other_arg);
  api::Context* const context = api::context();

  Tensor self = prepare_native_integral_buffer_input(self_arg);
  Tensor other = prepare_native_integral_buffer_input(other_arg);
  vTensor& v_self = convert(self);
  vTensor& v_other = convert(other);

  vTensor v_output{
      context,
      utils::broadcast_size(self_arg, other_arg),
      v_self.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct Block final {
    int32_t alpha;
  } block{
      alpha_arg ? alpha_arg->to<int32_t>() : 1,
  };

  api::UniformParamsBuffer params(context, block);
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
  api::UniformParamsBuffer other_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_other);

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
      in_meta.buffer(),
      v_other.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      other_meta.buffer(),
      params.buffer());

  return convert(v_output);
}

static Tensor binary_op_tensor_buffer_bool(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::AllocationScope allocation_scope("binary_op.bool_buffer");
  utils::is_broadcastable(self_arg, other_arg);
  api::Context* const context = api::context();

  Tensor self = prepare_native_integral_buffer_input(self_arg);
  Tensor other = prepare_native_integral_buffer_input(other_arg);
  vTensor& v_self = convert(self);
  vTensor& v_other = convert(other);

  vTensor v_output{
      context,
      utils::broadcast_size(self_arg, other_arg),
      api::kBool,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct Block final {
    int32_t alpha;
  } block{
      alpha_arg ? scalar_to_int32(*alpha_arg) : 1,
  };

  api::UniformParamsBuffer params(context, block);
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
  api::UniformParamsBuffer other_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_other);

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
      in_meta.buffer(),
      v_other.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      other_meta.buffer(),
      params.buffer());

  return convert(v_output);
}

static Tensor binary_op_scalar_buffer(
    const Tensor& self_arg,
    const Scalar& other,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::AllocationScope allocation_scope("binary_op.buffer");
  api::Context* const context = api::context();

  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  self = utils::ensure_buffer_storage(self);
  vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const float other_val = alpha_arg ? other.to<float>() * alpha_arg->to<float>()
                                    : other.to<float>();
  const struct Block final {
    float other;
  } block{
      other_val,
  };

  api::UniformParamsBuffer params(context, block);
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
      in_meta.buffer(),
      params.buffer());

  return convert(v_output);
}

static Tensor binary_op_scalar_buffer_integral(
    const Tensor& self_arg,
    const Scalar& other,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::AllocationScope allocation_scope("binary_op.integral_scalar_buffer");
  api::Context* const context = api::context();

  Tensor self = prepare_native_integral_buffer_input(self_arg);
  vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const int32_t other_val =
      alpha_arg ? safe_downcast<int32_t>(
                      static_cast<int64_t>(scalar_to_int32(other)) *
                      scalar_to_int32(*alpha_arg))
                : scalar_to_int32(other);

  const struct Block final {
    int32_t other;
  } block{
      other_val,
  };

  api::UniformParamsBuffer params(context, block);
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
      in_meta.buffer(),
      params.buffer());

  return convert(v_output);
}

static Tensor binary_op_scalar_buffer_bool(
    const Tensor& self_arg,
    const Scalar& other,
    const api::ShaderInfo& shader_descriptor) {
  api::AllocationScope allocation_scope("binary_op.bool_scalar_buffer");
  api::Context* const context = api::context();

  Tensor self = prepare_native_integral_buffer_input(self_arg);
  vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      api::kBool,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct Block final {
    int32_t other;
  } block{
      other.to<bool>() ? 1 : 0,
  };

  api::UniformParamsBuffer params(context, block);
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
      in_meta.buffer(),
      params.buffer());

  return convert(v_output);
}

static Tensor binary_op_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor,
    const api::ShaderInfo& buffer_shader_descriptor,
  const BinaryOpKind op_kind) {
  api::AllocationScope allocation_scope("binary_op");
  api::Context* const context = api::context();

  if (should_run_buffer_binary_scalar_integral(
          self_arg, other, alpha_arg, op_kind)) {
    return binary_op_scalar_buffer_integral(
        self_arg,
        other,
        alpha_arg,
        integral_buffer_scalar_shader(convert(self_arg).dtype(), op_kind));
  }

  if (should_run_buffer_binary_scalar_bool(self_arg, other, alpha_arg, op_kind)) {
    return binary_op_scalar_buffer_bool(
        self_arg, other, bool_buffer_scalar_shader(op_kind));
  }

  if (needs_binary_cpu_fallback(self_arg)) {
    return binary_op_scalar_cpu_fallback(self_arg, other, alpha_arg, op_kind);
  }

  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  if (should_run_buffer_binary_scalar(self)) {
    return binary_op_scalar_buffer(
        self, other, alpha_arg, buffer_shader_descriptor);
  }
  if (convert(self).storage_type() == api::StorageType::BUFFER) {
    self = utils::ensure_texture_storage(self);
  }
  const vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  const float other_val = alpha_arg ? other.to<float>() * alpha_arg->to<float>()
                                    : other.to<float>();
  const struct Block final {
    uvec3 extents;
    int fill0;
    float other;
  } block{
      v_self.extents(),
      0,
      other_val,
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

static Tensor binary_op_preprocess_other_arg(const Tensor& other_arg) {
  // Similar to binary_op_scalar where tensors is mapped to float, we
  // also map known integer types (but not quant types) tensor to float.

  // Such conversion can only to be done before moving to vulkan, since vulkan
  // doesn't yet support integer types.
  Tensor other = other_arg;
  if (!other.is_vulkan()) {
    switch (other.scalar_type()) {
      case at::kByte:
      case at::kChar:
      case at::kShort:
      case at::kInt:
      case at::kLong:
      case at::kDouble:
        other = other.to(kFloat);
        break;
      case at::kFloat:
        // No op for expected type.
        break;
      default:
        TORCH_CHECK(
            false,
            "binary_op_tensor, doesn't support type %s",
            other.scalar_type());
        break;
    }
    other = other.vulkan();
  }

  return other;
}

static Tensor& binary_op_scalar_(
    Tensor& self_arg,
    const Scalar& other,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::AllocationScope allocation_scope("binary_op_inplace");
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);
  TORCH_CHECK(
      v_self.storage_type() != api::StorageType::BUFFER,
      "In-place Vulkan binary ops do not yet support buffer-backed logical views");

  const float other_val = alpha_arg ? other.to<float>() * alpha_arg->to<float>()
                                    : other.to<float>();
  const struct Block final {
    uvec3 extents;
    int fill0;
    float other;
  } block{
      v_self.extents(),
      0,
      other_val,
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

static Tensor binary_op_tensor_buffer(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::AllocationScope allocation_scope("binary_op.buffer");
  utils::is_broadcastable(self_arg, other_arg);
  api::Context* const context = api::context();

  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();
  self = utils::ensure_buffer_storage(self);
  other = utils::ensure_buffer_storage(other);

  vTensor& v_self = convert(self);
  vTensor& v_other = convert(other);

  vTensor v_output{
      context,
      utils::broadcast_size(self_arg, other_arg),
      v_self.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct Block final {
    float alpha;
  } block{
      alpha_arg ? alpha_arg->to<float>() : 1.0f,
  };

  api::UniformParamsBuffer params(context, block);
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
  api::UniformParamsBuffer other_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_other);

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
      in_meta.buffer(),
      v_other.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      other_meta.buffer(),
      params.buffer());

  return convert(v_output);
}

static Tensor binary_op_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor,
    const api::ShaderInfo& buffer_shader_descriptor,
    const BinaryOpKind op_kind) {
  api::AllocationScope allocation_scope("binary_op");
  utils::is_broadcastable(self_arg, other_arg);
  api::Context* const context = api::context();

  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  if (should_run_buffer_binary_tensor_integral(
          self, other_arg, alpha_arg, op_kind)) {
    return binary_op_tensor_buffer_integral(
        self,
        other_arg,
        alpha_arg,
        integral_buffer_tensor_shader(convert(self).dtype(), op_kind));
  }

  if (should_run_buffer_binary_tensor_bool(self, other_arg, alpha_arg, op_kind)) {
    return binary_op_tensor_buffer_bool(
        self, other_arg, alpha_arg, bool_buffer_tensor_shader(op_kind));
  }

  if (needs_binary_cpu_fallback(self_arg) || needs_binary_cpu_fallback(other_arg)) {
    return binary_op_tensor_cpu_fallback(self_arg, other_arg, alpha_arg, op_kind);
  }

  Tensor other = binary_op_preprocess_other_arg(other_arg);
  if (should_run_buffer_binary_tensor(self, other)) {
    return binary_op_tensor_buffer(
        self, other, alpha_arg, buffer_shader_descriptor);
  }
  if (convert(self).storage_type() == api::StorageType::BUFFER) {
    self = utils::ensure_texture_storage(self);
  }
  const vTensor& v_self = convert(self);
  if (convert(other).storage_type() == api::StorageType::BUFFER) {
    other = utils::ensure_texture_storage(other);
  }

  const vTensor& v_other = convert(other);

  vTensor v_output{
      context,
      utils::broadcast_size(self_arg, other_arg),
      v_self.dtype(),
  };

  const double alpha = alpha_arg ? alpha_arg->to<double>() : 1.0;
  const struct Block final {
    uvec4 output_tensor_size;
    uvec4 input_tensor_size;
    uvec4 other_tensor_size;
    float alpha;
  } block{
      {get_dim<Dim4D::Width>(v_output),
       get_dim<Dim4D::Height>(v_output),
       get_dim<Dim4D::Channel>(v_output),
       get_dim<Dim4D::Batch>(v_output)},

      {get_dim<Dim4D::Width>(v_self),
       get_dim<Dim4D::Height>(v_self),
       get_dim<Dim4D::Channel>(v_self),
       get_dim<Dim4D::Batch>(v_self)},

      {get_dim<Dim4D::Width>(v_other),
       get_dim<Dim4D::Height>(v_other),
       get_dim<Dim4D::Channel>(v_other),
       get_dim<Dim4D::Batch>(v_other)},
      // alpha
      safe_downcast<float>(alpha),
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
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert(v_output);
}

static Tensor quantized_binary_op_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point,
    const api::ShaderInfo& shader_descriptor) {
  api::AllocationScope allocation_scope("qbinary_op");
  utils::is_broadcastable(self_arg, other_arg);
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);
  const Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();
  const vTensor& v_other = convert(other);

  TORCH_CHECK(v_self.is_quantized(), "Input tensor is not quantized");
  TORCH_CHECK(v_other.is_quantized(), "Input tensor is not quantized");

  vTensor v_output{
      context,
      utils::broadcast_size(self_arg, other_arg),
      scale,
      zero_point,
      api::kQUInt8,
  };

  const double scale1 = v_self.get_scale();
  const double scale2 = v_other.get_scale();
  const int64_t zero_point1 = v_self.get_zero_point();
  const int64_t zero_point2 = v_other.get_zero_point();
  const struct Block final {
    uvec3 extents;
    uint32_t channelSize;
    uvec3 input1Extents;
    uint32_t channelBatchSize1;
    uvec3 input2Extents;
    uint32_t channelBatchSize2;
    float scale1;
    float scale2;
    int32_t zeroPoint1;
    int32_t zeroPoint2;
    float scale;
    float fill1;
    int32_t zeroPoint;
    int32_t fill2;
  } block{
      v_output.extents(),
      get_dim<Dim4D::Channel>(v_output),
      v_self.extents(),
      get_dim<Dim4D::Channel>(self) * get_dim<Dim4D::Batch>(self),
      v_other.extents(),
      get_dim<Dim4D::Channel>(other) * get_dim<Dim4D::Batch>(other),
      safe_downcast<float>(scale1),
      safe_downcast<float>(scale2),
      safe_downcast<int32_t>(zero_point1),
      safe_downcast<int32_t>(zero_point2),
      safe_downcast<float>(scale),
      0.0f,
      safe_downcast<int32_t>(zero_point),
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
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert_quantized(v_output);
}

static Tensor& binary_op_tensor_(
    Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor) {
  TORCH_CHECK(
      get_dim<Dim4D::Batch>(self_arg) >= get_dim<Dim4D::Batch>(other_arg) &&
          get_dim<Dim4D::Channel>(self_arg) >=
              get_dim<Dim4D::Channel>(other_arg) &&
          get_dim<Dim4D::Height>(self_arg) >=
              get_dim<Dim4D::Height>(other_arg) &&
          get_dim<Dim4D::Width>(self_arg) >= get_dim<Dim4D::Width>(other_arg),
      "Dimensions of input tensor to Vulkan in-place binary elementwise op "
      "must be less than or equal the dimensions of the underlying tensor.");

  utils::is_broadcastable(self_arg, other_arg);

  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);
  TORCH_CHECK(
      v_self.storage_type() != api::StorageType::BUFFER,
      "In-place Vulkan binary ops do not yet support buffer-backed logical views");

  Tensor other = binary_op_preprocess_other_arg(other_arg);
  if (convert(other).storage_type() == api::StorageType::BUFFER) {
    other = utils::ensure_texture_storage(other);
  }

  const vTensor& v_other = convert(other);

  const double alpha = alpha_arg ? alpha_arg->to<double>() : 1.0;
  const struct Block final {
    uvec4 input_tensor_size;
    uvec4 other_tensor_size;
    float alpha;
  } block{
      {get_dim<Dim4D::Width>(v_self),
       get_dim<Dim4D::Height>(v_self),
       get_dim<Dim4D::Channel>(v_self),
       get_dim<Dim4D::Batch>(v_self)},

      {get_dim<Dim4D::Width>(v_other),
       get_dim<Dim4D::Height>(v_other),
       get_dim<Dim4D::Channel>(v_other),
       get_dim<Dim4D::Batch>(v_other)},
      // alpha
      safe_downcast<float>(alpha),
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
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return self_arg;
}

static Tensor add_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha) {
  return binary_op_scalar(
      self_arg,
      other,
      std::optional<Scalar>(alpha),
      VK_KERNEL(add_scalar),
      VK_KERNEL(buffer_add_scalar),
      BinaryOpKind::Add);
}

static Tensor& add_scalar_(
    Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  return binary_op_scalar_(
      self, other, std::optional<Scalar>(alpha), VK_KERNEL(add_scalar_inplace));
}

Tensor quantized_add(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_binary_op_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_add));
}

Tensor quantized_sub(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_binary_op_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_sub));
}

Tensor quantized_mul(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_binary_op_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_mul));
}

Tensor quantized_div(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_binary_op_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_div));
}

static Tensor add_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha) {
  return binary_op_tensor(
      self_arg,
      other_arg,
      std::optional<Scalar>(alpha),
      VK_KERNEL(add),
      VK_KERNEL(buffer_add),
      BinaryOpKind::Add);
}

static Tensor& add_tensor_(
    Tensor& self,
    const Tensor& other_arg,
    const Scalar& alpha) {
  return binary_op_tensor_(
      self, other_arg, std::optional<Scalar>(alpha), VK_KERNEL(add_inplace));
}

static Tensor sub_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha) {
  return binary_op_scalar(
      self_arg,
      other,
      std::optional<Scalar>(-1 * alpha.to<float>()),
      VK_KERNEL(add_scalar),
      VK_KERNEL(buffer_add_scalar),
      BinaryOpKind::Sub);
}

static Tensor& sub_scalar_(
    Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  return binary_op_scalar_(
      self,
      other,
      std::optional<Scalar>(-1 * alpha.to<float>()),
      VK_KERNEL(add_scalar_inplace));
}

static Tensor sub_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha) {
  return binary_op_tensor(
      self_arg,
      other_arg,
      std::optional<Scalar>(alpha),
      VK_KERNEL(sub),
      VK_KERNEL(buffer_sub),
      BinaryOpKind::Sub);
}

static Tensor& sub_tensor_(
    Tensor& self,
    const Tensor& other_arg,
    const Scalar& alpha) {
  return binary_op_tensor_(
      self, other_arg, std::optional<Scalar>(alpha), VK_KERNEL(sub_inplace));
}

static Tensor mul_scalar(const Tensor& self_arg, const Scalar& other) {
  return binary_op_scalar(
      self_arg,
      other,
      std::optional<Scalar>(),
      VK_KERNEL(mul_scalar),
      VK_KERNEL(buffer_mul_scalar),
      BinaryOpKind::Mul);
}

static Tensor& mul_scalar_(Tensor& self, const Scalar& other) {
  return binary_op_scalar_(
      self, other, std::optional<Scalar>(), VK_KERNEL(mul_scalar_inplace));
}

static Tensor mul_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  return binary_op_tensor(
      self_arg,
      other_arg,
      std::optional<Scalar>(),
      VK_KERNEL(mul),
      VK_KERNEL(buffer_mul),
      BinaryOpKind::Mul);
}

static Tensor& mul_tensor_(Tensor& self, const Tensor& other_arg) {
  return binary_op_tensor_(
      self, other_arg, std::optional<Scalar>(), VK_KERNEL(mul_inplace));
}

static Tensor div_scalar(const Tensor& self_arg, const Scalar& other) {
  return binary_op_scalar(
      self_arg,
      1.0 / other.to<float>(),
      std::optional<Scalar>(),
      VK_KERNEL(mul_scalar),
      VK_KERNEL(buffer_mul_scalar),
      BinaryOpKind::Div);
}

static Tensor& div_scalar_(Tensor& self, const Scalar& other) {
  return binary_op_scalar_(
      self,
      1.0 / other.to<float>(),
      std::optional<Scalar>(),
      VK_KERNEL(mul_scalar_inplace));
}

static Tensor div_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  return binary_op_tensor(
      self_arg,
      other_arg,
      std::optional<Scalar>(),
      VK_KERNEL(div),
      VK_KERNEL(buffer_div),
      BinaryOpKind::Div);
}

static Tensor& div_tensor_(Tensor& self, const Tensor& other_arg) {
  return binary_op_tensor_(
      self, other_arg, std::optional<Scalar>(), VK_KERNEL(div_inplace));
}

static Tensor pow(const Tensor& self, const Tensor& other) {
  return binary_op_tensor(
      self,
      other,
      std::optional<Scalar>(),
      VK_KERNEL(pow),
      VK_KERNEL(buffer_pow),
      BinaryOpKind::Pow);
}

static Tensor& pow_(Tensor& self, const Tensor& other) {
  return binary_op_tensor_(
      self, other, std::optional<Scalar>(), VK_KERNEL(pow_inplace));
}

static Tensor pow_tensor_scalar(const Tensor& self, const Scalar& other) {
  return binary_op_scalar(
      self,
      other,
      std::optional<Scalar>(),
      VK_KERNEL(pow_tensor_scalar),
      VK_KERNEL(buffer_pow_tensor_scalar),
      BinaryOpKind::Pow);
}

static Tensor& pow_tensor_scalar_(Tensor& self, const Scalar& other) {
  return binary_op_scalar_(
      self,
      other,
      std::optional<Scalar>(),
      VK_KERNEL(pow_tensor_scalar_inplace));
}

static Tensor pow_scalar_tensor(const Scalar& self, const Tensor& other) {
  return binary_op_scalar(
      other,
      self,
      std::optional<Scalar>(),
      VK_KERNEL(pow_scalar_tensor),
      VK_KERNEL(buffer_pow_tensor_scalar),
      BinaryOpKind::Pow);
}

static Tensor floor_divide_scalar(const Tensor& self, const Scalar& other) {
  TORCH_CHECK(
      other.to<float>() != 0.0f, "floor_divide_scalar: can't divide by zero");
  return binary_op_scalar(
      self,
      1.0 / other.to<float>(),
      std::optional<Scalar>(),
      VK_KERNEL(floor_mul_scalar),
      VK_KERNEL(buffer_floor_mul_scalar),
      BinaryOpKind::FloorDivide);
}

static Tensor& floor_divide_scalar_(Tensor& self, const Scalar& other) {
  TORCH_CHECK(
      other.to<float>() != 0.0f, "floor_divide_scalar_: can't divide by zero");
  return binary_op_scalar_(
      self,
      1.0 / other.to<float>(),
      std::optional<Scalar>(),
      VK_KERNEL(floor_mul_scalar_inplace));
}

static Tensor floor_divide_tensor(const Tensor& self, const Tensor& other) {
  return binary_op_tensor(
      self,
      other,
      std::optional<Scalar>(),
      VK_KERNEL(floor_divide),
      VK_KERNEL(buffer_floor_divide),
      BinaryOpKind::FloorDivide);
}

static Tensor& floor_divide_tensor_(Tensor& self, const Tensor& other_arg) {
  return binary_op_tensor_(
      self,
      other_arg,
      std::optional<Scalar>(),
      VK_KERNEL(floor_divide_inplace));
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::add.Scalar"), TORCH_FN(add_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::add_.Scalar"), TORCH_FN(add_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::add.Tensor"), TORCH_FN(add_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::add_.Tensor"), TORCH_FN(add_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub.Scalar"), TORCH_FN(sub_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub_.Scalar"), TORCH_FN(sub_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub.Tensor"), TORCH_FN(sub_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub_.Tensor"), TORCH_FN(sub_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul.Scalar"), TORCH_FN(mul_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul_.Scalar"), TORCH_FN(mul_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul.Tensor"), TORCH_FN(mul_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul_.Tensor"), TORCH_FN(mul_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::div.Scalar"), TORCH_FN(div_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::div_.Scalar"), TORCH_FN(div_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::div.Tensor"), TORCH_FN(div_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::div_.Tensor"), TORCH_FN(div_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::pow.Tensor_Tensor"), TORCH_FN(pow));
  m.impl(TORCH_SELECTIVE_NAME("aten::pow_.Tensor"), TORCH_FN(pow_));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::pow.Tensor_Scalar"),
      TORCH_FN(pow_tensor_scalar));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::pow_.Scalar"), TORCH_FN(pow_tensor_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::pow.Scalar"), TORCH_FN(pow_scalar_tensor));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::floor_divide.Scalar"),
      TORCH_FN(floor_divide_scalar));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::floor_divide_.Scalar"),
      TORCH_FN(floor_divide_scalar_));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::floor_divide"),
      TORCH_FN(floor_divide_tensor));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::floor_divide_.Tensor"),
      TORCH_FN(floor_divide_tensor_));
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
#endif /* USE_VULKAN_API */
