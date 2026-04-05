#include <ATen/InferSize.h>
#include <ATen/Functions.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <optional>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace {

bool is_contiguous_stride(
    IntArrayRef sizes,
    IntArrayRef strides) {
  return strides.equals(c10::contiguous_strides(sizes));
}

bool can_use_texture_contiguous_reshape(
    const vTensor& v_self,
    IntArrayRef output_size,
    IntArrayRef output_stride,
    const int64_t storage_offset) {
  if (
      v_self.storage_type() != api::StorageType::TEXTURE_3D ||
      v_self.gpu_memory_layout() !=
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED ||
      v_self.is_quantized() || v_self.sizes().size() > 4 ||
      output_size.size() > 4 || storage_offset != 0) {
    return false;
  }

  if (
      !is_contiguous_stride(v_self.sizes(), logical_strides(v_self)) ||
      !is_contiguous_stride(output_size, output_stride)) {
    return false;
  }

  return c10::multiply_integers(v_self.sizes()) ==
      c10::multiply_integers(output_size);
}

Tensor reshape_contiguous_texture(
    const Tensor& self_arg,
    IntArrayRef output_size) {
  api::AllocationScope allocation_scope("reshape");
  api::Context* const context = api::context();

  Tensor input = utils::prepare_vulkan_execution_tensor(
      self_arg, utils::VulkanExecutionPlanKind::TextureComputeInput);
  const vTensor& v_input = convert(input);

  vTensor v_output{
      context,
      output_size.vec(),
      convert_dtype(self_arg.scalar_type()),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
  };

  const api::utils::uvec4 out_tensor_size =
      api::utils::make_whcn_uvec4(output_size.vec());
  const api::utils::uvec4 in_tensor_size =
      api::utils::make_whcn_uvec4(v_input.sizes());

  const struct Block final {
    api::utils::ivec3 out_extents;
    int32_t fill0;
    api::utils::uvec4 out_tensor_size;
    api::utils::uvec4 in_tensor_size;
    api::utils::uvec2 aligned_channels;
    api::utils::uvec2 fill1;
  } block{
      api::utils::make_ivec3(v_output.extents()),
      0,
      out_tensor_size,
      in_tensor_size,
      {
          api::utils::align_up(out_tensor_size.data[2u], 4u),
          api::utils::align_up(in_tensor_size.data[2u], 4u),
      },
      {0u, 0u},
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(reshape_texture),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  utils::log_vulkan_op_hit("aten::view.texture_contiguous_reshape");
  return convert(v_output);
}

Tensor view_internal(
    const Tensor& self_arg,
    const IntArrayRef output_size,
    const IntArrayRef output_stride,
    const std::optional<int64_t> storage_offset = std::nullopt) {
  if (self_arg.is_vulkan()) {
    const vTensor& v_self = convert(self_arg);
    const int64_t resolved_storage_offset =
        storage_offset.value_or(v_self.storage_offset());
    const c10::DimVector v_self_logical_strides = logical_strides(v_self);
    if (
        v_self.storage_type() == api::StorageType::BUFFER &&
        c10::IntArrayRef(v_self_logical_strides) ==
            c10::IntArrayRef(v_self.gpu_strides()) &&
        utils::can_make_buffer_metadata_view(
            v_self,
            output_size,
            output_stride,
            output_stride,
            resolved_storage_offset)) {
      return utils::make_buffer_metadata_view(
          self_arg,
          output_size,
          output_stride,
          output_stride,
          resolved_storage_offset);
    }

    if (can_use_texture_contiguous_reshape(
            v_self, output_size, output_stride, resolved_storage_offset)) {
      return reshape_contiguous_texture(self_arg, output_size);
    }
  }

  // Vulkan views are not true metadata aliases yet. Use the proven CPU
  // reshape/as_strided path and rematerialize a fresh Vulkan tensor.
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);
  Tensor cpu = self_arg.cpu();
  Tensor cpu_view = storage_offset.has_value()
      ? cpu.as_strided(output_size.vec(), output_stride.vec(), *storage_offset)
      : cpu.as_strided(output_size.vec(), output_stride.vec());
  Tensor out = at::empty(
      output_size.vec(),
      self_arg.options().device(at::kVulkan));
  ops::copy_(out, cpu_view);
  return out;
}

} // namespace

inline Tensor view(const Tensor& self_arg, IntArrayRef shape) {
  at::DimVector inferred_size = at::infer_size_dv(shape, self_arg.numel());
  IntArrayRef base_sizes = self_arg.sizes();
  IntArrayRef base_strides = self_arg.strides();
  c10::DimVector base_logical_strides;
  if (self_arg.is_vulkan()) {
    const vTensor& v_self = convert(self_arg);
    base_logical_strides = logical_strides(v_self);
    base_sizes = v_self.sizes();
    base_strides = base_logical_strides;
  }
  auto inferred_stride = at::detail::computeStride(
      base_sizes,
      base_strides,
      inferred_size);
  TORCH_CHECK(
      inferred_stride.has_value(),
      "view size is not compatible with input tensor's size and stride");
  return view_internal(self_arg, inferred_size, *inferred_stride);
}

static Tensor _reshape_alias(
    const Tensor& self_arg,
    const IntArrayRef shape,
    const IntArrayRef strides) {
  return view_internal(self_arg, shape, strides);
}

static Tensor as_strided(
    const Tensor& self_arg,
    const IntArrayRef shape,
    const IntArrayRef strides,
    const std::optional<int64_t> storage_offset) {
  return view_internal(self_arg, shape, strides, storage_offset);
}

static Tensor im2col(
    const Tensor& self_arg,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  Tensor cpu = self_arg.cpu();
  Tensor cpu_result =
      at::im2col(cpu, kernel_size.vec(), dilation.vec(), padding.vec(), stride.vec());
  Tensor out = at::empty(
      cpu_result.sizes(),
      self_arg.options().device(at::kVulkan));
  ops::copy_(out, cpu_result);
  return out;
}

static Tensor& im2col_out(
    const Tensor& self_arg,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    Tensor& out) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  Tensor cpu = self_arg.cpu();
  Tensor cpu_result =
      at::im2col(cpu, kernel_size.vec(), dilation.vec(), padding.vec(), stride.vec());

  Tensor vulkan_result = at::empty(cpu_result.sizes(), out.options());
  ops::copy_(vulkan_result, cpu_result);
  return rebind_vulkan_output(out, vulkan_result);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::as_strided"), TORCH_FN(as_strided));
  m.impl(TORCH_SELECTIVE_NAME("aten::im2col"), TORCH_FN(im2col));
  m.impl(TORCH_SELECTIVE_NAME("aten::im2col.out"), TORCH_FN(im2col_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::view"), TORCH_FN(view));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_reshape_alias"), TORCH_FN(_reshape_alias));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
