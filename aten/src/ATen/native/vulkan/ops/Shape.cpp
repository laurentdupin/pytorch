#include <ATen/InferSize.h>
#include <ATen/Functions.h>
#include <c10/core/DispatchKeySet.h>
#include <ATen/ops/contiguous_ops.h>
#include <ATen/native/vulkan/ops/BinaryOp.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/ops/LayoutTransitions.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/Softmax.h>
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

bool is_vulkan_logically_contiguous(const Tensor& tensor) {
  if (!tensor.is_vulkan()) {
    return tensor.is_contiguous();
  }

  const vTensor& v_tensor = convert(tensor);
  return is_contiguous_stride(v_tensor.sizes(), logical_strides(v_tensor));
}

std::vector<int64_t> buffer_physical_sizes_for_contiguous_view(
    IntArrayRef sizes,
    const api::GPUMemoryLayout memory_layout) {
  std::vector<int64_t> physical_sizes(sizes.begin(), sizes.end());
  if (physical_sizes.empty()) {
    return physical_sizes;
  }

  switch (memory_layout) {
    case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
      physical_sizes.back() =
          api::utils::align_up(physical_sizes.back(), INT64_C(4));
      break;
    case api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED:
      if (physical_sizes.size() >= 2) {
        physical_sizes[physical_sizes.size() - 2] =
            api::utils::align_up(
                physical_sizes[physical_sizes.size() - 2], INT64_C(4));
      } else {
        physical_sizes.back() =
            api::utils::align_up(physical_sizes.back(), INT64_C(4));
      }
      break;
    case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
      if (physical_sizes.size() >= 3) {
        physical_sizes[physical_sizes.size() - 3] =
            api::utils::align_up(
                physical_sizes[physical_sizes.size() - 3], INT64_C(4));
      } else {
        physical_sizes.front() =
            api::utils::align_up(physical_sizes.front(), INT64_C(4));
      }
      break;
  }

  return physical_sizes;
}

bool can_use_buffer_preserved_contiguous_reshape(
    const vTensor& v_self,
    IntArrayRef output_size,
    IntArrayRef output_stride,
    const int64_t storage_offset) {
  if (
      v_self.storage_type() != api::StorageType::BUFFER ||
      !utils::supports_buffer_metadata_view_fast_path(v_self) ||
      storage_offset != 0) {
    return false;
  }

  if (
      !is_contiguous_stride(v_self.sizes(), logical_strides(v_self)) ||
      !is_contiguous_stride(output_size, output_stride)) {
    return false;
  }

  if (
      c10::multiply_integers(v_self.sizes()) !=
      c10::multiply_integers(output_size)) {
    return false;
  }

  const std::vector<int64_t> output_physical_sizes =
      buffer_physical_sizes_for_contiguous_view(
          output_size, v_self.gpu_memory_layout());
  if (
      c10::multiply_integers(output_physical_sizes) !=
      v_self.buffer_length()) {
    return false;
  }

  const auto output_physical_strides =
      c10::contiguous_strides(output_physical_sizes);
  return utils::can_make_buffer_metadata_view(
      v_self,
      output_size,
      output_stride,
      output_physical_strides,
      0);
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

bool can_use_buffer_materialized_contiguous_reshape(
    const vTensor& v_self,
    IntArrayRef output_size,
    IntArrayRef output_stride,
    const int64_t storage_offset) {
  if (
      v_self.storage_type() == api::StorageType::BUFFER ||
      !utils::supports_buffer_metadata_view_fast_path(v_self) ||
      output_size.size() > 5 ||
      storage_offset != 0) {
    return false;
  }

  if (
      !is_contiguous_stride(v_self.sizes(), logical_strides(v_self)) ||
      !is_contiguous_stride(output_size, output_stride)) {
    return false;
  }

  // Width-packed buffers insert padding whenever the logical width is not a
  // multiple of four. A metadata-only reshape would surface those padded slots
  // as logical values, so only use this path when the materialized buffer would
  // stay direct and gap-free.
  if (
      !v_self.sizes().empty() &&
      api::utils::align_up(v_self.sizes().back(), INT64_C(4)) !=
          v_self.sizes().back()) {
    return false;
  }

  return c10::multiply_integers(v_self.sizes()) ==
      c10::multiply_integers(output_size);
}

std::vector<int64_t> texture_gpu_sizes_for_contiguous_view(
    IntArrayRef sizes,
    const api::GPUMemoryLayout memory_layout) {
  const auto logical_val_at = [&](const int64_t index) {
    const int64_t resolved = static_cast<int64_t>(sizes.size()) + index;
    return resolved >= 0 && resolved < static_cast<int64_t>(sizes.size())
        ? sizes[resolved]
        : INT64_C(1);
  };
  std::vector<int64_t> gpu_sizes(4);
  gpu_sizes.at(0) = logical_val_at(-4);
  gpu_sizes.at(1) = logical_val_at(-3);
  gpu_sizes.at(2) = logical_val_at(-2);
  gpu_sizes.at(3) = logical_val_at(-1);

  switch (memory_layout) {
    case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
      gpu_sizes.at(3) = api::utils::align_up(gpu_sizes.at(3), INT64_C(4));
      break;
    case api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED:
      gpu_sizes.at(2) = api::utils::align_up(gpu_sizes.at(2), INT64_C(4));
      break;
    case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
      gpu_sizes.at(1) = api::utils::align_up(gpu_sizes.at(1), INT64_C(4));
      break;
  }

  return gpu_sizes;
}

bool can_use_texture_metadata_reshape(
    const vTensor& v_self,
    IntArrayRef output_size,
    IntArrayRef output_stride,
    const int64_t storage_offset) {
  if (
      v_self.storage_type() != api::StorageType::TEXTURE_3D ||
      v_self.gpu_memory_layout() !=
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED ||
      v_self.is_quantized() || storage_offset != 0) {
    return false;
  }

  if (
      !is_contiguous_stride(v_self.sizes(), logical_strides(v_self)) ||
      !is_contiguous_stride(output_size, output_stride)) {
    return false;
  }

  if (
      c10::multiply_integers(v_self.sizes()) !=
      c10::multiply_integers(output_size)) {
    return false;
  }

  return texture_gpu_sizes_for_contiguous_view(
             v_self.sizes(), v_self.gpu_memory_layout()) ==
      texture_gpu_sizes_for_contiguous_view(
             output_size, v_self.gpu_memory_layout());
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

  api::StorageBuffer staging(context, v_input.dtype(), v_input.numel());
  vTensor v_src = v_input;
  utils::pack_vtensor_to_staging(v_src, staging.buffer());
  api::PipelineBarrier pipeline_barrier{};
  add_buffer_barrier(
      pipeline_barrier,
      staging.buffer(),
      api::PipelineStage::COMPUTE | api::PipelineStage::TRANSFER,
      api::MemoryAccessType::WRITE,
      api::PipelineStage::COMPUTE | api::PipelineStage::TRANSFER,
      api::MemoryAccessType::READ);
  utils::pack_buffer_to_vtensor(staging.buffer(), v_output, pipeline_barrier);

  utils::log_vulkan_op_hit("aten::view.texture_contiguous_reshape");
  return convert(v_output);
}

Tensor reshape_contiguous_as_buffer_view(
    const Tensor& self_arg,
    IntArrayRef output_size,
    IntArrayRef output_stride) {
  Tensor buffer_input = utils::ensure_buffer_storage(
      self_arg, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  const vTensor& v_buffer_input = convert(buffer_input);

  TORCH_INTERNAL_ASSERT(
      utils::can_make_buffer_metadata_view(
          v_buffer_input,
          output_size,
          output_stride,
          output_stride,
          0),
      "Buffer-backed contiguous reshape expected a valid metadata view");

  utils::log_vulkan_op_hit("aten::view.texture_to_buffer_metadata_reshape");
  return make_buffer_metadata_view_checked(
      buffer_input,
      output_size,
      output_stride,
      output_stride,
      0,
      "aten::view");
}

Tensor view_internal(
    const Tensor& self_arg,
    const IntArrayRef output_size,
    const IntArrayRef output_stride,
    const std::optional<int64_t> storage_offset = std::nullopt) {
  const Tensor add_layer_norm_materialized =
      materialize_deferred_add_layer_norm_candidate_if_needed(self_arg);
  if (add_layer_norm_materialized.unsafeGetTensorImpl() !=
      self_arg.unsafeGetTensorImpl()) {
    return view_internal(
        add_layer_norm_materialized, output_size, output_stride, storage_offset);
  }

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
      Tensor output = make_buffer_metadata_view_checked(
          self_arg,
          output_size,
          output_stride,
          output_stride,
          resolved_storage_offset,
          "aten::view");
      move_decomposed_attention_candidate_to_alias(self_arg, output);
      move_deferred_attention_query_scale_candidate_to_alias(self_arg, output);
      move_deferred_linear_gelu_candidate_to_alias(self_arg, output);
      move_deferred_image_normalize_candidate_to_alias(self_arg, output);
      return output;
    }

    if (can_use_buffer_preserved_contiguous_reshape(
            v_self, output_size, output_stride, resolved_storage_offset)) {
      const std::vector<int64_t> output_physical_sizes =
          buffer_physical_sizes_for_contiguous_view(
              output_size, v_self.gpu_memory_layout());
      const auto output_physical_strides =
          c10::contiguous_strides(output_physical_sizes);
      utils::log_vulkan_op_hit("aten::view.buffer_preserve_padded_reshape");
      Tensor output = make_buffer_metadata_view_checked(
          self_arg,
          output_size,
          output_stride,
          output_physical_strides,
          resolved_storage_offset,
          "aten::view");
      move_decomposed_attention_candidate_to_alias(self_arg, output);
      move_deferred_attention_query_scale_candidate_to_alias(self_arg, output);
      move_deferred_linear_gelu_candidate_to_alias(self_arg, output);
      move_deferred_image_normalize_candidate_to_alias(self_arg, output);
      return output;
    }

    if (can_use_texture_metadata_reshape(
            v_self, output_size, output_stride, resolved_storage_offset)) {
      utils::log_vulkan_op_hit("aten::view.texture_metadata_reshape");
      Tensor output = convert(vTensor{
          v_self,
          output_size.vec(),
          output_stride.vec(),
          vTensor::PreservePhysicalView{},
      });
      move_decomposed_attention_candidate_to_alias(self_arg, output);
      move_deferred_attention_query_scale_candidate_to_alias(self_arg, output);
      move_deferred_linear_gelu_candidate_to_alias(self_arg, output);
      move_deferred_image_normalize_candidate_to_alias(self_arg, output);
      return output;
    }

    if (can_use_buffer_materialized_contiguous_reshape(
            v_self, output_size, output_stride, resolved_storage_offset)) {
      return reshape_contiguous_as_buffer_view(
          self_arg, output_size, output_stride);
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
  Tensor materialized_self =
      materialize_decomposed_attention_candidate_if_needed(self_arg);
  materialized_self =
      materialize_deferred_linear_gelu_candidate_if_needed(materialized_self);
  materialized_self =
      materialize_deferred_add_layer_norm_candidate_if_needed(materialized_self);
  materialized_self =
      materialize_deferred_image_normalize_candidate_if_needed(materialized_self);
  Tensor cpu = materialized_self.cpu();
  Tensor cpu_view = storage_offset.has_value()
      ? cpu.as_strided(output_size.vec(), output_stride.vec(), *storage_offset)
      : cpu.as_strided(output_size.vec(), output_stride.vec());
  Tensor out = at::empty(
      output_size.vec(),
      materialized_self.options().device(materialized_self.device()));
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

static Tensor contiguous(
    const Tensor& self_arg,
    c10::MemoryFormat memory_format) {
  TORCH_CHECK(
      memory_format == c10::MemoryFormat::Contiguous ||
          memory_format == c10::MemoryFormat::Preserve,
      "Vulkan contiguous supports Contiguous and Preserve memory formats");

  if (!self_arg.is_vulkan()) {
    return self_arg.contiguous(memory_format);
  }

  Tensor self = materialize_decomposed_attention_candidate_if_needed(self_arg);
  self = materialize_deferred_linear_gelu_candidate_if_needed(self);
  self = materialize_deferred_add_layer_norm_candidate_if_needed(self);
  self = materialize_deferred_image_normalize_candidate_if_needed(self);

  if (memory_format == c10::MemoryFormat::Preserve ||
      is_vulkan_logically_contiguous(self)) {
    return self;
  }

  const vTensor& v_self = convert(self);
  if (
      v_self.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_view_fast_path(v_self)) {
    utils::log_vulkan_op_hit("aten::contiguous.buffer_materialize");
    return utils::ensure_buffer_storage(
        self, v_self.gpu_memory_layout());
  }

  utils::log_vulkan_op_hit("aten::contiguous.clone_fallback");
  return at::clone(
      self, std::optional<c10::MemoryFormat>(c10::MemoryFormat::Contiguous));
}

static Tensor contiguous_autograd_other(
    c10::DispatchKeySet ks,
    const Tensor& self_arg,
    c10::MemoryFormat memory_format) {
  return at::_ops::contiguous::redispatch(
      ks & c10::after_autograd_keyset,
      self_arg,
      memory_format);
}

static Tensor _reshape_alias(
    const Tensor& self_arg,
    const IntArrayRef shape,
    const IntArrayRef strides) {
  return view_internal(self_arg, shape, strides);
}

static Tensor alias(const Tensor& self_arg) {
  if (!self_arg.is_vulkan()) {
    return at::alias(self_arg);
  }

  const vTensor& v_self = convert(self_arg);
  c10::DimVector sizes(v_self.sizes().begin(), v_self.sizes().end());
  c10::DimVector strides = logical_strides(v_self);
  return view_internal(self_arg, sizes, strides, v_self.storage_offset());
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
      self_arg.options().device(self_arg.device()));
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
  m.impl(TORCH_SELECTIVE_NAME("aten::alias"), TORCH_FN(alias));
  m.impl(TORCH_SELECTIVE_NAME("aten::as_strided"), TORCH_FN(as_strided));
  m.impl(TORCH_SELECTIVE_NAME("aten::contiguous"), TORCH_FN(contiguous));
  m.impl(TORCH_SELECTIVE_NAME("aten::im2col"), TORCH_FN(im2col));
  m.impl(TORCH_SELECTIVE_NAME("aten::im2col.out"), TORCH_FN(im2col_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::view"), TORCH_FN(view));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_reshape_alias"), TORCH_FN(_reshape_alias));
}

TORCH_LIBRARY_IMPL(aten, AutogradOther, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::contiguous"),
      TORCH_FN(contiguous_autograd_other));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
