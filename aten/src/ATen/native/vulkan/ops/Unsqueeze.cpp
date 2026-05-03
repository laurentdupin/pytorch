#include <ATen/Functions.h>
#include <ATen/native/vulkan/ops/BinaryOp.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/LayoutTransitions.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

struct Block final {
  ivec2 info;
};

Tensor unsqueeze_buffer_view(const at::Tensor& self, int64_t dim) {
  const vTensor& v_self = convert(self);
  const int64_t nDims = self.dim();
  const int64_t insert_dim = maybe_wrap_dim(dim, nDims + 1);

  c10::DimVector output_sizes(v_self.sizes().begin(), v_self.sizes().end());
  c10::DimVector output_logical_strides = logical_strides(v_self);
  c10::DimVector output_physical_strides(
      v_self.gpu_strides().begin(), v_self.gpu_strides().end());

  const int64_t inserted_logical_stride =
      insert_dim >= nDims
      ? 1
      : output_logical_strides[insert_dim] *
          std::max<int64_t>(output_sizes[insert_dim], 1);
  const int64_t inserted_physical_stride =
      insert_dim >= nDims
      ? 1
      : output_physical_strides[insert_dim] *
          std::max<int64_t>(output_sizes[insert_dim], 1);

  output_sizes.insert(output_sizes.begin() + insert_dim, 1);
  output_logical_strides.insert(
      output_logical_strides.begin() + insert_dim, inserted_logical_stride);
  output_physical_strides.insert(
      output_physical_strides.begin() + insert_dim, inserted_physical_stride);

  Tensor output = make_buffer_metadata_view_checked(
      self,
      output_sizes,
      output_logical_strides,
      output_physical_strides,
      v_self.storage_offset(),
      "aten::unsqueeze");
  move_deferred_image_normalize_candidate_to_alias(self, output);
  return output;
}

Tensor unsqueeze(const at::Tensor& self, int64_t dim) {
  TORCH_CHECK(
      dim >= -self.dim() - 1 && dim <= self.dim(),
      "Vulkan unsqueeze dimension out of range expected to be in range of [",
      -self.dim() - 1,
      ",",
      self.dim(),
      "], but got ",
      dim);

  const bool needs_cpu_fallback = [&]() {
    if (self.dim() > 3) {
      return true;
    }
    if (self.is_vulkan()) {
      const vTensor& v_self = convert(self);
      if (v_self.storage_type() == api::StorageType::BUFFER) {
        return !utils::supports_buffer_view_fast_path(v_self);
      }
    }
    return false;
  }();

  if (needs_cpu_fallback) {
    report_vulkan_cpu_fallback(
        "aten::unsqueeze", "cpu_fallback", {self});
    // Vulkan unsqueeze is not a true metadata-only view yet for higher-rank
    // tensors or buffer-backed tensors. Fall back to the proven CPU path and
    // rematerialize a fresh Vulkan tensor, matching the approach used by
    // view/as_strided.
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    Tensor cpu = self.cpu();
    Tensor cpu_unsqueezed = cpu.unsqueeze(dim);
    return record_tensor_write_and_return(
        convert(ops::to_vulkan(cpu_unsqueezed, api::StorageType::BUFFER)),
        "aten::unsqueeze",
        "cpu_fallback",
        {self});
  }

  if (self.is_vulkan()) {
    const vTensor& v_self = convert(self);
    if (v_self.storage_type() == api::StorageType::BUFFER) {
      return unsqueeze_buffer_view(self, dim);
    }
  }

  // Get the global Vulkan context
  api::Context* const context = api::context();

  // Cast the input Tensor to a vTensor
  const Tensor input = self.is_vulkan() ? self : self.vulkan();
  const vTensor& v_input = convert(input);

  // Create the output texture. For unsqueeze, add a dimension.
  std::vector<int64_t> output_size = v_input.sizes();
  if (dim < 0) {
    dim += (self.dim() + 1);
  }
  output_size.insert(output_size.begin() + dim, 1);
  // Create the output texture
  vTensor v_output{
      context,
      output_size,
      convert_dtype(self.scalar_type()),
  };

  // Required to determine how to insert memory barriers in the command buffer
  api::PipelineBarrier pipeline_barrier{};

  // Total number of work items is equal to the size of the output texture
  uvec3 global_size = v_output.extents();
  // Adaptively determine local work group size, will usually be {4, 4, 4}
  uvec3 local_size = adaptive_work_group_size(global_size);

  // When unsqueezing in the 0th dimension, only the metadata changes.
  // So we can perform a copy.
  if (dim == 0) {
    const vTensor& v_self = convert(self);
    uvec3 src_offset{};
    uvec3 dst_offset{};
    context->submit_copy<api::VulkanImage, api::VulkanImage>(
        // pipeline barrier
        pipeline_barrier,
        // images
        v_self.image(pipeline_barrier, api::PipelineStage::TRANSFER),
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::TRANSFER,
            api::MemoryAccessType::WRITE),
        // copy details
        v_self.extents(),
        src_offset,
        dst_offset,
        // fence handle
        VK_NULL_HANDLE);
    return record_tensor_write_and_return(
        convert(v_output), "aten::unsqueeze", "texture_copy_dim0", {input});
  }

  else {
    int channel_index = 1; // Channel dimension in a 3D tensor
    // Shift dim and channel_index for 1D, 2D tensors
    if (self.dim() < 3) {
      dim += (3 - self.dim());
      channel_index = 0;
    }

    // Create the params buffer
    struct Block block{{
        // Dimension to unsqueeze
        static_cast<int32_t>(dim),
        // Keep track of the channel in Image3D
        static_cast<int32_t>(
            std::ceil(static_cast<float>(output_size[channel_index]) / 4)),
    }};

    api::UniformParamsBuffer params(context, block);

    context->submit_compute_job(
        // shader descriptor
        VK_KERNEL(unsqueeze),
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
        // params buffer
        params.buffer());
    return record_tensor_write_and_return(
        convert(v_output), "aten::unsqueeze", "texture", {input});
  }
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::unsqueeze"), TORCH_FN(unsqueeze));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
