#include <ATen/native/vulkan/ops/Concat.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/cat.h>
#endif
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/LayoutTransitions.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <c10/util/irange.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

using namespace api::utils;

namespace {
inline int64_t normalize_dim(int64_t d, int64_t n) {
  return (d % n + n) % n;
}

std::vector<int64_t> calc_contiguous_strides(IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (int64_t idx = static_cast<int64_t>(sizes.size()) - 2; idx >= 0; --idx) {
    strides[idx] = strides[idx + 1] * std::max<int64_t>(sizes[idx + 1], 1);
  }
  return strides;
}

Tensor cat_cpu_fallback(
    const MaterializedITensorListRef& tensors,
    const int64_t in_dim) {
  std::vector<Tensor> fallback_inputs;
  fallback_inputs.reserve(tensors.size());
  for (const at::Tensor& t : tensors) {
    fallback_inputs.push_back(t);
  }
  report_vulkan_cpu_fallback(
      "aten::cat", "cpu_fallback", fallback_inputs);
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);
  std::vector<Tensor> cpu_tensors;
  cpu_tensors.reserve(tensors.size());
  for (const at::Tensor& t : tensors) {
    cpu_tensors.push_back(t.is_vulkan() ? t.cpu() : t);
  }
  const c10::Device output_device =
      !tensors.empty() && tensors[0].get().is_vulkan()
      ? tensors[0].get().device()
      : c10::Device(at::kVulkan, api::current_device());
  return at::cat(cpu_tensors, in_dim).to(output_device);
}

bool cat_requires_cpu_fallback(const MaterializedITensorListRef& tensors) {
  for (const at::Tensor& t : tensors) {
    if (!t.is_vulkan()) {
      return true;
    }
    const vTensor& v_t = convert(t);
    if (v_t.storage_type() == api::StorageType::BUFFER) {
      return true;
    }
  }
  return false;
}

bool can_use_buffer_cat_fast_path(
    const MaterializedITensorListRef& tensors,
    const int64_t dim) {
  if (tensors.empty()) {
    return false;
  }

  const Tensor& reference = tensors[0];
  if (
      reference.scalar_type() != kFloat ||
      dim < 0 || dim >= reference.dim() || dim == reference.dim() - 1) {
    return false;
  }

  bool has_buffer_input = false;
  for (const Tensor& tensor : tensors) {
    if (!tensor.is_vulkan() || tensor.dim() != reference.dim()) {
      return false;
    }
    const vTensor& v_tensor = convert(tensor);
    if (v_tensor.storage_type() == api::StorageType::BUFFER) {
      has_buffer_input = true;
    }
  }

  return has_buffer_input;
}

bool cat_buffer_direct_out_impl(
    at::ArrayRef<Tensor> tensors,
    const int64_t dim,
    Tensor& output_arg) {
  if (tensors.empty() || !output_arg.defined() || !output_arg.is_vulkan()) {
    return false;
  }

  api::AllocationScope allocation_scope("cat.buffer_direct");
  std::vector<Tensor> prepared_tensors;
  prepared_tensors.reserve(tensors.size());
  for (const Tensor& tensor : tensors) {
    Tensor prepared = tensor;
    const vTensor& v_tensor = convert(prepared);
    if (v_tensor.numel() == 0 || tensor.size(dim) == 0) {
      prepared_tensors.push_back(std::move(prepared));
      continue;
    }
    if (
        v_tensor.storage_type() != api::StorageType::BUFFER ||
        !utils::supports_buffer_elementwise_compute(v_tensor)) {
      prepared = utils::mark_tensor_execution(
          utils::ensure_buffer_storage(
              prepared, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
          api::ExecutionLayout::BUFFER_DIRECT);
      const vTensor& v_buffer = convert(prepared);
      TORCH_CHECK(
          v_buffer.storage_type() == api::StorageType::BUFFER &&
              v_buffer.has_direct_buffer_layout(),
          "Vulkan buffer cat requires buffer-backed inputs");
    }
    prepared_tensors.push_back(std::move(prepared));
  }

  const Tensor& reference = prepared_tensors[0];
  auto result_size = reference.sizes().vec();
  int64_t cat_dim_size = 0;
  for (const Tensor& tensor : prepared_tensors) {
    if (tensor.dim() != reference.dim() || dim < 0 || dim >= tensor.dim()) {
      return false;
    }
    for (const auto d : c10::irange(tensor.dim())) {
      if (d != dim && tensor.size(d) != reference.size(d)) {
        return false;
      }
    }
    cat_dim_size += tensor.size(dim);
  }
  result_size[dim] = cat_dim_size;
  if (output_arg.sizes().vec() != result_size) {
    return false;
  }

  api::Context* const context = api::context();
  Tensor output = utils::mark_tensor_execution(
      output_arg,
      utils::resolve_buffer_execution_layout(convert(output_arg)),
      false);
  vTensor& v_output = convert(output);
  if (
      v_output.storage_type() != api::StorageType::BUFFER ||
      !v_output.has_direct_buffer_layout()) {
    return false;
  }
  int64_t dst_dim_offset = 0;

  for (const Tensor& tensor : prepared_tensors) {
    vTensor& v_input = convert(tensor);
    if (v_input.numel() == 0 || tensor.size(dim) == 0) {
      continue;
    }
    const std::vector<int64_t> logical_strides =
        calc_contiguous_strides(tensor.sizes());
    const int64_t output_storage_offset =
        dst_dim_offset * v_output.gpu_strides()[dim];
    Tensor output_view = make_buffer_metadata_view_checked(
        output,
        tensor.sizes(),
        logical_strides,
        v_output.gpu_strides(),
        output_storage_offset,
        "aten::cat");
    vTensor& v_output_view = convert(output_view);

    api::PipelineBarrier pipeline_barrier{};
    const api::utils::uvec3 global_size = {
        api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_input.numel(), 1)),
        1u,
        1u,
    };
    note_vulkan_buffer_copy(
        VulkanBufferCopyReason::ViewMaterialization,
        v_input,
        v_output_view,
        "aten::cat",
        "buffer_to_buffer");
    context->submit_compute_job(
        VK_KERNEL(buffer_to_buffer),
        pipeline_barrier,
        global_size,
        adaptive_work_group_size(global_size),
        VK_NULL_HANDLE,
        v_output_view.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        utils::make_buffer_compute_metadata_ubo(context, v_output_view).buffer(),
        v_input.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        utils::make_buffer_compute_metadata_ubo(context, v_input).buffer());

    dst_dim_offset += tensor.size(dim);
  }

  output_arg = output;
  return true;
}

Tensor cat_buffer_direct(
    const MaterializedITensorListRef& tensors,
    const int64_t dim,
    IntArrayRef result_size) {
  Tensor output = utils::create_buffer_tensor(
      result_size,
      tensors[0].get().scalar_type(),
      /*persistent=*/false);
  std::vector<Tensor> tensor_vec(tensors.begin(), tensors.end());
  const bool success =
      cat_buffer_direct_out_impl(tensor_vec, dim, output);
  if (!success) {
    return cat_cpu_fallback(tensors, dim);
  }
  return output;
}

} // namespace

bool cat_buffer_out_vulkan(
    at::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& output) {
  return cat_buffer_direct_out_impl(tensors, dim, output);
}

Tensor cat_batch(const MaterializedITensorListRef& tensors, vTensor& v_output) {
  api::Context* const context = api::context();

  uvec3 src_offset{};
  uvec3 dst_offset{};

  for (const at::Tensor& tensor : tensors) {
    const Tensor self = tensor.is_vulkan() ? tensor : tensor.vulkan();
    const vTensor& v_self = convert(self);

    api::PipelineBarrier pipeline_barrier{};

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

    // Increment by the number of texels in the depth dimension
    dst_offset.data[2u] += v_self.extents().data[2u];
  }

  return convert(v_output);
}

Tensor cat_feature(
    const MaterializedITensorListRef& tensors,
    vTensor& v_output) {
  api::Context* const context = api::context();

  // Determine the channels of the output tensor
  uint32_t ch_total = 0;
  for (const at::Tensor& tensor : tensors) {
    ch_total += get_dim<Dim4D::Channel>(tensor);
  }

  // Running counter of the number of channels already appended.
  uint32_t ch_current = 0;
  for (const at::Tensor& tensor : tensors) {
    const Tensor self = tensor.is_vulkan() ? tensor : tensor.vulkan();
    const vTensor& v_self = convert(self);

    // Determine the number of channel texels that will be modified by
    // appending this input tensor
    uint32_t start_ch4 = ch_current / 4;

    uint32_t end_ch4 =
        api::utils::div_up(ch_current + get_dim<Dim4D::Channel>(v_self), 4u);

    uint32_t ch4_range = end_ch4 - start_ch4;
    uint32_t nc4_range = ch4_range * get_dim<Dim4D::Batch>(v_self);

    const struct Block final {
      ivec3 outExtents;
      int32_t fill0;
      ivec3 inExtents;
      int32_t fill1;
      uvec2 outChInfo;
      uvec2 inChInfo;
      uvec4 appendedChInfo;
    } block{
        api::utils::make_ivec3(v_output.extents()),
        0,
        api::utils::make_ivec3(v_self.extents()),
        0,
        {
            ch_total,
            api::utils::div_up(ch_total, 4u),
        },
        {
            get_dim<Dim4D::Channel>(v_self),
            api::utils::align_up(get_dim<Dim4D::Channel>(v_self), 4u),
        },
        {
            ch_current,
            start_ch4,
            ch4_range,
            0u,
        },
    };

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};

    context->submit_compute_job(
        // shader descriptor
        VK_KERNEL(cat_feature),
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {
            get_dim<Dim4D::Width>(v_output),
            get_dim<Dim4D::Height>(v_output),
            nc4_range,
        },
        // local work group size
        adaptive_work_group_size(v_self.extents()),
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
        v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());

    ch_current += get_dim<Dim4D::Channel>(v_self);
  }

  return convert(v_output);
}

Tensor cat_feature_mult4ch(
    const MaterializedITensorListRef& tensors,
    vTensor& v_output) {
  api::Context* const context = api::context();

  int64_t depth_size_allprior = 0;
  int64_t ch_interval = 0;
  for (const at::Tensor& tensor : tensors) {
    ch_interval += get_dim<Dim4D::Channel>(tensor);
  }
  const int64_t depth_interval = ch_interval / 4;

  uvec3 src_offset{};
  uvec3 dst_offset{};

  for (const at::Tensor& tensor_arg : tensors) {
    const Tensor tensor =
        tensor_arg.is_vulkan() ? tensor_arg : tensor_arg.vulkan();
    const vTensor& v_self = convert(tensor);

    const uint32_t depth_slice =
        safe_downcast<uint32_t>(get_dim<Dim4D::Channel>(tensor) / 4);

    uvec3 copy_extents{
        v_self.extents().data[0u], v_self.extents().data[1u], depth_slice};

    for (const auto b : c10::irange(get_dim<Dim4D::Batch>(tensor))) {
      src_offset.data[2u] = safe_downcast<uint32_t>(depth_slice * b);
      dst_offset.data[2u] =
          depth_size_allprior + safe_downcast<uint32_t>(depth_interval * b);

      api::PipelineBarrier pipeline_barrier{};

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
          copy_extents,
          src_offset,
          dst_offset,
          // fence handle
          VK_NULL_HANDLE);
    }

    depth_size_allprior += depth_slice;
  }

  return convert(v_output);
}

Tensor cat_width(const MaterializedITensorListRef& tensors, vTensor& v_output) {
  // TORCH_CHECK(false, "Vulkan cat not implemented for width dimension!");
  api::Context* const context = api::context();

  uvec3 src_offset{};
  uvec3 dst_offset{};

  for (const at::Tensor& tensor : tensors) {
    const Tensor self = tensor.is_vulkan() ? tensor : tensor.vulkan();
    const vTensor& v_self = convert(self);

    api::PipelineBarrier pipeline_barrier{};

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

    // Increment by width
    dst_offset.data[0u] += v_self.extents().data[0u];
  }

  return convert(v_output);
}

Tensor cat_height(
    const MaterializedITensorListRef& tensors,
    vTensor& v_output) {
  api::Context* const context = api::context();

  uvec3 src_offset{};
  uvec3 dst_offset{};

  for (const at::Tensor& tensor : tensors) {
    const Tensor self = tensor.is_vulkan() ? tensor : tensor.vulkan();
    const vTensor& v_self = convert(self);

    api::PipelineBarrier pipeline_barrier{};

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

    // Increment by height
    dst_offset.data[1u] += v_self.extents().data[1u];
  }

  return convert(v_output);
}

Tensor cat(const at::ITensorListRef& tensors, const int64_t in_dim) {
  api::AllocationScope allocation_scope("cat");

  TORCH_CHECK(!tensors.empty(), "Vulkan cat expects at least one tensor");
  auto materialized = tensors.materialize();
  TORCH_INTERNAL_ASSERT(!materialized.empty(), "Accessing empty array");
  const at::Tensor& tensor = materialized[0];
  auto ndim = safe_downcast<uint32_t>(tensor.dim());
  const int64_t dim = normalize_dim(in_dim, ndim);
  if (!c10::isFloatingType(tensor.scalar_type())) {
    return cat_cpu_fallback(materialized, in_dim);
  }
  int64_t cat_dim_size = 0;
  bool is_mult4ch = true;
  for (const at::Tensor& t : materialized) {
    TORCH_INTERNAL_ASSERT(
        t.dim() <= 4,
        "Vulkan cat expects inputs to have at most 4 dimensions, but got ",
        t.dim(),
        "d");

    if (ndim < 3 || get_dim<Dim4D::Channel>(t) % 4 != 0) {
      is_mult4ch = false;
    }

    for (const auto d : c10::irange(ndim)) {
      if (d == dim) {
        continue;
      }
      TORCH_INTERNAL_ASSERT(
          t.size(d) == tensor.size(d),
          "Vulkan cat inputs must have matching sizes except concatenated dimension");
    }
    cat_dim_size += t.size(dim);
  }

  auto result_size = tensor.sizes().vec();
  TORCH_INTERNAL_ASSERT(!result_size.empty(), "Accessing empty array");
  result_size[dim] = cat_dim_size;

  if (can_use_buffer_cat_fast_path(materialized, dim)) {
    return cat_buffer_direct(materialized, dim, result_size);
  }
  if (cat_requires_cpu_fallback(materialized)) {
    return cat_cpu_fallback(materialized, in_dim);
  }

  vTensor v_output{
      api::context(), result_size, convert_dtype(tensor.scalar_type())};

  if (dim == ndim - 1) {
    return cat_width(materialized, v_output);
  }
  if (dim == ndim - 2) {
    return cat_height(materialized, v_output);
  } else if (dim == ndim - 3) {
    if (is_mult4ch) {
      return cat_feature_mult4ch(materialized, v_output);
    }
    return cat_feature(materialized, v_output);
  }
  return cat_batch(materialized, v_output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::cat"), TORCH_FN(cat));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
