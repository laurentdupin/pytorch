#include <ATen/ATen.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/vulkan/Context.h>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace {

constexpr int64_t kLargeFloatMatrixNumelThreshold = 1 << 20;

const std::string& copy_sync_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_COPY_SYNC_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool copy_sync_logging_enabled() {
  return !copy_sync_log_path().empty();
}

bool should_force_buffer_storage_for_to_vulkan(const Tensor& src) {
  // Large 2D float tensors are typically inference weights such as embedding,
  // lm_head, and linear matrices. Keeping them buffer-backed avoids expensive
  // texture residency overhead; Vulkan linear/embedding paths can prepack or
  // gather from them later as needed.
  return src.scalar_type() == at::kFloat && src.dim() == 2 &&
      src.numel() >= kLargeFloatMatrixNumelThreshold;
}

const char* storage_type_name(const api::StorageType storage_type) {
  switch (storage_type) {
    case api::StorageType::TEXTURE_3D:
      return "TEXTURE_3D";
    case api::StorageType::TEXTURE_2D:
      return "TEXTURE_2D";
    case api::StorageType::BUFFER:
      return "BUFFER";
    case api::StorageType::UNKNOWN:
      return "UNKNOWN";
  }
  return "UNKNOWN";
}

std::string format_sizes(const std::vector<int64_t>& sizes) {
  std::ostringstream stream;
  stream << "[";
  for (size_t idx = 0; idx < sizes.size(); ++idx) {
    if (idx > 0) {
      stream << ",";
    }
    stream << sizes[idx];
  }
  stream << "]";
  return stream.str();
}

void log_copy_sync_event(
    const char* kind,
    const vTensor& tensor,
    const bool direct_buffer_layout) {
  if (!copy_sync_logging_enabled()) {
    return;
  }

  std::ofstream out(copy_sync_log_path(), std::ios::app);
  out << "kind=" << kind
      << " caller=" << api::current_allocation_label()
      << " storage=" << storage_type_name(tensor.storage_type())
      << " direct_buffer=" << (direct_buffer_layout ? 1 : 0)
      << " logical_bytes=" << tensor.nbytes()
      << " gpu_bytes=" << tensor.gpu_nbytes()
      << " sizes=" << format_sizes(tensor.sizes()) << '\n';
}

c10::MemoryFormat memory_format_for_buffer_layout(
    const api::GPUMemoryLayout memory_layout) {
  switch (memory_layout) {
    case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
      return c10::MemoryFormat::Contiguous;
    case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
      return c10::MemoryFormat::ChannelsLast;
    default:
      VK_THROW("Unsupported buffer memory layout");
  }
}

std::vector<int64_t> calc_logical_contiguous_strides(
    const IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (int idx = static_cast<int>(sizes.size()) - 2; idx >= 0; --idx) {
    strides[idx] = strides[idx + 1] * std::max<int64_t>(sizes[idx + 1], 1);
  }
  return strides;
}

template <typename T>
void pack_logical_tensor_to_buffer_mapping(
    const Tensor& src,
    api::MemoryMap& dst_mapping,
    const IntArrayRef physical_strides,
    const int64_t physical_numel,
    const int64_t storage_offset) {
  T* const dst = dst_mapping.data<T>();
  std::fill(dst, dst + physical_numel, T{});

  const T* const src_ptr = src.const_data_ptr<T>();
  const std::vector<int64_t> logical_strides =
      calc_logical_contiguous_strides(src.sizes());

  for (int64_t logical_idx = 0; logical_idx < src.numel(); ++logical_idx) {
    int64_t remaining = logical_idx;
    int64_t physical_idx = storage_offset;
    for (int64_t dim = 0; dim < src.dim(); ++dim) {
      const int64_t stride = logical_strides[dim];
      const int64_t coord = stride == 0 ? 0 : remaining / stride;
      remaining = stride == 0 ? 0 : remaining % stride;
      physical_idx += coord * physical_strides[dim];
    }
    dst[physical_idx] = src_ptr[logical_idx];
  }
}

template <typename T>
void unpack_buffer_mapping_to_logical_tensor(
    api::MemoryMap& src_mapping,
    Tensor& dst,
    const IntArrayRef physical_strides,
    const int64_t storage_offset) {
  const T* const src_ptr = src_mapping.data<T>();
  T* const dst_ptr = dst.mutable_data_ptr<T>();
  const std::vector<int64_t> logical_strides =
      calc_logical_contiguous_strides(dst.sizes());

  for (int64_t logical_idx = 0; logical_idx < dst.numel(); ++logical_idx) {
    int64_t remaining = logical_idx;
    int64_t physical_idx = storage_offset;
    for (int64_t dim = 0; dim < dst.dim(); ++dim) {
      const int64_t stride = logical_strides[dim];
      const int64_t coord = stride == 0 ? 0 : remaining / stride;
      remaining = stride == 0 ? 0 : remaining % stride;
      physical_idx += coord * physical_strides[dim];
    }
    dst_ptr[logical_idx] = src_ptr[physical_idx];
  }
}

void pack_logical_tensor_to_buffer_mapping_dispatch(
    const Tensor& src,
    api::MemoryMap& dst_mapping,
    const IntArrayRef physical_strides,
    const int64_t physical_numel,
    const int64_t storage_offset) {
  switch (src.scalar_type()) {
    case at::kFloat:
      pack_logical_tensor_to_buffer_mapping<float>(
          src,
          dst_mapping,
          physical_strides,
          physical_numel,
          storage_offset);
      return;
    case at::kHalf:
      pack_logical_tensor_to_buffer_mapping<c10::Half>(
          src,
          dst_mapping,
          physical_strides,
          physical_numel,
          storage_offset);
      return;
    case at::kBFloat16:
      pack_logical_tensor_to_buffer_mapping<c10::BFloat16>(
          src,
          dst_mapping,
          physical_strides,
          physical_numel,
          storage_offset);
      return;
    case at::kByte:
      pack_logical_tensor_to_buffer_mapping<uint8_t>(
          src,
          dst_mapping,
          physical_strides,
          physical_numel,
          storage_offset);
      return;
    case at::kChar:
      pack_logical_tensor_to_buffer_mapping<int8_t>(
          src,
          dst_mapping,
          physical_strides,
          physical_numel,
          storage_offset);
      return;
    case at::kInt:
      pack_logical_tensor_to_buffer_mapping<int32_t>(
          src,
          dst_mapping,
          physical_strides,
          physical_numel,
          storage_offset);
      return;
    case at::kLong:
      pack_logical_tensor_to_buffer_mapping<int64_t>(
          src,
          dst_mapping,
          physical_strides,
          physical_numel,
          storage_offset);
      return;
    case at::kBool:
      pack_logical_tensor_to_buffer_mapping<bool>(
          src,
          dst_mapping,
          physical_strides,
          physical_numel,
          storage_offset);
      return;
    default:
      TORCH_CHECK(
          false,
          "Unsupported scalar type for Vulkan buffer pack: ",
          src.scalar_type());
  }
}

void unpack_buffer_mapping_to_logical_tensor_dispatch(
    api::MemoryMap& src_mapping,
    Tensor& dst,
    const IntArrayRef physical_strides,
    const int64_t storage_offset) {
  switch (dst.scalar_type()) {
    case at::kFloat:
      unpack_buffer_mapping_to_logical_tensor<float>(
          src_mapping, dst, physical_strides, storage_offset);
      return;
    case at::kHalf:
      unpack_buffer_mapping_to_logical_tensor<c10::Half>(
          src_mapping, dst, physical_strides, storage_offset);
      return;
    case at::kBFloat16:
      unpack_buffer_mapping_to_logical_tensor<c10::BFloat16>(
          src_mapping, dst, physical_strides, storage_offset);
      return;
    case at::kByte:
      unpack_buffer_mapping_to_logical_tensor<uint8_t>(
          src_mapping, dst, physical_strides, storage_offset);
      return;
    case at::kChar:
      unpack_buffer_mapping_to_logical_tensor<int8_t>(
          src_mapping, dst, physical_strides, storage_offset);
      return;
    case at::kInt:
      unpack_buffer_mapping_to_logical_tensor<int32_t>(
          src_mapping, dst, physical_strides, storage_offset);
      return;
    case at::kLong:
      unpack_buffer_mapping_to_logical_tensor<int64_t>(
          src_mapping, dst, physical_strides, storage_offset);
      return;
    case at::kBool:
      unpack_buffer_mapping_to_logical_tensor<bool>(
          src_mapping, dst, physical_strides, storage_offset);
      return;
    default:
      TORCH_CHECK(
          false,
          "Unsupported scalar type for Vulkan buffer unpack: ",
          dst.scalar_type());
  }
}

void copy_staging_buffer_to_vtensor_buffer(
    api::Context* const context,
    api::StorageBuffer& staging,
    vTensor& dst,
    const VkFence fence_handle) {
  api::PipelineBarrier pipeline_barrier{};
  context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
      pipeline_barrier,
      staging.buffer(),
      dst.buffer(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::WRITE),
      {api::utils::safe_downcast<uint32_t>(staging.buffer().mem_size()), 0u, 0u},
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      fence_handle);
}

bool copy_vtensor_buffer_to_staging(
    api::Context* const context,
    vTensor& src,
    api::StorageBuffer& staging,
    const VkFence fence_handle) {
  const bool use_float_pack_shader =
      src.dtype() == api::kFloat && src.sizes().size() <= 4 &&
      src.storage_type() == api::StorageType::BUFFER &&
      src.last_write_was_compute();
  if (use_float_pack_shader) {
    return utils::pack_vtensor_to_staging(
        src, staging.buffer(), fence_handle);
  }

  api::PipelineBarrier pipeline_barrier{};
  return context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
      pipeline_barrier,
      src.buffer(pipeline_barrier, api::PipelineStage::TRANSFER),
      staging.buffer(),
      {api::utils::safe_downcast<uint32_t>(staging.buffer().mem_size()), 0u, 0u},
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      fence_handle);
}

} // namespace

//
// Utility functions for memcpy
//

void memcpy_to_mapping(const Tensor& src, api::MemoryMap& dst_mapping) {
  if (src.dtype() == at::kFloat) {
    memcpy_to_mapping_impl<float>(src, dst_mapping);
  } else if (src.dtype() == at::kHalf) {
    memcpy_to_mapping_impl<c10::Half>(src, dst_mapping);
  } else if (src.dtype() == at::kBFloat16) {
    memcpy_to_mapping_impl<c10::BFloat16>(src, dst_mapping);
  } else if (src.dtype() == at::kByte) {
    memcpy_to_mapping_impl<uint8_t>(src, dst_mapping);
  } else if (src.dtype() == at::kChar) {
    memcpy_to_mapping_impl<int8_t>(src, dst_mapping);
  } else if (src.dtype() == at::kInt) {
    memcpy_to_mapping_impl<int32_t>(src, dst_mapping);
  } else if (src.dtype() == at::kLong) {
    memcpy_to_mapping_impl<int64_t>(src, dst_mapping);
  } else if (src.dtype() == c10::kQUInt8) {
    memcpy_to_mapping_impl<c10::quint8>(src, dst_mapping);
  } else if (src.dtype() == c10::kQInt8) {
    memcpy_to_mapping_impl<c10::qint8>(src, dst_mapping);
  } else if (src.dtype() == c10::kQInt32) {
    memcpy_to_mapping_impl<c10::qint32>(src, dst_mapping);
  } else if (src.dtype() == c10::kBool) {
    memcpy_to_mapping_uint8(src, dst_mapping);
  } else {
    TORCH_CHECK(
        false,
        "Invalid Data Type: expected c10::kQInt32, c10::kQInt8, c10::kQUInt8,",
        " c10::kBool, at::kByte, at::kChar, at::kInt, at::kLong, at::kHalf,",
        " at::kBFloat16, or at::kFloat but got ",
        src.dtype());
  }
}

void memcpy_from_mapping(api::MemoryMap& src_mapping, Tensor& dst) {
  if (dst.dtype() == at::kFloat) {
    memcpy_from_mapping_impl<float>(src_mapping, dst);
  } else if (dst.dtype() == at::kHalf) {
    memcpy_from_mapping_impl<c10::Half>(src_mapping, dst);
  } else if (dst.dtype() == at::kBFloat16) {
    memcpy_from_mapping_impl<c10::BFloat16>(src_mapping, dst);
  } else if (dst.dtype() == at::kByte) {
    memcpy_from_mapping_impl<uint8_t>(src_mapping, dst);
  } else if (dst.dtype() == at::kChar) {
    memcpy_from_mapping_impl<int8_t>(src_mapping, dst);
  } else if (dst.dtype() == at::kInt) {
    memcpy_from_mapping_impl<int32_t>(src_mapping, dst);
  } else if (dst.dtype() == at::kLong) {
    memcpy_from_mapping_impl<int64_t>(src_mapping, dst);
  } else if (dst.dtype() == c10::kQUInt8) {
    memcpy_from_mapping_impl<c10::quint8>(src_mapping, dst);
  } else if (dst.dtype() == c10::kQInt8) {
    memcpy_from_mapping_impl<c10::qint8>(src_mapping, dst);
  } else if (dst.dtype() == c10::kQInt32) {
    memcpy_from_mapping_impl<c10::qint32>(src_mapping, dst);
  } else if (dst.dtype() == c10::kBool) {
    memcpy_from_mapping_bool(src_mapping, dst);
  } else {
    TORCH_CHECK(
        false,
        "Invalid Data Type: expected c10::kQInt32, c10::kQInt8, c10::kQUInt8,",
        " c10::kBool, at::kByte, at::kChar, at::kInt, at::kLong, at::kHalf,",
        " at::kBFloat16, or at::kFloat but got ",
        dst.dtype());
  }
}

//
// CPU <-> GPU copy implementations (these functions use Transfer commands)
//

void transfer_cpu_to_vulkan(const Tensor& src, vTensor& v_dst) {
  api::Context* const context = api::context();

  // Convert to dtype corresponding to the image format of the texture to
  // ensure that byte alignment is consistent when copying. In some cases
  // a 16 bit format will be used for at::kFloat.
  Tensor src_nc4hw =
      utils::nchw_to_nc4hw(src).to(convert_dtype(v_dst.texture_dtype()));

  api::StorageBuffer staging(context, v_dst.texture_dtype(), v_dst.gpu_numel());
  // Copy data into the staging buffer
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
    mapping.invalidate();

    memcpy_to_mapping(src_nc4hw, mapping);
  }

  api::PipelineBarrier pipeline_barrier{};
  utils::copy_buffer_to_vtensor(staging.buffer(), v_dst, pipeline_barrier);
}

void transfer_vulkan_to_cpu(vTensor& v_src, Tensor& dst) {
  api::Context* const context = api::context();

  // Temporary tensor to receive copied NC4HW data
  at::Tensor dst_tmp = utils::create_staging_tensor(v_src);

  api::StorageBuffer staging(context, v_src.texture_dtype(), v_src.gpu_numel());

  api::VulkanFence fence = context->fences().get_fence();

  {
    // Refer to comment in submit_compute_job. When syncing with the GPU, the
    // context must not allow other threads to record dispatches into it between
    // between calling vkQueueSubmit and flushing the context. Therefore,
    // cmd_mutex_ must be manually managed by the calling thread.
    std::unique_lock<std::mutex> context_lock(context->dispatch_lock());

    api::PipelineBarrier pipeline_barrier{};
    utils::copy_vtensor_to_buffer(
        v_src, staging.buffer(), pipeline_barrier, fence.get_submit_handle());

    fence.wait();

    log_copy_sync_event("transfer_vulkan_to_cpu", v_src, false);
    context->flush_after_fence_wait();
    // cmd_mutex_ will be released when exiting this scope.
  }

  // Copy data from buffer back to CPU tensor.
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
    mapping.invalidate();

    memcpy_from_mapping(mapping, dst_tmp);
  }

  context->fences().return_fence(fence);

  dst = utils::nc4hw_to_nchw(dst_tmp, v_src.sizes())
            .to(convert_dtype(v_src.dtype()));
}

static void transfer_vulkan_to_vulkan(vTensor& src, vTensor& dst) {
  api::Context* const context = api::context();

  api::PipelineBarrier pipeline_barrier{};

  context->submit_copy<api::VulkanImage, api::VulkanImage>(
      // pipeline barrier
      pipeline_barrier,
      // images
      src.image(pipeline_barrier, api::PipelineStage::TRANSFER),
      dst.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::WRITE),
      // copy details
      src.extents(),
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      VK_NULL_HANDLE);
}

//
// CPU <-> GPU copy implementations (these functions use compute shaders)
//

void pack_cpu_to_vulkan(const Tensor& src, vTensor& dst) {
  api::Context* const context = api::context();

  if (dst.storage_type() == api::StorageType::BUFFER) {
    const c10::MemoryFormat target_memory_format =
        memory_format_for_buffer_layout(dst.gpu_memory_layout());
    Tensor src_contig = src.contiguous(target_memory_format);
    api::StorageBuffer staging(
        context,
        convert_dtype(src_contig.scalar_type()),
        dst.gpu_numel());
    {
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
      if (dst.has_direct_buffer_layout()) {
        memcpy_to_mapping(src_contig, mapping);
      } else {
        pack_logical_tensor_to_buffer_mapping_dispatch(
            src_contig,
            mapping,
            dst.gpu_strides(),
            dst.gpu_numel(),
            dst.storage_offset());
      }
    }
    copy_staging_buffer_to_vtensor_buffer(context, staging, dst, VK_NULL_HANDLE);
    return;
  }

  // Ensure that src is contiguous in its memory format
  Tensor src_contig = src.contiguous(src.suggest_memory_format());

  // Note that the float data type has been enforced for the storage buffer
  // below. The reason for this is that the nchw_to_image and image_to_nchw
  // shaders which perform the transfer to/from an image texture expect a buffer
  // of floats as input. GLSL/Vulkan does not natively support 16 bit arithmetic
  // types, so for now storage buffers created for compute shaders must define
  // floats as their base data type.
  api::StorageBuffer staging(context, api::kFloat, dst.gpu_numel());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
    std::memset(mapping.data<uint8_t>(), 0, mapping.nbytes());

    // If the dtype() of src is at::kHalf, then first convert it to 32 bit
    // float. This is required since the nchw_to_image shader uses a float
    // buffer as input (note that at::kFloat is used to create the StorageBuffer
    // above).
    if (src.dtype() == at::kHalf) {
      memcpy_to_mapping(src_contig.to(at::kFloat), mapping);
    } else {
      memcpy_to_mapping(src_contig, mapping);
    }
  }
  utils::pack_staging_to_vtensor(staging.buffer(), dst);
}

void pack_vulkan_to_cpu(vTensor& src, Tensor& dst) {
  TORCH_CHECK(
      !src.is_quantized(),
      "Copy of vulkan quantized tensors to cpu is currently disabled!");
  api::Context* const context = api::context();

  if (src.storage_type() == api::StorageType::BUFFER) {
    const bool shader_packed_buffer =
        src.dtype() == api::kFloat && src.sizes().size() <= 4 &&
        src.storage_type() == api::StorageType::BUFFER &&
        src.last_write_was_compute();
    const int64_t staging_length =
        shader_packed_buffer
        ? api::utils::safe_downcast<int64_t>(src.numel())
        : src.has_direct_buffer_layout()
        ? api::utils::safe_downcast<int64_t>(src.gpu_numel())
        : src.buffer_length();
    api::StorageBuffer staging(context, src.dtype(), staging_length);
    api::VulkanFence fence = context->fences().get_fence();

    {
      std::unique_lock<std::mutex> context_lock(context->dispatch_lock());
      const bool submitted_to_gpu = copy_vtensor_buffer_to_staging(
          context, src, staging, fence.get_submit_handle());
      if (submitted_to_gpu) {
        fence.wait();
        log_copy_sync_event(
            "pack_vulkan_to_cpu_buffer", src, false);
        context->flush_after_fence_wait();
      } else {
        context->flush();
      }
    }

    Tensor dst_tmp = at::empty(
        src.sizes(),
        at::device(at::kCPU).dtype(convert_dtype(src.dtype())));
    if (src.has_direct_buffer_layout()) {
      dst_tmp = dst_tmp.to(memory_format_for_buffer_layout(src.gpu_memory_layout()));
    }

    {
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
      mapping.invalidate();
      if (src.has_direct_buffer_layout() || shader_packed_buffer) {
        memcpy_from_mapping(mapping, dst_tmp);
      } else {
        unpack_buffer_mapping_to_logical_tensor_dispatch(
            mapping, dst_tmp, src.gpu_strides(), src.storage_offset());
      }
    }

    context->fences().return_fence(fence);
    dst.copy_(dst_tmp);
    return;
  }

  // Refer to the comment in pack_cpu_to_vulkan for why at::kFloat is specified
  // for the storage buffer below.
  api::StorageBuffer staging(context, api::kFloat, src.gpu_numel());

  api::VulkanFence fence = context->fences().get_fence();

  {
    // Refer to comment in submit_compute_job. When syncing with the GPU, the
    // context must not allow other threads to record dispatches into it between
    // between calling vkQueueSubmit and flushing the context. Therefore,
    // cmd_mutex_ must be manually managed by the calling thread.
    std::unique_lock<std::mutex> context_lock(context->dispatch_lock());

    bool submitted_to_gpu = utils::pack_vtensor_to_staging(
        src, staging.buffer(), fence.get_submit_handle());

    // Only wait on the fence if work was actually submitted to the GPU.
    // Otherwise, it will hang indefinitely.
    if (submitted_to_gpu) {
      fence.wait();
      log_copy_sync_event("pack_vulkan_to_cpu_texture", src, false);
      context->flush_after_fence_wait();
    } else {
      context->flush();
    }
    // cmd_mutex_ will be released when exiting this scope.
  }

  // Copy data from buffer back to CPU tensor.
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
    mapping.invalidate();

    // If the dtype() of dst is at::kHalf, then copy the data into a float
    // version of it first, similar to pack_cpu_to_vulkan().
    if (dst.dtype() == at::kHalf) {
      Tensor dst_float = dst.to(at::kFloat);
      memcpy_from_mapping(mapping, dst_float);
      dst = dst_float.to(at::kHalf);
    } else {
      memcpy_from_mapping(mapping, dst);
    }
  }

  context->fences().return_fence(fence);
}

//
// Copy op implementations
//

Tensor& copy_(Tensor& dst, const Tensor& src) {
  // Check that sizes are equal
  TORCH_CHECK(
      dst.sizes() == src.sizes(), "Vulkan copy_: Tensor sizes are mismatched!");

  // X -> Vulkan
  if (at::kVulkan == dst.device().type()) {
    vTensor& v_self = convert(dst);

    // Vulkan -> Vulkan
    if (at::kVulkan == src.device().type()) {
      Tensor src_casted = src;
      if (src.scalar_type() != dst.scalar_type()) {
        src_casted = utils::cast_vulkan_tensor_dtype(src, dst.scalar_type());
      }

      vTensor& v_src = convert(src_casted);
      const bool can_direct_copy =
          v_src.dtype() == v_self.dtype() &&
          v_src.storage_type() != api::StorageType::BUFFER &&
          v_self.storage_type() != api::StorageType::BUFFER;
      if (can_direct_copy) {
        transfer_vulkan_to_vulkan(v_src, v_self);
      } else {
        c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
        c10::InferenceMode inference_mode_guard(false);
        Tensor cpu_src = from_vulkan(v_src);
        pack_cpu_to_vulkan(cpu_src, v_self);
      }
    }
    // CPU -> Vulkan
    else {
      Tensor cpu_src = src;
      if (cpu_src.scalar_type() != dst.scalar_type()) {
        cpu_src = cpu_src.to(dst.scalar_type());
      }
      pack_cpu_to_vulkan(cpu_src, v_self);
    }
  }
  // Vulkan -> X
  else if (at::kVulkan == src.device().type()) {
    vTensor& v_src = convert(src);

    // Vulkan -> CPU
    if (dst.device().is_cpu()) {
      pack_vulkan_to_cpu(v_src, dst);
    } else {
      TORCH_CHECK(false, "Unsupported!");
    }
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "Invalid code path taken! Either the source or the destination tensor "
        "was expected to be Vulkan a tensor!  Incorrect dispatch?");
  }

  return dst;
}

vTensor to_vulkan(at::Tensor& src, const api::StorageType storage_type) {
  TORCH_CHECK(
      src.device().type() == at::kCPU,
      "Vulkan to_vulkan(): input tensor must be a CPU tensor!")

  const api::StorageType resolved_storage_type =
      (api::requires_buffer_storage(convert_dtype(src.scalar_type()), src.dim()) ||
       should_force_buffer_storage_for_to_vulkan(src))
      ? api::StorageType::BUFFER
      : storage_type;

  vTensor v_ret{
      api::context(),
      src.sizes().vec(),
      convert_dtype(src.scalar_type()),
      resolved_storage_type,
      get_gpu_memory_layout(resolved_storage_type, src.suggest_memory_format()),
  };

  ops::pack_cpu_to_vulkan(src, v_ret);

  return v_ret;
}

at::Tensor to_vulkan_labeled(at::Tensor src, std::string label) {
  if (src.is_vulkan()) {
    return src;
  }
  TORCH_CHECK(
      src.device().type() == at::kCPU,
      "Vulkan to_vulkan_labeled(): input tensor must be a CPU or Vulkan tensor!");
  api::AllocationScope allocation_scope(std::move(label));
  vTensor v_ret = to_vulkan(src, api::StorageType::TEXTURE_3D);
  return convert(v_ret);
}

at::Tensor from_vulkan(vTensor& v_src) {
  at::TensorOptions opt(at::kCPU);
  opt = opt.dtype(convert_dtype(v_src.dtype()));

  c10::MemoryFormat v_src_memory_format = c10::MemoryFormat::Contiguous;

  switch (v_src.gpu_memory_layout()) {
    case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
      v_src_memory_format = c10::MemoryFormat::Contiguous;
      break;
    case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
      // ChannelsLast is only valid for rank-4 CPU tensors. Lower-rank Vulkan
      // tensors can still use channels-packed GPU layout, but the CPU staging
      // tensor for readback/conversion must stay contiguous.
      v_src_memory_format =
          v_src.sizes().size() == 4 ? c10::MemoryFormat::ChannelsLast
                                    : c10::MemoryFormat::Contiguous;
      break;
    default:
      TORCH_CHECK(false, "No corresponding memory format");
  }

  at::Tensor ret = at::empty(v_src.sizes(), opt).to(v_src_memory_format);
  ops::pack_vulkan_to_cpu(v_src, ret);
  return ret;
}

//
// VulkanImpl
//

struct VulkanImpl final : public at::vulkan::VulkanImplInterface {
  bool is_vulkan_available() const override {
    return api::available();
  }

  Tensor& vulkan_copy_(Tensor& self, const Tensor& src) const override {
    return vulkan::ops::copy_(self, src);
  }
};
static at::vulkan::VulkanImplRegistrar g_vulkan_impl(new VulkanImpl());

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
