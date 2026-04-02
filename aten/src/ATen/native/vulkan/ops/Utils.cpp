#include <ATen/native/vulkan/impl/Packing.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <cstdlib>
#include <fstream>
#include <sstream>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/zeros.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace utils {

using namespace api::utils;
namespace {

const std::string& materialize_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_MATERIALIZE_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool materialize_logging_enabled() {
  return !materialize_log_path().empty();
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

const char* memory_layout_name(const api::GPUMemoryLayout memory_layout) {
  switch (memory_layout) {
    case api::GPUMemoryLayout::TENSOR_WIDTH_PACKED:
      return "TENSOR_WIDTH_PACKED";
    case api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED:
      return "TENSOR_HEIGHT_PACKED";
    case api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED:
      return "TENSOR_CHANNELS_PACKED";
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

void append_materialize_log_line(const std::string& line) {
  if (!materialize_logging_enabled()) {
    return;
  }

  std::ofstream out(materialize_log_path(), std::ios::app);
  out << line << '\n';
}

void log_materialize_event(
    const char* kind,
    const vTensor& v_in,
    const api::StorageType dst_storage_type,
    const api::GPUMemoryLayout dst_memory_layout,
    const char* path) {
  if (!materialize_logging_enabled()) {
    return;
  }

  std::ostringstream stream;
  stream << "kind=" << kind
         << " caller=" << api::current_allocation_label()
         << " path=" << path
         << " src_storage=" << storage_type_name(v_in.storage_type())
         << " src_layout=" << memory_layout_name(v_in.gpu_memory_layout())
         << " dst_storage=" << storage_type_name(dst_storage_type)
         << " dst_layout=" << memory_layout_name(dst_memory_layout)
         << " direct_buffer=" << (v_in.has_direct_buffer_layout() ? 1 : 0)
         << " storage_offset=" << v_in.storage_offset()
         << " logical_bytes=" << v_in.nbytes()
         << " gpu_bytes=" << v_in.gpu_nbytes()
         << " sizes=" << format_sizes(v_in.sizes());
  append_materialize_log_line(stream.str());
}

} // namespace

/*
 * This function formats an input tensor in NCHW layout to NC4HW layout such
 * that the buffer of the formatted tensor can be directly copied into a GPU
 * texture. Conceptually, the formatting can be achieved via the following
 * steps:
 *
 * 1. Given that the src tensor has size {N,C,H,W}
 *
 * 2. Combine the batch and channel dims by reshaping to {N*C, H, W}
 *
 * 3. Determine the amount of padding to add: determine how many channels to add
 *    in order to align N*C to the next multiple of 4
 *
 * 4. Add padding to the tensor so that the batch-channel dimension is a
 *    multiple of four; the shape of the tensor is now {NC_aligned, H, W}
 *
 * 5. Split the batch-channel dimension into groups of 4 by reshaping the tensor
 *    to size {NC_aligned/4, 4, H, W}
 *
 * 6. The groups of 4 channels (dim 1) should be contiguous. Therefore, permute
 *    the dims of the tensor in the order {0, 2, 3, 1}
 *
 * 7. Finally, return a contiguous version of the tensor. The final shape of the
 *    tensor would be {NC_aligned/4, H, W, 4}
 */
Tensor nchw_to_nc4hw(const Tensor& src) {
  uint32_t N = get_dim<Dim4D::Batch>(src.sizes());
  uint32_t C = get_dim<Dim4D::Channel>(src.sizes());
  uint32_t H = get_dim<Dim4D::Height>(src.sizes());
  uint32_t W = get_dim<Dim4D::Width>(src.sizes());

  uint32_t C_aligned = api::utils::align_up(C, 4u);
  uint32_t NC4 = (N * C_aligned) / 4;

  // Add padding to the tensor so that the channel dim is a multiple of 4
  Tensor padding = at::zeros({N, C_aligned - C, H, W}, src.options());
  Tensor src_padded = at::cat({src.reshape({N, C, H, W}), padding}, 1);
  // Reshape to group channels into groups of 4 and permute so that the groups
  // are in the first dimension so that they are contiguous
  Tensor src_NC4HW = src_padded.reshape({NC4, 4, H, W}).permute({0, 2, 3, 1});

  // Return a contiguous version of the tensor
  return src_NC4HW.contiguous();
}

/*
 * Creates a staging tensor into which texture data, which will be in NC4HW
 * format, can be copied directly. The shape of the staging tensor will be the
 * same as the tensor produced by a call to format_src_tensor().
 */
Tensor create_staging_tensor(const vTensor& v_in) {
  uint32_t N = get_dim<Dim4D::Batch>(v_in.sizes());
  uint32_t C = get_dim<Dim4D::Channel>(v_in.sizes());
  uint32_t H = get_dim<Dim4D::Height>(v_in.sizes());
  uint32_t W = get_dim<Dim4D::Width>(v_in.sizes());

  uint32_t NC4 = N * api::utils::div_up(C, 4u);

  // Note that the dtype corresponding with the texture format of the vTensor is
  // used instead of options().dtype(). This is to ensure the number of bytes in
  // the staging tensor matches the number of bytes in the image texture. Refer
  // to comments for api::vk_format()
  return at::empty(
      {NC4, H, W, 4},
      at::device(at::kCPU).dtype(convert_dtype(v_in.texture_dtype())));
}

/*
 * After copying texture data, which will be in NC4HW format, to a staging
 * tensor created in create_staging_tensor(), this function reformats the tensor
 * to NCHW format. It essentially reverses the transformations made by
 * format_src_tensor().
 *
 * Note that the sizes of the original tensor must be passed in to fully restore
 * the properties of the original tensor.
 */
Tensor nc4hw_to_nchw(const Tensor& t_in, IntArrayRef sizes) {
  uint32_t N = get_dim<Dim4D::Batch>(sizes);
  uint32_t C = get_dim<Dim4D::Channel>(sizes);
  uint32_t H = get_dim<Dim4D::Height>(sizes);
  uint32_t W = get_dim<Dim4D::Width>(sizes);

  uint32_t C_aligned = api::utils::align_up(C, 4u);

  // Undo the permute step and channel grouping step
  Tensor t_in_padded = t_in.permute({0, 3, 1, 2}).reshape({N, C_aligned, H, W});
  // Remove the padding channels
  Tensor t_in_shaved =
      at::narrow(t_in_padded, /*dim=*/1, /*start*/ 0, /*end*/ C);

  // Reshape to original sizing and dtype and return a contiguous Tensor
  return t_in_shaved.reshape(sizes).contiguous();
}

bool supports_buffer_view_fast_path(const vTensor& v_in) {
  return !v_in.is_quantized() && v_in.dtype() == api::kFloat &&
      v_in.sizes().size() <= 4;
}

std::string describe_buffer_view_fast_path_failure(const vTensor& v_in) {
  std::ostringstream stream;
  stream
      << "Vulkan texture materialization from buffer views currently only "
      << "supports non-quantized float tensors with up to 4 dimensions"
      << " (caller=" << api::current_allocation_label()
      << ", sizes=" << format_sizes(v_in.sizes())
      << ", ndim=" << v_in.sizes().size()
      << ", dtype=" << static_cast<int>(v_in.dtype())
      << ", quantized=" << (v_in.is_quantized() ? 1 : 0)
      << ", storage=" << storage_type_name(v_in.storage_type())
      << ", layout=" << memory_layout_name(v_in.gpu_memory_layout())
      << ", direct_buffer=" << (v_in.has_direct_buffer_layout() ? 1 : 0)
      << ")";
  return stream.str();
}

vTensor materialize_to_contiguous_buffer(
    const vTensor& v_in,
    api::GPUMemoryLayout memory_layout) {
  TORCH_CHECK(
      supports_buffer_view_fast_path(v_in),
      describe_buffer_view_fast_path_failure(v_in));

  if (
      v_in.storage_type() == api::StorageType::BUFFER &&
      v_in.gpu_memory_layout() == memory_layout &&
      v_in.has_direct_buffer_layout()) {
    return v_in;
  }

  log_materialize_event(
      "materialize_to_contiguous_buffer",
      v_in,
      api::StorageType::BUFFER,
      memory_layout,
      "materialize");

  api::Context* const context = api::context();
  api::StorageBuffer staging(context, v_in.dtype(), v_in.numel());
  vTensor v_src = v_in;
  pack_vtensor_to_staging(v_src, staging.buffer());

  vTensor v_out{
      context,
      v_in.sizes(),
      v_in.dtype(),
      api::StorageType::BUFFER,
      memory_layout,
  };
  api::PipelineBarrier pipeline_barrier{};
  add_buffer_barrier(
      pipeline_barrier,
      staging.buffer(),
      api::PipelineStage::COMPUTE | api::PipelineStage::TRANSFER,
      api::MemoryAccessType::WRITE,
      api::PipelineStage::COMPUTE | api::PipelineStage::TRANSFER,
      api::MemoryAccessType::READ);
  pack_buffer_to_vtensor(staging.buffer(), v_out, pipeline_barrier);
  return v_out;
}

Tensor ensure_texture_storage(
    const Tensor& input_arg,
    api::GPUMemoryLayout memory_layout,
    api::StorageType storage_type) {
  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  vTensor v_input = convert(input);

  if (
      v_input.storage_type() == storage_type &&
      v_input.gpu_memory_layout() == memory_layout) {
    return input;
  }

  if (
      v_input.storage_type() != api::StorageType::BUFFER &&
      v_input.storage_type() == api::StorageType::TEXTURE_3D) {
    if (
        v_input.gpu_memory_layout() ==
        api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED) {
      if (memory_layout == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED) {
        log_materialize_event(
            "ensure_texture_storage",
            v_input,
            storage_type,
            memory_layout,
            "image_layout_convert_width");
        return convert(
            packing::convert_image_channels_packed_to_width_packed(v_input));
      }
      if (memory_layout == api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED) {
        log_materialize_event(
            "ensure_texture_storage",
            v_input,
            storage_type,
            memory_layout,
            "image_layout_convert_height");
        return convert(
            packing::convert_image_channels_packed_to_height_packed(v_input));
      }
    }
  }

  TORCH_CHECK(
      supports_buffer_view_fast_path(v_input),
      describe_buffer_view_fast_path_failure(v_input));

  api::Context* const context = api::context();
  const bool direct_buffer_path =
      v_input.storage_type() == api::StorageType::BUFFER &&
      v_input.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
      v_input.has_direct_buffer_layout();
  vTensor v_buffer =
      direct_buffer_path
      ? v_input
      : materialize_to_contiguous_buffer(
            v_input, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  log_materialize_event(
      "ensure_texture_storage",
      v_input,
      storage_type,
      memory_layout,
      direct_buffer_path ? "direct_buffer_to_texture"
                         : "buffer_materialize_to_texture");

  vTensor v_out{
      context,
      v_input.sizes(),
      v_input.dtype(),
      storage_type,
      memory_layout,
  };
  api::PipelineBarrier pipeline_barrier{};
  api::VulkanBuffer& src_buffer =
      v_buffer.buffer(pipeline_barrier, api::PipelineStage::COMPUTE);
  pack_buffer_to_vtensor(src_buffer, v_out, pipeline_barrier);
  return convert(v_out);
}

Tensor upcast_bfloat16_buffer_to_float(const Tensor& input) {
  TORCH_CHECK(input.is_vulkan(), "Input must be a Vulkan tensor");
  vTensor v_input = convert(input);
  TORCH_CHECK(
      v_input.storage_type() == api::StorageType::BUFFER,
      "BF16 buffer upcast requires a buffer-backed Vulkan tensor");
  TORCH_CHECK(
      v_input.dtype() == api::kBFloat16,
      "BF16 buffer upcast requires a BFloat16 Vulkan tensor");
  TORCH_CHECK(
      v_input.sizes().size() <= 4,
      "BF16 buffer upcast currently only supports tensors with up to 4 dimensions");

  // The experimental 16-bit buffer shader path is not numerically stable yet.
  // Fall back to a proven CPU widening route so BF16 Vulkan callers remain
  // correct while the device-native path is brought up safely.
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);
  return input.cpu().to(kFloat).vulkan();

  api::AllocationScope allocation_scope("bf16.buffer_to_float");
  api::Context* const context = api::context();
  vTensor v_out{
      context,
      v_input.sizes(),
      api::kFloat,
      api::StorageType::BUFFER,
      v_input.gpu_memory_layout(),
  };

  api::PipelineBarrier pipeline_barrier{};
  api::utils::uvec3 global_size = {
      api::utils::safe_downcast<uint32_t>(v_out.numel()),
      1u,
      1u,
  };
  api::utils::uvec3 local_size = {32u, 1u, 1u};

  context->submit_compute_job(
      VK_KERNEL(buffer_to_buffer_bfloat16_to_float),
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      v_out.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_out.buffer_metadata(),
      v_input.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ),
      v_input.buffer_metadata());

  return convert(v_out);
}

void copy_buffer_to_vtensor(
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst,
    api::PipelineBarrier& pipeline_barrier) {
  api::Context* const context = api::context();

  TORCH_CHECK(
      src_buffer.mem_size() == v_dst.gpu_nbytes(),
      "Vulkan copy_buffer_to_vtensor: source buffer and destination texture "
      "do not have the same number of bytes");

  context->submit_copy<api::VulkanBuffer, api::VulkanImage>(
      // pipeline barrier
      pipeline_barrier,
      // resources
      src_buffer,
      v_dst.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::WRITE),
      // copy details
      v_dst.extents(),
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      VK_NULL_HANDLE);
}

void copy_buffer_to_buffer(
    api::Context* const context,
    api::StorageBuffer& src,
    api::StorageBuffer& dst,
    VkFence fence_handle) {
  api::PipelineBarrier pipeline_barrier{};

  context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
      // pipeline barrier
      pipeline_barrier,
      // resources
      src.buffer(),
      dst.buffer(),
      // copy details
      {static_cast<uint32_t>(src.buffer().mem_size()), 0u, 0u},
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      fence_handle);
}

void copy_vtensor_to_buffer(
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer,
    api::PipelineBarrier& pipeline_barrier,
    const VkFence fence_handle) {
  api::Context* const context = api::context();

  TORCH_CHECK(
      v_src.gpu_nbytes() == dst_buffer.mem_size(),
      "Vulkan copy_vtensor_to_buffer: source texture and destination buffer "
      "do not have the same number of bytes");

  context->submit_copy<api::VulkanImage, api::VulkanBuffer>(
      // pipeline barrier
      pipeline_barrier,
      // resources
      v_src.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::READ),
      dst_buffer,
      // copy details
      v_src.extents(),
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      fence_handle);
}

void pack_buffer_to_vtensor(
    api::VulkanBuffer& buffer,
    vTensor& v_self,
    api::PipelineBarrier& pipeline_barrier) {
  api::Context* const context = api::context();

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    if (
        v_self.has_direct_buffer_layout() &&
        buffer.mem_size() == v_self.gpu_nbytes()) {
      add_buffer_barrier(
          pipeline_barrier,
          buffer,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::READ);
      context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
          pipeline_barrier,
          buffer,
          v_self.buffer(
              pipeline_barrier,
              api::PipelineStage::TRANSFER,
              api::MemoryAccessType::WRITE),
          {api::utils::safe_downcast<uint32_t>(buffer.mem_size()), 0u, 0u},
          {0u, 0u, 0u},
          {0u, 0u, 0u},
          VK_NULL_HANDLE);
      return;
    }
    packing::record_nchw_to_buffer_op(
        context, buffer, v_self, pipeline_barrier, VK_NULL_HANDLE);
  } else {
    api::ShaderInfo compute_shader = packing::get_nchw_to_image_shader(v_self);
    packing::record_nchw_to_image_op(
        context,
        compute_shader,
        buffer,
        v_self,
        pipeline_barrier,
        VK_NULL_HANDLE);
  }
}

void pack_staging_to_vtensor(api::VulkanBuffer& staging, vTensor& v_self) {
  api::PipelineBarrier pipeline_barrier{};
  pack_buffer_to_vtensor(staging, v_self, pipeline_barrier);
}

bool pack_vtensor_to_staging(
    vTensor& v_self,
    api::VulkanBuffer& staging,
    const VkFence fence_handle) {
  api::Context* const context = api::context();
  api::PipelineBarrier pipeline_barrier{};

  if (v_self.storage_type() == api::StorageType::BUFFER) {
    // Compute-written direct buffer tensors such as large embedding/index_select
    // gathers can read back incorrectly through the raw transfer-copy fast
    // path. Transfer-written buffers, such as large weights uploaded from CPU,
    // still rely on the direct path for exact roundtrips. Use the direct path
    // only when the buffer was not last written by a compute shader.
    if (
        v_self.has_direct_buffer_layout() &&
        !v_self.last_write_was_compute() &&
        v_self.gpu_nbytes() == staging.mem_size()) {
      return context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
          pipeline_barrier,
          v_self.buffer(pipeline_barrier, api::PipelineStage::TRANSFER),
          staging,
          {api::utils::safe_downcast<uint32_t>(staging.mem_size()), 0u, 0u},
          {0u, 0u, 0u},
          {0u, 0u, 0u},
          fence_handle);
    }
    return packing::record_buffer_to_nchw_op(
        context, v_self, staging, pipeline_barrier, fence_handle);
  } else {
    api::ShaderInfo compute_shader = packing::get_image_to_nchw_shader(v_self);
    return packing::record_image_to_nchw_op(
        context,
        compute_shader,
        v_self,
        staging,
        pipeline_barrier,
        fence_handle);
  }
}

/*
 * Broadcasting Utils
 */

// check if two tensors are broadcastable
void is_broadcastable(const Tensor& input1, const Tensor& input2) {
  TORCH_CHECK(
      input1.dim() <= 4 && input2.dim() <= 4,
      "Vulkan only supports tensors <= 4 dimensions");

  // check if the shapes of input tensors are broadcastable
  // see https://pytorch.org/docs/stable/notes/broadcasting.html
  // for broadcasting semantics
  const std::string broadcast_error_msg = "Tensors are not broadcastable!";

  if (get_dim<Dim4D::Batch>(input1) != get_dim<Dim4D::Batch>(input2)) {
    TORCH_CHECK(
        get_dim<Dim4D::Batch>(input1) == 1 ||
            get_dim<Dim4D::Batch>(input2) == 1,
        broadcast_error_msg);
  }
  if (get_dim<Dim4D::Channel>(input1) != get_dim<Dim4D::Channel>(input2)) {
    TORCH_CHECK(
        get_dim<Dim4D::Channel>(input1) == 1 ||
            get_dim<Dim4D::Channel>(input2) == 1,
        broadcast_error_msg);
  }
  if (get_dim<Dim4D::Height>(input1) != get_dim<Dim4D::Height>(input2)) {
    TORCH_CHECK(
        get_dim<Dim4D::Height>(input1) == 1 ||
            get_dim<Dim4D::Height>(input2) == 1,
        broadcast_error_msg);
  }
  if (get_dim<Dim4D::Width>(input1) != get_dim<Dim4D::Width>(input2)) {
    TORCH_CHECK(
        get_dim<Dim4D::Width>(input1) == 1 ||
            get_dim<Dim4D::Width>(input2) == 1,
        broadcast_error_msg);
  }
}

// compute the output shape by broadcasting the shapes of t1 and t2
std::vector<int64_t> broadcast_size(const Tensor& t1, const Tensor& t2) {
  int64_t t1_size = t1.dim();
  int64_t t2_size = t2.dim();

  std::vector<int64_t> out;
  if (t1_size > t2_size) {
    for (int64_t i = 0; i < t1_size; i++) {
      out.push_back(t1.sizes()[i]);
    }
  } else {
    for (int64_t i = 0; i < t2_size; i++) {
      out.push_back(t2.sizes()[i]);
    }
  }

  if (!out.empty()) {
    out[out.size() - 1] =
        std::max(get_dim<Dim4D::Width>(t1), get_dim<Dim4D::Width>(t2));
  }
  if (out.size() > 1) {
    out[out.size() - 2] =
        std::max(get_dim<Dim4D::Height>(t1), get_dim<Dim4D::Height>(t2));
  }
  if (out.size() > 2) {
    out[out.size() - 3] =
        std::max(get_dim<Dim4D::Channel>(t1), get_dim<Dim4D::Channel>(t2));
  }
  if (out.size() > 3) {
    out[out.size() - 4] =
        std::max(get_dim<Dim4D::Batch>(t1), get_dim<Dim4D::Batch>(t2));
  }

  return out;
}

api::utils::vec4 extract_texel(const Tensor& input, const ivec3& pos) {
  api::Context* const context = api::context();

  TORCH_CHECK(input.is_vulkan());
  const vTensor& v_input = convert(input);

  api::PipelineBarrier pipeline_barrier{};

  std::vector<int64_t> output_size{1, 1, 1};

  // x, y, z, w all using a single element tensor. We intend to pull
  // (0, 0, 0).x from each tensor. This allows us to isolate the effect
  // of most packing mechanism.
  api::ScalarType dtype = convert_dtype(input.scalar_type());
  vTensor v_outputs_x{context, output_size, dtype};
  vTensor v_outputs_y{context, output_size, dtype};
  vTensor v_outputs_z{context, output_size, dtype};
  vTensor v_outputs_w{context, output_size, dtype};

  const struct Block final {
    ivec3 pos;
  } block{
      pos,
  };

  api::UniformParamsBuffer params(context, block);

  context->submit_compute_job(
      VK_KERNEL(extract_texel),
      pipeline_barrier,
      {1, 1, 1},
      {1, 1, 1},
      VK_NULL_HANDLE,
      v_outputs_x.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_outputs_y.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_outputs_z.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_outputs_w.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  vec4 rv = {
      convert(v_outputs_x).cpu().data_ptr<float>()[0],
      convert(v_outputs_y).cpu().data_ptr<float>()[0],
      convert(v_outputs_z).cpu().data_ptr<float>()[0],
      convert(v_outputs_w).cpu().data_ptr<float>()[0],
  };

  return rv;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
