#include <ATen/native/vulkan/impl/Packing.h>
#include <ATen/native/vulkan/api/Diagnostics.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/LayoutTransitions.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/TensorState.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
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

bool can_native_buffer_cast_input(const vTensor& v_input) {
  const bool supports_dtype =
      v_input.dtype() == api::kFloat || v_input.dtype() == api::kInt ||
      v_input.dtype() == api::kBFloat16 ||
      (v_input.dtype() == api::kByte &&
       api::context()->adapter_ptr()->supports_int8_buffer_arithmetic());
  return supports_dtype &&
      v_input.storage_type() == api::StorageType::BUFFER &&
      v_input.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
      v_input.has_direct_buffer_layout() &&
      !v_input.is_quantized();
}

Tensor cast_vulkan_tensor_dtype_cpu_fallback(
    const Tensor& input,
    const ScalarType dtype) {
  ops::report_vulkan_cpu_fallback(
      "aten::to", "cpu_dtype_fallback", {input});
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);
  return record_tensor_write_and_return(
      input.cpu().to(dtype).vulkan(),
      "aten::to",
      "cpu_dtype_fallback",
      {input});
}

std::vector<int64_t> calc_logical_contiguous_strides(
    const std::vector<int64_t>& sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (int idx = safe_downcast<int>(sizes.size()) - 2; idx >= 0; --idx) {
    strides[idx] = strides[idx + 1] * std::max<int64_t>(sizes[idx + 1], 1);
  }
  return strides;
}

bool can_make_buffer_metadata_view_impl(
    const vTensor& v_input,
    IntArrayRef sizes,
    IntArrayRef logical_strides,
    IntArrayRef physical_strides,
    const int64_t storage_offset) {
  if (
      v_input.storage_type() != api::StorageType::BUFFER ||
      !supports_buffer_metadata_view_fast_path(v_input) ||
      sizes.size() != logical_strides.size() ||
      sizes.size() != physical_strides.size() || sizes.size() > 5 ||
      storage_offset < 0) {
    return false;
  }

  int64_t max_offset = storage_offset;
  bool is_empty = false;
  for (const auto idx : c10::irange(sizes.size())) {
    if (
        sizes[idx] < 0 || logical_strides[idx] < 0 ||
        physical_strides[idx] < 0) {
      return false;
    }
    if (sizes[idx] == 0) {
      is_empty = true;
      continue;
    }
    max_offset += (sizes[idx] - 1) * physical_strides[idx];
    if (max_offset < 0) {
      return false;
    }
  }

  return is_empty ? storage_offset <= v_input.buffer_length()
                  : max_offset < v_input.buffer_length();
}

bool can_make_typed_buffer_metadata_view_impl(
    const vTensor& v_input,
    const api::ScalarType dtype,
    IntArrayRef sizes,
    IntArrayRef logical_strides,
    IntArrayRef physical_strides,
    const int64_t storage_offset,
    const int64_t buffer_length_override) {
  if (
      v_input.storage_type() != api::StorageType::BUFFER ||
      v_input.is_quantized() || sizes.size() != logical_strides.size() ||
      sizes.size() != physical_strides.size() || sizes.size() > 4 ||
      storage_offset < 0 || buffer_length_override < 0 ||
      !api::supports_generic_buffer_view_ops(dtype, sizes.size())) {
    return false;
  }

  const int64_t source_buffer_bytes =
      v_input.buffer_length() *
      api::utils::safe_downcast<int64_t>(api::element_size(v_input.dtype()));
  const int64_t max_alias_buffer_length =
      source_buffer_bytes /
      api::utils::safe_downcast<int64_t>(api::element_size(dtype));
  if (buffer_length_override > max_alias_buffer_length) {
    return false;
  }

  int64_t max_offset = storage_offset;
  bool is_empty = false;
  for (const auto idx : c10::irange(sizes.size())) {
    if (
        sizes[idx] < 0 || logical_strides[idx] < 0 ||
        physical_strides[idx] < 0) {
      return false;
    }
    if (sizes[idx] == 0) {
      is_empty = true;
      continue;
    }
    max_offset += (sizes[idx] - 1) * physical_strides[idx];
    if (max_offset < 0) {
      return false;
    }
  }

  return is_empty ? storage_offset <= buffer_length_override
                  : max_offset < buffer_length_override;
}

Tensor cast_vulkan_tensor_dtype_buffer_native(
    const Tensor& input_arg,
    const ScalarType dtype,
    const api::ShaderInfo& shader_descriptor) {
  api::AllocationScope allocation_scope("cast.buffer");
  api::Context* const context = api::context();

  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  vTensor& v_input = convert(input);

  TORCH_CHECK(
      can_native_buffer_cast_input(v_input),
      "Native Vulkan buffer cast requires a supported buffer tensor");

  vTensor v_out{
      context,
      v_input.sizes(),
      convert_dtype(dtype),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(v_out.numel()),
      1u,
      1u,
  };
  api::UniformParamsBuffer out_meta =
      make_buffer_compute_metadata_ubo(context, v_out);
  api::UniformParamsBuffer in_meta =
      make_buffer_compute_metadata_ubo(context, v_input);

  context->submit_compute_job(
      shader_descriptor,
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_out.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer());

  return record_tensor_write_and_return(
      convert(v_out), "aten::to", "buffer_dtype_cast", {input});
}

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

std::string backing_allocation_label(const vTensor& tensor) {
  switch (tensor.storage_type()) {
    case api::StorageType::BUFFER:
      return tensor.buffer().allocation_label();
    case api::StorageType::TEXTURE_2D:
    case api::StorageType::TEXTURE_3D:
      return tensor.image().allocation_label();
    case api::StorageType::UNKNOWN:
      return std::string();
  }
  return std::string();
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
         << " runtime=" << api::current_runtime_label()
         << " backing_label=" << backing_allocation_label(v_in)
         << " path=" << path
         << " exec_layout=" << execution_layout_name(v_in.execution_layout())
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

std::string op_hit_log_path() {
  const char* env = std::getenv("PYTORCH_VULKAN_OP_HIT_LOG");
  return env ? std::string(env) : std::string();
}

bool op_hit_logging_enabled() {
  return !op_hit_log_path().empty();
}

std::mutex& op_hit_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

void log_vulkan_op_hit_impl(const char* op_name) {
  if (!op_hit_logging_enabled()) {
    return;
  }

  std::lock_guard<std::mutex> lock(op_hit_log_mutex());
  std::ofstream out(op_hit_log_path(), std::ios::app);
  out << "vulkan_op_hit op=" << op_name
      << " caller=" << api::current_allocation_label() << '\n';
}

} // namespace

void log_vulkan_op_hit(const char* op_name) {
  log_vulkan_op_hit_impl(op_name);
}

void log_vulkan_op_hit(const std::string& op_name) {
  log_vulkan_op_hit_impl(op_name.c_str());
}

std::optional<Tensor> normalized_optional_tensor(
    const std::optional<Tensor>& tensor) {
  if (tensor && tensor->defined()) {
    return tensor;
  }
  return std::nullopt;
}

bool same_optional_tensor(
    const std::optional<Tensor>& lhs,
    const std::optional<Tensor>& rhs) {
  if (lhs.has_value() != rhs.has_value()) {
    return false;
  }
  if (!lhs.has_value()) {
    return true;
  }
  return lhs->unsafeGetTensorImpl() == rhs->unsafeGetTensorImpl();
}

int64_t tensor_version_or_zero(const Tensor& tensor) {
  return tensor.is_inference() ? 0 : tensor._version();
}

bool has_inference_tensor(
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  const auto normalized_bias = normalized_optional_tensor(bias);
  return weight.is_inference() ||
      (normalized_bias && normalized_bias->is_inference());
}

LogicalBufferMetadata make_buffer_compute_metadata(const vTensor& tensor) {
  return {
      api::utils::make_whcn_uvec4(tensor.logical_sizes()),
      api::utils::make_whcn_uvec4(tensor.logical_strides()),
      api::utils::make_whcn_uvec4(tensor.physical_strides()),
      {
          api::utils::safe_downcast<uint32_t>(tensor.logical_sizes().size()),
          api::utils::safe_downcast<uint32_t>(tensor.numel()),
          api::utils::safe_downcast<uint32_t>(tensor.buffer_length()),
          api::utils::safe_downcast<uint32_t>(tensor.storage_offset()),
      },
  };
}

api::UniformParamsBuffer make_buffer_compute_metadata_ubo(
    api::Context* const context,
    const vTensor& tensor) {
  return api::UniformParamsBuffer(context, make_buffer_compute_metadata(tensor));
}

Tensor& fill_buffer_float_(Tensor& self, const float value, const char* op_name) {
  TORCH_CHECK(self.is_vulkan(), "Vulkan buffer fill expects a Vulkan tensor");
  TORCH_CHECK(
      self.scalar_type() == at::kFloat,
      "Vulkan buffer fill currently supports only float tensors");
  vTensor& v_self = convert(self);
  TORCH_CHECK(
      v_self.storage_type() == api::StorageType::BUFFER,
      "Vulkan buffer fill expects buffer storage");

  api::Context* const context = api::context();
  api::UniformParamsBuffer self_meta =
      make_buffer_compute_metadata_ubo(context, v_self);
  const struct Block final {
    float start;
    float step;
  } block{value, 0.0f};
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_self.numel(), 1)),
      1u,
      1u,
  };

  log_vulkan_op_hit(std::string(op_name) + ".buffer_float");
  context->submit_compute_job(
      VK_KERNEL(range_buffer_float),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_self.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      self_meta.buffer(),
      params.buffer());

  return self;
}

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
  return api::supports_generic_buffer_view_ops(
      v_in.dtype(), v_in.sizes().size(), v_in.is_quantized());
}

bool supports_buffer_metadata_view_fast_path(const vTensor& v_in) {
  return api::supports_generic_buffer_metadata_view_ops(
      v_in.dtype(), v_in.sizes().size(), v_in.is_quantized());
}

api::ExecutionLayout resolve_buffer_execution_layout(const vTensor& v_in) {
  return v_in.has_direct_buffer_layout() ? api::ExecutionLayout::BUFFER_DIRECT
                                         : api::ExecutionLayout::BUFFER_VIEW;
}

bool uses_buffer_execution(const vTensor& v_in) {
  return v_in.uses_buffer_execution();
}

bool uses_texture_execution(const vTensor& v_in) {
  return !v_in.uses_buffer_execution();
}

bool supports_buffer_elementwise_compute(const vTensor& v_in) {
  return supports_buffer_view_fast_path(v_in);
}

bool supports_buffer_reduction_compute(const vTensor& v_in) {
  return supports_buffer_view_fast_path(v_in) &&
      (v_in.dtype() == api::kFloat || v_in.dtype() == api::kBFloat16);
}

bool scalar_fits_vulkan_int32(const Scalar& scalar) {
  if (!scalar.isIntegral(true)) {
    return false;
  }
  const int64_t value = scalar.to<int64_t>();
  return value >= static_cast<int64_t>(std::numeric_limits<int32_t>::min()) &&
      value <= static_cast<int64_t>(std::numeric_limits<int32_t>::max());
}

int32_t scalar_to_vulkan_int32(const Scalar& scalar) {
  return safe_downcast<int32_t>(scalar.to<int64_t>());
}

bool last_dim_is_width_aligned(const Tensor& tensor) {
  return tensor.dim() == 0 || tensor.sizes().back() % 4 == 0;
}

bool supports_native_integral_buffer_compute_dtype(const api::ScalarType dtype) {
  switch (dtype) {
    case api::kInt:
      return true;
    case api::kByte:
    case api::kChar:
      return api::context()->adapter_ptr()->supports_int8_buffer_arithmetic();
    default:
      return false;
  }
}

bool supports_native_integral_buffer_compute(const Tensor& tensor) {
  if (!tensor.is_vulkan()) {
    return false;
  }
  const vTensor& v_tensor = convert(tensor);
  return supports_native_integral_buffer_compute_dtype(v_tensor.dtype()) &&
      v_tensor.storage_type() == api::StorageType::BUFFER &&
      v_tensor.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
      supports_buffer_elementwise_compute(v_tensor) && !v_tensor.is_quantized();
}

bool supports_native_bool_buffer_compute(const Tensor& tensor) {
  if (!tensor.is_vulkan()) {
    return false;
  }
  const vTensor& v_tensor = convert(tensor);
  return api::context()->adapter_ptr()->supports_int8_buffer_arithmetic() &&
      v_tensor.dtype() == api::kBool &&
      supports_buffer_elementwise_compute(v_tensor) && !v_tensor.is_quantized();
}

bool can_make_buffer_metadata_view(
    const vTensor& v_in,
    IntArrayRef sizes,
    IntArrayRef logical_strides,
    IntArrayRef physical_strides,
    int64_t storage_offset) {
  return can_make_buffer_metadata_view_impl(
      v_in, sizes, logical_strides, physical_strides, storage_offset);
}

Tensor make_buffer_metadata_view(
    const Tensor& input_arg,
    IntArrayRef sizes,
    IntArrayRef logical_strides,
    IntArrayRef physical_strides,
    int64_t storage_offset) {
  return make_buffer_metadata_view_checked(
      input_arg,
      sizes,
      logical_strides,
      physical_strides,
      storage_offset,
      "make_buffer_metadata_view");
}

bool can_make_typed_buffer_metadata_view(
    const vTensor& v_in,
    const ScalarType dtype,
    IntArrayRef sizes,
    IntArrayRef logical_strides,
    IntArrayRef physical_strides,
    const int64_t storage_offset,
    const int64_t buffer_length_override) {
  return can_make_typed_buffer_metadata_view_impl(
      v_in,
      convert_dtype(dtype),
      sizes,
      logical_strides,
      physical_strides,
      storage_offset,
      buffer_length_override);
}

Tensor make_typed_buffer_metadata_view(
    const Tensor& input_arg,
    const ScalarType dtype,
    IntArrayRef sizes,
    IntArrayRef logical_strides,
    IntArrayRef physical_strides,
    const int64_t storage_offset,
    const int64_t buffer_length_override,
    const api::ExecutionLayout execution_layout) {
  return make_typed_buffer_metadata_view_checked(
      input_arg,
      dtype,
      sizes,
      logical_strides,
      physical_strides,
      storage_offset,
      buffer_length_override,
      execution_layout,
      "make_typed_buffer_metadata_view");
}

std::string describe_buffer_view_fast_path_failure(const vTensor& v_in) {
  std::ostringstream stream;
  stream
      << "Vulkan texture materialization from buffer views currently only "
      << "supports non-quantized buffer-backed tensors with supported dtypes "
      << "and up to 4 dimensions"
      << " (caller=" << api::current_allocation_label()
      << ", sizes=" << format_sizes(v_in.sizes())
      << ", ndim=" << v_in.sizes().size()
      << ", dtype=" << api::to_string(v_in.dtype())
      << ", quantized=" << (v_in.is_quantized() ? 1 : 0)
      << ", exec_layout=" << execution_layout_name(v_in.execution_layout())
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
  vTensor v_out{
      context,
      v_in.sizes(),
      v_in.dtype(),
      api::StorageType::BUFFER,
      memory_layout,
  };

  if (
      v_in.storage_type() == api::StorageType::BUFFER &&
      v_in.dtype() == api::kFloat) {
    vTensor v_src = v_in;
    api::PipelineBarrier pipeline_barrier{};
    const api::utils::uvec3 global_size = {
        api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_out.numel(), 1)),
        1u,
        1u,
    };
    context->submit_compute_job(
        VK_KERNEL(buffer_to_buffer),
        pipeline_barrier,
        global_size,
        adaptive_work_group_size(global_size),
        VK_NULL_HANDLE,
        v_out.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        make_buffer_compute_metadata_ubo(context, v_out).buffer(),
        v_src.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::READ),
        make_buffer_compute_metadata_ubo(context, v_src).buffer());
    return v_out;
  }

  api::StorageBuffer staging(context, v_in.dtype(), v_in.numel());
  vTensor v_src = v_in;
  pack_vtensor_to_staging(v_src, staging.buffer());

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

Tensor ensure_buffer_storage(
    const Tensor& input_arg,
    api::GPUMemoryLayout memory_layout) {
  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  vTensor& v_input = convert(input);

  if (
      v_input.storage_type() == api::StorageType::BUFFER &&
      v_input.gpu_memory_layout() == memory_layout &&
      v_input.has_direct_buffer_layout()) {
    return input;
  }

  TORCH_CHECK(
      supports_buffer_view_fast_path(v_input),
      describe_buffer_view_fast_path_failure(v_input));

  log_materialize_event(
      "ensure_buffer_storage",
      v_input,
      api::StorageType::BUFFER,
      memory_layout,
      v_input.storage_type() == api::StorageType::BUFFER
          ? "buffer_relayout"
          : "texture_to_buffer");

  return record_tensor_write_and_return(
      convert(materialize_to_contiguous_buffer(v_input, memory_layout)),
      "ensure_buffer_storage",
      v_input.storage_type() == api::StorageType::BUFFER
          ? "buffer_relayout"
          : "texture_to_buffer",
      {input});
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
        return record_tensor_write_and_return(
            convert(
                packing::convert_image_channels_packed_to_width_packed(v_input)),
            "ensure_texture_storage",
            "image_layout_convert_width",
            {input});
      }
      if (memory_layout == api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED) {
        log_materialize_event(
            "ensure_texture_storage",
            v_input,
            storage_type,
            memory_layout,
            "image_layout_convert_height");
        return record_tensor_write_and_return(
            convert(
                packing::convert_image_channels_packed_to_height_packed(v_input)),
            "ensure_texture_storage",
            "image_layout_convert_height",
            {input});
      }
    }
  }

  TORCH_CHECK(
      supports_buffer_view_fast_path(v_input),
      describe_buffer_view_fast_path_failure(v_input));

  api::Context* const context = api::context();
  log_materialize_event(
      "ensure_texture_storage",
      v_input,
      storage_type,
      memory_layout,
      "buffer_to_texture_via_staging");

  api::StorageBuffer staging(context, v_input.dtype(), v_input.numel());
  vTensor v_src = v_input;
  pack_vtensor_to_staging(v_src, staging.buffer());

  vTensor v_out{
      context,
      v_input.sizes(),
      v_input.dtype(),
      storage_type,
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
  return record_tensor_write_and_return(
      convert(v_out), "ensure_texture_storage", "buffer_to_texture", {input});
}

Tensor create_buffer_tensor(
    IntArrayRef sizes,
    const ScalarType dtype,
    const bool persistent) {
  return mark_tensor_execution(
      convert(vTensor{
          api::context(),
          sizes.vec(),
          convert_dtype(dtype),
          api::StorageType::BUFFER,
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      }),
      api::ExecutionLayout::BUFFER_DIRECT,
      persistent);
}

Tensor& copy_buffer_tensor_direct_(Tensor& dst, const Tensor& src) {
  TORCH_CHECK(dst.is_vulkan(), "Destination must be a Vulkan tensor");
  TORCH_CHECK(src.is_vulkan(), "Source must be a Vulkan tensor");
  TORCH_CHECK(
      dst.sizes() == src.sizes(),
      "Vulkan direct buffer copy requires matching sizes");
  TORCH_CHECK(
      dst.scalar_type() == src.scalar_type(),
      "Vulkan direct buffer copy requires matching dtypes");

  vTensor& v_dst = convert(dst);
  vTensor v_src = convert(src);
  TORCH_CHECK(
      v_dst.storage_type() == api::StorageType::BUFFER &&
          v_src.storage_type() == api::StorageType::BUFFER,
      "Vulkan direct buffer copy requires buffer-backed tensors");
  TORCH_CHECK(
      v_dst.has_direct_buffer_layout() && v_src.has_direct_buffer_layout(),
      "Vulkan direct buffer copy requires direct buffer layout");
  TORCH_CHECK(
      v_dst.gpu_nbytes() == v_src.gpu_nbytes(),
      "Vulkan direct buffer copy requires matching physical byte sizes");
  if (!is_raw_buffer_copy_legal(src, dst)) {
    std::ostringstream detail;
    detail << "src={" << describe_tensor_state(src) << "} dst={"
           << describe_tensor_state(dst) << "}";
    api::fail_vulkan(
        api::VulkanFailureClass::RawCopyIllegal,
        "copy_buffer_tensor_direct_",
        "RawCopyIllegal",
        detail.str());
  }

  api::PipelineBarrier pipeline_barrier{};
  api::Context* const context = api::context();
  context->submit_copy<api::VulkanBuffer, api::VulkanBuffer>(
      pipeline_barrier,
      v_src.buffer(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::READ),
      v_dst.buffer(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::WRITE),
      {api::utils::safe_downcast<uint32_t>(v_src.gpu_nbytes()), 0u, 0u},
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      VK_NULL_HANDLE);
  record_tensor_write(dst, "copy_buffer_tensor_direct_", "raw_buffer_copy", {src});
  return dst;
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
      make_buffer_compute_metadata_ubo(context, v_out).buffer(),
      v_input.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ),
      make_buffer_compute_metadata_ubo(context, v_input).buffer());

  return record_tensor_write_and_return(
      convert(v_out), "aten::to", "bf16_buffer_to_float", {input});
}

Tensor mark_tensor_execution(
    const Tensor& input,
    const api::ExecutionLayout execution_layout,
    const bool persistent) {
  if (!input.is_vulkan()) {
    return input;
  }

  vTensor& v_input = convert(input);
  v_input.set_execution_layout(execution_layout);
  v_input.set_execution_persistent(persistent);
  return input;
}

Tensor contiguous_inference(
    const Tensor& tensor,
    const c10::MemoryFormat memory_format) {
  c10::InferenceMode inference_mode_guard;
  return tensor.contiguous(memory_format);
}

Tensor reshape_inference(const Tensor& tensor, IntArrayRef shape) {
  c10::InferenceMode inference_mode_guard;
  return tensor.reshape(shape);
}

Tensor cast_vulkan_tensor_dtype(const Tensor& input_arg, ScalarType dtype) {
  if (input_arg.scalar_type() == dtype) {
    return input_arg;
  }

  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  vTensor v_input = convert(input);

  switch (resolve_vulkan_cast_method(input.scalar_type(), dtype)) {
    case VulkanCastMethod::Identity:
      return input;
    case VulkanCastMethod::NativeBufferFloatToInt:
      return cast_vulkan_tensor_dtype_cpu_fallback(input, dtype);
    case VulkanCastMethod::NativeBufferIntToFloat:
      if (!can_native_buffer_cast_input(v_input)) {
        return cast_vulkan_tensor_dtype_cpu_fallback(input, dtype);
      }
      return cast_vulkan_tensor_dtype_buffer_native(
          input, dtype, VK_KERNEL(buffer_cast_int_to_float));
    case VulkanCastMethod::NativeBufferByteToFloat:
      if (!can_native_buffer_cast_input(v_input)) {
        return cast_vulkan_tensor_dtype_cpu_fallback(input, dtype);
      }
      return cast_vulkan_tensor_dtype_buffer_native(
          input, dtype, VK_KERNEL(buffer_cast_uint8_to_float));
    case VulkanCastMethod::NativeBufferFloatToBFloat16:
      if (!can_native_buffer_cast_input(v_input)) {
        return cast_vulkan_tensor_dtype_cpu_fallback(input, dtype);
      }
      return cast_vulkan_tensor_dtype_buffer_native(
          input, dtype, VK_KERNEL(buffer_cast_float_to_bfloat16));
    case VulkanCastMethod::NativeBufferBFloat16ToFloat:
      if (!can_native_buffer_cast_input(v_input)) {
        return cast_vulkan_tensor_dtype_cpu_fallback(input, dtype);
      }
      log_vulkan_op_hit("aten::cast.bfloat16_to_float_buffer_native");
      return upcast_bfloat16_buffer_to_float(input);
    case VulkanCastMethod::CpuFallback:
      return cast_vulkan_tensor_dtype_cpu_fallback(input, dtype);
    case VulkanCastMethod::Unsupported:
      {
        std::ostringstream detail;
        detail << "from=" << input.scalar_type() << " to=" << dtype;
        api::fail_vulkan(
            api::VulkanFailureClass::Unsupported,
            "aten::to",
            "UnsupportedVulkanCast",
            detail.str());
      }
  }

  api::fail_vulkan(
      api::VulkanFailureClass::Unsupported,
      "aten::to",
      "InvalidVulkanCastDispatchState");
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
    // The generic helper is used for both CPU-style logical staging buffers and
    // already packed GPU buffers. Only the former is safe to assume here, so
    // always repack through the metadata-aware shader path.
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
        fence_handle != VK_NULL_HANDLE &&
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
  const auto broadcast_error_msg = [&]() {
    std::ostringstream stream;
    stream << "Tensors are not broadcastable! input1={"
           << describe_tensor_state(input1) << "} input2={"
           << describe_tensor_state(input2) << "}";
    return api::report_vulkan_failure(
        api::VulkanFailureClass::TensorStateInvalid,
        "is_broadcastable",
        "BroadcastShapeInvalid",
        stream.str());
  };

  if (get_dim<Dim4D::Batch>(input1) != get_dim<Dim4D::Batch>(input2)) {
    TORCH_CHECK(
        get_dim<Dim4D::Batch>(input1) == 1 ||
            get_dim<Dim4D::Batch>(input2) == 1,
        broadcast_error_msg());
  }
  if (get_dim<Dim4D::Channel>(input1) != get_dim<Dim4D::Channel>(input2)) {
    TORCH_CHECK(
        get_dim<Dim4D::Channel>(input1) == 1 ||
            get_dim<Dim4D::Channel>(input2) == 1,
        broadcast_error_msg());
  }
  if (get_dim<Dim4D::Height>(input1) != get_dim<Dim4D::Height>(input2)) {
    TORCH_CHECK(
        get_dim<Dim4D::Height>(input1) == 1 ||
            get_dim<Dim4D::Height>(input2) == 1,
        broadcast_error_msg());
  }
  if (get_dim<Dim4D::Width>(input1) != get_dim<Dim4D::Width>(input2)) {
    TORCH_CHECK(
        get_dim<Dim4D::Width>(input1) == 1 ||
            get_dim<Dim4D::Width>(input2) == 1,
        broadcast_error_msg());
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
  report_vulkan_cpu_fallback(
      "vulkan::extract_texel",
      "utility_sync_readback",
      {input},
      VulkanCpuFallbackKind::SyncReadback);
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
      convert(v_outputs_x).cpu().const_data_ptr<float>()[0],
      convert(v_outputs_y).cpu().const_data_ptr<float>()[0],
      convert(v_outputs_z).cpu().const_data_ptr<float>()[0],
      convert(v_outputs_w).cpu().const_data_ptr<float>()[0],
  };

  return rv;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
