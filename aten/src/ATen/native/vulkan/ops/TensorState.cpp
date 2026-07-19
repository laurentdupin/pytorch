#include <ATen/native/vulkan/ops/TensorState.h>

#include <ATen/native/vulkan/api/Diagnostics.h>
#include <ATen/native/vulkan/ops/Convert.h>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace {

std::string tensor_state_log_path() {
  const char* env = std::getenv("PYTORCH_VULKAN_TENSOR_STATE_LOG");
  return env ? std::string(env) : std::string();
}

bool env_flag_enabled(const char* name) {
  const char* env = std::getenv(name);
  if (!env || !*env) {
    return false;
  }
  return std::string(env) != "0";
}

std::mutex& tensor_state_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

void append_tensor_state_log_line(const std::string& line) {
  if (!vulkan_tensor_state_logging_enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(tensor_state_log_mutex());
  std::ofstream out(tensor_state_log_path(), std::ios::app);
  out << line << '\n';
}

template <typename T>
void hash_combine_value(uint64_t& seed, const T& value) {
  seed ^= static_cast<uint64_t>(std::hash<T>{}(value)) +
      UINT64_C(0x9e3779b97f4a7c15) + (seed << 6u) + (seed >> 2u);
}

void hash_combine_sizes(uint64_t& seed, const std::vector<int64_t>& values) {
  hash_combine_value(seed, values.size());
  for (const int64_t value : values) {
    hash_combine_value(seed, value);
  }
}

std::string format_i64_vector(const std::vector<int64_t>& values) {
  std::ostringstream stream;
  stream << '[';
  for (size_t idx = 0; idx < values.size(); ++idx) {
    if (idx != 0u) {
      stream << ',';
    }
    stream << values[idx];
  }
  stream << ']';
  return stream.str();
}

uint64_t pointer_id(const void* ptr) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
}

uint64_t tensor_generation_or_zero(const Tensor& tensor) {
  return tensor.defined() && !tensor.is_inference()
      ? static_cast<uint64_t>(tensor._version())
      : 0u;
}

bool is_logical_physical_view(const vTensor& tensor) {
  return tensor.storage_offset() != 0 ||
      tensor.execution_layout() == api::ExecutionLayout::BUFFER_VIEW;
}

VulkanTensorRepr classify_vulkan_tensor(const vTensor& tensor) {
  if (tensor.storage_type() == api::StorageType::BUFFER) {
    if (tensor.is_packed_weight()) {
      return VulkanTensorRepr::PackedWeight;
    }
    if (tensor.execution_layout() == api::ExecutionLayout::BUFFER_VIEW ||
        is_logical_physical_view(tensor)) {
      return VulkanTensorRepr::BufferMetadataView;
    }
    if (tensor.buffer_uses_host_visible_allocation()) {
      return VulkanTensorRepr::HostVisibleUploadBuffer;
    }
    if (
        tensor.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
        tensor.physical_sizes() != tensor.logical_sizes()) {
      return VulkanTensorRepr::BufferWidthPacked;
    }
    if (tensor.has_direct_buffer_layout()) {
      return VulkanTensorRepr::BufferDirect;
    }
    return VulkanTensorRepr::BufferWidthPacked;
  }
  if (
      tensor.storage_type() == api::StorageType::TEXTURE_2D ||
      tensor.storage_type() == api::StorageType::TEXTURE_3D) {
    return VulkanTensorRepr::TexturePacked;
  }
  return VulkanTensorRepr::Invalid;
}

VulkanTensorStateValidation fail_validation(
    const char* op_name,
    const char* route_name,
    VulkanTensorUse use,
    const std::string& reason,
    const VulkanTensorStateDesc& desc) {
  const api::VulkanFailureClass failure_class =
      reason == "MetadataViewOutOfRange" ||
          reason == "RawCopyRequestedForMetadataView"
      ? api::VulkanFailureClass::MetadataViewInvalid
      : api::VulkanFailureClass::TensorStateInvalid;
  std::ostringstream detail;
  detail << describe_tensor_state(desc);
  if (route_name && route_name[0] != '\0') {
    detail << " route=" << route_name;
  }
  detail << " use=" << vulkan_tensor_use_name(use);
  return {
      false,
      reason,
      api::report_vulkan_failure(
          failure_class, op_name, reason.c_str(), detail.str())};
}

bool metadata_range_in_bounds(const VulkanTensorStateDesc& desc) {
  if (desc.storage_type != api::StorageType::BUFFER) {
    return true;
  }
  if (desc.storage_offset < 0 || desc.buffer_length < 0) {
    return false;
  }
  if (
      desc.logical_sizes.size() != desc.logical_strides.size() ||
      desc.logical_sizes.size() != desc.physical_strides.size()) {
    return false;
  }

  int64_t max_offset = desc.storage_offset;
  bool empty = false;
  for (size_t idx = 0; idx < desc.logical_sizes.size(); ++idx) {
    const int64_t size = desc.logical_sizes[idx];
    const int64_t logical_stride = desc.logical_strides[idx];
    const int64_t physical_stride = desc.physical_strides[idx];
    if (size < 0 || logical_stride < 0 || physical_stride < 0) {
      return false;
    }
    if (size == 0) {
      empty = true;
      continue;
    }
    const int64_t contribution = (size - 1) * physical_stride;
    if (
        contribution < 0 ||
        max_offset > std::numeric_limits<int64_t>::max() - contribution) {
      return false;
    }
    max_offset += contribution;
  }

  return empty ? desc.storage_offset <= desc.buffer_length
               : max_offset < desc.buffer_length;
}

} // namespace

const char* vulkan_tensor_repr_name(const VulkanTensorRepr repr) {
  switch (repr) {
    case VulkanTensorRepr::CpuTensor:
      return "CpuTensor";
    case VulkanTensorRepr::TexturePacked:
      return "TexturePacked";
    case VulkanTensorRepr::BufferDirect:
      return "BufferDirect";
    case VulkanTensorRepr::BufferWidthPacked:
      return "BufferWidthPacked";
    case VulkanTensorRepr::HostVisibleUploadBuffer:
      return "HostVisibleUploadBuffer";
    case VulkanTensorRepr::BufferMetadataView:
      return "BufferMetadataView";
    case VulkanTensorRepr::ReplayOwnedOutput:
      return "ReplayOwnedOutput";
    case VulkanTensorRepr::MaterializedContiguous:
      return "MaterializedContiguous";
    case VulkanTensorRepr::CpuReadbackStaging:
      return "CpuReadbackStaging";
    case VulkanTensorRepr::PackedWeight:
      return "PackedWeight";
    case VulkanTensorRepr::Invalid:
      return "Invalid";
  }
  return "Invalid";
}

const char* vulkan_tensor_use_name(const VulkanTensorUse use) {
  switch (use) {
    case VulkanTensorUse::Read:
      return "Read";
    case VulkanTensorUse::Write:
      return "Write";
    case VulkanTensorUse::ReadWrite:
      return "ReadWrite";
    case VulkanTensorUse::ShapeOnly:
      return "ShapeOnly";
    case VulkanTensorUse::RawCopySource:
      return "RawCopySource";
    case VulkanTensorUse::RawCopyDestination:
      return "RawCopyDestination";
    case VulkanTensorUse::KernelInput:
      return "KernelInput";
    case VulkanTensorUse::KernelOutput:
      return "KernelOutput";
    case VulkanTensorUse::ReplayExport:
      return "ReplayExport";
  }
  return "Unknown";
}

const char* vulkan_storage_type_name(const api::StorageType storage_type) {
  switch (storage_type) {
    case api::StorageType::BUFFER:
      return "BUFFER";
    case api::StorageType::TEXTURE_3D:
      return "TEXTURE_3D";
    case api::StorageType::TEXTURE_2D:
      return "TEXTURE_2D";
    case api::StorageType::UNKNOWN:
      return "UNKNOWN";
  }
  return "UNKNOWN";
}

const char* vulkan_memory_layout_name(
    const api::GPUMemoryLayout memory_layout) {
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

bool vulkan_tensor_state_validation_enabled() {
  return env_flag_enabled("PYTORCH_VULKAN_VALIDATE_TENSOR_STATE");
}

bool vulkan_tensor_state_logging_enabled() {
  return !tensor_state_log_path().empty();
}

uint64_t tensor_logical_desc_hash(const Tensor& tensor) {
  uint64_t seed = UINT64_C(0x84222325cbf29ce4);
  if (!tensor.defined()) {
    return seed;
  }

  hash_combine_value(seed, static_cast<int>(tensor.scalar_type()));
  if (tensor.is_vulkan()) {
    const vTensor& v_tensor = convert(tensor);
    hash_combine_sizes(seed, v_tensor.logical_sizes());
    hash_combine_sizes(seed, v_tensor.logical_strides());
    hash_combine_sizes(seed, v_tensor.physical_sizes());
    hash_combine_sizes(seed, v_tensor.physical_strides());
    hash_combine_value(seed, v_tensor.storage_offset());
    hash_combine_value(seed, static_cast<int>(v_tensor.storage_type()));
    hash_combine_value(seed, static_cast<int>(v_tensor.gpu_memory_layout()));
    hash_combine_value(seed, static_cast<int>(v_tensor.execution_layout()));
    hash_combine_value(seed, v_tensor.buffer_length());
  } else {
    hash_combine_sizes(seed, tensor.sizes().vec());
    hash_combine_sizes(seed, tensor.strides().vec());
    hash_combine_value(seed, tensor.storage_offset());
  }
  return seed;
}

uint64_t tensor_storage_identity(const Tensor& tensor) {
  if (!tensor.defined()) {
    return 0u;
  }
  if (tensor.is_vulkan()) {
    const vTensor& v_tensor = convert(tensor);
    if (v_tensor.storage_type() == api::StorageType::BUFFER) {
      return pointer_id(&v_tensor.buffer());
    }
    if (
        v_tensor.storage_type() == api::StorageType::TEXTURE_2D ||
        v_tensor.storage_type() == api::StorageType::TEXTURE_3D) {
      return pointer_id(&v_tensor.image());
    }
  }
  return pointer_id(tensor.unsafeGetTensorImpl());
}

VulkanTensorStateDesc inspect_tensor_state(const Tensor& tensor) {
  VulkanTensorStateDesc desc;
  if (!tensor.defined()) {
    desc.repr = VulkanTensorRepr::Invalid;
    return desc;
  }

  desc.dtype = tensor.scalar_type();
  desc.logical_sizes = tensor.sizes().vec();
  desc.logical_strides = tensor.strides().vec();
  desc.storage_offset = tensor.storage_offset();
  desc.view_id = pointer_id(tensor.unsafeGetTensorImpl());
  desc.generation = tensor_generation_or_zero(tensor);
  desc.logical_desc_hash = tensor_logical_desc_hash(tensor);
  desc.producer = api::current_allocation_label();
  desc.route = api::current_runtime_label();

  if (!tensor.is_vulkan()) {
    desc.repr = VulkanTensorRepr::CpuTensor;
    desc.physical_sizes = desc.logical_sizes;
    desc.physical_strides = desc.logical_strides;
    desc.storage_id = pointer_id(tensor.unsafeGetTensorImpl());
    return desc;
  }

  const vTensor& v_tensor = convert(tensor);
  desc.storage_type = v_tensor.storage_type();
  desc.memory_layout = v_tensor.gpu_memory_layout();
  desc.execution_layout = v_tensor.execution_layout();
  desc.logical_sizes = v_tensor.logical_sizes();
  desc.logical_strides = v_tensor.logical_strides();
  desc.physical_sizes = v_tensor.physical_sizes();
  desc.physical_strides = v_tensor.physical_strides();
  desc.storage_offset = v_tensor.storage_offset();
  desc.buffer_length = desc.storage_type == api::StorageType::BUFFER
      ? v_tensor.buffer_length()
      : static_cast<int64_t>(v_tensor.gpu_numel());
  desc.is_view = is_logical_physical_view(v_tensor);
  desc.is_direct_buffer = desc.storage_type == api::StorageType::BUFFER &&
      v_tensor.has_direct_buffer_layout();
  desc.host_visible = v_tensor.buffer_uses_host_visible_allocation();
  desc.persistent = v_tensor.execution_desc().persistent;
  desc.last_write_was_compute = v_tensor.last_write_was_compute();
  desc.storage_id = tensor_storage_identity(tensor);
  desc.repr = classify_vulkan_tensor(v_tensor);
  return desc;
}

VulkanTensorStateValidation validate_tensor_state(
    const Tensor& tensor,
    const VulkanTensorUse use,
    const char* op_name,
    const char* route_name) {
  const VulkanTensorStateDesc desc = inspect_tensor_state(tensor);
  if (desc.repr == VulkanTensorRepr::Invalid) {
    return fail_validation(op_name, route_name, use, "UndefinedTensor", desc);
  }
  if (!tensor.is_vulkan()) {
    return {};
  }
  if (desc.logical_sizes.size() != desc.logical_strides.size()) {
    return fail_validation(
        op_name, route_name, use, "LogicalSizesStridesRankMismatch", desc);
  }
  if (
      desc.storage_type == api::StorageType::BUFFER &&
      desc.logical_sizes.size() != desc.physical_strides.size()) {
    return fail_validation(
        op_name, route_name, use, "LogicalPhysicalRankMismatch", desc);
  }
  if (
      desc.storage_type == api::StorageType::BUFFER &&
      !api::uses_buffer_execution(desc.execution_layout)) {
    return fail_validation(
        op_name, route_name, use, "BufferStorageWithoutBufferExecution", desc);
  }
  if (
      desc.storage_type != api::StorageType::BUFFER &&
      desc.execution_layout != api::ExecutionLayout::TEXTURE) {
    return fail_validation(
        op_name, route_name, use, "TextureStorageWithoutTextureExecution", desc);
  }
  if (!metadata_range_in_bounds(desc)) {
    return fail_validation(
        op_name, route_name, use, "MetadataViewOutOfRange", desc);
  }
  if (
      (use == VulkanTensorUse::RawCopySource ||
       use == VulkanTensorUse::RawCopyDestination) &&
      desc.repr == VulkanTensorRepr::BufferMetadataView) {
    return fail_validation(
        op_name, route_name, use, "RawCopyRequestedForMetadataView", desc);
  }
  return {};
}

void assert_valid_tensor_state_debug(
    const Tensor& tensor,
    const VulkanTensorUse use,
    const char* op_name,
    const char* route_name) {
  const VulkanTensorStateValidation validation =
      validate_tensor_state(tensor, use, op_name, route_name);
  if (vulkan_tensor_state_validation_enabled()) {
    TORCH_CHECK(validation.ok, validation.message);
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(validation.ok, validation.message);
}

std::string describe_tensor_state(const Tensor& tensor) {
  return describe_tensor_state(inspect_tensor_state(tensor));
}

std::string describe_tensor_state(const VulkanTensorStateDesc& desc) {
  std::ostringstream stream;
  stream << "repr=" << vulkan_tensor_repr_name(desc.repr)
         << " storage=" << vulkan_storage_type_name(desc.storage_type)
         << " memory_layout=" << vulkan_memory_layout_name(desc.memory_layout)
         << " execution_layout=" << api::to_string(desc.execution_layout)
         << " dtype=" << desc.dtype
         << " logical_sizes=" << format_i64_vector(desc.logical_sizes)
         << " logical_strides=" << format_i64_vector(desc.logical_strides)
         << " physical_sizes=" << format_i64_vector(desc.physical_sizes)
         << " physical_strides=" << format_i64_vector(desc.physical_strides)
         << " storage_offset=" << desc.storage_offset
         << " buffer_length=" << desc.buffer_length
         << " is_view=" << (desc.is_view ? 1 : 0)
         << " direct=" << (desc.is_direct_buffer ? 1 : 0)
         << " host_visible=" << (desc.host_visible ? 1 : 0)
         << " persistent=" << (desc.persistent ? 1 : 0)
         << " replay_owned=" << (desc.replay_owned ? 1 : 0)
         << " last_write_compute=" << (desc.last_write_was_compute ? 1 : 0)
         << " storage_id=0x" << std::hex << desc.storage_id
         << " view_id=0x" << desc.view_id
         << " generation=" << std::dec << desc.generation
         << " logical_hash=0x" << std::hex << desc.logical_desc_hash
         << std::dec;
  if (!desc.producer.empty()) {
    stream << " producer=" << desc.producer;
  }
  if (!desc.route.empty()) {
    stream << " route=" << desc.route;
  }
  return stream.str();
}

void log_tensor_state(
    const Tensor& tensor,
    const VulkanTensorUse use,
    const char* op_name,
    const char* route_name) {
  if (!vulkan_tensor_state_logging_enabled()) {
    return;
  }
  std::ostringstream stream;
  stream << "vulkan_tensor_state";
  if (op_name && op_name[0] != '\0') {
    stream << " op=" << op_name;
  }
  if (route_name && route_name[0] != '\0') {
    stream << " route=" << route_name;
  }
  stream << " use=" << vulkan_tensor_use_name(use) << ' '
         << describe_tensor_state(tensor);
  append_tensor_state_log_line(stream.str());
}

bool is_raw_buffer_copy_legal(const Tensor& src, const Tensor& dst) {
  if (!src.defined() || !dst.defined() || !src.is_vulkan() || !dst.is_vulkan()) {
    return false;
  }
  const vTensor& v_src = convert(src);
  const vTensor& v_dst = convert(dst);
  return v_src.storage_type() == api::StorageType::BUFFER &&
      v_dst.storage_type() == api::StorageType::BUFFER &&
      v_src.storage_offset() == 0 && v_dst.storage_offset() == 0 &&
      v_src.buffer_length() == v_src.gpu_numel() &&
      v_dst.buffer_length() == v_dst.gpu_numel() &&
      v_src.dtype() == v_dst.dtype() &&
      v_src.gpu_memory_layout() == v_dst.gpu_memory_layout() &&
      v_src.gpu_nbytes() == v_dst.gpu_nbytes() &&
      v_src.logical_sizes() == v_dst.logical_sizes() &&
      v_src.gpu_sizes() == v_dst.gpu_sizes() &&
      v_src.physical_strides() == v_dst.physical_strides();
}

bool requires_logical_pack_shader(const Tensor& src, const Tensor& dst) {
  return !is_raw_buffer_copy_legal(src, dst);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
