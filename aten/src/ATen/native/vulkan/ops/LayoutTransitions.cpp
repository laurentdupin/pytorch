#include <ATen/native/vulkan/ops/LayoutTransitions.h>

#include <ATen/native/vulkan/api/Diagnostics.h>
#include <ATen/native/vulkan/ops/Convert.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/planning/TransitionPlanner.h>
#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace {

void validate_created_view(
    const Tensor& view,
    const char* op_name,
    const char* route_name) {
  const VulkanTensorStateValidation validation = validate_tensor_state(
      view, VulkanTensorUse::Read, op_name, route_name);
  TORCH_CHECK(validation.ok, validation.message);
  assert_valid_tensor_state_debug(
      view, VulkanTensorUse::Read, op_name, route_name);
}

utils::TransitionReason transition_reason_for_materialize_reason(
    const VulkanMaterializeReason reason) {
  switch (reason) {
    case VulkanMaterializeReason::KernelRequiresBuffer:
    case VulkanMaterializeReason::KernelRequiresDirectBuffer:
    case VulkanMaterializeReason::KernelRequiresTexture:
      return utils::TransitionReason::RequiredConsumerLayout;
    case VulkanMaterializeReason::RawCopyRequiresContiguous:
    case VulkanMaterializeReason::MetadataViewUnsupported:
      return utils::TransitionReason::RequiredContiguousMaterialization;
    case VulkanMaterializeReason::ReplayOutputEscaped:
      return utils::TransitionReason::RequiredCorrectnessMaterialization;
    case VulkanMaterializeReason::Readback:
      return utils::TransitionReason::RequiredFinalReadback;
    case VulkanMaterializeReason::Upload:
      return utils::TransitionReason::RequiredHostUpload;
    case VulkanMaterializeReason::DTypeCast:
      return utils::TransitionReason::RequiredDTypeCast;
    case VulkanMaterializeReason::MetadataViewCreated:
    case VulkanMaterializeReason::TypedMetadataViewCreated:
      return utils::TransitionReason::MetadataViewOnly;
    case VulkanMaterializeReason::Unknown:
      return utils::TransitionReason::UnknownTransitionReason;
  }
  return utils::TransitionReason::UnknownTransitionReason;
}

utils::TransitionKind transition_kind_for_materialize_reason(
    const VulkanMaterializeReason reason) {
  switch (reason) {
    case VulkanMaterializeReason::MetadataViewCreated:
    case VulkanMaterializeReason::TypedMetadataViewCreated:
      return utils::TransitionKind::MetadataView;
    case VulkanMaterializeReason::Readback:
    case VulkanMaterializeReason::Upload:
      return utils::TransitionKind::HostTransfer;
    case VulkanMaterializeReason::ReplayOutputEscaped:
      return utils::TransitionKind::SemanticMaterialization;
    case VulkanMaterializeReason::KernelRequiresBuffer:
    case VulkanMaterializeReason::KernelRequiresDirectBuffer:
    case VulkanMaterializeReason::KernelRequiresTexture:
    case VulkanMaterializeReason::RawCopyRequiresContiguous:
    case VulkanMaterializeReason::MetadataViewUnsupported:
    case VulkanMaterializeReason::DTypeCast:
      return utils::TransitionKind::LayoutMaterialization;
    case VulkanMaterializeReason::Unknown:
      return utils::TransitionKind::Unknown;
  }
  return utils::TransitionKind::Unknown;
}

} // namespace

const char* vulkan_materialize_reason_name(
    const VulkanMaterializeReason reason) {
  switch (reason) {
    case VulkanMaterializeReason::KernelRequiresBuffer:
      return "KernelRequiresBuffer";
    case VulkanMaterializeReason::KernelRequiresDirectBuffer:
      return "KernelRequiresDirectBuffer";
    case VulkanMaterializeReason::KernelRequiresTexture:
      return "KernelRequiresTexture";
    case VulkanMaterializeReason::RawCopyRequiresContiguous:
      return "RawCopyRequiresContiguous";
    case VulkanMaterializeReason::ReplayOutputEscaped:
      return "ReplayOutputEscaped";
    case VulkanMaterializeReason::MetadataViewUnsupported:
      return "MetadataViewUnsupported";
    case VulkanMaterializeReason::Readback:
      return "Readback";
    case VulkanMaterializeReason::Upload:
      return "Upload";
    case VulkanMaterializeReason::DTypeCast:
      return "DTypeCast";
    case VulkanMaterializeReason::MetadataViewCreated:
      return "MetadataViewCreated";
    case VulkanMaterializeReason::TypedMetadataViewCreated:
      return "TypedMetadataViewCreated";
    case VulkanMaterializeReason::Unknown:
      return "Unknown";
  }
  return "Unknown";
}

void log_layout_transition(
    const char* op_name,
    const VulkanMaterializeReason reason,
    const Tensor& src,
    const Tensor& dst) {
  if (!utils::transition_logging_enabled()) {
    return;
  }
  const bool host_transfer = reason == VulkanMaterializeReason::Upload ||
      reason == VulkanMaterializeReason::Readback;
  const bool metadata_view =
      reason == VulkanMaterializeReason::MetadataViewCreated ||
      reason == VulkanMaterializeReason::TypedMetadataViewCreated;
  const std::string src_state = describe_tensor_state(src);
  const std::string dst_state = describe_tensor_state(dst);
  const int64_t bytes = dst.is_vulkan() ? convert(dst).gpu_nbytes() : -1;
  utils::log_vulkan_transition(utils::VulkanTransitionRequest{
      "layout_transition",
      transition_reason_for_materialize_reason(reason),
      transition_kind_for_materialize_reason(reason),
      bytes,
      host_transfer,
      !metadata_view,
      host_transfer,
      !metadata_view || host_transfer,
      op_name ? op_name : "unknown",
      vulkan_materialize_reason_name(reason),
      nullptr,
      nullptr,
      {},
      {src_state.c_str(), nullptr, nullptr, nullptr},
      {},
      {dst_state.c_str(), nullptr, nullptr, nullptr},
  });
}

bool is_raw_buffer_readback_legal(const vTensor& src) {
  return src.storage_type() == api::StorageType::BUFFER &&
      src.has_direct_buffer_layout() && src.storage_offset() == 0 &&
      src.gpu_numel() == src.numel() && !src.last_write_was_compute();
}

bool is_raw_buffer_readback_legal(
    const vTensor& src,
    const size_t staging_size_bytes) {
  return is_raw_buffer_readback_legal(src) &&
      src.gpu_nbytes() == staging_size_bytes;
}

bool is_buffer_snapshot_readback_legal(
    const vTensor& src,
    const size_t staging_size_bytes) {
  return src.storage_type() == api::StorageType::BUFFER &&
      src.storage_offset() >= 0 && src.buffer_length() >= 0 &&
      api::element_size(src.dtype()) *
              static_cast<size_t>(src.buffer_length()) <=
          staging_size_bytes;
}

bool requires_logical_pack_shader_for_readback(const vTensor& src) {
  const bool transfer_written_direct_buffer_snapshot =
      src.storage_type() == api::StorageType::BUFFER &&
      src.has_direct_buffer_layout() && src.storage_offset() == 0 &&
      !src.last_write_was_compute();
  return src.storage_type() == api::StorageType::BUFFER &&
      (src.dtype() == api::kFloat || src.dtype() == api::kByte) &&
      src.sizes().size() <= 4 && !is_raw_buffer_readback_legal(src) &&
      !transfer_written_direct_buffer_snapshot;
}

Tensor make_buffer_metadata_view_checked(
    const Tensor& base_arg,
    IntArrayRef sizes,
    IntArrayRef logical_strides,
    IntArrayRef physical_strides,
    const int64_t storage_offset,
    const char* producer_op) {
  Tensor base = base_arg.is_vulkan() ? base_arg : base_arg.vulkan();
  const vTensor& v_base = convert(base);

  const bool valid_view = utils::can_make_buffer_metadata_view(
      v_base, sizes, logical_strides, physical_strides, storage_offset);
  if (!valid_view) {
    std::ostringstream detail;
    detail << "base={" << describe_tensor_state(base) << "} sizes=" << sizes
           << " logical_strides=" << logical_strides
           << " physical_strides=" << physical_strides
           << " storage_offset=" << storage_offset;
    api::fail_vulkan(
        api::VulkanFailureClass::MetadataViewInvalid,
        producer_op ? producer_op : "make_buffer_metadata_view_checked",
        "MetadataViewInvalid",
        detail.str());
  }

  Tensor view = convert(vTensor{
      v_base,
      sizes.vec(),
      logical_strides.vec(),
      physical_strides.vec(),
      storage_offset,
  });
  validate_created_view(
      view,
      producer_op ? producer_op : "make_buffer_metadata_view_checked",
      "metadata_view");
  log_layout_transition(
      producer_op ? producer_op : "make_buffer_metadata_view_checked",
      VulkanMaterializeReason::MetadataViewCreated,
      base,
      view);
  log_tensor_state(
      view,
      VulkanTensorUse::Read,
      producer_op ? producer_op : "make_buffer_metadata_view_checked",
      "metadata_view");
  return record_tensor_alias_and_return(
      view,
      base,
      producer_op ? producer_op : "make_buffer_metadata_view_checked",
      "metadata_view");
}

Tensor make_typed_buffer_metadata_view_checked(
    const Tensor& base_arg,
    const ScalarType dtype,
    IntArrayRef sizes,
    IntArrayRef logical_strides,
    IntArrayRef physical_strides,
    const int64_t storage_offset,
    const int64_t buffer_length_override,
    const api::ExecutionLayout execution_layout,
    const char* producer_op) {
  Tensor base = base_arg.is_vulkan() ? base_arg : base_arg.vulkan();
  const vTensor& v_base = convert(base);

  TORCH_CHECK(
      api::uses_buffer_execution(execution_layout),
      "Typed Vulkan buffer metadata view requires a buffer execution layout");
  const bool valid_view = utils::can_make_typed_buffer_metadata_view(
      v_base,
      dtype,
      sizes,
      logical_strides,
      physical_strides,
      storage_offset,
      buffer_length_override);
  if (!valid_view) {
    std::ostringstream detail;
    detail << "base={" << describe_tensor_state(base) << "} dtype=" << dtype
           << " sizes=" << sizes << " logical_strides=" << logical_strides
           << " physical_strides=" << physical_strides
           << " storage_offset=" << storage_offset
           << " buffer_length=" << buffer_length_override;
    api::fail_vulkan(
        api::VulkanFailureClass::MetadataViewInvalid,
        producer_op ? producer_op : "make_typed_buffer_metadata_view_checked",
        "TypedMetadataViewInvalid",
        detail.str());
  }

  Tensor view = convert(vTensor{
      v_base,
      convert_dtype(dtype),
      sizes.vec(),
      logical_strides.vec(),
      physical_strides.vec(),
      storage_offset,
      buffer_length_override,
  });
  view = utils::mark_tensor_execution(view, execution_layout);
  validate_created_view(
      view,
      producer_op ? producer_op : "make_typed_buffer_metadata_view_checked",
      "typed_metadata_view");
  log_layout_transition(
      producer_op ? producer_op : "make_typed_buffer_metadata_view_checked",
      VulkanMaterializeReason::TypedMetadataViewCreated,
      base,
      view);
  log_tensor_state(
      view,
      VulkanTensorUse::Read,
      producer_op ? producer_op : "make_typed_buffer_metadata_view_checked",
      "typed_metadata_view");
  return record_tensor_alias_and_return(
      view,
      base,
      producer_op ? producer_op : "make_typed_buffer_metadata_view_checked",
      "typed_metadata_view");
}

Tensor materialize_vulkan_tensor(
    const Tensor& input,
    const VulkanLayoutTarget& target,
    const VulkanMaterializeReason reason,
    const char* producer_op) {
  Tensor output;
  if (target.storage_type == api::StorageType::BUFFER) {
    output = utils::ensure_buffer_storage(input, target.memory_layout);
    if (target.execution_layout != convert(output).execution_layout()) {
      output = utils::mark_tensor_execution(
          output, target.execution_layout, target.persistent);
    }
  } else {
    output = utils::ensure_texture_storage(
        input, target.memory_layout, target.storage_type);
  }

  if (target.require_direct && output.is_vulkan()) {
    const vTensor& v_output = convert(output);
    TORCH_CHECK(
        target.storage_type != api::StorageType::BUFFER ||
            v_output.has_direct_buffer_layout(),
        "Vulkan layout target requires direct buffer output, got {",
        describe_tensor_state(output),
        "}");
  }
  log_layout_transition(
      producer_op ? producer_op : "materialize_vulkan_tensor",
      reason,
      input,
      output);
  return record_tensor_write_and_return(
      output,
      producer_op ? producer_op : "materialize_vulkan_tensor",
      vulkan_materialize_reason_name(reason),
      {input});
}

Tensor ensure_vulkan_layout(
    const Tensor& input,
    const VulkanLayoutTarget& target,
    const VulkanMaterializeReason reason,
    const char* op_name) {
  if (input.is_vulkan()) {
    const vTensor& v_input = convert(input);
    const bool storage_matches = v_input.storage_type() == target.storage_type;
    const bool memory_matches =
        v_input.gpu_memory_layout() == target.memory_layout;
    const bool execution_matches =
        v_input.execution_layout() == target.execution_layout;
    const bool direct_matches =
        !target.require_direct || v_input.has_direct_buffer_layout();
    if (
        storage_matches && memory_matches && execution_matches &&
        direct_matches) {
      log_tensor_state(input, VulkanTensorUse::Read, op_name, "layout_reuse");
      return input;
    }
  }

  return materialize_vulkan_tensor(input, target, reason, op_name);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
