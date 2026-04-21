#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/planning/Capabilities.h>

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

enum class RealizationKind : uint8_t {
  ExternalInput = 0u,
  Constant,
  Materialized,
  View,
  InPlaceVersion,
  Virtual,
};

enum class BoundaryRole : uint8_t {
  Internal = 0u,
  RegionOutput,
};

enum class StageKind : uint8_t {
  ImageEntry = 0u,
  Backbone,
  Capture,
  Decoder,
  Export,
  Unknown,
};

enum class ExecOpcode : uint8_t {
  Dispatch = 0u,
  Copy,
  Fill,
  Barrier,
  Export,
};

enum class DispatchKind : uint8_t {
  PatchEmbed = 0u,
  ImagePatchTokenInput,
  FeatureMapToTokens,
  ElementwiseAdd,
  Concat,
  PatchTokenInput,
  BackboneBlock,
  CapturePatchTokens,
  CaptureNormedPatchTokens,
  CaptureDecoderLayerPreprocess,
  DecoderLayerPreprocess,
  TokensToFeatureMap,
  DecoderProject,
  DecoderResize,
  DecoderPreprocess,
  DecoderHead,
  Unknown,
};

enum class ViewTransformKind : uint8_t {
  Identity = 0u,
  Reshape,
  Slice,
  Reinterpret,
  Opaque,
};

enum class MemoryClass : uint8_t {
  DeviceLocal = 0u,
  HostVisible,
  External,
};

struct LayoutContract final {
  ScalarType dtype{kFloat};
  api::StorageType storage_type{api::StorageType::BUFFER};
  api::GPUMemoryLayout memory_layout{
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  api::ExecutionLayout execution_layout{api::ExecutionLayout::BUFFER_DIRECT};
  int64_t width_alignment{1};
  bool pad_width{false};
  std::string capability_key;
  std::string debug_name;
};

struct PhysicalSlot final {
  size_t id{0u};
  std::optional<size_t> source_memory_slot;
  uint64_t byte_offset{0u};
  uint64_t byte_size{0u};
  ScalarType storage_dtype{kFloat};
  api::StorageType storage_type{api::StorageType::BUFFER};
  api::GPUMemoryLayout storage_layout{
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED};
  std::vector<int64_t> physical_sizes;
  MemoryClass memory_class{MemoryClass::DeviceLocal};
  uint64_t alignment{1u};
  bool dedicated{false};
  bool external{false};
};

struct ViewDesc final {
  std::optional<size_t> slot;
  ScalarType logical_dtype{kFloat};
  std::vector<int64_t> logical_sizes;
  std::vector<int64_t> logical_strides;
  int64_t storage_offset{0};
  ViewTransformKind transform{ViewTransformKind::Identity};
};

struct LoweredValue final {
  uint32_t ir_value{0u};
  std::string name;
  RealizationKind realization{RealizationKind::Materialized};
  BoundaryRole boundary_role{BoundaryRole::Internal};
  std::optional<size_t> slot;
  std::optional<uint32_t> base;
  ViewDesc view;
  uint32_t first_use_step{0u};
  uint32_t last_use_step{0u};
};

struct WorkspaceRef final {
  uint32_t id{0u};
  uint64_t bytes{0u};
  uint64_t alignment{1u};
};

struct DispatchStep final {
  uint32_t ir_op_index{0u};
  std::string name;
  std::string program_key;
  DispatchKind dispatch_kind{DispatchKind::Unknown};
  std::vector<uint32_t> reads;
  std::vector<uint32_t> constants;
  std::vector<uint32_t> temporaries;
  std::vector<uint32_t> writes;
  std::vector<WorkspaceRef> workspaces;
  std::string attributes_key;
};

struct CopyStep final {
  uint32_t source{0u};
  uint32_t destination{0u};
  std::string reason;
};

struct FillStep final {
  uint32_t target{0u};
  std::string reason;
};

struct BarrierStep final {
  std::vector<uint32_t> reads;
  std::vector<uint32_t> writes;
  std::string reason;
};

struct ExportStep final {
  uint32_t value{0u};
  size_t output_index{0u};
  std::string name;
};

struct ExecStep final {
  ExecOpcode opcode{ExecOpcode::Dispatch};
  std::variant<DispatchStep, CopyStep, FillStep, BarrierStep, ExportStep>
      payload;
};

struct StageRange final {
  StageKind kind{StageKind::Unknown};
  uint32_t begin_step{0u};
  uint32_t end_step{0u};
  std::optional<uint64_t> prerecorded_secondary;
};

struct RegionOutputBinding final {
  uint32_t value{0u};
  size_t output_index{0u};
  std::string name;
};

struct VulkanExecutableRegion final {
  std::string key;
  LayoutContract contract;
  std::vector<PhysicalSlot> slots;
  std::vector<LoweredValue> values;
  std::vector<ExecStep> steps;
  std::vector<StageRange> stages;
  std::vector<RegionOutputBinding> outputs;

  bool defined() const {
    return !values.empty() || !steps.empty() || !outputs.empty();
  }
};

inline const char* realization_kind_name(const RealizationKind kind) {
  switch (kind) {
    case RealizationKind::ExternalInput:
      return "ExternalInput";
    case RealizationKind::Constant:
      return "Constant";
    case RealizationKind::Materialized:
      return "Materialized";
    case RealizationKind::View:
      return "View";
    case RealizationKind::InPlaceVersion:
      return "InPlaceVersion";
    case RealizationKind::Virtual:
      return "Virtual";
  }
  return "UnknownRealizationKind";
}

inline const char* boundary_role_name(const BoundaryRole role) {
  switch (role) {
    case BoundaryRole::Internal:
      return "Internal";
    case BoundaryRole::RegionOutput:
      return "RegionOutput";
  }
  return "UnknownBoundaryRole";
}

inline const char* stage_kind_name(const StageKind kind) {
  switch (kind) {
    case StageKind::ImageEntry:
      return "ImageEntry";
    case StageKind::Backbone:
      return "Backbone";
    case StageKind::Capture:
      return "Capture";
    case StageKind::Decoder:
      return "Decoder";
    case StageKind::Export:
      return "Export";
    case StageKind::Unknown:
      return "Unknown";
  }
  return "UnknownStageKind";
}

inline const char* exec_opcode_name(const ExecOpcode opcode) {
  switch (opcode) {
    case ExecOpcode::Dispatch:
      return "Dispatch";
    case ExecOpcode::Copy:
      return "Copy";
    case ExecOpcode::Fill:
      return "Fill";
    case ExecOpcode::Barrier:
      return "Barrier";
    case ExecOpcode::Export:
      return "Export";
  }
  return "UnknownExecOpcode";
}

inline const char* dispatch_kind_name(const DispatchKind kind) {
  switch (kind) {
    case DispatchKind::PatchEmbed:
      return "PatchEmbed";
    case DispatchKind::ImagePatchTokenInput:
      return "ImagePatchTokenInput";
    case DispatchKind::FeatureMapToTokens:
      return "FeatureMapToTokens";
    case DispatchKind::ElementwiseAdd:
      return "ElementwiseAdd";
    case DispatchKind::Concat:
      return "Concat";
    case DispatchKind::PatchTokenInput:
      return "PatchTokenInput";
    case DispatchKind::BackboneBlock:
      return "BackboneBlock";
    case DispatchKind::CaptureNormedPatchTokens:
      return "CaptureNormedPatchTokens";
    case DispatchKind::CaptureDecoderLayerPreprocess:
      return "CaptureDecoderLayerPreprocess";
    case DispatchKind::DecoderLayerPreprocess:
      return "DecoderLayerPreprocess";
    case DispatchKind::TokensToFeatureMap:
      return "TokensToFeatureMap";
    case DispatchKind::DecoderProject:
      return "DecoderProject";
    case DispatchKind::DecoderResize:
      return "DecoderResize";
    case DispatchKind::DecoderPreprocess:
      return "DecoderPreprocess";
    case DispatchKind::DecoderHead:
      return "DecoderHead";
    case DispatchKind::Unknown:
      return "Unknown";
  }
  return "UnknownDispatchKind";
}

inline const char* view_transform_kind_name(const ViewTransformKind kind) {
  switch (kind) {
    case ViewTransformKind::Identity:
      return "Identity";
    case ViewTransformKind::Reshape:
      return "Reshape";
    case ViewTransformKind::Slice:
      return "Slice";
    case ViewTransformKind::Reinterpret:
      return "Reinterpret";
    case ViewTransformKind::Opaque:
      return "Opaque";
  }
  return "UnknownViewTransformKind";
}

inline const char* memory_class_name(const MemoryClass kind) {
  switch (kind) {
    case MemoryClass::DeviceLocal:
      return "DeviceLocal";
    case MemoryClass::HostVisible:
      return "HostVisible";
    case MemoryClass::External:
      return "External";
  }
  return "UnknownMemoryClass";
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
