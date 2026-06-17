#pragma once

#ifdef USE_VULKAN_API

#include <cstdint>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

enum class TransitionReason : uint8_t {
  MetadataViewOnly,
  DescriptorViewOnly,
  RequiredSemanticClone,
  RequiredSemanticCat,
  RequiredContiguousMaterialization,
  RequiredConsumerLayout,
  RequiredLayoutRepack,
  RequiredDTypeCast,
  RequiredHostUpload,
  RequiredFinalReadback,
  RequiredDebugReadback,
  RequiredCorrectnessMaterialization,
  TemporaryRegionScratchCopy,
  AvoidableRedundantCopy,
  AvoidableLayoutChurn,
  UnexpectedCpuStaging,
  UnexpectedIntermediateReadback,
  FallbackMaterialization,
  UnsupportedStrideForConsumer,
  UnsupportedStorageOffsetForConsumer,
  MissingLifetimeProof,
  BudgetBlocked,
  UnknownTransitionReason,
};

enum class TransitionKind : uint8_t {
  Unknown,
  MetadataView,
  DescriptorView,
  DeviceCopy,
  HostTransfer,
  LayoutMaterialization,
  SemanticMaterialization,
  RegionLifetime,
  Fallback,
};

enum class TransitionOutcome : uint8_t {
  Observed,
  Classified,
  Unknown,
};

const char* transition_reason_name(TransitionReason reason);
const char* transition_kind_name(TransitionKind kind);
const char* transition_outcome_name(TransitionOutcome outcome);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
