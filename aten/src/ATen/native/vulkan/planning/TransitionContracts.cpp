#include <ATen/native/vulkan/planning/TransitionContracts.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

const char* transition_reason_name(const TransitionReason reason) {
  switch (reason) {
    case TransitionReason::MetadataViewOnly:
      return "metadata_view_only";
    case TransitionReason::DescriptorViewOnly:
      return "descriptor_view_only";
    case TransitionReason::RequiredSemanticClone:
      return "required_semantic_clone";
    case TransitionReason::RequiredSemanticCat:
      return "required_semantic_cat";
    case TransitionReason::RequiredContiguousMaterialization:
      return "required_contiguous_materialization";
    case TransitionReason::RequiredConsumerLayout:
      return "required_consumer_layout";
    case TransitionReason::RequiredLayoutRepack:
      return "required_layout_repack";
    case TransitionReason::RequiredDTypeCast:
      return "required_dtype_cast";
    case TransitionReason::RequiredHostUpload:
      return "required_host_upload";
    case TransitionReason::RequiredFinalReadback:
      return "required_final_readback";
    case TransitionReason::RequiredDebugReadback:
      return "required_debug_readback";
    case TransitionReason::RequiredCorrectnessMaterialization:
      return "required_correctness_materialization";
    case TransitionReason::TemporaryRegionScratchCopy:
      return "temporary_region_scratch_copy";
    case TransitionReason::AvoidableRedundantCopy:
      return "avoidable_redundant_copy";
    case TransitionReason::AvoidableLayoutChurn:
      return "avoidable_layout_churn";
    case TransitionReason::UnexpectedCpuStaging:
      return "unexpected_cpu_staging";
    case TransitionReason::UnexpectedIntermediateReadback:
      return "unexpected_intermediate_readback";
    case TransitionReason::FallbackMaterialization:
      return "fallback_materialization";
    case TransitionReason::UnsupportedStrideForConsumer:
      return "unsupported_stride_for_consumer";
    case TransitionReason::UnsupportedStorageOffsetForConsumer:
      return "unsupported_storage_offset_for_consumer";
    case TransitionReason::MissingLifetimeProof:
      return "missing_lifetime_proof";
    case TransitionReason::BudgetBlocked:
      return "budget_blocked";
    case TransitionReason::UnknownTransitionReason:
      return "unknown_transition_reason";
  }
  return "unknown_transition_reason";
}

const char* transition_kind_name(const TransitionKind kind) {
  switch (kind) {
    case TransitionKind::Unknown:
      return "unknown";
    case TransitionKind::MetadataView:
      return "metadata_view";
    case TransitionKind::DescriptorView:
      return "descriptor_view";
    case TransitionKind::DeviceCopy:
      return "device_copy";
    case TransitionKind::HostTransfer:
      return "host_transfer";
    case TransitionKind::LayoutMaterialization:
      return "layout_materialization";
    case TransitionKind::SemanticMaterialization:
      return "semantic_materialization";
    case TransitionKind::RegionLifetime:
      return "region_lifetime";
    case TransitionKind::Fallback:
      return "fallback";
  }
  return "unknown";
}

const char* transition_outcome_name(const TransitionOutcome outcome) {
  switch (outcome) {
    case TransitionOutcome::Observed:
      return "observed";
    case TransitionOutcome::Classified:
      return "classified";
    case TransitionOutcome::Unknown:
      return "unknown";
  }
  return "unknown";
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
