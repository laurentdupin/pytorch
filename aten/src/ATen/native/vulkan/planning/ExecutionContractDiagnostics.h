#pragma once

#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <cstdint>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

enum class ContractAdmissionOutcome : uint8_t {
  Accept,
  Reject,
  Skip,
};

enum class ContractAdmissionPhase : uint8_t {
  Admitted,
  GeneratedOptions,
  GeneratedBounds,
  GeneratedRelationship,
  HandwrittenPolicy,
  MaterializationPolicy,
  ProvenanceLifetime,
  ResultAssembly,
};

struct ContractAdmissionDiagnostic final {
  const ExecutionContractMetadata* metadata{nullptr};
  const char* contract_name{nullptr};
  const char* family_name{nullptr};
  const char* tuple_id{nullptr};
  const char* predicate{nullptr};
  const char* reason_code{nullptr};
  const char* source{nullptr};
  ContractAdmissionOutcome outcome{ContractAdmissionOutcome::Reject};
  ContractAdmissionPhase phase{ContractAdmissionPhase::HandwrittenPolicy};
};

bool contract_admission_diagnostics_enabled();

void log_contract_admission(const ContractAdmissionDiagnostic& diagnostic);

void log_contract_accept(
    const ExecutionContractMetadata* metadata,
    const char* predicate);

void log_contract_reject(
    const ExecutionContractMetadata* metadata,
    ContractAdmissionPhase phase,
    const char* predicate,
    const char* reason_code,
    const char* source);

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
