#include <ATen/native/vulkan/planning/ExecutionContractDiagnostics.h>

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

const char* contract_admission_log_path() {
  const char* env = std::getenv("PYTORCH_VULKAN_CONTRACT_ADMISSION_LOG");
  return (env && *env) ? env : nullptr;
}

std::mutex& contract_admission_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

const char* admission_outcome_name(const ContractAdmissionOutcome outcome) {
  switch (outcome) {
    case ContractAdmissionOutcome::Accept:
      return "accept";
    case ContractAdmissionOutcome::Reject:
      return "reject";
    case ContractAdmissionOutcome::Skip:
      return "skip";
  }
  return "reject";
}

const char* admission_phase_name(const ContractAdmissionPhase phase) {
  switch (phase) {
    case ContractAdmissionPhase::Admitted:
      return "admitted";
    case ContractAdmissionPhase::GeneratedOptions:
      return "generated_options";
    case ContractAdmissionPhase::GeneratedBounds:
      return "generated_bounds";
    case ContractAdmissionPhase::GeneratedRelationship:
      return "generated_relationship";
    case ContractAdmissionPhase::HandwrittenPolicy:
      return "handwritten_policy";
    case ContractAdmissionPhase::MaterializationPolicy:
      return "materialization_policy";
    case ContractAdmissionPhase::ProvenanceLifetime:
      return "provenance_lifetime";
    case ContractAdmissionPhase::ResultAssembly:
      return "result_assembly";
  }
  return "handwritten_policy";
}

const char* first_non_null(const char* primary, const char* fallback) {
  return primary ? primary : (fallback ? fallback : "");
}

std::string json_escape(const char* value) {
  std::string escaped;
  for (const char c : std::string(first_non_null(value, ""))) {
    switch (c) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += c;
        break;
    }
  }
  return escaped;
}

void append_json_field(
    std::ofstream& out,
    const char* key,
    const char* value,
    bool& first) {
  if (!first) {
    out << ',';
  }
  first = false;
  out << '"' << key << "\":\"" << json_escape(value) << '"';
}

} // namespace

bool contract_admission_diagnostics_enabled() {
  return contract_admission_log_path() != nullptr;
}

void log_contract_admission(const ContractAdmissionDiagnostic& diagnostic) {
  const char* path = contract_admission_log_path();
  if (path == nullptr) {
    return;
  }

  const ExecutionContractMetadata* metadata = diagnostic.metadata;
  std::lock_guard<std::mutex> lock(contract_admission_log_mutex());
  std::ofstream out(path, std::ios::app);
  bool first = true;
  out << '{';
  append_json_field(out, "event", "vulkan_contract_admission", first);
  append_json_field(
      out,
      "contract_name",
      first_non_null(
          diagnostic.contract_name,
          metadata ? metadata->contract_name : nullptr),
      first);
  append_json_field(
      out,
      "family_name",
      first_non_null(
          diagnostic.family_name, metadata ? metadata->family_name : nullptr),
      first);
  append_json_field(
      out,
      "tuple_id",
      first_non_null(diagnostic.tuple_id, metadata ? metadata->tuple_id : nullptr),
      first);
  append_json_field(
      out, "outcome", admission_outcome_name(diagnostic.outcome), first);
  append_json_field(out, "phase", admission_phase_name(diagnostic.phase), first);
  append_json_field(out, "predicate", diagnostic.predicate, first);
  append_json_field(out, "reason_code", diagnostic.reason_code, first);
  append_json_field(out, "source", diagnostic.source, first);
  out << "}\n";
}

void log_contract_accept(
    const ExecutionContractMetadata* metadata,
    const char* predicate) {
  log_contract_admission(ContractAdmissionDiagnostic{
      metadata,
      nullptr,
      nullptr,
      nullptr,
      predicate,
      "matched",
      "handwritten",
      ContractAdmissionOutcome::Accept,
      ContractAdmissionPhase::Admitted});
}

void log_contract_reject(
    const ExecutionContractMetadata* metadata,
    const ContractAdmissionPhase phase,
    const char* predicate,
    const char* reason_code,
    const char* source) {
  log_contract_admission(ContractAdmissionDiagnostic{
      metadata,
      nullptr,
      nullptr,
      nullptr,
      predicate,
      reason_code,
      source,
      ContractAdmissionOutcome::Reject,
      phase});
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
