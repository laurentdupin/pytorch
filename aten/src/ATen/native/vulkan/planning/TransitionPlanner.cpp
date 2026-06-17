#include <ATen/native/vulkan/planning/TransitionPlanner.h>

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

const char* transition_log_path() {
  const char* env = std::getenv("PYTORCH_VULKAN_TRANSITION_LOG");
  return (env && *env) ? env : nullptr;
}

std::mutex& transition_log_mutex() {
  static std::mutex mutex;
  return mutex;
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

void append_json_string(
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

void append_json_bool(
    std::ofstream& out,
    const char* key,
    const bool value,
    bool& first) {
  if (!first) {
    out << ',';
  }
  first = false;
  out << '"' << key << "\":" << (value ? "true" : "false");
}

void append_json_i64(
    std::ofstream& out,
    const char* key,
    const int64_t value,
    bool& first) {
  if (!first) {
    out << ',';
  }
  first = false;
  out << '"' << key << "\":" << value;
}

} // namespace

bool transition_logging_enabled() {
  return transition_log_path() != nullptr;
}

VulkanTransitionAdmission classify_vulkan_transition(
    const VulkanTransitionRequest& request) {
  VulkanTransitionAdmission admission;
  admission.reason = request.reason;
  admission.kind = request.kind;
  admission.outcome =
      request.reason == TransitionReason::UnknownTransitionReason
      ? TransitionOutcome::Unknown
      : TransitionOutcome::Classified;
  admission.bytes = request.bytes;
  admission.host_transfer = request.host_transfer;
  admission.physical_copy = request.physical_copy;
  admission.sync_required = request.sync_required;
  admission.queue_submit_required = request.queue_submit_required;
  return admission;
}

void log_vulkan_transition(const VulkanTransitionRequest& request) {
  const char* path = transition_log_path();
  if (path == nullptr) {
    return;
  }

  const VulkanTransitionAdmission admission =
      classify_vulkan_transition(request);
  std::lock_guard<std::mutex> lock(transition_log_mutex());
  std::ofstream out(path, std::ios::app);
  bool first = true;
  out << '{';
  append_json_string(out, "event", "vulkan_transition", first);
  append_json_string(out, "phase", first_non_null(request.phase, "unknown"), first);
  append_json_string(
      out, "reason", transition_reason_name(admission.reason), first);
  append_json_string(out, "kind", transition_kind_name(admission.kind), first);
  append_json_string(
      out, "outcome", transition_outcome_name(admission.outcome), first);
  append_json_i64(out, "bytes", admission.bytes, first);
  append_json_bool(out, "host_transfer", admission.host_transfer, first);
  append_json_bool(out, "physical_copy", admission.physical_copy, first);
  append_json_bool(out, "sync_required", admission.sync_required, first);
  append_json_bool(
      out, "queue_submit_required", admission.queue_submit_required, first);
  append_json_string(
      out,
      "producer_schema",
      first_non_null(request.producer_schema, "unknown"),
      first);
  append_json_string(
      out,
      "consumer_schema",
      first_non_null(request.consumer_schema, "unknown"),
      first);
  append_json_string(
      out,
      "producer_contract",
      first_non_null(request.producer_contract, "unknown"),
      first);
  append_json_string(
      out,
      "consumer_contract",
      first_non_null(request.consumer_contract, "unknown"),
      first);
  append_json_string(
      out,
      "source_dtype",
      first_non_null(request.source_logical.dtype, "unknown"),
      first);
  append_json_string(
      out,
      "source_sizes",
      first_non_null(request.source_logical.sizes, "unknown"),
      first);
  append_json_string(
      out,
      "source_strides",
      first_non_null(request.source_logical.strides, "unknown"),
      first);
  append_json_string(
      out,
      "source_layout",
      first_non_null(request.source_physical.layout, "unknown"),
      first);
  append_json_string(
      out,
      "source_storage",
      first_non_null(request.source_physical.storage, "unknown"),
      first);
  append_json_string(
      out,
      "destination_dtype",
      first_non_null(request.destination_logical.dtype, "unknown"),
      first);
  append_json_string(
      out,
      "destination_sizes",
      first_non_null(request.destination_logical.sizes, "unknown"),
      first);
  append_json_string(
      out,
      "destination_strides",
      first_non_null(request.destination_logical.strides, "unknown"),
      first);
  append_json_string(
      out,
      "destination_layout",
      first_non_null(request.destination_physical.layout, "unknown"),
      first);
  append_json_string(
      out,
      "destination_storage",
      first_non_null(request.destination_physical.storage, "unknown"),
      first);
  out << "}\n";
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
