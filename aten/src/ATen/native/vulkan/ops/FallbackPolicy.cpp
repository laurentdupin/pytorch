#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Diagnostics.h>
#include <ATen/native/vulkan/api/Sync.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/planning/TransitionPlanner.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <atomic>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

std::atomic<uint64_t>& cpu_fallback_counter() {
  static std::atomic<uint64_t> counter{0};
  return counter;
}

std::atomic<uint64_t>& sync_readback_counter() {
  static std::atomic<uint64_t> counter{0};
  return counter;
}

struct VulkanFallbackPhaseCounters final {
  std::atomic<uint64_t> unknown{0};
  std::atomic<uint64_t> model_setup{0};
  std::atomic<uint64_t> owner_context_create{0};
  std::atomic<uint64_t> owner_forward{0};
  std::atomic<uint64_t> decoder_setup{0};
  std::atomic<uint64_t> positional_embedding_setup{0};
  std::atomic<uint64_t> readback{0};
  std::atomic<uint64_t> test_harness{0};
};

VulkanFallbackPhaseCounters& fallback_phase_counters() {
  static VulkanFallbackPhaseCounters counters;
  return counters;
}

VulkanFallbackPhaseCounters& timed_fallback_phase_counters() {
  static VulkanFallbackPhaseCounters counters;
  return counters;
}

VulkanFallbackPhase& fallback_phase_tls() {
  thread_local VulkanFallbackPhase phase = VulkanFallbackPhase::Unknown;
  return phase;
}

bool& timed_region_tls() {
  thread_local bool enabled = false;
  return enabled;
}

std::atomic<uint64_t>& fallback_phase_counter(
    VulkanFallbackPhaseCounters& counters,
    const VulkanFallbackPhase phase) {
  switch (phase) {
    case VulkanFallbackPhase::ModelSetup:
      return counters.model_setup;
    case VulkanFallbackPhase::OwnerContextCreate:
      return counters.owner_context_create;
    case VulkanFallbackPhase::OwnerForward:
      return counters.owner_forward;
    case VulkanFallbackPhase::DecoderSetup:
      return counters.decoder_setup;
    case VulkanFallbackPhase::PositionalEmbeddingSetup:
      return counters.positional_embedding_setup;
    case VulkanFallbackPhase::Readback:
      return counters.readback;
    case VulkanFallbackPhase::TestHarness:
      return counters.test_harness;
    case VulkanFallbackPhase::Unknown:
      return counters.unknown;
  }
  return counters.unknown;
}

bool env_flag_enabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  const std::string s(value);
  return !(s.empty() || s == "0" || s == "false" || s == "False" ||
           s == "FALSE" || s == "off" || s == "OFF");
}

std::string env_string(const char* name) {
  const char* value = std::getenv(name);
  return value ? std::string(value) : std::string();
}

bool cpu_fallback_log_enabled(const VulkanCpuFallbackKind kind) {
  return env_flag_enabled("PYTORCH_VULKAN_LOG_CPU_FALLBACK") ||
      env_flag_enabled("PYTORCH_VULKAN_LOG_FALLBACK") ||
      (kind == VulkanCpuFallbackKind::SyncReadback &&
       env_flag_enabled("PYTORCH_VULKAN_LOG_SYNC_READBACK"));
}

bool cpu_fallback_warn_enabled(const VulkanCpuFallbackKind kind) {
  const std::string policy = env_string("PYTORCH_VULKAN_CPU_FALLBACK");
  return policy == "warn" || policy == "WARN" || cpu_fallback_log_enabled(kind);
}

bool cpu_fallback_error_enabled(const VulkanCpuFallbackKind kind) {
  if (env_flag_enabled("PYTORCH_VULKAN_NO_CPU_FALLBACK")) {
    return true;
  }
  if (
      kind == VulkanCpuFallbackKind::SyncReadback &&
      env_flag_enabled("PYTORCH_VULKAN_FAIL_ON_SYNC_READBACK")) {
    return true;
  }
  const std::string policy = env_string("PYTORCH_VULKAN_CPU_FALLBACK");
  return policy == "error" || policy == "ERROR" || policy == "fail" ||
      policy == "FAIL";
}

const char* fallback_kind_name(const VulkanCpuFallbackKind kind) {
  switch (kind) {
    case VulkanCpuFallbackKind::OpFallback:
      return "cpu_fallback";
    case VulkanCpuFallbackKind::SyncReadback:
      return "sync_readback";
  }
  return "cpu_fallback";
}

const char* phase_name(const VulkanFallbackPhase phase) {
  switch (phase) {
    case VulkanFallbackPhase::Unknown:
      return "unknown";
    case VulkanFallbackPhase::ModelSetup:
      return "model_setup";
    case VulkanFallbackPhase::OwnerContextCreate:
      return "owner_context_create";
    case VulkanFallbackPhase::OwnerForward:
      return "owner_forward";
    case VulkanFallbackPhase::DecoderSetup:
      return "decoder_setup";
    case VulkanFallbackPhase::PositionalEmbeddingSetup:
      return "positional_embedding_setup";
    case VulkanFallbackPhase::Readback:
      return "readback";
    case VulkanFallbackPhase::TestHarness:
      return "test_harness";
  }
  return "unknown";
}

std::string tensor_detail(const Tensor& tensor) {
  std::ostringstream out;
  out << "device=" << tensor.device() << " dtype=" << tensor.scalar_type()
      << " sizes=" << tensor.sizes();
  if (tensor.is_vulkan()) {
    out << " layout=" << tensor.layout();
  }
  return out.str();
}

std::string fallback_detail(
    const VulkanCpuFallbackKind kind,
    ArrayRef<Tensor> tensors) {
  std::ostringstream out;
  out << "kind=" << fallback_kind_name(kind)
      << " phase=" << phase_name(fallback_phase_tls())
      << " inside_timed_forward=" << (timed_region_tls() ? 1 : 0);
  for (const auto i : c10::irange(tensors.size())) {
    out << " tensor" << i << "={" << tensor_detail(tensors[i]) << '}';
  }
  return out.str();
}

bool has_vulkan_tensor(ArrayRef<Tensor> tensors) {
  for (const Tensor& tensor : tensors) {
    if (tensor.defined() && tensor.is_vulkan()) {
      return true;
    }
  }
  return false;
}

int64_t fallback_tensor_bytes(ArrayRef<Tensor> tensors) {
  int64_t bytes = 0;
  bool found = false;
  for (const Tensor& tensor : tensors) {
    if (tensor.defined() && tensor.is_vulkan()) {
      bytes += static_cast<int64_t>(tensor.nbytes());
      found = true;
    }
  }
  return found ? bytes : -1;
}

void log_fallback_transition(
    const char* op_name,
    const char* reason,
    ArrayRef<Tensor> tensors,
    const VulkanCpuFallbackKind kind) {
  if (!utils::transition_logging_enabled()) {
    return;
  }
  utils::log_vulkan_transition(utils::VulkanTransitionRequest{
      phase_name(fallback_phase_tls()),
      utils::TransitionReason::FallbackMaterialization,
      utils::TransitionKind::Fallback,
      fallback_tensor_bytes(tensors),
      kind == VulkanCpuFallbackKind::SyncReadback,
      kind == VulkanCpuFallbackKind::SyncReadback,
      kind == VulkanCpuFallbackKind::SyncReadback,
      kind == VulkanCpuFallbackKind::SyncReadback,
      op_name ? op_name : "unknown",
      reason ? reason : fallback_kind_name(kind),
      nullptr,
      nullptr,
      {},
      {},
      {},
      {},
  });
}

} // namespace

uint64_t vulkan_cpu_fallback_count() {
  return cpu_fallback_counter().load(std::memory_order_relaxed);
}

uint64_t vulkan_sync_readback_count() {
  return sync_readback_counter().load(std::memory_order_relaxed);
}

void reset_vulkan_fallback_counters() {
  cpu_fallback_counter().store(0, std::memory_order_relaxed);
  sync_readback_counter().store(0, std::memory_order_relaxed);
}

std::vector<int64_t> vulkan_fallback_phase_counters_snapshot() {
  const auto& counters = fallback_phase_counters();
  return {
      static_cast<int64_t>(counters.unknown.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.model_setup.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.owner_context_create.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.owner_forward.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.decoder_setup.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.positional_embedding_setup.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.readback.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.test_harness.load(std::memory_order_relaxed)),
  };
}

std::vector<int64_t> vulkan_timed_fallback_phase_counters_snapshot() {
  const auto& counters = timed_fallback_phase_counters();
  return {
      static_cast<int64_t>(counters.unknown.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.model_setup.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.owner_context_create.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.owner_forward.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.decoder_setup.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.positional_embedding_setup.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.readback.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.test_harness.load(std::memory_order_relaxed)),
  };
}

void reset_vulkan_fallback_phase_counters() {
  auto& counters = fallback_phase_counters();
  counters.unknown.store(0, std::memory_order_relaxed);
  counters.model_setup.store(0, std::memory_order_relaxed);
  counters.owner_context_create.store(0, std::memory_order_relaxed);
  counters.owner_forward.store(0, std::memory_order_relaxed);
  counters.decoder_setup.store(0, std::memory_order_relaxed);
  counters.positional_embedding_setup.store(0, std::memory_order_relaxed);
  counters.readback.store(0, std::memory_order_relaxed);
  counters.test_harness.store(0, std::memory_order_relaxed);

  auto& timed_counters = timed_fallback_phase_counters();
  timed_counters.unknown.store(0, std::memory_order_relaxed);
  timed_counters.model_setup.store(0, std::memory_order_relaxed);
  timed_counters.owner_context_create.store(0, std::memory_order_relaxed);
  timed_counters.owner_forward.store(0, std::memory_order_relaxed);
  timed_counters.decoder_setup.store(0, std::memory_order_relaxed);
  timed_counters.positional_embedding_setup.store(0, std::memory_order_relaxed);
  timed_counters.readback.store(0, std::memory_order_relaxed);
  timed_counters.test_harness.store(0, std::memory_order_relaxed);
}

void set_vulkan_fallback_phase(const VulkanFallbackPhase phase) {
  fallback_phase_tls() = phase;
}

void set_vulkan_benchmark_timed_region(const bool enabled) {
  timed_region_tls() = enabled;
}

VulkanFallbackPhase current_vulkan_fallback_phase() {
  return fallback_phase_tls();
}

bool current_vulkan_benchmark_timed_region() {
  return timed_region_tls();
}

const char* vulkan_fallback_phase_name(const VulkanFallbackPhase phase) {
  return phase_name(phase);
}

void report_vulkan_cpu_fallback(
    const char* op_name,
    const char* reason,
    ArrayRef<Tensor> tensors,
    const VulkanCpuFallbackKind kind) {
  if (!tensors.empty() && !has_vulkan_tensor(tensors)) {
    return;
  }

  if (kind == VulkanCpuFallbackKind::SyncReadback) {
    sync_readback_counter().fetch_add(1, std::memory_order_relaxed);
    api::vulkan_sync_counters().fallback_sync_readback_count.fetch_add(
        1, std::memory_order_relaxed);
  } else {
    cpu_fallback_counter().fetch_add(1, std::memory_order_relaxed);
  }
  fallback_phase_counter(fallback_phase_counters(), fallback_phase_tls())
      .fetch_add(1, std::memory_order_relaxed);
  if (timed_region_tls()) {
    fallback_phase_counter(timed_fallback_phase_counters(), fallback_phase_tls())
        .fetch_add(1, std::memory_order_relaxed);
  }
  log_fallback_transition(op_name, reason, tensors, kind);

  const std::string detail = fallback_detail(kind, tensors);
  const std::string message = api::format_vulkan_failure(
      api::VulkanFailureClass::Unsupported, op_name, reason, detail);
  if (cpu_fallback_warn_enabled(kind)) {
    TORCH_WARN(message);
  }
  if (cpu_fallback_log_enabled(kind)) {
    api::log_vulkan_failure(
        api::VulkanFailureClass::Unsupported, op_name, reason, detail);
  }
  if (cpu_fallback_error_enabled(kind)) {
    api::fail_vulkan(
        api::VulkanFailureClass::Unsupported, op_name, reason, detail);
  }
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
