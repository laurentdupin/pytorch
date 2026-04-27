#include <ATen/native/vulkan/planning/ReplayTensorState.h>

#include <ATen/native/vulkan/api/Diagnostics.h>
#include <ATen/native/vulkan/ops/TensorState.h>

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

struct ReplayEpochState final {
  std::mutex mutex;
  std::unordered_map<const void*, uint64_t> generations;
  std::unordered_map<uint64_t, ReplayViewStamp> stamps_by_storage;
};

ReplayEpochState& replay_epoch_state() {
  static ReplayEpochState state;
  return state;
}

std::string replay_log_path() {
  const char* env = std::getenv("PYTORCH_VULKAN_REPLAY_LOG");
  return env ? std::string(env) : std::string();
}

std::mutex& replay_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

} // namespace

bool replay_logging_enabled() {
  return !replay_log_path().empty();
}

bool replay_materializes_escaping_outputs() {
  return true;
}

ReplayEpoch begin_replay_epoch(
    const void* session_identity,
    const char* allocation_label) {
  auto& state = replay_epoch_state();
  uint64_t generation = 0u;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    generation = ++state.generations[session_identity];
  }
  log_replay_event(
      "begin_epoch", session_identity, generation, allocation_label);
  return ReplayEpoch{session_identity, generation};
}

ReplayEpoch current_replay_epoch(const void* session_identity) {
  auto& state = replay_epoch_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  const auto it = state.generations.find(session_identity);
  return ReplayEpoch{
      session_identity,
      it == state.generations.end() ? 0u : it->second};
}

ReplayViewStamp stamp_replay_export(
    const Tensor& tensor,
    const void* session_identity,
    const uint32_t slot_id,
    const char* producer_op) {
  const ReplayEpoch epoch = current_replay_epoch(session_identity);
  ReplayViewStamp stamp{
      session_identity,
      slot_id,
      epoch.run_id,
      tensor_logical_desc_hash(tensor),
      tensor_storage_identity(tensor)};
  if (!replay_materializes_escaping_outputs() && stamp.storage_id != 0u) {
    auto& state = replay_epoch_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.stamps_by_storage[stamp.storage_id] = stamp;
  }
  std::ostringstream detail;
  detail << "slot=" << slot_id << " tensor={" << describe_tensor_state(tensor)
         << "}";
  log_replay_event(
      producer_op ? producer_op : "stamp_replay_export",
      session_identity,
      stamp.generation,
      nullptr,
      detail.str());
  return stamp;
}

void validate_replay_view_not_stale(
    const Tensor& tensor,
    const ReplayViewStamp& stamp,
    const char* consumer_op) {
  const ReplayEpoch epoch = current_replay_epoch(stamp.session_identity);
  const bool stale_generation = epoch.run_id != 0u &&
      stamp.generation != 0u && epoch.run_id != stamp.generation;
  const bool stale_descriptor =
      stamp.logical_desc_hash != 0u &&
      tensor_logical_desc_hash(tensor) != stamp.logical_desc_hash;
  const bool stale_storage = stamp.storage_id != 0u &&
      tensor_storage_identity(tensor) != stamp.storage_id;
  if (!stale_generation && !stale_descriptor && !stale_storage) {
    return;
  }

  std::ostringstream detail;
  detail << "slot=" << stamp.slot_id << " stamped_generation="
         << stamp.generation << " current_generation=" << epoch.run_id
         << " stamped_storage=0x" << std::hex << stamp.storage_id
         << " current_storage=0x" << tensor_storage_identity(tensor)
         << " stamped_hash=0x" << stamp.logical_desc_hash
         << " current_hash=0x" << tensor_logical_desc_hash(tensor)
         << std::dec << " tensor={" << describe_tensor_state(tensor) << "}";
  api::log_vulkan_failure(
      api::VulkanFailureClass::ReplayViewStale,
      consumer_op ? consumer_op : "validate_replay_view_not_stale",
      "ReplayViewStale",
      detail.str());
  TORCH_CHECK(
      false,
      api::format_vulkan_failure(
          api::VulkanFailureClass::ReplayViewStale,
          consumer_op ? consumer_op : "validate_replay_view_not_stale",
          "ReplayViewStale",
          detail.str()));
}

void validate_replay_tensor_not_stale(
    const Tensor& tensor,
    const char* consumer_op) {
  if (!tensor.defined()) {
    return;
  }
  const uint64_t storage_id = tensor_storage_identity(tensor);
  if (storage_id == 0u) {
    return;
  }

  ReplayViewStamp stamp;
  {
    auto& state = replay_epoch_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto it = state.stamps_by_storage.find(storage_id);
    if (it == state.stamps_by_storage.end()) {
      return;
    }
    stamp = it->second;
  }
  validate_replay_view_not_stale(tensor, stamp, consumer_op);
}

void clear_replay_tensor_stamp(const Tensor& tensor) {
  if (!tensor.defined()) {
    return;
  }
  const uint64_t storage_id = tensor_storage_identity(tensor);
  if (storage_id == 0u) {
    return;
  }

  auto& state = replay_epoch_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.stamps_by_storage.erase(storage_id);
}

void log_replay_event(
    const char* event,
    const void* session_identity,
    const uint64_t generation,
    const char* allocation_label,
    const std::string& detail) {
  if (!replay_logging_enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(replay_log_mutex());
  std::ofstream out(replay_log_path(), std::ios::app);
  out << "vulkan_replay event=" << (event ? event : "unknown")
      << " session=" << session_identity << " generation=" << generation;
  if (allocation_label && allocation_label[0] != '\0') {
    out << " label=" << allocation_label;
  }
  if (!detail.empty()) {
    out << " detail={" << detail << "}";
  }
  out << '\n';
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
