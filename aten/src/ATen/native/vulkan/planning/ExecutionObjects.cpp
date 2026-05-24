#include <ATen/native/vulkan/planning/ExecutionObjects.h>
#include <ATen/native/vulkan/planning/Runtime.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/narrow.h>
#include <ATen/ops/zeros.h>
#endif

#include <ATen/native/vulkan/ops/InferenceCache.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <c10/core/Storage.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

using namespace api::utils;

constexpr size_t kPackedWeightResidencyMaxEntries = 1024u;
constexpr size_t kPackedWeightResidencyLimitBytes =
    size_t{2} * 1024u * 1024u * 1024u;
constexpr size_t kLinearContextCacheSize = 128u;
constexpr size_t kExecutionObjectCacheSize = 64u;
constexpr uint64_t kLinearContextPruneInterval = 64u;
constexpr size_t kLinearContextPruneScanBudget = 32u;
constexpr size_t kLinearContextPruneEraseBudget = 8u;

template <typename T>
void hash_combine(size_t& seed, const T& value) {
  seed ^= std::hash<T>{}(value) + size_t{0x9e3779b97f4a7c15ull} + (seed << 6u) +
      (seed >> 2u);
}

void hash_combine_sizes(size_t& seed, const std::vector<int64_t>& sizes) {
  hash_combine(seed, sizes.size());
  for (const int64_t size : sizes) {
    hash_combine(seed, size);
  }
}

std::string format_size_list(const std::vector<int64_t>& sizes) {
  std::ostringstream stream;
  stream << "[";
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (i > 0u) {
      stream << ",";
    }
    stream << sizes[i];
  }
  stream << "]";
  return stream.str();
}

using TensorWeakRef = c10::weak_intrusive_ptr<TensorImpl, UndefinedTensorImpl>;
using StorageWeakRef = c10::weak_intrusive_ptr<c10::StorageImpl>;
using VulkanStorageWeakRef = std::weak_ptr<const vTensorStorage>;

std::optional<TensorWeakRef> make_tensor_weak_ref(const Tensor& tensor) {
  if (!tensor.defined()) {
    return std::nullopt;
  }
  return TensorWeakRef(tensor.getIntrusivePtr());
}

TensorBase packed_weight_identity_tensor(const Tensor& tensor) {
  if (tensor.defined() && tensor.is_view()) {
    const TensorBase base = tensor._base();
    if (base.defined()) {
      return base;
    }
  }
  return tensor;
}

std::optional<TensorWeakRef> make_packed_weight_weak_ref(
    const Tensor& tensor) {
  if (!tensor.defined()) {
    return std::nullopt;
  }
  const TensorBase identity = packed_weight_identity_tensor(tensor);
  return TensorWeakRef(identity.getIntrusivePtr());
}

bool packed_weight_ref_matches_tensor(
    const std::optional<TensorWeakRef>& ref,
    const Tensor& tensor) {
  if (!ref.has_value() || !tensor.defined()) {
    return false;
  }
  const TensorBase identity = packed_weight_identity_tensor(tensor);
  return !ref->expired() &&
      ref->_unsafe_get_target() == identity.unsafeGetTensorImpl();
}

std::optional<StorageWeakRef> make_storage_weak_ref(const Tensor& tensor) {
  if (!tensor.defined() || !tensor.unsafeGetTensorImpl()->has_storage()) {
    return std::nullopt;
  }
  return tensor.storage().getWeakStorageImpl();
}

std::optional<VulkanStorageWeakRef> make_vulkan_storage_weak_ref(
    const Tensor& tensor) {
  if (!tensor.defined() || !tensor.is_vulkan()) {
    return std::nullopt;
  }
  return convert(tensor).storage_weak_ref();
}

const void* tensor_storage_identity_ptr(const Tensor& tensor) {
  if (!tensor.defined() || !tensor.unsafeGetTensorImpl()->has_storage()) {
    return nullptr;
  }
  return static_cast<const void*>(tensor.storage().unsafeGetStorageImpl());
}

const void* tensor_vulkan_storage_identity_ptr(const Tensor& tensor) {
  if (!tensor.defined() || !tensor.is_vulkan()) {
    return nullptr;
  }
  return convert(tensor).storage_identity();
}

uint64_t tensor_packed_weight_source_key(const Tensor& tensor) {
  if (!tensor.defined() || !tensor.is_vulkan()) {
    return 0u;
  }
  return tensor_provenance_first_input_key(tensor);
}

bool weak_ref_matches_tensor(
    const std::optional<TensorWeakRef>& ref,
    const Tensor& tensor) {
  if (!ref.has_value() || !tensor.defined()) {
    return false;
  }
  return !ref->expired() && ref->_unsafe_get_target() ==
      tensor.unsafeGetTensorImpl();
}

bool weak_ref_matches_optional_tensor(
    const std::optional<TensorWeakRef>& ref,
    const std::optional<Tensor>& tensor) {
  if (ref.has_value() != tensor.has_value()) {
    return false;
  }
  if (!ref.has_value()) {
    return true;
  }
  return tensor.has_value() && !ref->expired() &&
      ref->_unsafe_get_target() == tensor->unsafeGetTensorImpl();
}

bool same_weak_tensor_ref(
    const std::optional<TensorWeakRef>& lhs,
    const std::optional<TensorWeakRef>& rhs) {
  if (lhs.has_value() != rhs.has_value()) {
    return false;
  }
  if (!lhs.has_value()) {
    return true;
  }
  return !lhs->expired() && !rhs->expired() &&
      lhs->_unsafe_get_target() == rhs->_unsafe_get_target();
}

bool weak_tensor_ref_alive(const std::optional<TensorWeakRef>& ref) {
  return ref.has_value() && !ref->expired();
}

bool weak_storage_ref_alive(const std::optional<StorageWeakRef>& ref) {
  return ref.has_value() && !ref->expired();
}

bool weak_vulkan_storage_ref_alive(
    const std::optional<VulkanStorageWeakRef>& ref) {
  return ref.has_value() && !ref->expired();
}

bool optional_weak_tensor_ref_alive(const std::optional<TensorWeakRef>& ref) {
  return !ref.has_value() || !ref->expired();
}

const void* tensor_identity_ptr(const Tensor& tensor) {
  return tensor.defined() ? static_cast<const void*>(tensor.unsafeGetTensorImpl())
                          : nullptr;
}

size_t align_up_size(const size_t value, const size_t alignment) {
  if (alignment <= 1u) {
    return value;
  }
  const size_t remainder = value % alignment;
  return remainder == 0u ? value : (value + alignment - remainder);
}

Tensor create_execution_object_storage(
    const std::vector<int64_t>& sizes,
    const ScalarType dtype,
    const api::ExecutionLayout execution_layout,
    const api::GPUMemoryLayout memory_layout,
    const api::StorageType storage_type,
    const bool persistent) {
  Tensor storage =
      at::zeros(sizes, at::device(at::kVulkan).dtype(dtype));
  if (api::uses_buffer_execution(execution_layout)) {
    storage = ensure_buffer_storage(storage, memory_layout);
  } else {
    storage = ensure_texture_storage(storage, memory_layout, storage_type);
  }
  return mark_tensor_execution(storage, execution_layout, persistent);
}

struct PackedWeightResidencyEntry final {
  std::optional<TensorWeakRef> weight_ref;
  std::optional<TensorWeakRef> bias_ref;
  std::optional<StorageWeakRef> weight_storage_ref;
  std::optional<VulkanStorageWeakRef> weight_vulkan_storage_ref;
  const void* weight_storage_identity;
  const void* weight_vulkan_storage_identity;
  uint64_t weight_source_key;
  int64_t weight_storage_offset;
  std::vector<int64_t> weight_strides;
  c10::ScalarType weight_dtype;
  int64_t weight_version;
  int64_t bias_version;
  uint64_t bias_source_key;
  std::vector<int64_t> logical_weight_sizes;
  PackedWeightKind kind;
  PackedWeightResidencyClass residency_class;
  bool quantized;
  uint64_t options_key;
  PackedWeightHandle handle;
};

struct RetiredPackedWeightMetadata final {
  std::optional<TensorWeakRef> weight_ref;
  std::optional<TensorWeakRef> bias_ref;
  std::vector<int64_t> logical_weight_sizes;
};

size_t packed_weight_cache_limit_bytes() {
  return kPackedWeightResidencyLimitBytes;
}

const std::string& packed_weight_cache_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_PACKED_WEIGHT_CACHE_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool packed_weight_cache_logging_enabled() {
  return !packed_weight_cache_log_path().empty();
}

size_t packed_weight_tensor_nbytes(const Tensor& tensor) {
  if (!tensor.defined() || !tensor.is_vulkan()) {
    return 0u;
  }
  return static_cast<size_t>(convert(tensor).gpu_nbytes());
}

size_t packed_weight_handle_nbytes(const Tensor& weight, const Tensor& bias) {
  return packed_weight_tensor_nbytes(weight) + packed_weight_tensor_nbytes(bias);
}

struct PackedWeightResidencyLogState final {
  std::atomic<uint64_t> lookups{0u};
  std::atomic<uint64_t> hits{0u};
  std::atomic<uint64_t> misses{0u};
  std::atomic<uint64_t> miss_empty_cache{0u};
  std::atomic<uint64_t> miss_no_match{0u};
  std::atomic<uint64_t> stores{0u};
  std::atomic<uint64_t> evictions{0u};
  std::atomic<uint64_t> persistent_evictions{0u};
  std::atomic<uint64_t> transient_evictions{0u};
  std::atomic<uint64_t> pruned_expired_sources{0u};
  std::atomic<uint64_t> cache_bytes{0u};
  std::atomic<uint64_t> peak_cache_bytes{0u};
  std::atomic<uint64_t> persistent_cache_bytes{0u};
  std::atomic<uint64_t> peak_persistent_cache_bytes{0u};

  void log() const {
    if (!packed_weight_cache_logging_enabled()) {
      return;
    }

    std::ofstream out(packed_weight_cache_log_path(), std::ios::app);
    out << "packed_weight_residency: lookups="
        << lookups.load(std::memory_order_relaxed)
        << " hits=" << hits.load(std::memory_order_relaxed)
        << " stores=" << stores.load(std::memory_order_relaxed)
        << " evictions=" << evictions.load(std::memory_order_relaxed)
        << " misses=" << misses.load(std::memory_order_relaxed)
        << " miss_empty_cache="
        << miss_empty_cache.load(std::memory_order_relaxed)
        << " miss_no_match=" << miss_no_match.load(std::memory_order_relaxed)
        << " persistent_evictions="
        << persistent_evictions.load(std::memory_order_relaxed)
        << " transient_evictions="
        << transient_evictions.load(std::memory_order_relaxed)
        << " pruned_expired_sources="
        << pruned_expired_sources.load(std::memory_order_relaxed)
        << " cache_bytes=" << cache_bytes.load(std::memory_order_relaxed)
        << " peak_cache_bytes="
        << peak_cache_bytes.load(std::memory_order_relaxed)
        << " persistent_cache_bytes="
        << persistent_cache_bytes.load(std::memory_order_relaxed)
        << " peak_persistent_cache_bytes="
        << peak_persistent_cache_bytes.load(std::memory_order_relaxed)
        << " cache_limit_bytes=" << packed_weight_cache_limit_bytes() << '\n';
  }

  ~PackedWeightResidencyLogState() {
    log();
  }
};

PackedWeightResidencyLogState& packed_weight_cache_log_state() {
  static PackedWeightResidencyLogState state;
  return state;
}

void reset_packed_weight_cache_log_state() {
  auto& state = packed_weight_cache_log_state();
  state.lookups.store(0u, std::memory_order_relaxed);
  state.hits.store(0u, std::memory_order_relaxed);
  state.misses.store(0u, std::memory_order_relaxed);
  state.miss_empty_cache.store(0u, std::memory_order_relaxed);
  state.miss_no_match.store(0u, std::memory_order_relaxed);
  state.stores.store(0u, std::memory_order_relaxed);
  state.evictions.store(0u, std::memory_order_relaxed);
  state.persistent_evictions.store(0u, std::memory_order_relaxed);
  state.transient_evictions.store(0u, std::memory_order_relaxed);
  state.pruned_expired_sources.store(0u, std::memory_order_relaxed);
  state.cache_bytes.store(0u, std::memory_order_relaxed);
  state.peak_cache_bytes.store(0u, std::memory_order_relaxed);
  state.persistent_cache_bytes.store(0u, std::memory_order_relaxed);
  state.peak_persistent_cache_bytes.store(0u, std::memory_order_relaxed);
}

std::mutex& retired_packed_weight_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::deque<PackedWeightHandle>& retired_packed_weight_handles() {
  static auto* handles = new std::deque<PackedWeightHandle>();
  return *handles;
}

std::deque<RetiredPackedWeightMetadata>& leaked_retired_packed_weight_metadata() {
  static auto* metadata = new std::deque<RetiredPackedWeightMetadata>();
  return *metadata;
}

std::deque<PackedWeightHandle>& quarantined_retired_packed_weight_handles() {
  static auto* handles = new std::deque<PackedWeightHandle>();
  return *handles;
}

void defer_retired_packed_weight_entries(
    std::deque<PackedWeightResidencyEntry>& retired_entries) {
  if (retired_entries.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(retired_packed_weight_mutex());
  auto& deferred_handles = retired_packed_weight_handles();
  auto& leaked_metadata = leaked_retired_packed_weight_metadata();
  for (auto& entry : retired_entries) {
    deferred_handles.emplace_back(std::move(entry.handle));
    leaked_metadata.emplace_back(RetiredPackedWeightMetadata{
        std::move(entry.weight_ref),
        std::move(entry.bias_ref),
        std::move(entry.logical_weight_sizes),
    });
  }
  retired_entries.clear();
}

bool release_retired_packed_weight_entries_impl() {
  std::deque<PackedWeightHandle> retired_handles;
  {
    std::lock_guard<std::mutex> lock(retired_packed_weight_mutex());
    retired_handles.swap(retired_packed_weight_handles());
  }
  if (retired_handles.empty()) {
    return false;
  }
  const size_t retired_count = retired_handles.size();
  {
    std::lock_guard<std::mutex> lock(retired_packed_weight_mutex());
    auto& quarantined_handles = quarantined_retired_packed_weight_handles();
    while (!retired_handles.empty()) {
      quarantined_handles.emplace_back(std::move(retired_handles.front()));
      retired_handles.pop_front();
    }
  }
  log_vulkan_op_hit(
      std::string(
          "vulkan_packed_weight_release.disabled_quarantined count=") +
      std::to_string(retired_count));
  return false;
}

class PackedWeightResidencyManager final {
 private:
  std::mutex mutex_;
  std::deque<PackedWeightResidencyEntry> cache_;
  size_t cache_bytes_{0u};
  size_t persistent_cache_bytes_{0u};

  static void release_retired_entries(
      std::deque<PackedWeightResidencyEntry>& retired_entries) {
    defer_retired_packed_weight_entries(retired_entries);
  }

  static bool source_refs_alive(const PackedWeightResidencyEntry& entry) {
    if (
        !weak_tensor_ref_alive(entry.weight_ref) &&
        !weak_storage_ref_alive(entry.weight_storage_ref) &&
        !weak_vulkan_storage_ref_alive(entry.weight_vulkan_storage_ref) &&
        entry.weight_source_key == 0u) {
      return false;
    }
    return optional_weak_tensor_ref_alive(entry.bias_ref) ||
        entry.bias_source_key != 0u;
  }

  static bool bias_matches_optional_tensor(
      const PackedWeightResidencyEntry& entry,
      const std::optional<Tensor>& normalized_bias) {
    if (!normalized_bias || !normalized_bias->defined()) {
      return !entry.bias_ref.has_value() && entry.bias_source_key == 0u;
    }
    return weak_ref_matches_tensor(entry.bias_ref, *normalized_bias) ||
        (entry.bias_source_key != 0u &&
         tensor_packed_weight_source_key(*normalized_bias) ==
             entry.bias_source_key);
  }

  static bool storage_view_matches_tensor(
      const PackedWeightResidencyEntry& entry,
      const Tensor& source_weight) {
    const bool c10_storage_matches =
        entry.weight_storage_identity != nullptr &&
        tensor_storage_identity_ptr(source_weight) == entry.weight_storage_identity;
    const bool vulkan_storage_matches =
        entry.weight_vulkan_storage_identity != nullptr &&
        tensor_vulkan_storage_identity_ptr(source_weight) ==
            entry.weight_vulkan_storage_identity;
    const bool provenance_source_matches = entry.weight_source_key != 0u &&
        tensor_packed_weight_source_key(source_weight) == entry.weight_source_key;
    if (
        (!c10_storage_matches && !vulkan_storage_matches &&
         !provenance_source_matches) ||
        entry.weight_storage_offset != source_weight.storage_offset() ||
        entry.weight_dtype != source_weight.scalar_type() ||
        entry.logical_weight_sizes.size() !=
            static_cast<size_t>(source_weight.dim()) ||
        entry.weight_strides.size() != static_cast<size_t>(source_weight.dim())) {
      return false;
    }
    return std::equal(
               entry.logical_weight_sizes.begin(),
               entry.logical_weight_sizes.end(),
               source_weight.sizes().begin()) &&
        std::equal(
               entry.weight_strides.begin(),
               entry.weight_strides.end(),
               source_weight.strides().begin());
  }

  static bool matches_entry(
      const PackedWeightResidencyEntry& entry,
      const Tensor& source_weight,
      const std::optional<Tensor>& normalized_bias,
      const int64_t weight_version,
      const int64_t bias_version,
      IntArrayRef logical_weight_sizes,
      const PackedWeightKind kind,
      const bool quantized,
      const uint64_t options_key) {
    const bool weight_matches =
        packed_weight_ref_matches_tensor(entry.weight_ref, source_weight) ||
        storage_view_matches_tensor(entry, source_weight);
    return weight_matches &&
        entry.weight_version == weight_version &&
        bias_matches_optional_tensor(entry, normalized_bias) &&
        entry.bias_version == bias_version &&
        entry.logical_weight_sizes.size() == logical_weight_sizes.size() &&
        std::equal(
            logical_weight_sizes.begin(),
            logical_weight_sizes.end(),
            entry.logical_weight_sizes.begin()) &&
        entry.kind == kind && entry.quantized == quantized &&
        entry.options_key == options_key;
  }

  void update_log_snapshot_locked() const {
    if (!packed_weight_cache_logging_enabled()) {
      return;
    }
    auto& log_state = packed_weight_cache_log_state();
    const auto cache_bytes = static_cast<uint64_t>(cache_bytes_);
    const auto persistent_cache_bytes =
        static_cast<uint64_t>(persistent_cache_bytes_);
    log_state.cache_bytes.store(cache_bytes, std::memory_order_relaxed);
    log_state.persistent_cache_bytes.store(
        persistent_cache_bytes, std::memory_order_relaxed);

    uint64_t observed_peak_cache =
        log_state.peak_cache_bytes.load(std::memory_order_relaxed);
    while (
        cache_bytes > observed_peak_cache &&
        !log_state.peak_cache_bytes.compare_exchange_weak(
            observed_peak_cache,
            cache_bytes,
            std::memory_order_relaxed,
            std::memory_order_relaxed)) {
    }

    uint64_t observed_peak_persistent =
        log_state.peak_persistent_cache_bytes.load(std::memory_order_relaxed);
    while (
        persistent_cache_bytes > observed_peak_persistent &&
        !log_state.peak_persistent_cache_bytes.compare_exchange_weak(
            observed_peak_persistent,
            persistent_cache_bytes,
            std::memory_order_relaxed,
            std::memory_order_relaxed)) {
    }
    log_state.log();
  }

  std::deque<PackedWeightResidencyEntry>::iterator retire_entry_locked(
      std::deque<PackedWeightResidencyEntry>::iterator entry_it,
      std::deque<PackedWeightResidencyEntry>& retired_entries,
      const bool count_eviction) {
    const bool persistent_entry =
        entry_it->residency_class ==
        PackedWeightResidencyClass::PersistentInference;
    cache_bytes_ -= entry_it->handle.resident_nbytes();
    if (persistent_entry) {
      persistent_cache_bytes_ -= entry_it->handle.resident_nbytes();
    }
    retired_entries.emplace_back(std::move(*entry_it));
    auto next_it = cache_.erase(entry_it);
    if (count_eviction) {
      auto& log_state = packed_weight_cache_log_state();
      log_state.evictions.fetch_add(1u, std::memory_order_relaxed);
      if (persistent_entry) {
        log_state.persistent_evictions.fetch_add(1u, std::memory_order_relaxed);
      } else {
        log_state.transient_evictions.fetch_add(1u, std::memory_order_relaxed);
      }
    }
    return next_it;
  }

  std::deque<PackedWeightResidencyEntry>::iterator
  select_transient_eviction_candidate_locked() {
    for (auto it = cache_.end(); it != cache_.begin();) {
      --it;
      if (
          it->residency_class == PackedWeightResidencyClass::Transient &&
          it->handle.defined()) {
        return it;
      }
    }
    return cache_.empty() ? cache_.end() : std::prev(cache_.end());
  }

  void trim_locked(std::deque<PackedWeightResidencyEntry>& retired_entries) {
    while (cache_bytes_ > packed_weight_cache_limit_bytes()) {
      auto victim = select_transient_eviction_candidate_locked();
      if (
          victim == cache_.end() ||
          victim->residency_class != PackedWeightResidencyClass::Transient) {
        break;
      }
      retire_entry_locked(victim, retired_entries, true);
    }
    while (cache_.size() > kPackedWeightResidencyMaxEntries) {
      auto victim = select_transient_eviction_candidate_locked();
      if (victim == cache_.end()) {
        break;
      }
      retire_entry_locked(victim, retired_entries, true);
    }
    update_log_snapshot_locked();
  }

 public:
  std::vector<std::string> snapshot() {
    std::vector<std::string> rows;
    std::lock_guard<std::mutex> lock(mutex_);
    rows.reserve(cache_.size() + 1u);

    size_t live_bytes = 0u;
    size_t live_persistent_bytes = 0u;
    for (const PackedWeightResidencyEntry& entry : cache_) {
      const size_t resident_bytes = entry.handle.resident_nbytes();
      live_bytes += resident_bytes;
      if (
          entry.residency_class ==
          PackedWeightResidencyClass::PersistentInference) {
        live_persistent_bytes += resident_bytes;
      }
      std::ostringstream stream;
      stream << "packed_weight_residency"
             << " state=live"
             << " kind=" << to_string(entry.kind)
             << " residency_class=" << to_string(entry.residency_class)
             << " bytes=" << resident_bytes
             << " logical_weight_shape="
             << format_size_list(entry.logical_weight_sizes)
             << " dtype=" << entry.weight_dtype
             << " quantized=" << (entry.quantized ? 1 : 0)
             << " options_key=" << entry.options_key
             << " source_tensor_alive="
             << (weak_tensor_ref_alive(entry.weight_ref) ? 1 : 0)
             << " source_storage_alive="
             << (weak_storage_ref_alive(entry.weight_storage_ref) ? 1 : 0)
             << " source_vulkan_storage_alive="
             << (weak_vulkan_storage_ref_alive(entry.weight_vulkan_storage_ref)
                     ? 1
                     : 0)
             << " raw_weight_storage_identity="
             << reinterpret_cast<uintptr_t>(entry.weight_storage_identity)
             << " raw_weight_vulkan_storage_identity="
             << reinterpret_cast<uintptr_t>(
                    entry.weight_vulkan_storage_identity)
             << " weight_source_key=" << entry.weight_source_key
             << " bias_source_key=" << entry.bias_source_key
             << " pack_identity="
             << reinterpret_cast<uintptr_t>(entry.handle.identity());
      rows.emplace_back(stream.str());
    }

    auto& log_state = packed_weight_cache_log_state();
    std::ostringstream summary;
    summary << "packed_weight_residency_summary"
            << " live_entries=" << cache_.size()
            << " live_bytes=" << live_bytes
            << " live_persistent_bytes=" << live_persistent_bytes
            << " cache_limit_bytes=" << packed_weight_cache_limit_bytes()
            << " manager_cache_bytes=" << cache_bytes_
            << " manager_persistent_cache_bytes=" << persistent_cache_bytes_
            << " lookups=" << log_state.lookups.load(std::memory_order_relaxed)
            << " hits=" << log_state.hits.load(std::memory_order_relaxed)
            << " misses=" << log_state.misses.load(std::memory_order_relaxed)
            << " miss_empty_cache="
            << log_state.miss_empty_cache.load(std::memory_order_relaxed)
            << " miss_no_match="
            << log_state.miss_no_match.load(std::memory_order_relaxed)
            << " stores=" << log_state.stores.load(std::memory_order_relaxed)
            << " evictions="
            << log_state.evictions.load(std::memory_order_relaxed)
            << " persistent_evictions="
            << log_state.persistent_evictions.load(std::memory_order_relaxed)
            << " transient_evictions="
            << log_state.transient_evictions.load(std::memory_order_relaxed)
            << " pruned_expired_sources="
            << log_state.pruned_expired_sources.load(std::memory_order_relaxed);
    rows.emplace_back(summary.str());
    return rows;
  }

  std::optional<PackedWeightHandle> lookup(
      const Tensor& source_weight,
      const std::optional<Tensor>& normalized_bias,
      IntArrayRef logical_weight_sizes,
      const PackedWeightKind kind,
      const bool quantized,
      const uint64_t options_key) {
    if (!source_weight.defined()) {
      return std::nullopt;
    }

    auto& log_state = packed_weight_cache_log_state();
    log_state.lookups.fetch_add(1u, std::memory_order_relaxed);

    const int64_t weight_version = tensor_version_or_zero(source_weight);
    const int64_t bias_version =
        normalized_bias ? tensor_version_or_zero(*normalized_bias) : 0u;

    std::deque<PackedWeightResidencyEntry> retired_entries;
    std::optional<PackedWeightHandle> result;
    bool cache_was_empty = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cache_was_empty = cache_.empty();
      for (auto it = cache_.begin(); it != cache_.end();) {
        if (!source_refs_alive(*it)) {
          log_state.pruned_expired_sources.fetch_add(
              1u, std::memory_order_relaxed);
          it = retire_entry_locked(it, retired_entries, true);
          continue;
        }
        if (!matches_entry(
                *it,
                source_weight,
                normalized_bias,
                weight_version,
                bias_version,
                logical_weight_sizes,
                kind,
                quantized,
                options_key)) {
          ++it;
          continue;
        }

        PackedWeightHandle handle = it->handle;
        if (it != cache_.begin()) {
          PackedWeightResidencyEntry entry = std::move(*it);
          cache_.erase(it);
          cache_.emplace_front(std::move(entry));
          handle = cache_.front().handle;
        }

        log_state.hits.fetch_add(1u, std::memory_order_relaxed);
        result = handle;
        break;
      }
      if (!result.has_value()) {
        log_state.misses.fetch_add(1u, std::memory_order_relaxed);
        if (cache_was_empty) {
          log_state.miss_empty_cache.fetch_add(1u, std::memory_order_relaxed);
        } else {
          log_state.miss_no_match.fetch_add(1u, std::memory_order_relaxed);
        }
      }
      update_log_snapshot_locked();
    }

    release_retired_entries(retired_entries);
    return result;
  }

  void store(
      const Tensor& source_weight,
      const std::optional<Tensor>& normalized_bias,
      IntArrayRef logical_weight_sizes,
      const PackedWeightKind kind,
      const PackedWeightHandle& handle,
      const bool quantized,
      const uint64_t options_key) {
    if (!source_weight.defined() || !handle.defined()) {
      return;
    }

    PackedWeightResidencyEntry entry;
    entry.weight_ref = make_packed_weight_weak_ref(source_weight);
    entry.bias_ref = normalized_bias ? make_tensor_weak_ref(*normalized_bias)
                                     : std::nullopt;
    entry.weight_storage_ref = make_storage_weak_ref(source_weight);
    entry.weight_vulkan_storage_ref =
        make_vulkan_storage_weak_ref(source_weight);
    entry.weight_storage_identity = tensor_storage_identity_ptr(source_weight);
    entry.weight_vulkan_storage_identity =
        tensor_vulkan_storage_identity_ptr(source_weight);
    entry.weight_source_key = tensor_packed_weight_source_key(source_weight);
    entry.weight_storage_offset = source_weight.storage_offset();
    entry.weight_strides = source_weight.strides().vec();
    entry.weight_dtype = source_weight.scalar_type();
    entry.weight_version = tensor_version_or_zero(source_weight);
    entry.bias_version =
        normalized_bias ? tensor_version_or_zero(*normalized_bias) : 0u;
    entry.bias_source_key =
        normalized_bias ? tensor_packed_weight_source_key(*normalized_bias) : 0u;
    entry.logical_weight_sizes = std::vector<int64_t>(
        logical_weight_sizes.begin(), logical_weight_sizes.end());
    entry.kind = kind;
    entry.residency_class = handle.residency_class();
    entry.quantized = quantized;
    entry.options_key = options_key;
    entry.handle = handle;

    packed_weight_cache_log_state().stores.fetch_add(
        1u, std::memory_order_relaxed);

    std::deque<PackedWeightResidencyEntry> retired_entries;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto it = cache_.begin(); it != cache_.end();) {
        if (!source_refs_alive(*it)) {
          packed_weight_cache_log_state().pruned_expired_sources.fetch_add(
              1u, std::memory_order_relaxed);
          it = retire_entry_locked(it, retired_entries, true);
          continue;
        }
        if (!matches_entry(
                *it,
                source_weight,
                normalized_bias,
                entry.weight_version,
                entry.bias_version,
                logical_weight_sizes,
                kind,
                quantized,
                options_key)) {
          ++it;
          continue;
        }
        retire_entry_locked(it, retired_entries, false);
        break;
      }

      cache_bytes_ += handle.resident_nbytes();
      if (
          handle.residency_class() ==
          PackedWeightResidencyClass::PersistentInference) {
        persistent_cache_bytes_ += handle.resident_nbytes();
      }
      cache_.emplace_front(std::move(entry));
      trim_locked(retired_entries);
    }
    release_retired_entries(retired_entries);
  }
};

PackedWeightResidencyManager& packed_weight_residency_manager() {
  static auto* manager = new PackedWeightResidencyManager();
  return *manager;
}

const std::string& linear_cache_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_LINEAR_CACHE_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool linear_cache_logging_enabled() {
  return !linear_cache_log_path().empty();
}

std::mutex& linear_cache_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

struct LinearCacheLogState final {
  std::atomic<uint64_t> lookups{0u};
  std::atomic<uint64_t> hits{0u};
  std::atomic<uint64_t> misses{0u};
  std::atomic<uint64_t> stores{0u};
  std::atomic<uint64_t> evictions{0u};
  std::atomic<uint64_t> prunes{0u};
  std::atomic<uint64_t> pruned_entries{0u};

  ~LinearCacheLogState() {
    if (!linear_cache_logging_enabled()) {
      return;
    }

    std::ofstream out(linear_cache_log_path(), std::ios::app);
    out << "linear_cache: lookups=" << lookups.load(std::memory_order_relaxed)
        << " hits=" << hits.load(std::memory_order_relaxed)
        << " misses=" << misses.load(std::memory_order_relaxed)
        << " stores=" << stores.load(std::memory_order_relaxed)
        << " evictions=" << evictions.load(std::memory_order_relaxed)
        << " prunes=" << prunes.load(std::memory_order_relaxed)
        << " pruned_entries="
        << pruned_entries.load(std::memory_order_relaxed) << '\n';
  }
};

LinearCacheLogState& linear_cache_log_state() {
  static LinearCacheLogState state;
  return state;
}

void log_linear_cache_event(
    const char* cache_kind,
    const char* event,
    const std::string& allocation_label = std::string(),
    const size_t evictions = 0u) {
  if (!linear_cache_logging_enabled()) {
    return;
  }

  std::lock_guard<std::mutex> lock(linear_cache_log_mutex());
  std::ofstream out(linear_cache_log_path(), std::ios::app);
  out << "linear_cache_event cache=" << cache_kind << " event=" << event;
  if (!allocation_label.empty()) {
    out << " label=" << allocation_label;
  }
  if (evictions > 0u) {
    out << " evictions=" << evictions;
  }
  out << '\n';
}

const std::string& execution_object_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_EXECUTION_OBJECT_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool execution_object_logging_enabled() {
  return !execution_object_log_path().empty();
}

std::mutex& execution_object_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

struct ExecutionObjectLogState final {
  std::atomic<uint64_t> kv_hits{0u};
  std::atomic<uint64_t> kv_stores{0u};
  std::atomic<uint64_t> kv_resets{0u};
  std::atomic<uint64_t> kv_sequence_updates{0u};
  std::atomic<uint64_t> kv_read_views{0u};
  std::atomic<uint64_t> kv_append_views{0u};
  std::atomic<uint64_t> scratch_hits{0u};
  std::atomic<uint64_t> scratch_stores{0u};
  std::atomic<uint64_t> scratch_resets{0u};
  std::atomic<uint64_t> scratch_reserves{0u};
  std::atomic<uint64_t> scratch_reserved_bytes{0u};
  std::atomic<uint64_t> scratch_peak_reserved_bytes{0u};
  std::atomic<uint64_t> readback_hits{0u};
  std::atomic<uint64_t> readback_stores{0u};

  ~ExecutionObjectLogState() {
    if (!execution_object_logging_enabled()) {
      return;
    }

    std::ofstream out(execution_object_log_path(), std::ios::app);
    out << "execution_object_summary kind=KVCache"
        << " hits=" << kv_hits.load(std::memory_order_relaxed)
        << " stores=" << kv_stores.load(std::memory_order_relaxed)
        << " resets=" << kv_resets.load(std::memory_order_relaxed)
        << " sequence_updates="
        << kv_sequence_updates.load(std::memory_order_relaxed)
        << " read_views=" << kv_read_views.load(std::memory_order_relaxed)
        << " append_views=" << kv_append_views.load(std::memory_order_relaxed)
        << '\n';
    out << "execution_object_summary kind=ScratchArena"
        << " hits=" << scratch_hits.load(std::memory_order_relaxed)
        << " stores=" << scratch_stores.load(std::memory_order_relaxed)
        << " resets=" << scratch_resets.load(std::memory_order_relaxed)
        << " reserves=" << scratch_reserves.load(std::memory_order_relaxed)
        << " reserved_bytes="
        << scratch_reserved_bytes.load(std::memory_order_relaxed)
        << " peak_reserved_bytes="
        << scratch_peak_reserved_bytes.load(std::memory_order_relaxed)
        << '\n';
    out << "execution_object_summary kind=ReadbackBuffer"
        << " hits=" << readback_hits.load(std::memory_order_relaxed)
        << " stores=" << readback_stores.load(std::memory_order_relaxed)
        << '\n';
  }
};

ExecutionObjectLogState& execution_object_log_state() {
  static ExecutionObjectLogState state;
  return state;
}

void log_execution_object_event(
    const char* kind,
    const char* event,
    const std::string& allocation_label,
    const void* identity,
    const size_t bytes = 0u) {
  if (!execution_object_logging_enabled()) {
    return;
  }

  std::lock_guard<std::mutex> lock(execution_object_log_mutex());
  std::ofstream out(execution_object_log_path(), std::ios::app);
  out << "execution_object_event kind=" << kind << " event=" << event;
  if (!allocation_label.empty()) {
    out << " label=" << allocation_label;
  }
  if (identity) {
    out << " identity=" << identity;
  }
  if (bytes > 0u) {
    out << " bytes=" << bytes;
  }
  out << '\n';
}

void record_scratch_reserved_bytes(const uint64_t bytes) {
  auto& log_state = execution_object_log_state();
  const uint64_t total_reserved = log_state.scratch_reserved_bytes.fetch_add(
                                      bytes, std::memory_order_relaxed) +
      bytes;
  uint64_t observed_peak =
      log_state.scratch_peak_reserved_bytes.load(std::memory_order_relaxed);
  while (
      total_reserved > observed_peak &&
      !log_state.scratch_peak_reserved_bytes.compare_exchange_weak(
          observed_peak,
          total_reserved,
          std::memory_order_relaxed,
          std::memory_order_relaxed)) {
  }
}

struct LinearContextCacheKey final {
  std::optional<TensorWeakRef> weight_ref;
  std::optional<TensorWeakRef> bias_ref;
  const void* weight_identity{nullptr};
  const void* bias_identity{nullptr};
  int64_t weight_version;
  int64_t bias_version;
};

bool same_linear_context_cache_key(
    const LinearContextCacheKey& lhs,
    const LinearContextCacheKey& rhs) {
  return lhs.weight_identity == rhs.weight_identity &&
      lhs.bias_identity == rhs.bias_identity &&
      same_weak_tensor_ref(lhs.weight_ref, rhs.weight_ref) &&
      lhs.weight_version == rhs.weight_version &&
      same_weak_tensor_ref(lhs.bias_ref, rhs.bias_ref) &&
      lhs.bias_version == rhs.bias_version;
}

size_t hash_linear_context_cache_key(const LinearContextCacheKey& key) {
  size_t seed = 0u;
  hash_combine(seed, reinterpret_cast<uintptr_t>(key.weight_identity));
  hash_combine(seed, reinterpret_cast<uintptr_t>(key.bias_identity));
  hash_combine(seed, key.weight_version);
  hash_combine(seed, key.bias_version);
  return seed;
}

bool linear_context_cache_key_sources_alive(
    const LinearContextCacheKey& key) {
  return weak_tensor_ref_alive(key.weight_ref) &&
      optional_weak_tensor_ref_alive(key.bias_ref);
}

InferenceLruCache<
    LinearContextCacheKey,
    c10::intrusive_ptr<LinearPackedContext>>&
linear_context_cache() {
  static auto* cache = new InferenceLruCache<
      LinearContextCacheKey,
      c10::intrusive_ptr<LinearPackedContext>>
      {kLinearContextCacheSize};
  return *cache;
}

struct LabeledLinearContextCacheKey final {
  std::optional<TensorWeakRef> weight_ref;
  std::optional<TensorWeakRef> bias_ref;
  const void* weight_identity{nullptr};
  const void* bias_identity{nullptr};
  int64_t weight_version;
  int64_t bias_version;
  std::string allocation_label;
};

bool same_labeled_linear_context_cache_key(
    const LabeledLinearContextCacheKey& lhs,
    const LabeledLinearContextCacheKey& rhs) {
  return lhs.weight_identity == rhs.weight_identity &&
      lhs.bias_identity == rhs.bias_identity &&
      same_weak_tensor_ref(lhs.weight_ref, rhs.weight_ref) &&
      lhs.weight_version == rhs.weight_version &&
      same_weak_tensor_ref(lhs.bias_ref, rhs.bias_ref) &&
      lhs.bias_version == rhs.bias_version &&
      lhs.allocation_label == rhs.allocation_label;
}

size_t hash_labeled_linear_context_cache_key(
    const LabeledLinearContextCacheKey& key) {
  size_t seed = hash_linear_context_cache_key(LinearContextCacheKey{
      key.weight_ref,
      key.bias_ref,
      key.weight_identity,
      key.bias_identity,
      key.weight_version,
      key.bias_version,
  });
  hash_combine(seed, key.allocation_label);
  return seed;
}

bool labeled_linear_context_cache_key_sources_alive(
    const LabeledLinearContextCacheKey& key) {
  return weak_tensor_ref_alive(key.weight_ref) &&
      optional_weak_tensor_ref_alive(key.bias_ref);
}

InferenceLruCache<
    LabeledLinearContextCacheKey,
    c10::intrusive_ptr<LinearPackedContext>>&
labeled_linear_context_cache() {
  static auto* cache = new InferenceLruCache<
      LabeledLinearContextCacheKey,
      c10::intrusive_ptr<LinearPackedContext>>
      {kLinearContextCacheSize};
  return *cache;
}

std::mutex& retired_linear_context_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::deque<c10::intrusive_ptr<LinearPackedContext>>& retired_linear_contexts() {
  static auto* contexts =
      new std::deque<c10::intrusive_ptr<LinearPackedContext>>();
  return *contexts;
}

void retire_linear_context_after_prune(
    c10::intrusive_ptr<LinearPackedContext>&& context) {
  if (!context) {
    return;
  }
  std::lock_guard<std::mutex> lock(retired_linear_context_mutex());
  retired_linear_contexts().emplace_back(std::move(context));
}

bool release_retired_linear_contexts_impl() {
  std::deque<c10::intrusive_ptr<LinearPackedContext>> retired_contexts;
  {
    std::lock_guard<std::mutex> lock(retired_linear_context_mutex());
    retired_contexts.swap(retired_linear_contexts());
  }
  if (retired_contexts.empty()) {
    return false;
  }
  retired_contexts.clear();
  return true;
}

void record_linear_cache_evictions(const size_t evictions) {
  if (evictions == 0u || !linear_cache_logging_enabled()) {
    return;
  }
  linear_cache_log_state().evictions.fetch_add(
      evictions, std::memory_order_relaxed);
}

std::atomic<uint64_t>& linear_cache_prune_ticks() {
  static std::atomic<uint64_t> ticks{0u};
  return ticks;
}

bool should_prune_linear_context_caches() {
  const uint64_t tick =
      linear_cache_prune_ticks().fetch_add(1u, std::memory_order_relaxed) + 1u;
  return tick % kLinearContextPruneInterval == 0u;
}

size_t prune_expired_linear_context_cache_entries() {
  const size_t evicted = linear_context_cache().erase_if_budgeted(
      [](const LinearContextCacheKey& key,
         const c10::intrusive_ptr<LinearPackedContext>&) {
        return !linear_context_cache_key_sources_alive(key);
      },
      kLinearContextPruneScanBudget,
      kLinearContextPruneEraseBudget,
      [](c10::intrusive_ptr<LinearPackedContext>&& context) {
        retire_linear_context_after_prune(std::move(context));
      });
  record_linear_cache_evictions(evicted);
  if (linear_cache_logging_enabled()) {
    auto& state = linear_cache_log_state();
    state.prunes.fetch_add(1u, std::memory_order_relaxed);
    state.pruned_entries.fetch_add(evicted, std::memory_order_relaxed);
    log_linear_cache_event("linear", "prune", std::string(), evicted);
  }
  return evicted;
}

size_t prune_expired_labeled_linear_context_cache_entries() {
  const size_t evicted = labeled_linear_context_cache().erase_if_budgeted(
      [](const LabeledLinearContextCacheKey& key,
         const c10::intrusive_ptr<LinearPackedContext>&) {
        return !labeled_linear_context_cache_key_sources_alive(key);
      },
      kLinearContextPruneScanBudget,
      kLinearContextPruneEraseBudget,
      [](c10::intrusive_ptr<LinearPackedContext>&& context) {
        retire_linear_context_after_prune(std::move(context));
      });
  record_linear_cache_evictions(evicted);
  if (linear_cache_logging_enabled()) {
    auto& state = linear_cache_log_state();
    state.prunes.fetch_add(1u, std::memory_order_relaxed);
    state.pruned_entries.fetch_add(evicted, std::memory_order_relaxed);
    log_linear_cache_event("labeled_linear", "prune", std::string(), evicted);
  }
  return evicted;
}

struct LabeledKVCacheKey final {
  std::string allocation_label;
  std::vector<int64_t> sizes;
  int64_t sequence_dim;
  ScalarType dtype;
  api::ExecutionLayout execution_layout;
  api::GPUMemoryLayout memory_layout;
  api::StorageType storage_type;
  bool persistent;
};

bool same_labeled_kv_cache_key(
    const LabeledKVCacheKey& lhs,
    const LabeledKVCacheKey& rhs) {
  return lhs.allocation_label == rhs.allocation_label &&
      lhs.sizes == rhs.sizes && lhs.sequence_dim == rhs.sequence_dim &&
      lhs.dtype == rhs.dtype &&
      lhs.execution_layout == rhs.execution_layout &&
      lhs.memory_layout == rhs.memory_layout &&
      lhs.storage_type == rhs.storage_type &&
      lhs.persistent == rhs.persistent;
}

size_t hash_labeled_kv_cache_key(const LabeledKVCacheKey& key) {
  size_t seed = 0u;
  hash_combine(seed, key.allocation_label);
  hash_combine_sizes(seed, key.sizes);
  hash_combine(seed, key.sequence_dim);
  hash_combine(seed, static_cast<int>(key.dtype));
  hash_combine(seed, static_cast<int>(key.execution_layout));
  hash_combine(seed, static_cast<int>(key.memory_layout));
  hash_combine(seed, static_cast<int>(key.storage_type));
  hash_combine(seed, key.persistent);
  return seed;
}

InferenceLruCache<LabeledKVCacheKey, KVCacheObject>& labeled_kv_cache() {
  static auto* cache = new InferenceLruCache<LabeledKVCacheKey, KVCacheObject>{
      kExecutionObjectCacheSize};
  return *cache;
}

struct LabeledScratchArenaKey final {
  std::string allocation_label;
  size_t num_bytes;
  uint32_t alignment;
  ScalarType dtype;
  api::ExecutionLayout execution_layout;
  api::GPUMemoryLayout memory_layout;
  api::StorageType storage_type;
  bool persistent;
};

bool same_labeled_scratch_arena_key(
    const LabeledScratchArenaKey& lhs,
    const LabeledScratchArenaKey& rhs) {
  return lhs.allocation_label == rhs.allocation_label &&
      lhs.num_bytes == rhs.num_bytes &&
      lhs.alignment == rhs.alignment &&
      lhs.dtype == rhs.dtype &&
      lhs.execution_layout == rhs.execution_layout &&
      lhs.memory_layout == rhs.memory_layout &&
      lhs.storage_type == rhs.storage_type &&
      lhs.persistent == rhs.persistent;
}

size_t hash_labeled_scratch_arena_key(const LabeledScratchArenaKey& key) {
  size_t seed = 0u;
  hash_combine(seed, key.allocation_label);
  hash_combine(seed, key.num_bytes);
  hash_combine(seed, key.alignment);
  hash_combine(seed, static_cast<int>(key.dtype));
  hash_combine(seed, static_cast<int>(key.execution_layout));
  hash_combine(seed, static_cast<int>(key.memory_layout));
  hash_combine(seed, static_cast<int>(key.storage_type));
  hash_combine(seed, key.persistent);
  return seed;
}

InferenceLruCache<LabeledScratchArenaKey, ScratchArena>&
labeled_scratch_arena_cache() {
  static auto* cache =
      new InferenceLruCache<LabeledScratchArenaKey, ScratchArena>{
      kExecutionObjectCacheSize};
  return *cache;
}

struct LabeledReadbackBufferKey final {
  std::string allocation_label;
  size_t num_bytes;
  bool persistent;
};

bool same_labeled_readback_buffer_key(
    const LabeledReadbackBufferKey& lhs,
    const LabeledReadbackBufferKey& rhs) {
  return lhs.allocation_label == rhs.allocation_label &&
      lhs.num_bytes == rhs.num_bytes &&
      lhs.persistent == rhs.persistent;
}

size_t hash_labeled_readback_buffer_key(const LabeledReadbackBufferKey& key) {
  size_t seed = 0u;
  hash_combine(seed, key.allocation_label);
  hash_combine(seed, key.num_bytes);
  hash_combine(seed, key.persistent);
  return seed;
}

InferenceLruCache<LabeledReadbackBufferKey, ReadbackBufferObject>&
labeled_readback_buffer_cache() {
  static auto* cache =
      new InferenceLruCache<LabeledReadbackBufferKey, ReadbackBufferObject>{
          kExecutionObjectCacheSize};
  return *cache;
}

} // namespace

bool release_retired_linear_contexts() {
  return release_retired_linear_contexts_impl();
}

bool release_retired_packed_weight_entries() {
  return release_retired_packed_weight_entries_impl();
}

std::vector<std::string> packed_weight_residency_snapshot() {
  return packed_weight_residency_manager().snapshot();
}

void reset_packed_weight_residency_snapshot() {
  reset_packed_weight_cache_log_state();
}

const char* execution_object_kind_name(const VulkanExecutionObjectKind kind) {
  switch (kind) {
    case VulkanExecutionObjectKind::PackedWeight:
      return "PackedWeight";
    case VulkanExecutionObjectKind::LinearContext:
      return "LinearContext";
    case VulkanExecutionObjectKind::KVCache:
      return "KVCache";
    case VulkanExecutionObjectKind::ScratchArena:
      return "ScratchArena";
    case VulkanExecutionObjectKind::ReadbackBuffer:
      return "ReadbackBuffer";
  }
  return "PackedWeight";
}

std::string make_vulkan_runtime_object_label(
    const VulkanPlanningRequest& request,
    const char* label_suffix) {
  const std::string& runtime_label = api::current_runtime_label();
  const std::string& current_label =
      runtime_label.empty() ? api::current_allocation_label() : runtime_label;
  if (!current_label.empty() && current_label != "unlabeled") {
    return current_label + "." + label_suffix;
  }

  std::ostringstream stream;
  stream << model_domain_name(request.model_domain) << "."
         << execution_phase_name(request.execution_phase) << "."
         << workload_class_name(request.workload_class) << "."
         << label_suffix;
  return stream.str();
}

const std::string& resolve_vulkan_linear_runtime_label(
    const std::string& allocation_label,
    const char* fallback_label) {
  if (!allocation_label.empty()) {
    return allocation_label;
  }

  static const std::string kLinearLabel = "linear";
  static const std::string kBmmLabel = "bmm";
  return std::string(fallback_label) == "bmm" ? kBmmLabel : kLinearLabel;
}

std::string make_vulkan_linear_pack_label(
    const std::string& allocation_label,
    const char* fallback_label) {
  if (allocation_label.empty()) {
    return fallback_label;
  }
  return allocation_label + "." + fallback_label;
}

bool KVCacheObject::defined() const {
  return state_ && state_->storage_.defined();
}

const Tensor& KVCacheObject::storage() const {
  TORCH_CHECK(state_, "KV cache object is not initialized");
  return state_->storage_;
}

const std::vector<int64_t>& KVCacheObject::sizes() const {
  TORCH_CHECK(state_, "KV cache object is not initialized");
  return state_->sizes_;
}

int64_t KVCacheObject::sequence_dim() const {
  TORCH_CHECK(state_, "KV cache object is not initialized");
  return state_->sequence_dim_;
}

int64_t KVCacheObject::max_sequence_length() const {
  TORCH_CHECK(state_, "KV cache object is not initialized");
  return state_->sizes_.at(state_->sequence_dim_);
}

int64_t KVCacheObject::sequence_length() const {
  TORCH_CHECK(state_, "KV cache object is not initialized");
  std::lock_guard<std::mutex> lock(state_->mutex_);
  return state_->sequence_length_;
}

void KVCacheObject::reset() {
  if (state_) {
    execution_object_log_state().kv_resets.fetch_add(
        1u, std::memory_order_relaxed);
    log_execution_object_event(
        "KVCache", "reset", std::string(),
        identity());
  }
  set_sequence_length(0);
}

void KVCacheObject::set_sequence_length(const int64_t sequence_length) {
  TORCH_CHECK(state_, "KV cache object is not initialized");
  TORCH_CHECK(
      sequence_length >= 0 && sequence_length <= max_sequence_length(),
      "Requested KV cache sequence length ",
      sequence_length,
      " exceeds cache capacity ",
      max_sequence_length());
  std::lock_guard<std::mutex> lock(state_->mutex_);
  state_->sequence_length_ = sequence_length;
  execution_object_log_state().kv_sequence_updates.fetch_add(
      1u, std::memory_order_relaxed);
  log_execution_object_event(
      "KVCache", "set_sequence_length", std::string(), identity());
}

Tensor KVCacheObject::read_view(const int64_t start, const int64_t length) const {
  TORCH_CHECK(state_, "KV cache object is not initialized");
  TORCH_CHECK(length >= 0, "KV cache read length must be non-negative");
  const int64_t current_length = sequence_length();
  TORCH_CHECK(
      start >= 0 && start + length <= current_length,
      "Requested KV cache read range [",
      start,
      ", ",
      start + length,
      ") exceeds current sequence length ",
      current_length);
  execution_object_log_state().kv_read_views.fetch_add(
      1u, std::memory_order_relaxed);
  log_execution_object_event(
      "KVCache", "read_view", std::string(), identity(), length);
  return at::narrow(storage(), state_->sequence_dim_, start, length);
}

Tensor KVCacheObject::append_view(const int64_t length) {
  TORCH_CHECK(state_, "KV cache object is not initialized");
  TORCH_CHECK(length >= 0, "KV cache append length must be non-negative");

  std::lock_guard<std::mutex> lock(state_->mutex_);
  const int64_t start = state_->sequence_length_;
  const int64_t end = start + length;
  TORCH_CHECK(
      end <= max_sequence_length(),
      "Requested KV cache append range [",
      start,
      ", ",
      end,
      ") exceeds cache capacity ",
      max_sequence_length());
  state_->sequence_length_ = end;
  execution_object_log_state().kv_append_views.fetch_add(
      1u, std::memory_order_relaxed);
  log_execution_object_event(
      "KVCache", "append_view", std::string(), identity(), length);
  return at::narrow(state_->storage_, state_->sequence_dim_, start, length);
}

api::ExecutionLayout KVCacheObject::execution_layout() const {
  TORCH_CHECK(state_, "KV cache object is not initialized");
  return state_->execution_layout_;
}

api::GPUMemoryLayout KVCacheObject::memory_layout() const {
  TORCH_CHECK(state_, "KV cache object is not initialized");
  return state_->memory_layout_;
}

api::StorageType KVCacheObject::storage_type() const {
  TORCH_CHECK(state_, "KV cache object is not initialized");
  return state_->storage_type_;
}

bool KVCacheObject::persistent() const {
  TORCH_CHECK(state_, "KV cache object is not initialized");
  return state_->persistent_;
}

const void* KVCacheObject::identity() const {
  return state_.get();
}

bool ScratchArena::defined() const {
  return state_ && state_->storage_.defined();
}

const Tensor& ScratchArena::storage() const {
  TORCH_CHECK(state_, "Scratch arena is not initialized");
  return state_->storage_;
}

size_t ScratchArena::size_bytes() const {
  TORCH_CHECK(state_, "Scratch arena is not initialized");
  return state_->size_bytes_;
}

size_t ScratchArena::used_bytes() const {
  TORCH_CHECK(state_, "Scratch arena is not initialized");
  std::lock_guard<std::mutex> lock(state_->mutex_);
  return state_->next_offset_bytes_;
}

size_t ScratchArena::available_bytes() const {
  TORCH_CHECK(state_, "Scratch arena is not initialized");
  const size_t used = used_bytes();
  return used <= state_->size_bytes_ ? (state_->size_bytes_ - used) : 0u;
}

uint32_t ScratchArena::alignment() const {
  TORCH_CHECK(state_, "Scratch arena is not initialized");
  return state_->default_alignment_;
}

void ScratchArena::reset() {
  TORCH_CHECK(state_, "Scratch arena is not initialized");
  std::lock_guard<std::mutex> lock(state_->mutex_);
  state_->next_offset_bytes_ = 0u;
  execution_object_log_state().scratch_resets.fetch_add(
      1u, std::memory_order_relaxed);
  log_execution_object_event(
      "ScratchArena", "reset", std::string(), identity());
}

VulkanScratchSlice ScratchArena::reserve(
    const size_t size_bytes,
    const uint32_t alignment) {
  TORCH_CHECK(state_, "Scratch arena is not initialized");
  const size_t alignment_bytes =
      std::max<size_t>(state_->default_alignment_, alignment == 0u ? 1u : alignment);
  std::lock_guard<std::mutex> lock(state_->mutex_);
  const size_t offset = align_up_size(state_->next_offset_bytes_, alignment_bytes);
  const size_t end = offset + size_bytes;
  TORCH_CHECK(
      end <= state_->size_bytes_,
      "Scratch arena allocation of ",
      size_bytes,
      " bytes exceeds remaining capacity ",
      (offset <= state_->size_bytes_ ? state_->size_bytes_ - offset : 0u));
  state_->next_offset_bytes_ = end;
  execution_object_log_state().scratch_reserves.fetch_add(
      1u, std::memory_order_relaxed);
  record_scratch_reserved_bytes(size_bytes);
  log_execution_object_event(
      "ScratchArena", "reserve", std::string(), identity(), size_bytes);
  return VulkanScratchSlice{offset, size_bytes};
}

api::ExecutionLayout ScratchArena::execution_layout() const {
  TORCH_CHECK(state_, "Scratch arena is not initialized");
  return state_->execution_layout_;
}

api::GPUMemoryLayout ScratchArena::memory_layout() const {
  TORCH_CHECK(state_, "Scratch arena is not initialized");
  return state_->memory_layout_;
}

bool ScratchArena::persistent() const {
  TORCH_CHECK(state_, "Scratch arena is not initialized");
  return state_->persistent_;
}

const void* ScratchArena::identity() const {
  return state_.get();
}

bool ReadbackBufferObject::defined() const {
  return state_ && state_->buffer_;
}

api::VulkanBuffer& ReadbackBufferObject::buffer() const {
  TORCH_CHECK(state_, "Readback buffer object is not initialized");
  return state_->buffer_;
}

size_t ReadbackBufferObject::size_bytes() const {
  TORCH_CHECK(state_, "Readback buffer object is not initialized");
  return state_->size_bytes_;
}

bool ReadbackBufferObject::persistent() const {
  TORCH_CHECK(state_, "Readback buffer object is not initialized");
  return state_->persistent_;
}

std::mutex& ReadbackBufferObject::mutex() const {
  TORCH_CHECK(state_, "Readback buffer object is not initialized");
  return state_->mutex_;
}

const void* ReadbackBufferObject::identity() const {
  return state_.get();
}

KVCacheObject create_vulkan_kv_cache_object(const VulkanKVCacheSpec& spec) {
  TORCH_CHECK(!spec.sizes.empty(), "KV cache spec requires non-empty sizes");
  TORCH_CHECK(
      spec.sequence_dim >= 0 &&
          spec.sequence_dim < safe_downcast<int64_t>(spec.sizes.size()),
      "Invalid KV cache sequence dimension");
  TORCH_CHECK(
      spec.sizes.at(spec.sequence_dim) >= 0,
      "KV cache sequence dimension must be non-negative");

  Tensor storage = create_execution_object_storage(
      spec.sizes,
      spec.dtype,
      spec.execution_layout,
      spec.memory_layout,
      spec.storage_type,
      spec.persistent);
  return KVCacheObject(std::make_shared<KVCacheObject::State>(
      std::move(storage),
      spec.sizes,
      spec.sequence_dim,
      spec.execution_layout,
      spec.memory_layout,
      spec.storage_type,
      spec.persistent));
}

ScratchArena create_vulkan_scratch_arena(const VulkanScratchArenaSpec& spec) {
  TORCH_CHECK(spec.num_bytes > 0u, "Scratch arena requires non-zero size");
  TORCH_CHECK(
      api::uses_buffer_execution(spec.execution_layout),
      "Scratch arena must use a buffer execution layout");

  Tensor storage = create_execution_object_storage(
      {safe_downcast<int64_t>(spec.num_bytes)},
      spec.dtype,
      spec.execution_layout,
      spec.memory_layout,
      spec.storage_type,
      spec.persistent);
  auto state = std::make_shared<ScratchArena::State>(
      std::move(storage),
      spec.num_bytes,
      std::max<uint32_t>(1u, spec.alignment),
      spec.execution_layout,
      spec.memory_layout,
      spec.persistent);
  if (state->storage_.is_vulkan()) {
    convert(state->storage_).set_stack_retire_provenance_source(
        api::VulkanStackRetireProvenanceSource::
            ProgramScratchArenaBackingStorage,
        reinterpret_cast<uint64_t>(state.get()),
        0u);
  }
  return ScratchArena(std::move(state));
}

ReadbackBufferObject create_vulkan_readback_buffer_object(
    const VulkanReadbackBufferSpec& spec) {
  TORCH_CHECK(
      spec.num_bytes > 0u, "Readback buffer requires a non-zero size");

  api::Context* const context = api::context();
  api::VulkanBuffer buffer =
      context->adapter_ptr()->vma().create_storage_buffer(
          spec.num_bytes,
          false,
          true,
          api::MemoryAllocator::BufferHostAccess::RandomRead);
  return ReadbackBufferObject(std::make_shared<ReadbackBufferObject::State>(
      std::move(buffer), spec.num_bytes, spec.persistent));
}

KVCacheObject lookup_or_create_labeled_kv_cache_object(
    const std::string& allocation_label,
    const VulkanKVCacheSpec& spec) {
  TORCH_CHECK(
      !allocation_label.empty(),
      "Labeled KV cache objects require a non-empty allocation label");
  const LabeledKVCacheKey key{
      allocation_label,
      spec.sizes,
      spec.sequence_dim,
      spec.dtype,
      spec.execution_layout,
      spec.memory_layout,
      spec.storage_type,
      spec.persistent,
  };
  if (const auto cached =
          labeled_kv_cache().lookup(
              key,
              hash_labeled_kv_cache_key,
              same_labeled_kv_cache_key)) {
    execution_object_log_state().kv_hits.fetch_add(
        1u, std::memory_order_relaxed);
    log_execution_object_event(
        "KVCache", "hit", allocation_label, cached->identity());
    return *cached;
  }
  KVCacheObject created = create_vulkan_kv_cache_object(spec);
  labeled_kv_cache().store(
      key,
      created,
      hash_labeled_kv_cache_key,
      same_labeled_kv_cache_key);
  execution_object_log_state().kv_stores.fetch_add(
      1u, std::memory_order_relaxed);
  log_execution_object_event(
      "KVCache", "store", allocation_label, created.identity());
  return created;
}

ScratchArena lookup_or_create_labeled_scratch_arena(
    const std::string& allocation_label,
    const VulkanScratchArenaSpec& spec) {
  TORCH_CHECK(
      !allocation_label.empty(),
      "Labeled scratch arenas require a non-empty allocation label");
  const LabeledScratchArenaKey key{
      allocation_label,
      spec.num_bytes,
      spec.alignment,
      spec.dtype,
      spec.execution_layout,
      spec.memory_layout,
      spec.storage_type,
      spec.persistent,
  };
  if (const auto cached = labeled_scratch_arena_cache().lookup(
          key,
          hash_labeled_scratch_arena_key,
          same_labeled_scratch_arena_key)) {
    execution_object_log_state().scratch_hits.fetch_add(
        1u, std::memory_order_relaxed);
    log_execution_object_event(
        "ScratchArena", "hit", allocation_label, cached->identity());
    return *cached;
  }
  ScratchArena created = create_vulkan_scratch_arena(spec);
  labeled_scratch_arena_cache().store(
      key,
      created,
      hash_labeled_scratch_arena_key,
      same_labeled_scratch_arena_key);
  execution_object_log_state().scratch_stores.fetch_add(
      1u, std::memory_order_relaxed);
  log_execution_object_event(
      "ScratchArena", "store", allocation_label, created.identity());
  return created;
}

ReadbackBufferObject lookup_or_create_labeled_readback_buffer_object(
    const std::string& allocation_label,
    const VulkanReadbackBufferSpec& spec) {
  TORCH_CHECK(
      !allocation_label.empty(),
      "Labeled readback buffers require a non-empty allocation label");
  const LabeledReadbackBufferKey key{
      allocation_label,
      spec.num_bytes,
      spec.persistent,
  };
  if (const auto cached = labeled_readback_buffer_cache().lookup(
          key,
          hash_labeled_readback_buffer_key,
          same_labeled_readback_buffer_key)) {
    execution_object_log_state().readback_hits.fetch_add(
        1u, std::memory_order_relaxed);
    log_execution_object_event(
        "ReadbackBuffer",
        "hit",
        allocation_label,
        cached->identity(),
        cached->size_bytes());
    return *cached;
  }

  api::AllocationScope allocation_scope(allocation_label);
  ReadbackBufferObject created = create_vulkan_readback_buffer_object(spec);
  labeled_readback_buffer_cache().store(
      key,
      created,
      hash_labeled_readback_buffer_key,
      same_labeled_readback_buffer_key);
  execution_object_log_state().readback_stores.fetch_add(
      1u, std::memory_order_relaxed);
  log_execution_object_event(
      "ReadbackBuffer",
      "store",
      allocation_label,
      created.identity(),
      created.size_bytes());
  return created;
}

std::optional<ScratchArena> prime_labeled_scratch_arena_for_request(
    const Tensor& reference,
    const VulkanPlanningRequest& request,
    const size_t requested_bytes,
    const char* label_suffix) {
  (void)reference;
  log_execution_object_event(
      "ScratchArena",
      "prime_request",
      make_vulkan_runtime_object_label(request, label_suffix),
      nullptr,
      requested_bytes);
  const auto policy = build_vulkan_runtime_policy(request);
  if (!policy.scratch_arena_plan.has_value()) {
    log_execution_object_event(
        "ScratchArena",
        "skip_no_plan",
        make_vulkan_runtime_object_label(policy.request, label_suffix),
        nullptr,
        requested_bytes);
    return std::nullopt;
  }

  const auto& desc = *policy.scratch_arena_plan;
  const size_t scratch_bytes = std::max(desc.min_arena_bytes, requested_bytes);
  auto arena = lookup_or_create_labeled_scratch_arena(
      make_vulkan_runtime_object_label(policy.request, label_suffix),
      VulkanScratchArenaSpec{
          kByte,
          scratch_bytes,
          desc.alignment,
          api::ExecutionLayout::BUFFER_DIRECT,
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
          api::StorageType::BUFFER,
          desc.prefer_reusable_arena,
      });
  arena.reset();
  arena.reserve(scratch_bytes, desc.alignment);
  return arena;
}

PackedWeightHandle make_packed_weight_handle(
    Tensor weight,
    Tensor bias,
    std::vector<int64_t> logical_weight_sizes,
    const PackedWeightKind kind,
    const bool bias_defined,
    const bool quantized,
    const PackedWeightResidencyClass residency_class) {
  const bool persistent =
      residency_class == PackedWeightResidencyClass::PersistentInference;
  const size_t resident_nbytes = packed_weight_handle_nbytes(weight, bias);
  return PackedWeightHandle(
      mark_tensor_execution(
          weight, api::ExecutionLayout::PACKED_WEIGHT, persistent),
      mark_tensor_execution(
          bias, api::ExecutionLayout::PACKED_WEIGHT, persistent),
      std::move(logical_weight_sizes),
      kind,
      bias_defined,
      residency_class,
      quantized,
      api::ExecutionLayout::PACKED_WEIGHT,
      resident_nbytes);
}

std::optional<PackedWeightHandle> lookup_packed_weight_handle(
    const Tensor& source_weight,
    const std::optional<Tensor>& source_bias,
    IntArrayRef logical_weight_sizes,
    const PackedWeightKind kind,
    const bool quantized,
    const uint64_t options_key) {
  return packed_weight_residency_manager().lookup(
      source_weight,
      normalized_optional_tensor(source_bias),
      logical_weight_sizes,
      kind,
      quantized,
      options_key);
}

void store_packed_weight_handle(
    const Tensor& source_weight,
    const std::optional<Tensor>& source_bias,
    IntArrayRef logical_weight_sizes,
    const PackedWeightKind kind,
    const PackedWeightHandle& handle,
    const bool quantized,
    const uint64_t options_key) {
  packed_weight_residency_manager().store(
      source_weight,
      normalized_optional_tensor(source_bias),
      logical_weight_sizes,
      kind,
      handle,
      quantized,
      options_key);
}

std::optional<c10::intrusive_ptr<LinearPackedContext>> lookup_linear_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  if (!weight.is_vulkan() || weight.dim() != 2) {
    return std::nullopt;
  }

  const auto normalized_bias = normalized_optional_tensor(bias);
  if (linear_cache_logging_enabled()) {
    linear_cache_log_state().lookups.fetch_add(1u, std::memory_order_relaxed);
  }

  const int64_t weight_version = tensor_version_or_zero(weight);
  const int64_t bias_version =
      normalized_bias ? tensor_version_or_zero(*normalized_bias) : 0u;

  const LinearContextCacheKey query{
      make_tensor_weak_ref(weight),
      normalized_bias ? make_tensor_weak_ref(*normalized_bias) : std::nullopt,
      tensor_identity_ptr(weight),
      normalized_bias ? tensor_identity_ptr(*normalized_bias) : nullptr,
      weight_version,
      bias_version,
  };
  bool pruned = false;
  if (should_prune_linear_context_caches()) {
    prune_expired_linear_context_cache_entries();
    pruned = true;
  }
  if (const auto cached =
          linear_context_cache().lookup(
              query,
              hash_linear_context_cache_key,
              same_linear_context_cache_key)) {
    if (linear_cache_logging_enabled()) {
      linear_cache_log_state().hits.fetch_add(1u, std::memory_order_relaxed);
      log_linear_cache_event(
          "linear", pruned ? "hit_after_prune" : "hit");
    }
    return cached;
  }
  if (!pruned) {
    prune_expired_linear_context_cache_entries();
    if (const auto cached = linear_context_cache().lookup(
            query,
            hash_linear_context_cache_key,
            same_linear_context_cache_key)) {
      if (linear_cache_logging_enabled()) {
        linear_cache_log_state().hits.fetch_add(1u, std::memory_order_relaxed);
        log_linear_cache_event("linear", "hit_after_miss_prune");
      }
      return cached;
    }
  }

  if (linear_cache_logging_enabled()) {
    linear_cache_log_state().misses.fetch_add(1u, std::memory_order_relaxed);
    log_linear_cache_event("linear", "miss");
  }

  return std::nullopt;
}

void store_linear_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const c10::intrusive_ptr<LinearPackedContext>& context) {
  if (!weight.is_vulkan() || weight.dim() != 2) {
    return;
  }

  const auto normalized_bias = normalized_optional_tensor(bias);
  if (linear_cache_logging_enabled()) {
    linear_cache_log_state().stores.fetch_add(1u, std::memory_order_relaxed);
    log_linear_cache_event("linear", "store");
  }

  if (should_prune_linear_context_caches()) {
    prune_expired_linear_context_cache_entries();
  }
  linear_context_cache().store(
      LinearContextCacheKey{
          make_tensor_weak_ref(weight),
          normalized_bias ? make_tensor_weak_ref(*normalized_bias)
                          : std::nullopt,
          tensor_identity_ptr(weight),
          normalized_bias ? tensor_identity_ptr(*normalized_bias) : nullptr,
          tensor_version_or_zero(weight),
          normalized_bias ? tensor_version_or_zero(*normalized_bias) : 0u,
      },
      context,
      hash_linear_context_cache_key,
      same_linear_context_cache_key);
}

std::optional<c10::intrusive_ptr<LinearPackedContext>>
lookup_labeled_linear_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const std::string& allocation_label) {
  if (!weight.is_vulkan() || weight.dim() != 2 || allocation_label.empty()) {
    return std::nullopt;
  }

  const auto normalized_bias = normalized_optional_tensor(bias);
  if (linear_cache_logging_enabled()) {
    linear_cache_log_state().lookups.fetch_add(1u, std::memory_order_relaxed);
  }

  const int64_t weight_version = tensor_version_or_zero(weight);
  const int64_t bias_version =
      normalized_bias ? tensor_version_or_zero(*normalized_bias) : 0u;

  const LabeledLinearContextCacheKey query{
      make_tensor_weak_ref(weight),
      normalized_bias ? make_tensor_weak_ref(*normalized_bias) : std::nullopt,
      tensor_identity_ptr(weight),
      normalized_bias ? tensor_identity_ptr(*normalized_bias) : nullptr,
      weight_version,
      bias_version,
      allocation_label,
  };
  bool pruned = false;
  if (should_prune_linear_context_caches()) {
    prune_expired_labeled_linear_context_cache_entries();
    pruned = true;
  }
  if (const auto cached = labeled_linear_context_cache().lookup(
          query,
          hash_labeled_linear_context_cache_key,
          same_labeled_linear_context_cache_key)) {
    if (linear_cache_logging_enabled()) {
      linear_cache_log_state().hits.fetch_add(1u, std::memory_order_relaxed);
      log_linear_cache_event(
          "labeled_linear",
          pruned ? "hit_after_prune" : "hit",
          allocation_label);
    }
    return cached;
  }
  if (!pruned) {
    prune_expired_labeled_linear_context_cache_entries();
    if (const auto cached = labeled_linear_context_cache().lookup(
            query,
            hash_labeled_linear_context_cache_key,
            same_labeled_linear_context_cache_key)) {
      if (linear_cache_logging_enabled()) {
        linear_cache_log_state().hits.fetch_add(1u, std::memory_order_relaxed);
        log_linear_cache_event(
            "labeled_linear", "hit_after_miss_prune", allocation_label);
      }
      return cached;
    }
  }

  if (linear_cache_logging_enabled()) {
    linear_cache_log_state().misses.fetch_add(1u, std::memory_order_relaxed);
    log_linear_cache_event("labeled_linear", "miss", allocation_label);
  }

  return std::nullopt;
}

void store_labeled_linear_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const std::string& allocation_label,
    const c10::intrusive_ptr<LinearPackedContext>& context) {
  if (!weight.is_vulkan() || weight.dim() != 2 || allocation_label.empty()) {
    return;
  }

  const auto normalized_bias = normalized_optional_tensor(bias);
  if (linear_cache_logging_enabled()) {
    linear_cache_log_state().stores.fetch_add(1u, std::memory_order_relaxed);
    log_linear_cache_event("labeled_linear", "store", allocation_label);
  }

  if (should_prune_linear_context_caches()) {
    prune_expired_labeled_linear_context_cache_entries();
  }
  labeled_linear_context_cache().store(
      LabeledLinearContextCacheKey{
          make_tensor_weak_ref(weight),
          normalized_bias ? make_tensor_weak_ref(*normalized_bias)
                          : std::nullopt,
          tensor_identity_ptr(weight),
          normalized_bias ? tensor_identity_ptr(*normalized_bias) : nullptr,
          tensor_version_or_zero(weight),
          normalized_bias ? tensor_version_or_zero(*normalized_bias) : 0u,
          allocation_label,
      },
      context,
      hash_labeled_linear_context_cache_key,
      same_labeled_linear_context_cache_key);
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
