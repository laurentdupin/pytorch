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
#include <ATen/native/vulkan/ops/Utils.h>

#include <atomic>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <mutex>
#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

using namespace api::utils;

constexpr size_t kPackedWeightResidencyMaxEntries = 256u;
constexpr size_t kLinearContextCacheSize = 128u;
constexpr size_t kExecutionObjectCacheSize = 64u;

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
  Tensor weight_ref;
  std::optional<Tensor> bias_ref;
  int64_t weight_version;
  int64_t bias_version;
  std::vector<int64_t> logical_weight_sizes;
  PackedWeightKind kind;
  PackedWeightResidencyClass residency_class;
  bool quantized;
  uint64_t options_key;
  PackedWeightHandle handle;
};

size_t packed_weight_cache_limit_bytes() {
  static const size_t limit_bytes = []() {
    constexpr size_t kDefaultLimitBytes = size_t{2} * 1024u * 1024u * 1024u;
    const char* env =
        std::getenv("PYTORCH_VULKAN_PACKED_WEIGHT_CACHE_LIMIT_MB");
    if (!env || *env == '\0') {
      return kDefaultLimitBytes;
    }

    std::istringstream stream(env);
    size_t limit_mb = 0u;
    stream >> limit_mb;
    if (!stream || limit_mb == 0u) {
      return kDefaultLimitBytes;
    }
    return limit_mb * 1024u * 1024u;
  }();
  return limit_bytes;
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
  std::atomic<uint64_t> stores{0u};
  std::atomic<uint64_t> evictions{0u};
  std::atomic<uint64_t> cache_bytes{0u};
  std::atomic<uint64_t> peak_cache_bytes{0u};
  std::atomic<uint64_t> persistent_cache_bytes{0u};
  std::atomic<uint64_t> peak_persistent_cache_bytes{0u};

  ~PackedWeightResidencyLogState() {
    if (!packed_weight_cache_logging_enabled()) {
      return;
    }

    std::ofstream out(packed_weight_cache_log_path(), std::ios::app);
    out << "packed_weight_residency: lookups="
        << lookups.load(std::memory_order_relaxed)
        << " hits=" << hits.load(std::memory_order_relaxed)
        << " stores=" << stores.load(std::memory_order_relaxed)
        << " evictions=" << evictions.load(std::memory_order_relaxed)
        << " cache_bytes=" << cache_bytes.load(std::memory_order_relaxed)
        << " peak_cache_bytes="
        << peak_cache_bytes.load(std::memory_order_relaxed)
        << " persistent_cache_bytes="
        << persistent_cache_bytes.load(std::memory_order_relaxed)
        << " peak_persistent_cache_bytes="
        << peak_persistent_cache_bytes.load(std::memory_order_relaxed)
        << " cache_limit_bytes=" << packed_weight_cache_limit_bytes() << '\n';
  }
};

PackedWeightResidencyLogState& packed_weight_cache_log_state() {
  static PackedWeightResidencyLogState state;
  return state;
}

class PackedWeightResidencyManager final {
 private:
  std::mutex mutex_;
  std::deque<PackedWeightResidencyEntry> cache_;
  size_t cache_bytes_{0u};
  size_t persistent_cache_bytes_{0u};

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
    return entry.weight_ref.unsafeGetTensorImpl() ==
            source_weight.unsafeGetTensorImpl() &&
        entry.weight_version == weight_version &&
        same_optional_tensor(entry.bias_ref, normalized_bias) &&
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
  }

  void erase_entry_locked(
      std::deque<PackedWeightResidencyEntry>::iterator entry_it,
      const bool count_eviction) {
    cache_bytes_ -= entry_it->handle.resident_nbytes();
    if (
        entry_it->residency_class ==
        PackedWeightResidencyClass::PersistentInference) {
      persistent_cache_bytes_ -= entry_it->handle.resident_nbytes();
    }
    cache_.erase(entry_it);
    if (count_eviction && packed_weight_cache_logging_enabled()) {
      packed_weight_cache_log_state().evictions.fetch_add(
          1u, std::memory_order_relaxed);
    }
  }

  std::deque<PackedWeightResidencyEntry>::iterator
  select_eviction_candidate_locked() {
    auto transient_it = cache_.end();
    for (auto it = cache_.end(); it != cache_.begin();) {
      --it;
      if (
          it->residency_class == PackedWeightResidencyClass::Transient &&
          it->handle.defined()) {
        transient_it = it;
        break;
      }
    }
    if (transient_it != cache_.end()) {
      return transient_it;
    }
    return cache_.empty() ? cache_.end() : std::prev(cache_.end());
  }

  void trim_locked() {
    while (
        cache_.size() > kPackedWeightResidencyMaxEntries ||
        cache_bytes_ > packed_weight_cache_limit_bytes()) {
      auto victim = select_eviction_candidate_locked();
      if (victim == cache_.end()) {
        break;
      }
      erase_entry_locked(victim, true);
    }
    update_log_snapshot_locked();
  }

 public:
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

    if (packed_weight_cache_logging_enabled()) {
      packed_weight_cache_log_state().lookups.fetch_add(
          1u, std::memory_order_relaxed);
    }

    const int64_t weight_version = tensor_version_or_zero(source_weight);
    const int64_t bias_version =
        normalized_bias ? tensor_version_or_zero(*normalized_bias) : 0u;

    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
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
        continue;
      }

      PackedWeightHandle handle = it->handle;
      if (it != cache_.begin()) {
        PackedWeightResidencyEntry entry = std::move(*it);
        cache_.erase(it);
        cache_.emplace_front(std::move(entry));
        handle = cache_.front().handle;
      }

      if (packed_weight_cache_logging_enabled()) {
        packed_weight_cache_log_state().hits.fetch_add(
            1u, std::memory_order_relaxed);
      }
      update_log_snapshot_locked();
      return handle;
    }

    update_log_snapshot_locked();
    return std::nullopt;
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
    entry.weight_ref = source_weight;
    entry.bias_ref = normalized_bias;
    entry.weight_version = tensor_version_or_zero(source_weight);
    entry.bias_version =
        normalized_bias ? tensor_version_or_zero(*normalized_bias) : 0u;
    entry.logical_weight_sizes = std::vector<int64_t>(
        logical_weight_sizes.begin(), logical_weight_sizes.end());
    entry.kind = kind;
    entry.residency_class = handle.residency_class();
    entry.quantized = quantized;
    entry.options_key = options_key;
    entry.handle = handle;

    if (packed_weight_cache_logging_enabled()) {
      packed_weight_cache_log_state().stores.fetch_add(
          1u, std::memory_order_relaxed);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = cache_.begin(); it != cache_.end(); ++it) {
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
        continue;
      }
      erase_entry_locked(it, false);
      break;
    }

    cache_bytes_ += handle.resident_nbytes();
    if (
        handle.residency_class() ==
        PackedWeightResidencyClass::PersistentInference) {
      persistent_cache_bytes_ += handle.resident_nbytes();
    }
    cache_.emplace_front(std::move(entry));
    trim_locked();
  }
};

PackedWeightResidencyManager& packed_weight_residency_manager() {
  static PackedWeightResidencyManager manager;
  return manager;
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

struct LinearCacheLogState final {
  std::atomic<uint64_t> lookups{0u};
  std::atomic<uint64_t> hits{0u};
  std::atomic<uint64_t> stores{0u};

  ~LinearCacheLogState() {
    if (!linear_cache_logging_enabled()) {
      return;
    }

    std::ofstream out(linear_cache_log_path(), std::ios::app);
    out << "linear_cache: lookups=" << lookups.load(std::memory_order_relaxed)
        << " hits=" << hits.load(std::memory_order_relaxed)
        << " stores=" << stores.load(std::memory_order_relaxed) << '\n';
  }
};

LinearCacheLogState& linear_cache_log_state() {
  static LinearCacheLogState state;
  return state;
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
  Tensor weight_ref;
  std::optional<Tensor> bias_ref;
  int64_t weight_version;
  int64_t bias_version;
};

bool same_linear_context_cache_key(
    const LinearContextCacheKey& lhs,
    const LinearContextCacheKey& rhs) {
  return lhs.weight_ref.unsafeGetTensorImpl() ==
          rhs.weight_ref.unsafeGetTensorImpl() &&
      lhs.weight_version == rhs.weight_version &&
      same_optional_tensor(lhs.bias_ref, rhs.bias_ref) &&
      lhs.bias_version == rhs.bias_version;
}

InferenceLruCache<
    LinearContextCacheKey,
    c10::intrusive_ptr<LinearPackedContext>>&
linear_context_cache() {
  static InferenceLruCache<
      LinearContextCacheKey,
      c10::intrusive_ptr<LinearPackedContext>>
      cache{kLinearContextCacheSize};
  return cache;
}

struct LabeledLinearContextCacheKey final {
  Tensor weight_ref;
  std::optional<Tensor> bias_ref;
  int64_t weight_version;
  int64_t bias_version;
  std::string allocation_label;
};

bool same_labeled_linear_context_cache_key(
    const LabeledLinearContextCacheKey& lhs,
    const LabeledLinearContextCacheKey& rhs) {
  return lhs.weight_ref.unsafeGetTensorImpl() ==
          rhs.weight_ref.unsafeGetTensorImpl() &&
      lhs.weight_version == rhs.weight_version &&
      same_optional_tensor(lhs.bias_ref, rhs.bias_ref) &&
      lhs.bias_version == rhs.bias_version &&
      lhs.allocation_label == rhs.allocation_label;
}

InferenceLruCache<
    LabeledLinearContextCacheKey,
    c10::intrusive_ptr<LinearPackedContext>>&
labeled_linear_context_cache() {
  static InferenceLruCache<
      LabeledLinearContextCacheKey,
      c10::intrusive_ptr<LinearPackedContext>>
      cache{kLinearContextCacheSize};
  return cache;
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

InferenceLruCache<LabeledKVCacheKey, KVCacheObject>& labeled_kv_cache() {
  static InferenceLruCache<LabeledKVCacheKey, KVCacheObject> cache{
      kExecutionObjectCacheSize};
  return cache;
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

InferenceLruCache<LabeledScratchArenaKey, ScratchArena>&
labeled_scratch_arena_cache() {
  static InferenceLruCache<LabeledScratchArenaKey, ScratchArena> cache{
      kExecutionObjectCacheSize};
  return cache;
}

std::string runtime_execution_object_label(
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

} // namespace

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
  }
  return "PackedWeight";
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
  return ScratchArena(std::make_shared<ScratchArena::State>(
      std::move(storage),
      spec.num_bytes,
      std::max<uint32_t>(1u, spec.alignment),
      spec.execution_layout,
      spec.memory_layout,
      spec.persistent));
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
          labeled_kv_cache().lookup(key, same_labeled_kv_cache_key)) {
    execution_object_log_state().kv_hits.fetch_add(
        1u, std::memory_order_relaxed);
    log_execution_object_event(
        "KVCache", "hit", allocation_label, cached->identity());
    return *cached;
  }
  KVCacheObject created = create_vulkan_kv_cache_object(spec);
  labeled_kv_cache().store(key, created, same_labeled_kv_cache_key);
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
          key, same_labeled_scratch_arena_key)) {
    execution_object_log_state().scratch_hits.fetch_add(
        1u, std::memory_order_relaxed);
    log_execution_object_event(
        "ScratchArena", "hit", allocation_label, cached->identity());
    return *cached;
  }
  ScratchArena created = create_vulkan_scratch_arena(spec);
  labeled_scratch_arena_cache().store(
      key, created, same_labeled_scratch_arena_key);
  execution_object_log_state().scratch_stores.fetch_add(
      1u, std::memory_order_relaxed);
  log_execution_object_event(
      "ScratchArena", "store", allocation_label, created.identity());
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
      runtime_execution_object_label(request, label_suffix),
      nullptr,
      requested_bytes);
  const auto policy = build_vulkan_runtime_policy(request);
  if (!policy.scratch_arena_plan.has_value()) {
    log_execution_object_event(
        "ScratchArena",
        "skip_no_plan",
        runtime_execution_object_label(policy.request, label_suffix),
        nullptr,
        requested_bytes);
    return std::nullopt;
  }

  const auto& desc = *policy.scratch_arena_plan;
  const size_t scratch_bytes = std::max(desc.min_arena_bytes, requested_bytes);
  auto arena = lookup_or_create_labeled_scratch_arena(
      runtime_execution_object_label(policy.request, label_suffix),
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
      weight,
      normalized_bias,
      weight_version,
      bias_version,
  };
  if (const auto cached =
          linear_context_cache().lookup(query, same_linear_context_cache_key)) {
    if (linear_cache_logging_enabled()) {
      linear_cache_log_state().hits.fetch_add(1u, std::memory_order_relaxed);
    }
    return cached;
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
  }

  linear_context_cache().store(
      LinearContextCacheKey{
          weight,
          normalized_bias,
          tensor_version_or_zero(weight),
          normalized_bias ? tensor_version_or_zero(*normalized_bias) : 0u,
      },
      context,
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
      weight,
      normalized_bias,
      weight_version,
      bias_version,
      allocation_label,
  };
  if (const auto cached = labeled_linear_context_cache().lookup(
          query, same_labeled_linear_context_cache_key)) {
    if (linear_cache_logging_enabled()) {
      linear_cache_log_state().hits.fetch_add(1u, std::memory_order_relaxed);
    }
    return cached;
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
  }

  labeled_linear_context_cache().store(
      LabeledLinearContextCacheKey{
          weight,
          normalized_bias,
          tensor_version_or_zero(weight),
          normalized_bias ? tensor_version_or_zero(*normalized_bias) : 0u,
          allocation_label,
      },
      context,
      same_labeled_linear_context_cache_key);
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
