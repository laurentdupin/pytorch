#include <ATen/native/vulkan/planning/PackedWeightCache.h>

#include <ATen/native/vulkan/ops/InferenceCache.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <c10/core/Storage.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>

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
constexpr uint64_t kLinearContextPruneInterval = 64u;
constexpr size_t kLinearContextPruneScanBudget = 32u;
constexpr size_t kLinearContextPruneEraseBudget = 8u;

template <typename T>
void hash_combine(size_t& seed, const T& value) {
  seed ^= std::hash<T>{}(value) + size_t{0x9e3779b97f4a7c15ull} + (seed << 6u) +
      (seed >> 2u);
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

std::string packed_weight_query_aggregate_key(
    IntArrayRef logical_weight_sizes,
    const ScalarType dtype,
    const PackedWeightKind kind,
    const bool quantized,
    const uint64_t options_key) {
  std::ostringstream stream;
  stream << "kind=" << to_string(kind)
         << " logical_weight_shape="
         << format_size_list(
                std::vector<int64_t>(
                    logical_weight_sizes.begin(), logical_weight_sizes.end()))
         << " dtype=" << dtype << " quantized=" << (quantized ? 1 : 0)
         << " options_key=" << options_key;
  return stream.str();
}

std::string packed_weight_query_aggregate_key(
    const Tensor& source_weight,
    IntArrayRef logical_weight_sizes,
    const PackedWeightKind kind,
    const bool quantized,
    const uint64_t options_key) {
  return packed_weight_query_aggregate_key(
      logical_weight_sizes,
      source_weight.defined() ? source_weight.scalar_type()
                              : ScalarType::Undefined,
      kind,
      quantized,
      options_key);
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

struct PackedWeightQueryAggregate final {
  uint64_t lookups{0u};
  uint64_t hits{0u};
  uint64_t misses{0u};
  uint64_t miss_empty_cache{0u};
  uint64_t miss_no_match{0u};
  uint64_t mismatch_tensor_impl{0u};
  uint64_t mismatch_storage_identity{0u};
  uint64_t mismatch_provenance_source{0u};
  uint64_t mismatch_shape_stride_dtype{0u};
  uint64_t mismatch_version{0u};
  uint64_t mismatch_bias{0u};
  uint64_t mismatch_context_device{0u};
  uint64_t stores{0u};
  uint64_t persistent_stores{0u};
  uint64_t transient_stores{0u};
  uint64_t stored_bytes{0u};
  uint64_t persistent_stored_bytes{0u};
  uint64_t transient_stored_bytes{0u};
  uint64_t store_skip_transient{0u};
  uint64_t store_skip_large{0u};
  uint64_t store_skip_other{0u};
  uint64_t skipped_bytes{0u};
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
  std::atomic<uint64_t> mismatch_tensor_impl{0u};
  std::atomic<uint64_t> mismatch_storage_identity{0u};
  std::atomic<uint64_t> mismatch_provenance_source{0u};
  std::atomic<uint64_t> mismatch_shape_stride_dtype{0u};
  std::atomic<uint64_t> mismatch_version{0u};
  std::atomic<uint64_t> mismatch_bias{0u};
  std::atomic<uint64_t> mismatch_context_device{0u};
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
        << " mismatch_tensor_impl="
        << mismatch_tensor_impl.load(std::memory_order_relaxed)
        << " mismatch_storage_identity="
        << mismatch_storage_identity.load(std::memory_order_relaxed)
        << " mismatch_provenance_source="
        << mismatch_provenance_source.load(std::memory_order_relaxed)
        << " mismatch_shape_stride_dtype="
        << mismatch_shape_stride_dtype.load(std::memory_order_relaxed)
        << " mismatch_version="
        << mismatch_version.load(std::memory_order_relaxed)
        << " mismatch_bias=" << mismatch_bias.load(std::memory_order_relaxed)
        << " mismatch_context_device="
        << mismatch_context_device.load(std::memory_order_relaxed)
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
  state.mismatch_tensor_impl.store(0u, std::memory_order_relaxed);
  state.mismatch_storage_identity.store(0u, std::memory_order_relaxed);
  state.mismatch_provenance_source.store(0u, std::memory_order_relaxed);
  state.mismatch_shape_stride_dtype.store(0u, std::memory_order_relaxed);
  state.mismatch_version.store(0u, std::memory_order_relaxed);
  state.mismatch_bias.store(0u, std::memory_order_relaxed);
  state.mismatch_context_device.store(0u, std::memory_order_relaxed);
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
  std::deque<RetiredPackedWeightMetadata> retired_metadata;
  {
    std::lock_guard<std::mutex> lock(retired_packed_weight_mutex());
    retired_handles.swap(retired_packed_weight_handles());
    retired_metadata.swap(leaked_retired_packed_weight_metadata());
  }
  if (retired_handles.empty()) {
    return false;
  }
  const size_t retired_count = retired_handles.size();
  if (c10::InferenceMode::is_enabled()) {
    c10::InferenceMode inference_mode_guard(false);
    retired_handles.clear();
  } else {
    retired_handles.clear();
  }
  retired_metadata.clear();
  log_vulkan_op_hit(
      std::string("vulkan_packed_weight_release.released count=") +
      std::to_string(retired_count));
  return true;
}

class PackedWeightResidencyManager final {
 private:
  static constexpr size_t kRecentMissRowsLimit = 64u;
  std::mutex mutex_;
  std::deque<PackedWeightResidencyEntry> cache_;
  std::deque<std::string> recent_misses_;
  std::unordered_map<std::string, PackedWeightQueryAggregate> query_aggregate_;
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

  struct MissBreakdown final {
    bool tensor_impl{false};
    bool storage_identity{false};
    bool provenance_source{false};
    bool shape_stride_dtype{false};
    bool version{false};
    bool bias{false};
    bool context_device{false};
    int score{0};
    std::string row;
  };

  static bool same_logical_sizes(
      IntArrayRef lhs,
      const std::vector<int64_t>& rhs) {
    return lhs.size() == rhs.size() &&
        std::equal(lhs.begin(), lhs.end(), rhs.begin());
  }

  static bool same_weight_shape_stride_dtype(
      const PackedWeightResidencyEntry& entry,
      const Tensor& source_weight) {
    return entry.weight_storage_offset == source_weight.storage_offset() &&
        entry.weight_dtype == source_weight.scalar_type() &&
        entry.logical_weight_sizes.size() ==
            static_cast<size_t>(source_weight.dim()) &&
        entry.weight_strides.size() == static_cast<size_t>(source_weight.dim()) &&
        std::equal(
               entry.logical_weight_sizes.begin(),
               entry.logical_weight_sizes.end(),
               source_weight.sizes().begin()) &&
        std::equal(
               entry.weight_strides.begin(),
               entry.weight_strides.end(),
               source_weight.strides().begin());
  }

  static MissBreakdown analyze_miss_candidate(
      const PackedWeightResidencyEntry& entry,
      const Tensor& source_weight,
      const std::optional<Tensor>& normalized_bias,
      const int64_t weight_version,
      const int64_t bias_version,
      IntArrayRef logical_weight_sizes,
      const PackedWeightKind kind,
      const bool quantized,
      const uint64_t options_key) {
    const bool tensor_impl_matches =
        packed_weight_ref_matches_tensor(entry.weight_ref, source_weight);
    const bool c10_storage_matches =
        entry.weight_storage_identity != nullptr &&
        tensor_storage_identity_ptr(source_weight) == entry.weight_storage_identity;
    const bool vulkan_storage_matches =
        entry.weight_vulkan_storage_identity != nullptr &&
        tensor_vulkan_storage_identity_ptr(source_weight) ==
            entry.weight_vulkan_storage_identity;
    const uint64_t source_weight_key =
        tensor_packed_weight_source_key(source_weight);
    const bool provenance_source_matches =
        entry.weight_source_key != 0u &&
        source_weight_key == entry.weight_source_key;
    const bool shape_stride_dtype_matches =
        same_weight_shape_stride_dtype(entry, source_weight) &&
        same_logical_sizes(logical_weight_sizes, entry.logical_weight_sizes);
    const bool version_matches = entry.weight_version == weight_version &&
        entry.bias_version == bias_version;
    const bool bias_matches =
        bias_matches_optional_tensor(entry, normalized_bias);
    const bool context_device_matches = entry.kind == kind &&
        entry.quantized == quantized && entry.options_key == options_key;

    MissBreakdown breakdown;
    breakdown.tensor_impl = !tensor_impl_matches;
    breakdown.storage_identity = !(c10_storage_matches || vulkan_storage_matches);
    breakdown.provenance_source = !provenance_source_matches;
    breakdown.shape_stride_dtype = !shape_stride_dtype_matches;
    breakdown.version = !version_matches;
    breakdown.bias = !bias_matches;
    breakdown.context_device = !context_device_matches;
    breakdown.score = (tensor_impl_matches ? 1 : 0) +
        ((c10_storage_matches || vulkan_storage_matches) ? 1 : 0) +
        (provenance_source_matches ? 1 : 0) +
        (shape_stride_dtype_matches ? 1 : 0) + (version_matches ? 1 : 0) +
        (bias_matches ? 1 : 0) + (context_device_matches ? 1 : 0);

    std::ostringstream stream;
    stream << "packed_weight_cache_miss"
           << " query_tensor_impl="
           << reinterpret_cast<uintptr_t>(tensor_identity_ptr(source_weight))
           << " query_base_tensor_impl="
           << reinterpret_cast<uintptr_t>(
                  tensor_identity_ptr(packed_weight_identity_tensor(source_weight)))
           << " query_storage_identity="
           << reinterpret_cast<uintptr_t>(tensor_storage_identity_ptr(source_weight))
           << " query_vulkan_storage_identity="
           << reinterpret_cast<uintptr_t>(
                  tensor_vulkan_storage_identity_ptr(source_weight))
           << " query_source_key=" << source_weight_key
           << " query_shape=" << format_size_list(source_weight.sizes().vec())
           << " query_stride=" << format_size_list(source_weight.strides().vec())
           << " query_dtype=" << source_weight.scalar_type()
           << " query_version=" << weight_version
           << " query_bias_present="
           << (normalized_bias && normalized_bias->defined() ? 1 : 0)
           << " query_bias_source_key="
           << (normalized_bias ? tensor_packed_weight_source_key(*normalized_bias)
                               : 0u)
           << " query_bias_version=" << bias_version
           << " stored_tensor_alive="
           << (weak_tensor_ref_alive(entry.weight_ref) ? 1 : 0)
           << " stored_storage_alive="
           << (weak_storage_ref_alive(entry.weight_storage_ref) ? 1 : 0)
           << " stored_vulkan_storage_alive="
           << (weak_vulkan_storage_ref_alive(entry.weight_vulkan_storage_ref)
                   ? 1
                   : 0)
           << " stored_storage_identity="
           << reinterpret_cast<uintptr_t>(entry.weight_storage_identity)
           << " stored_vulkan_storage_identity="
           << reinterpret_cast<uintptr_t>(entry.weight_vulkan_storage_identity)
           << " stored_source_key=" << entry.weight_source_key
           << " stored_shape=" << format_size_list(entry.logical_weight_sizes)
           << " stored_stride=" << format_size_list(entry.weight_strides)
           << " stored_dtype=" << entry.weight_dtype
           << " stored_version=" << entry.weight_version
           << " stored_bias_present="
           << (entry.bias_ref.has_value() || entry.bias_source_key != 0u ? 1 : 0)
           << " stored_bias_source_key=" << entry.bias_source_key
           << " stored_bias_version=" << entry.bias_version
           << " mismatch_tensor_impl=" << (breakdown.tensor_impl ? 1 : 0)
           << " mismatch_storage_identity="
           << (breakdown.storage_identity ? 1 : 0)
           << " mismatch_provenance_source="
           << (breakdown.provenance_source ? 1 : 0)
           << " mismatch_shape_stride_dtype="
           << (breakdown.shape_stride_dtype ? 1 : 0)
           << " mismatch_version=" << (breakdown.version ? 1 : 0)
           << " mismatch_bias=" << (breakdown.bias ? 1 : 0)
           << " mismatch_context_device="
           << (breakdown.context_device ? 1 : 0);
    breakdown.row = stream.str();
    return breakdown;
  }

  static void note_miss_breakdown(const MissBreakdown& breakdown) {
    auto& log_state = packed_weight_cache_log_state();
    if (breakdown.tensor_impl) {
      log_state.mismatch_tensor_impl.fetch_add(1u, std::memory_order_relaxed);
    }
    if (breakdown.storage_identity) {
      log_state.mismatch_storage_identity.fetch_add(
          1u, std::memory_order_relaxed);
    }
    if (breakdown.provenance_source) {
      log_state.mismatch_provenance_source.fetch_add(
          1u, std::memory_order_relaxed);
    }
    if (breakdown.shape_stride_dtype) {
      log_state.mismatch_shape_stride_dtype.fetch_add(
          1u, std::memory_order_relaxed);
    }
    if (breakdown.version) {
      log_state.mismatch_version.fetch_add(1u, std::memory_order_relaxed);
    }
    if (breakdown.bias) {
      log_state.mismatch_bias.fetch_add(1u, std::memory_order_relaxed);
    }
    if (breakdown.context_device) {
      log_state.mismatch_context_device.fetch_add(
          1u, std::memory_order_relaxed);
    }
  }

  static void note_aggregate_miss_breakdown(
      PackedWeightQueryAggregate& aggregate,
      const MissBreakdown& breakdown) {
    if (breakdown.tensor_impl) {
      ++aggregate.mismatch_tensor_impl;
    }
    if (breakdown.storage_identity) {
      ++aggregate.mismatch_storage_identity;
    }
    if (breakdown.provenance_source) {
      ++aggregate.mismatch_provenance_source;
    }
    if (breakdown.shape_stride_dtype) {
      ++aggregate.mismatch_shape_stride_dtype;
    }
    if (breakdown.version) {
      ++aggregate.mismatch_version;
    }
    if (breakdown.bias) {
      ++aggregate.mismatch_bias;
    }
    if (breakdown.context_device) {
      ++aggregate.mismatch_context_device;
    }
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
            << " mismatch_tensor_impl="
            << log_state.mismatch_tensor_impl.load(std::memory_order_relaxed)
            << " mismatch_storage_identity="
            << log_state.mismatch_storage_identity.load(std::memory_order_relaxed)
            << " mismatch_provenance_source="
            << log_state.mismatch_provenance_source.load(
                   std::memory_order_relaxed)
            << " mismatch_shape_stride_dtype="
            << log_state.mismatch_shape_stride_dtype.load(
                   std::memory_order_relaxed)
            << " mismatch_version="
            << log_state.mismatch_version.load(std::memory_order_relaxed)
            << " mismatch_bias="
            << log_state.mismatch_bias.load(std::memory_order_relaxed)
            << " mismatch_context_device="
            << log_state.mismatch_context_device.load(std::memory_order_relaxed)
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
    std::vector<std::pair<std::string, PackedWeightQueryAggregate>>
        aggregate_rows(
            query_aggregate_.begin(),
            query_aggregate_.end());
    std::sort(
        aggregate_rows.begin(),
        aggregate_rows.end(),
        [](const auto& lhs, const auto& rhs) {
          const auto lhs_pressure =
              lhs.second.misses + lhs.second.store_skip_transient +
              lhs.second.store_skip_large + lhs.second.store_skip_other;
          const auto rhs_pressure =
              rhs.second.misses + rhs.second.store_skip_transient +
              rhs.second.store_skip_large + rhs.second.store_skip_other;
          if (lhs_pressure != rhs_pressure) {
            return lhs_pressure > rhs_pressure;
          }
          return lhs.first < rhs.first;
        });
    for (const auto& item : aggregate_rows) {
      const PackedWeightQueryAggregate& aggregate = item.second;
      std::ostringstream stream;
      stream << "packed_weight_query_aggregate " << item.first
             << " lookups=" << aggregate.lookups
             << " hits=" << aggregate.hits
             << " misses=" << aggregate.misses
             << " miss_empty_cache=" << aggregate.miss_empty_cache
             << " miss_no_match=" << aggregate.miss_no_match
             << " mismatch_tensor_impl=" << aggregate.mismatch_tensor_impl
             << " mismatch_storage_identity="
             << aggregate.mismatch_storage_identity
             << " mismatch_provenance_source="
             << aggregate.mismatch_provenance_source
             << " mismatch_shape_stride_dtype="
             << aggregate.mismatch_shape_stride_dtype
             << " mismatch_version=" << aggregate.mismatch_version
             << " mismatch_bias=" << aggregate.mismatch_bias
             << " mismatch_context_device="
             << aggregate.mismatch_context_device
             << " stores=" << aggregate.stores
             << " persistent_stores=" << aggregate.persistent_stores
             << " transient_stores=" << aggregate.transient_stores
             << " stored_bytes=" << aggregate.stored_bytes
             << " persistent_stored_bytes="
             << aggregate.persistent_stored_bytes
             << " transient_stored_bytes=" << aggregate.transient_stored_bytes
             << " store_skip_transient=" << aggregate.store_skip_transient
             << " store_skip_large=" << aggregate.store_skip_large
             << " store_skip_other=" << aggregate.store_skip_other
             << " skipped_bytes=" << aggregate.skipped_bytes;
      rows.emplace_back(stream.str());
    }
    for (const std::string& row : recent_misses_) {
      rows.emplace_back(row);
    }
    return rows;
  }

  void reset_diagnostics() {
    std::lock_guard<std::mutex> lock(mutex_);
    recent_misses_.clear();
    query_aggregate_.clear();
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
    const std::string aggregate_key = packed_weight_query_aggregate_key(
        source_weight, logical_weight_sizes, kind, quantized, options_key);

    std::deque<PackedWeightResidencyEntry> retired_entries;
    std::optional<PackedWeightHandle> result;
    bool cache_was_empty = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      PackedWeightQueryAggregate& aggregate =
          query_aggregate_[aggregate_key];
      ++aggregate.lookups;
      cache_was_empty = cache_.empty();
      std::optional<MissBreakdown> best_miss;
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
          MissBreakdown breakdown = analyze_miss_candidate(
              *it,
              source_weight,
              normalized_bias,
              weight_version,
              bias_version,
              logical_weight_sizes,
              kind,
              quantized,
              options_key);
          if (!best_miss.has_value() || breakdown.score > best_miss->score) {
            best_miss = std::move(breakdown);
          }
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
        ++aggregate.hits;
        result = handle;
        break;
      }
      if (!result.has_value()) {
        log_state.misses.fetch_add(1u, std::memory_order_relaxed);
        ++aggregate.misses;
        if (cache_was_empty) {
          log_state.miss_empty_cache.fetch_add(1u, std::memory_order_relaxed);
          ++aggregate.miss_empty_cache;
        } else {
          log_state.miss_no_match.fetch_add(1u, std::memory_order_relaxed);
          ++aggregate.miss_no_match;
          if (best_miss.has_value()) {
            note_miss_breakdown(*best_miss);
            note_aggregate_miss_breakdown(aggregate, *best_miss);
            recent_misses_.emplace_back(std::move(best_miss->row));
            while (recent_misses_.size() > kRecentMissRowsLimit) {
              recent_misses_.pop_front();
            }
          }
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
    const std::string aggregate_key = packed_weight_query_aggregate_key(
        source_weight, logical_weight_sizes, kind, quantized, options_key);

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
      PackedWeightQueryAggregate& aggregate =
          query_aggregate_[aggregate_key];
      ++aggregate.stores;
      aggregate.stored_bytes +=
          static_cast<uint64_t>(handle.resident_nbytes());
      if (
          handle.residency_class() ==
          PackedWeightResidencyClass::PersistentInference) {
        persistent_cache_bytes_ += handle.resident_nbytes();
        ++aggregate.persistent_stores;
        aggregate.persistent_stored_bytes +=
            static_cast<uint64_t>(handle.resident_nbytes());
      } else {
        ++aggregate.transient_stores;
        aggregate.transient_stored_bytes +=
            static_cast<uint64_t>(handle.resident_nbytes());
      }
      cache_.emplace_front(std::move(entry));
      trim_locked(retired_entries);
    }
    release_retired_entries(retired_entries);
  }

  void note_store_skip(
      IntArrayRef logical_weight_sizes,
      const ScalarType dtype,
      const PackedWeightKind kind,
      const bool quantized,
      const uint64_t options_key,
      const char* reason,
      const size_t resident_nbytes) {
    const std::string aggregate_key = packed_weight_query_aggregate_key(
        logical_weight_sizes, dtype, kind, quantized, options_key);
    std::lock_guard<std::mutex> lock(mutex_);
    PackedWeightQueryAggregate& aggregate = query_aggregate_[aggregate_key];
    const std::string reason_string = reason ? reason : "";
    if (reason_string == "transient") {
      ++aggregate.store_skip_transient;
    } else if (reason_string == "large") {
      ++aggregate.store_skip_large;
    } else {
      ++aggregate.store_skip_other;
    }
    aggregate.skipped_bytes += static_cast<uint64_t>(resident_nbytes);
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
  packed_weight_residency_manager().reset_diagnostics();
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

void note_packed_weight_store_skip(
    IntArrayRef logical_weight_sizes,
    const ScalarType dtype,
    const PackedWeightKind kind,
    const bool quantized,
    const uint64_t options_key,
    const char* reason,
    const size_t resident_nbytes) {
  packed_weight_residency_manager().note_store_skip(
      logical_weight_sizes,
      dtype,
      kind,
      quantized,
      options_key,
      reason,
      resident_nbytes);
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
  if (
      context && context->packed_weight().defined() &&
      context->packed_weight().residency_class() ==
          PackedWeightResidencyClass::Transient) {
    if (linear_cache_logging_enabled()) {
      log_linear_cache_event("linear", "skip_transient");
    }
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
  if (
      context && context->packed_weight().defined() &&
      context->packed_weight().residency_class() ==
          PackedWeightResidencyClass::Transient) {
    if (linear_cache_logging_enabled()) {
      log_linear_cache_event(
          "labeled_linear", "skip_transient", allocation_label);
    }
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
