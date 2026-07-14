#include <ATen/native/vulkan/planning/ExecutionObjects.h>
#include <ATen/native/vulkan/planning/Runtime.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/narrow.h>
#include <ATen/ops/zeros.h>
#endif

#include <ATen/native/vulkan/ops/InferenceCache.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

using namespace api::utils;

constexpr size_t kExecutionObjectCacheSize = 64u;

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

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
