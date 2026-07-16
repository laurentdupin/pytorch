#include <ATen/native/vulkan/planning/ExecutionPrograms.h>

#include <ATen/native/vulkan/ops/InferenceCache.h>
#include <ATen/native/vulkan/ops/LayoutTransitions.h>
#include <ATen/native/vulkan/ops/Utils.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <mutex>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace utils {

namespace {

constexpr size_t kExecutionProgramCacheSize = 64u;
constexpr size_t kDefaultExecutionProgramCacheLimitBytes =
    size_t{512u} * 1024u * 1024u;

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

size_t execution_program_cache_limit_bytes() {
  static const size_t limit = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_EXECUTION_PROGRAM_CACHE_MB");
    if (!env || !*env) {
      return kDefaultExecutionProgramCacheLimitBytes;
    }
    char* end = nullptr;
    const unsigned long long mb = std::strtoull(env, &end, 10);
    if (!end || *end != '\0' || mb == 0ull) {
      return kDefaultExecutionProgramCacheLimitBytes;
    }
    return static_cast<size_t>(mb) * 1024u * 1024u;
  }();
  return limit;
}

std::string normalize_program_label(
    const std::string& allocation_label,
    const char* fallback) {
  if (!allocation_label.empty()) {
    return allocation_label;
  }
  return std::string(fallback);
}

std::string program_object_label(
    const std::string& allocation_label,
    const char* suffix) {
  return normalize_program_label(allocation_label, suffix) + "." + suffix;
}

const std::string& execution_program_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_EXECUTION_PROGRAM_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool execution_program_logging_enabled() {
  return !execution_program_log_path().empty();
}

std::mutex& execution_program_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

void log_execution_program_event(
    const VulkanExecutionProgramKind kind,
    const char* event,
    const std::string& allocation_label,
    const void* identity,
    const size_t bytes = 0u) {
  if (!execution_program_logging_enabled()) {
    return;
  }

  std::lock_guard<std::mutex> lock(execution_program_log_mutex());
  std::ofstream out(execution_program_log_path(), std::ios::app);
  out << "execution_program event=" << event << " kind="
      << execution_program_kind_name(kind) << " allocation_label="
      << allocation_label << " identity=" << identity;
  if (bytes > 0u) {
    out << " bytes=" << bytes;
  }
  out << '\n';
}

bool same_sizes(
    const std::vector<int64_t>& lhs,
    const std::vector<int64_t>& rhs) {
  return lhs.size() == rhs.size() &&
      std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool same_optional_sizes(
    const std::optional<std::vector<int64_t>>& lhs,
    const std::optional<std::vector<int64_t>>& rhs) {
  if (lhs.has_value() != rhs.has_value()) {
    return false;
  }
  if (!lhs.has_value()) {
    return true;
  }
  return same_sizes(*lhs, *rhs);
}

bool same_scratch_spec(
    const std::optional<VulkanScratchArenaSpec>& lhs,
    const std::optional<VulkanScratchArenaSpec>& rhs) {
  if (lhs.has_value() != rhs.has_value()) {
    return false;
  }
  if (!lhs.has_value()) {
    return true;
  }
  return lhs->dtype == rhs->dtype && lhs->num_bytes == rhs->num_bytes &&
      lhs->alignment == rhs->alignment &&
      lhs->execution_layout == rhs->execution_layout &&
      lhs->memory_layout == rhs->memory_layout &&
      lhs->storage_type == rhs->storage_type &&
      lhs->persistent == rhs->persistent;
}

size_t hash_optional_scratch_spec(
    const std::optional<VulkanScratchArenaSpec>& spec) {
  size_t seed = 0u;
  hash_combine(seed, spec.has_value());
  if (!spec.has_value()) {
    return seed;
  }
  hash_combine(seed, static_cast<int>(spec->dtype));
  hash_combine(seed, spec->num_bytes);
  hash_combine(seed, spec->alignment);
  hash_combine(seed, static_cast<int>(spec->execution_layout));
  hash_combine(seed, static_cast<int>(spec->memory_layout));
  hash_combine(seed, static_cast<int>(spec->storage_type));
  hash_combine(seed, spec->persistent);
  return seed;
}

struct AttentionRuntimeProgramKey final {
  std::string allocation_label;
  VulkanAttentionKernelFamily kernel_family{
      VulkanAttentionKernelFamily::TextureMath};
  std::optional<VulkanScratchArenaSpec> scratch_spec;
  bool persistent{true};
};

bool operator==(
    const AttentionRuntimeProgramKey& lhs,
    const AttentionRuntimeProgramKey& rhs) {
  return lhs.allocation_label == rhs.allocation_label &&
      lhs.kernel_family == rhs.kernel_family &&
      same_scratch_spec(lhs.scratch_spec, rhs.scratch_spec) &&
      lhs.persistent == rhs.persistent;
}

size_t hash_attention_runtime_program_key(
    const AttentionRuntimeProgramKey& key) {
  size_t seed = 0u;
  hash_combine(seed, key.allocation_label);
  hash_combine(seed, static_cast<int>(key.kernel_family));
  hash_combine(seed, hash_optional_scratch_spec(key.scratch_spec));
  hash_combine(seed, key.persistent);
  return seed;
}

InferenceLruCache<AttentionRuntimeProgramKey, AttentionRuntimeProgram>&
attention_runtime_program_cache() {
  static auto* cache =
      new InferenceLruCache<AttentionRuntimeProgramKey, AttentionRuntimeProgram>{
          kExecutionProgramCacheSize, execution_program_cache_limit_bytes()};
  return *cache;
}

struct VisionBackboneProgramKey final {
  std::string allocation_label;
  ScalarType dtype{kFloat};
  int64_t batch_size{1};
  int64_t token_count{1};
  int64_t embed_dim{1};
  int64_t hidden_dim{1};
  int64_t num_heads{1};
  std::optional<VulkanScratchArenaSpec> scratch_spec;
  bool persistent{true};
};

bool operator==(
    const VisionBackboneProgramKey& lhs,
    const VisionBackboneProgramKey& rhs) {
  return lhs.allocation_label == rhs.allocation_label &&
      lhs.dtype == rhs.dtype &&
      lhs.batch_size == rhs.batch_size &&
      lhs.token_count == rhs.token_count &&
      lhs.embed_dim == rhs.embed_dim && lhs.hidden_dim == rhs.hidden_dim &&
      lhs.num_heads == rhs.num_heads &&
      same_scratch_spec(lhs.scratch_spec, rhs.scratch_spec) &&
      lhs.persistent == rhs.persistent;
}

size_t hash_vision_backbone_program_key(const VisionBackboneProgramKey& key) {
  size_t seed = 0u;
  hash_combine(seed, key.allocation_label);
  hash_combine(seed, static_cast<int>(key.dtype));
  hash_combine(seed, key.batch_size);
  hash_combine(seed, key.token_count);
  hash_combine(seed, key.embed_dim);
  hash_combine(seed, key.hidden_dim);
  hash_combine(seed, key.num_heads);
  hash_combine(seed, hash_optional_scratch_spec(key.scratch_spec));
  hash_combine(seed, key.persistent);
  return seed;
}

InferenceLruCache<VisionBackboneProgramKey, VisionBackboneProgram>&
vision_backbone_program_cache() {
  static auto* cache =
      new InferenceLruCache<VisionBackboneProgramKey, VisionBackboneProgram>{
          kExecutionProgramCacheSize, execution_program_cache_limit_bytes()};
  return *cache;
}

struct VisionDecoderProgramKey final {
  std::string allocation_label;
  std::vector<int64_t> input_sizes;
  std::optional<std::vector<int64_t>> skip_sizes;
  std::vector<int64_t> target_sizes;
  int64_t out_channels{1};
  std::optional<VulkanScratchArenaSpec> scratch_spec;
  bool allocate_intermediate_outputs{true};
  bool persistent{true};
};

bool operator==(
    const VisionDecoderProgramKey& lhs,
    const VisionDecoderProgramKey& rhs) {
  return lhs.allocation_label == rhs.allocation_label &&
      same_sizes(lhs.input_sizes, rhs.input_sizes) &&
      same_optional_sizes(lhs.skip_sizes, rhs.skip_sizes) &&
      same_sizes(lhs.target_sizes, rhs.target_sizes) &&
      lhs.out_channels == rhs.out_channels &&
      same_scratch_spec(lhs.scratch_spec, rhs.scratch_spec) &&
      lhs.allocate_intermediate_outputs == rhs.allocate_intermediate_outputs &&
      lhs.persistent == rhs.persistent;
}

size_t hash_vision_decoder_program_key(const VisionDecoderProgramKey& key) {
  size_t seed = 0u;
  hash_combine(seed, key.allocation_label);
  hash_combine_sizes(seed, key.input_sizes);
  hash_combine(seed, key.skip_sizes.has_value());
  if (key.skip_sizes.has_value()) {
    hash_combine_sizes(seed, *key.skip_sizes);
  }
  hash_combine_sizes(seed, key.target_sizes);
  hash_combine(seed, key.out_channels);
  hash_combine(seed, hash_optional_scratch_spec(key.scratch_spec));
  hash_combine(seed, key.allocate_intermediate_outputs);
  hash_combine(seed, key.persistent);
  return seed;
}

InferenceLruCache<VisionDecoderProgramKey, VisionDecoderProgram>&
vision_decoder_program_cache() {
  static auto* cache =
      new InferenceLruCache<VisionDecoderProgramKey, VisionDecoderProgram>{
          kExecutionProgramCacheSize, execution_program_cache_limit_bytes()};
  return *cache;
}

Tensor create_program_buffer_tensor(
    IntArrayRef sizes,
    const ScalarType dtype,
    const bool persistent) {
  return mark_tensor_execution(
      convert(vTensor{
          api::context(),
          sizes.vec(),
          convert_dtype(dtype),
          api::StorageType::BUFFER,
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      }),
      api::ExecutionLayout::BUFFER_DIRECT,
      persistent);
}

std::vector<int64_t> calc_program_contiguous_strides(IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (int64_t idx = static_cast<int64_t>(sizes.size()) - 2; idx >= 0; --idx) {
    strides[idx] = strides[idx + 1] * std::max<int64_t>(sizes[idx + 1], 1);
  }
  return strides;
}

std::vector<int64_t> calc_program_width_packed_buffer_sizes(IntArrayRef sizes) {
  std::vector<int64_t> physical_sizes(sizes.begin(), sizes.end());
  if (!physical_sizes.empty()) {
    physical_sizes.back() =
        api::utils::align_up(physical_sizes.back(), INT64_C(4));
  }
  return physical_sizes;
}

size_t program_buffer_descriptor_nbytes(
    IntArrayRef sizes,
    const ScalarType dtype) {
  return static_cast<size_t>(
      api::element_size(convert_dtype(dtype)) *
      api::utils::multiply_integers(
          calc_program_width_packed_buffer_sizes(sizes)));
}

std::vector<int64_t> calc_program_width_packed_buffer_strides(
    IntArrayRef sizes) {
  return calc_program_contiguous_strides(
      calc_program_width_packed_buffer_sizes(sizes));
}

Tensor make_program_scratch_buffer_alias(
    const ScratchArena& arena,
    const VulkanScratchSlice& slice,
    IntArrayRef sizes,
    const ScalarType dtype) {
  const size_t required_bytes = program_buffer_descriptor_nbytes(sizes, dtype);
  TORCH_CHECK(
      required_bytes <= slice.size_bytes,
      "Execution-program scratch alias requested ",
      required_bytes,
      " bytes from a slice sized for ",
      slice.size_bytes,
      " bytes");

  const int64_t element_size =
      static_cast<int64_t>(c10::elementSize(dtype));
  TORCH_CHECK(
      element_size > 0,
      "Execution-program scratch alias requires a concrete element size");
  TORCH_CHECK(
      slice.offset_bytes % static_cast<size_t>(element_size) == 0u &&
          arena.size_bytes() % static_cast<size_t>(element_size) == 0u,
      "Execution-program scratch alias requires byte-aligned offsets for dtype ",
      dtype);

  const int64_t storage_offset =
      static_cast<int64_t>(slice.offset_bytes / static_cast<size_t>(element_size));
  const int64_t buffer_length_override =
      static_cast<int64_t>(arena.size_bytes() / static_cast<size_t>(element_size));
  const api::ExecutionLayout execution_layout =
      slice.offset_bytes == 0u ? api::ExecutionLayout::BUFFER_DIRECT
                               : api::ExecutionLayout::BUFFER_VIEW;
  return ::at::native::vulkan::ops::make_typed_buffer_metadata_view_checked(
      arena.storage(),
      dtype,
      sizes,
      calc_program_contiguous_strides(sizes),
      calc_program_width_packed_buffer_strides(sizes),
      storage_offset,
      buffer_length_override,
      execution_layout,
      "execution_program.scratch");
}

Tensor reserve_program_scratch_tensor(
    ScratchArena& arena,
    IntArrayRef sizes,
    const ScalarType dtype) {
  const size_t required_bytes = program_buffer_descriptor_nbytes(sizes, dtype);
  const VulkanScratchSlice slice = arena.reserve(
      required_bytes,
      std::max<uint32_t>(
          arena.alignment(),
          static_cast<uint32_t>(std::max<int64_t>(
              1, static_cast<int64_t>(c10::elementSize(dtype))))));
  return make_program_scratch_buffer_alias(arena, slice, sizes, dtype);
}

size_t tensor_resident_nbytes(const Tensor& tensor) {
  if (!tensor.defined() || !tensor.is_vulkan()) {
    return 0u;
  }
  return static_cast<size_t>(convert(tensor).gpu_nbytes());
}

size_t optional_scratch_resident_nbytes(
    const std::optional<ScratchArena>& scratch) {
  return scratch.has_value() ? tensor_resident_nbytes(scratch->storage()) : 0u;
}

} // namespace

struct AttentionRuntimeProgram::State final {
  std::optional<ScratchArena> scratch_arena_;

  explicit State(std::optional<ScratchArena> scratch_arena)
      : scratch_arena_(std::move(scratch_arena)) {}
};

struct VisionBackboneProgram::State final {
  int64_t num_heads_{1};
  std::optional<ScratchArena> scratch_arena_;
  Tensor norm1_output_;
  Tensor qkv_output_;
  Tensor merge_output_;
  Tensor proj_output_;
  Tensor norm2_output_;
  Tensor fc1_output_;
  Tensor fc2_output_;
  bool persistent_{true};

  State(
      const ScalarType dtype,
      const int64_t batch_size,
      const int64_t token_count,
      const int64_t embed_dim,
      const int64_t hidden_dim,
      const int64_t num_heads,
      std::optional<ScratchArena> scratch_arena,
      const bool persistent)
      : num_heads_(num_heads),
        scratch_arena_(std::move(scratch_arena)),
        persistent_(persistent) {
    const std::vector<int64_t> hidden_sizes{
        batch_size * token_count,
        embed_dim,
    };
    const std::vector<int64_t> qkv_sizes{
        batch_size * token_count,
        3 * embed_dim,
    };
    const std::vector<int64_t> fc1_sizes{
        batch_size * token_count,
        hidden_dim,
    };

    norm1_output_ = create_program_buffer_tensor(hidden_sizes, dtype, persistent_);
    qkv_output_ = create_program_buffer_tensor(qkv_sizes, dtype, persistent_);
    merge_output_ = create_program_buffer_tensor(hidden_sizes, dtype, persistent_);
    proj_output_ = create_program_buffer_tensor(hidden_sizes, dtype, persistent_);
    norm2_output_ = create_program_buffer_tensor(hidden_sizes, dtype, persistent_);
    fc1_output_ = create_program_buffer_tensor(fc1_sizes, dtype, persistent_);
    fc2_output_ = create_program_buffer_tensor(hidden_sizes, dtype, persistent_);
  }
};

struct VisionDecoderProgram::State final {
  std::optional<ScratchArena> scratch_arena_;
  Tensor skip_relu_output_;
  Tensor skip_conv1_output_;
  Tensor skip_conv2_output_;
  Tensor skip_res_output_;
  Tensor main_input_output_;
  Tensor main_relu_output_;
  Tensor main_conv1_output_;
  Tensor main_conv2_output_;
  Tensor main_res_output_;
  Tensor upsample_output_;
  Tensor out_conv_output_;

  State(
      const std::vector<int64_t>& input_sizes,
      const std::optional<std::vector<int64_t>>& skip_sizes,
      const std::vector<int64_t>& target_sizes,
      const int64_t out_channels,
      std::optional<ScratchArena> scratch_arena,
      const bool persistent,
      const bool allocate_intermediate_outputs)
      : scratch_arena_(std::move(scratch_arena)),
        out_conv_output_(create_program_buffer_tensor(
            {input_sizes.at(0), out_channels, target_sizes.at(0), target_sizes.at(1)},
            kFloat,
            persistent)) {
    const std::vector<int64_t> upsample_sizes{
        input_sizes.at(0),
        input_sizes.at(1),
        target_sizes.at(0),
        target_sizes.at(1),
    };

    if (!allocate_intermediate_outputs) {
      if (scratch_arena_.has_value()) {
        scratch_arena_->reset();
      }
      return;
    }

    if (scratch_arena_.has_value()) {
      scratch_arena_->reset();
      if (skip_sizes.has_value()) {
        skip_relu_output_ =
            reserve_program_scratch_tensor(*scratch_arena_, *skip_sizes, kFloat);
        skip_conv1_output_ =
            reserve_program_scratch_tensor(*scratch_arena_, *skip_sizes, kFloat);
        skip_conv2_output_ =
            reserve_program_scratch_tensor(*scratch_arena_, *skip_sizes, kFloat);
        skip_res_output_ =
            reserve_program_scratch_tensor(*scratch_arena_, *skip_sizes, kFloat);
        main_input_output_ =
            reserve_program_scratch_tensor(*scratch_arena_, input_sizes, kFloat);
      }
      main_relu_output_ =
          reserve_program_scratch_tensor(*scratch_arena_, input_sizes, kFloat);
      main_conv1_output_ =
          reserve_program_scratch_tensor(*scratch_arena_, input_sizes, kFloat);
      main_conv2_output_ =
          reserve_program_scratch_tensor(*scratch_arena_, input_sizes, kFloat);
      main_res_output_ =
          reserve_program_scratch_tensor(*scratch_arena_, input_sizes, kFloat);
      upsample_output_ =
          reserve_program_scratch_tensor(*scratch_arena_, upsample_sizes, kFloat);
      scratch_arena_->reset();
      return;
    }

    if (skip_sizes.has_value()) {
      skip_relu_output_ = create_program_buffer_tensor(*skip_sizes, kFloat, persistent);
      skip_conv1_output_ =
          create_program_buffer_tensor(*skip_sizes, kFloat, persistent);
      skip_conv2_output_ =
          create_program_buffer_tensor(*skip_sizes, kFloat, persistent);
      skip_res_output_ = create_program_buffer_tensor(*skip_sizes, kFloat, persistent);
      main_input_output_ =
          create_program_buffer_tensor(input_sizes, kFloat, persistent);
    }
    main_relu_output_ = create_program_buffer_tensor(input_sizes, kFloat, persistent);
    main_conv1_output_ = create_program_buffer_tensor(input_sizes, kFloat, persistent);
    main_conv2_output_ = create_program_buffer_tensor(input_sizes, kFloat, persistent);
    main_res_output_ = create_program_buffer_tensor(input_sizes, kFloat, persistent);
    upsample_output_ = create_program_buffer_tensor(upsample_sizes, kFloat, persistent);
  }
};

bool AttentionRuntimeProgram::defined() const {
  return static_cast<bool>(state_);
}

std::optional<ScratchArena>& AttentionRuntimeProgram::scratch_arena() {
  static std::optional<ScratchArena> empty;
  return state_ ? state_->scratch_arena_ : empty;
}

const std::optional<ScratchArena>& AttentionRuntimeProgram::scratch_arena()
    const {
  static const std::optional<ScratchArena> empty;
  return state_ ? state_->scratch_arena_ : empty;
}

size_t AttentionRuntimeProgram::resident_nbytes() const {
  if (!state_) {
    return 0u;
  }
  return optional_scratch_resident_nbytes(state_->scratch_arena_);
}

const void* AttentionRuntimeProgram::identity() const {
  return state_.get();
}

bool VisionBackboneProgram::defined() const {
  return static_cast<bool>(state_);
}

int64_t VisionBackboneProgram::num_heads() const {
  return state_ ? state_->num_heads_ : 1;
}

std::optional<ScratchArena>& VisionBackboneProgram::scratch_arena() {
  static std::optional<ScratchArena> empty;
  return state_ ? state_->scratch_arena_ : empty;
}

const std::optional<ScratchArena>& VisionBackboneProgram::scratch_arena()
    const {
  static const std::optional<ScratchArena> empty;
  return state_ ? state_->scratch_arena_ : empty;
}

Tensor& VisionBackboneProgram::norm1_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionBackboneProgram");
  return state_->norm1_output_;
}

Tensor& VisionBackboneProgram::qkv_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionBackboneProgram");
  return state_->qkv_output_;
}

Tensor& VisionBackboneProgram::merge_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionBackboneProgram");
  return state_->merge_output_;
}

Tensor& VisionBackboneProgram::proj_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionBackboneProgram");
  return state_->proj_output_;
}

Tensor& VisionBackboneProgram::norm2_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionBackboneProgram");
  return state_->norm2_output_;
}

Tensor& VisionBackboneProgram::fc1_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionBackboneProgram");
  return state_->fc1_output_;
}

Tensor& VisionBackboneProgram::fc2_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionBackboneProgram");
  return state_->fc2_output_;
}

bool VisionBackboneProgram::persistent() const {
  return state_ && state_->persistent_;
}

size_t VisionBackboneProgram::resident_nbytes() const {
  if (!state_) {
    return 0u;
  }
  return optional_scratch_resident_nbytes(state_->scratch_arena_) +
      tensor_resident_nbytes(state_->norm1_output_) +
      tensor_resident_nbytes(state_->qkv_output_) +
      tensor_resident_nbytes(state_->merge_output_) +
      tensor_resident_nbytes(state_->proj_output_) +
      tensor_resident_nbytes(state_->norm2_output_) +
      tensor_resident_nbytes(state_->fc1_output_) +
      tensor_resident_nbytes(state_->fc2_output_);
}

const void* VisionBackboneProgram::identity() const {
  return state_.get();
}

bool VisionDecoderProgram::defined() const {
  return static_cast<bool>(state_);
}

std::optional<ScratchArena>& VisionDecoderProgram::scratch_arena() {
  static std::optional<ScratchArena> empty;
  return state_ ? state_->scratch_arena_ : empty;
}

const std::optional<ScratchArena>& VisionDecoderProgram::scratch_arena() const {
  static const std::optional<ScratchArena> empty;
  return state_ ? state_->scratch_arena_ : empty;
}

Tensor& VisionDecoderProgram::skip_relu_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionDecoderProgram");
  return state_->skip_relu_output_;
}

Tensor& VisionDecoderProgram::skip_conv1_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionDecoderProgram");
  return state_->skip_conv1_output_;
}

Tensor& VisionDecoderProgram::skip_conv2_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionDecoderProgram");
  return state_->skip_conv2_output_;
}

Tensor& VisionDecoderProgram::skip_res_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionDecoderProgram");
  return state_->skip_res_output_;
}

Tensor& VisionDecoderProgram::main_input_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionDecoderProgram");
  return state_->main_input_output_;
}

Tensor& VisionDecoderProgram::main_relu_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionDecoderProgram");
  return state_->main_relu_output_;
}

Tensor& VisionDecoderProgram::main_conv1_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionDecoderProgram");
  return state_->main_conv1_output_;
}

Tensor& VisionDecoderProgram::main_conv2_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionDecoderProgram");
  return state_->main_conv2_output_;
}

Tensor& VisionDecoderProgram::main_res_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionDecoderProgram");
  return state_->main_res_output_;
}

Tensor& VisionDecoderProgram::upsample_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionDecoderProgram");
  return state_->upsample_output_;
}

Tensor& VisionDecoderProgram::out_conv_output() {
  TORCH_INTERNAL_ASSERT(state_, "Undefined VisionDecoderProgram");
  return state_->out_conv_output_;
}

size_t VisionDecoderProgram::resident_nbytes() const {
  if (!state_) {
    return 0u;
  }
  return optional_scratch_resident_nbytes(state_->scratch_arena_) +
      tensor_resident_nbytes(state_->skip_relu_output_) +
      tensor_resident_nbytes(state_->skip_conv1_output_) +
      tensor_resident_nbytes(state_->skip_conv2_output_) +
      tensor_resident_nbytes(state_->skip_res_output_) +
      tensor_resident_nbytes(state_->main_input_output_) +
      tensor_resident_nbytes(state_->main_relu_output_) +
      tensor_resident_nbytes(state_->main_conv1_output_) +
      tensor_resident_nbytes(state_->main_conv2_output_) +
      tensor_resident_nbytes(state_->main_res_output_) +
      tensor_resident_nbytes(state_->upsample_output_) +
      tensor_resident_nbytes(state_->out_conv_output_);
}

const void* VisionDecoderProgram::identity() const {
  return state_.get();
}

AttentionRuntimeProgram lookup_or_create_labeled_attention_runtime_program(
    const std::string& allocation_label,
    const VulkanAttentionKernelFamily kernel_family,
    const std::optional<VulkanScratchArenaSpec>& scratch_spec,
    const VulkanExecutionProgramPlanningDesc& program_plan) {
  const AttentionRuntimeProgramKey query{
      normalize_program_label(allocation_label, "attention_runtime"),
      kernel_family,
      scratch_spec,
      program_plan.persistent};
  if (const auto cached = attention_runtime_program_cache().lookup(
          query,
          hash_attention_runtime_program_key,
          [](const AttentionRuntimeProgramKey& lhs,
             const AttentionRuntimeProgramKey& rhs) { return lhs == rhs; })) {
    log_execution_program_event(
        VulkanExecutionProgramKind::AttentionRuntime,
        "hit",
        query.allocation_label,
        cached->identity(),
        cached->resident_nbytes());
    return *cached;
  }

  std::optional<ScratchArena> scratch_arena;
  if (scratch_spec.has_value()) {
    scratch_arena = lookup_or_create_labeled_scratch_arena(
        program_object_label(query.allocation_label, "scratch"),
        *scratch_spec);
  }

  AttentionRuntimeProgram created{std::make_shared<AttentionRuntimeProgram::State>(
      std::move(scratch_arena))};
  attention_runtime_program_cache().store(
      query,
      created,
      hash_attention_runtime_program_key,
      [](const AttentionRuntimeProgramKey& lhs,
         const AttentionRuntimeProgramKey& rhs) { return lhs == rhs; },
      [](const AttentionRuntimeProgram& program) {
        return program.resident_nbytes();
      });
  log_execution_program_event(
      VulkanExecutionProgramKind::AttentionRuntime,
      "store",
      query.allocation_label,
      created.identity(),
      created.resident_nbytes());
  return created;
}

VisionBackboneProgram lookup_or_create_labeled_vision_backbone_program(
    const std::string& allocation_label,
    const ScalarType dtype,
    const int64_t batch_size,
    const int64_t token_count,
    const int64_t embed_dim,
    const int64_t hidden_dim,
    const int64_t num_heads,
    const std::optional<VulkanScratchArenaSpec>& scratch_spec,
    const VulkanExecutionProgramPlanningDesc& program_plan) {
  const VisionBackboneProgramKey query{
      normalize_program_label(allocation_label, "vision_backbone"),
      dtype,
      batch_size,
      token_count,
      embed_dim,
      hidden_dim,
      num_heads,
      scratch_spec,
      program_plan.persistent};
  if (const auto cached = vision_backbone_program_cache().lookup(
          query,
          hash_vision_backbone_program_key,
          [](const VisionBackboneProgramKey& lhs,
             const VisionBackboneProgramKey& rhs) { return lhs == rhs; })) {
    log_execution_program_event(
        VulkanExecutionProgramKind::VisionBackbone,
        "hit",
        query.allocation_label,
        cached->identity(),
        cached->resident_nbytes());
    return *cached;
  }

  std::optional<ScratchArena> scratch_arena;
  if (scratch_spec.has_value()) {
    scratch_arena = lookup_or_create_labeled_scratch_arena(
        program_object_label(query.allocation_label, "scratch"),
        *scratch_spec);
  }

  VisionBackboneProgram created{
      std::make_shared<VisionBackboneProgram::State>(
          dtype,
          batch_size,
          token_count,
          embed_dim,
          hidden_dim,
          num_heads,
          std::move(scratch_arena),
          program_plan.persistent)};
  vision_backbone_program_cache().store(
      query,
      created,
      hash_vision_backbone_program_key,
      [](const VisionBackboneProgramKey& lhs,
         const VisionBackboneProgramKey& rhs) { return lhs == rhs; },
      [](const VisionBackboneProgram& program) {
        return program.resident_nbytes();
      });
  log_execution_program_event(
      VulkanExecutionProgramKind::VisionBackbone,
      "store",
      query.allocation_label,
      created.identity(),
      created.resident_nbytes());
  return created;
}

VisionDecoderProgram lookup_or_create_labeled_vision_decoder_program(
    const std::string& allocation_label,
    IntArrayRef input_sizes,
    const std::optional<std::vector<int64_t>>& skip_sizes,
    IntArrayRef target_sizes,
    const int64_t out_channels,
    const std::optional<VulkanScratchArenaSpec>& scratch_spec,
    const VulkanExecutionProgramPlanningDesc& program_plan,
    const bool allocate_intermediate_outputs) {
  const VisionDecoderProgramKey query{
      normalize_program_label(allocation_label, "vision_decoder"),
      input_sizes.vec(),
      skip_sizes,
      target_sizes.vec(),
      out_channels,
      scratch_spec,
      allocate_intermediate_outputs,
      program_plan.persistent};
  if (const auto cached = vision_decoder_program_cache().lookup(
          query,
          hash_vision_decoder_program_key,
          [](const VisionDecoderProgramKey& lhs,
             const VisionDecoderProgramKey& rhs) { return lhs == rhs; })) {
    log_execution_program_event(
        VulkanExecutionProgramKind::VisionDecoder,
        "hit",
        query.allocation_label,
        cached->identity(),
        cached->resident_nbytes());
    return *cached;
  }

  std::optional<ScratchArena> scratch_arena;
  if (scratch_spec.has_value()) {
    scratch_arena = lookup_or_create_labeled_scratch_arena(
        program_object_label(query.allocation_label, "scratch"),
        *scratch_spec);
  }

  VisionDecoderProgram created{
      std::make_shared<VisionDecoderProgram::State>(
          query.input_sizes,
          query.skip_sizes,
          query.target_sizes,
          query.out_channels,
          std::move(scratch_arena),
          program_plan.persistent,
          query.allocate_intermediate_outputs)};
  vision_decoder_program_cache().store(
      query,
      created,
      hash_vision_decoder_program_key,
      [](const VisionDecoderProgramKey& lhs,
         const VisionDecoderProgramKey& rhs) { return lhs == rhs; },
      [](const VisionDecoderProgram& program) {
        return program.resident_nbytes();
      });
  log_execution_program_event(
      VulkanExecutionProgramKind::VisionDecoder,
      "store",
      query.allocation_label,
      created.identity(),
      created.resident_nbytes());
  return created;
}

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
