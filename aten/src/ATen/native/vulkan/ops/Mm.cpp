#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/ops/LayoutTransitions.h>
#include <ATen/native/vulkan/ops/Softmax.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/TensorState.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/planning/DevicePolicy.h>
#include <ATen/native/vulkan/planning/DynamicProgramRuntime.h>
#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/ExecutionObjects.h>
#include <ATen/native/vulkan/planning/ReplayTensorState.h>

#include <ATen/Context.h>
#include <ATen/native/vulkan/api/Tensor.h>
#include <ATen/native/vulkan/api/Types.h>
#include <ATen/native/vulkan/impl/Packing.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/zeros.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/irange.h>
#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string_view>
#include <unordered_map>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;
using namespace at::native::vulkan::ops;

constexpr float kGeluBeta =
    static_cast<float>(M_SQRT2 * M_2_SQRTPI * 0.5);

enum class LinearPostOp : uint8_t {
  None,
  Gelu,
};

enum class VulkanLinearFastPath : uint8_t {
  Unknown = 0,
  FloatBuffer,
  BFloat16Buffer,
  BFloat16CooperativeMatrix,
  BFloat16CooperativeMatrixTailM,
  CpuFallback,
};

enum class VulkanLinearRejectReason : uint8_t {
  None = 0,
  InputNotVulkan,
  WeightNotVulkanOrPacked,
  UnsupportedDType,
  UnsupportedStorageType,
  UnsupportedLayout,
  KNotAligned,
  NNotAligned,
  MTailUnsupported,
  PostOpUnsupported,
  CapabilityMissing,
  ShapeUnsupported,
  Unknown,
};

struct VulkanLinearPlanDecision final {
  VulkanLinearFastPath selected = VulkanLinearFastPath::Unknown;
  VulkanLinearRejectReason reject = VulkanLinearRejectReason::None;
  int64_t m = 0;
  int64_t k = 0;
  int64_t n = 0;
  int64_t tile_m = 0;
  int64_t tile_k = 0;
  int64_t tile_n = 0;
  ScalarType input_dtype = ScalarType::Undefined;
  ScalarType weight_dtype = ScalarType::Undefined;
  ScalarType bias_dtype = ScalarType::Undefined;
  ScalarType output_dtype = ScalarType::Undefined;
  bool input_vulkan = false;
  bool weight_packed = false;
  bool weight_vulkan = false;
  bool bias_present = false;
  bool bias_vulkan = false;
  bool input_direct_buffer = false;
  bool output_direct_buffer = false;
  bool can_use_coop_matrix = false;
  bool has_post_op = false;
  bool m_tail = false;
  bool k_tail = false;
  bool n_tail = false;
  bool rejected_because_float_input = false;
  bool rejected_because_float_weight = false;
  bool rejected_because_bias_dtype = false;
  bool rejected_because_layout = false;
  bool rejected_because_not_packed = false;
};

struct VulkanLinearPlanCounters final {
  std::atomic<uint64_t> total{0};
  std::atomic<uint64_t> coop_hit{0};
  std::atomic<uint64_t> coop_tail_m_hit{0};
  std::atomic<uint64_t> reject_m_tail{0};
  std::atomic<uint64_t> reject_k_tail{0};
  std::atomic<uint64_t> reject_n_tail{0};
  std::atomic<uint64_t> reject_layout{0};
  std::atomic<uint64_t> reject_dtype{0};
  std::atomic<uint64_t> reject_capability{0};
  std::atomic<uint64_t> fallback_plain_bf16{0};
  std::atomic<uint64_t> fallback_float{0};
};

struct VulkanLinearPlanContractMatch final {
  bool matched = false;
  const char* contract_name = "none";
  const char* contract_family = "none";
  const char* contract_tuple_id = "none";
  bool prefer_vec2_tiled = false;
};

struct VulkanLinearAggregateValue final {
  uint64_t count = 0u;
  uint64_t input_bytes = 0u;
  uint64_t weight_bytes = 0u;
  uint64_t output_bytes = 0u;
};

struct VulkanLinearPackResidencyValue final {
  uint64_t count = 0u;
  uint64_t created = 0u;
  uint64_t reused = 0u;
  uint64_t packed_bytes = 0u;
  uint64_t raw_weight_bytes = 0u;
  uint64_t raw_bias_bytes = 0u;
  uint64_t raw_weight_vulkan = 0u;
  uint64_t retain_unpacked = 0u;
};

VulkanLinearPlanCounters& linear_plan_counters() {
  static VulkanLinearPlanCounters counters;
  return counters;
}

std::mutex& linear_aggregate_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::string, VulkanLinearAggregateValue>&
linear_aggregate() {
  static std::unordered_map<std::string, VulkanLinearAggregateValue> aggregate;
  return aggregate;
}

std::mutex& linear_pack_residency_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::string, VulkanLinearPackResidencyValue>&
linear_pack_residency_aggregate() {
  static auto* aggregate =
      new std::unordered_map<std::string, VulkanLinearPackResidencyValue>();
  return *aggregate;
}

std::string format_linear_shape(IntArrayRef shape) {
  std::ostringstream stream;
  stream << "[";
  for (const auto i : c10::irange(shape.size())) {
    if (i > 0) {
      stream << ",";
    }
    stream << shape[i];
  }
  stream << "]";
  return stream.str();
}

uint64_t tensor_nbytes_for_diagnostics(const Tensor& tensor) {
  if (!tensor.defined()) {
    return 0u;
  }
  return static_cast<uint64_t>(tensor.numel()) *
      static_cast<uint64_t>(tensor.element_size());
}

void note_linear_pack_residency(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const PackedWeightHandle& handle,
    const bool reused,
    const bool retain_unpacked,
    const bool use_batch,
    const bool use_buffer_packed_weights) {
  std::ostringstream key;
  key << "weight_shape=" << format_linear_shape(weight.sizes())
      << " dtype=" << weight.scalar_type()
      << " use_batch=" << (use_batch ? 1 : 0)
      << " buffer_packed=" << (use_buffer_packed_weights ? 1 : 0)
      << " has_bias=" << (bias && bias->defined() ? 1 : 0);
  std::lock_guard<std::mutex> lock(linear_pack_residency_mutex());
  auto& value = linear_pack_residency_aggregate()[key.str()];
  value.count += 1u;
  if (reused) {
    value.reused += 1u;
  } else {
    value.created += 1u;
  }
  value.packed_bytes += static_cast<uint64_t>(handle.resident_nbytes());
  value.raw_weight_bytes += tensor_nbytes_for_diagnostics(weight);
  value.raw_bias_bytes +=
      bias && bias->defined() ? tensor_nbytes_for_diagnostics(*bias) : 0u;
  value.raw_weight_vulkan += weight.is_vulkan() ? 1u : 0u;
  value.retain_unpacked += retain_unpacked ? 1u : 0u;
}

static inline bool is_aligned_i64(const int64_t value, const int64_t alignment) {
  return alignment <= 1 || value % alignment == 0;
}

const char* linear_reject_reason_name(const VulkanLinearRejectReason reason) {
  switch (reason) {
    case VulkanLinearRejectReason::None:
      return "none";
    case VulkanLinearRejectReason::InputNotVulkan:
      return "input_not_vulkan";
    case VulkanLinearRejectReason::WeightNotVulkanOrPacked:
      return "weight_not_vulkan_or_packed";
    case VulkanLinearRejectReason::UnsupportedDType:
      return "unsupported_dtype";
    case VulkanLinearRejectReason::UnsupportedStorageType:
      return "unsupported_storage_type";
    case VulkanLinearRejectReason::UnsupportedLayout:
      return "unsupported_layout";
    case VulkanLinearRejectReason::KNotAligned:
      return "k_not_aligned";
    case VulkanLinearRejectReason::NNotAligned:
      return "n_not_aligned";
    case VulkanLinearRejectReason::MTailUnsupported:
      return "m_tail_unsupported";
    case VulkanLinearRejectReason::PostOpUnsupported:
      return "post_op_unsupported";
    case VulkanLinearRejectReason::CapabilityMissing:
      return "capability_missing";
    case VulkanLinearRejectReason::ShapeUnsupported:
      return "shape_unsupported";
    default:
      return "unknown";
  }
}

const std::string& vulkan_linear_plan_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_LINEAR_PLAN_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

void append_vulkan_linear_plan_log(
    const VulkanLinearPlanDecision& decision,
    const char* label) {
  const std::string& path = vulkan_linear_plan_log_path();
  if (path.empty()) {
    return;
  }
  std::ofstream out(path, std::ios::app);
  out << "linear_plan"
      << " label=" << (label ? label : "unknown")
      << " selected=" << static_cast<int>(decision.selected)
      << " reject=" << linear_reject_reason_name(decision.reject)
      << " m=" << decision.m
      << " k=" << decision.k
      << " n=" << decision.n
      << " tile_m=" << decision.tile_m
      << " tile_k=" << decision.tile_k
      << " tile_n=" << decision.tile_n
      << " input_dtype=" << static_cast<int>(decision.input_dtype)
      << " weight_dtype=" << static_cast<int>(decision.weight_dtype)
      << " bias_dtype=" << static_cast<int>(decision.bias_dtype)
      << " output_dtype=" << static_cast<int>(decision.output_dtype)
      << " m_tail=" << (decision.m_tail ? 1 : 0)
      << " k_tail=" << (decision.k_tail ? 1 : 0)
      << " n_tail=" << (decision.n_tail ? 1 : 0)
      << " input_vulkan=" << (decision.input_vulkan ? 1 : 0)
      << " weight_packed=" << (decision.weight_packed ? 1 : 0)
      << " weight_vulkan=" << (decision.weight_vulkan ? 1 : 0)
      << " bias_present=" << (decision.bias_present ? 1 : 0)
      << " bias_vulkan=" << (decision.bias_vulkan ? 1 : 0)
      << " input_direct_buffer=" << (decision.input_direct_buffer ? 1 : 0)
      << " output_direct_buffer=" << (decision.output_direct_buffer ? 1 : 0)
      << " can_use_coop_matrix=" << (decision.can_use_coop_matrix ? 1 : 0)
      << " post_op=" << (decision.has_post_op ? 1 : 0)
      << " rejected_float_input="
      << (decision.rejected_because_float_input ? 1 : 0)
      << " rejected_float_weight="
      << (decision.rejected_because_float_weight ? 1 : 0)
      << " rejected_bias_dtype="
      << (decision.rejected_because_bias_dtype ? 1 : 0)
      << " rejected_layout="
      << (decision.rejected_because_layout ? 1 : 0)
      << " rejected_not_packed="
      << (decision.rejected_because_not_packed ? 1 : 0)
      << '\n';
}

void note_linear_plan_decision(const VulkanLinearPlanDecision& decision) {
  VulkanLinearPlanCounters& counters = linear_plan_counters();
  counters.total.fetch_add(1, std::memory_order_relaxed);
  switch (decision.selected) {
    case VulkanLinearFastPath::BFloat16CooperativeMatrix:
      counters.coop_hit.fetch_add(1, std::memory_order_relaxed);
      break;
    case VulkanLinearFastPath::BFloat16CooperativeMatrixTailM:
      counters.coop_tail_m_hit.fetch_add(1, std::memory_order_relaxed);
      break;
    case VulkanLinearFastPath::BFloat16Buffer:
      counters.fallback_plain_bf16.fetch_add(1, std::memory_order_relaxed);
      break;
    case VulkanLinearFastPath::FloatBuffer:
      counters.fallback_float.fetch_add(1, std::memory_order_relaxed);
      break;
    default:
      break;
  }
  switch (decision.reject) {
    case VulkanLinearRejectReason::MTailUnsupported:
      counters.reject_m_tail.fetch_add(1, std::memory_order_relaxed);
      break;
    case VulkanLinearRejectReason::KNotAligned:
      counters.reject_k_tail.fetch_add(1, std::memory_order_relaxed);
      break;
    case VulkanLinearRejectReason::NNotAligned:
      counters.reject_n_tail.fetch_add(1, std::memory_order_relaxed);
      break;
    case VulkanLinearRejectReason::UnsupportedLayout:
    case VulkanLinearRejectReason::UnsupportedStorageType:
      counters.reject_layout.fetch_add(1, std::memory_order_relaxed);
      break;
    case VulkanLinearRejectReason::UnsupportedDType:
      counters.reject_dtype.fetch_add(1, std::memory_order_relaxed);
      break;
    case VulkanLinearRejectReason::CapabilityMissing:
      counters.reject_capability.fetch_add(1, std::memory_order_relaxed);
      break;
    default:
      break;
  }
}

std::string format_linear_sizes(IntArrayRef sizes) {
  std::ostringstream stream;
  stream << '[';
  for (const auto i : c10::irange(sizes.size())) {
    if (i > 0) {
      stream << 'x';
    }
    stream << sizes[i];
  }
  stream << ']';
  return stream.str();
}

void append_linear_tensor_summary(
    std::ostringstream& stream,
    const char* name,
    const Tensor& tensor) {
  stream << ' ' << name << "=" << format_linear_sizes(tensor.sizes())
         << ' ' << name << "_vulkan=" << (tensor.is_vulkan() ? 1 : 0);
  if (!tensor.is_vulkan()) {
    return;
  }
  const vTensor& v_tensor = convert(tensor);
  stream << ' ' << name << "_storage="
         << static_cast<int>(v_tensor.storage_type())
         << ' ' << name << "_layout="
         << static_cast<int>(v_tensor.gpu_memory_layout())
         << ' ' << name << "_exec="
         << static_cast<int>(v_tensor.execution_layout())
         << ' ' << name << "_direct="
         << (v_tensor.has_direct_buffer_layout() ? 1 : 0)
         << ' ' << name << "_offset=" << v_tensor.storage_offset()
         << ' ' << name << "_bytes=" << tensor.nbytes();
}

void log_linear_context_checkpoint(
    const char* checkpoint,
    const Tensor& tensor,
    const LinearPostOp post_op,
    const bool quantized) {
  std::ostringstream stream;
  stream << "aten::linear." << checkpoint
         << " post=" << (post_op == LinearPostOp::Gelu ? "gelu" : "none")
         << " quantized=" << (quantized ? 1 : 0);
  append_linear_tensor_summary(stream, "input", tensor);
  utils::log_vulkan_op_hit(stream.str());
}

size_t linear_runtime_scratch_bytes(const Tensor& input) {
  return std::max<size_t>(
      128u * 1024u,
      static_cast<size_t>(std::max<int64_t>(1, input.numel())) *
          sizeof(float) * 4u);
}

Tensor upcast_half_linear_tensor_for_packing(const Tensor& tensor) {
  if (tensor.scalar_type() != kHalf && tensor.scalar_type() != kBFloat16) {
    return tensor;
  }

  if (!tensor.is_vulkan()) {
    return tensor.to(kFloat);
  }

  if (tensor.scalar_type() == kBFloat16) {
    constexpr int64_t kMaxSmallBFloat16LinearCpuWidenNumel = 65536;
    TORCH_CHECK(
        tensor.numel() <= kMaxSmallBFloat16LinearCpuWidenNumel,
        "Vulkan BF16 linear widening requires native BF16 buffer cast for large "
        "tensors, but that route is currently disabled because it is not "
        "correct for all buffer layouts");
    report_vulkan_cpu_fallback(
        "aten::linear", "bfloat16_widen_cpu_small", {tensor});
    utils::log_vulkan_op_hit("aten::linear.bfloat16_widen_cpu_small");
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    return tensor.cpu().to(kFloat).vulkan();
  }

  // Prefer the backend cast path for half tensors so they stay Vulkan-resident
  // when the source layout supports it.
  return utils::cast_vulkan_tensor_dtype(tensor, kFloat);
}

std::optional<Tensor> upcast_half_linear_tensor_for_packing(
    const std::optional<Tensor>& tensor) {
  if (!tensor || !tensor->defined()) {
    return tensor;
  }
  return upcast_half_linear_tensor_for_packing(*tensor);
}

const char* linear_role_from_label(
    const std::string& label,
    const LinearPostOp post_op) {
  if (label.find(".qkv") != std::string::npos) {
    return "qkv";
  }
  if (label.find(".proj") != std::string::npos) {
    return "proj";
  }
  if (label.find(".fc1") != std::string::npos) {
    return post_op == LinearPostOp::Gelu ? "fc1_gelu" : "fc1";
  }
  if (label.find(".fc2") != std::string::npos) {
    return "fc2";
  }
  if (label.find("patch") != std::string::npos) {
    return "patch_embed";
  }
  return label.empty() || label == "unlabeled" ? "unknown" : "other";
}

VulkanLinearPlanContractMatch match_vision_exact_tiled_linear_plan(
    const utils::VulkanRuntimePolicy& runtime_policy,
    const char* role,
    const ScalarType input_dtype,
    const int64_t m,
    const int64_t k,
    const int64_t n,
    const bool bias_defined,
    const LinearPostOp post_op) {
  (void)runtime_policy;
  (void)role;
  (void)input_dtype;
  (void)m;
  (void)k;
  (void)n;
  (void)bias_defined;
  (void)post_op;

  return {};
}

const char* linear_kernel_kind_from_name(const char* kernel_name) {
  if (kernel_name == nullptr) {
    return "unknown";
  }
  const std::string name(kernel_name);
  if (name.find("bmm_buffer_float") != std::string::npos) {
    return "bmm_buffer_float";
  }
  if (name.find("bfloat16_cooperative_matrix_tail_m") != std::string::npos) {
    return "bfloat16_coop_tail_m";
  }
  if (name.find("bfloat16_cooperative_matrix") != std::string::npos) {
    return "bfloat16_coop";
  }
  if (name.find("gelu") != std::string::npos) {
    return "mm_buffer_float_gelu";
  }
  if (name.find("bias") != std::string::npos) {
    return "mm_buffer_float_bias";
  }
  if (name.find("mm_buffer_float") != std::string::npos ||
      name.find("buffer_float") != std::string::npos) {
    return "mm_buffer_float";
  }
  if (name.find("raw_direct_weight") != std::string::npos) {
    return "mm_buffer_float";
  }
  return "other";
}

Tensor cpu_transposed_weight_for_packing(
    const Tensor& weight,
    const char* reason) {
  report_vulkan_cpu_fallback(
      "aten::linear",
      reason,
      {weight},
      VulkanCpuFallbackKind::SyncReadback);
  return weight.cpu().t().contiguous();
}

bool can_make_vulkan_linear_weight_transpose_view(
    const Tensor& weight,
    c10::DimVector& output_sizes,
    c10::DimVector& output_logical_strides,
    c10::DimVector& output_physical_strides) {
  if (!weight.is_vulkan() || weight.dim() != 2) {
    return false;
  }
  const vTensor& v_weight = convert(weight);
  if (
      v_weight.storage_type() != api::StorageType::BUFFER ||
      !v_weight.has_direct_buffer_layout() ||
      !utils::supports_buffer_metadata_view_fast_path(v_weight)) {
    return false;
  }

  output_sizes.assign(v_weight.sizes().begin(), v_weight.sizes().end());
  output_logical_strides = logical_strides(v_weight);
  output_physical_strides.assign(
      v_weight.gpu_strides().begin(), v_weight.gpu_strides().end());
  std::swap(output_sizes[0], output_sizes[1]);
  std::swap(output_logical_strides[0], output_logical_strides[1]);
  std::swap(output_physical_strides[0], output_physical_strides[1]);

  return utils::can_make_buffer_metadata_view(
      v_weight,
      output_sizes,
      output_logical_strides,
      output_physical_strides,
      v_weight.storage_offset());
}

Tensor vulkan_linear_weight_transpose_view_for_packing(
    const Tensor& weight,
    const c10::DimVector& output_sizes,
    const c10::DimVector& output_logical_strides,
    const c10::DimVector& output_physical_strides) {
  const vTensor& v_weight = convert(weight);
  utils::log_vulkan_op_hit("aten::linear.weight_transpose_metadata_view");
  return make_buffer_metadata_view_checked(
      weight,
      output_sizes,
      output_logical_strides,
      output_physical_strides,
      v_weight.storage_offset(),
      "aten::linear.weight_transpose_metadata_view");
}

Tensor transposed_linear_weight_for_packing(
    const Tensor& weight,
    const char* cpu_reason) {
  c10::DimVector output_sizes;
  c10::DimVector output_logical_strides;
  c10::DimVector output_physical_strides;
  if (can_make_vulkan_linear_weight_transpose_view(
          weight,
          output_sizes,
          output_logical_strides,
          output_physical_strides)) {
    return vulkan_linear_weight_transpose_view_for_packing(
        weight,
        output_sizes,
        output_logical_strides,
        output_physical_strides);
  }
  return cpu_transposed_weight_for_packing(weight, cpu_reason);
}

Tensor upload_linear_tensor_to_buffer(
    const Tensor& tensor,
    const api::GPUMemoryLayout memory_layout) {
  const Tensor source = tensor.requires_grad() ? tensor.detach() : tensor;

  if (source.is_vulkan()) {
    return utils::mark_tensor_execution(
        utils::ensure_buffer_storage(source, memory_layout),
        api::ExecutionLayout::BUFFER_DIRECT,
        true);
  }

  TORCH_CHECK(
      source.device().is_cpu(),
      "Vulkan linear buffer prepack expects CPU or Vulkan tensors");
  vTensor v_buffer{
      api::context(),
      source.sizes().vec(),
      convert_dtype(source.scalar_type()),
      api::StorageType::BUFFER,
      memory_layout,
  };
  pack_cpu_to_vulkan(source, v_buffer);
  Tensor result = utils::mark_tensor_execution(
      convert(v_buffer), api::ExecutionLayout::BUFFER_DIRECT, true);
  record_tensor_write(
      result,
      "aten::linear.buffer_upload",
      "cpu_to_vulkan_buffer_direct",
      {source});
  return result;
}

bool is_float_or_half_tensor(const Tensor& tensor) {
  return tensor.scalar_type() == kFloat || tensor.scalar_type() == kHalf ||
      tensor.scalar_type() == kBFloat16;
}

bool can_run_half_buffer_linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  if (
      input.scalar_type() != kHalf ||
      weight.scalar_type() != kHalf ||
      input.dim() < 1 ||
      weight.dim() != 2 ||
      input.requires_grad() ||
      weight.requires_grad() ||
      input.size(-1) != weight.size(1)) {
    return false;
  }

  if (bias && bias->defined()) {
    if (
        bias->requires_grad() ||
        (bias->scalar_type() != kHalf && bias->scalar_type() != kFloat)) {
      return false;
    }
  }

  return true;
}

c10::intrusive_ptr<LinearPackedContext> get_or_create_linear_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  if (utils::has_inference_tensor(weight, bias)) {
    const Tensor prepared_weight = (weight.is_vulkan() && weight.dim() == 2)
        ? transposed_linear_weight_for_packing(
              weight, "inference_tensor_weight_cpu_transpose")
        : weight.t();
    return c10::make_intrusive<LinearPackedContext>(
        LinearPackedContext(
            prepared_weight,
            bias,
            false,
            std::string(),
            false,
            true));
  }

  const Tensor prepared_weight =
      (c10::InferenceMode::is_enabled() && weight.is_vulkan() &&
       weight.dim() == 2)
      ? transposed_linear_weight_for_packing(
            weight, "inference_mode_weight_cpu_transpose")
      : weight.t();
  return c10::make_intrusive<LinearPackedContext>(
      LinearPackedContext(
          prepared_weight,
          bias,
          false,
          std::string(),
          false,
          true));
}

inline bool has_bias(const std::optional<Tensor>& bias) {
  return bias && bias->defined();
}

struct LinearPackedRunState final {
  const PackedWeightHandle& packed_weight;
  const vTensor& packed_v_weight;
  const vTensor& packed_v_bias;
  const std::vector<int64_t>& logical_weight_sizes;
  bool bias_defined;
};

void note_linear_aggregate(
    const char* kernel_name,
    const Tensor& input_arg_2d,
    const utils::VulkanRuntimePolicy& runtime_policy,
    const Tensor& packed_weight_tensor,
    const std::optional<Tensor>& packed_bias_tensor,
    const Tensor& output_tensor,
    const vTensor& v_input,
    const vTensor& v_output,
    const LinearPackedRunState& packed_state,
    IntArrayRef output_sizes,
    const api::utils::uvec3& global_size,
    const api::utils::uvec3& local_size,
    const LinearPostOp post_op) {
  const std::string& label = api::current_allocation_label();
  const char* role = linear_role_from_label(label, post_op);
  const char* kernel = linear_kernel_kind_from_name(kernel_name);
  const bool bias =
      packed_bias_tensor.has_value() && packed_bias_tensor->defined();
  const int64_t m = input_arg_2d.size(Layout::Parameter::height);
  const int64_t k = input_arg_2d.size(Layout::Parameter::width);
  const int64_t n = output_sizes[Layout::Parameter::width];
  const VulkanLinearPlanContractMatch contract =
      match_vision_exact_tiled_linear_plan(
          runtime_policy,
          role,
          input_arg_2d.scalar_type(),
          m,
          k,
          n,
          bias,
          post_op);

  std::ostringstream key;
  key << "linear_aggregate"
      << " op_family=linear"
      << " selected=FloatBufferLinear"
      << " reject=None"
      << " role=" << role
      << " kernel=" << kernel
      << " submit_kernel=" << (kernel_name ? kernel_name : "unknown")
      << " label=" << (label.empty() ? "unlabeled" : label)
      << " contract=" << contract.contract_name
      << " contract_family=" << contract.contract_family
      << " contract_tuple=" << contract.contract_tuple_id
      << " m=" << m
      << " k=" << k
      << " n=" << n
      << " input=[" << m << ',' << k << ']'
      << " weight=[" << k << ',' << n << ']'
      << " output=[" << m << ',' << n << ']'
      << " input_dtype=" << static_cast<int>(input_arg_2d.scalar_type())
      << " weight_dtype=" << static_cast<int>(packed_weight_tensor.scalar_type())
      << " bias_dtype="
      << static_cast<int>(
             bias ? packed_bias_tensor->scalar_type()
                  : ScalarType::Undefined)
      << " output_dtype=" << static_cast<int>(output_tensor.scalar_type())
      << " post_op=" << (post_op == LinearPostOp::Gelu ? 1 : 0)
      << " bias=" << (bias ? 1 : 0)
      << " input_storage=" << static_cast<int>(v_input.storage_type())
      << " weight_storage="
      << static_cast<int>(packed_state.packed_v_weight.storage_type())
      << " output_storage=" << static_cast<int>(v_output.storage_type())
      << " input_layout=" << static_cast<int>(v_input.gpu_memory_layout())
      << " weight_layout="
      << static_cast<int>(packed_state.packed_v_weight.gpu_memory_layout())
      << " output_layout=" << static_cast<int>(v_output.gpu_memory_layout())
      << " input_execution_layout="
      << static_cast<int>(v_input.execution_layout())
      << " weight_execution_layout="
      << static_cast<int>(packed_state.packed_v_weight.execution_layout())
      << " output_execution_layout="
      << static_cast<int>(v_output.execution_layout())
      << " input_direct=" << (v_input.has_direct_buffer_layout() ? 1 : 0)
      << " weight_direct="
      << (packed_state.packed_v_weight.has_direct_buffer_layout() ? 1 : 0)
      << " output_direct=" << (v_output.has_direct_buffer_layout() ? 1 : 0)
      << " weight_packed=1"
      << " input_offset=" << v_input.storage_offset()
      << " weight_offset=" << packed_state.packed_v_weight.storage_offset()
      << " output_offset=" << v_output.storage_offset()
      << " global=[" << global_size.data[0u] << ',' << global_size.data[1u]
      << ',' << global_size.data[2u] << ']'
      << " local=[" << local_size.data[0u] << ',' << local_size.data[1u]
      << ',' << local_size.data[2u] << ']';

  std::lock_guard<std::mutex> guard(linear_aggregate_mutex());
  VulkanLinearAggregateValue& value = linear_aggregate()[key.str()];
  value.count += 1u;
  value.input_bytes += static_cast<uint64_t>(input_arg_2d.nbytes());
  value.weight_bytes += static_cast<uint64_t>(packed_weight_tensor.nbytes());
  value.output_bytes += static_cast<uint64_t>(output_tensor.nbytes());
}

void note_raw_direct_linear_aggregate(
    const char* kernel_name,
    const Tensor& input_arg_2d,
    const utils::VulkanRuntimePolicy& runtime_policy,
    const Tensor& weight_tensor,
    const Tensor& output_tensor,
    const vTensor& v_input,
    const vTensor& v_weight,
    const vTensor& v_output,
    IntArrayRef output_sizes,
    const api::utils::uvec3& global_size,
    const api::utils::uvec3& local_size) {
  const std::string& label = api::current_allocation_label();
  const char* role = linear_role_from_label(label, LinearPostOp::None);
  const char* kernel = linear_kernel_kind_from_name(kernel_name);
  const int64_t m = input_arg_2d.size(Layout::Parameter::height);
  const int64_t k = input_arg_2d.size(Layout::Parameter::width);
  const int64_t n = output_sizes[Layout::Parameter::width];

  std::ostringstream key;
  key << "linear_aggregate"
      << " op_family=linear"
      << " selected=RawDirectWeightLinear"
      << " reject=None"
      << " role=" << role
      << " kernel=" << kernel
      << " submit_kernel=" << (kernel_name ? kernel_name : "unknown")
      << " label=" << (label.empty() ? "unlabeled" : label)
      << " contract=RawDirectWeightLinearPlan"
      << " contract_family=NoCacheFloatBuffer"
      << " contract_tuple=no_bias_inference_float_buffer"
      << " m=" << m
      << " k=" << k
      << " n=" << n
      << " input=[" << m << ',' << k << ']'
      << " weight=[" << k << ',' << n << ']'
      << " output=[" << m << ',' << n << ']'
      << " input_dtype=" << static_cast<int>(input_arg_2d.scalar_type())
      << " weight_dtype=" << static_cast<int>(weight_tensor.scalar_type())
      << " bias_dtype=" << static_cast<int>(ScalarType::Undefined)
      << " output_dtype=" << static_cast<int>(output_tensor.scalar_type())
      << " post_op=0"
      << " bias=0"
      << " input_storage=" << static_cast<int>(v_input.storage_type())
      << " weight_storage=" << static_cast<int>(v_weight.storage_type())
      << " output_storage=" << static_cast<int>(v_output.storage_type())
      << " input_layout=" << static_cast<int>(v_input.gpu_memory_layout())
      << " weight_layout=" << static_cast<int>(v_weight.gpu_memory_layout())
      << " output_layout=" << static_cast<int>(v_output.gpu_memory_layout())
      << " input_execution_layout="
      << static_cast<int>(v_input.execution_layout())
      << " weight_execution_layout="
      << static_cast<int>(v_weight.execution_layout())
      << " output_execution_layout="
      << static_cast<int>(v_output.execution_layout())
      << " input_direct=" << (v_input.has_direct_buffer_layout() ? 1 : 0)
      << " weight_direct=" << (v_weight.has_direct_buffer_layout() ? 1 : 0)
      << " output_direct=" << (v_output.has_direct_buffer_layout() ? 1 : 0)
      << " weight_packed=0"
      << " raw_direct=1"
      << " input_offset=" << v_input.storage_offset()
      << " weight_offset=" << v_weight.storage_offset()
      << " output_offset=" << v_output.storage_offset()
      << " global=[" << global_size.data[0u] << ',' << global_size.data[1u]
      << ',' << global_size.data[2u] << ']'
      << " local=[" << local_size.data[0u] << ',' << local_size.data[1u]
      << ',' << local_size.data[2u] << ']'
      << " domain="
      << static_cast<int>(runtime_policy.request.model_domain)
      << " phase="
      << static_cast<int>(runtime_policy.request.execution_phase);

  std::lock_guard<std::mutex> guard(linear_aggregate_mutex());
  VulkanLinearAggregateValue& value = linear_aggregate()[key.str()];
  value.count += 1u;
  value.input_bytes += static_cast<uint64_t>(input_arg_2d.nbytes());
  value.weight_bytes += static_cast<uint64_t>(weight_tensor.nbytes());
  value.output_bytes += static_cast<uint64_t>(output_tensor.nbytes());
}

void note_bmm_aggregate(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& output_tensor,
    const vTensor& v_mat1,
    const vTensor& v_mat2,
    const vTensor& v_output,
    const std::optional<Tensor>& bias,
    const api::utils::uvec3& global_size,
    const api::utils::uvec3& local_size) {
  const std::string& label = api::current_allocation_label();
  const bool bias_defined = bias && bias->defined();
  const int64_t batch = mat1.size(Layout::BatchMatrices::batch);
  const int64_t m = mat1.size(Layout::BatchMatrices::height);
  const int64_t k = mat1.size(Layout::BatchMatrices::width);
  const int64_t n = mat2.size(Layout::BatchMatrices::width);

  std::ostringstream key;
  key << "linear_aggregate"
      << " op_family=bmm"
      << " selected=FloatBufferBmm"
      << " reject=None"
      << " role=bmm"
      << " kernel=bmm_buffer_float"
      << " submit_kernel=aten::bmm.buffer_float"
      << " label=" << (label.empty() ? "unlabeled" : label)
      << " contract=none"
      << " contract_family=none"
      << " contract_tuple=none"
      << " batch=" << batch
      << " m=" << m
      << " k=" << k
      << " n=" << n
      << " input=[" << batch << ',' << m << ',' << k << ']'
      << " weight=[" << batch << ',' << k << ',' << n << ']'
      << " output=[" << batch << ',' << m << ',' << n << ']'
      << " input_dtype=" << static_cast<int>(mat1.scalar_type())
      << " weight_dtype=" << static_cast<int>(mat2.scalar_type())
      << " bias_dtype="
      << static_cast<int>(
             bias_defined ? bias->scalar_type() : ScalarType::Undefined)
      << " output_dtype=" << static_cast<int>(output_tensor.scalar_type())
      << " post_op=0"
      << " bias=" << (bias_defined ? 1 : 0)
      << " input_storage=" << static_cast<int>(v_mat1.storage_type())
      << " weight_storage=" << static_cast<int>(v_mat2.storage_type())
      << " output_storage=" << static_cast<int>(v_output.storage_type())
      << " input_layout=" << static_cast<int>(v_mat1.gpu_memory_layout())
      << " weight_layout=" << static_cast<int>(v_mat2.gpu_memory_layout())
      << " output_layout=" << static_cast<int>(v_output.gpu_memory_layout())
      << " input_execution_layout="
      << static_cast<int>(v_mat1.execution_layout())
      << " weight_execution_layout="
      << static_cast<int>(v_mat2.execution_layout())
      << " output_execution_layout="
      << static_cast<int>(v_output.execution_layout())
      << " input_direct=" << (v_mat1.has_direct_buffer_layout() ? 1 : 0)
      << " weight_direct=" << (v_mat2.has_direct_buffer_layout() ? 1 : 0)
      << " output_direct=" << (v_output.has_direct_buffer_layout() ? 1 : 0)
      << " weight_packed=0"
      << " input_offset=" << v_mat1.storage_offset()
      << " weight_offset=" << v_mat2.storage_offset()
      << " output_offset=" << v_output.storage_offset()
      << " global=[" << global_size.data[0u] << ',' << global_size.data[1u]
      << ',' << global_size.data[2u] << ']'
      << " local=[" << local_size.data[0u] << ',' << local_size.data[1u]
      << ',' << local_size.data[2u] << ']';

  std::lock_guard<std::mutex> guard(linear_aggregate_mutex());
  VulkanLinearAggregateValue& value = linear_aggregate()[key.str()];
  value.count += 1u;
  value.input_bytes += static_cast<uint64_t>(mat1.nbytes());
  value.weight_bytes += static_cast<uint64_t>(mat2.nbytes());
  value.output_bytes += static_cast<uint64_t>(output_tensor.nbytes());
}

void log_float_buffer_linear_submit(
    const char* kernel_name,
    const Tensor& input_arg,
    const Tensor& input_arg_2d,
    const vTensor& v_input,
    const vTensor& v_weight,
    const vTensor& v_output,
    const LinearPackedRunState& packed_state,
    IntArrayRef output_sizes,
    const utils::VulkanRuntimePolicy& runtime_policy,
    const bool bias_defined,
    const bool use_specialized_tiled_kernel,
    const bool use_vec2_tiled_kernel,
    const LinearPostOp post_op) {
  std::ostringstream stream;
  stream << "aten::linear.submit"
         << " kernel=" << kernel_name
         << " input=" << format_linear_sizes(input_arg.sizes())
         << " input2d=" << format_linear_sizes(input_arg_2d.sizes())
         << " output2d=" << format_linear_sizes(output_sizes)
         << " weight=" << format_linear_sizes(packed_state.logical_weight_sizes)
         << " bias=" << (bias_defined ? 1 : 0)
         << " post=" << (post_op == LinearPostOp::Gelu ? "gelu" : "none")
         << " tiled=" << (use_specialized_tiled_kernel ? 1 : 0)
         << " vec2=" << (use_vec2_tiled_kernel ? 1 : 0)
         << " input_direct=" << (v_input.has_direct_buffer_layout() ? 1 : 0)
         << " weight_direct=" << (v_weight.has_direct_buffer_layout() ? 1 : 0)
         << " output_direct=" << (v_output.has_direct_buffer_layout() ? 1 : 0)
         << " input_offset=" << v_input.storage_offset()
         << " weight_offset=" << v_weight.storage_offset()
         << " output_offset=" << v_output.storage_offset()
         << " domain="
         << static_cast<int>(runtime_policy.request.model_domain)
         << " phase="
         << static_cast<int>(runtime_policy.request.execution_phase);
  utils::log_vulkan_op_hit(stream.str());
}

constexpr uint64_t kLargeLinearCheckpointMinWeightBytes = 4ull * 1024ull * 1024ull;
constexpr uint64_t kLargeLinearCheckpointSubmitBudget = 48ull;
constexpr uint64_t kLargeLinearCheckpointByteBudget = 1024ull * 1024ull * 1024ull;

std::atomic<uint64_t>& large_linear_checkpoint_submit_count() {
  static std::atomic<uint64_t> count{0u};
  return count;
}

std::atomic<uint64_t>& large_linear_checkpoint_bytes() {
  static std::atomic<uint64_t> bytes{0u};
  return bytes;
}

void maybe_synchronize_after_large_linear_checkpoint(
    const Tensor& input,
    const Tensor& packed_weight,
    const Tensor& output) {
  if (!c10::InferenceMode::is_enabled() || !packed_weight.defined()) {
    return;
  }

  const uint64_t weight_bytes = static_cast<uint64_t>(packed_weight.nbytes());
  if (weight_bytes < kLargeLinearCheckpointMinWeightBytes) {
    return;
  }

  const uint64_t observed_submits =
      large_linear_checkpoint_submit_count().fetch_add(
          1u, std::memory_order_relaxed) +
      1u;
  const uint64_t observed_bytes = large_linear_checkpoint_bytes().fetch_add(
                                      static_cast<uint64_t>(input.nbytes()) +
                                          weight_bytes +
                                          static_cast<uint64_t>(output.nbytes()),
                                      std::memory_order_relaxed) +
      static_cast<uint64_t>(input.nbytes()) + weight_bytes +
      static_cast<uint64_t>(output.nbytes());

  if (
      observed_submits < kLargeLinearCheckpointSubmitBudget &&
      observed_bytes < kLargeLinearCheckpointByteBudget) {
    return;
  }

  const uint64_t checkpoint_submits =
      large_linear_checkpoint_submit_count().exchange(
          0u, std::memory_order_relaxed);
  const uint64_t checkpoint_bytes =
      large_linear_checkpoint_bytes().exchange(0u, std::memory_order_relaxed);

  std::ostringstream stream;
  stream << "aten::linear.large_stack_checkpoint"
         << " submits=" << checkpoint_submits
         << " bytes=" << checkpoint_bytes
         << " weight_bytes=" << weight_bytes;
  utils::log_vulkan_op_hit(stream.str());

  api::AllocationScope allocation_scope("linear.large_stack_checkpoint");
  api::Context* const context = api::context();
  context->synchronize_stream(context->current_c10_stream());
  utils::release_retired_packed_weight_entries();
  utils::release_retired_linear_contexts();
}

struct DeferredLinearGeluCandidate final {
  Tensor input_arg;
  Tensor buffer_input;
  c10::intrusive_ptr<LinearPackedContext> linear_context;
  utils::VulkanRuntimePolicy runtime_policy;
  std::vector<int64_t> output_sizes;
  uint64_t producer_storage_id{0};
  uint64_t producer_generation{0};
  uint64_t producer_logical_desc_hash{0};
  float alpha{1.0f};
  float beta{1.0f};
};

constexpr size_t kMaxDeferredLinearGeluCandidates = 128;

struct TensorProducerKey final {
  uint64_t base_storage_id{0};
  uint64_t generation{0};
  uint64_t logical_desc_hash{0};
  const char* producer_op{"aten::linear"};
};

bool operator==(const TensorProducerKey& lhs, const TensorProducerKey& rhs) {
  return lhs.base_storage_id == rhs.base_storage_id &&
      lhs.generation == rhs.generation &&
      lhs.logical_desc_hash == rhs.logical_desc_hash &&
      lhs.producer_op == rhs.producer_op;
}

struct TensorProducerKeyHash final {
  size_t operator()(const TensorProducerKey& key) const {
    size_t seed = 0;
    seed ^= std::hash<uint64_t>{}(key.base_storage_id) +
        size_t{0x9e3779b97f4a7c15ull} + (seed << 6u) + (seed >> 2u);
    seed ^= std::hash<uint64_t>{}(key.generation) +
        size_t{0x9e3779b97f4a7c15ull} + (seed << 6u) + (seed >> 2u);
    seed ^= std::hash<uint64_t>{}(key.logical_desc_hash) +
        size_t{0x9e3779b97f4a7c15ull} + (seed << 6u) + (seed >> 2u);
    seed ^= std::hash<const char*>{}(key.producer_op) +
        size_t{0x9e3779b97f4a7c15ull} + (seed << 6u) + (seed >> 2u);
    return seed;
  }
};

TensorProducerKey deferred_linear_gelu_key(const Tensor& tensor) {
  const VulkanTensorStateDesc state = inspect_tensor_state(tensor);
  return TensorProducerKey{
      state.storage_id,
      state.generation,
      state.logical_desc_hash,
      "aten::linear"};
}

std::mutex& deferred_linear_gelu_candidate_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<
    TensorProducerKey,
    DeferredLinearGeluCandidate,
    TensorProducerKeyHash>&
deferred_linear_gelu_candidates() {
  static std::unordered_map<
      TensorProducerKey,
      DeferredLinearGeluCandidate,
      TensorProducerKeyHash>
      candidates;
  return candidates;
}

bool can_retarget_deferred_linear_gelu_candidate(
    const Tensor& tensor,
    const DeferredLinearGeluCandidate& candidate) {
  const VulkanTensorStateDesc state = inspect_tensor_state(tensor);
  return state.storage_id == candidate.producer_storage_id &&
      state.generation == candidate.producer_generation &&
      state.logical_desc_hash == candidate.producer_logical_desc_hash;
}

std::optional<DeferredLinearGeluCandidate>
lookup_deferred_linear_gelu_candidate(const Tensor& tensor) {
  std::lock_guard<std::mutex> lock(deferred_linear_gelu_candidate_mutex());
  auto& candidates = deferred_linear_gelu_candidates();
  const auto it = candidates.find(deferred_linear_gelu_key(tensor));
  if (it == candidates.end()) {
    return std::nullopt;
  }
  if (!can_retarget_deferred_linear_gelu_candidate(tensor, it->second)) {
    utils::log_vulkan_op_hit("aten::linear_gelu_bridge.stale_candidate");
    candidates.erase(it);
    return std::nullopt;
  }
  return it->second;
}

std::optional<DeferredLinearGeluCandidate>
take_deferred_linear_gelu_candidate(const Tensor& tensor) {
  std::lock_guard<std::mutex> lock(deferred_linear_gelu_candidate_mutex());
  auto& candidates = deferred_linear_gelu_candidates();
  const auto it = candidates.find(deferred_linear_gelu_key(tensor));
  if (it == candidates.end()) {
    return std::nullopt;
  }
  if (!can_retarget_deferred_linear_gelu_candidate(tensor, it->second)) {
    utils::log_vulkan_op_hit("aten::linear_gelu_bridge.stale_candidate");
    candidates.erase(it);
    return std::nullopt;
  }
  DeferredLinearGeluCandidate candidate = it->second;
  candidates.erase(it);
  return candidate;
}

void register_deferred_linear_gelu_candidate(
    const Tensor& tensor,
    DeferredLinearGeluCandidate candidate) {
  std::lock_guard<std::mutex> lock(deferred_linear_gelu_candidate_mutex());
  auto& candidates = deferred_linear_gelu_candidates();
  if (candidates.size() >= kMaxDeferredLinearGeluCandidates) {
    utils::log_vulkan_op_hit("aten::linear_gelu_bridge.registry_clear");
    candidates.clear();
  }
  const VulkanTensorStateDesc state = inspect_tensor_state(tensor);
  candidate.producer_storage_id = state.storage_id;
  candidate.producer_generation = state.generation;
  candidate.producer_logical_desc_hash = state.logical_desc_hash;
  candidates[deferred_linear_gelu_key(tensor)] = std::move(candidate);
}

bool can_run_float_buffer_linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias);

static Tensor reshape_to_2d(const Tensor& input_arg);

LinearPackedRunState get_linear_packed_run_state(
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  const PackedWeightHandle& packed_weight = linear_context->packed_weight();
  return {
      packed_weight,
      packed_weight.weight_vtensor(),
      packed_weight.bias_vtensor(),
      packed_weight.logical_weight_sizes(),
      packed_weight.has_bias(),
  };
}

Tensor ensure_linear_buffer_output_tensor(
    Tensor& output,
    IntArrayRef sizes,
    const c10::ScalarType dtype) {
  bool needs_allocation = !output.defined() || !output.is_vulkan() ||
      output.scalar_type() != dtype || !output.sizes().equals(sizes);
  if (!needs_allocation) {
    const vTensor& v_output = convert(output);
    needs_allocation =
        v_output.storage_type() != api::StorageType::BUFFER ||
        v_output.gpu_memory_layout() !=
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED ||
        !v_output.has_direct_buffer_layout();
  }
  if (needs_allocation) {
    output = utils::mark_tensor_execution(
        convert(vTensor{
            api::context(),
            sizes.vec(),
            convert_dtype(dtype),
            api::StorageType::BUFFER,
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
        }),
        api::ExecutionLayout::BUFFER_DIRECT);
  }
  return output;
}

Tensor ensure_bmm_buffer_output_tensor(
    Tensor& output,
    IntArrayRef sizes,
    const c10::ScalarType dtype) {
  bool needs_allocation = !output.defined() || !output.is_vulkan() ||
      output.scalar_type() != dtype || !output.sizes().equals(sizes);
  if (!needs_allocation) {
    const vTensor& v_output = convert(output);
    needs_allocation =
        v_output.storage_type() != api::StorageType::BUFFER ||
        v_output.gpu_memory_layout() !=
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED ||
        !utils::supports_buffer_view_fast_path(v_output);
  }
  if (needs_allocation) {
    output = utils::mark_tensor_execution(
        convert(vTensor{
            api::context(),
            sizes.vec(),
            convert_dtype(dtype),
            api::StorageType::BUFFER,
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
        }),
        api::ExecutionLayout::BUFFER_DIRECT);
  } else {
    output = utils::mark_tensor_execution(
        output,
        utils::resolve_buffer_execution_layout(convert(output)));
  }
  return output;
}

bool can_fuse_linear_bias(
    const vTensor& v_output,
    const vTensor& v_bias,
    const std::vector<int64_t>& weight_sizes) {
  if (
      v_bias.storage_type() != api::StorageType::TEXTURE_3D ||
      v_bias.gpu_memory_layout() !=
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED) {
    return false;
  }

  const IntArrayRef bias_sizes = v_bias.sizes();
  if (bias_sizes.empty() || bias_sizes.size() > 2) {
    return false;
  }

  const int64_t output_width = weight_sizes[Layout::Parameter::width];
  const int64_t output_height = v_output.sizes()[Layout::Parameter::height];
  const int64_t bias_width = bias_sizes.back();
  const int64_t bias_height =
      bias_sizes.size() == 2 ? bias_sizes.front() : 1;

  return bias_width == output_width &&
      (bias_height == 1 || bias_height == output_height);
}

bool can_use_channel_packed_linear_input(
    const vTensor& v_input,
    const vTensor& packed_v_weight) {
  return v_input.dtype() == api::kFloat &&
      v_input.storage_type() == api::StorageType::TEXTURE_3D &&
      v_input.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED &&
      v_input.sizes().size() == 2 &&
      !v_input.is_quantized() &&
      packed_v_weight.dtype() == api::kFloat &&
      packed_v_weight.storage_type() == api::StorageType::TEXTURE_3D &&
      packed_v_weight.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED &&
      !packed_v_weight.is_quantized();
}

bool linear_kernel_family_allows_channel_packed_input(
    const utils::VulkanRuntimePolicy& runtime_policy) {
  if (
      runtime_policy.request.model_domain == utils::VulkanModelDomain::Generic &&
      runtime_policy.request.execution_phase ==
          utils::VulkanExecutionPhase::None) {
    return true;
  }

  switch (runtime_policy.linear_kernel_family) {
    case utils::VulkanLinearKernelFamily::TexturePacked:
      return false;
    case utils::VulkanLinearKernelFamily::UnifiedBufferView:
    case utils::VulkanLinearKernelFamily::PersistentPackedTexture:
    case utils::VulkanLinearKernelFamily::CooperativeMatrix:
      return true;
  }
  return true;
}

Tensor reshape_linear_output_if_needed(
    const Tensor& output,
    const Tensor& input_arg) {
  if (input_arg.dim() == 2) {
    return output;
  }

  std::vector<int64_t> shape;
  shape.reserve(static_cast<size_t>(std::max<int64_t>(0, input_arg.dim())));
  for (const auto i : c10::irange(input_arg.dim() - 1)) {
    shape.emplace_back(input_arg.size(i));
  }
  shape.emplace_back(output.size(-1));
  Tensor reshaped_output = utils::reshape_inference(output, shape);
  const bool large_buffer_linear_view =
      output.is_vulkan() && output.numel() >= (1 << 20);
  if ((c10::InferenceMode::is_enabled() || large_buffer_linear_view) &&
      reshaped_output.is_vulkan()) {
    const vTensor& v_reshaped_output = convert(reshaped_output);
    const bool needs_materialization =
        v_reshaped_output.storage_type() == api::StorageType::BUFFER &&
        !v_reshaped_output.has_direct_buffer_layout();
    if (needs_materialization) {
      if (large_buffer_linear_view) {
        utils::log_vulkan_op_hit("aten::linear.materialize_large_buffer_view");
        reshaped_output = utils::ensure_buffer_storage(
            reshaped_output, v_reshaped_output.gpu_memory_layout());
      } else {
        reshaped_output = reshaped_output.clone();
      }
    }
  } else if (c10::InferenceMode::is_enabled()) {
    reshaped_output = reshaped_output.clone();
  }
  return reshaped_output;
}

utils::LinearGeluBridgeTensorInfo linear_gelu_bridge_tensor_info(
    const Tensor& input_arg,
    const Tensor& input_arg_2d) {
  utils::LinearGeluBridgeTensorInfo info;
  info.input_rank = input_arg.dim();
  if (input_arg.dim() == 2) {
    info.input_rows = input_arg.size(0);
    info.input_features = input_arg.size(1);
  } else if (input_arg.dim() == 3) {
    info.input_batch = input_arg.size(0);
    info.input_rows = input_arg.size(1);
    info.input_features = input_arg.size(2);
  }
  info.flattened_rank = input_arg_2d.dim();
  if (input_arg_2d.dim() == 2) {
    info.flattened_rows = input_arg_2d.size(0);
    info.flattened_features = input_arg_2d.size(1);
  }
  return info;
}

utils::LinearGeluBridgeMatch match_linear_gelu_bridge_candidate(
    const Tensor& input_arg,
    const Tensor& input_arg_2d,
    const LinearPackedRunState& packed_state,
    const float alpha,
    const float beta,
    const LinearPostOp post_op,
    const Tensor* output_opt) {
  const Tensor& packed_weight_tensor = packed_state.packed_weight.weight();
  const std::optional<Tensor> packed_bias_tensor = packed_state.bias_defined
      ? std::optional<Tensor>(packed_state.packed_weight.bias())
      : std::nullopt;
  const utils::LinearGeluBridgePackedInfo packed_info{
      packed_state.logical_weight_sizes[Layout::Parameter::height],
      packed_state.logical_weight_sizes[Layout::Parameter::width],
      packed_state.bias_defined,
      can_run_float_buffer_linear(
          input_arg_2d, packed_weight_tensor, packed_bias_tensor)};
  const utils::LinearGeluBridgeOptions options{
      c10::InferenceMode::is_enabled(),
      output_opt != nullptr,
      post_op == LinearPostOp::None,
      alpha == 1.0f,
      beta == 1.0f};
  return utils::match_linear_gelu_bridge_contract(
      linear_gelu_bridge_tensor_info(input_arg, input_arg_2d),
      packed_info,
      options);
}

Tensor make_deferred_linear_gelu_placeholder(
    const Tensor& input_arg,
    const Tensor& input_arg_2d,
    const LinearPackedRunState& packed_state) {
  Tensor output_2d = utils::mark_tensor_execution(
      convert(vTensor{
          api::context(),
          {
              input_arg_2d.size(Layout::Parameter::height),
              packed_state.logical_weight_sizes[Layout::Parameter::width],
          },
          api::kFloat,
          api::StorageType::BUFFER,
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      }),
      api::ExecutionLayout::BUFFER_DIRECT);
  return reshape_linear_output_if_needed(output_2d, input_arg);
}

Tensor reshape_deferred_linear_gelu_output_if_needed(
    const Tensor& output,
    const DeferredLinearGeluCandidate& candidate) {
  if (output.sizes().vec() == candidate.output_sizes) {
    return output;
  }
  return utils::reshape_inference(output, candidate.output_sizes);
}

Tensor& ensure_linear_output_tensor(
    Tensor& output,
    IntArrayRef sizes,
    const c10::ScalarType dtype) {
  bool needs_allocation = !output.defined() || !output.is_vulkan() ||
      output.scalar_type() != dtype || !output.sizes().equals(sizes);
  if (!needs_allocation) {
    const vTensor& v_output = convert(output);
    needs_allocation =
        v_output.storage_type() != api::StorageType::TEXTURE_3D;
  }
  if (needs_allocation) {
    output = convert(vTensor{
        api::context(),
        sizes.vec(),
        convert_dtype(dtype),
    });
  }
  return output;
}

bool can_run_float_buffer_linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  if (
      input.device().type() != c10::DeviceType::Vulkan ||
      weight.device().type() != c10::DeviceType::Vulkan ||
      input.scalar_type() != kFloat ||
      weight.scalar_type() != kFloat ||
      input.dim() != 2 ||
      weight.dim() != 2 ||
      input.requires_grad() ||
      weight.requires_grad() ||
      input.size(Layout::Parameter::width) !=
          weight.size(Layout::Parameter::height)) {
    return false;
  }

  const vTensor& v_input = convert(input);
  const vTensor& v_weight = convert(weight);
  if (
      v_input.storage_type() != api::StorageType::BUFFER ||
      v_weight.storage_type() != api::StorageType::BUFFER ||
      v_input.gpu_memory_layout() != api::GPUMemoryLayout::TENSOR_WIDTH_PACKED ||
      v_weight.gpu_memory_layout() !=
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED ||
      !utils::supports_buffer_view_fast_path(v_input) ||
      !utils::supports_buffer_view_fast_path(v_weight)) {
    return false;
  }

  if (bias && bias->defined()) {
    if (
        bias->device().type() != c10::DeviceType::Vulkan ||
        bias->dim() > 2 ||
        bias->requires_grad() ||
        bias->scalar_type() != kFloat) {
      return false;
    }

    const vTensor& v_bias = convert(*bias);
    if (
        v_bias.storage_type() != api::StorageType::BUFFER ||
        !utils::supports_buffer_view_fast_path(v_bias)) {
      return false;
    }
  }

  return utils::admit_dynamic_program(
             utils::make_linear_or_matmul_direct_buffer_program_request(
                 input.sizes(),
                 weight.sizes(),
                 bias && bias->defined(),
                 input.scalar_type()))
      .accepted;
}

bool can_run_widened_half_buffer_linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  if (
      (input.scalar_type() != kHalf && input.scalar_type() != kBFloat16) ||
      weight.scalar_type() != kFloat ||
      input.dim() != 2 ||
      weight.dim() != 2 ||
      input.requires_grad() ||
      weight.requires_grad() ||
      input.size(Layout::Parameter::width) !=
          weight.size(Layout::Parameter::height)) {
    return false;
  }

  if (bias && bias->defined()) {
    if (
        bias->device().type() != c10::DeviceType::Vulkan ||
        bias->requires_grad() ||
        bias->scalar_type() != kFloat) {
      return false;
    }
  }

  return true;
}

Tensor widen_half_linear_tensor_to_float_buffer(const Tensor& tensor) {
  Tensor widened = upcast_half_linear_tensor_for_packing(tensor);
  Tensor vulkan_widened = widened.is_vulkan() ? widened : widened.vulkan();
  return utils::mark_tensor_execution(
      utils::ensure_buffer_storage(
          vulkan_widened, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
      api::ExecutionLayout::BUFFER_DIRECT);
}

bool can_run_float_buffer_bmm(const Tensor& mat1, const Tensor& mat2) {
  if (
      mat1.device().type() != c10::DeviceType::Vulkan ||
      mat2.device().type() != c10::DeviceType::Vulkan ||
      mat1.scalar_type() != kFloat ||
      mat2.scalar_type() != kFloat ||
      mat1.dim() != 3 ||
      mat2.dim() != 3 ||
      mat1.size(Layout::BatchMatrices::batch) !=
          mat2.size(Layout::BatchMatrices::batch) ||
      mat1.size(Layout::BatchMatrices::width) !=
          mat2.size(Layout::BatchMatrices::height)) {
    return false;
  }

  const vTensor& v_mat1 = convert(mat1);
  const vTensor& v_mat2 = convert(mat2);
  return v_mat1.storage_type() == api::StorageType::BUFFER &&
      v_mat2.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_view_fast_path(v_mat1) &&
      utils::supports_buffer_view_fast_path(v_mat2);
}

bool should_use_tiled_buffer_linear_kernel(
    const utils::VulkanRuntimePolicy& runtime_policy,
    const Tensor& input_arg_2d,
    IntArrayRef output_sizes,
    const LinearPostOp post_op,
    const bool bias_defined) {
  const int64_t input_height = input_arg_2d.size(Layout::Parameter::height);
  const int64_t input_width = input_arg_2d.size(Layout::Parameter::width);
  const int64_t output_width = output_sizes[Layout::Parameter::width];
  const char* role =
      linear_role_from_label(api::current_allocation_label(), post_op);
  const VulkanLinearPlanContractMatch exact_vision_tiled_contract =
      match_vision_exact_tiled_linear_plan(
          runtime_policy,
          role,
          input_arg_2d.scalar_type(),
          input_height,
          input_width,
          output_width,
          bias_defined,
          post_op);

  if (exact_vision_tiled_contract.matched) {
    return true;
  }

  const bool generic_large_buffer_matmul =
      runtime_policy.request.model_domain ==
          utils::VulkanModelDomain::Generic &&
      runtime_policy.request.execution_phase ==
          utils::VulkanExecutionPhase::None &&
      input_height >= 256 && input_width >= 128 && output_width >= 256;
  if (!generic_large_buffer_matmul) {
    return false;
  }

  // The generic tiled buffer-linear shader family corrupts diffusion-style
  // transformer linears such as [384,1280] x [1280,1280] on multiple devices.
  // Route generic and broad vision-labeled linears through the older Vulkan
  // buffer kernel until a contract-bounded tiled route is proven.
  return false;
}

Tensor run_float_buffer_linear(
    const Tensor& input_arg,
    const Tensor& input_arg_2d,
    const utils::VulkanRuntimePolicy& runtime_policy,
    const LinearPackedRunState& packed_state,
    const float alpha,
    const float beta,
    const LinearPostOp post_op,
    Tensor* output_opt = nullptr) {
  api::Context* const context = api::context();

  const Tensor& packed_weight_tensor = packed_state.packed_weight.weight();
  const std::optional<Tensor> packed_bias_tensor = packed_state.bias_defined
      ? std::optional<Tensor>(packed_state.packed_weight.bias())
      : std::nullopt;

  TORCH_INTERNAL_ASSERT(
      can_run_float_buffer_linear(
          input_arg_2d, packed_weight_tensor, packed_bias_tensor));

  Tensor input_tensor = input_arg_2d;
  Tensor weight_tensor = packed_weight_tensor;
  const std::vector<int64_t> output_sizes{
      input_arg_2d.sizes()[Layout::Parameter::height],
      packed_state.logical_weight_sizes[Layout::Parameter::width],
  };
  const bool should_use_tiled_kernel =
      should_use_tiled_buffer_linear_kernel(
          runtime_policy,
          input_tensor,
          output_sizes,
          post_op,
          packed_state.bias_defined);
  const vTensor& v_input_view = convert(input_tensor);
  if (
      v_input_view.storage_type() == api::StorageType::BUFFER &&
      !v_input_view.has_direct_buffer_layout() &&
      should_use_tiled_kernel) {
    utils::log_vulkan_op_hit(
        "aten::linear.materialize_tiled_input_view");
    input_tensor = utils::mark_tensor_execution(
        utils::ensure_buffer_storage(
            input_tensor, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
        api::ExecutionLayout::BUFFER_DIRECT);
  }
  vTensor& v_input = convert(input_tensor);
  vTensor& v_weight = convert(weight_tensor);
  Tensor output_tensor = output_opt
      ? ensure_linear_buffer_output_tensor(
            *output_opt, output_sizes, input_arg_2d.scalar_type())
      : utils::mark_tensor_execution(
            convert(vTensor{
                context,
                output_sizes,
                api::kFloat,
                api::StorageType::BUFFER,
                api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
            }),
            api::ExecutionLayout::BUFFER_DIRECT);
  vTensor& v_output = convert(output_tensor);

  const struct {
    int32_t out_width;
    int32_t out_height;
    int32_t inner_dim;
    int32_t reserved;
  } block{
      api::utils::safe_downcast<int32_t>(
          packed_state.logical_weight_sizes[Layout::Parameter::width]),
      api::utils::safe_downcast<int32_t>(
          input_arg_2d.size(Layout::Parameter::height)),
      api::utils::safe_downcast<int32_t>(
          input_arg_2d.size(Layout::Parameter::width)),
      0,
  };
  Tensor fused_bias_tensor;
  bool fuse_buffer_bias_gelu = false;
  bool fuse_buffer_bias = false;
  if (packed_state.bias_defined && alpha == 1.0f && beta == 1.0f) {
    fused_bias_tensor = packed_state.packed_weight.bias();
    const vTensor& v_bias = convert(fused_bias_tensor);
    const bool can_fuse_buffer_bias =
        v_bias.storage_type() == api::StorageType::BUFFER &&
        v_bias.gpu_memory_layout() ==
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
        v_bias.has_direct_buffer_layout() && v_bias.sizes().size() == 1 &&
        v_bias.sizes()[0] == output_sizes[Layout::Parameter::width];
    fuse_buffer_bias_gelu =
        can_fuse_buffer_bias && post_op == LinearPostOp::Gelu;
    fuse_buffer_bias =
        can_fuse_buffer_bias && post_op == LinearPostOp::None;
  }
  const bool use_specialized_tiled_kernel =
      should_use_tiled_kernel &&
      (!packed_state.bias_defined || fuse_buffer_bias || fuse_buffer_bias_gelu);
  const VulkanLinearPlanContractMatch exact_vision_tiled_contract =
      match_vision_exact_tiled_linear_plan(
          runtime_policy,
          linear_role_from_label(api::current_allocation_label(), post_op),
          input_arg_2d.scalar_type(),
          input_arg_2d.size(Layout::Parameter::height),
          input_arg_2d.size(Layout::Parameter::width),
          output_sizes[Layout::Parameter::width],
          packed_state.bias_defined,
          post_op);
  const bool use_vec2_tiled_kernel =
      use_specialized_tiled_kernel &&
      output_sizes[Layout::Parameter::width] >= 384 &&
      input_arg_2d.size(Layout::Parameter::width) % 16 == 0 &&
      (input_arg_2d.size(Layout::Parameter::height) >= 512 ||
       exact_vision_tiled_contract.prefer_vec2_tiled);

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(
          packed_state.logical_weight_sizes[Layout::Parameter::width]),
      api::utils::safe_downcast<uint32_t>(
          input_arg_2d.size(Layout::Parameter::height)),
      1u,
  };

  if (fuse_buffer_bias_gelu || fuse_buffer_bias) {
    const api::utils::uvec3 local_size =
        use_specialized_tiled_kernel ? api::utils::uvec3{16u, 16u, 1u}
                                     : api::utils::uvec3{16u, 4u, 1u};
    vTensor& v_bias = convert(fused_bias_tensor);
    const char* kernel_hit_name =
        use_vec2_tiled_kernel
            ? (fuse_buffer_bias_gelu
                   ? "aten::linear.buffer_float_tiled_bias_vec2_gelu"
                   : "aten::linear.buffer_float_tiled_bias_vec2")
        : use_specialized_tiled_kernel
            ? (fuse_buffer_bias_gelu
                   ? "aten::linear.buffer_float_tiled_bias_gelu"
                   : "aten::linear.buffer_float_tiled_bias")
            : (fuse_buffer_bias_gelu ? "aten::linear.buffer_float_bias_gelu"
                                      : "aten::linear.buffer_float_bias");
    log_float_buffer_linear_submit(
        kernel_hit_name,
        input_arg,
        input_arg_2d,
        v_input,
        v_weight,
        v_output,
        packed_state,
        output_sizes,
        runtime_policy,
        true,
        use_specialized_tiled_kernel,
        use_vec2_tiled_kernel,
        post_op);
    note_linear_aggregate(
        kernel_hit_name,
        input_arg_2d,
        runtime_policy,
        packed_weight_tensor,
        packed_bias_tensor,
        output_tensor,
        v_input,
        v_output,
        packed_state,
        output_sizes,
        global_size,
        local_size,
        post_op);
    utils::log_vulkan_op_hit(kernel_hit_name);
    context->submit_compute_job(
        use_vec2_tiled_kernel
            ? (fuse_buffer_bias_gelu
                   ? VK_KERNEL(mm_buffer_float_tiled_bias_vec2_gelu)
                   : VK_KERNEL(mm_buffer_float_tiled_bias_vec2))
        : use_specialized_tiled_kernel
            ? (fuse_buffer_bias_gelu ? VK_KERNEL(mm_buffer_float_tiled_bias_gelu)
                                     : VK_KERNEL(mm_buffer_float_tiled_bias))
            : (fuse_buffer_bias_gelu ? VK_KERNEL(mm_buffer_float_bias_gelu)
                                     : VK_KERNEL(mm_buffer_float_bias)),
        pipeline_barrier,
        global_size,
        local_size,
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_output.buffer_metadata(),
        v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_input.buffer_metadata(),
        v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_weight.buffer_metadata(),
        v_bias.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_bias.buffer_metadata(),
        params.buffer());
  } else {
    const api::utils::uvec3 local_size =
        use_specialized_tiled_kernel ? api::utils::uvec3{8u, 8u, 1u}
                                     : api::utils::uvec3{16u, 4u, 1u};
    const char* kernel_hit_name =
        use_specialized_tiled_kernel ? "aten::linear.buffer_float_tiled"
                                     : "aten::linear.buffer_float";
    log_float_buffer_linear_submit(
        kernel_hit_name,
        input_arg,
        input_arg_2d,
        v_input,
        v_weight,
        v_output,
        packed_state,
        output_sizes,
        runtime_policy,
        false,
        use_specialized_tiled_kernel,
        use_vec2_tiled_kernel,
        post_op);
    note_linear_aggregate(
        kernel_hit_name,
        input_arg_2d,
        runtime_policy,
        packed_weight_tensor,
        packed_bias_tensor,
        output_tensor,
        v_input,
        v_output,
        packed_state,
        output_sizes,
        global_size,
        local_size,
        post_op);
    utils::log_vulkan_op_hit(kernel_hit_name);
    context->submit_compute_job(
        use_specialized_tiled_kernel ? VK_KERNEL(mm_buffer_float_tiled)
                                     : VK_KERNEL(mm_buffer_float),
        pipeline_barrier,
        global_size,
        local_size,
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_output.buffer_metadata(),
        v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_input.buffer_metadata(),
        v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_weight.buffer_metadata(),
        params.buffer());
  }

  Tensor output = output_tensor;
  if (!fuse_buffer_bias_gelu && !fuse_buffer_bias && alpha != 1.0f) {
    output = output.mul(alpha);
  }
  if (!fuse_buffer_bias_gelu && !fuse_buffer_bias && packed_state.bias_defined) {
    Tensor bias = packed_state.packed_weight.bias();
    if (beta != 1.0f) {
      bias = bias.mul(beta);
    }
    output = output.add(bias);
  }
  if (!fuse_buffer_bias_gelu && post_op == LinearPostOp::Gelu) {
    output = at::gelu(output, "none");
  }
  if (output_opt && output.unsafeGetTensorImpl() != output_tensor.unsafeGetTensorImpl()) {
    *output_opt = output;
    output = *output_opt;
  }

  VulkanLinearPlanDecision decision;
  decision.selected = VulkanLinearFastPath::FloatBuffer;
  decision.reject = VulkanLinearRejectReason::None;
  decision.m = input_arg_2d.size(Layout::Parameter::height);
  decision.k = input_arg_2d.size(Layout::Parameter::width);
  decision.n = packed_state.logical_weight_sizes[Layout::Parameter::width];
  decision.input_dtype = input_arg_2d.scalar_type();
  decision.weight_dtype = packed_weight_tensor.scalar_type();
  decision.bias_dtype = packed_bias_tensor && packed_bias_tensor->defined()
      ? packed_bias_tensor->scalar_type()
      : ScalarType::Undefined;
  decision.output_dtype = output.scalar_type();
  decision.input_vulkan = input_arg_2d.is_vulkan();
  decision.weight_packed = true;
  decision.weight_vulkan = packed_weight_tensor.is_vulkan();
  decision.bias_present =
      packed_bias_tensor.has_value() && packed_bias_tensor->defined();
  decision.bias_vulkan =
      decision.bias_present && packed_bias_tensor->is_vulkan();
  decision.input_direct_buffer = v_input.has_direct_buffer_layout();
  decision.output_direct_buffer = v_output.has_direct_buffer_layout();
  decision.can_use_coop_matrix = false;
  decision.has_post_op = post_op != LinearPostOp::None;
  decision.m_tail = decision.m % 16 != 0;
  decision.k_tail = decision.k % 16 != 0;
  decision.n_tail = decision.n % 16 != 0;
  decision.rejected_because_float_input =
      decision.input_dtype == ScalarType::Float;
  decision.rejected_because_float_weight =
      decision.weight_dtype == ScalarType::Float;
  decision.rejected_because_bias_dtype =
      decision.bias_present && decision.bias_dtype != ScalarType::BFloat16;
  decision.rejected_because_layout =
      !decision.input_direct_buffer || !decision.output_direct_buffer;
  decision.rejected_because_not_packed = !decision.weight_packed;
  note_linear_plan_decision(decision);
  append_vulkan_linear_plan_log(decision, "aten::linear.float_buffer");
  maybe_synchronize_after_large_linear_checkpoint(
      input_arg_2d, packed_weight_tensor, output);

  return reshape_linear_output_if_needed(output, input_arg);
}

Tensor run_raw_direct_float_buffer_linear(
    const Tensor& input_arg,
    const Tensor& input_arg_2d,
    const utils::VulkanRuntimePolicy& runtime_policy,
    const Tensor& weight_tensor) {
  api::Context* const context = api::context();
  TORCH_INTERNAL_ASSERT(
      can_run_float_buffer_linear(input_arg_2d, weight_tensor, std::nullopt));

  vTensor& v_input = convert(input_arg_2d);
  vTensor& v_weight = convert(weight_tensor);
  const std::vector<int64_t> output_sizes{
      input_arg_2d.sizes()[Layout::Parameter::height],
      weight_tensor.sizes()[Layout::Parameter::width],
  };
  Tensor output_tensor = utils::mark_tensor_execution(
      convert(vTensor{
          context,
          output_sizes,
          api::kFloat,
          api::StorageType::BUFFER,
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      }),
      api::ExecutionLayout::BUFFER_DIRECT);
  vTensor& v_output = convert(output_tensor);

  const struct {
    int32_t out_width;
    int32_t out_height;
    int32_t inner_dim;
    int32_t reserved;
  } block{
      api::utils::safe_downcast<int32_t>(output_sizes[Layout::Parameter::width]),
      api::utils::safe_downcast<int32_t>(
          input_arg_2d.size(Layout::Parameter::height)),
      api::utils::safe_downcast<int32_t>(
          input_arg_2d.size(Layout::Parameter::width)),
      0,
  };
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(
          output_sizes[Layout::Parameter::width]),
      api::utils::safe_downcast<uint32_t>(
          input_arg_2d.size(Layout::Parameter::height)),
      1u,
  };
  const api::utils::uvec3 local_size{16u, 4u, 1u};
  const char* kernel_hit_name = "aten::linear.raw_direct_weight";
  std::ostringstream stream;
  stream << "aten::linear.submit"
         << " kernel=" << kernel_hit_name
         << " input=" << format_linear_sizes(input_arg.sizes())
         << " input2d=" << format_linear_sizes(input_arg_2d.sizes())
         << " output2d=" << format_linear_sizes(output_sizes)
         << " weight=" << format_linear_sizes(weight_tensor.sizes())
         << " bias=0"
         << " post=none"
         << " tiled=0"
         << " vec2=0"
         << " raw_direct=1"
         << " input_direct=" << (v_input.has_direct_buffer_layout() ? 1 : 0)
         << " weight_direct=" << (v_weight.has_direct_buffer_layout() ? 1 : 0)
         << " output_direct=" << (v_output.has_direct_buffer_layout() ? 1 : 0)
         << " input_offset=" << v_input.storage_offset()
         << " weight_offset=" << v_weight.storage_offset()
         << " output_offset=" << v_output.storage_offset()
         << " domain="
         << static_cast<int>(runtime_policy.request.model_domain)
         << " phase="
         << static_cast<int>(runtime_policy.request.execution_phase);
  utils::log_vulkan_op_hit(stream.str());
  note_raw_direct_linear_aggregate(
      kernel_hit_name,
      input_arg_2d,
      runtime_policy,
      weight_tensor,
      output_tensor,
      v_input,
      v_weight,
      v_output,
      output_sizes,
      global_size,
      local_size);
  utils::log_vulkan_op_hit(kernel_hit_name);
  context->submit_compute_job(
      VK_KERNEL(mm_buffer_float),
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_output.buffer_metadata(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_input.buffer_metadata(),
      v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.buffer_metadata(),
      params.buffer());

  Tensor output = output_tensor;
  VulkanLinearPlanDecision decision;
  decision.selected = VulkanLinearFastPath::FloatBuffer;
  decision.reject = VulkanLinearRejectReason::None;
  decision.m = input_arg_2d.size(Layout::Parameter::height);
  decision.k = input_arg_2d.size(Layout::Parameter::width);
  decision.n = output_sizes[Layout::Parameter::width];
  decision.input_dtype = input_arg_2d.scalar_type();
  decision.weight_dtype = weight_tensor.scalar_type();
  decision.bias_dtype = ScalarType::Undefined;
  decision.output_dtype = output.scalar_type();
  decision.input_vulkan = input_arg_2d.is_vulkan();
  decision.weight_packed = false;
  decision.weight_vulkan = weight_tensor.is_vulkan();
  decision.bias_present = false;
  decision.bias_vulkan = false;
  decision.input_direct_buffer = v_input.has_direct_buffer_layout();
  decision.output_direct_buffer = v_output.has_direct_buffer_layout();
  decision.can_use_coop_matrix = false;
  decision.has_post_op = false;
  decision.m_tail = decision.m % 16 != 0;
  decision.k_tail = decision.k % 16 != 0;
  decision.n_tail = decision.n % 16 != 0;
  decision.rejected_because_float_input =
      decision.input_dtype == ScalarType::Float;
  decision.rejected_because_float_weight =
      decision.weight_dtype == ScalarType::Float;
  decision.rejected_because_bias_dtype = false;
  decision.rejected_because_layout =
      !decision.input_direct_buffer || !decision.output_direct_buffer;
  decision.rejected_because_not_packed = false;
  note_linear_plan_decision(decision);
  append_vulkan_linear_plan_log(decision, "aten::linear.raw_direct_weight");
  maybe_synchronize_after_large_linear_checkpoint(
      input_arg_2d, weight_tensor, output);
  return reshape_linear_output_if_needed(output, input_arg);
}

std::optional<Tensor> try_run_raw_direct_weight_linear(
    const Tensor& input_arg,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  if (
      !c10::InferenceMode::is_enabled() ||
      !utils::current_vulkan_device_policy()
           .avoid_large_persistent_weight_cache ||
      !input_arg.is_vulkan() ||
      !weight.is_vulkan() ||
      input_arg.scalar_type() != kFloat ||
      weight.scalar_type() != kFloat ||
      weight.dim() != 2 ||
      (bias && bias->defined())) {
    return std::nullopt;
  }

  c10::DimVector weight_view_sizes;
  c10::DimVector weight_view_logical_strides;
  c10::DimVector weight_view_physical_strides;
  if (!can_make_vulkan_linear_weight_transpose_view(
          weight,
          weight_view_sizes,
          weight_view_logical_strides,
          weight_view_physical_strides)) {
    return std::nullopt;
  }

  const auto input_request =
      utils::make_vulkan_tensor_linear_request(
          input_arg, utils::VulkanTensorRole::Input);
  const auto runtime_policy = utils::build_vulkan_runtime_policy(input_request);
  const Tensor compute_input_arg = utils::prepare_vulkan_execution_tensor(
      input_arg,
      utils::VulkanExecutionPlanKind::LinearInputSource,
      input_request);
  const Tensor input_arg_2d =
      compute_input_arg.dim() == 2 ? compute_input_arg
                                   : reshape_to_2d(compute_input_arg);
  const Tensor input =
      input_arg_2d.is_vulkan() ? input_arg_2d : input_arg_2d.vulkan();
  const Tensor weight_view = vulkan_linear_weight_transpose_view_for_packing(
      weight,
      weight_view_sizes,
      weight_view_logical_strides,
      weight_view_physical_strides);
  if (!can_run_float_buffer_linear(input, weight_view, std::nullopt)) {
    return std::nullopt;
  }

  utils::log_vulkan_op_hit(
      "aten::linear.raw_direct_weight.accepted contract=RawDirectWeightLinearPlan family=NoCacheFloatBuffer");
  return run_raw_direct_float_buffer_linear(
      input_arg,
      input,
      runtime_policy,
      weight_view);
}

Tensor run_widened_half_buffer_linear(
    const Tensor& input_arg,
    const Tensor& input_arg_2d,
    const utils::VulkanRuntimePolicy& runtime_policy,
    const LinearPackedRunState& packed_state,
    const float alpha,
    const float beta,
    const LinearPostOp post_op,
    Tensor* output_opt = nullptr) {
  const c10::ScalarType output_dtype = input_arg.scalar_type();
  const Tensor float_input_2d =
      widen_half_linear_tensor_to_float_buffer(input_arg_2d);
  const std::optional<Tensor> packed_bias_tensor = packed_state.bias_defined
      ? std::optional<Tensor>(packed_state.packed_weight.bias())
      : std::nullopt;

  TORCH_INTERNAL_ASSERT(
      can_run_float_buffer_linear(
          float_input_2d, packed_state.packed_weight.weight(), packed_bias_tensor));

  Tensor float_output_2d = run_float_buffer_linear(
      input_arg_2d,
      float_input_2d,
      runtime_policy,
      packed_state,
      alpha,
      beta,
      post_op);
  Tensor output_2d = output_dtype == kFloat
      ? float_output_2d
      : float_output_2d.to(output_dtype);
  Tensor output = reshape_linear_output_if_needed(output_2d, input_arg);
  if (output_opt &&
      output.unsafeGetTensorImpl() != output_opt->unsafeGetTensorImpl()) {
    *output_opt = output;
    output = *output_opt;
  }
  return output;
}

Tensor materialize_deferred_linear_gelu_candidate_impl(const Tensor& tensor) {
  if (!tensor.is_vulkan()) {
    return tensor;
  }

  auto candidate = take_deferred_linear_gelu_candidate(tensor);
  if (!candidate.has_value()) {
    return tensor;
  }

  const LinearPackedRunState packed_state =
      get_linear_packed_run_state(candidate->linear_context);
  utils::log_vulkan_op_hit("aten::linear_gelu_bridge.materialize");
  Tensor output = run_float_buffer_linear(
      candidate->input_arg,
      candidate->buffer_input,
      candidate->runtime_policy,
      packed_state,
      candidate->alpha,
      candidate->beta,
      LinearPostOp::None);
  return reshape_deferred_linear_gelu_output_if_needed(output, *candidate);
}

void move_deferred_linear_gelu_candidate_to_alias_impl(
    const Tensor& source,
    const Tensor& alias) {
  if (!source.is_vulkan() || !alias.is_vulkan()) {
    return;
  }

  auto candidate = take_deferred_linear_gelu_candidate(source);
  if (!candidate.has_value()) {
    return;
  }

  candidate->output_sizes = alias.sizes().vec();
  register_deferred_linear_gelu_candidate(alias, std::move(*candidate));
  utils::log_vulkan_op_hit("aten::linear_gelu_bridge.alias");
}

std::optional<Tensor> try_consume_deferred_linear_gelu_impl(
    const Tensor& input,
    std::string_view approximate) {
  if (!utils::matches_linear_gelu_bridge_gelu_approximation_contract(
          approximate)) {
    return std::nullopt;
  }

  const auto candidate = lookup_deferred_linear_gelu_candidate(input);
  if (!candidate.has_value()) {
    return std::nullopt;
  }

  const LinearPackedRunState packed_state =
      get_linear_packed_run_state(candidate->linear_context);
  const std::optional<Tensor> packed_bias_tensor = packed_state.bias_defined
      ? std::optional<Tensor>(packed_state.packed_weight.bias())
      : std::nullopt;
  if (!can_run_float_buffer_linear(
          candidate->buffer_input,
          packed_state.packed_weight.weight(),
          packed_bias_tensor)) {
    return std::nullopt;
  }

  auto taken = take_deferred_linear_gelu_candidate(input);
  if (!taken.has_value()) {
    return std::nullopt;
  }

  const LinearPackedRunState taken_packed_state =
      get_linear_packed_run_state(taken->linear_context);
  utils::log_vulkan_op_hit("aten::linear_gelu_bridge.hit");
  Tensor output = run_float_buffer_linear(
      taken->input_arg,
      taken->buffer_input,
      taken->runtime_policy,
      taken_packed_state,
      taken->alpha,
      taken->beta,
      LinearPostOp::Gelu);
  return reshape_deferred_linear_gelu_output_if_needed(output, *taken);
}

Tensor run_float_buffer_bmm(
    const Tensor& mat1_arg,
    const Tensor& mat2_arg,
    const float alpha,
    const float beta,
    const std::optional<Tensor>& bias = std::nullopt,
    Tensor* output_opt = nullptr) {
  api::Context* const context = api::context();
  TORCH_INTERNAL_ASSERT(can_run_float_buffer_bmm(mat1_arg, mat2_arg));

  const Tensor mat1 = mat1_arg.requires_grad() ? mat1_arg.detach() : mat1_arg;
  const Tensor mat2 = mat2_arg.requires_grad() ? mat2_arg.detach() : mat2_arg;
  vTensor& v_mat1 = convert(mat1);
  vTensor& v_mat2 = convert(mat2);

  const std::vector<int64_t> output_sizes{
      mat1.size(Layout::BatchMatrices::batch),
      mat1.size(Layout::BatchMatrices::height),
      mat2.size(Layout::BatchMatrices::width),
  };
  Tensor output_tensor = output_opt
      ? ensure_bmm_buffer_output_tensor(*output_opt, output_sizes, kFloat)
      : utils::mark_tensor_execution(
            convert(vTensor{
                context,
                output_sizes,
                api::kFloat,
                api::StorageType::BUFFER,
                api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
            }),
            api::ExecutionLayout::BUFFER_DIRECT);
  vTensor& v_output = convert(output_tensor);

  const struct {
    int32_t out_width;
    int32_t out_height;
    int32_t inner_dim;
    int32_t batch;
  } block{
      api::utils::safe_downcast<int32_t>(
          mat2.size(Layout::BatchMatrices::width)),
      api::utils::safe_downcast<int32_t>(
          mat1.size(Layout::BatchMatrices::height)),
      api::utils::safe_downcast<int32_t>(
          mat1.size(Layout::BatchMatrices::width)),
      api::utils::safe_downcast<int32_t>(
          mat1.size(Layout::BatchMatrices::batch)),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(
          mat2.size(Layout::BatchMatrices::width)),
      api::utils::safe_downcast<uint32_t>(
          mat1.size(Layout::BatchMatrices::height)),
      api::utils::safe_downcast<uint32_t>(
          mat1.size(Layout::BatchMatrices::batch)),
  };
  const api::utils::uvec3 local_size{16u, 4u, 1u};
  note_bmm_aggregate(
      mat1,
      mat2,
      output_tensor,
      v_mat1,
      v_mat2,
      v_output,
      bias,
      global_size,
      local_size);

  context->submit_compute_job(
      VK_KERNEL(bmm_buffer_float),
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      utils::make_buffer_compute_metadata_ubo(context, v_output).buffer(),
      v_mat1.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      utils::make_buffer_compute_metadata_ubo(context, v_mat1).buffer(),
      v_mat2.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      utils::make_buffer_compute_metadata_ubo(context, v_mat2).buffer(),
      params.buffer());

  Tensor output = output_tensor;
  if (alpha != 1.0f) {
    output = output.mul(alpha);
  }
  if (bias && bias->defined()) {
    Tensor bias_tensor = bias->is_vulkan() ? *bias : bias->vulkan();
    if (beta != 1.0f) {
      bias_tensor = bias_tensor.mul(beta);
    }
    output = output.add(bias_tensor);
  }
  if (output_opt && output.unsafeGetTensorImpl() != output_tensor.unsafeGetTensorImpl()) {
    *output_opt = output;
    output = *output_opt;
  }
  return output;
}

bool can_run_half_buffer_bmm(const Tensor& mat1, const Tensor& mat2) {
  return mat1.scalar_type() == kHalf && mat2.scalar_type() == kHalf &&
      mat1.dim() == 3 && mat2.dim() == 3 &&
      mat1.size(Layout::BatchMatrices::batch) ==
          mat2.size(Layout::BatchMatrices::batch) &&
      mat1.size(Layout::BatchMatrices::width) ==
          mat2.size(Layout::BatchMatrices::height);
}

Tensor run_half_buffer_bmm(
    const Tensor& mat1,
    const Tensor& mat2,
    const float alpha,
    const float beta,
    const std::optional<Tensor>& bias = std::nullopt) {
  const Tensor float_mat1 = widen_half_linear_tensor_to_float_buffer(mat1);
  const Tensor float_mat2 = widen_half_linear_tensor_to_float_buffer(mat2);
  const std::optional<Tensor> float_bias =
      upcast_half_linear_tensor_for_packing(bias);
  return run_float_buffer_bmm(
      float_mat1, float_mat2, alpha, beta, float_bias);
}

Tensor run_addmm_context_channel_packed_input(
    const Tensor& input_arg,
    const Tensor& input_2d,
    const vTensor& v_input,
    const LinearPackedRunState& packed_state,
    const float alpha,
    const float beta,
    const LinearPostOp post_op,
    Tensor* output_opt = nullptr) {
  api::Context* const context = api::context();
  const vTensor& packed_v_weight = packed_state.packed_v_weight;
  const vTensor& packed_v_bias = packed_state.packed_v_bias;
  const std::vector<int64_t>& unpacked_weight_sizes =
      packed_state.logical_weight_sizes;
  const bool bias_defined = packed_state.bias_defined;
  const std::vector<int64_t> output_sizes{
      input_2d.sizes()[Layout::Parameter::height],
      unpacked_weight_sizes[Layout::Parameter::width],
  };
  Tensor output_tensor = output_opt
      ? ensure_linear_output_tensor(
            *output_opt, output_sizes, convert_dtype(v_input.dtype()))
      : convert(vTensor{context, output_sizes, v_input.dtype()});
  vTensor& v_output = convert(output_tensor);

  api::UniformParamsBuffer params;
  api::ShaderInfo compute_shader;
  const int step_size =
      div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(4));
  const bool fuse_bias =
      bias_defined &&
      can_fuse_linear_bias(v_output, packed_v_bias, unpacked_weight_sizes);
  const bool fuse_gelu = fuse_bias && post_op == LinearPostOp::Gelu;
  const api::utils::ivec4 input_sizes =
      api::utils::make_ivec4_prepadded1(v_input.sizes());

  if (fuse_gelu) {
    const struct {
      uvec4 shader_extents_and_step;
      ivec4 input_sizes;
      uvec4 bias_extents;
      vec4 multipliers_and_gelu;
    } block_with_bias_gelu{
        {
            v_output.extents().data[0u],
            v_output.extents().data[1u],
            v_output.extents().data[2u],
            safe_downcast<uint32_t>(step_size),
        },
        input_sizes,
        {
            packed_v_bias.extents().data[0u],
            packed_v_bias.extents().data[1u],
            packed_v_bias.extents().data[2u],
            0u,
        },
        {alpha, beta, kGeluBeta, 0.0f},
    };
    params = api::UniformParamsBuffer(context, block_with_bias_gelu);
    compute_shader = VK_KERNEL(mm_bias_gelu_channel_packed_input);
  } else if (fuse_bias) {
    const struct {
      uvec4 shader_extents_and_step;
      ivec4 input_sizes;
      uvec4 bias_extents;
      vec2 multipliers;
    } block_with_bias{
        {
            v_output.extents().data[0u],
            v_output.extents().data[1u],
            v_output.extents().data[2u],
            safe_downcast<uint32_t>(step_size),
        },
        input_sizes,
        {
            packed_v_bias.extents().data[0u],
            packed_v_bias.extents().data[1u],
            packed_v_bias.extents().data[2u],
            0u,
        },
        {alpha, beta},
    };
    params = api::UniformParamsBuffer(context, block_with_bias);
    compute_shader = VK_KERNEL(mm_bias_channel_packed_input);
  } else {
    const struct {
      uvec4 shader_extents_and_step;
      ivec4 input_sizes;
    } block_no_bias{
        {
            v_output.extents().data[0u],
            v_output.extents().data[1u],
            v_output.extents().data[2u],
            safe_downcast<uint32_t>(step_size),
        },
        input_sizes,
    };
    params = api::UniformParamsBuffer(context, block_no_bias);
    compute_shader = VK_KERNEL(mm_channel_packed_input);
  }

  api::PipelineBarrier pipeline_barrier{};
  if (fuse_bias) {
    context->submit_compute_job(
        compute_shader,
        pipeline_barrier,
        {
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::width], INT64_C(4))),
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::height], INT64_C(4))),
            1,
        },
        {8, 8, 1},
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  } else {
    context->submit_compute_job(
        compute_shader,
        pipeline_barrier,
        {
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::width], INT64_C(4))),
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::height], INT64_C(4))),
            1,
        },
        {8, 8, 1},
        VK_NULL_HANDLE,
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());
  }

  Tensor output = output_tensor;
  if (!fuse_bias && alpha != 1.0f) {
    output = output.mul(alpha);
  }
  if (!fuse_bias && bias_defined) {
    output = output.add(convert(packed_v_bias).mul(beta));
  }
  if (post_op == LinearPostOp::Gelu && !fuse_gelu) {
    output = at::gelu(output, "none");
  }
  if (output_opt && output.unsafeGetTensorImpl() != output_tensor.unsafeGetTensorImpl()) {
    output = rebind_vulkan_output(*output_opt, output);
  }

  return reshape_linear_output_if_needed(output, input_arg);
}

vTensor pack_cpu_float_weight_using_height_packing(const Tensor& weight_arg) {
  TORCH_INTERNAL_ASSERT(weight_arg.is_cpu());
  TORCH_INTERNAL_ASSERT(weight_arg.scalar_type() == kFloat);
  TORCH_INTERNAL_ASSERT(weight_arg.dim() == 2);

  api::Context* const context = api::context();
  const Tensor weight = weight_arg.contiguous();
  const int64_t height = weight.size(Layout::Parameter::height);
  const int64_t width = weight.size(Layout::Parameter::width);

  vTensor v_weight{
      context,
      weight.sizes().vec(),
      convert_dtype(weight.scalar_type()),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
  };

  api::StorageBuffer staging(context, api::kFloat, v_weight.gpu_numel());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
    float* const dst = mapping.template data<float>();
    const float* const src = weight.const_data_ptr<float>();
    std::fill_n(dst, v_weight.gpu_numel(), 0.0f);

    const api::utils::uvec3 extents = v_weight.extents();
    const int64_t texel_width =
        static_cast<int64_t>(extents.data[0u]);
    const int64_t texel_height =
        static_cast<int64_t>(extents.data[1u]);
    const int64_t texel_depth =
        static_cast<int64_t>(extents.data[2u]);

    for (const auto z : c10::irange(texel_depth)) {
      for (const auto y : c10::irange(texel_height)) {
        const int64_t src_base_h = y * 4;
        for (const auto x : c10::irange(texel_width)) {
          const int64_t texel_base =
              (((z * texel_height) + y) * texel_width + x) * 4;
          for (const auto c : c10::irange(int64_t{4})) {
            const int64_t src_h = src_base_h + c;
            if (src_h < height && x < width) {
              dst[texel_base + c] = src[src_h * width + x];
            }
          }
        }
      }
    }
  }

  api::PipelineBarrier pipeline_barrier{};
  utils::copy_buffer_to_vtensor(staging.buffer(), v_weight, pipeline_barrier);
  return v_weight;
}

vTensor pack_inputs_using_width_packing(
    const Tensor& input_arg,
    const utils::VulkanPlanningRequest& input_request) {
  TORCH_INTERNAL_ASSERT(
      !input_arg.is_quantized(),
      "Vulkan Linear not usable! "
      "Reason: Input packing only supports non-quantized tensors.");
  TORCH_INTERNAL_ASSERT(
      input_arg.dim() == 2 || input_arg.dim() == 3,
      "Vulkan Linear not usable! "
      "Reason: Input packing only supports 2D or 3D tensors.");

  const Tensor input = utils::prepare_vulkan_execution_tensor(
      input_arg,
      utils::VulkanExecutionPlanKind::LinearPackedInput,
      input_request);

  vTensor v_input = convert(input);

  TORCH_CHECK(
      v_input.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      "After packing, the v_input must be in TENSOR_WIDTH_PACKED format");

  return v_input;
}

vTensor pack_inputs_using_width_packing(const Tensor& input_arg) {
  return pack_inputs_using_width_packing(
      input_arg,
      utils::make_vulkan_tensor_linear_request(
          input_arg, utils::VulkanTensorRole::Input));
}

vTensor pack_weights_using_height_packing(const Tensor& weight_arg) {
  // Only non-batch, non-quantized tensors are supported
  TORCH_INTERNAL_ASSERT(
      !weight_arg.is_quantized(),
      "Vulkan Linear not usable! "
      "Reason: Weight packing only supports non-quantized tensors.");
  TORCH_INTERNAL_ASSERT(
      weight_arg.dim() == 2 || weight_arg.dim() == 3,
      "Vulkan Linear not usable! "
      "Reason: Weight packing only supports 2D or 3D tensors.");

  if (weight_arg.is_cpu() && weight_arg.scalar_type() == kFloat &&
      weight_arg.dim() == 2) {
    return pack_cpu_float_weight_using_height_packing(weight_arg);
  }

  const Tensor weight = utils::prepare_vulkan_execution_tensor(
      weight_arg,
      utils::VulkanExecutionPlanKind::LinearPackedWeight,
      utils::make_vulkan_linear_request(utils::VulkanTensorRole::Weight));

  vTensor v_weight = convert(weight);

  TORCH_CHECK(
      v_weight.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
      "After packing, the v_weight must be in TENSOR_HEIGHT_PACKED format");

  return v_weight;
}

vTensor pack_weights(const Tensor& weight_arg, const bool use_batch = false) {
  if (!weight_arg.is_quantized()) {
    return pack_weights_using_height_packing(weight_arg);
  }

  TORCH_CHECK(
      weight_arg.is_quantized(), "Only quantized weights logic after here");

  // Rest of the logic are either quantized or batched.

  api::Context* const context = api::context();

  const Tensor weight = weight_arg.contiguous();
  const IntArrayRef w_sizes = weight.sizes();
  if (use_batch) {
    TORCH_CHECK(
        w_sizes.size() == 3,
        "Vulkan Linear not usable! "
        "Reason: Unable to perform weight packing with batch; the input tensor of a batch of matrices should contain 3 dimensions: batch, height, width.");
  }
  /* Source */
  int64_t src_kb_sz = 0;
  int64_t src_kw_sz = 0;
  int64_t src_kh_sz = 0;
  /* Destination */
  int64_t dst_kb_sz = 0;
  int64_t dst_kw_sz = 0;
  int64_t dst_kh_sz = 0;
  std::vector<int64_t> dst_vtensor_sizes;
  /* Source */
  src_kb_sz = use_batch ? w_sizes[Layout::BatchMatrices::batch] : 1;
  src_kw_sz = use_batch ? w_sizes[Layout::BatchMatrices::width]
                        : w_sizes[Layout::Parameter::width];
  src_kh_sz = use_batch ? w_sizes[Layout::BatchMatrices::height]
                        : w_sizes[Layout::Parameter::height];

  /* Destination */
  dst_kb_sz = src_kb_sz;
  dst_kw_sz = div_up(src_kw_sz, INT64_C(2));
  dst_kh_sz = div_up(src_kh_sz, INT64_C(2));
  dst_vtensor_sizes = {
      dst_kb_sz,
      4,
      dst_kh_sz,
      dst_kw_sz,
  };

  vTensor v_weight{
      context, dst_vtensor_sizes, convert_dtype(weight_arg.scalar_type())};

  v_weight.set_is_quantized();
  v_weight.set_scale(weight_arg.q_scale());
  v_weight.set_zero_point(weight_arg.q_zero_point());

  stage_pack_weights<int8_t>(
      context,
      v_weight,
      weight,
      src_kb_sz,
      src_kh_sz,
      src_kw_sz,
      dst_kh_sz,
      dst_kw_sz);
  return v_weight;
}

vTensor pack_biases(
    const Tensor& weight_arg,
    const std::optional<Tensor>& bias_arg,
    const bool use_batch = false) {
  if (has_bias(bias_arg)) {
    Tensor bias = utils::prepare_vulkan_execution_tensor(
        *bias_arg,
        utils::VulkanExecutionPlanKind::LinearPackedBias,
        utils::make_vulkan_linear_request(utils::VulkanTensorRole::Bias));
    return convert(bias);
  } else {
    return convert(at::zeros(
        {1}, weight_arg.options().device(weight_arg.device()).dtype(at::kFloat)));
  }
}

// Old version of pack_biases that fixes issues with quantization and to be
// removed in the future.
vTensor pack_biases_quantized_weights(
    const Tensor& weight_arg,
    const std::optional<Tensor>& bias_arg,
    const bool use_batch = false) {
  TORCH_CHECK(
      weight_arg.is_quantized(),
      "pack_biases_quantized to be used only when using quantized linear ops");

  if (has_bias(bias_arg) && bias_arg->is_vulkan()) {
    Tensor bias = utils::prepare_vulkan_execution_tensor(
        *bias_arg,
        utils::VulkanExecutionPlanKind::TextureComputeInput,
        utils::make_vulkan_linear_request(utils::VulkanTensorRole::Bias));
    return convert(bias);
  }

  api::Context* const context = api::context();

  if (has_bias(bias_arg)) {
    const Tensor bias = bias_arg->contiguous();
    const IntArrayRef b_sizes = bias.sizes();
    const float* const src_bias_ptr = bias.const_data_ptr<float>();

    /* Source */
    int64_t src_kb_sz = 0;
    int64_t src_kw_sz = 0;
    int64_t src_kh_sz = 0;
    if (use_batch) {
      if (bias.sizes().size() == 3) {
        src_kb_sz = b_sizes[Layout::BatchMatrices::batch];
        src_kw_sz = b_sizes[Layout::BatchMatrices::width];
        src_kh_sz = b_sizes[Layout::BatchMatrices::height];
      } else if (bias.sizes().size() == 2) {
        // skip batch dim for broadcasting; index -1
        src_kb_sz = 1;
        src_kw_sz = b_sizes[Layout::BatchMatrices::height];
        src_kh_sz = b_sizes[Layout::BatchMatrices::batch];
      } else {
        // skip batch & height dim for broadcasting; index -2
        src_kb_sz = 1;
        src_kw_sz = b_sizes[Layout::BatchMatrices::batch];
        src_kh_sz = 1;
      }
    } else {
      src_kb_sz = 1;
      if (bias.sizes().size() == 2) {
        src_kw_sz = b_sizes[Layout::Parameter::width];
        src_kh_sz = b_sizes[Layout::Parameter::height];
      } else {
        src_kw_sz = b_sizes[Layout::Parameter::height];
        src_kh_sz = 1;
      }
    }
    const int64_t src_matrix_sz = src_kw_sz * src_kh_sz;

    /* Destination */
    const int64_t dst_kw_sz = div_up(src_kw_sz, INT64_C(2));
    const int64_t dst_kh_sz = div_up(src_kh_sz, INT64_C(2));
    const int64_t dst_plane_sz = dst_kw_sz * dst_kh_sz;
    const int64_t dst_matrix_sz = dst_plane_sz * 4;

    vTensor v_bias{
        context,
        {
            src_kb_sz,
            4,
            dst_kh_sz,
            dst_kw_sz,
        },
        convert_dtype(bias_arg->scalar_type()),
    };

    api::StorageBuffer staging(
        context, api::ScalarType::Float, v_bias.gpu_numel());
    {
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

      float* dst_bias_ptr = mapping.template data<float>();

      memset(dst_bias_ptr, 0, v_bias.nbytes());

      for (const auto src_b : c10::irange(src_kb_sz)) {
        for (const auto src_h : c10::irange(src_kh_sz == 1 ? 2 : src_kh_sz)) {
          for (const auto src_w :
               c10::irange((use_batch && src_kw_sz == 1) ? 2 : src_kw_sz)) {
            int64_t dst_plane = 2 * (src_h % 2) + (src_w % 2);
            int64_t dst_index = (src_h / 2) * dst_kw_sz + (src_w / 2);
            memcpy(
                dst_bias_ptr + src_b * dst_matrix_sz +
                    dst_plane * dst_plane_sz + dst_index,
                src_bias_ptr + src_b * src_matrix_sz +
                    (src_kh_sz == 1 ? 0 : src_h * src_kw_sz) +
                    ((use_batch && src_kw_sz == 1) ? 0 : src_w),
                sizeof(float));
          }
        }
      }
    }
    utils::pack_staging_to_vtensor(staging.buffer(), v_bias);

    return v_bias;
  } else {
    vTensor v_bias{
        api::context(),
        {1},
        convert_dtype(weight_arg.scalar_type()),
    };

    api::StorageBuffer staging(
        context, api::ScalarType::Float, v_bias.gpu_numel());
    {
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

      float* data_ptr = mapping.template data<float>();

      memset(
          data_ptr,
          // 2's complement integers and IEEE-754 floating point numbers both
          // have identical bit representations for 0, so can use memset which
          // only accepts uint8_t parameter.
          0,
          v_bias.nbytes());
    }
    utils::pack_staging_to_vtensor(staging.buffer(), v_bias);

    return v_bias;
  }
}

bool available_check_with_batch(
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  const bool weight_available = (3 == weight.ndimension()) &&
      (weight.size(Layout::BatchMatrices::batch) > 0) &&
      (weight.size(Layout::BatchMatrices::height) > 0) &&
      (weight.size(Layout::BatchMatrices::width) > 0) &&
      ((weight.device().is_cpu()) ||
       (c10::DeviceType::Vulkan == weight.device().type())) &&
      (kFloat == weight.scalar_type() || kHalf == weight.scalar_type());
  if (!weight_available) {
    return false;
  }

  if (!bias || !bias->defined()) {
    // no need to check bias since it is not used.
    return true;
  }

  bool bias_available = true;
  bias_available &= (bias->ndimension() > 0);
  bias_available &=
      ((bias->device().is_cpu()) ||
       (c10::DeviceType::Vulkan == bias->device().type()));
  bias_available &=
      (kFloat == bias->scalar_type() || kHalf == bias->scalar_type());
  // Only check the consistency of batch and width dimension. The height
  // dimension consistency is unchecked, due to the 2nd input which determines
  // the height is not passed into LinearPackedContext.
  if (bias->ndimension() == 3) {
    bias_available &=
        (bias->size(Layout::BatchMatrices::width) ==
             weight.size(Layout::BatchMatrices::width) ||
         bias->size(Layout::BatchMatrices::width) == 1);
    bias_available &=
        (bias->size(Layout::BatchMatrices::batch) ==
             weight.size(Layout::BatchMatrices::batch) ||
         bias->size(Layout::BatchMatrices::batch) == 1);
  } else if (bias->ndimension() == 2) {
    // skip batch dim for broadcasting; index -1
    bias_available &=
        (bias->size(Layout::BatchMatrices::height) ==
             weight.size(Layout::BatchMatrices::width) ||
         bias->size(Layout::BatchMatrices::height) == 1);
  } else {
    // skip batch & height dim for broadcasting; index -2
    bias_available &=
        (bias->size(Layout::BatchMatrices::batch) ==
             weight.size(Layout::BatchMatrices::width) ||
         bias->size(Layout::BatchMatrices::batch) == 1);
  }
  return bias_available;
}

bool available(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const bool use_batch = false) {
  if (!api::available()) {
    return false;
  }

  if (use_batch) {
    return available_check_with_batch(weight, bias);
  }

  const bool weight_available = (2 == weight.ndimension()) &&
      (weight.size(Layout::Parameter::height) > 0) &&
      (weight.size(Layout::Parameter::width) > 0) &&
      ((weight.device().is_cpu()) ||
       (c10::DeviceType::Vulkan == weight.device().type())) &&
      (kFloat == weight.scalar_type() || kHalf == weight.scalar_type() ||
       kBFloat16 == weight.scalar_type() ||
       kQInt8 == weight.scalar_type());
  if (!weight_available) {
    return false;
  }

  const bool bias_available =
      ((bias && bias.has_value() && bias->defined())
           ? ((bias->ndimension() > 0) &&
              ((bias->device().is_cpu()) ||
               (c10::DeviceType::Vulkan == bias->device().type())) &&
              (kFloat == bias->scalar_type() ||
               kHalf == bias->scalar_type() ||
               kBFloat16 == bias->scalar_type()) &&
              ((bias->ndimension() > 1)
                   ? (bias->size(Layout::Parameter::width) ==
                      weight.size(Layout::Parameter::width))
                   : true))
           : true);
  return bias_available;
}

bool usable_check_with_batch(
    const Tensor& input,
    const IntArrayRef unpacked_weight_sizes) {
  return (3 == input.ndimension()) &&
      (c10::DeviceType::Vulkan == input.device().type()) &&
      (kFloat == input.scalar_type() || kHalf == input.scalar_type()) &&
      (input.size(Layout::BatchMatrices::width) ==
       unpacked_weight_sizes[Layout::BatchMatrices::height]) &&
      (input.size(Layout::BatchMatrices::batch) ==
       unpacked_weight_sizes[Layout::BatchMatrices::batch]);
}

bool usable(
    const Tensor& input,
    const IntArrayRef unpacked_weight_sizes,
    const bool use_batch = false) {
  if (use_batch) {
    return usable_check_with_batch(input, unpacked_weight_sizes);
  }
  const auto v_input = convert(input);
  return (2 == input.ndimension()) &&
      (c10::DeviceType::Vulkan == input.device().type()) &&
      ((kFloat == input.scalar_type()) || (kHalf == input.scalar_type()) ||
       (kBFloat16 == input.scalar_type()) ||
       (v_input.is_quantized() &&
        (kQUInt8 == input.scalar_type() || kQInt8 == input.scalar_type()))) &&
      (input.size(Layout::Parameter::width) ==
       unpacked_weight_sizes[Layout::Parameter::height]);
}

static Tensor reshape_to_2d(const Tensor& input_arg) {
  TORCH_CHECK(
      input_arg.dim() >= 1,
      "Vulkan Linear op only supports input tensor with dim >= 1");

  Tensor reshape_input = input_arg;
  if (input_arg.is_vulkan() && c10::InferenceMode::is_enabled()) {
    const vTensor& v_input = convert(input_arg);
    const bool needs_materialization =
        v_input.storage_type() == api::StorageType::BUFFER &&
        !v_input.has_direct_buffer_layout();
    if (needs_materialization) {
      reshape_input =
          utils::contiguous_inference(input_arg, c10::MemoryFormat::Contiguous);
    }
  }

  if (reshape_input.dim() == 1) {
    return reshape_input.unsqueeze(0);
  }
  const IntArrayRef input_sizes = reshape_input.sizes();
  const auto d =
      c10::multiply_integers(input_sizes.cbegin(), input_sizes.end() - 1);
  return utils::reshape_inference(
      reshape_input, {d, reshape_input.size(-1)});
}

bool can_run_bfloat16_buffer_linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  if (
      input.device().type() != c10::DeviceType::Vulkan ||
      weight.device().type() != c10::DeviceType::Vulkan ||
      input.scalar_type() != kBFloat16 ||
      weight.scalar_type() != kBFloat16 ||
      input.dim() != 2 ||
      weight.dim() != 2 ||
      input.requires_grad() ||
      weight.requires_grad() ||
      input.size(Layout::Parameter::width) !=
          weight.size(Layout::Parameter::width)) {
    return false;
  }

  vTensor v_input = convert(input);
  vTensor v_weight = convert(weight);
  const bool valid_layout =
      v_input.storage_type() == api::StorageType::BUFFER &&
      v_weight.storage_type() == api::StorageType::BUFFER &&
      v_input.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
      v_weight.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
      utils::supports_buffer_view_fast_path(v_input) &&
      utils::supports_buffer_view_fast_path(v_weight);
  if (
      !valid_layout) {
    return false;
  }

  if (bias && bias->defined()) {
    if (
        bias->device().type() != c10::DeviceType::Vulkan ||
        bias->dim() > 2 ||
        bias->requires_grad()) {
      return false;
    }

    if (convert(*bias).storage_type() != api::StorageType::BUFFER) {
      return false;
    }

    if (bias->scalar_type() != kBFloat16 && bias->scalar_type() != kFloat) {
      return false;
    }

    if (!utils::supports_buffer_view_fast_path(convert(*bias))) {
      return false;
    }
  }

  return true;
}

Tensor run_bfloat16_buffer_linear(
    const Tensor& input_arg,
    const Tensor& input_compute_arg,
    const Tensor& weight_arg,
    const std::optional<Tensor>& bias_arg,
    const LinearPostOp post_op = LinearPostOp::None,
    Tensor* output_opt = nullptr) {
  api::AllocationScope allocation_scope("linear.bf16_buffer");
  api::Context* const context = api::context();

  const Tensor input_compute_arg_2d = input_compute_arg.dim() == 2
      ? input_compute_arg
      : reshape_to_2d(input_compute_arg);
  const Tensor input = input_compute_arg_2d.is_vulkan()
      ? input_compute_arg_2d
      : input_compute_arg_2d.vulkan();
  const Tensor weight = weight_arg.is_vulkan() ? weight_arg : weight_arg.vulkan();

  TORCH_INTERNAL_ASSERT(can_run_bfloat16_buffer_linear(input, weight, bias_arg));

  vTensor v_input = convert(input);
  vTensor v_weight = convert(weight);
  constexpr int64_t kCoopTileM = 16;
  constexpr int64_t kCoopTileN = 16;
  constexpr int64_t kCoopTileK = 16;
  VulkanLinearPlanDecision decision;
  decision.m = input_compute_arg_2d.size(Layout::Parameter::height);
  decision.k = input_compute_arg_2d.size(Layout::Parameter::width);
  decision.n = weight.size(Layout::Parameter::height);
  decision.tile_m = kCoopTileM;
  decision.tile_k = kCoopTileK;
  decision.tile_n = kCoopTileN;
  decision.input_vulkan = input.is_vulkan();
  decision.weight_packed = weight.is_vulkan();
  decision.input_direct_buffer = v_input.has_direct_buffer_layout();
  decision.has_post_op = post_op != LinearPostOp::None;
  const std::vector<int64_t> output_sizes{
      input_compute_arg_2d.sizes()[Layout::Parameter::height],
      weight.sizes()[Layout::Parameter::height],
  };
  Tensor output_tensor = output_opt
      ? ensure_linear_buffer_output_tensor(*output_opt, output_sizes, kFloat)
      : utils::mark_tensor_execution(
            convert(vTensor{
                context,
                output_sizes,
                api::kFloat,
                api::StorageType::BUFFER,
                api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
            }),
            api::ExecutionLayout::BUFFER_DIRECT);
  vTensor& v_output = convert(output_tensor);
  decision.output_direct_buffer = v_output.has_direct_buffer_layout();

  const struct {
    int32_t out_width;
    int32_t out_height;
    int32_t inner_dim;
    int32_t row_offset;
  } block{
      api::utils::safe_downcast<int32_t>(weight.size(Layout::Parameter::height)),
      api::utils::safe_downcast<int32_t>(
          input_compute_arg_2d.size(Layout::Parameter::height)),
      api::utils::safe_downcast<int32_t>(
          input_compute_arg_2d.size(Layout::Parameter::width)),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(weight.size(Layout::Parameter::height)),
      api::utils::safe_downcast<uint32_t>(
          input_compute_arg_2d.size(Layout::Parameter::height)),
      1u,
  };

  const bool m_aligned = is_aligned_i64(decision.m, kCoopTileM);
  const bool k_aligned = is_aligned_i64(decision.k, kCoopTileK);
  const bool n_aligned = is_aligned_i64(decision.n, kCoopTileN);
  decision.m_tail = !m_aligned;
  decision.k_tail = !k_aligned;
  decision.n_tail = !n_aligned;
  bool can_use_cooperative_matrix_kernel = false;
  if (api::Adapter* const adapter = context->adapter_ptr()) {
    can_use_cooperative_matrix_kernel =
        adapter->has_cooperative_matrix() &&
        adapter->has_compute_full_subgroups() &&
        adapter->supports_required_subgroup_size(
            VK_SHADER_STAGE_COMPUTE_BIT, 32u);
  }

  if (can_use_cooperative_matrix_kernel && k_aligned && n_aligned) {
    api::ShaderInfo coop_shader = VK_KERNEL(mm_buffer_bfloat16_cooperative_matrix);
    coop_shader.required_subgroup_size = 32u;
    coop_shader.require_full_subgroups = true;
    const api::utils::uvec3 coop_local_work_group{32u, 1u, 1u};
    const uint32_t coop_global_width =
        api::utils::safe_downcast<uint32_t>(decision.n) *
        coop_local_work_group.data[0u];
    if (m_aligned) {
      decision.selected = VulkanLinearFastPath::BFloat16CooperativeMatrix;
      utils::log_vulkan_op_hit(
          "aten::linear.buffer_bfloat16_cooperative_matrix");
      const api::utils::uvec3 coop_global_size{
          coop_global_width,
          api::utils::safe_downcast<uint32_t>(decision.m),
          1u,
      };
      context->submit_compute_job(
          coop_shader,
          pipeline_barrier,
          coop_global_size,
          coop_local_work_group,
          VK_NULL_HANDLE,
          v_output.buffer(
              pipeline_barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::WRITE),
          v_output.buffer_metadata(),
          v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
          v_input.buffer_metadata(),
          v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
          v_weight.buffer_metadata(),
          params.buffer());
    } else {
      decision.selected = VulkanLinearFastPath::BFloat16CooperativeMatrixTailM;
      utils::log_vulkan_op_hit(
          "aten::linear.buffer_bfloat16_cooperative_matrix_tail_m");
      const int64_t aligned_m = (decision.m / kCoopTileM) * kCoopTileM;
      if (aligned_m > 0) {
        const api::utils::uvec3 prefix_global_size{
            coop_global_width,
            api::utils::safe_downcast<uint32_t>(aligned_m),
            1u,
        };
        context->submit_compute_job(
            coop_shader,
            pipeline_barrier,
            prefix_global_size,
            coop_local_work_group,
            VK_NULL_HANDLE,
            v_output.buffer(
                pipeline_barrier,
                api::PipelineStage::COMPUTE,
                api::MemoryAccessType::WRITE),
            v_output.buffer_metadata(),
            v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
            v_input.buffer_metadata(),
            v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
            v_weight.buffer_metadata(),
            params.buffer());
      }
      const decltype(block) tail_block{
          block.out_width,
          block.out_height,
          block.inner_dim,
          api::utils::safe_downcast<int32_t>(aligned_m),
      };
      api::UniformParamsBuffer tail_params(context, tail_block);
      const api::utils::uvec3 tail_global_size{
          api::utils::safe_downcast<uint32_t>(decision.n),
          api::utils::safe_downcast<uint32_t>(decision.m - aligned_m),
          1u,
      };
      context->submit_compute_job(
          VK_KERNEL(mm_buffer_bfloat16_tail_m),
          pipeline_barrier,
          tail_global_size,
          adaptive_work_group_size(tail_global_size),
          VK_NULL_HANDLE,
          v_output.buffer(
              pipeline_barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::WRITE),
          v_output.buffer_metadata(),
          v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
          v_input.buffer_metadata(),
          v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
          v_weight.buffer_metadata(),
          tail_params.buffer());
    }
  } else {
    decision.selected = VulkanLinearFastPath::BFloat16Buffer;
    if (!can_use_cooperative_matrix_kernel) {
      decision.reject = VulkanLinearRejectReason::CapabilityMissing;
    } else if (!k_aligned) {
      decision.reject = VulkanLinearRejectReason::KNotAligned;
    } else if (!n_aligned) {
      decision.reject = VulkanLinearRejectReason::NNotAligned;
    }
    utils::log_vulkan_op_hit("aten::linear.buffer_bfloat16");
    context->submit_compute_job(
        VK_KERNEL(mm_buffer_bfloat16),
        pipeline_barrier,
        global_size,
        adaptive_work_group_size(global_size),
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_output.buffer_metadata(),
        v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_input.buffer_metadata(),
        v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        v_weight.buffer_metadata(),
        params.buffer());
  }
  note_linear_plan_decision(decision);
  append_vulkan_linear_plan_log(decision, "aten::linear.bfloat16_buffer");

  Tensor output = output_tensor;
  std::optional<Tensor> bias = bias_arg;
  if (bias && bias->defined()) {
    if (
        !bias->is_vulkan() || convert(*bias).storage_type() != api::StorageType::BUFFER ||
        !utils::supports_buffer_view_fast_path(convert(*bias))) {
      bias = utils::prepare_optional_vulkan_execution_tensor(
          bias_arg,
          utils::VulkanExecutionPlanKind::LinearBiasSource,
          utils::make_vulkan_linear_request(utils::VulkanTensorRole::Bias));
    }
    if (bias && bias->defined() && !bias->is_vulkan()) {
      *bias = bias->vulkan();
    }
    if (bias && bias->defined()) {
      output = output.add(*bias);
    }
  }
  if (post_op == LinearPostOp::Gelu) {
    output = at::gelu(output, "none");
  }
  if (output_opt &&
      output.unsafeGetTensorImpl() != output_tensor.unsafeGetTensorImpl()) {
    *output_opt = output;
    output = *output_opt;
  }
  return reshape_linear_output_if_needed(output, input_arg);
}

Tensor run_quantized_addmm_context(
    const Tensor& input_arg,
    const float alpha,
    const float beta,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context,
    double output_scale,
    int64_t output_zero_point) {
  api::Context* const context = api::context();

  const Tensor input_arg_2d =
      input_arg.dim() == 2 ? input_arg : reshape_to_2d(input_arg);
  const Tensor input =
      input_arg_2d.is_vulkan() ? input_arg_2d : input_arg_2d.vulkan();
  const vTensor& v_input = convert(input);
  const LinearPackedRunState packed_state =
      get_linear_packed_run_state(linear_context);
  const vTensor& packed_v_weight = packed_state.packed_v_weight;
  const vTensor& packed_v_bias = packed_state.packed_v_bias;
  const std::vector<int64_t>& unpacked_weight_sizes =
      packed_state.logical_weight_sizes;
  const bool bias_defined = packed_state.bias_defined;

  TORCH_CHECK(
      usable(input, unpacked_weight_sizes),
      "Vulkan Linear not usable! "
      "Reason: The provided input tensor is either invalid on its own, or its "
      "combination with the provided weight and bias tensors are unsupported by "
      "Vulkan impl.");

  TORCH_CHECK(
      (packed_v_weight.is_quantized() && v_input.is_quantized()),
      "run_quantized_addmm_context called for quantized version with unquantized input");

  vTensor v_output{
      context,
      {
          input_arg_2d.sizes()[Layout::Parameter::height],
          unpacked_weight_sizes[Layout::Parameter::width],
      },
      v_input.dtype(),
  };

  v_output.set_is_quantized();
  v_output.set_scale(output_scale);
  v_output.set_zero_point(output_zero_point);

  if (bias_defined) {
    api::UniformParamsBuffer params;
    api::ShaderInfo compute_shader;
    compute_shader = (kQInt8 == input_arg.scalar_type())
        ? VK_KERNEL(quantized_addmm_qint8)
        : VK_KERNEL(quantized_addmm_quint8);
    const struct {
      uvec3 size;
      int32_t K;
      uvec3 um1_size;
      int32_t K1;
      uvec3 um2_size;
      int32_t K2;
      uvec3 ut_size;
      int32_t K3;
      vec2 multiplier;
      vec2 input_scales;
      float out_scale;
      float _1;
      ivec2 input_zero_points;
      int32_t out_zero_point;
      int32_t _2;
    } block{
        v_output.extents(),
        safe_downcast<int32_t>(
            div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(2))),
        v_input.extents(),
        0u,
        packed_v_weight.extents(),
        0u,
        packed_v_bias.extents(),
        0u,
        {
            alpha,
            beta,
        },
        {
            safe_downcast<float>(v_input.get_scale()),
            safe_downcast<float>(packed_v_weight.get_scale()),
        },
        safe_downcast<float>(output_scale),
        0.0f,
        {
            safe_downcast<int32_t>(v_input.get_zero_point()),
            safe_downcast<int32_t>(packed_v_weight.get_zero_point()),
        },
        safe_downcast<int32_t>(output_zero_point),
        0u,
    };
    params = api::UniformParamsBuffer(context, block);

    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        // shader descriptor
        compute_shader,
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {
            safe_downcast<uint32_t>(
                div_up(v_output.sizes()[Layout::Parameter::width], INT64_C(2))),
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::height], INT64_C(2))),
            1,
        },
        // local work group size
        {8, 8, 1},
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());

  } else { // no bias
    api::UniformParamsBuffer params;
    api::ShaderInfo compute_shader;
    const struct {
      uvec3 size;
      int32_t K;
      uvec3 um1_size;
      int32_t K1;
      uvec3 um2_size;
      int32_t K2;
      vec2 input_scales;
      float out_scale;
      float _1;
      ivec2 input_zero_points;
      int32_t out_zero_point;
      int32_t _2;
    } block_no_bias{
        v_output.extents(),
        safe_downcast<int32_t>(
            div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(2))),
        v_input.extents(),
        0u,
        packed_v_weight.extents(),
        0u,
        {
            safe_downcast<float>(v_input.get_scale()),
            safe_downcast<float>(packed_v_weight.get_scale()),
        },
        safe_downcast<float>(output_scale),
        0.0f,
        {
            safe_downcast<int32_t>(v_input.get_zero_point()),
            safe_downcast<int32_t>(packed_v_weight.get_zero_point()),
        },
        safe_downcast<int32_t>(output_zero_point),
        0u,
    };
    params = api::UniformParamsBuffer(context, block_no_bias);
    compute_shader = (kQInt8 == input_arg.scalar_type())
        ? VK_KERNEL(quantized_mm_qint8)
        : VK_KERNEL(quantized_mm_quint8);

    api::PipelineBarrier pipeline_barrier{};

    context->submit_compute_job(
        // shader descriptor
        compute_shader,
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {
            safe_downcast<uint32_t>(
                div_up(v_output.sizes()[Layout::Parameter::width], INT64_C(2))),
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::height], INT64_C(2))),
            1,
        },
        // local work group size
        {8, 8, 1},
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());
  }
  Tensor output = convert(v_output);
  if (input_arg.dim() == 2) {
    return output;
  } else {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(std::max<int64_t>(0, input_arg.dim())));
    for (const auto i : c10::irange(input_arg.dim() - 1)) {
      shape.emplace_back(input_arg.size(i));
    }
    shape.emplace_back(output.size(-1));
    Tensor reshaped_output = utils::reshape_inference(output, shape);
    if (c10::InferenceMode::is_enabled()) {
      reshaped_output = reshaped_output.clone();
    }
    return reshaped_output;
  }
}

Tensor run_addmm_context(
    const Tensor& input_arg,
    const float alpha,
    const float beta,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context,
    bool quantized,
  double output_scale,
  int64_t output_zero_point,
  const LinearPostOp post_op = LinearPostOp::None,
  Tensor* output_opt = nullptr) {
  log_linear_context_checkpoint(
      "entry", input_arg, post_op, quantized);
  const Tensor input_for_compute =
      materialize_deferred_add_layer_norm_candidate_if_needed(input_arg);
  log_linear_context_checkpoint(
      "after_deferred_materialize",
      input_for_compute,
      post_op,
      quantized);
  const auto input_request =
      utils::make_vulkan_tensor_linear_request(
          input_for_compute, utils::VulkanTensorRole::Input);
  api::AllocationScope allocation_scope(
      utils::resolve_vulkan_linear_runtime_label(
          linear_context ? linear_context->allocation_label() : std::string(),
          "linear"));
  if (quantized) {
    return run_quantized_addmm_context(
        input_for_compute,
        alpha,
        beta,
        linear_context,
        output_scale,
        output_zero_point);
  }

  api::Context* const context = api::context();
  utils::prime_labeled_scratch_arena_for_request(
      input_for_compute,
      input_request,
      linear_runtime_scratch_bytes(input_for_compute),
      "linear_decode");
  const auto runtime_policy = utils::build_vulkan_runtime_policy(input_request);
  if (
      runtime_policy.request.model_domain != utils::VulkanModelDomain::Generic ||
      runtime_policy.request.execution_phase !=
          utils::VulkanExecutionPhase::None) {
    log_linear_kernel_family_choice(runtime_policy);
  }

  const Tensor source_input_arg =
      input_for_compute.is_vulkan() ? input_for_compute
                                    : input_for_compute.vulkan();
  log_linear_context_checkpoint(
      "source_input", source_input_arg, post_op, quantized);
  const Tensor compute_input_arg = utils::prepare_vulkan_execution_tensor(
      source_input_arg,
      utils::VulkanExecutionPlanKind::LinearInputSource,
      input_request);
  log_linear_context_checkpoint(
      "prepared_input", compute_input_arg, post_op, quantized);
  const Tensor input_arg_2d =
      compute_input_arg.dim() == 2 ? compute_input_arg
                                   : reshape_to_2d(compute_input_arg);
  log_linear_context_checkpoint(
      "input_2d", input_arg_2d, post_op, quantized);
  const Tensor input =
      input_arg_2d.is_vulkan() ? input_arg_2d : input_arg_2d.vulkan();
  log_linear_context_checkpoint(
      "input_ready", input, post_op, quantized);
  const LinearPackedRunState packed_state =
      get_linear_packed_run_state(linear_context);
  utils::log_vulkan_op_hit(
      std::string("aten::linear.packed_state_ready weight=") +
      format_linear_sizes(packed_state.logical_weight_sizes) +
      " bias=" + (packed_state.bias_defined ? std::string("1")
                                            : std::string("0")));
  const vTensor& packed_v_weight = packed_state.packed_v_weight;
  const vTensor& packed_v_bias = packed_state.packed_v_bias;
  const std::vector<int64_t>& unpacked_weight_sizes =
      packed_state.logical_weight_sizes;
  const bool bias_defined = packed_state.bias_defined;
  const vTensor& source_v_input = convert(input);

  TORCH_CHECK(
      usable(input, unpacked_weight_sizes),
      "Vulkan Linear not usable! "
      "Reason: The provided input tensor is either invalid on its own, or its "
      "combination with the provided weight and bias tensors are unsupported by "
      "Vulkan impl.");

  if (
      packed_v_weight.storage_type() == api::StorageType::BUFFER &&
      api::uses_buffer_execution(packed_state.packed_weight.execution_layout())) {
    const std::optional<Tensor> packed_bias_tensor = bias_defined
        ? std::optional<Tensor>(packed_state.packed_weight.bias())
        : std::nullopt;
    Tensor buffer_input = input.requires_grad() ? input.detach() : input;
    if (
        buffer_input.scalar_type() == kFloat &&
        packed_state.packed_weight.weight().scalar_type() == kBFloat16 &&
        source_input_arg.scalar_type() == kBFloat16) {
      const Tensor source_input_2d = source_input_arg.dim() == 2
          ? source_input_arg
          : reshape_to_2d(source_input_arg);
      buffer_input = utils::mark_tensor_execution(
          utils::ensure_buffer_storage(
              source_input_2d, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
          api::ExecutionLayout::BUFFER_DIRECT);
      utils::log_vulkan_op_hit("aten::linear.buffer_bfloat16_preserve_input");
    }
    if (buffer_input.unsafeGetTensorImpl() != input.unsafeGetTensorImpl()) {
      utils::log_vulkan_op_hit("aten::linear.buffer_forward_detach");
    }
    if (
        buffer_input.is_vulkan() &&
        convert(buffer_input).storage_type() == api::StorageType::BUFFER &&
        convert(buffer_input).gpu_memory_layout() ==
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED) {
      buffer_input = utils::mark_tensor_execution(
          buffer_input,
          utils::resolve_buffer_execution_layout(convert(buffer_input)));
    }
    log_linear_context_checkpoint(
        "buffer_input_marked", buffer_input, post_op, quantized);
    if (
        !can_run_float_buffer_linear(
            buffer_input,
            packed_state.packed_weight.weight(),
            packed_bias_tensor) &&
        buffer_input.scalar_type() == kFloat) {
      buffer_input = utils::mark_tensor_execution(
          utils::ensure_buffer_storage(
              buffer_input, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
          api::ExecutionLayout::BUFFER_DIRECT);
    }
    if (
        !can_run_bfloat16_buffer_linear(
            buffer_input,
            packed_state.packed_weight.weight(),
            packed_bias_tensor) &&
        buffer_input.scalar_type() == kBFloat16) {
      buffer_input = utils::mark_tensor_execution(
          utils::ensure_buffer_storage(
              buffer_input, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
          api::ExecutionLayout::BUFFER_DIRECT);
    }
    log_linear_context_checkpoint(
        "buffer_input_supported", buffer_input, post_op, quantized);
    if (can_run_bfloat16_buffer_linear(
            buffer_input,
            packed_state.packed_weight.weight(),
            packed_bias_tensor)) {
      return run_bfloat16_buffer_linear(
          input_for_compute,
          buffer_input,
          packed_state.packed_weight.weight(),
          packed_bias_tensor,
          post_op,
          output_opt);
    }
    if (can_run_float_buffer_linear(
            buffer_input,
            packed_state.packed_weight.weight(),
            packed_bias_tensor)) {
      const utils::LinearGeluBridgeMatch bridge_match =
          match_linear_gelu_bridge_candidate(
              input_for_compute,
              buffer_input,
              packed_state,
              alpha,
              beta,
              post_op,
              output_opt);
      if (bridge_match.matched && bridge_match.may_defer) {
        Tensor placeholder = make_deferred_linear_gelu_placeholder(
            input_for_compute, buffer_input, packed_state);
        DeferredLinearGeluCandidate candidate;
        candidate.input_arg = input_for_compute;
        candidate.buffer_input = buffer_input;
        candidate.linear_context = linear_context;
        candidate.runtime_policy = runtime_policy;
        candidate.output_sizes = placeholder.sizes().vec();
        candidate.alpha = alpha;
        candidate.beta = beta;
        register_deferred_linear_gelu_candidate(
            placeholder, std::move(candidate));
        utils::log_vulkan_op_hit("aten::linear_gelu_bridge.defer");
        return placeholder;
      }
      return run_float_buffer_linear(
          input_for_compute,
          buffer_input,
          runtime_policy,
          packed_state,
          alpha,
          beta,
          post_op,
          output_opt);
    }
    if (can_run_widened_half_buffer_linear(
            buffer_input,
            packed_state.packed_weight.weight(),
            packed_bias_tensor)) {
      utils::log_vulkan_op_hit("aten::linear.buffer_half_widened_float");
      return run_widened_half_buffer_linear(
          input_for_compute,
          buffer_input,
          runtime_policy,
          packed_state,
          alpha,
          beta,
          post_op,
          output_opt);
    }
    TORCH_CHECK(
        false,
        "Vulkan buffer-packed linear expects a supported float "
        "buffer-native execution path");
  }

  if (
      linear_kernel_family_allows_channel_packed_input(runtime_policy) &&
      can_use_channel_packed_linear_input(source_v_input, packed_v_weight)) {
    utils::log_vulkan_op_hit("aten::linear.channel_packed_family");
    return run_addmm_context_channel_packed_input(
        input_for_compute,
        input_arg_2d,
        source_v_input,
        packed_state,
        alpha,
        beta,
        post_op,
        output_opt);
  }

  const vTensor& v_input = pack_inputs_using_width_packing(input, input_request);

  TORCH_CHECK(
      v_input.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      "run_addmm_context must have width packed input");

  TORCH_CHECK(
      packed_v_weight.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
      "run_addmm_context must have height packed weight");

  const std::vector<int64_t> output_sizes{
      input_arg_2d.sizes()[Layout::Parameter::height],
      unpacked_weight_sizes[Layout::Parameter::width],
  };
  Tensor output_tensor = output_opt
      ? ensure_linear_output_tensor(
            *output_opt, output_sizes, convert_dtype(v_input.dtype()))
      : convert(vTensor{context, output_sizes, v_input.dtype()});
  vTensor& v_output = convert(output_tensor);

  api::UniformParamsBuffer params;
  api::ShaderInfo compute_shader;
  // Step size is the 2d input's w dimension / 4.
  int step_size = div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(4));
  const bool fuse_bias =
      bias_defined &&
      can_fuse_linear_bias(v_output, packed_v_bias, unpacked_weight_sizes);
  const bool fuse_gelu = fuse_bias && post_op == LinearPostOp::Gelu;

  if (fuse_gelu) {
    const struct {
      uvec4 shader_extents_and_step;
      uvec4 bias_extents;
      vec4 multipliers_and_gelu;
    } block_with_bias_gelu{
        {
            v_output.extents().data[0u],
            v_output.extents().data[1u],
            v_output.extents().data[2u],
            safe_downcast<uint32_t>(step_size),
        },
        {
            packed_v_bias.extents().data[0u],
            packed_v_bias.extents().data[1u],
            packed_v_bias.extents().data[2u],
            0u,
        },
        {alpha, beta, kGeluBeta, 0.0f},
    };
    params = api::UniformParamsBuffer(context, block_with_bias_gelu);
    compute_shader = VK_KERNEL(mm_bias_gelu);
  } else if (fuse_bias) {
    const struct {
      uvec4 shader_extents_and_step;
      uvec4 bias_extents;
      vec2 multipliers;
    } block_with_bias{
        {
            v_output.extents().data[0u],
            v_output.extents().data[1u],
            v_output.extents().data[2u],
            safe_downcast<uint32_t>(step_size),
        },
        {
            packed_v_bias.extents().data[0u],
            packed_v_bias.extents().data[1u],
            packed_v_bias.extents().data[2u],
            0u,
        },
        {alpha, beta},
    };
    params = api::UniformParamsBuffer(context, block_with_bias);
    compute_shader = VK_KERNEL(mm_bias);
  } else {
    const struct {
      uvec3 shader_extents;
      uint32_t mm_step_size;
    } block_no_bias{
        v_output.extents(),
        safe_downcast<uint32_t>(step_size),
    };
    params = api::UniformParamsBuffer(context, block_no_bias);
    compute_shader = VK_KERNEL(mm);
  }

  api::PipelineBarrier pipeline_barrier{};

  if (fuse_bias) {
    context->submit_compute_job(
        // shader descriptor
        compute_shader,
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::width],
                INT64_C(4))),
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::height],
                INT64_C(4))),
            1,
        },
        // local work group size
        {8, 8, 1},
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());
  } else {
    context->submit_compute_job(
        // shader descriptor
        compute_shader,
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::width],
                INT64_C(4))),
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::height],
                INT64_C(4))),
            1,
        },
        // local work group size
        {8, 8, 1},
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());
  }

  Tensor output = output_tensor;

  // addmm/linear operation, multiplying by alpha and adding bias when present.
  if (!fuse_bias && alpha != 1.0f) {
    output = output.mul(alpha);
  }
  if (!fuse_bias && bias_defined) {
    output = output.add(convert(packed_v_bias).mul(beta));
  }
  if (post_op == LinearPostOp::Gelu && !fuse_gelu) {
    output = at::gelu(output, "none");
  }
  if (output_opt && output.unsafeGetTensorImpl() != output_tensor.unsafeGetTensorImpl()) {
    output = rebind_vulkan_output(*output_opt, output);
  }
  return reshape_linear_output_if_needed(output, input_arg);
}

Tensor run_baddbmm_context(
    const Tensor& input_arg,
    const float alpha,
    const float beta,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  const auto input_request =
      utils::make_vulkan_tensor_linear_request(
          input_arg, utils::VulkanTensorRole::Input);
  api::AllocationScope allocation_scope("bmm");
  // TODO: Refactor run_baddbmm_context and run_addmm_context into one.
  api::Context* const context = api::context();

  TORCH_CHECK(
      input_arg.dim() == 3,
      "Vulkan Linear not usable! "
      "Reason: The input has the wrong dimension; the tensor of a batch of matrices should contain 3 dimensions: batch, height, width.");

  utils::prime_labeled_scratch_arena_for_request(
      input_arg,
      input_request,
      linear_runtime_scratch_bytes(input_arg),
      "bmm_decode");
  const Tensor compute_input_arg = utils::prepare_vulkan_execution_tensor(
      input_arg,
      utils::VulkanExecutionPlanKind::LinearInputSource,
      input_request);
  Tensor input =
      compute_input_arg.is_vulkan() ? compute_input_arg
                                    : compute_input_arg.vulkan();
  if (input.scalar_type() == kHalf) {
    // The current batched matmul path backing Vulkan SDPA is much more stable
    // when half inputs are widened before packing. Keep the model path running
    // on Vulkan until a true native half batch-matmul family exists.
    input = input.to(kFloat);
  }
  vTensor packed_v_input = pack_inputs_using_width_packing(input, input_request);

  const LinearPackedRunState packed_state =
      get_linear_packed_run_state(linear_context);
  const vTensor& packed_v_weight = packed_state.packed_v_weight;
  const vTensor& packed_v_bias = packed_state.packed_v_bias;
  const std::vector<int64_t>& unpacked_weight_sizes =
      packed_state.logical_weight_sizes;

  TORCH_CHECK(
      usable(input, unpacked_weight_sizes, true /*use batch*/),
      "Vulkan Linear not usable! "
      "Reason: The provided input tensor is either invalid on its own, or its "
      "combination with the provided weight and bias tensors are unsupported by "
      "Vulkan impl.");

  TORCH_CHECK(
      packed_v_input.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      "run_addmm_context called for non-quantized version with unpacked weight");

  TORCH_CHECK(
      packed_v_weight.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
      "run_addmm_context called for non-quantized version with unpacked weight");

  int64_t input_batch = packed_v_input.sizes()[Layout::BatchMatrices::batch];

  // Step size is the input's w dimension / 4.
  int64_t input_width = packed_v_input.sizes()[Layout::BatchMatrices::width];
  int64_t mm_step_size = div_up(input_width, INT64_C(4));

  vTensor v_output{
      context,
      {
          input_batch,
          packed_v_input.sizes()[Layout::BatchMatrices::height],
          unpacked_weight_sizes.back(), // "w" dimension in weight matrix
      },
      packed_v_input.dtype(),
  };

  const struct {
    uvec4 shader_extents_and_step;
    uvec4 batch_info;
  } block_no_bias{
      {
          v_output.extents().data[0u],
          v_output.extents().data[1u],
          v_output.extents().data[2u],
          safe_downcast<uint32_t>(mm_step_size),
      },
      {
          safe_downcast<uint32_t>(input_batch),
          0u,
          0u,
          0u,
      },
  };

  api::UniformParamsBuffer params(context, block_no_bias);

  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(bmm_channel_packed),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      {
          safe_downcast<uint32_t>(div_up(
              v_output.sizes()[Layout::BatchMatrices::width], INT64_C(4))),
          safe_downcast<uint32_t>(div_up(
              v_output.sizes()[Layout::BatchMatrices::height], INT64_C(4))),
          v_output.extents().data[2u],
      },
      // local work group size
      {8, 8, 1},
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      packed_v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  // The dedicated batched kernel writes up to four batch results directly into
  // each channel-packed output texel, so no post-slice is needed here.
  return convert(v_output).mul(alpha).add(convert(packed_v_bias).mul(beta));
}

Tensor addmm(
    const Tensor& bias,
    const Tensor& input,
    const Tensor& weight,
    const Scalar& beta,
    const Scalar& alpha) {
  const bool beta_zero = beta.to<double>() == 0.0;
  const bool use_packed_bias =
      bias.dim() == 1 && beta.to<double>() == 1.0 && bias.requires_grad();
  const std::optional<Tensor> optional_bias =
      use_packed_bias ? std::optional<Tensor>(bias) : std::nullopt;
  if (
      input.scalar_type() == kBFloat16 ||
      weight.scalar_type() == kBFloat16 ||
      bias.scalar_type() == kBFloat16) {
    utils::log_vulkan_op_hit("aten::linear.bfloat16_widen_addmm");
    const Tensor float_input = input.scalar_type() == kBFloat16
        ? (input.is_vulkan() ? utils::cast_vulkan_tensor_dtype(input, kFloat)
                             : input.to(kFloat))
        : input;
    const Tensor float_weight = weight.scalar_type() == kBFloat16
        ? (weight.is_vulkan() ? utils::cast_vulkan_tensor_dtype(weight, kFloat)
                              : weight.to(kFloat))
        : weight;
    const std::optional<Tensor> float_bias = use_packed_bias
        ? (bias.scalar_type() == kBFloat16
               ? std::optional<Tensor>(
                     bias.is_vulkan()
                         ? utils::cast_vulkan_tensor_dtype(bias, kFloat)
                         : bias.to(kFloat))
               : optional_bias)
        : std::nullopt;
    const auto linear_context = c10::make_intrusive<LinearPackedContext>(
        LinearPackedContext(float_weight, float_bias));
    Tensor output = run_addmm_context(
        float_input,
        alpha.to<float>(),
        1.0f,
        linear_context,
        false,
        0,
        0);
    if (!beta_zero && !use_packed_bias) {
      const Tensor bias_for_add = bias.scalar_type() == kBFloat16
          ? (bias.is_vulkan() ? utils::cast_vulkan_tensor_dtype(bias, kFloat)
                              : bias.to(kFloat))
          : bias;
      output = output.add(bias_for_add.mul(beta.to<float>()));
    }
    api::context()->flush_pending_cmds();
    return output;
  }

  const auto linear_context = c10::make_intrusive<LinearPackedContext>(
      LinearPackedContext(
          weight,
          optional_bias,
          false,
          std::string(),
          false,
          true));
  Tensor output = run_addmm_context(
      input,
      alpha.to<float>(),
      1.0f,
      linear_context,
      false,
      0,
      0);
  if (!beta_zero && !use_packed_bias) {
    output = output.add(bias.mul(beta.to<float>()));
  }
  api::context()->flush_pending_cmds();
  return output;
}

Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  utils::log_vulkan_op_hit("aten::linear.direct");
  if (std::optional<Tensor> raw_direct_output =
          try_run_raw_direct_weight_linear(input, weight, bias)) {
    api::context()->flush_pending_cmds();
    return *raw_direct_output;
  }
  const auto linear_context = get_or_create_linear_context(weight, bias);
  Tensor output = run_addmm_context(
      input,
      1.0f,
      1.0f,
      linear_context,
      false,
      0,
      0);
  api::context()->flush_pending_cmds();
  return output;
}

Tensor run_half_buffer_linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  const Tensor float_input = upcast_half_linear_tensor_for_packing(input);
  const Tensor float_weight = upcast_half_linear_tensor_for_packing(weight);
  const std::optional<Tensor> float_bias =
      upcast_half_linear_tensor_for_packing(bias);

  Tensor output = run_addmm_context(
      float_input,
      1.0f,
      1.0f,
      c10::make_intrusive<LinearPackedContext>(
          LinearPackedContext(float_weight, float_bias)),
      false,
      0,
      0);
  return output.to(kHalf);
}

Tensor linear_gelu(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  const auto linear_context = get_or_create_linear_context(weight, bias);
  Tensor output = run_addmm_context(
      input,
      1.0f,
      1.0f,
      linear_context,
      false,
      0,
      0,
      LinearPostOp::Gelu);
  api::context()->flush_pending_cmds();
  return output;
}

Tensor mm(const Tensor& mat1_arg, const Tensor& mat2_arg) {
  utils::log_vulkan_op_hit("aten::mm");
  return run_addmm_context(
      mat1_arg,
      1.0f,
      1.0f,
      c10::make_intrusive<LinearPackedContext>(
          LinearPackedContext(mat2_arg, std::optional<Tensor>())),
      false,
      0,
      0);
}

Tensor bmm(const Tensor& mat1_arg, const Tensor& mat2_arg) {
  utils::log_vulkan_op_hit("aten::bmm");
  Tensor mat1 = mat1_arg.is_vulkan() ? mat1_arg : mat1_arg.vulkan();
  Tensor mat2 = mat2_arg.is_vulkan() ? mat2_arg : mat2_arg.vulkan();
  if (auto fused = try_consume_decomposed_attention_probs(mat1, mat2)) {
    return *fused;
  }
  mat1 = materialize_decomposed_attention_candidate_if_needed(mat1);
  mat2 = materialize_decomposed_attention_candidate_if_needed(mat2);
  if (can_run_half_buffer_bmm(mat1, mat2)) {
    utils::log_vulkan_op_hit("aten::bmm.buffer_float");
    return run_half_buffer_bmm(mat1, mat2, 1.0f, 1.0f);
  }
  if (can_run_float_buffer_bmm(mat1, mat2)) {
    if (auto scores = try_start_decomposed_attention_scores(mat1, mat2)) {
      return *scores;
    }
    mat1 = materialize_deferred_attention_query_scale_candidate_if_needed(mat1);
    mat2 = materialize_deferred_attention_query_scale_candidate_if_needed(mat2);
    utils::log_vulkan_op_hit("aten::bmm.buffer_float");
    return run_float_buffer_bmm(mat1, mat2, 1.0f, 1.0f);
  }
  return run_baddbmm_context(
      mat1_arg,
      1.0f,
      1.0f,
      c10::make_intrusive<LinearPackedContext>(LinearPackedContext(
          mat2_arg, std::optional<Tensor>(), true /*use batch*/)));
}

Tensor bmm_buffer_out_vulkan_impl(
    const Tensor& mat1,
    const Tensor& mat2,
    Tensor& output) {
  TORCH_CHECK(
      can_run_float_buffer_bmm(mat1, mat2),
      "Vulkan bmm_buffer_out expects float rank-3 buffer-backed tensors");
  return run_float_buffer_bmm(
      mat1, mat2, 1.0f, 1.0f, std::nullopt, &output);
}

Tensor baddbmm(
    const Tensor& bias,
    const Tensor& input,
    const Tensor& weight,
    const Scalar& beta,
    const Scalar& alpha) {
  const Tensor mat1 = input.is_vulkan() ? input : input.vulkan();
  const Tensor mat2 = weight.is_vulkan() ? weight : weight.vulkan();
  if (can_run_half_buffer_bmm(mat1, mat2)) {
    utils::log_vulkan_op_hit("aten::bmm.buffer_float");
    return run_half_buffer_bmm(
        mat1, mat2, alpha.to<float>(), beta.to<float>(), bias);
  }
  if (can_run_float_buffer_bmm(mat1, mat2)) {
    utils::log_vulkan_op_hit("aten::bmm.buffer_float");
    return run_float_buffer_bmm(
        mat1, mat2, alpha.to<float>(), beta.to<float>(), bias);
  }
  return run_baddbmm_context(
      input,
      alpha.to<float>(),
      beta.to<float>(),
      c10::make_intrusive<LinearPackedContext>(
          LinearPackedContext(weight, bias, true /*use batch*/)));
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::linear"), TORCH_FN(linear));
  m.impl(TORCH_SELECTIVE_NAME("aten::addmm"), TORCH_FN(addmm));
  m.impl(TORCH_SELECTIVE_NAME("aten::mm"), TORCH_FN(mm));
  m.impl(TORCH_SELECTIVE_NAME("aten::bmm"), TORCH_FN(bmm));
  m.impl(TORCH_SELECTIVE_NAME("aten::baddbmm"), TORCH_FN(baddbmm));
}

#endif /* USE_VULKAN_API */

} // namespace

std::optional<Tensor> try_consume_deferred_linear_gelu(
    const Tensor& input,
    std::string_view approximate) {
  return try_consume_deferred_linear_gelu_impl(input, approximate);
}

Tensor materialize_deferred_linear_gelu_candidate_if_needed(
    const Tensor& tensor) {
  return materialize_deferred_linear_gelu_candidate_impl(tensor);
}

void move_deferred_linear_gelu_candidate_to_alias(
    const Tensor& source,
    const Tensor& alias) {
  move_deferred_linear_gelu_candidate_to_alias_impl(source, alias);
}

std::vector<int64_t> linear_plan_counters_snapshot() {
  const VulkanLinearPlanCounters& counters = linear_plan_counters();
  return {
      static_cast<int64_t>(counters.total.load(std::memory_order_relaxed)),
      static_cast<int64_t>(counters.coop_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.coop_tail_m_hit.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_m_tail.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_k_tail.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_n_tail.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_layout.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_dtype.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.reject_capability.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.fallback_plain_bf16.load(std::memory_order_relaxed)),
      static_cast<int64_t>(
          counters.fallback_float.load(std::memory_order_relaxed)),
  };
}

std::vector<std::string> linear_aggregate_snapshot() {
  std::vector<std::pair<std::string, VulkanLinearAggregateValue>> rows;
  {
    std::lock_guard<std::mutex> guard(linear_aggregate_mutex());
    rows.reserve(linear_aggregate().size());
    for (const auto& item : linear_aggregate()) {
      rows.emplace_back(item.first, item.second);
    }
  }
  std::sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
    const uint64_t lhs_bytes =
        lhs.second.input_bytes + lhs.second.weight_bytes + lhs.second.output_bytes;
    const uint64_t rhs_bytes =
        rhs.second.input_bytes + rhs.second.weight_bytes + rhs.second.output_bytes;
    if (lhs_bytes != rhs_bytes) {
      return lhs_bytes > rhs_bytes;
    }
    return lhs.first < rhs.first;
  });

  std::vector<std::string> snapshot;
  snapshot.reserve(rows.size());
  for (const auto& row : rows) {
    std::ostringstream out;
    out << row.first
        << " count=" << row.second.count
        << " input_bytes=" << row.second.input_bytes
        << " weight_bytes=" << row.second.weight_bytes
        << " output_bytes=" << row.second.output_bytes;
    snapshot.emplace_back(out.str());
  }
  return snapshot;
}

std::vector<std::string> linear_plan_key_snapshot() {
  std::vector<std::string> aggregate_rows = linear_aggregate_snapshot();
  if (aggregate_rows.empty()) {
    return {};
  }
  api::Context* const context = api::context();
  api::Adapter* const adapter = context->adapter_ptr();
  const VkPhysicalDeviceProperties& properties =
      adapter->physical_device().properties;
  const auto field_value = [](const std::string& row,
                              const std::string& key) -> std::string {
    const std::string needle = key + "=";
    const size_t begin = row.find(needle);
    if (begin == std::string::npos) {
      return "unknown";
    }
    const size_t value_begin = begin + needle.size();
    const size_t value_end = row.find(' ', value_begin);
    return row.substr(
        value_begin,
        value_end == std::string::npos ? std::string::npos
                                      : value_end - value_begin);
  };

  std::vector<std::string> snapshot;
  snapshot.reserve(aggregate_rows.size());
  for (const std::string& row : aggregate_rows) {
    const bool tiled = row.find("buffer_float_tiled") != std::string::npos;
    const bool float_buffer = row.find("buffer_float") != std::string::npos;
    const bool raw_direct = row.find("raw_direct=1") != std::string::npos;
    const std::string m = field_value(row, "m");
    const std::string k = field_value(row, "k");
    const std::string n = field_value(row, "n");
    const std::string input_direct = field_value(row, "input_direct");
    const std::string output_direct = field_value(row, "output_direct");
    const std::string weight_packed = field_value(row, "weight_packed");
    std::ostringstream out;
    out << "schema=VulkanLinearOrMatmulPlanKey.v0"
        << " source=linear_aggregate_snapshot"
        << " op_family=linear"
        << " selected="
        << (raw_direct
                ? "RawDirectWeightLinear"
                : (tiled ? "FloatBufferTiledLinear"
                         : (float_buffer ? "FloatBufferLinear"
                                         : "UnknownLinear")))
        << ' ' << row
        << " input=[" << m << ',' << k << ']'
        << " weight=[" << k << ',' << n << ']'
        << " output=[" << m << ',' << n << ']'
        << " input_storage=DirectBuffer"
        << " weight_storage="
        << (weight_packed == "1" ? "PackedWeightBuffer" : "DirectBuffer")
        << " output_storage=DirectBuffer"
        << " input_layout="
        << (input_direct == "1" ? "direct_buffer" : "unknown")
        << " weight_layout="
        << (weight_packed == "1" ? "packed_weight" : "direct_buffer")
        << " output_layout="
        << (output_direct == "1" ? "direct_buffer" : "unknown")
        << " weight_direct=" << (weight_packed == "1" ? 0 : 1)
        << " global=[" << n << ',' << m << ",1]"
        << " local=[" << (tiled ? "16,16,1" : "16,4,1") << ']'
        << " candidate_count=1"
        << " cacheable=1"
        << " tunable=" << (float_buffer ? 1 : 0)
        << " context_device_index="
        << static_cast<int64_t>(context->device_index())
        << " vendor_id=" << properties.vendorID
        << " device_id=" << properties.deviceID
        << " driver_version=" << properties.driverVersion
        << " api_version=" << adapter->api_version()
        << " subgroup_size=" << adapter->subgroup_size()
        << " min_subgroup_size=" << adapter->min_subgroup_size()
        << " max_subgroup_size=" << adapter->max_subgroup_size()
        << " max_compute_workgroup_subgroups="
        << adapter->max_compute_workgroup_subgroups()
        << " has_subgroup_size_control="
        << (adapter->has_subgroup_size_control() ? 1 : 0)
        << " has_compute_full_subgroups="
        << (adapter->has_compute_full_subgroups() ? 1 : 0)
        << " has_cooperative_matrix="
        << (adapter->has_cooperative_matrix() ? 1 : 0)
        << " cooperative_matrix_property_count="
        << adapter->cooperative_matrix_property_count()
        << " has_timeline_semaphore="
        << (adapter->has_timeline_semaphore() ? 1 : 0)
        << " has_synchronization2="
        << (adapter->has_synchronization2() ? 1 : 0);
    snapshot.emplace_back(out.str());
  }
  return snapshot;
}

std::vector<std::string> linear_pack_residency_snapshot() {
  std::vector<std::pair<std::string, VulkanLinearPackResidencyValue>> rows;
  {
    std::lock_guard<std::mutex> guard(linear_pack_residency_mutex());
    rows.reserve(linear_pack_residency_aggregate().size());
    for (const auto& item : linear_pack_residency_aggregate()) {
      rows.emplace_back(item.first, item.second);
    }
  }
  std::sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.second.packed_bytes != rhs.second.packed_bytes) {
      return lhs.second.packed_bytes > rhs.second.packed_bytes;
    }
    return lhs.first < rhs.first;
  });

  std::vector<std::string> snapshot;
  snapshot.reserve(rows.size());
  for (const auto& row : rows) {
    std::ostringstream out;
    out << "linear_pack_residency " << row.first
        << " count=" << row.second.count
        << " created=" << row.second.created
        << " reused=" << row.second.reused
        << " packed_bytes=" << row.second.packed_bytes
        << " raw_weight_bytes=" << row.second.raw_weight_bytes
        << " raw_bias_bytes=" << row.second.raw_bias_bytes
        << " raw_weight_vulkan=" << row.second.raw_weight_vulkan
        << " retain_unpacked=" << row.second.retain_unpacked;
    snapshot.emplace_back(out.str());
  }
  return snapshot;
}

void reset_linear_plan_counters() {
  VulkanLinearPlanCounters& counters = linear_plan_counters();
  counters.total.store(0, std::memory_order_relaxed);
  counters.coop_hit.store(0, std::memory_order_relaxed);
  counters.coop_tail_m_hit.store(0, std::memory_order_relaxed);
  counters.reject_m_tail.store(0, std::memory_order_relaxed);
  counters.reject_k_tail.store(0, std::memory_order_relaxed);
  counters.reject_n_tail.store(0, std::memory_order_relaxed);
  counters.reject_layout.store(0, std::memory_order_relaxed);
  counters.reject_dtype.store(0, std::memory_order_relaxed);
  counters.reject_capability.store(0, std::memory_order_relaxed);
  counters.fallback_plain_bf16.store(0, std::memory_order_relaxed);
  counters.fallback_float.store(0, std::memory_order_relaxed);
}

void reset_linear_aggregate() {
  std::lock_guard<std::mutex> guard(linear_aggregate_mutex());
  linear_aggregate().clear();
}

void reset_linear_pack_residency_snapshot() {
  std::lock_guard<std::mutex> guard(linear_pack_residency_mutex());
  linear_pack_residency_aggregate().clear();
}

PackedWeightResidencyClass linear_buffer_weight_residency_class(
    const size_t resident_nbytes,
    const std::vector<int64_t>& logical_weight_sizes) {
  const auto policy = utils::current_vulkan_device_policy();
  if (!policy.avoid_large_persistent_weight_cache) {
    const size_t transient_threshold =
        policy.transient_large_linear_weight_cache_threshold_bytes;
    if (transient_threshold == 0u || resident_nbytes < transient_threshold) {
      return PackedWeightResidencyClass::PersistentInference;
    }
    std::ostringstream stream;
    stream
        << "aten::linear.packed_weight_cache_transient.large_device_policy bytes="
        << resident_nbytes << " threshold=" << transient_threshold
        << " weight=[";
    for (size_t idx = 0; idx < logical_weight_sizes.size(); ++idx) {
      if (idx != 0u) {
        stream << ",";
      }
      stream << logical_weight_sizes[idx];
    }
    stream << "]";
    utils::log_vulkan_op_hit(stream.str());
    return PackedWeightResidencyClass::Transient;
  }

  std::ostringstream stream;
  stream << "aten::linear.packed_weight_cache_transient.device_policy bytes="
         << resident_nbytes << " weight=[";
  for (size_t idx = 0; idx < logical_weight_sizes.size(); ++idx) {
    if (idx != 0u) {
      stream << ",";
    }
    stream << logical_weight_sizes[idx];
  }
  stream << "]";
  utils::log_vulkan_op_hit(stream.str());
  return PackedWeightResidencyClass::Transient;
}

bool should_store_linear_buffer_packed_weight_handle(
    const PackedWeightHandle& handle,
    const uint64_t options_key) {
  if (handle.residency_class() == PackedWeightResidencyClass::Transient) {
    std::ostringstream stream;
    stream << "aten::linear.packed_weight_cache_skip.transient bytes="
           << handle.resident_nbytes() << " weight=[";
    const auto& logical_weight_sizes = handle.logical_weight_sizes();
    for (size_t idx = 0; idx < logical_weight_sizes.size(); ++idx) {
      if (idx != 0u) {
        stream << ",";
      }
      stream << logical_weight_sizes[idx];
    }
    stream << "]";
    utils::log_vulkan_op_hit(stream.str());
    utils::note_packed_weight_store_skip(
        handle.logical_weight_sizes(),
        handle.weight().scalar_type(),
        handle.kind(),
        handle.quantized(),
        options_key,
        "transient",
        handle.resident_nbytes());
    return false;
  }
  return true;
}

Tensor bmm_buffer_out_vulkan(
    const Tensor& mat1,
    const Tensor& mat2,
    Tensor& output) {
  return bmm_buffer_out_vulkan_impl(mat1, mat2, output);
}

LinearPackedContext::LinearPackedContext(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const bool use_batch,
    std::string allocation_label,
    const bool retain_unpacked,
    const bool use_packed_weight_cache)
    : unpacked_{c10::AnyType::get()} {
  allocation_label_ = std::move(allocation_label);
  api::AllocationScope allocation_scope(
      utils::make_vulkan_linear_pack_label(
          allocation_label_, use_batch ? "bmm.pack" : "linear.pack"));
  const auto normalized_bias = utils::normalized_optional_tensor(bias);
  const std::vector<int64_t> logical_weight_sizes = weight.sizes().vec();
  constexpr uint64_t kLinearBatchPackOption = 1u;
  constexpr uint64_t kLinearBufferPackOption = 2u;
  const bool use_buffer_packed_weights =
      !use_batch &&
      !weight.is_quantized() &&
      is_float_or_half_tensor(weight) &&
      (!bias || !bias->defined() || is_float_or_half_tensor(*bias));
  const uint64_t pack_options = (use_batch ? kLinearBatchPackOption : 0u) |
      (use_buffer_packed_weights ? kLinearBufferPackOption : 0u);
  std::optional<PackedWeightHandle> cached_packed_weight;
  if (use_packed_weight_cache) {
    cached_packed_weight = utils::lookup_packed_weight_handle(
          weight,
          normalized_bias,
          logical_weight_sizes,
          PackedWeightKind::Linear,
          weight.is_quantized(),
          pack_options);
  }
  const bool reused_packed_weight = cached_packed_weight.has_value();
  if (cached_packed_weight) {
    packed_weight_ = *cached_packed_weight;
  } else {
    const bool preserve_bfloat16_buffer_weight =
        use_buffer_packed_weights && weight.scalar_type() == kBFloat16;
    const Tensor pack_source_weight = preserve_bfloat16_buffer_weight
        ? weight
        : upcast_half_linear_tensor_for_packing(weight);
    std::optional<Tensor> pack_source_bias =
        upcast_half_linear_tensor_for_packing(bias);
    const Tensor compute_weight = pack_source_weight;
    const std::optional<Tensor> compute_bias = pack_source_bias;
    TORCH_CHECK(
        available(compute_weight, compute_bias, use_batch),
        "Vulkan Linear not available! "
        "Reason: The provided (weight, bias) parameters are either invalid "
        "individually or their combination is not supported by Vulkan Impl.");

    if (use_buffer_packed_weights) {
      const Tensor buffer_weight_source = preserve_bfloat16_buffer_weight
          ? pack_source_weight.t().contiguous()
          : compute_weight;
      Tensor buffer_weight = upload_linear_tensor_to_buffer(
          buffer_weight_source, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);

      Tensor buffer_bias_tensor;
      if (compute_bias && compute_bias->defined()) {
        buffer_bias_tensor = upload_linear_tensor_to_buffer(
            *compute_bias, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
      } else {
        buffer_bias_tensor = upload_linear_tensor_to_buffer(
            at::zeros({1}, at::device(at::kCPU).dtype(at::kFloat)),
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
      }

      const size_t resident_nbytes =
          convert(buffer_weight).gpu_nbytes() +
          (buffer_bias_tensor.defined() ? convert(buffer_bias_tensor).gpu_nbytes()
                                        : 0u);
      const PackedWeightResidencyClass residency_class =
          linear_buffer_weight_residency_class(
              resident_nbytes, logical_weight_sizes);
      packed_weight_ = PackedWeightHandle(
          std::move(buffer_weight),
          std::move(buffer_bias_tensor),
          logical_weight_sizes,
          PackedWeightKind::Linear,
          compute_bias && compute_bias->defined(),
          residency_class,
          false,
          api::ExecutionLayout::BUFFER_DIRECT,
          resident_nbytes);
    } else {
      const Tensor packed_weight = utils::prepare_vulkan_execution_tensor(
          pack_source_weight,
          utils::VulkanExecutionPlanKind::LinearWeightSource,
          utils::make_vulkan_linear_request(utils::VulkanTensorRole::Weight));
      const std::optional<Tensor> packed_bias =
          utils::prepare_optional_vulkan_execution_tensor(
              pack_source_bias,
              utils::VulkanExecutionPlanKind::LinearBiasSource,
              utils::make_vulkan_linear_request(utils::VulkanTensorRole::Bias));
      const Tensor texture_compute_weight =
          upcast_half_linear_tensor_for_packing(packed_weight);
      const std::optional<Tensor> texture_compute_bias =
          upcast_half_linear_tensor_for_packing(packed_bias);

      Tensor packed_bias_tensor = packed_weight.is_quantized()
          ? convert(pack_biases_quantized_weights(
                texture_compute_weight, texture_compute_bias, use_batch))
          : convert(
                pack_biases(texture_compute_weight, texture_compute_bias, use_batch));

      packed_weight_ = utils::make_packed_weight_handle(
          convert(pack_weights(texture_compute_weight, use_batch)),
          std::move(packed_bias_tensor),
          packed_weight.sizes().vec(),
          PackedWeightKind::Linear,
          texture_compute_bias && texture_compute_bias->defined(),
          packed_weight.is_quantized());
    }
    if (
        use_packed_weight_cache &&
        should_store_linear_buffer_packed_weight_handle(
            packed_weight_, pack_options)) {
      utils::store_packed_weight_handle(
          weight,
          normalized_bias,
          logical_weight_sizes,
          PackedWeightKind::Linear,
          packed_weight_,
          weight.is_quantized(),
          pack_options);
    }
  }
  note_linear_pack_residency(
      weight,
      normalized_bias,
      packed_weight_,
      reused_packed_weight,
      retain_unpacked && !at::globalContext().releaseWeightsWhenPrepacking(),
      use_batch,
      use_buffer_packed_weights);

  if (retain_unpacked && !at::globalContext().releaseWeightsWhenPrepacking()) {
    unpacked_.reserve(Unpacked::NumArgs);
    unpacked_.emplace_back(weight);
    unpacked_.emplace_back(normalized_bias);
  }
}

LinearPackedContext LinearPackedContext::pack(c10::impl::GenericList unpacked) {
  return LinearPackedContext(
      unpacked.get(Unpacked::Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Bias));
}

c10::intrusive_ptr<LinearPackedContext> create_linear_context(
    Tensor&& weight,
    std::optional<Tensor>&& bias) {
  return c10::make_intrusive<LinearPackedContext>(
      LinearPackedContext(weight, bias));
}

c10::intrusive_ptr<LinearPackedContext> create_linear_context_labeled(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    std::string label) {
  if (const auto cached_context =
          utils::lookup_labeled_linear_context(weight, bias, label)) {
    return *cached_context;
  }

  const Tensor prepared_weight =
      (c10::InferenceMode::is_enabled() && weight.is_vulkan() &&
       weight.dim() == 2)
      ? transposed_linear_weight_for_packing(
            weight, "labeled_weight_cpu_transpose")
      : weight.t();
  const auto context = c10::make_intrusive<LinearPackedContext>(
      LinearPackedContext(
          prepared_weight,
          bias,
          false,
          std::move(label),
          false,
          true));
  utils::store_labeled_linear_context(
      weight, bias, context->allocation_label(), context);
  return context;
}

Tensor run_linear_context(
    const Tensor& input,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  utils::log_vulkan_op_hit("vulkan_prepack::run_linear_context");
  utils::validate_replay_tensor_not_stale(
      input, "vulkan_prepack::run_linear_context");
  Tensor output =
      run_addmm_context(input, 1.0f, 1.0f, linear_context, false, 0, 0);
  return record_tensor_write_and_return(
      output, "vulkan_prepack::run_linear_context", "linear_context", {input});
}

Tensor run_linear_context_out(
    const Tensor& input,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context,
    Tensor& output) {
  utils::log_vulkan_op_hit("vulkan_prepack::run_linear_context");
  utils::validate_replay_tensor_not_stale(
      input, "vulkan_prepack::run_linear_context_out");
  Tensor result = run_addmm_context(
      input,
      1.0f,
      1.0f,
      linear_context,
      false,
      0,
      0,
      LinearPostOp::None,
      &output);
  return record_tensor_write_and_return(
      result,
      "vulkan_prepack::run_linear_context",
      "linear_context_out",
      {input});
}

Tensor run_linear_gelu_context(
    const Tensor& input,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  utils::validate_replay_tensor_not_stale(
      input, "vulkan_prepack::run_linear_gelu_context");
  Tensor output = run_addmm_context(
      input,
      1.0f,
      1.0f,
      linear_context,
      false,
      0,
      0,
      LinearPostOp::Gelu);
  return record_tensor_write_and_return(
      output,
      "vulkan_prepack::run_linear_gelu_context",
      "linear_gelu_context",
      {input});
}

Tensor run_linear_gelu_context_out(
    const Tensor& input,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context,
    Tensor& output) {
  utils::validate_replay_tensor_not_stale(
      input, "vulkan_prepack::run_linear_gelu_context_out");
  Tensor result = run_addmm_context(
      input,
      1.0f,
      1.0f,
      linear_context,
      false,
      0,
      0,
      LinearPostOp::Gelu,
      &output);
  return record_tensor_write_and_return(
      result,
      "vulkan_prepack::run_linear_gelu_context",
      "linear_gelu_context_out",
      {input});
}

Tensor run_qlinear_context(
    const Tensor& input_arg,
    double output_scale,
    int64_t output_zero_point,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  utils::validate_replay_tensor_not_stale(
      input_arg, "vulkan_prepack::run_qlinear_context");
  Tensor output = run_addmm_context(
      input_arg,
      1.0f,
      1.0f,
      linear_context,
      true,
      output_scale,
      output_zero_point);
  return record_tensor_write_and_return(
      output,
      "vulkan_prepack::run_qlinear_context",
      "qlinear_context",
      {input_arg});
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
