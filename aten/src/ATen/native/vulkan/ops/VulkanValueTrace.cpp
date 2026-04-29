#include <ATen/native/vulkan/ops/VulkanValueTrace.h>

#ifdef USE_VULKAN_API

#include <ATen/Functions.h>
#include <ATen/native/vulkan/api/Diagnostics.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/TensorState.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace {

struct VulkanValueTraceConfig final {
  bool enabled{false};
  std::string log_path;
  int64_t sample_count{64};
  double max_abs_threshold{0.0};
};

struct VulkanValueStats final {
  int64_t numel{0};
  double min{0.0};
  double max{0.0};
  double mean{0.0};
  int64_t nan_count{0};
  int64_t inf_count{0};
  uint64_t sample_hash{1469598103934665603ULL};
  std::vector<int64_t> sample_indices;
  std::vector<double> sample_values;
};

bool env_flag_enabled(const char* name) {
  const char* env = std::getenv(name);
  if (!env || env[0] == '\0') {
    return false;
  }
  const std::string value(env);
  return value != "0" && value != "false" && value != "False" &&
      value != "FALSE";
}

int64_t env_int64(const char* name, int64_t fallback) {
  const char* env = std::getenv(name);
  if (!env || env[0] == '\0') {
    return fallback;
  }
  try {
    return std::stoll(env);
  } catch (...) {
    return fallback;
  }
}

double env_double(const char* name, double fallback) {
  const char* env = std::getenv(name);
  if (!env || env[0] == '\0') {
    return fallback;
  }
  try {
    return std::stod(env);
  } catch (...) {
    return fallback;
  }
}

VulkanValueTraceConfig value_trace_config() {
  VulkanValueTraceConfig config;
  config.enabled = env_flag_enabled("PYTORCH_VULKAN_VALIDATE_VALUES") ||
      env_flag_enabled("PYTORCH_VULKAN_VALUE_TRACE");
  if (const char* path = std::getenv("PYTORCH_VULKAN_VALUE_TRACE_LOG")) {
    config.log_path = path;
    if (!config.log_path.empty()) {
      config.enabled = true;
    }
  }
  config.sample_count =
      std::max<int64_t>(0, env_int64("PYTORCH_VULKAN_VALUE_TRACE_SAMPLES", 64));
  config.max_abs_threshold =
      env_double("PYTORCH_VULKAN_VALUE_TRACE_MAX_ABS", 0.0);
  return config;
}

std::mutex& value_trace_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

bool& value_trace_guard() {
  static thread_local bool active = false;
  return active;
}

std::string json_escape(const std::string& value) {
  std::ostringstream stream;
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        stream << "\\\\";
        break;
      case '"':
        stream << "\\\"";
        break;
      case '\n':
        stream << "\\n";
        break;
      case '\r':
        stream << "\\r";
        break;
      case '\t':
        stream << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(static_cast<unsigned char>(ch))
                 << std::dec << std::setfill(' ');
        } else {
          stream << ch;
        }
        break;
    }
  }
  return stream.str();
}

template <typename T>
void write_json_array(std::ostringstream& stream, const std::vector<T>& values) {
  stream << '[';
  for (size_t idx = 0; idx < values.size(); ++idx) {
    if (idx != 0u) {
      stream << ',';
    }
    stream << values[idx];
  }
  stream << ']';
}

void write_json_string_array(
    std::ostringstream& stream,
    const std::vector<std::string>& values) {
  stream << '[';
  for (size_t idx = 0; idx < values.size(); ++idx) {
    if (idx != 0u) {
      stream << ',';
    }
    stream << '"' << json_escape(values[idx]) << '"';
  }
  stream << ']';
}

std::string json_double(const double value) {
  if (std::isnan(value)) {
    return "null";
  }
  if (std::isinf(value)) {
    return "null";
  }
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}

void write_json_double_array(
    std::ostringstream& stream,
    const std::vector<double>& values) {
  stream << '[';
  for (size_t idx = 0; idx < values.size(); ++idx) {
    if (idx != 0u) {
      stream << ',';
    }
    stream << json_double(values[idx]);
  }
  stream << ']';
}

uint64_t fnv_mix_u64(uint64_t hash, uint64_t value) {
  constexpr uint64_t kFnvPrime = 1099511628211ULL;
  for (int byte_idx = 0; byte_idx < 8; ++byte_idx) {
    hash ^= (value >> (byte_idx * 8)) & 0xffU;
    hash *= kFnvPrime;
  }
  return hash;
}

uint64_t double_bits_for_hash(const double value) {
  if (std::isnan(value)) {
    return 0x7ff8000000000000ULL;
  }
  if (std::isinf(value)) {
    return value > 0.0 ? 0x7ff0000000000000ULL : 0xfff0000000000000ULL;
  }
  const double quantized = std::round(value * 1000000.0) / 1000000.0;
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(quantized), "unexpected double size");
  std::memcpy(&bits, &quantized, sizeof(bits));
  return bits;
}

std::vector<int64_t> deterministic_sample_indices(
    const int64_t numel,
    const int64_t requested_samples) {
  std::vector<int64_t> indices;
  if (numel <= 0 || requested_samples <= 0) {
    return indices;
  }
  const int64_t count = std::min<int64_t>(numel, requested_samples);
  indices.reserve(static_cast<size_t>(count));
  if (count == 1) {
    indices.push_back(0);
    return indices;
  }
  for (int64_t idx = 0; idx < count; ++idx) {
    indices.push_back((idx * (numel - 1)) / (count - 1));
  }
  return indices;
}

Tensor stats_source_tensor(const Tensor& cpu_tensor) {
  if (cpu_tensor.scalar_type() == kHalf ||
      cpu_tensor.scalar_type() == kBFloat16) {
    return cpu_tensor.to(kDouble);
  }
  if (c10::isFloatingType(cpu_tensor.scalar_type())) {
    return cpu_tensor.to(kDouble);
  }
  if (c10::isIntegralType(cpu_tensor.scalar_type(), /*includeBool=*/true)) {
    return cpu_tensor.to(kDouble);
  }
  return Tensor();
}

VulkanValueStats compute_value_stats(
    const Tensor& tensor,
    const int64_t requested_samples) {
  VulkanValueStats stats;
  stats.numel = tensor.numel();
  if (!tensor.defined() || stats.numel == 0) {
    return stats;
  }

  const Tensor cpu_tensor = tensor.is_vulkan() ? tensor.cpu() : tensor.cpu();
  const Tensor source = stats_source_tensor(cpu_tensor);
  if (!source.defined()) {
    return stats;
  }

  const Tensor flat = source.reshape({-1}).contiguous();
  const auto* data = flat.const_data_ptr<double>();
  bool have_finite = false;
  double sum = 0.0;
  int64_t finite_count = 0;
  stats.min = std::numeric_limits<double>::infinity();
  stats.max = -std::numeric_limits<double>::infinity();

  for (int64_t idx = 0; idx < stats.numel; ++idx) {
    const double value = data[idx];
    if (std::isnan(value)) {
      ++stats.nan_count;
      continue;
    }
    if (std::isinf(value)) {
      ++stats.inf_count;
      continue;
    }
    have_finite = true;
    stats.min = std::min(stats.min, value);
    stats.max = std::max(stats.max, value);
    sum += value;
    ++finite_count;
  }

  if (have_finite && finite_count > 0) {
    stats.mean = sum / static_cast<double>(finite_count);
  } else {
    stats.min = 0.0;
    stats.max = 0.0;
    stats.mean = 0.0;
  }

  stats.sample_indices =
      deterministic_sample_indices(stats.numel, requested_samples);
  stats.sample_values.reserve(stats.sample_indices.size());
  for (const int64_t sample_idx : stats.sample_indices) {
    const double value = data[sample_idx];
    stats.sample_values.push_back(value);
    stats.sample_hash = fnv_mix_u64(stats.sample_hash, sample_idx);
    stats.sample_hash = fnv_mix_u64(stats.sample_hash, double_bits_for_hash(value));
  }

  return stats;
}

std::string scalar_type_name(ScalarType dtype) {
  std::ostringstream stream;
  stream << dtype;
  return stream.str();
}

std::string value_trace_record_json(
    const Tensor& output,
    const char* op_name,
    const char* route_name,
    ArrayRef<Tensor> inputs,
    const VulkanTensorStateDesc& state,
    const VulkanValueStats& stats) {
  std::vector<std::string> input_provenance;
  input_provenance.reserve(inputs.size());
  for (const Tensor& input : inputs) {
    input_provenance.emplace_back(describe_tensor_provenance(input));
  }

  std::ostringstream stream;
  stream << std::setprecision(17);
  stream << '{';
  stream << "\"event\":\"vulkan_value_write\"";
  stream << ",\"op\":\""
         << json_escape(op_name && op_name[0] != '\0' ? op_name : "<unknown>")
         << '"';
  stream << ",\"route\":\""
         << json_escape(
                route_name && route_name[0] != '\0' ? route_name
                                                     : "<unspecified>")
         << '"';
  stream << ",\"dtype\":\"" << json_escape(scalar_type_name(output.scalar_type()))
         << '"';
  stream << ",\"repr\":\"" << vulkan_tensor_repr_name(state.repr) << '"';
  stream << ",\"storage_type\":\""
         << vulkan_storage_type_name(state.storage_type) << '"';
  stream << ",\"memory_layout\":\""
         << vulkan_memory_layout_name(state.memory_layout) << '"';
  stream << ",\"execution_layout\":\""
         << api::to_string(state.execution_layout) << '"';
  stream << ",\"sizes\":";
  write_json_array(stream, state.logical_sizes);
  stream << ",\"strides\":";
  write_json_array(stream, state.logical_strides);
  stream << ",\"physical_sizes\":";
  write_json_array(stream, state.physical_sizes);
  stream << ",\"storage_offset\":" << state.storage_offset;
  stream << ",\"storage_id\":\"0x" << std::hex << state.storage_id << '"';
  stream << ",\"view_id\":\"0x" << state.view_id << '"';
  stream << ",\"generation\":" << std::dec << state.generation;
  stream << ",\"logical_desc_hash\":\"0x" << std::hex
         << state.logical_desc_hash << std::dec << '"';
  stream << ",\"numel\":" << stats.numel;
  stream << ",\"min\":" << json_double(stats.min);
  stream << ",\"max\":" << json_double(stats.max);
  stream << ",\"mean\":" << json_double(stats.mean);
  stream << ",\"nan_count\":" << stats.nan_count;
  stream << ",\"inf_count\":" << stats.inf_count;
  stream << ",\"sample_hash\":\"0x" << std::hex << stats.sample_hash
         << std::dec << '"';
  stream << ",\"sample_indices\":";
  write_json_array(stream, stats.sample_indices);
  stream << ",\"sample_values\":";
  write_json_double_array(stream, stats.sample_values);
  stream << ",\"output_provenance\":\""
         << json_escape(describe_tensor_provenance(output)) << '"';
  stream << ",\"input_provenance\":";
  write_json_string_array(stream, input_provenance);
  stream << '}';
  return stream.str();
}

void append_value_trace_log(const std::string& path, const std::string& record) {
  if (path.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(value_trace_log_mutex());
  std::ofstream out(path, std::ios::out | std::ios::app);
  out << record << '\n';
}

void fail_on_invalid_values(
    const Tensor& output,
    const char* op_name,
    const char* route_name,
    const VulkanTensorStateDesc& state,
    const VulkanValueStats& stats,
    const VulkanValueTraceConfig& config) {
  const bool has_nonfinite = stats.nan_count != 0 || stats.inf_count != 0;
  const double max_abs = std::max(std::abs(stats.min), std::abs(stats.max));
  const bool exceeds_range =
      config.max_abs_threshold > 0.0 && max_abs > config.max_abs_threshold;
  if (!has_nonfinite && !exceeds_range) {
    return;
  }

  std::ostringstream detail;
  detail << "op=" << (op_name && op_name[0] != '\0' ? op_name : "<unknown>")
         << " route="
         << (route_name && route_name[0] != '\0' ? route_name : "<unspecified>")
         << " nan_count=" << stats.nan_count
         << " inf_count=" << stats.inf_count
         << " min=" << stats.min
         << " max=" << stats.max
         << " mean=" << stats.mean
         << " sample_hash=0x" << std::hex << stats.sample_hash << std::dec
         << " state={" << describe_tensor_state(state) << "} "
         << describe_tensor_provenance(output);

  api::fail_vulkan(
      api::VulkanFailureClass::KernelIncorrect,
      op_name && op_name[0] != '\0' ? op_name : "vulkan_value_trace",
      has_nonfinite ? "ValueTraceNonFinite" : "ValueTraceRangeExceeded",
      detail.str());
}

} // namespace

bool vulkan_value_trace_enabled() {
  return value_trace_config().enabled;
}

void record_tensor_value_write(
    const Tensor& output,
    const char* op_name,
    const char* route_name,
    ArrayRef<Tensor> inputs) {
  const VulkanValueTraceConfig config = value_trace_config();
  if (!config.enabled || !output.defined() || output.numel() == 0) {
    return;
  }
  bool& guard = value_trace_guard();
  if (guard) {
    return;
  }

  guard = true;
  try {
    const VulkanTensorStateDesc state = inspect_tensor_state(output);
    const VulkanValueStats stats =
        compute_value_stats(output, config.sample_count);
    const std::string record =
        value_trace_record_json(output, op_name, route_name, inputs, state, stats);
    append_value_trace_log(config.log_path, record);
    fail_on_invalid_values(output, op_name, route_name, state, stats, config);
    guard = false;
  } catch (...) {
    guard = false;
    throw;
  }
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
