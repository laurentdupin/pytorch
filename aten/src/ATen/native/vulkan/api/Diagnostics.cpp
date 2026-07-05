#include <ATen/native/vulkan/api/Diagnostics.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Sync.h>

#include <c10/util/Exception.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace api {

namespace {

std::string failure_log_path() {
  const char* env = std::getenv("PYTORCH_VULKAN_FAILURE_LOG");
  return env ? std::string(env) : std::string();
}

std::mutex& failure_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::atomic<bool>& post_failure_recovery_required() {
  static std::atomic<bool> required{false};
  return required;
}

const std::string& lazy_chain_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_LAZY_CHAIN_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

const std::string& deferred_execution_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_DEFERRED_EXECUTION_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

const std::string& deferred_region_plan_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_DEFERRED_REGION_PLAN_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

const std::string& runtime_shader_compile_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_RUNTIME_SHADER_COMPILE_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

const std::string& runtime_shader_cache_dir() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_RUNTIME_SHADER_CACHE_DIR");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

const std::string& runtime_shader_glslc_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_RUNTIME_SHADER_GLSLC");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

const std::string& runtime_command_list_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_RUNTIME_COMMAND_LIST_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool lazy_chain_logging_enabled() {
  return !lazy_chain_log_path().empty();
}

bool deferred_execution_logging_enabled() {
  return !deferred_execution_log_path().empty();
}

bool deferred_region_plan_logging_enabled() {
  return !deferred_region_plan_log_path().empty();
}

bool runtime_shader_compile_logging_enabled() {
  return !runtime_shader_compile_log_path().empty() &&
      !runtime_shader_cache_dir().empty();
}

bool runtime_command_list_logging_enabled() {
  return !runtime_command_list_log_path().empty();
}

std::mutex& lazy_chain_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::mutex& deferred_execution_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::mutex& deferred_region_plan_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::mutex& runtime_shader_compile_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::mutex& runtime_command_list_log_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::mutex& runtime_shader_compile_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_set<std::string>& runtime_shader_compile_cache() {
  static auto* cache = new std::unordered_set<std::string>();
  return *cache;
}

struct DeferredTensorHandleRecord final {
  uint64_t handle_id = 0;
  uint64_t storage_id = 0;
  uint64_t view_id = 0;
  uint64_t generation = 0;
  uint64_t logical_desc_hash = 0;
  int64_t storage_offset = 0;
  int64_t buffer_length = 0;
  bool is_view = false;
  std::string state;
};

struct DeferredOpNodeRecord final {
  uint64_t node_id = 0;
  uint64_t output_handle_id = 0;
  std::string op;
  std::string route;
  uint64_t input_count = 0;
  uint64_t vulkan_input_count = 0;
  uint64_t missing_input_lease_count = 0;
};

struct LazyChainState final {
  uint64_t next_chain_id = 1;
  uint64_t next_op_id = 1;
  uint64_t next_deferred_region_id = 1;
  uint64_t next_deferred_event_id = 1;
  uint64_t next_deferred_region_plan_id = 1;
  uint64_t next_deferred_tensor_handle_id = 1;
  uint64_t next_deferred_op_node_id = 1;
  uint64_t pending_deferred_event_count = 0;
  uint64_t deferred_value_lease_count = 0;
  uint64_t deferred_missing_value_lease_count = 0;
  uint64_t deferred_alias_or_view_count = 0;
  bool pending_deferred_value_access_boundary = false;
  std::string deferred_value_access_boundary_kind;
  std::string deferred_value_access_reason;
  std::string deferred_value_access_kind;
  std::string deferred_value_access_source_state;
  std::string deferred_value_access_destination_state;
  uint64_t deferred_value_access_vulkan_source_count = 0;
  uint64_t deferred_value_access_cpu_destination_count = 0;
  std::vector<std::string> ops;
  std::vector<std::string> raw_ops;
  std::vector<DeferredTensorHandleRecord> deferred_tensor_handles;
  std::vector<DeferredOpNodeRecord> deferred_op_nodes;
};

LazyChainState& lazy_chain_state() {
  thread_local LazyChainState state;
  return state;
}

bool has_prefix(const char* value, const char* prefix) {
  return std::strncmp(value, prefix, std::strlen(prefix)) == 0;
}

bool is_lazy_chain_internal_bookkeeping(const char* op_name) {
  return has_prefix(op_name, "aten::copy_.retire_after") ||
      has_prefix(op_name, "aten::copy_.release_retired_contexts");
}

bool match_deferred_bridge_op(
    const char* op_name,
    std::string& family,
    std::string& action) {
  struct BridgePrefix final {
    const char* prefix;
    const char* family;
  };
  static constexpr BridgePrefix kPrefixes[] = {
      {"aten::image_normalize_bridge.", "ImageNormalizeDeferredBridge"},
      {"aten::linear_gelu_bridge.", "LinearGeluDeferredBridge"},
      {"aten::add_layer_norm_bridge.", "AddLayerNormDeferredBridge"},
      {"aten::layer_scale_bridge.", "LayerScaleDeferredBridge"},
      {"aten::attention_query_scale_bridge.", "AttentionQueryScaleDeferredBridge"},
      {"aten::decomposed_attention_bridge.", "DecomposedAttentionDeferredBridge"},
  };
  for (const BridgePrefix& candidate : kPrefixes) {
    if (!has_prefix(op_name, candidate.prefix)) {
      continue;
    }
    family = candidate.family;
    action = op_name + std::strlen(candidate.prefix);
    const size_t space_pos = action.find(' ');
    if (space_pos != std::string::npos) {
      action.resize(space_pos);
    }
    return !action.empty();
  }
  return false;
}

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

void append_json_field(
    std::ostringstream& out,
    const char* name,
    const std::string& value,
    bool& first) {
  if (!first) {
    out << ',';
  }
  first = false;
  out << '"' << name << "\":\"" << json_escape(value) << '"';
}

void append_json_field(
    std::ostringstream& out,
    const char* name,
    const char* value,
    bool& first) {
  append_json_field(out, name, value ? std::string(value) : std::string(), first);
}

void append_json_uint(
    std::ostringstream& out,
    const char* name,
    const uint64_t value,
    bool& first) {
  if (!first) {
    out << ',';
  }
  first = false;
  out << '"' << name << "\":" << value;
}

void append_deferred_execution_log_line(const std::string& line) {
  if (!deferred_execution_logging_enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(deferred_execution_log_mutex());
  std::ofstream out(deferred_execution_log_path(), std::ios::app);
  out << line << '\n';
}

void append_deferred_region_plan_log_line(const std::string& line) {
  if (!deferred_region_plan_logging_enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(deferred_region_plan_log_mutex());
  std::ofstream out(deferred_region_plan_log_path(), std::ios::app);
  out << line << '\n';
}

void append_runtime_shader_compile_log_line(const std::string& line) {
  if (!runtime_shader_compile_logging_enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(runtime_shader_compile_log_mutex());
  std::ofstream out(runtime_shader_compile_log_path(), std::ios::app);
  out << line << '\n';
}

void append_runtime_command_list_log_line(const std::string& line) {
  if (!runtime_command_list_logging_enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(runtime_command_list_log_mutex());
  std::ofstream out(runtime_command_list_log_path(), std::ios::app);
  out << line << '\n';
}

bool has_string_prefix(const std::string& value, const char* prefix) {
  return value.rfind(prefix, 0) == 0;
}

std::string hash_to_hex(const std::string& value) {
  std::ostringstream out;
  out << std::hex << std::hash<std::string>{}(value);
  return out.str();
}

std::string quote_command_arg(const std::string& value) {
  std::string out = "\"";
  for (const char ch : value) {
    if (ch == '"') {
      out += "\\\"";
    } else {
      out += ch;
    }
  }
  out += "\"";
  return out;
}

std::string extract_submit_field(
    const std::string& op_name,
    const char* field_name) {
  const std::string marker = std::string(field_name) + "=";
  const size_t start = op_name.find(marker);
  if (start == std::string::npos) {
    return std::string();
  }
  const size_t value_start = start + marker.size();
  const size_t value_end = op_name.find(' ', value_start);
  return op_name.substr(
      value_start,
      value_end == std::string::npos ? std::string::npos
                                    : value_end - value_start);
}

bool supported_runtime_elementwise_chain_op(const std::string& op) {
  return op == "add" || op == "sub" || op == "mul" || op == "div" ||
      op == "floor_divide" || op == "pow";
}

bool supported_runtime_unary_chain_op(const std::string& op) {
  return op == "exp" || op == "sqrt" || op == "log" || op == "sin" ||
      op == "cos" || op == "neg" || op == "rsqrt" || op == "silu";
}

std::string runtime_chain_operand_summary(
    const uint64_t tensor_rhs_count,
    const uint64_t scalar_rhs_count,
    const uint64_t unary_count) {
  if (tensor_rhs_count > 0u && scalar_rhs_count == 0u) {
    return "tensor_buffer";
  }
  if (scalar_rhs_count > 0u && tensor_rhs_count == 0u) {
    return unary_count > 0u ? "scalar_unary" : "scalar";
  }
  if (tensor_rhs_count == 0u && scalar_rhs_count == 0u && unary_count > 0u) {
    return "unary";
  }
  return "mixed";
}

struct RuntimeElementwiseStep final {
  std::string op;
  std::string arity;
  std::string operand_kind;
};

struct RuntimeElementwiseChain {
  std::vector<RuntimeElementwiseStep> steps;
  uint64_t tensor_rhs_count = 0;
  uint64_t scalar_rhs_count = 0;
  uint64_t unary_count = 0;

  std::vector<std::string> ops() const {
    std::vector<std::string> out;
    out.reserve(steps.size());
    for (const RuntimeElementwiseStep& step : steps) {
      out.emplace_back(step.op);
    }
    return out;
  }

  std::vector<std::string> arities() const {
    std::vector<std::string> out;
    out.reserve(steps.size());
    for (const RuntimeElementwiseStep& step : steps) {
      out.emplace_back(step.arity);
    }
    return out;
  }

  std::vector<std::string> operand_kinds() const {
    std::vector<std::string> out;
    out.reserve(steps.size());
    for (const RuntimeElementwiseStep& step : steps) {
      out.emplace_back(step.operand_kind);
    }
    return out;
  }

  std::string operand_summary() const {
    return runtime_chain_operand_summary(
        tensor_rhs_count, scalar_rhs_count, unary_count);
  }
};

struct RuntimeDeviceCopyChain final {
  uint64_t raw_op_count = 0;
  uint64_t copy_count = 0;
  bool direct_transfer_only = true;
};

RuntimeElementwiseChain extract_elementwise_chain_ops(
    const std::vector<std::string>& raw_ops) {
  RuntimeElementwiseChain chain;
  bool pending_elementwise_buffer = false;
  for (const std::string& op_name : raw_ops) {
    if (has_string_prefix(op_name, "aten::unary.submit")) {
      const std::string route = extract_submit_field(op_name, "route");
      const std::string op = extract_submit_field(op_name, "op");
      if (route != "buffer" || !supported_runtime_unary_chain_op(op)) {
        continue;
      }
      chain.steps.push_back({op, "unary", "none"});
      ++chain.unary_count;
      pending_elementwise_buffer = false;
      continue;
    }
    if (
        op_name == "aten::binary_op.scalar_buffer_float" ||
        op_name == "aten::binary_op.buffer_float") {
      pending_elementwise_buffer = true;
      continue;
    }
    if (!pending_elementwise_buffer) {
      continue;
    }
    if (!has_string_prefix(op_name, "aten::binary_op.submit")) {
      pending_elementwise_buffer = false;
      continue;
    }
    const std::string route = extract_submit_field(op_name, "route");
    const std::string op = extract_submit_field(op_name, "op");
    pending_elementwise_buffer = false;
    const bool scalar_route = route == "scalar_buffer" ||
        (route == "tensor_buffer" &&
         op_name.find(" other=[] ") != std::string::npos);
    const bool tensor_route = route == "tensor_buffer" && !scalar_route;
    if (
        (!scalar_route && !tensor_route) ||
        !supported_runtime_elementwise_chain_op(op)) {
      continue;
    }
    chain.steps.push_back(
        {op, "binary", tensor_route ? "tensor_buffer" : "scalar"});
    if (tensor_route) {
      ++chain.tensor_rhs_count;
    } else {
      ++chain.scalar_rhs_count;
    }
  }
  return chain;
}

bool classify_runtime_binary_submit_op(
    const std::string& op_name,
    std::string* op_out,
    std::string* operand_kind_out) {
  if (!has_string_prefix(op_name, "aten::binary_op.submit")) {
    return false;
  }
  const std::string route = extract_submit_field(op_name, "route");
  const std::string op = extract_submit_field(op_name, "op");
  const bool scalar_route = route == "scalar_buffer" ||
      (route == "tensor_buffer" &&
       op_name.find(" other=[] ") != std::string::npos);
  const bool tensor_route = route == "tensor_buffer" && !scalar_route;
  if (
      (!scalar_route && !tensor_route) ||
      !supported_runtime_elementwise_chain_op(op)) {
    return false;
  }
  if (op_out != nullptr) {
    *op_out = op;
  }
  if (operand_kind_out != nullptr) {
    *operand_kind_out = tensor_route ? "tensor_buffer" : "scalar";
  }
  return true;
}

bool classify_runtime_unary_submit_op(
    const std::string& op_name,
    std::string* op_out) {
  if (!has_string_prefix(op_name, "aten::unary.submit")) {
    return false;
  }
  const std::string route = extract_submit_field(op_name, "route");
  const std::string op = extract_submit_field(op_name, "op");
  if (route != "buffer" || !supported_runtime_unary_chain_op(op)) {
    return false;
  }
  if (op_out != nullptr) {
    *op_out = op;
  }
  return true;
}

bool runtime_elementwise_raw_chain_complete(
    const std::vector<std::string>& raw_ops) {
  for (size_t idx = 0; idx < raw_ops.size();) {
    if (classify_runtime_unary_submit_op(raw_ops[idx], nullptr)) {
      ++idx;
      continue;
    }
    if (
        raw_ops[idx] == "aten::binary_op.scalar_buffer_float" ||
        raw_ops[idx] == "aten::binary_op.buffer_float") {
      if (idx + 1 >= raw_ops.size() ||
          !classify_runtime_binary_submit_op(raw_ops[idx + 1], nullptr, nullptr)) {
        return false;
      }
      idx += 2;
      continue;
    }
    return false;
  }
  return true;
}

RuntimeDeviceCopyChain extract_device_copy_chain_ops(
    const std::vector<std::string>& raw_ops) {
  RuntimeDeviceCopyChain chain;
  for (const std::string& op_name : raw_ops) {
    if (has_string_prefix(op_name, "aten::copy_.buffer_to_buffer_submit")) {
      ++chain.raw_op_count;
      ++chain.copy_count;
      if (op_name.find(" path=direct_transfer ") == std::string::npos) {
        chain.direct_transfer_only = false;
      }
      continue;
    }
    if (op_name == "aten::copy_.buffer_to_buffer") {
      ++chain.raw_op_count;
      continue;
    }
    return RuntimeDeviceCopyChain{};
  }
  return chain;
}

std::string runtime_raw_op_token(const std::string& op_name) {
  const size_t end = op_name.find(' ');
  return op_name.substr(
      0,
      end == std::string::npos ? std::string::npos : end);
}

std::string runtime_command_for_raw_op(const std::string& op_name) {
  if (op_name.find("token_prefix_cat_add") != std::string::npos) {
    return "dispatch_existing_token_prefix_cat_add";
  }
  if (has_string_prefix(op_name, "aten::convolution.submit")) {
    return "dispatch_existing_convolution";
  }
  if (has_string_prefix(op_name, "aten::convolution.buffer_float_prepack")) {
    return "prepare_existing_convolution_prepack";
  }
  if (
      has_string_prefix(op_name, "aten::linear.") ||
      has_string_prefix(op_name, "vulkan_prepack::run_linear_context")) {
    return "dispatch_existing_linear";
  }
  if (
      has_string_prefix(op_name, "aten::bmm") ||
      op_name.find("scaled_dot_product_attention") != std::string::npos) {
    return "dispatch_existing_attention_or_bmm";
  }
  if (op_name.find("softmax") != std::string::npos) {
    return "dispatch_existing_softmax";
  }
  if (op_name.find("gelu") != std::string::npos) {
    return "dispatch_existing_gelu";
  }
  if (
      op_name.find("layer_norm") != std::string::npos ||
      op_name.find("native_layer_norm") != std::string::npos) {
    return "dispatch_existing_norm";
  }
  if (op_name.find("feature_map_to_tokens") != std::string::npos) {
    return "dispatch_feature_map_to_tokens";
  }
  if (op_name.find("upsample") != std::string::npos) {
    return "dispatch_existing_upsample";
  }
  if (op_name.find("cat") != std::string::npos) {
    return "dispatch_existing_cat";
  }
  if (
      has_string_prefix(op_name, "aten::view") ||
      has_string_prefix(op_name, "aten::_reshape_alias")) {
    return "apply_view_or_materialization";
  }
  if (has_string_prefix(op_name, "aten::copy_")) {
    return "record_copy_or_transfer";
  }
  if (
      has_string_prefix(op_name, "aten::binary_op") ||
      has_string_prefix(op_name, "aten::unary")) {
    return "dispatch_existing_elementwise";
  }
  return "record_existing_vulkan_op";
}

std::string runtime_multi_op_family(const std::vector<std::string>& raw_ops) {
  bool has_convolution = false;
  bool has_prepack = false;
  bool has_patch_tokens = false;
  bool has_upsample = false;
  bool has_pointwise = false;
  bool has_attention = false;
  bool has_softmax = false;
  bool has_linear = false;
  bool has_gelu = false;
  bool has_norm = false;
  bool has_token_prefix = false;
  bool has_cat = false;
  bool has_residual_norm = false;
  for (const std::string& op_name : raw_ops) {
    has_convolution =
        has_convolution || has_string_prefix(op_name, "aten::convolution.");
    has_prepack = has_prepack ||
        has_string_prefix(op_name, "aten::convolution.buffer_float_prepack");
    has_patch_tokens = has_patch_tokens ||
        op_name.find("patch_embed_feature_map_to_tokens") != std::string::npos ||
        op_name.find("feature_map_to_tokens") != std::string::npos;
    has_upsample = has_upsample || op_name.find("upsample") != std::string::npos;
    has_pointwise = has_pointwise ||
        op_name.find("pointwise_route") != std::string::npos ||
        op_name.find("buffer_float_1x1") != std::string::npos;
    has_attention = has_attention ||
        op_name.find("scaled_dot_product_attention") != std::string::npos ||
        has_string_prefix(op_name, "aten::bmm");
    has_softmax = has_softmax || op_name.find("softmax") != std::string::npos;
    has_linear = has_linear ||
        has_string_prefix(op_name, "aten::linear.") ||
        has_string_prefix(op_name, "vulkan_prepack::run_linear_context");
    has_gelu = has_gelu || op_name.find("gelu") != std::string::npos;
    has_norm = has_norm || op_name.find("layer_norm") != std::string::npos ||
        op_name.find("native_layer_norm") != std::string::npos;
    has_token_prefix =
        has_token_prefix || op_name.find("token_prefix_cat_add") != std::string::npos;
    has_cat = has_cat || op_name.find("cat") != std::string::npos;
    has_residual_norm = has_residual_norm ||
        op_name.find("add_layer_norm") != std::string::npos ||
        op_name.find("add_scaled_layer_norm") != std::string::npos;
  }
  if (has_token_prefix) {
    return "TokenPrefixBackboneCommandListRegion";
  }
  if (has_convolution && has_upsample && has_cat) {
    return "DecoderConvUpsampleCatCommandListRegion";
  }
  if (has_patch_tokens && has_upsample) {
    return "VisionPatchTokenPrepCommandListRegion";
  }
  if (has_patch_tokens || (has_convolution && has_patch_tokens)) {
    return "PatchEmbedFeatureMapToTokensCommandListRegion";
  }
  if (has_convolution && has_prepack) {
    return "ConvPrepackUploadCommandListRegion";
  }
  if (has_pointwise && has_upsample) {
    return "PointwiseUpsampleCommandListRegion";
  }
  if (has_attention && (has_softmax || has_linear || has_norm)) {
    return "TransformerBlockCommandListRegion";
  }
  if (has_linear && has_gelu) {
    return "LinearGeluMlpCommandListRegion";
  }
  if (has_residual_norm || (has_norm && has_cat)) {
    return "ResidualNormCommandListRegion";
  }
  if (has_upsample) {
    return "UpsampleCommandListRegion";
  }
  return "ObservedMultiOpCommandListRegion";
}

std::vector<std::string> runtime_multi_op_subfamily_tags(
    const std::vector<std::string>& raw_ops) {
  std::vector<std::string> tags;
  bool has_elementwise = false;
  bool has_copy = false;
  bool has_convolution = false;
  bool has_prepack = false;
  bool has_patch_tokens = false;
  bool has_upsample = false;
  bool has_attention = false;
  bool has_softmax = false;
  bool has_linear = false;
  bool has_gelu = false;
  bool has_norm = false;
  bool has_cat = false;
  bool has_token_prefix = false;
  for (const std::string& op_name : raw_ops) {
    has_elementwise = has_elementwise ||
        has_string_prefix(op_name, "aten::binary_op") ||
        has_string_prefix(op_name, "aten::unary");
    has_copy = has_copy || has_string_prefix(op_name, "aten::copy_");
    has_convolution =
        has_convolution || has_string_prefix(op_name, "aten::convolution.");
    has_prepack = has_prepack ||
        has_string_prefix(op_name, "aten::convolution.buffer_float_prepack");
    has_patch_tokens = has_patch_tokens ||
        op_name.find("patch_embed_feature_map_to_tokens") != std::string::npos ||
        op_name.find("feature_map_to_tokens") != std::string::npos;
    has_upsample = has_upsample || op_name.find("upsample") != std::string::npos;
    has_attention = has_attention ||
        op_name.find("scaled_dot_product_attention") != std::string::npos ||
        has_string_prefix(op_name, "aten::bmm");
    has_softmax = has_softmax || op_name.find("softmax") != std::string::npos;
    has_linear = has_linear ||
        has_string_prefix(op_name, "aten::linear.") ||
        has_string_prefix(op_name, "vulkan_prepack::run_linear_context");
    has_gelu = has_gelu || op_name.find("gelu") != std::string::npos;
    has_norm = has_norm || op_name.find("layer_norm") != std::string::npos ||
        op_name.find("native_layer_norm") != std::string::npos;
    has_cat = has_cat || op_name.find("cat") != std::string::npos;
    has_token_prefix =
        has_token_prefix || op_name.find("token_prefix_cat_add") != std::string::npos;
  }
  if (has_elementwise) {
    tags.emplace_back("contains_elementwise");
  }
  if (has_copy) {
    tags.emplace_back("contains_copy_or_transfer");
  }
  if (has_convolution) {
    tags.emplace_back("contains_convolution");
  }
  if (has_prepack) {
    tags.emplace_back("contains_conv_prepack");
  }
  if (has_patch_tokens) {
    tags.emplace_back("contains_patch_or_token_layout");
  }
  if (has_upsample) {
    tags.emplace_back("contains_upsample");
  }
  if (has_attention) {
    tags.emplace_back("contains_attention_or_bmm");
  }
  if (has_softmax) {
    tags.emplace_back("contains_softmax");
  }
  if (has_linear) {
    tags.emplace_back("contains_linear");
  }
  if (has_gelu) {
    tags.emplace_back("contains_gelu");
  }
  if (has_norm) {
    tags.emplace_back("contains_norm");
  }
  if (has_cat) {
    tags.emplace_back("contains_cat");
  }
  if (has_token_prefix) {
    tags.emplace_back("contains_token_prefix");
  }
  return tags;
}

std::string runtime_elementwise_chain_glsl(
    const RuntimeElementwiseChain& chain) {
  std::ostringstream glsl;
  glsl << "#version 450\n"
       << "layout(local_size_x_id = 0, local_size_y_id = 1, "
          "local_size_z_id = 2) in;\n"
       << "layout(set = 0, binding = 0) buffer OutBuffer { float out_data[]; };\n"
       << "layout(set = 0, binding = 1) readonly buffer InBuffer { "
          "float in_data[]; };\n";
  uint64_t tensor_rhs_idx = 0;
  for (const RuntimeElementwiseStep& step : chain.steps) {
    if (step.operand_kind == "tensor_buffer") {
      glsl << "layout(set = 0, binding = " << (tensor_rhs_idx + 2)
           << ") readonly buffer RhsBuffer" << tensor_rhs_idx
           << " { float rhs_data" << tensor_rhs_idx << "[]; };\n";
      ++tensor_rhs_idx;
    }
  }
  const uint64_t params_binding = chain.tensor_rhs_count + 2;
  glsl << "layout(set = 0, binding = " << params_binding
       << ") uniform Params {\n"
       << "  uint numel;\n";
  if (chain.scalar_rhs_count > 0u) {
    glsl << "  float scalars[" << chain.scalar_rhs_count << "];\n";
  }
  glsl << "} params;\n";
  glsl << "\n"
       << "void main() {\n"
       << "  uint idx = gl_GlobalInvocationID.x;\n"
       << "  if (idx >= params.numel) {\n"
       << "    return;\n"
       << "  }\n"
       << "  float value = in_data[idx];\n";
  tensor_rhs_idx = 0;
  uint64_t scalar_rhs_idx = 0;
  for (const RuntimeElementwiseStep& step : chain.steps) {
    const std::string& op = step.op;
    if (step.arity == "unary") {
      if (op == "exp") {
        glsl << "  value = exp(value);\n";
      } else if (op == "sqrt") {
        glsl << "  value = sqrt(value);\n";
      } else if (op == "log") {
        glsl << "  value = log(value);\n";
      } else if (op == "sin") {
        glsl << "  value = sin(value);\n";
      } else if (op == "cos") {
        glsl << "  value = cos(value);\n";
      } else if (op == "neg") {
        glsl << "  value = -value;\n";
      } else if (op == "rsqrt") {
        glsl << "  value = inversesqrt(value);\n";
      } else if (op == "silu") {
        glsl << "  value = value / (1.0 + exp(-value));\n";
      }
      continue;
    }
    std::string rhs;
    if (step.operand_kind == "tensor_buffer") {
      rhs = "rhs_data" + std::to_string(tensor_rhs_idx++) + "[idx]";
    } else {
      rhs = "params.scalars[" + std::to_string(scalar_rhs_idx++) + "]";
    }
    if (op == "add") {
      glsl << "  value = value + " << rhs << ";\n";
    } else if (op == "sub") {
      glsl << "  value = value - " << rhs << ";\n";
    } else if (op == "mul") {
      glsl << "  value = value * " << rhs << ";\n";
    } else if (op == "div") {
      glsl << "  value = value / " << rhs << ";\n";
    } else if (op == "floor_divide") {
      glsl << "  value = floor(value / " << rhs << ");\n";
    } else if (op == "pow") {
      glsl << "  value = pow(value, " << rhs << ");\n";
    }
  }
  glsl << "  out_data[idx] = value;\n"
       << "}\n";
  return glsl.str();
}

bool spv_header_looks_valid(const std::filesystem::path& spv_path) {
  std::ifstream in(spv_path, std::ios::binary);
  if (!in) {
    return false;
  }
  uint32_t words[2] = {0u, 0u};
  in.read(reinterpret_cast<char*>(words), sizeof(words));
  return in.gcount() == static_cast<std::streamsize>(sizeof(words)) &&
      words[0] == 0x07230203u && words[1] == 0x00010600u;
}

void append_json_string_array(
    std::ostringstream& out,
    const char* name,
    const std::vector<std::string>& values,
    bool& first) {
  if (!first) {
    out << ',';
  }
  first = false;
  out << '"' << name << "\":[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0u) {
      out << ',';
    }
    out << '"' << json_escape(values[i]) << '"';
  }
  out << ']';
}

void log_runtime_shader_compile_event(
    const char* status,
    const char* boundary_kind,
    const char* reason,
    const std::string& group_key,
    const RuntimeElementwiseChain& chain,
    const std::filesystem::path& glsl_path,
    const std::filesystem::path& spv_path,
    const int compile_exit_code,
    const bool cache_hit,
    const bool spv_valid) {
  std::ostringstream line;
  bool first = true;
  line << '{';
  append_json_field(line, "schema", "VulkanRuntimeShaderCompileTrace.v0", first);
  append_json_field(line, "event", "runtime_shader_compile", first);
  append_json_field(line, "status", status, first);
  append_json_field(line, "family", "ElementwiseChain", first);
  append_json_field(line, "group_key", group_key, first);
  append_json_uint(line, "op_count", chain.steps.size(), first);
  append_json_string_array(line, "ops", chain.ops(), first);
  append_json_string_array(line, "op_arities", chain.arities(), first);
  append_json_string_array(line, "op_operand_kinds", chain.operand_kinds(), first);
  append_json_uint(line, "tensor_rhs_count", chain.tensor_rhs_count, first);
  append_json_uint(line, "scalar_rhs_count", chain.scalar_rhs_count, first);
  append_json_uint(line, "unary_count", chain.unary_count, first);
  append_json_field(line, "operand_kind", chain.operand_summary(), first);
  append_json_field(
      line,
      "boundary_kind",
      boundary_kind ? boundary_kind : "unknown_boundary",
      first);
  append_json_field(line, "reason", reason ? reason : "unknown", first);
  append_json_field(line, "glsl_path", glsl_path.string(), first);
  append_json_field(line, "spv_path", spv_path.string(), first);
  append_json_field(line, "glslc", runtime_shader_glslc_path(), first);
  append_json_uint(line, "compile_exit_code", compile_exit_code, first);
  append_json_uint(line, "cache_hit", cache_hit ? 1u : 0u, first);
  append_json_uint(line, "spv_valid", spv_valid ? 1u : 0u, first);
  append_json_field(line, "behavior_change", "0", first);
  line << '}';
  append_runtime_shader_compile_log_line(line.str());
}

std::string runtime_elementwise_chain_group_key(
    const RuntimeElementwiseChain& chain) {
  std::ostringstream key_stream;
  key_stream << "ElementwiseChain:"
             << chain.operand_summary();
  for (const RuntimeElementwiseStep& step : chain.steps) {
    key_stream << ':' << step.arity << ':' << step.operand_kind << ':'
               << step.op;
  }
  return key_stream.str();
}

void log_runtime_command_list_plan_event(
    const char* boundary_kind,
    const char* reason,
    const std::string& group_key,
    const RuntimeElementwiseChain& chain) {
  std::vector<std::string> descriptor_slots = {
      "binding0:storage_buffer_write:output",
      "binding1:storage_buffer_read:input0",
  };
  if (chain.tensor_rhs_count > 0u) {
    for (uint64_t idx = 0; idx < chain.tensor_rhs_count; ++idx) {
      descriptor_slots.emplace_back(
          "binding" + std::to_string(idx + 2) +
          ":storage_buffer_read:rhs" + std::to_string(idx));
    }
  }
  const size_t params_binding = descriptor_slots.size();
  descriptor_slots.emplace_back(
      "binding" + std::to_string(params_binding) + ":uniform_buffer:params");
  const uint64_t descriptor_count = descriptor_slots.size();

  std::vector<std::string> params = {"numel:uint32"};
  if (chain.scalar_rhs_count > 0u) {
    params.emplace_back(
        "scalars:float32[" + std::to_string(chain.scalar_rhs_count) + "]");
  }

  const std::vector<std::string> barriers = {
      "before_dispatch:inputs=shader_read",
      "before_dispatch:output=shader_write",
  };
  const std::vector<std::string> commands = {
      "bind_compute_pipeline",
      "bind_descriptor_set",
      "bind_params_uniform_buffer",
      "dispatch_numel_1d",
  };
  const std::vector<std::string> missing_prerequisites = {
      "deferred_tensor_handle_capture",
      "generated_output_allocation",
      "runtime_shader_executor_hook",
      "params_uniform_buffer_executor",
      "alias_escape_proof",
  };

  std::ostringstream line;
  bool first = true;
  line << '{';
  append_json_field(line, "schema", "VulkanRuntimeCommandListPlanTrace.v0", first);
  append_json_field(line, "event", "runtime_command_list_plan", first);
  append_json_field(line, "status", "planned_not_executed", first);
  append_json_field(line, "family", "ElementwiseChain", first);
  append_json_field(line, "group_key", group_key, first);
  append_json_uint(line, "op_count", chain.steps.size(), first);
  append_json_string_array(line, "ops", chain.ops(), first);
  append_json_string_array(line, "op_arities", chain.arities(), first);
  append_json_string_array(line, "op_operand_kinds", chain.operand_kinds(), first);
  append_json_uint(line, "tensor_rhs_count", chain.tensor_rhs_count, first);
  append_json_uint(line, "scalar_rhs_count", chain.scalar_rhs_count, first);
  append_json_uint(line, "unary_count", chain.unary_count, first);
  append_json_field(line, "operand_kind", chain.operand_summary(), first);
  append_json_field(
      line,
      "boundary_kind",
      boundary_kind ? boundary_kind : "unknown_boundary",
      first);
  append_json_field(line, "reason", reason ? reason : "unknown", first);
  append_json_field(line, "program_key", group_key, first);
  append_json_field(line, "cache_key", group_key, first);
  append_json_field(line, "shader_family", "runtime_generated_glsl", first);
  append_json_field(line, "layout_contract", "direct_dense_buffer_required", first);
  append_json_field(
      line,
      "descriptor_layout_signature",
      "raw_dense_elementwise_params_ubo_v0",
      first);
  append_json_field(line, "shape_policy", "runtime_numel_params_ubo", first);
  append_json_field(line, "dispatch_geometry", "global=[numel,1,1]", first);
  append_json_field(line, "local_size_policy", "specialized_or_default_1d", first);
  append_json_uint(line, "descriptor_set_count", 1, first);
  append_json_uint(line, "descriptor_binding_count", descriptor_count, first);
  append_json_string_array(line, "descriptor_slots", descriptor_slots, first);
  append_json_field(line, "params_binding", "uniform_buffer_required", first);
  append_json_uint(line, "push_constants_supported", 0, first);
  append_json_string_array(line, "params", params, first);
  append_json_string_array(line, "barriers", barriers, first);
  append_json_string_array(line, "commands", commands, first);
  append_json_string_array(
      line, "missing_execution_prerequisites", missing_prerequisites, first);
  append_json_uint(line, "requires_deferred_tensor_handles", 1, first);
  append_json_uint(line, "requires_output_allocation", 1, first);
  append_json_uint(line, "requires_alias_escape_proof", 1, first);
  append_json_uint(line, "execution_enabled", 0, first);
  append_json_field(line, "behavior_change", "0", first);
  line << '}';
  append_runtime_command_list_log_line(line.str());
}

void log_runtime_device_copy_command_list_plan_event(
    const char* boundary_kind,
    const char* reason,
    const std::string& group_key,
    const RuntimeDeviceCopyChain& chain) {
  const std::vector<std::string> commands = {
      "insert_copy_read_barrier",
      "copy_buffer_to_buffer",
      "insert_copy_write_barrier",
  };
  const std::vector<std::string> missing_prerequisites = {
      "deferred_tensor_handle_capture",
      "source_destination_identity_capture",
      "copy_command_executor_hook",
      "alias_escape_proof",
  };

  std::ostringstream line;
  bool first = true;
  line << '{';
  append_json_field(line, "schema", "VulkanRuntimeCommandListPlanTrace.v0", first);
  append_json_field(line, "event", "runtime_command_list_plan", first);
  append_json_field(line, "status", "planned_not_executed", first);
  append_json_field(line, "family", "DeviceCopyChain", first);
  append_json_field(line, "group_key", group_key, first);
  append_json_uint(line, "op_count", chain.raw_op_count, first);
  append_json_uint(line, "copy_count", chain.copy_count, first);
  append_json_uint(
      line, "direct_transfer_only", chain.direct_transfer_only ? 1u : 0u, first);
  append_json_field(line, "operand_kind", "buffer_to_buffer", first);
  append_json_field(
      line,
      "boundary_kind",
      boundary_kind ? boundary_kind : "unknown_boundary",
      first);
  append_json_field(line, "reason", reason ? reason : "unknown", first);
  append_json_field(line, "program_key", group_key, first);
  append_json_field(line, "cache_key", group_key, first);
  append_json_field(line, "shader_family", "none_copy_command_list", first);
  append_json_field(line, "layout_contract", "buffer_copy_required", first);
  append_json_field(line, "shape_policy", "runtime_byte_range", first);
  append_json_uint(line, "descriptor_set_count", 0, first);
  append_json_uint(line, "descriptor_binding_count", 0, first);
  append_json_string_array(line, "commands", commands, first);
  append_json_string_array(
      line, "missing_execution_prerequisites", missing_prerequisites, first);
  append_json_uint(line, "requires_deferred_tensor_handles", 1, first);
  append_json_uint(line, "requires_output_allocation", 0, first);
  append_json_uint(line, "requires_alias_escape_proof", 1, first);
  append_json_uint(line, "execution_enabled", 0, first);
  append_json_field(line, "behavior_change", "0", first);
  line << '}';
  append_runtime_command_list_log_line(line.str());
}

void log_runtime_multi_op_command_list_plan_event(
    const char* boundary_kind,
    const char* reason,
    const std::vector<std::string>& raw_ops) {
  constexpr size_t kMaxLoggedCommandSequence = 128;
  std::vector<std::string> op_tokens;
  std::vector<std::string> commands;
  op_tokens.reserve(std::min(raw_ops.size(), kMaxLoggedCommandSequence));
  commands.reserve(std::min(raw_ops.size(), kMaxLoggedCommandSequence));
  for (size_t idx = 0;
       idx < raw_ops.size() && idx < kMaxLoggedCommandSequence;
       ++idx) {
    op_tokens.emplace_back(runtime_raw_op_token(raw_ops[idx]));
    commands.emplace_back(runtime_command_for_raw_op(raw_ops[idx]));
  }

  const std::string family = runtime_multi_op_family(raw_ops);
  const std::vector<std::string> subfamily_tags =
      runtime_multi_op_subfamily_tags(raw_ops);
  const std::string group_key =
      family + ":ops=" + std::to_string(raw_ops.size());
  const std::vector<std::string> missing_prerequisites = {
      "deferred_tensor_handle_capture",
      "producer_consumer_edge_capture",
      "descriptor_binding_capture",
      "barrier_plan_executor",
      "region_output_ownership",
      "alias_escape_proof",
  };

  std::ostringstream line;
  bool first = true;
  line << '{';
  append_json_field(line, "schema", "VulkanRuntimeCommandListPlanTrace.v0", first);
  append_json_field(line, "event", "runtime_command_list_plan", first);
  append_json_field(line, "status", "planned_not_executed", first);
  append_json_field(line, "family", family, first);
  append_json_field(line, "group_key", group_key, first);
  append_json_uint(line, "op_count", raw_ops.size(), first);
  append_json_uint(line, "logged_command_count", commands.size(), first);
  append_json_uint(
      line,
      "command_sequence_truncated",
      raw_ops.size() > kMaxLoggedCommandSequence ? 1u : 0u,
      first);
  append_json_string_array(line, "op_tokens", op_tokens, first);
  append_json_string_array(line, "subfamily_tags", subfamily_tags, first);
  append_json_field(line, "operand_kind", "multi_dispatch_region", first);
  append_json_field(
      line,
      "boundary_kind",
      boundary_kind ? boundary_kind : "unknown_boundary",
      first);
  append_json_field(line, "reason", reason ? reason : "unknown", first);
  append_json_field(line, "program_key", group_key, first);
  append_json_field(line, "cache_key", group_key, first);
  append_json_field(line, "shader_family", "multi_dispatch_existing_kernels", first);
  append_json_field(line, "layout_contract", "per_op_contracts_required", first);
  append_json_field(line, "shape_policy", "runtime_descriptors_per_op", first);
  append_json_uint(line, "descriptor_set_count", 0, first);
  append_json_uint(line, "descriptor_binding_count", 0, first);
  append_json_string_array(line, "commands", commands, first);
  append_json_string_array(
      line, "missing_execution_prerequisites", missing_prerequisites, first);
  append_json_uint(line, "requires_deferred_tensor_handles", 1, first);
  append_json_uint(line, "requires_output_allocation", 1, first);
  append_json_uint(line, "requires_alias_escape_proof", 1, first);
  append_json_uint(line, "execution_enabled", 0, first);
  append_json_field(line, "behavior_change", "0", first);
  line << '}';
  append_runtime_command_list_log_line(line.str());
}

void maybe_log_runtime_command_list_plan(
    const LazyChainState& state,
    const char* boundary_kind,
    const char* reason) {
  if (!runtime_command_list_logging_enabled() || state.raw_ops.empty()) {
    return;
  }

  const RuntimeDeviceCopyChain copy_chain =
      extract_device_copy_chain_ops(state.raw_ops);
  if (copy_chain.copy_count > 0u) {
    const std::string group_key =
        copy_chain.direct_transfer_only
        ? "DeviceCopyChain:buffer_to_buffer_direct_transfer"
        : "DeviceCopyChain:buffer_to_buffer";
    log_runtime_device_copy_command_list_plan_event(
        boundary_kind, reason, group_key, copy_chain);
    return;
  }

  if (!runtime_elementwise_raw_chain_complete(state.raw_ops)) {
    if (state.raw_ops.size() < 2u) {
      return;
    }
    log_runtime_multi_op_command_list_plan_event(
        boundary_kind, reason, state.raw_ops);
    return;
  }

  const RuntimeElementwiseChain chain = extract_elementwise_chain_ops(state.raw_ops);
  if (chain.steps.size() < 2u) {
    return;
  }

  log_runtime_command_list_plan_event(
      boundary_kind,
      reason,
      runtime_elementwise_chain_group_key(chain),
      chain);
}

void maybe_compile_runtime_shader_group(
    const LazyChainState& state,
    const char* boundary_kind,
    const char* reason) {
  if (!runtime_shader_compile_logging_enabled() || state.raw_ops.size() < 2u) {
    return;
  }

  if (!runtime_elementwise_raw_chain_complete(state.raw_ops)) {
    return;
  }

  const RuntimeElementwiseChain chain = extract_elementwise_chain_ops(state.raw_ops);
  if (chain.steps.size() < 2u) {
    return;
  }

  const std::string group_key = runtime_elementwise_chain_group_key(chain);
  const std::string key_hash = hash_to_hex(group_key);

  std::filesystem::path cache_dir(runtime_shader_cache_dir());
  std::filesystem::path glsl_path =
      cache_dir / ("runtime_elementwise_chain_" + key_hash + ".glsl");
  std::filesystem::path spv_path =
      cache_dir / ("runtime_elementwise_chain_" + key_hash + ".spv");

  bool cache_hit = false;
  {
    std::lock_guard<std::mutex> lock(runtime_shader_compile_cache_mutex());
    auto& cache = runtime_shader_compile_cache();
    cache_hit = cache.find(group_key) != cache.end();
    if (!cache_hit) {
      cache.insert(group_key);
    }
  }
  if (cache_hit && std::filesystem::exists(glsl_path)) {
    log_runtime_shader_compile_event(
        "cache_hit",
        boundary_kind,
        reason,
        group_key,
        chain,
        glsl_path,
        spv_path,
        0,
        true,
        std::filesystem::exists(spv_path) && spv_header_looks_valid(spv_path));
    return;
  }

  std::error_code ec;
  std::filesystem::create_directories(cache_dir, ec);
  if (ec) {
    log_runtime_shader_compile_event(
        "cache_dir_error",
        boundary_kind,
        reason,
        group_key,
        chain,
        glsl_path,
        spv_path,
        -1,
        false,
        false);
    return;
  }

  {
    std::ofstream glsl(glsl_path);
    glsl << runtime_elementwise_chain_glsl(chain);
  }

  int compile_exit_code = 0;
  bool spv_valid = false;
  const std::string& glslc_path = runtime_shader_glslc_path();
  const char* status = "generated_glsl_only";
  if (!glslc_path.empty()) {
    std::ostringstream command;
    command << quote_command_arg(glslc_path)
            << " -fshader-stage=compute "
            << quote_command_arg(glsl_path.string())
            << " -o " << quote_command_arg(spv_path.string())
            << " --target-env=vulkan1.3 --target-spv=spv1.6 -Werror";
#ifdef _WIN32
    std::string shell_command = "cmd.exe /S /C \"";
    shell_command += command.str();
    shell_command += "\"";
    compile_exit_code = std::system(shell_command.c_str());
#else
    compile_exit_code = std::system(command.str().c_str());
#endif
    spv_valid = compile_exit_code == 0 && std::filesystem::exists(spv_path) &&
        spv_header_looks_valid(spv_path);
    status = spv_valid ? "compiled_spv" : "compile_failed";
  }

  log_runtime_shader_compile_event(
      status,
      boundary_kind,
      reason,
      group_key,
      chain,
      glsl_path,
      spv_path,
      compile_exit_code,
      false,
      spv_valid);
}

void log_deferred_bridge_event(
    const char* op_name,
    const std::string& family,
    const std::string& action) {
  if (!deferred_execution_logging_enabled()) {
    return;
  }
  LazyChainState& state = lazy_chain_state();
  const uint64_t region_id = state.next_deferred_region_id;
  ++state.pending_deferred_event_count;

  std::ostringstream line;
  bool first = true;
  line << '{';
  append_json_field(line, "schema", "VulkanDeferredExecutionTrace.v0", first);
  append_json_field(line, "event", "deferred_bridge_event", first);
  append_json_uint(line, "region_id", region_id, first);
  append_json_uint(line, "event_id", state.next_deferred_event_id++, first);
  append_json_uint(
      line,
      "pending_deferred_event_count",
      state.pending_deferred_event_count,
      first);
  append_json_uint(
      line,
      "pending_lazy_chain_op_count",
      static_cast<uint64_t>(state.ops.size()),
      first);
  append_json_field(line, "family", family, first);
  append_json_field(line, "action", action, first);
  append_json_field(line, "op", op_name, first);
  append_json_field(
      line,
      "submit_phase",
      submit_phase_name(current_submit_phase()),
      first);
  append_json_field(
      line,
      "recent_op",
      recent_op_label().empty() ? "none" : recent_op_label(),
      first);
  if (!current_allocation_label().empty()) {
    append_json_field(line, "caller", current_allocation_label(), first);
  }
  if (!current_runtime_label().empty()) {
    append_json_field(line, "runtime", current_runtime_label(), first);
  }
  line << '}';
  append_deferred_execution_log_line(line.str());
}

void maybe_log_deferred_bridge_event(const char* op_name) {
  if (!deferred_execution_logging_enabled()) {
    return;
  }
  std::string family;
  std::string action;
  if (!match_deferred_bridge_op(op_name, family, action)) {
    return;
  }
  log_deferred_bridge_event(op_name, family, action);
}

void flush_deferred_execution_region(
    LazyChainState& state,
    const char* boundary_kind,
    const char* reason) {
  if (!deferred_execution_logging_enabled()) {
    return;
  }
  if (state.pending_deferred_event_count == 0) {
    return;
  }

  std::ostringstream line;
  bool first = true;
  line << '{';
  append_json_field(line, "schema", "VulkanDeferredExecutionTrace.v0", first);
  append_json_field(line, "event", "deferred_region_flush", first);
  append_json_uint(line, "region_id", state.next_deferred_region_id++, first);
  append_json_uint(line, "event_id", state.next_deferred_event_id++, first);
  append_json_uint(
      line,
      "pending_deferred_event_count",
      state.pending_deferred_event_count,
      first);
  append_json_uint(
      line,
      "pending_lazy_chain_op_count",
      static_cast<uint64_t>(state.ops.size()),
      first);
  append_json_field(
      line,
      "boundary_kind",
      boundary_kind ? boundary_kind : "unknown_boundary",
      first);
  append_json_field(line, "reason", reason ? reason : "unknown", first);
  append_json_field(
      line,
      "submit_phase",
      submit_phase_name(current_submit_phase()),
      first);
  append_json_field(
      line,
      "recent_op",
      recent_op_label().empty() ? "none" : recent_op_label(),
      first);
  append_json_field(line, "execution_mode", "observe_existing_bridges", first);
  append_json_field(line, "behavior_change", "0", first);
  line << '}';
  append_deferred_execution_log_line(line.str());
  state.pending_deferred_event_count = 0;
}

void clear_deferred_region_plan_state(LazyChainState& state) {
  state.deferred_tensor_handles.clear();
  state.deferred_op_nodes.clear();
  state.deferred_value_lease_count = 0;
  state.deferred_missing_value_lease_count = 0;
  state.deferred_alias_or_view_count = 0;
  state.pending_deferred_value_access_boundary = false;
  state.deferred_value_access_boundary_kind.clear();
  state.deferred_value_access_reason.clear();
  state.deferred_value_access_kind.clear();
  state.deferred_value_access_source_state.clear();
  state.deferred_value_access_destination_state.clear();
  state.deferred_value_access_vulkan_source_count = 0;
  state.deferred_value_access_cpu_destination_count = 0;
}

void flush_deferred_region_plan(
    LazyChainState& state,
    const char* boundary_kind,
    const char* reason) {
  if (!deferred_region_plan_logging_enabled()) {
    return;
  }
  if (state.deferred_op_nodes.empty()) {
    clear_deferred_region_plan_state(state);
    return;
  }

  std::vector<std::string> op_names;
  std::vector<std::string> route_names;
  op_names.reserve(state.deferred_op_nodes.size());
  route_names.reserve(state.deferred_op_nodes.size());
  for (const DeferredOpNodeRecord& node : state.deferred_op_nodes) {
    op_names.emplace_back(node.op);
    route_names.emplace_back(node.route);
  }

  std::vector<std::string> missing_prerequisites = {
      "barrier_plan_executor",
      "deferred_region_executor",
      "generated_output_allocation",
      "region_output_ownership",
      "runtime_shader_or_command_list_lowering",
      "value_lifetime_validation",
  };
  std::string top_blocker = "missing_deferred_region_executor";
  if (state.deferred_missing_value_lease_count > 0u) {
    top_blocker = "missing_value_lease_capture";
    missing_prerequisites.emplace_back("complete_value_lease_capture");
  }
  if (state.deferred_alias_or_view_count > 0u) {
    if (top_blocker == "missing_deferred_region_executor") {
      top_blocker = "alias_or_view_escape_proof_required";
    }
    missing_prerequisites.emplace_back("alias_escape_proof");
  }
  const std::vector<std::string> commands = {
      "capture_deferred_tensor_handles",
      "capture_value_lifetime_leases",
      "classify_boundary",
      "lower_region_at_flush_boundary",
  };

  std::ostringstream line;
  bool first = true;
  line << '{';
  append_json_field(line, "schema", "VulkanDeferredRegionPlanTrace.v0", first);
  append_json_field(line, "event", "deferred_region_plan_flush", first);
  append_json_field(line, "status", "planned_not_executed", first);
  append_json_uint(line, "region_id", state.next_deferred_region_plan_id++, first);
  append_json_uint(
      line,
      "op_node_count",
      static_cast<uint64_t>(state.deferred_op_nodes.size()),
      first);
  append_json_uint(
      line,
      "tensor_handle_count",
      static_cast<uint64_t>(state.deferred_tensor_handles.size()),
      first);
  append_json_uint(
      line, "value_lease_count", state.deferred_value_lease_count, first);
  append_json_uint(
      line,
      "missing_value_lease_count",
      state.deferred_missing_value_lease_count,
      first);
  append_json_uint(
      line,
      "alias_or_view_handle_count",
      state.deferred_alias_or_view_count,
      first);
  append_json_field(
      line,
      "boundary_kind",
      boundary_kind ? boundary_kind : "unknown_boundary",
      first);
  append_json_field(line, "reason", reason ? reason : "unknown", first);
  append_json_uint(
      line,
      "value_access_boundary_observed",
      state.pending_deferred_value_access_boundary ? 1u : 0u,
      first);
  if (state.pending_deferred_value_access_boundary) {
    append_json_field(
        line,
        "value_access_boundary_kind",
        state.deferred_value_access_boundary_kind,
        first);
    append_json_field(
        line,
        "value_access_reason",
        state.deferred_value_access_reason,
        first);
    append_json_field(
        line,
        "value_access_kind",
        state.deferred_value_access_kind,
        first);
    append_json_field(
        line,
        "value_access_source_state",
        state.deferred_value_access_source_state,
        first);
    append_json_field(
        line,
        "value_access_destination_state",
        state.deferred_value_access_destination_state,
        first);
    append_json_uint(
        line,
        "value_access_vulkan_source_count",
        state.deferred_value_access_vulkan_source_count,
        first);
    append_json_uint(
        line,
        "value_access_cpu_destination_count",
        state.deferred_value_access_cpu_destination_count,
        first);
  }
  append_json_field(
      line,
      "submit_phase",
      submit_phase_name(current_submit_phase()),
      first);
  append_json_field(
      line,
      "recent_op",
      recent_op_label().empty() ? "none" : recent_op_label(),
      first);
  append_json_string_array(line, "ops", op_names, first);
  append_json_string_array(line, "routes", route_names, first);
  append_json_string_array(line, "commands", commands, first);
  append_json_string_array(
      line, "missing_execution_prerequisites", missing_prerequisites, first);
  append_json_field(line, "top_blocker", top_blocker, first);
  append_json_uint(line, "requires_deferred_tensor_handles", 1u, first);
  append_json_uint(line, "requires_value_lifetime_leases", 1u, first);
  append_json_uint(line, "requires_output_ownership", 1u, first);
  append_json_uint(line, "execution_enabled", 0u, first);
  append_json_field(line, "behavior_change", "0", first);
  line << '}';
  append_deferred_region_plan_log_line(line.str());
  clear_deferred_region_plan_state(state);
}

void append_vulkan_failure_log(const std::string& message) {
  if (!vulkan_failure_logging_enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lock(failure_log_mutex());
  std::ofstream out(failure_log_path(), std::ios::app);
  out << message << '\n';
}

} // namespace

const char* vulkan_failure_class_name(
    const VulkanFailureClass failure_class) {
  switch (failure_class) {
    case VulkanFailureClass::TensorStateInvalid:
      return "TensorStateInvalid";
    case VulkanFailureClass::MetadataViewInvalid:
      return "MetadataViewInvalid";
    case VulkanFailureClass::RawCopyIllegal:
      return "RawCopyIllegal";
    case VulkanFailureClass::ReplayViewStale:
      return "ReplayViewStale";
    case VulkanFailureClass::RouteHardFail:
      return "RouteHardFail";
    case VulkanFailureClass::KernelIncorrect:
      return "KernelIncorrect";
    case VulkanFailureClass::DeviceLost:
      return "DeviceLost";
    case VulkanFailureClass::Unsupported:
      return "Unsupported";
    case VulkanFailureClass::ReplayHangRisk:
      return "ReplayHangRisk";
    case VulkanFailureClass::Unknown:
      return "Unknown";
  }
  return "Unknown";
}

bool vulkan_failure_logging_enabled() {
  return !failure_log_path().empty();
}

void mark_vulkan_post_failure_recovery_required() {
  post_failure_recovery_required().store(true, std::memory_order_release);
}

bool vulkan_post_failure_recovery_required() {
  return post_failure_recovery_required().load(std::memory_order_acquire);
}

void clear_vulkan_post_failure_recovery_required() {
  post_failure_recovery_required().store(false, std::memory_order_release);
}

bool vulkan_deferred_region_plan_logging_enabled() {
  return deferred_region_plan_logging_enabled();
}

void note_vulkan_deferred_region_tensor_write(
    const char* op_name,
    const char* route_name,
    const uint64_t output_storage_id,
    const uint64_t output_view_id,
    const uint64_t output_generation,
    const uint64_t output_logical_desc_hash,
    const int64_t output_storage_offset,
    const int64_t output_buffer_length,
    const bool output_is_view,
    const std::string& output_state,
    const std::vector<std::string>& input_states,
    const uint64_t vulkan_input_count,
    const uint64_t missing_input_lease_count) {
  if (
      !deferred_region_plan_logging_enabled() || op_name == nullptr ||
      op_name[0] == '\0' || is_lazy_chain_internal_bookkeeping(op_name)) {
    return;
  }

  LazyChainState& state = lazy_chain_state();
  const uint64_t output_handle_id = state.next_deferred_tensor_handle_id++;
  state.deferred_tensor_handles.push_back({
      output_handle_id,
      output_storage_id,
      output_view_id,
      output_generation,
      output_logical_desc_hash,
      output_storage_offset,
      output_buffer_length,
      output_is_view,
      output_state,
  });
  state.deferred_op_nodes.push_back({
      state.next_deferred_op_node_id++,
      output_handle_id,
      op_name,
      route_name && route_name[0] != '\0' ? route_name : "<unspecified>",
      static_cast<uint64_t>(input_states.size()),
      vulkan_input_count,
      missing_input_lease_count,
  });
  state.deferred_value_lease_count += vulkan_input_count;
  state.deferred_missing_value_lease_count += missing_input_lease_count;
  if (output_is_view) {
    ++state.deferred_alias_or_view_count;
  }
}

void note_vulkan_deferred_region_value_access_boundary(
    const char* boundary_kind,
    const char* reason,
    const char* access_kind,
    const std::string& source_state,
    const std::string& destination_state,
    const uint64_t vulkan_source_count,
    const uint64_t cpu_destination_count) {
  if (!deferred_region_plan_logging_enabled()) {
    return;
  }

  LazyChainState& state = lazy_chain_state();
  state.pending_deferred_value_access_boundary = true;
  state.deferred_value_access_boundary_kind =
      boundary_kind && boundary_kind[0] != '\0' ? boundary_kind
                                                : "unknown_boundary";
  state.deferred_value_access_reason =
      reason && reason[0] != '\0' ? reason : "unknown";
  state.deferred_value_access_kind =
      access_kind && access_kind[0] != '\0' ? access_kind : "unknown_access";
  state.deferred_value_access_source_state = source_state;
  state.deferred_value_access_destination_state = destination_state;
  state.deferred_value_access_vulkan_source_count = vulkan_source_count;
  state.deferred_value_access_cpu_destination_count = cpu_destination_count;
}

void note_vulkan_lazy_chain_op(const char* op_name) {
  if (
      (!lazy_chain_logging_enabled() &&
       !deferred_execution_logging_enabled() &&
       !runtime_shader_compile_logging_enabled() &&
       !runtime_command_list_logging_enabled()) ||
      op_name == nullptr || op_name[0] == '\0') {
    return;
  }
  maybe_log_deferred_bridge_event(op_name);
  if (is_lazy_chain_internal_bookkeeping(op_name)) {
    return;
  }
  LazyChainState& state = lazy_chain_state();
  state.raw_ops.emplace_back(op_name);
  if (!lazy_chain_logging_enabled()) {
    return;
  }
  std::ostringstream entry;
  entry << "{\"op_id\":" << state.next_op_id++ << ",\"op\":\""
        << json_escape(op_name) << "\"";
  if (!current_allocation_label().empty()) {
    entry << ",\"caller\":\"" << json_escape(current_allocation_label()) << "\"";
  }
  if (!current_runtime_label().empty()) {
    entry << ",\"runtime\":\"" << json_escape(current_runtime_label()) << "\"";
  }
  entry << '}';
  state.ops.emplace_back(entry.str());
}

void flush_vulkan_lazy_chain_boundary(
    const char* boundary_kind,
    const char* reason) {
  if (
      !lazy_chain_logging_enabled() &&
      !deferred_execution_logging_enabled() &&
      !deferred_region_plan_logging_enabled() &&
      !runtime_shader_compile_logging_enabled() &&
      !runtime_command_list_logging_enabled()) {
    return;
  }
  LazyChainState& state = lazy_chain_state();
  maybe_log_runtime_command_list_plan(state, boundary_kind, reason);
  maybe_compile_runtime_shader_group(state, boundary_kind, reason);
  flush_deferred_region_plan(state, boundary_kind, reason);
  flush_deferred_execution_region(state, boundary_kind, reason);
  if (!lazy_chain_logging_enabled()) {
    state.raw_ops.clear();
    return;
  }
  if (state.ops.empty() && (boundary_kind == nullptr || boundary_kind[0] == '\0')) {
    state.raw_ops.clear();
    return;
  }

  std::ostringstream line;
  bool first = true;
  line << '{';
  append_json_field(line, "schema", "VulkanLazyEagerChain.v0", first);
  append_json_field(line, "event", "mandatory_access_boundary", first);
  append_json_uint(line, "chain_id", state.next_chain_id++, first);
  append_json_uint(
      line, "op_count", static_cast<uint64_t>(state.ops.size()), first);
  append_json_field(
      line,
      "boundary_kind",
      boundary_kind ? boundary_kind : "unknown_boundary",
      first);
  append_json_field(line, "reason", reason ? reason : "unknown", first);
  append_json_field(
      line,
      "submit_phase",
      submit_phase_name(current_submit_phase()),
      first);
  append_json_field(
      line,
      "recent_op",
      recent_op_label().empty() ? "none" : recent_op_label(),
      first);
  if (!current_allocation_label().empty()) {
    append_json_field(line, "caller", current_allocation_label(), first);
  }
  if (!current_runtime_label().empty()) {
    append_json_field(line, "runtime", current_runtime_label(), first);
  }
  if (!first) {
    line << ',';
  }
  line << "\"ops\":[";
  for (size_t i = 0; i < state.ops.size(); ++i) {
    if (i > 0) {
      line << ',';
    }
    line << state.ops[i];
  }
  line << "]}";

  {
    std::lock_guard<std::mutex> lock(lazy_chain_log_mutex());
    std::ofstream out(lazy_chain_log_path(), std::ios::app);
    out << line.str() << '\n';
  }
  state.ops.clear();
  state.raw_ops.clear();
}

std::string format_vulkan_failure(
    const VulkanFailureClass failure_class,
    const char* op_name,
    const char* reason,
    const std::string& detail) {
  std::ostringstream out;
  out << "Vulkan failure"
      << " failure_class=" << vulkan_failure_class_name(failure_class);
  if (op_name && op_name[0] != '\0') {
    out << " op=" << op_name;
  }
  if (reason && reason[0] != '\0') {
    out << " reason=" << reason;
  }
  if (!current_allocation_label().empty()) {
    out << " caller=" << current_allocation_label();
  }
  if (!current_runtime_label().empty()) {
    out << " runtime=" << current_runtime_label();
  }
  if (!detail.empty()) {
    out << " detail={" << detail << '}';
  }
  return out.str();
}

void log_vulkan_failure(
    const VulkanFailureClass failure_class,
    const char* op_name,
    const char* reason,
    const std::string& detail) {
  append_vulkan_failure_log(
      format_vulkan_failure(failure_class, op_name, reason, detail));
}

std::string report_vulkan_failure(
    const VulkanFailureClass failure_class,
    const char* op_name,
    const char* reason,
    const std::string& detail) {
  const std::string message =
      format_vulkan_failure(failure_class, op_name, reason, detail);
  append_vulkan_failure_log(message);
  return message;
}

[[noreturn]] void fail_vulkan(
    const VulkanFailureClass failure_class,
    const char* op_name,
    const char* reason,
    const std::string& detail) {
  mark_vulkan_post_failure_recovery_required();
  TORCH_CHECK(false, report_vulkan_failure(failure_class, op_name, reason, detail));
  std::abort();
}

void check_vulkan(
    const bool condition,
    const VulkanFailureClass failure_class,
    const char* op_name,
    const char* reason,
    const std::string& detail) {
  if (!condition) {
    fail_vulkan(failure_class, op_name, reason, detail);
  }
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
