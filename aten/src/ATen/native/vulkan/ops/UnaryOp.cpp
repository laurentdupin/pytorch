#include <ATen/ArrayRef.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/atan.h>
#include <ATen/ops/cos.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/log.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/reciprocal.h>
#include <ATen/ops/rsqrt.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/sin.h>
#include <ATen/ops/sqrt.h>
#include <ATen/ops/tan.h>
#endif
#include <ATen/native/vulkan/ops/BinaryOp.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <c10/util/irange.h>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <torch/library.h>
#include <unordered_map>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {
using namespace api::utils;

Device vulkan_output_device(const Tensor& tensor) {
  return tensor.is_vulkan() ? tensor.device()
                            : Device(at::kVulkan, api::current_device());
}

enum class UnaryOpKind : uint8_t {
  Exp,
  Sqrt,
  Log,
  Sin,
  Cos,
  Tan,
  Atan,
  Neg,
  Reciprocal,
  Rsqrt,
  Silu,
};

const char* unary_op_kind_name(const UnaryOpKind op_kind) {
  switch (op_kind) {
    case UnaryOpKind::Exp:
      return "exp";
    case UnaryOpKind::Sqrt:
      return "sqrt";
    case UnaryOpKind::Log:
      return "log";
    case UnaryOpKind::Sin:
      return "sin";
    case UnaryOpKind::Cos:
      return "cos";
    case UnaryOpKind::Tan:
      return "tan";
    case UnaryOpKind::Atan:
      return "atan";
    case UnaryOpKind::Neg:
      return "neg";
    case UnaryOpKind::Reciprocal:
      return "reciprocal";
    case UnaryOpKind::Rsqrt:
      return "rsqrt";
    case UnaryOpKind::Silu:
      return "silu";
  }
  return "unknown";
}

std::string format_unary_sizes(IntArrayRef sizes) {
  std::ostringstream stream;
  stream << '[';
  for (const auto idx : c10::irange(sizes.size())) {
    if (idx > 0) {
      stream << 'x';
    }
    stream << sizes[idx];
  }
  stream << ']';
  return stream.str();
}

void log_unary_submit(
    const UnaryOpKind op_kind,
    const char* route,
    const vTensor& v_input,
    const vTensor& v_output,
    const api::utils::uvec3& global_size,
    const api::utils::uvec3& local_size) {
  std::ostringstream stream;
  stream << "aten::unary.submit"
         << " op=" << unary_op_kind_name(op_kind)
         << " route=" << route
         << " input=" << format_unary_sizes(v_input.sizes())
         << " output=" << format_unary_sizes(v_output.sizes())
         << " input_direct=" << (v_input.has_direct_buffer_layout() ? 1 : 0)
         << " output_direct=" << (v_output.has_direct_buffer_layout() ? 1 : 0)
         << " input_offset=" << v_input.storage_offset()
         << " output_offset=" << v_output.storage_offset()
         << " input_buffer_len="
         << (v_input.storage_type() == api::StorageType::BUFFER
                 ? v_input.buffer_length()
                 : -1)
         << " output_buffer_len="
         << (v_output.storage_type() == api::StorageType::BUFFER
                 ? v_output.buffer_length()
                 : -1)
         << " global=[" << global_size.data[0] << 'x' << global_size.data[1]
         << 'x' << global_size.data[2] << ']'
         << " local=[" << local_size.data[0] << 'x' << local_size.data[1]
         << 'x' << local_size.data[2] << ']';
  utils::log_vulkan_op_hit(stream.str());
}

std::string runtime_unary_json_quote(const std::string& value) {
  std::string quoted = "\"";
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        quoted += "\\\\";
        break;
      case '"':
        quoted += "\\\"";
        break;
      case '\n':
        quoted += "\\n";
        break;
      case '\r':
        quoted += "\\r";
        break;
      case '\t':
        quoted += "\\t";
        break;
      default:
        quoted += ch;
        break;
    }
  }
  quoted += "\"";
  return quoted;
}

std::string runtime_unary_env(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string{} : std::string{value};
}

std::string quote_runtime_unary_command_arg(const std::string& value) {
  std::string quoted = "\"";
  for (const char ch : value) {
    if (ch == '"') {
      quoted += "\\\"";
    } else {
      quoted += ch;
    }
  }
  quoted += "\"";
  return quoted;
}

bool runtime_unary_live_chain_execute_enabled() {
  const std::string value =
      runtime_unary_env("PYTORCH_VULKAN_RUNTIME_ELEMENTWISE_CHAIN_EXECUTE");
  return value == "1" || value == "true" || value == "TRUE";
}

std::string runtime_unary_live_chain_log_path() {
  return runtime_unary_env("PYTORCH_VULKAN_RUNTIME_ELEMENTWISE_CHAIN_LIVE_LOG");
}

const char* runtime_unary_expression(
    const UnaryOpKind op_kind,
    const char* value) {
  switch (op_kind) {
    case UnaryOpKind::Exp:
      return "exp(value)";
    case UnaryOpKind::Sqrt:
      return "sqrt(value)";
    case UnaryOpKind::Log:
      return "log(value)";
    case UnaryOpKind::Sin:
      return "sin(value)";
    case UnaryOpKind::Cos:
      return "cos(value)";
    case UnaryOpKind::Neg:
      return "-value";
    case UnaryOpKind::Reciprocal:
      return "1.0 / value";
    case UnaryOpKind::Rsqrt:
      return "inversesqrt(value)";
    case UnaryOpKind::Silu:
      return "value / (1.0 + exp(-value))";
    case UnaryOpKind::Tan:
    case UnaryOpKind::Atan:
      break;
  }
  TORCH_CHECK(false, "Unsupported runtime unary elementwise chain op: ", value);
  return "";
}

bool runtime_unary_chain_op_supported(const UnaryOpKind op_kind) {
  switch (op_kind) {
    case UnaryOpKind::Exp:
    case UnaryOpKind::Sqrt:
    case UnaryOpKind::Log:
    case UnaryOpKind::Sin:
    case UnaryOpKind::Cos:
    case UnaryOpKind::Neg:
    case UnaryOpKind::Reciprocal:
    case UnaryOpKind::Rsqrt:
    case UnaryOpKind::Silu:
      return true;
    case UnaryOpKind::Tan:
    case UnaryOpKind::Atan:
      return false;
  }
  return false;
}

std::string runtime_unary_chain_key(const std::vector<std::string>& ops) {
  std::ostringstream key;
  key << "unary";
  for (const std::string& op : ops) {
    key << '_' << op;
  }
  return key.str();
}

std::string runtime_unary_chain_glsl(const std::vector<UnaryOpKind>& ops) {
  TORCH_CHECK(!ops.empty(), "Runtime unary chain expects at least one op");
  TORCH_CHECK(
      ops.size() <= 8,
      "Runtime unary chain currently supports at most 8 ops");

  std::ostringstream glsl;
  glsl << R"(#version 450 core
layout(std430) buffer;

uint coord_to_idx(const uvec4 coord, const uvec4 strides) {
  const uvec4 linear_terms = coord * strides;
  return linear_terms.x + linear_terms.y + linear_terms.z + linear_terms.w;
}

uvec4 idx_to_coord(const uint idx, const uvec4 strides, const uvec4 sizes) {
  return uvec4(
      (idx / strides.x) % sizes.x,
      (idx / strides.y) % sizes.y,
      (idx / strides.z) % sizes.z,
      (idx / strides.w) % sizes.w);
}

layout(set = 0, binding = 0) buffer restrict writeonly OutBuffer {
  float data[];
} uOutput;
layout(set = 0, binding = 1) uniform restrict OutMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
} uOutMeta;
layout(set = 0, binding = 2) buffer restrict readonly InBuffer {
  float data[];
} uInput;
layout(set = 0, binding = 3) uniform restrict InMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
} uInMeta;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float read_input(const uvec4 coord) {
  const uint read_idx =
      coord_to_idx(coord, uInMeta.physical_strides) + uInMeta.info.w;
  if (read_idx >= uInMeta.info.z) {
    return 0.0;
  }
  return uInput.data[read_idx];
}

void zero_width_pack_padding(
    const uvec4 write_coord,
    const uint out_buf_length,
    const uint out_storage_offset) {
  const uint logical_channels = uOutMeta.logical_sizes.x;
  const uint physical_channels = uOutMeta.physical_strides.y;
  if (write_coord.x != 0u || logical_channels >= physical_channels) {
    return;
  }

  uvec4 pad_coord = write_coord;
  for (uint c = logical_channels; c < physical_channels; ++c) {
    pad_coord.x = c;
    const uint pad_idx =
        coord_to_idx(pad_coord, uOutMeta.physical_strides) + out_storage_offset;
    if (pad_idx < out_buf_length) {
      uOutput.data[pad_idx] = 0.0;
    }
  }
}

void main() {
  const uint write_idx = ivec3(gl_GlobalInvocationID).x;
  const uint out_numel = uOutMeta.info.y;
  const uint out_buf_length = uOutMeta.info.z;
  const uint out_storage_offset = uOutMeta.info.w;

  if (write_idx >= out_numel) {
    return;
  }

  const uvec4 coord =
      idx_to_coord(write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);
  float value = read_input(coord);
)";
  for (const UnaryOpKind op_kind : ops) {
    glsl << "  value = " << runtime_unary_expression(op_kind, "value")
         << ";\n";
  }
  glsl << R"(
  const uint actual_write_idx =
      coord_to_idx(coord, uOutMeta.physical_strides) + out_storage_offset;
  if (actual_write_idx < out_buf_length) {
    uOutput.data[actual_write_idx] = value;
  }

  zero_width_pack_padding(coord, out_buf_length, out_storage_offset);
}
)";
  return glsl.str();
}

std::vector<uint32_t> read_runtime_unary_spirv_file(
    const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  TORCH_CHECK(in, "Could not read runtime Vulkan SPIR-V file: ", path.string());
  in.seekg(0, std::ios::end);
  const std::streamoff byte_size = in.tellg();
  TORCH_CHECK(
      byte_size > 0 &&
          byte_size % static_cast<std::streamoff>(sizeof(uint32_t)) == 0,
      "Runtime Vulkan SPIR-V file has invalid byte size: ",
      path.string());
  in.seekg(0, std::ios::beg);
  std::vector<uint32_t> words(
      static_cast<size_t>(byte_size) / sizeof(uint32_t));
  in.read(
      reinterpret_cast<char*>(words.data()),
      static_cast<std::streamsize>(byte_size));
  TORCH_CHECK(
      !words.empty() && words[0] == 0x07230203u,
      "Runtime Vulkan shader compiler produced invalid SPIR-V: ",
      path.string());
  return words;
}

const std::vector<uint32_t>& runtime_unary_chain_spirv(
    const std::vector<UnaryOpKind>& kinds,
    const std::vector<std::string>& ops) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::vector<uint32_t>> cached_spirv;
  const std::string program_key = runtime_unary_chain_key(ops);
  std::lock_guard<std::mutex> lock(mutex);
  const auto cache_it = cached_spirv.find(program_key);
  if (cache_it != cached_spirv.end()) {
    return cache_it->second;
  }

  const std::string glslc_path =
      runtime_unary_env("PYTORCH_VULKAN_RUNTIME_SHADER_GLSLC");
  TORCH_CHECK(
      !glslc_path.empty(),
      "runtime unary elementwise chain requires ",
      "PYTORCH_VULKAN_RUNTIME_SHADER_GLSLC to point at glslc");

  std::filesystem::path cache_dir =
      runtime_unary_env("PYTORCH_VULKAN_RUNTIME_SHADER_CACHE_DIR");
  if (cache_dir.empty()) {
    cache_dir =
        std::filesystem::temp_directory_path() / "pytorch_vulkan_runtime_shaders";
  }

  std::error_code ec;
  std::filesystem::create_directories(cache_dir, ec);
  TORCH_CHECK(
      !ec,
      "Could not create Vulkan runtime shader cache directory ",
      cache_dir.string(),
      ": ",
      ec.message());

  const std::filesystem::path glsl_path =
      cache_dir / ("runtime_elementwise_chain_" + program_key + ".glsl");
  const std::filesystem::path spv_path =
      cache_dir / ("runtime_elementwise_chain_" + program_key + ".spv");
  {
    std::ofstream glsl(glsl_path);
    TORCH_CHECK(
        glsl,
        "Could not write runtime Vulkan GLSL file: ",
        glsl_path.string());
    glsl << runtime_unary_chain_glsl(kinds);
  }

  std::ostringstream command;
  command << quote_runtime_unary_command_arg(glslc_path)
          << " -fshader-stage=compute "
          << quote_runtime_unary_command_arg(glsl_path.string()) << " -o "
          << quote_runtime_unary_command_arg(spv_path.string())
          << " --target-env=vulkan1.3 --target-spv=spv1.6 -Werror";
#ifdef _WIN32
  std::string shell_command = "cmd.exe /S /C \"";
  shell_command += command.str();
  shell_command += "\"";
  const int compile_exit_code = std::system(shell_command.c_str());
#else
  const int compile_exit_code = std::system(command.str().c_str());
#endif
  TORCH_CHECK(
      compile_exit_code == 0,
      "Runtime Vulkan unary shader compilation failed for ",
      glsl_path.string(),
      " with exit code ",
      compile_exit_code);

  const auto insert_result = cached_spirv.emplace(
      program_key,
      read_runtime_unary_spirv_file(spv_path));
  return insert_result.first->second;
}

Tensor run_runtime_unary_chain(
    const Tensor& input,
    const std::vector<UnaryOpKind>& kinds,
    const std::vector<std::string>& ops) {
  TORCH_CHECK(
      !ops.empty() && ops.size() == kinds.size(),
      "runtime unary elementwise chain expects op names and kinds");
  TORCH_CHECK(
      ops.size() <= 8,
      "runtime unary elementwise chain currently supports 1 to 8 ops");
  for (const UnaryOpKind kind : kinds) {
    TORCH_CHECK(
        runtime_unary_chain_op_supported(kind),
        "runtime unary elementwise chain received unsupported op");
  }
  TORCH_CHECK(
      input.is_vulkan(),
      "runtime unary elementwise chain expects a Vulkan tensor");
  TORCH_CHECK(
      input.scalar_type() == at::kFloat,
      "runtime unary elementwise chain expects fp32 tensor");
  const vTensor& v_input_const = convert(input);
  TORCH_CHECK(
      v_input_const.storage_type() == api::StorageType::BUFFER &&
          utils::supports_buffer_elementwise_compute(v_input_const) &&
          !v_input_const.is_quantized(),
      "runtime unary elementwise chain expects supported Vulkan buffer tensor");

  api::AllocationScope allocation_scope(
      "runtime_elementwise_chain.unary");
  api::Context* const context = api::context();
  vTensor& v_input = convert(input);
  TORCH_CHECK(
      v_input.numel() > 0,
      "runtime unary elementwise chain expects a non-empty tensor");

  vTensor v_output{
      context,
      input.sizes().vec(),
      v_input.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const std::vector<uint32_t>& spirv = runtime_unary_chain_spirv(kinds, ops);
  api::ShaderInfo shader_descriptor{
      "runtime_elementwise_chain." + runtime_unary_chain_key(ops),
      std::vector<uint32_t>(spirv.begin(), spirv.end()),
      {
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      },
  };

  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(v_output.numel()),
      1u,
      1u,
  };
  const uvec3 local_size = adaptive_work_group_size(global_size);
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);

  utils::log_vulkan_op_hit("vulkan_prepack::runtime_unary_elementwise_chain");
  context->submit_compute_job(
      shader_descriptor,
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer());

  return record_tensor_write_and_return(
      convert(v_output),
      "vulkan_prepack::runtime_unary_elementwise_chain",
      "runtime_generated_unary_elementwise_chain",
      {input});
}

struct RuntimeUnaryLiveChain final {
  Tensor input;
  std::vector<UnaryOpKind> kinds;
  std::vector<std::string> ops;
};

struct RuntimeUnaryLiveChainState final {
  std::mutex mutex;
  std::unordered_map<const void*, RuntimeUnaryLiveChain> chains;
  size_t sequence{0};
};

RuntimeUnaryLiveChainState& runtime_unary_live_chain_state() {
  static RuntimeUnaryLiveChainState state;
  return state;
}

thread_local bool runtime_unary_live_chain_probe_active = false;

void append_runtime_unary_string_array(
    std::ostringstream& out,
    const std::vector<std::string>& values) {
  out << '[';
  for (const auto idx : c10::irange(values.size())) {
    if (idx > 0) {
      out << ',';
    }
    out << runtime_unary_json_quote(values[idx]);
  }
  out << ']';
}

void append_runtime_unary_operand_kind_array(
    std::ostringstream& out,
    const size_t count) {
  out << '[';
  for (const auto idx : c10::irange(count)) {
    if (idx > 0) {
      out << ',';
    }
    out << "\"unary\"";
  }
  out << ']';
}

void append_runtime_unary_shape_array(std::ostringstream& out, IntArrayRef sizes) {
  out << '[';
  for (const auto idx : c10::irange(sizes.size())) {
    if (idx > 0) {
      out << ',';
    }
    out << sizes[idx];
  }
  out << ']';
}

void log_runtime_unary_live_chain(
    const RuntimeUnaryLiveChain& chain,
    const Tensor& output,
    const size_t sequence,
    const char* status,
    const bool executed,
    const std::string& detail) {
  const std::string log_path = runtime_unary_live_chain_log_path();
  if (log_path.empty()) {
    return;
  }
  std::ofstream log(log_path, std::ios::app);
  if (!log) {
    return;
  }
  std::ostringstream row;
  row << "{\"schema\":\"VulkanRuntimeElementwiseLiveChainTrace.v0\"";
  row << ",\"sequence\":" << sequence;
  row << ",\"family\":\"ElementwiseChain\"";
  row << ",\"source\":\"unary_op_buffer\"";
  row << ",\"behavior_change\":0";
  row << ",\"normal_eager_output_preserved\":1";
  row << ",\"status\":" << runtime_unary_json_quote(status);
  row << ",\"executed\":" << (executed ? 1 : 0);
  row << ",\"chain_length\":" << chain.ops.size();
  row << ",\"ops\":";
  append_runtime_unary_string_array(row, chain.ops);
  row << ",\"operand_kinds\":";
  append_runtime_unary_operand_kind_array(row, chain.ops.size());
  row << ",\"tensor_rhs_count\":0";
  row << ",\"scalar_rhs_count\":0";
  row << ",\"input_shape\":";
  append_runtime_unary_shape_array(row, chain.input.sizes());
  row << ",\"output_shape\":";
  append_runtime_unary_shape_array(row, output.sizes());
  if (!detail.empty()) {
    row << ",\"detail\":" << runtime_unary_json_quote(detail);
  }
  row << "}\n";
  log << row.str();
}

void maybe_probe_runtime_unary_live_chain(
    const Tensor& self,
    const Tensor& output,
    const UnaryOpKind op_kind) {
  if (runtime_unary_live_chain_probe_active ||
      !runtime_unary_chain_op_supported(op_kind)) {
    return;
  }
  const bool execute = runtime_unary_live_chain_execute_enabled();
  if (!execute && runtime_unary_live_chain_log_path().empty()) {
    return;
  }
  if (!self.is_vulkan() || !output.is_vulkan() ||
      self.scalar_type() != kFloat || output.scalar_type() != kFloat ||
      self.sizes().vec() != output.sizes().vec()) {
    return;
  }

  RuntimeUnaryLiveChain chain;
  size_t sequence = 0;
  {
    RuntimeUnaryLiveChainState& state = runtime_unary_live_chain_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const void* self_key = convert(self).storage_identity();
    auto previous = state.chains.find(self_key);
    if (previous != state.chains.end()) {
      chain = previous->second;
    } else {
      chain.input = self;
    }
    chain.kinds.push_back(op_kind);
    chain.ops.emplace_back(unary_op_kind_name(op_kind));
    if (chain.ops.size() > 8u) {
      state.chains.erase(self_key);
      return;
    }
    const void* output_key = convert(output).storage_identity();
    state.chains[output_key] = chain;
    if (state.chains.size() > 256u) {
      state.chains.erase(state.chains.begin());
    }
    sequence = ++state.sequence;
  }

  bool executed = false;
  std::string status = "captured";
  std::string detail;
  if (execute && chain.ops.size() >= 2u) {
    runtime_unary_live_chain_probe_active = true;
    try {
      Tensor generated = run_runtime_unary_chain(
          chain.input, chain.kinds, chain.ops);
      (void)generated;
      executed = true;
      status = "executed";
    } catch (const c10::Error& error) {
      status = "execute_failed";
      detail = error.what_without_backtrace();
    } catch (const std::exception& error) {
      status = "execute_failed";
      detail = error.what();
    } catch (...) {
      status = "execute_failed";
      detail = "unknown runtime unary live-chain execution error";
    }
    runtime_unary_live_chain_probe_active = false;
  }
  log_runtime_unary_live_chain(
      chain, output, sequence, status.c_str(), executed, detail);
}

bool needs_unary_cpu_fallback(const Tensor& tensor) {
  if (!tensor.is_vulkan()) {
    return false;
  }

  const vTensor& v_tensor = convert(tensor);
  return v_tensor.storage_type() == api::StorageType::BUFFER &&
      !utils::supports_buffer_elementwise_compute(v_tensor);
}

Tensor unary_op_cpu_fallback(const Tensor& self_arg, const UnaryOpKind op_kind) {
  report_vulkan_cpu_fallback(
      "aten::unary_op", "cpu_fallback", {self_arg});
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
    switch (op_kind) {
      case UnaryOpKind::Exp:
        cpu_result = at::exp(self_cpu);
        break;
      case UnaryOpKind::Sqrt:
        cpu_result = at::sqrt(self_cpu);
        break;
      case UnaryOpKind::Log:
        cpu_result = at::log(self_cpu);
        break;
      case UnaryOpKind::Sin:
        cpu_result = at::sin(self_cpu);
        break;
      case UnaryOpKind::Cos:
        cpu_result = at::cos(self_cpu);
        break;
      case UnaryOpKind::Tan:
        cpu_result = at::tan(self_cpu);
        break;
      case UnaryOpKind::Atan:
        cpu_result = at::atan(self_cpu);
        break;
      case UnaryOpKind::Neg:
        cpu_result = at::neg(self_cpu);
        break;
      case UnaryOpKind::Reciprocal:
        cpu_result = at::reciprocal(self_cpu);
        break;
      case UnaryOpKind::Rsqrt:
        cpu_result = at::rsqrt(self_cpu);
        break;
      case UnaryOpKind::Silu:
        cpu_result = at::silu(self_cpu);
        break;
    }
  }
  return record_tensor_write_and_return(
      cpu_result.to(vulkan_output_device(self_arg)),
      "aten::unary",
      unary_op_kind_name(op_kind),
      {self_arg});
}

Tensor unary_op_buffer(
    const Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor,
    const UnaryOpKind op_kind) {
  api::AllocationScope allocation_scope("unary_op.buffer");
  api::Context* const context = api::context();

  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(v_output.numel()),
      1u,
      1u,
  };
  const uvec3 local_size = adaptive_work_group_size(global_size);
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_self);
  log_unary_submit(
      op_kind, "buffer", v_self, v_output, global_size, local_size);

  context->submit_compute_job(
      shader_descriptor,
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer());

  return record_tensor_write_and_return(
      convert(v_output),
      "aten::unary",
      unary_op_kind_name(op_kind),
      {self});
}

Tensor unary_op(
    const Tensor& self_arg,
    const api::ShaderInfo& shader_descriptor,
    const api::ShaderInfo& buffer_shader_descriptor,
    const UnaryOpKind op_kind) {
  api::Context* const context = api::context();

  if (needs_unary_cpu_fallback(self_arg)) {
    return unary_op_cpu_fallback(self_arg, op_kind);
  }

  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  Tensor logical_self = self;
  const auto plan = utils::build_vulkan_execution_plan(
      self, utils::VulkanExecutionPlanKind::ElementwiseInput);
  if (api::uses_buffer_execution(plan.execution_layout)) {
    self = utils::prepare_vulkan_direct_buffer_execution_tensor(self, plan);
    Tensor returned = unary_op_buffer(self, buffer_shader_descriptor, op_kind);
    note_runtime_elementwise_unary_live_chain(
        logical_self, returned, unary_op_kind_name(op_kind));
    return returned;
  }

  self = utils::prepare_vulkan_execution_tensor(
      self, utils::VulkanExecutionPlanKind::TextureComputeInput);

  const vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  const struct Block final {
    uvec3 extents;
    uint32_t fill0;
  } block{
      v_self.extents(),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = v_output.extents();
  const uvec3 local_size = adaptive_work_group_size(global_size);
  log_unary_submit(
      op_kind, "texture", v_self, v_output, global_size, local_size);

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      global_size,
      // local work group size
      local_size,
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return record_tensor_write_and_return(
      convert(v_output),
      "aten::unary",
      unary_op_kind_name(op_kind),
      {self});
}

Tensor& unary_op_(Tensor& self_arg, const api::ShaderInfo& shader_descriptor) {
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);

  const struct Block final {
    uvec3 extents;
    uint32_t fill0;
  } block{
      v_self.extents(),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_self.extents(),
      // local work group size
      adaptive_work_group_size(v_self.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_self.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ | api::MemoryAccessType::WRITE),
      // params buffer
      params.buffer());

  return self_arg;
}

Tensor exp(const Tensor& self_arg) {
  return unary_op(
      self_arg, VK_KERNEL(exp), VK_KERNEL(buffer_exp), UnaryOpKind::Exp);
}

Tensor& exp_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(exp_inplace));
}

Tensor sqrt(const Tensor& self_arg) {
  return unary_op(
      self_arg, VK_KERNEL(sqrt), VK_KERNEL(buffer_sqrt), UnaryOpKind::Sqrt);
}

Tensor& sqrt_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(sqrt_inplace));
}

Tensor log(const Tensor& self_arg) {
  return unary_op(
      self_arg, VK_KERNEL(log), VK_KERNEL(buffer_log), UnaryOpKind::Log);
}

Tensor& log_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(log_inplace));
}

Tensor sin(const Tensor& self_arg) {
  return unary_op(
      self_arg, VK_KERNEL(sin), VK_KERNEL(buffer_sin), UnaryOpKind::Sin);
}

Tensor& sin_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(sin_inplace));
}

Tensor cos(const Tensor& self_arg) {
  return unary_op(
      self_arg, VK_KERNEL(cos), VK_KERNEL(buffer_cos), UnaryOpKind::Cos);
}

Tensor& cos_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(cos_inplace));
}

Tensor tan(const Tensor& self_arg) {
  return unary_op_cpu_fallback(self_arg, UnaryOpKind::Tan);
}

Tensor atan(const Tensor& self_arg) {
  return unary_op_cpu_fallback(self_arg, UnaryOpKind::Atan);
}

Tensor neg(const Tensor& self_arg) {
  return unary_op(
      self_arg, VK_KERNEL(neg), VK_KERNEL(buffer_neg), UnaryOpKind::Neg);
}

Tensor& neg_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(neg_inplace));
}

Tensor reciprocal(const Tensor& self_arg) {
  return unary_op_cpu_fallback(self_arg, UnaryOpKind::Reciprocal);
}

Tensor rsqrt(const Tensor& self_arg) {
  return unary_op(
      self_arg, VK_KERNEL(rsqrt), VK_KERNEL(buffer_rsqrt), UnaryOpKind::Rsqrt);
}

Tensor& rsqrt_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(rsqrt_inplace));
}

Tensor silu(const Tensor& self_arg) {
  return unary_op(
      self_arg, VK_KERNEL(silu), VK_KERNEL(buffer_silu), UnaryOpKind::Silu);
}

Tensor& silu_(Tensor& self_arg) {
  return unary_op_(self_arg, VK_KERNEL(silu_inplace));
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::exp"), TORCH_FN(exp));
  m.impl(TORCH_SELECTIVE_NAME("aten::exp_"), TORCH_FN(exp_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sqrt"), TORCH_FN(sqrt));
  m.impl(TORCH_SELECTIVE_NAME("aten::sqrt_"), TORCH_FN(sqrt_));
  m.impl(TORCH_SELECTIVE_NAME("aten::log"), TORCH_FN(log));
  m.impl(TORCH_SELECTIVE_NAME("aten::log_"), TORCH_FN(log_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sin"), TORCH_FN(sin));
  m.impl(TORCH_SELECTIVE_NAME("aten::sin_"), TORCH_FN(sin_));
  m.impl(TORCH_SELECTIVE_NAME("aten::cos"), TORCH_FN(cos));
  m.impl(TORCH_SELECTIVE_NAME("aten::cos_"), TORCH_FN(cos_));
  m.impl("tan", TORCH_FN(tan));
  m.impl("atan", TORCH_FN(atan));
  m.impl(TORCH_SELECTIVE_NAME("aten::neg"), TORCH_FN(neg));
  m.impl(TORCH_SELECTIVE_NAME("aten::neg_"), TORCH_FN(neg_));
  m.impl(TORCH_SELECTIVE_NAME("aten::reciprocal"), TORCH_FN(reciprocal));
  m.impl(TORCH_SELECTIVE_NAME("aten::rsqrt"), TORCH_FN(rsqrt));
  m.impl(TORCH_SELECTIVE_NAME("aten::rsqrt_"), TORCH_FN(rsqrt_));
  m.impl(TORCH_SELECTIVE_NAME("aten::silu"), TORCH_FN(silu));
  m.impl(TORCH_SELECTIVE_NAME("aten::silu_"), TORCH_FN(silu_));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
