#ifdef USE_VULKAN_API
#include <ATen/ArrayRef.h>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/abs.h>
#include <ATen/ops/add.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/bitwise_and.h>
#include <ATen/ops/bitwise_not.h>
#include <ATen/ops/bitwise_or.h>
#include <ATen/ops/div.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/logical_and.h>
#include <ATen/ops/logical_not.h>
#include <ATen/ops/logical_or.h>
#include <ATen/ops/max.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/sub.h>
#endif
#include <ATen/native/vulkan/ops/BinaryOp.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convert.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/ops/LayoutTransitions.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <ATen/native/vulkan/ops/Softmax.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/TensorState.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/planning/ExecutionContracts.h>
#include <ATen/native/vulkan/planning/ReplayTensorState.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <c10/util/irange.h>
#include <exception>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <torch/library.h>
#include <unordered_map>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

using namespace api::utils;

namespace {

enum class BinaryOpKind : uint8_t {
  Add,
  Sub,
  Mul,
  Div,
  FloorDivide,
  Pow,
};

const char* binary_op_kind_name(const BinaryOpKind op_kind) {
  switch (op_kind) {
    case BinaryOpKind::Add:
      return "add";
    case BinaryOpKind::Sub:
      return "sub";
    case BinaryOpKind::Mul:
      return "mul";
    case BinaryOpKind::Div:
      return "div";
    case BinaryOpKind::FloorDivide:
      return "floor_divide";
    case BinaryOpKind::Pow:
      return "pow";
  }
  return "unknown";
}

std::string quote_runtime_shader_command_arg(const std::string& value) {
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

std::string runtime_shader_env(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string{} : std::string{value};
}

const char* runtime_elementwise_chain_op_symbol(const std::string& op) {
  if (op == "add") {
    return "+";
  }
  if (op == "mul") {
    return "*";
  }
  if (op == "sub") {
    return "-";
  }
  if (op == "div") {
    return "/";
  }
  TORCH_CHECK(false, "Unsupported runtime elementwise chain op: ", op);
  return "";
}

std::string runtime_elementwise_chain_key(const std::vector<std::string>& ops) {
  std::ostringstream key;
  for (const auto idx : c10::irange(ops.size())) {
    runtime_elementwise_chain_op_symbol(ops[idx]);
    if (idx > 0) {
      key << '_';
    }
    key << ops[idx];
  }
  return key.str();
}

std::string runtime_elementwise_chain_glsl(
    const std::vector<std::string>& ops) {
  TORCH_CHECK(!ops.empty(), "Runtime elementwise chain expects at least one op");
  TORCH_CHECK(
      ops.size() <= 16,
      "Runtime elementwise chain currently supports at most 16 ops");

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

)";
  for (const auto idx : c10::irange(ops.size() + 1u)) {
    const size_t buffer_binding = 2u + idx * 2u;
    const size_t meta_binding = buffer_binding + 1u;
    glsl << "layout(set = 0, binding = " << buffer_binding
         << ") buffer restrict readonly InBuffer" << idx << " {\n"
         << "  float data[];\n"
         << "} uInput" << idx << ";\n"
         << "layout(set = 0, binding = " << meta_binding
         << ") uniform restrict InMeta" << idx << " {\n"
         << "  uvec4 logical_sizes;\n"
         << "  uvec4 logical_strides;\n"
         << "  uvec4 physical_strides;\n"
         << "  uvec4 info;\n"
         << "} uInMeta" << idx << ";\n\n";
  }
  glsl << R"(layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

)";
  for (const auto idx : c10::irange(ops.size() + 1u)) {
    glsl << "float read_input" << idx << "(const uvec4 coord) {\n"
         << "  const uvec4 read_sizes = max(uInMeta" << idx
         << ".logical_sizes, uvec4(1));\n"
         << "  const uvec4 read_coord = min(coord, read_sizes - uvec4(1));\n"
         << "  const uint read_idx = coord_to_idx(read_coord, uInMeta" << idx
         << ".physical_strides) + uInMeta" << idx << ".info.w;\n"
         << "  if (read_idx >= uInMeta" << idx << ".info.z) {\n"
         << "    return 0.0;\n"
         << "  }\n"
         << "  return uInput" << idx << ".data[read_idx];\n"
         << "}\n\n";
  }
  glsl << R"(void zero_width_pack_padding(
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
  float value = read_input0(coord);
)";
  for (const auto idx : c10::irange(ops.size())) {
    glsl << "  value = value " << runtime_elementwise_chain_op_symbol(ops[idx])
         << " read_input" << (idx + 1u) << "(coord);\n";
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

std::vector<uint32_t> read_runtime_spirv_file(const std::filesystem::path& path) {
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

const std::vector<uint32_t>& runtime_elementwise_chain_spirv(
    const std::vector<std::string>& ops) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::vector<uint32_t>> cached_spirv;
  const std::string program_key = runtime_elementwise_chain_key(ops);
  std::lock_guard<std::mutex> lock(mutex);
  const auto cache_it = cached_spirv.find(program_key);
  if (cache_it != cached_spirv.end()) {
    return cache_it->second;
  }

  const std::string glslc_path =
      runtime_shader_env("PYTORCH_VULKAN_RUNTIME_SHADER_GLSLC");
  TORCH_CHECK(
      !glslc_path.empty(),
      "vulkan_prepack::runtime_elementwise_chain requires ",
      "PYTORCH_VULKAN_RUNTIME_SHADER_GLSLC to point at glslc");

  std::filesystem::path cache_dir =
      runtime_shader_env("PYTORCH_VULKAN_RUNTIME_SHADER_CACHE_DIR");
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
    glsl << runtime_elementwise_chain_glsl(ops);
  }

  std::ostringstream command;
  command << quote_runtime_shader_command_arg(glslc_path)
          << " -fshader-stage=compute "
          << quote_runtime_shader_command_arg(glsl_path.string()) << " -o "
          << quote_runtime_shader_command_arg(spv_path.string())
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
      "Runtime Vulkan shader compilation failed for ",
      glsl_path.string(),
      " with exit code ",
      compile_exit_code);

  const auto insert_result = cached_spirv.emplace(
      program_key,
      read_runtime_spirv_file(spv_path));
  return insert_result.first->second;
}

bool same_sizes_as(const Tensor& lhs, const Tensor& rhs) {
  return lhs.sizes().vec() == rhs.sizes().vec();
}

void check_runtime_elementwise_chain_tensor(
    const Tensor& tensor,
    const Tensor& reference,
    const char* name) {
  TORCH_CHECK(
      tensor.is_vulkan(),
      "runtime_elementwise_chain expects Vulkan tensor ",
      name);
  TORCH_CHECK(
      tensor.scalar_type() == at::kFloat,
      "runtime_elementwise_chain expects fp32 tensor ",
      name);
  TORCH_CHECK(
      same_sizes_as(tensor, reference) ||
          utils::broadcast_size(reference, tensor) == reference.sizes().vec(),
      "runtime_elementwise_chain expects RHS tensors broadcastable to input");
  const vTensor& v_tensor = convert(tensor);
  TORCH_CHECK(
      v_tensor.storage_type() == api::StorageType::BUFFER &&
          utils::supports_buffer_elementwise_compute(v_tensor) &&
          !v_tensor.is_quantized(),
      "runtime_elementwise_chain expects supported Vulkan buffer tensor ",
      name);
}

const char* runtime_elementwise_scalar_chain_op_symbol(const std::string& op) {
  if (op == "add") {
    return "+";
  }
  if (op == "mul") {
    return "*";
  }
  TORCH_CHECK(false, "Unsupported runtime scalar elementwise chain op: ", op);
  return "";
}

const char* runtime_elementwise_scalar_component(const size_t idx) {
  switch (idx) {
    case 0u:
      return "x";
    case 1u:
      return "y";
    case 2u:
      return "z";
    case 3u:
      return "w";
  }
  TORCH_CHECK(false, "Runtime scalar elementwise chain supports at most 4 ops");
  return "";
}

std::string runtime_elementwise_scalar_chain_key(
    const std::vector<std::string>& ops) {
  std::ostringstream key;
  key << "scalar";
  for (const std::string& op : ops) {
    runtime_elementwise_scalar_chain_op_symbol(op);
    key << '_' << op;
  }
  return key.str();
}

std::string runtime_elementwise_scalar_chain_glsl(
    const std::vector<std::string>& ops) {
  TORCH_CHECK(
      !ops.empty(),
      "Runtime scalar elementwise chain expects at least one op");
  TORCH_CHECK(
      ops.size() <= 4,
      "Runtime scalar elementwise chain currently supports at most 4 ops");

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
layout(set = 0, binding = 4) uniform restrict ScalarParams {
  vec4 scalars;
} uScalars;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float read_input(const uvec4 coord) {
  if (!all(lessThan(coord, uInMeta.logical_sizes))) {
    return 0.0;
  }
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
  for (const auto idx : c10::irange(ops.size())) {
    glsl << "  value = value "
         << runtime_elementwise_scalar_chain_op_symbol(ops[idx])
         << " uScalars.scalars."
         << runtime_elementwise_scalar_component(idx) << ";\n";
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

const std::vector<uint32_t>& runtime_elementwise_scalar_chain_spirv(
    const std::vector<std::string>& ops) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::vector<uint32_t>> cached_spirv;
  const std::string program_key = runtime_elementwise_scalar_chain_key(ops);
  std::lock_guard<std::mutex> lock(mutex);
  const auto cache_it = cached_spirv.find(program_key);
  if (cache_it != cached_spirv.end()) {
    return cache_it->second;
  }

  const std::string glslc_path =
      runtime_shader_env("PYTORCH_VULKAN_RUNTIME_SHADER_GLSLC");
  TORCH_CHECK(
      !glslc_path.empty(),
      "runtime scalar elementwise chain requires ",
      "PYTORCH_VULKAN_RUNTIME_SHADER_GLSLC to point at glslc");

  std::filesystem::path cache_dir =
      runtime_shader_env("PYTORCH_VULKAN_RUNTIME_SHADER_CACHE_DIR");
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
    glsl << runtime_elementwise_scalar_chain_glsl(ops);
  }

  std::ostringstream command;
  command << quote_runtime_shader_command_arg(glslc_path)
          << " -fshader-stage=compute "
          << quote_runtime_shader_command_arg(glsl_path.string()) << " -o "
          << quote_runtime_shader_command_arg(spv_path.string())
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
      "Runtime Vulkan scalar shader compilation failed for ",
      glsl_path.string(),
      " with exit code ",
      compile_exit_code);

  const auto insert_result = cached_spirv.emplace(
      program_key,
      read_runtime_spirv_file(spv_path));
  return insert_result.first->second;
}

Tensor run_runtime_elementwise_scalar_chain(
    const Tensor& input,
    const std::vector<float>& scalars,
    const std::vector<std::string>& ops) {
  TORCH_CHECK(
      !ops.empty() && ops.size() == scalars.size(),
      "runtime scalar elementwise chain expects one scalar per op");
  TORCH_CHECK(
      ops.size() <= 4,
      "runtime scalar elementwise chain currently supports 1 to 4 ops");
  for (const auto idx : c10::irange(ops.size())) {
    runtime_elementwise_scalar_chain_op_symbol(ops[idx]);
    TORCH_CHECK(
        std::isfinite(scalars[idx]),
        "runtime scalar elementwise chain expects finite scalar values");
  }
  check_runtime_elementwise_chain_tensor(input, input, "input");

  api::AllocationScope allocation_scope(
      "runtime_elementwise_chain.scalar_rhs");
  api::Context* const context = api::context();
  vTensor& v_input = convert(input);
  TORCH_CHECK(
      v_input.numel() > 0,
      "runtime scalar elementwise chain expects a non-empty tensor");

  vTensor v_output{
      context,
      input.sizes().vec(),
      v_input.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const std::vector<uint32_t>& spirv =
      runtime_elementwise_scalar_chain_spirv(ops);
  api::ShaderInfo shader_descriptor{
      "runtime_elementwise_chain." +
          runtime_elementwise_scalar_chain_key(ops),
      std::vector<uint32_t>(spirv.begin(), spirv.end()),
      {
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
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
  struct ScalarBlock final {
    api::utils::vec4 scalars;
  } block{{{0.0f, 0.0f, 0.0f, 0.0f}}};
  for (const auto idx : c10::irange(scalars.size())) {
    block.scalars.data[idx] = scalars[idx];
  }
  api::UniformParamsBuffer params(context, block);

  utils::log_vulkan_op_hit(
      "vulkan_prepack::runtime_elementwise_scalar_chain");
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
      in_meta.buffer(),
      params.buffer());

  return record_tensor_write_and_return(
      convert(v_output),
      "vulkan_prepack::runtime_elementwise_scalar_chain",
      "runtime_generated_elementwise_scalar_chain",
      {input});
}

utils::ElementwiseBroadcastOp elementwise_broadcast_op(
    const BinaryOpKind op_kind) {
  switch (op_kind) {
    case BinaryOpKind::Add:
      return utils::ElementwiseBroadcastOp::Add;
    case BinaryOpKind::Mul:
      return utils::ElementwiseBroadcastOp::Mul;
    case BinaryOpKind::Sub:
      return utils::ElementwiseBroadcastOp::Sub;
    case BinaryOpKind::Div:
    case BinaryOpKind::FloorDivide:
    case BinaryOpKind::Pow:
      return utils::ElementwiseBroadcastOp::Unsupported;
  }
  return utils::ElementwiseBroadcastOp::Unsupported;
}

TensorContractProvenance make_tensor_contract_provenance(
    const utils::ExecutionContractMetadata* metadata) {
  TensorContractProvenance provenance;
  if (metadata == nullptr) {
    return provenance;
  }
  provenance.contract_name = metadata->contract_name;
  provenance.contract_family = metadata->family_name;
  provenance.contract_tuple_id = metadata->tuple_id;
  provenance.contract_materialization_policy = metadata->materialization_policy;
  return provenance;
}

std::string format_binary_sizes(IntArrayRef sizes) {
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

void append_binary_tensor_summary(
    std::ostringstream& stream,
    const char* label,
    const vTensor& tensor) {
  stream << ' ' << label << '=' << format_binary_sizes(tensor.sizes())
         << ' ' << label
         << "_direct=" << (tensor.has_direct_buffer_layout() ? 1 : 0)
         << ' ' << label << "_offset=" << tensor.storage_offset()
         << ' ' << label << "_buffer_len="
         << (tensor.storage_type() == api::StorageType::BUFFER
                 ? tensor.buffer_length()
                 : -1);
}

void log_binary_submit(
    const BinaryOpKind op_kind,
    const char* route,
    const vTensor& v_self,
    const vTensor* v_other,
    const vTensor& v_output,
    const api::utils::uvec3& global_size,
    const api::utils::uvec3& local_size) {
  std::ostringstream stream;
  stream << "aten::binary_op.submit"
         << " op=" << binary_op_kind_name(op_kind)
         << " route=" << route;
  append_binary_tensor_summary(stream, "self", v_self);
  if (v_other != nullptr) {
    append_binary_tensor_summary(stream, "other", *v_other);
  }
  append_binary_tensor_summary(stream, "output", v_output);
  stream << " global=[" << global_size.data[0] << 'x' << global_size.data[1]
         << 'x' << global_size.data[2] << ']'
         << " local=[" << local_size.data[0] << 'x' << local_size.data[1]
         << 'x' << local_size.data[2] << ']';
  utils::log_vulkan_op_hit(stream.str());
}

struct RuntimeElementwiseLiveChain final {
  Tensor input;
  std::vector<Tensor> rhs_tensors;
  std::vector<float> scalar_rhs;
  std::vector<std::string> ops;
};

struct RuntimeElementwiseLiveChainState final {
  std::mutex mutex;
  std::unordered_map<const void*, RuntimeElementwiseLiveChain> chains;
  size_t sequence{0};
};

RuntimeElementwiseLiveChainState& runtime_elementwise_live_chain_state() {
  static RuntimeElementwiseLiveChainState state;
  return state;
}

thread_local bool runtime_elementwise_live_chain_probe_active = false;

bool runtime_elementwise_live_chain_execute_enabled() {
  const std::string value =
      runtime_shader_env("PYTORCH_VULKAN_RUNTIME_ELEMENTWISE_CHAIN_EXECUTE");
  return value == "1" || value == "true" || value == "TRUE";
}

bool runtime_elementwise_live_chain_check_output_enabled() {
  const std::string value = runtime_shader_env(
      "PYTORCH_VULKAN_RUNTIME_ELEMENTWISE_CHAIN_CHECK_OUTPUT");
  return value == "1" || value == "true" || value == "TRUE";
}

std::string runtime_elementwise_live_chain_log_path() {
  return runtime_shader_env("PYTORCH_VULKAN_RUNTIME_ELEMENTWISE_CHAIN_LIVE_LOG");
}

const char* runtime_elementwise_live_chain_op_name(
    const BinaryOpKind op_kind) {
  switch (op_kind) {
    case BinaryOpKind::Add:
      return "add";
    case BinaryOpKind::Sub:
      return "sub";
    case BinaryOpKind::Mul:
      return "mul";
    case BinaryOpKind::Div:
      return "div";
    case BinaryOpKind::FloorDivide:
    case BinaryOpKind::Pow:
      return nullptr;
  }
  return nullptr;
}

std::string runtime_json_quote(const std::string& value) {
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

void append_runtime_string_array(
    std::ostringstream& out,
    const std::vector<std::string>& values) {
  out << '[';
  for (const auto idx : c10::irange(values.size())) {
    if (idx > 0) {
      out << ',';
    }
    out << runtime_json_quote(values[idx]);
  }
  out << ']';
}

void append_runtime_operand_kind_array(
    std::ostringstream& out,
    const RuntimeElementwiseLiveChain& chain) {
  out << '[';
  const char* kind = chain.scalar_rhs.empty() ? "tensor" : "scalar";
  for (const auto idx : c10::irange(chain.ops.size())) {
    if (idx > 0) {
      out << ',';
    }
    out << runtime_json_quote(kind);
  }
  out << ']';
}

void append_runtime_shape_array(std::ostringstream& out, IntArrayRef sizes) {
  out << '[';
  for (const auto idx : c10::irange(sizes.size())) {
    if (idx > 0) {
      out << ',';
    }
    out << sizes[idx];
  }
  out << ']';
}

void log_runtime_elementwise_live_chain(
    const RuntimeElementwiseLiveChain& chain,
    const Tensor& output,
    const size_t sequence,
    const char* status,
    const bool executed,
    const std::string& detail) {
  const std::string log_path = runtime_elementwise_live_chain_log_path();
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
  row << ",\"source\":\"binary_op_tensor_buffer\"";
  row << ",\"behavior_change\":0";
  row << ",\"normal_eager_output_preserved\":1";
  row << ",\"status\":" << runtime_json_quote(status);
  row << ",\"executed\":" << (executed ? 1 : 0);
  row << ",\"chain_length\":" << chain.ops.size();
  row << ",\"ops\":";
  append_runtime_string_array(row, chain.ops);
  row << ",\"operand_kinds\":";
  append_runtime_operand_kind_array(row, chain);
  row << ",\"tensor_rhs_count\":" << chain.rhs_tensors.size();
  row << ",\"scalar_rhs_count\":" << chain.scalar_rhs.size();
  row << ",\"input_shape\":";
  append_runtime_shape_array(row, chain.input.sizes());
  row << ",\"output_shape\":";
  append_runtime_shape_array(row, output.sizes());
  if (!detail.empty()) {
    row << ",\"detail\":" << runtime_json_quote(detail);
  }
  row << "}\n";
  log << row.str();
}

bool runtime_live_chain_same_shape(
    const Tensor& lhs,
    const Tensor& rhs,
    const Tensor& output) {
  if (lhs.sizes().vec() != output.sizes().vec()) {
    return false;
  }
  try {
    utils::is_broadcastable(output, rhs);
    return utils::broadcast_size(output, rhs) == output.sizes().vec();
  } catch (const c10::Error&) {
    return false;
  }
}

bool runtime_live_chain_alpha_is_one(
    const std::optional<Scalar>& alpha_arg) {
  return !alpha_arg.has_value() || alpha_arg->to<float>() == 1.0f;
}

void maybe_probe_runtime_elementwise_live_chain(
    const Tensor& self,
    const Tensor& other,
    const Tensor& output,
    const std::optional<Scalar>& alpha_arg,
    const BinaryOpKind op_kind,
    const bool used_output_arg) {
  if (runtime_elementwise_live_chain_probe_active) {
    return;
  }
  const bool execute = runtime_elementwise_live_chain_execute_enabled();
  if (!execute && runtime_elementwise_live_chain_log_path().empty()) {
    return;
  }
  const char* op_name = runtime_elementwise_live_chain_op_name(op_kind);
  if (op_name == nullptr || used_output_arg ||
      !runtime_live_chain_alpha_is_one(alpha_arg) ||
      !runtime_live_chain_same_shape(self, other, output)) {
    return;
  }
  if (!self.is_vulkan() || !other.is_vulkan() || !output.is_vulkan() ||
      self.scalar_type() != kFloat || other.scalar_type() != kFloat ||
      output.scalar_type() != kFloat) {
    return;
  }

  RuntimeElementwiseLiveChain chain;
  size_t sequence = 0;
  {
    RuntimeElementwiseLiveChainState& state =
        runtime_elementwise_live_chain_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const void* self_key = self.unsafeGetTensorImpl();
    auto previous = state.chains.find(self_key);
    if (previous != state.chains.end()) {
      if (!previous->second.scalar_rhs.empty()) {
        return;
      }
      chain = previous->second;
    } else {
      chain.input = self;
    }
    chain.rhs_tensors.push_back(other);
    chain.ops.emplace_back(op_name);
    if (chain.ops.size() > 4u) {
      state.chains.erase(self_key);
      return;
    }
    const void* output_key = output.unsafeGetTensorImpl();
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
    runtime_elementwise_live_chain_probe_active = true;
    try {
      Tensor generated =
          run_runtime_elementwise_chain(chain.input, chain.rhs_tensors, chain.ops);
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
      detail = "unknown runtime elementwise live-chain execution error";
    }
    runtime_elementwise_live_chain_probe_active = false;
  }
  log_runtime_elementwise_live_chain(
      chain, output, sequence, status.c_str(), executed, detail);
}

bool runtime_live_scalar_op_supported(const BinaryOpKind op_kind) {
  return op_kind == BinaryOpKind::Add || op_kind == BinaryOpKind::Mul;
}

void maybe_probe_runtime_elementwise_live_chain_scalar(
    const Tensor& self,
    const Tensor& output,
    const float scalar,
    const std::optional<Scalar>& alpha_arg,
    const BinaryOpKind op_kind) {
  if (runtime_elementwise_live_chain_probe_active) {
    return;
  }
  const bool execute = runtime_elementwise_live_chain_execute_enabled();
  if (!execute && runtime_elementwise_live_chain_log_path().empty()) {
    return;
  }
  if (!runtime_live_scalar_op_supported(op_kind) ||
      !runtime_live_chain_alpha_is_one(alpha_arg) || !std::isfinite(scalar) ||
      !self.is_vulkan() || !output.is_vulkan() ||
      self.scalar_type() != kFloat || output.scalar_type() != kFloat ||
      self.sizes().vec() != output.sizes().vec()) {
    return;
  }

  RuntimeElementwiseLiveChain chain;
  size_t sequence = 0;
  {
    RuntimeElementwiseLiveChainState& state =
        runtime_elementwise_live_chain_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const void* self_key = self.unsafeGetTensorImpl();
    auto previous = state.chains.find(self_key);
    if (previous != state.chains.end()) {
      if (!previous->second.rhs_tensors.empty()) {
        return;
      }
      chain = previous->second;
    } else {
      chain.input = self;
    }
    chain.scalar_rhs.push_back(scalar);
    chain.ops.emplace_back(binary_op_kind_name(op_kind));
    if (chain.ops.size() > 4u) {
      state.chains.erase(self_key);
      return;
    }
    const void* output_key = output.unsafeGetTensorImpl();
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
    runtime_elementwise_live_chain_probe_active = true;
    try {
      Tensor generated = run_runtime_elementwise_scalar_chain(
          chain.input, chain.scalar_rhs, chain.ops);
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
      detail = "unknown runtime elementwise scalar live-chain execution error";
    }
    runtime_elementwise_live_chain_probe_active = false;
  }
  log_runtime_elementwise_live_chain(
      chain, output, sequence, status.c_str(), executed, detail);
}

enum class RuntimeElementwiseMixedOperandKind : uint8_t {
  Tensor,
  Unary,
};

struct RuntimeElementwiseMixedStep final {
  RuntimeElementwiseMixedOperandKind operand_kind{
      RuntimeElementwiseMixedOperandKind::Unary};
  std::string op;
  Tensor rhs;
};

struct RuntimeElementwiseMixedChain final {
  Tensor input;
  std::vector<RuntimeElementwiseMixedStep> steps;
};

struct RuntimeElementwiseMixedChainState final {
  std::mutex mutex;
  std::unordered_map<const void*, RuntimeElementwiseMixedChain> chains;
  size_t sequence{0};
};

RuntimeElementwiseMixedChainState& runtime_elementwise_mixed_chain_state() {
  static RuntimeElementwiseMixedChainState state;
  return state;
}

bool runtime_elementwise_mixed_unary_supported(const std::string& op) {
  return op == "exp" || op == "sqrt" || op == "log" || op == "sin" ||
      op == "cos" || op == "neg" || op == "reciprocal" || op == "rsqrt" ||
      op == "silu";
}

std::string runtime_elementwise_mixed_unary_expression(
    const std::string& op) {
  if (op == "exp") {
    return "exp(value)";
  }
  if (op == "sqrt") {
    return "sqrt(value)";
  }
  if (op == "log") {
    return "log(value)";
  }
  if (op == "sin") {
    return "sin(value)";
  }
  if (op == "cos") {
    return "cos(value)";
  }
  if (op == "neg") {
    return "-value";
  }
  if (op == "reciprocal") {
    return "1.0 / value";
  }
  if (op == "rsqrt") {
    return "inversesqrt(value)";
  }
  if (op == "silu") {
    return "value / (1.0 + exp(-value))";
  }
  TORCH_CHECK(false, "Unsupported runtime mixed unary op: ", op);
  return "";
}

size_t runtime_elementwise_mixed_tensor_rhs_count(
    const RuntimeElementwiseMixedChain& chain) {
  size_t count = 0;
  for (const RuntimeElementwiseMixedStep& step : chain.steps) {
    if (step.operand_kind == RuntimeElementwiseMixedOperandKind::Tensor) {
      ++count;
    }
  }
  return count;
}

bool runtime_elementwise_mixed_supported(
    const RuntimeElementwiseMixedChain& chain) {
  if (chain.steps.empty() || chain.steps.size() > 8u) {
    return false;
  }
  if (runtime_elementwise_mixed_tensor_rhs_count(chain) > 4u) {
    return false;
  }
  for (const RuntimeElementwiseMixedStep& step : chain.steps) {
    if (step.operand_kind == RuntimeElementwiseMixedOperandKind::Tensor) {
      try {
        runtime_elementwise_chain_op_symbol(step.op);
      } catch (const c10::Error&) {
        return false;
      }
    } else if (!runtime_elementwise_mixed_unary_supported(step.op)) {
      return false;
    }
  }
  return true;
}

std::string runtime_elementwise_mixed_chain_key(
    const RuntimeElementwiseMixedChain& chain) {
  std::ostringstream key;
  key << "mixed";
  for (const RuntimeElementwiseMixedStep& step : chain.steps) {
    key << '_'
        << (step.operand_kind == RuntimeElementwiseMixedOperandKind::Tensor
                ? "tensor"
                : "unary")
        << '_' << step.op;
  }
  return key.str();
}

std::string runtime_elementwise_mixed_chain_glsl(
    const RuntimeElementwiseMixedChain& chain) {
  TORCH_CHECK(
      runtime_elementwise_mixed_supported(chain),
      "Unsupported runtime mixed elementwise chain");
  const size_t rhs_count = runtime_elementwise_mixed_tensor_rhs_count(chain);

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

)";
  for (const auto idx : c10::irange(rhs_count)) {
    const size_t buffer_binding = 4u + idx * 2u;
    const size_t meta_binding = buffer_binding + 1u;
    glsl << "layout(set = 0, binding = " << buffer_binding
         << ") buffer restrict readonly RhsBuffer" << idx << " {\n"
         << "  float data[];\n"
         << "} uRhs" << idx << ";\n"
         << "layout(set = 0, binding = " << meta_binding
         << ") uniform restrict RhsMeta" << idx << " {\n"
         << "  uvec4 logical_sizes;\n"
         << "  uvec4 logical_strides;\n"
         << "  uvec4 physical_strides;\n"
         << "  uvec4 info;\n"
         << "} uRhsMeta" << idx << ";\n\n";
  }
  glsl << R"(layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float read_input(const uvec4 coord) {
  const uint read_idx =
      coord_to_idx(coord, uInMeta.physical_strides) + uInMeta.info.w;
  if (read_idx >= uInMeta.info.z) {
    return 0.0;
  }
  return uInput.data[read_idx];
}

)";
  for (const auto idx : c10::irange(rhs_count)) {
    glsl << "float read_rhs" << idx << "(const uvec4 coord) {\n"
         << "  const uvec4 read_sizes = max(uRhsMeta" << idx
         << ".logical_sizes, uvec4(1));\n"
         << "  const uvec4 read_coord = min(coord, read_sizes - uvec4(1));\n"
         << "  const uint read_idx = coord_to_idx(read_coord, uRhsMeta" << idx
         << ".physical_strides) + uRhsMeta" << idx << ".info.w;\n"
         << "  if (read_idx >= uRhsMeta" << idx << ".info.z) {\n"
         << "    return 0.0;\n"
         << "  }\n"
         << "  return uRhs" << idx << ".data[read_idx];\n"
         << "}\n\n";
  }
  glsl << R"(void zero_width_pack_padding(
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
  size_t rhs_idx = 0;
  for (const RuntimeElementwiseMixedStep& step : chain.steps) {
    if (step.operand_kind == RuntimeElementwiseMixedOperandKind::Tensor) {
      glsl << "  value = value " << runtime_elementwise_chain_op_symbol(step.op)
           << " read_rhs" << rhs_idx << "(coord);\n";
      ++rhs_idx;
    } else {
      glsl << "  value = "
           << runtime_elementwise_mixed_unary_expression(step.op) << ";\n";
    }
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

const std::vector<uint32_t>& runtime_elementwise_mixed_chain_spirv(
    const RuntimeElementwiseMixedChain& chain) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::vector<uint32_t>> cached_spirv;
  const std::string program_key = runtime_elementwise_mixed_chain_key(chain);
  std::lock_guard<std::mutex> lock(mutex);
  const auto cache_it = cached_spirv.find(program_key);
  if (cache_it != cached_spirv.end()) {
    return cache_it->second;
  }

  const std::string glslc_path =
      runtime_shader_env("PYTORCH_VULKAN_RUNTIME_SHADER_GLSLC");
  TORCH_CHECK(
      !glslc_path.empty(),
      "runtime mixed elementwise chain requires ",
      "PYTORCH_VULKAN_RUNTIME_SHADER_GLSLC to point at glslc");

  std::filesystem::path cache_dir =
      runtime_shader_env("PYTORCH_VULKAN_RUNTIME_SHADER_CACHE_DIR");
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
    glsl << runtime_elementwise_mixed_chain_glsl(chain);
  }

  std::ostringstream command;
  command << quote_runtime_shader_command_arg(glslc_path)
          << " -fshader-stage=compute "
          << quote_runtime_shader_command_arg(glsl_path.string()) << " -o "
          << quote_runtime_shader_command_arg(spv_path.string())
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
      "Runtime Vulkan mixed shader compilation failed for ",
      glsl_path.string(),
      " with exit code ",
      compile_exit_code);

  const auto insert_result = cached_spirv.emplace(
      program_key,
      read_runtime_spirv_file(spv_path));
  return insert_result.first->second;
}

void append_runtime_mixed_operand_kind_array(
    std::ostringstream& out,
    const RuntimeElementwiseMixedChain& chain) {
  out << '[';
  for (const auto idx : c10::irange(chain.steps.size())) {
    if (idx > 0) {
      out << ',';
    }
    out << runtime_json_quote(
        chain.steps[idx].operand_kind == RuntimeElementwiseMixedOperandKind::Tensor
            ? "tensor"
            : "unary");
  }
  out << ']';
}

std::vector<std::string> runtime_elementwise_mixed_ops(
    const RuntimeElementwiseMixedChain& chain) {
  std::vector<std::string> ops;
  ops.reserve(chain.steps.size());
  for (const RuntimeElementwiseMixedStep& step : chain.steps) {
    ops.push_back(step.op);
  }
  return ops;
}

const char* runtime_elementwise_mixed_source(
    const RuntimeElementwiseMixedChain& chain) {
  bool has_tensor = false;
  bool has_unary = false;
  for (const RuntimeElementwiseMixedStep& step : chain.steps) {
    has_tensor |= step.operand_kind == RuntimeElementwiseMixedOperandKind::Tensor;
    has_unary |= step.operand_kind == RuntimeElementwiseMixedOperandKind::Unary;
  }
  if (has_tensor && has_unary) {
    return "mixed_elementwise_live_chain";
  }
  if (has_unary) {
    return "unary_op_buffer";
  }
  return "binary_op_tensor_buffer";
}

void log_runtime_elementwise_mixed_live_chain(
    const RuntimeElementwiseMixedChain& chain,
    const Tensor& output,
    const size_t sequence,
    const char* status,
    const bool executed,
    const std::string& detail,
    const std::optional<double>& output_check_max_abs = std::nullopt) {
  const std::string log_path = runtime_elementwise_live_chain_log_path();
  if (log_path.empty()) {
    return;
  }
  std::ofstream log(log_path, std::ios::app);
  if (!log) {
    return;
  }
  const std::vector<std::string> ops = runtime_elementwise_mixed_ops(chain);
  const size_t tensor_rhs_count =
      runtime_elementwise_mixed_tensor_rhs_count(chain);
  std::ostringstream row;
  row << "{\"schema\":\"VulkanRuntimeElementwiseLiveChainTrace.v0\"";
  row << ",\"sequence\":" << sequence;
  row << ",\"family\":\"ElementwiseChain\"";
  row << ",\"source\":" << runtime_json_quote(
      runtime_elementwise_mixed_source(chain));
  row << ",\"behavior_change\":0";
  row << ",\"normal_eager_output_preserved\":1";
  row << ",\"status\":" << runtime_json_quote(status);
  row << ",\"executed\":" << (executed ? 1 : 0);
  row << ",\"chain_length\":" << chain.steps.size();
  row << ",\"ops\":";
  append_runtime_string_array(row, ops);
  row << ",\"operand_kinds\":";
  append_runtime_mixed_operand_kind_array(row, chain);
  row << ",\"tensor_rhs_count\":" << tensor_rhs_count;
  row << ",\"scalar_rhs_count\":0";
  row << ",\"input_shape\":";
  append_runtime_shape_array(row, chain.input.sizes());
  row << ",\"output_shape\":";
  append_runtime_shape_array(row, output.sizes());
  if (output_check_max_abs.has_value()) {
    row << ",\"output_check_max_abs\":" << *output_check_max_abs;
  }
  if (!detail.empty()) {
    row << ",\"detail\":" << runtime_json_quote(detail);
  }
  row << "}\n";
  log << row.str();
}

bool runtime_elementwise_mixed_input_supported(const Tensor& tensor) {
  if (!tensor.is_vulkan() || tensor.scalar_type() != kFloat) {
    return false;
  }
  const vTensor& v_tensor = convert(tensor);
  return v_tensor.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_elementwise_compute(v_tensor) &&
      !v_tensor.is_quantized();
}

Tensor run_runtime_elementwise_mixed_chain(
    const RuntimeElementwiseMixedChain& chain) {
  TORCH_CHECK(
      runtime_elementwise_mixed_supported(chain),
      "Unsupported runtime mixed elementwise chain");
  TORCH_CHECK(
      runtime_elementwise_mixed_input_supported(chain.input),
      "runtime mixed elementwise chain expects a Vulkan fp32 buffer input");

  std::vector<Tensor> rhs_tensors;
  rhs_tensors.reserve(runtime_elementwise_mixed_tensor_rhs_count(chain));
  for (const RuntimeElementwiseMixedStep& step : chain.steps) {
    if (step.operand_kind == RuntimeElementwiseMixedOperandKind::Tensor) {
      check_runtime_elementwise_chain_tensor(
          step.rhs, chain.input, "rhs");
      rhs_tensors.push_back(step.rhs);
    }
  }

  api::AllocationScope allocation_scope(
      "runtime_elementwise_chain.mixed");
  api::Context* const context = api::context();
  vTensor& v_input = convert(chain.input);
  TORCH_CHECK(
      v_input.numel() > 0,
      "runtime mixed elementwise chain expects a non-empty tensor");

  vTensor v_output{
      context,
      chain.input.sizes().vec(),
      v_input.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  std::vector<vTensor*> v_rhs;
  v_rhs.reserve(rhs_tensors.size());
  for (const Tensor& rhs : rhs_tensors) {
    v_rhs.push_back(&convert(rhs));
  }

  std::vector<VkDescriptorType> layout{
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
  };
  layout.resize(4u + v_rhs.size() * 2u);
  for (const auto idx : c10::irange(v_rhs.size())) {
    layout[4u + idx * 2u] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layout[5u + idx * 2u] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  }

  const std::vector<uint32_t>& spirv =
      runtime_elementwise_mixed_chain_spirv(chain);
  api::ShaderInfo shader_descriptor{
      "runtime_elementwise_chain." + runtime_elementwise_mixed_chain_key(chain),
      std::vector<uint32_t>(spirv.begin(), spirv.end()),
      layout,
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
  std::vector<api::UniformParamsBuffer> rhs_metas;
  rhs_metas.reserve(v_rhs.size());
  for (const vTensor* rhs : v_rhs) {
    rhs_metas.emplace_back(utils::make_buffer_compute_metadata_ubo(context, *rhs));
  }

  utils::log_vulkan_op_hit("vulkan_prepack::runtime_mixed_elementwise_chain");
  if (v_rhs.empty()) {
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
  } else if (v_rhs.size() == 1u) {
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
        in_meta.buffer(),
        v_rhs[0]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        rhs_metas[0].buffer());
  } else if (v_rhs.size() == 2u) {
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
        in_meta.buffer(),
        v_rhs[0]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        rhs_metas[0].buffer(),
        v_rhs[1]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        rhs_metas[1].buffer());
  } else if (v_rhs.size() == 3u) {
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
        in_meta.buffer(),
        v_rhs[0]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        rhs_metas[0].buffer(),
        v_rhs[1]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        rhs_metas[1].buffer(),
        v_rhs[2]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        rhs_metas[2].buffer());
  } else {
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
        in_meta.buffer(),
        v_rhs[0]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        rhs_metas[0].buffer(),
        v_rhs[1]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        rhs_metas[1].buffer(),
        v_rhs[2]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        rhs_metas[2].buffer(),
        v_rhs[3]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        rhs_metas[3].buffer());
  }

  std::vector<Tensor> provenance_inputs;
  provenance_inputs.reserve(rhs_tensors.size() + 1u);
  provenance_inputs.push_back(chain.input);
  provenance_inputs.insert(
      provenance_inputs.end(), rhs_tensors.begin(), rhs_tensors.end());
  return record_tensor_write_and_return(
      convert(v_output),
      "vulkan_prepack::runtime_mixed_elementwise_chain",
      "runtime_generated_mixed_elementwise_chain",
      provenance_inputs);
}

void record_runtime_elementwise_mixed_chain_step(
    const Tensor& self,
    const Tensor& output,
    RuntimeElementwiseMixedStep step) {
  if (runtime_elementwise_live_chain_probe_active) {
    return;
  }
  const bool execute = runtime_elementwise_live_chain_execute_enabled();
  if (!execute && runtime_elementwise_live_chain_log_path().empty()) {
    return;
  }
  if (!runtime_elementwise_mixed_input_supported(self) ||
      !runtime_elementwise_mixed_input_supported(output) ||
      self.sizes().vec() != output.sizes().vec()) {
    return;
  }
  if (step.operand_kind == RuntimeElementwiseMixedOperandKind::Tensor &&
      !runtime_live_chain_same_shape(self, step.rhs, output)) {
    return;
  }

  RuntimeElementwiseMixedChain chain;
  size_t sequence = 0;
  {
    RuntimeElementwiseMixedChainState& state =
        runtime_elementwise_mixed_chain_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    const void* self_key = self.unsafeGetTensorImpl();
    auto previous = state.chains.find(self_key);
    if (previous != state.chains.end()) {
      chain = previous->second;
    } else {
      chain.input = self;
    }
    chain.steps.push_back(std::move(step));
    if (!runtime_elementwise_mixed_supported(chain)) {
      state.chains.erase(self_key);
      return;
    }
    const void* output_key = output.unsafeGetTensorImpl();
    state.chains[output_key] = chain;
    if (state.chains.size() > 256u) {
      state.chains.erase(state.chains.begin());
    }
    sequence = ++state.sequence;
  }

  bool executed = false;
  std::string status = "captured";
  std::string detail;
  std::optional<double> output_check_max_abs;
  if (execute && chain.steps.size() >= 2u) {
    runtime_elementwise_live_chain_probe_active = true;
    try {
      Tensor generated = run_runtime_elementwise_mixed_chain(chain);
      if (runtime_elementwise_live_chain_check_output_enabled()) {
        const Tensor difference = at::abs(at::sub(generated.cpu(), output.cpu()));
        output_check_max_abs = at::max(difference).item<double>();
      }
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
      detail = "unknown runtime mixed live-chain execution error";
    }
    runtime_elementwise_live_chain_probe_active = false;
  }
  log_runtime_elementwise_mixed_live_chain(
      chain,
      output,
      sequence,
      status.c_str(),
      executed,
      detail,
      output_check_max_abs);
}

struct DeferredImageNormalizeCandidate final {
  Tensor input;
  Tensor mean;
  Tensor std;
  std::vector<int64_t> output_sizes;
  std::vector<int64_t> output_strides;
  std::vector<int64_t> output_physical_strides;
  int64_t storage_offset{0};
  uint64_t producer_storage_id{0};
  uint64_t producer_generation{0};
  uint64_t producer_logical_desc_hash{0};
  float scale{1.0f};
  bool has_mean{false};
  bool has_std{false};
};

constexpr size_t kMaxDeferredImageNormalizeCandidates = 64;
thread_local bool g_materializing_deferred_image_normalize = false;

struct DeferredTensorProducerKey final {
  uint64_t base_storage_id{0};
  uint64_t generation{0};
  uint64_t logical_desc_hash{0};
  const char* producer_op{"aten::image_normalize"};
};

bool operator==(
    const DeferredTensorProducerKey& lhs,
    const DeferredTensorProducerKey& rhs) {
  return lhs.base_storage_id == rhs.base_storage_id &&
      lhs.generation == rhs.generation &&
      lhs.logical_desc_hash == rhs.logical_desc_hash &&
      lhs.producer_op == rhs.producer_op;
}

struct DeferredTensorProducerKeyHash final {
  size_t operator()(const DeferredTensorProducerKey& key) const {
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

DeferredTensorProducerKey deferred_image_normalize_key(const Tensor& tensor) {
  const VulkanTensorStateDesc state = inspect_tensor_state(tensor);
  return DeferredTensorProducerKey{
      state.storage_id,
      state.generation,
      state.logical_desc_hash,
      "aten::image_normalize"};
}

std::mutex& deferred_image_normalize_candidate_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<
    DeferredTensorProducerKey,
    DeferredImageNormalizeCandidate,
    DeferredTensorProducerKeyHash>&
deferred_image_normalize_candidates() {
  static std::unordered_map<
      DeferredTensorProducerKey,
      DeferredImageNormalizeCandidate,
      DeferredTensorProducerKeyHash>
      candidates;
  return candidates;
}

bool can_retarget_deferred_image_normalize_candidate(
    const Tensor& tensor,
    const DeferredImageNormalizeCandidate& candidate) {
  const VulkanTensorStateDesc state = inspect_tensor_state(tensor);
  return state.storage_id == candidate.producer_storage_id &&
      state.generation == candidate.producer_generation &&
      state.logical_desc_hash == candidate.producer_logical_desc_hash;
}

std::optional<DeferredImageNormalizeCandidate>
lookup_deferred_image_normalize_candidate(const Tensor& tensor) {
  std::lock_guard<std::mutex> lock(
      deferred_image_normalize_candidate_mutex());
  auto& candidates = deferred_image_normalize_candidates();
  const auto it = candidates.find(deferred_image_normalize_key(tensor));
  if (it == candidates.end()) {
    return std::nullopt;
  }
  if (!can_retarget_deferred_image_normalize_candidate(tensor, it->second)) {
    utils::log_vulkan_op_hit("aten::image_normalize_bridge.stale_candidate");
    candidates.erase(it);
    return std::nullopt;
  }
  return it->second;
}

std::optional<DeferredImageNormalizeCandidate>
take_deferred_image_normalize_candidate(const Tensor& tensor) {
  std::lock_guard<std::mutex> lock(
      deferred_image_normalize_candidate_mutex());
  auto& candidates = deferred_image_normalize_candidates();
  const auto it = candidates.find(deferred_image_normalize_key(tensor));
  if (it == candidates.end()) {
    return std::nullopt;
  }
  if (!can_retarget_deferred_image_normalize_candidate(tensor, it->second)) {
    utils::log_vulkan_op_hit("aten::image_normalize_bridge.stale_candidate");
    candidates.erase(it);
    return std::nullopt;
  }
  DeferredImageNormalizeCandidate candidate = it->second;
  candidates.erase(it);
  return candidate;
}

void register_deferred_image_normalize_candidate(
    const Tensor& tensor,
    DeferredImageNormalizeCandidate candidate) {
  std::lock_guard<std::mutex> lock(
      deferred_image_normalize_candidate_mutex());
  auto& candidates = deferred_image_normalize_candidates();
  if (candidates.size() >= kMaxDeferredImageNormalizeCandidates) {
    utils::log_vulkan_op_hit("aten::image_normalize_bridge.registry_clear");
    candidates.clear();
  }
  const VulkanTensorStateDesc state = inspect_tensor_state(tensor);
  candidate.producer_storage_id = state.storage_id;
  candidate.producer_generation = state.generation;
  candidate.producer_logical_desc_hash = state.logical_desc_hash;
  candidates[deferred_image_normalize_key(tensor)] = std::move(candidate);
}

class DeferredImageNormalizeMaterializeGuard final {
 public:
  DeferredImageNormalizeMaterializeGuard() {
    previous_ = g_materializing_deferred_image_normalize;
    g_materializing_deferred_image_normalize = true;
  }

  ~DeferredImageNormalizeMaterializeGuard() {
    g_materializing_deferred_image_normalize = previous_;
  }

 private:
  bool previous_{false};
};

bool is_deep_desktop_hwc_rgb_float_buffer(const Tensor& tensor) {
  if (
      g_materializing_deferred_image_normalize || !tensor.is_vulkan() ||
      tensor.scalar_type() != kFloat || tensor.dim() != 3 ||
      tensor.size(0) < 64 || tensor.size(1) < 64 || tensor.size(2) != 3) {
    return false;
  }
  const vTensor& v_tensor = convert(tensor);
  return v_tensor.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_elementwise_compute(v_tensor);
}

bool is_rgb_vector_tensor(const Tensor& tensor) {
  if (!tensor.defined() || tensor.dim() != 1 || tensor.size(0) != 3) {
    return false;
  }
  if (tensor.is_vulkan()) {
    const vTensor& v_tensor = convert(tensor);
    return tensor.scalar_type() == kFloat &&
        v_tensor.storage_type() == api::StorageType::BUFFER &&
        utils::supports_buffer_elementwise_compute(v_tensor);
  }
  return tensor.scalar_type() == kFloat;
}

Tensor make_deferred_image_normalize_placeholder(
    IntArrayRef sizes,
    const ScalarType dtype) {
  api::Context* const context = api::context();
  return utils::mark_tensor_execution(
      convert(vTensor{
          context,
          sizes.vec(),
          convert_dtype(dtype),
          api::StorageType::BUFFER,
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      }),
      api::ExecutionLayout::BUFFER_DIRECT);
}

std::vector<int64_t> current_logical_strides(const Tensor& tensor) {
  if (!tensor.is_vulkan()) {
    return tensor.strides().vec();
  }
  const c10::DimVector strides = logical_strides(convert(tensor));
  return std::vector<int64_t>(strides.begin(), strides.end());
}

std::vector<int64_t> current_physical_strides(const Tensor& tensor) {
  if (!tensor.is_vulkan()) {
    return tensor.strides().vec();
  }
  const std::vector<int64_t>& strides = convert(tensor).gpu_strides();
  return std::vector<int64_t>(strides.begin(), strides.end());
}

void update_deferred_image_normalize_view(
    DeferredImageNormalizeCandidate& candidate,
    const Tensor& tensor) {
  candidate.output_sizes = tensor.sizes().vec();
  candidate.output_strides = current_logical_strides(tensor);
  candidate.output_physical_strides = current_physical_strides(tensor);
  candidate.storage_offset =
      tensor.is_vulkan() ? convert(tensor).storage_offset() : 0;
}

Tensor apply_deferred_image_normalize_view_if_needed(
    const Tensor& base,
    const DeferredImageNormalizeCandidate& candidate) {
  if (
      base.sizes().vec() == candidate.output_sizes &&
      current_logical_strides(base) == candidate.output_strides &&
      current_physical_strides(base) == candidate.output_physical_strides &&
      candidate.storage_offset == 0) {
    return base;
  }
  if (base.is_vulkan()) {
    return make_buffer_metadata_view_checked(
        base,
        candidate.output_sizes,
        candidate.output_strides,
        candidate.output_physical_strides,
        candidate.storage_offset,
        "aten::image_normalize_deferred_view");
  }
  return at::as_strided(
      base,
      candidate.output_sizes,
      candidate.output_strides,
      candidate.storage_offset);
}

Tensor run_deferred_image_normalize_fused(
    const DeferredImageNormalizeCandidate& candidate) {
  TORCH_INTERNAL_ASSERT(candidate.has_mean && candidate.has_std);
  api::Context* const context = api::context();

  Tensor input = utils::prepare_vulkan_execution_tensor(
      candidate.input, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  Tensor mean = candidate.mean.is_vulkan() ? candidate.mean : candidate.mean.vulkan();
  mean = utils::prepare_vulkan_execution_tensor(
      mean, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  Tensor std = candidate.std.is_vulkan() ? candidate.std : candidate.std.vulkan();
  std = utils::prepare_vulkan_execution_tensor(
      std, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);

  const vTensor& v_input = convert(input);
  const vTensor& v_mean = convert(mean);
  const vTensor& v_std = convert(std);
  vTensor v_output{
      context,
      candidate.input.sizes().vec(),
      api::kFloat,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct Block final {
    vec4 params;
  } block{{candidate.scale, 0.0f, 0.0f, 0.0f}};

  api::UniformParamsBuffer params(context, block);
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);
  api::UniformParamsBuffer mean_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_mean);
  api::UniformParamsBuffer std_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_std);
  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size{
      safe_downcast<uint32_t>(v_output.numel()),
      1u,
      1u,
  };

  utils::log_vulkan_op_hit("aten::image_normalize_bridge.fused");
  context->submit_compute_job(
      VK_KERNEL(buffer_image_normalize_float),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer(),
      v_mean.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      mean_meta.buffer(),
      v_std.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      std_meta.buffer(),
      params.buffer());

  return utils::mark_tensor_execution(
      convert(v_output), api::ExecutionLayout::BUFFER_DIRECT);
}

Tensor materialize_deferred_image_normalize_candidate_impl(
    const Tensor& tensor) {
  auto candidate = take_deferred_image_normalize_candidate(tensor);
  if (!candidate.has_value()) {
    return tensor;
  }

  if (candidate->has_mean && candidate->has_std) {
    Tensor normalized = run_deferred_image_normalize_fused(*candidate);
    return apply_deferred_image_normalize_view_if_needed(
        normalized, *candidate);
  }

  DeferredImageNormalizeMaterializeGuard guard;
  utils::log_vulkan_op_hit("aten::image_normalize_bridge.materialize");
  Tensor normalized = candidate->input.mul(candidate->scale);
  if (candidate->has_mean) {
    normalized = normalized.sub(candidate->mean);
  }
  if (candidate->has_std) {
    normalized = normalized.div(candidate->std);
  }
  return apply_deferred_image_normalize_view_if_needed(normalized, *candidate);
}

std::optional<Tensor> try_start_deferred_image_normalize_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const BinaryOpKind op_kind) {
  if (
      op_kind != BinaryOpKind::Div && op_kind != BinaryOpKind::Mul ||
      !is_deep_desktop_hwc_rgb_float_buffer(self_arg)) {
    return std::nullopt;
  }

  const float scale = other.to<float>();
  if (scale <= 0.0f || scale > 1.0f) {
    return std::nullopt;
  }

  Tensor placeholder =
      make_deferred_image_normalize_placeholder(self_arg.sizes(), kFloat);
  DeferredImageNormalizeCandidate candidate;
  candidate.input = self_arg;
  candidate.mean = Tensor();
  candidate.std = Tensor();
  candidate.output_sizes = placeholder.sizes().vec();
  candidate.output_strides = current_logical_strides(placeholder);
  candidate.output_physical_strides = current_physical_strides(placeholder);
  candidate.storage_offset = convert(placeholder).storage_offset();
  candidate.scale = scale;
  candidate.has_mean = false;
  candidate.has_std = false;
  register_deferred_image_normalize_candidate(
      placeholder, std::move(candidate));
  utils::log_vulkan_op_hit("aten::image_normalize_bridge.defer_scale");
  return placeholder;
}

std::optional<Tensor> try_start_deferred_image_normalize_tensor_scale(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const BinaryOpKind op_kind) {
  if (
      alpha_arg.has_value() ||
      (op_kind != BinaryOpKind::Div && op_kind != BinaryOpKind::Mul) ||
      !is_deep_desktop_hwc_rgb_float_buffer(self_arg) ||
      other_arg.is_vulkan() || other_arg.dim() != 0 ||
      !c10::isFloatingType(other_arg.scalar_type())) {
    return std::nullopt;
  }

  const float other_value = other_arg.item<float>();
  if (other_value <= 0.0f) {
    return std::nullopt;
  }
  const float scale =
      op_kind == BinaryOpKind::Div ? 1.0f / other_value : other_value;
  if (scale <= 0.0f || scale > 1.0f) {
    return std::nullopt;
  }

  Tensor placeholder =
      make_deferred_image_normalize_placeholder(self_arg.sizes(), kFloat);
  DeferredImageNormalizeCandidate candidate;
  candidate.input = self_arg;
  candidate.mean = Tensor();
  candidate.std = Tensor();
  candidate.output_sizes = placeholder.sizes().vec();
  candidate.output_strides = current_logical_strides(placeholder);
  candidate.output_physical_strides = current_physical_strides(placeholder);
  candidate.storage_offset = convert(placeholder).storage_offset();
  candidate.scale = scale;
  candidate.has_mean = false;
  candidate.has_std = false;
  register_deferred_image_normalize_candidate(
      placeholder, std::move(candidate));
  utils::log_vulkan_op_hit("aten::image_normalize_bridge.defer_scale");
  return placeholder;
}

std::optional<Tensor> try_start_deferred_attention_query_scale_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const BinaryOpKind op_kind) {
  if (
      alpha_arg.has_value() ||
      op_kind != BinaryOpKind::Mul ||
      other_arg.is_vulkan() ||
      other_arg.dim() != 0 ||
      !c10::isFloatingType(other_arg.scalar_type())) {
    return std::nullopt;
  }
  return try_start_deferred_attention_query_scale(
      self_arg, Scalar(other_arg.item<float>()));
}

std::optional<Tensor> try_update_deferred_image_normalize_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const BinaryOpKind op_kind) {
  auto candidate = lookup_deferred_image_normalize_candidate(self_arg);
  if (!candidate.has_value()) {
    return std::nullopt;
  }

  if (alpha_arg.has_value() && alpha_arg->to<float>() != 1.0f) {
    return std::nullopt;
  }

  Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();
  if (!is_rgb_vector_tensor(other)) {
    return std::nullopt;
  }

  if (op_kind == BinaryOpKind::Sub && !candidate->has_mean) {
    auto taken = take_deferred_image_normalize_candidate(self_arg);
    if (!taken.has_value()) {
      return std::nullopt;
    }
    Tensor placeholder =
        make_deferred_image_normalize_placeholder(self_arg.sizes(), kFloat);
    taken->mean = other;
    taken->has_mean = true;
    update_deferred_image_normalize_view(*taken, placeholder);
    register_deferred_image_normalize_candidate(
        placeholder, std::move(*taken));
    utils::log_vulkan_op_hit("aten::image_normalize_bridge.defer_mean");
    return placeholder;
  }

  if (
      op_kind == BinaryOpKind::Div && candidate->has_mean &&
      !candidate->has_std) {
    auto taken = take_deferred_image_normalize_candidate(self_arg);
    if (!taken.has_value()) {
      return std::nullopt;
    }
    Tensor placeholder =
        make_deferred_image_normalize_placeholder(self_arg.sizes(), kFloat);
    taken->std = other;
    taken->has_std = true;
    update_deferred_image_normalize_view(*taken, placeholder);
    register_deferred_image_normalize_candidate(
        placeholder, std::move(*taken));
    utils::log_vulkan_op_hit("aten::image_normalize_bridge.defer_std");
    return placeholder;
  }

  return std::nullopt;
}

const api::ShaderInfo& integral_buffer_scalar_shader(
    const api::ScalarType dtype,
    const BinaryOpKind op_kind) {
  switch (dtype) {
    case api::kInt:
      switch (op_kind) {
        case BinaryOpKind::Add:
          return VK_KERNEL(buffer_int_add_scalar);
        case BinaryOpKind::Sub:
          return VK_KERNEL(buffer_int_sub_scalar);
        case BinaryOpKind::Mul:
          return VK_KERNEL(buffer_int_mul_scalar);
        default:
          break;
      }
      break;
    case api::kChar:
      switch (op_kind) {
        case BinaryOpKind::Add:
          return VK_KERNEL(buffer_char_add_scalar);
        case BinaryOpKind::Sub:
          return VK_KERNEL(buffer_char_sub_scalar);
        case BinaryOpKind::Mul:
          return VK_KERNEL(buffer_char_mul_scalar);
        default:
          break;
      }
      break;
    case api::kByte:
      switch (op_kind) {
        case BinaryOpKind::Add:
          return VK_KERNEL(buffer_byte_add_scalar);
        case BinaryOpKind::Sub:
          return VK_KERNEL(buffer_byte_sub_scalar);
        case BinaryOpKind::Mul:
          return VK_KERNEL(buffer_byte_mul_scalar);
        default:
          break;
      }
      break;
    default:
      break;
  }
  VK_THROW("Unsupported integral buffer scalar binary op");
}

const api::ShaderInfo& integral_buffer_tensor_shader(
    const api::ScalarType dtype,
    const BinaryOpKind op_kind) {
  switch (dtype) {
    case api::kInt:
      switch (op_kind) {
        case BinaryOpKind::Add:
          return VK_KERNEL(buffer_int_add);
        case BinaryOpKind::Sub:
          return VK_KERNEL(buffer_int_sub);
        case BinaryOpKind::Mul:
          return VK_KERNEL(buffer_int_mul);
        default:
          break;
      }
      break;
    case api::kChar:
      switch (op_kind) {
        case BinaryOpKind::Add:
          return VK_KERNEL(buffer_char_add);
        case BinaryOpKind::Sub:
          return VK_KERNEL(buffer_char_sub);
        case BinaryOpKind::Mul:
          return VK_KERNEL(buffer_char_mul);
        default:
          break;
      }
      break;
    case api::kByte:
      switch (op_kind) {
        case BinaryOpKind::Add:
          return VK_KERNEL(buffer_byte_add);
        case BinaryOpKind::Sub:
          return VK_KERNEL(buffer_byte_sub);
        case BinaryOpKind::Mul:
          return VK_KERNEL(buffer_byte_mul);
        default:
          break;
      }
      break;
    default:
      break;
  }
  VK_THROW("Unsupported integral buffer tensor binary op");
}

const api::ShaderInfo& bool_buffer_scalar_shader(const BinaryOpKind op_kind) {
  switch (op_kind) {
    case BinaryOpKind::Add:
      return VK_KERNEL(buffer_bool_add_scalar);
    case BinaryOpKind::Mul:
      return VK_KERNEL(buffer_bool_mul_scalar);
    default:
      VK_THROW("Unsupported bool buffer scalar binary op");
  }
}

const api::ShaderInfo& bool_buffer_tensor_shader(const BinaryOpKind op_kind) {
  switch (op_kind) {
    case BinaryOpKind::Add:
      return VK_KERNEL(buffer_bool_add);
    case BinaryOpKind::Mul:
      return VK_KERNEL(buffer_bool_mul);
    default:
      VK_THROW("Unsupported bool buffer tensor binary op");
  }
}

bool needs_binary_cpu_fallback(const Tensor& tensor) {
  return tensor.is_vulkan() && convert(tensor).dtype() != api::kFloat;
}

bool can_promote_binary_operand_to_float(const ScalarType dtype) {
  switch (dtype) {
    case kBool:
    case kByte:
    case kChar:
    case kShort:
    case kInt:
    case kLong:
    case kHalf:
    case kBFloat16:
    case kFloat:
      return true;
    default:
      return false;
  }
}

bool should_promote_binary_operands_to_float(
    const Tensor& self,
    const Tensor& other) {
  const ScalarType promoted_dtype =
      promote_for_vulkan_binary(self.scalar_type(), other.scalar_type());
  return promoted_dtype == kFloat &&
      can_promote_binary_operand_to_float(self.scalar_type()) &&
      can_promote_binary_operand_to_float(other.scalar_type()) &&
      (self.scalar_type() != kFloat || other.scalar_type() != kFloat);
}

Tensor promote_binary_operand_to_vulkan_float(const Tensor& operand_arg) {
  Tensor operand = operand_arg;
  if (operand.scalar_type() != kFloat) {
    operand = operand.is_vulkan() ? utils::cast_vulkan_tensor_dtype(operand, kFloat)
                                  : operand.to(kFloat);
  }
  return operand.is_vulkan() ? operand : operand.vulkan();
}

bool scalar_is_integral_exponent(const Scalar& other) {
  if (other.isIntegral(false)) {
    return true;
  }
  if (!other.isFloatingPoint()) {
    return false;
  }
  const double value = other.to<double>();
  if (!std::isfinite(value)) {
    return false;
  }
  return std::nearbyint(value) == value;
}

int64_t scalar_to_integral_exponent(const Scalar& other) {
  if (other.isIntegral(false)) {
    return other.to<int64_t>();
  }
  return safe_downcast<int64_t>(std::nearbyint(other.to<double>()));
}

Tensor pow_tensor_scalar_integral_exponent(
    const Tensor& self_arg,
    const int64_t exponent) {
  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  Tensor result = at::ones_like(self);
  Tensor factor = self;

  const bool negative_exponent = exponent < 0;
  uint64_t power = negative_exponent
      ? static_cast<uint64_t>(-(exponent + 1)) + 1
      : static_cast<uint64_t>(exponent);

  while (power > 0) {
    if ((power & 1u) != 0u) {
      result = at::mul(result, factor);
    }
    power >>= 1u;
    if (power > 0) {
      factor = at::mul(factor, factor);
    }
  }

  if (negative_exponent) {
    Tensor numerator = at::ones_like(result);
    result = at::div(numerator, result);
  }

  return result;
}

bool should_run_buffer_binary_scalar(const Tensor& tensor) {
  if (!tensor.is_vulkan() || tensor.scalar_type() != c10::ScalarType::Float) {
    return false;
  }

  const vTensor& v_tensor = convert(tensor);
  return v_tensor.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_elementwise_compute(v_tensor);
}

bool is_integral_buffer_compute_candidate(const Tensor& tensor) {
  return utils::supports_native_integral_buffer_compute(tensor);
}

bool should_run_buffer_binary_tensor(const Tensor& self, const Tensor& other) {
  const ScalarType promoted_dtype =
      promote_for_vulkan_binary(self.scalar_type(), other.scalar_type());
  if (
      !self.is_vulkan() || !other.is_vulkan() ||
      promoted_dtype != c10::ScalarType::Float ||
      self.scalar_type() != promoted_dtype ||
      other.scalar_type() != promoted_dtype) {
    return false;
  }

  const vTensor& v_self = convert(self);
  const vTensor& v_other = convert(other);
  const bool self_buffer = v_self.storage_type() == api::StorageType::BUFFER;
  const bool other_buffer = v_other.storage_type() == api::StorageType::BUFFER;
  if (!self_buffer && !other_buffer) {
    return false;
  }
  if (
      !utils::supports_buffer_elementwise_compute(v_self) ||
      !utils::supports_buffer_elementwise_compute(v_other)) {
    return false;
  }
  return self_buffer || other_buffer;
}

bool should_run_add_scaled_buffer_out(
    const Tensor& self,
    const Tensor& other,
    const Tensor& scale,
    const Tensor& output) {
  if (
      !self.is_vulkan() || !other.is_vulkan() || !scale.is_vulkan() ||
      !output.defined() || !output.is_vulkan() ||
      self.scalar_type() != kFloat || other.scalar_type() != kFloat ||
      scale.scalar_type() != kFloat || output.scalar_type() != kFloat ||
      !self.sizes().equals(other.sizes()) ||
      !self.sizes().equals(output.sizes()) ||
      self.dim() < 1 || self.dim() > 4 || scale.dim() != 1 ||
      scale.size(0) != self.size(-1)) {
    return false;
  }

  const vTensor& v_self = convert(self);
  const vTensor& v_other = convert(other);
  const vTensor& v_scale = convert(scale);
  const vTensor& v_output = convert(output);
  return v_self.storage_type() == api::StorageType::BUFFER &&
      v_other.storage_type() == api::StorageType::BUFFER &&
      v_scale.storage_type() == api::StorageType::BUFFER &&
      v_output.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_elementwise_compute(v_self) &&
      utils::supports_buffer_elementwise_compute(v_other) &&
      utils::supports_buffer_elementwise_compute(v_scale) &&
      utils::supports_buffer_elementwise_compute(v_output);
}

bool should_run_add_relu_buffer_out(
    const Tensor& self,
    const Tensor& other,
    const Tensor& add_output,
    const Tensor& relu_output) {
  if (
      !self.is_vulkan() || !other.is_vulkan() || !add_output.defined() ||
      !relu_output.defined() || !add_output.is_vulkan() ||
      !relu_output.is_vulkan() || self.scalar_type() != kFloat ||
      other.scalar_type() != kFloat || add_output.scalar_type() != kFloat ||
      relu_output.scalar_type() != kFloat ||
      !self.sizes().equals(other.sizes()) ||
      !self.sizes().equals(add_output.sizes()) ||
      !self.sizes().equals(relu_output.sizes()) || self.dim() < 1 ||
      self.dim() > 4) {
    return false;
  }

  const vTensor& v_self = convert(self);
  const vTensor& v_other = convert(other);
  const vTensor& v_add_output = convert(add_output);
  const vTensor& v_relu_output = convert(relu_output);
  return v_self.storage_type() == api::StorageType::BUFFER &&
      v_other.storage_type() == api::StorageType::BUFFER &&
      v_add_output.storage_type() == api::StorageType::BUFFER &&
      v_relu_output.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_elementwise_compute(v_self) &&
      utils::supports_buffer_elementwise_compute(v_other) &&
      utils::supports_buffer_elementwise_compute(v_add_output) &&
      utils::supports_buffer_elementwise_compute(v_relu_output);
}

bool should_run_buffer_binary_tensor_integral(
    const Tensor& self,
    const Tensor& other,
    const std::optional<Scalar>& alpha_arg,
    const BinaryOpKind op_kind) {
  if (self.scalar_type() != other.scalar_type()) {
    return false;
  }
  if (self.is_vulkan() && !is_integral_buffer_compute_candidate(self)) {
    return false;
  }
  if (other.is_vulkan() && !is_integral_buffer_compute_candidate(other)) {
    return false;
  }
  if (self.sizes().vec() != other.sizes().vec()) {
    return false;
  }
  if (
      (self.scalar_type() == kChar || self.scalar_type() == kByte) &&
      !utils::last_dim_is_width_aligned(self)) {
    return false;
  }
  if (
      op_kind != BinaryOpKind::Add && op_kind != BinaryOpKind::Sub &&
      op_kind != BinaryOpKind::Mul) {
    return false;
  }
  if (!alpha_arg.has_value()) {
    return true;
  }
  return utils::scalar_fits_vulkan_int32(*alpha_arg);
}

bool should_run_buffer_binary_scalar_integral(
    const Tensor& self,
    const Scalar& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const BinaryOpKind op_kind) {
  if (!is_integral_buffer_compute_candidate(self)) {
    return false;
  }
  if (
      (self.scalar_type() == kChar || self.scalar_type() == kByte) &&
      !utils::last_dim_is_width_aligned(self)) {
    return false;
  }
  if (
      op_kind != BinaryOpKind::Add && op_kind != BinaryOpKind::Sub &&
      op_kind != BinaryOpKind::Mul) {
    return false;
  }
  if (!utils::scalar_fits_vulkan_int32(other_arg)) {
    return false;
  }

  if (!alpha_arg.has_value()) {
    return true;
  }
  return utils::scalar_fits_vulkan_int32(*alpha_arg);
}

bool is_bool_buffer_compute_candidate(const Tensor& tensor) {
  return utils::supports_native_bool_buffer_compute(tensor);
}

bool should_run_buffer_binary_tensor_bool(
    const Tensor& self,
    const Tensor& other,
    const std::optional<Scalar>& alpha_arg,
    const BinaryOpKind op_kind) {
  if (self.scalar_type() != c10::kBool || other.scalar_type() != c10::kBool) {
    return false;
  }
  if (self.is_vulkan() && !is_bool_buffer_compute_candidate(self)) {
    return false;
  }
  if (other.is_vulkan() && !is_bool_buffer_compute_candidate(other)) {
    return false;
  }
  if (self.sizes().vec() != other.sizes().vec()) {
    return false;
  }
  if (!utils::last_dim_is_width_aligned(self)) {
    return false;
  }
  if (op_kind != BinaryOpKind::Add && op_kind != BinaryOpKind::Mul) {
    return false;
  }
  return !alpha_arg.has_value() ||
      utils::scalar_fits_vulkan_int32(*alpha_arg);
}

bool should_run_buffer_binary_scalar_bool(
    const Tensor& self,
    const Scalar& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const BinaryOpKind op_kind) {
  if (self.scalar_type() != c10::kBool || !other_arg.isBoolean()) {
    return false;
  }
  if (!is_bool_buffer_compute_candidate(self)) {
    return false;
  }
  if (!utils::last_dim_is_width_aligned(self)) {
    return false;
  }
  if (op_kind != BinaryOpKind::Add && op_kind != BinaryOpKind::Mul) {
    return false;
  }
  return !alpha_arg.has_value() ||
      utils::scalar_fits_vulkan_int32(*alpha_arg);
}

bool should_run_bool_or_tensor_native(const Tensor& self, const Tensor& other) {
  return self.is_vulkan() && other.is_vulkan() && self.scalar_type() == kBool &&
      other.scalar_type() == kBool && self.dim() == 1 &&
      self.numel() == 1 && self.sizes().equals(other.sizes()) &&
      self.is_contiguous() &&
      other.is_contiguous() && self.storage_offset() == 0 &&
      other.storage_offset() == 0;
}

bool should_write_bool_or_tensor_native_out(
    const Tensor& out,
    const Tensor& self,
    const Tensor& other) {
  return should_run_bool_or_tensor_native(self, other) && out.is_vulkan() &&
      out.scalar_type() == kBool && out.dim() == 1 &&
      out.sizes().equals(self.sizes()) && out.is_contiguous() &&
      out.storage_offset() == 0 && convert(out).dtype() == api::kBool &&
      convert(out).storage_type() == api::StorageType::BUFFER;
}

Tensor bool_or_tensor_cpu_fallback(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const char* op_name,
    const bool logical) {
  report_vulkan_cpu_fallback(
      op_name, "bool_or_cpu_fallback", {self_arg, other_arg});
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
    const Tensor other_cpu = other_arg.is_vulkan() ? other_arg.cpu() : other_arg;
    cpu_result = logical ? at::logical_or(self_cpu, other_cpu)
                         : at::bitwise_or(self_cpu, other_cpu);
  }
  return record_tensor_write_and_return(
      cpu_result.vulkan(),
      op_name,
      "bool_or_cpu_fallback",
      {self_arg, other_arg});
}

Tensor bool_and_tensor_cpu_fallback(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const char* op_name,
    const bool logical) {
  report_vulkan_cpu_fallback(
      op_name, "bool_and_cpu_fallback", {self_arg, other_arg});
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
    const Tensor other_cpu = other_arg.is_vulkan() ? other_arg.cpu() : other_arg;
    cpu_result = logical ? at::logical_and(self_cpu, other_cpu)
                         : at::bitwise_and(self_cpu, other_cpu);
  }
  return record_tensor_write_and_return(
      cpu_result.vulkan(),
      op_name,
      "bool_and_cpu_fallback",
      {self_arg, other_arg});
}

Tensor bool_not_tensor_cpu_fallback(
    const Tensor& self_arg,
    const char* op_name,
    const bool logical) {
  report_vulkan_cpu_fallback(op_name, "bool_not_cpu_fallback", {self_arg});
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
    cpu_result = logical ? at::logical_not(self_cpu) : at::bitwise_not(self_cpu);
  }
  return record_tensor_write_and_return(
      cpu_result.vulkan(), op_name, "bool_not_cpu_fallback", {self_arg});
}

bool should_run_small_control_tensor_fallback(
    const Tensor& self_arg,
    const Tensor& other_arg) {
  return self_arg.numel() <= 16 && other_arg.numel() <= 16;
}

Tensor maximum_tensor_small_control_cpu_fallback(
    const Tensor& self_arg,
    const Tensor& other_arg) {
  TORCH_CHECK(
      should_run_small_control_tensor_fallback(self_arg, other_arg),
      "Vulkan aten::maximum currently supports only small control tensors; "
      "got self numel=",
      self_arg.numel(),
      " and other numel=",
      other_arg.numel());
  report_vulkan_cpu_fallback(
      "aten::maximum", "small_control_cpu_fallback", {self_arg, other_arg});
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
    const Tensor other_cpu = other_arg.is_vulkan() ? other_arg.cpu() : other_arg;
    cpu_result = at::maximum(self_cpu, other_cpu);
  }
  return record_tensor_write_and_return(
      cpu_result.vulkan(),
      "aten::maximum",
      "small_control_cpu_fallback",
      {self_arg, other_arg});
}

Tensor binary_op_scalar_cpu_fallback(
    const Tensor& self_arg,
    const Scalar& other,
    const std::optional<Scalar>& alpha_arg,
    const BinaryOpKind op_kind) {
  report_vulkan_cpu_fallback(
      "aten::binary_op", "scalar_cpu_fallback", {self_arg});
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
    switch (op_kind) {
      case BinaryOpKind::Add:
        cpu_result = at::add(self_cpu, other, alpha_arg.value_or(Scalar(1)));
        break;
      case BinaryOpKind::Sub:
        cpu_result = at::sub(self_cpu, other, alpha_arg.value_or(Scalar(1)));
        break;
      case BinaryOpKind::Mul:
        cpu_result = at::mul(self_cpu, other);
        break;
      case BinaryOpKind::Div:
        cpu_result = at::div(self_cpu, other);
        break;
      case BinaryOpKind::FloorDivide:
        cpu_result = at::floor_divide(self_cpu, other);
        break;
      case BinaryOpKind::Pow:
        cpu_result = at::pow(self_cpu, other);
        break;
    }
  }
  return record_tensor_write_and_return(
      cpu_result.vulkan(),
      "aten::binary_op",
      "scalar_cpu_fallback",
      {self_arg});
}

Tensor binary_op_tensor_cpu_fallback(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const BinaryOpKind op_kind) {
  report_vulkan_cpu_fallback(
      "aten::binary_op", "tensor_cpu_fallback", {self_arg, other_arg});
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
    const Tensor other_cpu = other_arg.is_vulkan() ? other_arg.cpu() : other_arg;
    switch (op_kind) {
      case BinaryOpKind::Add:
        cpu_result =
            at::add(self_cpu, other_cpu, alpha_arg.value_or(Scalar(1)));
        break;
      case BinaryOpKind::Sub:
        cpu_result =
            at::sub(self_cpu, other_cpu, alpha_arg.value_or(Scalar(1)));
        break;
      case BinaryOpKind::Mul:
        cpu_result = at::mul(self_cpu, other_cpu);
        break;
      case BinaryOpKind::Div:
        cpu_result = at::div(self_cpu, other_cpu);
        break;
      case BinaryOpKind::FloorDivide:
        cpu_result = at::floor_divide(self_cpu, other_cpu);
        break;
      case BinaryOpKind::Pow:
        cpu_result = at::pow(self_cpu, other_cpu);
        break;
    }
  }
  return record_tensor_write_and_return(
      cpu_result.vulkan(),
      "aten::binary_op",
      "tensor_cpu_fallback",
      {self_arg, other_arg});
}

Tensor prepare_native_integral_buffer_input(const Tensor& input_arg) {
  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  vTensor v_input = convert(input);
  if (
      v_input.dtype() != api::kByte && v_input.dtype() != api::kChar &&
      v_input.dtype() != api::kBool) {
    return utils::ensure_buffer_storage(input);
  }

  if (
      v_input.storage_type() == api::StorageType::BUFFER &&
      utils::last_dim_is_width_aligned(input) &&
      (v_input.dtype() == api::kBool
           ? utils::supports_native_bool_buffer_compute(input)
           : utils::supports_native_integral_buffer_compute(input))) {
    return input;
  }

  input = utils::ensure_buffer_storage(input);
  v_input = convert(input);
  if (
      !v_input.has_direct_buffer_layout() ||
      v_input.storage_type() != api::StorageType::BUFFER ||
      v_input.gpu_memory_layout() !=
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED) {
    return input;
  }

  api::AllocationScope allocation_scope("binary_op.narrow_buffer_materialize");
  api::Context* const context = api::context();
  api::StorageBuffer staging(context, v_input.dtype(), v_input.numel());
  vTensor v_src = v_input;
  utils::pack_vtensor_to_staging(v_src, staging.buffer());

  vTensor v_out{
      context,
      v_input.sizes(),
      v_input.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };
  api::PipelineBarrier pipeline_barrier{};
  add_buffer_barrier(
      pipeline_barrier,
      staging.buffer(),
      api::PipelineStage::COMPUTE | api::PipelineStage::TRANSFER,
      api::MemoryAccessType::WRITE,
      api::PipelineStage::COMPUTE | api::PipelineStage::TRANSFER,
      api::MemoryAccessType::READ);
  utils::pack_buffer_to_vtensor(staging.buffer(), v_out, pipeline_barrier);
  return convert(v_out);
}

} // namespace

void note_runtime_elementwise_binary_live_chain(
    const Tensor& self,
    const Tensor& other,
    const Tensor& output,
    const char* op_name) {
  if (op_name == nullptr) {
    return;
  }
  RuntimeElementwiseMixedStep step;
  step.operand_kind = RuntimeElementwiseMixedOperandKind::Tensor;
  step.op = op_name;
  step.rhs = other;
  record_runtime_elementwise_mixed_chain_step(self, output, std::move(step));
}

void note_runtime_elementwise_unary_live_chain(
    const Tensor& self,
    const Tensor& output,
    const char* op_name) {
  if (op_name == nullptr || !runtime_elementwise_mixed_unary_supported(op_name)) {
    return;
  }
  RuntimeElementwiseMixedStep step;
  step.operand_kind = RuntimeElementwiseMixedOperandKind::Unary;
  step.op = op_name;
  record_runtime_elementwise_mixed_chain_step(self, output, std::move(step));
}

Tensor run_runtime_elementwise_chain_add_mul_sub_div(
    const Tensor& input,
    const Tensor& add_rhs,
    const Tensor& mul_rhs,
    const Tensor& sub_rhs,
    const Tensor& div_rhs) {
  return run_runtime_elementwise_chain(
      input,
      std::vector<Tensor>{add_rhs, mul_rhs, sub_rhs, div_rhs},
      std::vector<std::string>{"add", "mul", "sub", "div"});
}

Tensor run_runtime_elementwise_chain(
    const Tensor& input,
    const std::vector<Tensor>& rhs_tensors,
    const std::vector<std::string>& ops) {
  TORCH_CHECK(
      !ops.empty() && ops.size() == rhs_tensors.size(),
      "runtime_elementwise_chain expects one RHS tensor per op");
  TORCH_CHECK(
      ops.size() <= 4,
      "runtime_elementwise_chain currently supports 1 to 4 tensor RHS ops");
  check_runtime_elementwise_chain_tensor(input, input, "input");
  for (const auto idx : c10::irange(rhs_tensors.size())) {
    runtime_elementwise_chain_op_symbol(ops[idx]);
    check_runtime_elementwise_chain_tensor(
        rhs_tensors[idx],
        input,
        "rhs");
  }

  api::AllocationScope allocation_scope(
      "runtime_elementwise_chain.tensor_rhs");
  api::Context* const context = api::context();
  vTensor& v_input = convert(input);
  TORCH_CHECK(
      v_input.numel() > 0,
      "runtime_elementwise_chain expects a non-empty tensor");

  std::vector<vTensor*> v_inputs;
  v_inputs.reserve(rhs_tensors.size() + 1u);
  v_inputs.push_back(&v_input);
  for (const Tensor& rhs : rhs_tensors) {
    v_inputs.push_back(&convert(rhs));
  }

  vTensor v_output{
      context,
      input.sizes().vec(),
      v_input.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  std::vector<VkDescriptorType> layout{
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
  };
  layout.resize(2u + v_inputs.size() * 2u);
  for (size_t idx = 0u; idx < v_inputs.size(); ++idx) {
    layout[2u + idx * 2u] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layout[3u + idx * 2u] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  }
  const std::vector<uint32_t>& spirv = runtime_elementwise_chain_spirv(ops);
  api::ShaderInfo shader_descriptor{
      "runtime_elementwise_chain." + runtime_elementwise_chain_key(ops),
      std::vector<uint32_t>(spirv.begin(), spirv.end()),
      layout,
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
  std::vector<api::UniformParamsBuffer> input_metas;
  input_metas.reserve(v_inputs.size());
  for (const vTensor* v_tensor : v_inputs) {
    input_metas.emplace_back(
        utils::make_buffer_compute_metadata_ubo(context, *v_tensor));
  }

  utils::log_vulkan_op_hit(
      "vulkan_prepack::runtime_elementwise_chain");
  if (ops.size() == 1u) {
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
        v_inputs[0]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        input_metas[0].buffer(),
        v_inputs[1]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        input_metas[1].buffer());
  } else if (ops.size() == 2u) {
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
        v_inputs[0]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        input_metas[0].buffer(),
        v_inputs[1]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        input_metas[1].buffer(),
        v_inputs[2]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        input_metas[2].buffer());
  } else if (ops.size() == 3u) {
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
        v_inputs[0]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        input_metas[0].buffer(),
        v_inputs[1]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        input_metas[1].buffer(),
        v_inputs[2]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        input_metas[2].buffer(),
        v_inputs[3]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        input_metas[3].buffer());
  } else {
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
        v_inputs[0]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        input_metas[0].buffer(),
        v_inputs[1]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        input_metas[1].buffer(),
        v_inputs[2]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        input_metas[2].buffer(),
        v_inputs[3]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        input_metas[3].buffer(),
        v_inputs[4]->buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        input_metas[4].buffer());
  }

  std::vector<Tensor> provenance_inputs;
  provenance_inputs.reserve(rhs_tensors.size() + 1u);
  provenance_inputs.push_back(input);
  provenance_inputs.insert(
      provenance_inputs.end(),
      rhs_tensors.begin(),
      rhs_tensors.end());

  return record_tensor_write_and_return(
      convert(v_output),
      "vulkan_prepack::runtime_elementwise_chain",
      "runtime_generated_elementwise_chain",
      provenance_inputs);
}

static Tensor binary_op_tensor_buffer_integral(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::AllocationScope allocation_scope("binary_op.integral_buffer");
  utils::is_broadcastable(self_arg, other_arg);
  api::Context* const context = api::context();

  Tensor self = prepare_native_integral_buffer_input(self_arg);
  Tensor other = prepare_native_integral_buffer_input(other_arg);
  vTensor& v_self = convert(self);
  vTensor& v_other = convert(other);

  vTensor v_output{
      context,
      utils::broadcast_size(self_arg, other_arg),
      v_self.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct Block final {
    int32_t alpha;
  } block{
      alpha_arg ? alpha_arg->to<int32_t>() : 1,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(v_output.numel()),
      1u,
      1u,
  };
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_self);
  api::UniformParamsBuffer other_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_other);

  context->submit_compute_job(
      shader_descriptor,
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer(),
      v_other.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      other_meta.buffer(),
      params.buffer());

  return record_tensor_write_and_return(
      convert(v_output), "aten::binary_op", "tensor_buffer_direct", {self, other});
}

static Tensor binary_op_tensor_buffer_bool(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::AllocationScope allocation_scope("binary_op.bool_buffer");
  utils::is_broadcastable(self_arg, other_arg);
  api::Context* const context = api::context();

  Tensor self = prepare_native_integral_buffer_input(self_arg);
  Tensor other = prepare_native_integral_buffer_input(other_arg);
  vTensor& v_self = convert(self);
  vTensor& v_other = convert(other);

  vTensor v_output{
      context,
      utils::broadcast_size(self_arg, other_arg),
      api::kBool,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct Block final {
    int32_t alpha;
  } block{
      alpha_arg ? utils::scalar_to_vulkan_int32(*alpha_arg) : 1,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(v_output.numel()),
      1u,
      1u,
  };
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_self);
  api::UniformParamsBuffer other_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_other);

  context->submit_compute_job(
      shader_descriptor,
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer(),
      v_other.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      other_meta.buffer(),
      params.buffer());

  return record_tensor_write_and_return(
      convert(v_output), "aten::binary_op", "scalar_buffer", {self});
}

static Tensor binary_op_scalar_buffer(
    const Tensor& self_arg,
    const Scalar& other,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor,
    const BinaryOpKind op_kind) {
  api::AllocationScope allocation_scope("binary_op.buffer");
  utils::log_vulkan_op_hit("aten::binary_op.scalar_buffer_float");
  utils::validate_replay_tensor_not_stale(
      self_arg, "aten::binary_op.scalar_buffer");
  api::Context* const context = api::context();

  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  self = utils::prepare_vulkan_execution_tensor(
      self, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const float other_val = alpha_arg ? other.to<float>() * alpha_arg->to<float>()
                                    : other.to<float>();
  const struct Block final {
    float other;
  } block{
      other_val,
  };

  api::UniformParamsBuffer params(context, block);
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
  log_binary_submit(
      op_kind, "scalar_buffer", v_self, nullptr, v_output, global_size, local_size);

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
      in_meta.buffer(),
      params.buffer());

  Tensor returned = record_tensor_write_and_return(
      convert(v_output), "aten::binary_op", "scalar_buffer_integral", {self});
  maybe_probe_runtime_elementwise_live_chain_scalar(
      self, returned, other_val, alpha_arg, op_kind);
  return returned;
}

static Tensor binary_op_scalar_buffer_integral(
    const Tensor& self_arg,
    const Scalar& other,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor) {
  api::AllocationScope allocation_scope("binary_op.integral_scalar_buffer");
  api::Context* const context = api::context();

  Tensor self = prepare_native_integral_buffer_input(self_arg);
  vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const int32_t other_val =
      alpha_arg ? safe_downcast<int32_t>(
                      static_cast<int64_t>(
                          utils::scalar_to_vulkan_int32(other)) *
                      utils::scalar_to_vulkan_int32(*alpha_arg))
                : utils::scalar_to_vulkan_int32(other);

  const struct Block final {
    int32_t other;
  } block{
      other_val,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(v_output.numel()),
      1u,
      1u,
  };
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_self);

  context->submit_compute_job(
      shader_descriptor,
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer(),
      params.buffer());

  return record_tensor_write_and_return(
      convert(v_output), "aten::binary_op", "scalar_buffer_bool", {self});
}

static Tensor binary_op_scalar_buffer_bool(
    const Tensor& self_arg,
    const Scalar& other,
    const api::ShaderInfo& shader_descriptor) {
  api::AllocationScope allocation_scope("binary_op.bool_scalar_buffer");
  api::Context* const context = api::context();

  Tensor self = prepare_native_integral_buffer_input(self_arg);
  vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      api::kBool,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct Block final {
    int32_t other;
  } block{
      other.to<bool>() ? 1 : 0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(v_output.numel()),
      1u,
      1u,
  };
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_self);

  context->submit_compute_job(
      shader_descriptor,
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer(),
      params.buffer());

  return record_tensor_write_and_return(
      convert(v_output), "aten::binary_op", "scalar_texture", {self});
}

static Tensor binary_op_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor,
    const api::ShaderInfo& buffer_shader_descriptor,
  const BinaryOpKind op_kind) {
  api::AllocationScope allocation_scope("binary_op");
  utils::validate_replay_tensor_not_stale(
      self_arg, "aten::binary_op.scalar");
  api::Context* const context = api::context();
  if (!alpha_arg.has_value()) {
    if (auto deferred = try_start_deferred_image_normalize_scalar(
            self_arg, other, op_kind)) {
      return *deferred;
    }
    if (op_kind == BinaryOpKind::Mul) {
      if (auto deferred =
              try_start_deferred_attention_query_scale(self_arg, other)) {
        return *deferred;
      }
    }
  }
  const Tensor self_materialized =
      materialize_decomposed_attention_candidate_if_needed(self_arg);
  const Tensor self_attention_query_scaled =
      materialize_deferred_attention_query_scale_candidate_if_needed(
          self_materialized);
  const Tensor self_image_normalized =
      materialize_deferred_image_normalize_candidate_if_needed(
          self_attention_query_scaled);
  const Tensor self_input =
      materialize_deferred_layer_scale_candidate_if_needed(
          materialize_deferred_add_layer_norm_candidate_if_needed(
              materialize_deferred_linear_gelu_candidate_if_needed(
                  self_image_normalized)));

  if (self_input.dim() > 4) {
    return binary_op_scalar_cpu_fallback(self_input, other, alpha_arg, op_kind);
  }

  if (should_run_buffer_binary_scalar_integral(
          self_input, other, alpha_arg, op_kind)) {
    return binary_op_scalar_buffer_integral(
        self_input,
        other,
        alpha_arg,
        integral_buffer_scalar_shader(convert(self_input).dtype(), op_kind));
  }

  if (should_run_buffer_binary_scalar_bool(
          self_input, other, alpha_arg, op_kind)) {
    return binary_op_scalar_buffer_bool(
        self_input, other, bool_buffer_scalar_shader(op_kind));
  }

  if (needs_binary_cpu_fallback(self_input)) {
    return binary_op_scalar_cpu_fallback(self_input, other, alpha_arg, op_kind);
  }

  Tensor self = self_input.is_vulkan() ? self_input : self_input.vulkan();
  if (should_run_buffer_binary_scalar(self)) {
    return binary_op_scalar_buffer(
        self, other, alpha_arg, buffer_shader_descriptor, op_kind);
  }
  self = utils::prepare_vulkan_execution_tensor(
      self, utils::VulkanExecutionPlanKind::TextureComputeInput);
  const vTensor& v_self = convert(self);

  vTensor v_output{
      context,
      v_self.sizes(),
      v_self.dtype(),
  };

  const float other_val = alpha_arg ? other.to<float>() * alpha_arg->to<float>()
                                    : other.to<float>();
  const struct Block final {
    uvec3 extents;
    int fill0;
    float other;
  } block{
      v_self.extents(),
      0,
      other_val,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = v_output.extents();
  const uvec3 local_size = adaptive_work_group_size(global_size);
  log_binary_submit(
      op_kind, "scalar_texture", v_self, nullptr, v_output, global_size, local_size);

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
      "aten::binary_op",
      "scalar_texture",
      {self});
}

static Tensor binary_op_preprocess_other_arg(const Tensor& other_arg) {
  // Similar to binary_op_scalar where tensors is mapped to float, we
  // also map known integer types (but not quant types) tensor to float.

  // Such conversion can only to be done before moving to vulkan, since vulkan
  // doesn't yet support integer types.
  Tensor other = other_arg;
  if (!other.is_vulkan()) {
    switch (other.scalar_type()) {
      case at::kByte:
      case at::kChar:
      case at::kShort:
      case at::kInt:
      case at::kLong:
      case at::kHalf:
      case at::kBFloat16:
      case at::kDouble:
        other = other.to(kFloat);
        break;
      case at::kFloat:
        // No op for expected type.
        break;
      default:
        TORCH_CHECK(
            false,
            "binary_op_tensor, doesn't support type %s",
            other.scalar_type());
        break;
    }
    other = other.vulkan();
  }

  return other;
}

static Tensor& binary_op_scalar_(
    Tensor& self_arg,
    const Scalar& other,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& inplace_shader_descriptor,
    const api::ShaderInfo& out_shader_descriptor,
    const api::ShaderInfo& buffer_shader_descriptor,
    const BinaryOpKind op_kind) {
  api::AllocationScope allocation_scope("binary_op_inplace");
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);
  if (v_self.storage_type() == api::StorageType::BUFFER) {
    Tensor result = binary_op_scalar(
        self_arg,
        other,
        alpha_arg,
        out_shader_descriptor,
        buffer_shader_descriptor,
        op_kind);
    ops::copy_(self_arg, result);
    return self_arg;
  }

  const float other_val = alpha_arg ? other.to<float>() * alpha_arg->to<float>()
                                    : other.to<float>();
  const struct Block final {
    uvec3 extents;
    int fill0;
    float other;
  } block{
      v_self.extents(),
      0,
      other_val,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      inplace_shader_descriptor,
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

static Tensor binary_op_tensor_buffer_impl(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor,
    const BinaryOpKind op_kind,
    Tensor* output_arg) {
  api::AllocationScope allocation_scope("binary_op.buffer");
  utils::log_vulkan_op_hit("aten::binary_op.buffer_float");
  utils::validate_replay_tensor_not_stale(
      self_arg, "aten::binary_op.tensor_buffer");
  utils::validate_replay_tensor_not_stale(
      other_arg, "aten::binary_op.tensor_buffer");
  const Tensor self_input =
      materialize_deferred_layer_scale_candidate_if_needed(self_arg);
  const Tensor other_input =
      materialize_deferred_layer_scale_candidate_if_needed(other_arg);
  utils::is_broadcastable(self_input, other_input);
  api::Context* const context = api::context();

  Tensor self = self_input.is_vulkan() ? self_input : self_input.vulkan();
  Tensor other = other_input.is_vulkan() ? other_input : other_input.vulkan();
  self = utils::prepare_vulkan_execution_tensor(
      self, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  other = utils::prepare_vulkan_execution_tensor(
      other, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);

  vTensor& v_self = convert(self);
  vTensor& v_other = convert(other);

  const std::vector<int64_t> output_sizes =
      utils::broadcast_size(self_input, other_input);
  Tensor output_tensor;
  vTensor* v_output_ptr = nullptr;
  vTensor owned_output;
  if (output_arg != nullptr) {
    TORCH_CHECK(
        output_arg->defined(),
        "Vulkan buffer binary out expects a defined output tensor");
    output_tensor = output_arg->is_vulkan() ? *output_arg : output_arg->vulkan();
    output_tensor = utils::mark_tensor_execution(
        output_tensor,
        utils::resolve_buffer_execution_layout(convert(output_tensor)),
        false);
    vTensor& v_output = convert(output_tensor);
    TORCH_CHECK(
        v_output.storage_type() == api::StorageType::BUFFER &&
            v_output.dtype() == api::kFloat &&
            utils::supports_buffer_elementwise_compute(v_output),
        "Vulkan buffer binary out expects float buffer-backed output");
    TORCH_CHECK(
        output_tensor.sizes().vec() == output_sizes,
        "Vulkan buffer binary out received mismatched output shape");
    v_output_ptr = &v_output;
  } else {
    owned_output = vTensor{
        context,
        output_sizes,
        v_self.dtype(),
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
    };
    v_output_ptr = &owned_output;
  }
  vTensor& v_output = *v_output_ptr;

  const struct Block final {
    float alpha;
  } block{
      alpha_arg ? alpha_arg->to<float>() : 1.0f,
  };

  api::UniformParamsBuffer params(context, block);
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
  api::UniformParamsBuffer other_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_other);
  log_binary_submit(
      op_kind, "tensor_buffer", v_self, &v_other, v_output, global_size, local_size);

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
      in_meta.buffer(),
      v_other.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      other_meta.buffer(),
      params.buffer());

  Tensor output = output_arg != nullptr ? output_tensor : convert(v_output);
  const utils::ElementwiseBroadcastMatch contract_match =
      utils::match_elementwise_broadcast_contract(
          self.sizes(),
          other.sizes(),
          self.scalar_type(),
          other.scalar_type(),
          output.scalar_type(),
          self.is_vulkan(),
          other.is_vulkan(),
          utils::supports_buffer_elementwise_compute(v_self),
          utils::supports_buffer_elementwise_compute(v_other),
          true,
          elementwise_broadcast_op(op_kind),
          block.alpha == 1.0f,
          output_arg != nullptr,
          false);
  const TensorContractProvenance contract_provenance =
      make_tensor_contract_provenance(contract_match.metadata);
  Tensor returned = record_tensor_write_and_return(
      output,
      "aten::binary_op",
      "tensor_buffer",
      {self, other},
      contract_match.matched ? &contract_provenance : nullptr);
  if (!output_arg && runtime_live_chain_alpha_is_one(alpha_arg)) {
    note_runtime_elementwise_binary_live_chain(
        self, other, returned, binary_op_kind_name(op_kind));
  }
  return returned;
}

static Tensor binary_op_tensor_buffer(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor,
    const BinaryOpKind op_kind) {
  return binary_op_tensor_buffer_impl(
      self_arg,
      other_arg,
      alpha_arg,
      shader_descriptor,
      op_kind,
      nullptr);
}

static Tensor binary_op_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& shader_descriptor,
    const api::ShaderInfo& buffer_shader_descriptor,
    const BinaryOpKind op_kind) {
  api::AllocationScope allocation_scope("binary_op");
  utils::validate_replay_tensor_not_stale(
      self_arg, "aten::binary_op.tensor");
  utils::validate_replay_tensor_not_stale(
      other_arg, "aten::binary_op.tensor");
  if (auto deferred_scale = try_start_deferred_image_normalize_tensor_scale(
          self_arg, other_arg, alpha_arg, op_kind)) {
    return *deferred_scale;
  }
  if (auto deferred_scale = try_start_deferred_attention_query_scale_tensor(
          self_arg, other_arg, alpha_arg, op_kind)) {
    return *deferred_scale;
  }
  if (auto deferred_scale = try_start_deferred_attention_query_scale_tensor(
          other_arg, self_arg, alpha_arg, op_kind)) {
    return *deferred_scale;
  }
  if (auto deferred = try_update_deferred_image_normalize_tensor(
          self_arg, other_arg, alpha_arg, op_kind)) {
    return *deferred;
  }
  const Tensor self_attention_materialized =
      materialize_decomposed_attention_candidate_if_needed(self_arg);
  const Tensor other_attention_materialized =
      materialize_decomposed_attention_candidate_if_needed(other_arg);
  const Tensor self_attention_query_scaled =
      materialize_deferred_attention_query_scale_candidate_if_needed(
          self_attention_materialized);
  const Tensor other_attention_query_scaled =
      materialize_deferred_attention_query_scale_candidate_if_needed(
          other_attention_materialized);
  const Tensor self_image_normalized =
      materialize_deferred_image_normalize_candidate_if_needed(
          self_attention_query_scaled);
  const Tensor other_image_normalized =
      materialize_deferred_image_normalize_candidate_if_needed(
          other_attention_query_scaled);
  const Tensor self_input =
      materialize_deferred_add_layer_norm_candidate_if_needed(
          materialize_deferred_linear_gelu_candidate_if_needed(
              self_image_normalized));
  const Tensor other_input =
      materialize_deferred_add_layer_norm_candidate_if_needed(
          materialize_deferred_linear_gelu_candidate_if_needed(
              other_image_normalized));

  if (self_input.dim() > 4 || other_input.dim() > 4) {
    return binary_op_tensor_cpu_fallback(
        self_input, other_input, alpha_arg, op_kind);
  }

  utils::is_broadcastable(self_input, other_input);
  api::Context* const context = api::context();

  Tensor self;
  Tensor other;
  if (should_promote_binary_operands_to_float(self_input, other_input)) {
    self = promote_binary_operand_to_vulkan_float(self_input);
    other = promote_binary_operand_to_vulkan_float(other_input);
  } else {
    self = self_input.is_vulkan() ? self_input : self_input.vulkan();
    if (should_run_buffer_binary_tensor_integral(
            self, other_input, alpha_arg, op_kind)) {
      return binary_op_tensor_buffer_integral(
          self,
          other_input,
          alpha_arg,
          integral_buffer_tensor_shader(convert(self).dtype(), op_kind));
    }

    if (should_run_buffer_binary_tensor_bool(
            self, other_input, alpha_arg, op_kind)) {
      return binary_op_tensor_buffer_bool(
          self, other_input, alpha_arg, bool_buffer_tensor_shader(op_kind));
    }

    if (needs_binary_cpu_fallback(self_input) ||
        needs_binary_cpu_fallback(other_input)) {
      return binary_op_tensor_cpu_fallback(
          self_input, other_input, alpha_arg, op_kind);
    }

    other = binary_op_preprocess_other_arg(other_input);
  }

  if (should_run_buffer_binary_tensor(self, other)) {
    if (op_kind == BinaryOpKind::Mul && !alpha_arg.has_value()) {
      if (auto deferred = try_start_deferred_layer_scale(self, other)) {
        return *deferred;
      }
      if (auto deferred = try_start_deferred_layer_scale(other, self)) {
        return *deferred;
      }
    }

    const bool can_defer_add_layer_norm =
        op_kind == BinaryOpKind::Add &&
        (!alpha_arg.has_value() || alpha_arg->to<float>() == 1.0f) &&
        self.sizes().equals(other.sizes());
    if (can_defer_add_layer_norm) {
      if (auto deferred = try_start_deferred_add_layer_norm(self, other)) {
        return *deferred;
      }
    }
    return binary_op_tensor_buffer(
        self, other, alpha_arg, buffer_shader_descriptor, op_kind);
  }
  self = utils::prepare_vulkan_execution_tensor(
      self, utils::VulkanExecutionPlanKind::TextureComputeInput);
  const vTensor& v_self = convert(self);
  other = utils::prepare_vulkan_execution_tensor(
      other, utils::VulkanExecutionPlanKind::TextureComputeInput);

  const vTensor& v_other = convert(other);

  vTensor v_output{
      context,
      utils::broadcast_size(self_input, other_input),
      v_self.dtype(),
  };

  const double alpha = alpha_arg ? alpha_arg->to<double>() : 1.0;
  const struct Block final {
    uvec4 output_tensor_size;
    uvec4 input_tensor_size;
    uvec4 other_tensor_size;
    float alpha;
  } block{
      {get_dim<Dim4D::Width>(v_output),
       get_dim<Dim4D::Height>(v_output),
       get_dim<Dim4D::Channel>(v_output),
       get_dim<Dim4D::Batch>(v_output)},

      {get_dim<Dim4D::Width>(v_self),
       get_dim<Dim4D::Height>(v_self),
       get_dim<Dim4D::Channel>(v_self),
       get_dim<Dim4D::Batch>(v_self)},

      {get_dim<Dim4D::Width>(v_other),
       get_dim<Dim4D::Height>(v_other),
       get_dim<Dim4D::Channel>(v_other),
       get_dim<Dim4D::Batch>(v_other)},
      // alpha
      safe_downcast<float>(alpha),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return record_tensor_write_and_return(
      convert(v_output), "aten::binary_op", "tensor_texture", {self, other});
}

static Tensor bool_or_tensor_native(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const char* op_name,
    const bool logical) {
  return bool_or_tensor_cpu_fallback(self_arg, other_arg, op_name, logical);
}

static Tensor bool_and_tensor_native(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const char* op_name,
    const bool logical) {
  return bool_and_tensor_cpu_fallback(self_arg, other_arg, op_name, logical);
}

static Tensor bool_not_tensor_native(
    const Tensor& self_arg,
    const char* op_name,
    const bool logical) {
  return bool_not_tensor_cpu_fallback(self_arg, op_name, logical);
}

static Tensor& bool_or_tensor_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out,
    const char* op_name,
    const bool logical) {
  report_vulkan_cpu_fallback(
      op_name, "bool_or_out_cpu_fallback", {self, other, out});
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    const Tensor self_cpu = self.is_vulkan() ? self.cpu() : self;
    const Tensor other_cpu = other.is_vulkan() ? other.cpu() : other;
    cpu_result =
        logical ? at::logical_or(self_cpu, other_cpu)
                : at::bitwise_or(self_cpu, other_cpu);
  }
  out.copy_(cpu_result);
  return out;
}

static Tensor& bool_and_tensor_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out,
    const char* op_name,
    const bool logical) {
  report_vulkan_cpu_fallback(
      op_name, "bool_and_out_cpu_fallback", {self, other, out});
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    const Tensor self_cpu = self.is_vulkan() ? self.cpu() : self;
    const Tensor other_cpu = other.is_vulkan() ? other.cpu() : other;
    cpu_result =
        logical ? at::logical_and(self_cpu, other_cpu)
                : at::bitwise_and(self_cpu, other_cpu);
  }
  out.copy_(cpu_result);
  return out;
}

static Tensor& bool_not_tensor_out(
    const Tensor& self,
    Tensor& out,
    const char* op_name,
    const bool logical) {
  report_vulkan_cpu_fallback(op_name, "bool_not_out_cpu_fallback", {self, out});
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    const Tensor self_cpu = self.is_vulkan() ? self.cpu() : self;
    cpu_result = logical ? at::logical_not(self_cpu) : at::bitwise_not(self_cpu);
  }
  out.copy_(cpu_result);
  return out;
}

static Tensor quantized_binary_op_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point,
    const api::ShaderInfo& shader_descriptor) {
  api::AllocationScope allocation_scope("qbinary_op");
  utils::is_broadcastable(self_arg, other_arg);
  api::Context* const context = api::context();

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);
  const Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();
  const vTensor& v_other = convert(other);

  TORCH_CHECK(v_self.is_quantized(), "Input tensor is not quantized");
  TORCH_CHECK(v_other.is_quantized(), "Input tensor is not quantized");

  vTensor v_output{
      context,
      utils::broadcast_size(self_arg, other_arg),
      scale,
      zero_point,
      api::kQUInt8,
  };

  const double scale1 = v_self.get_scale();
  const double scale2 = v_other.get_scale();
  const int64_t zero_point1 = v_self.get_zero_point();
  const int64_t zero_point2 = v_other.get_zero_point();
  const struct Block final {
    uvec3 extents;
    uint32_t channelSize;
    uvec3 input1Extents;
    uint32_t channelBatchSize1;
    uvec3 input2Extents;
    uint32_t channelBatchSize2;
    float scale1;
    float scale2;
    int32_t zeroPoint1;
    int32_t zeroPoint2;
    float scale;
    float fill1;
    int32_t zeroPoint;
    int32_t fill2;
  } block{
      v_output.extents(),
      get_dim<Dim4D::Channel>(v_output),
      v_self.extents(),
      get_dim<Dim4D::Channel>(self) * get_dim<Dim4D::Batch>(self),
      v_other.extents(),
      get_dim<Dim4D::Channel>(other) * get_dim<Dim4D::Batch>(other),
      safe_downcast<float>(scale1),
      safe_downcast<float>(scale2),
      safe_downcast<int32_t>(zero_point1),
      safe_downcast<int32_t>(zero_point2),
      safe_downcast<float>(scale),
      0.0f,
      safe_downcast<int32_t>(zero_point),
      0u,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      shader_descriptor,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      v_output.extents(),
      // local work group size
      adaptive_work_group_size(v_output.extents()),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_self.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return convert_quantized(v_output);
}

static Tensor& binary_op_tensor_(
    Tensor& self_arg,
    const Tensor& other_arg,
    const std::optional<Scalar>& alpha_arg,
    const api::ShaderInfo& inplace_shader_descriptor,
    const api::ShaderInfo& out_shader_descriptor,
    const api::ShaderInfo& buffer_shader_descriptor,
    const BinaryOpKind op_kind) {
  TORCH_CHECK(
      get_dim<Dim4D::Batch>(self_arg) >= get_dim<Dim4D::Batch>(other_arg) &&
          get_dim<Dim4D::Channel>(self_arg) >=
              get_dim<Dim4D::Channel>(other_arg) &&
          get_dim<Dim4D::Height>(self_arg) >=
              get_dim<Dim4D::Height>(other_arg) &&
          get_dim<Dim4D::Width>(self_arg) >= get_dim<Dim4D::Width>(other_arg),
      "Dimensions of input tensor to Vulkan in-place binary elementwise op "
      "must be less than or equal the dimensions of the underlying tensor.");

  utils::is_broadcastable(self_arg, other_arg);

  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan: In-place operator is only supported on Vulkan tensors.");

  api::Context* const context = api::context();

  vTensor& v_self = convert(self_arg);
  if (v_self.storage_type() == api::StorageType::BUFFER) {
    Tensor result = binary_op_tensor(
        self_arg,
        other_arg,
        alpha_arg,
        out_shader_descriptor,
        buffer_shader_descriptor,
        op_kind);
    ops::copy_(self_arg, result);
    return self_arg;
  }

  Tensor other = binary_op_preprocess_other_arg(other_arg);
  other = utils::prepare_vulkan_execution_tensor(
      other, utils::VulkanExecutionPlanKind::TextureComputeInput);

  const vTensor& v_other = convert(other);

  const double alpha = alpha_arg ? alpha_arg->to<double>() : 1.0;
  const struct Block final {
    uvec4 input_tensor_size;
    uvec4 other_tensor_size;
    float alpha;
  } block{
      {get_dim<Dim4D::Width>(v_self),
       get_dim<Dim4D::Height>(v_self),
       get_dim<Dim4D::Channel>(v_self),
       get_dim<Dim4D::Batch>(v_self)},

      {get_dim<Dim4D::Width>(v_other),
       get_dim<Dim4D::Height>(v_other),
       get_dim<Dim4D::Channel>(v_other),
       get_dim<Dim4D::Batch>(v_other)},
      // alpha
      safe_downcast<float>(alpha),
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      inplace_shader_descriptor,
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
      v_other.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return self_arg;
}

static Tensor add_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha) {
  return binary_op_scalar(
      self_arg,
      other,
      std::optional<Scalar>(alpha),
      VK_KERNEL(add_scalar),
      VK_KERNEL(buffer_add_scalar),
      BinaryOpKind::Add);
}

static Tensor& add_scalar_(
    Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  return binary_op_scalar_(
      self,
      other,
      std::optional<Scalar>(alpha),
      VK_KERNEL(add_scalar_inplace),
      VK_KERNEL(add_scalar),
      VK_KERNEL(buffer_add_scalar),
      BinaryOpKind::Add);
}

Tensor quantized_add(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_binary_op_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_add));
}

Tensor quantized_sub(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_binary_op_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_sub));
}

Tensor quantized_mul(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_binary_op_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_mul));
}

Tensor quantized_div(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const double scale,
    const int64_t zero_point) {
  return quantized_binary_op_tensor(
      self_arg, other_arg, scale, zero_point, VK_KERNEL(quantized_div));
}

static Tensor add_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha) {
  return binary_op_tensor(
      self_arg,
      other_arg,
      std::optional<Scalar>(alpha),
      VK_KERNEL(add),
      VK_KERNEL(buffer_add),
      BinaryOpKind::Add);
}

Tensor add_buffer_out_vulkan(
    const Tensor& self,
    const Tensor& other,
    Tensor& output,
    const std::optional<Scalar>& alpha) {
  TORCH_CHECK(
      should_run_buffer_binary_tensor(self, other),
      "Vulkan add_buffer_out expects float buffer-backed tensors");
  return binary_op_tensor_buffer_impl(
      self,
      other,
      alpha,
      VK_KERNEL(buffer_add),
      BinaryOpKind::Add,
      &output);
}

std::optional<Tensor> try_add_scaled_buffer_out_vulkan(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Tensor& scale_arg,
    Tensor& output_arg) {
  if (!scale_arg.defined() || !output_arg.defined() || !output_arg.is_vulkan()) {
    return std::nullopt;
  }

  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();
  Tensor scale = scale_arg.is_vulkan() ? scale_arg : scale_arg.vulkan();

  if (!should_run_add_scaled_buffer_out(self, other, scale, output_arg)) {
    return std::nullopt;
  }

  api::AllocationScope allocation_scope("add_scaled.buffer");
  utils::log_vulkan_op_hit("aten::add_scaled.buffer_float");
  api::Context* const context = api::context();

  self = utils::prepare_vulkan_execution_tensor(
      self, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  other = utils::prepare_vulkan_execution_tensor(
      other, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  scale = utils::prepare_vulkan_execution_tensor(
      scale, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  Tensor output = utils::mark_tensor_execution(
      output_arg,
      utils::resolve_buffer_execution_layout(convert(output_arg)),
      false);

  const vTensor& v_self = convert(self);
  const vTensor& v_other = convert(other);
  const vTensor& v_scale = convert(scale);
  vTensor& v_output = convert(output);

  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(v_output.numel()),
      1u,
      1u,
  };
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer self_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_self);
  api::UniformParamsBuffer other_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_other);
  api::UniformParamsBuffer scale_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_scale);

  context->submit_compute_job(
      VK_KERNEL(add_scaled_buffer_float),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      self_meta.buffer(),
      v_other.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      other_meta.buffer(),
      v_scale.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      scale_meta.buffer());

  return output;
}

std::optional<std::pair<Tensor, Tensor>> try_add_relu_buffer_out_vulkan(
    const Tensor& self_arg,
    const Tensor& other_arg,
    Tensor& add_output_arg,
    Tensor& relu_output_arg) {
  if (
      !add_output_arg.defined() || !relu_output_arg.defined() ||
      !add_output_arg.is_vulkan() || !relu_output_arg.is_vulkan()) {
    return std::nullopt;
  }

  Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  Tensor other = other_arg.is_vulkan() ? other_arg : other_arg.vulkan();

  if (!should_run_add_relu_buffer_out(
          self, other, add_output_arg, relu_output_arg)) {
    return std::nullopt;
  }

  api::AllocationScope allocation_scope("add_relu.buffer");
  utils::log_vulkan_op_hit("aten::add_relu.buffer_float");
  api::Context* const context = api::context();

  self = utils::prepare_vulkan_execution_tensor(
      self, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  other = utils::prepare_vulkan_execution_tensor(
      other, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  Tensor add_output = utils::mark_tensor_execution(
      add_output_arg,
      utils::resolve_buffer_execution_layout(convert(add_output_arg)),
      false);
  Tensor relu_output = utils::mark_tensor_execution(
      relu_output_arg,
      utils::resolve_buffer_execution_layout(convert(relu_output_arg)),
      false);

  const vTensor& v_self = convert(self);
  const vTensor& v_other = convert(other);
  vTensor& v_add_output = convert(add_output);
  vTensor& v_relu_output = convert(relu_output);

  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(v_add_output.numel()),
      1u,
      1u,
  };
  api::UniformParamsBuffer add_out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_add_output);
  api::UniformParamsBuffer relu_out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_relu_output);
  api::UniformParamsBuffer self_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_self);
  api::UniformParamsBuffer other_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_other);

  context->submit_compute_job(
      VK_KERNEL(add_relu_buffer_float),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_add_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      add_out_meta.buffer(),
      v_relu_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      relu_out_meta.buffer(),
      v_self.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      self_meta.buffer(),
      v_other.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      other_meta.buffer());

  return std::make_pair(add_output, relu_output);
}

static Tensor& add_tensor_(
    Tensor& self,
    const Tensor& other_arg,
    const Scalar& alpha) {
  return binary_op_tensor_(
      self,
      other_arg,
      std::optional<Scalar>(alpha),
      VK_KERNEL(add_inplace),
      VK_KERNEL(add),
      VK_KERNEL(buffer_add),
      BinaryOpKind::Add);
}

static Tensor sub_scalar(
    const Tensor& self_arg,
    const Scalar& other,
    const Scalar& alpha) {
  return binary_op_scalar(
      self_arg,
      other,
      std::optional<Scalar>(-1 * alpha.to<float>()),
      VK_KERNEL(add_scalar),
      VK_KERNEL(buffer_add_scalar),
      BinaryOpKind::Sub);
}

static Tensor& sub_scalar_(
    Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  return binary_op_scalar_(
      self,
      other,
      std::optional<Scalar>(-1 * alpha.to<float>()),
      VK_KERNEL(add_scalar_inplace),
      VK_KERNEL(add_scalar),
      VK_KERNEL(buffer_add_scalar),
      BinaryOpKind::Sub);
}

static Tensor sub_tensor(
    const Tensor& self_arg,
    const Tensor& other_arg,
    const Scalar& alpha) {
  return binary_op_tensor(
      self_arg,
      other_arg,
      std::optional<Scalar>(alpha),
      VK_KERNEL(sub),
      VK_KERNEL(buffer_sub),
      BinaryOpKind::Sub);
}

static Tensor& sub_tensor_(
    Tensor& self,
    const Tensor& other_arg,
    const Scalar& alpha) {
  return binary_op_tensor_(
      self,
      other_arg,
      std::optional<Scalar>(alpha),
      VK_KERNEL(sub_inplace),
      VK_KERNEL(sub),
      VK_KERNEL(buffer_sub),
      BinaryOpKind::Sub);
}

static Tensor mul_scalar(const Tensor& self_arg, const Scalar& other) {
  return binary_op_scalar(
      self_arg,
      other,
      std::optional<Scalar>(),
      VK_KERNEL(mul_scalar),
      VK_KERNEL(buffer_mul_scalar),
      BinaryOpKind::Mul);
}

static Tensor& mul_scalar_(Tensor& self, const Scalar& other) {
  return binary_op_scalar_(
      self,
      other,
      std::optional<Scalar>(),
      VK_KERNEL(mul_scalar_inplace),
      VK_KERNEL(mul_scalar),
      VK_KERNEL(buffer_mul_scalar),
      BinaryOpKind::Mul);
}

static Tensor mul_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  return binary_op_tensor(
      self_arg,
      other_arg,
      std::optional<Scalar>(),
      VK_KERNEL(mul),
      VK_KERNEL(buffer_mul),
      BinaryOpKind::Mul);
}

static Tensor& mul_tensor_(Tensor& self, const Tensor& other_arg) {
  return binary_op_tensor_(
      self,
      other_arg,
      std::optional<Scalar>(),
      VK_KERNEL(mul_inplace),
      VK_KERNEL(mul),
      VK_KERNEL(buffer_mul),
      BinaryOpKind::Mul);
}

static Tensor bitwise_or_tensor(const Tensor& self, const Tensor& other) {
  return bool_or_tensor_native(self, other, "aten::bitwise_or", false);
}

static Tensor& bitwise_or_tensor_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  return bool_or_tensor_out(self, other, out, "aten::bitwise_or", false);
}

static Tensor logical_or_tensor(const Tensor& self, const Tensor& other) {
  return bool_or_tensor_native(self, other, "aten::logical_or", true);
}

static Tensor& logical_or_tensor_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  return bool_or_tensor_out(self, other, out, "aten::logical_or", true);
}

static Tensor bitwise_and_tensor(const Tensor& self, const Tensor& other) {
  return bool_and_tensor_native(self, other, "aten::bitwise_and", false);
}

static Tensor& bitwise_and_tensor_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  return bool_and_tensor_out(self, other, out, "aten::bitwise_and", false);
}

static Tensor logical_and_tensor(const Tensor& self, const Tensor& other) {
  return bool_and_tensor_native(self, other, "aten::logical_and", true);
}

static Tensor& logical_and_tensor_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  return bool_and_tensor_out(self, other, out, "aten::logical_and", true);
}

static Tensor bitwise_not_tensor(const Tensor& self) {
  return bool_not_tensor_native(self, "aten::bitwise_not", false);
}

static Tensor& bitwise_not_tensor_out(const Tensor& self, Tensor& out) {
  return bool_not_tensor_out(self, out, "aten::bitwise_not", false);
}

static Tensor logical_not_tensor(const Tensor& self) {
  return bool_not_tensor_native(self, "aten::logical_not", true);
}

static Tensor& logical_not_tensor_out(const Tensor& self, Tensor& out) {
  return bool_not_tensor_out(self, out, "aten::logical_not", true);
}

static Tensor maximum_tensor(const Tensor& self, const Tensor& other) {
  return maximum_tensor_small_control_cpu_fallback(self, other);
}

static Tensor& maximum_tensor_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  const Tensor result = maximum_tensor_small_control_cpu_fallback(self, other);
  out.copy_(result);
  return out;
}

static Tensor div_scalar(const Tensor& self_arg, const Scalar& other) {
  return binary_op_scalar(
      self_arg,
      1.0 / other.to<float>(),
      std::optional<Scalar>(),
      VK_KERNEL(mul_scalar),
      VK_KERNEL(buffer_mul_scalar),
      BinaryOpKind::Div);
}

static Tensor& div_scalar_(Tensor& self, const Scalar& other) {
  return binary_op_scalar_(
      self,
      1.0 / other.to<float>(),
      std::optional<Scalar>(),
      VK_KERNEL(mul_scalar_inplace),
      VK_KERNEL(mul_scalar),
      VK_KERNEL(buffer_mul_scalar),
      BinaryOpKind::Div);
}

static Tensor div_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  return binary_op_tensor(
      self_arg,
      other_arg,
      std::optional<Scalar>(),
      VK_KERNEL(div),
      VK_KERNEL(buffer_div),
      BinaryOpKind::Div);
}

static Tensor& div_tensor_(Tensor& self, const Tensor& other_arg) {
  return binary_op_tensor_(
      self,
      other_arg,
      std::optional<Scalar>(),
      VK_KERNEL(div_inplace),
      VK_KERNEL(div),
      VK_KERNEL(buffer_div),
      BinaryOpKind::Div);
}

static Tensor div_scalar_mode(
    const Tensor& self_arg,
    const Scalar& other,
    std::optional<c10::string_view> rounding_mode) {
  if (!rounding_mode.has_value()) {
    return div_scalar(self_arg, other);
  }

  report_vulkan_cpu_fallback(
      "aten::div", "scalar_mode_cpu_fallback", {self_arg});
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
    cpu_result = at::div(self_cpu, other, rounding_mode);
  }
  return record_tensor_write_and_return(
      cpu_result.vulkan(), "aten::div", "scalar_mode_cpu_fallback", {self_arg});
}

static Tensor& div_scalar_mode_(
    Tensor& self,
    const Scalar& other,
    std::optional<c10::string_view> rounding_mode) {
  Tensor result = div_scalar_mode(self, other, rounding_mode);
  if (self.is_vulkan()) {
    ops::copy_(self, result);
  } else {
    self.copy_(result.cpu());
  }
  return self;
}

static Tensor& div_scalar_mode_out(
    const Tensor& self,
    const Scalar& other,
    std::optional<c10::string_view> rounding_mode,
    Tensor& out) {
  Tensor result = div_scalar_mode(self, other, rounding_mode);
  if (out.is_vulkan()) {
    ops::copy_(out, result);
  } else {
    out.copy_(result.cpu());
  }
  return out;
}

static Tensor div_tensor_mode(
    const Tensor& self_arg,
    const Tensor& other_arg,
    std::optional<c10::string_view> rounding_mode) {
  if (!rounding_mode.has_value()) {
    return div_tensor(self_arg, other_arg);
  }

  report_vulkan_cpu_fallback(
      "aten::div", "tensor_mode_cpu_fallback", {self_arg, other_arg});
  Tensor cpu_result;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
    const Tensor other_cpu = other_arg.is_vulkan() ? other_arg.cpu() : other_arg;
    cpu_result = at::div(self_cpu, other_cpu, rounding_mode);
  }
  return record_tensor_write_and_return(
      cpu_result.vulkan(),
      "aten::div",
      "tensor_mode_cpu_fallback",
      {self_arg, other_arg});
}

static Tensor& div_tensor_mode_(
    Tensor& self,
    const Tensor& other,
    std::optional<c10::string_view> rounding_mode) {
  Tensor result = div_tensor_mode(self, other, rounding_mode);
  if (self.is_vulkan()) {
    ops::copy_(self, result);
  } else {
    self.copy_(result.cpu());
  }
  return self;
}

static Tensor& div_tensor_mode_out(
    const Tensor& self,
    const Tensor& other,
    std::optional<c10::string_view> rounding_mode,
    Tensor& out) {
  Tensor result = div_tensor_mode(self, other, rounding_mode);
  if (out.is_vulkan()) {
    ops::copy_(out, result);
  } else {
    out.copy_(result.cpu());
  }
  return out;
}

static Tensor pow(const Tensor& self, const Tensor& other) {
  return binary_op_tensor(
      self,
      other,
      std::optional<Scalar>(),
      VK_KERNEL(pow),
      VK_KERNEL(buffer_pow),
      BinaryOpKind::Pow);
}

static Tensor& pow_(Tensor& self, const Tensor& other) {
  return binary_op_tensor_(
      self,
      other,
      std::optional<Scalar>(),
      VK_KERNEL(pow_inplace),
      VK_KERNEL(pow),
      VK_KERNEL(buffer_pow),
      BinaryOpKind::Pow);
}

static Tensor pow_tensor_scalar(const Tensor& self, const Scalar& other) {
  if (scalar_is_integral_exponent(other)) {
    return pow_tensor_scalar_integral_exponent(
        self, scalar_to_integral_exponent(other));
  }
  return binary_op_scalar(
      self,
      other,
      std::optional<Scalar>(),
      VK_KERNEL(pow_tensor_scalar),
      VK_KERNEL(buffer_pow_tensor_scalar),
      BinaryOpKind::Pow);
}

static Tensor& pow_tensor_scalar_(Tensor& self, const Scalar& other) {
  if (scalar_is_integral_exponent(other)) {
    Tensor result = pow_tensor_scalar_integral_exponent(
        self, scalar_to_integral_exponent(other));
    return rebind_vulkan_output(self, result);
  }
  return binary_op_scalar_(
      self,
      other,
      std::optional<Scalar>(),
      VK_KERNEL(pow_tensor_scalar_inplace),
      VK_KERNEL(pow_tensor_scalar),
      VK_KERNEL(buffer_pow_tensor_scalar),
      BinaryOpKind::Pow);
}

static Tensor pow_scalar_tensor(const Scalar& self, const Tensor& other) {
  return binary_op_scalar(
      other,
      self,
      std::optional<Scalar>(),
      VK_KERNEL(pow_scalar_tensor),
      VK_KERNEL(buffer_pow_tensor_scalar),
      BinaryOpKind::Pow);
}

static Tensor floor_divide_scalar(const Tensor& self, const Scalar& other) {
  TORCH_CHECK(
      other.to<float>() != 0.0f, "floor_divide_scalar: can't divide by zero");
  return binary_op_scalar(
      self,
      1.0 / other.to<float>(),
      std::optional<Scalar>(),
      VK_KERNEL(floor_mul_scalar),
      VK_KERNEL(buffer_floor_mul_scalar),
      BinaryOpKind::FloorDivide);
}

static Tensor& floor_divide_scalar_(Tensor& self, const Scalar& other) {
  TORCH_CHECK(
      other.to<float>() != 0.0f, "floor_divide_scalar_: can't divide by zero");
  return binary_op_scalar_(
      self,
      1.0 / other.to<float>(),
      std::optional<Scalar>(),
      VK_KERNEL(floor_mul_scalar_inplace),
      VK_KERNEL(floor_mul_scalar),
      VK_KERNEL(buffer_floor_mul_scalar),
      BinaryOpKind::FloorDivide);
}

static Tensor floor_divide_tensor(const Tensor& self, const Tensor& other) {
  return binary_op_tensor(
      self,
      other,
      std::optional<Scalar>(),
      VK_KERNEL(floor_divide),
      VK_KERNEL(buffer_floor_divide),
      BinaryOpKind::FloorDivide);
}

static Tensor& floor_divide_tensor_(Tensor& self, const Tensor& other_arg) {
  return binary_op_tensor_(
      self,
      other_arg,
      std::optional<Scalar>(),
      VK_KERNEL(floor_divide_inplace),
      VK_KERNEL(floor_divide),
      VK_KERNEL(buffer_floor_divide),
      BinaryOpKind::FloorDivide);
}

template <typename CompareFn>
static Tensor compare_tensor_tensor_cpu_fallback(
    const Tensor& self_arg,
    const Tensor& other_arg,
    CompareFn&& compare_fn) {
  report_vulkan_cpu_fallback(
      "aten::comparison", "tensor_cpu_fallback", {self_arg, other_arg});
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);
  const Tensor self_cpu = self_arg.cpu();
  const Tensor other_cpu = other_arg.cpu();
  const Tensor result_cpu = compare_fn(self_cpu, other_cpu);
  return result_cpu.to(
      self_arg.options().device(self_arg.device()).dtype(result_cpu.scalar_type()));
}

template <typename CompareFn>
static Tensor compare_tensor_scalar_cpu_fallback(
    const Tensor& self_arg,
    const Scalar& other,
    CompareFn&& compare_fn) {
  report_vulkan_cpu_fallback(
      "aten::comparison", "scalar_cpu_fallback", {self_arg});
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);
  const Tensor self_cpu = self_arg.cpu();
  const Tensor result_cpu = compare_fn(self_cpu, other);
  return result_cpu.to(
      self_arg.options().device(self_arg.device()).dtype(result_cpu.scalar_type()));
}

static Tensor lt_tensor(const Tensor& self, const Tensor& other) {
  return compare_tensor_tensor_cpu_fallback(
      self, other, [](const Tensor& lhs, const Tensor& rhs) {
        return lhs.lt(rhs);
      });
}

static Tensor lt_scalar(const Tensor& self, const Scalar& other) {
  return compare_tensor_scalar_cpu_fallback(
      self, other, [](const Tensor& lhs, const Scalar& rhs) {
        return lhs.lt(rhs);
      });
}

static Tensor le_tensor(const Tensor& self, const Tensor& other) {
  return compare_tensor_tensor_cpu_fallback(
      self, other, [](const Tensor& lhs, const Tensor& rhs) {
        return lhs.le(rhs);
      });
}

static Tensor le_scalar(const Tensor& self, const Scalar& other) {
  return compare_tensor_scalar_cpu_fallback(
      self, other, [](const Tensor& lhs, const Scalar& rhs) {
        return lhs.le(rhs);
      });
}

static Tensor gt_tensor(const Tensor& self, const Tensor& other) {
  return compare_tensor_tensor_cpu_fallback(
      self, other, [](const Tensor& lhs, const Tensor& rhs) {
        return lhs.gt(rhs);
      });
}

static Tensor gt_scalar(const Tensor& self, const Scalar& other) {
  return compare_tensor_scalar_cpu_fallback(
      self, other, [](const Tensor& lhs, const Scalar& rhs) {
        return lhs.gt(rhs);
      });
}

static Tensor ge_tensor(const Tensor& self, const Tensor& other) {
  return compare_tensor_tensor_cpu_fallback(
      self, other, [](const Tensor& lhs, const Tensor& rhs) {
        return lhs.ge(rhs);
      });
}

static Tensor ge_scalar(const Tensor& self, const Scalar& other) {
  return compare_tensor_scalar_cpu_fallback(
      self, other, [](const Tensor& lhs, const Scalar& rhs) {
        return lhs.ge(rhs);
      });
}

static Tensor eq_tensor(const Tensor& self, const Tensor& other) {
  return compare_tensor_tensor_cpu_fallback(
      self, other, [](const Tensor& lhs, const Tensor& rhs) {
        return lhs.eq(rhs);
      });
}

static Tensor eq_scalar(const Tensor& self, const Scalar& other) {
  return compare_tensor_scalar_cpu_fallback(
      self, other, [](const Tensor& lhs, const Scalar& rhs) {
        return lhs.eq(rhs);
      });
}

static Tensor ne_tensor(const Tensor& self, const Tensor& other) {
  return compare_tensor_tensor_cpu_fallback(
      self, other, [](const Tensor& lhs, const Tensor& rhs) {
        return lhs.ne(rhs);
      });
}

static Tensor ne_scalar(const Tensor& self, const Scalar& other) {
  return compare_tensor_scalar_cpu_fallback(
      self, other, [](const Tensor& lhs, const Scalar& rhs) {
        return lhs.ne(rhs);
      });
}

Tensor materialize_deferred_image_normalize_candidate_if_needed(
    const Tensor& tensor) {
  if (!tensor.is_vulkan()) {
    return tensor;
  }
  return materialize_deferred_image_normalize_candidate_impl(tensor);
}

void move_deferred_image_normalize_candidate_to_alias(
    const Tensor& source,
    const Tensor& alias) {
  if (!source.is_vulkan() || !alias.is_vulkan()) {
    return;
  }
  auto candidate = take_deferred_image_normalize_candidate(source);
  if (!candidate.has_value()) {
    return;
  }
  update_deferred_image_normalize_view(*candidate, alias);
  register_deferred_image_normalize_candidate(alias, std::move(*candidate));
  utils::log_vulkan_op_hit("aten::image_normalize_bridge.alias");
}

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::add.Scalar"), TORCH_FN(add_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::add_.Scalar"), TORCH_FN(add_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::add.Tensor"), TORCH_FN(add_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::add_.Tensor"), TORCH_FN(add_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub.Scalar"), TORCH_FN(sub_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub_.Scalar"), TORCH_FN(sub_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub.Tensor"), TORCH_FN(sub_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::sub_.Tensor"), TORCH_FN(sub_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul.Scalar"), TORCH_FN(mul_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul_.Scalar"), TORCH_FN(mul_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul.Tensor"), TORCH_FN(mul_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::mul_.Tensor"), TORCH_FN(mul_tensor_));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::bitwise_or.Tensor"),
      TORCH_FN(bitwise_or_tensor));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::bitwise_or.Tensor_out"),
      TORCH_FN(bitwise_or_tensor_out));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::logical_or"),
      TORCH_FN(logical_or_tensor));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::logical_or.out"),
      TORCH_FN(logical_or_tensor_out));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::bitwise_and.Tensor"),
      TORCH_FN(bitwise_and_tensor));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::bitwise_and.Tensor_out"),
      TORCH_FN(bitwise_and_tensor_out));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::logical_and"),
      TORCH_FN(logical_and_tensor));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::logical_and.out"),
      TORCH_FN(logical_and_tensor_out));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::bitwise_not"),
      TORCH_FN(bitwise_not_tensor));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::bitwise_not.out"),
      TORCH_FN(bitwise_not_tensor_out));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::logical_not"),
      TORCH_FN(logical_not_tensor));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::logical_not.out"),
      TORCH_FN(logical_not_tensor_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::maximum"), TORCH_FN(maximum_tensor));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::maximum.out"),
      TORCH_FN(maximum_tensor_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::div.Scalar"), TORCH_FN(div_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::div_.Scalar"), TORCH_FN(div_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::div.Tensor"), TORCH_FN(div_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::div_.Tensor"), TORCH_FN(div_tensor_));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::div.Scalar_mode"),
      TORCH_FN(div_scalar_mode));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::div_.Scalar_mode"),
      TORCH_FN(div_scalar_mode_));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::div.Scalar_mode_out"),
      TORCH_FN(div_scalar_mode_out));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::div.Tensor_mode"),
      TORCH_FN(div_tensor_mode));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::div_.Tensor_mode"),
      TORCH_FN(div_tensor_mode_));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::div.out_mode"),
      TORCH_FN(div_tensor_mode_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::pow.Tensor_Tensor"), TORCH_FN(pow));
  m.impl(TORCH_SELECTIVE_NAME("aten::pow_.Tensor"), TORCH_FN(pow_));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::pow.Tensor_Scalar"),
      TORCH_FN(pow_tensor_scalar));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::pow_.Scalar"), TORCH_FN(pow_tensor_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::pow.Scalar"), TORCH_FN(pow_scalar_tensor));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::floor_divide.Scalar"),
      TORCH_FN(floor_divide_scalar));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::floor_divide_.Scalar"),
      TORCH_FN(floor_divide_scalar_));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::floor_divide"),
      TORCH_FN(floor_divide_tensor));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::floor_divide_.Tensor"),
      TORCH_FN(floor_divide_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::lt.Tensor"), TORCH_FN(lt_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::lt.Scalar"), TORCH_FN(lt_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::le.Tensor"), TORCH_FN(le_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::le.Scalar"), TORCH_FN(le_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::gt.Tensor"), TORCH_FN(gt_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::gt.Scalar"), TORCH_FN(gt_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::ge.Tensor"), TORCH_FN(ge_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::ge.Scalar"), TORCH_FN(ge_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::eq.Tensor"), TORCH_FN(eq_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::eq.Scalar"), TORCH_FN(eq_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::ne.Tensor"), TORCH_FN(ne_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::ne.Scalar"), TORCH_FN(ne_scalar));
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
#endif /* USE_VULKAN_API */
