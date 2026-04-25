#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <algorithm>
#include <array>
#include <c10/util/irange.h>
#include <sstream>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

std::string format_pad_sizes(IntArrayRef sizes) {
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

void log_constant_pad_skip(
    const char* reason,
    const Tensor& self,
    IntArrayRef pad) {
  std::ostringstream stream;
  stream << "aten::constant_pad_nd.unsupported"
         << " reason=" << reason
         << " input=" << format_pad_sizes(self.sizes())
         << " pad=[";
  for (const auto idx : c10::irange(pad.size())) {
    if (idx > 0) {
      stream << ',';
    }
    stream << pad[idx];
  }
  stream << ']';
  if (self.is_vulkan()) {
    const vTensor& v_self = convert(self);
    stream << " storage="
           << (v_self.storage_type() == api::StorageType::BUFFER ? "buffer"
                                                                  : "texture")
           << " direct=" << (v_self.has_direct_buffer_layout() ? 1 : 0)
           << " offset=" << v_self.storage_offset();
  }
  utils::log_vulkan_op_hit(stream.str());
}

Tensor pad2d(
    const Tensor& self_arg,
    IntArrayRef padding,
    const api::ShaderInfo& shader_descriptor) {
  const int pad_dim = padding.size();
  const IntArrayRef input_size = self_arg.sizes();
  const int input_dim = input_size.size();

  TORCH_CHECK(
      pad_dim == 1 || pad_dim == 4,
      "Padding sizes must be a 1-tuple or 4-tuple!");
  TORCH_CHECK(input_dim >= 2, "Input tensor must have dim >= 2!");

  api::Context* const context = api::context();

  int pad_left = padding[0];
  int pad_right = padding[0];
  int pad_top = padding[0];
  int pad_bottom = padding[0];
  if (pad_dim == 4) {
    pad_right = padding[1];
    pad_top = padding[2];
    pad_bottom = padding[3];
  }

  const Tensor self = self_arg.is_vulkan() ? self_arg : self_arg.vulkan();
  const vTensor& v_self = convert(self);

  std::vector<int64_t> output_size(input_dim);
  for (const auto d : c10::irange(input_dim)) {
    if (d == input_dim - 1) {
      output_size[d] = input_size[d] + pad_right + pad_left;
    } else if (d == input_dim - 2) {
      output_size[d] = input_size[d] + pad_top + pad_bottom;
    } else {
      output_size[d] = input_size[d];
    }
  }

  vTensor v_output{
      context,
      output_size,
      v_self.dtype(),
  };

  const struct Block final {
    uvec3 extents;
    uint32_t _;
    uvec4 padding;
  } block{
      v_output.extents(),
      0u,
      {safe_downcast<uint32_t>(pad_left),
       safe_downcast<uint32_t>(pad_right),
       safe_downcast<uint32_t>(pad_top),
       safe_downcast<uint32_t>(pad_bottom)},
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
      // params buffer
      params.buffer());

  return convert(v_output);
}

Tensor reflection_pad2d(const Tensor& self_arg, IntArrayRef padding) {
  return pad2d(self_arg, padding, VK_KERNEL(reflection_pad2d));
}

Tensor replication_pad2d(const Tensor& self_arg, IntArrayRef padding) {
  return pad2d(self_arg, padding, VK_KERNEL(replication_pad2d));
}

Tensor constant_pad_nd_vulkan(
    const Tensor& self_arg,
    IntArrayRef pad,
    const Scalar& value) {
  utils::log_vulkan_op_hit("aten::constant_pad_nd");
  TORCH_CHECK(
      self_arg.is_vulkan(),
      "Vulkan constant_pad_nd expected a Vulkan input tensor");
  if (self_arg.dim() != 4) {
    log_constant_pad_skip("input_rank_not_4", self_arg, pad);
    TORCH_CHECK(
        false,
        "Vulkan constant_pad_nd currently supports only rank-4 tensors, got ",
        self_arg.sizes());
  }
  if (pad.size() % 2 != 0 || pad.size() > 8) {
    log_constant_pad_skip("invalid_pad_rank", self_arg, pad);
    TORCH_CHECK(
        false,
        "Vulkan constant_pad_nd expected an even pad list with at most 8 values, got ",
        pad);
  }
  for (const auto idx : c10::irange(pad.size())) {
    if (pad[idx] < 0) {
      log_constant_pad_skip("negative_padding", self_arg, pad);
      TORCH_CHECK(
          false,
          "Vulkan constant_pad_nd does not support negative/cropping padding yet, got ",
          pad);
    }
  }
  if (self_arg.scalar_type() != c10::ScalarType::Float) {
    log_constant_pad_skip("dtype_not_float", self_arg, pad);
    TORCH_CHECK(
        false,
        "Vulkan constant_pad_nd currently supports float tensors only, got ",
        self_arg.scalar_type());
  }

  Tensor self = utils::ensure_buffer_storage(
      self_arg, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  const vTensor& v_self = convert(self);
  TORCH_CHECK(
      v_self.storage_type() == api::StorageType::BUFFER &&
          utils::supports_buffer_elementwise_compute(v_self),
      "Vulkan constant_pad_nd requires a supported buffer-backed input");

  std::vector<int64_t> output_sizes = self.sizes().vec();
  std::array<int64_t, 4> pad_before_nchw{0, 0, 0, 0};
  for (const auto pair_idx : c10::irange(pad.size() / 2)) {
    const int64_t before = pad[2 * pair_idx];
    const int64_t after = pad[2 * pair_idx + 1];
    const int64_t dim = self.dim() - 1 - static_cast<int64_t>(pair_idx);
    output_sizes[dim] += before + after;
    pad_before_nchw[dim] = before;
  }
  for (const int64_t size : output_sizes) {
    TORCH_CHECK(
        size >= 0,
        "Vulkan constant_pad_nd produced a negative output size for input ",
        self.sizes(),
        " and pad ",
        pad);
  }

  api::Context* const context = api::context();
  Tensor output =
      utils::create_buffer_tensor(output_sizes, c10::ScalarType::Float);
  vTensor& v_output = convert(output);
  vTensor& v_input = convert(self);

  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);

  const float pad_value = value.to<float>();
  const struct Block final {
    ivec4 pad_before;
    vec4 values;
  } block{
      {safe_downcast<int32_t>(pad_before_nchw[3]),
       safe_downcast<int32_t>(pad_before_nchw[2]),
       safe_downcast<int32_t>(pad_before_nchw[1]),
       safe_downcast<int32_t>(pad_before_nchw[0])},
      {pad_value, 0.0f, 0.0f, 0.0f},
  };
  api::UniformParamsBuffer params(context, block);

  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
      1u,
      1u,
  };
  context->submit_compute_job(
      VK_KERNEL(buffer_constant_pad_nd),
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
      params.buffer());

  std::ostringstream stream;
  stream << "aten::constant_pad_nd.buffer"
         << " input=" << format_pad_sizes(self.sizes())
         << " output=" << format_pad_sizes(output.sizes())
         << " output_layout=width_packed"
         << " output_direct=" << (v_output.has_direct_buffer_layout() ? 1 : 0)
         << " pad=";
  for (const auto idx : c10::irange(pad.size())) {
    stream << (idx == 0 ? '[' : ',') << pad[idx];
  }
  stream << ']';
  utils::log_vulkan_op_hit(stream.str());
  return output;
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::constant_pad_nd"),
      TORCH_FN(constant_pad_nd_vulkan));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::reflection_pad2d"),
      TORCH_FN(reflection_pad2d));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::replication_pad2d"),
      TORCH_FN(replication_pad2d));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
