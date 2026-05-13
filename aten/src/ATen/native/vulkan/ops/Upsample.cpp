#include <ATen/native/UpSample.h>
#include <ATen/ops/_upsample_nearest_exact2d.h>
#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/FallbackPolicy.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/native/vulkan/ops/Upsample.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <c10/core/InferenceMode.h>
#include <torch/library.h>

#include <algorithm>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
using namespace api::utils;

Tensor prepare_upsample_texture_input(const Tensor& input_arg) {
  return utils::prepare_vulkan_execution_tensor(
      input_arg, utils::VulkanExecutionPlanKind::TextureComputeInput);
}

bool should_run_buffer_upsample(const Tensor& input_arg) {
  if (!input_arg.is_vulkan() || input_arg.scalar_type() != kFloat) {
    return false;
  }

  const vTensor& v_input = convert(input_arg);
  return v_input.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_elementwise_compute(v_input);
}

bool should_materialize_texture_bilinear_input_to_buffer(
    const Tensor& input_arg,
    const IntArrayRef output_sizes) {
  if (!input_arg.is_vulkan() || input_arg.scalar_type() != kFloat) {
    return false;
  }

  const vTensor& v_input = convert(input_arg);
  if (
      v_input.storage_type() == api::StorageType::BUFFER ||
      !utils::supports_buffer_view_fast_path(v_input)) {
    return false;
  }

  const int64_t input_h = get_dim<Dim4D::Height>(input_arg);
  const int64_t input_w = get_dim<Dim4D::Width>(input_arg);
  const int64_t output_h = output_sizes[Layout::Parameter::height];
  const int64_t output_w = output_sizes[Layout::Parameter::width];

  return get_dim<Dim4D::Batch>(input_arg) == 1 &&
      get_dim<Dim4D::Channel>(input_arg) <= 4 && output_h >= input_h &&
      output_w >= input_w;
}

bool should_run_buffer_nearest_upsample(const Tensor& input_arg) {
  if (!input_arg.is_vulkan()) {
    return false;
  }

  if (input_arg.scalar_type() == kFloat) {
    return true;
  }

  return input_arg.scalar_type() == kByte &&
      api::context()->adapter_ptr()->supports_int8_buffer_arithmetic();
}

Tensor upsample_bilinear2d_buffer_impl(
    const Tensor& input_arg,
    const IntArrayRef output_sizes,
    bool align_corners,
    const std::optional<double> scales_h,
    const std::optional<double> scales_w,
    Tensor* output_arg) {
  utils::log_vulkan_op_hit("aten::upsample_bilinear2d.buffer_float");
  api::Context* const context = api::context();
  const vTensor& v_input = convert(input_arg);

  const std::vector<int64_t> expected_output_sizes{
      get_dim<Dim4D::Batch>(v_input),
      get_dim<Dim4D::Channel>(v_input),
      output_sizes[Layout::Parameter::height],
      output_sizes[Layout::Parameter::width],
  };

  Tensor output_tensor;
  vTensor* v_output_ptr = nullptr;
  vTensor owned_output;
  if (output_arg != nullptr) {
    TORCH_CHECK(
        output_arg->defined(),
        "Vulkan bilinear upsample out expects a defined output tensor");
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
        "Vulkan bilinear upsample out expects float buffer-backed output");
    TORCH_CHECK(
        output_tensor.sizes().vec() == expected_output_sizes,
        "Vulkan bilinear upsample out received mismatched output shape");
    v_output_ptr = &v_output;
  } else {
    owned_output = vTensor{
        context,
        expected_output_sizes,
        v_input.dtype(),
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
    };
    v_output_ptr = &owned_output;
  }
  vTensor& v_output = *v_output_ptr;

  const struct Block final {
    ivec4 info;
    vec4 scale;
  } block{
      {
          safe_downcast<int32_t>(get_dim<Dim4D::Width>(input_arg) - 1),
          safe_downcast<int32_t>(get_dim<Dim4D::Height>(input_arg) - 1),
          safe_downcast<int32_t>(get_dim<Dim4D::Width>(v_output)),
          safe_downcast<int32_t>(get_dim<Dim4D::Height>(v_output)),
      },
      {
          compute_scales_value<float>(
              scales_w,
              get_dim<Dim4D::Width>(input_arg),
              get_dim<Dim4D::Width>(v_output)),
          compute_scales_value<float>(
              scales_h,
              get_dim<Dim4D::Height>(input_arg),
              get_dim<Dim4D::Height>(v_output)),
          0.0f,
          0.0f,
      },
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
      1u,
      1u,
  };
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);

  context->submit_compute_job(
      align_corners ? VK_KERNEL(upsample_bilinear2d_buffer_align_true)
                    : VK_KERNEL(upsample_bilinear2d_buffer_align_false),
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

  return record_tensor_write_and_return(
      output_arg != nullptr ? output_tensor : convert(v_output),
      "aten::upsample_bilinear2d",
      "buffer",
      {input_arg});
}

Tensor upsample_nearest2d_buffer_impl(
    const Tensor& input_arg,
    const IntArrayRef output_sizes,
    const std::optional<double> scales_h,
    const std::optional<double> scales_w) {
  utils::log_vulkan_op_hit("aten::upsample_nearest2d.buffer");
  api::AllocationScope allocation_scope("upsample_nearest.buffer");
  api::Context* const context = api::context();

  Tensor prepared = utils::prepare_vulkan_direct_buffer_execution_tensor(
      input_arg, utils::VulkanExecutionPlanKind::ElementwiseBufferInput);
  const vTensor& v_input = convert(prepared);

  const bool is_float = prepared.scalar_type() == kFloat;
  const bool is_uint8 = prepared.scalar_type() == kByte;
  TORCH_CHECK(
      (is_float && utils::supports_buffer_elementwise_compute(v_input)) ||
          (is_uint8 && utils::supports_native_integral_buffer_compute(prepared)),
      "Vulkan nearest upsample buffer path expects float or uint8 direct buffers");

  const std::vector<int64_t> output_tensor_sizes{
      get_dim<Dim4D::Batch>(v_input),
      get_dim<Dim4D::Channel>(v_input),
      output_sizes[Layout::Parameter::height],
      output_sizes[Layout::Parameter::width],
  };

  vTensor v_output{
      context,
      output_tensor_sizes,
      v_input.dtype(),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct Block final {
    ivec4 info;
    vec4 scale;
  } block{
      {
          safe_downcast<int32_t>(get_dim<Dim4D::Width>(prepared) - 1),
          safe_downcast<int32_t>(get_dim<Dim4D::Height>(prepared) - 1),
          safe_downcast<int32_t>(get_dim<Dim4D::Width>(v_output)),
          safe_downcast<int32_t>(get_dim<Dim4D::Height>(v_output)),
      },
      {
          compute_scales_value<float>(
              scales_w,
              get_dim<Dim4D::Width>(prepared),
              get_dim<Dim4D::Width>(v_output)),
          compute_scales_value<float>(
              scales_h,
              get_dim<Dim4D::Height>(prepared),
              get_dim<Dim4D::Height>(v_output)),
          0.0f,
          0.0f,
      },
  };

  api::UniformParamsBuffer params(context, block);
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);

  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
      1u,
      1u,
  };

  context->submit_compute_job(
      is_uint8 ? VK_KERNEL(upsample_nearest2d_buffer_uint8)
               : VK_KERNEL(upsample_nearest2d_buffer_float),
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

  return record_tensor_write_and_return(
      utils::mark_tensor_execution(
          convert(v_output), api::ExecutionLayout::BUFFER_DIRECT),
      "aten::upsample_nearest2d",
      "buffer",
      {prepared});
}

static Tensor upsample_nearest2d(
    const Tensor& input_arg,
    const IntArrayRef output_sizes,
    const std::optional<double> scales_h,
    const std::optional<double> scales_w) {
  if (should_run_buffer_nearest_upsample(input_arg)) {
    return upsample_nearest2d_buffer_impl(
        input_arg, output_sizes, scales_h, scales_w);
  }

  if (
      !input_arg.is_quantized() &&
      c10::isIntegralType(input_arg.scalar_type(), /*includeBool=*/true)) {
    Tensor float_input = utils::cast_vulkan_tensor_dtype(input_arg, kFloat);
    Tensor float_output =
        upsample_nearest2d(float_input, output_sizes, scales_h, scales_w);
    return utils::cast_vulkan_tensor_dtype(float_output, input_arg.scalar_type());
  }

  api::AllocationScope allocation_scope("upsample_nearest");
  api::Context* const context = api::context();

  TORCH_CHECK(
      (4 == input_arg.sizes().size()) && (2 == output_sizes.size()),
      "Invalid input!");

  const Tensor input = prepare_upsample_texture_input(input_arg);
  const vTensor& v_input = convert(input);
  const auto v_input_sizes = v_input.sizes();

  vTensor v_output{
      context,
      {
          v_input_sizes[Layout::Activation4D::batch],
          v_input_sizes[Layout::Activation4D::channels],
          output_sizes[Layout::Parameter::height],
          output_sizes[Layout::Parameter::width],
      },
      v_input.dtype(),
  };

  if (v_input.is_quantized()) {
    v_output.set_is_quantized();
    v_output.set_scale(v_input.get_scale());
    v_output.set_zero_point(v_input.get_zero_point());
  }

  const struct Block final {
    uvec3 extents;
    uint32_t fill0;
    ivec2 iextents;
    vec2 scale;
  } block{
      v_output.extents(),
      0u,
      {
          safe_downcast<int32_t>(
              input_arg.size(Layout::Activation4D::width) - 1),
          safe_downcast<int32_t>(
              input_arg.size(Layout::Activation4D::height) - 1),
      },
      {
          compute_scales_value<float>(
              scales_w,
              v_input_sizes[Layout::Activation4D::width],
              output_sizes[Layout::Parameter::width]),
          compute_scales_value<float>(
              scales_h,
              v_input_sizes[Layout::Activation4D::height],
              output_sizes[Layout::Parameter::height]),
      },
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      v_input.is_quantized() ? VK_KERNEL(quantized_upsample_nearest2d)
                             : VK_KERNEL(upsample_nearest2d),
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
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return record_tensor_write_and_return(
      convert(v_output), "aten::upsample_nearest2d", "texture", {input});
}

static Tensor upsample_nearest_exact2d_cpu_fallback(
    const Tensor& input_arg,
    const IntArrayRef output_sizes,
    const std::optional<double> scales_h,
    const std::optional<double> scales_w) {
  report_vulkan_cpu_fallback(
      "aten::_upsample_nearest_exact2d",
      "small_cpu_control_fallback",
      {input_arg});
  Tensor result_cpu;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    const Tensor input_cpu = input_arg.is_vulkan() ? input_arg.cpu() : input_arg;
    result_cpu = at::_upsample_nearest_exact2d(
        input_cpu,
        output_sizes,
        scales_h,
        scales_w);
  }
  Tensor result = record_tensor_write_and_return(
      result_cpu.to(input_arg.device()),
      "aten::_upsample_nearest_exact2d",
      "small_cpu_control_fallback",
      {input_arg});
  if (result.is_vulkan()) {
    api::context()->submit_pending_work_and_poll_retire();
  }
  return result;
}

static Tensor upsample_nearest_exact2d(
    const Tensor& input_arg,
    const IntArrayRef output_sizes,
    const std::optional<double> scales_h,
    const std::optional<double> scales_w) {
  utils::log_vulkan_op_hit("aten::_upsample_nearest_exact2d.cpu_fallback");
  return upsample_nearest_exact2d_cpu_fallback(
      input_arg,
      output_sizes,
      scales_h,
      scales_w);
}

static Tensor& upsample_nearest_exact2d_out(
    const Tensor& input_arg,
    const IntArrayRef output_sizes,
    const std::optional<double> scales_h,
    const std::optional<double> scales_w,
    Tensor& out) {
  Tensor result = upsample_nearest_exact2d(
      input_arg,
      output_sizes,
      scales_h,
      scales_w);
  if (out.is_vulkan()) {
    return rebind_vulkan_output(out, result);
  }
  out.copy_(result.cpu());
  return out;
}

static Tensor upsample_bilinear2d(
    const Tensor& input_arg,
    const IntArrayRef output_sizes,
    bool align_corners,
    const std::optional<double> scales_h,
    const std::optional<double> scales_w) {
  api::AllocationScope allocation_scope("upsample_bilinear");
  api::Context* const context = api::context();

  TORCH_CHECK(
      (4 == input_arg.sizes().size()) && (2 == output_sizes.size()),
      "Invalid input!");

  if (should_run_buffer_upsample(input_arg)) {
    return upsample_bilinear2d_buffer_impl(
        input_arg,
        output_sizes,
        align_corners,
        scales_h,
        scales_w,
        nullptr);
  }

  if (should_materialize_texture_bilinear_input_to_buffer(
          input_arg, output_sizes)) {
    utils::log_vulkan_op_hit("aten::upsample_bilinear2d.texture_to_buffer_float");
    return upsample_bilinear2d_buffer_impl(
        utils::ensure_buffer_storage(input_arg),
        output_sizes,
        align_corners,
        scales_h,
        scales_w,
        nullptr);
  }

  const Tensor input = prepare_upsample_texture_input(input_arg);
  const vTensor& v_input = convert(input);

  vTensor v_output{
      context,
      {
          get_dim<Dim4D::Batch>(v_input),
          get_dim<Dim4D::Channel>(v_input),
          output_sizes[Layout::Parameter::height],
          output_sizes[Layout::Parameter::width],
      },
      v_input.dtype(),
  };

  const api::utils::uvec3 output_extents = v_output.extents();
  const struct Block final {
    uvec3 oextents;
    uint32_t padding;
    ivec2 iextents;
    vec2 scale;
  } block{
      v_output.extents(), // oextents
      0u, // padding
      {
          safe_downcast<int32_t>(get_dim<Dim4D::Width>(input_arg) - 1),
          safe_downcast<int32_t>(get_dim<Dim4D::Height>(input_arg) - 1),
      }, // iextents
      {
          compute_scales_value<float>(
              scales_w,
              get_dim<Dim4D::Width>(input_arg),
              get_dim<Dim4D::Width>(v_output)),
          compute_scales_value<float>(
              scales_h,
              get_dim<Dim4D::Height>(input_arg),
              get_dim<Dim4D::Height>(v_output)),
      }, // scale
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  api::ShaderInfo shader_desc;
  if (align_corners) {
    shader_desc = VK_KERNEL(upsample_bilinear2d_align_true);
  } else {
    shader_desc = VK_KERNEL(upsample_bilinear2d_align_false);
  }
  context->submit_compute_job(
      // shader descriptor
      shader_desc,
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      output_extents,
      // local work group size
      adaptive_work_group_size(output_extents),
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  return record_tensor_write_and_return(
      convert(v_output), "aten::upsample_bilinear2d", "texture", {input});
}

static Tensor upsample_bicubic2d(
    const Tensor& input_arg,
    const IntArrayRef output_sizes,
    bool align_corners,
    const std::optional<double> scales_h,
    const std::optional<double> scales_w) {
  api::AllocationScope allocation_scope("upsample_bicubic");
  api::Context* const context = api::context();

  TORCH_CHECK(
      (4 == input_arg.sizes().size()) && (2 == output_sizes.size()),
      "Invalid input!");

  if (should_run_buffer_upsample(input_arg)) {
    utils::log_vulkan_op_hit("aten::upsample_bicubic2d.buffer_float");
    const vTensor& v_input = convert(input_arg);

    vTensor v_output{
        context,
        {
            get_dim<Dim4D::Batch>(v_input),
            get_dim<Dim4D::Channel>(v_input),
            output_sizes[Layout::Parameter::height],
            output_sizes[Layout::Parameter::width],
        },
        v_input.dtype(),
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
    };

    const struct Block final {
      ivec4 info;
      vec4 scale;
    } block{
        {
            safe_downcast<int32_t>(get_dim<Dim4D::Width>(input_arg) - 1),
            safe_downcast<int32_t>(get_dim<Dim4D::Height>(input_arg) - 1),
            safe_downcast<int32_t>(get_dim<Dim4D::Width>(v_output)),
            safe_downcast<int32_t>(get_dim<Dim4D::Height>(v_output)),
        },
        {
            compute_scales_value<float>(
                scales_w,
                get_dim<Dim4D::Width>(input_arg),
                get_dim<Dim4D::Width>(v_output)),
            compute_scales_value<float>(
                scales_h,
                get_dim<Dim4D::Height>(input_arg),
                get_dim<Dim4D::Height>(v_output)),
            0.0f,
            0.0f,
        },
    };

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};
    const api::utils::uvec3 global_size{
        safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
        1u,
        1u,
    };
    api::UniformParamsBuffer out_meta =
        utils::make_buffer_compute_metadata_ubo(context, v_output);
    api::UniformParamsBuffer in_meta =
        utils::make_buffer_compute_metadata_ubo(context, v_input);

    context->submit_compute_job(
        align_corners ? VK_KERNEL(upsample_bicubic2d_buffer_align_true)
                      : VK_KERNEL(upsample_bicubic2d_buffer_align_false),
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

    return record_tensor_write_and_return(
        convert(v_output), "aten::upsample_bicubic2d", "buffer", {input_arg});
  }

  Tensor input = prepare_upsample_texture_input(input_arg);
  const vTensor& v_input = convert(input);

  vTensor v_output{
      context,
      {
          get_dim<Dim4D::Batch>(v_input),
          get_dim<Dim4D::Channel>(v_input),
          output_sizes[Layout::Parameter::height],
          output_sizes[Layout::Parameter::width],
      },
      v_input.dtype(),
  };

  const api::utils::uvec3 output_extents = v_output.extents();
  const struct Block final {
    uvec3 oextents;
    uint32_t padding;
    ivec2 iextents;
    vec2 scale;
  } block{
      v_output.extents(), // oextents
      0u, // padding
      {
          safe_downcast<int32_t>(get_dim<Dim4D::Width>(input_arg) - 1),
          safe_downcast<int32_t>(get_dim<Dim4D::Height>(input_arg) - 1),
      }, // iextents
      {
          compute_scales_value<float>(
              scales_w,
              get_dim<Dim4D::Width>(input_arg),
              get_dim<Dim4D::Width>(v_output)),
          compute_scales_value<float>(
              scales_h,
              get_dim<Dim4D::Height>(input_arg),
              get_dim<Dim4D::Height>(v_output)),
      }, // scale
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  api::ShaderInfo shader_desc;
  if (align_corners) {
    shader_desc = VK_KERNEL(upsample_bicubic2d_align_true);
  } else {
    shader_desc = VK_KERNEL(upsample_bicubic2d_align_false);
  }
  context->submit_compute_job(
      shader_desc,
      pipeline_barrier,
      output_extents,
      adaptive_work_group_size(output_extents),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());

  return record_tensor_write_and_return(
      convert(v_output), "aten::upsample_bicubic2d", "texture", {input});
}

static Tensor& upsample_bicubic2d_out(
    const Tensor& input,
    const IntArrayRef output_sizes,
    bool align_corners,
    const std::optional<double> scales_h,
    const std::optional<double> scales_w,
    Tensor& out) {
  TORCH_CHECK(
      out.is_vulkan(),
      "Vulkan upsample_bicubic2d.out expects a Vulkan output tensor");
  return rebind_vulkan_output(
      out,
      upsample_bicubic2d(
          input,
          output_sizes,
          align_corners,
          scales_h,
          scales_w));
}

Tensor upsample_bilinear2d_buffer_out_vulkan(
    const Tensor& input,
    const IntArrayRef output_sizes,
    bool align_corners,
    const std::optional<double> scales_h,
    const std::optional<double> scales_w,
    Tensor& output) {
  TORCH_CHECK(
      should_run_buffer_upsample(input),
      "Vulkan bilinear upsample out expects float buffer-backed input");
  return upsample_bilinear2d_buffer_impl(
      input,
      output_sizes,
      align_corners,
      scales_h,
      scales_w,
      &output);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_nearest2d"),
      TORCH_FN(upsample_nearest2d));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_upsample_nearest_exact2d"),
      TORCH_FN(upsample_nearest_exact2d));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_upsample_nearest_exact2d.out"),
      TORCH_FN(upsample_nearest_exact2d_out));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_bilinear2d"),
      TORCH_FN(upsample_bilinear2d));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_bicubic2d"),
      TORCH_FN(upsample_bicubic2d));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::upsample_bicubic2d.out"),
      TORCH_FN(upsample_bicubic2d_out));
}

#endif /* USE_VULKAN_API */

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
