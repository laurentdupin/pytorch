#include <algorithm>
#include <ATen/Functions.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Reduction.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/irange.h>
#include <cstring>
#include <sstream>
#include <torch/library.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

constexpr uint32_t kGroupNormStatsLocalSizeX = 128u;
constexpr uint32_t kGroupNormStatsMaxWorkGroupsX = 65535u;
constexpr int64_t kParallelReduceAllMinNumel = 4096;

bool is_gtx_class_runtime_device() {
  const char* const device_name =
      api::context()->adapter_ptr()->physical_device().properties.deviceName;
  return device_name != nullptr && std::strstr(device_name, "GTX") != nullptr;
}

void maybe_sync_after_gtx_large_group_norm(
    api::Context* const context,
    const vTensor& v_output) {
  constexpr size_t kGtxLargeGroupNormSyncBytes = 128u * 1024u * 1024u;
  if (
      is_gtx_class_runtime_device() &&
      v_output.gpu_nbytes() >= kGtxLargeGroupNormSyncBytes) {
    utils::log_vulkan_op_hit("aten::group_norm.gtx_large_buffer_sync");
    context->sync_and_reclaim();
  }
}

Device vulkan_output_device(const Tensor& tensor) {
  return tensor.is_vulkan() ? tensor.device()
                            : Device(at::kVulkan, api::current_device());
}

std::string format_mean_sizes(IntArrayRef sizes) {
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

void append_group_norm_tensor_summary(
    std::ostringstream& stream,
    const char* label,
    const Tensor& tensor) {
  stream << ' ' << label << "_defined=" << (tensor.defined() ? 1 : 0);
  if (!tensor.defined()) {
    return;
  }
  stream << ' ' << label << '=' << format_mean_sizes(tensor.sizes());
  stream << ' ' << label << "_vulkan=" << (tensor.is_vulkan() ? 1 : 0);
  if (tensor.is_vulkan()) {
    const vTensor& v_tensor = convert(tensor);
    stream << ' ' << label
           << "_direct=" << (v_tensor.has_direct_buffer_layout() ? 1 : 0)
           << ' ' << label << "_offset=" << v_tensor.storage_offset()
           << ' ' << label << "_buffer_len="
           << (v_tensor.storage_type() == api::StorageType::BUFFER
                   ? v_tensor.buffer_length()
                   : -1);
  }
}

void log_group_norm_detail(
    const char* stage,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const int64_t num_groups,
    const int64_t group_size) {
  std::ostringstream stream;
  stream << "aten::group_norm.detail"
         << " stage=" << stage
         << " num_groups=" << num_groups
         << " group_size=" << group_size;
  append_group_norm_tensor_summary(stream, "input", input);
  append_group_norm_tensor_summary(stream, "weight", weight);
  append_group_norm_tensor_summary(stream, "bias", bias);
  utils::log_vulkan_op_hit(stream.str());
}

Tensor mean_dim_buffer_chunk(
    const Tensor& prepared_input,
    const std::vector<int64_t>& output_sizes) {
  api::Context* const context = api::context();
  vTensor& v_input = convert(prepared_input);

  vTensor v_output{
      context,
      output_sizes,
      api::kFloat,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);

  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
      1u,
      1u,
  };
  context->submit_compute_job(
      VK_KERNEL(buffer_mean_dim),
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
      in_meta.buffer());

  return convert(v_output);
}

Tensor finalize_bfloat16_mean_output(
    const Tensor& output,
    const std::optional<ScalarType> dtype) {
  const ScalarType target_dtype =
      dtype.has_value() ? *dtype : c10::ScalarType::BFloat16;
  if (target_dtype == c10::ScalarType::Float) {
    return output;
  }
  return utils::cast_vulkan_tensor_dtype(output, target_dtype);
}

Tensor mean_cpu_fallback(
    const Tensor& self_arg,
    const std::optional<ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
  return at::mean(self_cpu, dtype).to(vulkan_output_device(self_arg));
}

Tensor mean_dim_cpu_fallback(
    const Tensor& self_arg,
    int64_t dim,
    bool keepdim,
    const std::optional<ScalarType> dtype) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Tensor self_cpu = self_arg.is_vulkan() ? self_arg.cpu() : self_arg;
  return at::mean(self_cpu, dim, keepdim, dtype).to(vulkan_output_device(self_arg));
}

void check_group_norm_inputs(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const int64_t channels,
    const int64_t num_groups) {
  TORCH_CHECK(
      num_groups > 0, "Expected num_groups to be greater than 0, got ", num_groups);
  TORCH_CHECK(
      input.dim() >= 2,
      "Expected group_norm input to have at least 2 dimensions, got ",
      input.dim());
  TORCH_CHECK(
      channels % num_groups == 0,
      "Expected number of channels in input to be divisible by num_groups, got input of shape ",
      input.sizes(),
      " and num_groups=",
      num_groups);
  TORCH_CHECK(
      !weight.defined() || (weight.dim() == 1 && weight.numel() == channels),
      "Expected weight to be a vector of size equal to the number of channels in input, but got weight of shape ",
      weight.sizes(),
      " and input of shape ",
      input.sizes());
  TORCH_CHECK(
      !bias.defined() || (bias.dim() == 1 && bias.numel() == channels),
      "Expected bias to be a vector of size equal to the number of channels in input, but got bias of shape ",
      bias.sizes(),
      " and input of shape ",
      input.sizes());
}

Tensor maybe_to_vulkan(const Tensor& tensor) {
  return tensor.is_vulkan() ? tensor : tensor.to(vulkan_output_device(tensor));
}

Tensor maybe_to_compute_dtype(
    const Tensor& tensor,
    const ScalarType compute_dtype) {
  if (!tensor.defined()) {
    return tensor;
  }
  Tensor out = maybe_to_vulkan(tensor);
  if (out.scalar_type() != compute_dtype) {
    out = utils::cast_vulkan_tensor_dtype(out, compute_dtype);
  }
  return out;
}

void log_group_norm_fused_skip(
    const char* reason,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const int64_t num_groups,
    const int64_t group_size) {
  std::ostringstream stream;
  stream << "aten::group_norm.buffer_fused_skip"
         << " reason=" << reason
         << " num_groups=" << num_groups
         << " group_size=" << group_size;
  append_group_norm_tensor_summary(stream, "input", input);
  append_group_norm_tensor_summary(stream, "weight", weight);
  append_group_norm_tensor_summary(stream, "bias", bias);
  utils::log_vulkan_op_hit(stream.str());
}

struct GroupNormStats final {
  Tensor mean;
  Tensor rstd;
};

GroupNormStats run_group_norm_stats_buffer(
    const Tensor& reshaped,
    const int64_t row_count,
    const int64_t reduce_size,
    const double eps) {
  api::AllocationScope allocation_scope("group_norm.buffer_fused_stats");
  utils::log_vulkan_op_hit("aten::group_norm.buffer_fused_stats");

  api::Context* const context = api::context();
  vTensor& v_input = convert(reshaped);
  Tensor mean = utils::create_buffer_tensor(
      {1, row_count, 1}, c10::ScalarType::Float);
  Tensor rstd = utils::create_buffer_tensor(
      {1, row_count, 1}, c10::ScalarType::Float);
  vTensor& v_mean = convert(mean);
  vTensor& v_rstd = convert(rstd);

  api::UniformParamsBuffer mean_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_mean);
  api::UniformParamsBuffer rstd_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_rstd);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);

  const uint32_t row_count_u = safe_downcast<uint32_t>(row_count);
  const uint32_t reduce_size_u = safe_downcast<uint32_t>(reduce_size);
  const uint32_t rows_per_grid_x =
      std::min(row_count_u, kGroupNormStatsMaxWorkGroupsX);
  const uint32_t grid_y = api::utils::div_up(row_count_u, rows_per_grid_x);

  const struct Block final {
    uvec4 info;
    vec4 params;
  } block{
      {row_count_u, rows_per_grid_x, reduce_size_u, 0u},
      {static_cast<float>(eps), 0.0f, 0.0f, 0.0f},
  };
  api::UniformParamsBuffer params(context, block);

  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(
          static_cast<uint64_t>(rows_per_grid_x) *
          kGroupNormStatsLocalSizeX),
      grid_y,
      1u,
  };
  context->submit_compute_job(
      VK_KERNEL(buffer_group_norm_stats),
      pipeline_barrier,
      global_size,
      {kGroupNormStatsLocalSizeX, 1u, 1u},
      VK_NULL_HANDLE,
      v_mean.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      mean_meta.buffer(),
      v_rstd.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      rstd_meta.buffer(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      in_meta.buffer(),
      params.buffer());

  return {mean, rstd};
}

Tensor run_group_norm_affine_buffer(
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& weight,
    const Tensor& bias,
    const int64_t num_groups,
    const int64_t channels_per_group) {
  api::AllocationScope allocation_scope("group_norm.buffer_fused_affine");
  utils::log_vulkan_op_hit("aten::group_norm.buffer_fused_affine");

  api::Context* const context = api::context();
  Tensor output =
      utils::create_buffer_tensor(input.sizes(), c10::ScalarType::Float);
  vTensor& v_output = convert(output);
  vTensor& v_input = convert(input);
  vTensor& v_mean = convert(mean);
  vTensor& v_rstd = convert(rstd);
  vTensor& v_weight = convert(weight);
  vTensor& v_bias = convert(bias);

  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);
  api::UniformParamsBuffer mean_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_mean);
  api::UniformParamsBuffer rstd_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_rstd);
  api::UniformParamsBuffer weight_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_weight);
  api::UniformParamsBuffer bias_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_bias);

  const struct Block final {
    uvec4 info;
  } block{{
      safe_downcast<uint32_t>(num_groups),
      safe_downcast<uint32_t>(channels_per_group),
      safe_downcast<uint32_t>(input.size(1)),
      0u,
  }};
  api::UniformParamsBuffer params(context, block);

  api::PipelineBarrier pipeline_barrier{};
  const uvec3 global_size = {
      safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
      1u,
      1u,
  };
  context->submit_compute_job(
      VK_KERNEL(buffer_group_norm_affine),
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
      v_rstd.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      rstd_meta.buffer(),
      v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      weight_meta.buffer(),
      v_bias.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      bias_meta.buffer(),
      params.buffer());

  maybe_sync_after_gtx_large_group_norm(context, v_output);
  return output;
}

std::optional<Tensor> try_group_norm_buffer_fused(
    const Tensor& compute_input_arg,
    const Tensor& compute_weight_arg,
    const Tensor& compute_bias_arg,
    const int64_t num_groups,
    const int64_t group_size,
    const double eps,
    const ScalarType output_dtype) {
  if (compute_input_arg.dim() != 4) {
    log_group_norm_fused_skip(
        "input_rank_not_4",
        compute_input_arg,
        compute_weight_arg,
        compute_bias_arg,
        num_groups,
        group_size);
    return std::nullopt;
  }
  if (!compute_weight_arg.defined() || !compute_bias_arg.defined()) {
    log_group_norm_fused_skip(
        "missing_affine",
        compute_input_arg,
        compute_weight_arg,
        compute_bias_arg,
        num_groups,
        group_size);
    return std::nullopt;
  }
  if (compute_input_arg.size(0) <= 0 || group_size <= 0) {
    log_group_norm_fused_skip(
        "empty_input",
        compute_input_arg,
        compute_weight_arg,
        compute_bias_arg,
        num_groups,
        group_size);
    return std::nullopt;
  }
  const int64_t channels = compute_input_arg.size(1);
  const int64_t channels_per_group = channels / num_groups;
  if (channels_per_group <= 0) {
    log_group_norm_fused_skip(
        "invalid_channels_per_group",
        compute_input_arg,
        compute_weight_arg,
        compute_bias_arg,
        num_groups,
        group_size);
    return std::nullopt;
  }

  Tensor compute_input = utils::ensure_buffer_storage(
      compute_input_arg, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  Tensor compute_weight = utils::ensure_buffer_storage(
      compute_weight_arg, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  Tensor compute_bias = utils::ensure_buffer_storage(
      compute_bias_arg, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  const vTensor& v_input = convert(compute_input);
  const vTensor& v_weight = convert(compute_weight);
  const vTensor& v_bias = convert(compute_bias);
  if (
      v_input.storage_type() != api::StorageType::BUFFER ||
      v_weight.storage_type() != api::StorageType::BUFFER ||
      v_bias.storage_type() != api::StorageType::BUFFER ||
      !utils::supports_buffer_reduction_compute(v_input) ||
      !utils::supports_buffer_elementwise_compute(v_weight) ||
      !utils::supports_buffer_elementwise_compute(v_bias)) {
    log_group_norm_fused_skip(
        "unsupported_buffer_layout",
        compute_input,
        compute_weight,
        compute_bias,
        num_groups,
        group_size);
    return std::nullopt;
  }

  safe_downcast<uint32_t>(compute_input.numel());
  safe_downcast<uint32_t>(compute_input.size(0) * num_groups);
  safe_downcast<uint32_t>(group_size);

  Tensor reshaped =
      compute_input.reshape({1, compute_input.size(0) * num_groups, group_size});
  const GroupNormStats stats = run_group_norm_stats_buffer(
      reshaped, compute_input.size(0) * num_groups, group_size, eps);
  Tensor output = run_group_norm_affine_buffer(
      compute_input,
      stats.mean,
      stats.rstd,
      compute_weight,
      compute_bias,
      num_groups,
      channels_per_group);
  if (output.scalar_type() != output_dtype) {
    output = utils::cast_vulkan_tensor_dtype(output, output_dtype);
  }
  utils::log_vulkan_op_hit("aten::group_norm.buffer_fused");
  return output;
}

Tensor materialize_vulkan_metadata_view_for_cpu_readback(const Tensor& tensor) {
  if (!tensor.is_vulkan()) {
    return tensor;
  }
  const vTensor& v_tensor = convert(tensor);
  if (
      v_tensor.storage_type() == api::StorageType::BUFFER &&
      !v_tensor.has_direct_buffer_layout() &&
      utils::supports_buffer_view_fast_path(v_tensor)) {
    return utils::ensure_buffer_storage(tensor, v_tensor.gpu_memory_layout());
  }
  return tensor;
}

Tensor group_norm_vulkan(
    const Tensor& input_arg,
    int64_t num_groups,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps,
    bool /* cudnn_enabled */) {
  api::AllocationScope allocation_scope("group_norm");
  utils::log_vulkan_op_hit("aten::group_norm");
  Tensor input = maybe_to_vulkan(input_arg).contiguous();
  const Tensor weight = weight_opt.value_or(Tensor());
  const Tensor bias = bias_opt.value_or(Tensor());

  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  check_group_norm_inputs(input, weight, bias, C, num_groups);

  int64_t HxW = 1;
  for (const auto dim : c10::irange(2, input.dim())) {
    HxW *= input.size(dim);
  }
  const int64_t group_size = (C / num_groups) * HxW;
  log_group_norm_detail(
      "entry", input, weight, bias, num_groups, group_size);
  const ScalarType output_dtype = input.scalar_type();
  const ScalarType compute_dtype = c10::ScalarType::Float;

  Tensor compute_input = maybe_to_compute_dtype(input, compute_dtype);
  Tensor compute_weight = maybe_to_compute_dtype(weight, compute_dtype);
  Tensor compute_bias = maybe_to_compute_dtype(bias, compute_dtype);
  log_group_norm_detail(
      "compute_inputs",
      compute_input,
      compute_weight,
      compute_bias,
      num_groups,
      group_size);

  if (std::optional<Tensor> fused = try_group_norm_buffer_fused(
          compute_input,
          compute_weight,
          compute_bias,
          num_groups,
          group_size,
          eps,
          output_dtype)) {
    log_group_norm_detail(
        "buffer_fused",
        *fused,
        compute_weight,
        compute_bias,
        num_groups,
        group_size);
    return *fused;
  }

  Tensor reshaped = compute_input.reshape({1, N * num_groups, N ? group_size : 1});
  log_group_norm_detail(
      "reshaped", reshaped, Tensor(), Tensor(), num_groups, group_size);
  Tensor group_mean =
      at::mean(reshaped, /*dim=*/2, /*keepdim=*/true, c10::ScalarType::Float);
  log_group_norm_detail(
      "mean", group_mean, Tensor(), Tensor(), num_groups, group_size);
  Tensor centered = at::sub(reshaped, group_mean);
  log_group_norm_detail(
      "centered", centered, Tensor(), Tensor(), num_groups, group_size);
  Tensor group_var = at::mean(
      at::mul(centered, centered),
      /*dim=*/2,
      /*keepdim=*/true,
      c10::ScalarType::Float);
  log_group_norm_detail(
      "var", group_var, Tensor(), Tensor(), num_groups, group_size);
  Tensor group_rstd = at::rsqrt(at::add(group_var, eps));
  log_group_norm_detail(
      "rstd", group_rstd, Tensor(), Tensor(), num_groups, group_size);
  Tensor normalized =
      at::mul(centered, group_rstd).reshape(compute_input.sizes());
  log_group_norm_detail(
      "normalized", normalized, Tensor(), Tensor(), num_groups, group_size);

  std::vector<int64_t> affine_param_shape(input.dim(), 1);
  affine_param_shape[1] = C;
  if (compute_weight.defined()) {
    normalized =
        at::mul(normalized, compute_weight.reshape(affine_param_shape));
    log_group_norm_detail(
        "affine_weight", normalized, Tensor(), Tensor(), num_groups, group_size);
  }
  if (compute_bias.defined()) {
    normalized =
        at::add(normalized, compute_bias.reshape(affine_param_shape));
    log_group_norm_detail(
        "affine_bias", normalized, Tensor(), Tensor(), num_groups, group_size);
  }

  if (normalized.scalar_type() != output_dtype) {
    normalized = utils::cast_vulkan_tensor_dtype(normalized, output_dtype);
  }

  return normalized;
}

Tensor group_norm_cpu_fallback(
    const Tensor& input_arg,
    int64_t num_groups,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps,
    bool cudnn_enabled) {
  Tensor cpu_result;
  const Tensor input_for_cpu =
      materialize_vulkan_metadata_view_for_cpu_readback(input_arg);
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    const Tensor input_cpu =
        input_for_cpu.is_vulkan() ? input_for_cpu.cpu() : input_for_cpu;
    const std::optional<Tensor> weight_cpu =
        weight_opt && weight_opt->defined() && weight_opt->is_vulkan()
        ? std::optional<Tensor>(weight_opt->cpu())
        : weight_opt;
    const std::optional<Tensor> bias_cpu =
        bias_opt && bias_opt->defined() && bias_opt->is_vulkan()
        ? std::optional<Tensor>(bias_opt->cpu())
        : bias_opt;
    cpu_result = at::group_norm(
        input_cpu,
        num_groups,
        weight_cpu,
        bias_cpu,
        eps,
        cudnn_enabled);
  }
  Tensor output = cpu_result.to(vulkan_output_device(input_arg));
  if (input_arg.is_vulkan()) {
    const vTensor& v_input = convert(input_arg);
    if (v_input.storage_type() == api::StorageType::BUFFER) {
      output = utils::ensure_buffer_storage(output, v_input.gpu_memory_layout());
    }
  }
  return output;
}

Tensor group_norm_autograd_other(
    c10::DispatchKeySet ks,
    const Tensor& input_arg,
    int64_t num_groups,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    double eps,
    bool cudnn_enabled) {
  (void)ks;
  if (input_arg.is_vulkan()) {
    return group_norm_vulkan(
        input_arg,
        num_groups,
        weight_opt,
        bias_opt,
        eps,
        cudnn_enabled);
  }
  return group_norm_cpu_fallback(
      input_arg,
      num_groups,
      weight_opt,
      bias_opt,
      eps,
      cudnn_enabled);
}

Tensor mean_all_buffer(
    const Tensor& prepared_input_arg,
    const std::optional<ScalarType> dtype) {
  api::AllocationScope allocation_scope("mean.buffer_all");
  api::Context* const context = api::context();

  const ScalarType target_dtype =
      resolve_vulkan_mean_dtype(prepared_input_arg.scalar_type(), dtype);
  Tensor prepared = prepared_input_arg;
  bool is_bfloat16_input = prepared.scalar_type() == c10::ScalarType::BFloat16;
  if (!is_bfloat16_input && prepared.scalar_type() != c10::ScalarType::Float) {
    prepared = utils::cast_vulkan_tensor_dtype(prepared, c10::ScalarType::Float);
  }

  if (is_bfloat16_input) {
    prepared = utils::cast_vulkan_tensor_dtype(prepared, c10::ScalarType::Float);
    is_bfloat16_input = false;
  }
  vTensor& v_input = convert(prepared);

  if (prepared.numel() >= kParallelReduceAllMinNumel) {
    Tensor output =
        at::sum(prepared, c10::ScalarType::Float).div(prepared.numel());
    if (target_dtype != c10::ScalarType::Float) {
      output = utils::cast_vulkan_tensor_dtype(output, target_dtype);
    }
    return output;
  }

  vTensor v_output{
      context,
      {},
      api::kFloat,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  api::UniformParamsBuffer in_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_input);

  api::PipelineBarrier pipeline_barrier{};
  context->submit_compute_job(
      is_bfloat16_input ? VK_KERNEL(buffer_mean_all_bfloat16)
                        : VK_KERNEL(buffer_mean_all),
      pipeline_barrier,
      {1u, 1u, 1u},
      {1u, 1u, 1u},
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::READ),
      in_meta.buffer());

  Tensor output = convert(v_output);
  if (target_dtype != c10::ScalarType::Float) {
    output = utils::cast_vulkan_tensor_dtype(output, target_dtype);
  }
  return output;
}

Tensor mean_dim_buffer(
    const Tensor& prepared_input_arg,
    int64_t dim,
    bool keepdim,
    const std::optional<ScalarType> dtype) {
  api::AllocationScope allocation_scope("mean.buffer_dim");

  const ScalarType target_dtype =
      resolve_vulkan_mean_dtype(prepared_input_arg.scalar_type(), dtype);
  Tensor prepared = prepared_input_arg;
  if (prepared.scalar_type() == c10::ScalarType::BFloat16) {
    prepared = utils::cast_vulkan_tensor_dtype(prepared, c10::ScalarType::Float);
  }

  if (prepared.scalar_type() != c10::ScalarType::Float) {
    prepared = utils::cast_vulkan_tensor_dtype(prepared, c10::ScalarType::Float);
  }

  Tensor canonical = dim == safe_downcast<int64_t>(prepared.dim()) - 1
      ? prepared
      : reduction::canonicalize_buffer_reduction_input(prepared, dim);
  const vTensor& v_input = convert(canonical);
  const std::vector<int64_t> output_sizes =
      reduction::reduced_output_sizes(
          v_input.sizes(),
          safe_downcast<int64_t>(v_input.sizes().size()) - 1,
          keepdim);
  Tensor output = mean_dim_buffer_chunk(canonical, output_sizes);
  output = reduction::restore_buffer_reduction_output_layout(
      output, prepared.sizes(), dim, keepdim);

  if (target_dtype != c10::ScalarType::Float) {
    output = utils::cast_vulkan_tensor_dtype(output, target_dtype);
  }
  return output;
}

Tensor mean_dim(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    const std::optional<ScalarType> dtype) {
  if (self.dim() > 4) {
    return mean_dim_cpu_fallback(self, dim, keepdim, dtype);
  }

  if (self.scalar_type() == c10::ScalarType::BFloat16) {
    return finalize_bfloat16_mean_output(
        at::mean(
            utils::cast_vulkan_tensor_dtype(self, c10::ScalarType::Float),
            dim,
            keepdim,
            c10::ScalarType::Float),
        dtype);
  }

  TORCH_CHECK(
      self.dim() >= 2 && self.dim() <= 4,
      "Vulkan mean_dim supports 2d, 3d, 4d tensors as input!");
  TORCH_CHECK(
      dim >= -self.dim() && dim < self.dim(),
      "Vulkan mean.dim dimension out of range expected to be in range of [",
      -self.dim(),
      ",",
      self.dim() - 1,
      "], but got ",
      dim);

  const auto plan = utils::build_vulkan_execution_plan(
      self, utils::VulkanExecutionPlanKind::ReductionDimInput);
  if (api::uses_buffer_execution(plan.execution_layout)) {
    dim = utils::normalize(dim, self.dim());
    return mean_dim_buffer(
        utils::prepare_vulkan_direct_buffer_execution_tensor(self, plan),
        dim,
        keepdim,
        dtype);
  }

  // Get the global Vulkan context
  api::Context* const context = api::context();

  // Cast the input Tensor to a vTensor
  Tensor input = utils::execute_vulkan_execution_plan(self, plan);
  const vTensor& v_input = convert(input);

  // Normalize dim into range [0, self.dim()]
  dim = utils::normalize(dim, self.dim());

  // Create the output texture
  std::vector<int64_t> output_size = v_input.sizes();
  uint32_t dim_size = output_size[dim];
  if (keepdim) {
    output_size[dim] = 1;
  } else {
    output_size.erase(output_size.begin() + dim);
  }

  const ScalarType type = resolve_vulkan_mean_dtype(self.scalar_type(), dtype);

  vTensor v_output{
      context,
      output_size,
      convert_dtype(type),
  };

  // Required to determine how to insert memory barriers in the command buffer
  api::PipelineBarrier pipeline_barrier{};

  // Shift dim into 4d range
  if (self.dim() < 4) {
    dim += (4 - self.dim());
  }

  // Create the params buffer
  const struct Block final {
    uvec2 dim_info;
    int32_t channel;
  } block{
      {static_cast<uint32_t>(dim), dim_size},
      static_cast<int32_t>(get_dim<Dim4D::Channel>(v_input)),
  };

  api::UniformParamsBuffer params(context, block);

  context->submit_compute_job(
      // shader descriptor
      keepdim ? VK_KERNEL(mean_dim_keepdim) : VK_KERNEL(mean_dim),
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
  return convert(v_output);
}

Tensor mean_dim_IntList(
    const at::Tensor& self,
    const OptionalIntArrayRef opt_dim,
    bool keepdim,
    const std::optional<ScalarType> dtype) {
  if (
      !self.is_vulkan() ||
      (!is_vulkan_float_dtype(self.scalar_type()) &&
       self.scalar_type() != c10::ScalarType::BFloat16)) {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    const Tensor self_cpu = self.is_vulkan() ? self.cpu() : self;
    return at::mean(self_cpu, opt_dim, keepdim, dtype).vulkan();
  }

  TORCH_CHECK(
      opt_dim.has_value(), "Vulkan mean without a dim arg is not implemented");

  std::set<int64_t> dims_set;

  if (opt_dim.has_value()) {
    auto dims = opt_dim.value();
    for (const auto& d : dims) {
      TORCH_CHECK(
          d >= -self.dim() && d < self.dim(),
          "Vulkan mean.dim_IntList dimension out of range expected to be in range of [",
          -self.dim(),
          ",",
          self.dim() - 1,
          "], but got ",
          d);
      int64_t dim_normalized = utils::normalize(d, self.dim());
      if (dims_set.find(dim_normalized) != dims_set.end()) {
        TORCH_CHECK(
            false,
            "dim ",
            dim_normalized,
            " appears multiple times in the list of dims")
      }
      dims_set.insert(dim_normalized);
    }
    Tensor output = self;
    for (auto it = dims_set.rbegin(); it != dims_set.rend(); ++it) {
      output = mean_dim(output, *it, keepdim, dtype);
    }
    return output;
  }
  return self;
}

Tensor mean(const Tensor& self, const std::optional<ScalarType> dtype) {
  if (self.scalar_type() == c10::ScalarType::BFloat16) {
    return finalize_bfloat16_mean_output(
        at::mean(
            utils::cast_vulkan_tensor_dtype(self, c10::ScalarType::Float),
            c10::ScalarType::Float),
        dtype);
  }

  const auto plan = utils::build_vulkan_execution_plan(
      self, utils::VulkanExecutionPlanKind::ReductionAllInput);
  if (api::uses_buffer_execution(plan.execution_layout)) {
    return mean_all_buffer(
        utils::prepare_vulkan_direct_buffer_execution_tensor(self, plan), dtype);
  }

  return mean_cpu_fallback(self, dtype);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::mean.dim"), TORCH_FN(mean_dim_IntList));
  m.impl(TORCH_SELECTIVE_NAME("aten::mean"), TORCH_FN(mean));
  m.impl(TORCH_SELECTIVE_NAME("aten::group_norm"), TORCH_FN(group_norm_vulkan));
}

TORCH_LIBRARY_IMPL(aten, AutogradOther, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::group_norm"),
      TORCH_FN(group_norm_autograd_other));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
