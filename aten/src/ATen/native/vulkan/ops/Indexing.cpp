#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <optional>
#include <tuple>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/index.h>
#include <ATen/ops/index_add.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/scatter.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/topk.h>
#endif

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

bool buffer_allocation_is_host_visible(const vTensor& tensor) {
  return tensor.buffer_uses_host_visible_allocation();
}

Tensor upload_cpu_result_to_vulkan(
    const Tensor& cpu_result,
    IntArrayRef output_sizes,
    const Tensor& prototype) {
  const Tensor reshaped_cpu = cpu_result.reshape(output_sizes).contiguous();
  Tensor output = at::empty(
      output_sizes,
      prototype.options()
          .device(prototype.device())
          .dtype(reshaped_cpu.scalar_type()));
  ops::copy_(output, reshaped_cpu);
  return output;
}

Tensor gather_rows_2d(const Tensor& weight_arg, const Tensor& indices_arg) {
  TORCH_CHECK(weight_arg.is_vulkan(), "Vulkan gather expects weight on Vulkan");
  TORCH_CHECK(
      weight_arg.dim() == 2,
      "Vulkan gather_rows_2d expects a 2D weight tensor");
  Tensor indices_host = indices_arg;
  if (indices_host.is_vulkan()) {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    indices_host = indices_host.cpu();
  }
  TORCH_CHECK(
      indices_host.device().type() == kCPU,
      "Vulkan gather_rows_2d expects CPU or Vulkan indices");
  TORCH_CHECK(
      indices_host.dim() == 1 || indices_host.dim() == 2,
      "Vulkan gather_rows_2d expects 1D or 2D indices");
  TORCH_CHECK(
      indices_host.scalar_type() == kLong || indices_host.scalar_type() == kInt,
      "Vulkan gather_rows_2d expects int32 or int64 indices");

  Tensor weight = weight_arg;
  vTensor v_weight = convert(weight);

  const int64_t row_count = weight_arg.size(0);
  const int64_t row_width = weight_arg.size(1);
  const Tensor indices = indices_host.contiguous();
  const int64_t num_indices = indices.numel();

  api::Context* const context = api::context();
  api::StorageBuffer index_buffer(context, api::kInt, num_indices);
  {
    api::MemoryMap mapping(index_buffer.buffer(), api::MemoryAccessType::WRITE);
    int32_t* const dst = mapping.template data<int32_t>();

    if (indices.scalar_type() == kLong) {
      const int64_t* const src = indices.const_data_ptr<int64_t>();
      for (const auto idx : c10::irange(num_indices)) {
        const int64_t value = src[idx];
        TORCH_CHECK_INDEX(
            value >= 0 && value < row_count,
            "Vulkan gather_rows_2d: index ",
            value,
            " is out of bounds for dimension 0 with size ",
            row_count);
        dst[idx] = safe_downcast<int32_t>(value);
      }
    } else {
      const int32_t* const src = indices.const_data_ptr<int32_t>();
      for (const auto idx : c10::irange(num_indices)) {
        const int64_t value = src[idx];
        TORCH_CHECK_INDEX(
            value >= 0 && value < row_count,
            "Vulkan gather_rows_2d: index ",
            value,
            " is out of bounds for dimension 0 with size ",
            row_count);
        dst[idx] = src[idx];
      }
    }
  }

  std::vector<int64_t> output_sizes = indices.sizes().vec();
  output_sizes.push_back(row_width);

  if (
      v_weight.storage_type() == api::StorageType::BUFFER &&
      buffer_allocation_is_host_visible(v_weight)) {
    Tensor cpu_result =
        weight_arg.cpu().index_select(0, indices.reshape({num_indices}));
    return upload_cpu_result_to_vulkan(
        cpu_result, output_sizes, weight_arg);
  }

  if (
      v_weight.storage_type() == api::StorageType::BUFFER &&
      indices.dim() == 2) {
    // The flat buffer gather path is reliable for 1D index_select-style access,
    // but 2D embedding-style row gathers on large buffer-backed weights still
    // mis-materialize. Keep the embedding path correct by gathering on CPU and
    // moving the selected rows back to Vulkan.
    Tensor cpu_result =
        weight_arg.cpu().index_select(0, indices.reshape({num_indices}));
    return upload_cpu_result_to_vulkan(
        cpu_result, output_sizes, weight_arg);
  }

  if (weight_arg.scalar_type() != kFloat) {
    const bool can_use_nonfloat_texture_gather =
        v_weight.storage_type() != api::StorageType::BUFFER &&
        v_weight.gpu_memory_layout() ==
            api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;
    if (!can_use_nonfloat_texture_gather) {
      // Non-float 2D gathers currently only have a reliable texture gather
      // implementation for channel-packed layouts. Large half/bfloat16
      // embeddings now often stay in buffer-backed or width-packed Vulkan
      // storage to fit the residency budget, so gather them on CPU and move
      // the selected rows back to Vulkan.
      Tensor cpu_result =
          weight_arg.cpu().index_select(0, indices.reshape({num_indices}));
      return upload_cpu_result_to_vulkan(
          cpu_result, output_sizes, weight_arg);
    }
  }

  if (weight_arg.scalar_type() == kFloat && num_indices > 65535) {
    // Large 2D gathers such as BEiT's relative-position-bias lookup still
    // exceed the reliable Vulkan gather envelope on this backend. Materialize
    // the rows on CPU, then move the gathered result back to Vulkan.
    Tensor cpu_result =
        weight_arg.cpu().index_select(0, indices.reshape({num_indices}));
    return upload_cpu_result_to_vulkan(
        cpu_result, output_sizes, weight_arg);
  }

  if (weight_arg.scalar_type() == kFloat) {
    const bool can_use_texture_float_gather =
        v_weight.storage_type() != api::StorageType::BUFFER &&
        v_weight.gpu_memory_layout() ==
            api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;
    if (can_use_texture_float_gather) {
      vTensor v_output{
          context,
          output_sizes,
          convert_dtype(weight_arg.scalar_type()),
      };

      const int64_t index_rows = indices.dim() == 2 ? indices.size(0) : 1;
      const int64_t index_cols =
          indices.dim() == 2 ? indices.size(1) : num_indices;
      const struct Block final {
        ivec4 out_extents;
        ivec4 index_info;
      } block{
          {safe_downcast<int32_t>(v_output.extents().data[0u]),
           safe_downcast<int32_t>(v_output.extents().data[1u]),
           safe_downcast<int32_t>(v_output.extents().data[2u]),
           0},
          {safe_downcast<int32_t>(index_rows),
           safe_downcast<int32_t>(index_cols),
           safe_downcast<int32_t>(indices.dim()),
           0},
      };

      api::UniformParamsBuffer params(context, block);
      api::PipelineBarrier pipeline_barrier{};

      context->submit_compute_job(
          VK_KERNEL(gather_rows_2d),
          pipeline_barrier,
          v_output.extents(),
          adaptive_work_group_size(v_output.extents()),
          VK_NULL_HANDLE,
          v_output.image(
              pipeline_barrier,
              api::PipelineStage::COMPUTE,
              api::MemoryAccessType::WRITE),
          v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
          index_buffer.buffer(),
          params.buffer());

      return convert(v_output);
    }

    if (
        v_weight.storage_type() != api::StorageType::BUFFER ||
        !v_weight.has_direct_buffer_layout() ||
        v_weight.gpu_memory_layout() !=
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED) {
      v_weight = utils::materialize_to_contiguous_buffer(
          v_weight, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
    }

    constexpr int64_t kRowChunk = 4096;
    const int64_t dispatch_rows = std::min(num_indices, kRowChunk);
    const int64_t dispatch_depth = div_up(num_indices, kRowChunk);

    vTensor v_output{
        context,
        output_sizes,
        convert_dtype(weight_arg.scalar_type()),
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
    };

    const struct Block final {
      ivec4 info;
    } block{
        safe_downcast<int32_t>(row_width),
        safe_downcast<int32_t>(num_indices),
        safe_downcast<int32_t>(kRowChunk),
        0,
    };

    api::UniformParamsBuffer params(context, block);
    api::PipelineBarrier pipeline_barrier{};

    context->submit_compute_job(
        VK_KERNEL(gather_rows_2d_buffer),
        pipeline_barrier,
        {safe_downcast<uint32_t>(row_width),
         safe_downcast<uint32_t>(dispatch_rows),
         safe_downcast<uint32_t>(dispatch_depth)},
        adaptive_work_group_size(
            {safe_downcast<uint32_t>(row_width),
             safe_downcast<uint32_t>(dispatch_rows),
             safe_downcast<uint32_t>(dispatch_depth)}),
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        index_buffer.buffer(),
        params.buffer());

    return convert(v_output);
  }

  TORCH_CHECK(
      v_weight.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
      "Vulkan gather_rows_2d expects channel-packed 2D Vulkan weights");

  vTensor v_output{
      context,
      output_sizes,
      convert_dtype(weight_arg.scalar_type()),
  };

  const int64_t index_rows = indices.dim() == 2 ? indices.size(0) : 1;
  const int64_t index_cols = indices.dim() == 2 ? indices.size(1) : num_indices;
  const struct Block final {
    ivec4 out_extents;
    ivec4 index_info;
  } block{
      {safe_downcast<int32_t>(v_output.extents().data[0u]),
       safe_downcast<int32_t>(v_output.extents().data[1u]),
       safe_downcast<int32_t>(v_output.extents().data[2u]),
       0},
      {safe_downcast<int32_t>(index_rows),
       safe_downcast<int32_t>(index_cols),
       safe_downcast<int32_t>(indices.dim()),
       0},
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(gather_rows_2d),
      pipeline_barrier,
      v_output.extents(),
      adaptive_work_group_size(v_output.extents()),
      VK_NULL_HANDLE,
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      index_buffer.buffer(),
      params.buffer());

  return convert(v_output);
}

Tensor index_select(const Tensor& self, int64_t dim, const Tensor& index) {
  api::AllocationScope allocation_scope("index_select");

  TORCH_CHECK(self.dim() == 2, "Vulkan index_select currently supports 2D tensors only");
  const int64_t normalized_dim = maybe_wrap_dim(dim, self.dim());
  TORCH_CHECK(
      normalized_dim == 0,
      "Vulkan index_select currently only supports dim=0 for 2D tensors");
  TORCH_CHECK(
      index.dim() <= 1,
      "index_select(): Index is supposed to be a vector");

  const Tensor flat_index = index.dim() == 0 ? index.reshape({1}) : index;
  return gather_rows_2d(self, flat_index);
}

Tensor embedding(
    const Tensor& weight,
    const Tensor& indices,
    c10::SymInt /*padding_idx*/,
    bool /*scale_grad_by_freq*/,
    bool /*sparse*/) {
  api::AllocationScope allocation_scope("embedding");

  TORCH_CHECK(weight.dim() == 2, "'weight' must be 2-D");
  TORCH_CHECK(
      indices.scalar_type() == kLong || indices.scalar_type() == kInt,
      "embedding(): Expected dtype int32 or int64 for indices");
  TORCH_CHECK(
      indices.dim() == 1 || indices.dim() == 2,
      "Vulkan embedding currently supports 1D or 2D indices");

  return gather_rows_2d(weight, indices);
}

std::tuple<Tensor, Tensor> topk(
    const Tensor& self,
    c10::SymInt k,
    int64_t dim,
    bool largest,
    bool sorted) {
  Tensor values_cpu;
  Tensor indices_cpu;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    std::tie(values_cpu, indices_cpu) =
        at::topk(
            self.detach().cpu(),
            k.guard_int(__FILE__, __LINE__),
            dim,
            largest,
            sorted);
  }
  return std::make_tuple(
      upload_cpu_result_to_vulkan(values_cpu, values_cpu.sizes(), self),
      upload_cpu_result_to_vulkan(indices_cpu, indices_cpu.sizes(), self));
}

std::tuple<Tensor&, Tensor&> topk_out(
    const Tensor& self,
    c10::SymInt k,
    int64_t dim,
    bool largest,
    bool sorted,
    Tensor& values,
    Tensor& indices) {
  auto result = topk(self, k, dim, largest, sorted);
  Tensor result_values = std::get<0>(result);
  Tensor result_indices = std::get<1>(result);

  if (values.is_vulkan()) {
    ops::copy_(values, result_values);
  } else {
    values.copy_(result_values.cpu());
  }
  if (indices.is_vulkan()) {
    ops::copy_(indices, result_indices);
  } else {
    indices.copy_(result_indices.cpu());
  }
  return std::forward_as_tuple(values, indices);
}

Tensor scatter_value(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value) {
  Tensor result_cpu;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    result_cpu =
        at::scatter(self.detach().cpu(), dim, index.detach().cpu(), value);
  }
  return upload_cpu_result_to_vulkan(result_cpu, result_cpu.sizes(), self);
}

Tensor& scatter_value_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value,
    Tensor& out) {
  Tensor result_cpu;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    result_cpu =
        at::scatter(self.detach().cpu(), dim, index.detach().cpu(), value);
  }

  if (out.is_vulkan()) {
    Tensor result =
        upload_cpu_result_to_vulkan(result_cpu, result_cpu.sizes(), self);
    ops::copy_(out, result);
  } else {
    out.copy_(result_cpu);
  }
  return out;
}

std::tuple<Tensor, Tensor> sort_default(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  Tensor values_cpu;
  Tensor indices_cpu;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    std::tie(values_cpu, indices_cpu) =
        at::sort(self.detach().cpu(), dim, descending);
  }
  return std::make_tuple(
      upload_cpu_result_to_vulkan(values_cpu, values_cpu.sizes(), self),
      upload_cpu_result_to_vulkan(indices_cpu, indices_cpu.sizes(), self));
}

std::tuple<Tensor, Tensor> sort_stable(
    const Tensor& self,
    std::optional<bool> stable,
    int64_t dim,
    bool descending) {
  Tensor values_cpu;
  Tensor indices_cpu;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    std::tie(values_cpu, indices_cpu) =
        at::sort(self.detach().cpu(), stable, dim, descending);
  }
  return std::make_tuple(
      upload_cpu_result_to_vulkan(values_cpu, values_cpu.sizes(), self),
      upload_cpu_result_to_vulkan(indices_cpu, indices_cpu.sizes(), self));
}

std::tuple<Tensor&, Tensor&> sort_values_out(
    const Tensor& self,
    int64_t dim,
    bool descending,
    Tensor& values,
    Tensor& indices) {
  auto result = sort_default(self, dim, descending);
  Tensor result_values = std::get<0>(result);
  Tensor result_indices = std::get<1>(result);

  if (values.is_vulkan()) {
    ops::copy_(values, result_values);
  } else {
    values.copy_(result_values.cpu());
  }
  if (indices.is_vulkan()) {
    ops::copy_(indices, result_indices);
  } else {
    indices.copy_(result_indices.cpu());
  }
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor&, Tensor&> sort_values_stable_out(
    const Tensor& self,
    std::optional<bool> stable,
    int64_t dim,
    bool descending,
    Tensor& values,
    Tensor& indices) {
  auto result = sort_stable(self, stable, dim, descending);
  Tensor result_values = std::get<0>(result);
  Tensor result_indices = std::get<1>(result);

  if (values.is_vulkan()) {
    ops::copy_(values, result_values);
  } else {
    values.copy_(result_values.cpu());
  }
  if (indices.is_vulkan()) {
    ops::copy_(indices, result_indices);
  } else {
    indices.copy_(result_indices.cpu());
  }
  return std::forward_as_tuple(values, indices);
}

c10::List<std::optional<Tensor>> materialize_indices_on_cpu(
    const c10::List<std::optional<Tensor>>& indices) {
  c10::List<std::optional<Tensor>> cpu_indices;
  cpu_indices.reserve(indices.size());
  for (const auto i : c10::irange(indices.size())) {
    const auto index = indices.get(i);
    if (index.has_value()) {
      const Tensor& index_tensor = *index;
      const Tensor detached_index = index_tensor.detach();
      cpu_indices.push_back(
          detached_index.is_vulkan() ? detached_index.cpu() : detached_index);
    } else {
      cpu_indices.push_back(std::nullopt);
    }
  }
  return cpu_indices;
}

Tensor index_tensor(
    const Tensor& self,
    const c10::List<std::optional<Tensor>>& indices) {
  Tensor result_cpu;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    const Tensor self_cpu =
        self.is_vulkan() ? self.detach().cpu() : self.detach();
    result_cpu = at::index(self_cpu, materialize_indices_on_cpu(indices));
  }
  return upload_cpu_result_to_vulkan(result_cpu, result_cpu.sizes(), self);
}

Tensor& index_tensor_out(
    const Tensor& self,
    const c10::List<std::optional<Tensor>>& indices,
    Tensor& out) {
  Tensor result = index_tensor(self, indices);
  if (out.is_vulkan()) {
    ops::copy_(out, result);
  } else {
    out.copy_(result.cpu());
  }
  return out;
}

Tensor index_add_default(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha) {
  Tensor result_cpu;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);
    const Tensor self_cpu = self.is_vulkan() ? self.detach().cpu() : self.detach();
    const Tensor index_cpu =
        index.is_vulkan() ? index.detach().cpu() : index.detach();
    const Tensor source_cpu =
        source.is_vulkan() ? source.detach().cpu() : source.detach();
    result_cpu = at::index_add(self_cpu, dim, index_cpu, source_cpu, alpha);
  }
  return upload_cpu_result_to_vulkan(result_cpu, result_cpu.sizes(), self);
}

Tensor& index_add_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha,
    Tensor& out) {
  Tensor result = index_add_default(self, dim, index, source, alpha);
  if (out.is_vulkan()) {
    ops::copy_(out, result);
  } else {
    out.copy_(result.cpu());
  }
  return out;
}

Tensor& index_add_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha) {
  Tensor result = index_add_default(self, dim, index, source, alpha);
  if (self.is_vulkan()) {
    ops::copy_(self, result);
  } else {
    self.copy_(result.cpu());
  }
  return self;
}

Tensor nonzero_vulkan(const Tensor& self) {
  utils::log_vulkan_op_hit("aten::nonzero.cpu_fallback");
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);
  const Tensor self_cpu = self.is_vulkan() ? self.cpu() : self;
  return at::nonzero(self_cpu);
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::index_select"), TORCH_FN(index_select));
  m.impl(TORCH_SELECTIVE_NAME("aten::embedding"), TORCH_FN(embedding));
  m.impl(TORCH_SELECTIVE_NAME("aten::topk"), TORCH_FN(topk));
  m.impl(TORCH_SELECTIVE_NAME("aten::topk.values"), TORCH_FN(topk_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::scatter.value"), TORCH_FN(scatter_value));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::scatter.value_out"),
      TORCH_FN(scatter_value_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::sort"), TORCH_FN(sort_default));
  m.impl(TORCH_SELECTIVE_NAME("aten::sort.stable"), TORCH_FN(sort_stable));
  m.impl(TORCH_SELECTIVE_NAME("aten::sort.values"), TORCH_FN(sort_values_out));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::sort.values_stable"),
      TORCH_FN(sort_values_stable_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::index.Tensor"), TORCH_FN(index_tensor));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::index.Tensor_out"),
      TORCH_FN(index_tensor_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::index_add"), TORCH_FN(index_add_default));
  m.impl(TORCH_SELECTIVE_NAME("aten::index_add.out"), TORCH_FN(index_add_out));
  m.impl(TORCH_SELECTIVE_NAME("aten::index_add_"), TORCH_FN(index_add_));
  m.impl(TORCH_SELECTIVE_NAME("aten::nonzero"), TORCH_FN(nonzero_vulkan));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
