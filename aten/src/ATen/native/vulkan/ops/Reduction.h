#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <algorithm>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace reduction {

using namespace api::utils;

inline uvec4 make_logical_buffer_sizes(const std::vector<int64_t>& sizes) {
  return api::utils::make_whcn_uvec4(sizes);
}

inline std::vector<int64_t> calc_logical_contiguous_strides(
    const std::vector<int64_t>& sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (int idx = safe_downcast<int>(sizes.size()) - 2; idx >= 0; --idx) {
    strides[idx] = strides[idx + 1] * std::max<int64_t>(sizes[idx + 1], 1);
  }
  return strides;
}

inline uvec4 make_logical_buffer_strides(const std::vector<int64_t>& sizes) {
  return api::utils::make_whcn_uvec4(calc_logical_contiguous_strides(sizes));
}

inline uint32_t to_whcn_dim(const int64_t dim, const int64_t ndim) {
  return safe_downcast<uint32_t>(ndim - 1 - dim);
}

inline std::vector<int64_t> reduced_output_sizes(
    IntArrayRef input_sizes,
    const int64_t dim,
    const bool keepdim) {
  std::vector<int64_t> output_sizes(input_sizes.begin(), input_sizes.end());
  if (keepdim) {
    output_sizes.at(dim) = 1;
  } else {
    output_sizes.erase(output_sizes.begin() + dim);
  }
  return output_sizes;
}

inline std::vector<int64_t> move_dim_to_end_permutation(
    const int64_t ndim,
    const int64_t dim) {
  std::vector<int64_t> permutation;
  permutation.reserve(safe_downcast<size_t>(ndim));
  for (const auto idx : c10::irange(ndim)) {
    if (idx != dim) {
      permutation.push_back(idx);
    }
  }
  permutation.push_back(dim);
  return permutation;
}

inline std::vector<int64_t> inverse_permutation(
    const std::vector<int64_t>& permutation) {
  std::vector<int64_t> inverse(permutation.size(), 0);
  for (const auto idx : c10::irange(permutation.size())) {
    inverse.at(permutation.at(idx)) = safe_downcast<int64_t>(idx);
  }
  return inverse;
}

template <typename T>
inline std::vector<T> apply_permutation(
    const std::vector<T>& values,
    const std::vector<int64_t>& permutation) {
  std::vector<T> output;
  output.reserve(permutation.size());
  for (const auto index : permutation) {
    output.push_back(values.at(index));
  }
  return output;
}

inline Tensor canonicalize_buffer_reduction_input(
    const Tensor& prepared_input,
    const int64_t dim) {
  const vTensor& v_input = convert(prepared_input);
  TORCH_CHECK(
      v_input.storage_type() == api::StorageType::BUFFER,
      "Vulkan buffer reduction canonicalization requires a buffer-backed tensor");

  if (dim == safe_downcast<int64_t>(v_input.sizes().size()) - 1) {
    return prepared_input;
  }

  Tensor canonical = prepared_input;
  const std::vector<int64_t> permutation =
      move_dim_to_end_permutation(
          safe_downcast<int64_t>(v_input.sizes().size()), dim);
  canonical = utils::make_buffer_metadata_view(
      prepared_input,
      apply_permutation(v_input.sizes(), permutation),
      apply_permutation(v_input.logical_strides(), permutation),
      apply_permutation(v_input.physical_strides(), permutation),
      v_input.storage_offset());

  return convert(utils::materialize_to_contiguous_buffer(
      convert(canonical), api::GPUMemoryLayout::TENSOR_WIDTH_PACKED));
}

inline Tensor maybe_wrap_padded_buffer_output(const Tensor& output_arg) {
  const Tensor output = output_arg.is_vulkan() ? output_arg : output_arg.vulkan();
  const vTensor& v_output = convert(output);
  if (
      v_output.storage_type() != api::StorageType::BUFFER ||
      v_output.buffer_length() ==
          api::utils::safe_downcast<int64_t>(v_output.numel())) {
    return output;
  }

  return utils::make_buffer_metadata_view(
      output,
      v_output.sizes(),
      v_output.logical_strides(),
      v_output.gpu_strides(),
      v_output.storage_offset());
}

inline Tensor restore_buffer_reduction_output_layout(
    const Tensor& output_arg,
    IntArrayRef input_sizes,
    const int64_t dim,
    const bool keepdim) {
  const Tensor output = maybe_wrap_padded_buffer_output(output_arg);
  if (
      !keepdim ||
      dim == safe_downcast<int64_t>(input_sizes.size()) - 1) {
    return output;
  }

  const vTensor& v_output = convert(output);
  const std::vector<int64_t> permutation =
      move_dim_to_end_permutation(
          safe_downcast<int64_t>(input_sizes.size()), dim);
  const std::vector<int64_t> inverse = inverse_permutation(permutation);

  return utils::make_buffer_metadata_view(
      output,
      reduced_output_sizes(input_sizes, dim, true),
      apply_permutation(v_output.logical_strides(), inverse),
      apply_permutation(v_output.physical_strides(), inverse),
      v_output.storage_offset());
}

} // namespace reduction
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
