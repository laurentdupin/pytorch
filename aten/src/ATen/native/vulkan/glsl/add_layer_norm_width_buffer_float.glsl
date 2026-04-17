#version 450 core

#include "indexing.h"

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict writeonly ResidualOutBuffer {
  float data[];
}
uResidualOut;

layout(set = 0, binding = 1) uniform restrict ResidualOutMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uResidualOutMeta;

layout(set = 0, binding = 2) buffer restrict writeonly NormOutBuffer {
  float data[];
}
uNormOut;

layout(set = 0, binding = 3) uniform restrict NormOutMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uNormOutMeta;

layout(set = 0, binding = 4) buffer restrict readonly ResidualBuffer {
  float data[];
}
uResidual;

layout(set = 0, binding = 5) uniform restrict ResidualMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uResidualMeta;

layout(set = 0, binding = 6) buffer restrict readonly AddendBuffer {
  float data[];
}
uAddend;

layout(set = 0, binding = 7) uniform restrict AddendMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uAddendMeta;

layout(set = 0, binding = 8) buffer restrict readonly WeightBuffer {
  float data[];
}
uWeight;

layout(set = 0, binding = 9) uniform restrict WeightMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uWeightMeta;

layout(set = 0, binding = 10) buffer restrict readonly BiasBuffer {
  float data[];
}
uBias;

layout(set = 0, binding = 11) uniform restrict BiasMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uBiasMeta;

layout(set = 0, binding = 12) uniform restrict Params {
  vec4 data;
}
uParams;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint row = gl_GlobalInvocationID.x;
  const uint normalized_size = uResidualMeta.logical_sizes.x;
  if (normalized_size == 0u) {
    return;
  }

  const uint row_count = uNormOutMeta.info.y / normalized_size;
  if (row >= row_count) {
    return;
  }

  const uint h_size = max(uResidualMeta.logical_sizes.y, 1u);
  const uint c_size = max(uResidualMeta.logical_sizes.z, 1u);
  const uint h = row % h_size;
  const uint c = (row / h_size) % c_size;
  const uint n = row / (h_size * c_size);

  const uint residual_buf_length = uResidualMeta.info.z;
  const uint residual_storage_offset = uResidualMeta.info.w;
  const uint addend_buf_length = uAddendMeta.info.z;
  const uint addend_storage_offset = uAddendMeta.info.w;
  const uint residual_out_buf_length = uResidualOutMeta.info.z;
  const uint residual_out_storage_offset = uResidualOutMeta.info.w;
  const uint norm_out_buf_length = uNormOutMeta.info.z;
  const uint norm_out_storage_offset = uNormOutMeta.info.w;
  const uint weight_buf_length = uWeightMeta.info.z;
  const uint weight_storage_offset = uWeightMeta.info.w;
  const uint bias_buf_length = uBiasMeta.info.z;
  const uint bias_storage_offset = uBiasMeta.info.w;

  float sum = 0.0;
  float sumsq = 0.0;
  for (uint x = 0u; x < normalized_size; ++x) {
    const uvec4 coord = uvec4(x, h, c, n);
    const uint residual_idx =
        coord_to_idx(coord, uResidualMeta.physical_strides) +
        residual_storage_offset;
    const uint addend_idx =
        coord_to_idx(coord, uAddendMeta.physical_strides) +
        addend_storage_offset;
    if (residual_idx < residual_buf_length && addend_idx < addend_buf_length) {
      const float value =
          uResidual.data[residual_idx] + uAddend.data[addend_idx];
      sum += value;
      sumsq += value * value;
    }
  }

  const float denom = max(float(normalized_size), 1.0);
  const float mean = sum / denom;
  const float variance = max(sumsq / denom - mean * mean, 0.0);
  const float std_inv = inversesqrt(variance + uParams.data.x);

  for (uint x = 0u; x < normalized_size; ++x) {
    const uvec4 coord = uvec4(x, h, c, n);
    const uint residual_idx =
        coord_to_idx(coord, uResidualMeta.physical_strides) +
        residual_storage_offset;
    const uint addend_idx =
        coord_to_idx(coord, uAddendMeta.physical_strides) +
        addend_storage_offset;
    const uint residual_write_idx =
        coord_to_idx(coord, uResidualOutMeta.physical_strides) +
        residual_out_storage_offset;
    const uint norm_write_idx =
        coord_to_idx(coord, uNormOutMeta.physical_strides) +
        norm_out_storage_offset;
    const uint weight_idx =
        coord_to_idx(uvec4(x, 0u, 0u, 0u), uWeightMeta.physical_strides) +
        weight_storage_offset;
    const uint bias_idx =
        coord_to_idx(uvec4(x, 0u, 0u, 0u), uBiasMeta.physical_strides) +
        bias_storage_offset;
    if (
        residual_idx < residual_buf_length &&
        addend_idx < addend_buf_length &&
        residual_write_idx < residual_out_buf_length &&
        norm_write_idx < norm_out_buf_length &&
        weight_idx < weight_buf_length &&
        bias_idx < bias_buf_length) {
      const float residual_value =
          uResidual.data[residual_idx] + uAddend.data[addend_idx];
      uResidualOut.data[residual_write_idx] = residual_value;
      uNormOut.data[norm_write_idx] =
          (residual_value - mean) * std_inv * uWeight.data[weight_idx] +
          uBias.data[bias_idx];
    }
  }
}
