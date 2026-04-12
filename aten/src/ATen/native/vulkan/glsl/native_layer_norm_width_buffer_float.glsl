#version 450 core

#include "indexing.h"

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict writeonly OutBuffer {
  float data[];
}
uOutput;

layout(set = 0, binding = 1) uniform restrict OutMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uOutMeta;

layout(set = 0, binding = 2) buffer restrict writeonly MeanBuffer {
  float data[];
}
uMean;

layout(set = 0, binding = 3) uniform restrict MeanMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uMeanMeta;

layout(set = 0, binding = 4) buffer restrict writeonly StdInvBuffer {
  float data[];
}
uStdInv;

layout(set = 0, binding = 5) uniform restrict StdInvMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uStdInvMeta;

layout(set = 0, binding = 6) buffer restrict readonly InBuffer {
  float data[];
}
uInput;

layout(set = 0, binding = 7) uniform restrict InMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uInMeta;

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
  const uint normalized_size = uInMeta.logical_sizes.x;
  if (normalized_size == 0u) {
    return;
  }

  const uint row_count = uOutMeta.info.y / normalized_size;
  if (row >= row_count) {
    return;
  }

  const uint h_size = max(uInMeta.logical_sizes.y, 1u);
  const uint c_size = max(uInMeta.logical_sizes.z, 1u);
  const uint h = row % h_size;
  const uint c = (row / h_size) % c_size;
  const uint n = row / (h_size * c_size);

  const uint in_buf_length = uInMeta.info.z;
  const uint in_storage_offset = uInMeta.info.w;
  const uint out_buf_length = uOutMeta.info.z;
  const uint out_storage_offset = uOutMeta.info.w;
  const uint mean_buf_length = uMeanMeta.info.z;
  const uint mean_storage_offset = uMeanMeta.info.w;
  const uint std_inv_buf_length = uStdInvMeta.info.z;
  const uint std_inv_storage_offset = uStdInvMeta.info.w;
  const uint weight_buf_length = uWeightMeta.info.z;
  const uint weight_storage_offset = uWeightMeta.info.w;
  const uint bias_buf_length = uBiasMeta.info.z;
  const uint bias_storage_offset = uBiasMeta.info.w;

  float sum = 0.0;
  float sumsq = 0.0;
  for (uint x = 0u; x < normalized_size; ++x) {
    const uvec4 coord = uvec4(x, h, c, n);
    const uint read_idx =
        coord_to_idx(coord, uInMeta.physical_strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      const float value = uInput.data[read_idx];
      sum += value;
      sumsq += value * value;
    }
  }

  const float denom = max(float(normalized_size), 1.0);
  const float mean = sum / denom;
  const float variance = max(sumsq / denom - mean * mean, 0.0);
  const float std_inv = inversesqrt(variance + uParams.data.x);

  const uvec4 stats_coord = uvec4(0u, h, c, n);
  const uint mean_idx =
      coord_to_idx(stats_coord, uMeanMeta.physical_strides) +
      mean_storage_offset;
  const uint std_inv_idx =
      coord_to_idx(stats_coord, uStdInvMeta.physical_strides) +
      std_inv_storage_offset;
  if (mean_idx < mean_buf_length) {
    uMean.data[mean_idx] = mean;
  }
  if (std_inv_idx < std_inv_buf_length) {
    uStdInv.data[std_inv_idx] = std_inv;
  }

  for (uint x = 0u; x < normalized_size; ++x) {
    const uvec4 coord = uvec4(x, h, c, n);
    const uint read_idx =
        coord_to_idx(coord, uInMeta.physical_strides) + in_storage_offset;
    const uint write_idx =
        coord_to_idx(coord, uOutMeta.physical_strides) + out_storage_offset;
    const uint weight_idx =
        coord_to_idx(uvec4(x, 0u, 0u, 0u), uWeightMeta.physical_strides) +
        weight_storage_offset;
    const uint bias_idx =
        coord_to_idx(uvec4(x, 0u, 0u, 0u), uBiasMeta.physical_strides) +
        bias_storage_offset;
    if (
        read_idx < in_buf_length &&
        write_idx < out_buf_length &&
        weight_idx < weight_buf_length &&
        bias_idx < bias_buf_length) {
      uOutput.data[write_idx] =
          (uInput.data[read_idx] - mean) * std_inv *
              uWeight.data[weight_idx] +
          uBias.data[bias_idx];
    }
  }
}
