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

layout(set = 0, binding = 2) buffer restrict readonly InBuffer {
  float data[];
}
uInput;

layout(set = 0, binding = 3) uniform restrict InMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uInMeta;

layout(set = 0, binding = 4) buffer restrict readonly WeightBuffer {
  float data[];
}
uWeight;

layout(set = 0, binding = 5) uniform restrict WeightMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uWeightMeta;

layout(set = 0, binding = 6) uniform restrict Params {
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
  const uint weight_buf_length = uWeightMeta.info.z;
  const uint weight_storage_offset = uWeightMeta.info.w;

  float sumsq = 0.0;
  for (uint x = 0u; x < normalized_size; ++x) {
    const uvec4 coord = uvec4(x, h, c, n);
    const uint read_idx =
        coord_to_idx(coord, uInMeta.physical_strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      const float value = uInput.data[read_idx];
      sumsq += value * value;
    }
  }

  const float denom = max(float(normalized_size), 1.0);
  const float rstd = inversesqrt(sumsq / denom + uParams.data.x);

  for (uint x = 0u; x < normalized_size; ++x) {
    const uvec4 coord = uvec4(x, h, c, n);
    const uint read_idx =
        coord_to_idx(coord, uInMeta.physical_strides) + in_storage_offset;
    const uint write_idx =
        coord_to_idx(coord, uOutMeta.physical_strides) + out_storage_offset;
    const uint weight_idx =
        coord_to_idx(uvec4(x, 0u, 0u, 0u), uWeightMeta.physical_strides) +
        weight_storage_offset;
    if (
        read_idx < in_buf_length &&
        write_idx < out_buf_length &&
        weight_idx < weight_buf_length) {
      uOutput.data[write_idx] = uInput.data[read_idx] * rstd *
          uWeight.data[weight_idx];
    }
  }
}
