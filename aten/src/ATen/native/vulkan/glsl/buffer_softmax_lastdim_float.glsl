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

layout(set = 0, binding = 4) uniform restrict Block {
  uvec4 info;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

const uint MAX_LOCAL_SIZE_X = 128u;

shared float sPartials[MAX_LOCAL_SIZE_X];

void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint local_size = gl_WorkGroupSize.x;
  const uint row = gl_WorkGroupID.x + gl_WorkGroupID.y * uBlock.info.y;
  const uint row_count = uBlock.info.x;
  const uint reduce_size = uBlock.info.z;
  if (reduce_size == 0u) {
    return;
  }

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

  float row_max = -3.402823466e+38;
  for (uint x = lane; x < reduce_size; x += local_size) {
    const uvec4 coord = uvec4(x, h, c, n);
    const uint read_idx =
        coord_to_idx(coord, uInMeta.physical_strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      row_max = max(row_max, uInput.data[read_idx]);
    }
  }
  sPartials[lane] = row_max;
  barrier();

  for (uint stride = local_size >> 1u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      sPartials[lane] = max(sPartials[lane], sPartials[lane + stride]);
    }
    barrier();
  }
  row_max = sPartials[0];
  barrier();

  float denom = 0.0;
  for (uint x = lane; x < reduce_size; x += local_size) {
    const uvec4 coord = uvec4(x, h, c, n);
    const uint read_idx =
        coord_to_idx(coord, uInMeta.physical_strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      denom += exp(uInput.data[read_idx] - row_max);
    }
  }
  sPartials[lane] = denom;
  barrier();

  for (uint stride = local_size >> 1u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      sPartials[lane] += sPartials[lane + stride];
    }
    barrier();
  }

  const float inv_denom = 1.0 / max(sPartials[0], 1.0e-20);
  for (uint x = lane; x < reduce_size; x += local_size) {
    const uvec4 coord = uvec4(x, h, c, n);
    const uint read_idx =
        coord_to_idx(coord, uInMeta.physical_strides) + in_storage_offset;
    const uint write_idx =
        coord_to_idx(coord, uOutMeta.physical_strides) + out_storage_offset;
    if (read_idx < in_buf_length && write_idx < out_buf_length) {
      uOutput.data[write_idx] =
          exp(uInput.data[read_idx] - row_max) * inv_denom;
    }
  }
}
