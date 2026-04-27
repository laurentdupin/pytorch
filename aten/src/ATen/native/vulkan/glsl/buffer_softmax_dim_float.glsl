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

void main() {
  const uint write_linear_idx = gl_GlobalInvocationID.x;
  const uint out_numel = uBlock.info.x;
  const uint reduce_axis = uBlock.info.y;
  const uint reduce_size = uBlock.info.z;
  if (write_linear_idx >= out_numel || reduce_axis >= 4u || reduce_size == 0u) {
    return;
  }

  const uint in_buf_length = uInMeta.info.z;
  const uint in_storage_offset = uInMeta.info.w;
  const uint out_buf_length = uOutMeta.info.z;
  const uint out_storage_offset = uOutMeta.info.w;

  const uvec4 write_coord = idx_to_coord(
      write_linear_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);

  float row_max = -3.402823466e+38;
  for (uint r = 0u; r < reduce_size; ++r) {
    uvec4 reduce_coord = write_coord;
    reduce_coord[reduce_axis] = r;
    const uint read_idx =
        coord_to_idx(reduce_coord, uInMeta.physical_strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      row_max = max(row_max, uInput.data[read_idx]);
    }
  }

  float denom = 0.0;
  for (uint r = 0u; r < reduce_size; ++r) {
    uvec4 reduce_coord = write_coord;
    reduce_coord[reduce_axis] = r;
    const uint read_idx =
        coord_to_idx(reduce_coord, uInMeta.physical_strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      denom += exp(uInput.data[read_idx] - row_max);
    }
  }

  const uint read_idx =
      coord_to_idx(write_coord, uInMeta.physical_strides) + in_storage_offset;
  const uint write_idx =
      coord_to_idx(write_coord, uOutMeta.physical_strides) + out_storage_offset;
  if (read_idx < in_buf_length && write_idx < out_buf_length) {
    uOutput.data[write_idx] =
        exp(uInput.data[read_idx] - row_max) / max(denom, 1.0e-20);
  }
}
