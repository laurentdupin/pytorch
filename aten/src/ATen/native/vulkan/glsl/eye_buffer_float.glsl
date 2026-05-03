#version 450 core

#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#include "indexing.h"

layout(std430) buffer;

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float data[];
}
uOutput;

layout(set = 0, binding = 1) uniform PRECISION restrict OutMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uOutMeta;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint write_idx = ivec3(gl_GlobalInvocationID).x;
  const uint out_numel = uOutMeta.info.y;
  const uint out_buf_length = uOutMeta.info.z;
  const uint out_storage_offset = uOutMeta.info.w;

  if (write_idx >= out_numel) {
    return;
  }

  const uvec4 write_coord =
      idx_to_coord(
          write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);
  const uint actual_write_idx =
      coord_to_idx(write_coord, uOutMeta.physical_strides) +
      out_storage_offset;

  if (actual_write_idx < out_buf_length) {
    uOutput.data[actual_write_idx] =
        write_coord.x == write_coord.y ? 1.0 : 0.0;
  }
}
