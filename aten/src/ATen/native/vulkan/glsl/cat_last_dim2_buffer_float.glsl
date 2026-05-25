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

layout(set = 0, binding = 2) buffer PRECISION restrict readonly LeftBuffer {
  float data[];
}
uLeft;

layout(set = 0, binding = 3) uniform PRECISION restrict LeftMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uLeftMeta;

layout(set = 0, binding = 4) buffer PRECISION restrict readonly RightBuffer {
  float data[];
}
uRight;

layout(set = 0, binding = 5) uniform PRECISION restrict RightMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uRightMeta;

layout(set = 0, binding = 6) uniform PRECISION restrict Params {
  uint left_width;
  uint reserved0;
  uint reserved1;
  uint reserved2;
}
uParams;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint write_idx = gl_GlobalInvocationID.x;
  const uint out_numel = uOutMeta.info.y;
  const uint out_buf_length = uOutMeta.info.z;
  const uint out_storage_offset = uOutMeta.info.w;

  if (write_idx >= out_numel) {
    return;
  }

  const uvec4 write_coord =
      idx_to_coord(write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);

  float value = 0.0;
  if (write_coord.x < uParams.left_width) {
    const uint read_idx =
        coord_to_idx(write_coord, uLeftMeta.physical_strides) + uLeftMeta.info.w;
    if (read_idx < uLeftMeta.info.z) {
      value = uLeft.data[read_idx];
    }
  } else {
    uvec4 right_coord = write_coord;
    right_coord.x -= uParams.left_width;
    const uint read_idx =
        coord_to_idx(right_coord, uRightMeta.physical_strides) + uRightMeta.info.w;
    if (read_idx < uRightMeta.info.z) {
      value = uRight.data[read_idx];
    }
  }

  const uint actual_write_idx =
      coord_to_idx(write_coord, uOutMeta.physical_strides) + out_storage_offset;
  if (actual_write_idx < out_buf_length) {
    uOutput.data[actual_write_idx] = value;
  }
}
