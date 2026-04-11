#version 450 core

#define PRECISION ${PRECISION}

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

layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer {
  float data[];
}
uInput;

layout(set = 0, binding = 3) uniform PRECISION restrict InMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uInMeta;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint write_idx = gl_GlobalInvocationID.x;
  const uint out_numel = uOutMeta.info.y;
  const uint out_buf_length = uOutMeta.info.z;
  const uint out_storage_offset = uOutMeta.info.w;
  const uint in_buf_length = uInMeta.info.z;
  const uint in_storage_offset = uInMeta.info.w;

  if (write_idx >= out_numel) {
    return;
  }

  const uvec4 out_coord =
      idx_to_coord(write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);

  const uint width = uInMeta.logical_sizes.x;
  const uint token = out_coord.y;
  const uint h = token / width;
  const uint w = token - h * width;
  const uvec4 input_coord = uvec4(w, h, out_coord.x, out_coord.z);

  float value = 0.0;
  if (all(lessThan(input_coord, uInMeta.logical_sizes))) {
    const uint read_idx =
        coord_to_idx(input_coord, uInMeta.physical_strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      value = uInput.data[read_idx];
    }
  }

  const uint actual_write_idx =
      coord_to_idx(out_coord, uOutMeta.physical_strides) + out_storage_offset;
  if (actual_write_idx < out_buf_length) {
    uOutput.data[actual_write_idx] = value;
  }
}
