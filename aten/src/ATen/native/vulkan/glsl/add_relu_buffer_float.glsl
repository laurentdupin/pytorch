#version 450 core

#include "indexing.h"

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict writeonly AddOutBuffer {
  float data[];
}
uAddOut;

layout(set = 0, binding = 1) uniform restrict AddOutMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uAddOutMeta;

layout(set = 0, binding = 2) buffer restrict writeonly ReluOutBuffer {
  float data[];
}
uReluOut;

layout(set = 0, binding = 3) uniform restrict ReluOutMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uReluOutMeta;

layout(set = 0, binding = 4) buffer restrict readonly InBuffer {
  float data[];
}
uInput;

layout(set = 0, binding = 5) uniform restrict InMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uInMeta;

layout(set = 0, binding = 6) buffer restrict readonly OtherBuffer {
  float data[];
}
uOther;

layout(set = 0, binding = 7) uniform restrict OtherMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uOtherMeta;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint write_idx = gl_GlobalInvocationID.x;
  if (write_idx >= uAddOutMeta.info.y) {
    return;
  }

  const uvec4 coord = idx_to_coord(
      write_idx, uAddOutMeta.logical_strides, uAddOutMeta.logical_sizes);
  const uint add_write_idx =
      coord_to_idx(coord, uAddOutMeta.physical_strides) + uAddOutMeta.info.w;
  const uint relu_write_idx =
      coord_to_idx(coord, uReluOutMeta.physical_strides) + uReluOutMeta.info.w;
  const uint input_idx =
      coord_to_idx(coord, uInMeta.physical_strides) + uInMeta.info.w;
  const uint other_idx =
      coord_to_idx(coord, uOtherMeta.physical_strides) + uOtherMeta.info.w;

  if (
      add_write_idx < uAddOutMeta.info.z &&
      relu_write_idx < uReluOutMeta.info.z &&
      input_idx < uInMeta.info.z &&
      other_idx < uOtherMeta.info.z) {
    const float value = uInput.data[input_idx] + uOther.data[other_idx];
    uAddOut.data[add_write_idx] = value;
    uReluOut.data[relu_write_idx] = max(value, 0.0);
  }
}
