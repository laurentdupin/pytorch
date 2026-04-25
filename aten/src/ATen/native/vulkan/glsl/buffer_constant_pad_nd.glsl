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
  ivec4 pad_before; // W, H, C, N
  vec4 values; // constant pad value, unused, unused, unused
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint write_idx = uint(gl_GlobalInvocationID.x);
  const uint out_numel = uOutMeta.info.y;
  if (write_idx >= out_numel) {
    return;
  }

  const uvec4 out_coord =
      idx_to_coord(write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);
  const ivec4 in_coord_signed = ivec4(out_coord) - uBlock.pad_before;
  const bool in_bounds = all(greaterThanEqual(in_coord_signed, ivec4(0))) &&
      all(lessThan(uvec4(in_coord_signed), uInMeta.logical_sizes));

  float out_value = uBlock.values.x;
  if (in_bounds) {
    const uint read_idx =
        coord_to_idx(uvec4(in_coord_signed), uInMeta.physical_strides) +
        uInMeta.info.w;
    if (read_idx < uInMeta.info.z) {
      out_value = uInput.data[read_idx];
    }
  }

  const uint output_idx =
      coord_to_idx(out_coord, uOutMeta.physical_strides) + uOutMeta.info.w;
  if (output_idx < uOutMeta.info.z) {
    uOutput.data[output_idx] = out_value;
  }
}
