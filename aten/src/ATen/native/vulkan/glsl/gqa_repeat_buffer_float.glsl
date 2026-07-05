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

layout(set = 0, binding = 2) buffer restrict readonly InputBuffer {
  float data[];
}
uInput;

layout(set = 0, binding = 3) uniform restrict InputMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uInputMeta;

layout(set = 0, binding = 4) uniform restrict Block {
  ivec4 sizes;
  ivec4 repeat_info;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint write_idx = gl_GlobalInvocationID.x;
  if (write_idx >= uOutMeta.info.y) {
    return;
  }

  const uvec4 out_coord =
      idx_to_coord(write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);
  const uint repeat_factor = uint(uBlock.repeat_info.x);
  const uvec4 input_coord = uvec4(
      out_coord.x,
      out_coord.y / repeat_factor,
      out_coord.z,
      out_coord.w);

  const uint input_idx =
      coord_to_idx(input_coord, uInputMeta.physical_strides) + uInputMeta.info.w;
  const uint actual_write_idx =
      coord_to_idx(out_coord, uOutMeta.physical_strides) + uOutMeta.info.w;
  if (input_idx < uInputMeta.info.z && actual_write_idx < uOutMeta.info.z) {
    uOutput.data[actual_write_idx] = uInput.data[input_idx];
  }
}
