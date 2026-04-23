#version 450 core
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#define PRECISION ${PRECISION}

#include "indexing.h"

layout(std430) buffer;

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  uint8_t data[];
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
  uint8_t data[];
}
uInput;

layout(set = 0, binding = 3) uniform PRECISION restrict InMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uInMeta;

layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 info;
  vec4 scale;
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
  const ivec2 src_xy = clamp(
      ivec2(vec2(out_coord.xy) * uBlock.scale.xy),
      ivec2(0, 0),
      uBlock.info.xy);
  const uvec4 src_coord =
      uvec4(uint(src_xy.x), uint(src_xy.y), out_coord.z, out_coord.w);

  const uint read_idx =
      coord_to_idx(src_coord, uInMeta.physical_strides) + uInMeta.info.w;
  const uint actual_write_idx =
      coord_to_idx(out_coord, uOutMeta.physical_strides) + uOutMeta.info.w;

  if (actual_write_idx >= uOutMeta.info.z) {
    return;
  }
  if (read_idx >= uInMeta.info.z) {
    uOutput.data[actual_write_idx] = uint8_t(0);
    return;
  }

  uOutput.data[actual_write_idx] = uInput.data[read_idx];
}
