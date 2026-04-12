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

layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 info;
  vec4 scale;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float fetch_bounded(ivec2 pos_xy, uint channel, uint batch) {
  const ivec2 clamped_xy = clamp(pos_xy, ivec2(0, 0), uBlock.info.xy);
  const uvec4 coord =
      uvec4(uint(clamped_xy.x), uint(clamped_xy.y), channel, batch);
  const uint read_idx =
      coord_to_idx(coord, uInMeta.physical_strides) + uInMeta.info.w;
  if (read_idx >= uInMeta.info.z) {
    return 0.0;
  }
  return uInput.data[read_idx];
}

void main() {
  const uint write_idx = gl_GlobalInvocationID.x;
  if (write_idx >= uOutMeta.info.y) {
    return;
  }

  const uvec4 out_coord =
      idx_to_coord(write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);
  const vec2 out_minus_one =
      max(vec2(uBlock.info.zw - ivec2(1, 1)), vec2(1.0, 1.0));
  const vec2 src = vec2(out_coord.xy) * vec2(uBlock.info.xy) / out_minus_one;
  const ivec2 base = ivec2(floor(src));
  const ivec2 upper = ivec2(ceil(src));
  const vec2 alpha = src - vec2(base);

  const float top = mix(
      fetch_bounded(base, out_coord.z, out_coord.w),
      fetch_bounded(ivec2(upper.x, base.y), out_coord.z, out_coord.w),
      alpha.x);
  const float bottom = mix(
      fetch_bounded(ivec2(base.x, upper.y), out_coord.z, out_coord.w),
      fetch_bounded(upper, out_coord.z, out_coord.w),
      alpha.x);

  const uint actual_write_idx =
      coord_to_idx(out_coord, uOutMeta.physical_strides) + uOutMeta.info.w;
  if (actual_write_idx < uOutMeta.info.z) {
    uOutput.data[actual_write_idx] = mix(top, bottom, alpha.y);
  }
}
