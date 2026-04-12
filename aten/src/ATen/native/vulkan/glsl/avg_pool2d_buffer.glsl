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
  ivec4 kernel;
  ivec4 stride_padding;
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
  const ivec2 ipos =
      ivec2(out_coord.xy) * uBlock.stride_padding.xy - uBlock.stride_padding.zw;
  const ivec2 start = max(ivec2(0), ipos);
  const ivec2 end = min(ipos + uBlock.kernel.xy, uBlock.kernel.zw);

  float sum = 0.0;
  for (int y = start.y; y < end.y; ++y) {
    for (int x = start.x; x < end.x; ++x) {
      const uvec4 in_coord = uvec4(uint(x), uint(y), out_coord.z, out_coord.w);
      const uint read_idx =
          coord_to_idx(in_coord, uInMeta.physical_strides) + uInMeta.info.w;
      if (read_idx < uInMeta.info.z) {
        sum += uInput.data[read_idx];
      }
    }
  }

  const uint actual_write_idx =
      coord_to_idx(out_coord, uOutMeta.physical_strides) + uOutMeta.info.w;
  if (actual_write_idx < uOutMeta.info.z) {
    uOutput.data[actual_write_idx] = sum / float(uBlock.info.x);
  }
}
