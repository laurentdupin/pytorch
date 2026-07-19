#version 450 core

#include "indexing.h"

layout(std430) buffer;

layout(set = 0, binding = 0) buffer highp restrict writeonly OutBuffer {
  uint data[];
}
uOutput;

layout(set = 0, binding = 1) uniform highp restrict OutMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uOutMeta;

layout(set = 0, binding = 2) buffer highp restrict readonly LeftBuffer {
  uint data[];
}
uLeft;

layout(set = 0, binding = 3) uniform highp restrict LeftMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uLeftMeta;

layout(set = 0, binding = 4) buffer highp restrict readonly RightBuffer {
  uint data[];
}
uRight;

layout(set = 0, binding = 5) uniform highp restrict RightMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uRightMeta;

layout(set = 0, binding = 6) uniform highp restrict Params {
  uint left_seq;
  uint reserved0;
  uint reserved1;
  uint reserved2;
}
uParams;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

uint read_bfloat16(
    const uint relative_idx,
    const uint element_count,
    const uint storage_offset,
    const bool use_left) {
  if (relative_idx >= element_count) {
    return 0u;
  }
  const uint element_idx = relative_idx + storage_offset;
  const uint word = use_left ? uLeft.data[element_idx >> 1]
                             : uRight.data[element_idx >> 1];
  return ((element_idx & 1u) == 0u) ? (word & 0xFFFFu) : (word >> 16);
}

uint output_raw(const uint physical_idx, const uvec4 physical_sizes) {
  if (physical_idx >= uOutMeta.info.z) {
    return 0u;
  }
  const uvec4 write_coord =
      idx_to_coord(physical_idx, uOutMeta.physical_strides, physical_sizes);
  if (!all(lessThan(write_coord, uOutMeta.logical_sizes))) {
    return 0u;
  }
  if (write_coord.y < uParams.left_seq) {
    const uint read_idx =
        coord_to_idx(write_coord, uLeftMeta.physical_strides);
    return read_bfloat16(
        read_idx, uLeftMeta.info.z, uLeftMeta.info.w, true);
  }
  uvec4 right_coord = write_coord;
  right_coord.y -= uParams.left_seq;
  const uint read_idx =
      coord_to_idx(right_coord, uRightMeta.physical_strides);
  return read_bfloat16(
      read_idx, uRightMeta.info.z, uRightMeta.info.w, false);
}

void main() {
  const uint word_idx = uint(gl_GlobalInvocationID.x);
  const uint first_physical_idx = word_idx << 1;
  if (first_physical_idx >= uOutMeta.info.z) {
    return;
  }
  uvec4 physical_sizes = uOutMeta.logical_sizes;
  const uint outer_size =
      physical_sizes.y * physical_sizes.z * physical_sizes.w;
  physical_sizes.x = uOutMeta.info.z / max(outer_size, 1u);
  const uint low = output_raw(first_physical_idx, physical_sizes);
  const uint high = output_raw(first_physical_idx + 1u, physical_sizes);
  uOutput.data[word_idx] = low | (high << 16);
}
