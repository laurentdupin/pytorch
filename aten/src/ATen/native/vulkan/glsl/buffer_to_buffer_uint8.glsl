#version 450 core
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#define PRECISION ${PRECISION}

#include "indexing.h"

layout(std430) buffer;

/*
 * Output Buffer
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  uint8_t data[];
}
uOutput;

/*
 * Output Buffer Metadata
 */
layout(set = 0, binding = 1) uniform PRECISION restrict OutMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uOutMeta;

/*
 * Input Buffer
 */
layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer {
  uint8_t data[];
}
uInput;

/*
 * Input Buffer Metadata
 */
layout(set = 0, binding = 3) uniform PRECISION restrict InMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uInMeta;

/*
 * Local Work Group Size
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Copies data from the tensor at uInput to the tensor at uOutput based on 4D
 * coordinate. Each element at (x,y,c,n) in uInput will be copied to uOutput at
 * (x,y,c,n). If (x,y,c,n) is outside the bounds of uInput then 0 will be
 * written.
 *
 * Each shader invocation is responsible for one element of the output buffer.
 */
void main() {
  const uint write_idx = ivec3(gl_GlobalInvocationID).x;
  const uint out_numel = uOutMeta.info.y;
  const uint out_buf_length = uOutMeta.info.z;
  const uint out_storage_offset = uOutMeta.info.w;
  const uint in_buf_length = uInMeta.info.z;
  const uint in_storage_offset = uInMeta.info.w;

  if (write_idx >= out_numel) {
    return;
  }

  uvec4 write_coord =
      idx_to_coord(
          write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);

  uint8_t outval = uint8_t(0);
  if (all(lessThan(write_coord, uInMeta.logical_sizes))) {
    uint read_idx =
        coord_to_idx(write_coord, uInMeta.physical_strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      outval = uInput.data[read_idx];
    }
  }

  const uint actual_write_idx =
      coord_to_idx(write_coord, uOutMeta.physical_strides) + out_storage_offset;
  if (actual_write_idx < out_buf_length) {
    uOutput.data[actual_write_idx] = outval;
  }
}
