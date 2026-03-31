#version 450 core

#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#include "indexing.h"

layout(std430) buffer;

/*
 * Output Buffer
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float data[];
}
uOutput;

/*
 * Output Buffer Metadata
 */
layout(set = 0, binding = 1) uniform PRECISION restrict OutMeta {
  uvec4 sizes;
  uvec4 strides;
  uvec4 info;
}
uOutMeta;

/*
 * Input Buffer
 */
layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer {
  float data[];
}
uInput;

/*
 * Input Buffer Metadata
 */
layout(set = 0, binding = 3) uniform PRECISION restrict InMeta {
  uvec4 sizes;
  uvec4 strides;
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
  const uint out_buf_length = uOutMeta.info.y;
  const uint out_storage_offset = uOutMeta.info.z;
  const uint in_buf_length = uInMeta.info.y;
  const uint in_storage_offset = uInMeta.info.z;

  if (write_idx >= out_buf_length) {
    return;
  }

  // When the logical layouts match exactly, avoid the coordinate remap path.
  // This is both cheaper and avoids ambiguity when degenerate dimensions make
  // multiple stride values identical, e.g. large [N, C, 1, 1] tensors.
  if (all(equal(uOutMeta.sizes, uInMeta.sizes)) &&
      all(equal(uOutMeta.strides, uInMeta.strides)) &&
      out_storage_offset == in_storage_offset) {
    uOutput.data[write_idx + out_storage_offset] =
        uInput.data[write_idx + in_storage_offset];
    return;
  }

  uvec4 write_coord =
      idx_to_coord(write_idx, uOutMeta.strides, uOutMeta.sizes);

  float outval = 0.0;
  if (all(lessThan(write_coord, uInMeta.sizes))) {
    uint read_idx = coord_to_idx(write_coord, uInMeta.strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      outval = uInput.data[read_idx];
    }
  }

  const uint actual_write_idx = write_idx + out_storage_offset;
  if (actual_write_idx < out_buf_length) {
    uOutput.data[actual_write_idx] = outval;
  }
}
