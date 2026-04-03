#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define INIT ${INIT}
#define REDUCE(ACC, X) ${REDUCE}
#define FINALIZE(ACC, LEN) ${FINALIZE}
// clang-format on

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
  uvec4 map_out_sizes;
  uvec4 map_out_strides;
  uvec4 write_out_sizes;
  uvec4 write_out_strides;
  uvec4 info;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint write_idx = uint(gl_GlobalInvocationID.x);
  const uint reduce_dim = uBlock.info.x;
  const uint reduce_size = uBlock.info.y;
  const uint out_numel = uBlock.info.z;
  const uint reduce_offset = uBlock.info.w;

  if (write_idx >= out_numel) {
    return;
  }

  const uint out_buf_length = uOutMeta.info.z;
  const uint out_storage_offset = uOutMeta.info.w;
  const uint in_buf_length = uInMeta.info.z;
  const uint in_storage_offset = uInMeta.info.w;
  const uvec4 map_out_coord =
      idx_to_coord(write_idx, uBlock.map_out_strides, uBlock.map_out_sizes);
  const uvec4 write_out_coord =
      idx_to_coord(write_idx, uBlock.write_out_strides, uBlock.write_out_sizes);

  float acc = INIT;
  for (uint r = 0u; r < reduce_size; ++r) {
    uvec4 in_coord = map_out_coord;
    in_coord[reduce_dim] = reduce_offset + r;
    const uint read_idx =
        coord_to_idx(in_coord, uInMeta.physical_strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      acc = REDUCE(acc, uInput.data[read_idx]);
    }
  }

  const uint actual_write_idx =
      coord_to_idx(write_out_coord, uOutMeta.physical_strides) +
      out_storage_offset;
  if (actual_write_idx < out_buf_length) {
    uOutput.data[actual_write_idx] = FINALIZE(acc, float(reduce_size));
  }
}
