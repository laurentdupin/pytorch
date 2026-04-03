#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define DATA_T ${DATA_T}
#define ALPHA_T ${ALPHA_T}
#define OP(X, Y, A) ${OPERATOR}
// clang-format on

#include "indexing.h"

layout(std430) buffer;

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  DATA_T data[];
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
  DATA_T data[];
}
uInput;

layout(set = 0, binding = 3) uniform PRECISION restrict InMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uInMeta;

layout(set = 0, binding = 4) buffer PRECISION restrict readonly OtherBuffer {
  DATA_T data[];
}
uOther;

layout(set = 0, binding = 5) uniform PRECISION restrict OtherMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uOtherMeta;

layout(set = 0, binding = 6) uniform PRECISION restrict TensorBlock {
  ALPHA_T alpha;
}
uArgs;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

uvec4 map_output_coord_to_input_coord(
    const uvec4 out_coord,
    const uvec4 input_sizes) {
  return uvec4(
      input_sizes.x == 1u ? 0u : out_coord.x,
      input_sizes.y == 1u ? 0u : out_coord.y,
      input_sizes.z == 1u ? 0u : out_coord.z,
      input_sizes.w == 1u ? 0u : out_coord.w);
}

void main() {
  const uint write_idx = ivec3(gl_GlobalInvocationID).x;
  const uint out_numel = uOutMeta.info.y;
  const uint out_buf_length = uOutMeta.info.z;
  const uint out_storage_offset = uOutMeta.info.w;
  const uint in_buf_length = uInMeta.info.z;
  const uint in_storage_offset = uInMeta.info.w;
  const uint other_buf_length = uOtherMeta.info.z;
  const uint other_storage_offset = uOtherMeta.info.w;

  if (write_idx >= out_numel) {
    return;
  }

  const uvec4 write_coord =
      idx_to_coord(
          write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);
  const uvec4 input_coord =
      map_output_coord_to_input_coord(write_coord, uInMeta.logical_sizes);
  const uvec4 other_coord =
      map_output_coord_to_input_coord(write_coord, uOtherMeta.logical_sizes);

  DATA_T input_value = DATA_T(0);
  if (all(lessThan(input_coord, uInMeta.logical_sizes))) {
    const uint read_idx =
        coord_to_idx(input_coord, uInMeta.physical_strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      input_value = uInput.data[read_idx];
    }
  }

  DATA_T other_value = DATA_T(0);
  if (all(lessThan(other_coord, uOtherMeta.logical_sizes))) {
    const uint read_idx =
        coord_to_idx(other_coord, uOtherMeta.physical_strides) + other_storage_offset;
    if (read_idx < other_buf_length) {
      other_value = uOther.data[read_idx];
    }
  }

  const uint actual_write_idx =
      coord_to_idx(write_coord, uOutMeta.physical_strides) + out_storage_offset;
  if (actual_write_idx < out_buf_length) {
    uOutput.data[actual_write_idx] = OP(input_value, other_value, uArgs.alpha);
  }
}
