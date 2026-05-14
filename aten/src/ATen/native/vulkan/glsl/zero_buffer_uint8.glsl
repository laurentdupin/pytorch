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

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void zero_width_pack_padding(
    const uvec4 write_coord,
    const uint out_buf_length,
    const uint out_storage_offset) {
  const uint logical_channels = uOutMeta.logical_sizes.x;
  const uint physical_channels = uOutMeta.physical_strides.y;
  if (write_coord.x != 0u || logical_channels >= physical_channels) {
    return;
  }

  uvec4 pad_coord = write_coord;
  for (uint c = logical_channels; c < physical_channels; ++c) {
    pad_coord.x = c;
    const uint pad_idx =
        coord_to_idx(pad_coord, uOutMeta.physical_strides) + out_storage_offset;
    if (pad_idx < out_buf_length) {
      uOutput.data[pad_idx] = uint8_t(0);
    }
  }
}

void main() {
  const uint write_idx = ivec3(gl_GlobalInvocationID).x;
  const uint out_numel = uOutMeta.info.y;
  const uint out_buf_length = uOutMeta.info.z;
  const uint out_storage_offset = uOutMeta.info.w;

  if (write_idx >= out_numel) {
    return;
  }

  const uvec4 write_coord =
      idx_to_coord(
          write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);

  const uint actual_write_idx =
      coord_to_idx(write_coord, uOutMeta.physical_strides) + out_storage_offset;
  if (actual_write_idx < out_buf_length) {
    uOutput.data[actual_write_idx] = uint8_t(0);
  }

  zero_width_pack_padding(write_coord, out_buf_length, out_storage_offset);
}
