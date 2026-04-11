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

layout(set = 0, binding = 2) uniform PRECISION sampler3D uInput;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint write_idx = gl_GlobalInvocationID.x;
  const uint out_numel = uOutMeta.info.y;
  const uint out_buf_length = uOutMeta.info.z;
  const uint out_storage_offset = uOutMeta.info.w;

  if (write_idx >= out_numel) {
    return;
  }

  const uvec4 out_coord =
      idx_to_coord(write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);

  const int width = int(uOutMeta.logical_sizes.x);
  const int channels = int(uOutMeta.logical_sizes.z);
  const int token = int(out_coord.y) * width + int(out_coord.x);
  const int channel = int(out_coord.z);

  const ivec4 input_coord = ivec4(0, 0, token, channel);
  const ivec4 input_sizes =
      ivec4(1, 1, int(uOutMeta.logical_sizes.x * uOutMeta.logical_sizes.y), channels);
  const ivec4 input_pos =
      get_channel_packed_pos_from_index(input_coord, input_sizes);
  const vec4 texel = texelFetch(uInput, input_pos.xyz, 0);
  const float value = texel[input_pos.w];

  const uint actual_write_idx =
      coord_to_idx(out_coord, uOutMeta.physical_strides) + out_storage_offset;
  if (actual_write_idx < out_buf_length) {
    uOutput.data[actual_write_idx] = value;
  }
}
