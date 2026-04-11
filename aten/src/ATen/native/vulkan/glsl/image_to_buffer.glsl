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

layout(set = 0, binding = 2) uniform PRECISION sampler3D uImage;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint write_idx = gl_GlobalInvocationID.x;
  const uint out_numel = uOutMeta.info.y;
  const uint out_buf_length = uOutMeta.info.z;
  const uint out_storage_offset = uOutMeta.info.w;

  if (write_idx >= out_numel) {
    return;
  }

  const uvec4 coord =
      idx_to_coord(write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);
  const ivec4 nchw_coord = ivec4(
      int(coord.w),
      int(coord.z),
      int(coord.y),
      int(coord.x));
  const ivec4 nchw_sizes = ivec4(
      int(uOutMeta.logical_sizes.w),
      int(uOutMeta.logical_sizes.z),
      int(uOutMeta.logical_sizes.y),
      int(uOutMeta.logical_sizes.x));

  const ivec4 texel_pos =
      get_channel_packed_pos_from_index(nchw_coord, nchw_sizes);
  const vec4 texel = texelFetch(uImage, texel_pos.xyz, 0);
  const float value = texel[texel_pos.w];

  const uint actual_write_idx =
      coord_to_idx(coord, uOutMeta.physical_strides) + out_storage_offset;
  if (actual_write_idx < out_buf_length) {
    uOutput.data[actual_write_idx] = value;
  }
}
