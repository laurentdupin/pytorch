#version 450 core

#include "indexing.h"

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict writeonly OutBuffer {
  float data[];
}
uOutput;

layout(set = 0, binding = 1) uniform restrict OutMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uOutMeta;

layout(set = 0, binding = 2) buffer restrict readonly ResidualBuffer {
  float data[];
}
uResidual;

layout(set = 0, binding = 3) uniform restrict ResidualMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uResidualMeta;

layout(set = 0, binding = 4) buffer restrict readonly AddendBuffer {
  float data[];
}
uAddend;

layout(set = 0, binding = 5) uniform restrict AddendMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uAddendMeta;

layout(set = 0, binding = 6) buffer restrict readonly ScaleBuffer {
  float data[];
}
uScale;

layout(set = 0, binding = 7) uniform restrict ScaleMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uScaleMeta;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint write_idx = ivec3(gl_GlobalInvocationID).x;
  const uint out_numel = uOutMeta.info.y;
  const uint out_buf_length = uOutMeta.info.z;
  const uint out_storage_offset = uOutMeta.info.w;
  const uint residual_buf_length = uResidualMeta.info.z;
  const uint residual_storage_offset = uResidualMeta.info.w;
  const uint addend_buf_length = uAddendMeta.info.z;
  const uint addend_storage_offset = uAddendMeta.info.w;
  const uint scale_buf_length = uScaleMeta.info.z;
  const uint scale_storage_offset = uScaleMeta.info.w;

  if (write_idx >= out_numel) {
    return;
  }

  const uvec4 write_coord =
      idx_to_coord(write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);
  const uint residual_idx =
      coord_to_idx(write_coord, uResidualMeta.physical_strides) +
      residual_storage_offset;
  const uint addend_idx =
      coord_to_idx(write_coord, uAddendMeta.physical_strides) +
      addend_storage_offset;
  const uint scale_idx =
      coord_to_idx(uvec4(write_coord.x, 0u, 0u, 0u), uScaleMeta.physical_strides) +
      scale_storage_offset;
  const uint actual_write_idx =
      coord_to_idx(write_coord, uOutMeta.physical_strides) + out_storage_offset;

  if (
      residual_idx < residual_buf_length &&
      addend_idx < addend_buf_length &&
      scale_idx < scale_buf_length &&
      actual_write_idx < out_buf_length) {
    uOutput.data[actual_write_idx] =
        uResidual.data[residual_idx] +
        uAddend.data[addend_idx] * uScale.data[scale_idx];
  }
}
