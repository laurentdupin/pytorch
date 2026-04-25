#version 450 core

#include "indexing.h"

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict writeonly MeanBuffer {
  float data[];
}
uMean;

layout(set = 0, binding = 1) uniform restrict MeanMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uMeanMeta;

layout(set = 0, binding = 2) buffer restrict writeonly RstdBuffer {
  float data[];
}
uRstd;

layout(set = 0, binding = 3) uniform restrict RstdMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uRstdMeta;

layout(set = 0, binding = 4) buffer restrict readonly InBuffer {
  float data[];
}
uInput;

layout(set = 0, binding = 5) uniform restrict InMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uInMeta;

layout(set = 0, binding = 6) uniform restrict Block {
  uvec4 info; // row_count, rows_per_grid_x, reduce_size, reserved
  vec4 params; // eps, unused, unused, unused
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

const uint MAX_LOCAL_SIZE_X = 128u;

shared float sPartials[MAX_LOCAL_SIZE_X];

void main() {
  const uint lane = gl_LocalInvocationID.x;
  const uint local_size = gl_WorkGroupSize.x;
  const uint row =
      gl_WorkGroupID.x + gl_WorkGroupID.y * uBlock.info.y;
  const uint row_count = uBlock.info.x;
  const uint reduce_size = uBlock.info.z;
  if (row >= row_count || reduce_size == 0u) {
    return;
  }

  const uint in_buf_length = uInMeta.info.z;
  const uint in_storage_offset = uInMeta.info.w;

  float sum = 0.0;
  for (uint x = lane; x < reduce_size; x += local_size) {
    const uvec4 coord = uvec4(x, row, 0u, 0u);
    const uint read_idx =
        coord_to_idx(coord, uInMeta.physical_strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      sum += uInput.data[read_idx];
    }
  }
  sPartials[lane] = sum;
  barrier();

  for (uint stride = local_size >> 1u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      sPartials[lane] += sPartials[lane + stride];
    }
    barrier();
  }

  const float mean = sPartials[0] / max(float(reduce_size), 1.0);
  barrier();

  float var_sum = 0.0;
  for (uint x = lane; x < reduce_size; x += local_size) {
    const uvec4 coord = uvec4(x, row, 0u, 0u);
    const uint read_idx =
        coord_to_idx(coord, uInMeta.physical_strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      const float centered = uInput.data[read_idx] - mean;
      var_sum += centered * centered;
    }
  }
  sPartials[lane] = var_sum;
  barrier();

  for (uint stride = local_size >> 1u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      sPartials[lane] += sPartials[lane + stride];
    }
    barrier();
  }

  if (lane == 0u) {
    const float variance = sPartials[0] / max(float(reduce_size), 1.0);
    const float rstd = inversesqrt(max(variance, 0.0) + uBlock.params.x);
    const uvec4 stat_coord = uvec4(0u, row, 0u, 0u);
    const uint mean_idx =
        coord_to_idx(stat_coord, uMeanMeta.physical_strides) + uMeanMeta.info.w;
    const uint rstd_idx =
        coord_to_idx(stat_coord, uRstdMeta.physical_strides) + uRstdMeta.info.w;
    if (mean_idx < uMeanMeta.info.z) {
      uMean.data[mean_idx] = mean;
    }
    if (rstd_idx < uRstdMeta.info.z) {
      uRstd.data[rstd_idx] = rstd;
    }
  }
}
