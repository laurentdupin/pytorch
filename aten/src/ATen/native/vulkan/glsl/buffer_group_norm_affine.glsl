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

layout(set = 0, binding = 2) buffer restrict readonly InBuffer {
  float data[];
}
uInput;

layout(set = 0, binding = 3) uniform restrict InMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uInMeta;

layout(set = 0, binding = 4) buffer restrict readonly MeanBuffer {
  float data[];
}
uMean;

layout(set = 0, binding = 5) uniform restrict MeanMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uMeanMeta;

layout(set = 0, binding = 6) buffer restrict readonly RstdBuffer {
  float data[];
}
uRstd;

layout(set = 0, binding = 7) uniform restrict RstdMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uRstdMeta;

layout(set = 0, binding = 8) buffer restrict readonly WeightBuffer {
  float data[];
}
uWeight;

layout(set = 0, binding = 9) uniform restrict WeightMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uWeightMeta;

layout(set = 0, binding = 10) buffer restrict readonly BiasBuffer {
  float data[];
}
uBias;

layout(set = 0, binding = 11) uniform restrict BiasMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uBiasMeta;

layout(set = 0, binding = 12) uniform restrict Block {
  uvec4 info; // num_groups, channels_per_group, channels, reserved
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint write_idx = uint(gl_GlobalInvocationID.x);
  const uint out_numel = uOutMeta.info.y;
  if (write_idx >= out_numel) {
    return;
  }

  const uvec4 out_coord =
      idx_to_coord(write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);
  const uint c = out_coord.z;
  const uint n = out_coord.w;
  if (c >= uBlock.info.z) {
    return;
  }

  const uint group =
      n * uBlock.info.x + c / max(uBlock.info.y, 1u);
  const uvec4 stat_coord = uvec4(0u, group, 0u, 0u);
  const uvec4 channel_coord = uvec4(c, 0u, 0u, 0u);

  const uint input_idx =
      coord_to_idx(out_coord, uInMeta.physical_strides) + uInMeta.info.w;
  const uint mean_idx =
      coord_to_idx(stat_coord, uMeanMeta.physical_strides) + uMeanMeta.info.w;
  const uint rstd_idx =
      coord_to_idx(stat_coord, uRstdMeta.physical_strides) + uRstdMeta.info.w;
  const uint weight_idx =
      coord_to_idx(channel_coord, uWeightMeta.physical_strides) +
      uWeightMeta.info.w;
  const uint bias_idx =
      coord_to_idx(channel_coord, uBiasMeta.physical_strides) + uBiasMeta.info.w;
  const uint output_idx =
      coord_to_idx(out_coord, uOutMeta.physical_strides) + uOutMeta.info.w;

  if (input_idx >= uInMeta.info.z || mean_idx >= uMeanMeta.info.z ||
      rstd_idx >= uRstdMeta.info.z || weight_idx >= uWeightMeta.info.z ||
      bias_idx >= uBiasMeta.info.z || output_idx >= uOutMeta.info.z) {
    return;
  }

  const float normalized =
      (uInput.data[input_idx] - uMean.data[mean_idx]) * uRstd.data[rstd_idx];
  uOutput.data[output_idx] = normalized * uWeight.data[weight_idx] +
      uBias.data[bias_idx];
}
