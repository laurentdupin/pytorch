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

layout(set = 0, binding = 4) buffer restrict readonly WeightBuffer {
  float data[];
}
uWeight;

layout(set = 0, binding = 5) uniform restrict WeightMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uWeightMeta;

layout(set = 0, binding = 6) buffer restrict readonly BiasBuffer {
  float data[];
}
uBias;

layout(set = 0, binding = 7) uniform restrict BiasMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uBiasMeta;

layout(set = 0, binding = 8) buffer restrict readonly MeanBuffer {
  float data[];
}
uMean;

layout(set = 0, binding = 9) uniform restrict MeanMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uMeanMeta;

layout(set = 0, binding = 10) buffer restrict readonly VarBuffer {
  float data[];
}
uVar;

layout(set = 0, binding = 11) uniform restrict VarMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uVarMeta;

layout(set = 0, binding = 12) uniform restrict Block {
  uvec4 info; // channels, has_weight, has_bias, reserved
  float eps;
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
  if (c >= uBlock.info.x) {
    return;
  }

  const uvec4 channel_coord = uvec4(c, 0u, 0u, 0u);
  const uint input_idx =
      coord_to_idx(out_coord, uInMeta.physical_strides) + uInMeta.info.w;
  const uint output_idx =
      coord_to_idx(out_coord, uOutMeta.physical_strides) + uOutMeta.info.w;
  const uint weight_idx =
      coord_to_idx(channel_coord, uWeightMeta.physical_strides) +
      uWeightMeta.info.w;
  const uint bias_idx =
      coord_to_idx(channel_coord, uBiasMeta.physical_strides) + uBiasMeta.info.w;
  const uint mean_idx =
      coord_to_idx(channel_coord, uMeanMeta.physical_strides) + uMeanMeta.info.w;
  const uint var_idx =
      coord_to_idx(channel_coord, uVarMeta.physical_strides) + uVarMeta.info.w;

  if (input_idx >= uInMeta.info.z || output_idx >= uOutMeta.info.z ||
      mean_idx >= uMeanMeta.info.z || var_idx >= uVarMeta.info.z ||
      weight_idx >= uWeightMeta.info.z || bias_idx >= uBiasMeta.info.z) {
    return;
  }

  const float gamma = uBlock.info.y != 0u ? uWeight.data[weight_idx] : 1.0;
  const float beta = uBlock.info.z != 0u ? uBias.data[bias_idx] : 0.0;
  const float normalized =
      (uInput.data[input_idx] - uMean.data[mean_idx]) *
      inversesqrt(uVar.data[var_idx] + uBlock.eps);
  uOutput.data[output_idx] = normalized * gamma + beta;
}
