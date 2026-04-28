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

layout(set = 0, binding = 4) buffer PRECISION restrict readonly MeanBuffer {
  float data[];
}
uMean;

layout(set = 0, binding = 5) uniform PRECISION restrict MeanMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uMeanMeta;

layout(set = 0, binding = 6) buffer PRECISION restrict readonly StdBuffer {
  float data[];
}
uStd;

layout(set = 0, binding = 7) uniform PRECISION restrict StdMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uStdMeta;

layout(set = 0, binding = 8) uniform PRECISION restrict Block {
  vec4 params;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void zero_width_pack_padding(
    const uvec4 coord,
    const uint out_buf_length,
    const uint out_storage_offset) {
  const uint logical_channels = uOutMeta.logical_sizes.x;
  const uint physical_channels = uOutMeta.physical_strides.y;
  if (coord.x != 0u || logical_channels >= physical_channels) {
    return;
  }

  uvec4 pad_coord = coord;
  for (uint c = logical_channels; c < physical_channels; ++c) {
    pad_coord.x = c;
    const uint pad_idx =
        coord_to_idx(pad_coord, uOutMeta.physical_strides) + out_storage_offset;
    if (pad_idx < out_buf_length) {
      uOutput.data[pad_idx] = 0.0;
    }
  }
}

void main() {
  const uint write_idx = gl_GlobalInvocationID.x;
  if (write_idx >= uOutMeta.info.y) {
    return;
  }

  const uvec4 coord =
      idx_to_coord(write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);
  const uint input_idx =
      coord_to_idx(coord, uInMeta.physical_strides) + uInMeta.info.w;
  const uint output_idx =
      coord_to_idx(coord, uOutMeta.physical_strides) + uOutMeta.info.w;
  if (input_idx >= uInMeta.info.z || output_idx >= uOutMeta.info.z) {
    return;
  }

  const uint channel = coord.x;
  const uint mean_idx =
      channel * uMeanMeta.physical_strides.x + uMeanMeta.info.w;
  const uint std_idx =
      channel * uStdMeta.physical_strides.x + uStdMeta.info.w;
  const float mean_value =
      mean_idx < uMeanMeta.info.z ? uMean.data[mean_idx] : 0.0;
  const float std_value =
      std_idx < uStdMeta.info.z ? uStd.data[std_idx] : 1.0;
  uOutput.data[output_idx] =
      (uInput.data[input_idx] * uBlock.params.x - mean_value) / std_value;

  zero_width_pack_padding(coord, uOutMeta.info.z, uOutMeta.info.w);
}
