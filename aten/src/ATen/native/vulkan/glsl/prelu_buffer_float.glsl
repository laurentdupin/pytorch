#version 450 core

#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

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

layout(set = 0, binding = 4) buffer PRECISION restrict readonly WeightBuffer {
  float data[];
}
uWeight;

layout(set = 0, binding = 5) uniform PRECISION restrict WeightMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uWeightMeta;

layout(set = 0, binding = 6) uniform PRECISION restrict Params {
  uint weight_is_scalar;
  uint input_dim;
  uint reserved0;
  uint reserved1;
}
uParams;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void zero_width_pack_padding(
    const uvec4 write_coord,
    const uint out_buf_length,
    const uint out_storage_offset) {
  const uint logical_width = uOutMeta.logical_sizes.x;
  const uint physical_width = uOutMeta.physical_strides.y;
  if (write_coord.x != 0u || logical_width >= physical_width) {
    return;
  }

  uvec4 pad_coord = write_coord;
  for (uint w = logical_width; w < physical_width; ++w) {
    pad_coord.x = w;
    const uint pad_idx =
        coord_to_idx(pad_coord, uOutMeta.physical_strides) + out_storage_offset;
    if (pad_idx < out_buf_length) {
      uOutput.data[pad_idx] = 0.0;
    }
  }
}

uint channel_from_coord(const uvec4 coord) {
  if (uParams.input_dim == 2u) {
    return coord.x;
  }
  if (uParams.input_dim == 3u) {
    return coord.y;
  }
  return coord.z;
}

float weight_value(const uvec4 write_coord) {
  if (uParams.weight_is_scalar != 0u) {
    const uint scalar_idx = uWeightMeta.info.w;
    return scalar_idx < uWeightMeta.info.z ? uWeight.data[scalar_idx] : 0.0;
  }

  const uint weight_idx = uWeightMeta.info.w + channel_from_coord(write_coord);
  return weight_idx < uWeightMeta.info.z ? uWeight.data[weight_idx] : 0.0;
}

void main() {
  const uint write_idx = ivec3(gl_GlobalInvocationID).x;
  const uint out_numel = uOutMeta.info.y;
  const uint out_buf_length = uOutMeta.info.z;
  const uint out_storage_offset = uOutMeta.info.w;
  const uint in_buf_length = uInMeta.info.z;
  const uint in_storage_offset = uInMeta.info.w;

  if (write_idx >= out_numel) {
    return;
  }

  const uvec4 write_coord =
      idx_to_coord(write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);

  float input_value = 0.0;
  if (all(lessThan(write_coord, uInMeta.logical_sizes))) {
    const uint read_idx =
        coord_to_idx(write_coord, uInMeta.physical_strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      input_value = uInput.data[read_idx];
    }
  }

  const float slope = weight_value(write_coord);
  const float output_value =
      input_value > 0.0 ? input_value : input_value * slope;

  const uint actual_write_idx =
      coord_to_idx(write_coord, uOutMeta.physical_strides) + out_storage_offset;
  if (actual_write_idx < out_buf_length) {
    uOutput.data[actual_write_idx] = output_value;
  }

  zero_width_pack_padding(write_coord, out_buf_length, out_storage_offset);
}
