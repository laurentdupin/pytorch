#version 450 core

#include "indexing.h"

layout(std430) buffer;

layout(set = 0, binding = 0) buffer highp restrict writeonly OutBuffer {
  uint data[];
}
uOutput;

layout(set = 0, binding = 1) uniform highp restrict OutMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uOutMeta;

layout(set = 0, binding = 2) buffer highp restrict readonly InBuffer {
  uint data[];
}
uInput;

layout(set = 0, binding = 3) uniform highp restrict InMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uInMeta;

layout(set = 0, binding = 4) uniform highp restrict ScalarBlock {
  float other;
}
uArgs;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float read_bfloat16(const uint element_idx) {
  const uint word = uInput.data[element_idx >> 1];
  const uint raw = ((element_idx & 1u) == 0u) ? (word & 0xFFFFu)
                                              : (word >> 16);
  return uintBitsToFloat(raw << 16);
}

uint float_to_bfloat16(const float value) {
  const uint bits = floatBitsToUint(value);
  const uint lsb = (bits >> 16) & 1u;
  return ((bits + 0x7FFFu + lsb) >> 16) & 0xFFFFu;
}

uint compute_output_raw(const uint physical_idx, const uvec4 physical_sizes) {
  if (physical_idx >= uOutMeta.info.z) {
    return 0u;
  }
  const uvec4 coord =
      idx_to_coord(physical_idx, uOutMeta.physical_strides, physical_sizes);
  if (!all(lessThan(coord, uOutMeta.logical_sizes))) {
    return 0u;
  }
  const uint input_idx =
      coord_to_idx(coord, uInMeta.physical_strides);
  if (input_idx >= uInMeta.info.z) {
    return 0u;
  }
  return float_to_bfloat16(
      read_bfloat16(input_idx + uInMeta.info.w) * uArgs.other);
}

void main() {
  const uint word_idx = uint(gl_GlobalInvocationID.x);
  const uint first_physical_idx = word_idx << 1;
  if (first_physical_idx >= uOutMeta.info.z) {
    return;
  }
  uvec4 physical_sizes = uOutMeta.logical_sizes;
  const uint outer_size =
      physical_sizes.y * physical_sizes.z * physical_sizes.w;
  physical_sizes.x = uOutMeta.info.z / max(outer_size, 1u);
  const uint low = compute_output_raw(first_physical_idx, physical_sizes);
  const uint high =
      compute_output_raw(first_physical_idx + 1u, physical_sizes);
  uOutput.data[word_idx] = low | (high << 16);
}
