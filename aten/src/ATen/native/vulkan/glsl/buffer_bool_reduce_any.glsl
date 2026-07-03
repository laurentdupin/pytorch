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

layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
  uint8_t data[];
}
uInput;

layout(set = 0, binding = 2) uniform PRECISION restrict InMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uInMeta;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  if (gl_GlobalInvocationID.x != 0u || gl_GlobalInvocationID.y != 0u ||
      gl_GlobalInvocationID.z != 0u) {
    return;
  }

  const uint in_numel = uInMeta.info.y;
  const uint in_buf_length = uInMeta.info.z;
  const uint in_storage_offset = uInMeta.info.w;

  bool acc = false;
  for (uint logical_idx = 0u; logical_idx < in_numel; ++logical_idx) {
    const uint read_idx = logical_idx + in_storage_offset;
    if (read_idx < in_buf_length) {
      acc = acc || (uInput.data[read_idx] != uint8_t(0));
    }
  }

  uOutput.data[0] = acc ? uint8_t(1) : uint8_t(0);
}
