#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define INPUT_T ${INPUT_T}
#define OUTPUT_T ${OUTPUT_T}
#define CONVERT(X) ${CONVERT}
// clang-format on

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

#include "indexing.h"

layout(std430) buffer;

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  OUTPUT_T data[];
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
  INPUT_T data[];
}
uInput;

layout(set = 0, binding = 3) uniform PRECISION restrict InMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uInMeta;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float bfloat16_to_float(uint16_t raw) {
  return uintBitsToFloat(uint(raw) << 16);
}

uint16_t float_to_bfloat16(float value) {
  const uint bits = floatBitsToUint(value);
  const uint lsb = (bits >> 16) & 1u;
  const uint rounding_bias = 0x7FFFu + lsb;
  return uint16_t((bits + rounding_bias) >> 16);
}

void main() {
  const uint write_idx = uint(gl_GlobalInvocationID.x);
  const uint out_numel = uOutMeta.info.y;
  const uint out_buf_length = uOutMeta.info.z;
  const uint out_storage_offset = uOutMeta.info.w;
  const uint in_buf_length = uInMeta.info.z;
  const uint in_storage_offset = uInMeta.info.w;

  if (write_idx >= out_numel) {
    return;
  }

  const uvec4 write_coord =
      idx_to_coord(
          write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);

  OUTPUT_T outval = OUTPUT_T(0);
  if (all(lessThan(write_coord, uInMeta.logical_sizes))) {
    const uint read_idx =
        coord_to_idx(write_coord, uInMeta.physical_strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      outval = CONVERT(uInput.data[read_idx]);
    }
  }

  const uint actual_write_idx =
      coord_to_idx(write_coord, uOutMeta.physical_strides) + out_storage_offset;
  if (actual_write_idx < out_buf_length) {
    uOutput.data[actual_write_idx] = outval;
  }
}
