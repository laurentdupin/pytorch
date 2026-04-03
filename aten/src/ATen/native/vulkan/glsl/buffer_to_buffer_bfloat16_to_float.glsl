#version 450 core

#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

layout(std430) buffer;

/*
 * Output Buffer
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float data[];
}
uOutput;

/*
 * Output Buffer Metadata
 */
layout(set = 0, binding = 1) uniform PRECISION restrict OutMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uOutMeta;

/*
 * Input Buffer
 */
layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer {
  uint16_t data[];
}
uInput;

/*
 * Input Buffer Metadata
 */
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

uvec4 idx_to_coord_4d(const uint idx, const uvec4 strides, const uvec4 sizes) {
  return uvec4(
      (idx / strides.x) % sizes.x,
      (idx / strides.y) % sizes.y,
      (idx / strides.z) % sizes.z,
      (idx / strides.w) % sizes.w);
}

uint coord_to_idx_4d(const uvec4 coord, const uvec4 strides) {
  return coord.x * strides.x + coord.y * strides.y + coord.z * strides.z +
      coord.w * strides.w;
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

  uvec4 write_coord =
      idx_to_coord_4d(
          write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);

  float outval = 0.0;
  if (all(lessThan(write_coord, uInMeta.logical_sizes))) {
    uint read_idx =
        coord_to_idx_4d(write_coord, uInMeta.physical_strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      outval = bfloat16_to_float(uInput.data[read_idx]);
    }
  }

  const uint actual_write_idx =
      coord_to_idx_4d(write_coord, uOutMeta.physical_strides) + out_storage_offset;
  if (actual_write_idx < out_buf_length) {
    uOutput.data[actual_write_idx] = outval;
  }
}
