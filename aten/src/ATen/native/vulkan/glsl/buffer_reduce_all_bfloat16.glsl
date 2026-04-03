#version 450 core

#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

// clang-format off
#define INIT ${INIT}
#define REDUCE(ACC, X) ${REDUCE}
#define FINALIZE(ACC, LEN) ${FINALIZE}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float data[];
}
uOutput;

layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
  uint data[];
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

float bfloat16_to_float_raw(const uint raw) {
  return uintBitsToFloat((raw & 0xFFFFu) << 16);
}

float read_bfloat16(const uint idx) {
  const uint packed = uInput.data[idx >> 1];
  const uint raw = ((idx & 1u) == 0u) ? (packed & 0xFFFFu) : (packed >> 16);
  return bfloat16_to_float_raw(raw);
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
  if (gl_GlobalInvocationID.x != 0u || gl_GlobalInvocationID.y != 0u ||
      gl_GlobalInvocationID.z != 0u) {
    return;
  }

  const uint in_numel = uInMeta.info.y;
  const uint in_buf_length = uInMeta.info.z;
  const uint in_storage_offset = uInMeta.info.w;

  float acc = INIT;
  for (uint logical_idx = 0u; logical_idx < in_numel; ++logical_idx) {
    const uvec4 coord =
        idx_to_coord_4d(
            logical_idx, uInMeta.logical_strides, uInMeta.logical_sizes);
    const uint read_idx =
        coord_to_idx_4d(coord, uInMeta.physical_strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      acc = REDUCE(acc, read_bfloat16(read_idx));
    }
  }

  uOutput.data[0] = FINALIZE(acc, float(in_numel));
}
