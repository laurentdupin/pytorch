#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define INIT ${INIT}
#define REDUCE(ACC, X) ${REDUCE}
#define FINALIZE(ACC, LEN) ${FINALIZE}
// clang-format on

 #include "indexing.h"

layout(std430) buffer;

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float data[];
}
uOutput;

layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
  float data[];
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

  float acc = INIT;
  for (uint logical_idx = 0u; logical_idx < in_numel; ++logical_idx) {
    const uvec4 coord =
        idx_to_coord(logical_idx, uInMeta.logical_strides, uInMeta.logical_sizes);
    const uint read_idx =
        coord_to_idx(coord, uInMeta.physical_strides) + in_storage_offset;
    if (read_idx < in_buf_length) {
      acc = REDUCE(acc, uInput.data[read_idx]);
    }
  }

  uOutput.data[0] = FINALIZE(acc, float(in_numel));
}
