#version 450 core
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

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

layout(set = 0, binding = 2) buffer restrict readonly MaskBuffer {
  uint8_t data[];
}
uMask;

layout(set = 0, binding = 3) uniform restrict MaskMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uMaskMeta;

layout(set = 0, binding = 4) uniform restrict Block {
  uvec4 info;
  vec4 values;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

uvec4 mask_coord_for_output(const uvec4 out_coord) {
  const uint batch_head = out_coord.z;
  if (uBlock.info.z == 2u) {
    return uvec4(out_coord.x, out_coord.y, 0u, 0u);
  }
  if (uBlock.info.z == 3u) {
    const uint leading = uMaskMeta.logical_sizes.z == 1u
        ? 0u
        : (uBlock.info.w == 4u ? batch_head % uBlock.info.y
                               : batch_head / uBlock.info.y);
    return uvec4(out_coord.x, out_coord.y, leading, 0u);
  }

  const uint mask_head = uMaskMeta.logical_sizes.z == 1u
      ? 0u
      : batch_head % uBlock.info.y;
  const uint mask_batch = uMaskMeta.logical_sizes.w == 1u
      ? 0u
      : batch_head / uBlock.info.y;
  return uvec4(out_coord.x, out_coord.y, mask_head, mask_batch);
}

void main() {
  const uint write_idx = gl_GlobalInvocationID.x;
  if (write_idx >= uOutMeta.info.y) {
    return;
  }

  const uvec4 out_coord =
      idx_to_coord(write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);
  const uvec4 mask_coord = mask_coord_for_output(out_coord);
  const uint mask_idx =
      coord_to_idx(mask_coord, uMaskMeta.physical_strides) + uMaskMeta.info.w;
  const uint actual_write_idx =
      coord_to_idx(out_coord, uOutMeta.physical_strides) + uOutMeta.info.w;
  if (mask_idx < uMaskMeta.info.z && actual_write_idx < uOutMeta.info.z) {
    uOutput.data[actual_write_idx] =
        uMask.data[mask_idx] != uint8_t(0) ? uBlock.values.x : uBlock.values.y;
  }
}
