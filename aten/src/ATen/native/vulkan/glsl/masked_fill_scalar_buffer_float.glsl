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

layout(set = 0, binding = 4) buffer restrict readonly MaskBuffer {
  uint8_t data[];
}
uMask;

layout(set = 0, binding = 5) uniform restrict MaskMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uMaskMeta;

layout(set = 0, binding = 6) uniform restrict Block {
  float value;
  float fill0;
  float fill1;
  float fill2;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

uvec4 map_output_coord_to_input_coord(
    const uvec4 out_coord,
    const uvec4 input_sizes) {
  return uvec4(
      input_sizes.x == 1u ? 0u : out_coord.x,
      input_sizes.y == 1u ? 0u : out_coord.y,
      input_sizes.z == 1u ? 0u : out_coord.z,
      input_sizes.w == 1u ? 0u : out_coord.w);
}

void main() {
  const uint write_idx = gl_GlobalInvocationID.x;
  if (write_idx >= uOutMeta.info.y) {
    return;
  }

  const uvec4 write_coord =
      idx_to_coord(write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);
  const uvec4 input_coord =
      map_output_coord_to_input_coord(write_coord, uInMeta.logical_sizes);
  const uvec4 mask_coord =
      map_output_coord_to_input_coord(write_coord, uMaskMeta.logical_sizes);

  float input_value = 0.0;
  const uint input_idx =
      coord_to_idx(input_coord, uInMeta.physical_strides) + uInMeta.info.w;
  if (input_idx < uInMeta.info.z) {
    input_value = uInput.data[input_idx];
  }

  bool mask_value = false;
  const uint mask_idx =
      coord_to_idx(mask_coord, uMaskMeta.physical_strides) + uMaskMeta.info.w;
  if (mask_idx < uMaskMeta.info.z) {
    mask_value = uMask.data[mask_idx] != uint8_t(0);
  }

  const uint actual_write_idx =
      coord_to_idx(write_coord, uOutMeta.physical_strides) + uOutMeta.info.w;
  if (actual_write_idx < uOutMeta.info.z) {
    uOutput.data[actual_write_idx] = mask_value ? uBlock.value : input_value;
  }
}
