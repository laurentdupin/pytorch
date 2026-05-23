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

layout(set = 0, binding = 2) buffer restrict readonly ConditionBuffer {
  uint8_t data[];
}
uCondition;

layout(set = 0, binding = 3) uniform restrict ConditionMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uConditionMeta;

layout(set = 0, binding = 4) buffer restrict readonly SelfBuffer {
  float data[];
}
uSelf;

layout(set = 0, binding = 5) uniform restrict SelfMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uSelfMeta;

layout(set = 0, binding = 6) buffer restrict readonly OtherBuffer {
  float data[];
}
uOther;

layout(set = 0, binding = 7) uniform restrict OtherMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uOtherMeta;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint write_idx = gl_GlobalInvocationID.x;
  if (write_idx >= uOutMeta.info.y) {
    return;
  }

  const uvec4 write_coord =
      idx_to_coord(write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);

  bool condition_value = false;
  const uint condition_idx =
      coord_to_idx(write_coord, uConditionMeta.physical_strides) +
      uConditionMeta.info.w;
  if (condition_idx < uConditionMeta.info.z) {
    condition_value = uCondition.data[condition_idx] != uint8_t(0);
  }

  float self_value = 0.0;
  const uint self_idx =
      coord_to_idx(write_coord, uSelfMeta.physical_strides) + uSelfMeta.info.w;
  if (self_idx < uSelfMeta.info.z) {
    self_value = uSelf.data[self_idx];
  }

  float other_value = 0.0;
  const uint other_idx =
      coord_to_idx(write_coord, uOtherMeta.physical_strides) + uOtherMeta.info.w;
  if (other_idx < uOtherMeta.info.z) {
    other_value = uOther.data[other_idx];
  }

  const uint actual_write_idx =
      coord_to_idx(write_coord, uOutMeta.physical_strides) + uOutMeta.info.w;
  if (actual_write_idx < uOutMeta.info.z) {
    uOutput.data[actual_write_idx] =
        condition_value ? self_value : other_value;
  }
}
