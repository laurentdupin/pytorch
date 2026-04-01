#version 450 core

#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float data[];
}
uOutput;

layout(set = 0, binding = 1) uniform PRECISION restrict OutMeta {
  uvec4 sizes;
  uvec4 strides;
  uvec4 info;
}
uOutMeta;

layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer {
  uint16_t data[];
}
uInput;

layout(set = 0, binding = 3) uniform PRECISION restrict InMeta {
  uvec4 sizes;
  uvec4 strides;
  uvec4 info;
}
uInMeta;

layout(set = 0, binding = 4) buffer PRECISION restrict readonly WeightBuffer {
  uint16_t data[];
}
uWeight;

layout(set = 0, binding = 5) uniform PRECISION restrict WeightMeta {
  uvec4 sizes;
  uvec4 strides;
  uvec4 info;
}
uWeightMeta;

layout(set = 0, binding = 6) uniform PRECISION restrict Block {
  ivec4 info;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float bfloat16_to_float(uint16_t raw) {
  return uintBitsToFloat(uint(raw) << 16);
}

void main() {
  const uint out_col = gl_GlobalInvocationID.x;
  const uint out_row = gl_GlobalInvocationID.y;

  const uint out_width = uint(uBlock.info.x);
  const uint out_height = uint(uBlock.info.y);
  const uint inner_dim = uint(uBlock.info.z);

  if (out_col >= out_width || out_row >= out_height) {
    return;
  }

  const uint out_storage_offset = uOutMeta.info.z;
  const uint in_storage_offset = uInMeta.info.z;
  const uint weight_storage_offset = uWeightMeta.info.z;

  const uint out_idx = out_storage_offset + out_col * uOutMeta.strides.x +
      out_row * uOutMeta.strides.y;

  float acc = 0.0;
  for (uint k = 0u; k < inner_dim; ++k) {
    const uint input_idx = in_storage_offset + k * uInMeta.strides.x +
        out_row * uInMeta.strides.y;
    const uint weight_idx = weight_storage_offset + k * uWeightMeta.strides.x +
        out_col * uWeightMeta.strides.y;
    acc += bfloat16_to_float(uInput.data[input_idx]) *
        bfloat16_to_float(uWeight.data[weight_idx]);
  }

  uOutput.data[out_idx] = acc;
}
