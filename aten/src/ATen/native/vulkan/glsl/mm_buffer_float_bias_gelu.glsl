/*
 * TILE_SIZE = (4, 4, 1)
 */
#version 450 core

#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

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
  float data[];
}
uInput;

layout(set = 0, binding = 3) uniform PRECISION restrict InMeta {
  uvec4 sizes;
  uvec4 strides;
  uvec4 info;
}
uInMeta;

layout(set = 0, binding = 4) buffer PRECISION restrict readonly WeightBuffer {
  float data[];
}
uWeight;

layout(set = 0, binding = 5) uniform PRECISION restrict WeightMeta {
  uvec4 sizes;
  uvec4 strides;
  uvec4 info;
}
uWeightMeta;

layout(set = 0, binding = 6) buffer PRECISION restrict readonly BiasBuffer {
  float data[];
}
uBias;

layout(set = 0, binding = 7) uniform PRECISION restrict BiasMeta {
  uvec4 sizes;
  uvec4 strides;
  uvec4 info;
}
uBiasMeta;

layout(set = 0, binding = 8) uniform PRECISION restrict Block {
  ivec4 info;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

const uint TILE_SIZE = 4u;

float gelu_tanh(float value) {
  const float value_cube = value * value * value;
  const float inner = 0.7978845608028654 * (value + 0.044715 * value_cube);
  return 0.5 * value * (1.0 + tanh(clamp(inner, -15.0, 15.0)));
}

void main() {
  const uint out_col_base = gl_GlobalInvocationID.x * TILE_SIZE;
  const uint out_row_base = gl_GlobalInvocationID.y * TILE_SIZE;

  const uint out_width = uint(uBlock.info.x);
  const uint out_height = uint(uBlock.info.y);
  const uint inner_dim = uint(uBlock.info.z);

  if (out_col_base >= out_width || out_row_base >= out_height) {
    return;
  }

  const uint out_storage_offset = uOutMeta.info.z;
  const uint in_storage_offset = uInMeta.info.z;
  const uint weight_storage_offset = uWeightMeta.info.z;
  const uint bias_storage_offset = uBiasMeta.info.z;

  float acc[4][4];
  for (uint row = 0u; row < TILE_SIZE; ++row) {
    for (uint col = 0u; col < TILE_SIZE; ++col) {
      acc[row][col] = 0.0;
    }
  }

  for (uint k = 0u; k < inner_dim; ++k) {
    float input_values[4];
    float weight_values[4];

    for (uint row = 0u; row < TILE_SIZE; ++row) {
      const uint out_row = out_row_base + row;
      if (out_row < out_height) {
        const uint input_idx = in_storage_offset + k * uInMeta.strides.x +
            out_row * uInMeta.strides.y;
        input_values[row] = uInput.data[input_idx];
      } else {
        input_values[row] = 0.0;
      }
    }

    for (uint col = 0u; col < TILE_SIZE; ++col) {
      const uint out_col = out_col_base + col;
      if (out_col < out_width) {
        const uint weight_idx = weight_storage_offset +
            out_col * uWeightMeta.strides.x + k * uWeightMeta.strides.y;
        weight_values[col] = uWeight.data[weight_idx];
      } else {
        weight_values[col] = 0.0;
      }
    }

    for (uint row = 0u; row < TILE_SIZE; ++row) {
      for (uint col = 0u; col < TILE_SIZE; ++col) {
        acc[row][col] += input_values[row] * weight_values[col];
      }
    }
  }

  for (uint row = 0u; row < TILE_SIZE; ++row) {
    const uint out_row = out_row_base + row;
    if (out_row >= out_height) {
      continue;
    }
    for (uint col = 0u; col < TILE_SIZE; ++col) {
      const uint out_col = out_col_base + col;
      if (out_col >= out_width) {
        continue;
      }
      const uint out_idx = out_storage_offset + out_col * uOutMeta.strides.x +
          out_row * uOutMeta.strides.y;
      const uint bias_idx =
          bias_storage_offset + out_col * uBiasMeta.strides.x;
      uOutput.data[out_idx] = gelu_tanh(acc[row][col] + uBias.data[bias_idx]);
    }
  }
}
