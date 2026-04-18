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

const uint TILE_K = 8u;
shared float s_input[8][8];
shared float s_weight[8][8];

float gelu_tanh(float value) {
  const float value_cube = value * value * value;
  const float inner = 0.7978845608028654 * (value + 0.044715 * value_cube);
  return 0.5 * value * (1.0 + tanh(inner));
}

void main() {
  const uint out_col = gl_GlobalInvocationID.x;
  const uint out_row = gl_GlobalInvocationID.y;
  const uint local_col = gl_LocalInvocationID.x;
  const uint local_row = gl_LocalInvocationID.y;

  const uint out_width = uint(uBlock.info.x);
  const uint out_height = uint(uBlock.info.y);
  const uint inner_dim = uint(uBlock.info.z);

  const uint out_storage_offset = uOutMeta.info.z;
  const uint in_storage_offset = uInMeta.info.z;
  const uint weight_storage_offset = uWeightMeta.info.z;
  const uint bias_storage_offset = uBiasMeta.info.z;

  const bool valid_output = out_col < out_width && out_row < out_height;
  const bool valid_input_row = out_row < out_height;
  const bool valid_weight_col = out_col < out_width;
  float acc = 0.0;

  for (uint k_base = 0u; k_base < inner_dim; k_base += TILE_K) {
    const uint input_k = k_base + local_col;
    const uint weight_k = k_base + local_row;

    if (valid_input_row && input_k < inner_dim) {
      const uint input_idx = in_storage_offset + input_k * uInMeta.strides.x +
          out_row * uInMeta.strides.y;
      s_input[local_row][local_col] = uInput.data[input_idx];
    } else {
      s_input[local_row][local_col] = 0.0;
    }

    if (valid_weight_col && weight_k < inner_dim) {
      const uint weight_idx = weight_storage_offset +
          out_col * uWeightMeta.strides.x + weight_k * uWeightMeta.strides.y;
      s_weight[local_row][local_col] = uWeight.data[weight_idx];
    } else {
      s_weight[local_row][local_col] = 0.0;
    }

    barrier();

    if (valid_output) {
      for (uint k = 0u; k < TILE_K; ++k) {
        acc += s_input[local_row][k] * s_weight[k][local_col];
      }
    }

    barrier();
  }

  if (!valid_output) {
    return;
  }

  const uint out_idx =
      out_storage_offset + out_col * uOutMeta.strides.x +
      out_row * uOutMeta.strides.y;
  const uint bias_idx = bias_storage_offset + out_col * uBiasMeta.strides.x;
  uOutput.data[out_idx] = gelu_tanh(acc + uBias.data[bias_idx]);
}
