/*
 * TILE_SIZE = (4, 8, 1)
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
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uOutMeta;

layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer {
  float data[];
}
uInput;

layout(set = 0, binding = 3) uniform PRECISION restrict InMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uInMeta;

layout(set = 0, binding = 4) buffer PRECISION restrict readonly WeightBuffer {
  float data[];
}
uWeight;

layout(set = 0, binding = 5) uniform PRECISION restrict WeightMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uWeightMeta;

layout(set = 0, binding = 6) uniform PRECISION restrict Block {
  ivec4 info;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

const uint TILE_COLS = 4u;
const uint TILE_ROWS = 8u;

void main() {
  const uint out_col_base = gl_GlobalInvocationID.x * TILE_COLS;
  const uint out_row_base = gl_GlobalInvocationID.y * TILE_ROWS;
  const uint out_batch = gl_GlobalInvocationID.z;

  const uint out_width = uint(uBlock.info.x);
  const uint out_height = uint(uBlock.info.y);
  const uint inner_dim = uint(uBlock.info.z);
  const uint batch = uint(uBlock.info.w);

  if (
      out_col_base >= out_width || out_row_base >= out_height ||
      out_batch >= batch) {
    return;
  }

  const uint out_storage_offset = uOutMeta.info.w;
  const uint in_storage_offset = uInMeta.info.w;
  const uint weight_storage_offset = uWeightMeta.info.w;

  float acc[8][4];
  for (uint row = 0u; row < TILE_ROWS; ++row) {
    for (uint col = 0u; col < TILE_COLS; ++col) {
      acc[row][col] = 0.0;
    }
  }

  for (uint k = 0u; k < inner_dim; ++k) {
    float input_values[8];
    float weight_values[4];

    for (uint row = 0u; row < TILE_ROWS; ++row) {
      const uint out_row = out_row_base + row;
      if (out_row < out_height) {
        const uint input_idx = in_storage_offset +
            k * uInMeta.physical_strides.x +
            out_row * uInMeta.physical_strides.y +
            out_batch * uInMeta.physical_strides.z;
        input_values[row] = uInput.data[input_idx];
      } else {
        input_values[row] = 0.0;
      }
    }

    for (uint col = 0u; col < TILE_COLS; ++col) {
      const uint out_col = out_col_base + col;
      if (out_col < out_width) {
        const uint weight_idx = weight_storage_offset +
            out_col * uWeightMeta.physical_strides.x +
            k * uWeightMeta.physical_strides.y +
            out_batch * uWeightMeta.physical_strides.z;
        weight_values[col] = uWeight.data[weight_idx];
      } else {
        weight_values[col] = 0.0;
      }
    }

    for (uint row = 0u; row < TILE_ROWS; ++row) {
      for (uint col = 0u; col < TILE_COLS; ++col) {
        acc[row][col] += input_values[row] * weight_values[col];
      }
    }
  }

  for (uint row = 0u; row < TILE_ROWS; ++row) {
    const uint out_row = out_row_base + row;
    if (out_row >= out_height) {
      continue;
    }
    for (uint col = 0u; col < TILE_COLS; ++col) {
      const uint out_col = out_col_base + col;
      if (out_col >= out_width) {
        continue;
      }
      const uint out_idx = out_storage_offset +
          out_col * uOutMeta.physical_strides.x +
          out_row * uOutMeta.physical_strides.y +
          out_batch * uOutMeta.physical_strides.z;
      uOutput.data[out_idx] = acc[row][col];
    }
  }
}
