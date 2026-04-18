/*
 * TILE_SIZE = (16, 16, 1)
 */
#version 460 core

#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#extension GL_EXT_bfloat16 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_cooperative_matrix : require

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
  bfloat16_t data[];
}
uInput;

layout(set = 0, binding = 3) uniform PRECISION restrict InMeta {
  uvec4 sizes;
  uvec4 strides;
  uvec4 info;
}
uInMeta;

layout(set = 0, binding = 4) buffer PRECISION restrict readonly WeightBuffer {
  bfloat16_t data[];
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

const uint TILE_M = 16u;
const uint TILE_N = 16u;
const uint TILE_K = 16u;

void main() {
  const uint out_width = uint(uBlock.info.x);
  const uint out_height = uint(uBlock.info.y);
  const uint inner_dim = uint(uBlock.info.z);

  const uint tile_col = gl_WorkGroupID.x * TILE_N;
  const uint tile_row = gl_WorkGroupID.y * TILE_M;

  if (tile_col >= out_width || tile_row >= out_height) {
    return;
  }

  coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc =
      coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);

  const uint out_storage_offset = uOutMeta.info.z;
  const uint in_storage_offset = uInMeta.info.z;
  const uint weight_storage_offset = uWeightMeta.info.z;

  for (uint k_base = 0u; k_base < inner_dim; k_base += TILE_K) {
    coopmat<bfloat16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a;
    coopmat<bfloat16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b;

    const uint input_offset = in_storage_offset + tile_row * uInMeta.strides.y +
        k_base * uInMeta.strides.x;
    coopMatLoad(
        a,
        uInput.data,
        input_offset,
        uInMeta.strides.y,
        gl_CooperativeMatrixLayoutRowMajor);

    const uint weight_offset = weight_storage_offset +
        k_base * uWeightMeta.strides.x + tile_col * uWeightMeta.strides.y;
    coopMatLoad(
        b,
        uWeight.data,
        weight_offset,
        uWeightMeta.strides.y,
        gl_CooperativeMatrixLayoutColumnMajor);

    acc = coopMatMulAdd(a, b, acc);
  }

  const uint output_offset = out_storage_offset + tile_col * uOutMeta.strides.x +
      tile_row * uOutMeta.strides.y;
  coopMatStore(
      acc,
      uOutput.data,
      output_offset,
      uOutMeta.strides.y,
      gl_CooperativeMatrixLayoutRowMajor);
}
