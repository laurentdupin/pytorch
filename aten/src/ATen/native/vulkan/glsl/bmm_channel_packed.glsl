#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define FOUR 4
layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uM1;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uM2;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 shader_extents_and_step;
  ivec4 batch_info;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (all(lessThan(pos, uBlock.shader_extents_and_step.xyz))) {
    float results[FOUR][FOUR][FOUR];
    for (int b = 0; b < FOUR; ++b) {
      for (int row = 0; row < FOUR; ++row) {
        for (int col = 0; col < FOUR; ++col) {
          results[b][row][col] = 0.0;
        }
      }
    }

    const int batch_base = FOUR * pos.z;

    for (int j = 0; j < uBlock.shader_extents_and_step.w; ++j) {
      for (int b = 0; b < FOUR; ++b) {
        const int batch = batch_base + b;
        if (batch >= uBlock.batch_info.x) {
          continue;
        }

        vec4 uM1_partial_rows[FOUR];
        vec4 uM2_partial_cols[FOUR];

        for (int k = 0; k < FOUR; ++k) {
          const int pos_y_offset = (FOUR * pos.y) + k;
          uM1_partial_rows[k] = texelFetch(uM1, ivec3(j, pos_y_offset, batch), 0);
        }

        for (int k = 0; k < FOUR; ++k) {
          const int pos_x_offset = (FOUR * pos.x) + k;
          uM2_partial_cols[k] = texelFetch(uM2, ivec3(pos_x_offset, j, batch), 0);
        }

        for (int row = 0; row < FOUR; ++row) {
          for (int col = 0; col < FOUR; ++col) {
            results[b][row][col] += dot(uM1_partial_rows[row], uM2_partial_cols[col]);
          }
        }
      }
    }

    for (int row = 0; row < FOUR; ++row) {
      for (int col = 0; col < FOUR; ++col) {
        const ivec3 out_pos = ivec3(col + FOUR * pos.x, row + FOUR * pos.y, pos.z);
        imageStore(
            uOutput,
            out_pos,
            vec4(
                results[0][row][col],
                results[1][row][col],
                results[2][row][col],
                results[3][row][col]));
      }
    }
  }
}
