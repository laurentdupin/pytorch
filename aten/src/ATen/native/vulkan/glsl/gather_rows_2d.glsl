#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/*
 * Output Image
 */
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;

/*
 * Input Image
 */
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;

/*
 * Input Indices
 */
layout(set = 0, binding = 2) buffer PRECISION restrict readonly IndexBuffer {
  int data[];
}
uIndex;

/*
 * Params Buffer
 */
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 out_extents;
  ivec4 index_info;
}
uBlock;

/*
 * Local Work Group Size
 */
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, uBlock.out_extents.xyz))) {
    return;
  }

  const int index_rows = uBlock.index_info.x;
  const int index_cols = uBlock.index_info.y;
  const int index_rank = uBlock.index_info.z;

  if (index_rank == 1) {
    const int row = uIndex.data[pos.y];
    const vec4 texel = texelFetch(uInput, ivec3(pos.x, row, 0), 0);
    imageStore(uOutput, pos, vec4(texel.x, 0.0, 0.0, 0.0));
    return;
  }

  const int batch_base = pos.z * 4;
  const int token = pos.y;

  vec4 gathered = vec4(0.0);
  for (int component = 0; component < 4; ++component) {
    const int batch = batch_base + component;
    if (batch >= index_rows) {
      break;
    }

    const int row = uIndex.data[batch * index_cols + token];
    gathered[component] = texelFetch(uInput, ivec3(pos.x, row, 0), 0).x;
  }

  imageStore(uOutput, pos, gathered);
}
