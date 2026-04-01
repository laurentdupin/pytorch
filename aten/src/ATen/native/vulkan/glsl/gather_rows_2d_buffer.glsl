#version 450 core

#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/*
 * Output Buffer
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float data[];
}
uOutput;

/*
 * Input Buffer
 */
layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
  float data[];
}
uInput;

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
  ivec4 info;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const int row_width = uBlock.info.x;
  const int num_rows = uBlock.info.y;

  if (pos.x >= row_width || pos.y >= num_rows) {
    return;
  }

  const int src_row = uIndex.data[pos.y];
  const int src_idx = src_row * row_width + pos.x;
  const int dst_idx = pos.y * row_width + pos.x;
  uOutput.data[dst_idx] = uInput.data[src_idx];
}
