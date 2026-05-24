#version 450 core

#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float data[];
}
uOutput;

layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
  float data[];
}
uInput;

layout(set = 0, binding = 2) buffer PRECISION restrict readonly IndexBuffer {
  int data[];
}
uIndex;

layout(set = 0, binding = 3) buffer PRECISION restrict readonly SrcBuffer {
  float data[];
}
uSrc;

layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 info;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const int batch = pos.y;
  const int col = pos.x;
  const int input_cols = uBlock.info.x;
  const int scatter_cols = uBlock.info.y;
  const int batch_count = uBlock.info.z;

  if (batch >= batch_count || col >= input_cols) {
    return;
  }

  float value = uInput.data[batch * input_cols + col];
  for (int scatter_col = 0; scatter_col < scatter_cols; ++scatter_col) {
    const int scatter_offset = batch * scatter_cols + scatter_col;
    if (uIndex.data[scatter_offset] == col) {
      value = uSrc.data[scatter_offset];
    }
  }
  uOutput.data[batch * input_cols + col] = value;
}
