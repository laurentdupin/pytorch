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

layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 info;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const int batch = pos.y;
  const int col = pos.x;
  const int input_cols = uBlock.info.x;
  const int gather_cols = uBlock.info.y;
  const int batch_count = uBlock.info.z;

  if (batch >= batch_count || col >= gather_cols) {
    return;
  }

  const int index = uIndex.data[batch * gather_cols + col];
  uOutput.data[batch * gather_cols + col] =
      uInput.data[batch * input_cols + index];
}
