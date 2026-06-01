#version 450 core

#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float data[];
}
uOutput;

layout(set = 0, binding = 1) buffer PRECISION restrict readonly WeightBuffer {
  float data[];
}
uWeight;

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
  const int embedding_dim = uBlock.info.x;
  const int num_indices = uBlock.info.y;
  const int num_embeddings = uBlock.info.z;
  const int index_word_stride = uBlock.info.w;

  if (pos.x >= embedding_dim || pos.y >= num_indices) {
    return;
  }

  const int src_row = uIndex.data[pos.y * index_word_stride];
  if (src_row < 0 || src_row >= num_embeddings) {
    return;
  }

  uOutput.data[pos.y * embedding_dim + pos.x] =
      uWeight.data[src_row * embedding_dim + pos.x];
}
