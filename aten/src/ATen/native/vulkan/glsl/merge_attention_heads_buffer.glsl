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

layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 info;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint feature = gl_GlobalInvocationID.x;
  const uint token = gl_GlobalInvocationID.y;
  const uint batch_head = gl_GlobalInvocationID.z;

  const uint head_dim = uint(uBlock.info.x);
  const uint token_count = uint(uBlock.info.y);
  const uint num_heads = uint(uBlock.info.z);
  const uint batch_size = uint(uBlock.info.w);
  const uint batch_heads = batch_size * num_heads;

  if (feature >= head_dim || token >= token_count || batch_head >= batch_heads) {
    return;
  }

  const uint head = batch_head % num_heads;
  const uint batch = batch_head / num_heads;
  const uint out_row = batch * token_count + token;
  const uint out_col = head * head_dim + feature;

  const uint input_storage_offset = uInMeta.info.z;
  const uint output_storage_offset = uOutMeta.info.z;

  const uint input_idx = input_storage_offset +
      feature * uInMeta.strides.x +
      token * uInMeta.strides.y +
      batch_head * uInMeta.strides.z;
  const uint output_idx = output_storage_offset +
      out_col * uOutMeta.strides.x +
      out_row * uOutMeta.strides.y;

  uOutput.data[output_idx] = uInput.data[input_idx];
}
