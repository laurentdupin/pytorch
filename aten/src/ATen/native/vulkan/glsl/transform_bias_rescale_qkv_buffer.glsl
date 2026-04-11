#version 450 core

#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly QBuffer {
  float data[];
}
uQ;

layout(set = 0, binding = 1) uniform PRECISION restrict QMeta {
  uvec4 sizes;
  uvec4 strides;
  uvec4 info;
}
uQMeta;

layout(set = 0, binding = 2) buffer PRECISION restrict writeonly KBuffer {
  float data[];
}
uK;

layout(set = 0, binding = 3) uniform PRECISION restrict KMeta {
  uvec4 sizes;
  uvec4 strides;
  uvec4 info;
}
uKMeta;

layout(set = 0, binding = 4) buffer PRECISION restrict writeonly VBuffer {
  float data[];
}
uV;

layout(set = 0, binding = 5) uniform PRECISION restrict VMeta {
  uvec4 sizes;
  uvec4 strides;
  uvec4 info;
}
uVMeta;

layout(set = 0, binding = 6) buffer PRECISION restrict readonly QKVBuffer {
  float data[];
}
uQKV;

layout(set = 0, binding = 7) uniform PRECISION restrict QKVMeta {
  uvec4 sizes;
  uvec4 strides;
  uvec4 info;
}
uQKVMeta;

layout(set = 0, binding = 8) buffer PRECISION restrict readonly BiasBuffer {
  float data[];
}
uBias;

layout(set = 0, binding = 9) uniform PRECISION restrict BiasMeta {
  uvec4 sizes;
  uvec4 strides;
  uvec4 info;
}
uBiasMeta;

layout(set = 0, binding = 10) uniform PRECISION restrict Block {
  ivec4 sizes;
  vec4 scale;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

uint output_index(
    const uint head,
    const uint token,
    const uint feature,
    const uvec4 strides,
    const uvec4 info) {
  const uint storage_offset = info.z;
  return storage_offset + feature * strides.x + token * strides.y +
      head * strides.z;
}

void main() {
  const uint feature = gl_GlobalInvocationID.x;
  const uint token = gl_GlobalInvocationID.y;
  const uint head = gl_GlobalInvocationID.z;

  const uint head_dim = uint(uBlock.sizes.x);
  const uint token_count = uint(uBlock.sizes.y);
  const uint num_head = uint(uBlock.sizes.z);
  const uint embed_dim = uint(uBlock.sizes.w);

  if (feature >= head_dim || token >= token_count || head >= num_head) {
    return;
  }

  const uint qkv_storage_offset = uQKVMeta.info.z;
  const uint bias_storage_offset = uBiasMeta.info.z;
  const uint qkv_row_offset = qkv_storage_offset + token * uQKVMeta.strides.y;

  const uint q_base = head * head_dim + feature;
  const uint k_base = embed_dim + q_base;
  const uint v_base = (2u * embed_dim) + q_base;

  const uint q_idx = qkv_row_offset + q_base * uQKVMeta.strides.x;
  const uint k_idx = qkv_row_offset + k_base * uQKVMeta.strides.x;
  const uint v_idx = qkv_row_offset + v_base * uQKVMeta.strides.x;

  const uint q_bias_idx = bias_storage_offset + q_base * uBiasMeta.strides.x;
  const uint k_bias_idx = bias_storage_offset + k_base * uBiasMeta.strides.x;
  const uint v_bias_idx = bias_storage_offset + v_base * uBiasMeta.strides.x;

  const float q_value =
      (uQKV.data[q_idx] + uBias.data[q_bias_idx]) * uBlock.scale.x;
  const float k_value = uQKV.data[k_idx] + uBias.data[k_bias_idx];
  const float v_value = uQKV.data[v_idx] + uBias.data[v_bias_idx];

  uQ.data[output_index(head, token, feature, uQMeta.strides, uQMeta.info)] =
      q_value;
  uK.data[output_index(head, token, feature, uKMeta.strides, uKMeta.info)] =
      k_value;
  uV.data[output_index(head, token, feature, uVMeta.strides, uVMeta.info)] =
      v_value;
}
