#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uQ;
layout(set = 0, binding = 1, FORMAT) uniform PRECISION restrict writeonly image3D uK;
layout(set = 0, binding = 2, FORMAT) uniform PRECISION restrict writeonly image3D uV;

layout(set = 0, binding = 3) uniform PRECISION sampler3D uQKV;
layout(set = 0, binding = 4) uniform PRECISION sampler3D uBias;

layout(set = 0, binding = 5) uniform PRECISION restrict Block {
  ivec4 sizes;
  vec4 scale;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (pos.x >= uBlock.sizes.x || pos.y >= uBlock.sizes.y || pos.z >= uBlock.sizes.z) {
    return;
  }

  vec4 q_out = vec4(0.0);
  vec4 k_out = vec4(0.0);
  vec4 v_out = vec4(0.0);

  for (int lane = 0; lane < 4; ++lane) {
    const int head = pos.z * 4 + lane;
    if (head >= uBlock.sizes.z) {
      break;
    }

    const int head_offset = head * uBlock.sizes.x + pos.x;
    const int q_index = head_offset;
    const int k_index = uBlock.sizes.w + head_offset;
    const int v_index = (2 * uBlock.sizes.w) + head_offset;

    const float q_val =
        texelFetch(uQKV, ivec3(q_index, pos.y, 0), 0).x +
        texelFetch(uBias, ivec3(q_index, 0, 0), 0).x;
    const float k_val =
        texelFetch(uQKV, ivec3(k_index, pos.y, 0), 0).x +
        texelFetch(uBias, ivec3(k_index, 0, 0), 0).x;
    const float v_val =
        texelFetch(uQKV, ivec3(v_index, pos.y, 0), 0).x +
        texelFetch(uBias, ivec3(v_index, 0, 0), 0).x;

    q_out[lane] = q_val * uBlock.scale.x;
    k_out[lane] = k_val;
    v_out[lane] = v_val;
  }

  imageStore(uQ, pos, q_out);
  imageStore(uK, pos, k_out);
  imageStore(uV, pos, v_out);
}
