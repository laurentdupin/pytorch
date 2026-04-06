#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uWeight;

layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 output_extents;
  int normalized_size;
  float eps;
  ivec2 fill0;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (!all(lessThan(pos, uBlock.output_extents.xyz))) {
    return;
  }

  vec4 sumsq = vec4(0.0);
  for (int x = 0; x < uBlock.normalized_size; ++x) {
    const vec4 input_val = texelFetch(uInput, ivec3(x, pos.y, pos.z), 0);
    sumsq += input_val * input_val;
  }

  const float norm_size = max(float(uBlock.normalized_size), 1.0);
  const vec4 rstd = inversesqrt(sumsq / norm_size + vec4(uBlock.eps));

  for (int x = 0; x < uBlock.normalized_size; ++x) {
    const vec4 input_val = texelFetch(uInput, ivec3(x, pos.y, pos.z), 0);
    const float gamma = texelFetch(uWeight, ivec3(x, 0, 0), 0).x;
    imageStore(uOutput, ivec3(x, pos.y, pos.z), input_val * rstd * gamma);
  }
}
