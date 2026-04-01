#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

/* Qualifiers: layout - storage - precision - memory */

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D   uOutput;
layout(set = 0, binding = 1)         uniform PRECISION                    sampler3D uInput;
layout(set = 0, binding = 2)         uniform PRECISION restrict           Block {
  ivec4 size;
  float beta;
  float threshold;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (all(lessThan(pos, uBlock.size.xyz))) {
    const vec4 inval = texelFetch(uInput, pos, 0);
    const vec4 scaled = inval * vec4(uBlock.beta);
    const vec4 limited = min(scaled, vec4(uBlock.threshold));
    const vec4 soft = log(vec4(1.0) + exp(limited)) / vec4(uBlock.beta);
    const bvec4 pass_through = greaterThan(scaled, vec4(uBlock.threshold));
    imageStore(uOutput, pos, mix(soft, inval, pass_through));
  }
}
