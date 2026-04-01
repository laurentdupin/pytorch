#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;

layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 oextents;
  ivec2 iextents;
  vec2 scale;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float cubic_convolution1(float x, float a) {
  return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
}

float cubic_convolution2(float x, float a) {
  return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
}

vec4 cubic_coeffs(float t) {
  const float a = -0.75;
  const float x1 = t;
  const float x2 = 1.0 - t;
  return vec4(
      cubic_convolution2(x1 + 1.0, a),
      cubic_convolution1(x1, a),
      cubic_convolution1(x2, a),
      cubic_convolution2(x2 + 1.0, a));
}

vec4 cubic_interp1d(vec4 x0, vec4 x1, vec4 x2, vec4 x3, float t) {
  const vec4 coeffs = cubic_coeffs(t);
  return x0 * coeffs.x + x1 * coeffs.y + x2 * coeffs.z + x3 * coeffs.w;
}

vec4 fetch_bounded(ivec2 pos_xy, int z) {
  const ivec2 clamped_xy = clamp(pos_xy, ivec2(0, 0), uBlock.iextents.xy);
  return texelFetch(uInput, ivec3(clamped_xy, z), 0);
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThan(pos, uBlock.oextents.xyz))) {
    return;
  }

  const vec2 src = (vec2(pos.xy) + vec2(0.5, 0.5)) * uBlock.scale - vec2(0.5, 0.5);
  const ivec2 base = ivec2(floor(src));
  const vec2 t = src - vec2(base);

  vec4 rows[4];
  for (int k = 0; k < 4; ++k) {
    const int y = base.y - 1 + k;
    rows[k] = cubic_interp1d(
        fetch_bounded(ivec2(base.x - 1, y), pos.z),
        fetch_bounded(ivec2(base.x + 0, y), pos.z),
        fetch_bounded(ivec2(base.x + 1, y), pos.z),
        fetch_bounded(ivec2(base.x + 2, y), pos.z),
        t.x);
  }

  imageStore(uOutput, pos, cubic_interp1d(rows[0], rows[1], rows[2], rows[3], t.y));
}
