#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uQuery;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uKey;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uValue;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 sizes;      // batch, target_len, source_len, head_dim
  ivec4 tiled_info; // value_dim, local_size_x, max_outputs_per_thread, batch_groups
}
uBlock;

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

const float NEG_INF = -3.402823466e+38;
const float MIN_DENOM = 1.0e-20;
const int LOCAL_SIZE_X = 16;
const int MAX_OUTPUTS_PER_THREAD = 32;

shared vec4 sScorePartials[LOCAL_SIZE_X];
shared vec4 sRowMax;
shared vec4 sRowDenom;
shared vec4 sPrevScale;
shared vec4 sCurrScale;

void main() {
  const int lane = int(gl_LocalInvocationID.x);
  const int query_row = int(gl_WorkGroupID.y);
  const int batch_group = int(gl_WorkGroupID.z);

  if (query_row >= uBlock.sizes.y || batch_group >= uBlock.tiled_info.w) {
    return;
  }

  vec4 accumulators[MAX_OUTPUTS_PER_THREAD];
  int output_indices[MAX_OUTPUTS_PER_THREAD];
  int output_count = 0;

  for (int dx = lane; dx < uBlock.tiled_info.x; dx += LOCAL_SIZE_X) {
    if (output_count >= MAX_OUTPUTS_PER_THREAD) {
      break;
    }
    output_indices[output_count] = dx;
    accumulators[output_count] = vec4(0.0);
    ++output_count;
  }

  if (lane == 0) {
    sRowMax = vec4(NEG_INF);
    sRowDenom = vec4(0.0);
    sPrevScale = vec4(0.0);
    sCurrScale = vec4(0.0);
  }
  barrier();

  for (int source_index = 0; source_index < uBlock.sizes.z; ++source_index) {
    vec4 partial_score = vec4(0.0);
    for (int k = lane; k < uBlock.sizes.w; k += LOCAL_SIZE_X) {
      const vec4 query_texel =
          texelFetch(uQuery, ivec3(k, query_row, batch_group), 0);
      const vec4 key_texel =
          texelFetch(uKey, ivec3(k, source_index, batch_group), 0);
      partial_score += query_texel * key_texel;
    }

    sScorePartials[lane] = partial_score;
    barrier();

    for (int offset = LOCAL_SIZE_X / 2; offset > 0; offset /= 2) {
      if (lane < offset) {
        sScorePartials[lane] += sScorePartials[lane + offset];
      }
      barrier();
    }

    if (lane == 0) {
      const vec4 score = sScorePartials[0];
      const vec4 new_max = max(sRowMax, score);
      sPrevScale = exp(sRowMax - new_max);
      sCurrScale = exp(score - new_max);
      sRowDenom = sRowDenom * sPrevScale + sCurrScale;
      sRowMax = new_max;
    }
    barrier();

    for (int i = 0; i < output_count; ++i) {
      const vec4 value_texel =
          texelFetch(uValue, ivec3(output_indices[i], source_index, batch_group), 0);
      accumulators[i] =
          accumulators[i] * sPrevScale + sCurrScale * value_texel;
    }
    barrier();
  }

  const vec4 inverse_denom = 1.0 / max(sRowDenom, vec4(MIN_DENOM));
  for (int i = 0; i < output_count; ++i) {
    imageStore(
        uOutput,
        ivec3(output_indices[i], query_row, batch_group),
        accumulators[i] * inverse_denom);
  }
}
