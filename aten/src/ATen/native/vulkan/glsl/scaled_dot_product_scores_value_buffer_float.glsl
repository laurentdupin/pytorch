#version 450 core

#include "indexing.h"

layout(std430) buffer;

layout(set = 0, binding = 0) buffer restrict writeonly OutBuffer {
  float data[];
}
uOutput;

layout(set = 0, binding = 1) uniform restrict OutMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uOutMeta;

layout(set = 0, binding = 2) buffer restrict readonly QueryBuffer {
  float data[];
}
uQuery;

layout(set = 0, binding = 3) uniform restrict QueryMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uQueryMeta;

layout(set = 0, binding = 4) buffer restrict readonly KeyBuffer {
  float data[];
}
uKey;

layout(set = 0, binding = 5) uniform restrict KeyMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uKeyMeta;

layout(set = 0, binding = 6) buffer restrict readonly ValueBuffer {
  float data[];
}
uValue;

layout(set = 0, binding = 7) uniform restrict ValueMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uValueMeta;

layout(set = 0, binding = 8) uniform restrict Block {
  ivec4 sizes;      // batch_heads, target_len, source_len, head_dim
  ivec4 tiled_info; // value_dim, local_size_x, max_outputs_per_thread, unused
}
uBlock;

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

const float NEG_INF = -3.402823466e+38;
const float MIN_DENOM = 1.0e-20;
const int LOCAL_SIZE_X = 16;
const int MAX_OUTPUTS_PER_THREAD = 32;
const int MAX_QUERY_VALUES_PER_THREAD = 8;

shared float sScorePartials[LOCAL_SIZE_X];
shared float sPrevScale;
shared float sCurrScale;
shared float sInvDenom;

uint buffer_idx(const uvec4 coord, const uvec4 strides, const uint offset) {
  return coord_to_idx(coord, strides) + offset;
}

void main() {
  const int lane = int(gl_LocalInvocationID.x);
  const int query_row = int(gl_WorkGroupID.y);
  const int batch_group = int(gl_WorkGroupID.z);

  if (query_row >= uBlock.sizes.y || batch_group >= uBlock.sizes.x) {
    return;
  }

  float accumulators[MAX_OUTPUTS_PER_THREAD];
  int output_indices[MAX_OUTPUTS_PER_THREAD];
  int output_count = 0;

  for (int dx = lane; dx < uBlock.tiled_info.x; dx += LOCAL_SIZE_X) {
    if (output_count >= MAX_OUTPUTS_PER_THREAD) {
      break;
    }
    output_indices[output_count] = dx;
    accumulators[output_count] = 0.0;
    ++output_count;
  }

  const uint query_buf_length = uQueryMeta.info.z;
  const uint query_storage_offset = uQueryMeta.info.w;
  const uint key_buf_length = uKeyMeta.info.z;
  const uint key_storage_offset = uKeyMeta.info.w;
  const uint value_buf_length = uValueMeta.info.z;
  const uint value_storage_offset = uValueMeta.info.w;
  const uint out_buf_length = uOutMeta.info.z;
  const uint out_storage_offset = uOutMeta.info.w;

  float query_values[MAX_QUERY_VALUES_PER_THREAD];
  int query_count = 0;
  for (int k = lane; k < uBlock.sizes.w; k += LOCAL_SIZE_X) {
    if (query_count >= MAX_QUERY_VALUES_PER_THREAD) {
      break;
    }
    const uint query_idx = buffer_idx(
        uvec4(uint(k), uint(query_row), uint(batch_group), 0u),
        uQueryMeta.physical_strides,
        query_storage_offset);
    query_values[query_count] =
        query_idx < query_buf_length ? uQuery.data[query_idx] : 0.0;
    ++query_count;
  }

  float row_max = NEG_INF;
  float row_denom = 0.0;

  for (int source_index = 0; source_index < uBlock.sizes.z; ++source_index) {
    float partial_score = 0.0;
    for (int i = 0; i < query_count; ++i) {
      const int k = lane + i * LOCAL_SIZE_X;
      const uint key_idx = buffer_idx(
          uvec4(uint(k), uint(source_index), uint(batch_group), 0u),
          uKeyMeta.physical_strides,
          key_storage_offset);
      if (key_idx < key_buf_length) {
        partial_score += query_values[i] * uKey.data[key_idx];
      }
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
      const float score = sScorePartials[0];
      const float new_max = max(row_max, score);
      sPrevScale = exp(row_max - new_max);
      sCurrScale = exp(score - new_max);
      row_denom = row_denom * sPrevScale + sCurrScale;
      row_max = new_max;
    }
    barrier();

    const float prev_scale = sPrevScale;
    const float curr_scale = sCurrScale;
    for (int i = 0; i < output_count; ++i) {
      const uint value_idx = buffer_idx(
          uvec4(
              uint(output_indices[i]),
              uint(source_index),
              uint(batch_group),
              0u),
          uValueMeta.physical_strides,
          value_storage_offset);
      if (value_idx < value_buf_length) {
        accumulators[i] =
            accumulators[i] * prev_scale + curr_scale * uValue.data[value_idx];
      }
    }
    barrier();
  }

  if (lane == 0) {
    sInvDenom = 1.0 / max(row_denom, MIN_DENOM);
  }
  barrier();

  const float inverse_denom = sInvDenom;
  for (int i = 0; i < output_count; ++i) {
    const uint out_idx = buffer_idx(
        uvec4(uint(output_indices[i]), uint(query_row), uint(batch_group), 0u),
        uOutMeta.physical_strides,
        out_storage_offset);
    if (out_idx < out_buf_length) {
      uOutput.data[out_idx] = accumulators[i] * inverse_denom;
    }
  }
}
