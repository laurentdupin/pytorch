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
  ivec4 tiled_info; // value_dim, local_size_x, max_outputs_per_thread, max_query_values_per_thread
}
uBlock;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const float NEG_INF = -3.402823466e+38;
const float MIN_DENOM = 1.0e-20;
const int LOCAL_SIZE_X = 64;

shared float sScorePartials[LOCAL_SIZE_X];

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

  const uint query_buf_length = uQueryMeta.info.z;
  const uint query_storage_offset = uQueryMeta.info.w;
  const uint key_buf_length = uKeyMeta.info.z;
  const uint key_storage_offset = uKeyMeta.info.w;
  const uint value_buf_length = uValueMeta.info.z;
  const uint value_storage_offset = uValueMeta.info.w;
  const uint out_buf_length = uOutMeta.info.z;
  const uint out_storage_offset = uOutMeta.info.w;

  const bool has_head_value = lane < uBlock.sizes.w;
  const bool has_output_value = lane < uBlock.tiled_info.x;
  float query_value = 0.0;
  if (has_head_value) {
    const uint query_idx = buffer_idx(
        uvec4(uint(lane), uint(query_row), uint(batch_group), 0u),
        uQueryMeta.physical_strides,
        query_storage_offset);
    query_value = query_idx < query_buf_length ? uQuery.data[query_idx] : 0.0;
  }

  float accumulator = 0.0;
  float row_max = NEG_INF;
  float row_denom = 0.0;

  for (int source_index = 0; source_index < uBlock.sizes.z; ++source_index) {
    float partial_score = 0.0;
    if (has_head_value) {
      const uint key_idx = buffer_idx(
          uvec4(uint(lane), uint(source_index), uint(batch_group), 0u),
          uKeyMeta.physical_strides,
          key_storage_offset);
      partial_score = key_idx < key_buf_length
          ? query_value * uKey.data[key_idx]
          : 0.0;
    }

    sScorePartials[lane] = partial_score;
    barrier();

    for (int offset = LOCAL_SIZE_X / 2; offset > 0; offset /= 2) {
      if (lane < offset) {
        sScorePartials[lane] += sScorePartials[lane + offset];
      }
      barrier();
    }

    const float score = sScorePartials[0];
    const float new_max = max(row_max, score);
    const float previous_scale = exp(row_max - new_max);
    const float current_scale = exp(score - new_max);
    row_denom = row_denom * previous_scale + current_scale;
    row_max = new_max;

    if (has_output_value) {
      const uint value_idx = buffer_idx(
          uvec4(uint(lane), uint(source_index), uint(batch_group), 0u),
          uValueMeta.physical_strides,
          value_storage_offset);
      if (value_idx < value_buf_length) {
        accumulator =
            accumulator * previous_scale + current_scale * uValue.data[value_idx];
      }
    }
    barrier();
  }

  if (has_output_value) {
    const float inverse_denom = 1.0 / max(row_denom, MIN_DENOM);
    const uint out_idx = buffer_idx(
        uvec4(uint(lane), uint(query_row), uint(batch_group), 0u),
        uOutMeta.physical_strides,
        out_storage_offset);
    if (out_idx < out_buf_length) {
      uOutput.data[out_idx] = accumulator * inverse_denom;
    }
  }
}
