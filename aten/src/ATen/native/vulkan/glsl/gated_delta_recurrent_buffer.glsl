#version 450 core

#define PRECISION ${PRECISION}

#include "indexing.h"

layout(std430) buffer;

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float data[];
}
uOutput;

layout(set = 0, binding = 1) uniform PRECISION restrict OutMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uOutMeta;

layout(set = 0, binding = 2) buffer PRECISION restrict writeonly OutStateBuffer {
  float data[];
}
uOutputState;

layout(set = 0, binding = 3) uniform PRECISION restrict OutStateMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uOutStateMeta;

layout(set = 0, binding = 4) buffer PRECISION restrict readonly QueryBuffer {
  float data[];
}
uQuery;

layout(set = 0, binding = 5) uniform PRECISION restrict QueryMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uQueryMeta;

layout(set = 0, binding = 6) buffer PRECISION restrict readonly KeyBuffer {
  float data[];
}
uKey;

layout(set = 0, binding = 7) uniform PRECISION restrict KeyMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uKeyMeta;

layout(set = 0, binding = 8) buffer PRECISION restrict readonly ValueBuffer {
  float data[];
}
uValue;

layout(set = 0, binding = 9) uniform PRECISION restrict ValueMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uValueMeta;

layout(set = 0, binding = 10) buffer PRECISION restrict readonly GBuffer {
  float data[];
}
uG;

layout(set = 0, binding = 11) uniform PRECISION restrict GMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uGMeta;

layout(set = 0, binding = 12) buffer PRECISION restrict readonly BetaBuffer {
  float data[];
}
uBeta;

layout(set = 0, binding = 13) uniform PRECISION restrict BetaMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uBetaMeta;

layout(set = 0, binding = 14) buffer PRECISION restrict readonly InitialStateBuffer {
  float data[];
}
uInitialState;

layout(set = 0, binding = 15) uniform PRECISION restrict InitialStateMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uInitialStateMeta;

layout(set = 0, binding = 16) uniform PRECISION restrict Block {
  ivec4 sizes0;
  ivec4 sizes1;
  vec4 params;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

const int kMaxHeadDim = 128;

uint query_index(
    const int batch_idx,
    const int head_idx,
    const int seq_idx,
    const int k_idx) {
  const uvec4 coord = uvec4(
      uint(k_idx),
      uint(seq_idx),
      uint(head_idx),
      uint(batch_idx));
  return coord_to_idx(coord, uQueryMeta.physical_strides) + uQueryMeta.info.w;
}

uint key_index(
    const int batch_idx,
    const int head_idx,
    const int seq_idx,
    const int k_idx) {
  const uvec4 coord = uvec4(
      uint(k_idx),
      uint(seq_idx),
      uint(head_idx),
      uint(batch_idx));
  return coord_to_idx(coord, uKeyMeta.physical_strides) + uKeyMeta.info.w;
}

uint value_index(
    const int batch_idx,
    const int head_idx,
    const int seq_idx,
    const int v_idx) {
  const uvec4 coord = uvec4(
      uint(v_idx),
      uint(seq_idx),
      uint(head_idx),
      uint(batch_idx));
  return coord_to_idx(coord, uValueMeta.physical_strides) + uValueMeta.info.w;
}

uint scalar_index(const int batch_idx, const int head_idx, const int seq_idx) {
  const uvec4 coord = uvec4(0u, uint(seq_idx), uint(head_idx), uint(batch_idx));
  return coord_to_idx(coord, uGMeta.physical_strides) + uGMeta.info.w;
}

uint beta_index(const int batch_idx, const int head_idx, const int seq_idx) {
  const uvec4 coord = uvec4(0u, uint(seq_idx), uint(head_idx), uint(batch_idx));
  return coord_to_idx(coord, uBetaMeta.physical_strides) + uBetaMeta.info.w;
}

uint state_index(
    const int batch_idx,
    const int head_idx,
    const int k_idx,
    const int v_idx) {
  const uvec4 coord = uvec4(
      uint(v_idx),
      uint(k_idx),
      uint(head_idx),
      uint(batch_idx));
  return coord_to_idx(coord, uOutStateMeta.physical_strides) +
      uOutStateMeta.info.w;
}

uint initial_state_index(
    const int batch_idx,
    const int head_idx,
    const int k_idx,
    const int v_idx) {
  const uvec4 coord = uvec4(
      uint(v_idx),
      uint(k_idx),
      uint(head_idx),
      uint(batch_idx));
  return coord_to_idx(coord, uInitialStateMeta.physical_strides) +
      uInitialStateMeta.info.w;
}

float read_query(
    const int batch_idx,
    const int head_idx,
    const int seq_idx,
    const int k_idx) {
  const uint idx = query_index(batch_idx, head_idx, seq_idx, k_idx);
  return idx < uQueryMeta.info.z ? uQuery.data[idx] : 0.0;
}

float read_key(
    const int batch_idx,
    const int head_idx,
    const int seq_idx,
    const int k_idx) {
  const uint idx = key_index(batch_idx, head_idx, seq_idx, k_idx);
  return idx < uKeyMeta.info.z ? uKey.data[idx] : 0.0;
}

float read_g(const int batch_idx, const int head_idx, const int seq_idx) {
  const uint idx = scalar_index(batch_idx, head_idx, seq_idx);
  return idx < uGMeta.info.z ? uG.data[idx] : 0.0;
}

float read_beta(const int batch_idx, const int head_idx, const int seq_idx) {
  const uint idx = beta_index(batch_idx, head_idx, seq_idx);
  return idx < uBetaMeta.info.z ? uBeta.data[idx] : 0.0;
}

vec4 load_value_pack(
    const int batch_idx,
    const int head_idx,
    const int seq_idx,
    const int value_base) {
  vec4 value_pack = vec4(0.0);
  for (int lane = 0; lane < 4; ++lane) {
    const int value_idx = value_base + lane;
    if (value_idx < uBlock.sizes1.x) {
      const uint idx = value_index(batch_idx, head_idx, seq_idx, value_idx);
      value_pack[lane] = idx < uValueMeta.info.z ? uValue.data[idx] : 0.0;
    }
  }
  return value_pack;
}

vec4 load_state_pack(
    const int batch_idx,
    const int head_idx,
    const int k_idx,
    const int value_base) {
  vec4 state_pack = vec4(0.0);
  if (uBlock.sizes1.y == 0) {
    return state_pack;
  }
  for (int lane = 0; lane < 4; ++lane) {
    const int value_idx = value_base + lane;
    if (value_idx < uBlock.sizes1.x) {
      const uint idx =
          initial_state_index(batch_idx, head_idx, k_idx, value_idx);
      state_pack[lane] =
          idx < uInitialStateMeta.info.z ? uInitialState.data[idx] : 0.0;
    }
  }
  return state_pack;
}

void store_value_pack(
    const int batch_idx,
    const int head_idx,
    const int seq_idx,
    const int value_base,
    const vec4 value_pack) {
  for (int lane = 0; lane < 4; ++lane) {
    const int value_idx = value_base + lane;
    if (value_idx < uBlock.sizes1.x) {
      const uint idx = value_index(batch_idx, head_idx, seq_idx, value_idx);
      if (idx < uOutMeta.info.z) {
        uOutput.data[idx] = value_pack[lane];
      }
    }
  }
}

void store_state_pack(
    const int batch_idx,
    const int head_idx,
    const int k_idx,
    const int value_base,
    const vec4 state_pack) {
  if (uBlock.sizes1.z == 0) {
    return;
  }
  for (int lane = 0; lane < 4; ++lane) {
    const int value_idx = value_base + lane;
    if (value_idx < uBlock.sizes1.x) {
      const uint idx = state_index(batch_idx, head_idx, k_idx, value_idx);
      if (idx < uOutStateMeta.info.z) {
        uOutputState.data[idx] = state_pack[lane];
      }
    }
  }
}

void main() {
  const int value_pack_idx = int(gl_GlobalInvocationID.x);
  const int head_idx = int(gl_GlobalInvocationID.y);
  const int batch_idx = int(gl_GlobalInvocationID.z);
  const int value_base = value_pack_idx * 4;

  if (batch_idx >= uBlock.sizes0.x || head_idx >= uBlock.sizes0.y ||
      value_base >= uBlock.sizes1.x) {
    return;
  }

  vec4 recurrent_state[kMaxHeadDim];
  float query_cache[kMaxHeadDim];
  float key_cache[kMaxHeadDim];

  for (int k_idx = 0; k_idx < uBlock.sizes0.w; ++k_idx) {
    recurrent_state[k_idx] =
        load_state_pack(batch_idx, head_idx, k_idx, value_base);
  }

  for (int seq_idx = 0; seq_idx < uBlock.sizes0.z; ++seq_idx) {
    float query_norm_sq = 0.0;
    float key_norm_sq = 0.0;

    for (int k_idx = 0; k_idx < uBlock.sizes0.w; ++k_idx) {
      const float query_val = read_query(batch_idx, head_idx, seq_idx, k_idx);
      const float key_val = read_key(batch_idx, head_idx, seq_idx, k_idx);
      query_cache[k_idx] = query_val;
      key_cache[k_idx] = key_val;
      if (uBlock.sizes1.w != 0) {
        query_norm_sq += query_val * query_val;
        key_norm_sq += key_val * key_val;
      }
    }

    float query_multiplier = uBlock.params.x;
    float key_multiplier = 1.0;
    if (uBlock.sizes1.w != 0) {
      query_multiplier *= inversesqrt(query_norm_sq + uBlock.params.y);
      key_multiplier = inversesqrt(key_norm_sq + uBlock.params.y);
    }

    const float decay = exp(read_g(batch_idx, head_idx, seq_idx));
    const float beta = read_beta(batch_idx, head_idx, seq_idx);
    vec4 kv_memory = vec4(0.0);

    for (int k_idx = 0; k_idx < uBlock.sizes0.w; ++k_idx) {
      recurrent_state[k_idx] *= decay;
      key_cache[k_idx] *= key_multiplier;
      kv_memory += recurrent_state[k_idx] * key_cache[k_idx];
    }

    const vec4 value_vec =
        load_value_pack(batch_idx, head_idx, seq_idx, value_base);
    const vec4 delta = (value_vec - kv_memory) * beta;
    vec4 output_vec = vec4(0.0);

    for (int k_idx = 0; k_idx < uBlock.sizes0.w; ++k_idx) {
      recurrent_state[k_idx] += delta * key_cache[k_idx];
      output_vec += recurrent_state[k_idx] *
          (query_cache[k_idx] * query_multiplier);
    }

    store_value_pack(batch_idx, head_idx, seq_idx, value_base, output_vec);
  }

  for (int k_idx = 0; k_idx < uBlock.sizes0.w; ++k_idx) {
    store_state_pack(batch_idx, head_idx, k_idx, value_base, recurrent_state[k_idx]);
  }
}
