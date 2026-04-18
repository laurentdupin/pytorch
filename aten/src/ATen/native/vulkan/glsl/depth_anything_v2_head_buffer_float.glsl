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

layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer {
  float data[];
}
uInput;

layout(set = 0, binding = 3) uniform PRECISION restrict InMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uInMeta;

layout(set = 0, binding = 4) buffer PRECISION restrict readonly Weight1Buffer {
  float data[];
}
uWeight1;

layout(set = 0, binding = 5) uniform PRECISION restrict Weight1Meta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uWeight1Meta;

layout(set = 0, binding = 6) buffer PRECISION restrict readonly Bias1Buffer {
  float data[];
}
uBias1;

layout(set = 0, binding = 7) uniform PRECISION restrict Bias1Meta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uBias1Meta;

layout(set = 0, binding = 8) buffer PRECISION restrict readonly Weight2Buffer {
  float data[];
}
uWeight2;

layout(set = 0, binding = 9) uniform PRECISION restrict Weight2Meta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uWeight2Meta;

layout(set = 0, binding = 10) buffer PRECISION restrict readonly Bias2Buffer {
  float data[];
}
uBias2;

layout(set = 0, binding = 11) uniform PRECISION restrict Bias2Meta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uBias2Meta;

layout(set = 0, binding = 12) buffer PRECISION restrict readonly Weight3Buffer {
  float data[];
}
uWeight3;

layout(set = 0, binding = 13) uniform PRECISION restrict Weight3Meta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uWeight3Meta;

layout(set = 0, binding = 14) buffer PRECISION restrict readonly Bias3Buffer {
  float data[];
}
uBias3;

layout(set = 0, binding = 15) uniform PRECISION restrict Bias3Meta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uBias3Meta;

layout(set = 0, binding = 16) uniform PRECISION restrict Block {
  ivec4 info;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define HEAD_OUTPUT_CONV1_CHANNELS 32
#define HEAD_HIDDEN_CHANNELS 32
#define HEAD_OUTPUTS_PER_THREAD_X 2
#define HEAD_OUTPUTS_PER_THREAD_Y 1
#define HEAD_OUTPUT_SAMPLE_COLUMNS (HEAD_OUTPUTS_PER_THREAD_X + 2)
#define HEAD_OUTPUT_SAMPLE_ROWS 3
#define HEAD_LOWRES_TILE_WIDTH 21
#define HEAD_LOWRES_TILE_HEIGHT 12
#define HEAD_LOWRES_TILE_VALUES (HEAD_LOWRES_TILE_WIDTH * HEAD_LOWRES_TILE_HEIGHT * HEAD_OUTPUT_CONV1_CHANNELS)

shared int sLowX0;
shared int sLowY0;
shared int sLowWidth;
shared int sLowHeight;
shared int sTileValid;
shared float sConv1Tile[HEAD_LOWRES_TILE_VALUES];

uint conv1_tile_index(const int local_x, const int local_y, const int channel) {
  return uint(((local_y * HEAD_LOWRES_TILE_WIDTH) + local_x) *
      HEAD_OUTPUT_CONV1_CHANNELS + channel);
}

float source_index(const uint out_idx, const uint in_size, const uint out_size) {
  if (uBlock.info.x != 0) {
    if (in_size <= 1u || out_size <= 1u) {
      return 0.0;
    }
    return float(out_idx) * float(in_size - 1u) / float(out_size - 1u);
  }

  return ((float(out_idx) + 0.5) * float(in_size) / float(out_size)) - 0.5;
}

float read_path1_value(
    const int x,
    const int y,
    const int channel,
    const uint batch) {
  if (
      x < 0 || y < 0 || x >= int(uInMeta.logical_sizes.x) ||
      y >= int(uInMeta.logical_sizes.y)) {
    return 0.0;
  }

  const uvec4 coord = uvec4(uint(x), uint(y), uint(channel), batch);
  const uint read_idx =
      coord_to_idx(coord, uInMeta.physical_strides) + uInMeta.info.w;
  if (read_idx >= uInMeta.info.z) {
    return 0.0;
  }
  return uInput.data[read_idx];
}

float load_output_conv1_tile_value(
    const int low_x,
    const int low_y,
    const int out_channel,
    const uint batch) {
  float acc = 0.0;
  if (uBlock.info.y != 0) {
    const uint bias_idx = uBias1Meta.info.w +
        uint(out_channel) * uBias1Meta.physical_strides.x;
    acc = uBias1.data[bias_idx];
  }

  const uint in_width = uInMeta.logical_sizes.x;
  const uint in_height = uInMeta.logical_sizes.y;
  const int input_channels = int(uWeight1Meta.logical_sizes.z);
  for (int input_channel = 0; input_channel < input_channels; ++input_channel) {
    const uint input_channel_base = uInMeta.info.w +
        uint(input_channel) * uInMeta.physical_strides.z +
        batch * uInMeta.physical_strides.w;
    const uint weight_base = uWeight1Meta.info.w +
        uint(input_channel) * uWeight1Meta.physical_strides.z +
        uint(out_channel) * uWeight1Meta.physical_strides.w;

    const uint w00 = weight_base;
    const uint w01 = weight_base + uWeight1Meta.physical_strides.x;
    const uint w02 = w01 + uWeight1Meta.physical_strides.x;
    const uint w10 = weight_base + uWeight1Meta.physical_strides.y;
    const uint w11 = w10 + uWeight1Meta.physical_strides.x;
    const uint w12 = w11 + uWeight1Meta.physical_strides.x;
    const uint w20 = w10 + uWeight1Meta.physical_strides.y;
    const uint w21 = w20 + uWeight1Meta.physical_strides.x;
    const uint w22 = w21 + uWeight1Meta.physical_strides.x;

    if (low_x > 0 && low_y > 0 && low_x + 1 < int(in_width) &&
        low_y + 1 < int(in_height)) {
      const uint row0 = input_channel_base +
          uint(low_y - 1) * uInMeta.physical_strides.y +
          uint(low_x - 1) * uInMeta.physical_strides.x;
      const uint row1 = row0 + uInMeta.physical_strides.y;
      const uint row2 = row1 + uInMeta.physical_strides.y;

      acc += uInput.data[row0] * uWeight1.data[w00];
      acc += uInput.data[row0 + uInMeta.physical_strides.x] * uWeight1.data[w01];
      acc += uInput.data[row0 + 2u * uInMeta.physical_strides.x] * uWeight1.data[w02];
      acc += uInput.data[row1] * uWeight1.data[w10];
      acc += uInput.data[row1 + uInMeta.physical_strides.x] * uWeight1.data[w11];
      acc += uInput.data[row1 + 2u * uInMeta.physical_strides.x] * uWeight1.data[w12];
      acc += uInput.data[row2] * uWeight1.data[w20];
      acc += uInput.data[row2 + uInMeta.physical_strides.x] * uWeight1.data[w21];
      acc += uInput.data[row2 + 2u * uInMeta.physical_strides.x] * uWeight1.data[w22];
    } else {
      const int base_x = low_x - 1;
      const int base_y = low_y - 1;
      if (base_y >= 0) {
        const uint row0 =
            input_channel_base + uint(base_y) * uInMeta.physical_strides.y;
        if (base_x >= 0) {
          acc += uInput.data[row0 + uint(base_x) * uInMeta.physical_strides.x] *
              uWeight1.data[w00];
        }
        acc += uInput.data[row0 + uint(low_x) * uInMeta.physical_strides.x] *
            uWeight1.data[w01];
        if (uint(low_x) + 1u < in_width) {
          acc += uInput.data[row0 + (uint(low_x) + 1u) * uInMeta.physical_strides.x] *
              uWeight1.data[w02];
        }
      }
      {
        const uint row1 =
            input_channel_base + uint(low_y) * uInMeta.physical_strides.y;
        if (base_x >= 0) {
          acc += uInput.data[row1 + uint(base_x) * uInMeta.physical_strides.x] *
              uWeight1.data[w10];
        }
        acc += uInput.data[row1 + uint(low_x) * uInMeta.physical_strides.x] *
            uWeight1.data[w11];
        if (uint(low_x) + 1u < in_width) {
          acc += uInput.data[row1 + (uint(low_x) + 1u) * uInMeta.physical_strides.x] *
              uWeight1.data[w12];
        }
      }
      if (uint(low_y) + 1u < in_height) {
        const uint row2 =
            input_channel_base + (uint(low_y) + 1u) * uInMeta.physical_strides.y;
        if (base_x >= 0) {
          acc += uInput.data[row2 + uint(base_x) * uInMeta.physical_strides.x] *
              uWeight1.data[w20];
        }
        acc += uInput.data[row2 + uint(low_x) * uInMeta.physical_strides.x] *
            uWeight1.data[w21];
        if (uint(low_x) + 1u < in_width) {
          acc += uInput.data[row2 + (uint(low_x) + 1u) * uInMeta.physical_strides.x] *
              uWeight1.data[w22];
        }
      }
    }
  }

  return acc;
}

float read_conv1_tile_local(
    const int local_x,
    const int local_y,
    const int channel) {
  return sConv1Tile[conv1_tile_index(local_x, local_y, channel)];
}

void prepare_bilinear_x(
    const int hi_x,
    out ivec2 local_x_pair,
    out float alpha_x,
    out float valid) {
  if (hi_x < 0 || hi_x >= int(uOutMeta.logical_sizes.x)) {
    local_x_pair = ivec2(0);
    alpha_x = 0.0;
    valid = 0.0;
    return;
  }

  const float src_x =
      source_index(uint(hi_x), uInMeta.logical_sizes.x, uOutMeta.logical_sizes.x);
  const int base_x = int(floor(src_x));
  const int upper_x = int(ceil(src_x));
  const int clamped_base_x = clamp(base_x, 0, int(uInMeta.logical_sizes.x) - 1);
  const int clamped_upper_x = clamp(upper_x, 0, int(uInMeta.logical_sizes.x) - 1);
  local_x_pair = ivec2(clamped_base_x - sLowX0, clamped_upper_x - sLowX0);
  alpha_x = src_x - float(base_x);
  valid = 1.0;
}

void prepare_bilinear_y(
    const int hi_y,
    out ivec2 local_y_pair,
    out float alpha_y,
    out float valid) {
  if (hi_y < 0 || hi_y >= int(uOutMeta.logical_sizes.y)) {
    local_y_pair = ivec2(0);
    alpha_y = 0.0;
    valid = 0.0;
    return;
  }

  const float src_y =
      source_index(uint(hi_y), uInMeta.logical_sizes.y, uOutMeta.logical_sizes.y);
  const int base_y = int(floor(src_y));
  const int upper_y = int(ceil(src_y));
  const int clamped_base_y = clamp(base_y, 0, int(uInMeta.logical_sizes.y) - 1);
  const int clamped_upper_y = clamp(upper_y, 0, int(uInMeta.logical_sizes.y) - 1);
  local_y_pair = ivec2(clamped_base_y - sLowY0, clamped_upper_y - sLowY0);
  alpha_y = src_y - float(base_y);
  valid = 1.0;
}

float bilinear_sample_prepared(
    const ivec2 local_x_pair,
    const float alpha_x,
    const float x_valid,
    const ivec2 local_y_pair,
    const float alpha_y,
    const float y_valid,
    const int channel) {
  if (x_valid == 0.0 || y_valid == 0.0) {
    return 0.0;
  }

  const float top = mix(
      read_conv1_tile_local(local_x_pair.x, local_y_pair.x, channel),
      read_conv1_tile_local(local_x_pair.y, local_y_pair.x, channel),
      alpha_x);
  const float bottom = mix(
      read_conv1_tile_local(local_x_pair.x, local_y_pair.y, channel),
      read_conv1_tile_local(local_x_pair.y, local_y_pair.y, channel),
      alpha_x);
  return mix(top, bottom, alpha_y);
}

float compute_output_value_prepared(
    const int sample_x_offset,
    const uint out_channel,
    const ivec2 sample_x_pairs[HEAD_OUTPUT_SAMPLE_COLUMNS],
    const float sample_x_alpha[HEAD_OUTPUT_SAMPLE_COLUMNS],
    const float sample_x_valid[HEAD_OUTPUT_SAMPLE_COLUMNS],
    const ivec2 sample_y_pairs[HEAD_OUTPUT_SAMPLE_ROWS],
    const float sample_y_alpha[HEAD_OUTPUT_SAMPLE_ROWS],
    const float sample_y_valid[HEAD_OUTPUT_SAMPLE_ROWS]) {
  float hidden_acc[HEAD_HIDDEN_CHANNELS];
  for (int hidden_channel = 0; hidden_channel < HEAD_HIDDEN_CHANNELS;
       ++hidden_channel) {
    float value = 0.0;
    if (uBlock.info.z != 0) {
      const uint bias_idx = uBias2Meta.info.w +
          uint(hidden_channel) * uBias2Meta.physical_strides.x;
      value = uBias2.data[bias_idx];
    }
    hidden_acc[hidden_channel] = value;
  }

  for (int input_channel = 0; input_channel < HEAD_OUTPUT_CONV1_CHANNELS;
       ++input_channel) {
    float sample_rows[HEAD_OUTPUT_SAMPLE_ROWS][HEAD_OUTPUT_SAMPLE_COLUMNS];
    for (int sample_y = 0; sample_y < HEAD_OUTPUT_SAMPLE_ROWS; ++sample_y) {
      for (int sample_x = 0; sample_x < HEAD_OUTPUT_SAMPLE_COLUMNS; ++sample_x) {
        sample_rows[sample_y][sample_x] = bilinear_sample_prepared(
            sample_x_pairs[sample_x],
            sample_x_alpha[sample_x],
            sample_x_valid[sample_x],
            sample_y_pairs[sample_y],
            sample_y_alpha[sample_y],
            sample_y_valid[sample_y],
            input_channel);
      }
    }

    for (int hidden_channel = 0; hidden_channel < HEAD_HIDDEN_CHANNELS;
         ++hidden_channel) {
      const uint weight_base = uWeight2Meta.info.w +
          uint(input_channel) * uWeight2Meta.physical_strides.z +
          uint(hidden_channel) * uWeight2Meta.physical_strides.w;
      const uint w00 = weight_base;
      const uint w01 = weight_base + uWeight2Meta.physical_strides.x;
      const uint w02 = w01 + uWeight2Meta.physical_strides.x;
      const uint w10 = weight_base + uWeight2Meta.physical_strides.y;
      const uint w11 = w10 + uWeight2Meta.physical_strides.x;
      const uint w12 = w11 + uWeight2Meta.physical_strides.x;
      const uint w20 = w10 + uWeight2Meta.physical_strides.y;
      const uint w21 = w20 + uWeight2Meta.physical_strides.x;
      const uint w22 = w21 + uWeight2Meta.physical_strides.x;

      hidden_acc[hidden_channel] +=
          sample_rows[0][sample_x_offset + 0] * uWeight2.data[w00];
      hidden_acc[hidden_channel] +=
          sample_rows[0][sample_x_offset + 1] * uWeight2.data[w01];
      hidden_acc[hidden_channel] +=
          sample_rows[0][sample_x_offset + 2] * uWeight2.data[w02];
      hidden_acc[hidden_channel] +=
          sample_rows[1][sample_x_offset + 0] * uWeight2.data[w10];
      hidden_acc[hidden_channel] +=
          sample_rows[1][sample_x_offset + 1] * uWeight2.data[w11];
      hidden_acc[hidden_channel] +=
          sample_rows[1][sample_x_offset + 2] * uWeight2.data[w12];
      hidden_acc[hidden_channel] +=
          sample_rows[2][sample_x_offset + 0] * uWeight2.data[w20];
      hidden_acc[hidden_channel] +=
          sample_rows[2][sample_x_offset + 1] * uWeight2.data[w21];
      hidden_acc[hidden_channel] +=
          sample_rows[2][sample_x_offset + 2] * uWeight2.data[w22];
    }
  }

  float final_acc = 0.0;
  if (uBlock.info.w != 0) {
    const uint bias_idx =
        uBias3Meta.info.w + out_channel * uBias3Meta.physical_strides.x;
    final_acc = uBias3.data[bias_idx];
  }

  for (int hidden_channel = 0; hidden_channel < HEAD_HIDDEN_CHANNELS;
       ++hidden_channel) {
    const float activated = max(hidden_acc[hidden_channel], 0.0);
    const uint weight_idx = uWeight3Meta.info.w +
        uint(hidden_channel) * uWeight3Meta.physical_strides.z +
        out_channel * uWeight3Meta.physical_strides.w;
    final_acc += activated * uWeight3.data[weight_idx];
  }

  return max(final_acc, 0.0);
}

void main() {
  const uint out_width = uOutMeta.logical_sizes.x;
  const uint out_height = uOutMeta.logical_sizes.y;
  const uint out_channels = uOutMeta.logical_sizes.z;
  const uint batch_size = uOutMeta.logical_sizes.w;
  const uint group_output_width = gl_WorkGroupSize.x * HEAD_OUTPUTS_PER_THREAD_X;
  const uint group_output_height = gl_WorkGroupSize.y * HEAD_OUTPUTS_PER_THREAD_Y;
  const uint group_origin_x = gl_WorkGroupID.x * group_output_width;
  const uint group_origin_y = gl_WorkGroupID.y * group_output_height;
  const uint out_plane = gl_WorkGroupID.z;
  const uint local_linear_idx =
      gl_LocalInvocationID.x + gl_WorkGroupSize.x * gl_LocalInvocationID.y;
  const uint local_thread_count = gl_WorkGroupSize.x * gl_WorkGroupSize.y;

  if (group_origin_x >= out_width || group_origin_y >= out_height ||
      out_plane >= batch_size * out_channels) {
    return;
  }

  if (gl_LocalInvocationIndex == 0u) {
    const uint min_hi_x = group_origin_x > 0u ? group_origin_x - 1u : 0u;
    const uint min_hi_y = group_origin_y > 0u ? group_origin_y - 1u : 0u;
    const uint max_hi_x = min(group_origin_x + group_output_width, out_width - 1u);
    const uint max_hi_y = min(group_origin_y + group_output_height, out_height - 1u);
    const float src_x0 = source_index(min_hi_x, uInMeta.logical_sizes.x, out_width);
    const float src_x1 = source_index(max_hi_x, uInMeta.logical_sizes.x, out_width);
    const float src_y0 = source_index(min_hi_y, uInMeta.logical_sizes.y, out_height);
    const float src_y1 = source_index(max_hi_y, uInMeta.logical_sizes.y, out_height);
    const int low_x0 = clamp(
        int(floor(min(src_x0, src_x1))), 0, int(uInMeta.logical_sizes.x) - 1);
    const int low_y0 = clamp(
        int(floor(min(src_y0, src_y1))), 0, int(uInMeta.logical_sizes.y) - 1);
    const int low_x1 = clamp(
        int(ceil(max(src_x0, src_x1))), 0, int(uInMeta.logical_sizes.x) - 1);
    const int low_y1 = clamp(
        int(ceil(max(src_y0, src_y1))), 0, int(uInMeta.logical_sizes.y) - 1);

    sLowX0 = low_x0;
    sLowY0 = low_y0;
    sLowWidth = low_x1 - low_x0 + 1;
    sLowHeight = low_y1 - low_y0 + 1;
    sTileValid = (sLowWidth > 0 && sLowHeight > 0 &&
        sLowWidth <= HEAD_LOWRES_TILE_WIDTH &&
        sLowHeight <= HEAD_LOWRES_TILE_HEIGHT) ? 1 : 0;
  }
  barrier();

  if (sTileValid == 0) {
    return;
  }

  const uint batch = out_plane / out_channels;
  const int tile_value_count =
      sLowWidth * sLowHeight * HEAD_OUTPUT_CONV1_CHANNELS;
  for (int flat_idx = int(local_linear_idx); flat_idx < tile_value_count;
       flat_idx += int(local_thread_count)) {
    const int channel = flat_idx % HEAD_OUTPUT_CONV1_CHANNELS;
    const int spatial_idx = flat_idx / HEAD_OUTPUT_CONV1_CHANNELS;
    const int local_x = spatial_idx % sLowWidth;
    const int local_y = spatial_idx / sLowWidth;
    sConv1Tile[conv1_tile_index(local_x, local_y, channel)] =
        load_output_conv1_tile_value(
            sLowX0 + local_x, sLowY0 + local_y, channel, batch);
  }
  barrier();

  const uint out_channel = out_plane % out_channels;
  const uint base_out_x =
      group_origin_x + gl_LocalInvocationID.x * HEAD_OUTPUTS_PER_THREAD_X;
  const uint base_out_y =
      group_origin_y + gl_LocalInvocationID.y * HEAD_OUTPUTS_PER_THREAD_Y;
  if (base_out_y >= out_height) {
    return;
  }

  ivec2 sample_x_pairs[HEAD_OUTPUT_SAMPLE_COLUMNS];
  float sample_x_alpha[HEAD_OUTPUT_SAMPLE_COLUMNS];
  float sample_x_valid[HEAD_OUTPUT_SAMPLE_COLUMNS];
  for (int sample_x = 0; sample_x < HEAD_OUTPUT_SAMPLE_COLUMNS; ++sample_x) {
    prepare_bilinear_x(
        int(base_out_x) + sample_x - 1,
        sample_x_pairs[sample_x],
        sample_x_alpha[sample_x],
        sample_x_valid[sample_x]);
  }

  ivec2 sample_y_pairs[HEAD_OUTPUT_SAMPLE_ROWS];
  float sample_y_alpha[HEAD_OUTPUT_SAMPLE_ROWS];
  float sample_y_valid[HEAD_OUTPUT_SAMPLE_ROWS];
  for (int sample_y = 0; sample_y < HEAD_OUTPUT_SAMPLE_ROWS; ++sample_y) {
    prepare_bilinear_y(
        int(base_out_y) + sample_y - 1,
        sample_y_pairs[sample_y],
        sample_y_alpha[sample_y],
        sample_y_valid[sample_y]);
  }

  for (int local_x = 0; local_x < HEAD_OUTPUTS_PER_THREAD_X; ++local_x) {
    const uint out_x = base_out_x + uint(local_x);
    if (out_x >= out_width) {
      continue;
    }
    const uint write_idx = coord_to_idx(
        uvec4(out_x, base_out_y, out_channel, batch), uOutMeta.physical_strides) +
        uOutMeta.info.w;
    if (write_idx < uOutMeta.info.z) {
      uOutput.data[write_idx] = compute_output_value_prepared(
          local_x,
          out_channel,
          sample_x_pairs,
          sample_x_alpha,
          sample_x_valid,
          sample_y_pairs,
          sample_y_alpha,
          sample_y_valid);
    }
  }
}
