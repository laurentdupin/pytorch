#version 450 core

#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

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

layout(set = 0, binding = 4) buffer PRECISION restrict readonly WeightBuffer {
  float data[];
}
uWeight;

layout(set = 0, binding = 5) uniform PRECISION restrict WeightMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uWeightMeta;

layout(set = 0, binding = 6) buffer PRECISION restrict readonly BiasBuffer {
  float data[];
}
uBias;

layout(set = 0, binding = 7) uniform PRECISION restrict BiasMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uBiasMeta;

layout(set = 0, binding = 8) buffer PRECISION restrict readonly ResidualBuffer {
  float data[];
}
uResidual;

layout(set = 0, binding = 9) uniform PRECISION restrict ResidualMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uResidualMeta;

layout(set = 0, binding = 10) uniform PRECISION restrict Block {
  ivec4 stride_pad;
  ivec4 dilation_groups;
  vec4 clamp_thresh;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint out_x = gl_GlobalInvocationID.x;
  const uint out_y = gl_GlobalInvocationID.y;
  const uint out_z = gl_GlobalInvocationID.z;

  const uint out_width = uOutMeta.logical_sizes.x;
  const uint out_height = uOutMeta.logical_sizes.y;
  const uint out_channels = uOutMeta.logical_sizes.z;
  const uint batch_size = uOutMeta.logical_sizes.w;

  if (out_x >= out_width || out_y >= out_height ||
      out_z >= batch_size * out_channels) {
    return;
  }

  const uint out_channel = out_z % out_channels;
  const uint batch = out_z / out_channels;
  const bool has_bias = uBlock.dilation_groups.w != 0;
  const uint in_channels = uWeightMeta.logical_sizes.z;
  const uint in_width = uInMeta.logical_sizes.x;
  const uint in_height = uInMeta.logical_sizes.y;

  float acc = 0.0;
  for (uint in_channel = 0u; in_channel < in_channels; ++in_channel) {
    const uint input_channel_base = uInMeta.info.w +
        in_channel * uInMeta.physical_strides.z +
        batch * uInMeta.physical_strides.w;
    const uint weight_base = uWeightMeta.info.w +
        in_channel * uWeightMeta.physical_strides.z +
        out_channel * uWeightMeta.physical_strides.w;

    const uint w00 = weight_base;
    const uint w01 = weight_base + uWeightMeta.physical_strides.x;
    const uint w02 = w01 + uWeightMeta.physical_strides.x;
    const uint w10 = weight_base + uWeightMeta.physical_strides.y;
    const uint w11 = w10 + uWeightMeta.physical_strides.x;
    const uint w12 = w11 + uWeightMeta.physical_strides.x;
    const uint w20 = w10 + uWeightMeta.physical_strides.y;
    const uint w21 = w20 + uWeightMeta.physical_strides.x;
    const uint w22 = w21 + uWeightMeta.physical_strides.x;

    if (out_x > 0u && out_y > 0u && out_x + 1u < in_width &&
        out_y + 1u < in_height) {
      const uint row0 = input_channel_base +
          (out_y - 1u) * uInMeta.physical_strides.y +
          (out_x - 1u) * uInMeta.physical_strides.x;
      const uint row1 = row0 + uInMeta.physical_strides.y;
      const uint row2 = row1 + uInMeta.physical_strides.y;

      acc += uInput.data[row0] * uWeight.data[w00];
      acc += uInput.data[row0 + uInMeta.physical_strides.x] * uWeight.data[w01];
      acc += uInput.data[row0 + 2u * uInMeta.physical_strides.x] * uWeight.data[w02];
      acc += uInput.data[row1] * uWeight.data[w10];
      acc += uInput.data[row1 + uInMeta.physical_strides.x] * uWeight.data[w11];
      acc += uInput.data[row1 + 2u * uInMeta.physical_strides.x] * uWeight.data[w12];
      acc += uInput.data[row2] * uWeight.data[w20];
      acc += uInput.data[row2 + uInMeta.physical_strides.x] * uWeight.data[w21];
      acc += uInput.data[row2 + 2u * uInMeta.physical_strides.x] * uWeight.data[w22];
    } else {
      const int base_x = int(out_x) - 1;
      const int base_y = int(out_y) - 1;
      if (base_y >= 0) {
        const uint row0 = input_channel_base +
            uint(base_y) * uInMeta.physical_strides.y;
        if (base_x >= 0) {
          acc += uInput.data[row0 + uint(base_x) * uInMeta.physical_strides.x] *
              uWeight.data[w00];
        }
        acc += uInput.data[row0 + out_x * uInMeta.physical_strides.x] *
            uWeight.data[w01];
        if (out_x + 1u < in_width) {
          acc += uInput.data[row0 + (out_x + 1u) * uInMeta.physical_strides.x] *
              uWeight.data[w02];
        }
      }
      {
        const uint row1 = input_channel_base + out_y * uInMeta.physical_strides.y;
        if (base_x >= 0) {
          acc += uInput.data[row1 + uint(base_x) * uInMeta.physical_strides.x] *
              uWeight.data[w10];
        }
        acc += uInput.data[row1 + out_x * uInMeta.physical_strides.x] *
            uWeight.data[w11];
        if (out_x + 1u < in_width) {
          acc += uInput.data[row1 + (out_x + 1u) * uInMeta.physical_strides.x] *
              uWeight.data[w12];
        }
      }
      if (out_y + 1u < in_height) {
        const uint row2 = input_channel_base +
            (out_y + 1u) * uInMeta.physical_strides.y;
        if (base_x >= 0) {
          acc += uInput.data[row2 + uint(base_x) * uInMeta.physical_strides.x] *
              uWeight.data[w20];
        }
        acc += uInput.data[row2 + out_x * uInMeta.physical_strides.x] *
            uWeight.data[w21];
        if (out_x + 1u < in_width) {
          acc += uInput.data[row2 + (out_x + 1u) * uInMeta.physical_strides.x] *
              uWeight.data[w22];
        }
      }
    }
  }

  if (has_bias) {
    const uint bias_idx =
        uBiasMeta.info.w + out_channel * uBiasMeta.physical_strides.x;
    acc += uBias.data[bias_idx];
  }

  const uint residual_idx = uResidualMeta.info.w +
      out_x * uResidualMeta.physical_strides.x +
      out_y * uResidualMeta.physical_strides.y +
      out_channel * uResidualMeta.physical_strides.z +
      batch * uResidualMeta.physical_strides.w;
  const uint out_idx = uOutMeta.info.w +
      out_x * uOutMeta.physical_strides.x +
      out_y * uOutMeta.physical_strides.y +
      out_channel * uOutMeta.physical_strides.z +
      batch * uOutMeta.physical_strides.w;
  const float value = clamp(acc, uBlock.clamp_thresh.x, uBlock.clamp_thresh.y) +
      uResidual.data[residual_idx];
  uOutput.data[out_idx] = value;
}
