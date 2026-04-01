#version 450 core

#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

layout(std430) buffer;

layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float data[];
}
uOutput;

layout(set = 0, binding = 1) uniform PRECISION restrict OutMeta {
  uvec4 sizes;
  uvec4 strides;
  uvec4 info;
}
uOutMeta;

layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer {
  uint16_t data[];
}
uInput;

layout(set = 0, binding = 3) uniform PRECISION restrict InMeta {
  uvec4 sizes;
  uvec4 strides;
  uvec4 info;
}
uInMeta;

layout(set = 0, binding = 4) buffer PRECISION restrict readonly WeightBuffer {
  uint16_t data[];
}
uWeight;

layout(set = 0, binding = 5) uniform PRECISION restrict WeightMeta {
  uvec4 sizes;
  uvec4 strides;
  uvec4 info;
}
uWeightMeta;

layout(set = 0, binding = 6) buffer PRECISION restrict readonly BiasBuffer {
  float data[];
}
uBias;

layout(set = 0, binding = 7) uniform PRECISION restrict BiasMeta {
  uvec4 sizes;
  uvec4 strides;
  uvec4 info;
}
uBiasMeta;

layout(set = 0, binding = 8) uniform PRECISION restrict Block {
  ivec4 stride_pad;
  ivec4 dilation_groups;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float bfloat16_to_float(uint16_t raw) {
  return uintBitsToFloat(uint(raw) << 16);
}

void main() {
  const uint out_x = gl_GlobalInvocationID.x;
  const uint out_y = gl_GlobalInvocationID.y;
  const uint out_z = gl_GlobalInvocationID.z;

  const uint out_width = uOutMeta.sizes.x;
  const uint out_height = uOutMeta.sizes.y;
  const uint out_channels = uOutMeta.sizes.z;
  const uint batch_size = uOutMeta.sizes.w;

  if (out_x >= out_width || out_y >= out_height ||
      out_z >= batch_size * out_channels) {
    return;
  }

  const uint out_channel = out_z % out_channels;
  const uint batch = out_z / out_channels;

  const int stride_w = uBlock.stride_pad.x;
  const int stride_h = uBlock.stride_pad.y;
  const int pad_w = uBlock.stride_pad.z;
  const int pad_h = uBlock.stride_pad.w;
  const int dil_w = uBlock.dilation_groups.x;
  const int dil_h = uBlock.dilation_groups.y;
  const uint groups = uint(uBlock.dilation_groups.z);
  const bool has_bias = uBlock.dilation_groups.w != 0;

  const uint kernel_w = uWeightMeta.sizes.x;
  const uint kernel_h = uWeightMeta.sizes.y;
  const uint in_channels_per_group = uWeightMeta.sizes.z;
  const uint out_channels_per_group = out_channels / groups;
  const uint group_idx = out_channel / out_channels_per_group;
  const uint in_channel_start = group_idx * in_channels_per_group;

  float acc = 0.0;
  for (uint icg = 0u; icg < in_channels_per_group; ++icg) {
    const uint in_channel = in_channel_start + icg;
    for (uint ky = 0u; ky < kernel_h; ++ky) {
      const int in_y = int(out_y) * stride_h - pad_h + int(ky) * dil_h;
      if (in_y < 0 || in_y >= int(uInMeta.sizes.y)) {
        continue;
      }
      for (uint kx = 0u; kx < kernel_w; ++kx) {
        const int in_x = int(out_x) * stride_w - pad_w + int(kx) * dil_w;
        if (in_x < 0 || in_x >= int(uInMeta.sizes.x)) {
          continue;
        }

        const uint input_idx = uInMeta.info.z + uint(in_x) * uInMeta.strides.x +
            uint(in_y) * uInMeta.strides.y +
            in_channel * uInMeta.strides.z +
            batch * uInMeta.strides.w;
        const uint weight_idx = uWeightMeta.info.z + kx * uWeightMeta.strides.x +
            ky * uWeightMeta.strides.y +
            icg * uWeightMeta.strides.z +
            out_channel * uWeightMeta.strides.w;

        acc += bfloat16_to_float(uInput.data[input_idx]) *
            bfloat16_to_float(uWeight.data[weight_idx]);
      }
    }
  }

  if (has_bias) {
    const uint bias_idx = uBiasMeta.info.z + out_channel * uBiasMeta.strides.x;
    acc += uBias.data[bias_idx];
  }

  const uint out_idx = uOutMeta.info.z + out_x * uOutMeta.strides.x +
      out_y * uOutMeta.strides.y + out_channel * uOutMeta.strides.z +
      batch * uOutMeta.strides.w;
  uOutput.data[out_idx] = acc;
}
