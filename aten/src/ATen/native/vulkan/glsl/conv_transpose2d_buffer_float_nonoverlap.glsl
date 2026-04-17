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

layout(set = 0, binding = 8) uniform PRECISION restrict Block {
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

  const uint stride_w = uint(uBlock.stride_pad.x);
  const uint stride_h = uint(uBlock.stride_pad.y);
  const uint groups = uint(uBlock.dilation_groups.z);
  const bool has_bias = uBlock.dilation_groups.w != 0;

  const uint out_channels_per_group = uWeightMeta.logical_sizes.z;
  const uint in_channels = uInMeta.logical_sizes.z;
  const uint in_channels_per_group = in_channels / groups;
  const uint group_idx = out_channel / out_channels_per_group;
  const uint out_channel_in_group = out_channel % out_channels_per_group;
  const uint in_channel_start = group_idx * in_channels_per_group;

  const uint in_x = out_x / stride_w;
  const uint in_y = out_y / stride_h;
  const uint kernel_x = out_x % stride_w;
  const uint kernel_y = out_y % stride_h;

  float acc = 0.0;
  for (uint icg = 0u; icg < in_channels_per_group; ++icg) {
    const uint in_channel = in_channel_start + icg;
    const uint input_idx = uInMeta.info.w +
        in_x * uInMeta.physical_strides.x +
        in_y * uInMeta.physical_strides.y +
        in_channel * uInMeta.physical_strides.z +
        batch * uInMeta.physical_strides.w;
    const uint weight_idx = uWeightMeta.info.w +
        kernel_x * uWeightMeta.physical_strides.x +
        kernel_y * uWeightMeta.physical_strides.y +
        out_channel_in_group * uWeightMeta.physical_strides.z +
        in_channel * uWeightMeta.physical_strides.w;
    acc += uInput.data[input_idx] * uWeight.data[weight_idx];
  }

  if (has_bias) {
    const uint bias_idx =
        uBiasMeta.info.w + out_channel * uBiasMeta.physical_strides.x;
    acc += uBias.data[bias_idx];
  }

  const uint out_idx = uOutMeta.info.w +
      out_x * uOutMeta.physical_strides.x +
      out_y * uOutMeta.physical_strides.y +
      out_channel * uOutMeta.physical_strides.z +
      batch * uOutMeta.physical_strides.w;
  uOutput.data[out_idx] =
      clamp(acc, uBlock.clamp_thresh.x, uBlock.clamp_thresh.y);
}
