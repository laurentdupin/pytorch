#version 450 core

#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

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
  ivec4 size0;
  ivec4 size1;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint write_idx = gl_GlobalInvocationID.x;
  const uint out_numel = uOutMeta.info.y;
  if (write_idx >= out_numel) {
    return;
  }

  const uvec4 out_coord =
      idx_to_coord(write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);
  const uint out_l = out_coord.x;
  const uint out_c = out_coord.y;
  const uint batch = out_coord.z;

  const int in_length = uBlock.size0.x;
  const int kernel_size = uBlock.size0.y;
  const int stride = uBlock.size0.z;
  const int padding = uBlock.size0.w;
  const int dilation = uBlock.size1.x;
  const int in_group_size = uBlock.size1.y;
  const int out_group_size = uBlock.size1.z;

  const uint out_storage_offset = uOutMeta.info.w;
  const uint in_storage_offset = uInMeta.info.w;
  const uint weight_storage_offset = uWeightMeta.info.w;
  const uint bias_storage_offset = uBiasMeta.info.w;
  const uint in_buf_length = uInMeta.info.z;
  const uint weight_buf_length = uWeightMeta.info.z;
  const uint bias_buf_length = uBiasMeta.info.z;
  const uint out_buf_length = uOutMeta.info.z;

  float sum = 0.0;
  const uint bias_idx =
      bias_storage_offset + out_c * uBiasMeta.physical_strides.x;
  if (bias_idx < bias_buf_length) {
    sum = uBias.data[bias_idx];
  }

  const uint group = out_c / uint(out_group_size);
  const uint in_c_base = group * uint(in_group_size);
  const int in_l_base = int(out_l) * stride - padding;

  for (int local_in_c = 0; local_in_c < in_group_size; ++local_in_c) {
    const uint in_c = in_c_base + uint(local_in_c);
    for (int k = 0; k < kernel_size; ++k) {
      const int in_l = in_l_base + k * dilation;
      if (in_l < 0 || in_l >= in_length) {
        continue;
      }

      const uint input_idx = in_storage_offset +
          uint(in_l) * uInMeta.physical_strides.x +
          in_c * uInMeta.physical_strides.y +
          batch * uInMeta.physical_strides.z;
      const uint weight_idx = weight_storage_offset +
          uint(k) * uWeightMeta.physical_strides.x +
          uint(local_in_c) * uWeightMeta.physical_strides.y +
          out_c * uWeightMeta.physical_strides.z;

      if (input_idx < in_buf_length && weight_idx < weight_buf_length) {
        sum += uInput.data[input_idx] * uWeight.data[weight_idx];
      }
    }
  }

  const uint actual_write_idx = out_storage_offset +
      out_l * uOutMeta.physical_strides.x +
      out_c * uOutMeta.physical_strides.y +
      batch * uOutMeta.physical_strides.z;
  if (actual_write_idx < out_buf_length) {
    uOutput.data[actual_write_idx] = sum;
  }
}
