#version 450 core
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec3 out_extents;
  int fill0;
  uvec4 out_tensor_size;
  uvec4 in_tensor_size;
  uvec2 aligned_channels;
  uvec2 fill1;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (!all(lessThan(pos, uBlock.out_extents))) {
    return;
  }

  const uint aligned_out_channels = uBlock.aligned_channels.x;
  const uint aligned_in_channels = uBlock.aligned_channels.y;
  const uint max_dst_index = uBlock.out_tensor_size.w * aligned_out_channels;

  vec4 outval = vec4(0.0);
  for (uint lane = 0u; lane < 4u; ++lane) {
    const uint dst_index = uint(pos.z) * 4u + lane;
    if (dst_index >= max_dst_index) {
      break;
    }

    const uint n_out = dst_index / aligned_out_channels;
    const uint c_out = dst_index % aligned_out_channels;
    if (c_out >= uBlock.out_tensor_size.z) {
      continue;
    }

    uint linear_index =
        ((n_out * uBlock.out_tensor_size.z + c_out) * uBlock.out_tensor_size.y +
         uint(pos.y)) *
            uBlock.out_tensor_size.x +
        uint(pos.x);

    const uint w_in = linear_index % uBlock.in_tensor_size.x;
    linear_index /= uBlock.in_tensor_size.x;
    const uint h_in = linear_index % uBlock.in_tensor_size.y;
    linear_index /= uBlock.in_tensor_size.y;
    const uint c_in = linear_index % uBlock.in_tensor_size.z;
    linear_index /= uBlock.in_tensor_size.z;
    const uint n_in = linear_index;

    const uint src_index = n_in * aligned_in_channels + c_in;
    const ivec3 src_pos = ivec3(
        int(w_in),
        int(h_in),
        int(src_index / 4u));
    outval[lane] = texelFetch(uInput, src_pos, 0)[src_index % 4u];
  }

  imageStore(uOutput, pos, outval);
}
