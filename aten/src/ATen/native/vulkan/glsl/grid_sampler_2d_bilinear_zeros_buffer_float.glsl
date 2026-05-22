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

layout(set = 0, binding = 4) buffer PRECISION restrict readonly GridBuffer {
  float data[];
}
uGrid;

layout(set = 0, binding = 5) uniform PRECISION restrict GridMeta {
  uvec4 logical_sizes;
  uvec4 logical_strides;
  uvec4 physical_strides;
  uvec4 info;
}
uGridMeta;

layout(set = 0, binding = 6) uniform PRECISION restrict Params {
  uint align_corners;
  uint reserved0;
  uint reserved1;
  uint reserved2;
}
uParams;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

float unnormalize(float coord, uint size) {
  if (uParams.align_corners != 0u) {
    return ((coord + 1.0) * 0.5) * float(size - 1u);
  }
  return ((coord + 1.0) * float(size) - 1.0) * 0.5;
}

float load_grid(uint n, uint h, uint w, uint component) {
  const uvec4 coord = uvec4(component, w, h, n);
  const uint idx = coord_to_idx(coord, uGridMeta.physical_strides) +
      uGridMeta.info.w;
  return idx < uGridMeta.info.z ? uGrid.data[idx] : 0.0;
}

float load_input(int n, int c, int h, int w) {
  const int in_h = int(uInMeta.logical_sizes.y);
  const int in_w = int(uInMeta.logical_sizes.x);
  if (h < 0 || h >= in_h || w < 0 || w >= in_w) {
    return 0.0;
  }

  const uvec4 coord = uvec4(uint(w), uint(h), uint(c), uint(n));
  const uint idx = coord_to_idx(coord, uInMeta.physical_strides) +
      uInMeta.info.w;
  return idx < uInMeta.info.z ? uInput.data[idx] : 0.0;
}

void main() {
  const uint write_idx = gl_GlobalInvocationID.x;
  if (write_idx >= uOutMeta.info.y) {
    return;
  }

  const uvec4 out_coord =
      idx_to_coord(write_idx, uOutMeta.logical_strides, uOutMeta.logical_sizes);
  const uint w_out = out_coord.x;
  const uint h_out = out_coord.y;
  const uint c = out_coord.z;
  const uint n = out_coord.w;

  const float x = unnormalize(load_grid(n, h_out, w_out, 0u), uInMeta.logical_sizes.x);
  const float y = unnormalize(load_grid(n, h_out, w_out, 1u), uInMeta.logical_sizes.y);

  const int x0 = int(floor(x));
  const int y0 = int(floor(y));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const float wx = x - float(x0);
  const float wy = y - float(y0);

  const float v00 = load_input(int(n), int(c), y0, x0);
  const float v01 = load_input(int(n), int(c), y0, x1);
  const float v10 = load_input(int(n), int(c), y1, x0);
  const float v11 = load_input(int(n), int(c), y1, x1);

  const float top = mix(v00, v01, wx);
  const float bottom = mix(v10, v11, wx);
  const float output_value = mix(top, bottom, wy);

  const uint actual_write_idx =
      coord_to_idx(out_coord, uOutMeta.physical_strides) + uOutMeta.info.w;
  if (actual_write_idx < uOutMeta.info.z) {
    uOutput.data[actual_write_idx] = output_value;
  }
}
