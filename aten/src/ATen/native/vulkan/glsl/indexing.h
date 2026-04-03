/*
 * Computes a 4D tensor coordinate from a linearized index
 */
uvec4 idx_to_coord(const uint idx, const uvec4 strides, const uvec4 sizes) {
  return uvec4(
      (idx / strides.x) % sizes.x,
      (idx / strides.y) % sizes.y,
      (idx / strides.z) % sizes.z,
      (idx / strides.w) % sizes.w);
}

/*
 * Computes a linearized index from a 4D tensor coordinate
 */
uint coord_to_idx(const uvec4 coord, const uvec4 strides) {
  const uvec4 linear_terms = coord * strides;
  return linear_terms.x + linear_terms.y + linear_terms.z + linear_terms.w;
}

int align_up_4(int v) {
  return ((v + 4 - 1) / 4) * 4;
}

// Return the x, y, z and index value the channel-packed 3D tensor from the {n,
// c, h, w}-index.
ivec4 get_channel_packed_pos_from_index(ivec4 nchw, ivec4 sizes) {
  int n = nchw.x;
  int c = nchw.y;
  int h = nchw.z;
  int w = nchw.w;

  int aligned_c = align_up_4(sizes.y);
  int c_stride = aligned_c / 4;

  return ivec4(
      w, // x
      h, // y
      n * c_stride + c / 4, // z
      c % 4);
}
