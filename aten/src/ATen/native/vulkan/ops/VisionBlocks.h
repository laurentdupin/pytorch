#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <torch/library.h>

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class VisionBackboneBlockContext final : public torch::jit::CustomClassHolder {
 private:
  c10::impl::GenericList unpacked_{c10::AnyType::get()};
  uint64_t cache_id_{0u};
  std::string allocation_label_;
  c10::intrusive_ptr<LayernormPackedContext> norm1_context_;
  c10::intrusive_ptr<LinearPackedContext> qkv_context_;
  Tensor qkv_bias_;
  Tensor attention_bias_;
  int64_t num_heads_{0};
  c10::intrusive_ptr<LinearPackedContext> proj_context_;
  Tensor ls1_gamma_;
  c10::intrusive_ptr<LayernormPackedContext> norm2_context_;
  c10::intrusive_ptr<LinearPackedContext> fc1_context_;
  c10::intrusive_ptr<LinearPackedContext> fc2_context_;
  Tensor ls2_gamma_;

 public:
  VisionBackboneBlockContext(
      const Tensor& norm1_weight,
      const Tensor& norm1_bias,
      double norm1_eps,
      const Tensor& qkv_weight,
      const std::optional<Tensor>& qkv_bias,
      const std::optional<Tensor>& attention_bias,
      int64_t num_heads,
      const Tensor& proj_weight,
      const std::optional<Tensor>& proj_bias,
      const std::optional<Tensor>& ls1_gamma,
      const Tensor& norm2_weight,
      const Tensor& norm2_bias,
      double norm2_eps,
      const Tensor& fc1_weight,
      const std::optional<Tensor>& fc1_bias,
      const Tensor& fc2_weight,
      const std::optional<Tensor>& fc2_bias,
      const std::optional<Tensor>& ls2_gamma,
      std::string allocation_label = std::string());

  struct Unpacked final {
    static constexpr uint32_t Norm1Weight = 0u;
    static constexpr uint32_t Norm1Bias = 1u;
    static constexpr uint32_t Norm1Eps = 2u;
    static constexpr uint32_t QkvWeight = 3u;
    static constexpr uint32_t QkvBias = 4u;
    static constexpr uint32_t AttentionBias = 5u;
    static constexpr uint32_t NumHeads = 6u;
    static constexpr uint32_t ProjWeight = 7u;
    static constexpr uint32_t ProjBias = 8u;
    static constexpr uint32_t Ls1Gamma = 9u;
    static constexpr uint32_t Norm2Weight = 10u;
    static constexpr uint32_t Norm2Bias = 11u;
    static constexpr uint32_t Norm2Eps = 12u;
    static constexpr uint32_t Fc1Weight = 13u;
    static constexpr uint32_t Fc1Bias = 14u;
    static constexpr uint32_t Fc2Weight = 15u;
    static constexpr uint32_t Fc2Bias = 16u;
    static constexpr uint32_t Ls2Gamma = 17u;
    static constexpr uint32_t Label = 18u;
    static constexpr uint32_t NumArgs = 19u;
    static constexpr uint32_t LegacyNumArgs = 18u;
  };

  static VisionBackboneBlockContext pack(c10::impl::GenericList unpacked);

  const c10::impl::GenericList unpack() const {
    return unpacked_;
  }

  const std::string& allocation_label() const {
    return allocation_label_;
  }

  uint64_t cache_id() const {
    return cache_id_;
  }

  const c10::intrusive_ptr<LayernormPackedContext>& norm1_context() const {
    return norm1_context_;
  }

  const c10::intrusive_ptr<LinearPackedContext>& qkv_context() const {
    return qkv_context_;
  }

  const Tensor& qkv_bias() const {
    return qkv_bias_;
  }

  const Tensor& attention_bias() const {
    return attention_bias_;
  }

  int64_t num_heads() const {
    return num_heads_;
  }

  const c10::intrusive_ptr<LinearPackedContext>& proj_context() const {
    return proj_context_;
  }

  const Tensor& ls1_gamma() const {
    return ls1_gamma_;
  }

  const c10::intrusive_ptr<LayernormPackedContext>& norm2_context() const {
    return norm2_context_;
  }

  const c10::intrusive_ptr<LinearPackedContext>& fc1_context() const {
    return fc1_context_;
  }

  const c10::intrusive_ptr<LinearPackedContext>& fc2_context() const {
    return fc2_context_;
  }

  const Tensor& ls2_gamma() const {
    return ls2_gamma_;
  }
};

class VisionBackboneStackContext final : public torch::jit::CustomClassHolder {
 private:
  std::vector<c10::intrusive_ptr<VisionBackboneBlockContext>> blocks_;
  int64_t num_heads_{0};
  int64_t head_dim_{0};
  int64_t hidden_{0};
  int64_t mlp_hidden_{0};

 public:
  VisionBackboneStackContext(
      std::vector<c10::intrusive_ptr<VisionBackboneBlockContext>> blocks,
      int64_t num_heads,
      int64_t head_dim,
      int64_t hidden,
      int64_t mlp_hidden);

  const std::vector<c10::intrusive_ptr<VisionBackboneBlockContext>>& blocks()
      const {
    return blocks_;
  }

  int64_t num_heads() const {
    return num_heads_;
  }

  int64_t head_dim() const {
    return head_dim_;
  }

  int64_t hidden() const {
    return hidden_;
  }

  int64_t mlp_hidden() const {
    return mlp_hidden_;
  }
};

c10::intrusive_ptr<VisionBackboneBlockContext>
create_vision_backbone_block_context(
    Tensor&& norm1_weight,
    Tensor&& norm1_bias,
    double norm1_eps,
    Tensor&& qkv_weight,
    std::optional<Tensor>&& qkv_bias,
    int64_t num_heads,
    Tensor&& proj_weight,
    std::optional<Tensor>&& proj_bias,
    std::optional<Tensor>&& ls1_gamma,
    Tensor&& norm2_weight,
    Tensor&& norm2_bias,
    double norm2_eps,
    Tensor&& fc1_weight,
    std::optional<Tensor>&& fc1_bias,
    Tensor&& fc2_weight,
    std::optional<Tensor>&& fc2_bias,
    std::optional<Tensor>&& ls2_gamma,
    std::string label);

c10::intrusive_ptr<VisionBackboneBlockContext>
create_vision_backbone_block_context_with_attention_bias(
    Tensor&& norm1_weight,
    Tensor&& norm1_bias,
    double norm1_eps,
    Tensor&& qkv_weight,
    std::optional<Tensor>&& qkv_bias,
    std::optional<Tensor>&& attention_bias,
    int64_t num_heads,
    Tensor&& proj_weight,
    std::optional<Tensor>&& proj_bias,
    std::optional<Tensor>&& ls1_gamma,
    Tensor&& norm2_weight,
    Tensor&& norm2_bias,
    double norm2_eps,
    Tensor&& fc1_weight,
    std::optional<Tensor>&& fc1_bias,
    Tensor&& fc2_weight,
    std::optional<Tensor>&& fc2_bias,
    std::optional<Tensor>&& ls2_gamma,
    std::string label);

Tensor run_vision_backbone_block_context(
    const Tensor& input,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context);

c10::intrusive_ptr<VisionBackboneStackContext>
create_vision_backbone_stack_context(
    const c10::List<c10::intrusive_ptr<VisionBackboneBlockContext>>& blocks,
    int64_t num_heads,
    int64_t head_dim,
    int64_t hidden,
    int64_t mlp_hidden);

std::vector<Tensor> run_vision_backbone_stack_context(
    const Tensor& input,
    const c10::intrusive_ptr<VisionBackboneStackContext>& context,
    IntArrayRef capture_indices);

std::vector<int64_t> vision_owner_counters_snapshot();

void reset_vision_owner_counters();

std::vector<int64_t> vision_owner_context_counters_snapshot();

void reset_vision_owner_context_counters();

void record_vision_owner_context_cache_hit();

std::vector<int64_t> vision_owner_mlp_counters_snapshot();

void reset_vision_owner_mlp_counters();

std::vector<int64_t> vision_stack_owner_counters_snapshot();

void reset_vision_stack_owner_counters();

std::vector<int64_t> stack_attention_counters_snapshot();

void reset_stack_attention_counters();

void prime_vision_backbone_block_context_graph(
    const Tensor& input,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context);

class VisionDecoderFusionBlockContext final
    : public torch::jit::CustomClassHolder {
 private:
  c10::impl::GenericList unpacked_{c10::AnyType::get()};
  std::string allocation_label_;
  bool align_corners_{true};
  c10::intrusive_ptr<Conv2dPackedContext> res1_conv1_context_;
  c10::intrusive_ptr<Conv2dPackedContext> res1_conv2_context_;
  c10::intrusive_ptr<Conv2dPackedContext> res2_conv1_context_;
  c10::intrusive_ptr<Conv2dPackedContext> res2_conv2_context_;
  c10::intrusive_ptr<Conv2dPackedContext> out_conv_context_;

 public:
  VisionDecoderFusionBlockContext(
      const Tensor& res1_conv1_weight,
      const std::optional<Tensor>& res1_conv1_bias,
      const Tensor& res1_conv2_weight,
      const std::optional<Tensor>& res1_conv2_bias,
      const Tensor& res2_conv1_weight,
      const std::optional<Tensor>& res2_conv1_bias,
      const Tensor& res2_conv2_weight,
      const std::optional<Tensor>& res2_conv2_bias,
      const Tensor& out_conv_weight,
      const std::optional<Tensor>& out_conv_bias,
      bool align_corners,
      std::string allocation_label = std::string());

  struct Unpacked final {
    static constexpr uint32_t Res1Conv1Weight = 0u;
    static constexpr uint32_t Res1Conv1Bias = 1u;
    static constexpr uint32_t Res1Conv2Weight = 2u;
    static constexpr uint32_t Res1Conv2Bias = 3u;
    static constexpr uint32_t Res2Conv1Weight = 4u;
    static constexpr uint32_t Res2Conv1Bias = 5u;
    static constexpr uint32_t Res2Conv2Weight = 6u;
    static constexpr uint32_t Res2Conv2Bias = 7u;
    static constexpr uint32_t OutConvWeight = 8u;
    static constexpr uint32_t OutConvBias = 9u;
    static constexpr uint32_t AlignCorners = 10u;
    static constexpr uint32_t Label = 11u;
    static constexpr uint32_t NumArgs = 12u;
  };

  static VisionDecoderFusionBlockContext pack(c10::impl::GenericList unpacked);

  const c10::impl::GenericList unpack() const {
    return unpacked_;
  }

  const std::string& allocation_label() const {
    return allocation_label_;
  }

  bool align_corners() const {
    return align_corners_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& res1_conv1_context() const {
    return res1_conv1_context_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& res1_conv2_context() const {
    return res1_conv2_context_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& res2_conv1_context() const {
    return res2_conv1_context_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& res2_conv2_context() const {
    return res2_conv2_context_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& out_conv_context() const {
    return out_conv_context_;
  }
};

c10::intrusive_ptr<VisionDecoderFusionBlockContext>
create_vision_decoder_fusion_block_context(
    Tensor&& res1_conv1_weight,
    std::optional<Tensor>&& res1_conv1_bias,
    Tensor&& res1_conv2_weight,
    std::optional<Tensor>&& res1_conv2_bias,
    Tensor&& res2_conv1_weight,
    std::optional<Tensor>&& res2_conv1_bias,
    Tensor&& res2_conv2_weight,
    std::optional<Tensor>&& res2_conv2_bias,
    Tensor&& out_conv_weight,
    std::optional<Tensor>&& out_conv_bias,
    bool align_corners,
    std::string label);

Tensor run_vision_decoder_fusion_block_context(
    const Tensor& input,
    const std::optional<Tensor>& skip,
    const std::optional<std::vector<int64_t>>& size,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& context);

void prime_vision_decoder_fusion_block_context_graph(
    const Tensor& input,
    const std::optional<Tensor>& skip,
    const std::optional<std::vector<int64_t>>& size,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& context);

class VisionDecoderHeadContext final : public torch::jit::CustomClassHolder {
 private:
  c10::impl::GenericList unpacked_{c10::AnyType::get()};
  std::string allocation_label_;
  bool align_corners_{true};
  c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet4_context_;
  c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet3_context_;
  c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet2_context_;
  c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet1_context_;
  c10::intrusive_ptr<Conv2dPackedContext> output_conv1_context_;
  c10::intrusive_ptr<Conv2dPackedContext> output_conv2_conv1_context_;
  c10::intrusive_ptr<Conv2dPackedContext> output_conv2_conv2_context_;

 public:
  VisionDecoderHeadContext(
      c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet4_context,
      c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet3_context,
      c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet2_context,
      c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet1_context,
      c10::intrusive_ptr<Conv2dPackedContext> output_conv1_context,
      c10::intrusive_ptr<Conv2dPackedContext> output_conv2_conv1_context,
      c10::intrusive_ptr<Conv2dPackedContext> output_conv2_conv2_context,
      bool align_corners,
      std::string allocation_label = std::string());

  struct Unpacked final {
    static constexpr uint32_t Refinenet4Context = 0u;
    static constexpr uint32_t Refinenet3Context = 1u;
    static constexpr uint32_t Refinenet2Context = 2u;
    static constexpr uint32_t Refinenet1Context = 3u;
    static constexpr uint32_t OutputConv1Context = 4u;
    static constexpr uint32_t OutputConv2Conv1Context = 5u;
    static constexpr uint32_t OutputConv2Conv2Context = 6u;
    static constexpr uint32_t AlignCorners = 7u;
    static constexpr uint32_t Label = 8u;
    static constexpr uint32_t NumArgs = 9u;
  };

  static VisionDecoderHeadContext pack(c10::impl::GenericList unpacked);

  const c10::impl::GenericList unpack() const {
    return unpacked_;
  }

  const std::string& allocation_label() const {
    return allocation_label_;
  }

  bool align_corners() const {
    return align_corners_;
  }

  const c10::intrusive_ptr<VisionDecoderFusionBlockContext>&
  refinenet4_context() const {
    return refinenet4_context_;
  }

  const c10::intrusive_ptr<VisionDecoderFusionBlockContext>&
  refinenet3_context() const {
    return refinenet3_context_;
  }

  const c10::intrusive_ptr<VisionDecoderFusionBlockContext>&
  refinenet2_context() const {
    return refinenet2_context_;
  }

  const c10::intrusive_ptr<VisionDecoderFusionBlockContext>&
  refinenet1_context() const {
    return refinenet1_context_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& output_conv1_context() const {
    return output_conv1_context_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& output_conv2_conv1_context()
      const {
    return output_conv2_conv1_context_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& output_conv2_conv2_context()
      const {
    return output_conv2_conv2_context_;
  }
};

c10::intrusive_ptr<VisionDecoderHeadContext>
create_vision_decoder_head_context(
    const Tensor& prototype,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet4_context,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet3_context,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet2_context,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet1_context,
    c10::intrusive_ptr<Conv2dPackedContext> output_conv1_context,
    c10::intrusive_ptr<Conv2dPackedContext> output_conv2_conv1_context,
    c10::intrusive_ptr<Conv2dPackedContext> output_conv2_conv2_context,
    bool align_corners,
    std::string label);

Tensor run_vision_decoder_head_context(
    const Tensor& layer1,
    const Tensor& layer2,
    const Tensor& layer3,
    const Tensor& layer4,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderHeadContext>& context);

void prime_vision_decoder_head_context_graph(
    const Tensor& layer1,
    const Tensor& layer2,
    const Tensor& layer3,
    const Tensor& layer4,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderHeadContext>& context);

class VisionDecoderPreprocessHeadContext final
    : public torch::jit::CustomClassHolder {
 private:
  c10::impl::GenericList unpacked_{c10::AnyType::get()};
  std::string allocation_label_;
  c10::intrusive_ptr<Conv2dPackedContext> project1_context_;
  c10::intrusive_ptr<Conv2dPackedContext> project2_context_;
  c10::intrusive_ptr<Conv2dPackedContext> project3_context_;
  c10::intrusive_ptr<Conv2dPackedContext> project4_context_;
  c10::intrusive_ptr<Conv2dPackedContext> resize1_context_;
  c10::intrusive_ptr<Conv2dPackedContext> resize2_context_;
  c10::intrusive_ptr<Conv2dPackedContext> resize4_context_;
  c10::intrusive_ptr<Conv2dPackedContext> layer1_rn_context_;
  c10::intrusive_ptr<Conv2dPackedContext> layer2_rn_context_;
  c10::intrusive_ptr<Conv2dPackedContext> layer3_rn_context_;
  c10::intrusive_ptr<Conv2dPackedContext> layer4_rn_context_;
  c10::intrusive_ptr<VisionDecoderHeadContext> head_context_;

 public:
  VisionDecoderPreprocessHeadContext(
      c10::intrusive_ptr<Conv2dPackedContext> project1_context,
      c10::intrusive_ptr<Conv2dPackedContext> project2_context,
      c10::intrusive_ptr<Conv2dPackedContext> project3_context,
      c10::intrusive_ptr<Conv2dPackedContext> project4_context,
      c10::intrusive_ptr<Conv2dPackedContext> resize1_context,
      c10::intrusive_ptr<Conv2dPackedContext> resize2_context,
      c10::intrusive_ptr<Conv2dPackedContext> resize4_context,
      c10::intrusive_ptr<Conv2dPackedContext> layer1_rn_context,
      c10::intrusive_ptr<Conv2dPackedContext> layer2_rn_context,
      c10::intrusive_ptr<Conv2dPackedContext> layer3_rn_context,
      c10::intrusive_ptr<Conv2dPackedContext> layer4_rn_context,
      c10::intrusive_ptr<VisionDecoderHeadContext> head_context,
      std::string allocation_label = std::string());

  struct Unpacked final {
    static constexpr uint32_t Project1Context = 0u;
    static constexpr uint32_t Project2Context = 1u;
    static constexpr uint32_t Project3Context = 2u;
    static constexpr uint32_t Project4Context = 3u;
    static constexpr uint32_t Resize1Context = 4u;
    static constexpr uint32_t Resize2Context = 5u;
    static constexpr uint32_t Resize4Context = 6u;
    static constexpr uint32_t Layer1RnContext = 7u;
    static constexpr uint32_t Layer2RnContext = 8u;
    static constexpr uint32_t Layer3RnContext = 9u;
    static constexpr uint32_t Layer4RnContext = 10u;
    static constexpr uint32_t HeadContext = 11u;
    static constexpr uint32_t Label = 12u;
    static constexpr uint32_t NumArgs = 13u;
  };

  static VisionDecoderPreprocessHeadContext pack(
      c10::impl::GenericList unpacked);

  const c10::impl::GenericList unpack() const {
    return unpacked_;
  }

  const std::string& allocation_label() const {
    return allocation_label_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& project1_context() const {
    return project1_context_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& project2_context() const {
    return project2_context_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& project3_context() const {
    return project3_context_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& project4_context() const {
    return project4_context_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& resize1_context() const {
    return resize1_context_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& resize2_context() const {
    return resize2_context_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& resize4_context() const {
    return resize4_context_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& layer1_rn_context() const {
    return layer1_rn_context_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& layer2_rn_context() const {
    return layer2_rn_context_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& layer3_rn_context() const {
    return layer3_rn_context_;
  }

  const c10::intrusive_ptr<Conv2dPackedContext>& layer4_rn_context() const {
    return layer4_rn_context_;
  }

  const c10::intrusive_ptr<VisionDecoderHeadContext>& head_context() const {
    return head_context_;
  }
};

c10::intrusive_ptr<VisionDecoderPreprocessHeadContext>
create_vision_decoder_preprocess_head_context(
    const Tensor& prototype,
    c10::intrusive_ptr<Conv2dPackedContext> project1_context,
    c10::intrusive_ptr<Conv2dPackedContext> project2_context,
    c10::intrusive_ptr<Conv2dPackedContext> project3_context,
    c10::intrusive_ptr<Conv2dPackedContext> project4_context,
    c10::intrusive_ptr<Conv2dPackedContext> resize1_context,
    c10::intrusive_ptr<Conv2dPackedContext> resize2_context,
    c10::intrusive_ptr<Conv2dPackedContext> resize4_context,
    c10::intrusive_ptr<Conv2dPackedContext> layer1_rn_context,
    c10::intrusive_ptr<Conv2dPackedContext> layer2_rn_context,
    c10::intrusive_ptr<Conv2dPackedContext> layer3_rn_context,
    c10::intrusive_ptr<Conv2dPackedContext> layer4_rn_context,
    c10::intrusive_ptr<VisionDecoderHeadContext> head_context,
    std::string label);

Tensor run_vision_decoder_preprocess_head_context(
    const Tensor& layer1_tokens,
    const Tensor& layer2_tokens,
    const Tensor& layer3_tokens,
    const Tensor& layer4_tokens,
    int64_t patch_h,
    int64_t patch_w,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderPreprocessHeadContext>& context);

std::tuple<Tensor, Tensor> run_vision_backbone_decoder_replay_bundle_bridge(
    const Tensor& backbone_input,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& backbone_context,
    const Tensor& decoder_input,
    const std::optional<Tensor>& decoder_skip,
    const std::optional<std::vector<int64_t>>& decoder_size,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& decoder_context);

std::vector<Tensor> run_vision_backbone_stack_replay_bundle_bridge(
    const Tensor& input,
    const c10::List<c10::intrusive_ptr<VisionBackboneBlockContext>>& contexts,
    IntArrayRef capture_indices);

std::vector<Tensor> run_vision_backbone_stack_compiled_session_bridge(
    const Tensor& input,
    const c10::List<c10::intrusive_ptr<VisionBackboneBlockContext>>& contexts,
    IntArrayRef capture_indices);

std::vector<Tensor> run_vision_backbone_stack_norm_replay_bundle_bridge(
    const Tensor& input,
    const c10::List<c10::intrusive_ptr<VisionBackboneBlockContext>>& contexts,
    IntArrayRef capture_indices,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& norm_context);

std::vector<Tensor> run_vision_backbone_stack_norm_compiled_session_bridge(
    const Tensor& input,
    const c10::List<c10::intrusive_ptr<VisionBackboneBlockContext>>& contexts,
    IntArrayRef capture_indices,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& norm_context);

Tensor run_depth_anything_v2_compiled_session_bridge(
    const Tensor& input,
    const c10::List<c10::intrusive_ptr<VisionBackboneBlockContext>>& contexts,
    IntArrayRef capture_indices,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& norm_context,
    int64_t patch_h,
    int64_t patch_w,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderPreprocessHeadContext>&
        decoder_context);

Tensor run_depth_anything_v2_image_compiled_session_bridge(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dPackedContext>& patch_embed_context,
    const Tensor& prefix_token,
    const Tensor& patch_pos_encoding,
    const c10::List<c10::intrusive_ptr<VisionBackboneBlockContext>>& contexts,
    IntArrayRef capture_indices,
    IntArrayRef normalized_shape,
    const c10::intrusive_ptr<LayernormPackedContext>& norm_context,
    int64_t patch_h,
    int64_t patch_w,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderPreprocessHeadContext>&
        decoder_context);

Tensor tokens_to_feature_map(
    const Tensor& input,
    int64_t height,
    int64_t width);

Tensor feature_map_to_tokens(const Tensor& input);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
