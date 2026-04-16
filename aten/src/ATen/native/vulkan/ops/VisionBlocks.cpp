#include <ATen/Functions.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/ops/BinaryOp.h>
#include <ATen/native/vulkan/ops/Clamp.h>
#include <ATen/native/vulkan/ops/Softmax.h>
#include <ATen/native/vulkan/ops/Upsample.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/ops/VisionBlocks.h>
#include <ATen/native/vulkan/planning/ExecutionPrograms.h>
#include <ATen/native/vulkan/planning/InferenceGraphs.h>
#include <ATen/native/vulkan/planning/Request.h>
#include <ATen/native/vulkan/planning/Runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <tuple>
#include <utility>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

std::string child_label(const std::string& label, const char* suffix) {
  if (label.empty()) {
    return std::string(suffix);
  }
  return label + "." + suffix;
}

Tensor move_optional_to_vulkan_buffer(const std::optional<Tensor>& tensor) {
  if (!tensor.has_value() || !tensor->defined()) {
    return Tensor();
  }
  Tensor vulkan_tensor = tensor->is_vulkan() ? *tensor : tensor->vulkan();
  return utils::mark_tensor_execution(
      utils::ensure_buffer_storage(
          vulkan_tensor, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
      api::ExecutionLayout::BUFFER_DIRECT,
      true);
}

Tensor maybe_restore_tensor(
    const Tensor& tensor,
    const Device& device,
    const ScalarType scalar_type) {
  Tensor restored = device.type() == kVulkan ? tensor : tensor.cpu();
  if (restored.scalar_type() != scalar_type) {
    restored = restored.to(scalar_type);
  }
  return restored;
}

c10::intrusive_ptr<LayernormPackedContext> make_layernorm_context(
    const Tensor& weight,
    const Tensor& bias,
    const double eps,
    const std::string& label) {
  std::optional<Tensor> owned_weight(weight);
  std::optional<Tensor> owned_bias(bias);
  return create_layernorm_context_labeled(
      std::move(owned_weight), std::move(owned_bias), eps, label);
}

c10::intrusive_ptr<LinearPackedContext> make_linear_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const std::string& label) {
  Tensor owned_weight = weight;
  std::optional<Tensor> owned_bias =
      bias.has_value() ? std::optional<Tensor>(*bias) : std::nullopt;
  return create_linear_context_labeled(
      std::move(owned_weight), std::move(owned_bias), label);
}

c10::intrusive_ptr<LinearPackedContext> make_qkv_context(
    const Tensor& weight,
    const std::string& label) {
  Tensor owned_weight = weight;
  std::optional<Tensor> no_bias = std::nullopt;
  return create_linear_context_labeled(
      std::move(owned_weight), std::move(no_bias), label);
}

c10::intrusive_ptr<Conv2dPackedContext> make_conv2d_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding) {
  Tensor owned_weight = weight;
  std::optional<Tensor> owned_bias =
      bias.has_value() ? std::optional<Tensor>(*bias) : std::nullopt;
  std::vector<int64_t> dilation{1, 1};
  return create_conv2d_context(
      std::move(owned_weight),
      std::move(owned_bias),
      std::move(stride),
      std::move(padding),
      std::move(dilation),
      1);
}

Tensor maybe_apply_layerscale(const Tensor& input, const Tensor& gamma) {
  if (!gamma.defined()) {
    return input;
  }
  return at::mul(input, gamma);
}

int64_t vision_block_hidden_dim(
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context) {
  return context->unpack()
      .get(VisionBackboneBlockContext::Unpacked::Fc1Weight)
      .toTensor()
      .size(0);
}

std::string vision_backbone_program_base_label(const std::string& label) {
  if (label.empty()) {
    return "depth.dino.backbone.block";
  }

  constexpr const char* kDynamicBlockMarker = ".block.";
  const auto marker_pos = label.find(kDynamicBlockMarker);
  if (marker_pos != std::string::npos) {
    return label.substr(0, marker_pos + 6u);
  }

  return label;
}

std::string append_context_identity_suffix(
    const std::string& label,
    const void* identity) {
  if (identity == nullptr) {
    return label;
  }
  return label + ".ctx." +
      std::to_string(static_cast<unsigned long long>(
          reinterpret_cast<uintptr_t>(identity)));
}

std::string vision_backbone_program_label(
    const std::string& label,
    const void* identity) {
  return append_context_identity_suffix(
             vision_backbone_program_base_label(label), identity) +
      ".program";
}

std::string vision_backbone_execution_label(
    const std::string& label,
    const void* identity) {
  return vision_backbone_program_label(label, identity) + ".exec";
}

std::string vision_decoder_program_label(
    const std::string& label,
    const void* identity) {
  if (label.empty()) {
    return append_context_identity_suffix("depth.decoder.fusion", identity) +
        ".program";
  }
  return append_context_identity_suffix(label, identity) + ".program";
}

std::string vision_decoder_program_base_label(const std::string& label) {
  if (label.empty()) {
    return "depth.decoder";
  }

  constexpr const char* kDynamicFusionMarker = ".fusion.";
  const auto marker_pos = label.find(kDynamicFusionMarker);
  if (marker_pos != std::string::npos) {
    return label.substr(0, marker_pos + 7u);
  }

  return label;
}

std::string vision_decoder_head_program_label(
    const std::string& label,
    const void* identity) {
  return append_context_identity_suffix(
             vision_decoder_program_base_label(label), identity) +
      ".head";
}

bool has_explicit_runtime_capture_label() {
  const std::string& runtime_label = api::current_runtime_label();
  return !runtime_label.empty() && runtime_label != "unlabeled";
}

std::string current_graph_capture_label(
    const std::string& fallback_base_label,
    const char* default_label) {
  const std::string& runtime_label = api::current_runtime_label();
  if (!runtime_label.empty() && runtime_label != "unlabeled") {
    return runtime_label + ".graph";
  }
  if (!fallback_base_label.empty()) {
    return fallback_base_label + ".graph";
  }
  return std::string(default_label);
}

std::string current_phase_graph_capture_label(
    const std::string& phase_base_label,
    const char* default_label) {
  const std::string& runtime_label = api::current_runtime_label();
  if (!runtime_label.empty() && runtime_label != "unlabeled") {
    if (!phase_base_label.empty()) {
      return runtime_label + "." + phase_base_label + ".graph";
    }
    return runtime_label + ".graph";
  }
  if (!phase_base_label.empty()) {
    return phase_base_label + ".graph";
  }
  return std::string(default_label);
}

std::string vision_backbone_graph_label(const std::string& label) {
  return current_phase_graph_capture_label(
      vision_backbone_program_base_label(label),
      "depth.dino.backbone.graph");
}

std::string vision_decoder_graph_label(const std::string& label) {
  return current_phase_graph_capture_label(
      vision_decoder_program_base_label(label),
      "depth.decoder.graph");
}

std::vector<int64_t> calc_contiguous_strides(IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size(), 1);
  for (int64_t idx = static_cast<int64_t>(sizes.size()) - 2; idx >= 0; --idx) {
    strides[idx] = strides[idx + 1] * std::max<int64_t>(sizes[idx + 1], 1);
  }
  return strides;
}

std::vector<int64_t> calc_width_packed_buffer_sizes(IntArrayRef sizes) {
  std::vector<int64_t> physical_sizes(sizes.begin(), sizes.end());
  if (!physical_sizes.empty()) {
    physical_sizes.back() =
        api::utils::align_up(physical_sizes.back(), INT64_C(4));
  }
  return physical_sizes;
}

size_t buffer_descriptor_nbytes(IntArrayRef sizes, const ScalarType dtype) {
  return static_cast<size_t>(
      api::element_size(convert_dtype(dtype)) *
      api::utils::multiply_integers(calc_width_packed_buffer_sizes(sizes)));
}

std::vector<int64_t> calc_width_packed_buffer_strides(IntArrayRef sizes) {
  return calc_contiguous_strides(calc_width_packed_buffer_sizes(sizes));
}

size_t align_up_size(const size_t value, const size_t alignment) {
  if (alignment <= 1u) {
    return value;
  }
  const size_t remainder = value % alignment;
  return remainder == 0u ? value : (value + alignment - remainder);
}

size_t vision_attention_scratch_bytes(
    const int64_t batch_size,
    const int64_t token_count,
    const int64_t embed_dim,
    const int64_t num_heads,
    const ScalarType dtype,
    const bool has_qkv_bias,
    const uint32_t alignment) {
  if (
      dtype != kFloat || num_heads <= 0 || embed_dim <= 0 || token_count <= 0 ||
      embed_dim % num_heads != 0 || batch_size <= 0) {
    return 0u;
  }

  const int64_t head_dim = embed_dim / num_heads;
  const int64_t batch_heads = batch_size * num_heads;

  size_t total_bytes = 0u;
  const auto append_slice = [&](const size_t slice_bytes) {
    total_bytes = align_up_size(total_bytes, alignment);
    total_bytes += slice_bytes;
  };

  if (batch_size == 1 && dtype == kFloat && has_qkv_bias) {
    const size_t mixed_qkv_bytes =
        buffer_descriptor_nbytes({token_count, 3 * embed_dim}, dtype);
    const size_t qkv_projection_bytes =
        buffer_descriptor_nbytes({num_heads, token_count, head_dim}, dtype);
    append_slice(mixed_qkv_bytes);
    append_slice(qkv_projection_bytes);
    append_slice(qkv_projection_bytes);
    append_slice(qkv_projection_bytes);
  }

  if (dtype == kFloat) {
    const size_t attention_scores_bytes = buffer_descriptor_nbytes(
        {batch_heads, token_count, token_count}, dtype);
    const size_t attention_context_bytes = buffer_descriptor_nbytes(
        {batch_heads, token_count, head_dim}, dtype);
    const size_t merge_output_bytes = buffer_descriptor_nbytes(
        {batch_size * token_count, embed_dim}, dtype);
    append_slice(attention_scores_bytes);
    append_slice(attention_scores_bytes);
    append_slice(attention_context_bytes);
    append_slice(merge_output_bytes);
  }

  return total_bytes;
}

std::vector<int64_t> resolve_decoder_target_sizes(
    const Tensor& input,
    const std::optional<std::vector<int64_t>>& size) {
  if (size.has_value()) {
    TORCH_CHECK(
        size->size() == 2u,
        "Vision decoder fusion block expects size=[height, width]");
    return {size->at(0), size->at(1)};
  }
  TORCH_CHECK(
      input.dim() == 4,
      "Vision decoder fusion block expects rank-4 input for scale_factor=2");
  return {input.size(2) * 2, input.size(3) * 2};
}

size_t vision_decoder_fusion_block_scratch_bytes(
    const Tensor& input,
    const std::optional<Tensor>& skip,
    const std::vector<int64_t>& target_sizes) {
  if (
      !input.defined() || input.scalar_type() != kFloat || input.dim() != 4 ||
      target_sizes.size() != 2u) {
    return 0u;
  }

  size_t total_bytes = 0u;
  const auto append_slice = [&](IntArrayRef sizes) {
    total_bytes = align_up_size(total_bytes, 256u);
    total_bytes += buffer_descriptor_nbytes(sizes, kFloat);
  };

  if (skip.has_value() && skip->defined()) {
    append_slice(skip->sizes());
    append_slice(skip->sizes());
    append_slice(skip->sizes());
    append_slice(skip->sizes());
    append_slice(input.sizes());
  }

  append_slice(input.sizes());
  append_slice(input.sizes());
  append_slice(input.sizes());
  append_slice(input.sizes());
  append_slice(
      {input.size(0), input.size(1), target_sizes[0], target_sizes[1]});
  return total_bytes;
}

Tensor make_scratch_buffer_alias(
    const utils::ScratchArena& arena,
    const utils::VulkanScratchSlice& slice,
    IntArrayRef sizes,
    const ScalarType dtype) {
  const size_t required_bytes = buffer_descriptor_nbytes(sizes, dtype);
  TORCH_CHECK(
      required_bytes <= slice.size_bytes,
      "Scratch buffer alias requested ",
      required_bytes,
      " bytes from a slice sized for ",
      slice.size_bytes,
      " bytes");

  const int64_t element_size =
      static_cast<int64_t>(c10::elementSize(dtype));
  TORCH_CHECK(
      element_size > 0,
      "Scratch buffer alias requires a concrete element size");
  TORCH_CHECK(
      slice.offset_bytes % static_cast<size_t>(element_size) == 0u &&
          arena.size_bytes() % static_cast<size_t>(element_size) == 0u,
      "Scratch buffer alias requires byte-aligned offsets for dtype ",
      dtype);

  const int64_t storage_offset =
      static_cast<int64_t>(slice.offset_bytes / static_cast<size_t>(element_size));
  const int64_t buffer_length_override =
      static_cast<int64_t>(arena.size_bytes() / static_cast<size_t>(element_size));
  const api::ExecutionLayout execution_layout =
      slice.offset_bytes == 0u ? api::ExecutionLayout::BUFFER_DIRECT
                               : api::ExecutionLayout::BUFFER_VIEW;
  return utils::make_typed_buffer_metadata_view(
      arena.storage(),
      dtype,
      sizes,
      calc_contiguous_strides(sizes),
      calc_width_packed_buffer_strides(sizes),
      storage_offset,
      buffer_length_override,
      execution_layout);
}

std::pair<utils::VulkanScratchSlice, Tensor> reserve_scratch_buffer_tensor(
    utils::ScratchArena& arena,
    IntArrayRef sizes,
    const ScalarType dtype) {
  const size_t required_bytes = buffer_descriptor_nbytes(sizes, dtype);
  const utils::VulkanScratchSlice slice = arena.reserve(
      required_bytes,
      std::max<uint32_t>(
          arena.alignment(),
          static_cast<uint32_t>(std::max<int64_t>(
              1, static_cast<int64_t>(c10::elementSize(dtype))))));
  return {slice, make_scratch_buffer_alias(arena, slice, sizes, dtype)};
}

Tensor prepare_buffer_attention_tensor(const Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_vulkan(),
      "Vision attention workspace expects Vulkan tensors");
  const vTensor& v_tensor = convert(tensor);
  if (
      v_tensor.storage_type() == api::StorageType::BUFFER &&
      v_tensor.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
      utils::supports_buffer_view_fast_path(v_tensor)) {
    return utils::mark_tensor_execution(
        tensor, utils::resolve_buffer_execution_layout(v_tensor));
  }

  Tensor buffer_tensor = utils::ensure_buffer_storage(
      tensor, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  return utils::mark_tensor_execution(
      buffer_tensor,
      utils::resolve_buffer_execution_layout(convert(buffer_tensor)));
}

Tensor prepare_decoder_buffer_tensor(const Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_vulkan(),
      "Vision decoder fusion block expects Vulkan tensors");
  const vTensor& v_tensor = convert(tensor);
  if (
      v_tensor.storage_type() == api::StorageType::BUFFER &&
      v_tensor.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
      utils::supports_buffer_view_fast_path(v_tensor)) {
    return utils::mark_tensor_execution(
        tensor, utils::resolve_buffer_execution_layout(v_tensor), false);
  }

  Tensor buffer_tensor = utils::ensure_buffer_storage(
      tensor, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  return utils::mark_tensor_execution(
      buffer_tensor,
      utils::resolve_buffer_execution_layout(convert(buffer_tensor)),
      false);
}

int64_t vision_decoder_out_channels(
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& context) {
  const auto& logical_weight_sizes =
      context->out_conv_context()->packed_weight().logical_weight_sizes();
  TORCH_CHECK(
      logical_weight_sizes.size() == 4u,
      "Vision decoder fusion block expects rank-4 out_conv weights");
  return logical_weight_sizes[0];
}

struct VisionDecoderRunOutputs final {
  Tensor skip_relu_output;
  Tensor skip_conv1_output;
  Tensor skip_conv2_output;
  Tensor skip_res_output;
  Tensor main_input_output;
  Tensor main_relu_output;
  Tensor main_conv1_output;
  Tensor main_conv2_output;
  Tensor main_res_output;
  Tensor upsample_output;
  Tensor out_conv_output;
};

utils::VisionDecoderInferenceGraph prime_vision_decoder_graph(
    const Tensor& input,
    const utils::VulkanRuntimePolicy& runtime_policy,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& context) {
  if (
      !has_explicit_runtime_capture_label() ||
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionDecoder) {
    return {};
  }

  return utils::lookup_or_create_labeled_vision_decoder_inference_graph(
      vision_decoder_graph_label(context->allocation_label()),
      input.scalar_type(),
      runtime_policy.execution_program_plan->persistent);
}

utils::VisionBackboneInferenceGraph prime_vision_backbone_graph(
    const Tensor& input,
    const utils::VulkanRuntimePolicy& runtime_policy,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context) {
  if (
      !has_explicit_runtime_capture_label() ||
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionBackbone) {
    return {};
  }

  return utils::lookup_or_create_labeled_vision_backbone_inference_graph(
      vision_backbone_graph_label(context->allocation_label()),
      input.scalar_type(),
      runtime_policy.execution_program_plan->persistent);
}

utils::VisionDecoderProgram prime_vision_decoder_program(
    const Tensor& input,
    const std::optional<Tensor>& skip,
    IntArrayRef target_sizes,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& context,
    const utils::VulkanRuntimePolicy& runtime_policy,
    const bool use_external_scratch,
    const bool allocate_intermediate_outputs = true) {
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionDecoder) {
    return {};
  }

  const std::optional<utils::VulkanScratchArenaSpec> scratch_spec =
      !use_external_scratch && runtime_policy.scratch_arena_plan.has_value()
          ? [&]() -> std::optional<utils::VulkanScratchArenaSpec> {
              const auto requested_bytes = vision_decoder_fusion_block_scratch_bytes(
                  input, skip, target_sizes.vec());
              if (
                  requested_bytes == 0u ||
                  !runtime_policy.scratch_arena_plan->prefer_buffer_storage) {
                return std::nullopt;
              }
              return utils::VulkanScratchArenaSpec{
                  kByte,
                  std::max(
                      requested_bytes,
                      runtime_policy.scratch_arena_plan->min_arena_bytes),
                  runtime_policy.scratch_arena_plan->alignment,
                  api::ExecutionLayout::BUFFER_DIRECT,
                  api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
                  api::StorageType::BUFFER,
                  runtime_policy.scratch_arena_plan->prefer_reusable_arena,
              };
            }()
          : std::nullopt;

  return utils::lookup_or_create_labeled_vision_decoder_program(
      vision_decoder_program_label(context->allocation_label(), context.get()),
      input.sizes(),
      skip.has_value() ? std::optional<std::vector<int64_t>>(skip->sizes().vec())
                       : std::nullopt,
      target_sizes,
      vision_decoder_out_channels(context),
      scratch_spec,
      *runtime_policy.execution_program_plan,
      allocate_intermediate_outputs);
}

Tensor run_attention_with_workspace_fallback(
    const Tensor& query_arg,
    const Tensor& key_arg,
    const Tensor& value_arg,
    utils::VisionBackboneProgram* const vision_program,
    utils::ScratchArena* const scratch_override = nullptr) {
  const auto fallback = [&](const Tensor& query,
                            const Tensor& key,
                            const Tensor& value) -> Tensor {
    return at::scaled_dot_product_attention(
        query,
        key,
        value,
        std::nullopt,
        0.0,
        false,
        std::optional<double>(1.0),
        false);
  };

  const auto attention_policy = utils::build_vulkan_attention_policy(
      std::nullopt,
      /*is_causal=*/false,
      /*enable_gqa=*/false,
      /*use_kv_cache=*/false,
      /*cache_has_previous_state=*/false);
  const auto attention_runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_attention_request(
          attention_policy,
          query_arg,
          key_arg,
          value_arg,
          utils::VulkanTensorRole::Input));
  if (
      attention_runtime_policy.attention_execution_strategy ==
          utils::VulkanAttentionExecutionStrategy::RuntimeProgram &&
      attention_runtime_policy.execution_program_plan.has_value() &&
      attention_runtime_policy.execution_program_plan->kind ==
          utils::VulkanExecutionProgramKind::AttentionRuntime) {
    utils::log_vulkan_op_hit(
        "aten::vision_attention.runtime_program_dispatch");
    if (vision_program && vision_program->defined()) {
      return run_attention_runtime_buffer_math_program_bridge(
          query_arg, key_arg, value_arg);
    }
    return fallback(query_arg, key_arg, value_arg);
  }

  utils::ScratchArena* scratch_arena = scratch_override;
  if (
      !scratch_arena && vision_program && vision_program->defined() &&
      vision_program->scratch_arena().has_value()) {
    scratch_arena = &(*vision_program->scratch_arena());
  }
  if (!scratch_arena) {
    return fallback(query_arg, key_arg, value_arg);
  }

  Tensor query = prepare_buffer_attention_tensor(query_arg);
  Tensor key = prepare_buffer_attention_tensor(key_arg);
  Tensor value = prepare_buffer_attention_tensor(value_arg);
  if (
      query.scalar_type() != kFloat || key.scalar_type() != kFloat ||
      value.scalar_type() != kFloat || query.dim() != 3 || key.dim() != 3 ||
      value.dim() != 3 || query.size(0) != key.size(0) ||
      query.size(0) != value.size(0) || query.size(2) != key.size(2) ||
      key.size(1) != value.size(1)) {
    return fallback(query, key, value);
  }

  const vTensor& v_query = convert(query);
  const vTensor& v_key = convert(key);
  const vTensor& v_value = convert(value);
  if (
      v_query.storage_type() != api::StorageType::BUFFER ||
      v_key.storage_type() != api::StorageType::BUFFER ||
      v_value.storage_type() != api::StorageType::BUFFER ||
      !utils::supports_buffer_view_fast_path(v_query) ||
      !utils::supports_buffer_view_fast_path(v_key) ||
      !utils::supports_buffer_view_fast_path(v_value)) {
    return fallback(query, key, value);
  }

  const std::vector<int64_t> scores_sizes{
      query.size(0),
      query.size(1),
      key.size(1),
  };
  const std::vector<int64_t> output_sizes{
      query.size(0),
      query.size(1),
      value.size(2),
  };
  auto [scores_slice, scores_output] =
      reserve_scratch_buffer_tensor(*scratch_arena, scores_sizes, kFloat);
  auto [probs_slice, probs_output] =
      reserve_scratch_buffer_tensor(*scratch_arena, scores_sizes, kFloat);
  auto [context_slice, context_output] =
      reserve_scratch_buffer_tensor(*scratch_arena, output_sizes, kFloat);
  (void)scores_slice;
  (void)probs_slice;
  (void)context_slice;

  Tensor key_t = prepare_buffer_attention_tensor(key.transpose(1, 2));
  Tensor scores = bmm_buffer_out_vulkan(query, key_t, scores_output);
  Tensor probs = softmax_buffer_lastdim_out_vulkan(scores, probs_output);
  return bmm_buffer_out_vulkan(probs, value, context_output);
}

utils::VisionBackboneProgram prime_vision_backbone_program(
    const Tensor& input,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context,
    const utils::VulkanRuntimePolicy& runtime_policy,
    const bool use_external_scratch) {
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionBackbone) {
    return {};
  }

  const int64_t batch_size = input.dim() == 2 ? 1 : input.size(0);
  const int64_t token_count = input.dim() == 2 ? input.size(0) : input.size(1);
  const int64_t embed_dim = input.size(-1);
  const int64_t hidden_dim = vision_block_hidden_dim(context);
  const std::optional<utils::VulkanScratchArenaSpec> scratch_spec =
      !use_external_scratch && runtime_policy.scratch_arena_plan.has_value()
          ? [&]() -> std::optional<utils::VulkanScratchArenaSpec> {
              const auto requested_bytes = vision_attention_scratch_bytes(
                  batch_size,
                  token_count,
                  embed_dim,
                  context->num_heads(),
                  input.scalar_type(),
                  context->qkv_bias().defined(),
                  std::max<uint32_t>(
                      runtime_policy.scratch_arena_plan->alignment,
                      static_cast<uint32_t>(std::max<int64_t>(
                          1, static_cast<int64_t>(c10::elementSize(kFloat))))));
              if (
                  requested_bytes == 0u ||
                  !runtime_policy.scratch_arena_plan->prefer_buffer_storage) {
                return std::nullopt;
              }
              return utils::VulkanScratchArenaSpec{
                  kByte,
                  std::max(
                      requested_bytes,
                      runtime_policy.scratch_arena_plan->min_arena_bytes),
                  runtime_policy.scratch_arena_plan->alignment,
                  api::ExecutionLayout::BUFFER_DIRECT,
                  api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
                  api::StorageType::BUFFER,
                  runtime_policy.scratch_arena_plan->prefer_reusable_arena,
              };
            }()
          : std::nullopt;

  return utils::lookup_or_create_labeled_vision_backbone_program(
      vision_backbone_program_label(context->allocation_label(), context.get()),
      input.scalar_type(),
      batch_size,
      token_count,
      embed_dim,
      hidden_dim,
      context->num_heads(),
      scratch_spec,
      *runtime_policy.execution_program_plan);
}

VisionDecoderRunOutputs reserve_vision_decoder_graph_outputs(
    utils::ScratchArena& scratch_arena,
    const Tensor& input,
    const std::optional<Tensor>& skip,
    IntArrayRef target_sizes,
    const Tensor& out_conv_output) {
  VisionDecoderRunOutputs outputs;
  if (skip.has_value() && skip->defined()) {
    outputs.skip_relu_output =
        reserve_scratch_buffer_tensor(scratch_arena, skip->sizes(), kFloat).second;
    outputs.skip_conv1_output =
        reserve_scratch_buffer_tensor(scratch_arena, skip->sizes(), kFloat).second;
    outputs.skip_conv2_output =
        reserve_scratch_buffer_tensor(scratch_arena, skip->sizes(), kFloat).second;
    outputs.skip_res_output =
        reserve_scratch_buffer_tensor(scratch_arena, skip->sizes(), kFloat).second;
    outputs.main_input_output =
        reserve_scratch_buffer_tensor(scratch_arena, input.sizes(), kFloat).second;
  }

  outputs.main_relu_output =
      reserve_scratch_buffer_tensor(scratch_arena, input.sizes(), kFloat).second;
  outputs.main_conv1_output =
      reserve_scratch_buffer_tensor(scratch_arena, input.sizes(), kFloat).second;
  outputs.main_conv2_output =
      reserve_scratch_buffer_tensor(scratch_arena, input.sizes(), kFloat).second;
  outputs.main_res_output =
      reserve_scratch_buffer_tensor(scratch_arena, input.sizes(), kFloat).second;
  outputs.upsample_output = reserve_scratch_buffer_tensor(
                                scratch_arena,
                                {input.size(0), input.size(1), target_sizes[0], target_sizes[1]},
                                kFloat)
                                .second;
  outputs.out_conv_output = out_conv_output;
  return outputs;
}

bool can_use_decoder_replay(
    const Tensor& input,
    const std::optional<Tensor>& skip) {
  if (!input.defined() || !input.is_vulkan()) {
    return false;
  }
  const vTensor& v_input = convert(input);
  if (
      v_input.storage_type() != api::StorageType::BUFFER ||
      !v_input.has_direct_buffer_layout()) {
    return false;
  }
  if (skip.has_value() && skip->defined()) {
    const vTensor& v_skip = convert(*skip);
    if (
        v_skip.storage_type() != api::StorageType::BUFFER ||
        !v_skip.has_direct_buffer_layout()) {
      return false;
    }
  }
  return true;
}

Tensor run_vision_decoder_fusion_block_program(
    Tensor main_input,
    const std::optional<Tensor>& skip_tensor,
    IntArrayRef target_sizes,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& context,
    VisionDecoderRunOutputs outputs) {
  if (skip_tensor.has_value() && skip_tensor->defined()) {
    Tensor residual =
        relu_buffer_out_vulkan(*skip_tensor, outputs.skip_relu_output);
    residual = run_conv2d_context_out(
        residual,
        context->res1_conv1_context(),
        outputs.skip_conv1_output);
    residual =
        relu_buffer_out_vulkan(residual, outputs.skip_relu_output);
    residual =
        run_conv2d_context_out(
            residual,
            context->res1_conv2_context(),
            outputs.skip_conv2_output);
    residual = add_buffer_out_vulkan(
        residual, *skip_tensor, outputs.skip_res_output);
    main_input = add_buffer_out_vulkan(
        main_input, residual, outputs.main_input_output);
  }

  Tensor output =
      relu_buffer_out_vulkan(main_input, outputs.main_relu_output);
  output = run_conv2d_context_out(
      output,
      context->res2_conv1_context(),
      outputs.main_conv1_output);
  output =
      relu_buffer_out_vulkan(output, outputs.main_relu_output);
  output = run_conv2d_context_out(
      output,
      context->res2_conv2_context(),
      outputs.main_conv2_output);
  output = add_buffer_out_vulkan(
      output, main_input, outputs.main_res_output);
  output = upsample_bilinear2d_buffer_out_vulkan(
      output,
      target_sizes,
      context->align_corners(),
      std::nullopt,
      std::nullopt,
      outputs.upsample_output);
  return run_conv2d_context_out(
      output,
      context->out_conv_context(),
      outputs.out_conv_output);
}

VisionDecoderRunOutputs program_decoder_outputs(
    utils::VisionDecoderProgram& program) {
  return VisionDecoderRunOutputs{
      program.skip_relu_output(),
      program.skip_conv1_output(),
      program.skip_conv2_output(),
      program.skip_res_output(),
      program.main_input_output(),
      program.main_relu_output(),
      program.main_conv1_output(),
      program.main_conv2_output(),
      program.main_res_output(),
      program.upsample_output(),
      program.out_conv_output(),
  };
}

bool can_use_decoder_head_replay(
    const Tensor& layer1,
    const Tensor& layer2,
    const Tensor& layer3,
    const Tensor& layer4) {
  return layer1.defined() && layer1.is_vulkan() && layer2.defined() &&
      layer2.is_vulkan() && layer3.defined() && layer3.is_vulkan() &&
      layer4.defined() && layer4.is_vulkan();
}

void copy_tensor_for_replay(Tensor& dst, const Tensor& src) {
  if (dst.is_vulkan() && src.is_vulkan()) {
    const vTensor& v_dst = convert(dst);
    const vTensor& v_src = convert(src);
    if (
        v_dst.storage_type() == api::StorageType::BUFFER &&
        v_src.storage_type() == api::StorageType::BUFFER &&
        v_dst.has_direct_buffer_layout() && v_src.has_direct_buffer_layout()) {
      utils::copy_buffer_tensor_direct_(dst, src);
      return;
    }
  }
  dst.copy_(src);
}

Tensor run_vision_decoder_head_program(
    const Tensor& layer1,
    const Tensor& layer2,
    const Tensor& layer3,
    const Tensor& layer4,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderHeadContext>& context,
    utils::VisionDecoderProgram& refinenet4_program,
    utils::VisionDecoderProgram& refinenet3_program,
    utils::VisionDecoderProgram& refinenet2_program,
    utils::VisionDecoderProgram& refinenet1_program,
    Tensor& output_conv1_output,
    Tensor& upsample_output,
    Tensor& output_conv2_conv1_output,
    Tensor& output_conv2_relu1_output,
    Tensor& output_conv2_conv2_output,
    Tensor& output_slot) {
  const std::vector<int64_t> layer3_target{layer3.size(2), layer3.size(3)};
  const std::vector<int64_t> layer2_target{layer2.size(2), layer2.size(3)};
  const std::vector<int64_t> layer1_target{layer1.size(2), layer1.size(3)};
  const std::vector<int64_t> path1_target{
      layer1.size(2) * 2, layer1.size(3) * 2};

  Tensor path4 = run_vision_decoder_fusion_block_program(
      layer4,
      std::nullopt,
      layer3_target,
      context->refinenet4_context(),
      program_decoder_outputs(refinenet4_program));
  Tensor path3 = run_vision_decoder_fusion_block_program(
      path4,
      layer3,
      layer2_target,
      context->refinenet3_context(),
      program_decoder_outputs(refinenet3_program));
  Tensor path2 = run_vision_decoder_fusion_block_program(
      path3,
      layer2,
      layer1_target,
      context->refinenet2_context(),
      program_decoder_outputs(refinenet2_program));
  Tensor path1 = run_vision_decoder_fusion_block_program(
      path2,
      layer1,
      path1_target,
      context->refinenet1_context(),
      program_decoder_outputs(refinenet1_program));

  Tensor output = run_conv2d_context_out(
      path1, context->output_conv1_context(), output_conv1_output);
  output = upsample_bilinear2d_buffer_out_vulkan(
      output,
      output_size,
      context->align_corners(),
      std::nullopt,
      std::nullopt,
      upsample_output);
  output = run_conv2d_context_out(
      output,
      context->output_conv2_conv1_context(),
      output_conv2_conv1_output);
  output = relu_buffer_out_vulkan(output, output_conv2_relu1_output);
  output = run_conv2d_context_out(
      output,
      context->output_conv2_conv2_context(),
      output_conv2_conv2_output);
  return relu_buffer_out_vulkan(output, output_slot);
}

std::tuple<Tensor, Tensor, Tensor> reshape_qkv_for_attention(
    const Tensor& mixed_qkv,
    const int64_t batch_size,
    const int64_t token_count,
    const int64_t num_heads,
    const int64_t head_dim) {
  std::vector<Tensor> qkv = at::chunk(mixed_qkv, 3, 2);
  Tensor q =
      qkv[0].reshape({batch_size, token_count, num_heads, head_dim})
          .permute({0, 2, 1, 3})
          .reshape({batch_size * num_heads, token_count, head_dim});
  Tensor k =
      qkv[1].reshape({batch_size, token_count, num_heads, head_dim})
          .permute({0, 2, 1, 3})
          .reshape({batch_size * num_heads, token_count, head_dim});
  Tensor v =
      qkv[2].reshape({batch_size, token_count, num_heads, head_dim})
          .permute({0, 2, 1, 3})
          .reshape({batch_size * num_heads, token_count, head_dim});
  return std::make_tuple(std::move(q), std::move(k), std::move(v));
}

Tensor ensure_attention_merge_output_tensor(
    Tensor& output,
    const int64_t batch_size,
    const int64_t token_count,
    const int64_t embed_dim,
    const ScalarType dtype) {
  const std::vector<int64_t> output_sizes{
      batch_size * token_count,
      embed_dim,
  };
  bool needs_allocation = !output.defined() || !output.is_vulkan() ||
      output.scalar_type() != dtype ||
      !output.sizes().equals(IntArrayRef(output_sizes));
  if (!needs_allocation) {
    const vTensor& v_output = convert(output);
    needs_allocation =
        v_output.storage_type() != api::StorageType::BUFFER ||
        v_output.gpu_memory_layout() !=
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED ||
        !utils::supports_buffer_view_fast_path(v_output);
  }
  if (needs_allocation) {
    output = utils::mark_tensor_execution(
        convert(vTensor{
            api::context(),
            output_sizes,
            convert_dtype(dtype),
            api::StorageType::BUFFER,
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
        }),
        api::ExecutionLayout::BUFFER_DIRECT);
  } else {
    output = utils::mark_tensor_execution(
        output,
        utils::resolve_buffer_execution_layout(convert(output)));
  }
  return output;
}

Tensor merge_attention_heads_for_projection(
    const Tensor& attention_output_arg,
    const int64_t batch_size,
    const int64_t token_count,
    const int64_t num_heads,
    const int64_t head_dim,
    Tensor* output_opt = nullptr) {
  api::AllocationScope allocation_scope("attention_merge_heads");
  const int64_t batch_heads = batch_size * num_heads;
  const int64_t embed_dim = num_heads * head_dim;

  Tensor attention_output = attention_output_arg.is_vulkan()
      ? attention_output_arg
      : attention_output_arg.vulkan();
  {
    const vTensor& v_attention_output = convert(attention_output);
    if (
        v_attention_output.storage_type() == api::StorageType::BUFFER &&
        v_attention_output.gpu_memory_layout() ==
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED &&
        utils::supports_buffer_view_fast_path(v_attention_output)) {
      attention_output = utils::mark_tensor_execution(
          attention_output,
          utils::resolve_buffer_execution_layout(v_attention_output));
    } else {
      attention_output = utils::mark_tensor_execution(
          utils::ensure_buffer_storage(
              attention_output, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED),
          api::ExecutionLayout::BUFFER_DIRECT);
    }
  }

  TORCH_CHECK(
      attention_output.dim() == 3,
      "Vulkan attention head merge expects a rank-3 [B*H, T, D] tensor");
  TORCH_CHECK(
      attention_output.size(0) == batch_heads &&
          attention_output.size(1) == token_count &&
          attention_output.size(2) == head_dim,
      "Vulkan attention head merge received unexpected attention output sizes");

  vTensor& v_input = convert(attention_output);
  TORCH_CHECK(
      v_input.storage_type() == api::StorageType::BUFFER &&
          utils::supports_buffer_view_fast_path(v_input),
      "Vulkan attention head merge expects buffer-backed attention output");

  Tensor output_tensor = output_opt
      ? ensure_attention_merge_output_tensor(
            *output_opt, batch_size, token_count, embed_dim, attention_output.scalar_type())
      : utils::mark_tensor_execution(
            convert(vTensor{
                api::context(),
                {batch_size * token_count, embed_dim},
                convert_dtype(attention_output.scalar_type()),
                api::StorageType::BUFFER,
                api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
            }),
            api::ExecutionLayout::BUFFER_DIRECT);
  vTensor& v_output = convert(output_tensor);

  const struct Block final {
    int32_t head_dim;
    int32_t token_count;
    int32_t num_heads;
    int32_t batch_size;
  } block{
      api::utils::safe_downcast<int32_t>(head_dim),
      api::utils::safe_downcast<int32_t>(token_count),
      api::utils::safe_downcast<int32_t>(num_heads),
      api::utils::safe_downcast<int32_t>(batch_size),
  };

  api::UniformParamsBuffer params(api::context(), block);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(head_dim),
      api::utils::safe_downcast<uint32_t>(token_count),
      api::utils::safe_downcast<uint32_t>(batch_heads),
  };

  api::context()->submit_compute_job(
      VK_KERNEL(merge_attention_heads_buffer),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_output.buffer_metadata(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_input.buffer_metadata(),
      params.buffer());

  utils::log_vulkan_op_hit("aten::attention_merge_heads.buffer_native");
  return output_tensor;
}

Tensor run_attention_projection(
    const Tensor& input_2d,
    const int64_t batch_size,
    const int64_t token_count,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context,
    utils::VisionBackboneProgram* vision_program = nullptr,
    utils::ScratchArena* scratch_override = nullptr) {
  TORCH_CHECK(
      input_2d.dim() == 2,
      "Vision backbone attention projection expects flattened rank-2 input");

  const int64_t embed_dim = input_2d.size(-1);
  TORCH_CHECK(
      embed_dim % context->num_heads() == 0,
      "Vision backbone block context expects embed_dim divisible by num_heads");
  const int64_t head_dim = embed_dim / context->num_heads();
  utils::ScratchArena* scratch_arena = scratch_override;
  if (
      !scratch_arena && vision_program && vision_program->defined() &&
      vision_program->scratch_arena().has_value()) {
    scratch_arena = &(*vision_program->scratch_arena());
  }
  const bool use_program_scratch = scratch_arena != nullptr;
  Tensor attention_output;
  if (batch_size == 1) {
    const bool use_scratch_qkv_projection =
        use_program_scratch &&
        input_2d.scalar_type() == kFloat && context->qkv_bias().defined();

    std::optional<utils::VulkanScratchSlice> mixed_qkv_slice;
    Tensor mixed_qkv_output;
    if (use_scratch_qkv_projection) {
      auto scratch_qkv_output = reserve_scratch_buffer_tensor(
          *scratch_arena,
          {token_count, 3 * embed_dim},
          input_2d.scalar_type());
      mixed_qkv_slice = scratch_qkv_output.first;
      mixed_qkv_output = std::move(scratch_qkv_output.second);
    }

    Tensor mixed_qkv;
    if (vision_program && vision_program->defined()) {
      mixed_qkv = use_scratch_qkv_projection
          ? run_linear_context_out(
                input_2d, context->qkv_context(), mixed_qkv_output)
          : run_linear_context_out(
                input_2d, context->qkv_context(), vision_program->qkv_output());
    } else {
      mixed_qkv = run_linear_context(input_2d, context->qkv_context());
    }
    Tensor q;
    Tensor k;
    Tensor v;
    bool q_is_scaled = false;
    if (context->qkv_bias().defined()) {
      if (use_scratch_qkv_projection) {
        auto [q_slice, q_output] = reserve_scratch_buffer_tensor(
            *scratch_arena,
            {context->num_heads(), token_count, head_dim},
            kFloat);
        auto [k_slice, k_output] = reserve_scratch_buffer_tensor(
            *scratch_arena,
            {context->num_heads(), token_count, head_dim},
            kFloat);
        auto [v_slice, v_output] = reserve_scratch_buffer_tensor(
            *scratch_arena,
            {context->num_heads(), token_count, head_dim},
            kFloat);
        (void)q_slice;
        (void)k_slice;
        (void)v_slice;
        std::tie(q, k, v) = transform_bias_rescale_qkv_vulkan_out(
            mixed_qkv,
            context->qkv_bias(),
            context->num_heads(),
            q_output,
            k_output,
            v_output);
      } else {
        std::tie(q, k, v) = at::_transform_bias_rescale_qkv(
            mixed_qkv, context->qkv_bias(), context->num_heads());
      }
      q_is_scaled = true;
    } else {
      std::vector<Tensor> qkv = at::chunk(mixed_qkv, 3, 1);
      q = qkv[0].reshape({token_count, context->num_heads(), head_dim})
              .permute({1, 0, 2});
      k = qkv[1].reshape({token_count, context->num_heads(), head_dim})
              .permute({1, 0, 2});
      v = qkv[2].reshape({token_count, context->num_heads(), head_dim})
              .permute({1, 0, 2});
    }
    if (!q_is_scaled) {
      q = at::mul(
          q,
          static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim))));
    }
    attention_output = run_attention_with_workspace_fallback(
        q, k, v, vision_program, scratch_arena);
    Tensor scratch_merge_output;
    Tensor* merge_output_opt = nullptr;
    if (use_scratch_qkv_projection && mixed_qkv_slice.has_value()) {
      scratch_merge_output = make_scratch_buffer_alias(
          *scratch_arena,
          *mixed_qkv_slice,
          {batch_size * token_count, embed_dim},
          attention_output.scalar_type());
      merge_output_opt = &scratch_merge_output;
    } else if (use_program_scratch) {
      auto [merge_slice, merge_output] = reserve_scratch_buffer_tensor(
          *scratch_arena,
          {batch_size * token_count, embed_dim},
          attention_output.scalar_type());
      (void)merge_slice;
      scratch_merge_output = std::move(merge_output);
      merge_output_opt = &scratch_merge_output;
    } else if (vision_program && vision_program->defined()) {
      merge_output_opt = &vision_program->merge_output();
    }
    attention_output = merge_attention_heads_for_projection(
        attention_output,
        batch_size,
        token_count,
        context->num_heads(),
        head_dim,
        merge_output_opt);
    return vision_program && vision_program->defined()
        ? run_linear_context_out(
              attention_output,
              context->proj_context(),
              vision_program->proj_output())
        : run_linear_context(attention_output, context->proj_context());
  }

  Tensor mixed_qkv = vision_program && vision_program->defined()
      ? run_linear_context_out(
            input_2d, context->qkv_context(), vision_program->qkv_output())
      : run_linear_context(input_2d, context->qkv_context());
  if (context->qkv_bias().defined()) {
    mixed_qkv = mixed_qkv.add(context->qkv_bias());
  }
  mixed_qkv = mixed_qkv.reshape({batch_size, token_count, 3 * embed_dim});
  Tensor q;
  Tensor k;
  Tensor v;
  std::tie(q, k, v) = reshape_qkv_for_attention(
      mixed_qkv, batch_size, token_count, context->num_heads(), head_dim);
  q = at::mul(
      q,
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim))));
  attention_output = run_attention_with_workspace_fallback(
      q, k, v, vision_program, scratch_arena);
  Tensor scratch_merge_output;
  Tensor* merge_output_opt = nullptr;
  if (use_program_scratch) {
    auto [merge_slice, merge_output] = reserve_scratch_buffer_tensor(
        *scratch_arena,
        {batch_size * token_count, embed_dim},
        attention_output.scalar_type());
    (void)merge_slice;
    scratch_merge_output = std::move(merge_output);
    merge_output_opt = &scratch_merge_output;
  } else if (vision_program && vision_program->defined()) {
    merge_output_opt = &vision_program->merge_output();
  }
  attention_output = merge_attention_heads_for_projection(
      attention_output,
      batch_size,
      token_count,
      context->num_heads(),
      head_dim,
      merge_output_opt);
  return vision_program && vision_program->defined()
      ? run_linear_context_out(
            attention_output,
            context->proj_context(),
            vision_program->proj_output())
      : run_linear_context(attention_output, context->proj_context());
}

Tensor tokens_to_feature_map_fallback(
    const Tensor& input_arg,
    const int64_t height,
    const int64_t width) {
  Tensor input = input_arg;
  if (input.dim() == 2) {
    input = input.unsqueeze(0);
  }

  TORCH_CHECK(
      input.dim() == 3,
      "Vulkan tokens_to_feature_map expects a [N, C] or [B, N, C] tensor");
  TORCH_CHECK(
      input.size(1) == height * width,
      "Vulkan tokens_to_feature_map expected token count ",
      height * width,
      " but received ",
      input.size(1));

  Tensor output;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);

    Tensor cpu_input = input.is_vulkan() ? input.cpu() : input;
    output = cpu_input.reshape(
        {cpu_input.size(0), height, width, cpu_input.size(2)});
    output = output.permute({0, 3, 1, 2}).contiguous();
  }

  if (input_arg.is_vulkan()) {
    return output.vulkan();
  }
  return output;
}

Tensor feature_map_to_tokens_fallback(const Tensor& input_arg) {
  TORCH_CHECK(
      input_arg.dim() == 4,
      "Vulkan feature_map_to_tokens expects a [B, C, H, W] tensor");

  Tensor output;
  {
    c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
    c10::InferenceMode inference_mode_guard(false);

    Tensor cpu_input = input_arg.is_vulkan() ? input_arg.cpu() : input_arg;
    output = cpu_input.permute({0, 2, 3, 1})
                 .reshape(
                     {cpu_input.size(0),
                      cpu_input.size(2) * cpu_input.size(3),
                      cpu_input.size(1)})
                 .contiguous();
  }

  if (input_arg.is_vulkan()) {
    return output.vulkan();
  }
  return output;
}

Tensor run_vision_backbone_block_program(
    const Tensor& input,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context,
    utils::VisionBackboneProgram* const vision_program,
    utils::ScratchArena* const graph_scratch,
    Tensor* const output_slot = nullptr) {
  const bool use_2d_input = input.dim() == 2;
  const int64_t batch_size = use_2d_input ? 1 : input.size(0);
  const int64_t token_count = use_2d_input ? input.size(0) : input.size(1);
  const int64_t embed_dim = input.size(-1);
  const int64_t hidden_rows = batch_size * token_count;
  Tensor input_2d = use_2d_input ? input : input.reshape({hidden_rows, embed_dim});

  const std::array<int64_t, 1> normalized_shape = {embed_dim};
  Tensor attention_input = vision_program
      ? run_layernorm_context_out(
            input_2d,
            normalized_shape,
            context->norm1_context(),
            vision_program->norm1_output())
      : run_layernorm_context(input_2d, normalized_shape, context->norm1_context());
  Tensor attention_output = run_attention_projection(
      attention_input,
      batch_size,
      token_count,
      context,
      vision_program,
      graph_scratch);
  attention_output = maybe_apply_layerscale(attention_output, context->ls1_gamma());
  Tensor hidden_states = at::add(input_2d, attention_output);

  Tensor mlp_input = vision_program
      ? run_layernorm_context_out(
            hidden_states,
            normalized_shape,
            context->norm2_context(),
            vision_program->norm2_output())
      : run_layernorm_context(
            hidden_states, normalized_shape, context->norm2_context());
  Tensor mlp_output = vision_program
      ? run_linear_gelu_context_out(
            mlp_input, context->fc1_context(), vision_program->fc1_output())
      : run_linear_gelu_context(mlp_input, context->fc1_context());
  mlp_output = vision_program
      ? run_linear_context_out(
            mlp_output, context->fc2_context(), vision_program->fc2_output())
      : run_linear_context(mlp_output, context->fc2_context());
  mlp_output = maybe_apply_layerscale(mlp_output, context->ls2_gamma());

  if (output_slot && output_slot->defined() && hidden_states.scalar_type() == kFloat &&
      mlp_output.scalar_type() == kFloat) {
    Tensor add_output = use_2d_input
        ? *output_slot
        : output_slot->reshape({hidden_rows, embed_dim});
    (void)add_buffer_out_vulkan(hidden_states, mlp_output, add_output);
    return *output_slot;
  }

  Tensor output = at::add(hidden_states, mlp_output);
  if (!use_2d_input) {
    output = output.reshape({batch_size, token_count, embed_dim});
  }
  return output;
}

utils::ExecutionGraphReplayStep make_vision_backbone_replay_step(
    utils::VisionBackboneInferenceReplay backbone_replay,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& backbone_context,
    std::optional<utils::ScratchArena> backbone_graph_scratch) {
  const std::string execution_label = vision_backbone_execution_label(
      backbone_context->allocation_label(), backbone_context.get());
  return backbone_replay.phase_step(
      [backbone_replay,
       backbone_context,
       backbone_graph_scratch,
       execution_label]() mutable {
        api::RuntimeLabelScope runtime_scope(execution_label);
        if (backbone_graph_scratch.has_value()) {
          backbone_graph_scratch->reset();
        }
        (void)run_vision_backbone_block_program(
            backbone_replay.input_slot(),
            backbone_context,
            &backbone_replay.program(),
            backbone_graph_scratch.has_value() ? &(*backbone_graph_scratch)
                                              : nullptr,
            &backbone_replay.output_slot());
      });
}

utils::ExecutionGraphReplayStep make_chained_vision_backbone_replay_step(
    utils::VisionBackboneInferenceReplay previous_replay,
    utils::VisionBackboneInferenceReplay backbone_replay,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& backbone_context,
    std::optional<utils::ScratchArena> backbone_graph_scratch) {
  const std::string execution_label = vision_backbone_execution_label(
      backbone_context->allocation_label(), backbone_context.get());
  return backbone_replay.phase_step(
      [previous_replay,
       backbone_replay,
       backbone_context,
       backbone_graph_scratch,
       execution_label]() mutable {
        api::RuntimeLabelScope runtime_scope(execution_label);
        if (backbone_graph_scratch.has_value()) {
          backbone_graph_scratch->reset();
        }
        (void)run_vision_backbone_block_program(
            previous_replay.output_slot(),
            backbone_context,
            &backbone_replay.program(),
            backbone_graph_scratch.has_value() ? &(*backbone_graph_scratch)
                                              : nullptr,
            &backbone_replay.output_slot());
      });
}

utils::ExecutionGraphReplayStep make_vision_decoder_replay_step(
    utils::VisionDecoderInferenceReplay decoder_replay,
    std::vector<int64_t> decoder_target_sizes,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& decoder_context) {
  const VisionDecoderRunOutputs decoder_replay_outputs{
      decoder_replay.program().skip_relu_output(),
      decoder_replay.program().skip_conv1_output(),
      decoder_replay.program().skip_conv2_output(),
      decoder_replay.program().skip_res_output(),
      decoder_replay.program().main_input_output(),
      decoder_replay.program().main_relu_output(),
      decoder_replay.program().main_conv1_output(),
      decoder_replay.program().main_conv2_output(),
      decoder_replay.program().main_res_output(),
      decoder_replay.program().upsample_output(),
      decoder_replay.program().out_conv_output(),
  };
  return decoder_replay.phase_step(
      [decoder_replay,
       decoder_target_sizes = std::move(decoder_target_sizes),
       decoder_context,
       decoder_replay_outputs]() mutable {
        (void)run_vision_decoder_fusion_block_program(
            decoder_replay.input_slot(),
            decoder_replay.skip_slot(),
            decoder_target_sizes,
            decoder_context,
            decoder_replay_outputs);
      });
}

} // namespace

VisionBackboneBlockContext::VisionBackboneBlockContext(
    const Tensor& norm1_weight,
    const Tensor& norm1_bias,
    const double norm1_eps,
    const Tensor& qkv_weight,
    const std::optional<Tensor>& qkv_bias,
    const int64_t num_heads,
    const Tensor& proj_weight,
    const std::optional<Tensor>& proj_bias,
    const std::optional<Tensor>& ls1_gamma,
    const Tensor& norm2_weight,
    const Tensor& norm2_bias,
    const double norm2_eps,
    const Tensor& fc1_weight,
    const std::optional<Tensor>& fc1_bias,
    const Tensor& fc2_weight,
    const std::optional<Tensor>& fc2_bias,
    const std::optional<Tensor>& ls2_gamma,
    std::string allocation_label)
    : allocation_label_(std::move(allocation_label)),
      norm1_context_(make_layernorm_context(
          norm1_weight,
          norm1_bias,
          norm1_eps,
          child_label(allocation_label_, "norm1"))),
      qkv_context_(
          make_qkv_context(qkv_weight, child_label(allocation_label_, "qkv"))),
      qkv_bias_(move_optional_to_vulkan_buffer(qkv_bias)),
      num_heads_(num_heads),
      proj_context_(make_linear_context(
          proj_weight,
          proj_bias,
          child_label(allocation_label_, "proj"))),
      ls1_gamma_(move_optional_to_vulkan_buffer(ls1_gamma)),
      norm2_context_(make_layernorm_context(
          norm2_weight,
          norm2_bias,
          norm2_eps,
          child_label(allocation_label_, "norm2"))),
      fc1_context_(
          make_linear_context(fc1_weight, fc1_bias, child_label(allocation_label_, "fc1"))),
      fc2_context_(
          make_linear_context(fc2_weight, fc2_bias, child_label(allocation_label_, "fc2"))),
      ls2_gamma_(move_optional_to_vulkan_buffer(ls2_gamma)) {
  unpacked_.reserve(Unpacked::NumArgs);
  unpacked_.emplace_back(norm1_weight.cpu());
  unpacked_.emplace_back(norm1_bias.cpu());
  unpacked_.emplace_back(norm1_eps);
  unpacked_.emplace_back(qkv_weight.cpu());
  if (qkv_bias.has_value()) {
    unpacked_.emplace_back(qkv_bias->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(num_heads_);
  unpacked_.emplace_back(proj_weight.cpu());
  if (proj_bias.has_value()) {
    unpacked_.emplace_back(proj_bias->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  if (ls1_gamma.has_value()) {
    unpacked_.emplace_back(ls1_gamma->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(norm2_weight.cpu());
  unpacked_.emplace_back(norm2_bias.cpu());
  unpacked_.emplace_back(norm2_eps);
  unpacked_.emplace_back(fc1_weight.cpu());
  if (fc1_bias.has_value()) {
    unpacked_.emplace_back(fc1_bias->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(fc2_weight.cpu());
  if (fc2_bias.has_value()) {
    unpacked_.emplace_back(fc2_bias->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  if (ls2_gamma.has_value()) {
    unpacked_.emplace_back(ls2_gamma->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(allocation_label_);
}

VisionBackboneBlockContext VisionBackboneBlockContext::pack(
    c10::impl::GenericList unpacked) {
  return VisionBackboneBlockContext(
      unpacked.get(Unpacked::Norm1Weight).toTensor(),
      unpacked.get(Unpacked::Norm1Bias).toTensor(),
      unpacked.get(Unpacked::Norm1Eps).toDouble(),
      unpacked.get(Unpacked::QkvWeight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::QkvBias),
      unpacked.get(Unpacked::NumHeads).toInt(),
      unpacked.get(Unpacked::ProjWeight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::ProjBias),
      get_optional_tensor(unpacked, Unpacked::Ls1Gamma),
      unpacked.get(Unpacked::Norm2Weight).toTensor(),
      unpacked.get(Unpacked::Norm2Bias).toTensor(),
      unpacked.get(Unpacked::Norm2Eps).toDouble(),
      unpacked.get(Unpacked::Fc1Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Fc1Bias),
      unpacked.get(Unpacked::Fc2Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Fc2Bias),
      get_optional_tensor(unpacked, Unpacked::Ls2Gamma),
      unpacked.get(Unpacked::Label).toStringRef());
}

c10::intrusive_ptr<VisionBackboneBlockContext>
create_vision_backbone_block_context(
    Tensor&& norm1_weight,
    Tensor&& norm1_bias,
    const double norm1_eps,
    Tensor&& qkv_weight,
    std::optional<Tensor>&& qkv_bias,
    const int64_t num_heads,
    Tensor&& proj_weight,
    std::optional<Tensor>&& proj_bias,
    std::optional<Tensor>&& ls1_gamma,
    Tensor&& norm2_weight,
    Tensor&& norm2_bias,
    const double norm2_eps,
    Tensor&& fc1_weight,
    std::optional<Tensor>&& fc1_bias,
    Tensor&& fc2_weight,
    std::optional<Tensor>&& fc2_bias,
    std::optional<Tensor>&& ls2_gamma,
    std::string label) {
  return c10::make_intrusive<VisionBackboneBlockContext>(
      norm1_weight,
      norm1_bias,
      norm1_eps,
      qkv_weight,
      qkv_bias,
      num_heads,
      proj_weight,
      proj_bias,
      ls1_gamma,
      norm2_weight,
      norm2_bias,
      norm2_eps,
      fc1_weight,
      fc1_bias,
      fc2_weight,
      fc2_bias,
      ls2_gamma,
      std::move(label));
}

Tensor run_vision_backbone_block_context(
    const Tensor& input_arg,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context) {
  TORCH_CHECK(
      input_arg.dim() == 2 || input_arg.dim() == 3,
      "Vision backbone block context expects rank-2 or rank-3 input");

  const Device output_device = input_arg.device();
  const ScalarType output_dtype = input_arg.scalar_type();
  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  utils::VulkanPlanningRequestScope planning_scope(
      utils::make_vulkan_vision_backbone_request());
  const auto runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_vision_backbone_request());
  auto vision_graph = prime_vision_backbone_graph(input, runtime_policy, context);
  std::optional<utils::ScratchArena> graph_scratch = std::nullopt;
  if (vision_graph.defined() && runtime_policy.scratch_arena_plan.has_value()) {
    const int64_t batch_size = input.dim() == 2 ? 1 : input.size(0);
    const int64_t token_count = input.dim() == 2 ? input.size(0) : input.size(1);
    const int64_t embed_dim = input.size(-1);
    const uint32_t scratch_alignment = std::max<uint32_t>(
        runtime_policy.scratch_arena_plan->alignment,
        static_cast<uint32_t>(
            std::max<int64_t>(1, static_cast<int64_t>(c10::elementSize(kFloat)))));
    const size_t requested_bytes = vision_attention_scratch_bytes(
        batch_size,
        token_count,
        embed_dim,
        context->num_heads(),
        input.scalar_type(),
        context->qkv_bias().defined(),
        scratch_alignment);
    if (
        requested_bytes > 0u &&
        runtime_policy.scratch_arena_plan->prefer_buffer_storage) {
      graph_scratch = vision_graph.ensure_shared_scratch(
          std::max(
              requested_bytes,
              runtime_policy.scratch_arena_plan->min_arena_bytes),
          scratch_alignment,
          runtime_policy.execution_program_plan.has_value() &&
              runtime_policy.execution_program_plan->persistent);
    }
  }

  if (graph_scratch.has_value()) {
    graph_scratch->reset();
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_backbone_block_context.graph");
  }
  const int64_t batch_size = input.dim() == 2 ? 1 : input.size(0);
  const int64_t token_count = input.dim() == 2 ? input.size(0) : input.size(1);
  const int64_t embed_dim = input.size(-1);
  const int64_t hidden_dim = vision_block_hidden_dim(context);
  const std::string backbone_program_label =
      vision_backbone_program_label(context->allocation_label(), context.get());
  std::optional<api::RuntimeLabelScope> execution_runtime_scope;
  if (has_explicit_runtime_capture_label()) {
    execution_runtime_scope.emplace(
        vision_backbone_execution_label(context->allocation_label(), context.get()));
  }

  if (
      vision_graph.defined() &&
      runtime_policy.execution_program_plan.has_value() &&
      input.scalar_type() == kFloat) {
    auto vision_replay = vision_graph.lookup_or_create_replay(
        backbone_program_label,
        input.sizes(),
        token_count,
        embed_dim,
        hidden_dim,
        context->num_heads(),
        *runtime_policy.execution_program_plan);
    if (vision_replay.defined()) {
      copy_tensor_for_replay(vision_replay.input_slot(), input);
      api::context()->flush_pending_cmds();

      if (!vision_replay.recorded()) {
        Tensor warmup_output = utils::create_buffer_tensor(
            vision_replay.output_slot().sizes(),
            vision_replay.output_slot().scalar_type(),
            /*persistent=*/output_device.type() == kVulkan);
        if (graph_scratch.has_value()) {
          graph_scratch->reset();
        }
        (void)run_vision_backbone_block_program(
            vision_replay.input_slot(),
            context,
            &vision_replay.program(),
            graph_scratch.has_value() ? &(*graph_scratch) : nullptr,
            &vision_replay.output_slot());
        copy_tensor_for_replay(warmup_output, vision_replay.output_slot());
        api::context()->flush_pending_cmds();
        vision_replay.replay().record([&]() {
          if (graph_scratch.has_value()) {
            graph_scratch->reset();
          }
          (void)run_vision_backbone_block_program(
              vision_replay.input_slot(),
              context,
              &vision_replay.program(),
              graph_scratch.has_value() ? &(*graph_scratch) : nullptr,
              &vision_replay.output_slot());
        });
        utils::log_vulkan_op_hit(
            "vulkan_prepack::run_vision_backbone_block_context.replay_warmup");
        utils::log_vulkan_op_hit(
            "vulkan_prepack::run_vision_backbone_block_context");
        return maybe_restore_tensor(warmup_output, output_device, output_dtype);
      }

      vision_replay.replay().submit();
      Tensor output = utils::create_buffer_tensor(
          vision_replay.output_slot().sizes(),
          vision_replay.output_slot().scalar_type(),
          /*persistent=*/output_device.type() == kVulkan);
      copy_tensor_for_replay(output, vision_replay.output_slot());
      utils::log_vulkan_op_hit(
          "vulkan_prepack::run_vision_backbone_block_context.replay");
      utils::log_vulkan_op_hit(
          "vulkan_prepack::run_vision_backbone_block_context");
      return maybe_restore_tensor(output, output_device, output_dtype);
    }
  }

  auto vision_program = vision_graph.defined()
      ? vision_graph.lookup_or_create_program(
            backbone_program_label,
            input.scalar_type(),
            batch_size,
            token_count,
            embed_dim,
            hidden_dim,
            context->num_heads(),
            *runtime_policy.execution_program_plan)
      : prime_vision_backbone_program(
            input, context, runtime_policy, graph_scratch.has_value());
  if (vision_program.defined()) {
    if (!graph_scratch.has_value() && vision_program.scratch_arena().has_value()) {
      vision_program.scratch_arena()->reset();
    }
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_backbone_block_context.program");
  }
  utils::VisionBackboneProgram* const vision_program_ptr =
      vision_program.defined() ? &vision_program : nullptr;

  Tensor output = run_vision_backbone_block_program(
      input,
      context,
      vision_program_ptr,
      graph_scratch.has_value() ? &(*graph_scratch) : nullptr);
  utils::log_vulkan_op_hit("vulkan_prepack::run_vision_backbone_block_context");
  return maybe_restore_tensor(output, output_device, output_dtype);
}

void prime_vision_backbone_block_context_graph(
    const Tensor& input_arg,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context) {
  if (!input_arg.defined() || !input_arg.is_vulkan() || !context) {
    return;
  }

  TORCH_CHECK(
      input_arg.dim() == 2 || input_arg.dim() == 3,
      "Vision backbone block graph priming expects rank-2 or rank-3 input");

  utils::VulkanPlanningRequestScope planning_scope(
      utils::make_vulkan_vision_backbone_request());
  const auto runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_vision_backbone_request());
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionBackbone) {
    return;
  }
  if (input_arg.scalar_type() != kFloat) {
    return;
  }

  auto vision_graph = prime_vision_backbone_graph(input_arg, runtime_policy, context);
  if (!vision_graph.defined()) {
    return;
  }

  if (runtime_policy.scratch_arena_plan.has_value()) {
    const int64_t batch_size = input_arg.dim() == 2 ? 1 : input_arg.size(0);
    const int64_t token_count =
        input_arg.dim() == 2 ? input_arg.size(0) : input_arg.size(1);
    const int64_t embed_dim = input_arg.size(-1);
    const uint32_t scratch_alignment = std::max<uint32_t>(
        runtime_policy.scratch_arena_plan->alignment,
        static_cast<uint32_t>(std::max<int64_t>(
            1, static_cast<int64_t>(c10::elementSize(kFloat)))));
    const size_t requested_bytes = vision_attention_scratch_bytes(
        batch_size,
        token_count,
        embed_dim,
        context->num_heads(),
        input_arg.scalar_type(),
        context->qkv_bias().defined(),
        scratch_alignment);
    if (
        requested_bytes > 0u &&
        runtime_policy.scratch_arena_plan->prefer_buffer_storage) {
      vision_graph.note_shared_scratch_requirement(
          std::max(
              requested_bytes,
              runtime_policy.scratch_arena_plan->min_arena_bytes),
          scratch_alignment,
          runtime_policy.execution_program_plan->persistent);
    }
  }

  (void)vision_graph.lookup_or_create_replay(
      vision_backbone_program_label(
          context->allocation_label(), context.get()),
      input_arg.sizes(),
      input_arg.dim() == 2 ? input_arg.size(0) : input_arg.size(1),
      input_arg.size(-1),
      vision_block_hidden_dim(context),
      context->num_heads(),
      *runtime_policy.execution_program_plan);
  utils::log_vulkan_op_hit(
      "vulkan_prepack::prime_vision_backbone_block_context_graph");
}

VisionDecoderFusionBlockContext::VisionDecoderFusionBlockContext(
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
    const bool align_corners,
    std::string allocation_label)
    : allocation_label_(std::move(allocation_label)),
      align_corners_(align_corners),
      res1_conv1_context_(make_conv2d_context(
          res1_conv1_weight,
          res1_conv1_bias,
          {1, 1},
          {1, 1})),
      res1_conv2_context_(make_conv2d_context(
          res1_conv2_weight,
          res1_conv2_bias,
          {1, 1},
          {1, 1})),
      res2_conv1_context_(make_conv2d_context(
          res2_conv1_weight,
          res2_conv1_bias,
          {1, 1},
          {1, 1})),
      res2_conv2_context_(make_conv2d_context(
          res2_conv2_weight,
          res2_conv2_bias,
          {1, 1},
          {1, 1})),
      out_conv_context_(make_conv2d_context(
          out_conv_weight,
          out_conv_bias,
          {1, 1},
          {0, 0})) {
  unpacked_.reserve(Unpacked::NumArgs);
  unpacked_.emplace_back(res1_conv1_weight.cpu());
  if (res1_conv1_bias.has_value()) {
    unpacked_.emplace_back(res1_conv1_bias->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(res1_conv2_weight.cpu());
  if (res1_conv2_bias.has_value()) {
    unpacked_.emplace_back(res1_conv2_bias->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(res2_conv1_weight.cpu());
  if (res2_conv1_bias.has_value()) {
    unpacked_.emplace_back(res2_conv1_bias->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(res2_conv2_weight.cpu());
  if (res2_conv2_bias.has_value()) {
    unpacked_.emplace_back(res2_conv2_bias->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(out_conv_weight.cpu());
  if (out_conv_bias.has_value()) {
    unpacked_.emplace_back(out_conv_bias->cpu());
  } else {
    unpacked_.emplace_back(std::optional<Tensor>{});
  }
  unpacked_.emplace_back(align_corners_);
  unpacked_.emplace_back(allocation_label_);
}

VisionDecoderFusionBlockContext VisionDecoderFusionBlockContext::pack(
    c10::impl::GenericList unpacked) {
  return VisionDecoderFusionBlockContext(
      unpacked.get(Unpacked::Res1Conv1Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Res1Conv1Bias),
      unpacked.get(Unpacked::Res1Conv2Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Res1Conv2Bias),
      unpacked.get(Unpacked::Res2Conv1Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Res2Conv1Bias),
      unpacked.get(Unpacked::Res2Conv2Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Res2Conv2Bias),
      unpacked.get(Unpacked::OutConvWeight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::OutConvBias),
      unpacked.get(Unpacked::AlignCorners).toBool(),
      unpacked.get(Unpacked::Label).toStringRef());
}

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
    const bool align_corners,
    std::string label) {
  return c10::make_intrusive<VisionDecoderFusionBlockContext>(
      res1_conv1_weight,
      res1_conv1_bias,
      res1_conv2_weight,
      res1_conv2_bias,
      res2_conv1_weight,
      res2_conv1_bias,
      res2_conv2_weight,
      res2_conv2_bias,
      out_conv_weight,
      out_conv_bias,
      align_corners,
      std::move(label));
}

Tensor run_vision_decoder_fusion_block_context(
    const Tensor& input_arg,
    const std::optional<Tensor>& skip_arg,
    const std::optional<std::vector<int64_t>>& size,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& context) {
  TORCH_CHECK(
      input_arg.dim() == 4,
      "Vision decoder fusion block context expects rank-4 input");

  const Device output_device = input_arg.device();
  const ScalarType output_dtype = input_arg.scalar_type();
  const auto fallback =
      [&](const Tensor& input_tensor,
          const std::optional<Tensor>& skip_tensor) -> Tensor {
    Tensor main_input = input_tensor;
    if (skip_tensor.has_value() && skip_tensor->defined()) {
      Tensor residual = at::relu(*skip_tensor);
      residual = run_conv2d_context(residual, context->res1_conv1_context());
      residual = at::relu(residual);
      residual = run_conv2d_context(residual, context->res1_conv2_context());
      main_input = at::add(input_tensor, at::add(residual, *skip_tensor));
    }

    Tensor output = at::relu(main_input);
    output = run_conv2d_context(output, context->res2_conv1_context());
    output = at::relu(output);
    output = run_conv2d_context(output, context->res2_conv2_context());
    output = at::add(output, main_input);
    output = at::upsample_bilinear2d(
        output,
        resolve_decoder_target_sizes(input_tensor, size),
        context->align_corners(),
        std::nullopt,
        std::nullopt);
    output = run_conv2d_context(output, context->out_conv_context());
    return maybe_restore_tensor(output, output_device, output_dtype);
  };

  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();
  std::optional<Tensor> skip =
      (skip_arg.has_value() && skip_arg->defined())
      ? std::optional<Tensor>(skip_arg->is_vulkan() ? *skip_arg : skip_arg->vulkan())
      : std::nullopt;

  const std::vector<int64_t> target_sizes =
      resolve_decoder_target_sizes(input, size);
  utils::VulkanPlanningRequestScope planning_scope(
      utils::make_vulkan_vision_decoder_request());

  if (input.scalar_type() != kFloat || input.device().type() != kVulkan) {
    return fallback(input, skip);
  }

  Tensor main_input = prepare_decoder_buffer_tensor(input);
  if (main_input.dim() != 4) {
    return fallback(input, skip);
  }

  std::optional<Tensor> skip_tensor = std::nullopt;
  if (skip.has_value() && skip->defined()) {
    skip_tensor = prepare_decoder_buffer_tensor(*skip);
  }

  const auto runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_vision_decoder_request());
  auto vision_graph =
      prime_vision_decoder_graph(main_input, runtime_policy, context);
  if (
      vision_graph.defined() &&
      runtime_policy.execution_program_plan.has_value() &&
      can_use_decoder_replay(main_input, skip_tensor)) {
    auto vision_replay = vision_graph.lookup_or_create_replay(
        vision_decoder_program_label(
            context->allocation_label(), context.get()),
        main_input.sizes(),
        skip_tensor.has_value()
            ? std::optional<std::vector<int64_t>>(skip_tensor->sizes().vec())
            : std::nullopt,
        target_sizes,
        vision_decoder_out_channels(context),
        *runtime_policy.execution_program_plan);
    if (vision_replay.defined()) {
      auto& replay_program = vision_replay.program();
      const VisionDecoderRunOutputs replay_outputs{
          replay_program.skip_relu_output(),
          replay_program.skip_conv1_output(),
          replay_program.skip_conv2_output(),
          replay_program.skip_res_output(),
          replay_program.main_input_output(),
          replay_program.main_relu_output(),
          replay_program.main_conv1_output(),
          replay_program.main_conv2_output(),
          replay_program.main_res_output(),
          replay_program.upsample_output(),
          replay_program.out_conv_output(),
      };
      utils::copy_buffer_tensor_direct_(
          vision_replay.input_slot(), main_input);
      if (skip_tensor.has_value() && skip_tensor->defined()) {
        TORCH_INTERNAL_ASSERT(
            vision_replay.skip_slot().has_value(),
            "Vision decoder replay expected a skip slot");
        utils::copy_buffer_tensor_direct_(
            *vision_replay.skip_slot(), *skip_tensor);
      }
      api::context()->flush_pending_cmds();

      if (!vision_replay.recorded()) {
        Tensor warmup_output = utils::create_buffer_tensor(
            vision_replay.output_slot().sizes(),
            vision_replay.output_slot().scalar_type(),
            /*persistent=*/false);
        utils::copy_buffer_tensor_direct_(
            warmup_output,
            run_vision_decoder_fusion_block_program(
                vision_replay.input_slot(),
                vision_replay.skip_slot(),
                target_sizes,
                context,
                replay_outputs));
        api::context()->flush_pending_cmds();
        vision_replay.replay().record([&]() {
          (void)run_vision_decoder_fusion_block_program(
              vision_replay.input_slot(),
              vision_replay.skip_slot(),
              target_sizes,
              context,
              replay_outputs);
        });
        utils::log_vulkan_op_hit(
            "vulkan_prepack::run_vision_decoder_fusion_block_context.replay_warmup");
        utils::log_vulkan_op_hit(
            "vulkan_prepack::run_vision_decoder_fusion_block_context");
        return maybe_restore_tensor(warmup_output, output_device, output_dtype);
      }

      vision_replay.replay().submit();

      Tensor output = utils::create_buffer_tensor(
          vision_replay.output_slot().sizes(),
          vision_replay.output_slot().scalar_type(),
          /*persistent=*/false);
      utils::copy_buffer_tensor_direct_(
          output, vision_replay.output_slot());
      utils::log_vulkan_op_hit(
          "vulkan_prepack::run_vision_decoder_fusion_block_context.replay");
      utils::log_vulkan_op_hit(
          "vulkan_prepack::run_vision_decoder_fusion_block_context");
      return maybe_restore_tensor(output, output_device, output_dtype);
    }
  }
  std::optional<utils::ScratchArena> graph_scratch = std::nullopt;
  if (vision_graph.defined() && runtime_policy.scratch_arena_plan.has_value()) {
    const uint32_t scratch_alignment = std::max<uint32_t>(
        runtime_policy.scratch_arena_plan->alignment,
        static_cast<uint32_t>(std::max<int64_t>(
            1, static_cast<int64_t>(c10::elementSize(kFloat)))));
    const size_t requested_bytes = vision_decoder_fusion_block_scratch_bytes(
        main_input, skip_tensor, target_sizes);
    if (
        requested_bytes > 0u &&
        runtime_policy.scratch_arena_plan->prefer_buffer_storage) {
      graph_scratch = vision_graph.ensure_shared_scratch(
          std::max(
              requested_bytes,
              runtime_policy.scratch_arena_plan->min_arena_bytes),
          scratch_alignment,
          runtime_policy.execution_program_plan.has_value() &&
              runtime_policy.execution_program_plan->persistent);
    }
  }

  auto vision_program = vision_graph.defined()
      ? vision_graph.lookup_or_create_program(
            vision_decoder_program_label(
                context->allocation_label(), context.get()),
            main_input.sizes(),
            skip_tensor.has_value()
                ? std::optional<std::vector<int64_t>>(skip_tensor->sizes().vec())
                : std::nullopt,
            target_sizes,
            vision_decoder_out_channels(context),
            !graph_scratch.has_value(),
            *runtime_policy.execution_program_plan)
      : prime_vision_decoder_program(
            main_input,
            skip_tensor,
            target_sizes,
            context,
            runtime_policy,
            graph_scratch.has_value(),
            !graph_scratch.has_value());
  auto& program_ref = vision_program;
  if (!program_ref.defined()) {
    return fallback(main_input, skip_tensor);
  }
  if (graph_scratch.has_value()) {
    graph_scratch->reset();
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_decoder_fusion_block_context.graph");
  } else if (program_ref.scratch_arena().has_value()) {
    program_ref.scratch_arena()->reset();
  }

  VisionDecoderRunOutputs outputs = graph_scratch.has_value()
      ? reserve_vision_decoder_graph_outputs(
            *graph_scratch,
            main_input,
            skip_tensor,
            target_sizes,
            program_ref.out_conv_output())
      : VisionDecoderRunOutputs{
            program_ref.skip_relu_output(),
            program_ref.skip_conv1_output(),
            program_ref.skip_conv2_output(),
            program_ref.skip_res_output(),
            program_ref.main_input_output(),
            program_ref.main_relu_output(),
            program_ref.main_conv1_output(),
            program_ref.main_conv2_output(),
            program_ref.main_res_output(),
            program_ref.upsample_output(),
            program_ref.out_conv_output(),
        };
  Tensor output = run_vision_decoder_fusion_block_program(
      main_input,
      skip_tensor,
      target_sizes,
      context,
      outputs);
  utils::log_vulkan_op_hit("vulkan_prepack::run_vision_decoder_fusion_block_context");
  return maybe_restore_tensor(output, output_device, output_dtype);
}

void prime_vision_decoder_fusion_block_context_graph(
    const Tensor& input_arg,
    const std::optional<Tensor>& skip_arg,
    const std::optional<std::vector<int64_t>>& size,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& context) {
  if (!input_arg.defined() || !input_arg.is_vulkan() || !context) {
    return;
  }

  TORCH_CHECK(
      input_arg.dim() == 4,
      "Vision decoder fusion block graph priming expects rank-4 input");
  if (input_arg.scalar_type() != kFloat) {
    return;
  }

  std::optional<Tensor> skip =
      (skip_arg.has_value() && skip_arg->defined()) ? skip_arg : std::nullopt;
  const std::vector<int64_t> target_sizes =
      resolve_decoder_target_sizes(input_arg, size);

  utils::VulkanPlanningRequestScope planning_scope(
      utils::make_vulkan_vision_decoder_request());
  const auto runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_vision_decoder_request());
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionDecoder) {
    return;
  }

  auto vision_graph =
      prime_vision_decoder_graph(input_arg, runtime_policy, context);
  if (!vision_graph.defined()) {
    return;
  }

  bool use_graph_scratch = false;
  if (runtime_policy.scratch_arena_plan.has_value()) {
    const uint32_t scratch_alignment = std::max<uint32_t>(
        runtime_policy.scratch_arena_plan->alignment,
        static_cast<uint32_t>(std::max<int64_t>(
            1, static_cast<int64_t>(c10::elementSize(kFloat)))));
    const size_t requested_bytes = vision_decoder_fusion_block_scratch_bytes(
        input_arg, skip, target_sizes);
    if (
        requested_bytes > 0u &&
        runtime_policy.scratch_arena_plan->prefer_buffer_storage) {
      vision_graph.note_shared_scratch_requirement(
          std::max(
              requested_bytes,
              runtime_policy.scratch_arena_plan->min_arena_bytes),
          scratch_alignment,
          runtime_policy.execution_program_plan->persistent);
      use_graph_scratch = true;
    }
  }

  (void)vision_graph.lookup_or_create_program(
      vision_decoder_program_label(
          context->allocation_label(), context.get()),
      input_arg.sizes(),
      skip.has_value() ? std::optional<std::vector<int64_t>>(skip->sizes().vec())
                       : std::nullopt,
      target_sizes,
      vision_decoder_out_channels(context),
      !use_graph_scratch,
      *runtime_policy.execution_program_plan);
  utils::log_vulkan_op_hit(
      "vulkan_prepack::prime_vision_decoder_fusion_block_context_graph");
}

VisionDecoderHeadContext::VisionDecoderHeadContext(
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet4_context,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet3_context,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet2_context,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet1_context,
    c10::intrusive_ptr<Conv2dPackedContext> output_conv1_context,
    c10::intrusive_ptr<Conv2dPackedContext> output_conv2_conv1_context,
    c10::intrusive_ptr<Conv2dPackedContext> output_conv2_conv2_context,
    const bool align_corners,
    std::string allocation_label)
    : allocation_label_(std::move(allocation_label)),
      align_corners_(align_corners),
      refinenet4_context_(std::move(refinenet4_context)),
      refinenet3_context_(std::move(refinenet3_context)),
      refinenet2_context_(std::move(refinenet2_context)),
      refinenet1_context_(std::move(refinenet1_context)),
      output_conv1_context_(std::move(output_conv1_context)),
      output_conv2_conv1_context_(std::move(output_conv2_conv1_context)),
      output_conv2_conv2_context_(std::move(output_conv2_conv2_context)) {
  TORCH_CHECK(
      refinenet4_context_ && refinenet3_context_ && refinenet2_context_ &&
          refinenet1_context_ && output_conv1_context_ &&
          output_conv2_conv1_context_ && output_conv2_conv2_context_,
      "Vision decoder head context requires all sub-contexts to be defined");

  unpacked_.reserve(Unpacked::NumArgs);
  unpacked_.emplace_back(refinenet4_context_);
  unpacked_.emplace_back(refinenet3_context_);
  unpacked_.emplace_back(refinenet2_context_);
  unpacked_.emplace_back(refinenet1_context_);
  unpacked_.emplace_back(output_conv1_context_);
  unpacked_.emplace_back(output_conv2_conv1_context_);
  unpacked_.emplace_back(output_conv2_conv2_context_);
  unpacked_.emplace_back(align_corners_);
  unpacked_.emplace_back(allocation_label_);
}

VisionDecoderHeadContext VisionDecoderHeadContext::pack(
    c10::impl::GenericList unpacked) {
  return VisionDecoderHeadContext(
      unpacked.get(Unpacked::Refinenet4Context)
          .toCustomClass<VisionDecoderFusionBlockContext>(),
      unpacked.get(Unpacked::Refinenet3Context)
          .toCustomClass<VisionDecoderFusionBlockContext>(),
      unpacked.get(Unpacked::Refinenet2Context)
          .toCustomClass<VisionDecoderFusionBlockContext>(),
      unpacked.get(Unpacked::Refinenet1Context)
          .toCustomClass<VisionDecoderFusionBlockContext>(),
      unpacked.get(Unpacked::OutputConv1Context)
          .toCustomClass<Conv2dPackedContext>(),
      unpacked.get(Unpacked::OutputConv2Conv1Context)
          .toCustomClass<Conv2dPackedContext>(),
      unpacked.get(Unpacked::OutputConv2Conv2Context)
          .toCustomClass<Conv2dPackedContext>(),
      unpacked.get(Unpacked::AlignCorners).toBool(),
      unpacked.get(Unpacked::Label).toStringRef());
}

c10::intrusive_ptr<VisionDecoderHeadContext> create_vision_decoder_head_context(
    const Tensor& prototype,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet4_context,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet3_context,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet2_context,
    c10::intrusive_ptr<VisionDecoderFusionBlockContext> refinenet1_context,
    c10::intrusive_ptr<Conv2dPackedContext> output_conv1_context,
    c10::intrusive_ptr<Conv2dPackedContext> output_conv2_conv1_context,
    c10::intrusive_ptr<Conv2dPackedContext> output_conv2_conv2_context,
    const bool align_corners,
    std::string label) {
  (void)prototype;
  return c10::make_intrusive<VisionDecoderHeadContext>(
      std::move(refinenet4_context),
      std::move(refinenet3_context),
      std::move(refinenet2_context),
      std::move(refinenet1_context),
      std::move(output_conv1_context),
      std::move(output_conv2_conv1_context),
      std::move(output_conv2_conv2_context),
      align_corners,
      std::move(label));
}

Tensor run_vision_decoder_head_context(
    const Tensor& layer1_arg,
    const Tensor& layer2_arg,
    const Tensor& layer3_arg,
    const Tensor& layer4_arg,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderHeadContext>& context) {
  TORCH_CHECK(context, "Vision decoder head context must be defined");
  TORCH_CHECK(
      layer1_arg.dim() == 4 && layer2_arg.dim() == 4 && layer3_arg.dim() == 4 &&
          layer4_arg.dim() == 4,
      "Vision decoder head context expects rank-4 layer inputs");
  TORCH_CHECK(
      output_size.size() == 2,
      "Vision decoder head context expects a rank-1 output size with 2 entries");

  const Device output_device = layer1_arg.device();
  const ScalarType output_dtype = layer1_arg.scalar_type();

  Tensor layer1 = layer1_arg.is_vulkan() ? layer1_arg : layer1_arg.vulkan();
  Tensor layer2 = layer2_arg.is_vulkan() ? layer2_arg : layer2_arg.vulkan();
  Tensor layer3 = layer3_arg.is_vulkan() ? layer3_arg : layer3_arg.vulkan();
  Tensor layer4 = layer4_arg.is_vulkan() ? layer4_arg : layer4_arg.vulkan();
  if (layer1.scalar_type() != kFloat) {
    layer1 = layer1.to(kFloat);
  }
  if (layer2.scalar_type() != kFloat) {
    layer2 = layer2.to(kFloat);
  }
  if (layer3.scalar_type() != kFloat) {
    layer3 = layer3.to(kFloat);
  }
  if (layer4.scalar_type() != kFloat) {
    layer4 = layer4.to(kFloat);
  }

  const auto fallback = [&]() -> Tensor {
    Tensor path4 = run_vision_decoder_fusion_block_context(
        layer4,
        std::nullopt,
        std::optional<std::vector<int64_t>>({layer3.size(2), layer3.size(3)}),
        context->refinenet4_context());
    Tensor path3 = run_vision_decoder_fusion_block_context(
        path4,
        layer3,
        std::optional<std::vector<int64_t>>({layer2.size(2), layer2.size(3)}),
        context->refinenet3_context());
    Tensor path2 = run_vision_decoder_fusion_block_context(
        path3,
        layer2,
        std::optional<std::vector<int64_t>>({layer1.size(2), layer1.size(3)}),
        context->refinenet2_context());
    Tensor path1 = run_vision_decoder_fusion_block_context(
        path2, layer1, std::nullopt, context->refinenet1_context());
    Tensor output = run_conv2d_context(path1, context->output_conv1_context());
    output = at::upsample_bilinear2d(
        output,
        output_size.vec(),
        context->align_corners(),
        std::nullopt,
        std::nullopt);
    output =
        run_conv2d_context(output, context->output_conv2_conv1_context());
    output = at::relu(output);
    output =
        run_conv2d_context(output, context->output_conv2_conv2_context());
    output = at::relu(output);
    return maybe_restore_tensor(output, output_device, output_dtype);
  };

  Tensor layer1_buffer = prepare_decoder_buffer_tensor(layer1);
  Tensor layer2_buffer = prepare_decoder_buffer_tensor(layer2);
  Tensor layer3_buffer = prepare_decoder_buffer_tensor(layer3);
  Tensor layer4_buffer = prepare_decoder_buffer_tensor(layer4);
  if (
      layer1_buffer.dim() != 4 || layer2_buffer.dim() != 4 ||
      layer3_buffer.dim() != 4 || layer4_buffer.dim() != 4) {
    return fallback();
  }

  utils::VulkanPlanningRequestScope planning_scope(
      utils::make_vulkan_vision_decoder_request());
  const auto runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_vision_decoder_request());
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionDecoder ||
      !has_explicit_runtime_capture_label() ||
      !can_use_decoder_head_replay(
          layer1_buffer, layer2_buffer, layer3_buffer, layer4_buffer)) {
    return fallback();
  }

  const int64_t output_conv1_channels =
      context->output_conv1_context()
          ->unpack()
          .get(Conv2dPackedContext::Unpacked::Weight)
          .toTensor()
          .size(0);
  const int64_t output_conv2_channels =
      context->output_conv2_conv1_context()
          ->unpack()
          .get(Conv2dPackedContext::Unpacked::Weight)
          .toTensor()
          .size(0);
  const int64_t final_channels =
      context->output_conv2_conv2_context()
          ->unpack()
          .get(Conv2dPackedContext::Unpacked::Weight)
          .toTensor()
          .size(0);
  const std::vector<int64_t> output_sizes{
      layer1_buffer.size(0),
      final_channels,
      output_size[0],
      output_size[1],
  };

  auto vision_graph = utils::lookup_or_create_labeled_vision_decoder_inference_graph(
      vision_decoder_graph_label(context->allocation_label()),
      kFloat,
      runtime_policy.execution_program_plan->persistent);
  auto vision_replay = vision_graph.lookup_or_create_head_replay(
      vision_decoder_head_program_label(
          context->allocation_label(), context.get()),
      layer1_buffer.sizes(),
      layer2_buffer.sizes(),
      layer3_buffer.sizes(),
      layer4_buffer.sizes(),
      output_sizes,
      output_conv1_channels,
      output_conv2_channels,
      final_channels,
      *runtime_policy.execution_program_plan);

  if (!vision_replay.defined()) {
    return fallback();
  }

  copy_tensor_for_replay(vision_replay.layer1_slot(), layer1_buffer);
  copy_tensor_for_replay(vision_replay.layer2_slot(), layer2_buffer);
  copy_tensor_for_replay(vision_replay.layer3_slot(), layer3_buffer);
  copy_tensor_for_replay(vision_replay.layer4_slot(), layer4_buffer);
  api::context()->flush_pending_cmds();

  if (!vision_replay.recorded()) {
    Tensor warmup_output = utils::create_buffer_tensor(
        vision_replay.output_slot().sizes(),
        vision_replay.output_slot().scalar_type(),
        /*persistent=*/false);
    copy_tensor_for_replay(
        warmup_output,
        run_vision_decoder_head_program(
            vision_replay.layer1_slot(),
            vision_replay.layer2_slot(),
            vision_replay.layer3_slot(),
            vision_replay.layer4_slot(),
            output_size,
            context,
            vision_replay.refinenet4_program(),
            vision_replay.refinenet3_program(),
            vision_replay.refinenet2_program(),
            vision_replay.refinenet1_program(),
            vision_replay.output_conv1_output(),
            vision_replay.upsample_output(),
            vision_replay.output_conv2_conv1_output(),
            vision_replay.output_conv2_relu1_output(),
            vision_replay.output_conv2_conv2_output(),
            vision_replay.output_slot()));
    api::context()->flush_pending_cmds();
    vision_replay.replay().record([&]() {
      (void)run_vision_decoder_head_program(
          vision_replay.layer1_slot(),
          vision_replay.layer2_slot(),
          vision_replay.layer3_slot(),
          vision_replay.layer4_slot(),
          output_size,
          context,
          vision_replay.refinenet4_program(),
          vision_replay.refinenet3_program(),
          vision_replay.refinenet2_program(),
          vision_replay.refinenet1_program(),
          vision_replay.output_conv1_output(),
          vision_replay.upsample_output(),
          vision_replay.output_conv2_conv1_output(),
          vision_replay.output_conv2_relu1_output(),
          vision_replay.output_conv2_conv2_output(),
          vision_replay.output_slot());
    });
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_decoder_head_context.replay_warmup");
    utils::log_vulkan_op_hit("vulkan_prepack::run_vision_decoder_head_context");
    return maybe_restore_tensor(warmup_output, output_device, output_dtype);
  }

  vision_replay.replay().submit();
  Tensor output = utils::create_buffer_tensor(
      vision_replay.output_slot().sizes(),
      vision_replay.output_slot().scalar_type(),
      /*persistent=*/false);
  copy_tensor_for_replay(output, vision_replay.output_slot());
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_vision_decoder_head_context.replay");
  utils::log_vulkan_op_hit("vulkan_prepack::run_vision_decoder_head_context");
  return maybe_restore_tensor(output, output_device, output_dtype);
}

void prime_vision_decoder_head_context_graph(
    const Tensor& layer1,
    const Tensor& layer2,
    const Tensor& layer3,
    const Tensor& layer4,
    IntArrayRef output_size,
    const c10::intrusive_ptr<VisionDecoderHeadContext>& context) {
  if (
      !context || !layer1.defined() || !layer1.is_vulkan() ||
      !layer2.defined() || !layer2.is_vulkan() || !layer3.defined() ||
      !layer3.is_vulkan() || !layer4.defined() || !layer4.is_vulkan()) {
    return;
  }

  TORCH_CHECK(
      layer1.dim() == 4 && layer2.dim() == 4 && layer3.dim() == 4 &&
          layer4.dim() == 4,
      "Vision decoder head graph priming expects rank-4 layer inputs");
  TORCH_CHECK(
      output_size.size() == 2,
      "Vision decoder head graph priming expects a rank-1 output size with 2 entries");
  if (
      layer1.scalar_type() != kFloat || layer2.scalar_type() != kFloat ||
      layer3.scalar_type() != kFloat || layer4.scalar_type() != kFloat) {
    return;
  }

  utils::VulkanPlanningRequestScope planning_scope(
      utils::make_vulkan_vision_decoder_request());
  const auto runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_vision_decoder_request());
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionDecoder ||
      !has_explicit_runtime_capture_label()) {
    return;
  }

  const int64_t output_conv1_channels =
      context->output_conv1_context()
          ->unpack()
          .get(Conv2dPackedContext::Unpacked::Weight)
          .toTensor()
          .size(0);
  const int64_t output_conv2_channels =
      context->output_conv2_conv1_context()
          ->unpack()
          .get(Conv2dPackedContext::Unpacked::Weight)
          .toTensor()
          .size(0);
  const int64_t final_channels =
      context->output_conv2_conv2_context()
          ->unpack()
          .get(Conv2dPackedContext::Unpacked::Weight)
          .toTensor()
          .size(0);
  const std::vector<int64_t> output_sizes{
      layer1.size(0), final_channels, output_size[0], output_size[1]};

  auto vision_graph = utils::lookup_or_create_labeled_vision_decoder_inference_graph(
      vision_decoder_graph_label(context->allocation_label()),
      kFloat,
      runtime_policy.execution_program_plan->persistent);
  (void)vision_graph.lookup_or_create_head_replay(
      vision_decoder_head_program_label(
          context->allocation_label(), context.get()),
      layer1.sizes(),
      layer2.sizes(),
      layer3.sizes(),
      layer4.sizes(),
      output_sizes,
      output_conv1_channels,
      output_conv2_channels,
      final_channels,
      *runtime_policy.execution_program_plan);
  utils::log_vulkan_op_hit(
      "vulkan_prepack::prime_vision_decoder_head_context_graph");
}

std::tuple<Tensor, Tensor> run_vision_backbone_decoder_replay_bundle_bridge(
    const Tensor& backbone_input_arg,
    const c10::intrusive_ptr<VisionBackboneBlockContext>& backbone_context,
    const Tensor& decoder_input_arg,
    const std::optional<Tensor>& decoder_skip_arg,
    const std::optional<std::vector<int64_t>>& decoder_size,
    const c10::intrusive_ptr<VisionDecoderFusionBlockContext>& decoder_context) {
  TORCH_CHECK(
      backbone_context && decoder_context,
      "Vision backbone/decoder replay bundle bridge expects defined contexts");

  const Device backbone_output_device = backbone_input_arg.device();
  const ScalarType backbone_output_dtype = backbone_input_arg.scalar_type();
  const Device decoder_output_device = decoder_input_arg.device();
  const ScalarType decoder_output_dtype = decoder_input_arg.scalar_type();

  Tensor backbone_input =
      backbone_input_arg.is_vulkan() ? backbone_input_arg : backbone_input_arg.vulkan();
  Tensor decoder_input =
      decoder_input_arg.is_vulkan() ? decoder_input_arg : decoder_input_arg.vulkan();
  std::optional<Tensor> decoder_skip =
      (decoder_skip_arg.has_value() && decoder_skip_arg->defined())
      ? std::optional<Tensor>(
            decoder_skip_arg->is_vulkan() ? *decoder_skip_arg : decoder_skip_arg->vulkan())
      : std::nullopt;

  TORCH_CHECK(
      backbone_input.dim() == 2 || backbone_input.dim() == 3,
      "Vision backbone/decoder replay bundle bridge expects rank-2 or rank-3 "
      "backbone input");
  TORCH_CHECK(
      decoder_input.dim() == 4,
      "Vision backbone/decoder replay bundle bridge expects rank-4 decoder input");
  TORCH_CHECK(
      backbone_input.scalar_type() == kFloat &&
          decoder_input.scalar_type() == kFloat &&
          (!decoder_skip.has_value() || decoder_skip->scalar_type() == kFloat),
      "Vision backbone/decoder replay bundle bridge currently expects float inputs");

  utils::VulkanPlanningRequestScope backbone_scope(
      utils::make_vulkan_vision_backbone_request());
  const auto backbone_runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_vision_backbone_request());
  if (
      !backbone_runtime_policy.execution_program_plan.has_value() ||
      backbone_runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionBackbone) {
    return std::make_tuple(
        run_vision_backbone_block_context(backbone_input_arg, backbone_context),
        run_vision_decoder_fusion_block_context(
            decoder_input_arg, decoder_skip_arg, decoder_size, decoder_context));
  }

  utils::VulkanPlanningRequestScope decoder_scope(
      utils::make_vulkan_vision_decoder_request());
  const auto decoder_runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_vision_decoder_request());
  if (
      !decoder_runtime_policy.execution_program_plan.has_value() ||
      decoder_runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionDecoder) {
    return std::make_tuple(
        run_vision_backbone_block_context(backbone_input_arg, backbone_context),
        run_vision_decoder_fusion_block_context(
            decoder_input_arg, decoder_skip_arg, decoder_size, decoder_context));
  }

  auto backbone_graph =
      prime_vision_backbone_graph(backbone_input, backbone_runtime_policy, backbone_context);
  Tensor decoder_input_buffer = prepare_decoder_buffer_tensor(decoder_input);
  std::optional<Tensor> decoder_skip_buffer =
      (decoder_skip.has_value() && decoder_skip->defined())
      ? std::optional<Tensor>(prepare_decoder_buffer_tensor(*decoder_skip))
      : std::nullopt;
  auto decoder_graph =
      prime_vision_decoder_graph(decoder_input_buffer, decoder_runtime_policy, decoder_context);
  if (!backbone_graph.defined() || !decoder_graph.defined()) {
    return std::make_tuple(
        run_vision_backbone_block_context(backbone_input_arg, backbone_context),
        run_vision_decoder_fusion_block_context(
            decoder_input_arg, decoder_skip_arg, decoder_size, decoder_context));
  }

  std::optional<utils::ScratchArena> backbone_graph_scratch = std::nullopt;
  if (backbone_runtime_policy.scratch_arena_plan.has_value()) {
    const int64_t batch_size =
        backbone_input.dim() == 2 ? 1 : backbone_input.size(0);
    const int64_t token_count =
        backbone_input.dim() == 2 ? backbone_input.size(0) : backbone_input.size(1);
    const int64_t embed_dim = backbone_input.size(-1);
    const uint32_t scratch_alignment = std::max<uint32_t>(
        backbone_runtime_policy.scratch_arena_plan->alignment,
        static_cast<uint32_t>(std::max<int64_t>(
            1, static_cast<int64_t>(c10::elementSize(kFloat)))));
    const size_t requested_bytes = vision_attention_scratch_bytes(
        batch_size,
        token_count,
        embed_dim,
        backbone_context->num_heads(),
        backbone_input.scalar_type(),
        backbone_context->qkv_bias().defined(),
        scratch_alignment);
    if (
        requested_bytes > 0u &&
        backbone_runtime_policy.scratch_arena_plan->prefer_buffer_storage) {
      backbone_graph_scratch = backbone_graph.ensure_shared_scratch(
          std::max(
              requested_bytes,
              backbone_runtime_policy.scratch_arena_plan->min_arena_bytes),
          scratch_alignment,
          backbone_runtime_policy.execution_program_plan->persistent);
    }
  }

  const int64_t backbone_token_count =
      backbone_input.dim() == 2 ? backbone_input.size(0) : backbone_input.size(1);
  const int64_t backbone_embed_dim = backbone_input.size(-1);
  const int64_t backbone_hidden_dim = vision_block_hidden_dim(backbone_context);
  auto backbone_replay = backbone_graph.lookup_or_create_replay(
      vision_backbone_program_label(
          backbone_context->allocation_label(), backbone_context.get()),
      backbone_input.sizes(),
      backbone_token_count,
      backbone_embed_dim,
      backbone_hidden_dim,
      backbone_context->num_heads(),
      *backbone_runtime_policy.execution_program_plan);

  const std::vector<int64_t> decoder_target_sizes =
      resolve_decoder_target_sizes(decoder_input_buffer, decoder_size);
  auto decoder_replay = decoder_graph.lookup_or_create_replay(
      vision_decoder_program_label(
          decoder_context->allocation_label(), decoder_context.get()),
      decoder_input_buffer.sizes(),
      decoder_skip_buffer.has_value()
          ? std::optional<std::vector<int64_t>>(decoder_skip_buffer->sizes().vec())
          : std::nullopt,
      decoder_target_sizes,
      vision_decoder_out_channels(decoder_context),
      *decoder_runtime_policy.execution_program_plan);

  TORCH_CHECK(
      backbone_replay.defined() && decoder_replay.defined(),
      "Vision backbone/decoder replay bundle bridge expected defined replays");

  copy_tensor_for_replay(backbone_replay.input_slot(), backbone_input);
  utils::copy_buffer_tensor_direct_(
      decoder_replay.input_slot(), decoder_input_buffer);
  if (decoder_skip_buffer.has_value() && decoder_skip_buffer->defined()) {
    TORCH_INTERNAL_ASSERT(
        decoder_replay.skip_slot().has_value(),
        "Vision backbone/decoder replay bundle bridge expected decoder skip slot");
    utils::copy_buffer_tensor_direct_(
        *decoder_replay.skip_slot(), *decoder_skip_buffer);
  }
  api::context()->flush_pending_cmds();

  const std::string root_label =
      current_graph_capture_label("depth.vision", "depth.vision.graph");
  auto root = utils::lookup_or_create_labeled_execution_graph_root(
      root_label,
      kFloat,
      backbone_runtime_policy.execution_program_plan->persistent &&
          decoder_runtime_policy.execution_program_plan->persistent);
  auto replay_bundle = root.lookup_or_create_replay_bundle(
      "vision.backbone_decoder",
      [&]() -> utils::ExecutionGraphReplayBundle {
        std::vector<utils::ExecutionGraphReplayStep> steps;
        steps.reserve(2u);
        steps.push_back(make_vision_backbone_replay_step(
            backbone_replay, backbone_context, backbone_graph_scratch));
        steps.push_back(make_vision_decoder_replay_step(
            decoder_replay, decoder_target_sizes, decoder_context));
        return utils::make_execution_graph_replay_bundle(
            root.allocation_label() + ".vision.backbone_decoder.replay",
            kFloat,
            backbone_runtime_policy.execution_program_plan->persistent &&
                decoder_runtime_policy.execution_program_plan->persistent,
            std::move(steps));
      });
  TORCH_CHECK(
      replay_bundle.defined() && replay_bundle.size() == 2u,
      "Vision backbone/decoder replay bundle bridge expected a 2-phase bundle");

  if (!replay_bundle.recorded()) {
    Tensor warmup_backbone_output = utils::create_buffer_tensor(
        backbone_replay.output_slot().sizes(),
        backbone_replay.output_slot().scalar_type(),
        /*persistent=*/false);
    Tensor warmup_decoder_output = utils::create_buffer_tensor(
        decoder_replay.output_slot().sizes(),
        decoder_replay.output_slot().scalar_type(),
        /*persistent=*/false);
    if (backbone_graph_scratch.has_value()) {
      backbone_graph_scratch->reset();
    }
    api::RuntimeLabelScope backbone_runtime_scope(vision_backbone_execution_label(
        backbone_context->allocation_label(), backbone_context.get()));
    (void)run_vision_backbone_block_program(
        backbone_replay.input_slot(),
        backbone_context,
        &backbone_replay.program(),
        backbone_graph_scratch.has_value() ? &(*backbone_graph_scratch) : nullptr,
        &backbone_replay.output_slot());
    copy_tensor_for_replay(warmup_backbone_output, backbone_replay.output_slot());
    const VisionDecoderRunOutputs replay_outputs{
        decoder_replay.program().skip_relu_output(),
        decoder_replay.program().skip_conv1_output(),
        decoder_replay.program().skip_conv2_output(),
        decoder_replay.program().skip_res_output(),
        decoder_replay.program().main_input_output(),
        decoder_replay.program().main_relu_output(),
        decoder_replay.program().main_conv1_output(),
        decoder_replay.program().main_conv2_output(),
        decoder_replay.program().main_res_output(),
        decoder_replay.program().upsample_output(),
        decoder_replay.program().out_conv_output(),
    };
    utils::copy_buffer_tensor_direct_(
        warmup_decoder_output,
        run_vision_decoder_fusion_block_program(
            decoder_replay.input_slot(),
            decoder_replay.skip_slot(),
            decoder_target_sizes,
            decoder_context,
            replay_outputs));
    api::context()->flush_pending_cmds();
    replay_bundle.record();
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_backbone_decoder_replay_bundle_bridge.replay_warmup");
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_backbone_decoder_replay_bundle_bridge");
    return std::make_tuple(
        maybe_restore_tensor(
            warmup_backbone_output, backbone_output_device, backbone_output_dtype),
        maybe_restore_tensor(
            warmup_decoder_output, decoder_output_device, decoder_output_dtype));
  }

  replay_bundle.submit();

  Tensor backbone_output = utils::create_buffer_tensor(
      backbone_replay.output_slot().sizes(),
      backbone_replay.output_slot().scalar_type(),
      /*persistent=*/false);
  copy_tensor_for_replay(backbone_output, backbone_replay.output_slot());
  Tensor decoder_output = utils::create_buffer_tensor(
      decoder_replay.output_slot().sizes(),
      decoder_replay.output_slot().scalar_type(),
      /*persistent=*/false);
  utils::copy_buffer_tensor_direct_(
      decoder_output, decoder_replay.output_slot());
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_vision_backbone_decoder_replay_bundle_bridge");
  return std::make_tuple(
      maybe_restore_tensor(
          backbone_output, backbone_output_device, backbone_output_dtype),
      maybe_restore_tensor(
          decoder_output, decoder_output_device, decoder_output_dtype));
}

std::vector<Tensor> run_vision_backbone_stack_replay_bundle_bridge(
    const Tensor& input_arg,
    const c10::List<c10::intrusive_ptr<VisionBackboneBlockContext>>& contexts,
    IntArrayRef capture_indices) {
  TORCH_CHECK(
      contexts.size() > 0,
      "Vision backbone stack replay bundle bridge expects at least one context");

  const std::vector<int64_t> capture_indices_vec = capture_indices.vec();
  for (const int64_t capture_idx : capture_indices_vec) {
    TORCH_CHECK(
        capture_idx >= 0 &&
            capture_idx < static_cast<int64_t>(contexts.size()),
        "Vision backbone stack replay bundle bridge capture index ",
        capture_idx,
        " is out of range for ",
        contexts.size(),
        " contexts");
  }
  if (capture_indices_vec.empty()) {
    return {};
  }

  std::vector<c10::intrusive_ptr<VisionBackboneBlockContext>> backbone_contexts;
  backbone_contexts.reserve(contexts.size());
  for (const auto& context_ref : contexts) {
    c10::intrusive_ptr<VisionBackboneBlockContext> context = context_ref;
    TORCH_CHECK(
        static_cast<bool>(context),
        "Vision backbone stack replay bundle bridge expects defined contexts");
    backbone_contexts.push_back(std::move(context));
  }

  const auto sequential_fallback =
      [&]() -> std::vector<Tensor> {
    Tensor current = input_arg;
    std::vector<Tensor> outputs(capture_indices_vec.size());
    for (size_t idx = 0u; idx < backbone_contexts.size(); ++idx) {
      current = run_vision_backbone_block_context(current, backbone_contexts[idx]);
      for (size_t capture_pos = 0u; capture_pos < capture_indices_vec.size();
           ++capture_pos) {
        if (capture_indices_vec[capture_pos] == static_cast<int64_t>(idx)) {
          outputs[capture_pos] = current;
        }
      }
    }
    return outputs;
  };

  TORCH_CHECK(
      input_arg.dim() == 2 || input_arg.dim() == 3,
      "Vision backbone stack replay bundle bridge expects rank-2 or rank-3 input");

  const Device output_device = input_arg.device();
  const ScalarType output_dtype = input_arg.scalar_type();
  Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();

  utils::VulkanPlanningRequestScope planning_scope(
      utils::make_vulkan_vision_backbone_request());
  const auto runtime_policy = utils::build_vulkan_runtime_policy(
      utils::make_vulkan_vision_backbone_request());
  if (
      !runtime_policy.execution_program_plan.has_value() ||
      runtime_policy.execution_program_plan->kind !=
          utils::VulkanExecutionProgramKind::VisionBackbone ||
      input.scalar_type() != kFloat) {
    return sequential_fallback();
  }

  const int64_t batch_size = input.dim() == 2 ? 1 : input.size(0);
  const int64_t token_count = input.dim() == 2 ? input.size(0) : input.size(1);
  const int64_t embed_dim = input.size(-1);
  const uint32_t scratch_alignment =
      runtime_policy.scratch_arena_plan.has_value()
      ? std::max<uint32_t>(
            runtime_policy.scratch_arena_plan->alignment,
            static_cast<uint32_t>(std::max<int64_t>(
                1, static_cast<int64_t>(c10::elementSize(kFloat)))))
      : 1u;

  std::vector<std::optional<utils::ScratchArena>> graph_scratches;
  graph_scratches.reserve(backbone_contexts.size());
  std::vector<utils::VisionBackboneInferenceReplay> replays;
  replays.reserve(backbone_contexts.size());
  for (const auto& context : backbone_contexts) {
    auto vision_graph = prime_vision_backbone_graph(input, runtime_policy, context);
    if (!vision_graph.defined()) {
      return sequential_fallback();
    }
    std::optional<utils::ScratchArena> graph_scratch = std::nullopt;
    if (
        runtime_policy.scratch_arena_plan.has_value() &&
        runtime_policy.scratch_arena_plan->prefer_buffer_storage) {
      const size_t requested_bytes = vision_attention_scratch_bytes(
          batch_size,
          token_count,
          embed_dim,
          context->num_heads(),
          input.scalar_type(),
          context->qkv_bias().defined(),
          scratch_alignment);
      if (requested_bytes > 0u) {
        graph_scratch = vision_graph.ensure_shared_scratch(
            std::max(
                requested_bytes,
                runtime_policy.scratch_arena_plan->min_arena_bytes),
            scratch_alignment,
            runtime_policy.execution_program_plan->persistent);
      }
    }
    const int64_t hidden_dim = vision_block_hidden_dim(context);
    auto replay = vision_graph.lookup_or_create_replay(
        vision_backbone_program_label(
            context->allocation_label(), context.get()),
        input.sizes(),
        token_count,
        embed_dim,
        hidden_dim,
        context->num_heads(),
        *runtime_policy.execution_program_plan);
    if (!replay.defined()) {
      return sequential_fallback();
    }
    graph_scratches.push_back(std::move(graph_scratch));
    replays.push_back(std::move(replay));
  }

  std::string bundle_key = "vision.backbone_stack|count=" +
      std::to_string(backbone_contexts.size()) + "|capture=";
  for (size_t idx = 0u; idx < capture_indices_vec.size(); ++idx) {
    if (idx > 0u) {
      bundle_key += ",";
    }
    bundle_key += std::to_string(capture_indices_vec[idx]);
  }
  bundle_key += "|contexts=";
  for (size_t idx = 0u; idx < backbone_contexts.size(); ++idx) {
    if (idx > 0u) {
      bundle_key += ",";
    }
    bundle_key += std::to_string(static_cast<unsigned long long>(
        reinterpret_cast<uintptr_t>(backbone_contexts[idx].get())));
  }

  const std::string root_label =
      current_graph_capture_label("depth.vision", "depth.vision.graph");
  auto root = utils::lookup_or_create_labeled_execution_graph_root(
      root_label,
      kFloat,
      runtime_policy.execution_program_plan->persistent);
  auto replay_bundle = root.lookup_or_create_replay_bundle(
      bundle_key,
      [&]() -> utils::ExecutionGraphReplayBundle {
        std::vector<utils::ExecutionGraphReplayStep> steps;
        steps.reserve(replays.size());
        steps.push_back(make_vision_backbone_replay_step(
            replays[0], backbone_contexts[0], graph_scratches[0]));
        for (size_t idx = 1u; idx < replays.size(); ++idx) {
          steps.push_back(make_chained_vision_backbone_replay_step(
              replays[idx - 1u],
              replays[idx],
              backbone_contexts[idx],
              graph_scratches[idx]));
        }
        return utils::make_execution_graph_replay_bundle(
            root.allocation_label() + ".vision.backbone_stack.replay",
            kFloat,
            runtime_policy.execution_program_plan->persistent,
            std::move(steps));
      });
  TORCH_CHECK(
      replay_bundle.defined() && replay_bundle.size() == replays.size(),
      "Vision backbone stack replay bundle bridge expected a replay bundle "
      "matching the number of contexts");

  copy_tensor_for_replay(replays[0].input_slot(), input);
  api::context()->flush_pending_cmds();

  if (!replay_bundle.recorded()) {
    std::vector<Tensor> warmup_outputs(capture_indices_vec.size());
    for (size_t idx = 0u; idx < replays.size(); ++idx) {
      if (graph_scratches[idx].has_value()) {
        graph_scratches[idx]->reset();
      }
      api::RuntimeLabelScope runtime_scope(vision_backbone_execution_label(
          backbone_contexts[idx]->allocation_label(),
          backbone_contexts[idx].get()));
      const Tensor& replay_input =
          idx == 0u ? replays[idx].input_slot() : replays[idx - 1u].output_slot();
      (void)run_vision_backbone_block_program(
          replay_input,
          backbone_contexts[idx],
          &replays[idx].program(),
          graph_scratches[idx].has_value() ? &(*graph_scratches[idx]) : nullptr,
          &replays[idx].output_slot());
      for (size_t capture_pos = 0u; capture_pos < capture_indices_vec.size();
           ++capture_pos) {
        if (capture_indices_vec[capture_pos] != static_cast<int64_t>(idx)) {
          continue;
        }
        Tensor output = utils::create_buffer_tensor(
            replays[idx].output_slot().sizes(),
            replays[idx].output_slot().scalar_type(),
            /*persistent=*/true);
        copy_tensor_for_replay(output, replays[idx].output_slot());
        warmup_outputs[capture_pos] =
            maybe_restore_tensor(output, output_device, output_dtype);
      }
    }
    api::context()->flush_pending_cmds();
    replay_bundle.record();
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_backbone_stack_replay_bundle_bridge.replay_warmup");
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_backbone_stack_replay_bundle_bridge");
    return warmup_outputs;
  }

  replay_bundle.submit();

  std::vector<Tensor> outputs(capture_indices_vec.size());
  for (size_t capture_pos = 0u; capture_pos < capture_indices_vec.size();
       ++capture_pos) {
    const int64_t replay_idx = capture_indices_vec[capture_pos];
    Tensor output = utils::create_buffer_tensor(
        replays[replay_idx].output_slot().sizes(),
        replays[replay_idx].output_slot().scalar_type(),
        /*persistent=*/true);
    copy_tensor_for_replay(output, replays[replay_idx].output_slot());
    outputs[capture_pos] =
        maybe_restore_tensor(output, output_device, output_dtype);
  }
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_vision_backbone_stack_replay_bundle_bridge.replay");
  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_vision_backbone_stack_replay_bundle_bridge");
  return outputs;
}

Tensor tokens_to_feature_map(
    const Tensor& input_arg,
    const int64_t height,
    const int64_t width) {
  if (!input_arg.is_vulkan() || input_arg.scalar_type() != kFloat) {
    return tokens_to_feature_map_fallback(input_arg, height, width);
  }

  api::AllocationScope allocation_scope("tokens_to_feature_map");

  Tensor input = input_arg;
  const bool use_2d_input = input.dim() == 2;
  const int64_t batch_size = use_2d_input ? 1 : input.size(0);
  const int64_t token_count = use_2d_input ? input.size(0) : input.size(1);
  const int64_t channels = input.size(-1);

  TORCH_CHECK(
      input.dim() == 2 || input.dim() == 3,
      "Vulkan tokens_to_feature_map expects a [N, C] or [B, N, C] tensor");
  TORCH_CHECK(
      token_count == height * width,
      "Vulkan tokens_to_feature_map expected token count ",
      height * width,
      " but received ",
      token_count);

  utils::log_vulkan_op_hit("aten::tokens_to_feature_map");

  const vTensor& v_input_probe = convert(input);
  if (
      v_input_probe.storage_type() == api::StorageType::TEXTURE_3D &&
      v_input_probe.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED &&
      batch_size == 1) {
    const std::vector<int64_t> output_sizes{
        batch_size,
        channels,
        height,
        width,
    };

    vTensor v_output{
        api::context(),
        output_sizes,
        convert_dtype(input.scalar_type()),
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
    };

    api::PipelineBarrier pipeline_barrier{};
    const api::utils::uvec3 global_size{
        api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
        1u,
        1u,
    };
    api::UniformParamsBuffer out_meta =
        utils::make_buffer_compute_metadata_ubo(api::context(), v_output);

    api::context()->submit_compute_job(
        VK_KERNEL(tokens_to_feature_map_texture_to_buffer),
        pipeline_barrier,
        global_size,
        adaptive_work_group_size(global_size),
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        out_meta.buffer(),
        v_input_probe.image(pipeline_barrier, api::PipelineStage::COMPUTE));

    utils::log_vulkan_op_hit("aten::tokens_to_feature_map.texture_to_buffer");
    return convert(v_output);
  }

  if (
      v_input_probe.storage_type() != api::StorageType::BUFFER ||
      !utils::supports_buffer_elementwise_compute(v_input_probe)) {
    utils::log_vulkan_op_hit("aten::tokens_to_feature_map.texture_view_fallback");
    if (use_2d_input) {
      return tokens_to_feature_map_fallback(input_arg, height, width);
    }
    return input.permute({0, 2, 1})
        .reshape({batch_size, channels, height, width});
  }

  const vTensor& v_input = v_input_probe;

  const std::vector<int64_t> output_sizes{
      batch_size,
      channels,
      height,
      width,
  };

  vTensor v_output{
      api::context(),
      output_sizes,
      convert_dtype(input.scalar_type()),
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  api::UniformParamsBuffer input_meta =
      utils::make_buffer_compute_metadata_ubo(api::context(), v_input);
  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(api::context(), v_output);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
      1u,
      1u,
  };

  api::context()->submit_compute_job(
      VK_KERNEL(tokens_to_feature_map_buffer),
      pipeline_barrier,
      global_size,
      adaptive_work_group_size(global_size),
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      input_meta.buffer());

  utils::log_vulkan_op_hit("aten::tokens_to_feature_map.buffer_to_buffer");
  return convert(v_output);
}

Tensor feature_map_to_tokens(const Tensor& input_arg) {
  if (!input_arg.is_vulkan() || input_arg.scalar_type() != kFloat) {
    return feature_map_to_tokens_fallback(input_arg);
  }

  TORCH_CHECK(
      input_arg.dim() == 4,
      "Vulkan feature_map_to_tokens expects a [B, C, H, W] tensor");

  api::AllocationScope allocation_scope("feature_map_to_tokens");
  utils::log_vulkan_op_hit("aten::feature_map_to_tokens");

  const vTensor& v_input = convert(input_arg);
  if (
      v_input.storage_type() == api::StorageType::TEXTURE_3D &&
      v_input.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED &&
      input_arg.size(0) == 1) {
    const std::vector<int64_t> output_sizes{
        1,
        input_arg.size(2) * input_arg.size(3),
        input_arg.size(1),
    };

    vTensor v_output{
        api::context(),
        output_sizes,
        convert_dtype(input_arg.scalar_type()),
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
    };

    const struct Block final {
      api::utils::ivec4 info;
    } block{
        {
            api::utils::safe_downcast<int32_t>(input_arg.size(3)),
            api::utils::safe_downcast<int32_t>(input_arg.size(2)),
            api::utils::safe_downcast<int32_t>(input_arg.size(1)),
            api::utils::safe_downcast<int32_t>(input_arg.size(0)),
        },
    };

    api::UniformParamsBuffer params(api::context(), block);
    api::PipelineBarrier pipeline_barrier{};
    const api::utils::uvec3 global_size{
        api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
        1u,
        1u,
    };
    api::UniformParamsBuffer out_meta =
        utils::make_buffer_compute_metadata_ubo(api::context(), v_output);

    api::context()->submit_compute_job(
        VK_KERNEL(feature_map_to_tokens_texture_to_buffer),
        pipeline_barrier,
        global_size,
        adaptive_work_group_size(global_size),
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        out_meta.buffer(),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        params.buffer());

    utils::log_vulkan_op_hit("aten::feature_map_to_tokens.texture_to_buffer");
    return convert(v_output);
  }

  if (
      v_input.storage_type() == api::StorageType::BUFFER &&
      utils::supports_buffer_elementwise_compute(v_input)) {
    const std::vector<int64_t> output_sizes{
        input_arg.size(0),
        input_arg.size(2) * input_arg.size(3),
        input_arg.size(1),
    };

    vTensor v_output{
        api::context(),
        output_sizes,
        convert_dtype(input_arg.scalar_type()),
        api::StorageType::BUFFER,
        api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
    };

    api::PipelineBarrier pipeline_barrier{};
    const api::utils::uvec3 global_size{
        api::utils::safe_downcast<uint32_t>(std::max<int64_t>(v_output.numel(), 1)),
        1u,
        1u,
    };
    api::UniformParamsBuffer out_meta =
        utils::make_buffer_compute_metadata_ubo(api::context(), v_output);
    api::UniformParamsBuffer input_meta =
        utils::make_buffer_compute_metadata_ubo(api::context(), v_input);

    api::context()->submit_compute_job(
        VK_KERNEL(feature_map_to_tokens_buffer),
        pipeline_barrier,
        global_size,
        adaptive_work_group_size(global_size),
        VK_NULL_HANDLE,
        v_output.buffer(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        out_meta.buffer(),
        v_input.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
        input_meta.buffer());

    utils::log_vulkan_op_hit("aten::feature_map_to_tokens.buffer_to_buffer");
    return convert(v_output);
  }

  utils::log_vulkan_op_hit("aten::feature_map_to_tokens.fallback");
  return feature_map_to_tokens_fallback(input_arg);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
