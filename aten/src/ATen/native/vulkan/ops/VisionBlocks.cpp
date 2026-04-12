#include <ATen/Functions.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/ops/BinaryOp.h>
#include <ATen/native/vulkan/ops/Softmax.h>
#include <ATen/native/vulkan/ops/Upsample.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/ops/VisionBlocks.h>
#include <ATen/native/vulkan/planning/ExecutionPrograms.h>
#include <ATen/native/vulkan/planning/Request.h>
#include <ATen/native/vulkan/planning/Runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
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

std::string vision_backbone_program_label(const std::string& label) {
  return vision_backbone_program_base_label(label) + ".program";
}

std::string vision_decoder_program_label(const std::string& label) {
  if (label.empty()) {
    return "depth.decoder.fusion.program";
  }
  return label + ".program";
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
    append_slice(input.sizes());
  }

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

std::optional<utils::ScratchArena> maybe_create_vision_decoder_scratch(
    const std::string& label,
    const size_t requested_bytes) {
  if (requested_bytes == 0u) {
    return std::nullopt;
  }

  const auto request = utils::make_vulkan_vision_decoder_request(
      utils::VulkanTensorRole::Scratch);
  const auto runtime_policy = utils::build_vulkan_runtime_policy(request);
  if (!runtime_policy.scratch_arena_plan.has_value()) {
    return std::nullopt;
  }

  const auto& desc = *runtime_policy.scratch_arena_plan;
  return utils::lookup_or_create_labeled_scratch_arena(
      vision_decoder_program_label(label),
      utils::VulkanScratchArenaSpec{
          kByte,
          std::max(desc.min_arena_bytes, requested_bytes),
          desc.alignment,
          api::ExecutionLayout::BUFFER_DIRECT,
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
          api::StorageType::BUFFER,
          desc.prefer_reusable_arena,
      });
}

Tensor run_attention_with_workspace_fallback(
    const Tensor& query_arg,
    const Tensor& key_arg,
    const Tensor& value_arg,
    utils::VisionBackboneProgram* const vision_program) {
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

  if (
      !(vision_program && vision_program->defined() &&
        vision_program->scratch_arena().has_value())) {
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

  auto& scratch_arena = *vision_program->scratch_arena();
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
      reserve_scratch_buffer_tensor(scratch_arena, scores_sizes, kFloat);
  auto [probs_slice, probs_output] =
      reserve_scratch_buffer_tensor(scratch_arena, scores_sizes, kFloat);
  auto [context_slice, context_output] =
      reserve_scratch_buffer_tensor(scratch_arena, output_sizes, kFloat);
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
    const c10::intrusive_ptr<VisionBackboneBlockContext>& context) {
  const auto request = utils::make_vulkan_vision_backbone_request();
  const auto runtime_policy = utils::build_vulkan_runtime_policy(request);
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
      runtime_policy.scratch_arena_plan.has_value()
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
      vision_backbone_program_label(context->allocation_label()),
      batch_size,
      token_count,
      embed_dim,
      hidden_dim,
      context->num_heads(),
      scratch_spec,
      *runtime_policy.execution_program_plan);
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
    utils::VisionBackboneProgram* vision_program = nullptr) {
  TORCH_CHECK(
      input_2d.dim() == 2,
      "Vision backbone attention projection expects flattened rank-2 input");

  const int64_t embed_dim = input_2d.size(-1);
  TORCH_CHECK(
      embed_dim % context->num_heads() == 0,
      "Vision backbone block context expects embed_dim divisible by num_heads");
  const int64_t head_dim = embed_dim / context->num_heads();
  const bool use_program_scratch =
      vision_program && vision_program->defined() &&
      vision_program->scratch_arena().has_value();
  Tensor attention_output;
  if (batch_size == 1) {
    const bool use_scratch_qkv_projection =
        use_program_scratch &&
        input_2d.scalar_type() == kFloat && context->qkv_bias().defined();

    std::optional<utils::VulkanScratchSlice> mixed_qkv_slice;
    Tensor mixed_qkv_output;
    if (use_scratch_qkv_projection) {
      auto scratch_qkv_output = reserve_scratch_buffer_tensor(
          *vision_program->scratch_arena(),
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
            *vision_program->scratch_arena(),
            {context->num_heads(), token_count, head_dim},
            kFloat);
        auto [k_slice, k_output] = reserve_scratch_buffer_tensor(
            *vision_program->scratch_arena(),
            {context->num_heads(), token_count, head_dim},
            kFloat);
        auto [v_slice, v_output] = reserve_scratch_buffer_tensor(
            *vision_program->scratch_arena(),
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
        q, k, v, vision_program);
    Tensor scratch_merge_output;
    Tensor* merge_output_opt = nullptr;
    if (use_scratch_qkv_projection && mixed_qkv_slice.has_value()) {
      scratch_merge_output = make_scratch_buffer_alias(
          *vision_program->scratch_arena(),
          *mixed_qkv_slice,
          {batch_size * token_count, embed_dim},
          attention_output.scalar_type());
      merge_output_opt = &scratch_merge_output;
    } else if (use_program_scratch) {
      auto [merge_slice, merge_output] = reserve_scratch_buffer_tensor(
          *vision_program->scratch_arena(),
          {batch_size * token_count, embed_dim},
          attention_output.scalar_type());
      (void)merge_slice;
      scratch_merge_output = std::move(merge_output);
      merge_output_opt = &scratch_merge_output;
    } else if (vision_program && vision_program->defined()) {
      merge_output_opt = &vision_program->qkv_output();
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
      q, k, v, vision_program);
  Tensor scratch_merge_output;
  Tensor* merge_output_opt = nullptr;
  if (use_program_scratch) {
    auto [merge_slice, merge_output] = reserve_scratch_buffer_tensor(
        *vision_program->scratch_arena(),
        {batch_size * token_count, embed_dim},
        attention_output.scalar_type());
    (void)merge_slice;
    scratch_merge_output = std::move(merge_output);
    merge_output_opt = &scratch_merge_output;
  } else if (vision_program && vision_program->defined()) {
    merge_output_opt = &vision_program->qkv_output();
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
  auto vision_program = prime_vision_backbone_program(input, context);
  if (vision_program.defined()) {
    if (vision_program.scratch_arena().has_value()) {
      vision_program.scratch_arena()->reset();
    }
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_vision_backbone_block_context.program");
  }

  const bool use_2d_input = input.dim() == 2;
  const int64_t batch_size = use_2d_input ? 1 : input.size(0);
  const int64_t token_count = use_2d_input ? input.size(0) : input.size(1);
  const int64_t embed_dim = input.size(-1);
  const int64_t hidden_rows = batch_size * token_count;
  Tensor input_2d = use_2d_input ? input : input.reshape({hidden_rows, embed_dim});
  utils::VisionBackboneProgram* const vision_program_ptr =
      vision_program.defined() ? &vision_program : nullptr;

  const std::array<int64_t, 1> normalized_shape = {embed_dim};
  Tensor attention_input = vision_program_ptr
      ? run_layernorm_context_out(
            input_2d,
            normalized_shape,
            context->norm1_context(),
            vision_program.norm1_output())
      : run_layernorm_context(input_2d, normalized_shape, context->norm1_context());
  Tensor attention_output = run_attention_projection(
      attention_input,
      batch_size,
      token_count,
      context,
      vision_program_ptr);
  attention_output = maybe_apply_layerscale(attention_output, context->ls1_gamma());
  Tensor hidden_states = at::add(input_2d, attention_output);

  Tensor mlp_input = vision_program_ptr
      ? run_layernorm_context_out(
            hidden_states,
            normalized_shape,
            context->norm2_context(),
            vision_program.norm2_output())
      : run_layernorm_context(
            hidden_states, normalized_shape, context->norm2_context());
  Tensor mlp_output = vision_program_ptr
      ? run_linear_gelu_context_out(
            mlp_input, context->fc1_context(), vision_program.fc1_output())
      : run_linear_gelu_context(mlp_input, context->fc1_context());
  mlp_output = vision_program_ptr
      ? run_linear_context_out(
            mlp_output, context->fc2_context(), vision_program.fc2_output())
      : run_linear_context(mlp_output, context->fc2_context());
  mlp_output = maybe_apply_layerscale(mlp_output, context->ls2_gamma());

  Tensor output = at::add(hidden_states, mlp_output);
  if (!use_2d_input) {
    output = output.reshape({batch_size, token_count, embed_dim});
  }
  utils::log_vulkan_op_hit("vulkan_prepack::run_vision_backbone_block_context");
  return maybe_restore_tensor(output, output_device, output_dtype);
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

  const size_t requested_scratch_bytes = vision_decoder_fusion_block_scratch_bytes(
      main_input, skip_tensor, target_sizes);
  auto scratch_arena = maybe_create_vision_decoder_scratch(
      context->allocation_label(), requested_scratch_bytes);
  if (!scratch_arena.has_value()) {
    return fallback(main_input, skip_tensor);
  }

  scratch_arena->reset();

  if (skip_tensor.has_value() && skip_tensor->defined()) {
    auto [skip_conv1_slice, skip_conv1] = reserve_scratch_buffer_tensor(
        *scratch_arena, skip_tensor->sizes(), kFloat);
    auto [skip_conv2_slice, skip_conv2] = reserve_scratch_buffer_tensor(
        *scratch_arena, skip_tensor->sizes(), kFloat);
    auto [skip_res_slice, skip_res] = reserve_scratch_buffer_tensor(
        *scratch_arena, skip_tensor->sizes(), kFloat);
    auto [main_input_slice, main_input_buffer] = reserve_scratch_buffer_tensor(
        *scratch_arena, main_input.sizes(), kFloat);
    (void)skip_conv1_slice;
    (void)skip_conv2_slice;
    (void)skip_res_slice;
    (void)main_input_slice;

    Tensor residual = at::relu(*skip_tensor);
    residual =
        run_conv2d_context_out(residual, context->res1_conv1_context(), skip_conv1);
    residual = at::relu(residual);
    residual =
        run_conv2d_context_out(residual, context->res1_conv2_context(), skip_conv2);
    residual = add_buffer_out_vulkan(residual, *skip_tensor, skip_res);
    main_input = add_buffer_out_vulkan(main_input, residual, main_input_buffer);
  }

  auto [main_conv1_slice, main_conv1] =
      reserve_scratch_buffer_tensor(*scratch_arena, main_input.sizes(), kFloat);
  auto [main_conv2_slice, main_conv2] =
      reserve_scratch_buffer_tensor(*scratch_arena, main_input.sizes(), kFloat);
  auto [main_res_slice, main_res] =
      reserve_scratch_buffer_tensor(*scratch_arena, main_input.sizes(), kFloat);
  auto [upsample_slice, upsample_output] = reserve_scratch_buffer_tensor(
      *scratch_arena,
      {main_input.size(0), main_input.size(1), target_sizes[0], target_sizes[1]},
      kFloat);
  (void)main_conv1_slice;
  (void)main_conv2_slice;
  (void)main_res_slice;
  (void)upsample_slice;

  Tensor output = at::relu(main_input);
  output = run_conv2d_context_out(output, context->res2_conv1_context(), main_conv1);
  output = at::relu(output);
  output = run_conv2d_context_out(output, context->res2_conv2_context(), main_conv2);
  output = add_buffer_out_vulkan(output, main_input, main_res);
  output = upsample_bilinear2d_buffer_out_vulkan(
      output,
      target_sizes,
      context->align_corners(),
      std::nullopt,
      std::nullopt,
      upsample_output);
  output = run_conv2d_context(output, context->out_conv_context());
  utils::log_vulkan_op_hit("vulkan_prepack::run_vision_decoder_fusion_block_context");
  return maybe_restore_tensor(output, output_device, output_dtype);
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
