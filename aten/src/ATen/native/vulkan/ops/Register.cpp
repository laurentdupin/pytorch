#ifdef USE_VULKAN_API

#include <ATen/Functions.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/vulkan/ops/Batchnorm.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/vulkan/ops/GatedDelta.h>
#include <ATen/native/vulkan/ops/Gru.h>
#include <ATen/native/vulkan/ops/Layernorm.h>
#include <ATen/native/vulkan/ops/Lstm.h>
#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/QwenLinearAttention.h>
#include <ATen/native/vulkan/ops/QuantizedFunctions.h>
#include <ATen/native/vulkan/ops/Register.h>
#include <ATen/native/vulkan/planning/ExecutionObjects.h>
#include <ATen/native/vulkan/planning/Runtime.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <cmath>
#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

int register_vulkan_conv2d_packed_context() {
  static auto register_vulkan_conv2d_context =
      torch::selective_class_<Conv2dPackedContext>(
          "vulkan", TORCH_SELECTIVE_CLASS("Conv2dPackedContext"))
          .def_pickle(
              // __getstate__
              [](const c10::intrusive_ptr<Conv2dPackedContext>& context) {
                // context is packed
                return context->unpack();
              },
              // __setstate__
              [](c10::impl::GenericList state) {
                // state is unpacked
                return c10::make_intrusive<Conv2dPackedContext>(
                    Conv2dPackedContext::pack(state));
              });
  return 0;
}

int register_vulkan_conv1d_packed_context() {
  static auto register_vulkan_conv1d_context =
      torch::selective_class_<Conv1dPackedContext>(
          "vulkan", TORCH_SELECTIVE_CLASS("Conv1dPackedContext"))
          .def_pickle(
              // __getstate__
              [](const c10::intrusive_ptr<Conv1dPackedContext>& context) {
                // context is packed
                return context->unpack();
              },
              // __setstate__
              [](c10::impl::GenericList state) {
                // state is unpacked
                return c10::make_intrusive<Conv1dPackedContext>(
                    Conv1dPackedContext::pack(state));
              });
  return 0;
}

int register_vulkan_linear_packed_context() {
  static auto register_vulkan_linear_context =
      torch::selective_class_<LinearPackedContext>(
          "vulkan", TORCH_SELECTIVE_CLASS("LinearPackedContext"))
          .def_pickle(
              // __getstate__
              [](const c10::intrusive_ptr<LinearPackedContext>& context) {
                // context is packed
                return context->unpack();
              },
              // __setstate__
              [](c10::impl::GenericList state) {
                // state is unpacked
                return c10::make_intrusive<LinearPackedContext>(
                    LinearPackedContext::pack(state));
              });
  return 0;
}

int register_vulkan_layernorm_packed_context() {
  static auto register_vulkan_layernorm_context =
      torch::selective_class_<LayernormPackedContext>(
          "vulkan", TORCH_SELECTIVE_CLASS("LayernormPackedContext"))
          .def_pickle(
              // __getstate__
              [](const c10::intrusive_ptr<LayernormPackedContext>& context) {
                // context is packed
                return context->unpack();
              },
              // __setstate__
              [](c10::impl::GenericList state) {
                // state is unpacked
                return c10::make_intrusive<LayernormPackedContext>(
                    LayernormPackedContext::pack(state));
              });
  return 0;
}

int register_vulkan_qwen_linear_attention_prefill_packed_context() {
  static auto register_vulkan_qwen_linear_attention_prefill_context =
      torch::selective_class_<QwenLinearAttentionPrefillPackedContext>(
          "vulkan",
          TORCH_SELECTIVE_CLASS("QwenLinearAttentionPrefillPackedContext"))
          .def_pickle(
              [](const c10::intrusive_ptr<
                     QwenLinearAttentionPrefillPackedContext>& context) {
                return context->unpack();
              },
              [](c10::impl::GenericList state) {
                return c10::make_intrusive<
                    QwenLinearAttentionPrefillPackedContext>(
                    QwenLinearAttentionPrefillPackedContext::pack(state));
              });
  return 0;
}

namespace {

utils::VulkanPlanningRequest make_runtime_planning_request(
    const int64_t workload_class,
    const int64_t model_domain,
    const int64_t execution_phase,
    const int64_t tensor_role) {
  return utils::make_vulkan_planning_request(
      static_cast<utils::VulkanWorkloadClass>(workload_class),
      static_cast<utils::VulkanTensorRole>(tensor_role),
      static_cast<utils::VulkanModelDomain>(model_domain),
      static_cast<utils::VulkanExecutionPhase>(execution_phase));
}

std::vector<int64_t> query_runtime_policy(
    const Tensor& prototype,
    const int64_t workload_class,
    const int64_t model_domain,
    const int64_t execution_phase,
    const int64_t tensor_role) {
  (void)prototype;
  const auto request = make_runtime_planning_request(
      workload_class, model_domain, execution_phase, tensor_role);
  const auto policy = utils::build_vulkan_runtime_policy(request);
  const auto kv_cache_plan = policy.kv_cache_plan.value_or(
      utils::VulkanKVCachePlanningDesc{});
  const auto scratch_arena_plan = policy.scratch_arena_plan.value_or(
      utils::VulkanScratchArenaPlanningDesc{});
  const auto boundary_plan = policy.boundary_plan.value_or(
      utils::VulkanBoundaryPlan{});

  return {
      static_cast<int64_t>(policy.backend_route),
      policy.kv_cache_plan.has_value() ? 1 : 0,
      kv_cache_plan.prefer_persistent_object ? 1 : 0,
      kv_cache_plan.prefer_buffer_storage ? 1 : 0,
      kv_cache_plan.prefer_append_views ? 1 : 0,
      kv_cache_plan.prefer_decode_cursor ? 1 : 0,
      policy.scratch_arena_plan.has_value() ? 1 : 0,
      scratch_arena_plan.prefer_reusable_arena ? 1 : 0,
      scratch_arena_plan.prefer_buffer_storage ? 1 : 0,
      static_cast<int64_t>(scratch_arena_plan.min_arena_bytes),
      static_cast<int64_t>(scratch_arena_plan.alignment),
      static_cast<int64_t>(policy.linear_kernel_family),
      static_cast<int64_t>(policy.norm_kernel_family),
      static_cast<int64_t>(policy.attention_kernel_family),
      policy.boundary_plan.has_value() ? 1 : 0,
      static_cast<int64_t>(boundary_plan.kind),
      static_cast<int64_t>(boundary_plan.input_transfer_layout),
      static_cast<int64_t>(boundary_plan.output_transfer_layout),
      boundary_plan.prefer_backend_owned_execution ? 1 : 0,
      boundary_plan.requires_scratch_arena ? 1 : 0,
      static_cast<int64_t>(boundary_plan.preferred_cpu_threads),
  };
}

Tensor create_kv_cache_storage_for_request(
    const Tensor& prototype,
    IntArrayRef sizes,
    const int64_t sequence_dim,
    const int64_t workload_class,
    const int64_t model_domain,
    const int64_t execution_phase,
    const int64_t tensor_role) {
  const auto request = make_runtime_planning_request(
      workload_class, model_domain, execution_phase, tensor_role);
  const auto policy = utils::build_vulkan_runtime_policy(request);
  TORCH_CHECK(
      policy.kv_cache_plan.has_value(),
      "Vulkan runtime policy does not expose a KV cache plan for the requested workload");

  const auto& desc = *policy.kv_cache_plan;
  const auto storage_type =
      desc.prefer_buffer_storage ? api::StorageType::BUFFER
                                 : api::StorageType::TEXTURE_3D;
  const auto execution_layout =
      desc.prefer_buffer_storage ? api::ExecutionLayout::BUFFER_DIRECT
                                 : api::ExecutionLayout::TEXTURE;
  const auto memory_layout =
      desc.prefer_buffer_storage
      ? api::GPUMemoryLayout::TENSOR_WIDTH_PACKED
      : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;

  auto cache_object = utils::create_vulkan_kv_cache_object(
      utils::VulkanKVCacheSpec{
          prototype.scalar_type(),
          sizes.vec(),
          sequence_dim,
          execution_layout,
          memory_layout,
          storage_type,
          desc.prefer_persistent_object,
      });
  return cache_object.storage();
}

Tensor create_scratch_arena_storage_for_request(
    const Tensor& prototype,
    const int64_t num_bytes,
    const int64_t alignment,
    const int64_t workload_class,
    const int64_t model_domain,
    const int64_t execution_phase,
    const int64_t tensor_role) {
  (void)prototype;
  const auto request = make_runtime_planning_request(
      workload_class, model_domain, execution_phase, tensor_role);
  const auto policy = utils::build_vulkan_runtime_policy(request);
  TORCH_CHECK(
      policy.scratch_arena_plan.has_value(),
      "Vulkan runtime policy does not expose a scratch arena plan for the requested workload");

  const auto& desc = *policy.scratch_arena_plan;
  const uint32_t requested_alignment =
      alignment > 0 ? static_cast<uint32_t>(alignment) : desc.alignment;
  const auto storage_type =
      desc.prefer_buffer_storage ? api::StorageType::BUFFER
                                 : api::StorageType::TEXTURE_3D;
  const auto execution_layout =
      desc.prefer_buffer_storage ? api::ExecutionLayout::BUFFER_DIRECT
                                 : api::ExecutionLayout::TEXTURE;
  const auto memory_layout =
      desc.prefer_buffer_storage
      ? api::GPUMemoryLayout::TENSOR_WIDTH_PACKED
      : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;
  auto scratch_arena = utils::create_vulkan_scratch_arena(
      utils::VulkanScratchArenaSpec{
          kByte,
          static_cast<size_t>(std::max<int64_t>(num_bytes, desc.min_arena_bytes)),
          requested_alignment,
          execution_layout,
          memory_layout,
          storage_type,
          desc.prefer_reusable_arena,
      });
  return scratch_arena.storage();
}

Tensor maybe_move_runtime_tensor_to_device(
    const Tensor& tensor,
    const Device& device) {
  return device.type() == kCPU ? tensor : tensor.to(device);
}

Tensor create_causal_attention_mask_runtime(
    const Tensor& prototype,
    const int64_t batch_size,
    const int64_t q_length,
    const int64_t kv_length,
    const int64_t q_offset,
    const int64_t kv_offset,
    const bool float_mask) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  TORCH_CHECK(
      q_length >= 0 && kv_length >= 0,
      "vulkan_prepack::create_causal_attention_mask expects non-negative lengths");
  TORCH_CHECK(
      batch_size >= 0,
      "vulkan_prepack::create_causal_attention_mask expects a non-negative batch size");

  const Device output_device = prototype.device();
  const int64_t normalized_batch = std::max<int64_t>(batch_size, 1);
  const auto cpu_options = prototype.options().device(kCPU);

  Tensor q_positions = at::arange(q_length, cpu_options.dtype(kLong));
  Tensor kv_positions = at::arange(kv_length, cpu_options.dtype(kLong));
  if (q_offset != 0) {
    q_positions = at::add(q_positions, q_offset);
  }
  if (kv_offset != 0) {
    kv_positions = at::add(kv_positions, kv_offset);
  }
  const Tensor keep_mask = q_positions.unsqueeze(1)
                               .ge(kv_positions.unsqueeze(0))
                               .unsqueeze(0)
                               .unsqueeze(0)
                               .expand({normalized_batch, 1, q_length, kv_length})
                               .contiguous();

  if (!float_mask) {
    if (output_device.type() != kCPU) {
      utils::log_vulkan_op_hit("vulkan_prepack::create_causal_attention_mask");
    }
    return maybe_move_runtime_tensor_to_device(keep_mask, output_device);
  }

  ScalarType mask_dtype = prototype.scalar_type();
  if (!at::isFloatingType(mask_dtype)) {
    mask_dtype = kFloat;
  }
  Tensor additive_mask = at::zeros(
      {normalized_batch, 1, q_length, kv_length},
      cpu_options.dtype(mask_dtype));
  additive_mask.masked_fill_(
      keep_mask.logical_not(),
      -std::numeric_limits<float>::infinity());

  if (output_device.type() != kCPU) {
    utils::log_vulkan_op_hit("vulkan_prepack::create_causal_attention_mask");
  }
  return maybe_move_runtime_tensor_to_device(additive_mask, output_device);
}

Tensor slice_hidden_states_for_logits_runtime(
    const Tensor& hidden_states_arg,
    const int64_t logits_to_keep) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Device output_device = hidden_states_arg.device();
  const Tensor hidden_states =
      (hidden_states_arg.is_vulkan() ? hidden_states_arg.cpu() : hidden_states_arg)
          .contiguous();
  TORCH_CHECK(
      hidden_states.dim() == 3,
      "vulkan_prepack::slice_hidden_states_for_logits expects a [B, T, H] tensor");

  Tensor result = hidden_states;
  if (logits_to_keep > 0 && logits_to_keep < hidden_states.size(1)) {
    const int64_t start = std::max<int64_t>(hidden_states.size(1) - logits_to_keep, 0);
    result = hidden_states.narrow(1, start, hidden_states.size(1) - start);
  }

  if (output_device.type() != kCPU) {
    utils::log_vulkan_op_hit("vulkan_prepack::slice_hidden_states_for_logits");
  }
  return maybe_move_runtime_tensor_to_device(result, output_device);
}

Tensor index_select_hidden_states_for_logits_runtime(
    const Tensor& hidden_states_arg,
    const Tensor& index_arg) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Device output_device = hidden_states_arg.device();
  const Tensor hidden_states =
      (hidden_states_arg.is_vulkan() ? hidden_states_arg.cpu() : hidden_states_arg)
          .contiguous();
  Tensor index = index_arg.is_vulkan() ? index_arg.cpu() : index_arg;
  TORCH_CHECK(
      hidden_states.dim() == 3,
      "vulkan_prepack::index_select_hidden_states_for_logits expects a [B, T, H] tensor");
  TORCH_CHECK(
      index.dim() == 1,
      "vulkan_prepack::index_select_hidden_states_for_logits expects a 1D index tensor");
  TORCH_CHECK(
      index.scalar_type() == kLong || index.scalar_type() == kInt,
      "vulkan_prepack::index_select_hidden_states_for_logits expects int32 or int64 indices");

  index = index.contiguous().to(kLong);
  const Tensor result = at::index_select(hidden_states, 1, index);

  if (output_device.type() != kCPU) {
    utils::log_vulkan_op_hit(
        "vulkan_prepack::index_select_hidden_states_for_logits");
  }
  return maybe_move_runtime_tensor_to_device(result, output_device);
}

Tensor gather_hidden_states_by_batch_positions_runtime(
    const Tensor& hidden_states_arg,
    const Tensor& positions_arg) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Device output_device = hidden_states_arg.device();
  const Tensor hidden_states =
      (hidden_states_arg.is_vulkan() ? hidden_states_arg.cpu() : hidden_states_arg)
          .contiguous();
  Tensor positions = positions_arg.is_vulkan() ? positions_arg.cpu() : positions_arg;
  TORCH_CHECK(
      hidden_states.dim() == 3,
      "vulkan_prepack::gather_hidden_states_by_batch_positions expects a [B, T, H] tensor");
  TORCH_CHECK(
      positions.dim() == 1,
      "vulkan_prepack::gather_hidden_states_by_batch_positions expects a 1D positions tensor");
  TORCH_CHECK(
      positions.scalar_type() == kLong || positions.scalar_type() == kInt,
      "vulkan_prepack::gather_hidden_states_by_batch_positions expects int32 or int64 positions");
  TORCH_CHECK(
      positions.size(0) == hidden_states.size(0),
      "vulkan_prepack::gather_hidden_states_by_batch_positions expects one position per batch item");

  positions = positions.contiguous();
  std::vector<Tensor> gathered_rows;
  gathered_rows.reserve(hidden_states.size(0));
  if (positions.scalar_type() == kLong) {
    const int64_t* const pos_ptr = positions.const_data_ptr<int64_t>();
    for (const auto batch_idx : c10::irange(hidden_states.size(0))) {
      const int64_t token_idx = pos_ptr[batch_idx];
      TORCH_CHECK_INDEX(
          token_idx >= 0 && token_idx < hidden_states.size(1),
          "vulkan_prepack::gather_hidden_states_by_batch_positions index ",
          token_idx,
          " is out of bounds for sequence length ",
          hidden_states.size(1));
      gathered_rows.push_back(at::select(at::select(hidden_states, 0, batch_idx), 0, token_idx));
    }
  } else {
    const int32_t* const pos_ptr = positions.const_data_ptr<int32_t>();
    for (const auto batch_idx : c10::irange(hidden_states.size(0))) {
      const int64_t token_idx = pos_ptr[batch_idx];
      TORCH_CHECK_INDEX(
          token_idx >= 0 && token_idx < hidden_states.size(1),
          "vulkan_prepack::gather_hidden_states_by_batch_positions index ",
          token_idx,
          " is out of bounds for sequence length ",
          hidden_states.size(1));
      gathered_rows.push_back(at::select(at::select(hidden_states, 0, batch_idx), 0, token_idx));
    }
  }

  const Tensor result = at::stack(gathered_rows, 0);
  if (output_device.type() != kCPU) {
    utils::log_vulkan_op_hit(
        "vulkan_prepack::gather_hidden_states_by_batch_positions");
  }
  return maybe_move_runtime_tensor_to_device(result, output_device);
}

int64_t find_timestep_index_runtime(
    const Tensor& schedule_timesteps_arg,
    const Tensor& timestep_arg) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Tensor schedule_timesteps =
      (schedule_timesteps_arg.is_vulkan() ? schedule_timesteps_arg.cpu()
                                          : schedule_timesteps_arg)
          .contiguous();
  Tensor timestep = timestep_arg.is_vulkan() ? timestep_arg.cpu() : timestep_arg;
  TORCH_CHECK(
      schedule_timesteps.dim() == 1,
      "vulkan_prepack::find_timestep_index expects a 1D schedule tensor");
  TORCH_CHECK(
      timestep.numel() == 1,
      "vulkan_prepack::find_timestep_index expects a scalar timestep tensor");

  timestep = timestep.reshape({}).to(schedule_timesteps.scalar_type());
  const Tensor indices = at::eq(schedule_timesteps, timestep).nonzero();
  TORCH_CHECK(
      indices.numel() > 0,
      "vulkan_prepack::find_timestep_index could not find the requested timestep");

  const int64_t pos = indices.size(0) > 1 ? 1 : 0;
  return indices.select(0, pos).item<int64_t>();
}

std::tuple<Tensor, Tensor, Tensor> compute_moe_router_runtime(
    const Tensor& logits_arg,
    const int64_t top_k,
    const int64_t num_experts) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Device output_device = logits_arg.device();
  const Tensor logits =
      (logits_arg.is_vulkan() ? logits_arg.cpu() : logits_arg).contiguous().to(kFloat);
  TORCH_CHECK(
      logits.dim() == 2,
      "vulkan_prepack::compute_moe_router expects a [T, E] logits tensor");
  TORCH_CHECK(
      num_experts == logits.size(1),
      "vulkan_prepack::compute_moe_router expects num_experts to match the logits width");
  TORCH_CHECK(
      top_k > 0 && top_k <= num_experts,
      "vulkan_prepack::compute_moe_router expects top_k to be in [1, num_experts]");

  const auto topk = at::topk(logits, top_k, 1, true, true);
  const Tensor top_k_logits = std::get<0>(topk);
  const Tensor top_k_indices = std::get<1>(topk);
  const Tensor top_k_gates = at::softmax(top_k_logits, 1);
  const Tensor gates =
      at::zeros({top_k_indices.size(0), num_experts}, top_k_gates.options())
          .scatter(1, top_k_indices, 1);
  const Tensor expert_size = gates.to(kLong).sum(0).contiguous();

  const Tensor top_k_experts = top_k_indices.flatten();
  const auto sorted_experts = at::sort(top_k_experts, 0, false);
  const Tensor index_sorted_experts = std::get<1>(sorted_experts);
  const Tensor batch_index =
      at::floor_divide(index_sorted_experts, top_k).to(kLong).contiguous();
  const Tensor batch_gates =
      top_k_gates.flatten().index_select(0, index_sorted_experts).contiguous();

  if (output_device.type() != kCPU) {
    utils::log_vulkan_op_hit("vulkan_prepack::compute_moe_router");
  }

  return {
      batch_index,
      maybe_move_runtime_tensor_to_device(batch_gates, output_device),
      expert_size,
  };
}

Tensor accumulate_expert_outputs_runtime(
    const Tensor& expert_outputs_arg,
    const Tensor& batch_index_arg,
    const int64_t total_rows) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Device output_device = expert_outputs_arg.device();
  const Tensor expert_outputs =
      (expert_outputs_arg.is_vulkan() ? expert_outputs_arg.cpu() : expert_outputs_arg)
          .contiguous();
  const Tensor batch_index =
      (batch_index_arg.is_vulkan() ? batch_index_arg.cpu() : batch_index_arg)
          .contiguous()
          .to(kLong);
  TORCH_CHECK(
      expert_outputs.dim() == 2,
      "vulkan_prepack::accumulate_expert_outputs expects a [N, H] tensor");
  TORCH_CHECK(
      batch_index.dim() == 1 && batch_index.numel() == expert_outputs.size(0),
      "vulkan_prepack::accumulate_expert_outputs expects one row index per expert output");
  TORCH_CHECK(
      total_rows >= 0,
      "vulkan_prepack::accumulate_expert_outputs expects total_rows >= 0");

  const Tensor result =
      at::zeros({total_rows, expert_outputs.size(1)}, expert_outputs.options())
          .index_add(0, batch_index, expert_outputs);
  if (output_device.type() != kCPU) {
    utils::log_vulkan_op_hit("vulkan_prepack::accumulate_expert_outputs");
  }
  return maybe_move_runtime_tensor_to_device(result, output_device);
}

std::tuple<Tensor, Tensor> compute_rotary_cos_sin_runtime(
    const Tensor& prototype_arg,
    const Tensor& inv_freq_arg,
    const Tensor& position_ids_arg,
    const double attention_scaling) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Device output_device = prototype_arg.device();
  const ScalarType output_dtype = prototype_arg.scalar_type();
  const Tensor inv_freq =
      (inv_freq_arg.is_vulkan() ? inv_freq_arg.cpu() : inv_freq_arg).contiguous().to(kFloat);
  const Tensor position_ids =
      (position_ids_arg.is_vulkan() ? position_ids_arg.cpu() : position_ids_arg)
          .contiguous()
          .to(kFloat);
  TORCH_CHECK(
      inv_freq.dim() == 1,
      "vulkan_prepack::compute_rotary_cos_sin expects a 1D inv_freq tensor");
  TORCH_CHECK(
      position_ids.dim() == 2,
      "vulkan_prepack::compute_rotary_cos_sin expects a [B, T] position_ids tensor");

  const Tensor inv_freq_expanded =
      inv_freq.unsqueeze(0).unsqueeze(-1).expand({position_ids.size(0), -1, 1});
  const Tensor position_ids_expanded = position_ids.unsqueeze(1);
  const Tensor freqs = at::matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2);
  const Tensor emb = at::cat({freqs, freqs}, -1);
  Tensor cos = at::cos(emb);
  Tensor sin = at::sin(emb);
  if (attention_scaling != 1.0) {
    cos = cos.mul(attention_scaling);
    sin = sin.mul(attention_scaling);
  }

  if (output_device.type() != kCPU) {
    utils::log_vulkan_op_hit("vulkan_prepack::compute_rotary_cos_sin");
  }
  return {
      maybe_move_runtime_tensor_to_device(cos, output_device).to(output_dtype),
      maybe_move_runtime_tensor_to_device(sin, output_device).to(output_dtype),
  };
}

std::tuple<Tensor, Tensor> pose_encoding_to_extri_intri_runtime(
    const Tensor& pose_encoding_arg,
    const int64_t height,
    const int64_t width) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Device output_device = pose_encoding_arg.device();
  const Tensor pose_encoding =
      (pose_encoding_arg.is_vulkan() ? pose_encoding_arg.cpu() : pose_encoding_arg)
          .contiguous()
          .to(kFloat);
  TORCH_CHECK(
      pose_encoding.dim() == 3 && pose_encoding.size(-1) == 9,
      "vulkan_prepack::pose_encoding_to_extri_intri expects a [B, N, 9] tensor");

  const auto batch = pose_encoding.size(0);
  const auto views = pose_encoding.size(1);
  Tensor extrinsics = at::empty({batch, views, 3, 4}, pose_encoding.options());
  Tensor intrinsics = at::zeros({batch, views, 3, 3}, pose_encoding.options());

  const float* pose_ptr = pose_encoding.const_data_ptr<float>();
  float* extr_ptr = extrinsics.data_ptr<float>();
  float* intr_ptr = intrinsics.data_ptr<float>();

  for (const auto b : c10::irange(batch)) {
    for (const auto n : c10::irange(views)) {
      const int64_t pose_offset = (b * views + n) * 9;
      const int64_t extr_offset = (b * views + n) * 12;
      const int64_t intr_offset = (b * views + n) * 9;

      const float tx = pose_ptr[pose_offset + 0];
      const float ty = pose_ptr[pose_offset + 1];
      const float tz = pose_ptr[pose_offset + 2];
      const float i = pose_ptr[pose_offset + 3];
      const float j = pose_ptr[pose_offset + 4];
      const float k = pose_ptr[pose_offset + 5];
      const float r = pose_ptr[pose_offset + 6];
      const float fov_h = pose_ptr[pose_offset + 7];
      const float fov_w = pose_ptr[pose_offset + 8];

      const float two_s =
          2.0f / std::max(i * i + j * j + k * k + r * r, 1.0e-12f);

      extr_ptr[extr_offset + 0] = 1.0f - two_s * (j * j + k * k);
      extr_ptr[extr_offset + 1] = two_s * (i * j - k * r);
      extr_ptr[extr_offset + 2] = two_s * (i * k + j * r);
      extr_ptr[extr_offset + 3] = tx;
      extr_ptr[extr_offset + 4] = two_s * (i * j + k * r);
      extr_ptr[extr_offset + 5] = 1.0f - two_s * (i * i + k * k);
      extr_ptr[extr_offset + 6] = two_s * (j * k - i * r);
      extr_ptr[extr_offset + 7] = ty;
      extr_ptr[extr_offset + 8] = two_s * (i * k - j * r);
      extr_ptr[extr_offset + 9] = two_s * (j * k + i * r);
      extr_ptr[extr_offset + 10] = 1.0f - two_s * (i * i + j * j);
      extr_ptr[extr_offset + 11] = tz;

      const float fy = (static_cast<float>(height) * 0.5f) /
          std::max(std::tan(fov_h * 0.5f), 1.0e-6f);
      const float fx = (static_cast<float>(width) * 0.5f) /
          std::max(std::tan(fov_w * 0.5f), 1.0e-6f);

      intr_ptr[intr_offset + 0] = fx;
      intr_ptr[intr_offset + 1] = 0.0f;
      intr_ptr[intr_offset + 2] = static_cast<float>(width) * 0.5f;
      intr_ptr[intr_offset + 3] = 0.0f;
      intr_ptr[intr_offset + 4] = fy;
      intr_ptr[intr_offset + 5] = static_cast<float>(height) * 0.5f;
      intr_ptr[intr_offset + 6] = 0.0f;
      intr_ptr[intr_offset + 7] = 0.0f;
      intr_ptr[intr_offset + 8] = 1.0f;
    }
  }

  return {
      maybe_move_runtime_tensor_to_device(extrinsics, output_device),
      maybe_move_runtime_tensor_to_device(intrinsics, output_device),
  };
}

Tensor extri_intri_to_pose_encoding_runtime(
    const Tensor& extrinsics_arg,
    const Tensor& intrinsics_arg,
    const int64_t height,
    const int64_t width) {
  c10::impl::ExcludeDispatchKeyGuard no_vulkan(c10::DispatchKey::Vulkan);
  c10::InferenceMode inference_mode_guard(false);

  const Device output_device = extrinsics_arg.device();
  const Tensor extrinsics =
      (extrinsics_arg.is_vulkan() ? extrinsics_arg.cpu() : extrinsics_arg)
          .contiguous()
          .to(kFloat);
  const Tensor intrinsics =
      (intrinsics_arg.is_vulkan() ? intrinsics_arg.cpu() : intrinsics_arg)
          .contiguous()
          .to(kFloat);

  TORCH_CHECK(
      extrinsics.dim() == 4 && extrinsics.size(-2) == 3 && extrinsics.size(-1) == 4,
      "vulkan_prepack::extri_intri_to_pose_encoding expects extrinsics with shape [B, N, 3, 4]");
  TORCH_CHECK(
      intrinsics.dim() == 4 && intrinsics.size(-2) == 3 && intrinsics.size(-1) == 3,
      "vulkan_prepack::extri_intri_to_pose_encoding expects intrinsics with shape [B, N, 3, 3]");
  TORCH_CHECK(
      extrinsics.size(0) == intrinsics.size(0) &&
          extrinsics.size(1) == intrinsics.size(1),
      "vulkan_prepack::extri_intri_to_pose_encoding expects matching [B, N] dimensions");

  const auto batch = extrinsics.size(0);
  const auto views = extrinsics.size(1);
  Tensor pose_encoding = at::empty({batch, views, 9}, extrinsics.options());

  const float* extr_ptr = extrinsics.const_data_ptr<float>();
  const float* intr_ptr = intrinsics.const_data_ptr<float>();
  float* pose_ptr = pose_encoding.data_ptr<float>();

  for (const auto b : c10::irange(batch)) {
    for (const auto n : c10::irange(views)) {
      const int64_t extr_offset = (b * views + n) * 12;
      const int64_t intr_offset = (b * views + n) * 9;
      const int64_t pose_offset = (b * views + n) * 9;

      const float m00 = extr_ptr[extr_offset + 0];
      const float m01 = extr_ptr[extr_offset + 1];
      const float m02 = extr_ptr[extr_offset + 2];
      const float m10 = extr_ptr[extr_offset + 4];
      const float m11 = extr_ptr[extr_offset + 5];
      const float m12 = extr_ptr[extr_offset + 6];
      const float m20 = extr_ptr[extr_offset + 8];
      const float m21 = extr_ptr[extr_offset + 9];
      const float m22 = extr_ptr[extr_offset + 10];

      const float q_abs0 =
          std::sqrt(std::max(0.0f, 1.0f + m00 + m11 + m22));
      const float q_abs1 =
          std::sqrt(std::max(0.0f, 1.0f + m00 - m11 - m22));
      const float q_abs2 =
          std::sqrt(std::max(0.0f, 1.0f - m00 + m11 - m22));
      const float q_abs3 =
          std::sqrt(std::max(0.0f, 1.0f - m00 - m11 + m22));
      const float q_abs[4] = {q_abs0, q_abs1, q_abs2, q_abs3};

      int64_t best = 0;
      for (const auto candidate : c10::irange(1, 4)) {
        if (q_abs[candidate] > q_abs[best]) {
          best = candidate;
        }
      }

      float quat_rijk[4];
      switch (best) {
        case 0:
          quat_rijk[0] = q_abs0 * q_abs0;
          quat_rijk[1] = m21 - m12;
          quat_rijk[2] = m02 - m20;
          quat_rijk[3] = m10 - m01;
          break;
        case 1:
          quat_rijk[0] = m21 - m12;
          quat_rijk[1] = q_abs1 * q_abs1;
          quat_rijk[2] = m10 + m01;
          quat_rijk[3] = m02 + m20;
          break;
        case 2:
          quat_rijk[0] = m02 - m20;
          quat_rijk[1] = m10 + m01;
          quat_rijk[2] = q_abs2 * q_abs2;
          quat_rijk[3] = m12 + m21;
          break;
        default:
          quat_rijk[0] = m10 - m01;
          quat_rijk[1] = m20 + m02;
          quat_rijk[2] = m21 + m12;
          quat_rijk[3] = q_abs3 * q_abs3;
          break;
      }

      const float denom = 2.0f * std::max(q_abs[best], 0.1f);
      for (float& value : quat_rijk) {
        value /= denom;
      }

      float quat_xyzw[4] = {
          quat_rijk[1],
          quat_rijk[2],
          quat_rijk[3],
          quat_rijk[0],
      };
      if (quat_xyzw[3] < 0.0f) {
        for (float& value : quat_xyzw) {
          value = -value;
        }
      }

      pose_ptr[pose_offset + 0] = extr_ptr[extr_offset + 3];
      pose_ptr[pose_offset + 1] = extr_ptr[extr_offset + 7];
      pose_ptr[pose_offset + 2] = extr_ptr[extr_offset + 11];
      pose_ptr[pose_offset + 3] = quat_xyzw[0];
      pose_ptr[pose_offset + 4] = quat_xyzw[1];
      pose_ptr[pose_offset + 5] = quat_xyzw[2];
      pose_ptr[pose_offset + 6] = quat_xyzw[3];

      const float fy = intr_ptr[intr_offset + 4];
      const float fx = intr_ptr[intr_offset + 0];
      pose_ptr[pose_offset + 7] = 2.0f *
          std::atan((static_cast<float>(height) * 0.5f) / std::max(fy, 1.0e-6f));
      pose_ptr[pose_offset + 8] = 2.0f *
          std::atan((static_cast<float>(width) * 0.5f) / std::max(fx, 1.0e-6f));
    }
  }

  return maybe_move_runtime_tensor_to_device(pose_encoding, output_device);
}

TORCH_LIBRARY(vulkan, m) {
  m.class_<BatchNormPackedContext>("BatchNormPackedContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<BatchNormPackedContext>& context) {
            // context is packed
            return context->unpack();
          },
          // __setstate__
          [](c10::impl::GenericList state) {
            // state is unpacked
            return c10::make_intrusive<BatchNormPackedContext>(
                BatchNormPackedContext::pack(state));
          });
  m.class_<GruPackedContext>("GruPackedContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<GruPackedContext>& context) {
            // context is packed
            return context->unpack();
          },
          // __setstate__
          [](c10::impl::GenericList state) {
            // state is unpacked
            return c10::make_intrusive<GruPackedContext>(
                GruPackedContext::pack(state));
          });
  m.class_<LstmPackedContext>("LstmPackedContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<LstmPackedContext>& context) {
            // context is packed
            return context->unpack();
          },
          // __setstate__
          [](c10::impl::GenericList state) {
            // state is unpacked
            return c10::make_intrusive<LstmPackedContext>(
                LstmPackedContext::pack(state));
          });
  register_vulkan_conv2d_packed_context();
  register_vulkan_conv1d_packed_context();
  register_vulkan_linear_packed_context();
  register_vulkan_layernorm_packed_context();
  register_vulkan_qwen_linear_attention_prefill_packed_context();
  // To maintain backwards compatibility.
  m.class_<Conv2dOpContext>("Conv2dOpContext")
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<Conv2dOpContext>& context) {
            return context->unpack();
          },
          // __setstate__
          [](Conv2dOpContext::State state) {
            return std::apply(conv2d_clamp_prepack, std::move(state));
          });
}

TORCH_LIBRARY(vulkan_prepack, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_conv2d_context(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.vulkan.Conv2dPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA( // Backwards compatibility
      "vulkan_prepack::conv2d_clamp_prepack(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.vulkan.Conv2dOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_conv2d_context(Tensor X, "
      "__torch__.torch.classes.vulkan.Conv2dPackedContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA( // Backwards compatibility
      "vulkan_prepack::conv2d_clamp_run(Tensor X, "
      "__torch__.torch.classes.vulkan.Conv2dOpContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_tconv2d_context(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] output_padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.vulkan.Conv2dPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_tconv2d_context(Tensor X, "
      "__torch__.torch.classes.vulkan.Conv2dPackedContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_qconv2d_context(Tensor W, Tensor? B, "
      "int[2] stride, int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.vulkan.Conv2dPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_qconv2d_context(Tensor X, float scale, int zero_point, "
      "__torch__.torch.classes.vulkan.Conv2dPackedContext vk_context) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_conv1d_context(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups) "
      "-> __torch__.torch.classes.vulkan.Conv1dPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_conv1d_context(Tensor X, "
      "__torch__.torch.classes.vulkan.Conv1dPackedContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_qtconv2d_context(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] output_padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.vulkan.Conv2dPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_linear_context(Tensor W, Tensor? B) "
      "-> __torch__.torch.classes.vulkan.LinearPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_linear_context_labeled(Tensor W, Tensor? B, str label) "
      "-> __torch__.torch.classes.vulkan.LinearPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_qwen_linear_attention_prefill_context("
      "Tensor qkv_weight, Tensor z_weight, Tensor a_weight, Tensor b_weight, Tensor out_weight, "
      "Tensor conv_weight, Tensor? conv_bias, Tensor norm_weight, Tensor A_log, Tensor dt_bias, "
      "int key_dim, int value_dim, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads, "
      "int chunk_size=64, float norm_eps=1e-6, str label=\"\") "
      "-> __torch__.torch.classes.vulkan.QwenLinearAttentionPrefillPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_qwen_linear_attention_prefill_context("
      "Tensor X, __torch__.torch.classes.vulkan.QwenLinearAttentionPrefillPackedContext context) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_qwen_linear_attention_decode_context("
      "Tensor X, Tensor conv_state, Tensor recurrent_state, "
      "__torch__.torch.classes.vulkan.QwenLinearAttentionPrefillPackedContext context) -> (Tensor, Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_causal_attention_mask("
      "Tensor prototype, int batch_size, int q_length, int kv_length, int q_offset=0, int kv_offset=0, bool float_mask=True) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::slice_hidden_states_for_logits(Tensor hidden_states, int logits_to_keep) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::index_select_hidden_states_for_logits(Tensor hidden_states, Tensor index) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::gather_hidden_states_by_batch_positions(Tensor hidden_states, Tensor positions) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::find_timestep_index(Tensor schedule_timesteps, Tensor timestep) -> int"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::compute_moe_router(Tensor logits, int top_k, int num_experts) -> (Tensor, Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::accumulate_expert_outputs(Tensor expert_outputs, Tensor batch_index, int total_rows) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::compute_rotary_cos_sin(Tensor prototype, Tensor inv_freq, Tensor position_ids, float attention_scaling=1.0) -> (Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::to_vulkan_labeled(Tensor X, str label) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::query_runtime_policy(Tensor prototype, int workload_class, int model_domain, int execution_phase, int tensor_role) -> int[]"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_kv_cache_storage_for_request(Tensor prototype, int[] sizes, int sequence_dim, int workload_class, int model_domain, int execution_phase, int tensor_role) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_scratch_arena_storage_for_request(Tensor prototype, int num_bytes, int alignment, int workload_class, int model_domain, int execution_phase, int tensor_role) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::pose_encoding_to_extri_intri(Tensor pose_encoding, int height, int width) -> (Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::extri_intri_to_pose_encoding(Tensor extrinsics, Tensor intrinsics, int height, int width) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_scheduled_gated_delta_rule_chunk(Tensor query, Tensor key, Tensor value, Tensor g, Tensor beta, int chunk_size=64, Tensor? initial_state=None, bool output_final_state=False, bool use_qk_l2norm_in_kernel=False) -> (Tensor, Tensor?)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_scheduled_gated_delta_rule_recurrent(Tensor query, Tensor key, Tensor value, Tensor g, Tensor beta, Tensor? initial_state=None, bool output_final_state=False, bool use_qk_l2norm_in_kernel=False) -> (Tensor, Tensor?)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_linear_context(Tensor X, "
      "__torch__.torch.classes.vulkan.LinearPackedContext BW_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_linear_gelu_context(Tensor X, "
      "__torch__.torch.classes.vulkan.LinearPackedContext BW_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_qlinear_context(Tensor X, float scale, int zero_point, "
      "__torch__.torch.classes.vulkan.LinearPackedContext vk_context) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_layernorm_context(Tensor? W, Tensor? B, float eps) "
      "-> __torch__.torch.classes.vulkan.LayernormPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_layernorm_context(Tensor X, SymInt[] normalized_shape, "
      "__torch__.torch.classes.vulkan.LayernormPackedContext BW_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_gru_context(Tensor[] params_cpu, "
      "bool has_biases, "
      "int num_layers, "
      "float dropout, "
      "bool train, "
      "bool bidirectional, "
      "bool batch_first) "
      "-> __torch__.torch.classes.vulkan.GruPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_gru_context(Tensor input_vk, "
      "Tensor hx_vk, "
      "__torch__.torch.classes.vulkan.GruPackedContext G_prepack) -> (Tensor next_input, Tensor hidden_layer)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_lstm_context(Tensor[] params_cpu, "
      "bool has_biases, "
      "int num_layers, "
      "float dropout, "
      "bool train, "
      "bool bidirectional, "
      "bool batch_first) "
      "-> __torch__.torch.classes.vulkan.LstmPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_lstm_context(Tensor input_vk, "
      "Tensor hx_vk, "
      "Tensor cx_vk, "
      "__torch__.torch.classes.vulkan.LstmPackedContext L_prepack) -> (Tensor next_input, Tensor hidden_state, Tensor cell_state)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::create_batchnorm_context("
      "Tensor? weight_opt, "
      "Tensor? bias_opt, "
      "Tensor? running_mean_opt, "
      "Tensor? running_var_opt, "
      "bool training, "
      "float momentum, "
      "float eps, "
      "bool cudnn_enable) "
      "-> __torch__.torch.classes.vulkan.BatchNormPackedContext"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "vulkan_prepack::run_batchnorm_context("
      "Tensor input_vk, "
      "__torch__.torch.classes.vulkan.BatchNormPackedContext context) "
      "-> Tensor out"));
}

TORCH_LIBRARY_IMPL(vulkan_prepack, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_conv2d_context"),
      TORCH_FN(create_conv2d_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_clamp_prepack"),
      TORCH_FN(conv2d_clamp_prepack)); // Backwards compatibility
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_tconv2d_context"),
      TORCH_FN(create_tconv2d_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_conv1d_context"),
      TORCH_FN(create_conv1d_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_linear_context"),
      TORCH_FN(create_linear_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_linear_context_labeled"),
      TORCH_FN(create_linear_context_labeled));
  m.impl(
      TORCH_SELECTIVE_NAME(
          "vulkan_prepack::create_qwen_linear_attention_prefill_context"),
      TORCH_FN(create_qwen_linear_attention_prefill_context));
  m.impl(
      TORCH_SELECTIVE_NAME(
          "vulkan_prepack::run_qwen_linear_attention_prefill_context"),
      TORCH_FN(run_qwen_linear_attention_prefill_context));
  m.impl(
      TORCH_SELECTIVE_NAME(
          "vulkan_prepack::run_qwen_linear_attention_decode_context"),
      TORCH_FN(run_qwen_linear_attention_decode_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_causal_attention_mask"),
      TORCH_FN(create_causal_attention_mask_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::slice_hidden_states_for_logits"),
      TORCH_FN(slice_hidden_states_for_logits_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::index_select_hidden_states_for_logits"),
      TORCH_FN(index_select_hidden_states_for_logits_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::gather_hidden_states_by_batch_positions"),
      TORCH_FN(gather_hidden_states_by_batch_positions_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::find_timestep_index"),
      TORCH_FN(find_timestep_index_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::compute_moe_router"),
      TORCH_FN(compute_moe_router_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::accumulate_expert_outputs"),
      TORCH_FN(accumulate_expert_outputs_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::compute_rotary_cos_sin"),
      TORCH_FN(compute_rotary_cos_sin_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::to_vulkan_labeled"),
      TORCH_FN(to_vulkan_labeled));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::query_runtime_policy"),
      TORCH_FN(query_runtime_policy));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_kv_cache_storage_for_request"),
      TORCH_FN(create_kv_cache_storage_for_request));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_scratch_arena_storage_for_request"),
      TORCH_FN(create_scratch_arena_storage_for_request));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::pose_encoding_to_extri_intri"),
      TORCH_FN(pose_encoding_to_extri_intri_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::extri_intri_to_pose_encoding"),
      TORCH_FN(extri_intri_to_pose_encoding_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_scheduled_gated_delta_rule_chunk"),
      TORCH_FN(run_scheduled_gated_delta_rule_chunk));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_scheduled_gated_delta_rule_recurrent"),
      TORCH_FN(run_scheduled_gated_delta_rule_recurrent));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_layernorm_context"),
      TORCH_FN(create_layernorm_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_gru_context"),
      TORCH_FN(create_gru_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_lstm_context"),
      TORCH_FN(create_lstm_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_batchnorm_context"),
      TORCH_FN(create_batchnorm_context));
}

TORCH_LIBRARY_IMPL(vulkan_prepack, QuantizedCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_qconv2d_context"),
      TORCH_FN(create_qconv2d_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_qtconv2d_context"),
      TORCH_FN(create_qtconv2d_context));
}

TORCH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_linear_context"),
      TORCH_FN(create_linear_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_linear_context_labeled"),
      TORCH_FN(create_linear_context_labeled));
  m.impl(
      TORCH_SELECTIVE_NAME(
          "vulkan_prepack::create_qwen_linear_attention_prefill_context"),
      TORCH_FN(create_qwen_linear_attention_prefill_context));
  m.impl(
      TORCH_SELECTIVE_NAME(
          "vulkan_prepack::run_qwen_linear_attention_prefill_context"),
      TORCH_FN(run_qwen_linear_attention_prefill_context));
  m.impl(
      TORCH_SELECTIVE_NAME(
          "vulkan_prepack::run_qwen_linear_attention_decode_context"),
      TORCH_FN(run_qwen_linear_attention_decode_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_causal_attention_mask"),
      TORCH_FN(create_causal_attention_mask_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::slice_hidden_states_for_logits"),
      TORCH_FN(slice_hidden_states_for_logits_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::index_select_hidden_states_for_logits"),
      TORCH_FN(index_select_hidden_states_for_logits_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::gather_hidden_states_by_batch_positions"),
      TORCH_FN(gather_hidden_states_by_batch_positions_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::find_timestep_index"),
      TORCH_FN(find_timestep_index_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::compute_moe_router"),
      TORCH_FN(compute_moe_router_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::accumulate_expert_outputs"),
      TORCH_FN(accumulate_expert_outputs_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::compute_rotary_cos_sin"),
      TORCH_FN(compute_rotary_cos_sin_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::to_vulkan_labeled"),
      TORCH_FN(to_vulkan_labeled));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::query_runtime_policy"),
      TORCH_FN(query_runtime_policy));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_kv_cache_storage_for_request"),
      TORCH_FN(create_kv_cache_storage_for_request));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::create_scratch_arena_storage_for_request"),
      TORCH_FN(create_scratch_arena_storage_for_request));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::pose_encoding_to_extri_intri"),
      TORCH_FN(pose_encoding_to_extri_intri_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::extri_intri_to_pose_encoding"),
      TORCH_FN(extri_intri_to_pose_encoding_runtime));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_scheduled_gated_delta_rule_chunk"),
      TORCH_FN(run_scheduled_gated_delta_rule_chunk));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_scheduled_gated_delta_rule_recurrent"),
      TORCH_FN(run_scheduled_gated_delta_rule_recurrent));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_conv2d_context"),
      TORCH_FN(run_conv2d_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::conv2d_clamp_run"),
      TORCH_FN(conv2d_clamp_run)); // Backwards compatibility
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_tconv2d_context"),
      TORCH_FN(run_tconv2d_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_qconv2d_context"),
      TORCH_FN(run_qconv2d_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_conv1d_context"),
      TORCH_FN(run_conv1d_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_linear_context"),
      TORCH_FN(run_linear_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_linear_gelu_context"),
      TORCH_FN(run_linear_gelu_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_layernorm_context"),
      TORCH_FN(run_layernorm_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_qlinear_context"),
      TORCH_FN(run_qlinear_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_gru_context"),
      TORCH_FN(run_gru_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_lstm_context"),
      TORCH_FN(run_lstm_context));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_prepack::run_batchnorm_context"),
      TORCH_FN(run_batchnorm_context));
}

TORCH_LIBRARY(vulkan_quantized, m) {
  m.def(
      TORCH_SELECTIVE_SCHEMA("vulkan_quantized::add(Tensor qa, "
                             "Tensor qb, "
                             "float scale, "
                             "int zero_point) -> Tensor qc"));
  m.def(
      TORCH_SELECTIVE_SCHEMA("vulkan_quantized::sub(Tensor qa, "
                             "Tensor qb, "
                             "float scale, "
                             "int zero_point)-> Tensor qc"));
  m.def(
      TORCH_SELECTIVE_SCHEMA("vulkan_quantized::mul(Tensor qa, "
                             "Tensor qb, "
                             "float scale, "
                             "int zero_point)-> Tensor qc"));
  m.def(
      TORCH_SELECTIVE_SCHEMA("vulkan_quantized::div(Tensor qa, "
                             "Tensor qb, "
                             "float scale, "
                             "int zero_point)-> Tensor qc"));
}

TORCH_LIBRARY_IMPL(vulkan_quantized, Vulkan, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_quantized::add"), TORCH_FN(quantized_add));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_quantized::sub"), TORCH_FN(quantized_sub));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_quantized::mul"), TORCH_FN(quantized_mul));
  m.impl(
      TORCH_SELECTIVE_NAME("vulkan_quantized::div"), TORCH_FN(quantized_div));
}

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
