#include <ATen/native/vulkan/ops/GatedDelta.h>

#include <ATen/Functions.h>
#include <ATen/TensorIndexing.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/TensorProvenance.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/tril.h>
#include <ATen/ops/triu.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <ATen/native/vulkan/planning/ExecutionObjects.h>
#include <ATen/native/vulkan/planning/ExecutionPrograms.h>
#include <ATen/native/vulkan/planning/Request.h>
#include <ATen/native/vulkan/planning/Runtime.h>

#include <algorithm>
#include <cmath>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace at::indexing;
using namespace api::utils;

constexpr int64_t kMaxNativeGatedDeltaHeadDim = 128;
constexpr float kGatedDeltaL2NormEps = 1.0e-6f;

Tensor maybe_l2norm(
    const Tensor& x,
    const int64_t dim,
    const double eps,
    const bool enabled) {
  if (!enabled) {
    return x;
  }
  const Tensor inv_norm =
      at::rsqrt(at::add(at::sum(at::mul(x, x), dim, true), eps));
  return at::mul(x, inv_norm);
}

Tensor to_cpu_float(const Tensor& tensor) {
  return tensor.to(kCPU, kFloat);
}

std::optional<Tensor> move_optional_output_to_device(
    const std::optional<Tensor>& tensor,
    const Device& device) {
  if (!tensor.has_value()) {
    return std::nullopt;
  }
  return tensor->to(device, kFloat);
}

std::optional<Tensor> normalize_initial_state(
    const std::optional<Tensor>& initial_state) {
  if (!initial_state.has_value() || !initial_state->defined()) {
    return std::nullopt;
  }
  return to_cpu_float(*initial_state);
}

size_t tensor_nbytes(const Tensor& tensor) {
  if (!tensor.defined()) {
    return 0u;
  }
  return static_cast<size_t>(std::max<int64_t>(0, tensor.numel())) *
      tensor.element_size();
}

size_t gated_delta_scratch_bytes(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& initial_state) {
  size_t bytes = tensor_nbytes(query) + tensor_nbytes(key) + tensor_nbytes(value);
  if (initial_state.has_value() && initial_state->defined()) {
    bytes += tensor_nbytes(*initial_state);
  }
  return std::max<size_t>(64u * 1024u, bytes * 4u);
}

utils::VulkanRuntimePolicy prime_gated_delta_runtime(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const utils::VulkanExecutionPhase execution_phase,
    const std::optional<Tensor>& initial_state,
    const char* scratch_label_suffix) {
  const auto input_request =
      utils::make_vulkan_llm_runtime_request(
          execution_phase, utils::VulkanTensorRole::Input);
  const auto runtime_policy = utils::build_vulkan_runtime_policy(input_request);
  if (!query.is_vulkan()) {
    return runtime_policy;
  }

  utils::log_vulkan_op_hit(
      execution_phase == utils::VulkanExecutionPhase::Prefill
          ? "vulkan_prepack::run_scheduled_gated_delta_rule_chunk"
          : "vulkan_prepack::run_scheduled_gated_delta_rule_recurrent");
  if (
      runtime_policy.execution_program_plan.has_value() &&
      runtime_policy.execution_program_plan->kind ==
          utils::VulkanExecutionProgramKind::GatedDeltaSplit &&
      runtime_policy.boundary_plan.has_value()) {
    const std::optional<utils::VulkanScratchArenaSpec> scratch_spec =
        runtime_policy.scratch_arena_plan.has_value()
        ? std::optional<utils::VulkanScratchArenaSpec>(utils::VulkanScratchArenaSpec{
              kByte,
              std::max<size_t>(
                  gated_delta_scratch_bytes(query, key, value, initial_state),
                  runtime_policy.scratch_arena_plan->min_arena_bytes),
              runtime_policy.scratch_arena_plan->alignment,
              runtime_policy.scratch_arena_plan->prefer_buffer_storage
                  ? api::ExecutionLayout::BUFFER_DIRECT
                  : api::ExecutionLayout::TEXTURE,
              runtime_policy.scratch_arena_plan->prefer_buffer_storage
                  ? api::GPUMemoryLayout::TENSOR_WIDTH_PACKED
                  : api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
              runtime_policy.scratch_arena_plan->prefer_buffer_storage
                  ? api::StorageType::BUFFER
                  : api::StorageType::TEXTURE_3D,
              runtime_policy.scratch_arena_plan->prefer_reusable_arena,
          })
        : std::nullopt;
    (void)utils::lookup_or_create_labeled_gated_delta_split_program(
        api::current_allocation_label().empty()
            ? std::string(scratch_label_suffix)
            : api::current_allocation_label() + "." + scratch_label_suffix,
        *runtime_policy.boundary_plan,
        scratch_spec,
        *runtime_policy.execution_program_plan);
  }
  return runtime_policy;
}

bool use_scheduler_owned_gated_delta_boundary(
    const utils::VulkanRuntimePolicy& runtime_policy) {
  return runtime_policy.boundary_plan.has_value() &&
      runtime_policy.boundary_plan->kind ==
      utils::VulkanBoundaryKind::LLMLinearAttentionSplit;
}

bool use_single_chunk_recurrent_shortcut(
    const Tensor& query,
    const int64_t chunk_size) {
  return query.dim() >= 2 && query.size(1) <= chunk_size;
}

bool shapes_match(IntArrayRef lhs, IntArrayRef rhs) {
  return lhs.equals(rhs);
}

bool supports_native_recurrent_gated_delta(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const std::optional<Tensor>& initial_state) {
  if (!query.is_vulkan() || !key.is_vulkan() || !value.is_vulkan() ||
      !g.is_vulkan() || !beta.is_vulkan()) {
    return false;
  }

  if (
      query.scalar_type() != kFloat || key.scalar_type() != kFloat ||
      value.scalar_type() != kFloat || g.scalar_type() != kFloat ||
      beta.scalar_type() != kFloat) {
    return false;
  }

  if (
      query.dim() != 4 || key.dim() != 4 || value.dim() != 4 || g.dim() != 3 ||
      beta.dim() != 3) {
    return false;
  }

  if (!shapes_match(query.sizes(), key.sizes())) {
    return false;
  }

  if (
      query.size(0) != value.size(0) || query.size(1) != value.size(1) ||
      query.size(2) != value.size(2) || query.size(3) <= 0 ||
      value.size(3) <= 0) {
    return false;
  }

  if (
      g.size(0) != query.size(0) || g.size(1) != query.size(1) ||
      g.size(2) != query.size(2) || !shapes_match(g.sizes(), beta.sizes())) {
    return false;
  }

  if (query.size(3) > kMaxNativeGatedDeltaHeadDim) {
    return false;
  }

  if (initial_state.has_value() && initial_state->defined()) {
    if (!initial_state->is_vulkan() || initial_state->scalar_type() != kFloat ||
        initial_state->dim() != 4) {
      return false;
    }
    if (
        initial_state->size(0) != query.size(0) ||
        initial_state->size(1) != query.size(2) ||
        initial_state->size(2) != query.size(3) ||
        initial_state->size(3) != value.size(3)) {
      return false;
    }
  }

  return true;
}

Tensor make_vulkan_float_dummy_buffer(const Tensor& prototype) {
  return at::zeros({1, 1, 1, 1}, prototype.options().dtype(kFloat));
}

Tensor prepare_gated_delta_buffer_tensor(
    const Tensor& tensor,
    const utils::VulkanExecutionPhase execution_phase,
    const utils::VulkanTensorRole tensor_role) {
  return utils::prepare_vulkan_direct_buffer_execution_tensor(
      tensor,
      utils::VulkanExecutionPlanKind::ElementwiseBufferInput,
      utils::make_vulkan_llm_runtime_request(execution_phase, tensor_role));
}

std::tuple<Tensor, std::optional<Tensor>> run_gated_delta_rule_recurrent_native(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const std::optional<Tensor>& initial_state,
    const bool output_final_state,
    const bool use_qk_l2norm_in_kernel) {
  api::Context* const context = api::context();

  Tensor query_runtime =
      utils::contiguous_inference(query.transpose(1, 2));
  Tensor key_runtime =
      utils::contiguous_inference(key.transpose(1, 2));
  Tensor value_runtime =
      utils::contiguous_inference(value.transpose(1, 2));
  Tensor g_runtime = utils::contiguous_inference(
      g.transpose(1, 2).unsqueeze(-1));
  Tensor beta_runtime = utils::contiguous_inference(
      beta.transpose(1, 2).unsqueeze(-1));

  Tensor query_buffer = prepare_gated_delta_buffer_tensor(
      query_runtime,
      utils::VulkanExecutionPhase::Decode,
      utils::VulkanTensorRole::Input);
  Tensor key_buffer = prepare_gated_delta_buffer_tensor(
      key_runtime,
      utils::VulkanExecutionPhase::Decode,
      utils::VulkanTensorRole::Input);
  Tensor value_buffer = prepare_gated_delta_buffer_tensor(
      value_runtime,
      utils::VulkanExecutionPhase::Decode,
      utils::VulkanTensorRole::Input);
  Tensor g_buffer = prepare_gated_delta_buffer_tensor(
      g_runtime,
      utils::VulkanExecutionPhase::Decode,
      utils::VulkanTensorRole::Input);
  Tensor beta_buffer = prepare_gated_delta_buffer_tensor(
      beta_runtime,
      utils::VulkanExecutionPhase::Decode,
      utils::VulkanTensorRole::Input);

  Tensor initial_state_buffer = initial_state.has_value() && initial_state->defined()
      ? prepare_gated_delta_buffer_tensor(
            *initial_state,
            utils::VulkanExecutionPhase::Decode,
            utils::VulkanTensorRole::Cache)
      : prepare_gated_delta_buffer_tensor(
            make_vulkan_float_dummy_buffer(query),
            utils::VulkanExecutionPhase::Decode,
            utils::VulkanTensorRole::Cache);

  const std::vector<int64_t> output_sizes{
      query.size(0),
      query.size(2),
      query.size(1),
      value.size(3),
  };
  Tensor output_transposed = convert(vTensor{
      context,
      output_sizes,
      api::kFloat,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  });

  std::optional<Tensor> output_state = std::nullopt;
  Tensor output_state_tensor = output_final_state
      ? convert(vTensor{
            context,
            {query.size(0), query.size(2), query.size(3), value.size(3)},
            api::kFloat,
            api::StorageType::BUFFER,
            api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
        })
      : prepare_gated_delta_buffer_tensor(
            make_vulkan_float_dummy_buffer(query),
            utils::VulkanExecutionPhase::Decode,
            utils::VulkanTensorRole::Cache);
  if (output_final_state) {
    output_state = output_state_tensor;
  }

  const struct Block final {
    ivec4 sizes0;
    ivec4 sizes1;
    vec4 params;
  } block{
      {
          safe_downcast<int32_t>(query.size(0)),
          safe_downcast<int32_t>(query.size(2)),
          safe_downcast<int32_t>(query.size(1)),
          safe_downcast<int32_t>(query.size(3)),
      },
      {
          safe_downcast<int32_t>(value.size(3)),
          initial_state.has_value() && initial_state->defined() ? 1 : 0,
          output_final_state ? 1 : 0,
          use_qk_l2norm_in_kernel ? 1 : 0,
      },
      {
          static_cast<float>(
              1.0 / std::sqrt(static_cast<double>(query.size(3)))),
          kGatedDeltaL2NormEps,
          0.0f,
          0.0f,
      },
  };

  const uvec3 global_size = {
      safe_downcast<uint32_t>(div_up<int64_t>(value.size(3), 4)),
      safe_downcast<uint32_t>(query.size(2)),
      safe_downcast<uint32_t>(query.size(0)),
  };

  api::PipelineBarrier pipeline_barrier{};
  const vTensor& v_query = convert(query_buffer);
  const vTensor& v_key = convert(key_buffer);
  const vTensor& v_value = convert(value_buffer);
  const vTensor& v_g = convert(g_buffer);
  const vTensor& v_beta = convert(beta_buffer);
  const vTensor& v_initial_state = convert(initial_state_buffer);
  vTensor& v_output = convert(output_transposed);
  vTensor& v_output_state = convert(output_state_tensor);

  api::UniformParamsBuffer out_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output);
  api::UniformParamsBuffer out_state_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_output_state);
  api::UniformParamsBuffer query_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_query);
  api::UniformParamsBuffer key_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_key);
  api::UniformParamsBuffer value_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_value);
  api::UniformParamsBuffer g_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_g);
  api::UniformParamsBuffer beta_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_beta);
  api::UniformParamsBuffer initial_state_meta =
      utils::make_buffer_compute_metadata_ubo(context, v_initial_state);
  api::UniformParamsBuffer params(context, block);
  context->submit_compute_job(
      VK_KERNEL(gated_delta_recurrent_buffer),
      pipeline_barrier,
      global_size,
      {1u, 1u, 1u},
      VK_NULL_HANDLE,
      v_output.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_meta.buffer(),
      v_output_state.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      out_state_meta.buffer(),
      v_query.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      query_meta.buffer(),
      v_key.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      key_meta.buffer(),
      v_value.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      value_meta.buffer(),
      v_g.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      g_meta.buffer(),
      v_beta.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      beta_meta.buffer(),
      v_initial_state.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      initial_state_meta.buffer(),
      params.buffer());

  utils::log_vulkan_op_hit(
      "vulkan_prepack::run_scheduled_gated_delta_rule_recurrent.native_buffer");
  Tensor output = utils::contiguous_inference(output_transposed.transpose(1, 2));
  record_tensor_write(
      output,
      "vulkan_prepack::run_scheduled_gated_delta_rule_recurrent",
      "native_buffer",
      {query, key, value, g, beta});
  if (output_state.has_value()) {
    record_tensor_write(
        *output_state,
        "vulkan_prepack::run_scheduled_gated_delta_rule_recurrent",
        "native_buffer_final_state",
        {query, key, value, g, beta});
  }
  return {
      output,
      output_state};
}

} // namespace

std::tuple<Tensor, std::optional<Tensor>>
run_gated_delta_rule_recurrent_fallback(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const std::optional<Tensor>& initial_state,
    const bool output_final_state,
    const bool use_qk_l2norm_in_kernel);

std::tuple<Tensor, std::optional<Tensor>> run_gated_delta_rule_chunk_fallback(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const int64_t chunk_size,
    const std::optional<Tensor>& initial_state,
    const bool output_final_state,
    const bool use_qk_l2norm_in_kernel) {
  TORCH_CHECK(chunk_size > 0, "chunk_size must be > 0");

  if (use_single_chunk_recurrent_shortcut(query, chunk_size)) {
    utils::log_vulkan_op_hit(
        "vulkan_prepack::run_scheduled_gated_delta_rule_chunk.single_chunk_recurrent_shortcut");
    return run_gated_delta_rule_recurrent_fallback(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel);
  }

  const Device output_device = query.device();
  const ScalarType output_dtype = query.scalar_type();

  Tensor query_cpu = maybe_l2norm(to_cpu_float(query), -1, 1e-6, use_qk_l2norm_in_kernel);
  Tensor key_cpu = maybe_l2norm(to_cpu_float(key), -1, 1e-6, use_qk_l2norm_in_kernel);
  Tensor value_cpu = to_cpu_float(value);
  Tensor beta_cpu = to_cpu_float(beta);
  Tensor g_cpu = to_cpu_float(g);
  const auto initial_state_cpu = normalize_initial_state(initial_state);

  query_cpu = query_cpu.transpose(1, 2).contiguous();
  key_cpu = key_cpu.transpose(1, 2).contiguous();
  value_cpu = value_cpu.transpose(1, 2).contiguous();
  beta_cpu = beta_cpu.transpose(1, 2).contiguous();
  g_cpu = g_cpu.transpose(1, 2).contiguous();

  const auto batch_size = key_cpu.size(0);
  const auto num_heads = key_cpu.size(1);
  const auto sequence_length = key_cpu.size(2);
  const auto k_head_dim = key_cpu.size(3);
  const auto v_head_dim = value_cpu.size(3);

  const int64_t pad_size =
      (chunk_size - sequence_length % chunk_size) % chunk_size;
  if (pad_size > 0) {
    query_cpu = at::constant_pad_nd(query_cpu, {0, 0, 0, pad_size}, 0.0);
    key_cpu = at::constant_pad_nd(key_cpu, {0, 0, 0, pad_size}, 0.0);
    value_cpu = at::constant_pad_nd(value_cpu, {0, 0, 0, pad_size}, 0.0);
    beta_cpu = at::constant_pad_nd(beta_cpu, {0, pad_size}, 0.0);
    g_cpu = at::constant_pad_nd(g_cpu, {0, pad_size}, 0.0);
  }
  const int64_t total_sequence_length = sequence_length + pad_size;

  const double scale =
      1.0 / std::sqrt(static_cast<double>(query_cpu.size(-1)));
  query_cpu = at::mul(query_cpu, scale);

  Tensor v_beta = at::mul(value_cpu, beta_cpu.unsqueeze(-1));
  Tensor k_beta = at::mul(key_cpu, beta_cpu.unsqueeze(-1));

  query_cpu = query_cpu.reshape({batch_size, num_heads, -1, chunk_size, k_head_dim});
  key_cpu = key_cpu.reshape({batch_size, num_heads, -1, chunk_size, k_head_dim});
  value_cpu = value_cpu.reshape({batch_size, num_heads, -1, chunk_size, v_head_dim});
  k_beta = k_beta.reshape({batch_size, num_heads, -1, chunk_size, k_head_dim});
  v_beta = v_beta.reshape({batch_size, num_heads, -1, chunk_size, v_head_dim});
  g_cpu = g_cpu.reshape({batch_size, num_heads, -1, chunk_size});

  const Tensor mask = at::triu(
      at::ones({chunk_size, chunk_size}, query_cpu.options().dtype(kBool)),
      /*diagonal=*/0);

  g_cpu = g_cpu.cumsum(-1);
  Tensor decay_mask = at::tril(
      at::exp(at::sub(g_cpu.unsqueeze(-1), g_cpu.unsqueeze(-2))));
  decay_mask = at::tril(decay_mask).to(query_cpu.scalar_type());

  Tensor attn = at::neg(at::masked_fill(
      at::mul(at::matmul(k_beta, key_cpu.transpose(-1, -2)), decay_mask),
      mask,
      0));
  for (int64_t i = 1; i < chunk_size; ++i) {
    const Tensor row = attn.index({Ellipsis, i, Slice(None, i)}).clone();
    const Tensor sub =
        attn.index({Ellipsis, Slice(None, i), Slice(None, i)}).clone();
    const Tensor updated = at::add(
        row, at::sum(at::mul(row.unsqueeze(-1), sub), -2));
    at::indexing::set_item(attn, {Ellipsis, i, Slice(None, i)}, updated);
  }
  attn = at::add(attn, at::eye(chunk_size, attn.options()));

  value_cpu = at::matmul(attn, v_beta);
  Tensor k_cumdecay =
      at::matmul(attn, at::mul(k_beta, g_cpu.exp().unsqueeze(-1)));
  Tensor last_recurrent_state =
      initial_state_cpu.has_value()
      ? *initial_state_cpu
      : at::zeros(
            {batch_size, num_heads, k_head_dim, v_head_dim},
            value_cpu.options());
  Tensor core_attn_out = at::zeros_like(value_cpu);

  const Tensor upper_mask = at::triu(
      at::ones({chunk_size, chunk_size}, query_cpu.options().dtype(kBool)),
      /*diagonal=*/1);

  for (int64_t i = 0; i < total_sequence_length / chunk_size; ++i) {
    const Tensor q_i = query_cpu.index({Ellipsis, i, Slice(), Slice()});
    const Tensor k_i = key_cpu.index({Ellipsis, i, Slice(), Slice()});
    const Tensor v_i = value_cpu.index({Ellipsis, i, Slice(), Slice()});
    Tensor intra_attn = at::masked_fill(
        at::mul(
            at::matmul(q_i, k_i.transpose(-1, -2)),
            decay_mask.index({Ellipsis, i, Slice(), Slice()})),
        upper_mask,
        0);
    const Tensor v_prime = at::matmul(
        k_cumdecay.index({Ellipsis, i, Slice(), Slice()}),
        last_recurrent_state);
    const Tensor v_new = at::sub(v_i, v_prime);
    const Tensor attn_inter = at::matmul(
        at::mul(q_i, g_cpu.index({Ellipsis, i, Slice()}).exp().unsqueeze(-1)),
        last_recurrent_state);
    at::indexing::set_item(
        core_attn_out,
        {Ellipsis, i, Slice(), Slice()},
        at::add(attn_inter, at::matmul(intra_attn, v_new)));

    const Tensor g_last =
        g_cpu.index({Ellipsis, i, chunk_size - 1}).exp().unsqueeze(-1).unsqueeze(-1);
    const Tensor decay = at::exp(at::sub(
                                   g_cpu.index({Ellipsis, i, chunk_size - 1})
                                       .unsqueeze(-1),
                                   g_cpu.index({Ellipsis, i, Slice()})))
                             .unsqueeze(-1);
    last_recurrent_state = at::add(
        at::mul(last_recurrent_state, g_last),
        at::matmul(at::mul(k_i, decay).transpose(-1, -2), v_new));
  }

  core_attn_out = core_attn_out.reshape({batch_size, num_heads, -1, v_head_dim});
  core_attn_out = core_attn_out.index({Ellipsis, Slice(None, sequence_length), Slice()});
  core_attn_out =
      core_attn_out.transpose(1, 2).contiguous().to(output_device, output_dtype);
  record_tensor_write(
      core_attn_out,
      "vulkan_prepack::run_scheduled_gated_delta_rule_chunk",
      "fallback",
      {query, key, value, g, beta});
  std::optional<Tensor> output_state =
      output_final_state
          ? move_optional_output_to_device(last_recurrent_state, output_device)
          : std::optional<Tensor>{std::nullopt};
  if (output_state.has_value()) {
    record_tensor_write(
        *output_state,
        "vulkan_prepack::run_scheduled_gated_delta_rule_chunk",
        "fallback_final_state",
        {query, key, value, g, beta});
  }

  return {
      core_attn_out,
      output_state,
  };
}

std::tuple<Tensor, std::optional<Tensor>>
run_gated_delta_rule_recurrent_fallback(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const std::optional<Tensor>& initial_state,
    const bool output_final_state,
    const bool use_qk_l2norm_in_kernel) {
  const Device output_device = query.device();
  const ScalarType output_dtype = query.scalar_type();

  Tensor query_cpu = maybe_l2norm(to_cpu_float(query), -1, 1e-6, use_qk_l2norm_in_kernel);
  Tensor key_cpu = maybe_l2norm(to_cpu_float(key), -1, 1e-6, use_qk_l2norm_in_kernel);
  Tensor value_cpu = to_cpu_float(value);
  Tensor beta_cpu = to_cpu_float(beta);
  Tensor g_cpu = to_cpu_float(g);
  const auto initial_state_cpu = normalize_initial_state(initial_state);

  query_cpu = query_cpu.transpose(1, 2).contiguous();
  key_cpu = key_cpu.transpose(1, 2).contiguous();
  value_cpu = value_cpu.transpose(1, 2).contiguous();
  beta_cpu = beta_cpu.transpose(1, 2).contiguous();
  g_cpu = g_cpu.transpose(1, 2).contiguous();

  const auto batch_size = key_cpu.size(0);
  const auto num_heads = key_cpu.size(1);
  const auto sequence_length = key_cpu.size(2);
  const auto k_head_dim = key_cpu.size(3);
  const auto v_head_dim = value_cpu.size(3);

  const double scale =
      1.0 / std::sqrt(static_cast<double>(query_cpu.size(-1)));
  query_cpu = at::mul(query_cpu, scale);

  Tensor core_attn_out = at::zeros(
      {batch_size, num_heads, sequence_length, v_head_dim},
      value_cpu.options());
  Tensor last_recurrent_state =
      initial_state_cpu.has_value()
      ? *initial_state_cpu
      : at::zeros(
            {batch_size, num_heads, k_head_dim, v_head_dim},
            value_cpu.options());

  for (int64_t i = 0; i < sequence_length; ++i) {
    const Tensor q_t = query_cpu.index({Ellipsis, i, Slice()});
    const Tensor k_t = key_cpu.index({Ellipsis, i, Slice()});
    const Tensor v_t = value_cpu.index({Ellipsis, i, Slice()});
    const Tensor g_t =
        g_cpu.index({Ellipsis, i}).exp().unsqueeze(-1).unsqueeze(-1);
    const Tensor beta_t = beta_cpu.index({Ellipsis, i}).unsqueeze(-1);

    last_recurrent_state = at::mul(last_recurrent_state, g_t);
    const Tensor kv_mem =
        at::sum(at::mul(last_recurrent_state, k_t.unsqueeze(-1)), -2);
    const Tensor delta = at::mul(at::sub(v_t, kv_mem), beta_t);
    last_recurrent_state = at::add(
        last_recurrent_state,
        at::mul(k_t.unsqueeze(-1), delta.unsqueeze(-2)));
    at::indexing::set_item(
        core_attn_out,
        {Ellipsis, i, Slice()},
        at::sum(at::mul(last_recurrent_state, q_t.unsqueeze(-1)), -2));
  }

  core_attn_out =
      core_attn_out.transpose(1, 2).contiguous().to(output_device, output_dtype);
  record_tensor_write(
      core_attn_out,
      "vulkan_prepack::run_scheduled_gated_delta_rule_recurrent",
      "fallback",
      {query, key, value, g, beta});
  std::optional<Tensor> output_state =
      output_final_state
          ? move_optional_output_to_device(last_recurrent_state, output_device)
          : std::optional<Tensor>{std::nullopt};
  if (output_state.has_value()) {
    record_tensor_write(
        *output_state,
        "vulkan_prepack::run_scheduled_gated_delta_rule_recurrent",
        "fallback_final_state",
        {query, key, value, g, beta});
  }
  return {
      core_attn_out,
      output_state,
  };
}

std::tuple<Tensor, std::optional<Tensor>> run_scheduled_gated_delta_rule_chunk(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const int64_t chunk_size,
    const std::optional<Tensor>& initial_state,
    const bool output_final_state,
    const bool use_qk_l2norm_in_kernel) {
  const auto runtime_policy = prime_gated_delta_runtime(
      query,
      key,
      value,
      utils::VulkanExecutionPhase::Prefill,
      initial_state,
      "gated_delta_chunk");
  if (supports_native_recurrent_gated_delta(
          query, key, value, g, beta, initial_state)) {
    utils::log_vulkan_op_hit(
        use_single_chunk_recurrent_shortcut(query, chunk_size)
            ? "vulkan_prepack::run_scheduled_gated_delta_rule_chunk.native_single_chunk_recurrent"
            : "vulkan_prepack::run_scheduled_gated_delta_rule_chunk.native_full_sequence_recurrent");
    return run_gated_delta_rule_recurrent_native(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel);
  }
  if (use_scheduler_owned_gated_delta_boundary(runtime_policy)) {
    return run_gated_delta_rule_chunk_fallback(
        query,
        key,
        value,
        g,
        beta,
        chunk_size,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel);
  }
  return run_gated_delta_rule_chunk_fallback(
      query,
      key,
      value,
      g,
      beta,
      chunk_size,
      initial_state,
      output_final_state,
      use_qk_l2norm_in_kernel);
}

std::tuple<Tensor, std::optional<Tensor>>
run_scheduled_gated_delta_rule_recurrent(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    const std::optional<Tensor>& initial_state,
    const bool output_final_state,
    const bool use_qk_l2norm_in_kernel) {
  const auto runtime_policy = prime_gated_delta_runtime(
      query,
      key,
      value,
      utils::VulkanExecutionPhase::Decode,
      initial_state,
      "gated_delta_recurrent");
  if (supports_native_recurrent_gated_delta(
          query, key, value, g, beta, initial_state)) {
    return run_gated_delta_rule_recurrent_native(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel);
  }
  if (use_scheduler_owned_gated_delta_boundary(runtime_policy)) {
    return run_gated_delta_rule_recurrent_fallback(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel);
  }
  return run_gated_delta_rule_recurrent_fallback(
      query,
      key,
      value,
      g,
      beta,
      initial_state,
      output_final_state,
      use_qk_l2norm_in_kernel);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
