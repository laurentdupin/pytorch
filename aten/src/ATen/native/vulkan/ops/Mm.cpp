#include <ATen/native/vulkan/ops/Mm.h>
#include <ATen/native/vulkan/ops/Utils.h>

#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <ATen/native/vulkan/api/Tensor.h>
#include <ATen/native/vulkan/api/Types.h>
#include <ATen/native/vulkan/impl/Packing.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/irange.h>

#include <atomic>
#include <deque>
#include <fstream>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;
using namespace at::native::vulkan::ops;

constexpr size_t kLinearContextCacheSize = 128u;
constexpr float kGeluBeta =
    static_cast<float>(M_SQRT2 * M_2_SQRTPI * 0.5);

enum class LinearPostOp : uint8_t {
  None,
  Gelu,
};

const std::string& linear_cache_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_LINEAR_CACHE_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool linear_cache_logging_enabled() {
  return !linear_cache_log_path().empty();
}

struct LinearCacheLogState final {
  std::atomic<uint64_t> lookups{0u};
  std::atomic<uint64_t> hits{0u};
  std::atomic<uint64_t> stores{0u};

  ~LinearCacheLogState() {
    if (!linear_cache_logging_enabled()) {
      return;
    }

    std::ofstream out(linear_cache_log_path(), std::ios::app);
    out << "linear_cache: lookups=" << lookups.load(std::memory_order_relaxed)
        << " hits=" << hits.load(std::memory_order_relaxed)
        << " stores=" << stores.load(std::memory_order_relaxed) << '\n';
  }
};

LinearCacheLogState& linear_cache_log_state() {
  static LinearCacheLogState state;
  return state;
}

struct LinearContextCacheEntry final {
  Tensor weight_ref;
  std::optional<Tensor> bias_ref;
  int64_t weight_version;
  int64_t bias_version;
  c10::intrusive_ptr<LinearPackedContext> context;
};

thread_local std::deque<LinearContextCacheEntry> linear_context_cache;

std::optional<Tensor> normalized_optional_tensor(
    const std::optional<Tensor>& tensor) {
  if (tensor && tensor->defined()) {
    return tensor;
  }
  return std::nullopt;
}

bool same_optional_tensor_impl(
    const std::optional<Tensor>& lhs,
    const std::optional<Tensor>& rhs) {
  if (lhs.has_value() != rhs.has_value()) {
    return false;
  }
  if (!lhs.has_value()) {
    return true;
  }
  return lhs->unsafeGetTensorImpl() == rhs->unsafeGetTensorImpl();
}

int64_t tensor_version_or_zero(const Tensor& tensor) {
  return tensor.is_inference() ? 0 : tensor._version();
}

bool has_inference_tensor(
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  const auto normalized_bias = normalized_optional_tensor(bias);
  return weight.is_inference() ||
      (normalized_bias && normalized_bias->is_inference());
}

std::optional<c10::intrusive_ptr<LinearPackedContext>> lookup_linear_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  if (!weight.is_vulkan() || weight.dim() != 2) {
    return std::nullopt;
  }

  const auto normalized_bias = normalized_optional_tensor(bias);
  if (linear_cache_logging_enabled()) {
    linear_cache_log_state().lookups.fetch_add(1u, std::memory_order_relaxed);
  }

  const int64_t weight_version = tensor_version_or_zero(weight);
  const int64_t bias_version =
      normalized_bias ? tensor_version_or_zero(*normalized_bias) : 0u;

  for (auto it = linear_context_cache.begin(); it != linear_context_cache.end();
       ++it) {
    if (it->weight_ref.unsafeGetTensorImpl() != weight.unsafeGetTensorImpl() ||
        it->weight_version != weight_version ||
        !same_optional_tensor_impl(it->bias_ref, normalized_bias) ||
        it->bias_version != bias_version) {
      continue;
    }

    auto context = it->context;
    if (it != linear_context_cache.begin()) {
      LinearContextCacheEntry entry = std::move(*it);
      linear_context_cache.erase(it);
      linear_context_cache.emplace_front(std::move(entry));
      context = linear_context_cache.front().context;
    }
    if (linear_cache_logging_enabled()) {
      linear_cache_log_state().hits.fetch_add(1u, std::memory_order_relaxed);
    }
    return context;
  }

  return std::nullopt;
}

void store_linear_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const c10::intrusive_ptr<LinearPackedContext>& context) {
  if (!weight.is_vulkan() || weight.dim() != 2) {
    return;
  }

  const auto normalized_bias = normalized_optional_tensor(bias);
  if (linear_cache_logging_enabled()) {
    linear_cache_log_state().stores.fetch_add(1u, std::memory_order_relaxed);
  }

  LinearContextCacheEntry entry;
  entry.weight_ref = weight;
  entry.bias_ref = normalized_bias;
  entry.weight_version = tensor_version_or_zero(weight);
  entry.bias_version =
      normalized_bias ? tensor_version_or_zero(*normalized_bias) : 0u;
  entry.context = context;

  linear_context_cache.emplace_front(std::move(entry));
  if (linear_context_cache.size() > kLinearContextCacheSize) {
    linear_context_cache.pop_back();
  }
}

c10::intrusive_ptr<LinearPackedContext> get_or_create_linear_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  if (has_inference_tensor(weight, bias)) {
    const Tensor prepared_weight =
        (weight.is_vulkan() && weight.dim() == 2) ? weight.cpu().t().contiguous()
                                                  : weight.t();
    return c10::make_intrusive<LinearPackedContext>(
        LinearPackedContext(
            prepared_weight,
            bias,
            false,
            std::string(),
            false));
  }

  if (const auto cached_context = lookup_linear_context(weight, bias)) {
    return *cached_context;
  }

  const Tensor prepared_weight =
      (c10::InferenceMode::is_enabled() && weight.is_vulkan() &&
       weight.dim() == 2)
      ? weight.cpu().t().contiguous()
      : weight.t();
  const auto context = c10::make_intrusive<LinearPackedContext>(
      LinearPackedContext(
          prepared_weight,
          bias,
          false,
          std::string(),
          false));
  store_linear_context(weight, bias, context);
  return context;
}

const std::string& linear_runtime_label(
    const c10::intrusive_ptr<LinearPackedContext>& linear_context,
    const char* fallback) {
  if (
      linear_context &&
      !linear_context->allocation_label().empty()) {
    return linear_context->allocation_label();
  }
  static const std::string kLinearLabel = "linear";
  static const std::string kBmmLabel = "bmm";
  return std::string(fallback) == "bmm" ? kBmmLabel : kLinearLabel;
}

std::string linear_pack_label(
    const std::string& allocation_label,
    const bool use_batch) {
  if (allocation_label.empty()) {
    return use_batch ? "bmm.pack" : "linear.pack";
  }
  return allocation_label + ".pack";
}

inline bool has_bias(const std::optional<Tensor>& bias) {
  return bias && bias->defined();
}

bool can_fuse_linear_bias(
    const vTensor& v_output,
    const vTensor& v_bias,
    const std::vector<int64_t>& weight_sizes) {
  if (
      v_bias.storage_type() != api::StorageType::TEXTURE_3D ||
      v_bias.gpu_memory_layout() !=
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED) {
    return false;
  }

  const IntArrayRef bias_sizes = v_bias.sizes();
  if (bias_sizes.empty() || bias_sizes.size() > 2) {
    return false;
  }

  const int64_t output_width = weight_sizes[Layout::Parameter::width];
  const int64_t output_height = v_output.sizes()[Layout::Parameter::height];
  const int64_t bias_width = bias_sizes.back();
  const int64_t bias_height =
      bias_sizes.size() == 2 ? bias_sizes.front() : 1;

  return bias_width == output_width &&
      (bias_height == 1 || bias_height == output_height);
}

Tensor materialize_inference_vulkan_matrix_arg(const Tensor& tensor) {
  if (
      c10::InferenceMode::is_enabled() &&
      tensor.is_vulkan() &&
      tensor.dim() == 2 &&
      !tensor.is_contiguous_or_false()) {
    // A direct transpose view of a Vulkan parameter can later trip
    // version-counter propagation in the mm/addmm path. Re-materialize it via
    // the original layout first, then transpose back.
    return tensor.t().clone().t();
  }
  return tensor;
}

Tensor upcast_bfloat16_weight_for_vulkan_linear(const Tensor& tensor) {
  if (tensor.scalar_type() != kBFloat16) {
    return tensor;
  }
  if (tensor.is_vulkan()) {
    return convert(tensor).storage_type() == api::StorageType::BUFFER
        ? utils::upcast_bfloat16_buffer_to_float(tensor)
        : tensor.cpu().to(kFloat).vulkan();
  }
  return tensor.to(kFloat);
}

std::optional<Tensor> upcast_bfloat16_bias_for_vulkan_linear(
    const std::optional<Tensor>& tensor) {
  if (!tensor || !tensor->defined()) {
    return std::nullopt;
  }
  if (tensor->scalar_type() != kBFloat16) {
    return tensor;
  }
  if (tensor->is_vulkan()) {
    return convert(*tensor).storage_type() == api::StorageType::BUFFER
        ? utils::upcast_bfloat16_buffer_to_float(*tensor)
        : tensor->cpu().to(kFloat).vulkan();
  }
  return tensor->to(kFloat);
}

Tensor upcast_bfloat16_input_for_vulkan_linear(const Tensor& tensor) {
  if (tensor.scalar_type() != kBFloat16) {
    return tensor;
  }
  if (tensor.is_vulkan()) {
    return convert(tensor).storage_type() == api::StorageType::BUFFER
        ? utils::upcast_bfloat16_buffer_to_float(tensor)
        : tensor.cpu().to(kFloat).vulkan();
  }
  return tensor.to(kFloat);
}

vTensor pack_cpu_float_weight_using_height_packing(const Tensor& weight_arg) {
  TORCH_INTERNAL_ASSERT(weight_arg.is_cpu());
  TORCH_INTERNAL_ASSERT(weight_arg.scalar_type() == kFloat);
  TORCH_INTERNAL_ASSERT(weight_arg.dim() == 2);

  api::Context* const context = api::context();
  const Tensor weight = weight_arg.contiguous();
  const int64_t height = weight.size(Layout::Parameter::height);
  const int64_t width = weight.size(Layout::Parameter::width);

  vTensor v_weight{
      context,
      weight.sizes().vec(),
      convert_dtype(weight.scalar_type()),
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
  };

  api::StorageBuffer staging(context, api::kFloat, v_weight.gpu_numel());
  {
    api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
    float* const dst = mapping.template data<float>();
    const float* const src = weight.const_data_ptr<float>();
    std::fill_n(dst, v_weight.gpu_numel(), 0.0f);

    const api::utils::uvec3 extents = v_weight.extents();
    const int64_t texel_width =
        static_cast<int64_t>(extents.data[0u]);
    const int64_t texel_height =
        static_cast<int64_t>(extents.data[1u]);
    const int64_t texel_depth =
        static_cast<int64_t>(extents.data[2u]);

    for (const auto z : c10::irange(texel_depth)) {
      for (const auto y : c10::irange(texel_height)) {
        const int64_t src_base_h = y * 4;
        for (const auto x : c10::irange(texel_width)) {
          const int64_t texel_base =
              (((z * texel_height) + y) * texel_width + x) * 4;
          for (const auto c : c10::irange(int64_t{4})) {
            const int64_t src_h = src_base_h + c;
            if (src_h < height && x < width) {
              dst[texel_base + c] = src[src_h * width + x];
            }
          }
        }
      }
    }
  }

  api::PipelineBarrier pipeline_barrier{};
  utils::copy_buffer_to_vtensor(staging.buffer(), v_weight, pipeline_barrier);
  return v_weight;
}

vTensor pack_inputs_using_width_packing(const Tensor& input_arg) {
  TORCH_INTERNAL_ASSERT(
      !input_arg.is_quantized(),
      "Vulkan Linear not usable! "
      "Reason: Input packing only supports non-quantized tensors.");
  TORCH_INTERNAL_ASSERT(
      input_arg.dim() == 2 || input_arg.dim() == 3,
      "Vulkan Linear not usable! "
      "Reason: Input packing only supports 2D or 3D tensors.");

  Tensor input = input_arg;
  if (input.is_cpu()) {
    input = input.vulkan();
  }

  TORCH_CHECK(input.is_vulkan(), "Input must be on Vulkan device!");

  if (convert(input).storage_type() == api::StorageType::BUFFER) {
    input = utils::ensure_texture_storage(input);
  }

  vTensor v_input = convert(input);
  if (v_input.gpu_memory_layout() ==
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED) {
    v_input = packing::convert_image_channels_packed_to_width_packed(v_input);
  }

  TORCH_CHECK(
      v_input.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      "After packing, the v_input must be in TENSOR_WIDTH_PACKED format");

  return v_input;
}

vTensor pack_weights_using_height_packing(const Tensor& weight_arg) {
  // Only non-batch, non-quantized tensors are supported
  TORCH_INTERNAL_ASSERT(
      !weight_arg.is_quantized(),
      "Vulkan Linear not usable! "
      "Reason: Weight packing only supports non-quantized tensors.");
  TORCH_INTERNAL_ASSERT(
      weight_arg.dim() == 2 || weight_arg.dim() == 3,
      "Vulkan Linear not usable! "
      "Reason: Weight packing only supports 2D or 3D tensors.");

  if (weight_arg.is_cpu() && weight_arg.scalar_type() == kFloat &&
      weight_arg.dim() == 2) {
    return pack_cpu_float_weight_using_height_packing(weight_arg);
  }

  Tensor weight = weight_arg;

  if (weight.is_cpu()) {
    weight = weight.vulkan();
  }

  TORCH_CHECK(weight.is_vulkan(), "Weight must be on Vulkan device!");

  if (convert(weight).storage_type() == api::StorageType::BUFFER) {
    weight = utils::ensure_texture_storage(weight);
  }

  vTensor v_weight = convert(weight);
  if (v_weight.gpu_memory_layout() ==
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED) {
    v_weight =
        packing::convert_image_channels_packed_to_height_packed(v_weight);
  }

  TORCH_CHECK(
      v_weight.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
      "After packing, the v_weight must be in TENSOR_HEIGHT_PACKED format");

  return v_weight;
}

vTensor pack_weights(const Tensor& weight_arg, const bool use_batch = false) {
  if (!weight_arg.is_quantized()) {
    return pack_weights_using_height_packing(weight_arg);
  }

  TORCH_CHECK(
      weight_arg.is_quantized(), "Only quantized weights logic after here");

  // Rest of the logic are either quantized or batched.

  api::Context* const context = api::context();

  const Tensor weight = weight_arg.contiguous();
  const IntArrayRef w_sizes = weight.sizes();
  if (use_batch) {
    TORCH_CHECK(
        w_sizes.size() == 3,
        "Vulkan Linear not usable! "
        "Reason: Unable to perform weight packing with batch; the input tensor of a batch of matrices should contain 3 dimensions: batch, height, width.");
  }
  /* Source */
  int64_t src_kb_sz = 0;
  int64_t src_kw_sz = 0;
  int64_t src_kh_sz = 0;
  /* Destination */
  int64_t dst_kb_sz = 0;
  int64_t dst_kw_sz = 0;
  int64_t dst_kh_sz = 0;
  std::vector<int64_t> dst_vtensor_sizes;
  /* Source */
  src_kb_sz = use_batch ? w_sizes[Layout::BatchMatrices::batch] : 1;
  src_kw_sz = use_batch ? w_sizes[Layout::BatchMatrices::width]
                        : w_sizes[Layout::Parameter::width];
  src_kh_sz = use_batch ? w_sizes[Layout::BatchMatrices::height]
                        : w_sizes[Layout::Parameter::height];

  /* Destination */
  dst_kb_sz = src_kb_sz;
  dst_kw_sz = div_up(src_kw_sz, INT64_C(2));
  dst_kh_sz = div_up(src_kh_sz, INT64_C(2));
  dst_vtensor_sizes = {
      dst_kb_sz,
      4,
      dst_kh_sz,
      dst_kw_sz,
  };

  vTensor v_weight{
      context, dst_vtensor_sizes, convert_dtype(weight_arg.scalar_type())};

  v_weight.set_is_quantized();
  v_weight.set_scale(weight_arg.q_scale());
  v_weight.set_zero_point(weight_arg.q_zero_point());

  stage_pack_weights<int8_t>(
      context,
      v_weight,
      weight,
      src_kb_sz,
      src_kh_sz,
      src_kw_sz,
      dst_kh_sz,
      dst_kw_sz);
  return v_weight;
}

vTensor pack_biases(
    const Tensor& weight_arg,
    const std::optional<Tensor>& bias_arg,
    const bool use_batch = false) {
  if (has_bias(bias_arg)) {
    Tensor bias = *bias_arg;
    if (bias.is_cpu()) {
      bias = bias.vulkan();
    }
    if (bias.is_vulkan() && convert(bias).storage_type() == api::StorageType::BUFFER) {
      bias = utils::ensure_texture_storage(bias);
    }
    return convert(bias);
  } else {
    return convert(at::zeros({1}, at::device(at::kVulkan).dtype(at::kFloat)));
  }
}

// Old version of pack_biases that fixes issues with quantization and to be
// removed in the future.
vTensor pack_biases_quantized_weights(
    const Tensor& weight_arg,
    const std::optional<Tensor>& bias_arg,
    const bool use_batch = false) {
  TORCH_CHECK(
      weight_arg.is_quantized(),
      "pack_biases_quantized to be used only when using quantized linear ops");

  if (has_bias(bias_arg) && bias_arg->is_vulkan()) {
    Tensor bias = *bias_arg;
    if (convert(bias).storage_type() == api::StorageType::BUFFER) {
      bias = utils::ensure_texture_storage(bias);
    }
    return convert(bias);
  }

  api::Context* const context = api::context();

  if (has_bias(bias_arg)) {
    const Tensor bias = bias_arg->contiguous();
    const IntArrayRef b_sizes = bias.sizes();
    const float* const src_bias_ptr = bias.const_data_ptr<float>();

    /* Source */
    int64_t src_kb_sz = 0;
    int64_t src_kw_sz = 0;
    int64_t src_kh_sz = 0;
    if (use_batch) {
      if (bias.sizes().size() == 3) {
        src_kb_sz = b_sizes[Layout::BatchMatrices::batch];
        src_kw_sz = b_sizes[Layout::BatchMatrices::width];
        src_kh_sz = b_sizes[Layout::BatchMatrices::height];
      } else if (bias.sizes().size() == 2) {
        // skip batch dim for broadcasting; index -1
        src_kb_sz = 1;
        src_kw_sz = b_sizes[Layout::BatchMatrices::height];
        src_kh_sz = b_sizes[Layout::BatchMatrices::batch];
      } else {
        // skip batch & height dim for broadcasting; index -2
        src_kb_sz = 1;
        src_kw_sz = b_sizes[Layout::BatchMatrices::batch];
        src_kh_sz = 1;
      }
    } else {
      src_kb_sz = 1;
      if (bias.sizes().size() == 2) {
        src_kw_sz = b_sizes[Layout::Parameter::width];
        src_kh_sz = b_sizes[Layout::Parameter::height];
      } else {
        src_kw_sz = b_sizes[Layout::Parameter::height];
        src_kh_sz = 1;
      }
    }
    const int64_t src_matrix_sz = src_kw_sz * src_kh_sz;

    /* Destination */
    const int64_t dst_kw_sz = div_up(src_kw_sz, INT64_C(2));
    const int64_t dst_kh_sz = div_up(src_kh_sz, INT64_C(2));
    const int64_t dst_plane_sz = dst_kw_sz * dst_kh_sz;
    const int64_t dst_matrix_sz = dst_plane_sz * 4;

    vTensor v_bias{
        context,
        {
            src_kb_sz,
            4,
            dst_kh_sz,
            dst_kw_sz,
        },
        convert_dtype(bias_arg->scalar_type()),
    };

    api::StorageBuffer staging(
        context, api::ScalarType::Float, v_bias.gpu_numel());
    {
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

      float* dst_bias_ptr = mapping.template data<float>();

      memset(dst_bias_ptr, 0, v_bias.nbytes());

      for (const auto src_b : c10::irange(src_kb_sz)) {
        for (const auto src_h : c10::irange(src_kh_sz == 1 ? 2 : src_kh_sz)) {
          for (const auto src_w :
               c10::irange((use_batch && src_kw_sz == 1) ? 2 : src_kw_sz)) {
            int64_t dst_plane = 2 * (src_h % 2) + (src_w % 2);
            int64_t dst_index = (src_h / 2) * dst_kw_sz + (src_w / 2);
            memcpy(
                dst_bias_ptr + src_b * dst_matrix_sz +
                    dst_plane * dst_plane_sz + dst_index,
                src_bias_ptr + src_b * src_matrix_sz +
                    (src_kh_sz == 1 ? 0 : src_h * src_kw_sz) +
                    ((use_batch && src_kw_sz == 1) ? 0 : src_w),
                sizeof(float));
          }
        }
      }
    }
    utils::pack_staging_to_vtensor(staging.buffer(), v_bias);

    return v_bias;
  } else {
    vTensor v_bias{
        api::context(),
        {1},
        convert_dtype(weight_arg.scalar_type()),
    };

    api::StorageBuffer staging(
        context, api::ScalarType::Float, v_bias.gpu_numel());
    {
      api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);

      float* data_ptr = mapping.template data<float>();

      memset(
          data_ptr,
          // 2's complement integers and IEEE-754 floating point numbers both
          // have identical bit representations for 0, so can use memset which
          // only accepts uint8_t parameter.
          0,
          v_bias.nbytes());
    }
    utils::pack_staging_to_vtensor(staging.buffer(), v_bias);

    return v_bias;
  }
}

bool available_check_with_batch(
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  const bool weight_available = (3 == weight.ndimension()) &&
      (weight.size(Layout::BatchMatrices::batch) > 0) &&
      (weight.size(Layout::BatchMatrices::height) > 0) &&
      (weight.size(Layout::BatchMatrices::width) > 0) &&
      ((weight.device().is_cpu()) ||
       (c10::DeviceType::Vulkan == weight.device().type())) &&
      (kFloat == weight.scalar_type());
  if (!weight_available) {
    return false;
  }

  if (!bias || !bias->defined()) {
    // no need to check bias since it is not used.
    return true;
  }

  bool bias_available = true;
  bias_available &= (bias->ndimension() > 0);
  bias_available &=
      ((bias->device().is_cpu()) ||
       (c10::DeviceType::Vulkan == bias->device().type()));
  bias_available &= (kFloat == bias->scalar_type());
  // Only check the consistency of batch and width dimension. The height
  // dimension consistency is unchecked, due to the 2nd input which determines
  // the height is not passed into LinearPackedContext.
  if (bias->ndimension() == 3) {
    bias_available &=
        (bias->size(Layout::BatchMatrices::width) ==
             weight.size(Layout::BatchMatrices::width) ||
         bias->size(Layout::BatchMatrices::width) == 1);
    bias_available &=
        (bias->size(Layout::BatchMatrices::batch) ==
             weight.size(Layout::BatchMatrices::batch) ||
         bias->size(Layout::BatchMatrices::batch) == 1);
  } else if (bias->ndimension() == 2) {
    // skip batch dim for broadcasting; index -1
    bias_available &=
        (bias->size(Layout::BatchMatrices::height) ==
             weight.size(Layout::BatchMatrices::width) ||
         bias->size(Layout::BatchMatrices::height) == 1);
  } else {
    // skip batch & height dim for broadcasting; index -2
    bias_available &=
        (bias->size(Layout::BatchMatrices::batch) ==
             weight.size(Layout::BatchMatrices::width) ||
         bias->size(Layout::BatchMatrices::batch) == 1);
  }
  return bias_available;
}

bool available(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const bool use_batch = false) {
  if (!api::available()) {
    return false;
  }

  if (use_batch) {
    return available_check_with_batch(weight, bias);
  }

  const bool weight_available = (2 == weight.ndimension()) &&
      (weight.size(Layout::Parameter::height) > 0) &&
      (weight.size(Layout::Parameter::width) > 0) &&
      ((weight.device().is_cpu()) ||
       (c10::DeviceType::Vulkan == weight.device().type())) &&
      (kFloat == weight.scalar_type() || kQInt8 == weight.scalar_type());
  if (!weight_available) {
    return false;
  }

  const bool bias_available =
      ((bias && bias.has_value() && bias->defined())
           ? ((bias->ndimension() > 0) &&
              ((bias->device().is_cpu()) ||
               (c10::DeviceType::Vulkan == bias->device().type())) &&
              (kFloat == bias->scalar_type()) &&
              ((bias->ndimension() > 1)
                   ? (bias->size(Layout::Parameter::width) ==
                      weight.size(Layout::Parameter::width))
                   : true))
           : true);
  return bias_available;
}

bool usable_check_with_batch(
    const Tensor& input,
    const IntArrayRef unpacked_weight_sizes) {
  return (3 == input.ndimension()) &&
      (c10::DeviceType::Vulkan == input.device().type()) &&
      (kFloat == input.scalar_type()) &&
      (input.size(Layout::BatchMatrices::width) ==
       unpacked_weight_sizes[Layout::BatchMatrices::height]) &&
      (input.size(Layout::BatchMatrices::batch) ==
       unpacked_weight_sizes[Layout::BatchMatrices::batch]) &&
      !input.requires_grad() && true;
}

bool usable(
    const Tensor& input,
    const IntArrayRef unpacked_weight_sizes,
    const bool use_batch = false) {
  if (use_batch) {
    return usable_check_with_batch(input, unpacked_weight_sizes);
  }
  const auto v_input = convert(input);
  return (2 == input.ndimension()) &&
      (c10::DeviceType::Vulkan == input.device().type()) &&
      ((kFloat == input.scalar_type()) ||
       (v_input.is_quantized() &&
        (kQUInt8 == input.scalar_type() || kQInt8 == input.scalar_type()))) &&
      (input.size(Layout::Parameter::width) ==
       unpacked_weight_sizes[Layout::Parameter::height]) &&
      !input.requires_grad() && true;
}

static Tensor reshape_to_2d(const Tensor& input_arg) {
  TORCH_CHECK(
      input_arg.dim() >= 1,
      "Vulkan Linear op only supports input tensor with dim >= 1");

  Tensor reshape_input = input_arg;
  if (input_arg.is_vulkan() && c10::InferenceMode::is_enabled()) {
    // View-derived Vulkan tensors can carry logical layouts that are still
    // correct numerically but not yet suitable for the downstream linear packer.
    // Materialize once before the flattening reshape so permute/reshape ->
    // linear matches CPU without requiring model-side x = x + 0 workarounds.
    reshape_input = at::add(input_arg, 0.0);
  }

  if (reshape_input.dim() == 1) {
    return reshape_input.unsqueeze(0);
  }
  const IntArrayRef input_sizes = reshape_input.sizes();
  const auto d =
      c10::multiply_integers(input_sizes.cbegin(), input_sizes.end() - 1);
  return reshape_input.reshape({d, reshape_input.size(-1)});
}

static Tensor reshape_to_2d_buffer_linear(const Tensor& input_arg) {
  TORCH_CHECK(
      input_arg.dim() >= 1,
      "Vulkan Linear op only supports input tensor with dim >= 1");

  if (input_arg.dim() == 1) {
    return input_arg.unsqueeze(0);
  }

  const IntArrayRef input_sizes = input_arg.sizes();
  const auto d =
      c10::multiply_integers(input_sizes.cbegin(), input_sizes.end() - 1);
  return input_arg.reshape({d, input_arg.size(-1)});
}

bool can_run_bfloat16_buffer_linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  // Disabled for now. The initial buffer-native BF16 path is not yet reliable
  // enough for general eager execution, so BF16 linear currently widens through
  // the proven float path instead.
  return false;

  if (
      input.device().type() != c10::DeviceType::Vulkan ||
      weight.device().type() != c10::DeviceType::Vulkan ||
      input.scalar_type() != kBFloat16 ||
      weight.scalar_type() != kBFloat16 ||
      input.dim() != 2 ||
      weight.dim() != 2 ||
      input.requires_grad() ||
      weight.requires_grad() ||
      input.size(Layout::Parameter::width) !=
          weight.size(Layout::Parameter::width)) {
    return false;
  }

  vTensor v_input = convert(input);
  vTensor v_weight = convert(weight);
  if (
      v_input.storage_type() != api::StorageType::BUFFER ||
      v_weight.storage_type() != api::StorageType::BUFFER) {
    return false;
  }

  if (bias && bias->defined()) {
    if (
        bias->device().type() != c10::DeviceType::Vulkan ||
        bias->dim() > 2 ||
        bias->requires_grad()) {
      return false;
    }

    if (convert(*bias).storage_type() != api::StorageType::BUFFER) {
      return false;
    }

    if (bias->scalar_type() != kBFloat16 && bias->scalar_type() != kFloat) {
      return false;
    }
  }

  return true;
}

Tensor run_bfloat16_buffer_linear(
    const Tensor& input_arg,
    const Tensor& weight_arg,
    const std::optional<Tensor>& bias_arg) {
  api::AllocationScope allocation_scope("linear.bf16_buffer");
  api::Context* const context = api::context();

  const Tensor input_arg_2d =
      input_arg.dim() == 2 ? input_arg : reshape_to_2d_buffer_linear(input_arg);
  const Tensor input =
      input_arg_2d.is_vulkan() ? input_arg_2d : input_arg_2d.vulkan();
  const Tensor weight = weight_arg.is_vulkan() ? weight_arg : weight_arg.vulkan();

  TORCH_INTERNAL_ASSERT(can_run_bfloat16_buffer_linear(input, weight, bias_arg));

  vTensor v_input = convert(input);
  vTensor v_weight = convert(weight);

  vTensor v_output{
      context,
      {
          input_arg_2d.sizes()[Layout::Parameter::height],
          weight.sizes()[Layout::Parameter::height],
      },
      api::kFloat,
      api::StorageType::BUFFER,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
  };

  const struct {
    int32_t out_width;
    int32_t out_height;
    int32_t inner_dim;
    int32_t reserved;
  } block{
      api::utils::safe_downcast<int32_t>(weight.size(Layout::Parameter::height)),
      api::utils::safe_downcast<int32_t>(
          input_arg_2d.size(Layout::Parameter::height)),
      api::utils::safe_downcast<int32_t>(
          input_arg_2d.size(Layout::Parameter::width)),
      0,
  };

  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};
  const api::utils::uvec3 global_size{
      api::utils::safe_downcast<uint32_t>(weight.size(Layout::Parameter::height)),
      api::utils::safe_downcast<uint32_t>(
          input_arg_2d.size(Layout::Parameter::height)),
      1u,
  };

  context->submit_compute_job(
      VK_KERNEL(mm_buffer_bfloat16),
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
      v_weight.buffer(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.buffer_metadata(),
      params.buffer());

  Tensor output = convert(v_output);

  std::optional<Tensor> bias = upcast_bfloat16_bias_for_vulkan_linear(bias_arg);
  if (bias && bias->defined()) {
    if (!bias->is_vulkan()) {
      *bias = bias->vulkan();
    }
    output = output.add(*bias);
  }

  if (input_arg.dim() == 2) {
    return output;
  }

  std::vector<int64_t> shape;
  shape.reserve(static_cast<size_t>(std::max<int64_t>(0, input_arg.dim())));
  for (const auto i : c10::irange(input_arg.dim() - 1)) {
    shape.emplace_back(input_arg.size(i));
  }
  shape.emplace_back(output.size(-1));
  return output.reshape(shape);
}

Tensor run_quantized_addmm_context(
    const Tensor& input_arg,
    const float alpha,
    const float beta,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context,
    double output_scale,
    int64_t output_zero_point) {
  api::Context* const context = api::context();

  const Tensor input_arg_2d =
      input_arg.dim() == 2 ? input_arg : reshape_to_2d(input_arg);
  const Tensor input =
      input_arg_2d.is_vulkan() ? input_arg_2d : input_arg_2d.vulkan();
  const vTensor& v_input = convert(input);
  const vTensor& packed_v_weight = convert(
      linear_context->get_val(LinearPackedContext::Packed::Weight).toTensor());
  const vTensor& packed_v_bias = convert(
      linear_context->get_val(LinearPackedContext::Packed::Bias).toTensor());
  const std::vector<int64_t> unpacked_weight_sizes =
      linear_context->get_val(LinearPackedContext::Packed::WeightSizes)
          .toIntVector();
  const bool bias_defined =
      linear_context->get_val(LinearPackedContext::Packed::BiasDefined)
          .toBool();

  TORCH_CHECK(
      usable(input, unpacked_weight_sizes),
      "Vulkan Linear not usable! "
      "Reason: The provided input tensor is either invalid on its own, or its "
      "combination with the provided weight and bias tensors are unsupported by "
      "Vulkan impl.");

  TORCH_CHECK(
      (packed_v_weight.is_quantized() && v_input.is_quantized()),
      "run_quantized_addmm_context called for quantized version with unquantized input");

  vTensor v_output{
      context,
      {
          input_arg_2d.sizes()[Layout::Parameter::height],
          unpacked_weight_sizes[Layout::Parameter::width],
      },
      v_input.dtype(),
  };

  v_output.set_is_quantized();
  v_output.set_scale(output_scale);
  v_output.set_zero_point(output_zero_point);

  if (bias_defined) {
    api::UniformParamsBuffer params;
    api::ShaderInfo compute_shader;
    compute_shader = (kQInt8 == input_arg.scalar_type())
        ? VK_KERNEL(quantized_addmm_qint8)
        : VK_KERNEL(quantized_addmm_quint8);
    const struct {
      uvec3 size;
      int32_t K;
      uvec3 um1_size;
      int32_t K1;
      uvec3 um2_size;
      int32_t K2;
      uvec3 ut_size;
      int32_t K3;
      vec2 multiplier;
      vec2 input_scales;
      float out_scale;
      float _1;
      ivec2 input_zero_points;
      int32_t out_zero_point;
      int32_t _2;
    } block{
        v_output.extents(),
        safe_downcast<int32_t>(
            div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(2))),
        v_input.extents(),
        0u,
        packed_v_weight.extents(),
        0u,
        packed_v_bias.extents(),
        0u,
        {
            alpha,
            beta,
        },
        {
            safe_downcast<float>(v_input.get_scale()),
            safe_downcast<float>(packed_v_weight.get_scale()),
        },
        safe_downcast<float>(output_scale),
        0.0f,
        {
            safe_downcast<int32_t>(v_input.get_zero_point()),
            safe_downcast<int32_t>(packed_v_weight.get_zero_point()),
        },
        safe_downcast<int32_t>(output_zero_point),
        0u,
    };
    params = api::UniformParamsBuffer(context, block);

    api::PipelineBarrier pipeline_barrier{};
    context->submit_compute_job(
        // shader descriptor
        compute_shader,
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {
            safe_downcast<uint32_t>(
                div_up(v_output.sizes()[Layout::Parameter::width], INT64_C(2))),
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::height], INT64_C(2))),
            1,
        },
        // local work group size
        {8, 8, 1},
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());

  } else { // no bias
    api::UniformParamsBuffer params;
    api::ShaderInfo compute_shader;
    const struct {
      uvec3 size;
      int32_t K;
      uvec3 um1_size;
      int32_t K1;
      uvec3 um2_size;
      int32_t K2;
      vec2 input_scales;
      float out_scale;
      float _1;
      ivec2 input_zero_points;
      int32_t out_zero_point;
      int32_t _2;
    } block_no_bias{
        v_output.extents(),
        safe_downcast<int32_t>(
            div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(2))),
        v_input.extents(),
        0u,
        packed_v_weight.extents(),
        0u,
        {
            safe_downcast<float>(v_input.get_scale()),
            safe_downcast<float>(packed_v_weight.get_scale()),
        },
        safe_downcast<float>(output_scale),
        0.0f,
        {
            safe_downcast<int32_t>(v_input.get_zero_point()),
            safe_downcast<int32_t>(packed_v_weight.get_zero_point()),
        },
        safe_downcast<int32_t>(output_zero_point),
        0u,
    };
    params = api::UniformParamsBuffer(context, block_no_bias);
    compute_shader = (kQInt8 == input_arg.scalar_type())
        ? VK_KERNEL(quantized_mm_qint8)
        : VK_KERNEL(quantized_mm_quint8);

    api::PipelineBarrier pipeline_barrier{};

    context->submit_compute_job(
        // shader descriptor
        compute_shader,
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {
            safe_downcast<uint32_t>(
                div_up(v_output.sizes()[Layout::Parameter::width], INT64_C(2))),
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::height], INT64_C(2))),
            1,
        },
        // local work group size
        {8, 8, 1},
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());
  }
  Tensor output = convert(v_output);
  if (input_arg.dim() == 2) {
    return output;
  } else {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(std::max<int64_t>(0, input_arg.dim())));
    for (const auto i : c10::irange(input_arg.dim() - 1)) {
      shape.emplace_back(input_arg.size(i));
    }
    shape.emplace_back(output.size(-1));
    Tensor reshaped_output = output.reshape(shape);
    if (c10::InferenceMode::is_enabled()) {
      reshaped_output = reshaped_output.clone();
    }
    return reshaped_output;
  }
}

Tensor run_addmm_context(
    const Tensor& input_arg,
    const float alpha,
    const float beta,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context,
    bool quantized,
    double output_scale,
    int64_t output_zero_point,
    const LinearPostOp post_op = LinearPostOp::None) {
  api::AllocationScope allocation_scope(
      linear_runtime_label(linear_context, "linear"));
  if (quantized) {
    return run_quantized_addmm_context(
        input_arg,
        alpha,
        beta,
        linear_context,
        output_scale,
        output_zero_point);
  }

  api::Context* const context = api::context();

  const Tensor compute_input_arg =
      upcast_bfloat16_input_for_vulkan_linear(input_arg);
  const Tensor input_arg_2d =
      compute_input_arg.dim() == 2 ? compute_input_arg
                                   : reshape_to_2d(compute_input_arg);
  const Tensor input =
      input_arg_2d.is_vulkan() ? input_arg_2d : input_arg_2d.vulkan();
  const vTensor& v_input = pack_inputs_using_width_packing(input);

  const vTensor& packed_v_weight = convert(
      linear_context->get_val(LinearPackedContext::Packed::Weight).toTensor());
  const vTensor& packed_v_bias = convert(
      linear_context->get_val(LinearPackedContext::Packed::Bias).toTensor());
  const std::vector<int64_t> unpacked_weight_sizes =
      linear_context->get_val(LinearPackedContext::Packed::WeightSizes)
          .toIntVector();
  const bool bias_defined =
      linear_context->get_val(LinearPackedContext::Packed::BiasDefined)
          .toBool();

  TORCH_CHECK(
      usable(input, unpacked_weight_sizes),
      "Vulkan Linear not usable! "
      "Reason: The provided input tensor is either invalid on its own, or its "
      "combination with the provided weight and bias tensors are unsupported by "
      "Vulkan impl.");

  TORCH_CHECK(
      v_input.gpu_memory_layout() == api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      "run_addmm_context must have width packed input");

  TORCH_CHECK(
      packed_v_weight.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
      "run_addmm_context must have height packed weight");

  vTensor v_output{
      context,
      {
          input_arg_2d.sizes()[Layout::Parameter::height],
          unpacked_weight_sizes[Layout::Parameter::width],
      },
      v_input.dtype(),
  };

  api::UniformParamsBuffer params;
  api::ShaderInfo compute_shader;
  // Step size is the 2d input's w dimension / 4.
  int step_size = div_up(v_input.sizes()[Layout::Parameter::width], INT64_C(4));
  const bool fuse_bias =
      bias_defined &&
      can_fuse_linear_bias(v_output, packed_v_bias, unpacked_weight_sizes);
  const bool fuse_gelu = fuse_bias && post_op == LinearPostOp::Gelu;

  if (fuse_gelu) {
    const struct {
      uvec4 shader_extents_and_step;
      uvec4 bias_extents;
      vec4 multipliers_and_gelu;
    } block_with_bias_gelu{
        {
            v_output.extents().data[0u],
            v_output.extents().data[1u],
            v_output.extents().data[2u],
            safe_downcast<uint32_t>(step_size),
        },
        {
            packed_v_bias.extents().data[0u],
            packed_v_bias.extents().data[1u],
            packed_v_bias.extents().data[2u],
            0u,
        },
        {alpha, beta, kGeluBeta, 0.0f},
    };
    params = api::UniformParamsBuffer(context, block_with_bias_gelu);
    compute_shader = VK_KERNEL(mm_bias_gelu);
  } else if (fuse_bias) {
    const struct {
      uvec4 shader_extents_and_step;
      uvec4 bias_extents;
      vec2 multipliers;
    } block_with_bias{
        {
            v_output.extents().data[0u],
            v_output.extents().data[1u],
            v_output.extents().data[2u],
            safe_downcast<uint32_t>(step_size),
        },
        {
            packed_v_bias.extents().data[0u],
            packed_v_bias.extents().data[1u],
            packed_v_bias.extents().data[2u],
            0u,
        },
        {alpha, beta},
    };
    params = api::UniformParamsBuffer(context, block_with_bias);
    compute_shader = VK_KERNEL(mm_bias);
  } else {
    const struct {
      uvec3 shader_extents;
      uint32_t mm_step_size;
    } block_no_bias{
        v_output.extents(),
        safe_downcast<uint32_t>(step_size),
    };
    params = api::UniformParamsBuffer(context, block_no_bias);
    compute_shader = VK_KERNEL(mm);
  }

  api::PipelineBarrier pipeline_barrier{};

  if (fuse_bias) {
    context->submit_compute_job(
        // shader descriptor
        compute_shader,
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::width],
                INT64_C(4))),
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::height],
                INT64_C(4))),
            1,
        },
        // local work group size
        {8, 8, 1},
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());
  } else {
    context->submit_compute_job(
        // shader descriptor
        compute_shader,
        // pipeline barrier
        pipeline_barrier,
        // global work group size
        {
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::width],
                INT64_C(4))),
            safe_downcast<uint32_t>(div_up(
                v_output.sizes()[Layout::Parameter::height],
                INT64_C(4))),
            1,
        },
        // local work group size
        {8, 8, 1},
        // fence handle
        VK_NULL_HANDLE,
        // shader arguments
        v_output.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
        // params buffer
        params.buffer());
  }

  Tensor output = convert(v_output);

  // addmm/linear operation, multiplying by alpha and adding bias when present.
  if (!fuse_bias && alpha != 1.0f) {
    output = output.mul(alpha);
  }
  if (!fuse_bias && bias_defined) {
    output = output.add(convert(packed_v_bias).mul(beta));
  }
  if (post_op == LinearPostOp::Gelu && !fuse_gelu) {
    output = at::gelu(output, "none");
  }

  if (input_arg.dim() == 2) {
    return output;
  } else {
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(std::max<int64_t>(0, input_arg.dim())));
    for (const auto i : c10::irange(input_arg.dim() - 1)) {
      shape.emplace_back(input_arg.size(i));
    }
    shape.emplace_back(output.size(-1));
    return output.reshape(shape);
  }
}

Tensor run_baddbmm_context(
    const Tensor& input_arg,
    const float alpha,
    const float beta,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  api::AllocationScope allocation_scope("bmm");
  // TODO: Refactor run_baddbmm_context and run_addmm_context into one.
  api::Context* const context = api::context();

  TORCH_CHECK(
      input_arg.dim() == 3,
      "Vulkan Linear not usable! "
      "Reason: The input has the wrong dimension; the tensor of a batch of matrices should contain 3 dimensions: batch, height, width.");

  const Tensor compute_input_arg =
      upcast_bfloat16_input_for_vulkan_linear(input_arg);
  const Tensor input =
      compute_input_arg.is_vulkan() ? compute_input_arg
                                    : compute_input_arg.vulkan();
  vTensor packed_v_input = pack_inputs_using_width_packing(input);

  const vTensor& packed_v_weight = convert(
      linear_context->get_val(LinearPackedContext::Packed::Weight).toTensor());
  const vTensor& packed_v_bias = convert(
      linear_context->get_val(LinearPackedContext::Packed::Bias).toTensor());
  const std::vector<int64_t> unpacked_weight_sizes =
      linear_context->get_val(LinearPackedContext::Packed::WeightSizes)
          .toIntVector();

  TORCH_CHECK(
      usable(input, unpacked_weight_sizes, true /*use batch*/),
      "Vulkan Linear not usable! "
      "Reason: The provided input tensor is either invalid on its own, or its "
      "combination with the provided weight and bias tensors are unsupported by "
      "Vulkan impl.");

  TORCH_CHECK(
      packed_v_input.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_WIDTH_PACKED,
      "run_addmm_context called for non-quantized version with unpacked weight");

  TORCH_CHECK(
      packed_v_weight.gpu_memory_layout() ==
          api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED,
      "run_addmm_context called for non-quantized version with unpacked weight");

  int64_t input_batch = packed_v_input.sizes()[Layout::BatchMatrices::batch];

  // Step size is the input's w dimension / 4.
  int64_t input_width = packed_v_input.sizes()[Layout::BatchMatrices::width];
  int64_t mm_step_size = div_up(input_width, INT64_C(4));

  vTensor v_output{
      context,
      {
          input_batch,
          packed_v_input.sizes()[Layout::BatchMatrices::height],
          unpacked_weight_sizes.back(), // "w" dimension in weight matrix
      },
      packed_v_input.dtype(),
  };

  const struct {
    uvec4 shader_extents_and_step;
    uvec4 batch_info;
  } block_no_bias{
      {
          v_output.extents().data[0u],
          v_output.extents().data[1u],
          v_output.extents().data[2u],
          safe_downcast<uint32_t>(mm_step_size),
      },
      {
          safe_downcast<uint32_t>(input_batch),
          0u,
          0u,
          0u,
      },
  };

  api::UniformParamsBuffer params(context, block_no_bias);

  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      // shader descriptor
      VK_KERNEL(bmm_channel_packed),
      // pipeline barrier
      pipeline_barrier,
      // global work group size
      {
          safe_downcast<uint32_t>(div_up(
              v_output.sizes()[Layout::BatchMatrices::width], INT64_C(4))),
          safe_downcast<uint32_t>(div_up(
              v_output.sizes()[Layout::BatchMatrices::height], INT64_C(4))),
          v_output.extents().data[2u],
      },
      // local work group size
      {8, 8, 1},
      // fence handle
      VK_NULL_HANDLE,
      // shader arguments
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      packed_v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      packed_v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // params buffer
      params.buffer());

  // The dedicated batched kernel writes up to four batch results directly into
  // each channel-packed output texel, so no post-slice is needed here.
  return convert(v_output).mul(alpha).add(convert(packed_v_bias).mul(beta));
}

Tensor addmm(
    const Tensor& bias,
    const Tensor& input,
    const Tensor& weight,
    const Scalar& beta,
    const Scalar& alpha) {
  return run_addmm_context(
      input,
      alpha.to<float>(),
      beta.to<float>(),
      c10::make_intrusive<LinearPackedContext>(
          LinearPackedContext(weight, bias)),
      false,
      0,
      0);
}

Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  const Tensor linear_input =
      input.dim() == 2 ? input : reshape_to_2d_buffer_linear(input);
  const Tensor linear_weight = weight.is_vulkan() ? weight : weight.vulkan();
  const std::optional<Tensor> linear_bias =
      (bias && bias->defined() && !bias->is_vulkan()) ? bias->vulkan() : bias;

  if (can_run_bfloat16_buffer_linear(linear_input, linear_weight, linear_bias)) {
    return run_bfloat16_buffer_linear(input, linear_weight, linear_bias);
  }

  return run_addmm_context(
      input,
      1.0f,
      1.0f,
      get_or_create_linear_context(weight, bias),
      false,
      0,
      0);
}

Tensor linear_gelu(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias) {
  return run_addmm_context(
      input,
      1.0f,
      1.0f,
      get_or_create_linear_context(weight, bias),
      false,
      0,
      0,
      LinearPostOp::Gelu);
}

Tensor mm(const Tensor& mat1_arg, const Tensor& mat2_arg) {
  return run_addmm_context(
      mat1_arg,
      1.0f,
      1.0f,
      c10::make_intrusive<LinearPackedContext>(
          LinearPackedContext(mat2_arg, std::optional<Tensor>())),
      false,
      0,
      0);
}

Tensor bmm(const Tensor& mat1_arg, const Tensor& mat2_arg) {
  return run_baddbmm_context(
      mat1_arg,
      1.0f,
      1.0f,
      c10::make_intrusive<LinearPackedContext>(LinearPackedContext(
          mat2_arg, std::optional<Tensor>(), true /*use batch*/)));
}

Tensor baddbmm(
    const Tensor& bias,
    const Tensor& input,
    const Tensor& weight,
    const Scalar& beta,
    const Scalar& alpha) {
  return run_baddbmm_context(
      input,
      alpha.to<float>(),
      beta.to<float>(),
      c10::make_intrusive<LinearPackedContext>(
          LinearPackedContext(weight, bias, true /*use batch*/)));
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::addmm"), TORCH_FN(addmm));
  m.impl(TORCH_SELECTIVE_NAME("aten::linear"), TORCH_FN(linear));
  m.impl(TORCH_SELECTIVE_NAME("aten::mm"), TORCH_FN(mm));
  m.impl(TORCH_SELECTIVE_NAME("aten::bmm"), TORCH_FN(bmm));
  m.impl(TORCH_SELECTIVE_NAME("aten::baddbmm"), TORCH_FN(baddbmm));
}

#endif /* USE_VULKAN_API */

} // namespace

LinearPackedContext::LinearPackedContext(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const bool use_batch,
    std::string allocation_label,
    const bool retain_unpacked)
    : unpacked_{c10::AnyType::get()} {
  allocation_label_ = std::move(allocation_label);
  api::AllocationScope allocation_scope(
      linear_pack_label(allocation_label_, use_batch));
  const Tensor prepared_weight =
      use_batch ? weight : materialize_inference_vulkan_matrix_arg(weight);
  const Tensor packed_weight =
      upcast_bfloat16_weight_for_vulkan_linear(prepared_weight);
  const std::optional<Tensor> packed_bias =
      upcast_bfloat16_bias_for_vulkan_linear(bias);
  TORCH_CHECK(
      available(packed_weight, packed_bias, use_batch),
      "Vulkan Linear not available! "
      "Reason: The provided (weight, bias) parameters are either invalid "
      "individually or their combination is not supported by Vulkan Impl.");

  packed_.reserve(Packed::NumArgs);
  packed_.emplace_back(convert(pack_weights(packed_weight, use_batch)));
  const auto& packed_biases = packed_weight.is_quantized()
      ? pack_biases_quantized_weights(packed_weight, packed_bias, use_batch)
      : pack_biases(packed_weight, packed_bias, use_batch);
  packed_.emplace_back(convert(packed_biases));
  packed_.emplace_back(packed_weight.sizes());
  packed_.emplace_back(packed_bias && packed_bias->defined());

  if (retain_unpacked && !at::globalContext().releaseWeightsWhenPrepacking()) {
    unpacked_.reserve(Unpacked::NumArgs);
    unpacked_.emplace_back(packed_weight);
    unpacked_.emplace_back(packed_bias);
  }
}

LinearPackedContext LinearPackedContext::pack(c10::impl::GenericList unpacked) {
  return LinearPackedContext(
      unpacked.get(Unpacked::Weight).toTensor(),
      get_optional_tensor(unpacked, Unpacked::Bias));
}

c10::intrusive_ptr<LinearPackedContext> create_linear_context(
    Tensor&& weight,
    std::optional<Tensor>&& bias) {
  return c10::make_intrusive<LinearPackedContext>(
      LinearPackedContext(weight, bias));
}

c10::intrusive_ptr<LinearPackedContext> create_linear_context_labeled(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    std::string label) {
  const Tensor prepared_weight =
      (c10::InferenceMode::is_enabled() && weight.is_vulkan() &&
       weight.dim() == 2)
      ? weight.cpu().t().contiguous()
      : weight.t();
  return c10::make_intrusive<LinearPackedContext>(
      LinearPackedContext(
          prepared_weight, bias, false, std::move(label)));
}

Tensor run_linear_context(
    const Tensor& input,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  return run_addmm_context(input, 1.0f, 1.0f, linear_context, false, 0, 0);
}

Tensor run_linear_gelu_context(
    const Tensor& input,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  return run_addmm_context(
      input,
      1.0f,
      1.0f,
      linear_context,
      false,
      0,
      0,
      LinearPostOp::Gelu);
}

Tensor run_qlinear_context(
    const Tensor& input_arg,
    double output_scale,
    int64_t output_zero_point,
    const c10::intrusive_ptr<LinearPackedContext>& linear_context) {
  return run_addmm_context(
      input_arg,
      1.0f,
      1.0f,
      linear_context,
      true,
      output_scale,
      output_zero_point);
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
