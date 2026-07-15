#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/PackedWeight.h>
#include <c10/util/intrusive_ptr.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class LinearPackedContext;

namespace utils {

const std::string& resolve_vulkan_linear_runtime_label(
    const std::string& allocation_label,
    const char* fallback_label);

std::string make_vulkan_linear_pack_label(
    const std::string& allocation_label,
    const char* fallback_label);

PackedWeightHandle make_packed_weight_handle(
    Tensor,
    Tensor,
    std::vector<int64_t>,
    PackedWeightKind,
    bool bias_defined,
    bool quantized = false,
    PackedWeightResidencyClass residency_class =
        PackedWeightResidencyClass::PersistentInference);

std::optional<PackedWeightHandle> lookup_packed_weight_handle(
    const Tensor& source_weight,
    const std::optional<Tensor>& source_bias,
    IntArrayRef logical_weight_sizes,
    PackedWeightKind kind,
    bool quantized = false,
    uint64_t options_key = 0u);

void store_packed_weight_handle(
    const Tensor& source_weight,
    const std::optional<Tensor>& source_bias,
    IntArrayRef logical_weight_sizes,
    PackedWeightKind kind,
    const PackedWeightHandle& handle,
    bool quantized = false,
    uint64_t options_key = 0u);

void note_packed_weight_store_skip(
    IntArrayRef logical_weight_sizes,
    ScalarType dtype,
    PackedWeightKind kind,
    bool quantized,
    uint64_t options_key,
    const char* reason,
    size_t resident_nbytes);

std::vector<std::string> packed_weight_residency_snapshot();

void reset_packed_weight_residency_snapshot();

std::optional<c10::intrusive_ptr<LinearPackedContext>> lookup_linear_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias);

void store_linear_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const c10::intrusive_ptr<LinearPackedContext>& context);

std::optional<c10::intrusive_ptr<LinearPackedContext>>
lookup_labeled_linear_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const std::string& allocation_label);

void store_labeled_linear_context(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const std::string& allocation_label,
    const c10::intrusive_ptr<LinearPackedContext>& context);

bool release_retired_linear_contexts();
std::function<void()> take_retired_linear_context_cleanup();

bool release_retired_packed_weight_entries();
std::function<void()> take_retired_packed_weight_cleanup();

} // namespace utils
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
