#pragma once

#ifdef USE_VULKAN_API

#include <ATen/ArrayRef.h>
#include <ATen/native/vulkan/ops/Common.h>
#include <cstdint>
#include <string>
#include <vector>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

struct VulkanTensorProvenanceRecord final {
  uint64_t sequence{0};
  uint64_t storage_id{0};
  uint64_t view_id{0};
  uint64_t generation{0};
  uint64_t logical_desc_hash{0};
  std::string writer_op;
  std::string route;
  std::string output_state;
  std::vector<std::string> input_states;
  std::vector<std::string> input_writers;
};

void record_tensor_write(
    const Tensor& output,
    const char* op_name,
    const char* route_name = nullptr,
    ArrayRef<Tensor> inputs = {});

Tensor record_tensor_write_and_return(
    Tensor output,
    const char* op_name,
    const char* route_name = nullptr,
    ArrayRef<Tensor> inputs = {});

void record_tensor_alias(
    const Tensor& output,
    const Tensor& base,
    const char* op_name,
    const char* route_name = nullptr);

Tensor record_tensor_alias_and_return(
    Tensor output,
    const Tensor& base,
    const char* op_name,
    const char* route_name = nullptr);

std::string describe_tensor_provenance(const Tensor& tensor);

std::string tensor_provenance_writer(const Tensor& tensor);

std::string tensor_provenance_route(const Tensor& tensor);

bool check_tensor_finite(const Tensor& tensor, const char* consumer_op);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
