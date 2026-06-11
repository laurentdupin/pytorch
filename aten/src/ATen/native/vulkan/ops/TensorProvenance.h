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

struct TensorContractProvenance final {
  const char* contract_name{nullptr};
  const char* contract_family{nullptr};
  const char* contract_tuple_id{nullptr};
  const char* contract_materialization_policy{nullptr};
};

struct VulkanTensorProvenanceRecord final {
  uint64_t sequence{0};
  uint64_t storage_id{0};
  uint64_t view_id{0};
  uint64_t generation{0};
  uint64_t logical_desc_hash{0};
  uint64_t root_input_key{0};
  std::string writer_op;
  std::string route;
  std::string contract_name;
  std::string contract_family;
  std::string contract_tuple_id;
  std::string contract_materialization_policy;
  std::string output_state;
  std::vector<std::string> input_states;
  std::vector<std::string> input_writers;
  std::vector<uint64_t> input_state_keys;
};

void record_tensor_write(
    const Tensor& output,
    const char* op_name,
    const char* route_name = nullptr,
    ArrayRef<Tensor> inputs = {},
    const TensorContractProvenance* contract_provenance = nullptr);

Tensor record_tensor_write_and_return(
    Tensor output,
    const char* op_name,
    const char* route_name = nullptr,
    ArrayRef<Tensor> inputs = {},
    const TensorContractProvenance* contract_provenance = nullptr);

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

uint64_t tensor_provenance_first_input_key(const Tensor& tensor);

bool check_tensor_finite(const Tensor& tensor, const char* consumer_op);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
