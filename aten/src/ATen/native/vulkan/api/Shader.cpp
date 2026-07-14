#include <utility>

#include <ATen/native/vulkan/api/Shader.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

namespace {

constexpr uint32_t kSpvMagic = 0x07230203u;
constexpr uint32_t kSpvVersion16 = 0x00010600u;

} // namespace

//
// ShaderInfo
//

ShaderInfo::ShaderInfo()
    : src_code{
          nullptr,
          0u,
      } {}

ShaderInfo::ShaderInfo(
    std::string name,
    const uint32_t* const spirv_bin,
    const uint32_t size,
    std::vector<VkDescriptorType>  layout)
    : src_code{
          spirv_bin,
          size,
      },
      kernel_name{std::move(name)},
      kernel_layout{std::move(layout)} {}

ShaderInfo::ShaderInfo(
    std::string name,
    const uint32_t* const spirv_bin,
    const uint32_t size,
    std::vector<VkDescriptorType>  layout,
    const std::vector<uint32_t>& tile_size,
    const StorageType bias_storage_type,
    const StorageType weight_storage_type,
    std::string source_path,
    std::string target_env,
    uint32_t spv_version_word,
    std::vector<uint32_t> capabilities,
    std::vector<std::string> extensions,
    std::vector<uint32_t> execution_modes,
    bool uses_local_size_id)
    : src_code{
          spirv_bin,
          size,
      },
      kernel_name{std::move(name)},
      kernel_layout{std::move(layout)},
      source_path{std::move(source_path)},
      target_env{std::move(target_env)},
      spv_version_word{spv_version_word},
      capabilities{std::move(capabilities)},
      extensions{std::move(extensions)},
      execution_modes{std::move(execution_modes)},
      uses_local_size_id{uses_local_size_id},
      tile_size(tile_size),
      bias_storage_type(bias_storage_type),
      weight_storage_type(weight_storage_type) {
  for (uint64_t i = 0; i < tile_size.size(); ++i) {
    out_tile_size.data[i] = tile_size[i];
  }
}

bool operator==(const ShaderInfo& _1, const ShaderInfo& _2) {
  return (
      _1.src_code.bin == _2.src_code.bin &&
      _1.src_code.size == _2.src_code.size);
}

//
// ShaderLayout
//

ShaderLayout::ShaderLayout(
    VkDevice device,
    const ShaderLayout::Signature& signature)
    : device_(device), handle_{VK_NULL_HANDLE} {
  std::vector<VkDescriptorSetLayoutBinding> bindings;

  uint32_t binding_num = 0u;
  for (const VkDescriptorType type : signature) {
    bindings.push_back({
        binding_num++, // binding
        type, // descriptorType
        1u, // descriptorCount
        VK_SHADER_STAGE_COMPUTE_BIT, // stageFlags
        nullptr, // pImmutableSamplers
    });
  }

  const VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      static_cast<uint32_t>(bindings.size()), // bindingCount
      bindings.data(), // pBindings
  };

  VK_CHECK(vkCreateDescriptorSetLayout(
      device_, &descriptor_set_layout_create_info, nullptr, &handle_));
}

ShaderLayout::ShaderLayout(ShaderLayout&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

ShaderLayout::~ShaderLayout() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyDescriptorSetLayout(device_, handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

void swap(ShaderLayout& lhs, ShaderLayout& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkDescriptorSetLayout tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

//
// ShaderModule
//

ShaderModule::ShaderModule(VkDevice device, const ShaderInfo& source)
    : device_(device), handle_{VK_NULL_HANDLE} {
  const uint32_t* code = source.src_code.bin;
  uint32_t size = source.src_code.size;
  VK_CHECK_COND(
      code,
      "Shader ",
      source.kernel_name,
      " has null SPIR-V code.");
  VK_CHECK_COND(
      size >= 5u * sizeof(uint32_t) && (size % sizeof(uint32_t)) == 0u,
      "Shader ",
      source.kernel_name,
      " has invalid SPIR-V byte size ",
      size,
      ".");
  VK_CHECK_COND(
      code[0] == kSpvMagic,
      "Shader ",
      source.kernel_name,
      " has invalid SPIR-V magic word ",
      code[0],
      ".");
  VK_CHECK_COND(
      code[1] == kSpvVersion16,
      "Shader ",
      source.kernel_name,
      " was generated for unsupported SPIR-V version word ",
      code[1],
      "; expected ",
      kSpvVersion16,
      " for Vulkan 1.3 / SPIR-V 1.6.");

  const VkShaderModuleCreateInfo shader_module_create_info{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      size, // codeSize
      code, // pCode
  };

  const VkResult result = vkCreateShaderModule(
      device_, &shader_module_create_info, nullptr, &handle_);
  VK_CHECK_COND(
      result == VK_SUCCESS,
      "vkCreateShaderModule failed for shader=",
      source.kernel_name,
      " result=",
      result,
      " source=",
      source.source_path,
      " target_env=",
      source.target_env,
      " spv_version_word=",
      source.spv_version_word,
      " capabilities=",
      source.capabilities.size(),
      " extensions=",
      source.extensions.size(),
      " uses_local_size_id=",
      source.uses_local_size_id ? 1 : 0,
      ".");
}

ShaderModule::ShaderModule(ShaderModule&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

ShaderModule::~ShaderModule() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyShaderModule(device_, handle_, nullptr);
  handle_ = VK_NULL_HANDLE;
}

void swap(ShaderModule& lhs, ShaderModule& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkShaderModule tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

//
// ShaderLayoutCache
//

ShaderLayoutCache::ShaderLayoutCache(VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

ShaderLayoutCache::ShaderLayoutCache(ShaderLayoutCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_), cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);
}

ShaderLayoutCache::~ShaderLayoutCache() {
  purge();
}

VkDescriptorSetLayout ShaderLayoutCache::retrieve(
    const ShaderLayoutCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if (cache_.cend() == it) {
    it = cache_.insert({key, ShaderLayoutCache::Value(device_, key)}).first;
  }

  return it->second.handle();
}

void ShaderLayoutCache::purge() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_.clear();
}

//
// ShaderCache
//

ShaderCache::ShaderCache(VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

ShaderCache::ShaderCache(ShaderCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_), cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);
}

ShaderCache::~ShaderCache() {
  purge();
}

VkShaderModule ShaderCache::retrieve(const ShaderCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if (cache_.cend() == it) {
    it = cache_.insert({key, ShaderCache::Value(device_, key)}).first;
  }

  return it->second.handle();
}

void ShaderCache::purge() {
  cache_.clear();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
