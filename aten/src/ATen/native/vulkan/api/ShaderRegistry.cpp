#include <ATen/native/vulkan/api/ShaderRegistry.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

bool ShaderRegistry::has_shader(const std::string& shader_name) {
  const ShaderListing::const_iterator it = listings_.find(shader_name);
  return it != listings_.end();
}

void ShaderRegistry::register_shader(ShaderInfo&& shader_info) {
  if (has_shader(shader_info.kernel_name)) {
    VK_THROW(
        "Shader with name ", shader_info.kernel_name, "already registered");
  }
  listings_.emplace(shader_info.kernel_name, shader_info);
}

const ShaderInfo& ShaderRegistry::get_shader_info(
    const std::string& shader_name) {
  const ShaderListing::const_iterator it = listings_.find(shader_name);

  VK_CHECK_COND(
      it != listings_.end(),
      "Could not find ShaderInfo with name ",
      shader_name);

  return it->second;
}

ShaderRegistry& shader_registry() {
  static ShaderRegistry registry;
  return registry;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
