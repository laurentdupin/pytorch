#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Shader.h>

#include <string>
#include <unordered_map>

#define VK_KERNEL(shader_name) \
  ::at::native::vulkan::api::shader_registry().get_shader_info(#shader_name)

#define VK_KERNEL_FROM_STR(shader_name_str) \
  ::at::native::vulkan::api::shader_registry().get_shader_info(shader_name_str)

namespace at {
namespace native {
namespace vulkan {
namespace api {

class ShaderRegistry final {
  using ShaderListing = std::unordered_map<std::string, ShaderInfo>;

  ShaderListing listings_;

 public:
  /*
   * Check if the registry has a shader registered under the given name
   */
  bool has_shader(const std::string& shader_name);

  /*
   * Register a ShaderInfo to a given shader name
   */
  void register_shader(ShaderInfo&& shader_info);

  /*
   * Given a shader name, return the ShaderInfo which contains the SPIRV binary
   */
  const ShaderInfo& get_shader_info(const std::string& shader_name);
};

class ShaderRegisterInit final {
  using InitFn = void();

 public:
  ShaderRegisterInit(InitFn* init_fn) {
    init_fn();
  };
};

// The global shader registry is retrieved using this function, where it is
// declared as a static local variable.
ShaderRegistry& shader_registry();

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
