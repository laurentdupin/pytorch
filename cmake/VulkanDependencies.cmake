if(NOT USE_VULKAN)
  return()
endif()

set(PYTORCH_VULKAN_API_VERSION "1.3" CACHE STRING "Minimum Vulkan API version")
set(PYTORCH_VULKAN_TARGET_ENV "vulkan1.3" CACHE STRING "glslc target environment")
set(PYTORCH_VULKAN_TARGET_SPV "spv1.6" CACHE STRING "SPIR-V target version")
option(PYTORCH_VULKAN_STRICT_SPIRV_VERSION "Require generated SPIR-V 1.6" ON)
set(VULKAN_GLSLC_EXECUTABLE "" CACHE FILEPATH "Path to glslc")
set(VULKAN_SPIRV_VAL_EXECUTABLE "" CACHE FILEPATH "Path to spirv-val")

if(ANDROID)
  if(NOT ANDROID_NDK)
    message(FATAL_ERROR "USE_VULKAN requires ANDROID_NDK set.")
  endif()

  # Vulkan from ANDROID_NDK
  set(VULKAN_INCLUDE_DIR "${ANDROID_NDK}/sources/third_party/vulkan/src/include")
  message(STATUS "VULKAN_INCLUDE_DIR:${VULKAN_INCLUDE_DIR}")

  set(VULKAN_ANDROID_NDK_WRAPPER_DIR "${ANDROID_NDK}/sources/third_party/vulkan/src/common")
  message(STATUS "Vulkan_ANDROID_NDK_WRAPPER_DIR:${VULKAN_ANDROID_NDK_WRAPPER_DIR}")
  set(VULKAN_WRAPPER_DIR "${VULKAN_ANDROID_NDK_WRAPPER_DIR}")

  add_library(
    VulkanWrapper
    STATIC
    ${VULKAN_WRAPPER_DIR}/vulkan_wrapper.h
    ${VULKAN_WRAPPER_DIR}/vulkan_wrapper.cpp)

  target_include_directories(VulkanWrapper PUBLIC .)
  target_include_directories(VulkanWrapper PUBLIC "${VULKAN_INCLUDE_DIR}")
  target_link_libraries(VulkanWrapper ${CMAKE_DL_LIBS})

  string(APPEND Vulkan_DEFINES " -DUSE_VULKAN_WRAPPER")
  list(APPEND Vulkan_INCLUDES ${VULKAN_WRAPPER_DIR})
  list(APPEND Vulkan_LIBS VulkanWrapper)

else()
  find_package(Vulkan 1.3 REQUIRED)

  if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "USE_VULKAN requires either Vulkan installed on system path or environment var VULKAN_SDK set.")
  endif()

  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES ${Vulkan_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${Vulkan_LIBRARIES})
  check_cxx_source_compiles("
    #include <vulkan/vulkan.h>
    #ifndef VK_VERSION_1_3
    #error VK_VERSION_1_3 missing
    #endif
    int main() {
      uint32_t version = VK_API_VERSION_1_3;
      VkPhysicalDeviceVulkan13Features features{};
      features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
      auto enumerate = &vkEnumerateInstanceVersion;
      (void)version;
      (void)features;
      (void)enumerate;
      return 0;
    }
  " PYTORCH_VULKAN_HEADERS_SUPPORT_1_3)
  unset(CMAKE_REQUIRED_INCLUDES)
  unset(CMAKE_REQUIRED_LIBRARIES)
  if(NOT PYTORCH_VULKAN_HEADERS_SUPPORT_1_3)
    message(FATAL_ERROR "USE_VULKAN requires Vulkan 1.3 headers with VK_API_VERSION_1_3 and VkPhysicalDeviceVulkan13Features.")
  endif()

  list(APPEND Vulkan_INCLUDES ${Vulkan_INCLUDE_DIRS})
  list(APPEND Vulkan_LIBS ${Vulkan_LIBRARIES})

  set(GOOGLE_SHADERC_INCLUDE_SEARCH_PATH ${Vulkan_INCLUDE_DIR})
  set(GOOGLE_SHADERC_LIBRARY_SEARCH_PATH ${Vulkan_LIBRARY})
endif()
