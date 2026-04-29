# Shaders processing
if(NOT USE_VULKAN)
  return()
endif()

set(VULKAN_GEN_OUTPUT_PATH "${CMAKE_BINARY_DIR}/vulkan/ATen/native/vulkan")
set(VULKAN_GEN_ARG_ENV "")

set(PYTORCH_VULKAN_API_VERSION "1.3" CACHE STRING "Minimum Vulkan API version")
set(PYTORCH_VULKAN_TARGET_ENV "vulkan1.3" CACHE STRING "glslc target environment")
set(PYTORCH_VULKAN_TARGET_SPV "spv1.6" CACHE STRING "SPIR-V target version")
option(PYTORCH_VULKAN_STRICT_SPIRV_VERSION "Require generated SPIR-V 1.6" ON)
set(VULKAN_GLSLC_EXECUTABLE "" CACHE FILEPATH "Path to glslc")
set(VULKAN_SPIRV_VAL_EXECUTABLE "" CACHE FILEPATH "Path to spirv-val")

if(NOT PYTORCH_VULKAN_API_VERSION STREQUAL "1.3")
  message(FATAL_ERROR "PyTorch Vulkan backend requires Vulkan API 1.3; got ${PYTORCH_VULKAN_API_VERSION}")
endif()
if(NOT PYTORCH_VULKAN_TARGET_ENV STREQUAL "vulkan1.3")
  message(FATAL_ERROR "PyTorch Vulkan shader generation requires --target-env=vulkan1.3; got ${PYTORCH_VULKAN_TARGET_ENV}")
endif()
if(NOT PYTORCH_VULKAN_TARGET_SPV STREQUAL "spv1.6")
  message(FATAL_ERROR "PyTorch Vulkan shader generation requires --target-spv=spv1.6; got ${PYTORCH_VULKAN_TARGET_SPV}")
endif()

if(USE_VULKAN_RELAXED_PRECISION)
  list(APPEND VULKAN_GEN_ARG_ENV "PRECISION=mediump")
endif()
if(USE_VULKAN_FP16_INFERENCE)
  list(APPEND VULKAN_GEN_ARG_ENV "FLOAT_IMAGE_FORMAT=rgba16f")
else()
  list(APPEND VULKAN_GEN_ARG_ENV "FLOAT_IMAGE_FORMAT=rgba32f")
endif()

# Precompiling shaders
if(VULKAN_GLSLC_EXECUTABLE)
  set(GLSLC_PATH "${VULKAN_GLSLC_EXECUTABLE}")
elseif(ANDROID)
  if(NOT ANDROID_NDK)
    message(FATAL_ERROR "ANDROID_NDK not set")
  endif()

  set(GLSLC_PATH "${ANDROID_NDK}/shader-tools/${ANDROID_NDK_HOST_SYSTEM_NAME}/glslc")
else()
  find_program(
    GLSLC_PATH glslc
    PATHS
    ENV VULKAN_SDK
    PATHS "$ENV{VULKAN_SDK}/${CMAKE_HOST_SYSTEM_PROCESSOR}/bin"
    PATHS "$ENV{VULKAN_SDK}/bin"
  )

  if(NOT GLSLC_PATH)
    message(FATAL_ERROR "USE_VULKAN glslc not found")
  endif()
endif()

if(VULKAN_SPIRV_VAL_EXECUTABLE)
  set(SPIRV_VAL_PATH "${VULKAN_SPIRV_VAL_EXECUTABLE}")
else()
  find_program(
    SPIRV_VAL_PATH spirv-val
    PATHS
    ENV VULKAN_SDK
    PATHS "$ENV{VULKAN_SDK}/${CMAKE_HOST_SYSTEM_PROCESSOR}/bin"
    PATHS "$ENV{VULKAN_SDK}/bin")
endif()

if(PYTORCH_VULKAN_STRICT_SPIRV_VERSION AND NOT SPIRV_VAL_PATH)
  message(FATAL_ERROR "PYTORCH_VULKAN_STRICT_SPIRV_VERSION requires spirv-val. Set VULKAN_SPIRV_VAL_EXECUTABLE.")
endif()

set(VULKAN_SPV_PREFLIGHT_DIR "${CMAKE_BINARY_DIR}/vulkan/preflight")
file(MAKE_DIRECTORY "${VULKAN_SPV_PREFLIGHT_DIR}")
set(VULKAN_SPV_PREFLIGHT_GLSL "${VULKAN_SPV_PREFLIGHT_DIR}/preflight.comp")
set(VULKAN_SPV_PREFLIGHT_SPV "${VULKAN_SPV_PREFLIGHT_DIR}/preflight.spv")
file(WRITE "${VULKAN_SPV_PREFLIGHT_GLSL}" "#version 450\nlayout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\nvoid main() {}\n")
execute_process(
  COMMAND "${GLSLC_PATH}" --version
  OUTPUT_VARIABLE GLSLC_VERSION_OUTPUT
  ERROR_VARIABLE GLSLC_VERSION_ERROR
  RESULT_VARIABLE GLSLC_VERSION_RESULT)
if(GLSLC_VERSION_RESULT)
  message(FATAL_ERROR "Failed to run glslc --version using ${GLSLC_PATH}: ${GLSLC_VERSION_ERROR}")
endif()
execute_process(
  COMMAND "${GLSLC_PATH}"
    -fshader-stage=compute
    "${VULKAN_SPV_PREFLIGHT_GLSL}"
    -o "${VULKAN_SPV_PREFLIGHT_SPV}"
    "--target-env=${PYTORCH_VULKAN_TARGET_ENV}"
    "--target-spv=${PYTORCH_VULKAN_TARGET_SPV}"
    -Werror
  OUTPUT_VARIABLE GLSLC_PREFLIGHT_OUTPUT
  ERROR_VARIABLE GLSLC_PREFLIGHT_ERROR
  RESULT_VARIABLE GLSLC_PREFLIGHT_RESULT)
if(GLSLC_PREFLIGHT_RESULT)
  message(FATAL_ERROR "glslc cannot compile Vulkan shader for ${PYTORCH_VULKAN_TARGET_ENV}/${PYTORCH_VULKAN_TARGET_SPV}: ${GLSLC_PREFLIGHT_ERROR}")
endif()
file(READ "${VULKAN_SPV_PREFLIGHT_SPV}" VULKAN_SPV_PREFLIGHT_HEX HEX)
string(SUBSTRING "${VULKAN_SPV_PREFLIGHT_HEX}" 0 16 VULKAN_SPV_PREFLIGHT_HEADER)
if(PYTORCH_VULKAN_STRICT_SPIRV_VERSION AND NOT VULKAN_SPV_PREFLIGHT_HEADER STREQUAL "0302230700060100")
  message(FATAL_ERROR "glslc generated unexpected SPIR-V header ${VULKAN_SPV_PREFLIGHT_HEADER}; expected SPIR-V 1.6 header 0302230700060100")
endif()
if(SPIRV_VAL_PATH)
  execute_process(
    COMMAND "${SPIRV_VAL_PATH}" --target-env "${PYTORCH_VULKAN_TARGET_ENV}" "${VULKAN_SPV_PREFLIGHT_SPV}"
    OUTPUT_VARIABLE SPIRV_VAL_PREFLIGHT_OUTPUT
    ERROR_VARIABLE SPIRV_VAL_PREFLIGHT_ERROR
    RESULT_VARIABLE SPIRV_VAL_PREFLIGHT_RESULT)
  if(SPIRV_VAL_PREFLIGHT_RESULT)
    message(FATAL_ERROR "spirv-val rejected preflight shader for ${PYTORCH_VULKAN_TARGET_ENV}: ${SPIRV_VAL_PREFLIGHT_ERROR}")
  endif()
endif()
message(STATUS "Vulkan shader compiler: ${GLSLC_PATH}")
string(STRIP "${GLSLC_VERSION_OUTPUT}" GLSLC_VERSION_OUTPUT_STRIPPED)
message(STATUS "Vulkan glslc version: ${GLSLC_VERSION_OUTPUT_STRIPPED}")
message(STATUS "Vulkan shader target env: ${PYTORCH_VULKAN_TARGET_ENV}")
message(STATUS "Vulkan shader target SPIR-V: ${PYTORCH_VULKAN_TARGET_SPV}")
message(STATUS "Vulkan SPIR-V validator: ${SPIRV_VAL_PATH}")

set(VULKAN_GEN_VALIDATION_ARGS "")
if(PYTORCH_VULKAN_STRICT_SPIRV_VERSION)
  list(APPEND VULKAN_GEN_VALIDATION_ARGS "--strict-spv-version")
endif()
if(SPIRV_VAL_PATH)
  list(APPEND VULKAN_GEN_VALIDATION_ARGS "--spirv-val-path=${SPIRV_VAL_PATH}")
endif()

set(PYTHONPATH "$ENV{PYTHONPATH}")
set(NEW_PYTHONPATH ${PYTHONPATH})
list(APPEND NEW_PYTHONPATH "${CMAKE_CURRENT_LIST_DIR}/..")
set(ENV{PYTHONPATH} ${NEW_PYTHONPATH})
execute_process(
  COMMAND
  "${Python_EXECUTABLE}"
  ${CMAKE_CURRENT_LIST_DIR}/../tools/gen_vulkan_spv.py
  --glsl-paths ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/vulkan/glsl
  --output-path ${VULKAN_GEN_OUTPUT_PATH}
  --glslc-path=${GLSLC_PATH}
  --tmp-dir-path=${CMAKE_BINARY_DIR}/vulkan/spv
  --target-env=${PYTORCH_VULKAN_TARGET_ENV}
  --target-spv=${PYTORCH_VULKAN_TARGET_SPV}
  --manifest-path=${CMAKE_BINARY_DIR}/vulkan/vulkan_shader_manifest.json
  ${VULKAN_GEN_VALIDATION_ARGS}
  --env ${VULKAN_GEN_ARG_ENV}
  RESULT_VARIABLE error_code)
set(ENV{PYTHONPATH} ${PYTHONPATH})

  if(error_code)
    message(FATAL_ERROR "Failed to gen spv.h and spv.cpp with precompiled shaders for Vulkan backend")
  endif()

set(vulkan_generated_cpp ${VULKAN_GEN_OUTPUT_PATH}/spv.cpp)
