#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/Runtime.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/vulkan/Module.h>

#include <sstream>

namespace py = pybind11;

namespace torch::vulkan {

namespace {

using at::native::vulkan::api::PhysicalDevice;

uint64_t total_memory_bytes(const PhysicalDevice& physical_device) {
  const auto& memory_properties = physical_device.memory_properties;
  uint64_t total_device_local = 0u;
  uint64_t total_heap_bytes = 0u;
  for (uint32_t heap_i = 0u; heap_i < memory_properties.memoryHeapCount;
       ++heap_i) {
    total_heap_bytes += memory_properties.memoryHeaps[heap_i].size;
    if (memory_properties.memoryHeaps[heap_i].flags &
        VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      total_device_local += memory_properties.memoryHeaps[heap_i].size;
    }
  }
  return total_device_local > 0u ? total_device_local : total_heap_bytes;
}

std::string format_api_version(const uint32_t version) {
  std::ostringstream stream;
  stream << VK_VERSION_MAJOR(version) << "." << VK_VERSION_MINOR(version)
         << "." << VK_VERSION_PATCH(version);
  return stream.str();
}

std::string format_device_type(const VkPhysicalDeviceType device_type) {
  switch (device_type) {
    case VK_PHYSICAL_DEVICE_TYPE_OTHER:
      return "other";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return "integrated_gpu";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return "discrete_gpu";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return "virtual_gpu";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return "cpu";
    default:
      return "unknown";
  }
}

struct VulkanDeviceProperties final {
  int64_t index;
  std::string name;
  std::string type;
  uint32_t vendor_id;
  uint32_t device_id;
  std::string uuid;
  std::string luid;
  std::string pci_address;
  std::string pipeline_cache_uuid;
  std::string api_version;
  uint32_t api_version_raw;
  uint32_t driver_version;
  uint64_t total_memory;
  bool has_unified_memory;
  uint32_t num_compute_queues;
  bool has_timestamps;
  float timestamp_period;
  uint32_t max_image_dimension_2d;
  uint32_t max_image_dimension_3d;
  bool has_shader_bfloat16;
  bool has_shader_int8;
  bool has_storage_buffer_8bit;
  bool has_buffer_device_address;
  bool supports_push_descriptor;
  bool supports_descriptor_buffer;
  bool has_cooperative_matrix;
  bool has_subgroup_size_control;
  bool has_compute_full_subgroups;
  uint32_t min_subgroup_size;
  uint32_t max_subgroup_size;
  uint32_t max_compute_workgroup_subgroups;
  uint32_t required_subgroup_size_stages;
  uint32_t cooperative_matrix_supported_stages;
  uint32_t cooperative_matrix_property_count;

  explicit VulkanDeviceProperties(
      const c10::DeviceIndex device_index,
      const PhysicalDevice& physical_device)
      : index(device_index),
        name(physical_device.properties.deviceName),
        type(format_device_type(physical_device.properties.deviceType)),
        vendor_id(physical_device.properties.vendorID),
        device_id(physical_device.properties.deviceID),
        uuid(physical_device.uuid),
        luid(physical_device.luid),
        pci_address(physical_device.pci_address),
        pipeline_cache_uuid(physical_device.pipeline_cache_uuid),
        api_version(format_api_version(physical_device.properties.apiVersion)),
        api_version_raw(physical_device.properties.apiVersion),
        driver_version(physical_device.properties.driverVersion),
        total_memory(total_memory_bytes(physical_device)),
        has_unified_memory(physical_device.has_unified_memory),
        num_compute_queues(physical_device.num_compute_queues),
        has_timestamps(physical_device.has_timestamps),
        timestamp_period(physical_device.timestamp_period),
        max_image_dimension_2d(
            physical_device.properties.limits.maxImageDimension2D),
        max_image_dimension_3d(
            physical_device.properties.limits.maxImageDimension3D),
        has_shader_bfloat16(physical_device.has_shader_bfloat16),
        has_shader_int8(physical_device.has_shader_int8),
        has_storage_buffer_8bit(physical_device.has_storage_buffer_8bit),
        has_buffer_device_address(physical_device.has_buffer_device_address),
        supports_push_descriptor(physical_device.supports_push_descriptor),
        supports_descriptor_buffer(physical_device.supports_descriptor_buffer),
        has_cooperative_matrix(physical_device.has_cooperative_matrix),
        has_subgroup_size_control(physical_device.has_subgroup_size_control),
        has_compute_full_subgroups(physical_device.has_compute_full_subgroups),
        min_subgroup_size(physical_device.min_subgroup_size),
        max_subgroup_size(physical_device.max_subgroup_size),
        max_compute_workgroup_subgroups(
            physical_device.max_compute_workgroup_subgroups),
        required_subgroup_size_stages(
            physical_device.required_subgroup_size_stages),
        cooperative_matrix_supported_stages(
            physical_device.cooperative_matrix_supported_stages),
        cooperative_matrix_property_count(static_cast<uint32_t>(
            physical_device.cooperative_matrix_properties.size())) {}
};

void registerVulkanDeviceProperties(PyObject* module) {
  auto m = py::handle(module).cast<py::module_>();

  py::class_<VulkanDeviceProperties>(m, "_VulkanDeviceProperties")
      .def_readonly("index", &VulkanDeviceProperties::index)
      .def_readonly("name", &VulkanDeviceProperties::name)
      .def_readonly("type", &VulkanDeviceProperties::type)
      .def_readonly("vendor_id", &VulkanDeviceProperties::vendor_id)
      .def_readonly("device_id", &VulkanDeviceProperties::device_id)
      .def_readonly("uuid", &VulkanDeviceProperties::uuid)
      .def_readonly("luid", &VulkanDeviceProperties::luid)
      .def_readonly("pci_address", &VulkanDeviceProperties::pci_address)
      .def_readonly(
          "pipeline_cache_uuid", &VulkanDeviceProperties::pipeline_cache_uuid)
      .def_readonly("api_version", &VulkanDeviceProperties::api_version)
      .def_readonly("api_version_raw", &VulkanDeviceProperties::api_version_raw)
      .def_readonly("driver_version", &VulkanDeviceProperties::driver_version)
      .def_readonly("total_memory", &VulkanDeviceProperties::total_memory)
      .def_readonly(
          "has_unified_memory", &VulkanDeviceProperties::has_unified_memory)
      .def_readonly(
          "num_compute_queues", &VulkanDeviceProperties::num_compute_queues)
      .def_readonly(
          "has_timestamps", &VulkanDeviceProperties::has_timestamps)
      .def_readonly(
          "timestamp_period", &VulkanDeviceProperties::timestamp_period)
      .def_readonly(
          "max_image_dimension_2d",
          &VulkanDeviceProperties::max_image_dimension_2d)
      .def_readonly(
          "max_image_dimension_3d",
          &VulkanDeviceProperties::max_image_dimension_3d)
      .def_readonly(
          "has_shader_bfloat16",
          &VulkanDeviceProperties::has_shader_bfloat16)
      .def_readonly(
          "has_shader_int8", &VulkanDeviceProperties::has_shader_int8)
      .def_readonly(
          "has_storage_buffer_8bit",
          &VulkanDeviceProperties::has_storage_buffer_8bit)
      .def_readonly(
          "has_buffer_device_address",
          &VulkanDeviceProperties::has_buffer_device_address)
      .def_readonly(
          "supports_push_descriptor",
          &VulkanDeviceProperties::supports_push_descriptor)
      .def_readonly(
          "supports_descriptor_buffer",
          &VulkanDeviceProperties::supports_descriptor_buffer)
      .def_readonly(
          "has_cooperative_matrix",
          &VulkanDeviceProperties::has_cooperative_matrix)
      .def_readonly(
          "has_subgroup_size_control",
          &VulkanDeviceProperties::has_subgroup_size_control)
      .def_readonly(
          "has_compute_full_subgroups",
          &VulkanDeviceProperties::has_compute_full_subgroups)
      .def_readonly(
          "min_subgroup_size", &VulkanDeviceProperties::min_subgroup_size)
      .def_readonly(
          "max_subgroup_size", &VulkanDeviceProperties::max_subgroup_size)
      .def_readonly(
          "max_compute_workgroup_subgroups",
          &VulkanDeviceProperties::max_compute_workgroup_subgroups)
      .def_readonly(
          "required_subgroup_size_stages",
          &VulkanDeviceProperties::required_subgroup_size_stages)
      .def_readonly(
          "cooperative_matrix_supported_stages",
          &VulkanDeviceProperties::cooperative_matrix_supported_stages)
      .def_readonly(
          "cooperative_matrix_property_count",
          &VulkanDeviceProperties::cooperative_matrix_property_count)
      .def("__repr__", [](const VulkanDeviceProperties& properties) {
        std::ostringstream stream;
        stream << "_VulkanDeviceProperties(name='" << properties.name
               << "', index=" << properties.index << ", type='"
               << properties.type << "', uuid='" << properties.uuid
               << "', total_memory="
               << properties.total_memory << ", api_version='"
               << properties.api_version << "', num_compute_queues="
               << properties.num_compute_queues << ")";
        return stream.str();
      });
}

void initMethodBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module_>();
  m.def("_vulkan_getDeviceCount", []() {
    return at::native::vulkan::api::device_count();
  });
  m.def("_vulkan_getDevice", []() {
    return at::native::vulkan::api::current_device();
  });
  m.def("_vulkan_setDevice", [](c10::DeviceIndex device_index) {
    at::native::vulkan::api::set_current_device(device_index);
  });
  m.def("_vulkan_exchangeDevice", [](c10::DeviceIndex device_index) {
    return at::native::vulkan::api::exchange_device(device_index);
  });
  m.def("_vulkan_getDeviceProperties", [](c10::DeviceIndex device_index) {
    const c10::DeviceIndex resolved_device_index =
        device_index >= 0 ? device_index
                          : at::native::vulkan::api::current_device();
    return VulkanDeviceProperties(
        resolved_device_index,
        at::native::vulkan::api::runtime()->get_physical_device(
            resolved_device_index));
  });
}

} // namespace

void initModule(PyObject* module) {
  registerVulkanDeviceProperties(module);
  initMethodBindings(module);
}

} // namespace torch::vulkan
