#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <functional>
#include <memory>
#include <mutex>
#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/vk_api.h>

#include <ATen/native/vulkan/api/Adapter.h>
#include <c10/macros/Export.h>
#include <c10/core/Device.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// A Vulkan Runtime initializes a Vulkan instance and decouples the concept of
// Vulkan instance initialization from initialization of, and subsequent
// interactions with,  Vulkan [physical and logical] devices as a precursor to
// multi-GPU support.  The Vulkan Runtime can be queried for available Adapters
// (i.e. physical devices) in the system which in turn can be used for creation
// of a Vulkan Context (i.e. logical devices).  All Vulkan tensors in PyTorch
// are associated with a Context to make tensor <-> device affinity explicit.
//

enum AdapterSelector {
  First,
};

struct RuntimeConfiguration final {
  bool enableValidationMessages;
  bool initDefaultDevice;
  AdapterSelector defaultSelector;
  uint32_t numRequestedQueues;
};

class TORCH_API Runtime final {
 public:
  explicit Runtime(const RuntimeConfiguration);

  // Do not allow copying. There should be only one global instance of this
  // class.
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  Runtime(Runtime&&) noexcept = delete;
  Runtime& operator=(Runtime&&) = delete;

  ~Runtime();

  using DeviceMapping = std::pair<PhysicalDevice, int32_t>;
  using AdapterPtr = std::unique_ptr<Adapter>;

 private:
  RuntimeConfiguration config_;

  VkInstance instance_;

  std::vector<DeviceMapping> device_mappings_;
  std::vector<AdapterPtr> adapters_;
  std::mutex adapters_mutex_;
  uint32_t default_adapter_i_;
  c10::DeviceIndex default_device_i_;

  VkDebugReportCallbackEXT debug_report_callback_;

 public:
  inline VkInstance instance() const {
    return instance_;
  }

  inline Adapter* get_adapter_p() {
    return get_adapter_p_for_device(default_device_i_);
  }

  inline Adapter* get_adapter_p(uint32_t i) {
    VK_CHECK_COND(
        i < adapters_.size(),
        "Pytorch Vulkan Runtime: Adapter at index ",
        i,
        " is not available!");
    return adapters_[i].get();
  }

  inline uint32_t device_count() const {
    return utils::safe_downcast<uint32_t>(device_mappings_.size());
  }

  Adapter* get_adapter_p_for_device(c10::DeviceIndex device_index);
  const PhysicalDevice& get_physical_device(c10::DeviceIndex device_index) const;

  inline uint32_t default_adapter_i() const {
    return default_adapter_i_;
  }

  inline c10::DeviceIndex default_device_index() const {
    return default_device_i_;
  }

  using Selector =
      std::function<uint32_t(const std::vector<Runtime::DeviceMapping>&)>;
  uint32_t create_adapter(const Selector&);
  uint32_t create_adapter(uint32_t device_i);
};

// The global runtime is retrieved using this function, where it is declared as
// a static local variable.
TORCH_API Runtime* runtime();

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
