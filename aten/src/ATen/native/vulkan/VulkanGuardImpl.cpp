#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <ATen/native/vulkan/api/Context.h>

namespace at::detail {

namespace {

struct VulkanGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  VulkanGuardImpl() = default;

  // NOLINTNEXTLINE
  explicit VulkanGuardImpl(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == DeviceType::Vulkan);
  }

  DeviceType type() const override {
    return DeviceType::Vulkan;
  }
  Device exchangeDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == DeviceType::Vulkan);
    return Device(
        DeviceType::Vulkan,
        native::vulkan::api::exchange_device(d.index()));
  }
  Device getDevice() const override {
    return Device(DeviceType::Vulkan, native::vulkan::api::current_device());
  }
  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == DeviceType::Vulkan);
    native::vulkan::api::set_current_device(d.index());
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    try {
      if (d.type() == DeviceType::Vulkan && d.index() >= 0) {
        native::vulkan::api::set_current_device(d.index());
      }
    } catch (...) {
    }
  }
  Stream getStream(Device d) const noexcept override {
    try {
      const auto device_index =
          d.has_index() ? d.index() : native::vulkan::api::current_device();
      return Stream(Stream::DEFAULT, Device(DeviceType::Vulkan, device_index));
    } catch (...) {
      return Stream(Stream::DEFAULT, Device(DeviceType::Vulkan, -1));
    }
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const noexcept override {
    (void)s;
    return getStream(Device(DeviceType::Vulkan, -1));
  }
  DeviceIndex deviceCount() const noexcept override {
    try {
      return native::vulkan::api::device_count();
    } catch (...) {
      return 0;
    }
  }

  // Event-related functions
  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const EventFlag flag) const override {
    (void)event;
    (void)stream;
    (void)device_index;
    (void)flag;
    TORCH_CHECK(false, "VULKAN backend doesn't support events.");
  }
  void block(void* event, const Stream& stream) const override {
    (void)event;
    (void)stream;
    TORCH_CHECK(false, "VULKAN backend doesn't support events.")
  }
  bool queryEvent(void* event) const override {
    (void)event;
    TORCH_CHECK(false, "VULKAN backend doesn't support events.")
  }
  void destroyEvent(void* event, const DeviceIndex device_index)
      const noexcept override {
    (void)event;
    (void)device_index;
    // no-op
  }
};

} // namespace

C10_REGISTER_GUARD_IMPL(Vulkan, VulkanGuardImpl)

} // namespace at::detail
