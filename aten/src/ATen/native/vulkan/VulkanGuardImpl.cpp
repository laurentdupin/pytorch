#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/Diagnostics.h>
#include <ATen/native/vulkan/api/Event.h>
#include <ATen/native/vulkan/api/Stream.h>

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
      return native::vulkan::api::vulkan_stream_pool()
          .get_current_c10_stream(device_index);
    } catch (...) {
      return Stream(Stream::UNSAFE, Device(DeviceType::Vulkan, -1), 0);
    }
  }
  Stream getDefaultStream(Device d) const override {
    const auto device_index =
        d.has_index() ? d.index() : native::vulkan::api::current_device();
    return native::vulkan::api::vulkan_stream_pool().make_c10_stream(
        device_index,
        native::vulkan::api::vulkan_stream_pool()
            .get_default_stream(device_index)
            .id);
  }
  Stream getNewStream(Device d, int priority = 0) const override {
    (void)priority;
    const auto device_index =
        d.has_index() ? d.index() : native::vulkan::api::current_device();
    auto& stream =
        native::vulkan::api::vulkan_stream_pool().get_new_stream(device_index);
    return native::vulkan::api::vulkan_stream_pool().make_c10_stream(
        device_index, stream.id);
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const override {
    return native::vulkan::api::context(s.device_index())->exchange_stream(s);
  }
  bool queryStream(const Stream& stream) const override {
    return native::vulkan::api::context(stream.device_index())
        ->query_stream(stream);
  }
  void synchronizeStream(const Stream& stream) const override {
    native::vulkan::api::context(stream.device_index())
        ->synchronize_stream(stream);
  }
  void synchronizeDevice(const DeviceIndex device_index) const override {
    native::vulkan::api::context(device_index)->synchronize_device();
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
    (void)flag;
    if (*event == nullptr) {
      *event = new native::vulkan::api::VulkanEventState();
    }
    auto* state =
        static_cast<native::vulkan::api::VulkanEventState*>(*event);
    TORCH_CHECK(
        device_index < 0 || device_index == stream.device_index(),
        "Vulkan event device index ",
        device_index,
        " does not match stream device index ",
        stream.device_index());
    native::vulkan::api::record_vulkan_event(*state, stream);
  }
  void block(void* event, const Stream& stream) const override {
    if (event == nullptr) {
      return;
    }
    auto* state = static_cast<native::vulkan::api::VulkanEventState*>(event);
    native::vulkan::api::block_vulkan_event(*state, stream);
  }
  bool queryEvent(void* event) const override {
    if (event == nullptr) {
      return true;
    }
    auto* state = static_cast<native::vulkan::api::VulkanEventState*>(event);
    return native::vulkan::api::query_vulkan_event(*state);
  }
  void synchronizeEvent(void* event) const override {
    if (event == nullptr) {
      return;
    }
    auto* state = static_cast<native::vulkan::api::VulkanEventState*>(event);
    native::vulkan::api::synchronize_vulkan_event(*state);
  }
  void destroyEvent(void* event, const DeviceIndex device_index)
      const noexcept override {
    (void)device_index;
    delete static_cast<native::vulkan::api::VulkanEventState*>(event);
  }
};

} // namespace

C10_REGISTER_GUARD_IMPL(Vulkan, VulkanGuardImpl)

} // namespace at::detail
