#pragma once

#include <ATen/OpaqueTensorImpl.h>

namespace at {
// The only difference from OpaqueTensorImpl is faking strides(), stride(),
// is_contiguous(). The main intention for this is to be able to run torchscript
// model on Vulkan backend. Strides are not supported on Vulkan side, plan to
// support them.
template <typename OpaqueHandle>
struct VulkanOpaqueTensorImpl : public OpaqueTensorImpl<OpaqueHandle> {
  VulkanOpaqueTensorImpl(
      at::DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      c10::Device device,
      OpaqueHandle opaque_handle,
      c10::IntArrayRef sizes,
      c10::IntArrayRef strides,
      int64_t storage_offset = 0)
      : OpaqueTensorImpl<OpaqueHandle>(
            key_set,
            data_type,
            device,
            opaque_handle,
            sizes,
            false),
        strides_(strides.vec()),
        storage_offset_(storage_offset) {}

  IntArrayRef strides_custom() const override {
    return strides_;
  }

  SymIntArrayRef sym_strides_custom() const override {
    return c10::fromIntArrayRefKnownNonNegative(strides_);
  }

  int64_t storage_offset_custom() const override {
    return storage_offset_;
  }

  c10::SymInt sym_storage_offset_custom() const override {
    return c10::SymInt(storage_offset_);
  }

  c10::SymBool sym_is_contiguous_custom(
      c10::MemoryFormat memory_format) const override {
    (void)memory_format;
    return true;
  }

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<VulkanOpaqueTensorImpl<OpaqueHandle>>(
        this->key_set(),
        this->dtype(),
        this->device(),
        this->opaque_handle(),
        this->sizes(),
        strides_,
        storage_offset_);
    OpaqueTensorImpl<OpaqueHandle>::copy_tensor_metadata(
        this,
        impl.get(),
        version_counter,
        allow_tensor_metadata_change);
    impl->strides_ = strides_;
    impl->storage_offset_ = storage_offset_;
    impl->refresh_numel();
    return impl;
  }

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<VulkanOpaqueTensorImpl<OpaqueHandle>>(
        this->key_set(),
        this->dtype(),
        this->device(),
        this->opaque_handle(),
        this->sizes(),
        strides_,
        storage_offset_);
    OpaqueTensorImpl<OpaqueHandle>::copy_tensor_metadata(
        this,
        impl.get(),
        std::move(version_counter),
        allow_tensor_metadata_change);
    impl->strides_ = strides_;
    impl->storage_offset_ = storage_offset_;
    impl->refresh_numel();
    return impl;
  }

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override {
    AT_ASSERT(this->has_compatible_shallow_copy_type(impl->key_set()));
    auto opaque_impl =
        static_cast<const VulkanOpaqueTensorImpl<OpaqueHandle>*>(impl.get());
    OpaqueTensorImpl<OpaqueHandle>::copy_tensor_metadata(
        opaque_impl,
        this,
        this->version_counter(),
        this->allow_tensor_metadata_change());
    strides_ = opaque_impl->strides_;
    storage_offset_ = opaque_impl->storage_offset_;
    this->refresh_numel();
  }

 private:
  const char* tensorimpl_type_name() const override {
    return "VulkanOpaqueTensorImpl";
  }

  // TODO: storing strides separately is unnecessary, the base TensorImpl
  // has space for them
  SmallVector<int64_t, 5> strides_;
  int64_t storage_offset_;
};

} // namespace at
