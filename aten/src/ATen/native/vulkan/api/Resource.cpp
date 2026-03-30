#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <algorithm>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace {

struct AllocationTelemetryState final {
  std::mutex mutex;
  uint64_t live_bytes{0u};
  uint64_t high_water_bytes{0u};
  std::unordered_map<std::string, uint64_t> live_bytes_by_label;
  std::unordered_map<std::string, uint64_t> high_water_bytes_by_label;
  std::deque<std::string> recent_events;
};

std::string& mutable_current_allocation_label() {
  static thread_local std::string label = "unlabeled";
  return label;
}

AllocationTelemetryState& allocation_telemetry_state() {
  static AllocationTelemetryState state;
  return state;
}

const std::string& allocation_log_path() {
  static const std::string path = []() {
    const char* env = std::getenv("PYTORCH_VULKAN_ALLOC_LOG");
    return env ? std::string(env) : std::string();
  }();
  return path;
}

bool allocation_telemetry_enabled() {
  return !allocation_log_path().empty();
}

std::string format_bytes(const uint64_t bytes) {
  std::ostringstream stream;
  const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
  stream.setf(std::ios::fixed);
  stream.precision(2);
  stream << mib << " MiB";
  return stream.str();
}

const std::string& normalize_allocation_label(const std::string& label) {
  static const std::string unlabeled = "unlabeled";
  return label.empty() ? unlabeled : label;
}

void write_allocation_log_line_locked(
    AllocationTelemetryState& state,
    const std::string& line) {
  if (state.recent_events.size() >= 64u) {
    state.recent_events.pop_front();
  }
  state.recent_events.push_back(line);

  std::ofstream out(allocation_log_path(), std::ios::app);
  out << line << '\n';
}

std::string top_label_summary(
    const std::unordered_map<std::string, uint64_t>& live_bytes_by_label,
    const size_t limit = 4u) {
  if (live_bytes_by_label.empty()) {
    return {};
  }

  std::vector<std::pair<std::string, uint64_t>> entries(
      live_bytes_by_label.begin(), live_bytes_by_label.end());
  std::sort(
      entries.begin(),
      entries.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

  std::ostringstream stream;
  const size_t count = std::min(limit, entries.size());
  for (size_t idx = 0u; idx < count; ++idx) {
    if (idx > 0u) {
      stream << ", ";
    }
    stream << entries[idx].first << ":" << format_bytes(entries[idx].second);
  }
  return stream.str();
}

uint32_t format_element_size(const VkFormat format) {
  switch (format) {
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R32G32B32A32_SINT:
    case VK_FORMAT_R32G32B32A32_UINT:
      return 16u;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return 8u;
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R8G8B8A8_UINT:
      return 4u;
    default:
      return 0u;
  }
}

uint64_t estimate_image_bytes(const VulkanImage::ImageProperties& image_props) {
  const uint32_t element_size = format_element_size(image_props.image_format);
  return static_cast<uint64_t>(image_props.image_extents.width) *
      static_cast<uint64_t>(image_props.image_extents.height) *
      static_cast<uint64_t>(image_props.image_extents.depth) *
      static_cast<uint64_t>(element_size);
}

std::string heap_budgets_summary(const VmaAllocator allocator) {
  VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
  vmaGetHeapBudgets(allocator, budgets);

  const VkPhysicalDeviceMemoryProperties* mem_props = nullptr;
  vmaGetMemoryProperties(
      allocator,
      reinterpret_cast<const VkPhysicalDeviceMemoryProperties**>(&mem_props));

  std::ostringstream stream;
  for (uint32_t idx = 0; idx < mem_props->memoryHeapCount; ++idx) {
    if (idx > 0u) {
      stream << ", ";
    }
    stream << "heap" << idx << ": usage="
           << format_bytes(static_cast<uint64_t>(budgets[idx].usage))
           << " budget="
           << format_bytes(static_cast<uint64_t>(budgets[idx].budget));
  }
  return stream.str();
}

VkDevice allocator_device(const VmaAllocator allocator) {
  VmaAllocatorInfo allocator_info{};
  vmaGetAllocatorInfo(allocator, &allocator_info);
  return allocator_info.device;
}

std::string format_memory_type_bits(const uint32_t memory_type_bits) {
  std::ostringstream stream;
  stream << "0x" << std::hex << memory_type_bits;
  return stream.str();
}

std::string memory_requirements_summary(
    const VmaAllocator allocator,
    const VkMemoryRequirements& memory_requirements) {
  VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
  vmaGetHeapBudgets(allocator, budgets);

  const VkPhysicalDeviceMemoryProperties* mem_props = nullptr;
  vmaGetMemoryProperties(
      allocator,
      reinterpret_cast<const VkPhysicalDeviceMemoryProperties**>(&mem_props));

  std::ostringstream heaps_stream;
  VkDeviceSize best_available = 0u;
  bool found_compatible_heap = false;
  bool first_heap = true;

  for (uint32_t type_index = 0u; type_index < mem_props->memoryTypeCount;
       ++type_index) {
    if ((memory_requirements.memoryTypeBits & (1u << type_index)) == 0u) {
      continue;
    }

    const uint32_t heap_index = mem_props->memoryTypes[type_index].heapIndex;
    const VkDeviceSize usage = budgets[heap_index].usage;
    const VkDeviceSize budget = budgets[heap_index].budget;
    const VkDeviceSize available = budget > usage ? (budget - usage) : 0u;

    best_available = std::max(best_available, available);
    found_compatible_heap = true;

    if (!first_heap) {
      heaps_stream << "; ";
    }
    first_heap = false;
    heaps_stream << "heap" << heap_index << "(type=" << type_index
                 << ", avail=" << format_bytes(static_cast<uint64_t>(available))
                 << ", usage=" << format_bytes(static_cast<uint64_t>(usage))
                 << ", budget=" << format_bytes(static_cast<uint64_t>(budget))
                 << ")";
  }

  const int64_t deficit_vs_required =
      static_cast<int64_t>(memory_requirements.size) -
      static_cast<int64_t>(best_available);

  std::ostringstream stream;
  stream << " required_allocation="
         << format_bytes(static_cast<uint64_t>(memory_requirements.size))
         << " alignment=" << memory_requirements.alignment
         << " memoryTypeBits="
         << format_memory_type_bits(memory_requirements.memoryTypeBits)
         << " compatible_heaps={";
  if (found_compatible_heap) {
    stream << heaps_stream.str();
  }
  stream << "} best_compatible_available="
         << format_bytes(static_cast<uint64_t>(best_available))
         << " deficit_vs_required="
         << format_bytes(
                static_cast<uint64_t>(
                    deficit_vs_required >= 0 ? deficit_vs_required : 0));
  if (deficit_vs_required < 0) {
    stream << " headroom_vs_required="
           << format_bytes(static_cast<uint64_t>(-deficit_vs_required));
  }
  return stream.str();
}

std::string buffer_failure_requirements_details(
    const VmaAllocator allocator,
    const VkBufferCreateInfo& buffer_create_info) {
  const VkDevice device = allocator_device(allocator);
  VkBuffer buffer = VK_NULL_HANDLE;
  const VkResult create_result =
      vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer);
  if (VK_SUCCESS != create_result || VK_NULL_HANDLE == buffer) {
    std::ostringstream stream;
    stream << " preflight_buffer_create_result=" << create_result;
    return stream.str();
  }

  VkMemoryRequirements memory_requirements{};
  vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);
  vkDestroyBuffer(device, buffer, nullptr);
  return memory_requirements_summary(allocator, memory_requirements);
}

std::string image_failure_requirements_details(
    const VmaAllocator allocator,
    const VkImageCreateInfo& image_create_info) {
  const VkDevice device = allocator_device(allocator);
  VkImage image = VK_NULL_HANDLE;
  const VkResult create_result =
      vkCreateImage(device, &image_create_info, nullptr, &image);
  if (VK_SUCCESS != create_result || VK_NULL_HANDLE == image) {
    std::ostringstream stream;
    stream << " preflight_image_create_result=" << create_result;
    return stream.str();
  }

  VkMemoryRequirements memory_requirements{};
  vkGetImageMemoryRequirements(device, image, &memory_requirements);
  vkDestroyImage(device, image, nullptr);
  return memory_requirements_summary(allocator, memory_requirements);
}

void log_allocation_success(
    const char* kind,
    const uint64_t requested_bytes,
    const uint64_t actual_bytes,
    const std::string& details,
    const std::string& label) {
  if (!allocation_telemetry_enabled()) {
    return;
  }

  AllocationTelemetryState& state = allocation_telemetry_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  const std::string& normalized_label = normalize_allocation_label(label);

  state.live_bytes += actual_bytes;
  state.high_water_bytes = std::max(state.high_water_bytes, state.live_bytes);
  state.live_bytes_by_label[normalized_label] += actual_bytes;
  state.high_water_bytes_by_label[normalized_label] = std::max(
      state.high_water_bytes_by_label[normalized_label],
      state.live_bytes_by_label[normalized_label]);

  std::ostringstream stream;
  stream << "[alloc] kind=" << kind << " label=" << normalized_label
         << " requested="
         << format_bytes(requested_bytes) << " actual="
         << format_bytes(actual_bytes) << " live="
         << format_bytes(state.live_bytes) << " high_water="
         << format_bytes(state.high_water_bytes) << " label_live="
         << format_bytes(state.live_bytes_by_label[normalized_label])
         << " label_high_water="
         << format_bytes(state.high_water_bytes_by_label[normalized_label]);
  if (!details.empty()) {
    stream << " " << details;
  }

  write_allocation_log_line_locked(state, stream.str());
}

void log_allocation_free(
    const char* kind,
    const uint64_t actual_bytes,
    const std::string& details,
    const std::string& label) {
  if (!allocation_telemetry_enabled()) {
    return;
  }

  AllocationTelemetryState& state = allocation_telemetry_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  const std::string& normalized_label = normalize_allocation_label(label);

  state.live_bytes = state.live_bytes > actual_bytes
      ? (state.live_bytes - actual_bytes)
      : 0u;
  auto label_it = state.live_bytes_by_label.find(normalized_label);
  if (label_it != state.live_bytes_by_label.end()) {
    label_it->second = label_it->second > actual_bytes
        ? (label_it->second - actual_bytes)
        : 0u;
  }

  std::ostringstream stream;
  stream << "[free] kind=" << kind << " label=" << normalized_label
         << " actual=" << format_bytes(actual_bytes) << " live="
         << format_bytes(state.live_bytes) << " high_water="
         << format_bytes(state.high_water_bytes) << " label_live="
         << format_bytes(
                label_it != state.live_bytes_by_label.end() ? label_it->second
                                                            : 0u)
         << " label_high_water="
         << format_bytes(state.high_water_bytes_by_label[normalized_label]);
  if (!details.empty()) {
    stream << " " << details;
  }

  write_allocation_log_line_locked(state, stream.str());
}

void log_allocation_failure(
    const char* kind,
    const VkResult result,
    const uint64_t requested_bytes,
    const std::string& details,
    const VmaAllocator allocator,
    const std::string& label) {
  if (!allocation_telemetry_enabled()) {
    return;
  }

  AllocationTelemetryState& state = allocation_telemetry_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  const std::string& normalized_label = normalize_allocation_label(label);

  std::ostringstream stream;
  stream << "[alloc-fail] kind=" << kind << " label=" << normalized_label
         << " result=" << result
         << " requested=" << format_bytes(requested_bytes) << " live="
         << format_bytes(state.live_bytes) << " high_water="
         << format_bytes(state.high_water_bytes);
  if (!details.empty()) {
    stream << " " << details;
  }
  stream << " budgets={" << heap_budgets_summary(allocator) << "}";
  const std::string top_labels = top_label_summary(state.live_bytes_by_label);
  if (!top_labels.empty()) {
    stream << " top_live_labels={" << top_labels << "}";
  }

  write_allocation_log_line_locked(state, stream.str());

  const size_t recent_count = std::min<size_t>(state.recent_events.size(), 12u);
  if (recent_count == 0u) {
    return;
  }

  std::ofstream out(allocation_log_path(), std::ios::app);
  out << "[alloc-fail] recent-events-begin\n";
  for (size_t idx = state.recent_events.size() - recent_count;
       idx < state.recent_events.size();
       ++idx) {
    out << state.recent_events[idx] << '\n';
  }
  out << "[alloc-fail] recent-events-end\n";
}

VkDeviceSize query_allocation_size(
    const VmaAllocator allocator,
    const VmaAllocation allocation) {
  if (VK_NULL_HANDLE == allocation) {
    return 0u;
  }
  VmaAllocationInfo allocation_info{};
  vmaGetAllocationInfo(allocator, allocation, &allocation_info);
  return allocation_info.size;
}

std::string buffer_details(
    const VkDeviceSize size,
    const VkBufferUsageFlags usage) {
  std::ostringstream stream;
  stream << "size=" << size << " usage=0x" << std::hex << usage;
  return stream.str();
}

std::string image_details(const VulkanImage::ImageProperties& image_props) {
  std::ostringstream stream;
  stream << "extents=(" << image_props.image_extents.width << ","
         << image_props.image_extents.height << ","
         << image_props.image_extents.depth << ") format="
         << image_props.image_format << " usage=0x" << std::hex
         << image_props.image_usage;
  return stream.str();
}

} // namespace

const std::string& current_allocation_label() {
  return mutable_current_allocation_label();
}

AllocationScope::AllocationScope(const char* label)
    : previous_(current_allocation_label()) {
  mutable_current_allocation_label() =
      (label && label[0] != '\0') ? std::string(label) : std::string("unlabeled");
}

AllocationScope::AllocationScope(const std::string& label)
    : previous_(current_allocation_label()) {
  mutable_current_allocation_label() =
      label.empty() ? std::string("unlabeled") : label;
}

AllocationScope::~AllocationScope() {
  mutable_current_allocation_label() = previous_;
}

//
// MemoryBarrier
//

MemoryBarrier::MemoryBarrier(
    const VkAccessFlags src_access_flags,
    const VkAccessFlags dst_access_flags)
    : handle{
          VK_STRUCTURE_TYPE_MEMORY_BARRIER, // sType
          nullptr, // pNext
          src_access_flags, // srcAccessMask
          dst_access_flags, // dstAccessMask
      } {}

//
// MemoryAllocation
//

MemoryAllocation::MemoryAllocation()
    : memory_requirements{},
      create_info{},
      allocator(VK_NULL_HANDLE),
      allocation(VK_NULL_HANDLE) {}

MemoryAllocation::MemoryAllocation(
    VmaAllocator vma_allocator,
    const VkMemoryRequirements& mem_props,
    const VmaAllocationCreateInfo& create_info)
    : memory_requirements(mem_props),
      create_info(create_info),
      allocator(vma_allocator),
      allocation(VK_NULL_HANDLE) {
  VK_CHECK(vmaAllocateMemory(
      allocator, &memory_requirements, &create_info, &allocation, nullptr));
}

MemoryAllocation::MemoryAllocation(MemoryAllocation&& other) noexcept
    : memory_requirements(other.memory_requirements),
      create_info(other.create_info),
      allocator(other.allocator),
      allocation(other.allocation) {
  other.allocation = VK_NULL_HANDLE;
}

MemoryAllocation& MemoryAllocation::operator=(
    MemoryAllocation&& other) noexcept {
  VmaAllocation tmp_allocation = allocation;

  memory_requirements = other.memory_requirements;
  create_info = other.create_info;
  allocator = other.allocator;
  allocation = other.allocation;

  other.allocation = tmp_allocation;

  return *this;
}

MemoryAllocation::~MemoryAllocation() {
  if (VK_NULL_HANDLE != allocation) {
    vmaFreeMemory(allocator, allocation);
  }
}

//
// VulkanBuffer
//

VulkanBuffer::VulkanBuffer()
    : buffer_properties_{},
      allocator_(VK_NULL_HANDLE),
      memory_{},
      allocated_size_(0u),
      allocation_label_(),
      owns_memory_(false),
      handle_(VK_NULL_HANDLE) {}

VulkanBuffer::VulkanBuffer(
    VmaAllocator vma_allocator,
    const VkDeviceSize size,
    const VmaAllocationCreateInfo& allocation_create_info,
    const VkBufferUsageFlags usage,
    const bool allocate_memory)
    : buffer_properties_({
          size,
          0u,
          size,
          usage,
      }),
      allocator_(vma_allocator),
      memory_{},
      allocated_size_(0u),
      allocation_label_(current_allocation_label()),
      owns_memory_(allocate_memory),
      handle_(VK_NULL_HANDLE) {
  // Only allocate memory if the buffer has non-zero size
  if (size == 0) {
    return;
  }

  const VkBufferCreateInfo buffer_create_info{
      VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      size, // size
      buffer_properties_.buffer_usage, // usage
      VK_SHARING_MODE_EXCLUSIVE, // sharingMode
      0u, // queueFamilyIndexCount
      nullptr, // pQueueFamilyIndices
  };

  memory_.create_info = allocation_create_info;

  if (allocate_memory) {
    const VkResult create_result = vmaCreateBuffer(
        allocator_,
        &buffer_create_info,
        &allocation_create_info,
        &handle_,
        &(memory_.allocation),
        nullptr);
    if (VK_SUCCESS != create_result) {
      const std::string failure_details =
          buffer_details(size, buffer_properties_.buffer_usage) +
          buffer_failure_requirements_details(allocator_, buffer_create_info);
      log_allocation_failure(
          "buffer",
          create_result,
          static_cast<uint64_t>(size),
          failure_details,
          allocator_,
          allocation_label_);
    }
    VK_CHECK(create_result);
    allocated_size_ = query_allocation_size(allocator_, memory_.allocation);
    log_allocation_success(
        "buffer",
        static_cast<uint64_t>(size),
        static_cast<uint64_t>(allocated_size_),
        buffer_details(size, buffer_properties_.buffer_usage),
        allocation_label_);
  } else {
    VmaAllocatorInfo allocator_info{};
    vmaGetAllocatorInfo(allocator_, &allocator_info);
    VK_CHECK(vkCreateBuffer(
        allocator_info.device, &buffer_create_info, nullptr, &handle_));
  }
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept
    : buffer_properties_(other.buffer_properties_),
      allocator_(other.allocator_),
      memory_(std::move(other.memory_)),
      allocated_size_(other.allocated_size_),
      allocation_label_(std::move(other.allocation_label_)),
      owns_memory_(other.owns_memory_),
      handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
  other.allocated_size_ = 0u;
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept {
  VkBuffer tmp_buffer = handle_;
  bool tmp_owns_memory = owns_memory_;
  VkDeviceSize tmp_allocated_size = allocated_size_;
  std::string tmp_allocation_label = std::move(allocation_label_);

  buffer_properties_ = other.buffer_properties_;
  allocator_ = other.allocator_;
  memory_ = std::move(other.memory_);
  allocated_size_ = other.allocated_size_;
  allocation_label_ = std::move(other.allocation_label_);
  owns_memory_ = other.owns_memory_;
  handle_ = other.handle_;

  other.handle_ = tmp_buffer;
  other.owns_memory_ = tmp_owns_memory;
  other.allocated_size_ = tmp_allocated_size;
  other.allocation_label_ = std::move(tmp_allocation_label);

  return *this;
}

VulkanBuffer::~VulkanBuffer() {
  if (VK_NULL_HANDLE != handle_) {
    if (owns_memory_ && allocated_size_ > 0u) {
      log_allocation_free(
          "buffer",
          static_cast<uint64_t>(allocated_size_),
          buffer_details(buffer_properties_.size, buffer_properties_.buffer_usage),
          allocation_label_);
    }
    if (owns_memory_) {
      vmaDestroyBuffer(allocator_, handle_, memory_.allocation);
    } else {
      vkDestroyBuffer(this->device(), handle_, nullptr);
    }
    // Prevent the underlying memory allocation from being freed; it was either
    // freed by vmaDestroyBuffer, or this resource does not own the underlying
    // memory
    memory_.allocation = VK_NULL_HANDLE;
    allocated_size_ = 0u;
  }
}

VkMemoryRequirements VulkanBuffer::get_memory_requirements() const {
  VkMemoryRequirements memory_requirements;
  vkGetBufferMemoryRequirements(this->device(), handle_, &memory_requirements);
  return memory_requirements;
}

//
// MemoryMap
//

MemoryMap::MemoryMap(const VulkanBuffer& buffer, const uint8_t access)
    : access_(access),
      allocator_(buffer.vma_allocator()),
      allocation_(buffer.allocation()),
      data_(nullptr),
      data_len_{buffer.mem_size()} {
  if (allocation_) {
    VK_CHECK(vmaMapMemory(allocator_, allocation_, &data_));
  }
}

MemoryMap::MemoryMap(MemoryMap&& other) noexcept
    : access_(other.access_),
      allocator_(other.allocator_),
      allocation_(other.allocation_),
      data_(other.data_),
      data_len_{other.data_len_} {
  other.allocation_ = VK_NULL_HANDLE;
  other.data_ = nullptr;
}

MemoryMap::~MemoryMap() {
  if (!data_) {
    return;
  }

  if (allocation_) {
    if (access_ & MemoryAccessType::WRITE) {
      // Call will be ignored by implementation if the memory type this
      // allocation belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is
      // the behavior we want. Don't check the result here as the destructor
      // cannot throw.
      vmaFlushAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE);
    }

    vmaUnmapMemory(allocator_, allocation_);
  }
}

void MemoryMap::invalidate() {
  if (access_ & MemoryAccessType::READ && allocation_) {
    // Call will be ignored by implementation if the memory type this allocation
    // belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is the behavior
    // we want.
    VK_CHECK(
        vmaInvalidateAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE));
  }
}

//
// BufferMemoryBarrier
//

BufferMemoryBarrier::BufferMemoryBarrier(
    const VkAccessFlags src_access_flags,
    const VkAccessFlags dst_access_flags,
    const VulkanBuffer& buffer)
    : handle{
          VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, // sType
          nullptr, // pNext
          src_access_flags, // srcAccessMask
          dst_access_flags, // dstAccessMask
          VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
          VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
          buffer.handle_, // buffer
          buffer.buffer_properties_.mem_offset, // offset
          buffer.buffer_properties_.mem_range, // size
      } {}

//
// ImageSampler
//

static bool operator==(
    const ImageSampler::Properties& _1,
    const ImageSampler::Properties& _2) {
  return (
      _1.filter == _2.filter && _1.mipmap_mode == _2.mipmap_mode &&
      _1.address_mode == _2.address_mode && _1.border_color == _2.border_color);
}

ImageSampler::ImageSampler(
    VkDevice device,
    const ImageSampler::Properties& props)
    : device_(device), handle_(VK_NULL_HANDLE) {
  const VkSamplerCreateInfo sampler_create_info{
      VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      props.filter, // magFilter
      props.filter, // minFilter
      props.mipmap_mode, // mipmapMode
      props.address_mode, // addressModeU
      props.address_mode, // addressModeV
      props.address_mode, // addressModeW
      0.0f, // mipLodBias
      VK_FALSE, // anisotropyEnable
      1.0f, // maxAnisotropy,
      VK_FALSE, // compareEnable
      VK_COMPARE_OP_NEVER, // compareOp
      0.0f, // minLod
      VK_LOD_CLAMP_NONE, // maxLod
      props.border_color, // borderColor
      VK_FALSE, // unnormalizedCoordinates
  };

  VK_CHECK(vkCreateSampler(device_, &sampler_create_info, nullptr, &handle_));
}

ImageSampler::ImageSampler(ImageSampler&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

ImageSampler::~ImageSampler() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroySampler(device_, handle_, nullptr);
}

size_t ImageSampler::Hasher::operator()(
    const ImageSampler::Properties& props) const {
  size_t seed = 0;
  seed = utils::hash_combine(seed, std::hash<VkFilter>()(props.filter));
  seed = utils::hash_combine(
      seed, std::hash<VkSamplerMipmapMode>()(props.mipmap_mode));
  seed = utils::hash_combine(
      seed, std::hash<VkSamplerAddressMode>()(props.address_mode));
  seed =
      utils::hash_combine(seed, std::hash<VkBorderColor>()(props.border_color));
  return seed;
}

void swap(ImageSampler& lhs, ImageSampler& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkSampler tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

//
// VulkanImage
//

VulkanImage::VulkanImage()
    : image_properties_{},
      view_properties_{},
      sampler_properties_{},
      allocator_(VK_NULL_HANDLE),
      memory_{},
      allocated_size_(0u),
      owns_memory_(false),
      handles_{
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
      },
      layout_{} {}

VulkanImage::VulkanImage(
    VmaAllocator vma_allocator,
    const VmaAllocationCreateInfo& allocation_create_info,
    const ImageProperties& image_props,
    const ViewProperties& view_props,
    const SamplerProperties& sampler_props,
    const VkImageLayout layout,
    VkSampler sampler,
    const bool allocate_memory)
    : image_properties_(image_props),
      view_properties_(view_props),
      sampler_properties_(sampler_props),
      allocator_(vma_allocator),
      memory_{},
      allocated_size_(0u),
      allocation_label_(current_allocation_label()),
      owns_memory_{allocate_memory},
      handles_{
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
          sampler,
      },
      layout_(layout) {
  VmaAllocatorInfo allocator_info{};
  vmaGetAllocatorInfo(allocator_, &allocator_info);

  // If any dims are zero, then no memory will be allocated for the image.
  if (image_props.image_extents.width == 0 ||
      image_props.image_extents.height == 0 ||
      image_props.image_extents.depth == 0) {
    return;
  }

  const VkImageCreateInfo image_create_info{
      VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      image_properties_.image_type, // imageType
      image_properties_.image_format, // format
      image_properties_.image_extents, // extents
      1u, // mipLevels
      1u, // arrayLayers
      VK_SAMPLE_COUNT_1_BIT, // samples
      VK_IMAGE_TILING_OPTIMAL, // tiling
      image_properties_.image_usage, // usage
      VK_SHARING_MODE_EXCLUSIVE, // sharingMode
      0u, // queueFamilyIndexCount
      nullptr, // pQueueFamilyIndices
      layout_, // initialLayout
  };

  memory_.create_info = allocation_create_info;

  if (allocate_memory) {
    const VkResult create_result = vmaCreateImage(
        allocator_,
        &image_create_info,
        &allocation_create_info,
        &(handles_.image),
        &(memory_.allocation),
        nullptr);
    if (VK_SUCCESS != create_result) {
      const std::string failure_details =
          image_details(image_properties_) +
          image_failure_requirements_details(allocator_, image_create_info);
      log_allocation_failure(
          "image",
          create_result,
          estimate_image_bytes(image_properties_),
          failure_details,
          allocator_,
          allocation_label_);
    }
    VK_CHECK(create_result);
    allocated_size_ = query_allocation_size(allocator_, memory_.allocation);
    log_allocation_success(
        "image",
        estimate_image_bytes(image_properties_),
        static_cast<uint64_t>(allocated_size_),
        image_details(image_properties_),
        allocation_label_);
    // Only create the image view if the image has been bound to memory
    create_image_view();
  } else {
    VK_CHECK(vkCreateImage(
        allocator_info.device, &image_create_info, nullptr, &(handles_.image)));
  }
}

VulkanImage::VulkanImage(VulkanImage&& other) noexcept
    : image_properties_(other.image_properties_),
      view_properties_(other.view_properties_),
      sampler_properties_(other.sampler_properties_),
      allocator_(other.allocator_),
      memory_(std::move(other.memory_)),
      allocated_size_(other.allocated_size_),
      allocation_label_(std::move(other.allocation_label_)),
      owns_memory_(other.owns_memory_),
      handles_(other.handles_),
      layout_(other.layout_) {
  other.handles_.image = VK_NULL_HANDLE;
  other.handles_.image_view = VK_NULL_HANDLE;
  other.handles_.sampler = VK_NULL_HANDLE;
  other.owns_memory_ = false;
  other.allocated_size_ = 0u;
}

VulkanImage& VulkanImage::operator=(VulkanImage&& other) noexcept {
  VkImage tmp_image = handles_.image;
  VkImageView tmp_image_view = handles_.image_view;
  bool tmp_owns_memory = owns_memory_;
  VkDeviceSize tmp_allocated_size = allocated_size_;
  std::string tmp_allocation_label = std::move(allocation_label_);

  image_properties_ = other.image_properties_;
  view_properties_ = other.view_properties_;
  sampler_properties_ = other.sampler_properties_;
  allocator_ = other.allocator_;
  memory_ = std::move(other.memory_);
  allocated_size_ = other.allocated_size_;
  allocation_label_ = std::move(other.allocation_label_);
  owns_memory_ = other.owns_memory_;
  handles_ = other.handles_;
  layout_ = other.layout_;

  other.handles_.image = tmp_image;
  other.handles_.image_view = tmp_image_view;
  other.owns_memory_ = tmp_owns_memory;
  other.allocated_size_ = tmp_allocated_size;
  other.allocation_label_ = std::move(tmp_allocation_label);

  return *this;
}

VulkanImage::~VulkanImage() {
  if (VK_NULL_HANDLE != handles_.image_view) {
    vkDestroyImageView(this->device(), handles_.image_view, nullptr);
  }

  if (VK_NULL_HANDLE != handles_.image) {
    if (owns_memory_ && allocated_size_ > 0u) {
      log_allocation_free(
          "image",
          static_cast<uint64_t>(allocated_size_),
          image_details(image_properties_),
          allocation_label_);
    }
    if (owns_memory_) {
      vmaDestroyImage(allocator_, handles_.image, memory_.allocation);
    } else {
      vkDestroyImage(this->device(), handles_.image, nullptr);
    }
    // Prevent the underlying memory allocation from being freed; it was either
    // freed by vmaDestroyImage, or this resource does not own the underlying
    // memory
    memory_.allocation = VK_NULL_HANDLE;
    allocated_size_ = 0u;
  }
}

void VulkanImage::create_image_view() {
  VmaAllocatorInfo allocator_info{};
  vmaGetAllocatorInfo(allocator_, &allocator_info);

  const VkComponentMapping component_mapping{
      VK_COMPONENT_SWIZZLE_IDENTITY, // r
      VK_COMPONENT_SWIZZLE_IDENTITY, // g
      VK_COMPONENT_SWIZZLE_IDENTITY, // b
      VK_COMPONENT_SWIZZLE_IDENTITY, // a
  };

  const VkImageSubresourceRange subresource_range{
      VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
      0u, // baseMipLevel
      VK_REMAINING_MIP_LEVELS, // levelCount
      0u, // baseArrayLayer
      VK_REMAINING_ARRAY_LAYERS, // layerCount
  };

  const VkImageViewCreateInfo image_view_create_info{
      VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      handles_.image, // image
      view_properties_.view_type, // viewType
      view_properties_.view_format, // format
      component_mapping, // components
      subresource_range, // subresourceRange
  };

  VK_CHECK(vkCreateImageView(
      allocator_info.device,
      &(image_view_create_info),
      nullptr,
      &(handles_.image_view)));
}

VkMemoryRequirements VulkanImage::get_memory_requirements() const {
  VkMemoryRequirements memory_requirements;
  vkGetImageMemoryRequirements(
      this->device(), handles_.image, &memory_requirements);
  return memory_requirements;
}

//
// ImageMemoryBarrier
//

ImageMemoryBarrier::ImageMemoryBarrier(
    const VkAccessFlags src_access_flags,
    const VkAccessFlags dst_access_flags,
    const VkImageLayout src_layout_flags,
    const VkImageLayout dst_layout_flags,
    const VulkanImage& image)
    : handle{
          VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, // sType
          nullptr, // pNext
          src_access_flags, // srcAccessMask
          dst_access_flags, // dstAccessMask
          src_layout_flags, // oldLayout
          dst_layout_flags, // newLayout
          VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
          VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
          image.handles_.image, // image
          {
              // subresourceRange
              VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
              0u, // baseMipLevel
              VK_REMAINING_MIP_LEVELS, // levelCount
              0u, // baseArrayLayer
              VK_REMAINING_ARRAY_LAYERS, // layerCount
          },
      } {}

//
// SamplerCache
//

SamplerCache::SamplerCache(VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

SamplerCache::SamplerCache(SamplerCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_), cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);
}

SamplerCache::~SamplerCache() {
  purge();
}

VkSampler SamplerCache::retrieve(const SamplerCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if (cache_.cend() == it) {
    it = cache_.insert({key, SamplerCache::Value(device_, key)}).first;
  }

  return it->second.handle();
}

void SamplerCache::purge() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_.clear();
}

//
// MemoryAllocator
//

MemoryAllocator::MemoryAllocator(
    VkInstance instance,
    VkPhysicalDevice physical_device,
    VkDevice device)
    : instance_{},
      physical_device_(physical_device),
      device_(device),
      allocator_{VK_NULL_HANDLE} {
  VmaVulkanFunctions vk_functions{};
  vk_functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  vk_functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

  const VmaAllocatorCreateInfo allocator_create_info{
      0u, // flags
      physical_device_, // physicalDevice
      device_, // device
      0u, // preferredLargeHeapBlockSize
      nullptr, // pAllocationCallbacks
      nullptr, // pDeviceMemoryCallbacks
      nullptr, // pHeapSizeLimit
      &vk_functions, // pVulkanFunctions
      instance, // instance
      VK_API_VERSION_1_0, // vulkanApiVersion
      nullptr, // pTypeExternalMemoryHandleTypes
  };

  VK_CHECK(vmaCreateAllocator(&allocator_create_info, &allocator_));
}

MemoryAllocator::MemoryAllocator(MemoryAllocator&& other) noexcept
    : instance_(other.instance_),
      physical_device_(other.physical_device_),
      device_(other.device_),
      allocator_(other.allocator_) {
  other.allocator_ = VK_NULL_HANDLE;
  other.device_ = VK_NULL_HANDLE;
  other.physical_device_ = VK_NULL_HANDLE;
  other.instance_ = VK_NULL_HANDLE;
}

MemoryAllocator::~MemoryAllocator() {
  if (VK_NULL_HANDLE == allocator_) {
    return;
  }
  vmaDestroyAllocator(allocator_);
}

MemoryAllocation MemoryAllocator::create_allocation(
    const VkMemoryRequirements& memory_requirements,
    const VmaAllocationCreateInfo& create_info) {
  VmaAllocationCreateInfo alloc_create_info = create_info;
  // Protect against using VMA_MEMORY_USAGE_AUTO_* flags when allocating memory
  // directly, since those usage flags require that VkBufferCreateInfo and/or
  // VkImageCreateInfo also be available.
  switch (create_info.usage) {
    // The logic for the below usage options are too complex, therefore prevent
    // those from being used with direct memory allocation.
    case VMA_MEMORY_USAGE_AUTO:
    case VMA_MEMORY_USAGE_AUTO_PREFER_HOST:
      VK_THROW(
          "Only the VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE usage flag is compatible with create_allocation()");
      break;
    // Most of the time, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE will simply set the
    // DEVICE_LOCAL_BIT as a preferred memory flag. Therefore the below is a
    // decent approximation for VMA behaviour.
    case VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE:
      alloc_create_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      alloc_create_info.usage = VMA_MEMORY_USAGE_UNKNOWN;
      break;
    default:
      break;
  }

  return MemoryAllocation(allocator_, memory_requirements, alloc_create_info);
}

VulkanImage MemoryAllocator::create_image(
    const VkExtent3D& extents,
    const VkFormat image_format,
    const VkImageType image_type,
    const VkImageViewType image_view_type,
    const VulkanImage::SamplerProperties& sampler_props,
    VkSampler sampler,
    const bool allow_transfer,
    const bool allocate_memory) {
  VkImageUsageFlags usage =
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  if (allow_transfer) {
    usage |=
        (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  }

  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  const VulkanImage::ImageProperties image_props{
      image_type,
      image_format,
      extents,
      usage,
  };

  const VulkanImage::ViewProperties view_props{
      image_view_type,
      image_format,
  };

  const VkImageLayout initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;

  return VulkanImage(
      allocator_,
      alloc_create_info,
      image_props,
      view_props,
      sampler_props,
      initial_layout,
      sampler,
      allocate_memory);
}

VulkanBuffer MemoryAllocator::create_storage_buffer(
    const VkDeviceSize size,
    const bool gpu_only,
    const bool allocate_memory) {
  const VkBufferUsageFlags buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  // The create storage buffer will be accessed by both the CPU and GPU, so set
  // the appropriate flags to indicate that the host device will be accessing
  // the data from this buffer.
  if (!gpu_only) {
    // Deferred memory allocation should only be used for GPU only buffers.
    VK_CHECK_COND(
        allocate_memory,
        "Only GPU-only buffers should use deferred memory allocation");

    alloc_create_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    alloc_create_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    alloc_create_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }

  return VulkanBuffer(
      allocator_, size, alloc_create_info, buffer_usage, allocate_memory);
}

VulkanBuffer MemoryAllocator::create_staging_buffer(const VkDeviceSize size) {
  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;

  VkBufferUsageFlags buffer_usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  return VulkanBuffer(allocator_, size, alloc_create_info, buffer_usage);
}

VulkanBuffer MemoryAllocator::create_uniform_buffer(const VkDeviceSize size) {
  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY |
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;

  VkBufferUsageFlags buffer_usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

  VulkanBuffer uniform_buffer(
      allocator_, size, alloc_create_info, buffer_usage);
  return uniform_buffer;
}

//
// VulkanFence
//

VulkanFence::VulkanFence()
    : device_(VK_NULL_HANDLE), handle_(VK_NULL_HANDLE), waiting_(false) {}

VulkanFence::VulkanFence(VkDevice device)
    : device_(device), handle_(VK_NULL_HANDLE), waiting_(VK_NULL_HANDLE) {
  const VkFenceCreateInfo fence_create_info{
      VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
  };

  VK_CHECK(vkCreateFence(device_, &fence_create_info, nullptr, &handle_));
}

VulkanFence::VulkanFence(VulkanFence&& other) noexcept
    : device_(other.device_), handle_(other.handle_), waiting_(other.waiting_) {
  other.handle_ = VK_NULL_HANDLE;
  other.waiting_ = false;
}

VulkanFence& VulkanFence::operator=(VulkanFence&& other) noexcept {
  device_ = other.device_;
  handle_ = other.handle_;
  waiting_ = other.waiting_;

  other.device_ = VK_NULL_HANDLE;
  other.handle_ = VK_NULL_HANDLE;
  other.waiting_ = false;

  return *this;
}

VulkanFence::~VulkanFence() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyFence(device_, handle_, nullptr);
}

void VulkanFence::wait() {
  // if get_submit_handle() has not been called, then this will no-op
  if (waiting_) {
    VkResult fence_status = VK_NOT_READY;
    // Run the wait in a loop to keep the CPU hot. A single call to
    // vkWaitForFences with no timeout may cause the calling thread to be
    // scheduled out.
    do {
      // The timeout (last) arg is in units of ns
      fence_status = vkWaitForFences(device_, 1u, &handle_, VK_TRUE, 100000);

      VK_CHECK_COND(
          fence_status != VK_ERROR_DEVICE_LOST,
          "Vulkan Fence: Device lost while waiting for fence!");
    } while (fence_status != VK_SUCCESS);

    VK_CHECK(vkResetFences(device_, 1u, &handle_));

    waiting_ = false;
  }
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
