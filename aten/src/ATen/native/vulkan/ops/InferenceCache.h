#pragma once

#ifdef USE_VULKAN_API

#include <deque>
#include <mutex>
#include <optional>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

template <typename Key, typename Value>
class InferenceLruCache final {
 private:
  struct Entry final {
    Key key;
    Value value;
  };

  std::deque<Entry> entries_;
  mutable std::mutex mutex_;
  size_t max_entries_;

 public:
  explicit InferenceLruCache(const size_t max_entries)
      : max_entries_(max_entries) {}

  template <typename MatchFn>
  std::optional<Value> lookup(const Key& query, MatchFn&& match_fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (!match_fn(it->key, query)) {
        continue;
      }

      Value value = it->value;
      if (it != entries_.begin()) {
        Entry entry = std::move(*it);
        entries_.erase(it);
        entries_.emplace_front(std::move(entry));
        value = entries_.front().value;
      }
      return value;
    }
    return std::nullopt;
  }

  template <typename MatchFn>
  void store(Key key, Value value, MatchFn&& match_fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (!match_fn(it->key, key)) {
        continue;
      }
      entries_.erase(it);
      break;
    }

    entries_.emplace_front(Entry{std::move(key), std::move(value)});
    while (entries_.size() > max_entries_) {
      entries_.pop_back();
    }
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
  }
};

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
