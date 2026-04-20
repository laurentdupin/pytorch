#pragma once

#ifdef USE_VULKAN_API

#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <limits>
#include <list>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

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
    size_t hash{0u};
    size_t weight{1u};
  };

  using EntryList = std::list<Entry>;
  using EntryIterator = typename EntryList::iterator;

  EntryList entries_;
  std::unordered_multimap<size_t, EntryIterator> index_;
  mutable std::mutex mutex_;
  size_t max_entries_;
  size_t max_weight_;
  size_t total_weight_{0u};

  void erase_index_entry_locked(const EntryIterator& entry_it) {
    auto bucket = index_.equal_range(entry_it->hash);
    for (auto it = bucket.first; it != bucket.second; ++it) {
      if (it->second == entry_it) {
        index_.erase(it);
        return;
      }
    }
  }

  void erase_entry_locked(const EntryIterator& entry_it) {
    total_weight_ -= entry_it->weight;
    erase_index_entry_locked(entry_it);
    entries_.erase(entry_it);
  }

  void trim_locked() {
    while (
        (entries_.size() > max_entries_ || total_weight_ > max_weight_) &&
        !entries_.empty()) {
      erase_entry_locked(std::prev(entries_.end()));
    }
  }

 public:
  explicit InferenceLruCache(
      const size_t max_entries,
      const size_t max_weight = std::numeric_limits<size_t>::max())
      : max_entries_(std::max<size_t>(size_t{1u}, max_entries)),
        max_weight_(std::max<size_t>(size_t{1u}, max_weight)) {}

  template <typename MatchFn>
  std::optional<Value> lookup(const Key& query, MatchFn&& match_fn) {
    return lookup(
        query,
        [](const Key&) {
          return size_t{0u};
        },
        std::forward<MatchFn>(match_fn));
  }

  template <typename HashFn, typename MatchFn>
  std::optional<Value> lookup(
      const Key& query,
      HashFn&& hash_fn,
      MatchFn&& match_fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t query_hash = hash_fn(query);
    auto bucket = index_.equal_range(query_hash);
    for (auto it = bucket.first; it != bucket.second; ++it) {
      EntryIterator entry_it = it->second;
      if (!match_fn(entry_it->key, query)) {
        continue;
      }

      if (entry_it != entries_.begin()) {
        entries_.splice(entries_.begin(), entries_, entry_it);
        entry_it = entries_.begin();
      }
      return entry_it->value;
    }
    return std::nullopt;
  }

  template <typename MatchFn>
  void store(Key key, Value value, MatchFn&& match_fn) {
    store(
        std::move(key),
        std::move(value),
        [](const Key&) {
          return size_t{0u};
        },
        std::forward<MatchFn>(match_fn));
  }

  template <typename HashFn, typename MatchFn>
  void store(Key key, Value value, HashFn&& hash_fn, MatchFn&& match_fn) {
    store(
        std::move(key),
        std::move(value),
        std::forward<HashFn>(hash_fn),
        std::forward<MatchFn>(match_fn),
        [](const Value&) {
          return size_t{1u};
        });
  }

  template <typename HashFn, typename MatchFn, typename WeightFn>
  void store(
      Key key,
      Value value,
      HashFn&& hash_fn,
      MatchFn&& match_fn,
      WeightFn&& weight_fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t query_hash = hash_fn(key);
    auto bucket = index_.equal_range(query_hash);
    for (auto it = bucket.first; it != bucket.second; ++it) {
      EntryIterator entry_it = it->second;
      if (!match_fn(entry_it->key, key)) {
        continue;
      }

      total_weight_ -= entry_it->weight;
      entries_.erase(entry_it);
      index_.erase(it);
      break;
    }

    const size_t entry_weight =
        std::max<size_t>(size_t{1u}, weight_fn(value));
    entries_.emplace_front(Entry{
        std::move(key),
        std::move(value),
        query_hash,
        entry_weight,
    });
    index_.emplace(query_hash, entries_.begin());
    total_weight_ += entry_weight;

    trim_locked();
  }

  template <typename Predicate>
  size_t erase_if(Predicate&& predicate) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t erased = 0u;
    for (auto it = entries_.begin(); it != entries_.end();) {
      if (!predicate(it->key, it->value)) {
        ++it;
        continue;
      }

      total_weight_ -= it->weight;
      erase_index_entry_locked(it);
      it = entries_.erase(it);
      ++erased;
    }
    return erased;
  }

  template <typename Predicate>
  size_t erase_if_budgeted(
      Predicate&& predicate,
      const size_t max_scanned,
      const size_t max_erased) {
    return erase_if_budgeted(
        std::forward<Predicate>(predicate),
        max_scanned,
        max_erased,
        [](Value&&) {});
  }

  template <typename Predicate, typename RetireFn>
  size_t erase_if_budgeted(
      Predicate&& predicate,
      const size_t max_scanned,
      const size_t max_erased,
      RetireFn&& retire_fn) {
    EntryList retired_entries;
    std::unique_lock<std::mutex> lock(mutex_);
    if (entries_.empty() || max_scanned == 0u || max_erased == 0u) {
      return 0u;
    }

    size_t scanned = 0u;
    size_t erased = 0u;
    auto it = std::prev(entries_.end());
    while (scanned < max_scanned && erased < max_erased) {
      EntryIterator current = it;
      const bool at_begin = current == entries_.begin();
      if (!at_begin) {
        it = std::prev(current);
      }

      ++scanned;
      if (predicate(current->key, current->value)) {
        total_weight_ -= current->weight;
        erase_index_entry_locked(current);
        retired_entries.splice(retired_entries.begin(), entries_, current);
        ++erased;
      }

      if (at_begin) {
        break;
      }
    }
    lock.unlock();
    for (Entry& entry : retired_entries) {
      retire_fn(std::move(entry.value));
    }
    if (!retired_entries.empty() && c10::InferenceMode::is_enabled()) {
      c10::InferenceMode inference_mode_guard(false);
      retired_entries.clear();
    }
    return erased;
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
