#include <c10/util/env.h>
#include <torch/csrc/lazy/core/config.h>

C10_EXPORT bool FLAGS_torch_lazy_ir_debug = false;
C10_EXPORT bool FLAGS_torch_lazy_param_aliasing = true;
C10_EXPORT bool FLAGS_torch_lazy_handle_special_scalars = false;
C10_EXPORT bool FLAGS_torch_lazy_all_numbers_special_scalars = false;
C10_EXPORT bool FLAGS_torch_lazy_reuse_ir = false;
C10_EXPORT bool FLAGS_torch_lazy_use_thread_pool = false;
C10_EXPORT bool FLAGS_torch_lazy_enable_device_data_cache = true;

C10_EXPORT int FLAGS_torch_lazy_compilation_cache_size = 1024;
C10_EXPORT int FLAGS_torch_lazy_device_data_cache_size = 128;
C10_EXPORT int FLAGS_torch_lazy_io_thread_pool_size = 1;
C10_EXPORT int FLAGS_torch_lazy_metrics_samples = 1024;
C10_EXPORT int FLAGS_torch_lazy_trim_graph_check_frequency = 5000;
C10_EXPORT int FLAGS_torch_lazy_trim_graph_size = 100000;
C10_EXPORT int FLAGS_torch_lazy_shape_cache_size = 4096;

namespace torch::lazy {
const std::string& getTorchLazyMetricsPercentiles() {
  static const std::string config = "0.01:0.05:0.1:0.2:0.5:0.8:0.9:0.95:0.99";
  return config;
}

std::string& getLTCForceFallback() {
  static std::string config;
  static bool _ignore = [&]() {
    auto env = c10::utils::get_env("LTC_FORCE_FALLBACK");
    if (env.has_value()) {
      config = std::string(env.value());
    }
    return true;
  }();
  (void)_ignore; // avoid unused variables warning
  return config;
}

} // namespace torch::lazy
