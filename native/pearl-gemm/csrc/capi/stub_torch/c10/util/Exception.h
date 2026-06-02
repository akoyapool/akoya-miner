// Torch-shim header for the torch-free libpearl_gemm_capi.so build.
//
// Several shared kernel headers (csrc/gemm/error_check.hpp,
// csrc/gemm/pearl_gemm_host.h, csrc/tensor_hash/tensor_hash_host.hpp)
// hard-code `#include <c10/util/Exception.h>` and use TORCH_CHECK for
// CUDA error reporting. When building libpearl_gemm_capi.so we resolve
// that include to this stub instead by passing
// `-Icsrc/capi/stub_torch` first on the include path. No real libtorch /
// libc10 link is required.
//
// TORCH_CHECK(cond, ...) throws std::runtime_error on failure. The
// existing capi `try { ... } catch (const std::exception&)` blocks
// translate the throw into a non-zero return code.

#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace pearl_capi_torch_shim {

template <typename T>
inline void stream_arg(std::ostringstream& os, T&& v) {
  os << std::forward<T>(v);
}

template <typename... Args>
[[noreturn]] inline void torch_check_fail(Args&&... args) {
  std::ostringstream os;
  (stream_arg(os, std::forward<Args>(args)), ...);
  throw std::runtime_error(os.str());
}

}  // namespace pearl_capi_torch_shim

#ifndef TORCH_CHECK
#define TORCH_CHECK(cond, ...)                                                \
  do {                                                                        \
    if (!(cond)) {                                                            \
      ::pearl_capi_torch_shim::torch_check_fail(__VA_ARGS__);                 \
    }                                                                         \
  } while (0)
#endif

#ifndef TORCH_INTERNAL_ASSERT
#define TORCH_INTERNAL_ASSERT(cond, ...) TORCH_CHECK((cond), ##__VA_ARGS__)
#endif
