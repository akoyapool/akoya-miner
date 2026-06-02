// capi_prelude.h — force-included via `nvcc -include` for the torch-free
// libpearl_gemm_capi.so build. Pulls in the torch shim symbols so that
// shared kernel headers (csrc/gemm/error_check.hpp,
// csrc/gemm/pearl_gemm_host.h, csrc/tensor_hash/tensor_hash_host.hpp,
// csrc/gemm/static_switch_noisingA.h, …) compile without any real
// libtorch / libc10 headers on the include path.
//
// See csrc/capi/stub_torch/c10/util/Exception.h and
//     csrc/capi/stub_torch/torch/types.h for the actual stub implementations.

#pragma once

#include <c10/util/Exception.h>  // resolves to stub_torch/c10/util/Exception.h
#include <torch/types.h>          // resolves to stub_torch/torch/types.h
