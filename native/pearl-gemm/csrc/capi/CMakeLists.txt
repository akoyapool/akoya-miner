# pearl_gemm — Windows CMake build for the torch-free C-ABI shared library.
#
# Produces pearl_gemm_capi.dll. This is the WINDOWS counterpart to the GNU
# Makefile in this directory (which remains the canonical Linux build and is
# left untouched). It mirrors the Makefile's arch profiles, source sets,
# include paths and nvcc flags, translated for nvcc + the MSVC host compiler.
#
# Usage (from a VS Developer environment so nvcc finds cl.exe):
#   cmake -S csrc/capi -B csrc/capi/build-win -G Ninja \
#         -DPEARL_GEMM_ARCH=ampere -DCMAKE_BUILD_TYPE=Release
#   cmake --build csrc/capi/build-win
#   # → csrc/capi/build-win/pearl_gemm_capi.dll
#
# build.ps1 drives this end-to-end (arch auto-detect, CUTLASS submodule, etc).

cmake_minimum_required(VERSION 3.24)

project(pearl_gemm_capi LANGUAGES CXX CUDA)

# ── Paths (mirror the Makefile's ROOT/SRC/CUTLASS) ───────────────────────────
set(PEARL_GEMM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")   # native/pearl-gemm
set(SRC             "${PEARL_GEMM_ROOT}/csrc")
set(CUTLASS         "${PEARL_GEMM_ROOT}/third_party/cutlass")

if(NOT EXISTS "${CUTLASS}/include/cutlass/cutlass.h")
  message(FATAL_ERROR
    "CUTLASS not found at ${CUTLASS}. Fetch the submodule:\n"
    "  git submodule update --init --depth 1 native/pearl-gemm/third_party/cutlass")
endif()

# ── Architecture profile (mirror the Makefile's PEARL_GEMM_ARCH switch) ──────
set(PEARL_GEMM_ARCH "ampere" CACHE STRING
    "GPU profile: h100|volta|turing|portable|ampere|ada|blackwell|b200")

# Map profile → CMAKE_CUDA_ARCHITECTURES + the -DPEARL_GEMM_* defines and the
# arch-specific source set. The "a" suffixes (90a/100a/120a) need CMake 3.26+.
set(ARCH_DEFINES "")
if(PEARL_GEMM_ARCH STREQUAL "h100")
  set(CMAKE_CUDA_ARCHITECTURES 90a)
elseif(PEARL_GEMM_ARCH STREQUAL "volta")
  set(CMAKE_CUDA_ARCHITECTURES 70)
  set(ARCH_DEFINES PEARL_GEMM_PORTABLE PEARL_GEMM_VOLTA)
elseif(PEARL_GEMM_ARCH STREQUAL "turing")
  set(CMAKE_CUDA_ARCHITECTURES 75)
  set(ARCH_DEFINES PEARL_GEMM_PORTABLE PEARL_GEMM_TURING)
elseif(PEARL_GEMM_ARCH STREQUAL "portable")
  set(CMAKE_CUDA_ARCHITECTURES 80 86)
  set(ARCH_DEFINES PEARL_GEMM_PORTABLE)
elseif(PEARL_GEMM_ARCH STREQUAL "ampere")
  set(CMAKE_CUDA_ARCHITECTURES 80 86)
  set(ARCH_DEFINES PEARL_GEMM_PORTABLE PEARL_GEMM_AMPERE)
elseif(PEARL_GEMM_ARCH STREQUAL "ada")
  set(CMAKE_CUDA_ARCHITECTURES 89)
  set(ARCH_DEFINES PEARL_GEMM_PORTABLE PEARL_GEMM_ADA)
elseif(PEARL_GEMM_ARCH STREQUAL "blackwell")
  set(CMAKE_CUDA_ARCHITECTURES 120a)
  set(ARCH_DEFINES PEARL_GEMM_PORTABLE PEARL_GEMM_BLACKWELL)
elseif(PEARL_GEMM_ARCH STREQUAL "b200")
  set(CMAKE_CUDA_ARCHITECTURES 100a)
  set(ARCH_DEFINES PEARL_GEMM_PORTABLE PEARL_GEMM_B200)
else()
  message(FATAL_ERROR "PEARL_GEMM_ARCH=${PEARL_GEMM_ARCH} not recognised. "
    "Supported: h100, volta, turing, portable, ampere, ada, blackwell, b200")
endif()

# ── Source sets (mirror the Makefile's SRCS_* groups) ────────────────────────
set(SRCS_COMMON
  "${SRC}/capi/pearl_gemm_capi.cpp"
  "${SRC}/capi/pearl_gemm_capi_util.cu"
  "${SRC}/blake3/blake3.cu"
  "${SRC}/tensor_hash/tensor_hash.cu"
  "${SRC}/gemm/noise_generation.cu"
  "${SRC}/gemm/inner_hash_kernel.cu")

set(SRCS_HOPPER
  "${SRC}/gemm/instantiations/gemm_R64_bf16_128x256x128_3stages_cluster1x1.cu"
  "${SRC}/gemm/instantiations/gemm_R128_bf16_128x256x128_3stages_cluster1x1.cu"
  "${SRC}/gemm/instantiations/noisingA_R64_fp16_64x64_2stages.cu"
  "${SRC}/gemm/instantiations/noisingA_R128_fp16_64x64_2stages.cu"
  "${SRC}/gemm/instantiations/noisingB_R64_fp16_64x64_2stages.cu"
  "${SRC}/gemm/instantiations/noisingB_R128_fp16_64x64_2stages.cu")

set(SRCS_PORTABLE
  "${SRC}/portable/transcript_kernel.cu"
  "${SRC}/portable/transcript_gemm_kernel.cu"
  "${SRC}/portable/portable_int8_gemm.cu"
  "${SRC}/capi/portable_int8_helpers.cu")

set(SRCS_TURING
  "${SRC}/portable/transcript_kernel.cu"
  "${SRC}/portable/transcript_gemm_sm75.cu"
  "${SRC}/portable/portable_int8_gemm_sm75.cu"
  "${SRC}/capi/portable_int8_helpers.cu")

set(SRCS_LEGACY
  "${SRC}/portable/transcript_kernel.cu"
  "${SRC}/portable/transcript_gemm_dp4a.cu"
  "${SRC}/portable/portable_int8_gemm_dp4a.cu"
  "${SRC}/capi/portable_int8_helpers.cu")

# ampere/ada/blackwell share the consumer transcript lane (mirrors the
# Makefile's SRCS_ADA / SRCS_AMPERE / SRCS_BLACKWELL).
set(SRCS_CONSUMER
  "${SRC}/portable/transcript_kernel.cu"
  "${SRC}/consumer/transcript_gemm_kernel.cu"
  "${SRC}/portable/portable_int8_gemm.cu"
  "${SRC}/capi/portable_int8_helpers.cu")

set(SRCS_B200
  "${SRC}/portable/transcript_kernel.cu"
  "${SRC}/blackwell/transcript_gemm_sm100.cu"
  "${SRC}/portable/portable_int8_gemm.cu"
  "${SRC}/capi/portable_int8_helpers.cu")

if(PEARL_GEMM_ARCH STREQUAL "h100")
  set(ARCH_SRCS ${SRCS_HOPPER})
elseif(PEARL_GEMM_ARCH STREQUAL "volta")
  set(ARCH_SRCS ${SRCS_LEGACY})
elseif(PEARL_GEMM_ARCH STREQUAL "turing")
  set(ARCH_SRCS ${SRCS_TURING})
elseif(PEARL_GEMM_ARCH STREQUAL "b200")
  set(ARCH_SRCS ${SRCS_B200})
elseif(PEARL_GEMM_ARCH MATCHES "^(ampere|ada|blackwell)$")
  set(ARCH_SRCS ${SRCS_CONSUMER})
else() # portable
  set(ARCH_SRCS ${SRCS_PORTABLE})
endif()

# ── Generate the static-switch headers (Makefile: generate-static-switches) ──
# Run at configure time so the headers exist before any TU compiles. python is
# resolved by build.ps1 / the dev environment.
find_program(PEARL_PYTHON NAMES python python3 py REQUIRED)
execute_process(
  COMMAND "${PEARL_PYTHON}" "${SRC}/capi/generate_static_switches.py"
  RESULT_VARIABLE _gen_rc
  OUTPUT_VARIABLE _gen_out ERROR_VARIABLE _gen_out)
if(NOT _gen_rc EQUAL 0)
  message(FATAL_ERROR "generate_static_switches.py failed:\n${_gen_out}")
endif()

# ── The shared library ───────────────────────────────────────────────────────
add_library(pearl_gemm_capi SHARED ${SRCS_COMMON} ${ARCH_SRCS})

# pearl_gemm_capi.cpp is compiled as CUDA in the Makefile (`nvcc -x cu`); do the
# same here so its CUDA/CUTLASS includes resolve.
set_source_files_properties("${SRC}/capi/pearl_gemm_capi.cpp"
  PROPERTIES LANGUAGE CUDA)

set_target_properties(pearl_gemm_capi PROPERTIES
  PREFIX ""                       # no "lib" prefix → pearl_gemm_capi.dll
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  CUDA_SEPARABLE_COMPILATION OFF
  # Export the C ABI without per-symbol __declspec(dllexport): mirrors the
  # Linux build's "export everything" behaviour. The extern "C" pearl_capi_*
  # functions land in the DLL export table under their plain names.
  WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(pearl_gemm_capi PRIVATE
  "${SRC}/capi/stub_torch"        # torch shim — MUST precede real-looking paths
  "${SRC}"
  "${SRC}/gemm"
  "${SRC}/tensor_hash"
  "${SRC}/blake3"
  "${SRC}/capi"
  "${CUTLASS}/include"
  "${CUTLASS}/examples/common"
  "${CUTLASS}/tools/util/include")

target_compile_definitions(pearl_gemm_capi PRIVATE
  PEARL_GEMM_NO_TORCH
  CUTLASS_DEBUG_TRACE_LEVEL=0
  NDEBUG
  ${ARCH_DEFINES})

# nvcc flags carried over from the Makefile's NVCC_FLAGS (GCC-only -fPIC /
# -fvisibility / -ffile-prefix-map dropped; export handled by __declspec).
set(_prelude "${SRC}/capi/capi_prelude.h")
target_compile_options(pearl_gemm_capi PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_BFLOAT16_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    -U__CUDA_NO_BFLOAT162_OPERATORS__
    -U__CUDA_NO_BFLOAT162_CONVERSIONS__
    # Force-include the torch-stub prelude (Makefile: `nvcc -include <file>`).
    # Must be two separate tokens so nvcc recognises -include and translates it
    # to the host compiler's force-include (cl /FI); glued, cl ignores it.
    -include ${_prelude}
  >)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(pearl_gemm_capi PRIVATE CUDA::cudart CUDA::cuda_driver)

# Stage the DLL next to the build dir root for build.ps1 to pick up.
install(TARGETS pearl_gemm_capi RUNTIME DESTINATION .)
