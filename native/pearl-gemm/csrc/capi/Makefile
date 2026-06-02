# pearl_gemm — torch-free C-ABI shared library build.
#
# Produces libpearl_gemm_capi.so. Links only libcuda + libcudart;
# no libtorch / libpython / libc10 dependency. Used by akoya-miner
# (.NET / Native AOT) — see miner/akoya-miner/.
#
# Usage (from miner/pearl-gemm/):
#   make -f csrc/capi/Makefile                    # build (default: h100)
#   make -f csrc/capi/Makefile clean
#
# !! WSL / low-RAM note !!
#   NVCC_THREADS controls the number of parallel sub-processes nvcc
#   spawns internally (--threads N). Each thread can use 1-2 GiB of RAM
#   during template instantiation.  On WSL hosts with limited memory
#   (e.g. 8-16 GiB) keep this at 1 to avoid OOM:
#     NVCC=/usr/local/cuda-12.9/bin/nvcc NVCC_THREADS=1 make -f csrc/capi/Makefile ...
#
# Outputs:
#   csrc/capi/build/libpearl_gemm_capi.so

ROOT       := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
SRC        := $(ROOT)/csrc
CUTLASS    := $(ROOT)/third_party/cutlass
BUILD      := $(SRC)/capi/build
TARGET     := $(BUILD)/libpearl_gemm_capi.so

NVCC       ?= nvcc
NVCC_THREADS ?= 4
PYTHON     ?= python3
PEARL_GEMM_PORTABLE_MIN_BLOCKS ?=
PEARL_GEMM_AMPERE_BM ?=
PEARL_GEMM_AMPERE_BN ?=
PEARL_GEMM_AMPERE_KBLOCK ?=
PEARL_GEMM_AMPERE_STAGES ?=
PEARL_GEMM_AMPERE_SWIZZLE_BITS ?=
PEARL_GEMM_AMPERE_MIN_BLOCKS ?=
PEARL_GEMM_ADA_BM ?=
PEARL_GEMM_ADA_BN ?=
PEARL_GEMM_ADA_KBLOCK ?=
PEARL_GEMM_ADA_STAGES ?=
PEARL_GEMM_ADA_SWIZZLE_BITS ?=
PEARL_GEMM_ADA_MIN_BLOCKS ?=
PEARL_GEMM_BLACKWELL_STAGES ?=
PEARL_GEMM_BLACKWELL_CP_ASYNC_CACHE_ALWAYS ?=
PEARL_GEMM_BLACKWELL_B_CP_ASYNC_CACHE_ALWAYS ?=
PEARL_GEMM_BLACKWELL_KBLOCK ?=
PEARL_GEMM_BLACKWELL_SWIZZLE_BITS ?=
PEARL_GEMM_BLACKWELL_BM ?=
PEARL_GEMM_BLACKWELL_BN ?=
PEARL_GEMM_BLACKWELL_LOAD_POLICY ?= cp_async

# ── Architecture profile ─────────────────────────────────────────────────
# PEARL_GEMM_ARCH selects which GPU family we build for. Mirrors the
# _ARCH_CONFIGS dict in setup.py:
#   h100      → sm_90a only, full Hopper WGMMA pipeline (production for H100/H200/GH200)
#   volta     → sm_70 only, exact legacy DP4A transcript/helper kernels.
#   turing    → sm_75 only, exact SM75 int8 tensor-core transcript/helper
#               kernels without cp.async.
#   portable  → sm_80;sm_86 + PTX, per-warp mma.sync int8 portable
#               noisy_gemm/noise_B paths.
#               Runs on A100, RTX 3080/3090, and unknown future fallback.
#   ampere   → sm_80;sm_86, dedicated consumer headless transcript/XOR
#              mining path plus portable-family helper kernels.
#   ada       → sm_89 only, dedicated consumer lane using the headless
#               transcript/XOR mining path.
#   blackwell → sm_120a only, dedicated RTX 50-series transcript GEMM lane
#               plus portable-family helper kernels.
#   b200      → sm_100a native cubin. Uses the tcgen05/TMEM transcript GEMM
#               (csrc/blackwell/transcript_gemm_sm100.cu) for the hash
#               transcript path; the rest of the portable pipeline is shared.
PEARL_GEMM_ARCH ?= h100
PEARL_GEMM_BLACKWELL_MIN_BLOCKS ?=

ifeq ($(PEARL_GEMM_ARCH),h100)
  ARCH := -gencode=arch=compute_90a,code=sm_90a
  ARCH_DEFINES :=
else ifeq ($(PEARL_GEMM_ARCH),volta)
  ARCH := -gencode=arch=compute_70,code=sm_70
  ARCH_DEFINES := -DPEARL_GEMM_PORTABLE -DPEARL_GEMM_VOLTA
else ifeq ($(PEARL_GEMM_ARCH),turing)
  ARCH := -gencode=arch=compute_75,code=sm_75
  ARCH_DEFINES := -DPEARL_GEMM_PORTABLE -DPEARL_GEMM_TURING
else ifeq ($(PEARL_GEMM_ARCH),portable)
  ARCH := \
    -gencode=arch=compute_80,code=sm_80 \
    -gencode=arch=compute_86,code=sm_86 \
    -gencode=arch=compute_80,code=compute_80
  ARCH_DEFINES := -DPEARL_GEMM_PORTABLE
else ifeq ($(PEARL_GEMM_ARCH),ampere)
  ARCH := \
    -gencode=arch=compute_80,code=sm_80 \
    -gencode=arch=compute_86,code=sm_86
  ARCH_DEFINES := -DPEARL_GEMM_PORTABLE -DPEARL_GEMM_AMPERE
else ifeq ($(PEARL_GEMM_ARCH),ada)
  ARCH := -gencode=arch=compute_89,code=sm_89
  ARCH_DEFINES := -DPEARL_GEMM_PORTABLE -DPEARL_GEMM_ADA
else ifeq ($(PEARL_GEMM_ARCH),blackwell)
  ARCH := -gencode=arch=compute_120a,code=sm_120a
  ARCH_DEFINES := -DPEARL_GEMM_PORTABLE -DPEARL_GEMM_BLACKWELL
else ifeq ($(PEARL_GEMM_ARCH),b200)
  ARCH := -gencode=arch=compute_100a,code=sm_100a
  ARCH_DEFINES := -DPEARL_GEMM_PORTABLE -DPEARL_GEMM_B200
else
  $(error PEARL_GEMM_ARCH=$(PEARL_GEMM_ARCH) not recognised. Supported: h100, volta, turing, portable, ampere, ada, blackwell, b200)
endif

# Escape hatch for low-RAM dev hosts (e.g. WSL ~8 GiB) where the full
# portable multi-arch nvcc batch OOMs.  Setting
# PEARL_GEMM_LOCAL_ARCH=sm_86 (or sm_120a etc) overrides the gencode list
# above to that one cubin only.  Production builds must NOT set this.
ifneq ($(PEARL_GEMM_LOCAL_ARCH),)
  _LOCAL_COMPUTE := $(subst sm_,compute_,$(PEARL_GEMM_LOCAL_ARCH))
  ARCH := -gencode=arch=$(_LOCAL_COMPUTE),code=$(PEARL_GEMM_LOCAL_ARCH)
  $(info NOTE: PEARL_GEMM_LOCAL_ARCH override → $(ARCH))
endif

# Force-include the torch shim entry header so any TU that references
# TORCH_CHECK / torch::kFloat16 picks up the stubs at -Icsrc/capi/stub_torch.
PRELUDE    := $(SRC)/capi/capi_prelude.h

INCLUDES   := \
  -I$(SRC)/capi/stub_torch \
  -I$(SRC) \
  -I$(SRC)/gemm \
  -I$(SRC)/tensor_hash \
  -I$(SRC)/blake3 \
  -I$(SRC)/capi \
  -I$(CUTLASS)/include \
  -I$(CUTLASS)/examples/common \
  -I$(CUTLASS)/tools/util/include

DEFINES    := \
  -DPEARL_GEMM_NO_TORCH \
  -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
  -DNDEBUG \
  $(ARCH_DEFINES)

ifneq ($(PEARL_GEMM_PORTABLE_MIN_BLOCKS),)
  DEFINES += -DPEARL_PORTABLE_MIN_BLOCKS=$(PEARL_GEMM_PORTABLE_MIN_BLOCKS)
endif

ifneq (,$(filter $(PEARL_GEMM_ARCH),ampere ada blackwell))
  ifeq ($(PEARL_GEMM_ARCH),ampere)
    PEARL_CONSUMER_TILE_BM := $(PEARL_GEMM_AMPERE_BM)
    PEARL_CONSUMER_TILE_BN := $(PEARL_GEMM_AMPERE_BN)
  endif
  ifeq ($(PEARL_GEMM_ARCH),ada)
    PEARL_CONSUMER_TILE_BM := $(PEARL_GEMM_ADA_BM)
    PEARL_CONSUMER_TILE_BN := $(PEARL_GEMM_ADA_BN)
  endif
  ifeq ($(PEARL_GEMM_ARCH),blackwell)
    PEARL_CONSUMER_TILE_BM := $(PEARL_GEMM_BLACKWELL_BM)
    PEARL_CONSUMER_TILE_BN := $(PEARL_GEMM_BLACKWELL_BN)
  endif
  ifneq ($(PEARL_CONSUMER_TILE_BM),)
    ifneq ($(PEARL_CONSUMER_TILE_BM),128)
      $(error PEARL_GEMM_ARCH=$(PEARL_GEMM_ARCH) must use proof-canonical consumer tile BM=128, got $(PEARL_CONSUMER_TILE_BM))
    endif
  endif
  ifneq ($(PEARL_CONSUMER_TILE_BN),)
    ifneq ($(PEARL_CONSUMER_TILE_BN),256)
      $(error PEARL_GEMM_ARCH=$(PEARL_GEMM_ARCH) must use proof-canonical consumer tile BN=256, got $(PEARL_CONSUMER_TILE_BN))
    endif
  endif
endif

ifeq ($(PEARL_GEMM_ARCH),ampere)
  ifneq ($(PEARL_GEMM_AMPERE_BM),)
    DEFINES += -DPEARL_CONSUMER_BM=$(PEARL_GEMM_AMPERE_BM)
  endif
  ifneq ($(PEARL_GEMM_AMPERE_BN),)
    DEFINES += -DPEARL_CONSUMER_BN=$(PEARL_GEMM_AMPERE_BN)
  endif
  ifneq ($(PEARL_GEMM_AMPERE_KBLOCK),)
    DEFINES += -DPEARL_CONSUMER_KBLOCK=$(PEARL_GEMM_AMPERE_KBLOCK)
  endif
  ifneq ($(PEARL_GEMM_AMPERE_STAGES),)
    DEFINES += -DPEARL_CONSUMER_STAGES=$(PEARL_GEMM_AMPERE_STAGES)
  endif
  ifneq ($(PEARL_GEMM_AMPERE_SWIZZLE_BITS),)
    DEFINES += -DPEARL_CONSUMER_SWIZZLE_BITS=$(PEARL_GEMM_AMPERE_SWIZZLE_BITS)
  endif
  ifneq ($(PEARL_GEMM_AMPERE_MIN_BLOCKS),)
    DEFINES += -DPEARL_CONSUMER_MIN_BLOCKS=$(PEARL_GEMM_AMPERE_MIN_BLOCKS)
  endif
endif
ifeq ($(PEARL_GEMM_ARCH),ada)
  ifneq ($(PEARL_GEMM_ADA_BM),)
    DEFINES += -DPEARL_CONSUMER_BM=$(PEARL_GEMM_ADA_BM)
  endif
  ifneq ($(PEARL_GEMM_ADA_BN),)
    DEFINES += -DPEARL_CONSUMER_BN=$(PEARL_GEMM_ADA_BN)
  endif
  ifneq ($(PEARL_GEMM_ADA_KBLOCK),)
    DEFINES += -DPEARL_CONSUMER_KBLOCK=$(PEARL_GEMM_ADA_KBLOCK)
  endif
  ifneq ($(PEARL_GEMM_ADA_STAGES),)
    DEFINES += -DPEARL_CONSUMER_STAGES=$(PEARL_GEMM_ADA_STAGES)
  endif
  ifneq ($(PEARL_GEMM_ADA_SWIZZLE_BITS),)
    DEFINES += -DPEARL_CONSUMER_SWIZZLE_BITS=$(PEARL_GEMM_ADA_SWIZZLE_BITS)
  endif
  ifneq ($(PEARL_GEMM_ADA_MIN_BLOCKS),)
    DEFINES += -DPEARL_CONSUMER_MIN_BLOCKS=$(PEARL_GEMM_ADA_MIN_BLOCKS)
  endif
endif
ifeq ($(PEARL_GEMM_ARCH),blackwell)
  ifneq ($(PEARL_GEMM_BLACKWELL_MIN_BLOCKS),)
    DEFINES += -DPEARL_CONSUMER_MIN_BLOCKS=$(PEARL_GEMM_BLACKWELL_MIN_BLOCKS)
  endif
  ifneq ($(PEARL_GEMM_BLACKWELL_STAGES),)
    DEFINES += -DPEARL_CONSUMER_STAGES=$(PEARL_GEMM_BLACKWELL_STAGES)
  endif
  ifneq ($(PEARL_GEMM_BLACKWELL_CP_ASYNC_CACHE_ALWAYS),)
    DEFINES += -DPEARL_CONSUMER_CP_ASYNC_CACHE_ALWAYS=$(PEARL_GEMM_BLACKWELL_CP_ASYNC_CACHE_ALWAYS)
  endif
  ifneq ($(PEARL_GEMM_BLACKWELL_B_CP_ASYNC_CACHE_ALWAYS),)
    DEFINES += -DPEARL_CONSUMER_B_CP_ASYNC_CACHE_ALWAYS=$(PEARL_GEMM_BLACKWELL_B_CP_ASYNC_CACHE_ALWAYS)
  endif
  ifneq ($(PEARL_GEMM_BLACKWELL_KBLOCK),)
    DEFINES += -DPEARL_CONSUMER_KBLOCK=$(PEARL_GEMM_BLACKWELL_KBLOCK)
  endif
  ifneq ($(PEARL_GEMM_BLACKWELL_SWIZZLE_BITS),)
    DEFINES += -DPEARL_CONSUMER_SWIZZLE_BITS=$(PEARL_GEMM_BLACKWELL_SWIZZLE_BITS)
  endif
  ifneq ($(PEARL_GEMM_BLACKWELL_BM),)
    DEFINES += -DPEARL_CONSUMER_BM=$(PEARL_GEMM_BLACKWELL_BM)
  endif
  ifneq ($(PEARL_GEMM_BLACKWELL_BN),)
    DEFINES += -DPEARL_CONSUMER_BN=$(PEARL_GEMM_BLACKWELL_BN)
  endif
  ifeq ($(PEARL_GEMM_BLACKWELL_LOAD_POLICY),tma)
    DEFINES += -DPEARL_CONSUMER_USE_TMA_EXPERIMENT=1
  else ifneq ($(PEARL_GEMM_BLACKWELL_LOAD_POLICY),cp_async)
    $(error PEARL_GEMM_BLACKWELL_LOAD_POLICY=$(PEARL_GEMM_BLACKWELL_LOAD_POLICY) not recognised. Supported: cp_async, tma)
  endif
endif

NVCC_FLAGS := \
  -O3 -std=c++20 -Xcompiler -fPIC -Xcompiler -fvisibility=hidden \
  -Xcompiler -ffile-prefix-map=$(realpath $(SRC)/../..)=pearl \
  $(ARCH) \
  -U__CUDA_NO_HALF_OPERATORS__ \
  -U__CUDA_NO_HALF_CONVERSIONS__ \
  -U__CUDA_NO_BFLOAT16_OPERATORS__ \
  -U__CUDA_NO_BFLOAT16_CONVERSIONS__ \
  -U__CUDA_NO_BFLOAT162_OPERATORS__ \
  -U__CUDA_NO_BFLOAT162_CONVERSIONS__ \
  --expt-relaxed-constexpr \
  --expt-extended-lambda \
  --use_fast_math \
  --threads $(NVCC_THREADS) \
  -include $(PRELUDE) \
  $(INCLUDES) $(DEFINES)

# Sources — everything pearl_capi_* needs.
# Excludes: csrc/gemm/pearl_gemm_api.cpp (pybind shell, depends on libtorch)
#           csrc/gemm/quantize_kernel.cu (Python-only utility)
#           csrc/gemm/denoise_converter.cu (only used when SkipDenoising=false; pure-miner always skips)
SRCS_COMMON := \
  $(SRC)/capi/pearl_gemm_capi.cpp \
  $(SRC)/capi/pearl_gemm_capi_util.cu \
  $(SRC)/blake3/blake3.cu \
  $(SRC)/tensor_hash/tensor_hash.cu \
  $(SRC)/gemm/noise_generation.cu \
  $(SRC)/gemm/inner_hash_kernel.cu

# H100 path: full Hopper noisingA/B + matmul instantiations.
SRCS_HOPPER := \
  $(SRC)/gemm/instantiations/gemm_R64_bf16_128x256x128_3stages_cluster1x1.cu \
  $(SRC)/gemm/instantiations/gemm_R128_bf16_128x256x128_3stages_cluster1x1.cu \
  $(SRC)/gemm/instantiations/noisingA_R64_fp16_64x64_2stages.cu \
  $(SRC)/gemm/instantiations/noisingA_R128_fp16_64x64_2stages.cu \
  $(SRC)/gemm/instantiations/noisingB_R64_fp16_64x64_2stages.cu \
  $(SRC)/gemm/instantiations/noisingB_R128_fp16_64x64_2stages.cu

# Portable path: per-warp mma.sync int8 GEMM + native int8 helpers.
SRCS_PORTABLE := \
  $(SRC)/portable/transcript_kernel.cu \
  $(SRC)/portable/transcript_gemm_kernel.cu \
  $(SRC)/portable/portable_int8_gemm.cu \
  $(SRC)/capi/portable_int8_helpers.cu

SRCS_LEGACY := \
  $(SRC)/portable/transcript_kernel.cu \
  $(SRC)/portable/transcript_gemm_dp4a.cu \
  $(SRC)/portable/portable_int8_gemm_dp4a.cu \
  $(SRC)/capi/portable_int8_helpers.cu

SRCS_TURING := \
  $(SRC)/portable/transcript_kernel.cu \
  $(SRC)/portable/transcript_gemm_sm75.cu \
  $(SRC)/portable/portable_int8_gemm_sm75.cu \
  $(SRC)/capi/portable_int8_helpers.cu

SRCS_ADA := \
  $(SRC)/portable/transcript_kernel.cu \
  $(SRC)/consumer/transcript_gemm_kernel.cu \
  $(SRC)/portable/portable_int8_gemm.cu \
  $(SRC)/capi/portable_int8_helpers.cu

SRCS_AMPERE := \
  $(SRC)/portable/transcript_kernel.cu \
  $(SRC)/consumer/transcript_gemm_kernel.cu \
  $(SRC)/portable/portable_int8_gemm.cu \
  $(SRC)/capi/portable_int8_helpers.cu

SRCS_BLACKWELL := \
  $(SRC)/portable/transcript_kernel.cu \
  $(SRC)/consumer/transcript_gemm_kernel.cu \
  $(SRC)/portable/portable_int8_gemm.cu \
  $(SRC)/capi/portable_int8_helpers.cu

# B200 path: same portable helpers, but the sm_100a tcgen05/TMEM transcript
# GEMM replaces the per-warp mma.sync/consumer transcript GEMM.
SRCS_B200 := \
  $(SRC)/portable/transcript_kernel.cu \
  $(SRC)/blackwell/transcript_gemm_sm100.cu \
  $(SRC)/portable/portable_int8_gemm.cu \
  $(SRC)/capi/portable_int8_helpers.cu

STATIC_SWITCH_HEADERS := \
  $(SRC)/gemm/static_switch_matmul.h \
  $(SRC)/gemm/static_switch_noisingA.h \
  $(SRC)/gemm/static_switch_noisingB.h

ifeq ($(PEARL_GEMM_ARCH),h100)
  SRCS := $(SRCS_COMMON) $(SRCS_HOPPER)
  EXTRA_LDLIBS :=
else ifeq ($(PEARL_GEMM_ARCH),volta)
  SRCS := $(SRCS_COMMON) $(SRCS_LEGACY)
  EXTRA_LDLIBS :=
else ifeq ($(PEARL_GEMM_ARCH),turing)
  SRCS := $(SRCS_COMMON) $(SRCS_TURING)
  EXTRA_LDLIBS :=
else ifeq ($(PEARL_GEMM_ARCH),ada)
  SRCS := $(SRCS_COMMON) $(SRCS_ADA)
  EXTRA_LDLIBS :=
else ifeq ($(PEARL_GEMM_ARCH),ampere)
  SRCS := $(SRCS_COMMON) $(SRCS_AMPERE)
  EXTRA_LDLIBS :=
else ifeq ($(PEARL_GEMM_ARCH),blackwell)
  SRCS := $(SRCS_COMMON) $(SRCS_BLACKWELL)
  EXTRA_LDLIBS :=
else ifeq ($(PEARL_GEMM_ARCH),b200)
  SRCS := $(SRCS_COMMON) $(SRCS_B200)
  EXTRA_LDLIBS :=
else
  SRCS := $(SRCS_COMMON) $(SRCS_PORTABLE)
  EXTRA_LDLIBS :=
endif

OBJS := $(patsubst $(SRC)/%, $(BUILD)/%.o, $(SRCS))

LDLIBS := -lcuda -lcudart -lstdc++ $(EXTRA_LDLIBS)

.PHONY: all clean generate-static-switches
all: $(TARGET)
	@echo "==> built libpearl_gemm_capi.so for PEARL_GEMM_ARCH=$(PEARL_GEMM_ARCH)"

$(TARGET): $(STATIC_SWITCH_HEADERS) $(OBJS) | $(BUILD)
	$(NVCC) -shared -Xcompiler -fPIC -Xlinker --no-undefined \
	  $(ARCH) -o $@ $(OBJS) $(LDLIBS)
	@echo "==> $@"
	@ls -lh $@
	@echo "==> ldd:"
	@ldd $@ | sed 's/^/    /' || true

$(BUILD)/%.cu.o: $(SRC)/%.cu | $(BUILD)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD)/%.cpp.o: $(SRC)/%.cpp | $(BUILD)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -x cu -c $< -o $@

$(STATIC_SWITCH_HEADERS): generate-static-switches

generate-static-switches:
	$(PYTHON) $(SRC)/capi/generate_static_switches.py

$(BUILD):
	@mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
