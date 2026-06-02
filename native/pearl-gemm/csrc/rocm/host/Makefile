# ROCm host-integration build.
#
#   make            # build both shared libs
#
# Produces:
#   libpearl_gemm_capi.so  — ROCm backend for the pearl_capi_* C ABI
#   libcuda.so.1           — CUDA Driver API → HIP shim (lets Akoya.Cuda run unchanged)

HIPCC   ?= hipcc
ARCH    ?= gfx942
ROCM    ?= /opt/rocm-7.2.0
SRC     := $(abspath .)
ROOT    := $(abspath ../../..)/csrc

all: libpearl_gemm_capi.so libcuda.so.1

libpearl_gemm_capi.so: pearl_gemm_capi_rocm.cpp ../pearl_kernels.cuh ../blake3_device.cuh
	$(HIPCC) --offload-arch=$(ARCH) -O3 -fPIC -shared -I $(ROOT) -I .. $< -o $@

libcuda.so.1: cuda_hip_shim.cpp
	$(HIPCC) -O2 -fPIC -shared $< -o $@

clean:
	rm -f libpearl_gemm_capi.so libcuda.so.1


.PHONY: all clean
