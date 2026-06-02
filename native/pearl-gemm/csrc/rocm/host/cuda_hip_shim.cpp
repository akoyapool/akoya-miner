// cuda_hip_shim.cpp — a libcuda.so.1 drop-in that forwards the CUDA Driver API
// surface used by Akoya.Cuda.CudaDriver to HIP, so the C# host runs unchanged
// on AMD (gfx942). Build → libcuda.so.1, point the C# resolver at it.
//
//   hipcc -O2 -fPIC -shared cuda_hip_shim.cpp -o libcuda.so.1
//
// CUdeviceptr is a pointer-width handle; we represent it as void*. CUresult 0 ==
// Success and hipSuccess == 0, so we return the hipError_t code directly.

#include <hip/hip_runtime.h>
#include <cstring>

extern "C" {

int cuInit(unsigned int) { return hipInit(0); }
int cuDriverGetVersion(int* v){ return hipDriverGetVersion(v); }
int cuDeviceGetCount(int* c){ return hipGetDeviceCount(c); }
int cuDeviceGet(int* dev,int ordinal){ *dev=ordinal; return hipSuccess; }
int cuDeviceGetName(char* name,int len,int dev){ return hipDeviceGetName(name,len,dev); }
int cuDeviceTotalMem_v2(size_t* b,int dev){ hipDeviceProp_t p; int e=hipGetDeviceProperties(&p,dev); *b=p.totalGlobalMem; return e; }
int cuDeviceGetPCIBusId(char* s,int len,int dev){ return hipDeviceGetPCIBusId(s,len,dev); }

int cuDeviceComputeCapability(int* major,int* minor,int){ *major=9; *minor=0; return hipSuccess; }

// CUdevice_attribute → hipDeviceAttribute_t (only the few CudaDriver queries).
int cuDeviceGetAttribute(int* pi,int attr,int dev){
  hipDeviceAttribute_t a;
  switch(attr){
    case 16: a=hipDeviceAttributeMultiprocessorCount; break;     // MULTIPROCESSOR_COUNT
    case 13: a=hipDeviceAttributeClockRate; break;               // CLOCK_RATE (kHz)
    case 36: a=hipDeviceAttributeMemoryClockRate; break;         // MEMORY_CLOCK_RATE
    case 37: a=hipDeviceAttributeMemoryBusWidth; break;          // GLOBAL_MEMORY_BUS_WIDTH
    default: *pi=0; return hipSuccess;
  }
  return hipDeviceGetAttribute(pi,a,dev);
}

// Context management — HIP primary-context + legacy ctx API.
int cuCtxCreate_v2(void** ctx,unsigned int,int dev){ return hipCtxCreate((hipCtx_t*)ctx,0,dev); }
int cuCtxDestroy_v2(void* ctx){ return hipCtxDestroy((hipCtx_t)ctx); }
int cuCtxSetCurrent(void* ctx){ return hipCtxSetCurrent((hipCtx_t)ctx); }
int cuDevicePrimaryCtxRetain(void** ctx,int dev){ return hipDevicePrimaryCtxRetain((hipCtx_t*)ctx,dev); }
int cuDevicePrimaryCtxRelease_v2(int dev){ return hipDevicePrimaryCtxRelease(dev); }
int cuDevicePrimaryCtxSetFlags_v2(int dev,unsigned int f){ return hipDevicePrimaryCtxSetFlags(dev,f); }

// Memory.
int cuMemAlloc_v2(void** dptr,size_t n){ return hipMalloc(dptr,n); }
int cuMemFree_v2(void* dptr){ return hipFree(dptr); }
int cuMemGetInfo_v2(size_t* fr,size_t* tot){ return hipMemGetInfo(fr,tot); }
int cuMemHostAlloc(void** pp,size_t n,unsigned int){ return hipHostMalloc(pp,n,hipHostMallocDefault); }
int cuMemFreeHost(void* p){ return hipHostFree(p); }
int cuMemcpyHtoD_v2(void* dst,const void* src,size_t n){ return hipMemcpyHtoD((hipDeviceptr_t)dst,(void*)src,n); }
int cuMemcpyDtoH_v2(void* dst,const void* src,size_t n){ return hipMemcpyDtoH(dst,(hipDeviceptr_t)src,n); }
int cuMemcpyHtoDAsync_v2(void* dst,const void* src,size_t n,void* s){ return hipMemcpyHtoDAsync((hipDeviceptr_t)dst,(void*)src,n,(hipStream_t)s); }
int cuMemcpyDtoHAsync_v2(void* dst,const void* src,size_t n,void* s){ return hipMemcpyDtoHAsync(dst,(hipDeviceptr_t)src,n,(hipStream_t)s); }
int cuMemsetD8_v2(void* dst,unsigned char v,size_t n){ return hipMemset(dst,v,n); }
int cuMemsetD8Async(void* dst,unsigned char v,size_t n,void* s){ return hipMemsetAsync(dst,v,n,(hipStream_t)s); }

// Streams.
int cuStreamCreate(void** s,unsigned int){ return hipStreamCreateWithFlags((hipStream_t*)s,hipStreamNonBlocking); }
int cuStreamSynchronize(void* s){ return hipStreamSynchronize((hipStream_t)s); }
int cuStreamDestroy_v2(void* s){ return hipStreamDestroy((hipStream_t)s); }

// Events.
int cuEventCreate(void** e,unsigned int){ return hipEventCreate((hipEvent_t*)e); }
int cuEventRecord(void* e,void* s){ return hipEventRecord((hipEvent_t)e,(hipStream_t)s); }
int cuEventSynchronize(void* e){ return hipEventSynchronize((hipEvent_t)e); }
int cuEventElapsedTime(float* ms,void* a,void* b){ return hipEventElapsedTime(ms,(hipEvent_t)a,(hipEvent_t)b); }
int cuEventDestroy_v2(void* e){ return hipEventDestroy((hipEvent_t)e); }

} // extern "C"
