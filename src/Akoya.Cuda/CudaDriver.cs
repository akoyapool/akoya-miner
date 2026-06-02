// Minimal P/Invoke surface for the CUDA Driver API.
// We use the Driver API (cu*) rather than the Runtime API (cuda*) because
// it does not require linking libcudart and gives finer control over
// contexts and stream priorities.
//
// Linkage: at runtime we depend on libcuda.so.1 which is provided by the
// NVIDIA driver itself (not bundled in any toolkit package). Containers
// that use --gpus all already have it mounted into /usr/lib/x86_64-linux-gnu
// via the nvidia-container-runtime hook.
//
// AOT-friendly: every entrypoint uses [LibraryImport] (source-generated
// stubs) rather than [DllImport].

using System.Runtime.InteropServices;

namespace Akoya.Cuda;

public enum CUresult : int
{
    Success = 0,
}

public readonly record struct CUdevice(int Handle);
public readonly record struct CUcontext(nint Handle);
public readonly record struct CUstream(nint Handle);
public readonly record struct CUevent(nint Handle);
public readonly record struct CUdeviceptr(nint Handle);

public static partial class CudaDriver
{
    private const string Lib = "cuda";

    [LibraryImport(Lib, EntryPoint = "cuInit")]
    public static partial CUresult Init(uint flags);

    [LibraryImport(Lib, EntryPoint = "cuDeviceGetCount")]
    public static partial CUresult DeviceGetCount(out int count);

    [LibraryImport(Lib, EntryPoint = "cuDeviceGet")]
    public static partial CUresult DeviceGet(out CUdevice device, int ordinal);

    [LibraryImport(Lib, EntryPoint = "cuDeviceGetName", StringMarshalling = StringMarshalling.Utf8)]
    public static partial CUresult DeviceGetName(Span<byte> name, int len, CUdevice device);

    [LibraryImport(Lib, EntryPoint = "cuDeviceComputeCapability")]
    public static partial CUresult DeviceComputeCapability(out int major, out int minor, CUdevice device);

    [LibraryImport(Lib, EntryPoint = "cuCtxCreate_v2")]
    public static partial CUresult CtxCreate(out CUcontext ctx, uint flags, CUdevice device);

    [LibraryImport(Lib, EntryPoint = "cuCtxDestroy_v2")]
    public static partial CUresult CtxDestroy(CUcontext ctx);

    [LibraryImport(Lib, EntryPoint = "cuMemAlloc_v2")]
    public static partial CUresult MemAlloc(out CUdeviceptr dptr, nuint bytesize);

    [LibraryImport(Lib, EntryPoint = "cuMemFree_v2")]
    public static partial CUresult MemFree(CUdeviceptr dptr);

    [LibraryImport(Lib, EntryPoint = "cuStreamCreate")]
    public static partial CUresult StreamCreate(out CUstream stream, uint flags);

    [LibraryImport(Lib, EntryPoint = "cuStreamSynchronize")]
    public static partial CUresult StreamSynchronize(CUstream stream);

    [LibraryImport(Lib, EntryPoint = "cuStreamDestroy_v2")]
    public static partial CUresult StreamDestroy(CUstream stream);

    [LibraryImport(Lib, EntryPoint = "cuEventCreate")]
    public static partial CUresult EventCreate(out CUevent ev, uint flags);

    [LibraryImport(Lib, EntryPoint = "cuEventRecord")]
    public static partial CUresult EventRecord(CUevent ev, CUstream stream);

    [LibraryImport(Lib, EntryPoint = "cuEventSynchronize")]
    public static partial CUresult EventSynchronize(CUevent ev);

    [LibraryImport(Lib, EntryPoint = "cuEventElapsedTime")]
    public static partial CUresult EventElapsedTime(out float milliseconds, CUevent start, CUevent end);

    [LibraryImport(Lib, EntryPoint = "cuEventDestroy_v2")]
    public static partial CUresult EventDestroy(CUevent ev);

    [LibraryImport(Lib, EntryPoint = "cuMemcpyHtoD_v2")]
    public static partial CUresult MemcpyHtoD(CUdeviceptr dst, nint src, nuint bytesize);

    [LibraryImport(Lib, EntryPoint = "cuMemcpyDtoH_v2")]
    public static partial CUresult MemcpyDtoH(nint dst, CUdeviceptr src, nuint bytesize);

    [LibraryImport(Lib, EntryPoint = "cuMemcpyHtoDAsync_v2")]
    public static partial CUresult MemcpyHtoDAsync(CUdeviceptr dst, nint src, nuint bytesize, CUstream stream);

    [LibraryImport(Lib, EntryPoint = "cuMemcpyDtoHAsync_v2")]
    public static partial CUresult MemcpyDtoHAsync(nint dst, CUdeviceptr src, nuint bytesize, CUstream stream);

    [LibraryImport(Lib, EntryPoint = "cuMemsetD8_v2")]
    public static partial CUresult MemsetD8(CUdeviceptr dst, byte value, nuint count);

    // Stream-ordered variant: enqueues the zero on `stream` rather than
    // blocking all streams. Safe to use instead of MemsetD8 when the
    // same stream is used for all subsequent kernel launches that read
    // the zeroed memory (stream FIFO ordering guarantees the zero lands
    // before the first dependent kernel executes).
    [LibraryImport(Lib, EntryPoint = "cuMemsetD8Async")]
    public static partial CUresult MemsetD8Async(CUdeviceptr dst, byte value, nuint count, CUstream stream);

    [LibraryImport(Lib, EntryPoint = "cuCtxSetCurrent")]
    public static partial CUresult CtxSetCurrent(CUcontext ctx);

    [LibraryImport(Lib, EntryPoint = "cuDevicePrimaryCtxRetain")]
    public static partial CUresult DevicePrimaryCtxRetain(out CUcontext ctx, CUdevice device);

    [LibraryImport(Lib, EntryPoint = "cuDevicePrimaryCtxSetFlags_v2")]
    public static partial CUresult DevicePrimaryCtxSetFlags(CUdevice device, uint flags);

    public const uint CTX_SCHED_BLOCKING_SYNC = 0x04;

    [LibraryImport(Lib, EntryPoint = "cuDevicePrimaryCtxRelease_v2")]
    public static partial CUresult DevicePrimaryCtxRelease(CUdevice device);

    // Pinned host memory. Returned pointer is a host pointer (nint),
    // not a device pointer — pass it as host_signal_header_pinned to
    // pearl_capi_noisy_gemm.
    [LibraryImport(Lib, EntryPoint = "cuMemHostAlloc")]
    public static partial CUresult MemHostAlloc(out nint pp, nuint bytesize, uint flags);

    [LibraryImport(Lib, EntryPoint = "cuMemFreeHost")]
    public static partial CUresult MemFreeHost(nint p);

    [LibraryImport(Lib, EntryPoint = "cuMemGetInfo_v2")]
    public static partial CUresult MemGetInfo(out nuint free, out nuint total);

    [LibraryImport(Lib, EntryPoint = "cuDriverGetVersion")]
    public static partial CUresult DriverGetVersion(out int version);

    [LibraryImport(Lib, EntryPoint = "cuDeviceTotalMem_v2")]
    public static partial CUresult DeviceTotalMem(out nuint bytes, CUdevice device);

    [LibraryImport(Lib, EntryPoint = "cuDeviceGetAttribute")]
    public static partial CUresult DeviceGetAttribute(out int pi, int attribute, CUdevice device);

    [LibraryImport(Lib, EntryPoint = "cuDeviceGetPCIBusId", StringMarshalling = StringMarshalling.Utf8)]
    public static partial CUresult DeviceGetPCIBusId(Span<byte> buf, int len, CUdevice device);

    // CUdevice_attribute selectors used by Akoya.Miner.SweepBench.
    public const int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT      = 16;
    public const int CU_DEVICE_ATTRIBUTE_CLOCK_RATE                = 13;  // kHz
    public const int CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE         = 36;  // kHz
    public const int CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH   = 37;  // bits

    public const uint CU_MEMHOSTALLOC_PORTABLE = 0x01;
    public const uint CU_MEMHOSTALLOC_DEVICEMAP = 0x02;
    public const uint CU_MEMHOSTALLOC_WRITECOMBINED = 0x04;

    public static void Check(CUresult r, string op)
    {
        if (r != CUresult.Success)
            throw new InvalidOperationException($"CUDA driver call failed ({op}): {r}");
    }
}
