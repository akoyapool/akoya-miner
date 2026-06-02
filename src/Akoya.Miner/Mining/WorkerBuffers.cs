// WorkerBuffers — owns every CUDA device + pinned-host allocation a single
// GpuWorker needs for the mining loop. Lifted out of v1 MineBlocks.cs
// (lines ~609–653 and the Alloc helpers at ~1587–1602) so the buffer
// management has a name and a single Dispose path.
//
// Sizing comes from MineOptions (M/N/K/NoiseRank, MatmulsPerPoll) and is
// stable across σ rotations; allocations happen once per worker startup.
//
// All allocations are device-side except `HostHeaders[]` which is pinned host
// memory the GPU writes outcome headers into via the host-signal mechanism.

using System.Runtime.InteropServices;
using Akoya.Crypto;
using Akoya.Cuda;
using Akoya.PearlGemm;

namespace Akoya.Miner.Mining;

internal sealed class WorkerBuffers : IDisposable
{
    private const uint TENSOR_HASH_THREADS = 128;

    private readonly nint[] _hostHeaders;
    private nint _hostAPtr;
    private nint _hostALeafCvsPtr;
    private nint _hostASelectedPtr;
    private bool _disposed;

    public int M { get; }
    public int N { get; }
    public int K { get; }
    public int R { get; }
    public int MatmulsPerPoll { get; }
    public int HeaderSize { get; }
    public int SyncSize { get; }

    /// <summary>Pinned host buffer sized M×K — destination for the trigger-time
    /// D2H of A. Pre-pinned at construction so cuMemcpyDtoHAsync DMAs directly
    /// (≈1–2 ms at PCIe Gen4 for 16 MiB) instead of staging a 16 MiB managed
    /// pageable byte[] through CUDA's internal bounce buffer (~37 ms).
    /// Caller is responsible for copying out to its own owned buffer before
    /// the next D2H overwrites the slot.</summary>
    public nint HostAPtr => _hostAPtr;
    public long HostASize => (long)M * K;
    public nint HostALeafCvsPtr => _hostALeafCvsPtr;
    public long HostALeafCvsSize { get; }
    public nint HostASelectedPtr => _hostASelectedPtr;
    public long HostASelectedSize { get; }

    public CUdeviceptr A         { get; }
    public CUdeviceptr B         { get; }
    public CUdeviceptr Key       { get; }   // 32 B BLAKE3 keyed-hash key (jobKey)
    public CUdeviceptr EAL       { get; }
    public CUdeviceptr EALFp16   { get; }
    public CUdeviceptr EAR_R     { get; }
    public CUdeviceptr EAR_K     { get; }
    public CUdeviceptr EBL_R     { get; }
    public CUdeviceptr EBL_K     { get; }
    public CUdeviceptr EBR       { get; }
    public CUdeviceptr EBRFp16   { get; }
    public CUdeviceptr AxEBLFp16 { get; }
    public CUdeviceptr EARxBpEB  { get; }
    public CUdeviceptr ApEA      { get; }
    public CUdeviceptr BpEB      { get; }
    public CUdeviceptr AScales   { get; }   // FP32 ones for cuBLAS scale factor.
    public CUdeviceptr BScales   { get; }
    public CUdeviceptr C         { get; }   // Intentionally null in pure-miner mode; C stores are skipped.
    public CUdeviceptr PowTarget { get; }   // 8×u32 little-endian
    public CUdeviceptr Roots     { get; }   // tensor-hash scratchpad
    public CUdeviceptr ALeafCvs  { get; }   // 32 B per 1024 B A leaf
    public CUdeviceptr Sync      { get; }   // host-signal sync block
    public CUdeviceptr AHash     { get; }   // 32 B Merkle root of A
    public CUdeviceptr BHash     { get; }
    public CUdeviceptr CommitA   { get; }   // 32 B keyed commit (A)
    public CUdeviceptr CommitB   { get; }

    /// <summary>Per-iteration pinned host headers (mpp deep). GPU writes the
    /// host-signal header here; the CPU polls it without a sync.</summary>
    public ReadOnlySpan<nint> HostHeaders => _hostHeaders;

    public WorkerBuffers(int m, int n, int k, int noiseRank, int matmulsPerPoll)
    {
        if (m <= 0 || n <= 0 || k <= 0 || noiseRank <= 0 || matmulsPerPoll <= 0)
            throw new ArgumentOutOfRangeException(nameof(m), "M/N/K/R/MPP must all be > 0");

        M = m; N = n; K = k; R = noiseRank; MatmulsPerPoll = matmulsPerPoll;

        long bA          = (long)M * K;
        long bB          = (long)N * K;
        long bEAL        = (long)M * R;
        long bEAR_R      = (long)K * R;
        long bEAR_K      = (long)R * K;
        long bEBL_R      = (long)K * R;
        long bEBL_K      = (long)R * K;
        long bEBR        = (long)N * R;
        long bAxEBLFp16  = (long)M * R * 2;
        long bEARxBpEB16 = (long)N * R * 2;
        long aLeafCvsBytes = ((bA + Blake3.ChunkLen - 1) / Blake3.ChunkLen) * Blake3.DigestSize;
        long aSelectedBytes = (long)MiningConfiguration.DefaultRowsIndices.Length * K;

        SyncSize   = PearlGemmNative.GetHostSignalSyncSize();
        HeaderSize = PearlGemmNative.GetHostSignalHeaderSize();
        long rootsBytes = PearlGemmNative.GetRequiredScratchpadBytes(
            Math.Max(bA, bB), (int)TENSOR_HASH_THREADS);
        HostALeafCvsSize = aLeafCvsBytes;
        HostASelectedSize = aSelectedBytes;

        A         = Alloc(bA);
        B         = Alloc(bB);
        Key       = Alloc(32);
        EAL       = AllocZero(bEAL);
        EALFp16   = AllocZero(bEAL * 2);
        EAR_R     = AllocZero(bEAR_R);
        EAR_K     = AllocZero(bEAR_K);
        EBL_R     = AllocZero(bEBL_R);
        EBL_K     = AllocZero(bEBL_K);
        EBR       = AllocZero(bEBR);
        EBRFp16   = AllocZero(bEBR * 2);
        AxEBLFp16 = AllocZero(bAxEBLFp16);
        EARxBpEB  = AllocZero(bEARxBpEB16);
        ApEA      = AllocZero(bA);
        BpEB      = AllocZero(bB);
        AScales   = AllocFp32Ones(M);
        BScales   = AllocFp32Ones(N);
        // The pure-miner CAPI path is headless and passes C=nullptr into the
        // transcript GEMM. Keeping this null avoids a dead M*N*bf16 allocation
        // per half (512 MiB at 8192x32768, ~1 GiB per GPU worker).
        C         = default;
        PowTarget = Alloc(8 * 4);
        Roots     = AllocZero(rootsBytes);
        ALeafCvs  = AllocZero(aLeafCvsBytes);
        Sync      = AllocZero(SyncSize);
        AHash     = AllocZero(32);
        BHash     = AllocZero(32);
        CommitA   = AllocZero(32);
        CommitB   = AllocZero(32);

        _hostHeaders = new nint[MatmulsPerPoll];
        for (int i = 0; i < MatmulsPerPoll; i++)
        {
            CudaDriver.Check(
                CudaDriver.MemHostAlloc(out _hostHeaders[i], (nuint)HeaderSize,
                    CudaDriver.CU_MEMHOSTALLOC_PORTABLE),
                "host header alloc");
            unsafe { new Span<byte>((void*)_hostHeaders[i], HeaderSize).Clear(); }
        }

        // Pinned host scratch for the trigger-time D2H of A.
        CudaDriver.Check(
            CudaDriver.MemHostAlloc(out _hostAPtr, (nuint)bA,
                CudaDriver.CU_MEMHOSTALLOC_PORTABLE),
            "host A pinned alloc");
        CudaDriver.Check(
            CudaDriver.MemHostAlloc(out _hostALeafCvsPtr, (nuint)aLeafCvsBytes,
                CudaDriver.CU_MEMHOSTALLOC_PORTABLE),
            "host A leaf CV pinned alloc");
        CudaDriver.Check(
            CudaDriver.MemHostAlloc(out _hostASelectedPtr, (nuint)aSelectedBytes,
                CudaDriver.CU_MEMHOSTALLOC_PORTABLE),
            "host A selected pinned alloc");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        // Device buffers — fire-and-forget the frees: we want every buffer
        // released even if one fails. Best-effort by design.
        TryFree(A);          TryFree(B);          TryFree(Key);
        TryFree(EAL);        TryFree(EALFp16);
        TryFree(EAR_R);      TryFree(EAR_K);
        TryFree(EBL_R);      TryFree(EBL_K);
        TryFree(EBR);        TryFree(EBRFp16);
        TryFree(AxEBLFp16);  TryFree(EARxBpEB);
        TryFree(ApEA);       TryFree(BpEB);
        TryFree(AScales);    TryFree(BScales);
        TryFree(C);          TryFree(PowTarget);
        TryFree(Roots);      TryFree(ALeafCvs);
        TryFree(Sync);
        TryFree(AHash);      TryFree(BHash);
        TryFree(CommitA);    TryFree(CommitB);

        for (int i = 0; i < _hostHeaders.Length; i++)
        {
            if (_hostHeaders[i] != nint.Zero)
            {
                try { CudaDriver.MemFreeHost(_hostHeaders[i]); } catch { /* shutdown */ }
                _hostHeaders[i] = nint.Zero;
            }
        }

        if (_hostAPtr != nint.Zero)
        {
            try { CudaDriver.MemFreeHost(_hostAPtr); } catch { /* shutdown */ }
            _hostAPtr = nint.Zero;
        }

        if (_hostALeafCvsPtr != nint.Zero)
        {
            try { CudaDriver.MemFreeHost(_hostALeafCvsPtr); } catch { /* shutdown */ }
            _hostALeafCvsPtr = nint.Zero;
        }

        if (_hostASelectedPtr != nint.Zero)
        {
            try { CudaDriver.MemFreeHost(_hostASelectedPtr); } catch { /* shutdown */ }
            _hostASelectedPtr = nint.Zero;
        }
    }

    private static CUdeviceptr Alloc(long bytes)
    {
        CudaDriver.Check(CudaDriver.MemAlloc(out var p, (nuint)bytes), "MemAlloc");
        return p;
    }

    private static CUdeviceptr AllocZero(long bytes)
    {
        var p = Alloc(bytes);
        CudaDriver.Check(CudaDriver.MemsetD8(p, 0, (nuint)bytes), "MemsetD8");
        return p;
    }

    private static CUdeviceptr AllocFp32Ones(int n)
    {
        var p = Alloc((long)n * 4);
        var host = new float[n];
        Array.Fill(host, 1.0f);
        var bytes = MemoryMarshal.AsBytes<float>(host).ToArray();
        unsafe
        {
            fixed (byte* src = bytes)
                CudaDriver.Check(
                    CudaDriver.MemcpyHtoD(p, (nint)src, (nuint)bytes.Length),
                    "MemcpyHtoD ones");
        }
        return p;
    }

    private static void TryFree(CUdeviceptr p)
    {
        if (p.Handle == nint.Zero) return;
        try { CudaDriver.MemFree(p); } catch { /* shutdown best-effort */ }
    }
}
