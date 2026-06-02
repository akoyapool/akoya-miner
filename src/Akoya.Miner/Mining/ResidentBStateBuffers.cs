using System.Runtime.InteropServices;
using Akoya.Crypto;
using Akoya.Cuda;
using Akoya.PearlGemm;

namespace Akoya.Miner.Mining;

internal sealed class ResidentBStateBuffers : IDisposable
{
    private const uint TensorHashThreads = 128;
    private readonly CUstream _stream;
    private bool _disposed;

    public int N { get; }
    public int K { get; }
    public int R { get; }

    public CUdeviceptr B         { get; }
    public CUdeviceptr Key       { get; }
    public CUdeviceptr BHash     { get; }
    public CUdeviceptr Roots     { get; }
    public CUdeviceptr LeafCvs   { get; }
    public CUdeviceptr EAR_K     { get; }
    public CUdeviceptr EBL_R     { get; }
    public CUdeviceptr EBL_K     { get; }
    public CUdeviceptr EBR       { get; }
    public CUdeviceptr EBRFp16   { get; }
    public CUdeviceptr EARxBpEB  { get; }
    public CUdeviceptr BpEB      { get; }
    public CUdeviceptr BScales   { get; }
    public nint NoiseWorkspace   { get; private set; }
    public long LeafCvBytes      { get; }
    public bool BUploaded        { get; set; }

    public ResidentBStateBuffers(int n, int k, int r, CUstream stream)
    {
        if (n <= 0 || k <= 0 || r <= 0)
            throw new ArgumentOutOfRangeException(nameof(n), "N/K/R must all be > 0");

        N = n;
        K = k;
        R = r;
        _stream = stream;

        long bB          = (long)N * K;
        long bEAR_K      = (long)R * K;
        long bEBL_R      = (long)K * R;
        long bEBL_K      = (long)R * K;
        long bEBR        = (long)N * R;
        long bEARxBpEB16 = (long)N * R * 2;
        long rootsBytes  = PearlGemmNative.GetRequiredScratchpadBytes(bB, (int)TensorHashThreads);
        LeafCvBytes = ((bB + Blake3.ChunkLen - 1) / Blake3.ChunkLen) * Blake3.DigestSize;

        B         = Alloc(bB);
        Key       = Alloc(32);
        BHash     = AllocZero(32);
        Roots     = AllocZero(rootsBytes);
        LeafCvs   = AllocZero(LeafCvBytes);
        EAR_K     = AllocZero(bEAR_K);
        EBL_R     = AllocZero(bEBL_R);
        EBL_K     = AllocZero(bEBL_K);
        EBR       = AllocZero(bEBR);
        EBRFp16   = AllocZero(bEBR * 2);
        EARxBpEB  = AllocZero(bEARxBpEB16);
        BpEB      = AllocZero(bB);
        BScales   = AllocFp32Ones(N);

        unsafe
        {
            nint ws = IntPtr.Zero;
            Check("resident_b_workspace_alloc", PearlGemmNative.WorkspaceAlloc(
                m: 0,
                n: N,
                k: K,
                r: R,
                withNoiseA: 0,
                withNoiseB: 1,
                outWorkspace: &ws,
                stream: stream.Handle));
            NoiseWorkspace = ws;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (NoiseWorkspace != IntPtr.Zero)
        {
            try { _ = PearlGemmNative.WorkspaceFree(NoiseWorkspace, _stream.Handle); } catch { /* shutdown */ }
            NoiseWorkspace = IntPtr.Zero;
        }

        TryFree(B);        TryFree(Key);
        TryFree(BHash);    TryFree(Roots);
        TryFree(LeafCvs);
        TryFree(EAR_K);    TryFree(EBL_R);
        TryFree(EBL_K);    TryFree(EBR);
        TryFree(EBRFp16);  TryFree(EARxBpEB);
        TryFree(BpEB);     TryFree(BScales);
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

    private static void Check(string op, int rc)
    {
        if (rc != 0) throw new InvalidOperationException($"{op} failed rc={rc}");
    }
}
