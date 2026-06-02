// Akoya.PearlGemm — P/Invoke surface for the pearl-gemm C-ABI shim.
//
// All entry points return int status (0 = success, <0 = error). All raw
// pointers are device pointers (CUdeviceptr / nint).

using System.Runtime.InteropServices;

namespace Akoya.PearlGemm;

public static partial class PearlGemmNative
{
    public const string Lib = "pearl_gemm_capi";

    [LibraryImport(Lib, EntryPoint = "pearl_capi_abi_version")]
    public static partial int AbiVersion();

    [LibraryImport(Lib, EntryPoint = "pearl_capi_build_profile")]
    public static partial nint BuildProfilePtr();

    public static string BuildProfile()
        => Marshal.PtrToStringUTF8(BuildProfilePtr()) ?? "unknown";

    [LibraryImport(Lib, EntryPoint = "pearl_capi_supports_sm")]
    public static partial int SupportsSm(int major, int minor);

    [LibraryImport(Lib, EntryPoint = "pearl_capi_get_host_signal_sync_size")]
    public static partial int GetHostSignalSyncSize();

    [LibraryImport(Lib, EntryPoint = "pearl_capi_get_host_signal_header_size")]
    public static partial int GetHostSignalHeaderSize();

    [LibraryImport(Lib, EntryPoint = "pearl_capi_get_required_scratchpad_bytes")]
    public static partial long GetRequiredScratchpadBytes(long matrixBytes, int threadsPerBlock);

    // tensor_hash: data, key, out, roots are all device pointers.
    [LibraryImport(Lib, EntryPoint = "pearl_capi_tensor_hash")]
    public static partial int TensorHash(
        nint data, uint dataSize,
        nint outHash,
        nint key,
        uint numBlocks,
        uint threadsPerBlock,
        uint numStages,
        uint leavesPerMtBlock,
        nint roots,
        int deviceId,
        nint stream);

    [LibraryImport(Lib, EntryPoint = "pearl_capi_tensor_hash_leaf_cvs")]
    public static partial int TensorHashLeafCvs(
        nint data, uint dataSize,
        nint outHash,
        nint key,
        uint numBlocks,
        uint threadsPerBlock,
        uint numStages,
        uint leavesPerMtBlock,
        nint roots,
        nint leafCvs,
        int deviceId,
        nint stream);

    // Fused BSeed expansion + tensor_hash for B-state install. bSeed is a
    // pinned host pointer; data/outHash/key/roots are device pointers.
    [LibraryImport(Lib, EntryPoint = "pearl_capi_bseed_expand_and_tensor_hash")]
    public static unsafe partial int BSeedExpandAndTensorHash(
        byte* bSeed,
        nint data,
        uint dataSize,
        nint outHash,
        nint key,
        uint numBlocks,
        uint threadsPerBlock,
        uint numStages,
        uint leavesPerMtBlock,
        nint roots,
        int deviceId,
        nint stream);

    [LibraryImport(Lib, EntryPoint = "pearl_capi_bseed_expand_and_tensor_hash_leaf_cvs")]
    public static unsafe partial int BSeedExpandAndTensorHashLeafCvs(
        byte* bSeed,
        nint data,
        uint dataSize,
        nint outHash,
        nint key,
        uint numBlocks,
        uint threadsPerBlock,
        uint numStages,
        uint leavesPerMtBlock,
        nint roots,
        nint leafCvs,
        int deviceId,
        nint stream);

    [LibraryImport(Lib, EntryPoint = "pearl_capi_commitment_hash_from_merkle_roots")]
    public static partial int CommitmentHashFromMerkleRoots(
        nint aMerkleRoot, nint bMerkleRoot,
        nint key,
        nint aCommitmentHash, nint bCommitmentHash,
        int deviceId,
        nint stream);

    [LibraryImport(Lib, EntryPoint = "pearl_capi_noise_gen")]
    public static partial int NoiseGen(
        int r,
        int m, int n, int k,
        nint eal, nint ealFp16,
        nint earRMajor, nint earKMajor,
        nint ebrRMajor, nint ebrKMajor,
        nint ebr, nint ebrFp16,
        nint keyA, nint keyB,
        nint stream);

    [StructLayout(LayoutKind.Sequential)]
    public struct NoiseBParams
    {
        public int N, K, R;
        public nint B, EAR_K_major, EBL_R_major, EBR, EARxBpEB, BpEB;
        // ABI v2: optional pre-allocated workspace handle (IntPtr.Zero =
        // fall back to per-call cudaMallocAsync inside the .so).
        public nint Workspace;
    }

    [LibraryImport(Lib, EntryPoint = "pearl_capi_noise_B")]
    public static unsafe partial int NoiseB(NoiseBParams* p, nint stream);

    [StructLayout(LayoutKind.Sequential)]
    public struct InstallBParams
    {
        public int M, N, K, R;
        public int ExpandBSeed;
        public uint ThNumBlocks;
        public uint ThThreads;
        public uint ThStages;
        public uint ThLeaves;
        public int DeviceId;

        public nint BSeed;
        public nint B;
        public nint BHash;
        public nint Key;
        public nint Roots;
        public nint AHash;
        public nint CommitA;
        public nint CommitB;
        public nint EAR_K_major;
        public nint EBL_R_major;
        public nint EBL_K_major;
        public nint EBR;
        public nint EBR_fp16;
        public nint EARxBpEB;
        public nint BpEB;
        public nint Workspace;
        public nint LeafCvs;
    }

    [LibraryImport(Lib, EntryPoint = "pearl_capi_install_B")]
    public static unsafe partial int InstallB(InstallBParams* p, nint stream);

    [StructLayout(LayoutKind.Sequential)]
    public struct NoisyGemmParams
    {
        public int M, N, K, R;
        public int BM, BN, BK, CM, CN;

        public nint A, B, EAL, EAL_fp16, EBR, EBR_fp16;
        public nint EAR_R_major, EBL_R_major, EAR_K_major, EBL_K_major;
        public nint AxEBL_fp16, EARxBpEB_fp16;
        public nint ApEA, BpEB, A_scales, B_scales, C;
        public nint HostSignalHeaderPinned, HostSignalSync;
        public nint PowTarget, PowKey;
        // ABI v2: optional pre-allocated workspace handle.
        public nint Workspace;
    }

    [LibraryImport(Lib, EntryPoint = "pearl_capi_noisy_gemm")]
    public static unsafe partial int NoisyGemm(NoisyGemmParams* p, nint stream);

    // ABI v2: per-σ workspace pool. Allocate once after noise_gen at
    // σ-refresh, pass the handle through every NoiseB / NoisyGemm call, free
    // on σ-rotation. Saves the per-iter cudaMallocAsync/Free pair inside the
    // portable noisy_gemm path (measured ~+10 % on RTX 3080 / 5090).
    [LibraryImport(Lib, EntryPoint = "pearl_capi_workspace_alloc")]
    public static unsafe partial int WorkspaceAlloc(
        int m, int n, int k, int r,
        int withNoiseA, int withNoiseB,
        nint* outWorkspace, nint stream);

    [LibraryImport(Lib, EntryPoint = "pearl_capi_workspace_free")]
    public static partial int WorkspaceFree(nint workspace, nint stream);

    // Deterministic int7 ([-63, +63]) device fill, keyed by (seedLo, seedHi).
    // Host replay lives in Akoya.Crypto.LcgInt7 — both are byte-identical so
    // proof-time A recovery does not need to keep snapshot buffers around.
    [LibraryImport(Lib, EntryPoint = "pearl_capi_lcg_int7_fill")]
    public static partial int LcgInt7Fill(nint dst, long n, ulong seedLo, ulong seedHi, nint stream);

    // BSeed XOF expansion directly into a device buffer. bSeed is a pinned
    // host pointer to the 32-byte seed; dst is a device pointer.
    [LibraryImport(Lib, EntryPoint = "pearl_capi_bseed_expand_raw_device")]
    public static unsafe partial int BSeedExpandRawDevice(byte* bSeed, nint dst, long n, nint stream);

    [LibraryImport(Lib, EntryPoint = "pearl_capi_bseed_expand_range_raw_device")]
    public static unsafe partial int BSeedExpandRangeRawDevice(
        byte* bSeed,
        ulong byteOffset,
        nint dst,
        long n,
        nint stream);

    // ── Per-σ constant cache — eliminates per-iter argument marshalling ──────
    //
    // Call WorkspaceInstallParams() ONCE after WorkspaceAlloc() and after all
    // device pointers are stable (i.e. at σ-install time). The workspace then
    // caches ALL constants so the per-iter hot path can use the minimal
    // Iter() call (4 args, 1 P/Invoke) instead of 5 calls × 40 args.
    //
    // WorkspaceParams mirrors PearlCapiWorkspaceParams in pearl_gemm_capi.h.
    // Must be [StructLayout(Sequential)] — passed by pointer to the C ABI.
    [StructLayout(LayoutKind.Sequential)]
    public struct WorkspaceParams
    {
        // Dimensions
        public int M, N, K, R;
        public int BM, BN, BK, CM, CN;

        // TensorHash constants (= TENSOR_HASH_THREADS/STAGES/LEAVES)
        public uint ThNumBlocks;   // = ceil(M*K / (ThThreads * 1024))
        public uint ThThreads;     // = 128
        public uint ThStages;      // = 2
        public uint ThLeaves;      // = 512

        // seed_hi for lcg_int7_fill (= σ seed, constant within σ lifetime)
        public ulong SigmaSeed;

        // Device pointers — content changes per-iter, pointer values are const
        public nint A, B, AHash, BHash, Key, Roots, CommitA, CommitB;
        public nint EAL, EAL_fp16, EBR, EBR_fp16;
        public nint EAR_R_major, EBL_R_major, EAR_K_major, EBL_K_major;
        public nint AxEBL_fp16, EARxBpEB_fp16;
        public nint ApEA, BpEB;
        public nint A_scales, B_scales, C;
        public nint HostSignalSync;   // device — dSync coordination block
        public nint PowTarget;        // device uint32[8]
        public nint PowKey;           // device uint32[8]
    }

    // Install constant per-σ params into the workspace.  Must be called before
    // the first Iter() call.  Safe to call again on σ-rotation.
    [LibraryImport(Lib, EntryPoint = "pearl_capi_workspace_install_params")]
    public static unsafe partial int WorkspaceInstallParams(nint workspace, WorkspaceParams* p);

    // Per-iteration hot path — replaces 5 separate CAPI calls per iter.
    // Internally: lcg_int7_fill → tensor_hash → commitment_hash →
    // noise_gen_A → noisy_gemm, reading all constants from the installed params.
    // Only seedLo (nonce counter) and hostSignalHeaderPinned (pinned host slot)
    // change between iterations.
    [LibraryImport(Lib, EntryPoint = "pearl_capi_iter")]
    public static partial int Iter(nint workspace, ulong seedLo, nint hostSignalHeaderPinned, nint stream);

    // Batched variant of Iter(): launches `count` consecutive nonces starting
    // at seedLoStart, using hostSignalHeaderPinnedBatch[i] as the pinned slot
    // for iter i. Reduces managed/native transition overhead in QueueBatch.
    [LibraryImport(Lib, EntryPoint = "pearl_capi_iter_batch")]
    public static unsafe partial int IterBatch(
        nint workspace,
        ulong seedLoStart,
        nint* hostSignalHeaderPinnedBatch,
        int count,
        nint stream);

    [LibraryImport(Lib, EntryPoint = "pearl_capi_iter_batch_graph_prepare")]
    public static unsafe partial int IterBatchGraphPrepare(
        nint workspace,
        nint* hostSignalHeaderPinnedBatch,
        int count,
        nint stream);

    [LibraryImport(Lib, EntryPoint = "pearl_capi_iter_batch_graph_launch")]
    public static partial int IterBatchGraphLaunch(
        nint workspace,
        ulong seedLoStart,
        nint stream);
}
