using System.Runtime.InteropServices;
using Akoya.Crypto;
using Akoya.Mining;

namespace Akoya.Miner.Mining;

internal sealed class NativeBSeedMerkleTreeHandle : IMerkleTreeHandle, IDisposable
{
    private unsafe void* _handle;
    private int _refCount;

    public byte[] Root { get; }
    public uint TotalLeaves { get; }

    private unsafe NativeBSeedMerkleTreeHandle(void* handle, byte[] root, uint totalLeaves)
    {
        _handle = handle;
        Root = root;
        TotalLeaves = totalLeaves;
        _refCount = 1;
    }

    public static unsafe NativeBSeedMerkleTreeHandle BuildFromLeafCvs(
        ReadOnlySpan<byte> leafCvs,
        ReadOnlySpan<byte> bSeed,
        ReadOnlySpan<byte> key,
        int numRows,
        int rowWidth)
    {
        if (bSeed.Length != SigmaContext.BSeedSize) throw new ArgumentException("BSeed must be 32 bytes.", nameof(bSeed));
        if (key.Length != Blake3.DigestSize) throw new ArgumentException("Key must be 32 bytes.", nameof(key));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numRows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rowWidth);
        if (leafCvs.IsEmpty || leafCvs.Length % Blake3.DigestSize != 0)
            throw new ArgumentException("leafCvs must be a non-empty multiple of 32 bytes.", nameof(leafCvs));

        void* handle = null;
        var root = new byte[Blake3.DigestSize];
        uint totalLeaves = 0;
        byte* errMsg = null;

        try
        {
            fixed (byte* pLeafCvs = leafCvs)
            fixed (byte* pBSeed = bSeed)
            fixed (byte* pKey = key)
            fixed (byte* pRoot = root)
            {
                int rc = PearlMiningNative.BSeedMerkleBuildTreeFromLeafCvs(
                    pLeafCvs,
                    (nuint)leafCvs.Length,
                    pBSeed,
                    pKey,
                    (nuint)numRows,
                    (nuint)rowWidth,
                    &handle,
                    pRoot,
                    &totalLeaves,
                    &errMsg);

                if (rc != 0)
                {
                    var msg = errMsg != null ? Marshal.PtrToStringUTF8((IntPtr)errMsg) ?? "(no message)" : "(no message)";
                    throw new MerkleRootAndProofException(rc, msg);
                }
            }

            return new NativeBSeedMerkleTreeHandle(handle, root, totalLeaves);
        }
        finally
        {
            if (errMsg != null) PearlMiningNative.FreeString(errMsg);
        }
    }

    public void Acquire()
    {
        if (Interlocked.Increment(ref _refCount) <= 1)
        {
            Interlocked.Decrement(ref _refCount);
            throw new ObjectDisposedException(nameof(NativeBSeedMerkleTreeHandle));
        }
    }

    public unsafe void Release()
    {
        int after = Interlocked.Decrement(ref _refCount);
        if (after > 0) return;
        if (after < 0)
        {
            Interlocked.Exchange(ref _refCount, 0);
            return;
        }
        FreeNative();
    }

    [System.Diagnostics.CodeAnalysis.SuppressMessage("Usage", "CA1816:Dispose methods should call SuppressFinalize",
        Justification = "Refcounted; SuppressFinalize happens inside FreeNative() on the final decrement only.")]
    private unsafe void FreeNative()
    {
        if (_handle != null)
        {
            PearlMiningNative.BSeedMerkleTreeFree(_handle);
            _handle = null;
        }
        GC.SuppressFinalize(this);
    }

    public unsafe MerkleRootAndProofResult Proof(ReadOnlySpan<uint> rowIndices)
    {
        ObjectDisposedException.ThrowIf(_refCount <= 0 || _handle == null, this);
        if (rowIndices.IsEmpty) throw new ArgumentException("rowIndices must be non-empty.", nameof(rowIndices));

        byte* leafDataPtr = null; nuint leafCount = 0;
        uint* leafIdxPtr = null; nuint leafIdxLen = 0;
        byte* siblingsPtr = null; nuint sibCount = 0;
        byte* errMsg = null;

        try
        {
            fixed (uint* pRows = rowIndices)
            {
                int rc = PearlMiningNative.BSeedMerkleProofForHandle(
                    _handle,
                    pRows, (nuint)rowIndices.Length,
                    &leafDataPtr, &leafCount,
                    &leafIdxPtr, &leafIdxLen,
                    &siblingsPtr, &sibCount,
                    &errMsg);

                if (rc != 0)
                {
                    var msg = errMsg != null ? Marshal.PtrToStringUTF8((IntPtr)errMsg) ?? "(no message)" : "(no message)";
                    throw new MerkleRootAndProofException(rc, msg);
                }
            }

            var leafData = new byte[leafCount][];
            for (nuint i = 0; i < leafCount; i++)
            {
                var chunk = new byte[Blake3.ChunkLen];
                Marshal.Copy((IntPtr)(leafDataPtr + i * Blake3.ChunkLen), chunk, 0, Blake3.ChunkLen);
                leafData[i] = chunk;
            }

            var leafIndices = new uint[leafIdxLen];
            if (leafIdxLen > 0)
                new ReadOnlySpan<uint>(leafIdxPtr, (int)leafIdxLen).CopyTo(leafIndices);

            var siblings = new byte[sibCount][];
            for (nuint i = 0; i < sibCount; i++)
            {
                var sib = new byte[Blake3.DigestSize];
                Marshal.Copy((IntPtr)(siblingsPtr + i * Blake3.DigestSize), sib, 0, Blake3.DigestSize);
                siblings[i] = sib;
            }

            return new MerkleRootAndProofResult(Root, leafData, leafIndices, TotalLeaves, siblings);
        }
        finally
        {
            if (leafDataPtr != null) PearlMiningNative.FreeBuffer(leafDataPtr, leafCount * Blake3.ChunkLen);
            if (leafIdxPtr != null) PearlMiningNative.FreeU32Buffer(leafIdxPtr, leafIdxLen);
            if (siblingsPtr != null) PearlMiningNative.FreeBuffer(siblingsPtr, sibCount * Blake3.DigestSize);
            if (errMsg != null) PearlMiningNative.FreeString(errMsg);
        }
    }

    public unsafe byte[] AuditPaths(ReadOnlySpan<uint> leafIndices)
    {
        ObjectDisposedException.ThrowIf(_refCount <= 0 || _handle == null, this);
        if (leafIndices.IsEmpty) return [];

        byte* siblingsPtr = null; nuint sibBytes = 0;
        byte* errMsg = null;

        try
        {
            fixed (uint* pIdx = leafIndices)
            {
                int rc = PearlMiningNative.BSeedMerkleAuditPathsForHandle(
                    _handle,
                    pIdx, (nuint)leafIndices.Length,
                    &siblingsPtr, &sibBytes,
                    &errMsg);

                if (rc != 0)
                {
                    var msg = errMsg != null ? Marshal.PtrToStringUTF8((IntPtr)errMsg) ?? "(no message)" : "(no message)";
                    throw new MerkleRootAndProofException(rc, msg);
                }
            }

            var siblings = new byte[(int)sibBytes];
            if (sibBytes > 0)
                new ReadOnlySpan<byte>(siblingsPtr, (int)sibBytes).CopyTo(siblings);
            return siblings;
        }
        finally
        {
            if (siblingsPtr != null) PearlMiningNative.FreeBuffer(siblingsPtr, sibBytes);
            if (errMsg != null) PearlMiningNative.FreeString(errMsg);
        }
    }

    [System.Diagnostics.CodeAnalysis.SuppressMessage("Usage", "CA1816:Dispose methods should call SuppressFinalize",
        Justification = "Refcounted; Release() suppresses finalization only on the final decrement.")]
    public void Dispose() => Release();

    ~NativeBSeedMerkleTreeHandle()
    {
        unsafe { if (_handle != null) PearlMiningNative.BSeedMerkleTreeFree(_handle); }
    }
}
