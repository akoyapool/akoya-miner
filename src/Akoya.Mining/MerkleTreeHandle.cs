// MerkleTreeHandle — owning wrapper over a pre-built Rust keyed-BLAKE3
// Merkle tree. Built once per σ install (the matrix is constant for the
// life of σ), then queried with the per-trigger column indices to produce
// inclusion proofs without re-hashing the entire matrix.
//
// Block-find latency motivation: ShareBuilder.Build was calling
// MerkleRootAndProof.Compute(bBytes, ...) on every trigger, which rebuilt
// the full 64 MiB B-tree each time (~130 ms measured). Caching the tree
// here cuts that to a few microseconds for proof extraction.

using System.Runtime.InteropServices;

namespace Akoya.Mining;

/// <summary>
/// Owning handle to a pre-built keyed-BLAKE3 Merkle tree living in the
/// Rust runtime. Refcounted so the per-σ owner (GpuWorker) can hand a
/// live reference to ShareFinalizer (which runs on a separate task and
/// may still be opening proofs against the tree when σ rotates).
///
/// Refcount semantics:
///   - Newly built handle has refcount = 1 (the builder's reference).
///   - <see cref="Acquire"/> increments; <see cref="Release"/>
///     decrements and frees the native handle when count reaches 0.
///   - <see cref="Dispose"/> is an alias for <see cref="Release"/>
///     so existing <c>using</c>/<c>?.Dispose()</c> sites still work
///     unchanged.
///   - The finalizer remains as a safety net for leaks.
/// </summary>
public sealed class MerkleTreeHandle : IMerkleTreeHandle, IDisposable
{
    private unsafe void* _handle;
    private int _refCount;

    /// <summary>The 32-byte root computed at build time.</summary>
    public byte[] Root { get; }

    /// <summary>Total number of 1024-byte leaves in the tree.</summary>
    public uint TotalLeaves { get; }

    private unsafe MerkleTreeHandle(void* handle, byte[] root, uint totalLeaves)
    {
        _handle = handle;
        Root = root;
        TotalLeaves = totalLeaves;
        _refCount = 1;
    }

    /// <summary>Increment the refcount. Must be paired with exactly one
    /// <see cref="Release"/> from the same logical owner.</summary>
    public void Acquire()
    {
        if (Interlocked.Increment(ref _refCount) <= 1)
        {
            // The pre-increment value was 0 or negative — the handle has
            // already been freed. Resurrection is not supported.
            Interlocked.Decrement(ref _refCount);
            throw new ObjectDisposedException(nameof(MerkleTreeHandle));
        }
    }

    /// <summary>Decrement the refcount; free the native handle when it
    /// reaches zero. Idempotent on an already-freed handle (no-op).</summary>
    public unsafe void Release()
    {
        int after = Interlocked.Decrement(ref _refCount);
        if (after > 0) return;
        if (after < 0)
        {
            // Defensive: refcount went negative. Restore to 0 and bail.
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
            PearlMiningNative.MerkleTreeFree(_handle);
            _handle = null;
        }
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Build the keyed-BLAKE3 Merkle tree over <paramref name="data"/> once.
    /// Equivalent to the build half of <see cref="MerkleRootAndProof.Compute"/>,
    /// but returns a reusable handle.
    /// </summary>
    public static unsafe MerkleTreeHandle Build(
        ReadOnlySpan<byte> data,
        ReadOnlySpan<byte> key,
        int rowWidth)
    {
        if (key.Length != 32) throw new ArgumentException("Key must be 32 bytes.", nameof(key));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rowWidth);

        void* handle = null;
        var root = new byte[32];
        uint totalLeaves = 0;
        byte* errMsg = null;
        int rc;

        try
        {
            fixed (byte* pData = data)
            fixed (byte* pKey = key)
            fixed (byte* pRoot = root)
            {
                rc = PearlMiningNative.MerkleBuildTree(
                    pData, (nuint)data.Length,
                    pKey,
                    (nuint)rowWidth,
                    &handle,
                    pRoot,
                    &totalLeaves,
                    &errMsg);
            }

            if (rc != 0)
            {
                var msg = errMsg != null ? Marshal.PtrToStringUTF8((IntPtr)errMsg) ?? "(no message)" : "(no message)";
                throw new MerkleRootAndProofException(rc, msg);
            }

            return new MerkleTreeHandle(handle, root, totalLeaves);
        }
        finally
        {
            if (errMsg != null) PearlMiningNative.FreeString(errMsg);
        }
    }

    /// <summary>
    /// Extract an inclusion proof for the given row indices. The handle is
    /// NOT consumed; call this as many times as you have distinct trigger
    /// index sets.
    /// </summary>
    public unsafe MerkleRootAndProofResult Proof(ReadOnlySpan<uint> rowIndices)
    {
        ObjectDisposedException.ThrowIf(_refCount <= 0 || _handle == null, this);
        if (rowIndices.IsEmpty) throw new ArgumentException("rowIndices must be non-empty.", nameof(rowIndices));

        byte* leafDataPtr = null; nuint leafCount = 0;
        uint* leafIdxPtr = null;  nuint leafIdxLen = 0;
        byte* siblingsPtr = null; nuint sibCount = 0;
        byte* errMsg = null;
        int rc;

        try
        {
            fixed (uint* pRows = rowIndices)
            {
                rc = PearlMiningNative.MerkleProofForHandle(
                    _handle,
                    pRows, (nuint)rowIndices.Length,
                    &leafDataPtr, &leafCount,
                    &leafIdxPtr, &leafIdxLen,
                    &siblingsPtr, &sibCount,
                    &errMsg);
            }

            if (rc != 0)
            {
                var msg = errMsg != null ? Marshal.PtrToStringUTF8((IntPtr)errMsg) ?? "(no message)" : "(no message)";
                throw new MerkleRootAndProofException(rc, msg);
            }

            var leafData = new byte[leafCount][];
            for (nuint i = 0; i < leafCount; i++)
            {
                var chunk = new byte[1024];
                Marshal.Copy((IntPtr)(leafDataPtr + i * 1024), chunk, 0, 1024);
                leafData[i] = chunk;
            }

            var leafIndices = new uint[leafIdxLen];
            if (leafIdxLen > 0)
            {
                new ReadOnlySpan<uint>(leafIdxPtr, (int)leafIdxLen).CopyTo(leafIndices);
            }

            var siblings = new byte[sibCount][];
            for (nuint i = 0; i < sibCount; i++)
            {
                var sib = new byte[32];
                Marshal.Copy((IntPtr)(siblingsPtr + i * 32), sib, 0, 32);
                siblings[i] = sib;
            }

            return new MerkleRootAndProofResult(Root, leafData, leafIndices, TotalLeaves, siblings);
        }
        finally
        {
            if (leafDataPtr != null) PearlMiningNative.FreeBuffer(leafDataPtr, leafCount * 1024);
            if (leafIdxPtr != null) PearlMiningNative.FreeU32Buffer(leafIdxPtr, leafIdxLen);
            if (siblingsPtr != null) PearlMiningNative.FreeBuffer(siblingsPtr, sibCount * 32);
            if (errMsg != null) PearlMiningNative.FreeString(errMsg);
        }
    }

    /// <summary>
    /// Open K independent leaf-major Merkle audit paths against this handle,
    /// formatted exactly as the audit_proof v1 wire field expects:
    /// <c>leafIndices.Length × levels × 32</c> contiguous bytes, leaf→root
    /// within each opening, openings in caller-supplied order. This is the
    /// fast-path equivalent of <c>AuditProofBuilder.Build</c> — produces a
    /// byte-identical output by reusing the keyed-BLAKE3 tree that was
    /// already built at σ install time (no rehashing of B).
    ///
    /// Latency-critical: block-find shares must not pay the managed
    /// builder's O(N) CV-table reconstruction cost; this routine is O(K log N).
    /// </summary>
    public unsafe byte[] AuditPaths(ReadOnlySpan<uint> leafIndices)
    {
        ObjectDisposedException.ThrowIf(_refCount <= 0 || _handle == null, this);

        if (leafIndices.IsEmpty) return [];

        byte* siblingsPtr = null; nuint sibBytes = 0;
        byte* errMsg = null;
        int rc;

        try
        {
            fixed (uint* pIdx = leafIndices)
            {
                rc = PearlMiningNative.MerkleAuditPathsForHandle(
                    _handle,
                    pIdx, (nuint)leafIndices.Length,
                    &siblingsPtr, &sibBytes,
                    &errMsg);
            }

            if (rc != 0)
            {
                var msg = errMsg != null ? Marshal.PtrToStringUTF8((IntPtr)errMsg) ?? "(no message)" : "(no message)";
                throw new MerkleRootAndProofException(rc, msg);
            }

            var siblings = new byte[(int)sibBytes];
            if (sibBytes > 0)
            {
                new ReadOnlySpan<byte>(siblingsPtr, (int)sibBytes).CopyTo(siblings);
            }
            return siblings;
        }
        finally
        {
            if (siblingsPtr != null) PearlMiningNative.FreeBuffer(siblingsPtr, sibBytes);
            if (errMsg != null) PearlMiningNative.FreeString(errMsg);
        }
    }

    /// <summary>Alias for <see cref="Release"/> so existing
    /// <c>using</c>/<c>?.Dispose()</c> call sites work unchanged.</summary>
    /// <remarks>CA1816 (call <c>GC.SuppressFinalize</c> here) does not apply
    /// to refcounted handles: when refcount &gt; 1 the native resource is
    /// still live and the finalizer must remain registered to catch leaks.
    /// <see cref="Release"/> only suppresses the finalizer once the count
    /// drops to zero and the native handle has actually been freed.</remarks>
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Usage", "CA1816:Dispose methods should call SuppressFinalize",
        Justification = "Refcounted; SuppressFinalize happens inside Release() only on the last decrement.")]
    public void Dispose() => Release();

    ~MerkleTreeHandle()
    {
        // Safety net for leaks: if the refcount machinery was bypassed
        // (e.g. an Acquire without a paired Release), the GC finalizer
        // still frees the native handle.
        unsafe { if (_handle != null) PearlMiningNative.MerkleTreeFree(_handle); }
    }
}
