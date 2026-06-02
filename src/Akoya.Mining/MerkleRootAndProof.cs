// MerkleRootAndProof — managed wrapper over pearl_capi_merkle_root_and_proof.
//
// Fuses what used to be three independent scans over the same 32-128 MB matrix
// (one keyed-BLAKE3 Merkle root + one Merkle tree build for the A proof + one
// for the B proof) into a single Rust call per matrix that builds the tree
// once and emits both outputs. The Rust BLAKE3 has AVX2/AVX-512 SIMD; the C#
// implementation it replaces was hand-rolled scalar reference code.

using System.Runtime.InteropServices;

namespace Akoya.Mining;

public sealed class MerkleRootAndProofException : Exception
{
    public int Code { get; }
    public MerkleRootAndProofException(int code, string msg)
        : base($"pearl_capi_merkle_root_and_proof failed (code={code}): {msg}")
        => Code = code;
}

/// <summary>
/// Result of <see cref="MerkleRootAndProof.Compute"/>. Field shapes match the
/// pool wire format (leaf chunks 1024 B
/// each, siblings 32 B each).
/// </summary>
public sealed record MerkleRootAndProofResult(
    byte[]   Root,
    byte[][] LeafData,
    uint[]   LeafIndices,
    uint     TotalLeaves,
    byte[][] Siblings);

public static class MerkleRootAndProof
{
    /// <summary>
    /// Build the keyed BLAKE3 Merkle tree over <paramref name="data"/> once
    /// and return both the root and an inclusion proof for the matrix rows
    /// listed in <paramref name="rowIndices"/>.
    /// </summary>
    public static unsafe MerkleRootAndProofResult Compute(
        ReadOnlySpan<byte> data,
        ReadOnlySpan<byte> key,
        ReadOnlySpan<uint> rowIndices,
        int rowWidth)
    {
        if (key.Length != 32) throw new ArgumentException("Key must be 32 bytes.", nameof(key));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rowWidth);
        if (rowIndices.IsEmpty) throw new ArgumentException("rowIndices must be non-empty.", nameof(rowIndices));

        var root = new byte[32];
        uint totalLeaves = 0;
        byte* leafDataPtr = null;     nuint leafCount = 0;
        uint* leafIdxPtr  = null;     nuint leafIdxLen = 0;
        byte* siblingsPtr = null;     nuint sibCount = 0;
        byte* errMsg      = null;
        int rc;

        try
        {
            fixed (byte* pData = data)
            fixed (byte* pKey  = key)
            fixed (uint* pRows = rowIndices)
            fixed (byte* pRoot = root)
            {
                rc = PearlMiningNative.MerkleRootAndProof(
                    pData, (nuint)data.Length,
                    pKey,
                    pRows, (nuint)rowIndices.Length,
                    (nuint)rowWidth,
                    pRoot,
                    &totalLeaves,
                    &leafDataPtr, &leafCount,
                    &leafIdxPtr,  &leafIdxLen,
                    &siblingsPtr, &sibCount,
                    &errMsg);
            }

            if (rc != 0)
            {
                var msg = errMsg != null ? Marshal.PtrToStringUTF8((IntPtr)errMsg) ?? "(no message)" : "(no message)";
                throw new MerkleRootAndProofException(rc, msg);
            }

            // ---- Marshal leaf_data: leaf_count × 1024 bytes ----
            var leafData = new byte[leafCount][];
            for (nuint i = 0; i < leafCount; i++)
            {
                var chunk = new byte[1024];
                Marshal.Copy((IntPtr)(leafDataPtr + i * 1024), chunk, 0, 1024);
                leafData[i] = chunk;
            }

            // ---- Marshal leaf_indices: leaf_count × u32 ----
            var leafIndices = new uint[leafIdxLen];
            if (leafIdxLen > 0)
            {
                new ReadOnlySpan<uint>(leafIdxPtr, (int)leafIdxLen).CopyTo(leafIndices);
            }

            // ---- Marshal siblings: sibling_count × 32 bytes ----
            var siblings = new byte[sibCount][];
            for (nuint i = 0; i < sibCount; i++)
            {
                var sib = new byte[32];
                Marshal.Copy((IntPtr)(siblingsPtr + i * 32), sib, 0, 32);
                siblings[i] = sib;
            }

            return new MerkleRootAndProofResult(root, leafData, leafIndices, totalLeaves, siblings);
        }
        finally
        {
            if (leafDataPtr != null) PearlMiningNative.FreeBuffer(leafDataPtr, leafCount * 1024);
            if (leafIdxPtr  != null) PearlMiningNative.FreeU32Buffer(leafIdxPtr, leafIdxLen);
            if (siblingsPtr != null) PearlMiningNative.FreeBuffer(siblingsPtr, sibCount * 32);
            if (errMsg      != null) PearlMiningNative.FreeString(errMsg);
        }
    }
}
