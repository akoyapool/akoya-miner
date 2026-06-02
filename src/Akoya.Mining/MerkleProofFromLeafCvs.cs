using System.Runtime.InteropServices;

namespace Akoya.Mining;

public static class MerkleProofFromLeafCvs
{
    private const int DigestSize = 32;
    private const int ChunkLen = 1024;

    public static unsafe MerkleRootAndProofResult Compute(
        ReadOnlySpan<byte> leafCvs,
        ReadOnlySpan<byte> leafData,
        ReadOnlySpan<byte> key,
        ReadOnlySpan<uint> rowIndices,
        int numRows,
        int rowWidth)
    {
        if (key.Length != DigestSize) throw new ArgumentException("Key must be 32 bytes.", nameof(key));
        if (leafCvs.IsEmpty || leafCvs.Length % DigestSize != 0)
            throw new ArgumentException("leafCvs must be a non-empty multiple of 32 bytes.", nameof(leafCvs));
        if (leafData.IsEmpty || leafData.Length % ChunkLen != 0)
            throw new ArgumentException("leafData must be a non-empty multiple of 1024 bytes.", nameof(leafData));
        if (rowIndices.IsEmpty) throw new ArgumentException("rowIndices must be non-empty.", nameof(rowIndices));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numRows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rowWidth);

        var root = new byte[DigestSize];
        uint totalLeaves = 0;
        uint* leafIdxPtr = null;  nuint leafIdxLen = 0;
        byte* siblingsPtr = null; nuint sibCount = 0;
        byte* errMsg = null;

        try
        {
            fixed (byte* pLeafCvs = leafCvs)
            fixed (byte* pLeafData = leafData)
            fixed (byte* pKey = key)
            fixed (uint* pRows = rowIndices)
            fixed (byte* pRoot = root)
            {
                int rc = PearlMiningNative.MerkleProofFromLeafCvs(
                    pLeafCvs,
                    (nuint)leafCvs.Length,
                    pLeafData,
                    (nuint)leafData.Length,
                    pKey,
                    pRows,
                    (nuint)rowIndices.Length,
                    (nuint)numRows,
                    (nuint)rowWidth,
                    pRoot,
                    &totalLeaves,
                    &leafIdxPtr,
                    &leafIdxLen,
                    &siblingsPtr,
                    &sibCount,
                    &errMsg);

                if (rc != 0)
                {
                    var msg = errMsg != null
                        ? Marshal.PtrToStringUTF8((IntPtr)errMsg) ?? "(no message)"
                        : "(no message)";
                    throw new MerkleRootAndProofException(rc, msg);
                }
            }

            if (leafData.Length != checked((int)leafIdxLen * ChunkLen))
                throw new MerkleRootAndProofException(
                    -1,
                    $"native proof returned {leafIdxLen} leaf indices for {leafData.Length} leaf-data bytes");

            var leafChunks = new byte[leafIdxLen][];
            for (nuint i = 0; i < leafIdxLen; i++)
            {
                var chunk = new byte[ChunkLen];
                leafData.Slice(checked((int)i) * ChunkLen, ChunkLen).CopyTo(chunk);
                leafChunks[i] = chunk;
            }

            var leafIndices = new uint[leafIdxLen];
            if (leafIdxLen > 0)
                new ReadOnlySpan<uint>(leafIdxPtr, (int)leafIdxLen).CopyTo(leafIndices);

            var siblings = new byte[sibCount][];
            for (nuint i = 0; i < sibCount; i++)
            {
                var sib = new byte[DigestSize];
                Marshal.Copy((IntPtr)(siblingsPtr + i * DigestSize), sib, 0, DigestSize);
                siblings[i] = sib;
            }

            return new MerkleRootAndProofResult(root, leafChunks, leafIndices, totalLeaves, siblings);
        }
        finally
        {
            if (leafIdxPtr != null) PearlMiningNative.FreeU32Buffer(leafIdxPtr, leafIdxLen);
            if (siblingsPtr != null) PearlMiningNative.FreeBuffer(siblingsPtr, sibCount * DigestSize);
            if (errMsg != null) PearlMiningNative.FreeString(errMsg);
        }
    }
}
