// MerkleProofBuilder — builds BLAKE3 keyed Merkle inclusion proofs.
//
// Generates proofs that the pool's MerkleProofVerifier.VerifyProof accepts.
// The sibling ordering must exactly match the verifier's consumption order:
// it iterates known nodes in sorted order, and for each:
//   - even index: if index+1 is NOT also known, consume next sibling (right)
//   - odd index:  if index-1 is NOT also known, consume next sibling (left)
// At the final level (2 nodes), unknown sides consume from siblings too.

namespace Akoya.Crypto;

/// <summary>
/// Result of Merkle proof generation — maps 1:1 to pool's MerkleProofData.
/// </summary>
public sealed class MerkleProof
{
    public byte[][] LeafData { get; set; } = [];
    public uint[] LeafIndices { get; set; } = [];
    public uint TotalLeaves { get; set; }
    public byte[][] Siblings { get; set; } = [];
}

public static class MerkleProofBuilder
{
    private const int ChunkLen = 1024;

    /// <summary>
    /// Build a Merkle inclusion proof for the given row indices.
    /// </summary>
    public static MerkleProof BuildProof(
        ReadOnlySpan<byte> data,
        ReadOnlySpan<byte> key,
        ReadOnlySpan<uint> rowIndices,
        int rowWidth)
    {
        if (key.Length != 32) throw new ArgumentException("Key must be 32 bytes.", nameof(key));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rowWidth);

        // Pad data to chunk boundary
        int paddedLen = ((data.Length + ChunkLen - 1) / ChunkLen) * ChunkLen;
        uint totalLeaves = (uint)(paddedLen / ChunkLen);

        // Handle single-chunk special case
        if (totalLeaves <= 1)
        {
            var chunk = new byte[ChunkLen];
            data.Slice(0, Math.Min(data.Length, ChunkLen)).CopyTo(chunk);
            return new MerkleProof
            {
                LeafData = [chunk],
                LeafIndices = [0],
                TotalLeaves = 1,
                Siblings = [],
            };
        }

        // Determine which chunk indices are needed to cover the requested rows
        var requiredChunks = new SortedSet<uint>();
        foreach (var rowIdx in rowIndices)
        {
            int byteStart = (int)rowIdx * rowWidth;
            int byteEnd = byteStart + rowWidth - 1;
            uint chunkStart = (uint)(byteStart / ChunkLen);
            uint chunkEnd = (uint)(byteEnd / ChunkLen);
            for (uint c = chunkStart; c <= chunkEnd && c < totalLeaves; c++)
                requiredChunks.Add(c);
        }

        var leafIndices = requiredChunks.ToArray();

        // Pad data
        byte[] paddedData;
        if (paddedLen == data.Length)
        {
            paddedData = data.ToArray();
        }
        else
        {
            paddedData = new byte[paddedLen];
            data.CopyTo(paddedData);
        }

        // Extract leaf data chunks
        var leafData = new byte[leafIndices.Length][];
        for (int i = 0; i < leafIndices.Length; i++)
        {
            leafData[i] = new byte[ChunkLen];
            paddedData.AsSpan((int)leafIndices[i] * ChunkLen, ChunkLen).CopyTo(leafData[i]);
        }

        // Build ALL chunk CVs at the leaf level
        var allCvs = new byte[totalLeaves][];
        for (uint i = 0; i < totalLeaves; i++)
            allCvs[i] = Blake3.ChunkCv(paddedData.AsSpan((int)i * ChunkLen, ChunkLen), i, key);

        // Now walk the tree bottom-up, mirroring the verifier's exact logic.
        // At each level we track which indices are "known" (from proof leaves)
        // and collect siblings in the exact order the verifier would consume them.
        var siblings = new List<byte[]>();
        var known = new SortedDictionary<uint, byte[]>();
        for (int i = 0; i < leafIndices.Length; i++)
            known[leafIndices[i]] = allCvs[leafIndices[i]];

        uint levelLen = totalLeaves;

        while (levelLen > 2)
        {
            var next = new SortedDictionary<uint, byte[]>();

            // Mirror the verifier: iterate known nodes in sorted order
            foreach (var (index, cv) in known)
            {
                if ((index & 1) == 0)
                {
                    // Even: we are left child. Need right sibling.
                    byte[]? rightCv;
                    if (known.TryGetValue(index + 1, out var inlineRight))
                    {
                        // Both sides known — no sibling needed
                        rightCv = inlineRight;
                    }
                    else if (index + 1 < levelLen)
                    {
                        // Right side is NOT known — emit as sibling
                        rightCv = allCvs[index + 1];
                        siblings.Add(rightCv);
                    }
                    else
                    {
                        rightCv = null; // odd node at end, promoted
                    }

                    next[index / 2] = rightCv is null
                        ? cv
                        : Blake3.ParentCv(cv, rightCv, key);
                }
                else
                {
                    // Odd: we are right child. Check if left was already handled.
                    if (known.ContainsKey(index - 1))
                        continue; // already processed as even's pair

                    // Left side is NOT known — emit as sibling
                    var leftCv = allCvs[index - 1];
                    siblings.Add(leftCv);
                    next[index / 2] = Blake3.ParentCv(leftCv, cv, key);
                }
            }

            // Advance: compute ALL CVs for next level (we need them for future siblings)
            uint nextLevelLen = (levelLen >> 1) + (levelLen & 1u);
            var nextAllCvs = new byte[nextLevelLen][];
            for (uint i = 0; i < levelLen; i += 2)
            {
                if (i + 1 < levelLen)
                    nextAllCvs[i / 2] = Blake3.ParentCv(allCvs[i], allCvs[i + 1], key);
                else
                    nextAllCvs[i / 2] = allCvs[i]; // odd promoted
            }

            allCvs = nextAllCvs;
            known = next;
            levelLen = nextLevelLen;
        }

        // Final two nodes — mirror TryResolveFinalNode
        if (!known.ContainsKey(0))
            siblings.Add(allCvs[0]);
        if (!known.ContainsKey(1) && allCvs.Length > 1 && allCvs[1] is not null)
            siblings.Add(allCvs[1]);

        return new MerkleProof
        {
            LeafData = leafData,
            LeafIndices = leafIndices,
            TotalLeaves = totalLeaves,
            Siblings = siblings.ToArray(),
        };
    }

    /// <summary>
    /// Compute the Merkle root from data and key.
    /// </summary>
    public static byte[] ComputeRoot(ReadOnlySpan<byte> data, ReadOnlySpan<byte> key)
        => Blake3.MerkleRoot(data, key);
}
