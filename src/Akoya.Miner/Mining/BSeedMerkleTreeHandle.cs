using Akoya.Crypto;
using Akoya.Mining;

namespace Akoya.Miner.Mining;

internal sealed class BSeedMerkleTreeHandle : IMerkleTreeHandle, IDisposable
{
    private const int LeafSize = Blake3.ChunkLen;
    private const int HashSize = Blake3.DigestSize;

    private readonly byte[] _bSeed;
    private readonly byte[] _key;
    private readonly int _numRows;
    private readonly int _rowWidth;
    private readonly long _dataBytes;
    private readonly byte[][] _layers;
    private int _refCount = 1;

    public byte[] Root { get; }
    public uint TotalLeaves { get; }

    private BSeedMerkleTreeHandle(
        byte[] bSeed,
        byte[] key,
        int numRows,
        int rowWidth,
        uint totalLeaves,
        byte[] root,
        byte[][] layers)
    {
        _bSeed = bSeed;
        _key = key;
        _numRows = numRows;
        _rowWidth = rowWidth;
        _dataBytes = checked((long)numRows * rowWidth);
        TotalLeaves = totalLeaves;
        Root = root;
        _layers = layers;
    }

    public static BSeedMerkleTreeHandle BuildFromLeafCvs(
        byte[] leafCvs,
        ReadOnlySpan<byte> bSeed,
        ReadOnlySpan<byte> key,
        int numRows,
        int rowWidth)
    {
        if (bSeed.Length != SigmaContext.BSeedSize) throw new ArgumentException("BSeed must be 32 bytes.", nameof(bSeed));
        if (key.Length != Blake3.DigestSize) throw new ArgumentException("Key must be 32 bytes.", nameof(key));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numRows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rowWidth);

        long dataBytes = checked((long)numRows * rowWidth);
        long leavesLong = (dataBytes + LeafSize - 1) / LeafSize;
        if (leavesLong <= 0 || leavesLong > uint.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(numRows), $"Invalid B Merkle leaf count {leavesLong}.");
        if (leafCvs.LongLength != leavesLong * HashSize)
            throw new ArgumentException(
                $"leafCvs must be exactly {leavesLong * HashSize:N0} bytes for {leavesLong:N0} leaves.",
                nameof(leafCvs));

        var keyBytes = key.ToArray();
        var layers = BuildLayers(leafCvs, (int)leavesLong, keyBytes);
        var root = ComputeRoot(layers, keyBytes, bSeed.ToArray(), dataBytes);

        return new BSeedMerkleTreeHandle(
            bSeed.ToArray(),
            keyBytes,
            numRows,
            rowWidth,
            (uint)leavesLong,
            root,
            layers);
    }

    public void Acquire()
    {
        if (Interlocked.Increment(ref _refCount) <= 1)
        {
            Interlocked.Decrement(ref _refCount);
            throw new ObjectDisposedException(nameof(BSeedMerkleTreeHandle));
        }
    }

    public void Release()
    {
        int after = Interlocked.Decrement(ref _refCount);
        if (after >= 0) return;
        Interlocked.Exchange(ref _refCount, 0);
    }

    public MerkleRootAndProofResult Proof(ReadOnlySpan<uint> rowIndices)
    {
        ThrowIfDisposed();
        if (rowIndices.IsEmpty) throw new ArgumentException("rowIndices must be non-empty.", nameof(rowIndices));

        foreach (uint row in rowIndices)
        {
            if (row >= (uint)_numRows)
                throw new MerkleRootAndProofException(-1, $"row index {row} out of bounds (num_rows={_numRows}, row_width={_rowWidth})");
        }

        var leafIndices = ComputeLeafIndicesFromRows(rowIndices, _numRows, _rowWidth);
        var leafData = new byte[leafIndices.Length][];
        for (int i = 0; i < leafIndices.Length; i++)
            leafData[i] = ExpandLeaf(leafIndices[i]);

        var siblings = new List<byte[]>();
        var current = new SortedSet<uint>(leafIndices);
        uint levelLen = TotalLeaves;
        int level = 0;

        while (levelLen > 1 && current.Count > 0)
        {
            var levelNodes = _layers[level];
            foreach (uint idx in current)
            {
                if ((idx & 1u) == 1u)
                {
                    if (!current.Contains(idx - 1u))
                        siblings.Add(CopyDigest(levelNodes, idx - 1u));
                }
                else if (!current.Contains(idx + 1u) && idx + 1u < levelLen)
                {
                    siblings.Add(CopyDigest(levelNodes, idx + 1u));
                }
            }

            var next = new SortedSet<uint>();
            foreach (uint idx in current) next.Add(idx / 2u);
            current = next;
            levelLen = (levelLen + 1u) / 2u;
            level++;
        }

        return new MerkleRootAndProofResult(
            (byte[])Root.Clone(),
            leafData,
            leafIndices,
            TotalLeaves,
            siblings.ToArray());
    }

    public byte[] AuditPaths(ReadOnlySpan<uint> leafIndices)
    {
        ThrowIfDisposed();
        if (leafIndices.IsEmpty) return [];

        if (!TotalLeaves.IsPowerOfTwo())
            throw new MerkleRootAndProofException(
                -1,
                $"total_leaves ({TotalLeaves}) must be a power of two for audit_proof v1");

        int levels = System.Numerics.BitOperations.Log2(TotalLeaves);
        var outBytes = new byte[checked(leafIndices.Length * levels * HashSize)];

        for (int k = 0; k < leafIndices.Length; k++)
        {
            uint idx = leafIndices[k];
            if (idx >= TotalLeaves)
                throw new MerkleRootAndProofException(-1, $"leaf index {idx} out of bounds (total_leaves={TotalLeaves})");

            int blockBase = k * levels * HashSize;
            for (int level = 0; level < levels; level++)
            {
                uint sibling = idx ^ 1u;
                _layers[level].AsSpan((int)sibling * HashSize, HashSize)
                    .CopyTo(outBytes.AsSpan(blockBase + level * HashSize, HashSize));
                idx >>= 1;
            }
        }

        return outBytes;
    }

    public void Dispose() => Release();

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _refCount) <= 0, this);
    }

    private static byte[][] BuildLayers(byte[] leafCvs, int totalLeaves, ReadOnlySpan<byte> key)
    {
        var layers = new List<byte[]> { leafCvs };
        var current = leafCvs;
        int levelLen = totalLeaves;

        while (levelLen > 2)
        {
            int nextLen = (levelLen + 1) / 2;
            var next = GC.AllocateUninitializedArray<byte>(nextLen * HashSize);
            for (int i = 0; i < levelLen; i += 2)
            {
                var dst = next.AsSpan((i / 2) * HashSize, HashSize);
                if (i + 1 < levelLen)
                {
                    Blake3.ParentCv(
                        current.AsSpan(i * HashSize, HashSize),
                        current.AsSpan((i + 1) * HashSize, HashSize),
                        key,
                        dst);
                }
                else
                {
                    current.AsSpan(i * HashSize, HashSize).CopyTo(dst);
                }
            }

            layers.Add(next);
            current = next;
            levelLen = nextLen;
        }

        return layers.ToArray();
    }

    private static byte[] ComputeRoot(
        byte[][] layers,
        ReadOnlySpan<byte> key,
        ReadOnlySpan<byte> bSeed,
        long dataBytes)
    {
        var top = layers[^1];
        int topCount = top.Length / HashSize;
        var root = new byte[HashSize];

        if (topCount == 1)
        {
            var leaf = new byte[LeafSize];
            int valid = (int)Math.Min(dataBytes, LeafSize);
            if (valid > 0)
                NativeBSeedExpansion.ExpandRangeRaw(bSeed, 0, leaf.AsSpan(0, valid));
            return Blake3.KeyedHash(key, leaf);
        }

        Blake3.RootCv(top.AsSpan(0, HashSize), top.AsSpan(HashSize, HashSize), key, root);
        return root;
    }

    private static uint[] ComputeLeafIndicesFromRows(
        ReadOnlySpan<uint> rowIndices,
        int numRows,
        int rowWidth)
    {
        var set = new SortedSet<uint>();
        foreach (uint row in rowIndices)
        {
            if (row >= (uint)numRows)
                throw new MerkleRootAndProofException(-1, $"row index {row} out of bounds (num_rows={numRows}, row_width={rowWidth})");

            ulong byteStart = (ulong)row * (ulong)rowWidth;
            ulong byteEnd = byteStart + (ulong)rowWidth - 1UL;
            uint first = (uint)(byteStart / LeafSize);
            uint last = (uint)(byteEnd / LeafSize);
            for (uint idx = first; idx <= last; idx++)
                set.Add(idx);
        }
        return set.ToArray();
    }

    private byte[] ExpandLeaf(uint leafIndex)
    {
        if (leafIndex >= TotalLeaves)
            throw new MerkleRootAndProofException(-1, $"leaf index {leafIndex} out of bounds (total_leaves={TotalLeaves})");

        var chunk = new byte[LeafSize];
        ulong offset = (ulong)leafIndex * LeafSize;
        long remaining = _dataBytes - (long)offset;
        if (remaining > 0)
        {
            int valid = (int)Math.Min(LeafSize, remaining);
            NativeBSeedExpansion.ExpandRangeRaw(_bSeed, offset, chunk.AsSpan(0, valid));
        }
        return chunk;
    }

    private static byte[] CopyDigest(byte[] layer, uint index)
    {
        var digest = new byte[HashSize];
        layer.AsSpan((int)index * HashSize, HashSize).CopyTo(digest);
        return digest;
    }
}

file static class MerkleUIntExtensions
{
    public static bool IsPowerOfTwo(this uint value)
        => value != 0 && (value & (value - 1u)) == 0;
}
