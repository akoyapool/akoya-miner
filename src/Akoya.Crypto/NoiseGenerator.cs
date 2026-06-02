// NoiseGenerator — deterministic Pearl noise generation using keyed BLAKE3 counter mode.

using System.Buffers.Binary;
using System.Numerics;

namespace Akoya.Crypto;

public static class NoiseGenerator
{
    private const int DigestSize = 32;
    private const int UniformRange = 64;
    private const int ZeroPoint = UniformRange / 2;
    private const byte RangeMask = UniformRange - 1;
    private const int BytesPerLine = 4;
    private const int LinesPerHash = DigestSize / BytesPerLine;

    public readonly record struct PermutationPair(uint FirstIndex, uint SecondIndex);

    public static uint MulHiU32(uint a, uint b) => (uint)(((ulong)a * b) >> 32);

    public static byte[] GetRandomHash(uint index, ReadOnlySpan<byte> seed, ReadOnlySpan<byte> key, int prependIndex)
    {
        var message = new byte[64];
        BinaryPrimitives.WriteUInt32LittleEndian(
            message.AsSpan(prependIndex * sizeof(uint), sizeof(uint)),
            checked(index + 1));
        seed.CopyTo(message.AsSpan(32));
        return Blake3.KeyedHash(key, message);
    }

    public static sbyte[][] GenerateUniformRandomMatrix(
        ReadOnlySpan<byte> seed,
        ReadOnlySpan<byte> key,
        IReadOnlyList<uint> rowIndices,
        int numCols)
    {
        var rows = new sbyte[rowIndices.Count][];
        for (int row = 0; row < rowIndices.Count; row++)
        {
            uint rowIndex = rowIndices[row];
            long startIndex = (long)rowIndex * numCols;
            long endIndex = startIndex + numCols;
            int startBlock = (int)(startIndex / DigestSize);
            int endBlock = (int)((endIndex + DigestSize - 1) / DigestSize);

            var values = new List<sbyte>(numCols);
            for (int block = startBlock; block < endBlock; block++)
            {
                var hash = GetRandomHash((uint)block, seed, key, 0);
                for (int i = 0; i < hash.Length; i++)
                {
                    long absIdx = (long)block * DigestSize + i;
                    if (absIdx < startIndex || absIdx >= endIndex) continue;
                    values.Add(unchecked((sbyte)((hash[i] & RangeMask) - ZeroPoint)));
                }
            }
            rows[row] = values.ToArray();
        }
        return rows;
    }

    public static PermutationPair[] GeneratePermutationMatrix(
        ReadOnlySpan<byte> seed,
        ReadOnlySpan<byte> key,
        int k,
        int noiseRank)
    {
        var rankMask = (uint)(noiseRank - 1);
        var result = new PermutationPair[k];

        for (int chunk = 0; chunk * LinesPerHash < k; chunk++)
        {
            var hash = GetRandomHash((uint)chunk, seed, key, 1);
            int len = Math.Min(LinesPerHash, k - chunk * LinesPerHash);

            for (int j = 0; j < len; j++)
            {
                uint rand = BinaryPrimitives.ReadUInt32LittleEndian(
                    hash.AsSpan(j * BytesPerLine, BytesPerLine));
                uint first = rand & rankMask;
                uint second = first ^ (1u + MulHiU32((uint)(noiseRank - 1), rand));
                result[chunk * LinesPerHash + j] = new PermutationPair(first, second);
            }
        }
        return result;
    }

    public static sbyte[] ApplySparsePermutation(
        IReadOnlyList<PermutationPair> perm, ReadOnlySpan<sbyte> vector)
    {
        var result = new sbyte[perm.Count];
        for (int i = 0; i < perm.Count; i++)
        {
            var p = perm[i];
            result[i] = unchecked((sbyte)(vector[(int)p.FirstIndex] - vector[(int)p.SecondIndex]));
        }
        return result;
    }
}

public static class SeedLabels
{
    private static readonly byte[] _a =
    [
        (byte)'A', (byte)'_', (byte)'t', (byte)'e', (byte)'n', (byte)'s', (byte)'o', (byte)'r',
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ];
    private static readonly byte[] _b =
    [
        (byte)'B', (byte)'_', (byte)'t', (byte)'e', (byte)'n', (byte)'s', (byte)'o', (byte)'r',
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ];

    public static ReadOnlySpan<byte> Eal => _a;
    public static ReadOnlySpan<byte> Ear => _a;
    public static ReadOnlySpan<byte> Ebl => _b;
    public static ReadOnlySpan<byte> Ebr => _b;
}
