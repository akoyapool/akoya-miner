// JackpotComputer — computes Pearl jackpot accumulator words and hash.

using System.Buffers.Binary;
using System.Numerics;

namespace Akoya.Crypto;

public static class JackpotComputer
{
    public const int JackpotSize = 16;
    private const int RotateLeft = 13;

    /// <summary>
    /// Compute the 16 jackpot words from signal and noise slices.
    /// </summary>
    public static uint[] Compute(
        int h, int w, int k, int r,
        sbyte[][] secretA, sbyte[][] secretB,
        sbyte[][] noiseA, sbyte[][] noiseB)
    {
        var tile = new int[h][];
        for (int i = 0; i < h; i++) tile[i] = new int[w];

        var msg = new uint[JackpotSize];
        for (int ll = r; ll <= k; ll += r)
        {
            for (int u = 0; u < h; u++)
                for (int v = 0; v < w; v++)
                    for (int l = ll - r; l < ll; l++)
                        tile[u][v] += (secretA[u][l] + noiseA[u][l]) * (secretB[v][l] + noiseB[v][l]);

            uint xored = 0;
            for (int u = 0; u < h; u++)
                for (int v = 0; v < w; v++)
                    xored ^= unchecked((uint)tile[u][v]);

            int tid = ((ll / r) - 1) % JackpotSize;
            msg[tid] = BitOperations.RotateLeft(msg[tid], RotateLeft) ^ xored;
        }
        return msg;
    }

    /// <summary>
    /// Compute jackpot from raw E matrices (generates noise internally).
    /// </summary>
    public static uint[] Compute(
        int h, int w, int k, int r,
        sbyte[][] secretA, sbyte[][] secretB,
        sbyte[][] eAl,
        NoiseGenerator.PermutationPair[] eAr,
        NoiseGenerator.PermutationPair[] eBl,
        sbyte[][] eBr)
    {
        var noiseA = new sbyte[eAl.Length][];
        for (int i = 0; i < eAl.Length; i++)
            noiseA[i] = NoiseGenerator.ApplySparsePermutation(eAr, eAl[i]);

        var noiseB = new sbyte[eBr.Length][];
        for (int i = 0; i < eBr.Length; i++)
            noiseB[i] = NoiseGenerator.ApplySparsePermutation(eBl, eBr[i]);

        return Compute(h, w, k, r, secretA, secretB, noiseA, noiseB);
    }

    /// <summary>
    /// Compute the keyed BLAKE3 jackpot hash from 16 jackpot words.
    /// </summary>
    public static byte[] Hash(uint[] jackpotWords, ReadOnlySpan<byte> aNoiseSeed)
    {
        var buf = new byte[JackpotSize * sizeof(uint)];
        for (int i = 0; i < jackpotWords.Length; i++)
            BinaryPrimitives.WriteUInt32LittleEndian(buf.AsSpan(i * 4, 4), jackpotWords[i]);
        return Blake3.KeyedHash(aNoiseSeed, buf);
    }
}
