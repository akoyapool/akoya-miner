using System;

namespace Akoya.Crypto;

/// <summary>
/// Host-side replay of <c>pearl_capi_lcg_int7_fill</c>.
///
/// Byte-identical to the device kernel in
/// <c>miner/pearl-gemm/csrc/capi/pearl_gemm_capi_util.cu</c> and to the
/// Python reference in <c>scripts/lcg_int7_ref.py</c>. Used at proof-build
/// time to recover the exact A matrix any iteration used, without keeping
/// per-iter device snapshots around.
///
/// Output bytes are signed int7 ∈ [-63, +63] (matches
/// <c>torch.randint(-63, 64, ...)</c> range used by pure-miner's
/// <c>matrix_factory.fresh_A</c>).
/// </summary>
public static class LcgInt7
{
    private static ulong SplitMix64(ulong z)
    {
        z = unchecked(z + 0x9E3779B97F4A7C15UL);
        z = unchecked((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL);
        z = unchecked((z ^ (z >> 27)) * 0x94D049BB133111EBUL);
        return z ^ (z >> 31);
    }

    /// <summary>Fill <paramref name="dst"/> with deterministic int7 bytes.</summary>
    public static void Fill(Span<sbyte> dst, ulong seedLo, ulong seedHi)
    {
        ulong baseSeed = SplitMix64(seedLo ^ SplitMix64(seedHi));
        long n = dst.Length;
        long n8 = n / 8;
        for (long i = 0; i < n8; i++)
        {
            ulong z = SplitMix64(unchecked(baseSeed + (ulong)i));
            for (int b = 0; b < 8; b++)
            {
                uint v = (uint)((z >> (b * 8)) & 0xFFu);
                uint r = v % 127u;
                dst[(int)(i * 8 + b)] = (sbyte)((int)r - 63);
            }
        }
        long tailOff = n8 * 8;
        long tailLen = n - tailOff;
        if (tailLen > 0)
        {
            ulong z = SplitMix64(unchecked(baseSeed + (ulong)n8));
            for (int b = 0; b < (int)tailLen; b++)
            {
                uint v = (uint)((z >> (b * 8)) & 0xFFu);
                uint r = v % 127u;
                dst[(int)(tailOff + b)] = (sbyte)((int)r - 63);
            }
        }
    }

    /// <summary>Allocate + fill — convenience for proof-time replay.</summary>
    public static byte[] FillNew(long n, ulong seedLo, ulong seedHi)
    {
        var arr = new byte[n];
        var sbytes = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, sbyte>(arr.AsSpan());
        Fill(sbytes, seedLo, seedHi);
        return arr;
    }
}
