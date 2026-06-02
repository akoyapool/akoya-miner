// MiningConfiguration — verbatim wire layout of the Pearl proof system's
// MiningConfiguration.
//
// Wire format (52 bytes, little-endian):
//   common_dim     u32  (4)
//   rank           u16  (2)
//   mma_type       u16  (2)   0 = Int7xInt7ToInt32 (only valid value today)
//   rows_pattern        (6)   PeriodicPattern.to_bytes
//   cols_pattern        (6)
//   reserved            (32)  must be all zeros

using System.Buffers.Binary;

namespace Akoya.Crypto;

public enum MMAType : ushort
{
    Int7xInt7ToInt32 = 0,
}

/// <summary>
/// Three-dim periodic pattern: an ordered list of (stride, length)
/// triples. The Rust impl stores `shape: [(u32, u32); 3]` and serializes
/// each dim as 2 bytes (factor-1, length-1), where factor = stride /
/// min_stride and min_stride starts at 1 then becomes stride*length.
/// </summary>
public readonly record struct PeriodicPattern(
    (uint Stride, uint Length) D0,
    (uint Stride, uint Length) D1,
    (uint Stride, uint Length) D2)
{
    public const int SerializedSize = 6;

    /// <summary>Identity pattern (one element): all dims (1,1).</summary>
    public static readonly PeriodicPattern Identity =
        new((1, 1), (1, 1), (1, 1));

    public byte[] ToBytes()
    {
        var data = new byte[SerializedSize];
        uint minStride = 1;
        var dims = new[] { D0, D1, D2 };
        for (int i = 0; i < 3; i++)
        {
            var (stride, length) = dims[i];
            uint factor = stride / minStride;
            data[2 * i]     = (byte)(factor - 1);
            data[2 * i + 1] = (byte)(length - 1);
            minStride = stride * length;
        }
        return data;
    }

    /// <summary>
    /// Given a sorted, zero-rooted list of indices, reverse-engineer the (stride,
    /// length) shape array. Pads with trailing (period, 1) tuples to NUM_DIMS=3.
    /// </summary>
    public static PeriodicPattern FromIndices(IReadOnlyList<uint> pattern)
    {
        if (pattern.Count == 0) throw new ArgumentException("Pattern cannot be empty");
        for (int i = 1; i < pattern.Count; i++)
            if (pattern[i] <= pattern[i - 1])
                throw new ArgumentException("Pattern must be sorted and have no duplicates");
        if (pattern[0] != 0) throw new ArgumentException("Pattern must start at 0");

        var p = new List<uint>(pattern);
        var shape = new List<(uint Stride, uint Length)>();

        while (p.Count > 1)
        {
            bool found = false;
            for (int period = 1; period < p.Count; period++)
            {
                if (p.Count % period != 0) continue;
                uint s = p[period];
                bool isPeriodic = true;
                for (int i = 0; i + period < p.Count; i++)
                {
                    if (p[i] + s != p[i + period]) { isPeriodic = false; break; }
                }
                if (isPeriodic)
                {
                    shape.Add((s, (uint)(p.Count / period)));
                    p.RemoveRange(period, p.Count - period);
                    found = true;
                    break;
                }
            }
            if (!found) throw new ArgumentException("Pattern is not periodic");
        }

        shape.Reverse();
        uint trailing = shape.Count == 0 ? 1u : shape[^1].Stride * shape[^1].Length;
        while (shape.Count < 3) shape.Add((trailing, 1));
        return new PeriodicPattern(shape[0], shape[1], shape[2]);
    }

    /// <summary>Number of elements in the pattern (product of lengths).</summary>
    public uint Size => D0.Length * D1.Length * D2.Length;
}

public sealed record MiningConfiguration(
    uint CommonDim,
    ushort Rank,
    MMAType MmaType,
    PeriodicPattern RowsPattern,
    PeriodicPattern ColsPattern)
{
    public const int SerializedSize = 52;
    public const int ReservedSize   = 32;

    /// <summary>
    /// Default MinerSettings rows/cols patterns (matching
    /// miner_base.settings.MinerSettings — H100 hash-tile pattern).
    /// rows = [0, 8] (size 2, stride 8); cols = 64-element [0,1,8,9,...248,249].
    /// </summary>
    public static readonly uint[] DefaultRowsIndices = { 0, 8 };

    public static readonly uint[] DefaultColsIndices = BuildDefaultCols();

    private static uint[] BuildDefaultCols()
    {
        // Same pattern as miner_base.settings.MinerSettings.cols_pattern:
        //   for r in 0..32: emit (8r, 8r+1)  → 64 values total.
        var c = new uint[64];
        for (int r = 0; r < 32; r++)
        {
            c[2 * r]     = (uint)(8 * r);
            c[2 * r + 1] = (uint)(8 * r + 1);
        }
        return c;
    }

    /// <summary>
    /// Parse a comma-separated index list from an env var (e.g. AKOYA_ROWS_PATTERN
    /// = "0,1,2,...,15"); returns <paramref name="fallback"/> if unset/blank.
    /// The committed hash-tile pattern MUST match what the GPU kernel actually
    /// XORs per tile, or shares are rejected. The ROCm (MI300X) kernel computes a
    /// contiguous 16x16 tile, so that deployment sets both vars to "0..15".
    /// </summary>
    private static uint[] PatternFromEnv(string envName, uint[] fallback)
    {
        var s = System.Environment.GetEnvironmentVariable(envName);
        if (string.IsNullOrWhiteSpace(s)) return fallback;
        var parts = s.Split(',', System.StringSplitOptions.RemoveEmptyEntries | System.StringSplitOptions.TrimEntries);
        var idx = new uint[parts.Length];
        for (int i = 0; i < parts.Length; i++) idx[i] = uint.Parse(parts[i]);
        return idx;
    }

    /// <summary>
    /// Default MiningConfiguration matching pearl-pure-miner: common_dim=k,
    /// rank=128, MMAType.Int7xInt7ToInt32. Rows/cols patterns default to the H100
    /// hash-tile pattern but can be overridden via AKOYA_ROWS_PATTERN /
    /// AKOYA_COLS_PATTERN (the ROCm path sets both to the contiguous 16x16 the
    /// MI300X kernel computes). The pattern is committed in job_key and verified
    /// per-share, so any valid periodic AP is accepted by the network.
    /// </summary>
    public static MiningConfiguration Default(uint commonDim, ushort rank = 128)
        => new(
            CommonDim: commonDim,
            Rank: rank,
            MmaType: MMAType.Int7xInt7ToInt32,
            RowsPattern: PeriodicPattern.FromIndices(PatternFromEnv("AKOYA_ROWS_PATTERN", DefaultRowsIndices)),
            ColsPattern: PeriodicPattern.FromIndices(PatternFromEnv("AKOYA_COLS_PATTERN", DefaultColsIndices)));

    /// <summary>
    /// Protocol dot-product length used by the per-tile target scaling.
    /// For today's Int7 MMA this is the common dimension rounded down to the
    /// verifier's dot-product quantum (floor-div, NOT round-up).
    /// </summary>
    public uint DotProductLength()
    {
        uint quantum = MmaType switch
        {
            MMAType.Int7xInt7ToInt32 => 128,
            _ => throw new NotSupportedException($"Unsupported MMA type {MmaType}")
        };

        return (CommonDim / quantum) * quantum;
    }

    /// <summary>
    /// Difficulty adjustment factor:
    ///   hash_tile_h * hash_tile_w * dot_product_length
    /// where hash_tile_h = rows.size, hash_tile_w = cols.size.
    /// </summary>
    public ulong DifficultyAdjustmentFactor()
        => (ulong)RowsPattern.Size * ColsPattern.Size * DotProductLength();

    public byte[] ToBytes()
    {
        var buf = new byte[SerializedSize];
        var span = buf.AsSpan();
        BinaryPrimitives.WriteUInt32LittleEndian(span.Slice(0, 4), CommonDim);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(4, 2), Rank);
        BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(6, 2), (ushort)MmaType);
        RowsPattern.ToBytes().CopyTo(span.Slice(8, 6));
        ColsPattern.ToBytes().CopyTo(span.Slice(14, 6));
        // span[20..52] is reserved — already zero from `new byte[52]`.
        return buf;
    }
}

public static class CommitmentHasher
{
    public const int KeySize = 32;

    /// <summary>
    /// Derive the V2 canonical 32-byte commitment key (jobKey).
    /// <c>key = BLAKE3(incomplete_header_bytes ‖ mining_config.to_bytes())</c>.
    /// </summary>
    /// <remarks>
    /// Must byte-for-byte match the pool's job-key derivation, which is itself
    /// the same construction the network uses to re-derive jobKey on block-find
    /// consensus. There is NO minerId mixed in: the network has no concept of
    /// miner identity, and the audit_proof Merkle tree is keyed by this jobKey
    /// — any extra mix-in produces a different root and every share fails
    /// <c>audit_proof_merkle_mismatch</c> on the pool verifier (and silently
    /// fails block-find at the network layer).
    /// </remarks>
    public static byte[] GetKey(ReadOnlySpan<byte> incompleteHeaderBytes, MiningConfiguration miningConfig)
    {
        var cfgBytes = miningConfig.ToBytes();
        var input = new byte[incompleteHeaderBytes.Length + cfgBytes.Length];
        incompleteHeaderBytes.CopyTo(input);
        cfgBytes.CopyTo(input.AsSpan(incompleteHeaderBytes.Length));
        return Blake3.Hash(input);
    }

    /// <summary>
    /// Derive a per-miner jobKey by mixing the wallet GUID in big-endian.
    /// </summary>
    /// <remarks>
    /// <b>Do NOT use in production.</b> The V2 canonical jobKey is the 2-arg
    /// overload <see cref="GetKey(ReadOnlySpan{byte}, MiningConfiguration)"/>
    /// — the network's block-find consensus check has no concept of minerId and
    /// will reject blocks computed with this overload. Kept only for the
    /// regression test that proves the byte-order trap (native vs big-endian
    /// Guid serialisation) hasn't crept back in elsewhere in the codebase.
    /// </remarks>
    [Obsolete("V2 jobKey is BLAKE3(σ ‖ configBytes) only — no minerId.")]
    public static byte[] GetKey(
        ReadOnlySpan<byte> incompleteHeaderBytes,
        MiningConfiguration miningConfig,
        Guid minerId)
    {
        var cfgBytes = miningConfig.ToBytes();
        var totalLen = incompleteHeaderBytes.Length + cfgBytes.Length + 16;

        // 76 + 52 + 16 = 144 bytes typical; always safe to stackalloc.
        Span<byte> input = stackalloc byte[256].Slice(0, totalLen);
        incompleteHeaderBytes.CopyTo(input);
        cfgBytes.CopyTo(input.Slice(incompleteHeaderBytes.Length));
        minerId.TryWriteBytes(
            input.Slice(incompleteHeaderBytes.Length + cfgBytes.Length, 16),
            bigEndian: true, out _);

        return Blake3.Hash(input);
    }

    /// <summary>
    /// Raw-bytes variant of the per-miner 3-arg overload.
    /// </summary>
    /// <remarks>
    /// <b>Do NOT use in production</b> — see remarks on
    /// <see cref="GetKey(ReadOnlySpan{byte}, MiningConfiguration, Guid)"/>.
    /// network consensus requires the 2-arg jobKey; this overload is retained
    /// only for regression-testing the historical big-endian-vs-native Guid
    /// byte-order trap.
    /// </remarks>
    [Obsolete("V2 jobKey is BLAKE3(σ ‖ configBytes) only — no minerId.")]
    public static byte[] GetKey(
        ReadOnlySpan<byte> incompleteHeaderBytes,
        ReadOnlySpan<byte> miningConfigBytes,
        Guid minerId)
    {
        if (miningConfigBytes.Length != MiningConfiguration.SerializedSize)
            throw new ArgumentException(
                $"miningConfigBytes must be {MiningConfiguration.SerializedSize} B (got {miningConfigBytes.Length})",
                nameof(miningConfigBytes));

        var totalLen = incompleteHeaderBytes.Length + miningConfigBytes.Length + 16;
        Span<byte> input = stackalloc byte[256].Slice(0, totalLen);
        incompleteHeaderBytes.CopyTo(input);
        miningConfigBytes.CopyTo(input.Slice(incompleteHeaderBytes.Length));
        minerId.TryWriteBytes(
            input.Slice(incompleteHeaderBytes.Length + miningConfigBytes.Length, 16),
            bigEndian: true, out _);
        return Blake3.Hash(input);
    }

    /// <summary>
    /// Compute keyed commitment hashes for A and B slices.
    /// hashA = BLAKE3_keyed(jobKey, aSlice), hashB = BLAKE3_keyed(jobKey, bSlice)
    /// </summary>
    public static (byte[] HashA, byte[] HashB) ComputeCommitmentHashes(
        ReadOnlySpan<byte> jobKey,
        ReadOnlySpan<byte> aSlice,
        ReadOnlySpan<byte> bSlice)
    {
        return (Blake3.KeyedHash(jobKey, aSlice), Blake3.KeyedHash(jobKey, bSlice));
    }

    /// <summary>
    /// Derive chained noise seeds from jobKey and commitment hashes.
    /// bNoiseSeed = BLAKE3(jobKey ‖ hashB)
    /// aNoiseSeed = BLAKE3(bNoiseSeed ‖ hashA)
    /// </summary>
    public static (byte[] BNoiseSeed, byte[] ANoiseSeed) DeriveNoiseSeeds(
        ReadOnlySpan<byte> jobKey,
        ReadOnlySpan<byte> hashA,
        ReadOnlySpan<byte> hashB)
    {
        var bInput = new byte[64];
        jobKey.CopyTo(bInput);
        hashB.CopyTo(bInput.AsSpan(32));
        var bNoiseSeed = Blake3.Hash(bInput);

        var aInput = new byte[64];
        bNoiseSeed.CopyTo(aInput);
        hashA.CopyTo(aInput.AsSpan(32));
        var aNoiseSeed = Blake3.Hash(aInput);

        return (bNoiseSeed, aNoiseSeed);
    }
}
