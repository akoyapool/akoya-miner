using System.Buffers.Binary;
using System.Diagnostics;
using Akoya.Crypto;
using Akoya.Mining;
using Google.Protobuf;
using PearlPool.Proto.V2;
using ProtoMerkleProof = PearlPool.Proto.V2.MerkleProof;

namespace Akoya.Miner.Mining;

internal static class ShareBuilder
{
    /// <summary>
    /// Per-step timings populated by <see cref="Build"/>. Used by
    /// HandleTrigger to log block-find latency attribution.
    /// </summary>
    public readonly record struct Timings(
        double SliceMs,
        double AMerkleMs,
        double BMerkleMs,
        double NoiseMs,
        double JackpotMs,
        double HashMs,
        double AuditMs,
        double PackMs);

    /// <summary>
    /// Build a wire-ready <see cref="ShareSubmission"/> from raw mining
    /// state at trigger time. Returns the message; caller hands it to
    /// <c>MiningSession.EnqueueAsync(new MinerEvent { Share = ... })</c>.
    /// </summary>
    public static ShareSubmission Build(
        ReadOnlySpan<byte> sigma,
        ReadOnlySpan<byte> configBytes,
        ReadOnlySpan<byte> jobKey,
        ReadOnlySpan<byte> aBytes,
        ReadOnlySpan<byte> bBytes,
        uint[] aRowIndices,
        uint[] bColIndices,
        int tileRow,
        int tileCol,
        int m,
        int n,
        int k,
        int r,
        uint claimedDifficultyNbits,
        byte[] hashB,
        out Timings timings,
        bool collectTimings = true)
    {
        int h = aRowIndices.Length;
        int w = bColIndices.Length;

        // 1. Extract A/B slices (row-major; K bytes per row).
        long stageStart = TimingStart(collectTimings);
        var aSlice = ExtractSlice(aBytes, aRowIndices, k);
        var bSlice = ExtractSlice(bBytes, bColIndices, k);
        double msSlice = TimingElapsedMs(collectTimings, stageStart);

        // 2./3. Keyed-BLAKE3 Merkle root + inclusion proof for both matrices.
        //       One Rust FFI call per matrix, SIMD all the way down.
        stageStart = TimingStart(collectTimings);
        var aRoot = MerkleRootAndProof.Compute(aBytes, jobKey, aRowIndices, k);
        double msAMerkle = TimingElapsedMs(collectTimings, stageStart);

        stageStart = TimingStart(collectTimings);
        var bRoot = MerkleRootAndProof.Compute(bBytes, jobKey, bColIndices, k);
        double msBMerkle = TimingElapsedMs(collectTimings, stageStart);
        var hashA = aRoot.Root;

        // 4. Noise seeds, jackpot, claimed_hash.
        stageStart = TimingStart(collectTimings);
        var (bNoiseSeed, aNoiseSeed) = CommitmentHasher.DeriveNoiseSeeds(jobKey, hashA, hashB);

        var secretA = ParseRows(aSlice, h, k);
        var secretB = ParseRows(bSlice, w, k);

        var eAl = NoiseGenerator.GenerateUniformRandomMatrix(SeedLabels.Eal, aNoiseSeed, aRowIndices, r);
        var eAr = NoiseGenerator.GeneratePermutationMatrix(SeedLabels.Ear, aNoiseSeed, k, r);
        var eBl = NoiseGenerator.GeneratePermutationMatrix(SeedLabels.Ebl, bNoiseSeed, k, r);
        var eBr = NoiseGenerator.GenerateUniformRandomMatrix(SeedLabels.Ebr, bNoiseSeed, bColIndices, r);
        double msNoise = TimingElapsedMs(collectTimings, stageStart);

        stageStart = TimingStart(collectTimings);
        var jackpotWords = JackpotComputer.Compute(h, w, k, r, secretA, secretB, eAl, eAr, eBl, eBr);
        double msJackpot = TimingElapsedMs(collectTimings, stageStart);

        stageStart = TimingStart(collectTimings);
        var claimedHash  = JackpotComputer.Hash(jackpotWords, aNoiseSeed);
        double msHash = TimingElapsedMs(collectTimings, stageStart);

        // 5. Pack into the proto.
        stageStart = TimingStart(collectTimings);
        var msg = new ShareSubmission
        {
            Sigma                  = ByteString.CopyFrom(sigma),
            ConfigBytes            = ByteString.CopyFrom(configBytes),
            HashA                  = ByteString.CopyFrom(hashA),
            HashB                  = ByteString.CopyFrom(hashB),
            ASlice                 = ByteString.CopyFrom(aSlice),
            BSlice                 = ByteString.CopyFrom(bSlice),
            AProof                 = ToProtoMerkleProof(aRoot),
            BProof                 = ToProtoMerkleProof(bRoot),
            ClaimedHash            = ByteString.CopyFrom(claimedHash),
            ClaimedDifficultyNbits = claimedDifficultyNbits,
            TileRow                = tileRow,
            TileCol                = tileCol,
            M                      = (uint)m,
            N                      = (uint)n,
        };
        double msPack = TimingElapsedMs(collectTimings, stageStart);

        timings = new Timings(msSlice, msAMerkle, msBMerkle, msNoise, msJackpot, msHash, 0, msPack);
        return msg;
    }

    /// <summary>
    /// Hot-path build that consumes a pre-computed B-merkle inclusion proof
    /// (extracted on the mining thread from the cached <see cref="MerkleTreeHandle"/>
    /// at trigger time — see <c>GpuWorker.HandleTrigger</c>) plus the live
    /// tree handle for opening audit-proof v1 paths against (lifetime
    /// managed by refcount; caller acquires the ref pre-enqueue and
    /// <see cref="ShareFinalizer.Finalize"/> releases after this method
    /// returns).
    /// </summary>
    public static ShareSubmission Build(
        ReadOnlySpan<byte> sigma,
        ReadOnlySpan<byte> configBytes,
        ReadOnlySpan<byte> jobKey,
        ReadOnlySpan<byte> aBytes,
        MerkleRootAndProofResult bProof,
        IMerkleTreeHandle bMerkleTree,
        ReadOnlySpan<byte> bSeed,
        uint auditK,
        uint[] aRowIndices,
        uint[] bColIndices,
        int tileRow,
        int tileCol,
        int m,
        int n,
        int k,
        int r,
        uint claimedDifficultyNbits,
        out Timings timings,
        bool collectTimings = true)
    {
        int h = aRowIndices.Length;
        int w = bColIndices.Length;

        long stageStart = TimingStart(collectTimings);
        var aSlice = ExtractSlice(aBytes, aRowIndices, k);
        var bSlice = ExpandBSeedRows(bSeed, bColIndices, n, k);
        double msSlice = TimingElapsedMs(collectTimings, stageStart);

        stageStart = TimingStart(collectTimings);
        var aRoot = MerkleRootAndProof.Compute(aBytes, jobKey, aRowIndices, k);
        double msAMerkle = TimingElapsedMs(collectTimings, stageStart);

        // B proof was pre-extracted on the mining thread; no FFI here.
        double msBMerkle = 0;
        var hashA = aRoot.Root;
        var hashB = bProof.Root;

        stageStart = TimingStart(collectTimings);
        var (bNoiseSeed, aNoiseSeed) = CommitmentHasher.DeriveNoiseSeeds(jobKey, hashA, hashB);

        var secretA = ParseRows(aSlice, h, k);
        var secretB = ParseRows(bSlice, w, k);

        var eAl = NoiseGenerator.GenerateUniformRandomMatrix(SeedLabels.Eal, aNoiseSeed, aRowIndices, r);
        var eAr = NoiseGenerator.GeneratePermutationMatrix(SeedLabels.Ear, aNoiseSeed, k, r);
        var eBl = NoiseGenerator.GeneratePermutationMatrix(SeedLabels.Ebl, bNoiseSeed, k, r);
        var eBr = NoiseGenerator.GenerateUniformRandomMatrix(SeedLabels.Ebr, bNoiseSeed, bColIndices, r);
        double msNoise = TimingElapsedMs(collectTimings, stageStart);

        stageStart = TimingStart(collectTimings);
        var jackpotWords = JackpotComputer.Compute(h, w, k, r, secretA, secretB, eAl, eAr, eBl, eBr);
        double msJackpot = TimingElapsedMs(collectTimings, stageStart);

        stageStart = TimingStart(collectTimings);
        var claimedHash  = JackpotComputer.Hash(jackpotWords, aNoiseSeed);
        double msHash = TimingElapsedMs(collectTimings, stageStart);

        // Audit-proof v1 opening (audit_proof v1 schematic §3). When
        // K=0 (rollout-safe default) we skip both index derivation and
        // path opening so the wire shape exactly matches the
        // pre-audit ShareSubmission with the BSeed echo added.
        stageStart = TimingStart(collectTimings);
        AuditProof? auditProof = null;
        if (auditK > 0)
        {
            var auditIndices = Akoya.Crypto.AuditIndexDeriver.Derive(
                claimedHash, bSeed, auditK, bMerkleTree.TotalLeaves);
            var auditSiblings = bMerkleTree.AuditPaths(auditIndices);
            auditProof = new AuditProof { Siblings = ByteString.CopyFrom(auditSiblings) };
        }
        double msAudit = TimingElapsedMs(collectTimings, stageStart);

        stageStart = TimingStart(collectTimings);
        var msg = new ShareSubmission
        {
            Sigma                  = ByteString.CopyFrom(sigma),
            ConfigBytes            = ByteString.CopyFrom(configBytes),
            HashA                  = ByteString.CopyFrom(hashA),
            HashB                  = ByteString.CopyFrom(hashB),
            ASlice                 = ByteString.CopyFrom(aSlice),
            BSlice                 = ByteString.CopyFrom(bSlice),
            AProof                 = ToProtoMerkleProof(aRoot),
            BProof                 = ToProtoMerkleProof(bProof),
            ClaimedHash            = ByteString.CopyFrom(claimedHash),
            ClaimedDifficultyNbits = claimedDifficultyNbits,
            TileRow                = tileRow,
            TileCol                = tileCol,
            M                      = (uint)m,
            N                      = (uint)n,
            BSeed                  = ByteString.CopyFrom(bSeed),
        };
        if (auditProof is not null) msg.AuditProof = auditProof;
        double msPack = TimingElapsedMs(collectTimings, stageStart);

        timings = new Timings(msSlice, msAMerkle, msBMerkle, msNoise, msJackpot, msHash, msAudit, msPack);
        return msg;
    }

    /// <summary>
    /// Hot-path build that consumes opened A leaf data plus the full A leaf-CV
    /// table produced by GPU tensor_hash. This avoids copying full A to host
    /// and avoids a full CPU A Merkle rebuild on every share.
    /// </summary>
    public static ShareSubmission Build(
        ReadOnlySpan<byte> sigma,
        ReadOnlySpan<byte> configBytes,
        ReadOnlySpan<byte> jobKey,
        byte[] aSlice,
        byte[] aLeafCvs,
        MerkleRootAndProofResult bProof,
        IMerkleTreeHandle bMerkleTree,
        ReadOnlySpan<byte> bSeed,
        uint auditK,
        uint[] aRowIndices,
        uint[] bColIndices,
        int tileRow,
        int tileCol,
        int m,
        int n,
        int k,
        int r,
        uint claimedDifficultyNbits,
        out Timings timings,
        bool collectTimings = true)
    {
        int h = aRowIndices.Length;
        int w = bColIndices.Length;

        long stageStart = TimingStart(collectTimings);
        var bSlice = ExpandBSeedRows(bSeed, bColIndices, n, k);
        double msSlice = TimingElapsedMs(collectTimings, stageStart);

        stageStart = TimingStart(collectTimings);
        var aRoot = MerkleProofFromLeafCvs.Compute(aLeafCvs, aSlice, jobKey, aRowIndices, m, k);
        double msAMerkle = TimingElapsedMs(collectTimings, stageStart);

        double msBMerkle = 0;
        var hashA = aRoot.Root;
        var hashB = bProof.Root;

        stageStart = TimingStart(collectTimings);
        var (bNoiseSeed, aNoiseSeed) = CommitmentHasher.DeriveNoiseSeeds(jobKey, hashA, hashB);

        var secretA = ParseRows(aSlice, h, k);
        var secretB = ParseRows(bSlice, w, k);

        var eAl = NoiseGenerator.GenerateUniformRandomMatrix(SeedLabels.Eal, aNoiseSeed, aRowIndices, r);
        var eAr = NoiseGenerator.GeneratePermutationMatrix(SeedLabels.Ear, aNoiseSeed, k, r);
        var eBl = NoiseGenerator.GeneratePermutationMatrix(SeedLabels.Ebl, bNoiseSeed, k, r);
        var eBr = NoiseGenerator.GenerateUniformRandomMatrix(SeedLabels.Ebr, bNoiseSeed, bColIndices, r);
        double msNoise = TimingElapsedMs(collectTimings, stageStart);

        stageStart = TimingStart(collectTimings);
        var jackpotWords = JackpotComputer.Compute(h, w, k, r, secretA, secretB, eAl, eAr, eBl, eBr);
        double msJackpot = TimingElapsedMs(collectTimings, stageStart);

        stageStart = TimingStart(collectTimings);
        var claimedHash = JackpotComputer.Hash(jackpotWords, aNoiseSeed);
        double msHash = TimingElapsedMs(collectTimings, stageStart);

        stageStart = TimingStart(collectTimings);
        AuditProof? auditProof = null;
        if (auditK > 0)
        {
            var auditIndices = Akoya.Crypto.AuditIndexDeriver.Derive(
                claimedHash, bSeed, auditK, bMerkleTree.TotalLeaves);
            var auditSiblings = bMerkleTree.AuditPaths(auditIndices);
            auditProof = new AuditProof { Siblings = ByteString.CopyFrom(auditSiblings) };
        }
        double msAudit = TimingElapsedMs(collectTimings, stageStart);

        stageStart = TimingStart(collectTimings);
        var msg = new ShareSubmission
        {
            Sigma                  = ByteString.CopyFrom(sigma),
            ConfigBytes            = ByteString.CopyFrom(configBytes),
            HashA                  = ByteString.CopyFrom(hashA),
            HashB                  = ByteString.CopyFrom(hashB),
            ASlice                 = ByteString.CopyFrom(aSlice),
            BSlice                 = ByteString.CopyFrom(bSlice),
            AProof                 = ToProtoMerkleProof(aRoot),
            BProof                 = ToProtoMerkleProof(bProof),
            ClaimedHash            = ByteString.CopyFrom(claimedHash),
            ClaimedDifficultyNbits = claimedDifficultyNbits,
            TileRow                = tileRow,
            TileCol                = tileCol,
            M                      = (uint)m,
            N                      = (uint)n,
            BSeed                  = ByteString.CopyFrom(bSeed),
        };
        if (auditProof is not null) msg.AuditProof = auditProof;
        double msPack = TimingElapsedMs(collectTimings, stageStart);

        timings = new Timings(msSlice, msAMerkle, msBMerkle, msNoise, msJackpot, msHash, msAudit, msPack);
        return msg;
    }

    /// <summary>Overload without timings — used by tests and any non-hot-path callers.</summary>
    public static ShareSubmission Build(
        ReadOnlySpan<byte> sigma,
        ReadOnlySpan<byte> configBytes,
        ReadOnlySpan<byte> jobKey,
        ReadOnlySpan<byte> aBytes,
        ReadOnlySpan<byte> bBytes,
        uint[] aRowIndices,
        uint[] bColIndices,
        int tileRow,
        int tileCol,
        int m,
        int n,
        int k,
        int r,
        uint claimedDifficultyNbits,
        byte[] hashB)
        => Build(sigma, configBytes, jobKey, aBytes, bBytes, aRowIndices, bColIndices,
                 tileRow, tileCol, m, n, k, r, claimedDifficultyNbits, hashB, out _);

    private static ProtoMerkleProof ToProtoMerkleProof(MerkleRootAndProofResult r)
    {
        var p = new ProtoMerkleProof { TotalLeaves = r.TotalLeaves };
        foreach (var chunk in r.LeafData)    p.LeafData.Add(ByteString.CopyFrom(chunk));
        foreach (var idx   in r.LeafIndices) p.LeafIndices.Add(idx);
        foreach (var sib   in r.Siblings)    p.Siblings.Add(ByteString.CopyFrom(sib));
        return p;
    }

    private static double ElapsedMsSince(long startTimestamp)
        => (Stopwatch.GetTimestamp() - startTimestamp) * 1000.0 / Stopwatch.Frequency;

    private static long TimingStart(bool collectTimings)
        => collectTimings ? Stopwatch.GetTimestamp() : 0;

    private static double TimingElapsedMs(bool collectTimings, long startTimestamp)
        => collectTimings ? ElapsedMsSince(startTimestamp) : 0.0;

    private static byte[] ExpandBSeedRows(ReadOnlySpan<byte> bSeed, uint[] indices, int rowCount, int rowWidth)
    {
        if (bSeed.Length != SigmaContext.BSeedSize)
            throw new ArgumentException("BSeed must be 32 bytes.", nameof(bSeed));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rowCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rowWidth);
        if (indices.Length == 0)
            throw new ArgumentException("indices must be non-empty.", nameof(indices));

        var slice = new byte[checked(indices.Length * rowWidth)];
        for (int i = 0; i < indices.Length; i++)
        {
            uint row = indices[i];
            if (row >= (uint)rowCount)
                throw new ArgumentOutOfRangeException(nameof(indices), $"Row index {row} >= row count {rowCount}.");

            ulong byteOffset = checked((ulong)row * (ulong)rowWidth);
            NativeBSeedExpansion.ExpandRangeRaw(bSeed, byteOffset, slice.AsSpan(i * rowWidth, rowWidth));
        }

        return slice;
    }

    private static byte[] ExtractSlice(ReadOnlySpan<byte> matrix, uint[] indices, int rowWidth)
    {
        var slice = new byte[indices.Length * rowWidth];
        for (int i = 0; i < indices.Length; i++)
        {
            int srcOffset = (int)indices[i] * rowWidth;
            matrix.Slice(srcOffset, rowWidth).CopyTo(slice.AsSpan(i * rowWidth));
        }
        return slice;
    }

    private static sbyte[][] ParseRows(ReadOnlySpan<byte> slice, int numRows, int cols)
    {
        var rows = new sbyte[numRows][];
        for (int i = 0; i < numRows; i++)
        {
            rows[i] = new sbyte[cols];
            for (int j = 0; j < cols; j++)
                rows[i][j] = unchecked((sbyte)slice[i * cols + j]);
        }
        return rows;
    }
}
