// AuditProofBuilder — builds the flat sibling buffer for K audit openings
// against a B-matrix already expanded in memory.
//
// Wire layout (audit_proof v1 spec §4, power-of-two only):
//
//   AuditProof.siblings = sibling[0] ‖ sibling[1] ‖ ... ‖ sibling[K-1]
//   where sibling[i] = sib(i, 0) ‖ sib(i, 1) ‖ ... ‖ sib(i, levels-1)
//
// Each sib(i, level) is the 32-byte CV of the subtree sibling encountered
// when walking from leaf audit_idx[i] up to the root, leaf→root order.
// Total length: K * levels * 32 bytes.
//
// IMPORTANT: this is a *reference* / fallback implementation. It walks the
// keyed-BLAKE3 Merkle tree in managed code via Akoya.Crypto.Blake3 primitives,
// which mirror pearl_blake3::MerkleTree byte-for-byte. The production hot
// path will be a thin FFI wrapper that extracts paths from the cached
// MerkleTreeHandle in Rust (TBD — wire integration is gated on the pool's
// proto branch landing). For correctness this is the source of truth and
// will be used in tests on both sides.

using System.Buffers;

namespace Akoya.Crypto;

public static class AuditProofBuilder
{
    private const int LeafSize = 1024;
    private const int HashSize = 32;

    /// <summary>
    /// Build the flat <c>AuditProof.siblings</c> buffer for K audit openings.
    ///
    /// Length of returned buffer: <c>K * levels * 32</c>, where
    /// <c>levels = log2(total_leaves)</c>. <c>total_leaves</c> must be a power
    /// of two (audit_proof v1 §3.2(a) constraint).
    /// </summary>
    /// <param name="bBytes">Full B matrix, raw int7 bytes (sbyte reinterpreted as byte), row-major n×k.</param>
    /// <param name="jobKey">The 32-byte jobKey used to key the Merkle tree.</param>
    /// <param name="auditIndices">K leaf indices in <c>[0, total_leaves)</c>. May contain duplicates (spec permits).</param>
    public static byte[] Build(
        ReadOnlySpan<byte> bBytes,
        ReadOnlySpan<byte> jobKey,
        ReadOnlySpan<uint> auditIndices)
    {
        if (jobKey.Length != 32)
            throw new ArgumentException("jobKey must be 32 bytes.", nameof(jobKey));
        if (auditIndices.Length > AuditIndexDeriver.AuditKMax)
            throw new ArgumentOutOfRangeException(nameof(auditIndices),
                $"K must be ≤ {AuditIndexDeriver.AuditKMax} (spec hard cap).");

        if (auditIndices.IsEmpty)
            return [];

        // Pad to chunk boundary — same rule as Blake3.MerkleRoot.
        int paddedLen = ((bBytes.Length + LeafSize - 1) / LeafSize) * LeafSize;
        if (paddedLen == 0) paddedLen = LeafSize;
        int totalLeaves = paddedLen / LeafSize;

        if (totalLeaves < 2)
            throw new ArgumentException(
                "AuditProofBuilder requires total_leaves >= 2 (single-leaf path uses HashSingleChunkRoot, no siblings).",
                nameof(bBytes));
        if ((totalLeaves & (totalLeaves - 1)) != 0)
            throw new ArgumentException(
                $"total_leaves must be a power of two (audit_proof v1 §3.2(a)); got {totalLeaves}.",
                nameof(bBytes));

        int levels = BitOperations_Log2((uint)totalLeaves); // levels = log2(N) for power-of-two N
        var siblings = new byte[auditIndices.Length * levels * HashSize];

        // Pre-compute every leaf CV once (O(N) work, one-shot, mirrors
        // Blake3.MerkleRoot's leaf loop).
        var leafCvs = new byte[totalLeaves][];
        byte[]? rented = null;
        ReadOnlySpan<byte> padded;
        if (paddedLen == bBytes.Length)
        {
            padded = bBytes;
        }
        else
        {
            rented = ArrayPool<byte>.Shared.Rent(paddedLen);
            rented.AsSpan(0, paddedLen).Clear();
            bBytes.CopyTo(rented);
            padded = rented.AsSpan(0, paddedLen);
        }

        try
        {
            for (int i = 0; i < totalLeaves; i++)
                leafCvs[i] = Blake3.ChunkCv(padded.Slice(i * LeafSize, LeafSize), (ulong)i, jobKey);
        }
        finally
        {
            if (rented is not null)
                ArrayPool<byte>.Shared.Return(rented, clearArray: true);
        }

        // Pre-compute the full inner-node CV table. We need every level so
        // we can read sibling CVs at arbitrary positions without re-hashing.
        // For a power-of-two tree of N leaves: total inner nodes = N - 1
        // (N/2 + N/4 + ... + 1). Memory: ~2N * 32 bytes. For N=524288 (5090
        // profile) that's 32 MiB — comfortably one-shot at σ-install
        // cadence (~30s), not on the share-build hot path.
        var levelsCvs = new byte[levels + 1][][];
        levelsCvs[0] = leafCvs;
        for (int lvl = 0; lvl < levels; lvl++)
        {
            var cur = levelsCvs[lvl];
            var next = new byte[cur.Length / 2][];
            for (int i = 0; i < cur.Length; i += 2)
            {
                bool isTopCombine = lvl == levels - 1;
                next[i / 2] = isTopCombine
                    ? Blake3.RootCv(cur[i], cur[i + 1], jobKey)
                    : Blake3.ParentCv(cur[i], cur[i + 1], jobKey);
            }
            levelsCvs[lvl + 1] = next;
        }

        // Sanity: the top of the table is the Merkle root.
        // (Not used here; available to callers if they want to assert it
        //  matches BProof.Root before shipping the share.)

        // Extract per-opening sibling paths, leaf→root.
        for (int i = 0; i < auditIndices.Length; i++)
        {
            uint idx = auditIndices[i];
            if (idx >= totalLeaves)
                throw new ArgumentOutOfRangeException(nameof(auditIndices),
                    $"audit index {idx} out of range for total_leaves={totalLeaves}.");

            int blockOff = i * levels * HashSize;
            uint cur = idx;
            for (int lvl = 0; lvl < levels; lvl++)
            {
                uint sibIdx = cur ^ 1u;
                Buffer.BlockCopy(
                    levelsCvs[lvl][sibIdx], 0,
                    siblings, blockOff + lvl * HashSize,
                    HashSize);
                cur >>= 1;
            }
        }

        return siblings;
    }

    private static int BitOperations_Log2(uint n) => System.Numerics.BitOperations.Log2(n);
}
