// AuditProofVerifier — reference implementation matching the pool's audit
// verifier (power-of-two only, v1 frozen).
//
// This lives in Akoya.Crypto on the miner side primarily so that
// AuditProofBuilder ↔ AuditProofVerifier can round-trip in unit tests and
// catch any divergence between the two paths before any wire bytes ever
// flow. In production the miner does not run this verifier — the pool
// does. Keep this byte-exact with the pool's verifier.
//
// Verification steps per audit_proof v1 spec §5 (corrected):
//   1. Validate sizes.
//   2. Derive audit indices via AuditIndexDeriver.
//   3. For each opening i ∈ [0, K):
//        a. Re-derive leaf bytes from b_seed via seekable BLAKE3-XOF.
//        b. Apply (b % 127) - 63 transform to recover the int7 layout.
//        c. Compute leaf_cv = chunk_cv(leaf_bytes, idx, jobKey).
//        d. Walk levels 0..levels-1 with parent_cv (NOT root_cv).
//        e. Final combine uses root_cv (sets ROOT flag).
//        f. Compare against the share's HashB Merkle root.

namespace Akoya.Crypto;

public static class AuditProofVerifier
{
    private const int LeafSize = 1024;
    private const int HashSize = 32;

    public enum AuditError
    {
        Ok = 0,
        WrongSiblingLength,
        TotalLeavesNotPowerOfTwo,
        MerklePathMismatch,
        KAboveSpecMax,
    }

    /// <summary>
    /// Verify an audit proof. Returns <see cref="AuditError.Ok"/> on success,
    /// or the first failure encountered. <paramref name="failingIndex"/> is
    /// set to the offending opening index when the failure is per-opening
    /// (i.e. <see cref="AuditError.MerklePathMismatch"/>).
    /// </summary>
    public static AuditError Verify(
        ReadOnlySpan<byte> claimedHash,
        ReadOnlySpan<byte> bSeed,
        ReadOnlySpan<byte> jobKey,
        ReadOnlySpan<byte> hashB,
        uint auditK,
        ulong totalLeaves,
        ReadOnlySpan<byte> siblings,
        out int failingIndex)
    {
        failingIndex = -1;

        if (auditK > AuditIndexDeriver.AuditKMax)
            return AuditError.KAboveSpecMax;
        if (totalLeaves == 0 || (totalLeaves & (totalLeaves - 1)) != 0)
            return AuditError.TotalLeavesNotPowerOfTwo;
        if (auditK == 0)
        {
            // Spec §6: "audit_proof present when audit_k == 0" — ignore silently.
            // For a strict verifier, callers should short-circuit before calling Verify.
            return siblings.IsEmpty ? AuditError.Ok : AuditError.WrongSiblingLength;
        }

        int levels = totalLeaves == 1
            ? 0
            : System.Numerics.BitOperations.Log2((uint)totalLeaves);
        int expectedSiblingsLen = (int)auditK * levels * HashSize;
        if (siblings.Length != expectedSiblingsLen)
            return AuditError.WrongSiblingLength;

        var indices = AuditIndexDeriver.Derive(claimedHash, bSeed, auditK, totalLeaves);

        Span<byte> leafBytes = stackalloc byte[LeafSize];

        for (int i = 0; i < (int)auditK; i++)
        {
            ulong idx = indices[i];

            // (a, b) Re-derive leaf bytes at byte offset idx*1024.
            BSeedExpander.ExpandRangeRaw(bSeed, idx * LeafSize, leafBytes);

            // (c) Leaf CV.
            var node = Blake3.ChunkCv(leafBytes, idx, jobKey);

            // (d) Walk inner levels with parent_cv.
            int sibBlock = i * levels * HashSize;
            uint cur = (uint)idx;
            for (int lvl = 0; lvl < levels - 1; lvl++)
            {
                var sibling = siblings.Slice(sibBlock + lvl * HashSize, HashSize);
                bool weAreLeft = (cur & 1) == 0;
                node = weAreLeft
                    ? Blake3.ParentCv(node, sibling, jobKey)
                    : Blake3.ParentCv(sibling, node, jobKey);
                cur >>= 1;
            }

            // (e) Final combine — root_cv (sets ROOT flag). If levels=0 the
            //     single-leaf path applies (see (f) below).
            byte[] root;
            if (levels == 0)
            {
                // Single-leaf tree — root is just keyed BLAKE3 over the leaf
                // bytes (FlagKeyedHash | FlagChunkStart | FlagChunkEnd | FlagRoot
                // all set by the normal Blake3.KeyedHash path).
                root = Blake3.KeyedHash(jobKey, leafBytes);
            }
            else
            {
                var topSibling = siblings.Slice(sibBlock + (levels - 1) * HashSize, HashSize);
                bool weAreLeft = (cur & 1) == 0;
                root = weAreLeft
                    ? Blake3.RootCv(node, topSibling, jobKey)
                    : Blake3.RootCv(topSibling, node, jobKey);
            }

            // (f) Match against share.HashB.
            if (!root.AsSpan().SequenceEqual(hashB))
            {
                failingIndex = i;
                return AuditError.MerklePathMismatch;
            }
        }

        return AuditError.Ok;
    }
}
