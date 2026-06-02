// AuditIndexDeriver — pure deterministic function from
//   (claimed_hash, b_seed, audit_k, total_leaves)
// to K audit-leaf indices in [0, total_leaves).
//
// Implements the audit_proof v1 scheme. The cryptographic core; nothing here
// is negotiable without a v2 of the domain separator.
//
// XOF input layout (exactly 82 bytes, no separators):
//
//   offset 0..14   :  "akoya-audit-v1"        (14 bytes, ASCII)
//   offset 14..46  :  claimed_hash            (32 bytes)
//   offset 46..78  :  b_seed                  (32 bytes)
//   offset 78..82  :  audit_k (u32 LE)        (4 bytes)
//
// Then BLAKE3-XOF over those bytes is read for K*4 bytes and decoded as
// K little-endian u32 words; each is taken mod total_leaves.
//
// We accept the small modulo bias (≤ 2^32 / total_leaves) per the spec
// §3.4 — negligible cryptographic effect, rejection sampling would
// complicate test vectors.

namespace Akoya.Crypto;

public static class AuditIndexDeriver
{
    /// <summary>Spec domain separator. ASCII, no NUL, 14 bytes.</summary>
    public static ReadOnlySpan<byte> DomainSeparator =>
        "akoya-audit-v1"u8;

    /// <summary>Spec-imposed hard cap on K (audit_proof v1 §10/§3.2).</summary>
    public const int AuditKMax = 64;

    /// <summary>
    /// Derive K audit-leaf indices per the audit_proof v1 spec.
    /// Returns a freshly-allocated array of length <paramref name="auditK"/>.
    /// </summary>
    /// <exception cref="ArgumentException">Sizes wrong, K out of range, or total_leaves not a power of two.</exception>
    public static uint[] Derive(
        ReadOnlySpan<byte> claimedHash,
        ReadOnlySpan<byte> bSeed,
        uint auditK,
        ulong totalLeaves)
    {
        if (claimedHash.Length != 32)
            throw new ArgumentException("claimedHash must be 32 bytes.", nameof(claimedHash));
        if (bSeed.Length != 32)
            throw new ArgumentException("bSeed must be 32 bytes.", nameof(bSeed));
        if (auditK == 0)
            return [];
        if (auditK > AuditKMax)
            throw new ArgumentOutOfRangeException(nameof(auditK), auditK,
                $"audit_k must be in [0, {AuditKMax}] (spec §3.2 hard cap).");
        if (totalLeaves == 0)
            throw new ArgumentOutOfRangeException(nameof(totalLeaves), "total_leaves must be > 0.");
        if ((totalLeaves & (totalLeaves - 1)) != 0)
            throw new ArgumentException(
                $"total_leaves must be a power of two (spec §3.2(a) v1 constraint); got {totalLeaves}.",
                nameof(totalLeaves));

        Span<byte> xofInput = stackalloc byte[82];
        DomainSeparator.CopyTo(xofInput);
        claimedHash.CopyTo(xofInput[14..]);
        bSeed.CopyTo(xofInput[46..]);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(xofInput[78..], auditK);

        Span<byte> xofOut = stackalloc byte[AuditKMax * 4]; // ≤ 256 bytes, always safe
        var xofUsed = xofOut[..(int)(auditK * 4u)];
        Blake3.Xof(xofInput, xofUsed);

        var indices = new uint[auditK];
        for (int i = 0; i < (int)auditK; i++)
        {
            uint word = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(xofUsed.Slice(i * 4, 4));
            indices[i] = (uint)((ulong)word % totalLeaves);
        }
        return indices;
    }
}
