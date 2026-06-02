// BSeedExpander — deterministic per-miner B matrix generation.
//
// Algorithm:
//   1. BLAKE3_XOF(bSeed) → stream n*k bytes
//   2. Map each byte: (byte % 127) - 63 → int7 in [-63, +63]
//   3. Row-major layout: B[i,j] at offset i*k + j
//
// HashB = BLAKE3 keyed Merkle tree root over B (row-major, zero-padded
// to 1024-byte chunk boundary), keyed with jobKey.

using System.Buffers;

namespace Akoya.Crypto;

public static class BSeedExpander
{
    /// <summary>
    /// Expands a 32-byte BSeed into n*k bytes of int7 values in [-63, +63].
    /// Row-major: B[i,j] at offset i*k + j (same order as XOF stream).
    /// Returns the raw bytes (sbyte reinterpreted as byte).
    /// </summary>
    public static byte[] ExpandRaw(ReadOnlySpan<byte> bSeed, int n, int k)
    {
        var result = new byte[checked(n * k)];
        ExpandRaw(bSeed, n, k, result);
        return result;
    }

    /// <summary>
    /// Expands BSeed into a caller-supplied buffer (must be at least n*k bytes).
    /// Avoids allocation on repeated calls with the same dimensions.
    /// </summary>
    public static void ExpandRaw(ReadOnlySpan<byte> bSeed, int n, int k, Span<byte> destination)
    {
        if (bSeed.Length != 32) throw new ArgumentException("BSeed must be 32 bytes.", nameof(bSeed));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(n);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);

        int totalBytes = checked(n * k);
        if (destination.Length < totalBytes)
            throw new ArgumentException($"Destination must be at least {totalBytes} bytes.", nameof(destination));

        // Get raw XOF output
        var xofBuffer = ArrayPool<byte>.Shared.Rent(totalBytes);
        try
        {
            Blake3.Xof(bSeed, xofBuffer.AsSpan(0, totalBytes));

            for (int i = 0; i < totalBytes; i++)
                destination[i] = unchecked((byte)(sbyte)((xofBuffer[i] % 127) - 63));
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(xofBuffer, clearArray: true);
        }
    }

    /// <summary>
    /// Random-access expansion: re-derive the int7-encoded B bytes for the
    /// half-open byte range <c>[byteOffset, byteOffset + destination.Length)</c>
    /// of the conceptual full-length expansion, without materialising the
    /// rest. Output is the same byte layout as <see cref="ExpandRaw(ReadOnlySpan{byte}, int, int)"/>
    /// — i.e. <c>(byte)(sbyte)((xof_byte % 127) - 63)</c>.
    ///
    /// Used by AuditProofVerifier (and the audit_proof spec's pool-side leaf
    /// re-derivation) to recover the bytes of a single 1024-byte leaf at
    /// <c>leaf_idx * 1024</c>.
    /// </summary>
    public static void ExpandRangeRaw(ReadOnlySpan<byte> bSeed, ulong byteOffset, Span<byte> destination)
    {
        if (bSeed.Length != 32) throw new ArgumentException("BSeed must be 32 bytes.", nameof(bSeed));
        if (destination.IsEmpty) return;

        var xofBuffer = ArrayPool<byte>.Shared.Rent(destination.Length);
        try
        {
            var xofSpan = xofBuffer.AsSpan(0, destination.Length);
            Blake3.XofAt(bSeed, byteOffset, xofSpan);

            for (int i = 0; i < destination.Length; i++)
                destination[i] = unchecked((byte)(sbyte)((xofSpan[i] % 127) - 63));
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(xofBuffer, clearArray: true);
        }
    }

    /// <summary>
    /// Expands BSeed and computes the expected HashB as BLAKE3 keyed Merkle root.
    /// </summary>
    public static byte[] ExpandAndComputeHashB(ReadOnlySpan<byte> bSeed, int n, int k, ReadOnlySpan<byte> jobKey)
    {
        if (jobKey.Length != 32) throw new ArgumentException("Job key must be 32 bytes.", nameof(jobKey));

        var bBytes = ExpandRaw(bSeed, n, k);
        return Blake3.MerkleRoot(bBytes, jobKey);
    }
}
