using System.Runtime.InteropServices;
using Akoya.Mining;

namespace Akoya.Miner.Mining;

internal static class NativeBSeedExpansion
{
    public static unsafe void ExpandRaw(ReadOnlySpan<byte> bSeed, int n, int k, Span<byte> destination)
    {
        if (bSeed.Length != SigmaContext.BSeedSize)
            throw new ArgumentException("BSeed must be 32 bytes.", nameof(bSeed));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(n);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);

        int totalBytes = checked(n * k);
        if (destination.Length < totalBytes)
            throw new ArgumentException($"Destination must be at least {totalBytes} bytes.", nameof(destination));

        ExpandNative(bSeed, n, k, destination[..totalBytes]);
    }

    public static unsafe void ExpandRangeRaw(ReadOnlySpan<byte> bSeed, ulong byteOffset, Span<byte> destination)
    {
        if (bSeed.Length != SigmaContext.BSeedSize)
            throw new ArgumentException("BSeed must be 32 bytes.", nameof(bSeed));
        if (destination.IsEmpty)
            return;

        byte* errMsg = null;
        try
        {
            fixed (byte* pSeed = bSeed)
            fixed (byte* pOut = destination)
            {
                int rc = PearlMiningNative.BSeedExpandRangeRaw(
                    pSeed,
                    byteOffset,
                    pOut,
                    (nuint)destination.Length,
                    &errMsg);

                if (rc == 0) return;

                var msg = errMsg != null
                    ? Marshal.PtrToStringUTF8((IntPtr)errMsg) ?? "(no message)"
                    : "(no message)";
                throw new MerkleRootAndProofException(rc, msg);
            }
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException(
                "Native BSeed range expansion requires pearl_capi_bseed_expand_range_raw. Rebuild the pearl mining CAPI library.",
                ex);
        }
        finally
        {
            if (errMsg != null) PearlMiningNative.FreeString(errMsg);
        }
    }

    private static unsafe void ExpandNative(ReadOnlySpan<byte> bSeed, int n, int k, Span<byte> destination)
    {
        byte* errMsg = null;
        try
        {
            fixed (byte* pSeed = bSeed)
            fixed (byte* pOut = destination)
            {
                int rc = PearlMiningNative.BSeedExpandRaw(
                    pSeed,
                    (nuint)n,
                    (nuint)k,
                    pOut,
                    (nuint)destination.Length,
                    &errMsg);

                if (rc == 0) return;

                var msg = errMsg != null
                    ? Marshal.PtrToStringUTF8((IntPtr)errMsg) ?? "(no message)"
                    : "(no message)";
                throw new MerkleRootAndProofException(rc, msg);
            }
        }
        catch (EntryPointNotFoundException ex)
        {
            throw new InvalidOperationException(
                "Native BSeed expansion requires pearl_capi_bseed_expand_raw. Rebuild the pearl mining CAPI library.",
                ex);
        }
        finally
        {
            if (errMsg != null) PearlMiningNative.FreeString(errMsg);
        }
    }
}
