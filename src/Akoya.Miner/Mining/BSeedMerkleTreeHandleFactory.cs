using Akoya.Mining;

namespace Akoya.Miner.Mining;

internal static class BSeedMerkleTreeHandleFactory
{
    public static IMerkleTreeHandle BuildFromLeafCvs(
        byte[] leafCvs,
        ReadOnlySpan<byte> bSeed,
        ReadOnlySpan<byte> key,
        int numRows,
        int rowWidth)
    {
        try
        {
            return NativeBSeedMerkleTreeHandle.BuildFromLeafCvs(
                leafCvs,
                bSeed,
                key,
                numRows,
                rowWidth);
        }
        catch (EntryPointNotFoundException)
        {
            return BSeedMerkleTreeHandle.BuildFromLeafCvs(
                leafCvs,
                bSeed,
                key,
                numRows,
                rowWidth);
        }
    }
}
