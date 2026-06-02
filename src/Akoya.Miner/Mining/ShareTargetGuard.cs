using System.Numerics;
using Akoya.Crypto;
using Akoya.Mining;

namespace Akoya.Miner.Mining;

/// <summary>
/// Pure predicate for "does this claimed-hash still clear the live target?"
///
/// Background: A GPU iteration that triggers because its claimed hash &lt; target
/// can race with a vardiff that has just tightened the target. By the time
/// we ShareBuilder.Build and call SubmitAsync, the tile may be invalid.
/// Submitting anyway wastes a slot AND signals to the pool that we have a
/// CPU/GPU disagreement (loud noise, no upside). Drop client-side.
///
/// Extracted from <see cref="GpuWorker"/>.OnTrigger so this invariant has
/// a unit test (the trigger path itself requires a real GPU).
/// </summary>
internal static class ShareTargetGuard
{
    /// <summary>
    /// Returns true if the share is still valid — i.e. the claimed hash
    /// (little-endian, unsigned, 32 bytes) is ≤ the adjusted live target
    /// derived from <paramref name="installedNbits"/> × difficulty-adjustment
    /// factor in <paramref name="cfg"/>. The adjusted target is clamped to
    /// 2^256-1 so a DAF > 1.0 at the maximum-difficulty boundary doesn't
    /// overflow the 256-bit space.
    /// </summary>
    public static bool ClearsLiveTarget(
        ReadOnlySpan<byte> claimedHashLittleEndian,
        uint installedNbits,
        MiningConfiguration cfg)
    {
        var liveTarget = SigmaContext.NbitsToTarget(installedNbits)
                       * cfg.DifficultyAdjustmentFactor();
        if (liveTarget >= BigInteger.One << 256)
            liveTarget = (BigInteger.One << 256) - 1;

        var hashInt = new BigInteger(claimedHashLittleEndian, isUnsigned: true, isBigEndian: false);
        return hashInt <= liveTarget;
    }
}
