// Minimal seam so WorkerLivenessWatchdog can be tested without needing a
// real GpuWorker (which requires a CUDA device, mining loop, etc).
//
// GpuWorker implements this — see GpuWorker.cs. Tests substitute a
// FakeLivenessTarget that lets them script LastProgressTicks.

namespace Akoya.Miner.Mining;

internal interface ILivenessTarget
{
    int GpuIndex { get; }
    long LastProgressTicks { get; }

    /// <summary>
    /// Monotonic <see cref="Environment.TickCount64"/> of the most recent
    /// event that resets the trigger-rate clock for this worker. These are:
    ///   • the FIRST σ install (arms the clock — gives the watchdog a baseline)
    ///   • a vardiff retarget (the share target changed, so the Poisson
    ///     trigger rate changed — old measurements are no longer valid)
    ///   • an actual share trigger (proves the current target is reachable)
    ///
    /// Plain σ rotations do NOT reset this — the share target nbits is
    /// unchanged across rotations, so the expected trigger rate is unchanged
    /// and we'd be hiding a stuck-target bug if we did.
    ///
    /// <see cref="WorkerLivenessWatchdog"/>'s trigger-rate guard is aggregate
    /// across all workers. It trips when no armed worker has advanced this
    /// tick within the no-trigger budget — that indicates the pool's
    /// aggregate share target is impossibly hard (typically a buggy vardiff)
    /// and we should reconnect.
    ///
    /// Special value <c>0</c> means "never had a σ installed"; the watchdog
    /// must NOT trip on this — there's nothing to measure yet.
    /// </summary>
    long LastTriggerOrSigmaTicks { get; }
}
