// JobBus — broadcast-latest channel of SigmaContext for N GpuWorkers.
//
// Semantics required by the mining loop:
//   • Workers don't queue old jobs. Only the LATEST σ matters; everything
//     else is stale and wasted work.
//   • Late-joining workers must observe the current σ immediately on attach.
//   • Subscribers should be wakeable without spin-polling.
//
// Implementation: a single atomic reference to the current SigmaContext plus
// a "version" counter; each worker stashes the version it last consumed and
// awaits a TaskCompletionSource that fires on each Publish. New TCS is
// allocated per publish — workers always await the *next* event.

namespace Akoya.Miner.Mining;

internal sealed class JobBus
{
    private SigmaContext? _current;
    private long _version;
    private TaskCompletionSource _signal = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly object _gate = new();

    public SigmaContext? Current => Volatile.Read(ref _current);
    public long Version => Interlocked.Read(ref _version);

    public void Publish(SigmaContext ctx)
    {
        TaskCompletionSource prev;
        lock (_gate)
        {
            Volatile.Write(ref _current, ctx);
            Interlocked.Increment(ref _version);
            prev = _signal;
            _signal = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        }
        prev.TrySetResult();
    }

    /// <summary>Returns immediately if a newer σ than <paramref name="seenVersion"/> exists,
    /// otherwise awaits the next Publish. Returns (ctx, version).</summary>
    public async ValueTask<(SigmaContext ctx, long version)> WaitForJobAsync(
        long seenVersion, CancellationToken ct)
    {
        while (true)
        {
            TaskCompletionSource toAwait;
            SigmaContext? ctx;
            long ver;
            lock (_gate)
            {
                ctx = _current;
                ver = Interlocked.Read(ref _version);
                if (ctx is not null && ver > seenVersion) return (ctx, ver);
                toAwait = _signal;
            }
            // CRITICAL: do NOT use ct.Register(... TrySetCanceled(toAwait) ...)
            // here. `toAwait` is a SHARED TCS used to wake every waiter on
            // the bus; cancelling it would poison the wake signal for all
            // other GpuWorkers AND for the next Publish (which calls
            // TrySetResult on the same instance). Task.WaitAsync(ct)
            // observes cancellation WITHOUT touching the underlying task.
            await toAwait.Task.WaitAsync(ct).ConfigureAwait(false);
            ct.ThrowIfCancellationRequested();
        }
    }
}
