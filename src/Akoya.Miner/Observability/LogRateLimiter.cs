namespace Akoya.Miner.Observability;

public sealed class LogRateLimiter
{
    private readonly long _intervalMs;
    private long _nextAllowedTicks;
    private long _suppressedSinceLast;

    public LogRateLimiter(TimeSpan interval)
    {
        if (interval <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(interval), "interval must be positive");
        _intervalMs = (long)interval.TotalMilliseconds;
        _nextAllowedTicks = 0; // first call always logs
    }

    /// <summary>
    /// True if the caller should emit a log line right now. Outputs
    /// <paramref name="suppressedSinceLast"/> = count of TryLog calls
    /// that returned false since the last true return, so the caller can
    /// surface "suppressed N similar" to the operator.
    /// </summary>
    public bool TryLog(out int suppressedSinceLast)
    {
        var now = Environment.TickCount64;
        var next = Volatile.Read(ref _nextAllowedTicks);
        if (now < next)
        {
            Interlocked.Increment(ref _suppressedSinceLast);
            suppressedSinceLast = 0;
            return false;
        }
        // Claim the slot. CompareExchange to avoid a thundering-herd: only
        // one caller across N threads gets to emit the line in the window.
        var claimed = Interlocked.CompareExchange(
            ref _nextAllowedTicks, now + _intervalMs, next);
        if (claimed != next)
        {
            Interlocked.Increment(ref _suppressedSinceLast);
            suppressedSinceLast = 0;
            return false;
        }
        suppressedSinceLast = (int)Math.Min(int.MaxValue,
            Interlocked.Exchange(ref _suppressedSinceLast, 0));
        return true;
    }

    /// <summary>Drain the suppressed counter without consuming a slot —
    /// useful for periodic "still failing" health reports.</summary>
    public int SnapshotSuppressed()
        => (int)Math.Min(int.MaxValue, Volatile.Read(ref _suppressedSinceLast));
}
