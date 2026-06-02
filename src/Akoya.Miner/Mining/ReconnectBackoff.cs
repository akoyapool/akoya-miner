namespace Akoya.Miner.Mining;

/// <summary>
/// Pure functions over the outer reconnect-loop backoff math. Extracted from
/// Program.cs so each invariant has a deterministic unit test. The invariant
/// we MUST hold:
///
///   • A flapping gateway (RST every second) must never let us spin at line
///     speed. Backoff grows exponentially with attempt count and caps at
///     <see cref="CapSeconds"/>.
///   • Independent miners must not thunder-herd a recovering pool: every
///     delay gets ±<see cref="JitterFraction"/> uniform jitter.
///   • A server-supplied ReconnectHint can park us much longer (planned
///     drain) but must be clamped to <see cref="MaxReconnectHintSeconds"/>
///     so a buggy/hostile gateway can't wedge us for hours.
///
/// All functions are deterministic given a supplied jitter sample in
/// [-1.0, +1.0] (the caller wires up Random.Shared in production and a
/// fixed value in tests).
/// </summary>
internal static class ReconnectBackoff
{
    /// <summary>Cap on the base exponential before jitter. The cap kicks
    /// in at attempt 6 (2^6 = 64 → 60s).</summary>
    public const double CapSeconds = 60.0;

    /// <summary>Floor on the resulting delay. Below 500ms the loop would
    /// be effectively spin-retrying.</summary>
    public const double FloorSeconds = 0.5;

    /// <summary>Symmetric jitter fraction applied to the exponential base.
    /// 0.25 = ±25%.</summary>
    public const double JitterFraction = 0.25;

    /// <summary>Maximum number of attempts we let influence the exponent.
    /// Beyond this the cap is in force anyway, but we clamp to avoid
    /// overflow paranoia.</summary>
    public const int MaxAttemptForExponent = 6;

    /// <summary>Hard ceiling on a server-supplied ReconnectHint. 300s is
    /// well above the longest legitimate deploy window quoted by the pool
    /// integration doc and short enough that Ctrl-C is never far away.</summary>
    public const double MaxReconnectHintSeconds = 300.0;

    /// <summary>±10% jitter on ReconnectHint waits. Smaller than the
    /// generic-failure jitter because the pool already coordinated the
    /// timing — we just don't want every fleet member to reconnect at
    /// exactly the same wall-clock second.</summary>
    public const double HintJitterFraction = 0.10;

    /// <summary>
    /// Compute the delay before the next reconnect attempt after a
    /// failure. <paramref name="jitterSample"/> must be in [-1.0, +1.0];
    /// the production caller passes <c>(Random.Shared.NextDouble()*2)-1</c>.
    /// </summary>
    public static TimeSpan ComputeDelay(int attempt, double jitterSample)
    {
        if (attempt < 1) attempt = 1;
        var exp     = System.Math.Min(attempt, MaxAttemptForExponent);
        var baseSec = System.Math.Min(CapSeconds, System.Math.Pow(2, exp));
        var jittered = baseSec * (1.0 + (jitterSample * JitterFraction));
        var clamped  = System.Math.Max(FloorSeconds, jittered);
        return TimeSpan.FromSeconds(clamped);
    }

    /// <summary>
    /// Apply the ReconnectHint clamp + jitter. <paramref name="jitterSample"/>
    /// in [-1.0, +1.0]. Returns null if the hint is non-positive (means
    /// "no hint").
    /// </summary>
    public static TimeSpan? ApplyHint(double hintSeconds, double jitterSample)
    {
        if (hintSeconds <= 0) return null;
        var clamped  = System.Math.Min(hintSeconds, MaxReconnectHintSeconds);
        var jittered = clamped * (1.0 + (jitterSample * HintJitterFraction));
        return TimeSpan.FromSeconds(System.Math.Max(0.0, jittered));
    }

    /// <summary>Was the hint clamped by <see cref="MaxReconnectHintSeconds"/>?
    /// Used by the caller to emit a warning log.</summary>
    public static bool HintWasClamped(double hintSeconds) =>
        hintSeconds > MaxReconnectHintSeconds;
}
