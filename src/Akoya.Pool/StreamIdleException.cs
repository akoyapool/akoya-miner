// Thrown by MiningSession.RunStreamAsync when the stream-liveness watchdog
// trips — i.e. no PoolEvent has arrived within AKOYA_POOL_STREAM_WATCHDOG_SEC
// (default 90s). The outer reconnect loop in Program.cs catches this via the
// generic `catch (Exception ex)` branch and applies exponential backoff
// before the next Register/Resume attempt.

namespace Akoya.Pool;

public sealed class StreamIdleException : Exception
{
    public StreamIdleException(string message) : base(message) { }
}
