// Monotonic uint64 source for MinerEvent.seq / PoolEvent.seq.
//
// V2 protocol requires both directions to carry a per-stream monotonically
// increasing sequence number starting at 1. Used by the server to detect
// dropped messages on a noisy connection and by us to detect server-side
// replays after a reconnect.
//
// Lifecycle: one SequenceCounter per MiningStream. Disposed when the stream
// is torn down — a Resume opens a fresh stream and gets a fresh counter
// (the server's seq numbering is also per-stream, see MinerServiceImpl.cs).

namespace Akoya.Pool;

public sealed class SequenceCounter
{
    private long _value;   // long for Interlocked; cast to ulong on Next()

    /// <summary>Returns the next seq starting at 1. Wraps after ~9.2e18; in
    /// practice the connection will be torn down for unrelated reasons long
    /// before that.</summary>
    public ulong Next() => unchecked((ulong)Interlocked.Increment(ref _value));

    /// <summary>Current last-issued seq, or 0 if Next() has not been called.
    /// Intended for diagnostics / metrics only.</summary>
    public ulong Current => unchecked((ulong)Interlocked.Read(ref _value));
}
