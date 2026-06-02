// Narrow abstraction over MiningSession.EnqueueAsync so that PingPump,
// HeartbeatPump and any other producer can be tested without standing up a
// real gRPC connection. MiningSession implements this directly.
//
// Intentionally tiny: the only thing pumps need is "shove a MinerEvent onto
// the outbound queue". Anything bigger (e.g. accessing MinerId, SessionToken)
// belongs on MiningSession itself.

using PearlPool.Proto.V2;

namespace Akoya.Pool;

public interface IEventSink
{
    /// <summary>
    /// Enqueue a <see cref="MinerEvent"/> for transmission. May block if the
    /// transport's outbound buffer is full (back-pressures the caller).
    /// </summary>
    ValueTask EnqueueAsync(MinerEvent ev, CancellationToken ct);
}
