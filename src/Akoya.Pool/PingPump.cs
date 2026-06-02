// PingPump — periodic PingEvent → MiningSession; on Pong, records RTT.
//
// V2 introduces a separate Ping/Pong RTT probe (was bundled into v1
// Heartbeat). Cadence is operator-configurable (AKOYA_POOL_PING_INTERVAL_SEC).
//
// Ping/Pong are unmatched at the protocol level (both messages carry only a
// timestamp). We assume FIFO under a single in-flight stream and treat any
// Pong as a response to the most-recent unanswered Ping. Wall-clock isn't
// guaranteed to match between miner and pool, so RTT is computed from the
// LOCAL monotonic clock: we stash the local send-ts in `_pingSentTicks` and
// compute `now - sent` on Pong, ignoring the timestamp inside the Pong.
//
// The pump injects RTT into `IRttSink` (typically `Metrics.SetPoolLatencyMs`).

using Microsoft.Extensions.Logging;
using PearlPool.Proto.V2;

namespace Akoya.Pool;

public interface IRttSink
{
    void RecordRttMs(double ms);
}

public sealed class PingPump : IAsyncDisposable
{
    private readonly IEventSink _session;
    private readonly TimeSpan _interval;
    private readonly IRttSink _rtt;
    private readonly ILogger<PingPump> _log;
    private readonly CancellationTokenSource _cts = new();
    private Task? _loopTask;

    // Most-recent ping send-time, local monotonic ticks. Read by HandlePong,
    // written by the loop. Interlocked so the read is torn-free.
    private long _pingSentTicks;

    public PingPump(IEventSink session, TimeSpan interval, IRttSink rtt,
                    ILogger<PingPump> log)
    {
        _session  = session;
        _interval = interval > TimeSpan.Zero ? interval : TimeSpan.FromSeconds(15);
        _rtt      = rtt;
        _log      = log;
    }

    public void Start(CancellationToken external)
    {
        var linked = CancellationTokenSource.CreateLinkedTokenSource(external, _cts.Token);
        _loopTask = Task.Run(() => Loop(linked.Token), linked.Token);
    }

    /// <summary>Hook this up to <see cref="MiningSessionCallbacks.OnPong"/>.</summary>
    public ValueTask HandlePong(PongEvent _)
    {
        var sent = Interlocked.Read(ref _pingSentTicks);
        if (sent == 0) return ValueTask.CompletedTask;
        // _pingSentTicks holds Environment.TickCount64, which is already in
        // milliseconds. (NOT 100ns ticks — TimeSpan.FromTicks would be wrong.)
        var ms = (double)(Environment.TickCount64 - sent);
        if (ms is >= 0 and < 60_000)
        {
            _rtt.RecordRttMs(ms);
        }
        return ValueTask.CompletedTask;
    }

    private async Task Loop(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                Interlocked.Exchange(ref _pingSentTicks, Environment.TickCount64);
                var ev = new MinerEvent
                {
                    Ping = new PingEvent { Timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() },
                };
                await _session.EnqueueAsync(ev, ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException) { break; }
            catch (Exception e) { _log.LogDebug("ping: send failed ({Err})", e.Message); }

            try { await Task.Delay(_interval, ct).ConfigureAwait(false); }
            catch (OperationCanceledException) { break; }
        }
    }

    public async ValueTask DisposeAsync()
    {
        try { _cts.Cancel(); } catch { /* already disposed */ }
        if (_loopTask is not null)
        {
            try { await _loopTask.ConfigureAwait(false); } catch { /* shutdown */ }
        }
        _cts.Dispose();
    }
}
