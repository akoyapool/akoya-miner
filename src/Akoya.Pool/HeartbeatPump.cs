// HeartbeatPump — periodic Heartbeat with per-GPU hashrate.
//
// V2 Heartbeat carries:
//   • timestamp (unix-ms)
//   • repeated PerGpuHashrate { gpu_uuid, hashrate_5m, shares_5m }
//   • current_hashrate  (sum across GPUs)
//   • latency           (optional — last-known RTT)
//
// The pump pulls a stats snapshot from `IHeartbeatSource` and emits a
// MinerEvent.Heartbeat. We don't track 5m windows here — that's the source's
// concern. (In practice this is fed by Metrics.GetSnapshot()-derived values.)

using Microsoft.Extensions.Logging;
using PearlPool.Proto.V2;

namespace Akoya.Pool;

public readonly record struct GpuHashrateSample(string GpuUuid, double Hashrate5m, uint Shares5m);

public sealed record HeartbeatSnapshot(
    IReadOnlyList<GpuHashrateSample> PerGpu,
    double TotalHashrate,
    double LatencyMs);

public interface IHeartbeatSource
{
    HeartbeatSnapshot Sample();
}

public sealed class HeartbeatPump : IAsyncDisposable
{
    private readonly IEventSink _session;
    private readonly TimeSpan _interval;
    private readonly IHeartbeatSource _source;
    private readonly ILogger<HeartbeatPump> _log;
    private readonly CancellationTokenSource _cts = new();
    private Task? _loopTask;

    public HeartbeatPump(IEventSink session, TimeSpan interval,
                         IHeartbeatSource source, ILogger<HeartbeatPump> log)
    {
        _session  = session;
        _interval = interval > TimeSpan.Zero ? interval : TimeSpan.FromSeconds(30);
        _source   = source;
        _log      = log;
    }

    public void Start(CancellationToken external)
    {
        var linked = CancellationTokenSource.CreateLinkedTokenSource(external, _cts.Token);
        _loopTask = Task.Run(() => Loop(linked.Token), linked.Token);
    }

    private async Task Loop(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                var snap = _source.Sample();
                var hb = new Heartbeat
                {
                    Timestamp       = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                    CurrentHashrate = snap.TotalHashrate,
                    Latency         = snap.LatencyMs,
                    // SequenceNumber is set inside MiningSession.EnqueueAsync from
                    // the outer MinerEvent.seq; we leave the inner field 0.
                };
                foreach (var g in snap.PerGpu)
                {
                    hb.GpuHashrates.Add(new PerGpuHashrate
                    {
                        GpuUuid   = g.GpuUuid,
                        Hashrate5M = g.Hashrate5m,
                        Shares5M   = g.Shares5m,
                    });
                }

                var ev = new MinerEvent { Heartbeat = hb };
                await _session.EnqueueAsync(ev, ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException) { break; }
            catch (Exception e) { _log.LogDebug("heartbeat: send failed ({Err})", e.Message); }

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
