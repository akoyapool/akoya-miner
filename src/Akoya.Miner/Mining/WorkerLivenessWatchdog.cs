// WorkerLivenessWatchdog — guards against GPU-side deadlocks AND mining
// stalls caused by a too-hard share target.
//
// Each GpuWorker exposes two monotonic ticks via ILivenessTarget:
//
//   1. LastProgressTicks — bumped every loop iteration. If this goes stale
//      past `progressBudget`, the GPU is wedged (kernel hang, driver crash,
//      stuck cuStreamSync). We trip the orchestrator's cancellation source
//      to force a full reconnect.
//
//   2. LastTriggerOrSigmaTicks — bumped on first σ install / vardiff retarget /
//      actual share trigger. The no-trigger guard is aggregate across all
//      workers: V2 has one pool connection and one vardiff target sized for
//      the whole rig, so a single GPU can be share-silent for a long time
//      while the miner is healthy. If every armed worker goes stale past
//      `noTriggerBudget`, the pool has handed us a target we can't meet —
//      typically a bad vardiff. Same response: trip the CTS so we reconnect
//      and (hopefully) negotiate a sane target.
//
// Hard escalation: if the workers are still stale `HardKillGrace` after we
// cancelled (i.e. they're wedged in an uninterruptible native call and won't
// honour the CTS), we Environment.Exit(70) so the container supervisor
// (Docker / systemd / HiveOS) restarts the whole process. This is the only
// way out of a truly deadlocked CUDA driver. Note: hard-kill only applies
// to the progress-budget path — a no-trigger trip is by definition not
// wedged, so the orchestrator's reconnect path will unwind cleanly.

using System.Diagnostics;
using Microsoft.Extensions.Logging;

namespace Akoya.Miner.Mining;

internal sealed class WorkerLivenessWatchdog : IAsyncDisposable
{
    private static readonly TimeSpan DefaultHardKillGrace = TimeSpan.FromSeconds(30);

    private readonly IReadOnlyList<ILivenessTarget> _workers;
    private readonly TimeSpan _budget;
    private readonly TimeSpan _noTriggerBudget;
    private readonly TimeSpan _hardKillGrace;
    private readonly CancellationTokenSource _tripCts;
    private readonly ILogger _log;
    private readonly Action<int> _exitAction;
    private readonly Action<string, Exception?>? _tripReasonSink;
    private readonly CancellationTokenSource _ownCts = new();
    private Task? _loop;

    // Test hook — when true, the hard-kill escalation loop polls every
    // 100ms instead of 1s. Production never touches this; tests set it via
    // the dedicated constructor below.
    private readonly TimeSpan _hardKillPollInterval;

    public WorkerLivenessWatchdog(
        IReadOnlyList<ILivenessTarget> workers,
        TimeSpan budget,
        TimeSpan noTriggerBudget,
        CancellationTokenSource tripCts,
        ILogger log,
        Action<string, Exception?>? tripReasonSink = null)
        : this(workers, budget, noTriggerBudget, tripCts, log,
               exitAction: Environment.Exit,
               hardKillGrace: DefaultHardKillGrace,
               hardKillPollInterval: TimeSpan.FromSeconds(1),
               tripReasonSink: tripReasonSink) { }

    // Test-only ctor: lets a test inject a non-process-killing exit action
    // and tune the hard-kill grace / poll interval so the bounded-budget
    // assertions stay sub-second.
    internal WorkerLivenessWatchdog(
        IReadOnlyList<ILivenessTarget> workers,
        TimeSpan budget,
        TimeSpan noTriggerBudget,
        CancellationTokenSource tripCts,
        ILogger log,
        Action<int> exitAction,
        TimeSpan hardKillGrace,
        TimeSpan hardKillPollInterval,
        Action<string, Exception?>? tripReasonSink = null)
    {
        _workers              = workers;
        _budget               = budget;
        _noTriggerBudget      = noTriggerBudget;
        _tripCts              = tripCts;
        _log                  = log;
        _exitAction           = exitAction;
        _hardKillGrace        = hardKillGrace;
        _hardKillPollInterval = hardKillPollInterval;
        _tripReasonSink       = tripReasonSink;
    }

    public void Start()
    {
        if (_workers.Count == 0) return;
        if (_budget <= TimeSpan.Zero && _noTriggerBudget <= TimeSpan.Zero) return;
        // Poll cadence is driven by the tighter of the two budgets.
        var driveBudget = _budget > TimeSpan.Zero
            ? (_noTriggerBudget > TimeSpan.Zero ? (_budget < _noTriggerBudget ? _budget : _noTriggerBudget) : _budget)
            : _noTriggerBudget;
        var poll = TimeSpan.FromMilliseconds(Math.Max(500, driveBudget.TotalMilliseconds / 3.0));
        _loop = Task.Run(() => LoopAsync(poll, _ownCts.Token));
    }

    private async Task LoopAsync(TimeSpan poll, CancellationToken ct)
    {
        // Prime every worker so we don't false-trip before the first iteration.
        var primeBy = Environment.TickCount64;
        try
        {
            while (!ct.IsCancellationRequested)
            {
                await Task.Delay(poll, ct).ConfigureAwait(false);

                var now = Environment.TickCount64;

                // ── Progress budget (GPU-wedge guard) ──────────────────────
                long stalest = -1; int stalestGpu = -1;
                if (_budget > TimeSpan.Zero)
                {
                    foreach (var w in _workers)
                    {
                        var last = w.LastProgressTicks;
                        var since = now - (last == 0 ? primeBy : last);
                        if (since > stalest) { stalest = since; stalestGpu = w.GpuIndex; }
                    }
                }

                if (_budget > TimeSpan.Zero
                    && stalest > (long)_budget.TotalMilliseconds
                    && !_tripCts.IsCancellationRequested)
                {
                    _log.LogError(
                        "worker-watchdog: GPU {Gpu} stalled {Stalled}ms (> budget {Budget}ms) — forcing reconnect",
                        stalestGpu, stalest, (long)_budget.TotalMilliseconds);
                    _tripReasonSink?.Invoke("worker_watchdog_progress", null);
                    try { _tripCts.Cancel(); } catch { /* already disposed */ }

                    // Hard-kill escalation: if workers still haven't budged
                    // `_hardKillGrace` after cancellation, they're wedged in
                    // an uninterruptible native call. Bail the process so
                    // the supervisor restarts us cleanly.
                    var killDeadline = Stopwatch.StartNew();
                    while (killDeadline.Elapsed < _hardKillGrace)
                    {
                        await Task.Delay(_hardKillPollInterval, CancellationToken.None).ConfigureAwait(false);
                        var nowK = Environment.TickCount64;
                        bool anyAlive = false;
                        foreach (var w in _workers)
                        {
                            if (nowK - w.LastProgressTicks < (long)_budget.TotalMilliseconds)
                            { anyAlive = true; break; }
                        }
                        if (anyAlive) return; // workers recovered, our job is done
                    }

                    _log.LogCritical(
                        "worker-watchdog: workers still wedged {Grace}s after cancel — Environment.Exit(70)",
                        (int)_hardKillGrace.TotalSeconds);
                    _exitAction(70);
                    return;
                }

                // ── Aggregate no-trigger budget (bad-vardiff / impossible-target guard)
                //
                // Workers are iterating fine (progress-budget would have
                // caught a dead GPU otherwise) but the miner as a whole has
                // not triggered for noTriggerBudget. In V2 the pool target is
                // sized for aggregate hashrate across all GPUs, not each GPU
                // independently; using the worst individual worker here
                // causes false reconnects when one card is simply unlucky.
                // Ignore workers whose LastTriggerOrSigmaTicks == 0 — that
                // means they've never had a σ installed yet (just connected,
                // or pool hasn't sent JobAssignment), so there's nothing to
                // measure.
                if (_noTriggerBudget > TimeSpan.Zero && !_tripCts.IsCancellationRequested)
                {
                    long newestTriggerOrSigma = 0;
                    foreach (var w in _workers)
                    {
                        var last = w.LastTriggerOrSigmaTicks;
                        if (last == 0) continue; // never had σ — skip
                        if (last > newestTriggerOrSigma) newestTriggerOrSigma = last;
                    }

                    if (newestTriggerOrSigma != 0)
                    {
                        var aggregateNoTrigger = now - newestTriggerOrSigma;
                        if (aggregateNoTrigger > (long)_noTriggerBudget.TotalMilliseconds)
                        {
                            _log.LogError(
                                "worker-watchdog: miner produced no aggregate triggers for {Stalled}ms (> {Budget}ms) — "
                                + "share target likely too hard (bad vardiff). Forcing reconnect",
                                aggregateNoTrigger, (long)_noTriggerBudget.TotalMilliseconds);
                            _tripReasonSink?.Invoke("worker_watchdog_no_trigger", null);
                            try { _tripCts.Cancel(); } catch { /* already disposed */ }
                            // No hard-kill — workers aren't wedged, just unlucky or on a bad target.
                            return;
                        }
                    }
                }
            }
        }
        catch (OperationCanceledException) { /* normal shutdown */ }
        catch (Exception ex)
        {
            // CRITICAL: a fatal in the watchdog itself is a worst-case
            // outcome — if we silently log-and-exit, NOTHING is watching
            // the workers any more. They could hang and the process would
            // sit there forever. So: log Critical, capture for diagnostics,
            // AND trip the reconnect CTS. A reconnect cycle tears down
            // workers + watchdog + recreates them; better to recycle
            // unnecessarily than to run unwatched.
            _log.LogCritical(ex,
                "worker-watchdog: loop crashed — tripping reconnect so workers are not left unwatched");
            Volatile.Write(ref _lastFatal, ex);
            _tripReasonSink?.Invoke("worker_watchdog_loop_crashed", ex);
            try { _tripCts.Cancel(); } catch { /* already disposed */ }
        }
    }

    // Most recent fatal that crashed the LoopAsync, if any. Tests / operators
    // can inspect this to confirm the trip path fired for the right reason
    // (rather than via the progress or no-trigger budget).
    private Exception? _lastFatal;
    public Exception? LastFatal => Volatile.Read(ref _lastFatal);

    public async ValueTask DisposeAsync()
    {
        try { _ownCts.Cancel(); } catch { }
        if (_loop is not null)
        {
            try { await _loop.ConfigureAwait(false); } catch { }
        }
        _ownCts.Dispose();
    }
}
