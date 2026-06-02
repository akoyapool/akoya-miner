// Shutdown-deadline guard. After cancellation is requested (SIGTERM /
// SIGINT / SIGHUP / SIGQUIT / Ctrl-C), the main loop MUST exit within a
// bounded time. If it doesn't — typically because a worker's CUDA call
// or a hung native handle won't return — we have to force-exit ourselves
// before systemd/k8s SIGKILLs us mid-share-submit.
//
// Why this matters operationally:
//   • k8s default terminationGracePeriodSeconds is 30s, then SIGKILL.
//   • systemd default TimeoutStopSec is 90s.
//   • HiveOS sends SIGTERM then SIGKILL after ~10s in some configs.
// In every case, a process that ignores SIGTERM and gets SIGKILLed loses
// any in-flight share + leaves the GPU in a less-clean state than a
// graceful Environment.Exit. We'd rather take the shutdown into our own
// hands at ~30s than wait for the supervisor's hammer at 30-90s.
//
// Contract:
//   • Arm() returns an IDisposable. Dispose to cancel the deadline (the
//     main loop exited cleanly before the deadline).
//   • When `ct` cancels, a one-shot timer is started. If it fires before
//     the returned disposable is disposed, `onExpire` runs.
//   • Thread-safe; Arm/Dispose can race without harm.
//   • Pure helper: `onExpire` is injected so tests can observe it without
//     terminating the test runner.
//
// Default integration: Program.cs passes `() => Environment.Exit(124)`.
// 124 matches GNU coreutils `timeout`(1) — operators can grep for it.

using Microsoft.Extensions.Logging;

namespace Akoya.Miner.Observability;

public static class ShutdownDeadline
{
    /// <summary>Process exit code when the deadline fires. 124 matches
    /// GNU <c>timeout</c>(1) so operators can pattern-match.</summary>
    public const int HardExitCode = 124;

    [System.Diagnostics.CodeAnalysis.SuppressMessage("Design", "CA1068:CancellationToken parameters must come last",
        Justification = "CT is the subject of the deadline — it reads more naturally first; preserved for API stability.")]
    public static IDisposable Arm(
        CancellationToken shutdownCt,
        TimeSpan deadline,
        Action onExpire,
        ILogger log)
    {
        if (deadline <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(deadline), "deadline must be positive");
        ArgumentNullException.ThrowIfNull(onExpire);
        ArgumentNullException.ThrowIfNull(log);

        var cancelDeadline = new CancellationTokenSource();
        CancellationTokenRegistration reg = default;
        reg = shutdownCt.Register(() =>
        {
            // Fire-and-forget: a delayed force-exit. If the main loop
            // unwinds cleanly first, Dispose() on the returned handle
            // cancels this delay.
            _ = Task.Run(async () =>
            {
                try
                {
                    await Task.Delay(deadline, cancelDeadline.Token).ConfigureAwait(false);
                    log.LogCritical(
                        "shutdown: deadline ({D:F1}s) elapsed after cancellation — force-exiting (exit {Code})",
                        deadline.TotalSeconds, HardExitCode);
                    try { onExpire(); }
                    catch (Exception ex)
                    {
                        // Last-ditch: onExpire itself threw. Nothing left
                        // to try except a hard fall-through.
                        log.LogCritical(ex, "shutdown: onExpire threw — process will exit via runtime");
                    }
                }
                catch (OperationCanceledException) { /* clean shutdown beat the deadline */ }
            });
        });

        return new DisposableAction(() =>
        {
            cancelDeadline.Cancel();
            reg.Dispose();
            cancelDeadline.Dispose();
        });
    }

    private sealed class DisposableAction : IDisposable
    {
        private Action? _onDispose;
        public DisposableAction(Action onDispose) => _onDispose = onDispose;
        public void Dispose() => Interlocked.Exchange(ref _onDispose, null)?.Invoke();
    }
}
