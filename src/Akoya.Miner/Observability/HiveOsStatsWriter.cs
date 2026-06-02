// HiveOsStatsWriter — periodic JSON stats file for HiveOS integration.
//
// Writes /run/hive/akoya-miner-stats.json (or a configurable path) every N
// seconds. HiveOS's h-stats.sh reads this file and reformats it for the
// dashboard. Atomic write via temp+rename.

using System.Diagnostics;
using System.Text.Json;
using Akoya.MinerCore;
using Microsoft.Extensions.Logging;

namespace Akoya.Miner.Observability;

internal sealed class HiveOsStatsWriter : IDisposable
{
    private readonly string _statsPath;
    private readonly TimeSpan _interval;
    private readonly MetricsSampler _sampler;
    private readonly Stopwatch _uptime;
    private readonly Func<bool> _isConnected;
    private readonly ILogger _log;
    private readonly CancellationTokenSource _cts = new();
    private readonly Thread _thread;
    // Rate-limit the "failed to write stats file" warning. At a 5s
    // interval and a permanently-broken FS path, the raw path would emit
    // 17,280 identical warnings per day. The pool/share telemetry isn't
    // affected by this failure — only the HiveOS dashboard is — so heavy
    // throttling is the right call. One warning every 5 minutes preserves
    // signal without spam.
    private readonly LogRateLimiter _writeFailureRl = new(TimeSpan.FromMinutes(5));
    // Set to true once we determine the stats directory can never be
    // written to (e.g. /run/hive doesn't exist on a non-HiveOS host and
    // we lack the permissions to create it). In that mode the Loop
    // becomes a no-op — we don't want to spam warnings for a feature
    // the operator simply isn't using.
    private readonly bool _disabled;

    public HiveOsStatsWriter(
        string statsPath,
        TimeSpan interval,
        MetricsSampler sampler,
        Stopwatch uptime,
        Func<bool> isConnected,
        ILogger log)
    {
        _statsPath   = statsPath;
        _interval    = interval;
        _sampler     = sampler;
        _uptime      = uptime;
        _isConnected = isConnected;
        _log         = log;

        // Up-front feasibility check: if the stats directory doesn't
        // exist AND we cannot create it, this writer is permanently
        // disabled. The HiveOS integration is an opt-in feature — on
        // dev machines, plain containers, etc. the directory is absent
        // by design. Logging one info line and then going silent is
        // strictly better than warning every 5 minutes for the life of
        // the process.
        _disabled = !CanWriteToStatsDir(statsPath, log);

        _thread = new Thread(Loop) { IsBackground = true, Name = "hiveos-stats" };
        _thread.Start();
    }

    private static bool CanWriteToStatsDir(string statsPath, ILogger log)
    {
        var dir = Path.GetDirectoryName(statsPath);
        if (string.IsNullOrEmpty(dir))
            return true; // CWD-relative path; assume usable.

        if (Directory.Exists(dir))
            return true;

        try
        {
            Directory.CreateDirectory(dir);
            return true;
        }
        catch (Exception ex)
        {
            log.LogInformation(
                "hiveos: stats directory {Dir} not present and not creatable " +
                "({Reason}) — HiveOS integration disabled for this run",
                dir, ex.GetType().Name);
            return false;
        }
    }

    private void Loop()
    {
        if (_disabled) return;

        // Give the miner a few seconds to produce real data before first write.
        try { Thread.Sleep(3000); }
        catch (ThreadInterruptedException) { return; }

        while (!_cts.IsCancellationRequested)
        {
            try { WriteStats(); }
            catch (Exception ex)
            {
                if (_writeFailureRl.TryLog(out var suppressed))
                {
                    _log.LogWarning(ex,
                        "hiveos: failed to write stats file (suppressed {N} similar in last 5m)",
                        suppressed);
                }
            }

            try { Thread.Sleep(_interval); }
            catch (ThreadInterruptedException) { break; }
        }
    }

    private void WriteStats()
    {
        var gpuTelemetry = _sampler.LatestGpuStats;
        var snapshot     = Metrics.GetSnapshot();

        double totalTmads = 0;
        double totalExpectedOpens = 0;
        double totalTiles = 0;
        long totalAccepted = 0, totalRejected = 0;
        for (int i = 0; i < snapshot.GpuCount; i++)
        {
            totalTmads         += snapshot.TmadsPerSec[i];
            totalExpectedOpens += snapshot.ExpectedOpensPerSec[i];
            totalTiles         += snapshot.TilesPerSec[i];
            totalAccepted      += snapshot.Accepted[i];
            totalRejected      += snapshot.Rejected[i];
        }

        using var ms = new MemoryStream(1024);
        using var w = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = false });

        w.WriteStartObject();
        w.WriteString("version", VersionInfo.GitSha);
        w.WriteNumber("uptime_seconds", (long)_uptime.Elapsed.TotalSeconds);
        w.WriteNumber("total_tmads_per_sec", Math.Round(totalTmads, 2));
        w.WriteNumber("total_expected_opens_per_sec", totalExpectedOpens);
        w.WriteNumber("total_tiles_per_sec", Math.Round(totalTiles, 2));

        w.WriteStartObject("shares");
        w.WriteNumber("accepted", totalAccepted);
        w.WriteNumber("rejected", totalRejected);
        w.WriteNumber("stale", 0);
        w.WriteEndObject();

        w.WriteString("connection", _isConnected() ? "connected" : "disconnected");

        w.WriteStartArray("gpus");
        for (int i = 0; i < snapshot.GpuCount; i++)
        {
            var tel = i < gpuTelemetry.Length ? gpuTelemetry[i] : default;
            w.WriteStartObject();
            w.WriteNumber("index", i);
            w.WriteNumber("pci_bus_id", tel.Index);
            w.WriteNumber("tmads_per_sec", Math.Round(snapshot.TmadsPerSec[i], 2));
            w.WriteNumber("expected_opens_per_sec", snapshot.ExpectedOpensPerSec[i]);
            w.WriteNumber("tiles_per_sec", Math.Round(snapshot.TilesPerSec[i], 2));
            w.WriteNumber("temp_c", tel.TempC);
            w.WriteNumber("fan_pct", tel.FanPct);
            w.WriteNumber("power_w", Math.Round(tel.PowerW, 0));
            w.WriteNumber("uptime_seconds", (long)_uptime.Elapsed.TotalSeconds);
            w.WriteEndObject();
        }
        w.WriteEndArray();

        w.WriteEndObject();
        w.Flush();

        var dir = Path.GetDirectoryName(_statsPath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        {
            try { Directory.CreateDirectory(dir); }
            catch { /* non-fatal: HiveOS dirs typically pre-exist */ }
        }

        var tmpPath = _statsPath + ".tmp";
        File.WriteAllBytes(tmpPath, ms.ToArray());
        File.Move(tmpPath, _statsPath, overwrite: true);
    }

    public void Dispose()
    {
        _cts.Cancel();
        _thread.Interrupt();
        try { _thread.Join(TimeSpan.FromSeconds(2)); } catch { /* shutdown */ }
        _cts.Dispose();
    }
}
