// MetricsSampler — 1 Hz CPU/GPU sampler for the hot-loop.
//
// Emits structured ILogger lines like:
//   metric { cpuPct: 12.3, rssMB: 1834, gpu0: 99%/67°C/350W, gpu1: 98%/72°C/340W }
//
// Implementation:
//   • CPU%: delta of Process.TotalProcessorTime over wall-clock interval,
//     normalised by Environment.ProcessorCount → 100% means a single core
//     fully pinned (i.e. raw process CPU%, not normalised).
//   • RSS MB: Process.WorkingSet64.
//   • GPU stats: shells out to `nvidia-smi --query-gpu=… --format=csv,noheader,nounits`
//     once per sample. ~5–20 ms per call; cheap at 1 Hz, AOT-safe.
//     Queries all visible GPUs (multi-row CSV output).
//
// Why nvidia-smi rather than NVML P/Invoke: avoids an extra native dep,
// works inside the akoya-miner Native-AOT binary without binding gymnastics.
// 1 Hz polling has negligible perf impact.

using System.Diagnostics;
using System.Globalization;
using Microsoft.Extensions.Logging;

namespace Akoya.MinerCore;

public record struct GpuStats(int Index, int UtilPct, int MemMB, int TempC, double PowerW, int FanPct);

public sealed class MetricsSampler : IDisposable
{
    private readonly ILogger _log;
    private readonly TimeSpan _period;
    private readonly int _logEveryNSamples;
    private readonly CancellationTokenSource _cts = new();
    private readonly Task _loop;
    private readonly Process _self = Process.GetCurrentProcess();

    private volatile GpuStats[] _latestGpuStats = Array.Empty<GpuStats>();

    /// <summary>Latest per-GPU telemetry snapshot (updated at sample rate).</summary>
    public GpuStats[] LatestGpuStats => _latestGpuStats;

    /// <param name="period">Sampling cadence (default 1 s). Snapshots feed
    /// HiveOS writer + Prometheus at this rate regardless of log emission.</param>
    /// <param name="logEveryNSamples">Emit one INFO "metric …" log line every
    /// N samples. Default 5 → ~one log line every 5 s at 1 Hz sampling.
    /// Tests pass 1 for per-sample emission.</param>
    public MetricsSampler(ILogger log, TimeSpan? period = null, int logEveryNSamples = 5)
    {
        _log = log;
        _period = period ?? TimeSpan.FromSeconds(1);
        _logEveryNSamples = Math.Max(1, logEveryNSamples);
        _loop = Task.Run(LoopAsync);
    }

    private async Task LoopAsync()
    {
        var swWall = Stopwatch.StartNew();
        var prevCpu = _self.TotalProcessorTime;
        var prevWall = TimeSpan.Zero;
        var ct = _cts.Token;
        // Log every Nth sample for liveness watchers; sample at full rate so
        // the HiveOS writer and Prometheus snapshot stay fresh.
        int sampleNum = 0;

        while (!ct.IsCancellationRequested)
        {
            try { await Task.Delay(_period, ct).ConfigureAwait(false); }
            catch (TaskCanceledException) { break; }

            try
            {
                _self.Refresh();
                var nowCpu  = _self.TotalProcessorTime;
                var nowWall = swWall.Elapsed;
                var dCpu    = (nowCpu  - prevCpu ).TotalMilliseconds;
                var dWall   = (nowWall - prevWall).TotalMilliseconds;
                prevCpu = nowCpu; prevWall = nowWall;

                double cpuPct = dWall > 0 ? 100.0 * dCpu / dWall : 0;
                long rssMB    = _self.WorkingSet64 / (1024 * 1024);

                var gpus = ReadAllGpuStats();
                _latestGpuStats = gpus;

                bool shouldLog = (++sampleNum % _logEveryNSamples) == 0;
                if (!shouldLog)
                    continue;

                if (gpus.Length == 1)
                {
                    var g = gpus[0];
                    _log.LogInformation(
                        "metric cpuPct={CpuPct:F1} rssMB={RssMB} gpuUtilPct={GpuUtilPct} gpuMemMB={GpuMemMB} gpuTempC={GpuTempC} gpuPowerW={GpuPowerW:F0}",
                        cpuPct, rssMB, g.UtilPct, g.MemMB, g.TempC, g.PowerW);
                }
                else if (gpus.Length > 1)
                {
                    var summary = string.Join(" ", gpus.Select(g =>
                        $"gpu{g.Index}={g.UtilPct}%/{g.TempC}°C/{g.PowerW:F0}W"));
                    _log.LogInformation(
                        "metric cpuPct={CpuPct:F1} rssMB={RssMB} {GpuSummary}",
                        cpuPct, rssMB, summary);
                }
                else
                {
                    _log.LogInformation(
                        "metric cpuPct={CpuPct:F1} rssMB={RssMB} gpu=unavailable",
                        cpuPct, rssMB);
                }
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "metric sample failed");
            }
        }
    }

    internal static GpuStats[] ReadAllGpuStats()
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "nvidia-smi",
                Arguments = "--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw,fan.speed --format=csv,noheader,nounits",
                RedirectStandardOutput = true,
                RedirectStandardError  = true,
                UseShellExecute = false,
                CreateNoWindow  = true,
            };
            using var p = Process.Start(psi);
            if (p == null) return Array.Empty<GpuStats>();
            string output = p.StandardOutput.ReadToEnd().Trim();
            p.WaitForExit(500);
            return ParseGpuStatsCsv(output);
        }
        catch
        {
            return Array.Empty<GpuStats>();
        }
    }

    internal static GpuStats[] ParseGpuStatsCsv(string output)
    {
        if (string.IsNullOrWhiteSpace(output)) return Array.Empty<GpuStats>();

        var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var result = new GpuStats[lines.Length];
        for (int i = 0; i < lines.Length; i++)
        {
            var parts = lines[i].Split(',', StringSplitOptions.TrimEntries);
            int idx   = parts.Length > 0 ? ParseInt(parts[0]) : i;
            int util  = parts.Length > 1 ? ParseInt(parts[1]) : 0;
            int mem   = parts.Length > 2 ? ParseInt(parts[2]) : 0;
            int temp  = parts.Length > 3 ? ParseInt(parts[3]) : 0;
            double pw = parts.Length > 4 ? ParseDouble(parts[4]) : 0;
            int fan   = parts.Length > 5 ? ParseInt(parts[5]) : 0;
            result[i] = new GpuStats(idx, util, mem, temp, pw, fan);
        }
        return result;
    }

    private static int ParseInt(string s) =>
        int.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out var v) ? v : 0;

    private static double ParseDouble(string s) =>
        double.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out var v) ? v : 0;

    public void Dispose()
    {
        _cts.Cancel();
        try { _loop.Wait(TimeSpan.FromSeconds(2)); } catch { /* shutdown */ }
        _cts.Dispose();
        _self.Dispose();
    }
}
