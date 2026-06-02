using System.Globalization;
using System.Net;
using System.Text;
using Microsoft.Extensions.Logging;

namespace Akoya.Miner.Observability;

internal static class Metrics
{
    private static long[] _iters             = Array.Empty<long>();
    private static long[] _triggers          = Array.Empty<long>();
    private static long[] _blocksAccepted    = Array.Empty<long>();
    private static long[] _blocksRejected    = Array.Empty<long>();
    private static long[] _itersPerSec       = Array.Empty<long>();
    private static long[] _tmadsPerSec       = Array.Empty<long>();
    private static long[] _hashesPerSec      = Array.Empty<long>();
    private static long[] _tilesPerSec       = Array.Empty<long>();
    private static long[] _expectedOpensPerSec = Array.Empty<long>();
    private static long[] _iterMs            = Array.Empty<long>();
    private static long[] _sigmaRotations    = Array.Empty<long>();
    private static long[] _sigmaRotationLatestMs = Array.Empty<long>();
    private static long[] _sigmaRotationMaxMs = Array.Empty<long>();
    private static long[] _sigmaRotationDrainMs = Array.Empty<long>();
    private static long[] _sigmaRotationInstallMs = Array.Empty<long>();
    private static long[] _sigmaRotationBMerkleMs = Array.Empty<long>();
    private static long[] _sigmaRotationLostIters = Array.Empty<long>();
    private static long[] _sigmaRotationBSeedChanged = Array.Empty<long>();

    private static long[] _heartbeatTicks    = Array.Empty<long>();

    private static long   _blockFinds;
    private static long   _poolConnected;
    private static long   _poolLatencyMsBits;

    private static int    _gpuCount;
    private static HttpListener? _listener;
    private static Thread? _serverThread;

    public static void Init(int gpuCount, long[] heartbeats)
    {
        _gpuCount         = gpuCount;
        _iters            = new long[gpuCount];
        _triggers         = new long[gpuCount];
        _blocksAccepted   = new long[gpuCount];
        _blocksRejected   = new long[gpuCount];
        _itersPerSec      = new long[gpuCount];
        _tmadsPerSec      = new long[gpuCount];
        _hashesPerSec     = new long[gpuCount];
        _tilesPerSec      = new long[gpuCount];
        _expectedOpensPerSec = new long[gpuCount];
        _iterMs           = new long[gpuCount];
        _sigmaRotations   = new long[gpuCount];
        _sigmaRotationLatestMs = new long[gpuCount];
        _sigmaRotationMaxMs = new long[gpuCount];
        _sigmaRotationDrainMs = new long[gpuCount];
        _sigmaRotationInstallMs = new long[gpuCount];
        _sigmaRotationBMerkleMs = new long[gpuCount];
        _sigmaRotationLostIters = new long[gpuCount];
        _sigmaRotationBSeedChanged = new long[gpuCount];
        _heartbeatTicks   = heartbeats;
    }

    public static void IncIters(int gpu, long n)
    {
        if ((uint)gpu < (uint)_iters.Length)        Interlocked.Add(ref _iters[gpu], n);
    }
    public static void IncTriggers(int gpu)
    {
        if ((uint)gpu < (uint)_triggers.Length)     Interlocked.Increment(ref _triggers[gpu]);
    }
    public static void IncShareAccepted(int gpu)
    {
        if ((uint)gpu < (uint)_blocksAccepted.Length) Interlocked.Increment(ref _blocksAccepted[gpu]);
    }
    public static void IncShareRejected(int gpu)
    {
        if ((uint)gpu < (uint)_blocksRejected.Length) Interlocked.Increment(ref _blocksRejected[gpu]);
    }
    public static void IncBlockFind()                   => Interlocked.Increment(ref _blockFinds);

    public static void SetThroughput(
        int gpu,
        double itersPerSec,
        double tmadsPerSec,
        double hashesPerSec,
        double iterMs,
        double tilesPerSec = 0.0,
        double expectedOpensPerSec = 0.0)
    {
        if ((uint)gpu >= (uint)_itersPerSec.Length) return;
        Interlocked.Exchange(ref _itersPerSec[gpu],          BitConverter.DoubleToInt64Bits(itersPerSec));
        Interlocked.Exchange(ref _tmadsPerSec[gpu],          BitConverter.DoubleToInt64Bits(tmadsPerSec));
        Interlocked.Exchange(ref _hashesPerSec[gpu],         BitConverter.DoubleToInt64Bits(hashesPerSec));
        Interlocked.Exchange(ref _tilesPerSec[gpu],          BitConverter.DoubleToInt64Bits(tilesPerSec));
        Interlocked.Exchange(ref _expectedOpensPerSec[gpu],  BitConverter.DoubleToInt64Bits(expectedOpensPerSec));
        Interlocked.Exchange(ref _iterMs[gpu],               BitConverter.DoubleToInt64Bits(iterMs));
    }

    public static void RecordSigmaRotation(
        int gpu,
        double totalMs,
        double drainMs,
        double installMs,
        double bMerkleMs,
        double lostIters,
        bool bSeedChanged)
    {
        if ((uint)gpu >= (uint)_sigmaRotations.Length) return;

        Interlocked.Increment(ref _sigmaRotations[gpu]);
        Interlocked.Exchange(ref _sigmaRotationLatestMs[gpu], BitConverter.DoubleToInt64Bits(totalMs));
        Interlocked.Exchange(ref _sigmaRotationDrainMs[gpu], BitConverter.DoubleToInt64Bits(drainMs));
        Interlocked.Exchange(ref _sigmaRotationInstallMs[gpu], BitConverter.DoubleToInt64Bits(installMs));
        Interlocked.Exchange(ref _sigmaRotationBMerkleMs[gpu], BitConverter.DoubleToInt64Bits(bMerkleMs));
        Interlocked.Exchange(ref _sigmaRotationLostIters[gpu], BitConverter.DoubleToInt64Bits(lostIters));
        Interlocked.Exchange(ref _sigmaRotationBSeedChanged[gpu], BitConverter.DoubleToInt64Bits(bSeedChanged ? 1.0 : 0.0));

        long nextBits = BitConverter.DoubleToInt64Bits(totalMs);
        while (true)
        {
            long curBits = Volatile.Read(ref _sigmaRotationMaxMs[gpu]);
            double cur = BitConverter.Int64BitsToDouble(curBits);
            if (double.IsFinite(cur) && cur >= totalMs) break;
            if (Interlocked.CompareExchange(ref _sigmaRotationMaxMs[gpu], nextBits, curBits) == curBits) break;
        }
    }

    public static void SetPoolConnected(bool connected)
        => Interlocked.Exchange(ref _poolConnected, connected ? 1L : 0L);

    public static void SetPoolLatencyMs(double ms)
        => Interlocked.Exchange(ref _poolLatencyMsBits, BitConverter.DoubleToInt64Bits(ms));

    public static double GetPoolLatencyMs()
    {
        var v = BitConverter.Int64BitsToDouble(Interlocked.Read(ref _poolLatencyMsBits));
        return double.IsFinite(v) ? v : 0.0;
    }

    public static bool IsPoolConnected => Interlocked.Read(ref _poolConnected) == 1L;

    public static bool TryStart(int port, ILogger log, CancellationToken ct)
    {
        try
        {
            _listener = new HttpListener();
            _listener.Prefixes.Add($"http://*:{port}/");
            _listener.Start();
        }
        catch (Exception e)
        {
            log.LogWarning("metrics: failed to bind 0.0.0.0:{Port} ({Err}) — Prometheus disabled", port, e.Message);
            _listener = null;
            return false;
        }

        _serverThread = new Thread(() => ServeLoop(log, ct)) { IsBackground = true, Name = "metrics-http" };
        _serverThread.Start();
        log.LogInformation("metrics: Prometheus exposer on :{Port}/metrics", port);
        return true;
    }

    public static void Stop()
    {
        try { _listener?.Stop(); _listener?.Close(); } catch { /* shutdown */ }
    }

    public readonly record struct Snapshot(
        int GpuCount,
        long[] Accepted,
        long[] Rejected,
        double[] TmadsPerSec,
        double[] HashesPerSec,
        double[] ItersPerSec,
        double[] TilesPerSec,
        double[] ExpectedOpensPerSec);

    public static Snapshot GetSnapshot()
    {
        int n = _gpuCount;
        var accepted    = new long[n];
        var rejected    = new long[n];
        var tmads       = new double[n];
        var hashes      = new double[n];
        var iters       = new double[n];
        var tiles       = new double[n];
        var expected    = new double[n];
        for (int g = 0; g < n; g++)
        {
            accepted[g] = Volatile.Read(ref _blocksAccepted[g]);
            rejected[g] = Volatile.Read(ref _blocksRejected[g]);
            tmads[g]    = BitConverter.Int64BitsToDouble(Volatile.Read(ref _tmadsPerSec[g]));
            hashes[g]   = BitConverter.Int64BitsToDouble(Volatile.Read(ref _hashesPerSec[g]));
            iters[g]    = BitConverter.Int64BitsToDouble(Volatile.Read(ref _itersPerSec[g]));
            tiles[g]    = BitConverter.Int64BitsToDouble(Volatile.Read(ref _tilesPerSec[g]));
            expected[g] = BitConverter.Int64BitsToDouble(Volatile.Read(ref _expectedOpensPerSec[g]));
            if (!double.IsFinite(tmads[g]))  tmads[g] = 0;
            if (!double.IsFinite(hashes[g])) hashes[g] = 0;
            if (!double.IsFinite(iters[g]))  iters[g] = 0;
            if (!double.IsFinite(tiles[g]))  tiles[g] = 0;
            if (!double.IsFinite(expected[g])) expected[g] = 0;
        }
        return new Snapshot(n, accepted, rejected, tmads, hashes, iters, tiles, expected);
    }

    private static void ServeLoop(ILogger log, CancellationToken ct)
    {
        var l = _listener!;
        while (!ct.IsCancellationRequested && l.IsListening)
        {
            HttpListenerContext ctx;
            try { ctx = l.GetContext(); }
            catch { break; }

            try
            {
                if (ctx.Request.Url?.AbsolutePath == "/metrics")
                {
                    var body = Encoding.UTF8.GetBytes(Render());
                    ctx.Response.ContentType = "text/plain; version=0.0.4";
                    ctx.Response.ContentLength64 = body.Length;
                    ctx.Response.OutputStream.Write(body, 0, body.Length);
                }
                else
                {
                    ctx.Response.StatusCode = 404;
                }
                ctx.Response.Close();
            }
            catch (Exception e) { log.LogDebug("metrics: serve err {Err}", e.Message); }
        }
    }

    private static string Render()
    {
        var sb = new StringBuilder(4096);
        var inv = CultureInfo.InvariantCulture;

        sb.Append("# HELP akoya_miner_info Build metadata.\n");
        sb.Append("# TYPE akoya_miner_info gauge\n");
        sb.Append("akoya_miner_info{git_sha=\"").Append(VersionInfo.GitSha).Append("\"} 1\n");

        Counter(sb, "akoya_miner_iters_total",            "Total host-signal poll iterations.",     _iters);
        Counter(sb, "akoya_miner_triggers_total",         "Total GPU triggers (tile met σ target).", _triggers);
        Counter(sb, "akoya_miner_sigma_rotations_total",  "Total observed sigma installs or retargets.", _sigmaRotations);

        sb.Append("# HELP akoya_miner_blocks_submitted_total Submitted shares by pool result (V2: shares; V1: blocks).\n");
        sb.Append("# TYPE akoya_miner_blocks_submitted_total counter\n");
        for (int g = 0; g < _gpuCount; g++)
        {
            sb.Append("akoya_miner_blocks_submitted_total{gpu=\"").Append(g).Append("\",result=\"accepted\"} ")
              .Append(Volatile.Read(ref _blocksAccepted[g]).ToString(inv)).Append('\n');
            sb.Append("akoya_miner_blocks_submitted_total{gpu=\"").Append(g).Append("\",result=\"rejected\"} ")
              .Append(Volatile.Read(ref _blocksRejected[g]).ToString(inv)).Append('\n');
        }

        Gauge(sb, "akoya_miner_iters_per_second",  "Per-worker iterations per second (gauge).", _itersPerSec);
        Gauge(sb, "akoya_miner_tmads_per_second",  "Per-worker TMADs/s (gauge).",                _tmadsPerSec);
        Gauge(sb, "akoya_miner_hashes_per_second", "Per-worker hashes/s (gauge, tiles*DAF).",    _hashesPerSec);
        Gauge(sb, "akoya_miner_expected_opens_per_second", "Per-worker expected opens/s at current adjusted target.", _expectedOpensPerSec);
        Gauge(sb, "akoya_miner_tiles_per_second",  "Per-worker CTA output tiles/s (diagnostic; target-normalized opens track TMADs/s).", _tilesPerSec);
        Gauge(sb, "akoya_miner_iter_ms",           "Per-worker mean iteration latency (ms).",    _iterMs);
        Gauge(sb, "akoya_miner_sigma_rotation_latest_ms", "Latest worker-observed sigma rotation wall time from job observation to first new batch queued.", _sigmaRotationLatestMs);
        Gauge(sb, "akoya_miner_sigma_rotation_max_ms", "Maximum worker-observed sigma rotation wall time in this process.", _sigmaRotationMaxMs);
        Gauge(sb, "akoya_miner_sigma_rotation_drain_ms", "Latest old-batch drain time before sigma install.", _sigmaRotationDrainMs);
        Gauge(sb, "akoya_miner_sigma_rotation_install_ms", "Latest sigma install time excluding old-batch drain and first queue.", _sigmaRotationInstallMs);
        Gauge(sb, "akoya_miner_sigma_rotation_b_merkle_ms", "Latest B Merkle handle build time during sigma install.", _sigmaRotationBMerkleMs);
        Gauge(sb, "akoya_miner_sigma_rotation_lost_iters", "Latest sigma rotation time expressed as mean iterations lost.", _sigmaRotationLostIters);
        Gauge(sb, "akoya_miner_sigma_rotation_bseed_changed", "1 if the latest sigma rotation changed BSeed, else 0.", _sigmaRotationBSeedChanged);

        sb.Append("# HELP akoya_miner_block_finds_total Shares that the pool flagged is_block_find=true.\n");
        sb.Append("# TYPE akoya_miner_block_finds_total counter\n");
        sb.Append("akoya_miner_block_finds_total ").Append(Volatile.Read(ref _blockFinds).ToString(inv)).Append('\n');

        if (_heartbeatTicks.Length > 0)
        {
            sb.Append("# HELP akoya_miner_heartbeat_age_seconds Wall seconds since worker last ticked.\n");
            sb.Append("# TYPE akoya_miner_heartbeat_age_seconds gauge\n");
            long nowTicks = DateTime.UtcNow.Ticks;
            for (int g = 0; g < _gpuCount; g++)
            {
                long hb = Interlocked.Read(ref _heartbeatTicks[g]);
                double ageSec = hb == 0 ? 0.0 : (nowTicks - hb) / (double)TimeSpan.TicksPerSecond;
                sb.Append("akoya_miner_heartbeat_age_seconds{gpu=\"").Append(g).Append("\"} ")
                  .Append(ageSec.ToString("F3", inv)).Append('\n');
            }
        }

        sb.Append("# HELP akoya_miner_pool_connected 1 if the gRPC MiningStream is currently open, 0 otherwise.\n");
        sb.Append("# TYPE akoya_miner_pool_connected gauge\n");
        sb.Append("akoya_miner_pool_connected ").Append(Interlocked.Read(ref _poolConnected).ToString(inv)).Append('\n');

        sb.Append("# HELP akoya_miner_pool_latency_ms Last Ping/Pong round-trip time in milliseconds.\n");
        sb.Append("# TYPE akoya_miner_pool_latency_ms gauge\n");
        double rtt = BitConverter.Int64BitsToDouble(Interlocked.Read(ref _poolLatencyMsBits));
        sb.Append("akoya_miner_pool_latency_ms ")
          .Append(double.IsFinite(rtt) ? rtt.ToString("F3", inv) : "0").Append('\n');

        return sb.ToString();
    }

    private static void Counter(StringBuilder sb, string name, string help, long[] arr)
    {
        sb.Append("# HELP ").Append(name).Append(' ').Append(help).Append('\n');
        sb.Append("# TYPE ").Append(name).Append(" counter\n");
        for (int g = 0; g < arr.Length; g++)
            sb.Append(name).Append("{gpu=\"").Append(g).Append("\"} ")
              .Append(Volatile.Read(ref arr[g]).ToString(CultureInfo.InvariantCulture)).Append('\n');
    }

    private static void Gauge(StringBuilder sb, string name, string help, long[] bitsArr)
    {
        sb.Append("# HELP ").Append(name).Append(' ').Append(help).Append('\n');
        sb.Append("# TYPE ").Append(name).Append(" gauge\n");
        for (int g = 0; g < bitsArr.Length; g++)
        {
            double v = BitConverter.Int64BitsToDouble(Volatile.Read(ref bitsArr[g]));
            sb.Append(name).Append("{gpu=\"").Append(g).Append("\"} ")
              .Append(double.IsFinite(v) ? v.ToString("G", CultureInfo.InvariantCulture) : "0").Append('\n');
        }
    }
}
