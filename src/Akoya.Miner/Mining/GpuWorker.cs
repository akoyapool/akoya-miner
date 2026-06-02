// GpuWorker — owns one CUDA device and runs the V2 mining loop against the
// current σ from JobBus. Emits ShareSubmission protos to ShareSink.
// One process → one MiningSession → N GpuWorker (one per GPU). 

using System.Diagnostics;
using System.Numerics;
using Akoya.Crypto;
using Akoya.Cuda;
using Akoya.Miner.Config;
using Akoya.Miner.Observability;
using Akoya.MinerCore;
using Akoya.Mining;
using Microsoft.Extensions.Logging;
using PearlPool.Proto.V2;

namespace Akoya.Miner.Mining;

internal interface IShareSink
{
    /// <summary>Hand a freshly-built ShareSubmission off to the session.
    /// Implementations enqueue onto MiningSession's outbound channel.</summary>
    ValueTask SubmitAsync(ShareSubmission share, CancellationToken ct);
}

internal sealed class GpuWorker : IAsyncDisposable, ILivenessTarget
{
    // Tensor-hash kernel parameters — match v1 MineBlocks.cs / pearl-pure-miner.
    private const uint TENSOR_HASH_THREADS = 128;
    private const uint TENSOR_HASH_STAGES  = 2;
    private const uint TENSOR_HASH_LEAVES  = 512;
    private static readonly BigInteger TargetSpace = BigInteger.One << 256;
    private static readonly double TargetSpaceAsDouble = Math.Pow(2.0, 256);

    private readonly record struct SigmaRotationTiming(
        bool BSeedChanged,
        double OldBatchDrainMs,
        double BSeedExpandMs,
        double InstallGpuMs,
        double SyncPingMs,
        double SyncPongMs,
        double GraphPrepareMs,
        double BMerkleMs,
        double FirstQueueMs,
        double JobSeenToFirstNewBatchQueuedMs)
    {
        public double InstallMs =>
            BSeedExpandMs + InstallGpuMs + SyncPingMs + SyncPongMs + GraphPrepareMs;
    }

    private sealed class SigmaRotationTimingBuilder
    {
        public bool BSeedChanged;
        public double OldBatchDrainMs;
        public double BSeedExpandMs;
        public double InstallGpuMs;
        public double SyncPingMs;
        public double SyncPongMs;
        public double GraphPrepareMs;
        public double BMerkleMs;

        public SigmaRotationTiming Build(double firstQueueMs, double jobSeenToFirstNewBatchQueuedMs)
            => new(
                BSeedChanged,
                OldBatchDrainMs,
                BSeedExpandMs,
                InstallGpuMs,
                SyncPingMs,
                SyncPongMs,
                GraphPrepareMs,
                BMerkleMs,
                firstQueueMs,
                jobSeenToFirstNewBatchQueuedMs);
    }

    // Noisy-GEMM CTA tile parameters — match v1 MineBlocks.cs.
    // BM/BN are the GEMM kernel CTA output tile; the kernel requires
    // M%BM==0 and N%BN==0. They are NOT the same as the protocol's
    // proof-pattern tile (rows×cols) used for hash-tile accounting —
    // see TilesPerIter / ValidateCtaTileDivisibility below.
    private const int BM = 128, BN = 256, BK = 128, CM = 1, CN = 1;

    // Used by the throwaway lcg_int7_fill that primes dA before the first
    // tensor_hash_A in a σ install (so the commit computation has SOMETHING
    // to hash — the actual mining iters use real seeds derived from the
    // monotonic counter).
    private const ulong THROWAWAY_A_SEED_LO = 0xFFFF_FFFF_FFFF_FFFFul;

    // Periodic stats log cadence.
    private const double StatsIntervalSeconds = 5.0;
#if PERF_TIMINGS
    private const bool RuntimePerfTimings = true;
#else
    private const bool RuntimePerfTimings = false;
#endif

    private readonly int _gpuIndex;
    private readonly int _deviceOrdinal;
    private readonly string _gpuUuid;
    private readonly JobBus _bus;
    private readonly IShareSink _sink;
    private readonly ShareFinalizer _finalizer;
    private readonly ReadOnlyMemory<byte> _minerId;
    private readonly MineOptions _mine;
    private readonly ILogger _log;
    private readonly TaskCompletionSource _exited = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly CancellationTokenSource _cts = new();
    private Thread? _thread;

    // Rate-limit per-trigger error logging. A systematic post-trigger fault
    // (e.g. a kernel that's producing torn shares because of GPU ECC
    // corruption) can fire HandleTrigger -> catch -> log on every batch.
    // At ~mpp=10 and ~5 batches/sec that's 50/s of identical-looking
    // errors — enough to swamp the log shipper. Cap to 1/30s; the worker
    // is still attempting recovery between, so we're not hiding anything,
    // just merging duplicates.
    private readonly Akoya.Miner.Observability.LogRateLimiter _triggerErrorRl =
        new(TimeSpan.FromSeconds(30));

    // Liveness tick: monotonic TickCount64 of the most-recent observable
    // progress on this worker (loop iter, σ-rotation observed, share built).
    // Read by WorkerLivenessWatchdog, written by the worker thread on every
    // pass. Volatile-paired so a torn read never happens on 32-bit.
    private long _lastProgressTicks;

    // Trigger / σ-install tick: monotonic TickCount64 of the most recent
    // "productive" event — first σ install, vardiff retarget, or a real
    // trigger. The watchdog evaluates this across all workers to detect
    // "the rig is iterating but never producing shares" — typically a vardiff
    // that locked us at an impossible aggregate target. Initial 0 means
    // "no σ ever installed"; the watchdog ignores 0.
    private long _lastTriggerOrSigmaTicks;

    // Mirror of state.SeenJobVersion exposed for diagnostics + tests. Bumped
    // every time the worker drains a new (ctx, version) from the JobBus —
    // i.e. immediately before InstallSigma runs for that version. Allows
    // operators (and integration tests) to confirm the worker is keeping up
    // with the bus without poking at internal WorkerState.
    private long _lastSeenJobVersion = -1;

    // Last observed iter latency, stored as double bits. Used to translate
    // sigma-rotation stalls into the equivalent number of lost iterations.
    private long _lastIterMsBits;

    // Most recent fatal exception that caused Loop() to exit, if any.
    // Surfaced for tests / diagnostics; production reads via logs.
    private Exception? _lastFatal;
    public Exception? LastFatal => Volatile.Read(ref _lastFatal);

    // Whether DisposeAsync observed a clean exit (true) or had to proceed
    // past the grace timeout (false). Surfaced so the orchestrator can log
    // critical when a worker had to be orphaned — orphaned-but-running
    // CUDA work could collide with the next reconnect's fresh worker.
    private int _exitedCleanly;
    public bool DisposeWasClean => Volatile.Read(ref _exitedCleanly) != 0;

    /// <summary>Maximum time <see cref="DisposeAsync"/> will wait for the
    /// worker thread to honour cancellation before proceeding anyway. A
    /// stuck <c>cuStreamSynchronize</c> blocks the kernel — we can't force
    /// the thread to exit, but we can't let it pin the reconnect either.
    /// 10s is comfortably above worst-case kernel duration at production
    /// M/N/K (sub-200ms per iter on a 3090; the watchdog would trip on
    /// real hangs before this anyway).</summary>
    public static readonly TimeSpan DisposeGrace = TimeSpan.FromSeconds(10);

    public Task ExitedTask => _exited.Task;
    public int GpuIndex => _gpuIndex;
    public string GpuUuid => _gpuUuid;

    /// <summary>Monotonic <see cref="Environment.TickCount64"/> of the
    /// worker's last observable progress event. The orchestrator's
    /// <see cref="WorkerLivenessWatchdog"/> compares this to "now" to
    /// detect GPU-side deadlocks (kernel hang, driver wedge, stuck await).</summary>
    public long LastProgressTicks => Volatile.Read(ref _lastProgressTicks);

    /// <summary>Monotonic <see cref="Environment.TickCount64"/> of the most
    /// recent σ install / vardiff / actual trigger. <c>0</c> means "never
    /// had σ" — the trigger-rate watchdog treats that as "nothing to check
    /// yet". See <see cref="ILivenessTarget.LastTriggerOrSigmaTicks"/>.</summary>
    public long LastTriggerOrSigmaTicks => Volatile.Read(ref _lastTriggerOrSigmaTicks);

    /// <summary>Highest <see cref="JobBus"/> version this worker has drained
    /// and acted on (i.e. picked up + completed InstallSigma against). Starts
    /// at -1 before the first job. Useful for diagnostics ("is the worker
    /// keeping up with bus publishes?") and for tests that need to sequence
    /// publishes deterministically without observing private state.</summary>
    public long LastSeenJobVersion => Volatile.Read(ref _lastSeenJobVersion);

    private void TouchProgress() => Volatile.Write(ref _lastProgressTicks, Environment.TickCount64);

    private void TouchTriggerOrSigma() =>
        Volatile.Write(ref _lastTriggerOrSigmaTicks, Environment.TickCount64);

    private void RecordSeenJobVersion(long version) =>
        Volatile.Write(ref _lastSeenJobVersion, version);

    // Arms the trigger-rate clock the first time we install σ — gives the
    // watchdog something to compare against. After that, only an actual
    // trigger or a vardiff retarget should reset it; ordinary σ rotations
    // do NOT (they don't change the underlying target difficulty, so the
    // Poisson trigger rate is unchanged — resetting would mask a stuck
    // pool target indefinitely).
    private void ArmTriggerClockIfFirst()
    {
        if (Volatile.Read(ref _lastTriggerOrSigmaTicks) == 0)
            Volatile.Write(ref _lastTriggerOrSigmaTicks, Environment.TickCount64);
    }

    public GpuWorker(
        int gpuIndex,
        int deviceOrdinal,
        string gpuUuid,
        JobBus bus,
        IShareSink sink,
        ReadOnlyMemory<byte> minerId,
        MineOptions mine,
        ILogger log,
        Action<GpuWorker, Exception>? onFatal = null)
    {
        _gpuIndex      = gpuIndex;
        _deviceOrdinal = deviceOrdinal;
        _gpuUuid       = gpuUuid;
        _bus           = bus;
        _sink          = sink;
        _minerId       = minerId;
        _mine          = mine;
        _log           = log;
        _onFatal       = onFatal;
        _finalizer     = new ShareFinalizer(sink, log);
    }

    private readonly Action<GpuWorker, Exception>? _onFatal;

    public void Start(CancellationToken external)
    {
        var linkedCt = CancellationTokenSource.CreateLinkedTokenSource(external, _cts.Token).Token;
        _thread = new Thread(() => Loop(linkedCt))
        {
            IsBackground = true,
            Name = $"gpu-worker-{_gpuIndex}",
        };
        _thread.Start();
    }

    // ─── State carried across iterations within one Loop() invocation ─────

    private sealed class WorkerState
    {
        public WorkerHalf Ping = new();
        public WorkerHalf Pong = new();
        public ResidentBStateBuffers BState = null!;
        public CUcontext CudaContext;
        public CUstream BMerkleCopyStream;
        public MiningConfiguration MiningConfig = null!;

        // Per-σ state. Re-derived on every σ rotation; CommitmentHasher.GetKey
        // is the source of truth for jobKey (must use big-endian minerId path).
        public byte[] InstalledSigma = [];       // 128 B; empty == no σ installed
        public byte[] InstalledJobKey = [];      // 32 B; matches InstalledSigma
        public byte[] InstalledHashB = [];       // 32 B; keyed-merkle root of bBytes under InstalledJobKey
        public byte[] InstalledBSeed = [];       // 32 B; pool-supplied opaque B seed for the current σ
        public uint   InstalledAuditK;           // audit_proof v1 K parameter (0 = disabled)
        // Pre-built B Merkle tree under InstalledJobKey. Cached at σ install so
        // per-trigger share build extracts inclusion proofs without re-hashing
        // 64 MiB. Refcounted: σ rotation Releases the worker's reference; any
        // in-flight ShareFinalizer payload holds its own Acquired reference.
        public IMerkleTreeHandle? BMerkleTree;
        public Task<IMerkleTreeHandle>? BMerkleTreeBuildTask;
        public uint   InstalledTargetNbits;      // pool difficulty nbits we built dPowTarget against
        public ulong  SigmaSeed;                 // BitConverter.ToUInt64(σ[0..8]) — drives lcg_int7

        // Stats window.
        public ulong GlobalIterIdx;
        public ulong ItersAtLastReport;
        public ulong TriggersTotal;
        public ulong SharesEmitted;
        public long StatsWindowStartedAt = Stopwatch.GetTimestamp();
        public long SigmaInstalledAtTimestamp = Stopwatch.GetTimestamp();
        public long SeenJobVersion = -1;         // JobBus.Version we last acted on
    }

    // One side of the double buffer: independent WorkerBuffers + CUDA stream + workspace.
    // The loop alternates: Ping executes on GPU while CPU sets up Pong, then swaps.
    private sealed class WorkerHalf
    {
        public int           DeviceId;
        public WorkerBuffers Buffers = null!;
        public CUstream      Stream;
        public nint          Workspace;
        public bool          BUploaded;   // dB has been H2D'd to this half
        public bool          GraphReady;  // workspace owns prepared iter-batch graph
        public nint[]?       HeaderPtrBatch; // reused scratch for IterBatch header ptr array
        public CUevent       BatchStartEvent;
        public CUevent       BatchStopEvent;
        public bool          BatchEventsCreated;
    }

    // ─── Main loop ─────────────────────────────────────────────────────────

    private void Loop(CancellationToken ct)
    {
        _log.LogInformation("worker[{Gpu}]: starting on device ordinal={Ord} uuid={Uuid}",
            _gpuIndex, _deviceOrdinal, _gpuUuid);

        WorkerState? state = null;
        // CUDA cleanup state — declared outside the try so the finally block
        // can release everything regardless of where we throw.
        //
        // We use the *primary* context (cuDevicePrimaryCtxRetain) rather than
        // a fresh cuCtxCreate. The primary context is reference-counted by
        // the driver, so:
        //   • Reconnect cycles can start fresh workers on the same thread/
        //     process without leaking contexts (each retain pairs with a
        //     release on shutdown).
        //   • Multi-GPU works naturally — one primary per device.
        //   • Tests that stand up + tear down workers repeatedly don't
        //     accumulate dangling contexts and hit ERROR_OUT_OF_MEMORY.
        // V1 used cuCtxCreate and leaked on every reconnect — do not regress.
        bool deviceRetained = false;
        CUdevice dev = default;
        CUstream stream0 = default, stream1 = default, stream2 = default;
        bool stream0Created = false, stream1Created = false, stream2Created = false;
        try
        {
            // 1. Bind to CUDA device + retain the driver's primary context.
            CudaDriver.Check(CudaDriver.Init(0), "cuInit");
            CudaDriver.Check(CudaDriver.DeviceGet(out dev, _deviceOrdinal), "cuDeviceGet");
            CudaDriver.Check(CudaDriver.DevicePrimaryCtxRetain(out var primary, dev),
                "cuDevicePrimaryCtxRetain");
            deviceRetained = true;
            CudaDriver.Check(CudaDriver.CtxSetCurrent(primary), "cuCtxSetCurrent");
            CudaDriver.Check(CudaDriver.StreamCreate(out stream0, 0), "cuStreamCreate [ping]");
            stream0Created = true;
            CudaDriver.Check(CudaDriver.StreamCreate(out stream1, 0), "cuStreamCreate [pong]");
            stream1Created = true;
            CudaDriver.Check(CudaDriver.StreamCreate(out stream2, 0), "cuStreamCreate [b-merkle]");
            stream2Created = true;
            TouchProgress();

            // 2. Wait for the first job — buffer sizes depend on σ-carried K/R.
            var (firstCtx, firstVer) = _bus.WaitForJobAsync(-1, ct).AsTask().GetAwaiter().GetResult();
            TouchProgress();
            ValidateCtaTileDivisibility(_mine.M, _mine.N);

            state = new WorkerState
            {
                MiningConfig   = MiningConfiguration.Default(firstCtx.CommonDim, firstCtx.Rank),
                SeenJobVersion = firstVer,
                CudaContext = primary,
                BMerkleCopyStream = stream2,
                // ──────────────────────────────────────────────────────────────
                //  Per-GPU search-space partitioning.
                //
                //  The kernel derives A from (winSeed, SigmaSeed) where
                //  winSeed == GlobalIterIdx at the moment the matmul is queued
                //  and SigmaSeed is σ[0..8] (a pool-controlled value all GPUs
                //  in the rig necessarily share). With both workers starting at
                //  GlobalIterIdx = 0 they walked the same iter sequence and
                //  derived the IDENTICAL A matrix at every iter — observed in
                //  the wild as two GPUs finding the same share (same tile,
                //  same σ, identical hash_a) every time, with the second
                //  submission silently deduped server-side. That's a wasted
                //  half of the rig's hashrate.
                //
                //  Pool-side verification reads the A matrix from the share's
                //  a_slice (committed via hash_a) — there is NO winSeed field
                //  on the wire (see proto/v2/miner.proto §ShareSubmission). So
                //  the miner is free to start at any GlobalIterIdx it likes,
                //  as long as no two GPUs in the rig collide.
                //
                //  2^48 ≈ 2.8e14 iters per slot. An H100 at ~1000 of these
                //  iter steps per second would take ~9000 years to exhaust
                //  one slot. Safe ceiling, and we still have room for
                //  2^16 = 65536 GPUs in a single rig before wrapping —
                //  vastly above any realistic per-process ordinal limit.
                GlobalIterIdx = (ulong)_gpuIndex << 48,
            };
            state.Ping.Stream = stream0;
            state.Pong.Stream = stream1;
            state.Ping.DeviceId = _deviceOrdinal;
            state.Pong.DeviceId = _deviceOrdinal;
            state.Ping.Buffers = new WorkerBuffers(
                _mine.M, _mine.N, (int)firstCtx.CommonDim, firstCtx.Rank,
                _mine.MatmulsPerPoll);
            state.Pong.Buffers = new WorkerBuffers(
                _mine.M, _mine.N, (int)firstCtx.CommonDim, firstCtx.Rank,
                _mine.MatmulsPerPoll);
            state.BState = new ResidentBStateBuffers(
                _mine.N, (int)firstCtx.CommonDim, firstCtx.Rank, stream0);

            _log.LogInformation(
                "worker[{Gpu}]: buffers allocated M={M} N={N} K={K} R={R} mpp={Mpp}; bBytes={BLen} (seed=ctx.BSeed[0..8]={BSeedHex}); iter-base=0x{IterBase:X16}",
                _gpuIndex, state.Ping.Buffers.M, state.Ping.Buffers.N, state.Ping.Buffers.K, state.Ping.Buffers.R,
                state.Ping.Buffers.MatmulsPerPoll, (long)_mine.N * (long)firstCtx.CommonDim,
                Convert.ToHexString(firstCtx.BSeed.AsSpan(0, Math.Min(8, firstCtx.BSeed.Length))),
                state.GlobalIterIdx);

            // 4. Install the first σ; from this point on every batch can run.
            //    Record the drained version BEFORE InstallSigma so observers
            //    (watchdog, tests, logs) can distinguish "worker hasn't seen
            //    the publish yet" from "worker saw it but install threw".
            RecordSeenJobVersion(state.SeenJobVersion);
            long firstRotationStart = Stopwatch.GetTimestamp();
            var firstTimingBuilder = InstallSigma(state, firstCtx, oldBatchDrainMs: 0.0);

            // 5. Prime: queue the first batch on Ping before entering the loop.
            //    Loop invariant: Ping ALWAYS has a queued batch in-flight at the
            //    top of the while loop. Pong is always idle. The only exception
            //    is after a σ-rotation forced drain (where we re-queue Ping
            //    immediately before the continue).
            int inflight = ComputeBatchSize(state);
            long firstQueueStart = Stopwatch.GetTimestamp();
            QueueBatch(state.Ping, inflight, state.GlobalIterIdx);
            double firstQueueMs = ElapsedMsSince(firstQueueStart);
            double firstRotationMs = ElapsedMsSince(firstRotationStart);
            StartBMerkleForInstalledSigma(state, firstCtx, firstTimingBuilder);
            RecordSigmaRotation(firstCtx, firstTimingBuilder.Build(
                firstQueueMs,
                firstRotationMs));

            // 6. Steady-state double-buffered mining loop.
            while (!ct.IsCancellationRequested)
            {
                TouchProgress();

                // (A) New σ/vardiff: sync + scan Ping, handle trigger,
                //     install σ on both halves, re-queue Ping.
                var currentVer = _bus.Version;
                if (currentVer > state.SeenJobVersion)
                {
                    long rotationStart = Stopwatch.GetTimestamp();
                    var ctx = _bus.Current!;
                    state.SeenJobVersion = currentVer;
                    RecordSeenJobVersion(currentVer);

                    long drainStart = Stopwatch.GetTimestamp();
                    int winKb    = SyncAndScan(state.Ping, inflight);
                    double drainMs = ElapsedMsSince(drainStart);
                    ulong startB = state.GlobalIterIdx;
                    state.GlobalIterIdx  += (ulong)inflight;
                    MaybeLogStats(state);
                    TouchProgress();

                    if (winKb >= 0)
                    {
                        state.TriggersTotal++;
                        Metrics.IncTriggers(_gpuIndex);
                        TouchTriggerOrSigma();
                        ulong winSeedB = startB + (ulong)winKb;
                        try { HandleTrigger(state.Ping, state, winKb, winSeedB, ct); }
                        catch (Exception ex)
                        {
                            if (_triggerErrorRl.TryLog(out var suppressed))
                            {
                                _log.LogError(ex,
                                    "worker[{Gpu}]: trigger handling failed (winSeed={Seed}, suppressed {N} similar in last 30s) — share dropped",
                                    _gpuIndex, winSeedB, suppressed);
                            }
                        }
                        TouchProgress();
                    }

                    state.MiningConfig = MiningConfiguration.Default(ctx.CommonDim, ctx.Rank);
                    // InstallSigma drains both streams.
                    var timingBuilder = InstallSigma(state, ctx, drainMs);
                    inflight = ComputeBatchSize(state);
                    long queueStart = Stopwatch.GetTimestamp();
                    QueueBatch(state.Ping, inflight, state.GlobalIterIdx);
                    double queueMs = ElapsedMsSince(queueStart);
                    double rotationMs = ElapsedMsSince(rotationStart);
                    StartBMerkleForInstalledSigma(state, ctx, timingBuilder);
                    RecordSigmaRotation(ctx, timingBuilder.Build(
                        queueMs,
                        rotationMs));
                    continue;
                }

                // (B) Queue Pong batch while Ping is in-flight on the GPU.
                //     CPU overlap: GPU runs Ping while CPU sets up Pong's headers
                //     and enqueues kernels on the independent Pong stream.
                //
                //     Emergency switch: if AKOYA_DISABLE_PONG=1, force pongSize=0
                //     to fall back to V1-equivalent single-stream behaviour. Use
                //     this if bursty `claimedHash > liveTarget` pre-submit skips
                //     appear clustered in a single σ window (a symptom that
                //     points to a concurrent-stream state hazard). Costs
                //     ~30–40% throughput; preserves correctness.
                int pongSize   = _mine.DisablePong ? 0 : ComputeBatchSize(state);
                ulong pongStart = state.GlobalIterIdx + (ulong)inflight;
                if (pongSize > 0)
                    QueueBatch(state.Pong, pongSize, pongStart);

                // (C) Sync + scan Ping, update counters, handle trigger.
                {
                    ulong pingBatchStart = state.GlobalIterIdx;
                    int winK = SyncAndScan(state.Ping, inflight);
                    TouchProgress();

                    state.GlobalIterIdx  += (ulong)inflight;
                    MaybeLogStats(state);

                    if (winK >= 0)
                    {
                        state.TriggersTotal++;
                        Metrics.IncTriggers(_gpuIndex);
                        TouchTriggerOrSigma();

                        ulong winSeed = pingBatchStart + (ulong)winK;
                        try
                        {
                            HandleTrigger(state.Ping, state, winK, winSeed, ct);
                        }
                        catch (Exception ex)
                        {
                            // Don't kill the worker on a single bad share — log and continue.
                            // Rate-limited: a systematic post-trigger fault can fire
                            // this on every batch (see _triggerErrorRl comment).
                            if (_triggerErrorRl.TryLog(out var suppressed))
                            {
                                _log.LogError(ex,
                                    "worker[{Gpu}]: trigger handling failed (winSeed={Seed}, suppressed {N} similar in last 30s) — share dropped",
                                    _gpuIndex, winSeed, suppressed);
                            }
                        }
                        TouchProgress();
                    }
                }

                if (pongSize == 0)
                {
                    QueueBatch(state.Ping, inflight, state.GlobalIterIdx);
                    continue;
                }

                // (D) Swap Ping/Pong. The former Pong batch is now "in-flight"
                //     on the new Ping stream; the former Ping is now idle Pong.
                (state.Ping, state.Pong) = (state.Pong, state.Ping);
                inflight = pongSize;
            }

            // Teardown: drain any remaining in-flight work on both streams
            // so device memory is quiesced before FreeHalf releases it.
            CudaDriver.Check(CudaDriver.StreamSynchronize(state.Ping.Stream), "sync shutdown [ping]");
            CudaDriver.Check(CudaDriver.StreamSynchronize(state.Pong.Stream), "sync shutdown [pong]");
        }
        catch (OperationCanceledException) { /* shutdown */ }
        catch (Exception ex)
        {
            // Fatal in the mining loop. We do NOT silently swallow — instead:
            //   1. Log the failure with full context.
            //   2. Capture it on _lastFatal for diagnostics / tests.
            //   3. Fire _onFatal so the orchestrator can trip its
            //      workerTripCts and recycle the pool connection (which
            //      also restarts every worker, including this one, against
            //      a freshly-retained CUDA primary context).
            // The worker thread itself still unwinds cleanly (finally block
            // releases device resources); but the *process* learns about
            // the failure and recovers via the reconnect path instead of
            // sitting wedged with a dead GPU.
            _log.LogError(ex, "worker[{Gpu}]: fatal — exiting, tripping reconnect", _gpuIndex);
            Volatile.Write(ref _lastFatal, ex);
            try { _onFatal?.Invoke(this, ex); }
            catch (Exception cbEx)
            {
                _log.LogError(cbEx, "worker[{Gpu}]: onFatal callback itself threw", _gpuIndex);
            }
        }
        finally
        {
            // Cleanup order matters: drain streams + free workspace + buffers
            // MUST happen while the CUDA context is still current. Only after
            // those are gone do we release the primary context (which
            // decrements the driver's ref-count; if we're the last holder,
            // the context is destroyed). Getting this order wrong leaks
            // device memory across worker restarts → eventual OOM on long
            // reconnect cycles.
            if (state is not null)
            {
                // Drain any in-flight work before freeing resources (handles
                // the OperationCanceledException path where the loop may have
                // exited mid-flight or before the teardown syncs above ran).
                if (stream0Created)
                    try { CudaDriver.StreamSynchronize(state.Ping.Stream); } catch { /* best-effort */ }
                if (stream1Created)
                    try { CudaDriver.StreamSynchronize(state.Pong.Stream); } catch { /* best-effort */ }
                if (stream2Created)
                    try { CudaDriver.StreamSynchronize(state.BMerkleCopyStream); } catch { /* best-effort */ }
                FreeHalf(state.Ping);
                FreeHalf(state.Pong);
                try { state.BState?.Dispose(); } catch { /* best-effort */ }
                RetireBMerkleState(state, waitForPendingBuild: true);
            }

            if (stream0Created)
            {
                try { CudaDriver.StreamDestroy(stream0); } catch { /* best-effort */ }
            }
            if (stream1Created)
            {
                try { CudaDriver.StreamDestroy(stream1); } catch { /* best-effort */ }
            }
            if (stream2Created)
            {
                try { CudaDriver.StreamDestroy(stream2); } catch { /* best-effort */ }
            }

            if (deviceRetained)
            {
                try { CudaDriver.DevicePrimaryCtxRelease(dev); } catch { /* best-effort */ }
            }

            _exited.TrySetResult();
            // Drain the share finalizer last — any in-flight share build
            // must complete (or be dropped) before the worker logs "stopped".
            try { _finalizer.DisposeAsync().AsTask().GetAwaiter().GetResult(); }
            catch (Exception ex) { _log.LogWarning(ex, "worker[{Gpu}]: ShareFinalizer dispose threw", _gpuIndex); }
            _log.LogInformation("worker[{Gpu}]: stopped", _gpuIndex);
        }
    }

    // ─── σ install (heavy path) ────────────────────────────────────────────

    private readonly record struct QueueBatchTiming(
        double TotalMs,
        double HeaderClearMs,
        double HeaderPtrMs,
        double DeviceClearEnqueueMs,
        double IterEnqueueMs,
        double GraphLaunchMs,
        bool UsedGraph);

    private readonly record struct SyncScanTiming(
        int Winner,
        double SyncMs,
        double ScanMs);

    // Queue one batch: header clear + IterBatch enqueue. No sync.
    private static void QueueBatch(WorkerHalf half, int batch, ulong batchStart)
    {
        var b = half.Buffers;
        // INVARIANT 1: every pHeader must be host-cleared BEFORE launching
        // anything that will write to it. A stale status==1 from the previous
        // batch left in a slot we don't overwrite this round = phantom trigger.
        for (int k = 0; k < batch; k++)
            unsafe { new Span<byte>((void*)b.HostHeaders[k], b.HeaderSize).Clear(); }

        // The graph path captures the dSync clear as the first graph node.
        // Legacy IterBatch still needs the clear enqueued explicitly here.
        var headers = EnsureHeaderPtrBatch(half, batch);
        if (half.GraphReady && batch == b.MatmulsPerPoll)
        {
            int rc = PearlGemm.PearlGemmNative.IterBatchGraphLaunch(
                half.Workspace, batchStart, half.Stream.Handle);
            if (rc == 0) return;

            // Launch failure should be rare and usually indicates the graph
            // cannot replay on this driver/path. Fall back without dropping
            // the batch; the legacy path will clear dSync below.
            half.GraphReady = false;
        }

        // INVARIANT 2: dSync is the device-side coordination block shared by
        // every iter in the batch — zero it before any noisy_gemm executes.
        // The async zero is enqueued on the SAME stream as all subsequent
        // kernels, so CUDA's stream FIFO ordering guarantees the zero lands
        // before the first noisy_gemm reads dSync. Never pass CUstream.Null here.
        CudaDriver.Check(CudaDriver.MemsetD8Async(b.Sync, 0, (nuint)b.SyncSize, half.Stream), "memset dSync");

        // One CAPI call per batch (was 1 call per iter). All constants were
        // installed into the workspace at σ-install; only seedLo range and
        // the header slot pointers vary.
        unsafe
        {
            fixed (nint* pHeaders = headers)
                Check("iter_batch", PearlGemm.PearlGemmNative.IterBatch(
                    half.Workspace, batchStart, pHeaders, batch, half.Stream.Handle));
        }
    }

    private static QueueBatchTiming QueueBatchTimed(WorkerHalf half, int batch, ulong batchStart)
    {
        long totalStart = Stopwatch.GetTimestamp();
        var b = half.Buffers;
        // INVARIANT 1: every pHeader must be host-cleared BEFORE launching
        // anything that will write to it. A stale status==1 from the previous
        // batch left in a slot we don't overwrite this round = phantom trigger.
        long stageStart = Stopwatch.GetTimestamp();
        for (int k = 0; k < batch; k++)
            unsafe { new Span<byte>((void*)b.HostHeaders[k], b.HeaderSize).Clear(); }
        double headerClearMs = ElapsedMsSince(stageStart);

        // The graph path captures the dSync clear as the first graph node.
        // Legacy IterBatch still needs the clear enqueued explicitly here.
        stageStart = Stopwatch.GetTimestamp();
        var headers = EnsureHeaderPtrBatch(half, batch);
        double headerPtrMs = ElapsedMsSince(stageStart);
        if (half.GraphReady && batch == b.MatmulsPerPoll)
        {
            stageStart = Stopwatch.GetTimestamp();
            int rc = PearlGemm.PearlGemmNative.IterBatchGraphLaunch(
                half.Workspace, batchStart, half.Stream.Handle);
            double graphLaunchMs = ElapsedMsSince(stageStart);
            if (rc == 0)
            {
                return new QueueBatchTiming(
                    ElapsedMsSince(totalStart),
                    headerClearMs,
                    headerPtrMs,
                    DeviceClearEnqueueMs: 0.0,
                    IterEnqueueMs: 0.0,
                    GraphLaunchMs: graphLaunchMs,
                    UsedGraph: true);
            }

            // Launch failure should be rare and usually indicates the graph
            // cannot replay on this driver/path. Fall back without dropping
            // the batch; the legacy path will clear dSync below.
            half.GraphReady = false;
        }

        // INVARIANT 2: dSync is the device-side coordination block shared by
        // every iter in the batch — zero it before any noisy_gemm executes.
        // The async zero is enqueued on the SAME stream as all subsequent
        // kernels, so CUDA's stream FIFO ordering guarantees the zero lands
        // before the first noisy_gemm reads dSync. Never pass CUstream.Null here.
        stageStart = Stopwatch.GetTimestamp();
        CudaDriver.Check(CudaDriver.MemsetD8Async(b.Sync, 0, (nuint)b.SyncSize, half.Stream), "memset dSync");
        double deviceClearEnqueueMs = ElapsedMsSince(stageStart);

        // One CAPI call per batch (was 1 call per iter). All constants were
        // installed into the workspace at σ-install; only seedLo range and
        // the header slot pointers vary.
        stageStart = Stopwatch.GetTimestamp();
        unsafe
        {
            fixed (nint* pHeaders = headers)
                Check("iter_batch", PearlGemm.PearlGemmNative.IterBatch(
                    half.Workspace, batchStart, pHeaders, batch, half.Stream.Handle));
        }
        double iterEnqueueMs = ElapsedMsSince(stageStart);

        return new QueueBatchTiming(
            ElapsedMsSince(totalStart),
            headerClearMs,
            headerPtrMs,
            deviceClearEnqueueMs,
            iterEnqueueMs,
            GraphLaunchMs: 0.0,
            UsedGraph: false);
    }

    private static nint[] EnsureHeaderPtrBatch(WorkerHalf half, int batch)
    {
        if (half.HeaderPtrBatch is null || half.HeaderPtrBatch.Length < batch)
            half.HeaderPtrBatch = new nint[batch];
        var headers = half.HeaderPtrBatch;
        for (int k = 0; k < batch; k++)
            headers[k] = half.Buffers.HostHeaders[k];
        return headers;
    }

    private static void PrepareGraphIfEnabled(
        WorkerHalf half,
        bool enabled,
        bool required,
        ILogger log,
        string label)
    {
        half.GraphReady = false;
        if (!enabled) return;
        if (half.Workspace == IntPtr.Zero) return;

        int count = half.Buffers.MatmulsPerPoll;
        var headers = EnsureHeaderPtrBatch(half, count);
        int rc;
        unsafe
        {
            fixed (nint* pHeaders = headers)
                rc = PearlGemm.PearlGemmNative.IterBatchGraphPrepare(
                    half.Workspace, pHeaders, count, half.Stream.Handle);
        }

        if (rc == 0)
        {
            half.GraphReady = true;
            log.LogInformation("CUDA graph iter path prepared for {Half} (mpp={Mpp})", label, count);
            return;
        }

        if (required)
            throw new InvalidOperationException($"CUDA graph iter prepare failed for {label}: rc={rc}");

        log.LogWarning(
            "CUDA graph iter prepare failed for {Half}: rc={Rc}; falling back to IterBatch",
            label, rc);
    }

    // Sync stream, scan headers. Returns winning slot index or -1.
    // INVARIANT 3: SINGLE host↔device memory barrier for the whole batch.
    // Read a pHeader before this and you'll race the GPU writeback.
    private static int SyncAndScan(WorkerHalf half, int batch)
    {
        CudaDriver.Check(CudaDriver.StreamSynchronize(half.Stream), "sync batch");
        return ScanHeaders(half, batch);
    }

    private static SyncScanTiming SyncAndScanTimed(WorkerHalf half, int batch)
    {
        long start = Stopwatch.GetTimestamp();
        CudaDriver.Check(CudaDriver.StreamSynchronize(half.Stream), "sync batch");
        double syncMs = ElapsedMsSince(start);

        start = Stopwatch.GetTimestamp();
        int winner = ScanHeaders(half, batch);

        return new SyncScanTiming(winner, syncMs, ElapsedMsSince(start));
    }

    private static int ScanHeaders(WorkerHalf half, int batch)
    {
        for (int k = 0; k < batch; k++)
        {
            int status;
            unsafe { status = *(int*)half.Buffers.HostHeaders[k]; }
            if (status == 1) return k;
        }
        return -1;
    }

    private static void EnsureBatchEvents(WorkerHalf half)
    {
        if (half.BatchEventsCreated) return;
        CudaDriver.Check(CudaDriver.EventCreate(out half.BatchStartEvent, flags: 0), "cuEventCreate batch start");
        CudaDriver.Check(CudaDriver.EventCreate(out half.BatchStopEvent, flags: 0), "cuEventCreate batch stop");
        half.BatchEventsCreated = true;
    }

    private static QueueBatchTiming QueueBatchTimedWithEvents(WorkerHalf half, int batch, ulong batchStart)
    {
        EnsureBatchEvents(half);
        CudaDriver.Check(CudaDriver.EventRecord(half.BatchStartEvent, half.Stream), "cuEventRecord batch start");
        var timing = QueueBatchTimed(half, batch, batchStart);
        CudaDriver.Check(CudaDriver.EventRecord(half.BatchStopEvent, half.Stream), "cuEventRecord batch stop");
        return timing;
    }

    private static double ReadCompletedBatchGpuMs(WorkerHalf half)
    {
        if (!half.BatchEventsCreated) return 0.0;
        CudaDriver.Check(CudaDriver.EventElapsedTime(out float milliseconds, half.BatchStartEvent, half.BatchStopEvent),
            "cuEventElapsedTime batch");
        return milliseconds;
    }

    private static int ComputeBatchSize(WorkerState s) => s.Ping.Buffers.MatmulsPerPoll;

    private void RecordSigmaRotation(SigmaContext ctx, SigmaRotationTiming timing)
    {
        double iterMs = BitConverter.Int64BitsToDouble(Volatile.Read(ref _lastIterMsBits));
        double lostIters = double.IsFinite(iterMs) && iterMs > 0.0
            ? timing.JobSeenToFirstNewBatchQueuedMs / iterMs
            : 0.0;

        Metrics.RecordSigmaRotation(
            _gpuIndex,
            timing.JobSeenToFirstNewBatchQueuedMs,
            timing.OldBatchDrainMs,
            timing.InstallMs,
            timing.BMerkleMs,
            lostIters,
            timing.BSeedChanged);

        _log.LogInformation(
            "worker[{Gpu}]: σ rotation ready job={Job} sigma={SigmaPrefix} total={Total:F1}ms lost_iters={Lost:F2} drain={Drain:F1}ms install={Install:F1}ms bseed_changed={BSeedChanged} bseed_expand={BSeedExpand:F1}ms install_enqueue={InstallGpu:F1}ms sync_ping={SyncPing:F1}ms sync_pong={SyncPong:F1}ms graph={Graph:F1}ms b_merkle={BMerkle:F1}ms first_queue={FirstQueue:F3}ms",
            _gpuIndex,
            ctx.JobId,
            Convert.ToHexString(ctx.Sigma.AsSpan(0, Math.Min(8, ctx.Sigma.Length))),
            timing.JobSeenToFirstNewBatchQueuedMs,
            lostIters,
            timing.OldBatchDrainMs,
            timing.InstallMs,
            timing.BSeedChanged,
            timing.BSeedExpandMs,
            timing.InstallGpuMs,
            timing.SyncPingMs,
            timing.SyncPongMs,
            timing.GraphPrepareMs,
            timing.BMerkleMs,
            timing.FirstQueueMs);
    }

    private static void RetireBMerkleState(WorkerState s, bool waitForPendingBuild)
    {
        var pending = s.BMerkleTreeBuildTask;
        s.BMerkleTreeBuildTask = null;

        if (pending is not null)
        {
            try { CudaDriver.StreamSynchronize(s.BMerkleCopyStream); } catch { /* rotation/shutdown best-effort */ }
            if (waitForPendingBuild)
            {
                try { pending.GetAwaiter().GetResult().Release(); } catch { /* shutdown/rotation best-effort */ }
            }
            else
            {
                _ = pending.ContinueWith(
                    t =>
                    {
                        if (t.Status == TaskStatus.RanToCompletion)
                        {
                            try { t.Result.Release(); } catch { /* best-effort */ }
                        }
                    },
                    CancellationToken.None,
                    TaskContinuationOptions.ExecuteSynchronously,
                    TaskScheduler.Default);
            }
        }

        try { s.BMerkleTree?.Release(); } catch { /* best-effort */ }
        s.BMerkleTree = null;
        s.InstalledHashB = [];
    }

    private static void StartBMerkleBuild(WorkerState s, byte[] bSeed, byte[] jobKey, int n, int rowWidth)
    {
        var bSeedSnapshot = (byte[])bSeed.Clone();
        var keySnapshot = (byte[])jobKey.Clone();
        s.BMerkleTreeBuildTask = StartBMerkleBuildTask(
            s.BState,
            bSeedSnapshot,
            keySnapshot,
            n,
            rowWidth,
            s.BMerkleCopyStream,
            s.CudaContext);
    }

    private static IMerkleTreeHandle EnsureBMerkleTreeReady(WorkerState s)
    {
        if (s.BMerkleTree is not null) return s.BMerkleTree;
        if (s.BMerkleTreeBuildTask is null)
            throw new InvalidOperationException("B Merkle tree is not available for installed sigma.");

        var tree = s.BMerkleTreeBuildTask.GetAwaiter().GetResult();
        s.BMerkleTreeBuildTask = null;
        s.BMerkleTree = tree;
        s.InstalledHashB = tree.Root;
        return tree;
    }

    private static Task<IMerkleTreeHandle> StartBMerkleBuildTask(
        ResidentBStateBuffers bState,
        byte[] bSeedSnapshot,
        byte[] keySnapshot,
        int n,
        int rowWidth,
        CUstream copyStream,
        CUcontext context)
    {
        if (bState.LeafCvBytes > int.MaxValue)
            throw new InvalidOperationException($"B leaf CV buffer is too large ({bState.LeafCvBytes:N0} bytes)");

        int leafCvBytes = (int)bState.LeafCvBytes;
        var leafCvs = GC.AllocateUninitializedArray<byte>(leafCvBytes, pinned: true);
        unsafe
        {
            fixed (byte* pLeafCvs = leafCvs)
            {
                CudaDriver.Check(
                    CudaDriver.MemcpyDtoHAsync((nint)pLeafCvs, bState.LeafCvs, (nuint)leafCvBytes, copyStream),
                    "dtoh async B leaf CVs");
            }
        }

        return Task.Run(() =>
        {
            CudaDriver.Check(CudaDriver.CtxSetCurrent(context), "cuCtxSetCurrent b-merkle");
            CudaDriver.Check(CudaDriver.StreamSynchronize(copyStream), "sync B leaf CV copy");
            return BSeedMerkleTreeHandleFactory.BuildFromLeafCvs(
                leafCvs,
                bSeedSnapshot,
                keySnapshot,
                n,
                rowWidth);
        });
    }

    // Free one half's workspace + buffers.
    private static void FreeHalf(WorkerHalf half)
    {
        if (half.BatchEventsCreated)
        {
            try { CudaDriver.EventDestroy(half.BatchStartEvent); } catch { /* best-effort */ }
            try { CudaDriver.EventDestroy(half.BatchStopEvent); } catch { /* best-effort */ }
            half.BatchEventsCreated = false;
        }

        if (half.Workspace != IntPtr.Zero)
        {
            try { _ = PearlGemm.PearlGemmNative.WorkspaceFree(half.Workspace, half.Stream.Handle); }
            catch { /* best-effort */ }
            half.Workspace = IntPtr.Zero;
            half.GraphReady = false;
        }
        try { half.Buffers?.Dispose(); } catch { /* best-effort */ }
    }

    // Install resident per-σ B state once per GPU worker. This state is not tied
    // to whichever ping/pong half happens to be current, so σ rotation never
    // needs duplicate B uploads just to defend against future half swaps.
    private static void InstallResidentBState(WorkerHalf scratchHalf, SigmaContext ctx, WorkerState s)
    {
        var a = scratchHalf.Buffers;
        var bs = s.BState;
        long bA = (long)a.M * a.K;

        H2D(bs.Key, ctx.JobKey);
        Check("lcg_int7 A throwaway", PearlGemm.PearlGemmNative.LcgInt7Fill(
            a.A.Handle, bA, THROWAWAY_A_SEED_LO, s.SigmaSeed, scratchHalf.Stream.Handle));
        Check("tensor_hash A seed", PearlGemm.PearlGemmNative.TensorHash(
            a.A.Handle, (uint)bA, a.AHash.Handle, bs.Key.Handle,
            NumBlocks(bA), TENSOR_HASH_THREADS, TENSOR_HASH_STAGES,
            TENSOR_HASH_LEAVES, a.Roots.Handle, scratchHalf.DeviceId, scratchHalf.Stream.Handle));

        bool expandB = !bs.BUploaded;
        InstallNativeBProcess(expandB ? ctx.BSeed : null, expandB, a, bs, scratchHalf.DeviceId, scratchHalf.Stream);
        if (expandB)
        {
            bs.BUploaded = true;
        }
    }

    // Install per-σ A-side workspace on one half's stream (no sync — caller
    // syncs after queuing both halves). SigmaSeed and resident B state must
    // already be installed in s before call.
    private static void InstallSigmaHalf(WorkerHalf half, SigmaContext ctx, WorkerState s)
    {
        var b = half.Buffers;
        var bs = s.BState;

        half.GraphReady = false;

        var diffTarget = SigmaContext.NbitsToTarget(ctx.TargetNbits);
        var adjusted   = diffTarget * s.MiningConfig.DifficultyAdjustmentFactor();
        if (adjusted >= BigInteger.One << 256) adjusted = (BigInteger.One << 256) - 1;
        H2D(b.PowTarget, TargetToUint32LE(adjusted));

        if (half.Workspace == IntPtr.Zero)
        {
            unsafe
            {
                nint ws = IntPtr.Zero;
                Check("workspace_alloc", PearlGemm.PearlGemmNative.WorkspaceAlloc(
                    b.M, b.N, b.K, b.R,
                    withNoiseA: 1, withNoiseB: 0,
                    outWorkspace: &ws, half.Stream.Handle));
                half.Workspace = ws;
            }
        }

        unsafe
        {
            var wp = new PearlGemm.PearlGemmNative.WorkspaceParams
            {
                M = b.M, N = b.N, K = b.K, R = b.R,
                BM = BM, BN = BN, BK = BK, CM = CM, CN = CN,
                ThNumBlocks = NumBlocks((long)b.M * b.K),
                ThThreads   = TENSOR_HASH_THREADS,
                ThStages    = TENSOR_HASH_STAGES,
                ThLeaves    = TENSOR_HASH_LEAVES,
                SigmaSeed   = s.SigmaSeed,

                // ── A-side: always this half's own buffers ───────────────────
                // These are written per-iter by pearl_capi_iter and must not be
                // shared between concurrent streams.
                A           = b.A.Handle,
                AHash       = b.AHash.Handle,
                Roots       = b.Roots.Handle,  // scratch for per-iter tensor_hash A
                CommitA     = b.CommitA.Handle,
                CommitB     = b.CommitB.Handle, // written per-iter; NOT shared
                EAL         = b.EAL.Handle,
                EAL_fp16    = b.EALFp16.Handle,
                EAR_R_major = b.EAR_R.Handle,
                EAR_K_major = b.EAR_K.Handle,
                AxEBL_fp16  = b.AxEBLFp16.Handle,
                ApEA        = b.ApEA.Handle,
                A_scales    = b.AScales.Handle,
                C           = b.C.Handle,
                HostSignalSync = b.Sync.Handle,
                PowTarget   = b.PowTarget.Handle,
                PowKey      = b.CommitA.Handle,

                // ── B-side: resident shared state ───────────────────────────
                // Written once at σ-install, then read-only for both halves for
                // the σ lifetime. Not tied to ping/pong swap parity.
                B           = bs.B.Handle,
                BHash       = bs.BHash.Handle,
                Key         = bs.Key.Handle,
                EBR         = bs.EBR.Handle,
                EBR_fp16    = bs.EBRFp16.Handle,
                EBL_R_major = bs.EBL_R.Handle,
                EBL_K_major = bs.EBL_K.Handle,
                EARxBpEB_fp16 = bs.EARxBpEB.Handle,
                BpEB        = bs.BpEB.Handle,
                B_scales    = bs.BScales.Handle,
            };
            Check("workspace_install_params",
                PearlGemm.PearlGemmNative.WorkspaceInstallParams(half.Workspace, &wp));
        }
    }

    private SigmaRotationTimingBuilder InstallSigma(WorkerState s, SigmaContext ctx, double oldBatchDrainMs)
    {
        var timing = new SigmaRotationTimingBuilder { OldBatchDrainMs = oldBatchDrainMs };
        bool sameAsInstalled = s.InstalledSigma.Length > 0
            && s.InstalledSigma.AsSpan().SequenceEqual(ctx.Sigma.AsSpan());

        if (sameAsInstalled && s.InstalledTargetNbits == ctx.TargetNbits)
        {
            return timing;
        }

        // Vardiff fast path: same σ, different target — update PowTarget on both halves.
        // Caller guarantees both streams are idle (Ping was drained, Pong was never queued).
        if (sameAsInstalled && s.InstalledTargetNbits != ctx.TargetNbits)
        {
            _log.LogInformation(
                "worker[{Gpu}]: vardiff retarget {Old:X8} → {New:X8} (same σ)",
                _gpuIndex, s.InstalledTargetNbits, ctx.TargetNbits);
            var diffTargetVd = SigmaContext.NbitsToTarget(ctx.TargetNbits);
            var adjustedVd   = diffTargetVd * s.MiningConfig.DifficultyAdjustmentFactor();
            if (adjustedVd >= BigInteger.One << 256) adjustedVd = (BigInteger.One << 256) - 1;
            var targetBytes = TargetToUint32LE(adjustedVd);
            long vardiffStart = Stopwatch.GetTimestamp();
            H2D(s.Ping.Buffers.PowTarget, targetBytes);
            H2D(s.Pong.Buffers.PowTarget, targetBytes);
            timing.InstallGpuMs = ElapsedMsSince(vardiffStart);
            s.InstalledTargetNbits = ctx.TargetNbits;
            TouchProgress();
            TouchTriggerOrSigma();
            return timing;
        }

        _log.LogInformation(
            "worker[{Gpu}]: σ install job={Job} height={H} nbits=0x{Nbits:X8} (rotate={Rot}) sigma={SigmaPrefix} jobKey={JobKeyPrefix} bSeed={BSeedPrefix} auditK={AuditK}",
            _gpuIndex, ctx.JobId, ctx.BlockHeight, ctx.TargetNbits, !sameAsInstalled,
            Convert.ToHexString(ctx.Sigma.AsSpan(0, Math.Min(8, ctx.Sigma.Length))),
            Convert.ToHexString(ctx.JobKey.AsSpan(0, Math.Min(8, ctx.JobKey.Length))),
            Convert.ToHexString(ctx.BSeed.AsSpan(0, Math.Min(8, ctx.BSeed.Length))),
            ctx.AuditK);

        // The B Merkle handle is keyed by jobKey, so it is rebuilt for every
        // full σ install. BSeed changes no longer force a synchronous host
        // expansion: resident dB is generated directly on the GPU, while proof
        // infrastructure builds from BSeed on a background thread.
        RetireBMerkleState(s, waitForPendingBuild: false);
        if (!s.InstalledBSeed.AsSpan().SequenceEqual(ctx.BSeed.AsSpan()))
        {
            timing.BSeedChanged = true;
            timing.BSeedExpandMs = 0.0; // device expansion is included in install/sync timing.
            s.InstalledBSeed = (byte[])ctx.BSeed.Clone();
            // Force resident B state to regenerate on this σ.
            s.BState.BUploaded = false;
        }

        // SigmaSeed must be set before InstallSigmaHalf is called (both halves read it).
        s.SigmaSeed = BitConverter.ToUInt64(ctx.Sigma.AsSpan(0, 8));

        // Install resident B state once, then point both halves at it.
        long installStart = Stopwatch.GetTimestamp();
        InstallResidentBState(s.Ping, ctx, s);
        InstallSigmaHalf(s.Ping, ctx, s);
        InstallSigmaHalf(s.Pong, ctx, s);
        timing.InstallGpuMs = ElapsedMsSince(installStart);

        long syncStart = Stopwatch.GetTimestamp();
        CudaDriver.Check(CudaDriver.StreamSynchronize(s.Ping.Stream), "sync σ install [ping]");
        timing.SyncPingMs = ElapsedMsSince(syncStart);

        syncStart = Stopwatch.GetTimestamp();
        CudaDriver.Check(CudaDriver.StreamSynchronize(s.Pong.Stream), "sync σ install [pong]");
        timing.SyncPongMs = ElapsedMsSince(syncStart);

        long graphStart = Stopwatch.GetTimestamp();
        PrepareGraphIfEnabled(s.Ping, _mine.CudaGraphIter, _mine.CudaGraphRequired, _log, "ping");
        PrepareGraphIfEnabled(s.Pong, _mine.CudaGraphIter, _mine.CudaGraphRequired, _log, "pong");
        timing.GraphPrepareMs = ElapsedMsSince(graphStart);

        s.InstalledSigma       = (byte[])ctx.Sigma.Clone();
        s.InstalledJobKey      = (byte[])ctx.JobKey.Clone();
        s.InstalledAuditK      = ctx.AuditK;
        s.InstalledTargetNbits = ctx.TargetNbits;
        s.SigmaInstalledAtTimestamp = Stopwatch.GetTimestamp();

        TouchProgress();
        // σ rotation does NOT change trigger probability (Poisson rate is a
        // function of the share target, which only changes on vardiff). We
        // only arm the clock on the FIRST σ install so the watchdog has a
        // baseline; subsequent rotations leave the clock running.
        ArmTriggerClockIfFirst();
        return timing;
    }

    private void StartBMerkleForInstalledSigma(
        WorkerState s,
        SigmaContext ctx,
        SigmaRotationTimingBuilder timing)
    {
        long bMerkleStart = Stopwatch.GetTimestamp();
        StartBMerkleBuild(s, ctx.BSeed, ctx.JobKey, _mine.N, (int)s.Ping.Buffers.K);
        timing.BMerkleMs = ElapsedMsSince(bMerkleStart);
    }

    // ─── Trigger handling: rebuild A for winning seed, D2H, ShareBuilder ──

    private void HandleTrigger(WorkerHalf half, WorkerState s, int winK, ulong winSeed, CancellationToken ct)
    {
        // Per-step timestamp instrumentation — block-find latency matters
        // (a network-target share is ~$5K). We log per-step millis so we
        // can attribute real costs and optimize the dominant step.
        long totalStart = RuntimeTimingStart();

        var b = half.Buffers;
        long bA = (long)b.M * b.K;

        // Snapshot the pinned header into a heap buffer — we're about to
        // launch more device work that may stomp the pinned slot.
        var headerBytes = new byte[b.HeaderSize];
        unsafe { new ReadOnlySpan<byte>((void*)b.HostHeaders[winK], b.HeaderSize).CopyTo(headerBytes); }

        var idxs = HostSignalHeaderLayout.ExtractIndices(headerBytes);
        var dbg  = HostSignalHeaderLayout.DebugInfo(headerBytes);

        _log.LogInformation(
            "worker[{Gpu}]: ✦ trigger at iter={Iter} tile=({Tr},{Tc})",
            _gpuIndex, winSeed, dbg.TileRow, dbg.TileCol);

        // CRITICAL: intermediate iters in this batch overwrote dA. We MUST
        // re-derive dA for the winning seed before D2H, or the proof's A
        // bytes won't match the GPU-side hashes and the pool rejects the
        // share with `a_merkle_mismatch`.
        //
        // Profiling: split this into two measurements. The first
        // StreamSynchronize drains everything already queued on this half's
        // stream (up to mpp matmuls × iter_ms each). That wait is what
        // actually dominates msRegen — the LCG kernel itself is sub-ms.
        // Surfacing the two separately lets us prove the queue depth is
        // the cost driver, and confirms the benchmark-driven mpp cap is
        // working as intended.
        long stageStart = RuntimeTimingStart();
        CudaDriver.Check(CudaDriver.StreamSynchronize(half.Stream), "drain pre-regen");
        double msDrainWait = RuntimeTimingElapsedMs(stageStart);

        stageStart = RuntimeTimingStart();
        if (b.MatmulsPerPoll > 1 || winSeed != s.GlobalIterIdx - 1)
        {
            Check("lcg_int7 A regen", PearlGemm.PearlGemmNative.LcgInt7Fill(
                b.A.Handle, bA, winSeed, s.SigmaSeed, half.Stream.Handle));
            CudaDriver.Check(CudaDriver.StreamSynchronize(half.Stream), "sync A regen");
        }
        double msLcgKernel = RuntimeTimingElapsedMs(stageStart);
        double msRegen = msDrainWait + msLcgKernel;

        var aRowsU32 = Array.ConvertAll(idxs.ARowIndices, i => (uint)i);
        var bColsU32 = Array.ConvertAll(idxs.BColumnIndices, i => (uint)i);

        byte[]? aBytes;
        byte[]? aSlice;
        byte[]? aLeafCvs;

        // Fast path for the production shapes: compute A's full leaf-CV table
        // on GPU, D2H only that compact CV table plus the opened A rows, and
        // let ShareFinalizer build the proof from CVs. This replaces a full-A
        // D2H and full CPU Merkle rebuild on each share. Keep the old full-A
        // path for non-1024-aligned row widths or unexpected row patterns.
        stageStart = RuntimeTimingStart();
        if (CanUseFastAProofPath(b, aRowsU32, bA))
        {
            Check("tensor_hash A leaf_cvs", PearlGemm.PearlGemmNative.TensorHashLeafCvs(
                b.A.Handle, (uint)bA, b.AHash.Handle, s.BState.Key.Handle,
                NumBlocks(bA), TENSOR_HASH_THREADS, TENSOR_HASH_STAGES,
                TENSOR_HASH_LEAVES, b.Roots.Handle, b.ALeafCvs.Handle, half.DeviceId, half.Stream.Handle));

            CudaDriver.Check(
                CudaDriver.MemcpyDtoHAsync(b.HostALeafCvsPtr, b.ALeafCvs, (nuint)b.HostALeafCvsSize, half.Stream),
                "D2H A leaf CVs async");

            long selectedOffset = 0;
            for (int i = 0; i < aRowsU32.Length; i++)
            {
                long rowOffset = (long)aRowsU32[i] * b.K;
                CudaDriver.Check(
                    CudaDriver.MemcpyDtoHAsync(
                        b.HostASelectedPtr + (nint)selectedOffset,
                        DevicePtrOffset(b.A, rowOffset),
                        (nuint)b.K,
                        half.Stream),
                    "D2H A selected row async");
                selectedOffset += b.K;
            }

            CudaDriver.Check(CudaDriver.StreamSynchronize(half.Stream), "sync D2H A proof data");
            aLeafCvs = CopyPinnedHost(b.HostALeafCvsPtr, b.HostALeafCvsSize);
            aSlice = CopyPinnedHost(b.HostASelectedPtr, checked((long)aRowsU32.Length * b.K));
            aBytes = null;
        }
        else
        {
            // D2H the regenerated A via the pre-pinned host buffer + async memcpy.
            // Pageable byte[] dest forces CUDA to stage through a hidden bounce
            // buffer (~37 ms for 16 MiB). Pre-pinned direct DMA is ~1–2 ms.
            // We then copy out to a managed byte[] (host-to-host, ~3 ms via the
            // memory subsystem) so the SharePayload owns its bytes free-and-clear
            // and the pinned slot is reusable for the next trigger.
            CudaDriver.Check(
                CudaDriver.MemcpyDtoHAsync(b.HostAPtr, b.A, (nuint)bA, half.Stream),
                "D2H A async");
            CudaDriver.Check(CudaDriver.StreamSynchronize(half.Stream), "sync D2H A");
            aBytes = CopyPinnedHost(b.HostAPtr, bA);
            aSlice = null;
            aLeafCvs = null;
        }
        double msD2H = RuntimeTimingElapsedMs(stageStart);

        // Extract the B-merkle inclusion proof on the mining thread BEFORE
        // enqueueing the payload. Reasons:
        //   1. s.BMerkleTree is Dispose()d in InstallSigma on the next σ
        //      rotation — handing the live handle to ShareFinalizer (which
        //      runs on a separate task) is a use-after-free hazard.
        //      MerkleRootAndProofResult is owned plain byte arrays, safe
        //      to cross the channel.
        //   2. The proof is cheap (microseconds — the tree is cached) so
        //      we don't lose meaningful overlap by doing it here.
        //   3. Snapshotting at this point binds the proof to the *same*
        //      σ/jobKey/BTree state the GPU just used for the matmul.
        //      Any later state mutation cannot decorrelate inputs.
        var bTree = EnsureBMerkleTreeReady(s);
        var bProof = bTree.Proof(bColsU32);

        // Acquire a refcount on the live B-merkle tree so the consumer
        // task can open the audit-proof v1 K-paths after we've already
        // moved on. Released inside ShareFinalizer.Finalize (or on
        // TryEnqueue failure → ShareFinalizer drops + Releases). If σ
        // rotates while this share is in flight, InstallSigma will
        // Release the worker's reference but the native handle survives
        // until this Acquire is paired off.
        var bTreeForPayload = bTree;
        bTreeForPayload.Acquire();

        // Hand off everything from here on to ShareFinalizer so the mining
        // loop can resume immediately. Build (37–42 ms) and submit run on a
        // dedicated consumer task; we incur only the channel-write cost
        // (sub-microsecond) on the hot path. Block-find latency to the pool
        // is unchanged (work still has to complete before submit), but GPU
        // throughput recovers ~40 ms per trigger by overlapping with the
        // next iteration batch.
        var payload = new SharePayload(
            GpuIndex:               _gpuIndex,
            Sigma:                  (byte[])s.InstalledSigma.Clone(),
            ConfigBytes:            s.MiningConfig.ToBytes(),
            JobKey:                 (byte[])s.InstalledJobKey.Clone(),
            ABytes:                 aBytes,
            ASlice:                 aSlice,
            ALeafCvs:               aLeafCvs,
            BProof:                 bProof,
            BMerkleTree:            bTreeForPayload,
            BSeed:                  (byte[])s.InstalledBSeed.Clone(),
            AuditK:                 s.InstalledAuditK,
            ARowIndices:            aRowsU32,
            BColIndices:            bColsU32,
            // wire field name is `tile_row`/`tile_col` but the proto semantic
            // is the ABSOLUTE STARTING row/col offset — i.e. min(aRowIndices)
            // — not the GPU tile coord.
            // Verifier re-derives the opened rows via
            //   RowsPattern.IndicesWithOffset(t_rows)
            // and that must equal what our Rust-canonical merkle proof opens.
            // aRowsU32[0] is min(aRowIndices) (ExtractIndices builds it from
            // a SortedSet, ascending). Same for bColsU32[0].
            TileRow:                (int)aRowsU32[0],
            TileCol:                (int)bColsU32[0],
            M:                      b.M,
            N:                      b.N,
            K:                      b.K,
            R:                      b.R,
            ClaimedDifficultyNbits: s.InstalledTargetNbits,
            MiningConfig:           s.MiningConfig,
            MsTotalSoFar:           RuntimeTimingElapsedMs(totalStart),
            MsDrainWait:            msDrainWait,
            MsLcgKernel:            msLcgKernel,
            MsD2H:                  msD2H);

        s.SharesEmitted++;
        _finalizer.TryEnqueue(payload);
    }

    // ─── Stats logging ─────────────────────────────────────────────────────

    private void MaybeLogStats(WorkerState s)
    {
        double dWall = ElapsedSecondsSince(s.StatsWindowStartedAt);
        if (dWall < StatsIntervalSeconds) return;
        if (s.GlobalIterIdx <= s.ItersAtLastReport) return;

        ulong dIters  = s.GlobalIterIdx - s.ItersAtLastReport;
        double ips    = dIters / dWall;

        long tilesPerIter = TilesPerIter(s.Ping.Buffers.M, s.Ping.Buffers.N, s.MiningConfig);
        double tilesPerSec= ips * tilesPerIter;
        double madsPerIter= (double)s.Ping.Buffers.M * s.Ping.Buffers.N * s.Ping.Buffers.K;
        double tmadsPerSec= (madsPerIter * dIters / dWall) / 1e12;
        double hashesPerSec = tilesPerSec * s.MiningConfig.DifficultyAdjustmentFactor();
        double expectedOpensPerSec = ExpectedOpensPerSecond(
            tilesPerSec, s.InstalledTargetNbits, s.MiningConfig);
        double iterMs     = dWall * 1000.0 / dIters;
        Volatile.Write(ref _lastIterMsBits, BitConverter.DoubleToInt64Bits(iterMs));

        _log.LogInformation(
            "worker[{Gpu}] [stats] iters/s={Ips:F2} iter_ms={Ms:F1} tiles/s={Tps:E2} tmads/s={Tmads:F2} hashes/s={Hps} triggers={Trig} shares={Sh} σ_age={Age:F0}s expected_opens/s={Eos:E3}",
            _gpuIndex, ips, iterMs, tilesPerSec, tmadsPerSec, FormatHashRate(hashesPerSec),
            s.TriggersTotal, s.SharesEmitted, ElapsedSecondsSince(s.SigmaInstalledAtTimestamp),
            expectedOpensPerSec);

        Metrics.SetThroughput(_gpuIndex, ips, tmadsPerSec, hashesPerSec, iterMs,
            tilesPerSec, expectedOpensPerSec);
        Metrics.IncIters(_gpuIndex, (long)dIters);

        s.ItersAtLastReport = s.GlobalIterIdx;
        s.StatsWindowStartedAt = Stopwatch.GetTimestamp();
    }

    // ─── Static helpers (small, lifted from v1 MineBlocks.cs verbatim) ────

    private static uint NumBlocks(long bytes)
    {
        uint bpb = TENSOR_HASH_THREADS * 1024;
        return (uint)((bytes + bpb - 1) / bpb);
    }

    private static bool CanUseFastAProofPath(WorkerBuffers b, uint[] rowIndices, long aBytes)
    {
        if (rowIndices.Length == 0) return false;
        if (b.K % Blake3.ChunkLen != 0) return false;
        if (aBytes <= 0 || aBytes > uint.MaxValue) return false;
        if (b.HostALeafCvsSize <= 0 || b.HostALeafCvsSize > int.MaxValue) return false;

        long selectedBytes = checked((long)rowIndices.Length * b.K);
        if (selectedBytes <= 0 || selectedBytes > b.HostASelectedSize || selectedBytes > int.MaxValue)
            return false;

        foreach (uint row in rowIndices)
        {
            if (row >= (uint)b.M) return false;
        }
        return true;
    }

    private static CUdeviceptr DevicePtrOffset(CUdeviceptr ptr, long offsetBytes)
        => new(ptr.Handle + (nint)offsetBytes);

    private static byte[] CopyPinnedHost(nint ptr, long bytes)
    {
        if (bytes < 0 || bytes > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(bytes), $"Cannot copy {bytes:N0} bytes into a managed array.");

        var dst = GC.AllocateUninitializedArray<byte>((int)bytes);
        unsafe { new ReadOnlySpan<byte>((void*)ptr, (int)bytes).CopyTo(dst); }
        return dst;
    }

    private static byte[] TargetToUint32LE(BigInteger target)
    {
        var raw = target.ToByteArray(isUnsigned: true, isBigEndian: false);
        var buf = new byte[32];
        Array.Copy(raw, buf, Math.Min(raw.Length, 32));
        return buf;
    }

    private static void H2D(CUdeviceptr dst, byte[] src)
    {
        unsafe
        {
            fixed (byte* p = src)
                CudaDriver.Check(CudaDriver.MemcpyHtoD(dst, (nint)p, (nuint)src.Length), "H2D");
        }
    }

    private static void InstallNativeBProcess(
        byte[]? bSeed,
        bool expandBSeed,
        WorkerBuffers a,
        ResidentBStateBuffers bState,
        int deviceId,
        CUstream stream)
    {
        if (expandBSeed && (bSeed is null || bSeed.Length != SigmaContext.BSeedSize))
            throw new ArgumentException($"BSeed must be {SigmaContext.BSeedSize} bytes.", nameof(bSeed));

        byte[] seedBuffer = bSeed ?? Array.Empty<byte>();
        unsafe
        {
            fixed (byte* pSeed = seedBuffer)
            {
                var p = new PearlGemm.PearlGemmNative.InstallBParams
                {
                    M = a.M,
                    N = bState.N,
                    K = bState.K,
                    R = bState.R,
                    ExpandBSeed = expandBSeed ? 1 : 0,
                    ThNumBlocks = NumBlocks((long)bState.N * bState.K),
                    ThThreads = TENSOR_HASH_THREADS,
                    ThStages = TENSOR_HASH_STAGES,
                    ThLeaves = TENSOR_HASH_LEAVES,
                    DeviceId = deviceId,

                    BSeed = expandBSeed ? (nint)pSeed : nint.Zero,
                    B = bState.B.Handle,
                    BHash = bState.BHash.Handle,
                    Key = bState.Key.Handle,
                    Roots = bState.Roots.Handle,
                    AHash = a.AHash.Handle,
                    CommitA = a.CommitA.Handle,
                    CommitB = a.CommitB.Handle,
                    EAR_K_major = bState.EAR_K.Handle,
                    EBL_R_major = bState.EBL_R.Handle,
                    EBL_K_major = bState.EBL_K.Handle,
                    EBR = bState.EBR.Handle,
                    EBR_fp16 = bState.EBRFp16.Handle,
                    EARxBpEB = bState.EARxBpEB.Handle,
                    BpEB = bState.BpEB.Handle,
                    Workspace = bState.NoiseWorkspace,
                    LeafCvs = bState.LeafCvs.Handle,
                };
                Check("install_B", PearlGemm.PearlGemmNative.InstallB(&p, stream.Handle));
            }
        }
    }

    private static void Check(string op, int rc)
    {
        if (rc != 0) throw new InvalidOperationException($"{op} rc={rc}");
    }

    private static string FormatHashRate(double hps)
    {
        if (!double.IsFinite(hps) || hps <= 0) return "0 H/s";
        string[] units = { "H/s", "kH/s", "MH/s", "GH/s", "TH/s", "PH/s", "EH/s" };
        int i = 0;
        while (hps >= 1000.0 && i < units.Length - 1) { hps /= 1000.0; i++; }
        return $"{hps:F2} {units[i]}";
    }

    private static double ElapsedMsSince(long startTimestamp)
        => ElapsedMs(startTimestamp, Stopwatch.GetTimestamp());

    private static double ElapsedMs(long startTimestamp, long endTimestamp)
        => (endTimestamp - startTimestamp) * 1000.0 / Stopwatch.Frequency;

    private static TimeSpan ElapsedSince(long startTimestamp)
        => TimeSpan.FromSeconds(ElapsedSecondsSince(startTimestamp));

    private static double ElapsedSecondsSince(long startTimestamp)
        => (Stopwatch.GetTimestamp() - startTimestamp) / (double)Stopwatch.Frequency;

    private static long RuntimeTimingStart()
        => RuntimePerfTimings ? Stopwatch.GetTimestamp() : 0;

    private static double RuntimeTimingElapsedMs(long startTimestamp)
        => RuntimePerfTimings ? ElapsedMsSince(startTimestamp) : 0.0;

    private static double Percentile(List<double> values, double p)
    {
        if (values.Count == 0) return 0.0;
        var sorted = values.ToArray();
        Array.Sort(sorted);
        int index = (int)Math.Ceiling(Math.Clamp(p, 0.0, 1.0) * (sorted.Length - 1));
        return sorted[index];
    }

    public async ValueTask DisposeAsync()
    {
        try { _cts.Cancel(); } catch { /* already disposed */ }
        if (_thread is not null)
        {
            var winner = await Task.WhenAny(_exited.Task, Task.Delay(DisposeGrace))
                                   .ConfigureAwait(false);
            if (winner == _exited.Task)
            {
                Volatile.Write(ref _exitedCleanly, 1);
            }
            else
            {
                // The thread is wedged past the grace budget — almost
                // certainly inside a CUDA call that won't return. We CAN'T
                // force a managed Thread to exit without risking stuck
                // finalizers / orphaned native state. Surface this loudly:
                // the next reconnect's fresh worker will race the orphan
                // for the GPU's primary context. The watchdog should have
                // already tripped at this point; this log line is the
                // smoking gun for forensics.
                _log.LogCritical(
                    "worker[{Gpu}]: did not exit within {Grace}s of cancellation — orphaning thread (state={State})",
                    _gpuIndex, DisposeGrace.TotalSeconds, _thread.ThreadState);
            }
        }
        else
        {
            // Never started — trivially clean.
            Volatile.Write(ref _exitedCleanly, 1);
        }
        _cts.Dispose();
    }

    // ─── Hashrate benchmark (used to populate RegisterRequest.claimed_total_hashrate) ──
    //
    // Runs the full GEMM pipeline (σ install + repeated RunBatch) against a
    // SYNTHETIC σ for `duration` seconds and reports throughput. Used at
    // startup (before session.ConnectAsync) so we can give the pool an
    // accurate hashrate estimate in Register / GpuCard.hashrate.
    //
    // The synthetic σ uses:
    //   • jobKey  = BLAKE3("akoya-hashrate-benchmark-v1")           (deterministic)
    //   • bBytes  = BSeedExpander(BLAKE3("akoya-hashrate-bench-B"), N, K)
    //   • PowTarget = 0³² (all zeros)  → noisy_gemm can never trigger,
    //                                    so we run pure GEMM throughput
    //                                    and the trigger path is never
    //                                    exercised (no torn share, no D2H).
    //
    // This is the same pipeline GpuWorker.Loop runs minus the JobBus poll,
    // share submit, and pHeader scan. Identical kernel mix → identical
    // performance characteristics, so the reported TMADs/sec is what
    // real-job mining will deliver.
    public readonly record struct BenchmarkResult(
        double TmadsPerSec,
        double ItersPerSec,
        ulong  IterCount,
        double DurationSec,
        long   TilesPerIter,
        double TilesPerSec)
    {
        public BenchmarkResult(
            double tmadsPerSec,
            double itersPerSec,
            ulong  iterCount,
            double durationSec)
            : this(tmadsPerSec, itersPerSec, iterCount, durationSec, 0, 0.0)
        {
        }

        /// <summary>Mean wall time per matmul iteration, in milliseconds.
        /// Used by the orchestrator to size <c>MatmulsPerPoll</c> so the
        /// trigger-detection budget (queue drain wait on a share trigger)
        /// stays below the configured ceiling.</summary>
        public double IterMs => ItersPerSec > 0.0 ? 1000.0 / ItersPerSec : double.PositiveInfinity;
    }

    public readonly record struct SigmaRotateBenchmarkResult(
        string Case,
        int Run,
        bool BSeedChanged,
        int Mpp,
        double JobSeenToFirstNewBatchQueuedMs,
        double OldBatchDrainMs,
        double BSeedExpandMs,
        double BH2DMs,
        double BProcessMs,
        double TensorHashBMs,
        double ASeedHashMs,
        double CommitMs,
        double NoiseGenMs,
        double WorkspaceAllocMs,
        double NoiseBMs,
        double WorkspaceInstallMs,
        double GraphPrepareMs,
        double BMerkleTreeMs,
        double FirstQueueMs);

    public readonly record struct IterLoopBenchmarkResult(
        int Run,
        int Mpp,
        bool GraphIter,
        bool DisablePong,
        int Samples,
        ulong IterCount,
        double DurationSec,
        double ItersPerSec,
        double IterMs,
        double LoopBatchWallP50Ms,
        double LoopBatchWallP95Ms,
        double QueueP50Ms,
        double QueueP95Ms,
        double HeaderClearP50Ms,
        double HeaderPtrP50Ms,
        double DeviceClearEnqueueP50Ms,
        double IterEnqueueP50Ms,
        double GraphLaunchP50Ms,
        double SyncWaitP50Ms,
        double SyncWaitP95Ms,
        double ScanP50Ms,
        double GpuBatchP50Ms,
        double GpuBatchP95Ms,
        double HostNonWaitP50Ms,
        int GraphLaunchCount);

    public static IReadOnlyList<SigmaRotateBenchmarkResult> RunSigmaRotateBenchmark(
        int deviceOrdinal,
        MineOptions mine,
        int repeat,
        ILogger log,
        CancellationToken ct = default)
    {
        ValidateCtaTileDivisibility(mine.M, mine.N);
        repeat = Math.Max(1, repeat);

        CudaDriver.Check(CudaDriver.Init(0), "cuInit");
        CudaDriver.Check(CudaDriver.DeviceGet(out var dev, deviceOrdinal), "cuDeviceGet");
        CudaDriver.Check(CudaDriver.DevicePrimaryCtxRetain(out var primary, dev),
            "cuDevicePrimaryCtxRetain");
        bool deviceRetained = true;
        CudaDriver.Check(CudaDriver.CtxSetCurrent(primary), "cuCtxSetCurrent");

        CUstream stream0 = default, stream1 = default, stream2 = default;
        bool stream0Created = false, stream1Created = false, stream2Created = false;
        var ping = new WorkerHalf { DeviceId = deviceOrdinal };
        var pong = new WorkerHalf { DeviceId = deviceOrdinal };
        ResidentBStateBuffers? bState = null;
        IMerkleTreeHandle? bTree = null;

        try
        {
            CudaDriver.Check(CudaDriver.StreamCreate(out stream0, 0), "cuStreamCreate [sigma ping]");
            stream0Created = true;
            CudaDriver.Check(CudaDriver.StreamCreate(out stream1, 0), "cuStreamCreate [sigma pong]");
            stream1Created = true;
            CudaDriver.Check(CudaDriver.StreamCreate(out stream2, 0), "cuStreamCreate [sigma b-merkle]");
            stream2Created = true;
            ping.Stream = stream0;
            pong.Stream = stream1;
            ping.Buffers = new WorkerBuffers(mine.M, mine.N, mine.K, mine.NoiseRank, mine.MatmulsPerPoll);
            pong.Buffers = new WorkerBuffers(mine.M, mine.N, mine.K, mine.NoiseRank, mine.MatmulsPerPoll);
            bState = new ResidentBStateBuffers(mine.N, mine.K, mine.NoiseRank, stream0);

            long bA = (long)mine.M * mine.K;

            var powTarget = new byte[32];
            var currentBSeed = Blake3.Hash(System.Text.Encoding.ASCII.GetBytes("akoya-sigma-rotate-B-0"));

            var initialJobKey = Blake3.Hash(System.Text.Encoding.ASCII.GetBytes("akoya-sigma-rotate-job-0"));
            ulong sigmaSeed = 0xA1A2_A3A4_A5A6_A7A8ul;
            InstallBenchmarkResidentBState(ping, bState, initialJobKey, currentBSeed, sigmaSeed);
            InstallBenchmarkHalf(ping, bState, powTarget, sigmaSeed);
            InstallBenchmarkHalf(pong, bState, powTarget, sigmaSeed);
            CudaDriver.Check(CudaDriver.StreamSynchronize(ping.Stream), "sync sigma bench initial [ping]");
            CudaDriver.Check(CudaDriver.StreamSynchronize(pong.Stream), "sync sigma bench initial [pong]");

            int batch = mine.MatmulsPerPoll;
            ulong nextSeed = 0;
            QueueBatch(ping, batch, nextSeed);
            nextSeed += (ulong)batch;

            var results = new List<SigmaRotateBenchmarkResult>(repeat * 2);
            foreach (bool bSeedChanged in new[] { false, true })
            {
                string caseName = bSeedChanged ? "bseed_changed" : "bseed_stable";
                for (int run = 1; run <= repeat && !ct.IsCancellationRequested; run++)
                {
                    var total = Stopwatch.StartNew();

                    var oldDrainMs = MeasureMs(() =>
                    {
                        _ = SyncAndScan(ping, batch);
                    });

                    var jobKey = Blake3.Hash(System.Text.Encoding.ASCII.GetBytes(
                        $"akoya-sigma-rotate-job-{caseName}-{run}"));
                    sigmaSeed += 0x9E37_79B9_7F4A_7C15ul;

                    if (bSeedChanged)
                    {
                        currentBSeed = Blake3.Hash(System.Text.Encoding.ASCII.GetBytes(
                            $"akoya-sigma-rotate-B-{run}"));
                        bState.BUploaded = false;
                    }

                    H2D(ping.Buffers.PowTarget, powTarget);
                    H2D(pong.Buffers.PowTarget, powTarget);
                    H2D(bState.Key, jobKey);

                    double bH2DMs = 0.0;
                    double bProcessMs = 0.0;
                    double bSeedExpandMs = 0.0;
                    double tensorHashBMs = 0.0;
                    double commitMs = 0.0;
                    double noiseGenMs = 0.0;
                    double noiseBMs = 0.0;

                    double aSeedHashMs = MeasureStreamStageMs(ping.Stream, () =>
                    {
                        Check("lcg_int7 A throwaway", PearlGemm.PearlGemmNative.LcgInt7Fill(
                            ping.Buffers.A.Handle, bA, THROWAWAY_A_SEED_LO, sigmaSeed, ping.Stream.Handle));
                        Check("tensor_hash A seed", PearlGemm.PearlGemmNative.TensorHash(
                            ping.Buffers.A.Handle, (uint)bA, ping.Buffers.AHash.Handle, bState.Key.Handle,
                            NumBlocks(bA), TENSOR_HASH_THREADS, TENSOR_HASH_STAGES,
                            TENSOR_HASH_LEAVES, ping.Buffers.Roots.Handle, ping.DeviceId, ping.Stream.Handle));
                    });

                    bool expandB = !bState.BUploaded;
                    bProcessMs = MeasureStreamStageMs(ping.Stream, () =>
                        InstallNativeBProcess(
                            expandB ? currentBSeed : null,
                            expandB,
                            ping.Buffers,
                            bState,
                            ping.DeviceId,
                            ping.Stream));
                    if (expandB)
                    {
                        bState.BUploaded = true;
                    }

                    double workspaceAllocMs = MeasureMs(() =>
                    {
                        unsafe
                        {
                            if (ping.Workspace == IntPtr.Zero)
                            {
                                nint ws = IntPtr.Zero;
                                Check("workspace_alloc ping", PearlGemm.PearlGemmNative.WorkspaceAlloc(
                                    ping.Buffers.M, ping.Buffers.N, ping.Buffers.K, ping.Buffers.R,
                                    withNoiseA: 1, withNoiseB: 0,
                                    outWorkspace: &ws, ping.Stream.Handle));
                                ping.Workspace = ws;
                            }

                            if (pong.Workspace == IntPtr.Zero)
                            {
                                nint ws = IntPtr.Zero;
                                Check("workspace_alloc pong", PearlGemm.PearlGemmNative.WorkspaceAlloc(
                                    pong.Buffers.M, pong.Buffers.N, pong.Buffers.K, pong.Buffers.R,
                                    withNoiseA: 1, withNoiseB: 0,
                                    outWorkspace: &ws, pong.Stream.Handle));
                                pong.Workspace = ws;
                            }
                        }
                        CudaDriver.Check(CudaDriver.StreamSynchronize(ping.Stream), "sync workspace alloc ping");
                        CudaDriver.Check(CudaDriver.StreamSynchronize(pong.Stream), "sync workspace alloc pong");
                    });

                    double workspaceInstallMs = MeasureMs(() =>
                    {
                        InstallBenchmarkWorkspaceParams(ping, sigmaSeed, bState);
                        InstallBenchmarkWorkspaceParams(pong, sigmaSeed, bState);
                    });

                    double graphPrepareMs = MeasureMs(() =>
                    {
                        PrepareGraphIfEnabled(ping, mine.CudaGraphIter, mine.CudaGraphRequired, log, "sigma-bench-ping");
                        PrepareGraphIfEnabled(pong, mine.CudaGraphIter, mine.CudaGraphRequired, log, "sigma-bench-pong");
                    });

                    double firstQueueMs = MeasureMs(() => QueueBatch(ping, batch, nextSeed));
                    nextSeed += (ulong)batch;
                    total.Stop();

                    bTree?.Release();
                    bTree = null;
                    var swBMerkleFull = Stopwatch.StartNew();
                    var bTreeTask = StartBMerkleBuildTask(
                        bState,
                        currentBSeed,
                        jobKey,
                        mine.N,
                        mine.K,
                        stream2,
                        primary);
                    bTree = bTreeTask.GetAwaiter().GetResult();
                    swBMerkleFull.Stop();
                    double bMerkleTreeMs = swBMerkleFull.Elapsed.TotalMilliseconds;

                    results.Add(new SigmaRotateBenchmarkResult(
                        caseName,
                        run,
                        bSeedChanged,
                        batch,
                        total.Elapsed.TotalMilliseconds,
                        oldDrainMs,
                        bSeedExpandMs,
                        bH2DMs,
                        bProcessMs,
                        tensorHashBMs,
                        aSeedHashMs,
                        commitMs,
                        noiseGenMs,
                        workspaceAllocMs,
                        noiseBMs,
                        workspaceInstallMs,
                        graphPrepareMs,
                        bMerkleTreeMs,
                        firstQueueMs));
                }
            }

            return results;
        }
        finally
        {
            try { bTree?.Release(); } catch { }
            if (stream0Created)
                try { CudaDriver.StreamSynchronize(stream0); } catch { }
            if (stream1Created)
                try { CudaDriver.StreamSynchronize(stream1); } catch { }
            if (stream2Created)
                try { CudaDriver.StreamSynchronize(stream2); } catch { }

            FreeHalf(ping);
            FreeHalf(pong);
            try { bState?.Dispose(); } catch { }

            if (stream0Created)
                try { CudaDriver.StreamDestroy(stream0); } catch { }
            if (stream1Created)
                try { CudaDriver.StreamDestroy(stream1); } catch { }
            if (stream2Created)
                try { CudaDriver.StreamDestroy(stream2); } catch { }
            if (deviceRetained)
                try { CudaDriver.DevicePrimaryCtxRelease(dev); } catch { }
        }
    }

    public static IterLoopBenchmarkResult RunIterLoopBenchmark(
        int deviceOrdinal,
        MineOptions mine,
        TimeSpan duration,
        ILogger log,
        int run = 1,
        CancellationToken ct = default)
    {
        if (duration <= TimeSpan.Zero) duration = TimeSpan.FromSeconds(10);
        ValidateCtaTileDivisibility(mine.M, mine.N);

        log.LogInformation(
            "iter-loop[{Ord}]: starting {Sec:F0}s raw-loop sample",
            deviceOrdinal, duration.TotalSeconds);

        CudaDriver.Check(CudaDriver.Init(0), "cuInit");
        CudaDriver.Check(CudaDriver.DeviceGet(out var dev, deviceOrdinal), "cuDeviceGet");
        CudaDriver.Check(CudaDriver.DevicePrimaryCtxRetain(out var primary, dev),
            "cuDevicePrimaryCtxRetain");
        bool deviceRetained = true;
        CudaDriver.Check(CudaDriver.CtxSetCurrent(primary), "cuCtxSetCurrent");

        CUstream stream0 = default, stream1 = default;
        bool stream0Created = false, stream1Created = false;
        var ping = new WorkerHalf { DeviceId = deviceOrdinal };
        var pong = new WorkerHalf { DeviceId = deviceOrdinal };
        ResidentBStateBuffers? bState = null;

        try
        {
            CudaDriver.Check(CudaDriver.StreamCreate(out stream0, 0), "cuStreamCreate [iter ping]");
            stream0Created = true;
            CudaDriver.Check(CudaDriver.StreamCreate(out stream1, 0), "cuStreamCreate [iter pong]");
            stream1Created = true;
            ping.Stream = stream0;
            pong.Stream = stream1;
            ping.Buffers = new WorkerBuffers(mine.M, mine.N, mine.K, mine.NoiseRank, mine.MatmulsPerPoll);
            pong.Buffers = new WorkerBuffers(mine.M, mine.N, mine.K, mine.NoiseRank, mine.MatmulsPerPoll);
            bState = new ResidentBStateBuffers(mine.N, mine.K, mine.NoiseRank, stream0);

            var jobKey = Blake3.Hash(System.Text.Encoding.ASCII.GetBytes("akoya-iter-loop-benchmark-v1"));
            var bSeed = Blake3.Hash(System.Text.Encoding.ASCII.GetBytes("akoya-iter-loop-bench-B"));
            var powTarget = new byte[32];
            ulong sigmaSeed = 0xA1A2_A3A4_A5A6_A7A8ul;

            InstallBenchmarkResidentBState(ping, bState, jobKey, bSeed, sigmaSeed);
            InstallBenchmarkHalf(ping, bState, powTarget, sigmaSeed);
            InstallBenchmarkHalf(pong, bState, powTarget, sigmaSeed);
            CudaDriver.Check(CudaDriver.StreamSynchronize(ping.Stream), "sync iter install [ping]");
            CudaDriver.Check(CudaDriver.StreamSynchronize(pong.Stream), "sync iter install [pong]");
            PrepareGraphIfEnabled(ping, mine.CudaGraphIter, mine.CudaGraphRequired, log, "iter-ping");
            PrepareGraphIfEnabled(pong, mine.CudaGraphIter, mine.CudaGraphRequired, log, "iter-pong");

            int batch = mine.MatmulsPerPoll;
            ulong nextSeed = 0;

            QueueBatch(ping, batch, nextSeed);
            nextSeed += (ulong)batch;
            _ = SyncAndScan(ping, batch);
            if (!mine.DisablePong)
            {
                QueueBatch(pong, batch, nextSeed);
                nextSeed += (ulong)batch;
                _ = SyncAndScan(pong, batch);
            }

            var loopBatchWall = new List<double>(4096);
            var queueTotal = new List<double>(4096);
            var headerClear = new List<double>(4096);
            var headerPtr = new List<double>(4096);
            var deviceClear = new List<double>(4096);
            var iterEnqueue = new List<double>(4096);
            var graphLaunch = new List<double>(4096);
            var syncWait = new List<double>(4096);
            var scan = new List<double>(4096);
            var gpuBatch = new List<double>(4096);
            var hostNonWait = new List<double>(4096);
            int graphLaunchCount = 0;

            QueueBatchTiming pingQueueTiming = QueueBatchTimedWithEvents(ping, batch, nextSeed);
            nextSeed += (ulong)batch;

            long clockStart = Stopwatch.GetTimestamp();
            ulong iters = 0;
            while (ElapsedSince(clockStart) < duration && !ct.IsCancellationRequested)
            {
                QueueBatchTiming? nextPingQueueTiming = null;
                long loopStart = Stopwatch.GetTimestamp();
                if (!mine.DisablePong)
                {
                    nextPingQueueTiming = QueueBatchTimedWithEvents(pong, batch, nextSeed);
                    nextSeed += (ulong)batch;
                }

                var syncTiming = SyncAndScanTimed(ping, batch);
                double gpuMs = ReadCompletedBatchGpuMs(ping);
                double loopMs = ElapsedMsSince(loopStart);

                queueTotal.Add(pingQueueTiming.TotalMs);
                headerClear.Add(pingQueueTiming.HeaderClearMs);
                headerPtr.Add(pingQueueTiming.HeaderPtrMs);
                deviceClear.Add(pingQueueTiming.DeviceClearEnqueueMs);
                iterEnqueue.Add(pingQueueTiming.IterEnqueueMs);
                graphLaunch.Add(pingQueueTiming.GraphLaunchMs);
                if (pingQueueTiming.UsedGraph) graphLaunchCount++;
                syncWait.Add(syncTiming.SyncMs);
                scan.Add(syncTiming.ScanMs);
                gpuBatch.Add(gpuMs);
                loopBatchWall.Add(loopMs);
                hostNonWait.Add(pingQueueTiming.TotalMs + syncTiming.ScanMs);

                iters += (ulong)batch;
                if (ElapsedSince(clockStart) >= duration || ct.IsCancellationRequested) break;

                if (mine.DisablePong)
                {
                    pingQueueTiming = QueueBatchTimedWithEvents(ping, batch, nextSeed);
                    nextSeed += (ulong)batch;
                }
                else
                {
                    (ping, pong) = (pong, ping);
                    pingQueueTiming = nextPingQueueTiming.GetValueOrDefault();
                }
            }

            double sec = ElapsedSince(clockStart).TotalSeconds;
            double ips = sec > 0 ? iters / sec : 0.0;
            return new IterLoopBenchmarkResult(
                run,
                batch,
                mine.CudaGraphIter,
                mine.DisablePong,
                loopBatchWall.Count,
                iters,
                sec,
                ips,
                ips > 0.0 ? 1000.0 / ips : double.PositiveInfinity,
                Percentile(loopBatchWall, 0.50),
                Percentile(loopBatchWall, 0.95),
                Percentile(queueTotal, 0.50),
                Percentile(queueTotal, 0.95),
                Percentile(headerClear, 0.50),
                Percentile(headerPtr, 0.50),
                Percentile(deviceClear, 0.50),
                Percentile(iterEnqueue, 0.50),
                Percentile(graphLaunch, 0.50),
                Percentile(syncWait, 0.50),
                Percentile(syncWait, 0.95),
                Percentile(scan, 0.50),
                Percentile(gpuBatch, 0.50),
                Percentile(gpuBatch, 0.95),
                Percentile(hostNonWait, 0.50),
                graphLaunchCount);
        }
        finally
        {
            if (stream0Created)
                try { CudaDriver.StreamSynchronize(stream0); } catch { /* best-effort */ }
            if (stream1Created)
                try { CudaDriver.StreamSynchronize(stream1); } catch { /* best-effort */ }

            FreeHalf(ping);
            FreeHalf(pong);
            try { bState?.Dispose(); } catch { /* best-effort */ }

            if (stream0Created)
                try { CudaDriver.StreamDestroy(stream0); } catch { /* best-effort */ }
            if (stream1Created)
                try { CudaDriver.StreamDestroy(stream1); } catch { /* best-effort */ }
            if (deviceRetained)
                try { CudaDriver.DevicePrimaryCtxRelease(dev); } catch { /* best-effort */ }
        }
    }

    public static BenchmarkResult RunBenchmark(
        int deviceOrdinal,
        MineOptions mine,
        TimeSpan duration,
        ILogger log,
        CancellationToken ct = default)
    {
        if (duration <= TimeSpan.Zero) duration = TimeSpan.FromSeconds(10);
        ValidateCtaTileDivisibility(mine.M, mine.N);

        log.LogInformation(
            "benchmark[{Ord}]: starting {Sec:F0}s hashrate sample",
            deviceOrdinal, duration.TotalSeconds);

        CudaDriver.Check(CudaDriver.Init(0), "cuInit");
        CudaDriver.Check(CudaDriver.DeviceGet(out var dev, deviceOrdinal), "cuDeviceGet");
        // Primary context: ref-counted by the driver, so repeated benchmark
        // calls (e.g. integration tests, or one-shot post-reconnect re-
        // sampling) don't leak contexts. We release in finally.
        CudaDriver.Check(CudaDriver.DevicePrimaryCtxRetain(out var primary, dev),
            "cuDevicePrimaryCtxRetain");
        bool deviceRetained = true;
        CudaDriver.Check(CudaDriver.CtxSetCurrent(primary), "cuCtxSetCurrent");
        CUstream stream0 = default, stream1 = default;
        bool stream0Created = false, stream1Created = false;
        var ping = new WorkerHalf { DeviceId = deviceOrdinal };
        var pong = new WorkerHalf { DeviceId = deviceOrdinal };
        ResidentBStateBuffers? bState = null;

        try
        {
            CudaDriver.Check(CudaDriver.StreamCreate(out stream0, 0), "cuStreamCreate [bench ping]");
            stream0Created = true;
            CudaDriver.Check(CudaDriver.StreamCreate(out stream1, 0), "cuStreamCreate [bench pong]");
            stream1Created = true;
            ping.Stream = stream0;
            pong.Stream = stream1;
            ping.Buffers = new WorkerBuffers(mine.M, mine.N, mine.K, mine.NoiseRank, mine.MatmulsPerPoll);
            pong.Buffers = new WorkerBuffers(mine.M, mine.N, mine.K, mine.NoiseRank, mine.MatmulsPerPoll);
            bState = new ResidentBStateBuffers(mine.N, mine.K, mine.NoiseRank, stream0);

            // ── Synthetic σ install (mirrors production ping/pong path) ──
            var jobKey   = Blake3.Hash(System.Text.Encoding.ASCII.GetBytes("akoya-hashrate-benchmark-v1"));
            var bSeed    = Blake3.Hash(System.Text.Encoding.ASCII.GetBytes("akoya-hashrate-bench-B")); // stable per-miner-style B seed
            var powTarget = new byte[32]; // all zeros → no triggers ever
            // Deterministic SigmaSeed — any non-zero value works.
            ulong sigmaSeed = 0xA1A2_A3A4_A5A6_A7A8ul;
            InstallBenchmarkResidentBState(ping, bState, jobKey, bSeed, sigmaSeed);
            InstallBenchmarkHalf(ping, bState, powTarget, sigmaSeed);
            InstallBenchmarkHalf(pong, bState, powTarget, sigmaSeed);
            CudaDriver.Check(CudaDriver.StreamSynchronize(ping.Stream), "sync bench install [ping]");
            CudaDriver.Check(CudaDriver.StreamSynchronize(pong.Stream), "sync bench install [pong]");
            PrepareGraphIfEnabled(ping, mine.CudaGraphIter, mine.CudaGraphRequired, log, "bench-ping");
            PrepareGraphIfEnabled(pong, mine.CudaGraphIter, mine.CudaGraphRequired, log, "bench-pong");

            int batch = mine.MatmulsPerPoll;

            // Warm-up both streams once so timed loop reflects steady-state ping/pong overlap.
            ulong nextSeed = 0;
            QueueBatch(ping, batch, nextSeed);
            QueueBatch(pong, batch, nextSeed + (ulong)batch);
            SyncAndScan(ping, batch);
            nextSeed += (ulong)batch;
            (ping, pong) = (pong, ping);
            SyncAndScan(ping, batch);
            nextSeed += (ulong)batch;

            // Prime timed loop with one in-flight batch on Ping.
            QueueBatch(ping, batch, nextSeed);

            var clock = Stopwatch.StartNew();
            ulong iters = 0;
            while (clock.Elapsed < duration && !ct.IsCancellationRequested)
            {
                QueueBatch(pong, batch, nextSeed + (ulong)batch);
                SyncAndScan(ping, batch);
                iters += (ulong)batch;
                nextSeed += (ulong)batch;
                if (clock.Elapsed >= duration || ct.IsCancellationRequested) break;
                (ping, pong) = (pong, ping);
            }
            clock.Stop();

            double sec   = clock.Elapsed.TotalSeconds;
            double ips   = iters / sec;
            // Match the per-worker stats log: count proof-pattern tiles
            // (rows×cols) so hashes/s == tmads/s × 1e12 (the wire unit).
            // mine.K is the protocol's CommonDim — same value used to
            // build MiningConfiguration during real mining.
            var benchCfg = MiningConfiguration.Default(commonDim: (uint)mine.K, rank: (ushort)mine.NoiseRank);
            long tilesPerIter = TilesPerIter(mine.M, mine.N, benchCfg);
            double tilesPerSec = ips * tilesPerIter;
            // TMADs/sec: each iter does one M×N×K multiply-add chain.
            double tmads = ((double)mine.M * mine.N * mine.K * iters) / sec / 1e12;
            double hashesPerSec = tilesPerSec * benchCfg.DifficultyAdjustmentFactor();

            log.LogInformation(
                "benchmark[{Ord}]: {Iters} iters in {Sec:F2}s → {Ips:F2} iters/s, {Tmads:F2} TMADs/s, {Tps:E2} tiles/s, hashes/s={Hps}",
                deviceOrdinal, iters, sec, ips, tmads, tilesPerSec, FormatHashRate(hashesPerSec));

            return new BenchmarkResult(tmads, ips, iters, sec, tilesPerIter, tilesPerSec);
        }
        finally
        {
            // Drain + free while context is still current.
            if (stream0Created)
                try { CudaDriver.StreamSynchronize(stream0); } catch { /* best-effort */ }
            if (stream1Created)
                try { CudaDriver.StreamSynchronize(stream1); } catch { /* best-effort */ }

            FreeHalf(ping);
            FreeHalf(pong);
            try { bState?.Dispose(); } catch { /* best-effort */ }

            if (stream0Created)
                try { CudaDriver.StreamDestroy(stream0); } catch { /* best-effort */ }
            if (stream1Created)
                try { CudaDriver.StreamDestroy(stream1); } catch { /* best-effort */ }
            if (deviceRetained)
            {
                try { CudaDriver.DevicePrimaryCtxRelease(dev); } catch { /* best-effort */ }
            }
        }
    }

    internal static long TilesPerIter(MineOptions mine)
        => TilesPerIter(mine.M, mine.N, defaultCfg: null);

    internal static long TilesPerIter(int m, int n)
        => TilesPerIter(m, n, defaultCfg: null);

    internal static long TilesPerIter(int m, int n, MiningConfiguration? defaultCfg)
    {
        ValidateCtaTileDivisibility(m, n);
        var cfg = defaultCfg ?? MiningConfiguration.Default(commonDim: 2048, rank: 128);
        int patternH = (int)cfg.RowsPattern.Size;
        int patternW = (int)cfg.ColsPattern.Size;
        return ((long)m / patternH) * ((long)n / patternW);
    }

    internal static double ExpectedOpensPerSecond(
        double tilesPerSec,
        uint targetNbits,
        MiningConfiguration miningConfig)
    {
        if (!double.IsFinite(tilesPerSec) || tilesPerSec <= 0.0)
            return 0.0;

        var adjustedTarget = SigmaContext.NbitsToTarget(targetNbits)
                             * miningConfig.DifficultyAdjustmentFactor();
        if (adjustedTarget <= BigInteger.Zero)
            return 0.0;
        if (adjustedTarget >= TargetSpace)
            return tilesPerSec;

        return tilesPerSec * ((double)adjustedTarget / TargetSpaceAsDouble);
    }

    private static void ValidateCtaTileDivisibility(int m, int n)
    {
        if (m % BM != 0)
            throw new ArgumentException($"M={m} must be divisible by CTA tile M={BM}", nameof(m));
        if (n % BN != 0)
            throw new ArgumentException($"N={n} must be divisible by CTA tile N={BN}", nameof(n));
    }

    // Per-benchmark σ install for one half. Mirrors production InstallSigmaHalf:
    // Ping computes full B-side once; Pong shares Ping's B-side pointers.
    private static double MeasureMs(Action action)
    {
        var sw = Stopwatch.StartNew();
        action();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds;
    }

    private static double MeasureStreamStageMs(CUstream stream, Action enqueue)
    {
        var sw = Stopwatch.StartNew();
        enqueue();
        CudaDriver.Check(CudaDriver.StreamSynchronize(stream), "sync measured stage");
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds;
    }

    private static void FreeBenchmarkWorkspace(WorkerHalf half)
    {
        if (half.Workspace == IntPtr.Zero) return;
        Check("workspace_free (sigma bench rotate)",
            PearlGemm.PearlGemmNative.WorkspaceFree(half.Workspace, half.Stream.Handle));
        half.Workspace = IntPtr.Zero;
        half.GraphReady = false;
    }

    private static void InstallBenchmarkWorkspaceParams(
        WorkerHalf half,
        ulong sigmaSeed,
        WorkerBuffers? bSideSource = null)
    {
        var b = half.Buffers;
        var bs = bSideSource ?? b;
        unsafe
        {
            var wp = new PearlGemm.PearlGemmNative.WorkspaceParams
            {
                M = b.M, N = b.N, K = b.K, R = b.R,
                BM = BM, BN = BN, BK = BK, CM = CM, CN = CN,
                ThNumBlocks = NumBlocks((long)b.M * b.K),
                ThThreads = TENSOR_HASH_THREADS,
                ThStages = TENSOR_HASH_STAGES,
                ThLeaves = TENSOR_HASH_LEAVES,
                SigmaSeed = sigmaSeed,

                A = b.A.Handle,
                AHash = b.AHash.Handle,
                Roots = b.Roots.Handle,
                CommitA = b.CommitA.Handle,
                CommitB = b.CommitB.Handle,
                EAL = b.EAL.Handle,
                EAL_fp16 = b.EALFp16.Handle,
                EAR_R_major = b.EAR_R.Handle,
                EAR_K_major = b.EAR_K.Handle,
                AxEBL_fp16 = b.AxEBLFp16.Handle,
                ApEA = b.ApEA.Handle,
                A_scales = b.AScales.Handle,
                C = b.C.Handle,
                HostSignalSync = b.Sync.Handle,
                PowTarget = b.PowTarget.Handle,
                PowKey = b.CommitA.Handle,

                B = bs.B.Handle,
                BHash = bs.BHash.Handle,
                Key = bs.Key.Handle,
                EBR = bs.EBR.Handle,
                EBR_fp16 = bs.EBRFp16.Handle,
                EBL_R_major = bs.EBL_R.Handle,
                EBL_K_major = bs.EBL_K.Handle,
                EARxBpEB_fp16 = bs.EARxBpEB.Handle,
                BpEB = bs.BpEB.Handle,
                B_scales = bs.BScales.Handle,
            };
            Check("workspace_install_params",
                PearlGemm.PearlGemmNative.WorkspaceInstallParams(half.Workspace, &wp));
        }
    }

    private static void InstallBenchmarkWorkspaceParams(
        WorkerHalf half,
        ulong sigmaSeed,
        ResidentBStateBuffers bState)
    {
        var b = half.Buffers;
        unsafe
        {
            var wp = new PearlGemm.PearlGemmNative.WorkspaceParams
            {
                M = b.M, N = b.N, K = b.K, R = b.R,
                BM = BM, BN = BN, BK = BK, CM = CM, CN = CN,
                ThNumBlocks = NumBlocks((long)b.M * b.K),
                ThThreads = TENSOR_HASH_THREADS,
                ThStages = TENSOR_HASH_STAGES,
                ThLeaves = TENSOR_HASH_LEAVES,
                SigmaSeed = sigmaSeed,

                A = b.A.Handle,
                AHash = b.AHash.Handle,
                Roots = b.Roots.Handle,
                CommitA = b.CommitA.Handle,
                CommitB = b.CommitB.Handle,
                EAL = b.EAL.Handle,
                EAL_fp16 = b.EALFp16.Handle,
                EAR_R_major = b.EAR_R.Handle,
                EAR_K_major = b.EAR_K.Handle,
                AxEBL_fp16 = b.AxEBLFp16.Handle,
                ApEA = b.ApEA.Handle,
                A_scales = b.AScales.Handle,
                C = b.C.Handle,
                HostSignalSync = b.Sync.Handle,
                PowTarget = b.PowTarget.Handle,
                PowKey = b.CommitA.Handle,

                B = bState.B.Handle,
                BHash = bState.BHash.Handle,
                Key = bState.Key.Handle,
                EBR = bState.EBR.Handle,
                EBR_fp16 = bState.EBRFp16.Handle,
                EBL_R_major = bState.EBL_R.Handle,
                EBL_K_major = bState.EBL_K.Handle,
                EARxBpEB_fp16 = bState.EARxBpEB.Handle,
                BpEB = bState.BpEB.Handle,
                B_scales = bState.BScales.Handle,
            };
            Check("workspace_install_params",
                PearlGemm.PearlGemmNative.WorkspaceInstallParams(half.Workspace, &wp));
        }
    }

    private static void InstallBenchmarkResidentBState(
        WorkerHalf scratchHalf,
        ResidentBStateBuffers bState,
        byte[] jobKey,
        byte[] bSeed,
        ulong sigmaSeed)
    {
        var a = scratchHalf.Buffers;
        long bA = (long)a.M * a.K;

        H2D(bState.Key, jobKey);
        Check("lcg_int7 A throwaway", PearlGemm.PearlGemmNative.LcgInt7Fill(
            a.A.Handle, bA, THROWAWAY_A_SEED_LO, sigmaSeed, scratchHalf.Stream.Handle));
        Check("tensor_hash A seed", PearlGemm.PearlGemmNative.TensorHash(
            a.A.Handle, (uint)bA, a.AHash.Handle, bState.Key.Handle,
            NumBlocks(bA), TENSOR_HASH_THREADS, TENSOR_HASH_STAGES,
            TENSOR_HASH_LEAVES, a.Roots.Handle, scratchHalf.DeviceId, scratchHalf.Stream.Handle));

        bool expandB = !bState.BUploaded;
        InstallNativeBProcess(expandB ? bSeed : null, expandB, a, bState, scratchHalf.DeviceId, scratchHalf.Stream);
        if (expandB)
        {
            bState.BUploaded = true;
        }
    }

    private static void InstallBenchmarkHalf(
        WorkerHalf half,
        ResidentBStateBuffers bState,
        byte[] powTarget,
        ulong sigmaSeed)
    {
        var b = half.Buffers;
        half.GraphReady = false;

        H2D(b.PowTarget, powTarget);

        if (half.Workspace == IntPtr.Zero)
        {
            unsafe
            {
                nint ws = IntPtr.Zero;
                Check("workspace_alloc", PearlGemm.PearlGemmNative.WorkspaceAlloc(
                    b.M, b.N, b.K, b.R,
                    withNoiseA: 1, withNoiseB: 0,
                    outWorkspace: &ws, half.Stream.Handle));
                half.Workspace = ws;
            }
        }

        InstallBenchmarkWorkspaceParams(half, sigmaSeed, bState);
    }

    private static void InstallBenchmarkHalf(
        WorkerHalf half, byte[] jobKey, byte[] bBytes, byte[] powTarget, ulong sigmaSeed,
        WorkerBuffers? bSideSource = null)
    {
        var b = half.Buffers;
        long bA = (long)b.M * b.K, bB = (long)b.N * b.K;
        bool isPrimary = bSideSource is null;

        if (half.Workspace != IntPtr.Zero)
        {
            Check("workspace_free (bench rotate)",
                PearlGemm.PearlGemmNative.WorkspaceFree(half.Workspace, half.Stream.Handle));
            half.Workspace = IntPtr.Zero;
            half.GraphReady = false;
        }

        H2D(b.PowTarget, powTarget);

        if (isPrimary)
        {
            H2D(b.Key, jobKey);
            if (!half.BUploaded)
            {
                H2D(b.B, bBytes);
                half.BUploaded = true;
            }

            Check("tensor_hash B", PearlGemm.PearlGemmNative.TensorHash(
                b.B.Handle, (uint)bB, b.BHash.Handle, b.Key.Handle,
                NumBlocks(bB), TENSOR_HASH_THREADS, TENSOR_HASH_STAGES,
                TENSOR_HASH_LEAVES, b.Roots.Handle, half.DeviceId, half.Stream.Handle));

            Check("lcg_int7 A throwaway", PearlGemm.PearlGemmNative.LcgInt7Fill(
                b.A.Handle, bA, THROWAWAY_A_SEED_LO, sigmaSeed, half.Stream.Handle));
            Check("tensor_hash A seed", PearlGemm.PearlGemmNative.TensorHash(
                b.A.Handle, (uint)bA, b.AHash.Handle, b.Key.Handle,
                NumBlocks(bA), TENSOR_HASH_THREADS, TENSOR_HASH_STAGES,
                TENSOR_HASH_LEAVES, b.Roots.Handle, half.DeviceId, half.Stream.Handle));
            Check("commit_seed", PearlGemm.PearlGemmNative.CommitmentHashFromMerkleRoots(
                b.AHash.Handle, b.BHash.Handle, b.Key.Handle,
                b.CommitA.Handle, b.CommitB.Handle, half.DeviceId, half.Stream.Handle));
            Check("noise_gen full", PearlGemm.PearlGemmNative.NoiseGen(
                b.R, b.M, b.N, b.K,
                b.EAL.Handle, b.EALFp16.Handle,
                b.EAR_R.Handle, b.EAR_K.Handle,
                b.EBL_R.Handle, b.EBL_K.Handle,
                b.EBR.Handle, b.EBRFp16.Handle,
                b.CommitA.Handle, b.CommitB.Handle,
                half.Stream.Handle));

            unsafe
            {
                nint ws = IntPtr.Zero;
                Check("workspace_alloc", PearlGemm.PearlGemmNative.WorkspaceAlloc(
                    b.M, b.N, b.K, b.R,
                    withNoiseA: 1, withNoiseB: 1,
                    outWorkspace: &ws, half.Stream.Handle));
                half.Workspace = ws;

                var nb = new PearlGemm.PearlGemmNative.NoiseBParams
                {
                    N           = b.N,
                    K           = b.K,
                    R           = b.R,
                    B           = b.B.Handle,
                    EAR_K_major = b.EAR_K.Handle,
                    EBL_R_major = b.EBL_R.Handle,
                    EBR         = b.EBR.Handle,
                    EARxBpEB    = b.EARxBpEB.Handle,
                    BpEB        = b.BpEB.Handle,
                    Workspace   = half.Workspace,
                };
                Check("noise_B", PearlGemm.PearlGemmNative.NoiseB(&nb, half.Stream.Handle));
            }
        }
        else
        {
            half.BUploaded = true;
            unsafe
            {
                nint ws = IntPtr.Zero;
                Check("workspace_alloc", PearlGemm.PearlGemmNative.WorkspaceAlloc(
                    b.M, b.N, b.K, b.R,
                    withNoiseA: 1, withNoiseB: 0,
                    outWorkspace: &ws, half.Stream.Handle));
                half.Workspace = ws;
            }
        }

        var bs = bSideSource ?? b;
        unsafe
        {
            var wp = new PearlGemm.PearlGemmNative.WorkspaceParams
            {
                M = b.M, N = b.N, K = b.K, R = b.R,
                BM = BM, BN = BN, BK = BK, CM = CM, CN = CN,
                ThNumBlocks = NumBlocks((long)b.M * b.K),
                ThThreads   = TENSOR_HASH_THREADS,
                ThStages    = TENSOR_HASH_STAGES,
                ThLeaves    = TENSOR_HASH_LEAVES,
                SigmaSeed   = sigmaSeed,

                A           = b.A.Handle,
                AHash       = b.AHash.Handle,
                Roots       = b.Roots.Handle,
                CommitA     = b.CommitA.Handle,
                CommitB     = b.CommitB.Handle,
                EAL         = b.EAL.Handle,
                EAL_fp16    = b.EALFp16.Handle,
                EAR_R_major = b.EAR_R.Handle,
                EAR_K_major = b.EAR_K.Handle,
                AxEBL_fp16  = b.AxEBLFp16.Handle,
                ApEA        = b.ApEA.Handle,
                A_scales    = b.AScales.Handle,
                C           = b.C.Handle,
                HostSignalSync = b.Sync.Handle,
                PowTarget   = b.PowTarget.Handle,
                PowKey      = b.CommitA.Handle,

                B           = bs.B.Handle,
                BHash       = bs.BHash.Handle,
                Key         = bs.Key.Handle,
                EBR         = bs.EBR.Handle,
                EBR_fp16    = bs.EBRFp16.Handle,
                EBL_R_major = bs.EBL_R.Handle,
                EBL_K_major = bs.EBL_K.Handle,
                EARxBpEB_fp16 = bs.EARxBpEB.Handle,
                BpEB        = bs.BpEB.Handle,
                B_scales    = bs.BScales.Handle,
            };
            Check("workspace_install_params",
                PearlGemm.PearlGemmNative.WorkspaceInstallParams(half.Workspace, &wp));
        }
    }
}
