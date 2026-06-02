using System.Diagnostics;
using System.Threading.Channels;
using Akoya.Mining;
using Akoya.Miner.Observability;
using Microsoft.Extensions.Logging;
using PearlPool.Proto.V2;

namespace Akoya.Miner.Mining;

/// <summary>
/// Immutable snapshot passed from the mining thread to the finalize task.
///
/// Owned-bytes fields (Sigma, ConfigBytes, JobKey, ABytes/ASlice/ALeafCvs,
/// BProof, BSeed, indices …) cross the thread boundary safely.
///
/// <see cref="BMerkleTree"/> is an <see cref="Akoya.Mining.IMerkleTreeHandle"/> reference
/// shared with <c>GpuWorker</c>. The hot-path acquires a refcount before
/// enqueueing this payload, and <c>ShareFinalizer.Finalize</c> releases
/// it after the audit-proof opening — so the proof handle survives any
/// concurrent σ rotation that may Dispose the worker's reference.
/// </summary>
internal sealed record SharePayload(
    int             GpuIndex,
    byte[]          Sigma,
    byte[]          ConfigBytes,
    byte[]          JobKey,
    byte[]?         ABytes,
    byte[]?         ASlice,
    byte[]?         ALeafCvs,
    Akoya.Mining.MerkleRootAndProofResult BProof,
    Akoya.Mining.IMerkleTreeHandle BMerkleTree,
    byte[]          BSeed,
    uint            AuditK,
    uint[]          ARowIndices,
    uint[]          BColIndices,
    int             TileRow,
    int             TileCol,
    int             M,
    int             N,
    int             K,
    int             R,
    uint            ClaimedDifficultyNbits,
    Akoya.Crypto.MiningConfiguration MiningConfig,
    double          MsTotalSoFar,
    double          MsDrainWait,
    double          MsLcgKernel,
    double          MsD2H);

internal sealed class ShareFinalizer : IAsyncDisposable
{
#if PERF_TIMINGS
    private const bool RuntimePerfTimings = true;
#else
    private const bool RuntimePerfTimings = false;
#endif

    private readonly IShareSink _sink;
    private readonly ILogger _log;
    private readonly Channel<SharePayload> _channel;
    private readonly Task _consumer;
    private readonly CancellationTokenSource _cts = new();
    private long _droppedTotal;

    public ShareFinalizer(IShareSink sink, ILogger log)
    {
        _sink = sink;
        _log = log;
        _channel = Channel.CreateBounded<SharePayload>(new BoundedChannelOptions(4)
        {
            FullMode = BoundedChannelFullMode.DropWrite,
            SingleReader = true,
            SingleWriter = false,
        });
        _consumer = Task.Run(ConsumeLoopAsync);
    }

    /// <summary>
    /// Non-blocking enqueue. Returns <c>false</c> if the channel was full
    /// (caller should treat as a dropped share and log). On drop we release
    /// the BMerkleTree refcount the caller acquired pre-enqueue so the
    /// tree doesn't leak its native allocation.
    /// </summary>
    public bool TryEnqueue(SharePayload payload)
    {
        if (_channel.Writer.TryWrite(payload)) return true;
        try { payload.BMerkleTree.Release(); } catch { /* best-effort */ }
        var n = Interlocked.Increment(ref _droppedTotal);
        _log.LogWarning(
            "worker[{Gpu}]: ShareFinalizer queue full — share dropped (total dropped={N})",
            payload.GpuIndex, n);
        return false;
    }

    private async Task ConsumeLoopAsync()
    {
        try
        {
            await foreach (var p in _channel.Reader.ReadAllAsync(_cts.Token).ConfigureAwait(false))
            {
                try
                {
                    Finalize(p);
                }
                catch (Exception ex)
                {
                    _log.LogError(ex,
                        "worker[{Gpu}]: share-finalize threw — share dropped",
                        p.GpuIndex);
                }
            }
        }
        catch (OperationCanceledException) { /* shutdown */ }
    }

    private void Finalize(SharePayload p)
    {
        ShareSubmission share;
        ShareBuilder.Timings buildTimings;
        double msBuild;
        long buildStart = TimingStart();
        try
        {
            if (p.ASlice is not null && p.ALeafCvs is not null)
            {
                share = ShareBuilder.Build(
                    sigma:                  p.Sigma,
                    configBytes:            p.ConfigBytes,
                    jobKey:                 p.JobKey,
                    aSlice:                 p.ASlice,
                    aLeafCvs:               p.ALeafCvs,
                    bProof:                 p.BProof,
                    bMerkleTree:            p.BMerkleTree,
                    bSeed:                  p.BSeed,
                    auditK:                 p.AuditK,
                    aRowIndices:            p.ARowIndices,
                    bColIndices:            p.BColIndices,
                    tileRow:                p.TileRow,
                    tileCol:                p.TileCol,
                    m:                      p.M,
                    n:                      p.N,
                    k:                      p.K,
                    r:                      p.R,
                    claimedDifficultyNbits: p.ClaimedDifficultyNbits,
                    timings:                out buildTimings,
                    collectTimings:         RuntimePerfTimings);
            }
            else if (p.ABytes is not null)
            {
                share = ShareBuilder.Build(
                    sigma:                  p.Sigma,
                    configBytes:            p.ConfigBytes,
                    jobKey:                 p.JobKey,
                    aBytes:                 p.ABytes,
                    bProof:                 p.BProof,
                    bMerkleTree:            p.BMerkleTree,
                    bSeed:                  p.BSeed,
                    auditK:                 p.AuditK,
                    aRowIndices:            p.ARowIndices,
                    bColIndices:            p.BColIndices,
                    tileRow:                p.TileRow,
                    tileCol:                p.TileCol,
                    m:                      p.M,
                    n:                      p.N,
                    k:                      p.K,
                    r:                      p.R,
                    claimedDifficultyNbits: p.ClaimedDifficultyNbits,
                    timings:                out buildTimings,
                    collectTimings:         RuntimePerfTimings);
            }
            else
            {
                throw new InvalidOperationException("Share payload contained neither ABytes nor ASlice+ALeafCvs.");
            }
            msBuild = TimingElapsedMs(buildStart);
        }
        finally
        {
            // Release the refcount acquired pre-enqueue. After this point
            // the mining thread is free to retire the corresponding σ.
            try { p.BMerkleTree.Release(); } catch { /* best-effort */ }
        }

        if (!ShareTargetGuard.ClearsLiveTarget(
                share.ClaimedHash.Span, p.ClaimedDifficultyNbits, p.MiningConfig))
        {
            // Diagnostic surfacing: the docstring on ShareTargetGuard once
            // explained this skip as a "vardiff race tightened the target",
            // but in practice (see docs and the σ=ACFB0748 incident) the
            // skip can come from a CPU↔GPU jackpot disagreement that has
            // NOTHING to do with target movement. Log enough hex to tell
            // those two cases apart on the next occurrence.
            string claimedHex = Convert.ToHexString(
                share.ClaimedHash.Span.Slice(0, Math.Min(8, share.ClaimedHash.Length)));
            string sigmaHex = Convert.ToHexString(
                p.Sigma.AsSpan(0, Math.Min(8, p.Sigma.Length)));
            string jobKeyHex = Convert.ToHexString(
                p.JobKey.AsSpan(0, Math.Min(8, p.JobKey.Length)));
            _log.LogWarning(
                "worker[{Gpu}]: share skipped — claimedHash > liveTarget (nbits=0x{Nbits:X8} DAF={Daf} claimedHash[0..8]={Claimed} sigma[0..8]={Sigma} jobKey[0..8]={Job} tile=({Tr},{Tc}))",
                p.GpuIndex, p.ClaimedDifficultyNbits, p.MiningConfig.DifficultyAdjustmentFactor(),
                claimedHex, sigmaHex, jobKeyHex, p.TileRow, p.TileCol);
            return;
        }

        long queueStart = TimingStart();
        _sink.SubmitAsync(share, _cts.Token).AsTask().GetAwaiter().GetResult();
        double msQueueWait = TimingElapsedMs(queueStart);

        // NOTE: Metrics.IncShareAccepted is NOT called here despite the
        // previous "✓ share submitted" name suggesting otherwise. That
        // increment happens in WorkerOrchestrator.OnShareResult, keyed on
        // the pool's actual verdict — which is the only source of truth for
        // share counts. Incrementing here AND there double-counted accepted
        // shares (and over-counted by anything that never reached the wire
        // due to e.g. a stream watchdog trip).
        // NOTE: This is a queue-handoff log, not a wire-send confirmation.
        // SubmitAsync above just writes to MiningSession._outbound; the actual
        // wire send happens in MiningSession.OutboundWriterLoop, which emits
        // "✓ share on wire" once WriteAsync to gRPC returns. The final
        // pool-ACK confirmation is the "share-result" Info line in
        // WorkerOrchestrator.OnShareResult. Demoted to Debug so an operator
        // grepping `✓ share` sees only events that actually made the wire.
#if PERF_TIMINGS
        double msTotal = p.MsTotalSoFar + msBuild + msQueueWait;
        _log.LogDebug(
            "worker[{Gpu}]: share queued tile=({Tr},{Tc}) sigma={SigmaPrefix} [total={Tot:F1}ms build={Build:F1} queueWait={Q:F1}]",
            p.GpuIndex, p.TileRow, p.TileCol,
            Convert.ToHexString(p.Sigma.AsSpan(0, Math.Min(8, p.Sigma.Length))),
            msTotal, msBuild, msQueueWait);
        if (_log.IsEnabled(LogLevel.Debug))
        {
            _log.LogDebug(
                "worker[{Gpu}]: share timings tile=({Tr},{Tc}) drainWait={Dw:F1} lcg={Lcg:F1} d2h={D2H:F1} build={Build:F1}(slice={Sl:F1} aMerk={Am:F1} bMerk={Bm:F1} noise={No:F1} jack={Jk:F1} hash={Hs:F1} audit={Au:F1} pack={Pk:F1}) queueWait={Q:F1}",
                p.GpuIndex, p.TileRow, p.TileCol,
                p.MsDrainWait, p.MsLcgKernel, p.MsD2H, msBuild,
                buildTimings.SliceMs, buildTimings.AMerkleMs, buildTimings.BMerkleMs,
                buildTimings.NoiseMs, buildTimings.JackpotMs, buildTimings.HashMs, buildTimings.AuditMs, buildTimings.PackMs,
                msQueueWait);
        }
#endif
    }

    public async ValueTask DisposeAsync()
    {
        // Stop accepting new payloads; let the consumer drain the queue
        // (in-flight share submits should complete). Bound the drain to
        // avoid hanging shutdown on a wedged pool — after 5s we cancel.
        _channel.Writer.TryComplete();
        var drain = Task.WhenAny(_consumer, Task.Delay(TimeSpan.FromSeconds(5)));
        var winner = await drain.ConfigureAwait(false);
        if (winner != _consumer)
        {
            _log.LogWarning("ShareFinalizer: drain timed out after 5s — cancelling in-flight submit");
            try { _cts.Cancel(); } catch (ObjectDisposedException) { /* fine */ }
            try { await _consumer.ConfigureAwait(false); }
            catch (OperationCanceledException) { /* expected */ }
        }
        // Anything still sitting in the channel after the consumer has
        // stopped reading is now orphaned — release its tree refcounts
        // so the native handles are freed promptly instead of waiting
        // for the GC finalizer.
        while (_channel.Reader.TryRead(out var orphan))
        {
            try { orphan.BMerkleTree.Release(); } catch { /* best-effort */ }
        }
        _cts.Dispose();
    }

    private static double ElapsedMsSince(long startTimestamp)
        => (Stopwatch.GetTimestamp() - startTimestamp) * 1000.0 / Stopwatch.Frequency;

    private static long TimingStart()
        => RuntimePerfTimings ? Stopwatch.GetTimestamp() : 0;

    private static double TimingElapsedMs(long startTimestamp)
        => RuntimePerfTimings ? ElapsedMsSince(startTimestamp) : 0.0;
}
