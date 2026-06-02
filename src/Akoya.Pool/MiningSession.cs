// Owns the V2 logical session: Register/Resume orchestration, the bidi
// MiningStream, and an outbound queue that any thread can enqueue events to.
//
// State machine (per process lifetime, many gRPC streams):
//
//   Connecting ── Register or Resume ──▶ Authenticated ── MiningStream open ──▶ Streaming
//        ▲                                                                         │
//        └──── stream error / ReconnectHint(wait) / fatal PoolError ◀──────────────┘
//
// Threading model:
//   * Single producer/consumer model for the outbound stream. Callers enqueue
//     MinerEvents via TrySendAsync; a dedicated writer task drains the channel
//     into the stream. This is required because IClientStreamWriter is
//     single-threaded by contract.
//   * Inbound PoolEvents are dispatched to whatever handler the caller wires
//     up via subscribe-style callbacks (OnJob, OnShareResult, OnVardiff,
//     OnPong, OnError, OnReconnect).
//   * seq numbers are assigned by SequenceCounter at enqueue time so the
//     consumer order on the wire matches the producer's logical order.

using System.Threading.Channels;
using Grpc.Core;
using Microsoft.Extensions.Logging;
using PearlPool.Proto.V2;

namespace Akoya.Pool;

/// <summary>
/// Callbacks invoked on PoolEvent receipt. All run on the inbound reader
/// task — do not block; hand work off to other tasks/channels.
/// </summary>
public sealed class MiningSessionCallbacks
{
    public Func<JobAssignment, ValueTask>?   OnJob          { get; init; }
    public Func<ShareResult, ValueTask>?     OnShareResult  { get; init; }
    public Func<DifficultyAdjust, ValueTask>? OnVardiff     { get; init; }
    public Func<PongEvent, ValueTask>?       OnPong         { get; init; }
    public Func<PoolError, ValueTask>?       OnError        { get; init; }
    public Func<ReconnectHint, ValueTask>?   OnReconnect    { get; init; }
}

/// <summary>Carries everything needed to call Register or Resume.</summary>
public sealed record SessionIdentity(
    string WalletAddress,
    string WorkerName,
    string MinerVersion,
    string GitSha,
    uint K,
    double ClaimedTotalHashrate,
    IReadOnlyList<GpuCard> GpuCards);

public sealed class MiningSession : IAsyncDisposable, IEventSink
{
    private readonly PoolConnection _pool;
    private readonly SessionStore _store;
    private readonly SequenceCounter _seq = new();
    private readonly ILogger<MiningSession> _log;

    // Bounded so a runaway producer (e.g. a misbehaving Heartbeat pump) can't
    // OOM us. 1024 is comfortably more than peak rate: shares ≤ ~few/s,
    // heartbeats every 30s, pings every 15s.
    private readonly Channel<MinerEvent> _outbound =
        Channel.CreateBounded<MinerEvent>(new BoundedChannelOptions(1024)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = false,
        });

    private AsyncDuplexStreamingCall<MinerEvent, PoolEvent>? _stream;
    private byte[]? _minerId;     // 16 B
    private string? _sessionToken;
    private string? _identityKey;

    // Stream-liveness watchdog: monotonic TickCount64 of last inbound frame.
    // Volatile-written from the reader, Volatile-read from the watchdog loop.
    private long _lastInboundTicks;

    // Set by StreamWatchdogLoop when it cancels the linked CTS. Checked in
    // RunStreamAsync's finally so we surface a StreamIdleException instead
    // of a silent clean return. Critical for the outer reconnect loop —
    // a silently-returning watchdog trip would skip exp backoff and let
    // us hammer a degrading gateway with Register attempts every ~90s.
    private bool _watchdogTripped;
    private string? _watchdogTripReason;

    // Layer 2 — per-ping pong-deadline tracking. Pings are not protocol-matched
    // (any Pong satisfies the oldest unanswered Ping, FIFO). We track the
    // TickCount64 of the OLDEST unanswered ping so a stalled application
    // layer can be detected long before the inbound-idle watchdog trips.
    //   * MarkPingSent — called from OutboundWriterLoop after a Ping has been
    //     handed to the wire. Sets _oldestPendingPingTicks if currently 0.
    //   * MarkPongReceived — called from InboundReaderLoop on PongEvent. Pops
    //     one pending entry: if count reaches 0 we clear the timestamp;
    //     otherwise we refresh it to "now" (we don't know the exact second-
    //     oldest, but it's guaranteed not older than now).
    // Default deadline 20s = ~1.3× ping interval. Override via
    // AKOYA_POOL_PONG_TIMEOUT_SEC; 0 disables this check.
    private long _oldestPendingPingTicks;
    private int  _pendingPingCount;

    public MiningSession(PoolConnection pool, SessionStore store, ILogger<MiningSession> log)
    {
        _pool = pool;
        _store = store;
        _log = log;
    }

    public ReadOnlyMemory<byte> MinerId => _minerId is null ? ReadOnlyMemory<byte>.Empty : _minerId;
    public string? SessionToken => _sessionToken;
    public SequenceCounter Sequence => _seq;

    /// <summary>
    /// Establish a logical session, preferring Resume if a persisted token
    /// exists. Falls back to Register on Resume-failure. Throws on
    /// unrecoverable error (network down, invalid wallet, etc.) — the caller
    /// decides whether to back off and retry.
    ///
    /// On success the returned ResumeResponse carries the initial job
    /// (Resume path), OR null (Register path — first JobAssignment arrives
    /// over the MiningStream).
    /// </summary>
    public async Task<ResumeResponse?> ConnectAsync(SessionIdentity identity, CancellationToken ct)
    {
        var stored = _store.TryLoad(_pool.Endpoint);
        ResumeResponse? initialJob = null;

        if (stored is not null)
        {
            var resumeResp = await TryResumeAsync(stored, ct).ConfigureAwait(false);
            if (resumeResp is { Success: true })
            {
                _minerId      = HexToBytes(stored.MinerIdHex);
                _sessionToken = resumeResp.SessionTokenRefreshed;
                _identityKey  = stored.IdentityKey;
                Persist(stored.MinerIdHex);
                initialJob = resumeResp;
            }
            else
            {
                _log.LogInformation(
                    "session: Resume failed ({Reason}) — falling back to Register",
                    resumeResp?.ErrorMessage ?? "no response");
                _store.Delete();
            }
        }

        if (_sessionToken is null)
        {
            await RegisterAsync(identity, stored?.IdentityKey, ct).ConfigureAwait(false);
        }

        return initialJob;
    }

    /// <summary>Unary deadline for Register / Resume. The pool accepting the
    /// TCP/TLS handshake but black-holing the gRPC call would otherwise hang
    /// us indefinitely — we'd never reach the bidi stream and the stream
    /// watchdog wouldn't be running yet. 30s is comfortably above measured
    /// p99 for both RPCs (sub-200ms) and well below the outer reconnect
    /// backoff ceiling.</summary>
    public static readonly TimeSpan RegisterResumeDeadline = TimeSpan.FromSeconds(30);

    private async Task<ResumeResponse?> TryResumeAsync(SessionRecord stored, CancellationToken ct)
    {
        try
        {
            var req = new ResumeRequest
            {
                MinerId      = Google.Protobuf.ByteString.CopyFrom(HexToBytes(stored.MinerIdHex)),
                SessionToken = stored.SessionToken,
            };
            return await _pool.Client.ResumeAsync(
                req,
                deadline: DateTime.UtcNow + RegisterResumeDeadline,
                cancellationToken: ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested) { throw; }
        catch (RpcException e)
        {
            _log.LogInformation("session: Resume RPC failed ({Status}): {Detail}", e.StatusCode, e.Status.Detail);
            return null;
        }
        catch (OperationCanceledException oce)
        {
            // gRPC deadline expired without the channel ever becoming ready.
            // Treat as "no usable session to resume" — Connect() will fall
            // back to Register, which will re-translate its own version of
            // this into a PoolUnreachableException for the reconnect loop.
            _log.LogInformation(
                "session: Resume timed out after {Sec:F0}s (channel never ready): {Msg}",
                RegisterResumeDeadline.TotalSeconds, oce.Message);
            return null;
        }
    }

    private async Task RegisterAsync(SessionIdentity id, string? reclaimIdentityKey, CancellationToken ct)
    {
        var req = new RegisterRequest
        {
            WalletAddress         = id.WalletAddress,
            WorkerName            = id.WorkerName,
            MinerVersion          = id.MinerVersion,
            GitSha                = id.GitSha,
            ProtocolVersion       = 2,
            K                     = id.K,
            ClaimedTotalHashrate  = id.ClaimedTotalHashrate,
        };
        if (!string.IsNullOrEmpty(reclaimIdentityKey)) req.IdentityKey = reclaimIdentityKey;
        foreach (var g in id.GpuCards) req.GpuCards.Add(g);

        _log.LogInformation("session: Register wallet={Wallet} worker={Worker} gpus={N}",
            id.WalletAddress, id.WorkerName, id.GpuCards.Count);

        RegisterResponse resp;
        try
        {
            resp = await _pool.Client.RegisterAsync(
                req,
                deadline: DateTime.UtcNow + RegisterResumeDeadline,
                cancellationToken: ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested) { throw; }
        catch (RpcException e) when (e.StatusCode == StatusCode.DeadlineExceeded
                                  || e.StatusCode == StatusCode.Unavailable)
        {
            // Channel layer signalled timeout / no-route — same operator
            // failure mode as the bare TaskCanceledException case below,
            // just with a typed status. Translate to the same diagnostic.
            throw new PoolUnreachableException(
                _pool.Endpoint,
                $"pool unreachable: gRPC {e.StatusCode} after {RegisterResumeDeadline.TotalSeconds:F0}s " +
                $"connecting to {_pool.Endpoint} — verify host/port and firewall " +
                $"(AKOYA_POOL_HOST / AKOYA_POOL_PORT)",
                e);
        }
        catch (OperationCanceledException oce)
        {
            // ct was NOT cancelled (handled above) — so this is the gRPC
            // deadline expiring while the load balancer was still picking
            // a sub-channel. Almost always: nothing is listening on
            // host:port. See PoolUnreachableException.cs for context.
            throw new PoolUnreachableException(
                _pool.Endpoint,
                $"pool unreachable: timed out after {RegisterResumeDeadline.TotalSeconds:F0}s " +
                $"connecting to {_pool.Endpoint} (gRPC channel never became ready) — " +
                $"verify host/port and firewall (AKOYA_POOL_HOST / AKOYA_POOL_PORT)",
                oce);
        }

        if (!resp.Success)
        {
            throw new InvalidOperationException($"Register rejected by pool: {resp.ErrorMessage}");
        }

        _minerId      = resp.MinerId.ToByteArray();
        _sessionToken = resp.SessionToken;
        _identityKey  = resp.IdentityKey;

        Persist(BytesToHex(_minerId));
        _log.LogInformation(
            "session: Registered miner_id={MinerId} initial_nbits={Nbits}",
            BytesToHex(_minerId), resp.InitialDifficultyNbits);
    }

    /// <summary>
    /// Open the bidi MiningStream and run inbound + outbound pumps until
    /// either the cancellation token fires, the server closes the stream, OR
    /// the stream-liveness watchdog trips (no inbound PoolEvent within
    /// <paramref name="streamWatchdog"/>). Sends the mandatory AuthEvent as
    /// the first frame.
    ///
    /// The watchdog exists because the only acceptable failure mode for a
    /// long-running miner is "reconnect and try again". A silently-wedged
    /// stream (TCP black-holed by CF, server-side pod stuck, h2 keepalive
    /// silently lost) MUST be force-closed so the caller's reconnect loop
    /// fires. Pings every ~15s mean Pongs should arrive every ~15s; a
    /// default watchdog of 90s leaves 6× margin before declaring death.
    ///
    /// Pass <c>TimeSpan.Zero</c> or a negative value to disable.
    /// </summary>
    public async Task RunStreamAsync(
        MiningSessionCallbacks callbacks,
        TimeSpan streamWatchdog,
        TimeSpan pongTimeout,
        int outboundDepthTrip,
        CancellationToken ct)
    {
        if (_minerId is null || _sessionToken is null)
            throw new InvalidOperationException("Call ConnectAsync first.");

        _watchdogTripped = false;
        _watchdogTripReason = null;
        Interlocked.Exchange(ref _oldestPendingPingTicks, 0);
        Interlocked.Exchange(ref _pendingPingCount, 0);
        _stream = _pool.Client.MiningStream(cancellationToken: ct);

        // Auth must be the first frame on the stream — server side will
        // close the stream with FailedPrecondition if anything else arrives
        // first.
        await WriteAuthAsync(ct).ConfigureAwait(false);

        Volatile.Write(ref _lastInboundTicks, Environment.TickCount64);

        // Linked CTS: the watchdog cancels it on staleness, the inbound
        // reader bails out cleanly via OperationCanceledException.
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct);
        var writerTask   = Task.Run(() => OutboundWriterLoop(linked.Token), linked.Token);
        var watchdogTask = streamWatchdog > TimeSpan.Zero
            ? Task.Run(() => StreamWatchdogLoop(streamWatchdog, pongTimeout, outboundDepthTrip, linked), linked.Token)
            : Task.CompletedTask;
        try
        {
            await InboundReaderLoop(callbacks, linked.Token).ConfigureAwait(false);
        }
        finally
        {
            // Make sure both helper tasks unblock even if the reader exits
            // cleanly without the watchdog ever firing.
            linked.Cancel();
            _outbound.Writer.TryComplete();
            try { await writerTask.ConfigureAwait(false); }   catch { /* logged in loop */ }
            try { await watchdogTask.ConfigureAwait(false); } catch { /* shutdown */ }
            // Bound the half-close: on a wedged TCP / black-holed CF route,
            // CompleteAsync can hang indefinitely waiting for the peer to
            // acknowledge GOAWAY. We abandon after 2s — the channel-level
            // dispose / new stream will tear down whatever's left.
            try
            {
                var complete = _stream.RequestStream.CompleteAsync();
                var winner   = await Task.WhenAny(complete, Task.Delay(TimeSpan.FromSeconds(2), CancellationToken.None))
                                         .ConfigureAwait(false);
                if (winner != complete)
                    _log.LogDebug("session: RequestStream.CompleteAsync exceeded 2s — abandoning");
                else
                    await complete.ConfigureAwait(false); // observe any exception
            }
            catch { /* peer gone */ }
        }

        // Watchdog cancellation is the only path where _watchdogTripped is
        // set. We surface it as a typed exception so the outer reconnect
        // loop applies exp backoff — a degrading gateway that connects but
        // goes silent must NOT be hammered at the watchdog cadence. Note
        // that an EXTERNAL caller-driven cancellation (Ctrl-C, worker
        // watchdog tripping workerTripCts) does NOT set _watchdogTripped,
        // so those paths still return normally and get the right treatment
        // in Program.cs.
        if (_watchdogTripped && !ct.IsCancellationRequested)
        {
            throw new StreamIdleException(
                _watchdogTripReason ??
                $"stream-liveness watchdog tripped after {streamWatchdog.TotalSeconds:F0}s without inbound traffic");
        }
    }

    /// <summary>Legacy overload: defaults to a 90 s liveness watchdog,
    /// 20 s pong deadline, and 16-item outbound-depth trip.</summary>
    public Task RunStreamAsync(MiningSessionCallbacks callbacks, CancellationToken ct)
        => RunStreamAsync(callbacks,
                          TimeSpan.FromSeconds(90),
                          TimeSpan.FromSeconds(20),
                          outboundDepthTrip: 16,
                          ct);

    /// <summary>Back-compat overload (no pong-deadline / depth-trip).</summary>
    public Task RunStreamAsync(
        MiningSessionCallbacks callbacks,
        TimeSpan streamWatchdog,
        CancellationToken ct)
        => RunStreamAsync(callbacks, streamWatchdog,
                          TimeSpan.FromSeconds(20),
                          outboundDepthTrip: 16,
                          ct);

    private async Task StreamWatchdogLoop(
        TimeSpan budget,
        TimeSpan pongTimeout,
        int depthTrip,
        CancellationTokenSource linked)
    {
        // Poll at 1/3 of the inbound-idle budget so we never overshoot the
        // threshold by more than ~budget/3. The pong/depth checks ride the
        // same tick. Cap the tick at 5s so a generous inbound budget
        // (e.g. 90s) still yields responsive pong-deadline detection.
        var tickMs = Math.Max(1000, Math.Min(5000, budget.TotalMilliseconds / 3));
        var tick = TimeSpan.FromMilliseconds(tickMs);
        try
        {
            while (!linked.IsCancellationRequested)
            {
                await Task.Delay(tick, linked.Token).ConfigureAwait(false);
                var now = Environment.TickCount64;

                // (a) Inbound-idle — original signal. Catches "peer went
                // silent" including transport wedges that h2 keepalive
                // alone can't see (CF↔origin half-open).
                var lastMs = Volatile.Read(ref _lastInboundTicks);
                var idle = now - lastMs;
                if (idle > budget.TotalMilliseconds)
                {
                    _watchdogTripReason =
                        $"stream-liveness watchdog tripped after {budget.TotalSeconds:F0}s without inbound traffic";
                    _log.LogWarning(
                        "session: stream watchdog tripped (idle {Idle:F0}ms > {Budget:F0}ms) — forcing reconnect",
                        idle, budget.TotalMilliseconds);
                    _watchdogTripped = true;
                    linked.Cancel();
                    return;
                }

                // (b) Per-ping pong deadline — Layer 2. Direct round-trip
                // probe; fires faster than (a) when the wedge is in the
                // application layer but other inbound frames (Job, Vardiff)
                // are still trickling through. Detect time bounded by
                // ping-interval + pongTimeout.
                if (pongTimeout > TimeSpan.Zero)
                {
                    var oldest = Volatile.Read(ref _oldestPendingPingTicks);
                    if (oldest != 0)
                    {
                        var age = now - oldest;
                        if (age > pongTimeout.TotalMilliseconds)
                        {
                            _watchdogTripReason =
                                $"pong deadline exceeded: oldest unanswered ping is {age:F0}ms old (> {pongTimeout.TotalSeconds:F0}s)";
                            _log.LogWarning(
                                "session: pong deadline exceeded — oldest unanswered ping {Age:F0}ms (> {Budget:F0}ms), pending={Pending} — forcing reconnect",
                                age, pongTimeout.TotalMilliseconds,
                                Volatile.Read(ref _pendingPingCount));
                            _watchdogTripped = true;
                            linked.Cancel();
                            return;
                        }
                    }
                }

                // (c) Outbound-channel depth — Layer 3. If the writer is
                // parked inside RequestStream.WriteAsync (HTTP/2 flow-
                // control stall, half-open) producers keep enqueuing and
                // the channel swells. Detect that backlog directly so we
                // tear down before more events are abandoned. Channel.Reader
                // .Count is supported on BoundedChannel since .NET 6.
                if (depthTrip > 0 && _outbound.Reader.CanCount)
                {
                    var depth = _outbound.Reader.Count;
                    if (depth > depthTrip)
                    {
                        _watchdogTripReason =
                            $"outbound channel backed up: depth {depth} > {depthTrip}";
                        _log.LogWarning(
                            "session: outbound channel backed up (depth={Depth} > {Trip}) — writer likely parked, forcing reconnect",
                            depth, depthTrip);
                        _watchdogTripped = true;
                        linked.Cancel();
                        return;
                    }
                }
            }
        }
        catch (OperationCanceledException) { /* shutdown */ }
    }

    private void MarkPingSent()
    {
        Interlocked.Increment(ref _pendingPingCount);
        // CompareExchange: only stamp if currently 0 — preserves the OLDEST
        // outstanding ping's timestamp across multiple pings in flight.
        Interlocked.CompareExchange(ref _oldestPendingPingTicks, Environment.TickCount64, 0);
    }

    private void MarkPongReceived()
    {
        // FIFO assumption per PingPump.cs: any Pong matches the oldest
        // outstanding Ping. Decrement count; if anyone is still pending we
        // can't know the exact send-tick, but it's no older than "now" —
        // refresh the timestamp so we don't false-trip on a stale value.
        var remaining = Interlocked.Decrement(ref _pendingPingCount);
        if (remaining <= 0)
        {
            Interlocked.Exchange(ref _pendingPingCount, 0);
            Interlocked.Exchange(ref _oldestPendingPingTicks, 0);
        }
        else
        {
            Interlocked.Exchange(ref _oldestPendingPingTicks, Environment.TickCount64);
        }
    }

    /// <summary>Enqueue an event for sending. Blocks if the outbound queue
    /// is full (back-pressures the caller). Assigns seq automatically.</summary>
    public ValueTask EnqueueAsync(MinerEvent ev, CancellationToken ct)
    {
        ev.Seq = _seq.Next();
        return _outbound.Writer.WriteAsync(ev, ct);
    }

    private async Task WriteAuthAsync(CancellationToken ct)
    {
        var auth = new AuthEvent
        {
            MinerId      = Google.Protobuf.ByteString.CopyFrom(_minerId!),
            SessionToken = _sessionToken!,
        };
        var ev = new MinerEvent { Seq = _seq.Next(), Auth = auth };
        await _stream!.RequestStream.WriteAsync(ev, ct).ConfigureAwait(false);
        _log.LogDebug("session: stream auth sent (seq={Seq})", ev.Seq);
    }

    private async Task OutboundWriterLoop(CancellationToken ct)
    {
        Exception? terminal = null;
        try
        {
            await foreach (var ev in _outbound.Reader.ReadAllAsync(ct).ConfigureAwait(false))
            {
                await _stream!.RequestStream.WriteAsync(ev, ct).ConfigureAwait(false);

                // Truth-in-logging: only events that actually made it through
                // RequestStream.WriteAsync are "on the wire". Per-share INFO
                // here pairs 1:1 with the pool's receive log and with the
                // subsequent "share-result" Info line in WorkerOrchestrator;
                // grepping `✓ share on wire` gives the canonical client-side
                // share-submission count for reconciliation against the pool's
                // records.
                if (ev.EventCase == MinerEvent.EventOneofCase.Share)
                {
                    var share = ev.Share;
                    _log.LogInformation(
                        "session: ✓ share on wire seq={Seq} tile=({Tr},{Tc}) sigma={SigmaPrefix}",
                        ev.Seq, share.TileRow, share.TileCol,
                        Convert.ToHexString(share.Sigma.Span.Slice(0,
                            Math.Min(8, share.Sigma.Length))));
                }
                else if (ev.EventCase == MinerEvent.EventOneofCase.Ping)
                {
                    // Truth-in-logging for liveness: prove our keepalive is
                    // actually hitting the wire (not just enqueued) so the
                    // 90s stream watchdog "idle Xms" figure can be
                    // distinguished from a possible client-side outbound
                    // wedge. Paired with the Pong-received log on the
                    // inbound side gives full RTT visibility. Debug-level
                    // so it doesn't flood INFO every 15s; promote when
                    // diagnosing stream stalls.
                    _log.LogDebug(
                        "session: ✓ ping on wire seq={Seq} ts={Ts}",
                        ev.Seq, ev.Ping.Timestamp);
                    MarkPingSent();
                }
            }
        }
        catch (OperationCanceledException) { /* expected on shutdown */ }
        catch (Exception e)
        {
            terminal = e;
            _log.LogWarning("session: outbound writer ended ({Err})", e.Message);
        }
        finally
        {
            // Critical hang-safety: if the writer dies mid-stream (e.g.
            // RpcException because the stream got reset by the peer) the
            // bounded channel is still WriteAsync-blocking ANY producer at
            // capacity. Completing the channel here makes future producers
            // throw ChannelClosedException immediately instead of silently
            // sitting on a full buffer forever. Pump catches are generic
            // and survive the exception; without this, they'd wedge.
            _outbound.Writer.TryComplete(terminal);

            // Truth-in-logging on tear-down: count anything still queued
            // but never flushed to the wire (e.g. shares enqueued in the
            // last few seconds before a watchdog trip). Pre-fix this was
            // the silent-loss path — we'd "submit" the share locally but
            // the channel got dropped on reconnect with the events still
            // inside. Now we surface the loss with a per-EventCase count.
            int totalAbandoned = 0, sharesAbandoned = 0,
                pingsAbandoned = 0, otherAbandoned = 0;
            while (_outbound.Reader.TryRead(out var stranded))
            {
                totalAbandoned++;
                switch (stranded.EventCase)
                {
                    case MinerEvent.EventOneofCase.Share: sharesAbandoned++; break;
                    case MinerEvent.EventOneofCase.Ping:  pingsAbandoned++;  break;
                    default: otherAbandoned++; break;
                }
            }
            if (totalAbandoned > 0)
            {
                _log.LogWarning(
                    "session: outbound channel drained on tear-down — abandoned {Total} events ({Shares} shares, {Pings} pings, {Other} other)",
                    totalAbandoned, sharesAbandoned, pingsAbandoned, otherAbandoned);
            }
        }
    }

    private async Task InboundReaderLoop(MiningSessionCallbacks cb, CancellationToken ct)
    {
        try
        {
            await foreach (var ev in _stream!.ResponseStream.ReadAllAsync(ct).ConfigureAwait(false))
            {
                // Touch the watchdog on EVERY inbound frame, including ones
                // we don't otherwise handle. The invariant is "the stream is
                // alive", not "we got something interesting".
                Volatile.Write(ref _lastInboundTicks, Environment.TickCount64);

                // Per-ping pong-deadline tracking — clear unconditionally so
                // pongs continue to satisfy the deadline even if no OnPong
                // callback is wired up (e.g. PingPump still constructing).
                if (ev.EventCase == PoolEvent.EventOneofCase.Pong)
                    MarkPongReceived();

                switch (ev.EventCase)
                {
                    case PoolEvent.EventOneofCase.Job when cb.OnJob is not null:
                        await cb.OnJob(ev.Job).ConfigureAwait(false);
                        break;
                    case PoolEvent.EventOneofCase.ShareResult when cb.OnShareResult is not null:
                        await cb.OnShareResult(ev.ShareResult).ConfigureAwait(false);
                        break;
                    case PoolEvent.EventOneofCase.Vardiff when cb.OnVardiff is not null:
                        await cb.OnVardiff(ev.Vardiff).ConfigureAwait(false);
                        break;
                    case PoolEvent.EventOneofCase.Pong when cb.OnPong is not null:
                        _log.LogDebug(
                            "session: ✓ pong received ts={Ts}",
                            ev.Pong.Timestamp);
                        await cb.OnPong(ev.Pong).ConfigureAwait(false);
                        break;
                    case PoolEvent.EventOneofCase.Error:
                        if (cb.OnError is not null) await cb.OnError(ev.Error).ConfigureAwait(false);
                        if (ev.Error.Fatal)
                        {
                            _log.LogError("session: fatal PoolError {Code}: {Msg} — closing stream",
                                ev.Error.Code, ev.Error.Message);
                            return;
                        }
                        break;
                    case PoolEvent.EventOneofCase.Reconnect when cb.OnReconnect is not null:
                        await cb.OnReconnect(ev.Reconnect).ConfigureAwait(false);
                        return; // surrender the stream; caller orchestrates reconnect.
                    default:
                        _log.LogDebug("session: ignored PoolEvent case={Case} seq={Seq}", ev.EventCase, ev.Seq);
                        break;
                }
            }
        }
        catch (OperationCanceledException) { /* shutdown */ }
        catch (RpcException e)
        {
            _log.LogInformation("session: stream ended ({Status}): {Detail}", e.StatusCode, e.Status.Detail);
        }
    }

    private void Persist(string minerIdHex)
    {
        try
        {
            _store.Save(new SessionRecord(
                MinerIdHex:    minerIdHex,
                SessionToken:  _sessionToken!,
                IdentityKey:   _identityKey ?? string.Empty,
                PoolEndpoint:  _pool.Endpoint,
                LastSeenUtc:   DateTime.UtcNow.ToString("O")));
        }
        catch (Exception e)
        {
            _log.LogWarning("session: persist failed ({Err}) — restart will Re-Register", e.Message);
        }
    }

    public async ValueTask DisposeAsync()
    {
        _outbound.Writer.TryComplete();
        if (_stream is not null)
        {
            try { _stream.Dispose(); } catch { /* shutdown */ }
        }
        await ValueTask.CompletedTask;
    }

    // --- byte/hex helpers -------------------------------------------------
    private static string BytesToHex(byte[] b) => Convert.ToHexString(b).ToLowerInvariant();
    private static byte[] HexToBytes(string s) => Convert.FromHexString(s);
}
