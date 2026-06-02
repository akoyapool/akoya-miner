// gRPC channel + MinerService.MinerServiceClient owner.
//
// Single responsibility: hold one GrpcChannel for the pool, configured with
// the proxy-friendly h2 keepalive defaults that long-running bidi streams
// need to survive proxy idle timeouts. Re-creation of the channel
// (e.g. after a fatal PoolError) is the caller's job — PoolConnection itself
// is immutable after construction.
//
// AOT note: Grpc.Net.Client is fully AOT-compatible since 2.60. We avoid
// the SocketsHttpHandler reflection paths by passing an explicit
// SocketsHttpHandler with our knobs already set.

using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using Grpc.Net.Client;
using Microsoft.Extensions.Logging;
using PearlPool.Proto.V2;

namespace Akoya.Pool;

public sealed class PoolConnection : IAsyncDisposable
{
    /// <summary>Keep the long-lived h2 stream warm enough that neither the
    /// gateway nor the underlying TCP route goes idle between mining events.
    /// Ten seconds is intentionally latency-biased: it prevents
    /// slow-start-after-idle without meaningful wire overhead.</summary>
    public static readonly TimeSpan DefaultKeepAlivePingDelay   = TimeSpan.FromSeconds(10);

    public static readonly TimeSpan DefaultKeepAlivePingTimeout = TimeSpan.FromSeconds(10);
    public static readonly TimeSpan DefaultPooledConnectionLifetime    = TimeSpan.FromHours(24);
    public static readonly TimeSpan DefaultPooledConnectionIdleTimeout = TimeSpan.FromHours(24);

    private readonly GrpcChannel _channel;
    private readonly ILogger<PoolConnection> _log;

    public string Endpoint { get; }
    public MinerService.MinerServiceClient Client { get; }

    private PoolConnection(GrpcChannel channel, string endpoint, ILogger<PoolConnection> log)
    {
        _channel = channel;
        _log = log;
        Endpoint = endpoint;
        Client = new MinerService.MinerServiceClient(channel);
    }

    /// <summary>Build a GrpcChannel for the configured pool. Does NOT issue
    /// any RPCs — first network I/O happens when MiningSession.OpenAsync
    /// dials Register/Resume.</summary>
    public static PoolConnection Create(
        string host,
        int port,
        bool useTls,
        ILogger<PoolConnection> log,
        bool tlsInsecure = false,
        TimeSpan? keepAlivePingDelay = null,
        TimeSpan? keepAlivePingTimeout = null)
    {
        var scheme = useTls ? "https" : "http";
        var address = $"{scheme}://{host}:{port}";

        var handler = CreateHttpHandler(
            useTls,
            tlsInsecure,
            log,
            keepAlivePingDelay,
            keepAlivePingTimeout);

        var channel = GrpcChannel.ForAddress(address, new GrpcChannelOptions
        {
            HttpHandler = new GrpcContentTypeHandler { InnerHandler = handler },
            DisposeHttpClient = true,
            // 16 MiB inbound: a ShareSubmission echo carries 2*hashSlice + 2 proofs.
            // V1 worst-case payload was ~3 MiB; pad generously.
            MaxReceiveMessageSize = 16 * 1024 * 1024,
            MaxSendMessageSize    = 16 * 1024 * 1024,
            // Let the gRPC layer keep retrying transient unavailability;
            // we add explicit Resume/Register orchestration above this in
            // MiningSession when a logical reconnect is needed.
            ThrowOperationCanceledOnCancellation = true,
        });

        log.LogInformation(
            "pool: gRPC channel {Address} (tls={Tls}, h2_keepalive={DelaySeconds}s/{TimeoutSeconds}s, tcp_nodelay=true)",
            address,
            useTls,
            (keepAlivePingDelay ?? DefaultKeepAlivePingDelay).TotalSeconds,
            (keepAlivePingTimeout ?? DefaultKeepAlivePingTimeout).TotalSeconds);
        return new PoolConnection(channel, address, log);
    }

    internal static SocketsHttpHandler CreateHttpHandler(
        bool useTls,
        bool tlsInsecure,
        ILogger<PoolConnection> log,
        TimeSpan? keepAlivePingDelay = null,
        TimeSpan? keepAlivePingTimeout = null)
    {
        // SocketsHttpHandler owns the h2 keepalive knobs. Constructing it
        // explicitly keeps the AOT analyzer happy (the GrpcChannelOptions
        // default path uses reflection to probe the handler type).
        var handler = new SocketsHttpHandler
        {
            EnableMultipleHttp2Connections = true,
            KeepAlivePingDelay   = keepAlivePingDelay   ?? DefaultKeepAlivePingDelay,
            KeepAlivePingTimeout = keepAlivePingTimeout ?? DefaultKeepAlivePingTimeout,
            KeepAlivePingPolicy  = HttpKeepAlivePingPolicy.WithActiveRequests,
            ConnectCallback      = ConnectWithNoDelayAsync,
            // 24h: never silently rotate the underlying TCP socket mid-stream.
            // If the proxy or the route drops us, surface it as a stream error
            // so MiningSession can drive a clean Resume.
            PooledConnectionLifetime    = DefaultPooledConnectionLifetime,
            PooledConnectionIdleTimeout = DefaultPooledConnectionIdleTimeout,
            // h2c (plaintext HTTP/2) for local dev — Grpc.Net.Client requires
            // this on the channel options below; SocketsHttpHandler does not
            // negotiate ALPN so the handler defaults work for both modes.
        };

        if (useTls && tlsInsecure)
        {
            // Escape hatch for testnet / self-signed / private-CA scenarios.
            // V1 had no equivalent; V2 adds AKOYA_POOL_TLS_INSECURE=1.
            // NEVER set this in production — it disables peer cert validation
            // entirely and exposes the SessionToken to MitM.
            log.LogWarning(
                "pool: AKOYA_POOL_TLS_INSECURE=1 — TLS cert validation DISABLED. " +
                "Do not use in production.");
            handler.SslOptions = new System.Net.Security.SslClientAuthenticationOptions
            {
#pragma warning disable CA5359 // intentional opt-in via AKOYA_POOL_TLS_INSECURE=1; gated above
                RemoteCertificateValidationCallback = (_, _, _, _) => true,
#pragma warning restore CA5359
            };
        }

        return handler;
    }

    internal static Socket CreateLowLatencySocket() =>
        new(SocketType.Stream, ProtocolType.Tcp) { NoDelay = true };

    private static async ValueTask<Stream> ConnectWithNoDelayAsync(
        SocketsHttpConnectionContext context,
        CancellationToken ct)
    {
        var socket = CreateLowLatencySocket();
        try
        {
            await socket.ConnectAsync(context.DnsEndPoint, ct).ConfigureAwait(false);
            return new NetworkStream(socket, ownsSocket: true);
        }
        catch
        {
            socket.Dispose();
            throw;
        }
    }

    public async ValueTask DisposeAsync()
    {
        // Bound the gRPC channel shutdown: on a wedged TCP route, ShutdownAsync
        // can hang waiting for in-flight RPCs to drain / GOAWAY to be acked.
        // Abandon after 2s and force-dispose; the OS will tear the socket down.
        try
        {
            var shutdown = _channel.ShutdownAsync();
            var winner   = await Task.WhenAny(shutdown, Task.Delay(TimeSpan.FromSeconds(2)))
                                     .ConfigureAwait(false);
            if (winner != shutdown)
                _log.LogDebug("pool: channel shutdown exceeded 2s — abandoning");
            else
                await shutdown.ConfigureAwait(false);
        }
        catch (Exception e) { _log.LogDebug("pool: channel shutdown err {Err}", e.Message); }
        _channel.Dispose();
    }

    // Force the Content-Type to "application/grpc+proto" on outgoing HTTP/2
    // requests. The Grpc.Net.Client default is plain "application/grpc"
    // (spec-valid, no subtype), but some L7 middleboxes only enable their
    // gRPC-aware long-lived-stream path when the subtype is explicit. We
    // were seeing silent stream wedges on the production gateway with no
    // GOAWAY / RST_STREAM — symptom of an LB demoting the connection to
    // generic HTTP/2 and idle-evicting. The "+proto" suffix is the standard
    // hint per https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md
    private sealed class GrpcContentTypeHandler : DelegatingHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken ct)
        {
            var ct0 = request.Content?.Headers.ContentType;
            if (ct0 is not null && ct0.MediaType == "application/grpc")
                ct0.MediaType = "application/grpc+proto";
            return base.SendAsync(request, ct);
        }
    }
}
