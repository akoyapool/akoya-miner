using Akoya.Crypto;
using PearlPool.Proto.V2;

namespace Akoya.Miner.Mining;

internal sealed class SigmaContext
{
    public const int HeaderSize = 76;
    public const int ConfigSize = 52;
    public const int BSeedSize = 32;

    public Guid JobId { get; }
    public byte[] Sigma { get; }                    // 76 B incomplete header
    public byte[] ConfigBytes { get; }              // 52 B MiningConfiguration.ToBytes()
    public uint CommonDim { get; }                  // = K (miner-chosen)
    public ushort Rank { get; }                     // = R (miner-chosen, default 128)
    public byte[] JobKey { get; }                   // 32 B BLAKE3 keyed merkle key
    public byte[] BSeed { get; }
    /// <summary>Audit-proof v1 K parameter (0 disabled, ≤64 per spec). When &gt;0
    /// every ShareSubmission carries a K-opening AuditProof keyed by
    /// (claimed_hash, b_seed, K) per the audit_proof v1 schematic.</summary>
    public uint AuditK { get; }
    public uint TargetNbits { get; }
    public uint NetworkTargetNbits { get; }
    public long BlockHeight { get; }

    private SigmaContext(
        Guid jobId, byte[] sigma, byte[] configBytes,
        uint commonDim, ushort rank, byte[] jobKey,
        byte[] bSeed, uint auditK,
        uint targetNbits, uint networkTargetNbits, long blockHeight)
    {
        JobId              = jobId;
        Sigma              = sigma;
        ConfigBytes        = configBytes;
        CommonDim          = commonDim;
        Rank               = rank;
        JobKey             = jobKey;
        BSeed              = bSeed;
        AuditK             = auditK;
        TargetNbits        = targetNbits;
        NetworkTargetNbits = networkTargetNbits;
        BlockHeight        = blockHeight;
    }

    /// <summary>
    /// True iff <paramref name="job"/> carries the structural minimum required
    /// to build a <see cref="SigmaContext"/> — i.e. a 16-byte UUID job_id and a
    /// header-sized sigma. Used as a pre-publish guard on paths where the pool
    /// may return a "session resumed but no current job yet" response
    /// (Resume Success=true with empty job_id/sigma): the orchestrator must
    /// accept the session without crashing and wait for the next OnJob over
    /// the bidi stream, rather than blow up and trip reconnect on a perfectly
    /// authenticated session. Pool integration doc §2.4 commits Resume to
    /// always carry the current job if one exists; if pool returns success
    /// without it, treat it as "no current job — stream will deliver".
    /// </summary>
    public static bool IsValidInitialJob(JobAssignment job)
    {
        if (job is null) return false;
        if (job.JobId is null || job.JobId.Length != 16) return false;
        if (job.Sigma is null || job.Sigma.Length != HeaderSize) return false;
        // audit_proof v1 requires a 32 B b_seed. If the pool's ResumeResponse
        // omitted it (e.g. an older pool version), fall through to the
        // MiningStream OnJob path instead of crashing the orchestrator.
        if (job.BSeed is null || job.BSeed.Length != BSeedSize) return false;
        return true;
    }

    /// <summary>
    /// Parse a JobAssignment into the immutable per-σ snapshot.
    /// </summary>
    /// <param name="job">The JobAssignment the pool just pushed.</param>
    /// <param name="minerId">16-byte minerId assigned by Register (little-endian
    /// Guid byte layout — i.e. what arrives on the wire as RegisterResponse.miner_id).
    /// The 3-arg <see cref="CommitmentHasher.GetKey"/> overload handles the
    /// RFC 4122 big-endian re-serialisation required by the pool.</param>
    /// <param name="commonDim">Miner's chosen K.</param>
    /// <param name="rank">Miner's chosen R (default 128).</param>
    public static SigmaContext FromJobAssignment(
        JobAssignment job,
        ReadOnlySpan<byte> minerId,
        uint commonDim,
        ushort rank)
    {
        if (job.Sigma.Length != HeaderSize)
            throw new InvalidOperationException(
                $"JobAssignment.sigma length {job.Sigma.Length} != expected {HeaderSize} (V2: header-only)");
        if (minerId.Length != 16)
            throw new ArgumentException("minerId must be 16 B", nameof(minerId));
        if (job.JobId.Length != 16)
            throw new InvalidOperationException("JobAssignment.job_id must be 16 B");
        if (commonDim == 0)
            throw new ArgumentOutOfRangeException(nameof(commonDim), "commonDim (K) must be non-zero");
        if (rank == 0)
            throw new ArgumentOutOfRangeException(nameof(rank), "rank (R) must be non-zero");
        if (job.BSeed.Length != BSeedSize)
            throw new InvalidOperationException(
                $"JobAssignment.b_seed length {job.BSeed.Length} != expected {BSeedSize} (audit_proof v1)");
        if (job.AuditK > Akoya.Crypto.AuditIndexDeriver.AuditKMax)
            throw new InvalidOperationException(
                $"JobAssignment.audit_k {job.AuditK} > spec cap {Akoya.Crypto.AuditIndexDeriver.AuditKMax}");

        var sigma = job.Sigma.ToByteArray(); // 76 B header
        var configBytes = MiningConfiguration.Default(commonDim, rank).ToBytes();

        _ = minerId; // accepted + length-validated above; not part of jobKey.
#pragma warning disable CS0618 // 2-arg overload is canonical for V2.
        var jobKey = CommitmentHasher.GetKey(sigma, MiningConfiguration.Default(commonDim, rank));
#pragma warning restore CS0618

        return new SigmaContext(
            jobId:              new Guid(job.JobId.Span),
            sigma:              sigma,
            configBytes:        configBytes,
            commonDim:          commonDim,
            rank:               rank,
            jobKey:             jobKey,
            bSeed:              job.BSeed.ToByteArray(),
            auditK:             job.AuditK,
            targetNbits:        job.TargetNbits,
            networkTargetNbits: job.NetworkTargetNbits,
            blockHeight:        job.BlockHeight);
    }

    /// <summary>
    /// Return a copy of this context with the share-difficulty <c>nbits</c>
    /// replaced. Used by the vardiff handler: when the pool nudges difficulty
    /// up/down mid-σ, we republish on the JobBus with the same σ but the new
    /// target so every <see cref="GpuWorker"/> picks it up on its next
    /// drain — without waiting for an actual σ rotation from the pool.
    /// </summary>
    public SigmaContext WithTargetNbits(uint newTargetNbits) => new(
        jobId:              JobId,
        sigma:              Sigma,
        configBytes:        ConfigBytes,
        commonDim:          CommonDim,
        rank:               Rank,
        jobKey:             JobKey,
        bSeed:              BSeed,
        auditK:             AuditK,
        targetNbits:        newTargetNbits,
        networkTargetNbits: NetworkTargetNbits,
        blockHeight:        BlockHeight);

    /// <summary>Convert compact NBits target representation to a full 256-bit target.</summary>
    public static System.Numerics.BigInteger NbitsToTarget(uint nbits)
    {
        int exp = (int)(nbits >> 24);
        uint mantissa = nbits & 0x00FFFFFFu;
        if (exp <= 3) return new System.Numerics.BigInteger(mantissa >> (8 * (3 - exp)));
        return new System.Numerics.BigInteger(mantissa) << (8 * (exp - 3));
    }

    /// <summary>SHA-256(σ) first 8 B — mirrors the server's sigma fingerprint
    /// so we can compare quickly across heartbeats without round-tripping σ.</summary>
    public static byte[] Fingerprint(ReadOnlySpan<byte> sigma)
    {
        if (sigma.IsEmpty) return [];
        Span<byte> full = stackalloc byte[32];
        System.Security.Cryptography.SHA256.HashData(sigma, full);
        var fp = new byte[8];
        full[..8].CopyTo(fp);
        return fp;
    }
}
