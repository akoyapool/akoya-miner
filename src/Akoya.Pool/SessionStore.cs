// Persistent session state: what we need to resume a MiningStream after a
// miner restart without burning a fresh registration.
//
// On-disk format is JSON (System.Text.Json, AOT-safe via source-gen). The
// file is written atomically via temp + rename, owner-only perms (0600 on
// POSIX). Default path: $HOME/.akoya/session.json (configurable via
// AKOYA_SESSION_FILE).
//
// Lifecycle:
//   - Read once at startup; if present, MiningSession tries Resume first.
//   - Written once per successful Register or successful Resume (token gets
//     refreshed every 4h on the server side; we persist the new token).
//   - On hard auth failure (e.g. "session expired"), DeleteAsync() is called
//     so the next start does a clean Register.
//
// NOT included on-disk: wallet/worker (lives in env), GPU UUIDs (re-derived
// from CUDA at every start), miner_version (build-time constant).

using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;

namespace Akoya.Pool;

/// <summary>Immutable session record persisted across miner restarts.</summary>
public sealed record SessionRecord(
    /// <summary>16-byte miner_id assigned by the pool on first Register.
    /// Hex-encoded (32 chars, no dashes) for human readability.</summary>
    string MinerIdHex,

    /// <summary>Pool-issued HMAC session token. 4h TTL on the server.</summary>
    string SessionToken,

    /// <summary>Long-lived identity key. Survives session-token rotation;
    /// used to re-Register without losing accumulated worker history.</summary>
    string IdentityKey,

    /// <summary>The pool host:port this session was issued by. If the operator
    /// repoints AKOYA_POOL_HOST to a different pool, we discard the file
    /// (different pool = different miner_id namespace).</summary>
    string PoolEndpoint,

    /// <summary>UTC ISO-8601 of the last successful Register or Resume. Purely
    /// informational; the server is the authority on token expiry.</summary>
    string LastSeenUtc);

/// <summary>JSON source-generation context — keeps SessionStore AOT-clean.</summary>
[JsonSourceGenerationOptions(
    WriteIndented = true,
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(SessionRecord))]
internal sealed partial class SessionJsonContext : JsonSerializerContext { }

public sealed class SessionStore
{
    private readonly string _path;
    private readonly ILogger<SessionStore> _log;

    public SessionStore(string path, ILogger<SessionStore> log)
    {
        _path = path;
        _log = log;
    }

    public string Path => _path;

    /// <summary>Load the session if present and parseable. Returns null on
    /// missing-file, corrupt-file, or wrong-pool (caller should then
    /// Register fresh). Never throws.</summary>
    public SessionRecord? TryLoad(string currentPoolEndpoint)
    {
        if (!File.Exists(_path))
        {
            _log.LogDebug("session: no file at {Path} — first start", _path);
            return null;
        }

        try
        {
            using var fs = File.OpenRead(_path);
            var rec = JsonSerializer.Deserialize(fs, SessionJsonContext.Default.SessionRecord);
            if (rec is null)
            {
                _log.LogWarning("session: file at {Path} parsed to null — ignoring", _path);
                return null;
            }
            if (!string.Equals(rec.PoolEndpoint, currentPoolEndpoint, StringComparison.OrdinalIgnoreCase))
            {
                _log.LogWarning(
                    "session: stored pool {Stored} differs from current {Current} — discarding session",
                    rec.PoolEndpoint, currentPoolEndpoint);
                return null;
            }
            var schemaErr = Validate(rec);
            if (schemaErr is not null)
            {
                _log.LogWarning(
                    "session: stored record failed schema validation ({Err}) — discarding, will re-Register",
                    schemaErr);
                return null;
            }
            _log.LogInformation("session: loaded miner_id={MinerId} last_seen={LastSeen}",
                rec.MinerIdHex, rec.LastSeenUtc);
            return rec;
        }
        catch (Exception e)
        {
            _log.LogWarning("session: failed to load {Path}: {Err} — will re-Register", _path, e.Message);
            return null;
        }
    }

    /// <summary>Atomically write the session record (temp file + rename).
    /// On POSIX, sets file mode 0600 before the rename. Throws on I/O failure
    /// because losing a session record means the next restart re-Registers
    /// and breaks worker continuity — better to surface the problem.</summary>
    public void Save(SessionRecord rec)
    {
        ValidateOrThrow(rec);
        var dir = System.IO.Path.GetDirectoryName(_path)!;
        Directory.CreateDirectory(dir);

        var tmp = _path + ".tmp";
        using (var fs = new FileStream(tmp, FileMode.Create, FileAccess.Write, FileShare.None))
        {
            JsonSerializer.Serialize(fs, rec, SessionJsonContext.Default.SessionRecord);
            // Force the bytes to disk BEFORE the rename. Without this, on
            // ext4 default (data=ordered) a power-cut between the rename
            // and the next fsync can leave a zero-length file at _path —
            // the classic "atomic rename, non-durable contents" bug. The
            // miner would then fail to Resume on next boot and silently
            // re-Register (worker history reset). fsync(2) the file
            // contents; rename(2) on the directory entry is already
            // crash-atomic.
            fs.Flush(flushToDisk: true);
        }
        TrySetOwnerOnlyPerms(tmp);
        // File.Move with overwrite = true is atomic on POSIX (rename(2)).
        File.Move(tmp, _path, overwrite: true);
        _log.LogDebug("session: persisted miner_id={MinerId} to {Path}", rec.MinerIdHex, _path);
    }

    // ─── Schema validation ────────────────────────────────────────────────
    //
    // Defends against:
    //   • Hand-edited session files with wrong field shapes.
    //   • Forward-compatibility footgun: if a future format change ever
    //     loosens the JSON contract, this is the single chokepoint that
    //     catches it before we put bad bytes on the wire to the pool.
    //
    // Used by BOTH Save (we never want to persist garbage) and TryLoad
    // (we never want to ship garbage to the pool from a corrupted file).
    private const int MinerIdHexLength = 32;       // 16 bytes hex-encoded
    private const int IdentityKeyMinLength = 1;    // pool implementation-defined; non-empty
    private const int SessionTokenMaxLength = 4096;
    private const int IdentityKeyMaxLength = 4096;

    private static void ValidateOrThrow(SessionRecord rec)
    {
        var err = Validate(rec);
        if (err is not null) throw new InvalidDataException("session record invalid: " + err);
    }

    /// <summary>Returns null on valid, error message on invalid.</summary>
    public static string? Validate(SessionRecord rec)
    {
        if (rec is null) return "null record";
        if (string.IsNullOrEmpty(rec.MinerIdHex) || rec.MinerIdHex.Length != MinerIdHexLength)
            return $"MinerIdHex must be {MinerIdHexLength} hex chars (got {rec.MinerIdHex?.Length ?? 0})";
        foreach (var c in rec.MinerIdHex)
        {
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
                return "MinerIdHex contains non-hex chars";
        }
        // SessionToken may legitimately be empty (cleared after unauth_*);
        // identity_key is the one we MUST preserve. Cap both for sanity.
        if (rec.SessionToken is null) return "SessionToken is null";
        if (rec.SessionToken.Length > SessionTokenMaxLength) return "SessionToken exceeds max length";
        if (string.IsNullOrEmpty(rec.IdentityKey) || rec.IdentityKey.Length < IdentityKeyMinLength)
            return "IdentityKey is empty";
        if (rec.IdentityKey.Length > IdentityKeyMaxLength) return "IdentityKey exceeds max length";
        if (string.IsNullOrEmpty(rec.PoolEndpoint)) return "PoolEndpoint is empty";
        if (string.IsNullOrEmpty(rec.LastSeenUtc)) return "LastSeenUtc is empty";
        return null;
    }

    /// <summary>Delete the session file (e.g. after a "session expired" error
    /// from the server, so next start does a clean Register). No-op if file
    /// is already gone.</summary>
    public void Delete()
    {
        try { File.Delete(_path); _log.LogInformation("session: deleted {Path}", _path); }
        catch (Exception e) { _log.LogWarning("session: delete failed for {Path}: {Err}", _path, e.Message); }
    }

    /// <summary>
    /// Drop only the cached <c>SessionToken</c> (so we can't Resume) while
    /// preserving <c>MinerId</c> + <c>IdentityKey</c>. Use on
    /// <c>PoolError.fatal=true</c> with code <c>unauthenticated_*</c>: per
    /// V2 integration doc §7, the right response is "re-Register" — and
    /// re-Register WITH the identity_key reclaims the same miner_id.
    /// Wiping everything (via <see cref="Delete"/>) would force the pool
    /// onto Tier-2 identity matching and risk a new minerId.
    /// </summary>
    public void ClearSessionToken()
    {
        var existing = TryLoadAny();
        if (existing is null) return;
        try
        {
            Save(existing with { SessionToken = string.Empty });
            _log.LogInformation("session: cleared SessionToken (kept identity_key + miner_id)");
        }
        catch (Exception e)
        {
            _log.LogWarning("session: clear-token failed: {Err}", e.Message);
        }
    }

    // Pool-endpoint-agnostic load — ClearSessionToken needs the current
    // record regardless of endpoint match (it just zeroes the token in place).
    private SessionRecord? TryLoadAny()
    {
        try
        {
            if (!File.Exists(_path)) return null;
            using var fs = File.OpenRead(_path);
            return JsonSerializer.Deserialize(fs, SessionJsonContext.Default.SessionRecord);
        }
        catch { return null; }
    }

    private static void TrySetOwnerOnlyPerms(string path)
    {
        if (!OperatingSystem.IsWindows())
        {
            try
            {
                // 0600 = owner R/W only. SessionToken is auth material;
                // operator scripts often `chmod -R` whole homedirs and we
                // want to detect/fail in that scenario rather than silently
                // leak. But best-effort here.
                File.SetUnixFileMode(path,
                    UnixFileMode.UserRead | UnixFileMode.UserWrite);
            }
            catch { /* old runtime / non-POSIX filesystem — silently OK */ }
        }
    }
}
