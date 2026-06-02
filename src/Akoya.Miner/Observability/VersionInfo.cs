// Version banner + capi handshake.

using System.Reflection;
using Akoya.Mining;
using Akoya.PearlGemm;
using Microsoft.Extensions.Logging;

namespace Akoya.Miner.Observability;

internal static class VersionInfo
{
    /// <summary>Bumped whenever the C# miner relies on a new capi entry point or
    /// changes the on-wire format. Compared against pearl_capi_*_version.</summary>
    public const int RequiredGemmAbi   = 2;
    public const int RequiredMiningAbi = 2;

    public static string MinerVersion =>
        Assembly.GetExecutingAssembly()
            .GetCustomAttributes<AssemblyMetadataAttribute>()
            .FirstOrDefault(a => a.Key == "MinerVersion")?.Value
            ?? "0.0.0";

    public static string GitSha =>
        Assembly.GetExecutingAssembly()
            .GetCustomAttributes<AssemblyMetadataAttribute>()
            .FirstOrDefault(a => a.Key == "GitSha")?.Value
            ?? "unknown";

    public static int Run(string[] args)
    {
        Console.WriteLine($"akoya-miner");
        Console.WriteLine($"  version      : {MinerVersion}");
        Console.WriteLine($"  git_sha      : {GitSha}");
        Console.WriteLine($"  runtime      : {Environment.Version} ({System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription})");
        try
        {
            int gemmAbi = PearlGemmNative.AbiVersion();
            Console.WriteLine($"  pearl_gemm   : abi v{gemmAbi}{(gemmAbi == RequiredGemmAbi ? "" : $" (mismatch — required v{RequiredGemmAbi})")}");
        }
        catch (Exception e)
        {
            Console.WriteLine($"  pearl_gemm   : FAILED to load — {e.Message}");
        }
        try
        {
            uint miningAbi = PearlMiningNative.Version();
            Console.WriteLine($"  pearl_mining : abi v{miningAbi}{(miningAbi == RequiredMiningAbi ? "" : $" (mismatch — required v{RequiredMiningAbi})")}");
        }
        catch (Exception e)
        {
            Console.WriteLine($"  pearl_mining : FAILED to load — {e.Message}");
        }
        return 0;
    }

    /// <summary>Probe both .so libs and abort if either ABI doesn't match the
    /// version this miner was built against. Logs the versions on success.</summary>
    public static void LogAndCheck(ILogger log)
    {
        int gemmAbi;
        uint miningAbi;
        try { gemmAbi = PearlGemmNative.AbiVersion(); }
        catch (Exception e) { throw new InvalidOperationException($"failed to load libpearl_gemm_capi.so: {e.Message}", e); }
        try { miningAbi = PearlMiningNative.Version(); }
        catch (Exception e) { throw new InvalidOperationException($"failed to load libpearl_mining_capi.so: {e.Message}", e); }

        log.LogInformation("version git_sha={Sha} pearl_gemm_abi=v{Gemm} pearl_mining_abi=v{Mining}",
            GitSha, gemmAbi, miningAbi);

        if (gemmAbi != RequiredGemmAbi)
            throw new InvalidOperationException(
                $"libpearl_gemm_capi.so ABI v{gemmAbi} != required v{RequiredGemmAbi} — rebuild the .so against this miner's source tree");
        if (miningAbi != RequiredMiningAbi)
            throw new InvalidOperationException(
                $"libpearl_mining_capi.so ABI v{miningAbi} != required v{RequiredMiningAbi} — rebuild the .so against this miner's source tree");
    }
}
