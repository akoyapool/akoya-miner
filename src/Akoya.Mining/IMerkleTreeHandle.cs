namespace Akoya.Mining;

public interface IMerkleTreeHandle
{
    byte[] Root { get; }
    uint TotalLeaves { get; }

    void Acquire();
    void Release();
    MerkleRootAndProofResult Proof(ReadOnlySpan<uint> rowIndices);
    byte[] AuditPaths(ReadOnlySpan<uint> leafIndices);
}
