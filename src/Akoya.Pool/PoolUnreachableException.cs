// Thrown by MiningSession.RegisterAsync / TryResumeAsync when the gRPC
// channel never reaches a ready state within RegisterResumeDeadline (30s).

namespace Akoya.Pool;

public sealed class PoolUnreachableException : Exception
{
    public string Endpoint { get; }
    public PoolUnreachableException(string endpoint, string message, Exception inner)
        : base(message, inner)
    {
        Endpoint = endpoint;
    }
}
