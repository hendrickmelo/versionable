using System.Diagnostics;

namespace Versionable.Engine;

/// <summary>
/// Where the engine reports the situations Python reports with <c>logger.warning</c>.
/// </summary>
/// <remarks>
/// Python counterpart: the module-level <c>logger</c> in <c>src/versionable/_api.py</c> and
/// <c>src/versionable/_types.py</c>. Those warnings are observable behavior, not debug noise —
/// a file with no <c>version</c> key, a literal outside its declared options, an enum value the
/// code no longer defines — so the C# engine has to surface them somewhere.
/// <para>
/// Deliberately not <c>Microsoft.Extensions.Logging</c>: that would put a dependency on the
/// package for four call sites, and a serialization library should not dictate a host's logging
/// stack. Messages go to <see cref="Trace"/> (on by default in SDK-style builds) and to
/// <see cref="Warning"/> for hosts that want to route them. Subscribers are invoked
/// synchronously on the thread doing the load; keep handlers cheap and do not throw from them.
/// </para>
/// </remarks>
public static class VersionableLog
{
    /// <summary>Raised for every warning the engine emits. Handlers must not throw.</summary>
    public static event Action<string>? Warning;

    internal static void Warn(string message)
    {
        Warning?.Invoke(message);
        Trace.TraceWarning(message);
    }
}
