using Xunit;

namespace Versionable.Tests;

/// <summary>
/// The xUnit collection every test class that touches a process-wide registry belongs to.
/// </summary>
/// <remarks>
/// <see cref="VersionableRegistry"/>, <c>Backends.BackendRegistry</c>, and
/// <c>Converters.ConverterRegistry</c> are static, are populated by <c>[ModuleInitializer]</c>
/// methods that run exactly once per process, and carry an internal <c>Reset</c> test seam.
/// A reset is therefore destructive in a way an ordinary fixture is not — nothing re-runs a
/// module initializer — and a class doing it while another class reads the same static is the
/// failure mode that passes in isolation and fails in a full run.
/// <para>
/// xUnit runs the classes of one collection sequentially, so naming this collection is what makes
/// those resets safe, and it is also what lets cross-collection parallelism stay on for everything
/// else: the Roslyn suites, which are by far the slowest, touch no registry and run alongside.
/// There is deliberately no <c>xunit.runner.json</c> — the default (<c>parallelizeTestCollections:
/// true</c>) is the intended configuration.
/// </para>
/// <para>
/// Membership rule: a class belongs here if it resets a registry, or if it reads one — directly,
/// or through <c>VersionableFile</c>, <c>WireValues</c>, or a converter lookup. Classes that only
/// touch <c>BuiltinConverters.All</c> (a plain static array) or that run inside their own
/// <c>AssemblyLoadContext</c> do not: they see no shared state.
/// </para>
/// </remarks>
[CollectionDefinition(Name)]
public sealed class RegistryCollection
{
    /// <summary>The collection name, as <c>[Collection]</c> spells it.</summary>
    public const string Name = "Versionable registries";
}
