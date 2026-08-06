using System.Globalization;
using System.Runtime.CompilerServices;

namespace Versionable.Tests;

/// <summary>
/// Resolves the root of the golden corpus every corpus-reading test class reads from.
/// </summary>
/// <remarks>
/// The corpus under <c>conformance/golden/</c> is Python-written and is the contract
/// (<c>docs/plans/csharp-port.md</c>, § Conformance). Tests read the bytes out of the repository
/// rather than a copied build artefact, so a fixture regenerated on the Python side is picked up
/// without a corresponding change here.
/// <para>
/// <b>Why the environment override exists.</b> Setting <c>VERSIONABLE_GOLDEN_ROOT</c> points the
/// whole suite at a different corpus directory, which is what makes the bidirectional conformance
/// job in <c>.github/workflows/ci.yml</c> possible: Python regenerates every fixture into a scratch
/// directory with <c>conformance/generate_golden.py --output</c>, CI exports the variable, and the
/// existing JSON/YAML/TOML/HDF5 corpus suites — manifest assertions and all — run against the fresh
/// bytes instead of the committed ones. No parallel set of tests has to be maintained for the
/// Python-writes/C#-reads direction; it is the same suite pointed somewhere else.
/// </para>
/// <para>
/// The override must name a directory containing <c>index.json</c>; a variable that points at
/// nothing is an error rather than a silent fall back to the committed corpus, because a CI job
/// that quietly re-tested the checked-in bytes would report success for a comparison it never made.
/// </para>
/// </remarks>
internal static class GoldenCorpus
{
    /// <summary>The environment variable that redirects the suite to another corpus directory.</summary>
    internal const string RootVariable = "VERSIONABLE_GOLDEN_ROOT";

    /// <summary>The corpus directory: the committed one, or whatever <see cref="RootVariable"/> names.</summary>
    internal static string Root { get; } = Resolve();

    private static string Resolve([CallerFilePath] string thisFile = "")
    {
        string? overridden = Environment.GetEnvironmentVariable(RootVariable);
        if (!string.IsNullOrWhiteSpace(overridden))
        {
            string root = Path.GetFullPath(overridden);
            return File.Exists(Path.Combine(root, "index.json"))
                ? root
                : throw new DirectoryNotFoundException(
                    string.Create(
                        CultureInfo.InvariantCulture,
                        $"{RootVariable} is set to '{root}', which has no index.json."));
        }

        // Walk up rather than hard-coding a depth, and try the source location as well as the
        // output one: bin/<config>/<tfm>/ sits under the repository for `dotnet test`, but not
        // when the suite is compiled into a scratch project elsewhere.
        return SearchUp(AppContext.BaseDirectory)
            ?? SearchUp(Path.GetDirectoryName(thisFile))
            ?? throw new DirectoryNotFoundException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"No conformance/golden/index.json above '{AppContext.BaseDirectory}' or '{thisFile}'."));
    }

    private static string? SearchUp(string? start)
    {
        for (DirectoryInfo? directory = start is null ? null : new DirectoryInfo(start);
            directory is not null;
            directory = directory.Parent)
        {
            string candidate = Path.Combine(directory.FullName, "conformance", "golden");
            if (File.Exists(Path.Combine(candidate, "index.json")))
            {
                return candidate;
            }
        }

        return null;
    }
}
