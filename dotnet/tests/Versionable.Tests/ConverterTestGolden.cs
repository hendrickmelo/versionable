using System.Runtime.CompilerServices;
using System.Text.Json;

namespace Versionable.Tests;

/// <summary>
/// Locates and parses the checked-in golden corpus.
/// </summary>
/// <remarks>
/// The corpus under <c>conformance/golden/</c> is Python-written and is the contract
/// (<c>docs/plans/csharp-port.md</c>, § Conformance): each fixture pairs a
/// <c>manifest.json</c> of expected values with the four backend files that hold them. Tests
/// read the bytes from the repository rather than from a copied build artefact, so a fixture
/// regenerated on the Python side is picked up without a corresponding change here.
/// </remarks>
internal static class ConverterTestGolden
{
    private static readonly string _goldenRoot = FindGoldenRoot();

    /// <summary>Reads a file out of one fixture directory.</summary>
    /// <param name="fixture">Fixture directory name, e.g. <c>arrays</c>.</param>
    /// <param name="fileName">File name within it, e.g. <c>arrays.json</c>.</param>
    /// <returns>The parsed document.</returns>
    internal static JsonDocument Read(string fixture, string fileName) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(_goldenRoot, fixture, fileName)));

    /// <summary>Reads a fixture's expected-value manifest.</summary>
    /// <param name="fixture">Fixture directory name.</param>
    /// <returns>The <c>values</c> object of <c>manifest.json</c>.</returns>
    internal static JsonElement Manifest(string fixture)
    {
        // Cloned, so the element outlives the document it came from.
        using JsonDocument document = Read(fixture, "manifest.json");
        return document.RootElement.GetProperty("values").Clone();
    }

    /// <summary>Reads a fixture's JSON wire file.</summary>
    /// <param name="fixture">Fixture directory name.</param>
    /// <returns>The root object.</returns>
    internal static JsonElement Wire(string fixture)
    {
        using JsonDocument document = Read(fixture, fixture + ".json");
        return document.RootElement.Clone();
    }

    private static string FindGoldenRoot([CallerFilePath] string thisFile = "")
    {
        // Walk up rather than hard-coding a depth, and try the source location as well as the
        // output one: bin/<config>/<tfm>/ sits under the repository for `dotnet test`, but not
        // when the suite is compiled into a scratch project elsewhere.
        return SearchUp(AppContext.BaseDirectory)
            ?? SearchUp(Path.GetDirectoryName(thisFile))
            ?? throw new DirectoryNotFoundException(
                $"No conformance/golden/index.json above {AppContext.BaseDirectory} or {thisFile}.");
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
