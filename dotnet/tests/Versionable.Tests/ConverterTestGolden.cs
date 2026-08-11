using System.Text.Json;

namespace Versionable.Tests;

/// <summary>
/// Parses files out of the golden corpus <see cref="GoldenCorpus"/> resolves.
/// </summary>
/// <remarks>
/// Each fixture pairs a <c>manifest.json</c> of expected values with the four backend files that
/// hold them; this reads either. The corpus is Python-written and is the contract
/// (<c>docs/plans/csharp-port.md</c>, § Conformance).
/// </remarks>
internal static class ConverterTestGolden
{
    private static readonly string _goldenRoot = GoldenCorpus.Root;

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
}
