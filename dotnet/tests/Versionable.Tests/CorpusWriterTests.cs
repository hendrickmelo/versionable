using System.Globalization;
using System.Text.Json;
using Versionable.Engine;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// Writes the whole corpus with the C# serializers, for the Python side to read back.
/// </summary>
/// <remarks>
/// This is the writer half of the bidirectional conformance job in
/// <c>.github/workflows/ci.yml</c>. The other three quarters of that matrix need no entry point of
/// their own: both suites already read the committed corpus, and <see cref="GoldenCorpus"/>'
/// environment override points this suite at a freshly Python-generated one. Only
/// C#-writes/Python-reads has nothing to run, because nothing in the C# suite emits files a
/// separate process can find afterwards. This does.
/// <para>
/// <b>Values come from the committed corpus, bytes come from C#.</b> Each fixture is loaded out of
/// its committed JSON file and re-saved through all four backends, rather than being constructed
/// from a second copy of the fixture data held here. That is deliberate: a hand-written copy would
/// be a third place the fixture values live (after <c>conformance/golden_schemas.py</c> and the
/// manifests) and could drift out of agreement with the manifests the Python reader asserts
/// against, turning a real serializer bug into an unexplained value mismatch. Loading first means
/// the values are the manifest's by construction and the only thing under test is what the C#
/// writers do with them.
/// </para>
/// <para>
/// <b>Migration sources are excluded</b> and cannot be included: they are files stamped with an
/// older version and hash, and no writer emits an envelope for a version other than the type's
/// current one. Old-file handling is covered by the other direction, where Python's committed
/// v1/v2 files migrate forward in the C# suite.
/// </para>
/// <para>
/// <b>It runs on every ordinary <c>dotnet test</c></b>, writing to a temporary directory it then
/// deletes. Only the destination changes when <c>VERSIONABLE_CORPUS_OUT</c> is set. A writer that
/// ran only under a CI environment variable would rot unnoticed between the times anyone looked at
/// the conformance job.
/// </para>
/// </remarks>
[Collection(RegistryCollection.Name)]
public class CorpusWriterTests
{
    /// <summary>Destination for the written corpus; a self-deleting temporary directory when unset.</summary>
    internal const string OutputVariable = "VERSIONABLE_CORPUS_OUT";

    public CorpusWriterTests() => GoldenSchemas.EnsureRegistered();

    /// <summary>
    /// Every fixture, saved by every backend, and read back into the same values it went in as.
    /// </summary>
    /// <remarks>
    /// The reload assertion is what makes this a test rather than a script: it fails here, with a
    /// fixture and backend named, instead of surfacing as a confusing Python-side error about a
    /// file this process wrote. The cross-language claim is still the Python reader's to make — a
    /// C# round trip proves only that C# agrees with itself.
    /// </remarks>
    [Fact]
    public void every_fixture_is_written_by_every_backend_and_reads_back_unchanged()
    {
        string? destination = Environment.GetEnvironmentVariable(OutputVariable);
        bool temporary = string.IsNullOrWhiteSpace(destination);
        string output = temporary
            ? Path.Combine(Path.GetTempPath(), $"versionable-corpus-{Path.GetRandomFileName()}")
            : Path.GetFullPath(destination!);

        try
        {
            IReadOnlyList<string> written = WriteCorpus(output);

            // 11 fixtures x 4 backends. A silently shrinking corpus would otherwise pass.
            Assert.Equal(Fixtures().Count * Backends().Count, written.Count);
        }
        finally
        {
            if (temporary && Directory.Exists(output))
            {
                Directory.Delete(output, recursive: true);
            }
        }
    }

    /// <summary>Writes every fixture to every backend under <paramref name="output"/>.</summary>
    /// <param name="output">Destination root; fixture subdirectories are created inside it.</param>
    /// <returns>The paths written.</returns>
    private static IReadOnlyList<string> WriteCorpus(string output)
    {
        List<string> written = [];

        foreach (string fixture in Fixtures())
        {
            string directory = Path.Combine(output, fixture);
            Directory.CreateDirectory(directory);

            // Read from JSON specifically: it is the one backend with no lazy-load or
            // native-type-mapping behaviour to opt out of, so the in-memory object is fully
            // materialized before anything writes it.
            object loaded = VersionableFile.LoadDynamic(
                Path.Combine(GoldenCorpus.Root, fixture, fixture + ".json"));

            foreach ((string backend, string extension) in Backends())
            {
                string path = Path.Combine(directory, fixture + extension);
                VersionableFile.Save(loaded, path);

                Assert.True(
                    File.Exists(path),
                    string.Create(CultureInfo.InvariantCulture, $"{fixture}/{backend}: nothing was written"));

                object reloaded = VersionableFile.LoadDynamic(
                    path,
                    options: new VersionableLoadOptions { PreloadAll = true });

                Assert.Equal(Canonical(loaded), Canonical(reloaded));
                written.Add(path);
            }
        }

        return written;
    }

    /// <summary>The fixture directory names, from the corpus index.</summary>
    /// <returns>The names, in index order.</returns>
    private static IReadOnlyList<string> Fixtures()
    {
        using JsonDocument index = JsonDocument.Parse(
            File.ReadAllBytes(Path.Combine(GoldenCorpus.Root, "index.json")));

        return
        [
            .. index.RootElement.GetProperty("fixtures")
                .EnumerateArray()
                .Select(entry => entry.GetProperty("fixture").GetString()!),
        ];
    }

    /// <summary>The backend key to file-extension map, from the corpus index.</summary>
    /// <returns>The map, in index order.</returns>
    private static IReadOnlyList<(string Backend, string Extension)> Backends()
    {
        using JsonDocument index = JsonDocument.Parse(
            File.ReadAllBytes(Path.Combine(GoldenCorpus.Root, "index.json")));

        return
        [
            .. index.RootElement.GetProperty("backends")
                .EnumerateObject()
                .Select(entry => (entry.Name, entry.Value.GetString()!)),
        ];
    }

    /// <summary>Renders an object's wire form deterministically, for comparing two instances.</summary>
    /// <param name="value">A loaded object.</param>
    /// <returns>The canonical rendering.</returns>
    private static string Canonical(object value) => Render(WireValues.Write(value));

    private static string Render(object? wire) =>
        wire switch
        {
            null => "null",
            string text => $"'{text}'",
            bool flag => flag ? "true" : "false",
            IReadOnlyDictionary<string, object?> map =>
                "{" + string.Join(
                    ",",
                    map.OrderBy(entry => entry.Key, StringComparer.Ordinal)
                        .Select(entry => $"{entry.Key}:{Render(entry.Value)}")) + "}",
            IEnumerable<object?> items => "[" + string.Join(",", items.Select(Render)) + "]",
            _ => Convert.ToString(wire, CultureInfo.InvariantCulture) ?? wire.GetType().Name,
        };
}
