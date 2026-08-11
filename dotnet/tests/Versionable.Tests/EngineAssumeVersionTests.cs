using Versionable.Engine;
using Versionable.Errors;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// <see cref="VersionableLoadOptions.AssumeVersion"/>: what a file that records no version is
/// taken to be.
/// </summary>
/// <remarks>
/// Python counterpart: the <c>assumeVersion</c> branch of <c>load()</c> in
/// <c>src/versionable/_api.py</c>, and the <c>test_assumeVersionOverride</c> case each of the
/// JSON, YAML, and TOML backend suites carries. Only the three text formats are covered here, as
/// in Python: HDF5 files are only ever produced by a writer, and every writer stamps a version.
/// <para>
/// The three rules the option has, all pinned below: it applies <em>only</em> when the file
/// records no version, it applies to the root object only, and an assumed version newer than the
/// type is refused exactly as a recorded one would be.
/// </para>
/// </remarks>
[Collection(RegistryCollection.Name)]
public class EngineAssumeVersionTests : IDisposable
{
    private readonly string _directory =
        Path.Combine(Path.GetTempPath(), $"versionable-assume-{Path.GetRandomFileName()}");

    public EngineAssumeVersionTests()
    {
        GoldenSchemas.EnsureRegistered();
        Directory.CreateDirectory(_directory);
    }

    public void Dispose()
    {
        Directory.Delete(_directory, recursive: true);
        GC.SuppressFinalize(this);
    }

    [Theory]
    [InlineData(".json")]
    [InlineData(".yaml")]
    [InlineData(".toml")]
    public void an_assumed_version_applies_the_migrations_a_version_less_file_needs(string extension)
    {
        string path = VersionLess(extension);

        PolyOwl loaded = VersionableFile.Load<PolyOwl>(path, options: new VersionableLoadOptions { AssumeVersion = 1 });

        Assert.Equal("HOOT", loaded.Sound);
    }

    [Theory]
    [InlineData(".json")]
    [InlineData(".yaml")]
    [InlineData(".toml")]
    public void a_version_less_file_with_no_assumption_is_treated_as_current_and_warned_about(string extension)
    {
        // Current is the only guess that loads at all, and it silently skips every migration —
        // which is right for a hand-written file and wrong for one written by older code, so the
        // warning is what tells the two apart.
        string path = VersionLess(extension);

        List<string> warnings = [];
        PolyOwl loaded = Capturing(warnings, () => VersionableFile.Load<PolyOwl>(path));

        Assert.Equal("hoot", loaded.Sound);
        Assert.Contains(warnings, message => message.Contains("No version found", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData(".json")]
    [InlineData(".yaml")]
    [InlineData(".toml")]
    public void an_assumption_is_silent_where_the_warning_would_have_been(string extension)
    {
        string path = VersionLess(extension);

        List<string> warnings = [];
        Capturing(warnings, () => VersionableFile.Load<PolyOwl>(
            path, options: new VersionableLoadOptions { AssumeVersion = 1 }));

        Assert.DoesNotContain(warnings, message => message.Contains("No version found", StringComparison.Ordinal));
    }

    [Fact]
    public void a_version_the_file_records_wins_over_the_assumption()
    {
        // The file says 2, so no migration runs and `call` is an unknown field. If the assumption
        // won instead, the rename would fire and overwrite `sound` with the value under `call`.
        string path = Path.Combine(_directory, "current.json");
        File.WriteAllText(
            path,
            """
            {
              "__versionable__": {"object": "PolyOwl", "version": 2, "hash": ""},
              "name": "Hedwig",
              "sound": "SET",
              "call": "IGNORED"
            }
            """);

        PolyOwl loaded = VersionableFile.Load<PolyOwl>(path, options: new VersionableLoadOptions { AssumeVersion = 1 });

        Assert.Equal("SET", loaded.Sound);
    }

    [Fact]
    public void an_assumed_version_newer_than_the_type_is_refused()
    {
        string path = VersionLess(".json");

        VersionException error = Assert.Throws<VersionException>(
            () => VersionableFile.Load<PolyOwl>(path, options: new VersionableLoadOptions { AssumeVersion = 9 }));

        Assert.Contains("newer than class version", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void an_assumption_does_not_reach_a_nested_object()
    {
        // Python's assumeVersion is an argument to load(), consumed where the root's version is
        // decided; a nested object reads its own envelope and, finding no version, assumes its own
        // current version and warns. Propagating the root's assumption downward would migrate a
        // nested value from a version its own file never claimed.
        string path = Path.Combine(_directory, "nested.json");
        File.WriteAllText(
            path,
            """
            {
              "__versionable__": {"object": "PolyZoo", "version": 1, "hash": ""},
              "animals": [{"__versionable__": {"object": "PolyOwl"}, "name": "Hedwig", "call": "HOOT"}],
              "star": null
            }
            """);

        List<string> warnings = [];
        PolyZoo loaded = Capturing(
            warnings,
            () => VersionableFile.Load<PolyZoo>(path, options: new VersionableLoadOptions { AssumeVersion = 1 }));

        Assert.Equal("hoot", Assert.IsType<PolyOwl>(loaded.Animals[0]).Sound);
        Assert.Contains(warnings, message => message.Contains("Nested PolyOwl", StringComparison.Ordinal));
    }

    private static T Capturing<T>(List<string> warnings, Func<T> action)
    {
        void Capture(string message) => warnings.Add(message);
        VersionableLog.Warning += Capture;
        try
        {
            return action();
        }
        finally
        {
            VersionableLog.Warning -= Capture;
        }
    }

    /// <summary>A version-1 <c>PolyOwl</c> file whose envelope records no version.</summary>
    private string VersionLess(string extension)
    {
        string path = Path.Combine(_directory, $"owl{extension}");
        File.WriteAllText(
            path,
            extension switch
            {
                ".json" => """
                    {
                      "__versionable__": {"object": "PolyOwl", "hash": ""},
                      "name": "Hedwig",
                      "call": "HOOT"
                    }
                    """,
                ".yaml" => """
                    __versionable__:
                      object: PolyOwl
                      hash: ''
                    name: Hedwig
                    call: HOOT
                    """,
                ".toml" => """
                    name = "Hedwig"
                    call = "HOOT"

                    [__versionable__]
                    object = "PolyOwl"
                    hash = ""
                    """,
                _ => throw new ArgumentOutOfRangeException(nameof(extension), extension, "No hand-authored form."),
            });

        return path;
    }
}
