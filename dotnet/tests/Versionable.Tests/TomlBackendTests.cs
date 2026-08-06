using System.Text;
using Versionable.Backends;
using Versionable.Backends.Toml;
using Versionable.Errors;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// The TOML backend against real files: what it writes, what it reads, and how it fails.
/// </summary>
[Collection(RegistryCollection.Name)]
public class TomlBackendTests : IDisposable
{
    private readonly string _directory =
        Path.Combine(Path.GetTempPath(), $"versionable-toml-{Path.GetRandomFileName()}");

    public TomlBackendTests()
    {
        GoldenSchemas.EnsureRegistered();
        Directory.CreateDirectory(_directory);
    }

    public void Dispose()
    {
        Directory.Delete(_directory, recursive: true);
        GC.SuppressFinalize(this);
    }

    [Fact]
    public void the_extension_resolves_to_this_backend_without_an_explicit_one()
    {
        Assert.IsType<TomlBackend>(BackendRegistry.Resolve(Path.Combine(_directory, "config.toml")));
        Assert.Contains(".toml", BackendRegistry.RegisteredExtensions(), StringComparer.Ordinal);
    }

    [Fact]
    public void a_saved_object_loads_back_equal()
    {
        EngineNode node = new(
            "root",
            new EngineLeaf("centre", 2.5),
            [new EngineLeaf("left", 1.0), new EngineLeaf("right", 2.0)],
            new Dictionary<string, EngineLeaf>(StringComparer.Ordinal) { ["primary"] = new("primary", 3.5) },
            new HashSet<string>(["alpha", "beta"], StringComparer.Ordinal),
            new EngineLeaf("maybe", 0.25));
        string path = Path.Combine(_directory, "node.toml");

        VersionableFile.Save(node, path);
        EngineNode loaded = VersionableFile.Load<EngineNode>(path);

        Assert.Equal(node.Label, loaded.Label);
        Assert.Equal(node.Leaf.Name, loaded.Leaf.Name);
        Assert.Equal(node.Leaves.Select(leaf => leaf.Name), loaded.Leaves.Select(leaf => leaf.Name));
        Assert.Equal(node.ByName["primary"].Weight, loaded.ByName["primary"].Weight);
        Assert.Equal(node.Tags.Order(StringComparer.Ordinal), loaded.Tags.Order(StringComparer.Ordinal));
        Assert.Equal(node.OptionalLeaf!.Name, loaded.OptionalLeaf!.Name);
    }

    /// <summary>
    /// Nested objects use table syntax, and a list of them uses TOML's array-of-tables headers
    /// rather than one long inline array — which is what the golden corpus holds and what makes a
    /// generated file readable.
    /// </summary>
    [Fact]
    public void nested_objects_are_written_as_tables_and_lists_of_them_as_array_of_tables()
    {
        string path = Path.Combine(_directory, "node.toml");
        VersionableFile.Save(
            new EngineNode(
                "root",
                new EngineLeaf("centre", 2.5),
                [new EngineLeaf("left", 1.0)],
                new Dictionary<string, EngineLeaf>(StringComparer.Ordinal) { ["primary"] = new("primary", 3.5) },
                new HashSet<string>(["alpha"], StringComparer.Ordinal),
                null),
            path);

        string text = File.ReadAllText(path);
        Assert.Contains("[leaf]\n", text, StringComparison.Ordinal);
        Assert.Contains("[leaf.__versionable__]\n", text, StringComparison.Ordinal);
        Assert.Contains("[[leaves]]\n", text, StringComparison.Ordinal);
        Assert.Contains("[leaves.__versionable__]\n", text, StringComparison.Ordinal);

        // A table holding nothing but tables gets no header of its own, as tomlkit renders it.
        Assert.Contains("[byName.primary]\n", text, StringComparison.Ordinal);
        Assert.DoesNotContain("[byName]\n", text, StringComparison.Ordinal);
    }

    /// <summary>
    /// The envelope is the first table in the file, so a reader sees what the file is before it
    /// sees what is in it.
    /// </summary>
    [Fact]
    public void the_envelope_is_the_first_table_and_holds_the_declared_schema()
    {
        string path = Path.Combine(_directory, "leaf.toml");

        VersionableFile.Save(new EngineLeaf("tip", 1.5), path);

        string text = File.ReadAllText(path);
        Assert.Equal(
            """
            name = "tip"
            weight = 1.5

            [__versionable__]
            object = "EngineLeaf"
            version = 1
            hash = "aaaaaa"

            """.ReplaceLineEndings("\n"),
            text);
    }

    /// <summary>
    /// TOML has no null literal, so an absent optional is absent from the file and the type's
    /// default refills it. The <c>optionals</c> golden fixture is the cross-language half of this.
    /// </summary>
    [Fact]
    public void a_null_field_is_omitted_and_comes_back_from_the_default()
    {
        string path = Path.Combine(_directory, "node.toml");
        VersionableFile.Save(
            new EngineNode(
                "root",
                new EngineLeaf("centre", 2.5),
                [],
                new Dictionary<string, EngineLeaf>(StringComparer.Ordinal),
                new HashSet<string>(StringComparer.Ordinal),
                optionalLeaf: null),
            path);

        Assert.DoesNotContain("optionalLeaf", File.ReadAllText(path), StringComparison.Ordinal);
        Assert.Null(VersionableFile.Load<EngineNode>(path).OptionalLeaf);
    }

    [Fact]
    public void the_file_is_utf8_without_a_byte_order_mark()
    {
        string path = Path.Combine(_directory, "leaf.toml");

        VersionableFile.Save(new EngineLeaf("é", 1.5), path);

        byte[] bytes = File.ReadAllBytes(path);
        Assert.NotEqual<byte[]>([0xEF, 0xBB, 0xBF], bytes.Take(3).ToArray());
        Assert.Contains("é", Encoding.UTF8.GetString(bytes), StringComparison.Ordinal);
        Assert.Equal("é", VersionableFile.Load<EngineLeaf>(path).Name);
    }

    /// <summary>
    /// <c>nan</c> and <c>inf</c> are TOML float literals, so no repair pass is needed the way the
    /// JSON backend needs one.
    /// </summary>
    /// <remarks>
    /// Positive infinity is the one where the two languages spell the same value differently:
    /// Tomlyn writes <c>+inf</c> and tomlkit writes <c>inf</c> (as it writes <c>e+</c> where
    /// Tomlyn writes <c>E+</c>). Both are TOML 1.0 §Float — the leading <c>+</c> is optional, not
    /// forbidden — and each reader accepts the other's spelling, which is asserted here by loading
    /// the written file and in <see cref="TomlGoldenCorpusTests"/> by loading Python's. It is also
    /// why no fixture holding a non-finite float is in
    /// <see cref="TomlGoldenCorpusTests.rewriting_a_golden_fixture_reproduces_tomlkits_bytes"/>.
    /// </remarks>
    [Fact]
    public void non_finite_floats_round_trip_as_toml_float_literals()
    {
        string path = Path.Combine(_directory, "leaf.toml");

        VersionableFile.Save(new EngineLeaf("edge", double.NaN), path);
        Assert.Contains("weight = nan", File.ReadAllText(path), StringComparison.Ordinal);
        Assert.True(double.IsNaN(VersionableFile.Load<EngineLeaf>(path).Weight));

        VersionableFile.Save(new EngineLeaf("edge", double.PositiveInfinity), path);
        Assert.Contains("weight = +inf", File.ReadAllText(path), StringComparison.Ordinal);
        Assert.Equal(double.PositiveInfinity, VersionableFile.Load<EngineLeaf>(path).Weight);

        VersionableFile.Save(new EngineLeaf("edge", double.NegativeInfinity), path);
        Assert.Contains("weight = -inf", File.ReadAllText(path), StringComparison.Ordinal);
        Assert.Equal(double.NegativeInfinity, VersionableFile.Load<EngineLeaf>(path).Weight);

        // tomlkit's spelling of the same value, which a Python-written file holds.
        File.WriteAllText(path, "name = \"edge\"\nweight = inf\n");
        Assert.Equal(double.PositiveInfinity, VersionableFile.Load<EngineLeaf>(path).Weight);
    }

    /// <summary>
    /// A file with no <c>[__versionable__]</c> table still loads, at the current version.
    /// </summary>
    /// <remarks>
    /// Python counterpart: <c>TestTomlMissingVersion</c>. This is the hand-written-config case the
    /// format exists for, so it has to work.
    /// </remarks>
    [Fact]
    public void a_hand_written_file_without_an_envelope_loads_at_the_current_version()
    {
        string path = Path.Combine(_directory, "plain.toml");
        File.WriteAllText(path, "name = \"tip\"\nweight = 1.5\n");

        EngineLeaf loaded = VersionableFile.Load<EngineLeaf>(path);

        Assert.Equal("tip", loaded.Name);
        Assert.Equal(1.5, loaded.Weight);
    }

    /// <summary>
    /// TOML's native date and time tokens are accepted, even though nothing writes them.
    /// </summary>
    /// <remarks>
    /// Every temporal type reaches the wire as an ISO 8601 string from its converter — that is
    /// what <c>conformance/golden/temporal/temporal.toml</c> holds. A hand-written config may
    /// still use TOML's own tokens, and they are handed on as the same ISO 8601 text so the
    /// ordinary converters accept them. Python cannot read such a file (tomlkit hands back a
    /// <c>datetime</c> and <c>fromisoformat</c> rejects it); the divergence only ever adds files
    /// that load.
    /// </remarks>
    [Fact]
    public void a_native_toml_datetime_is_read_as_its_iso_8601_text()
    {
        string path = Path.Combine(_directory, "temporal.toml");
        File.WriteAllText(
            path,
            """
            naive = 2026-08-05T14:30:15.123456
            aware = 2026-08-05T14:30:15.123456-05:00
            day = 2026-08-05
            clock = 23:59:58.500000
            elapsed = 90061.5

            [__versionable__]
            object = "GoldenTemporal"
            version = 1
            hash = "3856a2"

            """);

        GoldenTemporal loaded = VersionableFile.Load<GoldenTemporal>(path);

        Assert.Equal(new DateTime(2026, 8, 5, 14, 30, 15, DateTimeKind.Unspecified).AddTicks(1234560), loaded.Naive);
        Assert.Equal(new TimeSpan(-5, 0, 0), loaded.Aware.Offset);
        Assert.Equal(new DateOnly(2026, 8, 5), loaded.Day);
        Assert.Equal(new TimeOnly(23, 59, 58, 500), loaded.Clock);
    }

    /// <summary>The pre-0.2 <c>__json__</c> wrapper is still unwrapped on read.</summary>
    [Fact]
    public void the_legacy_json_wrapper_is_still_read()
    {
        string path = Path.Combine(_directory, "legacy.toml");
        File.WriteAllText(
            path,
            """
            label = "legacy"

            [__versionable__]
            object = "EngineNode"
            version = 1
            hash = "aaaaaa"

            [leaf]
            __json__ = "{\"__versionable__\": {\"object\": \"EngineLeaf\", \"version\": 1}, \"name\": \"tip\", \"weight\": 1.5}"

            """);

        EngineNode loaded = VersionableFile.Load<EngineNode>(path);

        Assert.Equal("tip", loaded.Leaf.Name);
        Assert.Equal(1.5, loaded.Leaf.Weight);
    }

    [Fact]
    public void a_malformed_file_reports_the_path()
    {
        string path = Path.Combine(_directory, "broken.toml");
        File.WriteAllText(path, "name = \n");

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.Load<EngineLeaf>(path));

        Assert.Contains("broken.toml", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_missing_file_reports_the_path()
    {
        string path = Path.Combine(_directory, "absent.toml");

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.Load<EngineLeaf>(path));

        Assert.Contains("absent.toml", error.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// A file that declares a wire format this build does not know is refused rather than misread.
    /// </summary>
    [Fact]
    public void an_unknown_format_is_refused()
    {
        string path = Path.Combine(_directory, "future.toml");
        File.WriteAllText(
            path,
            """
            name = "tip"
            weight = 1.5

            [__versionable__]
            object = "EngineLeaf"
            version = 1
            format = "2"

            """);

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.Load<EngineLeaf>(path));

        Assert.Contains("format", error.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// TOML integers are signed 64-bit, so a value above that range is refused with an explanation
    /// rather than silently widened to a float.
    /// </summary>
    [Fact]
    public void an_integer_wider_than_toml_allows_is_refused()
    {
        BackendException error = Assert.Throws<BackendException>(
            () => TomlEmitterProbe.Render(ulong.MaxValue));

        Assert.Contains("64-bit", error.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// A null inside an array is refused: dropping it would renumber the elements after it, and
    /// TOML has nothing to put in its place.
    /// </summary>
    [Fact]
    public void a_null_inside_an_array_is_refused()
    {
        BackendException error = Assert.Throws<BackendException>(
            () => TomlEmitterProbe.Safe(new List<object?> { "a", null }));

        Assert.Contains("null inside an array", error.Message, StringComparison.Ordinal);
    }

    /// <summary>Reaches the two edges of the wire layer that no schema in the suite can produce.</summary>
    private static class TomlEmitterProbe
    {
        internal static string Render(object? value) => TomlEmitter.RenderKeyValue("k", value);

        internal static object? Safe(object? value) => TomlWire.ToTomlSafe(value);
    }
}
