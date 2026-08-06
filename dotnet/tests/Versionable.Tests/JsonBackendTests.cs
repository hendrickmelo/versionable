using System.Text;
using System.Text.Json;
using Versionable.Backends;
using Versionable.Backends.Json;
using Versionable.Errors;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// The JSON backend against real files: what it writes, what it reads, and how it fails.
/// </summary>
[Collection(RegistryCollection.Name)]
public class JsonBackendTests : IDisposable
{
    private readonly string _directory =
        Path.Combine(Path.GetTempPath(), $"versionable-json-{Path.GetRandomFileName()}");

    public JsonBackendTests()
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
    public void a_saved_object_loads_back_equal()
    {
        EngineNode node = new(
            "root",
            new EngineLeaf("centre", 2.5),
            [new EngineLeaf("left", 1.0), new EngineLeaf("right", 2.0)],
            new Dictionary<string, EngineLeaf>(StringComparer.Ordinal) { ["primary"] = new("primary", 3.5) },
            new HashSet<string>(["alpha", "beta"], StringComparer.Ordinal),
            new EngineLeaf("maybe", 0.25));
        string path = Path.Combine(_directory, "node.json");

        VersionableFile.Save(node, path);
        EngineNode loaded = VersionableFile.Load<EngineNode>(path);

        Assert.Equal(node.Label, loaded.Label);
        Assert.Equal(node.Leaf.Name, loaded.Leaf.Name);
        Assert.Equal(node.Leaf.Weight, loaded.Leaf.Weight);
        Assert.Equal(node.Leaves.Select(leaf => leaf.Name), loaded.Leaves.Select(leaf => leaf.Name));
        Assert.Equal(node.ByName["primary"].Weight, loaded.ByName["primary"].Weight);
        Assert.Equal(node.Tags.Order(StringComparer.Ordinal), loaded.Tags.Order(StringComparer.Ordinal));
        Assert.Equal(node.OptionalLeaf!.Name, loaded.OptionalLeaf!.Name);
    }

    [Fact]
    public void the_envelope_is_written_first_and_holds_the_declared_schema()
    {
        string path = Path.Combine(_directory, "leaf.json");

        VersionableFile.Save(new EngineLeaf("tip", 1.5), path);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(path));
        JsonProperty first = document.RootElement.EnumerateObject().First();
        Assert.Equal(VersionableEnvelope.WrappedKey, first.Name);
        Assert.Equal("EngineLeaf", first.Value.GetProperty("object").GetString());
        Assert.Equal(1, first.Value.GetProperty("version").GetInt32());
        Assert.Equal("aaaaaa", first.Value.GetProperty("hash").GetString());
    }

    [Fact]
    public void the_file_is_utf8_without_a_byte_order_mark_and_indented_like_python()
    {
        string path = Path.Combine(_directory, "leaf.json");

        VersionableFile.Save(new EngineLeaf("é", 1.5), path);

        byte[] bytes = File.ReadAllBytes(path);
        Assert.NotEqual<byte[]>([0xEF, 0xBB, 0xBF], bytes.Take(3).ToArray());

        string text = Encoding.UTF8.GetString(bytes);
        Assert.StartsWith(
            "{\n  \"__versionable__\": {\n    \"object\":",
            text.ReplaceLineEndings("\n"),
            StringComparison.Ordinal);
    }

    [Fact]
    public void fields_are_written_in_declaration_order()
    {
        string path = Path.Combine(_directory, "settings.json");

        VersionableFile.Save(new EngineSettings("named", 1234, "needed"), path);

        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(path));
        Assert.Equal(
            [VersionableEnvelope.WrappedKey, "name", "port", "required"],
            document.RootElement.EnumerateObject().Select(property => property.Name));
    }

    [Fact]
    public void a_file_written_by_python_in_the_legacy_layout_still_loads()
    {
        // 0.1.x wrote the envelope as top-level dunder keys. Nothing writes them now; everything
        // still reads them.
        string path = Path.Combine(_directory, "legacy.json");
        File.WriteAllText(
            path,
            """
            {
              "__OBJECT__": "EngineLeaf",
              "__VERSION__": 1,
              "__HASH__": "aaaaaa",
              "name": "from-0.1.x",
              "weight": 2.0
            }
            """,
            new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

        EngineLeaf leaf = VersionableFile.Load<EngineLeaf>(path);

        Assert.Equal("from-0.1.x", leaf.Name);
        Assert.Equal(2.0, leaf.Weight);
    }

    [Fact]
    public void the_dynamic_entry_point_resolves_the_type_from_the_envelope()
    {
        string path = Path.Combine(_directory, "leaf.json");
        VersionableFile.Save(new EngineLeaf("tip", 1.5), path);

        object loaded = VersionableFile.Load(path);

        Assert.Equal("tip", Assert.IsType<EngineLeaf>(loaded).Name);
    }

    [Fact]
    public void a_dynamic_load_of_an_unregistered_name_is_refused()
    {
        string path = Path.Combine(_directory, "stranger.json");
        File.WriteAllText(
            path,
            """{"__versionable__": {"object": "NotRegistered", "version": 1, "hash": "000000"}}""");

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.Load(path));

        Assert.Contains("Unknown object type 'NotRegistered'", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void malformed_json_is_reported_as_a_backend_failure()
    {
        string path = Path.Combine(_directory, "broken.json");
        File.WriteAllText(path, "{\"name\": ");

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.Load<EngineLeaf>(path));

        Assert.Contains("Failed to read JSON", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_json_document_that_is_not_an_object_is_reported()
    {
        string path = Path.Combine(_directory, "array.json");
        File.WriteAllText(path, "[1, 2, 3]");

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.Load<EngineLeaf>(path));

        Assert.Contains("Expected a JSON object", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_missing_file_is_reported_as_a_backend_failure()
    {
        BackendException error = Assert.Throws<BackendException>(
            () => VersionableFile.Load<EngineLeaf>(Path.Combine(_directory, "absent.json")));

        Assert.Contains("Failed to read JSON", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void an_unclaimed_extension_names_the_ones_that_are_claimed()
    {
        BackendException error = Assert.Throws<BackendException>(
            () => VersionableFile.Save(new EngineLeaf("tip", 1.5), Path.Combine(_directory, "leaf.unknown")));

        Assert.Contains(".json", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void an_explicit_backend_overrides_the_extension()
    {
        string path = Path.Combine(_directory, "leaf.unknown");

        VersionableFile.Save(new EngineLeaf("tip", 1.5), path, new JsonBackend());
        EngineLeaf loaded = VersionableFile.Load<EngineLeaf>(path, new JsonBackend());

        Assert.Equal("tip", loaded.Name);
    }

    [Fact]
    public void integers_survive_the_round_trip_as_integers()
    {
        // Regression: the number reader used a conditional expression whose best common type was
        // double, so every integer in every file was silently widened — which showed up as a
        // Literal[1, 2, 3] field rejecting the 3 in its own golden file.
        string path = Path.Combine(_directory, "numbers.json");
        EngineNumbers numbers = new(long.MaxValue, int.MinValue, ulong.MaxValue, 9.8125);

        VersionableFile.Save(numbers, path);
        EngineNumbers loaded = VersionableFile.Load<EngineNumbers>(path);

        Assert.Equal(long.MaxValue, loaded.Big);
        Assert.Equal(int.MinValue, loaded.Small);
        Assert.Equal(ulong.MaxValue, loaded.Huge);
        Assert.Equal(9.8125, loaded.Ratio);
    }

    [Fact]
    public void an_integer_past_the_exact_range_of_a_double_keeps_every_bit()
    {
        // 2^53 + 1 is the first integer a double cannot represent, so a reader that went through
        // double would hand back 2^53 and lose the odd bit without any error.
        const long beyondDoublePrecision = (1L << 53) + 1;
        string path = Path.Combine(_directory, "precise.json");

        VersionableFile.Save(new EngineNumbers(beyondDoublePrecision, 0, 0, 0.5), path);

        Assert.Contains("9007199254740993", File.ReadAllText(path), StringComparison.Ordinal);
        Assert.Equal(beyondDoublePrecision, VersionableFile.Load<EngineNumbers>(path).Big);
    }

    [Fact]
    public void a_ulong_above_long_maxvalue_is_read_without_overflowing()
    {
        string path = Path.Combine(_directory, "unsigned.json");
        File.WriteAllText(
            path,
            """
            {
              "__versionable__": {"object": "EngineNumbers", "version": 1, "hash": "333333"},
              "big": 0, "small": 0, "huge": 18446744073709551615, "ratio": 0.5
            }
            """);

        Assert.Equal(ulong.MaxValue, VersionableFile.Load<EngineNumbers>(path).Huge);
    }

    [Fact]
    public void non_finite_numbers_are_written_the_way_python_writes_them()
    {
        // json.dumps emits bare NaN / Infinity / -Infinity, which is not RFC 8259. Matching it is
        // what lets Python read the file; see NonFiniteJson.
        string path = Path.Combine(_directory, "nonfinite.json");

        VersionableFile.Save(new EngineNumbers(0, 0, 0, double.NaN), path);
        Assert.Contains("\"ratio\": NaN", File.ReadAllText(path).ReplaceLineEndings("\n"), StringComparison.Ordinal);

        VersionableFile.Save(new EngineNumbers(0, 0, 0, double.NegativeInfinity), path);
        Assert.Contains("\"ratio\": -Infinity", File.ReadAllText(path), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("NaN", double.NaN)]
    [InlineData("Infinity", double.PositiveInfinity)]
    [InlineData("-Infinity", double.NegativeInfinity)]
    public void a_python_written_non_finite_number_is_read_back(string token, double expected)
    {
        string path = Path.Combine(_directory, "nonfinite.json");
        File.WriteAllText(
            path,
            $$"""
            {
              "__versionable__": {"object": "EngineNumbers", "version": 1, "hash": "333333"},
              "big": 0, "small": 0, "huge": 0, "ratio": {{token}}
            }
            """);

        Assert.Equal(expected, VersionableFile.Load<EngineNumbers>(path).Ratio);
    }

    [Fact]
    public void the_non_finite_repair_leaves_string_values_alone()
    {
        // The repair pass rewrites bare tokens only. A field whose value is the word NaN is a
        // string, and rewriting it would corrupt the very file the pass exists to rescue.
        string path = Path.Combine(_directory, "nan-named.json");
        File.WriteAllText(
            path,
            """
            {
              "__versionable__": {"object": "EngineLeaf", "version": 1, "hash": "aaaaaa"},
              "name": "NaN", "weight": NaN
            }
            """);

        EngineLeaf loaded = VersionableFile.Load<EngineLeaf>(path);

        Assert.Equal("NaN", loaded.Name);
        Assert.Equal(double.NaN, loaded.Weight);
    }

    [Fact]
    public void a_corrupt_envelope_table_is_reported_rather_than_ignored()
    {
        // Python raises here too (_json_backend.py). Falling through to the flat layout would
        // report a corrupt envelope as a missing one and load the file as version-less.
        string path = Path.Combine(_directory, "corrupt.json");
        File.WriteAllText(path, """{"__versionable__": "GoldenInner", "name": "x", "weight": 1.0}""");

        BackendException error = Assert.Throws<BackendException>(() => VersionableFile.Load<EngineLeaf>(path));

        Assert.Contains("must hold the envelope table", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void json_stores_nothing_natively_so_every_value_goes_through_the_walker()
    {
        // Python's JsonBackend.nativeTypes is an empty set for the same reason.
        Assert.Empty(new JsonBackend().NativeTypes);
    }

    [Fact]
    public void json_never_reports_a_lazy_field()
    {
        string path = Path.Combine(_directory, "leaf.json");
        VersionableFile.Save(new EngineLeaf("tip", 1.5), path);

        BackendLoadResult result = new JsonBackend().Load(path, new BackendLoadOptions { MetadataOnly = true });

        Assert.Null(result.LazyFields);
        Assert.Equal("tip", result.Fields["name"]);
    }
}
