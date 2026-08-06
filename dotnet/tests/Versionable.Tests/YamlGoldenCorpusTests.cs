using System.Globalization;
using System.Numerics;
using System.Numerics.Tensors;
using System.Text.Json;
using Versionable.Backends.Yaml;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// The YAML half of the golden-corpus contract: the <c>.yaml</c> bytes Python wrote, read here and
/// compared against the same manifests the Python suite compares against.
/// </summary>
/// <remarks>
/// Python counterpart: <c>tests/test_golden_corpus.py</c>. Structurally a mirror of
/// <see cref="JsonGoldenCorpusTests"/> and deliberately so — the two backends are handed the same
/// values by the same walker, and every place the assertions differ is a place the <em>format</em>
/// differs, which is precisely what is worth testing:
/// <list type="bullet">
/// <item>arrays are read out of an embedded JSON string rather than a nested mapping
/// (<see cref="EmbeddedNdarrayJson"/>);</item>
/// <item>the envelope sits at the end of the file rather than the start;</item>
/// <item>every temporal and decimal value is quoted in the file, because PyYAML would otherwise
/// resolve it to a date or a float — the read side of that is <see cref="YamlSchema"/>.</item>
/// </list>
/// <para>
/// Nothing here regenerates anything; the committed files are the contract. What is <em>not</em>
/// repeated from the JSON suite is the schema-mirror theory (that the C# hashes match the
/// manifests): that is a property of the types, not of a backend, and asserting it twice would
/// only mean fixing it twice.
/// </para>
/// </remarks>
[Collection(RegistryCollection.Name)]
public class YamlGoldenCorpusTests
{
    private static readonly string _goldenRoot = FindGoldenRoot();

    public YamlGoldenCorpusTests()
    {
        GoldenSchemas.EnsureRegistered();
    }

    /// <summary>
    /// Every fixture in the corpus has a <c>.yaml</c> file and it is read by this backend.
    /// </summary>
    /// <remarks>
    /// The guard against a fixture being added to the corpus and quietly skipped here: the
    /// per-fixture tests below name their files, so only this one notices a twelfth directory.
    /// </remarks>
    [Fact]
    public void the_corpus_declares_a_yaml_file_for_every_fixture_and_all_of_them_parse()
    {
        using JsonDocument index = JsonDocument.Parse(
            File.ReadAllBytes(Path.Combine(_goldenRoot, "index.json")));

        int seen = 0;
        foreach (JsonElement entry in index.RootElement.GetProperty("fixtures").EnumerateArray())
        {
            string fixture = entry.GetProperty("fixture").GetString()!;
            JsonElement manifest = Manifest(fixture);

            foreach (string file in FilesOf(manifest))
            {
                string path = Path.Combine(_goldenRoot, fixture, file);
                Assert.True(File.Exists(path), $"{fixture}: {file} is missing from the corpus.");

                object loaded = VersionableFile.LoadDynamic(path);
                Assert.Equal(
                    manifest.GetProperty("serializationName").GetString(),
                    VersionableRegistry.TryGetByType(loaded.GetType(), out VersionableMetadata? metadata)
                        ? metadata.Name
                        : null);
                seen++;
            }
        }

        // Eleven fixtures plus the two older migration-chain files.
        Assert.Equal(13, seen);
    }

    // ------------------------------------------------------------------
    // Values, fixture by fixture
    // ------------------------------------------------------------------

    [Fact]
    public void containers_loads_every_container_form_python_wrote()
    {
        JsonElement values = Values("containers");
        GoldenContainers loaded = Load<GoldenContainers>("containers", "containers.yaml");

        Assert.Equal(Strings(values.GetProperty("names")), loaded.Names);
        Assert.Equal(Doubles(values.GetProperty("readings")), loaded.Readings);
        Assert.Equal(Ints(values.GetProperty("counts")), loaded.Counts);
        Assert.Equal(
            values.GetProperty("flags").EnumerateArray().Select(item => item.GetBoolean()),
            loaded.Flags);

        Assert.Equal(
            Pairs(values.GetProperty("lookup"))
                .ToDictionary(pair => pair.Key.GetString()!, pair => pair.Value.GetInt32()),
            loaded.Lookup);

        // The integer keys are quoted in the file ('1', '2'), because PyYAML would otherwise read
        // them back as integers and the mapping would no longer be str-keyed on the Python side.
        Assert.Equal(
            Pairs(values.GetProperty("byIndex"))
                .ToDictionary(pair => pair.Key.GetInt32(), pair => pair.Value.GetString()!),
            loaded.ByIndex);

        Assert.Equal(
            Strings(values.GetProperty("tags").GetProperty("$set")).Order(StringComparer.Ordinal),
            loaded.Tags.Order(StringComparer.Ordinal));
        Assert.Equal(Ints(values.GetProperty("ids").GetProperty("$set")).Order(), loaded.Ids.Order());

        Assert.Equal(Ints(values.GetProperty("pair").GetProperty("$tuple")), [loaded.Pair.First, loaded.Pair.Second]);
        Assert.Equal(Doubles(values.GetProperty("samples").GetProperty("$tuple")), loaded.Samples);

        Assert.Equal(
            values.GetProperty("matrix").EnumerateArray().Select(row => Doubles(row).ToList()),
            loaded.Matrix);
        Assert.Equal(
            Pairs(values.GetProperty("grouped")).ToDictionary(
                pair => pair.Key.GetString()!, pair => Doubles(pair.Value).ToList()),
            loaded.Grouped);
    }

    [Fact]
    public void enums_load_as_their_member_values()
    {
        JsonElement values = Values("enums");
        GoldenEnums loaded = Load<GoldenEnums>("enums", "enums.yaml");

        Assert.Equal(Colour(values.GetProperty("colour")), loaded.Colour);
        Assert.Equal(
            (GoldenPriority)values.GetProperty("priority").GetProperty("$enum").GetProperty("value").GetInt32(),
            loaded.Priority);
        Assert.Equal(values.GetProperty("palette").EnumerateArray().Select(Colour), loaded.Palette);
        Assert.Equal(
            Pairs(values.GetProperty("byName"))
                .ToDictionary(pair => pair.Key.GetString()!, pair => Colour(pair.Value)),
            loaded.ByName);
    }

    [Fact]
    public void literals_load_and_keep_the_member_kind_python_chose()
    {
        JsonElement values = Values("literals");
        GoldenLiterals loaded = Load<GoldenLiterals>("literals", "literals.yaml");

        Assert.Equal(values.GetProperty("mode").GetString(), loaded.Mode);
        Assert.Equal(values.GetProperty("level").GetInt32(), loaded.Level);

        // `flag: false` has to stay a bool and `tag: 1` an integer. Both are bare tokens in the
        // file, so this is the schema resolver's answer, not a quoted string's.
        Assert.Equal(values.GetProperty("flag").GetBoolean(), loaded.Flag);
        Assert.Equal(values.GetProperty("tag").GetInt64(), loaded.Tag);
    }

    [Fact]
    public void nested_objects_load_from_their_own_envelopes()
    {
        JsonElement values = Values("nested");
        GoldenNested loaded = Load<GoldenNested>("nested", "nested.yaml");

        Assert.Equal(values.GetProperty("label").GetString(), loaded.Label);
        AssertInner(values.GetProperty("inner"), loaded.Inner);

        JsonElement[] points = [.. values.GetProperty("points").EnumerateArray()];
        Assert.Equal(points.Length, loaded.Points.Count);
        for (int index = 0; index < points.Length; index++)
        {
            AssertInner(points[index], loaded.Points[index]);
        }

        foreach ((JsonElement key, JsonElement value) in Pairs(values.GetProperty("byName")))
        {
            AssertInner(value, loaded.ByName[key.GetString()!]);
        }

        Assert.NotNull(loaded.OptionalInner);
        AssertInner(values.GetProperty("optionalInner"), loaded.OptionalInner);
    }

    [Fact]
    public void a_base_typed_collection_comes_back_as_the_subclasses_the_file_names()
    {
        JsonElement values = Values("polymorphic");
        GoldenPolymorphic loaded = Load<GoldenPolymorphic>("polymorphic", "polymorphic.yaml");

        JsonElement[] expected = [.. values.GetProperty("shapes").EnumerateArray()];
        Assert.Equal(expected.Length, loaded.Shapes.Count);

        for (int index = 0; index < expected.Length; index++)
        {
            JsonElement declared = expected[index].GetProperty("$object");
            JsonElement fields = declared.GetProperty("fields");
            GoldenShape shape = loaded.Shapes[index];

            Assert.Equal(fields.GetProperty("label").GetString(), shape.Label);
            switch (declared.GetProperty("name").GetString())
            {
                case "GoldenCircle":
                    Assert.Equal(fields.GetProperty("radius").GetDouble(), Assert.IsType<GoldenCircle>(shape).Radius);
                    break;
                case "GoldenSquare":
                    Assert.Equal(fields.GetProperty("side").GetDouble(), Assert.IsType<GoldenSquare>(shape).Side);
                    break;
                default:
                    Assert.Fail($"unexpected shape {declared.GetProperty("name")}");
                    break;
            }
        }
    }

    [Fact]
    public void the_current_version_of_the_migration_fixture_loads_unmigrated()
    {
        AssertWorker(Values("migration-chain"), Load<GoldenWorker>("migration-chain", "migration-chain.yaml"));
    }

    [Fact]
    public void older_files_migrate_forward_to_the_current_schema()
    {
        JsonElement manifest = Manifest("migration-chain");

        foreach (JsonElement source in manifest.GetProperty("migrationSources").EnumerateArray())
        {
            string file = source.GetProperty("files").GetProperty("yaml").GetString()!;
            AssertWorker(source.GetProperty("expected"), Load<GoldenWorker>("migration-chain", file));
        }
    }

    [Fact]
    public void scalars_load_every_scalar_token()
    {
        JsonElement values = Values("scalars");
        GoldenScalars loaded = Load<GoldenScalars>("scalars", "scalars.yaml");

        Assert.Equal(values.GetProperty("text").GetString(), loaded.Text);
        Assert.Equal(values.GetProperty("count").GetInt32(), loaded.Count);
        Assert.Equal(values.GetProperty("ratio").GetDouble(), loaded.Ratio);
        Assert.Equal(values.GetProperty("enabled").GetBoolean(), loaded.Enabled);

        double[] parts = [.. Doubles(values.GetProperty("phase").GetProperty("$complex"))];
        Assert.Equal(new Complex(parts[0], parts[1]), loaded.Phase);

        // Base64 goes through untouched: `AAH+/w==` is a plain scalar in the file and has to come
        // back as that exact string rather than as anything the schema might make of it.
        Assert.Equal(
            Convert.FromBase64String(values.GetProperty("blob").GetProperty("$bytes").GetString()!),
            loaded.Blob);
    }

    [Fact]
    public void optionals_load_including_a_null_and_a_multi_member_union()
    {
        JsonElement values = Values("optionals");
        GoldenOptionals loaded = Load<GoldenOptionals>("optionals", "optionals.yaml");

        Assert.Equal(values.GetProperty("present").GetString(), loaded.Present);
        Assert.Equal(JsonValueKind.Null, values.GetProperty("absent").ValueKind);
        Assert.Null(loaded.Absent);
        Assert.Equal(values.GetProperty("maybeCount").GetInt32(), loaded.MaybeCount);
        Assert.Equal(Decimal(values.GetProperty("money")), loaded.Money);
        Assert.Equal(values.GetProperty("either").GetString(), loaded.Either);
    }

    [Fact]
    public void temporal_values_load_with_microsecond_precision()
    {
        JsonElement values = Values("temporal");
        GoldenTemporal loaded = Load<GoldenTemporal>("temporal", "temporal.yaml");

        // Every one of these is a quoted scalar in the file. Unquoted, PyYAML resolves the first
        // four to datetime/date objects and the schema here leaves them as text; either way the
        // converters see a string, which is the point of the quoting.
        Assert.Equal(
            DateTime.Parse(
                values.GetProperty("naive").GetProperty("$datetime").GetString()!,
                CultureInfo.InvariantCulture,
                DateTimeStyles.None),
            loaded.Naive);
        Assert.Equal(
            DateTimeOffset.Parse(
                values.GetProperty("aware").GetProperty("$datetime").GetString()!,
                CultureInfo.InvariantCulture),
            loaded.Aware);
        Assert.Equal(
            DateOnly.Parse(values.GetProperty("day").GetProperty("$date").GetString()!, CultureInfo.InvariantCulture),
            loaded.Day);
        Assert.Equal(
            TimeOnly.Parse(values.GetProperty("clock").GetProperty("$time").GetString()!, CultureInfo.InvariantCulture),
            loaded.Clock);
        Assert.Equal(
            TimeSpan.FromSeconds(values.GetProperty("elapsed").GetProperty("$timedeltaSeconds").GetDouble()),
            loaded.Elapsed);
    }

    [Fact]
    public void stdlib_converter_types_load()
    {
        JsonElement values = Values("stdlib");
        GoldenStdlib loaded = Load<GoldenStdlib>("stdlib", "stdlib.yaml");

        Assert.Equal(PathValue(values.GetProperty("filePath")), loaded.FilePath.Value);
        Assert.Equal(PathValue(values.GetProperty("posixPath")), loaded.PosixPath.Value);

        // `C:\Devices\probe.cfg` is a bare plain scalar in the file: YAML gives no meaning to a
        // backslash outside double quotes, so it survives with no unescaping.
        Assert.Equal(PathValue(values.GetProperty("windowsPath")), loaded.WindowsPath.Value);

        Assert.Equal(Decimal(values.GetProperty("amount")), loaded.Amount);
        Assert.Equal(Guid.Parse(values.GetProperty("deviceId").GetProperty("$uuid").GetString()!), loaded.DeviceId);
        Assert.Equal(
            values.GetProperty("serialPattern").GetProperty("$pattern").GetString(),
            loaded.SerialPattern.ToString());
    }

    [Fact]
    public void arrays_load_from_the_npz_payload_embedded_as_json()
    {
        JsonElement values = Values("arrays");
        GoldenArrays loaded = Load<GoldenArrays>("arrays", "arrays.yaml");

        AssertTensor(values.GetProperty("signal"), "float64", loaded.Signal, element => element.GetDouble());
        AssertTensor(values.GetProperty("weights"), "float32", loaded.Weights, element => element.GetSingle());
        AssertTensor(values.GetProperty("counts"), "int32", loaded.Counts, element => element.GetInt32());
        AssertTensor(values.GetProperty("image"), "uint8", loaded.Image, element => element.GetByte());
        AssertTensor(values.GetProperty("mask"), "bool", loaded.Mask, element => element.GetBoolean());
        AssertTensor(values.GetProperty("matrix"), "float64", loaded.Matrix, element => element.GetDouble());

        JsonElement[] traces = [.. values.GetProperty("traces").EnumerateArray()];
        Assert.Equal(traces.Length, loaded.Traces.Count);
        for (int index = 0; index < traces.Length; index++)
        {
            AssertTensor(traces[index], "float64", loaded.Traces[index], element => element.GetDouble());
        }

        foreach ((JsonElement key, JsonElement value) in Pairs(values.GetProperty("channels")))
        {
            AssertTensor(value, "float64", loaded.Channels[key.GetString()!], element => element.GetDouble());
        }
    }

    /// <summary>
    /// Python's array fixture holds embedded JSON, not YAML mappings — the shape the C# writer has
    /// to reproduce.
    /// </summary>
    /// <remarks>
    /// Asserted on the raw bytes because it is the one structural difference between this
    /// backend's wire form and the JSON backend's, and every other test in this file would pass
    /// just as happily if the arrays were written as ordinary nested mappings — which Python could
    /// then not read.
    /// <para>
    /// This is the <em>read</em> side of the claim: the bytes are Python's. The matching statement
    /// about what C# writes is
    /// <c>YamlBackendTests.an_array_field_written_by_csharp_carries_the_embedded_json_wrapper</c>,
    /// and only the pair of them says the two languages agree.
    /// </para>
    /// </remarks>
    [Fact]
    public void the_corpus_stores_an_array_field_as_a_single_embedded_json_wrapper()
    {
        string text = File.ReadAllText(Path.Combine(_goldenRoot, "arrays", "arrays.yaml"));

        Assert.Contains("signal:\n  __ver_json__:", text.ReplaceLineEndings("\n"), StringComparison.Ordinal);
        Assert.DoesNotContain("dtype: float64", text, StringComparison.Ordinal);
    }

    /// <summary>The envelope is the last thing in a Python-written YAML file, not the first.</summary>
    [Fact]
    public void the_envelope_sits_at_the_end_of_every_golden_file()
    {
        using JsonDocument index = JsonDocument.Parse(
            File.ReadAllBytes(Path.Combine(_goldenRoot, "index.json")));

        foreach (JsonElement entry in index.RootElement.GetProperty("fixtures").EnumerateArray())
        {
            string fixture = entry.GetProperty("fixture").GetString()!;
            foreach (string file in FilesOf(Manifest(fixture)))
            {
                string[] lines = File.ReadAllLines(Path.Combine(_goldenRoot, fixture, file));
                int envelope = Array.FindIndex(
                    lines,
                    line => line.StartsWith($"{VersionableEnvelope.WrappedKey}:", StringComparison.Ordinal));

                Assert.True(envelope >= 0, $"{file} has no top-level envelope.");
                Assert.Equal(lines.Length - 4, envelope);
            }
        }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static IEnumerable<string> FilesOf(JsonElement manifest)
    {
        yield return manifest.GetProperty("files").GetProperty("yaml").GetString()!;

        // Only the migration fixture declares older files; the rest omit the key entirely.
        if (manifest.TryGetProperty("migrationSources", out JsonElement sources))
        {
            foreach (JsonElement source in sources.EnumerateArray())
            {
                yield return source.GetProperty("files").GetProperty("yaml").GetString()!;
            }
        }
    }

    private static void AssertTensor<T>(
        JsonElement tagged,
        string dtype,
        Tensor<T> tensor,
        Func<JsonElement, T> element)
    {
        JsonElement declared = tagged.GetProperty("$ndarray");
        Assert.Equal(dtype, declared.GetProperty("dtype").GetString());

        int[] shape = new int[tensor.Rank];
        for (int axis = 0; axis < shape.Length; axis++)
        {
            shape[axis] = (int)tensor.Lengths[axis];
        }

        Assert.Equal(Ints(declared.GetProperty("shape")), shape);

        T[] flat = new T[(int)tensor.FlattenedLength];
        tensor.FlattenTo(flat);
        Assert.Equal(FlattenScalars(declared.GetProperty("data")).Select(element), flat);
    }

    private static IEnumerable<JsonElement> FlattenScalars(JsonElement array)
    {
        foreach (JsonElement item in array.EnumerateArray())
        {
            if (item.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement nested in FlattenScalars(item))
                {
                    yield return nested;
                }
            }
            else
            {
                yield return item;
            }
        }
    }

    private static decimal Decimal(JsonElement tagged) =>
        decimal.Parse(tagged.GetProperty("$decimal").GetString()!, CultureInfo.InvariantCulture);

    private static string PathValue(JsonElement tagged) => tagged.GetProperty("$path").GetProperty("value").GetString()!;

    private static void AssertInner(JsonElement tagged, GoldenInner inner)
    {
        JsonElement fields = tagged.GetProperty("$object").GetProperty("fields");
        Assert.Equal(fields.GetProperty("x").GetDouble(), inner.X);
        Assert.Equal(fields.GetProperty("y").GetDouble(), inner.Y);
    }

    private static void AssertWorker(JsonElement expected, GoldenWorker worker)
    {
        Assert.Equal(expected.GetProperty("name").GetString(), worker.Name);
        Assert.Equal(expected.GetProperty("retries").GetInt32(), worker.Retries);
        Assert.Equal(expected.GetProperty("timeout_ms").GetInt32(), worker.TimeoutMs);
    }

    private static GoldenColour Colour(JsonElement tagged) =>
        (GoldenColour)Engine.WireValues.Read(
            tagged.GetProperty("$enum").GetProperty("value").GetString(), typeof(GoldenColour))!;

    private static IEnumerable<string> Strings(JsonElement array) =>
        array.EnumerateArray().Select(item => item.GetString()!);

    private static IEnumerable<int> Ints(JsonElement array) => array.EnumerateArray().Select(item => item.GetInt32());

    private static IEnumerable<double> Doubles(JsonElement array) =>
        array.EnumerateArray().Select(item => item.GetDouble());

    private static IEnumerable<(JsonElement Key, JsonElement Value)> Pairs(JsonElement tagged) =>
        tagged.GetProperty("$dict").EnumerateArray().Select(entry =>
        {
            JsonElement[] pair = [.. entry.EnumerateArray()];
            return (pair[0], pair[1]);
        });

    private static T Load<T>(string fixture, string file)
        where T : IVersionableMetadataProvider =>
        VersionableFile.Load<T>(Path.Combine(_goldenRoot, fixture, file));

    private static JsonElement Values(string fixture) => Manifest(fixture).GetProperty("values");

    private static JsonElement Manifest(string fixture)
    {
        using JsonDocument document = JsonDocument.Parse(
            File.ReadAllBytes(Path.Combine(_goldenRoot, fixture, "manifest.json")));

        return document.RootElement.Clone();
    }

    private static string FindGoldenRoot()
    {
        for (DirectoryInfo? directory = new(AppContext.BaseDirectory);
            directory is not null;
            directory = directory.Parent)
        {
            string candidate = Path.Combine(directory.FullName, "conformance", "golden");
            if (File.Exists(Path.Combine(candidate, "index.json")))
            {
                return candidate;
            }
        }

        throw new InvalidOperationException(
            string.Create(
                CultureInfo.InvariantCulture,
                $"conformance/golden not found above '{AppContext.BaseDirectory}'."));
    }
}
