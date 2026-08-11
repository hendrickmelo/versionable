using System.Globalization;
using System.Numerics;
using System.Numerics.Tensors;
using System.Text.Json;
using Versionable.Backends.Toml;
using Versionable.Engine;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// The TOML half of the golden-corpus contract: the <c>.toml</c> bytes Python wrote, read here and
/// compared against the same manifests the Python suite compares against — then written back out
/// and read again.
/// </summary>
/// <remarks>
/// Python counterpart: <c>tests/test_golden_corpus.py</c>. Nothing here regenerates anything; the
/// committed files are the contract.
/// <para>
/// Every fixture runs the same shape: load the checked-in file, assert it against the manifest,
/// save the loaded object to a fresh <c>.toml</c>, load <em>that</em>, and assert it against the
/// manifest again. The second half is what makes this a writer test as well as a reader test —
/// the assertions are the corpus's, not the writer's, so a save that quietly reshaped a value
/// fails against Python's numbers rather than against its own output.
/// </para>
/// <para>
/// Both loads go through <see cref="VersionableFile.LoadDynamic(string, Versionable.Backends.IVersionableBackend?,
/// VersionableLoadOptions?)"/> — the envelope-driven overload — so the object name written into
/// <c>[__versionable__]</c> is exercised on every fixture rather than being supplied by the test.
/// </para>
/// </remarks>
[Collection(RegistryCollection.Name)]
public class TomlGoldenCorpusTests : IDisposable
{
    private static readonly string _goldenRoot = GoldenCorpus.Root;

    private readonly string _directory =
        Path.Combine(Path.GetTempPath(), $"versionable-toml-golden-{Path.GetRandomFileName()}");

    public TomlGoldenCorpusTests()
    {
        GoldenSchemas.EnsureRegistered();
        Directory.CreateDirectory(_directory);
    }

    public void Dispose()
    {
        Directory.Delete(_directory, recursive: true);
        GC.SuppressFinalize(this);
    }

    /// <summary>Every fixture in the corpus loads from TOML with the manifest's values.</summary>
    /// <param name="fixture">Corpus directory name.</param>
    [Theory]
    [InlineData("scalars")]
    [InlineData("containers")]
    [InlineData("optionals")]
    [InlineData("enums")]
    [InlineData("literals")]
    [InlineData("temporal")]
    [InlineData("stdlib")]
    [InlineData("arrays")]
    [InlineData("nested")]
    [InlineData("polymorphic")]
    [InlineData("migration-chain")]
    public void the_golden_toml_loads_with_the_manifest_values(string fixture)
    {
        JsonElement manifest = Manifest(fixture);
        string file = manifest.GetProperty("files").GetProperty("toml").GetString()!;

        object loaded = VersionableFile.LoadDynamic(Path.Combine(_goldenRoot, fixture, file));

        AssertFixture(fixture, manifest.GetProperty("values"), loaded);
    }

    /// <summary>
    /// Every fixture the writer can express survives a save-and-reload with the manifest's values
    /// intact.
    /// </summary>
    /// <param name="fixture">Corpus directory name.</param>
    /// <remarks>
    /// The assertions are the corpus's, not the writer's, so a save that quietly reshaped a value
    /// fails against Python's numbers rather than against its own output.
    /// </remarks>
    [Theory]
    [InlineData("scalars")]
    [InlineData("containers")]
    [InlineData("optionals")]
    [InlineData("enums")]
    [InlineData("literals")]
    [InlineData("temporal")]
    [InlineData("stdlib")]
    [InlineData("arrays")]
    [InlineData("nested")]
    [InlineData("polymorphic")]
    [InlineData("migration-chain")]
    public void a_saved_toml_file_reloads_with_the_manifest_values(string fixture)
    {
        JsonElement manifest = Manifest(fixture);
        string file = manifest.GetProperty("files").GetProperty("toml").GetString()!;

        object loaded = VersionableFile.LoadDynamic(Path.Combine(_goldenRoot, fixture, file));
        string written = Path.Combine(_directory, $"{fixture}.toml");
        VersionableFile.Save(loaded, written);

        object reloaded = VersionableFile.LoadDynamic(written);

        AssertFixture(fixture, manifest.GetProperty("values"), reloaded);
    }

    /// <summary>
    /// Rewriting a golden fixture reproduces tomlkit's bytes exactly.
    /// </summary>
    /// <remarks>
    /// Byte-identity is not the contract — the contract is that both languages read both files —
    /// but where it happens to hold it is the tightest available statement of writer parity, and
    /// it is the only thing that would notice a Tomlyn upgrade changing how it spells a float, an
    /// escape, or the blank line before a table header. Those changes break no assertion above:
    /// every value still loads.
    /// <para>
    /// Three fixtures are excluded and none for an emitter reason. <c>stdlib</c> and
    /// <c>containers</c> hash differently in C# (GRAMMAR §9 and §5), so their envelope line cannot
    /// match however the values are written. <c>arrays</c> differs inside the <c>__ver_json__</c>
    /// string, where <c>Utf8JsonWriter</c> compacts and <c>json.dumps</c> spaces, and in the NPZ
    /// payload bytes themselves.
    /// </para>
    /// <para>
    /// If this fails after a deliberate change, check that Python still reads the new output —
    /// <see cref="a_saved_toml_file_reloads_with_the_manifest_values"/> and the Python suite are
    /// the tests that must not be relaxed — and then update the expectation here.
    /// </para>
    /// </remarks>
    /// <param name="fixture">Corpus directory name.</param>
    [Theory]
    [InlineData("scalars")]
    [InlineData("optionals")]
    [InlineData("enums")]
    [InlineData("literals")]
    [InlineData("temporal")]
    [InlineData("nested")]
    [InlineData("polymorphic")]
    [InlineData("migration-chain")]
    public void rewriting_a_golden_fixture_reproduces_tomlkits_bytes(string fixture)
    {
        string file = Manifest(fixture).GetProperty("files").GetProperty("toml").GetString()!;
        string source = Path.Combine(_goldenRoot, fixture, file);
        string written = Path.Combine(_directory, $"identity-{file}");

        VersionableFile.Save(VersionableFile.LoadDynamic(source), written);

        Assert.Equal(File.ReadAllBytes(source), File.ReadAllBytes(written));
    }

    /// <summary>
    /// The corpus's older <c>migration-chain</c> files migrate forward from TOML exactly as they
    /// do from JSON.
    /// </summary>
    [Fact]
    public void older_toml_files_migrate_forward_to_the_current_schema()
    {
        JsonElement manifest = Manifest("migration-chain");
        int sources = 0;

        foreach (JsonElement source in manifest.GetProperty("migrationSources").EnumerateArray())
        {
            string file = source.GetProperty("files").GetProperty("toml").GetString()!;
            GoldenWorker loaded = VersionableFile.Load<GoldenWorker>(
                Path.Combine(_goldenRoot, "migration-chain", file));

            AssertWorker(source.GetProperty("expected"), loaded);
            sources++;
        }

        Assert.Equal(2, sources);
    }

    /// <summary>
    /// The one representational choice TOML forces: a null field is absent from the file, and
    /// comes back from the type's default rather than from a sentinel.
    /// </summary>
    /// <remarks>
    /// The manifest records <c>absent: null</c> for every backend; only TOML has no token for it.
    /// Asserting on the bytes as well as on the loaded value is deliberate — a future writer that
    /// invented <c>absent = ""</c> would still load as null through the union walk and would only
    /// be caught here.
    /// </remarks>
    [Fact]
    public void a_null_field_is_omitted_from_the_file_and_refilled_from_the_default()
    {
        string text = File.ReadAllText(Path.Combine(_goldenRoot, "optionals", "optionals.toml"));
        Assert.DoesNotContain("absent", text, StringComparison.Ordinal);

        GoldenOptionals loaded = VersionableFile.Load<GoldenOptionals>(
            Path.Combine(_goldenRoot, "optionals", "optionals.toml"));
        Assert.Null(loaded.Absent);

        string written = Path.Combine(_directory, "optionals.toml");
        VersionableFile.Save(loaded, written);
        Assert.DoesNotContain("absent", File.ReadAllText(written), StringComparison.Ordinal);
    }

    /// <summary>
    /// Arrays travel in the <c>__ver_json__</c> wrapper Python writes, not as loose TOML tables.
    /// </summary>
    [Fact]
    public void an_array_is_written_as_the_json_wrapper_python_reads()
    {
        GoldenArrays loaded = VersionableFile.Load<GoldenArrays>(
            Path.Combine(_goldenRoot, "arrays", "arrays.toml"));
        string written = Path.Combine(_directory, "arrays.toml");
        VersionableFile.Save(loaded, written);

        string text = File.ReadAllText(written);
        Assert.Contains("__ver_json__ = ", text, StringComparison.Ordinal);
        Assert.Contains("__ver_ndarray__", text, StringComparison.Ordinal);

        // The marker never leaks out as a bare TOML key: that would be an unwrapped payload.
        Assert.DoesNotContain("__ver_ndarray__ = ", text, StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------
    // Per-fixture assertions, run against the corpus file and its round trip alike
    // ------------------------------------------------------------------

    private static void AssertFixture(string fixture, JsonElement values, object loaded)
    {
        switch (fixture)
        {
            case "scalars":
                AssertScalars(values, Assert.IsType<GoldenScalars>(loaded));
                break;
            case "containers":
                AssertContainers(values, Assert.IsType<GoldenContainers>(loaded));
                break;
            case "optionals":
                AssertOptionals(values, Assert.IsType<GoldenOptionals>(loaded));
                break;
            case "enums":
                AssertEnums(values, Assert.IsType<GoldenEnums>(loaded));
                break;
            case "literals":
                AssertLiterals(values, Assert.IsType<GoldenLiterals>(loaded));
                break;
            case "temporal":
                AssertTemporal(values, Assert.IsType<GoldenTemporal>(loaded));
                break;
            case "stdlib":
                AssertStdlib(values, Assert.IsType<GoldenStdlib>(loaded));
                break;
            case "arrays":
                AssertArrays(values, Assert.IsType<GoldenArrays>(loaded));
                break;
            case "nested":
                AssertNested(values, Assert.IsType<GoldenNested>(loaded));
                break;
            case "polymorphic":
                AssertPolymorphic(values, Assert.IsType<GoldenPolymorphic>(loaded));
                break;
            case "migration-chain":
                AssertWorker(values, Assert.IsType<GoldenWorker>(loaded));
                break;
            default:
                Assert.Fail($"no assertions for fixture '{fixture}'");
                break;
        }
    }

    private static void AssertScalars(JsonElement values, GoldenScalars loaded)
    {
        Assert.Equal(values.GetProperty("text").GetString(), loaded.Text);
        Assert.Equal(values.GetProperty("count").GetInt32(), loaded.Count);
        Assert.Equal(values.GetProperty("ratio").GetDouble(), loaded.Ratio);
        Assert.Equal(values.GetProperty("enabled").GetBoolean(), loaded.Enabled);

        double[] parts = [.. Doubles(values.GetProperty("phase").GetProperty("$complex"))];
        Assert.Equal(new Complex(parts[0], parts[1]), loaded.Phase);

        Assert.Equal(
            Convert.FromBase64String(values.GetProperty("blob").GetProperty("$bytes").GetString()!),
            loaded.Blob);
    }

    private static void AssertContainers(JsonElement values, GoldenContainers loaded)
    {
        Assert.Equal(Strings(values.GetProperty("names")), loaded.Names);
        Assert.Equal(Doubles(values.GetProperty("readings")), loaded.Readings);
        Assert.Equal(Ints(values.GetProperty("counts")), loaded.Counts);
        Assert.Equal(
            values.GetProperty("flags").EnumerateArray().Select(item => item.GetBoolean()),
            loaded.Flags);

        Assert.Equal(
            Pairs(values.GetProperty("lookup")).ToDictionary(pair => pair.Key.GetString()!, pair => pair.Value.GetInt32()),
            loaded.Lookup);
        Assert.Equal(
            Pairs(values.GetProperty("byIndex")).ToDictionary(pair => pair.Key.GetInt32(), pair => pair.Value.GetString()!),
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
            Pairs(values.GetProperty("grouped"))
                .ToDictionary(pair => pair.Key.GetString()!, pair => Doubles(pair.Value).ToList()),
            loaded.Grouped);
    }

    private static void AssertOptionals(JsonElement values, GoldenOptionals loaded)
    {
        Assert.Equal(values.GetProperty("present").GetString(), loaded.Present);
        Assert.Equal(JsonValueKind.Null, values.GetProperty("absent").ValueKind);
        Assert.Null(loaded.Absent);
        Assert.Equal(values.GetProperty("maybeCount").GetInt32(), loaded.MaybeCount);
        Assert.Equal(Decimal(values.GetProperty("money")), loaded.Money);
        Assert.Equal(values.GetProperty("either").GetString(), loaded.Either);
    }

    private static void AssertEnums(JsonElement values, GoldenEnums loaded)
    {
        Assert.Equal(Colour(values.GetProperty("colour")), loaded.Colour);
        Assert.Equal(
            (GoldenPriority)values.GetProperty("priority").GetProperty("$enum").GetProperty("value").GetInt32(),
            loaded.Priority);
        Assert.Equal(values.GetProperty("palette").EnumerateArray().Select(Colour), loaded.Palette);
        Assert.Equal(
            Pairs(values.GetProperty("byName")).ToDictionary(pair => pair.Key.GetString()!, pair => Colour(pair.Value)),
            loaded.ByName);
    }

    private static void AssertLiterals(JsonElement values, GoldenLiterals loaded)
    {
        Assert.Equal(values.GetProperty("mode").GetString(), loaded.Mode);
        Assert.Equal(values.GetProperty("level").GetInt32(), loaded.Level);
        Assert.Equal(values.GetProperty("flag").GetBoolean(), loaded.Flag);
        Assert.Equal(values.GetProperty("tag").GetInt64(), loaded.Tag);
    }

    private static void AssertTemporal(JsonElement values, GoldenTemporal loaded)
    {
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

    private static void AssertStdlib(JsonElement values, GoldenStdlib loaded)
    {
        Assert.Equal(PathValue(values.GetProperty("filePath")), loaded.FilePath.Value);
        Assert.Equal(PathValue(values.GetProperty("posixPath")), loaded.PosixPath.Value);
        Assert.Equal(PathValue(values.GetProperty("windowsPath")), loaded.WindowsPath.Value);

        Assert.Equal(Decimal(values.GetProperty("amount")), loaded.Amount);
        Assert.Equal(Guid.Parse(values.GetProperty("deviceId").GetProperty("$uuid").GetString()!), loaded.DeviceId);
        Assert.Equal(
            values.GetProperty("serialPattern").GetProperty("$pattern").GetString(),
            loaded.SerialPattern.ToString());
    }

    private static void AssertArrays(JsonElement values, GoldenArrays loaded)
    {
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

    private static void AssertNested(JsonElement values, GoldenNested loaded)
    {
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

    private static void AssertPolymorphic(JsonElement values, GoldenPolymorphic loaded)
    {
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

    private static void AssertWorker(JsonElement expected, GoldenWorker worker)
    {
        Assert.Equal(expected.GetProperty("name").GetString(), worker.Name);
        Assert.Equal(expected.GetProperty("retries").GetInt32(), worker.Retries);
        Assert.Equal(expected.GetProperty("timeout_ms").GetInt32(), worker.TimeoutMs);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

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

    private static void AssertInner(JsonElement tagged, GoldenInner inner)
    {
        JsonElement fields = tagged.GetProperty("$object").GetProperty("fields");
        Assert.Equal(fields.GetProperty("x").GetDouble(), inner.X);
        Assert.Equal(fields.GetProperty("y").GetDouble(), inner.Y);
    }

    private static GoldenColour Colour(JsonElement tagged) =>
        (GoldenColour)WireValues.Read(
            tagged.GetProperty("$enum").GetProperty("value").GetString(), typeof(GoldenColour))!;

    private static decimal Decimal(JsonElement tagged) =>
        decimal.Parse(tagged.GetProperty("$decimal").GetString()!, CultureInfo.InvariantCulture);

    private static string PathValue(JsonElement tagged) => tagged.GetProperty("$path").GetProperty("value").GetString()!;

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

    private static JsonElement Manifest(string fixture)
    {
        using JsonDocument document = JsonDocument.Parse(
            File.ReadAllBytes(Path.Combine(_goldenRoot, fixture, "manifest.json")));

        return document.RootElement.Clone();
    }
}
