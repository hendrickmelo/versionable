using System.Globalization;
using System.Numerics;
using System.Numerics.Tensors;
using System.Text.Json;
using Versionable.Engine;
using Versionable.Migrations;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// The C# half of the golden-corpus contract: the checked-in bytes Python wrote, read here and
/// compared against the same manifests the Python suite compares against.
/// </summary>
/// <remarks>
/// Python counterpart: <c>tests/test_golden_corpus.py</c>. Nothing here regenerates anything —
/// the committed files are the contract, and a failure means either a reader broke or a
/// serializer changed behavior (regenerate deliberately with
/// <c>pixi run -- python conformance/generate_golden.py</c> and review the diff).
/// <para>
/// All eleven fixtures load, and ten of the eleven load through <b>generator-emitted</b> metadata:
/// the schemas in <c>EngineGoldenSchemas.cs</c> are real <c>[Versionable]</c> partial types
/// compiled by the source generator in this assembly, so their hashes are checked by the analyzer
/// at build time rather than merely asserted here. <c>optionals</c> is the exception and says why
/// at its declaration.
/// </para>
/// <para>
/// The five that go through the converter set — scalars, optionals, temporal, stdlib, arrays —
/// reach it with no generated help at all: the generator emits no <c>WireReader</c> for a field a
/// registered converter already handles, so what they prove is that the engine's own dispatch
/// finds the right converter.
/// </para>
/// </remarks>
[Collection(RegistryCollection.Name)]
public class JsonGoldenCorpusTests
{
    private static readonly string _goldenRoot = GoldenCorpus.Root;

    public JsonGoldenCorpusTests() => GoldenSchemas.EnsureRegistered();

    // ------------------------------------------------------------------
    // The corpus and the C# schemas agree about what they are
    // ------------------------------------------------------------------

    [Theory]
    [InlineData("enums", "GoldenEnums")]
    [InlineData("literals", "GoldenLiterals")]
    [InlineData("nested", "GoldenNested")]
    [InlineData("polymorphic", "GoldenPolymorphic")]
    [InlineData("migration-chain", "GoldenWorker")]
    [InlineData("scalars", "GoldenScalars")]
    [InlineData("optionals", "GoldenOptionals")]
    [InlineData("temporal", "GoldenTemporal")]
    [InlineData("arrays", "GoldenArrays")]
    // `containers` and `stdlib` are in the theory below instead: each declares a Python construct
    // C# has no spelling for, so their C# hashes are legitimately different.
    public void the_csharp_schema_mirrors_the_manifest(string fixture, string serializationName)
    {
        JsonElement manifest = Manifest(fixture);
        VersionableMetadata metadata = MetadataFor(serializationName);

        Assert.Equal(manifest.GetProperty("serializationName").GetString(), metadata.Name);
        Assert.Equal(manifest.GetProperty("schemaHash").GetString(), metadata.Hash);
        Assert.Equal(manifest.GetProperty("version").GetInt32(), metadata.Version);
    }

    /// <summary>
    /// The two fixtures whose Python declaration has no C# spelling still agree on everything the
    /// wire depends on — and are asserted to disagree on the hash, so a future C# feature that
    /// closes the gap fails here rather than passing silently.
    /// </summary>
    /// <param name="fixture">Corpus directory name.</param>
    /// <param name="serializationName">Serialization Name both sides declare.</param>
    /// <param name="reason">The Python construct C# cannot spell.</param>
    [Theory]
    [InlineData("containers", "GoldenContainers", "tuple[float, ...] renders list[float] in C#")]
    [InlineData("stdlib", "GoldenStdlib", "PurePosixPath and PureWindowsPath both render Path in C#")]
    public void a_fixture_csharp_cannot_spell_mirrors_everything_but_its_hash(
        string fixture,
        string serializationName,
        string reason)
    {
        JsonElement manifest = Manifest(fixture);
        VersionableMetadata metadata = MetadataFor(serializationName);

        Assert.Equal(manifest.GetProperty("serializationName").GetString(), metadata.Name);
        Assert.Equal(manifest.GetProperty("version").GetInt32(), metadata.Version);
        Assert.False(
            manifest.GetProperty("schemaHash").GetString() == metadata.Hash,
            $"{fixture}: the C# hash now matches Python's, so the divergence is gone — {reason}. "
                + "Move this fixture into the_csharp_schema_mirrors_the_manifest.");
    }

    /// <summary>
    /// Every fixture but <c>optionals</c> resolves to the metadata the <em>generator</em> emitted,
    /// not to something hand-built alongside it.
    /// </summary>
    /// <remarks>
    /// This is what makes the rest of the file a test of the compile-time pipeline: reference
    /// equality with <c>T.VersionableMetadata</c> holds only if the registry entry came from the
    /// generated <c>[ModuleInitializer]</c> (or from re-registering that same instance), and
    /// <c>T.VersionableMetadata</c> exists only because the generator emitted it.
    /// </remarks>
    [Fact]
    public void the_corpus_loads_through_generator_emitted_metadata()
    {
        (string Name, VersionableMetadata Generated)[] generated =
        [
            ("GoldenContainers", GoldenContainers.VersionableMetadata),
            ("GoldenEnums", GoldenEnums.VersionableMetadata),
            ("GoldenLiterals", GoldenLiterals.VersionableMetadata),
            ("GoldenInner", GoldenInner.VersionableMetadata),
            ("GoldenNested", GoldenNested.VersionableMetadata),
            ("GoldenShape", GoldenShape.VersionableMetadata),
            ("GoldenCircle", GoldenCircle.VersionableMetadata),
            ("GoldenSquare", GoldenSquare.VersionableMetadata),
            ("GoldenPolymorphic", GoldenPolymorphic.VersionableMetadata),
            ("GoldenWorker", GoldenWorker.VersionableMetadata),
            ("GoldenScalars", GoldenScalars.VersionableMetadata),
            ("GoldenTemporal", GoldenTemporal.VersionableMetadata),
            ("GoldenStdlib", GoldenStdlib.VersionableMetadata),
            ("GoldenArrays", GoldenArrays.VersionableMetadata),
        ];

        foreach ((string name, VersionableMetadata metadata) in generated)
        {
            Assert.Same(metadata, MetadataFor(name));
        }

        // The chain in the metadata is composed by the generator out of GoldenWorker.Migrate's V1
        // and V2 builder members: the fixture proves the emission, and the older corpus files
        // below prove the composition runs.
        MigrationChain chain = Assert.IsType<MigrationChain>(GoldenWorker.VersionableMetadata.Migrations);
        Assert.Equal([1, 2], chain.FromVersions);
        Assert.Same(GoldenWorker.Migrate.V1, Assert.Single(chain.Steps, step => step.FromVersion == 1).Declarative);
    }

    // ------------------------------------------------------------------
    // Save is the other half of the contract
    // ------------------------------------------------------------------

    /// <summary>
    /// Every fixture survives a C#-written round trip, not just a Python-written read.
    /// </summary>
    /// <remarks>
    /// The corpus proves C# can read what Python wrote; this proves C# can write what C# can read,
    /// which is the half a load-only suite silently skips. <c>containers</c> is the reason it is a
    /// theory over every fixture rather than a handful: its <c>pair</c> field is a
    /// <c>ValueTuple</c>, and the walker had no tuple lowering, so saving it — and only saving it —
    /// threw on every text backend.
    /// <para>
    /// The comparison is the walker's own wire form of both objects rather than the two files'
    /// bytes: a C#-written array carries a C#-written NPZ payload, whose ZIP metadata is allowed to
    /// differ from numpy's by design, and comparing wire forms compares the values instead of the
    /// container they arrived in.
    /// </para>
    /// </remarks>
    /// <param name="fixture">Corpus directory name.</param>
    /// <param name="file">The JSON file inside it.</param>
    [Theory]
    [InlineData("containers", "containers.json")]
    [InlineData("scalars", "scalars.json")]
    [InlineData("optionals", "optionals.json")]
    [InlineData("enums", "enums.json")]
    [InlineData("literals", "literals.json")]
    [InlineData("temporal", "temporal.json")]
    [InlineData("stdlib", "stdlib.json")]
    [InlineData("arrays", "arrays.json")]
    [InlineData("nested", "nested.json")]
    [InlineData("polymorphic", "polymorphic.json")]
    [InlineData("migration-chain", "migration-chain.json")]
    public void a_fixture_csharp_wrote_loads_back_identical(string fixture, string file)
    {
        object fromCorpus = VersionableFile.LoadDynamic(Path.Combine(_goldenRoot, fixture, file));

        string directory = Path.Combine(Path.GetTempPath(), $"versionable-golden-{Path.GetRandomFileName()}");
        Directory.CreateDirectory(directory);
        try
        {
            string written = Path.Combine(directory, file);
            VersionableFile.Save(fromCorpus, written);

            Assert.Equal(Canonical(fromCorpus), Canonical(VersionableFile.LoadDynamic(written)));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    // ------------------------------------------------------------------
    // Fixtures that load with the converter set as it stands
    // ------------------------------------------------------------------

    [Fact]
    public void containers_loads_every_container_form_python_wrote()
    {
        JsonElement values = Values("containers");
        GoldenContainers loaded = Load<GoldenContainers>("containers", "containers.json");

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
        Assert.Equal(
            Pairs(values.GetProperty("byIndex"))
                .ToDictionary(pair => pair.Key.GetInt32(), pair => pair.Value.GetString()!),
            loaded.ByIndex);

        // Sets are unordered on the wire; the manifest orders them for comparison only.
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
        GoldenEnums loaded = Load<GoldenEnums>("enums", "enums.json");

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
        GoldenLiterals loaded = Load<GoldenLiterals>("literals", "literals.json");

        Assert.Equal(values.GetProperty("mode").GetString(), loaded.Mode);
        Assert.Equal(values.GetProperty("level").GetInt32(), loaded.Level);
        Assert.Equal(values.GetProperty("flag").GetBoolean(), loaded.Flag);

        // `tag` is Literal['auto', 0, 'off', 1] and the file holds the integer 1, not the string.
        Assert.Equal(values.GetProperty("tag").GetInt64(), loaded.Tag);
    }

    [Fact]
    public void nested_objects_load_from_their_own_envelopes()
    {
        JsonElement values = Values("nested");
        GoldenNested loaded = Load<GoldenNested>("nested", "nested.json");

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
        GoldenPolymorphic loaded = Load<GoldenPolymorphic>("polymorphic", "polymorphic.json");

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
        JsonElement values = Values("migration-chain");
        GoldenWorker loaded = Load<GoldenWorker>("migration-chain", "migration-chain.json");

        AssertWorker(values, loaded);
    }

    [Fact]
    public void older_files_migrate_forward_to_the_current_schema()
    {
        JsonElement manifest = Manifest("migration-chain");

        foreach (JsonElement source in manifest.GetProperty("migrationSources").EnumerateArray())
        {
            string file = source.GetProperty("files").GetProperty("json").GetString()!;
            GoldenWorker loaded = Load<GoldenWorker>("migration-chain", file);

            AssertWorker(source.GetProperty("expected"), loaded);
        }
    }

    // ------------------------------------------------------------------
    // Fixtures that reach the wire through the converter set
    // ------------------------------------------------------------------

    [Fact]
    public void scalars_load_every_scalar_token()
    {
        JsonElement values = Values("scalars");
        GoldenScalars loaded = Load<GoldenScalars>("scalars", "scalars.json");

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

    [Fact]
    public void optionals_load_including_a_null_and_a_multi_member_union()
    {
        JsonElement values = Values("optionals");
        GoldenOptionals loaded = Load<GoldenOptionals>("optionals", "optionals.json");

        Assert.Equal(values.GetProperty("present").GetString(), loaded.Present);
        Assert.Equal(JsonValueKind.Null, values.GetProperty("absent").ValueKind);
        Assert.Null(loaded.Absent);
        Assert.Equal(values.GetProperty("maybeCount").GetInt32(), loaded.MaybeCount);
        Assert.Equal(Decimal(values.GetProperty("money")), loaded.Money);

        // `either` is int | str and the file holds a string, so the int member has to fail and the
        // str member has to win — the union walk, not a cast.
        Assert.Equal(values.GetProperty("either").GetString(), loaded.Either);
    }

    [Fact]
    public void temporal_values_load_with_microsecond_precision()
    {
        JsonElement values = Values("temporal");
        GoldenTemporal loaded = Load<GoldenTemporal>("temporal", "temporal.json");

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
        GoldenStdlib loaded = Load<GoldenStdlib>("stdlib", "stdlib.json");

        // All three path fields are FilePath on this side: C# v1 has no PurePosixPath or
        // PureWindowsPath (GRAMMAR §9). The wire form is the same bare string, so the values
        // round-trip; only the canonical rendering differs, which is why this fixture is absent
        // from the hash-mirror theory above.
        Assert.Equal(PathValue(values.GetProperty("filePath")), loaded.FilePath.Value);
        Assert.Equal(PathValue(values.GetProperty("posixPath")), loaded.PosixPath.Value);
        Assert.Equal(PathValue(values.GetProperty("windowsPath")), loaded.WindowsPath.Value);

        Assert.Equal(Decimal(values.GetProperty("amount")), loaded.Amount);
        Assert.Equal(Guid.Parse(values.GetProperty("deviceId").GetProperty("$uuid").GetString()!), loaded.DeviceId);
        Assert.Equal(
            values.GetProperty("serialPattern").GetProperty("$pattern").GetString(),
            loaded.SerialPattern.ToString());
    }

    [Fact]
    public void arrays_load_from_their_base64_npz_payloads()
    {
        JsonElement values = Values("arrays");
        GoldenArrays loaded = Load<GoldenArrays>("arrays", "arrays.json");

        AssertTensor(values.GetProperty("signal"), "float64", loaded.Signal, element => element.GetDouble());
        AssertTensor(values.GetProperty("weights"), "float32", loaded.Weights, element => element.GetSingle());
        AssertTensor(values.GetProperty("counts"), "int32", loaded.Counts, element => element.GetInt32());
        AssertTensor(values.GetProperty("image"), "uint8", loaded.Image, element => element.GetByte());
        AssertTensor(values.GetProperty("mask"), "bool", loaded.Mask, element => element.GetBoolean());

        // Shape is erased from the hash but not from the file: a 2x3 has to come back a 2x3.
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

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// <summary>Asserts a loaded tensor against the manifest's <c>$ndarray</c> form.</summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="tagged">The tagged element.</param>
    /// <param name="dtype">The dtype token the manifest must declare.</param>
    /// <param name="tensor">The loaded tensor.</param>
    /// <param name="element">Reads one element from the manifest.</param>
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

    /// <summary>Yields the scalars of an arbitrarily nested manifest data array, in C order.</summary>
    /// <param name="array">The manifest's <c>data</c> element.</param>
    /// <returns>The scalars, flattened.</returns>
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

    /// <summary>
    /// Renders an object's wire form as a deterministic string, for comparing two instances.
    /// </summary>
    /// <remarks>
    /// Dictionary keys are sorted so a comparison does not depend on hash order; lists are not,
    /// because element order is part of the value. Numbers go through the invariant culture, and
    /// both sides come from the same CLR types, so no formatting drift can creep in between them.
    /// </remarks>
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

    private static decimal Decimal(JsonElement tagged) =>
        decimal.Parse(tagged.GetProperty("$decimal").GetString()!, CultureInfo.InvariantCulture);

    private static string PathValue(JsonElement tagged) =>
        tagged.GetProperty("$path").GetProperty("value").GetString()!;

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

    /// <summary>Reads the manifest's <c>$enum</c> form through the same path a file takes.</summary>
    /// <param name="tagged">The tagged element.</param>
    /// <returns>The member the wire value names.</returns>
    private static GoldenColour Colour(JsonElement tagged) =>
        (GoldenColour)WireValues.Read(
            tagged.GetProperty("$enum").GetProperty("value").GetString(), typeof(GoldenColour))!;

    private static IEnumerable<string> Strings(JsonElement array) =>
        array.EnumerateArray().Select(item => item.GetString()!);

    private static IEnumerable<int> Ints(JsonElement array) => array.EnumerateArray().Select(item => item.GetInt32());

    private static IEnumerable<double> Doubles(JsonElement array) =>
        array.EnumerateArray().Select(item => item.GetDouble());

    /// <summary>Unpacks the manifest's <c>$dict</c> form: an array of two-element key/value arrays.</summary>
    /// <param name="tagged">The tagged element.</param>
    /// <returns>The entries, in manifest order.</returns>
    private static IEnumerable<(JsonElement Key, JsonElement Value)> Pairs(JsonElement tagged) =>
        tagged.GetProperty("$dict").EnumerateArray().Select(entry =>
        {
            JsonElement[] pair = [.. entry.EnumerateArray()];
            return (pair[0], pair[1]);
        });

    private static T Load<T>(string fixture, string file)
        where T : IVersionableMetadataProvider =>
        VersionableFile.Load<T>(Path.Combine(_goldenRoot, fixture, file));

    private static VersionableMetadata MetadataFor(string serializationName) =>
        VersionableRegistry.TryGetByName(serializationName, out VersionableMetadata? metadata)
            ? metadata
            : throw new InvalidOperationException($"no C# fixture registered as '{serializationName}'");

    private static JsonElement Values(string fixture) => Manifest(fixture).GetProperty("values");

    private static JsonElement Manifest(string fixture)
    {
        using JsonDocument document = JsonDocument.Parse(
            File.ReadAllBytes(Path.Combine(_goldenRoot, fixture, "manifest.json")));

        // Cloned so the element outlives the document, which the using disposes.
        return document.RootElement.Clone();
    }
}
