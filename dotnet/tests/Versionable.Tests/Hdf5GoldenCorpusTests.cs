using System.Globalization;
using System.Numerics;
using System.Numerics.Tensors;
using System.Text.Json;
using Versionable.Backends;
using Versionable.Backends.Hdf5;
using Versionable.Engine;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// The HDF5 half of the golden-corpus contract: the <c>.h5</c> files h5py wrote, read here and
/// compared against the manifests the Python suite compares against.
/// </summary>
/// <remarks>
/// Python counterpart: <c>tests/test_golden_corpus.py</c>, HDF5 rows. Nothing here regenerates
/// anything — the committed files are the contract.
/// <para>
/// This is not a second copy of <see cref="JsonGoldenCorpusTests"/> with a different extension.
/// The HDF5 files hold a <em>different structure</em>: no envelope key, no base64, no NPZ, no
/// JSON containers. Every value below therefore travels a path the JSON suite never exercises —
/// attributes for scalars and converter output, real datasets for arrays and scalar sequences,
/// groups for containers and nested objects — and arriving at the same manifest values is what
/// proves the two layouts describe one wire format.
/// </para>
/// </remarks>
[Collection(RegistryCollection.Name)]
public class Hdf5GoldenCorpusTests
{
    private static readonly string _goldenRoot = FindGoldenRoot();

    public Hdf5GoldenCorpusTests()
    {
        GoldenSchemas.EnsureRegistered();
    }

    // ------------------------------------------------------------------
    // Structural facts the whole corpus depends on
    // ------------------------------------------------------------------

    [Theory]
    [InlineData(".h5")]
    [InlineData(".hdf5")]
    public void the_backend_claims_both_hdf5_extensions(string extension)
    {
        Assert.IsType<Hdf5Backend>(BackendRegistry.Resolve($"whatever{extension}"));
    }

    /// <summary>
    /// Every fixture file in the corpus resolves its own type from the envelope in the file.
    /// </summary>
    /// <remarks>
    /// The dynamic <c>Load(path)</c> overload probes with <c>MetadataOnly</c>, so this is also
    /// the only test that runs the metadata-only path over the whole corpus — including
    /// <c>arrays</c>, where it is what stops the probe reading six datasets to answer a question
    /// the envelope already answers.
    /// </remarks>
    [Theory]
    [InlineData("containers", "containers.h5", typeof(GoldenContainers))]
    [InlineData("optionals", "optionals.h5", typeof(GoldenOptionals))]
    [InlineData("enums", "enums.h5", typeof(GoldenEnums))]
    [InlineData("literals", "literals.h5", typeof(GoldenLiterals))]
    [InlineData("nested", "nested.h5", typeof(GoldenNested))]
    [InlineData("polymorphic", "polymorphic.h5", typeof(GoldenPolymorphic))]
    [InlineData("migration-chain", "migration-chain.h5", typeof(GoldenWorker))]
    [InlineData("scalars", "scalars.h5", typeof(GoldenScalars))]
    [InlineData("temporal", "temporal.h5", typeof(GoldenTemporal))]
    [InlineData("stdlib", "stdlib.h5", typeof(GoldenStdlib))]
    [InlineData("arrays", "arrays.h5", typeof(GoldenArrays))]
    public void every_fixture_resolves_its_type_from_the_envelope_group(
        string fixture,
        string file,
        Type expected)
    {
        object loaded = VersionableFile.Load(Path.Combine(_goldenRoot, fixture, file));
        Assert.IsType(expected, loaded);
    }

    /// <summary>
    /// The envelope is a <c>__versionable__</c> child group, not attributes on the object.
    /// </summary>
    /// <remarks>
    /// Python's HDF5 backend puts the envelope in a child group precisely so a Versionable group
    /// is distinguishable from a plain collection group by structure alone, and both readers
    /// dispatch on that. A reader that looked for the envelope on the group's own attributes
    /// would still pass every value assertion below and then misread the first
    /// <c>dict[str, Inner]</c> it met.
    /// </remarks>
    [Fact]
    public void the_envelope_is_read_from_the_versionable_child_group()
    {
        string path = Path.Combine(_goldenRoot, "arrays", "arrays.h5");
        BackendLoadResult result = new Hdf5Backend().Load(
            path,
            new BackendLoadOptions { MetadataOnly = true, TargetMetadata = GoldenArrays.VersionableMetadata });

        Assert.Equal("GoldenArrays", result.Envelope.ObjectName);
        Assert.Equal(1, result.Envelope.Version);
        Assert.Equal("b76a00", result.Envelope.Hash);

        // MetadataOnly leaves the array fields unread, and says which they were.
        Assert.DoesNotContain("signal", result.Fields.Keys);
        Assert.Contains("signal", result.LazyFields!);
        Assert.Contains("traces", result.LazyFields!);
        Assert.Contains("channels", result.LazyFields!);

        // The probe VersionableFile.Load(path) runs has no TargetMetadata — it is reading the
        // file to find out what type to ask for. An unannotated dataset is assumed to be array
        // data there, which is the whole reason the probe is cheap; Python's _isArrayField makes
        // the same assumption for the same reason.
        BackendLoadResult probe = new Hdf5Backend().Load(path, new BackendLoadOptions { MetadataOnly = true });
        Assert.Equal("GoldenArrays", probe.Envelope.ObjectName);
        Assert.Contains("signal", probe.LazyFields!);
        Assert.DoesNotContain("signal", probe.Fields.Keys);

        // A list[float]-shaped dataset is not array data and stays eager, as it does in Python.
        BackendLoadResult containers = new Hdf5Backend().Load(
            Path.Combine(_goldenRoot, "containers", "containers.h5"),
            new BackendLoadOptions { MetadataOnly = true, TargetMetadata = GoldenContainers.VersionableMetadata });
        Assert.Contains("readings", containers.Fields.Keys);
        Assert.Empty(containers.LazyFields!);
    }

    // ------------------------------------------------------------------
    // Fixtures
    // ------------------------------------------------------------------

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void containers_load_from_datasets_groups_and_attributes(bool roundTrip)
    {
        JsonElement values = Values("containers");
        GoldenContainers loaded = Fixture<GoldenContainers>("containers", "containers.h5", roundTrip);

        Assert.Equal(Strings(values.GetProperty("names")), loaded.Names);
        Assert.Equal(Doubles(values.GetProperty("readings")), loaded.Readings);
        Assert.Equal(Ints(values.GetProperty("counts")), loaded.Counts);
        Assert.Equal(
            values.GetProperty("flags").EnumerateArray().Select(item => item.GetBoolean()),
            loaded.Flags);

        // Scalar dict values live in the subgroup's attributes, not in datasets.
        Assert.Equal(
            Pairs(values.GetProperty("lookup"))
                .ToDictionary(pair => pair.Key.GetString()!, pair => pair.Value.GetInt32()),
            loaded.Lookup);
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

        // A list of lists is a group of datasets; a dict of lists is the same group with names.
        Assert.Equal(
            values.GetProperty("matrix").EnumerateArray().Select(row => Doubles(row).ToList()),
            loaded.Matrix);
        Assert.Equal(
            Pairs(values.GetProperty("grouped")).ToDictionary(
                pair => pair.Key.GetString()!, pair => Doubles(pair.Value).ToList()),
            loaded.Grouped);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void enums_load_from_attributes_including_inside_containers(bool roundTrip)
    {
        JsonElement values = Values("enums");
        GoldenEnums loaded = Fixture<GoldenEnums>("enums", "enums.h5", roundTrip);

        Assert.Equal(Colour(values.GetProperty("colour")), loaded.Colour);
        Assert.Equal(
            (GoldenPriority)values.GetProperty("priority").GetProperty("$enum").GetProperty("value").GetInt32(),
            loaded.Priority);

        // A list of enums is a group whose elements are integer-named *attributes*, which is the
        // one place the sequence reader has to merge attributes with children.
        Assert.Equal(values.GetProperty("palette").EnumerateArray().Select(Colour), loaded.Palette);
        Assert.Equal(
            Pairs(values.GetProperty("byName"))
                .ToDictionary(pair => pair.Key.GetString()!, pair => Colour(pair.Value)),
            loaded.ByName);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void literals_load_and_keep_the_member_kind_python_chose(bool roundTrip)
    {
        JsonElement values = Values("literals");
        GoldenLiterals loaded = Fixture<GoldenLiterals>("literals", "literals.h5", roundTrip);

        Assert.Equal(values.GetProperty("mode").GetString(), loaded.Mode);
        Assert.Equal(values.GetProperty("level").GetInt32(), loaded.Level);
        Assert.Equal(values.GetProperty("flag").GetBoolean(), loaded.Flag);
        Assert.Equal(values.GetProperty("tag").GetInt64(), loaded.Tag);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void nested_objects_load_from_their_own_metadata_groups(bool roundTrip)
    {
        JsonElement values = Values("nested");
        GoldenNested loaded = Fixture<GoldenNested>("nested", "nested.h5", roundTrip);

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

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void a_base_typed_collection_comes_back_as_the_subclasses_the_file_names(bool roundTrip)
    {
        JsonElement values = Values("polymorphic");
        GoldenPolymorphic loaded = Fixture<GoldenPolymorphic>("polymorphic", "polymorphic.h5", roundTrip);

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

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void the_current_version_of_the_migration_fixture_loads_unmigrated(bool roundTrip)
    {
        AssertWorker(
            Values("migration-chain"),
            Fixture<GoldenWorker>("migration-chain", "migration-chain.h5", roundTrip));
    }

    [Fact]
    public void older_hdf5_files_migrate_forward_to_the_current_schema()
    {
        JsonElement manifest = Manifest("migration-chain");
        int sources = 0;

        foreach (JsonElement source in manifest.GetProperty("migrationSources").EnumerateArray())
        {
            string file = source.GetProperty("files").GetProperty("hdf5").GetString()!;
            AssertWorker(source.GetProperty("expected"), Load<GoldenWorker>("migration-chain", file));

            // The migrated object saved back out is a v3 file, and reads as one.
            AssertWorker(
                source.GetProperty("expected"),
                Fixture<GoldenWorker>("migration-chain", file, roundTrip: true));
            sources++;
        }

        Assert.Equal(2, sources);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void scalars_load_from_attributes_with_converter_output_inline(bool roundTrip)
    {
        JsonElement values = Values("scalars");
        GoldenScalars loaded = Fixture<GoldenScalars>("scalars", "scalars.h5", roundTrip);

        Assert.Equal(values.GetProperty("text").GetString(), loaded.Text);
        Assert.Equal(values.GetProperty("count").GetInt32(), loaded.Count);
        Assert.Equal(values.GetProperty("ratio").GetDouble(), loaded.Ratio);
        Assert.Equal(values.GetProperty("enabled").GetBoolean(), loaded.Enabled);

        // complex is a two-element float64 attribute here, not a JSON array and not a dataset:
        // Python's converter arm runs before its container arm, and so does the C# one.
        double[] parts = [.. Doubles(values.GetProperty("phase").GetProperty("$complex"))];
        Assert.Equal(new Complex(parts[0], parts[1]), loaded.Phase);

        Assert.Equal(
            Convert.FromBase64String(values.GetProperty("blob").GetProperty("$bytes").GetString()!),
            loaded.Blob);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void optionals_load_including_the_null_dataspace_attribute(bool roundTrip)
    {
        JsonElement values = Values("optionals");
        GoldenOptionals loaded = Fixture<GoldenOptionals>("optionals", "optionals.h5", roundTrip);

        Assert.Equal(values.GetProperty("present").GetString(), loaded.Present);

        // h5py.Empty("f") — a null dataspace, the one HDF5 spelling of "no value".
        Assert.Equal(JsonValueKind.Null, values.GetProperty("absent").ValueKind);
        Assert.Null(loaded.Absent);

        Assert.Equal(values.GetProperty("maybeCount").GetInt32(), loaded.MaybeCount);
        Assert.Equal(Decimal(values.GetProperty("money")), loaded.Money);
        Assert.Equal(values.GetProperty("either").GetString(), loaded.Either);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void temporal_values_load_with_microsecond_precision(bool roundTrip)
    {
        JsonElement values = Values("temporal");
        GoldenTemporal loaded = Fixture<GoldenTemporal>("temporal", "temporal.h5", roundTrip);

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

        // timedelta is the one converter whose output is a number, so it lands in a float64
        // attribute where every other converter type lands in a string one.
        Assert.Equal(
            TimeSpan.FromSeconds(values.GetProperty("elapsed").GetProperty("$timedeltaSeconds").GetDouble()),
            loaded.Elapsed);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void stdlib_converter_types_load_from_string_attributes(bool roundTrip)
    {
        JsonElement values = Values("stdlib");
        GoldenStdlib loaded = Fixture<GoldenStdlib>("stdlib", "stdlib.h5", roundTrip);

        Assert.Equal(PathValue(values.GetProperty("filePath")), loaded.FilePath.Value);
        Assert.Equal(PathValue(values.GetProperty("posixPath")), loaded.PosixPath.Value);
        Assert.Equal(PathValue(values.GetProperty("windowsPath")), loaded.WindowsPath.Value);

        Assert.Equal(Decimal(values.GetProperty("amount")), loaded.Amount);
        Assert.Equal(Guid.Parse(values.GetProperty("deviceId").GetProperty("$uuid").GetString()!), loaded.DeviceId);
        Assert.Equal(
            values.GetProperty("serialPattern").GetProperty("$pattern").GetString(),
            loaded.SerialPattern.ToString());
    }

    /// <summary>
    /// The fixture this backend exists for: five dtypes, a 2-D array, and arrays in containers.
    /// </summary>
    /// <remarks>
    /// Every one of these is a real HDF5 dataset with its dtype in the header — no NPZ, no
    /// base64. <c>mask</c> is the sharp case: numpy's <c>bool_</c> is an HDF5 enumeration, not a
    /// <c>uint8</c>, and reading it as the latter would fail the dtype check numpy's safe-cast
    /// table imposes on a <c>Tensor&lt;bool&gt;</c> field.
    /// </remarks>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void arrays_load_from_native_datasets_with_their_dtypes_and_shapes(bool roundTrip)
    {
        JsonElement values = Values("arrays");
        GoldenArrays loaded = Fixture<GoldenArrays>("arrays", "arrays.h5", roundTrip);

        AssertTensor(values.GetProperty("signal"), "float64", loaded.Signal, element => element.GetDouble());
        AssertTensor(values.GetProperty("weights"), "float32", loaded.Weights, element => element.GetSingle());
        AssertTensor(values.GetProperty("counts"), "int32", loaded.Counts, element => element.GetInt32());
        AssertTensor(values.GetProperty("image"), "uint8", loaded.Image, element => element.GetByte());
        AssertTensor(values.GetProperty("mask"), "bool", loaded.Mask, element => element.GetBoolean());

        // Shape is erased from the hash but not from the dataset header: a 2x3 stays a 2x3.
        AssertTensor(values.GetProperty("matrix"), "float64", loaded.Matrix, element => element.GetDouble());
        Assert.Equal(2, loaded.Matrix.Rank);

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
        (GoldenColour)WireValues.Read(
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

    /// <summary>
    /// The fixture, either as Python wrote it or after a trip through the C# writer.
    /// </summary>
    /// <remarks>
    /// Every value assertion in this file runs twice against this: once over h5py's bytes and
    /// once over PureHDF's. The second pass is the round-trip gate, and it is deliberately
    /// asserted against the <em>manifest</em> rather than against the first pass — an object
    /// compared only with itself would pass even if both directions were wrong in the same way.
    /// </remarks>
    /// <typeparam name="T">The fixture type.</typeparam>
    /// <param name="fixture">Corpus directory name.</param>
    /// <param name="file">File name inside it.</param>
    /// <param name="roundTrip">Whether to save and reload before asserting.</param>
    /// <returns>The loaded instance.</returns>
    private static T Fixture<T>(string fixture, string file, bool roundTrip)
        where T : IVersionableMetadataProvider
    {
        T loaded = Load<T>(fixture, file);
        if (!roundTrip)
        {
            return loaded;
        }

        using Hdf5TempFile temp = new();
        VersionableFile.Save(loaded, temp.Path);
        return VersionableFile.Load<T>(temp.Path);
    }

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
