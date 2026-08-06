using System.Collections.Frozen;
using System.Numerics.Tensors;
using Versionable.Backends;
using Versionable.Backends.Hdf5;
using Versionable.Errors;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// Fields a backend was told not to read: what the object comes back holding, and what it does not
/// come back saying.
/// </summary>
/// <remarks>
/// The HDF5 backend is the only one with anything to skip, so the end-to-end cases go through it —
/// writing the file through the backend API rather than reading a corpus file, so these stay
/// independent of the corpus and of HDF5's own Python-interchange suites. The engine-level cases
/// use a stub backend, because the behaviour is the materializer's, not HDF5's.
/// </remarks>
[Collection(RegistryCollection.Name)]
public class EngineLazyFieldTests : IDisposable
{
    private readonly string _directory =
        Path.Combine(Path.GetTempPath(), $"versionable-lazy-{Path.GetRandomFileName()}");

    public EngineLazyFieldTests()
    {
        GoldenSchemas.EnsureRegistered();
        Directory.CreateDirectory(_directory);
    }

    public void Dispose()
    {
        Directory.Delete(_directory, recursive: true);
        GC.SuppressFinalize(this);
    }

    // ------------------------------------------------------------------
    // Through the HDF5 backend, which is the one that actually skips reads
    // ------------------------------------------------------------------

    [Fact]
    public void a_metadata_only_load_reads_the_scalars_and_defaults_the_skipped_array()
    {
        string path = Path.Combine(_directory, "lazy.h5");
        VersionableFile.Save(
            new EngineLazyHolder("probe-A", 42, Tensor.Create(new[] { 1.0, 2.0, 3.0, 4.0 }, [(nint)4])), path);

        EngineLazyHolder loaded = VersionableFile.Load<EngineLazyHolder>(
            path, options: new VersionableLoadOptions { MetadataOnly = true });

        Assert.Equal("probe-A", loaded.Label);
        Assert.Equal(42, loaded.Count);

        // The declared default, not the four elements on disk: the load was told not to read them.
        Assert.Same(EngineLazyHolder.EmptySamples, loaded.Samples);
    }

    [Fact]
    public void the_same_file_loaded_eagerly_carries_the_array()
    {
        // The other half of the previous test: the array is on disk and readable, so the default
        // above is the option's doing rather than a write that never happened.
        string path = Path.Combine(_directory, "lazy.h5");
        VersionableFile.Save(
            new EngineLazyHolder("probe-A", 42, Tensor.Create(new[] { 1.0, 2.0, 3.0, 4.0 }, [(nint)4])), path);

        EngineLazyHolder loaded = VersionableFile.Load<EngineLazyHolder>(
            path, options: new VersionableLoadOptions { PreloadAll = true });

        Assert.Equal(4, loaded.Samples.FlattenedLength);
    }

    [Fact]
    public void a_metadata_only_load_of_a_field_with_no_default_names_the_option_as_the_cause()
    {
        // Was: "Field 'signal' is missing from the file and has no default value", which sends a
        // reader looking for a corrupt file instead of at the option they passed.
        string path = Path.Combine(_directory, "arrays.h5");
        VersionableFile.Save(SampleArrays(), path);

        ArrayNotLoadedException error = Assert.Throws<ArrayNotLoadedException>(
            () => VersionableFile.Load<GoldenArrays>(
                path, options: new VersionableLoadOptions { MetadataOnly = true }));

        Assert.Contains("was skipped by this load, not missing from the file", error.Message, StringComparison.Ordinal);
        Assert.Contains("MetadataOnly", error.Message, StringComparison.Ordinal);
        Assert.Contains("PreloadAll", error.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("is missing from the file and has no default", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_metadata_only_read_still_yields_the_envelope_and_names_what_it_skipped()
    {
        // What MetadataOnly is for: the envelope and the scalars, without paying for the arrays.
        string path = Path.Combine(_directory, "arrays.h5");
        VersionableFile.Save(SampleArrays(), path);

        BackendLoadResult result = new Hdf5Backend().Load(
            path,
            new BackendLoadOptions
            {
                TargetMetadata = GoldenArrays.VersionableMetadata,
                Preload = FrozenSet<string>.Empty,
                MetadataOnly = true,
            });

        Assert.Equal("GoldenArrays", result.Envelope.ObjectName);
        Assert.Equal(1, result.Envelope.Version);
        Assert.NotNull(result.LazyFields);
        Assert.Contains("signal", result.LazyFields!);
        Assert.DoesNotContain("signal", result.Fields.Keys);
    }

    [Fact]
    public void the_arrays_fixture_loads_whole_when_nothing_is_skipped()
    {
        string path = Path.Combine(_directory, "arrays.h5");
        VersionableFile.Save(SampleArrays(), path);

        GoldenArrays loaded = VersionableFile.Load<GoldenArrays>(
            path, options: new VersionableLoadOptions { PreloadAll = true });

        Assert.Equal(3, loaded.Signal.FlattenedLength);
        Assert.Equal(2, loaded.Traces.Count);
    }

    // ------------------------------------------------------------------
    // Nested objects, where a skip has to say which object it belongs to
    // ------------------------------------------------------------------

    [Fact]
    public void a_nested_skipped_field_is_recorded_path_qualified()
    {
        // The convention on BackendLoadResult.LazyFields: a bare name is the root, `inner/values`
        // is one level down. Without the qualification the engine cannot tell which object skipped
        // what, and a nested skip is indistinguishable from a field the file never had.
        string path = Path.Combine(_directory, "nested.h5");
        VersionableFile.Save(new EngineNestedStrict("outer", new EngineStrictArray(Samples())), path);

        BackendLoadResult result = new Hdf5Backend().Load(
            path,
            new BackendLoadOptions
            {
                TargetMetadata = EngineNestedStrict.Metadata,
                Preload = FrozenSet<string>.Empty,
                MetadataOnly = true,
            });

        Assert.Equal(["inner/values"], result.LazyFields!.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void a_nested_skipped_field_with_a_default_takes_the_default()
    {
        string path = Path.Combine(_directory, "nested-lazy.h5");
        VersionableFile.Save(new EngineNestedLazy("outer", new EngineLazyHolder("probe-A", 42, Samples())), path);

        EngineNestedLazy loaded = VersionableFile.Load<EngineNestedLazy>(
            path, options: new VersionableLoadOptions { MetadataOnly = true });

        Assert.Equal("outer", loaded.Label);
        Assert.Equal("probe-A", loaded.Inner.Label);
        Assert.Same(EngineLazyHolder.EmptySamples, loaded.Inner.Samples);
    }

    [Fact]
    public void a_nested_skipped_field_with_no_default_names_the_path_rather_than_the_bare_field()
    {
        // Reproduced by the reviewer before the fix as the verbatim "missing from the file"
        // message, one level down from where it had already been eliminated.
        string path = Path.Combine(_directory, "nested-strict.h5");
        VersionableFile.Save(new EngineNestedStrict("outer", new EngineStrictArray(Samples())), path);

        ArrayNotLoadedException error = Assert.Throws<ArrayNotLoadedException>(
            () => VersionableFile.Load<EngineNestedStrict>(
                path, options: new VersionableLoadOptions { MetadataOnly = true }));

        Assert.Contains("'inner/values'", error.Message, StringComparison.Ordinal);
        Assert.Contains("skipped by this load, not missing from the file", error.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("is missing from the file and has no default", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_skip_two_levels_down_inside_a_collection_still_resolves()
    {
        // The generated container reader hands each element to the engine without saying which
        // element it is, so the recorded path carries no index: `items/samples`, once, whatever
        // the element count. Every element shares one declared type, so a skip in one is a skip
        // in all.
        string path = Path.Combine(_directory, "deep.h5");
        VersionableFile.Save(
            new EngineDeepLazy(
                "outer",
                [new EngineLazyHolder("first", 1, Samples()), new EngineLazyHolder("second", 2, Samples())]),
            path);

        // One entry, not one per element: an index is not part of a skip's identity, and the
        // engine could never match one back — a generated container reader hands it an element,
        // never which element.
        BackendLoadResult probe = new Hdf5Backend().Load(
            path,
            new BackendLoadOptions
            {
                TargetMetadata = EngineDeepLazy.Metadata,
                Preload = FrozenSet<string>.Empty,
                MetadataOnly = true,
            });
        Assert.Equal(["items/samples"], probe.LazyFields!.Order(StringComparer.Ordinal));

        EngineDeepLazy loaded = VersionableFile.Load<EngineDeepLazy>(
            path, options: new VersionableLoadOptions { MetadataOnly = true });

        Assert.Equal(["first", "second"], loaded.Items.Select(item => item.Label));
        Assert.All(loaded.Items, item => Assert.Same(EngineLazyHolder.EmptySamples, item.Samples));
    }

    [Fact]
    public void a_skip_two_containers_down_resolves_without_any_depth_arithmetic()
    {
        // Dictionary<string, List<Inner>>: the case that broke index-qualified paths. The reader
        // records `groups/samples` once, and one segment of narrowing carries it to every element.
        string path = Path.Combine(_directory, "grouped.h5");
        VersionableFile.Save(
            new EngineGroupedLazy(new Dictionary<string, List<EngineLazyHolder>>(StringComparer.Ordinal)
            {
                ["g1"] = [new EngineLazyHolder("first", 1, Samples()), new EngineLazyHolder("second", 2, Samples())],
                ["g2"] = [new EngineLazyHolder("third", 3, Samples())],
            }),
            path);

        BackendLoadResult probe = new Hdf5Backend().Load(
            path,
            new BackendLoadOptions
            {
                TargetMetadata = EngineGroupedLazy.Metadata,
                Preload = FrozenSet<string>.Empty,
                MetadataOnly = true,
            });
        Assert.Equal(["groups/samples"], probe.LazyFields!.Order(StringComparer.Ordinal));

        EngineGroupedLazy loaded = VersionableFile.Load<EngineGroupedLazy>(
            path, options: new VersionableLoadOptions { MetadataOnly = true });

        Assert.Equal(["first", "second"], loaded.Groups["g1"].Select(item => item.Label));
        Assert.Equal(["third"], loaded.Groups["g2"].Select(item => item.Label));
        Assert.All(
            loaded.Groups.Values.SelectMany(list => list),
            item => Assert.Same(EngineLazyHolder.EmptySamples, item.Samples));
    }

    [Fact]
    public void a_skip_two_containers_down_with_no_default_still_names_the_path()
    {
        string path = Path.Combine(_directory, "grouped-strict.h5");
        VersionableFile.Save(
            new EngineGroupedStrict(new Dictionary<string, List<EngineStrictArray>>(StringComparer.Ordinal)
            {
                ["g1"] = [new EngineStrictArray(Samples())],
            }),
            path);

        ArrayNotLoadedException error = Assert.Throws<ArrayNotLoadedException>(
            () => VersionableFile.Load<EngineGroupedStrict>(
                path, options: new VersionableLoadOptions { MetadataOnly = true }));

        Assert.Contains("'groups/values'", error.Message, StringComparison.Ordinal);
        Assert.Contains("skipped by this load, not missing from the file", error.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("is missing from the file and has no default", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void three_container_levels_behave_exactly_like_one()
    {
        // List<Dictionary<string, Inner>>. Depth-independence is the point: the recorded path is
        // the same shape it would be with no containers at all.
        string path = Path.Combine(_directory, "layered.h5");
        VersionableFile.Save(
            new EngineLayeredLazy(
            [
                new Dictionary<string, EngineLazyHolder>(StringComparer.Ordinal)
                {
                    ["a"] = new EngineLazyHolder("first", 1, Samples()),
                },
                new Dictionary<string, EngineLazyHolder>(StringComparer.Ordinal)
                {
                    ["b"] = new EngineLazyHolder("second", 2, Samples()),
                },
            ]),
            path);

        BackendLoadResult probe = new Hdf5Backend().Load(
            path,
            new BackendLoadOptions
            {
                TargetMetadata = EngineLayeredLazy.Metadata,
                Preload = FrozenSet<string>.Empty,
                MetadataOnly = true,
            });
        Assert.Equal(["layers/samples"], probe.LazyFields!.Order(StringComparer.Ordinal));

        EngineLayeredLazy loaded = VersionableFile.Load<EngineLayeredLazy>(
            path, options: new VersionableLoadOptions { MetadataOnly = true });

        Assert.Equal(["first", "second"], loaded.Layers.Select(layer => layer.Values.Single().Label));
        Assert.All(
            loaded.Layers.SelectMany(layer => layer.Values),
            item => Assert.Same(EngineLazyHolder.EmptySamples, item.Samples));
    }

    [Fact]
    public void nesting_does_not_leak_a_skip_into_a_sibling_of_the_same_name()
    {
        // `label` is declared on both the outer and the inner type. Scoping by prefix is what stops
        // a skip of one from being read as a skip of the other; nothing here is skipped at the
        // root, so a leak would show up as the outer label going missing.
        string path = Path.Combine(_directory, "siblings.h5");
        VersionableFile.Save(new EngineNestedLazy("outer", new EngineLazyHolder("inner", 7, Samples())), path);

        EngineNestedLazy loaded = VersionableFile.Load<EngineNestedLazy>(
            path, options: new VersionableLoadOptions { MetadataOnly = true });

        Assert.Equal("outer", loaded.Label);
        Assert.Equal("inner", loaded.Inner.Label);
        Assert.Equal(7, loaded.Inner.Count);
    }

    // ------------------------------------------------------------------
    // The materializer's own behaviour, with no file in the way
    // ------------------------------------------------------------------

    [Fact]
    public void a_skipped_field_is_not_reported_as_a_missing_one()
    {
        // EngineSettings.required has no default, so the two paths produce different exceptions
        // from the same absence — which is the whole point.
        SkippingBackend skipping = new(["required"]);
        ArrayNotLoadedException skipped = Assert.Throws<ArrayNotLoadedException>(
            () => VersionableFile.Load<EngineSettings>("memory.stub", skipping));
        Assert.Contains("skipped by this load", skipped.Message, StringComparison.Ordinal);

        SkippingBackend silent = new([]);
        BackendException missing = Assert.Throws<BackendException>(
            () => VersionableFile.Load<EngineSettings>("memory.stub", silent));
        Assert.Contains("is missing from the file", missing.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_skipped_field_with_a_default_takes_the_default()
    {
        SkippingBackend backend = new(["name", "port"], ("required", "present"));

        EngineSettings loaded = VersionableFile.Load<EngineSettings>("memory.stub", backend);

        Assert.Equal("anon", loaded.Name);
        Assert.Equal(8080, loaded.Port);
    }

    private static Tensor<double> Samples() => Tensor.Create(new[] { 1.0, 2.0, 3.0, 4.0 }, [(nint)4]);

    private static GoldenArrays SampleArrays() => new()
    {
        Signal = Tensor.Create(new[] { 0.5, -1.25, 2.0 }, [(nint)3]),
        Weights = Tensor.Create(new[] { 0.5f, 0.25f }, [(nint)2]),
        Counts = Tensor.Create(new[] { -2, 7 }, [(nint)2]),
        Image = Tensor.Create(new byte[] { 0, 255 }, [(nint)2]),
        Mask = Tensor.Create(new[] { true, false }, [(nint)2]),
        Matrix = Tensor.Create(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, [(nint)2, 3]),
        Traces =
        [
            Tensor.Create(new[] { 1.0, 2.0 }, [(nint)2]),
            Tensor.Create(new[] { 3.0, 4.0, 5.0 }, [(nint)3]),
        ],
        Channels = new Dictionary<string, Tensor<double>>(StringComparer.Ordinal)
        {
            ["ch0"] = Tensor.Create(new[] { 0.25, 0.5 }, [(nint)2]),
        },
    };

    /// <summary>A backend that reports fields as skipped without supplying a value for them.</summary>
    /// <remarks>
    /// The shape the contract describes and the HDF5 backend produces: a name in
    /// <see cref="BackendLoadResult.LazyFields"/> and nothing under it in
    /// <see cref="BackendLoadResult.Fields"/>.
    /// </remarks>
    private sealed class SkippingBackend(IEnumerable<string> skipped, params (string Key, object? Value)[] present)
        : IVersionableBackend
    {
        public IReadOnlySet<Type> NativeTypes => FrozenSet<Type>.Empty;

        public void Save(
            IReadOnlyDictionary<string, object?> fields,
            EnvelopeMetadata envelope,
            string path,
            VersionableMetadata metadata,
            BackendSaveOptions options) => throw new NotSupportedException();

        public BackendLoadResult Load(string path, BackendLoadOptions options)
        {
            Dictionary<string, object?> fields = new(StringComparer.Ordinal);
            foreach ((string key, object? value) in present)
            {
                fields[key] = value;
            }

            return new BackendLoadResult(
                fields,
                new EnvelopeMetadata("EngineSettings", 1, "ffffff"),
                skipped.ToFrozenSet(StringComparer.Ordinal));
        }
    }
}
