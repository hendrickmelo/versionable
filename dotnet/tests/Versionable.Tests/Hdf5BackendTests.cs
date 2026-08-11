using System.Numerics;
using System.Numerics.Tensors;
using PureHDF;
using PureHDF.VOL.Native;
using Versionable.Backends;
using Versionable.Backends.Hdf5;
using Versionable.Errors;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// What the golden corpus cannot say: the shape of the bytes the C# writer produces, the
/// compression settings on them, and the dtype rules on the way back in.
/// </summary>
/// <remarks>
/// The golden suite proves the two implementations agree about <em>values</em>. This one proves
/// the C# side agrees with the format about <em>structure</em> — that an array is a chunked,
/// filtered dataset and not a base64 string in an attribute, that a null is a null dataspace,
/// that a dictionary key with a slash in it survives HDF5's path syntax. A backend can pass
/// every value assertion in the corpus and still write files no other tool can use.
/// </remarks>
[Collection(RegistryCollection.Name)]
public class Hdf5BackendTests
{
    public Hdf5BackendTests()
    {
        GoldenSchemas.EnsureRegistered();
        VersionableRegistry.Register(Hdf5Shapes.VersionableMetadata);
        VersionableRegistry.Register(Hdf5Single.VersionableMetadata);
        VersionableRegistry.Register(Hdf5Double.VersionableMetadata);
    }

    // ------------------------------------------------------------------
    // Native type mapping
    // ------------------------------------------------------------------

    /// <summary>
    /// Every field lands on the HDF5 construct Python's backend puts it on.
    /// </summary>
    /// <remarks>
    /// Read with PureHDF's raw API rather than through the engine, so the assertions are about
    /// the file and not about whether this backend can read what it wrote.
    /// </remarks>
    [Fact]
    public void every_field_maps_onto_the_native_hdf5_construct()
    {
        using Hdf5TempFile temp = new();
        VersionableFile.Save(Sample(), temp.Path);

        using NativeFile file = H5File.OpenRead(temp.Path);

        // The envelope is a child group, which is what distinguishes an object group from a
        // collection group.
        IH5Group envelope = file.Group(VersionableEnvelope.WrappedKey);
        Assert.Equal("Hdf5Shapes", envelope.Attribute("object").Read<string[]>()[0]);
        Assert.Equal(1L, envelope.Attribute("version").Read<long[]>()[0]);
        Assert.Equal(Hdf5Shapes.VersionableMetadata.Hash, envelope.Attribute("hash").Read<string[]>()[0]);

        // Scalars are attributes on the object group itself.
        Assert.Equal("probe", file.Attribute("label").Read<string[]>()[0]);

        // A null is a null dataspace — h5py.Empty, the one HDF5 spelling of "no value".
        Assert.Equal(H5DataspaceType.Null, file.Attribute("note").Space.Type);

        // A list of scalars is a real 1-D dataset, typed int64 as a Python int would be.
        IH5Dataset counts = file.Dataset("counts");
        Assert.Equal(H5DataTypeClass.FixedPoint, counts.Type.Class);
        Assert.Equal(8, counts.Type.Size);
        Assert.Equal([3UL], counts.Space.Dimensions);
        Assert.Equal<long>([2, 3, 5], counts.Read<long[]>());

        // Arrays keep their own dtype: float64, float16, and a complex compound.
        Assert.Equal(H5DataTypeClass.FloatingPoint, file.Dataset("signal").Type.Class);
        Assert.Equal(8, file.Dataset("signal").Type.Size);
        Assert.Equal([2UL, 3UL], file.Dataset("signal").Space.Dimensions);
        Assert.Equal(2, file.Dataset("wide").Type.Size);
        Assert.Equal(H5DataTypeClass.Compound, file.Dataset("spin").Type.Class);
        Assert.Equal(16, file.Dataset("spin").Type.Size);

        // A dict is a group whose child names are its keys, with scalar values as attributes.
        IH5Group keyed = file.Group("keyed");
        Assert.Equal(0.5, keyed.Attribute("alpha").Read<double[]>()[0]);

        // '/' is HDF5's path separator, so a key holding one is percent-encoded — and '%'
        // has to be escaped too or the escaping would not be reversible.
        Assert.True(keyed.AttributeExists("a%2Fb"));
        Assert.True(keyed.AttributeExists("100%25"));

        // An empty sequence still gets a dataset, typed from the declaration.
        IH5Dataset empty = file.Dataset("empty");
        Assert.Equal([0UL], empty.Space.Dimensions);
        Assert.Equal(H5DataTypeClass.VariableLength, empty.Type.Class);
    }

    /// <summary>Nothing in an HDF5 file is base64, JSON, or NPY.</summary>
    /// <remarks>
    /// The negative half of the mapping, and the one a lazy implementation would fail: writing
    /// an array through the text backends' <c>Tensor</c> converter would produce a perfectly
    /// round-trippable file whose arrays no other HDF5 tool could read.
    /// </remarks>
    [Fact]
    public void arrays_are_datasets_rather_than_encoded_payloads()
    {
        using Hdf5TempFile temp = new();
        VersionableFile.Save(Sample(), temp.Path);

        using NativeFile file = H5File.OpenRead(temp.Path);
        Assert.False(file.AttributeExists("signal"));
        Assert.IsAssignableFrom<IH5Dataset>(file.Get("signal"));

        // The NPZ marker key the JSON/YAML/TOML wire form carries appears nowhere.
        string contents = System.Text.Encoding.ASCII.GetString(File.ReadAllBytes(temp.Path));
        Assert.DoesNotContain("__ver_ndarray__", contents, StringComparison.Ordinal);
        Assert.DoesNotContain("PK", contents, StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------
    // Compression
    // ------------------------------------------------------------------

    /// <summary>Array datasets are chunked and filtered; container datasets are not.</summary>
    /// <remarks>
    /// Python compresses what <c>create_dataset</c> is called with from the ndarray arm and
    /// leaves the scalar-sequence arm alone, and the golden files show exactly that split. It
    /// matters: a <c>list[float]</c> of three elements pays a chunk index and a filter pipeline
    /// to save nothing.
    /// <para>
    /// <b>The filter pipeline itself is not assertable from C#.</b> <c>IH5DataLayout</c> exposes
    /// <c>Class</c> and <c>Chunks</c> and nothing about filters, and PureHDF keeps the pipeline
    /// message internal — so a reader cannot ask a dataset which filters it carries. The C# side
    /// therefore checks layout, chunk shape, and (in
    /// <see cref="the_default_pipeline_compresses_and_the_data_survives_it"/>) file size; naming
    /// the filters <c>gzip</c> and <c>shuffle</c> on a C#-written file is left to an out-of-band
    /// h5py check, since h5py can introspect a dataset's filter pipeline directly and PureHDF
    /// cannot.
    /// </para>
    /// </remarks>
    [Fact]
    public void arrays_are_chunked_and_filtered_and_containers_are_not()
    {
        using Hdf5TempFile temp = new();
        VersionableFile.Save(Sample(), temp.Path);

        using NativeFile file = H5File.OpenRead(temp.Path);

        IH5Dataset signal = file.Dataset("signal");
        Assert.Equal(H5DataLayoutClass.Chunked, signal.Layout.Class);
        Assert.Equal([2U, 3U], signal.Layout.Chunks);

        Assert.Equal(H5DataLayoutClass.Contiguous, file.Dataset("counts").Layout.Class);
    }

    [Fact]
    public void uncompressed_writes_contiguous_datasets_that_still_round_trip()
    {
        using Hdf5TempFile temp = new();
        VersionableFile.Save(
            Sample(),
            temp.Path,
            options: new BackendSaveOptions
            {
                BackendOptions = new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    [Hdf5Compression.OptionKey] = Hdf5Compression.Uncompressed,
                },
            });

        using (NativeFile file = H5File.OpenRead(temp.Path))
        {
            Assert.Equal(H5DataLayoutClass.Contiguous, file.Dataset("signal").Layout.Class);
        }

        Assert.Equal(Sample().Signal.ToArray(), VersionableFile.Load<Hdf5Shapes>(temp.Path).Signal.ToArray());
    }

    /// <summary>
    /// The default filter pipeline actually compresses, and the data survives it.
    /// </summary>
    /// <remarks>
    /// Size is the only thing available: PureHDF exposes no filter pipeline on a dataset it
    /// reads, and correct data coming back proves only that the filters are symmetric, not that
    /// they ran. An out-of-band h5py check against a C#-written file covers the other half — that
    /// the filters are named <c>gzip</c> and <c>shuffle</c>.
    /// </remarks>
    [Fact]
    public void the_default_pipeline_compresses_and_the_data_survives_it()
    {
        Hdf5Double large = new() { Values = Tensor.Create([.. Enumerable.Range(0, 20_000).Select(i => i * 0.5)]) };

        using Hdf5TempFile compressed = new();
        using Hdf5TempFile plain = new();
        VersionableFile.Save(large, compressed.Path);
        VersionableFile.Save(
            large,
            plain.Path,
            options: new BackendSaveOptions
            {
                BackendOptions = new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    [Hdf5Compression.OptionKey] = Hdf5Compression.Uncompressed,
                },
            });

        Assert.True(
            compressed.Length * 2 < plain.Length,
            $"gzip+shuffle wrote {compressed.Length} bytes against {plain.Length} uncompressed, which is "
                + "not the order-of-magnitude difference a working pipeline gives on a linear ramp.");

        Assert.Equal(large.Values.ToArray(), VersionableFile.Load<Hdf5Double>(compressed.Path).Values.ToArray());
    }

    /// <summary>
    /// Blosc2 round-trips where its native library exists, and says so where it does not.
    /// </summary>
    /// <remarks>
    /// <c>Blosc2.PInvoke</c> ships binaries for win-x86, win-x64, and linux-x64 only, so this
    /// exercises the filter on CI and the diagnostic on an Apple-silicon workstation. Asserting
    /// only one of the two would either fail every local run or pass without ever compressing
    /// anything.
    /// </remarks>
    [Fact]
    public void blosc_round_trips_through_the_registered_filter()
    {
        Hdf5Double large = new() { Values = Tensor.Create([.. Enumerable.Range(0, 5_000).Select(i => i * 0.25)]) };

        using Hdf5TempFile temp = new();
        BackendSaveOptions options = new()
        {
            BackendOptions = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                [Hdf5Compression.OptionKey] = Hdf5Compression.Blosc,
            },
        };

        try
        {
            VersionableFile.Save(large, temp.Path, options: options);
        }
        catch (BackendException error) when (HasMissingNativeLibrary(error))
        {
            Assert.Contains("native library", error.Message, StringComparison.Ordinal);
            Assert.Contains("Hdf5Compression.Gzip", error.Message, StringComparison.Ordinal);
            return;
        }

        using (NativeFile file = H5File.OpenRead(temp.Path))
        {
            Assert.Equal(H5DataLayoutClass.Chunked, file.Dataset("values").Layout.Class);
        }

        Assert.Equal(large.Values.ToArray(), VersionableFile.Load<Hdf5Double>(temp.Path).Values.ToArray());
    }

    private static bool HasMissingNativeLibrary(Exception error)
    {
        for (Exception? cause = error; cause is not null; cause = cause.InnerException)
        {
            if (cause is DllNotFoundException)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// The gzip level a file records is the nearest one PureHDF can apply.
    /// </summary>
    /// <remarks>
    /// PureHDF's deflate filter wraps <see cref="System.IO.Compression.ZLibStream"/>, whose four
    /// levels — 0, 1, 6, 9 — are the only writable ones. Python's default of 4 is not among them
    /// and rounds to 6, zlib's own default. This pins the mapping so the divergence stays a
    /// documented number rather than a surprise in a file someone diffs.
    /// </remarks>
    [Theory]
    [InlineData(null, 6)]
    [InlineData(0, 0)]
    [InlineData(1, 1)]
    [InlineData(3, 1)]
    [InlineData(4, 6)]
    [InlineData(6, 6)]
    [InlineData(7, 6)]
    [InlineData(8, 9)]
    [InlineData(9, 9)]
    public void the_requested_gzip_level_maps_onto_one_purehdf_can_write(int? requested, int effective)
    {
        Assert.Equal(effective, new Hdf5Compression { Level = requested }.EffectiveGzipLevel);
    }

    /// <summary>Every level the mapping can produce actually writes and reads back.</summary>
    /// <remarks>
    /// The mapping above is arithmetic; this is the claim that matters — that PureHDF accepts
    /// each of the four and that the data survives. It is also where a PureHDF upgrade that
    /// widened or narrowed the accepted set would surface.
    /// </remarks>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(9)]
    public void every_mapped_gzip_level_writes_and_reads_back(int level)
    {
        Hdf5Double values = new() { Values = Tensor.Create([.. Enumerable.Range(0, 2_000).Select(i => i * 0.5)]) };

        using Hdf5TempFile temp = new();
        VersionableFile.Save(
            values,
            temp.Path,
            options: new BackendSaveOptions
            {
                BackendOptions = new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    [Hdf5Compression.OptionKey] = new Hdf5Compression { Level = level },
                },
            });

        Assert.Equal(values.Values.ToArray(), VersionableFile.Load<Hdf5Double>(temp.Path).Values.ToArray());
    }

    // ------------------------------------------------------------------
    // Dtypes
    // ------------------------------------------------------------------

    /// <summary>A widening the numpy table calls safe is applied without comment.</summary>
    [Fact]
    public void a_safe_dtype_widening_is_applied_on_load()
    {
        using Hdf5TempFile temp = new();
        VersionableFile.Save(new Hdf5Single { Values = Tensor.Create([1.5f, -2.25f]) }, temp.Path);

        Hdf5Double widened = VersionableFile.Load<Hdf5Double>(temp.Path);
        Assert.Equal<double>([1.5, -2.25], widened.Values.ToArray());
    }

    /// <summary>
    /// A narrowing it does not is a <see cref="DtypeMismatchException"/> at load, not at first
    /// element access.
    /// </summary>
    /// <remarks>
    /// The dtype lives in the dataset header, so it is knowable before a single element is
    /// read — which is the whole reason Python checks it there too rather than letting the cast
    /// fail somewhere downstream with no field name attached.
    /// </remarks>
    [Fact]
    public void an_unsafe_dtype_narrowing_fails_the_load()
    {
        using Hdf5TempFile temp = new();
        VersionableFile.Save(new Hdf5Double { Values = Tensor.Create([1.5, -2.25]) }, temp.Path);

        DtypeMismatchException error = Assert.Throws<DtypeMismatchException>(
            () => VersionableFile.Load<Hdf5Single>(temp.Path));

        Assert.Contains("float32", error.Message, StringComparison.Ordinal);
        Assert.Contains("float64", error.Message, StringComparison.Ordinal);
        Assert.Contains("values", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void an_empty_array_round_trips_as_an_empty_array()
    {
        using Hdf5TempFile temp = new();
        VersionableFile.Save(new Hdf5Double { Values = Tensor.Create<double>([], [0]) }, temp.Path);

        Hdf5Double loaded = VersionableFile.Load<Hdf5Double>(temp.Path);
        Assert.Equal(0, loaded.Values.FlattenedLength);
    }

    // ------------------------------------------------------------------
    // Failure modes
    // ------------------------------------------------------------------

    /// <summary>
    /// A field this backend skipped is reported as skipped, not as missing from the file.
    /// </summary>
    /// <remarks>
    /// The end-to-end half of the <c>MetadataOnly</c> contract, and the one that needed an engine
    /// change to hold: the backend leaves the array out of
    /// <c>BackendLoadResult.Fields</c> and names it in <c>LazyFields</c>, and until
    /// <c>Engine.ObjectMaterializer</c> learned to read that set, a required array field surfaced
    /// as "missing from the file and has no default" — a message that sends the reader looking
    /// for a corrupt file instead of at their own load options. It now raises
    /// <see cref="ArrayNotLoadedException"/> naming the field and the way out.
    /// </remarks>
    [Fact]
    public void metadata_only_reports_a_skipped_array_field_as_skipped()
    {
        using Hdf5TempFile temp = new();
        VersionableFile.Save(new Hdf5Double { Values = Tensor.Create([1.5, -2.25]) }, temp.Path);

        ArrayNotLoadedException error = Assert.Throws<ArrayNotLoadedException>(
            () => VersionableFile.Load<Hdf5Double>(
                temp.Path, options: new VersionableLoadOptions { MetadataOnly = true }));

        Assert.Contains("values", error.Message, StringComparison.Ordinal);
        Assert.Contains("skipped by this load", error.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("missing from the file and has no default", error.Message, StringComparison.Ordinal);

        // The same file loads whole the moment the caller asks for the data.
        Hdf5Double loaded = VersionableFile.Load<Hdf5Double>(
            temp.Path, options: new VersionableLoadOptions { PreloadAll = true });
        Assert.Equal<double>([1.5, -2.25], loaded.Values.ToArray());
    }

    [Fact]
    public void a_missing_file_is_a_backend_exception()
    {
        BackendException error = Assert.Throws<BackendException>(
            () => VersionableFile.Load<Hdf5Double>(Path.Combine(Path.GetTempPath(), "versionable-absent.h5")));

        Assert.Contains("Failed to read HDF5", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_file_that_is_not_hdf5_is_a_backend_exception()
    {
        using Hdf5TempFile temp = new();
        File.WriteAllText(temp.Path, "not an HDF5 file");

        Assert.Throws<BackendException>(() => VersionableFile.Load<Hdf5Double>(temp.Path));
    }

    [Fact]
    public void the_backend_declares_every_tensor_element_type_native()
    {
        IReadOnlySet<Type> native = new Hdf5Backend().NativeTypes;

        Assert.Equal(13, native.Count);
        Assert.Contains(typeof(Tensor<double>), native);
        Assert.Contains(typeof(Tensor<bool>), native);
        Assert.Contains(typeof(Tensor<Complex>), native);
        Assert.DoesNotContain(typeof(double[]), native);
    }

    private static Hdf5Shapes Sample() => new()
    {
        Label = "probe",
        Counts = [2, 3, 5],
        Keyed = new Dictionary<string, double>(StringComparer.Ordinal)
        {
            ["alpha"] = 0.5,
            ["a/b"] = 1.5,
            ["100%"] = 2.5,
        },
        Note = null,
        Signal = Tensor.Create([1.0, 2, 3, 4, 5, 6], [2, 3]),
        Wide = Tensor.Create([(Half)1.5, (Half)2.5]),
        Spin = Tensor.Create([new Complex(1, 2), new Complex(-3, 4)]),
        Empty = [],
    };
}

/// <summary>Covers the constructs the golden corpus exercises only from the read side.</summary>
[Versionable(Version = 1, Hash = "7dd31b")]
internal sealed partial class Hdf5Shapes
{
    /// <summary>A string, which becomes an attribute.</summary>
    [VersionableField("label")]
    public required string Label { get; init; }

    /// <summary>A list of integers, which becomes a contiguous int64 dataset.</summary>
    [VersionableField("counts")]
    public required List<int> Counts { get; init; }

    /// <summary>A dictionary whose keys need HDF5 name escaping.</summary>
    [VersionableField("keyed")]
    public required Dictionary<string, double> Keyed { get; init; }

    /// <summary>A null, which becomes a null-dataspace attribute.</summary>
    [VersionableField("note")]
    public string? Note { get; init; }

    /// <summary>A 2-D float64 array.</summary>
    [VersionableField("signal")]
    public required Tensor<double> Signal { get; init; }

    /// <summary>A float16 array, which no golden fixture carries.</summary>
    [VersionableField("wide")]
    public required Tensor<Half> Wide { get; init; }

    /// <summary>A complex128 array, stored as HDF5's two-member compound.</summary>
    [VersionableField("spin")]
    public required Tensor<Complex> Spin { get; init; }

    /// <summary>An empty sequence, whose dataset type comes from the declaration alone.</summary>
    [VersionableField("empty")]
    public required List<string> Empty { get; init; }
}

/// <summary>A float32 array under a wire name <see cref="Hdf5Double"/> also declares.</summary>
[Versionable(Version = 1, Hash = "9af342")]
internal sealed partial class Hdf5Single
{
    /// <summary>The array.</summary>
    [VersionableField("values")]
    public required Tensor<float> Values { get; init; }
}

/// <summary>A float64 array under a wire name <see cref="Hdf5Single"/> also declares.</summary>
[Versionable(Version = 1, Hash = "b34d4a")]
internal sealed partial class Hdf5Double
{
    /// <summary>The array.</summary>
    [VersionableField("values")]
    public required Tensor<double> Values { get; init; }
}
