using System.Collections.Frozen;
using System.Runtime.CompilerServices;
using PureHDF.Filters;
using Versionable.Engine;

namespace Versionable.Backends.Hdf5;

/// <summary>
/// Reads and writes HDF5 with native type mapping: no JSON, no base64, no NPY.
/// </summary>
/// <remarks>
/// Python counterpart: <c>Hdf5Backend</c> in <c>src/versionable/_hdf5_backend.py</c>. Every
/// field maps onto a real HDF5 construct — scalars and converter output onto attributes, arrays
/// onto chunked and compressed datasets, sequences of scalars onto 1-D datasets, containers and
/// nested objects onto groups — which is the point of the backend: an <c>h5ls</c> of a saved
/// object shows the object, and MATLAB or a plotting tool can read a dataset without knowing
/// anything about Versionable.
/// <para>
/// <b>Loads are eager (v1).</b> Python's HDF5 backend hands back <c>LazyArray</c> sentinels for
/// array fields and materializes them on first access; the C# contract keeps the seam —
/// <see cref="BackendLoadOptions.Preload"/> and <see cref="BackendLoadResult.LazyFields"/> —
/// but this implementation reads array data eagerly and leaves <c>LazyFields</c> empty unless
/// <see cref="BackendLoadOptions.MetadataOnly"/> is set. The consequence is honest and bounded:
/// <c>Load&lt;T&gt;()</c> on a large file costs what the file costs, where Python would defer
/// it. Lazy slicing is Tier 3 and post-v1; nothing in the contract has to change to add it,
/// because the seam it needs is the one being left empty here.
/// </para>
/// <para>
/// <see cref="BackendLoadOptions.MetadataOnly"/> <em>is</em> honored, and not as a courtesy:
/// <see cref="VersionableFile.Load(string, IVersionableBackend?, VersionableLoadOptions?)"/>
/// probes the envelope with it before it knows what type to build, so a backend that ignored it
/// would read every array of every file twice.
/// </para>
/// <para>
/// <b>Compression.</b> Array datasets are chunked and filtered per
/// <see cref="Hdf5Compression"/> — gzip with shuffle by default, matching Python. Pass a
/// different setting through
/// <c>BackendSaveOptions.BackendOptions[Hdf5Compression.OptionKey]</c>. Reading needs no option:
/// the filter pipeline is in the file.
/// </para>
/// <para>
/// <b>Which filters are readable.</b> Gzip and shuffle are built into HDF5. Beyond those, the
/// module initializer registers <c>PureHDF.Filters.Blosc2Filter</c>, which despite its package
/// name implements filter <b>32001</b> — blosc <em>v1</em>, what Python's
/// <c>hdf5plugin.Blosc</c> writes. Python's own <c>blosc</c> preset goes through
/// <c>hdf5plugin.Blosc2</c> and writes filter <b>32026</b>, which nothing here registers, so
/// such a file fails to read on every platform. So do Python's <c>zstd</c> (32015) and
/// <c>lzf</c> (32000) presets. A read that trips one of these gets a message naming the gap
/// (<see cref="Hdf5Diagnostics"/>); closing it is tracked for phase 5.
/// </para>
/// </remarks>
public sealed class Hdf5Backend : IVersionableBackend
{
    private static readonly FrozenSet<Type> _nativeTypes = Hdf5Arrays.TensorTypes().ToFrozenSet();

    /// <summary>File extensions this backend claims.</summary>
    public static IReadOnlyList<string> Extensions { get; } = [".h5", ".hdf5"];

    /// <inheritdoc/>
    /// <remarks>
    /// The thirteen closed <c>Tensor&lt;T&gt;</c> types. Declaring them native is what makes the
    /// engine hand arrays through to this backend intact instead of lowering them to a base64
    /// NPZ, and — on the way back — what makes <c>Converters.TensorConverter{T}</c> accept the
    /// tensor this backend built from the dataset. Python's counterpart is the one-element
    /// <c>nativeTypes = {np.ndarray}</c>.
    /// </remarks>
    public IReadOnlySet<Type> NativeTypes => _nativeTypes;

    /// <inheritdoc/>
    public void Save(
        IReadOnlyDictionary<string, object?> fields,
        EnvelopeMetadata envelope,
        string path,
        VersionableMetadata metadata,
        BackendSaveOptions options)
    {
        ArgumentNullException.ThrowIfNull(fields);
        ArgumentNullException.ThrowIfNull(envelope);
        ArgumentNullException.ThrowIfNull(metadata);
        ArgumentNullException.ThrowIfNull(options);

        IReadOnlyDictionary<string, object?> wire = WireValues.WriteFields(fields, metadata, NativeTypes);
        Hdf5Writer.Write(wire, envelope, metadata, path, CompressionFrom(options));
    }

    /// <inheritdoc/>
    public BackendLoadResult Load(string path, BackendLoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return Hdf5Reader.Read(path, options);
    }

    /// <summary>Registers this backend for <see cref="Extensions"/>.</summary>
    /// <remarks>
    /// Public for the same reason <c>Json.JsonBackend.RegisterExtensions</c> is: anything that
    /// clears <see cref="BackendRegistry"/> needs a way to put the built-ins back, and a module
    /// initializer cannot be re-run.
    /// </remarks>
    public static void RegisterExtensions()
    {
        BackendRegistry.Register(Extensions, static () => new Hdf5Backend());
        RegisterFilters();
    }

    /// <summary>
    /// Registers the third-party filters PureHDF does not ship enabled.
    /// </summary>
    /// <remarks>
    /// <see cref="H5Filter.Register"/> is process-wide, which is why this runs once from the
    /// module initializer rather than per file. <c>Blosc2Filter</c> registers HDF5 filter
    /// <b>32001</b> (blosc v1), not 32026 — see the note on the type. Nothing registers 32026,
    /// 32015 (zstd), or 32000 (lzf), so a file using one of those reports an unregistered filter
    /// rather than decoding to noise.
    /// </remarks>
    private static void RegisterFilters() => H5Filter.Register(new Blosc2Filter());

    // CA2255 warns off module initializers in libraries. The intent is the same as
    // JsonBackend's: BackendRegistry's contract is that every C# backend ships in the one
    // package and registers unconditionally, so `Save(config, "run.h5")` works with no
    // initialization call — which is what Python's `import versionable.hdf5` buys there.
#pragma warning disable CA2255
    [ModuleInitializer]
#pragma warning restore CA2255
    internal static void Register() => RegisterExtensions();

    private static Hdf5Compression CompressionFrom(BackendSaveOptions options) =>
        options.BackendOptions.TryGetValue(Hdf5Compression.OptionKey, out object? value)
            && value is Hdf5Compression compression
            ? compression
            : Hdf5Compression.Default;
}
