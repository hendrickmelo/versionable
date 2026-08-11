using PureHDF.Filters;

namespace Versionable.Backends.Hdf5;

/// <summary>Compression filter applied to array datasets.</summary>
/// <remarks>
/// Python counterpart: the <c>Hdf5CompressionAlgorithm</c> literal alias in
/// <c>src/versionable/_hdf5_compression.py</c>. Python's <c>lzf</c> and <c>zstd</c> have no
/// member here: PureHDF ships neither filter, so offering them would only produce files this
/// build cannot read back.
/// </remarks>
public enum Hdf5CompressionAlgorithm
{
    /// <summary>No filter. Datasets are written contiguous rather than chunked.</summary>
    None = 0,

    /// <summary>The built-in HDF5 deflate filter, which every HDF5 tool can read.</summary>
    Gzip = 1,

    /// <summary>
    /// Blosc <b>version 1</b> (HDF5 filter 32001), through <c>PureHDF.Filters.Blosc2</c>.
    /// </summary>
    /// <remarks>
    /// <b>Not the same filter as Python's <c>blosc</c> preset, despite the package name.</b>
    /// <c>PureHDF.Filters.Blosc2Filter.Id</c> is <b>32001</b> — the blosc v1 container format,
    /// which the Blosc2 library can also produce. Python's
    /// <c>Hdf5Compression(algorithm="blosc")</c> routes through <c>hdf5plugin.Blosc2</c>, which
    /// writes filter <b>32026</b>. The two directions are therefore asymmetric:
    /// <list type="bullet">
    ///   <item>
    ///     <description>
    ///     A file this backend writes is readable in Python through <c>hdf5plugin.Blosc</c>
    ///     (v1), which registers 32001. Plain <c>h5py</c> without that plugin cannot read it.
    ///     </description>
    ///   </item>
    ///   <item>
    ///     <description>
    ///     A file Python's <c>BLOSC_DEFAULT</c> writes is <b>not readable here on any
    ///     platform</b>: nothing registers 32026, so the read fails with a filter-id mismatch.
    ///     This is a genuine interchange gap, not a platform issue; it is tracked for phase 5.
    ///     </description>
    ///   </item>
    /// </list>
    /// <para>
    /// <b>Separately, the filter is not available on every platform.</b> It calls into
    /// <c>libblosc2</c> through <c>Blosc2.PInvoke</c>, which ships native binaries for win-x86,
    /// win-x64, and linux-x64 only — there is no macOS build in 2.7.3. Saving with this on a Mac
    /// raises a <see cref="Errors.BackendException"/> naming the missing library and pointing at
    /// <see cref="Hdf5CompressionAlgorithm.Gzip"/>, which is built into the HDF5 format and
    /// needs nothing native.
    /// </para>
    /// </remarks>
    Blosc = 2,
}

/// <summary>Inner codec Blosc2 compresses each block with.</summary>
/// <remarks>
/// Python counterpart: the <c>BloscCompressor</c> literal alias, whose values are the strings
/// Blosc2 itself uses. The names round-trip through
/// <see cref="Blosc2Filter.COMPRESSOR_CODE"/>.
/// </remarks>
public enum Hdf5BloscCompressor
{
    /// <summary><c>blosclz</c>.</summary>
    BloscLz = 0,

    /// <summary><c>lz4</c>.</summary>
    Lz4 = 1,

    /// <summary><c>lz4hc</c>.</summary>
    Lz4Hc = 2,

    /// <summary><c>zlib</c>.</summary>
    Zlib = 3,

    /// <summary><c>zstd</c>.</summary>
    Zstd = 4,
}

/// <summary>
/// How <see cref="Hdf5Backend"/> compresses array datasets.
/// </summary>
/// <remarks>
/// Python counterpart: the <c>Hdf5Compression</c> dataclass and its presets in
/// <c>src/versionable/_hdf5_compression.py</c>. Pass one through
/// <c>BackendSaveOptions.BackendOptions[Hdf5Compression.OptionKey]</c>, the C# spelling of
/// Python's <c>versionable.save(obj, path, compression=...)</c>.
/// <para>
/// <b>The gzip level is not written verbatim.</b> Python defaults to gzip level 4; PureHDF's
/// deflate filter is built on <see cref="System.IO.Compression.ZLibStream"/>, whose four
/// <see cref="System.IO.Compression.CompressionLevel"/> values are the only ones it accepts —
/// <b>0, 1, 6, and 9</b> (the fourth is <c>Optimal</c>, which PureHDF passes as -1 and records
/// in the file as 6). A requested level is mapped to the nearest of those, see
/// <see cref="EffectiveGzipLevel"/>, and it is that level the file records. The filter is still
/// ordinary gzip, so h5py reads it as <c>compression='gzip'</c> and every HDF5 tool decompresses
/// it; only <c>compression_opts</c> can differ from a file Python wrote, and for the default it
/// differs by one step (6 against Python's 4).
/// </para>
/// </remarks>
public sealed record Hdf5Compression
{
    /// <summary>
    /// Key under which a <see cref="Hdf5Compression"/> is read from
    /// <see cref="BackendSaveOptions.BackendOptions"/>.
    /// </summary>
    public const string OptionKey = "compression";

    /// <summary>The filter to apply. Defaults to <see cref="Hdf5CompressionAlgorithm.Gzip"/>.</summary>
    public Hdf5CompressionAlgorithm Algorithm { get; init; } = Hdf5CompressionAlgorithm.Gzip;

    /// <summary>
    /// Compression level, or <see langword="null"/> for the algorithm's own default (4 for
    /// gzip, 5 for Blosc2 — the same defaults Python uses).
    /// </summary>
    public int? Level { get; init; }

    /// <summary>Whether the byte-shuffle filter runs before the compressor.</summary>
    public bool Shuffle { get; init; } = true;

    /// <summary>Inner Blosc2 codec. Ignored unless <see cref="Algorithm"/> is Blosc.</summary>
    public Hdf5BloscCompressor BloscCompressor { get; init; } = Hdf5BloscCompressor.Zstd;

    /// <summary>Gzip level 4 with shuffle. Python counterpart: <c>GZIP_DEFAULT</c>.</summary>
    public static Hdf5Compression Gzip { get; } = new()
    {
        Algorithm = Hdf5CompressionAlgorithm.Gzip,
        Level = 4,
    };

    /// <summary>Blosc2/zstd level 5 with shuffle. Python counterpart: <c>BLOSC_DEFAULT</c>.</summary>
    public static Hdf5Compression Blosc { get; } = new()
    {
        Algorithm = Hdf5CompressionAlgorithm.Blosc,
        Level = 5,
        BloscCompressor = Hdf5BloscCompressor.Zstd,
    };

    /// <summary>No compression at all. Python counterpart: <c>UNCOMPRESSED</c>.</summary>
    public static Hdf5Compression Uncompressed { get; } = new()
    {
        Algorithm = Hdf5CompressionAlgorithm.None,
        Level = null,
        Shuffle = false,
    };

    /// <summary>What a save with no explicit setting uses. Python counterpart: <c>DEFAULT_COMPRESSION</c>.</summary>
    public static Hdf5Compression Default => Gzip;

    /// <summary>Whether array datasets are chunked and filtered rather than contiguous.</summary>
    public bool IsEnabled => Algorithm != Hdf5CompressionAlgorithm.None;

    /// <summary>
    /// The gzip level actually written, which is the nearest one PureHDF can apply.
    /// </summary>
    /// <remarks>
    /// PureHDF writes 0, 1, 6, and 9 — the four <see cref="System.IO.Compression.CompressionLevel"/>
    /// values. Everything else rounds to the nearest of them: 1-3 to 1, 4-7 to 6, 8-9 to 9.
    /// Python's default of 4 therefore writes 6, which is zlib's own default and the closest
    /// available effort to what Python spends.
    /// </remarks>
    public int EffectiveGzipLevel => (Level ?? 4) switch
    {
        <= 0 => 0,
        <= 3 => 1,
        <= 7 => 6,
        _ => 9,
    };

    /// <summary>Builds the PureHDF filter pipeline for this setting.</summary>
    /// <returns>The filters in application order, or <see langword="null"/> when uncompressed.</returns>
    internal List<H5Filter>? BuildFilters()
    {
        switch (Algorithm)
        {
            case Hdf5CompressionAlgorithm.None:
                return null;

            case Hdf5CompressionAlgorithm.Blosc:
                // Blosc2 owns its own shuffle stage, so the standalone shuffle filter is not
                // stacked on top of it — exactly as Python passes `filters=Blosc2.SHUFFLE`
                // rather than `shuffle=True` alongside.
                return
                [
                    new H5Filter(
                        Blosc2Filter.Id,
                        new Dictionary<string, object>(StringComparer.Ordinal)
                        {
                            [Blosc2Filter.COMPRESSION_LEVEL] = Level ?? 5,
                            [Blosc2Filter.COMPRESSOR_CODE] = BloscCode(BloscCompressor),
                            [Blosc2Filter.SHUFFLE] = Shuffle
                                ? Blosc2ShuffleMode.Shuffle
                                : Blosc2ShuffleMode.NoShuffle,
                        }),
                ];

            default:
                List<H5Filter> filters = [];
                if (Shuffle)
                {
                    filters.Add(new H5Filter(ShuffleFilter.Id, new Dictionary<string, object>(StringComparer.Ordinal)));
                }

                filters.Add(new H5Filter(
                    DeflateFilter.Id,
                    new Dictionary<string, object>(StringComparer.Ordinal)
                    {
                        [DeflateFilter.COMPRESSION_LEVEL] = EffectiveGzipLevel,
                    }));

                return filters;
        }
    }

    private static string BloscCode(Hdf5BloscCompressor compressor) => compressor switch
    {
        Hdf5BloscCompressor.BloscLz => "blosclz",
        Hdf5BloscCompressor.Lz4 => "lz4",
        Hdf5BloscCompressor.Lz4Hc => "lz4hc",
        Hdf5BloscCompressor.Zlib => "zlib",
        _ => "zstd",
    };
}
