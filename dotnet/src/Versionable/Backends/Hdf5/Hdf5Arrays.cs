using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using PureHDF;
using PureHDF.VOL.Native;
using Versionable.Errors;
using Versionable.Numerics;

namespace Versionable.Backends.Hdf5;

/// <summary>
/// The HDF5 boolean, as h5py spells it: a one-byte enumeration with members <c>FALSE</c> and
/// <c>TRUE</c>.
/// </summary>
/// <remarks>
/// <b>The member names are the interchange contract, not a style choice.</b> numpy's
/// <c>bool_</c> has no HDF5 primitive; h5py stores it as an enumeration and recognises one
/// coming back only when its members are named exactly <c>FALSE</c> and <c>TRUE</c>. Writing a
/// plain <see cref="byte"/> instead — which is what PureHDF does for a CLR
/// <see cref="bool"/> — produces a <c>uint8</c> dataset, and a Python <c>ndarray[bool]</c>
/// field then fails its dtype check, because numpy does not consider <c>uint8 -&gt; bool</c> a
/// safe cast.
/// <para>
/// The base type is <em>unsigned</em> here where h5py's is <c>int8</c>. That is a PureHDF
/// constraint — an <c>sbyte</c>-backed enum fails to encode — and it costs nothing: h5py's bool
/// detection looks at the member names and the one-byte width, not at the sign, so both files
/// read back as <c>numpy.bool_</c>.
/// </para>
/// </remarks>
internal enum Hdf5Bool : byte
{
    /// <summary><see langword="false"/>.</summary>
    FALSE = 0,

    /// <summary><see langword="true"/>.</summary>
    TRUE = 1,
}

/// <summary>
/// The HDF5 complex number: a compound of two floats named <c>r</c> and <c>i</c>.
/// </summary>
/// <remarks>
/// The layout numpy and h5py use for <c>complex128</c>. The member names are matched by name
/// when PureHDF maps the compound onto this struct, so they carry
/// <see cref="H5NameAttribute"/> rather than being spelled lowercase.
/// </remarks>
[StructLayout(LayoutKind.Sequential)]
internal struct Hdf5Complex
{
    /// <summary>Real part.</summary>
    [H5Name("r")]
    public double Real;

    /// <summary>Imaginary part.</summary>
    [H5Name("i")]
    public double Imaginary;
}

/// <summary>
/// Single-precision complex, which HDF5 files may hold and .NET has no element type for.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct Hdf5Complex64
{
    /// <summary>Real part.</summary>
    [H5Name("r")]
    public float Real;

    /// <summary>Imaginary part.</summary>
    [H5Name("i")]
    public float Imaginary;
}

/// <summary>
/// Maps <c>Tensor&lt;T&gt;</c> onto real HDF5 datasets, in both directions.
/// </summary>
/// <remarks>
/// Python counterpart: the <c>np.ndarray</c> arms of <c>_writeValue</c> and <c>_readDataset</c>
/// in <c>src/versionable/_hdf5_backend.py</c>. Nothing here touches the NPY/NPZ codec: the
/// point of HDF5 is that the array <em>is</em> the dataset, with its dtype in the header where
/// every HDF5 tool can see it.
/// <para>
/// The dtype switches are written out rather than reflected over. There are thirteen element
/// types and each closes one generic instantiation the trimmer can see, which is what
/// <c>IsAotCompatible</c> costs and buys.
/// </para>
/// </remarks>
internal static class Hdf5Arrays
{
    // HDF5 caps a chunk at 4 GiB and the whole chunk is decompressed to read one element, so a
    // single-chunk multi-gigabyte array is both illegal and useless. h5py's auto-chunker targets
    // a comparable band; the golden arrays are far below it, so chunks still equal the shape
    // there and the files match Python's layout exactly.
    //
    // ChunkDimensions halves the *largest* remaining axis until the bound holds, which always
    // terminates at one element per axis and therefore always meets the bound for the element
    // types in the dtype table (16 bytes at worst). Halving a fixed axis instead would leave a
    // shape like (1, 400_000_000) above the bound with nothing left to shrink.
    private const long _maxChunkBytes = 8L * 1024 * 1024;

    /// <summary>Whether <paramref name="value"/> is a <c>Tensor&lt;T&gt;</c>.</summary>
    /// <param name="value">A wire value.</param>
    /// <returns><see langword="true"/> when it is.</returns>
    internal static bool IsTensor(object? value) =>
        value is not null && Hdf5TypeShape.TryTensorElement(value.GetType(), out _);

    /// <summary>Every closed <c>Tensor&lt;T&gt;</c> the backend stores natively.</summary>
    /// <returns>The thirteen element types of GRAMMAR §7, closed over <c>Tensor&lt;T&gt;</c>.</returns>
    internal static IEnumerable<Type> TensorTypes()
    {
        yield return typeof(Tensor<bool>);
        yield return typeof(Tensor<sbyte>);
        yield return typeof(Tensor<short>);
        yield return typeof(Tensor<int>);
        yield return typeof(Tensor<long>);
        yield return typeof(Tensor<byte>);
        yield return typeof(Tensor<ushort>);
        yield return typeof(Tensor<uint>);
        yield return typeof(Tensor<ulong>);
        yield return typeof(Tensor<Half>);
        yield return typeof(Tensor<float>);
        yield return typeof(Tensor<double>);
        yield return typeof(Tensor<Complex>);
    }

    /// <summary>Builds the PureHDF dataset for a tensor.</summary>
    /// <param name="tensor">A <c>Tensor&lt;T&gt;</c>.</param>
    /// <param name="compression">Filter settings.</param>
    /// <param name="path">Field path, for error messages.</param>
    /// <returns>The object to add to the parent group.</returns>
    /// <exception cref="BackendException">The element type has no HDF5 mapping.</exception>
    internal static object ToDataset(object tensor, Hdf5Compression compression, string path) => tensor switch
    {
        Tensor<bool> t => Build(t, compression, static v => v ? Hdf5Bool.TRUE : Hdf5Bool.FALSE),
        Tensor<sbyte> t => Build(t, compression),
        Tensor<short> t => Build(t, compression),
        Tensor<int> t => Build(t, compression),
        Tensor<long> t => Build(t, compression),
        Tensor<byte> t => Build(t, compression),
        Tensor<ushort> t => Build(t, compression),
        Tensor<uint> t => Build(t, compression),
        Tensor<ulong> t => Build(t, compression),
        Tensor<Half> t => Build(t, compression),
        Tensor<float> t => Build(t, compression),
        Tensor<double> t => Build(t, compression),
        Tensor<Complex> t => Build(
            t, compression, static v => new Hdf5Complex { Real = v.Real, Imaginary = v.Imaginary }),
        _ => throw new BackendException(
            $"Cannot write '{tensor.GetType()}' at field {Describe(path)} to HDF5. Supported "
                + $"element types: {string.Join(", ", Dtypes.All.Select(Dtypes.ToToken))} (GRAMMAR §7)."),
    };

    /// <summary>Reads a dataset as a <c>Tensor&lt;T&gt;</c> of the declared element type.</summary>
    /// <param name="dataset">The dataset.</param>
    /// <param name="elementType">The element type the field declares.</param>
    /// <param name="path">Field path, for error messages.</param>
    /// <returns>A boxed <c>Tensor&lt;T&gt;</c>.</returns>
    /// <exception cref="DtypeMismatchException">The stored dtype cannot be cast to the declared one.</exception>
    /// <exception cref="BackendException">The dataset's type has no dtype token.</exception>
    internal static object ReadTensor(IH5Dataset dataset, Type elementType, string path)
    {
        DtypeToken stored = TokenFor(dataset.Type, path);

        if (!Dtypes.TryFromClrType(elementType, out DtypeToken declared))
        {
            throw new BackendException(
                $"Field {Describe(path)} declares Tensor<{elementType.Name}>, which has no canonical "
                    + $"dtype token. Supported element types: "
                    + $"{string.Join(", ", Dtypes.All.Select(Dtypes.ToToken))} (GRAMMAR §7).");
        }

        // The dtype lives in the dataset header, so an array whose element type drifted from the
        // schema fails here rather than at the first element access — the same guarantee Python's
        // `_lazyCastDtype` gives.
        Dtypes.EnsureCastable(stored, declared, path, DtypeContext.Load);

        ulong[] dimensions = dataset.Space.Dimensions;
        if (dimensions.Length == 0)
        {
            throw new BackendException(
                $"Field {Describe(path)} holds a 0-d (scalar) HDF5 dataset, which Tensor<T> cannot "
                    + "represent — its rank is always at least 1. Store the value as a scalar field, "
                    + "or as a shape-(1,) array.");
        }

        Array flat = ReadFlat(dataset, stored, path);
        nint[] lengths = new nint[dimensions.Length];
        for (int axis = 0; axis < lengths.Length; axis++)
        {
            lengths[axis] = checked((nint)dimensions[axis]);
        }

        return Create(elementType, flat, lengths, path);
    }

    /// <summary>Reads a dataset as plain wire values: a list, nested one level per extra rank.</summary>
    /// <param name="dataset">The dataset.</param>
    /// <param name="path">Field path, for error messages.</param>
    /// <returns>A <see cref="List{T}"/> of wire values.</returns>
    internal static object ReadValues(IH5Dataset dataset, string path)
    {
        List<object?> flat = ReadWireElements(dataset, path);
        ulong[] dimensions = dataset.Space.Dimensions;
        return dimensions.Length <= 1 ? flat : Nest(flat, dimensions, 0, 0, out _);
    }

    /// <summary>The dtype token an HDF5 datatype carries.</summary>
    /// <param name="type">The dataset or attribute type.</param>
    /// <param name="path">Field path, for error messages.</param>
    /// <returns>The token.</returns>
    /// <exception cref="BackendException">The type is outside the closed dtype set.</exception>
    internal static DtypeToken TokenFor(IH5DataType type, string path)
    {
        switch (type.Class)
        {
            case H5DataTypeClass.FloatingPoint:
                return type.Size switch
                {
                    2 => DtypeToken.Float16,
                    4 => DtypeToken.Float32,
                    8 => DtypeToken.Float64,
                    _ => throw Unsupported(type, path),
                };

            case H5DataTypeClass.FixedPoint:
                return FixedPointToken(type.FixedPoint.IsSigned, type.Size, type, path);

            case H5DataTypeClass.Enumerated:
                // h5py writes numpy's bool_ as a one-byte enumeration. Anything wider is an
                // ordinary enumeration and reads as its base integer.
                IH5DataType baseType = type.Enumeration.BaseType;
                return baseType.Size == 1
                    ? DtypeToken.Bool
                    : FixedPointToken(baseType.FixedPoint.IsSigned, baseType.Size, type, path);

            case H5DataTypeClass.Compound:
                return type.Size switch
                {
                    8 => DtypeToken.Complex64,
                    16 => DtypeToken.Complex128,
                    _ => throw Unsupported(type, path),
                };

            default:
                throw Unsupported(type, path);
        }
    }

    private static DtypeToken FixedPointToken(bool signed, int size, IH5DataType type, string path) =>
        (signed, size) switch
        {
            (true, 1) => DtypeToken.Int8,
            (true, 2) => DtypeToken.Int16,
            (true, 4) => DtypeToken.Int32,
            (true, 8) => DtypeToken.Int64,
            (false, 1) => DtypeToken.UInt8,
            (false, 2) => DtypeToken.UInt16,
            (false, 4) => DtypeToken.UInt32,
            (false, 8) => DtypeToken.UInt64,
            _ => throw Unsupported(type, path),
        };

    private static BackendException Unsupported(IH5DataType type, string path) =>
        new($"Field {Describe(path)} has HDF5 type {type.Class} of {type.Size} bytes, which is "
            + $"outside the dtype set the wire format admits: "
            + $"{string.Join(", ", Dtypes.All.Select(Dtypes.ToToken))} (GRAMMAR §7).");

    private static Array ReadFlat(IH5Dataset dataset, DtypeToken stored, string path)
    {
        try
        {
            return stored switch
            {
                DtypeToken.Bool => dataset.Read<bool[]>(),
                DtypeToken.Int8 => dataset.Read<sbyte[]>(),
                DtypeToken.Int16 => dataset.Read<short[]>(),
                DtypeToken.Int32 => dataset.Read<int[]>(),
                DtypeToken.Int64 => dataset.Read<long[]>(),
                DtypeToken.UInt8 => dataset.Read<byte[]>(),
                DtypeToken.UInt16 => dataset.Read<ushort[]>(),
                DtypeToken.UInt32 => dataset.Read<uint[]>(),
                DtypeToken.UInt64 => dataset.Read<ulong[]>(),
                DtypeToken.Float16 => dataset.Read<Half[]>(),
                DtypeToken.Float32 => dataset.Read<float[]>(),
                DtypeToken.Float64 => dataset.Read<double[]>(),
                DtypeToken.Complex64 => Array.ConvertAll(
                    dataset.Read<Hdf5Complex64[]>(), v => new Complex(v.Real, v.Imaginary)),
                _ => Array.ConvertAll(dataset.Read<Hdf5Complex[]>(), v => new Complex(v.Real, v.Imaginary)),
            };
        }
        catch (Exception error) when (error is not VersionableException)
        {
            throw new BackendException(
                $"Failed to read the {Dtypes.ToToken(stored)} dataset at field {Describe(path)}: "
                    + $"{error.Message}", error);
        }
    }

    private static object Create(Type elementType, Array flat, nint[] lengths, string path)
    {
        if (elementType == typeof(bool))
        {
            return Tensor.Create(Dtypes.CastElements<bool>(flat), lengths);
        }

        if (elementType == typeof(sbyte))
        {
            return Tensor.Create(Dtypes.CastElements<sbyte>(flat), lengths);
        }

        if (elementType == typeof(short))
        {
            return Tensor.Create(Dtypes.CastElements<short>(flat), lengths);
        }

        if (elementType == typeof(int))
        {
            return Tensor.Create(Dtypes.CastElements<int>(flat), lengths);
        }

        if (elementType == typeof(long))
        {
            return Tensor.Create(Dtypes.CastElements<long>(flat), lengths);
        }

        if (elementType == typeof(byte))
        {
            return Tensor.Create(Dtypes.CastElements<byte>(flat), lengths);
        }

        if (elementType == typeof(ushort))
        {
            return Tensor.Create(Dtypes.CastElements<ushort>(flat), lengths);
        }

        if (elementType == typeof(uint))
        {
            return Tensor.Create(Dtypes.CastElements<uint>(flat), lengths);
        }

        if (elementType == typeof(ulong))
        {
            return Tensor.Create(Dtypes.CastElements<ulong>(flat), lengths);
        }

        if (elementType == typeof(Half))
        {
            return Tensor.Create(Dtypes.CastElements<Half>(flat), lengths);
        }

        if (elementType == typeof(float))
        {
            return Tensor.Create(Dtypes.CastElements<float>(flat), lengths);
        }

        if (elementType == typeof(double))
        {
            return Tensor.Create(Dtypes.CastElements<double>(flat), lengths);
        }

        if (elementType == typeof(Complex))
        {
            return Tensor.Create(Dtypes.CastElements<Complex>(flat), lengths);
        }

        throw new BackendException(
            $"Field {Describe(path)} declares Tensor<{elementType.Name}>, which HDF5 cannot "
                + "materialize. Supported element types: "
                + $"{string.Join(", ", Dtypes.All.Select(Dtypes.ToToken))} (GRAMMAR §7).");
    }

    private static List<object?> ReadWireElements(IH5Dataset dataset, string path)
    {
        if (dataset.Type.Class is H5DataTypeClass.String or H5DataTypeClass.VariableLength)
        {
            return [.. Hdf5Values.ReadStrings(dataset, path)];
        }

        DtypeToken stored = TokenFor(dataset.Type, path);
        Array flat = ReadFlat(dataset, stored, path);
        List<object?> values = new(flat.Length);
        foreach (object? element in flat)
        {
            values.Add(Hdf5Values.Normalize(element));
        }

        return values;
    }

    private static object Nest(List<object?> flat, ulong[] dimensions, int axis, int offset, out int consumed)
    {
        int length = checked((int)dimensions[axis]);
        if (axis == dimensions.Length - 1)
        {
            consumed = length;
            return flat.GetRange(offset, length);
        }

        List<object?> rows = new(length);
        consumed = 0;
        for (int index = 0; index < length; index++)
        {
            rows.Add(Nest(flat, dimensions, axis + 1, offset + consumed, out int inner));
            consumed += inner;
        }

        return rows;
    }

    private static object Build<TSource, TStored>(
        Tensor<TSource> tensor,
        Hdf5Compression compression,
        Converter<TSource, TStored> convert)
        where TStored : unmanaged
    {
        TSource[] source = Flatten(tensor);
        TStored[] stored = Array.ConvertAll(source, convert);
        return Dataset(stored, Dimensions(tensor), compression, Unsafe.SizeOf<TStored>());
    }

    private static object Build<T>(Tensor<T> tensor, Hdf5Compression compression)
        where T : unmanaged =>
        Dataset(Flatten(tensor), Dimensions(tensor), compression, Unsafe.SizeOf<T>());

    private static object Dataset<T>(T[] data, ulong[] dimensions, Hdf5Compression compression, int itemSize)
    {
        uint[]? chunks = ChunkDimensions(dimensions, itemSize, compression);
        H5DatasetCreation creation = new(Filters: chunks is null ? null : compression.BuildFilters());
        return new H5Dataset<T[]>(data, chunks: chunks, fileDims: dimensions, datasetCreation: creation);
    }

    private static T[] Flatten<T>(Tensor<T> tensor)
    {
        if (tensor.FlattenedLength > int.MaxValue)
        {
            throw new BackendException(
                $"An ndarray of {tensor.FlattenedLength} elements cannot be written: the dataset is "
                    + "built through a single CLR array, which tops out at int.MaxValue elements.");
        }

        T[] flat = new T[(int)tensor.FlattenedLength];
        tensor.FlattenTo(flat);
        return flat;
    }

    private static ulong[] Dimensions<T>(Tensor<T> tensor)
    {
        ulong[] dimensions = new ulong[tensor.Rank];
        for (int axis = 0; axis < dimensions.Length; axis++)
        {
            dimensions[axis] = (ulong)tensor.Lengths[axis];
        }

        return dimensions;
    }

    /// <summary>
    /// Chunk shape for a dataset, or <see langword="null"/> to write it contiguous.
    /// </summary>
    /// <remarks>
    /// One chunk per dataset, as h5py's auto-chunker produces for the golden arrays, except that
    /// the largest axis is halved until the chunk fits <see cref="_maxChunkBytes"/>. A degenerate
    /// axis makes chunking impossible — a chunk may not be larger than the dataset and may not be
    /// zero — so an empty array is written contiguous and uncompressed, which costs nothing to
    /// compress anyway.
    /// </remarks>
    private static uint[]? ChunkDimensions(ulong[] dimensions, int itemSize, Hdf5Compression compression)
    {
        if (!compression.IsEnabled || dimensions.Length == 0)
        {
            return null;
        }

        uint[] chunks = new uint[dimensions.Length];
        for (int axis = 0; axis < chunks.Length; axis++)
        {
            // A zero-length axis means no elements to store, so the dataset is written
            // contiguous and unfiltered: HDF5 rejects a chunk dimension of 0, and chunking an
            // empty dataset would buy compression on nothing while costing a chunk index.
            if (dimensions[axis] == 0)
            {
                return null;
            }

            chunks[axis] = dimensions[axis] > uint.MaxValue ? uint.MaxValue : (uint)dimensions[axis];
        }

        while (ChunkBytes(chunks, itemSize) > _maxChunkBytes)
        {
            int widest = 0;
            for (int axis = 1; axis < chunks.Length; axis++)
            {
                if (chunks[axis] > chunks[widest])
                {
                    widest = axis;
                }
            }

            if (chunks[widest] <= 1)
            {
                // Every axis is down to one element and the chunk is still over the bound, which
                // takes an element wider than any dtype in the table. Nothing left to shrink.
                break;
            }

            chunks[widest] = (chunks[widest] + 1) / 2;
        }

        return chunks;
    }

    private static long ChunkBytes(uint[] chunks, int itemSize)
    {
        long total = itemSize;
        foreach (uint length in chunks)
        {
            total = total > long.MaxValue / length ? long.MaxValue : total * length;
        }

        return total;
    }

    private static string Describe(string path) => path.Length == 0 ? "<root>" : path;
}
