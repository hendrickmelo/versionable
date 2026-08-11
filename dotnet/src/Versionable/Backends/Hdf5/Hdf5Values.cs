using System.Globalization;
using System.Numerics;
using PureHDF;
using Versionable.Errors;

namespace Versionable.Backends.Hdf5;

/// <summary>
/// The leaf level of the HDF5 mapping: link names, attributes, and scalar 1-D datasets.
/// </summary>
/// <remarks>
/// Python counterparts: <c>_keyToStr</c> / <c>_strToKey</c>, <c>_readAttr</c>, and
/// <c>_dtypeForElementType</c> in <c>src/versionable/_hdf5_backend.py</c>.
/// <para>
/// Integers are widened to <see cref="long"/> and floats to <see cref="double"/> on the way
/// out, because that is what a Python <c>int</c> and <c>float</c> become in an h5py attribute
/// and the golden files record. A C# <c>int</c> field written as <c>int32</c> would still load
/// in Python, but two files written from the same schema in the two languages would differ in
/// their headers for no reason.
/// </para>
/// </remarks>
internal static class Hdf5Values
{
    /// <summary>Encodes a wire key as an HDF5 link name.</summary>
    /// <param name="key">The wire key.</param>
    /// <param name="path">Field path, for error messages.</param>
    /// <returns>The link name.</returns>
    /// <exception cref="BackendException">The key has no valid HDF5 name.</exception>
    /// <remarks>
    /// Python counterpart: <c>_keyToStr</c>. Only <c>/</c> and <c>%</c> are escaped — HDF5 reads
    /// <c>/</c> as a path separator, and <c>%</c> has to go too or the escaping would not be
    /// reversible — plus a bare <c>.</c>, which names the current group.
    /// </remarks>
    internal static string EncodeName(string key, string path)
    {
        if (key.Length == 0)
        {
            throw new BackendException(
                $"Dictionary key at field {Describe(path)} is empty — HDF5 requires non-empty names.");
        }

        if (key.Contains('\0', StringComparison.Ordinal))
        {
            throw new BackendException(
                $"Dictionary key '{key}' at field {Describe(path)} contains a null byte, which HDF5 "
                    + "names cannot hold.");
        }

        string encoded = key
            .Replace("%", "%25", StringComparison.Ordinal)
            .Replace("/", "%2F", StringComparison.Ordinal);

        return encoded == "." ? "%2E" : encoded;
    }

    /// <summary>Decodes an HDF5 link name back to its wire key.</summary>
    /// <param name="name">The link name.</param>
    /// <returns>The wire key.</returns>
    /// <remarks>
    /// Python counterpart: <c>_strToKey</c>, which calls <c>urllib.parse.unquote</c> — a single
    /// pass, so <c>%2525</c> decodes to <c>%25</c> and not to <c>%</c>.
    /// </remarks>
    internal static string DecodeName(string name) => Uri.UnescapeDataString(name);

    /// <summary>Builds the PureHDF attribute for a wire value.</summary>
    /// <param name="value">The wire value; <see langword="null"/> becomes a null dataspace.</param>
    /// <param name="path">Field path, for error messages.</param>
    /// <returns>The attribute.</returns>
    /// <exception cref="BackendException">The value has no attribute form.</exception>
    internal static H5Attribute ToAttribute(object? value, string path)
    {
        if (value is null)
        {
            // Python writes `h5py.Empty("f")` — a null dataspace typed float32 — and reads any
            // null dataspace back as None. PureHDF spells the same thing as a null-dataspace
            // attribute over float; the array element type is what makes it take that overload.
            return new H5Attribute<float[]>(default(H5OpaqueInfo)!);
        }

        if (value is IReadOnlyList<object?> list)
        {
            return new H5Attribute(ScalarArray(list, elementType: null, path), null, null);
        }

        return new H5Attribute(Scalar(value, path), null, null);
    }

    /// <summary>Reads an attribute as a wire value.</summary>
    /// <param name="attribute">The attribute.</param>
    /// <param name="path">Field path, for error messages.</param>
    /// <returns>A wire scalar, a <see cref="List{T}"/> of them, or <see langword="null"/>.</returns>
    internal static object? FromAttribute(IH5Attribute attribute, string path)
    {
        if (attribute.Space.Type == H5DataspaceType.Null)
        {
            return null;
        }

        List<object?> values = ReadAttributeElements(attribute, path);

        // A scalar dataspace is not a one-element array: `complex` writes a shape-(2,) attribute
        // and `count` writes a scalar one, and unwrapping by length would make a one-element
        // list[float] load as a bare float.
        return attribute.Space.Type == H5DataspaceType.Scalar
            ? values.Count == 0 ? null : values[0]
            : values;
    }

    /// <summary>Reads a string dataset or attribute.</summary>
    /// <param name="source">A dataset or attribute whose type is a string.</param>
    /// <param name="path">Field path, for error messages.</param>
    /// <returns>The strings, in file order.</returns>
    internal static string?[] ReadStrings(object source, string path)
    {
        try
        {
            return source switch
            {
                IH5Dataset dataset => dataset.Read<string[]>(),
                IH5Attribute attribute => attribute.Read<string[]>(),
                _ => throw new BackendException($"Field {Describe(path)} is neither a dataset nor an attribute."),
            };
        }
        catch (Exception error) when (error is not VersionableException)
        {
            throw new BackendException($"Failed to read the string data at field {Describe(path)}.", error);
        }
    }

    /// <summary>Builds the 1-D dataset a sequence of scalars is written as.</summary>
    /// <param name="values">The wire elements.</param>
    /// <param name="elementType">The declared element type, which types an empty sequence.</param>
    /// <param name="path">Field path, for error messages.</param>
    /// <returns>The dataset. Always contiguous: Python compresses arrays, not containers.</returns>
    internal static object ToScalarDataset(IReadOnlyList<object?> values, Type? elementType, string path)
    {
        object data = ScalarArray(values, elementType, path);
        ulong[] dimensions = [(ulong)values.Count];

        return data switch
        {
            string[] strings => new H5Dataset<string[]>(strings, chunks: null, fileDims: dimensions),
            Hdf5Bool[] flags => new H5Dataset<Hdf5Bool[]>(flags, chunks: null, fileDims: dimensions),
            long[] integers => new H5Dataset<long[]>(integers, chunks: null, fileDims: dimensions),
            ulong[] integers => new H5Dataset<ulong[]>(integers, chunks: null, fileDims: dimensions),
            _ => new H5Dataset<double[]>((double[])data, chunks: null, fileDims: dimensions),
        };
    }

    /// <summary>Widens a CLR value read out of HDF5 to the wire form the engine expects.</summary>
    /// <param name="value">A raw element.</param>
    /// <returns>The wire value.</returns>
    internal static object? Normalize(object? value) => value switch
    {
        null => null,
        bool or string => value,
        sbyte or short or int or long => Convert.ToInt64(value, CultureInfo.InvariantCulture),
        byte or ushort or uint => Convert.ToInt64(value, CultureInfo.InvariantCulture),
        ulong big => big <= long.MaxValue ? (long)big : value,
        Half or float or double => Convert.ToDouble(value, CultureInfo.InvariantCulture),
        Complex complex => new List<object?> { complex.Real, complex.Imaginary },
        _ => value,
    };

    private static List<object?> ReadAttributeElements(IH5Attribute attribute, string path)
    {
        if (attribute.Type.Class is H5DataTypeClass.String or H5DataTypeClass.VariableLength)
        {
            return [.. ReadStrings(attribute, path)];
        }

        List<object?> values = [];
        foreach (object? element in ReadAttributeArray(attribute, path))
        {
            values.Add(Normalize(element));
        }

        return values;
    }

    private static Array ReadAttributeArray(IH5Attribute attribute, string path)
    {
        Numerics.DtypeToken token = Hdf5Arrays.TokenFor(attribute.Type, path);
        try
        {
            return token switch
            {
                Numerics.DtypeToken.Bool => attribute.Read<bool[]>(),
                Numerics.DtypeToken.Int8 => attribute.Read<sbyte[]>(),
                Numerics.DtypeToken.Int16 => attribute.Read<short[]>(),
                Numerics.DtypeToken.Int32 => attribute.Read<int[]>(),
                Numerics.DtypeToken.Int64 => attribute.Read<long[]>(),
                Numerics.DtypeToken.UInt8 => attribute.Read<byte[]>(),
                Numerics.DtypeToken.UInt16 => attribute.Read<ushort[]>(),
                Numerics.DtypeToken.UInt32 => attribute.Read<uint[]>(),
                Numerics.DtypeToken.UInt64 => attribute.Read<ulong[]>(),
                Numerics.DtypeToken.Float16 => attribute.Read<Half[]>(),
                Numerics.DtypeToken.Float32 => attribute.Read<float[]>(),
                Numerics.DtypeToken.Float64 => attribute.Read<double[]>(),
                Numerics.DtypeToken.Complex64 => Array.ConvertAll(
                    attribute.Read<Hdf5Complex64[]>(), v => new Complex(v.Real, v.Imaginary)),
                _ => Array.ConvertAll(attribute.Read<Hdf5Complex[]>(), v => new Complex(v.Real, v.Imaginary)),
            };
        }
        catch (Exception error) when (error is not VersionableException)
        {
            throw new BackendException($"Failed to read the attribute at field {Describe(path)}.", error);
        }
    }

    private static object Scalar(object value, string path) => value switch
    {
        string text => text,
        bool flag => flag ? Hdf5Bool.TRUE : Hdf5Bool.FALSE,
        sbyte or short or int or long => Convert.ToInt64(value, CultureInfo.InvariantCulture),
        byte or ushort or uint => Convert.ToInt64(value, CultureInfo.InvariantCulture),
        ulong big => big <= long.MaxValue ? (long)big : value,
        Half or float or double => Convert.ToDouble(value, CultureInfo.InvariantCulture),
        _ => throw new BackendException(
            $"Cannot store field {Describe(path)} of type {value.GetType().Name} in HDF5. Register "
                + "an IWireConverter that lowers it to a string or a number, or use a dict-based "
                + "backend (JSON, YAML, TOML)."),
    };

    /// <summary>
    /// Chooses one HDF5 element type for a whole sequence and materializes the CLR array.
    /// </summary>
    /// <remarks>
    /// The kind comes from the values when there are any and from the declared element type
    /// when there are not — Python's <c>_dtypeForElementType</c>, which exists for exactly the
    /// empty case. A mixed sequence has no single dataset type; it never reaches here, because
    /// the writer sends a non-scalar element type to a group instead.
    /// </remarks>
    private static object ScalarArray(IReadOnlyList<object?> values, Type? elementType, string path)
    {
        object? first = values.Count == 0 ? null : values[0];

        if (first is string || (values.Count == 0 && elementType == typeof(string)))
        {
            string?[] strings = new string?[values.Count];
            for (int index = 0; index < values.Count; index++)
            {
                strings[index] = values[index] as string
                    ?? throw Mixed(values[index], "string", path);
            }

            return strings;
        }

        if (first is bool || (values.Count == 0 && elementType == typeof(bool)))
        {
            Hdf5Bool[] flags = new Hdf5Bool[values.Count];
            for (int index = 0; index < values.Count; index++)
            {
                flags[index] = values[index] is bool flag
                    ? flag ? Hdf5Bool.TRUE : Hdf5Bool.FALSE
                    : throw Mixed(values[index], "bool", path);
            }

            return flags;
        }

        if (IsIntegral(first) || (values.Count == 0 && IsIntegralType(elementType)))
        {
            // int64 is what a Python int becomes, and what the golden files hold. A value above
            // long.MaxValue can only have come from a C# ulong field, and it gets a uint64
            // dataset rather than an overflow — Python reads that as an int just the same.
            if (values.Any(value => value is ulong big && big > long.MaxValue))
            {
                return values.Select(value => Convert.ToUInt64(value, CultureInfo.InvariantCulture)).ToArray();
            }

            long[] integers = new long[values.Count];
            for (int index = 0; index < values.Count; index++)
            {
                integers[index] = IsIntegral(values[index])
                    ? Convert.ToInt64(values[index], CultureInfo.InvariantCulture)
                    : throw Mixed(values[index], "int", path);
            }

            return integers;
        }

        double[] numbers = new double[values.Count];
        for (int index = 0; index < values.Count; index++)
        {
            numbers[index] = values[index] is (float or double or Half or sbyte or short or int or long
                or byte or ushort or uint or ulong)
                ? Convert.ToDouble(values[index], CultureInfo.InvariantCulture)
                : throw Mixed(values[index], "float", path);
        }

        return numbers;
    }

    private static BackendException Mixed(object? value, string expected, string path) =>
        new($"Field {Describe(path)} mixes element types: an HDF5 dataset holds one type, and this "
            + $"sequence has a {value?.GetType().Name ?? "null"} among its {expected} elements.");

    private static bool IsIntegral(object? value) =>
        value is sbyte or short or int or long or byte or ushort or uint or ulong;

    private static bool IsIntegralType(Type? type) =>
        type == typeof(sbyte) || type == typeof(short) || type == typeof(int) || type == typeof(long)
        || type == typeof(byte) || type == typeof(ushort) || type == typeof(uint) || type == typeof(ulong);

    private static string Describe(string path) => path.Length == 0 ? "<root>" : path;
}
