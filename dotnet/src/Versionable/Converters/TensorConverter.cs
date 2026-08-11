using System.Collections;
using System.Globalization;
using System.Numerics.Tensors;
using Versionable.Errors;
using Versionable.Numerics;

namespace Versionable.Converters;

/// <summary>
/// <c>Tensor&lt;T&gt;</c> ⇄ the ndarray wire dict, whose payload is a base64 NPZ.
/// </summary>
/// <typeparam name="T">The element type; one of the thirteen with a dtype token (GRAMMAR §7).</typeparam>
/// <remarks>
/// Python counterparts: <c>_serializeNdarray</c> and <c>_deserializeNdarray</c> in
/// <c>src/versionable/_types.py</c>. The wire value is a dict with four keys —
/// <c>__ver_ndarray__</c>, <c>dtype</c>, <c>shape</c>, <c>data</c> — of which only
/// <c>data</c> is read back: the NPZ inside carries its own dtype and shape, and letting the
/// sidecar fields override them would make a hand-edited file decode as something its bytes
/// do not say. They are written because a human reading the JSON needs them, and because the
/// golden manifests describe them.
/// <para>
/// One converter instance per element type, all registered exactly. That is thirteen closed
/// generic instantiations rather than one reflective converter, which is what keeps the
/// library AOT-safe.
/// </para>
/// <para>
/// <strong>Dtype validation is load-only.</strong> A <c>Tensor&lt;double&gt;</c> field cannot
/// hold anything but <c>float64</c>, so the save-side check Python needs (its annotation and
/// its runtime dtype can disagree) has nothing to test here. On load the file's dtype may
/// differ, and then numpy's rule applies unchanged: a safe cast is silent, anything else is a
/// <see cref="DtypeMismatchException"/>.
/// </para>
/// </remarks>
internal sealed class TensorConverter<T> : WireConverter<Tensor<T>>
{
    /// <summary>Key marking a dict as an ndarray payload.</summary>
    internal const string MarkerKey = "__ver_ndarray__";

    /// <summary>Pre-0.2 marker key, still read.</summary>
    internal const string LegacyMarkerKey = "__ndarray__";

    /// <summary>Key holding the base64 NPZ.</summary>
    internal const string DataKey = "data";

    /// <summary>Key holding the dtype token, written for readers, not read back.</summary>
    internal const string DtypeKey = "dtype";

    /// <summary>Key holding the shape, written for readers, not read back.</summary>
    internal const string ShapeKey = "shape";

    private readonly DtypeToken _dtype;

    /// <summary>Initializes a new instance of the <see cref="TensorConverter{T}"/> class.</summary>
    /// <exception cref="UnsupportedTypeException"><typeparamref name="T"/> has no dtype token.</exception>
    internal TensorConverter()
    {
        if (!Dtypes.TryFromClrType(typeof(T), out _dtype))
        {
            throw new UnsupportedTypeException(
                $"Tensor<{typeof(T).Name}> has no canonical dtype token, so it cannot appear in a "
                + $"schema hash. Supported element types: "
                + $"{string.Join(", ", Dtypes.All.Select(Dtypes.ToToken))} (GRAMMAR §7).");
        }
    }

    /// <summary>
    /// The canonical array type string, <c>ndarray[float64]</c> — parameterized, unlike every
    /// other converter's bare name, because array dtype is hash-significant (ADR-0002).
    /// </summary>
    public override string SerializationName => $"ndarray[{Dtypes.ToToken(_dtype)}]";

    /// <summary>The dtype this converter's element type declares.</summary>
    internal DtypeToken Dtype => _dtype;

    /// <inheritdoc/>
    protected override object ToWireCore(Tensor<T> value)
    {
        long[] shape = new long[value.Rank];
        for (int i = 0; i < shape.Length; i++)
        {
            shape[i] = value.Lengths[i];
        }

        // FlattenTo walks the tensor in C order and materialises a dense copy, so a strided
        // view (a slice of a larger tensor) writes the elements it logically holds rather than
        // the buffer it points into.
        if (value.FlattenedLength > int.MaxValue)
        {
            throw new ConverterException(
                $"An ndarray of {value.FlattenedLength} elements cannot be written: the NPZ payload "
                + "is built through a single CLR array, which tops out at int.MaxValue elements.");
        }

        T[] flat = new T[(int)value.FlattenedLength];
        value.FlattenTo(flat);

        return new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            [MarkerKey] = true,
            [DtypeKey] = Dtypes.ToToken(_dtype),
            [ShapeKey] = shape.Select(d => (object?)d).ToList(),
            [DataKey] = Convert.ToBase64String(NpzCodec.Encode(_dtype, shape, flat)),
        };
    }

    /// <inheritdoc/>
    protected override object FromWireCore(object wireValue, Type targetType)
    {
        // Backends that hand arrays back natively (HDF5) skip the NPZ entirely.
        if (wireValue is Tensor<T> tensor)
        {
            return tensor;
        }

        if (wireValue is T[] flatArray)
        {
            return Tensor.Create(flatArray, [(nint)flatArray.Length]);
        }

        if (wireValue is not IDictionary dict)
        {
            throw new ConverterException(
                $"An ndarray is written as a dict with a '{DataKey}' key holding a base64 NPZ; the "
                + $"file holds {wireValue.GetType().Name}.");
        }

        // Python's _deserializeNdarray requires a truthy '__ver_ndarray__' or '__ndarray__'
        // before it will look at 'data', and raises otherwise. Without the same gate a plain
        // dict field whose value happened to carry a 'data' string would be decoded as an
        // array instead of failing.
        if (!IsTruthy(dict[MarkerKey]) && !IsTruthy(dict[LegacyMarkerKey]))
        {
            throw new ConverterException(
                $"Dict is not an ndarray payload: neither '{MarkerKey}' nor '{LegacyMarkerKey}' is "
                + $"present and truthy. Keys present: {DescribeKeys(dict)}.");
        }

        if (dict[DataKey] is not string base64)
        {
            throw new ConverterException(
                $"ndarray dict has no string '{DataKey}' key. Keys present: {DescribeKeys(dict)}.");
        }

        byte[] npz;
        try
        {
            npz = Convert.FromBase64String(base64);
        }
        catch (FormatException e)
        {
            throw new ConverterException($"ndarray '{DataKey}' is not valid base64.", e);
        }

        NpyArray decoded = NpzCodec.Decode(npz);

        // Validation only — it throws DtypeMismatchException when the cast would lose data and
        // returns nothing. The conversion itself happens in CastElements, which knows the
        // target type statically.
        Dtypes.EnsureCastable(decoded.Dtype, _dtype, string.Empty, DtypeContext.Load);

        if (decoded.Shape.Length == 0)
        {
            throw new ConverterException(
                "The file holds a 0-d (scalar) ndarray, which Tensor<T> cannot represent — its "
                + "rank is always at least 1. Store the value as a scalar field, or as a shape-(1,) "
                + "array.");
        }

        T[] elements = Dtypes.CastElements<T>(decoded.Values);
        nint[] lengths = new nint[decoded.Shape.Length];
        for (int i = 0; i < lengths.Length; i++)
        {
            lengths[i] = checked((nint)decoded.Shape[i]);
        }

        return Tensor.Create(elements, lengths);
    }

    /// <summary>Python's notion of truthiness, over the CLR types a backend can hand back.</summary>
    /// <remarks>
    /// Python tests the marker with <c>or</c>, so <c>False</c>, <c>0</c>, and <c>""</c> all read
    /// as absent. Matching that means a file Python rejects is not silently accepted here.
    /// </remarks>
    private static bool IsTruthy(object? value) => value switch
    {
        null => false,
        bool flag => flag,
        string text => text.Length > 0,
        IConvertible number => number.ToDouble(CultureInfo.InvariantCulture) != 0,
        _ => true,
    };

    private static string DescribeKeys(IDictionary dict) =>
        string.Join(", ", dict.Keys.Cast<object>().Select(k => k?.ToString()));
}
