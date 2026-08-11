using System.Numerics;
using Versionable.Errors;

namespace Versionable.Numerics;

/// <summary>
/// The dtype table: grammar tokens, NPY <c>descr</c> strings, CLR element types, and numpy's
/// safe-cast relation.
/// </summary>
/// <remarks>
/// Python counterparts: <c>_DTYPE_TOKENS</c> in <c>src/versionable/_arrays.py</c> for the
/// token names and <c>numpy.can_cast(..., casting="safe")</c> for
/// <see cref="CanCastSafely"/>.
/// </remarks>
internal static class Dtypes
{
    /// <summary>Every token, in <see cref="DtypeToken"/> order.</summary>
    internal static readonly DtypeToken[] All =
    [
        DtypeToken.Bool,
        DtypeToken.Int8,
        DtypeToken.Int16,
        DtypeToken.Int32,
        DtypeToken.Int64,
        DtypeToken.UInt8,
        DtypeToken.UInt16,
        DtypeToken.UInt32,
        DtypeToken.UInt64,
        DtypeToken.Float16,
        DtypeToken.Float32,
        DtypeToken.Float64,
        DtypeToken.Complex64,
        DtypeToken.Complex128,
    ];

    // Indexed by (int)DtypeToken. GRAMMAR §7.
    private static readonly string[] _tokens =
    [
        "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64", "complex64", "complex128",
    ];

    // NPY `descr` strings. The byte-order character is '|' for one-byte types (numpy writes
    // '|' where endianness is meaningless) and '<' otherwise: everything this codec writes is
    // little-endian.
    private static readonly string[] _descrs =
    [
        "|b1", "|i1", "<i2", "<i4", "<i8", "|u1", "<u2", "<u4", "<u8",
        "<f2", "<f4", "<f8", "<c8", "<c16",
    ];

    private static readonly int[] _itemSizes = [1, 1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 8, 8, 16];

    private static readonly Type[] _clrTypes =
    [
        typeof(bool), typeof(sbyte), typeof(short), typeof(int), typeof(long),
        typeof(byte), typeof(ushort), typeof(uint), typeof(ulong),
        typeof(Half), typeof(float), typeof(double), typeof(Complex), typeof(Complex),
    ];

    // numpy's can_cast(from, to, casting="safe"), transcribed verbatim from numpy 2.4.3.
    // Rows are the source dtype, columns the destination, both in DtypeToken order. Kept as a
    // literal grid rather than derived from a rule ("wider is safe") because numpy's relation
    // is not that rule: int64 -> float64 is safe despite losing integers above 2^53,
    // int32 -> complex64 is unsafe despite complex64 being 8 bytes wide, and uint8 -> int16 is
    // safe while int8 -> uint16 is not. Regenerate with:
    //   [int(np.can_cast(np.dtype(a), np.dtype(b), casting="safe")) for b in tokens]
    private static readonly string[] _safeCasts =
    [
        // bool i8 i16 i32 i64 u8 u16 u32 u64 f16 f32 f64 c64 c128
        "11111111111111", // bool
        "01111000011111", // int8
        "00111000001111", // int16
        "00011000000101", // int32
        "00001000000101", // int64
        "00111111111111", // uint8
        "00011011101111", // uint16
        "00001001100101", // uint32
        "00000000100101", // uint64
        "00000000011111", // float16
        "00000000001111", // float32
        "00000000000101", // float64
        "00000000000011", // complex64
        "00000000000001", // complex128
    ];

    /// <summary>Returns the canonical grammar token for <paramref name="dtype"/>.</summary>
    /// <param name="dtype">The dtype.</param>
    /// <returns>A token such as <c>float64</c>.</returns>
    internal static string ToToken(DtypeToken dtype) => _tokens[(int)dtype];

    /// <summary>Parses a canonical grammar token.</summary>
    /// <param name="token">A token such as <c>float64</c>.</param>
    /// <param name="dtype">The parsed dtype.</param>
    /// <returns><see langword="true"/> when <paramref name="token"/> is in the closed set.</returns>
    internal static bool TryParseToken(string? token, out DtypeToken dtype)
    {
        int index = token is null ? -1 : Array.IndexOf(_tokens, token);
        dtype = index < 0 ? default : (DtypeToken)index;
        return index >= 0;
    }

    /// <summary>Returns the NPY <c>descr</c> string this codec writes for <paramref name="dtype"/>.</summary>
    /// <param name="dtype">The dtype.</param>
    /// <returns>A descr such as <c>&lt;f8</c>.</returns>
    internal static string ToDescr(DtypeToken dtype) => _descrs[(int)dtype];

    /// <summary>Parses an NPY <c>descr</c> string.</summary>
    /// <param name="descr">The <c>descr</c> value read from an NPY header.</param>
    /// <param name="dtype">The parsed dtype.</param>
    /// <returns>
    /// <see langword="true"/> when <paramref name="descr"/> names a supported dtype whose
    /// bytes are readable as little-endian. Big-endian multi-byte data (<c>&gt;</c>) and every
    /// dtype outside the closed set return <see langword="false"/>.
    /// </returns>
    internal static bool TryParseDescr(string? descr, out DtypeToken dtype)
    {
        dtype = default;
        if (string.IsNullOrEmpty(descr))
        {
            return false;
        }

        // numpy writes '<' (little), '>' (big), '=' (native), or '|' (not applicable). A
        // one-byte type may carry any of them and still mean the same bytes.
        char order = descr[0];
        bool hasOrder = order is '<' or '>' or '=' or '|';
        string rest = hasOrder ? descr[1..] : descr;
        int index = Array.FindIndex(_descrs, d => string.Equals(d[1..], rest, StringComparison.Ordinal));
        if (index < 0)
        {
            return false;
        }

        dtype = (DtypeToken)index;
        return order != '>' || ItemSize(dtype) == 1;
    }

    /// <summary>Returns the on-disk size in bytes of one element.</summary>
    /// <param name="dtype">The dtype.</param>
    /// <returns>The element size in bytes.</returns>
    internal static int ItemSize(DtypeToken dtype) => _itemSizes[(int)dtype];

    /// <summary>Returns the CLR type one element decodes to.</summary>
    /// <param name="dtype">The dtype.</param>
    /// <returns>
    /// The element type. <see cref="DtypeToken.Complex64"/> and
    /// <see cref="DtypeToken.Complex128"/> both map to <see cref="Complex"/>: .NET has no
    /// single-precision complex, so complex64 widens on read.
    /// </returns>
    internal static Type ToClrType(DtypeToken dtype) => _clrTypes[(int)dtype];

    /// <summary>Returns the dtype a <c>Tensor&lt;T&gt;</c> element type declares.</summary>
    /// <param name="clrType">The element type.</param>
    /// <param name="dtype">The declared dtype.</param>
    /// <returns>
    /// <see langword="false"/> for element types with no grammar token. Note
    /// <see cref="Complex"/> declares <see cref="DtypeToken.Complex128"/>, never
    /// <c>complex64</c> — GRAMMAR §7 leaves complex64 without a C# element type.
    /// </returns>
    internal static bool TryFromClrType(Type clrType, out DtypeToken dtype)
    {
        // Complex occupies two slots in _clrTypes, and IndexOf would find the complex64 one.
        // Special-cased rather than reordering the table, whose order is the grammar's.
        if (clrType == typeof(Complex))
        {
            dtype = DtypeToken.Complex128;
            return true;
        }

        int index = Array.IndexOf(_clrTypes, clrType);
        dtype = index < 0 ? default : (DtypeToken)index;
        return index >= 0;
    }

    /// <summary>
    /// Whether numpy considers casting <paramref name="from"/> to <paramref name="to"/> safe.
    /// </summary>
    /// <param name="from">The dtype found in the data.</param>
    /// <param name="to">The declared dtype.</param>
    /// <returns><see langword="true"/> when the cast is applied silently.</returns>
    internal static bool CanCastSafely(DtypeToken from, DtypeToken to) =>
        _safeCasts[(int)from][(int)to] == '1';

    /// <summary>
    /// Throws unless an array of <paramref name="actual"/> may be read as
    /// <paramref name="declared"/>.
    /// </summary>
    /// <param name="actual">The dtype the data carries.</param>
    /// <param name="declared">The dtype the field declares.</param>
    /// <param name="fieldPath">Dotted path of the field, for the error message.</param>
    /// <param name="context">Whether the check runs on save or on load.</param>
    /// <exception cref="DtypeMismatchException">The cast would lose data.</exception>
    /// <remarks>
    /// Python counterpart: <c>resolveCast</c> in <c>src/versionable/_arrays.py</c>, minus its
    /// return value. Python has to hand the target dtype back because it applies the cast with
    /// <c>astype</c> at the call site; here the target is <c>Tensor&lt;T&gt;</c>'s element type,
    /// known statically, and <see cref="CastElements{T}"/> converts to it. Returning a dtype no
    /// caller could use would only invite the assumption that this method had applied it.
    /// </remarks>
    internal static void EnsureCastable(
        DtypeToken actual,
        DtypeToken declared,
        string fieldPath = "",
        DtypeContext context = DtypeContext.Load)
    {
        if (actual == declared || CanCastSafely(actual, declared))
        {
            return;
        }

        throw new DtypeMismatchException(ToToken(declared), ToToken(actual), fieldPath, context);
    }

    /// <summary>Reinterprets a flat element array as <typeparamref name="T"/>, casting if needed.</summary>
    /// <typeparam name="T">The declared element type.</typeparam>
    /// <param name="source">A flat array whose element type is <see cref="ToClrType"/> of its dtype.</param>
    /// <returns>
    /// A flat <typeparamref name="T"/> array — <paramref name="source"/> itself when it is
    /// already one.
    /// </returns>
    /// <exception cref="ConverterException">No conversion exists for the pair.</exception>
    /// <remarks>
    /// Only reached after <see cref="EnsureCastable"/> has approved the pair, so every conversion
    /// that runs is one numpy calls safe and no <c>CreateChecked</c> call can overflow.
    /// <see cref="bool"/> needs no arm of its own: it is safely castable only from itself, and
    /// that case is the identity fast path.
    /// </remarks>
    internal static T[] CastElements<T>(Array source)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source is T[] alreadyTyped)
        {
            return alreadyTyped;
        }

        object converted = typeof(T) switch
        {
            Type t when t == typeof(sbyte) => Widen<sbyte>(source),
            Type t when t == typeof(short) => Widen<short>(source),
            Type t when t == typeof(int) => Widen<int>(source),
            Type t when t == typeof(long) => Widen<long>(source),
            Type t when t == typeof(byte) => Widen<byte>(source),
            Type t when t == typeof(ushort) => Widen<ushort>(source),
            Type t when t == typeof(uint) => Widen<uint>(source),
            Type t when t == typeof(ulong) => Widen<ulong>(source),
            Type t when t == typeof(Half) => Widen<Half>(source),
            Type t when t == typeof(float) => Widen<float>(source),
            Type t when t == typeof(double) => Widen<double>(source),
            Type t when t == typeof(Complex) => Widen<Complex>(source),
            _ => throw NoConversion(source, typeof(T)),
        };

        return (T[])converted;
    }

    /// <summary>Allocates a flat element array of the CLR type <paramref name="dtype"/> decodes to.</summary>
    /// <param name="dtype">The dtype.</param>
    /// <param name="length">Element count.</param>
    /// <returns>A zero-filled array.</returns>
    internal static Array CreateElementArray(DtypeToken dtype, int length) => dtype switch
    {
        DtypeToken.Bool => new bool[length],
        DtypeToken.Int8 => new sbyte[length],
        DtypeToken.Int16 => new short[length],
        DtypeToken.Int32 => new int[length],
        DtypeToken.Int64 => new long[length],
        DtypeToken.UInt8 => new byte[length],
        DtypeToken.UInt16 => new ushort[length],
        DtypeToken.UInt32 => new uint[length],
        DtypeToken.UInt64 => new ulong[length],
        DtypeToken.Float16 => new Half[length],
        DtypeToken.Float32 => new float[length],
        DtypeToken.Float64 => new double[length],
        DtypeToken.Complex64 or DtypeToken.Complex128 => new Complex[length],
        _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
    };

    private static TTo[] Widen<TTo>(Array source)
        where TTo : INumberBase<TTo> => source switch
        {
            bool[] a => Array.ConvertAll(a, v => v ? TTo.One : TTo.Zero),
            sbyte[] a => Array.ConvertAll(a, TTo.CreateChecked),
            short[] a => Array.ConvertAll(a, TTo.CreateChecked),
            int[] a => Array.ConvertAll(a, TTo.CreateChecked),
            long[] a => Array.ConvertAll(a, TTo.CreateChecked),
            byte[] a => Array.ConvertAll(a, TTo.CreateChecked),
            ushort[] a => Array.ConvertAll(a, TTo.CreateChecked),
            uint[] a => Array.ConvertAll(a, TTo.CreateChecked),
            ulong[] a => Array.ConvertAll(a, TTo.CreateChecked),
            Half[] a => Array.ConvertAll(a, TTo.CreateChecked),
            float[] a => Array.ConvertAll(a, TTo.CreateChecked),
            double[] a => Array.ConvertAll(a, TTo.CreateChecked),
            _ => throw NoConversion(source, typeof(TTo)),
        };

    private static ConverterException NoConversion(Array source, Type target)
    {
        string from = source.GetType().GetElementType()?.Name ?? source.GetType().Name;
        return new ConverterException(
            $"No ndarray element conversion from {from} to {target.Name}. Supported element "
            + $"types: {string.Join(", ", _tokens)}.");
    }
}
