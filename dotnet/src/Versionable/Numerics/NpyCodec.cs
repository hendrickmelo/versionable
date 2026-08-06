using System.Buffers.Binary;
using System.Globalization;
using System.Numerics;
using System.Text;
using Versionable.Errors;

namespace Versionable.Numerics;

/// <summary>
/// Reads and writes the numpy <c>.npy</c> container for the closed dtype set (GRAMMAR §7).
/// </summary>
/// <remarks>
/// Reference: numpy's <c>numpy/lib/format.py</c>, which is the format's normative
/// description. There is no mainstream .NET NPY package, so the codec is in-library.
/// <para>
/// <strong>Reads</strong> format 1.0, 2.0 and 3.0 headers (they differ only in the width and
/// encoding of the header-length field). <strong>Writes</strong> 1.0 only — 2.0 exists for
/// headers past 64 KiB, which needs roughly five thousand dimensions.
/// </para>
/// <para>
/// C-order only. A Fortran-ordered file is rejected rather than transposed: transposing would
/// silently change which element a given index reaches, and numpy only writes
/// <c>fortran_order: True</c> for an array that was explicitly built that way.
/// </para>
/// <para>
/// Little-endian only, which is what <c>numpy.save</c> emits on every platform this ships to.
/// Big-endian multi-byte data is rejected by <see cref="Dtypes.TryParseDescr"/>.
/// </para>
/// </remarks>
internal static class NpyCodec
{
    /// <summary>The six-byte magic prefix, <c>\x93NUMPY</c>.</summary>
    internal static ReadOnlySpan<byte> Magic =>
        [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];

    // numpy's ARRAY_ALIGN: the header is padded so the data starts on a 64-byte boundary.
    private const int _arrayAlign = 64;

    // numpy's MAGIC_LEN: the magic prefix plus the two version bytes.
    private const int _magicLen = 8;

    // numpy's GROWTH_AXIS_MAX_DIGITS: spare spaces left after the header dict so the leading
    // axis can be grown in place up to 21 digits without relocating the data. It is part of
    // the byte layout, not an optimisation — omitting it changes the header length numpy
    // computes for the same array (a shape of twenty 1s lands on 182 with it, 118 without).
    private const int _growthAxisMaxDigits = 21;

    /// <summary>Decodes an NPY payload.</summary>
    /// <param name="bytes">The complete <c>.npy</c> file.</param>
    /// <returns>The dtype, shape, and flat element buffer.</returns>
    /// <exception cref="ConverterException">The payload is malformed or uses an unsupported feature.</exception>
    internal static NpyArray Decode(byte[] bytes)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        if (bytes.Length < _magicLen || !bytes.AsSpan(0, Magic.Length).SequenceEqual(Magic))
        {
            throw new ConverterException(
                "Not an NPY payload: the six-byte \\x93NUMPY magic prefix is missing.");
        }

        int major = bytes[6];
        int minor = bytes[7];

        // The length field's width is the only difference between the versions, so the
        // minimum size to read it differs too — and it has to be checked before the read
        // rather than after, or an 11-byte 2.0 stub faults instead of reporting a truncation.
        (int lengthWidth, int headerStart) = major switch
        {
            1 => (2, 10),
            2 or 3 => (4, 12),
            _ => throw new ConverterException(
                $"Unsupported NPY format version {major}.{minor}. Supported: 1.x, 2.x, 3.x."),
        };

        if (bytes.Length < _magicLen + lengthWidth)
        {
            throw new ConverterException(
                $"Truncated NPY payload: a {major}.{minor} header needs a {lengthWidth}-byte length "
                + $"field, and the payload is only {bytes.Length} bytes.");
        }

        long headerLength = lengthWidth == 2
            ? BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(8, 2))
            : BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(8, 4));

        if (headerStart + headerLength > bytes.Length)
        {
            throw new ConverterException(
                $"Truncated NPY payload: the header claims {headerLength} bytes but only "
                + $"{bytes.Length - headerStart} remain.");
        }

        // 1.0 and 2.0 headers are latin-1, 3.0 is UTF-8. Decoding the older two as UTF-8 would
        // turn a high byte into U+FFFD instead of the character it names; every byte value is
        // valid latin-1, so the two decoders only agree on ASCII. The keys this codec reads are
        // ASCII in every version, but a dtype or a stray byte need not be.
        Encoding encoding = major == 3 ? Encoding.UTF8 : Encoding.Latin1;
        string header = encoding.GetString(bytes, headerStart, (int)headerLength);
        NpyHeader parsed = NpyHeader.Parse(header);

        int dataStart = headerStart + (int)headerLength;
        long elementCount;
        try
        {
            elementCount = CountElements(parsed.Shape);
        }
        catch (OverflowException e)
        {
            throw new ConverterException(
                $"NPY shape ({string.Join(", ", parsed.Shape)}) describes more elements than a "
                + "64-bit count can hold.",
                e);
        }

        if (elementCount > int.MaxValue)
        {
            throw new ConverterException(
                $"NPY shape ({string.Join(", ", parsed.Shape)}) describes {elementCount} elements; "
                + "the decoder materialises a single CLR array, which tops out at int.MaxValue.");
        }

        long expectedBytes = elementCount * Dtypes.ItemSize(parsed.Dtype);
        long availableBytes = bytes.Length - dataStart;
        if (availableBytes < expectedBytes)
        {
            throw new ConverterException(
                $"Truncated NPY payload: shape ({string.Join(", ", parsed.Shape)}) of "
                + $"{Dtypes.ToToken(parsed.Dtype)} needs {expectedBytes} data bytes, "
                + $"{availableBytes} present.");
        }

        Array values = ReadElements(bytes.AsSpan(dataStart), parsed.Dtype, (int)elementCount);
        return new NpyArray(parsed.Dtype, parsed.Shape, values);
    }

    /// <summary>Encodes a flat element buffer as an NPY 1.0 payload.</summary>
    /// <param name="dtype">The element dtype.</param>
    /// <param name="shape">The C-order shape; empty writes a 0-d array.</param>
    /// <param name="values">The flat element buffer, in C order.</param>
    /// <returns>The complete <c>.npy</c> file, byte-identical to <c>numpy.save</c>'s output.</returns>
    /// <exception cref="ConverterException">The header would exceed the 1.0 format's 64 KiB limit.</exception>
    internal static byte[] Encode(DtypeToken dtype, long[] shape, Array values)
    {
        ArgumentNullException.ThrowIfNull(shape);
        ArgumentNullException.ThrowIfNull(values);

        long elementCount = CountElements(shape);
        if (elementCount != values.Length)
        {
            throw new ConverterException(
                $"Shape ({string.Join(", ", shape)}) describes {elementCount} elements but the "
                + $"buffer holds {values.Length}.");
        }

        byte[] header = BuildHeader(dtype, shape);
        byte[] result = new byte[header.Length + (elementCount * Dtypes.ItemSize(dtype))];
        header.CopyTo(result, 0);
        WriteElements(result.AsSpan(header.Length), dtype, values);
        return result;
    }

    /// <summary>Multiplies out a shape. An empty shape is 0-d and holds exactly one element.</summary>
    /// <param name="shape">The C-order shape.</param>
    /// <returns>The element count.</returns>
    internal static long CountElements(long[] shape)
    {
        long count = 1;
        foreach (long dimension in shape)
        {
            count = checked(count * dimension);
        }

        return count;
    }

    /// <summary>Builds the magic prefix, version, length field, and padded header dict.</summary>
    /// <param name="dtype">The element dtype.</param>
    /// <param name="shape">The C-order shape.</param>
    /// <returns>The header bytes, whose length is a multiple of 64.</returns>
    internal static byte[] BuildHeader(DtypeToken dtype, long[] shape)
    {
        string dict =
            $"{{'descr': '{Dtypes.ToDescr(dtype)}', 'fortran_order': False, 'shape': {ShapeRepr(shape)}, }}";

        // numpy._write_array_header, verbatim: spare space sized from the leading axis, then
        // pad so magic + length field + header + '\n' lands on a 64-byte boundary.
        int growth = shape.Length > 0
            ? Math.Max(0, _growthAxisMaxDigits - shape[0].ToString(CultureInfo.InvariantCulture).Length)
            : 0;
        int contentLength = dict.Length + growth + 1;
        int padding = _arrayAlign - ((_magicLen + 2 + contentLength) % _arrayAlign);
        int declaredLength = contentLength + padding;
        if (declaredLength > ushort.MaxValue)
        {
            throw new ConverterException(
                $"NPY header is {declaredLength} bytes, past the {ushort.MaxValue}-byte limit of "
                + "format 1.0. Reduce the number of dimensions.");
        }

        byte[] result = new byte[_magicLen + 2 + declaredLength];
        Magic.CopyTo(result);
        result[6] = 1;
        result[7] = 0;
        BinaryPrimitives.WriteUInt16LittleEndian(result.AsSpan(8, 2), (ushort)declaredLength);

        int written = _magicLen + 2;
        written += Encoding.ASCII.GetBytes(dict, 0, dict.Length, result, written);
        result.AsSpan(written, growth + padding).Fill((byte)' ');
        result[^1] = (byte)'\n';
        return result;
    }

    private static string ShapeRepr(long[] shape) => shape.Length switch
    {
        // Python tuple repr: '()', '(4,)', '(2, 3)'.
        0 => "()",
        1 => $"({shape[0].ToString(CultureInfo.InvariantCulture)},)",
        _ => $"({string.Join(", ", shape.Select(d => d.ToString(CultureInfo.InvariantCulture)))})",
    };

    private static Array ReadElements(ReadOnlySpan<byte> data, DtypeToken dtype, int count)
    {
        Array result = Dtypes.CreateElementArray(dtype, count);
        int size = Dtypes.ItemSize(dtype);
        for (int i = 0; i < count; i++)
        {
            ReadOnlySpan<byte> element = data.Slice(i * size, size);
            switch (dtype)
            {
                case DtypeToken.Bool:
                    ((bool[])result)[i] = element[0] != 0;
                    break;
                case DtypeToken.Int8:
                    ((sbyte[])result)[i] = (sbyte)element[0];
                    break;
                case DtypeToken.UInt8:
                    ((byte[])result)[i] = element[0];
                    break;
                case DtypeToken.Int16:
                    ((short[])result)[i] = BinaryPrimitives.ReadInt16LittleEndian(element);
                    break;
                case DtypeToken.UInt16:
                    ((ushort[])result)[i] = BinaryPrimitives.ReadUInt16LittleEndian(element);
                    break;
                case DtypeToken.Int32:
                    ((int[])result)[i] = BinaryPrimitives.ReadInt32LittleEndian(element);
                    break;
                case DtypeToken.UInt32:
                    ((uint[])result)[i] = BinaryPrimitives.ReadUInt32LittleEndian(element);
                    break;
                case DtypeToken.Int64:
                    ((long[])result)[i] = BinaryPrimitives.ReadInt64LittleEndian(element);
                    break;
                case DtypeToken.UInt64:
                    ((ulong[])result)[i] = BinaryPrimitives.ReadUInt64LittleEndian(element);
                    break;
                case DtypeToken.Float16:
                    ((Half[])result)[i] = BitConverter.UInt16BitsToHalf(
                        BinaryPrimitives.ReadUInt16LittleEndian(element));
                    break;
                case DtypeToken.Float32:
                    ((float[])result)[i] = BitConverter.UInt32BitsToSingle(
                        BinaryPrimitives.ReadUInt32LittleEndian(element));
                    break;
                case DtypeToken.Float64:
                    ((double[])result)[i] = BitConverter.UInt64BitsToDouble(
                        BinaryPrimitives.ReadUInt64LittleEndian(element));
                    break;
                case DtypeToken.Complex64:
                    ((Complex[])result)[i] = new Complex(
                        BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(element)),
                        BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(element[4..])));
                    break;
                case DtypeToken.Complex128:
                    ((Complex[])result)[i] = new Complex(
                        BitConverter.UInt64BitsToDouble(BinaryPrimitives.ReadUInt64LittleEndian(element)),
                        BitConverter.UInt64BitsToDouble(BinaryPrimitives.ReadUInt64LittleEndian(element[8..])));
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(dtype));
            }
        }

        return result;
    }

    private static void WriteElements(Span<byte> destination, DtypeToken dtype, Array values)
    {
        int size = Dtypes.ItemSize(dtype);
        for (int i = 0; i < values.Length; i++)
        {
            Span<byte> element = destination.Slice(i * size, size);
            switch (dtype)
            {
                case DtypeToken.Bool:
                    element[0] = ((bool[])values)[i] ? (byte)1 : (byte)0;
                    break;
                case DtypeToken.Int8:
                    element[0] = (byte)((sbyte[])values)[i];
                    break;
                case DtypeToken.UInt8:
                    element[0] = ((byte[])values)[i];
                    break;
                case DtypeToken.Int16:
                    BinaryPrimitives.WriteInt16LittleEndian(element, ((short[])values)[i]);
                    break;
                case DtypeToken.UInt16:
                    BinaryPrimitives.WriteUInt16LittleEndian(element, ((ushort[])values)[i]);
                    break;
                case DtypeToken.Int32:
                    BinaryPrimitives.WriteInt32LittleEndian(element, ((int[])values)[i]);
                    break;
                case DtypeToken.UInt32:
                    BinaryPrimitives.WriteUInt32LittleEndian(element, ((uint[])values)[i]);
                    break;
                case DtypeToken.Int64:
                    BinaryPrimitives.WriteInt64LittleEndian(element, ((long[])values)[i]);
                    break;
                case DtypeToken.UInt64:
                    BinaryPrimitives.WriteUInt64LittleEndian(element, ((ulong[])values)[i]);
                    break;
                case DtypeToken.Float16:
                    BinaryPrimitives.WriteUInt16LittleEndian(
                        element, BitConverter.HalfToUInt16Bits(((Half[])values)[i]));
                    break;
                case DtypeToken.Float32:
                    BinaryPrimitives.WriteUInt32LittleEndian(
                        element, BitConverter.SingleToUInt32Bits(((float[])values)[i]));
                    break;
                case DtypeToken.Float64:
                    BinaryPrimitives.WriteUInt64LittleEndian(
                        element, BitConverter.DoubleToUInt64Bits(((double[])values)[i]));
                    break;
                case DtypeToken.Complex64:
                    BinaryPrimitives.WriteUInt32LittleEndian(
                        element, BitConverter.SingleToUInt32Bits((float)((Complex[])values)[i].Real));
                    BinaryPrimitives.WriteUInt32LittleEndian(
                        element[4..], BitConverter.SingleToUInt32Bits((float)((Complex[])values)[i].Imaginary));
                    break;
                case DtypeToken.Complex128:
                    BinaryPrimitives.WriteUInt64LittleEndian(
                        element, BitConverter.DoubleToUInt64Bits(((Complex[])values)[i].Real));
                    BinaryPrimitives.WriteUInt64LittleEndian(
                        element[8..], BitConverter.DoubleToUInt64Bits(((Complex[])values)[i].Imaginary));
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(dtype));
            }
        }
    }
}
