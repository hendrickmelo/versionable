using System.Numerics;
using System.Text;
using Versionable.Errors;
using Versionable.Numerics;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// The NPY container: header layout, the closed dtype set, and the shapes that are easy to
/// get wrong.
/// </summary>
/// <remarks>
/// Expected header lengths are numpy's, read off <c>numpy.save</c> output (numpy 2.4.3) — the
/// format is defined by what numpy writes, not by prose.
/// </remarks>
public class NpyCodecTests
{
    [Fact]
    public void header_matches_numpys_layout()
    {
        byte[] header = NpyCodec.BuildHeader(DtypeToken.Float64, [4]);

        Assert.Equal(128, header.Length);
        Assert.Equal(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' }, header[..6]);
        Assert.Equal(1, header[6]);
        Assert.Equal(0, header[7]);

        // numpy pads so the data starts on a 64-byte boundary: 10 prefix bytes + 118.
        Assert.Equal(118, header[8] | (header[9] << 8));

        string text = Encoding.ASCII.GetString(header, 10, 118);
        Assert.Equal("{'descr': '<f8', 'fortran_order': False, 'shape': (4,), }", text.TrimEnd(' ', '\n'));
        Assert.EndsWith("\n", text, StringComparison.Ordinal);
        Assert.All(text[..^1], c => Assert.NotEqual('\n', c));
    }

    [Fact]
    public void header_reserves_numpys_growth_space_for_the_leading_axis()
    {
        // numpy leaves 21 - len(repr(shape[0])) spare spaces after the dict so the leading
        // axis can be grown in place. Usually the 64-byte alignment swallows them; a
        // twenty-dimensional shape is the case where it does not, and numpy's own header for
        // it is 182 bytes rather than 118.
        long[] shape = [.. Enumerable.Repeat(1L, 20)];
        byte[] header = NpyCodec.BuildHeader(DtypeToken.UInt8, shape);

        Assert.Equal(192, header.Length);
        Assert.Equal(182, header[8] | (header[9] << 8));
    }

    [Theory]
    [InlineData("bool", "|b1")]
    [InlineData("int8", "|i1")]
    [InlineData("uint8", "|u1")]
    [InlineData("int16", "<i2")]
    [InlineData("int32", "<i4")]
    [InlineData("int64", "<i8")]
    [InlineData("uint16", "<u2")]
    [InlineData("uint32", "<u4")]
    [InlineData("uint64", "<u8")]
    [InlineData("float16", "<f2")]
    [InlineData("float32", "<f4")]
    [InlineData("float64", "<f8")]
    [InlineData("complex64", "<c8")]
    [InlineData("complex128", "<c16")]
    public void every_dtype_writes_the_descr_numpy_writes(string token, string descr)
    {
        // Grammar tokens rather than the internal enum: the token set is the public
        // vocabulary (GRAMMAR §7), and an internal enum cannot appear in a public signature.
        Assert.True(Dtypes.TryParseToken(token, out DtypeToken dtype));

        byte[] header = NpyCodec.BuildHeader(dtype, [1]);
        Assert.Contains($"'descr': '{descr}'", Encoding.ASCII.GetString(header), StringComparison.Ordinal);
        Assert.True(Dtypes.TryParseDescr(descr, out DtypeToken parsed));
        Assert.Equal(dtype, parsed);
    }

    [Fact]
    public void every_dtype_round_trips_its_values()
    {
        AssertRoundTrip(DtypeToken.Bool, new[] { true, false, true });
        AssertRoundTrip(DtypeToken.Int8, new sbyte[] { -128, 0, 127 });
        AssertRoundTrip(DtypeToken.UInt8, new byte[] { 0, 128, 255 });
        AssertRoundTrip(DtypeToken.Int16, new short[] { short.MinValue, 0, short.MaxValue });
        AssertRoundTrip(DtypeToken.UInt16, new ushort[] { 0, ushort.MaxValue });
        AssertRoundTrip(DtypeToken.Int32, new[] { int.MinValue, 0, int.MaxValue });
        AssertRoundTrip(DtypeToken.UInt32, new[] { 0u, uint.MaxValue });
        AssertRoundTrip(DtypeToken.Int64, new[] { long.MinValue, 0L, long.MaxValue });
        AssertRoundTrip(DtypeToken.UInt64, new[] { 0ul, ulong.MaxValue });
        AssertRoundTrip(DtypeToken.Float16, new[] { (Half)0.5f, Half.NegativeInfinity, Half.NaN });
        AssertRoundTrip(DtypeToken.Float32, new[] { 0.5f, -1.25f, float.NaN });
        AssertRoundTrip(DtypeToken.Float64, new[] { 0.5, -1.25, double.PositiveInfinity });
        AssertRoundTrip(DtypeToken.Complex128, new[] { new Complex(1.5, -2.5) });
        AssertRoundTrip(DtypeToken.Complex64, new[] { new Complex(1.5, -2.5) });
    }

    [Fact]
    public void a_zero_d_array_holds_one_element()
    {
        // shape () is not shape (0,): numpy's scalar array has exactly one element, and the
        // '()' repr has no dimensions to multiply out.
        byte[] encoded = NpyCodec.Encode(DtypeToken.Float64, [], new[] { 42.0 });
        Assert.Equal(136, encoded.Length);

        NpyArray decoded = NpyCodec.Decode(encoded);
        Assert.Empty(decoded.Shape);
        Assert.Equal(new[] { 42.0 }, (double[])decoded.Values);
    }

    [Fact]
    public void an_empty_array_holds_no_elements()
    {
        byte[] encoded = NpyCodec.Encode(DtypeToken.Float64, [0], Array.Empty<double>());
        Assert.Equal(128, encoded.Length);

        NpyArray decoded = NpyCodec.Decode(encoded);
        Assert.Equal(new long[] { 0 }, decoded.Shape);
        Assert.Empty((double[])decoded.Values);
    }

    [Fact]
    public void a_multidimensional_shape_survives_and_stays_in_c_order()
    {
        double[] values = [1, 2, 3, 4, 5, 6];
        NpyArray decoded = NpyCodec.Decode(NpyCodec.Encode(DtypeToken.Float64, [2, 3], values));

        Assert.Equal(new long[] { 2, 3 }, decoded.Shape);
        Assert.Equal(values, (double[])decoded.Values);
    }

    [Fact]
    public void a_shape_that_disagrees_with_the_buffer_is_rejected()
    {
        Assert.Throws<ConverterException>(
            () => NpyCodec.Encode(DtypeToken.Float64, [2, 3], new double[5]));
    }

    [Fact]
    public void fortran_order_is_rejected_rather_than_transposed()
    {
        byte[] payload = WithHeaderText("{'descr': '<f8', 'fortran_order': True, 'shape': (2, 3), }", 48);
        ConverterException error = Assert.Throws<ConverterException>(() => NpyCodec.Decode(payload));
        Assert.Contains("Fortran", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void big_endian_multibyte_dtypes_are_rejected()
    {
        Assert.False(Dtypes.TryParseDescr(">f8", out _));
        Assert.False(Dtypes.TryParseDescr(">i4", out _));

        // A one-byte type carries no byte order, so any prefix means the same bytes.
        Assert.True(Dtypes.TryParseDescr(">u1", out DtypeToken uint8));
        Assert.Equal(DtypeToken.UInt8, uint8);
    }

    [Theory]
    [InlineData("<f10")] // not a width numpy has
    [InlineData("<U4")] // unicode strings are outside the closed set
    [InlineData("<m8[ns]")] // datetime64/timedelta64 likewise
    public void dtypes_outside_the_closed_set_are_rejected(string descr)
    {
        Assert.False(Dtypes.TryParseDescr(descr, out _));
    }

    [Fact]
    public void a_payload_without_the_magic_prefix_is_rejected()
    {
        Assert.Throws<ConverterException>(() => NpyCodec.Decode([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
    }

    [Theory]
    [InlineData(1, 9)] // a 1.0 header needs 2 length bytes
    [InlineData(2, 11)] // a 2.0 header needs 4
    [InlineData(3, 11)]
    public void a_payload_too_short_for_its_length_field_is_rejected(byte major, int length)
    {
        // The guard has to run before the read, or the wider length field faults with an
        // ArgumentOutOfRangeException instead of reporting a truncation.
        byte[] stub = new byte[length];
        NpyCodec.Magic.CopyTo(stub);
        stub[6] = major;

        Assert.Throws<ConverterException>(() => NpyCodec.Decode(stub));
    }

    [Fact]
    public void a_high_byte_in_a_1_0_header_is_read_as_latin_1()
    {
        // 1.0 and 2.0 headers are latin-1, not UTF-8. A lone high byte is valid latin-1 and
        // invalid UTF-8, so decoding as UTF-8 would replace it with U+FFFD; either way the
        // dtype is outside the closed set, but the failure must be the dtype's, not a decoder
        // fault.
        byte[] payload = WithHeaderText("{'descr': '<\u00e9 8', 'fortran_order': False, 'shape': (1,), }", 8);
        Assert.Throws<ConverterException>(() => NpyCodec.Decode(payload));
    }

    [Fact]
    public void a_truncated_data_section_is_rejected()
    {
        byte[] encoded = NpyCodec.Encode(DtypeToken.Float64, [4], new double[4]);
        Assert.Throws<ConverterException>(() => NpyCodec.Decode(encoded[..^8]));
    }

    [Fact]
    public void a_format_2_header_is_read()
    {
        // 2.0 differs from 1.0 only in the width of the length field. numpy switches to it
        // for headers past 64 KiB; C# never writes one but must read one.
        byte[] one = NpyCodec.Encode(DtypeToken.Int32, [3], new[] { -2, 0, 7 });
        int headerLength = one[8] | (one[9] << 8);

        byte[] two = new byte[one.Length + 2];
        one.AsSpan(0, 8).CopyTo(two);
        two[6] = 2;
        BitConverter.GetBytes(headerLength).CopyTo(two, 8);
        one.AsSpan(10).CopyTo(two.AsSpan(12));

        NpyArray decoded = NpyCodec.Decode(two);
        Assert.Equal(new long[] { 3 }, decoded.Shape);
        Assert.Equal(new[] { -2, 0, 7 }, (int[])decoded.Values);
    }

    private static void AssertRoundTrip(DtypeToken dtype, Array values)
    {
        byte[] encoded = NpyCodec.Encode(dtype, [values.Length], values);
        NpyArray decoded = NpyCodec.Decode(encoded);

        Assert.Equal(dtype, decoded.Dtype);
        Assert.Equal(new long[] { values.Length }, decoded.Shape);
        Assert.Equal(values, decoded.Values);
    }

    /// <summary>Builds a valid 1.0 payload around a hand-written header dict.</summary>
    private static byte[] WithHeaderText(string dict, int dataBytes)
    {
        int contentLength = dict.Length + 1;
        int padding = 64 - ((10 + contentLength) % 64);
        int declaredLength = contentLength + padding;

        byte[] result = new byte[10 + declaredLength + dataBytes];
        NpyCodec.Magic.CopyTo(result);
        result[6] = 1;
        result[8] = (byte)(declaredLength & 0xFF);
        result[9] = (byte)(declaredLength >> 8);
        Encoding.Latin1.GetBytes(dict).CopyTo(result, 10);
        result.AsSpan(10 + dict.Length, padding).Fill((byte)' ');
        result[10 + declaredLength - 1] = (byte)'\n';
        return result;
    }
}
