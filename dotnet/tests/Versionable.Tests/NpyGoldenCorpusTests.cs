using System.IO.Compression;
using System.Text.Json;
using Versionable.Errors;
using Versionable.Numerics;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// The NPZ codec against the golden corpus, which is the byte-level oracle: every payload in
/// <c>conformance/golden/arrays/arrays.json</c> was written by
/// <c>numpy.savez_compressed</c> through Python's <c>versionable</c>.
/// </summary>
public class NpyGoldenCorpusTests
{
    /// <summary>Every array field of the <c>arrays</c> fixture, with the manifest path to it.</summary>
    public static TheoryData<string> ArrayFields =>
        ["signal", "weights", "counts", "image", "mask", "matrix"];

    [Theory]
    [MemberData(nameof(ArrayFields))]
    public void golden_npz_payloads_decode_to_the_manifest_values(string field)
    {
        JsonElement expected = ConverterTestGolden.Manifest("arrays")
            .GetProperty(field)
            .GetProperty("$ndarray");
        NpyArray decoded = NpzCodec.Decode(PayloadOf(field));

        Assert.Equal(expected.GetProperty("dtype").GetString(), Dtypes.ToToken(decoded.Dtype));
        Assert.Equal(
            expected.GetProperty("shape").EnumerateArray().Select(e => e.GetInt64()),
            decoded.Shape);
        Assert.Equal(Flatten(expected.GetProperty("data")), AsDoubles(decoded.Values));
    }

    [Fact]
    public void golden_npz_entries_are_a_single_deflated_data_npy()
    {
        // The entry name and the compression are the interchange contract, derived from
        // numpy.savez_compressed(buf, data=arr) in _serializeNdarray.
        using MemoryStream source = new(PayloadOf("signal"), writable: false);
        using ZipArchive archive = new(source, ZipArchiveMode.Read);

        ZipArchiveEntry entry = Assert.Single(archive.Entries);
        Assert.Equal("data.npy", entry.FullName);
        Assert.True(entry.CompressedLength < entry.Length);
    }

    [Theory]
    [MemberData(nameof(ArrayFields))]
    public void re_encoding_a_golden_array_reproduces_numpys_npy_bytes(string field)
    {
        // Byte identity is required of the NPY payload, not of the ZIP around it: numpy's
        // zipfile writes ZIP64 sentinel sizes into the local header and .NET's does not.
        byte[] original = NpyBytesOf(field);
        NpyArray decoded = NpyCodec.Decode(original);

        Assert.Equal(original, NpyCodec.Encode(decoded.Dtype, decoded.Shape, decoded.Values));
    }

    [Fact]
    public void nested_golden_arrays_decode_too()
    {
        // 'traces' is a list of arrays and 'channels' a dict of them; the payloads are the
        // same shape as a top-level field's.
        JsonElement wire = ConverterTestGolden.Wire("arrays");

        double[][] traces = [.. wire.GetProperty("traces").EnumerateArray()
            .Select(e => AsDoubles(NpzCodec.Decode(PayloadOf(e)).Values))];
        Assert.Equal(new double[][] { [1.0, 2.0], [3.0, 4.0, 5.0] }, traces);

        NpyArray channel0 = NpzCodec.Decode(PayloadOf(wire.GetProperty("channels").GetProperty("ch0")));
        Assert.Equal(new[] { 0.25, 0.5 }, AsDoubles(channel0.Values));
    }

    [Fact]
    public void a_written_npz_reads_back()
    {
        double[] values = [0.5, -1.25, 2.0, 3.75];
        byte[] npz = NpzCodec.Encode(DtypeToken.Float64, [2, 2], values);

        NpyArray decoded = NpzCodec.Decode(npz);
        Assert.Equal(DtypeToken.Float64, decoded.Dtype);
        Assert.Equal(new long[] { 2, 2 }, decoded.Shape);
        Assert.Equal(values, (double[])decoded.Values);
    }

    [Fact]
    public void a_written_npz_does_not_embed_the_wall_clock()
    {
        // The same array must encode to the same base64 on every run, or every save would
        // show a diff.
        byte[] first = NpzCodec.Encode(DtypeToken.Int32, [2], new[] { 1, 2 });
        byte[] second = NpzCodec.Encode(DtypeToken.Int32, [2], new[] { 1, 2 });
        Assert.Equal(first, second);
    }

    [Fact]
    public void an_npz_without_the_data_entry_is_rejected()
    {
        using MemoryStream buffer = new();
        using (ZipArchive archive = new(buffer, ZipArchiveMode.Create, leaveOpen: true))
        {
            archive.CreateEntry("other.npy");
        }

        ConverterException error = Assert.Throws<ConverterException>(
            () => NpzCodec.Decode(buffer.ToArray()));
        Assert.Contains("other.npy", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_payload_that_is_not_a_zip_is_rejected()
    {
        Assert.Throws<ConverterException>(() => NpzCodec.Decode([0, 1, 2, 3, 4, 5, 6, 7]));
    }

    private static byte[] PayloadOf(string field) =>
        PayloadOf(ConverterTestGolden.Wire("arrays").GetProperty(field));

    private static byte[] PayloadOf(JsonElement ndarray)
    {
        Assert.True(ndarray.GetProperty("__ver_ndarray__").GetBoolean());
        return Convert.FromBase64String(ndarray.GetProperty("data").GetString()!);
    }

    private static byte[] NpyBytesOf(string field)
    {
        using MemoryStream source = new(PayloadOf(field), writable: false);
        using ZipArchive archive = new(source, ZipArchiveMode.Read);
        using Stream entry = archive.GetEntry("data.npy")!.Open();
        using MemoryStream buffer = new();
        entry.CopyTo(buffer);
        return buffer.ToArray();
    }

    /// <summary>Reads a manifest <c>data</c> value, of any nesting depth, into a flat list.</summary>
    private static double[] Flatten(JsonElement data)
    {
        if (data.ValueKind != JsonValueKind.Array)
        {
            return [ToDouble(data)];
        }

        return [.. data.EnumerateArray().SelectMany(Flatten)];
    }

    private static double ToDouble(JsonElement value) => value.ValueKind switch
    {
        JsonValueKind.True => 1,
        JsonValueKind.False => 0,
        _ => value.GetDouble(),
    };

    /// <summary>Widens a decoded buffer of any dtype to doubles so one assertion covers all.</summary>
    private static double[] AsDoubles(Array values) => values switch
    {
        bool[] a => [.. a.Select(v => v ? 1.0 : 0.0)],
        _ => [.. values.Cast<object>().Select(v => Convert.ToDouble(v, null))],
    };
}
