using System.Collections;
using System.Numerics;
using System.Numerics.Tensors;
using System.Text.Json;
using Versionable.Converters;
using Versionable.Errors;
using Versionable.Numerics;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// <c>Tensor&lt;T&gt;</c> ⇄ the ndarray wire dict, and the dtype rules that govern the load
/// side of it.
/// </summary>
public class TensorConverterTests
{
    /// <summary>Every (from, to) dtype pair numpy calls a safe cast, plus the identities.</summary>
    /// <remarks>
    /// Transcribed from <c>numpy.can_cast(a, b, casting="safe")</c> (numpy 2.4.3) rather than
    /// derived, because the relation is not "wider wins": <c>int64 → float64</c> is safe
    /// though it drops integers above 2^53, and <c>int32 → complex64</c> is not though
    /// complex64 is twice as wide.
    /// </remarks>
    public static TheoryData<string, string, bool> CastMatrix
    {
        get
        {
            // Grammar tokens rather than the internal DtypeToken: the token set is the public
            // vocabulary (GRAMMAR §7), and an internal enum cannot appear in a public
            // signature.
            string[] tokens = [.. Dtypes.All.Select(Dtypes.ToToken)];
            TheoryData<string, string, bool> data = [];
            foreach ((string from, string row) in tokens.Zip(SafeCastRows))
            {
                foreach ((string to, char safe) in tokens.Zip(row))
                {
                    data.Add(from, to, safe == '1');
                }
            }

            return data;
        }
    }

    private static string[] SafeCastRows =>
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

    [Theory]
    [MemberData(nameof(CastMatrix))]
    public void the_safe_cast_matrix_matches_numpy(string from, string to, bool safe)
    {
        Assert.Equal(safe, Dtypes.CanCastSafely(Dtype(from), Dtype(to)));
    }

    [Theory]
    // Pairs where the answer is not the one a "wider wins" intuition gives, written out by
    // hand from numpy rather than transcribed from a table.
    [InlineData("int64", "float64", true)] // safe, though it drops integers above 2^53
    [InlineData("uint64", "float64", true)]
    [InlineData("int32", "complex64", false)] // wider, but the components are float32
    [InlineData("int16", "complex64", true)]
    [InlineData("uint8", "int16", true)] // an unsigned byte fits a signed short
    [InlineData("int8", "uint16", false)] // a signed byte does not fit an unsigned short
    [InlineData("uint64", "int64", false)]
    [InlineData("int8", "float16", true)] // ±127 is exact in 11 mantissa bits
    [InlineData("int16", "float16", false)]
    [InlineData("float64", "float32", false)]
    [InlineData("bool", "int8", true)]
    [InlineData("int8", "bool", false)]
    [InlineData("complex128", "complex64", false)]
    public void notable_cast_pairs_follow_numpys_rule(string from, string to, bool safe)
    {
        Assert.Equal(safe, Dtypes.CanCastSafely(Dtype(from), Dtype(to)));
    }

    [Fact]
    public void every_dtype_is_safely_castable_to_itself()
    {
        foreach (DtypeToken dtype in Dtypes.All)
        {
            Assert.True(Dtypes.CanCastSafely(dtype, dtype));
        }
    }

    [Fact]
    public void serialization_names_are_the_parameterized_array_form()
    {
        // GRAMMAR §7: array dtype is hash-significant, so the name is not bare.
        Assert.Equal("ndarray[float64]", new TensorConverter<double>().SerializationName);
        Assert.Equal("ndarray[int32]", new TensorConverter<int>().SerializationName);
        Assert.Equal("ndarray[uint8]", new TensorConverter<byte>().SerializationName);
        Assert.Equal("ndarray[float16]", new TensorConverter<Half>().SerializationName);

        // Tensor<Complex> is complex128; complex64 has no C# element type.
        Assert.Equal("ndarray[complex128]", new TensorConverter<Complex>().SerializationName);
    }

    [Fact]
    public void an_element_type_with_no_dtype_token_is_rejected_at_construction()
    {
        Assert.Throws<UnsupportedTypeException>(() => new TensorConverter<string>());
        Assert.Throws<UnsupportedTypeException>(() => new TensorConverter<decimal>());
    }

    [Fact]
    public void the_wire_dict_carries_the_four_keys_python_writes()
    {
        Tensor<double> tensor = Tensor.Create(new[] { 0.5, -1.25, 2.0, 3.75 }, [(nint)4]);
        IDictionary wire = Assert.IsAssignableFrom<IDictionary>(new TensorConverter<double>().ToWire(tensor));

        Assert.Equal(true, wire["__ver_ndarray__"]);
        Assert.Equal("float64", wire["dtype"]);
        Assert.Equal(new List<object?> { 4L }, wire["shape"]);
        Assert.IsType<string>(wire["data"]);
    }

    [Fact]
    public void a_tensor_round_trips_through_the_wire_dict()
    {
        TensorConverter<double> converter = new();
        Tensor<double> tensor = Tensor.Create(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, [(nint)2, 3]);

        Tensor<double> restored = Assert.IsType<Tensor<double>>(
            converter.FromWire(converter.ToWire(tensor)!, typeof(Tensor<double>)));

        Assert.Equal(new nint[] { 2, 3 }, restored.Lengths.ToArray());
        Assert.Equal(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, Flatten(restored));
    }

    [Fact]
    public void an_empty_tensor_round_trips()
    {
        TensorConverter<int> converter = new();
        Tensor<int> tensor = Tensor.Create(Array.Empty<int>(), [(nint)0]);

        Tensor<int> restored = Assert.IsType<Tensor<int>>(
            converter.FromWire(converter.ToWire(tensor)!, typeof(Tensor<int>)));
        Assert.Equal(new nint[] { 0 }, restored.Lengths.ToArray());
    }

    [Fact]
    public void a_strided_view_writes_the_elements_it_holds()
    {
        // FlattenTo materialises the logical elements, so a slice does not leak the buffer it
        // points into.
        TensorConverter<int> converter = new();
        Tensor<int> full = Tensor.Create(new[] { 1, 2, 3, 4, 5, 6 }, [(nint)2, 3]);
        Tensor<int> slice = full.Slice([0..1, 0..3]);

        Tensor<int> restored = Assert.IsType<Tensor<int>>(
            converter.FromWire(converter.ToWire(slice)!, typeof(Tensor<int>)));
        Assert.Equal(new nint[] { 1, 3 }, restored.Lengths.ToArray());
        Assert.Equal(new[] { 1, 2, 3 }, Flatten(restored));
    }

    [Fact]
    public void a_safe_dtype_mismatch_is_cast_silently()
    {
        // float32 on disk, float64 declared: numpy calls that safe, so it is applied without
        // a word, exactly as Python's coerceArrayDtype does.
        object wire = new TensorConverter<float>().ToWire(Tensor.Create(new[] { 0.5f, 0.25f }, [(nint)2]))!;

        Tensor<double> restored = Assert.IsType<Tensor<double>>(
            new TensorConverter<double>().FromWire(wire, typeof(Tensor<double>)));
        Assert.Equal(new[] { 0.5, 0.25 }, Flatten(restored));
    }

    [Fact]
    public void an_unsafe_dtype_mismatch_raises()
    {
        object wire = new TensorConverter<double>().ToWire(Tensor.Create(new[] { 0.5 }, [(nint)1]))!;

        DtypeMismatchException error = Assert.Throws<DtypeMismatchException>(
            () => new TensorConverter<float>().FromWire(wire, typeof(Tensor<float>)));
        Assert.Equal("float32", error.Declared);
        Assert.Equal("float64", error.Actual);
        Assert.Equal(DtypeContext.Load, error.Context);
    }

    [Fact]
    public void an_integer_widening_that_numpy_calls_unsafe_raises()
    {
        // int32 -> int16 loses range; int32 -> int64 does not.
        object wire = new TensorConverter<int>().ToWire(Tensor.Create(new[] { 7 }, [(nint)1]))!;

        Assert.Throws<DtypeMismatchException>(
            () => new TensorConverter<short>().FromWire(wire, typeof(Tensor<short>)));
        Assert.IsType<Tensor<long>>(new TensorConverter<long>().FromWire(wire, typeof(Tensor<long>)));
    }

    [Fact]
    public void a_complex64_payload_widens_into_a_complex_tensor()
    {
        // complex64 has no C# element type, so it can only ever arrive from Python — and
        // widening it to complex128 is a cast numpy calls safe.
        byte[] npz = NpzCodec.Encode(DtypeToken.Complex64, [2], new[] { new Complex(1.5, -2.5), Complex.One });
        Dictionary<string, object?> wire = Payload(npz);

        Tensor<Complex> restored = Assert.IsType<Tensor<Complex>>(
            new TensorConverter<Complex>().FromWire(wire, typeof(Tensor<Complex>)));
        Assert.Equal(new[] { new Complex(1.5, -2.5), Complex.One }, Flatten(restored));

        Assert.Throws<DtypeMismatchException>(
            () => new TensorConverter<double>().FromWire(wire, typeof(Tensor<double>)));
    }

    [Fact]
    public void a_zero_d_payload_is_rejected_with_an_explanation()
    {
        // Tensor<T> normalises an empty shape to rank 1 length 0, so a numpy scalar array has
        // no faithful representation and is refused rather than silently reshaped.
        byte[] npz = NpzCodec.Encode(DtypeToken.Float64, [], new[] { 42.0 });
        Dictionary<string, object?> wire = Payload(npz);

        ConverterException error = Assert.Throws<ConverterException>(
            () => new TensorConverter<double>().FromWire(wire, typeof(Tensor<double>)));
        Assert.Contains("0-d", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_native_tensor_passes_straight_through()
    {
        // The HDF5 backend hands arrays back without an NPZ in between.
        Tensor<double> tensor = Tensor.Create(new[] { 1.0 }, [(nint)1]);
        Assert.Same(tensor, new TensorConverter<double>().FromWire(tensor, typeof(Tensor<double>)));
    }

    [Fact]
    public void a_dict_without_an_ndarray_marker_is_rejected()
    {
        // Python's _deserializeNdarray gates on the marker before it reads 'data', so a plain
        // dict field that happens to hold a 'data' string must not decode as an array here
        // either.
        byte[] npz = NpzCodec.Encode(DtypeToken.Float64, [1], new[] { 1.0 });
        Dictionary<string, object?> wire = new() { ["data"] = Convert.ToBase64String(npz) };

        ConverterException error = Assert.Throws<ConverterException>(
            () => new TensorConverter<double>().FromWire(wire, typeof(Tensor<double>)));
        Assert.Contains("__ver_ndarray__", error.Message, StringComparison.Ordinal);
        Assert.Contains("__ndarray__", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void the_pre_0_2_marker_key_is_still_accepted()
    {
        // Python reads either key; files written before 0.2 carry the short one.
        byte[] npz = NpzCodec.Encode(DtypeToken.Float64, [2], new[] { 1.0, 2.0 });
        Dictionary<string, object?> wire = new()
        {
            ["__ndarray__"] = true,
            ["data"] = Convert.ToBase64String(npz),
        };

        Tensor<double> restored = Assert.IsType<Tensor<double>>(
            new TensorConverter<double>().FromWire(wire, typeof(Tensor<double>)));
        Assert.Equal(new[] { 1.0, 2.0 }, Flatten(restored));
    }

    [Fact]
    public void a_falsy_marker_reads_as_absent()
    {
        // Python tests the marker with `or`, so False and 0 are not markers.
        byte[] npz = NpzCodec.Encode(DtypeToken.Float64, [1], new[] { 1.0 });
        Dictionary<string, object?> wire = new()
        {
            ["__ver_ndarray__"] = false,
            ["data"] = Convert.ToBase64String(npz),
        };

        Assert.Throws<ConverterException>(
            () => new TensorConverter<double>().FromWire(wire, typeof(Tensor<double>)));
    }

    [Fact]
    public void a_dict_without_a_data_key_is_rejected()
    {
        Dictionary<string, object?> wire = new()
        {
            ["__ver_ndarray__"] = true,
            ["dtype"] = "float64",
            ["shape"] = new List<object?> { 1L },
        };
        Assert.Throws<ConverterException>(
            () => new TensorConverter<double>().FromWire(wire, typeof(Tensor<double>)));
    }

    [Fact]
    public void the_sidecar_dtype_does_not_override_the_payload()
    {
        // dtype and shape are written for human readers; the NPZ inside is authoritative, so
        // a hand-edited sidecar cannot make a file decode as something its bytes do not say.
        byte[] npz = NpzCodec.Encode(DtypeToken.Float64, [2], new[] { 1.0, 2.0 });
        Dictionary<string, object?> wire = new()
        {
            ["__ver_ndarray__"] = true,
            ["dtype"] = "int32",
            ["shape"] = new List<object?> { 99L },
            ["data"] = Convert.ToBase64String(npz),
        };

        Tensor<double> restored = Assert.IsType<Tensor<double>>(
            new TensorConverter<double>().FromWire(wire, typeof(Tensor<double>)));
        Assert.Equal(new nint[] { 2 }, restored.Lengths.ToArray());
    }

    [Theory]
    [InlineData("signal", "float64")]
    [InlineData("weights", "float32")]
    [InlineData("counts", "int32")]
    [InlineData("image", "uint8")]
    [InlineData("mask", "bool")]
    [InlineData("matrix", "float64")]
    public void golden_payloads_load_into_the_declared_tensor_type(string field, string dtype)
    {
        JsonElement wire = ConverterTestGolden.Wire("arrays").GetProperty(field);
        Dictionary<string, object?> asDict = new()
        {
            ["__ver_ndarray__"] = wire.GetProperty("__ver_ndarray__").GetBoolean(),
            ["data"] = wire.GetProperty("data").GetString(),
        };
        JsonElement expected = ConverterTestGolden.Manifest("arrays")
            .GetProperty(field)
            .GetProperty("$ndarray");

        object restored = ConverterFor(dtype).FromWire(asDict, typeof(object));
        Assert.Equal(
            expected.GetProperty("shape").EnumerateArray().Select(e => (nint)e.GetInt64()),
            LengthsOf(restored));
    }

    private static IWireConverter ConverterFor(string dtype) => dtype switch
    {
        "float64" => new TensorConverter<double>(),
        "float32" => new TensorConverter<float>(),
        "int32" => new TensorConverter<int>(),
        "uint8" => new TensorConverter<byte>(),
        "bool" => new TensorConverter<bool>(),
        _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
    };

    private static nint[] LengthsOf(object tensor) => tensor switch
    {
        Tensor<double> t => t.Lengths.ToArray(),
        Tensor<float> t => t.Lengths.ToArray(),
        Tensor<int> t => t.Lengths.ToArray(),
        Tensor<byte> t => t.Lengths.ToArray(),
        Tensor<bool> t => t.Lengths.ToArray(),
        _ => throw new ArgumentOutOfRangeException(nameof(tensor)),
    };

    private static DtypeToken Dtype(string token)
    {
        Assert.True(Dtypes.TryParseToken(token, out DtypeToken dtype));
        return dtype;
    }

    /// <summary>Builds the wire dict a backend would hand back for <paramref name="npz"/>.</summary>
    private static Dictionary<string, object?> Payload(byte[] npz) => new()
    {
        ["__ver_ndarray__"] = true,
        ["data"] = Convert.ToBase64String(npz),
    };

    private static T[] Flatten<T>(Tensor<T> tensor)
    {
        T[] flat = new T[(int)tensor.FlattenedLength];
        tensor.FlattenTo(flat);
        return flat;
    }
}
