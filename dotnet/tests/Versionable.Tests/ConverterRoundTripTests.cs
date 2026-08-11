using System.Globalization;
using System.Numerics;
using System.Text.Json;
using System.Text.RegularExpressions;
using Versionable.Converters;
using Versionable.Errors;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// The scalar converter set: wire shape, round trips, and the failure modes that are
/// deliberate rather than accidental.
/// </summary>
public class ConverterRoundTripTests
{
    // Reflection over the internal converters would make the tests agree with whatever the
    // code does; naming them keeps the expected wire form written out.
    private static readonly GuidConverter _guid = new();
    private static readonly DecimalConverter _decimal = new();
    private static readonly BytesConverter _bytes = new();
    private static readonly ComplexConverter _complex = new();
    private static readonly DateTimeConverter _dateTime = new();
    private static readonly DateTimeOffsetConverter _dateTimeOffset = new();
    private static readonly DateOnlyConverter _dateOnly = new();
    private static readonly TimeOnlyConverter _timeOnly = new();
    private static readonly TimeSpanConverter _timeSpan = new();
    private static readonly RegexConverter _regex = new();
    private static readonly FilePathConverter _filePath = new();

    [Fact]
    public void canonical_names_match_the_grammar_table()
    {
        // conformance/GRAMMAR.md §9, "Built-in converter types and their canonical names".
        Assert.Equal("UUID", _guid.SerializationName);
        Assert.Equal("Decimal", _decimal.SerializationName);
        Assert.Equal("bytes", _bytes.SerializationName);
        Assert.Equal("complex", _complex.SerializationName);
        Assert.Equal("datetime", _dateTime.SerializationName);
        Assert.Equal("datetime", _dateTimeOffset.SerializationName);
        Assert.Equal("date", _dateOnly.SerializationName);
        Assert.Equal("time", _timeOnly.SerializationName);
        Assert.Equal("timedelta", _timeSpan.SerializationName);
        Assert.Equal("Pattern", _regex.SerializationName);
        Assert.Equal("Path", _filePath.SerializationName);
    }

    [Fact]
    public void the_builtin_set_covers_every_converter_type_in_the_grammar()
    {
        // GRAMMAR §9's converter table plus §7's thirteen Tensor element types. The registry
        // itself is not touched: it is a process-wide static that another test class resets,
        // and BuiltinConverters.All is what that reset would put back.
        string[] names = [.. BuiltinConverters.All.Select(c => c.SerializationName).Order(StringComparer.Ordinal)];

        Assert.Equal(
            [
                "Decimal", "Path", "Pattern", "UUID", "bytes", "complex", "date", "datetime",
                "datetime", "ndarray[bool]", "ndarray[complex128]", "ndarray[float16]",
                "ndarray[float32]", "ndarray[float64]", "ndarray[int16]", "ndarray[int32]",
                "ndarray[int64]", "ndarray[int8]", "ndarray[uint16]", "ndarray[uint32]",
                "ndarray[uint64]", "ndarray[uint8]", "time", "timedelta",
            ],
            names);

        // Every CLR type is claimed once, so a registration can never quietly shadow another.
        Assert.Equal(
            BuiltinConverters.All.Length,
            BuiltinConverters.All.Select(c => c.ClrType).Distinct().Count());
    }

    [Fact]
    public void only_regex_matches_subclasses()
    {
        // Python sets matchSubclasses=True for re.Pattern and pathlib.Path. Only Regex needs
        // it here: FilePath is sealed, so an exact match already covers every value it can
        // hold, whereas Regex is not and a derived regex type must still route through the
        // converter.
        Assert.True(_regex.MatchSubclasses);
        Assert.False(_filePath.MatchSubclasses);
        Assert.False(_guid.MatchSubclasses);
        Assert.False(_dateTime.MatchSubclasses);
    }

    [Fact]
    public void guid_wires_as_a_lowercase_hyphenated_uuid()
    {
        Guid value = new("6BA7B810-9DAD-11D1-80B4-00C04FD430C8");
        Assert.Equal("6ba7b810-9dad-11d1-80b4-00c04fd430c8", _guid.ToWire(value));
        Assert.Equal(value, _guid.FromWire("6ba7b810-9dad-11d1-80b4-00c04fd430c8", typeof(Guid)));
    }

    [Theory]
    [InlineData("6ba7b8109dad11d180b400c04fd430c8")] // unhyphenated: uuid.UUID takes it, no writer emits it
    [InlineData("{6ba7b810-9dad-11d1-80b4-00c04fd430c8}")]
    [InlineData("not-a-uuid")]
    public void guid_rejects_forms_no_writer_produces(string wire)
    {
        Assert.Throws<ConverterException>(() => _guid.FromWire(wire, typeof(Guid)));
    }

    [Fact]
    public void decimal_wires_as_a_string_keeping_trailing_zeros()
    {
        Assert.Equal("1234.5678", _decimal.ToWire(1234.5678m));
        Assert.Equal(1234.5678m, _decimal.FromWire("1234.5678", typeof(decimal)));

        // The scale is part of a System.Decimal's value, as it is part of a Python Decimal's.
        Assert.Equal("1.500", _decimal.ToWire(1.500m));
    }

    [Fact]
    public void decimal_reads_the_exponent_form_python_can_write()
    {
        Assert.Equal(12300000m, _decimal.FromWire("1.23E+7", typeof(decimal)));
    }

    [Theory]
    [InlineData("1E+50")] // Python's Decimal exponent is unbounded; System.Decimal's is not
    [InlineData("-1E+50")]
    [InlineData("NaN")]
    [InlineData("Infinity")]
    [InlineData("")]
    public void decimal_rejects_values_outside_system_decimal(string wire)
    {
        Assert.Throws<ConverterException>(() => _decimal.FromWire(wire, typeof(decimal)));
    }

    [Fact]
    public void bytes_wire_as_padded_base64()
    {
        byte[] value = [0x00, 0x01, 0xFE, 0xFF];
        Assert.Equal("AAH+/w==", _bytes.ToWire(value));
        Assert.Equal(value, _bytes.FromWire("AAH+/w==", typeof(byte[])));
        Assert.Equal(Array.Empty<byte>(), _bytes.FromWire(string.Empty, typeof(byte[])));
    }

    [Fact]
    public void bytes_reject_non_base64()
    {
        Assert.Throws<ConverterException>(() => _bytes.FromWire("not base64!", typeof(byte[])));
    }

    [Fact]
    public void complex_wires_as_a_two_element_list()
    {
        Complex value = new(1.5, -2.25);
        Assert.Equal(new List<object?> { 1.5, -2.25 }, _complex.ToWire(value));
        Assert.Equal(value, _complex.FromWire(new List<object?> { 1.5, -2.25 }, typeof(Complex)));

        // Backends decode numbers to whatever their parser picks; a TOML [1, 2] arrives long.
        Assert.Equal(new Complex(1, 2), _complex.FromWire(new object?[] { 1L, 2 }, typeof(Complex)));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void complex_needs_exactly_two_components(int count)
    {
        List<object?> wire = [.. Enumerable.Repeat((object?)1.0, count)];
        Assert.Throws<ConverterException>(() => _complex.FromWire(wire, typeof(Complex)));
    }

    [Theory]
    // Python's isoformat() writes no fraction at all on a whole second, and exactly six
    // digits otherwise; C# matches both rather than merely round-tripping.
    [InlineData(0, "2026-08-05T14:30:15")]
    [InlineData(5_000_000, "2026-08-05T14:30:15.500000")]
    [InlineData(1_234_560, "2026-08-05T14:30:15.123456")]
    // A tick that is not a whole microsecond has no six-digit form, so the seventh appears.
    [InlineData(1_234_567, "2026-08-05T14:30:15.1234567")]
    [InlineData(1, "2026-08-05T14:30:15.0000001")]
    public void datetime_fraction_digits_follow_the_value(long subsecondTicks, string expected)
    {
        DateTime value = new DateTime(2026, 8, 5, 14, 30, 15, DateTimeKind.Unspecified)
            .AddTicks(subsecondTicks);
        Assert.Equal(expected, _dateTime.ToWire(value));
        Assert.Equal(value, _dateTime.FromWire(expected, typeof(DateTime)));
    }

    [Fact]
    public void datetime_rejects_an_aware_wire_value()
    {
        // Silently shifting into local time would make the same file load differently per
        // machine; the field should be a DateTimeOffset instead.
        ConverterException error = Assert.Throws<ConverterException>(
            () => _dateTime.FromWire("2026-08-05T14:30:15.123456-05:00", typeof(DateTime)));
        Assert.Contains("DateTimeOffset", error.Message, StringComparison.Ordinal);

        Assert.Throws<ConverterException>(
            () => _dateTime.FromWire("2026-08-05T14:30:15Z", typeof(DateTime)));
    }

    [Theory]
    [InlineData("08/05/2026 14:30:15")] // .NET's general TryParse takes these; fromisoformat
    [InlineData("Aug 5, 2026 2:30:15 PM")] // does not, so neither may this converter
    [InlineData("2026-8-5T14:30:15")]
    [InlineData("")]
    public void datetime_rejects_what_python_cannot_read(string wire)
    {
        Assert.Throws<ConverterException>(() => _dateTime.FromWire(wire, typeof(DateTime)));
    }

    [Fact]
    public void the_compact_iso_form_is_rejected_across_the_temporal_set()
    {
        // fromisoformat has read the compact forms since Python 3.11, so this is the one place
        // the converters are narrower than Python. It is excluded everywhere rather than in
        // some converters, because a set where datetime took a compact string and date did not
        // would be harder to predict than one that says no throughout. Nothing writes them.
        Assert.Throws<ConverterException>(() => _dateTime.FromWire("20260805T143015", typeof(DateTime)));
        Assert.Throws<ConverterException>(() => _dateOnly.FromWire("20260805", typeof(DateOnly)));
        Assert.Throws<ConverterException>(() => _timeOnly.FromWire("143015", typeof(TimeOnly)));
    }

    [Theory]
    // No writer emits these, but datetime.fromisoformat reads them, and refusing a form
    // Python accepts breaks interchange in the direction that matters.
    [InlineData("2026-08-05", "2026-08-05T00:00:00")]
    [InlineData("2026-08-05 14:30:15", "2026-08-05T14:30:15")]
    [InlineData("2026-08-05T14:30", "2026-08-05T14:30:00")]
    [InlineData("2026-08-05T14:30:15.5", "2026-08-05T14:30:15.500000")]
    public void datetime_reads_every_form_python_reads(string wire, string canonical)
    {
        DateTime value = Assert.IsType<DateTime>(_dateTime.FromWire(wire, typeof(DateTime)));
        Assert.Equal(canonical, _dateTime.ToWire(value));
    }

    [Fact]
    public void datetimeoffset_wires_the_offset_as_hh_mm()
    {
        DateTimeOffset value = new(2026, 8, 5, 14, 30, 15, TimeSpan.FromHours(-5));
        value = value.AddTicks(1_234_560);
        Assert.Equal("2026-08-05T14:30:15.123456-05:00", _dateTimeOffset.ToWire(value));
        Assert.Equal(
            value,
            _dateTimeOffset.FromWire("2026-08-05T14:30:15.123456-05:00", typeof(DateTimeOffset)));
    }

    [Fact]
    public void datetimeoffset_writes_utc_as_plus_zero_not_z()
    {
        // Python's isoformat() never emits 'Z'.
        DateTimeOffset value = new(2026, 8, 5, 14, 30, 15, TimeSpan.Zero);
        Assert.Equal("2026-08-05T14:30:15+00:00", _dateTimeOffset.ToWire(value));
    }

    [Fact]
    public void datetimeoffset_rejects_a_naive_wire_value()
    {
        ConverterException error = Assert.Throws<ConverterException>(
            () => _dateTimeOffset.FromWire("2026-08-05T14:30:15.123456", typeof(DateTimeOffset)));
        Assert.Contains("DateTime", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("08/05/2026 14:30:15 -05:00")]
    [InlineData("Aug 5, 2026 2:30:15 PM -05:00")]
    public void datetimeoffset_rejects_what_python_cannot_read(string wire)
    {
        Assert.Throws<ConverterException>(
            () => _dateTimeOffset.FromWire(wire, typeof(DateTimeOffset)));
    }

    [Theory]
    // Again fromisoformat's set, not isoformat's: the colon-less offset and the Zulu
    // designator are both read by Python and neither is written by anything.
    [InlineData("2026-08-05T14:30:15-0500", "2026-08-05T14:30:15-05:00")]
    [InlineData("2026-08-05T14:30:15Z", "2026-08-05T14:30:15+00:00")]
    [InlineData("2026-08-05 14:30:15+00:00", "2026-08-05T14:30:15+00:00")]
    [InlineData("2026-08-05T14:30-05:00", "2026-08-05T14:30:00-05:00")]
    public void datetimeoffset_reads_every_form_python_reads(string wire, string canonical)
    {
        DateTimeOffset value = Assert.IsType<DateTimeOffset>(
            _dateTimeOffset.FromWire(wire, typeof(DateTimeOffset)));
        Assert.Equal(canonical, _dateTimeOffset.ToWire(value));
    }

    [Fact]
    public void date_and_time_wire_as_iso_8601()
    {
        Assert.Equal("2026-08-05", _dateOnly.ToWire(new DateOnly(2026, 8, 5)));
        Assert.Equal(new DateOnly(2026, 8, 5), _dateOnly.FromWire("2026-08-05", typeof(DateOnly)));

        TimeOnly clock = new(23, 59, 58, 500);
        Assert.Equal("23:59:58.500000", _timeOnly.ToWire(clock));
        Assert.Equal(clock, _timeOnly.FromWire("23:59:58.500000", typeof(TimeOnly)));
        Assert.Equal(new TimeOnly(23, 59, 58), _timeOnly.FromWire("23:59:58", typeof(TimeOnly)));
    }

    [Theory]
    [InlineData("20260805")] // date.fromisoformat takes the compact form; nothing writes it
    [InlineData("2026-8-5")]
    [InlineData("08/05/2026")]
    public void date_rejects_forms_no_writer_produces(string wire)
    {
        Assert.Throws<ConverterException>(() => _dateOnly.FromWire(wire, typeof(DateOnly)));
    }

    [Theory]
    [InlineData("11:59 PM")]
    [InlineData("23:59")]
    public void time_rejects_forms_no_writer_produces(string wire)
    {
        Assert.Throws<ConverterException>(() => _timeOnly.FromWire(wire, typeof(TimeOnly)));
    }

    [Fact]
    public void timespan_wires_as_total_seconds()
    {
        TimeSpan value = new(1, 1, 1, 1, 500);
        Assert.Equal(90061.5, _timeSpan.ToWire(value));
        Assert.Equal(value, _timeSpan.FromWire(90061.5, typeof(TimeSpan)));
    }

    [Fact]
    public void timespan_keeps_sub_millisecond_precision()
    {
        // Python's timedelta resolves to a microsecond, so 0.123456 s must survive the trip;
        // the converter rebuilds from ticks rather than relying on TimeSpan.FromSeconds,
        // whose rounding has changed across .NET versions.
        TimeSpan value = TimeSpan.FromTicks(1_234_560);
        double wire = Assert.IsType<double>(_timeSpan.ToWire(value));
        Assert.Equal(0.123456, wire);
        Assert.Equal(value, _timeSpan.FromWire(wire, typeof(TimeSpan)));
    }

    [Fact]
    public void timespan_accepts_whichever_numeric_type_a_backend_decoded()
    {
        Assert.Equal(TimeSpan.FromSeconds(90), _timeSpan.FromWire(90L, typeof(TimeSpan)));
        Assert.Equal(TimeSpan.FromSeconds(90), _timeSpan.FromWire(90, typeof(TimeSpan)));
        Assert.Throws<ConverterException>(() => _timeSpan.FromWire("90", typeof(TimeSpan)));
    }

    [Fact]
    public void timespan_rejects_non_finite_and_out_of_range_values()
    {
        Assert.Throws<ConverterException>(() => _timeSpan.FromWire(double.NaN, typeof(TimeSpan)));
        Assert.Throws<ConverterException>(
            () => _timeSpan.FromWire(double.PositiveInfinity, typeof(TimeSpan)));
        Assert.Throws<ConverterException>(() => _timeSpan.FromWire(1e30, typeof(TimeSpan)));
    }

    [Fact]
    public void regex_wires_the_pattern_and_drops_the_options()
    {
        Regex value = new(@"^SN-\d{6}$", RegexOptions.IgnoreCase);
        Assert.Equal(@"^SN-\d{6}$", _regex.ToWire(value));

        Regex restored = Assert.IsType<Regex>(_regex.FromWire(@"^SN-\d{6}$", typeof(Regex)));
        Assert.Equal(RegexOptions.None, restored.Options);
        Assert.Matches(value, "sn-123456");
        Assert.DoesNotMatch(restored, "sn-123456");
    }

    [Fact]
    public void regex_rejects_a_pattern_dotnet_cannot_compile()
    {
        Assert.Throws<ConverterException>(() => _regex.FromWire("(unclosed", typeof(Regex)));
    }

    [Fact]
    public void filepath_crosses_verbatim_in_both_directions()
    {
        Assert.Equal("data/run-01.h5", _filePath.ToWire(new FilePath("data/run-01.h5")));
        Assert.Equal(@"C:\Devices\probe.cfg", _filePath.ToWire(new FilePath(@"C:\Devices\probe.cfg")));

        FilePath restored = Assert.IsType<FilePath>(_filePath.FromWire("data/run-01.h5", typeof(FilePath)));
        Assert.Equal(new FilePath("data/run-01.h5"), restored);
    }

    [Fact]
    public void converters_reject_a_value_of_the_wrong_type()
    {
        Assert.Throws<ConverterException>(() => _guid.ToWire("6ba7b810-9dad-11d1-80b4-00c04fd430c8"));
        Assert.Throws<ConverterException>(() => _guid.FromWire(42, typeof(Guid)));
    }

    [Fact]
    public void golden_stdlib_wire_values_match_the_manifest()
    {
        JsonElement wire = ConverterTestGolden.Wire("stdlib");
        JsonElement manifest = ConverterTestGolden.Manifest("stdlib");

        AssertPathRoundTrip(wire, manifest, "filePath");
        AssertStringRoundTrip(_decimal, wire, manifest, "amount", "$decimal", static s => decimal.Parse(s, CultureInfo.InvariantCulture));
        AssertStringRoundTrip(_guid, wire, manifest, "deviceId", "$uuid", Guid.Parse);
        AssertStringRoundTrip(_regex, wire, manifest, "serialPattern", "$pattern", static s => new Regex(s));

        // GRAMMAR §9 leaves PurePosixPath and PureWindowsPath without a C# counterpart, so
        // those two fields of the fixture are read as plain strings by a C# mirror.
        Assert.Equal("/var/log/device.log", wire.GetProperty("posixPath").GetString());
        Assert.Equal(@"C:\Devices\probe.cfg", wire.GetProperty("windowsPath").GetString());
    }

    [Fact]
    public void golden_temporal_wire_values_match_the_manifest()
    {
        JsonElement wire = ConverterTestGolden.Wire("temporal");
        JsonElement manifest = ConverterTestGolden.Manifest("temporal");

        string naive = manifest.GetProperty("naive").GetProperty("$datetime").GetString()!;
        Assert.Equal(naive, wire.GetProperty("naive").GetString());
        Assert.Equal(naive, _dateTime.ToWire(DateTime.Parse(naive, CultureInfo.InvariantCulture)));
        Assert.Equal(
            DateTime.Parse(naive, CultureInfo.InvariantCulture),
            _dateTime.FromWire(naive, typeof(DateTime)));

        string aware = manifest.GetProperty("aware").GetProperty("$datetime").GetString()!;
        Assert.Equal(aware, wire.GetProperty("aware").GetString());
        DateTimeOffset awareValue = Assert.IsType<DateTimeOffset>(
            _dateTimeOffset.FromWire(aware, typeof(DateTimeOffset)));
        Assert.Equal(TimeSpan.FromHours(-5), awareValue.Offset);
        Assert.Equal(aware, _dateTimeOffset.ToWire(awareValue));

        AssertStringRoundTrip(
            _dateOnly, wire, manifest, "day", "$date",
            static s => DateOnly.ParseExact(s, "yyyy-MM-dd", CultureInfo.InvariantCulture));
        AssertStringRoundTrip(
            _timeOnly, wire, manifest, "clock", "$time",
            static s => TimeOnly.Parse(s, CultureInfo.InvariantCulture));

        double elapsed = manifest.GetProperty("elapsed").GetProperty("$timedeltaSeconds").GetDouble();
        Assert.Equal(elapsed, wire.GetProperty("elapsed").GetDouble());
        Assert.Equal(elapsed, _timeSpan.ToWire(_timeSpan.FromWire(elapsed, typeof(TimeSpan))));
    }

    private static void AssertPathRoundTrip(JsonElement wire, JsonElement manifest, string field)
    {
        string expected = manifest.GetProperty(field).GetProperty("$path").GetProperty("value").GetString()!;
        Assert.Equal(expected, wire.GetProperty(field).GetString());
        Assert.Equal(expected, _filePath.ToWire(new FilePath(expected)));
        Assert.Equal(new FilePath(expected), _filePath.FromWire(expected, typeof(FilePath)));
    }

    private static void AssertStringRoundTrip<T>(
        IWireConverter converter,
        JsonElement wire,
        JsonElement manifest,
        string field,
        string manifestTag,
        Func<string, T> parse)
        where T : notnull
    {
        string expected = manifest.GetProperty(field).GetProperty(manifestTag).GetString()!;
        Assert.Equal(expected, wire.GetProperty(field).GetString());
        Assert.Equal(expected, converter.ToWire(parse(expected)));
        Assert.Equal(expected, converter.ToWire(converter.FromWire(expected, typeof(T))));
    }
}
