using System.Text.Json;
using Versionable.Converters;
using Versionable.Engine;
using Versionable.Errors;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// Enum wire values: bare numbers, <see cref="EnumValueAttribute"/> strings, and the
/// <see cref="EnumFallbackAttribute"/> member.
/// </summary>
public class ConverterEnumTests
{
    /// <summary>Mirrors the golden corpus's <c>GoldenColour</c>, a Python string-valued enum.</summary>
    public enum Colour
    {
        [EnumValue("red")]
        Red = 0,

        [EnumValue("green")]
        Green = 1,

        [EnumValue("blue")]
        Blue = 2,
    }

    /// <summary>Mirrors the golden corpus's <c>GoldenPriority</c>, a Python int-valued enum.</summary>
    public enum Priority
    {
        Low = 1,
        Medium = 2,
        High = 3,
    }

    /// <summary>A string-valued enum with a fallback, the analogue of VERSIONABLE_FALLBACK.</summary>
    public enum Status
    {
        [EnumValue("active")]
        Active = 0,

        [EnumValue("unknown")]
        [EnumFallback]
        Unknown = 1,
    }

    /// <summary>An enum with a fallback but no explicit wire strings.</summary>
    public enum Mode
    {
        Fast = 1,

        [EnumFallback]
        Unset = 0,
    }

    /// <summary>A byte-backed enum: the wire number must be the underlying value, not an int.</summary>
    public enum Channel : byte
    {
        A = 7,
    }

    [Fact]
    public void string_valued_members_wire_as_their_enum_value_string()
    {
        Assert.Equal("green", EnumConverter.ToWire(Colour.Green));
        Assert.Equal(Colour.Green, EnumConverter.FromWire("green", typeof(Colour)));
    }

    [Fact]
    public void plain_members_wire_as_their_number()
    {
        Assert.Equal(3, EnumConverter.ToWire(Priority.High));
        Assert.Equal(Priority.High, EnumConverter.FromWire(3, typeof(Priority)));
    }

    [Fact]
    public void the_wire_number_has_the_enums_underlying_type()
    {
        object wire = EnumConverter.ToWire(Channel.A);
        Assert.IsType<byte>(wire);
        Assert.Equal((byte)7, wire);
    }

    [Fact]
    public void a_number_from_any_backend_resolves()
    {
        // Backends decode numbers to whichever CLR type their parser picks.
        Assert.Equal(Priority.Medium, EnumConverter.FromWire(2L, typeof(Priority)));
        Assert.Equal(Priority.Medium, EnumConverter.FromWire(2.0, typeof(Priority)));
    }

    [Fact]
    public void an_undefined_number_is_not_silently_admitted()
    {
        // Python's Enum(value) raises for a value with no member; only a fallback saves it.
        Assert.Throws<ConverterException>(() => EnumConverter.FromWire(99, typeof(Priority)));
    }

    [Fact]
    public void an_unknown_value_falls_back_when_a_member_is_marked()
    {
        Assert.Equal(Status.Unknown, EnumConverter.FromWire("retired", typeof(Status)));
        Assert.Equal(Mode.Unset, EnumConverter.FromWire(42, typeof(Mode)));
    }

    [Fact]
    public void an_unknown_value_falling_back_raises_a_versionable_log_warning()
    {
        List<string> warnings = [];
        void Capture(string message) => warnings.Add(message);
        VersionableLog.Warning += Capture;
        try
        {
            EnumConverter.FromWire("retired", typeof(Status));
        }
        finally
        {
            VersionableLog.Warning -= Capture;
        }

        Assert.Contains(warnings, message => message.Contains("Unknown", StringComparison.Ordinal)
            && message.Contains("fallback", StringComparison.Ordinal));
    }

    [Fact]
    public void an_unknown_value_raises_when_no_member_is_marked()
    {
        ConverterException error = Assert.Throws<ConverterException>(
            () => EnumConverter.FromWire("teal", typeof(Colour)));
        Assert.Contains("EnumFallback", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_member_name_is_not_a_wire_value()
    {
        // Python matches on member value, never on member name; "Green" is neither.
        Assert.Throws<ConverterException>(() => EnumConverter.FromWire("Green", typeof(Colour)));
    }

    [Fact]
    public void non_enum_target_types_are_rejected()
    {
        Assert.Throws<ConverterException>(() => EnumConverter.FromWire(1, typeof(int)));
    }

    [Fact]
    public void golden_enum_wire_values_match_the_manifest()
    {
        JsonElement wire = ConverterTestGolden.Wire("enums");
        JsonElement manifest = ConverterTestGolden.Manifest("enums");

        // The manifest records the member name and its value; the wire file holds the value.
        Assert.Equal("GREEN", manifest.GetProperty("colour").GetProperty("$enum").GetProperty("member").GetString());
        string colour = manifest.GetProperty("colour").GetProperty("$enum").GetProperty("value").GetString()!;
        Assert.Equal(colour, wire.GetProperty("colour").GetString());
        Assert.Equal(colour, EnumConverter.ToWire(Colour.Green));
        Assert.Equal(Colour.Green, EnumConverter.FromWire(colour, typeof(Colour)));

        int priority = manifest.GetProperty("priority").GetProperty("$enum").GetProperty("value").GetInt32();
        Assert.Equal(priority, wire.GetProperty("priority").GetInt32());
        Assert.Equal(priority, EnumConverter.ToWire(Priority.High));
        Assert.Equal(Priority.High, EnumConverter.FromWire(priority, typeof(Priority)));

        string[] palette = [.. wire.GetProperty("palette").EnumerateArray().Select(e => e.GetString()!)];
        Assert.Equal(["red", "blue"], palette);
        Assert.Equal(
            [Colour.Red, Colour.Blue],
            palette.Select(v => (Colour)EnumConverter.FromWire(v, typeof(Colour))));
    }
}
