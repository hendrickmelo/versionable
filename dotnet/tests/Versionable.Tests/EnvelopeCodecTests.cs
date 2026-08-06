using Versionable.Engine;
using Versionable.Errors;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// Both envelope layouts: the wrapped one every writer produces, and the 0.1.x flat dunders that
/// only ever have to be read.
/// </summary>
[Collection(RegistryCollection.Name)]
public class EnvelopeCodecTests
{
    public EnvelopeCodecTests() => GoldenSchemas.EnsureRegistered();

    [Fact]
    public void the_wrapped_layout_is_read_from_the_nested_table()
    {
        Dictionary<string, object?> data = new(StringComparer.Ordinal)
        {
            [VersionableEnvelope.WrappedKey] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["object"] = "GoldenInner",
                ["version"] = 1L,
                ["hash"] = "e37514",
            },
            ["x"] = 1.5,
        };

        EnvelopeMetadata envelope = EnvelopeCodec.Read(data);

        Assert.Equal("GoldenInner", envelope.ObjectName);
        Assert.Equal(1, envelope.Version);
        Assert.Equal("e37514", envelope.Hash);
    }

    [Fact]
    public void the_legacy_flat_layout_is_still_readable()
    {
        // 0.1.x files, read-only back-compat. Nothing writes these keys any more.
        Dictionary<string, object?> data = new(StringComparer.Ordinal)
        {
            [VersionableEnvelope.LegacyObjectKey] = "GoldenInner",
            [VersionableEnvelope.LegacyVersionKey] = 1L,
            [VersionableEnvelope.LegacyHashKey] = "e37514",
            ["x"] = 1.5,
        };

        EnvelopeMetadata envelope = EnvelopeCodec.Read(data);

        Assert.Equal("GoldenInner", envelope.ObjectName);
        Assert.Equal(1, envelope.Version);
        Assert.Equal("e37514", envelope.Hash);
    }

    [Fact]
    public void a_file_with_no_envelope_reads_as_all_absent()
    {
        EnvelopeMetadata envelope = EnvelopeCodec.Read(
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["x"] = 1.5 });

        Assert.Null(envelope.ObjectName);
        Assert.Null(envelope.Version);
        Assert.Null(envelope.Hash);
    }

    [Fact]
    public void stripping_removes_every_reserved_key_from_both_layouts()
    {
        Dictionary<string, object?> data = new(StringComparer.Ordinal)
        {
            [VersionableEnvelope.WrappedKey] = new Dictionary<string, object?>(StringComparer.Ordinal),
            [VersionableEnvelope.LegacyObjectKey] = "GoldenInner",
            [VersionableEnvelope.LegacyVersionKey] = 1L,
            [VersionableEnvelope.LegacyHashKey] = "e37514",
            [VersionableEnvelope.LegacyFormatBigEndianKey] = true,
            [VersionableEnvelope.LegacySharedRefsKey] = new List<object?>(),
            ["x"] = 1.5,
            ["y"] = -2.5,
        };

        Dictionary<string, object?> fields = EnvelopeCodec.Strip(data);

        Assert.Equal(["x", "y"], fields.Keys.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void a_version_recorded_as_text_is_still_a_version()
    {
        // Backends hand back whatever their reader produced, and a hand-edited file may quote it.
        EnvelopeMetadata envelope = EnvelopeCodec.Read(Wrapped(("version", "2")));

        Assert.Equal(2, envelope.Version);
    }

    [Fact]
    public void a_version_that_is_not_a_number_is_reported()
    {
        BackendException error = Assert.Throws<BackendException>(
            () => EnvelopeCodec.Read(Wrapped(("version", "tuesday"))));

        Assert.Contains("not an integer", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void envelope_words_are_only_envelope_words_inside_the_wrapped_table()
    {
        // `object`, `version`, and `hash` are ordinary field names, and no writer has ever put them
        // at the top level. Reading them there would misread a type that happens to declare one.
        Dictionary<string, object?> data = new(StringComparer.Ordinal)
        {
            ["object"] = "not-an-envelope",
            ["version"] = 7L,
            ["hash"] = "abcdef",
            ["format"] = "csv",
        };

        EnvelopeMetadata envelope = EnvelopeCodec.Read(data);

        Assert.Null(envelope.ObjectName);
        Assert.Null(envelope.Version);
        Assert.Null(envelope.Hash);
        Assert.Equal(
            ["format", "hash", "object", "version"],
            EnvelopeCodec.Strip(data).Keys.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void a_file_from_a_future_wire_revision_is_refused()
    {
        BackendException error = Assert.Throws<BackendException>(
            () => EnvelopeCodec.Read(Wrapped(("format", "2"))));

        Assert.Contains("versionable format", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_legacy_format_marker_is_refused_wherever_it_sits()
    {
        // __FORMAT__ is a reserved key, so unlike `format` it can never be a field name.
        BackendException error = Assert.Throws<BackendException>(() => EnvelopeCodec.Read(
            new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                [VersionableEnvelope.LegacyFormatKey] = "2",
            }));

        Assert.Contains("versionable format", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void has_envelope_sees_either_layout()
    {
        Assert.True(EnvelopeCodec.HasEnvelope(new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            [VersionableEnvelope.WrappedKey] = new Dictionary<string, object?>(StringComparer.Ordinal),
        }));
        Assert.True(EnvelopeCodec.HasEnvelope(new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            [VersionableEnvelope.LegacyObjectKey] = "GoldenInner",
        }));
        Assert.False(EnvelopeCodec.HasEnvelope(
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["x"] = 1.5 }));
    }

    [Fact]
    public void a_nested_object_in_the_legacy_layout_still_materializes()
    {
        Dictionary<string, object?> legacy = new(StringComparer.Ordinal)
        {
            [VersionableEnvelope.LegacyObjectKey] = "EngineLeaf",
            [VersionableEnvelope.LegacyVersionKey] = 1L,
            [VersionableEnvelope.LegacyHashKey] = "aaaaaa",
            ["name"] = "from-0.1.x",
            ["weight"] = 2.0,
        };

        EngineLeaf leaf = (EngineLeaf)WireValues.ReadVersionable(legacy, EngineLeaf.Metadata);

        Assert.Equal("from-0.1.x", leaf.Name);
        Assert.Equal(2.0, leaf.Weight);
    }

    [Fact]
    public void a_nested_object_resolves_its_concrete_type_from_its_own_envelope()
    {
        object shape = WireValues.ReadVersionable(
            NestedShape("GoldenCircle", "8e5e7c", "radius", 2.5), GoldenShape.VersionableMetadata);

        GoldenCircle circle = Assert.IsType<GoldenCircle>(shape);
        Assert.Equal("c1", circle.Label);
        Assert.Equal(2.5, circle.Radius);
    }

    [Fact]
    public void a_nested_object_naming_an_unregistered_type_is_refused()
    {
        BackendException error = Assert.Throws<BackendException>(() => WireValues.ReadVersionable(
            NestedShape("GoldenTriangle", "000000", "sides", 3.0), GoldenShape.VersionableMetadata));

        Assert.Contains("Unknown nested object type 'GoldenTriangle'", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_nested_object_naming_an_unrelated_type_is_refused()
    {
        BackendException error = Assert.Throws<BackendException>(() => WireValues.ReadVersionable(
            NestedShape("GoldenInner", "e37514", "x", 1.0), GoldenShape.VersionableMetadata));

        Assert.Contains("not assignable", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void a_nested_object_with_no_version_falls_back_to_the_current_one_and_warns()
    {
        List<string> warnings = [];
        void Capture(string message) => warnings.Add(message);
        VersionableLog.Warning += Capture;
        try
        {
            Dictionary<string, object?> data = new(StringComparer.Ordinal)
            {
                ["name"] = "hand-written",
                ["weight"] = 1.0,
            };

            EngineLeaf leaf = (EngineLeaf)WireValues.ReadVersionable(data, EngineLeaf.Metadata);

            Assert.Equal("hand-written", leaf.Name);
        }
        finally
        {
            VersionableLog.Warning -= Capture;
        }

        Assert.Contains(warnings, message => message.Contains("no version in the envelope", StringComparison.Ordinal));
    }

    private static Dictionary<string, object?> Wrapped(params (string Key, object? Value)[] entries)
    {
        Dictionary<string, object?> table = new(StringComparer.Ordinal);
        foreach ((string key, object? value) in entries)
        {
            table[key] = value;
        }

        return new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            [VersionableEnvelope.WrappedKey] = table,
        };
    }

    private static Dictionary<string, object?> NestedShape(string name, string hash, string field, double value) =>
        new(StringComparer.Ordinal)
        {
            [VersionableEnvelope.WrappedKey] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["object"] = name,
                ["version"] = 1L,
                ["hash"] = hash,
            },
            ["label"] = "c1",
            [field] = value,
        };
}
