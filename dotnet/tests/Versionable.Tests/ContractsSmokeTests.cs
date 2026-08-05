using System.Reflection;
using Versionable.Backends;
using Versionable.Converters;
using Versionable.Errors;
using Xunit;

namespace Versionable.Tests;

/// <summary>
/// Proves the scaffold builds and the core contracts compose. Behavior lands with the
/// implementation tasks; these assert only on the pinned surface.
/// </summary>
/// <remarks>
/// The three registries are process-wide statics, so each test starts by resetting them
/// through the internal test seam. xUnit runs the tests of one class serially, which is what
/// makes that safe; a second test class touching the registries would need its own xUnit
/// collection to stay off this one's toes.
/// </remarks>
public class ContractsSmokeTests
{
    public ContractsSmokeTests()
    {
        VersionableRegistry.Reset();
        BackendRegistry.Reset();
        ConverterRegistry.Reset();
    }

    [Fact]
    public void envelope_keys_match_the_python_wire_format()
    {
        // Source of truth: _serializeVersionable and _ENVELOPE_KEYS in src/versionable/_types.py.
        Assert.Equal("__versionable__", VersionableEnvelope.WrappedKey);
        Assert.Equal("object", VersionableEnvelope.ObjectKey);
        Assert.Equal("version", VersionableEnvelope.VersionKey);
        Assert.Equal("hash", VersionableEnvelope.HashKey);

        // Exactly the members of _ENVELOPE_KEYS: the wrapped key plus the 0.1.x flat dunders.
        Assert.Equal(
            [
                "__FORMAT_BE__",
                "__FORMAT__",
                "__HASH__",
                "__OBJECT__",
                "__SHARED_REFS__",
                "__VERSION__",
                "__versionable__",
            ],
            VersionableEnvelope.ReservedKeys.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void every_exception_roots_at_versionable_exception()
    {
        Assert.IsAssignableFrom<VersionableException>(new BackendException("boom"));
        Assert.IsAssignableFrom<VersionableException>(new MigrationException("boom"));
        Assert.IsAssignableFrom<VersionableException>(new UnknownFieldException("boom"));
        Assert.IsAssignableFrom<VersionableException>(new UpgradeRequiredException("boom"));
        Assert.IsAssignableFrom<VersionableException>(new UnsupportedTypeException("boom"));
        Assert.IsAssignableFrom<VersionableException>(new VersionException("boom"));
        Assert.IsAssignableFrom<VersionableException>(new ArrayNotLoadedException("boom"));

        // DtypeMismatchError derives from ConverterError in src/versionable/errors.py.
        Assert.IsAssignableFrom<ConverterException>(
            new DtypeMismatchException("float32", "float64", "samples", DtypeContext.Load));
    }

    [Fact]
    public void hash_mismatch_message_names_the_computed_hash()
    {
        HashMismatchException error = new("Config", "74a182", "a1b2c3");

        Assert.Equal("74a182", error.Declared);
        Assert.Equal("a1b2c3", error.Computed);
        Assert.Contains("a1b2c3", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void backend_registry_resolves_by_extension_and_reports_unknown_ones()
    {
        BackendRegistry.Register([".smoke"], () => new StubBackend());

        Assert.IsType<StubBackend>(BackendRegistry.Resolve("/tmp/config.SMOKE"));
        Assert.Contains(".smoke", BackendRegistry.RegisteredExtensions());

        BackendException error = Assert.Throws<BackendException>(() => BackendRegistry.Resolve("/tmp/config.nope"));
        Assert.Contains(".smoke", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void an_explicit_backend_wins_over_extension_detection()
    {
        // Mirrors Python's getBackend(explicit=) / save(backend=): the caller can write a
        // format the path's extension does not imply, or one that was never registered.
        StubBackend chosen = new();

        Assert.Same(chosen, BackendRegistry.Resolve("/tmp/config.unregistered", chosen));
    }

    [Fact]
    public void converter_registry_prefers_an_exact_match_over_a_subclass_match()
    {
        ConverterRegistry.Register(new StubConverter(typeof(Stream), "stream", matchSubclasses: true));
        ConverterRegistry.Register(new StubConverter(typeof(MemoryStream), "memoryStream", matchSubclasses: false));

        Assert.True(ConverterRegistry.TryResolve(typeof(MemoryStream), out IWireConverter? exact));
        Assert.Equal("memoryStream", exact.SerializationName);

        Assert.True(ConverterRegistry.TryResolve(typeof(FileStream), out IWireConverter? bySubclass));
        Assert.Equal("stream", bySubclass.SerializationName);

        Assert.False(ConverterRegistry.TryResolve(typeof(Uri), out IWireConverter? missing));
        Assert.Null(missing);
    }

    [Fact]
    public void registry_rejects_two_types_claiming_one_serialization_name()
    {
        VersionableRegistry.Register(Describe(typeof(FirstClaimant), "Contested"));

        VersionableException error = Assert.Throws<VersionableException>(
            () => VersionableRegistry.Register(Describe(typeof(SecondClaimant), "Contested")));
        Assert.Contains("Contested", error.Message, StringComparison.Ordinal);

        Assert.True(VersionableRegistry.TryGetByName("Contested", out VersionableMetadata? found));
        Assert.Equal(typeof(FirstClaimant), found.ClrType);
    }

    [Fact]
    public void a_collision_on_an_old_name_registers_nothing_at_all()
    {
        // Validate-before-mutate: Python reads every name before touching _REGISTRY
        // (src/versionable/_base.py), so a collision on the *second* old name must not leave
        // the primary name — or the CLR type — half-registered.
        VersionableRegistry.Register(Describe(typeof(FirstClaimant), "Taken"));

        Assert.Throws<VersionableException>(
            () => VersionableRegistry.Register(
                Describe(typeof(SecondClaimant), "Fresh", oldNames: ["AlsoFresh", "Taken"])));

        Assert.False(VersionableRegistry.TryGetByName("Fresh", out _));
        Assert.False(VersionableRegistry.TryGetByName("AlsoFresh", out _));
        Assert.False(VersionableRegistry.TryGetByType(typeof(SecondClaimant), out _));
    }

    [Fact]
    public void register_type_only_keeps_a_type_serializable_without_claiming_a_name()
    {
        // The Register=false path. Python keeps _serializer_meta_ on the class while leaving
        // it out of _REGISTRY, so the type still serializes but never resolves from an
        // envelope name.
        VersionableRegistry.RegisterTypeOnly(Describe(typeof(FirstClaimant), "Unclaimed"));

        Assert.True(VersionableRegistry.TryGetByType(typeof(FirstClaimant), out VersionableMetadata? byType));
        Assert.Equal("Unclaimed", byType.Name);
        Assert.False(VersionableRegistry.TryGetByName("Unclaimed", out _));

        // The name stays free for a type that does want it.
        VersionableRegistry.Register(Describe(typeof(SecondClaimant), "Unclaimed"));
        Assert.True(VersionableRegistry.TryGetByName("Unclaimed", out VersionableMetadata? byName));
        Assert.Equal(typeof(SecondClaimant), byName.ClrType);
    }

    [Fact]
    public void the_registry_hands_back_the_very_instance_it_was_given()
    {
        // Identity contract: generated [ModuleInitializer] code passes T.VersionableMetadata
        // straight to Register, and the registry stores that instance rather than copying it.
        // So ReferenceEquals(T.VersionableMetadata, registryResult) holds, and engine code can
        // cache either one. Asserting the registry half here — the T.VersionableMetadata half
        // needs the generator, which lands in 2b.
        VersionableMetadata registered = Describe(typeof(SmokeConfig), "SmokeConfig", oldNames: ["OldSmoke"]);
        VersionableRegistry.Register(registered);

        Assert.True(VersionableRegistry.TryGetByType(typeof(SmokeConfig), out VersionableMetadata? byType));
        Assert.True(VersionableRegistry.TryGetByName("SmokeConfig", out VersionableMetadata? byName));
        Assert.True(VersionableRegistry.TryGetByName("OldSmoke", out VersionableMetadata? byOldName));

        Assert.Same(registered, byType);
        Assert.Same(registered, byName);
        Assert.Same(registered, byOldName);
        Assert.Same(registered, VersionableRegistry.RegisteredTypes()["SmokeConfig"]);
    }

    [Fact]
    public void metadata_carries_generated_accessors_for_a_versionable_type()
    {
        VersionableMetadata metadata = Describe(typeof(SmokeConfig), "SmokeConfig");
        SmokeConfig instance = (SmokeConfig)metadata.Factory(["prod", "fast"]);

        Assert.Equal("prod", instance.Name);
        Assert.Equal("prod", metadata.Fields[0].Getter(instance));
        Assert.Equal("str", metadata.Fields[0].CanonicalType);
        Assert.Equal(1, metadata.Version);
        Assert.Equal("74a182", metadata.Hash);
        Assert.Equal(UnknownFieldPolicy.Ignore, metadata.Unknown);
    }

    [Fact]
    public void a_literal_field_carries_its_options_in_canonical_order()
    {
        // Options are order-significant per conformance/GRAMMAR.md §8 — Literal['fast', 'slow']
        // and Literal['slow', 'fast'] are different schemas. The descriptor must preserve the
        // declared order, not sort it.
        FieldDescriptor mode = Describe(typeof(SmokeConfig), "SmokeConfig").Fields[1];

        Assert.Equal(["fast", "slow"], mode.LiteralOptions);
        Assert.Equal("Literal['fast', 'slow']", mode.CanonicalType);
        Assert.True(mode.HasLiteralFallback);
        Assert.Equal("fast", mode.LiteralFallback);
    }

    [Fact]
    public void the_wire_reader_seam_materializes_without_reflection()
    {
        // The generator emits a closed delegate per field that needs construction, so the
        // engine never calls Activator.CreateInstance over a user type (IsAotCompatible).
        FieldDescriptor tags = new()
        {
            WireName = "tags",
            ClrName = "Tags",
            ClrType = typeof(List<string>),
            CanonicalType = "list[str]",
            Getter = instance => instance,
            WireReader = wire => new List<string>(((IEnumerable<object?>)wire!).Select(v => (string)v!)),
        };

        object? materialized = tags.WireReader!(new object?[] { "a", "b" });

        Assert.Equal(new List<string> { "a", "b" }, materialized);
    }

    [Fact]
    public void the_source_generator_runs_in_this_compilation()
    {
        // Proves the ADR-0003 pipeline is wired as a real analyzer reference, not merely
        // referenced as an assembly: this attribute exists only because the generator emitted
        // it into this compilation.
        AssemblyMetadataAttribute? marker = typeof(ContractsSmokeTests).Assembly
            .GetCustomAttributes<AssemblyMetadataAttribute>()
            .SingleOrDefault(attribute => attribute.Key == "Versionable.Generator");

        Assert.NotNull(marker);
        Assert.Equal("VersionableMetadataGenerator", marker.Value);
    }

    private static VersionableMetadata Describe(Type clrType, string name, string[]? oldNames = null) => new()
    {
        ClrType = clrType,
        Name = name,
        Version = 1,
        Hash = "74a182",
        Fields =
        [
            new FieldDescriptor
            {
                WireName = "name",
                ClrName = "Name",
                ClrType = typeof(string),
                CanonicalType = "str",
                Getter = instance => ((SmokeConfig)instance).Name,
            },
            new FieldDescriptor
            {
                WireName = "mode",
                ClrName = "Mode",
                ClrType = typeof(string),
                CanonicalType = "Literal['fast', 'slow']",
                Getter = instance => ((SmokeConfig)instance).Mode,
                LiteralOptions = ["fast", "slow"],
                HasLiteralFallback = true,
                LiteralFallback = "fast",
            },
        ],
        Factory = values => new SmokeConfig((string)values[0]!, (string)values[1]!),
        OldNames = oldNames,
    };

    private sealed class StubBackend : IVersionableBackend
    {
        public IReadOnlySet<Type> NativeTypes { get; } = new HashSet<Type>();

        public void Save(
            IReadOnlyDictionary<string, object?> fields,
            EnvelopeMetadata envelope,
            string path,
            VersionableMetadata metadata,
            BackendSaveOptions options) => throw new NotSupportedException();

        public BackendLoadResult Load(string path, BackendLoadOptions options) => throw new NotSupportedException();
    }

    private sealed class StubConverter(Type clrType, string serializationName, bool matchSubclasses) : IWireConverter
    {
        public Type ClrType { get; } = clrType;

        public string SerializationName { get; } = serializationName;

        public bool MatchSubclasses { get; } = matchSubclasses;

        public object? ToWire(object value) => throw new NotSupportedException();

        public object FromWire(object wireValue, Type targetType) => throw new NotSupportedException();
    }

    private sealed class FirstClaimant;

    private sealed class SecondClaimant;
}

/// <summary>
/// Minimal declaration proving the attribute surface binds and composes. Hash is a
/// placeholder — nothing computes hashes until task 2b.
/// </summary>
/// <remarks>
/// <c>partial</c> because the generator implements
/// <see cref="IVersionableMetadataProvider"/> by emitting another part of the type, and a
/// static abstract member cannot be implemented from outside its declaring type. Every
/// <c>[Versionable]</c> type must be declared this way.
/// </remarks>
[Versionable(Version = 1, Hash = "74a182", Unknown = UnknownFieldPolicy.Error)]
[SerializationName("SmokeConfig")]
public sealed partial class SmokeConfig(string name, string mode)
{
    /// <summary>Wire name is the property name unless overridden.</summary>
    [VersionableField("name")]
    public string Name { get; } = name;

    /// <summary>Argument order is the canonical order and is hash-significant.</summary>
    [LiteralValues("fast", "slow", Fallback = "fast")]
    public string Mode { get; } = mode;
}

/// <summary>Proves an enum can carry a Serialization Name, wire values, and a fallback.</summary>
[SerializationName("SmokeStatus")]
public enum SmokeStatus
{
    /// <summary>Serializes as the string <c>active</c>, not as <c>0</c>.</summary>
    [EnumValue("active")]
    Active = 0,

    /// <summary>Substituted for any value this enum does not define.</summary>
    [EnumValue("unknown")]
    [EnumFallback]
    Unknown = 1,
}
