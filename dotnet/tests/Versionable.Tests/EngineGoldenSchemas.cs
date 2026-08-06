using System.Collections.Frozen;
using System.Numerics;
using System.Numerics.Tensors;
using System.Text.RegularExpressions;
using Versionable.Backends.Hdf5;
using Versionable.Backends.Json;
using Versionable.Backends.Toml;
using Versionable.Backends.Yaml;
using Versionable.Converters;
using Versionable.Engine;
using Versionable.Errors;
using Versionable.Migrations;

namespace Versionable.Tests;

/// <summary>
/// The C# half of the golden-corpus contract: real <c>[Versionable]</c> types, compiled by the
/// source generator in this very assembly.
/// </summary>
/// <remarks>
/// Python counterpart: <c>conformance/golden_schemas.py</c>. Serialization Names, versions, and
/// schema hashes are hardcoded string literals on both sides on purpose — a rendering drift then
/// fails at compile time here and at class-definition time there, rather than quietly producing a
/// file the other language cannot read. The analyzer recomputes every <c>Hash</c> below from the
/// declared members on every build, so these literals are proof rather than documentation.
/// <para>
/// <b>Three fixtures cannot be mirrored exactly, and each says so at its declaration.</b> Python
/// constructs with no C# spelling do not stop a file loading — the wire form is unaffected — but
/// they do change what the C# schema hashes to:
/// </para>
/// <list type="bullet">
///   <item>
///     <description>
///     <see cref="GoldenContainers"/> — Python's <c>tuple[float, ...]</c> has no C# equivalent
///     (GRAMMAR §5), so <c>samples</c> is a <c>double[]</c> rendering <c>list[float]</c>. Loads
///     identically; hashes differently.
///     </description>
///   </item>
///   <item>
///     <description>
///     <see cref="GoldenStdlib"/> — C# v1 has no <c>PurePosixPath</c> / <c>PureWindowsPath</c>
///     (GRAMMAR §9), so all three path fields are <see cref="FilePath"/>, rendering <c>Path</c>.
///     Loads identically; hashes differently.
///     </description>
///   </item>
///   <item>
///     <description>
///     <see cref="GoldenOptionals"/> — a union of two non-<see langword="null"/> members
///     (<c>int | str</c>) has no C# type at all, which is why four conformance vectors carry
///     <c>csharpDeclarable: false</c>. This one keeps hand-written metadata, and so also covers
///     the manual-construction path of the <see cref="VersionableMetadata"/> contract.
///     </description>
///   </item>
/// </list>
/// <para>
/// <see cref="JsonGoldenCorpusTests"/> excludes the first two from its hash-mirror theory, naming
/// the reason, rather than asserting a hash C# cannot produce.
/// </para>
/// </remarks>
internal static class GoldenSchemas
{
    private static readonly object _registrationLock = new();

    /// <summary>Every fixture type's metadata, in the order a fresh process would register it.</summary>
    internal static IReadOnlyList<VersionableMetadata> All { get; } =
    [
        GoldenContainers.VersionableMetadata,
        GoldenEnums.VersionableMetadata,
        GoldenLiterals.VersionableMetadata,
        GoldenInner.VersionableMetadata,
        GoldenNested.VersionableMetadata,
        GoldenShape.VersionableMetadata,
        GoldenCircle.VersionableMetadata,
        GoldenSquare.VersionableMetadata,
        GoldenPolymorphic.VersionableMetadata,
        GoldenWorker.VersionableMetadata,
        GoldenScalars.VersionableMetadata,
        GoldenTemporal.VersionableMetadata,
        GoldenStdlib.VersionableMetadata,
        GoldenArrays.VersionableMetadata,
        SmokeConfig.VersionableMetadata,
        GoldenOptionals.Metadata,
        EngineLeaf.Metadata,
        EngineNode.Metadata,
        EngineChain.Metadata,
        EngineLiteralHolder.Metadata,
        EngineUnionHolder.Metadata,
        EngineSettings.Metadata,
        EngineVersioned.Metadata,
        EngineEnumHolder.Metadata,
        EngineNumbers.Metadata,
        EngineLiteralList.Metadata,
        EngineTupleHolder.Metadata,
        EngineLazyHolder.Metadata,
        EngineStrictArray.Metadata,
        EngineNestedLazy.Metadata,
        EngineNestedStrict.Metadata,
        EngineDeepLazy.Metadata,
        EngineGroupedLazy.Metadata,
        EngineGroupedStrict.Metadata,
        EngineLayeredLazy.Metadata,
    ];

    /// <summary>
    /// Puts every fixture, converter, and backend back in the process-wide registries.
    /// </summary>
    /// <remarks>
    /// All three registries are static with a <c>Reset</c> test seam, and the registrations they
    /// normally hold come from <c>[ModuleInitializer]</c> methods — which run once per process and
    /// can never be re-run. So a reset is only survivable if something can put the entries back by
    /// hand, and this is that something: the generated metadata of every <c>[Versionable]</c> type
    /// in this assembly, plus every module initializer <c>Versionable.dll</c> itself carries.
    /// <para>
    /// <b>All four backends, not just JSON.</b> The registry is one process-wide table, so a test
    /// class that only re-registered the backend it happens to use left the other three missing for
    /// whatever ran next in the same collection — which made the suite's result depend on class
    /// order. Restoring the same set a fresh process would hold is what makes the collection
    /// order-independent, and it is why no test class registers a backend of its own.
    /// </para>
    /// <para>
    /// Called from the constructor of every test class in the registry collection, and from
    /// <see cref="ContractsSmokeTests.Dispose"/> — the one place that resets. Registering the same
    /// metadata twice is a no-op; registering a <em>different</em> type under a name already
    /// claimed is the collision the registry exists to catch.
    /// </para>
    /// </remarks>
    internal static void EnsureRegistered()
    {
        lock (_registrationLock)
        {
            foreach (VersionableMetadata metadata in All)
            {
                VersionableRegistry.Register(metadata);
            }

            // The polymorphism and SkipDefaults fixtures live in their own file but share this
            // process-wide registry, so a reset has to put them back too.
            foreach (VersionableMetadata metadata in ParitySchemas.All)
            {
                VersionableRegistry.Register(metadata);
            }

            foreach (VersionableMetadata metadata in ParitySchemas.Unregistered)
            {
                VersionableRegistry.RegisterTypeOnly(metadata);
            }

            BuiltinConverters.RegisterAll();
            JsonBackend.RegisterExtensions();
            YamlBackend.RegisterExtensions();
            TomlBackend.RegisterExtensions();
            Hdf5Backend.RegisterExtensions();
        }
    }
}

/// <summary>
/// A string-valued enum: the member names differ from what goes on the wire.
/// </summary>
/// <remarks>
/// Python counterpart: <c>class GoldenColour(Enum): RED = "red"</c>. The engine reads the
/// attributes through <c>EnumConverter</c>, so nothing about this enum needs a hand-written
/// mapping — which is the point: an <c>[EnumValue]</c> enum reaching the engine unaided has to
/// write <c>"green"</c>, not an ordinal.
/// </remarks>
internal enum GoldenColour
{
    /// <summary>Wires as <c>red</c>.</summary>
    [EnumValue("red")]
    Red,

    /// <summary>Wires as <c>green</c>.</summary>
    [EnumValue("green")]
    Green,

    /// <summary>Wires as <c>blue</c>.</summary>
    [EnumValue("blue")]
    Blue,
}

/// <summary>An integer-valued enum, which needs no wire mapping at all.</summary>
internal enum GoldenPriority
{
    /// <summary>Wires as <c>1</c>.</summary>
    Low = 1,

    /// <summary>Wires as <c>2</c>.</summary>
    Medium = 2,

    /// <summary>Wires as <c>3</c>.</summary>
    High = 3,
}

/// <summary>
/// Every container form the grammar defines, including a non-string dictionary key.
/// </summary>
/// <remarks>
/// <b>Not a hash mirror.</b> Python declares <c>samples: tuple[float, ...]</c> and hashes
/// <c>846dc8</c>. A homogeneous variadic tuple has no C# spelling (GRAMMAR §5 lists it as
/// "no equivalent"), so <see cref="Samples"/> is a <c>double[]</c> and renders <c>list[float]</c>.
/// Both spellings read the same JSON array, so the fixture loads and every value compares equal;
/// only the canonical rendering — and therefore the hash below — differs.
/// </remarks>
[Versionable(Version = 1, Hash = "51917e")]
internal sealed partial class GoldenContainers
{
    /// <summary>A list of strings.</summary>
    [VersionableField("names")]
    public required List<string> Names { get; init; }

    /// <summary>A list of floats.</summary>
    [VersionableField("readings")]
    public required List<double> Readings { get; init; }

    /// <summary>A list of integers.</summary>
    [VersionableField("counts")]
    public required List<int> Counts { get; init; }

    /// <summary>A list of booleans.</summary>
    [VersionableField("flags")]
    public required List<bool> Flags { get; init; }

    /// <summary>A string-keyed dictionary.</summary>
    [VersionableField("lookup")]
    public required Dictionary<string, int> Lookup { get; init; }

    /// <summary>An integer-keyed dictionary, which the wire stringifies.</summary>
    [VersionableField("byIndex")]
    public required Dictionary<int, string> ByIndex { get; init; }

    /// <summary>A set.</summary>
    [VersionableField("tags")]
    public required HashSet<string> Tags { get; init; }

    /// <summary>A frozen set.</summary>
    [VersionableField("ids")]
    public required FrozenSet<int> Ids { get; init; }

    /// <summary>A fixed, homogeneous tuple.</summary>
    [VersionableField("pair")]
    public required (int First, int Second) Pair { get; init; }

    /// <summary>Python's variadic tuple, which C# can only hold as a list.</summary>
    [VersionableField("samples")]
    public required double[] Samples { get; init; }

    /// <summary>A list of lists.</summary>
    [VersionableField("matrix")]
    public required List<List<double>> Matrix { get; init; }

    /// <summary>A dictionary of lists.</summary>
    [VersionableField("grouped")]
    public required Dictionary<string, List<double>> Grouped { get; init; }
}

/// <summary>Enums standalone and inside containers.</summary>
[Versionable(Version = 1, Hash = "660511")]
internal sealed partial class GoldenEnums
{
    /// <summary>A string-valued enum.</summary>
    [VersionableField("colour")]
    public required GoldenColour Colour { get; init; }

    /// <summary>An integer-valued enum.</summary>
    [VersionableField("priority")]
    public required GoldenPriority Priority { get; init; }

    /// <summary>Enums in a list.</summary>
    [VersionableField("palette")]
    public required List<GoldenColour> Palette { get; init; }

    /// <summary>Enums in a dictionary.</summary>
    [VersionableField("byName")]
    public required Dictionary<string, GoldenColour> ByName { get; init; }
}

/// <summary>Literal member kinds: string, integer, boolean, and a mixed set.</summary>
[Versionable(Version = 1, Hash = "2085b2")]
internal sealed partial class GoldenLiterals
{
    /// <summary>A string literal.</summary>
    [VersionableField("mode")]
    [LiteralValues("fast", "slow")]
    public required string Mode { get; init; }

    /// <summary>An integer literal.</summary>
    [VersionableField("level")]
    [LiteralValues(1, 2, 3)]
    public required int Level { get; init; }

    /// <summary>A boolean literal.</summary>
    [VersionableField("flag")]
    [LiteralValues(true, false)]
    public required bool Flag { get; init; }

    /// <summary>
    /// A mixed literal, whose chosen value in the corpus is the integer 1.
    /// </summary>
    /// <remarks>
    /// Typed <see cref="object"/> because no single C# type spans the declared members. That is
    /// the one place a bare <see cref="object"/> is accepted: a literal field renders from its
    /// options, never from its declared type, so it never reaches the rejection VSN0002 applies
    /// to <see cref="object"/> elsewhere.
    /// </remarks>
    [VersionableField("tag")]
    [LiteralValues("auto", 0, "off", 1)]
    public required object Tag { get; init; }
}

/// <summary>The leaf object of the nested fixture.</summary>
[Versionable(Version = 1, Hash = "e37514")]
internal sealed partial class GoldenInner
{
    /// <summary>X coordinate.</summary>
    [VersionableField("x")]
    public required double X { get; init; }

    /// <summary>Y coordinate.</summary>
    [VersionableField("y")]
    public required double Y { get; init; }
}

/// <summary>A nested object standalone, in a list, in a dictionary, and as an optional.</summary>
[Versionable(Version = 1, Hash = "705e61")]
internal sealed partial class GoldenNested
{
    /// <summary>A plain string field.</summary>
    [VersionableField("label")]
    public required string Label { get; init; }

    /// <summary>A nested object.</summary>
    [VersionableField("inner")]
    public required GoldenInner Inner { get; init; }

    /// <summary>Nested objects in a list.</summary>
    [VersionableField("points")]
    public required List<GoldenInner> Points { get; init; }

    /// <summary>Nested objects in a dictionary.</summary>
    [VersionableField("byName")]
    public required Dictionary<string, GoldenInner> ByName { get; init; }

    /// <summary>A nested object that may be absent.</summary>
    [VersionableField("optionalInner")]
    public GoldenInner? OptionalInner { get; init; }
}

/// <summary>
/// Polymorphic base: a collection declared as this may hold any registered subclass.
/// </summary>
/// <remarks>
/// Concrete rather than abstract, mirroring Python, so the base is loadable in its own right.
/// Each subclass gets its own generated metadata and its own registry entry, which is what lets a
/// <c>list[GoldenShape]</c> resolve each element from the element's own envelope.
/// </remarks>
[Versionable(Version = 1, Hash = "357f27")]
internal partial class GoldenShape
{
    /// <summary>Shape label.</summary>
    [VersionableField("label")]
    public required string Label { get; init; }
}

/// <summary>A concrete shape.</summary>
[Versionable(Version = 1, Hash = "8e5e7c")]
internal sealed partial class GoldenCircle : GoldenShape
{
    /// <summary>Circle radius.</summary>
    [VersionableField("radius")]
    public required double Radius { get; init; }
}

/// <summary>Another concrete shape.</summary>
[Versionable(Version = 1, Hash = "42fff3")]
internal sealed partial class GoldenSquare : GoldenShape
{
    /// <summary>Square side length.</summary>
    [VersionableField("side")]
    public required double Side { get; init; }
}

/// <summary>A collection declared as the base type but holding concrete subclasses.</summary>
[Versionable(Version = 1, Hash = "cfb9bf")]
internal sealed partial class GoldenPolymorphic
{
    /// <summary>The shapes, each resolved from its own envelope.</summary>
    [VersionableField("shapes")]
    public required List<GoldenShape> Shapes { get; init; }
}

/// <summary>The current schema of the migration-chain fixture, at version 3.</summary>
[Versionable(Version = 3, Hash = "aac8a2")]
internal sealed partial class GoldenWorker
{
    /// <summary>Worker name; <c>title</c> before the v1 migration renamed it.</summary>
    [VersionableField("name")]
    public required string Name { get; init; }

    /// <summary>Retry count.</summary>
    [VersionableField("retries")]
    public required int Retries { get; init; }

    /// <summary>Timeout, added at v3 and defaulted to 0 for older files.</summary>
    [VersionableField("timeout_ms")]
    public required int TimeoutMs { get; init; }

    /// <summary>
    /// The migration chain, which the generator composes into
    /// <see cref="VersionableMetadata.Migrations"/>.
    /// </summary>
    /// <remarks>
    /// Python counterpart: <c>GoldenWorker.Migrate</c> in <c>conformance/golden_schemas.py</c> —
    /// <c>v1 = Migration().rename("title", "name").drop("debug")</c> and
    /// <c>v2 = Migration().add("timeout_ms", default=0)</c> — transcribed op for op, which is the
    /// point: the corpus files were written by the Python implementation and are read here through
    /// the C# builder.
    /// <para>
    /// The v2 step defaults old files to 0 rather than to the schema default of 30000, which is
    /// the whole point of the fixture: a migration decides what a field meant before it existed.
    /// </para>
    /// </remarks>
    public static class Migrate
    {
        /// <summary>Takes a version 1 file to version 2.</summary>
        public static readonly Migration V1 = new Migration().Rename("title", "name").Drop("debug");

        /// <summary>Takes a version 2 file to version 3.</summary>
        public static readonly Migration V2 = new Migration().Add("timeout_ms", 0);
    }
}

/// <summary>Every scalar token that has a C# counterpart.</summary>
[Versionable(Version = 1, Hash = "c51ec9")]
internal sealed partial class GoldenScalars
{
    /// <summary>A string.</summary>
    [VersionableField("text")]
    public required string Text { get; init; }

    /// <summary>An integer.</summary>
    [VersionableField("count")]
    public required int Count { get; init; }

    /// <summary>A float.</summary>
    [VersionableField("ratio")]
    public required double Ratio { get; init; }

    /// <summary>A boolean.</summary>
    [VersionableField("enabled")]
    public required bool Enabled { get; init; }

    /// <summary>A complex number, which wires as <c>[re, im]</c>.</summary>
    [VersionableField("phase")]
    public required Complex Phase { get; init; }

    /// <summary>Bytes, which wire as base64.</summary>
    [VersionableField("blob")]
    public required byte[] Blob { get; init; }
}

/// <summary>The datetime family, naive and offset-aware.</summary>
[Versionable(Version = 1, Hash = "3856a2")]
internal sealed partial class GoldenTemporal
{
    /// <summary>A naive datetime.</summary>
    [VersionableField("naive")]
    public required DateTime Naive { get; init; }

    /// <summary>An offset-aware datetime.</summary>
    [VersionableField("aware")]
    public required DateTimeOffset Aware { get; init; }

    /// <summary>A date.</summary>
    [VersionableField("day")]
    public required DateOnly Day { get; init; }

    /// <summary>A time of day.</summary>
    [VersionableField("clock")]
    public required TimeOnly Clock { get; init; }

    /// <summary>A duration, which wires as total seconds.</summary>
    [VersionableField("elapsed")]
    public required TimeSpan Elapsed { get; init; }
}

/// <summary>
/// Registered converter types.
/// </summary>
/// <remarks>
/// <b>Not a hash mirror.</b> Python declares <c>posixPath: PurePosixPath</c> and
/// <c>windowsPath: PureWindowsPath</c> and hashes <c>c3ff88</c>; C# v1 has neither type
/// (GRAMMAR §9 lists both as "(none)"), so all three path fields are <see cref="FilePath"/> and
/// all three render <c>Path</c>. The wire form is the same bare path string, so the fixture loads
/// and every value compares equal; only the canonical rendering — and therefore the hash below —
/// differs.
/// </remarks>
[Versionable(Version = 1, Hash = "b38e82")]
internal sealed partial class GoldenStdlib
{
    /// <summary>A path.</summary>
    [VersionableField("filePath")]
    public required FilePath FilePath { get; init; }

    /// <summary>Python's <c>PurePosixPath</c>, which C# can only hold as a path.</summary>
    [VersionableField("posixPath")]
    public required FilePath PosixPath { get; init; }

    /// <summary>Python's <c>PureWindowsPath</c>, which C# can only hold as a path.</summary>
    [VersionableField("windowsPath")]
    public required FilePath WindowsPath { get; init; }

    /// <summary>A decimal, which wires as a string so no precision is lost.</summary>
    [VersionableField("amount")]
    public required decimal Amount { get; init; }

    /// <summary>A UUID.</summary>
    [VersionableField("deviceId")]
    public required Guid DeviceId { get; init; }

    /// <summary>A regex, which wires as its pattern with flags dropped.</summary>
    [VersionableField("serialPattern")]
    public required Regex SerialPattern { get; init; }
}

/// <summary>Five hash-significant dtypes, a 2-D array, and arrays inside containers.</summary>
[Versionable(Version = 1, Hash = "b76a00")]
internal sealed partial class GoldenArrays
{
    /// <summary>A float64 array.</summary>
    [VersionableField("signal")]
    public required Tensor<double> Signal { get; init; }

    /// <summary>A float32 array.</summary>
    [VersionableField("weights")]
    public required Tensor<float> Weights { get; init; }

    /// <summary>An int32 array.</summary>
    [VersionableField("counts")]
    public required Tensor<int> Counts { get; init; }

    /// <summary>A uint8 array.</summary>
    [VersionableField("image")]
    public required Tensor<byte> Image { get; init; }

    /// <summary>A bool array.</summary>
    [VersionableField("mask")]
    public required Tensor<bool> Mask { get; init; }

    /// <summary>A 2-D float64 array; shape is erased from the hash but kept on disk.</summary>
    [VersionableField("matrix")]
    public required Tensor<double> Matrix { get; init; }

    /// <summary>Arrays in a list.</summary>
    [VersionableField("traces")]
    public required List<Tensor<double>> Traces { get; init; }

    /// <summary>Arrays in a dictionary.</summary>
    [VersionableField("channels")]
    public required Dictionary<string, Tensor<double>> Channels { get; init; }
}

/// <summary>
/// Optionals populated and null, plus a multi-member union.
/// </summary>
/// <remarks>
/// <b>The one golden fixture that keeps hand-written metadata, and the only one that has to.</b>
/// Python declares <c>either: int | str</c>. C# spells optionality as <c>T?</c> — a two-member
/// union with <see langword="null"/> — and has no type-level spelling for a union of two
/// non-<see langword="null"/> members; a bare <see cref="object"/> property is rejected outright
/// by VSN0002. The same fact is recorded as <c>csharpDeclarable: false</c> on four conformance
/// vectors. Declaring <c>either</c> as <c>string</c> would compile and load, but it would delete
/// the only golden coverage of <c>WireValues.ReadUnion</c> — the file holds a string, so the
/// <c>int</c> member has to fail and the <c>str</c> member has to win — which is precisely what
/// this fixture exists to prove.
/// <para>
/// Doubling as the required coverage of the manual-construction path: everything here is built
/// exactly as the <see cref="FieldDescriptor"/> contract documents, with closed delegates and no
/// reflection, so the engine is still exercised against a metadata object no generator produced.
/// </para>
/// </remarks>
internal sealed class GoldenOptionals : IVersionableMetadataProvider
{
    /// <summary>An optional holding a value.</summary>
    public string? Present { get; init; }

    /// <summary>An optional holding null.</summary>
    public string? Absent { get; init; }

    /// <summary>An optional value type.</summary>
    public int? MaybeCount { get; init; }

    /// <summary>An optional converter type.</summary>
    public decimal? Money { get; init; }

    /// <summary>A union of two non-null members, which has no CLR type of its own.</summary>
    public required object Either { get; init; }

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;

    /// <summary>Metadata, in the shape the generator emits for every other fixture here.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(GoldenOptionals),
        Name = "GoldenOptionals",
        Version = 1,
        Hash = "70c93b",
        Fields =
        [
            TestMetadata.Field<GoldenOptionals>(
                "present", "Present", typeof(string), "Union[None, str]", value => value.Present, () => null),
            TestMetadata.Field<GoldenOptionals>(
                "absent", "Absent", typeof(string), "Union[None, str]", value => value.Absent, () => null),
            TestMetadata.Field<GoldenOptionals>(
                "maybeCount", "MaybeCount", typeof(int?), "Union[None, int]", value => value.MaybeCount, () => null),
            TestMetadata.Field<GoldenOptionals>(
                "money", "Money", typeof(decimal?), "Union[Decimal, None]", value => value.Money, () => null),
            TestMetadata.Field<GoldenOptionals>(
                "either", "Either", typeof(object), "Union[int, str]",
                value => value.Either,
                () => 0,
                wire => WireValues.ReadUnion(
                    wire,
                    value => WireValues.Read(value, typeof(int)),
                    value => WireValues.Read(value, typeof(string)))),
        ],
        Factory = values => new GoldenOptionals
        {
            Present = (string?)values[0],
            Absent = (string?)values[1],
            MaybeCount = (int?)values[2],
            Money = (decimal?)values[3],
            Either = values[4]!,
        },
    };
}
