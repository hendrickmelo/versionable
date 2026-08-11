using System.Numerics.Tensors;
using Versionable.Engine;
using Versionable.Migrations;

namespace Versionable.Tests;

/// <summary>
/// Builds <see cref="FieldDescriptor"/> instances the way the source generator does.
/// </summary>
/// <remarks>
/// The golden-corpus fixtures in <c>EngineGoldenSchemas.cs</c> are real <c>[Versionable]</c>
/// types now, so the cross-language contract runs on generated metadata. The engine fixtures
/// below stay hand-written on purpose, and cover two things generated types cannot:
/// <list type="bullet">
///   <item>
///     <description>
///     <b>The manual-construction path.</b> <see cref="VersionableMetadata"/> is public,
///     <c>init</c>-only, and documented as buildable by hand; nothing else would notice if that
///     stopped working.
///     </description>
///   </item>
///   <item>
///     <description>
///     <b>Metadata a declaration cannot express.</b> Several fixtures here vary one member of an
///     otherwise identical schema — <c>SkipDefaults</c>, <c>Unknown</c>, <c>ValidateLiterals</c>,
///     a substituted migration chain — which an attribute fixes at compile time and a
///     <c>with</c> expression does not.
///     </description>
///   </item>
/// </list>
/// Everything here follows the shape the <see cref="FieldDescriptor"/> and
/// <see cref="VersionableMetadata"/> contracts document: closed delegates, no reflection.
/// </remarks>
internal static class TestMetadata
{
    /// <summary>Builds one field descriptor, closing over a typed getter.</summary>
    /// <typeparam name="TOwner">The declaring type.</typeparam>
    /// <param name="wireName">Key on the wire.</param>
    /// <param name="clrName">Property name in C#.</param>
    /// <param name="clrType">Declared CLR type.</param>
    /// <param name="canonicalType">Canonical grammar rendering, as the generator computes it.</param>
    /// <param name="getter">Reads the property from an instance.</param>
    /// <param name="defaultFactory">Produces the declared default, when there is one.</param>
    /// <param name="wireReader">Materialization delegate, for types the engine cannot construct.</param>
    /// <param name="wireWriter">Lowering delegate, for types the engine cannot walk.</param>
    /// <param name="literalOptions">Declared literal options, in canonical order.</param>
    /// <param name="hasLiteralFallback">Whether <paramref name="literalFallback"/> is meaningful.</param>
    /// <param name="literalFallback">Value substituted for an out-of-range literal.</param>
    /// <returns>The descriptor.</returns>
    internal static FieldDescriptor Field<TOwner>(
        string wireName,
        string clrName,
        Type clrType,
        string canonicalType,
        Func<TOwner, object?> getter,
        Func<object?>? defaultFactory = null,
        Func<object?, object?>? wireReader = null,
        Func<object?, object?>? wireWriter = null,
        IReadOnlyList<object?>? literalOptions = null,
        bool hasLiteralFallback = false,
        object? literalFallback = null) =>
        new()
        {
            WireName = wireName,
            ClrName = clrName,
            ClrType = clrType,
            CanonicalType = canonicalType,
            Getter = owner => getter((TOwner)owner),
            HasDefault = defaultFactory is not null,
            DefaultFactory = defaultFactory,
            WireReader = wireReader,
            WireWriter = wireWriter,
            LiteralOptions = literalOptions,
            HasLiteralFallback = hasLiteralFallback,
            LiteralFallback = literalFallback,
        };

    /// <summary>Reads a wire list into a typed list, as a generated container reader does.</summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="wire">The wire value.</param>
    /// <param name="element">Reads one element.</param>
    /// <returns>The materialized list.</returns>
    internal static List<T> ReadList<T>(object? wire, Func<object?, T> element) =>
        [.. WireValues.AsList(wire).Select(element)];

    /// <summary>Reads a wire mapping into a typed dictionary.</summary>
    /// <typeparam name="TKey">Key type.</typeparam>
    /// <typeparam name="TValue">Value type.</typeparam>
    /// <param name="wire">The wire value.</param>
    /// <param name="key">Converts a wire key.</param>
    /// <param name="value">Converts a wire value.</param>
    /// <returns>The materialized dictionary.</returns>
    internal static Dictionary<TKey, TValue> ReadMap<TKey, TValue>(
        object? wire,
        Func<string, TKey> key,
        Func<object?, TValue> value)
        where TKey : notnull =>
        WireValues.AsMap(wire).ToDictionary(entry => key(entry.Key), entry => value(entry.Value));

    /// <summary>Reads one scalar of a declared type.</summary>
    /// <typeparam name="T">The declared type.</typeparam>
    /// <param name="wire">The wire value.</param>
    /// <returns>The materialized value.</returns>
    internal static T Scalar<T>(object? wire) => (T)WireValues.Read(wire, typeof(T))!;
}

/// <summary>A leaf object, nested inside <see cref="EngineNode"/>.</summary>
internal sealed class EngineLeaf(string name, double weight) : IVersionableMetadataProvider
{
    /// <summary>Leaf name.</summary>
    public string Name { get; } = name;

    /// <summary>Leaf weight.</summary>
    public double Weight { get; } = weight;

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineLeaf),
        Name = "EngineLeaf",
        Version = 1,
        Hash = "aaaaaa",
        Fields =
        [
            TestMetadata.Field<EngineLeaf>("name", "Name", typeof(string), "str", leaf => leaf.Name, () => ""),
            TestMetadata.Field<EngineLeaf>("weight", "Weight", typeof(double), "float", leaf => leaf.Weight, () => 0.0),
        ],
        Factory = values => new EngineLeaf((string)values[0]!, (double)values[1]!),
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;
}

/// <summary>Nested objects, containers, and an optional, in one type.</summary>
internal sealed class EngineNode(
    string label,
    EngineLeaf leaf,
    List<EngineLeaf> leaves,
    Dictionary<string, EngineLeaf> byName,
    HashSet<string> tags,
    EngineLeaf? optionalLeaf) : IVersionableMetadataProvider
{
    /// <summary>Node label.</summary>
    public string Label { get; } = label;

    /// <summary>A nested object.</summary>
    public EngineLeaf Leaf { get; } = leaf;

    /// <summary>Nested objects in a list.</summary>
    public List<EngineLeaf> Leaves { get; } = leaves;

    /// <summary>Nested objects in a dictionary.</summary>
    public Dictionary<string, EngineLeaf> ByName { get; } = byName;

    /// <summary>A set of strings.</summary>
    public HashSet<string> Tags { get; } = tags;

    /// <summary>A nested object that may be absent.</summary>
    public EngineLeaf? OptionalLeaf { get; } = optionalLeaf;

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineNode),
        Name = "EngineNode",
        Version = 1,
        Hash = "bbbbbb",
        Fields =
        [
            TestMetadata.Field<EngineNode>("label", "Label", typeof(string), "str", node => node.Label, () => ""),
            TestMetadata.Field<EngineNode>(
                "leaf", "Leaf", typeof(EngineLeaf), "EngineLeaf", node => node.Leaf),
            TestMetadata.Field<EngineNode>(
                "leaves",
                "Leaves",
                typeof(List<EngineLeaf>),
                "list[EngineLeaf]",
                node => node.Leaves,
                () => new List<EngineLeaf>(),
                wire => TestMetadata.ReadList(
                    wire, item => (EngineLeaf)WireValues.ReadVersionable(item, EngineLeaf.Metadata))),
            TestMetadata.Field<EngineNode>(
                "byName",
                "ByName",
                typeof(Dictionary<string, EngineLeaf>),
                "dict[str, EngineLeaf]",
                node => node.ByName,
                () => new Dictionary<string, EngineLeaf>(StringComparer.Ordinal),
                wire => TestMetadata.ReadMap(
                    wire,
                    key => key,
                    value => (EngineLeaf)WireValues.ReadVersionable(value, EngineLeaf.Metadata))),
            TestMetadata.Field<EngineNode>(
                "tags",
                "Tags",
                typeof(HashSet<string>),
                "set[str]",
                node => node.Tags,
                () => new HashSet<string>(StringComparer.Ordinal),
                wire => new HashSet<string>(
                    WireValues.AsList(wire).Select(TestMetadata.Scalar<string>), StringComparer.Ordinal)),
            TestMetadata.Field<EngineNode>(
                "optionalLeaf",
                "OptionalLeaf",
                typeof(EngineLeaf),
                "Union[None, EngineLeaf]",
                node => node.OptionalLeaf,
                () => null),
        ],
        Factory = values => new EngineNode(
            (string)values[0]!,
            (EngineLeaf)values[1]!,
            (List<EngineLeaf>)values[2]!,
            (Dictionary<string, EngineLeaf>)values[3]!,
            (HashSet<string>)values[4]!,
            (EngineLeaf?)values[5]),
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;
}

/// <summary>A self-referencing type, for cycle detection.</summary>
internal sealed class EngineChain(string name) : IVersionableMetadataProvider
{
    /// <summary>Link name.</summary>
    public string Name { get; } = name;

    /// <summary>The next link, or <see langword="null"/>.</summary>
    public EngineChain? Next { get; set; }

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineChain),
        Name = "EngineChain",
        Version = 1,
        Hash = "cccccc",
        Fields =
        [
            TestMetadata.Field<EngineChain>("name", "Name", typeof(string), "str", link => link.Name, () => ""),
            TestMetadata.Field<EngineChain>(
                "next", "Next", typeof(EngineChain), "Union[None, EngineChain]", link => link.Next, () => null),
        ],
        Factory = values => new EngineChain((string)values[0]!) { Next = (EngineChain?)values[1] },
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;
}

/// <summary>Literal-typed fields, with and without a fallback.</summary>
internal sealed class EngineLiteralHolder(string mode, int level) : IVersionableMetadataProvider
{
    /// <summary>A literal with no fallback.</summary>
    public string Mode { get; } = mode;

    /// <summary>A literal with a fallback.</summary>
    public int Level { get; } = level;

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineLiteralHolder),
        Name = "EngineLiteralHolder",
        Version = 1,
        Hash = "dddddd",
        Fields =
        [
            TestMetadata.Field<EngineLiteralHolder>(
                "mode",
                "Mode",
                typeof(string),
                "Literal['fast', 'slow']",
                holder => holder.Mode,
                () => "fast",
                literalOptions: ["fast", "slow"]),
            TestMetadata.Field<EngineLiteralHolder>(
                "level",
                "Level",
                typeof(int),
                "Literal[1, 2, 3]",
                holder => holder.Level,
                () => 1,
                literalOptions: [1, 2, 3],
                hasLiteralFallback: true,
                literalFallback: 1),
        ],
        Factory = values => new EngineLiteralHolder((string)values[0]!, (int)values[1]!),
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;

    /// <summary>The same schema with literal validation switched off.</summary>
    public static VersionableMetadata Unvalidated { get; } = Metadata with { ValidateLiterals = false };
}

/// <summary>A multi-member union (<c>int | str</c>), which has no CLR type of its own.</summary>
internal sealed class EngineUnionHolder(object either) : IVersionableMetadataProvider
{
    /// <summary>The union-typed value.</summary>
    public object Either { get; } = either;

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineUnionHolder),
        Name = "EngineUnionHolder",
        Version = 1,
        Hash = "eeeeee",
        Fields =
        [
            TestMetadata.Field<EngineUnionHolder>(
                "either",
                "Either",
                typeof(object),
                "Union[int, str]",
                holder => holder.Either,
                () => 0,
                wire => WireValues.ReadUnion(
                    wire,
                    value => WireValues.Read(value, typeof(int)),
                    value => WireValues.Read(value, typeof(string)))),
        ],
        Factory = values => new EngineUnionHolder(values[0]!),
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;
}

/// <summary>Defaults, missing fields, and the unknown-field policies.</summary>
internal sealed class EngineSettings(string name, int port, string required) : IVersionableMetadataProvider
{
    /// <summary>A field with a default.</summary>
    public string Name { get; } = name;

    /// <summary>Another field with a default.</summary>
    public int Port { get; } = port;

    /// <summary>A field with no default, which a file must supply.</summary>
    public string Required { get; } = required;

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineSettings),
        Name = "EngineSettings",
        Version = 1,
        Hash = "ffffff",
        Fields =
        [
            TestMetadata.Field<EngineSettings>(
                "name", "Name", typeof(string), "str", settings => settings.Name, () => "anon"),
            TestMetadata.Field<EngineSettings>(
                "port", "Port", typeof(int), "int", settings => settings.Port, () => 8080),
            TestMetadata.Field<EngineSettings>(
                "required", "Required", typeof(string), "str", settings => settings.Required),
        ],
        Factory = values => new EngineSettings((string)values[0]!, (int)values[1]!, (string)values[2]!),
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;

    /// <summary>The same schema declared <c>SkipDefaults</c>.</summary>
    public static VersionableMetadata SkippingDefaults { get; } = Metadata with { SkipDefaults = true };

    /// <summary>The same schema declared <c>unknown = Error</c>.</summary>
    public static VersionableMetadata RejectingUnknown { get; } =
        Metadata with { Unknown = UnknownFieldPolicy.Error };
}

/// <summary>
/// A chain that renames one field per step, for the version-dispatch tests.
/// </summary>
/// <remarks>
/// Hand-written rather than built with <see cref="Migration"/>: these tests are about what
/// <c>MigrationRunner</c> decides — which steps run, in what order, and which files it refuses —
/// so the chain records what it was asked to do instead of doing anything interesting. The
/// builder's own ops are covered by <c>MigrationBuilderTests</c>, and the composed chain by the
/// golden corpus.
/// </remarks>
internal sealed class EngineRenameChain(IReadOnlyList<int> fromVersions, int? minReversibleVersion)
    : IMigrationChain
{
    /// <inheritdoc/>
    public IReadOnlyList<int> FromVersions { get; } = fromVersions;

    /// <inheritdoc/>
    public int? MinReversibleVersion { get; } = minReversibleVersion;

    /// <summary>Source versions this chain was actually asked to run.</summary>
    public List<int> Applied { get; } = [];

    /// <inheritdoc/>
    public IDictionary<string, object?> Apply(
        IDictionary<string, object?> fields,
        int fromVersion,
        int toVersion,
        bool upgradeInPlace)
    {
        for (int version = fromVersion; version < toVersion; version++)
        {
            Applied.Add(version);
            if (fields.Remove($"name_v{version}", out object? value))
            {
                fields[$"name_v{version + 1}"] = value;
            }
        }

        return fields;
    }
}

/// <summary>A versioned type whose field name changes with every schema version.</summary>
internal sealed class EngineVersioned(string name) : IVersionableMetadataProvider
{
    /// <summary>The renamed field, at version 3.</summary>
    public string Name { get; } = name;

    /// <summary>Metadata at version 3, with a chain reaching back to version 1.</summary>
    public static VersionableMetadata Metadata { get; } = At(new EngineRenameChain([1, 2], null));

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;

    /// <summary>Builds the version-3 metadata around a specific chain.</summary>
    /// <param name="chain">The migration chain to declare, or <see langword="null"/> for none.</param>
    /// <returns>The metadata.</returns>
    public static VersionableMetadata At(IMigrationChain? chain) => new()
    {
        ClrType = typeof(EngineVersioned),
        Name = "EngineVersioned",
        Version = 3,
        Hash = "111111",
        Fields =
        [
            TestMetadata.Field<EngineVersioned>(
                "name_v3", "Name", typeof(string), "str", versioned => versioned.Name, () => ""),
        ],
        Factory = values => new EngineVersioned((string)values[0]!),
        Migrations = chain,
    };
}

/// <summary>
/// A string-valued enum that names a fallback for values a newer schema may write.
/// </summary>
/// <remarks>
/// Python counterpart: a string-valued <c>Enum</c> with <c>VERSIONABLE_FALLBACK</c> assigned after
/// the body.
/// </remarks>
internal enum EngineStatus
{
    /// <summary>Wires as <c>active</c>.</summary>
    [EnumValue("active")]
    Active,

    /// <summary>Wires as <c>retired</c>.</summary>
    [EnumValue("retired")]
    Retired,

    /// <summary>Wires as <c>unknown</c>, and absorbs any value this build does not define.</summary>
    [EnumValue("unknown")]
    [EnumFallback]
    Unknown,
}

/// <summary>A string-valued enum with no fallback: an unknown value is an error.</summary>
internal enum EngineMode
{
    /// <summary>Wires as <c>fast</c>.</summary>
    [EnumValue("fast")]
    Fast,

    /// <summary>Wires as <c>slow</c>.</summary>
    [EnumValue("slow")]
    Slow,
}

/// <summary>Enums reaching the engine with no generated help, standalone and in a container.</summary>
internal sealed class EngineEnumHolder(EngineStatus status, EngineMode mode, List<EngineStatus> history)
    : IVersionableMetadataProvider
{
    /// <summary>An enum with a fallback member.</summary>
    public EngineStatus Status { get; } = status;

    /// <summary>An enum without one.</summary>
    public EngineMode Mode { get; } = mode;

    /// <summary>Enums inside a list, which the walker lowers element by element.</summary>
    public List<EngineStatus> History { get; } = history;

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineEnumHolder),
        Name = "EngineEnumHolder",
        Version = 1,
        Hash = "222222",
        Fields =
        [
            TestMetadata.Field<EngineEnumHolder>(
                "status", "Status", typeof(EngineStatus), "EngineStatus",
                holder => holder.Status, () => EngineStatus.Active),
            TestMetadata.Field<EngineEnumHolder>(
                "mode", "Mode", typeof(EngineMode), "EngineMode",
                holder => holder.Mode, () => EngineMode.Fast),
            // Reader only, no writer: constructing a List<T> needs generated help, lowering one
            // does not — the walker enumerates it and routes each member through EnumConverter.
            TestMetadata.Field<EngineEnumHolder>(
                "history", "History", typeof(List<EngineStatus>), "list[EngineStatus]",
                holder => holder.History,
                () => new List<EngineStatus>(),
                wire => TestMetadata.ReadList(wire, TestMetadata.Scalar<EngineStatus>)),
        ],
        Factory = values => new EngineEnumHolder(
            (EngineStatus)values[0]!, (EngineMode)values[1]!, (List<EngineStatus>)values[2]!),
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;
}

/// <summary>Integral widths that a JSON reader has to keep straight.</summary>
internal sealed class EngineNumbers(long big, int small, ulong huge, double ratio) : IVersionableMetadataProvider
{
    /// <summary>A 64-bit integer, including values no double can hold exactly.</summary>
    public long Big { get; } = big;

    /// <summary>A 32-bit integer.</summary>
    public int Small { get; } = small;

    /// <summary>An unsigned 64-bit integer, whose top half is above <see cref="long.MaxValue"/>.</summary>
    public ulong Huge { get; } = huge;

    /// <summary>A float, which must stay a float.</summary>
    public double Ratio { get; } = ratio;

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineNumbers),
        Name = "EngineNumbers",
        Version = 1,
        Hash = "333333",
        Fields =
        [
            TestMetadata.Field<EngineNumbers>(
                "big", "Big", typeof(long), "int", numbers => numbers.Big, () => 0L),
            TestMetadata.Field<EngineNumbers>(
                "small", "Small", typeof(int), "int", numbers => numbers.Small, () => 0),
            TestMetadata.Field<EngineNumbers>(
                "huge", "Huge", typeof(ulong), "int", numbers => numbers.Huge, () => 0UL),
            TestMetadata.Field<EngineNumbers>(
                "ratio", "Ratio", typeof(double), "float", numbers => numbers.Ratio, () => 0.0),
        ],
        Factory = values => new EngineNumbers(
            (long)values[0]!, (int)values[1]!, (ulong)values[2]!, (double)values[3]!),
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;
}

/// <summary>
/// A container of literals, whose elements the generated reader validates.
/// </summary>
/// <remarks>
/// The seam described on <c>ObjectMaterializer.ValidateLiteral</c>: the descriptor leaves
/// <see cref="FieldDescriptor.LiteralOptions"/> null — the engine would otherwise compare the whole
/// list against <c>'fast'</c> and reject every file — and the
/// <see cref="FieldDescriptor.WireReader"/> checks each element instead.
/// </remarks>
internal sealed class EngineLiteralList(List<string> modes) : IVersionableMetadataProvider
{
    private static readonly string[] _allowed = ["fast", "slow"];

    /// <summary>A list whose every element must be a declared option.</summary>
    public List<string> Modes { get; } = modes;

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineLiteralList),
        Name = "EngineLiteralList",
        Version = 1,
        Hash = "444444",
        Fields =
        [
            TestMetadata.Field<EngineLiteralList>(
                "modes", "Modes", typeof(List<string>), "list[Literal['fast', 'slow']]",
                holder => holder.Modes,
                () => new List<string>(),
                wire => TestMetadata.ReadList(wire, ReadOption)),
        ],
        Factory = values => new EngineLiteralList((List<string>)values[0]!),
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;

    private static string ReadOption(object? wire)
    {
        string value = TestMetadata.Scalar<string>(wire);
        return Array.IndexOf(_allowed, value) >= 0
            ? value
            : throw new Errors.ConverterException(
                $"EngineLiteralList.modes: value '{value}' is not a valid Literal option. "
                    + $"Allowed values: ['{string.Join("', '", _allowed)}'].");
    }
}

/// <summary>
/// Tuple-valued fields, which the walker has to lower without a generated writer.
/// </summary>
/// <remarks>
/// A tuple gets a <see cref="FieldDescriptor.WireReader"/> — constructing one needs generated help
/// — but no <see cref="FieldDescriptor.WireWriter"/>, because lowering it does not: the engine
/// walks it through <see cref="System.Runtime.CompilerServices.ITuple"/>. The descriptors below
/// leave <c>wireWriter</c> null on purpose; filling it in would hide the engine path this fixture
/// exists to exercise.
/// </remarks>
internal sealed class EngineTupleHolder((int Code, string Label) pair, (int Id, (double X, string Tag) Inner) nested)
    : IVersionableMetadataProvider
{
    /// <summary>A flat two-element tuple of mixed element types.</summary>
    public (int Code, string Label) Pair { get; } = pair;

    /// <summary>A tuple holding another tuple, which recurses through the same path.</summary>
    public (int Id, (double X, string Tag) Inner) Nested { get; } = nested;

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineTupleHolder),
        Name = "EngineTupleHolder",
        Version = 1,
        Hash = "555555",
        Fields =
        [
            TestMetadata.Field<EngineTupleHolder>(
                "pair", "Pair", typeof((int, string)), "tuple[int, str]",
                holder => holder.Pair,
                () => default((int, string)),
                wire =>
                {
                    IReadOnlyList<object?> items = WireValues.AsList(wire);
                    return (TestMetadata.Scalar<int>(items[0]), TestMetadata.Scalar<string>(items[1]));
                }),
            TestMetadata.Field<EngineTupleHolder>(
                "nested", "Nested", typeof((int, (double, string))), "tuple[int, tuple[float, str]]",
                holder => holder.Nested,
                () => default((int, (double, string))),
                wire =>
                {
                    IReadOnlyList<object?> items = WireValues.AsList(wire);
                    IReadOnlyList<object?> inner = WireValues.AsList(items[1]);
                    return (
                        TestMetadata.Scalar<int>(items[0]),
                        (TestMetadata.Scalar<double>(inner[0]), TestMetadata.Scalar<string>(inner[1])));
                }),
        ],
        Factory = values => new EngineTupleHolder(
            ((int, string))values[0]!, ((int, (double, string)))values[1]!),
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;
}

/// <summary>
/// An array field that declares a default, for the <c>MetadataOnly</c> path.
/// </summary>
/// <remarks>
/// The counterpart of <c>GoldenArrays</c>, whose array fields are <c>required</c> with no
/// initializer. A backend told to skip a field hands the materializer a name with no value, and
/// what happens next turns entirely on whether the schema said what an unset value is — so both
/// halves need a fixture.
/// </remarks>
internal sealed class EngineLazyHolder(string label, int count, Tensor<double> samples)
    : IVersionableMetadataProvider
{
    /// <summary>A scalar, which <c>MetadataOnly</c> still reads.</summary>
    public string Label { get; } = label;

    /// <summary>Another scalar.</summary>
    public int Count { get; } = count;

    /// <summary>An array, which <c>MetadataOnly</c> skips — and which declares a default.</summary>
    public Tensor<double> Samples { get; } = samples;

    /// <summary>The value a skipped <see cref="Samples"/> comes back as.</summary>
    public static Tensor<double> EmptySamples { get; } = Tensor.Create(Array.Empty<double>(), [(nint)0]);

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineLazyHolder),
        Name = "EngineLazyHolder",
        Version = 1,
        Hash = "666666",
        Fields =
        [
            TestMetadata.Field<EngineLazyHolder>(
                "label", "Label", typeof(string), "str", holder => holder.Label, () => ""),
            TestMetadata.Field<EngineLazyHolder>(
                "count", "Count", typeof(int), "int", holder => holder.Count, () => 0),
            TestMetadata.Field<EngineLazyHolder>(
                "samples", "Samples", typeof(Tensor<double>), "ndarray[float64]",
                holder => holder.Samples,
                () => EmptySamples),
        ],
        Factory = values => new EngineLazyHolder(
            (string)values[0]!, (int)values[1]!, (Tensor<double>)values[2]!),
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;
}

/// <summary>An array field with no default, for the strict half of the skipped-field semantic.</summary>
internal sealed class EngineStrictArray(Tensor<double> values) : IVersionableMetadataProvider
{
    /// <summary>An array the schema gives no stand-in for.</summary>
    public Tensor<double> Values { get; } = values;

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineStrictArray),
        Name = "EngineStrictArray",
        Version = 1,
        Hash = "777777",
        Fields =
        [
            TestMetadata.Field<EngineStrictArray>(
                "values", "Values", typeof(Tensor<double>), "ndarray[float64]", holder => holder.Values),
        ],
        Factory = values => new EngineStrictArray((Tensor<double>)values[0]!),
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;
}

/// <summary>An object whose nested object has a skippable array with a default.</summary>
internal sealed class EngineNestedLazy(string label, EngineLazyHolder inner) : IVersionableMetadataProvider
{
    /// <summary>A scalar at the root.</summary>
    public string Label { get; } = label;

    /// <summary>The nested object, whose own array field is what gets skipped.</summary>
    public EngineLazyHolder Inner { get; } = inner;

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineNestedLazy),
        Name = "EngineNestedLazy",
        Version = 1,
        Hash = "888888",
        Fields =
        [
            TestMetadata.Field<EngineNestedLazy>(
                "label", "Label", typeof(string), "str", outer => outer.Label, () => ""),
            TestMetadata.Field<EngineNestedLazy>(
                "inner", "Inner", typeof(EngineLazyHolder), "EngineLazyHolder", outer => outer.Inner),
        ],
        Factory = values => new EngineNestedLazy((string)values[0]!, (EngineLazyHolder)values[1]!),
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;
}

/// <summary>An object whose nested object has a skippable array with no default.</summary>
internal sealed class EngineNestedStrict(string label, EngineStrictArray inner) : IVersionableMetadataProvider
{
    /// <summary>A scalar at the root.</summary>
    public string Label { get; } = label;

    /// <summary>The nested object with the unsubstitutable array.</summary>
    public EngineStrictArray Inner { get; } = inner;

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineNestedStrict),
        Name = "EngineNestedStrict",
        Version = 1,
        Hash = "999999",
        Fields =
        [
            TestMetadata.Field<EngineNestedStrict>(
                "label", "Label", typeof(string), "str", outer => outer.Label, () => ""),
            TestMetadata.Field<EngineNestedStrict>(
                "inner", "Inner", typeof(EngineStrictArray), "EngineStrictArray", outer => outer.Inner),
        ],
        Factory = values => new EngineNestedStrict((string)values[0]!, (EngineStrictArray)values[1]!),
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;
}

/// <summary>
/// A collection of nested objects, each with a skippable array — two levels down, through a
/// generated container reader that never sees an element's index.
/// </summary>
internal sealed class EngineDeepLazy(string label, List<EngineLazyHolder> items) : IVersionableMetadataProvider
{
    /// <summary>A scalar at the root.</summary>
    public string Label { get; } = label;

    /// <summary>The elements, whose skipped array field records once, as <c>items/samples</c>.</summary>
    public List<EngineLazyHolder> Items { get; } = items;

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineDeepLazy),
        Name = "EngineDeepLazy",
        Version = 1,
        Hash = "aabbcc",
        Fields =
        [
            TestMetadata.Field<EngineDeepLazy>(
                "label", "Label", typeof(string), "str", outer => outer.Label, () => ""),
            TestMetadata.Field<EngineDeepLazy>(
                "items", "Items", typeof(List<EngineLazyHolder>), "list[EngineLazyHolder]",
                outer => outer.Items,
                () => new List<EngineLazyHolder>(),
                wire => TestMetadata.ReadList(
                    wire, item => (EngineLazyHolder)WireValues.ReadVersionable(item, EngineLazyHolder.Metadata))),
        ],
        Factory = values => new EngineDeepLazy((string)values[0]!, (List<EngineLazyHolder>)values[1]!),
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;
}

/// <summary>
/// A dictionary of lists of nested objects: the skipped array is two containers down.
/// </summary>
/// <remarks>
/// The shape that broke index-qualified paths. With indexes recorded, the reader wrote
/// <c>groups/g1/0/samples</c> and no amount of prefix arithmetic could match it back to an element
/// the engine only knows as "an element". With indexes collapsed it records <c>groups/samples</c>,
/// and one segment of narrowing carries it to every element at any depth.
/// </remarks>
internal sealed class EngineGroupedLazy(Dictionary<string, List<EngineLazyHolder>> groups)
    : IVersionableMetadataProvider
{
    /// <summary>Nested objects two containers down.</summary>
    public Dictionary<string, List<EngineLazyHolder>> Groups { get; } = groups;

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineGroupedLazy),
        Name = "EngineGroupedLazy",
        Version = 1,
        Hash = "bbccdd",
        Fields =
        [
            TestMetadata.Field<EngineGroupedLazy>(
                "groups", "Groups", typeof(Dictionary<string, List<EngineLazyHolder>>),
                "dict[str, list[EngineLazyHolder]]",
                outer => outer.Groups,
                () => new Dictionary<string, List<EngineLazyHolder>>(StringComparer.Ordinal),
                wire => TestMetadata.ReadMap(
                    wire,
                    key => key,
                    list => TestMetadata.ReadList(
                        list, item => (EngineLazyHolder)WireValues.ReadVersionable(item, EngineLazyHolder.Metadata)))),
        ],
        Factory = values => new EngineGroupedLazy((Dictionary<string, List<EngineLazyHolder>>)values[0]!),
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;
}

/// <summary>The same shape, with an element type that declares no default.</summary>
internal sealed class EngineGroupedStrict(Dictionary<string, List<EngineStrictArray>> groups)
    : IVersionableMetadataProvider
{
    /// <summary>Nested objects two containers down, none of which can stand in for a skipped array.</summary>
    public Dictionary<string, List<EngineStrictArray>> Groups { get; } = groups;

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineGroupedStrict),
        Name = "EngineGroupedStrict",
        Version = 1,
        Hash = "ccddee",
        Fields =
        [
            TestMetadata.Field<EngineGroupedStrict>(
                "groups", "Groups", typeof(Dictionary<string, List<EngineStrictArray>>),
                "dict[str, list[EngineStrictArray]]",
                outer => outer.Groups,
                () => new Dictionary<string, List<EngineStrictArray>>(StringComparer.Ordinal),
                wire => TestMetadata.ReadMap(
                    wire,
                    key => key,
                    list => TestMetadata.ReadList(
                        list,
                        item => (EngineStrictArray)WireValues.ReadVersionable(
                            item, EngineStrictArray.Metadata)))),
        ],
        Factory = values => new EngineGroupedStrict((Dictionary<string, List<EngineStrictArray>>)values[0]!),
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;
}

/// <summary>
/// A list of dictionaries of nested objects — three container levels, to prove depth does not
/// matter.
/// </summary>
internal sealed class EngineLayeredLazy(List<Dictionary<string, EngineLazyHolder>> layers)
    : IVersionableMetadataProvider
{
    /// <summary>Nested objects behind a list and then a dictionary.</summary>
    public List<Dictionary<string, EngineLazyHolder>> Layers { get; } = layers;

    /// <summary>Metadata, as the generator would emit it.</summary>
    public static VersionableMetadata Metadata { get; } = new()
    {
        ClrType = typeof(EngineLayeredLazy),
        Name = "EngineLayeredLazy",
        Version = 1,
        Hash = "ddeeff",
        Fields =
        [
            TestMetadata.Field<EngineLayeredLazy>(
                "layers", "Layers", typeof(List<Dictionary<string, EngineLazyHolder>>),
                "list[dict[str, EngineLazyHolder]]",
                outer => outer.Layers,
                () => new List<Dictionary<string, EngineLazyHolder>>(),
                wire => TestMetadata.ReadList(
                    wire,
                    layer => TestMetadata.ReadMap(
                        layer,
                        key => key,
                        item => (EngineLazyHolder)WireValues.ReadVersionable(item, EngineLazyHolder.Metadata)))),
        ],
        Factory = values => new EngineLayeredLazy((List<Dictionary<string, EngineLazyHolder>>)values[0]!),
    };

    static VersionableMetadata IVersionableMetadataProvider.VersionableMetadata => Metadata;
}
