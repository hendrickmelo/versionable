namespace Versionable;

/// <summary>
/// One serializable field of a <c>[Versionable]</c> type, as emitted by the source generator.
/// </summary>
/// <remarks>
/// Python counterpart: an entry of the <c>{name: type}</c> mapping returned by
/// <c>_resolveFields</c> in <c>src/versionable/_base.py</c>, plus the
/// <c>dataclasses.Field</c> default information consulted by <c>save()</c> in
/// <c>src/versionable/_api.py</c>, plus the <c>Literal</c> option list that
/// <c>deserialize()</c> validates against in <c>src/versionable/_types.py</c>. C# resolves
/// all of it at compile time, so the engine never reflects over user types (ADR-0003).
/// <para>
/// Field order in <see cref="VersionableMetadata.Fields"/> is declaration order, and it is
/// what <see cref="VersionableMetadata.Factory"/> expects. Hash payload order is separate:
/// the grammar sorts pairs by wire name (<c>conformance/GRAMMAR.md</c> §1).
/// </para>
/// <para>
/// Members are <c>init</c>-only with <c>required</c> on the ones the generator must always
/// supply, so new optional members can be added later without breaking generated code.
/// </para>
/// </remarks>
public sealed record FieldDescriptor
{
    /// <summary>Key written to and read from the file. Hash-significant.</summary>
    public required string WireName { get; init; }

    /// <summary>Declaring property or field name in C#.</summary>
    public required string ClrName { get; init; }

    /// <summary>Declared CLR type of the field.</summary>
    public required Type ClrType { get; init; }

    /// <summary>
    /// Canonical grammar rendering of <see cref="ClrType"/> — for example <c>list[int]</c>,
    /// <c>Union[None, str]</c>, <c>ndarray[float64]</c>, <c>Literal['fast', 'slow']</c>.
    /// This exact string goes into the hash payload (<c>conformance/GRAMMAR.md</c>).
    /// </summary>
    public required string CanonicalType { get; init; }

    /// <summary>Reads the field from an instance. Generated, not reflective.</summary>
    public required Func<object, object?> Getter { get; init; }

    /// <summary>
    /// Writes the field on an instance, or <see langword="null"/> for init-only and
    /// positional-record members, which must go through
    /// <see cref="VersionableMetadata.Factory"/>.
    /// </summary>
    public Action<object, object?>? Setter { get; init; }

    /// <summary>
    /// Whether the field has a declared default. Drives <c>SkipDefaults</c> on save and the
    /// TOML/YAML <c>CommentDefaults</c> option.
    /// </summary>
    public bool HasDefault { get; init; }

    /// <summary>
    /// Produces the default value, or <see langword="null"/> when <see cref="HasDefault"/> is
    /// <see langword="false"/>. Mirrors Python's <c>default</c> / <c>default_factory</c>; a
    /// delegate rather than a constant so mutable defaults are not shared between instances.
    /// </summary>
    public Func<object?>? DefaultFactory { get; init; }

    /// <summary>
    /// Declared options for a literal field, in canonical order, or <see langword="null"/>
    /// when the field is not a literal.
    /// </summary>
    /// <remarks>
    /// Populated from <see cref="LiteralValuesAttribute"/>. Order is the attribute's argument
    /// order and is hash-significant — <c>Literal['fast', 'slow']</c> and
    /// <c>Literal['slow', 'fast']</c> are different schemas
    /// (<c>conformance/GRAMMAR.md</c> §8). Members are <see cref="string"/>,
    /// <see cref="int"/>, <see cref="bool"/>, or <see langword="null"/>; nothing else is
    /// representable.
    /// </remarks>
    public IReadOnlyList<object?>? LiteralOptions { get; init; }

    /// <summary>
    /// Whether <see cref="LiteralFallback"/> holds a value to substitute when a loaded value
    /// is outside <see cref="LiteralOptions"/>. Distinguishes "no fallback" from "the
    /// fallback is <see langword="null"/>".
    /// </summary>
    public bool HasLiteralFallback { get; init; }

    /// <summary>
    /// Value substituted for an out-of-range literal, when
    /// <see cref="HasLiteralFallback"/> is set. Python counterpart: the value passed to
    /// <c>literalFallback()</c> in <c>src/versionable/_types.py</c>. Without a fallback,
    /// an out-of-range value is an error whenever
    /// <see cref="VersionableMetadata.ValidateLiterals"/> is set.
    /// </summary>
    public object? LiteralFallback { get; init; }

    /// <summary>
    /// Materializes this field's CLR value from the raw value a backend read, or
    /// <see langword="null"/> when the engine can handle the field without generated help.
    /// </summary>
    /// <remarks>
    /// The AOT seam. Constructing a <c>List&lt;T&gt;</c>, a <c>Dictionary&lt;K, V&gt;</c>, a
    /// nested <c>[Versionable]</c> instance, or an enum value at runtime would otherwise mean
    /// <c>Activator.CreateInstance</c> over a type the trimmer cannot see —
    /// <c>IsAotCompatible=true</c> plus <c>TreatWarningsAsErrors</c> rejects that. The
    /// generator therefore emits a closed delegate here for every field whose type needs
    /// construction: containers, nested Versionable types, enums, and arrays. It leaves this
    /// <see langword="null"/> for scalars and for types a registered
    /// <see cref="Converters.IWireConverter"/> already handles, which the engine converts
    /// without constructing anything.
    /// <para>Python has no counterpart: <c>deserialize()</c> reflects over the annotation.</para>
    /// </remarks>
    public Func<object?, object?>? WireReader { get; init; }

    /// <summary>
    /// Lowers this field's CLR value to the raw value a backend writes, or
    /// <see langword="null"/> when the engine can handle the field without generated help.
    /// </summary>
    /// <remarks>
    /// The inverse of <see cref="WireReader"/>, emitted under the same conditions. Backends
    /// that store values natively rather than through a wire form — HDF5 — may bypass it;
    /// see <see cref="Backends.IVersionableBackend.NativeTypes"/>.
    /// </remarks>
    public Func<object?, object?>? WireWriter { get; init; }
}
