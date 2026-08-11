using Versionable.Migrations;

namespace Versionable;

/// <summary>
/// Everything the runtime engine needs to save, load, and migrate one <c>[Versionable]</c>
/// type. Emitted per type by the source generator and registered from a
/// <c>[ModuleInitializer]</c>.
/// </summary>
/// <remarks>
/// Python counterpart: <c>VersionableMetadata</c> and <c>_InternalMeta</c> in
/// <c>src/versionable/_base.py</c>. The C# record carries more than the Python one because it
/// also stands in for what Python obtains by reflection at runtime: typed accessors, an
/// instance factory, and the canonical type string per field (ADR-0003).
/// <para>
/// Reached two ways, both reflection-free: <c>T.VersionableMetadata</c> via
/// <see cref="IVersionableMetadataProvider"/> where the type is known at compile time, and
/// <see cref="VersionableRegistry"/> where only a <see cref="Type"/> or an envelope name is.
/// Both hand back the <em>same instance</em> — the registry stores what it was given and never
/// copies — so reference equality is a valid identity check. See
/// <see cref="IVersionableMetadataProvider"/> for the full identity contract.
/// </para>
/// <para>
/// Members are <c>init</c>-only, with <c>required</c> on the ones the generator must always
/// supply, so optional members can be added later without breaking generated code or
/// disturbing an argument order.
/// </para>
/// </remarks>
public sealed record VersionableMetadata
{
    /// <summary>The type this metadata describes.</summary>
    public required Type ClrType { get; init; }

    /// <summary>
    /// Serialization Name written to the envelope's <c>object</c> key. Defaults to the bare
    /// type name; overridden by <see cref="SerializationNameAttribute"/>. Bare names — no
    /// namespace — keep hashes stable across file moves and across languages (ADR-0001).
    /// </summary>
    public required string Name { get; init; }

    /// <summary>Schema version written to the envelope's <c>version</c> key.</summary>
    public required int Version { get; init; }

    /// <summary>Declared schema hash, validated at compile time by the analyzer.</summary>
    public required string Hash { get; init; }

    /// <summary>
    /// Serializable fields in declaration order, which is the order
    /// <see cref="Factory"/> expects. Hash payload order is separate: the grammar sorts pairs
    /// by wire name (<c>conformance/GRAMMAR.md</c> §1).
    /// </summary>
    public required IReadOnlyList<FieldDescriptor> Fields { get; init; }

    /// <summary>
    /// Constructs an instance from field values ordered to match <see cref="Fields"/>. Needed
    /// for init-only and positional-record members, which have no setter.
    /// </summary>
    public required Func<object?[], object> Factory { get; init; }

    /// <summary>
    /// Previously used Serialization Names that still resolve to this type on load. Mirrors
    /// Python's <c>old_names</c>.
    /// </summary>
    public IReadOnlyList<string>? OldNames { get; init; }

    /// <summary>Mirrors Python's <c>skip_defaults</c>.</summary>
    public bool SkipDefaults { get; init; }

    /// <summary>Mirrors Python's <c>unknown</c>.</summary>
    public UnknownFieldPolicy Unknown { get; init; } = UnknownFieldPolicy.Ignore;

    /// <summary>Mirrors Python's <c>validate_literals</c>.</summary>
    public bool ValidateLiterals { get; init; } = true;

    /// <summary>
    /// Migration chain for this type, or <see langword="null"/> when the type declares none.
    /// </summary>
    public IMigrationChain? Migrations { get; init; }
}
