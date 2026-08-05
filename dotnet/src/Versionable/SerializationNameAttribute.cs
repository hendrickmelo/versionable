namespace Versionable;

/// <summary>
/// Overrides a type's Serialization Name — the bare identifier it renders as in schema
/// hashes and writes into envelopes.
/// </summary>
/// <remarks>
/// Without it the Serialization Name is the bare type name, with no namespace, assembly, or
/// enclosing-type qualification, so moving a file or renaming a namespace never changes a
/// hash (<c>conformance/GRAMMAR.md</c> §9, ADR-0001). Declare an override when two types in
/// different namespaces would otherwise collide — Serialization Names must be unique across
/// every type reachable from a schema — or when a type is renamed but its files must keep
/// loading.
/// <para>
/// Python counterpart: three mechanisms unified into one here — <c>name=</c> on a
/// <c>Versionable</c> class, the <c>VERSIONABLE_NAME</c> class attribute on an enum, and
/// <c>setSerializationName()</c> / <c>registerConverter(name=)</c> for converter types
/// (<c>src/versionable/_hash.py</c>, <c>src/versionable/_types.py</c>). C# needs one
/// attribute rather than three because attributes apply uniformly to classes, structs, and
/// enums.
/// </para>
/// <para>
/// This is why the attribute is separate from <see cref="VersionableAttribute"/>: enums and
/// converter types need Serialization Names but have no version and no hash, so folding the
/// name into <c>[Versionable]</c> would mean declaring meaningless version and hash values on
/// an enum.
/// </para>
/// </remarks>
[AttributeUsage(
    AttributeTargets.Class | AttributeTargets.Struct | AttributeTargets.Enum | AttributeTargets.Interface,
    Inherited = false)]
public sealed class SerializationNameAttribute : Attribute
{
    /// <summary>Initializes a new instance of the <see cref="SerializationNameAttribute"/> class.</summary>
    /// <param name="name">
    /// Bare identifier this type hashes and serializes as. Must contain no namespace or
    /// enclosing-type qualification.
    /// </param>
    public SerializationNameAttribute(string name) => Name = name;

    /// <summary>Bare identifier this type hashes and serializes as. Hash-significant.</summary>
    public string Name { get; }
}
