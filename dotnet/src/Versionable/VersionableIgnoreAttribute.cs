namespace Versionable;

/// <summary>
/// Keeps a public property or field off the wire: out of the schema hash, out of the
/// generated field descriptors, and out of every saved file.
/// </summary>
/// <remarks>
/// Python needs no counterpart. A dataclass's fields are exactly its annotated class
/// attributes, so anything that should not persist simply goes unannotated, or is declared
/// <c>ClassVar</c>, or lives behind a <c>@property</c> — and <c>_resolveFields</c> in
/// <c>src/versionable/_base.py</c> never sees it. C# has no such marker: a persistable member
/// and an incidental one look identical, so the generator's rule is "public, non-static, and
/// stores state", and this attribute is the opt-out that rule needs.
/// <para>
/// Computed members — a get-only property with a body or an expression body — are excluded
/// without it, because a derived value is not state. This is for the case the shape cannot
/// distinguish: a public settable property holding a cache, a handle, or a back-reference.
/// </para>
/// <para>
/// <b>Applying or removing it changes the schema hash</b>, because it changes the field set.
/// That is deliberate: a field silently leaving the wire is exactly the drift the hash exists
/// to catch.
/// </para>
/// <example>
/// <code>
/// [Versionable(Version = 1, Hash = "a2f240")]
/// public sealed partial class Session
/// {
///     public string Name { get; init; } = "";
///
///     [VersionableIgnore]
///     public Stream? OpenFile { get; set; }
/// }
/// </code>
/// </example>
/// </remarks>
[AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, Inherited = false)]
public sealed class VersionableIgnoreAttribute : Attribute;
