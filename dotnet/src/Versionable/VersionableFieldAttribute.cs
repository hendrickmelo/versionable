namespace Versionable;

/// <summary>
/// Overrides the wire name of a field. Without it, the property name is written verbatim.
/// </summary>
/// <remarks>
/// Python counterpart: none — Python writes dataclass field names verbatim
/// (<c>_resolveFields</c> in <c>src/versionable/_base.py</c>). This attribute exists so a
/// C# <c>PascalCase</c> property can map onto a <c>snake_case</c> key written by Python.
/// It is the only naming override; there are no naming policies on either side
/// (<c>docs/plans/csharp-port.md</c>, § "C# architecture").
/// </remarks>
[AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, Inherited = false)]
public sealed class VersionableFieldAttribute : Attribute
{
    /// <summary>Initializes a new instance of the <see cref="VersionableFieldAttribute"/> class.</summary>
    /// <param name="name">Wire name for this field, hash-significant.</param>
    public VersionableFieldAttribute(string name) => Name = name;

    /// <summary>Wire name for this field. Hash-significant (ADR-0001).</summary>
    public string Name { get; }
}
