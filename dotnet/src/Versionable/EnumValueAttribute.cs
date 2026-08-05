namespace Versionable;

/// <summary>
/// Gives one enum member an explicit string value on the wire.
/// </summary>
/// <remarks>
/// Python counterpart: a string-valued enum — <c>class Status(Enum): ACTIVE = "active"</c> —
/// whose member <em>values</em> are what serialize. A C# enum member's value is always
/// integral, so writing files a Python string-valued enum can read needs this attribute.
/// <para>
/// Members without it serialize as their bare numeric value. Enum member values are not
/// hash-significant: an enum hashes by its Serialization Name alone, so adding, removing, or
/// changing the value of a member never changes a schema hash
/// (<c>conformance/GRAMMAR.md</c> §9).
/// </para>
/// <example>
/// <code>
/// public enum Status
/// {
///     [EnumValue("active")] Active,
///     [EnumValue("unknown")][EnumFallback] Unknown,
/// }
/// </code>
/// </example>
/// </remarks>
[AttributeUsage(AttributeTargets.Field, Inherited = false)]
public sealed class EnumValueAttribute : Attribute
{
    /// <summary>Initializes a new instance of the <see cref="EnumValueAttribute"/> class.</summary>
    /// <param name="value">String written to and matched from the file for this member.</param>
    public EnumValueAttribute(string value) => Value = value;

    /// <summary>String written to and matched from the file for this member.</summary>
    public string Value { get; }
}
