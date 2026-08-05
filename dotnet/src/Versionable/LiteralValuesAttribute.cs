namespace Versionable;

/// <summary>
/// Declares a closed set of allowed values for a plain property, the C# spelling of Python's
/// <c>Literal[...]</c>.
/// </summary>
/// <remarks>
/// Python counterpart: a field annotated <c>Literal['fast', 'slow']</c>, rendered by
/// <c>src/versionable/_hash.py</c> and validated by <c>deserialize()</c> in
/// <c>src/versionable/_types.py</c>. C# has no literal types, so the option set is declared
/// as attribute arguments on a normally-typed property
/// (<c>conformance/GRAMMAR.md</c> §8).
/// <para>
/// <b>Argument order is the canonical order and is hash-significant.</b> Options are
/// <em>not</em> sorted: <c>Literal['fast', 'slow']</c> and <c>Literal['slow', 'fast']</c> are
/// different schemas with different hashes. Reordering the arguments is a schema change.
/// </para>
/// <para>
/// Only <see cref="string"/>, <see cref="int"/>, <see cref="bool"/>, and <see langword="null"/>
/// options are representable; the grammar's member list is closed. Anything else — a float,
/// a byte array, an arbitrary object — must be rejected at schema-definition time rather than
/// rendered in a language-specific way. Enum-valued literal members have no C# counterpart in
/// v1.
/// </para>
/// <example>
/// <code>
/// [LiteralValues("fast", "slow", Fallback = "fast")]
/// public string Mode { get; init; } = "fast";
/// </code>
/// </example>
/// </remarks>
[AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, Inherited = false)]
public sealed class LiteralValuesAttribute : Attribute
{
    /// <summary>Initializes a new instance of the <see cref="LiteralValuesAttribute"/> class.</summary>
    /// <param name="values">
    /// The allowed values, in canonical order. Each must be a <see cref="string"/>,
    /// <see cref="int"/>, <see cref="bool"/>, or <see langword="null"/>.
    /// </param>
    public LiteralValuesAttribute(params object?[] values) => Values = values;

    /// <summary>The allowed values, in canonical order. Hash-significant.</summary>
    public IReadOnlyList<object?> Values { get; }

    /// <summary>
    /// Value substituted when a file holds something outside <see cref="Values"/>, instead of
    /// failing the load. Must itself be one of <see cref="Values"/>.
    /// </summary>
    /// <remarks>
    /// Python counterpart: <c>literalFallback()</c> used as a field default in
    /// <c>src/versionable/_types.py</c>. Not hash-significant — it governs load behavior, not
    /// the schema. Leaving it unset means an out-of-range value is an error whenever the
    /// declaring type has <c>ValidateLiterals</c> set.
    /// </remarks>
    public object? Fallback { get; init; }
}
