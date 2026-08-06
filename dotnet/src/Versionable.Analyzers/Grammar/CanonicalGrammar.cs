using System.Collections.Generic;

namespace Versionable.Analyzers.Grammar;

/// <summary>
/// The fixed tables of <c>conformance/GRAMMAR.md</c>: scalar tokens, the C# types that map
/// onto a built-in converter's Serialization Name, and the closed <c>ndarray</c> dtype set.
/// </summary>
/// <remarks>
/// Keyed by fully-qualified metadata name rather than by resolved <c>ITypeSymbol</c> so the
/// renderer never needs a <c>Compilation</c> and never silently falls back when a reference
/// is missing — an unresolved <c>DateTime</c> would otherwise render as a bare
/// Serialization Name and produce a hash Python cannot reproduce.
/// </remarks>
internal static class CanonicalGrammar
{
    /// <summary>Separator between type arguments inside brackets — comma plus one space (§5).</summary>
    internal const string ArgumentSeparator = ", ";

    /// <summary>Separator between <c>name:type</c> pairs in the payload — a bare comma (§1).</summary>
    internal const string PairSeparator = ",";

    /// <summary>The canonical token for the null/absent value (§4).</summary>
    internal const string NoneToken = "None";

    /// <summary>Characters a wire name may not contain, because they structure the payload (§2).</summary>
    internal static readonly char[] ReservedWireNameCharacters = { ':', ',', '[', ']' };

    /// <summary>
    /// Types whose canonical name is a built-in converter's Serialization Name (§9), keyed by
    /// fully-qualified name. <c>decimal</c> lands here rather than on the <c>float</c> scalar:
    /// it is a converter type, not a numeric width.
    /// </summary>
    internal static readonly IReadOnlyDictionary<string, string> ConverterNames = new Dictionary<string, string>
    {
        ["System.DateTime"] = "datetime",
        ["System.DateTimeOffset"] = "datetime",
        ["System.DateOnly"] = "date",
        ["System.TimeOnly"] = "time",
        ["System.TimeSpan"] = "timedelta",
        ["System.Guid"] = "UUID",
        ["System.Decimal"] = "Decimal",
        ["System.Text.RegularExpressions.Regex"] = "Pattern",

        // The library's own path wrapper. Both namespaces are accepted because the type is
        // owned by the converter work; whichever it lands in, the canonical name is `Path`.
        // A [SerializationName] on the type would also work — that is checked first.
        ["Versionable.FilePath"] = "Path",
        ["Versionable.Converters.FilePath"] = "Path",
    };

    /// <summary>
    /// The closed <c>ndarray</c> dtype token table (§7), keyed by the fully-qualified element
    /// type of <c>Tensor&lt;T&gt;</c>. Array dtypes are hash-significant and are never
    /// width-erased, so anything outside this table is rejected rather than approximated
    /// (ADR-0002).
    /// </summary>
    internal static readonly IReadOnlyDictionary<string, string> DtypeTokens = new Dictionary<string, string>
    {
        ["System.Boolean"] = "bool",
        ["System.SByte"] = "int8",
        ["System.Int16"] = "int16",
        ["System.Int32"] = "int32",
        ["System.Int64"] = "int64",
        ["System.Byte"] = "uint8",
        ["System.UInt16"] = "uint16",
        ["System.UInt32"] = "uint32",
        ["System.UInt64"] = "uint64",
        ["System.Half"] = "float16",
        ["System.Single"] = "float32",
        ["System.Double"] = "float64",
        ["System.Numerics.Complex"] = "complex128",
    };

    /// <summary>Renders a <c>Union</c> from already-rendered members: deduplicated, ordinally sorted (§6).</summary>
    /// <param name="members">Rendered member strings, in any order.</param>
    /// <returns>The union rendering, or the single member when the set collapses to one.</returns>
    internal static string RenderUnion(IEnumerable<string> members)
    {
        SortedSet<string> sorted = new(System.StringComparer.Ordinal);
        foreach (string member in members)
        {
            sorted.Add(member);
        }

        if (sorted.Count == 1)
        {
            foreach (string only in sorted)
            {
                return only;
            }
        }

        return "Union[" + string.Join(ArgumentSeparator, sorted) + "]";
    }

    /// <summary>
    /// Escapes a <c>Literal</c> string member: backslash first, then single quote (§8). The
    /// payload is UTF-8, so nothing else is escaped.
    /// </summary>
    /// <param name="value">The raw string value.</param>
    /// <returns>The quoted, escaped rendering.</returns>
    internal static string QuoteLiteralString(string value) =>
        "'" + value.Replace("\\", "\\\\").Replace("'", "\\'") + "'";
}
