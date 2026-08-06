using Versionable.Errors;

namespace Versionable.Converters;

/// <summary>
/// <see cref="Guid"/> ⇄ a lowercase hyphenated UUID string.
/// </summary>
/// <remarks>
/// Python counterpart: <c>registerConverter(uuid.UUID, str, uuid.UUID)</c> in
/// <c>src/versionable/_types.py</c>. <c>str(UUID)</c> is always the lowercase 8-4-4-4-12
/// form, which is what <c>"D"</c> produces here.
/// <para>
/// Reads accept upper-case hex — <c>TryParseExact</c> with <c>"D"</c> is case-insensitive —
/// but not the braced, parenthesised, or unhyphenated forms .NET's plain <c>Guid.Parse</c>
/// takes. Python's <c>uuid.UUID(str)</c> is looser still; nothing on either side ever
/// <em>writes</em> those forms, and accepting them here would mean C# silently reading files
/// Python's own reader would also take but that no writer produces.
/// </para>
/// </remarks>
internal sealed class GuidConverter : WireConverter<Guid>
{
    /// <inheritdoc/>
    public override string SerializationName => "UUID";

    /// <inheritdoc/>
    protected override object ToWireCore(Guid value) => value.ToString("D", null);

    /// <inheritdoc/>
    protected override object FromWireCore(object wireValue, Type targetType)
    {
        string text = RequireString(wireValue);
        if (!Guid.TryParseExact(text, "D", out Guid result))
        {
            throw new ConverterException(
                $"'{text}' is not a hyphenated UUID (8-4-4-4-12 hex digits).");
        }

        return result;
    }
}
