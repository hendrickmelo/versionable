using System.Globalization;
using Versionable.Errors;

namespace Versionable.Converters;

/// <summary>
/// <see cref="DateOnly"/> ⇄ an ISO 8601 date string, <c>2026-08-05</c>.
/// </summary>
/// <remarks>
/// Python counterpart: <c>registerConverter(datetime.date, isoformat, fromisoformat)</c> in
/// <c>src/versionable/_types.py</c>. Parsed exactly, so the compact <c>20260805</c> form
/// <c>date.fromisoformat</c> also accepts is rejected here — nothing writes it, and admitting
/// it would let a file be read by one implementation and not the other.
/// </remarks>
internal sealed class DateOnlyConverter : WireConverter<DateOnly>
{
    /// <inheritdoc/>
    public override string SerializationName => "date";

    /// <inheritdoc/>
    protected override object ToWireCore(DateOnly value) =>
        value.ToString(Iso8601.DatePattern, CultureInfo.InvariantCulture);

    /// <inheritdoc/>
    protected override object FromWireCore(object wireValue, Type targetType)
    {
        string text = RequireString(wireValue);
        if (!DateOnly.TryParseExact(
                text,
                Iso8601.DatePattern,
                CultureInfo.InvariantCulture,
                DateTimeStyles.None,
                out DateOnly result))
        {
            throw new ConverterException($"'{text}' is not an ISO 8601 date (yyyy-MM-dd).");
        }

        return result;
    }
}
