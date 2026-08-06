using System.Globalization;
using Versionable.Errors;

namespace Versionable.Converters;

/// <summary>
/// <see cref="DateTimeOffset"/> ⇄ an ISO 8601 date-and-time string with a UTC offset.
/// </summary>
/// <remarks>
/// The aware half of Python's <c>datetime</c> (see <see cref="DateTimeConverter"/> for the
/// naive half). Python writes the offset as <c>±HH:MM</c>, which is what <c>zzz</c> produces;
/// UTC is <c>+00:00</c> on both sides, never <c>Z</c>.
/// <para>
/// A wire value with no offset is rejected. .NET would happily parse it by assuming the
/// reading machine's local offset, which would make the same file load to different instants
/// on different machines. Declare the field <see cref="DateTime"/> for naive values.
/// </para>
/// <para>
/// Python's <c>tzinfo</c> is richer than an offset — a <c>ZoneInfo</c> knows the zone's rules,
/// not just its current shift — but only the offset reaches the wire, so nothing is lost in
/// interchange that Python itself would have kept.
/// </para>
/// </remarks>
internal sealed class DateTimeOffsetConverter : WireConverter<DateTimeOffset>
{
    /// <inheritdoc/>
    public override string SerializationName => "datetime";

    /// <inheritdoc/>
    protected override object ToWireCore(DateTimeOffset value) =>
        value.ToString(
            Iso8601.DateTimePattern + Iso8601.FractionPattern(value.Ticks) + Iso8601.OffsetPattern,
            CultureInfo.InvariantCulture);

    /// <inheritdoc/>
    protected override object FromWireCore(object wireValue, Type targetType)
    {
        string text = Iso8601.NormalizeZulu(RequireString(wireValue));
        if (DateTimeOffset.TryParseExact(
                text,
                Iso8601.AwareDateTimePatterns,
                CultureInfo.InvariantCulture,
                DateTimeStyles.None,
                out DateTimeOffset result))
        {
            return result;
        }

        if (!Iso8601.CarriesOffset(text))
        {
            throw new ConverterException(
                $"'{text}' has no UTC offset, so it is a naive datetime; DateTimeOffset models the "
                + "aware form. Declare the field as DateTime.");
        }

        throw new ConverterException(
            $"'{text}' is not an ISO 8601 date-and-time with a UTC offset.");
    }
}
