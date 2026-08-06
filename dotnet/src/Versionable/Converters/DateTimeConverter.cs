using System.Globalization;
using Versionable.Errors;

namespace Versionable.Converters;

/// <summary>
/// <see cref="DateTime"/> ⇄ an ISO 8601 date-and-time string with no UTC offset.
/// </summary>
/// <remarks>
/// Python counterpart: <c>registerConverter(datetime.datetime, isoformat, fromisoformat)</c>
/// in <c>src/versionable/_types.py</c>. <see cref="DateTime"/> mirrors a <em>naive</em>
/// Python datetime; the aware form is <see cref="DateTimeOffset"/>
/// (<see cref="DateTimeOffsetConverter"/>). Both canonicalise to <c>datetime</c>, so the two
/// are interchangeable as far as the schema hash is concerned and a Python schema does not
/// have to say which it meant.
/// <para>
/// A wire value that <em>does</em> carry an offset or a trailing <c>Z</c> is rejected rather
/// than shifted into local time or stripped. Both silent alternatives move the instant a
/// value denotes, and which way it moves would depend on the reading machine's time zone.
/// The fix is to declare the field <see cref="DateTimeOffset"/>, which reads the same file.
/// </para>
/// <para>
/// <see cref="DateTime.Kind"/> is not part of the wire format and is not preserved: a value
/// written as <see cref="DateTimeKind.Utc"/> reads back as
/// <see cref="DateTimeKind.Unspecified"/>, because the string it wrote carried no offset for
/// it to be recovered from. Python has the same property — a naive datetime carries no
/// <c>tzinfo</c>.
/// </para>
/// </remarks>
internal sealed class DateTimeConverter : WireConverter<DateTime>
{
    /// <inheritdoc/>
    public override string SerializationName => "datetime";

    /// <inheritdoc/>
    protected override object ToWireCore(DateTime value) =>
        value.ToString(
            Iso8601.DateTimePattern + Iso8601.FractionPattern(value.Ticks),
            CultureInfo.InvariantCulture);

    /// <inheritdoc/>
    protected override object FromWireCore(object wireValue, Type targetType)
    {
        string text = RequireString(wireValue);
        if (DateTime.TryParseExact(
                text,
                Iso8601.NaiveDateTimePatterns,
                CultureInfo.InvariantCulture,
                DateTimeStyles.None,
                out DateTime result))
        {
            return result;
        }

        if (Iso8601.CarriesOffset(text))
        {
            throw new ConverterException(
                $"'{text}' carries a UTC offset, so it is an aware datetime; DateTime models the "
                + "naive form. Declare the field as DateTimeOffset.");
        }

        throw new ConverterException(
            $"'{text}' is not an ISO 8601 date-and-time (yyyy-MM-dd, optionally followed by "
            + "THH:mm[:ss[.fraction]]).");
    }
}
