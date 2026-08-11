using System.Globalization;
using Versionable.Errors;

namespace Versionable.Converters;

/// <summary>
/// <see cref="TimeOnly"/> ⇄ an ISO 8601 time-of-day string, <c>23:59:58.500000</c>.
/// </summary>
/// <remarks>
/// Python counterpart: <c>registerConverter(datetime.time, isoformat, fromisoformat)</c> in
/// <c>src/versionable/_types.py</c>.
/// <para>
/// Python's <c>time</c> may carry a <c>tzinfo</c>, in which case <c>isoformat()</c> appends an
/// offset. <see cref="TimeOnly"/> has no offset to put it in, so such a value is rejected
/// rather than silently dropped — an offset-bearing time-of-day has no C# counterpart in v1.
/// </para>
/// </remarks>
internal sealed class TimeOnlyConverter : WireConverter<TimeOnly>
{
    // Fixed patterns rather than TimeOnly.TryParse: the general parse accepts locale-shaped
    // input such as '11:59 PM' and, worse, a bare 'HH:mm', which would read a file no writer
    // produces. The three forms below are exactly what Python's time.isoformat() emits, plus
    // the trimmed fractions .NET itself can write.
    private static readonly string[] _patterns =
    [
        Iso8601.TimePattern,
        Iso8601.TimePattern + ".FFFFFFF",
    ];

    /// <inheritdoc/>
    public override string SerializationName => "time";

    /// <inheritdoc/>
    protected override object ToWireCore(TimeOnly value) =>
        value.ToString(
            Iso8601.TimePattern + Iso8601.FractionPattern(value.Ticks),
            CultureInfo.InvariantCulture);

    /// <inheritdoc/>
    protected override object FromWireCore(object wireValue, Type targetType)
    {
        string text = RequireString(wireValue);
        if (!TimeOnly.TryParseExact(
                text,
                _patterns,
                CultureInfo.InvariantCulture,
                DateTimeStyles.None,
                out TimeOnly result))
        {
            throw new ConverterException(
                $"'{text}' is not an ISO 8601 time of day (HH:mm:ss with an optional fraction). "
                + "A time carrying a UTC offset has no TimeOnly representation.");
        }

        return result;
    }
}
