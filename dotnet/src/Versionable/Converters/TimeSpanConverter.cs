using Versionable.Errors;

namespace Versionable.Converters;

/// <summary>
/// <see cref="TimeSpan"/> ⇄ total seconds as a floating-point number.
/// </summary>
/// <remarks>
/// Python counterpart:
/// <c>registerConverter(datetime.timedelta, total_seconds, timedelta(seconds=v))</c> in
/// <c>src/versionable/_types.py</c>. A number, not an ISO 8601 duration string.
/// <para>
/// The value is rebuilt through ticks rather than
/// <see cref="TimeSpan.FromSeconds(double)"/>, which rounds its argument to the nearest
/// millisecond and would quietly discard the microseconds a Python <c>timedelta</c> can
/// carry.
/// </para>
/// <para>
/// A <see cref="double"/> holds a 53-bit significand, so a duration keeps tick precision out
/// to roughly ±104 days and degrades gradually beyond that; a <c>timedelta</c> of a thousand
/// years lands on the nearest few hundred nanoseconds. The wire format is Python's, and this
/// is its limit, not the encoding's.
/// </para>
/// <para>
/// <strong>Sub-microsecond values do not round the same way on both sides.</strong> A
/// <c>timedelta</c> resolves to the microsecond and rounds half-to-even when it is built from
/// a float, where this converter resolves to the 100 ns tick and rounds half-away-from-zero.
/// So a wire value of <c>0.0000005</c> becomes 5 ticks here and <c>0:00:00.000000</c> in
/// Python — the divergence is one of resolution first and tie-breaking second, and it is
/// bounded by half a microsecond. Every value either side actually <em>writes</em> is exact
/// in both: Python cannot produce a sub-microsecond duration at all, and a C# duration that
/// is a whole number of microseconds converts without a tie. This mirrors the sub-microsecond
/// caveat on <see cref="Iso8601"/>.
/// </para>
/// </remarks>
internal sealed class TimeSpanConverter : WireConverter<TimeSpan>
{
    /// <inheritdoc/>
    public override string SerializationName => "timedelta";

    /// <inheritdoc/>
    protected override object ToWireCore(TimeSpan value) => value.TotalSeconds;

    /// <inheritdoc/>
    protected override object FromWireCore(object wireValue, Type targetType)
    {
        double seconds = RequireDouble(wireValue);
        if (double.IsNaN(seconds) || double.IsInfinity(seconds))
        {
            throw new ConverterException($"timedelta cannot be {seconds}.");
        }

        double ticks = Math.Round(seconds * TimeSpan.TicksPerSecond, MidpointRounding.AwayFromZero);

        // The upper bound is exclusive because (double)long.MaxValue rounds up to 2^63: a
        // ticks value comparing equal to it would still overflow the cast below.
        if (ticks < long.MinValue || ticks >= long.MaxValue)
        {
            throw new ConverterException(
                $"{seconds} seconds is outside the range of a TimeSpan (±10,675,199 days).");
        }

        return TimeSpan.FromTicks((long)ticks);
    }
}
