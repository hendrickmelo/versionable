namespace Versionable.Converters;

/// <summary>
/// Format strings shared by the four temporal converters.
/// </summary>
/// <remarks>
/// The target is Python's <c>isoformat()</c>, which writes either no fractional part or
/// exactly six digits, because <c>datetime</c> resolution is the microsecond.
/// <para>
/// .NET's resolution is the 100-nanosecond tick, so a value can carry a seventh digit Python
/// cannot express. <see cref="FractionPattern"/> writes six digits whenever the value happens
/// to be a whole number of microseconds — which every Python-written and almost every
/// C#-written value is — and only widens to seven when the tick is not, so ordinary files
/// come out byte-identical to Python's rather than merely equivalent.
/// </para>
/// <para>
/// <strong>Sub-microsecond ticks do not survive a round trip through Python.</strong>
/// <c>datetime.fromisoformat</c> truncates the seventh digit, so C# → Python → C# loses up to
/// 99 ns. This is inherent to the interchange format, not to the encoding: there is no
/// representation of a 100 ns tick in a Python <c>datetime</c>.
/// </para>
/// </remarks>
internal static class Iso8601
{
    /// <summary>Date pattern: <c>2026-08-05</c>.</summary>
    internal const string DatePattern = "yyyy-MM-dd";

    /// <summary>Date-and-time pattern without the fractional part.</summary>
    internal const string DateTimePattern = "yyyy-MM-ddTHH:mm:ss";

    /// <summary>Time-of-day pattern without the fractional part.</summary>
    internal const string TimePattern = "HH:mm:ss";

    /// <summary>UTC offset pattern: <c>-05:00</c>.</summary>
    internal const string OffsetPattern = "zzz";

    // The time-of-day tails fromisoformat accepts, coarsest first. Declared before the
    // pattern tables below because static initializers run in textual order.
    private static readonly string[] _timeSuffixes =
    [
        "HH:mm",
        TimePattern,
        TimePattern + ".FFFFFFF",
    ];

    /// <summary>The exact date-time forms <c>datetime.fromisoformat</c> reads, without an offset.</summary>
    /// <remarks>
    /// Parsing is exact, as it is for <c>Guid</c> and <c>DateOnly</c>: .NET's general
    /// <c>TryParse</c> accepts culture-shaped input such as <c>08/05/2026</c> and
    /// <c>Aug 5 2026</c>, which no writer emits and which Python cannot read — admitting them
    /// would let a file load in C# and fail in Python.
    /// <para>
    /// The set is <c>fromisoformat</c>'s, not <c>isoformat</c>'s, and the difference is
    /// deliberate. A writer only ever produces <c>yyyy-MM-ddTHH:mm:ss</c> with an optional
    /// six-digit fraction, so the extra forms are reachable only by hand-editing — but
    /// rejecting a form Python accepts is the one failure that breaks interchange in the
    /// direction that matters, and each extra pattern is unambiguous.
    /// </para>
    /// <para>
    /// <c>FFFFFFF</c> matches one to seven fractional digits, covering Python's six and the
    /// seventh .NET can produce; the fraction-less forms need their own patterns because the
    /// decimal point is a literal. Python truncates past six digits.
    /// </para>
    /// <para>
    /// One exclusion is deliberate: the <strong>compact</strong> forms (<c>20260805</c>,
    /// <c>20260805T143015</c>), which <c>fromisoformat</c> has read since Python 3.11. They
    /// are excluded across every temporal converter rather than in some of them, because
    /// <see cref="DateOnly"/> and <see cref="TimeOnly"/> exclude them too and a set where
    /// <c>datetime</c> accepted a compact string but <c>date</c> did not would be harder to
    /// predict than one that says no everywhere. Nothing on either side writes them.
    /// </para>
    /// </remarks>
    internal static readonly string[] NaiveDateTimePatterns =
    [
        DatePattern,
        .. _timeSuffixes.Select(suffix => DatePattern + "T" + suffix),

        // fromisoformat takes any single separator character; a space is the one that occurs
        // in practice, because it is what str(datetime) prints.
        .. _timeSuffixes.Select(suffix => DatePattern + " " + suffix),
    ];

    /// <summary>The same forms with a UTC offset, which <c>datetime.isoformat()</c> writes for an aware value.</summary>
    /// <remarks>
    /// <c>zzz</c> reads <c>±HH:MM</c> and <c>±HHMM</c> alike. A bare <c>±HH</c>, which
    /// <c>fromisoformat</c> also accepts, is not covered; nothing writes it.
    /// </remarks>
    internal static readonly string[] AwareDateTimePatterns =
    [
        .. _timeSuffixes.Select(suffix => DatePattern + "T" + suffix + OffsetPattern),
        .. _timeSuffixes.Select(suffix => DatePattern + " " + suffix + OffsetPattern),
    ];

    /// <summary>Whether an ISO 8601 date-time string carries a UTC offset or a <c>Z</c>.</summary>
    /// <param name="text">The wire value.</param>
    /// <returns><see langword="true"/> when a time zone is designated.</returns>
    /// <remarks>
    /// Used only to choose between two error messages, so it looks past the date — whose own
    /// hyphens must not be mistaken for a negative offset — and accepts either separator.
    /// </remarks>
    internal static bool CarriesOffset(string text) =>
        text.Length > DatePattern.Length
        && text.AsSpan(DatePattern.Length + 1).IndexOfAny("+-Zz") >= 0;

    /// <summary>Rewrites a trailing <c>Z</c> as the <c>+00:00</c> the patterns expect.</summary>
    /// <param name="text">The wire value.</param>
    /// <returns>The value with any Zulu designator expanded.</returns>
    /// <remarks>
    /// Python's <c>isoformat()</c> never writes <c>Z</c>, but <c>fromisoformat</c> reads it and
    /// so must this. Rewriting is simpler than a second set of patterns with a literal
    /// <c>'Z'</c>, which would need <c>AssumeUniversal</c> to land on the right offset.
    /// </remarks>
    internal static string NormalizeZulu(string text) =>
        text.Length > 0 && (text[^1] == 'Z' || text[^1] == 'z')
            ? string.Concat(text.AsSpan(0, text.Length - 1), "+00:00")
            : text;

    /// <summary>Returns the fractional-seconds pattern for <paramref name="ticks"/>.</summary>
    /// <param name="ticks">The value's tick count; only the sub-second remainder is read.</param>
    /// <returns>
    /// <c>".ffffff"</c>, <c>".fffffff"</c>, or the empty string when the value falls on a whole
    /// second. Fixed-width <c>f</c> rather than trimming <c>F</c>, so a value such as
    /// <c>.500000</c> keeps the six digits Python writes.
    /// </returns>
    internal static string FractionPattern(long ticks)
    {
        long subsecond = Math.Abs(ticks % TimeSpan.TicksPerSecond);
        if (subsecond == 0)
        {
            return string.Empty;
        }

        return subsecond % 10 == 0 ? ".ffffff" : ".fffffff";
    }
}
