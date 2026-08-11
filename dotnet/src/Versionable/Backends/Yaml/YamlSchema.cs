using System.Globalization;
using System.Numerics;
using System.Text.RegularExpressions;

namespace Versionable.Backends.Yaml;

/// <summary>
/// PyYAML's implicit-typing schema: what an unquoted scalar means, and what therefore has to be
/// quoted when writing one.
/// </summary>
/// <remarks>
/// Python counterpart: <c>yaml.resolver.Resolver</c> and the <c>construct_yaml_*</c> methods of
/// <c>yaml.constructor.SafeConstructor</c>. The regular expressions below are transcribed
/// character for character from PyYAML 6's <c>add_implicit_resolver</c> calls and are matched with
/// <see cref="RegexOptions.IgnorePatternWhitespace"/>, which is the <c>re.X</c> those calls use —
/// so they can be diffed against the Python source rather than re-derived.
/// <para>
/// <b>This is YAML 1.1, not 1.2</b>, because PyYAML is. The differences are load-bearing for
/// interchange: <c>yes</c>/<c>no</c>/<c>on</c>/<c>off</c> are booleans, <c>012</c> is octal 10
/// while the 1.2 spelling <c>0o12</c> is a plain string, <c>1_000</c> is 1000, <c>1e5</c> is a
/// <em>string</em> (PyYAML's float pattern requires a sign on the exponent), and <c>1:30</c> is
/// the sexagesimal integer 90. A reader that applied 1.2 rules would disagree with Python about
/// files Python wrote, and a writer that did would emit <c>yes</c> for the string "yes".
/// </para>
/// <para>
/// <b>One deliberate divergence: timestamps stay strings.</b> PyYAML resolves an unquoted
/// <c>2026-08-05</c> to a <c>datetime.date</c>. Nothing in versionable wants that — every
/// temporal value reaches the wire as a string through a converter (<c>$datetime</c>,
/// <c>$date</c>, <c>$time</c> in the golden manifests), and Python's own writer quotes them for
/// exactly this reason. Handing the converters a <see cref="DateTime"/> instead of the text they
/// parse would only give them something to undo, so a timestamp-shaped scalar is read as its
/// text. It is still reported as implicitly typed, because the <em>writer</em> must quote it or
/// Python would read a date where C# wrote a string.
/// </para>
/// </remarks>
internal static partial class YamlSchema
{
    /// <summary>The merge key, which pulls another mapping's entries into this one.</summary>
    internal const string MergeKey = "<<";

    /// <summary>
    /// Resolves a plain (unquoted) scalar, PyYAML's way.
    /// </summary>
    /// <param name="text">The scalar's text, exactly as the parser produced it.</param>
    /// <param name="value">
    /// What the scalar means. For a timestamp or the <c>=</c> value key this is
    /// <paramref name="text"/> itself — see the note on the type.
    /// </param>
    /// <returns>
    /// <see langword="true"/> when the text carries an implicit type, which is both what a reader
    /// needs in order to produce the right CLR value and what forces a writer to quote a string
    /// spelled the same way. <see langword="false"/> means the text is just itself.
    /// </returns>
    internal static bool TryResolveImplicit(string text, out object? value)
    {
        // Order is PyYAML's registration order (bool, float, int, merge, null, timestamp, value),
        // which is the order its per-first-character resolver lists are tried in. The patterns are
        // anchored and mutually exclusive, but keeping the order makes the two readable side by
        // side.
        if (BoolPattern().IsMatch(text))
        {
            value = ParseBool(text);
            return true;
        }

        if (FloatPattern().IsMatch(text))
        {
            value = ParseFloat(text);
            return true;
        }

        if (IntPattern().IsMatch(text))
        {
            value = ParseInt(text);
            return true;
        }

        if (text is MergeKey or "=")
        {
            // The merge key is handled where mappings are built; `=` has no SafeConstructor at all
            // in Python (it raises), so it is kept as text. Both are reported as implicit so that
            // a field whose *value* is the string "<<" is written quoted.
            value = text;
            return true;
        }

        if (NullPattern().IsMatch(text))
        {
            value = null;
            return true;
        }

        if (TimestampPattern().IsMatch(text))
        {
            value = text;
            return true;
        }

        value = text;
        return false;
    }

    /// <summary>Whether a string can be written unquoted and read back as the same string.</summary>
    /// <param name="text">The string about to be written.</param>
    /// <returns><see langword="true"/> when no implicit type claims it.</returns>
    internal static bool IsPlainSafe(string text) => !TryResolveImplicit(text, out _);

    /// <summary>Renders a double the way PyYAML's float representer does.</summary>
    /// <remarks>
    /// Two rules, both of which matter. Non-finite values are <c>.inf</c>, <c>-.inf</c>, and
    /// <c>.nan</c> — YAML has spellings for these, unlike JSON, so there is no repair pass here.
    /// And a whole number gets a <c>.0</c>: .NET renders 2.0 as <c>2</c>, which YAML would read
    /// back as an integer, and <c>1E+30</c> as an exponent with no point, which PyYAML's float
    /// pattern does not even match — it would come back a string. Python's representer inserts the
    /// same <c>.0</c> for the same reason, and spells the exponent in lower case; both are
    /// <see cref="EnsurePoint"/>'s job.
    /// </remarks>
    /// <param name="value">The value to render.</param>
    /// <returns>A plain scalar that reads back as <paramref name="value"/>.</returns>
    internal static string RenderReal(double value)
    {
        if (double.IsNaN(value))
        {
            return ".nan";
        }

        if (double.IsPositiveInfinity(value))
        {
            return ".inf";
        }

        if (double.IsNegativeInfinity(value))
        {
            return "-.inf";
        }

        return EnsurePoint(value.ToString("R", CultureInfo.InvariantCulture));
    }

    /// <summary>Renders a float, keeping its own shortest round-trip form.</summary>
    /// <remarks>
    /// Widening to <see cref="double"/> first would turn <c>0.1f</c> into
    /// <c>0.10000000149011612</c>: the nearest double to the nearest float, rendered to full
    /// double precision. Formatting the float directly gives <c>0.1</c>, which reads back as the
    /// same float.
    /// </remarks>
    /// <param name="value">The value to render.</param>
    /// <returns>A plain scalar that reads back as <paramref name="value"/>.</returns>
    internal static string RenderReal(float value)
    {
        if (float.IsNaN(value))
        {
            return ".nan";
        }

        if (float.IsPositiveInfinity(value))
        {
            return ".inf";
        }

        if (float.IsNegativeInfinity(value))
        {
            return "-.inf";
        }

        return EnsurePoint(value.ToString("R", CultureInfo.InvariantCulture));
    }

    /// <summary>Renders a decimal, which YAML has no type for beyond its float.</summary>
    /// <remarks>
    /// Unreachable through a declared <c>decimal</c> field — <c>DecimalConverter</c> writes those
    /// as strings, which is what keeps their exactness — but a backend must not throw on a CLR
    /// type the walker legitimately passed through, so this renders it as a YAML float.
    /// </remarks>
    /// <param name="value">The value to render.</param>
    /// <returns>A plain scalar.</returns>
    internal static string RenderReal(decimal value) =>
        EnsurePoint(value.ToString(CultureInfo.InvariantCulture));

    /// <summary>
    /// Gives a rendered number the decimal point PyYAML's float pattern requires, and spells the
    /// exponent the way PyYAML spells it.
    /// </summary>
    /// <remarks>
    /// .NET renders 1e30 as <c>1E+30</c>; Python's <c>repr(1e30)</c> is <c>1e+30</c> and its float
    /// representer adds the <c>.0</c>, so the same value is <c>1.0e+30</c> in a Python-written
    /// file. Both spellings read back as the same double in both languages — the pattern accepts
    /// <c>[eE]</c> — so the lower case is emitter parity, not correctness: it is what lets a
    /// C#-written file diff clean against a Python-written one.
    /// </remarks>
    /// <param name="rendered">A number as the CLR formatted it.</param>
    /// <returns>The same number in PyYAML's spelling.</returns>
    private static string EnsurePoint(string rendered)
    {
        int exponent = rendered.IndexOfAny(['e', 'E']);
        if (exponent < 0)
        {
            return rendered.Contains('.', StringComparison.Ordinal) ? rendered : rendered + ".0";
        }

        ReadOnlySpan<char> mantissa = rendered.AsSpan(0, exponent);
        ReadOnlySpan<char> tail = rendered.AsSpan(exponent + 1);
        return mantissa.Contains('.')
            ? string.Concat(mantissa, "e", tail)
            : string.Concat(mantissa, ".0e", tail);
    }

    private static bool ParseBool(string text) =>
        text.ToLowerInvariant() is "yes" or "true" or "on";

    private static object ParseFloat(string text)
    {
        // construct_yaml_float: underscores dropped, case folded, sign taken off the front.
        string body = text.Replace("_", string.Empty, StringComparison.Ordinal).ToLowerInvariant();
        double sign = body.StartsWith('-') ? -1 : 1;
        if (body.StartsWith('-') || body.StartsWith('+'))
        {
            body = body[1..];
        }

        if (body == ".inf")
        {
            return sign * double.PositiveInfinity;
        }

        if (body == ".nan")
        {
            // Unsigned on purpose: Python returns its own nan whatever the sign was.
            return double.NaN;
        }

        if (body.Contains(':', StringComparison.Ordinal))
        {
            double total = 0;
            foreach (string part in body.Split(':'))
            {
                total = (total * 60) + double.Parse(part, CultureInfo.InvariantCulture);
            }

            return sign * total;
        }

        return sign * double.Parse(body, NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    private static object ParseInt(string text)
    {
        // construct_yaml_int, branch for branch.
        string body = text.Replace("_", string.Empty, StringComparison.Ordinal);
        bool negative = body.StartsWith('-');
        if (negative || body.StartsWith('+'))
        {
            body = body[1..];
        }

        BigInteger magnitude =
            body == "0" ? BigInteger.Zero
            : body.StartsWith("0b", StringComparison.Ordinal) ? FromDigits(body[2..], 2)
            : body.StartsWith("0x", StringComparison.Ordinal) ? FromDigits(body[2..], 16)
            : body.StartsWith('0') ? FromDigits(body, 8)
            : body.Contains(':', StringComparison.Ordinal) ? FromSexagesimal(body)
            : BigInteger.Parse(body, CultureInfo.InvariantCulture);

        BigInteger result = negative ? -magnitude : magnitude;

        // Python's int is unbounded, so the file may hold a value no CLR integer can take. The
        // narrowing order matches the JSON backend's: long, then ulong for the top half of the
        // 64-bit range, then double — which loses bits but is the only thing left, and is what
        // reaching a field would have done anyway.
        if (result >= long.MinValue && result <= long.MaxValue)
        {
            return (long)result;
        }

        if (result >= 0 && result <= ulong.MaxValue)
        {
            return (ulong)result;
        }

        return (double)result;
    }

    private static BigInteger FromDigits(string digits, int radix)
    {
        // The pattern has already restricted the digit set, so no validation is needed here — only
        // the letters of base 16 need mapping, and they arrive in either case.
        BigInteger total = BigInteger.Zero;
        foreach (char digit in digits)
        {
            int place = digit <= '9' ? digit - '0' : (char.ToLowerInvariant(digit) - 'a') + 10;
            total = (total * radix) + place;
        }

        return total;
    }

    private static BigInteger FromSexagesimal(string body)
    {
        BigInteger total = BigInteger.Zero;
        foreach (string part in body.Split(':'))
        {
            total = (total * 60) + BigInteger.Parse(part, CultureInfo.InvariantCulture);
        }

        return total;
    }

    [GeneratedRegex(
        """
        ^(?:yes|Yes|YES|no|No|NO
            |true|True|TRUE|false|False|FALSE
            |on|On|ON|off|Off|OFF)$
        """,
        RegexOptions.IgnorePatternWhitespace | RegexOptions.CultureInvariant)]
    private static partial Regex BoolPattern();

    [GeneratedRegex(
        """
        ^(?: ~
            |null|Null|NULL
            | )$
        """,
        RegexOptions.IgnorePatternWhitespace | RegexOptions.CultureInvariant)]
    private static partial Regex NullPattern();

    [GeneratedRegex(
        """
        ^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
            |\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
            |[-+]?\.(?:inf|Inf|INF)
            |\.(?:nan|NaN|NAN))$
        """,
        RegexOptions.IgnorePatternWhitespace | RegexOptions.CultureInvariant)]
    private static partial Regex FloatPattern();

    [GeneratedRegex(
        """
        ^(?:[-+]?0b[0-1_]+
            |[-+]?0[0-7_]+
            |[-+]?(?:0|[1-9][0-9_]*)
            |[-+]?0x[0-9a-fA-F_]+
            |[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$
        """,
        RegexOptions.IgnorePatternWhitespace | RegexOptions.CultureInvariant)]
    private static partial Regex IntPattern();

    [GeneratedRegex(
        """
        ^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]
            |[0-9][0-9][0-9][0-9] -[0-9][0-9]? -[0-9][0-9]?
             (?:[Tt]|[ \t]+)[0-9][0-9]?
             :[0-9][0-9] :[0-9][0-9] (?:\.[0-9]*)?
             (?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)$
        """,
        RegexOptions.IgnorePatternWhitespace | RegexOptions.CultureInvariant)]
    private static partial Regex TimestampPattern();
}
