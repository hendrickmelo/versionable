using System.Text;
using System.Text.Json;

namespace Versionable.Backends.Json;

/// <summary>
/// Python's non-finite number tokens, which are not JSON.
/// </summary>
/// <remarks>
/// <c>json.dumps</c> writes <c>NaN</c>, <c>Infinity</c>, and <c>-Infinity</c> as bare tokens by
/// default, and <c>json.loads</c> reads them back. RFC 8259 has no such literals, so
/// <c>System.Text.Json</c> refuses them in both directions: <c>Utf8JsonWriter.WriteNumberValue</c>
/// throws on a non-finite double, and <c>JsonDocument.Parse</c> fails on the token.
/// <para>
/// Interchange wins over strictness here. A physics or instrument schema with a
/// <c>float('nan')</c> in it is ordinary, Python has been writing those files for years, and a C#
/// build that could not read them would fail at the one thing this port exists for. So the writer
/// emits the same three tokens Python does, and the reader accepts them.
/// </para>
/// <para>
/// <b>The output is therefore not strict JSON</b>, exactly as Python's is not. A third-party
/// parser that rejects the tokens will reject a file holding a non-finite value — from either
/// language.
/// </para>
/// <para>
/// <c>JsonNumberHandling.AllowNamedFloatingPointLiterals</c> does not help: it is a serializer
/// setting, and it accepts the <em>quoted</em> forms (<c>"NaN"</c>), which is not what Python
/// writes. <see cref="JsonDocument"/> has no equivalent knob, hence the repair pass below.
/// </para>
/// </remarks>
internal static class NonFiniteJson
{
    /// <summary>
    /// Sentinel a repaired <c>NaN</c> token becomes. The NUL prefix keeps it unwritable from a schema.
    /// </summary>
    internal const string NanSentinel = "\0NaN";

    /// <summary>Sentinel a repaired <c>Infinity</c> token becomes.</summary>
    internal const string PositiveInfinitySentinel = "\0Infinity";

    /// <summary>Sentinel a repaired <c>-Infinity</c> token becomes.</summary>
    internal const string NegativeInfinitySentinel = "\0-Infinity";

    private static readonly (string Token, string Sentinel)[] _tokens =
    [
        // Longest first: `-Infinity` has to match before `Infinity` would, and the leading `-`
        // means it can only ever start where a value starts.
        ("-Infinity", NegativeInfinitySentinel),
        ("Infinity", PositiveInfinitySentinel),
        ("NaN", NanSentinel),
    ];

    /// <summary>
    /// The token Python writes for <paramref name="value"/>, or <see langword="null"/> when finite.
    /// </summary>
    /// <param name="value">The value about to be written.</param>
    /// <returns>The raw JSON token to emit, or <see langword="null"/>.</returns>
    internal static string? TokenFor(double value) =>
        double.IsNaN(value) ? "NaN"
        : double.IsPositiveInfinity(value) ? "Infinity"
        : double.IsNegativeInfinity(value) ? "-Infinity"
        : null;

    /// <summary>Maps a repaired sentinel string back to the value it stood for.</summary>
    /// <param name="text">A string read from a repaired document.</param>
    /// <param name="value">The non-finite value, when <paramref name="text"/> is a sentinel.</param>
    /// <returns><see langword="true"/> when <paramref name="text"/> was a sentinel.</returns>
    internal static bool TryFromSentinel(string text, out double value)
    {
        switch (text)
        {
            case NanSentinel:
                value = double.NaN;
                return true;
            case PositiveInfinitySentinel:
                value = double.PositiveInfinity;
                return true;
            case NegativeInfinitySentinel:
                value = double.NegativeInfinity;
                return true;
            default:
                value = 0;
                return false;
        }
    }

    /// <summary>
    /// Rewrites Python's bare non-finite tokens as quoted sentinels so a strict parser can read
    /// the document.
    /// </summary>
    /// <remarks>
    /// Runs only after a strict parse has already failed, so a well-formed file never pays for it.
    /// String contents are skipped — with escape handling — so a field whose <em>value</em> is the
    /// word <c>NaN</c> is left alone.
    /// </remarks>
    /// <param name="utf8">The document bytes.</param>
    /// <param name="repaired">The rewritten bytes, when at least one token was found.</param>
    /// <returns><see langword="true"/> when something was rewritten.</returns>
    internal static bool TryRepair(ReadOnlySpan<byte> utf8, out byte[]? repaired)
    {
        repaired = null;
        List<byte> output = new(utf8.Length + 16);
        bool changed = false;
        bool inString = false;

        for (int index = 0; index < utf8.Length; index++)
        {
            byte current = utf8[index];

            if (inString)
            {
                output.Add(current);
                if (current == (byte)'\\' && index + 1 < utf8.Length)
                {
                    output.Add(utf8[++index]);
                }
                else if (current == (byte)'"')
                {
                    inString = false;
                }

                continue;
            }

            if (current == (byte)'"')
            {
                inString = true;
                output.Add(current);
                continue;
            }

            if (TryMatchToken(utf8, index, out string? sentinel, out int length))
            {
                output.AddRange(Encoding.UTF8.GetBytes(Quote(sentinel!)));
                index += length - 1;
                changed = true;
                continue;
            }

            output.Add(current);
        }

        if (!changed)
        {
            return false;
        }

        repaired = [.. output];
        return true;
    }

    private static bool TryMatchToken(ReadOnlySpan<byte> utf8, int index, out string? sentinel, out int length)
    {
        foreach ((string token, string replacement) in _tokens)
        {
            if (index + token.Length <= utf8.Length && Matches(utf8, index, token))
            {
                sentinel = replacement;
                length = token.Length;
                return true;
            }
        }

        sentinel = null;
        length = 0;
        return false;
    }

    private static bool Matches(ReadOnlySpan<byte> utf8, int index, string token)
    {
        for (int offset = 0; offset < token.Length; offset++)
        {
            if (utf8[index + offset] != (byte)token[offset])
            {
                return false;
            }
        }

        return true;
    }

    private static string Quote(string sentinel) =>
        // The sentinels hold a NUL, which JSON requires escaped; nothing else in them needs it.
        $"\"\\u0000{sentinel[1..]}\"";
}
