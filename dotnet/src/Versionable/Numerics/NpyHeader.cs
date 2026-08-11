using System.Globalization;
using Versionable.Errors;

namespace Versionable.Numerics;

/// <summary>
/// The three fields of an NPY header dict, parsed out of its Python-literal source.
/// </summary>
/// <remarks>
/// The header is a Python dict literal — <c>{'descr': '&lt;f8', 'fortran_order': False,
/// 'shape': (4,), }</c> — which numpy itself parses with <c>ast.literal_eval</c>. This reads
/// the three known keys directly instead of implementing a literal evaluator: the format
/// fixes the key set, and a general evaluator would be a parser for attacker-controlled input
/// that buys nothing.
/// </remarks>
internal sealed class NpyHeader
{
    private NpyHeader(DtypeToken dtype, long[] shape)
    {
        Dtype = dtype;
        Shape = shape;
    }

    /// <summary>The element dtype, from <c>descr</c>.</summary>
    internal DtypeToken Dtype { get; }

    /// <summary>The C-order shape, from <c>shape</c>. Empty for a 0-d array.</summary>
    internal long[] Shape { get; }

    /// <summary>Parses an NPY header dict.</summary>
    /// <param name="header">The header source, padding and trailing newline included.</param>
    /// <returns>The parsed fields.</returns>
    /// <exception cref="ConverterException">
    /// A key is missing or malformed, the dtype is outside the closed set or big-endian, or
    /// the array is Fortran-ordered.
    /// </exception>
    internal static NpyHeader Parse(string header)
    {
        string descr = ReadQuoted(header, "descr");
        if (!Dtypes.TryParseDescr(descr, out DtypeToken dtype))
        {
            throw new ConverterException(
                $"NPY dtype '{descr}' is not supported. The wire format admits only "
                + $"little-endian {string.Join(", ", Dtypes.All.Select(Dtypes.ToToken))} "
                + "(GRAMMAR §7).");
        }

        if (ReadBare(header, "fortran_order") is not "False")
        {
            throw new ConverterException(
                "NPY payload is Fortran-ordered. Only C-order arrays are supported; transposing "
                + "on read would silently change which element each index reaches.");
        }

        return new NpyHeader(dtype, ReadShape(header));
    }

    private static int ValueStart(string header, string key)
    {
        // Accept either quote style: numpy writes single quotes, but the format does not
        // require them and a hand-written header may use double.
        int keyIndex = header.IndexOf($"'{key}'", StringComparison.Ordinal);
        if (keyIndex < 0)
        {
            keyIndex = header.IndexOf($"\"{key}\"", StringComparison.Ordinal);
        }

        if (keyIndex < 0)
        {
            throw new ConverterException($"NPY header has no '{key}' key: {header.Trim()}");
        }

        int colon = header.IndexOf(':', keyIndex + key.Length + 2);
        if (colon < 0)
        {
            throw new ConverterException($"NPY header key '{key}' has no value: {header.Trim()}");
        }

        int start = colon + 1;
        while (start < header.Length && char.IsWhiteSpace(header[start]))
        {
            start++;
        }

        return start;
    }

    private static string ReadQuoted(string header, string key)
    {
        int start = ValueStart(header, key);
        if (start >= header.Length || (header[start] != '\'' && header[start] != '"'))
        {
            throw new ConverterException($"NPY header key '{key}' is not a string: {header.Trim()}");
        }

        int end = header.IndexOf(header[start], start + 1);
        if (end < 0)
        {
            throw new ConverterException($"NPY header key '{key}' has an unterminated value: {header.Trim()}");
        }

        return header[(start + 1)..end];
    }

    private static string ReadBare(string header, string key)
    {
        int start = ValueStart(header, key);
        int end = start;
        while (end < header.Length && (char.IsLetterOrDigit(header[end]) || header[end] == '_'))
        {
            end++;
        }

        return header[start..end];
    }

    private static long[] ReadShape(string header)
    {
        int start = ValueStart(header, "shape");
        if (start >= header.Length || header[start] != '(')
        {
            throw new ConverterException($"NPY header 'shape' is not a tuple: {header.Trim()}");
        }

        int end = header.IndexOf(')', start);
        if (end < 0)
        {
            throw new ConverterException($"NPY header 'shape' is unterminated: {header.Trim()}");
        }

        string body = header[(start + 1)..end];
        List<long> dimensions = [];
        foreach (string part in body.Split(','))
        {
            string trimmed = part.Trim();
            if (trimmed.Length == 0)
            {
                // The trailing comma of a 1-tuple, or an empty 0-d '()'.
                continue;
            }

            if (!long.TryParse(trimmed, NumberStyles.None, CultureInfo.InvariantCulture, out long dimension))
            {
                throw new ConverterException(
                    $"NPY header 'shape' has a non-numeric dimension '{trimmed}': {header.Trim()}");
            }

            dimensions.Add(dimension);
        }

        return [.. dimensions];
    }
}
