using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;

namespace Versionable.Analyzers.Grammar;

/// <summary>
/// The Schema Hash algorithm of <c>conformance/GRAMMAR.md</c> §1: sort by wire name, join
/// <c>name:canonicalType</c> pairs with a bare comma, SHA-256 the UTF-8 payload, keep the
/// first six lowercase hex characters.
/// </summary>
/// <remarks>
/// Python counterpart: <c>computeHash()</c> in <c>src/versionable/_hash.py</c>.
/// <para>
/// Two traps are load-bearing here and both are locked by vectors. Sorting is
/// <see cref="StringComparer.Ordinal"/>, never culture-aware — <c>"Mid" &lt; "alpha"</c>
/// only holds by code point. And the sort key is the wire <em>name</em>, not the assembled
/// pair: sorting pairs reorders any name that is a prefix of another whose next character
/// sorts below <c>':'</c> (vector <c>sort-by-name-not-pair</c>).
/// </para>
/// </remarks>
internal static class SchemaHash
{
    /// <summary>Builds the canonical payload from wire-name/canonical-type pairs.</summary>
    /// <param name="fields">Field pairs in any order; duplicates are kept, so a duplicate wire name is visible.</param>
    /// <returns>The <c>,</c>-joined, name-sorted payload. Empty when there are no fields.</returns>
    internal static string ComputePayload(IEnumerable<KeyValuePair<string, string>> fields)
    {
        // A stable sort on the name alone. Array.Sort is unstable, so index-decorate to keep
        // duplicate names in declaration order rather than in an arbitrary one.
        KeyValuePair<string, string>[] items = new List<KeyValuePair<string, string>>(fields).ToArray();
        int[] indices = new int[items.Length];
        for (int i = 0; i < indices.Length; i++)
        {
            indices[i] = i;
        }

        Array.Sort(indices, (left, right) =>
        {
            int byName = string.CompareOrdinal(items[left].Key, items[right].Key);
            return byName != 0 ? byName : left.CompareTo(right);
        });

        StringBuilder payload = new();
        for (int i = 0; i < indices.Length; i++)
        {
            if (i > 0)
            {
                payload.Append(CanonicalGrammar.PairSeparator);
            }

            KeyValuePair<string, string> field = items[indices[i]];
            payload.Append(field.Key).Append(':').Append(field.Value);
        }

        return payload.ToString();
    }

    /// <summary>Hashes a canonical payload.</summary>
    /// <param name="payload">The payload from <see cref="ComputePayload"/>.</param>
    /// <returns>The first six characters of the lowercase hex SHA-256 digest.</returns>
    internal static string ComputeHash(string payload)
    {
        // UTF-8, never UTF-16: the payload's bytes are part of the cross-language contract
        // (vector `non-ascii-utf8`).
        byte[] bytes = Encoding.UTF8.GetBytes(payload);
        using SHA256 sha256 = SHA256.Create();
        byte[] digest = sha256.ComputeHash(bytes);

        StringBuilder hex = new(6);
        for (int i = 0; i < 3; i++)
        {
            hex.Append(digest[i].ToString("x2", System.Globalization.CultureInfo.InvariantCulture));
        }

        return hex.ToString();
    }
}
