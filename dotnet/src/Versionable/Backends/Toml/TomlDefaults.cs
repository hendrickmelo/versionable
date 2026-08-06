using System.Collections.Frozen;
using System.Globalization;
using Versionable.Engine;
using Versionable.Errors;

namespace Versionable.Backends.Toml;

/// <summary>
/// The default side of <c>commentDefaults</c>: what a type's untouched fields look like on the
/// TOML wire, and whether a value written there is one of them.
/// </summary>
/// <remarks>
/// Python counterpart: <c>_classDefaultsToml</c> in <c>src/versionable/_toml_backend.py</c>, and
/// the <c>value == defaults[key]</c> test in <c>_addContainerWithDefaults</c>.
/// <para>
/// Defaults are compared <em>on the wire</em>, not as CLR values, and that is the whole trick: a
/// <see cref="decimal"/> default and the <see cref="decimal"/> in the object are two boxes that
/// never compare equal by reference, but both lower to the same string, and it is the string the
/// file would have held. Lowering them through the same
/// <see cref="WireValues.WriteFields(IReadOnlyDictionary{string, object?}, VersionableMetadata, IReadOnlySet{Type})"/>
/// call the real values went through is what keeps the two sides comparable at all.
/// </para>
/// </remarks>
internal static class TomlDefaults
{
    /// <summary>The empty default set, for a table with no metadata behind it.</summary>
    internal static IReadOnlyDictionary<string, object?> None { get; } =
        FrozenDictionary<string, object?>.Empty;

    /// <summary>Computes the TOML-safe wire form of every default <paramref name="metadata"/> declares.</summary>
    /// <param name="metadata">The type whose defaults to lower.</param>
    /// <param name="nativeTypes">The backend's native types.</param>
    /// <returns>
    /// Default wire values keyed by wire name. Fields with no default, and fields whose default
    /// lowers to <see langword="null"/> or to something TOML cannot hold, are absent — a field
    /// that could never be written at its default is never "at its default".
    /// </returns>
    internal static IReadOnlyDictionary<string, object?> For(
        VersionableMetadata metadata,
        IReadOnlySet<Type> nativeTypes)
    {
        Dictionary<string, object?> raw = new(metadata.Fields.Count, StringComparer.Ordinal);
        foreach (FieldDescriptor field in metadata.Fields)
        {
            if (field.HasDefault && field.DefaultFactory is not null)
            {
                raw[field.WireName] = field.DefaultFactory();
            }
        }

        if (raw.Count == 0)
        {
            return None;
        }

        IReadOnlyDictionary<string, object?> wire = WireValues.WriteFields(raw, metadata, nativeTypes);

        Dictionary<string, object?> safe = new(wire.Count, StringComparer.Ordinal);
        foreach (KeyValuePair<string, object?> entry in wire)
        {
            if (entry.Value is null)
            {
                // Python skips a default that serializes to None for the same reason: the field is
                // omitted from the file outright, so there is no line to comment.
                continue;
            }

            try
            {
                safe[entry.Key] = TomlWire.ToTomlSafe(entry.Value);
            }
            catch (BackendException)
            {
                // A default TOML cannot express (a list holding a null, say). The real value will
                // fail on its own terms if it has the same shape; suppressing it here only means
                // the field is never treated as at-default.
            }
        }

        return safe;
    }

    /// <summary>Structural equality over TOML-safe wire values.</summary>
    /// <param name="left">A value about to be written.</param>
    /// <param name="right">The corresponding default.</param>
    /// <returns><see langword="true"/> when the two would render as the same TOML.</returns>
    /// <remarks>
    /// Python compares with <c>==</c>, which is structural over dicts and lists and numeric across
    /// <c>int</c>/<c>float</c>. This matches that: <c>1</c> equals <c>1.0</c>, because the two
    /// spellings denote one value and a schema is free to default an int field from a float
    /// literal.
    /// </remarks>
    internal static bool WireEquals(object? left, object? right)
    {
        if (left is null || right is null)
        {
            return left is null && right is null;
        }

        if (left is IReadOnlyDictionary<string, object?> leftMap)
        {
            if (right is not IReadOnlyDictionary<string, object?> rightMap || leftMap.Count != rightMap.Count)
            {
                return false;
            }

            foreach (KeyValuePair<string, object?> entry in leftMap)
            {
                if (!rightMap.TryGetValue(entry.Key, out object? other) || !WireEquals(entry.Value, other))
                {
                    return false;
                }
            }

            return true;
        }

        if (left is string leftText)
        {
            return right is string rightText && string.Equals(leftText, rightText, StringComparison.Ordinal);
        }

        if (left is IReadOnlyList<object?> leftList)
        {
            if (right is not IReadOnlyList<object?> rightList || leftList.Count != rightList.Count)
            {
                return false;
            }

            for (int index = 0; index < leftList.Count; index++)
            {
                if (!WireEquals(leftList[index], rightList[index]))
                {
                    return false;
                }
            }

            return true;
        }

        if (IsIntegral(left) && IsIntegral(right))
        {
            // Not through double: two ulongs a few bits apart round to the same double, and
            // "close enough to a default" is not a thing a config file should decide.
            return Convert.ToDecimal(left, CultureInfo.InvariantCulture)
                == Convert.ToDecimal(right, CultureInfo.InvariantCulture);
        }

        if (IsNumeric(left) && IsNumeric(right))
        {
            double leftValue = Convert.ToDouble(left, CultureInfo.InvariantCulture);
            double rightValue = Convert.ToDouble(right, CultureInfo.InvariantCulture);
            return leftValue.Equals(rightValue);
        }

        return left.Equals(right);
    }

    private static bool IsIntegral(object value) =>
        value is sbyte or byte or short or ushort or int or uint or long or ulong;

    private static bool IsNumeric(object value) => IsIntegral(value) || value is float or double or decimal;
}
