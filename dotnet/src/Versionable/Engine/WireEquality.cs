using System.Collections;
using System.Globalization;
using Versionable.Errors;

namespace Versionable.Engine;

/// <summary>
/// Structural equality over wire values, and the one question that needs it: is this field still
/// at the value its schema declares?
/// </summary>
/// <remarks>
/// Python counterpart: the plain <c>==</c> that <c>save()</c> uses for <c>skip_defaults</c>
/// (<c>src/versionable/_api.py</c>) and that <c>_addContainerWithDefaults</c> uses for
/// <c>commentDefaults</c> (<c>src/versionable/_toml_backend.py</c>).
/// <para>
/// <b>Why the comparison happens on the wire rather than on the CLR values.</b> Python's
/// <c>==</c> is structural all the way down: a <c>list</c> equals an equal <c>list</c>, a
/// <c>dict</c> an equal <c>dict</c>, a dataclass an equal dataclass, and <c>1 == 1.0</c>. C#'s
/// <see cref="object.Equals(object?, object?)"/> is none of those things for the types a schema
/// actually defaults — <c>new List&lt;int&gt;()</c> never equals another empty list — so
/// comparing CLR values would make <c>SkipDefaults</c> a no-op for every container and every
/// nested object, which is most of what it exists for. Lowering both sides through the same
/// <see cref="WireValues.WriteFields(IReadOnlyDictionary{string, object?}, VersionableMetadata,
/// IReadOnlySet{Type})"/> call the real value is about to take turns the question into "would
/// these two write the same file", which is both structural and exactly what the option promises.
/// </para>
/// </remarks>
internal static class WireEquality
{
    /// <summary>Structural equality over wire values.</summary>
    /// <param name="left">A value about to be written.</param>
    /// <param name="right">The value to compare it against, normally a lowered default.</param>
    /// <returns><see langword="true"/> when the two would render identically.</returns>
    /// <remarks>
    /// Mirrors Python's <c>==</c>: structural over mappings and sequences, numeric across
    /// integral and floating spellings — <c>1</c> equals <c>1.0</c>, because a schema is free to
    /// default an int field from a float literal and the file would hold the same number either
    /// way.
    /// <para>
    /// <b>One divergence, recorded rather than fixed.</b> <see cref="double.Equals(double)"/> is
    /// <see langword="true"/> for two NaNs; Python's <c>nan == nan</c> is <see langword="false"/>,
    /// so Python would never treat a NaN field as being at a NaN default and this would. The case
    /// is currently unreachable from either caller: a NaN default has to be written
    /// <c>double.NaN</c> or <c>0.0 / 0.0</c>, neither of which is a literal initializer, so the
    /// generator reports <see cref="FieldDescriptor.HasDefault"/> <see langword="false"/> and no
    /// comparison happens. Stated here so that a future widening of the default detection does not
    /// import the divergence silently.
    /// </para>
    /// </remarks>
    internal static bool Equal(object? left, object? right)
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
                if (!rightMap.TryGetValue(entry.Key, out object? other) || !Equal(entry.Value, other))
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
                if (!Equal(leftList[index], rightList[index]))
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

    /// <summary>
    /// Whether <paramref name="value"/> is still what <paramref name="field"/> declares as its
    /// default, judged by what the two would write.
    /// </summary>
    /// <remarks>
    /// A field with no declared default is never at its default — there is nothing to be at. Two
    /// cheap checks run before anything is serialized: <see cref="object.Equals(object?, object?)"/>,
    /// which settles primitives, strings, and records, and a length comparison for two collections,
    /// which settles the common case of a populated field against an empty declared default. Only
    /// when neither answers does the wire comparison run — which is the path a container or nested
    /// object of matching size needs, and the one that would otherwise lower a million-element array
    /// just to discover it is not empty.
    /// <para>
    /// <b>What the gates do not cover.</b> Both are shaped for scalars and collections, so a
    /// <em>nested-object</em> field with a visible default pays the double lowering every time it
    /// is saved: it is never <see cref="object.Equals(object?, object?)"/>-equal to a freshly built
    /// default (that is the whole reason the wire comparison exists) and it is not an
    /// <see cref="ICollection"/>, so both fast paths fall through. That is the known cost of the
    /// option, and it is bounded by the size of the nested object — fine for the config-shaped
    /// types that declare defaults at all, and worth knowing before putting a large object graph
    /// behind one.
    /// </para>
    /// <para>
    /// A default that cannot be lowered at all — a type with no converter, say — is treated as
    /// "not at its default" rather than allowed to fail the save. The real value takes the same
    /// path a moment later and reports the problem itself, with the backend's own context.
    /// </para>
    /// </remarks>
    /// <param name="field">The field being written.</param>
    /// <param name="value">Its current value.</param>
    /// <param name="metadata">The declaring type's metadata; supplies the field's wire writer.</param>
    /// <param name="nativeTypes">The backend's native types, so both sides lower the same way.</param>
    /// <returns><see langword="true"/> when the field may be omitted from the file.</returns>
    internal static bool IsAtDefault(
        FieldDescriptor field,
        object? value,
        VersionableMetadata metadata,
        IReadOnlySet<Type> nativeTypes)
    {
        if (!field.HasDefault || field.DefaultFactory is null)
        {
            return false;
        }

        object? declared = field.DefaultFactory();
        if (Equals(value, declared))
        {
            return true;
        }

        if (value is ICollection populated && declared is ICollection empty && populated.Count != empty.Count)
        {
            return false;
        }

        try
        {
            return Equal(
                Lower(field, value, metadata, nativeTypes),
                Lower(field, declared, metadata, nativeTypes));
        }
        catch (VersionableException)
        {
            return false;
        }
    }

    private static object? Lower(
        FieldDescriptor field,
        object? value,
        VersionableMetadata metadata,
        IReadOnlySet<Type> nativeTypes)
    {
        Dictionary<string, object?> one = new(1, StringComparer.Ordinal) { [field.WireName] = value };
        IReadOnlyDictionary<string, object?> wire = WireValues.WriteFields(one, metadata, nativeTypes);
        return wire.TryGetValue(field.WireName, out object? lowered) ? lowered : null;
    }

    private static bool IsIntegral(object value) =>
        value is sbyte or byte or short or ushort or int or uint or long or ulong;

    private static bool IsNumeric(object value) => IsIntegral(value) || value is float or double or decimal;
}
