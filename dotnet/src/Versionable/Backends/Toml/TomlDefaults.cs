using System.Collections.Frozen;
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
    /// <see cref="WireEquality"/> owns the comparison, because <c>SkipDefaults</c> asks the same
    /// question of the same values and the two must not be able to disagree about what "still at
    /// its default" means.
    /// </remarks>
    internal static bool WireEquals(object? left, object? right) => WireEquality.Equal(left, right);
}
