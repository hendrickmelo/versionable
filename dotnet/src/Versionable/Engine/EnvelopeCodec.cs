using System.Globalization;
using Versionable.Errors;

namespace Versionable.Engine;

/// <summary>
/// Reads and writes the metadata envelope that wraps every serialized object, in both wire
/// layouts.
/// </summary>
/// <remarks>
/// Python counterpart: <c>_serializeVersionable</c>, <c>_readNestedEnvelope</c>, and
/// <c>_stripEnvelope</c> in <c>src/versionable/_types.py</c> (lines 648-740). The key literals
/// live on <see cref="VersionableEnvelope"/>; this type is the behavior around them.
/// <para>
/// Only the wrapped layout is ever written. The 0.1.x flat-dunder layout is read-only
/// back-compat, and it is honored at the top level of a file as well as on a nested object;
/// Python's JSON backend looks for the dunders only <em>inside</em> a <c>__versionable__</c>
/// table, a place no writer has ever put them, so matching it exactly would mean implementing a
/// path that cannot be reached.
/// </para>
/// <para>
/// The unprefixed keys — <c>object</c>, <c>version</c>, <c>hash</c>, <c>format</c> — are honored
/// only <em>inside</em> the wrapped table. They are ordinary words that a schema may well declare
/// as fields, and no layout has ever placed them at the top level, so reading them there would
/// misread a legitimate file in order to support a file that does not exist. Python's
/// <c>_readNestedEnvelope</c> does read them from an unwrapped dictionary, which is why a nested
/// Python object with a field called <c>version</c> is misread there; the divergence is
/// deliberate and costs no readable file.
/// </para>
/// </remarks>
public static class EnvelopeCodec
{
    /// <summary>
    /// Extracts the envelope from an object's wire dictionary, in either layout.
    /// </summary>
    /// <param name="data">
    /// The object's wire dictionary — the whole file for a root object, the nested dictionary
    /// for a nested one.
    /// </param>
    /// <returns>
    /// The envelope. Members are <see langword="null"/> when the corresponding key is absent,
    /// which is what a hand-written or pre-envelope file looks like.
    /// </returns>
    /// <exception cref="BackendException">
    /// The envelope declares a <c>format</c> this build does not know how to read.
    /// </exception>
    public static EnvelopeMetadata Read(IReadOnlyDictionary<string, object?> data)
    {
        ArgumentNullException.ThrowIfNull(data);

        bool isWrapped = data.TryGetValue(VersionableEnvelope.WrappedKey, out object? wrapped);
        if (isWrapped && wrapped is not IReadOnlyDictionary<string, object?>)
        {
            // Python counterpart: the `isinstance(metaTable, dict)` guard in
            // src/versionable/_json_backend.py. Falling through to the flat layout instead would
            // report a corrupt envelope as a missing one, and the file would load as version-less.
            throw new BackendException(
                $"'{VersionableEnvelope.WrappedKey}' must hold the envelope table, but the file has "
                    + $"{wrapped?.GetType().ToString() ?? "null"} there.");
        }

        IReadOnlyDictionary<string, object?> envelope = isWrapped
            ? (IReadOnlyDictionary<string, object?>)wrapped!
            : data;

        RejectUnknownFormat(envelope, isWrapped);

        // Outside the wrapped table only the dunder keys count. `object`, `version`, and `hash` are
        // ordinary words, and a type is free to declare a field called any of them; no writer has
        // ever put them at the top level, so treating them as an envelope there would misread a
        // legitimate file to support a layout that does not exist. The dunders carry no such risk —
        // they are reserved keys, stripped before fields are read either way.
        return new EnvelopeMetadata(
            AsString(Lookup(envelope, VersionableEnvelope.ObjectKey, VersionableEnvelope.LegacyObjectKey, isWrapped)),
            AsVersion(Lookup(
                envelope, VersionableEnvelope.VersionKey, VersionableEnvelope.LegacyVersionKey, isWrapped)),
            AsString(Lookup(envelope, VersionableEnvelope.HashKey, VersionableEnvelope.LegacyHashKey, isWrapped)));
    }

    /// <summary>
    /// Returns <paramref name="data"/> without its envelope keys, leaving only field keys.
    /// </summary>
    /// <remarks>
    /// Migrations run over field-name keys, so every envelope key — both layouts — has to go
    /// first. Python counterpart: <c>_stripEnvelope</c>.
    /// </remarks>
    /// <param name="data">An object's wire dictionary.</param>
    /// <returns>A new dictionary holding only the field entries.</returns>
    public static Dictionary<string, object?> Strip(IReadOnlyDictionary<string, object?> data)
    {
        ArgumentNullException.ThrowIfNull(data);

        Dictionary<string, object?> stripped = new(data.Count, StringComparer.Ordinal);
        foreach (KeyValuePair<string, object?> entry in data)
        {
            if (!VersionableEnvelope.ReservedKeys.Contains(entry.Key))
            {
                stripped[entry.Key] = entry.Value;
            }
        }

        return stripped;
    }

    /// <summary>Builds the nested dictionary written under <c>__versionable__</c>.</summary>
    /// <param name="envelope">The envelope to write.</param>
    /// <returns>A dictionary with the <c>object</c>, <c>version</c>, and <c>hash</c> keys.</returns>
    public static Dictionary<string, object?> Wrap(EnvelopeMetadata envelope)
    {
        ArgumentNullException.ThrowIfNull(envelope);

        return new Dictionary<string, object?>(3, StringComparer.Ordinal)
        {
            [VersionableEnvelope.ObjectKey] = envelope.ObjectName,
            [VersionableEnvelope.VersionKey] = envelope.Version,
            [VersionableEnvelope.HashKey] = envelope.Hash,
        };
    }

    /// <summary>Whether <paramref name="data"/> carries an envelope in either layout.</summary>
    /// <param name="data">An object's wire dictionary.</param>
    /// <returns><see langword="true"/> when at least one envelope key is present.</returns>
    public static bool HasEnvelope(IReadOnlyDictionary<string, object?> data)
    {
        ArgumentNullException.ThrowIfNull(data);

        foreach (string key in VersionableEnvelope.ReservedKeys)
        {
            if (data.ContainsKey(key))
            {
                return true;
            }
        }

        return false;
    }

    private static object? Lookup(
        IReadOnlyDictionary<string, object?> envelope,
        string key,
        string legacyKey,
        bool plainKeysApply) =>
        plainKeysApply && envelope.TryGetValue(key, out object? value) ? value
        : envelope.TryGetValue(legacyKey, out object? legacy) ? legacy
        : null;

    private static void RejectUnknownFormat(IReadOnlyDictionary<string, object?> envelope, bool isWrapped)
    {
        object? format = Lookup(envelope, "format", VersionableEnvelope.LegacyFormatKey, isWrapped);
        if (format is not null)
        {
            // Python counterpart: the same check in JsonBackend.load. A `format` key is how a
            // future wire revision announces itself, so refusing it is what stops an old build
            // from silently misreading a newer file.
            throw new BackendException(
                $"File declares versionable format '{format}', but this build only reads format-less files. "
                    + "Upgrade Versionable to read it.");
        }
    }

    private static string? AsString(object? value) =>
        value switch
        {
            null => null,
            string text => text,
            _ => Convert.ToString(value, CultureInfo.InvariantCulture),
        };

    private static int? AsVersion(object? value)
    {
        if (value is null)
        {
            return null;
        }

        try
        {
            // Backends hand back whatever their reader produced — long from System.Text.Json,
            // int from a native HDF5 attribute, string from a hand-edited file.
            return Convert.ToInt32(value, CultureInfo.InvariantCulture);
        }
        catch (Exception error) when (error is FormatException or InvalidCastException or OverflowException)
        {
            throw new BackendException($"Envelope version '{value}' is not an integer.", error);
        }
    }
}
