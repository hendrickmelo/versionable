using System.Text;
using System.Text.Json;
using Tomlyn;
using Tomlyn.Helpers;
using Tomlyn.Serialization;
using Versionable.Backends.Json;
using Versionable.Errors;

namespace Versionable.Backends.Toml;

/// <summary>
/// Moves between Tomlyn's streaming reader/writer and the plain CLR primitives, lists, and
/// dictionaries the engine's walker speaks — plus the two shape changes TOML forces on that wire.
/// </summary>
/// <remarks>
/// Python counterpart: <c>_toTomlSafe</c> / <c>_fromTomlSafe</c> in
/// <c>src/versionable/_toml_backend.py</c>, and the tomlkit calls around them.
/// <para>
/// <b>TOML has no null and no arbitrary nesting for binary blobs</b>, so two lowerings happen
/// here that no other text backend needs:
/// </para>
/// <list type="number">
///   <item>
///     <description>
///     A <see langword="null"/> value is <em>omitted</em>. There is no TOML token for it, and
///     inventing a sentinel would make an absent optional and a string that happens to spell the
///     sentinel indistinguishable. The field comes back from the type's default on load, which is
///     what the <c>optionals</c> golden fixture asserts.
///     </description>
///   </item>
///   <item>
///     <description>
///     An ndarray wire dictionary is wrapped as <c>{ __ver_json__ = "&lt;json&gt;" }</c>. Its
///     payload is a base64 NPZ string beside a dtype and a shape; nesting that as a TOML table
///     would work but would read back as a table of loose keys rather than the one blob it is, and
///     Python has always written the JSON wrapper. Byte-identity with Python's <c>json.dumps</c>
///     is not attempted (it emits <c>", "</c> separators, <c>Utf8JsonWriter</c> emits <c>","</c>);
///     both parse to the same object, which is the contract.
///     </description>
///   </item>
/// </list>
/// <para>
/// <b>Reading is deliberately more permissive than writing.</b> Nothing here ever writes a native
/// TOML datetime — every temporal type reaches the wire as an ISO 8601 string from its converter,
/// which is what the golden <c>temporal</c> fixture holds. A hand-written file may still use
/// TOML's native date/time tokens, and those are handed on as their ISO 8601 text so the ordinary
/// converters accept them. Python cannot do this (tomlkit hands back a <c>datetime</c> object and
/// <c>fromisoformat</c> rejects it), so such a file loads here and fails there; the divergence
/// only ever adds files that load.
/// </para>
/// </remarks>
internal static class TomlWire
{
    /// <summary>Key of the JSON-in-TOML wrapper an ndarray blob is written under.</summary>
    internal const string JsonWrapperKey = "__ver_json__";

    /// <summary>The 0.1.x spelling of <see cref="JsonWrapperKey"/>, read but never written.</summary>
    internal const string LegacyJsonWrapperKey = "__json__";

    /// <summary>The ndarray marker that decides whether a table becomes a JSON blob.</summary>
    /// <remarks>
    /// Aliased through a closed generic so the literal has exactly one definition: the marker is
    /// the same string for every element type, and a rename there has to break the build here.
    /// Only the current marker is matched, as Python's <c>_toTomlSafe</c> does — a file carrying
    /// the pre-0.2 marker is being read, not written, so it never reaches this side.
    /// </remarks>
    internal const string NdarrayMarkerKey = Converters.TensorConverter<double>.MarkerKey;

    private static readonly TomlSerializerOptions _options = new() { NewLine = TomlNewLineKind.Lf };

    /// <summary>Tomlyn options shared by the reader and every scratch writer.</summary>
    internal static TomlSerializerOptions Options => _options;

    // ------------------------------------------------------------------
    // Read
    // ------------------------------------------------------------------

    /// <summary>Parses <paramref name="path"/> into engine-shaped CLR values.</summary>
    /// <param name="path">The file to read.</param>
    /// <returns>The root table, keys in file order.</returns>
    /// <exception cref="BackendException">The read failed or the file is not a TOML table.</exception>
    internal static Dictionary<string, object?> ReadDocument(string path)
    {
        string text;
        try
        {
            text = File.ReadAllText(path, Encoding.UTF8);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException)
        {
            throw new BackendException($"Failed to read TOML from '{path}': {error.Message}", error);
        }

        try
        {
            TomlReader reader = TomlReader.Create(text, _options);
            if (!reader.Read() || reader.TokenType != TomlTokenType.StartDocument)
            {
                throw new BackendException($"Expected a TOML document in '{path}'.");
            }

            if (!reader.Read() || reader.TokenType != TomlTokenType.StartTable)
            {
                throw new BackendException($"Expected a TOML table at the root of '{path}'.");
            }

            return ReadTable(reader);
        }
        catch (TomlException error)
        {
            throw new BackendException($"Failed to read TOML from '{path}': {error.Message}", error);
        }
    }

    /// <summary>Unwraps every <see cref="JsonWrapperKey"/> blob in a parsed table.</summary>
    /// <param name="value">A value straight off the reader.</param>
    /// <returns>The same shape with wrappers replaced by the objects they hold.</returns>
    internal static object? FromTomlSafe(object? value)
    {
        if (value is IReadOnlyDictionary<string, object?> map)
        {
            if (map.Count == 1)
            {
                if (map.TryGetValue(JsonWrapperKey, out object? wrapped)
                    || map.TryGetValue(LegacyJsonWrapperKey, out wrapped))
                {
                    return ParseJsonBlob(wrapped);
                }
            }

            Dictionary<string, object?> unwrapped = new(map.Count, StringComparer.Ordinal);
            foreach (KeyValuePair<string, object?> entry in map)
            {
                unwrapped[entry.Key] = FromTomlSafe(entry.Value);
            }

            return unwrapped;
        }

        if (value is List<object?> items)
        {
            List<object?> unwrapped = new(items.Count);
            foreach (object? item in items)
            {
                unwrapped.Add(FromTomlSafe(item));
            }

            return unwrapped;
        }

        return value;
    }

    // ------------------------------------------------------------------
    // Write
    // ------------------------------------------------------------------

    /// <summary>
    /// Lowers a wire value to something TOML can hold: nulls dropped, ndarray blobs wrapped.
    /// </summary>
    /// <param name="value">A value produced by the engine walker.</param>
    /// <returns>The TOML-safe value, or <see langword="null"/> when the value itself was null.</returns>
    /// <exception cref="BackendException">An array element is null, which TOML cannot express.</exception>
    internal static object? ToTomlSafe(object? value)
    {
        if (value is IReadOnlyDictionary<string, object?> map)
        {
            if (map.ContainsKey(NdarrayMarkerKey))
            {
                return new Dictionary<string, object?>(1, StringComparer.Ordinal)
                {
                    [JsonWrapperKey] = ToJsonBlob(map),
                };
            }

            Dictionary<string, object?> safe = new(map.Count, StringComparer.Ordinal);
            foreach (KeyValuePair<string, object?> entry in map)
            {
                // Python drops null table entries exactly here, not only at the top level: a
                // nested object's optional field has to vanish the same way a root one does, or
                // the nested table would carry a key TOML has no token for.
                if (entry.Value is not null)
                {
                    safe[entry.Key] = ToTomlSafe(entry.Value);
                }
            }

            return safe;
        }

        if (value is string or bool)
        {
            return value;
        }

        if (value is IEnumerable<object?> sequence)
        {
            List<object?> safe = [];
            foreach (object? item in sequence)
            {
                // No null-dropping here, deliberately: a list is positional, so removing an
                // element would silently renumber the rest. Python hits a tomlkit TypeError on
                // the same input; this says which field and why.
                safe.Add(item is null
                    ? throw new BackendException(
                        "TOML cannot store a null inside an array. Drop the element, or use a "
                            + "backend with a null literal (JSON, YAML).")
                    : ToTomlSafe(item));
            }

            return safe;
        }

        return value;
    }

    /// <summary>Writes one wire value through <paramref name="writer"/>.</summary>
    /// <param name="writer">A writer positioned to accept a value.</param>
    /// <param name="value">A TOML-safe value from <see cref="ToTomlSafe"/>.</param>
    /// <exception cref="BackendException">The value is not something TOML can hold.</exception>
    internal static void WriteValue(TomlWriter writer, object? value)
    {
        switch (value)
        {
            case null:
                throw new BackendException("TOML has no null literal; null values are omitted before this point.");
            case string text:
                writer.WriteStringValue(text);
                break;
            case bool flag:
                writer.WriteBooleanValue(flag);
                break;
            case sbyte or short or int or long:
                writer.WriteIntegerValue(Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture));
                break;
            case byte or ushort or uint:
                writer.WriteIntegerValue(Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture));
                break;
            case ulong unsigned:
                // TOML integers are signed 64-bit (spec §Integer). Widening to a float to fit
                // would lose the low bits of exactly the values that need them.
                writer.WriteIntegerValue(unsigned <= long.MaxValue
                    ? (long)unsigned
                    : throw new BackendException(
                        $"{unsigned} exceeds the signed 64-bit range TOML integers occupy. Store it "
                            + "as a string, or use a backend with a wider integer (HDF5)."));
                break;
            case float single:
                writer.WriteFloatValue(single);
                break;
            case double real:
                writer.WriteFloatValue(real);
                break;
            case IReadOnlyDictionary<string, object?> map:
                writer.WriteStartTable();
                foreach (KeyValuePair<string, object?> entry in map)
                {
                    writer.WritePropertyName(entry.Key);
                    WriteValue(writer, entry.Value);
                }

                writer.WriteEndTable();
                break;
            case IEnumerable<object?> sequence:
                writer.WriteStartArray();
                foreach (object? item in sequence)
                {
                    WriteValue(writer, item);
                }

                writer.WriteEndArray();
                break;
            default:
                throw new BackendException(
                    $"The TOML backend cannot write '{value.GetType()}'. The walker should have "
                        + "lowered it to a primitive, list, or dictionary first.");
        }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static Dictionary<string, object?> ReadTable(TomlReader reader)
    {
        Dictionary<string, object?> map = new(StringComparer.Ordinal);
        while (reader.Read())
        {
            if (reader.TokenType == TomlTokenType.EndTable)
            {
                return map;
            }

            if (reader.TokenType != TomlTokenType.PropertyName)
            {
                throw new BackendException($"Expected a TOML key, found {reader.TokenType}.");
            }

            // Tomlyn is compiled without nullable annotations, so the compiler cannot see that a
            // PropertyName token always carries one.
            string name = reader.PropertyName ?? throw new BackendException("A TOML key has no name.");
            if (!reader.Read())
            {
                throw new BackendException($"TOML key '{name}' has no value.");
            }

            map[name] = ReadValue(reader);
        }

        throw new BackendException("The TOML document ended inside a table.");
    }

    private static List<object?> ReadArray(TomlReader reader)
    {
        List<object?> items = [];
        while (reader.Read())
        {
            if (reader.TokenType == TomlTokenType.EndArray)
            {
                return items;
            }

            items.Add(ReadValue(reader));
        }

        throw new BackendException("The TOML document ended inside an array.");
    }

    private static object? ReadValue(TomlReader reader) =>
        reader.TokenType switch
        {
            TomlTokenType.StartTable => ReadTable(reader),
            TomlTokenType.StartArray => ReadArray(reader),
            TomlTokenType.String => reader.GetString(),
            TomlTokenType.Integer => reader.GetInt64(),
            TomlTokenType.Float => reader.GetDouble(),
            TomlTokenType.Boolean => reader.GetBoolean(),

            // Handed on as ISO 8601 text, which is the wire form every temporal converter reads;
            // see the note on the type.
            TomlTokenType.DateTime => TomlFormatHelper.ToString(reader.GetTomlDateTime()),
            _ => throw new BackendException($"Unexpected TOML token {reader.TokenType}."),
        };

    private static string ToJsonBlob(IReadOnlyDictionary<string, object?> map)
    {
        using MemoryStream stream = new();
        using (Utf8JsonWriter writer = new(stream))
        {
            JsonWire.WriteObject(writer, map);
        }

        return Encoding.UTF8.GetString(stream.ToArray());
    }

    private static object? ParseJsonBlob(object? wrapped)
    {
        if (wrapped is not string json)
        {
            throw new BackendException(
                $"'{JsonWrapperKey}' must hold a JSON string, but the file has "
                    + $"{wrapped?.GetType().ToString() ?? "null"} there.");
        }

        try
        {
            using JsonDocument document = JsonDocument.Parse(json);
            return JsonWire.ToClr(document.RootElement);
        }
        catch (JsonException error)
        {
            throw new BackendException($"'{JsonWrapperKey}' does not hold valid JSON: {error.Message}", error);
        }
    }
}
