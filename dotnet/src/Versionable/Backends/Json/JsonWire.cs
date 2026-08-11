using System.Text.Json;
using Versionable.Errors;

namespace Versionable.Backends.Json;

/// <summary>
/// Moves between <c>System.Text.Json</c>'s reader/writer and the plain CLR primitives, lists, and
/// dictionaries the engine's walker speaks.
/// </summary>
/// <remarks>
/// Hand-written rather than <c>JsonSerializer.Serialize(object)</c>: that overload reflects over
/// the runtime type, which <c>IsAotCompatible</c> rejects, and it would also let a CLR type the
/// walker never lowered slip into a file unnoticed.
/// </remarks>
internal static class JsonWire
{
    /// <summary>Converts a parsed element into engine-shaped CLR values.</summary>
    /// <param name="element">A parsed JSON element.</param>
    /// <param name="repaired">
    /// Whether the document went through <see cref="NonFiniteJson.TryRepair"/>, in which case the
    /// sentinel strings it introduced are turned back into their non-finite values. Only then:
    /// mapping them unconditionally would rewrite a legitimate string in an untouched file.
    /// </param>
    /// <returns>
    /// <see cref="string"/>, <see cref="bool"/>, <see cref="long"/>, <see cref="ulong"/>,
    /// <see cref="double"/>, <see cref="Dictionary{TKey, TValue}"/>, <see cref="List{T}"/>, or
    /// <see langword="null"/>.
    /// </returns>
    internal static object? ToClr(JsonElement element, bool repaired = false)
    {
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                Dictionary<string, object?> map = new(StringComparer.Ordinal);
                foreach (JsonProperty property in element.EnumerateObject())
                {
                    map[property.Name] = ToClr(property.Value, repaired);
                }

                return map;

            case JsonValueKind.Array:
                List<object?> items = [];
                foreach (JsonElement item in element.EnumerateArray())
                {
                    items.Add(ToClr(item, repaired));
                }

                return items;

            case JsonValueKind.String:
                string? text = element.GetString();
                if (repaired && text is not null && NonFiniteJson.TryFromSentinel(text, out double nonFinite))
                {
                    return nonFinite;
                }

                return text;

            case JsonValueKind.Number:
                // TryGetInt64 reads the raw token, so `2` comes back integral and `2.0` does not —
                // which is what keeps a literal option written `1` comparable to the `1` in the
                // file. Not a conditional expression: its best common type would be double, which
                // would quietly widen every integer in every file.
                if (element.TryGetInt64(out long integer))
                {
                    return integer;
                }

                // Above long.MaxValue and still integral: a ulong field written by either language
                // lands here, and going straight to double would lose the low bits and then
                // overflow on the way into the field.
                if (element.TryGetUInt64(out ulong unsigned))
                {
                    return unsigned;
                }

                return element.GetDouble();

            case JsonValueKind.True:
                return true;

            case JsonValueKind.False:
                return false;

            default:
                return null;
        }
    }

    /// <summary>Writes an engine-shaped wire value.</summary>
    /// <param name="writer">The writer positioned to accept a value.</param>
    /// <param name="value">A value produced by the engine walker.</param>
    /// <exception cref="BackendException">The value is not something JSON can hold.</exception>
    internal static void Write(Utf8JsonWriter writer, object? value)
    {
        switch (value)
        {
            case null:
                writer.WriteNullValue();
                break;
            case string text:
                writer.WriteStringValue(text);
                break;
            case bool flag:
                writer.WriteBooleanValue(flag);
                break;
            case sbyte or short or int or long:
                writer.WriteNumberValue(Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture));
                break;
            case byte or ushort or uint or ulong:
                writer.WriteNumberValue(Convert.ToUInt64(value, System.Globalization.CultureInfo.InvariantCulture));
                break;
            case float single:
                WriteReal(writer, single);
                break;
            case double real:
                WriteReal(writer, real);
                break;
            case decimal money:
                writer.WriteNumberValue(money);
                break;
            case IReadOnlyDictionary<string, object?> map:
                WriteObject(writer, map);
                break;
            case IEnumerable<object?> sequence:
                writer.WriteStartArray();
                foreach (object? item in sequence)
                {
                    Write(writer, item);
                }

                writer.WriteEndArray();
                break;
            default:
                throw new BackendException(
                    $"The JSON backend cannot write '{value.GetType()}'. The walker should have "
                        + "lowered it to a primitive, list, or dictionary first.");
        }
    }

    private static void WriteReal(Utf8JsonWriter writer, double value)
    {
        string? token = NonFiniteJson.TokenFor(value);
        if (token is null)
        {
            writer.WriteNumberValue(value);
            return;
        }

        // WriteNumberValue throws on a non-finite double. Python writes these three tokens bare,
        // and a file it can write is a file this has to produce; see NonFiniteJson.
        writer.WriteRawValue(token, skipInputValidation: true);
    }

    /// <summary>Writes a dictionary as a JSON object.</summary>
    /// <param name="writer">The writer positioned to accept a value.</param>
    /// <param name="map">Entries to write, in enumeration order.</param>
    internal static void WriteObject(Utf8JsonWriter writer, IReadOnlyDictionary<string, object?> map)
    {
        writer.WriteStartObject();
        foreach (KeyValuePair<string, object?> entry in map)
        {
            writer.WritePropertyName(entry.Key);
            Write(writer, entry.Value);
        }

        writer.WriteEndObject();
    }
}
