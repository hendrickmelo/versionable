using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using Versionable.Backends.Json;
using Versionable.Converters;
using Versionable.Errors;

namespace Versionable.Backends.Yaml;

/// <summary>
/// The one place YAML's wire form differs from JSON's: an ndarray payload is stored as an embedded
/// JSON string rather than as a YAML mapping.
/// </summary>
/// <remarks>
/// Python counterpart: <c>_toYamlSafe</c> and <c>_fromYamlSafe</c> in
/// <c>src/versionable/_yaml_backend.py</c>. The four-key ndarray dict becomes
/// <c>{__ver_json__: '&lt;json&gt;'}</c> on the way out and is unwrapped on the way back in, so a
/// <c>conformance/golden/arrays/arrays.yaml</c> field reads:
/// <code>
/// signal:
///   __ver_json__: '{"__ver_ndarray__": true, "dtype": "float64", ...}'
/// </code>
/// <para>
/// This is not a size optimization — the base64 NPZ dominates either way. It keeps the array out
/// of the YAML structure a human is meant to read and edit, which is the whole reason to choose
/// this backend, and it is what Python writes, which is the reason it is not up for
/// reconsideration.
/// </para>
/// <para>
/// The transform is recursive and structural: it fires on any mapping carrying the
/// <c>__ver_ndarray__</c> marker, wherever it sits, so an array inside a nested object or a list
/// is wrapped too. Legacy files are read with the pre-0.2 <c>__json__</c> wrapper as well.
/// </para>
/// <para>
/// <b>The payload text is not byte-identical to Python's, and permanently will not be.</b>
/// <see cref="Utf8JsonWriter"/> compacts — <c>{"dtype":"float64"}</c> — where Python's
/// <c>json.dumps</c> defaults to a space after each <c>:</c> and <c>,</c>, and PyYAML then folds
/// the longer scalar at 80 columns, so the same array is one line here and several there. Both
/// decode to the same four-key mapping through <c>json.loads</c> and through
/// <see cref="Unwrap"/>, which is the contract; matching the spacing would mean hand-rolling a
/// JSON writer to reproduce a formatting default. This is the standing reason
/// <c>conformance/golden/arrays</c> is excluded from the YAML byte-identity theory in
/// <c>YamlBackendTests</c> — the exclusion is about this file, not about the emitter.
/// </para>
/// </remarks>
internal static class EmbeddedNdarrayJson
{
    /// <summary>Key of the wrapper written around an ndarray payload.</summary>
    internal const string WrapperKey = "__ver_json__";

    /// <summary>Pre-0.2 wrapper key, still read.</summary>
    internal const string LegacyWrapperKey = "__json__";

    // The marker constant lives on the generic converter, so it can only be reached through some
    // instantiation; every one of them folds to the same literal, and `byte` is an arbitrary pick.
    // Hoisting it onto a non-generic holder is a converters-side change, so it is reported rather
    // than made here.
    private const string _ndarrayMarkerKey = TensorConverter<byte>.MarkerKey;

    // The default encoder escapes `+`, `<`, `>`, and `&` for the benefit of JSON embedded in HTML
    // or JavaScript. This payload is embedded in a single-quoted YAML scalar, where none of those
    // characters mean anything, and roughly one base64 character in fifty is a `+` — so the
    // default turns a readable blob into a wall of `+`. The relaxed encoder is the right one
    // here and nothing it stops escaping can appear: the payload holds only base64, a dtype token,
    // and a shape.
    private static readonly JsonWriterOptions _payloadOptions =
        new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    /// <summary>Wraps every ndarray payload in <paramref name="value"/>.</summary>
    /// <param name="value">A wire value from the walker.</param>
    /// <returns>
    /// The same value with ndarray mappings replaced. Returns <paramref name="value"/> itself when
    /// nothing matched, so an array-free document is not rebuilt node by node.
    /// </returns>
    internal static object? Wrap(object? value)
    {
        switch (value)
        {
            case IReadOnlyDictionary<string, object?> map:
                if (map.ContainsKey(_ndarrayMarkerKey))
                {
                    return new Dictionary<string, object?>(1, StringComparer.Ordinal)
                    {
                        [WrapperKey] = ToJson(map),
                    };
                }

                Dictionary<string, object?>? rewrittenMap = null;
                foreach (KeyValuePair<string, object?> entry in map)
                {
                    object? wrapped = Wrap(entry.Value);
                    if (!ReferenceEquals(wrapped, entry.Value) && rewrittenMap is null)
                    {
                        rewrittenMap = new Dictionary<string, object?>(map, StringComparer.Ordinal);
                    }

                    if (rewrittenMap is not null)
                    {
                        rewrittenMap[entry.Key] = wrapped;
                    }
                }

                return rewrittenMap ?? value;

            case IReadOnlyList<object?> list:
                List<object?>? rewrittenList = null;
                for (int index = 0; index < list.Count; index++)
                {
                    object? wrapped = Wrap(list[index]);
                    if (!ReferenceEquals(wrapped, list[index]) && rewrittenList is null)
                    {
                        rewrittenList = [.. list];
                    }

                    if (rewrittenList is not null)
                    {
                        rewrittenList[index] = wrapped;
                    }
                }

                return rewrittenList ?? value;

            default:
                return value;
        }
    }

    /// <summary>Unwraps every embedded-JSON payload in <paramref name="value"/>.</summary>
    /// <param name="value">A value read from a file.</param>
    /// <returns>The same value with wrappers replaced by the mapping they encode.</returns>
    /// <exception cref="BackendException">A wrapper does not hold decodable JSON.</exception>
    internal static object? Unwrap(object? value)
    {
        switch (value)
        {
            case Dictionary<string, object?> map:
                // Length 1 is Python's own guard: a mapping that merely happens to declare a field
                // called `__ver_json__` alongside others is a schema's business, not a wrapper.
                if (map.Count == 1
                    && (map.TryGetValue(WrapperKey, out object? payload)
                        || map.TryGetValue(LegacyWrapperKey, out payload)))
                {
                    return FromJson(payload);
                }

                foreach (string key in map.Keys.ToArray())
                {
                    map[key] = Unwrap(map[key]);
                }

                return map;

            case List<object?> list:
                for (int index = 0; index < list.Count; index++)
                {
                    list[index] = Unwrap(list[index]);
                }

                return list;

            default:
                return value;
        }
    }

    private static string ToJson(IReadOnlyDictionary<string, object?> map)
    {
        using MemoryStream buffer = new();
        using (Utf8JsonWriter writer = new(buffer, _payloadOptions))
        {
            // JsonWire, not a YAML-specific encoder: the payload has to be readable by
            // `json.loads` on the Python side and by the JSON backend's own reader, and there is
            // exactly one right answer for how a wire value becomes JSON.
            JsonWire.WriteObject(writer, map);
        }

        return Encoding.UTF8.GetString(buffer.ToArray());
    }

    private static object? FromJson(object? payload)
    {
        if (payload is not string json)
        {
            throw new BackendException(
                $"'{WrapperKey}' must hold a JSON string, but the file has "
                    + $"{payload?.GetType().ToString() ?? "null"} there.");
        }

        try
        {
            using JsonDocument document = JsonDocument.Parse(json);
            return JsonWire.ToClr(document.RootElement);
        }
        catch (JsonException error)
        {
            throw new BackendException($"Failed to decode embedded JSON: {error.Message}", error);
        }
    }
}
