using System.Collections.Frozen;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Versionable.Engine;
using Versionable.Errors;

namespace Versionable.Backends.Json;

/// <summary>
/// Reads and writes pretty-printed JSON, the format every other backend is compared against.
/// </summary>
/// <remarks>
/// Python counterpart: <c>JsonBackend</c> in <c>src/versionable/_json_backend.py</c>. Two-space
/// indentation and UTF-8 match <c>json.dumps(data, indent=2)</c>, and the envelope goes first,
/// but byte-identity with Python is not a goal and is not achievable: Python renders a whole
/// float as <c>2.0</c> where .NET renders the shortest round-trippable form <c>2</c>, and
/// <c>System.Text.Json</c>'s default encoder escapes <c>&lt;</c>, <c>&gt;</c>, and <c>&amp;</c>
/// where Python escapes non-ASCII instead. Both files parse to the same values in both languages,
/// which is the actual contract (<c>conformance/golden/</c>).
/// <para>
/// <see cref="NativeTypes"/> is empty, as Python's is: JSON holds no CLR type natively, so every
/// value goes through the walker.
/// </para>
/// </remarks>
public sealed class JsonBackend : IVersionableBackend
{
    private static readonly JsonWriterOptions _writerOptions = new() { Indented = true };

    private static readonly JsonDocumentOptions _documentOptions = new()
    {
        CommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
    };

    /// <summary>File extensions this backend claims.</summary>
    public static IReadOnlyList<string> Extensions { get; } = [".json"];

    /// <inheritdoc/>
    public IReadOnlySet<Type> NativeTypes => FrozenSet<Type>.Empty;

    /// <inheritdoc/>
    public void Save(
        IReadOnlyDictionary<string, object?> fields,
        EnvelopeMetadata envelope,
        string path,
        VersionableMetadata metadata,
        BackendSaveOptions options)
    {
        ArgumentNullException.ThrowIfNull(fields);
        ArgumentNullException.ThrowIfNull(envelope);
        ArgumentNullException.ThrowIfNull(metadata);

        IReadOnlyDictionary<string, object?> wire = WireValues.WriteFields(fields, metadata, NativeTypes);

        try
        {
            using FileStream stream = File.Create(path);
            using Utf8JsonWriter writer = new(stream, _writerOptions);

            writer.WriteStartObject();
            writer.WritePropertyName(VersionableEnvelope.WrappedKey);
            JsonWire.WriteObject(writer, EnvelopeCodec.Wrap(envelope));

            // Declaration order first, so a diff between two saves of the same schema lines up;
            // anything else the caller preserved goes after it.
            HashSet<string> written = new(wire.Count, StringComparer.Ordinal);
            foreach (FieldDescriptor field in metadata.Fields)
            {
                if (wire.TryGetValue(field.WireName, out object? value))
                {
                    written.Add(field.WireName);
                    writer.WritePropertyName(field.WireName);
                    JsonWire.Write(writer, value);
                }
            }

            foreach (KeyValuePair<string, object?> extra in wire)
            {
                if (!written.Contains(extra.Key))
                {
                    writer.WritePropertyName(extra.Key);
                    JsonWire.Write(writer, extra.Value);
                }
            }

            writer.WriteEndObject();
            writer.Flush();
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException
            or NotSupportedException or ArgumentException)
        {
            throw new BackendException($"Failed to write JSON to '{path}': {error.Message}", error);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// JSON has no lazy story, so <see cref="BackendLoadOptions.Preload"/> and
    /// <see cref="BackendLoadOptions.MetadataOnly"/> are ignored and everything is materialized —
    /// which is exactly what <see cref="BackendLoadResult.LazyFields"/> being empty reports.
    /// </remarks>
    public BackendLoadResult Load(string path, BackendLoadOptions options)
    {
        Dictionary<string, object?> document = ReadDocument(path);
        EnvelopeMetadata envelope = EnvelopeCodec.Read(document);
        return new BackendLoadResult(EnvelopeCodec.Strip(document), envelope);
    }

    /// <summary>Registers this backend for <see cref="Extensions"/>.</summary>
    /// <remarks>
    /// Also runs from a module initializer, so a consumer that never touches
    /// <see cref="BackendRegistry"/> can still save to a <c>.json</c> path. Public because
    /// <see cref="BackendRegistry.Reset"/> exists: anything that clears the registry needs a way
    /// to put the built-ins back.
    /// </remarks>
    public static void RegisterExtensions() => BackendRegistry.Register(Extensions, static () => new JsonBackend());

    // CA2255 warns off module initializers in libraries because they run on assembly load with no
    // way for a consumer to opt out. That is the intent here: BackendRegistry's contract says
    // every C# backend ships in the one package and registers unconditionally, so that
    // `Save(config, "config.json")` works without an initialization call Python does not need
    // either. The work is one dictionary write.
#pragma warning disable CA2255
    [ModuleInitializer]
#pragma warning restore CA2255
    internal static void Register() => RegisterExtensions();

    private static Dictionary<string, object?> ReadDocument(string path)
    {
        byte[] bytes;
        try
        {
            bytes = File.ReadAllBytes(path);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException)
        {
            throw new BackendException($"Failed to read JSON from '{path}': {error.Message}", error);
        }

        try
        {
            return Parse(bytes, path, repaired: false);
        }
        catch (JsonException error)
        {
            // Python writes NaN and ±Infinity as bare tokens, which no strict JSON parser accepts.
            // The repair pass runs only here, so a well-formed file never pays for it.
            if (NonFiniteJson.TryRepair(bytes, out byte[]? repaired))
            {
                try
                {
                    return Parse(repaired!, path, repaired: true);
                }
                catch (JsonException repairedError)
                {
                    throw new BackendException(
                        $"Failed to read JSON from '{path}': {repairedError.Message}", repairedError);
                }
            }

            throw new BackendException($"Failed to read JSON from '{path}': {error.Message}", error);
        }
    }

    private static Dictionary<string, object?> Parse(byte[] bytes, string path, bool repaired)
    {
        using JsonDocument document = JsonDocument.Parse(bytes, _documentOptions);
        if (document.RootElement.ValueKind != JsonValueKind.Object)
        {
            throw new BackendException(
                $"Expected a JSON object in '{path}', found {document.RootElement.ValueKind}.");
        }

        return (Dictionary<string, object?>)JsonWire.ToClr(document.RootElement, repaired)!;
    }
}
