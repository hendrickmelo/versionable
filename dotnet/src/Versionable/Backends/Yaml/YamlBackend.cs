using System.Collections.Frozen;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using Versionable.Engine;
using Versionable.Errors;
using YamlDotNet.Core;
using YamlDotNet.Core.Events;

namespace Versionable.Backends.Yaml;

/// <summary>
/// Reads and writes block-style YAML, the format for files a person is expected to open.
/// </summary>
/// <remarks>
/// Python counterpart: <c>YamlBackend</c> in <c>src/versionable/_yaml_backend.py</c>, which calls
/// <c>yaml.dump(data, default_flow_style=False, sort_keys=False)</c>. The emitter settings here
/// reproduce that shape — two-space indent, block collections, sequences flush with their key —
/// and the wire values are the same ones the JSON backend writes, with one exception the format
/// forces (see <see cref="EmbeddedNdarrayJson"/>).
/// <para>
/// <b>The envelope goes last</b>, unlike JSON's. That is Python's layout, not an accident of
/// ordering: the point of a YAML file is that the fields are the first thing you see, and the
/// bookkeeping sits at the bottom. Nested objects still carry their envelope first, because those
/// come from the walker rather than from here.
/// </para>
/// <para>
/// <see cref="NativeTypes"/> is empty, as Python's is: YAML holds no CLR type natively, so every
/// value goes through the walker.
/// </para>
/// <para>
/// <b>Byte-identity with Python is not a goal.</b> PyYAML wraps long lines at 80 columns and this
/// does not; either file parses to the same values in both languages, which is the actual contract
/// (<c>conformance/golden/</c>).
/// </para>
/// </remarks>
public sealed class YamlBackend : IVersionableBackend
{
    /// <summary>File extensions this backend claims.</summary>
    public static IReadOnlyList<string> Extensions { get; } = [".yaml", ".yml"];

    /// <inheritdoc/>
    public IReadOnlySet<Type> NativeTypes => FrozenSet<Type>.Empty;

    /// <inheritdoc/>
    /// <remarks>
    /// <see cref="BackendSaveOptions.CommentDefaults"/> is honored: a top-level field still at its
    /// declared default is written as commented-out lines, so the file shows what a value would
    /// look like without setting it. The block is commented whole — Python leaves a nested
    /// object's <c>__versionable__</c> table uncommented and comments only its data fields, which
    /// produces an envelope for an object with no fields; commenting the field out entirely is
    /// what "this field is not set" already means everywhere else.
    /// </remarks>
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
        ArgumentNullException.ThrowIfNull(options);

        IReadOnlyDictionary<string, object?> wire = WireValues.WriteFields(fields, metadata, NativeTypes);
        List<KeyValuePair<string, object?>> entries = Order(wire, metadata);
        entries.Add(new KeyValuePair<string, object?>(
            VersionableEnvelope.WrappedKey, EnvelopeCodec.Wrap(envelope)));

        string content = options.CommentDefaults
            ? EmitCommented(entries, metadata)
            : Emit(entries);

        try
        {
            // No BOM: `path.write_text(content, encoding="utf-8")` writes none, and PyYAML's
            // reader would take one for content.
            File.WriteAllText(path, content, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException
            or NotSupportedException or ArgumentException)
        {
            throw new BackendException($"Failed to write YAML to '{path}': {error.Message}", error);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// YAML has no lazy story, so <see cref="BackendLoadOptions.Preload"/> and
    /// <see cref="BackendLoadOptions.MetadataOnly"/> are ignored and everything is materialized —
    /// which is exactly what <see cref="BackendLoadResult.LazyFields"/> being empty reports.
    /// </remarks>
    public BackendLoadResult Load(string path, BackendLoadOptions options)
    {
        Dictionary<string, object?> document = ReadDocument(path);
        EnvelopeMetadata envelope = EnvelopeCodec.Read(document);

        Dictionary<string, object?> stripped = EnvelopeCodec.Strip(document);
        foreach (string key in stripped.Keys.ToArray())
        {
            stripped[key] = EmbeddedNdarrayJson.Unwrap(stripped[key]);
        }

        return new BackendLoadResult(stripped, envelope);
    }

    /// <summary>Registers this backend for <see cref="Extensions"/>.</summary>
    /// <remarks>
    /// Also runs from a module initializer, so a consumer that never touches
    /// <see cref="BackendRegistry"/> can still save to a <c>.yaml</c> path. Public because
    /// <see cref="BackendRegistry.Reset"/> exists: anything that clears the registry needs a way
    /// to put the built-ins back.
    /// </remarks>
    public static void RegisterExtensions() => BackendRegistry.Register(Extensions, static () => new YamlBackend());

    // CA2255 warns off module initializers in libraries because they run on assembly load with no
    // way for a consumer to opt out. That is the intent here, for the reason given on
    // JsonBackend.Register: every C# backend ships in the one package and registers
    // unconditionally, so `Save(config, "config.yaml")` works with no initialization call.
#pragma warning disable CA2255
    [ModuleInitializer]
#pragma warning restore CA2255
    internal static void Register() => RegisterExtensions();

    /// <summary>
    /// Declaration order first, then anything a <c>Preserve</c> load carried through, with every
    /// ndarray payload already wrapped.
    /// </summary>
    private static List<KeyValuePair<string, object?>> Order(
        IReadOnlyDictionary<string, object?> wire,
        VersionableMetadata metadata)
    {
        List<KeyValuePair<string, object?>> entries = new(wire.Count + 1);
        HashSet<string> written = new(wire.Count, StringComparer.Ordinal);

        foreach (FieldDescriptor field in metadata.Fields)
        {
            if (wire.TryGetValue(field.WireName, out object? value))
            {
                written.Add(field.WireName);
                entries.Add(new KeyValuePair<string, object?>(
                    field.WireName, EmbeddedNdarrayJson.Wrap(value)));
            }
        }

        foreach (KeyValuePair<string, object?> extra in wire)
        {
            if (!written.Contains(extra.Key))
            {
                entries.Add(new KeyValuePair<string, object?>(
                    extra.Key, EmbeddedNdarrayJson.Wrap(extra.Value)));
            }
        }

        return entries;
    }

    private static string Emit(IReadOnlyList<KeyValuePair<string, object?>> entries)
    {
        StringWriter output = new(new StringBuilder(), CultureInfo.InvariantCulture) { NewLine = "\n" };

        Emitter emitter = new(output, EmitterSettings.Default.WithNewLine("\n"));
        emitter.Emit(new StreamStart());
        emitter.Emit(new DocumentStart(null, null, isImplicit: true));
        emitter.Emit(new MappingStart(AnchorName.Empty, TagName.Empty, true, MappingStyle.Block));

        foreach (KeyValuePair<string, object?> entry in entries)
        {
            YamlWire.Write(emitter, entry.Key);
            YamlWire.Write(emitter, entry.Value);
        }

        emitter.Emit(new MappingEnd());
        emitter.Emit(new DocumentEnd(isImplicit: true));
        emitter.Emit(new StreamEnd());

        return output.ToString();
    }

    /// <summary>
    /// Emits each top-level entry on its own and comments out the ones still at their default.
    /// </summary>
    /// <remarks>
    /// A block mapping at the root has no state that crosses entries — every entry renders the
    /// same alone as it does in place — so emitting them one at a time is what makes it possible
    /// to say which lines belong to which field without parsing the output back, which is how
    /// Python does it and where its version's fragility comes from.
    /// <para>
    /// A field counts as defaulted when its block is byte-identical to the block its declared
    /// default would produce. Comparing the rendered text rather than the CLR values is
    /// deliberate: <c>Equals</c> on a freshly built <c>List&lt;T&gt;</c> default is reference
    /// equality, so a collection field could never match. Python compares dumped text for the same
    /// reason.
    /// </para>
    /// </remarks>
    private static string EmitCommented(
        IReadOnlyList<KeyValuePair<string, object?>> entries,
        VersionableMetadata metadata)
    {
        Dictionary<string, FieldDescriptor> byWireName = new(metadata.Fields.Count, StringComparer.Ordinal);
        foreach (FieldDescriptor field in metadata.Fields)
        {
            byWireName[field.WireName] = field;
        }

        StringBuilder content = new();
        foreach (KeyValuePair<string, object?> entry in entries)
        {
            string block = Emit([entry]);

            // The envelope is never commented: without it the file loads version-less.
            if (entry.Key != VersionableEnvelope.WrappedKey
                && byWireName.TryGetValue(entry.Key, out FieldDescriptor? field)
                && IsAtDefault(field, block))
            {
                foreach (string line in block.Split('\n'))
                {
                    if (line.Length > 0)
                    {
                        content.Append("# ").Append(line).Append('\n');
                    }
                }

                continue;
            }

            content.Append(block);
        }

        return content.ToString();
    }

    private static bool IsAtDefault(FieldDescriptor field, string block)
    {
        if (!field.HasDefault || field.DefaultFactory is null)
        {
            return false;
        }

        try
        {
            object? wire = field.WireWriter is not null
                ? field.WireWriter(field.DefaultFactory())
                : WireValues.Write(field.DefaultFactory(), FrozenSet<Type>.Empty);

            return Emit([new KeyValuePair<string, object?>(field.WireName, EmbeddedNdarrayJson.Wrap(wire))])
                == block;
        }
        catch (VersionableException)
        {
            // A default the walker cannot lower is simply not a default worth commenting; the
            // field's own value has already been written by the time this runs.
            return false;
        }
    }

    private static Dictionary<string, object?> ReadDocument(string path)
    {
        string text;
        try
        {
            text = File.ReadAllText(path);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException)
        {
            throw new BackendException($"Failed to read YAML from '{path}': {error.Message}", error);
        }

        try
        {
            return YamlWire.ReadDocument(new Parser(new StringReader(text)), path)
                ?? throw new BackendException($"Expected a YAML mapping in '{path}', found an empty document.");
        }
        catch (YamlException error)
        {
            throw new BackendException($"Failed to read YAML from '{path}': {error.Message}", error);
        }
    }
}
