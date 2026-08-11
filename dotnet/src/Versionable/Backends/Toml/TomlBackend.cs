using System.Collections.Frozen;
using System.Runtime.CompilerServices;
using System.Text;
using Versionable.Engine;
using Versionable.Errors;

namespace Versionable.Backends.Toml;

/// <summary>
/// Reads and writes TOML, the config-file format of the four backends.
/// </summary>
/// <remarks>
/// Python counterpart: <c>TomlBackend</c> in <c>src/versionable/_toml_backend.py</c> (tomlkit).
/// The envelope goes in a <c>[__versionable__]</c> table and every nested object carries its own
/// <c>[field.__versionable__]</c>, so a file written here loads in Python and the reverse, which
/// <c>conformance/golden/*/[fixture].toml</c> pins.
/// <para>
/// <b>TOML costs the wire two things, and both are visible in the corpus.</b> There is no null
/// literal, so a field holding <see langword="null"/> is omitted and comes back from the type's
/// default — the <c>optionals</c> fixture has no <c>absent</c> key at all. And there is no
/// natural spelling for an array blob, so an ndarray is wrapped as a one-key table holding JSON;
/// see <see cref="TomlWire"/> for both.
/// </para>
/// <para>
/// <see cref="NativeTypes"/> is empty, as Python's is: every value goes through the walker.
/// </para>
/// </remarks>
public sealed class TomlBackend : IVersionableBackend
{
    /// <summary>File extensions this backend claims.</summary>
    public static IReadOnlyList<string> Extensions { get; } = [".toml"];

    /// <inheritdoc/>
    public IReadOnlySet<Type> NativeTypes => FrozenSet<Type>.Empty;

    /// <inheritdoc/>
    /// <remarks>
    /// <see cref="BackendSaveOptions.CommentDefaults"/> is honored here: fields still holding the
    /// value their schema declares are written as <c>#</c> lines instead of live keys, which makes
    /// a generated config file self-documenting and lets a user turn a setting on by deleting one
    /// character. The envelope and every section header stay live — commenting a header out would
    /// take its whole table with it.
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

        Dictionary<string, object?> data = new(wire.Count + 1, StringComparer.Ordinal)
        {
            [VersionableEnvelope.WrappedKey] = EnvelopeCodec.Wrap(envelope),
        };

        // Declaration order first so two saves of one schema diff cleanly, then whatever an
        // `unknown = Preserve` load left behind — the same two passes the JSON backend makes.
        foreach (FieldDescriptor field in metadata.Fields)
        {
            if (wire.TryGetValue(field.WireName, out object? value) && value is not null)
            {
                data[field.WireName] = TomlWire.ToTomlSafe(value);
            }
        }

        foreach (KeyValuePair<string, object?> extra in wire)
        {
            if (extra.Value is not null && !data.ContainsKey(extra.Key))
            {
                data[extra.Key] = TomlWire.ToTomlSafe(extra.Value);
            }
        }

        string text = TomlEmitter.Render(data, metadata, options.CommentDefaults, NativeTypes);

        try
        {
            // No BOM: Python reads with encoding="utf-8", which does not strip one, so a BOM would
            // become part of the first key name there.
            File.WriteAllText(path, text, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException
            or NotSupportedException or ArgumentException)
        {
            throw new BackendException($"Failed to write TOML to '{path}': {error.Message}", error);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// TOML has no lazy story, so <see cref="BackendLoadOptions.Preload"/> and
    /// <see cref="BackendLoadOptions.MetadataOnly"/> are ignored and everything is materialized —
    /// which is what <see cref="BackendLoadResult.LazyFields"/> being empty reports.
    /// </remarks>
    public BackendLoadResult Load(string path, BackendLoadOptions options)
    {
        Dictionary<string, object?> document = TomlWire.ReadDocument(path);
        EnvelopeMetadata envelope = EnvelopeCodec.Read(document);

        Dictionary<string, object?> fields = EnvelopeCodec.Strip(document);
        Dictionary<string, object?> unwrapped = new(fields.Count, StringComparer.Ordinal);
        foreach (KeyValuePair<string, object?> entry in fields)
        {
            unwrapped[entry.Key] = TomlWire.FromTomlSafe(entry.Value);
        }

        return new BackendLoadResult(unwrapped, envelope);
    }

    /// <summary>Registers this backend for <see cref="Extensions"/>.</summary>
    /// <remarks>
    /// Also runs from a module initializer; public because <see cref="BackendRegistry.Reset"/>
    /// exists and anything that clears the registry needs a way to put the built-ins back.
    /// </remarks>
    public static void RegisterExtensions() => BackendRegistry.Register(Extensions, static () => new TomlBackend());

    // CA2255 warns off module initializers in libraries because they run on assembly load with no
    // way for a consumer to opt out. That is the intent: BackendRegistry's contract is that every
    // C# backend ships in the one package and registers unconditionally, so `Save(config,
    // "config.toml")` works without an initialization call Python does not need either.
#pragma warning disable CA2255
    [ModuleInitializer]
#pragma warning restore CA2255
    internal static void Register() => RegisterExtensions();
}
