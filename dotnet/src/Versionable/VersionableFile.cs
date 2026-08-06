using System.Collections.Frozen;
using Versionable.Backends;
using Versionable.Engine;
using Versionable.Errors;

namespace Versionable;

/// <summary>
/// Saves and loads <c>[Versionable]</c> objects. The entry point the whole library exists for.
/// </summary>
/// <remarks>
/// Python counterpart: the <c>save()</c> and <c>load()</c> free functions in
/// <c>src/versionable/_api.py</c>, reached as <c>versionable.save(obj, path)</c>.
/// <para>
/// <b>Why this name.</b> C# has no free functions, so the pair has to hang off a type, and that
/// type cannot be called <c>Versionable</c> — it is the namespace. Extension methods were the
/// other candidate (<c>obj.Save(path)</c>), but they would put <c>Save</c> on every type in scope
/// while giving <c>Load</c> nowhere to live. A static class named after the resource, as
/// <see cref="System.IO.File"/> is, keeps both halves together and reads the way the Python calls
/// do: <c>VersionableFile.Save(config, "config.json")</c>.
/// </para>
/// <para>
/// The generic overloads are the reflection-free path (ADR-0003): a <c>where T :
/// IVersionableMetadataProvider</c> constraint is the only way to reach <c>T.VersionableMetadata</c>,
/// which is why it is on the signature. The non-generic ones exist for callers holding an
/// <see cref="object"/> or nothing at all, and go through <see cref="VersionableRegistry"/>.
/// </para>
/// </remarks>
public static class VersionableFile
{
    /// <summary>Serializes <paramref name="value"/> to <paramref name="path"/>.</summary>
    /// <typeparam name="T">The declared type of <paramref name="value"/>.</typeparam>
    /// <param name="value">The object to save.</param>
    /// <param name="path">Output path; its extension selects the backend.</param>
    /// <param name="backend">Backend to use instead of selecting by extension.</param>
    /// <param name="options">Backend save options.</param>
    /// <exception cref="BackendException">No backend claims the extension, or the write failed.</exception>
    public static void Save<T>(
        T value,
        string path,
        IVersionableBackend? backend = null,
        BackendSaveOptions? options = null)
        where T : IVersionableMetadataProvider
    {
        ArgumentNullException.ThrowIfNull(value);

        // The runtime type wins where it is known: `Save<Shape>(circle, ...)` has to write a
        // Circle envelope, exactly as Python's `type(obj)` does. T.VersionableMetadata is the
        // fallback for a type whose module initializer has not run — and the no-lookup path the
        // constraint is here to provide.
        VersionableMetadata metadata = VersionableRegistry.TryGetByType(value.GetType(), out VersionableMetadata? found)
            ? found
            : T.VersionableMetadata;

        SaveCore(value, metadata, path, backend, options);
    }

    /// <summary>
    /// Serializes <paramref name="value"/> to <paramref name="path"/>, resolving its metadata by
    /// runtime type.
    /// </summary>
    /// <param name="value">The object to save; its type must be <c>[Versionable]</c>.</param>
    /// <param name="path">Output path; its extension selects the backend.</param>
    /// <param name="backend">Backend to use instead of selecting by extension.</param>
    /// <param name="options">Backend save options.</param>
    /// <exception cref="VersionableException">The runtime type has no registered metadata.</exception>
    public static void Save(
        object value,
        string path,
        IVersionableBackend? backend = null,
        BackendSaveOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(value);

        if (!VersionableRegistry.TryGetByType(value.GetType(), out VersionableMetadata? metadata))
        {
            throw new VersionableException(
                $"'{value.GetType()}' has no registered metadata. Declare it [Versionable] and "
                    + "partial so the source generator can emit and register it.");
        }

        SaveCore(value, metadata, path, backend, options);
    }

    /// <summary>Loads an object of type <typeparamref name="T"/> from <paramref name="path"/>.</summary>
    /// <typeparam name="T">The type to load into.</typeparam>
    /// <param name="path">Input path; its extension selects the backend.</param>
    /// <param name="backend">Backend to use instead of selecting by extension.</param>
    /// <param name="options">Load options; see <see cref="VersionableLoadOptions"/>.</param>
    /// <returns>The loaded instance.</returns>
    /// <remarks>
    /// The file's envelope names a type, and this deliberately ignores it in favour of
    /// <typeparamref name="T"/> — a caller who asked for a <c>Config</c> gets a <c>Config</c> or
    /// an error. Python's <c>load(cls, path)</c> behaves the same way; resolving the type from the
    /// file is what <see cref="Load(string, IVersionableBackend?, VersionableLoadOptions?)"/> is
    /// for. Envelope-driven resolution still happens for <em>nested</em> values, which is what
    /// makes polymorphic collections work.
    /// </remarks>
    /// <exception cref="VersionException">The file's version cannot be reconciled with the type's.</exception>
    /// <exception cref="BackendException">The read failed, or a required field is missing.</exception>
    public static T Load<T>(
        string path,
        IVersionableBackend? backend = null,
        VersionableLoadOptions? options = null)
        where T : IVersionableMetadataProvider =>
        (T)LoadCore(T.VersionableMetadata, path, backend, options);

    /// <summary>
    /// Loads an object whose type is named by the file's envelope.
    /// </summary>
    /// <remarks>
    /// Python counterpart: <c>loadDynamic()</c>. The envelope is read first, the Serialization
    /// Name resolved through <see cref="VersionableRegistry"/>, and the file then read in full —
    /// two reads, as Python does, because a backend that maps values onto declared field types
    /// needs to know the type before it can do the real read.
    /// </remarks>
    /// <param name="path">Input path; its extension selects the backend.</param>
    /// <param name="backend">Backend to use instead of selecting by extension.</param>
    /// <param name="options">Load options; see <see cref="VersionableLoadOptions"/>.</param>
    /// <returns>The loaded instance.</returns>
    /// <exception cref="BackendException">
    /// The file names no type, or names one that is not registered.
    /// </exception>
    public static object Load(
        string path,
        IVersionableBackend? backend = null,
        VersionableLoadOptions? options = null)
    {
        IVersionableBackend resolved = BackendRegistry.Resolve(path, backend);
        BackendLoadResult probe = resolved.Load(
            path,
            new BackendLoadOptions { Preload = FrozenSet<string>.Empty, MetadataOnly = true });

        string? name = probe.Envelope.ObjectName;
        if (string.IsNullOrEmpty(name))
        {
            throw new BackendException(
                $"'{path}' records no object name, so its type cannot be resolved. Load it with "
                    + $"{nameof(Load)}<T>() instead.");
        }

        if (!VersionableRegistry.TryGetByName(name, out VersionableMetadata? metadata))
        {
            throw new BackendException(
                $"Unknown object type '{name}' in '{path}'. The type is not registered, it has been "
                    + "removed, or it is declared Register = false.");
        }

        return LoadCore(metadata, path, resolved, options);
    }

    private static void SaveCore(
        object value,
        VersionableMetadata metadata,
        string path,
        IVersionableBackend? backend,
        BackendSaveOptions? options)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        IVersionableBackend resolved = BackendRegistry.Resolve(path, backend);

        Dictionary<string, object?> raw = new(metadata.Fields.Count, StringComparer.Ordinal);
        foreach (FieldDescriptor field in metadata.Fields)
        {
            object? fieldValue = field.Getter(value);

            if (metadata.SkipDefaults && field.HasDefault && field.DefaultFactory is not null
                && Equals(fieldValue, field.DefaultFactory()))
            {
                continue;
            }

            raw[field.WireName] = fieldValue;
        }

        resolved.Save(
            raw,
            new EnvelopeMetadata(metadata.Name, metadata.Version, metadata.Hash),
            path,
            metadata,
            options ?? new BackendSaveOptions());
    }

    private static object LoadCore(
        VersionableMetadata metadata,
        string path,
        IVersionableBackend? backend,
        VersionableLoadOptions? options)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        IVersionableBackend resolved = BackendRegistry.Resolve(path, backend);
        BackendLoadResult result = resolved.Load(path, ToBackendOptions(options, metadata));

        int version = ResolveVersion(result.Envelope, metadata, path, options?.AssumeVersion);

        IDictionary<string, object?> fields = MigrationRunner.Run(
            new Dictionary<string, object?>(result.Fields, StringComparer.Ordinal),
            metadata,
            version,
            options?.UpgradeInPlace ?? false,
            metadata.Name);

        ReadScope scope = new()
        {
            NativeTypes = resolved.NativeTypes,
            UpgradeInPlace = options?.UpgradeInPlace ?? false,
        };

        using (WireValues.EnterReadScope(scope))
        {
            return ObjectMaterializer.Materialize(fields, metadata, result.LazyFields, scope);
        }
    }

    private static int ResolveVersion(
        EnvelopeMetadata envelope,
        VersionableMetadata metadata,
        string path,
        int? assumeVersion)
    {
        if (envelope.Version is int recorded)
        {
            return recorded;
        }

        if (assumeVersion is int assumed)
        {
            return assumed;
        }

        VersionableLog.Warn(
            $"No version found in '{path}'. Treating it as the current version ({metadata.Version}) "
                + $"for {metadata.Name}. If the file was written by older code, pass "
                + $"{nameof(VersionableLoadOptions.AssumeVersion)} so the right migrations run.");

        return metadata.Version;
    }

    /// <summary>
    /// Translates caller intent into the backend contract, whose no-preference default is the
    /// opposite of Python's.
    /// </summary>
    /// <remarks>
    /// <c>new BackendLoadOptions()</c> means "materialize everything"; Python's <c>load()</c> with
    /// no <c>preload</c> means "materialize no arrays" on the one backend that can be lazy.
    /// Passing the backend default straight through would read every array of every HDF5 file
    /// anyone ever loaded without asking for it — a silent parity break that only shows up as a
    /// memory bill. So no caller preference becomes an <em>empty</em> preload set, which lazy
    /// backends honor and eager ones ignore.
    /// </remarks>
    private static BackendLoadOptions ToBackendOptions(VersionableLoadOptions? options, VersionableMetadata metadata)
    {
        IReadOnlySet<string>? preload =
            options?.PreloadAll == true ? null
            : options?.Preload ?? FrozenSet<string>.Empty;

        BackendLoadOptions backendOptions = new()
        {
            TargetMetadata = metadata,
            Preload = preload,
            MetadataOnly = options?.MetadataOnly ?? false,
        };

        return options?.BackendOptions is { Count: > 0 } extras
            ? backendOptions with { BackendOptions = extras }
            : backendOptions;
    }
}
