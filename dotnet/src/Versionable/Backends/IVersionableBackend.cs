namespace Versionable.Backends;

/// <summary>
/// A storage backend: turns raw field values plus type metadata into a file, and back.
/// </summary>
/// <remarks>
/// Python counterpart: the <c>Backend</c> ABC in <c>src/versionable/_backend.py</c>.
/// <para>
/// The contract deliberately hands the backend <em>raw</em> field values, not pre-serialized
/// ones, together with the type's <see cref="VersionableMetadata"/>. Each backend owns its
/// own serialization: JSON, YAML, and TOML walk values through
/// <see cref="Converters.IWireConverter"/> and the generated
/// <see cref="FieldDescriptor.WireWriter"/> delegates, while HDF5 maps them onto native HDF5
/// types and never forms a text wire value at all.
/// </para>
/// </remarks>
public interface IVersionableBackend
{
    /// <summary>
    /// CLR types this backend stores natively, which must therefore not be run through a wire
    /// converter. Python counterpart: <c>Backend.nativeTypes</c>.
    /// </summary>
    IReadOnlySet<Type> NativeTypes { get; }

    /// <summary>
    /// Writes <paramref name="fields"/> and <paramref name="envelope"/> to
    /// <paramref name="path"/>.
    /// </summary>
    /// <param name="fields">
    /// Raw field values keyed by wire name, in <see cref="VersionableMetadata.Fields"/> order.
    /// Fields omitted by <c>SkipDefaults</c> are already absent.
    /// </param>
    /// <param name="envelope">Serialization Name, version, and hash to record.</param>
    /// <param name="path">Output file path.</param>
    /// <param name="metadata">Metadata of the type being saved; supplies field types.</param>
    /// <param name="options">Backend-independent and backend-specific save options.</param>
    /// <exception cref="Errors.BackendException">The write failed.</exception>
    void Save(
        IReadOnlyDictionary<string, object?> fields,
        EnvelopeMetadata envelope,
        string path,
        VersionableMetadata metadata,
        BackendSaveOptions options);

    /// <summary>Reads fields and envelope back from <paramref name="path"/>.</summary>
    /// <param name="path">Input file path.</param>
    /// <param name="options">
    /// Which fields to materialize, and the target type's metadata. Backends with no lazy
    /// story ignore the laziness members and load everything.
    /// </param>
    /// <returns>The raw field values and the envelope read from the file.</returns>
    /// <exception cref="Errors.BackendException">The read failed or the file is malformed.</exception>
    BackendLoadResult Load(string path, BackendLoadOptions options);
}
